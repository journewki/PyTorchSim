# Experiment Log: GLUMBConv Spike Segfault Investigation & Fix
**Date:** 2026-03-20
**Branch:** dit_training
**Test:** `tests/Sana_Falqon/test_sana_falqon.py --device npu --tests mixffn`

---

## Problem

Running `run_sana_mixffn_test` crashes Spike with a user load segfault:

```
User load segfault @ 0x00000000d14af200
[Spike] Command failed with exit code 255
RuntimeError: UNKNOWN_ERROR
```

The fault occurs inside the `GLUMBConv` kernel — SANA's gated depthwise
conv MLP block.

---

## Root Cause Analysis

### Step 1: Identify the kernel

The kernel hash `fix7yzup6xr` was the failing one. Its MLIR header shows:

```
// Single Batch Conv2D kernel
// BATCH = 1, I_C = 320, O_C = 320, K_H = 3, K_W = 3
// TILE_K = 384, TILE_N = 384
memref.global @W_spad : memref<3x1x384x384xf32, 1>   // 1.7 MB !!
```

### Step 2: GLUMBConv is a depthwise conv

`GLUMBConv` uses `nn.Conv2d(..., groups=dim)` where `groups == in_channels`.
This is a **depthwise convolution**: each input channel has its own 3×3 filter,
with no cross-channel mixing.

Actual weight shape (PyTorch): `[320, 1, 3, 3]`
→ O_C=320, I_per_group=1, K_H=3, K_W=3

### Step 3: `MLIRConvTemplate` doesn't handle groups

In `mlir_conv_common.py::extract_info()`:
```python
BATCH, I_C, I_H, I_W = X.layout.size      # I_C = 320 (TOTAL channels)
O_C, _, K_H, K_W   = W.layout.size        # _ = 1 (I_per_group, IGNORED)
```

The template uses `I_C = 320` (total channels from X shape) in the
K-loop: `affine.for %tile_k = 0 to 320 step 384`. But for a depthwise
conv the weight dimension for K should be `I_per_group = 1`, not 320.

This causes the template to generate:
- `W_spad: memref<3x1x384x384xf32>` — treating weight as K=384 × N=384
- `sram_stride=[147456, 147456, 1, 384]` — stride for a 384×384 weight matrix

### Step 4: MVIN writes past the scratchpad boundary

The DMA (MVIN) instruction is:
```
dram_stride=[307200, 102400, 320, 1]
sram_stride=[147456, 147456, 1, 384]
subtile_size=[1, 1, 384, 128]
```

With vlane_split_axis=W, vlane_stride=128, n_vu=128:
- `used_vlane = ceil(384/128) = 3` (only lanes 0,1,2 load real data)
- MVIN still iterates `vlane_idx` from 0 to 127
- `s_addr = W_spad_base + s_idx*4 + vlane_idx * vu_sram_byte`

For vlane_idx=127, s_idx_max=147455:
```
s_addr = 0xD00003C0 + 147455*4 + 127*131072
       = 0xD00003C0 + 589820 + 16645120
       = 0xD106EBBC          ← beyond 16MB virtual spad limit (0xD1000000)
```

The MMU has no mapping for `0xD1xxxxxx` → page fault → Spike exits 255.

### Step 5: Why spad validation didn't catch it

The pre-flight spad overflow check in `extension_codecache.py` (line 185-190)
computes spad_usage from the **validation binary's** `.spad` section:

```
riscv64-unknown-elf-readelf -s validation_binary | grep spad_end
→ spad_end at 0xD0003CC0 = 15.2 KB per lane
```

The check `spad_size (128KB) >= spad_usage (15.2KB)` passes. But the
MVIN instruction's `s_idx` range (spanning all KH=3 slices via
`block_stride`) exceeds this 15.2KB per-lane budget at runtime.

The per-lane allocation is correct (`X:960B + W:13824B + Y:768B = 15.5KB`),
but the MVIN with wrong strides accesses the full 1.7MB range.

---

## Fix

The fundamental bug is in `MLIRConvTemplate`: it does not support grouped
or depthwise convolutions (`groups > 1`). Fixing the template correctly
would require implementing per-group tile logic.

**Fix applied in `PyTorchSimFrontend/mlir/mlir_lowering.py`:**

Added a `groups > 1` guard at the top of `convolution()` that routes any
grouped/depthwise conv through inductor's `fallback_handler`, which runs
the op on CPU while the rest of the compiled graph executes on NPU.

```python
# In convolution(), before template selection:
if groups > 1:
    return fallback_handler(aten.convolution.default, add_to_fallback_set=False)(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )
```

**Note:** Skipping `torch.compile` entirely does NOT work because NPU
tensors in eager mode hit `convolution_overrideable` which PyTorchSimDevice
does not implement, raising `NotImplementedError`.

**Trade-off:** GLUMBConv is not cycle-simulated on NPU. All other ops in
the test (linear attention, cross-attention, transformer block, etc.) are
unaffected.

---

## Files Modified
| File | Change |
|---|---|
| `PyTorchSimFrontend/mlir/mlir_lowering.py` | `convolution()`: add `groups > 1` guard, import `fallback_handler` |

---

## Future Work

The `MLIRConvTemplate` itself would need a proper depthwise conv implementation
to simulate these ops cycle-accurately on NPU. Until then, the `fallback_handler`
approach correctly routes grouped/depthwise convs to CPU within the compiled graph.

---

## Key Technical Details

| Item | Value |
|---|---|
| Fault address | `0xD14AF200` |
| Scratchpad virtual base | `0xD0000000` |
| Scratchpad virtual limit | `0xD1000000` (16MB = 128 lanes × 128KB) |
| W_spad wrong allocation | `3×1×384×384 = 1.7MB total` |
| W_spad correct per-lane | `3×1×3×384 = 13.5KB` (3 channels/lane) |
| MVIN max s_addr | `0xD106EBBC` (exceeds limit by ~6MB) |
| Root cause | `MLIRConvTemplate` ignores `groups` parameter |
| Exit code | 255 (page fault, not 200 INVALID_SPAD_ACCESS) |
