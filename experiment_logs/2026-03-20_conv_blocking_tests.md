# Experiment Log: Conv Bugs Blocking Sana/Falqon Tests
**Date:** 2026-03-20
**Branch:** dit_training
**Test:** `tests/Sana_Falqon/test_sana_falqon.py --device npu`

---

## Status Summary

| Test | Status | Blocker |
|---|---|---|
| `falqon_linear` (fwd + bwd) | ✅ PASSED | — |
| `linear_attention` | ✅ PASSED | — |
| `cross_attention` | ✅ PASSED | — |
| `mixffn` | ❌ BLOCKED | Bug #1: depthwise conv (groups > 1) |
| `transformer_block` | ❌ BLOCKED | Bug #2: `StorageBox` has no `.name` |
| `falqon_block` | ❌ BLOCKED | Bug #2 (contains transformer block) |
| `transformer_forward` | ❌ BLOCKED | Bug #2 (contains transformer block) |
| `training_step` | ❌ BLOCKED | Bug #2 (contains transformer block) |
| `flow_matching` | ❌ BLOCKED | Bug #2 (contains transformer block) |

3 of 9 tests pass. 6 blocked by 2 separate conv bugs.

---

## Bug #1 — Depthwise Conv Not Supported (mixffn)

**Op:** `aten.convolution.default` with `groups=320` (depthwise, `groups == in_channels`)

**Root cause:** `MLIRConvTemplate` ignores the `groups` parameter. It reads
`I_C=320` from X shape instead of `I_C_per_group=1` from W shape. This
generates a W scratchpad of 1.7MB and MVIN addresses that overflow the 16MB
virtual spad limit → Spike page fault (exit 255).

**Attempted fix:** `fallback_handler(aten.convolution.default)` in `mlir_lowering.py`.
**Why it failed:** `fallback_handler` emits `torch.ops.aten.convolution(npu_tensor, ...)`
at runtime — still on NPU. `PyTorchSimDevice` does not implement
`convolution_overrideable`, so it raises `NotImplementedError`.

**What's needed:** An explicit device-transfer fallback — move tensors to CPU,
run conv, move result back to NPU — expressed inside the inductor graph.
Options: register a `torch.library.custom_op` that does the transfer, or
implement `MLIRConvDepthwiseTemplate` for proper NPU simulation.

---

## Bug #2 — `StorageBox` Has No `.name` (transformer_block and later)

**Op:** `aten.convolution.default` with `groups=1` (regular 1×1 conv,
`[320, 64, 1, 1]` weight)

**Error:**
```
AttributeError: 'StorageBox' object has no attribute 'name'
  in mlir_conv_common.py:116  get_arg_attributes()
    arg_attributes.append([X.data.data.name, ...])
```

**Root cause:** `get_arg_attributes()` assumes the IR chain is always
`TensorBox → StorageBox → Buffer(name=...)` (two `.data` hops). But when
the input has been through a reshape/reinterpret before the conv, the chain
becomes `TensorBox → ReinterpretView → StorageBox → Buffer(name=...)` (three
hops). `StorageBox` has no `.name`; the name lives one level deeper.

**Fix (one line):** Traverse `.data` until a node with `.name` is found:
```python
# mlir_conv_common.py line 116 — replace:
arg_attributes.append([X.data.data.name, [...]])

# with:
inner = X.data
while not hasattr(inner, 'name'):
    inner = inner.data
arg_attributes.append([inner.name, [...]])
```

This fix is mechanical and low-risk. Once applied, transformer_block and all
downstream tests can proceed (subject to any further issues in those tests).

---

## Key Technical Context

- All three passing tests (falqon_linear, linear_attention, cross_attention) are
  **matmul-only** — they never call `aten.convolution`.
- All six failing tests call `aten.convolution` at least once.
- Bug #2 blocks regular conv (groups=1); Bug #1 blocks depthwise conv (groups>1).
- Fixing Bug #2 alone unblocks 5 tests (transformer_block through flow_matching),
  assuming no other issues in those tests.
- Bug #1 (mixffn) requires a more involved solution.
