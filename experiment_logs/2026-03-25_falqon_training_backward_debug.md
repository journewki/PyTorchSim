# FALQON Training Backward Debug

**Date:** 2026-03-25
**Branch:** `dit_training`
**Goal:** Fix Tests 7 (`training_step`) and 8 (`flow_matching`) — make `loss.backward()` propagate gradients to FALQON B matrices on NPU.

---

## Summary of Work Done This Session

### 1. Spike DMA Negative Stride Fix (DONE)
Fixed segfault in Spike for `weight.flip([-2,-1])` used in `decompose_convolution_backward`.
- Files fixed: `/workspace/riscv-isa-sim/riscv/insns/torchsim_mvin_common.h` and `torchsim_mvout.h`
- The strides are stored as 32-bit values; cast to `int32_t` first for sign extension before `int64_t`.
- Spike was rebuilt and deployed.

### 2. `convolution_backward` Decomposition (DONE)
`decompose_convolution_backward` is in `PyTorchSimFrontend/mlir/mlir_decomposition.py` (line ~173).
Registered via `decompositions[aten.convolution_backward.default] = decompose_convolution_backward`.

### 3. Non-Contiguous `A_out` Fix (DONE)
In `FalqonMatmul.forward` (previously used), `A_out = res[:, out_features:]` had strides `(68,1)` causing Spike segfault.
Fixed by `.contiguous()`. Then moved on to bigger problem below.

### 4. `FalqonMatmul` Custom Function → LoRA-style Forward (CURRENT)
**Root problem**: `FalqonMatmul` was a `torch.autograd.Function` where B was passed but NOT used in the forward computation. With `torch.compile` + AOT autograd, B is not in the computation graph, so B.grad = None after backward.

**Fix applied in this session**:
Replaced `FalqonLinearEmulated.forward` to use explicit LoRA-style computation where B is actually used:

```python
def forward(self, x):
    W_main = self.weight[:self.out_features, :]  # [out, in]
    A = self.weight[self.out_features:, :]        # [rank, in]
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    A_out = x_2d @ A.t()                         # [batch, rank]
    out = x_2d @ W_main.t() + A_out @ self.B.t() # [batch, out]
    out = out.reshape(*orig_shape[:-1], self.out_features)
    if self.bias is not None:
        out = out + self.bias
    return out
```

This gives `grad_B = grad_output^T @ A_out` automatically via autograd — same formula as the custom Function's manual computation. Numerically equivalent since B starts at zero.

**Isolated test shows it works** (`FalqonLinearEmulated` alone on NPU → B.grad is non-zero).
**Full model test still fails** (B.grad = None after `model_compiled(SANA)` backward).

---

## Current State

**Test 7 failure**: `All B matrices have non-zero gradients: False` — B.grad is None for all B matrices.

**Debug info added** (at the end of this session, not yet run):
```python
print(f"pred.requires_grad: {pred.requires_grad}, grad_fn: {type(pred.grad_fn).__name__}")
print(f"trainable[0] id={id(trainable[0])}, requires_grad={trainable[0].requires_grad}")
```

---

## Key Hypothesis for Why Full Model Fails

The isolated single-layer NPU test works (B.grad is non-None and non-zero). The full SANA model test fails. The difference:

1. **`torch.compiler.is_compiling = lambda: True`** on line 898 — this might cause some ops to behave differently
2. **Graph breaks** — the full model may have dynamo graph breaks that cause parts to run in eager mode. When the FalqonLinearEmulated layers run in eager mode, their ops might go through different dispatch.
3. **B parameter not lifted** — in the full model graph, B might be treated as a constant (frozen) rather than a leaf variable that needs grad.
4. **The debug we need**: Check `pred.requires_grad` and `pred.grad_fn` to confirm grad flows to the output at all. Also compare `id(trainable[0])` with `id(model.B)` for the same module.

---

## Next Steps

### Step 1: Run the diagnostic (add output for debugging)

In `run_falqon_training_step_test`, before `loss.backward()`:
```python
print(f"pred.requires_grad: {pred.requires_grad}")
print(f"pred.grad_fn: {type(pred.grad_fn).__name__ if pred.grad_fn else None}")
for p in trainable[:2]:
    print(f"  trainable B: id={id(p)}, req_grad={p.requires_grad}, device={p.device}")
for m in model.modules():
    if hasattr(m, 'B') and m.B.requires_grad:
        print(f"  model.B: id={id(m.B)}, same={id(m.B)==id(trainable[0])}")
        break
```

Run: `python tests/Sana_Falqon/test_sana_falqon.py --device npu --tests training_step`

### Step 2: Check for graph breaks

Add `TORCH_LOGS="+dynamo"` or `TORCHDYNAMO_PRINT_GRAPH_BREAKS=1` to see if dynamo breaks the graph.

### Step 3: Check if `torch.no_grad()` is used anywhere

Search for `no_grad` in the SANA model or test:
```bash
grep -n "no_grad\|torch.inference_mode" tests/Sana_Falqon/test_sana_falqon.py
```

### Step 4: Try wrapping only FalqonLinearEmulated

Instead of compiling the whole model, check if compiling just the FALQON forward and doing the rest in eager mode gives correct B.grad:
```python
# Test: don't compile the whole model, just check eager backward
pred = model(hidden_states=noisy_input, ...)  # non-compiled
loss.backward()
```

### Step 5: Check if there's an issue with `model.requires_grad_(False)` + new modules

In the SANA model test, `model.requires_grad_(False)` is called before `convert_to_falqon`. Verify that new FalqonLinearEmulated B parameters are still `requires_grad=True` after the full model is on device:
```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"{name}: req_grad={p.requires_grad}")
```

### Step 6: Alternative approach — use `register_hook` for grad accumulation

If torch.compile can't properly handle B's gradient in the full model context, an alternative is to manually accumulate grad_B using a tensor hook:

```python
def forward(self, x):
    W_main = self.weight[:self.out_features, :]
    A = self.weight[self.out_features:, :]
    x_2d = x.reshape(-1, x.shape[-1])
    A_out = x_2d @ A.t()
    out = x_2d @ W_main.t()
    # Use .detach() + hook to accumulate grad_B without it being in the forward graph
    if self.B.requires_grad:
        A_out_for_hook = A_out.detach()
        def accum_B_grad(grad):
            g = grad.reshape(-1, grad.shape[-1]).t() @ A_out_for_hook
            if self.B.grad is None:
                self.B.grad = g
            else:
                self.B.grad.add_(g)
        out.register_hook(accum_B_grad)
    out = out.reshape(*x.shape[:-1], self.out_features)
    if self.bias is not None:
        out = out + self.bias
    return out
```

**Caution**: This won't work with torch.compile since hooks are not compiled. The hook runs in eager mode on NPU tensors. Eager NPU mm might not produce correct values.

### Step 7: Alternative approach — move B to CPU, gather outputs to CPU for grad

Actually the most robust approach might be to compute the B gradient correction OUTSIDE torch.compile:

```python
# After loss.backward(), recompute grad_B manually:
with torch.no_grad():
    for layer in falqon_layers:
        # A_out was computed during forward, need to re-run or save it
        pass
```

This is complex.

### Step 8: The ACTUAL fix might be simpler

Look at whether `@torch._dynamo.disable` should be used on `FalqonLinearEmulated.forward` to run it in eager mode. In eager mode, autograd's standard Function mechanism works. The mm ops in eager NPU mode would produce zero outputs currently (as was the issue), BUT:

With the NEW LoRA-style forward that includes B explicitly:
- In eager NPU mode, `A_out = x_2d @ A.t()` runs as eager NPU mm (might return zeros)
- But B.grad would be computed as `grad_output.t() @ A_out` = zeros if A_out is zero
- Still zero...

Actually wait — from the session summary: "Eager NPU mm doesn't go through Spike, returns zeros." But looking at the log for the isolated test — the isolated test DOES show Spike running and B.grad is non-zero. So COMPILED NPU mm through Spike works correctly.

The issue must be that in the FULL model, the FalqonLinearEmulated forward is somehow NOT going through our compiler, OR there's something else.

---

## Files Modified This Session

| File | Change |
|---|---|
| `tests/Sana_Falqon/test_sana_falqon.py` | Replaced `FalqonMatmul` custom Function forward with explicit LoRA-style forward in `FalqonLinearEmulated.forward`; added debug output before assertion |
| `tests/Sana_Falqon/test_sana_falqon.py` | Changed `p.grad.abs().sum() > 0` to `p.grad.cpu().abs().sum().item() > 0` in the B-grad check |
| (Previous session) `riscv-isa-sim/.../torchsim_mvin_common.h` | Signed arithmetic fix for negative DMA strides |
| (Previous session) `riscv-isa-sim/.../torchsim_mvout.h` | Same signed arithmetic fix |
| (Previous session) `PyTorchSimFrontend/mlir/mlir_decomposition.py` | Added `decompose_convolution_backward` |

---

## Diagnostic Output Added (not yet run to completion)

At line ~706 in `run_falqon_training_step_test`:
```python
print(f"pred.requires_grad: {pred.requires_grad}, pred.grad_fn: {type(pred.grad_fn).__name__ if pred.grad_fn else None}")
print(f"trainable[0] id={id(trainable[0])}, requires_grad={trainable[0].requires_grad}, device={trainable[0].device}")
for m in model.modules():
    if hasattr(m, 'B') and m.B.requires_grad:
        print(f"model.B id={id(m.B)}, same={id(m.B)==id(trainable[0])}")
        break
```

These should help diagnose whether:
1. pred.requires_grad=False (grad doesn't flow at all)
2. trainable[0] != model.B (parameter aliasing issue)
3. B.requires_grad was changed somehow
