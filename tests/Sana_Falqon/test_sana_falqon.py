"""
SANA Diffusion Transformer + FALQON (FP8 LoRA) PyTorchSim Test

Bottom-up block-level tests following test_llama.py / test_diffusion.py patterns.
Each block is independently compiled and compared (CPU vs Device) for cycle measurement.

Phase 1: conda diffusers + CUDA verification (--device cuda)
Phase 2: PyTorchSim NPU simulation          (--device npu)

Usage:
    python test_sana_falqon.py --device cuda --tests all
    python test_sana_falqon.py --device cuda --tests falqon_linear
    python test_sana_falqon.py --device npu  --tests all
"""

import os
import sys
import re
import copy
import math
import argparse
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from diffusers (SANA model components)
# ---------------------------------------------------------------------------
from diffusers.models.transformers.sana_transformer import (
    SanaTransformer2DModel,
    SanaTransformerBlock,
    GLUMBConv,
    SanaAttnProcessor2_0,
    SanaModulatedNorm,
)
from diffusers.models.attention_processor import Attention, SanaLinearAttnProcessor2_0
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm


# ===================================================================
# Small model config for simulation
# ===================================================================
SMALL_SANA_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_attention_heads=4,
    attention_head_dim=16,       # inner_dim = 4*16 = 64
    num_layers=2,
    num_cross_attention_heads=2,
    cross_attention_head_dim=32, # cross inner_dim = 2*32 = 64
    cross_attention_dim=64,
    caption_channels=64,
    mlp_ratio=2.5,
    dropout=0.0,
    attention_bias=True,
    sample_size=8,               # 8x8 latent
    patch_size=1,
    norm_elementwise_affine=False,
    norm_eps=1e-6,
    guidance_embeds=False,
    qk_norm=None,
    timestep_scale=1.0,
)

INNER_DIM = SMALL_SANA_CONFIG["num_attention_heads"] * SMALL_SANA_CONFIG["attention_head_dim"]  # 64
FALQON_RANK = 4


# ===================================================================
# Test result comparison (same pattern as test_diffusion.py / test_llama.py)
# ===================================================================
def test_result(name, out, cpu_out, rtol=1e-3, atol=1e-3):
    if torch.allclose(out.cpu().float(), cpu_out.cpu().float(), rtol=rtol, atol=atol):
        msg = f"|{name} Test Passed|"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
    else:
        msg = f"|{name} Test Failed|"
        print("-" * len(msg))
        print(msg)
        print("-" * len(msg))
        diff = torch.max(torch.abs(out.cpu().float() - cpu_out.cpu().float())).item()
        print("device out:", out.detach().cpu())
        print("cpu ref   :", cpu_out.detach().cpu())
        print(f"Max abs diff: {diff}")
        sys.exit(1)


# ===================================================================
# FALQON Emulated Linear Layer
# ===================================================================
def _low_rank_decomposition(weight, rank):
    """SVD-based low-rank decomposition: weight ≈ L @ R"""
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    sqrt_S = torch.sqrt(S[:rank])
    L = U[:, :rank] * sqrt_S[None, :]      # [out_features, rank]
    R = sqrt_S[:, None] * Vh[:rank, :]      # [rank, in_features]
    return L, R


def _quantize_emulate(weight):
    """Emulate FP8 quantization error via round-trip cast to float16.

    In real FALQON, weight is cast to FP8 (e4m3fn) then back.
    For BF16/FP32 emulation we approximate this with float16 truncation,
    which introduces a small quantization error suitable for SVD decomposition.
    """
    return weight.half().float()


class FalqonMatmul(torch.autograd.Function):
    """Custom autograd for FALQON's single-matmul-then-split pattern.

    Forward:  res = input @ [W | A]^T  ->  split -> main_output
    Backward: grad_B = grad_output^T @ A_output   (B is the only trainable param)
              grad_input = grad_output @ W_main^T  (propagate to previous layer)
    """

    @staticmethod
    def forward(ctx, input, weight, B, out_features, rank):
        # input: [*, in_features]
        # weight: [out_features + rank, in_features]  (frozen)
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])

        # Single matmul: [batch, in] @ [in, out+rank] -> [batch, out+rank]
        res = input_2d @ weight.t()

        main_out = res[:, :out_features]
        A_out = res[:, out_features:]

        ctx.save_for_backward(weight[:out_features, :], A_out)
        ctx.orig_shape = orig_shape

        return main_out.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        W_main, A_out = ctx.saved_tensors
        orig_shape = ctx.orig_shape

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # grad_input = grad_output @ W_main  -> propagate to previous layer
        grad_input = grad_output_2d @ W_main
        grad_input = grad_input.reshape(orig_shape)

        # grad_B = grad_output^T @ A_out  -> shape [out_features, rank]
        grad_B = grad_output_2d.t() @ A_out

        return grad_input, None, grad_B, None, None


class FalqonLinearEmulated(nn.Module):
    """FALQON FP8 Linear emulation in BF16/FP32.

    Preserves FALQON's core structure:
    - weight = [Q(W) | A] concatenated, frozen
    - B = [out_features, rank], only trainable parameter
    - Forward: single matmul -> split -> main output
    - Backward: grad flows only to B
    - Weight fusion: merge B @ A^T into W, reset B to zero

    Original FALQON uses FP8; this version uses the same dtype as the model
    with float16-based quantization emulation for SVD initialization.
    """

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank


        # [out_features + rank, in_features] — frozen
        self.weight = nn.Parameter(
            torch.zeros(out_features + rank, in_features), requires_grad=False
        )
        # [out_features, rank] — trainable
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int):
        """Convert nn.Linear to FALQON structure with SVD initialization."""
        in_f = linear.in_features
        out_f = linear.out_features
        has_bias = linear.bias is not None

        module = cls(in_f, out_f, rank, bias=has_bias)

        W = linear.weight.data.float()  # [out_features, in_features]

        # Emulate FP8 quantization
        Q_W = _quantize_emulate(W)
        E = W - Q_W  # quantization error

        # SVD decomposition of error: E ≈ L @ R
        L, R = _low_rank_decomposition(E, rank)  # L:[out,rank], R:[rank,in]

        A = R   # frozen, embedded in weight
        B = torch.zeros(out_f, rank, dtype=W.dtype)

        # Concatenate: weight = [Q(W); A]  shape [out+rank, in]
        merged = torch.cat([Q_W, A], dim=0)
        module.weight.data.copy_(merged)
        module.B.data.copy_(B)

        if has_bias:
            module.bias.data.copy_(linear.bias.data)

        return module

    def forward(self, x):
        out = FalqonMatmul.apply(x, self.weight, self.B, self.out_features, self.rank)
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def apply_delta_weight(self):
        """FALQON weight fusion: merge B @ A^T into main weight, reset B."""
        W_main = self.weight.data[:self.out_features, :]   # [out, in]
        A = self.weight.data[self.out_features:, :]         # [rank, in]

        # delta = B @ A  -> [out, in]
        delta = self.B.data @ A

        # Update main weight
        W_new = W_main + delta

        # Re-emulate quantization (simulates FP8 re-quantization)
        W_new_q = _quantize_emulate(W_new)

        # Store residual for next merge (quantization error accumulation)
        # (simplified: we don't carry residual in this emulation)

        self.weight.data[:self.out_features, :] = W_new_q
        self.B.data.zero_()

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


# ===================================================================
# SANA model FALQON conversion
# ===================================================================
def convert_to_falqon(model, target_modules, rank):
    """Replace target nn.Linear modules with FalqonLinearEmulated.

    Args:
        model: SanaTransformer2DModel or sub-block
        target_modules: list of module name suffixes, e.g. ["to_q", "to_k", "to_v", "to_out.0"]
        rank: FALQON low-rank dimension
    """
    pattern = re.compile(r".*(" + "|".join(re.escape(m) for m in target_modules) + r")$")
    replacements = []

    for full_name, module in model.named_modules():
        if pattern.match(full_name) and isinstance(module, nn.Linear):
            replacements.append(full_name)

    for full_name in replacements:
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        attr = parts[-1]
        old_linear = getattr(parent, attr)
        new_module = FalqonLinearEmulated.from_linear(old_linear, rank)
        setattr(parent, attr, new_module)

    return model, replacements


def apply_delta_weight_all(model):
    """Apply weight fusion on all FalqonLinearEmulated modules."""
    for module in model.modules():
        if isinstance(module, FalqonLinearEmulated):
            module.apply_delta_weight()


# ===================================================================
# Test 1: FALQON Linear — single layer forward + backward
# ===================================================================
@torch.no_grad()
def run_falqon_linear_forward_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 1] FalqonLinearEmulated Forward")
    print("=" * 60)

    in_f, out_f, rank = 64, 64, FALQON_RANK
    base_linear = nn.Linear(in_f, out_f, bias=True)

    cpu_mod = FalqonLinearEmulated.from_linear(base_linear, rank).eval()
    dev_mod = copy.deepcopy(cpu_mod).to(device).eval()
    dev_mod_compiled = torch.compile(dev_mod, dynamic=False)

    g = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(2, 8, in_f, generator=g)
    x_dev = x_cpu.to(device)

    out_cpu = cpu_mod(x_cpu)
    out_dev = dev_mod_compiled(x_dev)

    test_result("FalqonLinear forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("FalqonLinear forward test done.")


def run_falqon_linear_backward_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 1b] FalqonLinearEmulated Backward (B.grad)")
    print("=" * 60)

    in_f, out_f, rank = 64, 64, FALQON_RANK
    base_linear = nn.Linear(in_f, out_f, bias=True)

    cpu_mod = FalqonLinearEmulated.from_linear(base_linear, rank)
    cpu_mod.weight.requires_grad_(False)
    cpu_mod.B.requires_grad_(True)

    dev_mod = copy.deepcopy(cpu_mod).to(device)
    dev_mod.weight.requires_grad_(False)
    dev_mod.B.requires_grad_(True)

    g = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(2, 8, in_f, generator=g)
    target_cpu = torch.randn(2, 8, out_f, generator=g)
    x_dev = x_cpu.to(device)
    target_dev = target_cpu.to(device)

    # CPU forward + backward
    out_cpu = cpu_mod(x_cpu)
    loss_cpu = F.mse_loss(out_cpu, target_cpu)
    loss_cpu.backward()

    # Device forward + backward
    out_dev = dev_mod(x_dev)
    loss_dev = F.mse_loss(out_dev, target_dev)
    loss_dev.backward()

    test_result("FalqonLinear B.grad", dev_mod.B.grad, cpu_mod.B.grad, rtol=rtol, atol=atol)
    print("Max grad diff >", torch.max(torch.abs(dev_mod.B.grad.cpu().float() - cpu_mod.B.grad.float())).item())
    assert cpu_mod.weight.grad is None, "weight should not have gradient"
    print("Confirmed: weight.grad is None (frozen)")
    print("FalqonLinear backward test done.")


# ===================================================================
# Test 2: SANA Linear Attention block
# ===================================================================
@torch.no_grad()
def run_sana_linear_attention_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 2] SANA Linear Attention (SanaLinearAttnProcessor2_0)")
    print("=" * 60)

    dim = INNER_DIM
    heads = SMALL_SANA_CONFIG["num_attention_heads"]
    head_dim = SMALL_SANA_CONFIG["attention_head_dim"]

    base_attn = Attention(
        query_dim=dim,
        heads=heads,
        dim_head=head_dim,
        dropout=0.0,
        bias=True,
        cross_attention_dim=None,  # self-attention
        processor=SanaLinearAttnProcessor2_0(),
    ).eval()

    cpu_attn = copy.deepcopy(base_attn).eval()
    dev_attn = base_attn.to(device).eval()
    dev_attn_compiled = torch.compile(dev_attn, dynamic=False)

    batch, seq = 1, 64  # 8x8 patches
    g = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(batch, seq, dim, generator=g)
    x_dev = x_cpu.to(device)

    out_cpu = cpu_attn(x_cpu)
    out_dev = dev_attn_compiled(x_dev)

    test_result("SanaLinearAttn forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SANA Linear Attention test done.")


# ===================================================================
# Test 3: SANA Cross Attention block
# ===================================================================
@torch.no_grad()
def run_sana_cross_attention_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 3] SANA Cross Attention (SanaAttnProcessor2_0)")
    print("=" * 60)

    dim = INNER_DIM
    cross_heads = SMALL_SANA_CONFIG["num_cross_attention_heads"]
    cross_head_dim = SMALL_SANA_CONFIG["cross_attention_head_dim"]
    cross_dim = SMALL_SANA_CONFIG["cross_attention_dim"]

    base_attn = Attention(
        query_dim=dim,
        heads=cross_heads,
        dim_head=cross_head_dim,
        dropout=0.0,
        bias=True,
        cross_attention_dim=cross_dim,
        out_bias=True,
        processor=SanaAttnProcessor2_0(),
    ).eval()

    cpu_attn = copy.deepcopy(base_attn).eval()
    dev_attn = base_attn.to(device).eval()
    dev_attn_compiled = torch.compile(dev_attn, dynamic=False)

    batch, seq, enc_seq = 1, 64, 16
    g = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(batch, seq, dim, generator=g)
    enc_cpu = torch.randn(batch, enc_seq, cross_dim, generator=g)
    x_dev = x_cpu.to(device)
    enc_dev = enc_cpu.to(device)

    out_cpu = cpu_attn(x_cpu, encoder_hidden_states=enc_cpu)
    out_dev = dev_attn_compiled(x_dev, encoder_hidden_states=enc_dev)

    test_result("SanaCrossAttn forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SANA Cross Attention test done.")


# ===================================================================
# Test 4: SANA Mix-FFN (GLUMBConv)
# ===================================================================
@torch.no_grad()
def run_sana_mixffn_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 4] SANA Mix-FFN (GLUMBConv)")
    print("=" * 60)

    dim = INNER_DIM
    ratio = SMALL_SANA_CONFIG["mlp_ratio"]

    base_ff = GLUMBConv(dim, dim, ratio, norm_type=None, residual_connection=False).eval()

    cpu_ff = copy.deepcopy(base_ff).eval()
    dev_ff = base_ff.to(device).eval()

    batch, h, w = 1, 8, 8
    g = torch.Generator().manual_seed(0)
    # GLUMBConv expects [B, C, H, W]
    x_cpu = torch.randn(batch, dim, h, w, generator=g)
    x_dev = x_cpu.to(device)

    out_cpu = cpu_ff(x_cpu)

    dev_ff_compiled = torch.compile(dev_ff, dynamic=False)
    out_dev = dev_ff_compiled(x_dev)

    test_result("GLUMBConv (Mix-FFN) forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SANA Mix-FFN test done.")


# ===================================================================
# Test 5: SANA Transformer Block (single block)
# ===================================================================
@torch.no_grad()
def run_sana_transformer_block_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 5] SanaTransformerBlock (single block)")
    print("=" * 60)

    dim = INNER_DIM
    cfg = SMALL_SANA_CONFIG

    base_block = SanaTransformerBlock(
        dim=dim,
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        dropout=cfg["dropout"],
        num_cross_attention_heads=cfg["num_cross_attention_heads"],
        cross_attention_head_dim=cfg["cross_attention_head_dim"],
        cross_attention_dim=cfg["cross_attention_dim"],
        attention_bias=cfg["attention_bias"],
        norm_elementwise_affine=cfg["norm_elementwise_affine"],
        norm_eps=cfg["norm_eps"],
        mlp_ratio=cfg["mlp_ratio"],
        qk_norm=cfg["qk_norm"],
    ).eval()

    cpu_block = copy.deepcopy(base_block).eval()
    dev_block = base_block.to(device).eval()
    dev_block_compiled = torch.compile(dev_block, dynamic=False)

    batch, seq, h, w = 1, 64, 8, 8
    enc_seq = 16
    g = torch.Generator().manual_seed(0)

    hidden_cpu = torch.randn(batch, seq, dim, generator=g)
    enc_cpu = torch.randn(batch, enc_seq, cfg["cross_attention_dim"], generator=g)
    # timestep modulation: [B, 6*dim] from AdaLayerNormSingle
    timestep_cpu = torch.randn(batch, 6, dim, generator=g)

    hidden_dev = hidden_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    timestep_dev = timestep_cpu.to(device)

    out_cpu = cpu_block(
        hidden_states=hidden_cpu,
        encoder_hidden_states=enc_cpu,
        timestep=timestep_cpu,
        height=h,
        width=w,
    )
    out_dev = dev_block_compiled(
        hidden_states=hidden_dev,
        encoder_hidden_states=enc_dev,
        timestep=timestep_dev,
        height=h,
        width=w,
    )

    test_result("SanaTransformerBlock forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SanaTransformerBlock test done.")


# ===================================================================
# Test 6: Full SanaTransformer2DModel forward
# ===================================================================
@torch.no_grad()
def run_sana_transformer_forward_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 6] SanaTransformer2DModel forward (full model)")
    print("=" * 60)

    cfg = SMALL_SANA_CONFIG
    base_model = SanaTransformer2DModel(**cfg).eval()

    cpu_model = copy.deepcopy(base_model).eval()
    dev_model = base_model.to(device).eval()
    dev_model_compiled = torch.compile(dev_model, dynamic=False)

    batch = 1
    h = w = cfg["sample_size"]  # 8
    in_ch = cfg["in_channels"]  # 4
    enc_seq = 16
    g = torch.Generator().manual_seed(0)

    latent_cpu = torch.randn(batch, in_ch, h, w, generator=g)
    enc_cpu = torch.randn(batch, enc_seq, cfg["caption_channels"], generator=g)
    timestep_cpu = torch.tensor([500.0])

    latent_dev = latent_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    timestep_dev = timestep_cpu.to(device)

    out_cpu = cpu_model(
        hidden_states=latent_cpu,
        encoder_hidden_states=enc_cpu,
        timestep=timestep_cpu,
        return_dict=False,
    )[0]

    out_dev = dev_model_compiled(
        hidden_states=latent_dev,
        encoder_hidden_states=enc_dev,
        timestep=timestep_dev,
        return_dict=False,
    )[0]

    test_result("SanaTransformer2DModel forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SanaTransformer2DModel forward test done.")


# ===================================================================
# Test 7: FALQON Training Step (forward + backward + optimizer + weight fusion)
# ===================================================================
def run_falqon_training_step_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 7] FALQON Training Step (full training cycle)")
    print("=" * 60)

    cfg = SMALL_SANA_CONFIG
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    # Build model
    model = SanaTransformer2DModel(**cfg)
    model.requires_grad_(False)

    # Convert to FALQON
    model, converted = convert_to_falqon(model, target_modules, rank)
    print(f"Converted {len(converted)} layers to FalqonLinearEmulated")
    for name in converted[:4]:
        print(f"  - {name}")
    if len(converted) > 4:
        print(f"  ... and {len(converted) - 4} more")

    # Only B parameters are trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)} (all B matrices)")
    total_params = sum(p.numel() for p in trainable)
    print(f"Total trainable params: {total_params}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    # Synthetic data (generate on CPU, move to device)
    batch = 1
    h = w = cfg["sample_size"]
    in_ch = cfg["in_channels"]
    enc_seq = 16
    g = torch.Generator().manual_seed(42)

    latent = torch.randn(batch, in_ch, h, w, generator=g).to(device)
    enc = torch.randn(batch, enc_seq, cfg["caption_channels"], generator=g).to(device)
    timestep = torch.tensor([500.0], device=device)
    noise = torch.randn(batch, in_ch, h, w, generator=g).to(device)

    # Flow matching target
    sigma = torch.tensor([0.5], device=device).reshape(1, 1, 1, 1)
    noisy_input = (1.0 - sigma) * latent + sigma * noise
    target = noise - latent  # velocity target

    # --- Training Step ---
    print("\n--- Step 1 ---")
    optimizer.zero_grad()

    pred = model(
        hidden_states=noisy_input,
        encoder_hidden_states=enc,
        timestep=timestep,
        return_dict=False,
    )[0]

    loss = F.mse_loss(pred, target)
    print(f"Loss: {loss.item():.6f}")
    loss.backward()

    # Check B gradients exist
    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    print(f"All B matrices have non-zero gradients: {has_grad}")
    assert has_grad, "B matrices should have gradients after backward"

    optimizer.step()

    # Weight fusion (outside compile graph)
    apply_delta_weight_all(model)

    # Verify B is zeroed after fusion
    all_zero = all(
        m.B.data.abs().sum() == 0
        for m in model.modules()
        if isinstance(m, FalqonLinearEmulated)
    )
    print(f"All B matrices zeroed after fusion: {all_zero}")
    assert all_zero, "B should be zero after apply_delta_weight"

    # --- Step 2: verify model still works after fusion ---
    print("\n--- Step 2 ---")
    optimizer.zero_grad()

    pred2 = model(
        hidden_states=noisy_input,
        encoder_hidden_states=enc,
        timestep=timestep,
        return_dict=False,
    )[0]

    loss2 = F.mse_loss(pred2, target)
    print(f"Loss: {loss2.item():.6f}")
    loss2.backward()
    optimizer.step()
    apply_delta_weight_all(model)

    print(f"\nLoss change: {loss.item():.6f} -> {loss2.item():.6f}")
    print("FALQON Training Step test done.")


# ===================================================================
# Test 8: Flow Matching Training Loop (multiple steps)
# ===================================================================
def run_falqon_flow_matching_test(device, num_steps=5, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print(f"[Test 8] FALQON Flow Matching Training Loop ({num_steps} steps)")
    print("=" * 60)

    cfg = SMALL_SANA_CONFIG
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    # Build and convert model
    model = SanaTransformer2DModel(**cfg)
    model.requires_grad_(False)
    model, converted = convert_to_falqon(model, target_modules, rank)
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    batch = 1
    h = w = cfg["sample_size"]
    in_ch = cfg["in_channels"]
    enc_seq = 16

    losses = []
    for step in range(num_steps):
        g = torch.Generator().manual_seed(step)
        latent = torch.randn(batch, in_ch, h, w, generator=g).to(device)
        noise = torch.randn(batch, in_ch, h, w, generator=g).to(device)
        enc = torch.randn(batch, enc_seq, cfg["caption_channels"], generator=g).to(device)

        # Flow matching: sample random sigma
        u = torch.rand(batch, generator=torch.Generator().manual_seed(step + 100))
        sigma = u.reshape(batch, 1, 1, 1).to(device)

        noisy_input = (1.0 - sigma) * latent + sigma * noise
        target = noise - latent
        timestep = (u * 1000).to(device)

        # Training step
        optimizer.zero_grad()
        pred = model(
            hidden_states=noisy_input,
            encoder_hidden_states=enc,
            timestep=timestep,
            return_dict=False,
        )[0]
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        # FALQON weight fusion
        apply_delta_weight_all(model)

        losses.append(loss.item())
        print(f"  Step {step+1}/{num_steps}: loss = {loss.item():.6f}")

    print(f"\nLoss trajectory: {' -> '.join(f'{l:.4f}' for l in losses)}")
    print(f"Loss change: {losses[0]:.6f} -> {losses[-1]:.6f} (delta={losses[-1]-losses[0]:.6f})")
    print("FALQON Flow Matching test done.")


# ===================================================================
# Test 5b: SANA Transformer Block with FALQON conversion
# ===================================================================
@torch.no_grad()
def run_sana_falqon_block_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 5b] SanaTransformerBlock + FALQON conversion")
    print("=" * 60)

    dim = INNER_DIM
    cfg = SMALL_SANA_CONFIG
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    base_block = SanaTransformerBlock(
        dim=dim,
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        dropout=cfg["dropout"],
        num_cross_attention_heads=cfg["num_cross_attention_heads"],
        cross_attention_head_dim=cfg["cross_attention_head_dim"],
        cross_attention_dim=cfg["cross_attention_dim"],
        attention_bias=cfg["attention_bias"],
        norm_elementwise_affine=cfg["norm_elementwise_affine"],
        norm_eps=cfg["norm_eps"],
        mlp_ratio=cfg["mlp_ratio"],
        qk_norm=cfg["qk_norm"],
    ).eval()

    # Convert to FALQON
    base_block.requires_grad_(False)
    base_block, converted = convert_to_falqon(base_block, target_modules, rank)
    print(f"Converted {len(converted)} layers in block")

    cpu_block = copy.deepcopy(base_block).eval()
    dev_block = base_block.to(device).eval()
    dev_block_compiled = torch.compile(dev_block, dynamic=False)

    batch, seq, h, w = 1, 64, 8, 8
    enc_seq = 16
    g = torch.Generator().manual_seed(0)

    hidden_cpu = torch.randn(batch, seq, dim, generator=g)
    enc_cpu = torch.randn(batch, enc_seq, cfg["cross_attention_dim"], generator=g)
    timestep_cpu = torch.randn(batch, 6, dim, generator=g)

    hidden_dev = hidden_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    timestep_dev = timestep_cpu.to(device)

    out_cpu = cpu_block(
        hidden_states=hidden_cpu,
        encoder_hidden_states=enc_cpu,
        timestep=timestep_cpu,
        height=h,
        width=w,
    )
    out_dev = dev_block_compiled(
        hidden_states=hidden_dev,
        encoder_hidden_states=enc_dev,
        timestep=timestep_dev,
        height=h,
        width=w,
    )

    test_result("SanaTransformerBlock+FALQON forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("SanaTransformerBlock+FALQON test done.")


# ===================================================================
# Device setup
# ===================================================================
def get_device(device_str):
    if device_str == "npu":
        sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
        from Scheduler.scheduler import PyTorchSimRunner
        PyTorchSimRunner.setup_device()
        torch.compiler.is_compiling = lambda: True  # FIXME: same workaround as test_llama.py
        return torch.device("npu:0")
    elif device_str == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


# ===================================================================
# Main
# ===================================================================
TEST_REGISTRY = {
    "falqon_linear": [run_falqon_linear_forward_test, run_falqon_linear_backward_test],
    "linear_attention": [run_sana_linear_attention_test],
    "cross_attention": [run_sana_cross_attention_test],
    "mixffn": [run_sana_mixffn_test],
    "transformer_block": [run_sana_transformer_block_test],
    "falqon_block": [run_sana_falqon_block_test],
    "transformer_forward": [run_sana_transformer_forward_test],
    "training_step": [run_falqon_training_step_test],
    "flow_matching": [run_falqon_flow_matching_test],
}

ALL_TEST_ORDER = [
    "falqon_linear",
    "linear_attention",
    "cross_attention",
    "mixffn",
    "transformer_block",
    "falqon_block",
    "transformer_forward",
    "training_step",
    "flow_matching",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SANA + FALQON PyTorchSim Test")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "npu", "cpu"],
                        help="Device: cuda (Phase 1), npu (Phase 2 PyTorchSim), cpu")
    parser.add_argument("--tests", type=str, default="all",
                        help="Comma-separated test names or 'all'. "
                             f"Available: {', '.join(ALL_TEST_ORDER)}")
    parser.add_argument("--rank", type=int, default=4, help="FALQON LoRA rank")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Number of steps for flow_matching test")
    args = parser.parse_args()

    FALQON_RANK = args.rank
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"FALQON rank: {FALQON_RANK}")
    print(f"Model inner_dim: {INNER_DIM}")

    # Select tests
    if args.tests == "all":
        selected = ALL_TEST_ORDER
    else:
        selected = [t.strip() for t in args.tests.split(",")]
        for t in selected:
            if t not in TEST_REGISTRY:
                print(f"Unknown test: {t}. Available: {', '.join(ALL_TEST_ORDER)}")
                sys.exit(1)

    # Run selected tests
    for test_name in selected:
        for test_fn in TEST_REGISTRY[test_name]:
            if test_fn == run_falqon_flow_matching_test:
                test_fn(device, num_steps=args.num_steps, rtol=args.rtol, atol=args.atol)
            else:
                test_fn(device, rtol=args.rtol, atol=args.atol)

    print("\n" + "=" * 60)
    print("ALL SELECTED TESTS PASSED")
    print("=" * 60)