"""
FLUX.2-Klein-4B Diffusion Transformer + FALQON (FP8 LoRA) PyTorchSim Test

Bottom-up block-level tests following test_sana_falqon.py patterns.
Each block is independently compiled and compared (CPU vs Device) for cycle measurement.

Note: Uses FluxTransformer2DModel (diffusers 0.34.x) as the closest available
approximation to FLUX.2-klein-4B architecture (Flux2Transformer2DModel requires
diffusers >= 0.37.0).

Phase 1: conda diffusers + CUDA verification (--device cuda)
Phase 2: PyTorchSim NPU simulation          (--device npu)

Usage:
    python test_flux2_klein.py --device cuda --tests all
    python test_flux2_klein.py --device cuda --tests falqon_linear
    python test_flux2_klein.py --device npu  --tests all
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
# Imports from diffusers (FLUX transformer components)
# ---------------------------------------------------------------------------
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
    FluxPosEmbed,
)


# ===================================================================
# Small model config for simulation
# Mirrors FLUX.2-klein-4B architecture but scaled down for NPU sim.
# Key constraints: sum(axes_dims_rope) == attention_head_dim
# ===================================================================
SMALL_FLUX2_KLEIN_CONFIG = dict(
    patch_size=1,
    in_channels=16,              # packed latent channels (VAE 16-ch)
    out_channels=16,
    num_layers=2,                # dual-stream MMDiT blocks (real: 5)
    num_single_layers=4,         # single-stream blocks (real: 20)
    attention_head_dim=64,       # per-head dim (real: 128)
    num_attention_heads=4,       # heads (real: 24); hidden_dim=4*64=256
    joint_attention_dim=256,     # text encoder output dim (real: 7680)
    pooled_projection_dim=64,    # pooled text dim (real: 3072)
    guidance_embeds=False,       # distilled, no CFG
    axes_dims_rope=(16, 24, 24), # RoPE per-axis dims; must sum to attention_head_dim=64
)

HIDDEN_DIM = (
    SMALL_FLUX2_KLEIN_CONFIG["num_attention_heads"]
    * SMALL_FLUX2_KLEIN_CONFIG["attention_head_dim"]
)  # 256
FALQON_RANK = 4

# Latent spatial size for simulation (H x W = img_seq_len)
SIM_H = 4
SIM_W = 4
SIM_TXT_SEQ = 8


# ===================================================================
# Helpers: positional IDs and RoPE embeddings
# ===================================================================
def make_img_ids(height, width):
    """Returns img_ids of shape (height*width, 3) — 2D, unbatched."""
    ids = torch.zeros(height * width, 3)
    ids[:, 1] = torch.arange(height * width).float() // width  # row
    ids[:, 2] = torch.arange(height * width).float() % width   # col
    return ids


def make_txt_ids(txt_seq):
    """Returns txt_ids of shape (txt_seq, 3) — 2D, unbatched."""
    return torch.zeros(txt_seq, 3)


def make_image_rotary_emb(img_seq, txt_seq, axes_dims_rope, device="cpu"):
    """Compute RoPE (cos, sin) for combined img+txt sequence."""
    pe = FluxPosEmbed(theta=10000, axes_dim=list(axes_dims_rope)).to(device)
    ids = torch.zeros(img_seq + txt_seq, 3, device=device)
    ids[:img_seq, 1] = torch.arange(img_seq, device=device).float() // SIM_W
    ids[:img_seq, 2] = torch.arange(img_seq, device=device).float() % SIM_W
    with torch.no_grad():
        cos, sin = pe(ids)
    return cos, sin


# ===================================================================
# Test result comparison (same pattern as test_sana_falqon.py)
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
# FALQON Emulated Linear Layer (same as test_sana_falqon.py)
# ===================================================================
def _low_rank_decomposition(weight, rank):
    """SVD-based low-rank decomposition: weight ≈ L @ R"""
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    sqrt_S = torch.sqrt(S[:rank])
    L = U[:, :rank] * sqrt_S[None, :]
    R = sqrt_S[:, None] * Vh[:rank, :]
    return L, R


def _quantize_emulate(weight):
    """Emulate FP8 quantization via round-trip cast to float16."""
    return weight.half().float()


class FalqonMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, B, out_features, rank):
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
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
        grad_input = grad_output_2d @ W_main
        grad_input = grad_input.reshape(orig_shape)
        grad_B = grad_output_2d.t() @ A_out
        return grad_input, None, grad_B, None, None


class FalqonLinearEmulated(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = nn.Parameter(
            torch.zeros(out_features + rank, in_features), requires_grad=False
        )
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int):
        in_f = linear.in_features
        out_f = linear.out_features
        has_bias = linear.bias is not None
        module = cls(in_f, out_f, rank, bias=has_bias)
        W = linear.weight.data.float()
        Q_W = _quantize_emulate(W)
        E = W - Q_W
        L, R = _low_rank_decomposition(E, rank)
        A = R
        B = torch.zeros(out_f, rank, dtype=W.dtype)
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
        W_main = self.weight.data[:self.out_features, :]
        A = self.weight.data[self.out_features:, :]
        delta = self.B.data @ A
        W_new = W_main + delta
        W_new_q = _quantize_emulate(W_new)
        self.weight.data[:self.out_features, :] = W_new_q
        self.B.data.zero_()

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


# ===================================================================
# FLUX model FALQON conversion
# ===================================================================
def convert_to_falqon(model, target_modules, rank):
    """Replace target nn.Linear modules with FalqonLinearEmulated."""
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

    in_f, out_f, rank = HIDDEN_DIM, HIDDEN_DIM, FALQON_RANK
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

    in_f, out_f, rank = HIDDEN_DIM, HIDDEN_DIM, FALQON_RANK
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

    out_cpu = cpu_mod(x_cpu)
    loss_cpu = F.mse_loss(out_cpu, target_cpu)
    loss_cpu.backward()

    out_dev = dev_mod(x_dev)
    loss_dev = F.mse_loss(out_dev, target_dev)
    loss_dev.backward()

    test_result("FalqonLinear B.grad", dev_mod.B.grad, cpu_mod.B.grad, rtol=rtol, atol=atol)
    print("Max grad diff >", torch.max(torch.abs(dev_mod.B.grad.cpu().float() - cpu_mod.B.grad.float())).item())
    assert cpu_mod.weight.grad is None, "weight should not have gradient"
    print("Confirmed: weight.grad is None (frozen)")
    print("FalqonLinear backward test done.")


# ===================================================================
# Test 2: FLUX Double-stream block (FluxTransformerBlock)
# ===================================================================
@torch.no_grad()
def run_flux_double_stream_block_test(device, rtol=0.15, atol=0.15):
    print("\n" + "=" * 60)
    print("[Test 2] FLUX Double-stream Block (FluxTransformerBlock)")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    dim = HIDDEN_DIM
    heads = cfg["num_attention_heads"]
    head_dim = cfg["attention_head_dim"]

    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ

    base_block = FluxTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
    ).eval()

    cpu_block = copy.deepcopy(base_block).eval()
    dev_block = base_block.to(device).eval()
    dev_block_compiled = torch.compile(dev_block, dynamic=False)

    g = torch.Generator().manual_seed(0)
    hidden_cpu = torch.randn(B, img_seq, dim, generator=g)
    enc_cpu = torch.randn(B, txt_seq, dim, generator=g)
    temb_cpu = torch.randn(B, dim, generator=g)

    cos_cpu, sin_cpu = make_image_rotary_emb(img_seq, txt_seq, cfg["axes_dims_rope"], device="cpu")
    image_rotary_emb_cpu = (cos_cpu, sin_cpu)

    hidden_dev = hidden_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    image_rotary_emb_dev = (cos_cpu.to(device), sin_cpu.to(device))

    out_cpu = cpu_block(
        hidden_states=hidden_cpu,
        encoder_hidden_states=enc_cpu,
        temb=temb_cpu,
        image_rotary_emb=image_rotary_emb_cpu,
    )
    out_dev = dev_block_compiled(
        hidden_states=hidden_dev,
        encoder_hidden_states=enc_dev,
        temb=temb_dev,
        image_rotary_emb=image_rotary_emb_dev,
    )

    # out = (encoder_hidden_states, hidden_states) ordering from FluxTransformerBlock
    enc_out_dev, hidden_out_dev = out_dev[0], out_dev[1]
    enc_out_cpu, hidden_out_cpu = out_cpu[0], out_cpu[1]

    test_result("FluxDoubleStreamBlock hidden", hidden_out_dev, hidden_out_cpu, rtol=rtol, atol=atol)
    test_result("FluxDoubleStreamBlock encoder", enc_out_dev, enc_out_cpu, rtol=rtol, atol=atol)
    print("Max hidden diff >", torch.max(torch.abs(hidden_out_dev.cpu().float() - hidden_out_cpu.float())).item())
    print("Max encoder diff >", torch.max(torch.abs(enc_out_dev.cpu().float() - enc_out_cpu.float())).item())
    print("FLUX Double-stream Block test done.")


# ===================================================================
# Test 3: FLUX Single-stream block (FluxSingleTransformerBlock)
# ===================================================================
@torch.no_grad()
def run_flux_single_stream_block_test(device, rtol=0.15, atol=0.15):
    print("\n" + "=" * 60)
    print("[Test 3] FLUX Single-stream Block (FluxSingleTransformerBlock)")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    dim = HIDDEN_DIM
    heads = cfg["num_attention_heads"]
    head_dim = cfg["attention_head_dim"]

    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ
    seq = img_seq + txt_seq  # combined sequence for single-stream blocks

    base_block = FluxSingleTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
    ).eval()

    cpu_block = copy.deepcopy(base_block).eval()
    dev_block = base_block.to(device).eval()
    dev_block_compiled = torch.compile(dev_block, dynamic=False)

    g = torch.Generator().manual_seed(0)
    hidden_cpu = torch.randn(B, seq, dim, generator=g)
    temb_cpu = torch.randn(B, dim, generator=g)

    cos_cpu, sin_cpu = make_image_rotary_emb(img_seq, txt_seq, cfg["axes_dims_rope"], device="cpu")
    image_rotary_emb_cpu = (cos_cpu, sin_cpu)

    hidden_dev = hidden_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    image_rotary_emb_dev = (cos_cpu.to(device), sin_cpu.to(device))

    out_cpu = cpu_block(
        hidden_states=hidden_cpu,
        temb=temb_cpu,
        image_rotary_emb=image_rotary_emb_cpu,
    )
    out_dev = dev_block_compiled(
        hidden_states=hidden_dev,
        temb=temb_dev,
        image_rotary_emb=image_rotary_emb_dev,
    )

    test_result("FluxSingleStreamBlock forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("FLUX Single-stream Block test done.")


# ===================================================================
# Test 4: Full FluxTransformer2DModel forward
# ===================================================================
@torch.no_grad()
def run_flux_transformer_forward_test(device, rtol=0.15, atol=0.15):
    print("\n" + "=" * 60)
    print("[Test 4] FluxTransformer2DModel forward (full model)")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ

    base_model = FluxTransformer2DModel(**cfg).eval()

    cpu_model = copy.deepcopy(base_model).eval()
    dev_model = base_model.to(device).eval()

    # FluxPosEmbed uses torch.outer which creates index patterns the PyTorchSim
    # MLIR backend cannot lower. Disable dynamo for pos_embed so it runs in
    # eager mode (graph break), while the transformer blocks compile normally.
    import torch._dynamo
    dev_model.pos_embed.forward = torch._dynamo.disable(dev_model.pos_embed.forward)

    dev_model_compiled = torch.compile(dev_model, dynamic=False)

    g = torch.Generator().manual_seed(0)
    hidden_cpu = torch.randn(B, img_seq, cfg["in_channels"], generator=g)
    enc_cpu = torch.randn(B, txt_seq, cfg["joint_attention_dim"], generator=g)
    pooled_cpu = torch.randn(B, cfg["pooled_projection_dim"], generator=g)
    timestep_cpu = torch.tensor([0.5])

    img_ids_cpu = make_img_ids(SIM_H, SIM_W)
    txt_ids_cpu = make_txt_ids(txt_seq)

    hidden_dev = hidden_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    pooled_dev = pooled_cpu.to(device)
    timestep_dev = timestep_cpu.to(device)
    img_ids_dev = img_ids_cpu.to(device)
    txt_ids_dev = txt_ids_cpu.to(device)

    out_cpu = cpu_model(
        hidden_states=hidden_cpu,
        encoder_hidden_states=enc_cpu,
        pooled_projections=pooled_cpu,
        timestep=timestep_cpu,
        img_ids=img_ids_cpu,
        txt_ids=txt_ids_cpu,
        return_dict=False,
    )[0]

    out_dev = dev_model_compiled(
        hidden_states=hidden_dev,
        encoder_hidden_states=enc_dev,
        pooled_projections=pooled_dev,
        timestep=timestep_dev,
        img_ids=img_ids_dev,
        txt_ids=txt_ids_dev,
        return_dict=False,
    )[0]

    test_result("FluxTransformer2DModel forward", out_dev, out_cpu, rtol=rtol, atol=atol)
    print("Max diff >", torch.max(torch.abs(out_dev.cpu().float() - out_cpu.float())).item())
    print("FluxTransformer2DModel forward test done.")


# ===================================================================
# Test 5: FLUX Double-stream Block + FALQON conversion
# ===================================================================
@torch.no_grad()
def run_flux_falqon_block_test(device, rtol=0.15, atol=0.15):
    print("\n" + "=" * 60)
    print("[Test 5] FluxTransformerBlock + FALQON conversion")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    dim = HIDDEN_DIM
    heads = cfg["num_attention_heads"]
    head_dim = cfg["attention_head_dim"]
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    base_block = FluxTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
    ).eval()
    base_block.requires_grad_(False)
    base_block, converted = convert_to_falqon(base_block, target_modules, rank)
    print(f"Converted {len(converted)} layers in block")

    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ

    cpu_block = copy.deepcopy(base_block).eval()
    dev_block = base_block.to(device).eval()
    dev_block_compiled = torch.compile(dev_block, dynamic=False)

    g = torch.Generator().manual_seed(0)
    hidden_cpu = torch.randn(B, img_seq, dim, generator=g)
    enc_cpu = torch.randn(B, txt_seq, dim, generator=g)
    temb_cpu = torch.randn(B, dim, generator=g)

    cos_cpu, sin_cpu = make_image_rotary_emb(img_seq, txt_seq, cfg["axes_dims_rope"], device="cpu")
    image_rotary_emb_cpu = (cos_cpu, sin_cpu)

    hidden_dev = hidden_cpu.to(device)
    enc_dev = enc_cpu.to(device)
    temb_dev = temb_cpu.to(device)
    image_rotary_emb_dev = (cos_cpu.to(device), sin_cpu.to(device))

    out_cpu = cpu_block(
        hidden_states=hidden_cpu,
        encoder_hidden_states=enc_cpu,
        temb=temb_cpu,
        image_rotary_emb=image_rotary_emb_cpu,
    )
    out_dev = dev_block_compiled(
        hidden_states=hidden_dev,
        encoder_hidden_states=enc_dev,
        temb=temb_dev,
        image_rotary_emb=image_rotary_emb_dev,
    )

    enc_out_dev, hidden_out_dev = out_dev[0], out_dev[1]
    enc_out_cpu, hidden_out_cpu = out_cpu[0], out_cpu[1]

    test_result("FluxDoubleBlock+FALQON hidden", hidden_out_dev, hidden_out_cpu, rtol=rtol, atol=atol)
    test_result("FluxDoubleBlock+FALQON encoder", enc_out_dev, enc_out_cpu, rtol=rtol, atol=atol)
    print("Max hidden diff >", torch.max(torch.abs(hidden_out_dev.cpu().float() - hidden_out_cpu.float())).item())
    print("FluxTransformerBlock+FALQON test done.")


# ===================================================================
# Test 6: FALQON Training Step (forward + backward + optimizer + weight fusion)
# ===================================================================
def run_falqon_training_step_test(device, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print("[Test 6] FALQON Training Step (full training cycle)")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    model = FluxTransformer2DModel(**cfg)
    model.requires_grad_(False)
    model, converted = convert_to_falqon(model, target_modules, rank)
    print(f"Converted {len(converted)} layers to FalqonLinearEmulated")
    for name in converted[:4]:
        print(f"  - {name}")
    if len(converted) > 4:
        print(f"  ... and {len(converted) - 4} more")

    # Collect trainable refs AFTER .to(device): NPU .to() creates new Parameter objects
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable)} (all B matrices)")
    total_params = sum(p.numel() for p in trainable)
    print(f"Total trainable params: {total_params}")

    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ
    g = torch.Generator().manual_seed(42)

    hidden = torch.randn(B, img_seq, cfg["in_channels"], generator=g).to(device)
    enc = torch.randn(B, txt_seq, cfg["joint_attention_dim"], generator=g).to(device)
    pooled = torch.randn(B, cfg["pooled_projection_dim"], generator=g).to(device)
    timestep = torch.tensor([0.5], device=device)
    noise = torch.randn(B, img_seq, cfg["in_channels"], generator=g).to(device)

    # Flow matching: velocity target
    sigma = torch.tensor([0.5], device=device).reshape(1, 1, 1)
    noisy_input = (1.0 - sigma) * hidden + sigma * noise
    target = noise - hidden

    img_ids = make_img_ids(SIM_H, SIM_W).to(device)
    txt_ids = make_txt_ids(txt_seq).to(device)

    print("\n--- Step 1 ---")
    optimizer.zero_grad()

    pred = model(
        hidden_states=noisy_input,
        encoder_hidden_states=enc,
        pooled_projections=pooled,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]

    loss = F.mse_loss(pred, target)
    print(f"Loss: {loss.item():.6f}")
    loss.backward()

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    print(f"All B matrices have non-zero gradients: {has_grad}")
    assert has_grad, "B matrices should have gradients after backward"

    optimizer.step()
    apply_delta_weight_all(model)

    all_zero = all(
        m.B.data.abs().sum() == 0
        for m in model.modules()
        if isinstance(m, FalqonLinearEmulated)
    )
    print(f"All B matrices zeroed after fusion: {all_zero}")
    assert all_zero, "B should be zero after apply_delta_weight"

    print("\n--- Step 2 ---")
    optimizer.zero_grad()

    pred2 = model(
        hidden_states=noisy_input,
        encoder_hidden_states=enc,
        pooled_projections=pooled,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
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
# Test 7: Flow Matching Training Loop (multiple steps)
# ===================================================================
def run_falqon_flow_matching_test(device, num_steps=5, rtol=1e-3, atol=1e-3):
    print("\n" + "=" * 60)
    print(f"[Test 7] FALQON Flow Matching Training Loop ({num_steps} steps)")
    print("=" * 60)

    cfg = SMALL_FLUX2_KLEIN_CONFIG
    rank = FALQON_RANK
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    model = FluxTransformer2DModel(**cfg)
    model.requires_grad_(False)
    model, converted = convert_to_falqon(model, target_modules, rank)
    # Collect trainable refs AFTER .to(device): NPU .to() creates new Parameter objects
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    B = 1
    img_seq = SIM_H * SIM_W
    txt_seq = SIM_TXT_SEQ

    img_ids = make_img_ids(SIM_H, SIM_W).to(device)
    txt_ids = make_txt_ids(txt_seq).to(device)

    losses = []
    for step in range(num_steps):
        g = torch.Generator().manual_seed(step)
        hidden = torch.randn(B, img_seq, cfg["in_channels"], generator=g).to(device)
        noise = torch.randn(B, img_seq, cfg["in_channels"], generator=g).to(device)
        enc = torch.randn(B, txt_seq, cfg["joint_attention_dim"], generator=g).to(device)
        pooled = torch.randn(B, cfg["pooled_projection_dim"], generator=g).to(device)

        u = torch.rand(1, generator=torch.Generator().manual_seed(step + 100))
        sigma = u.reshape(1, 1, 1).to(device)
        noisy_input = (1.0 - sigma) * hidden + sigma * noise
        target = noise - hidden
        timestep = u.to(device)

        optimizer.zero_grad()
        pred = model(
            hidden_states=noisy_input,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        apply_delta_weight_all(model)

        losses.append(loss.item())
        print(f"  Step {step+1}/{num_steps}: loss = {loss.item():.6f}")

    print(f"\nLoss trajectory: {' -> '.join(f'{l:.4f}' for l in losses)}")
    print(f"Loss change: {losses[0]:.6f} -> {losses[-1]:.6f} (delta={losses[-1]-losses[0]:.6f})")
    print("FALQON Flow Matching test done.")


# ===================================================================
# Device setup
# ===================================================================
def get_device(device_str):
    if device_str == "npu":
        sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
        from Scheduler.scheduler import PyTorchSimRunner
        PyTorchSimRunner.setup_device()
        torch.compiler.is_compiling = lambda: True  # same workaround as test_llama.py
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
    "double_stream_block": [run_flux_double_stream_block_test],
    "single_stream_block": [run_flux_single_stream_block_test],
    "transformer_forward": [run_flux_transformer_forward_test],
    "falqon_block": [run_flux_falqon_block_test],
    "training_step": [run_falqon_training_step_test],
    "flow_matching": [run_falqon_flow_matching_test],
}

ALL_TEST_ORDER = [
    "falqon_linear",
    "double_stream_block",
    "single_stream_block",
    "transformer_forward",
    "falqon_block",
    "training_step",
    "flow_matching",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX.2-Klein + FALQON PyTorchSim Test")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "npu", "cpu"],
                        help="Device: cuda (Phase 1), npu (Phase 2 PyTorchSim), cpu")
    parser.add_argument("--tests", type=str, default="all",
                        help="Comma-separated test names or 'all'. "
                             f"Available: {', '.join(ALL_TEST_ORDER)}")
    parser.add_argument("--rank", type=int, default=4, help="FALQON LoRA rank")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=None,
                        help="Override rtol for all tests (default: use per-test tolerance)")
    parser.add_argument("--atol", type=float, default=None,
                        help="Override atol for all tests (default: use per-test tolerance)")
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Number of steps for flow_matching test")
    args = parser.parse_args()

    FALQON_RANK = args.rank
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"FALQON rank: {FALQON_RANK}")
    print(f"Model hidden_dim: {HIDDEN_DIM}")
    print(f"Config: {SMALL_FLUX2_KLEIN_CONFIG}")

    if args.tests == "all":
        selected = ALL_TEST_ORDER
    else:
        selected = [t.strip() for t in args.tests.split(",")]
        for t in selected:
            if t not in TEST_REGISTRY:
                print(f"Unknown test: {t}. Available: {', '.join(ALL_TEST_ORDER)}")
                sys.exit(1)

    for test_name in selected:
        for test_fn in TEST_REGISTRY[test_name]:
            kwargs = {}
            if args.rtol is not None:
                kwargs["rtol"] = args.rtol
            if args.atol is not None:
                kwargs["atol"] = args.atol
            if test_fn == run_falqon_flow_matching_test:
                test_fn(device, num_steps=args.num_steps, **kwargs)
            else:
                test_fn(device, **kwargs)

    print("\n" + "=" * 60)
    print("ALL SELECTED TESTS PASSED")
    print("=" * 60)
