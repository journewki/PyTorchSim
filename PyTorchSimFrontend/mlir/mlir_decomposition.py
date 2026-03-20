# Registers decomposition rules for composite PyTorch ops (e.g., multi-head
# attention) into simpler operations that the MLIR backend can lower and compile.

import math
import torch
import torch.nn.functional as F
from torch._inductor.decomposition import register_decomposition

aten = torch.ops.aten

@register_decomposition(aten._native_multi_head_attention.default)
def decompose_native_multi_head_attention(
    query,
    key,
    value,
    embed_dim: int,
    num_heads: int,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights: bool = False,
):
    """
    Decompose _native_multi_head_attention into scaled_dot_product_attention operations.

    Based on F.scaled_dot_product_attention and nn.MultiheadAttention implementation:
    1. QKV projection (if needed - but query/key/value may already be projected)
    2. Reshape to multi-head format
    3. Scaled dot product: Q @ K^T / sqrt(head_dim)
    4. Softmax
    5. Attention @ V
    6. Reshape back and output projection
    """
    head_dim = embed_dim // num_heads
    scale_factor = 1.0 / math.sqrt(head_dim)

    # Get input shapes - assuming [batch, seq_len, embed_dim] format
    query_shape = query.shape
    if len(query_shape) == 3:
        # [batch, seq_len, embed_dim] format
        batch_size = query_shape[0]
        seq_len = query_shape[1]
    elif len(query_shape) == 2:
        # [seq_len, embed_dim] -> add batch dimension
        batch_size = 1
        seq_len = query_shape[0]
        query = query.unsqueeze(0)  # [1, seq_len, embed_dim]
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    else:
        # Fallback: assume first dim is batch, second is seq_len
        batch_size = query_shape[0] if len(query_shape) > 0 else 1
        seq_len = query_shape[1] if len(query_shape) > 1 else query_shape[0]

    # Step 1: QKV projection (if query/key/value are not already projected)
    # In many cases, query/key/value are already projected, so we check if qkv_weight is used
    # For now, assume they might need projection
    # Note: In practice, _native_multi_head_attention often receives already projected inputs

    # Reshape for projection: [batch, seq_len, embed_dim] -> [batch*seq_len, embed_dim]
    if len(query.shape) == 3:
        query_flat = query.view(-1, embed_dim)
        key_flat = key.view(-1, embed_dim)
        value_flat = value.view(-1, embed_dim)
    else:
        query_flat = query
        key_flat = key
        value_flat = value

    # QKV projection using qkv_weight and qkv_bias
    # Check if GQA (Grouped Query Attention) is used
    # Standard MHA: qkv_weight shape = [3*embed_dim, embed_dim]
    # GQA: qkv_weight shape = [embed_dim + 2*kv_embed_dim, embed_dim] where kv_embed_dim < embed_dim
    qkv_weight_total = qkv_weight.shape[0]

    # Determine if GQA: if qkv_weight is not exactly 3*embed_dim, it might be GQA
    if qkv_weight_total == 3 * embed_dim:
        # Standard MHA: split equally
        qkv_weight_q, qkv_weight_k, qkv_weight_v = torch.split(qkv_weight, embed_dim, dim=0)
        if qkv_bias is not None:
            qkv_bias_q, qkv_bias_k, qkv_bias_v = torch.split(qkv_bias, embed_dim, dim=0)
        else:
            qkv_bias_q = qkv_bias_k = qkv_bias_v = None
        kv_embed_dim = embed_dim
        kv_heads = num_heads
    else:
        # GQA: Q has embed_dim, K and V share the rest
        # Assume Q = embed_dim, K = V = (qkv_weight_total - embed_dim) / 2
        q_dim = embed_dim
        kv_dim = (qkv_weight_total - embed_dim) // 2
        qkv_weight_q = qkv_weight[:q_dim]
        qkv_weight_k = qkv_weight[q_dim:q_dim + kv_dim]
        qkv_weight_v = qkv_weight[q_dim + kv_dim:]
        if qkv_bias is not None:
            qkv_bias_q = qkv_bias[:q_dim]
            qkv_bias_k = qkv_bias[q_dim:q_dim + kv_dim]
            qkv_bias_v = qkv_bias[q_dim + kv_dim:]
        else:
            qkv_bias_q = qkv_bias_k = qkv_bias_v = None
        kv_embed_dim = kv_dim
        kv_heads = kv_embed_dim // head_dim  # Number of KV heads

    # Project Q, K, V
    q = torch.nn.functional.linear(query_flat, qkv_weight_q, qkv_bias_q)
    k = torch.nn.functional.linear(key_flat, qkv_weight_k, qkv_bias_k)
    v = torch.nn.functional.linear(value_flat, qkv_weight_v, qkv_bias_v)

    # Reshape back: [batch*seq_len, embed_dim] -> [batch, seq_len, embed_dim]
    q = q.view(batch_size, seq_len, embed_dim)
    k = k.view(batch_size, seq_len, kv_embed_dim)
    v = v.view(batch_size, seq_len, kv_embed_dim)

    # Step 2: Reshape to multi-head format
    # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, kv_heads, head_dim)
    v = v.view(batch_size, seq_len, kv_heads, head_dim)

    # Transpose to [batch, num_heads, seq_len, head_dim] for bmm
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)  # [batch, kv_heads, seq_len, head_dim]
    v = v.transpose(1, 2)  # [batch, kv_heads, seq_len, head_dim]

    # GQA: If key/value have fewer heads, repeat them to match query heads
    if kv_heads < num_heads:
        repeat_factor = num_heads // kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)  # [batch, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(repeat_factor, dim=1)  # [batch, num_heads, seq_len, head_dim]

    # Step 3: Scaled dot product attention
    # Scale Q
    q_scaled = q * scale_factor

    # Q @ K^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    k_transposed = k.transpose(-2, -1)  # [batch, num_heads, head_dim, seq_len]
    scores = torch.matmul(q_scaled, k_transposed)  # [batch, num_heads, seq_len, seq_len]

    # Step 4: Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias = mask + attn_bias

    # Step 5: Softmax along the last dimension (seq_len dimension)
    attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

    # Step 6: Attention @ V
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    # -> [batch, num_heads, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, v)

    # Step 7: Reshape back to [batch, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

    # Step 8: Output projection
    attn_output_flat = attn_output.view(-1, embed_dim)
    output = torch.nn.functional.linear(attn_output_flat, proj_weight, proj_bias)
    output = output.view(batch_size, seq_len, embed_dim)

    if need_weights:
        # Return attention weights: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
        attn_weights_mean = attn_weights.mean(dim=1)  # Average over heads
        return output, attn_weights_mean
    else:
        return (output, None)