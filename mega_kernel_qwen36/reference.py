from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    return (x_f * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def moe_decode_reference(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    *,
    eps: float = 1e-6,
    top_k: int = 8,
    add_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference MoE decode path.

    Shapes:
    - x: [batch, hidden]
    - router_weight: [experts, hidden]
    - expert_gate/expert_up: [experts, intermediate, hidden]
    - expert_down: [experts, hidden, intermediate]
    - shared_gate/shared_up: [shared_intermediate, hidden]
    - shared_down: [hidden, shared_intermediate]
    """

    x_norm = rms_norm(x, norm_weight, eps).float()
    logits = x_norm @ router_weight.float().t()
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)

    batch, hidden = x.shape
    out = torch.zeros((batch, hidden), device=x.device, dtype=torch.float32)

    for b in range(batch):
        for slot in range(top_k):
            expert = int(topk_indices[b, slot])
            weight = topk_weights[b, slot]
            gate = F.silu(expert_gate[expert].float() @ x_norm[b])
            up = expert_up[expert].float() @ x_norm[b]
            out[b] += weight * (expert_down[expert].float() @ (gate * up))

    shared = F.silu(x_norm @ shared_gate.float().t()) * (x_norm @ shared_up.float().t())
    out += shared @ shared_down.float().t()
    if add_residual:
        out += x.float()
    return out.to(x.dtype), topk_indices, topk_weights


def dense_ffn_decode_reference(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    add_residual: bool = True,
) -> torch.Tensor:
    """Reference dense SwiGLU decode FFN for Qwen/Qwen3.6-27B.

    Shapes:
    - x: [batch, hidden]
    - gate_weight/up_weight: [intermediate, hidden]
    - down_weight: [hidden, intermediate]
    """

    x_norm = rms_norm(x, norm_weight, eps).float()
    act = F.silu(x_norm @ gate_weight.float().t()) * (x_norm @ up_weight.float().t())
    out = act @ down_weight.float().t()
    if add_residual:
        out += x.float()
    return out.to(x.dtype)


def apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None or rotary_dim <= 0:
        return q, k

    def _rotate_half(t: torch.Tensor) -> torch.Tensor:
        left, right = t[..., : rotary_dim // 2], t[..., rotary_dim // 2 : rotary_dim]
        return torch.cat((-right, left), dim=-1)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos.to(device=q.device, dtype=torch.float32)
    sin = sin.to(device=q.device, dtype=torch.float32)
    while cos.ndim < q_rot.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_out = q_rot.float() * cos + _rotate_half(q_rot.float()) * sin
    k_out = k_rot.float() * cos + _rotate_half(k_rot.float()) * sin
    return torch.cat((q_out.to(q.dtype), q_pass), dim=-1), torch.cat((k_out.to(k.dtype), k_pass), dim=-1)


def full_attention_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference one-token GQA attention.

    Shapes:
    - q: [batch, query_heads, head_dim]
    - k_cache/v_cache: [batch, seq_len, kv_heads, head_dim]
    """

    batch, query_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    if query_heads % kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    group = query_heads // kv_heads
    outputs = []
    for qh in range(query_heads):
        kvh = qh // group
        scores = torch.einsum("bd,btd->bt", q[:, qh].float(), k_cache[:, :, kvh].float()) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("bt,btd->bd", probs, v_cache[:, :, kvh].float()))
    return torch.stack(outputs, dim=1).to(q.dtype)


def linear_attention_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    *,
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic FP32 recurrent linear-attention update.

    Shapes:
    - q/k: [batch, heads, key_dim]
    - v: [batch, heads, value_dim]
    - state: [batch, heads, key_dim, value_dim], FP32 preferred
    """

    new_state = state.float().mul(decay) + torch.einsum("bhd,bhv->bhdv", k.float(), v.float())
    out = torch.einsum("bhd,bhdv->bhv", q.float(), new_state)
    return out.to(v.dtype), new_state


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    bias_f = None if bias is None else bias.float()
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias_f, eps).to(x.dtype)


def vision_patch_embed_reference(
    pixel_values: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: Optional[torch.Tensor] = None,
    *,
    in_channels: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int = 16,
) -> torch.Tensor:
    """Qwen3.6 vision Conv3D patch embed.

    ``pixel_values`` follows the Hugging Face processor layout:
    ``[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]``.
    """

    patches = pixel_values.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
    out = F.conv3d(patches.to(dtype=proj_weight.dtype), proj_weight, proj_bias)
    return out.view(-1, proj_weight.shape[0])


def vision_make_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Build Qwen3-VL frame-level cumulative sequence lengths from ``grid_thw``."""

    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


def vision_fast_pos_embed_interpolate_reference(
    grid_thw: torch.Tensor,
    pos_embed_weight: torch.Tensor,
    *,
    spatial_merge_size: int = 2,
    num_position_embeddings: Optional[int] = None,
) -> torch.Tensor:
    """Reference Qwen3-VL absolute position interpolation for patch tokens."""

    if num_position_embeddings is None:
        num_position_embeddings = pos_embed_weight.shape[0]
    num_grid_per_side = int(num_position_embeddings**0.5)
    device = pos_embed_weight.device
    dtype = pos_embed_weight.dtype

    idx_chunks: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_chunks: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.to(device=device):
        frames = int(t.item())
        height = int(h.item())
        width = int(w.item())
        h_idxs = torch.linspace(0, num_grid_per_side - 1, height, device=device, dtype=torch.float32)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, width, device=device, dtype=torch.float32)

        h_floor = h_idxs.floor().long()
        w_floor = w_idxs.floor().long()
        h_ceil = (h_floor + 1).clamp(max=num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clamp(max=num_grid_per_side - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()
        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side

        indices = (
            (base_h[:, None] + w_floor[None, :]).flatten(),
            (base_h[:, None] + w_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
        )
        weights = (
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        )
        for i in range(4):
            idx_chunks[i].append(indices[i])
            weight_chunks[i].append(weights[i])

    idx_tensor = torch.stack([torch.cat(chunks) for chunks in idx_chunks])
    weight_tensor = torch.stack([torch.cat(chunks) for chunks in weight_chunks]).to(dtype=dtype)
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds.sum(dim=0)

    split_sizes = [int(h.item() * w.item()) for h, w in grid_thw[:, 1:].to(device="cpu")]
    patch_pos_embeds = patch_pos_embeds.split(split_sizes)

    reordered = []
    for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_thw.to(device="cpu")):
        frames = int(t.item())
        height = int(h.item())
        width = int(w.item())
        pos_embed = pos_embed.repeat(frames, 1)
        pos_embed = (
            pos_embed.view(
                frames,
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        reordered.append(pos_embed)
    return torch.cat(reordered, dim=0)


def vision_rotary_position_embeddings_reference(
    grid_thw: torch.Tensor,
    *,
    head_dim: int,
    spatial_merge_size: int = 2,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Qwen3-VL vision rotary ``cos``/``sin`` embeddings with shape ``[tokens, head_dim]``."""

    device = grid_thw.device
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    max_hw = int(grid_thw[:, 1:].max().item())
    seq = torch.arange(max_hw, device=device, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        frames = int(num_frames.item())
        h = int(height.item())
        w = int(width.item())
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(spatial_merge_size, device=device)
        intra_col = torch.arange(spatial_merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * spatial_merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * spatial_merge_size + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, spatial_merge_size, spatial_merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if frames > 1:
            coords = coords.repeat(frames, 1)

        count = coords.shape[0]
        pos_ids[offset : offset + count] = coords
        offset += count

    rotary_pos_emb = freq_table[pos_ids].flatten(1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb_vision_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_f = q.float()
    k_f = k.float()
    q_embed = (q_f * cos) + (torch.cat((-q_f[..., q.shape[-1] // 2 :], q_f[..., : q.shape[-1] // 2]), dim=-1) * sin)
    k_embed = (k_f * cos) + (torch.cat((-k_f[..., k.shape[-1] // 2 :], k_f[..., : k.shape[-1] // 2]), dim=-1) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def vision_attention_encode_reference(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference non-causal Qwen3-VL vision self-attention over frame chunks."""

    seq_length, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    qkv = F.linear(hidden_states.float(), qkv_weight.float(), qkv_bias.float())
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    query, key, value = qkv.unbind(0)
    query, key = apply_rotary_pos_emb_vision_reference(query, key, cos, sin)

    cu = cu_seqlens.detach().cpu().tolist()
    outputs = []
    for start, end in zip(cu[:-1], cu[1:]):
        q = query[start:end].transpose(0, 1).float()
        k = key[start:end].transpose(0, 1).float()
        v = value[start:end].transpose(0, 1).float()
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v).transpose(0, 1))

    attn = torch.cat(outputs, dim=0).reshape(seq_length, hidden_size)
    out = F.linear(attn, proj_weight.float(), proj_bias.float())
    return out.to(hidden_states.dtype)


def vision_mlp_encode_reference(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    *,
    hidden_act: str = "gelu_pytorch_tanh",
) -> torch.Tensor:
    pre = F.linear(hidden_states.float(), fc1_weight.float(), fc1_bias.float())
    if hidden_act == "gelu_pytorch_tanh":
        act = F.gelu(pre, approximate="tanh")
    elif hidden_act == "gelu":
        act = F.gelu(pre)
    else:
        raise ValueError(f"unsupported vision activation: {hidden_act}")
    return F.linear(act, fc2_weight.float(), fc2_bias.float()).to(hidden_states.dtype)


def vision_patch_merger_reference(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    *,
    use_postshuffle_norm: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    merged_hidden = fc1_weight.shape[1]
    if use_postshuffle_norm:
        x = hidden_states.view(-1, merged_hidden)
        x = layer_norm_reference(x, norm_weight, norm_bias, eps=eps)
    else:
        x = layer_norm_reference(hidden_states, norm_weight, norm_bias, eps=eps).view(-1, merged_hidden)
    x = F.linear(x.float(), fc1_weight.float(), fc1_bias.float())
    x = F.gelu(x)
    return F.linear(x, fc2_weight.float(), fc2_bias.float()).to(hidden_states.dtype)


def vision_encoder_block_reference(
    hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    hidden_act: str = "gelu_pytorch_tanh",
    eps: float = 1e-6,
) -> torch.Tensor:
    normed = layer_norm_reference(hidden_states, norm1_weight, norm1_bias, eps=eps)
    hidden_states = hidden_states + vision_attention_encode_reference(
        normed,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        cu_seqlens,
        cos,
        sin,
        num_heads=num_heads,
    )
    normed = layer_norm_reference(hidden_states, norm2_weight, norm2_bias, eps=eps)
    hidden_states = hidden_states + vision_mlp_encode_reference(
        normed,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        hidden_act=hidden_act,
    )
    return hidden_states
