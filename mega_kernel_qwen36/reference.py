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
