from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .config import qwen36_layer_type
from .reference import (
    apply_partial_rope,
    dense_ffn_decode_reference,
    full_attention_decode_reference,
    linear_attention_decode_reference,
    moe_decode_reference,
)

try:
    from . import _C
except Exception:  # pragma: no cover - exercised when extension is not built.
    _C = None


@dataclass(frozen=True)
class MoeWeights:
    norm_weight: torch.Tensor
    router_weight: torch.Tensor
    expert_gate: torch.Tensor
    expert_up: torch.Tensor
    expert_down: torch.Tensor
    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor


@dataclass(frozen=True)
class DenseFfnWeights:
    norm_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor


@dataclass(frozen=True)
class FullAttentionInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    rotary_dim: int = 64
    scale: Optional[float] = None


@dataclass(frozen=True)
class LinearAttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    state: torch.Tensor
    decay: float = 1.0


@dataclass(frozen=True)
class DecodeLayerResult:
    hidden: torch.Tensor
    router_indices: Optional[torch.Tensor] = None
    router_weights: Optional[torch.Tensor] = None
    attention_output: Optional[torch.Tensor] = None
    linear_output: Optional[torch.Tensor] = None
    linear_state: Optional[torch.Tensor] = None


def extension_available() -> bool:
    return _C is not None


def _use_extension(*tensors: torch.Tensor) -> bool:
    return _C is not None and all(t.is_cuda for t in tensors)


def moe_decode(
    x: torch.Tensor,
    weights: MoeWeights,
    *,
    eps: float = 1e-6,
    top_k: int = 8,
    add_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (
        x,
        weights.norm_weight,
        weights.router_weight,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
        weights.shared_gate,
        weights.shared_up,
        weights.shared_down,
    )
    if _use_extension(*tensors):
        return _C.moe_decode(*tensors, float(eps), int(top_k), bool(add_residual))
    return moe_decode_reference(
        x,
        weights.norm_weight,
        weights.router_weight,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
        weights.shared_gate,
        weights.shared_up,
        weights.shared_down,
        eps=eps,
        top_k=top_k,
        add_residual=add_residual,
    )


def dense_ffn_decode(
    x: torch.Tensor,
    weights: DenseFfnWeights,
    *,
    eps: float = 1e-6,
    add_residual: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    tensors = (
        x,
        weights.norm_weight,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
    )
    if backend not in {"auto", "cuda", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, cuda, triton, reference")
    if backend in {"auto", "cuda"} and _use_extension(*tensors):
        return _C.dense_ffn_decode(*tensors, float(eps), bool(add_residual))
    if backend == "cuda":
        raise RuntimeError("CUDA extension is unavailable for dense_ffn_decode")
    if backend == "triton":
        from .triton_ops import dense_ffn_decode_triton

        return dense_ffn_decode_triton(
            x,
            weights.norm_weight,
            weights.gate_weight,
            weights.up_weight,
            weights.down_weight,
            eps=eps,
            add_residual=add_residual,
        )
    return dense_ffn_decode_reference(
        x,
        weights.norm_weight,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
        eps=eps,
        add_residual=add_residual,
    )


def full_attention_decode(inputs: FullAttentionInputs) -> torch.Tensor:
    q, k_cache = apply_partial_rope(inputs.q, inputs.k_cache, inputs.cos, inputs.sin, inputs.rotary_dim)
    scale = inputs.scale
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if _use_extension(q, k_cache, inputs.v_cache):
        return _C.full_attention_decode(q.contiguous(), k_cache.contiguous(), inputs.v_cache.contiguous(), float(scale))
    return full_attention_decode_reference(q, k_cache, inputs.v_cache, scale=scale)


def linear_attention_decode(inputs: LinearAttentionInputs) -> tuple[torch.Tensor, torch.Tensor]:
    if _use_extension(inputs.q, inputs.k, inputs.v, inputs.state):
        return _C.linear_attention_decode(
            inputs.q.contiguous(),
            inputs.k.contiguous(),
            inputs.v.contiguous(),
            inputs.state.contiguous(),
            float(inputs.decay),
        )
    return linear_attention_decode_reference(inputs.q, inputs.k, inputs.v, inputs.state, decay=inputs.decay)


def decode_layer(
    layer_idx: int,
    hidden: torch.Tensor,
    moe: Optional[MoeWeights] = None,
    *,
    dense_ffn: Optional[DenseFfnWeights] = None,
    full_attention: Optional[FullAttentionInputs] = None,
    linear_attention: Optional[LinearAttentionInputs] = None,
    layer_type: Optional[str] = None,
    eps: float = 1e-6,
    top_k: int = 8,
) -> DecodeLayerResult:
    kind = layer_type or qwen36_layer_type(layer_idx)
    attention_output = None
    linear_output = None
    linear_state = None
    residual = hidden

    if kind == "full_attention":
        if full_attention is None:
            raise ValueError("full_attention inputs are required for full_attention layers")
        attention_output = full_attention_decode(full_attention).reshape_as(hidden)
        residual = residual + attention_output
    elif kind == "linear_attention":
        if linear_attention is None:
            raise ValueError("linear_attention inputs are required for linear_attention layers")
        linear_output, linear_state = linear_attention_decode(linear_attention)
        linear_output = linear_output.reshape_as(hidden)
        residual = residual + linear_output
    else:
        raise ValueError(f"unsupported layer_type: {kind}")

    if dense_ffn is not None:
        out = dense_ffn_decode(residual, dense_ffn, eps=eps, add_residual=True)
        router_indices = None
        router_weights = None
    else:
        if moe is None:
            raise ValueError("either moe or dense_ffn weights are required")
        out, router_indices, router_weights = moe_decode(residual, moe, eps=eps, top_k=top_k, add_residual=True)
    return DecodeLayerResult(
        hidden=out,
        router_indices=router_indices,
        router_weights=router_weights,
        attention_output=attention_output,
        linear_output=linear_output,
        linear_state=linear_state,
    )
