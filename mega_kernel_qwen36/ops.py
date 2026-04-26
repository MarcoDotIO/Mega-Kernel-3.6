from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .config import Qwen36VisionConfig, qwen36_27b_vision_config, qwen36_layer_type
from .reference import (
    apply_partial_rope,
    dense_ffn_decode_reference,
    full_attention_decode_reference,
    layer_norm_reference,
    linear_attention_decode_reference,
    moe_decode_reference,
    rms_norm,
    vision_attention_encode_reference,
    vision_encoder_block_reference,
    vision_fast_pos_embed_interpolate_reference,
    vision_make_cu_seqlens,
    vision_mlp_encode_reference,
    vision_patch_embed_reference,
    vision_patch_merger_reference,
    vision_rotary_position_embeddings_reference,
)

try:
    from . import _C
except Exception:  # pragma: no cover - exercised when extension is not built.
    _C = None

_FLASH_ATTN_VARLEN = None
_FLASH_ATTN_IMPORT_ATTEMPTED = False


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
class VisionPatchEmbedWeights:
    proj_weight: torch.Tensor
    proj_bias: torch.Tensor


@dataclass(frozen=True)
class VisionAttentionWeights:
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    proj_weight: torch.Tensor
    proj_bias: torch.Tensor


@dataclass(frozen=True)
class VisionMlpWeights:
    fc1_weight: torch.Tensor
    fc1_bias: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_bias: torch.Tensor


@dataclass(frozen=True)
class VisionPatchMergerWeights:
    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_bias: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_bias: torch.Tensor


@dataclass(frozen=True)
class VisionBlockWeights:
    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    attention: VisionAttentionWeights
    norm2_weight: torch.Tensor
    norm2_bias: torch.Tensor
    mlp: VisionMlpWeights


@dataclass(frozen=True)
class VisionEncoderWeights:
    patch_embed: VisionPatchEmbedWeights
    pos_embed_weight: torch.Tensor
    blocks: Sequence[VisionBlockWeights]
    merger: VisionPatchMergerWeights
    deepstack_mergers: Sequence[VisionPatchMergerWeights] = ()


@dataclass(frozen=True)
class DecodeLayerResult:
    hidden: torch.Tensor
    router_indices: Optional[torch.Tensor] = None
    router_weights: Optional[torch.Tensor] = None
    attention_output: Optional[torch.Tensor] = None
    linear_output: Optional[torch.Tensor] = None
    linear_state: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class VisionEncoderResult:
    hidden: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...] = ()


def extension_available() -> bool:
    return _C is not None


def _use_extension(*tensors: torch.Tensor) -> bool:
    return _C is not None and all(t.is_cuda for t in tensors)


def _use_low_precision_matmul(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in {torch.float16, torch.bfloat16}


def _matmul_input_dtype(x: torch.Tensor) -> torch.dtype:
    return x.dtype if _use_low_precision_matmul(x) else torch.float32


def _moe_decode_grouped_torch(
    x: torch.Tensor,
    weights: MoeWeights,
    *,
    eps: float,
    top_k: int,
    add_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_norm_f = rms_norm(x, weights.norm_weight, eps).float()
    logits = x_norm_f @ weights.router_weight.float().t()
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)

    batch, hidden = x.shape
    intermediate = weights.expert_gate.shape[1]
    flat_indices = topk_indices.reshape(-1)
    matmul_dtype = _matmul_input_dtype(x)
    x_mm = x_norm_f.to(matmul_dtype)
    x_selected = x_mm[:, None, :].expand(batch, top_k, hidden).reshape(batch * top_k, hidden).contiguous()

    gate_w = weights.expert_gate.index_select(0, flat_indices).reshape(batch * top_k, intermediate, hidden)
    up_w = weights.expert_up.index_select(0, flat_indices).reshape(batch * top_k, intermediate, hidden)
    gate = torch.bmm(gate_w, x_selected.unsqueeze(-1)).squeeze(-1).float()
    up = torch.bmm(up_w, x_selected.unsqueeze(-1)).squeeze(-1).float()
    routed_act = F.silu(gate) * up * topk_weights.reshape(batch * top_k, 1)

    down_w = weights.expert_down.index_select(0, flat_indices).reshape(batch * top_k, hidden, intermediate)
    routed = torch.bmm(down_w, routed_act.to(matmul_dtype).unsqueeze(-1)).squeeze(-1).float()
    routed = routed.view(batch, top_k, hidden).sum(dim=1)

    shared_gate = F.linear(x_mm, weights.shared_gate).float()
    shared_up = F.linear(x_mm, weights.shared_up).float()
    shared_act = F.silu(shared_gate) * shared_up
    shared = F.linear(shared_act.to(matmul_dtype), weights.shared_down).float()

    out = routed + shared
    if add_residual:
        out = out + x.float()
    return out.to(x.dtype), topk_indices, topk_weights


def _dense_ffn_decode_torch(
    x: torch.Tensor,
    weights: DenseFfnWeights,
    *,
    eps: float,
    add_residual: bool,
) -> torch.Tensor:
    x_norm = rms_norm(x, weights.norm_weight, eps)
    matmul_dtype = _matmul_input_dtype(x)
    x_mm = x_norm.to(matmul_dtype)
    gate = F.linear(x_mm, weights.gate_weight).float()
    up = F.linear(x_mm, weights.up_weight).float()
    act = F.silu(gate) * up
    out = F.linear(act.to(matmul_dtype), weights.down_weight).float()
    if add_residual:
        out = out + x.float()
    return out.to(x.dtype)


def _flash_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float],
) -> Optional[torch.Tensor]:
    global _FLASH_ATTN_IMPORT_ATTEMPTED, _FLASH_ATTN_VARLEN
    if not query.is_cuda:
        return None
    if not _FLASH_ATTN_IMPORT_ATTEMPTED:
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception:
            _FLASH_ATTN_VARLEN = None
        else:
            _FLASH_ATTN_VARLEN = flash_attn_varlen_func
        _FLASH_ATTN_IMPORT_ATTEMPTED = True
    if _FLASH_ATTN_VARLEN is None:
        return None

    cu = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    kwargs = {
        "dropout_p": 0.0,
        "softmax_scale": scale,
        "causal": False,
    }
    try:
        return _FLASH_ATTN_VARLEN(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            **kwargs,
        )
    except TypeError:
        return _FLASH_ATTN_VARLEN(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            0.0,
            scale,
            False,
        )
    except Exception:
        return None


def moe_decode(
    x: torch.Tensor,
    weights: MoeWeights,
    *,
    eps: float = 1e-6,
    top_k: int = 8,
    add_residual: bool = True,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend not in {"auto", "cuda", "grouped", "reference"}:
        raise ValueError("backend must be one of: auto, cuda, grouped, reference")
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
    if backend in {"auto", "grouped"} and all(t.is_cuda for t in tensors):
        return _moe_decode_grouped_torch(x, weights, eps=eps, top_k=top_k, add_residual=add_residual)
    if backend == "grouped":
        return _moe_decode_grouped_torch(x, weights, eps=eps, top_k=top_k, add_residual=add_residual)
    if backend in {"auto", "cuda"} and _use_extension(*tensors):
        return _C.moe_decode(*tensors, float(eps), int(top_k), bool(add_residual))
    if backend == "cuda":
        raise RuntimeError("CUDA extension is unavailable for moe_decode")
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
    if backend not in {"auto", "cuda", "torch", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, cuda, torch, triton, reference")
    if backend == "auto" and all(t.is_cuda for t in tensors):
        return _dense_ffn_decode_torch(x, weights, eps=eps, add_residual=add_residual)
    if backend == "torch":
        return _dense_ffn_decode_torch(x, weights, eps=eps, add_residual=add_residual)
    if backend == "cuda" and _use_extension(*tensors):
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


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
    backend: str,
) -> torch.Tensor:
    if backend == "triton":
        from .triton_ops import layer_norm_triton

        return layer_norm_triton(x, weight, bias, eps)
    if backend == "reference":
        return layer_norm_reference(x, weight, bias, eps=eps)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def vision_patch_embed(
    pixel_values: torch.Tensor,
    weights: VisionPatchEmbedWeights,
    *,
    config: Optional[Qwen36VisionConfig] = None,
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "torch", "reference"}:
        raise ValueError("backend must be one of: auto, torch, reference")
    cfg = config or qwen36_27b_vision_config()
    if backend == "reference":
        return vision_patch_embed_reference(
            pixel_values,
            weights.proj_weight,
            weights.proj_bias,
            in_channels=cfg.in_channels,
            temporal_patch_size=cfg.temporal_patch_size,
            patch_size=cfg.patch_size,
        )
    patches = pixel_values.view(-1, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
    return F.conv3d(patches.to(dtype=weights.proj_weight.dtype), weights.proj_weight, weights.proj_bias).view(
        -1, weights.proj_weight.shape[0]
    )


def vision_attention_encode(
    hidden_states: torch.Tensor,
    weights: VisionAttentionWeights,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    scale: Optional[float] = None,
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "sdpa", "flash", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, sdpa, flash, triton, reference")
    if backend == "reference":
        return vision_attention_encode_reference(
            hidden_states,
            weights.qkv_weight,
            weights.qkv_bias,
            weights.proj_weight,
            weights.proj_bias,
            cu_seqlens,
            cos,
            sin,
            num_heads=num_heads,
            scale=scale,
        )

    seq_length, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads
    qkv = F.linear(hidden_states, weights.qkv_weight, weights.qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    query, key, value = qkv.unbind(0)
    query, key = _apply_rotary_pos_emb_vision(query, key, cos, sin)

    if backend in {"auto", "flash"}:
        flash_out = _flash_attn_varlen(query, key, value, cu_seqlens, scale)
        if flash_out is not None:
            attn = flash_out.reshape(seq_length, hidden_size)
            return F.linear(attn, weights.proj_weight, weights.proj_bias)
        if backend == "flash":
            raise RuntimeError("flash_attn_varlen_func is unavailable for vision_attention_encode")

    cu = cu_seqlens.detach().cpu().tolist()
    if len(cu) == 2:
        q = query.transpose(0, 1).unsqueeze(0)
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)
        if scale is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
        attn = out.squeeze(0).transpose(0, 1).reshape(seq_length, hidden_size)
        return F.linear(attn, weights.proj_weight, weights.proj_bias)

    outputs = []
    for start, end in zip(cu[:-1], cu[1:]):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        if scale is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
        outputs.append(out.squeeze(0).transpose(0, 1))
    attn = torch.cat(outputs, dim=0).reshape(seq_length, hidden_size)
    return F.linear(attn, weights.proj_weight, weights.proj_bias)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    q_f = q.float()
    k_f = k.float()
    cos_f = cos.to(device=q.device).unsqueeze(-2).float()
    sin_f = sin.to(device=q.device).unsqueeze(-2).float()
    q_half = torch.cat((-q_f[..., q.shape[-1] // 2 :], q_f[..., : q.shape[-1] // 2]), dim=-1)
    k_half = torch.cat((-k_f[..., k.shape[-1] // 2 :], k_f[..., : k.shape[-1] // 2]), dim=-1)
    return (q_f * cos_f + q_half * sin_f).to(q_dtype), (k_f * cos_f + k_half * sin_f).to(k_dtype)


def vision_mlp_encode(
    hidden_states: torch.Tensor,
    weights: VisionMlpWeights,
    *,
    hidden_act: str = "gelu_pytorch_tanh",
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "torch", "reference"}:
        raise ValueError("backend must be one of: auto, torch, reference")
    if backend == "reference":
        return vision_mlp_encode_reference(
            hidden_states,
            weights.fc1_weight,
            weights.fc1_bias,
            weights.fc2_weight,
            weights.fc2_bias,
            hidden_act=hidden_act,
        )
    x = F.linear(hidden_states, weights.fc1_weight, weights.fc1_bias)
    if hidden_act == "gelu_pytorch_tanh":
        x = F.gelu(x, approximate="tanh")
    elif hidden_act == "gelu":
        x = F.gelu(x)
    else:
        raise ValueError(f"unsupported vision activation: {hidden_act}")
    return F.linear(x, weights.fc2_weight, weights.fc2_bias)


def vision_patch_merger(
    hidden_states: torch.Tensor,
    weights: VisionPatchMergerWeights,
    *,
    use_postshuffle_norm: bool = False,
    eps: float = 1e-6,
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "torch", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, torch, triton, reference")
    if backend == "reference":
        return vision_patch_merger_reference(
            hidden_states,
            weights.norm_weight,
            weights.norm_bias,
            weights.fc1_weight,
            weights.fc1_bias,
            weights.fc2_weight,
            weights.fc2_bias,
            use_postshuffle_norm=use_postshuffle_norm,
            eps=eps,
        )
    norm_backend = "triton" if backend == "triton" and hidden_states.is_cuda else "auto"
    merged_hidden = weights.fc1_weight.shape[1]
    if use_postshuffle_norm:
        x = hidden_states.view(-1, merged_hidden)
        x = _layer_norm(x, weights.norm_weight, weights.norm_bias, eps=eps, backend=norm_backend)
    else:
        x = _layer_norm(hidden_states, weights.norm_weight, weights.norm_bias, eps=eps, backend=norm_backend)
        x = x.view(-1, merged_hidden)
    x = F.linear(x, weights.fc1_weight, weights.fc1_bias)
    x = F.gelu(x)
    return F.linear(x, weights.fc2_weight, weights.fc2_bias)


def vision_encoder_block(
    hidden_states: torch.Tensor,
    weights: VisionBlockWeights,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_heads: int,
    hidden_act: str = "gelu_pytorch_tanh",
    eps: float = 1e-6,
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "sdpa", "flash", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, sdpa, flash, triton, reference")
    if backend == "reference":
        return vision_encoder_block_reference(
            hidden_states,
            weights.norm1_weight,
            weights.norm1_bias,
            weights.attention.qkv_weight,
            weights.attention.qkv_bias,
            weights.attention.proj_weight,
            weights.attention.proj_bias,
            weights.norm2_weight,
            weights.norm2_bias,
            weights.mlp.fc1_weight,
            weights.mlp.fc1_bias,
            weights.mlp.fc2_weight,
            weights.mlp.fc2_bias,
            cu_seqlens,
            cos,
            sin,
            num_heads=num_heads,
            hidden_act=hidden_act,
            eps=eps,
        )

    norm_backend = "triton" if backend == "triton" and hidden_states.is_cuda else "auto"
    attention_backend = "flash" if backend == "flash" else ("sdpa" if backend == "sdpa" else "auto")
    normed = _layer_norm(hidden_states, weights.norm1_weight, weights.norm1_bias, eps=eps, backend=norm_backend)
    hidden_states = hidden_states + vision_attention_encode(
        normed,
        weights.attention,
        cu_seqlens,
        cos,
        sin,
        num_heads=num_heads,
        backend=attention_backend,
    )
    normed = _layer_norm(hidden_states, weights.norm2_weight, weights.norm2_bias, eps=eps, backend=norm_backend)
    hidden_states = hidden_states + vision_mlp_encode(normed, weights.mlp, hidden_act=hidden_act)
    return hidden_states


def vision_encode(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    weights: VisionEncoderWeights,
    *,
    config: Optional[Qwen36VisionConfig] = None,
    eps: float = 1e-6,
    backend: str = "auto",
) -> VisionEncoderResult:
    if backend not in {"auto", "sdpa", "flash", "triton", "reference"}:
        raise ValueError("backend must be one of: auto, sdpa, flash, triton, reference")
    cfg = config or qwen36_27b_vision_config()
    grid_thw = grid_thw.to(device=pixel_values.device)

    hidden_states = vision_patch_embed(pixel_values, weights.patch_embed, config=cfg, backend="reference" if backend == "reference" else "auto")
    pos_embeds = vision_fast_pos_embed_interpolate_reference(
        grid_thw,
        weights.pos_embed_weight,
        spatial_merge_size=cfg.spatial_merge_size,
        num_position_embeddings=cfg.num_position_embeddings,
    )
    hidden_states = hidden_states + pos_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
    sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)
    cu_seqlens = vision_make_cu_seqlens(grid_thw)

    deepstack_features = []
    for layer_idx, block_weights in enumerate(weights.blocks):
        hidden_states = vision_encoder_block(
            hidden_states,
            block_weights,
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            hidden_act=cfg.hidden_act,
            eps=eps,
            backend=backend,
        )
        if layer_idx in cfg.deepstack_visual_indexes:
            merger_idx = cfg.deepstack_visual_indexes.index(layer_idx)
            if merger_idx >= len(weights.deepstack_mergers):
                raise ValueError("missing deepstack merger weights")
            deepstack_features.append(
                vision_patch_merger(
                    hidden_states,
                    weights.deepstack_mergers[merger_idx],
                    use_postshuffle_norm=True,
                    eps=eps,
                    backend="reference" if backend == "reference" else ("triton" if backend == "triton" else "auto"),
                )
            )

    hidden_states = vision_patch_merger(
        hidden_states,
        weights.merger,
        use_postshuffle_norm=False,
        eps=eps,
        backend="reference" if backend == "reference" else ("triton" if backend == "triton" else "auto"),
    )
    return VisionEncoderResult(hidden=hidden_states, deepstack_features=tuple(deepstack_features))


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
