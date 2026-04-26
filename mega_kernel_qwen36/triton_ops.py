from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional Linux GPU stack.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(x, weight, out, hidden: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / hidden
        normalized = values * tl.rsqrt(variance + eps) * weights
        tl.store(out + row * hidden + offsets, normalized, mask=mask)

    @triton.jit
    def _layer_norm_kernel(
        x,
        weight,
        bias,
        out,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(values, axis=0) / hidden
        centered = tl.where(mask, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = centered * tl.rsqrt(variance + eps) * weights
        if has_bias:
            normalized += tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + row * hidden + offsets, normalized, mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise ValueError("rms_norm_triton requires a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be [batch, hidden]")
    hidden = x.shape[-1]
    block = triton.next_power_of_2(hidden)
    out = torch.empty_like(x)
    _rms_norm_kernel[(x.shape[0],)](x, weight, out, hidden, float(eps), block, num_warps=8)
    return out


def layer_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise ValueError("layer_norm_triton requires a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be [rows, hidden]")
    hidden = x.shape[-1]
    block = triton.next_power_of_2(hidden)
    out = torch.empty_like(x)
    _layer_norm_kernel[(x.shape[0],)](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        hidden,
        float(eps),
        block,
        bias is not None,
        num_warps=8,
    )
    return out


def dense_ffn_decode_triton(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    add_residual: bool = True,
) -> torch.Tensor:
    x_norm = rms_norm_triton(x, norm_weight, eps).float()
    act = torch.nn.functional.silu(x_norm @ gate_weight.float().t()) * (x_norm @ up_weight.float().t())
    out = act @ down_weight.float().t()
    if add_residual:
        out += x.float()
    return out.to(x.dtype)
