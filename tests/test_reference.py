import torch

import mega_kernel_qwen36 as mk
from mega_kernel_qwen36.reference import (
    full_attention_decode_reference,
    linear_attention_decode_reference,
    moe_decode_reference,
)


def _small_moe(device="cpu", dtype=torch.float32):
    torch.manual_seed(7)
    batch, hidden, experts, intermediate, top_k = 2, 16, 6, 12, 3
    scale = 0.08
    x = (torch.randn(batch, hidden, device=device) * scale).to(dtype)
    weights = mk.MoeWeights(
        norm_weight=(torch.randn(hidden, device=device) * scale + 1.0).to(dtype),
        router_weight=(torch.randn(experts, hidden, device=device) * scale).to(dtype),
        expert_gate=(torch.randn(experts, intermediate, hidden, device=device) * scale).to(dtype),
        expert_up=(torch.randn(experts, intermediate, hidden, device=device) * scale).to(dtype),
        expert_down=(torch.randn(experts, hidden, intermediate, device=device) * scale).to(dtype),
        shared_gate=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        shared_up=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        shared_down=(torch.randn(hidden, intermediate, device=device) * scale).to(dtype),
    )
    return x, weights, top_k


def test_moe_reference_shapes_and_topk():
    x, weights, top_k = _small_moe()
    out, idx, router_weights = moe_decode_reference(
        x,
        weights.norm_weight,
        weights.router_weight,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
        weights.shared_gate,
        weights.shared_up,
        weights.shared_down,
        top_k=top_k,
    )
    assert out.shape == x.shape
    assert idx.shape == (x.shape[0], top_k)
    assert router_weights.shape == (x.shape[0], top_k)
    torch.testing.assert_close(router_weights.sum(dim=-1), torch.ones(x.shape[0]))


def test_attention_reference_shapes():
    torch.manual_seed(11)
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 9, 2, 16)
    v = torch.randn(2, 9, 2, 16)
    out = full_attention_decode_reference(q, k, v)
    assert out.shape == q.shape


def test_linear_attention_reference_shapes():
    torch.manual_seed(13)
    q = torch.randn(2, 3, 5)
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 7)
    state = torch.randn(2, 3, 5, 7)
    out, new_state = linear_attention_decode_reference(q, k, v, state, decay=0.9)
    assert out.shape == v.shape
    assert new_state.shape == state.shape


def test_decode_layer_dispatch_reference_linear():
    x, weights, top_k = _small_moe()
    q = torch.randn(2, 4, 4)
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    state = torch.zeros(2, 4, 4, 4)
    result = mk.decode_layer(
        0,
        x,
        weights,
        linear_attention=mk.LinearAttentionInputs(q=q, k=k, v=v, state=state),
        top_k=top_k,
    )
    assert result.hidden.shape == x.shape
    assert result.linear_state is not None
