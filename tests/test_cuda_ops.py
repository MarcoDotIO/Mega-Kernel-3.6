import pytest
import torch

import mega_kernel_qwen36 as mk
from tests.test_reference import _small_moe


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mk.extension_available(),
    reason="CUDA extension is not available",
)


def _assert_close_bf16(actual, expected, atol=3e-2, rtol=5e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def test_h100_capability_smoke():
    major, minor = torch.cuda.get_device_capability()
    assert (major, minor) == (9, 0)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_moe_decode_cuda_matches_reference(batch):
    x, weights, top_k = _small_moe(device="cuda", dtype=torch.bfloat16)
    x = x[:1].repeat(batch, 1).contiguous()
    out, idx, router_weights = mk.moe_decode(x, weights, top_k=top_k)
    ref, ref_idx, ref_router_weights = mk.moe_decode(
        x.cpu().float(),
        mk.MoeWeights(*(t.cpu().float() for t in weights.__dict__.values())),
        top_k=top_k,
    )
    assert idx.cpu().tolist() == ref_idx.tolist()
    _assert_close_bf16(router_weights.cpu(), ref_router_weights)
    _assert_close_bf16(out.cpu(), ref)


@pytest.mark.parametrize("seq_len", [1, 128, 4096])
def test_full_attention_decode_cuda_matches_reference(seq_len):
    torch.manual_seed(17)
    batch, query_heads, kv_heads, head_dim = 2, 4, 2, 32
    q = (torch.randn(batch, query_heads, head_dim, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16()
    out = mk.full_attention_decode(mk.FullAttentionInputs(q=q, k_cache=k, v_cache=v))
    ref = mk.full_attention_decode(
        mk.FullAttentionInputs(q=q.cpu().float(), k_cache=k.cpu().float(), v_cache=v.cpu().float())
    )
    _assert_close_bf16(out.cpu(), ref)


def test_linear_attention_decode_cuda_matches_reference():
    torch.manual_seed(19)
    batch, heads, key_dim, value_dim = 2, 3, 8, 10
    q = (torch.randn(batch, heads, key_dim, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(batch, heads, key_dim, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(batch, heads, value_dim, device="cuda") * 0.1).bfloat16()
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda") * 0.1
    out, new_state = mk.linear_attention_decode(mk.LinearAttentionInputs(q=q, k=k, v=v, state=state, decay=0.92))
    ref_out, ref_state = mk.linear_attention_decode(
        mk.LinearAttentionInputs(
            q=q.cpu().float(),
            k=k.cpu().float(),
            v=v.cpu().float(),
            state=state.cpu(),
            decay=0.92,
        )
    )
    _assert_close_bf16(out.cpu(), ref_out)
    torch.testing.assert_close(new_state.cpu(), ref_state, atol=3e-2, rtol=5e-2)
