import pytest
import torch

import mega_kernel_qwen36 as mk
from tests.test_reference import _small_dense, _small_moe, _small_vision


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not mk.extension_available(),
    reason="CUDA extension is not available",
)


def _assert_close_bf16(actual, expected, atol=3e-2, rtol=5e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def test_cuda_capability_smoke():
    major, minor = torch.cuda.get_device_capability()
    assert (major, minor) in {(9, 0), (12, 0)}


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


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_dense_ffn_decode_cuda_matches_reference(batch):
    x, weights = _small_dense(device="cuda", dtype=torch.bfloat16)
    x = x[:1].repeat(batch, 1).contiguous()
    out = mk.dense_ffn_decode(x, weights)
    ref = mk.dense_ffn_decode(
        x.cpu().float(),
        mk.DenseFfnWeights(*(t.cpu().float() for t in weights.__dict__.values())),
    )
    _assert_close_bf16(out.cpu(), ref)


def test_dense_ffn_decode_triton_matches_reference():
    pytest.importorskip("triton")
    x, weights = _small_dense(device="cuda", dtype=torch.bfloat16)
    out = mk.dense_ffn_decode(x, weights, backend="triton")
    ref = mk.dense_ffn_decode(
        x.cpu().float(),
        mk.DenseFfnWeights(*(t.cpu().float() for t in weights.__dict__.values())),
    )
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


def test_vision_attention_sdpa_cuda_matches_reference():
    cfg, grid_thw, pixel_values, weights = _small_vision(device="cuda", dtype=torch.bfloat16)
    hidden = mk.vision_patch_embed(pixel_values, weights.patch_embed, config=cfg)
    from mega_kernel_qwen36.reference import vision_make_cu_seqlens, vision_rotary_position_embeddings_reference

    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cos = cos.to(dtype=hidden.dtype)
    sin = sin.to(dtype=hidden.dtype)
    cu = vision_make_cu_seqlens(grid_thw)
    out = mk.vision_attention_encode(hidden, weights.blocks[0].attention, cu, cos, sin, num_heads=cfg.num_heads)
    ref = mk.vision_attention_encode(
        hidden,
        weights.blocks[0].attention,
        cu,
        cos,
        sin,
        num_heads=cfg.num_heads,
        backend="reference",
    )
    _assert_close_bf16(out, ref)


def test_vision_patch_merger_triton_cuda_matches_reference():
    pytest.importorskip("triton")
    cfg, _, _, weights = _small_vision(device="cuda", dtype=torch.bfloat16)
    hidden = (torch.randn(16, cfg.hidden_size, device="cuda") * 0.04).bfloat16()
    out = mk.vision_patch_merger(hidden, weights.merger, backend="triton")
    ref = mk.vision_patch_merger(hidden, weights.merger, backend="reference")
    _assert_close_bf16(out, ref)


def test_vision_encode_cuda_matches_reference():
    cfg, grid_thw, pixel_values, weights = _small_vision(device="cuda", dtype=torch.bfloat16)
    out = mk.vision_encode(pixel_values, grid_thw, weights, config=cfg)
    ref = mk.vision_encode(pixel_values, grid_thw, weights, config=cfg, backend="reference")
    assert out.hidden.shape == (4, cfg.out_hidden_size)
    _assert_close_bf16(out.hidden, ref.hidden)
