import math

import pytest
import torch

import mega_kernel_qwen36 as mk
from mega_kernel_qwen36.reference import rms_norm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _require_runtime_parity(request):
    if not request.config.getoption("--run-runtime-parity"):
        pytest.skip("pass --run-runtime-parity to run vLLM/SGLang parity tests")


def _assert_close(actual, expected, atol=3e-2, rtol=5e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def test_vllm_rms_norm_matches_reference(request):
    _require_runtime_parity(request)
    ops = pytest.importorskip("vllm._custom_ops")
    torch.manual_seed(101)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = (torch.randn(512, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
    out = torch.empty_like(x)
    ops.rms_norm(out, x.contiguous(), weight, 1e-6)
    _assert_close(out, rms_norm(x, weight, 1e-6))


def test_sglang_rms_norm_matches_reference(request):
    _require_runtime_parity(request)
    norm = pytest.importorskip("sglang.jit_kernel.norm")
    torch.manual_seed(103)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = (torch.randn(512, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
    out = torch.empty_like(x)
    norm.rmsnorm(x.contiguous(), weight, out=out, eps=1e-6)
    _assert_close(out, rms_norm(x, weight, 1e-6))


def test_vllm_triton_decode_attention_matches_mega_kernel(request):
    _require_runtime_parity(request)
    tda = pytest.importorskip("vllm.v1.attention.ops.triton_decode_attention")
    torch.manual_seed(107)
    batch, seq_len, query_heads, kv_heads, head_dim = 2, 257, 4, 2, 32
    q = (torch.randn(batch, query_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    k_cache = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    v_cache = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    expected = mk.full_attention_decode(mk.FullAttentionInputs(q=q, k_cache=k_cache, v_cache=v_cache))

    k_buffer = k_cache.reshape(batch * seq_len, kv_heads, head_dim).contiguous()
    v_buffer = v_cache.reshape(batch * seq_len, kv_heads, head_dim).contiguous()
    req_to_token = torch.arange(batch * seq_len, device="cuda", dtype=torch.int32).reshape(batch, seq_len)
    b_seq_len = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    num_kv_splits = 4
    attn_logits = torch.empty((batch, query_heads, num_kv_splits, head_dim + 1), device="cuda", dtype=torch.float32)
    out = torch.empty_like(q)
    lse = torch.empty((batch, query_heads), device="cuda", dtype=torch.float32)
    tda.decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        out,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        1.0 / math.sqrt(head_dim),
        page_size=1,
    )
    _assert_close(out, expected)


def test_sglang_flash_attention_matches_mega_kernel_if_native_kernel_loads(request):
    _require_runtime_parity(request)
    try:
        from sglang.jit_kernel.flash_attention_v4 import flash_attn_with_kvcache
    except Exception as exc:
        pytest.skip(f"SGLang FlashAttention import unavailable: {exc}")

    torch.manual_seed(109)
    batch, seq_len, query_heads, kv_heads, head_dim = 2, 128, 4, 2, 32
    q = (torch.randn(batch, query_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    k_cache = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    v_cache = (torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda") * 0.1).bfloat16().contiguous()
    expected = mk.full_attention_decode(mk.FullAttentionInputs(q=q, k_cache=k_cache, v_cache=v_cache))
    try:
        out = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=torch.full((batch,), seq_len, device="cuda", dtype=torch.int32),
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
        ).squeeze(1)
    except Exception as exc:
        pytest.skip(f"SGLang FlashAttention runtime unavailable: {exc}")
    _assert_close(out, expected)
