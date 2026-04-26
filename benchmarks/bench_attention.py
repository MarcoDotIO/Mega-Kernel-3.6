from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mega_kernel_qwen36 as mk
from bench_utils import cuda_time
from mega_kernel_qwen36.reference import full_attention_decode_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    q = (torch.randn(args.batch, args.query_heads, args.head_dim, device="cuda") * args.scale).to(dtype)
    k = (torch.randn(args.batch, args.seq_len, args.kv_heads, args.head_dim, device="cuda") * args.scale).to(dtype)
    v = (torch.randn(args.batch, args.seq_len, args.kv_heads, args.head_dim, device="cuda") * args.scale).to(dtype)
    inputs = mk.FullAttentionInputs(q=q, k_cache=k, v_cache=v)
    ext_result = cuda_time(lambda: mk.full_attention_decode(inputs), iters=args.iters, tokens=args.batch)
    ref_result = cuda_time(
        lambda: full_attention_decode_reference(q, k, v),
        warmup=max(2, args.iters // 10),
        iters=max(5, args.iters // 2),
        tokens=args.batch,
    )
    print(f"extension_available={mk.extension_available()}")
    print(f"extension_median_ms={ext_result.median_ms:.4f}")
    print(f"extension_p95_ms={ext_result.p95_ms:.4f}")
    print(f"reference_median_ms={ref_result.median_ms:.4f}")
    print(f"speedup={ref_result.median_ms / ext_result.median_ms:.2f}x")
    print(f"tokens_per_sec={ext_result.tokens_per_sec:.2f}")
    print(f"max_memory_mb={ext_result.max_memory_mb:.2f}")


if __name__ == "__main__":
    main()
