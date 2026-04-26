from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mega_kernel_qwen36 as mk
from bench_utils import cuda_time
from mega_kernel_qwen36.reference import dense_ffn_decode_reference


def make_inputs(args):
    cfg = mk.qwen36_27b_config()
    hidden = cfg.hidden_size if args.official else args.hidden
    intermediate = cfg.intermediate_size if args.official else args.intermediate
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    device = "cuda"
    scale = args.scale
    x = (torch.randn(args.batch, hidden, device=device) * scale).to(dtype)
    weights = mk.DenseFfnWeights(
        norm_weight=(torch.randn(hidden, device=device) * scale + 1.0).to(dtype),
        gate_weight=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        up_weight=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        down_weight=(torch.randn(hidden, intermediate, device=device) * scale).to(dtype),
    )
    return x, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--backend", choices=["auto", "torch", "cuda", "triton", "reference"], default="auto")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    x, weights = make_inputs(args)
    ext_result = cuda_time(
        lambda: mk.dense_ffn_decode(x, weights, backend=args.backend),
        iters=args.iters,
        tokens=args.batch,
        cuda_graph=args.cuda_graph,
    )
    ref_result = cuda_time(
        lambda: dense_ffn_decode_reference(
            x,
            weights.norm_weight,
            weights.gate_weight,
            weights.up_weight,
            weights.down_weight,
        ),
        warmup=max(2, args.iters // 10),
        iters=max(5, args.iters // 2),
        tokens=args.batch,
    )
    print(f"extension_available={mk.extension_available()}")
    print(f"backend={args.backend}")
    print(f"cuda_graph={args.cuda_graph}")
    print(f"optimized_median_ms={ext_result.median_ms:.4f}")
    print(f"optimized_p95_ms={ext_result.p95_ms:.4f}")
    print(f"reference_median_ms={ref_result.median_ms:.4f}")
    print(f"speedup={ref_result.median_ms / ext_result.median_ms:.2f}x")
    print(f"tokens_per_sec={ext_result.tokens_per_sec:.2f}")
    print(f"max_memory_mb={ext_result.max_memory_mb:.2f}")


if __name__ == "__main__":
    main()
