from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mega_kernel_qwen36 as mk
from bench_utils import cuda_time
from mega_kernel_qwen36.reference import moe_decode_reference


def make_inputs(args):
    cfg = mk.qwen36_35b_a3b_config()
    if args.official:
        hidden = cfg.hidden_size
        experts = cfg.num_experts
        intermediate = cfg.moe_intermediate_size
        top_k = cfg.num_experts_per_tok
    else:
        hidden = args.hidden
        experts = args.experts
        intermediate = args.intermediate
        top_k = args.top_k

    torch.manual_seed(args.seed)
    scale = args.scale
    dtype = torch.bfloat16
    device = "cuda"
    x = (torch.randn(args.batch, hidden, device=device) * scale).to(dtype)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--backend", choices=["auto", "grouped", "persistent", "reference"], default="auto")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    x, weights, top_k = make_inputs(args)
    ext_result = cuda_time(
        lambda: mk.moe_decode(x, weights, top_k=top_k, backend=args.backend),
        iters=args.iters,
        tokens=args.batch,
        cuda_graph=args.cuda_graph,
    )
    ref_result = cuda_time(
        lambda: moe_decode_reference(
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
