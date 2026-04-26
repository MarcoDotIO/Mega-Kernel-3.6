from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import replace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mega_kernel_qwen36 as mk
from bench_utils import cuda_time
from mega_kernel_qwen36.reference import (
    vision_fast_pos_embed_interpolate_reference,
    vision_make_cu_seqlens,
    vision_rotary_position_embeddings_reference,
)


def _weight(shape, *, device, dtype, scale):
    return (torch.randn(*shape, device=device) * scale).to(dtype)


def _bias(size, *, device, dtype, scale):
    return (torch.randn(size, device=device) * scale).to(dtype)


def make_inputs(args):
    dtype = torch.bfloat16
    device = "cuda"
    if args.official:
        cfg = replace(mk.qwen36_27b_vision_config(), depth=args.layers)
    else:
        cfg = mk.Qwen36VisionConfig(
            depth=args.layers,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            num_heads=args.heads,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            out_hidden_size=args.out_hidden,
            num_position_embeddings=args.position_embeddings,
        )

    if args.height % cfg.spatial_merge_size or args.width % cfg.spatial_merge_size:
        raise ValueError("height and width must be divisible by spatial_merge_size")
    if cfg.hidden_size % cfg.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")

    torch.manual_seed(args.seed)
    scale = args.scale
    grid_thw = torch.tensor([[args.frames, args.height, args.width]], device=device, dtype=torch.long)
    seq_len = int(grid_thw.prod(dim=1).sum().item())
    pixel_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = _weight((seq_len, pixel_dim), device=device, dtype=dtype, scale=scale)

    def merger(postshuffle_norm=False):
        norm_hidden = cfg.merged_hidden_size if postshuffle_norm else cfg.hidden_size
        return mk.VisionPatchMergerWeights(
            norm_weight=(_weight((norm_hidden,), device=device, dtype=dtype, scale=scale) + 1.0).to(dtype),
            norm_bias=_bias(norm_hidden, device=device, dtype=dtype, scale=scale),
            fc1_weight=_weight((cfg.merged_hidden_size, cfg.merged_hidden_size), device=device, dtype=dtype, scale=scale),
            fc1_bias=_bias(cfg.merged_hidden_size, device=device, dtype=dtype, scale=scale),
            fc2_weight=_weight((cfg.out_hidden_size, cfg.merged_hidden_size), device=device, dtype=dtype, scale=scale),
            fc2_bias=_bias(cfg.out_hidden_size, device=device, dtype=dtype, scale=scale),
        )

    blocks = []
    for _ in range(cfg.depth):
        blocks.append(
            mk.VisionBlockWeights(
                norm1_weight=(_weight((cfg.hidden_size,), device=device, dtype=dtype, scale=scale) + 1.0).to(dtype),
                norm1_bias=_bias(cfg.hidden_size, device=device, dtype=dtype, scale=scale),
                attention=mk.VisionAttentionWeights(
                    qkv_weight=_weight((cfg.hidden_size * 3, cfg.hidden_size), device=device, dtype=dtype, scale=scale),
                    qkv_bias=_bias(cfg.hidden_size * 3, device=device, dtype=dtype, scale=scale),
                    proj_weight=_weight((cfg.hidden_size, cfg.hidden_size), device=device, dtype=dtype, scale=scale),
                    proj_bias=_bias(cfg.hidden_size, device=device, dtype=dtype, scale=scale),
                ),
                norm2_weight=(_weight((cfg.hidden_size,), device=device, dtype=dtype, scale=scale) + 1.0).to(dtype),
                norm2_bias=_bias(cfg.hidden_size, device=device, dtype=dtype, scale=scale),
                mlp=mk.VisionMlpWeights(
                    fc1_weight=_weight((cfg.intermediate_size, cfg.hidden_size), device=device, dtype=dtype, scale=scale),
                    fc1_bias=_bias(cfg.intermediate_size, device=device, dtype=dtype, scale=scale),
                    fc2_weight=_weight((cfg.hidden_size, cfg.intermediate_size), device=device, dtype=dtype, scale=scale),
                    fc2_bias=_bias(cfg.hidden_size, device=device, dtype=dtype, scale=scale),
                ),
            )
        )

    weights = mk.VisionEncoderWeights(
        patch_embed=mk.VisionPatchEmbedWeights(
            proj_weight=_weight(
                (cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
                device=device,
                dtype=dtype,
                scale=scale,
            ),
            proj_bias=_bias(cfg.hidden_size, device=device, dtype=dtype, scale=scale),
        ),
        pos_embed_weight=_weight((cfg.num_position_embeddings, cfg.hidden_size), device=device, dtype=dtype, scale=scale),
        blocks=tuple(blocks),
        merger=merger(),
    )
    return cfg, grid_thw, pixel_values, weights


def make_block_state(cfg, grid_thw, pixel_values, weights):
    hidden = mk.vision_patch_embed(pixel_values, weights.patch_embed, config=cfg)
    hidden = hidden + vision_fast_pos_embed_interpolate_reference(
        grid_thw,
        weights.pos_embed_weight,
        spatial_merge_size=cfg.spatial_merge_size,
        num_position_embeddings=cfg.num_position_embeddings,
    ).to(dtype=hidden.dtype)
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cos = cos.to(dtype=hidden.dtype)
    sin = sin.to(dtype=hidden.dtype)
    return hidden, vision_make_cu_seqlens(grid_thw), cos, sin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--mode", choices=["attention", "block", "encoder"], default="block")
    parser.add_argument("--backend", choices=["auto", "sdpa", "triton"], default="auto")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--out-hidden", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument("--position-embeddings", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    cfg, grid_thw, pixel_values, weights = make_inputs(args)
    seq_len = int(grid_thw.prod(dim=1).sum().item())
    hidden, cu_seqlens, cos, sin = make_block_state(cfg, grid_thw, pixel_values, weights)

    if args.mode == "attention":
        opt_fn = lambda: mk.vision_attention_encode(
            hidden,
            weights.blocks[0].attention,
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            backend=args.backend,
        )
        ref_fn = lambda: mk.vision_attention_encode(
            hidden,
            weights.blocks[0].attention,
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            backend="reference",
        )
        tokens = seq_len
    elif args.mode == "block":
        opt_fn = lambda: mk.vision_encoder_block(
            hidden,
            weights.blocks[0],
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            hidden_act=cfg.hidden_act,
            backend=args.backend,
        )
        ref_fn = lambda: mk.vision_encoder_block(
            hidden,
            weights.blocks[0],
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            hidden_act=cfg.hidden_act,
            backend="reference",
        )
        tokens = seq_len
    else:
        opt_fn = lambda: mk.vision_encode(pixel_values, grid_thw, weights, config=cfg, backend=args.backend).hidden
        ref_fn = lambda: mk.vision_encode(pixel_values, grid_thw, weights, config=cfg, backend="reference").hidden
        tokens = seq_len // (cfg.spatial_merge_size**2)

    opt_result = cuda_time(opt_fn, iters=args.iters, tokens=tokens)
    ref_result = cuda_time(ref_fn, warmup=max(2, args.iters // 10), iters=max(5, args.iters // 2), tokens=tokens)
    print(f"mode={args.mode}")
    print(f"backend={args.backend}")
    print(f"official={args.official}")
    print(f"layers={cfg.depth}")
    print(f"seq_len={seq_len}")
    print(f"hidden={cfg.hidden_size}")
    print(f"heads={cfg.num_heads}")
    print(f"optimized_median_ms={opt_result.median_ms:.4f}")
    print(f"optimized_p95_ms={opt_result.p95_ms:.4f}")
    print(f"reference_median_ms={ref_result.median_ms:.4f}")
    print(f"speedup={ref_result.median_ms / opt_result.median_ms:.2f}x")
    print(f"tokens_per_sec={opt_result.tokens_per_sec:.2f}")
    print(f"max_memory_mb={opt_result.max_memory_mb:.2f}")


if __name__ == "__main__":
    main()
