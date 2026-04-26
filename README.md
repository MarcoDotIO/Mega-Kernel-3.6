# Mega Kernel Qwen3.6

CUDA/Triton decode-path kernels plus optimized vision-encoder forward
primitives and PyTorch references for Qwen3.6 models, with
`Qwen/Qwen3.6-27B` as the primary dense/VL target and
`Qwen/Qwen3.6-35B-A3B` MoE support kept available.

The implementation targets single-GPU Qwen3.6 decode on H100 and RTX PRO 6000
Blackwell-class GPUs. It now has two layers:

- MPK-style planning: `TaskGraph`, SM-level task/event IR, graph normalization,
  event fusion, launch linearization, batch-specialized plans, and Mirage
  skeleton export.
- Kernel implementations: legacy per-stage CUDA kernels, faster PyTorch/Triton
  prototype backends, and a cooperative persistent MoE prototype whose internal
  stages are `__device__` tasks under one `__global__` mega-kernel entrypoint.

The public runtime APIs expose:

- `dense_ffn_decode`: RMSNorm, dense SwiGLU FFN, and residual for
  `Qwen/Qwen3.6-27B`. The default CUDA path uses tensor-core-friendly PyTorch
  matmuls; the original scalar CUDA prototype remains available with
  `backend="cuda"`.
- `moe_decode`: RMSNorm, router top-k, routed MoE, shared expert, and FFN residual.
  The default CUDA path uses a grouped/batched tensor-core prototype; the
  original scalar CUDA extension remains available with `backend="cuda"`, and
  the MPK-shaped single-launch prototype is available with
  `backend="persistent"`.
- `full_attention_decode`: one-token GQA attention over an existing KV cache.
- `linear_attention_decode`: FP32 recurrent-state linear attention primitive.
- `decode_layer`: small Python dispatcher for the Qwen3.6 layer pattern.
- `vision_patch_embed`, `vision_attention_encode`, `vision_mlp_encode`,
  `vision_patch_merger`, `vision_encoder_block`, and `vision_encode`:
  Qwen3.6-27B vision encoder forward primitives with SDPA attention and
  optional Triton LayerNorm.

## Viper Setup

Use local scratch/cache space on viper because the network home directory is
nearly full.

```bash
cd /tmp/mega-kernel-qwen36
python3 -m venv --system-site-packages /tmp/mega-kernel-venv
source /tmp/mega-kernel-venv/bin/activate
python -m pip install -U pip ninja pytest pybind11 triton transformers huggingface_hub safetensors

# H100 is sm_90; RTX PRO 6000 Blackwell is sm_120. CUDA 12.8+ is required for sm_120.
export TORCH_CUDA_ARCH_LIST="9.0;12.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export HF_HOME=/tmp/hf_home
export XDG_CACHE_HOME=/tmp/xdg_cache
export TMPDIR=/tmp/mega-kernel-build
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

python -m pip install -e . --no-build-isolation
python setup.py build_ext --inplace
pytest
```

## Quick Smoke

```bash
python - <<'PY'
import torch
import mega_kernel_qwen36 as mk

print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(mk.qwen36_27b_config().hidden_size)
PY
```

## Benchmarks

The benchmark defaults use smaller synthetic dimensions so they are fast during
kernel iteration. Pass `--official` for official Qwen3.6 dimensions.

```bash
python benchmarks/bench_moe.py
python benchmarks/bench_moe.py --official --backend auto --cuda-graph
python benchmarks/bench_moe.py --backend persistent
python benchmarks/bench_dense_ffn.py
python benchmarks/bench_dense_ffn.py --official --backend auto --cuda-graph
python benchmarks/bench_attention.py --seq-len 32768
python benchmarks/bench_attention.py --seq-len 32768 --cuda-graph
python benchmarks/bench_vision_encoder.py --mode block
python benchmarks/bench_vision_encoder.py --official --height 32 --width 32 --mode block
```

## MPK / Mirage Shape

The paper-backed direction is not to write every stage as its own `__global__`
kernel. The MPK-style path is:

1. Lower Qwen layers into a fine-grained task graph where tasks are SM-sized
   units and events encode task dependencies.
2. Normalize and fuse equivalent events to avoid unnecessary barriers.
3. Linearize the graph into AOT/JIT task launches for worker and scheduler SMs.
4. Execute task bodies inside a single persistent kernel, with task functions as
   `__device__` routines rather than standalone CUDA launches.
5. Optionally export a Mirage `PersistentKernel` skeleton for a future native
   MPK backend.

Inspect the current tGraph:

```bash
python scripts/inspect_mpk_plan.py --model 27b --layers 4
python scripts/inspect_mpk_plan.py --model 35b-a3b --layers 1
python scripts/inspect_mpk_plan.py --model vision --layers 2
python scripts/inspect_mpk_plan.py --model 27b --layers 1 --mirage-skeleton
```

## Runtime Parity

Parity tests are opt-in because vLLM and SGLang are large runtime dependencies
and not installed in the base viper environment.

```bash
# Recommended on viper: install runtime parity dependencies in an isolated venv.
python3 -m venv /tmp/mega-kernel-parity-venv
source /tmp/mega-kernel-parity-venv/bin/activate
python -m pip install -U pip ninja pytest pybind11 triton transformers huggingface_hub safetensors
python -m pip install vllm==0.19.1
python -m pip install --no-deps sglang==0.5.10.post1

cd /tmp/mega-kernel-qwen36-parity
export TORCH_CUDA_ARCH_LIST="9.0;12.0"
python -m pip install -e . --no-build-isolation
python setup.py build_ext --inplace
pytest tests/test_runtime_parity.py --run-runtime-parity
pytest tests/test_transformers_parity.py --run-transformers-parity
```

These tests compare shared numerical primitives against vLLM/SGLang when their
public or importable kernels are available, and skip with an explicit reason
when a runtime does not expose a compatible kernel on the installed version.
Keep the parity checkout separate from the core checkout because the extension
is ABI-specific to the active PyTorch version.
In the current viper parity environment, `sgl-kernel 0.3.21` installs but its
native `common_ops` library fails to load against the selected Torch ABI, so
SGLang native FlashAttention parity is skipped while SGLang Python/JIT RMSNorm
parity remains testable.
The Transformers parity tests instantiate the official Qwen3.6-27B vision
architecture, copy live Hugging Face module weights into the local dataclasses,
and compare block/model forward outputs without downloading the full 27B
checkpoint.

## Verified H100 Baseline

Measured on `marnett5@viper.cs.kent.edu` with an NVIDIA H100 NVL, CUDA 12.8,
PyTorch 2.7.0, BF16 inputs, and `TORCH_CUDA_ARCH_LIST="9.0;12.0"`.

- Tests: `14 passed`.
- Small MoE synthetic: `0.1650 ms`, `2.96x` over the PyTorch reference.
- Official MoE synthetic, batch 1: `1.2492 ms`, `0.64x`; this path is
  correctness-first and still needs tensor-core or grouped-GEMM work.
- Official MoE synthetic, batch 2: `1.2736 ms`, `1.15x`.
- Official MoE synthetic, batch 4: `1.3224 ms`, `1.92x`.
- Full attention decode, 32K context: `0.7408 ms`, `2.53x`.

## Verified Qwen3.6-27B / Runtime Parity Update

Measured on the same H100 after adding `Qwen/Qwen3.6-27B`, `sm_120`, Triton
RMSNorm/LayerNorm, grouped decode backends, vision encoder primitives, CUDA
Graph benchmark replay, and runtime parity hooks.

- Core tests: `59 passed, 6 skipped`.
- Transformers architecture parity: `2 passed`.
- Runtime parity tests with `vllm 0.19.1` and `sglang 0.5.10.post1`:
  `3 passed, 1 skipped`.
- Passed parity: vLLM RMSNorm, SGLang RMSNorm, and vLLM Triton decode attention.
- Skipped parity: SGLang FlashAttention, because `flash_attn.cute` is not
  available in the installed runtime stack.
- Qwen3.6-35B-A3B official MoE synthetic, batch 1, default grouped backend:
  `0.3546 ms`, `2.31x`; with CUDA Graph replay: `0.2511 ms`, `3.16x`.
- Qwen3.6-35B-A3B official MoE synthetic, batch 1, old scalar CUDA backend:
  `1.2491 ms`, `0.64x`.
- Qwen3.6-35B-A3B official MoE synthetic, batch 1, single-launch persistent
  prototype, before cooperative worker tiling: `18.2959 ms`, `0.04x`.
- Qwen3.6-35B-A3B official MoE synthetic, cooperative persistent MoE:
  batch 1 `1.5922 ms`, `0.51x`; batch 2 `1.9240 ms`, `0.75x`;
  batch 4 `2.2278 ms`, `1.18x`; batch 8 `2.7972 ms`, `1.82x`.
  This is still scalar-FMA tile work rather than native tensor-core WMMA, but
  the MPK-style single launch now distributes router, routed expert, shared
  expert, and down-projection work across cooperative CTA workers.
- Qwen3.6-27B official dense FFN synthetic, batch 1, default torch backend:
  `0.2708 ms`, `5.37x`; with CUDA Graph replay: `0.2133 ms`, `6.84x`.
- Qwen3.6-27B official dense FFN synthetic, batch 1, old scalar CUDA backend:
  `2.8912 ms`, `0.51x`.
- Full attention decode, 32K context: `0.7405 ms`, `2.53x`; with CUDA Graph
  replay: `0.7356 ms`, `2.53x`.
- Qwen3.6-27B official vision attention, 32x32 grid: `0.2094 ms`, `3.81x`.
- Qwen3.6-27B official vision block, 32x32 grid: `0.2948 ms`, `5.60x`.
- Qwen3.6-27B official 27-layer vision encoder wrapper, 32x32 grid:
  `9.0645 ms`, `5.12x`.
- Triton LayerNorm vision block path is numerically correct but measured slower
  than the PyTorch LayerNorm path on H100 for the 1152-wide official block:
  `0.3376 ms`, `4.89x`.
