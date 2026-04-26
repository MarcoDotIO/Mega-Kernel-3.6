# Mega Kernel Qwen3.6

CUDA decode-path kernels and PyTorch references for the text path of
`Qwen/Qwen3.6-35B-A3B`.

The first implementation targets a single H100 and exposes:

- `moe_decode`: RMSNorm, router top-k, routed MoE, shared expert, and FFN residual.
- `full_attention_decode`: one-token GQA attention over an existing KV cache.
- `linear_attention_decode`: FP32 recurrent-state linear attention primitive.
- `decode_layer`: small Python dispatcher for the Qwen3.6 layer pattern.

## Viper Setup

Use local scratch/cache space on viper because the network home directory is
nearly full.

```bash
cd /tmp/mega-kernel-qwen36
python3 -m venv --system-site-packages /tmp/mega-kernel-venv
source /tmp/mega-kernel-venv/bin/activate
python -m pip install -U pip ninja pytest pybind11 transformers huggingface_hub safetensors

export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export HF_HOME=/tmp/hf_home
export XDG_CACHE_HOME=/tmp/xdg_cache
export TMPDIR=/tmp/mega-kernel-build
mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

python -m pip install -e . --no-build-isolation
pytest
```

## Quick Smoke

```bash
python - <<'PY'
import torch
import mega_kernel_qwen36 as mk

print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(mk.qwen36_35b_a3b_config().hidden_size)
PY
```

## Benchmarks

The benchmark defaults use smaller synthetic dimensions so they are fast during
kernel iteration. Pass `--official` for Qwen3.6-35B-A3B dimensions.

```bash
python benchmarks/bench_moe.py
python benchmarks/bench_attention.py --seq-len 32768
```

## Verified H100 Baseline

Measured on `marnett5@viper.cs.kent.edu` with an NVIDIA H100 NVL, CUDA 12.8,
PyTorch 2.7.0, BF16 inputs, and `TORCH_CUDA_ARCH_LIST=9.0`.

- Tests: `14 passed`.
- Small MoE synthetic: `0.1650 ms`, `2.96x` over the PyTorch reference.
- Official MoE synthetic, batch 1: `1.2492 ms`, `0.64x`; this path is
  correctness-first and still needs tensor-core or grouped-GEMM work.
- Official MoE synthetic, batch 2: `1.2736 ms`, `1.15x`.
- Official MoE synthetic, batch 4: `1.3224 ms`, `1.92x`.
- Full attention decode, 32K context: `0.7408 ms`, `2.53x`.
