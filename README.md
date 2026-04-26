# Mega Kernel Qwen3.6

CUDA/Triton decode kernels, a cooperative persistent MoE path, Qwen3.6-27B
vision encoder primitives, and PyTorch references for correctness testing.

The repo targets single-GPU H100 and RTX PRO 6000 class hardware with
`TORCH_CUDA_ARCH_LIST="9.0;12.0"`.

## Runtime APIs

- `moe_decode`: Qwen3.6-35B-A3B MoE decode. `backend="auto"` uses grouped
  PyTorch matmuls; `backend="persistent"` uses the cooperative single-launch
  MoE kernel.
- `dense_ffn_decode`: Qwen3.6-27B dense SwiGLU decode. `backend="auto"` uses
  tensor-core-friendly PyTorch matmuls; `backend="triton"` uses Triton
  normalization plus PyTorch matmuls.
- `full_attention_decode`: one-token GQA attention over an existing KV cache.
- `linear_attention_decode`: recurrent linear attention with FP32 state.
- `decode_layer`: small dispatcher for Qwen3.6 layer wiring.
- `vision_encode` and component helpers: Qwen3.6-27B vision forward path.
- `build_qwen36_decode_tgraph` / `build_qwen36_vision_tgraph`: MPK-style task
  graph planning for decode and vision paths.

## Viper Setup

Use scratch/cache paths on viper.

```bash
cd /tmp/mega-kernel-qwen36
python3 -m venv --system-site-packages /tmp/mega-kernel-venv
source /tmp/mega-kernel-venv/bin/activate
python -m pip install -U pip ninja pytest pybind11 triton transformers huggingface_hub safetensors

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

## Benchmarks

Defaults are small synthetic shapes. Pass `--official` for Qwen3.6 dimensions.

```bash
python benchmarks/bench_moe.py --official --backend auto --cuda-graph
python benchmarks/bench_moe.py --official --backend persistent
python benchmarks/bench_dense_ffn.py --official --backend auto --cuda-graph
python benchmarks/bench_attention.py --seq-len 32768 --cuda-graph
python benchmarks/bench_vision_encoder.py --official --height 32 --width 32 --mode block
python scripts/inspect_mpk_plan.py --model 27b --layers 4
python scripts/inspect_mpk_plan.py --model 35b-a3b --layers 1
python scripts/inspect_mpk_plan.py --model vision --layers 2
```

## Runtime Parity

Parity tests are opt-in because vLLM and SGLang are large dependencies.

```bash
python -m pip install vllm==0.19.1
python -m pip install --no-deps sglang==0.5.10.post1
pytest tests/test_runtime_parity.py --run-runtime-parity
pytest tests/test_transformers_parity.py --run-transformers-parity
```

The runtime tests compare shared primitives against vLLM/SGLang when compatible
kernels are importable. The Transformers tests instantiate Qwen3.6-27B vision
modules, copy their live weights into local dataclasses, and compare outputs.

## Verified H100 Baseline

Measured on `marnett5@viper.cs.kent.edu` with an NVIDIA H100 NVL, CUDA 12.8,
PyTorch 2.7.0, BF16 inputs, and `TORCH_CUDA_ARCH_LIST="9.0;12.0"`.

- Core tests: `50 passed, 6 skipped`.
- Transformers architecture parity: `2 passed`.
- Qwen3.6-35B-A3B MoE grouped backend, batch 1: `0.3646 ms`; with CUDA Graph:
  `0.2511 ms`.
- Qwen3.6-35B-A3B cooperative persistent MoE: batch 1 `1.5884 ms`, batch 2
  `1.9240 ms`, batch 4 `2.2278 ms`, batch 8 `2.7972 ms`.
- Qwen3.6-27B dense FFN, batch 1: `0.2707 ms`; with CUDA Graph: `0.2133 ms`.
- Full attention decode, 32K context: `0.7426 ms`; with CUDA Graph:
  `0.7356 ms`.
- Qwen3.6-27B vision block, 32x32 grid: `0.2948 ms`.
- Qwen3.6-27B 27-layer vision wrapper, 32x32 grid: `9.0645 ms`.
