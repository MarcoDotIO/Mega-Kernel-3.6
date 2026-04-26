#!/usr/bin/env bash
set -euo pipefail

VENV=${VENV:-/tmp/mega-kernel-parity-venv}
export TMPDIR=${TMPDIR:-/tmp/mega-kernel-parity-build}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/tmp/mega-kernel-pip-cache}
export HF_HOME=${HF_HOME:-/tmp/mega-kernel-hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/mega-kernel-xdg}

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
python3 -m venv "$VENV"
. "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -U ninja pytest pybind11 triton transformers huggingface_hub safetensors
python -m pip install vllm==0.19.1
python -m pip install --no-deps sglang==0.5.10.post1

python - <<'PY'
for name in ["torch", "triton", "vllm", "sglang"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "installed"))
PY
