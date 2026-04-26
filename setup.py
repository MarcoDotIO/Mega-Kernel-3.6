import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    import pybind11
except Exception:
    include_dirs = []
else:
    include_dirs = [pybind11.get_include()]


if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"


setup(
    name="mega-kernel-qwen36",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="mega_kernel_qwen36._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/moe_decode.cu",
                "csrc/attention_decode.cu",
                "csrc/linear_attention_decode.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
