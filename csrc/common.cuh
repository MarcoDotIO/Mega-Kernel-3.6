#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#define MK_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define MK_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define MK_CHECK_INPUT(x) \
    MK_CHECK_CUDA(x);     \
    MK_CHECK_CONTIGUOUS(x)

template <typename T>
__device__ __forceinline__ float mk_to_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T mk_from_float(float value) {
    return static_cast<T>(value);
}

__device__ __forceinline__ float mk_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float mk_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float mk_block_sum(float value) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    value = mk_warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = tid < 8 ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = mk_warp_sum(value);
    }
    __syncthreads();
    if (tid == 0) {
        shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}
