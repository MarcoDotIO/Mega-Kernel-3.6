#include "common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum_256(float value, float* shared) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    __syncthreads();
    return value;
}

constexpr int kChunkSize = 128;

template <typename scalar_t>
__global__ void attention_chunk_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int batch,
    int query_heads,
    int seq_len,
    int kv_heads,
    int head_dim,
    int chunks,
    float scale) {
    int b = blockIdx.x;
    int qh = blockIdx.y;
    int chunk = blockIdx.z;
    int tid = threadIdx.x;
    int group = query_heads / kv_heads;
    int kvh = qh / group;
    int start = chunk * kChunkSize;
    int end = min(seq_len, start + kChunkSize);
    int partial_base = ((b * query_heads + qh) * chunks + chunk);

    extern __shared__ float shared[];
    float max_score = -3.4028234663852886e38f;
    for (int t = start; t < end; ++t) {
        float local = 0.0f;
        if (tid < head_dim) {
            float qv = mk_to_float(q[(b * query_heads + qh) * head_dim + tid]);
            float kv = mk_to_float(k_cache[((b * seq_len + t) * kv_heads + kvh) * head_dim + tid]);
            local = qv * kv;
        }
        float dot = block_reduce_sum_256(local, shared);
        if (tid == 0) {
            max_score = fmaxf(max_score, dot * scale);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared[0] = max_score;
    }
    __syncthreads();
    max_score = shared[0];
    __syncthreads();

    float denom = 0.0f;
    float acc = 0.0f;
    for (int t = start; t < end; ++t) {
        float local = 0.0f;
        if (tid < head_dim) {
            float qv = mk_to_float(q[(b * query_heads + qh) * head_dim + tid]);
            float kv = mk_to_float(k_cache[((b * seq_len + t) * kv_heads + kvh) * head_dim + tid]);
            local = qv * kv;
        }
        float dot = block_reduce_sum_256(local, shared);
        float weight = __expf(dot * scale - max_score);
        if (tid == 0) {
            denom += weight;
        }
        if (tid < head_dim) {
            acc += weight * mk_to_float(v_cache[((b * seq_len + t) * kv_heads + kvh) * head_dim + tid]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_m[partial_base] = max_score;
        partial_l[partial_base] = denom;
    }
    if (tid < head_dim) {
        partial_out[(partial_base * head_dim) + tid] = acc;
    }
}

template <typename scalar_t>
__global__ void attention_merge_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    scalar_t* __restrict__ out,
    int batch,
    int query_heads,
    int head_dim,
    int chunks) {
    int b = blockIdx.x;
    int qh = blockIdx.y;
    int tid = threadIdx.x;
    int base = b * query_heads + qh;
    extern __shared__ float shared[];

    if (tid == 0) {
        float max_m = -3.4028234663852886e38f;
        for (int c = 0; c < chunks; ++c) {
            max_m = fmaxf(max_m, partial_m[base * chunks + c]);
        }
        float denom = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            denom += __expf(partial_m[base * chunks + c] - max_m) * partial_l[base * chunks + c];
        }
        shared[0] = max_m;
        shared[1] = denom;
    }
    __syncthreads();

    if (tid < head_dim) {
        float max_m = shared[0];
        float denom = shared[1];
        float acc = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            float scale = __expf(partial_m[base * chunks + c] - max_m);
            acc += scale * partial_out[(base * chunks + c) * head_dim + tid];
        }
        out[base * head_dim + tid] = mk_from_float<scalar_t>(acc / denom);
    }
}

}  // namespace

torch::Tensor full_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    double scale) {
    MK_CHECK_INPUT(q);
    MK_CHECK_INPUT(k_cache);
    MK_CHECK_INPUT(v_cache);
    TORCH_CHECK(q.dim() == 3, "q must be [batch, query_heads, head_dim]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [batch, seq_len, kv_heads, head_dim]");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache shape must match k_cache");
    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and k_cache dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q and v_cache dtype mismatch");

    int batch = static_cast<int>(q.size(0));
    int query_heads = static_cast<int>(q.size(1));
    int head_dim = static_cast<int>(q.size(2));
    int seq_len = static_cast<int>(k_cache.size(1));
    int kv_heads = static_cast<int>(k_cache.size(2));
    TORCH_CHECK(k_cache.size(0) == batch, "batch mismatch");
    TORCH_CHECK(k_cache.size(3) == head_dim, "head_dim mismatch");
    TORCH_CHECK(query_heads % kv_heads == 0, "query_heads must be divisible by kv_heads");
    TORCH_CHECK(head_dim <= 256, "head_dim > 256 is not supported by this kernel");
    TORCH_CHECK(seq_len > 0, "seq_len must be positive");

    auto out = torch::empty_like(q);
    int chunks = (seq_len + kChunkSize - 1) / kChunkSize;
    auto float_opts = q.options().dtype(torch::kFloat32);
    auto partial_out = torch::empty({batch, query_heads, chunks, head_dim}, float_opts);
    auto partial_m = torch::empty({batch, query_heads, chunks}, float_opts);
    auto partial_l = torch::empty({batch, query_heads, chunks}, float_opts);
    constexpr int threads = 256;
    dim3 chunk_grid(batch, query_heads, chunks);
    dim3 merge_grid(batch, query_heads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    size_t shared_bytes = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "full_attention_decode_cuda", [&] {
        attention_chunk_kernel<scalar_t><<<chunk_grid, threads, shared_bytes, stream>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            partial_out.data_ptr<float>(),
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            batch,
            query_heads,
            seq_len,
            kv_heads,
            head_dim,
            chunks,
            static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        attention_merge_kernel<scalar_t><<<merge_grid, threads, shared_bytes, stream>>>(
            partial_out.data_ptr<float>(),
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            batch,
            query_heads,
            head_dim,
            chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out;
}
