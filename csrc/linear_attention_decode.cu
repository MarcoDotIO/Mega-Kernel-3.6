#include "common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

template <typename scalar_t>
__global__ void linear_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ state,
    scalar_t* __restrict__ out,
    float* __restrict__ new_state,
    int batch,
    int heads,
    int key_dim,
    int value_dim,
    float decay) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * value_dim;
    if (n >= total) {
        return;
    }

    int vd = n % value_dim;
    int head = (n / value_dim) % heads;
    int b = n / (heads * value_dim);
    int base_bh = (b * heads + head);
    float vv = mk_to_float(v[base_bh * value_dim + vd]);
    float acc = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        int state_idx = ((base_bh * key_dim + kd) * value_dim + vd);
        float updated = state[state_idx] * decay + mk_to_float(k[base_bh * key_dim + kd]) * vv;
        new_state[state_idx] = updated;
        acc += mk_to_float(q[base_bh * key_dim + kd]) * updated;
    }
    out[n] = mk_from_float<scalar_t>(acc);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> linear_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    double decay) {
    MK_CHECK_INPUT(q);
    MK_CHECK_INPUT(k);
    MK_CHECK_INPUT(v);
    MK_CHECK_INPUT(state);
    TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, key_dim]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
    TORCH_CHECK(v.dim() == 3, "v must be [batch, heads, value_dim]");
    TORCH_CHECK(state.dim() == 4, "state must be [batch, heads, key_dim, value_dim]");
    TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be float32");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v dtype mismatch");

    int batch = static_cast<int>(q.size(0));
    int heads = static_cast<int>(q.size(1));
    int key_dim = static_cast<int>(q.size(2));
    int value_dim = static_cast<int>(v.size(2));
    TORCH_CHECK(v.size(0) == batch && v.size(1) == heads, "v batch/head mismatch");
    TORCH_CHECK(state.size(0) == batch && state.size(1) == heads && state.size(2) == key_dim && state.size(3) == value_dim, "state shape mismatch");

    auto out = torch::empty({batch, heads, value_dim}, q.options());
    auto new_state = torch::empty_like(state);
    constexpr int threads = 256;
    int total = batch * heads * value_dim;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "linear_attention_decode_cuda", [&] {
        linear_attention_kernel<scalar_t><<<(total + threads - 1) / threads, threads, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            state.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            new_state.data_ptr<float>(),
            batch,
            heads,
            key_dim,
            value_dim,
            static_cast<float>(decay));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return std::make_tuple(out, new_state);
}
