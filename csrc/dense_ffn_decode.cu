#include "common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void dense_normalize_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ norm_weight,
    float* __restrict__ x_norm,
    int batch,
    int hidden,
    float eps) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int h = tid; h < hidden; h += blockDim.x) {
        float v = mk_to_float(x[b * hidden + h]);
        local += v * v;
    }
    float sum = mk_block_sum(local);
    float inv_rms = rsqrtf(sum / static_cast<float>(hidden) + eps);
    for (int h = tid; h < hidden; h += blockDim.x) {
        x_norm[b * hidden + h] = mk_to_float(x[b * hidden + h]) * inv_rms * mk_to_float(norm_weight[h]);
    }
}

template <typename scalar_t>
__global__ void dense_activation_kernel(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ up_weight,
    float* __restrict__ activations,
    int batch,
    int hidden,
    int intermediate) {
    int n = blockIdx.x;
    int i = n % intermediate;
    int b = n / intermediate;
    int tid = threadIdx.x;

    const float* xb = x_norm + b * hidden;
    const scalar_t* gate = gate_weight + i * hidden;
    const scalar_t* up = up_weight + i * hidden;
    float gate_local = 0.0f;
    float up_local = 0.0f;
    for (int h = tid; h < hidden; h += blockDim.x) {
        float xv = xb[h];
        gate_local += xv * mk_to_float(gate[h]);
        up_local += xv * mk_to_float(up[h]);
    }
    float gate_sum = mk_block_sum(gate_local);
    float up_sum = mk_block_sum(up_local);
    if (tid == 0) {
        activations[b * intermediate + i] = mk_silu(gate_sum) * up_sum;
    }
}

template <typename scalar_t>
__global__ void dense_output_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ activations,
    const scalar_t* __restrict__ down_weight,
    scalar_t* __restrict__ out,
    int batch,
    int hidden,
    int intermediate,
    bool add_residual) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (n >= total) {
        return;
    }

    int h = n % hidden;
    int b = n / hidden;
    float acc = add_residual ? mk_to_float(x[n]) : 0.0f;
    const scalar_t* down = down_weight + h * intermediate;
    const float* act = activations + b * intermediate;
    for (int i = 0; i < intermediate; ++i) {
        acc += act[i] * mk_to_float(down[i]);
    }
    out[n] = mk_from_float<scalar_t>(acc);
}

}  // namespace

torch::Tensor dense_ffn_decode_cuda(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor gate_weight,
    torch::Tensor up_weight,
    torch::Tensor down_weight,
    double eps,
    bool add_residual) {
    MK_CHECK_INPUT(x);
    MK_CHECK_INPUT(norm_weight);
    MK_CHECK_INPUT(gate_weight);
    MK_CHECK_INPUT(up_weight);
    MK_CHECK_INPUT(down_weight);
    TORCH_CHECK(x.dim() == 2, "x must be [batch, hidden]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be [hidden]");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be [intermediate, hidden]");
    TORCH_CHECK(up_weight.sizes() == gate_weight.sizes(), "up_weight shape must match gate_weight");
    TORCH_CHECK(down_weight.dim() == 2, "down_weight must be [hidden, intermediate]");
    TORCH_CHECK(x.scalar_type() == norm_weight.scalar_type(), "x and norm_weight dtype mismatch");
    TORCH_CHECK(x.scalar_type() == gate_weight.scalar_type(), "x and gate_weight dtype mismatch");
    TORCH_CHECK(x.scalar_type() == up_weight.scalar_type(), "x and up_weight dtype mismatch");
    TORCH_CHECK(x.scalar_type() == down_weight.scalar_type(), "x and down_weight dtype mismatch");

    int batch = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    int intermediate = static_cast<int>(gate_weight.size(0));
    TORCH_CHECK(norm_weight.size(0) == hidden, "norm_weight hidden mismatch");
    TORCH_CHECK(gate_weight.size(1) == hidden, "gate_weight hidden mismatch");
    TORCH_CHECK(down_weight.size(0) == hidden && down_weight.size(1) == intermediate, "down_weight shape mismatch");

    auto float_opts = x.options().dtype(torch::kFloat32);
    auto x_norm = torch::empty({batch, hidden}, float_opts);
    auto activations = torch::empty({batch, intermediate}, float_opts);
    auto out = torch::empty_like(x);

    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "dense_ffn_decode_cuda", [&] {
        dense_normalize_kernel<scalar_t><<<batch, threads, shared_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            norm_weight.data_ptr<scalar_t>(),
            x_norm.data_ptr<float>(),
            batch,
            hidden,
            static_cast<float>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dense_activation_kernel<scalar_t><<<batch * intermediate, threads, shared_bytes, stream>>>(
            x_norm.data_ptr<float>(),
            gate_weight.data_ptr<scalar_t>(),
            up_weight.data_ptr<scalar_t>(),
            activations.data_ptr<float>(),
            batch,
            hidden,
            intermediate);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int total = batch * hidden;
        dense_output_kernel<scalar_t><<<(total + threads - 1) / threads, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            activations.data_ptr<float>(),
            down_weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            hidden,
            intermediate,
            add_residual);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return out;
}
