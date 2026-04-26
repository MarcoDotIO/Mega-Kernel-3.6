#include "common.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

template <typename scalar_t>
__global__ void normalize_kernel(
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
__global__ void router_logits_kernel(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ router_weight,
    float* __restrict__ logits,
    int batch,
    int hidden,
    int experts) {
    int b = blockIdx.x;
    int e = blockIdx.y;
    int tid = threadIdx.x;
    float local = 0.0f;
    const scalar_t* router = router_weight + e * hidden;
    const float* xb = x_norm + b * hidden;
    for (int h = tid; h < hidden; h += blockDim.x) {
        local += xb[h] * mk_to_float(router[h]);
    }
    float sum = mk_block_sum(local);
    if (tid == 0) {
        logits[b * experts + e] = sum;
    }
}

__global__ void topk_softmax_kernel(
    const float* __restrict__ logits,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    int batch,
    int experts,
    int top_k) {
    int b = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const float* row = logits + b * experts;
    int64_t* out_idx = indices + b * top_k;
    float* out_w = weights + b * top_k;

    for (int k = 0; k < top_k; ++k) {
        float best = -3.4028234663852886e38f;
        int best_idx = 0;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int prev = 0; prev < k; ++prev) {
                used = used || (out_idx[prev] == e);
            }
            float value = used ? -3.4028234663852886e38f : row[e];
            if (value > best) {
                best = value;
                best_idx = e;
            }
        }
        out_idx[k] = static_cast<int64_t>(best_idx);
        out_w[k] = best;
    }

    float max_v = out_w[0];
    for (int k = 1; k < top_k; ++k) {
        max_v = fmaxf(max_v, out_w[k]);
    }
    float denom = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        out_w[k] = __expf(out_w[k] - max_v);
        denom += out_w[k];
    }
    for (int k = 0; k < top_k; ++k) {
        out_w[k] /= denom;
    }
}

template <typename scalar_t>
__global__ void expert_activation_kernel(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ expert_gate,
    const scalar_t* __restrict__ expert_up,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ activations,
    int batch,
    int hidden,
    int top_k,
    int intermediate) {
    int n = blockIdx.x;
    int i = n % intermediate;
    int slot = (n / intermediate) % top_k;
    int b = n / (intermediate * top_k);
    int tid = threadIdx.x;
    int expert = static_cast<int>(indices[b * top_k + slot]);

    const float* xb = x_norm + b * hidden;
    const scalar_t* gate = expert_gate + (expert * intermediate + i) * hidden;
    const scalar_t* up = expert_up + (expert * intermediate + i) * hidden;
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
        activations[(b * top_k + slot) * intermediate + i] =
            weights[b * top_k + slot] * mk_silu(gate_sum) * up_sum;
    }
}

template <typename scalar_t>
__global__ void shared_activation_kernel(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ shared_gate,
    const scalar_t* __restrict__ shared_up,
    float* __restrict__ shared_activations,
    int batch,
    int hidden,
    int shared_intermediate) {
    int n = blockIdx.x;
    int i = n % shared_intermediate;
    int b = n / shared_intermediate;
    int tid = threadIdx.x;

    const float* xb = x_norm + b * hidden;
    const scalar_t* gate = shared_gate + i * hidden;
    const scalar_t* up = shared_up + i * hidden;
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
        shared_activations[b * shared_intermediate + i] = mk_silu(gate_sum) * up_sum;
    }
}

template <typename scalar_t>
__global__ void moe_output_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ activations,
    const float* __restrict__ shared_activations,
    const scalar_t* __restrict__ expert_down,
    const scalar_t* __restrict__ shared_down,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ indices,
    int batch,
    int hidden,
    int top_k,
    int intermediate,
    int shared_intermediate,
    bool add_residual) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (n >= total) {
        return;
    }

    int h = n % hidden;
    int b = n / hidden;
    float acc = add_residual ? mk_to_float(x[n]) : 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        int expert = static_cast<int>(indices[b * top_k + slot]);
        const scalar_t* down = expert_down + (expert * hidden + h) * intermediate;
        const float* act = activations + (b * top_k + slot) * intermediate;
        for (int i = 0; i < intermediate; ++i) {
            acc += act[i] * mk_to_float(down[i]);
        }
    }

    const scalar_t* shared_row = shared_down + h * shared_intermediate;
    const float* shared_act = shared_activations + b * shared_intermediate;
    for (int i = 0; i < shared_intermediate; ++i) {
        acc += shared_act[i] * mk_to_float(shared_row[i]);
    }
    out[n] = mk_from_float<scalar_t>(acc);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_decode_cuda(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    torch::Tensor expert_gate,
    torch::Tensor expert_up,
    torch::Tensor expert_down,
    torch::Tensor shared_gate,
    torch::Tensor shared_up,
    torch::Tensor shared_down,
    double eps,
    int64_t top_k,
    bool add_residual) {
    MK_CHECK_INPUT(x);
    MK_CHECK_INPUT(norm_weight);
    MK_CHECK_INPUT(router_weight);
    MK_CHECK_INPUT(expert_gate);
    MK_CHECK_INPUT(expert_up);
    MK_CHECK_INPUT(expert_down);
    MK_CHECK_INPUT(shared_gate);
    MK_CHECK_INPUT(shared_up);
    MK_CHECK_INPUT(shared_down);
    TORCH_CHECK(x.dim() == 2, "x must be [batch, hidden]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be [hidden]");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be [experts, hidden]");
    TORCH_CHECK(expert_gate.dim() == 3, "expert_gate must be [experts, intermediate, hidden]");
    TORCH_CHECK(expert_up.sizes() == expert_gate.sizes(), "expert_up shape must match expert_gate");
    TORCH_CHECK(expert_down.dim() == 3, "expert_down must be [experts, hidden, intermediate]");
    TORCH_CHECK(shared_gate.dim() == 2, "shared_gate must be [shared_intermediate, hidden]");
    TORCH_CHECK(shared_up.sizes() == shared_gate.sizes(), "shared_up shape must match shared_gate");
    TORCH_CHECK(shared_down.dim() == 2, "shared_down must be [hidden, shared_intermediate]");
    TORCH_CHECK(x.scalar_type() == norm_weight.scalar_type(), "x and norm_weight dtype mismatch");
    TORCH_CHECK(x.scalar_type() == router_weight.scalar_type(), "x and router_weight dtype mismatch");
    TORCH_CHECK(x.scalar_type() == expert_gate.scalar_type(), "x and expert_gate dtype mismatch");
    TORCH_CHECK(x.scalar_type() == expert_up.scalar_type(), "x and expert_up dtype mismatch");
    TORCH_CHECK(x.scalar_type() == expert_down.scalar_type(), "x and expert_down dtype mismatch");
    TORCH_CHECK(x.scalar_type() == shared_gate.scalar_type(), "x and shared_gate dtype mismatch");
    TORCH_CHECK(x.scalar_type() == shared_up.scalar_type(), "x and shared_up dtype mismatch");
    TORCH_CHECK(x.scalar_type() == shared_down.scalar_type(), "x and shared_down dtype mismatch");

    int batch = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    int experts = static_cast<int>(router_weight.size(0));
    int intermediate = static_cast<int>(expert_gate.size(1));
    int shared_intermediate = static_cast<int>(shared_gate.size(0));
    TORCH_CHECK(top_k > 0 && top_k <= experts, "top_k must be in [1, experts]");
    TORCH_CHECK(norm_weight.size(0) == hidden, "norm_weight hidden mismatch");
    TORCH_CHECK(router_weight.size(1) == hidden, "router_weight hidden mismatch");
    TORCH_CHECK(expert_gate.size(0) == experts && expert_gate.size(2) == hidden, "expert_gate shape mismatch");
    TORCH_CHECK(expert_down.size(0) == experts && expert_down.size(1) == hidden && expert_down.size(2) == intermediate, "expert_down shape mismatch");
    TORCH_CHECK(shared_gate.size(1) == hidden, "shared_gate hidden mismatch");
    TORCH_CHECK(shared_down.size(0) == hidden && shared_down.size(1) == shared_intermediate, "shared_down shape mismatch");

    auto float_opts = x.options().dtype(torch::kFloat32);
    auto index_opts = x.options().dtype(torch::kInt64);
    auto x_norm = torch::empty({batch, hidden}, float_opts);
    auto logits = torch::empty({batch, experts}, float_opts);
    auto indices = torch::empty({batch, top_k}, index_opts);
    auto weights = torch::empty({batch, top_k}, float_opts);
    auto activations = torch::empty({batch, top_k, intermediate}, float_opts);
    auto shared_activations = torch::empty({batch, shared_intermediate}, float_opts);
    auto out = torch::empty_like(x);

    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "moe_decode_cuda", [&] {
        normalize_kernel<scalar_t><<<batch, threads, shared_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            norm_weight.data_ptr<scalar_t>(),
            x_norm.data_ptr<float>(),
            batch,
            hidden,
            static_cast<float>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dim3 router_grid(batch, experts);
        router_logits_kernel<scalar_t><<<router_grid, threads, shared_bytes, stream>>>(
            x_norm.data_ptr<float>(),
            router_weight.data_ptr<scalar_t>(),
            logits.data_ptr<float>(),
            batch,
            hidden,
            experts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        topk_softmax_kernel<<<batch, 1, 0, stream>>>(
            logits.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            batch,
            experts,
            static_cast<int>(top_k));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int expert_blocks = batch * static_cast<int>(top_k) * intermediate;
        expert_activation_kernel<scalar_t><<<expert_blocks, threads, shared_bytes, stream>>>(
            x_norm.data_ptr<float>(),
            expert_gate.data_ptr<scalar_t>(),
            expert_up.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            activations.data_ptr<float>(),
            batch,
            hidden,
            static_cast<int>(top_k),
            intermediate);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int shared_blocks = batch * shared_intermediate;
        shared_activation_kernel<scalar_t><<<shared_blocks, threads, shared_bytes, stream>>>(
            x_norm.data_ptr<float>(),
            shared_gate.data_ptr<scalar_t>(),
            shared_up.data_ptr<scalar_t>(),
            shared_activations.data_ptr<float>(),
            batch,
            hidden,
            shared_intermediate);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        int total = batch * hidden;
        moe_output_kernel<scalar_t><<<(total + threads - 1) / threads, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            activations.data_ptr<float>(),
            shared_activations.data_ptr<float>(),
            expert_down.data_ptr<scalar_t>(),
            shared_down.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            batch,
            hidden,
            static_cast<int>(top_k),
            intermediate,
            shared_intermediate,
            add_residual);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return std::make_tuple(out, indices, weights);
}
