#include "common.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <tuple>

namespace {
namespace cg = cooperative_groups;

template <typename scalar_t>
__device__ void mpk_task_normalize(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ norm_weight,
    float* __restrict__ x_norm,
    int b,
    int hidden,
    float eps) {
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

// Fallback task bodies for devices that cannot use cooperative grid barriers.
template <typename scalar_t>
__device__ void mpk_task_router_topk_warp_tiled(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ router_weight,
    float* __restrict__ logits,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    int b,
    int hidden,
    int experts,
    int top_k) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    const float* xb = x_norm + b * hidden;
    float* row_logits = logits + b * experts;

    for (int e = warp; e < experts; e += num_warps) {
        const scalar_t* router = router_weight + e * hidden;
        float local = 0.0f;
        for (int h = lane; h < hidden; h += 32) {
            local += xb[h] * mk_to_float(router[h]);
        }
        float logit = mk_warp_sum(local);
        if (lane == 0) {
            row_logits[e] = logit;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int64_t* out_idx = indices + b * top_k;
        float* out_w = weights + b * top_k;
        for (int k = 0; k < top_k; ++k) {
            out_idx[k] = 0;
            out_w[k] = -3.4028234663852886e38f;
        }

        for (int e = 0; e < experts; ++e) {
            float logit = row_logits[e];
            int insert_at = -1;
            for (int k = 0; k < top_k; ++k) {
                if (logit > out_w[k]) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at >= 0) {
                for (int k = top_k - 1; k > insert_at; --k) {
                    out_w[k] = out_w[k - 1];
                    out_idx[k] = out_idx[k - 1];
                }
                out_w[insert_at] = logit;
                out_idx[insert_at] = static_cast<int64_t>(e);
            }
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
}

template <typename scalar_t>
__device__ void mpk_task_routed_experts_warp_tiled(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ expert_gate,
    const scalar_t* __restrict__ expert_up,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ activations,
    int b,
    int hidden,
    int top_k,
    int intermediate) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    const float* xb = x_norm + b * hidden;

    for (int slot = 0; slot < top_k; ++slot) {
        int expert = static_cast<int>(indices[b * top_k + slot]);
        float route_weight = weights[b * top_k + slot];
        for (int i = warp; i < intermediate; i += num_warps) {
            const scalar_t* gate = expert_gate + (expert * intermediate + i) * hidden;
            const scalar_t* up = expert_up + (expert * intermediate + i) * hidden;
            float gate_local = 0.0f;
            float up_local = 0.0f;
            for (int h = lane; h < hidden; h += 32) {
                float xv = xb[h];
                gate_local += xv * mk_to_float(gate[h]);
                up_local += xv * mk_to_float(up[h]);
            }
            float gate_sum = mk_warp_sum(gate_local);
            float up_sum = mk_warp_sum(up_local);
            if (lane == 0) {
                activations[(b * top_k + slot) * intermediate + i] = route_weight * mk_silu(gate_sum) * up_sum;
            }
        }
    }
}

template <typename scalar_t>
__device__ void mpk_task_shared_expert_warp_tiled(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ shared_gate,
    const scalar_t* __restrict__ shared_up,
    float* __restrict__ shared_activations,
    int b,
    int hidden,
    int shared_intermediate) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    const float* xb = x_norm + b * hidden;

    for (int i = warp; i < shared_intermediate; i += num_warps) {
        const scalar_t* gate = shared_gate + i * hidden;
        const scalar_t* up = shared_up + i * hidden;
        float gate_local = 0.0f;
        float up_local = 0.0f;
        for (int h = lane; h < hidden; h += 32) {
            float xv = xb[h];
            gate_local += xv * mk_to_float(gate[h]);
            up_local += xv * mk_to_float(up[h]);
        }
        float gate_sum = mk_warp_sum(gate_local);
        float up_sum = mk_warp_sum(up_local);
        if (lane == 0) {
            shared_activations[b * shared_intermediate + i] = mk_silu(gate_sum) * up_sum;
        }
    }
}

template <typename scalar_t>
__device__ void mpk_task_moe_output_warp_tiled(
    const scalar_t* __restrict__ x,
    const float* __restrict__ activations,
    const float* __restrict__ shared_activations,
    const scalar_t* __restrict__ expert_down,
    const scalar_t* __restrict__ shared_down,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ indices,
    int b,
    int hidden,
    int top_k,
    int intermediate,
    int shared_intermediate,
    bool add_residual) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;

    for (int h = warp; h < hidden; h += num_warps) {
        float local = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            int expert = static_cast<int>(indices[b * top_k + slot]);
            const scalar_t* down = expert_down + (expert * hidden + h) * intermediate;
            const float* act = activations + (b * top_k + slot) * intermediate;
            for (int i = lane; i < intermediate; i += 32) {
                local += act[i] * mk_to_float(down[i]);
            }
        }

        const scalar_t* shared_row = shared_down + h * shared_intermediate;
        const float* shared_act = shared_activations + b * shared_intermediate;
        for (int i = lane; i < shared_intermediate; i += 32) {
            local += shared_act[i] * mk_to_float(shared_row[i]);
        }
        float acc = mk_warp_sum(local);
        if (lane == 0) {
            int n = b * hidden + h;
            if (add_residual) {
                acc += mk_to_float(x[n]);
            }
            out[n] = mk_from_float<scalar_t>(acc);
        }
    }
}

// Cooperative task bodies spread one decode row across many resident CTAs.
template <typename scalar_t>
__device__ void mpk_task_normalize_coop(
    const scalar_t* __restrict__ x,
    float* __restrict__ worker_scratch,
    int b,
    int worker,
    int workers,
    int hidden) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int h = worker * blockDim.x + tid; h < hidden; h += workers * blockDim.x) {
        float v = mk_to_float(x[b * hidden + h]);
        local += v * v;
    }
    float partial = mk_block_sum(local);
    if (tid == 0) {
        worker_scratch[b * workers + worker] = partial;
    }
}

__device__ void mpk_task_finish_normalize_coop(
    const float* __restrict__ worker_scratch,
    float* __restrict__ norm_scales,
    int b,
    int workers,
    int hidden,
    float eps) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int w = tid; w < workers; w += blockDim.x) {
        local += worker_scratch[b * workers + w];
    }
    float sum = mk_block_sum(local);
    if (tid == 0) {
        norm_scales[b] = rsqrtf(sum / static_cast<float>(hidden) + eps);
    }
}

template <typename scalar_t>
__device__ void mpk_task_write_normalized_coop(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ norm_weight,
    const float* __restrict__ norm_scales,
    float* __restrict__ x_norm,
    int b,
    int worker,
    int workers,
    int hidden) {
    int tid = threadIdx.x;
    float inv_rms = norm_scales[b];
    for (int h = worker * blockDim.x + tid; h < hidden; h += workers * blockDim.x) {
        x_norm[b * hidden + h] = mk_to_float(x[b * hidden + h]) * inv_rms * mk_to_float(norm_weight[h]);
    }
}

template <typename scalar_t>
__device__ void mpk_task_router_logits_coop(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ router_weight,
    float* __restrict__ logits,
    int b,
    int worker,
    int workers,
    int hidden,
    int experts) {
    int tid = threadIdx.x;
    const float* xb = x_norm + b * hidden;
    for (int e = worker; e < experts; e += workers) {
        const scalar_t* router = router_weight + e * hidden;
        float local = 0.0f;
        for (int h = tid; h < hidden; h += blockDim.x) {
            local += xb[h] * mk_to_float(router[h]);
        }
        float logit = mk_block_sum(local);
        if (tid == 0) {
            logits[b * experts + e] = logit;
        }
    }
}

__device__ void mpk_task_router_topk_from_logits_coop(
    const float* __restrict__ logits,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    int b,
    int experts,
    int top_k) {
    int tid = threadIdx.x;
    if (tid != 0) {
        return;
    }

    const float* row_logits = logits + b * experts;
    int64_t* out_idx = indices + b * top_k;
    float* out_w = weights + b * top_k;
    for (int k = 0; k < top_k; ++k) {
        out_idx[k] = 0;
        out_w[k] = -3.4028234663852886e38f;
    }

    for (int e = 0; e < experts; ++e) {
        float logit = row_logits[e];
        int insert_at = -1;
        for (int k = 0; k < top_k; ++k) {
            if (logit > out_w[k]) {
                insert_at = k;
                break;
            }
        }
        if (insert_at >= 0) {
            for (int k = top_k - 1; k > insert_at; --k) {
                out_w[k] = out_w[k - 1];
                out_idx[k] = out_idx[k - 1];
            }
            out_w[insert_at] = logit;
            out_idx[insert_at] = static_cast<int64_t>(e);
        }
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
__device__ void mpk_task_routed_experts_coop(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ expert_gate,
    const scalar_t* __restrict__ expert_up,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ activations,
    int b,
    int worker,
    int workers,
    int hidden,
    int top_k,
    int intermediate) {
    int tid = threadIdx.x;
    const float* xb = x_norm + b * hidden;
    int tasks = top_k * intermediate;
    for (int task = worker; task < tasks; task += workers) {
        int slot = task / intermediate;
        int i = task - slot * intermediate;
        int expert = static_cast<int>(indices[b * top_k + slot]);
        float route_weight = weights[b * top_k + slot];
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
            activations[(b * top_k + slot) * intermediate + i] = route_weight * mk_silu(gate_sum) * up_sum;
        }
    }
}

template <typename scalar_t>
__device__ void mpk_task_shared_expert_coop(
    const float* __restrict__ x_norm,
    const scalar_t* __restrict__ shared_gate,
    const scalar_t* __restrict__ shared_up,
    float* __restrict__ shared_activations,
    int b,
    int worker,
    int workers,
    int hidden,
    int shared_intermediate) {
    int tid = threadIdx.x;
    const float* xb = x_norm + b * hidden;
    for (int i = worker; i < shared_intermediate; i += workers) {
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
}

template <typename scalar_t>
__device__ void mpk_task_moe_output_coop(
    const scalar_t* __restrict__ x,
    const float* __restrict__ activations,
    const float* __restrict__ shared_activations,
    const scalar_t* __restrict__ expert_down,
    const scalar_t* __restrict__ shared_down,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ indices,
    int b,
    int worker,
    int workers,
    int hidden,
    int top_k,
    int intermediate,
    int shared_intermediate,
    bool add_residual) {
    int tid = threadIdx.x;
    for (int h = worker; h < hidden; h += workers) {
        float local = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            int expert = static_cast<int>(indices[b * top_k + slot]);
            const scalar_t* down = expert_down + (expert * hidden + h) * intermediate;
            const float* act = activations + (b * top_k + slot) * intermediate;
            for (int i = tid; i < intermediate; i += blockDim.x) {
                local += act[i] * mk_to_float(down[i]);
            }
        }

        const scalar_t* shared_row = shared_down + h * shared_intermediate;
        const float* shared_act = shared_activations + b * shared_intermediate;
        for (int i = tid; i < shared_intermediate; i += blockDim.x) {
            local += shared_act[i] * mk_to_float(shared_row[i]);
        }
        float acc = mk_block_sum(local);
        if (tid == 0) {
            int n = b * hidden + h;
            if (add_residual) {
                acc += mk_to_float(x[n]);
            }
            out[n] = mk_from_float<scalar_t>(acc);
        }
    }
}

template <typename scalar_t>
__global__ void moe_decode_persistent_coop_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ norm_weight,
    const scalar_t* __restrict__ router_weight,
    const scalar_t* __restrict__ expert_gate,
    const scalar_t* __restrict__ expert_up,
    const scalar_t* __restrict__ expert_down,
    const scalar_t* __restrict__ shared_gate,
    const scalar_t* __restrict__ shared_up,
    const scalar_t* __restrict__ shared_down,
    float* __restrict__ x_norm,
    float* __restrict__ router_logits,
    float* __restrict__ worker_scratch,
    float* __restrict__ norm_scales,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    float* __restrict__ activations,
    float* __restrict__ shared_activations,
    scalar_t* __restrict__ out,
    int batch,
    int hidden,
    int experts,
    int top_k,
    int intermediate,
    int shared_intermediate,
    float eps,
    bool add_residual,
    int workers_per_batch) {
    cg::grid_group grid = cg::this_grid();
    int b = blockIdx.x / workers_per_batch;
    int worker = blockIdx.x - b * workers_per_batch;

    mpk_task_normalize_coop(
        x,
        worker_scratch,
        b,
        worker,
        workers_per_batch,
        hidden);
    grid.sync();
    if (worker == 0) {
        mpk_task_finish_normalize_coop(worker_scratch, norm_scales, b, workers_per_batch, hidden, eps);
    }
    grid.sync();
    mpk_task_write_normalized_coop(x, norm_weight, norm_scales, x_norm, b, worker, workers_per_batch, hidden);
    grid.sync();

    mpk_task_router_logits_coop(
        x_norm,
        router_weight,
        router_logits,
        b,
        worker,
        workers_per_batch,
        hidden,
        experts);
    grid.sync();
    if (worker == 0) {
        mpk_task_router_topk_from_logits_coop(router_logits, indices, weights, b, experts, top_k);
    }
    grid.sync();

    mpk_task_routed_experts_coop(
        x_norm,
        expert_gate,
        expert_up,
        indices,
        weights,
        activations,
        b,
        worker,
        workers_per_batch,
        hidden,
        top_k,
        intermediate);
    grid.sync();
    mpk_task_shared_expert_coop(
        x_norm,
        shared_gate,
        shared_up,
        shared_activations,
        b,
        worker,
        workers_per_batch,
        hidden,
        shared_intermediate);
    grid.sync();
    mpk_task_moe_output_coop(
        x,
        activations,
        shared_activations,
        expert_down,
        shared_down,
        out,
        indices,
        b,
        worker,
        workers_per_batch,
        hidden,
        top_k,
        intermediate,
        shared_intermediate,
        add_residual);
}

template <typename scalar_t>
__global__ void moe_decode_persistent_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ norm_weight,
    const scalar_t* __restrict__ router_weight,
    const scalar_t* __restrict__ expert_gate,
    const scalar_t* __restrict__ expert_up,
    const scalar_t* __restrict__ expert_down,
    const scalar_t* __restrict__ shared_gate,
    const scalar_t* __restrict__ shared_up,
    const scalar_t* __restrict__ shared_down,
    float* __restrict__ x_norm,
    float* __restrict__ router_logits,
    int64_t* __restrict__ indices,
    float* __restrict__ weights,
    float* __restrict__ activations,
    float* __restrict__ shared_activations,
    scalar_t* __restrict__ out,
    int batch,
    int hidden,
    int experts,
    int top_k,
    int intermediate,
    int shared_intermediate,
    float eps,
    bool add_residual) {
    int b = blockIdx.x;
    if (b >= batch) {
        return;
    }

    mpk_task_normalize(x, norm_weight, x_norm, b, hidden, eps);
    __syncthreads();
    mpk_task_router_topk_warp_tiled(x_norm, router_weight, router_logits, indices, weights, b, hidden, experts, top_k);
    __syncthreads();
    mpk_task_routed_experts_warp_tiled(
        x_norm,
        expert_gate,
        expert_up,
        indices,
        weights,
        activations,
        b,
        hidden,
        top_k,
        intermediate);
    __syncthreads();
    mpk_task_shared_expert_warp_tiled(x_norm, shared_gate, shared_up, shared_activations, b, hidden, shared_intermediate);
    __syncthreads();
    mpk_task_moe_output_warp_tiled(
        x,
        activations,
        shared_activations,
        expert_down,
        shared_down,
        out,
        indices,
        b,
        hidden,
        top_k,
        intermediate,
        shared_intermediate,
        add_residual);
}

void check_moe_inputs(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    torch::Tensor expert_gate,
    torch::Tensor expert_up,
    torch::Tensor expert_down,
    torch::Tensor shared_gate,
    torch::Tensor shared_up,
    torch::Tensor shared_down,
    int64_t top_k) {
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
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_decode_persistent_cuda(
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
    check_moe_inputs(
        x,
        norm_weight,
        router_weight,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        top_k);

    int batch = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    int experts = static_cast<int>(router_weight.size(0));
    int intermediate = static_cast<int>(expert_gate.size(1));
    int shared_intermediate = static_cast<int>(shared_gate.size(0));

    auto float_opts = x.options().dtype(torch::kFloat32);
    auto index_opts = x.options().dtype(torch::kInt64);
    auto x_norm = torch::empty({batch, hidden}, float_opts);
    auto router_logits = torch::empty({batch, experts}, float_opts);
    auto indices = torch::empty({batch, top_k}, index_opts);
    auto weights = torch::empty({batch, top_k}, float_opts);
    auto activations = torch::empty({batch, top_k, intermediate}, float_opts);
    auto shared_activations = torch::empty({batch, shared_intermediate}, float_opts);
    auto out = torch::empty_like(x);

    constexpr int threads = 256;
    int target_workers_per_batch = batch == 1 ? 128 : (batch <= 4 ? 64 : 32);
    auto worker_scratch = torch::empty({batch, target_workers_per_batch}, float_opts);
    auto norm_scales = torch::empty({batch}, float_opts);
    size_t shared_bytes = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int device = x.get_device();
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int cooperative_launch = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "moe_decode_persistent_cuda", [&] {
        int active_blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            moe_decode_persistent_coop_kernel<scalar_t>,
            threads,
            shared_bytes));
        int max_cooperative_blocks = active_blocks_per_sm * prop.multiProcessorCount;
        int workers_per_batch = std::min(target_workers_per_batch, max_cooperative_blocks / std::max(batch, 1));

        if (cooperative_launch && workers_per_batch > 1) {
            const scalar_t* x_ptr = x.data_ptr<scalar_t>();
            const scalar_t* norm_weight_ptr = norm_weight.data_ptr<scalar_t>();
            const scalar_t* router_weight_ptr = router_weight.data_ptr<scalar_t>();
            const scalar_t* expert_gate_ptr = expert_gate.data_ptr<scalar_t>();
            const scalar_t* expert_up_ptr = expert_up.data_ptr<scalar_t>();
            const scalar_t* expert_down_ptr = expert_down.data_ptr<scalar_t>();
            const scalar_t* shared_gate_ptr = shared_gate.data_ptr<scalar_t>();
            const scalar_t* shared_up_ptr = shared_up.data_ptr<scalar_t>();
            const scalar_t* shared_down_ptr = shared_down.data_ptr<scalar_t>();
            float* x_norm_ptr = x_norm.data_ptr<float>();
            float* router_logits_ptr = router_logits.data_ptr<float>();
            float* worker_scratch_ptr = worker_scratch.data_ptr<float>();
            float* norm_scales_ptr = norm_scales.data_ptr<float>();
            int64_t* indices_ptr = indices.data_ptr<int64_t>();
            float* weights_ptr = weights.data_ptr<float>();
            float* activations_ptr = activations.data_ptr<float>();
            float* shared_activations_ptr = shared_activations.data_ptr<float>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            int batch_arg = batch;
            int hidden_arg = hidden;
            int experts_arg = experts;
            int top_k_arg = static_cast<int>(top_k);
            int intermediate_arg = intermediate;
            int shared_intermediate_arg = shared_intermediate;
            float eps_arg = static_cast<float>(eps);
            bool add_residual_arg = add_residual;
            int workers_per_batch_arg = workers_per_batch;
            void* args[] = {
                &x_ptr,
                &norm_weight_ptr,
                &router_weight_ptr,
                &expert_gate_ptr,
                &expert_up_ptr,
                &expert_down_ptr,
                &shared_gate_ptr,
                &shared_up_ptr,
                &shared_down_ptr,
                &x_norm_ptr,
                &router_logits_ptr,
                &worker_scratch_ptr,
                &norm_scales_ptr,
                &indices_ptr,
                &weights_ptr,
                &activations_ptr,
                &shared_activations_ptr,
                &out_ptr,
                &batch_arg,
                &hidden_arg,
                &experts_arg,
                &top_k_arg,
                &intermediate_arg,
                &shared_intermediate_arg,
                &eps_arg,
                &add_residual_arg,
                &workers_per_batch_arg,
            };
            C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(moe_decode_persistent_coop_kernel<scalar_t>),
                dim3(batch * workers_per_batch),
                dim3(threads),
                args,
                shared_bytes,
                stream));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            moe_decode_persistent_kernel<scalar_t><<<batch, threads, shared_bytes, stream>>>(
                x.data_ptr<scalar_t>(),
                norm_weight.data_ptr<scalar_t>(),
                router_weight.data_ptr<scalar_t>(),
                expert_gate.data_ptr<scalar_t>(),
                expert_up.data_ptr<scalar_t>(),
                expert_down.data_ptr<scalar_t>(),
                shared_gate.data_ptr<scalar_t>(),
                shared_up.data_ptr<scalar_t>(),
                shared_down.data_ptr<scalar_t>(),
                x_norm.data_ptr<float>(),
                router_logits.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                activations.data_ptr<float>(),
                shared_activations.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                batch,
                hidden,
                experts,
                static_cast<int>(top_k),
                intermediate,
                shared_intermediate,
                static_cast<float>(eps),
                add_residual);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    return std::make_tuple(out, indices, weights);
}
