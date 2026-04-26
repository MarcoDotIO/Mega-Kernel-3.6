#include <torch/extension.h>

#include <tuple>

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
    bool add_residual);

torch::Tensor full_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    double scale);

std::tuple<torch::Tensor, torch::Tensor> linear_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    double decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_decode", &moe_decode_cuda, "Qwen3.6 MoE decode kernel");
    m.def("full_attention_decode", &full_attention_decode_cuda, "Qwen3.6 full-attention decode kernel");
    m.def("linear_attention_decode", &linear_attention_decode_cuda, "Qwen3.6 linear-attention recurrent decode kernel");
}
