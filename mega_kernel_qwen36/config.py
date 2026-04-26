from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen36TextConfig:
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    is_moe: bool = True
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: int = 10_000_000
    full_attention_interval: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    intermediate_size: int = 0
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(qwen36_layer_type(i) for i in range(self.num_hidden_layers))


def qwen36_layer_type(layer_idx: int) -> str:
    """Return the Qwen3.6 text layer kind for a zero-based layer index."""

    return "full_attention" if layer_idx % 4 == 3 else "linear_attention"


def qwen36_35b_a3b_config() -> Qwen36TextConfig:
    """Official text-path dimensions for Qwen/Qwen3.6-35B-A3B."""

    return Qwen36TextConfig()


def qwen36_27b_config() -> Qwen36TextConfig:
    """Official text-path dimensions for Qwen/Qwen3.6-27B."""

    return Qwen36TextConfig(
        model_name="Qwen/Qwen3.6-27B",
        is_moe=False,
        hidden_size=5120,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        linear_num_value_heads=48,
        intermediate_size=17408,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
    )
