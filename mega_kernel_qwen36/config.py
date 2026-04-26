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


@dataclass(frozen=True)
class Qwen36VisionConfig:
    """Official Qwen/Qwen3.6-27B vision encoder dimensions."""

    model_name: str = "Qwen/Qwen3.6-27B"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 5120
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * (self.spatial_merge_size**2)


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


def qwen36_27b_vision_config() -> Qwen36VisionConfig:
    """Official vision encoder dimensions for Qwen/Qwen3.6-27B."""

    return Qwen36VisionConfig()
