import mega_kernel_qwen36 as mk


def test_qwen36_35b_a3b_config_values():
    cfg = mk.qwen36_35b_a3b_config()
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 40
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 512
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 256
    assert cfg.rotary_dim == 64
    assert cfg.max_position_embeddings == 262144
    assert cfg.layer_types[:8] == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )


def test_qwen36_27b_config_values():
    cfg = mk.qwen36_27b_config()
    assert cfg.model_name == "Qwen/Qwen3.6-27B"
    assert not cfg.is_moe
    assert cfg.hidden_size == 5120
    assert cfg.num_hidden_layers == 64
    assert cfg.intermediate_size == 17408
    assert cfg.num_attention_heads == 24
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 256
    assert cfg.rotary_dim == 64
    assert cfg.linear_num_value_heads == 48
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[4] == "linear_attention"


def test_qwen36_27b_vision_config_values():
    cfg = mk.qwen36_27b_vision_config()
    assert cfg.model_name == "Qwen/Qwen3.6-27B"
    assert cfg.depth == 27
    assert cfg.hidden_size == 1152
    assert cfg.intermediate_size == 4304
    assert cfg.num_heads == 16
    assert cfg.head_dim == 72
    assert cfg.patch_size == 16
    assert cfg.temporal_patch_size == 2
    assert cfg.spatial_merge_size == 2
    assert cfg.merged_hidden_size == 4608
    assert cfg.out_hidden_size == 5120
    assert cfg.num_position_embeddings == 2304
    assert cfg.deepstack_visual_indexes == ()
