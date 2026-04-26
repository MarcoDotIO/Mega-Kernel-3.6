import pytest
import torch

import mega_kernel_qwen36 as mk
from mega_kernel_qwen36.reference import (
    vision_make_cu_seqlens,
    vision_rotary_position_embeddings_reference,
)


def _require_transformers_parity(request):
    if not request.config.getoption("--run-transformers-parity"):
        pytest.skip("pass --run-transformers-parity to run Hugging Face Transformers parity tests")


def _hf_vision_config(depth=1):
    cfg = mk.qwen36_27b_vision_config()
    hf_config_mod = pytest.importorskip("transformers.models.qwen3_vl.configuration_qwen3_vl")
    hf_cfg = hf_config_mod.Qwen3VLVisionConfig(
        depth=depth,
        hidden_size=cfg.hidden_size,
        hidden_act=cfg.hidden_act,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        in_channels=cfg.in_channels,
        patch_size=cfg.patch_size,
        spatial_merge_size=cfg.spatial_merge_size,
        temporal_patch_size=cfg.temporal_patch_size,
        out_hidden_size=cfg.out_hidden_size,
        num_position_embeddings=cfg.num_position_embeddings,
        deepstack_visual_indexes=list(cfg.deepstack_visual_indexes),
    )
    hf_cfg._attn_implementation = "eager"
    return cfg, hf_cfg


def _block_weights(block):
    return mk.VisionBlockWeights(
        norm1_weight=block.norm1.weight.detach().clone(),
        norm1_bias=block.norm1.bias.detach().clone(),
        attention=mk.VisionAttentionWeights(
            qkv_weight=block.attn.qkv.weight.detach().clone(),
            qkv_bias=block.attn.qkv.bias.detach().clone(),
            proj_weight=block.attn.proj.weight.detach().clone(),
            proj_bias=block.attn.proj.bias.detach().clone(),
        ),
        norm2_weight=block.norm2.weight.detach().clone(),
        norm2_bias=block.norm2.bias.detach().clone(),
        mlp=mk.VisionMlpWeights(
            fc1_weight=block.mlp.linear_fc1.weight.detach().clone(),
            fc1_bias=block.mlp.linear_fc1.bias.detach().clone(),
            fc2_weight=block.mlp.linear_fc2.weight.detach().clone(),
            fc2_bias=block.mlp.linear_fc2.bias.detach().clone(),
        ),
    )


def _merger_weights(merger):
    return mk.VisionPatchMergerWeights(
        norm_weight=merger.norm.weight.detach().clone(),
        norm_bias=merger.norm.bias.detach().clone(),
        fc1_weight=merger.linear_fc1.weight.detach().clone(),
        fc1_bias=merger.linear_fc1.bias.detach().clone(),
        fc2_weight=merger.linear_fc2.weight.detach().clone(),
        fc2_bias=merger.linear_fc2.bias.detach().clone(),
    )


def _encoder_weights(model):
    return mk.VisionEncoderWeights(
        patch_embed=mk.VisionPatchEmbedWeights(
            proj_weight=model.patch_embed.proj.weight.detach().clone(),
            proj_bias=model.patch_embed.proj.bias.detach().clone(),
        ),
        pos_embed_weight=model.pos_embed.weight.detach().clone(),
        blocks=tuple(_block_weights(block) for block in model.blocks),
        merger=_merger_weights(model.merger),
        deepstack_mergers=tuple(_merger_weights(merger) for merger in model.deepstack_merger_list),
    )


def test_transformers_vision_block_matches_mega_kernel_reference(request):
    _require_transformers_parity(request)
    cfg, hf_cfg = _hf_vision_config(depth=1)
    modeling = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    torch.manual_seed(401)
    block = modeling.Qwen3VLVisionBlock(hf_cfg).eval()
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    hidden = torch.randn(16, cfg.hidden_size) * 0.02
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cu_seqlens = vision_make_cu_seqlens(grid_thw)

    with torch.no_grad():
        hf_out = block(hidden.clone(), cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
        mk_out = mk.vision_encoder_block(
            hidden,
            _block_weights(block),
            cu_seqlens,
            cos,
            sin,
            num_heads=cfg.num_heads,
            hidden_act=cfg.hidden_act,
            backend="reference",
        )
    torch.testing.assert_close(mk_out, hf_out, atol=1e-5, rtol=1e-4)


def test_transformers_vision_model_matches_mega_kernel_reference(request):
    _require_transformers_parity(request)
    cfg, hf_cfg = _hf_vision_config(depth=1)
    modeling = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    torch.manual_seed(409)
    model = modeling.Qwen3VLVisionModel(hf_cfg).eval()
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixel_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = torch.randn(16, pixel_dim) * 0.02

    with torch.no_grad():
        hf_out = model(pixel_values, grid_thw)
        mk_out = mk.vision_encode(pixel_values, grid_thw, _encoder_weights(model), config=cfg, backend="reference")
    assert mk_out.deepstack_features == tuple(hf_out.deepstack_features)
    torch.testing.assert_close(mk_out.hidden, hf_out.pooler_output, atol=1e-5, rtol=1e-4)
