import torch

import mega_kernel_qwen36 as mk
from mega_kernel_qwen36.reference import (
    dense_ffn_decode_reference,
    full_attention_decode_reference,
    linear_attention_decode_reference,
    moe_decode_reference,
    vision_attention_encode_reference,
    vision_fast_pos_embed_interpolate_reference,
    vision_make_cu_seqlens,
    vision_patch_embed_reference,
    vision_patch_merger_reference,
    vision_rotary_position_embeddings_reference,
)


def _small_moe(device="cpu", dtype=torch.float32):
    torch.manual_seed(7)
    batch, hidden, experts, intermediate, top_k = 2, 16, 6, 12, 3
    scale = 0.08
    x = (torch.randn(batch, hidden, device=device) * scale).to(dtype)
    weights = mk.MoeWeights(
        norm_weight=(torch.randn(hidden, device=device) * scale + 1.0).to(dtype),
        router_weight=(torch.randn(experts, hidden, device=device) * scale).to(dtype),
        expert_gate=(torch.randn(experts, intermediate, hidden, device=device) * scale).to(dtype),
        expert_up=(torch.randn(experts, intermediate, hidden, device=device) * scale).to(dtype),
        expert_down=(torch.randn(experts, hidden, intermediate, device=device) * scale).to(dtype),
        shared_gate=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        shared_up=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        shared_down=(torch.randn(hidden, intermediate, device=device) * scale).to(dtype),
    )
    return x, weights, top_k


def _small_dense(device="cpu", dtype=torch.float32):
    torch.manual_seed(23)
    batch, hidden, intermediate = 2, 16, 24
    scale = 0.08
    x = (torch.randn(batch, hidden, device=device) * scale).to(dtype)
    weights = mk.DenseFfnWeights(
        norm_weight=(torch.randn(hidden, device=device) * scale + 1.0).to(dtype),
        gate_weight=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        up_weight=(torch.randn(intermediate, hidden, device=device) * scale).to(dtype),
        down_weight=(torch.randn(hidden, intermediate, device=device) * scale).to(dtype),
    )
    return x, weights


def _small_vision(device="cpu", dtype=torch.float32, layers=2):
    torch.manual_seed(31)
    cfg = mk.Qwen36VisionConfig(
        depth=layers,
        hidden_size=32,
        intermediate_size=48,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    scale = 0.04
    grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    pixel_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = (torch.randn(16, pixel_dim, device=device) * scale).to(dtype)

    def bias(size):
        return (torch.randn(size, device=device) * scale).to(dtype)

    def weight(*shape):
        return (torch.randn(*shape, device=device) * scale).to(dtype)

    def merger(postshuffle_norm=False):
        norm_hidden = cfg.merged_hidden_size if postshuffle_norm else cfg.hidden_size
        return mk.VisionPatchMergerWeights(
            norm_weight=(torch.randn(norm_hidden, device=device) * scale + 1.0).to(dtype),
            norm_bias=bias(norm_hidden),
            fc1_weight=weight(cfg.merged_hidden_size, cfg.merged_hidden_size),
            fc1_bias=bias(cfg.merged_hidden_size),
            fc2_weight=weight(cfg.out_hidden_size, cfg.merged_hidden_size),
            fc2_bias=bias(cfg.out_hidden_size),
        )

    blocks = []
    for _ in range(layers):
        blocks.append(
            mk.VisionBlockWeights(
                norm1_weight=(torch.randn(cfg.hidden_size, device=device) * scale + 1.0).to(dtype),
                norm1_bias=bias(cfg.hidden_size),
                attention=mk.VisionAttentionWeights(
                    qkv_weight=weight(cfg.hidden_size * 3, cfg.hidden_size),
                    qkv_bias=bias(cfg.hidden_size * 3),
                    proj_weight=weight(cfg.hidden_size, cfg.hidden_size),
                    proj_bias=bias(cfg.hidden_size),
                ),
                norm2_weight=(torch.randn(cfg.hidden_size, device=device) * scale + 1.0).to(dtype),
                norm2_bias=bias(cfg.hidden_size),
                mlp=mk.VisionMlpWeights(
                    fc1_weight=weight(cfg.intermediate_size, cfg.hidden_size),
                    fc1_bias=bias(cfg.intermediate_size),
                    fc2_weight=weight(cfg.hidden_size, cfg.intermediate_size),
                    fc2_bias=bias(cfg.hidden_size),
                ),
            )
        )

    weights = mk.VisionEncoderWeights(
        patch_embed=mk.VisionPatchEmbedWeights(
            proj_weight=weight(cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size),
            proj_bias=bias(cfg.hidden_size),
        ),
        pos_embed_weight=weight(cfg.num_position_embeddings, cfg.hidden_size),
        blocks=tuple(blocks),
        merger=merger(),
    )
    return cfg, grid_thw, pixel_values, weights


def test_moe_reference_shapes_and_topk():
    x, weights, top_k = _small_moe()
    out, idx, router_weights = moe_decode_reference(
        x,
        weights.norm_weight,
        weights.router_weight,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
        weights.shared_gate,
        weights.shared_up,
        weights.shared_down,
        top_k=top_k,
    )
    assert out.shape == x.shape
    assert idx.shape == (x.shape[0], top_k)
    assert router_weights.shape == (x.shape[0], top_k)
    torch.testing.assert_close(router_weights.sum(dim=-1), torch.ones(x.shape[0]))


def test_moe_grouped_backend_matches_reference_cpu():
    x, weights, top_k = _small_moe()
    out, idx, router_weights = mk.moe_decode(x, weights, top_k=top_k, backend="grouped")
    ref, ref_idx, ref_router_weights = mk.moe_decode(x, weights, top_k=top_k, backend="reference")
    assert idx.tolist() == ref_idx.tolist()
    torch.testing.assert_close(router_weights, ref_router_weights)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_dense_ffn_reference_shapes():
    x, weights = _small_dense()
    out = dense_ffn_decode_reference(
        x,
        weights.norm_weight,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
    )
    assert out.shape == x.shape


def test_dense_ffn_torch_backend_matches_reference_cpu():
    x, weights = _small_dense()
    out = mk.dense_ffn_decode(x, weights, backend="torch")
    ref = mk.dense_ffn_decode(x, weights, backend="reference")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_reference_shapes():
    torch.manual_seed(11)
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 9, 2, 16)
    v = torch.randn(2, 9, 2, 16)
    out = full_attention_decode_reference(q, k, v)
    assert out.shape == q.shape


def test_linear_attention_reference_shapes():
    torch.manual_seed(13)
    q = torch.randn(2, 3, 5)
    k = torch.randn(2, 3, 5)
    v = torch.randn(2, 3, 7)
    state = torch.randn(2, 3, 5, 7)
    out, new_state = linear_attention_decode_reference(q, k, v, state, decay=0.9)
    assert out.shape == v.shape
    assert new_state.shape == state.shape


def test_decode_layer_dispatch_reference_linear():
    x, weights, top_k = _small_moe()
    q = torch.randn(2, 4, 4)
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    state = torch.zeros(2, 4, 4, 4)
    result = mk.decode_layer(
        0,
        x,
        weights,
        linear_attention=mk.LinearAttentionInputs(q=q, k=k, v=v, state=state),
        top_k=top_k,
    )
    assert result.hidden.shape == x.shape
    assert result.linear_state is not None


def test_decode_layer_dispatch_reference_dense():
    x, weights = _small_dense()
    q = torch.randn(2, 4, 4)
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    state = torch.zeros(2, 4, 4, 4)
    result = mk.decode_layer(
        0,
        x,
        dense_ffn=weights,
        linear_attention=mk.LinearAttentionInputs(q=q, k=k, v=v, state=state),
    )
    assert result.hidden.shape == x.shape
    assert result.router_indices is None


def test_vision_reference_helpers_shapes():
    cfg, grid_thw, pixel_values, weights = _small_vision()
    patch = vision_patch_embed_reference(
        pixel_values,
        weights.patch_embed.proj_weight,
        weights.patch_embed.proj_bias,
        in_channels=cfg.in_channels,
        temporal_patch_size=cfg.temporal_patch_size,
        patch_size=cfg.patch_size,
    )
    pos = vision_fast_pos_embed_interpolate_reference(
        grid_thw,
        weights.pos_embed_weight,
        spatial_merge_size=cfg.spatial_merge_size,
        num_position_embeddings=cfg.num_position_embeddings,
    )
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cu = vision_make_cu_seqlens(grid_thw)
    assert patch.shape == (16, cfg.hidden_size)
    assert pos.shape == patch.shape
    assert cos.shape == (16, cfg.head_dim)
    assert sin.shape == (16, cfg.head_dim)
    assert cu.tolist() == [0, 16]


def test_vision_attention_reference_shapes():
    cfg, grid_thw, pixel_values, weights = _small_vision()
    hidden = mk.vision_patch_embed(pixel_values, weights.patch_embed, config=cfg, backend="reference")
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    out = vision_attention_encode_reference(
        hidden,
        weights.blocks[0].attention.qkv_weight,
        weights.blocks[0].attention.qkv_bias,
        weights.blocks[0].attention.proj_weight,
        weights.blocks[0].attention.proj_bias,
        vision_make_cu_seqlens(grid_thw),
        cos,
        sin,
        num_heads=cfg.num_heads,
    )
    assert out.shape == hidden.shape


def test_vision_attention_sdpa_matches_reference_cpu():
    cfg, grid_thw, pixel_values, weights = _small_vision()
    hidden = mk.vision_patch_embed(pixel_values, weights.patch_embed, config=cfg, backend="reference")
    cos, sin = vision_rotary_position_embeddings_reference(
        grid_thw,
        head_dim=cfg.head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    cu = vision_make_cu_seqlens(grid_thw)
    out = mk.vision_attention_encode(hidden, weights.blocks[0].attention, cu, cos, sin, num_heads=cfg.num_heads)
    ref = mk.vision_attention_encode(
        hidden,
        weights.blocks[0].attention,
        cu,
        cos,
        sin,
        num_heads=cfg.num_heads,
        backend="reference",
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_vision_patch_merger_reference_shapes():
    cfg, _, _, weights = _small_vision()
    hidden = torch.randn(16, cfg.hidden_size)
    out = vision_patch_merger_reference(
        hidden,
        weights.merger.norm_weight,
        weights.merger.norm_bias,
        weights.merger.fc1_weight,
        weights.merger.fc1_bias,
        weights.merger.fc2_weight,
        weights.merger.fc2_bias,
    )
    assert out.shape == (4, cfg.out_hidden_size)


def test_vision_encode_matches_reference_cpu():
    cfg, grid_thw, pixel_values, weights = _small_vision()
    out = mk.vision_encode(pixel_values, grid_thw, weights, config=cfg)
    ref = mk.vision_encode(pixel_values, grid_thw, weights, config=cfg, backend="reference")
    assert out.hidden.shape == (4, cfg.out_hidden_size)
    assert out.deepstack_features == ()
    torch.testing.assert_close(out.hidden, ref.hidden, atol=1e-5, rtol=1e-4)
