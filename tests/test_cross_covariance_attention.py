from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.cross_covariance_attention import (
    SharedProjectionCrossCovarianceAttention,
)
from trkh.models.model import (
    CustomTransformerEncoderLayer,
    MultiHeadSelfAttention,
    create_model,
)
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=3,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="1,2",
        token_keep_rates="0.75,0.50",
        cross_covariance_attention=True,
        cross_covariance_attention_layers="1,3",
        cross_covariance_attention_residual_scale=0.10,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_xca_matches_explicit_channel_covariance_equation() -> None:
    torch.manual_seed(7)
    module = SharedProjectionCrossCovarianceAttention(dim=16, num_heads=4)
    qkv = nn.Linear(16, 48)
    projection = nn.Linear(16, 16)
    tokens = torch.randn(2, 9, 16)

    output, attention = module(
        tokens,
        qkv_projection=qkv,
        output_projection=projection,
        return_attention=True,
    )

    normalized = module.norm(tokens)
    explicit_qkv = qkv(normalized).reshape(2, 9, 3, 4, 4).permute(2, 0, 3, 4, 1)
    query, key, value = explicit_qkv.unbind(dim=0)
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    expected_attention = (
        (query @ key.transpose(-2, -1)) * module.temperature
    ).softmax(dim=-1)
    expected = expected_attention @ value
    expected = projection(expected.permute(0, 3, 1, 2).reshape(2, 9, 16))

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(attention, expected_attention)
    assert attention.shape == (2, 4, 4, 4)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
    )
    trace = module.trace()
    expected_entropy = -(
        attention.float().clamp_min(1e-12)
        * attention.float().clamp_min(1e-12).log()
    ).sum(dim=-1).mean() / math.log(4.0)
    torch.testing.assert_close(trace["normalized_entropy"], expected_entropy)
    assert float(trace["query_norm_max_error"]) < 1e-5
    assert float(trace["key_norm_max_error"]) < 1e-5


def test_xca_is_patch_only_and_all_parameter_families_receive_gradients() -> None:
    torch.manual_seed(11)
    block = CustomTransformerEncoderLayer(
        dim=32,
        num_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        cross_covariance_attention=True,
        cross_covariance_attention_residual_scale=0.10,
    )
    tokens = torch.randn(2, 13, 32, requires_grad=True)
    prefix = tokens[:, :4].detach().clone()

    updated, channel_attention = block.apply_cross_covariance_attention(
        tokens,
        prefix_count=4,
        return_attention=True,
    )

    torch.testing.assert_close(updated[:, :4], prefix, rtol=0.0, atol=0.0)
    assert channel_attention.shape == (2, 4, 8, 8)
    updated.square().mean().backward()
    required = {
        "attn.qkv.weight": block.attn.qkv.weight,
        "attn.proj.weight": block.attn.proj.weight,
        "xca.temperature": block.cross_covariance_attention.temperature,
        "xca.norm.weight": block.cross_covariance_attention.norm.weight,
        "xca.norm.bias": block.cross_covariance_attention.norm.bias,
        "input": tokens,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert int(torch.count_nonzero(parameter.grad).item()) > 0, name


def test_model_wiring_keeps_spatial_attention_and_supports_pruning_trace() -> None:
    torch.manual_seed(13)
    model = create_model(num_classes=5, model_config=_tiny_config()).eval()
    assert all(isinstance(block.attn, MultiHeadSelfAttention) for block in model.blocks)
    assert [
        index + 1
        for index, block in enumerate(model.blocks)
        if block.cross_covariance_attention is not None
    ] == [1, 3]
    images = torch.randn(2, 3, 32, 32)

    with torch.inference_mode():
        spatial_features = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
        pruned_features = model.forward_features(images, return_trace=True)

    assert set(spatial_features["attentions"]) == {0, 1, 2}
    assert spatial_features["attention_representation"] == "mhsa_probability"
    for attention in spatial_features["attentions"].values():
        assert attention.shape == (2, 4, 19, 19)
    trace = spatial_features["trace"]
    assert trace["cross_covariance_attention_layers"].tolist() == [1, 3]
    assert trace["cross_covariance_temperature_mean"].shape == (2,)
    assert trace["cross_covariance_normalized_entropy"].shape == (2,)
    assert trace["cross_covariance_diagonal_mass"].shape == (2,)
    assert trace["cross_covariance_residual_norm_ratio"].shape == (2,)
    assert pruned_features["patches"].shape[1] == 8
    assert [int(item["layer"]) for item in pruned_features["trace"]["pruning"]] == [1, 2]


def test_xca_checkpoint_extension_reuses_qkv_proj_and_has_exact_missing_keys() -> None:
    torch.manual_seed(17)
    source = create_model(
        num_classes=5,
        model_config=_tiny_config(cross_covariance_attention=False),
    )
    torch.manual_seed(17)
    target = create_model(num_classes=5, model_config=_tiny_config())

    summary = _load_model_state_allowing_extensions(
        target,
        source.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    assert summary["unexpected_keys"] == []
    assert set(summary["allowed_missing_keys"]) == {
        "blocks.0.cross_covariance_attention.temperature",
        "blocks.0.cross_covariance_attention.norm.weight",
        "blocks.0.cross_covariance_attention.norm.bias",
        "blocks.2.cross_covariance_attention.temperature",
        "blocks.2.cross_covariance_attention.norm.weight",
        "blocks.2.cross_covariance_attention.norm.bias",
    }
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        torch.testing.assert_close(target_state[key], value, rtol=0.0, atol=0.0)
    assert not any(
        ".cross_covariance_attention.qkv" in key
        or ".cross_covariance_attention.proj" in key
        for key in target_state
    )


def test_xca_constructor_and_forward_do_not_advance_rng_streams() -> None:
    seed = 23
    torch.manual_seed(seed)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(cross_covariance_attention=False),
    ).eval()
    control_constructor_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = create_model(num_classes=5, model_config=_tiny_config()).eval()
    candidate_constructor_rng = torch.get_rng_state().clone()
    assert torch.equal(control_constructor_rng, candidate_constructor_rng)

    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(29)
    with torch.inference_mode():
        control(images)
    control_forward_rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    with torch.inference_mode():
        candidate(images)
    candidate_forward_rng = torch.get_rng_state().clone()
    assert torch.equal(control_forward_rng, candidate_forward_rng)


def test_xca_layer_ablation_changes_logits() -> None:
    torch.manual_seed(31)
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(token_pruning=False),
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        baseline = model(images)
        original_scale = model.blocks[0].cross_covariance_attention_residual_scale
        model.blocks[0].cross_covariance_attention_residual_scale = 0.0
        ablated = model(images)
        model.blocks[0].cross_covariance_attention_residual_scale = original_scale
    assert float((baseline - ablated).abs().amax()) >= 1e-6


def test_xca_defaults_conflicts_and_launcher_controls() -> None:
    defaults = ModelConfig()
    assert defaults.cross_covariance_attention is False
    assert defaults.cross_covariance_attention_layers == "2,5"
    assert defaults.cross_covariance_attention_residual_scale == pytest.approx(0.10)
    with pytest.raises(ValueError, match="Visual-Contrast Attention"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(
                visual_contrast_attention=True,
                visual_contrast_attention_layers="1,2,3",
                visual_contrast_tokens=4,
                token_pruning=False,
            ),
        )
    with pytest.raises(ValueError, match=">= 0"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(
                cross_covariance_attention_residual_scale=-0.01,
            ),
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$CrossCovarianceAttention = $false" in script
    assert "[string]$CrossCovarianceAttentionLayers = \"2,5\"" in script
    assert '"--cross-covariance-attention"' in script
    assert '"--cross-covariance-attention-layers"' in script
    assert '"--cross-covariance-attention-residual-scale"' in script
