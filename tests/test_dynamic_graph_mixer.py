from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.dynamic_graph_mixer import MaxRelativeDynamicGraphMixer
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
        dynamic_graph_mixer=True,
        dynamic_graph_mixer_layers="1,3",
        dynamic_graph_mixer_bottleneck_dim=8,
        dynamic_graph_mixer_k=3,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _activate_output(module: MaxRelativeDynamicGraphMixer) -> None:
    with torch.no_grad():
        values = torch.linspace(
            -0.03,
            0.03,
            module.output_projection.weight.numel(),
        ).reshape_as(module.output_projection.weight)
        module.output_projection.weight.copy_(values)
        module.output_projection.bias.copy_(
            torch.linspace(-0.01, 0.01, module.dim)
        )


def test_dynamic_graph_matches_explicit_max_relative_equation() -> None:
    torch.manual_seed(7)
    module = MaxRelativeDynamicGraphMixer(dim=16, bottleneck_dim=4, k=3)
    _activate_output(module)
    tokens = torch.randn(2, 9, 16)

    residual, details = module(tokens, return_details=True, grid_size=(3, 3))

    normalized = module.norm(tokens)
    projected = F.linear(normalized, module.input_projection.weight)
    graph_nodes = F.normalize(projected.detach().float(), p=2.0, dim=-1)
    squared_norm = graph_nodes.square().sum(dim=-1, keepdim=True)
    distance = (
        squared_norm
        - 2.0 * graph_nodes @ graph_nodes.transpose(1, 2)
        + squared_norm.transpose(1, 2)
    ).clamp_min(0.0)
    expected_distance, expected_indices = torch.topk(
        -distance,
        k=3,
        dim=-1,
        largest=True,
        sorted=True,
    )
    expected_neighbors = projected.gather(
        1,
        expected_indices.reshape(2, -1).unsqueeze(-1).expand(-1, -1, 4),
    ).reshape(2, 9, 3, 4)
    expected_message = (expected_neighbors - projected.unsqueeze(2)).amax(dim=2)
    expected_residual = F.gelu(
        F.linear(
            torch.cat((projected, expected_message), dim=-1),
            module.output_projection.weight,
            module.output_projection.bias,
        )
    )

    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(details["projected_nodes"], projected)
    torch.testing.assert_close(details["neighbor_indices"], expected_indices)
    torch.testing.assert_close(details["neighbor_distances"], -expected_distance)
    torch.testing.assert_close(details["max_relative_message"], expected_message)
    center = torch.arange(9).view(1, 9, 1)
    assert torch.all((expected_indices == center).any(dim=-1))
    trace = module.trace()
    assert float(trace["nonself_neighbor_count"]) == pytest.approx(2.0)
    assert 0.0 <= float(trace["nonlocal_neighbor_fraction"]) <= 1.0
    assert 0.0 <= float(trace["neighbor_selection_entropy"]) <= 1.0


def test_graph_zero_init_is_exact_patch_only_and_parameter_count_is_locked() -> None:
    module = MaxRelativeDynamicGraphMixer(dim=256, bottleneck_dim=64, k=9)
    assert sum(parameter.numel() for parameter in module.parameters()) == 49_920
    tokens = torch.randn(2, 13, 256)
    block = CustomTransformerEncoderLayer(
        dim=256,
        num_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        dynamic_graph_mixer=True,
        dynamic_graph_mixer_bottleneck_dim=64,
        dynamic_graph_mixer_k=9,
    )
    prefix = tokens[:, :4].clone()

    updated = block.apply_dynamic_graph_mixer(
        tokens,
        prefix_count=4,
        grid_size=(3, 3),
        patch_indices=None,
    )

    torch.testing.assert_close(updated, tokens, rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated[:, :4], prefix, rtol=0.0, atol=0.0)


def test_dynamic_graph_all_parameter_families_receive_gradients_when_active() -> None:
    torch.manual_seed(11)
    module = MaxRelativeDynamicGraphMixer(dim=32, bottleneck_dim=8, k=3)
    _activate_output(module)
    tokens = torch.randn(2, 16, 32, requires_grad=True)

    residual = module(tokens, grid_size=(4, 4))
    residual.square().mean().backward()

    required = {
        "norm.weight": module.norm.weight,
        "norm.bias": module.norm.bias,
        "input_projection.weight": module.input_projection.weight,
        "output_projection.weight": module.output_projection.weight,
        "output_projection.bias": module.output_projection.bias,
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
        if block.dynamic_graph_mixer is not None
    ] == [1, 3]
    for block in model.blocks:
        if block.dynamic_graph_mixer is not None:
            _activate_output(block.dynamic_graph_mixer)
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
    trace = spatial_features["trace"]
    assert trace["dynamic_graph_mixer_layers"].tolist() == [1, 3]
    for key in (
        "residual_norm_ratio",
        "projected_node_norm",
        "neighbor_distance_mean",
        "nonself_neighbor_count",
        "nonlocal_neighbor_fraction",
        "neighbor_selection_entropy",
        "patch_count",
        "neighbor_count",
    ):
        assert trace[f"dynamic_graph_{key}"].shape == (2,)
    assert pruned_features["patches"].shape[1] == 8
    assert [int(item["layer"]) for item in pruned_features["trace"]["pruning"]] == [1, 2]


def test_graph_checkpoint_extension_has_exact_missing_keys_and_existing_state() -> None:
    torch.manual_seed(17)
    source = create_model(
        num_classes=5,
        model_config=_tiny_config(dynamic_graph_mixer=False),
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
    expected_suffixes = {
        "norm.weight",
        "norm.bias",
        "input_projection.weight",
        "output_projection.weight",
        "output_projection.bias",
    }
    assert set(summary["allowed_missing_keys"]) == {
        f"blocks.{block}.dynamic_graph_mixer.{suffix}"
        for block in (0, 2)
        for suffix in expected_suffixes
    }
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        torch.testing.assert_close(target_state[key], value, rtol=0.0, atol=0.0)


def test_graph_constructor_forward_rng_and_zero_init_model_are_exact() -> None:
    seed = 23
    torch.manual_seed(seed)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(dynamic_graph_mixer=False),
    ).eval()
    control_constructor_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = create_model(num_classes=5, model_config=_tiny_config()).eval()
    candidate_constructor_rng = torch.get_rng_state().clone()
    assert torch.equal(control_constructor_rng, candidate_constructor_rng)

    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(29)
    with torch.inference_mode():
        control_logits = control(images)
    control_forward_rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    with torch.inference_mode():
        candidate_logits = candidate(images)
    candidate_forward_rng = torch.get_rng_state().clone()
    assert torch.equal(control_forward_rng, candidate_forward_rng)
    torch.testing.assert_close(candidate_logits, control_logits, rtol=0.0, atol=0.0)


def test_graph_defaults_conflicts_and_launcher_controls() -> None:
    defaults = ModelConfig()
    assert defaults.dynamic_graph_mixer is False
    assert defaults.dynamic_graph_mixer_layers == "2,5"
    assert defaults.dynamic_graph_mixer_bottleneck_dim == 64
    assert defaults.dynamic_graph_mixer_k == 9
    with pytest.raises(ValueError, match="cross-covariance attention"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(
                cross_covariance_attention=True,
                cross_covariance_attention_layers="1,3",
            ),
        )
    with pytest.raises(ValueError, match="divide embed_dim"):
        create_model(
            num_classes=5,
            model_config=_tiny_config(dynamic_graph_mixer_bottleneck_dim=7),
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$DynamicGraphMixer = $false" in script
    assert "[string]$DynamicGraphMixerLayers = \"2,5\"" in script
    assert "[int]$DynamicGraphMixerBottleneckDim = 64" in script
    assert "[int]$DynamicGraphMixerK = 9" in script
    assert '"--dynamic-graph-mixer"' in script
    assert '"--dynamic-graph-mixer-layers"' in script
    assert '"--dynamic-graph-mixer-bottleneck-dim"' in script
    assert '"--dynamic-graph-mixer-k"' in script


def test_projection_initialization_matches_locked_group_average() -> None:
    module = MaxRelativeDynamicGraphMixer(dim=16, bottleneck_dim=4, k=3)
    expected = torch.zeros(4, 16)
    for row in range(4):
        expected[row, row * 4 : (row + 1) * 4] = 0.5
    torch.testing.assert_close(module.input_projection.weight, expected)
    assert torch.count_nonzero(module.output_projection.weight) == 0
    assert torch.count_nonzero(module.output_projection.bias) == 0
    assert math.isclose(float(expected.square().sum(dim=1).mean()), 1.0)
