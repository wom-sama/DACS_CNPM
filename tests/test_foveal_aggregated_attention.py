from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.foveal_aggregated_attention import FovealAggregatedAttention
from trkh.models.model import MultiHeadSelfAttention, create_model


OFFICIAL_ATTENTION_PATH = Path(
    r"D:\DataAI\external_sources\official\transnext-cvpr2024"
) / "classification" / "attention_native.py"


def _tiny_model_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="1",
        token_keep_rates="0.5",
        foveal_aggregated_attention=True,
        foveal_aggregated_attention_layers="1",
        foveal_aggregated_attention_window_size=3,
        foveal_aggregated_attention_pool_size=2,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    return inputs, indices


def _load_official_attention_class():
    if not OFFICIAL_ATTENTION_PATH.is_file():
        pytest.skip(f"Locked official TransNeXt source is absent: {OFFICIAL_ATTENTION_PATH}")
    spec = importlib.util.spec_from_file_location(
        "locked_transnext_attention_native",
        OFFICIAL_ATTENTION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked official TransNeXt attention source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AggregatedAttention


def _copy_candidate_to_official(
    candidate: FovealAggregatedAttention,
    official: torch.nn.Module,
) -> None:
    dim = candidate.dim
    with torch.no_grad():
        official.q.weight.copy_(candidate.qkv.weight[:dim])
        official.q.bias.copy_(candidate.qkv.bias[:dim])
        official.kv.weight.copy_(candidate.qkv.weight[dim:])
        official.kv.bias.copy_(candidate.qkv.bias[dim:])
        official.proj.load_state_dict(candidate.proj.state_dict())
        official.temperature.copy_(candidate.temperature)
        official.query_embedding.copy_(candidate.query_embedding)
        official.sr.load_state_dict(candidate.sr.state_dict())
        official.norm.load_state_dict(candidate.pool_norm.state_dict())
        official.cpb_fc1.load_state_dict(candidate.cpb_fc1.state_dict())
        official.cpb_fc2.load_state_dict(candidate.cpb_fc2.state_dict())
        official.relative_pos_bias_local.copy_(candidate.relative_position_bias_local)
        official.learnable_tokens.copy_(candidate.learnable_tokens)
        official.learnable_bias.copy_(candidate.learnable_bias)


def test_faa_patch_only_equation_and_input_gradient_match_official_native() -> None:
    official_class = _load_official_attention_class()
    torch.manual_seed(41)
    candidate = FovealAggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    ).eval()
    official = official_class(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    ).eval()
    _copy_candidate_to_official(candidate, official)

    torch.manual_seed(19)
    candidate_input = torch.randn(2, 16, 32, requires_grad=True)
    official_input = candidate_input.detach().clone().requires_grad_(True)
    candidate_output = candidate(
        candidate_input,
        grid_size=(4, 4),
        prefix_count=0,
    )
    official_output = official(
        official_input,
        4,
        4,
        candidate.relative_position_index,
        candidate.relative_coords_table,
    )
    torch.testing.assert_close(candidate_output, official_output, atol=1e-6, rtol=1e-6)
    candidate_output.square().mean().backward()
    official_output.square().mean().backward()
    torch.testing.assert_close(
        candidate_input.grad,
        official_input.grad,
        atol=1e-6,
        rtol=1e-6,
    )


def test_faa_prefix_shape_dense_attention_and_route_trace() -> None:
    inputs, indices = _inputs()
    module = FovealAggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    ).eval()
    output, attention = module(
        inputs,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    assert output.shape == inputs.shape
    assert attention.shape == (2, 4, 21, 21)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention).all()
    assert bool((attention >= 0).all())
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )
    assert int(torch.count_nonzero(attention[:, :, 5:, :5]).item()) == 0
    trace = module.trace()
    assert trace["local_mass_map"].shape == (2, 16)
    assert trace["pooled_mass_map"].shape == (2, 16)
    torch.testing.assert_close(
        trace["local_mass_map"] + trace["pooled_mass_map"],
        torch.ones_like(trace["local_mass_map"]),
        atol=1e-6,
        rtol=0.0,
    )
    assert 0.0 < float(trace["local_mass_mean"]) < 1.0
    assert 0.0 < float(trace["pooled_mass_mean"]) < 1.0
    assert float(trace["local_mass_mean"] + trace["pooled_mass_mean"]) == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert 0.0 <= float(trace["dual_route_fraction"]) <= 1.0


def test_faa_component_families_receive_finite_nonzero_gradients() -> None:
    inputs, indices = _inputs()
    module = FovealAggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    )
    output = module(
        inputs,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    output.square().mean().backward()
    parameters = dict(module.named_parameters())
    required = {
        "qkv.weight",
        "proj.weight",
        "temperature",
        "query_embedding",
        "sr.weight",
        "pool_norm.weight",
        "cpb_fc1.weight",
        "cpb_fc2.weight",
        "relative_position_bias_local",
        "learnable_tokens",
        "learnable_bias",
    }
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient).item()) > 0, name


def test_faa_control_candidate_common_initial_state_is_bit_exact() -> None:
    torch.manual_seed(42)
    control = create_model(
        num_classes=5,
        model_config=_tiny_model_config(foveal_aggregated_attention=False),
    )
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=_tiny_model_config())
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    common_keys = sorted(set(control_state).intersection(candidate_state))
    assert common_keys
    for key in common_keys:
        assert torch.equal(control_state[key], candidate_state[key]), key
    assert isinstance(control.blocks[0].attn, MultiHeadSelfAttention)
    assert isinstance(candidate.blocks[0].attn, FovealAggregatedAttention)
    assert isinstance(candidate.blocks[1].attn, MultiHeadSelfAttention)
    parameter_delta = sum(value.numel() for value in candidate.parameters()) - sum(
        value.numel() for value in control.parameters()
    )
    assert 0 < parameter_delta <= 100_000


def test_faa_model_wiring_trace_and_post_block_pruning() -> None:
    torch.manual_seed(13)
    model = create_model(num_classes=5, model_config=_tiny_model_config()).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        features = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
    assert features["attention_representation"] == "mixed"
    assert features["attention_representations"] == {
        0: "faa_local_pool_proxy",
        1: "mhsa_probability",
    }
    trace = features["trace"]
    assert trace["foveal_aggregated_attention_layers"].tolist() == [1]
    assert trace["foveal_patch_count"].tolist() == [16.0]
    assert trace["foveal_local_mass_mean"].shape == (1,)
    assert trace["foveal_pooled_mass_mean"].shape == (1,)

    with torch.inference_mode():
        pruned = model.forward_features(images, return_trace=True)
    assert pruned["patches"].shape[1] == 8


def test_faa_rejects_sparse_grid_and_locked_route_conflicts() -> None:
    inputs, indices = _inputs()
    module = FovealAggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        fixed_pool_size=2,
    )
    with pytest.raises(ValueError, match="complete dense patch grid"):
        module(inputs[:, :-1], grid_size=(4, 4), prefix_count=5)
    with pytest.raises(ValueError, match="identity-ordered"):
        module(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices.flip(1),
        )
    with pytest.raises(ValueError, match="locked to transformer layer 1"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                foveal_aggregated_attention_layers="2",
            ),
        )
    with pytest.raises(ValueError, match="Visual-Contrast"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                visual_contrast_attention=True,
                visual_contrast_attention_layers="1",
                visual_contrast_tokens=4,
                token_pruning=False,
            ),
        )


def test_v8_launcher_exposes_faa_controls() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$FovealAggregatedAttention = $false" in script
    assert '"--foveal-aggregated-attention"' in script
    assert '"--foveal-aggregated-attention-layers"' in script
    assert '"--foveal-aggregated-attention-window-size"' in script
    assert '"--foveal-aggregated-attention-pool-size"' in script
