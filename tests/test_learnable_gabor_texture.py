from __future__ import annotations

import copy
import math

import torch

from trkh.core.config import ModelConfig
from trkh.models.learnable_gabor_texture import (
    LearnableGaborTextureEncoder,
    LearnableGaborTextureResidual,
)
from trkh.models.model import build_model_from_checkpoint, create_model, load_model_state
from trkh.training.train import parse_args


def _model_config(
    *,
    enabled: bool = False,
    semantic_fusion: bool = False,
) -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=64,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=True,
        branch_color_tokens=1,
        branch_edge_tokens=1,
        branch_cnn_tokens=0,
        branch_token_dropout=0.0,
        token_pruning=False,
        learnable_gabor_texture_residual=enabled,
        learnable_gabor_texture_semantic_fusion=semantic_fusion,
    )


def _inputs(batch_size: int = 2):
    image = torch.randn(batch_size, 3, 64, 64)
    bbox = torch.tensor(
        [[0.50, 0.50, 0.55, 0.60], [0.42, 0.58, 0.35, 0.40]],
        dtype=torch.float32,
    )[:batch_size]
    return image, bbox


def test_gabor_constraints_histogram_and_materialized_kernels() -> None:
    torch.manual_seed(42)
    module = LearnableGaborTextureResidual(embed_dim=64)
    image, bbox = _inputs()
    components = module.forward_components(image, bbox)
    parameters = module.constrained_parameters()

    assert sum(parameter.numel() for parameter in module.parameters()) <= 100_000
    assert components["texture_token"].shape == (2, 64)
    assert components["magnitude"].shape == (2, 32, 16, 16)
    assert components["counts"].shape == (2, 32, 8)
    assert torch.allclose(
        components["counts"].sum(dim=-1),
        torch.ones(2, 32),
        atol=1e-5,
        rtol=0.0,
    )
    expected_position = torch.matmul(
        components["assignment"].flatten(-2),
        module.position_features,
    ) / float(module.response_size * module.response_size)
    assert torch.equal(components["position_descriptor"], expected_position)
    assert bool((parameters["theta"] >= 0.0).all())
    assert bool((parameters["theta"] <= math.pi).all())
    assert bool((parameters["frequency"][:16] <= module.frequency_upper[:16]).all())
    assert bool((parameters["frequency"][16:] >= module.frequency_lower[16:]).all())
    assert bool((parameters["sigma_x"] >= parameters["sigma_x_lower"]).all())
    assert bool((parameters["sigma_x"] <= parameters["sigma_x_upper"] + 1e-6).all())
    assert bool((parameters["sigma_y"] >= module.sigma_y_bounds[0]).all())
    assert bool((parameters["sigma_y"] <= module.sigma_y_bounds[1]).all())

    materialized = module.materialized_kernels()
    replay = module.forward_components(image, bbox, kernel_override=materialized)
    assert torch.allclose(
        components["texture_token"], replay["texture_token"], atol=1e-6, rtol=0.0
    )


def test_gabor_construction_is_rng_neutral_and_seed_deterministic() -> None:
    torch.manual_seed(42)
    state_before = torch.get_rng_state().clone()
    first = LearnableGaborTextureResidual(embed_dim=64)
    state_after = torch.get_rng_state().clone()

    torch.manual_seed(42)
    second = LearnableGaborTextureResidual(embed_dim=64)

    assert torch.equal(state_before, state_after)
    assert set(first.state_dict()) == set(second.state_dict())
    assert all(
        torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key in first.state_dict()
    )


def test_gabor_mechanism_ablations_change_descriptor() -> None:
    torch.manual_seed(42)
    module = LearnableGaborTextureResidual(embed_dim=64).eval()
    image, bbox = _inputs()
    with torch.no_grad():
        full = module.forward_components(image, bbox)["texture_token"]
        low = module.forward_components(image, bbox, use_high=False)["texture_token"]
        high = module.forward_components(image, bbox, use_low=False)["texture_token"]
        no_position = module.forward_components(
            image,
            bbox,
            use_lho_position=False,
        )["texture_token"]
        no_parameter_encoding = module.forward_components(
            image,
            bbox,
            use_filter_parameter_encoding=False,
        )["texture_token"]

    for candidate in (low, high, no_position, no_parameter_encoding):
        assert float((full - candidate).abs().mean()) > 1e-6


def test_zero_gate_and_active_gate_gradients_are_live() -> None:
    torch.manual_seed(42)
    module = LearnableGaborTextureResidual(embed_dim=64)
    image, bbox = _inputs()
    target = torch.randn(2, 64)

    residual = module(image, bbox)
    assert torch.count_nonzero(residual) == 0
    (residual * target).sum().backward()
    assert module.raw_gate.grad is not None
    assert torch.isfinite(module.raw_gate.grad)
    assert float(module.raw_gate.grad.abs()) > 0.0

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.raw_gate.fill_(torch.atanh(torch.tensor(0.2)))
    module(image, bbox).square().mean().backward()
    required = (
        module.raw_theta,
        module.raw_frequency,
        module.raw_sigma_x,
        module.raw_sigma_y,
        module.level_projection.weight,
        module.position_projection.weight,
        module.lho_attention.qkv.weight,
        module.filter_parameter_projection.weight,
        module.fcm_attention.qkv.weight,
        module.output_projection.weight,
    )
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert int(torch.count_nonzero(parameter.grad)) > 0


def test_trkh_zero_gate_is_exact_and_checkpoint_safe() -> None:
    control_config = _model_config(enabled=False)
    candidate_config = copy.deepcopy(control_config)
    candidate_config.learnable_gabor_texture_residual = True
    torch.manual_seed(42)
    control = create_model(num_classes=5, model_config=control_config)
    control_post_constructor_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_post_constructor_rng = torch.get_rng_state().clone()
    missing, unexpected = load_model_state(candidate, control.state_dict(), strict=False)

    assert torch.equal(control_post_constructor_rng, candidate_post_constructor_rng)
    assert missing
    assert all(key.startswith("gabor_texture_residual.") for key in missing)
    assert unexpected == []
    control.eval()
    candidate.eval()
    image, bbox = _inputs()
    with torch.no_grad():
        control_features = control.forward_features(image, bbox_token_prior=bbox)
        candidate_features = candidate.forward_features(image, bbox_token_prior=bbox)
        control_logits = control.head(control_features["pooled"])
        candidate_logits = candidate.head(candidate_features["pooled"])
    assert torch.equal(control_logits, candidate_logits)

    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(candidate_config),
            "model_state": candidate.state_dict(),
        }
    )
    assert isinstance(
        restored.gabor_texture_residual,
        LearnableGaborTextureResidual,
    )
    assert set(restored.state_dict()) == set(candidate.state_dict())


def test_gabor_requires_edge_token_and_cli_flag_is_default_off() -> None:
    config = _model_config(enabled=True)
    config.branch_edge_tokens = 0
    try:
        create_model(num_classes=5, model_config=config)
    except ValueError as error:
        assert "edge token" in str(error)
    else:
        raise AssertionError("Expected missing edge-token rejection")

    assert parse_args([]).learnable_gabor_texture_residual is False
    assert parse_args([]).learnable_gabor_texture_semantic_fusion is False
    assert (
        parse_args(["--learnable-gabor-texture-residual"]).learnable_gabor_texture_residual
        is True
    )


def test_active_gabor_encoder_has_no_residual_gate_and_emits_trace() -> None:
    torch.manual_seed(42)
    encoder = LearnableGaborTextureEncoder(embed_dim=64).eval()
    image, bbox = _inputs()

    with torch.no_grad():
        texture, trace = encoder(image, bbox, return_trace=True)

    assert not hasattr(encoder, "raw_gate")
    assert texture.shape == (2, 64)
    assert torch.equal(texture, trace["texture_token"])
    assert trace["fcm_attention"].shape == (2, 4, 32, 32)
    assert torch.isfinite(texture).all()


def test_active_gabor_semantic_fusion_is_rng_neutral_and_direct() -> None:
    control_config = _model_config()
    candidate_config = _model_config(semantic_fusion=True)

    torch.manual_seed(42)
    control = create_model(num_classes=5, model_config=control_config)
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = torch.get_rng_state().clone()

    assert torch.equal(control_rng, candidate_rng)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    shared_names = [name for name in candidate_state if name in control_state]
    assert shared_names
    assert all(torch.equal(candidate_state[name], control_state[name]) for name in shared_names)
    extra_names = [name for name in candidate_state if name not in control_state]
    assert extra_names
    assert all(name.startswith("gabor_texture_semantic_encoder.") for name in extra_names)

    control.eval()
    candidate.eval()
    image, bbox = _inputs()
    with torch.no_grad():
        control_features = control.forward_features(image, bbox_token_prior=bbox)
        candidate_features = candidate.forward_features(
            image,
            bbox_token_prior=bbox,
            return_trace=True,
        )
        control_logits = control(image)
        candidate_head_input = candidate.head_input_from_features(candidate_features)
        candidate_logits = candidate.head(candidate_head_input)

    expected = (
        candidate_features["pooled"]
        + candidate_features["gabor_texture_semantic_feature"].to(
            dtype=candidate_features["pooled"].dtype
        )
    )
    assert torch.equal(candidate_head_input, expected)
    assert not torch.equal(control_logits, candidate_logits)
    assert candidate.gabor_texture_residual is None
    assert isinstance(
        candidate.gabor_texture_semantic_encoder,
        LearnableGaborTextureEncoder,
    )
    assert "gabor_texture_semantic_fused_norm" in candidate_features["trace"]

    restored = build_model_from_checkpoint(
        {
            "class_names": [f"class_{index}" for index in range(5)],
            "model_config": vars(candidate_config),
            "model_state": candidate.state_dict(),
        }
    )
    assert isinstance(
        restored.gabor_texture_semantic_encoder,
        LearnableGaborTextureEncoder,
    )
    assert set(restored.state_dict()) == set(candidate.state_dict())


def test_active_and_residual_gabor_modes_are_mutually_exclusive() -> None:
    config = _model_config(enabled=True, semantic_fusion=True)
    try:
        create_model(num_classes=5, model_config=config)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("Expected mutually exclusive Gabor modes to fail")

    parsed = parse_args(["--learnable-gabor-texture-semantic-fusion"])
    assert parsed.learnable_gabor_texture_semantic_fusion is True
