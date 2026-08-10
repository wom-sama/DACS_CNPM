from __future__ import annotations

import copy

import pytest
import timm
import torch
import torch.nn.functional as F

from trkh.models.swiftformer_surfacefold_b17 import (
    SURFACEFOLD_MEAN_CONTROL_B17_MODE,
    SURFACEFOLD_OFF_B17_MODE,
    SURFACEFOLD_SPATIAL_B17_MODE,
    SURFACEFOLD_RELATION_EDGES,
    SwiftFormerSurfaceFoldB17,
    dino_s3_relation_target_b17,
    factorized_delta_parameter_count_b17,
    surfacefold_s3_relation_loss_b17,
)


STOCK_PARAMETERS = 3_035_570
FROZEN_DOWNSAMPLE_PARAMETERS = 221_980
FACTOR_PARAMETERS = 25_648


def _backbone() -> torch.nn.Module:
    return timm.create_model(
        "swiftformer_xs",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )


def _tensor(shape: tuple[int, ...], seed: int, *, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def test_parameter_counts_freeze_and_one_percent_initialization() -> None:
    base = _backbone()
    off = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_OFF_B17_MODE
    )
    candidate = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
    )

    assert sum(parameter.numel() for parameter in base.parameters()) == STOCK_PARAMETERS
    assert sum(parameter.numel() for parameter in off.parameters()) == STOCK_PARAMETERS
    assert factorized_delta_parameter_count_b17() == FACTOR_PARAMETERS
    assert (
        sum(parameter.numel() for parameter in candidate.parameters())
        == STOCK_PARAMETERS + FACTOR_PARAMETERS
    )
    assert sum(parameter.numel() for parameter in off.parameters() if parameter.requires_grad) == (
        STOCK_PARAMETERS - FROZEN_DOWNSAMPLE_PARAMETERS
    )
    assert sum(
        parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
    ) == STOCK_PARAMETERS - FROZEN_DOWNSAMPLE_PARAMETERS + FACTOR_PARAMETERS

    projection = candidate.backbone.stages[3].downsample.proj
    assert not projection.weight.requires_grad
    assert projection.bias is not None and not projection.bias.requires_grad
    assert candidate.factor_p is not None and candidate.factor_d is not None
    assert torch.equal(candidate.factor_d, torch.full_like(candidate.factor_d, 1.0 / 9.0))
    delta_norm = candidate.effective_delta_weight().double().flatten(1).norm(dim=1)
    base_norm = projection.weight.double().flatten(1).norm(dim=1)
    assert torch.allclose(delta_norm / base_norm, torch.full_like(base_norm, 0.01), atol=2e-8, rtol=2e-6)


def test_constructor_preserves_rng_and_candidate_control_start_equal() -> None:
    base = _backbone()
    rng_before = torch.random.get_rng_state().clone()
    candidate = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    control = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_MEAN_CONTROL_B17_MODE
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    candidate_state = candidate.state_dict()
    control_state = control.state_dict()
    assert candidate_state.keys() == control_state.keys()
    assert all(
        torch.equal(candidate_state[key], control_state[key])
        for key in candidate_state
        if key != "_surfacefold_mode_id"
    )
    assert int(candidate_state["_surfacefold_mode_id"]) == 2
    assert int(control_state["_surfacefold_mode_id"]) == 1
    s2 = _tensor((2, 112, 14, 14), 101)
    candidate.eval()
    control.eval()
    with torch.no_grad():
        candidate_s3 = candidate.forward_stage3_downsample(s2)
        control_s3 = control.forward_stage3_downsample(s2)
    assert torch.equal(candidate_s3, control_s3)


def test_strict_load_rejects_cross_mode_checkpoint() -> None:
    base = _backbone()
    candidate = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
    )
    control = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_MEAN_CONTROL_B17_MODE
    )

    with pytest.raises(RuntimeError, match="immutable B17 mode contract"):
        control.load_state_dict(candidate.state_dict(), strict=True)
    assert control.mode == SURFACEFOLD_MEAN_CONTROL_B17_MODE
    assert int(control._surfacefold_mode_id) == 1


@pytest.mark.parametrize(
    ("mode", "expect_centered_gradient"),
    [
        (SURFACEFOLD_SPATIAL_B17_MODE, True),
        (SURFACEFOLD_MEAN_CONTROL_B17_MODE, False),
    ],
)
def test_first_backward_activates_both_factors(
    mode: str, expect_centered_gradient: bool
) -> None:
    model = SwiftFormerSurfaceFoldB17(_backbone(), mode).eval()
    s2 = _tensor((2, 112, 14, 14), 211)
    teacher = _tensor((2, 384, 16, 16), 223)
    target = dino_s3_relation_target_b17(teacher)
    loss = surfacefold_s3_relation_loss_b17(
        model.forward_stage3_downsample(s2), target
    )
    loss.backward()

    assert model.factor_p is not None and model.factor_p.grad is not None
    assert model.factor_d is not None and model.factor_d.grad is not None
    assert torch.isfinite(model.factor_p.grad).all()
    assert torch.isfinite(model.factor_d.grad).all()
    assert float(model.factor_p.grad.norm()) > 0.0
    assert float(model.factor_d.grad.norm()) > 0.0
    centered = model.factor_d.grad - model.factor_d.grad.mean(
        dim=(1, 2), keepdim=True
    )
    if expect_centered_gradient:
        assert float(centered.norm()) > 1e-8
    else:
        assert float(centered.abs().max()) <= 2e-12


@pytest.mark.parametrize(
    "mode", [SURFACEFOLD_SPATIAL_B17_MODE, SURFACEFOLD_MEAN_CONTROL_B17_MODE]
)
def test_float64_factorization_oracle(mode: str) -> None:
    model = SwiftFormerSurfaceFoldB17(_backbone().double(), mode).eval()
    assert model.factor_p is not None and model.factor_d is not None
    s2 = _tensor((2, 112, 14, 14), 307, dtype=torch.float64)
    factor_d = model.factor_d
    if mode == SURFACEFOLD_MEAN_CONTROL_B17_MODE:
        factor_d = factor_d.mean(dim=(1, 2), keepdim=True).expand_as(factor_d)
    depthwise = F.conv2d(
        s2,
        factor_d[:, None, :, :],
        stride=2,
        padding=1,
        groups=112,
    )
    separable = F.conv2d(depthwise, model.factor_p[:, :, None, None])
    expanded = F.conv2d(
        s2, model.effective_delta_weight(), stride=2, padding=1
    )
    assert float((separable - expanded).abs().max()) <= 1e-10


def test_off_mode_is_exact_stock() -> None:
    stock = _backbone().eval()
    model = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(stock), SURFACEFOLD_OFF_B17_MODE
    ).eval()
    inputs = _tensor((1, 3, 224, 224), 401)
    with torch.no_grad():
        expected = stock(inputs)
        actual = model(inputs)
    assert torch.equal(actual, expected)
    assert model.factor_p is None and model.factor_d is None


def test_fold_to_deploy_matches_logits_and_removes_factors() -> None:
    model = SwiftFormerSurfaceFoldB17(
        _backbone(), SURFACEFOLD_SPATIAL_B17_MODE
    ).eval()
    assert model.factor_p is not None and model.factor_d is not None
    with torch.no_grad():
        model.factor_d.add_(
            torch.linspace(-0.02, 0.02, model.factor_d.numel()).reshape_as(
                model.factor_d
            )
        )
        model.factor_p.mul_(1.7)
    deployed = model.fold_to_deploy().eval()
    inputs = _tensor((1, 3, 224, 224), 419)
    with torch.no_grad():
        expected = model(inputs)
        actual = deployed(inputs)
    assert float((actual - expected).abs().max()) <= 1e-6
    assert sum(parameter.numel() for parameter in deployed.parameters()) == STOCK_PARAMETERS
    assert all("factor_p" not in name and "factor_d" not in name for name in deployed.state_dict())
    assert not isinstance(deployed, SwiftFormerSurfaceFoldB17)


def test_s3_relation_target_has_all_edges_and_is_shared() -> None:
    teacher = _tensor((2, 384, 16, 16), 503).requires_grad_(True)
    target = dino_s3_relation_target_b17(teacher)
    assert target.shape == (2, SURFACEFOLD_RELATION_EDGES)
    assert target.dtype == torch.float32
    assert not target.requires_grad

    base = _backbone()
    candidate = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_SPATIAL_B17_MODE
    ).eval()
    control = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_MEAN_CONTROL_B17_MODE
    ).eval()
    stock = SwiftFormerSurfaceFoldB17(
        copy.deepcopy(base), SURFACEFOLD_OFF_B17_MODE
    ).eval()
    s2 = _tensor((2, 112, 14, 14), 521)
    losses = [
        surfacefold_s3_relation_loss_b17(
            arm.forward_stage3_downsample(s2), target
        )
        for arm in (stock, control, candidate)
    ]
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses)
    assert teacher.grad is None
