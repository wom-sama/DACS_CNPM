from __future__ import annotations

import math

import pytest
import torch

from trkh.models.dinov3_cgaer_bridge_b11 import (
    CGAER_B11_CANDIDATE_MODE,
    CGAER_B11_CONTROL_MODE,
    CGAER_B11_DENSE_MACS,
    CGAER_B11_PARAMETER_COUNT,
    CGAER_B11_RESIDUAL_L2_CAP,
    DinoV3CGAERBridgeB11,
)


def _paired() -> tuple[DinoV3CGAERBridgeB11, DinoV3CGAERBridgeB11]:
    torch.manual_seed(1101)
    candidate = DinoV3CGAERBridgeB11(CGAER_B11_CANDIDATE_MODE)
    control = DinoV3CGAERBridgeB11(CGAER_B11_CONTROL_MODE)
    control.load_state_dict(candidate.state_dict(), strict=True)
    return candidate, control


def test_b11_shape_finite_capacity_and_zero_initial_state() -> None:
    candidate, control = _paired()
    z = torch.randn(7, 384)
    base = torch.randn(7, 5)

    for model in (candidate, control):
        logits, trace = model(z, base, return_trace=True)
        assert logits.shape == (7, 5)
        assert torch.isfinite(logits).all()
        assert torch.equal(logits, base)
        assert torch.count_nonzero(trace["raw_outputs"]) == 0
        assert torch.count_nonzero(trace["residual"]) == 0
        assert torch.equal(trace["gate"], torch.full((7,), 0.5))
        assert model.added_parameter_count() == CGAER_B11_PARAMETER_COUNT
        assert model.added_parameter_count(trainable_only=True) == 3_125
        assert model.dense_macs() == CGAER_B11_DENSE_MACS
        assert not model.input_norm.elementwise_affine
        assert not list(model.input_norm.parameters())
        assert torch.count_nonzero(model.output_projection.weight) == 0
        assert torch.count_nonzero(model.output_projection.bias) == 0

    assert {
        name: tuple(parameter.shape)
        for name, parameter in candidate.named_parameters()
    } == {
        "input_projection.weight": (8, 384),
        "input_projection.bias": (8,),
        "output_projection.weight": (5, 8),
        "output_projection.bias": (5,),
    }


@pytest.mark.parametrize(
    "mode", (CGAER_B11_CANDIDATE_MODE, CGAER_B11_CONTROL_MODE)
)
def test_b11_branch_off_is_exact_and_does_not_evaluate_trainable_path(
    mode: str,
) -> None:
    model = DinoV3CGAERBridgeB11(mode)
    with torch.no_grad():
        model.output_projection.weight.normal_()
        model.output_projection.bias.normal_()
    z = torch.randn(3, 384, requires_grad=True)
    base = torch.randn(3, 5, requires_grad=True)

    logits, trace = model(z, base, branch_off=True, return_trace=True)
    assert torch.equal(logits, base)
    assert trace["branch_off"] is True
    assert torch.count_nonzero(trace["residual"]) == 0
    assert not logits.requires_grad


def test_b11_fixed_G_and_ordered_bases_are_orthonormal_and_zero_sum() -> None:
    model = DinoV3CGAERBridgeB11()
    G = model.fixed_categorical_basis()
    ordered_basis = model.ordered_basis
    gains = model.ordered_basis_gains()

    assert G.shape == (5, 2)
    assert torch.allclose(G.transpose(0, 1) @ G, torch.eye(2), atol=1e-7)
    assert torch.allclose(G.sum(dim=0), torch.zeros(2), atol=1e-7)
    assert torch.allclose(
        G[:3, 0], torch.full((3,), math.sqrt(2.0 / 15.0)), atol=1e-7
    )
    assert torch.allclose(ordered_basis.T @ ordered_basis, torch.eye(2), atol=1e-7)
    assert torch.allclose(ordered_basis.sum(dim=0), torch.zeros(2), atol=1e-7)
    assert torch.allclose(
        gains,
        torch.tensor((8.0 * math.sqrt(2.0) / 9.0, math.sqrt(6.0) / 27.0)),
        atol=1e-7,
    )


def test_b11_candidate_and_control_zero_state_jacobian_singular_values_match() -> None:
    candidate, control = (model.double() for model in _paired())
    gate = torch.tensor([0.5], dtype=torch.float64)

    def candidate_map(coordinates: torch.Tensor) -> torch.Tensor:
        return candidate.candidate_ordered_energy(
            coordinates[None, 0], coordinates[None, 1], gate
        )[0]

    def control_map(coordinates: torch.Tensor) -> torch.Tensor:
        return control.control_ordered_contrasts(coordinates[None])[0]

    origin = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    candidate_jacobian = torch.autograd.functional.jacobian(candidate_map, origin)
    control_jacobian = torch.autograd.functional.jacobian(control_map, origin)
    expected = torch.tensor(
        ((-8.0 / 9.0, -1.0 / 27.0), (0.0, 2.0 / 27.0),
         (8.0 / 9.0, -1.0 / 27.0)),
        dtype=torch.float64,
    )

    assert torch.allclose(candidate_jacobian, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(control_jacobian, expected, atol=2e-8, rtol=2e-8)
    assert torch.allclose(
        torch.linalg.svdvals(candidate_jacobian),
        torch.linalg.svdvals(control_jacobian),
        atol=2e-8,
        rtol=2e-8,
    )


@pytest.mark.parametrize(
    "mode", (CGAER_B11_CANDIDATE_MODE, CGAER_B11_CONTROL_MODE)
)
def test_b11_common_smooth_cap_and_force_gate_ablation(mode: str) -> None:
    model = DinoV3CGAERBridgeB11(mode)
    raw = torch.tensor(
        ((20.0, -20.0, 100.0, -100.0, 4.0),
         (-30.0, 30.0, -80.0, 70.0, -4.0))
    )
    base = torch.tensor(
        ((8.0, -3.0, 1.0, 0.2, -0.4), (-6.0, 2.0, 5.0, -0.1, 0.3))
    )

    residual, trace = model.residual_from_raw_outputs(
        raw, base, return_trace=True
    )
    assert torch.isfinite(residual).all()
    assert torch.all(trace["residual_l2_norm"] < CGAER_B11_RESIDUAL_L2_CAP)
    assert torch.all(trace["uncapped_residual_l2_norm"] > 1.0)
    expected = trace["uncapped_residual"] / torch.sqrt(
        1.0
        + torch.square(
            trace["uncapped_residual_l2_norm"][:, None]
            / CGAER_B11_RESIDUAL_L2_CAP
        )
    )
    assert torch.allclose(residual, expected, atol=2e-6, rtol=2e-6)

    forced, forced_trace = model.residual_from_raw_outputs(
        raw, base, force_gate=0.5, return_trace=True
    )
    raw_zero_gate = raw.clone()
    raw_zero_gate[:, 4] = 0.0
    neutral = model.residual_from_raw_outputs(raw_zero_gate, base)
    assert torch.allclose(forced, neutral, atol=1e-7, rtol=1e-7)
    assert torch.equal(forced_trace["gate"], torch.full((2,), 0.5))
    assert torch.equal(forced_trace["conflict_logit"], raw[:, 4])


def test_b11_control_coordinates_are_unbounded_before_common_cap() -> None:
    model = DinoV3CGAERBridgeB11(CGAER_B11_CONTROL_MODE)
    base = torch.zeros(1, 5)
    raw = torch.tensor(((1_000.0, -1_000.0, 0.0, 0.0, 0.0),))
    uncapped = model.uncapped_residual_from_raw_outputs(raw, base)
    capped = model.residual_from_raw_outputs(raw, base)

    assert float(torch.linalg.vector_norm(uncapped)) > 1_000.0
    assert float(torch.linalg.vector_norm(capped)) < 0.5


def test_b11_low_precision_cap_does_not_overflow_to_zero() -> None:
    model = DinoV3CGAERBridgeB11(CGAER_B11_CONTROL_MODE).half()
    base = torch.zeros(1, 5, dtype=torch.float16)
    raw = torch.tensor(
        ((1_000.0, -1_000.0, 0.0, 0.0, 0.0),),
        dtype=torch.float16,
    )
    residual = model.residual_from_raw_outputs(raw, base)
    norm = torch.linalg.vector_norm(residual.float())

    assert torch.isfinite(residual).all()
    assert 0.49 < float(norm) < CGAER_B11_RESIDUAL_L2_CAP


@pytest.mark.parametrize(
    "mode", (CGAER_B11_CANDIDATE_MODE, CGAER_B11_CONTROL_MODE)
)
def test_b11_active_gradients_are_finite_and_frozen_inputs_do_not_receive_grad(
    mode: str,
) -> None:
    torch.manual_seed(1129)
    model = DinoV3CGAERBridgeB11(mode).train()
    with torch.no_grad():
        model.output_projection.weight.normal_(0.0, 0.05)
        model.output_projection.bias.normal_(0.0, 0.01)
    z = torch.randn(11, 384, requires_grad=True)
    base = torch.randn(11, 5, requires_grad=True)

    logits, trace = model(z, base, return_trace=True)
    loss = logits.square().mean() + 0.1 * trace["conflict_logit"].mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert z.grad is None
    assert base.grad is None
