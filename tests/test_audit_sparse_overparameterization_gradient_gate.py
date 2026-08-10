import torch
import torch.nn.functional as F

from trkh.tools.audit_sparse_overparameterization_gradient_gate import (
    EXPECTED_CLASS1_TP,
    EXPECTED_FIT_ROWS,
    EXPECTED_RESTRICTED_FP,
    NUM_CLASSES,
    _finite_difference_diagnostics,
    assess_gradient_gate,
    sop_independent_equation,
    sop_official_equation,
)


def _equation_inputs(dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(8, NUM_CLASSES, generator=generator, dtype=dtype)
    targets = F.one_hot(torch.tensor([0, 1, 2, 3, 4, 1, 0, 2]), NUM_CLASSES).to(
        dtype
    )
    u = torch.randn(8, 1, generator=generator, dtype=dtype) * 0.08
    v = torch.randn(8, NUM_CLASSES, generator=generator, dtype=dtype) * 0.08
    return logits, targets, u, v


def test_independent_sop_equation_matches_official_expression_and_gradients() -> None:
    logits, targets, u, v = _equation_inputs()
    first = [value.detach().clone().requires_grad_(True) for value in (logits, u, v)]
    second = [value.detach().clone().requires_grad_(True) for value in (logits, u, v)]
    official_loss, official_terms = sop_official_equation(
        first[0], targets, first[1], first[2]
    )
    independent_loss, independent_terms = sop_independent_equation(
        second[0], targets, second[1], second[2]
    )
    official_gradients = torch.autograd.grad(official_loss, first)
    independent_gradients = torch.autograd.grad(independent_loss, second)
    assert torch.allclose(official_loss, independent_loss, rtol=0.0, atol=1e-10)
    assert torch.allclose(
        official_terms["corrected_probability"],
        independent_terms["corrected_probability"],
        rtol=0.0,
        atol=1e-10,
    )
    for official, independent in zip(official_gradients, independent_gradients):
        assert torch.allclose(official, independent, rtol=0.0, atol=1e-9)


def test_sop_finite_difference_diagnostics_pass_locked_tolerance() -> None:
    result = _finite_difference_diagnostics()
    assert result["passed"] is True
    assert result["maximum_absolute_error"] <= 1e-6


def test_zero_noise_sop_preserves_standard_cross_entropy_logit_gradient() -> None:
    logits, targets, _, _ = _equation_inputs(dtype=torch.float32)
    logits = logits.requires_grad_(True)
    u = torch.zeros(logits.shape[0], 1, requires_grad=True)
    v = torch.zeros_like(logits, requires_grad=True)
    loss, terms = sop_official_equation(logits, targets, u, v)
    observed = torch.autograd.grad(loss, logits)[0] * logits.shape[0]
    expected = F.softmax(logits.detach(), dim=1) - targets
    assert torch.allclose(observed, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(terms["u_effective"]) == 0
    assert torch.count_nonzero(terms["v_effective"]) == 0


def _gate_payload(*, active_v: bool, attenuate_restricted: bool = False):
    rows = EXPECTED_FIT_ROWS
    targets = torch.zeros(rows, dtype=torch.long)
    predictions = torch.zeros(rows, dtype=torch.long)
    targets[:EXPECTED_CLASS1_TP] = 1
    predictions[:EXPECTED_CLASS1_TP] = 1
    restricted_start = EXPECTED_CLASS1_TP
    restricted_stop = restricted_start + EXPECTED_RESTRICTED_FP
    predictions[restricted_start:restricted_stop] = 1
    predictions[restricted_stop] = 2
    ce_gradient = torch.full((rows, NUM_CLASSES), 0.1)
    ce_gradient[:EXPECTED_CLASS1_TP, 1] = -0.2
    sop_gradient = ce_gradient.clone()
    if attenuate_restricted:
        sop_gradient[restricted_start:restricted_stop, 1] *= 0.1
    corrected = torch.full((rows, NUM_CLASSES), 0.05)
    corrected[:, 0] = 0.8
    corrected[:, 1] = 0.05
    corrected[:, 2] = 0.04
    corrected[:, 3] = 0.04
    corrected[:, 4] = 0.07
    corrected[:EXPECTED_CLASS1_TP, 0] = 0.05
    corrected[:EXPECTED_CLASS1_TP, 1] = 0.8
    u = torch.full((rows, 1), 0.01)
    v = torch.full((rows, NUM_CLASSES), 0.01)
    u_effective = torch.zeros(rows, NUM_CLASSES)
    v_effective = torch.zeros(rows, NUM_CLASSES)
    if active_v:
        active_rows = (rows + 99) // 100
        v_effective[:active_rows, 2] = 2e-6
    noise_energy = torch.full((rows,), 1e-8)
    noise_energy[restricted_start:restricted_stop] = 0.5e-8
    return {
        "targets": targets,
        "raw_predictions": predictions,
        "corrected_probability": corrected,
        "ce_gradient": ce_gradient,
        "sop_gradient": sop_gradient,
        "u": u,
        "v": v,
        "u_effective": u_effective,
        "v_effective": v_effective,
        "noise_energy": noise_energy,
        "replay_passed": True,
        "source_replay_passed": True,
    }


def test_gradient_gate_passes_complete_synthetic_contract() -> None:
    result = assess_gradient_gate(**_gate_payload(active_v=True))
    assert result["all_gates_passed"] is True
    assert result["trainer_integration_authorized"] is True


def test_gradient_gate_rejects_inactive_noise_and_restricted_fp_attenuation() -> None:
    result = assess_gradient_gate(
        **_gate_payload(active_v=False, attenuate_restricted=True)
    )
    assert result["all_gates_passed"] is False
    assert result["trainer_integration_authorized"] is False
    assert "effective_v_active_at_least_1pct" in result["failed_checks"]
    assert "restricted_fp_median_ratio_at_least_0p95" in result["failed_checks"]
    assert "restricted_fp_half_loss_at_most_5pct" in result["failed_checks"]
