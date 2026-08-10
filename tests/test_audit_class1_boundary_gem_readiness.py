from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_class1_boundary_gem_readiness import (
    _class1_margin_loss,
    _rank_boundary_strata,
    _scipy_dual_reference,
    assess_stage_a,
    project_gem_gradient,
    solve_nonnegative_dual_active_set,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow


def test_gem_single_constraint_matches_orthogonal_projection() -> None:
    current = [torch.tensor([1.0, -2.0])]
    memory = [torch.tensor([1.0, 1.0])]
    projected, telemetry = project_gem_gradient(
        current,
        [
            memory,
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([-1.0, -1.0])],
            [torch.tensor([2.0, 0.0])],
        ],
    )
    observed = projected[0]
    assert telemetry["raw_violation_count"] == 1
    assert torch.allclose(observed, torch.tensor([1.5, -1.5]), atol=1e-6)
    for dot in telemetry["projected_dots"]:
        assert dot >= -1e-6
    assert torch.isfinite(observed).all()


def test_gem_multi_constraint_projects_to_quadrant_origin() -> None:
    current = [torch.tensor([-1.0, -1.0])]
    memories = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0])],
        [torch.tensor([1.0, 1.0])],
        [torch.tensor([2.0, 1.0])],
    ]
    projected, telemetry = project_gem_gradient(current, memories)
    assert torch.allclose(projected[0], torch.zeros(2), atol=1e-6)
    assert min(telemetry["projected_dots"]) >= -1e-6
    assert telemetry["scipy_direction_relative_error"] <= 1e-7


def test_nonviolating_dual_keeps_zero_multipliers() -> None:
    gram = np.eye(4, dtype=np.float64)
    linear = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = solve_nonnegative_dual_active_set(gram, linear)
    assert np.array_equal(result["dual"], np.zeros(4))
    assert result["active"] == []


def test_active_set_matches_independent_scipy_nnls() -> None:
    gram = np.asarray(
        [[2.0, 0.4, 0.2, 0.0], [0.4, 1.5, 0.1, 0.2], [0.2, 0.1, 1.2, 0.3], [0.0, 0.2, 0.3, 1.0]],
        dtype=np.float64,
    )
    linear = np.asarray([-1.0, 0.2, -0.3, 0.1], dtype=np.float64)
    active = solve_nonnegative_dual_active_set(gram, linear)
    reference = _scipy_dual_reference(gram, linear)
    delta = np.asarray(active["dual"]) - np.asarray(reference["dual"])
    assert float(delta @ gram @ delta) <= 1e-14


def test_class1_margin_loss_matches_softplus_of_argmax_margin() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0, -1.0, 0.5]], requires_grad=True)
    targets = torch.tensor([1])
    loss = _class1_margin_loss(logits, targets)
    assert loss.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(-1.0)).item())
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_rank_boundary_strata_uses_margin_then_sample_index() -> None:
    def row(index: int, p1: float, p0: float) -> CleanTrainRow:
        path = Path(f"D:/dataset/train/{index}.jpg")
        remaining = (1.0 - p1 - p0) / 3.0
        probabilities = (p0, p1, remaining, remaining, remaining)
        return CleanTrainRow(
            sample_index=index,
            source_stem=str(index),
            image_path=path,
            fold=1,
            target=1,
            keeper_prediction=int(np.argmax(probabilities)),
            keeper_probabilities=probabilities,
        )

    rows = [row(3, 0.4, 0.3), row(1, 0.2, 0.5), row(2, 0.3, 0.3), row(0, 0.8, 0.1)]
    groups = _rank_boundary_strata(rows, fit_fold=0, focus_class=1, strata=2)
    assert [value.sample_index for value in groups[0]] == [1, 2]
    assert [value.sample_index for value in groups[1]] == [3, 0]


def _comparison_payload(
    *,
    class1_f1_delta: float = 0.006,
    class1_recall_delta: float = 0.0,
    tp_breaks: int = 0,
) -> dict[str, object]:
    return {
        "control": {"predicted_support": [300, 100, 300, 600, 543]},
        "candidate": {"predicted_support": [301, 99, 300, 600, 543]},
        "delta": {
            "macro_f1": 0.001,
            "class1_f1": class1_f1_delta,
            "class1_precision": 0.010,
            "class1_recall": class1_recall_delta,
        },
        "transitions": {
            "restricted_focus_fp_reduction": 4,
            "focus_fn_rescue": 1,
            "focus_tp_break": tp_breaks,
            "candidate_correction": 4,
            "candidate_harm": 1,
        },
        "maximum_nonfocus_f1_drop": 0.005,
    }


def test_stage_a_gate_passes_complete_synthetic_case() -> None:
    raw_candidate = _comparison_payload()
    raw_control = _comparison_payload(class1_f1_delta=0.003, class1_recall_delta=-0.005, tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=0.003, class1_recall_delta=0.005)
    prior_candidate = _comparison_payload(class1_f1_delta=0.006)
    illumination = []
    prior_illumination = []
    for condition in ("lighting_dim", "lighting_bright", "low_contrast"):
        value = _comparison_payload()
        value["condition"] = condition
        illumination.append(value)
        prior = _comparison_payload(class1_f1_delta=0.025)
        prior["condition"] = condition
        prior_illumination.append(prior)
    result = assess_stage_a(
        structural_checks={"structural": True},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        clean_prior_agem_candidate=prior_candidate,
        illumination_raw_candidate=illumination,
        illumination_prior_agem_candidate=prior_illumination,
    )
    assert result["all_gates_passed"] is True
    assert result["stage_b_authorized"] is True


def test_stage_a_gate_rejects_raw_and_comparator_failures() -> None:
    raw_candidate = _comparison_payload(class1_f1_delta=-0.001, tp_breaks=2)
    raw_control = _comparison_payload(tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=-0.001, class1_recall_delta=-0.01)
    prior_candidate = _comparison_payload(class1_f1_delta=0.0)
    illumination = []
    prior_illumination = []
    for condition in ("lighting_dim", "lighting_bright", "low_contrast"):
        value = _comparison_payload()
        value["condition"] = condition
        illumination.append(value)
        prior = _comparison_payload(class1_f1_delta=0.0)
        prior["condition"] = condition
        prior_illumination.append(prior)
    result = assess_stage_a(
        structural_checks={"qp": False},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        clean_prior_agem_candidate=prior_candidate,
        illumination_raw_candidate=illumination,
        illumination_prior_agem_candidate=prior_illumination,
    )
    assert result["all_gates_passed"] is False
    assert "qp" in result["failed_checks"]
    assert "class1_f1_delta_gte_0p005" in result["failed_checks"]
    assert "candidate_f1_exceeds_aggregate_control" in result["failed_checks"]
