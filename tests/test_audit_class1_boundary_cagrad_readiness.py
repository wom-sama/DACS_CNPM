import numpy as np
import pytest
import torch

from trkh.tools.audit_class1_boundary_cagrad_readiness import (
    assess_stage_a,
    cagrad_dual_terms,
    compose_cagrad_direction,
    solve_cagrad_dual,
)


def test_cagrad_dual_jacobian_matches_finite_difference() -> None:
    matrix = np.asarray(
        [
            [2.0, 0.2, 0.1, 0.0],
            [0.2, 1.5, 0.3, 0.1],
            [0.1, 0.3, 1.8, 0.2],
            [0.0, 0.1, 0.2, 1.2],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.15, 0.25, 0.35, 0.25], dtype=np.float64)
    _, observed = cagrad_dual_terms(weights, matrix, c=0.4)
    epsilon = 1e-6
    numerical = np.empty(4, dtype=np.float64)
    for index in range(4):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            cagrad_dual_terms(plus, matrix, c=0.4)[0]
            - cagrad_dual_terms(minus, matrix, c=0.4)[0]
        ) / (2.0 * epsilon)
    assert np.allclose(observed, numerical, rtol=1e-7, atol=1e-8)


def test_cagrad_solver_recovers_symmetric_uniform_solution() -> None:
    weights, rows, telemetry = solve_cagrad_dual(np.eye(4), c=0.4)
    assert np.allclose(weights, np.full(4, 0.25), rtol=0.0, atol=1e-12)
    assert len(rows) == 9
    assert telemetry["primary_success"] is True
    assert telemetry["all_starts_success"] is True
    assert telemetry["objective_spread"] <= 1e-10
    assert telemetry["maximum_kkt_residual"] <= 1e-7


def test_cagrad_direction_satisfies_ball_and_paper_scale_invariance() -> None:
    gradients = []
    for index in range(4):
        value = torch.zeros(4)
        value[index] = 1.0
        gradients.append([value])
    erm, candidate, telemetry, _ = compose_cagrad_direction(gradients, c=0.4)
    assert torch.allclose(erm[0], torch.full((4,), 0.25))
    assert torch.allclose(candidate[0], torch.full((4,), 0.35))
    assert telemetry["weights"] == pytest.approx([0.25] * 4)
    assert telemetry["ball_ratio"] == pytest.approx(0.4, abs=1e-7)
    assert telemetry["direction_vs_erm_relative_l2"] == pytest.approx(
        0.4, abs=1e-7
    )
    assert telemetry["paper_scale_relative_error"] <= 1e-10
    assert telemetry["official_direction_relative_error"] <= 1e-6
    assert telemetry["minimum_task_dot_gain"] > 0.0


def _comparison_payload(
    *,
    class1_f1_delta: float = 0.006,
    class1_precision_delta: float = 0.010,
    class1_recall_delta: float = 0.0,
    control_f1: float = 0.70,
    tp_breaks: int = 0,
    fp_reduction: int = 4,
) -> dict[str, object]:
    return {
        "control": {
            "predicted_support": [300, 100, 300, 600, 543],
            "per_class_f1": [0.90, control_f1, 0.90, 0.90, 0.90],
        },
        "candidate": {
            "predicted_support": [301, 99, 300, 600, 543],
            "per_class_f1": [
                0.90,
                control_f1 + class1_f1_delta,
                0.90,
                0.90,
                0.90,
            ],
        },
        "delta": {
            "macro_f1": 0.001,
            "class1_f1": class1_f1_delta,
            "class1_precision": class1_precision_delta,
            "class1_recall": class1_recall_delta,
        },
        "transitions": {
            "restricted_focus_fp_reduction": fp_reduction,
            "focus_fn_rescue": 1,
            "focus_tp_break": tp_breaks,
            "candidate_correction": 4,
            "candidate_harm": 1,
        },
        "maximum_nonfocus_f1_drop": 0.005,
    }


def _illumination_payload(
    condition: str,
    *,
    control_f1: float,
    class1_f1_delta: float = 0.012,
) -> dict[str, object]:
    value = _comparison_payload(
        class1_f1_delta=class1_f1_delta,
        control_f1=control_f1,
    )
    value["condition"] = condition
    return value


def test_stage_a_gate_passes_complete_synthetic_case() -> None:
    raw_candidate = _comparison_payload(class1_f1_delta=0.006)
    raw_erm = _comparison_payload(class1_f1_delta=-0.020, tp_breaks=1)
    raw_vrex = _comparison_payload(class1_f1_delta=-0.020, tp_breaks=1)
    erm_candidate = _comparison_payload(class1_f1_delta=0.006)
    vrex_candidate = _comparison_payload(class1_f1_delta=0.006)
    margin_candidate = _comparison_payload(class1_f1_delta=-0.001)
    conditions = ("lighting_dim", "lighting_bright", "low_contrast")
    illumination_raw = [
        _illumination_payload(condition, control_f1=0.50 + 0.05 * index)
        for index, condition in enumerate(conditions)
    ]
    illumination_erm = [
        _illumination_payload(condition, control_f1=0.49 + 0.05 * index)
        for index, condition in enumerate(conditions)
    ]
    illumination_vrex = [
        _illumination_payload(condition, control_f1=0.49 + 0.05 * index)
        for index, condition in enumerate(conditions)
    ]
    risk_table = {
        "boundary_erm": {"mean": 0.50, "worst": 0.60},
        "boundary_cagrad": {"mean": 0.49, "worst": 0.59},
    }
    result = assess_stage_a(
        structural_checks={"structural": True},
        clean_raw_candidate=raw_candidate,
        clean_raw_erm=raw_erm,
        clean_raw_vrex=raw_vrex,
        clean_erm_candidate=erm_candidate,
        clean_vrex_candidate=vrex_candidate,
        clean_margin_candidate=margin_candidate,
        illumination_raw_candidate=illumination_raw,
        illumination_erm_candidate=illumination_erm,
        illumination_vrex_candidate=illumination_vrex,
        risk_table=risk_table,
    )
    assert result["all_gates_passed"] is True
    assert result["stage_b_authorized"] is True


def test_stage_a_gate_rejects_geometry_behavior_and_risk_failures() -> None:
    raw_candidate = _comparison_payload(
        class1_f1_delta=-0.010,
        class1_precision_delta=-0.010,
        tp_breaks=2,
        fp_reduction=-1,
    )
    comparator = _comparison_payload(class1_f1_delta=-0.020, tp_breaks=1)
    candidate_comparator = _comparison_payload(class1_f1_delta=-0.001)
    conditions = ("lighting_dim", "lighting_bright", "low_contrast")
    illumination = [
        _illumination_payload(
            condition,
            control_f1=0.50,
            class1_f1_delta=-0.020,
        )
        for condition in conditions
    ]
    risk_table = {
        "boundary_erm": {"mean": 0.50, "worst": 0.60},
        "boundary_cagrad": {"mean": 0.51, "worst": 0.61},
    }
    result = assess_stage_a(
        structural_checks={"solver": False},
        clean_raw_candidate=raw_candidate,
        clean_raw_erm=comparator,
        clean_raw_vrex=comparator,
        clean_erm_candidate=candidate_comparator,
        clean_vrex_candidate=candidate_comparator,
        clean_margin_candidate=candidate_comparator,
        illumination_raw_candidate=illumination,
        illumination_erm_candidate=illumination,
        illumination_vrex_candidate=illumination,
        risk_table=risk_table,
    )
    assert result["all_gates_passed"] is False
    assert "solver" in result["failed_checks"]
    assert "class1_f1_delta_gte_0p005" in result["failed_checks"]
    assert "candidate_worst_boundary_risk_lte_erm" in result["failed_checks"]
