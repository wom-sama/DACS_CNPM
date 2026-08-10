import math

import pytest
import torch

from trkh.tools.audit_class1_boundary_vrex_readiness import (
    _balanced_environment_gradient,
    assess_stage_a,
    boundary_risk_from_rows,
    compose_vrex_gradients,
)


def test_balanced_environment_gradient_is_equal_weight_mean() -> None:
    hard = [torch.tensor([2.0, -2.0])]
    positive = [torch.tensor([0.0, 4.0])]
    observed, error = _balanced_environment_gradient(hard, positive, weight=0.5)
    assert torch.equal(observed[0], torch.tensor([1.0, 1.0]))
    assert error == 0.0


def test_vrex_composition_matches_population_variance_equation() -> None:
    risks = [1.0, 2.0, 3.0, 4.0]
    gradients = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0])],
        [torch.tensor([1.0, 1.0])],
        [torch.tensor([-1.0, 2.0])],
    ]
    erm, variance, vrex, telemetry = compose_vrex_gradients(
        risks, gradients, beta=1.0
    )
    expected_erm = sum((value[0] for value in gradients), torch.zeros(2)) / 4.0
    expected_variance = (
        -0.75 * gradients[0][0]
        - 0.25 * gradients[1][0]
        + 0.25 * gradients[2][0]
        + 0.75 * gradients[3][0]
    )
    assert torch.allclose(erm[0], expected_erm)
    assert torch.allclose(variance[0], expected_variance)
    assert torch.allclose(vrex[0], expected_erm + expected_variance)
    assert telemetry["risk_mean"] == pytest.approx(2.5)
    assert telemetry["risk_population_variance"] == pytest.approx(1.25)
    assert max(telemetry["flat_max_abs_errors"].values()) == 0.0


def test_vrex_beta_zero_matches_erm() -> None:
    gradients = [[torch.tensor([float(index), 1.0])] for index in range(1, 5)]
    erm, _, vrex, telemetry = compose_vrex_gradients(
        [0.1, 0.2, 0.3, 0.4], gradients, beta=0.0
    )
    assert torch.equal(erm[0], vrex[0])
    assert telemetry["vrex_weights"] == telemetry["erm_weights"]


def test_boundary_risk_uses_equal_cohort_means() -> None:
    rows = [
        {
            "sample_index": 1,
            "target": 0,
            "prob_0": 0.5,
            "prob_1": 0.5,
            "prob_2": 0.0,
            "prob_3": 0.0,
            "prob_4": 0.0,
        },
        {
            "sample_index": 2,
            "target": 1,
            "prob_0": 0.0,
            "prob_1": 0.25,
            "prob_2": 0.75,
            "prob_3": 0.0,
            "prob_4": 0.0,
        },
    ]
    result = boundary_risk_from_rows(
        rows, hard_indices=[1], positive_indices=[2], weight=0.5
    )
    assert result["hard_cross_entropy"] == pytest.approx(-math.log(0.5))
    assert result["positive_cross_entropy"] == pytest.approx(-math.log(0.25))
    assert result["boundary_risk"] == pytest.approx(
        0.5 * (-math.log(0.5) - math.log(0.25))
    )


def _comparison_payload(
    *,
    class1_f1_delta: float = 0.006,
    class1_recall_delta: float = 0.0,
    tp_breaks: int = 0,
    control_f1: float = 0.70,
) -> dict[str, object]:
    return {
        "control": {
            "predicted_support": [300, 100, 300, 600, 543],
            "per_class_f1": [0.90, control_f1, 0.90, 0.90, 0.90],
        },
        "candidate": {
            "predicted_support": [301, 99, 300, 600, 543],
            "per_class_f1": [0.90, control_f1 + class1_f1_delta, 0.90, 0.90, 0.90],
        },
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


def _illumination_payload(
    condition: str,
    *,
    control_f1: float = 0.50,
    class1_f1_delta: float = 0.012,
):
    value = _comparison_payload(
        class1_f1_delta=class1_f1_delta, control_f1=control_f1
    )
    value["condition"] = condition
    return value


def test_stage_a_gate_passes_complete_synthetic_case() -> None:
    raw_candidate = _comparison_payload(class1_f1_delta=0.006)
    raw_control = _comparison_payload(class1_f1_delta=0.003, tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=0.003)
    margin_candidate = _comparison_payload(class1_f1_delta=-0.001)
    illumination_candidate = [
        _illumination_payload(condition, control_f1=0.50 + index * 0.05)
        for index, condition in enumerate(
            ("lighting_dim", "lighting_bright", "low_contrast")
        )
    ]
    illumination_control = [
        _illumination_payload(
            condition,
            control_f1=0.50 + index * 0.05,
            class1_f1_delta=0.002,
        )
        for index, condition in enumerate(
            ("lighting_dim", "lighting_bright", "low_contrast")
        )
    ]
    risk_table = {
        "boundary_erm": {"mean": 0.50, "population_variance": 0.010},
        "boundary_vrex": {"mean": 0.49, "population_variance": 0.008},
    }
    result = assess_stage_a(
        structural_checks={"structural": True},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        clean_margin_candidate=margin_candidate,
        illumination_raw_candidate=illumination_candidate,
        illumination_raw_control=illumination_control,
        risk_table=risk_table,
    )
    assert result["all_gates_passed"] is True
    assert result["stage_b_authorized"] is True


def test_stage_a_gate_rejects_comparator_and_risk_failures() -> None:
    raw_candidate = _comparison_payload(class1_f1_delta=-0.001, tp_breaks=2)
    raw_control = _comparison_payload(class1_f1_delta=0.003, tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=-0.001)
    margin_candidate = _comparison_payload(class1_f1_delta=-0.003)
    illumination_candidate = [
        _illumination_payload(condition, control_f1=0.50)
        for condition in ("lighting_dim", "lighting_bright", "low_contrast")
    ]
    illumination_control = [
        _illumination_payload(condition, control_f1=0.52)
        for condition in ("lighting_dim", "lighting_bright", "low_contrast")
    ]
    risk_table = {
        "boundary_erm": {"mean": 0.50, "population_variance": 0.010},
        "boundary_vrex": {"mean": 0.51, "population_variance": 0.011},
    }
    result = assess_stage_a(
        structural_checks={"equation": False},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        clean_margin_candidate=margin_candidate,
        illumination_raw_candidate=illumination_candidate,
        illumination_raw_control=illumination_control,
        risk_table=risk_table,
    )
    assert result["all_gates_passed"] is False
    assert "equation" in result["failed_checks"]
    assert "candidate_f1_gain_vs_erm_gte_0p002" in result["failed_checks"]
    assert "candidate_boundary_risk_variance_lte_90pct_erm" in result["failed_checks"]
