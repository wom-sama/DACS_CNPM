from pathlib import Path

import pytest
import torch
from torch import nn

from trkh.tools.audit_class1_reference_agem_readiness import (
    _apply_normalized_parameter_step,
    _select_cohorts,
    assess_stage_a,
    project_agem_gradient,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow


def test_active_agem_projection_matches_equation() -> None:
    current = [torch.tensor([1.0, -2.0]), torch.tensor([0.5])]
    reference = [torch.tensor([1.0, 1.0]), torch.tensor([0.0])]

    projected, telemetry = project_agem_gradient(current, reference)

    observed = torch.cat([value.flatten() for value in projected])
    expected = torch.tensor([1.5, -1.5, 0.5])
    assert torch.equal(observed, expected)
    assert telemetry["violation_active"] is True
    assert telemetry["raw_dot"] == pytest.approx(-1.0)
    assert telemetry["projected_dot"] == pytest.approx(0.0, abs=1e-12)
    assert telemetry["direct_flat_max_abs_error"] == 0.0


def test_nonconflicting_agem_gradient_is_unchanged() -> None:
    current = [torch.tensor([2.0, 0.0])]
    reference = [torch.tensor([1.0, 0.0])]

    projected, telemetry = project_agem_gradient(current, reference)

    assert torch.equal(projected[0], current[0])
    assert telemetry["violation_active"] is False
    assert telemetry["projection_coefficient"] == 0.0


def test_agem_rejects_zero_reference_gradient() -> None:
    with pytest.raises(ValueError, match="reference gradient has zero norm"):
        project_agem_gradient(
            [torch.tensor([1.0, 2.0])],
            [torch.tensor([0.0, 0.0])],
        )


def test_normalized_steps_have_exact_matched_ratio() -> None:
    left = nn.Linear(3, 2, bias=True).double()
    right = nn.Linear(3, 2, bias=True).double()
    right.load_state_dict(left.state_dict())
    direction_left = [torch.ones_like(value) for value in left.parameters()]
    direction_right = [
        torch.arange(1, value.numel() + 1, dtype=value.dtype).reshape(value.shape)
        for value in right.parameters()
    ]

    left_step = _apply_normalized_parameter_step(
        left, direction_left, parameter_ratio=1e-4
    )
    right_step = _apply_normalized_parameter_step(
        right, direction_right, parameter_ratio=1e-4
    )

    assert left_step["actual_ratio"] == pytest.approx(1e-4, abs=1e-12)
    assert right_step["actual_ratio"] == pytest.approx(1e-4, abs=1e-12)
    assert left_step["actual_step_norm"] == pytest.approx(
        right_step["actual_step_norm"], rel=1e-12
    )


def _comparison_payload(
    *,
    class1_f1_delta: float = 0.006,
    macro_delta: float = 0.001,
    tp_breaks: int = 0,
) -> dict[str, object]:
    return {
        "control": {
            "predicted_support": [300, 100, 300, 600, 543],
            "per_class_f1": [0.98, 0.97, 0.98, 0.99, 0.98],
        },
        "candidate": {
            "predicted_support": [301, 99, 300, 600, 543],
            "per_class_f1": [0.98, 0.976, 0.98, 0.99, 0.98],
        },
        "delta": {
            "macro_f1": macro_delta,
            "class1_f1": class1_f1_delta,
            "class1_precision": 0.010,
            "class1_recall": 0.0,
        },
        "transitions": {
            "restricted_focus_fp_reduction": 2,
            "focus_fn_rescue": 1,
            "focus_tp_break": tp_breaks,
            "candidate_correction": 3,
            "candidate_harm": 1,
        },
        "maximum_nonfocus_f1_drop": 0.005,
    }


def test_assess_stage_a_passes_complete_synthetic_gate() -> None:
    raw_candidate = _comparison_payload()
    raw_control = _comparison_payload(class1_f1_delta=0.004, tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=0.002)
    illumination = [_comparison_payload() for _ in range(3)]

    result = assess_stage_a(
        structural_checks={"structural": True},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        illumination_raw_candidate=illumination,
    )

    assert result["all_gates_passed"] is True
    assert result["stage_b_authorized"] is True
    assert result["full_train_authorized"] is False


def test_assess_stage_a_rejects_raw_collapse_and_structural_failure() -> None:
    raw_candidate = _comparison_payload(macro_delta=-0.001)
    raw_control = _comparison_payload(tp_breaks=1)
    control_candidate = _comparison_payload(class1_f1_delta=0.010)

    result = assess_stage_a(
        structural_checks={"active_conflict": False},
        clean_raw_candidate=raw_candidate,
        clean_raw_control=raw_control,
        clean_control_candidate=control_candidate,
        illumination_raw_candidate=[_comparison_payload() for _ in range(3)],
    )

    assert result["all_gates_passed"] is False
    assert "active_conflict" in result["failed_checks"]
    assert "macro_f1_delta_nonnegative" in result["failed_checks"]


def test_select_cohorts_uses_only_fit_class1_and_verified_restricted_fp() -> None:
    def row(index: int, *, fold: int, target: int, prediction: int) -> CleanTrainRow:
        path = Path(f"D:/dataset/train/{index}.jpg")
        return CleanTrainRow(
            sample_index=index,
            source_stem=str(index),
            image_path=path,
            fold=fold,
            target=target,
            keeper_prediction=prediction,
            keeper_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
        )

    rows = [
        row(0, fold=0, target=0, prediction=1),
        row(1, fold=1, target=1, prediction=1),
        row(2, fold=2, target=0, prediction=1),
        row(3, fold=3, target=3, prediction=1),
        row(4, fold=4, target=2, prediction=0),
        row(5, fold=4, target=4, prediction=1),
    ]

    result = _select_cohorts(rows, fold=0, focus_class=1)

    assert result["holdout_indices"] == [0]
    assert result["reference_indices"] == [1]
    assert result["hard_indices"] == [2, 5]
    assert result["holdout_restricted_fp"] == 1
