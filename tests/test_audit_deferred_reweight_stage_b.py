from __future__ import annotations

from copy import deepcopy

from trkh.tools.audit_deferred_reweight_stage_b import (
    _per_class_from_confusion,
    assess_clean,
    assess_robustness,
    assess_trace,
    assess_xai,
)


def _metrics(confusion: list[list[int]]) -> dict[str, object]:
    per_class = _per_class_from_confusion(confusion)
    return {
        "macro_f1": sum(float(row["f1"]) for row in per_class) / 5.0,
        "per_class": per_class,
    }


def test_clean_gate_accepts_balanced_precision_and_recall() -> None:
    confusion = [
        [499, 50, 0, 0, 0],
        [51, 100, 0, 0, 0],
        [0, 0, 544, 0, 0],
        [0, 0, 0, 712, 0],
        [0, 0, 0, 0, 650],
    ]
    result = assess_clean(_metrics(confusion), confusion)
    assert result["passed"] is True
    assert result["class1"]["false_positives"] == 50


def test_clean_gate_rejects_natural_prior_class1_suppression() -> None:
    confusion = [
        [521, 9, 11, 0, 8],
        [70, 30, 40, 0, 11],
        [5, 4, 485, 40, 10],
        [0, 1, 63, 646, 2],
        [42, 11, 23, 3, 571],
    ]
    result = assess_clean(_metrics(confusion), confusion)
    assert result["passed"] is False
    assert result["class1"]["false_positives"] == 25
    assert "class1_recall_at_least_0_635232" in result["failed_checks"]
    assert "class1_precision_at_least_0_403702" not in result["failed_checks"]


def _robustness(macro: float, class1_tp: int) -> dict[str, object]:
    confusion = [
        [549, 0, 0, 0, 0],
        [151 - class1_tp, class1_tp, 0, 0, 0],
        [0, 0, 544, 0, 0],
        [0, 0, 0, 712, 0],
        [0, 0, 0, 0, 650],
    ]
    return {
        "conditions": {
            name: {"metrics": {"macro_f1": macro, "confusion_matrix": confusion}}
            for name in (
                "clean",
                "occlusion_center",
                "lighting_dim",
                "lighting_bright",
                "low_contrast",
            )
        }
    }


def test_robustness_gate_uses_existing_three_of_five_and_tp_rule() -> None:
    keeper = _robustness(0.80, 100)
    candidate = _robustness(0.81, 99)
    assert assess_robustness(candidate, keeper)["passed"] is True
    rejected = _robustness(0.79, 97)
    result = assess_robustness(rejected, keeper)
    assert result["passed"] is False
    assert result["macro_f1_win_count"] == 0


def _paired_xai() -> dict[str, object]:
    deltas = {
        "gradcam_foreground_mass": -0.01,
        "grad_rollout_foreground_mass": -0.02,
        "background_gray_original_prediction_drop": 0.01,
        "background_blur_original_prediction_drop": 0.01,
        "center_occlusion_original_prediction_drop": 0.02,
    }
    return {
        "cases": 12,
        "attention_provenance": {"known_sources_only": True},
        "contact_sheets": {"all": {"missing_tiles": 0}},
        "categories": {"all": {"candidate_minus_keeper": deltas}},
    }


def test_xai_gate_rejects_material_foreground_regression() -> None:
    assert assess_xai(_paired_xai())["passed"] is True
    rejected = deepcopy(_paired_xai())
    rejected["categories"]["all"]["candidate_minus_keeper"][
        "gradcam_foreground_mass"
    ] = -0.051
    result = assess_xai(rejected)
    assert result["passed"] is False
    assert "gradcam_foreground_drop_at_most_0_05" in result["failed_checks"]


def test_trace_requires_exactly_one_sample_per_class() -> None:
    passed = assess_trace({"samples": [{"class_id": value} for value in range(5)]})
    assert passed["passed"] is True
    failed = assess_trace({"samples": [{"class_id": value} for value in (0, 1, 1, 3, 4)]})
    assert failed["passed"] is False
