from __future__ import annotations

import copy

from trkh.tools.audit_ibn_a_shallow_stem_smoke_pair import (
    _deep_differences,
    _normalized_train_arguments,
    assess_ibn_a_smoke_pair,
)


def _metrics(
    *,
    macro_f1: float,
    class1_precision: float,
    class1_recall: float,
    class1_f1: float,
):
    return {
        "macro_f1": macro_f1,
        "per_class": [
            {"f1": 0.90, "precision": 0.90, "recall": 0.90},
            {
                "f1": class1_f1,
                "precision": class1_precision,
                "recall": class1_recall,
            },
            {"f1": 0.90, "precision": 0.90, "recall": 0.90},
            {"f1": 0.90, "precision": 0.90, "recall": 0.90},
            {"f1": 0.90, "precision": 0.90, "recall": 0.90},
        ],
    }


def _passing_inputs():
    control = _metrics(
        macro_f1=0.870,
        class1_precision=0.535,
        class1_recall=0.840,
        class1_f1=0.654,
    )
    candidate = copy.deepcopy(control)
    candidate["macro_f1"] = 0.875
    candidate["per_class"][1].update(
        {"precision": 0.560, "recall": 0.830, "f1": 0.668}
    )
    candidate["per_class"][0]["f1"] = 0.895
    candidate["per_class"][2]["f1"] = 0.891
    candidate["per_class"][3]["f1"] = 0.890
    candidate["per_class"][4]["f1"] = 0.889
    transitions = {
        "corrections": 12,
        "harms": 10,
        "focus_false_positive_net_reduction": 4,
        "focus_false_negatives_rescued": 3,
        "focus_true_positives_broken": 2,
    }
    return control, candidate, transitions


def test_ibn_a_smoke_gate_accepts_exact_protocol_boundaries() -> None:
    control, candidate, transitions = _passing_inputs()
    result = assess_ibn_a_smoke_pair(
        control_metrics=control,
        candidate_metrics=candidate,
        transitions=transitions,
        runtime_ratio=1.30,
        aligned_rows=2606,
        probabilities_valid=True,
        provenance_valid=True,
    )
    assert result["metric_gate_passed"] is True
    assert result["post_smoke_audit_required"] is True
    assert result["five_epoch_probe_permission"] is False
    assert result["full_train_permission"] is False
    assert result["test_permission"] is False
    assert result["route_closed"] is False


def test_ibn_a_smoke_gate_rejects_precision_gain_from_tp_breaks() -> None:
    control, candidate, transitions = _passing_inputs()
    candidate["per_class"][1]["recall"] = 0.820
    transitions["focus_false_negatives_rescued"] = 1
    transitions["focus_true_positives_broken"] = 4
    result = assess_ibn_a_smoke_pair(
        control_metrics=control,
        candidate_metrics=candidate,
        transitions=transitions,
        runtime_ratio=1.10,
        aligned_rows=2606,
        probabilities_valid=True,
        provenance_valid=True,
    )
    assert result["metric_gate_passed"] is False
    assert "class1_recall_preserved" in result["failed_checks"]
    assert "fn_rescues_cover_tp_breaks" in result["failed_checks"]
    assert result["post_smoke_audit_required"] is False
    assert result["route_closed"] is True


def test_ibn_a_smoke_gate_is_conjunctive() -> None:
    control, candidate, transitions = _passing_inputs()
    transitions["corrections"] = 9
    transitions["harms"] = 10
    transitions["focus_false_positive_net_reduction"] = 3
    result = assess_ibn_a_smoke_pair(
        control_metrics=control,
        candidate_metrics=candidate,
        transitions=transitions,
        runtime_ratio=1.301,
        aligned_rows=2606,
        probabilities_valid=True,
        provenance_valid=False,
    )
    assert result["metric_gate_passed"] is False
    assert set(result["failed_checks"]) >= {
        "provenance_and_trace_locked",
        "focus_false_positives_reduced",
        "corrections_cover_harms",
        "runtime_ratio_bounded",
    }


def test_ibn_a_argument_normalization_allows_only_role_values() -> None:
    control = ["--run-name", "control", "--stem-normalization", "batch", "--seed", "42"]
    candidate = [
        "--run-name",
        "candidate",
        "--stem-normalization",
        "ibn_a_first",
        "--seed",
        "42",
    ]
    assert _normalized_train_arguments(control) == _normalized_train_arguments(candidate)
    candidate[-1] = "7"
    assert _normalized_train_arguments(control) != _normalized_train_arguments(candidate)


def test_ibn_a_deep_config_diff_reports_exact_paths() -> None:
    control = {
        "run_name": "control",
        "run_dir": "runs/control",
        "model_config": {"stem_normalization": "batch", "depth": 8},
        "train_config": {"seed": 42},
    }
    candidate = copy.deepcopy(control)
    candidate["run_name"] = "candidate"
    candidate["run_dir"] = "runs/candidate"
    candidate["model_config"]["stem_normalization"] = "ibn_a_first"
    assert _deep_differences(control, candidate) == [
        "model_config.stem_normalization",
        "run_dir",
        "run_name",
    ]
