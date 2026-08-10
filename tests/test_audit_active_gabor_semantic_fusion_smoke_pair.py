from __future__ import annotations

import copy

from trkh.tools.audit_active_gabor_semantic_fusion_smoke_pair import (
    _is_control_candidate_decision_change,
    _source_group_forensics,
    assess_smoke_pair,
)

import numpy as np


def _metrics(
    *,
    macro_f1: float,
    class1_precision: float,
    class1_recall: float,
    class1_f1: float,
    class1_tp: int,
):
    per_class = [
        {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 549},
        {
            "precision": class1_precision,
            "recall": class1_recall,
            "f1": class1_f1,
            "support": 151,
        },
        {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 544},
        {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 712},
        {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 650},
    ]
    confusion = [[0] * 5 for _ in range(5)]
    confusion[1][1] = class1_tp
    return {"macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": confusion}


def _passing_inputs():
    keeper = _metrics(
        macro_f1=0.882925,
        class1_precision=0.603093,
        class1_recall=0.774834,
        class1_f1=0.678261,
        class1_tp=117,
    )
    control = _metrics(
        macro_f1=0.870,
        class1_precision=0.535,
        class1_recall=0.840,
        class1_f1=0.654,
        class1_tp=127,
    )
    candidate = copy.deepcopy(control)
    candidate["macro_f1"] = 0.876
    candidate["per_class"][0]["f1"] = 0.895
    candidate["per_class"][1].update(
        {"precision": 0.560, "recall": 0.825, "f1": 0.667}
    )
    candidate["per_class"][2]["f1"] = 0.891
    candidate["per_class"][3]["f1"] = 0.892
    candidate["per_class"][4]["f1"] = 0.895
    candidate["confusion_matrix"][1][1] = 125
    transitions = {
        "corrections": 18,
        "harms": 10,
        "focus_false_positives_removed": 12,
        "focus_false_positives_created": 3,
        "focus_false_negatives_rescued": 3,
        "focus_true_positives_broken": 2,
        "new_3_to_2_harms": 2,
    }
    return keeper, control, candidate, transitions


def test_active_gabor_smoke_gate_accepts_only_material_precision_win() -> None:
    keeper, control, candidate, transitions = _passing_inputs()
    result = assess_smoke_pair(
        keeper_metrics=keeper,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions_vs_control=transitions,
        runtime_ratio=1.12,
        peak_vram_gib=2.58,
        aligned_rows=2606,
        probabilities_valid=True,
    )
    assert result["metric_gate_passed"] is True
    assert result["post_smoke_audit_required"] is True
    assert result["ten_epoch_permission"] is False
    assert result["full_train_permission"] is False
    assert result["test_permission"] is False


def test_active_gabor_smoke_gate_rejects_recall_for_precision_trade() -> None:
    keeper, control, candidate, transitions = _passing_inputs()
    candidate["per_class"][1]["recall"] = 0.819
    candidate["confusion_matrix"][1][1] = 119
    transitions["focus_false_negatives_rescued"] = 1
    transitions["focus_true_positives_broken"] = 8
    result = assess_smoke_pair(
        keeper_metrics=keeper,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions_vs_control=transitions,
        runtime_ratio=1.12,
        peak_vram_gib=2.58,
        aligned_rows=2606,
        probabilities_valid=True,
    )
    assert result["metric_gate_passed"] is False
    assert "class1_recall_preserved" in result["failed_checks"]
    assert "class1_tp_retained_95pct" in result["failed_checks"]
    assert "fn_rescues_cover_tp_breaks" in result["failed_checks"]


def test_changed_case_artifact_uses_control_candidate_decisions_only() -> None:
    assert _is_control_candidate_decision_change(1, 0) is True
    assert _is_control_candidate_decision_change(0, 0) is False


def test_active_gabor_source_group_forensics_keeps_multi_object_rows_together() -> None:
    rows = [
        {"key": (0, "source_a", 0)},
        {"key": (1, "source_a", 1)},
        {"key": (2, "source_b", 0)},
    ]
    labels = np.asarray([0, 1, 4], dtype=np.int64)
    control = np.asarray([1, 1, 4], dtype=np.int64)
    candidate = np.asarray([0, 0, 1], dtype=np.int64)
    group_rows, summary = _source_group_forensics(
        rows=rows,
        labels=labels,
        control=control,
        candidate=candidate,
    )
    assert len(group_rows) == 2
    assert summary["multi_object_source_groups"] == 1
    assert summary["focus_fp_removal_source_groups"] == 1
    assert summary["focus_fp_creation_source_groups"] == 1
    assert summary["focus_tp_break_source_groups"] == 1
