from __future__ import annotations

import copy

from trkh.tools.audit_learnable_gabor_texture_smoke_pair import assess_smoke_pair


def _metrics(
    *,
    macro_f1: float,
    class1_precision: float,
    class1_recall: float,
    class1_f1: float,
    class1_tp: int,
):
    per_class = [
        {"precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 549},
        {
            "precision": class1_precision,
            "recall": class1_recall,
            "f1": class1_f1,
            "support": 151,
        },
        {"precision": 0.92, "recall": 0.92, "f1": 0.92, "support": 544},
        {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 712},
        {"precision": 0.96, "recall": 0.96, "f1": 0.96, "support": 650},
    ]
    confusion = [[0] * 5 for _ in range(5)]
    confusion[1][1] = class1_tp
    return {"macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": confusion}


def test_gabor_smoke_gate_passes_only_locked_precision_candidate() -> None:
    keeper = _metrics(
        macro_f1=0.882925,
        class1_precision=0.603093,
        class1_recall=0.774834,
        class1_f1=0.678261,
        class1_tp=117,
    )
    control = _metrics(
        macro_f1=0.8830,
        class1_precision=0.605,
        class1_recall=0.775,
        class1_f1=0.680,
        class1_tp=117,
    )
    candidate = _metrics(
        macro_f1=0.8860,
        class1_precision=0.620,
        class1_recall=0.781,
        class1_f1=0.690,
        class1_tp=118,
    )
    transitions = {
        "focus_false_positives_removed": 5,
        "corrections": 9,
        "harms": 3,
        "focus_false_negatives_rescued": 2,
        "focus_true_positives_broken": 1,
        "new_3_to_2_harms": 1,
    }
    result = assess_smoke_pair(
        keeper_metrics=keeper,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions_vs_keeper=transitions,
        runtime_ratio=1.10,
        peak_vram_gib=2.6,
        aligned_rows=2606,
        probabilities_valid=True,
    )
    assert result["metric_gate_passed"] is True
    assert result["five_epoch_permission"] is False
    assert result["post_smoke_audit_required"] is True

    rejected = copy.deepcopy(candidate)
    rejected["per_class"][1]["precision"] = 0.610
    failed = assess_smoke_pair(
        keeper_metrics=keeper,
        control_metrics=control,
        candidate_metrics=rejected,
        transitions_vs_keeper=transitions,
        runtime_ratio=1.10,
        peak_vram_gib=2.6,
        aligned_rows=2606,
        probabilities_valid=True,
    )
    assert failed["metric_gate_passed"] is False
    assert "class1_precision_absolute" in failed["failed_checks"]
