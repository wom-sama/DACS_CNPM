from __future__ import annotations

from trkh.tools.audit_visual_contrast_smoke_pair import assess_smoke_pair


def _metrics(class1_precision: float, class1_recall: float, class1_f1: float, macro: float):
    per_class = [
        {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10}
        for _ in range(5)
    ]
    per_class[1] = {
        "precision": class1_precision,
        "recall": class1_recall,
        "f1": class1_f1,
        "support": 10,
    }
    return {"macro_f1": macro, "per_class": per_class}


def test_gate_accepts_precision_positive_candidate() -> None:
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.60, 0.85, 0.70, 0.88),
        candidate_metrics=_metrics(0.63, 0.84, 0.72, 0.879),
        transitions={
            "focus_fp_reduction": 7,
            "focus_tp_breaks": 3,
            "focus_fn_rescues": 2,
            "corrections": 20,
            "harms": 12,
            "new_3_to_2_harms": 2,
        },
        runtime_ratio=1.2,
        stage_a_peak_vram_gib=3.0,
        aligned_rows=2606,
    )
    assert gate["metric_gate_passed"]
    assert not gate["five_epoch_permission"]


def test_gate_rejects_precision_recall_and_runtime_regression() -> None:
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.60, 0.85, 0.70, 0.88),
        candidate_metrics=_metrics(0.58, 0.80, 0.67, 0.86),
        transitions={
            "focus_fp_reduction": -2,
            "focus_tp_breaks": 8,
            "focus_fn_rescues": 1,
            "corrections": 5,
            "harms": 20,
            "new_3_to_2_harms": 7,
        },
        runtime_ratio=1.5,
        stage_a_peak_vram_gib=8.0,
        aligned_rows=2500,
    )
    assert not gate["metric_gate_passed"]
    assert "class1_precision_gain" in gate["failed_checks"]
    assert "class1_recall_preserved" in gate["failed_checks"]
    assert "runtime_ratio_bounded" in gate["failed_checks"]
    assert "full_validation_support" in gate["failed_checks"]
