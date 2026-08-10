from trkh.tools.audit_inceptionnext_atto_smoke_pair import assess_smoke_pair


def _metrics(macro_f1: float, class1_precision: float, class1_recall: float, class1_f1: float):
    per_class = [
        {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 1}
        for _ in range(5)
    ]
    per_class[1] = {
        "precision": class1_precision,
        "recall": class1_recall,
        "f1": class1_f1,
        "support": 1,
    }
    return {"macro_f1": macro_f1, "per_class": per_class}


def test_inceptionnext_smoke_gate_accepts_locked_precision_improvement():
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.80, 0.50, 0.70, 0.58),
        candidate_metrics=_metrics(0.802, 0.526, 0.681, 0.591),
        transitions={
            "focus_fp_reduction": 5,
            "focus_tp_breaks": 4,
            "focus_fn_rescues": 1,
            "corrections": 11,
            "harms": 10,
            "new_3_to_2_harms": 5,
        },
        runtime_ratio=1.50,
        stage_a_peak_vram_gib=7.75,
        aligned_rows=2606,
    )
    assert gate["metric_gate_passed"] is True
    assert gate["failed_checks"] == []


def test_inceptionnext_smoke_gate_rejects_precision_runtime_and_tp_failures():
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.80, 0.50, 0.70, 0.58),
        candidate_metrics=_metrics(0.796, 0.524, 0.679, 0.589),
        transitions={
            "focus_fp_reduction": 4,
            "focus_tp_breaks": 6,
            "focus_fn_rescues": 1,
            "corrections": 8,
            "harms": 8,
            "new_3_to_2_harms": 6,
        },
        runtime_ratio=1.501,
        stage_a_peak_vram_gib=7.751,
        aligned_rows=2605,
    )
    assert gate["metric_gate_passed"] is False
    assert "class1_precision_gain" in gate["failed_checks"]
    assert "runtime_ratio_bounded" in gate["failed_checks"]
    assert "focus_tp_breaks_bounded" in gate["failed_checks"]
    assert "full_validation_support" in gate["failed_checks"]
