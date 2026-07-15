from __future__ import annotations

from trkh.tools.audit_tokenizer_smoke_pair import assess_smoke_pair


def _metrics(
    macro_f1: float,
    class1_precision: float,
    class1_recall: float,
    class1_f1: float,
) -> dict[str, object]:
    per_class = [
        {"precision": 0.80, "recall": 0.80, "f1": 0.80, "support": 1}
        for _ in range(5)
    ]
    per_class[1] = {
        "precision": class1_precision,
        "recall": class1_recall,
        "f1": class1_f1,
        "support": 1,
    }
    return {"macro_f1": macro_f1, "per_class": per_class}


def _transitions(**overrides: int) -> dict[str, int]:
    values = {
        "focus_fp_reduction": 4,
        "focus_tp_breaks": 2,
        "focus_fn_rescues": 2,
        "corrections": 8,
        "harms": 7,
        "new_3_to_2_harms": 3,
    }
    values.update(overrides)
    return values


def test_octave_smoke_gate_accepts_locked_precision_boundary() -> None:
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.800, 0.600, 0.780, 0.680),
        candidate_metrics=_metrics(0.803, 0.610, 0.770, 0.685),
        transitions=_transitions(),
        runtime_ratio=1.50,
        stage_a_peak_vram_gib=3.25,
        aligned_rows=2606,
        gate_profile="octave_conv_precision",
    )

    assert gate["metric_gate_passed"] is True
    assert gate["profile"] == "octave_conv_precision"
    assert gate["failed_checks"] == []


def test_octave_smoke_gate_rejects_each_strict_boundary_failure() -> None:
    gate = assess_smoke_pair(
        control_metrics=_metrics(0.800, 0.600, 0.780, 0.680),
        candidate_metrics=_metrics(0.8029, 0.6099, 0.7699, 0.6849),
        transitions=_transitions(
            focus_fp_reduction=3,
            focus_tp_breaks=3,
            corrections=7,
            harms=7,
            new_3_to_2_harms=4,
        ),
        runtime_ratio=1.5001,
        stage_a_peak_vram_gib=3.2501,
        aligned_rows=2605,
        gate_profile="octave_conv_precision",
    )

    assert gate["metric_gate_passed"] is False
    assert set(gate["failed_checks"]) == {
        "full_validation_support",
        "macro_f1_preserved",
        "class1_f1_gain",
        "class1_precision_gain",
        "class1_recall_preserved",
        "focus_false_positives_reduced",
        "focus_tp_breaks_bounded",
        "net_corrections_positive",
        "new_3_to_2_harms_bounded",
        "runtime_ratio_bounded",
        "stage_a_vram_bounded",
    }
