from __future__ import annotations

import numpy as np

from trkh.tools.audit_natural_prior_continuation_smoke_pair import (
    FOCUS_MILESTONE,
    KEEPER_FOCUS_F1,
    KEEPER_MACRO_F1,
    METHOD,
    PROTOCOL_STAGE,
    _transition_stats,
    assess_natural_prior_continuation,
    parse_args,
)


def _metrics(
    *,
    macro_f1: float,
    focus_precision: float,
    focus_recall: float,
    focus_f1: float,
    nonfocus_f1: float = 0.92,
) -> dict[str, object]:
    per_class = []
    for index in range(5):
        if index == 1:
            precision = focus_precision
            recall = focus_recall
            f1 = focus_f1
            support = 151
        else:
            precision = recall = f1 = nonfocus_f1
            support = (549, 151, 544, 712, 650)[index]
        per_class.append(
            {
                "class_index": index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return {"macro_f1": macro_f1, "per_class": per_class}


def test_locked_constants_and_parser() -> None:
    assert METHOD == "natural_prior_full_model_continuation"
    assert PROTOCOL_STAGE == "matched_60b_2e_full_validation"
    assert KEEPER_MACRO_F1 == 0.8829248547554016
    assert KEEPER_FOCUS_F1 == 0.678260862827301
    assert FOCUS_MILESTONE == 0.70
    parsed = parse_args(
        [
            "--keeper-predictions",
            "keeper.csv",
            "--scratch-predictions",
            "scratch.csv",
            "--control-predictions",
            "control.csv",
            "--candidate-predictions",
            "candidate.csv",
            "--control-run-summary",
            "control_summary.json",
            "--candidate-run-summary",
            "candidate_summary.json",
            "--control-resolved-config",
            "control_config.json",
            "--candidate-resolved-config",
            "candidate_config.json",
            "--locked-protocol",
            "protocol.json",
            "--output-dir",
            "out",
        ]
    )
    assert parsed.output_dir.name == "out"


def test_transition_stats_counts_precision_and_recall_actions() -> None:
    labels = np.asarray([0, 2, 4, 1, 1, 3], dtype=np.int64)
    control = np.asarray([1, 1, 4, 0, 1, 3], dtype=np.int64)
    candidate = np.asarray([0, 1, 1, 1, 0, 2], dtype=np.int64)
    result = _transition_stats(labels, control, candidate)
    assert result["changed_decisions"] == 5
    assert result["corrections"] == 2
    assert result["harms"] == 3
    assert result["focus_false_positives_removed"] == 1
    assert result["focus_false_positives_created"] == 1
    assert result["focus_false_negatives_rescued"] == 1
    assert result["focus_true_positives_broken"] == 1
    assert result["new_3_to_2_harms"] == 1


def test_gate_accepts_precision_gain_with_recall_protection() -> None:
    keeper = _metrics(
        macro_f1=0.883,
        focus_precision=0.61,
        focus_recall=0.78,
        focus_f1=0.68,
    )
    scratch = _metrics(
        macro_f1=0.874,
        focus_precision=0.535,
        focus_recall=0.841,
        focus_f1=0.655,
    )
    control = _metrics(
        macro_f1=0.880,
        focus_precision=0.550,
        focus_recall=0.840,
        focus_f1=0.665,
    )
    candidate = _metrics(
        macro_f1=0.886,
        focus_precision=0.610,
        focus_recall=0.839,
        focus_f1=0.706,
        nonfocus_f1=0.919,
    )
    gate = assess_natural_prior_continuation(
        keeper_metrics=keeper,
        scratch_metrics=scratch,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions_vs_control={
            "focus_false_positive_net_reduction": 12,
            "corrections": 20,
            "harms": 8,
            "focus_false_negatives_rescued": 2,
            "focus_true_positives_broken": 1,
        },
        aligned_rows=2606,
        probabilities_valid=True,
        runtime_ratio=1.01,
    )
    assert gate["probe_permission"] is True
    assert gate["failed_checks"] == []
    assert gate["test_permission"] is False


def test_gate_rejects_precision_from_class_suppression() -> None:
    keeper = _metrics(
        macro_f1=0.883,
        focus_precision=0.61,
        focus_recall=0.78,
        focus_f1=0.68,
    )
    scratch = _metrics(
        macro_f1=0.874,
        focus_precision=0.535,
        focus_recall=0.841,
        focus_f1=0.655,
    )
    control = _metrics(
        macro_f1=0.878,
        focus_precision=0.55,
        focus_recall=0.84,
        focus_f1=0.665,
    )
    candidate = _metrics(
        macro_f1=0.879,
        focus_precision=0.65,
        focus_recall=0.70,
        focus_f1=0.674,
    )
    gate = assess_natural_prior_continuation(
        keeper_metrics=keeper,
        scratch_metrics=scratch,
        control_metrics=control,
        candidate_metrics=candidate,
        transitions_vs_control={
            "focus_false_positive_net_reduction": 30,
            "corrections": 25,
            "harms": 20,
            "focus_false_negatives_rescued": 0,
            "focus_true_positives_broken": 20,
        },
        aligned_rows=2606,
        probabilities_valid=True,
        runtime_ratio=1.0,
    )
    assert gate["probe_permission"] is False
    assert "candidate_focus_recall_preserved_vs_scratch" in gate["failed_checks"]
    assert "candidate_focus_reaches_0p70" in gate["failed_checks"]
    assert "focus_fn_rescues_gte_tp_breaks" in gate["failed_checks"]
