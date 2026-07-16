from __future__ import annotations

from copy import deepcopy

import pytest

from trkh.tools import audit_foveal_aggregated_attention_pair as common
from trkh.tools.audit_inattentive_token_fusion_pair import (
    EXPECTED_HOLDOUT_ROWS,
    _context_condition_summary,
    _gate_checks,
    _normalize_train_args,
)


def test_normalize_train_args_removes_only_locked_pair_differences() -> None:
    control = ["--run-name", "control", "--epochs", "5", "--seed", "42"]
    candidate = [
        "--run-name",
        "candidate",
        "--epochs",
        "5",
        "--seed",
        "42",
        "--inattentive-token-fusion",
    ]

    assert _normalize_train_args(control) == _normalize_train_args(candidate)
    with pytest.raises(ValueError, match="more than once"):
        _normalize_train_args(
            [*candidate, "--inattentive-token-fusion"]
        )


def test_context_condition_summary_replays_mass_and_cohort_statistics() -> None:
    rows = [
        {
            "tiny_edge_cohort": True,
            "dropped_object_mass_nonzero": True,
            "clean_context_cosine": "",
            "finite_row": True,
            "standard_trace_maximum_error": 1e-7,
            "standard_trace_prediction_match": True,
            "partitions_valid": True,
            "stage1_positive_finite_mass": True,
            "stage2_positive_finite_mass": True,
            "stage1_mass_replay_error": 2e-8,
            "stage2_mass_replay_error": 3e-8,
            "context_norm": 2.0,
            "weighted_object_fraction": 0.4,
            "raw_dropped_object_fraction": 0.3,
            "weighted_object_gain": 0.1,
            "outside_context_fraction": 0.5,
        },
        {
            "tiny_edge_cohort": True,
            "dropped_object_mass_nonzero": False,
            "clean_context_cosine": 0.8,
            "finite_row": True,
            "standard_trace_maximum_error": 2e-7,
            "standard_trace_prediction_match": True,
            "partitions_valid": True,
            "stage1_positive_finite_mass": True,
            "stage2_positive_finite_mass": False,
            "stage1_mass_replay_error": 4e-8,
            "stage2_mass_replay_error": 5e-8,
            "context_norm": 4.0,
            "weighted_object_fraction": 0.2,
            "raw_dropped_object_fraction": 0.2,
            "weighted_object_gain": 0.0,
            "outside_context_fraction": 0.3,
        },
    ]

    summary = _context_condition_summary(rows)

    assert summary["rows"] == 2
    assert summary["stage2_positive_finite_mass_fraction"] == pytest.approx(0.5)
    assert summary["maximum_mass_replay_error"] == pytest.approx(5e-8)
    assert summary["tiny_edge_nonzero_object_mass_fraction"] == pytest.approx(0.5)
    assert summary["weighted_object_gain_mean"] == pytest.approx(0.05)
    assert summary["clean_context_cosine_mean"] == pytest.approx(0.8)


def _passing_gate_payloads() -> dict[str, object]:
    confusion = [[0] * 5 for _ in range(5)]
    confusion[1][1] = 10
    candidate_confusion = deepcopy(confusion)
    candidate_confusion[1][1] = 8
    comparisons = {}
    for condition, _, _ in common.CONDITIONS:
        comparisons[condition] = {
            "control": {"confusion_matrix": confusion},
            "candidate": {"confusion_matrix": candidate_confusion},
            "delta": {
                "macro_f1": 0.0,
                "class1_f1": 0.005,
                "class1_precision": 0.015,
            },
            "transitions": {
                "restricted_focus_fp_control": 40,
                "restricted_focus_fp_reduction": 4,
                "candidate_correction": 5,
                "candidate_harm": 4,
            },
            "maximum_nonfocus_f1_drop": 0.015,
        }
    selectivity_conditions = {
        "clean": {"candidate": {"auroc": 0.65}}
    }
    context_conditions = {}
    for condition, _, _ in common.CONDITIONS:
        selectivity_conditions.setdefault(
            condition, {"candidate": {"auroc": 0.60}}
        )
        context_conditions[condition] = {
            "finite_rows": EXPECTED_HOLDOUT_ROWS,
            "standard_trace_maximum_error": 1e-6,
            "standard_trace_prediction_matches": EXPECTED_HOLDOUT_ROWS,
            "clean_context_cosine_mean": None if condition == "clean" else 0.65,
        }
    context_conditions["clean"].update(
        {
            "partitions_valid_fraction": 1.0,
            "stage1_positive_finite_mass_fraction": 0.95,
            "stage2_positive_finite_mass_fraction": 0.95,
            "maximum_mass_replay_error": 1e-6,
            "tiny_edge_nonzero_object_mass_fraction": 0.20,
            "weighted_object_gain_mean": 0.010,
            "outside_context_fraction_mean": 0.20,
        }
    )
    return {
        "provenance": {"all_checks_pass": True},
        "comparisons": comparisons,
        "cohort": {"delta": {"macro_f1": 0.0, "class1_f1": 0.0}},
        "selectivity": {"conditions": selectivity_conditions},
        "context": {
            "rows": len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
            "conditions": context_conditions,
        },
        "perturbation": {
            "roles": {"candidate": {"object_more_causal": True}}
        },
        "xai": {
            "request_rows": 3,
            "stem_foreground_mass_delta": -0.05,
            "all_event_rows_covered": True,
            "all_representatives_covered": True,
            "maps_finite": True,
            "control_standard_prediction_matches": 3,
            "candidate_standard_prediction_matches": 3,
        },
        "replay": {
            "metrics_exact": True,
            "rows": 2 * len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
        },
    }


def test_gate_checks_enforce_locked_precision_threshold() -> None:
    payloads = _passing_gate_payloads()
    checks = _gate_checks(**payloads)
    assert checks and all(checks.values())

    payloads["comparisons"]["clean"]["delta"]["class1_precision"] = 0.014
    checks = _gate_checks(**payloads)
    assert not checks["clean_class1_precision_delta_gte_0p015"]
    assert checks["mean_condition_class1_precision_delta_gte_0p010"]
