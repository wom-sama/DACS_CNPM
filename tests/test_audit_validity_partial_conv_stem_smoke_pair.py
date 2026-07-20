from __future__ import annotations

import copy

import numpy as np

from trkh.tools.audit_validity_partial_conv_stem_smoke_pair import (
    ALLOWED_ROLE_CONFIG_DIFFERENCES,
    _deep_differences,
    _normalized_train_arguments,
    assess_validity_partial_smoke_pair,
    build_mask_derangement,
    build_source_group_folds,
)


def _metrics(
    *,
    macro_f1: float,
    class1_precision: float,
    class1_recall: float,
    class1_f1: float,
) -> dict[str, object]:
    per_class = [
        {
            "class_index": index,
            "precision": 0.90,
            "recall": 0.90,
            "f1": 0.90,
            "support": 100,
        }
        for index in range(5)
    ]
    per_class[1].update(
        {
            "precision": class1_precision,
            "recall": class1_recall,
            "f1": class1_f1,
        }
    )
    return {"macro_f1": macro_f1, "per_class": per_class}


def _passing_gate_inputs() -> dict[str, object]:
    control = _metrics(
        macro_f1=0.880,
        class1_precision=0.600,
        class1_recall=0.780,
        class1_f1=0.680,
    )
    candidate = _metrics(
        macro_f1=0.878,
        class1_precision=0.615,
        class1_recall=0.768,
        class1_f1=0.688,
    )
    placebo = _metrics(
        macro_f1=0.875,
        class1_precision=0.607,
        class1_recall=0.760,
        class1_f1=0.683,
    )
    return {
        "control_metrics": control,
        "candidate_metrics": candidate,
        "placebo_metrics": placebo,
        "candidate_control_transitions": {
            "corrections": 10,
            "harms": 10,
            "focus_false_positive_net_reduction": 5,
            "focus_false_negatives_rescued": 2,
            "focus_true_positives_broken": 4,
        },
        "candidate_placebo_transitions": {
            "focus_false_positive_net_reduction": 3,
        },
        "source_group_summary": {
            "adequate_groups": 5,
            "precision_nonworse_groups": 4,
            "required_nonworse_groups": 4,
            "passed": True,
        },
        "runtime_ratio": 1.35,
        "structural_checks": {
            "full_validation_support": True,
            "provenance_exact": True,
            "no_test_output": True,
        },
    }


def test_validity_partial_smoke_gate_accepts_exact_boundaries() -> None:
    gate = assess_validity_partial_smoke_pair(**_passing_gate_inputs())
    assert gate["metric_gate_passed"] is True
    assert gate["post_smoke_audit_required"] is True
    assert gate["five_epoch_probe_permission"] is False
    assert gate["full_train_permission"] is False
    assert gate["test_permission"] is False
    assert gate["route_closed"] is False


def test_validity_partial_smoke_gate_is_conjunctive() -> None:
    inputs = _passing_gate_inputs()
    inputs["candidate_metrics"]["per_class"][1]["precision"] = 0.6149
    inputs["candidate_control_transitions"]["focus_false_positive_net_reduction"] = 4
    inputs["candidate_placebo_transitions"]["focus_false_positive_net_reduction"] = 2
    inputs["source_group_summary"]["passed"] = False
    inputs["runtime_ratio"] = 1.3501
    inputs["structural_checks"]["provenance_exact"] = False
    gate = assess_validity_partial_smoke_pair(**inputs)
    assert gate["metric_gate_passed"] is False
    assert set(gate["failed_checks"]) >= {
        "class1_precision_gain",
        "restricted_focus_fp_net_reduction",
        "aligned_beats_placebo_restricted_fp",
        "source_group_precision_stable",
        "runtime_ratio_bounded",
        "provenance_exact",
    }
    assert gate["route_closed"] is True


def test_validity_partial_smoke_gate_replay_is_input_order_independent() -> None:
    first = _passing_gate_inputs()
    first["structural_checks"] = {"z_check": False, "a_check": False}
    second = copy.deepcopy(first)
    second["structural_checks"] = {"a_check": False, "z_check": False}
    first_gate = assess_validity_partial_smoke_pair(**first)
    second_gate = assess_validity_partial_smoke_pair(**second)
    assert first_gate == second_gate
    assert first_gate["failed_checks"] == sorted(first_gate["failed_checks"])


def test_mask_derangement_is_bijective_and_changes_source_and_shape() -> None:
    rows_per_class = 24
    labels = np.repeat(np.arange(5), rows_per_class)
    row_ids = np.arange(labels.size)
    sources = np.asarray([f"source_{index}" for index in row_ids], dtype=object)
    invalid_counts = labels * 10_000 + row_ids % rows_per_class
    mask_hashes = np.asarray([f"mask_{index}" for index in row_ids], dtype=object)
    mapping, bins, summary = build_mask_derangement(
        labels, sources, invalid_counts, mask_hashes
    )
    assert summary["passed"] is True
    assert np.unique(mapping).size == labels.size
    assert np.array_equal(labels[mapping], labels)
    assert np.array_equal(bins[mapping], bins)
    assert np.all(sources[mapping] != sources)
    assert np.all(mask_hashes[mapping] != mask_hashes)


def test_mask_derangement_rejects_partition_without_distinct_mask() -> None:
    labels = np.zeros(8, dtype=np.int64)
    sources = np.asarray([f"source_{index}" for index in range(8)], dtype=object)
    invalid_counts = np.arange(8)
    mask_hashes = np.asarray(["same"] * 8, dtype=object)
    try:
        build_mask_derangement(
            labels, sources, invalid_counts, mask_hashes, rank_bins=1
        )
    except RuntimeError as exc:
        assert "No valid mask derangement" in str(exc)
    else:
        raise AssertionError("Expected an impossible mask derangement to fail")


def test_source_group_folds_are_source_disjoint_and_repeatable() -> None:
    labels = np.tile(np.arange(5), 120)
    sources = np.asarray([f"source_{index}" for index in range(labels.size)])
    first_folds, first_summary = build_source_group_folds(labels, sources)
    second_folds, second_summary = build_source_group_folds(labels, sources)
    assert np.array_equal(first_folds, second_folds)
    assert first_summary == second_summary
    assert first_summary["source_overlap"] == 0
    assert sorted(np.unique(first_folds).tolist()) == [0, 1, 2, 3, 4]


def test_argument_normalization_allows_only_matched_role_fields() -> None:
    control = [
        "--run-name",
        "control",
        "--stem-convolution",
        "standard",
        "--data-cartography-output",
        "control.csv",
        "--seed",
        "42",
    ]
    candidate = [
        "--run-name",
        "candidate",
        "--stem-convolution",
        "validity_partial",
        "--data-cartography-output",
        "candidate.csv",
        "--seed",
        "42",
    ]
    assert _normalized_train_arguments(control) == _normalized_train_arguments(
        candidate
    )
    candidate[-1] = "7"
    assert _normalized_train_arguments(control) != _normalized_train_arguments(
        candidate
    )


def test_deep_config_diff_reports_only_exact_paths() -> None:
    control = {
        "data": {
            "data_cartography": {
                "occurrence_output": "control_occurrence.json",
                "output": "control.csv",
            }
        },
        "run_name": "control",
        "run_dir": "runs/control",
        "model_config": {"stem_convolution": "standard", "depth": 8},
        "train_config": {
            "seed": 42,
            "data_cartography_output": "control.csv",
        },
    }
    candidate = copy.deepcopy(control)
    candidate["data"]["data_cartography"]["occurrence_output"] = (
        "candidate_occurrence.json"
    )
    candidate["data"]["data_cartography"]["output"] = "candidate.csv"
    candidate["run_name"] = "candidate"
    candidate["run_dir"] = "runs/candidate"
    candidate["model_config"]["stem_convolution"] = "validity_partial"
    candidate["train_config"]["data_cartography_output"] = "candidate.csv"
    assert _deep_differences(control, candidate) == [
        "data.data_cartography.occurrence_output",
        "data.data_cartography.output",
        "model_config.stem_convolution",
        "run_dir",
        "run_name",
        "train_config.data_cartography_output",
    ]
    assert set(_deep_differences(control, candidate)) == set(
        ALLOWED_ROLE_CONFIG_DIFFERENCES
    )
