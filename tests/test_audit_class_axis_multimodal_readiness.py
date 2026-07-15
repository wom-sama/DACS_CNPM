from __future__ import annotations

import copy

import numpy as np

from trkh.tools.audit_class_axis_multimodal_readiness import (
    AXIS_DIMENSIONS_PER_CLASS,
    GMM_COMPONENTS,
    SEED,
    assess_candidate_readiness,
    balanced_class_axis_assignment,
    class_axis_probabilities,
    multimodal_gmm_probabilities,
    parse_args,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert args.preflight_only is False
    assert AXIS_DIMENSIONS_PER_CLASS == 51
    assert GMM_COMPONENTS == 10
    assert SEED == 20260715
    assert "train_embeddings" in str(args.train_cache)
    assert "val" not in args.train_cache.name.casefold()
    assert "test" not in args.train_cache.name.casefold()


def test_balanced_axis_assignment_recovers_disjoint_synthetic_blocks() -> None:
    rows = []
    labels = []
    for class_index in range(3):
        for sample_index in range(20):
            value = np.full(7, 0.05 + 0.001 * sample_index, dtype=np.float64)
            value[2 * class_index : 2 * class_index + 2] = 3.0 + 0.01 * sample_index
            rows.append(value)
            labels.append(class_index)
    features = np.stack(rows)
    targets = np.asarray(labels, dtype=np.int64)
    assignments, telemetry = balanced_class_axis_assignment(
        features,
        targets,
        class_count=3,
        dimensions_per_class=2,
    )
    assert [value.tolist() for value in assignments] == [[0, 1], [2, 3], [4, 5]]
    assert telemetry["unique_assignment"] is True
    assert telemetry["unassigned_dimensions"] == [6]

    probabilities, probability_telemetry = class_axis_probabilities(
        features,
        features[[0, 20, 40]],
        assignments,
    )
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]
    assert probability_telemetry["probability_sum_max_error"] <= 1e-12


def test_multimodal_gmm_is_deterministic_and_normalized() -> None:
    rng = np.random.default_rng(7)
    first = np.concatenate(
        [rng.normal((-3.0, 0.0), 0.2, size=(40, 2)), rng.normal((-1.0, 0.0), 0.2, size=(40, 2))]
    )
    second = np.concatenate(
        [rng.normal((1.0, 0.0), 0.2, size=(40, 2)), rng.normal((3.0, 0.0), 0.2, size=(40, 2))]
    )
    fit = np.concatenate([first, second])
    labels = np.asarray([0] * len(first) + [1] * len(second), dtype=np.int64)
    evaluate = np.asarray([[-2.8, 0.0], [-1.1, 0.0], [1.1, 0.0], [2.9, 0.0]])

    first_probabilities, first_telemetry = multimodal_gmm_probabilities(
        fit,
        labels,
        evaluate,
        class_count=2,
        components=2,
        seed=11,
    )
    second_probabilities, second_telemetry = multimodal_gmm_probabilities(
        fit,
        labels,
        evaluate,
        class_count=2,
        components=2,
        seed=11,
    )
    assert np.array_equal(first_probabilities, second_probabilities)
    assert first_probabilities.argmax(axis=1).tolist() == [0, 0, 1, 1]
    assert np.allclose(first_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert first_telemetry["all_converged"] is True
    assert first_telemetry == second_telemetry


def _passing_evidence() -> dict[str, object]:
    return {
        "delta_vs_raw": {
            "macro_f1": 0.002,
            "class1_f1": 0.006,
            "class1_precision": 0.012,
            "class1_recall": -0.004,
        },
        "class1_f1_delta_vs_margin_agem": -0.001,
        "transitions_vs_raw": {
            "focus_true_positive_broken": 0,
            "corrections": 8,
            "harms": 3,
        },
        "direction_vs_raw": {"auc_fn_positive": 0.66},
        "restricted_focus_false_positives": {
            "raw": 20,
            "candidate": 15,
            "reduction": 5,
        },
    }


def test_candidate_gate_passes_only_complete_precision_safe_case() -> None:
    result = assess_candidate_readiness(
        _passing_evidence(), structural_checks={"structural": True}
    )
    assert result["shared_trainer_authorized"] is True
    assert result["failed_checks"] == []

    rejected = _passing_evidence()
    rejected["delta_vs_raw"]["class1_recall"] = -0.02
    rejected["transitions_vs_raw"]["focus_true_positive_broken"] = 2
    result = assess_candidate_readiness(
        copy.deepcopy(rejected), structural_checks={"structural": True}
    )
    assert result["shared_trainer_authorized"] is False
    assert "class1_recall_delta_gte_minus_0p005" in result["failed_checks"]
    assert "no_focus_true_positive_broken" in result["failed_checks"]
