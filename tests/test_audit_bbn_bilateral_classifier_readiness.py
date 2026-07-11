from __future__ import annotations

import math

import numpy as np

from trkh.tools.audit_bbn_bilateral_classifier_readiness import (
    FOCUS_MILESTONE,
    FOLDS,
    INFERENCE_ALPHA,
    LOGISTIC_C,
    MAX_ITERATIONS,
    SEED,
    assess_bbn_classifier_readiness,
    bbn_reversed_sample_weights,
    fuse_bilateral_logits,
    parse_args,
)


def test_protocol_defaults_are_locked() -> None:
    args = parse_args(
        [
            "--train-cache",
            "train.npz",
            "--val-cache",
            "val.npz",
            "--output-dir",
            "out",
        ]
    )
    assert args.folds == FOLDS == 5
    assert args.logistic_c == LOGISTIC_C == 0.3
    assert args.max_iterations == MAX_ITERATIONS == 2000
    assert args.inference_alpha == INFERENCE_ALPHA == 0.5
    assert args.seed == SEED == 20260712
    assert FOCUS_MILESTONE == 0.70


def test_reversed_weights_reproduce_inverse_class_probability() -> None:
    labels = np.asarray([0] * 8 + [1] * 2 + [2] * 4, dtype=np.int64)
    weights, summary = bbn_reversed_sample_weights(labels, class_count=3)
    class_mass = np.asarray(
        [weights[labels == index].sum() for index in range(3)],
        dtype=np.float64,
    )
    realized = class_mass / class_mass.sum()
    expected = np.asarray([1 / 8, 1 / 2, 1 / 4], dtype=np.float64)
    expected /= expected.sum()
    assert np.allclose(realized, expected)
    assert np.allclose(realized, summary["target_class_sampling_probabilities"])
    assert math.isclose(float(weights.mean()), 1.0, rel_tol=1e-12)
    assert weights[labels == 1][0] > weights[labels == 2][0] > weights[labels == 0][0]


def test_bilateral_logit_fusion_uses_fixed_inference_alpha() -> None:
    conventional = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    reversed_branch = np.asarray([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    fused = fuse_bilateral_logits(conventional, reversed_branch, alpha=0.5)
    assert np.allclose(fused, np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32))


def _metrics(macro: float, focus_f1: float, focus_recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus_f1, "precision": 0.72, "recall": focus_recall},
        ],
    }


def _transitions() -> dict:
    return {
        "changed": 10,
        "corrections": 7,
        "harms": 2,
        "neutral": 1,
        "focus_false_positive_removed": 4,
        "focus_false_positive_created": 1,
        "focus_false_negative_rescued": 3,
        "focus_true_positive_broken": 1,
    }


def test_gate_accepts_transferable_bilateral_gain() -> None:
    result = assess_bbn_classifier_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=0,
        train_val_source_overlap=0,
        folds_with_focus_gain=4,
        fold_count=5,
        all_models_converged=True,
        oof_conventional=_metrics(0.90, 0.72, 0.78),
        oof_bilateral=_metrics(0.904, 0.73, 0.79),
        val_keeper=_metrics(0.884, 0.686, 0.78),
        val_conventional=_metrics(0.883, 0.690, 0.78),
        val_bilateral=_metrics(0.887, 0.705, 0.79),
        transitions_vs_keeper=_transitions(),
        transitions_vs_conventional=_transitions(),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["image_smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_tail_gain_that_breaks_keeper_recall() -> None:
    transitions = _transitions()
    transitions.update(
        {
            "corrections": 2,
            "harms": 8,
            "focus_false_positive_removed": 6,
            "focus_false_positive_created": 1,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 9,
        }
    )
    result = assess_bbn_classifier_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=0,
        train_val_source_overlap=0,
        folds_with_focus_gain=4,
        fold_count=5,
        all_models_converged=True,
        oof_conventional=_metrics(0.90, 0.72, 0.78),
        oof_bilateral=_metrics(0.904, 0.73, 0.79),
        val_keeper=_metrics(0.884, 0.686, 0.78),
        val_conventional=_metrics(0.883, 0.690, 0.78),
        val_bilateral=_metrics(0.879, 0.71, 0.70),
        transitions_vs_keeper=transitions,
        transitions_vs_conventional=transitions,
        focus_class_index=1,
        test_split_used=False,
    )
    assert not result["image_smoke_permission"]
    assert "val_macro_preserves_keeper_within_0p001" in result["failed_checks"]
    assert "val_focus_recall_preserved_within_0p01" in result["failed_checks"]
    assert "candidate_focus_fn_rescued_ge_tp_broken_vs_keeper" in result["failed_checks"]


def test_gate_rejects_test_access_and_source_overlap() -> None:
    result = assess_bbn_classifier_readiness(
        train_rows=9215,
        val_rows=2606,
        fold_source_overlap=1,
        train_val_source_overlap=1,
        folds_with_focus_gain=4,
        fold_count=5,
        all_models_converged=True,
        oof_conventional=_metrics(0.90, 0.72, 0.78),
        oof_bilateral=_metrics(0.904, 0.73, 0.79),
        val_keeper=_metrics(0.884, 0.686, 0.78),
        val_conventional=_metrics(0.883, 0.690, 0.78),
        val_bilateral=_metrics(0.887, 0.705, 0.79),
        transitions_vs_keeper=_transitions(),
        transitions_vs_conventional=_transitions(),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["image_smoke_permission"]
    assert "test_not_used" in result["failed_checks"]
    assert "source_group_folds_disjoint" in result["failed_checks"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
