from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from trkh.tools.audit_mcr2_embedding_readiness import (
    BATCH_SIZE,
    EPOCHS,
    EPSILON,
    FEATURE_DIM,
    FOLDS,
    LEARNING_RATE,
    PCA_COMPONENTS,
    _prediction_rows,
    assess_mcr2_readiness,
    maximal_coding_rate_reduction,
    nearest_subspace_probabilities,
    parse_args,
)


def test_protocol_defaults_match_locked_official_readiness_recipe() -> None:
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
    assert args.epochs == EPOCHS == 30
    assert args.batch_size == BATCH_SIZE == 1000
    assert args.feature_dim == FEATURE_DIM == 128
    assert args.pca_components == PCA_COMPONENTS == 30
    assert args.learning_rate == LEARNING_RATE == 0.01
    assert args.epsilon == EPSILON == 0.5


def test_mcr2_loss_matches_dense_membership_formula() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
        ],
        dtype=torch.float64,
    )
    features = torch.nn.functional.normalize(features, dim=1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    loss, global_rate, compressive_rate = maximal_coding_rate_reduction(
        features,
        labels,
        class_count=2,
        epsilon=0.5,
    )
    w = features.T
    identity = torch.eye(3, dtype=torch.float64)
    dense_global = 0.5 * torch.logdet(identity + (3.0 / (4.0 * 0.5)) * (w @ w.T))
    dense_compressive = torch.zeros((), dtype=torch.float64)
    for class_index in range(2):
        membership = torch.diag((labels == class_index).to(dtype=torch.float64))
        dense_compressive = dense_compressive + 0.5 * 0.5 * torch.logdet(
            identity + (3.0 / (2.0 * 0.5)) * (w @ membership @ w.T)
        )
    assert torch.allclose(global_rate, dense_global, atol=1e-10)
    assert torch.allclose(compressive_rate, dense_compressive, atol=1e-10)
    assert torch.allclose(loss, -dense_global + dense_compressive, atol=1e-10)


def test_mcr2_rate_reduction_prefers_correct_subspace_membership() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        dtype=torch.float64,
    )
    features = torch.nn.functional.normalize(features, dim=1).requires_grad_(True)
    correct = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    mixed = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    correct_loss, _, _ = maximal_coding_rate_reduction(features, correct, class_count=2)
    mixed_loss, _, _ = maximal_coding_rate_reduction(features, mixed, class_count=2)
    assert correct_loss < mixed_loss
    correct_loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_nearest_subspace_recovers_two_synthetic_classes() -> None:
    fit = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.6, 0.4],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    evaluate = np.asarray(
        [[0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.7, 0.3]],
        dtype=np.float32,
    )
    probabilities, residuals, protocol = nearest_subspace_probabilities(
        fit,
        labels,
        evaluate,
        class_count=2,
        pca_components=1,
    )
    assert probabilities.argmax(axis=1).tolist() == [0, 1]
    assert residuals.shape == (2, 2)
    assert protocol["effective_components"] == [1, 1]


def test_prediction_rows_expose_canonical_candidate_fields() -> None:
    cache = {
        "sample_index": np.asarray([7], dtype=np.int64),
        "source_stems": np.asarray(["source_7"], dtype=object),
        "paths": np.asarray(["image_7.jpg"], dtype=object),
        "labels": np.asarray([1], dtype=np.int64),
        "probabilities": np.asarray([[0.8, 0.2]], dtype=np.float32),
    }
    control = np.asarray([[0.7, 0.3]], dtype=np.float32)
    candidate = np.asarray([[0.1, 0.9]], dtype=np.float32)
    row = _prediction_rows(
        split="val",
        cache=cache,
        fold_assignment=np.asarray([-1], dtype=np.int64),
        control=control,
        candidate=candidate,
    )[0]
    assert row["target_index"] == row["y_true"] == 1
    assert row["prediction_index"] == row["y_pred"] == 1
    assert row["candidate_prediction_index"] == 1


def _metrics(macro: float, focus: float, recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus, "precision": 0.72, "recall": recall},
        ],
    }


def _transitions(*, winner: bool) -> dict:
    return {
        "changed": 12,
        "corrections": 8 if winner else 3,
        "harms": 3 if winner else 8,
        "neutral": 1,
        "focus_false_positive_removed": 4 if winner else 1,
        "focus_false_positive_created": 1 if winner else 4,
        "focus_false_negative_rescued": 3 if winner else 1,
        "focus_true_positive_broken": 1 if winner else 4,
    }


def _direction(value: Optional[float]) -> dict:
    return {"auc_fn_positive": value, "samples": 20, "false_positives": 10, "false_negatives": 10}


def test_readiness_gate_accepts_recall_safe_mcr2_winner() -> None:
    result = assess_mcr2_readiness(
        train_rows=9215,
        val_rows=2606,
        train_source_groups=8064,
        train_val_source_overlap=0,
        maximum_fold_source_overlap=0,
        all_training_finite=True,
        all_losses_decreased=True,
        minimum_batch_class_support=20,
        folds_with_focus_gain=4,
        fold_count=5,
        oof_control_metrics=_metrics(0.82, 0.61, 0.65),
        oof_candidate_metrics=_metrics(0.84, 0.64, 0.69),
        val_control_metrics=_metrics(0.82, 0.61, 0.65),
        val_candidate_metrics=_metrics(0.889, 0.71, 0.79),
        val_keeper_metrics=_metrics(0.884, 0.686, 0.781),
        oof_transitions=_transitions(winner=True),
        val_transitions=_transitions(winner=True),
        keeper_transitions=_transitions(winner=True),
        oof_direction=_direction(0.68),
        val_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["image_smoke_permission"]
    assert result["failed_checks"] == []


def test_readiness_gate_fails_closed_on_unsafe_or_missing_direction() -> None:
    result = assess_mcr2_readiness(
        train_rows=9215,
        val_rows=2606,
        train_source_groups=8064,
        train_val_source_overlap=1,
        maximum_fold_source_overlap=1,
        all_training_finite=False,
        all_losses_decreased=False,
        minimum_batch_class_support=0,
        folds_with_focus_gain=1,
        fold_count=5,
        oof_control_metrics=_metrics(0.82, 0.61, 0.65),
        oof_candidate_metrics=_metrics(0.80, 0.58, 0.55),
        val_control_metrics=_metrics(0.82, 0.61, 0.65),
        val_candidate_metrics=_metrics(0.80, 0.58, 0.55),
        val_keeper_metrics=_metrics(0.884, 0.686, 0.781),
        oof_transitions=_transitions(winner=False),
        val_transitions=_transitions(winner=False),
        keeper_transitions=_transitions(winner=False),
        oof_direction=_direction(None),
        val_direction=_direction(0.40),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["image_smoke_permission"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
    assert "fold_sources_disjoint" in result["failed_checks"]
    assert "all_batches_have_every_class" in result["failed_checks"]
    assert "directions_available" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]
