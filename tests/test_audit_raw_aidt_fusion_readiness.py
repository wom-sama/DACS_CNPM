from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch

from trkh.tools.audit_raw_aidt_fusion_readiness import (
    BATCH_SIZE,
    EPOCHS,
    FOLDS,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    RESNET_DIM,
    VIT_DIM,
    WEIGHT_DECAY,
    AIDTReadout,
    _align_keeper_probabilities,
    _prediction_rows,
    _selected_control,
    assess_raw_aidt_fusion_readiness,
    normalize_component_features,
    parse_args,
    split_component_features,
    sqrt_inverse_class_weights,
    train_readout,
)


def test_protocol_defaults_match_locked_raw_aidt_readiness_recipe() -> None:
    args = parse_args(
        [
            "--train-npz",
            "train.npz",
            "--val-npz",
            "val.npz",
            "--keeper-val-cache",
            "keeper.npz",
            "--output-dir",
            "out",
        ]
    )
    assert args.folds == FOLDS == 5
    assert args.epochs == EPOCHS == 30
    assert args.batch_size == BATCH_SIZE == 512
    assert args.resnet_dim == RESNET_DIM == 2048
    assert args.vit_dim == VIT_DIM == 768
    assert args.hidden_dim_1 == HIDDEN_DIM_1 == 1024
    assert args.hidden_dim_2 == HIDDEN_DIM_2 == 512
    assert args.learning_rate == LEARNING_RATE == 3e-4
    assert args.weight_decay == WEIGHT_DECAY == 0.05
    assert args.label_smoothing == LABEL_SMOOTHING == 0.05
    assert args.feature_normalization == "raw"


def test_aidt_readout_matches_local_head_shape_and_parameter_count() -> None:
    model = AIDTReadout(2816, 5)
    logits = model.eval()(torch.zeros((2, 2816), dtype=torch.float32))
    assert logits.shape == (2, 5)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_415_045


def test_split_component_features_preserves_exact_column_order() -> None:
    features = np.arange(30, dtype=np.float32).reshape(3, 10)
    views = split_component_features(features, resnet_dim=6, vit_dim=4)
    assert np.array_equal(views["resnet"], features[:, :6])
    assert np.array_equal(views["vit"], features[:, 6:])
    assert np.shares_memory(views["fusion"], features)
    with pytest.raises(ValueError, match="dimension mismatch"):
        split_component_features(features, resnet_dim=5, vit_dim=4)


def test_block_l2_normalization_equalizes_branch_norm_without_mixing_columns() -> None:
    features = np.asarray(
        [[3.0, 4.0, 0.0, 6.0, 8.0], [0.0, 5.0, 12.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    views = split_component_features(features, resnet_dim=3, vit_dim=2)
    normalized = normalize_component_features(views, mode="block_l2")
    assert np.allclose(np.linalg.norm(normalized["resnet"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(normalized["vit"], axis=1), 1.0)
    assert np.allclose(normalized["fusion"][:, :3], normalized["resnet"])
    assert np.allclose(normalized["fusion"][:, 3:], normalized["vit"])
    assert np.allclose(np.linalg.norm(normalized["fusion"], axis=1), np.sqrt(2.0))


def test_sqrt_inverse_weights_have_mean_one_and_upweight_rare_class() -> None:
    labels = np.asarray([0] * 16 + [1] * 4 + [2], dtype=np.int64)
    weights = sqrt_inverse_class_weights(labels, 3)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[2] > weights[1] > weights[0]


def test_small_readout_training_is_finite_and_does_not_write_a_model() -> None:
    generator = np.random.default_rng(7)
    labels = np.tile(np.arange(5, dtype=np.int64), 16)
    features = generator.normal(size=(len(labels), 12)).astype(np.float32)
    model, curve, telemetry = train_readout(
        features,
        labels,
        class_count=5,
        hidden_dim_1=16,
        hidden_dim_2=16,
        epochs=2,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_epochs=1,
        label_smoothing=0.0,
        seed=11,
        device=torch.device("cpu"),
    )
    assert isinstance(model, AIDTReadout)
    assert len(curve) == telemetry["epochs"] == 2
    assert telemetry["finite"]
    assert all(np.isfinite(row["loss"]) for row in curve)


def test_keeper_alignment_uses_sample_index_and_rejects_duplicates() -> None:
    feature_cache = {
        "sample_index": np.asarray([7, 3], dtype=np.int64),
        "labels": np.asarray([1, 0], dtype=np.int64),
    }
    keeper_cache = {
        "sample_index": np.asarray([3, 7], dtype=np.int64),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "probabilities": np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
    }
    aligned = _align_keeper_probabilities(feature_cache, keeper_cache)
    assert np.allclose(aligned, [[0.2, 0.8], [0.9, 0.1]])
    keeper_cache["sample_index"] = np.asarray([3, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="not unique"):
        _align_keeper_probabilities(feature_cache, keeper_cache)


def test_control_selection_uses_only_oof_macro_and_focus_score() -> None:
    metrics = {
        "resnet": _metrics(0.84, 0.61, 0.66),
        "vit": _metrics(0.82, 0.66, 0.68),
    }
    assert _selected_control(metrics, focus_class_index=1) == "vit"


def test_prediction_rows_expose_canonical_fusion_fields() -> None:
    cache = {
        "sample_index": np.asarray([7], dtype=np.int64),
        "source_stem": np.asarray(["source_7"], dtype=object),
        "object_index": np.asarray([2], dtype=np.int64),
        "paths": np.asarray(["image_7.jpg"], dtype=object),
        "labels": np.asarray([1], dtype=np.int64),
    }
    probabilities = {
        "resnet": np.asarray([[0.7, 0.3]], dtype=np.float32),
        "vit": np.asarray([[0.6, 0.4]], dtype=np.float32),
        "fusion": np.asarray([[0.1, 0.9]], dtype=np.float32),
    }
    row = _prediction_rows(
        split="val",
        cache=cache,
        fold_assignment=np.asarray([-1], dtype=np.int64),
        probabilities=probabilities,
        selected_control="vit",
    )[0]
    assert row["target_index"] == row["y_true"] == 1
    assert row["prediction_index"] == row["y_pred"] == 1
    assert row["fusion_prediction_index"] == 1


def _metrics(macro: float, focus: float, recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus, "precision": 0.76, "recall": recall},
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
    return {
        "auc_fn_positive": value,
        "samples": 20,
        "false_positives": 10,
        "false_negatives": 10,
    }


def _gate_inputs(*, winner: bool) -> dict:
    return {
        "train_rows": 9215,
        "val_rows": 2606,
        "train_source_groups": 8064,
        "train_val_source_overlap": 0 if winner else 1,
        "maximum_fold_source_overlap": 0 if winner else 1,
        "all_training_finite": winner,
        "all_losses_decreased": winner,
        "minimum_batch_class_support": 20 if winner else 0,
        "folds_fusion_wins_focus": 4 if winner else 1,
        "fold_count": 5,
        "oof_control_metrics": _metrics(0.82, 0.61, 0.65),
        "oof_fusion_metrics": _metrics(0.84 if winner else 0.80, 0.64 if winner else 0.58, 0.69),
        "val_control_metrics": _metrics(0.82, 0.61, 0.65),
        "val_fusion_metrics": _metrics(0.889 if winner else 0.80, 0.71 if winner else 0.58, 0.79 if winner else 0.55),
        "val_keeper_metrics": _metrics(0.884, 0.686, 0.781),
        "oof_transitions": _transitions(winner=winner),
        "val_control_transitions": _transitions(winner=winner),
        "val_keeper_transitions": _transitions(winner=winner),
        "oof_direction": _direction(0.68 if winner else None),
        "val_direction": _direction(0.66 if winner else 0.40),
        "focus_class_index": 1,
        "test_split_used": not winner,
    }


def test_readiness_gate_accepts_fold_safe_raw_aidt_winner() -> None:
    result = assess_raw_aidt_fusion_readiness(**_gate_inputs(winner=True))
    assert result["fold_safe_teacher_permission"]
    assert result["failed_checks"] == []


def test_readiness_gate_fails_closed_on_leakage_and_missing_direction() -> None:
    result = assess_raw_aidt_fusion_readiness(**_gate_inputs(winner=False))
    assert not result["fold_safe_teacher_permission"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
    assert "fold_sources_disjoint" in result["failed_checks"]
    assert "directions_available" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]
