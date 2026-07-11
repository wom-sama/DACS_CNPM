from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_two_stage_reedl_readiness import (
    BATCH_SIZE,
    FOLDS,
    LEARNING_RATE,
    PRIOR_WEIGHT,
    SEED,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    WEIGHT_DECAY,
    _load_cache,
    assess_two_stage_reedl_readiness,
    parse_args,
    reedl_projected_probabilities,
)


def test_reedl_projected_probability_matches_reference_formula() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float64)
    probabilities, uncertainty, evidence = reedl_projected_probabilities(
        logits,
        prior_weight=1.0,
    )
    expected_evidence = torch.nn.functional.softplus(logits)
    expected_alpha = expected_evidence + 1.0
    expected_strength = expected_alpha.sum(dim=1)
    assert torch.allclose(evidence, expected_evidence)
    assert torch.allclose(probabilities, expected_alpha / expected_strength[:, None])
    assert torch.allclose(uncertainty, 3.0 / expected_strength)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, dtype=torch.float64))
    assert 0.0 < float(uncertainty.item()) <= 1.0


def test_two_stage_reedl_protocol_defaults_are_locked() -> None:
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
    assert args.stage1_epochs == STAGE1_EPOCHS == 20
    assert args.stage2_epochs == STAGE2_EPOCHS == 10
    assert args.batch_size == BATCH_SIZE == 512
    assert args.learning_rate == LEARNING_RATE == 3e-3
    assert args.weight_decay == WEIGHT_DECAY == 1e-4
    assert args.prior_weight == PRIOR_WEIGHT == 1.0
    assert args.seed == SEED == 20260712
    assert args.device == "cpu"


def _model_payload(macro: float, focus_f1: float, recall: float, auc: float) -> dict:
    per_class = [
        {"f1": macro, "recall": recall},
        {"f1": focus_f1, "recall": recall},
    ]
    return {
        "metrics": {"macro_f1": macro, "per_class": per_class},
        "risk": {
            "error_auroc": auc,
            "aurc": 0.10,
            "coverage_rows": [
                {"coverage": 0.80, "risk": 0.08},
                {"coverage": 1.00, "risk": 0.10},
            ],
        },
    }


def _split_payload(
    keeper: tuple[float, float, float, float],
    control: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> dict:
    return {
        "models": {
            "keeper": _model_payload(*keeper),
            "ce_control": _model_payload(*control),
            "two_stage_reedl": _model_payload(*candidate),
        },
        "direction": {"auc_fn_positive": 0.60},
        "transitions_vs_keeper": {
            "corrections": 3,
            "harms": 1,
            "focus_false_positive_removed": 2,
            "focus_false_positive_created": 1,
            "focus_false_negative_rescued": 2,
            "focus_true_positive_broken": 1,
        },
    }


def test_two_stage_reedl_gate_rejects_non_incremental_candidate() -> None:
    train = _split_payload(
        (0.94, 0.81, 0.98, 0.70),
        (0.90, 0.70, 0.80, 0.68),
        (0.90, 0.70, 0.80, 0.68),
    )
    val = _split_payload(
        (0.884, 0.686, 0.78, 0.71),
        (0.885, 0.688, 0.78, 0.70),
        (0.885, 0.688, 0.78, 0.70),
    )
    result = assess_two_stage_reedl_readiness(
        train_summary=train,
        val_summary=val,
        matched_source_folds=True,
        focus_class_index=1,
    )
    assert not result["smoke_permission"]
    assert "oof_macro_gain_ge_0p002" in result["failed_checks"]
    assert "val_focus_gain_ge_0p015" in result["failed_checks"]
    assert "val_uncertainty_auc_gain_ge_0p02" in result["failed_checks"]


def test_two_stage_reedl_gate_accepts_consistent_incremental_candidate() -> None:
    train = _split_payload(
        (0.90, 0.65, 0.75, 0.69),
        (0.91, 0.68, 0.76, 0.70),
        (0.915, 0.70, 0.77, 0.73),
    )
    val = _split_payload(
        (0.884, 0.686, 0.78, 0.71),
        (0.88, 0.67, 0.77, 0.70),
        (0.89, 0.71, 0.79, 0.73),
    )
    result = assess_two_stage_reedl_readiness(
        train_summary=train,
        val_summary=val,
        matched_source_folds=True,
        focus_class_index=1,
    )
    assert result["smoke_permission"]
    assert result["failed_checks"] == []


def test_cache_loader_rejects_test_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "candidate.npz"
    probabilities = np.asarray([[0.7, 0.3]], dtype=np.float32)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        probabilities=probabilities,
        labels=np.asarray([0], dtype=np.int64),
        base_predictions=np.asarray([0], dtype=np.int64),
        paths=np.asarray(["D:/dataset/images/test/image_1.jpg"], dtype=object),
        sample_index=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="Test rows"):
        _load_cache(cache_path, split="val")
