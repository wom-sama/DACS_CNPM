from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (
    CLASSES,
    DINO_FEATURE_DIM,
    DINO_PREFIX_TOKENS,
    FOLDS,
    _TrainLedgerDataset,
    _assert_output_outside_data_root,
    _dependency_contract,
    _dino_descriptor,
    _export_onnx,
    _fast_f1_from_predictions,
    _parse_args,
    _preflight_check_contract,
    _train_content_contract,
    assess_gate,
    classification_summary,
    fit_oof_readout,
    paired_component_bootstrap,
)


def _separable_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.tile(np.arange(CLASSES, dtype=np.int64), 20)
    folds = np.repeat(np.arange(FOLDS, dtype=np.int64), 20)
    groups = np.arange(labels.size, dtype=np.int64)
    features = np.eye(CLASSES, dtype=np.float32)[labels]
    features = np.concatenate((features, 0.01 * np.arange(labels.size)[:, None]), axis=1)
    return features, labels, folds, groups


def test_parser_locks_train_only_defaults() -> None:
    args = _parse_args(
        [
            "--data",
            "data.yaml",
            "--cache-dir",
            "cache",
            "--assignment-csv",
            "folds.csv",
            "--efficientvim-checkpoint",
            "m1.pth",
            "--output-dir",
            "out",
        ]
    )
    assert args.batch_size == 32
    assert args.workers == 0
    assert not args.preflight_only
    assert args.preflight_artifact is None


def test_output_must_stay_outside_immutable_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "class_f"
    data_root.mkdir()
    with pytest.raises(ValueError, match="outside immutable"):
        _assert_output_outside_data_root(data_root, data_root)
    with pytest.raises(ValueError, match="outside immutable"):
        _assert_output_outside_data_root(data_root / "runs" / "b13", data_root)
    _assert_output_outside_data_root(tmp_path / "project_runs" / "b13", data_root)


def test_train_content_contract_detects_byte_changes(tmp_path: Path) -> None:
    first = tmp_path / "train" / "0" / "a.bin"
    second = tmp_path / "train" / "1" / "b.bin"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    relative = ["train/0/a.bin", "train/1/b.bin"]
    before = _train_content_contract(relative, [first, second])
    assert before["files"] == 2
    assert before["bytes"] == 9
    second.write_bytes(b"BETA")
    after = _train_content_contract(relative, [first, second])
    assert after["sha256"] != before["sha256"]


def test_dino_descriptor_removes_prefix_and_averages_patches() -> None:
    tokens = torch.arange(
        (DINO_PREFIX_TOKENS + 256) * DINO_FEATURE_DIM, dtype=torch.float32
    ).reshape(1, DINO_PREFIX_TOKENS + 256, DINO_FEATURE_DIM)

    class Model:
        @staticmethod
        def forward_features(_images: torch.Tensor) -> torch.Tensor:
            return tokens

    observed = _dino_descriptor(Model(), torch.zeros(1, 3, 256, 256))
    expected = tokens[:, DINO_PREFIX_TOKENS:].mean(dim=1)
    torch.testing.assert_close(observed, expected)


def test_dino_descriptor_rejects_geometry_drift() -> None:
    class Model:
        @staticmethod
        def forward_features(_images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 260, DINO_FEATURE_DIM)

    with pytest.raises(ValueError, match="geometry"):
        _dino_descriptor(Model(), torch.zeros(1, 3, 256, 256))


def test_onnx_export_wraps_optional_model_signature(tmp_path: Path) -> None:
    class OptionalCausalAttention(torch.nn.Module):
        def forward(
            self,
            images: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(
                images,
                images,
                images,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )

    result = _export_onnx(
        OptionalCausalAttention(),
        torch.randn(1, 1, 2, 4),
        tmp_path / "optional_signature.onnx",
    )
    assert result["operator_domains"] == [""]
    assert result["export_signature"] == "single_tensor_forward_wrapper"


def test_dependency_contract_records_installed_onnxruntime_distribution() -> None:
    runtime = _dependency_contract()["onnxruntime"]
    assert runtime["distribution"] in {"onnxruntime", "onnxruntime-gpu"}
    assert runtime["version"] == ort.__version__


def test_preflight_check_contract_uses_positive_safety_conditions() -> None:
    checks = _preflight_check_contract(
        efficientvim_digest=(
            "c04c83b982a9a136cec8dca98c5397540bb7d8b18acaa22e559edca042c937a6"
        ),
        dino_shape=(1, 384),
        efficientvim_shape=(1, 320),
        mobile_passed=True,
    )
    assert checks["validation_not_constructed"]
    assert checks["test_not_constructed"]
    assert all(checks.values())


def test_train_ledger_dataset_preserves_index_and_label(tmp_path: Path) -> None:
    path = tmp_path / "train" / "0" / "sample.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)
    dataset = _TrainLedgerDataset(
        [path],
        np.asarray([3], dtype=np.int64),
        lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1),
    )
    image, label, index = dataset[0]
    assert image.shape == (3, 8, 8)
    assert (label, index) == (3, 0)


def test_matched_oof_readout_is_complete_and_converged() -> None:
    features, labels, folds, _groups = _separable_rows()
    result = fit_oof_readout(features, labels, folds)
    assert result["scores"].shape == (labels.size, CLASSES)
    assert np.isfinite(result["scores"]).all()
    assert result["converged"]
    assert len(result["fold_records"]) == FOLDS
    summary = classification_summary(labels, result["scores"], folds)
    assert summary["accuracy"] > 0.99
    assert summary["class1_f1"] > 0.99
    assert len(summary["folds"]) == FOLDS


def test_fast_f1_matches_perfect_predictions() -> None:
    labels = np.tile(np.arange(CLASSES), 4).astype(np.int64)
    macro, class1 = _fast_f1_from_predictions(labels, labels.copy())
    assert macro == pytest.approx(1.0)
    assert class1 == pytest.approx(1.0)


def test_component_bootstrap_is_deterministic_and_fold_safe() -> None:
    _features, labels, folds, groups = _separable_rows()
    scores = np.full((labels.size, CLASSES), -2.0, dtype=np.float64)
    scores[np.arange(labels.size), labels] = 2.0
    first = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=scores,
        candidate_scores=scores,
        replicates=8,
        seed=123,
    )
    second = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=scores,
        candidate_scores=scores,
        replicates=8,
        seed=123,
    )
    assert first == second
    assert first["intervals"]["macro_f1_delta"] == {"lower": 0.0, "upper": 0.0}

    harmed_scores = scores.copy()
    class1_rows = np.flatnonzero(labels == 1)
    harmed_scores[class1_rows, 1] = -3.0
    harmed_scores[class1_rows, 0] = 3.0
    harmed = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=scores,
        candidate_scores=harmed_scores,
        replicates=8,
        seed=123,
    )
    assert harmed["intervals"]["class1_f1_delta"]["upper"] < 0.0
    assert harmed["intervals"]["mean_pair_auroc_delta"]["upper"] < 0.0

    bad_groups = groups.copy()
    bad_groups[20] = bad_groups[0]
    with pytest.raises(ValueError, match="crosses folds"):
        paired_component_bootstrap(
            labels=labels,
            folds=folds,
            groups=bad_groups,
            dino_scores=scores,
            candidate_scores=scores,
            replicates=1,
        )


def _gate_summary(
    *, macro: float, class1: float, recall: float, fp_rate: float, pair_auc: float
) -> dict[str, object]:
    return {
        "macro_f1": macro,
        "class1_f1": class1,
        "class1_recall": recall,
        "restricted_fp_rate": fp_rate,
        "mean_pair_auroc": pair_auc,
        "folds": [
            {"fold": fold, "class1_f1": class1 - 0.002 * fold}
            for fold in range(FOLDS)
        ],
    }


def test_gate_requires_every_preregistered_condition() -> None:
    dino = _gate_summary(
        macro=0.85, class1=0.70, recall=0.75, fp_rate=0.02, pair_auc=0.99
    )
    candidate = _gate_summary(
        macro=0.84, class1=0.68, recall=0.70, fp_rate=0.021, pair_auc=0.987
    )
    bootstrap = {
        "intervals": {
            "mean_pair_auroc_delta": {"lower": -0.008, "upper": 0.001},
            "macro_f1_delta": {"lower": -0.03, "upper": 0.01},
            "class1_f1_delta": {"lower": -0.05, "upper": 0.01},
        }
    }
    passed = assess_gate(
        dino=dino,
        candidate=candidate,
        bootstrap=bootstrap,
        integrity_complete=True,
        readouts_converged=True,
    )
    assert passed["signal_gate_passed"]
    assert passed["fine_tune_protocol_permission"]
    assert not passed["validation_permission"]
    assert not passed["test_permission"]

    failed_candidate = dict(candidate)
    failed_candidate["class1_recall"] = 0.50
    failed = assess_gate(
        dino=dino,
        candidate=failed_candidate,
        bootstrap=bootstrap,
        integrity_complete=True,
        readouts_converged=True,
    )
    assert not failed["signal_gate_passed"]
    assert "class1_recall_retention" in failed["failed_checks"]
