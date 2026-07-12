from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_dinov2_dense_patch_readiness import (
    dino_dense_patch_descriptors,
)
from trkh.tools.audit_dinov3_dense_patch_readiness import (
    DINO_V2_MODEL,
    DINO_V3_MODEL,
    _load_variant_probabilities,
    _parse_args,
    _prepare_output_dir,
    _protocol,
    _validate_dinov2_summary,
    assess_dinov3_readiness,
    select_oof_feature,
)


def test_dino_descriptor_excludes_register_tokens_from_patch_statistics() -> None:
    cls = torch.tensor([[[10.0, 20.0]]])
    registers = torch.full((1, 4, 2), 999.0)
    patches = torch.arange(1, 33, dtype=torch.float32).reshape(1, 16, 2)
    descriptors = dino_dense_patch_descriptors(
        torch.cat((cls, registers, patches), dim=1),
        grid_size=(4, 4),
        prefix_tokens=5,
        spatial_pool_size=2,
    )
    np.testing.assert_allclose(descriptors["cls"].numpy(), [[10.0, 20.0]])
    np.testing.assert_allclose(
        descriptors["dense_patch"][:, :2].numpy(), patches.mean(dim=1).numpy()
    )
    assert tuple(descriptors["cls_plus_dense_patch"].shape) == (1, 14)
    assert float(descriptors["dense_patch"].max()) < 999.0


def test_dinov3_protocol_defaults_are_locked() -> None:
    args = _parse_args(
        (
            "--classification-root",
            "class_f",
            "--yolo-data",
            "data.yaml",
            "--base-train-csv",
            "train.csv",
            "--base-val-csv",
            "val.csv",
            "--dinov2-audit-dir",
            "dinov2",
            "--output-dir",
            "output",
        )
    )
    protocol = _protocol()
    assert args.model == DINO_V3_MODEL
    assert args.folds == 5
    assert args.batch_size == 128
    assert protocol["candidate_image_size"] == 256
    assert protocol["candidate_prefix_tokens"] == 5
    assert protocol["candidate_selection_uses_validation"] is False
    assert protocol["descriptor_grid_sweep"] is False


def test_dinov3_output_dir_allows_preflight_only_and_rejects_stale_payload(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "audit"
    _prepare_output_dir(output_dir)
    (output_dir / "preflight.json").write_text("{}", encoding="utf-8")
    _prepare_output_dir(output_dir)
    (output_dir / "stale.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale audit payloads"):
        _prepare_output_dir(output_dir)


def test_oof_feature_selection_uses_fixed_objective() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    cls = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32
    )
    dense = np.asarray(
        [[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6]], dtype=np.float32
    )
    selected = select_oof_feature(
        {"cls": cls, "cls_plus_dense_patch": dense},
        labels=labels,
        class_names=("zero", "one"),
    )
    assert selected["selected_feature"] == "cls"
    assert selected["criterion"] == "train_oof_macro_f1_plus_class1_f1"


def _compatible_dinov2_summary() -> dict:
    return {
        "mode": "dinov2_dense_patch_readiness",
        "protocol": {
            "model": DINO_V2_MODEL,
            "folds": 5,
            "seed": 20260712,
            "matched_source_folds": True,
            "primary_readout": "rbf_svc",
        },
        "model": {"model_name": DINO_V2_MODEL},
        "alignment": {"train": {"yolo_rows": 9215}, "val": {"yolo_rows": 2606}},
        "test_split_used": False,
    }


def test_dinov2_reference_validation_is_fail_closed() -> None:
    summary = _compatible_dinov2_summary()
    _validate_dinov2_summary(summary, expected_train_rows=9215, expected_val_rows=2606)
    summary["protocol"]["seed"] = 1
    with pytest.raises(ValueError, match="seed"):
        _validate_dinov2_summary(
            summary, expected_train_rows=9215, expected_val_rows=2606
        )


def test_variant_probability_loader_preserves_sample_identity(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    fields = [
        "sample_index",
        "target",
        "path",
        "cls_prob_0",
        "cls_prob_1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_index": 0,
                "target": 0,
                "path": r"D:\data\train\a.jpg",
                "cls_prob_0": 0.8,
                "cls_prob_1": 0.2,
            }
        )
        writer.writerow(
            {
                "sample_index": 1,
                "target": 1,
                "path": r"D:\data\train\b.jpg",
                "cls_prob_0": 0.1,
                "cls_prob_1": 0.9,
            }
        )
    probabilities = _load_variant_probabilities(
        path,
        variant="cls",
        expected_labels=np.asarray([0, 1]),
        expected_sample_index=np.asarray([0, 1]),
        expected_paths=np.asarray(
            [r"D:\data\train\a.jpg", r"D:\data\train\b.jpg"], dtype=object
        ),
        class_count=2,
    )
    np.testing.assert_allclose(probabilities, [[0.8, 0.2], [0.1, 0.9]])
    with pytest.raises(ValueError, match="Path mismatch"):
        _load_variant_probabilities(
            path,
            variant="cls",
            expected_labels=np.asarray([0, 1]),
            expected_sample_index=np.asarray([0, 1]),
            expected_paths=np.asarray(["wrong.jpg", r"D:\data\train\b.jpg"], dtype=object),
            class_count=2,
        )


def _passing_gate_arguments() -> dict:
    control_oof = {"macro_f1": 0.88, "focus_f1": 0.62}
    candidate_oof = {"macro_f1": 0.90, "focus_f1": 0.66}
    control_val = {"macro_f1": 0.89, "focus_f1": 0.65}
    candidate_val = {
        "macro_f1": 0.91,
        "focus_f1": 0.73,
        "focus_recall": 0.79,
    }
    keeper_val = {
        "macro_f1": 0.884,
        "focus_f1": 0.684,
        "focus_recall": 0.781,
    }
    folds = [
        {"class1_gain": 0.02},
        {"class1_gain": 0.03},
        {"class1_gain": 0.01},
        {"class1_gain": 0.04},
        {"class1_gain": -0.01},
    ]
    transitions = {
        "corrections": 30,
        "harms": 10,
        "class1_fn_rescued": 10,
        "class1_tp_broken": 5,
        "class1_fp_removed": 20,
        "class1_fp_created": 5,
    }
    return {
        "train_samples": 9215,
        "val_samples": 2606,
        "source_groups": 8064,
        "cross_split_source_overlap": 0,
        "descriptor_finite": True,
        "dense_effective_rank": 50.0,
        "energy_entropy": 0.9,
        "energy_border_mass": 0.25,
        "control_oof_metrics": control_oof,
        "candidate_oof_metrics": candidate_oof,
        "control_val_metrics": control_val,
        "candidate_val_metrics": candidate_val,
        "keeper_val_metrics": keeper_val,
        "fold_comparison": folds,
        "keeper_oracle": {"f1": 0.75},
        "train_direction_auc": 0.65,
        "val_direction_auc": 0.66,
        "keeper_transitions": transitions,
    }


def test_dinov3_gate_accepts_only_consistent_recall_safe_gain() -> None:
    result = assess_dinov3_readiness(**_passing_gate_arguments())
    assert result["smoke_permission"] is True
    assert result["failed_checks"] == []


def test_dinov3_gate_rejects_class1_suppression() -> None:
    arguments = _passing_gate_arguments()
    arguments["candidate_val_metrics"] = {
        "macro_f1": 0.90,
        "focus_f1": 0.70,
        "focus_recall": 0.60,
    }
    arguments["keeper_transitions"] = {
        "corrections": 20,
        "harms": 25,
        "class1_fn_rescued": 2,
        "class1_tp_broken": 25,
        "class1_fp_removed": 30,
        "class1_fp_created": 2,
    }
    result = assess_dinov3_readiness(**arguments)
    assert result["smoke_permission"] is False
    assert "keeper_class1_recall_preserved" in result["failed_checks"]
    assert "keeper_validation_class1_recall_protected" in result["failed_checks"]
