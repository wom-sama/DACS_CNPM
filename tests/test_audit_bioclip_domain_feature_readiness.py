from __future__ import annotations

import numpy as np
import pytest

from trkh.tools.audit_bioclip_domain_feature_readiness import (
    BATCH_SIZE,
    BIOCLIP_REVISION,
    BIOCLIP_SHA256,
    CONTROL_REVISION,
    CONTROL_SHA256,
    EMBEDDING_DIMENSION,
    FOLDS,
    LOGISTIC_C,
    MAX_ITERATIONS,
    OCCLUSION_GRID,
    SEED,
    XAI_CASES,
    _model_spec,
    _normalized_source_group,
    _assert_aligned_class_names,
    _validate_protocol,
    assess_bioclip_domain_readiness,
    paired_feature_shift,
    parse_args,
    select_xai_cases,
)


def test_protocol_defaults_and_revisions_are_locked() -> None:
    args = parse_args(
        [
            "--classification-root",
            "class_f",
            "--yolo-data",
            "data.yaml",
            "--keeper-train-cache",
            "train.npz",
            "--keeper-val-cache",
            "val.npz",
            "--output-dir",
            "out",
        ]
    )
    assert args.batch_size == BATCH_SIZE == 64
    assert args.folds == FOLDS == 5
    assert args.logistic_c == LOGISTIC_C == 0.3
    assert args.max_iterations == MAX_ITERATIONS == 400
    assert args.seed == SEED == 20260712
    assert args.occlusion_grid == OCCLUSION_GRID == 4
    assert args.xai_cases == XAI_CASES == 12
    assert EMBEDDING_DIMENSION == 512
    assert BIOCLIP_REVISION == "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c"
    assert CONTROL_REVISION == "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d"
    assert BIOCLIP_SHA256 == "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"
    assert CONTROL_SHA256 == "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"


def test_model_specs_use_equal_capacity_official_encoders() -> None:
    control = _model_spec("control")
    bioclip = _model_spec("bioclip")
    assert control["model_name"] == "ViT-B-16-quickgelu"
    assert bioclip["model_name"] == "ViT-B-16"
    assert control["revision"] == CONTROL_REVISION
    assert bioclip["revision"] == BIOCLIP_REVISION
    assert control["checkpoint_sha256"] == CONTROL_SHA256
    assert bioclip["checkpoint_sha256"] == BIOCLIP_SHA256
    assert _normalized_source_group("Image_1") == _normalized_source_group("image_1")


def test_protocol_rejects_non_class1_focus(tmp_path) -> None:
    args = parse_args(
        [
            "--classification-root",
            str(tmp_path / "class_f"),
            "--yolo-data",
            str(tmp_path / "yolo_f" / "data.yaml"),
            "--keeper-train-cache",
            str(tmp_path / "train.npz"),
            "--keeper-val-cache",
            str(tmp_path / "val.npz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--focus-class-index",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="class 1 only"):
        _validate_protocol(args)


def test_alignment_compares_class_names_not_folder_numeric_order() -> None:
    folder_classes = ["class_b", "class_a"]
    target_classes = ["class_a", "class_b"]
    _assert_aligned_class_names(
        np.asarray([1, 0, 1], dtype=np.int64),
        folder_classes,
        np.asarray([0, 1, 0], dtype=np.int64),
        target_classes,
    )


def _metrics(macro: float, focus_f1: float, focus_recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.85, "recall": 0.85},
            {"f1": focus_f1, "precision": 0.75, "recall": focus_recall},
        ],
    }


def _transitions() -> dict:
    return {
        "changed": 20,
        "corrections": 12,
        "harms": 5,
        "neutral": 3,
        "focus_false_positive_removed": 8,
        "focus_false_positive_created": 2,
        "focus_false_negative_rescued": 5,
        "focus_true_positive_broken": 1,
    }


def _direction(auc: float) -> dict:
    return {"auc_fn_positive": auc, "samples": 30}


def test_gate_accepts_transferable_domain_representation() -> None:
    result = assess_bioclip_domain_readiness(
        train_rows=9215,
        val_rows=2606,
        train_val_source_overlap=0,
        fold_source_overlap=0,
        transform_max_abs_diff=0.0,
        equal_visual_parameter_count=True,
        embedding_dimension=512,
        features_finite=True,
        max_norm_error=1e-6,
        minimum_effective_rank=128.0,
        all_readouts_converged=True,
        folds_with_focus_gain=4,
        fold_count=5,
        control_oof=_metrics(0.88, 0.68, 0.75),
        bioclip_oof=_metrics(0.90, 0.72, 0.79),
        keeper_val=_metrics(0.884, 0.684, 0.781),
        control_val=_metrics(0.88, 0.67, 0.76),
        bioclip_val=_metrics(0.90, 0.72, 0.79),
        transitions_vs_keeper=_transitions(),
        domain_oof_direction=_direction(0.68),
        domain_val_direction=_direction(0.66),
        focus_class_index=1,
        test_split_used=False,
    )
    assert result["representation_transfer_smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_source_overlap_and_class1_suppressor() -> None:
    transitions = _transitions()
    transitions.update(
        {
            "corrections": 4,
            "harms": 16,
            "focus_false_negative_rescued": 0,
            "focus_true_positive_broken": 12,
        }
    )
    result = assess_bioclip_domain_readiness(
        train_rows=9000,
        val_rows=1000,
        train_val_source_overlap=2,
        fold_source_overlap=1,
        transform_max_abs_diff=0.1,
        equal_visual_parameter_count=False,
        embedding_dimension=256,
        features_finite=False,
        max_norm_error=0.2,
        minimum_effective_rank=10.0,
        all_readouts_converged=False,
        folds_with_focus_gain=1,
        fold_count=5,
        control_oof=_metrics(0.90, 0.72, 0.78),
        bioclip_oof=_metrics(0.88, 0.66, 0.65),
        keeper_val=_metrics(0.884, 0.684, 0.781),
        control_val=_metrics(0.88, 0.67, 0.75),
        bioclip_val=_metrics(0.86, 0.62, 0.60),
        transitions_vs_keeper=transitions,
        domain_oof_direction=_direction(0.45),
        domain_val_direction=_direction(0.80),
        focus_class_index=1,
        test_split_used=True,
    )
    assert not result["representation_transfer_smoke_permission"]
    assert "train_val_sources_disjoint" in result["failed_checks"]
    assert "source_group_folds_disjoint" in result["failed_checks"]
    assert "val_focus_recall_preserved_within_0p01" in result["failed_checks"]
    assert "candidate_focus_fn_rescued_ge_tp_broken_vs_keeper" in result["failed_checks"]
    assert "test_not_used" in result["failed_checks"]


def test_paired_feature_shift_reports_classwise_change() -> None:
    control = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    candidate = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    result = paired_feature_shift(control, candidate, labels)
    assert 0.0 < result["paired_cosine_mean"] < 1.0
    assert result["displacement_l2_mean"] > 0.0
    assert [row["samples"] for row in result["per_class"]] == [2, 1]


def test_xai_selection_balances_focus_transition_types() -> None:
    labels = np.asarray([1, 1, 0, 2, 1, 4, 0, 2], dtype=np.int64)
    keeper_predictions = np.asarray([0, 1, 1, 0, 2, 1, 0, 2], dtype=np.int64)
    candidate_predictions = np.asarray([1, 0, 0, 1, 1, 4, 1, 0], dtype=np.int64)
    keeper = np.eye(5, dtype=np.float32)[keeper_predictions]
    candidate = np.eye(5, dtype=np.float32)[candidate_predictions]
    selected = select_xai_cases(labels, keeper, candidate, maximum_cases=8)
    categories = {row["category"] for row in selected}
    assert "focus_fn_rescued" in categories
    assert "focus_tp_broken" in categories
    assert "focus_fp_removed" in categories
    assert "focus_fp_created" in categories
    assert len({row["row_index"] for row in selected}) == len(selected)


def test_xai_selection_falls_back_to_stable_references() -> None:
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    probabilities = np.eye(5, dtype=np.float32)
    selected = select_xai_cases(labels, probabilities, probabilities, maximum_cases=4)
    assert len(selected) == 4
    assert all(row["category"] == "stable_reference" for row in selected)
