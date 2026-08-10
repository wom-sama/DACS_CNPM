from __future__ import annotations

import math

import torch

from trkh.tools.audit_joint_cls_localization_gradient_readiness import (
    AUXILIARY_GRADIENT_RATIO,
    BATCH_SIZE,
    FOLDS,
    HEAD_WARMUP_BATCHES,
    HEAD_WARMUP_BATCH_SIZE,
    HEAD_WARMUP_LEARNING_RATE,
    SAMPLES_PER_CLASS_PER_FOLD,
    SEED,
    STEP_PARAMETER_RATIO,
    PairedSource,
    assess_joint_gradient_readiness,
    parse_args,
    project_conflicting_auxiliary_gradient,
    select_stratified_audit_folds,
)


def test_protocol_defaults_are_locked() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "keeper.pt",
            "--data-yaml",
            "data.yaml",
            "--output-dir",
            "out",
        ]
    )
    assert args.folds == FOLDS == 5
    assert args.samples_per_class_per_fold == SAMPLES_PER_CLASS_PER_FOLD == 12
    assert args.batch_size == BATCH_SIZE == 4
    assert args.head_warmup_batch_size == HEAD_WARMUP_BATCH_SIZE == 8
    assert args.head_warmup_batches == HEAD_WARMUP_BATCHES == 120
    assert args.head_warmup_learning_rate == HEAD_WARMUP_LEARNING_RATE == 2e-4
    assert args.step_parameter_ratio == STEP_PARAMETER_RATIO == 1e-4
    assert args.auxiliary_gradient_ratio == AUXILIARY_GRADIENT_RATIO == 0.25
    assert args.seed == SEED == 20260712
    assert args.device == "cuda"


def test_pcgrad_projects_only_conflicting_component() -> None:
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([-1.0, 1.0])]
    projected, summary = project_conflicting_auxiliary_gradient(primary, auxiliary)
    assert torch.allclose(projected[0], torch.tensor([0.0, 1.0]))
    assert summary["raw_conflict"]
    assert math.isclose(float(summary["projected_dot"]), 0.0, abs_tol=1e-7)
    assert math.isclose(
        float(summary["retained_norm_ratio"]),
        1.0 / math.sqrt(2.0),
        rel_tol=1e-6,
    )

    aligned = [torch.tensor([0.5, 1.0])]
    unchanged, aligned_summary = project_conflicting_auxiliary_gradient(primary, aligned)
    assert torch.equal(unchanged[0], aligned[0])
    assert not aligned_summary["raw_conflict"]
    assert aligned_summary["retained_norm_ratio"] == 1.0


def test_stratified_audit_folds_are_class_balanced_and_source_disjoint() -> None:
    rows = [
        PairedSource(
            source_stem=f"class{class_index}_source{row_index}",
            label=class_index,
            classification_index=class_index * 100 + row_index,
            detection_index=class_index * 100 + row_index,
        )
        for class_index in range(3)
        for row_index in range(12)
    ]
    folds = select_stratified_audit_folds(
        rows,
        class_count=3,
        folds=3,
        samples_per_class_per_fold=4,
        seed=7,
    )
    assert len(folds) == 3
    all_stems = []
    for fold in folds:
        assert len(fold) == 12
        assert [sum(row.label == class_index for row in fold) for class_index in range(3)] == [4, 4, 4]
        all_stems.extend(row.source_stem for row in fold)
    assert len(all_stems) == len(set(all_stems))


def _alignment(raw_cosine: float, retained: float) -> dict:
    return {
        "raw_cosine": raw_cosine,
        "retained_norm_ratio": retained,
    }


def _influence(
    *,
    loss: float,
    positive_margin: float,
    negative_margin: float,
    true_margin: float = 0.25,
) -> dict:
    return {
        "loss": loss,
        "mean_true_margin": true_margin,
        "focus_positive_margin": positive_margin,
        "focus_negative_margin": negative_margin,
    }


def _fold_payload(
    *,
    raw_cosine: float = 0.10,
    total_retained: float = 0.90,
    localization_retained: float = 0.85,
    focus_retained: float = 0.80,
    classification_loss: float = 0.80,
    naive_loss: float = 0.802,
    pcgrad_loss: float = 0.799,
    classification_positive_margin: float = 0.20,
    pcgrad_positive_margin: float = 0.205,
    classification_negative_margin: float = 0.30,
    pcgrad_negative_margin: float = 0.305,
) -> dict:
    return {
        "natural_alignment": {
            "det_total": _alignment(raw_cosine, total_retained),
            "det_localization": _alignment(raw_cosine, localization_retained),
        },
        "focus_alignment": _alignment(raw_cosine, focus_retained),
        "influence": {
            "classification_only": _influence(
                loss=classification_loss,
                positive_margin=classification_positive_margin,
                negative_margin=classification_negative_margin,
            ),
            "naive_joint": _influence(
                loss=naive_loss,
                positive_margin=classification_positive_margin,
                negative_margin=classification_negative_margin,
            ),
            "pcgrad_joint": _influence(
                loss=naive_loss,
                positive_margin=classification_positive_margin,
                negative_margin=classification_negative_margin,
            ),
            "pcgrad_spatial": _influence(
                loss=pcgrad_loss,
                positive_margin=pcgrad_positive_margin,
                negative_margin=pcgrad_negative_margin,
            ),
        },
    }


def test_gate_accepts_consistent_classification_safe_signal() -> None:
    result = assess_joint_gradient_readiness(
        fold_rows=[_fold_payload() for _ in range(5)],
        warmup_summary={"last_to_first_loss_ratio": 0.75},
        source_overlap=0,
        test_split_used=False,
    )
    assert result["smoke_permission"]
    assert result["failed_checks"] == []


def test_gate_rejects_conflict_and_class1_margin_damage() -> None:
    rows = [
        _fold_payload(
            raw_cosine=-0.80,
            total_retained=0.40,
            localization_retained=0.30,
            focus_retained=0.20,
            pcgrad_loss=0.82,
            pcgrad_positive_margin=0.10,
            pcgrad_negative_margin=0.20,
        )
        for _ in range(5)
    ]
    result = assess_joint_gradient_readiness(
        fold_rows=rows,
        warmup_summary={"last_to_first_loss_ratio": 0.98},
        source_overlap=3,
        test_split_used=True,
    )
    assert not result["smoke_permission"]
    assert "train_only_no_test" in result["failed_checks"]
    assert "head_warmup_audit_source_disjoint" in result["failed_checks"]
    assert "total_projection_retained_median_ge_0p70" in result["failed_checks"]
    assert "focus_positive_margin_delta_ge_minus_0p01" in result["failed_checks"]


def test_gate_rejects_safe_but_non_incremental_auxiliary() -> None:
    rows = [
        _fold_payload(
            classification_loss=0.80,
            naive_loss=0.80,
            pcgrad_loss=0.80,
            classification_positive_margin=0.20,
            pcgrad_positive_margin=0.20,
            classification_negative_margin=0.30,
            pcgrad_negative_margin=0.30,
        )
        for _ in range(5)
    ]
    result = assess_joint_gradient_readiness(
        fold_rows=rows,
        warmup_summary={"last_to_first_loss_ratio": 0.75},
        source_overlap=0,
        test_split_used=False,
    )
    assert not result["smoke_permission"]
    assert "spatial_pcgrad_has_measurable_incremental_signal" in result["failed_checks"]
