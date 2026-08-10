from __future__ import annotations

from trkh.tools.audit_lpa3_hard_positive_a0 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DATA_SHA256,
    FOLDS,
    assess_a0_gate,
    parse_args,
    summarize_rows,
)


def _row(
    cohort: str,
    clean: int,
    augmented: int,
    target: int,
    *,
    distance: float = 0.2,
) -> dict[str, object]:
    return {
        "audit_cohort": cohort,
        "clean_prediction_index": clean,
        "aug_prediction_index": augmented,
        "target_index": target,
        "constraint_satisfied": True,
        "selected_step": 2,
        "feature_distance": distance,
        "true_log_probability_drop": 0.01,
        "max_delta_rgb": 2.0 / 255.0,
        "max_outside_mask_delta": 0.0,
        "empty_mask": False,
    }


def test_parser_defaults_to_full_train_only_protocol() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "keeper.pt",
            "--data",
            "data.yaml",
            "--output-dir",
            "output",
        ]
    )
    assert args.seed == 20260724
    assert args.max_train_samples == 0
    assert args.allow_preflight is False


def test_row_summary_counts_class1_safety_and_fp_rejection() -> None:
    rows = [
        _row("protect_correct", 1, 1, 1),
        _row("protect_correct", 1, 0, 1),
        _row("suppress_correct", 0, 0, 0),
        _row("suppress_correct", 2, 1, 2),
        _row("negative_error", 1, 0, 0),
        _row("negative_error", 1, 1, 2),
    ]
    summary = summarize_rows(rows)
    assert summary["class1_tp_retention"] == 0.5
    assert summary["restricted_rival_retention"] == 0.5
    assert summary["restricted_fp_rejection"] == 0.5
    assert summary["class1_tp_broken"] == 1
    assert summary["restricted_fp_removed"] == 1
    assert summary["restricted_fp_created"] == 1


def test_gate_requires_fold_stability_and_every_causal_delta() -> None:
    candidate = summarize_rows(
        [_row("protect_correct", 1, 1, 1, distance=0.30) for _ in range(100)]
        + [_row("suppress_correct", 0, 0, 0, distance=0.30) for _ in range(100)]
        + [_row("negative_error", 1, 0, 0, distance=0.30) for _ in range(100)]
    )
    random_control = summarize_rows(
        [_row("protect_correct", 1, 1, 1, distance=0.10) for _ in range(100)]
        + [_row("suppress_correct", 0, 0, 0, distance=0.10) for _ in range(100)]
        + [_row("negative_error", 1, 1, 0, distance=0.10) for _ in range(100)]
    )
    feature_only = summarize_rows(
        [_row("protect_correct", 1, 0, 1, distance=0.35) for _ in range(100)]
        + [_row("suppress_correct", 0, 0, 0, distance=0.35) for _ in range(100)]
        + [_row("negative_error", 1, 0, 0, distance=0.35) for _ in range(100)]
    )
    fold_rows = []
    for fold in range(FOLDS):
        fold_rows.extend(
            (
                {
                    "variant": "candidate",
                    "source_fold": fold,
                    "class1_tp_retention": 1.0,
                    "restricted_fp_rejection": 1.0,
                },
                {
                    "variant": "random_feasible",
                    "source_fold": fold,
                    "class1_tp_retention": 1.0,
                    "restricted_fp_rejection": 0.0,
                },
                {
                    "variant": "feature_only",
                    "source_fold": fold,
                    "class1_tp_retention": 0.0,
                    "restricted_fp_rejection": 1.0,
                },
            )
        )
    gate = assess_a0_gate(
        preflight=False,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        data_sha256=EXPECTED_DATA_SHA256,
        train_support=9215,
        source_fold_crossings=0,
        replay={"max_abs_delta": 0.0, "selected_steps_equal": True},
        variant_summaries={
            "candidate": candidate,
            "random_feasible": random_control,
            "feature_only": feature_only,
        },
        fold_rows=fold_rows,
    )
    assert gate["automated_a0_permission"] is True
    assert gate["trainer_integration_permission"] is False
    assert gate["visual_review_required"] is True
