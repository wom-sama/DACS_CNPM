from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trkh.tools.audit_same_class_fourier_amplitude_mix_readiness import (
    CROSS_CLASS_MAP,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_KEEPER_VAL_CLASS1,
    EXPECTED_KEEPER_VAL_MACRO,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    assess_smoke_permission,
    assign_source_grouped_folds,
    build_peer_plan,
    focus_cohort,
    fourier_amplitude_mix_rgb,
    summarize_view,
    transition_outcome,
)


def test_fourier_identity_reconstructs_rgb_and_rejects_invalid_settings() -> None:
    generator = torch.Generator().manual_seed(7)
    rgb = torch.rand((3, 3, 16, 12), generator=generator)
    reconstructed, diagnostics = fourier_amplitude_mix_rgb(rgb, rgb)
    assert torch.allclose(reconstructed, rgb, atol=2e-6, rtol=0.0)
    assert diagnostics["finite"] is True
    assert float(diagnostics["max_imaginary_residual"]) <= 1e-6

    with pytest.raises(ValueError, match="mix_lambda"):
        fourier_amplitude_mix_rgb(rgb, rgb, mix_lambda=1.1)
    with pytest.raises(ValueError, match="spectrum_ratio"):
        fourier_amplitude_mix_rgb(rgb, rgb, spectrum_ratio=0.0)


def test_fourier_mix_uses_peer_amplitude_and_is_bounded() -> None:
    target = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    target[:, :, 2:6, 2:6] = 0.25
    peer = torch.zeros_like(target)
    peer[:, 0] = 0.9
    mixed, diagnostics = fourier_amplitude_mix_rgb(target, peer)
    assert mixed.shape == target.shape
    assert torch.isfinite(mixed).all()
    assert float(mixed.min()) >= 0.0
    assert float(mixed.max()) <= 1.0
    assert float(diagnostics["mean_abs_delta_rgb"]) > 0.0


def test_peer_plan_is_same_class_source_distinct_and_deterministic() -> None:
    labels = [class_index for class_index in range(5) for _ in range(3)]
    paths = [
        Path(f"D:/dataset/train/source_{class_index}_{copy}.jpg")
        for class_index in range(5)
        for copy in range(3)
    ]
    first, stats = build_peer_plan(
        target_split="train",
        peer_split="train",
        target_indices=list(range(len(labels))),
        target_labels=labels,
        target_paths=paths,
        peer_labels=labels,
        peer_paths=paths,
    )
    second, _ = build_peer_plan(
        target_split="train",
        peer_split="train",
        target_indices=list(range(len(labels))),
        target_labels=labels,
        target_paths=paths,
        peer_labels=labels,
        peer_paths=paths,
    )
    assert first == second
    assert stats["same_label_matches"] == len(labels)
    assert stats["same_source_distinct"] == len(labels)
    assert stats["cross_label_matches"] == len(labels)
    assert stats["cross_source_distinct"] == len(labels)
    for row in first:
        assert row["same_peer_label"] == row["target_label"]
        assert row["cross_peer_label"] == CROSS_CLASS_MAP[int(row["target_label"])]
        assert row["same_peer_source_stem"] != row["target_source_stem"]


def _row(
    sample_index: int,
    source: str,
    target: int,
    clean: int,
    same: int,
    delta: float,
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "source_stem": source,
        "target_index": target,
        "clean_prediction_index": clean,
        "same_prediction_index": same,
        "cross_prediction_index": same,
        "focus_cohort": focus_cohort(target, clean),
        "same_outcome": transition_outcome(target, clean, same),
        "cross_outcome": transition_outcome(target, clean, same),
        "delta_p1_same": delta,
        "delta_p1_cross": delta,
    }


def test_view_summary_counts_focus_transitions_and_direction() -> None:
    rows = [
        _row(0, "a", 1, 0, 1, 0.20),
        _row(1, "b", 1, 1, 1, 0.01),
        _row(2, "c", 0, 1, 0, -0.20),
        _row(3, "d", 2, 2, 2, -0.01),
    ]
    summary = summarize_view(rows, view="same")
    assert summary["class1_fn_rescued"] == 1
    assert summary["class1_tp_broken"] == 0
    assert summary["class1_fp_removed"] == 1
    assert summary["class1_fp_created"] == 0
    assert summary["class1_fn_vs_fp_delta_auc"] == pytest.approx(1.0)
    assert summary["directional_pass"] is True


def test_source_grouped_folds_never_split_a_source() -> None:
    rows = []
    for index in range(100):
        label = index % 5
        rows.append(
            _row(
                index,
                f"source_{index // 2:03d}",
                label,
                label,
                label,
                0.01,
            )
        )
    assignments = assign_source_grouped_folds(rows, folds=5)
    assert len(assignments) == len(rows)
    source_folds: dict[str, set[int]] = {}
    for row in rows:
        source_folds.setdefault(str(row["source_stem"]), set()).add(
            assignments[int(row["sample_index"])]
        )
    assert all(len(values) == 1 for values in source_folds.values())


def _metrics(macro: float, class1: float) -> dict[str, object]:
    return {
        "macro_f1": macro,
        "per_class": [
            {"class_index": index, "f1": class1 if index == 1 else 0.9}
            for index in range(5)
        ],
    }


def _pairing(split: str, support: int) -> dict[str, object]:
    return {
        "target_split": split,
        "peer_split": "train",
        "support": support,
        "same_label_matches": support,
        "same_source_distinct": support,
        "cross_label_matches": support,
        "cross_source_distinct": support,
    }


def _passing_gate_arguments() -> dict[str, object]:
    val_same = {
        "clean_correct_retention": 0.98,
        "class1_tp_retention": 0.97,
        "class1_fn_mean_delta_p1": 0.02,
        "class1_fp_mean_delta_p1": -0.01,
        "class1_fn_vs_fp_delta_auc": 0.70,
        "class1_fn_rescued": 5,
        "class1_tp_broken": 2,
        "class1_fp_removed": 8,
        "class1_fp_created": 3,
    }
    return {
        "preflight": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "train_support": EXPECTED_TRAIN_COUNT,
        "val_support": EXPECTED_VAL_COUNT,
        "source_overlap_count": 0,
        "train_pairing": _pairing("train", EXPECTED_TRAIN_COUNT),
        "val_pairing": _pairing("val", EXPECTED_VAL_COUNT),
        "numerical": {
            "identity_max_error_rgb": 1e-7,
            "max_imaginary_residual": 1e-7,
            "finite": True,
            "same_mean_rgb_mae": 0.05,
            "same_clip_fraction": 0.01,
        },
        "clean_val_metrics": _metrics(
            EXPECTED_KEEPER_VAL_MACRO, EXPECTED_KEEPER_VAL_CLASS1
        ),
        "same_val_metrics": _metrics(0.884, 0.69),
        "train_same": {"directional_pass": True},
        "val_same": val_same,
        "val_cross": {
            "clean_correct_retention": 0.90,
            "class1_tp_retention": 0.88,
        },
        "fold_rows": [{"directional_pass": True} for _ in range(5)],
    }


def test_smoke_gate_is_conjunctive_and_preflight_fails_closed() -> None:
    arguments = _passing_gate_arguments()
    gate = assess_smoke_permission(**arguments)
    assert gate["smoke_permission"] is True
    assert gate["passed"] == gate["total"]

    arguments["preflight"] = True
    preflight = assess_smoke_permission(**arguments)
    assert preflight["smoke_permission"] is False
    assert preflight["failed_checks"] == ["full_run"]

    arguments = _passing_gate_arguments()
    arguments["val_same"] = dict(arguments["val_same"])
    arguments["val_same"]["class1_fp_mean_delta_p1"] = 0.01
    failed = assess_smoke_permission(**arguments)
    assert failed["smoke_permission"] is False
    assert "val_fp_delta_p1" in failed["failed_checks"]
