from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.audit_vicregl_crop_match_readiness import (
    CROP_A,
    CROP_B,
    INTERIOR_PRIOR_THRESHOLD,
    LOCATION_MAX_DISTANCE,
    NUM_MATCHES,
    OFFICIAL_SOURCE_COMMIT,
    _validate_protocol,
    assess_vicregl_crop_match_readiness,
    batch_crop_match_stats,
    parse_args,
    patch_locations,
    select_xai_cases,
    summarize_local_signal,
    transform_bbox_for_crop,
)


def _args(tmp_path, *extra: str):
    return parse_args(
        [
            "--data",
            str(tmp_path / "dataset" / "data.yaml"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_protocol_defaults_are_locked(tmp_path) -> None:
    args = _args(tmp_path)
    assert args.num_matches == NUM_MATCHES == 20
    assert args.location_max_distance == LOCATION_MAX_DISTANCE == 0.05
    assert args.interior_prior_threshold == INTERIOR_PRIOR_THRESHOLD == 0.50
    assert CROP_A == (0, 0, 240)
    assert CROP_B == (16, 16, 240)
    assert OFFICIAL_SOURCE_COMMIT == "803ae4c8cd1649a820f03afb4793763e95317620"
    _validate_protocol(args)


def test_protocol_rejects_swept_matching_and_nonclass1(tmp_path) -> None:
    with pytest.raises(ValueError, match="gamma=20"):
        _validate_protocol(_args(tmp_path, "--num-matches", "10"))
    with pytest.raises(ValueError, match="class 1 only"):
        _validate_protocol(_args(tmp_path, "--focus-class-index", "2"))


def test_transform_bbox_for_crop_clips_and_rescales() -> None:
    bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    transformed = transform_bbox_for_crop(
        bbox,
        top=0,
        left=0,
        crop_size=128,
        image_size=256,
    )
    assert torch.allclose(
        transformed,
        torch.tensor([[0.75, 0.75, 0.5, 0.5]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_patch_locations_track_crop_coordinates() -> None:
    indices = torch.tensor([[0, 3, 12, 15]], dtype=torch.long)
    locations = patch_locations(
        indices,
        grid_size=(4, 4),
        top=16,
        left=16,
        crop_size=224,
        image_size=256,
    )
    assert locations.shape == (1, 4, 2)
    assert torch.allclose(locations[0, 0], torch.tensor([0.171875, 0.171875]))
    assert torch.allclose(locations[0, -1], torch.tensor([0.828125, 0.828125]))


def test_perfect_crop_match_has_unit_cosine_and_retrieval() -> None:
    torch.manual_seed(7)
    features = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    prior = torch.ones((1, 4), dtype=torch.float32)
    result = batch_crop_match_stats(
        features,
        indices,
        prior,
        features.clone(),
        indices.clone(),
        prior.clone(),
        grid_size=(2, 2),
        crop_a=(0, 0, 256),
        crop_b=(0, 0, 256),
        num_matches=4,
        location_max_distance=0.01,
    )
    assert int(result["match_count"].item()) == 4
    assert float(result["location_match_cosine"].item()) == pytest.approx(1.0)
    assert float(result["retrieval_top1"].item()) == pytest.approx(1.0)
    assert torch.count_nonzero(result["mismatch_map"]) == 0


def _payload() -> dict:
    labels = np.array([1, 1, 0, 2], dtype=np.int64)
    base = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    return {
        "labels": labels,
        "base_probabilities": base,
        "match_count": np.full(4, 20),
        "location_distance": np.full(4, 0.01),
        "location_match_cosine": np.array([0.98, 0.90, 0.94, 0.97]),
        "feature_match_cosine": np.full(4, 0.99),
        "retrieval_top1": np.array([0.8, 0.5, 0.6, 0.75]),
        "feature_match_location_distance": np.full(4, 0.02),
        "eligible_fraction": np.full(4, 0.8),
    }


def test_local_signal_summarizes_class1_error_direction() -> None:
    summary = summarize_local_signal(_payload())
    assert summary["categories"]["class1_true_positive"]["rows"] == 1
    assert summary["categories"]["class1_false_negative"]["rows"] == 1
    assert summary["categories"]["class1_false_positive"]["rows"] == 1
    assert summary["overall"]["fn_minus_tp_mismatch"] == pytest.approx(0.08)
    assert summary["overall"]["fn_vs_fp_mismatch_auc"] == pytest.approx(1.0)


def _metrics(macro: float, focus_f1: float, recall: float) -> dict:
    return {
        "macro_f1": macro,
        "per_class": [
            {"f1": macro, "precision": 0.9, "recall": 0.9},
            {"f1": focus_f1, "precision": 0.75, "recall": recall},
        ],
    }


def _signal(retrieval: float, cosine: float, deficit: float, auc: float) -> dict:
    return {
        "overall": {
            "minimum_match_count": 20,
            "location_match_cosine_mean": cosine,
            "retrieval_top1_mean": retrieval,
            "fn_minus_tp_mismatch": deficit,
            "fn_vs_fp_mismatch_auc": auc,
        }
    }


def _transitions() -> dict:
    return {
        "changed": 20,
        "corrections": 12,
        "harms": 5,
        "neutral": 3,
        "focus_false_positive_removed": 6,
        "focus_false_positive_created": 2,
        "focus_false_negative_rescued": 4,
        "focus_true_positive_broken": 1,
    }


def test_gate_accepts_label_safe_fn_specific_local_headroom() -> None:
    result = assess_vicregl_crop_match_readiness(
        train_rows=9215,
        val_rows=2606,
        train_val_source_overlap=0,
        features_finite=True,
        train_signal=_signal(0.70, 0.94, 0.02, 0.72),
        val_signal=_signal(0.72, 0.95, 0.018, 0.70),
        base_val=_metrics(0.884, 0.686, 0.78),
        crop_average_val=_metrics(0.885, 0.690, 0.78),
        transitions_vs_base=_transitions(),
        prediction_agreement=0.97,
        test_split_used=False,
    )
    assert result["image_smoke_permission"] is True
    assert result["failed_checks"] == []


def test_gate_rejects_saturated_non_directional_signal() -> None:
    transitions = _transitions()
    transitions["harms"] = 20
    result = assess_vicregl_crop_match_readiness(
        train_rows=9215,
        val_rows=2606,
        train_val_source_overlap=0,
        features_finite=True,
        train_signal=_signal(0.90, 0.985, -0.001, 0.48),
        val_signal=_signal(0.91, 0.986, 0.001, 0.51),
        base_val=_metrics(0.884, 0.686, 0.78),
        crop_average_val=_metrics(0.880, 0.670, 0.74),
        transitions_vs_base=transitions,
        prediction_agreement=0.96,
        test_split_used=False,
    )
    assert result["image_smoke_permission"] is False
    assert "val_retrieval_has_headroom_le_0p85" in result["failed_checks"]
    assert "val_fn_vs_fp_auc_ge_0p60" in result["failed_checks"]


def test_xai_selection_prioritizes_class1_transitions() -> None:
    labels = np.array([1, 1, 0, 2, 0], dtype=np.int64)
    base = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
        ]
    )
    crop = np.array(
        [
            [0.2, 0.7, 0.1],
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ]
    )
    cases = select_xai_cases(labels, base, crop, maximum_cases=4)
    categories = {case["category"] for case in cases}
    assert "focus_fn_rescued" in categories
    assert "focus_tp_broken" in categories
    assert "focus_fp_removed" in categories
    assert "focus_fp_created" in categories
