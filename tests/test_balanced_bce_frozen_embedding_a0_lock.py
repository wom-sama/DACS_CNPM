from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import build_trkh_balanced_bce_frozen_embedding_a0_lock as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_balanced_bce_lock_rebuilds_exactly() -> None:
    generated = builder.build_lock()
    expected_text = builder.LOCK_PATH.read_text(encoding="utf-8")
    assert builder._serialize(generated) == expected_text

    sidecar = builder.LOCK_SHA_PATH.read_text(encoding="ascii")
    assert sidecar == (
        f"{_sha256(builder.LOCK_PATH)}  {builder.LOCK_PATH.name}\n"
    )


def test_balanced_bce_lock_is_prospective_and_train_only() -> None:
    lock = json.loads(builder.LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["lock_state"] == (
        "prospective_before_candidate_implementation_training_or_metrics"
    )
    assert lock["claim_boundary"]["candidate_metrics_observed_before_lock"] is False
    assert lock["dataset_declaration"]["outer_source_overlap"] == 0
    assert lock["dataset_declaration"]["cache_split_allowlist"] == ["train"]
    assert lock["dataset_declaration"]["cache_split_denylist"] == [
        "val",
        "valid",
        "validation",
        "test",
    ]
    assert lock["dataset_declaration"]["image_augmentation_used"] is False
    assert lock["dataset_declaration"]["synthetic_data_used"] is False
    assert lock["data_access_audit"]["validation_open_count_max"] == 0
    assert lock["data_access_audit"]["test_open_count_max"] == 0
    assert lock["pass_policy"]["test_allowed"] is False
    assert lock["pass_policy"]["probe_allowed"] is False
    assert lock["pass_policy"]["full_train_allowed"] is False
    assert lock["pass_policy"]["current_best_update_allowed"] is False


def test_balanced_bce_fold_priors_biases_and_orders_are_locked() -> None:
    lock = json.loads(builder.LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["dataset_declaration"]["outer_assignment_sha256"] == (
        "512226167ac354ceb5886ed478f17bc4d27d60017bca444ceb19b76a0fa424e1"
    )
    assert lock["visual_anchors"]["ordered_indices_sha256"] == (
        "cf6bfe611947a72075d5d85525e74c4a74d103cdfb276a5f7ee9b2d27f8a901e"
    )

    for fold in lock["outer_folds"]:
        fit_counts = np.asarray(fold["fit_class_counts"], dtype=np.int64)
        held_counts = np.asarray(fold["held_class_counts"], dtype=np.int64)
        prior = np.asarray(fold["fit_prior_float64"], dtype=np.float64)
        bias = np.asarray(
            fold["balanced_bce_bias_float64"], dtype=np.float64
        )
        reverse = np.asarray(
            fold["reversed_prior_bias_float64"], dtype=np.float64
        )

        assert int(fit_counts.sum()) == fold["fit_rows"]
        assert int(held_counts.sum()) == fold["held_rows"]
        assert fold["fit_rows"] + fold["held_rows"] == 9215
        assert np.allclose(prior, fit_counts / fit_counts.sum(), atol=0.0)
        assert np.allclose(
            bias, np.log(prior) - np.log1p(-prior), atol=1e-15
        )
        assert np.array_equal(reverse, -bias)
        assert fold["source_overlap"] == 0

        primary = fold["primary_orders"]
        repeat = fold["repeat_orders"]
        assert len(primary["per_epoch_sha256"]) == 30
        assert len(repeat["per_epoch_sha256"]) == 30
        assert primary["all_epochs_sha256"] != repeat["all_epochs_sha256"]
        assert (
            fold["primary_initial_head"]["combined_sha256"]
            != fold["repeat_initial_head"]["combined_sha256"]
        )


def test_balanced_bce_roles_and_equations_are_fixed() -> None:
    lock = json.loads(builder.LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["roles"] == [
        "ce_control",
        "plain_bce_control",
        "balanced_bce_candidate",
        "balanced_bce_seed_repeat",
        "balanced_softmax_tau025_control",
        "reversed_prior_bce_control",
    ]
    equation = lock["equation"]
    assert equation["balanced_bce_candidate"]["tau"] == 1.0
    assert equation["balanced_bce_candidate"]["inference"] == (
        "argmax(raw_logits)"
    )
    assert equation["balanced_softmax_tau025_control"]["tau"] == 0.25
    assert equation["label_smoothing"] == 0.0
    assert equation["pos_weight"] is None
    assert equation["class_weights"] is None
    assert equation["test_prior_term"] is False
    assert equation["threshold_calibration"] is False
