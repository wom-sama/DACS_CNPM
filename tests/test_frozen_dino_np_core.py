from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.frozen_dino_np_core import (
    DESCRIPTOR_WIDTH,
    MIN_CALIBRATION_GROUPS,
    VARIANCE_FLOOR,
    class2_conformity_score,
    frozen_prefix_descriptor,
    maximum_group_threshold,
    maximum_order_tolerance_upper,
    readiness_from_metrics,
    selective_2to1_predictions,
    source_balanced_diagonal_gaussian,
)
from trkh.tools.precheck_dinov3_frozen_np_2to1_train_oof import run_oof_audit


def test_frozen_prefix_descriptor_contract_and_register_permutation() -> None:
    generator = torch.Generator().manual_seed(20260801)
    tokens = torch.randn(3, 261, 384, generator=generator)
    descriptor = frozen_prefix_descriptor(tokens)
    assert descriptor.shape == (3, DESCRIPTOR_WIDTH)
    permuted = tokens.clone()
    permuted[:, 1:5] = permuted[:, torch.tensor([4, 2, 1, 3])]
    torch.testing.assert_close(
        frozen_prefix_descriptor(permuted), descriptor, rtol=0.0, atol=1e-6
    )


def test_zero_residual_descriptor_is_exactly_zero() -> None:
    patch = torch.randn(2, 256, 384, generator=torch.Generator().manual_seed(7))
    mean = patch.mean(dim=1, keepdim=True)
    tokens = torch.cat((mean.repeat(1, 5, 1), patch), dim=1)
    assert torch.count_nonzero(frozen_prefix_descriptor(tokens)).item() == 0


def test_group_balancing_gives_each_source_equal_total_weight() -> None:
    values = np.zeros((3, DESCRIPTOR_WIDTH), dtype=np.float32)
    values[0] = 2.0
    values[1] = 4.0
    values[2] = 10.0
    mean, variance, report = source_balanced_diagonal_gaussian(
        values, ["a", "a", "b"]
    )
    # Group a mean is 3 and group b mean is 10, so group-balanced mean is 6.5.
    np.testing.assert_allclose(mean, 6.5, rtol=0.0, atol=1e-6)
    assert report["source_groups"] == 2
    assert report["maximum_group_weight_error"] <= 1e-12
    assert float(variance.min()) >= VARIANCE_FLOOR


def test_conformity_direction_and_strict_group_max_threshold() -> None:
    fit = np.vstack(
        (
            np.zeros((2, DESCRIPTOR_WIDTH), dtype=np.float32),
            np.ones((2, DESCRIPTOR_WIDTH), dtype=np.float32),
        )
    )
    mean, variance, _ = source_balanced_diagonal_gaussian(
        fit, ["a", "a", "b", "b"]
    )
    scores = class2_conformity_score(
        np.vstack((mean, mean + 5.0)).astype(np.float32), mean, variance
    )
    assert scores[0] > scores[1]
    threshold, report = maximum_group_threshold(
        np.asarray([-3.0, -2.0, -4.0]), ["g0", "g0", "g1"]
    )
    assert threshold == -2.0
    assert report["calibration_group_exceedances"] == 0


def test_maximum_order_bound_matches_locked_support() -> None:
    assert maximum_order_tolerance_upper(MIN_CALIBRATION_GROUPS) <= 0.008
    assert maximum_order_tolerance_upper(MIN_CALIBRATION_GROUPS - 1) > 0.008
    assert maximum_order_tolerance_upper(384) == pytest.approx(
        0.0077710342963546175, abs=1e-15
    )


def test_selective_routing_changes_only_strict_eligible_rows() -> None:
    logits = np.asarray(
        [
            [0.0, 3.0, 2.0, 1.0, 0.0],  # eligible and above
            [0.0, 3.0, 2.0, 1.0, 0.0],  # equal threshold: no action
            [0.0, 3.0, 1.0, 2.0, 0.0],  # runner-up 3
            [0.0, 2.0, 3.0, 1.0, 0.0],  # base 2
        ],
        dtype=np.float32,
    )
    base, runner, action, corrected = selective_2to1_predictions(
        logits, np.asarray([0.2, 0.1, 0.3, 0.4]), threshold=0.1
    )
    assert base.tolist() == [1, 1, 1, 2]
    assert runner.tolist() == [2, 2, 3, 2]
    assert action.tolist() == [True, False, False, False]
    assert corrected.tolist() == [2, 1, 1, 2]


def test_readiness_gate_fails_closed_on_one_missing_effect_gate() -> None:
    aligned = {
        "class1_tp_retention": 1.0,
        "corrected_2to1": 14,
        "fp_2to1_reduction": 0.11,
        "new_0to1_or_4to1": 0,
        "outside_domain_changes": 0,
    }
    controls = {
        "zero": {"fp_2to1_reduction": 0.0},
        "deranged": {"fp_2to1_reduction": 0.05},
    }
    folds = [
        {
            "aligned_class1_recall_delta": 0.0,
            "aligned_corrected_2to1": 1,
            "aligned_fp_2to1_reduction": 0.1,
            "zero_fp_2to1_reduction": 0.0,
            "deranged_fp_2to1_reduction": 0.0,
        }
        for _ in range(5)
    ]
    ready = readiness_from_metrics(
        aligned=aligned,
        controls=controls,
        fold_rows=folds,
        integrity_checks={"provenance": True},
    )
    assert ready["frozen_dino_np_2to1_ready"] is True
    aligned["corrected_2to1"] = 13
    failed = readiness_from_metrics(
        aligned=aligned,
        controls=controls,
        fold_rows=folds,
        integrity_checks={"provenance": True},
    )
    assert failed["frozen_dino_np_2to1_ready"] is False
    assert "minimum_2to1_corrections" in failed["failed_checks"]


def test_small_oof_audit_has_complete_rows_and_frozen_parameters(tmp_path) -> None:
    # Two unique source groups for every label/fold keeps each label-preserving
    # derangement partition feasible while exercising the complete OOF path.
    labels = np.asarray(
        [label for fold in range(5) for label in range(5) for _ in range(2)],
        dtype=np.int64,
    )
    assignments = np.asarray(
        [fold for fold in range(5) for _label in range(5) for _ in range(2)],
        dtype=np.int64,
    )
    groups = np.asarray([f"source_{index}" for index in range(labels.size)], dtype=object)
    paths = [f"image_{index}.jpg" for index in range(labels.size)]
    rng = np.random.default_rng(20260801)
    descriptors = rng.normal(size=(labels.size, DESCRIPTOR_WIDTH)).astype(np.float32)
    logits = np.full((labels.size, 5), -2.0, dtype=np.float32)
    logits[np.arange(labels.size), labels] = 2.0
    # Make a few real 2->1 baseline errors with class 2 as runner-up.
    class2 = np.flatnonzero(labels == 2)[:5]
    logits[class2, 1] = 3.0
    audit, fold_rows, prediction_rows, readiness = run_oof_audit(
        descriptors=descriptors,
        b2_logits=logits,
        labels=labels,
        groups=groups,
        assignments=assignments,
        paths=paths,
        output_dir=tmp_path,
    )
    assert len(fold_rows) == 5
    assert len(prediction_rows) == labels.size
    assert len(audit["parameter_artifacts"]) == 15
    assert all((tmp_path / "fold_parameters").glob("*.npy"))
    # This tiny fixture deliberately cannot satisfy the locked 373-group gate.
    assert readiness["frozen_dino_np_2to1_ready"] is False
