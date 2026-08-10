from __future__ import annotations

import numpy as np

from trkh.tools.audit_deferred_reweight_prior_a0 import (
    DEFERRED_BETA,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_STRICT_EXPOSURE,
    _apply_prior_factor,
    _classification_metrics,
    _effective_number_weights,
)


def test_classification_metrics_tracks_focus_support_and_restricted_fp() -> None:
    labels = np.asarray([0, 1, 1, 2, 3, 4], dtype=np.int64)
    predictions = np.asarray([1, 1, 0, 1, 3, 1], dtype=np.int64)

    metrics = _classification_metrics(labels, predictions)

    assert metrics["focus_predicted_support"] == 4
    assert metrics["support"][1] == 2
    assert metrics["focus_support_ratio"] == 2.0
    assert metrics["focus_precision"] == 0.25
    assert metrics["focus_recall"] == 0.5
    assert metrics["restricted_focus_false_positives"] == 3


def test_prior_factor_changes_argmax_without_mutating_probabilities() -> None:
    probabilities = np.asarray(
        [[0.30, 0.40, 0.10, 0.10, 0.10], [0.10, 0.30, 0.20, 0.25, 0.15]],
        dtype=np.float64,
    )
    original = probabilities.copy()
    factor = np.asarray([1.0, 0.25, 1.0, 1.0, 1.0], dtype=np.float64)

    predictions = _apply_prior_factor(probabilities, factor)

    assert predictions.tolist() == [0, 3]
    np.testing.assert_array_equal(probabilities, original)


def test_effective_number_weights_are_bounded_and_mean_normalized() -> None:
    weights = _effective_number_weights(EXPECTED_CLASS_COUNTS, beta=DEFERRED_BETA)

    assert abs(float(weights.mean()) - 1.0) < 1e-12
    assert 1.70 < float(weights[1]) < 1.75
    assert float(weights[1]) == float(weights.max())


def test_locked_exposure_is_less_aggressive_than_strict_after_reweight() -> None:
    counts = np.asarray(EXPECTED_CLASS_COUNTS, dtype=np.float64)
    weights = _effective_number_weights(EXPECTED_CLASS_COUNTS)
    weighted_share = counts * weights / np.sum(counts * weights)
    strict_share = np.asarray(EXPECTED_STRICT_EXPOSURE, dtype=np.float64)
    strict_share /= strict_share.sum()
    natural_share = counts / counts.sum()

    assert natural_share[1] < weighted_share[1] < strict_share[1]
