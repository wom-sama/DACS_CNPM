import numpy as np

from trkh.tools.probe_relative_mahalanobis import (
    _change_summary,
    _class_means,
    _diag_mahalanobis_scores,
    _guard_class1_predictions,
    _score_margin_to_class1,
    _shared_diag_variance,
)


def test_diag_mahalanobis_prefers_nearest_class_mean() -> None:
    train = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
            [3.2, 3.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    means = _class_means(train, labels, class_count=2)
    variance = _shared_diag_variance(train, variance_floor=1e-4)

    scores = _diag_mahalanobis_scores(
        np.asarray([[0.1, 0.1], [3.1, 2.9]], dtype=np.float32),
        means,
        variance,
    )

    assert scores.argmin(axis=1).tolist() == [0, 1]


def test_guard_class1_predictions_switches_only_allowed_base_class1_rows() -> None:
    base = np.asarray([1, 1, 0, 2, 1], dtype=np.int64)
    nearest = np.asarray([0, 2, 1, 1, 1], dtype=np.int64)

    assert _guard_class1_predictions(base, nearest, class1_index=1).tolist() == [
        0,
        2,
        0,
        2,
        1,
    ]
    assert _guard_class1_predictions(
        base,
        nearest,
        class1_index=1,
        allowed_targets=[0],
    ).tolist() == [0, 1, 0, 2, 1]


def test_class1_margin_positive_means_rival_is_closer() -> None:
    scores = np.asarray(
        [
            [1.0, 3.0, 4.0],
            [4.0, 0.5, 2.0],
        ],
        dtype=np.float32,
    )
    margins = _score_margin_to_class1(scores, class1_index=1)

    assert margins[0] > 0.0
    assert margins[1] < 0.0


def test_change_summary_counts_corrections_harms_and_neutral() -> None:
    targets = np.asarray([0, 1, 2, 2], dtype=np.int64)
    before = np.asarray([1, 1, 2, 0], dtype=np.int64)
    after = np.asarray([0, 0, 2, 1], dtype=np.int64)

    summary = _change_summary(targets, before, after)

    assert summary["changed"] == 3
    assert summary["corrections"] == 1
    assert summary["harms"] == 1
    assert summary["neutral"] == 1
    assert summary["transitions"]["0:1->0"] == 1
