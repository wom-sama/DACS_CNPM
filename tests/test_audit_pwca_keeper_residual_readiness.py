from __future__ import annotations

import numpy as np
import pytest

from trkh.tools.audit_pwca_keeper_residual_readiness import (
    _guard_non_test_path,
    apply_keeper_relative_residual,
    select_oof_residual_alpha,
)


def test_zero_alpha_is_exact_keeper_identity() -> None:
    base = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32)
    control = np.asarray([[0.5, 0.4, 0.1], [0.3, 0.6, 0.1]], dtype=np.float32)
    pwca = np.asarray([[0.4, 0.5, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    actual = apply_keeper_relative_residual(base, control, pwca, 0.0)
    np.testing.assert_allclose(actual, base, atol=1e-7, rtol=1e-7)


def test_oof_selection_keeps_identity_when_nonzero_residual_breaks_recall() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    base = np.asarray(
        [[0.2, 0.8], [0.3, 0.7], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32
    )
    control = np.full((4, 2), 0.5, dtype=np.float32)
    pwca = np.asarray(
        [[0.8, 0.2], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=np.float32
    )
    alpha, rows = select_oof_residual_alpha(
        labels,
        base,
        control,
        pwca,
        class_names=["0", "1"],
        alpha_grid=np.asarray([0.0, 1.0]),
    )
    assert alpha == 0.0
    assert [bool(row["eligible"]) for row in rows] == [True, False]


def test_test_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="test data"):
        _guard_non_test_path(
            "D:/DataAI/AIEx/newdataset/yolo_f/test/predictions.csv", label="input"
        )
