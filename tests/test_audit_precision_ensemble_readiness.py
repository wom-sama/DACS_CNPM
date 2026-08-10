from pathlib import Path

import numpy as np
import pytest

from trkh.tools.audit_precision_ensemble_readiness import (
    _choose_fold_parameters,
    _float_grid,
    _reject_test_input,
    _transition_audit,
    _validate_prediction_row_split,
    _validated_row_probabilities,
    apply_precision_ensemble,
)


CLASS_NAMES = ["c0", "c1", "c2", "c3", "c4"]


def test_float_grid_includes_declared_endpoints() -> None:
    values = _float_grid(0.0, 0.10, 0.02)

    assert np.allclose(values, [0.0, 0.02, 0.04, 0.06, 0.08, 0.10])


def test_reject_test_input_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        _reject_test_input(Path("runs/model/test/predictions.csv"))

    _reject_test_input(Path("runs/model/val/predictions.csv"))


@pytest.mark.parametrize(
    "values, message",
    [
        (["nan", "1"], "Non-finite"),
        (["-0.1", "1.1"], "Negative"),
        (["0", "0"], "positive sum"),
    ],
)
def test_validated_row_probabilities_rejects_invalid_values(values, message) -> None:
    row = {f"prob_{index}": value for index, value in enumerate(values)}

    with pytest.raises(ValueError, match=message):
        _validated_row_probabilities(row, list(row))


def test_validated_row_probabilities_normalizes_finite_nonnegative_values() -> None:
    result = _validated_row_probabilities(
        {"prob_0": "2", "prob_1": "3"},
        ["prob_0", "prob_1"],
    )

    assert result == pytest.approx([0.4, 0.6])


def test_prediction_row_split_is_verified_from_each_image_path() -> None:
    _validate_prediction_row_split(
        [{"image_path": "D:/dataset/images/val/example.jpg"}],
        expected_split="validation",
    )

    with pytest.raises(ValueError, match="do not belong to split=val"):
        _validate_prediction_row_split(
            [{"image_path": "D:/dataset/images/test/example.jpg"}],
            expected_split="val",
        )


def test_apply_precision_ensemble_uses_focus_margin_offset_only_for_decision() -> None:
    keeper = np.asarray([[0.30, 0.35, 0.20, 0.10, 0.05]], dtype=np.float64)
    candidate = np.asarray([[0.25, 0.40, 0.20, 0.10, 0.05]], dtype=np.float64)

    blended, prediction, focus_margin = apply_precision_ensemble(
        keeper,
        candidate,
        candidate_weight=0.40,
        focus_margin_offset=0.10,
        focus_class=1,
    )

    assert np.allclose(blended, [[0.28, 0.37, 0.20, 0.10, 0.05]])
    assert prediction.tolist() == [0]
    assert focus_margin.tolist() == pytest.approx([-0.01])


def test_apply_precision_ensemble_rejects_nonfinite_direct_input() -> None:
    keeper = np.asarray([[0.5, float("nan"), 0.2, 0.2, 0.1]])
    candidate = np.asarray([[0.5, 0.0, 0.2, 0.2, 0.1]])

    with pytest.raises(ValueError, match="non-finite"):
        apply_precision_ensemble(
            keeper,
            candidate,
            candidate_weight=0.4,
            focus_margin_offset=0.0,
        )


def test_transition_audit_separates_precision_and_recall_changes() -> None:
    targets = np.asarray([1, 1, 0, 2, 4, 3])
    reference = np.asarray([0, 1, 1, 2, 4, 3])
    candidate = np.asarray([1, 0, 0, 1, 4, 2])

    summary, categories = _transition_audit(
        targets,
        reference,
        candidate,
        focus_class=1,
    )

    assert summary == {
        "changed": 5,
        "corrections": 2,
        "harms": 3,
        "neutral_changed": 0,
        "focus_fn_rescued": 1,
        "focus_tp_broken": 1,
        "focus_fp_removed": 1,
        "focus_fp_created": 1,
    }
    assert categories.tolist() == [
        "correction",
        "harm",
        "correction",
        "harm",
        "unchanged",
        "harm",
    ]


def test_fold_parameter_selection_respects_precision_floor() -> None:
    targets = np.asarray([1] * 8 + [0] * 8 + [2] * 8 + [3] * 8 + [4] * 8)
    keeper = np.full((40, 5), 0.02, dtype=np.float64)
    candidate = np.full((40, 5), 0.02, dtype=np.float64)
    for index, target in enumerate(targets):
        keeper[index, target] = 0.78
        candidate[index, target] = 0.78

    # Keeper has four weak class-1 false positives and four class-1 false negatives.
    for index in (8, 9, 16, 17):
        keeper[index] = [0.37, 0.42, 0.11, 0.06, 0.04]
    for index in (0, 1, 2, 3):
        keeper[index] = [0.42, 0.37, 0.11, 0.06, 0.04]

    # Candidate strengthens true class-1 rows but is more recall-oriented on negatives.
    for index in range(8):
        candidate[index] = [0.27, 0.52, 0.11, 0.06, 0.04]
    for index in (8, 9, 16, 17):
        candidate[index] = [0.34, 0.45, 0.11, 0.06, 0.04]

    weight, offset, selected, keeper_metrics, feasible_count = _choose_fold_parameters(
        targets,
        keeper,
        candidate,
        np.arange(len(targets)),
        class_names=CLASS_NAMES,
        focus_class=1,
        candidate_weights=[0.0, 0.5],
        focus_margin_offsets=[0.0, 0.04, 0.08],
        minimum_focus_precision=0.65,
        macro_tolerance=0.10,
        precision_weight=0.10,
    )

    assert feasible_count > 0
    assert weight in {0.0, 0.5}
    assert offset in {0.0, 0.04, 0.08}
    assert selected["focus"]["precision"] >= 0.65
    assert selected["focus"]["f1"] >= keeper_metrics["focus"]["f1"]
