from __future__ import annotations

from trkh.tools.review_surface_blob_morphology_changed_cases import (
    reproduction_batch_windows,
    select_review_rows,
    transition_kind,
    transition_summary,
)


def _row(index: int, target: int, keeper: int, candidate: int) -> dict:
    return {
        "sample_index": index,
        "target_index": target,
        "keeper_prediction_index": keeper,
        "keeper_candidate_residual_prediction_index": candidate,
    }


def test_transition_kind_prioritizes_class1_actions() -> None:
    assert transition_kind(_row(0, 1, 0, 1)) == "class1_fn_rescued"
    assert transition_kind(_row(1, 1, 1, 0)) == "class1_tp_broken"
    assert transition_kind(_row(2, 0, 1, 0)) == "class1_fp_removed"
    assert transition_kind(_row(3, 0, 0, 1)) == "class1_fp_created"
    assert transition_kind(_row(4, 2, 3, 2)) == "correction"
    assert transition_kind(_row(5, 2, 2, 3)) == "harm"
    assert transition_kind(_row(6, 2, 3, 4)) == "neutral"


def test_transition_summary_matches_locked_accounting() -> None:
    rows = [
        _row(0, 1, 0, 1),
        _row(1, 1, 1, 0),
        _row(2, 0, 1, 0),
        _row(3, 0, 0, 1),
        _row(4, 2, 3, 2),
        _row(5, 2, 2, 3),
        _row(6, 2, 3, 4),
    ]
    summary = transition_summary(rows)
    assert summary == {
        "changed": 7,
        "corrections": 3,
        "harms": 3,
        "neutral": 1,
        "class1_fn_rescued": 1,
        "class1_tp_broken": 1,
        "class1_fp_removed": 1,
        "class1_fp_created": 1,
    }


def test_review_selection_round_robins_transition_kinds() -> None:
    rows = []
    for index in range(4):
        rows.extend(
            (
                _row(index * 10 + 0, 1, 0, 1),
                _row(index * 10 + 1, 1, 1, 0),
                _row(index * 10 + 2, 0, 1, 0),
                _row(index * 10 + 3, 0, 0, 1),
                _row(index * 10 + 4, 2, 3, 2),
                _row(index * 10 + 5, 2, 2, 3),
                _row(index * 10 + 6, 2, 3, 4),
            )
        )
    selected = select_review_rows(rows, maximum=14)
    kinds = [transition_kind(row) for row in selected]
    assert len(selected) == 14
    assert kinds[:7] == [
        "class1_fn_rescued",
        "class1_tp_broken",
        "class1_fp_removed",
        "class1_fp_created",
        "correction",
        "harm",
        "neutral",
    ]
    assert kinds[7:] == kinds[:7]


def test_reproduction_batch_windows_restore_original_context() -> None:
    windows = reproduction_batch_windows(
        [130, 1, 63, 64, 129],
        dataset_size=131,
        batch_size=64,
    )
    assert windows == [
        (0, 64, [1, 63]),
        (64, 128, [64]),
        (128, 131, [129, 130]),
    ]
