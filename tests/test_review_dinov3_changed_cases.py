from __future__ import annotations

from trkh.tools.review_dinov3_changed_cases import (
    TRANSITION_ORDER,
    select_review_rows,
    transition_kind,
)


def test_transition_kind_covers_four_class1_actions() -> None:
    assert (
        transition_kind(target=1, keeper_prediction=0, candidate_prediction=1)
        == "class1_fn_rescued"
    )
    assert (
        transition_kind(target=1, keeper_prediction=1, candidate_prediction=0)
        == "class1_tp_broken"
    )
    assert (
        transition_kind(target=0, keeper_prediction=1, candidate_prediction=0)
        == "class1_fp_removed"
    )
    assert (
        transition_kind(target=0, keeper_prediction=0, candidate_prediction=1)
        == "class1_fp_created"
    )


def test_transition_kind_preserves_generic_correction_and_harm() -> None:
    assert transition_kind(target=2, keeper_prediction=0, candidate_prediction=2) == "correction"
    assert transition_kind(target=2, keeper_prediction=2, candidate_prediction=0) == "harm"
    assert transition_kind(target=2, keeper_prediction=0, candidate_prediction=3) == "neutral"


def test_balanced_review_selection_uses_largest_absolute_p1_delta() -> None:
    rows = []
    sample_index = 0
    for kind in TRANSITION_ORDER:
        for delta in (0.1, -0.7, 0.4):
            rows.append(
                {
                    "sample_index": sample_index,
                    "transition_kind": kind,
                    "p1_delta": delta,
                }
            )
            sample_index += 1
    selected = select_review_rows(rows, per_transition=2)
    assert len(selected) == 8
    for kind in TRANSITION_ORDER:
        bucket = [row for row in selected if row["transition_kind"] == kind]
        assert [abs(float(row["p1_delta"])) for row in bucket] == [0.7, 0.4]


def test_review_selection_rejects_invalid_limit() -> None:
    try:
        select_review_rows([], per_transition=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid per-transition limit to fail")
