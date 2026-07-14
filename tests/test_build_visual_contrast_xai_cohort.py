from __future__ import annotations

from trkh.tools.build_visual_contrast_xai_cohort import (
    classify_change,
    select_cohort_rows,
)


def _row(
    sample_index: int,
    *,
    target: int,
    control: int,
    candidate: int,
    delta: float,
    correction: int = 0,
    harm: int = 0,
    rescue: int = 0,
    break_tp: int = 0,
    new_3_to_2: int = 0,
) -> dict[str, str]:
    return {
        "sample_index": str(sample_index),
        "source_stem": f"image_{sample_index}",
        "object_index": "0",
        "image_path": f"D:/data/images/val/image_{sample_index}.jpg",
        "target": str(target),
        "control_prediction": str(control),
        "candidate_prediction": str(candidate),
        "correction": str(correction),
        "harm": str(harm),
        "focus_fn_rescue": str(rescue),
        "focus_tp_break": str(break_tp),
        "new_3_to_2_harm": str(new_3_to_2),
        "control_prob_1": "0.2",
        "candidate_prob_1": str(0.2 + delta),
    }


def test_classification_prioritizes_focus_error_semantics() -> None:
    assert classify_change(_row(1, target=0, control=1, candidate=0, delta=-0.1)) == (
        "focus_fp_removed"
    )
    assert classify_change(_row(2, target=2, control=2, candidate=1, delta=0.1)) == (
        "focus_fp_added"
    )
    assert classify_change(
        _row(3, target=1, control=0, candidate=1, delta=0.1, correction=1, rescue=1)
    ) == "focus_fn_rescue"
    assert classify_change(
        _row(4, target=1, control=1, candidate=0, delta=-0.1, harm=1, break_tp=1)
    ) == "focus_tp_break"


def test_selection_honors_category_quota_then_probability_delta() -> None:
    rows = [
        _row(1, target=0, control=1, candidate=0, delta=-0.10),
        _row(2, target=0, control=1, candidate=0, delta=-0.30),
        _row(3, target=0, control=1, candidate=0, delta=-0.20),
        _row(4, target=0, control=1, candidate=0, delta=-0.40),
        _row(5, target=2, control=2, candidate=1, delta=0.50),
        _row(6, target=1, control=0, candidate=1, delta=0.25, correction=1, rescue=1),
    ]
    selected = select_cohort_rows(
        rows,
        max_cases=5,
        category_quotas=(("focus_fp_removed", 3), ("focus_fp_added", 1)),
    )
    assert [row["sample_index"] for row in selected] == ["4", "2", "3", "5", "6"]
    assert selected[0]["target_index"] == "0"
    assert selected[0]["control_prediction_index"] == "1"
    assert selected[0]["candidate_prediction_index"] == "0"
