from __future__ import annotations

import csv

import pytest

from trkh.tools.build_tokenizer_xai_cohort import build_cohort


def test_tokenizer_cohort_excludes_probability_only_changes(tmp_path) -> None:
    changed_cases = tmp_path / "changed_cases.csv"
    fields = [
        "sample_index",
        "image_path",
        "target",
        "control_prediction",
        "candidate_prediction",
        "focus_fn_rescue",
        "focus_tp_break",
        "new_3_to_2_harm",
        "correction",
        "harm",
        "control_prob_1",
        "candidate_prob_1",
    ]
    rows = [
        {
            "sample_index": "1",
            "image_path": str(tmp_path / "val" / "one.jpg"),
            "target": "0",
            "control_prediction": "0",
            "candidate_prediction": "0",
            "focus_fn_rescue": "0",
            "focus_tp_break": "0",
            "new_3_to_2_harm": "0",
            "correction": "0",
            "harm": "0",
            "control_prob_1": "0.10",
            "candidate_prob_1": "0.40",
        },
        {
            "sample_index": "2",
            "image_path": str(tmp_path / "val" / "two.jpg"),
            "target": "0",
            "control_prediction": "1",
            "candidate_prediction": "0",
            "focus_fn_rescue": "0",
            "focus_tp_break": "0",
            "new_3_to_2_harm": "0",
            "correction": "1",
            "harm": "0",
            "control_prob_1": "0.60",
            "candidate_prob_1": "0.20",
        },
    ]
    with changed_cases.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_cohort(
        changed_cases=changed_cases,
        output_dir=tmp_path / "cohort",
        max_cases=16,
        mode="test_exact_changed_decisions",
    )
    assert summary["source_row_count"] == 2
    assert summary["changed_case_count"] == 1
    assert summary["excluded_unchanged_decisions"] == 1
    with open(summary["cohort"], encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in selected] == ["2"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("control_prediction", ""), ("candidate_prediction", "class_1")),
)
def test_tokenizer_cohort_rejects_invalid_prediction_schema(
    tmp_path, field: str, value: str
) -> None:
    changed_cases = tmp_path / "changed_cases.csv"
    row = {
        "sample_index": "1",
        "image_path": str(tmp_path / "val" / "one.jpg"),
        "target": "0",
        "control_prediction": "1",
        "candidate_prediction": "0",
        "correction": "1",
        "harm": "0",
        "control_prob_1": "0.60",
        "candidate_prob_1": "0.20",
    }
    row[field] = value
    with changed_cases.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match=field):
        build_cohort(
            changed_cases=changed_cases,
            output_dir=tmp_path / "cohort",
            max_cases=16,
            mode="test_invalid_prediction_schema",
        )
