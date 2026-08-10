from __future__ import annotations

import numpy as np
import pytest

from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    EVENT_NAMES,
    _assign_source_folds,
    _classification_metrics,
    _require_empty_output,
    directional_event_masks,
    select_balanced_indices,
    summarize_directional_rows,
)


def test_directional_event_masks_keep_rescue_and_harm_separate() -> None:
    targets = np.asarray([1, 1, 0, 2, 0, 4])
    keeper = np.asarray([0, 1, 1, 2, 0, 4])
    candidate = np.asarray([1, 0, 0, 1, 1, 3])
    masks = directional_event_masks(targets, keeper, candidate)

    assert np.flatnonzero(masks["focus_fn_rescue"]).tolist() == [0]
    assert np.flatnonzero(masks["focus_tp_break"]).tolist() == [1]
    assert np.flatnonzero(masks["focus_fp_remove_correct"]).tolist() == [2]
    assert np.flatnonzero(masks["focus_fp_create"]).tolist() == [3, 4]
    assert np.flatnonzero(masks["candidate_correction"]).tolist() == [0, 2]
    assert np.flatnonzero(masks["candidate_harm"]).tolist() == [1, 3, 4, 5]


def test_directional_event_masks_reject_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        directional_event_masks(
            np.asarray([0, 1]), np.asarray([0]), np.asarray([0, 1])
        )


def test_balanced_probe_selection_covers_rare_classes() -> None:
    labels = [0] * 10 + [1] * 2 + [2] * 4 + [3] * 5 + [4] * 3
    selected = select_balanced_indices(labels, 15)
    selected_labels = [labels[index] for index in selected]
    assert len(selected) == 15
    assert set(selected_labels) == {0, 1, 2, 3, 4}
    assert selected == select_balanced_indices(labels, 15)
    assert select_balanced_indices(labels, 0) == list(range(len(labels)))


def test_source_fold_assignment_keeps_groups_together() -> None:
    labels = np.asarray([class_index for class_index in range(5) for _ in range(10)])
    groups = np.asarray(
        [f"class{class_index}_source{row // 2}" for class_index in range(5) for row in range(10)]
    )
    assignments, summary = _assign_source_folds(
        labels, groups, folds=5, seed=20260714
    )
    assert summary["assignment_complete"] is True
    assert summary["source_overlap"] == 0
    for group in set(groups.tolist()):
        assert len(set(assignments[groups == group].tolist())) == 1


def test_directional_summary_counts_unique_rows_sources_and_folds() -> None:
    rows = []
    for condition in ("clean", "lighting_dim", "lighting_bright"):
        for sample_index in range(6):
            row = {
                "condition": condition,
                "sample_index": sample_index,
                "source_stem": f"source_{sample_index}",
                "fold": sample_index % 3,
            }
            row.update({event: False for event in EVENT_NAMES})
            rows.append(row)

    rows[0]["focus_fn_rescue"] = True
    for row in rows:
        if row["condition"] != "clean" and row["sample_index"] in {0, 1, 2}:
            row["focus_fn_rescue"] = True
            row["candidate_correction"] = True
        if row["condition"] != "clean" and row["sample_index"] == 3:
            row["focus_tp_break"] = True
            row["candidate_harm"] = True
        if row["condition"] != "clean" and row["sample_index"] == 4:
            row["focus_fp_create"] = True
            row["candidate_harm"] = True

    summary = summarize_directional_rows(rows, fold_count=3)
    assert summary["event_counts"]["focus_fn_rescue"] == 6
    assert summary["unique_sample_counts"]["focus_fn_rescue"] == 3
    assert summary["source_group_counts"]["focus_fn_rescue"] == 3
    assert summary["clean_unique_focus_fn_rescue"] == 1
    assert summary["maximum_rescue_source_share"] == pytest.approx(2 / 6)
    assert summary["folds_with_focus_rescue"] == 3
    assert summary["folds_rescue_exceeds_tp_break"] == 2


def test_classification_metrics_preserve_focus_precision_recall() -> None:
    metrics = _classification_metrics(
        np.asarray([0, 1, 1, 2]),
        np.asarray([0, 1, 0, 2]),
        num_classes=3,
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class_precision"][1] == pytest.approx(1.0)
    assert metrics["per_class_recall"][1] == pytest.approx(0.5)


def test_output_directory_must_be_empty(tmp_path) -> None:
    output = tmp_path / "audit"
    assert _require_empty_output(output) == output.resolve()
    (output / "existing.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        _require_empty_output(output)
