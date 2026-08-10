from __future__ import annotations

import pytest

from trkh.tools.probe_crop_margin_sensitivity import (
    build_changed_prediction_rows,
    build_sweep_row,
    margin_key,
    parse_margin_grid,
    summarize_changed_predictions,
)


def test_parse_margin_grid_sorts_unique_and_validates() -> None:
    assert parse_margin_grid("0.05,0,0.05;0.12") == [0.0, 0.05, 0.12]
    assert margin_key(0.05) == "m0p05"
    assert margin_key(0.0) == "m0"
    with pytest.raises(ValueError):
        parse_margin_grid("-0.1")


def test_changed_prediction_rows_count_corrections_harms_and_neutral() -> None:
    class_names = ["0", "1", "2"]
    baseline = [
        {"sample_index": 0, "target_index": 0, "prediction_index": 1, "confidence": 0.7},
        {"sample_index": 1, "target_index": 1, "prediction_index": 1, "confidence": 0.8},
        {"sample_index": 2, "target_index": 2, "prediction_index": 0, "confidence": 0.6},
        {"sample_index": 3, "target_index": 2, "prediction_index": 2, "confidence": 0.9},
    ]
    current = [
        {"sample_index": 0, "target_index": 0, "prediction_index": 0, "confidence": 0.6},
        {"sample_index": 1, "target_index": 1, "prediction_index": 0, "confidence": 0.7},
        {"sample_index": 2, "target_index": 2, "prediction_index": 1, "confidence": 0.5},
        {"sample_index": 3, "target_index": 2, "prediction_index": 2, "confidence": 0.9},
    ]
    rows, counts = build_changed_prediction_rows(
        baseline_records=baseline,
        margin_records=current,
        margin=0.12,
        class_names=class_names,
    )
    assert counts == {"corrections": 1, "harms": 1, "neutral": 1}
    assert [row["change_type"] for row in rows] == ["correction", "harm", "neutral"]
    assert [row["transition"] for row in rows] == ["0:1->0", "1:1->0", "2:0->1"]

    summary = summarize_changed_predictions(
        baseline_records=baseline,
        margin_records=current,
        margin=0.12,
        class_names=class_names,
    )
    assert summary["changed"] == 3
    assert summary["transitions"]["0:1->0"] == 1


def test_build_sweep_row_extracts_class1_and_deltas() -> None:
    metrics = {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "weighted_f1": 0.85,
        "per_class": [
            {"class_index": 0, "precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"class_index": 1, "precision": 0.7, "recall": 0.6, "f1": 0.64},
        ],
    }
    row = build_sweep_row(
        margin=0.08,
        metrics=metrics,
        sample_count=10,
        baseline_macro_f1=0.79,
        baseline_class1_f1=0.62,
    )
    assert row["samples"] == 10
    assert row["class1_f1"] == pytest.approx(0.64)
    assert row["delta_macro_f1_vs_baseline"] == pytest.approx(0.01)
    assert row["delta_class1_f1_vs_baseline"] == pytest.approx(0.02)
