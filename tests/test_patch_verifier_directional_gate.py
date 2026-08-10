from __future__ import annotations

import csv
from pathlib import Path

from trkh.tools.apply_patch_verifier_directional_gate import (
    apply_thresholds,
    read_prediction_csv,
    select_thresholds,
)


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "changed",
        "change_pair",
        "verifier_confidence",
        "prob_0",
        "prob_1",
        "prob_2",
        "verifier_0_1_prob_0",
        "verifier_0_1_prob_1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(payload)


def test_directional_gate_disables_harmful_direction(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    _write_predictions(
        csv_path,
        [
            {
                "split": "train",
                "sample_index": 0,
                "image_path": "a.jpg",
                "target_index": 0,
                "base_prediction": 1,
                "prob_0": 0.20,
                "prob_1": 0.30,
                "prob_2": 0.50,
                "verifier_0_1_prob_0": 0.90,
                "verifier_0_1_prob_1": 0.10,
            },
            {
                "split": "train",
                "sample_index": 1,
                "image_path": "b.jpg",
                "target_index": 0,
                "base_prediction": 1,
                "prob_0": 0.25,
                "prob_1": 0.30,
                "prob_2": 0.45,
                "verifier_0_1_prob_0": 0.80,
                "verifier_0_1_prob_1": 0.20,
            },
            {
                "split": "train",
                "sample_index": 2,
                "image_path": "c.jpg",
                "target_index": 1,
                "base_prediction": 1,
                "prob_0": 0.22,
                "prob_1": 0.32,
                "prob_2": 0.46,
                "verifier_0_1_prob_0": 0.60,
                "verifier_0_1_prob_1": 0.40,
            },
            {
                "split": "train",
                "sample_index": 3,
                "image_path": "d.jpg",
                "target_index": 0,
                "base_prediction": 0,
                "prob_0": 0.31,
                "prob_1": 0.29,
                "prob_2": 0.40,
                "verifier_0_1_prob_0": 0.10,
                "verifier_0_1_prob_1": 0.90,
            },
        ],
    )

    rows = read_prediction_csv(csv_path, (0, 1))
    thresholds = select_thresholds(
        rows,
        pair=(0, 1),
        directions=["1->0", "0->1"],
        min_pair_probability=0.02,
        max_pair_margin=0.40,
        min_correction_precision=0.60,
        min_net_corrections=1,
        min_corrections=1,
    )

    assert thresholds["1->0"]["enabled"] is True
    assert thresholds["0->1"]["enabled"] is False
    assert thresholds["1->0"]["threshold"] == 0.8

    final_predictions, changes = apply_thresholds(
        rows,
        pair=(0, 1),
        thresholds=thresholds,
        min_pair_probability=0.02,
        max_pair_margin=0.40,
    )
    assert final_predictions == [0, 0, 1, 0]
    assert [change["change_type"] for change in changes] == ["correction", "correction"]
