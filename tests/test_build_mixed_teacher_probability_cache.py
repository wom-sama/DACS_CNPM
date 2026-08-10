from __future__ import annotations

import csv
from pathlib import Path

from trkh.tools.build_mixed_teacher_probability_cache import (
    build_mixed_indices,
    build_mixed_teacher_probability_cache,
)


def _write_teacher_csv(path: Path, prefix: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "path", "prob_0", "prob_1"])
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "sample_index": str(index),
                    "path": str(path.parent / f"{prefix}_{index}.jpg"),
                    "prob_0": f"{0.8 - 0.1 * (index % 2):.3f}",
                    "prob_1": f"{0.2 + 0.1 * (index % 2):.3f}",
                }
            )


def test_build_mixed_teacher_probability_cache_uses_mixed_indices(tmp_path: Path) -> None:
    primary_csv = tmp_path / "primary.csv"
    auxiliary_csv = tmp_path / "auxiliary.csv"
    output_csv = tmp_path / "mixed" / "teacher.csv"
    _write_teacher_csv(primary_csv, "primary", 4)
    _write_teacher_csv(auxiliary_csv, "auxiliary", 4)

    summary = build_mixed_teacher_probability_cache(
        primary_csv=primary_csv,
        auxiliary_csv=auxiliary_csv,
        output_csv=output_csv,
        primary_weight=1.0,
        auxiliary_weight=0.5,
        seed=7,
        primary_name="yolo",
        auxiliary_name="classf",
    )

    expected_indices = build_mixed_indices([4, 4], [1.0, 0.5], seed=7)
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["mixed_rows"] == len(expected_indices)
    assert [int(row["sample_index"]) for row in rows] == list(range(len(rows)))
    assert [
        (0 if row["source_name"] == "yolo" else 1, int(row["source_sample_index"]))
        for row in rows
    ] == expected_indices
    assert rows[0]["path"].endswith(f"{rows[0]['source_name'].replace('yolo', 'primary').replace('classf', 'auxiliary')}_{rows[0]['source_sample_index']}.jpg")
    assert output_csv.with_suffix(".summary.json").is_file()
