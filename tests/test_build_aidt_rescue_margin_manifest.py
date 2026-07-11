import csv
from pathlib import Path

from trkh.tools.build_aidt_rescue_margin_manifest import build_manifest


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_aidt_rescue_margin_manifest_keeps_only_teacher_rescues(tmp_path):
    image_root = tmp_path / "images" / "train"
    image_root.mkdir(parents=True)
    baseline_csv = tmp_path / "baseline.csv"
    teacher_csv = tmp_path / "teacher.csv"
    output_csv = tmp_path / "manifest.csv"
    base_fields = [
        "sample_index",
        "image_path",
        "target_index",
        "prediction_index",
        "confidence",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
    ]
    teacher_fields = [
        "sample_index",
        "image_path",
        "target_index",
        "prediction_index",
        "confidence",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
    ]
    rows = []
    teacher_rows = []
    for index in range(4):
        image_path = image_root / f"Image_{index}.jpg"
        image_path.write_bytes(b"x")
        rows.append(
            {
                "sample_index": index,
                "image_path": str(image_path),
                "target_index": [1, 0, 1, 2][index],
                "prediction_index": [0, 1, 1, 1][index],
                "confidence": "0.40",
                "prob_0": "0.20",
                "prob_1": "0.30",
                "prob_2": "0.20",
                "prob_3": "0.15",
                "prob_4": "0.15",
            }
        )
        teacher_rows.append(
            {
                "sample_index": index,
                "image_path": str(image_path),
                "target_index": [1, 0, 1, 2][index],
                "prediction_index": [1, 0, 1, 4][index],
                "confidence": ["0.95", "0.93", "0.99", "0.96"][index],
                "prob_0": ["0.02", "0.93", "0.01", "0.01"][index],
                "prob_1": ["0.95", "0.02", "0.99", "0.01"][index],
                "prob_2": ["0.01", "0.02", "0.00", "0.02"][index],
                "prob_3": ["0.01", "0.01", "0.00", "0.00"][index],
                "prob_4": ["0.01", "0.02", "0.00", "0.96"][index],
            }
        )
    _write_csv(baseline_csv, rows, base_fields)
    _write_csv(teacher_csv, teacher_rows, teacher_fields)

    summary = build_manifest(
        baseline_predictions=baseline_csv,
        teacher_csv=teacher_csv,
        output=output_csv,
        num_classes=5,
        focus_class_index=1,
        negative_classes={0, 2, 4},
        min_teacher_confidence=0.90,
        min_teacher_margin=0.30,
        target_margin=0.10,
        false_positive_base_weight=0.70,
        false_negative_base_weight=0.55,
        confidence_weight=0.0,
        margin_weight=0.0,
        max_weight=1.25,
        max_samples=100,
        max_per_pair=100,
        allow_non_train_paths=False,
        dry_run=False,
    )

    assert summary["rows"] == 2
    assert summary["by_reason"] == {
        "aidt_rescue_false_negative": 1,
        "aidt_suppress_false_positive": 1,
    }
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows_out = list(csv.DictReader(handle))
    assert {int(row["sample_index"]) for row in rows_out} == {0, 1}
    assert {row["reason"] for row in rows_out} == {
        "aidt_rescue_false_negative",
        "aidt_suppress_false_positive",
    }
