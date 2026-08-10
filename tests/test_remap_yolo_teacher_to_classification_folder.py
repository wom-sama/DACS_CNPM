import csv
import sys
from pathlib import Path

from PIL import Image

from trkh.tools import remap_yolo_teacher_to_classification_folder as remap


def test_remap_yolo_teacher_writes_router_metadata(tmp_path, monkeypatch):
    data_root = tmp_path / "class_f"
    class_dir = data_root / "train" / "c1"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), (120, 160, 80)).save(class_dir / "Image_10_box001.jpg")
    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "format: classification_folder",
                f"path: {data_root.as_posix()}",
                "train: train",
                "val: train",
                "names:",
                "  0: c0",
                "  1: c1",
            ]
        ),
        encoding="utf-8",
    )

    teacher_csv = tmp_path / "teacher.csv"
    with teacher_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "prob_0", "prob_1"])
        writer.writeheader()
        writer.writerow({"image_path": "Image_10.jpg", "prob_0": "0.9", "prob_1": "0.1"})
        writer.writerow({"image_path": "Image_10.jpg", "prob_0": "0.2", "prob_1": "0.8"})

    output_csv = tmp_path / "remapped.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remap",
            "--classification-data",
            str(data_yaml),
            "--split",
            "train",
            "--yolo-teacher-csv",
            str(teacher_csv),
            "--output-csv",
            str(output_csv),
            "--class-name-mode",
            "raw",
            "--expected-num-classes",
            "2",
            "--strict",
        ],
    )

    remap.main()

    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["target_index"] == "1"
    assert row["target_name"] == "c1"
    assert row["true_name"] == "c1"
    assert row["prediction_index"] == "1"
    assert row["prediction_name"] == "c1"
    assert row["pred_name"] == "c1"
    assert abs(float(row["confidence"]) - 0.8) < 1e-6
    assert abs(float(row["prob_0"]) + float(row["prob_1"]) - 1.0) < 1e-6
