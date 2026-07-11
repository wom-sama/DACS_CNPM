import csv
from pathlib import Path

from trkh.tools.remap_classification_teacher_to_yolo import (
    remap_classification_teacher_to_yolo,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_remap_classification_teacher_to_yolo_uses_object_sample_index_and_class_order(tmp_path):
    data_root = tmp_path / "yolo"
    images_dir = data_root / "images" / "train"
    labels_dir = data_root / "labels" / "train"
    _touch(images_dir / "Image_A.jpg")
    _touch(images_dir / "Image_B.jpg")
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "Image_A.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "Image_B.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: A",
                "  1: B",
                "  2: C",
            ]
        ),
        encoding="utf-8",
    )

    teacher_csv = tmp_path / "teacher.csv"
    with teacher_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "prob_0", "prob_1", "prob_2"])
        writer.writeheader()
        # Source teacher order is C,A,B. Output must be remapped to A,B,C.
        writer.writerow({"path": "Image_A_box000.jpg", "prob_0": "0.05", "prob_1": "0.90", "prob_2": "0.05"})
        writer.writerow({"path": "Image_A_box001.jpg", "prob_0": "0.10", "prob_1": "0.20", "prob_2": "0.70"})
        writer.writerow({"path": "Image_B_box000.jpg", "prob_0": "0.80", "prob_1": "0.10", "prob_2": "0.10"})

    output_csv = tmp_path / "teacher_yolo.csv"
    summary = remap_classification_teacher_to_yolo(
        teacher_csv=teacher_csv,
        yolo_data=data_yaml,
        split="train",
        output_csv=output_csv,
        source_class_names="C,A,B",
        expected_num_classes=3,
        strict=True,
    )

    assert summary["mapped_rows"] == 3
    assert summary["missing_rows"] == 0
    assert summary["teacher_label_agreement"] == 1.0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["sample_index"]) for row in rows] == [0, 1, 2]
    assert [int(row["object_index"]) for row in rows] == [0, 1, 0]
    assert [int(row["prediction_index"]) for row in rows] == [0, 1, 2]
    assert float(rows[0]["prob_0"]) == 0.9
    assert float(rows[1]["prob_1"]) == 0.7
    assert float(rows[2]["prob_2"]) == 0.8
