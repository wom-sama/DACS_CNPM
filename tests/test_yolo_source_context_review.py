from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from trkh.tools.build_yolo_source_context_review import build_yolo_source_context_review


def test_build_yolo_source_context_review_maps_crop_to_yolo_box(tmp_path: Path) -> None:
    yolo_root = tmp_path / "yolo"
    (yolo_root / "images" / "train").mkdir(parents=True)
    (yolo_root / "labels" / "train").mkdir(parents=True)
    (yolo_root / "images" / "val").mkdir(parents=True)
    (yolo_root / "labels" / "val").mkdir(parents=True)

    Image.new("RGB", (120, 80), (40, 110, 70)).save(yolo_root / "images" / "train" / "Image_1.jpg")
    (yolo_root / "labels" / "train" / "Image_1.txt").write_text(
        "1 0.500000 0.500000 0.400000 0.500000\n",
        encoding="utf-8",
    )
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "class_name_mode: raw",
                "names:",
                "  0: class0",
                "  1: class1",
            ]
        ),
        encoding="utf-8",
    )

    crop_dir = tmp_path / "class_f" / "train" / "class1"
    crop_dir.mkdir(parents=True)
    crop_path = crop_dir / "Image_1_box000.jpg"
    Image.new("RGB", (48, 48), (90, 140, 80)).save(crop_path)

    review_csv = tmp_path / "review.csv"
    fieldnames = [
        "review_id",
        "split",
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "confidence",
        "target_probability",
        "top2_margin",
        "boundary_pair",
        "reason",
        "severity",
        "manual_label_status",
        "review_notes",
    ]
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "train_00001",
                "split": "train",
                "image_path": str(crop_path),
                "target_index": "1",
                "target_name": "class1",
                "prediction_index": "0",
                "prediction_name": "class0",
                "confidence": "0.91",
                "target_probability": "0.05",
                "top2_margin": "0.86",
                "boundary_pair": "0-1",
                "reason": "oof_cleanlab_1->0",
                "severity": "0.86",
            }
        )

    output_dir = tmp_path / "out"
    summary = build_yolo_source_context_review(
        review_csv=review_csv,
        yolo_data=data_yaml,
        output_dir=output_dir,
        split="train",
        focus_class_index=1,
        max_rows=10,
    )

    manifest = output_dir / "source_context_review_manifest.csv"
    html = output_dir / "source_context_review.html"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    assert summary["output_rows"] == 1
    assert rows[0]["source_id"] == "Image_1"
    assert rows[0]["box_index"] == "0"
    assert rows[0]["yolo_object_count"] == "1"
    assert rows[0]["yolo_label_index"] == "1"
    assert rows[0]["yolo_bbox_area"] == "0.20000000"
    assert rows[0]["class_label_matches_yolo"] == "True"
    assert "manual_expected_class" in rows[0]
    assert "quality_lighting" in rows[0]
    assert "quality_dirty_obstacle" in rows[0]
    assert "quality_partial_fruit" in rows[0]
    assert "quality_background_mask" in rows[0]
    assert Path(rows[0]["context_image_path"]).is_file()
    assert html.is_file()
    html_text = html.read_text(encoding="utf-8")
    assert "YOLO Source Context Boundary Review" in html_text
    assert "Image_1 box0" in html_text
