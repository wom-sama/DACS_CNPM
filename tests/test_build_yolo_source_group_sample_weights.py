import csv
from pathlib import Path

import pytest

from trkh.tools.build_yolo_source_group_sample_weights import build_manifest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_data_yaml(root: Path) -> Path:
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/train",
                "test: images/train",
                "names:",
                "  0: A",
                "  1: B",
                "  2: C",
                "  3: D",
                "  4: E",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_build_yolo_source_group_sample_weights_uses_dataset_sample_order(tmp_path):
    root = tmp_path / "yolo"
    labels_dir = root / "labels" / "train"
    images_dir = root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    _touch(images_dir / "Image_A.jpg")
    _touch(images_dir / "Image_B.jpg")
    _touch(images_dir / "Image_C.jpg")
    (labels_dir / "Image_A.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "Image_B.txt").write_text(
        "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "Image_C.txt").write_text(
        "4 0.4 0.4 0.2 0.2\n4 0.6 0.6 0.2 0.2\n",
        encoding="utf-8",
    )
    data_yaml = _write_data_yaml(root)

    summary = build_manifest(
        data=data_yaml,
        output_dir=tmp_path / "out",
        mixed_label_weight=0.65,
        mixed_label_class1_weight=0.85,
        same_label_multi_weight=0.90,
        same_label_class4_weight=0.75,
    )

    assert summary["objects_total"] == 5
    assert summary["written_rows"] == 4
    assert summary["weighted_class_counts"] == {"0": 1, "1": 1, "4": 2}
    with (tmp_path / "out" / "source_group_sample_weights_train_only.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_index"] for row in rows] == ["1", "2", "3", "4"]
    assert [row["sample_weight"] for row in rows] == ["0.65", "0.85", "0.75", "0.75"]
    assert rows[1]["reason"] == "mixed_label_class1_protected"
    assert rows[2]["reason"] == "same_label_multi_class4"


def test_build_yolo_source_group_sample_weights_rejects_non_train(tmp_path):
    root = tmp_path / "yolo"
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    data_yaml = _write_data_yaml(root)

    with pytest.raises(ValueError, match="train split only"):
        build_manifest(data=data_yaml, output_dir=tmp_path / "out", split="val")
