from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision.transforms import functional as TF

from trkh.core.config import load_data_spec
from trkh.tools.probe_foreground_surface_stats import (
    _build_probe_dataset,
    _collate_classification,
    _dataset_records,
)


def _tensor_transform(image, target=None):
    return TF.to_tensor(image), target


def _write_yolo_data_yaml(root: Path) -> Path:
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "format: yolo",
                "path: .",
                "train: images/train",
                "val: images/val",
                "nc: 5",
                "class_name_mode: raw",
                "names:",
                "  0: c0",
                "  1: c1",
                "  2: c2",
                "  3: c3",
                "  4: c4",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_probe_surface_stats_keeps_yolo_object_records(tmp_path: Path) -> None:
    data_yaml = _write_yolo_data_yaml(tmp_path)
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (80, 64), (120, 180, 60)).save(images_dir / "same.jpg")
    (labels_dir / "same.txt").write_text(
        "\n".join(
            [
                "1 0.300000 0.500000 0.250000 0.500000",
                "4 0.700000 0.500000 0.250000 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    dataset = _build_probe_dataset(
        data_spec=data_spec,
        split="train",
        transform=_tensor_transform,
        crop_margin_ratio=0.0,
    )
    records = _dataset_records(dataset)

    assert len(dataset) == 2
    assert [record["sample_index"] for record in records] == [0, 1]
    assert [record["object_index"] for record in records] == [0, 1]
    assert [record["label"] for record in records] == [1, 4]
    assert records[0]["image_path"] == records[1]["image_path"]
    assert records[0]["bbox_0"] == 0.3
    assert records[1]["bbox_0"] == 0.7

    images, labels = _collate_classification([dataset[0], dataset[1]])

    assert tuple(images.shape[:2]) == (2, 3)
    assert labels.tolist() == [1, 4]
