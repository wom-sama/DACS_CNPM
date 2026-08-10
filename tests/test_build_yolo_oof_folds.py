import json
from pathlib import Path

import yaml

from trkh.tools.build_yolo_oof_folds import build_folds


CLASS_NAMES = [
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
]


def _make_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_index in range(10):
        image_path = root / "images" / "train" / f"img_{image_index:02d}.jpg"
        image_path.write_bytes(b"not-a-real-image")
        lines = [
            f"{class_index} 0.{class_index + 1} 0.5 0.1 0.1"
            for class_index in range(len(CLASS_NAMES))
        ]
        (root / "labels" / "train" / f"img_{image_index:02d}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "format": "yolo",
                "nc": len(CLASS_NAMES),
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
                "class_name_mode": "raw",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_build_yolo_oof_folds_groups_by_image_without_leakage(tmp_path: Path) -> None:
    data_yaml = _make_yolo_dataset(tmp_path / "data")
    output_root = tmp_path / "folds"

    summary = build_folds(
        data_yaml=data_yaml,
        output_root=output_root,
        source_split="train",
        folds=2,
        seed=7,
        link_mode="copy",
        overwrite=False,
    )

    assert summary["source_image_count"] == 10
    assert summary["source_object_count"] == 50
    assert summary["source_class_object_counts"] == [10, 10, 10, 10, 10]
    assert summary["folds"] == 2
    assert summary["materialized_counts"]["copy"] == 60

    all_val_stems = []
    for fold in summary["folds_detail"]:
        fold_root = Path(str(fold["root"]))
        assert (fold_root / "data.yaml").is_file()
        assert fold["train_image_count"] + fold["val_image_count"] == 10
        assert fold["train_object_count"] + fold["val_object_count"] == 50

        train_images = {path.stem for path in (fold_root / "images" / "train").glob("*.jpg")}
        val_images = {path.stem for path in (fold_root / "images" / "val").glob("*.jpg")}
        test_images = {path.stem for path in (fold_root / "images" / "test").glob("*.jpg")}
        assert train_images
        assert val_images
        assert train_images.isdisjoint(val_images)
        assert test_images == val_images

        fold_yaml = yaml.safe_load((fold_root / "data.yaml").read_text(encoding="utf-8"))
        assert fold_yaml["format"] == "yolo"
        assert fold_yaml["train"] == "images/train"
        assert fold_yaml["val"] == "images/val"
        all_val_stems.extend(fold["val_stems"])

    assert sorted(all_val_stems) == [f"img_{index:02d}" for index in range(10)]
    persisted = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert persisted["source_object_count"] == 50


def test_build_yolo_oof_folds_balanced_max_images_cap(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_index in range(20):
        label = 0 if image_index < 10 else (image_index - 10) % len(CLASS_NAMES)
        (root / "images" / "train" / f"img_{image_index:02d}.jpg").write_bytes(b"not-a-real-image")
        (root / "labels" / "train" / f"img_{image_index:02d}.txt").write_text(
            f"{label} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "format": "yolo",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
                "class_name_mode": "raw",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = build_folds(
        data_yaml=data_yaml,
        output_root=tmp_path / "folds",
        source_split="train",
        folds=2,
        seed=11,
        link_mode="copy",
        max_images=10,
        overwrite=False,
    )

    assert summary["source_image_count"] == 10
    assert all(count >= 2 for count in summary["source_class_object_counts"])
