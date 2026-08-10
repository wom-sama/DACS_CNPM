import json
from pathlib import Path

from trkh.tools.build_classification_oof_folds import build_folds


CLASS_NAMES = [
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
]


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            count = 4 if split == "train" else 1
            for index in range(count):
                (class_dir / f"{split}_{index:02d}.jpg").write_bytes(b"not-a-real-image")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        json.dumps(
            {
                "path": str(root),
                "train": "train",
                "val": "val",
                "test": "test",
                "format": "classification_folder",
                "names": {index: name for index, name in enumerate(CLASS_NAMES)},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_build_classification_oof_folds_uses_source_split_only(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path / "data")
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

    assert summary["source_sample_count"] == 20
    assert summary["folds"] == 2
    assert summary["materialized_counts"]["copy"] == 60
    for fold in summary["folds_detail"]:
        fold_root = Path(str(fold["root"]))
        assert (fold_root / "data.yaml").is_file()
        payload = json.loads((fold_root / "data.yaml").read_text(encoding="utf-8"))
        assert Path(payload["path"]).is_absolute()
        assert fold["train_count"] == 10
        assert fold["val_count"] == 10
        for split in ("train", "val", "test"):
            for class_name in CLASS_NAMES:
                assert (fold_root / split / class_name).is_dir()
        val_files = sorted((fold_root / "val").glob("*/*.jpg"))
        assert val_files
        assert all(path.name.startswith("train_") for path in val_files)
