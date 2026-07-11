import csv
from pathlib import Path

import yaml

from trkh.tools.build_yolo_oof_folds import build_folds
from trkh.tools.stitch_yolo_oof_predictions import stitch_oof_predictions


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
        (root / "images" / "train" / f"img_{image_index:02d}.jpg").write_bytes(b"not-a-real-image")
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
                "names": CLASS_NAMES,
                "class_name_mode": "raw",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def _write_perfect_prediction_csv(path: Path, fold_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "source_stem",
        "object_index",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        *[f"prob_{index}_{class_name}" for index, class_name in enumerate(CLASS_NAMES)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        sample_index = 0
        for label_path in sorted((fold_root / "labels" / "val").glob("*.txt")):
            image_path = fold_root / "images" / "val" / f"{label_path.stem}.jpg"
            for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                label = int(line.split()[0])
                row = {
                    "sample_index": sample_index,
                    "image_path": str(image_path),
                    "source_stem": label_path.stem,
                    "object_index": object_index,
                    "target_index": label,
                    "target_name": CLASS_NAMES[label],
                    "prediction_index": label,
                    "prediction_name": CLASS_NAMES[label],
                }
                row.update(
                    {
                        f"prob_{index}_{class_name}": 1.0 if index == label else 0.0
                        for index, class_name in enumerate(CLASS_NAMES)
                    }
                )
                writer.writerow(row)
                sample_index += 1


def test_stitch_yolo_oof_predictions_uses_object_index_not_path_only(tmp_path: Path) -> None:
    data_yaml = _make_yolo_dataset(tmp_path / "data")
    folds_root = tmp_path / "folds"
    fold_summary = build_folds(
        data_yaml=data_yaml,
        output_root=folds_root,
        source_split="train",
        folds=2,
        seed=3,
        link_mode="copy",
        overwrite=False,
    )
    inputs = []
    for fold in fold_summary["folds_detail"]:
        fold_name = f"fold_{int(fold['fold']):02d}"
        fold_root = Path(str(fold["root"]))
        csv_path = tmp_path / "predictions" / fold_name / "predictions.csv"
        _write_perfect_prediction_csv(csv_path, fold_root)
        inputs.append(f"{fold_name}={csv_path}")

    summary = stitch_oof_predictions(
        data_yaml=data_yaml,
        inputs=inputs,
        output_dir=tmp_path / "stitched",
        source_split="train",
    )

    assert summary["rows"] == 50
    assert summary["source_object_count"] == 50
    assert summary["missing_source_object_count"] == 0
    assert summary["metrics"]["macro_f1"] == 1.0

    with Path(str(summary["output_csv"])).open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    assert [int(row["sample_index"]) for row in rows] == list(range(50))
    img00_rows = [row for row in rows if row["source_stem"] == "img_00"]
    assert len(img00_rows) == 5
    assert sorted(int(row["object_index"]) for row in img00_rows) == [0, 1, 2, 3, 4]
