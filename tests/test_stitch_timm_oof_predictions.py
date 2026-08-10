import csv
import json
from pathlib import Path

import pytest

from trkh.tools.stitch_timm_oof_predictions import stitch_oof_predictions


TARGET_CLASSES = [
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
]

SOURCE_CLASSES = sorted(TARGET_CLASSES)


def _write_data_yaml(root: Path) -> Path:
    train_root = root / "train"
    for class_name in TARGET_CLASSES:
        class_dir = train_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "sample.jpg").write_bytes(b"fake")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        json.dumps(
            {
                "path": str(root),
                "train": "train",
                "val": "train",
                "test": "train",
                "format": "classification_folder",
                "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_stitch_oof_predictions_maps_source_class_order_to_data_yaml_order(tmp_path: Path) -> None:
    data_yaml = _write_data_yaml(tmp_path / "data")
    fold_dir = tmp_path / "fold_00" / "val" / "Xoai_Song_ChuaNhe_CoNguyCo"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_image = fold_dir / "sample.jpg"
    fold_image.write_bytes(b"fake")
    prediction_csv = tmp_path / "predictions_val.csv"

    source_index = {class_name: index for index, class_name in enumerate(SOURCE_CLASSES)}
    probabilities = [0.01 for _ in SOURCE_CLASSES]
    probabilities[source_index["Xoai_Song_ChuaNhe_CoNguyCo"]] = 0.90
    with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "y_true", "y_pred", "true_name", "pred_name", "confidence"] + [
            f"prob_{index}" for index in range(len(SOURCE_CLASSES))
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "path": str(fold_image),
                "y_true": source_index["Xoai_Song_ChuaNhe_CoNguyCo"],
                "y_pred": source_index["Xoai_Song_ChuaNhe_CoNguyCo"],
                "true_name": "Xoai_Song_ChuaNhe_CoNguyCo",
                "pred_name": "Xoai_Song_ChuaNhe_CoNguyCo",
                "confidence": 0.90,
                **{f"prob_{index}": probabilities[index] for index in range(len(SOURCE_CLASSES))},
            }
        )
    (tmp_path / "metrics_val.json").write_text(
        json.dumps({"classes": SOURCE_CLASSES}),
        encoding="utf-8",
    )

    summary = stitch_oof_predictions(
        data_yaml=data_yaml,
        inputs=[f"fold00={prediction_csv}"],
        output_dir=tmp_path / "stitched",
    )

    assert summary["rows"] == 1
    with Path(str(summary["output_csv"])).open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["path"].endswith(r"train\Xoai_Song_ChuaNhe_CoNguyCo\sample.jpg") or row["path"].endswith(
        "train/Xoai_Song_ChuaNhe_CoNguyCo/sample.jpg"
    )
    assert row["target_index"] == "1"
    assert row["prediction_index"] == "1"
    assert float(row["prob_1_Xoai_Song_ChuaNhe_CoNguyCo"]) == pytest.approx(0.90 / 0.94)
