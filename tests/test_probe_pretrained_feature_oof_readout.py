import csv
import json
from pathlib import Path

import numpy as np

from trkh.tools.probe_pretrained_feature_oof_readout import run_probe


def _write_npz(path: Path, features, labels, stems, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        sample_index=np.arange(len(labels), dtype=np.int64),
        paths=np.asarray([f"Image_{idx}.jpg" for idx in range(len(labels))], dtype=object),
        source_stem=np.asarray(stems, dtype=object),
        classes=np.asarray(classes, dtype=object),
    )


def test_probe_pretrained_feature_oof_readout_writes_selected_predictions(tmp_path):
    classes = ["A", "B", "C"]
    train_features = [
        [3.0, 0.0, 0.0],
        [2.8, 0.1, 0.0],
        [0.0, 3.0, 0.0],
        [0.1, 2.7, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.2, 2.8],
    ]
    train_labels = [0, 0, 1, 1, 2, 2]
    val_features = [[3.1, 0.0, 0.0], [0.0, 2.9, 0.1], [0.1, 0.0, 2.9]]
    val_labels = [0, 1, 2]
    train_npz = tmp_path / "train_features.npz"
    val_npz = tmp_path / "val_features.npz"
    _write_npz(
        train_npz,
        train_features,
        train_labels,
        [f"train_{idx}" for idx in range(len(train_labels))],
        classes,
    )
    _write_npz(
        val_npz,
        val_features,
        val_labels,
        [f"val_{idx}" for idx in range(len(val_labels))],
        classes,
    )

    out_dir = tmp_path / "probe"
    summary = run_probe(
        train_npz=train_npz,
        val_npz=val_npz,
        output_dir=out_dir,
        candidates="logreg_balanced_c1",
        folds=2,
        seed=7,
        n_jobs=1,
        focus_class=1,
    )

    assert summary["selected_variant"] == "logreg_balanced_c1"
    assert summary["selection_source"] == "train_oof"
    assert summary["class_names"] == classes
    assert summary["train_samples"] == 6
    assert summary["val_samples"] == 3
    assert (out_dir / "variant_metrics.csv").is_file()
    assert (out_dir / "summary.json").is_file()
    with (out_dir / "val_predictions_selected.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["variant"] for row in rows} == {"logreg_balanced_c1"}
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["folds"] == 2
