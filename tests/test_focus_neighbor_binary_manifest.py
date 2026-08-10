import csv
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from trkh.data.dataset import FocusNeighborBinaryDataset
from trkh.tools.build_focus_neighbor_binary_manifest import build_manifest
from trkh.training.train import (
    _focus_neighbor_binary_loss_from_logits,
    _load_focus_neighbor_binary_manifest,
)


def _write_predictions(path: Path) -> None:
    fieldnames = [
        "image_path",
        "target_index",
        "prediction_index",
        "confidence",
        "prob_0_a",
        "prob_1_b",
        "prob_2_c",
        "prob_4_d",
    ]
    rows = [
        {
            "image_path": str(path.parent / "train" / "a.jpg"),
            "target_index": "0",
            "prediction_index": "1",
            "confidence": "0.62",
            "prob_0_a": "0.31",
            "prob_1_b": "0.62",
            "prob_2_c": "0.04",
            "prob_4_d": "0.03",
        },
        {
            "image_path": str(path.parent / "train" / "b.jpg"),
            "target_index": "1",
            "prediction_index": "0",
            "confidence": "0.55",
            "prob_0_a": "0.55",
            "prob_1_b": "0.40",
            "prob_2_c": "0.03",
            "prob_4_d": "0.02",
        },
        {
            "image_path": str(path.parent / "train" / "c.jpg"),
            "target_index": "1",
            "prediction_index": "1",
            "confidence": "0.51",
            "prob_0_a": "0.48",
            "prob_1_b": "0.51",
            "prob_2_c": "0.01",
            "prob_4_d": "0.00",
        },
        {
            "image_path": str(path.parent / "val" / "leak.jpg"),
            "target_index": "0",
            "prediction_index": "1",
            "confidence": "0.80",
            "prob_0_a": "0.10",
            "prob_1_b": "0.80",
            "prob_2_c": "0.05",
            "prob_4_d": "0.05",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_focus_neighbor_binary_manifest_is_train_only(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions_detailed.csv"
    _write_predictions(predictions)
    summary = build_manifest(
        predictions=predictions,
        output_dir=tmp_path / "manifest",
        focus_class_index=1,
        neighbor_classes=[0, 2, 4],
        margin_threshold=0.08,
        false_positive_weight=1.5,
        false_negative_weight=1.2,
        low_margin_positive_weight=1.0,
        low_margin_negative_weight=1.1,
        positive_anchor_weight=0.7,
        include_focus_positive_anchors=False,
        max_weight=2.0,
        max_samples=0,
        max_per_bucket=0,
        dry_run=False,
    )

    assert summary["rows"] == 3
    assert summary["skipped_non_train"] == 1
    rows = list(
        csv.DictReader(
            (tmp_path / "manifest" / "focus_neighbor_binary_train_only.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert {row["reason"] for row in rows} == {
        "focus_false_positive",
        "focus_false_negative",
        "focus_low_margin_positive",
    }
    assert all("val" not in row["image_path"].replace("\\", "/").split("/") for row in rows)


def test_focus_neighbor_binary_loader_rejects_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "binary_target", "binary_weight", "neighbor_index"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(tmp_path / "val" / "a.jpg"),
                "binary_target": "0",
                "binary_weight": "1.0",
                "neighbor_index": "0",
            }
        )

    with pytest.raises(ValueError, match="train split"):
        _load_focus_neighbor_binary_manifest(
            str(manifest),
            default_weight=1.0,
            max_weight=2.0,
        )


class _TinyDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.paths = [root / "train" / "a.jpg", root / "train" / "b.jpg"]

    def __len__(self) -> int:
        return len(self.paths)

    def sample_paths(self):
        return list(self.paths)

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), int(index)


def test_focus_neighbor_binary_dataset_adds_metadata(tmp_path: Path) -> None:
    dataset = _TinyDataset(tmp_path)
    wrapped = FocusNeighborBinaryDataset(
        dataset,
        {
            str((tmp_path / "train" / "a.jpg").resolve()).lower(): {
                "binary_target": 0.0,
                "weight": 1.5,
                "neighbor_index": 0.0,
            }
        },
        default_weight=1.0,
        max_weight=2.0,
    )

    _, _, metadata = wrapped[0]
    assert metadata["focus_neighbor_binary_target"].item() == 0.0
    assert metadata["focus_neighbor_binary_weight"].item() == pytest.approx(1.5)
    assert metadata["focus_neighbor_binary_neighbor"].item() == 0
    _, _, missing = wrapped[1]
    assert missing["focus_neighbor_binary_weight"].item() == 0.0


def test_focus_neighbor_binary_loss_uses_focus_vs_neighbor_logit() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    loss, fraction, focus_probability, weight_mean = _focus_neighbor_binary_loss_from_logits(
        logits=logits,
        binary_targets=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([1.0, 1.0]),
        neighbor_indices=torch.tensor([0, 0]),
        focus_class=1,
        neighbor_classes="0,2",
    )

    assert fraction == pytest.approx(1.0)
    assert weight_mean.item() == pytest.approx(1.0)
    assert focus_probability.item() == pytest.approx(torch.sigmoid(torch.tensor(0.0)).item(), abs=0.4)
    assert loss.item() < 0.2
