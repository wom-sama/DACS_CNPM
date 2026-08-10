import csv
import json
from pathlib import Path

import torch

from trkh.training.train import (
    DataCartographyRecorder,
    _expanded_dataset_labels,
    _expanded_dataset_sample_paths,
)


class DummySample:
    def __init__(self, image_path: Path, label: int) -> None:
        self.image_path = image_path
        self.label = label


class DummyDataset:
    def __init__(self, root: Path) -> None:
        self.samples = [
            DummySample(root / "a.jpg", 0),
            DummySample(root / "b.jpg", 1),
            DummySample(root / "c.jpg", 2),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def sample_paths(self):
        return [sample.image_path for sample in self.samples]

    def labels(self):
        return [sample.label for sample in self.samples]


class IndexedDummyDataset:
    def __init__(self, dataset: DummyDataset) -> None:
        self.dataset = dataset
        self.indices = [2, 0, 2]

    def __len__(self) -> int:
        return len(self.indices)


def test_expanded_dataset_helpers_follow_wrapper_indices(tmp_path):
    dataset = IndexedDummyDataset(DummyDataset(tmp_path))

    assert _expanded_dataset_sample_paths(dataset) == [
        tmp_path / "c.jpg",
        tmp_path / "a.jpg",
        tmp_path / "c.jpg",
    ]
    assert _expanded_dataset_labels(dataset) == [2, 0, 2]


def test_data_cartography_recorder_writes_targets_and_unseen_rows(tmp_path):
    output_path = tmp_path / "cartography.csv"
    recorder = DataCartographyRecorder(
        sample_paths=[tmp_path / "class1.jpg", tmp_path / "class0.jpg", tmp_path / "unseen.jpg"],
        labels=[1, 0, 2],
        class_names=["zero", "one", "two"],
        output_path=output_path,
    )

    recorder.start_epoch(3)
    recorder.update(
        sample_indices=torch.tensor([0, 1], dtype=torch.long),
        logits=torch.tensor([[0.1, 2.0, -1.0], [2.0, 0.1, -1.0]], dtype=torch.float32),
        labels=torch.tensor([1, 0], dtype=torch.long),
    )
    recorder.write(epoch=3)

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8", newline="")))

    assert rows[0]["target_index"] == "1"
    assert rows[0]["target_name"] == "one"
    assert rows[0]["seen_count"] == "1"
    assert rows[0]["cartography_bucket"] == "easy_clean"
    assert rows[0]["prediction_index"] == "1"
    assert rows[0]["prediction_name"] == "one"
    assert rows[0]["top2_index"] == "0"
    assert float(rows[0]["prob_1"]) > 0.80
    assert float(rows[0]["confidence_mean"]) > 0.80

    assert rows[2]["target_index"] == "2"
    assert rows[2]["target_name"] == "two"
    assert rows[2]["seen_count"] == "0"
    assert rows[2]["cartography_bucket"] == "unseen"
    assert rows[2]["prediction_index"] == "-1"
    assert rows[2]["prob_2"] == "0.00000000"
    assert rows[2]["last_epoch"] == "3"

    occurrence = json.loads(
        (tmp_path / "cartography_occurrence_hashes.json").read_text(encoding="utf-8")
    )
    assert occurrence["hash_record"] == "sample_index:target_index\\n in dataloader order"
    assert occurrence["epochs"] == [
        {
            "class_counts": [1, 1, 0],
            "epoch": 3,
            "occurrences": 2,
            "ordered_sample_target_sha256": (
                "4785aee12ff54ef90bb49b6efcf2778a4b32a318e24537a56fcf8992f5e7d022"
            ),
            "unique_sample_indices": 2,
        }
    ]
