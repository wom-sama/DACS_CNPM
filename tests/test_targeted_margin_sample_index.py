import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from trkh.data.dataset import TargetedMarginDataset
from trkh.training.train import _load_targeted_margin_manifest


class _DuplicatePathDataset(Dataset):
    def __init__(self, image_path: Path) -> None:
        self._paths = [image_path, image_path]

    def __len__(self) -> int:
        return len(self._paths)

    def sample_paths(self):
        return list(self._paths)

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), int(index)


def test_targeted_margin_manifest_can_key_by_sample_index(tmp_path):
    image_path = tmp_path / "images" / "train" / "same.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")
    manifest = tmp_path / "targeted_margin.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "image_path",
                "target_index",
                "negative_index",
                "targeted_margin",
                "targeted_margin_weight",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_index": "0",
                "image_path": str(image_path),
                "target_index": "0",
                "negative_index": "1",
                "targeted_margin": "0.11",
                "targeted_margin_weight": "0.7",
            }
        )
        writer.writerow(
            {
                "sample_index": "1",
                "image_path": str(image_path),
                "target_index": "1",
                "negative_index": "2",
                "targeted_margin": "0.13",
                "targeted_margin_weight": "0.9",
            }
        )

    by_path, by_sample_index, summary = _load_targeted_margin_manifest(
        str(manifest),
        num_classes=5,
        default_margin=0.12,
        default_weight=1.0,
        max_weight=2.0,
        return_sample_index_specs=True,
    )

    assert by_path == {}
    assert set(by_sample_index) == {0, 1}
    assert summary["key_mode"] == "sample_index"
    assert summary["sample_indices"] == 2

    wrapped = TargetedMarginDataset(
        _DuplicatePathDataset(image_path),
        by_path,
        margin_specs_by_sample_index=by_sample_index,
        default_margin=0.12,
        default_weight=1.0,
        max_weight=2.0,
    )
    _, _, meta0 = wrapped[0]
    _, _, meta1 = wrapped[1]

    assert int(meta0["targeted_margin_negative"]) == 1
    assert int(meta1["targeted_margin_negative"]) == 2
    assert torch.isclose(meta0["targeted_margin_margin"], torch.tensor(0.11))
    assert torch.isclose(meta1["targeted_margin_margin"], torch.tensor(0.13))
    assert wrapped.targeted_margin_summary()["matched_samples"] == 2
