from __future__ import annotations

from PIL import Image
import pytest
import torch

from trkh.evaluation.robustness_eval import (
    CorruptedDataset,
    IdentityCorruption,
    _unpack_classification_sample,
)


class _ClassificationDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return (
            Image.new("RGB", (12, 10), color=(20, 40, 60)),
            1,
            {"bbox": torch.tensor([0.5, 0.5, 0.4, 0.6])},
        )


def _transform(image: Image.Image, *, target):
    assert image.size == (12, 10)
    transformed_target = {
        "labels": target["labels"],
        "boxes": target["boxes"] * 0.5,
        "image_mask": torch.ones(8, 8, dtype=torch.bool),
    }
    return torch.zeros(3, 8, 8), transformed_target


def test_corrupted_dataset_preserves_classification_label_and_bbox() -> None:
    dataset = CorruptedDataset(
        base_dataset=_ClassificationDataset(),
        corruption=IdentityCorruption(),
        transform=_transform,
    )
    image, label, metadata = dataset[0]
    assert image.shape == (3, 8, 8)
    assert label == 1
    assert torch.allclose(metadata["bbox"], torch.tensor([0.5, 0.5, 0.4, 0.6]))
    assert torch.allclose(metadata["crop_bbox"], torch.tensor([0.25, 0.25, 0.2, 0.3]))
    assert metadata["image_mask"].all()


@pytest.mark.parametrize(
    "sample",
    [
        (Image.new("RGB", (2, 2)), 1),
        (torch.zeros(3, 2, 2), 1, {"bbox": torch.ones(4)}),
        (Image.new("RGB", (2, 2)), 1, {}),
    ],
)
def test_unpack_classification_sample_rejects_invalid_contract(sample) -> None:
    with pytest.raises(ValueError):
        _unpack_classification_sample(sample)
