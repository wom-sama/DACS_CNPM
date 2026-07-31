from __future__ import annotations

from PIL import Image
import pytest
import torch

from trkh.evaluation.robustness_eval import (
    CorruptedDataset,
    IdentityCorruption,
    _unpack_classification_sample,
    evaluate_condition,
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


def test_unpack_classification_folder_sample_uses_full_image_bbox() -> None:
    image, label, metadata, bbox = _unpack_classification_sample(
        (Image.new("RGB", (12, 10), color=(20, 40, 60)), 3)
    )

    assert isinstance(image, Image.Image)
    assert label == 3
    assert metadata == {}
    assert torch.equal(bbox, torch.tensor([0.5, 0.5, 1.0, 1.0]))


def test_corrupted_classification_folder_sample_does_not_invent_model_bbox() -> None:
    class _FolderDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return Image.new("RGB", (12, 10), color=(20, 40, 60)), 3

    dataset = CorruptedDataset(
        base_dataset=_FolderDataset(),
        corruption=IdentityCorruption(),
        transform=_transform,
    )
    _, label, metadata = dataset[0]

    assert label == 3
    assert "bbox" not in metadata
    assert torch.equal(metadata["crop_bbox"], torch.tensor([0.25, 0.25, 0.5, 0.5]))


def test_evaluate_condition_reports_actual_bounded_sample_count() -> None:
    class _ConstantModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(images.shape[0], 2)
            logits[:, 0] = 1.0
            return logits

    dataset = torch.utils.data.TensorDataset(
        torch.zeros(5, 3, 8, 8),
        torch.tensor([0, 1, 0, 1, 0]),
    )
    metrics = evaluate_condition(
        model=_ConstantModel(),
        dataset=dataset,
        class_names=["zero", "one"],
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        max_batches=1,
    )

    assert metrics["evaluated_samples"] == 2
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 2


@pytest.mark.parametrize(
    "sample",
    [
        (torch.zeros(3, 2, 2), 1, {"bbox": torch.ones(4)}),
        (Image.new("RGB", (2, 2)),),
        (Image.new("RGB", (2, 2)), 1, {}, "extra"),
    ],
)
def test_unpack_classification_sample_rejects_invalid_contract(sample) -> None:
    with pytest.raises(ValueError):
        _unpack_classification_sample(sample)
