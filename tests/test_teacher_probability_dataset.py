from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from trkh.data.dataset import TeacherProbabilityDataset


class _DuplicatePathDataset(Dataset):
    def __init__(self, image_path: Path) -> None:
        self._image_path = Path(image_path)
        self._labels = [1, 2]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return torch.zeros(3, 4, 4), self._labels[int(index)], {}

    def sample_paths(self):
        return [self._image_path, self._image_path]

    def labels(self):
        return list(self._labels)


def test_teacher_probability_dataset_prefers_sample_index_for_duplicate_paths(tmp_path):
    image_path = tmp_path / "multi_object.jpg"
    image_path.write_bytes(b"not-an-image")
    dataset = _DuplicatePathDataset(image_path)

    wrapped = TeacherProbabilityDataset(
        dataset,
        {str(image_path): [0.9, 0.05, 0.05]},
        num_classes=3,
        probabilities_by_sample_index={
            0: [0.1, 0.8, 0.1],
            1: [0.1, 0.2, 0.7],
        },
    )

    _, _, metadata0 = wrapped[0]
    _, _, metadata1 = wrapped[1]

    assert torch.allclose(metadata0["teacher_probs"], torch.tensor([0.1, 0.8, 0.1]))
    assert torch.allclose(metadata1["teacher_probs"], torch.tensor([0.1, 0.2, 0.7]))
    summary = wrapped.teacher_probability_summary()
    assert summary["key_mode"] == "sample_index"
    assert summary["sample_index_path_overlap_ratio"] == 1.0


def test_teacher_probability_dataset_supports_custom_metadata_key(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"not-an-image")
    dataset = _DuplicatePathDataset(image_path)

    wrapped = TeacherProbabilityDataset(
        dataset,
        {str(image_path): [0.9, 0.05, 0.05]},
        num_classes=3,
        probabilities_by_sample_index={
            0: [0.1, 0.8, 0.1],
            1: [0.1, 0.2, 0.7],
        },
        metadata_key="patch_router_teacher_probs",
    )

    _, _, metadata = wrapped[0]
    summary = wrapped.teacher_probability_summary()

    assert "teacher_probs" not in metadata
    assert torch.allclose(
        metadata["patch_router_teacher_probs"],
        torch.tensor([0.1, 0.8, 0.1]),
    )
    assert summary["metadata_key"] == "patch_router_teacher_probs"


def test_teacher_probability_dataset_rejects_sample_index_from_other_view(tmp_path):
    image_path = tmp_path / "class_f_crop.jpg"
    image_path.write_bytes(b"not-an-image")
    other_view_path = tmp_path / "yolo_source.jpg"
    other_view_path.write_bytes(b"not-an-image")
    dataset = _DuplicatePathDataset(image_path)

    with pytest.raises(ValueError, match="sample_index cua view khac"):
        TeacherProbabilityDataset(
            dataset,
            {str(other_view_path): [0.9, 0.05, 0.05]},
            num_classes=3,
            probabilities_by_sample_index={
                0: [0.8, 0.1, 0.1],
                1: [0.8, 0.1, 0.1],
            },
        )
