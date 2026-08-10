from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from trkh.data.dataset import TeacherFeatureDataset
from trkh.models.model import TeacherFeatureProjectionAdapter
from trkh.tools.remap_classification_features_to_yolo import (
    remap_classification_features_to_yolo,
)
from trkh.training.train import _teacher_feature_rkd_loss_from_features
from trkh.training.train import _teacher_feature_contrastive_loss_from_features


class TinyDataset(Dataset):
    def __init__(self, paths):
        self._paths = [Path(path) for path in paths]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        return torch.zeros(3, 4, 4), int(index), {"sample_index": torch.tensor(index)}

    def sample_paths(self):
        return list(self._paths)

    def labels(self):
        return [0 for _ in self._paths]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_teacher_feature_dataset_attaches_sample_index_features(tmp_path):
    dataset = TinyDataset([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    wrapped = TeacherFeatureDataset(
        dataset,
        {},
        features_by_sample_index={
            0: [1.0, 2.0, 3.0],
            1: [4.0, 5.0, 6.0],
        },
    )

    _, _, metadata = wrapped[1]

    assert torch.equal(metadata["teacher_features"], torch.tensor([4.0, 5.0, 6.0]))
    summary = wrapped.teacher_feature_summary()
    assert summary["key_mode"] == "sample_index"
    assert summary["feature_dim"] == 3


def test_teacher_feature_dataset_requires_full_sample_index_coverage(tmp_path):
    dataset = TinyDataset([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    with pytest.raises(ValueError, match="khong phu het"):
        TeacherFeatureDataset(dataset, {}, features_by_sample_index={0: [1.0, 2.0]})


def test_teacher_feature_rkd_loss_is_dimension_agnostic_and_backpropagates():
    student = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    loss, stats = _teacher_feature_rkd_loss_from_features(
        model=torch.nn.Identity(),
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        distance_weight=1.0,
        angle_weight=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["count"] == 3
    assert stats["distance_loss"] >= 0.0
    assert stats["angle_loss"] >= 0.0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_teacher_feature_rkd_boundary_mode_masks_unlisted_pairs():
    student = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1000.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    boundary_loss, boundary_stats = _teacher_feature_rkd_loss_from_features(
        model=torch.nn.Identity(),
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        distance_weight=1.0,
        angle_weight=0.0,
        hard_labels=labels,
        pair_mode="boundary",
        pairs="0-1",
        num_classes=3,
    )
    all_loss, all_stats = _teacher_feature_rkd_loss_from_features(
        model=torch.nn.Identity(),
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        distance_weight=1.0,
        angle_weight=0.0,
        hard_labels=labels,
        pair_mode="all",
        pairs="0-1",
        num_classes=3,
    )

    assert boundary_stats["count"] == 3
    assert boundary_stats["pair_count"] == 2
    assert all_stats["pair_count"] == 0
    assert boundary_loss.item() < 1e-6
    assert all_loss.item() > boundary_loss.item()


def test_teacher_feature_contrastive_loss_filters_boundary_pairs_and_backpropagates():
    student = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [0.0, -2.0], [-2.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [-2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 4], dtype=torch.long)

    loss, stats = _teacher_feature_contrastive_loss_from_features(
        model=torch.nn.Identity(),
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        temperature=0.2,
        projection_dim=4,
        hard_labels=labels,
        pair_mode="boundary",
        pairs="0-1",
        num_classes=5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["count"] == 2
    assert stats["negative_count"] == 1.0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_teacher_feature_contrastive_loss_returns_zero_without_valid_negatives():
    student = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 5)
    labels = torch.tensor([0, 0, 0], dtype=torch.long)

    loss, stats = _teacher_feature_contrastive_loss_from_features(
        model=torch.nn.Identity(),
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        temperature=0.2,
        projection_dim=4,
        hard_labels=labels,
        pair_mode="boundary",
        pairs="0-1",
        num_classes=5,
    )

    assert loss.item() == 0.0
    assert stats["count"] == 3
    assert stats["negative_count"] == 0.0


def test_teacher_feature_contrastive_loss_uses_learned_projection_adapter():
    student = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [0.0, -2.0], [-2.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [-2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    model = torch.nn.Identity()
    model.teacher_feature_projection_adapter = TeacherFeatureProjectionAdapter(
        student_dim=2,
        teacher_dim=3,
        projection_dim=4,
        dropout=0.0,
    )

    loss, stats = _teacher_feature_contrastive_loss_from_features(
        model=model,
        features={"pooled": student},
        teacher_features=teacher,
        source="head",
        temperature=0.2,
        projection_dim=4,
        hard_labels=labels,
        pair_mode="boundary",
        pairs="0-1",
        num_classes=5,
    )
    loss.backward()

    adapter = model.teacher_feature_projection_adapter
    assert torch.isfinite(loss)
    assert stats["projection_adapter"] is True
    assert adapter.student_projector[-1].weight.grad is not None
    assert adapter.teacher_projector[-1].weight.grad is not None
    assert torch.isfinite(adapter.student_projector[-1].weight.grad).all()
    assert torch.isfinite(adapter.teacher_projector[-1].weight.grad).all()


def test_remap_classification_features_to_yolo_outputs_sample_index_npz(tmp_path):
    data_root = tmp_path / "yolo"
    images_dir = data_root / "images" / "train"
    labels_dir = data_root / "labels" / "train"
    _touch(images_dir / "Image_A.jpg")
    _touch(images_dir / "Image_B.jpg")
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "Image_A.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    (labels_dir / "Image_B.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: A",
                "  1: B",
                "  2: C",
            ]
        ),
        encoding="utf-8",
    )

    feature_npz = tmp_path / "classf_features.npz"
    np.savez_compressed(
        feature_npz,
        features=np.asarray(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]],
            dtype=np.float32,
        ),
        paths=np.asarray(
            ["Image_A_box000.jpg", "Image_A_box001.jpg", "Image_B_box000.jpg"],
            dtype=object,
        ),
        classes=np.asarray(["A", "B", "C"], dtype=object),
    )
    output_npz = tmp_path / "yolo_features.npz"

    summary = remap_classification_features_to_yolo(
        teacher_feature_npz=feature_npz,
        yolo_data=data_yaml,
        split="train",
        output_npz=output_npz,
        expected_num_classes=3,
        strict=True,
    )

    assert summary["mapped_rows"] == 3
    with np.load(output_npz, allow_pickle=True) as data:
        assert data["features"].shape == (3, 2)
        assert data["sample_index"].tolist() == [0, 1, 2]
        assert data["labels"].tolist() == [0, 1, 2]
        assert data["object_index"].tolist() == [0, 1, 0]
