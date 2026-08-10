import torch
import torch.nn.functional as F

from trkh.training.train import (
    _foreground_counterexample_mix_batch,
    _foreground_snapmix_batch,
)


def test_foreground_snapmix_creates_bbox_soft_targets() -> None:
    torch.manual_seed(4)
    labels = torch.tensor([0, 1, 0, 1, 2, 1, 4, 1], dtype=torch.long)
    images = torch.zeros((labels.numel(), 3, 32, 32), dtype=torch.float32)
    for index in range(labels.numel()):
        images[index].fill_(float(index + 1) / 10.0)
    targets = F.one_hot(labels, num_classes=5).to(dtype=torch.float32)
    bboxes = torch.tensor([[0.5, 0.5, 0.75, 0.75]] * labels.numel(), dtype=torch.float32)

    mixed_images, mixed_targets, selected_indices, stats = _foreground_snapmix_batch(
        images=images,
        targets=targets,
        labels=labels,
        bboxes=bboxes,
        num_classes=5,
        probability=1.0,
        alpha=1.0,
        pairs="0-1,1-2,1-4",
        min_area_ratio=0.16,
        max_area_ratio=0.16,
        bbox_margin_ratio=0.0,
    )

    assert mixed_images is not None
    assert mixed_targets is not None
    assert selected_indices is not None
    assert stats["count"] > 0
    assert torch.allclose(mixed_targets.sum(dim=1), torch.ones(mixed_targets.size(0)))
    assert (mixed_targets > 0).sum(dim=1).max().item() >= 2
    original_selected = images.index_select(0, selected_indices)
    assert not torch.allclose(mixed_images, original_selected)


def test_foreground_snapmix_requires_bbox_metadata() -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)
    images = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    mixed_images, mixed_targets, selected_indices, stats = _foreground_snapmix_batch(
        images=images,
        targets=labels,
        labels=labels,
        bboxes=None,
        num_classes=5,
        probability=1.0,
        alpha=1.0,
        pairs="0-1",
        min_area_ratio=0.10,
        max_area_ratio=0.10,
        bbox_margin_ratio=0.0,
    )

    assert mixed_images is None
    assert mixed_targets is None
    assert selected_indices is None
    assert stats["fraction"] == 0.0


def test_foreground_counterexample_mix_keeps_target_labels() -> None:
    torch.manual_seed(7)
    labels = torch.tensor([0, 1, 2, 4, 1, 0], dtype=torch.long)
    images = torch.zeros((labels.numel(), 3, 32, 32), dtype=torch.float32)
    for index in range(labels.numel()):
        images[index].fill_(float(index + 1) / 10.0)
    targets = F.one_hot(labels, num_classes=5).to(dtype=torch.float32)
    bboxes = torch.tensor([[0.5, 0.5, 0.75, 0.75]] * labels.numel(), dtype=torch.float32)

    mixed_images, mixed_targets, selected_indices, stats = (
        _foreground_counterexample_mix_batch(
            images=images,
            targets=targets,
            labels=labels,
            bboxes=bboxes,
            num_classes=5,
            probability=1.0,
            source_class=1,
            target_classes="0,2,4",
            alpha=1.0,
            min_area_ratio=0.16,
            max_area_ratio=0.16,
            bbox_margin_ratio=0.0,
        )
    )

    assert mixed_images is not None
    assert mixed_targets is not None
    assert selected_indices is not None
    assert stats["count"] == selected_indices.numel()
    assert not torch.any(labels.index_select(0, selected_indices) == 1)
    assert torch.allclose(mixed_targets, targets.index_select(0, selected_indices))
    original_selected = images.index_select(0, selected_indices)
    assert not torch.allclose(mixed_images, original_selected)


def test_foreground_counterexample_mix_requires_source_class() -> None:
    labels = torch.tensor([0, 2, 4], dtype=torch.long)
    images = torch.zeros((3, 3, 16, 16), dtype=torch.float32)
    bboxes = torch.tensor([[0.5, 0.5, 0.75, 0.75]] * labels.numel(), dtype=torch.float32)

    mixed_images, mixed_targets, selected_indices, stats = (
        _foreground_counterexample_mix_batch(
            images=images,
            targets=labels,
            labels=labels,
            bboxes=bboxes,
            num_classes=5,
            probability=1.0,
            source_class=1,
            target_classes="0,2,4",
            alpha=1.0,
            min_area_ratio=0.10,
            max_area_ratio=0.10,
            bbox_margin_ratio=0.0,
        )
    )

    assert mixed_images is None
    assert mixed_targets is None
    assert selected_indices is None
    assert stats["fraction"] == 0.0
