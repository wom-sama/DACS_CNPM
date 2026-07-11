import torch

from trkh.training.train import _dcl_region_shuffle_batch


def test_dcl_region_shuffle_changes_bbox_grid_and_keeps_targets() -> None:
    torch.manual_seed(7)
    images = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    labels = torch.tensor([1])
    bboxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)

    shuffled, targets, indices, stats = _dcl_region_shuffle_batch(
        images=images,
        targets=labels,
        labels=labels,
        bboxes=bboxes,
        num_classes=5,
        probability=1.0,
        grid_size=4,
        bbox_margin_ratio=0.0,
        classes="0,1,2,3,4",
    )

    assert shuffled is not None
    assert targets is not None
    assert indices is not None
    assert shuffled.shape == images.shape
    assert not torch.equal(shuffled, images)
    assert torch.equal(indices.cpu(), torch.tensor([0]))
    assert torch.allclose(targets, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert stats["count"] == 1.0
    assert stats["fraction"] == 1.0
    assert stats["grid_size"] == 4.0


def test_dcl_region_shuffle_respects_class_filter() -> None:
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 2])

    shuffled, targets, indices, stats = _dcl_region_shuffle_batch(
        images=images,
        targets=labels,
        labels=labels,
        bboxes=None,
        num_classes=5,
        probability=1.0,
        grid_size=4,
        bbox_margin_ratio=0.0,
        classes="1",
    )

    assert shuffled is None
    assert targets is None
    assert indices is None
    assert stats["count"] == 0.0
