import torch
from torch import nn

from trkh.training.train import _bbox_token_label_loss_from_features


class _DummyPatchClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 3)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(3))
            self.head.bias.zero_()


def test_bbox_token_label_loss_lower_for_aligned_patch_tokens() -> None:
    model = _DummyPatchClassifier()
    targets = torch.tensor([1, 0])
    bbox_prior = torch.ones(2, 2)
    aligned = torch.tensor(
        [
            [[0.0, 3.0, 0.0], [0.0, 2.5, 0.0]],
            [[3.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ]
    )
    swapped = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 2.5, 0.0]],
        ]
    )

    aligned_loss, aligned_stats = _bbox_token_label_loss_from_features(
        model=model,
        features={"patches": aligned, "patch_bbox_prior": bbox_prior},
        targets=targets,
        min_prior=0.5,
        classes="0,1",
    )
    swapped_loss, _ = _bbox_token_label_loss_from_features(
        model=model,
        features={"patches": swapped, "patch_bbox_prior": bbox_prior},
        targets=targets,
        min_prior=0.5,
        classes="0,1",
    )

    assert aligned_loss.item() < swapped_loss.item()
    assert aligned_stats["bbox_token_label_fraction"] == 1.0
    assert aligned_stats["bbox_token_label_accuracy"] == 1.0


def test_bbox_token_label_loss_respects_class_filter() -> None:
    model = _DummyPatchClassifier()
    patches = torch.randn(2, 4, 3)
    bbox_prior = torch.ones(2, 4)
    loss, stats = _bbox_token_label_loss_from_features(
        model=model,
        features={"patches": patches, "patch_bbox_prior": bbox_prior},
        targets=torch.tensor([2, 2]),
        min_prior=0.5,
        classes="0,1",
    )

    assert loss.item() == 0.0
    assert stats["bbox_token_label_fraction"] == 0.0


def test_bbox_token_label_loss_detaches_classifier_but_updates_patches() -> None:
    model = _DummyPatchClassifier()
    patches = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 0.8, 0.0]]],
        requires_grad=True,
    )
    bbox_prior = torch.ones(1, 2)
    loss, _ = _bbox_token_label_loss_from_features(
        model=model,
        features={"patches": patches, "patch_bbox_prior": bbox_prior},
        targets=torch.tensor([1]),
        min_prior=0.5,
        classes="1",
        detach_classifier=True,
    )
    loss.backward()

    assert patches.grad is not None
    assert float(patches.grad.abs().sum().item()) > 0.0
    assert model.head.weight.grad is None
    assert model.head.bias.grad is None
