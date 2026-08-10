from __future__ import annotations

import torch
from torch import nn

from trkh.training.train import _confusion_pair_mixup_loss_from_features


class _TinyHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 3)


def test_confusion_pair_mixup_loss_uses_boundary_pairs_and_backprops() -> None:
    torch.manual_seed(7)
    model = _TinyHeadModel()
    head_input = torch.randn(6, 4, requires_grad=True)
    features = {"pooled": head_input}
    targets = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long)

    loss, fraction, count, lambda_mean = _confusion_pair_mixup_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        alpha=0.4,
        pairs="0-1,1-2",
        max_pairs=4,
        num_classes=3,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction > 0.0
    assert count > 0.0
    assert 0.5 <= lambda_mean <= 1.0

    loss.backward()
    assert head_input.grad is not None
    assert torch.isfinite(head_input.grad).all()
    assert head_input.grad.abs().sum().item() > 0.0


def test_confusion_pair_mixup_loss_is_zero_without_available_pair() -> None:
    model = _TinyHeadModel()
    head_input = torch.randn(4, 4, requires_grad=True)
    features = {"pooled": head_input}
    targets = torch.tensor([0, 0, 2, 2], dtype=torch.long)

    loss, fraction, count, lambda_mean = _confusion_pair_mixup_loss_from_features(
        model=model,
        features=features,
        targets=targets,
        alpha=0.4,
        pairs="0-1",
        max_pairs=8,
        num_classes=3,
    )

    assert loss.item() == 0.0
    assert fraction == 0.0
    assert count == 0.0
    assert lambda_mean == 0.0
