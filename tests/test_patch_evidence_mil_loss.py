from __future__ import annotations

import torch

from trkh.training.train import _patch_evidence_mil_loss_from_features


class _IdentityHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))


def test_patch_evidence_mil_loss_uses_bbox_topk_pair_margin() -> None:
    model = _IdentityHeadModel()
    patches = torch.tensor(
        [
            [[0.0, 3.0], [0.0, 2.0], [3.0, 0.0]],
            [[3.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            [[1.0, 1.0], [1.0, 0.5], [0.5, 1.0]],
        ],
        requires_grad=True,
    )
    bbox_prior = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    loss, stats = _patch_evidence_mil_loss_from_features(
        model=model,
        features={"patches": patches, "patch_bbox_prior": bbox_prior},
        targets=torch.tensor([1, 0, 2]),
        pair="0-1",
        top_k=2,
        bbox_threshold=0.05,
    )

    assert float(loss.item()) < 0.10
    assert abs(stats["patch_evidence_mil_fraction"] - (2.0 / 3.0)) < 1e-6
    assert stats["patch_evidence_mil_positive_score"] > 2.0
    assert stats["patch_evidence_mil_negative_score"] < -2.0
    assert stats["patch_evidence_mil_score_margin"] > 4.0
    loss.backward()
    assert patches.grad is not None


def test_patch_evidence_mil_loss_returns_zero_when_pair_absent() -> None:
    model = _IdentityHeadModel()
    patches = torch.randn(2, 3, 2, requires_grad=True)
    loss, stats = _patch_evidence_mil_loss_from_features(
        model=model,
        features={"patches": patches},
        targets=torch.tensor([2, 2]),
        pair="0-1",
    )

    assert loss.item() == 0.0
    assert stats["patch_evidence_mil_fraction"] == 0.0
    loss.backward()
    assert patches.grad is not None
