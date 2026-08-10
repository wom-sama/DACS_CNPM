from __future__ import annotations

import pytest
import torch

from trkh.tools.coteach_finetune import coteaching_cross_losses, peer_small_loss_indices


def test_peer_small_loss_indices_keeps_lowest_losses() -> None:
    losses = torch.tensor([0.7, 0.1, 0.4, 0.2], dtype=torch.float32)

    selected = peer_small_loss_indices(losses, remember_rate=0.50)

    assert set(selected.tolist()) == {1, 3}


def test_coteaching_cross_losses_use_peer_selection() -> None:
    losses_a = torch.tensor([0.9, 0.1, 0.2, 0.8], dtype=torch.float32)
    losses_b = torch.tensor([0.1, 0.8, 0.9, 0.2], dtype=torch.float32)

    loss_a, loss_b, stats = coteaching_cross_losses(losses_a, losses_b, remember_rate=0.50)

    assert loss_a.item() == pytest.approx((0.9 + 0.8) / 2.0)
    assert loss_b.item() == pytest.approx((0.8 + 0.9) / 2.0)
    assert stats["selected_count"] == 2.0
    assert stats["selected_overlap_fraction"] == 0.0
