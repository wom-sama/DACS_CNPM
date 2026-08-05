from __future__ import annotations

import pytest
import torch

from trkh.models.trust_margin_distillation_b30 import (
    GAIN_CAP,
    orchestrated_margin_targets,
    trust_margin_distillation_loss,
)


def test_harmful_teacher_margin_is_ignored() -> None:
    primary = torch.tensor([[2.0, 0.0, -1.0, -2.0, -3.0]])
    teacher = torch.tensor([[0.0, 3.0, -1.0, -2.0, -3.0]])
    labels = torch.tensor([0])
    target, rivals, gain = orchestrated_margin_targets(primary, teacher, labels)
    expected = primary[:, :1] - primary
    assert torch.equal(target[rivals], expected[rivals])
    assert torch.count_nonzero(gain[rivals]) == 0


def test_helpful_teacher_gain_is_nonnegative_and_capped() -> None:
    primary = torch.zeros(2, 5)
    teacher = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    target, rivals, gain = orchestrated_margin_targets(primary, teacher, labels)
    assert torch.all(gain[rivals] >= 0)
    assert float(gain.max()) == GAIN_CAP
    assert torch.equal(target, gain)


def test_loss_is_finite_and_reaches_student_logits() -> None:
    generator = torch.Generator().manual_seed(20260805)
    primary = torch.randn(4, 5, generator=generator)
    teacher = primary + torch.randn(4, 5, generator=generator)
    logits = primary.clone().requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 1])
    loss, parts = trust_margin_distillation_loss(logits, primary, teacher, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert set(parts) == {
        "task",
        "kd",
        "retention",
        "total",
        "positive_gain_fraction",
        "gain_mean",
        "class1_positive_gain_fraction",
    }


def test_contract_rejects_wrong_score_shape() -> None:
    with pytest.raises(ValueError):
        orchestrated_margin_targets(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
