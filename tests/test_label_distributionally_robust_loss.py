import math

import pytest
import torch
import torch.nn.functional as F

from trkh.training.losses import LabelDistributionallyRobustLoss


def _official_ldr_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    target_probs = F.one_hot(targets, num_classes=logits.size(1)).to(logits.dtype)
    scores = F.softplus(logits)
    scores = scores / scores.mean(dim=1, keepdim=True)
    target_scores = (scores * target_probs).sum(dim=1, keepdim=True)
    differences = margin * (1.0 - target_probs) + scores - target_scores
    return temperature * (
        torch.logsumexp(differences / temperature, dim=1)
        - math.log(float(logits.size(1)))
    )


def test_ldr_kl_matches_official_formula_for_hard_targets() -> None:
    logits = torch.tensor(
        [[1.2, -0.4, 0.3], [-0.7, 0.1, 1.5]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 2])
    criterion = LabelDistributionallyRobustLoss(margin=2.0, temperature=0.8)

    actual = criterion.per_sample_loss(logits, targets)
    expected = _official_ldr_reference(logits, targets, margin=2.0, temperature=0.8)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_ldr_kl_prefers_a_larger_target_logit_and_backpropagates() -> None:
    logits = torch.tensor([[0.0, 0.8, -0.2]], requires_grad=True)
    target = torch.tensor([0])
    criterion = LabelDistributionallyRobustLoss(margin=2.0, temperature=1.0)

    base_loss = criterion(logits, target)
    improved_loss = criterion(logits + torch.tensor([[1.5, 0.0, 0.0]]), target)
    assert improved_loss < base_loss

    base_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0.0


def test_ldr_kl_supports_soft_targets_and_class_weights() -> None:
    logits = torch.tensor([[0.4, -0.2, 0.8], [0.1, 0.6, -0.5]])
    targets = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.4, 0.6]])
    plain = LabelDistributionallyRobustLoss(margin=1.5, temperature=0.7)
    weighted = LabelDistributionallyRobustLoss(
        weight=torch.tensor([1.0, 2.0, 3.0]),
        margin=1.5,
        temperature=0.7,
    )

    plain_losses = plain.per_sample_loss(logits, targets)
    weighted_losses = weighted.per_sample_loss(logits, targets)
    expected_weights = torch.tensor([1.3, 2.6])

    torch.testing.assert_close(weighted_losses, plain_losses * expected_weights)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"margin": -0.1}, "margin"),
        ({"temperature": 0.0}, "temperature"),
    ],
)
def test_ldr_kl_rejects_invalid_parameters(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LabelDistributionallyRobustLoss(**kwargs)
