import torch

from trkh.training.losses import SeesawCrossEntropyLoss


def test_seesaw_compensation_increases_false_positive_penalty() -> None:
    logits = torch.tensor([[0.0, 3.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0])
    base = SeesawCrossEntropyLoss(
        class_counts=[100, 10, 100],
        mitigation_power=0.0,
        compensation_power=0.0,
    )
    compensated = SeesawCrossEntropyLoss(
        class_counts=[100, 10, 100],
        mitigation_power=0.0,
        compensation_power=2.0,
    )

    base_loss = base(logits, targets)
    compensated_loss = compensated(logits, targets)

    assert compensated_loss.item() > base_loss.item()
    compensated_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_seesaw_mitigation_reduces_rare_negative_penalty() -> None:
    logits = torch.tensor([[0.0, 3.0, 0.0]])
    targets = torch.tensor([0])
    no_mitigation = SeesawCrossEntropyLoss(
        class_counts=[100, 10, 100],
        mitigation_power=0.0,
        compensation_power=0.0,
    )
    mitigated = SeesawCrossEntropyLoss(
        class_counts=[100, 10, 100],
        mitigation_power=0.8,
        compensation_power=0.0,
    )

    assert mitigated(logits, targets).item() < no_mitigation(logits, targets).item()


def test_seesaw_supports_soft_targets() -> None:
    logits = torch.tensor([[1.0, 0.5, -0.25]], requires_grad=True)
    soft_targets = torch.tensor([[0.8, 0.2, 0.0]])
    criterion = SeesawCrossEntropyLoss(
        class_counts=[100, 10, 100],
        mitigation_power=0.0,
        compensation_power=1.5,
        label_smoothing=0.01,
    )

    loss = criterion(logits, soft_targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
