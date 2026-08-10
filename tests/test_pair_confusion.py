import pytest
import torch

from trkh.training.pair_confusion import bidirectional_pair_confusion_mean


def test_pair_confusion_mean_updates_both_unequal_directions() -> None:
    logits = torch.tensor(
        [[0.0, 3.0], [1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])
    result = bidirectional_pair_confusion_mean(logits, targets)
    gradient = torch.autograd.grad(result.loss, logits)[0]

    assert result.directional_confusions[1] > result.directional_confusions[0]
    assert gradient[0, 0] < 0 and gradient[0, 1] > 0
    assert gradient[1, 0] > 0 and gradient[1, 1] < 0
    assert bool((gradient.abs().sum(dim=1) > 0).all().item())


def test_pair_confusion_mean_detaches_ema_history() -> None:
    logits = torch.tensor(
        [[0.0, 2.0], [2.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    history = torch.tensor(
        [[0.0, 0.2], [0.7, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    result = bidirectional_pair_confusion_mean(
        logits,
        torch.tensor([0, 1]),
        previous_ema=history,
        momentum=0.5,
    )
    result.loss.backward()

    assert history.grad is None
    assert logits.grad is not None
    assert bool((logits.grad.abs().sum(dim=1) > 0).all().item())


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (torch.randn(3, 5), torch.tensor([0, 1, 0]), "shape"),
        (torch.randn(3, 2), torch.tensor([0, 0, 0]), "both pair classes"),
        (torch.randn(2, 2), torch.tensor([0, 2]), "only 0 and 1"),
    ],
)
def test_pair_confusion_mean_rejects_invalid_pair_batches(
    logits: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bidirectional_pair_confusion_mean(logits, targets)
