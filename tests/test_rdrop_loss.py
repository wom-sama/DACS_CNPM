import torch

from trkh.training.train import _rdrop_symmetric_kl_loss


def test_rdrop_loss_zero_for_identical_logits():
    logits = torch.randn(4, 5, requires_grad=True)

    loss = _rdrop_symmetric_kl_loss(logits, logits, temperature=1.0)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_rdrop_loss_backpropagates_to_both_logits():
    logits_a = torch.randn(4, 5, requires_grad=True)
    logits_b = torch.randn(4, 5, requires_grad=True)

    loss = _rdrop_symmetric_kl_loss(logits_a, logits_b, temperature=0.7)
    loss.backward()

    assert loss.item() >= 0.0
    assert logits_a.grad is not None
    assert logits_b.grad is not None
    assert torch.isfinite(logits_a.grad).all()
    assert torch.isfinite(logits_b.grad).all()
    assert logits_a.grad.abs().sum().item() > 0.0
    assert logits_b.grad.abs().sum().item() > 0.0
