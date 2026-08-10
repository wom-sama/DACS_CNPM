import torch

from trkh.training.losses import LogitNormCrossEntropyLoss


def test_logit_norm_loss_is_invariant_to_positive_logit_scale():
    loss_fn = LogitNormCrossEntropyLoss(temperature=0.04)
    logits = torch.tensor(
        [[2.0, -0.5, 0.25], [-1.0, 1.5, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])

    base_loss = loss_fn(logits, targets)
    scaled_loss = loss_fn(logits * 7.0, targets)

    assert torch.allclose(base_loss, scaled_loss, atol=1e-6)
    base_loss.backward()
    assert logits.grad is not None


def test_logit_norm_loss_supports_soft_targets_and_class_weights():
    loss_fn = LogitNormCrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0, 0.5]),
        label_smoothing=0.02,
        temperature=0.05,
    )
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor(
        [
            [0.90, 0.10, 0.00],
            [0.20, 0.70, 0.10],
            [0.00, 0.25, 0.75],
            [0.33, 0.33, 0.34],
        ],
        dtype=torch.float32,
    )

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
