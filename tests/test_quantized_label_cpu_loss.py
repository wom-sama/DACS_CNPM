import torch
import torch.nn.functional as F

from trkh.training.train import _quantized_label_cpu_loss_from_logits


def test_quantized_label_cpu_loss_prefers_correct_logits() -> None:
    targets = torch.tensor([0, 1, 2, 4, 0, 1, 2, 4], dtype=torch.long)
    good_logits = torch.full((targets.numel(), 5), -2.0)
    good_logits[torch.arange(targets.numel()), targets] = 3.0
    bad_logits = -good_logits

    good_loss, good_stats = _quantized_label_cpu_loss_from_logits(
        good_logits,
        targets,
        classes="0,1,2,4",
        min_prior=0.02,
        max_prior=0.60,
    )
    bad_loss, bad_stats = _quantized_label_cpu_loss_from_logits(
        bad_logits,
        targets,
        classes="0,1,2,4",
        min_prior=0.02,
        max_prior=0.60,
    )

    assert good_stats["class_count"] == 4.0
    assert bad_stats["class_count"] == 4.0
    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert good_loss.item() < bad_loss.item()


def test_quantized_label_cpu_loss_supports_soft_targets_and_gradients() -> None:
    hard_targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    soft_targets = F.one_hot(hard_targets, num_classes=5).to(dtype=torch.float32)
    soft_targets[1] = torch.tensor([0.25, 0.50, 0.25, 0.0, 0.0])
    logits = torch.randn(4, 5, requires_grad=True)

    loss, stats = _quantized_label_cpu_loss_from_logits(
        logits,
        soft_targets,
        classes="0,1,2,4",
        min_prior=0.02,
        max_prior=0.60,
    )
    loss.backward()

    assert stats["class_count"] == 4.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_quantized_label_cpu_loss_skips_missing_positive_classes() -> None:
    targets = torch.tensor([0, 0, 0], dtype=torch.long)
    logits = torch.randn(3, 5)

    loss, stats = _quantized_label_cpu_loss_from_logits(
        logits,
        targets,
        classes="2,4",
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert stats["class_count"] == 0.0
