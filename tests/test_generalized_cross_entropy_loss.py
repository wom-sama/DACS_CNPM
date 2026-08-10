import torch
import torch.nn.functional as F

from trkh.training.losses import GeneralizedCrossEntropyLoss


def test_gce_per_sample_caps_confident_wrong_loss() -> None:
    logits = torch.tensor([[6.0, -4.0], [6.0, -4.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    criterion = GeneralizedCrossEntropyLoss(q=0.7)

    per_sample = criterion.per_sample_loss(logits, targets)
    ce_per_sample = F.cross_entropy(logits, targets, reduction="none")

    assert per_sample.shape == (2,)
    assert torch.isfinite(per_sample).all()
    assert per_sample[1] < ce_per_sample[1]
    assert per_sample[1] > per_sample[0]


def test_gce_supports_soft_targets_and_class_multipliers() -> None:
    logits = torch.tensor([[2.0, -0.5, 0.0], [-0.5, 2.0, 0.0]], dtype=torch.float32)
    soft_targets = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.80, 0.20]], dtype=torch.float32)
    criterion = GeneralizedCrossEntropyLoss(q=0.8, label_smoothing=0.01)

    base = criterion.per_sample_loss(logits, soft_targets)
    criterion.set_class_weight_multipliers(torch.tensor([1.0, 2.0, 1.0]))
    weighted = criterion.per_sample_loss(logits, soft_targets)

    assert torch.isfinite(base).all()
    assert torch.isfinite(weighted).all()
    assert weighted.mean() > base.mean()


def test_ldam_gce_has_per_sample_loss() -> None:
    logits = torch.tensor([[1.0, 0.5, -0.25], [-0.25, 0.5, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)
    criterion = GeneralizedCrossEntropyLoss(
        class_counts=[10, 2, 5],
        q=0.7,
        max_margin=0.3,
        scale=8.0,
    )

    per_sample = criterion.per_sample_loss(logits, targets)

    assert per_sample.shape == (2,)
    assert torch.isfinite(per_sample).all()
