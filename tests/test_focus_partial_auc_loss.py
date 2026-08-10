import torch

from trkh.training.train import _focus_partial_auc_loss_from_logits


def test_focus_partial_auc_targets_high_risk_negative_tail():
    targets = torch.tensor([0, 0, 2, 4, 3], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [3.0, -1.0, 0.0, 0.0, 0.1],
            [2.5, -0.8, 0.0, 0.0, 0.2],
            [0.0, -0.6, 2.8, 0.0, 0.0],
            [0.1, -0.7, 0.0, 0.0, 2.7],
            [0.0, 0.2, 0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bad_logits = good_logits.clone()
    bad_logits[1, 1] = 3.0

    good_loss, good_stats = _focus_partial_auc_loss_from_logits(
        logits=good_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        negative_fraction=0.25,
        positive_fraction=0.50,
        min_negative_probability=0.0,
    )
    bad_loss, bad_stats = _focus_partial_auc_loss_from_logits(
        logits=bad_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        negative_fraction=0.25,
        positive_fraction=0.50,
        min_negative_probability=0.0,
    )

    assert good_loss < bad_loss
    assert good_stats["negative_count"].item() == 4.0
    assert good_stats["selected_negative_count"].item() == 1.0
    assert good_stats["selected_negative_fraction"].item() == 0.25
    assert bad_stats["negative_score"].item() > good_stats["negative_score"].item()

    differentiable_logits = bad_logits.clone().requires_grad_(True)
    loss, _ = _focus_partial_auc_loss_from_logits(
        logits=differentiable_logits,
        targets=targets,
        focus_class=1,
        negative_classes=[0, 2, 4],
        negative_fraction=0.25,
        min_negative_probability=0.0,
    )
    loss.backward()
    assert torch.isfinite(differentiable_logits.grad).all()
    assert differentiable_logits.grad[1, 1].item() > 0.0


def test_focus_partial_auc_protects_low_margin_focus_positives():
    targets = torch.tensor([1, 1, 3], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [0.0, 3.2, 0.1, 0.0, 0.0],
            [0.2, 2.9, 0.0, 0.0, 0.1],
            [0.0, 0.5, 0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bad_logits = torch.tensor(
        [
            [2.8, 0.6, 0.1, 0.0, 0.0],
            [0.2, 0.5, 2.7, 0.0, 0.1],
            [0.0, 0.5, 0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    good_loss, good_stats = _focus_partial_auc_loss_from_logits(
        logits=good_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        negative_fraction=0.25,
        positive_fraction=0.50,
        positive_weight=1.0,
    )
    bad_loss, bad_stats = _focus_partial_auc_loss_from_logits(
        logits=bad_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        negative_fraction=0.25,
        positive_fraction=0.50,
        positive_weight=1.0,
    )

    assert good_loss < bad_loss
    assert good_stats["positive_count"].item() == 2.0
    assert good_stats["selected_positive_count"].item() == 1.0
    assert good_stats["selected_positive_fraction"].item() == 0.5
    assert good_stats["positive_score"].item() > bad_stats["positive_score"].item()


def test_focus_partial_auc_noops_without_focus_or_configured_negatives():
    logits = torch.tensor(
        [
            [2.0, -0.5, 0.0, 0.0, 0.0],
            [0.0, -0.3, 2.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    loss, stats = _focus_partial_auc_loss_from_logits(
        logits=logits,
        targets=targets,
        focus_class=1,
        negative_classes="4",
    )

    assert loss.item() == 0.0
    assert stats["positive_count"].item() == 0.0
    assert stats["negative_count"].item() == 0.0
    assert stats["selected_negative_count"].item() == 0.0
