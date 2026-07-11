import torch

from trkh.training.train import _focus_auc_rank_loss_from_logits


def test_focus_auc_rank_loss_prefers_positive_scores_above_negatives():
    targets = torch.tensor([1, 1, 0, 2], dtype=torch.long)
    good_logits = torch.tensor(
        [
            [0.0, 3.4, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.1, 0.0, 0.0],
            [3.0, -0.5, 0.0, 0.0, 1.0],
            [0.0, -0.4, 3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bad_logits = torch.tensor(
        [
            [2.6, -0.2, 0.0, 0.0, 0.5],
            [0.0, -0.4, 2.4, 0.0, 0.0],
            [0.1, 2.6, 0.0, 0.0, 0.0],
            [0.0, 2.3, 0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    good_loss, good_stats = _focus_auc_rank_loss_from_logits(
        logits=good_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        hard_fraction=0.5,
    )
    bad_loss, bad_stats = _focus_auc_rank_loss_from_logits(
        logits=bad_logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
        hard_fraction=0.5,
    )

    assert good_loss < bad_loss
    assert good_stats["pair_count"].item() == 4
    assert good_stats["selected_pair_fraction"].item() == 0.5
    assert good_stats["score_gap"].item() > bad_stats["score_gap"].item()


def test_focus_auc_rank_loss_noops_without_positive_or_negative_pairs():
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -0.8, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    loss, stats = _focus_auc_rank_loss_from_logits(
        logits=logits,
        targets=targets,
        focus_class=1,
        negative_classes="0,2,4",
    )

    assert loss.item() == 0.0
    assert stats["positive_count"].item() == 0.0
    assert stats["pair_count"].item() == 0.0
