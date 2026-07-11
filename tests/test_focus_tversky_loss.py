import torch

from trkh.training.train import _focus_tversky_loss_from_logits


def test_focus_tversky_loss_prefers_correct_focus_predictions():
    targets = torch.tensor([1, 0, 2, 1])
    good_logits = torch.tensor(
        [
            [0.0, 4.0, 0.0],
            [3.0, -2.0, 0.0],
            [0.0, -2.0, 3.0],
            [0.0, 4.0, 0.0],
        ]
    )
    bad_logits = torch.tensor(
        [
            [3.0, -2.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 3.0],
            [3.0, -2.0, 0.0],
        ]
    )

    good_loss, good_stats = _focus_tversky_loss_from_logits(
        logits=good_logits,
        targets=targets,
        focus_class=1,
    )
    bad_loss, bad_stats = _focus_tversky_loss_from_logits(
        logits=bad_logits,
        targets=targets,
        focus_class=1,
    )

    assert good_loss < bad_loss
    assert good_stats["index"] > bad_stats["index"]
    assert good_stats["tp"] > bad_stats["tp"]


def test_focus_tversky_alpha_penalizes_false_positive_focus_mass():
    targets = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor(
        [
            [0.0, 3.0],
            [0.0, 2.5],
            [0.0, 3.0],
            [0.0, 3.0],
        ],
        requires_grad=True,
    )

    low_fp_loss, _ = _focus_tversky_loss_from_logits(
        logits=logits,
        targets=targets,
        focus_class=1,
        alpha=0.30,
        beta=0.70,
    )
    high_fp_loss, _ = _focus_tversky_loss_from_logits(
        logits=logits,
        targets=targets,
        focus_class=1,
        alpha=0.80,
        beta=0.20,
    )

    assert high_fp_loss > low_fp_loss
    high_fp_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
