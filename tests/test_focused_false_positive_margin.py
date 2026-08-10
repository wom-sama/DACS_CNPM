import torch
import pytest
from torch import nn

import trkh.training.train as train_module
from trkh.training.train import (
    _background_focus_suppression_loss,
    _focused_false_positive_margin_loss_from_logits,
    _teacher_pairwise_margin_loss_from_features,
    _teacher_focus_binary_loss_from_logits,
    _teacher_focus_margin_loss_from_logits,
)


def test_focused_false_positive_margin_targets_only_configured_negative_classes():
    logits = torch.tensor(
        [
            [1.0, 1.2, 0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.4],
            [0.0, 1.5, 0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 4, 3], dtype=torch.long)

    loss, fraction, focus_probability = _focused_false_positive_margin_loss_from_logits(
        logits=logits,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes="0,4",
        margin=0.1,
        min_probability=0.05,
        probability_power=1.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction == 0.75
    assert 0.0 < focus_probability.item() < 1.0


def test_focused_false_positive_margin_is_smaller_when_focus_logit_is_well_below_target():
    separated_logits = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0, 1.8],
        ],
        dtype=torch.float32,
    )
    violating_logits = torch.tensor(
        [
            [1.0, 1.2, 0.0, 0.0, 0.0],
            [0.0, 1.1, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 4], dtype=torch.long)

    separated_loss, separated_fraction, _ = _focused_false_positive_margin_loss_from_logits(
        logits=separated_logits,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes=[0, 4],
        margin=0.1,
        min_probability=0.0,
        probability_power=1.5,
    )
    violating_loss, violating_fraction, _ = _focused_false_positive_margin_loss_from_logits(
        logits=violating_logits,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes=[0, 4],
        margin=0.1,
        min_probability=0.0,
        probability_power=1.5,
    )

    assert torch.isfinite(separated_loss)
    assert torch.isfinite(violating_loss)
    assert separated_loss.item() < violating_loss.item()
    assert separated_fraction == 1.0
    assert violating_fraction == 1.0


def test_teacher_focus_margin_uses_teacher_gate_and_agreement():
    logits = torch.tensor(
        [
            [1.0, 1.2, 0.0, 0.0, 0.0],
            [0.0, 1.3, 1.1, 0.0, 0.0],
            [0.0, 1.4, 0.0, 0.0, 1.2],
            [0.0, 1.6, 0.0, 1.7, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2, 4, 3], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.82, 0.04, 0.05, 0.04, 0.05],
            [0.04, 0.03, 0.85, 0.04, 0.04],
            [0.05, 0.62, 0.05, 0.04, 0.24],
            [0.04, 0.02, 0.04, 0.86, 0.04],
        ],
        dtype=torch.float32,
    )

    loss, fraction, focus_probability, teacher_focus_probability = (
        _teacher_focus_margin_loss_from_logits(
            logits=logits,
            teacher_probabilities=teacher_probabilities,
            hard_labels=labels,
            targets=labels,
            focus_class=1,
            negative_classes="0,2,4",
            teacher_max_probability=0.20,
            margin=0.08,
            min_probability=0.05,
            probability_power=1.0,
            require_agreement=True,
        )
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction == pytest.approx(2.0 / 4.0)
    assert 0.0 < focus_probability.item() < 1.0
    assert 0.0 < teacher_focus_probability.item() < 1.0


def test_teacher_focus_margin_is_smaller_when_focus_logit_is_separated():
    labels = torch.tensor([0, 2], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.90, 0.02, 0.03, 0.02, 0.03],
            [0.03, 0.02, 0.90, 0.02, 0.03],
        ],
        dtype=torch.float32,
    )
    separated_logits = torch.tensor(
        [
            [1.8, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.2, 1.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    violating_logits = torch.tensor(
        [
            [1.0, 1.3, 0.0, 0.0, 0.0],
            [0.0, 1.4, 1.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    separated_loss, separated_fraction, _, _ = _teacher_focus_margin_loss_from_logits(
        logits=separated_logits,
        teacher_probabilities=teacher_probabilities,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes=[0, 2],
        teacher_max_probability=0.20,
        margin=0.08,
        min_probability=0.0,
        probability_power=1.0,
        require_agreement=True,
    )
    violating_loss, violating_fraction, _, _ = _teacher_focus_margin_loss_from_logits(
        logits=violating_logits,
        teacher_probabilities=teacher_probabilities,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes=[0, 2],
        teacher_max_probability=0.20,
        margin=0.08,
        min_probability=0.0,
        probability_power=1.0,
        require_agreement=True,
    )

    assert torch.isfinite(separated_loss)
    assert torch.isfinite(violating_loss)
    assert separated_loss.item() < violating_loss.item()
    assert separated_fraction == 1.0
    assert violating_fraction == 1.0


def test_teacher_focus_binary_uses_teacher_focus_probability_and_agreement():
    logits = torch.tensor(
        [
            [1.3, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.8, 0.0, 0.0],
            [0.1, 1.5, 0.0, 0.0, 0.0],
            [0.0, 1.4, 0.0, 1.7, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.86, 0.04, 0.04, 0.03, 0.03],
            [0.04, 0.08, 0.80, 0.04, 0.04],
            [0.08, 0.78, 0.06, 0.04, 0.04],
            [0.04, 0.02, 0.04, 0.86, 0.04],
        ],
        dtype=torch.float32,
    )

    loss, fraction, focus_probability, teacher_focus_probability = (
        _teacher_focus_binary_loss_from_logits(
            logits=logits,
            teacher_probabilities=teacher_probabilities,
            hard_labels=labels,
            targets=labels,
            focus_class=1,
            classes="0,1,2",
            teacher_min_confidence=0.50,
            error_power=0.0,
            hard_target_blend=0.0,
            require_agreement=True,
        )
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction == pytest.approx(3.0 / 4.0)
    assert 0.0 < focus_probability.item() < 1.0
    assert 0.0 < teacher_focus_probability.item() < 1.0


def test_teacher_focus_binary_is_smaller_when_student_matches_teacher_focus():
    labels = torch.tensor([0, 1], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.90, 0.03, 0.03, 0.02, 0.02],
            [0.06, 0.82, 0.04, 0.04, 0.04],
        ],
        dtype=torch.float32,
    )
    matching_logits = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.8, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mismatched_logits = torch.tensor(
        [
            [0.0, 2.8, 0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    matching_loss, matching_fraction, _, _ = _teacher_focus_binary_loss_from_logits(
        logits=matching_logits,
        teacher_probabilities=teacher_probabilities,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        classes=[0, 1],
        teacher_min_confidence=0.0,
        error_power=0.0,
        hard_target_blend=0.0,
        require_agreement=True,
    )
    mismatched_loss, mismatched_fraction, _, _ = _teacher_focus_binary_loss_from_logits(
        logits=mismatched_logits,
        teacher_probabilities=teacher_probabilities,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        classes=[0, 1],
        teacher_min_confidence=0.0,
        error_power=0.0,
        hard_target_blend=0.0,
        require_agreement=True,
    )

    assert torch.isfinite(matching_loss)
    assert torch.isfinite(mismatched_loss)
    assert matching_loss.item() < mismatched_loss.item()
    assert matching_fraction == 1.0
    assert mismatched_fraction == 1.0


class _PairwiseMarginModel(nn.Module):
    pairwise_margin_pairs = [(0, 1), (1, 2), (2, 3), (-1, 4)]


def test_teacher_pairwise_margin_uses_teacher_pair_probabilities_and_agreement():
    pairwise_logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    teacher_probabilities = torch.tensor(
        [
            [0.82, 0.12, 0.02, 0.02, 0.02],
            [0.10, 0.78, 0.08, 0.02, 0.02],
            [0.02, 0.14, 0.76, 0.06, 0.02],
            [0.04, 0.05, 0.04, 0.02, 0.85],
        ],
        dtype=torch.float32,
    )

    loss, fraction, student_probability, teacher_probability = (
        _teacher_pairwise_margin_loss_from_features(
            model=_PairwiseMarginModel(),
            features={"pairwise_margin_logits": pairwise_logits},
            teacher_probabilities=teacher_probabilities,
            hard_labels=labels,
            targets=labels,
            teacher_mass_threshold=0.10,
            error_power=0.0,
            hard_target_blend=0.10,
            require_agreement=True,
        )
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction > 0.0
    assert student_probability.item() == pytest.approx(0.5)
    assert 0.0 < teacher_probability.item() < 1.0
    loss.backward()
    assert pairwise_logits.grad is not None
    assert torch.isfinite(pairwise_logits.grad).all()


class _MeanFocusModel(nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.size(0), 5, device=images.device, dtype=images.dtype)
        logits[:, 0] = 0.2
        logits[:, 1] = images.flatten(1).mean(dim=1) * 4.0
        logits[:, 4] = 0.1
        return logits


def test_background_focus_suppression_targets_negative_background_delta(monkeypatch):
    def fake_counterfactual_images(images, **kwargs):
        return torch.zeros_like(images)

    monkeypatch.setattr(
        train_module,
        "_background_counterfactual_images",
        fake_counterfactual_images,
    )
    images = torch.ones(3, 3, 8, 8, dtype=torch.float32)
    logits = torch.tensor(
        [
            [0.4, 1.6, 0.0, 0.0, 0.0],
            [0.0, 1.7, 0.0, 1.9, 0.0],
            [0.0, 1.5, 0.0, 0.0, 0.3],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 3, 4], dtype=torch.long)

    loss, fraction, focus_probability, delta = _background_focus_suppression_loss(
        model=_MeanFocusModel(),
        images=images,
        logits=logits,
        hard_labels=labels,
        targets=labels,
        bbox_metadata=None,
        probability=1.0,
        focus_class=1,
        negative_classes="0,4",
        margin=0.01,
        min_probability=0.05,
        probability_power=1.0,
        mode="mean",
        counterfactual_margin=0.08,
        blur_kernel=3,
        amp=False,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert fraction == pytest.approx(2.0 / 3.0)
    assert 0.0 < focus_probability.item() < 1.0
    assert delta.item() > 0.0
