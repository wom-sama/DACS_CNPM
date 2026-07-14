from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

from trkh.tools.audit_confusion_aware_spectral_readiness import (
    _directional_probability_derivative,
    _summarize_values,
)
from trkh.training.confusion_spectral import (
    ConfusionSpectralEMAState,
    confusion_aware_spectral_regularizer,
    confusion_frequency_weights,
    differentiable_margin_confusion,
)
from trkh.training.train import _forward_train_loss, build_configs, parse_args


def test_frequency_weights_prioritize_the_rare_class() -> None:
    weights = confusion_frequency_weights([100, 10, 80], smoothing=0.2)

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert weights[1] > weights[2] > weights[0]


def test_differentiable_confusion_is_off_diagonal_and_has_finite_gradients() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5, -0.5],
            [0.2, 1.4, 0.1],
            [0.3, 0.8, 1.1],
            [0.9, 1.0, 0.2],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2, 0])

    confusion = differentiable_margin_confusion(logits, targets, margin=0.1)
    confusion.sum().backward()

    assert confusion.shape == (3, 3)
    assert torch.equal(torch.diagonal(confusion), torch.zeros(3))
    assert bool((confusion >= 0.0).all())
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ema_history_is_constant_for_the_current_gradient() -> None:
    logits = torch.tensor(
        [[1.2, 0.1], [0.4, 1.1], [0.7, 0.9], [1.0, 0.6]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 1, 0])
    previous = torch.full((2, 2), 0.2, requires_grad=True)

    result = confusion_aware_spectral_regularizer(
        logits,
        targets,
        class_counts=[30, 10],
        previous_ema=previous,
        momentum=0.5,
    )
    result.loss.backward()

    assert previous.grad is None
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not result.ema_confusion.requires_grad


def test_bicar_strengthens_rare_focus_false_positive_suppression() -> None:
    probabilities = torch.tensor(
        [
            [0.35, 0.55, 0.04, 0.03, 0.03],
            [0.48, 0.42, 0.04, 0.03, 0.03],
            [0.05, 0.20, 0.65, 0.05, 0.05],
            [0.04, 0.08, 0.08, 0.75, 0.05],
            [0.05, 0.12, 0.05, 0.03, 0.75],
        ],
        dtype=torch.float32,
    )
    targets = torch.arange(5)
    class_counts = [1941, 541, 1920, 2520, 2293]

    derivatives = {}
    for method, bidirectional in (("car", False), ("bicar", True)):
        logits = probabilities.log().requires_grad_(True)
        result = confusion_aware_spectral_regularizer(
            logits,
            targets,
            class_counts=class_counts,
            momentum=0.5,
            smoothing=0.2,
            margin=0.1,
            bidirectional=bidirectional,
        )
        gradient = torch.autograd.grad(result.loss, logits)[0]
        derivatives[method] = _directional_probability_derivative(
            probabilities,
            gradient,
            focus_class=1,
        )

    assert derivatives["car"][0] < 0.0
    assert derivatives["bicar"][0] < derivatives["car"][0]
    assert derivatives["car"][1] > 0.0
    assert derivatives["bicar"][1] > 0.0


def test_direction_summary_uses_the_declared_sign() -> None:
    positive = _summarize_values([0.1, 0.2, -0.1], desired_positive=True)
    negative = _summarize_values([-0.1, -0.2, 0.1], desired_positive=False)

    assert positive["desired_sign_fraction"] == 2.0 / 3.0
    assert negative["desired_sign_fraction"] == 2.0 / 3.0


def test_confusion_spectral_state_round_trip_is_checkpoint_safe() -> None:
    state = ConfusionSpectralEMAState()
    state.update(torch.eye(5))

    restored = ConfusionSpectralEMAState.from_state_dict(
        state.state_dict(),
        num_classes=5,
        device=torch.device("cpu"),
    )

    assert restored.updates == 1
    assert restored.ema_confusion is not None
    assert torch.equal(restored.ema_confusion, torch.eye(5))


def test_bicar_config_uses_the_parser_sam_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trkh.training.train",
            "--confusion-spectral-loss-weight",
            "0.5",
            "--confusion-spectral-bidirectional",
        ],
    )
    _, train_config, _ = build_configs(parse_args())

    assert train_config.confusion_spectral_loss_weight == 0.5
    assert train_config.confusion_spectral_bidirectional is True
    assert train_config.use_sam is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trkh.training.train",
            "--confusion-spectral-loss-weight",
            "0.5",
            "--sam",
        ],
    )
    with pytest.raises(ValueError, match="khong ho tro SAM"):
        build_configs(parse_args())


def test_train_loss_updates_bicar_ema_once_and_replay_does_not_advance_it() -> None:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 5))
    images = torch.randn(10, 1, 2, 2)
    targets = torch.arange(5).repeat_interleave(2)
    state = ConfusionSpectralEMAState()

    loss, _, logits, details, _ = _forward_train_loss(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        images=images,
        labels=targets,
        targets=targets,
        device=torch.device("cpu"),
        amp=False,
        epoch_index=1,
        confusion_spectral_state=state,
        confusion_spectral_loss_weight=0.5,
        confusion_spectral_class_counts=[1941, 541, 1920, 2520, 2293],
        confusion_spectral_ema_momentum=0.5,
        confusion_spectral_frequency_smoothing=0.2,
        confusion_spectral_margin=0.1,
        confusion_spectral_start_epoch=1,
        confusion_spectral_bidirectional=True,
    )
    loss.backward()

    assert logits.shape == (10, 5)
    assert state.updates == 1
    assert state.ema_confusion is not None
    assert details["confusion_spectral_loss"] > 0.0
    assert details["confusion_spectral_weighted_loss"] > 0.0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    ema_before_replay = state.ema_confusion.clone()
    _forward_train_loss(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        images=images,
        labels=targets,
        targets=targets,
        device=torch.device("cpu"),
        amp=False,
        epoch_index=1,
        confusion_spectral_state=state,
        confusion_spectral_loss_weight=0.5,
        confusion_spectral_class_counts=[1941, 541, 1920, 2520, 2293],
        confusion_spectral_ema_momentum=0.5,
        confusion_spectral_frequency_smoothing=0.2,
        confusion_spectral_margin=0.1,
        confusion_spectral_start_epoch=1,
        confusion_spectral_bidirectional=True,
        confusion_spectral_update_state=False,
    )

    assert state.updates == 1
    assert torch.equal(state.ema_confusion, ema_before_replay)
