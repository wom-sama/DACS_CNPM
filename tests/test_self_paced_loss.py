import torch
import pytest

from trkh.training.train import (
    _cyflod_damped_classification_loss_from_per_sample,
    _self_paced_classification_loss_from_per_sample,
)


def test_self_paced_loss_downweights_high_loss_samples_and_keeps_gradient():
    per_sample = torch.tensor([0.2, 0.4, 1.0, 2.0], requires_grad=True)

    loss, stats = _self_paced_classification_loss_from_per_sample(
        per_sample,
        percentile=0.50,
        gamma=1.0,
        min_weight=0.25,
    )

    assert torch.isfinite(loss)
    assert loss.item() < per_sample.mean().item()
    assert 0.0 < stats["weight_mean"].item() < 1.0
    assert 0.0 < stats["high_loss_weight_mean"].item() < 1.0
    assert stats["selected_fraction"].item() == 0.5

    loss.backward()
    assert per_sample.grad is not None
    assert torch.isfinite(per_sample.grad).all()
    assert per_sample.grad[-1].item() < per_sample.grad[0].item()


def test_self_paced_loss_composes_sample_weights_and_ignores_zero_weight_rows():
    per_sample = torch.tensor([0.2, 0.6, 3.0], requires_grad=True)
    sample_weights = torch.tensor([1.0, 0.0, 1.0])

    loss, stats = _self_paced_classification_loss_from_per_sample(
        per_sample,
        sample_weights=sample_weights,
        percentile=0.50,
        gamma=1.0,
        min_weight=0.20,
    )

    assert torch.isfinite(loss)
    assert stats["selected_fraction"].item() == 0.5
    assert stats["weight_mean"].item() < 1.0

    loss.backward()
    assert per_sample.grad is not None
    assert per_sample.grad[1].item() == 0.0


def test_self_paced_loss_class_balanced_selects_within_each_class():
    per_sample = torch.tensor([0.1, 0.2, 0.3, 2.0, 2.1, 5.0], requires_grad=True)
    targets = torch.tensor([0, 0, 0, 1, 1, 1])

    loss, stats = _self_paced_classification_loss_from_per_sample(
        per_sample,
        target_indices=targets,
        percentile=0.50,
        gamma=1.0,
        min_weight=0.10,
        class_balanced=True,
    )

    assert torch.isfinite(loss)
    assert stats["class_count"].item() == 2.0
    assert stats["selected_fraction"].item() == pytest.approx(4.0 / 6.0)
    assert 0.0 <= stats["normalized_loss"].item() <= 1.0

    loss.backward()
    assert per_sample.grad is not None
    assert per_sample.grad[3].item() > per_sample.grad[5].item()


def test_cyflod_loss_damping_downweights_low_target_probability_samples():
    per_sample = torch.tensor([0.1, 0.5, 2.0, 4.0], requires_grad=True)

    loss, stats = _cyflod_damped_classification_loss_from_per_sample(
        per_sample,
        delta=0.50,
        cycle_epochs=2,
        epoch_index=1,
        min_weight=0.05,
    )

    assert torch.isfinite(loss)
    assert loss.item() < per_sample.mean().item()
    assert stats["active_delta"].item() == 0.50
    assert 0.0 < stats["weight_mean"].item() < 1.0
    assert 0.0 < stats["damped_fraction"].item() < 1.0
    assert 0.0 < stats["damped_weight_mean"].item() < 1.0

    loss.backward()
    assert per_sample.grad is not None
    assert torch.isfinite(per_sample.grad).all()
    assert per_sample.grad[-1].item() < per_sample.grad[0].item()


def test_cyflod_loss_damping_composes_sample_weights():
    per_sample = torch.tensor([0.1, 3.0, 4.0], requires_grad=True)
    sample_weights = torch.tensor([1.0, 0.0, 1.0])

    loss, stats = _cyflod_damped_classification_loss_from_per_sample(
        per_sample,
        sample_weights=sample_weights,
        delta=0.50,
        cycle_epochs=2,
        epoch_index=1,
        min_weight=0.10,
    )

    assert torch.isfinite(loss)
    assert stats["damped_fraction"].item() == 0.5

    loss.backward()
    assert per_sample.grad is not None
    assert per_sample.grad[1].item() == 0.0
