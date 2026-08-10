from __future__ import annotations

import pytest
import torch
from torch import Tensor

from trkh.training.label_preserving_hard_positive import (
    generate_projected_hard_positives,
    generate_random_feasible_hard_positives,
)


MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


def _forward(images: Tensor) -> tuple[Tensor, Tensor]:
    channel_means = images.mean(dim=(2, 3))
    features = torch.cat(
        (
            channel_means,
            images[:, :, :2, :2].flatten(1),
        ),
        dim=1,
    )
    logits = torch.stack(
        (
            channel_means[:, 0] - channel_means[:, 1],
            channel_means[:, 1] - channel_means[:, 2],
            channel_means[:, 2] - channel_means[:, 0],
        ),
        dim=1,
    )
    return logits, features


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(19)
    rgb = torch.rand((3, 3, 8, 8), generator=generator) * 0.5 + 0.25
    images = (rgb - 0.5) / 0.25
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    mask = torch.zeros((3, 1, 8, 8), dtype=torch.bool)
    mask[:, :, 1:7, 2:6] = True
    sample_indices = torch.tensor([7, 11, 101], dtype=torch.long)
    return images, labels, mask, sample_indices


def test_projected_hard_positive_is_bounded_masked_and_label_preserving() -> None:
    images, labels, mask, sample_indices = _inputs()
    result = generate_projected_hard_positives(
        images=images,
        labels=labels,
        attack_mask=mask,
        sample_indices=sample_indices,
        forward_logits_features=_forward,
        mean=MEAN,
        std=STD,
        epsilon=2.0 / 255.0,
        steps=5,
        label_margin=0.05,
        seed=20260724,
    )

    assert result.augmented_images.shape == images.shape
    assert float(result.delta_rgb.abs().max()) <= 2.0 / 255.0 + 1e-7
    assert torch.count_nonzero(result.delta_rgb * (~mask).float()) == 0
    assert bool(result.constraint_satisfied.all())
    assert bool((result.true_log_probability_drop <= 0.0500001).all())
    assert bool((result.selected_step >= 0).all())


def test_projected_hard_positive_replay_is_batch_order_independent() -> None:
    images, labels, mask, sample_indices = _inputs()
    kwargs = {
        "forward_logits_features": _forward,
        "mean": MEAN,
        "std": STD,
        "epsilon": 2.0 / 255.0,
        "steps": 5,
        "label_margin": 0.05,
        "seed": 20260724,
    }
    first = generate_projected_hard_positives(
        images=images,
        labels=labels,
        attack_mask=mask,
        sample_indices=sample_indices,
        **kwargs,
    )
    order = torch.tensor([2, 0, 1], dtype=torch.long)
    second = generate_projected_hard_positives(
        images=images[order],
        labels=labels[order],
        attack_mask=mask[order],
        sample_indices=sample_indices[order],
        **kwargs,
    )
    inverse = torch.argsort(order)
    assert torch.equal(first.delta_rgb, second.delta_rgb[inverse])
    assert torch.equal(first.selected_step, second.selected_step[inverse])


def test_random_feasible_control_replays_exactly() -> None:
    images, labels, mask, sample_indices = _inputs()
    kwargs = {
        "images": images,
        "labels": labels,
        "attack_mask": mask,
        "sample_indices": sample_indices,
        "forward_logits_features": _forward,
        "mean": MEAN,
        "std": STD,
        "epsilon": 2.0 / 255.0,
        "candidates": 5,
        "label_margin": 0.05,
        "seed": 20260724,
    }
    first = generate_random_feasible_hard_positives(**kwargs)
    second = generate_random_feasible_hard_positives(**kwargs)
    assert torch.equal(first.delta_rgb, second.delta_rgb)
    assert torch.equal(first.feature_distance, second.feature_distance)
    assert bool(first.constraint_satisfied.all())


def test_hard_positive_rejects_empty_masks() -> None:
    images, labels, mask, sample_indices = _inputs()
    mask[1] = False
    with pytest.raises(ValueError, match="non-empty"):
        generate_projected_hard_positives(
            images=images,
            labels=labels,
            attack_mask=mask,
            sample_indices=sample_indices,
            forward_logits_features=_forward,
            mean=MEAN,
            std=STD,
            epsilon=2.0 / 255.0,
            steps=5,
            label_margin=0.05,
            seed=20260724,
        )
