from __future__ import annotations

import torch

import trkh.training.train as train_module
from trkh.training.train import (
    _surface_amplified_boundary_margin_loss,
    _surface_counterfactual_images,
)


def test_surface_amplified_boundary_margin_penalizes_close_neighbor() -> None:
    targets = torch.tensor([1, 2, 4, 0])
    separated_logits = torch.tensor(
        [
            [0.2, 2.0, 0.1, -0.2, -0.5],
            [0.1, 0.0, 2.2, 0.4, -0.3],
            [0.3, 0.2, 0.1, 0.0, 2.1],
            [1.9, 0.3, 0.1, -0.1, 0.0],
        ]
    )
    close_logits = torch.tensor(
        [
            [0.2, 0.5, 0.48, -0.2, -0.5],
            [0.1, 0.0, 0.45, 0.44, -0.3],
            [0.44, 0.43, 0.42, 0.41, 0.45],
            [0.5, 0.49, 0.1, -0.1, 0.0],
        ]
    )

    separated_loss, separated_terms = _surface_amplified_boundary_margin_loss(
        separated_logits,
        targets,
        pairs="0-1,1-2,2-3,4-rest",
        margin=0.10,
        num_classes=5,
    )
    close_loss, close_terms = _surface_amplified_boundary_margin_loss(
        close_logits,
        targets,
        pairs="0-1,1-2,2-3,4-rest",
        margin=0.10,
        num_classes=5,
    )

    assert separated_terms == 4
    assert close_terms == 4
    assert float(close_loss.item()) > float(separated_loss.item())


def test_surface_counterfactual_images_preserve_shape_and_finiteness() -> None:
    images = torch.rand(3, 3, 64, 64) * 1.6 - 0.8
    amplified = _surface_counterfactual_images(
        images,
        mode="foreground_luma",
        strength=0.35,
        blur_kernel=7,
    )

    assert tuple(amplified.shape) == tuple(images.shape)
    assert torch.isfinite(amplified).all()
    assert float((amplified - images).abs().mean().item()) > 0.001


def test_surface_amplified_auxiliary_forward_keeps_bbox_metadata(monkeypatch) -> None:
    calls = {}

    def fake_forward_model_outputs(
        model,
        images,
        image_valid_mask=None,
        bbox_metadata=None,
        bbox_token_prior=None,
        **kwargs,
    ):
        calls["image_valid_mask"] = image_valid_mask.clone()
        calls["bbox_metadata"] = bbox_metadata.clone()
        calls["bbox_token_prior"] = bbox_token_prior.clone()
        return None, torch.zeros(images.size(0), 5, dtype=images.dtype)

    monkeypatch.setattr(train_module, "_forward_model_outputs", fake_forward_model_outputs)

    images = torch.rand(2, 3, 32, 32)
    targets = torch.tensor([1, 2])
    image_valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    bbox_metadata = torch.tensor([[0.50, 0.50, 0.60, 0.70], [0.40, 0.45, 0.35, 0.42]])
    bbox_token_prior = torch.tensor([[0.52, 0.51, 0.55, 0.64], [0.44, 0.48, 0.30, 0.38]])

    _, _, fraction, _ = train_module._surface_amplified_supervised_losses(
        model=torch.nn.Identity(),
        criterion=torch.nn.CrossEntropyLoss(),
        images=images,
        targets=targets,
        image_valid_mask=image_valid_mask,
        bbox_metadata=bbox_metadata,
        bbox_token_prior=bbox_token_prior,
        probability=1.0,
        mode="foreground_luma",
        strength=0.20,
        blur_kernel=7,
        boundary_pairs="0-1,1-2",
        boundary_margin=0.10,
        amp=False,
        device=torch.device("cpu"),
        num_classes=5,
    )

    assert fraction == 1.0
    assert torch.equal(calls["image_valid_mask"], image_valid_mask)
    assert torch.equal(calls["bbox_metadata"], bbox_metadata)
    assert torch.equal(calls["bbox_token_prior"], bbox_token_prior)


def test_surface_counterfactual_consistency_forward_keeps_bbox_metadata(monkeypatch) -> None:
    calls = {}

    def fake_forward_model_outputs(
        model,
        images,
        image_valid_mask=None,
        bbox_metadata=None,
        bbox_token_prior=None,
        **kwargs,
    ):
        calls["image_valid_mask"] = image_valid_mask.clone()
        calls["bbox_metadata"] = bbox_metadata.clone()
        calls["bbox_token_prior"] = bbox_token_prior.clone()
        return None, torch.zeros(images.size(0), 5, dtype=images.dtype)

    monkeypatch.setattr(train_module, "_forward_model_outputs", fake_forward_model_outputs)

    images = torch.rand(2, 3, 32, 32)
    logits = torch.tensor([[0.1, 0.9, 0.0, -0.2, -0.3], [0.2, 0.1, 0.8, -0.4, 0.0]])
    image_valid_mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    bbox_metadata = torch.tensor([[0.50, 0.50, 0.60, 0.70], [0.40, 0.45, 0.35, 0.42]])
    bbox_token_prior = torch.tensor([[0.52, 0.51, 0.55, 0.64], [0.44, 0.48, 0.30, 0.38]])

    _, fraction = train_module._surface_counterfactual_consistency_loss(
        model=torch.nn.Identity(),
        images=images,
        logits=logits,
        image_valid_mask=image_valid_mask,
        bbox_metadata=bbox_metadata,
        bbox_token_prior=bbox_token_prior,
        probability=1.0,
        mode="foreground_luma",
        strength=0.20,
        blur_kernel=7,
        temperature=1.0,
        amp=False,
        device=torch.device("cpu"),
    )

    assert fraction == 1.0
    assert torch.equal(calls["image_valid_mask"], image_valid_mask)
    assert torch.equal(calls["bbox_metadata"], bbox_metadata)
    assert torch.equal(calls["bbox_token_prior"], bbox_token_prior)
