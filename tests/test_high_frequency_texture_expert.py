from __future__ import annotations

import torch
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import classification_logits_from_features, create_model
from trkh.training.train import (
    _high_frequency_texture_auxiliary_loss_from_features,
    _high_frequency_texture_pairwise_loss_from_features,
)


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_high_frequency_texture_expert_emits_logits_and_trace() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            high_frequency_texture_expert=True,
            high_frequency_texture_hidden_dim=32,
            high_frequency_texture_dropout=0.0,
            high_frequency_texture_analysis_size=32,
            high_frequency_texture_logit_scale=0.20,
            high_frequency_texture_routing=False,
        ),
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)

    assert tuple(logits.shape) == (2, 5)
    assert tuple(features["high_frequency_texture_logits"].shape) == (2, 5)
    assert tuple(features["trace"]["high_frequency_texture_descriptor"].shape) == (2, 66)
    assert tuple(features["trace"]["high_frequency_texture_high_pass"].shape[:2]) == (2, 1)
    assert tuple(features["trace"]["high_frequency_texture_gradient"].shape[:2]) == (2, 1)
    assert tuple(features["trace"]["high_frequency_texture_laplacian"].shape[:2]) == (2, 1)
    assert tuple(features["trace"]["high_frequency_texture_foreground_detail"].shape[:2]) == (2, 1)
    assert tuple(features["trace"]["high_frequency_texture_route_weights"].shape) == (2,)


def test_high_frequency_texture_expert_can_extend_checkpoint_without_changing_logits() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    high_frequency = create_model(
        num_classes=5,
        model_config=_tiny_config(
            high_frequency_texture_expert=True,
            high_frequency_texture_hidden_dim=32,
            high_frequency_texture_dropout=0.0,
            high_frequency_texture_analysis_size=32,
        ),
    )
    missing, unexpected = high_frequency.load_flexible_state_dict(
        base.state_dict(),
        strict=False,
    )
    assert not unexpected
    assert any(str(key).startswith("high_frequency_texture_expert.") for key in missing)

    base.eval()
    high_frequency.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), high_frequency(images), atol=1e-6)


def test_high_frequency_texture_auxiliary_loss_is_finite() -> None:
    features = {"high_frequency_texture_logits": torch.randn(4, 5)}
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    loss = _high_frequency_texture_auxiliary_loss_from_features(
        features=features,
        targets=targets,
        criterion=nn.CrossEntropyLoss(),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_high_frequency_texture_pairwise_loss_is_finite() -> None:
    features = {
        "high_frequency_texture_logits": torch.tensor(
            [
                [1.0, 0.2, -0.3, -0.5, 0.0],
                [0.1, 0.8, -0.4, -0.6, 0.0],
                [-0.2, 0.3, 0.7, -0.1, 0.0],
                [-0.3, 0.1, 0.0, 0.9, 0.0],
            ],
            dtype=torch.float32,
        )
    }
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = _high_frequency_texture_pairwise_loss_from_features(
        features=features,
        targets=targets,
        pairs="0-1,1-2,2-3",
        num_classes=5,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
