from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import classification_logits_from_features, create_model
from trkh.training.train import _patch_objectness_loss_from_features


def _tiny_model_config(**overrides):
    config = dict(
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
    config.update(overrides)
    return ModelConfig(**config)


def test_patch_objectness_head_can_extend_existing_checkpoint() -> None:
    base = create_model(num_classes=5, model_config=_tiny_model_config())
    extended = create_model(
        num_classes=5,
        model_config=_tiny_model_config(
            patch_objectness_guided_head=True,
            patch_objectness_dropout=0.0,
        ),
    )

    missing, unexpected = extended.load_flexible_state_dict(base.state_dict(), strict=False)

    assert not unexpected
    assert any(str(key).startswith("patch_objectness_guided_head.") for key in missing)

    base.eval()
    extended.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), extended(images), atol=1e-6)


def test_patch_objectness_logits_are_used_traced_and_lossable() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_model_config(
            patch_objectness_guided_head=True,
            patch_objectness_dropout=0.0,
            patch_objectness_logit_scale=0.20,
        ),
    )
    with torch.no_grad():
        model.patch_objectness_guided_head.net[-1].bias.fill_(0.25)
    model.eval()

    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.35, 0.45, 0.40, 0.60],
        ],
        dtype=torch.float32,
    )
    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=True)
    logits = classification_logits_from_features(model, features)
    base_logits = model.head(model.head_input_from_features(features))
    loss = _patch_objectness_loss_from_features(
        features=features,
        positive_weight=1.25,
    )

    assert torch.allclose(logits - base_logits, torch.full_like(logits, 0.05), atol=1e-5)
    assert tuple(features["patch_objectness_class_logits"].shape) == (2, 5)
    assert tuple(features["patch_objectness_logits"].shape) == (2, 16)
    assert tuple(features["patch_bbox_prior"].shape) == (2, 16)
    assert tuple(features["trace"]["patch_objectness_attention"].shape) == (2, 16)
    assert tuple(features["trace"]["patch_objectness_probability"].shape) == (2, 16)
    assert torch.allclose(
        features["trace"]["patch_objectness_attention"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.isfinite(loss)


def test_patch_objectness_training_path_keeps_token_logits_without_trace() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_model_config(
            patch_objectness_guided_head=True,
            patch_objectness_dropout=0.0,
        ),
    )
    model.train()

    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.35, 0.45, 0.40, 0.60],
        ],
        dtype=torch.float32,
    )
    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=False)
    classification_logits_from_features(model, features)

    assert "trace" not in features
    assert tuple(features["patch_objectness_logits"].shape) == (2, 16)
    assert tuple(features["patch_objectness_probability"].shape) == (2, 16)


def test_patch_objectness_loss_rewards_bbox_aligned_logits() -> None:
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    good = {
        "patch_objectness_logits": torch.tensor([[-4.0, 4.0, -4.0, 4.0]]),
        "patch_bbox_prior": target,
    }
    bad = {
        "patch_objectness_logits": torch.tensor([[4.0, -4.0, 4.0, -4.0]]),
        "patch_bbox_prior": target,
    }

    assert _patch_objectness_loss_from_features(features=good) < _patch_objectness_loss_from_features(
        features=bad
    )
