from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models import model as model_module
from trkh.models.model import classification_logits_from_features, create_model


def _tiny_config(**overrides) -> ModelConfig:
    values = {
        "model_type": "vit_registers",
        "image_size": 32,
        "patch_size": 8,
        "use_cnn_stem": False,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "num_registers": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_bbox_prior_patch_context_head_prefers_bbox_patch() -> None:
    head = model_module.BBoxPriorPatchContextHead(
        dim=8,
        num_classes=5,
        hidden_dim=16,
        dropout=0.0,
        temperature=0.50,
    )
    patch_tokens = torch.zeros(1, 8, 8)
    bbox_prior = torch.zeros(1, 8)
    bbox_prior[0, 3] = 1.0

    _, trace = head(patch_tokens, bbox_prior=bbox_prior, return_trace=True)

    object_attention = trace["object_attention"][0]
    background_attention = trace["background_attention"][0]
    assert int(object_attention.argmax().item()) == 3
    assert torch.allclose(object_attention.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(background_attention.sum(), torch.tensor(1.0), atol=1e-6)
    assert float(background_attention[3].item()) < 1e-6


def test_bbox_prior_patch_context_head_can_extend_existing_checkpoint() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    extended = create_model(
        num_classes=5,
        model_config=_tiny_config(
            bbox_prior_patch_context_head=True,
            bbox_prior_patch_context_dropout=0.0,
        ),
    )

    missing, unexpected = extended.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("bbox_prior_patch_context_head.") for key in missing)

    base.eval()
    extended.eval()
    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        extended_features = extended.forward_features(images, bbox_token_prior=bbox)
        assert torch.allclose(
            base(images),
            classification_logits_from_features(extended, extended_features),
            atol=1e-6,
        )


def test_bbox_prior_patch_context_logits_are_used_and_traced() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            bbox_prior_patch_context_head=True,
            bbox_prior_patch_context_dropout=0.0,
            bbox_prior_patch_context_logit_scale=0.20,
        ),
    )
    with torch.no_grad():
        model.bbox_prior_patch_context_head.net[-1].bias.fill_(0.25)
    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.375, 0.375, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )

    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=True)
    logits = classification_logits_from_features(model, features)
    base_logits = model.head(model.head_input_from_features(features))

    assert torch.allclose(logits - base_logits, torch.full_like(logits, 0.05), atol=1e-5)
    assert tuple(features["bbox_prior_patch_context_logits"].shape) == (2, 5)
    assert tuple(features["patch_bbox_prior"].shape) == (2, 16)
    assert tuple(features["trace"]["bbox_prior_patch_object_attention"].shape) == (2, 16)
    assert tuple(features["trace"]["bbox_prior_patch_background_attention"].shape) == (2, 16)
    assert torch.allclose(
        features["trace"]["bbox_prior_patch_object_attention"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        features["trace"]["bbox_prior_patch_background_attention"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
