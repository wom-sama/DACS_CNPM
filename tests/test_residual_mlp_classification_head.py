from __future__ import annotations

import torch

from trkh.models.model import (
    VisionTransformerWithRegisters,
    classification_logits_from_features,
)
from trkh.training.train import _load_model_state_allowing_extensions


def _tiny_model(*, classification_mlp_head: bool) -> VisionTransformerWithRegisters:
    return VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        classification_mlp_head=classification_mlp_head,
        classification_mlp_hidden_dim=48,
        classification_mlp_dropout=0.0,
        classification_mlp_residual_scale=0.20,
    )


def test_residual_mlp_head_preserves_linear_logits_at_zero_init() -> None:
    base = _tiny_model(classification_mlp_head=False)
    extended = _tiny_model(classification_mlp_head=True)
    with torch.no_grad():
        extended.head.weight.copy_(base.head.weight)
        extended.head.bias.copy_(base.head.bias)

    features = {"pooled": torch.randn(4, 32)}

    base_logits = classification_logits_from_features(base, dict(features))
    extended_logits = classification_logits_from_features(extended, dict(features))

    assert torch.allclose(extended_logits, base_logits, atol=1e-6)


def test_residual_mlp_head_resume_allows_only_residual_extension_keys() -> None:
    base = _tiny_model(classification_mlp_head=False)
    extended = _tiny_model(classification_mlp_head=True)

    summary = _load_model_state_allowing_extensions(
        extended,
        base.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    missing = set(summary["allowed_missing_keys"])
    assert missing
    assert all(key.startswith("head.residual_") for key in missing)
    assert summary["unexpected_keys"] == []
    assert torch.equal(extended.head.weight, base.head.weight)
    assert torch.equal(extended.head.bias, base.head.bias)
