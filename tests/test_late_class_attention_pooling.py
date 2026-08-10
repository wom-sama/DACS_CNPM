from __future__ import annotations

import torch

from trkh.models.model import (
    LateClassAttentionPooling,
    VisionTransformerWithRegisters,
    classification_logits_from_features,
)
from trkh.training.train import _apply_trainable_module_prefixes


def test_late_class_attention_pooling_is_noop_but_trainable() -> None:
    module = LateClassAttentionPooling(
        dim=32,
        num_heads=4,
        dropout=0.0,
        mlp_ratio=1.5,
        residual_scale=0.10,
    )
    module.zero_init_residual()
    head_input = torch.randn(2, 32, requires_grad=True)
    patch_tokens = torch.randn(2, 6, 32)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, False, False, False, False],
        ]
    )

    pooled, trace = module(
        head_input,
        patch_tokens,
        valid_mask=valid_mask,
        return_trace=True,
    )

    assert pooled.shape == head_input.shape
    assert torch.allclose(pooled, head_input)
    assert trace["attention"].shape == (2, 6)
    assert torch.all(trace["attention"].masked_select(~valid_mask) == 0.0)

    pooled.sum().backward()
    assert module.residual_projection.weight.grad is not None
    assert module.residual_projection.weight.grad.abs().sum() > 0


def test_late_class_attention_pooling_integrates_with_classifier_trace() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        late_class_attention_pooling=True,
        late_class_attention_heads=4,
        late_class_attention_dropout=0.0,
    )
    features = model.forward_features(torch.rand(2, 3, 32, 32), return_trace=True)
    logits = classification_logits_from_features(model, features)

    assert logits.shape == (2, 3)
    assert features["late_class_attention_feature"].shape == (2, 32)
    assert features["trace"]["late_class_attention"].shape == (2, 16)


def test_trainable_module_prefixes_supports_late_class_attention_pool() -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        late_class_attention_pooling=True,
    )

    summary = _apply_trainable_module_prefixes(
        model,
        ["late_class_attention_pool"],
    )

    assert summary["enabled"] is True
    assert summary["trainable_parameters"] > 0
    assert summary["frozen_parameters"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith("late_class_attention_pool.")
