from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    ComplementaryPatchSuppressionHead,
    VisionTransformerWithRegisters,
    classification_logits_from_features,
    create_model,
)
from trkh.training.train import _apply_trainable_module_prefixes


def test_complementary_patch_suppression_head_is_zero_init_trainable() -> None:
    module = ComplementaryPatchSuppressionHead(
        dim=24,
        num_classes=3,
        top_k=2,
        hidden_dim=32,
        dropout=0.0,
        suppression_strength=0.75,
        bbox_weight=0.30,
    )
    module.zero_init_residual()
    head_input = torch.randn(2, 24, requires_grad=True)
    patch_tokens = torch.randn(2, 5, 24)
    bbox_prior = torch.tensor(
        [
            [1.0, 0.8, 0.2, 0.0, 0.0],
            [0.0, 0.2, 0.9, 0.7, 0.0],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )

    logits, trace = module(
        head_input,
        patch_tokens,
        bbox_prior=bbox_prior,
        valid_mask=valid_mask,
        return_trace=True,
    )

    assert logits.shape == (2, 3)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    assert trace["suppressed_mask"].shape == (2, 5)
    assert trace["complement_attention"].shape == (2, 5)
    assert torch.all(trace["complement_attention"].masked_select(~valid_mask) == 0.0)

    logits.sum().backward()
    final_linear = module.net[-1]
    assert isinstance(final_linear, torch.nn.Linear)
    assert final_linear.weight.grad is not None
    assert final_linear.weight.grad.abs().sum() > 0


def test_complementary_patch_suppression_integrates_as_noop_trace() -> None:
    torch.manual_seed(13)
    base = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
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
        attention_dropout=0.0,
        drop_path_rate=0.0,
        complementary_patch_suppression_head=True,
        complementary_patch_suppression_top_k=2,
        complementary_patch_suppression_dropout=0.0,
        complementary_patch_suppression_logit_scale=0.12,
    )
    missing, unexpected = model.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("complementary_patch_suppression_head.") for key in missing)

    images = torch.randn(2, 3, 32, 32)
    base.eval()
    model.eval()
    with torch.inference_mode():
        base_features = base.forward_features(images)
        base_logits = classification_logits_from_features(base, base_features)
        features = model.forward_features(images, return_trace=True)
        features["patch_bbox_prior"] = torch.ones(2, 16)
        logits = classification_logits_from_features(model, features)

    assert torch.allclose(base_logits, logits, atol=1e-6)
    assert features["complementary_patch_suppression_logits"].shape == (2, 3)
    assert features["trace"]["complementary_patch_attention"].shape == (2, 16)
    assert features["trace"]["complementary_patch_suppressed_indices"].shape == (2, 2)


def test_trainable_module_prefixes_supports_complementary_patch_suppression() -> None:
    model = create_model(
        num_classes=3,
        model_config=ModelConfig(
            model_type="vit_registers",
            image_size=32,
            patch_size=8,
            use_cnn_stem=False,
            embed_dim=32,
            depth=1,
            num_heads=4,
            num_registers=1,
            complementary_patch_suppression_head=True,
        ),
    )

    summary = _apply_trainable_module_prefixes(
        model,
        ["complementary_patch_suppression_head"],
    )

    assert summary["enabled"] is True
    assert summary["trainable_parameters"] > 0
    assert summary["frozen_parameters"] > 0
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith(
            "complementary_patch_suppression_head."
        )
