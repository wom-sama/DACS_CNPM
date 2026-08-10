from __future__ import annotations

import torch

from trkh.models.model import (
    VisionTransformerWithRegisters,
    classification_logits_from_features,
    paired_view_fused_logits_from_features,
    source_context_fused_logits_from_features,
)


def test_source_context_feature_fusion_is_zero_init_residual() -> None:
    torch.manual_seed(7)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=4,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        source_context_feature_fusion=True,
        source_context_fusion_hidden_dim=16,
        source_context_fusion_dropout=0.0,
        source_context_fusion_logit_scale=0.25,
    )
    model.train()
    primary = torch.randn(2, 3, 32, 32)
    context = torch.randn(2, 3, 32, 32)

    primary_features = model.forward_features(primary)
    context_features = model.forward_features(context)
    primary_logits = classification_logits_from_features(model, primary_features)
    context_logits = classification_logits_from_features(model, context_features)

    fused_logits, trace = source_context_fused_logits_from_features(
        model,
        primary_features,
        context_features,
        primary_logits,
        context_logits,
        return_trace=True,
    )

    assert fused_logits.shape == primary_logits.shape
    assert torch.allclose(fused_logits, primary_logits)
    assert torch.allclose(trace["adjustment"], torch.zeros_like(primary_logits))
    assert 0.0 < float(trace["gate"].mean()) < 0.5

    loss = fused_logits[:, 1].sum()
    loss.backward()
    delta_final = model.source_context_fusion_head.delta[-1]
    assert delta_final.weight.grad is not None
    assert float(delta_final.weight.grad.abs().sum()) > 0.0


def test_paired_view_feature_fusion_is_zero_init_residual() -> None:
    torch.manual_seed(11)
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=4,
        use_cnn_stem=False,
        num_classes=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_registers=1,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        paired_view_feature_fusion=True,
        paired_view_fusion_hidden_dim=16,
        paired_view_fusion_dropout=0.0,
        paired_view_fusion_logit_scale=0.25,
    )
    model.train()
    primary = torch.randn(2, 3, 32, 32)
    paired = torch.randn(2, 3, 32, 32)

    primary_features = model.forward_features(primary)
    paired_features = model.forward_features(paired)
    primary_logits = classification_logits_from_features(model, primary_features)
    paired_logits = classification_logits_from_features(model, paired_features)

    fused_logits, trace = paired_view_fused_logits_from_features(
        model,
        primary_features,
        paired_features,
        primary_logits,
        paired_logits,
        return_trace=True,
    )

    assert fused_logits.shape == primary_logits.shape
    assert torch.allclose(fused_logits, primary_logits)
    assert torch.allclose(trace["adjustment"], torch.zeros_like(primary_logits))
    assert 0.0 < float(trace["gate"].mean()) < 0.5

    loss = fused_logits[:, 1].sum()
    loss.backward()
    delta_final = model.paired_view_fusion_head.delta[-1]
    assert delta_final.weight.grad is not None
    assert float(delta_final.weight.grad.abs().sum()) > 0.0
