from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models import model as model_module
from trkh.models.model import classification_logits_from_features, create_model


def test_micro_detail_expert_prefers_high_detail_foreground_patch() -> None:
    expert = model_module.MicroDetailPatchExpert(
        dim=8,
        num_classes=5,
        top_k=2,
        hidden_dim=16,
        dropout=0.0,
        temperature=0.05,
    )
    patch_tokens = torch.zeros(1, 16, 8)
    detail_map = torch.zeros(1, 1, 4, 4)
    detail_map[0, 0, 1, 1] = 1.0
    patch_indices = torch.arange(16).view(1, 16)
    foreground_prior = torch.ones(1, 16)

    _, trace = expert(
        patch_tokens,
        detail_map=detail_map,
        patch_indices=patch_indices,
        foreground_prior=foreground_prior,
        return_trace=True,
    )

    attention = trace["attention"][0]
    assert int(attention.argmax().item()) == 5
    assert torch.isclose(attention.sum(), torch.tensor(1.0), atol=1e-6)


def test_micro_detail_patch_expert_can_extend_existing_checkpoint() -> None:
    base = create_model(
        num_classes=5,
        model_config=ModelConfig(
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
            detail_patch_enhancement=True,
        ),
    )
    micro = create_model(
        num_classes=5,
        model_config=ModelConfig(
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
            detail_patch_enhancement=True,
            micro_detail_patch_expert=True,
            micro_detail_top_k=3,
            micro_detail_dropout=0.0,
        ),
    )
    missing, unexpected = micro.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("micro_detail_patch_expert.") for key in missing)

    base.eval()
    micro.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), micro(images), atol=1e-6)


def test_micro_detail_patch_expert_logits_are_used_and_traced() -> None:
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
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
            detail_patch_enhancement=True,
            micro_detail_patch_expert=True,
            micro_detail_top_k=4,
            micro_detail_dropout=0.0,
            micro_detail_logit_scale=0.25,
            micro_detail_routing=False,
        ),
    )
    with torch.no_grad():
        model.micro_detail_patch_expert.net[-1].bias.fill_(0.2)
    images = torch.randn(2, 3, 32, 32)
    features = model.forward_features(images, return_trace=True)
    logits = classification_logits_from_features(model, features)
    base_logits = model.head(model.head_input_from_features(features))

    assert torch.allclose(logits - base_logits, torch.full_like(logits, 0.05), atol=1e-5)
    assert tuple(features["micro_detail_logits"].shape) == (2, 5)
    assert tuple(features["trace"]["micro_detail_attention"].shape) == (2, 16)
    assert tuple(features["trace"]["micro_detail_selected_indices"].shape) == (2, 4)
    assert torch.allclose(
        features["trace"]["micro_detail_attention"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
