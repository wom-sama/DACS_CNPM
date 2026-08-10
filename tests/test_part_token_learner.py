from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models import model as model_module
from trkh.models.model import classification_logits_from_features, create_model


def test_part_token_learner_prefers_bbox_prior_patch() -> None:
    learner = model_module.AdaptivePartTokenLearner(
        dim=8,
        num_classes=5,
        part_count=3,
        hidden_dim=16,
        dropout=0.0,
        temperature=0.20,
        foreground_power=1.0,
        bbox_weight=1.0,
    )
    patch_tokens = torch.zeros(1, 8, 8)
    bbox_prior = torch.zeros(1, 8)
    bbox_prior[0, 3] = 1.0

    _, trace = learner(
        patch_tokens,
        bbox_prior=bbox_prior,
        return_trace=True,
    )

    attention = trace["attention"][0]
    assert tuple(attention.shape) == (3, 8)
    assert torch.equal(attention.argmax(dim=1), torch.full((3,), 3))
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)


def test_part_token_learner_can_extend_existing_checkpoint() -> None:
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
        ),
    )
    part_model = create_model(
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
            part_token_learner=True,
            part_token_count=3,
            part_token_dropout=0.0,
        ),
    )
    missing, unexpected = part_model.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("part_token_learner.") for key in missing)

    base.eval()
    part_model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), part_model(images), atol=1e-6)


def test_part_token_logits_are_used_routed_and_traced() -> None:
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
            part_token_learner=True,
            part_token_count=3,
            part_token_dropout=0.0,
            part_token_logit_scale=0.20,
            part_token_routing=False,
        ),
    )
    with torch.no_grad():
        model.part_token_learner.net[-1].bias.fill_(0.25)
    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=True)
    logits = classification_logits_from_features(model, features)
    base_logits = model.head(model.head_input_from_features(features))

    assert torch.allclose(logits - base_logits, torch.full_like(logits, 0.05), atol=1e-5)
    assert tuple(features["part_token_logits"].shape) == (2, 5)
    assert tuple(features["patch_bbox_prior"].shape) == (2, 16)
    assert tuple(features["trace"]["part_token_attention"].shape) == (2, 3, 16)
    assert tuple(features["trace"]["part_token_foreground_mass"].shape) == (2, 3)
    assert torch.allclose(
        features["trace"]["part_token_attention"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )


def test_part_token_routing_limits_updates_to_configured_pairs() -> None:
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="vit_registers",
            image_size=32,
            patch_size=8,
            use_cnn_stem=False,
            embed_dim=32,
            depth=1,
            num_heads=4,
            num_registers=1,
            dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=0.0,
            part_token_learner=True,
            part_token_route_pairs="0-1",
            part_token_route_max_probability_margin=0.25,
        ),
    )
    like_logits = torch.tensor(
        [
            [2.0, 1.95, 0.0, -1.0, -2.0],
            [2.0, 0.1, 1.95, -1.0, -2.0],
        ]
    )

    route_weights = model.part_token_route_weights(like_logits)

    assert route_weights[0] > 0.0
    assert route_weights[1] == 0.0


def test_part_token_pairwise_head_can_extend_existing_checkpoint() -> None:
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
        ),
    )
    pairwise_model = create_model(
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
            part_token_pairwise_head=True,
            part_token_pairwise_pairs="0-1,1-2",
            part_token_count=3,
            part_token_pairwise_dropout=0.0,
        ),
    )

    missing, unexpected = pairwise_model.load_flexible_state_dict(
        base.state_dict(),
        strict=False,
    )

    assert not unexpected
    assert any(str(key).startswith("part_token_pairwise_learner.") for key in missing)

    base.eval()
    pairwise_model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), pairwise_model(images), atol=1e-6)


def test_part_token_pairwise_logits_are_used_routed_and_traced() -> None:
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
            part_token_pairwise_head=True,
            part_token_pairwise_pairs="0-1",
            part_token_pairwise_logit_scale=0.20,
            part_token_pairwise_routing=False,
            part_token_count=3,
            part_token_pairwise_dropout=0.0,
        ),
    )
    with torch.no_grad():
        model.part_token_pairwise_learner.net[-1].bias.fill_(0.50)
    images = torch.randn(2, 3, 32, 32)
    bbox = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    features = model.forward_features(images, bbox_token_prior=bbox, return_trace=True)
    logits = classification_logits_from_features(model, features)
    base_logits = model.head(model.head_input_from_features(features))
    expected = torch.zeros_like(logits)
    expected[:, 0] = -0.10
    expected[:, 1] = 0.10

    assert torch.allclose(logits - base_logits, expected, atol=1e-5)
    assert tuple(features["part_token_pairwise_logits"].shape) == (2, 1)
    assert tuple(features["trace"]["part_token_pairwise_attention"].shape) == (2, 3, 16)
    assert tuple(features["trace"]["part_token_pairwise_route_weights"].shape) == (2, 1)
    assert torch.allclose(
        features["trace"]["part_token_pairwise_attention"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )


def test_part_token_pairwise_routing_limits_updates_per_pair() -> None:
    model = create_model(
        num_classes=5,
        model_config=ModelConfig(
            model_type="vit_registers",
            image_size=32,
            patch_size=8,
            use_cnn_stem=False,
            embed_dim=32,
            depth=1,
            num_heads=4,
            num_registers=1,
            dropout=0.0,
            attention_dropout=0.0,
            drop_path_rate=0.0,
            part_token_pairwise_head=True,
            part_token_pairwise_pairs="0-1,1-2",
            part_token_pairwise_route_max_probability_margin=0.25,
        ),
    )
    like_logits = torch.tensor(
        [
            [2.0, 1.95, 0.0, -1.0, -2.0],
            [0.0, 2.0, 1.95, -1.0, -2.0],
            [2.0, 0.1, 1.95, -1.0, -2.0],
        ]
    )

    route_weights = model.part_token_pairwise_route_weights(like_logits)

    assert tuple(route_weights.shape) == (3, 2)
    assert route_weights[0, 0] > 0.0
    assert route_weights[0, 1] == 0.0
    assert route_weights[1, 0] == 0.0
    assert route_weights[1, 1] > 0.0
    assert torch.all(route_weights[2] == 0.0)
