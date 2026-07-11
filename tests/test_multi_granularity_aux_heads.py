from __future__ import annotations

import torch
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import create_model
from trkh.training.losses import SupervisedContrastiveLoss
from trkh.training.train import (
    _multi_granularity_auxiliary_loss_from_features,
    _multi_granularity_contrastive_loss_from_features,
    _multi_granularity_refinement_loss_from_features,
)


def _tiny_config(**overrides) -> ModelConfig:
    values = dict(
        model_type="vit_registers",
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        embed_dim=32,
        depth=3,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_multi_granularity_aux_heads_emit_layer_logits() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            multi_granularity_aux_heads=True,
            multi_granularity_aux_layers="1,3",
            multi_granularity_aux_dropout=0.0,
        ),
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = model.forward_features(images, return_trace=True)

    auxiliary_logits = features["multi_granularity_logits"]
    auxiliary_features = features["multi_granularity_features"]
    assert set(auxiliary_logits) == {"1", "3"}
    assert set(auxiliary_features) == {"1", "3"}
    assert tuple(auxiliary_logits["1"].shape) == (2, 5)
    assert tuple(auxiliary_logits["3"].shape) == (2, 5)
    assert tuple(auxiliary_features["1"].shape) == (2, 32)
    assert tuple(auxiliary_features["3"].shape) == (2, 32)
    assert features["trace"]["multi_granularity_layers"].tolist() == [1, 3]
    assert features["trace"]["multi_granularity_feature_shapes"] == [(2, 32), (2, 32)]


def test_multi_granularity_aux_heads_can_extend_checkpoint_without_changing_logits() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    auxiliary = create_model(
        num_classes=5,
        model_config=_tiny_config(
            multi_granularity_aux_heads=True,
            multi_granularity_aux_layers="2",
            multi_granularity_aux_dropout=0.0,
        ),
    )
    missing, unexpected = auxiliary.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("multi_granularity_aux_heads.") for key in missing)

    base.eval()
    auxiliary.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), auxiliary(images), atol=1e-6)


def test_multi_granularity_auxiliary_loss_is_finite() -> None:
    features = {
        "multi_granularity_logits": {
            "1": torch.randn(4, 5),
            "3": torch.randn(4, 5),
        }
    }
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    loss = _multi_granularity_auxiliary_loss_from_features(
        features=features,
        targets=targets,
        criterion=nn.CrossEntropyLoss(),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_multi_granularity_refinement_loss_distills_final_logits() -> None:
    teacher_logits = torch.randn(4, 5)
    features = {
        "multi_granularity_logits": {
            "1": teacher_logits + 0.1 * torch.randn(4, 5),
            "3": torch.randn(4, 5),
        }
    }

    loss, stats = _multi_granularity_refinement_loss_from_features(
        features=features,
        teacher_logits=teacher_logits,
        temperature=16.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert stats["count"] == 2
    assert stats["used_layers"] == ["1", "3"]
    assert stats["teacher_entropy"] > 0.0


def test_multi_granularity_refinement_loss_zero_without_aux_logits() -> None:
    teacher_logits = torch.randn(3, 5)

    loss, stats = _multi_granularity_refinement_loss_from_features(
        features={},
        teacher_logits=teacher_logits,
        temperature=16.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert stats["count"] == 0


def test_multi_granularity_contrastive_loss_uses_teacher_reliability() -> None:
    targets = torch.tensor([0, 1, 1, 2, 0, 4], dtype=torch.long)
    teacher_probabilities = torch.zeros(6, 5)
    teacher_probabilities[torch.arange(6), targets] = torch.tensor(
        [0.92, 0.86, 0.88, 0.91, 0.84, 0.89]
    )
    teacher_probabilities = teacher_probabilities + 0.01
    teacher_probabilities = teacher_probabilities / teacher_probabilities.sum(dim=1, keepdim=True)
    features = {
        "multi_granularity_features": {
            "1": torch.randn(6, 12),
            "3": torch.randn(6, 12),
        }
    }

    loss, stats = _multi_granularity_contrastive_loss_from_features(
        features=features,
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=SupervisedContrastiveLoss(temperature=0.20, class_balanced=True),
        pairs="0-1,1-2,1-4",
        teacher_min_confidence=0.0,
        require_agreement=True,
        weight_mode="confidence_margin",
        teacher_confidence_power=0.5,
        num_classes=5,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert stats["count"] == 6
    assert stats["positive_pairs"] > 0
    assert stats["used_layers"] == ["1", "3"]
    assert stats["reliability_mean"] > 0.0


def test_multi_granularity_contrastive_loss_zero_when_teacher_rejects() -> None:
    targets = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    teacher_probabilities = torch.zeros(4, 5)
    teacher_probabilities[:, 3] = 1.0
    features = {
        "multi_granularity_features": {
            "2": torch.randn(4, 8),
        }
    }

    loss, stats = _multi_granularity_contrastive_loss_from_features(
        features=features,
        targets=targets,
        teacher_probabilities=teacher_probabilities,
        criterion=SupervisedContrastiveLoss(temperature=0.20),
        pairs="0-1,1-2",
        teacher_min_confidence=0.0,
        require_agreement=True,
        weight_mode="filter",
        num_classes=5,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert stats["count"] == 0


def test_layer_token_fusion_emits_intermediate_token_trace() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            layer_token_fusion=True,
            layer_token_fusion_layers="1,3",
            layer_token_fusion_top_k=2,
            layer_token_fusion_blend=0.20,
            layer_token_fusion_bbox_weight=0.0,
            layer_token_fusion_foreground_weight=0.0,
        ),
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = model.forward_features(images, return_trace=True)

    assert tuple(features["layer_token_fusion_feature"].shape) == (2, 32)
    trace = features["trace"]
    assert trace["layer_token_fusion_layers"].tolist() == [1, 3]
    assert len(trace["layer_token_fusion_indices"]) == 2
    assert tuple(trace["layer_token_fusion_indices"][0].shape) == (2, 2)
    assert torch.isfinite(trace["layer_token_fusion_feature"]).all()


def test_layer_token_fusion_zero_blend_preserves_checkpoint_logits() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    fusion = create_model(
        num_classes=5,
        model_config=_tiny_config(
            layer_token_fusion=True,
            layer_token_fusion_layers="1,3",
            layer_token_fusion_top_k=2,
            layer_token_fusion_blend=0.0,
        ),
    )
    missing, unexpected = fusion.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not missing
    assert not unexpected

    base.eval()
    fusion.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), fusion(images), atol=1e-6)
