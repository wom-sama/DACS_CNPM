from __future__ import annotations

import torch
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import (
    LocalZoomImageExpert,
    classification_logits_from_features,
    create_model,
)
from trkh.training.train import _local_zoom_auxiliary_loss_from_features


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


def test_local_zoom_expert_emits_logits_and_trace() -> None:
    model = create_model(
        num_classes=5,
        model_config=_tiny_config(
            local_zoom_image_expert=True,
            local_zoom_crop_size=32,
            local_zoom_crop_scale=0.50,
            local_zoom_hidden_dim=32,
            local_zoom_dropout=0.0,
            local_zoom_logit_scale=0.20,
            local_zoom_routing=False,
        ),
    )
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)

    assert tuple(logits.shape) == (2, 5)
    assert tuple(features["local_zoom_logits"].shape) == (2, 5)
    assert tuple(features["trace"]["local_zoom_score_map"].shape[:2]) == (2, 1)
    assert tuple(features["trace"]["local_zoom_crop_boxes"].shape) == (2, 4)
    assert tuple(features["trace"]["local_zoom_descriptor"].shape) == (2, 38)
    assert tuple(features["trace"]["local_zoom_route_weights"].shape) == (2,)


def test_local_zoom_expert_can_extend_checkpoint_without_changing_logits() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    zoom = create_model(
        num_classes=5,
        model_config=_tiny_config(
            local_zoom_image_expert=True,
            local_zoom_crop_size=32,
            local_zoom_hidden_dim=32,
            local_zoom_dropout=0.0,
        ),
    )
    missing, unexpected = zoom.load_flexible_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert any(str(key).startswith("local_zoom_image_expert.") for key in missing)

    base.eval()
    zoom.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(base(images), zoom(images), atol=1e-6)


def test_local_zoom_auxiliary_loss_is_finite() -> None:
    features = {"local_zoom_logits": torch.randn(4, 5)}
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    loss = _local_zoom_auxiliary_loss_from_features(
        features=features,
        targets=targets,
        criterion=nn.CrossEntropyLoss(),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_local_zoom_defect_spot_score_crops_dark_brown_region() -> None:
    rgb = torch.full((1, 3, 64, 64), 0.62, dtype=torch.float32)
    rgb[:, 0] = 0.66
    rgb[:, 1] = 0.72
    rgb[:, 2] = 0.25
    rgb[:, :, 34:46, 44:56] = torch.tensor([0.18, 0.09, 0.03]).view(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    image = (rgb - mean) / std
    expert = LocalZoomImageExpert(
        num_classes=5,
        hidden_dim=32,
        crop_size=32,
        crop_scale=0.40,
        score_mode="defect_spot",
        dropout=0.0,
        score_size=32,
    )
    expert.eval()

    with torch.no_grad():
        _logits, trace = expert(image, return_trace=True)

    score = trace["score_map"][0, 0]
    max_index = int(score.flatten().argmax().item())
    max_y = max_index // int(score.shape[1])
    max_x = max_index % int(score.shape[1])
    box = trace["crop_boxes"][0]

    assert max_x >= 20
    assert max_y >= 16
    assert float(box[0]) > 0.45
    assert float(box[1]) > 0.35


def test_local_zoom_defect_spot_interior_suppresses_border_band() -> None:
    rgb = torch.full((1, 3, 64, 64), 0.62, dtype=torch.float32)
    rgb[:, 0] = 0.66
    rgb[:, 1] = 0.72
    rgb[:, 2] = 0.25
    rgb[:, :, 54:64, :] = torch.tensor([0.15, 0.07, 0.025]).view(1, 3, 1, 1)
    rgb[:, :, 26:38, 38:50] = torch.tensor([0.18, 0.09, 0.03]).view(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    image = (rgb - mean) / std
    expert = LocalZoomImageExpert(
        num_classes=5,
        hidden_dim=32,
        crop_size=32,
        crop_scale=0.40,
        score_mode="defect_spot_interior",
        dropout=0.0,
        score_size=32,
    )
    expert.eval()

    with torch.no_grad():
        _logits, trace = expert(image, return_trace=True)

    score = trace["score_map"][0, 0]
    max_index = int(score.flatten().argmax().item())
    max_y = max_index // int(score.shape[1])
    max_x = max_index % int(score.shape[1])
    box = trace["crop_boxes"][0]

    assert 12 <= max_y <= 22
    assert 18 <= max_x <= 28
    assert float(box[1]) < 0.55
    assert float(box[3]) < 0.85
