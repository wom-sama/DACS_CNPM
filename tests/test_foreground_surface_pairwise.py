from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import classification_logits_from_features, create_model


def _small_pairwise_model():
    return create_model(
        5,
        ModelConfig(
            model_type="vit_registers",
            image_size=64,
            patch_size=16,
            stem_channels=8,
            embed_dim=32,
            depth=1,
            num_heads=4,
            num_registers=2,
            dropout=0.0,
            drop_path_rate=0.0,
            foreground_surface_pairwise_head=True,
            foreground_surface_pairwise_pairs="0-1,1-2,1-4",
            foreground_surface_pairwise_logit_scale=0.18,
            foreground_surface_pairwise_dropout=0.0,
        ),
    )


def test_foreground_surface_pairwise_forward_trace_shapes() -> None:
    model = _small_pairwise_model()
    model.eval()
    images = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        features = model.forward_features(images, return_trace=True)
        logits = classification_logits_from_features(model, features)

    assert tuple(logits.shape) == (2, 5)
    assert tuple(features["foreground_surface_pairwise_logits"].shape) == (2, 3)
    assert "foreground_surface_pairwise_stats" in features["trace"]
    assert tuple(features["trace"]["foreground_surface_pairwise_route_weights"].shape) == (2, 3)


def test_foreground_surface_pairwise_routing_matches_boundary_top2() -> None:
    model = _small_pairwise_model()
    like_logits = torch.tensor(
        [
            [2.0, 1.95, -1.0, -2.0, -3.0],
            [-2.0, 1.80, 1.75, -1.0, -3.0],
            [-2.0, 1.80, -1.0, -3.0, 1.77],
            [3.0, -1.0, -2.0, 2.8, -3.0],
        ],
        dtype=torch.float32,
    )

    weights = model.foreground_surface_pairwise_route_weights(like_logits)

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0
    assert weights[1, 1] > 0.0
    assert weights[2, 2] > 0.0
    assert torch.all(weights[3] == 0.0)
