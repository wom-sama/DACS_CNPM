from __future__ import annotations

import torch

from trkh.core.config import ModelConfig
from trkh.models.model import classification_logits_from_features, create_model
from trkh.training.train import _interior_boundary_pairwise_loss_from_features


def _small_interior_boundary_model():
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
            interior_boundary_pairwise_head=True,
            interior_boundary_pairwise_pairs="0-1,4-1",
            interior_boundary_pairwise_logit_scale=0.14,
            interior_boundary_pairwise_dropout=0.0,
            interior_boundary_pairwise_hidden_dim=32,
            interior_boundary_pairwise_erode_kernel=5,
        ),
    )


def test_interior_boundary_pairwise_forward_trace_shapes() -> None:
    model = _small_interior_boundary_model()
    model.eval()
    images = torch.rand(2, 3, 64, 64)
    bboxes = torch.tensor(
        [
            [0.50, 0.50, 0.50, 0.50],
            [0.45, 0.55, 0.40, 0.45],
        ],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        features = model.forward_features(
            images,
            bbox_token_prior=bboxes,
            return_trace=True,
        )
        logits = classification_logits_from_features(model, features)

    assert tuple(logits.shape) == (2, 5)
    assert tuple(features["interior_boundary_pairwise_logits"].shape) == (2, 2)
    trace = features["trace"]
    assert tuple(trace["interior_boundary_pairwise_stats"].shape) == (2, 234)
    assert tuple(trace["interior_boundary_pairwise_route_weights"].shape) == (2, 2)
    assert tuple(trace["interior_boundary_foreground_mask"].shape) == (2, 64, 64)
    assert tuple(trace["interior_boundary_bbox_mask"].shape) == (2, 64, 64)
    assert tuple(trace["interior_boundary_interior_weight_map"].shape) == (2, 1, 64, 64)
    assert tuple(trace["interior_boundary_boundary_weight_map"].shape) == (2, 1, 64, 64)


def test_interior_boundary_pairwise_routing_matches_boundary_top2() -> None:
    model = _small_interior_boundary_model()
    like_logits = torch.tensor(
        [
            [2.0, 1.95, -1.0, -2.0, -3.0],
            [-2.0, 1.80, -1.0, -3.0, 1.77],
            [3.0, -1.0, -2.0, 2.8, -3.0],
        ],
        dtype=torch.float32,
    )

    weights = model.interior_boundary_pairwise_route_weights(like_logits)

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0
    assert weights[1, 1] > 0.0
    assert torch.all(weights[2] == 0.0)


def test_interior_boundary_pairwise_can_extend_checkpoint_without_changing_logits() -> None:
    base = create_model(
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
        ),
    )
    extended = _small_interior_boundary_model()
    missing, unexpected = extended.load_flexible_state_dict(base.state_dict(), strict=False)

    assert not unexpected
    assert any(str(key).startswith("interior_boundary_pairwise_head.") for key in missing)

    base.eval()
    extended.eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(base(images), extended(images), atol=1e-6)


def test_interior_boundary_pairwise_loss_is_finite() -> None:
    model = _small_interior_boundary_model()
    features = {"interior_boundary_pairwise_logits": torch.randn(4, 2)}
    targets = torch.tensor([0, 1, 4, 2], dtype=torch.long)

    loss = _interior_boundary_pairwise_loss_from_features(
        model=model,
        features=features,
        targets=targets,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
