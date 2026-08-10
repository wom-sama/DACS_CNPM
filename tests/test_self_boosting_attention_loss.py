from __future__ import annotations

import torch
from torch import nn

from trkh.core.config import ModelConfig
from trkh.models.model import create_model
from trkh.training.train import (
    _load_model_state_allowing_extensions,
    _self_boosting_attention_loss_from_features,
)


class _CamModel(nn.Module):
    def __init__(self, *, projection_weight: float = 1.0, with_projection: bool = True) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))
        self.self_boosting_attention_projection = nn.Linear(2, 1, bias=False) if with_projection else None
        if self.self_boosting_attention_projection is not None:
            with torch.no_grad():
                self.self_boosting_attention_projection.weight.copy_(
                    torch.tensor([[float(projection_weight), 0.0]])
                )


def test_self_boosting_attention_loss_prefers_projection_matching_cam() -> None:
    patches = torch.tensor(
        [
            [[4.0, 0.0], [1.0, 0.0], [-2.0, 0.0]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0], dtype=torch.long)

    good_loss, good_stats = _self_boosting_attention_loss_from_features(
        model=_CamModel(projection_weight=1.0),
        features={"patches": patches},
        targets=targets,
        classes="0,1",
        temperature=0.40,
    )
    bad_loss, bad_stats = _self_boosting_attention_loss_from_features(
        model=_CamModel(projection_weight=-1.0),
        features={"patches": patches},
        targets=targets,
        classes="0,1",
        temperature=0.40,
    )

    assert torch.isfinite(good_loss)
    assert torch.isfinite(bad_loss)
    assert good_loss.item() < 1.0e-6
    assert bad_loss.item() > good_loss.item() + 0.10
    assert good_stats["self_boosting_attention_valid_fraction"] == 1.0
    assert good_stats["self_boosting_attention_map_similarity"] > bad_stats[
        "self_boosting_attention_map_similarity"
    ]
    good_loss.backward()
    assert patches.grad is not None


def test_self_boosting_attention_loss_respects_class_filter() -> None:
    patches = torch.randn(2, 3, 2, requires_grad=True)
    loss, stats = _self_boosting_attention_loss_from_features(
        model=_CamModel(projection_weight=1.0),
        features={"patches": patches},
        targets=torch.tensor([0, 1], dtype=torch.long),
        classes="1",
        temperature=0.40,
    )

    assert torch.isfinite(loss)
    assert stats["self_boosting_attention_valid_fraction"] == 0.5
    loss.backward()
    assert patches.grad is not None


def test_self_boosting_attention_loss_zero_without_projection() -> None:
    patches = torch.randn(2, 3, 2, requires_grad=True)
    loss, stats = _self_boosting_attention_loss_from_features(
        model=_CamModel(with_projection=False),
        features={"patches": patches},
        targets=torch.tensor([0, 1], dtype=torch.long),
        classes="0,1",
        temperature=0.40,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert stats["self_boosting_attention_valid_fraction"] == 0.0
    loss.backward()
    assert patches.grad is not None


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


def test_self_boosting_attention_head_is_allowed_resume_extension() -> None:
    base = create_model(num_classes=5, model_config=_tiny_config())
    extended = create_model(
        num_classes=5,
        model_config=_tiny_config(self_boosting_attention_head=True),
    )

    summary = _load_model_state_allowing_extensions(
        extended,
        base.state_dict(),
        allow_extensions=True,
    )

    assert summary is not None
    missing = summary["allowed_missing_keys"]
    assert any(str(key).startswith("self_boosting_attention_projection.") for key in missing)
