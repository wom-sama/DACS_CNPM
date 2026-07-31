from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from trkh.core.utils import build_optimizer_param_groups
from trkh.models.model import (
    VisionTransformerWithRegisters,
    build_model_from_checkpoint,
    create_model,
)
from trkh.training.train import _load_model_state_allowing_extensions


class _TinyTokenBackbone(nn.Module):
    num_features = 8
    embed_dim = 8
    num_prefix_tokens = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 8, bias=False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1))
        token = self.projection(pooled).unsqueeze(1)
        offsets = torch.arange(6, device=images.device, dtype=token.dtype).view(1, 6, 1)
        return token + offsets / 10.0

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = bool(enable)


def _keeper_config() -> dict[str, object]:
    return {
        "image_size": 32,
        "patch_size": 16,
        "use_cnn_stem": False,
        "embed_dim": 16,
        "depth": 1,
        "num_heads": 4,
        "num_registers": 1,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "head_pooling": "cls",
    }


def _hybrid_config() -> dict[str, object]:
    return {
        **_keeper_config(),
        "model_type": "vit_registers_pretrained_hybrid",
        "research_track": "pretrained",
        "pretrained": True,
        "timm_model_name": "tiny-token-vit",
        "pretrained_checkpoint_path": "logical-model.safetensors",
        "pretrained_checkpoint_sha256": "a" * 64,
        "pretrained_source_url": "https://example.test/model",
        "pretrained_source_revision": "revision-1",
        "pretrained_source_license": "test-license",
        "pretrained_semantic_expected_prefix_tokens": 2,
        "pretrained_semantic_expected_embed_dim": 8,
        "pretrained_semantic_expected_patch_count": 4,
        "pretrained_semantic_dropout": 0.0,
        "pretrained_semantic_initial_scale": 0.0,
        "pretrained_semantic_max_scale": 0.25,
    }


def _fake_verified_loader(**_kwargs):
    backbone = _TinyTokenBackbone()
    provenance = {
        "model_name": "tiny-token-vit",
        "checkpoint": {"sha256": "a" * 64},
    }
    backbone.pretrained_source_provenance = deepcopy(provenance)
    return backbone, provenance


def test_hybrid_zero_gate_preserves_keeper_logits_and_uses_split_lr() -> None:
    torch.manual_seed(19)
    keeper = VisionTransformerWithRegisters(num_classes=5, **_keeper_config()).eval()
    with patch(
        "trkh.models.model.load_verified_local_timm_model",
        side_effect=_fake_verified_loader,
    ):
        hybrid = create_model(num_classes=5, model_config=_hybrid_config()).eval()

    incompatible = hybrid.load_state_dict(keeper.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        name.startswith("pretrained_semantic_branch.")
        for name in incompatible.missing_keys
    )

    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        keeper_logits = keeper(images)
        hybrid_logits = hybrid(images)
    assert torch.equal(hybrid_logits, keeper_logits)
    assert float(
        hybrid.pretrained_semantic_branch.effective_gate().detach().item()
    ) == 0.0

    groups = build_optimizer_param_groups(
        hybrid,
        weight_decay=0.05,
        learning_rate=2.0e-4,
        backbone_lr_scale=0.1,
    )
    backbone_groups = [group for group in groups if group["name"].startswith("backbone_")]
    head_groups = [group for group in groups if group["name"].startswith("head_")]
    assert backbone_groups and head_groups
    assert all(group["lr"] == 2.0e-5 for group in backbone_groups)
    assert all("lr" not in group for group in head_groups)


def test_hybrid_zero_gate_preserves_training_rng_and_logits() -> None:
    torch.manual_seed(23)
    keeper = VisionTransformerWithRegisters(num_classes=5, **_keeper_config())
    config = _hybrid_config()
    config["pretrained_semantic_dropout"] = 0.5
    with patch(
        "trkh.models.model.load_verified_local_timm_model",
        side_effect=_fake_verified_loader,
    ):
        hybrid = create_model(num_classes=5, model_config=config)
    hybrid.load_state_dict(keeper.state_dict(), strict=False)

    keeper.head = nn.Sequential(nn.Dropout(0.5), keeper.head)
    hybrid.head = deepcopy(keeper.head)
    keeper.train()
    hybrid.train()
    images = torch.randn(2, 3, 32, 32)

    torch.manual_seed(97)
    keeper_logits = keeper(images)
    torch.manual_seed(97)
    hybrid_logits = hybrid(images)

    assert torch.equal(hybrid_logits, keeper_logits)


def test_hybrid_resume_rejects_partially_missing_semantic_branch() -> None:
    with patch(
        "trkh.models.model.load_verified_local_timm_model",
        side_effect=_fake_verified_loader,
    ):
        hybrid = create_model(num_classes=5, model_config=_hybrid_config())
    corrupt_state = dict(hybrid.state_dict())
    removed_key = next(
        key
        for key in corrupt_state
        if key.startswith("pretrained_semantic_branch.projector.")
    )
    corrupt_state.pop(removed_key)

    with pytest.raises(RuntimeError, match="partial load"):
        _load_model_state_allowing_extensions(
            hybrid,
            corrupt_state,
            allow_extensions=True,
        )


def test_checkpoint_rebuild_does_not_replay_external_initialization() -> None:
    config = _hybrid_config()
    with patch(
        "trkh.models.model.load_verified_local_timm_model",
        side_effect=_fake_verified_loader,
    ):
        original = create_model(num_classes=5, model_config=config).eval()

    checkpoint = {
        "class_names": [f"class_{index}" for index in range(5)],
        "model_config": deepcopy(config),
        "model_state": original.state_dict(),
        "pretrained_provenance": deepcopy(original.pretrained_provenance),
    }
    with patch(
        "trkh.models.model.load_verified_local_timm_model",
        side_effect=AssertionError("external initialization was replayed"),
    ), patch("timm.create_model", return_value=_TinyTokenBackbone()):
        restored = build_model_from_checkpoint(checkpoint).eval()

    assert restored.model_type == "vit_registers_pretrained_hybrid"
    assert restored.research_track == "pretrained"
    assert restored.pretrained_provenance["external_initialization_replayed"] is False
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(restored(images), original(images))
