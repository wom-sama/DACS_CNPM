from __future__ import annotations

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.model import (
    VisionTransformerWithRegisters,
    classification_logits_from_features,
    create_model,
)


def _config(*, fusion: bool, **overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "model_type": "vit_registers",
        "image_size": 64,
        "patch_size": 16,
        "stem_channels": 8,
        "embed_dim": 32,
        "depth": 5,
        "num_heads": 4,
        "num_registers": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
        "multi_branch_fusion": True,
        "branch_color_tokens": 1,
        "branch_edge_tokens": 1,
        "branch_cnn_tokens": 0,
        "branch_token_dropout": 0.0,
        "fine_grained_pooling": True,
        "fine_grained_pooling_dropout": 0.0,
        "head_pooling": "cls_branch_register_mean",
        "token_pruning": True,
        "token_prune_layers": "2,4",
        "token_keep_rates": "0.75,0.50",
        "token_prune_foreground_weight": 0.35,
        "inattentive_token_fusion": fusion,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _model(*, fusion: bool) -> VisionTransformerWithRegisters:
    return create_model(num_classes=5, model_config=_config(fusion=fusion))


def test_official_first_stage_weighted_sum_and_gradients() -> None:
    patch_tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    patch_tokens.requires_grad_(True)
    attention = torch.tensor(
        [[0.10, 0.20, 0.30, 0.15, 0.25], [0.05, 0.35, 0.10, 0.30, 0.20]],
        dtype=torch.float32,
    )
    selected = torch.tensor([[0, 2, 4], [1, 3, 4]], dtype=torch.long)

    context, dropped, weights, mass, previous_mass = (
        VisionTransformerWithRegisters._fuse_inattentive_context(
            patch_tokens=patch_tokens,
            cls_attention=attention,
            selected_local=selected,
        )
    )
    selected_mask = torch.zeros_like(attention).scatter(
        1, selected, torch.ones_like(selected, dtype=attention.dtype)
    )
    expected = (
        patch_tokens * (attention * (1.0 - selected_mask)).unsqueeze(-1)
    ).sum(dim=1, keepdim=True)

    assert torch.allclose(context, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(weights.sum(dim=1, keepdim=True), mass)
    assert torch.count_nonzero(previous_mass) == 0
    for row in range(2):
        assert set(dropped[row].tolist()).isdisjoint(set(selected[row].tolist()))
        assert set(dropped[row].tolist()) | set(selected[row].tolist()) == set(range(5))

    context.sum().backward()
    expected_grad = (attention * (1.0 - selected_mask)).unsqueeze(-1).expand_as(
        patch_tokens
    )
    assert torch.allclose(patch_tokens.grad, expected_grad, atol=0.0, rtol=0.0)


def test_later_stage_refolds_previous_context_with_current_attention() -> None:
    patch_tokens = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], requires_grad=True
    )
    attention = torch.tensor([[0.2, 0.3, 0.1]])
    selected = torch.tensor([[1, 2]])
    previous_context = torch.tensor([[[7.0, 11.0]]], requires_grad=True)
    previous_attention = torch.tensor([[0.4]])

    context, _, _, mass, previous_mass = (
        VisionTransformerWithRegisters._fuse_inattentive_context(
            patch_tokens=patch_tokens,
            cls_attention=attention,
            selected_local=selected,
            previous_context=previous_context,
            previous_context_attention=previous_attention,
        )
    )
    expected = patch_tokens[:, 0:1] * 0.2 + previous_context * 0.4
    assert torch.allclose(context, expected)
    assert torch.allclose(mass, torch.tensor([[0.6]]))
    assert torch.equal(previous_mass, previous_attention)

    context.sum().backward()
    assert torch.allclose(previous_context.grad, torch.full_like(previous_context, 0.4))
    assert torch.allclose(patch_tokens.grad[:, 0], torch.full_like(patch_tokens[:, 0], 0.2))
    assert torch.count_nonzero(patch_tokens.grad[:, 1:]) == 0


def test_fusion_is_parameter_free_and_preserves_public_spatial_layout() -> None:
    torch.manual_seed(31)
    control = _model(fusion=False).eval()
    candidate = _model(fusion=True).eval()
    candidate.load_state_dict(control.state_dict(), strict=True)
    assert list(control.state_dict()) == list(candidate.state_dict())
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )

    torch.manual_seed(9)
    images = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        control_features = control.forward_features(images, return_trace=True)
        candidate_features = candidate.forward_features(images, return_trace=True)

    base_prefixes = int(candidate.num_prefix_tokens)
    assert candidate_features["inattentive_context"].shape == (3, 1, 32)
    assert candidate_features["patches"].shape == (3, 8, 32)
    assert candidate_features["patch_indices"].shape == (3, 8)
    assert candidate_features["tokens"].shape == (3, base_prefixes + 8, 32)
    assert candidate_features["trace"]["active_prefix_count"].item() == base_prefixes + 1
    assert "inattentive_context" not in control_features

    control_pruning = control_features["trace"]["pruning"]
    candidate_pruning = candidate_features["trace"]["pruning"]
    assert torch.equal(control_pruning[0]["kept_indices"], candidate_pruning[0]["kept_indices"])
    assert [int(item["after_count"]) for item in candidate_pruning] == [12, 8]

    previous = torch.arange(16).view(1, -1).expand(images.size(0), -1)
    for stage in candidate_pruning:
        kept = stage["kept_indices"]
        dropped = stage["dropped_indices"]
        for row in range(images.size(0)):
            kept_set = set(kept[row].tolist())
            dropped_set = set(dropped[row].tolist())
            assert kept_set.isdisjoint(dropped_set)
            assert kept_set | dropped_set == set(previous[row].tolist())
        assert torch.isfinite(stage["context_token"]).all()
        assert torch.isfinite(stage["context_attention_mass"]).all()
        assert (stage["fusion_weights"] >= 0).all()
        previous = kept

    for indices, norms in zip(
        candidate_features["trace"]["block_patch_indices"],
        candidate_features["trace"]["block_patch_norms"],
    ):
        assert indices.shape == norms.shape


def test_standard_and_trace_logits_match_for_fusion_candidate() -> None:
    torch.manual_seed(17)
    model = _model(fusion=True).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        standard = model.forward_features(images)
        traced = model.forward_features(images, return_trace=True)
        standard_logits = classification_logits_from_features(model, standard)
        traced_logits = classification_logits_from_features(model, traced)

    assert torch.allclose(standard_logits, traced_logits, atol=1e-6, rtol=0.0)
    assert torch.equal(standard["patch_indices"], traced["patch_indices"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_pruning": False},
        {"early_token_mask_keep_rate": 0.75},
        {"deep_class_prompt": True},
        {
            "concurrent_local_global_coupling": True,
            "concurrent_local_global_layers": "1,2,3,4,5",
        },
        {"late_member_branch": True, "late_member_fork_after_block": 3},
    ],
)
def test_incompatible_dynamic_prefix_configurations_fail_fast(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="inattentive_token_fusion"):
        create_model(
            num_classes=5,
            model_config=_config(fusion=True, **overrides),
        )
