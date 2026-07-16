from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from trkh.core.config import ModelConfig
from trkh.models.bi_level_routing_attention import BiLevelRoutingAttention
from trkh.models.model import MultiHeadSelfAttention, create_model


OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\biformer-cvpr2023")


def _tiny_model_config(**overrides) -> ModelConfig:
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
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="2",
        token_keep_rates="0.5",
        bi_level_routing_attention=True,
        bi_level_routing_attention_layers="2",
        bi_level_routing_attention_regions_per_axis=4,
        bi_level_routing_attention_topk=4,
        bi_level_routing_attention_local_context_kernel_size=5,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    return inputs, indices


def _official_class():
    if not (OFFICIAL_ROOT / "ops" / "bra_nchw.py").is_file():
        pytest.skip(f"Locked official BiFormer source is absent: {OFFICIAL_ROOT}")
    root = str(OFFICIAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from ops.bra_nchw import nchwBRA

    return nchwBRA


def _copy_candidate_to_official(
    candidate: BiLevelRoutingAttention,
    official: torch.nn.Module,
) -> None:
    with torch.no_grad():
        official.qkv_linear.weight.copy_(candidate.qkv.weight[:, :, None, None])
        official.qkv_linear.bias.copy_(candidate.qkv.bias)
        official.output_linear.weight.copy_(candidate.proj.weight[:, :, None, None])
        official.output_linear.bias.copy_(candidate.proj.bias)
        official.lepe.weight.copy_(candidate.local_context.weight)
        official.lepe.bias.copy_(candidate.local_context.bias)


def test_bra_patch_equation_attention_and_gradients_match_official() -> None:
    official_class = _official_class()
    torch.manual_seed(41)
    candidate = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=2,
        topk=2,
        local_context_kernel_size=3,
    ).eval()
    official = official_class(
        dim=32,
        num_heads=4,
        n_win=2,
        qk_scale=candidate.scale,
        topk=2,
        side_dwconv=3,
    ).eval()
    _copy_candidate_to_official(candidate, official)

    torch.manual_seed(19)
    official_input = torch.randn(2, 32, 4, 4, requires_grad=True)
    candidate_input = (
        official_input.detach()
        .flatten(2)
        .transpose(1, 2)
        .clone()
        .requires_grad_(True)
    )
    official_output, official_attention = official(
        official_input,
        ret_attn_mask=True,
    )
    candidate_output = candidate(
        candidate_input,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=0,
        patch_indices=torch.arange(16).unsqueeze(0).expand(2, -1),
    )
    candidate_output, _ = candidate_output
    candidate_map = candidate_output.transpose(1, 2).reshape_as(official_output)
    trace = candidate.trace()
    torch.testing.assert_close(candidate_map, official_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        trace["native_sparse_attention"],
        official_attention,
        atol=1e-6,
        rtol=1e-6,
    )

    candidate_map.square().mean().backward()
    official_output.square().mean().backward()
    candidate_input_gradient = candidate_input.grad.transpose(1, 2).reshape_as(
        official_input
    )
    torch.testing.assert_close(
        candidate_input_gradient,
        official_input.grad,
        atol=1e-6,
        rtol=1e-6,
    )
    candidate_parameters = dict(candidate.named_parameters())
    official_parameters = dict(official.named_parameters())
    mappings = {
        "qkv.weight": ("qkv_linear.weight", lambda value: value[:, :, None, None]),
        "qkv.bias": ("qkv_linear.bias", lambda value: value),
        "proj.weight": ("output_linear.weight", lambda value: value[:, :, None, None]),
        "proj.bias": ("output_linear.bias", lambda value: value),
        "local_context.weight": ("lepe.weight", lambda value: value),
        "local_context.bias": ("lepe.bias", lambda value: value),
    }
    for candidate_name, (official_name, transform) in mappings.items():
        candidate_gradient = candidate_parameters[candidate_name].grad
        official_gradient = official_parameters[official_name].grad
        assert candidate_gradient is not None, candidate_name
        assert official_gradient is not None, official_name
        torch.testing.assert_close(
            transform(candidate_gradient),
            official_gradient,
            atol=1e-6,
            rtol=1e-6,
        )


def test_bra_all_region_control_matches_dense_mhsa_when_lce_is_zero() -> None:
    inputs, indices = _inputs()
    torch.manual_seed(23)
    standard = MultiHeadSelfAttention(dim=32, num_heads=4).eval()
    torch.manual_seed(99)
    control = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=16,
        local_context_kernel_size=5,
    ).eval()
    control.copy_shared_projections_from(standard)
    with torch.no_grad():
        control.local_context.weight.zero_()
        control.local_context.bias.zero_()
    expected, expected_attention = standard(inputs, return_attention=True)
    observed, observed_attention = control(
        inputs,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        observed_attention,
        expected_attention,
        atol=1e-6,
        rtol=1e-6,
    )


def test_bra_sparse_attention_scatter_is_exact_and_normalized() -> None:
    inputs, indices = _inputs()
    module = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=4,
        local_context_kernel_size=5,
    ).eval()
    output, attention = module(
        inputs,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    trace = module.trace()
    assert output.shape == inputs.shape
    assert attention.shape == (2, 4, 21, 21)
    assert torch.isfinite(attention).all()
    assert bool((attention >= 0).all())
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )
    selected = trace["selected_patch_indices"]
    patch_attention = attention[:, :, 5:, 5:]
    support = torch.zeros(2, 16, 16, dtype=torch.bool)
    support.scatter_(2, selected, True)
    support = support[:, None].expand(-1, 4, -1, -1)
    assert int(torch.count_nonzero(patch_attention.masked_select(~support)).item()) == 0
    assert bool((attention[:, :, 5:, :5] > 0).all())
    torch.testing.assert_close(attention, trace["dense_attention"])


def test_bra_component_families_receive_finite_nonzero_gradients() -> None:
    inputs, indices = _inputs()
    module = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=4,
        local_context_kernel_size=5,
    )
    output = module(
        inputs,
        collect_trace=True,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    output, _ = output
    output.square().mean().backward()
    for name in (
        "qkv.weight",
        "qkv.bias",
        "proj.weight",
        "proj.bias",
        "local_context.weight",
        "local_context.bias",
    ):
        gradient = dict(module.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient).item()) > 0, name
    assert not module.trace()["route_indices"].requires_grad


def test_bra_matched_roles_have_identical_state_inventory_and_rng() -> None:
    torch.manual_seed(42)
    control = create_model(
        num_classes=5,
        model_config=_tiny_model_config(
            bi_level_routing_attention_topk=16,
        ),
    )
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=_tiny_model_config())
    candidate_rng = torch.random.get_rng_state().clone()
    assert torch.equal(control_rng, candidate_rng)
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for key, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    assert isinstance(control.blocks[1].attn, BiLevelRoutingAttention)
    assert isinstance(candidate.blocks[1].attn, BiLevelRoutingAttention)
    assert control.blocks[1].attn.topk == 16
    assert candidate.blocks[1].attn.topk == 4
    assert sum(value.numel() for value in control.parameters()) == sum(
        value.numel() for value in candidate.parameters()
    )


def test_bra_model_trace_and_pruning_after_block_two() -> None:
    torch.manual_seed(13)
    model = create_model(num_classes=5, model_config=_tiny_model_config()).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        features = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
    assert features["attention_representation"] == "mixed"
    assert features["attention_representations"] == {
        0: "mhsa_probability",
        1: "bi_level_routing_sparse_probability",
        2: "mhsa_probability",
    }
    trace = features["trace"]
    assert trace["bi_level_routing_attention_layers"].tolist() == [2]
    assert trace["bi_level_routing_topk"].tolist() == [4.0]
    assert trace["bi_level_routing_region_count"].tolist() == [16.0]
    assert trace["bi_level_routing_distinct_route_sets_mean"].shape == (1,)
    with torch.inference_mode():
        pruned = model.forward_features(images, return_trace=True)
    assert pruned["patches"].shape[1] == 8


def test_bra_rejects_sparse_grid_and_locked_route_conflicts() -> None:
    inputs, indices = _inputs()
    module = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=4,
        local_context_kernel_size=5,
    )
    with pytest.raises(ValueError, match="complete dense patch grid"):
        module(inputs[:, :-1], grid_size=(4, 4), prefix_count=5)
    with pytest.raises(ValueError, match="identity-ordered"):
        module(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices.flip(1),
        )
    with pytest.raises(ValueError, match="locked to transformer layer 2"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                bi_level_routing_attention_layers="1",
            ),
        )
    with pytest.raises(ValueError, match="topk 4 or 16"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                bi_level_routing_attention_topk=8,
            ),
        )
    with pytest.raises(ValueError, match="Deformable Spatial"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                deformable_spatial_attention=True,
                deformable_spatial_attention_layers="2",
            ),
        )


def test_v8_launcher_exposes_locked_bra_controls() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$BiLevelRoutingAttention = $false" in script
    assert '"--bi-level-routing-attention"' in script
    assert '"--bi-level-routing-attention-layers"' in script
    assert '"--bi-level-routing-attention-regions-per-axis"' in script
    assert '"--bi-level-routing-attention-topk"' in script
    assert '"--bi-level-routing-attention-local-context-kernel-size"' in script
