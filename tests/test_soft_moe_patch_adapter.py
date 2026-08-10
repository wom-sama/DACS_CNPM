from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.model import MultiHeadSelfAttention, create_model
from trkh.models.soft_moe_patch_adapter import SoftMoEPatchAdapter
from trkh.training.train import _load_model_state_allowing_extensions


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
        head_pooling="cls_register_mean",
        token_pruning=True,
        token_prune_layers="1,2",
        token_keep_rates="0.75,0.50",
        soft_moe_patch_adapter=True,
        soft_moe_patch_adapter_layers="1,3",
        soft_moe_hidden_dim=8,
        soft_moe_num_experts=4,
        soft_moe_residual_scale=0.10,
        soft_moe_router_scale_init=10.0,
        soft_moe_init_seed=20260715,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _activate_outputs(module: SoftMoEPatchAdapter) -> None:
    with torch.no_grad():
        for expert_index, expert in enumerate(module.experts):
            values = torch.linspace(
                -0.02,
                0.02,
                expert.output_projection.weight.numel(),
            ).reshape_as(expert.output_projection.weight)
            expert.output_projection.weight.copy_(
                values * float(expert_index + 1)
            )
            expert.output_projection.bias.copy_(
                torch.linspace(-0.01, 0.01, module.dim)
            )


def test_soft_moe_matches_locked_dispatch_expert_combine_equation() -> None:
    torch.manual_seed(7)
    module = SoftMoEPatchAdapter(
        dim=16,
        hidden_dim=4,
        num_experts=4,
        residual_scale=0.10,
        router_scale_init=10.0,
        init_seed=101,
    )
    _activate_outputs(module)
    patches = torch.randn(2, 9, 16)

    updated, details = module.mix_patches(patches, return_details=True)

    normalized_tokens = F.normalize(patches.float(), p=2.0, dim=-1, eps=1e-6)
    normalized_slots = F.normalize(
        module.router_slots.float(), p=2.0, dim=0, eps=1e-6
    )
    logits = torch.einsum(
        "bnd,de->bne",
        normalized_tokens,
        normalized_slots * module.router_scale.float(),
    )
    dispatch = logits.softmax(dim=1)
    combine = logits.softmax(dim=2)
    slots = torch.einsum("bne,bnd->bed", dispatch, patches)
    expert_outputs = torch.stack(
        [
            expert.output_projection(
                F.gelu(expert.input_projection(slots[:, expert_index]))
            )
            for expert_index, expert in enumerate(module.experts)
        ],
        dim=1,
    )
    residual = torch.einsum("bne,bed->bnd", combine, expert_outputs)
    expected = patches + 0.10 * residual

    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(details["router_logits"], logits)
    torch.testing.assert_close(details["dispatch_weights"], dispatch)
    torch.testing.assert_close(details["combine_weights"], combine)
    torch.testing.assert_close(details["slots"], slots)
    torch.testing.assert_close(details["expert_outputs"], expert_outputs)
    torch.testing.assert_close(details["residual"], residual)
    torch.testing.assert_close(dispatch.sum(dim=1), torch.ones(2, 4))
    torch.testing.assert_close(combine.sum(dim=2), torch.ones(2, 9))


def test_locked_parameter_count_zero_init_identity_and_prefix_bypass() -> None:
    module = SoftMoEPatchAdapter(
        dim=256,
        hidden_dim=64,
        num_experts=4,
        residual_scale=0.10,
    )
    assert module.added_parameter_count == 133_377
    tokens = torch.randn(2, 21, 256)
    prefix = tokens[:, :5].clone()

    updated = module(tokens, prefix_count=5)

    torch.testing.assert_close(updated, tokens, rtol=0.0, atol=0.0)
    torch.testing.assert_close(updated[:, :5], prefix, rtol=0.0, atol=0.0)
    for expert in module.experts:
        assert torch.count_nonzero(expert.output_projection.weight) == 0
        assert torch.count_nonzero(expert.output_projection.bias) == 0


def test_all_parameter_families_receive_gradient_after_one_warm_update() -> None:
    torch.manual_seed(11)
    module = SoftMoEPatchAdapter(
        dim=32,
        hidden_dim=8,
        num_experts=4,
        residual_scale=0.10,
        init_seed=103,
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    tokens = torch.randn(3, 13, 32)
    target = torch.randn_like(tokens)

    loss = F.mse_loss(module(tokens, prefix_count=3), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = F.mse_loss(module(tokens, prefix_count=3), target)
    second_loss.backward()
    required = {
        "router_slots": module.router_slots,
        "router_scale": module.router_scale,
    }
    for expert_index, expert in enumerate(module.experts):
        required.update(
            {
                f"expert{expert_index}.input.weight": expert.input_projection.weight,
                f"expert{expert_index}.input.bias": expert.input_projection.bias,
                f"expert{expert_index}.output.weight": expert.output_projection.weight,
                f"expert{expert_index}.output.bias": expert.output_projection.bias,
            }
        )
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert int(torch.count_nonzero(parameter.grad).item()) > 0, name


def test_constructor_restores_cpu_and_all_visible_cuda_rng() -> None:
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(19)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    SoftMoEPatchAdapter(dim=32, hidden_dim=8, num_experts=4, init_seed=107)

    assert torch.equal(cpu_before, torch.get_rng_state())
    if torch.cuda.is_available():
        cuda_after = torch.cuda.get_rng_state_all()
        assert len(cuda_before) == len(cuda_after)
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_before, cuda_after)
        )


def test_trace_reports_noncollapsed_normalized_routing() -> None:
    torch.manual_seed(23)
    module = SoftMoEPatchAdapter(
        dim=32,
        hidden_dim=8,
        num_experts=4,
        init_seed=109,
    )
    _activate_outputs(module)
    module(torch.randn(4, 20, 32), prefix_count=4)

    trace = module.trace()
    assert trace["dispatch_weights"].shape == (4, 16, 4)
    assert trace["combine_weights"].shape == (4, 16, 4)
    assert trace["combine_mass"].shape == (4, 4)
    assert trace["combine_entropy"].shape == (4,)
    assert trace["residual_norm"].shape == (4, 16)
    assert trace["patch_indices"].shape == (4, 16)
    assert trace["dispatch_similarity_off_diagonal"].shape == (4, 12)
    assert torch.isfinite(trace["residual_norm_ratio"])
    torch.testing.assert_close(
        trace["dispatch_weights"].sum(dim=1),
        torch.ones(4, 4),
    )
    torch.testing.assert_close(
        trace["combine_weights"].sum(dim=2),
        torch.ones(4, 16),
    )
    assert bool((trace["combine_entropy"] >= 0.0).all())
    assert bool((trace["combine_entropy"] <= 1.0 + 1e-6).all())


def test_invalid_configuration_and_prefix_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SoftMoEPatchAdapter(dim=16, hidden_dim=4, num_experts=1)
    with pytest.raises(ValueError, match=">= 0"):
        SoftMoEPatchAdapter(dim=16, hidden_dim=4, residual_scale=-0.1)
    module = SoftMoEPatchAdapter(dim=16, hidden_dim=4)
    with pytest.raises(ValueError, match="leave at least one"):
        module(torch.randn(2, 5, 16), prefix_count=5)
    assert math.isclose(float(module.router_scale), 10.0)


def test_model_wiring_keeps_native_attention_pruning_and_soft_moe_trace() -> None:
    torch.manual_seed(29)
    model = create_model(num_classes=5, model_config=_tiny_config()).eval()
    assert all(isinstance(block.attn, MultiHeadSelfAttention) for block in model.blocks)
    assert [
        index + 1
        for index, block in enumerate(model.blocks)
        if block.soft_moe_patch_adapter is not None
    ] == [1, 3]
    for block in model.blocks:
        if block.soft_moe_patch_adapter is not None:
            _activate_outputs(block.soft_moe_patch_adapter)

    with torch.inference_mode():
        images = torch.randn(2, 3, 32, 32)
        features = model.forward_features(
            images,
            return_attention=True,
            return_trace=True,
        )
        pruned_features = model.forward_features(images, return_trace=True)

    assert set(features["attentions"]) == {0, 1, 2}
    assert features["attention_representation"] == "mhsa_probability"
    assert features["trace"]["soft_moe_patch_adapter_layers"].tolist() == [1, 3]
    assert set(features["trace"]["soft_moe_router_logits"]) == {1, 3}
    assert set(features["trace"]["soft_moe_patch_indices"]) == {1, 3}
    assert features["trace"]["soft_moe_combine_mass"].shape == (2, 2, 4)
    assert features["trace"]["soft_moe_combine_entropy"].shape == (2, 2)
    assert [
        int(item["layer"]) for item in pruned_features["trace"]["pruning"]
    ] == [1, 2]
    assert pruned_features["patches"].shape[1] == int(
        pruned_features["trace"]["pruning"][-1]["after_count"]
    )


def test_model_constructor_checkpoint_extension_and_zero_init_are_exact() -> None:
    seed = 31
    torch.manual_seed(seed)
    control = create_model(
        num_classes=5,
        model_config=_tiny_config(soft_moe_patch_adapter=False),
    ).eval()
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = create_model(num_classes=5, model_config=_tiny_config()).eval()
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(control_rng, candidate_rng)

    summary = _load_model_state_allowing_extensions(
        candidate,
        control.state_dict(),
        allow_extensions=True,
    )
    assert summary is not None
    assert summary["unexpected_keys"] == []
    expected_suffixes = {"router_slots", "router_scale"}
    for expert_index in range(4):
        expected_suffixes.update(
            {
                f"experts.{expert_index}.input_projection.weight",
                f"experts.{expert_index}.input_projection.bias",
                f"experts.{expert_index}.output_projection.weight",
                f"experts.{expert_index}.output_projection.bias",
            }
        )
    assert set(summary["allowed_missing_keys"]) == {
        f"blocks.{block}.soft_moe_patch_adapter.{suffix}"
        for block in (0, 2)
        for suffix in expected_suffixes
    }
    for key, value in control.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0.0, atol=0.0)

    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        control_logits = control(images)
        candidate_logits = candidate(images)
    torch.testing.assert_close(candidate_logits, control_logits, rtol=0.0, atol=0.0)


def test_model_defaults_and_v8_launcher_controls() -> None:
    defaults = ModelConfig()
    assert defaults.soft_moe_patch_adapter is False
    assert defaults.soft_moe_patch_adapter_layers == "2,5"
    assert defaults.soft_moe_hidden_dim == 64
    assert defaults.soft_moe_num_experts == 4
    assert defaults.soft_moe_residual_scale == pytest.approx(0.10)
    assert defaults.soft_moe_router_scale_init == pytest.approx(10.0)
    assert defaults.soft_moe_init_seed == 20260715

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$SoftMoePatchAdapter = $false" in script
    assert '[string]$SoftMoePatchAdapterLayers = "2,5"' in script
    assert "[int]$SoftMoeHiddenDim = 64" in script
    assert "[int]$SoftMoeNumExperts = 4" in script
    assert '"--soft-moe-patch-adapter"' in script
    assert '"--soft-moe-patch-adapter-layers"' in script
    assert '"--soft-moe-hidden-dim"' in script
    assert '"--soft-moe-num-experts"' in script
