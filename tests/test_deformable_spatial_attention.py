from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from trkh.core.config import ModelConfig
from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.models.model import MultiHeadSelfAttention, create_model


OFFICIAL_ATTENTION_PATH = Path(
    r"D:\DataAI\external_sources\official\dat-cvpr2022"
) / "models" / "dat_blocks.py"


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
        deformable_spatial_attention=True,
        deformable_spatial_attention_layers="2",
        deformable_spatial_attention_groups=2,
        deformable_spatial_attention_kernel_size=5,
        deformable_spatial_attention_offset_range=2.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    return inputs, indices


def _load_official_attention_class():
    if not OFFICIAL_ATTENTION_PATH.is_file():
        pytest.skip(f"Locked official DAT source is absent: {OFFICIAL_ATTENTION_PATH}")
    spec = importlib.util.spec_from_file_location(
        "locked_cvpr2022_dat_blocks",
        OFFICIAL_ATTENTION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked official DAT attention source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DAttentionBaseline


def _copy_candidate_to_official(
    candidate: DeformableSpatialAttention,
    official: torch.nn.Module,
) -> None:
    dim = candidate.dim
    with torch.no_grad():
        official.proj_q.weight.copy_(candidate.qkv.weight[:dim].reshape(dim, dim, 1, 1))
        official.proj_q.bias.copy_(candidate.qkv.bias[:dim])
        official.proj_k.weight.copy_(
            candidate.qkv.weight[dim : 2 * dim].reshape(dim, dim, 1, 1)
        )
        official.proj_k.bias.copy_(candidate.qkv.bias[dim : 2 * dim])
        official.proj_v.weight.copy_(candidate.qkv.weight[2 * dim :].reshape(dim, dim, 1, 1))
        official.proj_v.bias.copy_(candidate.qkv.bias[2 * dim :])
        official.proj_out.weight.copy_(candidate.proj.weight.reshape(dim, dim, 1, 1))
        official.proj_out.bias.copy_(candidate.proj.bias)
        official.conv_offset.load_state_dict(candidate.conv_offset.state_dict())
        official.rpe_table.copy_(candidate.relative_position_bias_table)


def _candidate_parameter_for_official_name(
    candidate: DeformableSpatialAttention,
    official_name: str,
) -> torch.nn.Parameter:
    named = dict(candidate.named_parameters())
    mapping = {
        "rpe_table": "relative_position_bias_table",
        "conv_offset.0.weight": "conv_offset.0.weight",
        "conv_offset.0.bias": "conv_offset.0.bias",
        "conv_offset.1.norm.weight": "conv_offset.1.norm.weight",
        "conv_offset.1.norm.bias": "conv_offset.1.norm.bias",
        "conv_offset.3.weight": "conv_offset.3.weight",
        "proj_q.weight": "qkv.weight",
        "proj_q.bias": "qkv.bias",
        "proj_k.weight": "qkv.weight",
        "proj_k.bias": "qkv.bias",
        "proj_v.weight": "qkv.weight",
        "proj_v.bias": "qkv.bias",
        "proj_out.weight": "proj.weight",
        "proj_out.bias": "proj.bias",
    }
    return named[mapping[official_name]]


def _candidate_gradient_slice(
    candidate: DeformableSpatialAttention,
    official_name: str,
) -> torch.Tensor:
    dim = candidate.dim
    parameter = _candidate_parameter_for_official_name(candidate, official_name)
    gradient = parameter.grad
    if gradient is None:
        raise AssertionError(f"Missing candidate gradient for {official_name}")
    if official_name == "proj_q.weight":
        return gradient[:dim].reshape(dim, dim, 1, 1)
    if official_name == "proj_q.bias":
        return gradient[:dim]
    if official_name == "proj_k.weight":
        return gradient[dim : 2 * dim].reshape(dim, dim, 1, 1)
    if official_name == "proj_k.bias":
        return gradient[dim : 2 * dim]
    if official_name == "proj_v.weight":
        return gradient[2 * dim :].reshape(dim, dim, 1, 1)
    if official_name == "proj_v.bias":
        return gradient[2 * dim :]
    if official_name == "proj_out.weight":
        return gradient.reshape(dim, dim, 1, 1)
    return gradient


def test_dat_patch_equation_positions_and_gradients_match_official() -> None:
    official_class = _load_official_attention_class()
    torch.manual_seed(41)
    candidate = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    ).eval()
    official = official_class(
        (4, 4),
        (4, 4),
        4,
        8,
        2,
        0.0,
        0.0,
        1,
        2.0,
        True,
        False,
        False,
        False,
        2,
    ).eval()
    _copy_candidate_to_official(candidate, official)

    torch.manual_seed(19)
    official_input = torch.randn(2, 32, 4, 4, requires_grad=True)
    candidate_input = official_input.detach().permute(0, 2, 3, 1).reshape(
        2, 16, 32
    ).clone().requires_grad_(True)
    official_output, official_positions, official_reference = official(official_input)
    candidate_output = candidate(
        candidate_input,
        grid_size=(4, 4),
        prefix_count=0,
        patch_indices=torch.arange(16).unsqueeze(0).expand(2, -1),
    )
    candidate_map = candidate_output.reshape(2, 4, 4, 32).permute(0, 3, 1, 2)
    trace = candidate.trace()
    torch.testing.assert_close(candidate_map, official_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        trace["positions"], official_positions, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        trace["reference_positions"], official_reference, atol=1e-6, rtol=1e-6
    )

    candidate_map.square().mean().backward()
    official_output.square().mean().backward()
    candidate_input_gradient = candidate_input.grad.reshape(
        2, 4, 4, 32
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(
        candidate_input_gradient,
        official_input.grad,
        atol=1e-6,
        rtol=1e-6,
    )
    for official_name, official_parameter in official.named_parameters():
        assert official_parameter.grad is not None, official_name
        torch.testing.assert_close(
            _candidate_gradient_slice(candidate, official_name),
            official_parameter.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_dat_prefix_matches_standard_and_proxy_is_normalized() -> None:
    inputs, indices = _inputs()
    torch.manual_seed(23)
    standard = MultiHeadSelfAttention(
        dim=32,
        num_heads=4,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ).eval()
    torch.manual_seed(99)
    candidate = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    ).eval()
    candidate.copy_shared_projections_from(standard)
    standard_output, _ = standard(inputs, return_attention=True)
    candidate_output, attention = candidate(
        inputs,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    torch.testing.assert_close(
        candidate_output[:, :5],
        standard_output[:, :5],
        atol=1e-6,
        rtol=1e-6,
    )
    assert candidate_output.shape == inputs.shape
    assert attention.shape == (2, 4, 21, 21)
    assert torch.isfinite(attention).all()
    assert bool((attention >= 0).all())
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-6,
        rtol=0.0,
    )
    assert int(torch.count_nonzero(attention[:, :, 5:, :5]).item()) == 0


def test_dat_bilinear_proxy_matches_independent_grid_sample_basis() -> None:
    inputs, indices = _inputs()
    module = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    ).eval()
    module(
        inputs,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    trace = module.trace()
    positions = trace["positions"]
    torch.manual_seed(29)
    sample_attention = torch.rand(2, 4, 16, 16)
    sample_attention = sample_attention / sample_attention.sum(dim=-1, keepdim=True)
    dense, _ = module.bilinear_scatter_attention(
        sample_attention,
        positions.reshape(4, 4, 4, 2),
    )
    basis = torch.eye(16).reshape(16, 1, 4, 4)
    sampled_basis = []
    for batch_index in range(2):
        group_rows = []
        for group_index in range(2):
            sampled = F.grid_sample(
                basis,
                positions[batch_index, group_index][..., (1, 0)].unsqueeze(0).expand(
                    16, -1, -1, -1
                ),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            group_rows.append(sampled[:, 0].flatten(1).transpose(0, 1))
        sampled_basis.append(torch.stack(group_rows, dim=0))
    sampled_basis_tensor = torch.stack(sampled_basis, dim=0)
    sampled_basis_heads = sampled_basis_tensor.unsqueeze(2).expand(
        -1, -1, 2, -1, -1
    ).reshape(2, 4, 16, 16)
    independent = sample_attention @ sampled_basis_heads
    independent = independent / independent.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    torch.testing.assert_close(
        dense,
        independent,
        atol=1e-6,
        rtol=1e-6,
    )


def test_dat_component_families_receive_finite_nonzero_gradients() -> None:
    inputs, indices = _inputs()
    module = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    )
    output = module(
        inputs,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=indices,
    )
    output.square().mean().backward()
    parameters = dict(module.named_parameters())
    for name in (
        "qkv.weight",
        "proj.weight",
        "conv_offset.0.weight",
        "conv_offset.1.norm.weight",
        "conv_offset.3.weight",
        "relative_position_bias_table",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert int(torch.count_nonzero(gradient).item()) > 0, name


def test_dat_control_candidate_common_state_and_rng_are_bit_exact() -> None:
    torch.manual_seed(42)
    control = create_model(
        num_classes=5,
        model_config=_tiny_model_config(deformable_spatial_attention=False),
    )
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=_tiny_model_config())
    candidate_rng = torch.random.get_rng_state().clone()
    assert torch.equal(control_rng, candidate_rng)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    common_keys = sorted(set(control_state).intersection(candidate_state))
    assert common_keys
    for key in common_keys:
        assert torch.equal(control_state[key], candidate_state[key]), key
    assert isinstance(control.blocks[1].attn, MultiHeadSelfAttention)
    assert isinstance(candidate.blocks[0].attn, MultiHeadSelfAttention)
    assert isinstance(candidate.blocks[1].attn, DeformableSpatialAttention)
    assert isinstance(candidate.blocks[2].attn, MultiHeadSelfAttention)
    parameter_delta = sum(value.numel() for value in candidate.parameters()) - sum(
        value.numel() for value in control.parameters()
    )
    assert 0 < parameter_delta <= 50_000


def test_dat_model_trace_and_pruning_after_block_two() -> None:
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
        1: "deformable_bilinear_sample_proxy",
        2: "mhsa_probability",
    }
    trace = features["trace"]
    assert trace["deformable_spatial_attention_layers"].tolist() == [2]
    assert trace["deformable_offset_rms"].shape == (1,)
    assert trace["deformable_offset_rms_per_group"].shape == (1, 2)
    assert trace["deformable_valid_interpolation_mass_mean"].shape == (1,)
    with torch.inference_mode():
        pruned = model.forward_features(images, return_trace=True)
    assert pruned["patches"].shape[1] == 8


def test_dat_rejects_sparse_grid_and_locked_route_conflicts() -> None:
    inputs, indices = _inputs()
    module = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    )
    with pytest.raises(ValueError, match="complete patch grid"):
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
                deformable_spatial_attention_layers="1",
            ),
        )
    with pytest.raises(ValueError, match="Foveal Aggregated"):
        create_model(
            num_classes=5,
            model_config=_tiny_model_config(
                foveal_aggregated_attention=True,
                foveal_aggregated_attention_layers="1",
                foveal_aggregated_attention_pool_size=2,
            ),
        )


def test_v8_launcher_exposes_locked_dat_controls() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")
    assert "[bool]$DeformableSpatialAttention = $false" in script
    assert '"--deformable-spatial-attention"' in script
    assert '"--deformable-spatial-attention-layers"' in script
    assert '"--deformable-spatial-attention-groups"' in script
    assert '"--deformable-spatial-attention-kernel-size"' in script
    assert '"--deformable-spatial-attention-offset-range"' in script
