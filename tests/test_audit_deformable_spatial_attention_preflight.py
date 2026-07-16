from __future__ import annotations

import json
from pathlib import Path

import torch

from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.tools.audit_deformable_spatial_attention_preflight import (
    _candidate_config,
    _control_config,
    _independent_bilinear_scatter,
    _official_equation_replay,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ATTENTION = Path(
    r"D:\DataAI\external_sources\official\dat-cvpr2022\models\dat_blocks.py"
)


def test_dat_preflight_configs_lock_only_the_candidate_attention() -> None:
    source = {"pretrained": True, "early_token_mask_keep_rate": 0.7}
    candidate = _candidate_config(source)
    control = _control_config(source)
    assert candidate["pretrained"] is False
    assert candidate["early_token_mask_keep_rate"] == 1.0
    assert candidate["deformable_spatial_attention"] is True
    assert candidate["deformable_spatial_attention_layers"] == "2"
    assert candidate["deformable_spatial_attention_groups"] == 2
    assert candidate["deformable_spatial_attention_kernel_size"] == 5
    assert candidate["deformable_spatial_attention_offset_range"] == 2.0
    assert control["deformable_spatial_attention"] is False
    for key in (
        "gated_relative_position_attention",
        "visual_contrast_attention",
        "foveal_aggregated_attention",
        "cross_covariance_attention",
        "dynamic_graph_mixer",
    ):
        assert candidate[key] is False
        assert control[key] is False


def test_dat_preflight_official_equation_replays_all_gradients() -> None:
    assert OFFICIAL_ATTENTION.is_file()
    result = _official_equation_replay(OFFICIAL_ATTENTION)
    assert result["passed"] is True
    assert result["output_maximum_absolute_error"] <= 1e-6
    assert result["position_maximum_absolute_error"] <= 1e-6
    assert result["reference_maximum_absolute_error"] <= 1e-6
    assert result["input_gradient_maximum_absolute_error"] <= 1e-6
    assert result["parameter_gradient_maximum_absolute_error"] <= 1e-6
    assert set(result["parameter_gradient_errors"]) == {
        "conv_offset.0.bias",
        "conv_offset.0.weight",
        "conv_offset.1.norm.bias",
        "conv_offset.1.norm.weight",
        "conv_offset.3.weight",
        "proj_k.bias",
        "proj_k.weight",
        "proj_out.bias",
        "proj_out.weight",
        "proj_q.bias",
        "proj_q.weight",
        "proj_v.bias",
        "proj_v.weight",
        "rpe_table",
    }


def test_dat_preflight_proxy_replay_is_independent_and_exact() -> None:
    torch.manual_seed(17)
    module = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    ).eval()
    inputs = torch.randn(2, 21, 32)
    indices = torch.arange(16).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        module(
            inputs,
            grid_size=(4, 4),
            prefix_count=5,
            patch_indices=indices,
        )
    positions = module.trace()["positions"]
    sample_attention = torch.rand(2, 4, 16, 16)
    sample_attention /= sample_attention.sum(dim=-1, keepdim=True)
    implementation, implementation_mass = module.bilinear_scatter_attention(
        sample_attention,
        positions.reshape(4, 4, 4, 2),
    )
    independent, independent_mass = _independent_bilinear_scatter(
        sample_attention,
        positions,
    )
    torch.testing.assert_close(implementation, independent, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        implementation_mass, independent_mass, atol=1e-6, rtol=1e-6
    )
    fast_mass = module.sampled_valid_interpolation_mass(
        sample_attention,
        positions.reshape(4, 4, 4, 2),
    )
    torch.testing.assert_close(fast_mass, implementation_mass, atol=1e-6, rtol=1e-6)


def test_dat_a1_launcher_keeps_train_only_gate_and_locked_recipe() -> None:
    script = (
        REPO_ROOT / "scripts" / "run_trkh_deformable_spatial_attention_a1.ps1"
    ).read_text(encoding="utf-8")
    for fragment in (
        "audit_deformable_spatial_attention_preflight",
        "formal_pair_permission",
        "Epochs = 5",
        "SchedulerTotalEpochs = 5",
        "BatchSize = 32",
        "GradAccumSteps = 2",
        "DisableBalancedEpochSampling = $true",
        "LearningRate = 2.5e-4",
        "SkipFinalTest = $true",
        "DeformableSpatialAttentionLayers = \"2\"",
        "DeformableSpatialAttentionGroups = 2",
        "DeformableSpatialAttentionKernelSize = 5",
        "DeformableSpatialAttentionOffsetRange = 2.0",
    ):
        assert fragment in script
    assert "RunFinalTest" not in script


def test_locked_source_config_remains_parseable() -> None:
    path = (
        REPO_ROOT
        / "runs"
        / "full_v8_yolof_randominit_30e_20260714_105524"
        / "resolved_config.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload["model_config"], dict)
    assert payload["model_config"]["depth"] == 8
