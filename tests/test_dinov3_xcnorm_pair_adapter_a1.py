from __future__ import annotations

import pytest
import torch

from trkh.models.dinov3_xcnorm_pair_adapter_a1 import (
    LOCAL_CONV_A1_MODE,
    LOCAL_XCNORM_A1_MODE,
    XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT,
    DinoV3XCNormPairA1ContractError,
    DinoV3XCNormPairAdapterA1,
)


def _classifier_weight() -> torch.Tensor:
    weight = torch.zeros(5, 384)
    weight[0, 0] = -1.0
    weight[2, :2] = torch.tensor((-0.7, -1.0))
    weight[4, :3] = torch.tensor((-0.4, -0.6, -1.0))
    weight[3, 3] = 0.5
    return weight


def _adapter(mode: str) -> DinoV3XCNormPairAdapterA1:
    adapter = DinoV3XCNormPairAdapterA1(mode)
    adapter.bind_classifier_weight(_classifier_weight())
    return adapter


def test_a1_modes_have_exact_paired_state_geometry_and_capacity() -> None:
    torch.manual_seed(101)
    conv = _adapter(LOCAL_CONV_A1_MODE)
    xcnorm = _adapter(LOCAL_XCNORM_A1_MODE)
    xcnorm.load_state_dict(conv.state_dict(), strict=True)

    conv_geometry = {
        name: tuple(tensor.shape)
        for name, tensor in conv.state_dict().items()
    }
    xcnorm_geometry = {
        name: tuple(tensor.shape)
        for name, tensor in xcnorm.state_dict().items()
    }
    assert conv_geometry == xcnorm_geometry
    assert {
        name: tuple(parameter.shape)
        for name, parameter in conv.named_parameters()
    } == {
        "core.local_weight": (8, 8, 3, 3),
        "core.token_projection.weight": (8, 384),
        "core.pair_projection.weight": (3, 8),
    }
    assert conv.added_parameter_count() == XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT
    assert xcnorm.added_parameter_count() == XCNORM_PAIR_A1_ADDED_PARAMETER_COUNT
    assert conv.parameter_telemetry()["added_parameter_count"] == 3672
    assert xcnorm.parameter_telemetry()["added_parameter_count"] == 3672


@pytest.mark.parametrize("mode", (LOCAL_CONV_A1_MODE, LOCAL_XCNORM_A1_MODE))
def test_a1_zero_initialization_cap_and_branch_off_stream_identity(mode: str) -> None:
    torch.manual_seed(103)
    adapter = _adapter(mode).eval()
    z_patch = 100.0 * torch.randn(2, 256, 384)
    post_mhsa = torch.randn(2, 261, 384)
    u_patch = post_mhsa[:, 5:]
    classifier_weight = _classifier_weight()

    with torch.no_grad():
        residual, trace = adapter.residual_from_final_norm1(
            z_patch,
            u_patch,
            classifier_weight,
            return_trace=True,
        )
        merged = adapter.merge_parallel_final_mhsa(
            post_mhsa,
            z_patch,
            classifier_weight,
        )
    assert residual.shape == z_patch.shape
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(merged, post_mhsa)
    assert trace["input_tensor"] == "final_block_norm1_patch_tokens"

    with torch.no_grad():
        adapter.core.pair_projection.weight.normal_(0.0, 0.2)
        active, active_trace = adapter.residual_from_final_norm1(
            z_patch,
            u_patch,
            classifier_weight,
            return_trace=True,
        )
        off = adapter.residual_from_final_norm1(
            z_patch,
            u_patch,
            classifier_weight,
            branch_off=True,
        )
        merged_active = adapter.merge_parallel_final_mhsa(
            post_mhsa,
            z_patch,
            classifier_weight,
        )
        merged_off = adapter.merge_parallel_final_mhsa(
            post_mhsa,
            z_patch,
            classifier_weight,
            branch_off=True,
        )

    assert torch.count_nonzero(active) > 0
    assert torch.isfinite(active).all()
    actual_ratio = active.float().norm(dim=-1) / u_patch.float().norm(
        dim=-1
    ).clamp_min(1e-12)
    assert torch.allclose(
        active_trace["residual_norm_ratio"], actual_ratio, atol=2e-6, rtol=2e-6
    )
    assert float(actual_ratio.max()) < 0.04
    assert torch.equal(off, torch.zeros_like(u_patch))
    assert torch.equal(merged_off, post_mhsa)
    assert torch.equal(merged_active[:, :5], post_mhsa[:, :5])
    assert torch.count_nonzero(merged_active[:, 5:] - post_mhsa[:, 5:]) > 0


@pytest.mark.parametrize("mode", (LOCAL_CONV_A1_MODE, LOCAL_XCNORM_A1_MODE))
def test_a1_active_residual_gradients_are_finite_and_inputs_stay_frozen(
    mode: str,
) -> None:
    torch.manual_seed(107)
    adapter = _adapter(mode).train()
    with torch.no_grad():
        adapter.core.pair_projection.weight.normal_(0.0, 0.1)
    z_patch = torch.randn(2, 256, 384, requires_grad=True)
    residual_reference = torch.randn(2, 256, 384, requires_grad=True)
    classifier_weight = _classifier_weight()

    residual = adapter(z_patch, residual_reference, classifier_weight)
    loss = residual.float().square().mean() + residual.float().mean()
    loss.backward()

    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert z_patch.grad is None
    assert residual_reference.grad is None
    assert classifier_weight.grad is None


def test_a1_provenance_preserves_core_safety_and_limits_margin_claim() -> None:
    adapter = _adapter(LOCAL_XCNORM_A1_MODE)
    provenance = adapter.fusion_provenance()

    assert provenance["placement"] == "parallel_to_frozen_final_mhsa"
    assert provenance["merge_point"] == (
        "post_final_mhsa_residual_before_frozen_final_mlp_and_final_norm"
    )
    assert provenance["backbone_contract"] == "frozen_eval"
    assert provenance["classifier_contract"] == "frozen_and_explicitly_bound"
    assert provenance["classifier_axis_semantics"] == (
        "fixed_injection_space_prior_not_exact_output_margin_mapping_"
        "after_frozen_mlp_and_final_norm"
    )
    assert provenance["local_score_input"] == (
        "z_patch=frozen_final_block_norm1_patch_tokens"
    )
    assert provenance["residual_bound_reference"] == (
        "u_patch=frozen_post_final_mhsa_residual_patch_tokens"
    )
    assert "u_full=x_pre+drop_path1" in provenance["cache_precheck_attestation"]
    assert provenance["residual_ratio_cap_semantics"] == (
        "norm(r_patch)_2/norm(u_patch)_2<0.04_per_patch"
    )
    assert provenance["pooling_contract"] == (
        "frozen_final_mlp_then_final_norm_then_mean_patch_pool_then_head"
    )
    assert provenance["xcnorm_fixed_response_scale"] == (
        "sqrt(8)_analytic_kaiming_rms_match_no_data_calibration"
    )
    assert "degenerate_energy_zero_mask" in provenance["xcnorm_denominator"]
    assert provenance["observed_added_parameter_count"] == 3672


def test_a1_binding_rejects_a_trainable_classifier() -> None:
    adapter = DinoV3XCNormPairAdapterA1(LOCAL_XCNORM_A1_MODE)
    classifier_weight = _classifier_weight().requires_grad_()
    with pytest.raises(DinoV3XCNormPairA1ContractError, match="frozen"):
        adapter.bind_classifier_weight(classifier_weight)


def test_a1_residual_reference_contract_fails_closed() -> None:
    adapter = _adapter(LOCAL_XCNORM_A1_MODE)
    z_patch = torch.randn(1, 256, 384)
    classifier_weight = _classifier_weight()

    with pytest.raises(DinoV3XCNormPairA1ContractError, match="shape"):
        adapter.residual_from_final_norm1(
            z_patch,
            torch.randn(1, 255, 384),
            classifier_weight,
        )
    with pytest.raises(DinoV3XCNormPairA1ContractError, match="dtype"):
        adapter.residual_from_final_norm1(
            z_patch,
            torch.randn(1, 256, 384, dtype=torch.float64),
            classifier_weight,
        )
    if torch.cuda.is_available():
        with pytest.raises(DinoV3XCNormPairA1ContractError, match="device"):
            adapter.residual_from_final_norm1(
                z_patch.cuda(),
                torch.randn(1, 256, 384),
                classifier_weight,
            )
