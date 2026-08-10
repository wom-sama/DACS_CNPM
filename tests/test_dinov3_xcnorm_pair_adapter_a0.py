from __future__ import annotations

import math

import pytest
import torch

from trkh.models.dinov3_xcnorm_pair_adapter_a0 import (
    LOCAL_CONV_A0_MODE,
    LOCAL_XCNORM_A0_MODE,
    XCNORM_PAIR_ADDED_PARAMETER_COUNT,
    XCNORM_PAIR_RESIDUAL_RATIO_CAP,
    DinoV3XCNormPairAdapterA0,
    DinoV3XCNormPairContractError,
)


def _classifier_weight() -> torch.Tensor:
    weight = torch.zeros(5, 384)
    weight[0, 0] = -1.0
    weight[2, :2] = torch.tensor((-0.7, -1.0))
    weight[4, :3] = torch.tensor((-0.4, -0.6, -1.0))
    weight[3, 3] = 0.5
    return weight


def _adapter(mode: str = LOCAL_XCNORM_A0_MODE) -> DinoV3XCNormPairAdapterA0:
    module = DinoV3XCNormPairAdapterA0(mode)
    module.bind_classifier_weight(_classifier_weight())
    return module


def _nonzero_gradient(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


@pytest.mark.parametrize("mode", (LOCAL_CONV_A0_MODE, LOCAL_XCNORM_A0_MODE))
def test_exact_capacity_step_zero_identity_shapes_and_bound(mode: str) -> None:
    torch.manual_seed(11)
    adapter = _adapter(mode).eval()
    tokens = torch.randn(2, 256, 384)
    classifier_weight = _classifier_weight()
    base_logits = torch.randn(2, 5)

    with torch.no_grad():
        fused, trace = adapter.forward_with_trace(tokens, classifier_weight)
        logits = adapter.logits_from_cached_tokens(
            tokens,
            classifier_weight,
            base_logits=base_logits,
        )

    named_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in adapter.named_parameters()
    }
    assert named_shapes == {
        "local_weight": (8, 8, 3, 3),
        "token_projection.weight": (8, 384),
        "pair_projection.weight": (3, 8),
    }
    assert adapter.added_parameter_count() == XCNORM_PAIR_ADDED_PARAMETER_COUNT
    assert adapter.parameter_telemetry() == {
        "added_parameter_count": 3672,
        "trainable_added_parameter_count": 3672,
        "token_projection_parameter_count": 3072,
        "local_template_parameter_count": 576,
        "pair_projection_parameter_count": 24,
    }
    assert torch.equal(fused, tokens)
    assert torch.equal(logits, base_logits)
    assert trace["projected_feature_map"].shape == (2, 8, 16, 16)
    assert trace["local_response_valid"].shape == (2, 8, 14, 14)
    assert trace["activated_local_map"].shape == (2, 8, 16, 16)
    assert trace["pair_scores"].shape == (2, 256, 3)
    assert trace["residual"].shape == tokens.shape
    assert torch.count_nonzero(trace["pair_scores"]) == 0
    assert torch.count_nonzero(trace["residual"]) == 0
    assert torch.isfinite(trace["local_response_valid"]).all()
    assert adapter.classifier_binding_matches(classifier_weight)

    with torch.no_grad():
        adapter.pair_projection.weight.normal_(0.0, 0.2)
        _fused, active_trace = adapter.forward_with_trace(tokens, classifier_weight)
    ratio = active_trace["residual_norm_ratio"]
    assert torch.isfinite(ratio).all()
    assert float(ratio.max()) < XCNORM_PAIR_RESIDUAL_RATIO_CAP
    assert float(ratio.max()) > 0.0
    activated = active_trace["activated_local_map"]
    assert torch.count_nonzero(activated[:, :, 0, :]) == 0
    assert torch.count_nonzero(activated[:, :, -1, :]) == 0
    assert torch.count_nonzero(activated[:, :, :, 0]) == 0
    assert torch.count_nonzero(activated[:, :, :, -1]) == 0


def test_pair_margin_dual_map_decouples_all_three_classifier_margins() -> None:
    adapter = _adapter()
    tokens = torch.randn(1, 256, 384)
    classifier_weight = _classifier_weight()
    rows = classifier_weight[1:2] - classifier_weight[[0, 2, 4]]

    for pair_index in range(3):
        scores = torch.zeros(1, 256, 3)
        scores[0, 73, pair_index] = 0.4
        residual, _trace = adapter.bounded_pair_residual(
            scores,
            tokens,
            classifier_weight,
        )
        margin_change = torch.matmul(residual[0, 73].float(), rows.T)
        assert float(margin_change[pair_index]) > 0.0
        other = torch.cat(
            (margin_change[:pair_index], margin_change[pair_index + 1 :])
        )
        assert torch.allclose(other, torch.zeros_like(other), atol=2e-6, rtol=0.0)

    epsilon = 1e-3
    signed_derivatives = []
    for pair_index in range(3):
        positive = torch.zeros(1, 256, 3)
        negative = torch.zeros_like(positive)
        positive[0, 73, pair_index] = epsilon
        negative[0, 73, pair_index] = -epsilon
        plus, _ = adapter.bounded_pair_residual(
            positive,
            tokens,
            classifier_weight,
        )
        minus, _ = adapter.bounded_pair_residual(
            negative,
            tokens,
            classifier_weight,
        )
        finite_difference = (plus[0, 73] - minus[0, 73]) / (2.0 * epsilon)
        signed_derivatives.append(float(torch.dot(finite_difference, rows[pair_index])))
    assert all(math.isfinite(value) and value > 0.0 for value in signed_derivatives)


def test_xcnorm_positive_affine_invariance_conv_noninvariance_and_scale() -> None:
    torch.manual_seed(23)
    xcnorm = _adapter(LOCAL_XCNORM_A0_MODE).eval()
    conv = _adapter(LOCAL_CONV_A0_MODE).eval()
    conv.load_state_dict(xcnorm.state_dict(), strict=True)
    feature_map = torch.randn(2, 8, 16, 16)
    affine_map = 2.3 * feature_map + 0.7

    with torch.no_grad():
        xcnorm_base = xcnorm.local_response(feature_map)
        xcnorm_affine = xcnorm.local_response(affine_map)
        conv_base = conv.local_response(feature_map)
        conv_affine = conv.local_response(affine_map)

    assert torch.allclose(xcnorm_base, xcnorm_affine, atol=3e-5, rtol=3e-5)
    assert float((conv_base - conv_affine).abs().mean()) > 1e-2

    # Scale parity is defined on A0's actual fixed-LN -> Kaiming projection,
    # not on an arbitrary unit-variance feature map that bypasses it.
    patch_tokens = torch.randn(8, 256, 384)
    with torch.no_grad():
        projected = xcnorm.project_token_grid(patch_tokens)
        xcnorm_matched = xcnorm.local_response(projected)
        conv_matched = conv.local_response(projected)
    xcnorm_rms = float(xcnorm_matched.square().mean().sqrt())
    conv_rms = float(conv_matched.square().mean().sqrt())
    response_scale_ratio = xcnorm_rms / conv_rms
    assert 0.75 <= response_scale_ratio <= 1.33
    assert xcnorm.fusion_provenance()["xcnorm_fixed_response_scale"] == (
        "sqrt(8)_analytic_kaiming_rms_match_no_data_calibration"
    )


def test_zero_output_opens_upstream_gradients_only_after_first_update() -> None:
    torch.manual_seed(37)
    adapter = _adapter(LOCAL_XCNORM_A0_MODE).train()
    tokens = torch.randn(2, 256, 384)
    classifier_weight = _classifier_weight()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.25)

    first_logits = adapter.logits_from_cached_tokens(tokens, classifier_weight)
    first_loss = -(first_logits[:, 1] - first_logits[:, 0]).mean()
    first_loss.backward()
    assert _nonzero_gradient(adapter.pair_projection.weight)
    assert not _nonzero_gradient(adapter.token_projection.weight)
    assert not _nonzero_gradient(adapter.local_weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_logits = adapter.logits_from_cached_tokens(tokens, classifier_weight)
    second_loss = -(second_logits[:, 1] - second_logits[:, 0]).mean()
    second_loss.backward()
    assert _nonzero_gradient(adapter.pair_projection.weight)
    assert _nonzero_gradient(adapter.token_projection.weight)
    assert _nonzero_gradient(adapter.local_weight)


def test_branch_off_is_bit_exact_after_adapter_is_active() -> None:
    torch.manual_seed(41)
    adapter = _adapter()
    with torch.no_grad():
        adapter.pair_projection.weight.normal_(0.0, 0.4)
    tokens = torch.randn(2, 256, 384)
    classifier_weight = _classifier_weight()
    base_logits = torch.randn(2, 5)

    with torch.no_grad():
        fused, trace = adapter.forward_with_trace(
            tokens,
            classifier_weight,
            branch_off=True,
        )
        logits = adapter.logits_from_cached_tokens(
            tokens,
            classifier_weight,
            base_logits=base_logits,
            branch_off=True,
        )

    assert torch.equal(fused, tokens)
    assert torch.equal(logits, base_logits)
    assert trace["branch_off"] is True
    assert torch.count_nonzero(trace["residual"]) == 0


def test_xcnorm_fp32_statistics_remain_finite_for_bf16_near_constant_input() -> None:
    torch.manual_seed(53)
    adapter = _adapter(LOCAL_XCNORM_A0_MODE).eval()
    feature_map = (
        torch.full((2, 8, 16, 16), 0.25)
        + torch.randn(2, 8, 16, 16) * 5e-3
    ).to(torch.bfloat16)

    with torch.no_grad(), torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
    ):
        response = adapter.local_response(feature_map)

    assert response.dtype == torch.bfloat16
    assert response.shape == (2, 8, 14, 14)
    assert torch.isfinite(response.float()).all()


@pytest.mark.parametrize("constant", (0.0, 0.1, 1.0, 100.0))
def test_xcnorm_degenerate_constant_map_is_exact_zero_with_finite_gradients(
    constant: float,
) -> None:
    adapter = _adapter(LOCAL_XCNORM_A0_MODE).train()
    feature_map = torch.full(
        (2, 8, 16, 16),
        constant,
        requires_grad=True,
    )

    response = adapter.local_response(feature_map)
    response.sum().backward()

    assert torch.count_nonzero(response) == 0
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert adapter.local_weight.grad is not None
    assert torch.isfinite(adapter.local_weight.grad).all()


def test_active_branch_requires_explicit_frozen_classifier_binding() -> None:
    adapter = DinoV3XCNormPairAdapterA0(LOCAL_XCNORM_A0_MODE)
    tokens = torch.randn(1, 256, 384)
    with pytest.raises(DinoV3XCNormPairContractError, match="bind_classifier_weight"):
        adapter(tokens, _classifier_weight())
    assert torch.equal(adapter(tokens, _classifier_weight(), branch_off=True), tokens)


def test_active_branch_rejects_classifier_drift_after_binding() -> None:
    adapter = _adapter()
    tokens = torch.randn(1, 256, 384)
    drifted_weight = _classifier_weight()
    drifted_weight[0, 0] += 0.1

    with pytest.raises(DinoV3XCNormPairContractError, match="no longer matches"):
        adapter(tokens, drifted_weight)
    assert torch.equal(adapter(tokens, drifted_weight, branch_off=True), tokens)


def test_cached_frozen_inputs_and_head_never_receive_gradients() -> None:
    torch.manual_seed(67)
    classifier_weight = _classifier_weight().requires_grad_()
    adapter = DinoV3XCNormPairAdapterA0(LOCAL_XCNORM_A0_MODE).train()
    adapter.bind_classifier_weight(classifier_weight)
    with torch.no_grad():
        adapter.pair_projection.weight.normal_(0.0, 0.2)
    tokens = torch.randn(2, 256, 384, requires_grad=True)
    base_logits = torch.randn(2, 5, requires_grad=True)

    logits = adapter.logits_from_cached_tokens(
        tokens,
        classifier_weight,
        base_logits=base_logits,
    )
    logits.square().mean().backward()

    assert tokens.grad is None
    assert classifier_weight.grad is None
    assert base_logits.grad is None
    assert _nonzero_gradient(adapter.pair_projection.weight)
    assert _nonzero_gradient(adapter.token_projection.weight)
    assert _nonzero_gradient(adapter.local_weight)


def test_classifier_binding_survives_exact_state_round_trip() -> None:
    classifier_weight = _classifier_weight()
    source = _adapter()
    restored = DinoV3XCNormPairAdapterA0(LOCAL_XCNORM_A0_MODE)
    restored.load_state_dict(source.state_dict(), strict=True)

    assert restored.classifier_binding_matches(classifier_weight)
    tiny_drift = classifier_weight.clone()
    tiny_drift[0, 0] += 1e-7
    assert not restored.classifier_binding_matches(tiny_drift)
