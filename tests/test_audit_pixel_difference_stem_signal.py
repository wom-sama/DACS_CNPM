from __future__ import annotations

import os

import torch
from torch import nn

from trkh.models.model import HybridConvStem
from trkh.tools.audit_pixel_difference_stem_signal import (
    AD_PERMUTATION,
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    FEATURE_NAMES,
    OBJECT_FEATURE_INDICES,
    OPERATOR_ORDER,
    RD_NEGATIVE,
    RD_POSITIVE,
    FrozenStemTrace,
    activation_descriptors,
    assess_information_gate,
    convert_ad_weight,
    convert_cd_weight,
    convert_rd_weight,
    independent_conversion_oracle,
    parse_args,
    region_masks,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.seed == 42
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert OPERATOR_ORDER == ("cd", "ad", "rd")
    assert len(FEATURE_NAMES) == 24
    assert len(OBJECT_FEATURE_INDICES) == 15


def test_cd_conversion_matches_independent_equation() -> None:
    weight = torch.arange(18, dtype=torch.float64).reshape(2, 1, 3, 3) / 10.0
    converted = convert_cd_weight(weight)
    oracle = independent_conversion_oracle(weight, "cd")
    expected_center = weight[..., 1, 1] - weight.sum(dim=(-2, -1))
    assert torch.allclose(converted, oracle, atol=1e-12, rtol=0.0)
    assert torch.allclose(converted[..., 1, 1], expected_center, atol=1e-12, rtol=0.0)
    mask = torch.ones((3, 3), dtype=torch.bool)
    mask[1, 1] = False
    assert torch.equal(converted[..., mask], weight[..., mask])


def test_ad_conversion_uses_official_clockwise_permutation() -> None:
    weight = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    converted = convert_ad_weight(weight).flatten()
    expected = weight.flatten() - weight.flatten()[list(AD_PERMUTATION)]
    assert torch.equal(converted, expected)
    assert torch.equal(convert_ad_weight(weight), independent_conversion_oracle(weight, "ad"))


def test_rd_conversion_uses_source_indices_one_to_eight() -> None:
    weight = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)
    converted = convert_rd_weight(weight).flatten()
    oracle = independent_conversion_oracle(weight, "rd").flatten()
    assert torch.equal(converted, oracle)
    assert converted[12].item() == 0.0
    for source_index, (positive, negative) in enumerate(
        zip(RD_POSITIVE, RD_NEGATIVE), start=1
    ):
        assert converted[positive].item() == float(source_index)
        assert converted[negative].item() == -float(source_index)

    changed_unused = weight.clone()
    changed_unused.flatten()[0] = 1000.0
    assert torch.equal(convert_rd_weight(changed_unused), convert_rd_weight(weight))
    changed_row_major_center = weight.clone()
    changed_row_major_center.flatten()[4] = 1000.0
    assert not torch.equal(
        convert_rd_weight(changed_row_major_center), convert_rd_weight(weight)
    )


def test_converted_weight_gradients_match_direct_equations() -> None:
    torch.manual_seed(19)
    images = torch.randn(2, 3, 9, 11, dtype=torch.float64)
    for operator in OPERATOR_ORDER:
        weight = torch.randn(4, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        converted = {
            "cd": convert_cd_weight,
            "ad": convert_ad_weight,
            "rd": convert_rd_weight,
        }[operator](weight)
        output = nn.functional.conv2d(
            images, converted, padding=converted.shape[-1] // 2
        )
        loss = output.square().mean()
        gradient = torch.autograd.grad(loss, weight)[0]
        assert torch.isfinite(output).all()
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_frozen_trace_preserves_shapes_and_uses_5x5_rd() -> None:
    torch.manual_seed(23)
    source = HybridConvStem(
        in_channels=3,
        stem_channels=4,
        embed_dim=16,
        pooling_mode="max",
    ).eval()
    native = FrozenStemTrace(source, ("cv", "cv", "cv"))
    pdc = FrozenStemTrace(source, OPERATOR_ORDER)
    images = torch.randn(2, 3, 32, 40)
    with torch.inference_mode():
        native_outputs = native(images)
        pdc_outputs = pdc(images)
    assert [tuple(value.shape) for value in native_outputs] == [
        tuple(value.shape) for value in pdc_outputs
    ]
    assert [value.shape[-2:] for value in pdc_outputs] == [
        torch.Size((16, 20)),
        torch.Size((8, 10)),
        torch.Size((4, 5)),
    ]
    assert pdc.blocks[2].block.conv.kernel_size == (5, 5)
    assert all(not parameter.requires_grad for parameter in pdc.parameters())


def test_region_descriptors_are_finite_and_geometry_sensitive() -> None:
    activation = torch.zeros(2, 3, 16, 16)
    activation[:, :, 4:12, 4:12] = 1.0
    activation[:, :, 6:10, 6:10] = 2.0
    bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(2, 1)
    core, boundary, outside = region_masks(bboxes, 16, 16)
    assert torch.all(core.flatten(1).any(dim=1))
    assert torch.all(boundary.flatten(1).any(dim=1))
    assert torch.all(outside.flatten(1).any(dim=1))
    descriptors = activation_descriptors(activation, bboxes)
    assert descriptors.shape == (2, 8)
    assert torch.isfinite(descriptors).all()
    assert torch.all(descriptors[:, 0] > descriptors[:, 2])
    assert torch.all(descriptors[:, 2] > descriptors[:, 4])


def _condition_metrics(
    *, auroc: float, tp: float, fp: float, tp_median: float, fp_median: float
) -> dict[str, float]:
    return {
        "auroc": auroc,
        "tp_retention": tp,
        "fp_rejection": fp,
        "tp_median_score": tp_median,
        "fp_median_score": fp_median,
    }


def _passing_metrics() -> dict[str, object]:
    conditions = ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    native_object = {
        condition: _condition_metrics(
            auroc=0.62, tp=0.98, fp=0.12, tp_median=0.65, fp_median=0.45
        )
        for condition in conditions
    }
    pdc_object = {
        condition: _condition_metrics(
            auroc=0.68, tp=0.98 if condition == "clean" else 0.93,
            fp=0.25 if condition == "clean" else 0.18,
            tp_median=0.70,
            fp_median=0.40,
        )
        for condition in conditions
    }
    pdc_context = {
        condition: _condition_metrics(
            auroc=0.69, tp=0.98, fp=0.24, tp_median=0.70, fp_median=0.40
        )
        for condition in conditions
    }
    return {
        "native": {"object": {"conditions": native_object}},
        "pdc": {
            "object": {"conditions": pdc_object},
            "context": {"conditions": pdc_context},
        },
    }


def test_information_gate_accepts_locked_gain_and_rejects_precision_failure() -> None:
    responses = {"passing_block_count": 2}
    result = assess_information_gate(
        metrics=_passing_metrics(), response_comparisons=responses
    )
    assert result["passed"] is True
    assert result["failed_checks"] == []

    metrics = _passing_metrics()
    metrics["pdc"]["object"]["conditions"]["clean"]["fp_rejection"] = 0.15
    result = assess_information_gate(
        metrics=metrics, response_comparisons=responses
    )
    assert result["passed"] is False
    assert "clean_fp_rejection_gte_0p20" in result["failed_checks"]
