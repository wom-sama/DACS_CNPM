from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from trkh.models.model import HybridConvStem
from trkh.tools.audit_push_pull_illumination_support_a0 import (
    BATCH_SIZE,
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_REPEATS,
    BENCHMARK_WARMUPS,
    FEATURE_NAMES,
    INHIBITION_STRENGTH,
    MECHANISM_NAMES,
    OBJECT_FEATURE_INDICES,
    PUSH_SCALE,
    FrozenFirstBlockTrace,
    PushPullConv2d,
    _finite_difference_error,
    _mechanism_summary,
    _replace_first_conv,
    _structure_audit,
    activation_descriptors,
    assess_information_gate,
    independent_push_pull_oracle,
    parse_args,
    pull_kernel_size,
)


def _source_conv() -> nn.Conv2d:
    torch.manual_seed(11)
    return nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1, bias=False)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.seed == 42
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert BENCHMARK_BATCH_SIZE == 32
    assert BENCHMARK_WARMUPS == 5
    assert PUSH_SCALE == 2.0
    assert INHIBITION_STRENGTH == 1.0
    assert len(FEATURE_NAMES) == 22
    assert len(OBJECT_FEATURE_INDICES) == 14
    assert len(MECHANISM_NAMES) == 10


def test_pull_kernel_size_matches_paper_formula() -> None:
    assert pull_kernel_size(3, 1.0) == 3
    assert pull_kernel_size(3, 1.5) == 5
    assert pull_kernel_size(3, 2.0) == 7


def test_h1_is_exact_native_identity_for_output_and_gradients() -> None:
    source = _source_conv().double()
    candidate = PushPullConv2d(source, scale=1.0, alpha=1.0).double()
    images = torch.randn(2, 3, 13, 15, dtype=torch.float64, requires_grad=True)
    identity = candidate(images)
    native = source(images)
    assert torch.equal(identity, native)

    identity_input, identity_weight = torch.autograd.grad(
        identity.square().mean(),
        (images, candidate.weight),
        retain_graph=True,
    )
    native_input, native_weight = torch.autograd.grad(
        native.square().mean(), (images, source.weight)
    )
    assert torch.equal(identity_input, native_input)
    assert torch.equal(identity_weight, native_weight)


def test_h2_matches_independent_oracle_and_is_nondegenerate() -> None:
    source = _source_conv().double()
    candidate = PushPullConv2d(source, scale=2.0, alpha=1.0).double()
    images = torch.randn(2, 3, 17, 19, dtype=torch.float64, requires_grad=True)
    observed = candidate(images)
    _, _, expected = independent_push_pull_oracle(
        images, candidate.weight, scale=2.0, alpha=1.0
    )
    native = source(images)
    assert torch.equal(observed, expected)
    assert float((observed - native).abs().max()) > 1e-4
    gradient = torch.autograd.grad(observed.square().mean(), candidate.weight)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)
    assert candidate.pull_size == 7
    assert candidate.pull_padding == (3, 3)


def test_finite_difference_error_is_within_lock() -> None:
    assert _finite_difference_error() <= 1e-4


def test_first_block_trace_preserves_shape_and_h1_block_identity() -> None:
    torch.manual_seed(17)
    stem = HybridConvStem(
        in_channels=3,
        stem_channels=4,
        embed_dim=16,
        pooling_mode="max",
    ).eval()
    source = stem.blocks[0]
    native = FrozenFirstBlockTrace(source, push_pull=False).eval()
    candidate = FrozenFirstBlockTrace(source, push_pull=True).eval()
    images = torch.randn(2, 3, 32, 40)
    with torch.inference_mode():
        native_pre, native_block, _, _ = native.forward_components(images)
        candidate_pre, candidate_block, push, pull = candidate.forward_components(
            images
        )
    assert native_pre.shape == candidate_pre.shape == torch.Size([2, 4, 32, 40])
    assert native_block.shape == candidate_block.shape == torch.Size([2, 4, 16, 20])
    assert push is not None and pull is not None
    assert push.shape == pull.shape == candidate_pre.shape
    assert all(not parameter.requires_grad for parameter in candidate.parameters())


def test_activation_descriptors_include_signed_object_and_context_stats() -> None:
    activation = torch.zeros(2, 3, 16, 16)
    activation[:, :, 4:12, 4:12] = 1.0
    activation[:, :, 6:10, 6:10] = -2.0
    bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(2, 1)
    descriptors = activation_descriptors(activation, bboxes)
    assert descriptors.shape == (2, 11)
    assert torch.isfinite(descriptors).all()
    assert torch.all(descriptors[:, 1] > descriptors[:, 7])
    assert torch.all(descriptors[:, 2] > descriptors[:, 8])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = HybridConvStem(
            in_channels=3,
            stem_channels=32,
            embed_dim=256,
            pooling_mode="max",
        )
        self.head = nn.Linear(256, 5)


def test_first_conv_replacement_keeps_parameter_count_and_source_untouched() -> None:
    source = _DummyModel().eval()
    source_weight = source.stem.blocks[0].block.conv.weight.detach().clone()
    candidate = _replace_first_conv(source, scale=2.0)
    assert isinstance(candidate.stem.blocks[0].block.conv, PushPullConv2d)
    assert isinstance(source.stem.blocks[0].block.conv, nn.Conv2d)
    assert torch.equal(source.stem.blocks[0].block.conv.weight, source_weight)
    assert sum(value.numel() for value in source.parameters()) == sum(
        value.numel() for value in candidate.parameters()
    )


def test_structure_audit_inspects_replaced_operator_without_revalidating_as_conv() -> None:
    source = _DummyModel().eval()
    first_block, identity, candidate, summary = _structure_audit(source)
    assert first_block is source.stem.blocks[0]
    assert isinstance(identity.stem.blocks[0].block.conv, PushPullConv2d)
    assert isinstance(candidate.stem.blocks[0].block.conv, PushPullConv2d)
    assert summary["parameter_counts_equal"] is True
    assert summary["candidate_pull_size"] == 7


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
            auroc=0.62,
            tp=0.98 if condition == "clean" else 0.95,
            fp=0.10 if condition == "clean" else 0.12,
            tp_median=0.64,
            fp_median=0.46,
        )
        for condition in conditions
    }
    candidate_object = {
        condition: _condition_metrics(
            auroc=0.68,
            tp=0.98 if condition == "clean" else 0.95,
            fp=0.24 if condition == "clean" else 0.18,
            tp_median=0.70,
            fp_median=0.40,
        )
        for condition in conditions
    }
    candidate_context = {
        condition: _condition_metrics(
            auroc=0.69,
            tp=0.98,
            fp=0.23,
            tp_median=0.70,
            fp_median=0.40,
        )
        for condition in conditions
    }
    return {
        "native_identity_h1": {"object": {"conditions": native_object}},
        "push_pull_h2": {
            "object": {"conditions": candidate_object},
            "context": {"conditions": candidate_context},
        },
    }


def _passing_mechanism() -> dict[str, object]:
    conditions = {
        name: {
            "both_cohorts_object_gt_outside": True,
            "fp_minus_tp_pull_ratio": 0.02,
        }
        for name in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }
    return {
        "conditions": conditions,
        "shifted_object_gt_outside_count": 3,
        "shifted_fp_inhibition_gap_gte_0p01_count": 3,
    }


def test_information_gate_accepts_support_safe_gain_and_rejects_tp_loss() -> None:
    result = assess_information_gate(
        metrics=_passing_metrics(), mechanism_summary=_passing_mechanism()
    )
    assert result["passed"] is True
    assert result["failed_checks"] == []

    metrics = _passing_metrics()
    metrics["push_pull_h2"]["object"]["conditions"]["lighting_dim"][
        "tp_retention"
    ] = 0.88
    result = assess_information_gate(
        metrics=metrics, mechanism_summary=_passing_mechanism()
    )
    assert result["passed"] is False
    assert "all_shifted_tp_retention_gte_0p93" in result["failed_checks"]


def test_mechanism_summary_detects_fp_specific_inhibition() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    matrix = np.zeros((4, len(MECHANISM_NAMES)), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(MECHANISM_NAMES)}
    matrix[:, lookup["inhibited_object_rms"]] = [2.0, 2.2, 1.8, 2.0]
    matrix[:, lookup["inhibited_outside_rms"]] = [1.0, 1.1, 0.9, 1.0]
    matrix[:, lookup["object_pull_to_push_ratio"]] = [0.20, 0.22, 0.35, 0.37]
    mechanism = {
        condition: matrix.copy()
        for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast")
    }
    summary = _mechanism_summary(mechanism, labels)
    assert summary["conditions"]["clean"]["both_cohorts_object_gt_outside"]
    assert summary["conditions"]["clean"]["fp_minus_tp_pull_ratio"] > 0.01
    assert summary["shifted_object_gt_outside_count"] == 3
    assert summary["shifted_fp_inhibition_gap_gte_0p01_count"] == 3
