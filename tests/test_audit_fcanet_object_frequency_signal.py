from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.models.model import BBoxSpatialPriorFusion
from trkh.tools.audit_fcanet_object_frequency_signal import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    CHANNELS,
    CONDITIONS,
    FIT_FOLDS,
    GRID_SIZE,
    MultiSpectralDCTLayer,
    ObjectAlignedDCTExtractor,
    _center_mask_support,
    _load_official_dct_class,
    align_object_feature_map,
    assess_signal_gate,
    bbox_geometry_descriptor,
    get_frequency_indices,
    parse_args,
    select_recall_constrained_threshold,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 7
    assert args.seed == 42
    assert len(CONDITIONS) == 4


def test_locked_frequency_indices_match_fcanet_top16_order() -> None:
    mapper_x, mapper_y = get_frequency_indices("top16")
    assert mapper_x == [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
    assert mapper_y == [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
    assert get_frequency_indices("top1") == ([0], [0])


def test_zero_frequency_dct_is_proportional_to_gap() -> None:
    torch.manual_seed(17)
    features = torch.randn(3, 4, 8, 8, dtype=torch.float64)
    layer = MultiSpectralDCTLayer(8, 8, [0], [0], 4).double()
    observed = layer(features)
    expected = features.mean(dim=(2, 3)) * 8.0
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_full_bbox_alignment_is_identity_and_object_dct_has_gradients() -> None:
    torch.manual_seed(19)
    small = torch.randn(2, 3, GRID_SIZE, GRID_SIZE)
    full_boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]]).repeat(2, 1)
    aligned = align_object_feature_map(small, full_boxes)
    assert torch.allclose(aligned, small, atol=2e-6, rtol=0.0)

    features = torch.randn(
        2, CHANNELS, GRID_SIZE, GRID_SIZE, requires_grad=True
    )
    boxes = torch.tensor(
        [[0.5, 0.5, 0.7, 0.8], [0.4, 0.6, 0.5, 0.4]]
    )
    descriptor = ObjectAlignedDCTExtractor("top16")(features, boxes)
    descriptor.square().mean().backward()
    assert descriptor.shape == (2, CHANNELS)
    assert torch.isfinite(descriptor).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0

    with torch.inference_mode():
        reference = ObjectAlignedDCTExtractor("top16")(features.detach(), boxes)
        low_precision = ObjectAlignedDCTExtractor("top16")(
            features.detach().to(dtype=torch.bfloat16),
            boxes,
        )
    assert low_precision.dtype == torch.float32
    assert float((reference - low_precision).abs().max()) <= 0.05


def test_center_support_and_bbox_geometry_match_project_control() -> None:
    boxes = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.3, 0.4]],
        dtype=torch.float32,
    )
    object_count, context_count = _center_mask_support(boxes, GRID_SIZE)
    assert torch.all(object_count > 0)
    assert torch.all(context_count > 0)
    control = BBoxSpatialPriorFusion(num_classes=5).extract_stats(boxes)
    observed = bbox_geometry_descriptor(boxes)
    assert observed.shape == (2, 18)
    assert torch.equal(observed, control)


def test_official_ast_loader_replays_real_source_when_available() -> None:
    path = Path(r"D:\DataAI\external_sources\official\fcanet-iccv2021\model\layer.py")
    if not path.is_file():
        pytest.skip("Official FcaNet source is not installed in this environment.")
    official_indices, official_layer = _load_official_dct_class(path)
    assert official_indices("top16") == get_frequency_indices("top16")
    mapper_x, mapper_y = get_frequency_indices("top16")
    mapper_x = [value * (GRID_SIZE // 7) for value in mapper_x]
    mapper_y = [value * (GRID_SIZE // 7) for value in mapper_y]
    local = MultiSpectralDCTLayer(
        GRID_SIZE, GRID_SIZE, mapper_x, mapper_y, CHANNELS
    )
    official = official_layer(
        GRID_SIZE, GRID_SIZE, mapper_x, mapper_y, CHANNELS
    )
    assert torch.equal(local.weight, official.weight)


def test_recall_constrained_threshold_uses_highest_valid_score() -> None:
    scores = np.asarray([0.95, 0.90, 0.80, 0.70, 0.60, 0.55])
    labels = np.asarray([1, 1, 1, 1, 1, 0])
    threshold = select_recall_constrained_threshold(
        scores, labels, min_tp_retention=0.80
    )
    assert threshold == 0.70


def _role_metric(
    auroc: float, *, tp: float = 0.92, fp: float = 0.25
) -> dict[str, object]:
    return {
        "auroc": auroc,
        "tp_retention": tp,
        "fp_rejection": fp,
        "fold_auroc": {str(fold): auroc for fold in FIT_FOLDS},
    }


def _passing_signal_inputs() -> tuple[dict[str, object], dict[str, object]]:
    conditions: dict[str, object] = {}
    for name, _, _ in CONDITIONS:
        candidate_auroc = 0.70 if name == "clean" else 0.69
        conditions[name] = {
            "full_gap": _role_metric(0.64),
            "object_gap": _role_metric(0.66, fp=0.18),
            "full_top16": _role_metric(0.68),
            "object_top16": _role_metric(candidate_auroc, fp=0.25),
            "bbox_geometry": _role_metric(0.63),
        }
    clean = conditions["clean"]
    clean["object_top16"]["fold_auroc"] = {
        "1": 0.70,
        "2": 0.69,
        "3": 0.68,
        "4": 0.67,
    }
    clean["object_gap"]["fold_auroc"] = {
        "1": 0.66,
        "2": 0.66,
        "3": 0.67,
        "4": 0.68,
    }
    telemetry = {
        "all_finite": True,
        "object_top16_gap_max_abs_delta": 0.1,
        "all_frequency_groups_nonzero_variance": True,
    }
    return {"conditions": conditions}, telemetry


def test_signal_gate_accepts_all_locked_gains_and_rejects_fp_failure() -> None:
    metrics, telemetry = _passing_signal_inputs()
    result = assess_signal_gate(metrics, telemetry)
    assert result["passed"] is True

    metrics["conditions"]["clean"]["object_top16"]["fp_rejection"] = 0.19
    result = assess_signal_gate(metrics, telemetry)
    assert result["passed"] is False
    assert result["checks"]["clean_fp_rejection_at_least_020"] is False
