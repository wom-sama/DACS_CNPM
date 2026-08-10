from __future__ import annotations

import numpy as np
import torch
from torch import nn

from trkh.tools.audit_ddhts_positive_evidence_readiness import (
    DESCRIPTOR_DIM,
    _iuwt_contrast_views,
    _locked_args_exact,
    _normalized_code_histogram,
    apply_positive_evidence_veto,
    assess_ddhts_gate,
    build_ddhts_descriptor,
    parse_args,
)


class _TinyStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GELU(), nn.AvgPool2d(2)),
            nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.GELU(), nn.AvgPool2d(2)),
            nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.GELU(), nn.AvgPool2d(2)),
        )


def test_iuwt_constant_input_maps_to_half_contrast() -> None:
    rgb = torch.full((2, 3, 24, 24), 0.4)
    views = _iuwt_contrast_views(rgb)
    assert len(views) == 3
    for view in views:
        assert view.shape == (2, 1, 24, 24)
        assert torch.isfinite(view).all()
        assert torch.allclose(view, torch.full_like(view, 0.5), atol=1e-5)


def test_code_histogram_is_normalized() -> None:
    codes = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    histogram = _normalized_code_histogram(codes, bins=4)
    assert histogram.shape == (1, 4)
    assert torch.equal(histogram, torch.full((1, 4), 0.25))
    assert torch.equal(histogram.sum(dim=1), torch.ones(1))


def test_ddhts_descriptor_has_locked_components_and_is_deterministic() -> None:
    stem = _TinyStem().eval()
    images = torch.linspace(-1.0, 1.0, steps=2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    first, first_diagnostics, _ = build_ddhts_descriptor(
        stem,
        images,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        preserved_channels=4,
        grid_size=2,
        amp=False,
    )
    second, second_diagnostics, _ = build_ddhts_descriptor(
        stem,
        images,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        preserved_channels=4,
        grid_size=2,
        amp=False,
    )
    assert first.shape == (2, DESCRIPTOR_DIM)
    assert first_diagnostics["inter_dim"] == 24
    assert first_diagnostics["intra_dim"] == 16
    assert first_diagnostics["segment_count"] == 7
    assert first_diagnostics["segment_sum_max_abs_error"] <= 1e-7
    assert first_diagnostics == second_diagnostics
    assert torch.equal(first, second)


def test_positive_evidence_veto_only_changes_raw_class1_below_threshold() -> None:
    raw = np.asarray(
        [
            [0.10, 0.60, 0.20, 0.05, 0.05],
            [0.60, 0.10, 0.20, 0.05, 0.05],
            [0.20, 0.50, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    candidate, veto = apply_positive_evidence_veto(
        raw, np.asarray([-0.5, -1.0, 0.5]), threshold=0.0
    )
    assert veto.tolist() == [True, False, False]
    assert candidate.argmax(axis=1).tolist() == [2, 0, 1]
    assert np.array_equal(candidate[1], raw[1])
    assert np.array_equal(candidate[2], raw[2])
    assert np.allclose(candidate.sum(axis=1), 1.0)


def _passing_comparison() -> dict[str, object]:
    return {
        "delta": {
            "macro_f1": 0.01,
            "class1_f1": 0.02,
            "class1_precision": 0.03,
            "class1_recall": 0.0,
        },
        "transitions": {
            "focus_true_positive_broken": 0,
            "focus_false_positive_created": 0,
            "restricted_focus_fp_reduction": 4,
            "corrections": 4,
            "harms": 0,
        },
    }


def test_gate_requires_zero_true_positive_breaks_in_every_condition() -> None:
    comparisons = {
        "clean": _passing_comparison(),
        "lighting_dim": _passing_comparison(),
        "lighting_bright": _passing_comparison(),
        "low_contrast": _passing_comparison(),
    }
    gate = assess_ddhts_gate(
        structural_checks={"structure": True},
        comparisons=comparisons,
        clean_direction_auc=0.70,
    )
    assert gate["stage_b_authorized"] is True

    comparisons["lighting_bright"]["transitions"]["focus_true_positive_broken"] = 1
    failed = assess_ddhts_gate(
        structural_checks={"structure": True},
        comparisons=comparisons,
        clean_direction_auc=0.70,
    )
    assert failed["stage_b_authorized"] is False
    assert "lighting_bright_breaks_zero_class1_tp" in failed["failed_checks"]


def test_default_cli_matches_locked_protocol() -> None:
    assert _locked_args_exact(parse_args([]))
