from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trkh.models.moga_surface_tokenizer import MogaXTTokenizer
from trkh.tools.audit_moga_tokenizer_readiness import (
    BENCHMARK_REPEATS,
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
    assess_moga_tokenizer_readiness,
    parse_args,
    summarize_moga_gradients,
)


class _PatchProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(96, 8, kernel_size=1)


class _GradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = MogaXTTokenizer()
        self.patch_embed = _PatchProjection()


def test_protocol_defaults_are_locked() -> None:
    args = parse_args(["--output-dir", "out"])

    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.seed == 42
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert args.max_peak_vram_gib == MAX_PEAK_VRAM_GIB == 7.75
    assert args.max_runtime_ratio == MAX_RUNTIME_RATIO == 1.75
    assert args.device == "cuda"
    assert isinstance(args.checkpoint, Path)
    assert isinstance(args.source_launcher_args, Path)


def test_gradient_summary_requires_every_family() -> None:
    model = _GradientModel()
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()

    summary = summarize_moga_gradients(model)
    assert set(summary) == {
        "stage_embeddings",
        "low_order",
        "middle_order",
        "high_order",
        "gate",
        "spatial_projection",
        "spatial_decomposition",
        "channel_ffn",
        "channel_decomposition",
        "layer_scales",
        "patch_projection",
    }
    assert all(value["passed"] for value in summary.values())

    model.stem.blocks1[0].spatial.value.dw_conv_high.weight.grad = None
    assert not summarize_moga_gradients(model)["high_order"]["passed"]


def test_gate_rejects_failed_check_vram_or_runtime() -> None:
    passed = assess_moga_tokenizer_readiness(
        checks={"functional": True, "train_only": True},
        peak_vram_gib=7.0,
        max_peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=1.5,
        max_runtime_ratio=MAX_RUNTIME_RATIO,
    )
    assert passed["smoke_permission"]
    assert not passed["full_train_permission"]

    failed = assess_moga_tokenizer_readiness(
        checks={"functional": False, "train_only": True},
        peak_vram_gib=7.8,
        max_peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=1.8,
        max_runtime_ratio=MAX_RUNTIME_RATIO,
    )
    assert not failed["smoke_permission"]
    assert set(failed["failed_checks"]) == {
        "functional",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }
