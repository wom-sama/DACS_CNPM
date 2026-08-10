from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trkh.models.starnet_s2_tokenizer import StarNetS2Tokenizer
from trkh.tools.audit_moga_tokenizer_readiness import (
    BENCHMARK_REPEATS,
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
)
from trkh.tools.audit_starnet_s2_tokenizer_readiness import (
    _architecture_summary,
    assess_starnet_readiness,
    parse_args,
    summarize_starnet_gradients,
)


class _PatchProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(128, 8, kernel_size=1)


class _GradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = StarNetS2Tokenizer()
        self.patch_embed = _PatchProjection()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed.proj(self.stem(x))


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


def test_gradient_summary_requires_every_star_family_and_weight() -> None:
    model = _GradientModel()
    output = model(torch.randn(2, 3, 64, 64))
    output.square().mean().backward()

    summary = summarize_starnet_gradients(model)
    assert set(summary) == {
        "stem_embedding",
        "stage_downsampling",
        "first_depthwise",
        "f1_projection",
        "f2_projection",
        "g_projection",
        "second_depthwise",
        "patch_projection",
    }
    assert all(value["passed"] for value in summary.values())
    model.stem.stages[2].blocks[5].f2.conv.weight.grad = None
    assert not summarize_starnet_gradients(model)["f2_projection"]["passed"]


def test_architecture_summary_matches_official_s2_stages() -> None:
    summary = _architecture_summary(StarNetS2Tokenizer())

    assert summary["stage_depths"] == [1, 2, 6]
    assert summary["stage_channels"] == [32, 64, 128]
    assert summary["block_count"] == 9
    assert summary["all_official_s2_settings_locked"]
    assert summary["actual_official_initialization_valid"]
    assert all(block["activation"] == "ReLU6" for block in summary["blocks"])


def test_gate_rejects_failed_check_vram_or_runtime() -> None:
    passed = assess_starnet_readiness(
        checks={"functional": True, "train_only": True},
        peak_vram_gib=7.0,
        max_peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=1.5,
        max_runtime_ratio=MAX_RUNTIME_RATIO,
    )
    assert passed["smoke_permission"]
    assert not passed["full_train_permission"]

    failed = assess_starnet_readiness(
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
