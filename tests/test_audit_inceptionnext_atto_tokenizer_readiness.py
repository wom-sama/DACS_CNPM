from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trkh.models.inceptionnext_atto_tokenizer import InceptionNeXtAttoTokenizer
from trkh.tools.audit_inceptionnext_atto_tokenizer_readiness import (
    assess_inceptionnext_readiness,
    parse_args,
    summarize_inceptionnext_gradients,
)
from trkh.tools.audit_moga_tokenizer_readiness import (
    BENCHMARK_REPEATS,
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
)


class _PatchProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(160, 8, kernel_size=1)


class _GradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = InceptionNeXtAttoTokenizer()
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


def test_gradient_summary_requires_every_family() -> None:
    model = _GradientModel()
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()

    summary = summarize_inceptionnext_gradients(model)
    assert set(summary) == {
        "stem_embedding",
        "stage_downsampling",
        "square_branch",
        "horizontal_branch",
        "vertical_branch",
        "block_normalization",
        "mlp_input_projection",
        "mlp_output_projection",
        "layer_scales",
        "patch_projection",
    }
    assert all(value["passed"] for value in summary.values())
    model.stem.stages[0].blocks[0].token_mixer.dwconv_w.weight.grad = None
    assert not summarize_inceptionnext_gradients(model)["horizontal_branch"][
        "passed"
    ]


def test_gate_rejects_failed_check_vram_or_runtime() -> None:
    passed = assess_inceptionnext_readiness(
        checks={"functional": True, "train_only": True},
        peak_vram_gib=7.0,
        max_peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=1.5,
        max_runtime_ratio=MAX_RUNTIME_RATIO,
    )
    assert passed["smoke_permission"]
    assert not passed["full_train_permission"]

    failed = assess_inceptionnext_readiness(
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
