from __future__ import annotations

import torch

from trkh.models.learnable_gabor_texture import LearnableGaborTextureEncoder
from trkh.tools.audit_active_gabor_semantic_fusion_readiness import (
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
    _mean_off_diagonal_cosine,
    assess_active_readiness,
    parse_args,
    summarize_active_gradients,
)


def test_mean_off_diagonal_cosine_distinguishes_diversity() -> None:
    identical = torch.ones(1, 3, 4)
    orthogonal = torch.eye(3).unsqueeze(0)

    assert _mean_off_diagonal_cosine(identical) == 1.0
    assert abs(_mean_off_diagonal_cosine(orthogonal)) < 1e-7


def test_active_gradient_summary_requires_every_family() -> None:
    torch.manual_seed(42)
    module = LearnableGaborTextureEncoder(embed_dim=64)
    image = torch.randn(2, 3, 64, 64)
    bbox = torch.tensor(
        [[0.5, 0.5, 0.6, 0.6], [0.4, 0.6, 0.5, 0.4]],
        dtype=torch.float32,
    )
    module(image, bbox).square().mean().backward()

    summary = summarize_active_gradients(module)

    assert summary
    assert all(value["passed"] for value in summary.values())
    assert all(value["l2_norm"] > 0.0 for value in summary.values())


def test_active_readiness_is_fail_closed() -> None:
    passed = assess_active_readiness(
        checks={"functional": True, "micro": True},
        peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=MAX_RUNTIME_RATIO,
    )
    failed = assess_active_readiness(
        checks={"functional": True, "micro": False},
        peak_vram_gib=MAX_PEAK_VRAM_GIB + 0.01,
        runtime_ratio=MAX_RUNTIME_RATIO + 0.01,
    )

    assert passed["smoke_permission"] is True
    assert passed["test_permission"] is False
    assert failed["smoke_permission"] is False
    assert set(failed["failed_checks"]) == {
        "micro",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }


def test_active_readiness_cli_defaults_are_locked() -> None:
    args = parse_args(["--output-dir", "runs/audit"])

    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.seed == 42
    assert args.benchmark_repeats == 3
