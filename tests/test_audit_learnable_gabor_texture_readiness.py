from __future__ import annotations

import torch

from trkh.models.learnable_gabor_texture import LearnableGaborTextureResidual
from trkh.tools.audit_learnable_gabor_texture_readiness import (
    BENCHMARK_REPEATS,
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
    assess_readiness,
    parse_args,
    summarize_gabor_gradients,
)


def test_gabor_readiness_parser_locks_train_only_defaults(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path)])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.benchmark_repeats == BENCHMARK_REPEATS
    assert "yolo_f" in str(args.data)
    assert "probe_v8_yolof" in str(args.checkpoint)


def test_gabor_readiness_gate_is_conjunctive() -> None:
    passed = assess_readiness(
        checks={"functional": True, "export": True},
        peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=MAX_RUNTIME_RATIO,
    )
    assert passed["smoke_permission"] is True
    assert passed["probe_permission"] is False
    assert passed["full_train_permission"] is False
    assert passed["test_permission"] is False

    failed = assess_readiness(
        checks={"functional": True, "export": False},
        peak_vram_gib=MAX_PEAK_VRAM_GIB + 0.01,
        runtime_ratio=MAX_RUNTIME_RATIO + 0.01,
    )
    assert failed["smoke_permission"] is False
    assert set(failed["failed_checks"]) == {
        "export",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }


def test_gabor_gradient_summary_requires_every_family() -> None:
    torch.manual_seed(42)
    module = LearnableGaborTextureResidual(embed_dim=64)
    with torch.no_grad():
        module.raw_gate.fill_(torch.atanh(torch.tensor(0.2)))
    image = torch.randn(2, 3, 64, 64)
    bbox = torch.tensor(
        [[0.5, 0.5, 0.6, 0.6], [0.4, 0.6, 0.3, 0.4]],
        dtype=torch.float32,
    )
    module(image, bbox).square().mean().backward()
    summary = summarize_gabor_gradients(module)

    assert summary
    assert all(value["passed"] for value in summary.values())
    module.raw_theta.grad = None
    failed = summarize_gabor_gradients(module)
    assert failed["gabor_parameters"]["passed"] is False
