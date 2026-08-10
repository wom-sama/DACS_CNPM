from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trkh.tools.audit_visual_contrast_attention_readiness import (
    MAX_PEAK_VRAM_GIB,
    assess_visual_contrast_readiness,
    parse_args,
    summarize_vca_gradients,
)


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(2, 6)
        self.positive_embedding = nn.Parameter(torch.ones(1, 1, 2))
        self.negative_embedding = nn.Parameter(torch.ones(1, 1, 2))
        self.stage1_lambda_q1 = nn.Parameter(torch.ones(2))
        self.stage2_lambda_q1 = nn.Parameter(torch.ones(2))
        self.stage1_norm = nn.LayerNorm(2)
        self.stage2_norm = nn.LayerNorm(2)
        self.depthwise_value = nn.Conv2d(2, 2, 1, groups=2)
        self.proj = nn.Linear(2, 2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = _TinyAttention()


def test_protocol_defaults_are_locked() -> None:
    args = parse_args(["--output-dir", "out"])
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.seed == 42
    assert args.visual_contrast_tokens == 64
    assert args.max_peak_vram_gib == MAX_PEAK_VRAM_GIB == 7.75
    assert args.device == "cuda"
    assert isinstance(args.checkpoint, Path)


def test_gradient_summary_requires_every_family() -> None:
    model = _TinyModel()
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    summary = summarize_vca_gradients(model)
    assert set(summary) == {
        "qkv",
        "positive_embedding",
        "negative_embedding",
        "stage1_lambda",
        "stage2_lambda",
        "stage1_norm",
        "stage2_norm",
        "depthwise_value",
        "projection",
    }
    assert all(value["passed"] for value in summary.values())
    model.blocks[0].attn.negative_embedding.grad = None
    assert not summarize_vca_gradients(model)["negative_embedding"]["passed"]


def test_gate_rejects_any_failed_check_or_vram_overflow() -> None:
    passed = assess_visual_contrast_readiness(
        checks={"functional": True, "train_only": True},
        peak_vram_gib=7.0,
        max_peak_vram_gib=7.75,
    )
    assert passed["smoke_permission"]
    assert not passed["full_train_permission"]

    failed = assess_visual_contrast_readiness(
        checks={"functional": True, "train_only": False},
        peak_vram_gib=7.8,
        max_peak_vram_gib=7.75,
    )
    assert not failed["smoke_permission"]
    assert set(failed["failed_checks"]) == {
        "train_only",
        "peak_vram_within_budget",
    }
