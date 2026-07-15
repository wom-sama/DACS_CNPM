from __future__ import annotations

import pytest
import torch

from trkh.models.model import create_model
from trkh.tools.audit_octave_conv_stem_readiness import (
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
    _candidate_config,
    _flatten_cosine,
    _normalized_total_variation,
    assess_octave_readiness,
    summarize_octave_gradients,
)


def _small_config() -> dict[str, object]:
    return {
        "model_type": "vit_registers",
        "image_size": 64,
        "patch_size": 16,
        "in_channels": 3,
        "use_cnn_stem": True,
        "stem_channels": 16,
        "stem_architecture": "conv_pool",
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 4,
        "num_registers": 1,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
        "multi_branch_fusion": False,
        "token_pruning": False,
    }


def test_octave_readiness_gate_requires_every_check_and_resource_limit() -> None:
    passed = assess_octave_readiness(
        checks={"mechanism": True, "no_test": True},
        peak_vram_gib=MAX_PEAK_VRAM_GIB,
        runtime_ratio=MAX_RUNTIME_RATIO,
    )
    failed = assess_octave_readiness(
        checks={"mechanism": False, "no_test": True},
        peak_vram_gib=MAX_PEAK_VRAM_GIB + 0.01,
        runtime_ratio=MAX_RUNTIME_RATIO + 0.01,
    )

    assert passed["smoke_permission"] is True
    assert passed["full_train_permission"] is False
    assert failed["smoke_permission"] is False
    assert set(failed["failed_checks"]) == {
        "mechanism",
        "peak_vram_within_budget",
        "runtime_ratio_within_budget",
    }


def test_octave_candidate_config_is_default_off_and_scoped() -> None:
    source = _small_config()
    candidate = _candidate_config(source)

    assert source["stem_architecture"] == "conv_pool"
    assert candidate["stem_architecture"] == "octave_conv"
    assert candidate["pretrained"] is False
    assert candidate["visual_contrast_attention"] is False


def test_octave_gradient_summary_covers_every_path() -> None:
    torch.manual_seed(42)
    model = create_model(num_classes=5, model_config=_candidate_config(_small_config()))
    logits = model(torch.randn(2, 3, 64, 64))
    logits.square().mean().backward()
    summary = summarize_octave_gradients(model)

    assert len(summary) == 9
    assert all(value["passed"] for value in summary.values())


def test_octave_cosine_and_total_variation_diagnostics_are_directional() -> None:
    high = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    same = high.clone()
    flat = torch.ones_like(high)

    assert _flatten_cosine(high, same)["maximum_absolute"] == pytest.approx(
        1.0, abs=1e-6
    )
    flat_tv = _normalized_total_variation(flat)
    high_tv = _normalized_total_variation(high)
    assert flat_tv.shape == (2,)
    assert high_tv.shape == (2,)
    assert torch.equal(flat_tv, torch.zeros_like(flat_tv))
    assert bool(torch.all(high_tv > 0.0))
