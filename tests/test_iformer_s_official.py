from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.models.iformer_s_official import (
    DESCRIPTOR_DIM,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_SOURCE_SHA256,
    PARAMETERS_1000,
    PARAMETERS_5,
    _safe_checkpoint,
    build_official_iformer_s,
    build_official_iformer_s_5class,
    copy_iformer_s_backbone_5class,
    load_official_checkpoint,
    load_official_iformer_s_1000,
    validate_official_source,
)


ROOT = Path(
    os.environ.get(
        "TRKH_IFORMER_OFFICIAL_ROOT",
        r"D:\DataAI\external_sources\official\iformer_iclr2025",
    )
)
SOURCE = ROOT / "models" / "iformer.py"
CHECKPOINT = Path(
    os.environ.get(
        "TRKH_IFORMER_CHECKPOINT",
        str(
            Path(__file__).resolve().parents[1]
            / "runs"
            / "pretrained_assets"
            / "iformer_s"
            / "iFormer_s.pth"
        ),
    )
)
ASSETS_PRESENT = SOURCE.is_file() and CHECKPOINT.is_file()


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned iFormer assets are absent")
def test_safe_strict_load_is_scoped_and_ignores_stale_args() -> None:
    source = validate_official_source(ROOT)
    assert source["source_sha256"] == OFFICIAL_SOURCE_SHA256
    safe_before = list(torch.serialization.get_safe_globals())
    artifact = load_official_checkpoint(CHECKPOINT)
    assert set(torch.serialization.get_safe_globals()) == set(safe_before)
    assert len(artifact["state"]) == 505

    allowlist = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        argparse.Namespace,
    ]
    with torch.serialization.safe_globals(allowlist):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["args"].model == "convnext_t1_v35"
    assert checkpoint["args"].layer_scale_init_value == pytest.approx(1e-6)

    torch.manual_seed(91)
    rng_before = torch.random.get_rng_state().clone()
    model = build_official_iformer_s(ROOT, CHECKPOINT)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS_1000
    wrapped = load_official_iformer_s_1000(ROOT, CHECKPOINT)
    assert wrapped.source_sha256 == OFFICIAL_SOURCE_SHA256
    assert wrapped.checkpoint_sha256 == OFFICIAL_CHECKPOINT_SHA256
    assert sum(p.numel() for p in wrapped.parameters()) == PARAMETERS_1000
    assert not any(key.endswith(".gamma") for key in wrapped.model.state_dict())


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned iFormer assets are absent")
def test_pooled_descriptor_reproduces_official_logits_exactly() -> None:
    wrapped = load_official_iformer_s_1000(ROOT, CHECKPOINT).eval()
    torch.manual_seed(20260805)
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        expected = wrapped.model(images)
        actual, descriptor = wrapped.forward_with_descriptor(images)
        direct_descriptor = wrapped(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(direct_descriptor, descriptor, rtol=0, atol=0)
    assert descriptor.shape == (2, DESCRIPTOR_DIM)
    assert torch.isfinite(descriptor).all()


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned iFormer assets are absent")
def test_five_class_copy_changes_only_linear_head() -> None:
    released_model = build_official_iformer_s(ROOT, CHECKPOINT)
    copied_model = copy_iformer_s_backbone_5class(released_model)
    released = load_official_iformer_s_1000(ROOT, CHECKPOINT).eval()
    transferred = build_official_iformer_s_5class(ROOT, CHECKPOINT).eval()
    assert sum(p.numel() for p in copied_model.parameters()) == PARAMETERS_5
    assert sum(p.numel() for p in transferred.parameters()) == PARAMETERS_5

    source_state = released.model.state_dict()
    target_state = transferred.model.state_dict()
    changed = {
        "classifier.classifier.l.weight",
        "classifier.classifier.l.bias",
    }
    assert set(source_state) == set(target_state)
    for key in source_state.keys() - changed:
        assert torch.equal(source_state[key], target_state[key]), key
    assert target_state["classifier.classifier.l.weight"].shape == (5, 320)
    assert target_state["classifier.classifier.l.bias"].shape == (5,)
    assert torch.count_nonzero(target_state["classifier.classifier.l.bias"]) == 0
    transferred_again = build_official_iformer_s_5class(ROOT, CHECKPOINT).eval()
    assert torch.equal(
        target_state["classifier.classifier.l.weight"],
        transferred_again.model.state_dict()["classifier.classifier.l.weight"],
    )

    images = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        expected = transferred.model(images)
        actual, descriptor = transferred.forward_with_descriptor(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 5)
    assert descriptor.shape == (1, 320)


def test_hash_guards_fail_before_loading_untrusted_content(tmp_path: Path) -> None:
    bad_source = tmp_path / "iformer.py"
    bad_source.write_text("def iFormer_s(): pass\n", encoding="utf-8")
    bad_checkpoint = tmp_path / "iFormer_s.pth"
    bad_checkpoint.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        load_official_iformer_s_1000(bad_source, bad_checkpoint)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _safe_checkpoint(bad_checkpoint)
