from __future__ import annotations

import numpy as np
import pytest
import torch

from trkh.tools.probe_locality_self_attention import (
    parse_layer_numbers,
    parse_variant_spec,
    parse_variant_specs,
    patch_attention_statistics,
    summarize_prediction_changes,
    transform_patch_attention,
)


def test_parse_variants_keeps_identity_first_and_validates_ranges() -> None:
    parsed = parse_variant_spec("sharp=1.25,0.5")
    assert parsed == {
        "name": "sharp",
        "temperature_multiplier": 1.25,
        "self_suppression": 0.5,
    }
    variants = parse_variant_specs(["sharp=1.25,0.5"])
    assert [item["name"] for item in variants] == ["baseline", "sharp"]
    with pytest.raises(ValueError):
        parse_variant_spec("bad=0,0")
    with pytest.raises(ValueError):
        parse_variant_spec("bad=1,1.1")


def test_parse_layers_rejects_out_of_range_values() -> None:
    assert parse_layer_numbers("4,2,2", block_count=8) == [2, 4]
    with pytest.raises(ValueError):
        parse_layer_numbers("0,2", block_count=8)


def test_identity_transform_is_exact() -> None:
    attention = torch.softmax(torch.randn(2, 3, 7, 7), dim=-1)
    transformed = transform_patch_attention(
        attention,
        prefix_count=2,
        temperature_multiplier=1.0,
        self_suppression=0.0,
    )
    assert transformed.data_ptr() == attention.data_ptr()
    torch.testing.assert_close(transformed, attention, rtol=0.0, atol=0.0)


def test_patch_transform_preserves_prefix_and_patch_mass() -> None:
    attention = torch.softmax(torch.randn(2, 3, 8, 8), dim=-1)
    transformed = transform_patch_attention(
        attention,
        prefix_count=2,
        temperature_multiplier=1.4,
        self_suppression=0.75,
    )
    torch.testing.assert_close(transformed[:, :, :2], attention[:, :, :2])
    torch.testing.assert_close(transformed[:, :, 2:, :2], attention[:, :, 2:, :2])
    torch.testing.assert_close(
        transformed[:, :, 2:, 2:].sum(dim=-1),
        attention[:, :, 2:, 2:].sum(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        transformed.sum(dim=-1),
        attention.sum(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_sharpening_lowers_entropy_and_diagonal_suppression_lowers_self_mass() -> None:
    attention = torch.softmax(torch.randn(4, 2, 9, 9), dim=-1)
    before = patch_attention_statistics(attention, prefix_count=1)
    sharpened = transform_patch_attention(
        attention,
        prefix_count=1,
        temperature_multiplier=1.5,
        self_suppression=0.0,
    )
    suppressed = transform_patch_attention(
        attention,
        prefix_count=1,
        temperature_multiplier=1.0,
        self_suppression=1.0,
    )
    sharp_stats = patch_attention_statistics(sharpened, prefix_count=1)
    suppressed_stats = patch_attention_statistics(suppressed, prefix_count=1)
    assert torch.all(sharp_stats["normalized_entropy"] < before["normalized_entropy"])
    assert torch.all(suppressed_stats["conditional_self_fraction"] < 1e-7)
    assert torch.all(
        suppressed_stats["conditional_self_fraction"]
        < before["conditional_self_fraction"]
    )


def test_prediction_change_summary_tracks_class1_risk() -> None:
    targets = np.asarray([1, 1, 0, 2, 4, 3])
    baseline = np.asarray([0, 1, 1, 2, 0, 3])
    variant = np.asarray([1, 0, 0, 1, 0, 3])
    summary = summarize_prediction_changes(
        targets=targets,
        baseline_predictions=baseline,
        variant_predictions=variant,
    )
    assert summary == {
        "changed": 4,
        "corrections": 2,
        "harms": 2,
        "neutral": 0,
        "class1_fn_rescues": 1,
        "class1_correct_breaks": 1,
        "class1_fp_removed": 1,
        "class1_fp_created": 1,
    }
