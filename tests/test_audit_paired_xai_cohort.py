import math

import pytest

from trkh.tools.audit_paired_xai_cohort import (
    _attention_title,
    _delta_means,
    _grad_rollout_title,
    _known_attention_source,
    _validate_case_alignment,
    flatten_xai_case,
)


def _case(sample_index: int, foreground: float, drop: float) -> dict:
    return {
        "sample_index": sample_index,
        "target_index": 1,
        "prediction_index": 1,
        "confidence": 0.7,
        "margin": 0.2,
        "correct": 1,
        "review_flags": ["flag"],
        "viz": {
            "attention_source": "feature_map_fallback",
            "grad_rollout_provenance": {
                "selected_layer_count": 8,
                "gradient_layer_count": 6,
                "fallback_layer_count": 2,
                "missing_gradient_layer_count": 2,
                "zero_weight_fallback_layer_count": 0,
            },
            "heatmap_focus": {
                "gradcam": {
                    "foreground_mass": foreground,
                    "background_mass": 1.0 - foreground,
                    "border_mass": 0.1,
                    "entropy": 0.8,
                }
            },
        },
        "robustness": {
            "object_desaturate": {
                "original_prediction_drop": drop,
                "target_probability_drop": drop,
            }
        },
    }


def test_flatten_xai_case_preserves_provenance_and_missing_metrics() -> None:
    row = flatten_xai_case(_case(7, 0.9, 0.12))

    assert row["sample_index"] == 7
    assert row["attention_source"] == "feature_map_fallback"
    assert row["gradcam_foreground_mass"] == pytest.approx(0.9)
    assert row["object_desaturate_original_prediction_drop"] == pytest.approx(0.12)
    assert math.isnan(row["rollout_foreground_mass"])
    assert row["grad_rollout_gradient_layer_count"] == pytest.approx(6)
    assert row["grad_rollout_fallback_layer_count"] == pytest.approx(2)


def test_grad_rollout_title_discloses_fallback() -> None:
    cases = {"7": _case(7, 0.9, 0.12)}
    assert _grad_rollout_title("candidate", cases, ["7"]) == (
        "candidate rollout fallback"
    )


def test_attention_title_discloses_feature_map_fallback() -> None:
    fallback = {"7": _case(7, 0.9, 0.12)}
    native_case = _case(8, 0.9, 0.12)
    native_case["viz"]["attention_source"] = (
        "forward_features.return_attention.blocks[7].mhsa_probability"
    )
    native = {"8": native_case}
    mixed = {**fallback, **native}

    assert _attention_title("keeper", fallback, ["7"]) == (
        "keeper feature-map fallback"
    )
    assert _attention_title("candidate", native, ["8"]) == (
        "candidate native attention"
    )
    assert _attention_title("pair", mixed, ["7", "8"]) == "pair attention (mixed)"


def test_delta_means_is_right_minus_left_and_ignores_nonfinite() -> None:
    rows = [
        {"left_confidence": 0.4, "right_confidence": 0.6},
        {"left_confidence": float("nan"), "right_confidence": 0.8},
    ]

    deltas = _delta_means(rows, left_prefix="left", right_prefix="right")

    assert deltas["confidence"] == pytest.approx(0.2)


def test_known_attention_source_accepts_full_grid_return_attention_provenance() -> None:
    assert _known_attention_source("forward_features.return_attention.blocks[7]")
    assert _known_attention_source(
        "forward_features.return_attention.late_member_blocks[7]"
    )
    assert _known_attention_source(
        "forward_features.return_attention.blocks[7].vca_effective_positive"
    )
    assert _known_attention_source(
        "forward_features.return_attention.blocks[3].mhsa_probability"
    )
    assert _known_attention_source("feature_map_fallback")
    assert not _known_attention_source("")
    assert not _known_attention_source("forward_features.return_attention.blocks[bad]")
    assert not _known_attention_source("unknown_hook")


def test_paired_audit_rejects_xai_target_that_differs_from_cohort() -> None:
    flattened = flatten_xai_case(_case(7, 0.9, 0.12))

    with pytest.raises(ValueError, match="XAI/cohort target mismatch"):
        _validate_case_alignment(
            key="7",
            cohort_row={
                "target_index": "0",
                "keeper_prediction_index": "1",
                "candidate_prediction_index": "1",
            },
            left=flattened,
            right=flattened,
            left_prefix="keeper",
            right_prefix="candidate",
        )
