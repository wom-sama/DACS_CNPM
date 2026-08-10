from __future__ import annotations

import pytest

from trkh.tools.audit_visual_contrast_trace import audit_trace_payload


def _sample(class_id: int) -> dict[str, object]:
    return {
        "class_id": class_id,
        "visual_contrast_attention_layers": [1, 2],
        "visual_contrast_lambda_stage1": [0.2, 0.4],
        "visual_contrast_lambda_stage2": [0.3, 0.5],
        "visual_contrast_stage1_positive_mass": [0.7, 0.52],
        "visual_contrast_stage1_negative_mass": [0.3, 0.48],
        "visual_contrast_stage2_positive_mass": [0.51, 0.5],
        "visual_contrast_stage2_negative_mass": [0.49, 0.5],
        "visual_contrast_stage1_contrast_norm": [2.0, 1.0],
        "visual_contrast_stage2_contrast_norm": [1.5, 0.8],
    }


def test_trace_audit_records_nonzero_contrast_and_mass_balance() -> None:
    summary = audit_trace_payload(
        {"split_used": "train", "samples": [_sample(0), _sample(1)]}
    )
    assert summary["checks"]["stage1_contrast_nonzero"]
    assert summary["checks"]["stage2_contrast_nonzero"]
    assert summary["diagnostics"]["stage1_near_balanced_fraction"] == 0.5
    assert summary["diagnostics"]["stage2_near_balanced_fraction"] == 1.0
    assert summary["layer_metrics"][0]["stage1_mass_gap_mean"] == pytest.approx(0.4)


def test_trace_audit_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="test-split"):
        audit_trace_payload({"split_used": "test", "samples": [_sample(0)]})
