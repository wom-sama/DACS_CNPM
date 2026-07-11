from pathlib import Path

import pytest

from trkh.tools.audit_deep_abstention_risk_coverage import (
    _reject_test_input,
    analyze_risk_coverage,
)


def _row(target: int, prediction: int, abstention: float):
    return {
        "target_index": target,
        "prediction_index": prediction,
        "abstention_probability": abstention,
    }


def test_risk_coverage_rewards_abstention_on_errors() -> None:
    rows = [
        _row(0, 0, 0.01),
        _row(1, 1, 0.02),
        _row(2, 2, 0.03),
        _row(1, 2, 0.90),
    ]
    summary, curve, ranked = analyze_risk_coverage(
        rows,
        focus_class=1,
        coverages=(1.0, 0.75),
    )

    assert summary["error_detection"]["auroc"] == 1.0
    assert summary["error_detection"]["top_10_percent_error_capture"] == 1.0
    assert summary["coverage_checkpoints"][1]["risk"] == 0.0
    assert curve[-1]["risk"] == 0.25
    assert ranked[0]["error"] == 1


def test_risk_coverage_rejects_test_inputs() -> None:
    with pytest.raises(ValueError, match="test-derived"):
        _reject_test_input(Path("runs/model_test_final/predictions.csv"), "val")
    with pytest.raises(ValueError, match="validation-only"):
        _reject_test_input(Path("runs/model_val/predictions.csv"), "test")
