from __future__ import annotations

from trkh.tools.run_b23_b9_guarded_convpass import (
    EPOCHS,
    EXPECTED_B9_CONFUSION,
    _promotion_gate,
)


def _metrics(*, c1_f1: float, macro: float, accuracy: float, tp: int, two: int):
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "class1": {"f1": c1_f1, "tp": tp},
        "restricted_fp_into_class1": 68,
        "transitions_into_class1": {"2->1": two},
    }


def test_b23_schedule_is_screen_aligned() -> None:
    assert EPOCHS == 6


def test_b9_confusion_contract_has_expected_support() -> None:
    assert sum(sum(row) for row in EXPECTED_B9_CONFUSION) == 2479
    assert EXPECTED_B9_CONFUSION[1][1] == 122


def test_promotion_requires_joint_quality_and_transition_gates() -> None:
    base = _metrics(c1_f1=0.70, macro=0.876, accuracy=0.912, tp=122, two=41)
    passing = _metrics(c1_f1=0.706, macro=0.875, accuracy=0.909, tp=120, two=40)
    failing = _metrics(c1_f1=0.706, macro=0.875, accuracy=0.909, tp=120, two=42)
    assert _promotion_gate(base, passing)["passed"] is True
    assert _promotion_gate(base, failing)["passed"] is False

