from __future__ import annotations

from trkh.tools.audit_active_gabor_semantic_fusion_post_smoke import (
    CONDITIONS,
    assess_robustness,
)


def _condition_metrics(*, wins: int, tp_delta: int):
    payload = {}
    for index, condition in enumerate(CONDITIONS):
        control_confusion = [[0] * 5 for _ in range(5)]
        candidate_confusion = [[0] * 5 for _ in range(5)]
        control_confusion[1][1] = 120
        candidate_confusion[1][1] = 120 + tp_delta
        payload[condition] = {
            "control": {"macro_f1": 0.80, "confusion_matrix": control_confusion},
            "candidate": {
                "macro_f1": 0.81 if index < wins else 0.79,
                "confusion_matrix": candidate_confusion,
            },
        }
    return payload


def test_active_gabor_robustness_accepts_three_wins_and_two_tp_losses() -> None:
    result = assess_robustness(_condition_metrics(wins=3, tp_delta=-2))
    assert result["passed"] is True
    assert result["macro_f1_win_count"] == 3


def test_active_gabor_robustness_rejects_insufficient_wins_or_tp_retention() -> None:
    insufficient_wins = assess_robustness(_condition_metrics(wins=2, tp_delta=0))
    assert insufficient_wins["passed"] is False
    assert "macro_f1_wins_at_least_three_of_five" in insufficient_wins["failed_checks"]

    insufficient_tp = assess_robustness(_condition_metrics(wins=5, tp_delta=-3))
    assert insufficient_tp["passed"] is False
    assert (
        "class1_tp_loss_at_most_two_each_condition"
        in insufficient_tp["failed_checks"]
    )
