from __future__ import annotations

from trkh.tools.audit_learnable_gabor_texture_post_smoke import (
    CONDITIONS,
    assess_robustness,
)


def _condition(macro: float, tp: int):
    confusion = [[0] * 5 for _ in range(5)]
    confusion[1][1] = tp
    return {"macro_f1": macro, "confusion_matrix": confusion}


def test_gabor_robustness_requires_three_wins_and_tp_retention() -> None:
    metrics = {}
    for index, condition in enumerate(CONDITIONS):
        metrics[condition] = {
            "control": _condition(0.88, 117),
            "candidate": _condition(0.881 if index < 3 else 0.879, 116),
        }
    result = assess_robustness(metrics)
    assert result["passed"] is True
    assert result["macro_f1_win_count"] == 3

    metrics["bright"]["candidate"] = _condition(0.879, 115)
    failed = assess_robustness(metrics)
    assert failed["passed"] is False
    assert "class1_tp_loss_at_most_one_each_condition" in failed["failed_checks"]
