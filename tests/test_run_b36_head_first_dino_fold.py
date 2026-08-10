from __future__ import annotations

from trkh.tools.run_b36_head_first_dino_fold import _gate


def _metrics(
    *,
    accuracy: float,
    macro: float,
    c1: float,
    tp: int,
    fp: int,
    zero_to_one: int,
    pair: float,
) -> dict:
    confusion = [[0] * 5 for _ in range(5)]
    confusion[0][1] = zero_to_one
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "class1_f1": c1,
        "class1_tp": tp,
        "restricted_fp": fp,
        "mean_pair_auroc": pair,
        "confusion_matrix": confusion,
    }


def _conditions(delta: float = 0.0) -> dict:
    return {
        "clean": _metrics(
            accuracy=0.94 + delta,
            macro=0.91 + delta,
            c1=0.77 + delta,
            tp=75 + int(delta > 0),
            fp=18 - int(delta > 0),
            zero_to_one=10 - int(delta > 0),
            pair=0.98 + delta,
        ),
        "lighting_dim": _metrics(
            accuracy=0.90,
            macro=0.87,
            c1=0.65 + delta,
            tp=60,
            fp=30,
            zero_to_one=15,
            pair=0.95,
        ),
        "lighting_bright": _metrics(
            accuracy=0.86,
            macro=0.81,
            c1=0.53 + delta,
            tp=55,
            fp=60,
            zero_to_one=40 - int(delta > 0),
            pair=0.93,
        ),
    }


def _invariance(delta: float = 0.0) -> dict:
    return {
        "lighting_dim": {"class1_median_cosine": 0.90 + delta},
        "lighting_bright": {"class1_median_cosine": 0.85 + delta},
    }


def test_gate_passes_only_for_clean_and_robust_head_first_gain() -> None:
    result = _gate(
        _conditions(),
        _conditions(0.012),
        _invariance(),
        _invariance(0.001),
    )

    assert result["passed"] is True
    assert result["next_permission"] == "confirm_head_first_on_fresh_component_fold"


def test_gate_rejects_clean_only_gain_without_robust_or_feature_preservation() -> None:
    control = _conditions()
    candidate = _conditions(0.012)
    candidate["lighting_dim"]["class1_f1"] = control["lighting_dim"]["class1_f1"]
    candidate["lighting_bright"]["class1_f1"] = control["lighting_bright"]["class1_f1"]
    result = _gate(control, candidate, _invariance(), _invariance(-0.001))

    assert result["passed"] is False
    assert result["checks"]["robust_mean_gain"] is False
    assert result["checks"]["bright_feature_preservation"] is False
