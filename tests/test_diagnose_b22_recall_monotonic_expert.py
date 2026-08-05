from __future__ import annotations

import copy

import pytest
import torch

from trkh.models.recall_monotonic_expert_b22 import (
    BIAS_ONLY,
    FEATURE_LOGITS,
    LOGITS_ONLY,
)
from trkh.tools.diagnose_b22_recall_monotonic_expert import (
    evaluate_diagnostic_gate,
    focus_log_odds,
)


def _metrics(*, accuracy: float, macro: float, f1: float, recall: float, fp: int, two: int):
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "class1": {"f1": f1, "recall": recall, "tp": 66, "fp": fp},
        "transitions_into_class1": {"2->1": two},
    }


def _arm(metrics, parameters: int):
    return {
        "held_metrics": metrics,
        "parameter_count": parameters,
        "finite": True,
        "existing_focus_predictions_preserved": True,
    }


def _passing_payload():
    spatial = _metrics(
        accuracy=0.8804168952276468,
        macro=0.8467748826448902,
        f1=0.6740331491712708,
        recall=0.6354166666666666,
        fp=24,
        two=17,
    )
    arms = {
        BIAS_ONLY: _arm(
            _metrics(accuracy=0.879, macro=0.846, f1=0.686, recall=0.66, fp=27, two=18),
            1,
        ),
        LOGITS_ONLY: _arm(
            _metrics(accuracy=0.880, macro=0.847, f1=0.689, recall=0.67, fp=28, two=19),
            6,
        ),
        FEATURE_LOGITS: _arm(
            _metrics(accuracy=0.881, macro=0.850, f1=0.695, recall=0.69, fp=29, two=20),
            390,
        ),
    }
    return arms, spatial


def test_focus_log_odds_matches_binary_log_probability_ratio() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, -1.0, 0.5]])
    observed = focus_log_odds(logits)
    expected = logits[:, 1] - torch.logsumexp(logits[:, [0, 2, 3, 4]], dim=1)
    assert torch.equal(observed, expected)


def test_gate_passes_only_feature_specific_recall_safe_gain() -> None:
    arms, spatial = _passing_payload()
    result = evaluate_diagnostic_gate(arms, spatial_metrics=spatial)
    assert result["passed"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "mutation",
    ("no_feature_gain", "recall_loss", "fp_over_budget", "wrong_parameters"),
)
def test_gate_fails_closed(mutation: str) -> None:
    arms, spatial = _passing_payload()
    arms = copy.deepcopy(arms)
    if mutation == "no_feature_gain":
        arms[FEATURE_LOGITS]["held_metrics"]["class1"]["f1"] = 0.691
    elif mutation == "recall_loss":
        arms[FEATURE_LOGITS]["held_metrics"]["class1"]["recall"] = 0.63
    elif mutation == "fp_over_budget":
        arms[FEATURE_LOGITS]["held_metrics"]["class1"]["fp"] = 30
    else:
        arms[FEATURE_LOGITS]["parameter_count"] = 391
    result = evaluate_diagnostic_gate(arms, spatial_metrics=spatial)
    assert result["passed"] is False

