from __future__ import annotations

import copy

import numpy as np
import torch

from trkh.tools.screen_b37_dinov3_cls_patch_head_fold import (
    _gate,
    _sattolo_indices,
    _split_official_features,
)


def test_official_features_use_cls_and_patch_mean_but_not_registers() -> None:
    tokens = torch.zeros(2, 261, 384)
    tokens[:, 0] = 1.0
    tokens[:, 1:5] = 100.0
    tokens[:, 5:] = 3.0
    cls, patch, official = _split_official_features(tokens)
    assert torch.equal(cls, torch.ones_like(cls))
    assert torch.equal(patch, torch.full_like(patch, 3.0))
    assert torch.equal(official[:, :384], cls)
    assert torch.equal(official[:, 384:], patch)
    assert not bool((official == 100.0).any())


def test_sattolo_control_is_reproducible_and_has_no_fixed_points() -> None:
    first = _sattolo_indices(101, 42)
    second = _sattolo_indices(101, 42)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(101))
    assert not bool(np.any(first == np.arange(101)))


def _condition(
    *,
    accuracy: float,
    macro: float,
    class1: float,
    pair: float,
    tp: int,
    fp: int,
    zero_to_one: int,
) -> dict[str, object]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    matrix[0][1] = zero_to_one
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "class1_f1": class1,
        "mean_pair_auroc": pair,
        "class1_tp": tp,
        "restricted_fp": fp,
        "confusion_matrix": matrix,
    }


def _passing_metrics() -> dict[str, dict[str, dict[str, object]]]:
    raw = {
        "clean": _condition(
            accuracy=0.90, macro=0.85, class1=0.70, pair=0.95, tp=70, fp=20, zero_to_one=10
        ),
        "lighting_dim": _condition(
            accuracy=0.85, macro=0.80, class1=0.60, pair=0.92, tp=60, fp=30, zero_to_one=20
        ),
        "lighting_bright": _condition(
            accuracy=0.80, macro=0.75, class1=0.50, pair=0.90, tp=50, fp=40, zero_to_one=30
        ),
    }
    patch = copy.deepcopy(raw)
    deranged = copy.deepcopy(raw)
    deranged["clean"]["class1_f1"] = 0.69
    candidate = copy.deepcopy(raw)
    candidate["clean"].update(
        accuracy=0.901,
        macro_f1=0.851,
        class1_f1=0.71,
        mean_pair_auroc=0.951,
        class1_tp=71,
        restricted_fp=19,
    )
    candidate["lighting_dim"]["class1_f1"] = 0.61
    candidate["lighting_bright"]["class1_f1"] = 0.51
    return {
        "native_primary": raw,
        "patch_readout": patch,
        "deranged_cls_patch": deranged,
        "official_cls_patch": candidate,
    }


def test_gate_requires_information_gain_and_safety() -> None:
    passing = _passing_metrics()
    result = _gate(passing)
    assert result["passed"] is True
    assert result["validation_permission"] is False
    assert result["test_permission"] is False

    failing = copy.deepcopy(passing)
    failing["official_cls_patch"]["clean"]["restricted_fp"] = 21
    failed = _gate(failing)
    assert failed["passed"] is False
    assert "clean_fp_nonincrease" in failed["failed"]
