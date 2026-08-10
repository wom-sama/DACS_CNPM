from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from trkh.tools.balanced_bce_frozen_embedding_a0_engine import (
    BASE_LR,
    BASE_SEED,
    CLASS_COUNT,
    EPOCHS,
    REPEAT_SEED_OFFSET,
    ROLE_NAMES,
    balanced_bce_bias,
    build_epoch_orders,
    equation_oracles,
    initial_head_evidence,
    learning_rate,
    role_loss,
)
from trkh.tools.audit_balanced_bce_frozen_embedding_a0 import LOCK_PATH


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_equation_oracles_and_bias_direction() -> None:
    evidence = equation_oracles()
    assert evidence["passed"]
    assert all(evidence["checks"].values())

    prior = torch.tensor(
        [0.20, 0.06, 0.21, 0.28, 0.25], dtype=torch.float64
    )
    expected = torch.log(prior) - torch.log1p(-prior)
    assert torch.equal(balanced_bce_bias(prior), expected)

    logits = torch.tensor(
        [[-1.0, 0.4, 1.2, -0.2, 0.1]], dtype=torch.float64
    )
    target = torch.tensor([2], dtype=torch.long)
    one_hot = F.one_hot(target, num_classes=CLASS_COUNT).to(torch.float64)
    expected_candidate = CLASS_COUNT * F.binary_cross_entropy_with_logits(
        logits + expected, one_hot, reduction="mean"
    )
    expected_reversed = CLASS_COUNT * F.binary_cross_entropy_with_logits(
        logits - expected, one_hot, reduction="mean"
    )
    assert torch.equal(
        role_loss(
            "balanced_bce_candidate", logits, target, prior=prior
        ),
        expected_candidate,
    )
    assert torch.equal(
        role_loss(
            "reversed_prior_bce_control", logits, target, prior=prior
        ),
        expected_reversed,
    )


def test_initial_states_and_orders_match_machine_lock() -> None:
    lock = _lock()
    for fold in lock["outer_folds"]:
        fold_index = int(fold["outer_fold"])
        fit = np.arange(int(fold["fit_rows"]), dtype=np.int64)
        # Order hashes depend on the actual global fit indices and are covered
        # by the auditor test; this unit verifies deterministic stream reuse.
        first = build_epoch_orders(fit, seed=BASE_SEED + fold_index)
        second = build_epoch_orders(fit, seed=BASE_SEED + fold_index)
        repeat = build_epoch_orders(
            fit, seed=BASE_SEED + REPEAT_SEED_OFFSET + fold_index
        )
        assert len(first) == EPOCHS
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        assert any(not np.array_equal(a, b) for a, b in zip(first, repeat))
        assert initial_head_evidence(
            seed=BASE_SEED + fold_index
        ) == fold["primary_initial_head"]
        assert initial_head_evidence(
            seed=BASE_SEED + REPEAT_SEED_OFFSET + fold_index
        ) == fold["repeat_initial_head"]


def test_learning_rate_is_fixed_warmup_then_cosine() -> None:
    rates = [learning_rate(epoch) for epoch in range(EPOCHS)]
    assert rates[:5] == pytest.approx(
        [0.006, 0.012, 0.018, 0.024, BASE_LR], abs=1e-15
    )
    assert rates[5] == pytest.approx(BASE_LR, abs=1e-15)
    assert rates[-1] == pytest.approx(0.0, abs=1e-15)
    assert all(
        rates[index] >= rates[index + 1]
        for index in range(5, EPOCHS - 1)
    )
    with pytest.raises(ValueError):
        learning_rate(-1)
    with pytest.raises(ValueError):
        learning_rate(EPOCHS)


def test_roles_are_exact_and_inference_remains_raw() -> None:
    lock = _lock()
    assert tuple(lock["roles"]) == ROLE_NAMES
    for name in (
        "balanced_bce_candidate",
        "balanced_softmax_tau025_control",
        "reversed_prior_bce_control",
    ):
        assert lock["equation"][name]["inference"] == "argmax(raw_logits)"
