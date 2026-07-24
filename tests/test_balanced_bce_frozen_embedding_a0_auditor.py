from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trkh.tools.audit_balanced_bce_frozen_embedding_a0 import (
    DataAccessLedger,
    LOCK_PATH,
    VISUAL_ANCHORS,
    _load_audit_inputs,
    _jsonable,
    _recursive_numeric_difference,
    _verify_engine_contract,
    analyze_predictions,
    assess_performance_gates,
    load_locked_train_cache,
)
from trkh.tools.balanced_bce_frozen_embedding_a0_engine import ROLE_NAMES


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_locked_inputs_rebuild_fold_protocol_without_candidate_metrics() -> None:
    lock = _lock()
    immutable = lock["immutable_inputs"]
    cache = load_locked_train_cache(
        Path(immutable["train_embedding_cache"]["path"])
    )
    loaded, folds, sources, evidence, protocol = _load_audit_inputs(lock)
    assert np.array_equal(loaded["labels"], cache["labels"])
    assert evidence["assignment_sha256"] == (
        lock["dataset_declaration"]["outer_assignment_sha256"]
    )
    assert len(protocol) == 5
    assert sum(int(row["held_rows"]) for row in protocol) == 9215
    assert all(int(row["source_overlap"]) == 0 for row in protocol)
    assert len(folds) == len(sources) == 9215
    assert _verify_engine_contract(lock)


def test_dynamic_ledger_blocks_forbidden_split_before_open() -> None:
    lock = _lock()
    ledger = DataAccessLedger(lock, lock_path=LOCK_PATH)
    forbidden = (
        r"D:\DataAI\AIEx\newdataset\yolo_f\images"
        r"\validation\forbidden.jpg"
    )
    with pytest.raises(PermissionError, match="forbidden_split_component"):
        ledger.observe_open((forbidden, "r", 0))
    snapshot = ledger.snapshot()
    assert snapshot["blocked_attempt_count"] == 1
    assert snapshot["validation_open_count"] == 1


def test_visual_authorization_is_exact_train_only() -> None:
    lock = _lock()
    ledger = DataAccessLedger(lock, lock_path=LOCK_PATH)
    immutable = lock["immutable_inputs"]
    cache = load_locked_train_cache(
        Path(immutable["train_embedding_cache"]["path"])
    )
    paths = [str(cache["paths"][index]) for index in VISUAL_ANCHORS]
    ledger.authorize_visual_paths(paths)
    ledger.observe_open((paths[0], "r", 0))
    with pytest.raises(PermissionError, match="undeclared_input"):
        ledger.observe_open(
            (
                r"D:\DataAI\AIEx\newdataset\yolo_f"
                r"\images\train\not_locked.jpg",
                "r",
                0,
            )
        )


def test_recursive_comparison_normalizes_only_boolean_scalars() -> None:
    assert _recursive_numeric_difference(True, np.bool_(True)) == 0.0
    assert _recursive_numeric_difference(False, np.bool_(True)) == np.inf
    assert _recursive_numeric_difference(True, 1) == np.inf
    assert _recursive_numeric_difference({"a": [1.0]}, {"a": [1.0]}) == 0.0


def test_jsonable_normalizes_numpy_without_weakening_boolean_comparison() -> None:
    payload = _jsonable(
        {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "values": np.asarray([1.0, 2.0], dtype=np.float32),
        }
    )
    assert payload == {"flag": True, "count": 3, "values": [1.0, 2.0]}
    assert isinstance(payload["flag"], bool)
    assert isinstance(payload["count"], int)


def test_gate_logic_rejects_identical_candidate_and_controls() -> None:
    lock = _lock()
    targets = np.tile(np.arange(5, dtype=np.int64), 20)
    folds = np.arange(targets.size, dtype=np.int64) % 5
    probabilities = np.full((targets.size, 5), 0.01, dtype=np.float32)
    probabilities[np.arange(targets.size), targets] = 0.96
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    role_probabilities = {
        role: probabilities.copy() for role in ROLE_NAMES
    }
    role_logits = {
        role: np.log(probabilities).astype(np.float32) for role in ROLE_NAMES
    }
    analysis = analyze_predictions(
        targets=targets,
        folds=folds,
        logits=role_logits,
        probabilities=role_probabilities,
    )
    analysis["_candidate_actions"] = role_probabilities[
        "balanced_bce_candidate"
    ].argmax(axis=1)
    analysis["_reversed_actions"] = role_probabilities[
        "reversed_prior_bce_control"
    ].argmax(axis=1)
    gate = assess_performance_gates(analysis, lock)
    assert not gate["performance_pass"]
    assert not gate["groups"][
        "candidate_vs_historical_and_sign_controls"
    ]["prediction_array_differs_from_reversed"]
