from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trkh.tools import precheck_dinov3_cgaer_b11_sourcefold as preflight_module
from trkh.tools.precheck_dinov3_cgaer_b11_sourcefold import (
    EXPECTED_ASSIGNMENT_CSV_SHA256,
    EXPECTED_CACHE_HASHES,
    EXPECTED_CACHE_ORDER_FOLD_SHA256,
    EXPECTED_CACHE_ORDER_GROUP_SHA256,
    EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256,
    PROTOCOL_ID,
    _architecture_smoke,
    _readiness_diagnostic,
    _sha256,
    _source_hashes,
    component_conflict_targets,
    validate_runner_preflight,
)


def test_component_conflict_is_normalized_and_singletons_are_masked() -> None:
    labels = np.asarray([4, 0, 0, 1, 1, 2], dtype=np.int64)
    groups = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int64)
    result = component_conflict_targets(labels, groups)
    assert not result["eligible"][0]
    assert result["targets"][1] == 0.0
    assert result["targets"][3] == pytest.approx(8.0 / 9.0)
    assert result["inverse_sizes"][1] == pytest.approx(0.5)
    assert result["inverse_sizes"][3] == pytest.approx(1.0 / 3.0)


def test_architecture_smoke_locks_zero_cap_gradients_and_jacobian() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 384)).astype(np.float32)
    logits = rng.normal(scale=0.2, size=(20, 5)).astype(np.float32)
    labels = np.tile(np.arange(5), 4).astype(np.int64)
    result = _architecture_smoke(features, logits, labels)
    assert result["passed"]
    assert result["parameter_count_each"] == 3_125
    assert result["active_zero_max_abs_error"] == 0.0
    assert result["candidate_control_jacobian_max_abs_error"] <= 1e-7
    assert result["stress_residual_l2_max"] < 0.5


def test_fixed_linear_readiness_is_fold_oof_and_finite() -> None:
    rng = np.random.default_rng(9)
    n = 250
    folds = np.repeat(np.arange(5), n // 5)
    labels = np.tile(np.arange(5), n // 5).astype(np.int64)
    mixed = (np.arange(n) % 3 == 0)
    features = rng.normal(size=(n, 384)).astype(np.float32)
    features[:, 0] = np.where(mixed, 4.0, -4.0) + rng.normal(scale=0.1, size=n)
    logits = np.zeros((n, 5), dtype=np.float32)
    logits[np.arange(n), labels] = 1.0
    boundary_errors = mixed & (labels == 0)
    logits[boundary_errors, 1] = 2.0
    conflict = {
        "mixed": mixed,
        "eligible": np.ones(n, dtype=bool),
    }
    result = _readiness_diagnostic(features, logits, labels, folds, conflict)
    assert result["passed"]
    assert result["mixed_vs_pure_auroc"] > 0.90
    assert result["b9_class1_boundary_error_auroc"] > 0.75


def _valid_payload() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "preflight_passed": True,
        "source_files": _source_hashes(),
        "locked_inputs": {
            "a0_cache_manifest_sha256": EXPECTED_CACHE_HASHES["manifest"],
            "a0_tokens_sha256": EXPECTED_CACHE_HASHES["tokens"],
            "a0_logits_sha256": EXPECTED_CACHE_HASHES["logits"],
            "a0_labels_sha256": EXPECTED_CACHE_HASHES["labels"],
            "a0_paths_sha256": EXPECTED_CACHE_HASHES["paths"],
            "assignment_csv_sha256": EXPECTED_ASSIGNMENT_CSV_SHA256,
            "cache_order_fold_sha256": EXPECTED_CACHE_ORDER_FOLD_SHA256,
            "cache_order_path_fold_sha256": EXPECTED_CACHE_ORDER_PATH_FOLD_SHA256,
            "cache_order_group_sha256": EXPECTED_CACHE_ORDER_GROUP_SHA256,
        },
        "architecture_smoke": {"passed": True},
        "readiness_diagnostic": {"passed": True},
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "candidate_oof_training_run": False,
        "full_validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
    }


def test_runner_validator_rehashes_sources_and_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    payload = validate_runner_preflight(path, expected_sha256=_sha256(path))
    assert payload["preflight_passed"]
    unsafe = _valid_payload()
    unsafe["test_permission"] = True
    path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="stale/unsafe"):
        validate_runner_preflight(path, expected_sha256=_sha256(path))


def test_runner_validator_binds_exact_runner_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    changed = dict(_source_hashes())
    changed["runner_sha256"] = "0" * 64
    monkeypatch.setattr(preflight_module, "_source_hashes", lambda: changed)
    with pytest.raises(ValueError, match="sources_current"):
        validate_runner_preflight(path, expected_sha256=_sha256(path))
