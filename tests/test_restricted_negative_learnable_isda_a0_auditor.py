from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from trkh.tools import audit_restricted_negative_learnable_isda_a0 as audit


def _one_hot_probabilities(predictions: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(predictions), 5), dtype=np.float64)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities


def test_equation_diagnostics_match_independent_oracle() -> None:
    diagnostics = audit.equation_diagnostics()
    assert diagnostics["equation_oracle_max_error"] <= 1e-12
    assert diagnostics["clean_class_delta_exact_zero"]
    assert diagnostics["nonfocus_delta_exact_zero"]
    assert diagnostics["initial_covariance_max_error"] == 0.0


def test_classification_metrics_and_transition_accounting() -> None:
    targets = np.asarray([1, 1, 0, 2, 4], dtype=np.int64)
    perfect = audit.classification_metrics(
        targets, _one_hot_probabilities(targets)
    )
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == 0.8
    assert perfect["class1_tp"] == 2
    assert perfect["restricted_fp"] == 0

    base = _one_hot_probabilities(
        np.asarray([1, 0, 1, 1, 4], dtype=np.int64)
    )
    candidate = _one_hot_probabilities(
        np.asarray([1, 1, 0, 2, 1], dtype=np.int64)
    )
    transition = audit.transition_stats(targets, base, candidate)
    assert transition["corrections"] == 3
    assert transition["harms"] == 1
    assert transition["class1_tp_net"] == 1
    assert transition["restricted_fp_removals"] == 2
    assert transition["restricted_fp_creations"] == 1
    assert transition["restricted_fp_net_removal"] == 1


def test_analysis_and_conjunctive_gate_have_all_locked_groups() -> None:
    targets = np.tile(np.arange(5, dtype=np.int64), 10)
    folds = np.repeat(np.arange(5, dtype=np.int64), 10)
    probabilities = np.full((len(targets), 5), 0.025, dtype=np.float32)
    probabilities[np.arange(len(targets)), targets] = 0.9
    outputs = {
        role: probabilities.copy() for role in audit.ROLE_NAMES
    }
    analysis = audit.analyze_predictions(targets, folds, outputs)
    lock = json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))
    gate = audit.assess_performance_gates(analysis, lock)
    assert set(gate["groups"]) == {
        "control_compatibility",
        "candidate_vs_ce_and_fold",
        "candidate_vs_classwise",
        "causal_controls",
        "paired_covnet_seed_repeat",
    }
    assert gate["total"] > 20
    assert not gate["performance_pass"]


def test_initial_state_hashes_match_machine_lock() -> None:
    lock = json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))
    rows = audit._initial_state_evidence(lock)
    assert len(rows) == 5
    assert all(len(row["primary_covnet_sha256"]) == 64 for row in rows)
    assert all(
        row["primary_head_sha256"] == row["repeat_head_sha256"]
        and row["primary_covnet_sha256"] != row["repeat_covnet_sha256"]
        for row in rows
    )
    assert rows == audit._initial_state_evidence(lock)


def test_engine_contract_matches_lock_and_rejects_role_drift() -> None:
    lock = json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))
    checks = audit._verify_engine_lock(lock)
    assert checks and all(checks.values())
    lock["roles"] = list(reversed(lock["roles"]))
    with pytest.raises(ValueError, match="roles_exact"):
        audit._verify_engine_lock(lock)

    changed = json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))
    changed["optimization"]["cublas_workspace_config"] = ":16:8"
    with pytest.raises(ValueError, match="determinism_exact"):
        audit._verify_engine_lock(changed)


def test_runtime_lock_matches_and_rejects_version_drift() -> None:
    lock_path = audit.LOCK_PATH
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = audit._verify_runtime_lock(lock)
    assert result["passed"]
    assert all(result["checks"].values())

    changed = json.loads(lock_path.read_text(encoding="utf-8"))
    changed["runtime"]["numpy"] = "0.0.invalid"
    with pytest.raises(ValueError, match="runtime differs"):
        audit._verify_runtime_lock(changed)


def test_cublas_workspace_is_locked_before_formal_cuda_context() -> None:
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    code = "\n".join(
        [
            "import json",
            "import torch",
            "from trkh.tools import "
            "audit_restricted_negative_learnable_isda_a0 as audit",
            "lock = json.loads(audit.LOCK_PATH.read_text(encoding='utf-8'))",
            "audit._verify_runtime_lock(lock)",
            "audit._configure_torch(torch.device('cuda'))",
            "left = torch.ones((4, 4), device='cuda')",
            "right = left @ left",
            "assert float(right.sum().item()) == 64.0",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=audit.REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_engineering_benchmark_hash_matches_machine_lock() -> None:
    lock = json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))
    verified = audit._verify_file(
        audit.REPO_ROOT,
        lock["engineering_benchmark"],
        label="engineering_benchmark",
    )
    assert verified["passed"]
    assert verified["sha256"] == lock["engineering_benchmark"]["sha256"]


def test_recursive_and_array_replay_comparisons() -> None:
    left = {"a": [1.0, {"b": 2}], "c": True}
    right = {"a": [1.0 + 1e-12, {"b": 2}], "c": True}
    assert audit._recursive_numeric_difference(left, right) == pytest.approx(
        1e-12
    )
    arrays = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    repeated = {"x": np.asarray([1.0, 2.0], dtype=np.float32)}
    assert audit._maximum_array_error(arrays, repeated) == 0.0
    with pytest.raises(ValueError):
        audit._maximum_array_error(
            arrays, {"x": np.asarray([1.0, 2.0], dtype=np.float64)}
        )
    assert (
        audit._recursive_numeric_difference(True, np.bool_(True)) == 0.0
    )
    with pytest.raises(ValueError, match="Replay boolean differs"):
        audit._recursive_numeric_difference(True, np.bool_(False))
    with pytest.raises(ValueError, match="Replay boolean differs"):
        audit._recursive_numeric_difference(True, 1)


def test_replay_visual_builder_matches_formal_read_set_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    formal_contact = output / "fixed_visual_contact_sheet.png"
    formal_contact.write_bytes(b"formal")
    observed: dict[str, object] = {}

    def fake_build_visual_evidence(**kwargs: object):
        replay_contact = Path(kwargs["contact_sheet_path"])
        observed["contact_sheet_path"] = replay_contact
        observed["render_contact"] = kwargs["render_contact"]
        replay_contact.write_bytes(b"replay")
        return (
            {"values": np.asarray([1.0], dtype=np.float32)},
            {"contact_sheet": str(replay_contact), "rows": []},
        )

    monkeypatch.setattr(
        audit, "build_visual_evidence", fake_build_visual_evidence
    )
    arrays, metadata = audit._build_replay_visual_evidence(
        cache={},
        folds=np.asarray([], dtype=np.int64),
        sources=np.asarray([], dtype=str),
        outputs={},
        views={},
        lock={},
        output=output,
    )

    replay_contact = Path(observed["contact_sheet_path"])
    assert observed["render_contact"] is True
    assert replay_contact != formal_contact
    assert not replay_contact.exists()
    assert formal_contact.read_bytes() == b"formal"
    assert metadata["contact_sheet"] == str(formal_contact)
    assert np.array_equal(
        arrays["values"], np.asarray([1.0], dtype=np.float32)
    )


def test_manifest_round_trip_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"\x00\x01")
    manifest = audit._write_manifest(
        tmp_path, manifest_name="artifact_manifest.json"
    )
    verified = audit._verify_manifest(
        tmp_path, "artifact_manifest.json"
    )
    assert manifest["payload_count"] == 2
    assert verified["passed"]
    extra = tmp_path / "unexpected.txt"
    extra.write_text("not manifested\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest file set differs"):
        audit._verify_manifest(tmp_path, "artifact_manifest.json")
    extra.unlink()
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit._verify_manifest(tmp_path, "artifact_manifest.json")


def test_repository_state_parses_nul_paths_and_hashes_protected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = tmp_path / "deep research.md"
    protected.write_text("locked\n", encoding="utf-8")
    expected_sha256 = audit._sha256(protected)

    def fake_git(root: Path, *arguments: str) -> str:
        assert root == tmp_path
        if arguments[:2] == ("status", "--porcelain=v1"):
            return "?? deep research.md\0"
        if arguments == ("rev-parse", "HEAD"):
            return "abc123"
        if arguments == ("rev-parse", "@{u}"):
            return "abc123"
        raise AssertionError(arguments)

    monkeypatch.setattr(audit, "_run_git", fake_git)
    state = audit._repository_state(
        tmp_path, {"deep research.md": expected_sha256}
    )
    assert state["passed"]
    assert state["protected_untracked_hashes_exact"]

    protected.write_text("mutated\n", encoding="utf-8")
    mutated = audit._repository_state(
        tmp_path, {"deep research.md": expected_sha256}
    )
    assert not mutated["passed"]
    assert not mutated["protected_untracked_hashes_exact"]


def test_oof_csv_round_trip_preserves_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_ROWS", 3)
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    cache = {
        "labels": labels,
        "paths": np.asarray(
            [
                r"D:\dataset\images\train\a.jpg",
                r"D:\dataset\images\train\b.jpg",
                r"D:\dataset\images\train\c.jpg",
            ]
        ),
    }
    folds = np.asarray([0, 1, 2], dtype=np.int64)
    sources = np.asarray(["a", "b", "c"])
    probabilities = np.full((3, 5), 0.025, dtype=np.float32)
    probabilities[np.arange(3), labels] = 0.9
    outputs = {
        role: probabilities.copy() for role in audit.ROLE_NAMES
    }
    path = tmp_path / "oof.csv"
    audit._write_oof_rows(
        path,
        cache=cache,
        folds=folds,
        sources=sources,
        outputs=outputs,
    )
    loaded = audit._read_oof_rows(
        path,
        cache=cache,
        folds=folds,
        sources=sources,
    )
    for role in audit.ROLE_NAMES:
        np.testing.assert_allclose(loaded[role], probabilities, atol=1e-9)
        np.testing.assert_array_equal(
            loaded[f"{role}__actions"], labels
        )


def test_peak_resource_monitor_returns_without_hanging() -> None:
    with audit.PeakResourceMonitor(interval_seconds=0.01) as monitor:
        payload = bytearray(1024 * 1024)
        assert len(payload) == 1024 * 1024
        time.sleep(0.03)
    assert monitor.peak_rss_bytes > 0
    assert 0.0 <= monitor.peak_virtual_memory_fraction <= 1.0


def test_raw_train_metadata_hash_changes_with_train_label(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "sample.jpg"
    label = labels / "sample.txt"
    data_yaml = tmp_path / "data.yaml"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
    data_yaml.write_text("train: images/train\n", encoding="utf-8")
    before, count = audit._raw_train_metadata_sha256(
        [str(image)], data_yaml
    )
    time.sleep(0.01)
    label.write_text("1 0.5 0.5 1 1\n", encoding="utf-8")
    after, after_count = audit._raw_train_metadata_sha256(
        [str(image)], data_yaml
    )
    assert count == after_count == 3
    assert before != after


def test_data_access_evidence_is_derived_from_declared_paths() -> None:
    cache = {
        "paths": np.asarray(
            [
                r"D:\dataset\images\train\a.jpg",
                r"D:\dataset\images\train\b.jpg",
            ]
        )
    }
    evidence = audit._data_access_evidence(
        cache,
        {
            "declared_train_image_rows": 2,
            "forbidden_split_path_count": 0,
        },
        {
            "hook_installed": True,
            "hook_probe_seen": True,
            "observed_event_count": 3,
            "unique_path_count": 2,
            "blocked_attempt_count": 0,
            "validation_open_count": 0,
            "test_open_count": 0,
            "ordered_events_sha256": "a" * 64,
            "passed": True,
        },
    )
    assert evidence["cache_train_image_rows"] == 2
    assert evidence["cidt_declared_train_image_rows"] == 2
    assert evidence["forbidden_split_path_count"] == 0
    assert evidence["dynamic_hook_installed"]
    assert evidence["dynamic_hook_probe_seen"]
    assert evidence["observed_data_domain_open_count"] == 3
    assert evidence["blocked_attempt_count"] == 0
    assert evidence["validation_open_count"] == 0
    assert evidence["test_open_count"] == 0
    assert evidence["dynamic_ledger_pass"]


def _data_access_test_lock(
    dataset_root: Path,
    runs_root: Path,
    allowed_run: Path,
    data_yaml: Path,
) -> dict:
    return {
        "data_access_audit": {
            "roots": [str(dataset_root), str(runs_root)],
            "forbidden_complete_components": [
                "val",
                "valid",
                "validation",
                "test",
            ],
        },
        "immutable_inputs": {
            "train_cache": {"path": str(allowed_run)},
            "data_yaml": {"path": str(data_yaml)},
        },
    }


def test_dynamic_data_access_ledger_is_exact_for_allowed_train_reads(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    yolo = dataset / "yolo_f"
    runs = tmp_path / "runs"
    image = yolo / "images" / "train" / "sample.jpg"
    data_yaml = yolo / "data.yaml"
    allowed_run = runs / "train_cache.npz"
    image.parent.mkdir(parents=True)
    runs.mkdir()
    image.write_bytes(b"image")
    data_yaml.write_text("train: images/train\n", encoding="utf-8")
    allowed_run.write_bytes(b"cache")
    lock = _data_access_test_lock(dataset, runs, allowed_run, data_yaml)

    snapshots = []
    for _ in range(2):
        ledger = audit.DataAccessLedger(lock)
        with ledger:
            assert image.read_bytes() == b"image"
            assert allowed_run.read_bytes() == b"cache"
            assert data_yaml.read_text(encoding="utf-8").startswith("train")
        snapshots.append(ledger.snapshot())

    assert snapshots[0]["passed"]
    assert snapshots[0]["blocked_attempt_count"] == 0
    assert snapshots[0]["observed_event_count"] > 0
    assert (
        snapshots[0]["ordered_events_sha256"]
        == snapshots[1]["ordered_events_sha256"]
    )
    assert snapshots[0]["path_counts"] == snapshots[1]["path_counts"]


def test_dynamic_data_access_ledger_blocks_split_undeclared_and_write_access(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    yolo = dataset / "yolo_f"
    runs = tmp_path / "runs"
    train_image = yolo / "images" / "train" / "sample.jpg"
    val_image = yolo / "images" / "val" / "sample.jpg"
    misleading_train = yolo / "derived" / "images" / "train" / "sample.jpg"
    other_view = dataset / "class_f" / "images" / "train" / "sample.jpg"
    data_yaml = yolo / "data.yaml"
    allowed_run = runs / "train_cache.npz"
    undeclared_run = runs / "aggregate_manifest.json"
    train_image.parent.mkdir(parents=True)
    val_image.parent.mkdir(parents=True)
    misleading_train.parent.mkdir(parents=True)
    other_view.parent.mkdir(parents=True)
    runs.mkdir()
    train_image.write_bytes(b"train")
    val_image.write_bytes(b"val")
    misleading_train.write_bytes(b"misleading")
    other_view.write_bytes(b"other")
    data_yaml.write_text("train: images/train\n", encoding="utf-8")
    allowed_run.write_bytes(b"cache")
    undeclared_run.write_text("{}", encoding="utf-8")
    lock = _data_access_test_lock(dataset, runs, allowed_run, data_yaml)

    ledger = audit.DataAccessLedger(lock)
    with ledger:
        with pytest.raises(PermissionError, match="forbidden_split_component"):
            val_image.read_bytes()
        with pytest.raises(PermissionError, match="undeclared_runs_artifact"):
            undeclared_run.read_text(encoding="utf-8")
        with pytest.raises(PermissionError, match="undeclared_dataset_artifact"):
            other_view.read_bytes()
        with pytest.raises(PermissionError, match="undeclared_dataset_artifact"):
            misleading_train.read_bytes()
        with pytest.raises(PermissionError, match="write_to_locked_data_domain"):
            train_image.write_bytes(b"mutated")

    snapshot = ledger.snapshot()
    assert not snapshot["passed"]
    assert snapshot["blocked_attempt_count"] == 5
    assert snapshot["validation_open_count"] == 1
    assert snapshot["test_open_count"] == 0
    assert train_image.read_bytes() == b"train"


def test_argument_parser_requires_explicit_phase() -> None:
    args = audit.parse_args(["--preflight-only"])
    assert args.preflight_only
    with pytest.raises(ValueError):
        audit.run_audit(audit.parse_args([]))
