from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trkh.tools.audit_iformer_s_frozen_transfer_b14 as b14


def _valid_b13_summary() -> dict[str, object]:
    git_state = {"head": b14.B13_HEAD}
    return {
        "protocol_id": b14.B13_PROTOCOL_ID,
        "integrity_complete": True,
        "git": {"start": dict(git_state), "end": dict(git_state)},
        "dataset": {
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_rows": b14.EXPECTED_TRAIN_ROWS,
        },
        "assignment": {
            "csv_sha256": b14.EXPECTED_ASSIGNMENT_SHA256,
            "assignment_int64_sha256": b14.EXPECTED_FOLD_VECTOR_SHA256,
            "group_vector_int64_sha256": b14.EXPECTED_GROUP_VECTOR_SHA256,
        },
    }


def test_b13_summary_guard_rejects_test_or_integrity_drift() -> None:
    payload = _valid_b13_summary()
    b14._validate_b13_summary_payload(payload)

    bad_test = _valid_b13_summary()
    bad_test["dataset"]["test_split_used"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="summary contract failed"):
        b14._validate_b13_summary_payload(bad_test)

    bad_integrity = _valid_b13_summary()
    bad_integrity["integrity_complete"] = False
    with pytest.raises(ValueError, match="summary contract failed"):
        b14._validate_b13_summary_payload(bad_integrity)


def test_comparator_reader_never_requests_efficientvim_scores() -> None:
    class GuardedArchive:
        def __init__(self) -> None:
            self.accessed: list[str] = []
            self.values = {
                "labels": np.array([1], dtype=np.int64),
                "folds": np.array([0], dtype=np.int64),
                "groups": np.array([3], dtype=np.int64),
                "dino_scores": np.zeros((1, 5), dtype=np.float64),
            }

        def __getitem__(self, key: str) -> np.ndarray:
            self.accessed.append(key)
            if key == "efficientvim_scores":
                raise AssertionError("B14 attempted to read forbidden EfficientViM scores")
            return self.values[key]

    archive = GuardedArchive()
    arrays = b14._read_allowed_b13_arrays(archive)
    assert set(arrays) == {"labels", "folds", "groups", "dino_scores"}
    assert archive.accessed == ["labels", "folds", "groups", "dino_scores"]
    assert "efficientvim_scores" not in archive.accessed


def test_train_path_guard_rejects_every_non_train_form(tmp_path: Path) -> None:
    class_name = b14.EXPECTED_CLASSES[0]
    valid = [f"train/{class_name}/image.jpg"]
    labels = np.array([0], dtype=np.int64)
    resolved = b14._resolve_train_paths(
        valid, labels, tmp_path, require_files=False
    )
    assert resolved[0].is_relative_to((tmp_path / "train").resolve())

    invalid = (
        f"val/{class_name}/image.jpg",
        f"test/{class_name}/image.jpg",
        f"train/{class_name}/../other/image.jpg",
        "../train/image.jpg",
        r"C:\outside\image.jpg",
    )
    for path in invalid:
        with pytest.raises(ValueError, match="non-TRAIN path"):
            b14._resolve_train_paths(
                [path], labels, tmp_path, require_files=False
            )

    with pytest.raises(ValueError, match="class directory/label mismatch"):
        b14._resolve_train_paths(
            [f"train/{b14.EXPECTED_CLASSES[1]}/image.jpg"],
            labels,
            tmp_path,
            require_files=False,
        )


def test_release_asset_selection_is_exact() -> None:
    asset = {
        "id": b14.OFFICIAL_CHECKPOINT_ASSET_ID,
        "name": "iFormer_s.pth",
        "size": b14.OFFICIAL_CHECKPOINT_BYTES,
        "browser_download_url": b14.OFFICIAL_CHECKPOINT_URL,
    }
    selected = b14._select_release_asset(
        {"tag_name": b14.OFFICIAL_TAG, "assets": [asset]}
    )
    assert selected == asset

    wrong = dict(asset, size=b14.OFFICIAL_CHECKPOINT_BYTES - 1)
    with pytest.raises(ValueError, match="metadata changed"):
        b14._select_release_asset(
            {"tag_name": b14.OFFICIAL_TAG, "assets": [wrong]}
        )


def test_locked_locations_reject_an_alternate_dataset() -> None:
    defaults = b14._defaults()
    args = SimpleNamespace(**defaults)
    b14._assert_locked_locations(args, include_data=True)
    args.data_root = Path(r"D:\DataAI\alternate_class_f")
    with pytest.raises(ValueError, match="locked artifact locations changed"):
        b14._assert_locked_locations(args, include_data=True)


def test_preflight_checks_are_positive_and_gate_name_is_iformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = {
        "branch": b14.EXPECTED_BRANCH,
        "tracked_worktree_clean": True,
    }
    checks = b14._preflight_checks(project_git=clean, mobile_passed=True)
    assert all(checks.values())
    assert checks["validation_not_constructed"]
    assert checks["test_not_constructed"]
    assert not b14._preflight_checks(project_git=clean, mobile_passed=False)[
        "mobile_precondition"
    ]

    monkeypatch.setattr(
        b14,
        "assess_gate",
        lambda **_kwargs: {
            "signal_gate_passed": False,
            "exact_m1_final_feature_route_closed": True,
        },
    )
    gate = b14._gate_for_iformer(
        dino={},
        candidate={},
        bootstrap={},
        integrity_complete=True,
        readouts_converged=True,
    )
    assert "exact_m1_final_feature_route_closed" not in gate
    assert gate["exact_iformer_s_pooled320_route_closed"] is True


def test_focused_test_guard_rejects_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        b14.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="4 passed, 1 skipped", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="failed or skipped"):
        b14._run_focused_tests(Path("source"), Path("checkpoint"))


def test_partial_quarantine_records_retry_boundary(tmp_path: Path) -> None:
    output = tmp_path / "formal_r1"
    partial = tmp_path / "formal_r1.partial"
    partial.mkdir()
    (partial / "run_owner.json").write_text(
        json.dumps({"protocol_id": b14.PROTOCOL_ID}), encoding="utf-8"
    )
    (partial / "descriptor.npy").write_bytes(b"evidence")
    b14._quarantine_partial(
        SimpleNamespace(output_dir=output), RuntimeError("infrastructure")
    )
    manifest = json.loads(
        (partial / "failure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "QUARANTINED_PARTIAL"
    assert manifest["metrics_constructed"] is False
    assert manifest["retry_allowed"] is True
    assert manifest["files"][0]["sha256"] == b14._sha256(
        partial / "descriptor.npy"
    )

    output2 = tmp_path / "formal_r2"
    partial2 = tmp_path / "formal_r2.partial"
    partial2.mkdir()
    (partial2 / "run_owner.json").write_text(
        json.dumps({"protocol_id": b14.PROTOCOL_ID}), encoding="utf-8"
    )
    (partial2 / "stage_metrics_constructed.json").write_text("{}", encoding="utf-8")
    b14._quarantine_partial(
        SimpleNamespace(output_dir=output2), RuntimeError("after metrics")
    )
    manifest2 = json.loads(
        (partial2 / "failure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest2["metrics_constructed"] is True
    assert manifest2["retry_allowed"] is False
