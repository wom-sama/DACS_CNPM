import json
import os
from pathlib import Path

import pytest

from trkh.tools.audit_trkh_artifact_retention import (
    DEFAULT_ESSENTIAL_FILE_LOCKS,
    DEFAULT_PRESENTATION_ROOT_LOCKS,
    _classify_directory,
    _scan_file_inventory,
    _sha256_file,
    audit_trkh_artifact_retention,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _touch(path: Path, payload: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _make_protected_tree(repo_root: Path) -> list[str]:
    protected = [
        r"runs\keeper\summary.json",
        r"runs\gate\summary.json",
        r"..\newdataset\class_f\data.yaml",
        r"..\newdataset\yolo_f\data.yaml",
    ]
    for relative in protected:
        _touch(repo_root / relative)
    return protected


def _file_lock(path: Path, repo_root: Path, *, role: str = "essential") -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.relative_to(repo_root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _root_lock(path: Path, repo_root: Path, *, role: str = "presentation") -> dict[str, object]:
    inventory = _scan_file_inventory(path)
    assert inventory["read_errors"] == []
    return {
        "role": role,
        "path": str(path.relative_to(repo_root)),
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
        "inventory_sha256": inventory["inventory_sha256"],
    }


def test_retention_audit_passes_when_compacted_originals_are_absent(tmp_path: Path) -> None:
    repo_root = tmp_path / "TRKH"
    runs_root = repo_root / "runs"
    protected = _make_protected_tree(repo_root)
    _touch(runs_root / "yolof_oof_folds_train5_20260704" / "summary.json")
    _touch(runs_root / "evidence_legacy" / "summary.json")
    manifest = _write_json(
        runs_root / "cleanup_manifest.json",
        {"compacted_names": ["embedding_retrieval_old_20260627", "selector_old_20260611"]},
    )

    summary = audit_trkh_artifact_retention(
        runs_root=runs_root,
        output_dir=runs_root / "artifact_retention",
        protected_relative_paths=protected,
        essential_file_locks=(),
        presentation_root_locks=(),
        compaction_manifests=[manifest],
        created_at="2026-07-06T23:50:00",
    )

    assert summary["retention_audit_passed"] is True
    assert summary["deleted_anything"] is False
    assert summary["raw_dataset_touched"] is False
    assert summary["test_split_used_for_new_diagnostics"] is False
    assert summary["trainable_manifest_written"] is False
    assert summary["compacted_original_checks"]["remaining_original_count"] == 0
    assert (runs_root / "artifact_retention" / "summary.json").is_file()
    assert (runs_root / "artifact_retention" / "largest_directories.csv").is_file()
    assert (runs_root / "artifact_retention" / "future_cleanup_review_candidates.csv").is_file()
    assert (runs_root / "artifact_retention" / "presentation_root_file_inventory.csv").is_file()


def test_retention_audit_blocks_stale_compacted_originals(tmp_path: Path) -> None:
    repo_root = tmp_path / "TRKH"
    runs_root = repo_root / "runs"
    protected = _make_protected_tree(repo_root)
    stale = runs_root / "embedding_retrieval_old_20260627"
    _touch(stale / "summary.json")
    manifest = _write_json(runs_root / "cleanup_manifest.json", {"compacted_names": [stale.name]})

    summary = audit_trkh_artifact_retention(
        runs_root=runs_root,
        output_dir=runs_root / "artifact_retention",
        protected_relative_paths=protected,
        essential_file_locks=(),
        presentation_root_locks=(),
        compaction_manifests=[manifest],
        created_at="2026-07-06T23:55:00",
    )

    assert summary["retention_audit_passed"] is False
    assert "compacted_original_directory_remaining" in summary["blockers"]
    assert summary["compacted_original_checks"]["remaining_original_count"] == 1
    assert summary["future_cleanup_review_candidate_count"] == 1
    candidate = summary["future_cleanup_review_candidates"][0]
    assert candidate["name"] == stale.name
    assert candidate["retention_decision"] == "STALE_COMPACTED_ORIGINAL_PRESENT"


def test_current_presentation_and_scientific_artifacts_are_explicitly_protected() -> None:
    expected = {
        "full_v8_yolof_randominit_30e_20260714_105524": "KEEP_FULL_SCRATCH_BASELINE",
        "paper_pretrained_compare_val_reuse_20260729_001455": "KEEP_PRESENTATION_COMPARISON",
        "audit_pair_surface_ddf_a0_synthetic_20260729_003200": "KEEP_ENGINEERING_EVIDENCE",
        "audit_pair_surface_ddf_v1_fold_leakage_20260729": "KEEP_PROTOCOL_SUPERSESSION_EVIDENCE",
        "audit_pair_surface_ddf_v2_geometry_metadata_20260729": "KEEP_PROSPECTIVE_PROTOCOL_EVIDENCE",
        "presentation_keeper_bundle_20260729": "KEEP_PRESENTATION_BUNDLE",
    }

    for name, decision in expected.items():
        assert _classify_directory(name, {})[0] == decision

    assert len(DEFAULT_PRESENTATION_ROOT_LOCKS) == 10
    assert {str(item["role"]) for item in DEFAULT_ESSENTIAL_FILE_LOCKS} >= {
        "pretrained_resnet50_checkpoint",
        "pretrained_mobilenetv3_checkpoint",
        "pretrained_efficientnetv2_s_checkpoint",
        "pretrained_convnext_tiny_checkpoint",
        "pretrained_aidt_fusion_checkpoint",
        "cvat_yolo_exporter",
        "canonical_yolo_manifest",
    }


def test_retention_audit_exposes_complete_deterministic_hash_inventory(tmp_path: Path) -> None:
    repo_root = tmp_path / "TRKH"
    runs_root = repo_root / "runs"
    presentation_root = runs_root / "presentation"
    first = _touch(presentation_root / "b.txt", "beta")
    second = _touch(presentation_root / "nested" / "a.txt", "alpha")
    essential = _touch(repo_root / "external" / "checkpoint.pt", "weights")
    root_lock = _root_lock(presentation_root, repo_root)
    file_lock = _file_lock(essential, repo_root)

    summary = audit_trkh_artifact_retention(
        runs_root=runs_root,
        output_dir=runs_root / "audit",
        protected_relative_paths=(),
        essential_file_locks=[file_lock],
        presentation_root_locks=[root_lock],
        absent_prefixes=(),
        compaction_manifests=(),
        created_at="2026-07-29T02:00:00",
    )

    assert summary["retention_audit_passed"] is True
    essential_check = summary["essential_file_checks"][0]
    assert essential_check["bytes_match"] is True
    assert essential_check["sha256_match"] is True
    root_check = summary["presentation_root_checks"][0]
    assert root_check["file_count_match"] is True
    assert root_check["total_bytes_match"] is True
    assert root_check["inventory_sha256_match"] is True
    assert root_check["files"] == [
        {
            "relative_path": str(first.relative_to(presentation_root)).replace("\\", "/"),
            "bytes": first.stat().st_size,
            "sha256": _sha256_file(first),
        },
        {
            "relative_path": str(second.relative_to(presentation_root)).replace("\\", "/"),
            "bytes": second.stat().st_size,
            "sha256": _sha256_file(second),
        },
    ]
    assert summary["presentation_root_inventory_file_count"] == 2
    persisted = json.loads((runs_root / "audit" / "summary.json").read_text(encoding="utf-8"))
    assert persisted["presentation_root_checks"] == summary["presentation_root_checks"]


def test_retention_audit_fails_closed_on_hash_inventory_and_missing_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path / "TRKH"
    runs_root = repo_root / "runs"
    presentation_root = runs_root / "presentation"
    _touch(presentation_root / "kept.txt", "kept")
    essential = _touch(repo_root / "external" / "checkpoint.pt", "same-size-a")
    root_lock = _root_lock(presentation_root, repo_root)
    file_lock = _file_lock(essential, repo_root)

    essential.write_text("same-size-b", encoding="utf-8")
    _touch(presentation_root / "unexpected.txt", "new")
    missing_root_lock = dict(root_lock, role="missing", path="runs/missing")
    missing_file_lock = dict(file_lock, role="missing", path="external/missing.pt")
    summary = audit_trkh_artifact_retention(
        runs_root=runs_root,
        output_dir=runs_root / "audit",
        protected_relative_paths=(),
        essential_file_locks=[file_lock, missing_file_lock],
        presentation_root_locks=[root_lock, missing_root_lock],
        absent_prefixes=(),
        compaction_manifests=(),
    )

    assert summary["retention_audit_passed"] is False
    assert set(summary["blockers"]) >= {
        "essential_file_missing",
        "essential_file_sha256_mismatch",
        "presentation_root_missing",
        "presentation_root_file_count_mismatch",
        "presentation_root_total_bytes_mismatch",
        "presentation_root_inventory_sha256_mismatch",
    }
    assert summary["essential_file_checks"][0]["bytes_match"] is True
    assert summary["essential_file_checks"][0]["sha256_match"] is False
    assert summary["presentation_root_checks"][0]["files"][-1]["relative_path"] == "unexpected.txt"


def test_retention_audit_fails_closed_on_symlinks(tmp_path: Path) -> None:
    repo_root = tmp_path / "TRKH"
    runs_root = repo_root / "runs"
    presentation_root = runs_root / "presentation"
    target = _touch(repo_root / "targets" / "artifact.bin", "content")
    _touch(presentation_root / "regular.txt", "regular")
    root_lock = _root_lock(presentation_root, repo_root)
    linked_in_root = presentation_root / "linked.bin"
    essential_link = repo_root / "external" / "checkpoint.pt"
    essential_link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, linked_in_root)
        os.symlink(target, essential_link)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    essential_lock = {
        "role": "linked-essential",
        "path": str(essential_link.relative_to(repo_root)),
        "bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
    }

    summary = audit_trkh_artifact_retention(
        runs_root=runs_root,
        output_dir=runs_root / "audit",
        protected_relative_paths=(),
        essential_file_locks=[essential_lock],
        presentation_root_locks=[root_lock],
        absent_prefixes=(),
        compaction_manifests=(),
    )

    assert summary["retention_audit_passed"] is False
    assert "essential_file_symlink_detected" in summary["blockers"]
    assert "presentation_root_symlink_detected" in summary["blockers"]
    assert summary["essential_file_checks"][0]["is_symlink"] is True
    assert summary["presentation_root_checks"][0]["symlinks"] == ["linked.bin"]
