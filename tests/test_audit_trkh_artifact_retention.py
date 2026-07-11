import json
from pathlib import Path

from trkh.tools.audit_trkh_artifact_retention import audit_trkh_artifact_retention


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
