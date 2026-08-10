from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.execute_reviewed_cleanup_manifest import (
    CleanupContractError,
    execute_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, *, target: Path, protected: Path | None = None) -> Path:
    payload = {
        "schema_version": 1,
        "status": "approved_ready",
        "workspace_root": str(root),
        "protected_dataset_roots": [str(protected)] if protected is not None else [],
        "delete_groups": [
            {
                "reason": "test",
                "entries": [
                    {
                        "path": str(target),
                        "kind": "file" if target.is_file() else "directory",
                        "logical_bytes": target.stat().st_size if target.is_file() else 0,
                        "sha256": _sha256(target) if target.is_file() else None,
                    }
                ],
            }
        ],
        "cache_directories": {"logical_bytes": 0, "paths": []},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dry_run_then_execute_exact_file(tmp_path: Path) -> None:
    target = tmp_path / "stale.bin"
    target.write_bytes(b"stale")
    manifest = _manifest(tmp_path, target=target)
    digest = _sha256(manifest)

    dry = execute_manifest(
        manifest,
        expected_sha256=digest,
        expected_root=tmp_path,
        execute=False,
    )
    assert dry["present_before"] == 1
    assert target.is_file()

    result = execute_manifest(
        manifest,
        expected_sha256=digest,
        expected_root=tmp_path,
        execute=True,
    )
    assert result["deleted"] == [str(target)]
    assert not target.exists()


def test_rejects_dataset_target(tmp_path: Path) -> None:
    protected = tmp_path / "dataset"
    protected.mkdir()
    target = protected / "image.jpg"
    target.write_bytes(b"data")
    manifest = _manifest(tmp_path, target=target, protected=protected)

    with pytest.raises(CleanupContractError, match="Dataset cleanup is forbidden"):
        execute_manifest(
            manifest,
            expected_sha256=_sha256(manifest),
            expected_root=tmp_path,
            execute=False,
        )


def test_rejects_changed_manifest_or_file(tmp_path: Path) -> None:
    target = tmp_path / "stale.bin"
    target.write_bytes(b"stale")
    manifest = _manifest(tmp_path, target=target)
    digest = _sha256(manifest)
    target.write_bytes(b"changed")

    with pytest.raises(CleanupContractError, match="size changed"):
        execute_manifest(
            manifest,
            expected_sha256=digest,
            expected_root=tmp_path,
            execute=False,
        )

    with pytest.raises(CleanupContractError, match="Manifest SHA-256 mismatch"):
        execute_manifest(
            manifest,
            expected_sha256="0" * 64,
            expected_root=tmp_path,
            execute=False,
        )
