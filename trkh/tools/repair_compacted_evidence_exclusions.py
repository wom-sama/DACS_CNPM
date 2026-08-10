from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from trkh.core.config import project_dir
from trkh.tools.compact_rejected_run_evidence import (
    _is_within,
    _matches_any,
    _require_child,
    _sha256,
    _write_json,
    combine_exclude_globs,
    refresh_compacted_evidence_metadata,
)


def _validated_relative_path(value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return relative


def _verify_inventory_file(path: Path, row: Dict[str, object]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing compacted evidence payload: {path}")
    expected_bytes = int(row["bytes"])
    observed_bytes = int(path.stat().st_size)
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"Compacted payload size mismatch for {path}: "
            f"{observed_bytes} != {expected_bytes}"
        )
    expected_sha = str(row["sha256"]).lower()
    observed_sha = _sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"Compacted payload hash mismatch for {path}: "
            f"{observed_sha} != {expected_sha}"
        )


def repair_compacted_evidence_exclusions(
    *,
    runs_root: Path,
    evidence_dir: Path,
    cleanup_manifest: Path,
    additional_exclude_globs: Sequence[str] = (),
    repaired_at: str | None = None,
) -> Dict[str, object]:
    """Remove mistakenly retained excluded payloads after strict hash verification."""
    runs_root = Path(runs_root).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    evidence_dir = _require_child(Path(evidence_dir), runs_root, "evidence_dir")
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"Evidence directory does not exist: {evidence_dir}")
    cleanup_manifest = _require_child(
        Path(cleanup_manifest), runs_root, "cleanup_manifest"
    )
    if _is_within(cleanup_manifest, evidence_dir):
        raise ValueError("Cleanup manifest must be outside the evidence directory.")

    inventory_path = evidence_dir / "source_inventory.json"
    summary_path = evidence_dir / "summary.json"
    for required in (inventory_path, summary_path, cleanup_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"Missing compaction metadata: {required}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    if Path(str(inventory.get("output_dir", ""))).resolve() != evidence_dir:
        raise ValueError("Source inventory points at a different evidence directory.")
    if Path(str(cleanup.get("evidence_root", ""))).resolve() != evidence_dir:
        raise ValueError("Cleanup manifest points at a different evidence directory.")
    if cleanup.get("status") != "completed" or not cleanup.get("deletion_verified"):
        raise ValueError("Cleanup manifest must describe a completed verified deletion.")

    existing_globs = tuple(str(value) for value in inventory.get("exclude_globs", ()))
    exclude_globs = combine_exclude_globs(
        (*existing_globs, *tuple(additional_exclude_globs))
    )
    removed_paths: List[str] = []
    removed_bytes = 0
    planned_removals: List[tuple[Path, Dict[str, object], str]] = []
    copied_files = 0
    copied_bytes = 0
    excluded_files = 0
    excluded_bytes = 0

    sources = inventory.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source inventory has no sources.")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Source inventory entry must be an object.")
        alias = _validated_relative_path(source.get("alias"), "source alias")
        if len(alias.parts) != 1:
            raise ValueError(f"Source alias must contain one path component: {alias}")
        rows = source.get("files")
        if not isinstance(rows, list):
            raise ValueError(f"Source inventory files are invalid for alias {alias}.")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid inventory row for alias {alias}.")
            relative = _validated_relative_path(
                row.get("relative_path"), "inventory relative_path"
            )
            expected_evidence_path = alias / relative
            now_excluded = _matches_any(relative, exclude_globs)
            evidence_path_value = row.get("evidence_path")
            if now_excluded:
                excluded_files += 1
                excluded_bytes += int(row["bytes"])
                if evidence_path_value is not None:
                    evidence_relative = _validated_relative_path(
                        evidence_path_value, "inventory evidence_path"
                    )
                    if evidence_relative.as_posix() != expected_evidence_path.as_posix():
                        raise ValueError(
                            "Inventory evidence_path does not match alias/relative_path: "
                            f"{evidence_relative} != {expected_evidence_path}"
                        )
                    payload = _require_child(
                        evidence_dir / evidence_relative,
                        evidence_dir,
                        "compacted payload",
                    )
                    _verify_inventory_file(payload, row)
                    planned_removals.append(
                        (payload, row, evidence_relative.as_posix())
                    )
                row["excluded"] = True
                continue

            if evidence_path_value is None:
                raise ValueError(
                    f"Non-excluded inventory row has no evidence_path: {expected_evidence_path}"
                )
            evidence_relative = _validated_relative_path(
                evidence_path_value, "inventory evidence_path"
            )
            if evidence_relative.as_posix() != expected_evidence_path.as_posix():
                raise ValueError(
                    "Inventory evidence_path does not match alias/relative_path: "
                    f"{evidence_relative} != {expected_evidence_path}"
                )
            payload = _require_child(
                evidence_dir / evidence_relative,
                evidence_dir,
                "compacted payload",
            )
            _verify_inventory_file(payload, row)
            row["excluded"] = False
            copied_files += 1
            copied_bytes += int(row["bytes"])

    if copied_files == 0:
        raise ValueError("Repair would leave no retained evidence payloads.")

    # Verify every retained and removable payload before the first unlink.
    for payload, row, relative_text in planned_removals:
        payload.unlink()
        removed_paths.append(relative_text)
        removed_bytes += int(row["bytes"])
        row.pop("evidence_path", None)

    for directory in sorted(
        (path for path in evidence_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    repaired_at = repaired_at or datetime.now().astimezone().isoformat()
    repair_record: Dict[str, object] = {
        "repaired_at": repaired_at,
        "reason": "Restore mandatory binary exclusions after custom CLI globs replaced defaults.",
        "removed_files": len(removed_paths),
        "removed_bytes": removed_bytes,
        "removed_paths": removed_paths,
        "exclude_globs": list(exclude_globs),
        "sha256_verified_before_delete": True,
    }
    inventory["exclude_globs"] = list(exclude_globs)
    inventory["exclusion_repair"] = repair_record
    summary.update(
        {
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "excluded_files": excluded_files,
            "excluded_bytes": excluded_bytes,
            "exclude_globs": list(exclude_globs),
            "exclusion_repair": repair_record,
        }
    )
    cleanup.update(
        {
            "excluded_files": excluded_files,
            "excluded_bytes": excluded_bytes,
            "exclusion_repair": repair_record,
        }
    )
    _write_json(inventory_path, inventory)
    _write_json(summary_path, summary)
    _write_json(cleanup_manifest, cleanup)
    refreshed = refresh_compacted_evidence_metadata(evidence_dir, cleanup_manifest)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["repair_refresh"] = refreshed
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash-verify and repair exclusions in compacted run evidence."
    )
    parser.add_argument("--runs-root", type=Path, default=project_dir() / "runs")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--cleanup-manifest", type=Path, required=True)
    parser.add_argument("--exclude-glob", action="append", default=[])
    parser.add_argument("--repaired-at", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root).resolve()
    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = runs_root / evidence_dir
    cleanup_manifest = Path(args.cleanup_manifest)
    if not cleanup_manifest.is_absolute():
        cleanup_manifest = runs_root / cleanup_manifest
    result = repair_compacted_evidence_exclusions(
        runs_root=runs_root,
        evidence_dir=evidence_dir,
        cleanup_manifest=cleanup_manifest,
        additional_exclude_globs=args.exclude_glob,
        repaired_at=args.repaired_at,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
