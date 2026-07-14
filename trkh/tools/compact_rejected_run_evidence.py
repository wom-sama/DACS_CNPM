from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from trkh.core.config import project_dir


DEFAULT_EXCLUDE_GLOBS = (
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.engine",
    "*.onnx",
    "*.plan",
)


def combine_exclude_globs(additional_globs: Sequence[str] = ()) -> Tuple[str, ...]:
    """Append user exclusions to the binary defaults without case duplicates."""
    combined: List[str] = []
    seen = set()
    for pattern in (*DEFAULT_EXCLUDE_GLOBS, *tuple(additional_globs)):
        normalized = str(pattern).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(normalized)
    return tuple(combined)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _require_child(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not _is_within(resolved, root):
        raise ValueError(f"{label} must be a strict child of {root}: {resolved}")
    return resolved


def _matches_any(relative_path: Path, patterns: Sequence[str]) -> bool:
    relative = relative_path.as_posix().lower()
    name = relative_path.name.lower()
    return any(
        fnmatch.fnmatch(relative, pattern.lower())
        or fnmatch.fnmatch(name, pattern.lower())
        for pattern in patterns
    )


def _source_file_rows(source: Path, exclude_globs: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    items = sorted(source.rglob("*"))
    links = [path for path in items if path.is_symlink()]
    if links:
        raise ValueError(f"Symbolic links are not allowed in cleanup sources: {links[0]}")
    for path in (item for item in items if item.is_file()):
        relative = path.relative_to(source)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
                "excluded": _matches_any(relative, exclude_globs),
            }
        )
    return rows


def _verify_payload_manifest(evidence_dir: Path, manifest_path: Path) -> int:
    verified = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid payload manifest line: {line!r}")
        payload = evidence_dir / relative
        if not payload.is_file():
            raise FileNotFoundError(f"Missing compacted payload: {payload}")
        observed = _sha256(payload)
        if observed != digest:
            raise ValueError(
                f"Compacted payload hash mismatch for {relative}: {observed} != {digest}"
            )
        verified += 1
    if verified == 0:
        raise ValueError("Compacted payload manifest is empty.")
    return verified


def _evidence_payload_paths(evidence_dir: Path) -> List[Path]:
    excluded_root_files = {"file_manifest.sha256.txt", "summary.json"}
    return sorted(
        path
        for path in evidence_dir.rglob("*")
        if path.is_file()
        and path.relative_to(evidence_dir).as_posix() not in excluded_root_files
    )


def _write_payload_manifest(evidence_dir: Path) -> Tuple[int, str]:
    evidence_dir = Path(evidence_dir).resolve()
    manifest_path = evidence_dir / "file_manifest.sha256.txt"
    payload_paths = _evidence_payload_paths(evidence_dir)
    manifest_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(evidence_dir).as_posix()}\n"
            for path in payload_paths
        ),
        encoding="utf-8",
    )
    verified = _verify_payload_manifest(evidence_dir, manifest_path)
    return verified, _sha256(manifest_path)


def refresh_compacted_evidence_metadata(
    evidence_dir: Path,
    cleanup_manifest: Path | None = None,
) -> Dict[str, object]:
    evidence_dir = Path(evidence_dir).resolve()
    summary_path = evidence_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing evidence summary: {summary_path}")
    verified, manifest_sha256 = _write_payload_manifest(evidence_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["verified_payloads"] = verified
    summary["payload_manifest_sha256"] = manifest_sha256
    _write_json(summary_path, summary)
    if cleanup_manifest is not None:
        cleanup_manifest = Path(cleanup_manifest).resolve()
        cleanup = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
        if Path(str(cleanup.get("evidence_root", ""))).resolve() != evidence_dir:
            raise ValueError("Cleanup manifest points at a different evidence root.")
        cleanup["evidence_payloads"] = verified
        cleanup["evidence_manifest_sha256"] = manifest_sha256
        cleanup["evidence_metadata_refreshed_at"] = datetime.now().astimezone().isoformat()
        _write_json(cleanup_manifest, cleanup)
    return {
        "evidence_dir": str(evidence_dir),
        "verified_payloads": verified,
        "payload_manifest_sha256": manifest_sha256,
        "cleanup_manifest": str(cleanup_manifest) if cleanup_manifest is not None else None,
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compact_rejected_run_evidence(
    *,
    runs_root: Path,
    output_dir: Path,
    sources: Sequence[Tuple[str, Path]],
    exclude_globs: Sequence[str] = DEFAULT_EXCLUDE_GLOBS,
    notes: Sequence[str] = (),
    test_data_used: bool,
    protected_paths: Sequence[Path] = (),
    delete_sources: bool = False,
    cleanup_manifest: Path | None = None,
    created_at: str | None = None,
) -> Dict[str, object]:
    runs_root = Path(runs_root).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    output_dir = _require_child(Path(output_dir), runs_root, "output_dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evidence output must be absent or empty: {output_dir}")
    if not sources:
        raise ValueError("At least one source is required.")

    protected = [Path(path).resolve() for path in protected_paths]
    resolved_sources: List[Tuple[str, Path]] = []
    aliases = set()
    for alias, source_value in sources:
        alias = str(alias).strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", alias):
            raise ValueError(f"Invalid evidence alias: {alias!r}")
        if alias in aliases:
            raise ValueError(f"Duplicate evidence alias: {alias}")
        aliases.add(alias)
        source = _require_child(Path(source_value), runs_root, f"source {alias}")
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")
        if _is_within(output_dir, source) or _is_within(source, output_dir):
            raise ValueError(f"Evidence output and source overlap: {source}")
        for protected_path in protected:
            if _is_within(source, protected_path) or _is_within(protected_path, source):
                raise ValueError(f"Source overlaps protected path: {source} / {protected_path}")
        resolved_sources.append((alias, source))

    source_paths = [source for _, source in resolved_sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("Duplicate source directories are not allowed.")
    for index, source in enumerate(source_paths):
        for other in source_paths[index + 1 :]:
            if _is_within(source, other) or _is_within(other, source):
                raise ValueError(f"Nested source directories are not allowed: {source}, {other}")

    created_at = created_at or datetime.now().astimezone().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_sources: List[Dict[str, object]] = []
    copied_files = 0
    copied_bytes = 0
    excluded_files = 0
    excluded_bytes = 0
    for alias, source in resolved_sources:
        rows = _source_file_rows(source, exclude_globs)
        for row in rows:
            if bool(row["excluded"]):
                excluded_files += 1
                excluded_bytes += int(row["bytes"])
                continue
            relative = Path(str(row["relative_path"]))
            destination = output_dir / alias / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
            if _sha256(destination) != str(row["sha256"]):
                raise ValueError(f"Copied payload hash mismatch: {destination}")
            row["evidence_path"] = (Path(alias) / relative).as_posix()
            copied_files += 1
            copied_bytes += int(row["bytes"])
        inventory_sources.append(
            {
                "alias": alias,
                "source_path": str(source),
                "source_name": source.name,
                "file_count": len(rows),
                "total_bytes": sum(int(row["bytes"]) for row in rows),
                "files": rows,
            }
        )
    if copied_files == 0:
        raise ValueError("No non-excluded evidence payload was copied.")

    inventory = {
        "created_at": created_at,
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "exclude_globs": list(exclude_globs),
        "sources": inventory_sources,
    }
    _write_json(output_dir / "source_inventory.json", inventory)
    summary: Dict[str, object] = {
        "created_at": created_at,
        "status": "compacted_verified",
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "source_count": len(resolved_sources),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "excluded_files": excluded_files,
        "excluded_bytes": excluded_bytes,
        "exclude_globs": list(exclude_globs),
        "notes": [str(note) for note in notes],
        "test_data_used": bool(test_data_used),
    }
    _write_json(output_dir / "summary.json", summary)
    readme_lines = [
        "# Compacted Rejected-Run Evidence",
        "",
        "This directory preserves reproducible metrics, predictions, plots, XAI, trace,",
        "configuration, and provenance from rejected runs. Model/deployment binaries are",
        "listed with hashes in `source_inventory.json` but intentionally not copied.",
        "",
        *[f"- {note}" for note in notes],
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    verified_payloads, manifest_sha256 = _write_payload_manifest(output_dir)
    summary.update(
        {
            "verified_payloads": verified_payloads,
            "payload_manifest_sha256": manifest_sha256,
        }
    )

    if delete_sources:
        if cleanup_manifest is None:
            raise ValueError("cleanup_manifest is required with delete_sources=True.")
        cleanup_manifest = _require_child(
            Path(cleanup_manifest), runs_root, "cleanup_manifest"
        )
        if _is_within(cleanup_manifest, output_dir):
            raise ValueError("Cleanup manifest must be outside the evidence directory.")
        if any(_is_within(cleanup_manifest, source) for _, source in resolved_sources):
            raise ValueError("Cleanup manifest must be outside every source directory.")
        if cleanup_manifest.exists():
            raise FileExistsError(f"Cleanup manifest already exists: {cleanup_manifest}")

        for inventory_source, (_, source) in zip(inventory_sources, resolved_sources):
            current_rows = _source_file_rows(source, exclude_globs)
            expected_rows = inventory_source["files"]
            comparable = lambda rows: [
                {
                    "relative_path": row["relative_path"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                    "excluded": row["excluded"],
                }
                for row in rows
            ]
            if comparable(current_rows) != comparable(expected_rows):
                raise ValueError(f"Source changed after compaction; refusing deletion: {source}")

        cleanup_payload: Dict[str, object] = {
            "created_at": created_at,
            "status": "verified_pending_delete",
            "runs_root": str(runs_root),
            "evidence_root": str(output_dir),
            "evidence_payloads": verified_payloads,
            "evidence_manifest_sha256": manifest_sha256,
            "source_roots": [
                {
                    "alias": item["alias"],
                    "name": item["source_name"],
                    "path": item["source_path"],
                    "files": item["file_count"],
                    "bytes": item["total_bytes"],
                }
                for item in inventory_sources
            ],
            "compacted_names": [item["source_name"] for item in inventory_sources],
            "excluded_files": excluded_files,
            "excluded_bytes": excluded_bytes,
            "deletion_verified": False,
        }
        _write_json(cleanup_manifest, cleanup_payload)
        reread = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
        if reread.get("status") != "verified_pending_delete":
            raise ValueError("Cleanup manifest verification failed before deletion.")

        free_before = shutil.disk_usage(runs_root).free
        for _, source in resolved_sources:
            shutil.rmtree(source)
        remaining = [str(source) for _, source in resolved_sources if source.exists()]
        if remaining:
            raise RuntimeError(f"Compacted source directories remain after deletion: {remaining}")
        free_after = shutil.disk_usage(runs_root).free
        cleanup_payload.update(
            {
                "status": "completed",
                "deleted_at": datetime.now().astimezone().isoformat(),
                "deletion_verified": True,
                "observed_free_bytes_delta": int(free_after - free_before),
            }
        )
        _write_json(cleanup_manifest, cleanup_payload)
        summary["cleanup_manifest"] = str(cleanup_manifest)
        summary["deleted_sources"] = len(resolved_sources)

    _write_json(output_dir / "summary.json", summary)
    # Reverify retained payloads after any optional source deletion. The root
    # summary is derived metadata and is intentionally outside this manifest.
    final_verified, final_manifest_sha256 = _write_payload_manifest(output_dir)
    summary["verified_payloads"] = final_verified
    summary["payload_manifest_sha256"] = final_manifest_sha256
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_source(value: str, runs_root: Path) -> Tuple[str, Path]:
    alias, separator, source_text = str(value).partition("=")
    if not separator or not alias.strip() or not source_text.strip():
        raise argparse.ArgumentTypeError("--source must use ALIAS=PATH.")
    source = Path(source_text.strip())
    if not source.is_absolute():
        source = runs_root / source
    return alias.strip(), source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact rejected run evidence, verify hashes, then optionally delete sources."
    )
    parser.add_argument("--runs-root", type=Path, default=project_dir() / "runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="ALIAS=PATH")
    parser.add_argument("--exclude-glob", action="append", default=None)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--test-data-used",
        choices=("yes", "no"),
        required=True,
        help="Explicit leakage declaration for the compacted sources.",
    )
    parser.add_argument("--protected-path", type=Path, action="append", default=[])
    parser.add_argument("--delete-sources", action="store_true")
    parser.add_argument("--cleanup-manifest", type=Path, default=None)
    parser.add_argument("--created-at", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = runs_root / output_dir
    cleanup_manifest = args.cleanup_manifest
    if cleanup_manifest is not None and not Path(cleanup_manifest).is_absolute():
        cleanup_manifest = runs_root / Path(cleanup_manifest)
    sources = [_parse_source(value, runs_root) for value in args.source]
    summary = compact_rejected_run_evidence(
        runs_root=runs_root,
        output_dir=output_dir,
        sources=sources,
        exclude_globs=combine_exclude_globs(args.exclude_glob or ()),
        notes=args.note,
        test_data_used=args.test_data_used == "yes",
        protected_paths=args.protected_path,
        delete_sources=bool(args.delete_sources),
        cleanup_manifest=cleanup_manifest,
        created_at=args.created_at,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
