from __future__ import annotations

"""Execute one exact, hash-locked cleanup manifest with fail-closed guards."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence


class CleanupContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _within(path: Path, root: Path) -> bool:
    value = _normal(path)
    prefix = _normal(root).rstrip("\\/")
    return value == prefix or value.startswith(prefix + os.sep)


def _absolute(raw: object) -> Path:
    text = str(raw)
    if not text or any(character in text for character in "*?[]"):
        raise CleanupContractError(f"Cleanup path is empty or contains a wildcard: {text!r}")
    path = Path(text)
    if not path.is_absolute():
        raise CleanupContractError(f"Cleanup path is not absolute: {text}")
    return Path(os.path.abspath(os.fspath(path)))


def _flatten_targets(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    groups = manifest.get("delete_groups")
    if not isinstance(groups, list):
        raise CleanupContractError("delete_groups must be a list")
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("entries"), list):
            raise CleanupContractError("Every delete group must contain an entries list")
        for entry in group["entries"]:
            if not isinstance(entry, Mapping):
                raise CleanupContractError("Cleanup entry must be an object")
            targets.append(dict(entry))
    cache = manifest.get("cache_directories")
    if not isinstance(cache, Mapping) or not isinstance(cache.get("paths"), list):
        raise CleanupContractError("cache_directories.paths must be a list")
    targets.extend({"path": value, "kind": "directory", "cache": True} for value in cache["paths"])
    return targets


def validate_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != str(expected_sha256).strip().lower():
        raise CleanupContractError(
            f"Manifest SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or int(manifest.get("schema_version", 0)) != 1:
        raise CleanupContractError("Unsupported cleanup manifest schema")
    if manifest.get("status") != "approved_ready":
        raise CleanupContractError("Cleanup manifest is not approved_ready")

    root = _absolute(manifest.get("workspace_root"))
    expected = _absolute(expected_root)
    if _normal(root) != _normal(expected):
        raise CleanupContractError(f"Workspace root mismatch: manifest={root} expected={expected}")
    if not root.is_dir():
        raise CleanupContractError(f"Workspace root is missing: {root}")

    protected = [_absolute(value) for value in manifest.get("protected_dataset_roots", [])]
    targets = _flatten_targets(manifest)
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for entry in targets:
        path = _absolute(entry.get("path"))
        key = _normal(path)
        if key in seen:
            raise CleanupContractError(f"Duplicate cleanup path: {path}")
        seen.add(key)
        if not _within(path, root):
            raise CleanupContractError(f"Cleanup path is outside workspace root: {path}")
        if any(part.casefold() == ".git" for part in path.parts):
            raise CleanupContractError(f"Git metadata cleanup is forbidden: {path}")
        if any(_within(path, protected_root) for protected_root in protected):
            raise CleanupContractError(f"Dataset cleanup is forbidden: {path}")
        if path.is_symlink():
            raise CleanupContractError(f"Symlink cleanup target is forbidden: {path}")

        kind = str(entry.get("kind", ""))
        if kind not in {"file", "directory"}:
            raise CleanupContractError(f"Invalid cleanup kind for {path}: {kind}")
        exists = path.exists()
        if exists:
            resolved = path.resolve(strict=True)
            if not _within(resolved, root):
                raise CleanupContractError(f"Resolved cleanup path escaped workspace: {path} -> {resolved}")
            if (kind == "directory") != path.is_dir():
                raise CleanupContractError(f"Cleanup type mismatch: {path}")
            if kind == "file":
                expected_bytes = int(entry.get("logical_bytes", -1))
                if path.stat().st_size != expected_bytes:
                    raise CleanupContractError(f"Cleanup file size changed: {path}")
                expected_hash = str(entry.get("sha256", "")).strip().lower()
                if expected_hash and sha256_file(path) != expected_hash:
                    raise CleanupContractError(f"Cleanup file hash changed: {path}")
        checked.append({**entry, "path": str(path), "exists": exists})
    return manifest, checked


def _remove_readonly(function: Any, path: str, _error: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def execute_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_root: Path,
    execute: bool,
) -> dict[str, Any]:
    manifest, targets = validate_manifest(
        manifest_path,
        expected_sha256=expected_sha256,
        expected_root=expected_root,
    )
    present = [entry for entry in targets if bool(entry["exists"])]
    absent = [str(entry["path"]) for entry in targets if not bool(entry["exists"])]
    deleted: list[str] = []
    if execute:
        for entry in sorted(present, key=lambda item: len(Path(str(item["path"])).parts), reverse=True):
            path = Path(str(entry["path"]))
            if str(entry["kind"]) == "directory":
                shutil.rmtree(path, onerror=_remove_readonly)
            else:
                path.unlink()
            if os.path.lexists(path):
                raise CleanupContractError(f"Cleanup target still exists after deletion: {path}")
            deleted.append(str(path))
    return {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path.expanduser().resolve(strict=True)),
        "mode": "execute" if execute else "dry_run",
        "planned_targets": len(targets),
        "present_before": len(present),
        "already_absent": absent,
        "deleted": deleted,
        "protected_dataset_roots": manifest["protected_dataset_roots"],
    }


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        result = execute_manifest(
            args.manifest,
            expected_sha256=args.expected_manifest_sha256,
            expected_root=args.expected_root,
            execute=bool(args.execute),
        )
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 2
    print(json.dumps({"status": "ok", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
