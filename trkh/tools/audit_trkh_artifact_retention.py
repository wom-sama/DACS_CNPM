from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


DEFAULT_GUARDRAIL = (
    "Artifact-retention audit only. No deletion, raw-data edits, test tuning, "
    "training launch, relabeling, sample weights, soft targets, or trainable manifest."
)

DEFAULT_PROTECTED_RELATIVE_PATHS = (
    r"runs\yolof_oof_folds_train5_20260704\summary.json",
    r"runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    r"runs\eval_patch_linear_verifier_softboost001_full_val_20260705\metrics_detailed.json",
    r"runs\eval_yolof_teacherfocusbinary015_best_test_final_20260702\metrics_detailed.json",
    r"runs\audit_trkh_smoke_gate_current_rerun_20260706\summary.json",
    r"runs\diagnostic_trkh_signal_gap_readiness_current_20260706\summary.json",
    r"runs\review_worklist_readiness_matrix_20260706\summary.json",
    r"runs\review_minimum_fill_plan_current_20260706\summary.json",
    r"runs\review_minimum_fill_plan_current_20260706\priority_fill_plan_review.html",
    r"runs\evidence_embedding_retrieval_legacy_20260706",
    r"runs\evidence_legacy_quality_boundary_reviews_20260706",
    r"runs\evidence_legacy_selectors_posthoc_20260706",
    r"runs\full_v8_yolof_current_best_30e_20260728_184850\summary.json",
    r"runs\full_v8_yolof_randominit_30e_20260714_105524\summary.json",
    r"runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt",
    r"runs\probe_v8_scratch_natural_120b_10e_20260728_223907\summary.json",
    r"runs\paper_pretrained_compare_val_reuse_20260729_001455\RUN_README.md",
    r"runs\audit_cross_colour_ratio_surface_a0_materialized_20260725\summary.json",
    r"runs\audit_pair_surface_ddf_a0_synthetic_20260729_003200\summary.json",
    r"runs\audit_pair_surface_ddf_v1_fold_leakage_20260729\summary.json",
    r"runs\audit_pair_surface_ddf_v2_geometry_metadata_20260729\summary.json",
    r"runs\presentation_keeper_bundle_20260729\source_snapshot_summary.json",
    r"docs\TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json",
    r"..\newdataset\class_f\data.yaml",
    r"..\newdataset\yolo_f\data.yaml",
)

# These locks intentionally cover files outside the TRKH Git repository.  They
# are comparison/provenance inputs, not trainable TRKH parameters.  Byte counts
# make truncation visible before the (authoritative) SHA-256 comparison.
DEFAULT_ESSENTIAL_FILE_LOCKS: tuple[dict[str, object], ...] = (
    {
        "role": "trkh_keeper_checkpoint",
        "path": r"runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
        "bytes": 58190990,
        "sha256": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    },
    {
        "role": "trkh_random_init_full_checkpoint",
        "path": r"runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt",
        "bytes": 58209870,
        "sha256": "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549",
    },
    {
        "role": "pretrained_resnet50_checkpoint",
        "path": r"D:\DataAI\AIEx\image_baseline_experiments\outputs\resnet50\best.pt",
        "bytes": 94375846,
        "sha256": "abf96e70735dcb9d37911776dfe759428450c35615fb707ccf48bc941a717929",
    },
    {
        "role": "pretrained_mobilenetv3_checkpoint",
        "path": r"D:\DataAI\AIEx\image_baseline_experiments\outputs\mobilenetv3\best.pt",
        "bytes": 17026702,
        "sha256": "9d04d979aff163e4ff1f06996f9a6084ef5b48ca10033abe7d7150d7cf54eef4",
    },
    {
        "role": "pretrained_efficientnetv2_s_checkpoint",
        "path": r"D:\DataAI\AIEx\image_baseline_experiments\outputs\efficientnetv2_s\best.pt",
        "bytes": 81583520,
        "sha256": "0b581a8f264007407db1d88927bf2539c9eaf8bca00b300d9e0381d8dfc92cf9",
    },
    {
        "role": "pretrained_convnext_tiny_checkpoint",
        "path": r"D:\DataAI\AIEx\image_baseline_experiments\outputs\convnext_tiny\best.pt",
        "bytes": 111354200,
        "sha256": "0b0613e2e96ef6c9569726bcd1a33b64a1d21f7164c4c6b196d15e81bb143650",
    },
    {
        "role": "pretrained_aidt_fusion_checkpoint",
        "path": r"D:\DataAI\AIDT\runs\resnet50_vit_b16_class_f_5class_pretrained\best.pt",
        "bytes": 451259634,
        "sha256": "f902cefc19218ca4db8b5acfd998e6fea8ff6c3dfa30070aa91550b189cf7a4f",
    },
    {
        "role": "cvat_yolo_exporter",
        "path": r"D:\DataAI\AIEx\CVAT_\src\cvat_nhai\yolo_editor.py",
        "bytes": 44941,
        "sha256": "65e6056dac1793eaa9775b458a448828bfd0debd7adc4d9acb58807b7b74e4b5",
    },
    {
        "role": "canonical_yolo_manifest",
        "path": r"..\newdataset\yolo_f\manifest.csv",
        "bytes": 2391429,
        "sha256": "eb16e09cd20fff8480b50aa79e7d5fec779e4614398911e2e257ba6a37642ff2",
    },
)

# A root lock is the SHA-256 of canonical compact JSON containing every regular
# file as {relative_path, bytes, sha256}, sorted by case-preserving POSIX path.
# Directories themselves are intentionally not hashed; any extra/missing file,
# byte change, or path rename changes at least one pinned value below.
DEFAULT_PRESENTATION_ROOT_LOCKS: tuple[dict[str, object], ...] = (
    {
        "role": "trkh_keeper",
        "path": r"runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701",
        "file_count": 460,
        "total_bytes": 363864736,
        "inventory_sha256": "bfa7ceb1c7d1d079f73e7a00c5e26bc959a9f49d77d0b7add99b12d2806c42e3",
    },
    {
        "role": "trkh_random_init_full",
        "path": r"runs\full_v8_yolof_randominit_30e_20260714_105524",
        "file_count": 1860,
        "total_bytes": 306449952,
        "inventory_sha256": "22cd65c8849b60745434dae92b20ad971835b70b9a0680f5ae56d97dda6159fe",
    },
    {
        "role": "warm_start_plateau_diagnostic",
        "path": r"runs\full_v8_yolof_current_best_30e_20260728_184850",
        "file_count": 1411,
        "total_bytes": 373476714,
        "inventory_sha256": "aa633db24ac8f8d949465073c364729e9a3034c9bf0d1f1c3402b67a5aae43ed",
    },
    {
        "role": "natural_only_negative_control",
        "path": r"runs\probe_v8_scratch_natural_120b_10e_20260728_223907",
        "file_count": 1089,
        "total_bytes": 346326192,
        "inventory_sha256": "cf78a91f19681e69fa1b27c7efff31a1bcef68127edda0a2dca8c08663bbc09b",
    },
    {
        "role": "seven_model_validation_comparison",
        "path": r"runs\paper_pretrained_compare_val_reuse_20260729_001455",
        "file_count": 142,
        "total_bytes": 19671898,
        "inventory_sha256": "fc4f5741a3a907e76ee7259cb709ed3da46bd81c864c0a6116d2765d28fc72b4",
    },
    {
        "role": "ccr_materialized_cache",
        "path": r"runs\audit_cross_colour_ratio_surface_a0_materialized_20260725",
        "file_count": 8,
        "total_bytes": 157897360,
        "inventory_sha256": "12beb1f70abfeceff2f74945a931168f5f1d46779a5fc52dd5d0abe5764b3d77",
    },
    {
        "role": "ddf_synthetic_engineering",
        "path": r"runs\audit_pair_surface_ddf_a0_synthetic_20260729_003200",
        "file_count": 1,
        "total_bytes": 19779,
        "inventory_sha256": "760c9f71b782b5da3833c67ab17dd786b056a8742c26ee76a9aa9843b476dccd",
    },
    {
        "role": "ddf_v1_fold_leakage_audit",
        "path": r"runs\audit_pair_surface_ddf_v1_fold_leakage_20260729",
        "file_count": 1,
        "total_bytes": 679293,
        "inventory_sha256": "03f45df65f82364799738d24a24eeec78e24046a597328f878f2474597081a10",
    },
    {
        "role": "ddf_v2_geometry_metadata_audit",
        "path": r"runs\audit_pair_surface_ddf_v2_geometry_metadata_20260729",
        "file_count": 2,
        "total_bytes": 6980,
        "inventory_sha256": "0d73969e818b3fb632c3bb5ec76d3b373878cb11ffb7d0c1a6d8585a77b925dd",
    },
    {
        "role": "presentation_external_source_bundle",
        "path": r"runs\presentation_keeper_bundle_20260729",
        "file_count": 33,
        "total_bytes": 212697,
        "inventory_sha256": "d0a7cba8814eae6671fb43fdce1cb9840a8fc6616cb977d5a94a4f13364f0079",
    },
)

DEFAULT_ABSENT_PREFIXES = (
    "eval_smoke_",
    "oof_mobilenetv3_fold",
)

DEFAULT_COMPACTION_MANIFEST_NAMES = (
    "cleanup_manifest_20260706_legacy_embedding_retrieval_compacted.json",
    "cleanup_manifest_20260706_legacy_quality_boundary_reviews_compacted.json",
    "cleanup_manifest_20260706_legacy_selectors_posthoc_compacted.json",
)

KEEP_EXACT_REASONS: Dict[str, tuple[str, str]] = {
    "yolof_oof_folds_train5_20260704": (
        "KEEP_LIVE",
        "current grouped yolo_f OOF materialization for fold-safe diagnostics",
    ),
    "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701": (
        "KEEP_LIVE",
        "best no-pretrain TRKH validation anchor and checkpoint",
    ),
    "eval_patch_linear_verifier_softboost001_full_val_20260705": (
        "KEEP_CURRENT_DIAGNOSTIC",
        "current best TRKH-native validation hook and conservative false-positive diagnostic",
    ),
    "eval_yolof_teacherfocusbinary015_best_test_final_20260702": (
        "KEEP_FINAL",
        "final no-pretrain TRKH test audit; do not tune from it",
    ),
    "patch_evidence_mil_yolof_keeper_01only_t060_m040_top4_cropbbox_test_final_20260703": (
        "KEEP_FINAL_DIAGNOSTIC",
        "fixed patch-evidence final diagnostic reference; do not tune thresholds from it",
    ),
    "audit_trkh_smoke_gate_current_rerun_20260706": (
        "KEEP_CURRENT_GATE",
        "current pre-smoke gate after corrected signal-gap rerun",
    ),
    "diagnostic_trkh_signal_gap_readiness_current_20260706": (
        "KEEP_CURRENT_DIAGNOSTIC",
        "current signal-gap crosswalk with rival2/4 review coverage",
    ),
    "review_worklist_readiness_matrix_20260706": (
        "KEEP_CURRENT_REVIEW",
        "current manual-review readiness matrix",
    ),
    "review_minimum_fill_plan_current_20260706": (
        "KEEP_CURRENT_REVIEW",
        "current priority manual-review packet",
    ),
    "evidence_embedding_retrieval_legacy_20260706": (
        "KEEP_COMPACT_EVIDENCE",
        "compacted legacy embedding-retrieval evidence root",
    ),
    "evidence_legacy_quality_boundary_reviews_20260706": (
        "KEEP_COMPACT_EVIDENCE",
        "compacted legacy quality/boundary review evidence root",
    ),
    "evidence_legacy_selectors_posthoc_20260706": (
        "KEEP_COMPACT_EVIDENCE",
        "compacted legacy selector/posthoc evidence root",
    ),
    "full_v8_yolof_current_best_30e_20260728_184850": (
        "KEEP_REJECTED_DIAGNOSTIC",
        "pre-fix warm-start run retained to explain the apparent early peak and plateau",
    ),
    "full_v8_yolof_randominit_30e_20260714_105524": (
        "KEEP_FULL_SCRATCH_BASELINE",
        "completed strict-balanced random-init baseline with checkpoint and final audits",
    ),
    "probe_v8_scratch_natural_120b_10e_20260728_223907": (
        "KEEP_NEGATIVE_CONTROL",
        "completed natural-only scratch negative control; do not resume or extend",
    ),
    "paper_pretrained_compare_val_reuse_20260729_001455": (
        "KEEP_PRESENTATION_COMPARISON",
        "validation-only seven-model comparison with locked checkpoint hashes",
    ),
    "audit_cross_colour_ratio_surface_a0_materialized_20260725": (
        "KEEP_FORMAL_CACHE_DEPENDENCY",
        "exact replayed train-only materialization retained for prospective mechanism audits",
    ),
    "audit_pair_surface_ddf_a0_synthetic_20260729_003200": (
        "KEEP_ENGINEERING_EVIDENCE",
        "Pair-Surface DDF synthetic engineering evidence; mechanism remains unproven",
    ),
    "audit_pair_surface_ddf_v1_fold_leakage_20260729": (
        "KEEP_PROTOCOL_SUPERSESSION_EVIDENCE",
        "metadata-only evidence that supersedes the original DDF A0 folds before formal execution",
    ),
    "audit_pair_surface_ddf_v2_geometry_metadata_20260729": (
        "KEEP_PROSPECTIVE_PROTOCOL_EVIDENCE",
        "locked train-table geometry preflight for the validity-aware DDF v2 protocol",
    ),
    "presentation_keeper_bundle_20260729": (
        "KEEP_PRESENTATION_BUNDLE",
        "source-only snapshots and hashes for external comparison repositories that are not under Git",
    ),
}


def _repo_root_from_runs_root(runs_root: Path) -> Path:
    runs_root = Path(runs_root).resolve()
    if runs_root.name.lower() == "runs":
        return runs_root.parent
    return runs_root


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _size_stats(path: Path) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            try:
                total_size += item.stat().st_size
            except OSError:
                continue
    return total_size, file_count


def _round_mb(size_bytes: int) -> float:
    return round(float(size_bytes) / (1024.0 * 1024.0), 3)


def _safe_read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(files: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(files),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_file_inventory(root: Path) -> dict[str, object]:
    """Hash a tree without following links and retain every file-level fact."""

    root = Path(root)
    files: list[dict[str, object]] = []
    symlinks: list[str] = []
    special_entries: list[str] = []
    read_errors: list[dict[str, str]] = []

    if _symlink_path_components(root):
        if root.is_symlink():
            symlinks.append(".")
        return {
            "files": files,
            "file_count": 0,
            "total_bytes": 0,
            "inventory_sha256": _inventory_sha256(files),
            "symlinks": symlinks,
            "special_entries": special_entries,
            "read_errors": read_errors,
        }
    if not root.exists() or not root.is_dir():
        return {
            "files": files,
            "file_count": 0,
            "total_bytes": 0,
            "inventory_sha256": _inventory_sha256(files),
            "symlinks": symlinks,
            "special_entries": special_entries,
            "read_errors": read_errors,
        }

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            with os.scandir(directory) as handle:
                entries = sorted(handle, key=lambda entry: entry.name)
        except OSError as exc:
            relative = relative_directory.as_posix() if relative_directory.parts else "."
            read_errors.append({"relative_path": relative, "error": f"{type(exc).__name__}: {exc}"})
            return

        for entry in entries:
            relative = relative_directory / entry.name
            relative_text = relative.as_posix()
            try:
                if entry.is_symlink():
                    symlinks.append(relative_text)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), relative)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    special_entries.append(relative_text)
                    continue

                before = os.stat(entry.path, follow_symlinks=False)
                sha256 = _sha256_file(Path(entry.path))
                after = os.stat(entry.path, follow_symlinks=False)
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity:
                    read_errors.append(
                        {
                            "relative_path": relative_text,
                            "error": "file_changed_during_inventory",
                        }
                    )
                    continue
                files.append(
                    {
                        "relative_path": relative_text,
                        "bytes": int(before.st_size),
                        "sha256": sha256,
                    }
                )
            except OSError as exc:
                read_errors.append(
                    {
                        "relative_path": relative_text,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    visit(root, Path())
    files.sort(key=lambda item: str(item["relative_path"]))
    symlinks.sort()
    special_entries.sort()
    read_errors.sort(key=lambda item: (item["relative_path"], item["error"]))
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "inventory_sha256": _inventory_sha256(files),
        "symlinks": symlinks,
        "special_entries": special_entries,
        "read_errors": read_errors,
    }


def _locked_path(repo_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else repo_root / path


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _symlink_path_components(path: Path) -> list[str]:
    absolute = _lexical_absolute(path)
    candidates = list(reversed((absolute, *absolute.parents)))
    return [str(candidate) for candidate in candidates if candidate.is_symlink()]


def _essential_file_checks(
    repo_root: Path,
    locks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for lock in locks:
        path = _locked_path(repo_root, lock["path"])
        is_symlink = path.is_symlink()
        symlink_path_components = _symlink_path_components(path)
        exists = path.exists()
        is_file = exists and path.is_file() and not symlink_path_components
        expected_bytes = int(lock["bytes"])
        expected_sha256 = str(lock["sha256"]).lower()
        observed_bytes: Optional[int] = None
        observed_sha256: Optional[str] = None
        read_error: Optional[str] = None
        if is_file:
            try:
                before = path.stat()
                observed_sha256 = _sha256_file(path)
                after = path.stat()
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    read_error = "file_changed_during_hash"
                    observed_sha256 = None
                observed_bytes = int(after.st_size)
            except OSError as exc:
                read_error = f"{type(exc).__name__}: {exc}"

        bytes_match = observed_bytes == expected_bytes
        sha256_match = observed_sha256 == expected_sha256
        passed = bool(
            exists
            and is_file
            and not symlink_path_components
            and read_error is None
            and bytes_match
            and sha256_match
        )
        checks.append(
            {
                "role": str(lock.get("role", "")),
                "path": str(lock["path"]),
                "resolved_path": str(_lexical_absolute(path)),
                "exists": exists,
                "is_file": is_file,
                "is_symlink": is_symlink,
                "symlink_path_components": symlink_path_components,
                "expected_bytes": expected_bytes,
                "observed_bytes": observed_bytes,
                "bytes_match": bytes_match,
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
                "sha256_match": sha256_match,
                "read_error": read_error,
                "passed": passed,
            }
        )
    return checks


def _presentation_root_checks(
    repo_root: Path,
    locks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for lock in locks:
        path = _locked_path(repo_root, lock["path"])
        is_symlink = path.is_symlink()
        symlink_path_components = _symlink_path_components(path)
        exists = path.exists()
        is_dir = exists and path.is_dir() and not symlink_path_components
        inventory = _scan_file_inventory(path)
        expected_file_count = int(lock["file_count"])
        expected_total_bytes = int(lock["total_bytes"])
        expected_inventory_sha256 = str(lock["inventory_sha256"]).lower()
        file_count_match = int(inventory["file_count"]) == expected_file_count
        total_bytes_match = int(inventory["total_bytes"]) == expected_total_bytes
        inventory_sha256_match = str(inventory["inventory_sha256"]) == expected_inventory_sha256
        passed = bool(
            exists
            and is_dir
            and not symlink_path_components
            and not inventory["symlinks"]
            and not inventory["special_entries"]
            and not inventory["read_errors"]
            and file_count_match
            and total_bytes_match
            and inventory_sha256_match
        )
        checks.append(
            {
                "role": str(lock.get("role", "")),
                "path": str(lock["path"]),
                "resolved_path": str(_lexical_absolute(path)),
                "exists": exists,
                "is_dir": is_dir,
                "is_symlink": is_symlink,
                "symlink_path_components": symlink_path_components,
                "expected_file_count": expected_file_count,
                "observed_file_count": int(inventory["file_count"]),
                "file_count_match": file_count_match,
                "expected_total_bytes": expected_total_bytes,
                "observed_total_bytes": int(inventory["total_bytes"]),
                "total_bytes_match": total_bytes_match,
                "expected_inventory_sha256": expected_inventory_sha256,
                "observed_inventory_sha256": str(inventory["inventory_sha256"]),
                "inventory_sha256_match": inventory_sha256_match,
                "symlinks": inventory["symlinks"],
                "special_entries": inventory["special_entries"],
                "read_errors": inventory["read_errors"],
                "files": inventory["files"],
                "passed": passed,
            }
        )
    return checks


def _manifest_compacted_names(manifest_paths: Iterable[Path]) -> dict[str, str]:
    names: dict[str, str] = {}
    for manifest in manifest_paths:
        if not Path(manifest).is_file():
            continue
        payload = _safe_read_json(Path(manifest))
        compacted = payload.get("compacted_names", [])
        if not isinstance(compacted, list):
            continue
        for name in compacted:
            text = str(name).strip()
            if text:
                names[text] = Path(manifest).name
    return names


def _classify_directory(name: str, compacted_source_names: Mapping[str, str]) -> tuple[str, str]:
    if name in compacted_source_names:
        return (
            "STALE_COMPACTED_ORIGINAL_PRESENT",
            f"original source directory listed in {compacted_source_names[name]} still exists",
        )
    if name in KEEP_EXACT_REASONS:
        return KEEP_EXACT_REASONS[name]
    if name.startswith("artifact_retention_audit_"):
        return ("KEEP_CURRENT_INVENTORY", "artifact-retention audit output")
    if name.startswith("evidence_"):
        return ("KEEP_COMPACT_EVIDENCE", "compacted evidence root")
    if name.startswith(("top5_", "aidt_", "efficientnet", "dinov2_", "pretrained_", "router_exports_")):
        return ("KEEP_TEACHER_DIAGNOSTIC", "external/teacher diagnostic or comparison artifact")
    if name.startswith(("manual_", "review_", "diagnostic_")):
        return ("KEEP_RECENT_DIAGNOSTIC_OR_REVIEW", "diagnostic/review artifact; keep unless a later audit supersedes it")
    if name.startswith(("xai_", "transition_summary_")):
        return ("KEEP_XAI_REVIEW", "XAI or visual review evidence")
    if name.startswith(("embedding_retrieval_", "selector_", "focus_class1_specialist_", "posthoc_q34", "posthoc_scratch_q34")):
        return (
            "FUTURE_CLEANUP_REVIEW",
            "historical candidate; compact separately after preserving summaries and checking docs",
        )
    if name.startswith(("quality_group", "quality_groups", "quality_cartography", "boundary_review_v16", "boundary_review_groupclean")):
        return (
            "FUTURE_CLEANUP_REVIEW",
            "historical quality/boundary review candidate; compact separately after evidence preservation",
        )
    return ("REVIEW_REQUIRED", "not automatically classified; keep until manually audited")


def _directory_rows(
    runs_root: Path,
    repo_root: Path,
    compacted_source_names: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    for child in sorted(runs_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        size_bytes, file_count = _size_stats(child)
        decision, reason = _classify_directory(child.name, compacted_source_names)
        try:
            last_write = datetime.fromtimestamp(child.stat().st_mtime).isoformat(timespec="seconds")
        except OSError:
            last_write = ""
        rows.append(
            {
                "name": child.name,
                "path": str(child.resolve()),
                "repo_relative_path": _to_repo_relative(child, repo_root),
                "size_bytes": int(size_bytes),
                "size_mb": _round_mb(size_bytes),
                "file_count": int(file_count),
                "last_write_time": last_write,
                "retention_decision": decision,
                "retention_reason": reason,
            }
        )
    rows.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    return rows


def _decision_counts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    counts: Counter[str] = Counter(str(row.get("retention_decision", "")) for row in rows)
    size_by_decision: Counter[str] = Counter()
    for row in rows:
        size_by_decision[str(row.get("retention_decision", ""))] += int(row.get("size_bytes", 0))
    return [
        {
            "decision": decision,
            "count": int(counts[decision]),
            "size_mb": _round_mb(int(size_by_decision[decision])),
        }
        for decision, _ in counts.most_common()
    ]


def _protected_checks(repo_root: Path, relative_paths: Sequence[str]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for relative in relative_paths:
        path = repo_root / relative
        symlink_path_components = _symlink_path_components(path)
        checks.append(
            {
                "path": relative,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "is_file": path.is_file(),
                "is_symlink": path.is_symlink(),
                "symlink_path_components": symlink_path_components,
            }
        )
    return checks


def _absent_prefix_checks(runs_root: Path, prefixes: Sequence[str]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for prefix in prefixes:
        remaining = sorted(child.name for child in runs_root.iterdir() if child.is_dir() and child.name.startswith(prefix))
        checks.append({"prefix": prefix, "remaining_count": len(remaining), "remaining_names": remaining[:50]})
    return checks


def _compacted_original_checks(runs_root: Path, compacted_source_names: Mapping[str, str]) -> dict[str, object]:
    remaining: list[dict[str, str]] = []
    for name, manifest_name in sorted(compacted_source_names.items()):
        if (runs_root / name).exists():
            remaining.append({"name": name, "manifest": manifest_name})
    by_manifest: Counter[str] = Counter(row["manifest"] for row in remaining)
    return {
        "expected_absent_count": len(compacted_source_names),
        "remaining_original_count": len(remaining),
        "remaining_by_manifest": dict(by_manifest),
        "remaining_originals": remaining,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def audit_trkh_artifact_retention(
    *,
    runs_root: Path,
    output_dir: Path,
    protected_relative_paths: Sequence[str] = DEFAULT_PROTECTED_RELATIVE_PATHS,
    essential_file_locks: Sequence[Mapping[str, object]] = DEFAULT_ESSENTIAL_FILE_LOCKS,
    presentation_root_locks: Sequence[Mapping[str, object]] = DEFAULT_PRESENTATION_ROOT_LOCKS,
    absent_prefixes: Sequence[str] = DEFAULT_ABSENT_PREFIXES,
    compaction_manifests: Optional[Sequence[Path]] = None,
    top_directory_count: int = 120,
    created_at: Optional[str] = None,
) -> Dict[str, object]:
    runs_root = Path(runs_root).resolve()
    repo_root = _repo_root_from_runs_root(runs_root)
    if compaction_manifests is None:
        compaction_manifests = tuple(runs_root / name for name in DEFAULT_COMPACTION_MANIFEST_NAMES)
    compacted_source_names = _manifest_compacted_names(compaction_manifests)

    directory_rows = _directory_rows(runs_root, repo_root, compacted_source_names)
    largest_rows = directory_rows[: max(0, int(top_directory_count))]
    cleanup_candidates = [
        row
        for row in directory_rows
        if str(row.get("retention_decision")) in {"FUTURE_CLEANUP_REVIEW", "STALE_COMPACTED_ORIGINAL_PRESENT"}
    ]
    compacted_checks = _compacted_original_checks(runs_root, compacted_source_names)
    protected = _protected_checks(repo_root, protected_relative_paths)
    essential_file_checks = _essential_file_checks(repo_root, essential_file_locks)
    presentation_root_checks = _presentation_root_checks(repo_root, presentation_root_locks)
    absent = _absent_prefix_checks(runs_root, absent_prefixes)
    free = shutil.disk_usage(runs_root).free

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "largest_directories.csv",
        largest_rows,
        (
            "name",
            "repo_relative_path",
            "size_mb",
            "file_count",
            "last_write_time",
            "retention_decision",
            "retention_reason",
        ),
    )
    _write_csv(
        output_dir / "future_cleanup_review_candidates.csv",
        cleanup_candidates,
        (
            "name",
            "repo_relative_path",
            "size_mb",
            "file_count",
            "last_write_time",
            "retention_decision",
            "retention_reason",
        ),
    )
    presentation_inventory_rows = [
        {
            "role": check["role"],
            "root_path": check["path"],
            "relative_path": item["relative_path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for check in presentation_root_checks
        for item in check["files"]
    ]
    _write_csv(
        output_dir / "presentation_root_file_inventory.csv",
        presentation_inventory_rows,
        ("role", "root_path", "relative_path", "bytes", "sha256"),
    )

    blockers: list[str] = []

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if any(not bool(item["exists"]) for item in protected):
        block("missing_protected_artifact")
    if any(bool(item["symlink_path_components"]) for item in protected):
        block("protected_artifact_symlink_detected")
    if any(not bool(item["exists"]) for item in essential_file_checks):
        block("essential_file_missing")
    if any(bool(item["exists"]) and not bool(item["is_file"]) for item in essential_file_checks):
        block("essential_file_type_mismatch")
    if any(bool(item["symlink_path_components"]) for item in essential_file_checks):
        block("essential_file_symlink_detected")
    if any(item["read_error"] is not None for item in essential_file_checks):
        block("essential_file_read_error")
    if any(item["observed_bytes"] is not None and not bool(item["bytes_match"]) for item in essential_file_checks):
        block("essential_file_size_mismatch")
    if any(item["observed_sha256"] is not None and not bool(item["sha256_match"]) for item in essential_file_checks):
        block("essential_file_sha256_mismatch")
    if any(not bool(item["exists"]) for item in presentation_root_checks):
        block("presentation_root_missing")
    if any(bool(item["exists"]) and not bool(item["is_dir"]) for item in presentation_root_checks):
        block("presentation_root_type_mismatch")
    if any(
        bool(item["symlink_path_components"]) or bool(item["symlinks"])
        for item in presentation_root_checks
    ):
        block("presentation_root_symlink_detected")
    if any(bool(item["special_entries"]) for item in presentation_root_checks):
        block("presentation_root_special_entry_detected")
    if any(bool(item["read_errors"]) for item in presentation_root_checks):
        block("presentation_root_read_error")
    if any(bool(item["is_dir"]) and not bool(item["file_count_match"]) for item in presentation_root_checks):
        block("presentation_root_file_count_mismatch")
    if any(bool(item["is_dir"]) and not bool(item["total_bytes_match"]) for item in presentation_root_checks):
        block("presentation_root_total_bytes_mismatch")
    if any(bool(item["is_dir"]) and not bool(item["inventory_sha256_match"]) for item in presentation_root_checks):
        block("presentation_root_inventory_sha256_mismatch")
    if any(int(item["remaining_count"]) > 0 for item in absent):
        block("deleted_prefix_directory_remaining")
    if int(compacted_checks["remaining_original_count"]) > 0:
        block("compacted_original_directory_remaining")

    summary: Dict[str, object] = {
        "mode": "artifact_retention_audit_current_trkh_rerun",
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "guardrail": DEFAULT_GUARDRAIL,
        "runs_root": str(runs_root),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "raw_dataset_touched": False,
        "test_split_used_for_new_diagnostics": False,
        "trainable_manifest_written": False,
        "deleted_anything": False,
        "directory_count": len(directory_rows),
        "free_gb": round(float(free) / (1024.0**3), 3),
        "top_directory_count": int(top_directory_count),
        "decision_counts": _decision_counts(directory_rows),
        "protected_checks": protected,
        "essential_file_checks": essential_file_checks,
        "presentation_root_checks": presentation_root_checks,
        "absent_prefix_checks": absent,
        "compacted_original_checks": compacted_checks,
        "future_cleanup_review_candidate_count": len(cleanup_candidates),
        "future_cleanup_review_candidate_size_mb": _round_mb(sum(int(row["size_bytes"]) for row in cleanup_candidates)),
        "largest_directories_csv": str(output_dir / "largest_directories.csv"),
        "future_cleanup_review_candidates_csv": str(output_dir / "future_cleanup_review_candidates.csv"),
        "presentation_root_file_inventory_csv": str(output_dir / "presentation_root_file_inventory.csv"),
        "presentation_root_inventory_file_count": len(presentation_inventory_rows),
        "largest_directories": largest_rows,
        "future_cleanup_review_candidates": cleanup_candidates[:100],
        "blockers": blockers,
        "retention_audit_passed": not blockers,
        "decision": (
            "No deletion performed. Current TRKH keeper/gate/review artifacts are protected; "
            "compact original directories are expected to be absent before further cleanup."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = "\n".join(
        [
            "# TRKH Artifact Retention Audit",
            "",
            f"Created: `{summary['created_at']}`",
            "",
            f"Retention audit passed: `{str(summary['retention_audit_passed']).lower()}`",
            f"Directory count: `{summary['directory_count']}`",
            f"Future cleanup candidates: `{summary['future_cleanup_review_candidate_count']}`",
            f"Compacted originals remaining: `{compacted_checks['remaining_original_count']}`",
            "",
            "This audit is read-only. It does not delete files, touch raw datasets, tune on test, or write trainable manifests.",
        ]
    )
    (output_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-directory-count", type=int, default=120)
    parser.add_argument(
        "--compaction-manifest",
        type=Path,
        action="append",
        default=None,
        help="Cleanup manifest with compacted_names to verify original dirs are absent. Repeatable.",
    )
    parser.add_argument("--created-at", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = audit_trkh_artifact_retention(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        compaction_manifests=args.compaction_manifest,
        top_directory_count=args.top_directory_count,
        created_at=args.created_at,
    )
    print(json.dumps({key: summary[key] for key in ("retention_audit_passed", "directory_count", "blockers")}, indent=2))


if __name__ == "__main__":
    main()
