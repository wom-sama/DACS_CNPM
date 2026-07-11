from __future__ import annotations

import argparse
import csv
import json
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
    r"..\newdataset\class_f\data.yaml",
    r"..\newdataset\yolo_f\data.yaml",
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
        path = (repo_root / relative).resolve()
        checks.append(
            {
                "path": relative,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "is_file": path.is_file(),
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

    blockers = []
    if any(not bool(item["exists"]) for item in protected):
        blockers.append("missing_protected_artifact")
    if any(int(item["remaining_count"]) > 0 for item in absent):
        blockers.append("deleted_prefix_directory_remaining")
    if int(compacted_checks["remaining_original_count"]) > 0:
        blockers.append("compacted_original_directory_remaining")

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
        "absent_prefix_checks": absent,
        "compacted_original_checks": compacted_checks,
        "future_cleanup_review_candidate_count": len(cleanup_candidates),
        "future_cleanup_review_candidate_size_mb": _round_mb(sum(int(row["size_bytes"]) for row in cleanup_candidates)),
        "largest_directories_csv": str(output_dir / "largest_directories.csv"),
        "future_cleanup_review_candidates_csv": str(output_dir / "future_cleanup_review_candidates.csv"),
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
