from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_FINALIZATION_LOCK_20260724.json"
)
DEFAULT_GENERATION_ROOT = (
    REPO_ROOT
    / "runs"
    / "audit_saspa_dual_view_synthetic_a0_20260723"
    / "f1_tiny_output_v2"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash-locked finalization of the rejected SaSPA A0 F1."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=DEFAULT_GENERATION_ROOT,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-finalize", action="store_true")
    mode.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _resolve_inside(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    path = (root / Path(relative_path)).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes generation root: {relative_path}") from exc
    return path


def validate_fixed_artifacts(
    *,
    lock: Mapping[str, object],
    generation_root: Path,
) -> dict[str, object]:
    fixed = _as_mapping(lock.get("fixed_artifacts"), "fixed_artifacts")
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for relative_path, expected_hash in fixed.items():
        path = _resolve_inside(generation_root, str(relative_path))
        if not path.is_file():
            failures.append(f"missing:{relative_path}")
            continue
        observed = _sha256(path)
        if observed != str(expected_hash):
            failures.append(f"hash:{relative_path}")
        rows.append(
            {
                "relative_path": str(relative_path),
                "bytes": int(path.stat().st_size),
                "sha256": observed,
            }
        )
    return {
        "passed": not failures,
        "failures": failures,
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def evaluate_stage_checks(
    *,
    generation: Mapping[str, object],
    fidelity: Mapping[str, object],
    blind: Mapping[str, object],
    duplicate: Mapping[str, object],
    duplicate_replay: Mapping[str, object],
    xai: Mapping[str, object],
    xai_replay: Mapping[str, object],
    visual_review: Mapping[str, object],
) -> dict[str, object]:
    checks = {
        "generation_passed": bool(generation.get("passed_generation"))
        and int(generation.get("saved_output_count", -1)) == 10,
        "fidelity_passed": bool(fidelity.get("passed"))
        and int(fidelity.get("row_count", -1)) == 10,
        "blind_review_passed": bool(blind.get("passed")),
        "duplicate_audit_passed": bool(duplicate.get("passed"))
        and bool(duplicate_replay.get("passed"))
        and int(duplicate.get("reference_file_count", -1)) == 24996,
        "xai_execution_and_replay_complete": (
            str(xai.get("status")) == "diagnostic_complete_upstream_rejected"
            and int(xai.get("case_count", -1)) == 10
            and bool(xai_replay.get("passed"))
        ),
        "xai_visual_review_complete": (
            str(visual_review.get("decision_authority")) == "diagnostic_only"
            and visual_review.get("a1_authorized") is False
            and visual_review.get("training_authorized") is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "decision": "Passed" if not failed else "Rejected",
    }


def _load_evidence(root: Path) -> dict[str, Mapping[str, object]]:
    paths = {
        "generation": "generation_summary.json",
        "fidelity": "generated_fidelity.json",
        "blind": "blind_review_unblinded.json",
        "duplicate": "duplicate_audit/duplicate_report.json",
        "duplicate_replay": "duplicate_audit/duplicate_replay.json",
        "xai": "xai_audit/xai_summary.json",
        "xai_replay": "xai_audit/xai_replay.json",
        "visual_review": "f1_xai_visual_review.locked.json",
    }
    return {
        name: _as_mapping(
            _read_json(_resolve_inside(root, relative_path)),
            name,
        )
        for name, relative_path in paths.items()
    }


def _decision_from_evidence(evidence: Mapping[str, Mapping[str, object]]):
    return evaluate_stage_checks(
        generation=evidence["generation"],
        fidelity=evidence["fidelity"],
        blind=evidence["blind"],
        duplicate=evidence["duplicate"],
        duplicate_replay=evidence["duplicate_replay"],
        xai=evidence["xai"],
        xai_replay=evidence["xai_replay"],
        visual_review=evidence["visual_review"],
    )


def _repo_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "upstream": run("rev-parse", "@{upstream}"),
        "tracked_status": run("status", "--short", "--untracked-files=no"),
    }


def _compact_metrics(evidence: Mapping[str, Mapping[str, object]]):
    generation = evidence["generation"]
    fidelity = evidence["fidelity"]
    blind = evidence["blind"]
    duplicate = evidence["duplicate"]
    xai = evidence["xai"]
    return {
        "generation": {
            "passed": bool(generation.get("passed_generation")),
            "saved_output_count": int(generation.get("saved_output_count", -1)),
        },
        "fidelity": {
            "passed": bool(fidelity.get("passed")),
            "passed_count": int(fidelity.get("passed_count", -1)),
            "row_count": int(fidelity.get("row_count", -1)),
        },
        "blind_review": {
            "passed": bool(blind.get("passed")),
            "counts": blind.get("counts"),
            "limits": blind.get("limits"),
            "checks": blind.get("checks"),
        },
        "duplicate_audit": {
            "passed": bool(duplicate.get("passed")),
            "reference_file_count": int(
                duplicate.get("reference_file_count", -1)
            ),
            "output_count": len(duplicate.get("output_rows", [])),
            "synthetic_pair_count": len(duplicate.get("synthetic_pairs", [])),
        },
        "xai": {
            "case_count": int(xai.get("case_count", -1)),
            "correct_count": int(xai.get("correct_count", -1)),
            "accuracy": float(xai.get("accuracy", 0.0)),
            "per_class": xai.get("per_class"),
            "mean_gradcam_bbox_mass": float(
                xai.get("mean_gradcam_bbox_mass", 0.0)
            ),
            "mean_input_gradient_bbox_mass": float(
                xai.get("mean_input_gradient_bbox_mass", 0.0)
            ),
            "robustness_mean": xai.get("robustness_mean"),
        },
    }


def _closure_manifest(
    *,
    generation_root: Path,
    fixed_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = [dict(row) for row in fixed_rows]
    for relative_path in ("f1_final_summary.json", "f1_final_replay.json"):
        path = _resolve_inside(generation_root, relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    rows.sort(key=lambda row: str(row["relative_path"]))
    rows_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "trkh_saspa_f1_final_manifest_v1",
        "root": str(generation_root.resolve()),
        "file_count": len(rows),
        "files_sha256": rows_sha256,
        "files": rows,
        "a1_authorized": False,
        "training_authorized": False,
        "current_best_command_updated": False,
    }


def preflight(args: argparse.Namespace) -> dict[str, object]:
    lock = _as_mapping(_read_json(args.lock), "lock")
    fixed = validate_fixed_artifacts(
        lock=lock,
        generation_root=args.generation_root,
    )
    evidence = _load_evidence(args.generation_root) if fixed["passed"] else {}
    decision = _decision_from_evidence(evidence) if evidence else None
    result = {
        "schema": "trkh_saspa_f1_finalization_preflight_v1",
        "lock_sha256": _sha256(args.lock),
        "fixed_artifacts": fixed,
        "decision": decision,
        "passed": (
            bool(fixed["passed"])
            and isinstance(decision, Mapping)
            and decision.get("decision") == str(lock.get("expected_locked_outcome"))
        ),
        "images_decoded": False,
        "model_inference_performed": False,
        "validation_test_pixels_opened": False,
        "a1_authorized": False,
        "training_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("SaSPA F1 finalization preflight failed.")
    return result


def formal_finalize(args: argparse.Namespace) -> dict[str, object]:
    summary_path = args.generation_root / "f1_final_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Final summary already exists: {summary_path}")
    lock = _as_mapping(_read_json(args.lock), "lock")
    fixed = validate_fixed_artifacts(
        lock=lock,
        generation_root=args.generation_root,
    )
    if not fixed["passed"]:
        raise RuntimeError(f"Fixed artifacts failed: {fixed['failures']}")
    repo_state = _repo_state()
    if repo_state["tracked_status"]:
        raise RuntimeError("Formal finalization requires a clean tracked worktree.")
    if repo_state["head"] != repo_state["upstream"]:
        raise RuntimeError("Formal finalization requires HEAD to equal upstream.")
    evidence = _load_evidence(args.generation_root)
    decision = _decision_from_evidence(evidence)
    expected = str(lock.get("expected_locked_outcome"))
    if decision["decision"] != expected:
        raise RuntimeError(
            f"Locked final outcome differs: {decision['decision']} != {expected}"
        )
    summary = {
        "schema": "trkh_saspa_f1_final_summary_v1",
        "status": "Rejected",
        "method": "SaSPA-derived dual-view synthetic augmentation A0 F1",
        "protocol_id": str(lock.get("protocol_id")),
        "started_unix": time.time(),
        "finished_unix": time.time(),
        "finalization_lock": str(args.lock.resolve()),
        "finalization_lock_sha256": _sha256(args.lock),
        "generation_root": str(args.generation_root.resolve()),
        "repo_state": repo_state,
        "fixed_artifact_count": fixed["artifact_count"],
        "fixed_artifacts": fixed["artifacts"],
        "stage_decision": decision,
        "metrics": _compact_metrics(evidence),
        "scientific_conclusion": (
            "Generation and duplicate-safety execution passed, but all 10 "
            "outputs failed the locked class-specific fidelity gate, blind "
            "review passed only 7/10 with 4 severe artifacts, and frozen-keeper "
            "XAI reached only 6/10 with weak bbox attribution. The exact SaSPA "
            "configuration does not preserve TRKH fine-grained class evidence."
        ),
        "exact_configuration_closed": True,
        "prompt_seed_step_threshold_or_ratio_sweep_authorized": False,
        "a1_authorized": False,
        "training_authorized": False,
        "full_train_authorized": False,
        "current_best_command_updated": False,
        "raw_dataset_modified": False,
        "images_decoded_by_finalizer": False,
        "model_inference_performed_by_finalizer": False,
        "source_train_pixels_opened_by_finalizer": False,
        "validation_test_pixels_opened_by_finalizer": False,
        "validation_test_labels_opened_by_finalizer": False,
    }
    _write_json(summary_path, summary)
    result = {
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "decision": summary["status"],
        "failed_checks": decision["failed_checks"],
        "a1_authorized": False,
        "training_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return summary


def replay(summary_path: Path) -> dict[str, object]:
    summary_path = summary_path.resolve()
    generation_root = summary_path.parent
    summary = _as_mapping(_read_json(summary_path), "summary")
    lock_path = Path(str(summary["finalization_lock"]))
    lock = _as_mapping(_read_json(lock_path), "lock")
    fixed = validate_fixed_artifacts(
        lock=lock,
        generation_root=generation_root,
    )
    evidence = _load_evidence(generation_root)
    decision = _decision_from_evidence(evidence)
    failures: list[str] = []
    if not fixed["passed"]:
        failures.extend(f"fixed:{item}" for item in fixed["failures"])
    if summary.get("status") != decision["decision"]:
        failures.append("decision")
    if summary.get("stage_decision") != decision:
        failures.append("stage_decision")
    if summary.get("metrics") != _compact_metrics(evidence):
        failures.append("metrics")
    if bool(summary.get("a1_authorized")) or bool(
        summary.get("training_authorized")
    ):
        failures.append("premature_authorization")
    if bool(summary.get("validation_test_pixels_opened_by_finalizer")):
        failures.append("split_boundary")
    result = {
        "schema": "trkh_saspa_f1_final_replay_v1",
        "summary_sha256": _sha256(summary_path),
        "finalization_lock_sha256": _sha256(lock_path),
        "fixed_artifact_checks": fixed,
        "recomputed_decision": decision,
        "failures": failures,
        "passed": not failures,
        "images_decoded": False,
        "model_inference_performed": False,
        "source_train_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "a1_authorized": False,
        "training_authorized": False,
        "current_best_command_updated": False,
    }
    replay_path = generation_root / "f1_final_replay.json"
    _write_json(replay_path, result)
    manifest = _closure_manifest(
        generation_root=generation_root,
        fixed_rows=fixed["artifacts"],
    )
    manifest_path = generation_root / "f1_final_manifest.json"
    _write_json(manifest_path, manifest)
    printed = dict(result)
    printed["replay_sha256"] = _sha256(replay_path)
    printed["final_manifest_sha256"] = _sha256(manifest_path)
    print(json.dumps(printed, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"SaSPA F1 final replay failed: {failures}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        preflight(args)
    elif args.formal_finalize:
        formal_finalize(args)
    else:
        replay(args.replay_summary)


if __name__ == "__main__":
    main()
