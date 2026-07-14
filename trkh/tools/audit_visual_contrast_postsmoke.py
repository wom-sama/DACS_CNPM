from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


EXPECTED_CONDITIONS = (
    "clean",
    "occlusion_center",
    "lighting_dim",
    "lighting_bright",
    "low_contrast",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _condition_summary(root: Path, condition: str) -> tuple[Path, Mapping[str, object]]:
    path = Path(root) / condition / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, _read_json(path)


def run_audit(
    *,
    pair_summary: Path,
    trace_audit: Path,
    paired_xai: Path,
    robustness_control: Path,
    robustness_candidate: Path,
    forensics_control: Path,
    forensics_candidate: Path,
    confusions_control: Path,
    confusions_candidate: Path,
    output_dir: Path,
) -> Dict[str, object]:
    pair_summary = Path(pair_summary).resolve()
    trace_audit = Path(trace_audit).resolve()
    paired_xai = Path(paired_xai).resolve()
    robustness_control = Path(robustness_control).resolve()
    robustness_candidate = Path(robustness_candidate).resolve()
    forensics_control = Path(forensics_control).resolve()
    forensics_candidate = Path(forensics_candidate).resolve()
    confusions_control = Path(confusions_control).resolve()
    confusions_candidate = Path(confusions_candidate).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Post-smoke output already exists and is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = _read_json(pair_summary)
    trace = _read_json(trace_audit)
    xai = _read_json(paired_xai)
    pair_sources = pair.get("sources", {})
    if not isinstance(pair_sources, Mapping) or bool(pair_sources.get("test_used", True)):
        raise ValueError("Pair summary must explicitly declare test_used=false")
    if bool(xai.get("test_data_used", True)):
        raise ValueError("Paired XAI must explicitly declare test_data_used=false")
    trace_checks = trace.get("checks", {})
    if not isinstance(trace_checks, Mapping) or bool(trace_checks.get("test_used", True)):
        raise ValueError("Trace audit must explicitly declare test_used=false")

    control_forensics = _read_json(forensics_control)
    candidate_forensics = _read_json(forensics_candidate)
    control_confusions = _read_json(confusions_control)
    candidate_confusions = _read_json(confusions_candidate)
    sample_counts = {
        int(payload.get("samples", 0))
        for payload in (
            control_forensics,
            candidate_forensics,
            control_confusions,
            candidate_confusions,
        )
    }
    if sample_counts != {2606}:
        raise ValueError(f"Forensics/confusion cohort mismatch: {sorted(sample_counts)}")

    control_focus = control_confusions.get("focus_metrics")
    candidate_focus = candidate_confusions.get("focus_metrics")
    if not isinstance(control_focus, Mapping) or not isinstance(candidate_focus, Mapping):
        raise ValueError("Confusion summaries are missing focus_metrics")
    focus_delta = {
        field: float(candidate_focus[field]) - float(control_focus[field])
        for field in ("tp", "fp", "fn", "precision", "recall", "f1")
    }
    control_errors = control_confusions.get("error_counts", {})
    candidate_errors = candidate_confusions.get("error_counts", {})
    if not isinstance(control_errors, Mapping) or not isinstance(candidate_errors, Mapping):
        raise ValueError("Confusion summaries are missing error_counts")
    error_count_deltas = {
        key: int(candidate_errors.get(key, 0)) - int(control_errors.get(key, 0))
        for key in sorted(set(control_errors) | set(candidate_errors))
    }
    control_pairs = {
        str(row["pair"]): float(row["f1"])
        for row in control_forensics.get("pair_metrics", [])
        if isinstance(row, Mapping) and row.get("f1") is not None
    }
    candidate_pairs = {
        str(row["pair"]): float(row["f1"])
        for row in candidate_forensics.get("pair_metrics", [])
        if isinstance(row, Mapping) and row.get("f1") is not None
    }
    pair_f1_deltas = {
        key: candidate_pairs[key] - control_pairs[key]
        for key in sorted(set(control_pairs) & set(candidate_pairs))
    }

    robustness_rows = []
    robustness_sources = []
    for condition in EXPECTED_CONDITIONS:
        control_path, control = _condition_summary(robustness_control, condition)
        candidate_path, candidate = _condition_summary(robustness_candidate, condition)
        control_metrics = control.get("metrics")
        candidate_metrics = candidate.get("metrics")
        if not isinstance(control_metrics, Mapping) or not isinstance(
            candidate_metrics, Mapping
        ):
            raise ValueError(f"Missing robustness metrics for condition={condition}")
        row = {
            "condition": condition,
            "control_accuracy": float(control_metrics["accuracy"]),
            "candidate_accuracy": float(candidate_metrics["accuracy"]),
            "accuracy_delta": float(candidate_metrics["accuracy"])
            - float(control_metrics["accuracy"]),
            "control_macro_f1": float(control_metrics["macro_f1"]),
            "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
            "macro_f1_delta": float(candidate_metrics["macro_f1"])
            - float(control_metrics["macro_f1"]),
        }
        robustness_rows.append(row)
        robustness_sources.extend(
            (
                {"path": str(control_path.resolve()), "sha256": _sha256(control_path)},
                {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            )
        )

    robustness_path = output_dir / "robustness_comparison.csv"
    with robustness_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
        writer.writeheader()
        writer.writerows(robustness_rows)

    gate = pair.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Pair summary is missing the locked gate")
    xai_categories = xai.get("categories")
    if not isinstance(xai_categories, Mapping) or not isinstance(
        xai_categories.get("all"), Mapping
    ):
        raise ValueError("Paired XAI is missing the all-category summary")
    all_xai = xai_categories["all"]
    xai_delta = all_xai.get("candidate_minus_control", {})
    grad_provenance = xai.get("grad_rollout_provenance", {})
    candidate_grad_fallback_cases = 0
    if isinstance(grad_provenance, Mapping):
        fallback_counts = grad_provenance.get("fallback_case_count", {})
        if isinstance(fallback_counts, Mapping):
            candidate_grad_fallback_cases = int(fallback_counts.get("candidate", 0))
    candidate_grad_valid = candidate_grad_fallback_cases == 0
    valid_xai_delta = dict(xai_delta) if isinstance(xai_delta, Mapping) else {}
    invalid_xai_delta_fields = []
    if not candidate_grad_valid:
        invalid_xai_delta_fields = sorted(
            key for key in valid_xai_delta if str(key).startswith("grad_rollout_")
        )
        for key in invalid_xai_delta_fields:
            valid_xai_delta.pop(key, None)

    robustness_wins = sum(
        float(row["macro_f1_delta"]) > 0.0 for row in robustness_rows
    )
    summary: Dict[str, object] = {
        "mode": "visual_contrast_postsmoke_closure",
        "protocol_stage": "C_postsmoke_audit",
        "decision": {
            "metric_gate_passed": bool(gate.get("metric_gate_passed", False)),
            "five_epoch_permission": bool(gate.get("five_epoch_permission", False)),
            "full_train_permission": bool(gate.get("full_train_permission", False)),
            "close_method_without_hyperparameter_sweep": not bool(
                gate.get("metric_gate_passed", False)
            ),
            "failed_checks": list(gate.get("failed_checks", [])),
        },
        "robustness": {
            "conditions": robustness_rows,
            "macro_f1_win_count": int(robustness_wins),
            "condition_count": len(robustness_rows),
            "best_macro_f1_delta": max(
                float(row["macro_f1_delta"]) for row in robustness_rows
            ),
            "worst_macro_f1_delta": min(
                float(row["macro_f1_delta"]) for row in robustness_rows
            ),
            "comparison_csv": str(robustness_path),
            "comparison_csv_sha256": _sha256(robustness_path),
        },
        "trace": {
            "checks": dict(trace_checks),
            "diagnostics": trace.get("diagnostics", {}),
        },
        "forensics": {
            "validation_rows": 2606,
            "control_focus": dict(control_focus),
            "candidate_focus": dict(candidate_focus),
            "candidate_minus_control_focus": focus_delta,
            "pair_f1_deltas": pair_f1_deltas,
            "error_count_deltas": error_count_deltas,
            "calibration": {
                "control": control_forensics.get("calibration", {}),
                "candidate": candidate_forensics.get("calibration", {}),
            },
        },
        "xai": {
            "cohort_cases": int(xai.get("cases", 0)),
            "candidate_minus_control": valid_xai_delta,
            "invalid_candidate_minus_control_fields": invalid_xai_delta_fields,
            "candidate_grad_rollout_fallback_cases": candidate_grad_fallback_cases,
            "candidate_grad_rollout_valid_for_attribution": candidate_grad_valid,
            "attention_provenance": xai.get("attention_provenance", {}),
            "grad_rollout_provenance": grad_provenance,
        },
        "sources": {
            "pair_summary": {"path": str(pair_summary), "sha256": _sha256(pair_summary)},
            "trace_audit": {"path": str(trace_audit), "sha256": _sha256(trace_audit)},
            "paired_xai": {"path": str(paired_xai), "sha256": _sha256(paired_xai)},
            "forensics_control": {
                "path": str(forensics_control),
                "sha256": _sha256(forensics_control),
            },
            "forensics_candidate": {
                "path": str(forensics_candidate),
                "sha256": _sha256(forensics_candidate),
            },
            "confusions_control": {
                "path": str(confusions_control),
                "sha256": _sha256(confusions_control),
            },
            "confusions_candidate": {
                "path": str(confusions_candidate),
                "sha256": _sha256(confusions_candidate),
            },
            "robustness_summaries": robustness_sources,
        },
        "raw_dataset_modified": False,
        "test_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close a VCA smoke with locked metric, robustness, trace, and XAI evidence."
    )
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--trace-audit", type=Path, required=True)
    parser.add_argument("--paired-xai", type=Path, required=True)
    parser.add_argument("--robustness-control", type=Path, required=True)
    parser.add_argument("--robustness-candidate", type=Path, required=True)
    parser.add_argument("--forensics-control", type=Path, required=True)
    parser.add_argument("--forensics-candidate", type=Path, required=True)
    parser.add_argument("--confusions-control", type=Path, required=True)
    parser.add_argument("--confusions-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(
        pair_summary=args.pair_summary,
        trace_audit=args.trace_audit,
        paired_xai=args.paired_xai,
        robustness_control=args.robustness_control,
        robustness_candidate=args.robustness_candidate,
        forensics_control=args.forensics_control,
        forensics_candidate=args.forensics_candidate,
        confusions_control=args.confusions_control,
        confusions_candidate=args.confusions_candidate,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
