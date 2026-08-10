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
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _condition_summary(root: Path, condition: str) -> tuple[Path, Mapping[str, object]]:
    path = Path(root) / condition / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, _read_json(path)


def _focus_recall(metrics: Mapping[str, object]) -> float:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 5:
        raise ValueError("Robustness metrics require a 5x5 confusion matrix")
    row = matrix[1]
    if not isinstance(row, list) or len(row) != 5:
        raise ValueError("Robustness class-1 row is invalid")
    support = sum(int(value) for value in row)
    if support != 151:
        raise ValueError(f"Robustness class-1 support mismatch: {support}")
    return int(row[1]) / support


def assess_postsmoke(
    *,
    pair_gate_passed: bool,
    robustness_rows: Sequence[Mapping[str, float]],
    trace_passed: bool,
    candidate_xai_means: Mapping[str, float],
    attention_sources: Mapping[str, int],
    grad_rollout_fallback_cases: int,
) -> Dict[str, object]:
    robustness_preserved = sum(
        float(row["macro_f1_delta"]) >= 0.0 for row in robustness_rows
    )
    worst_recall_delta = min(float(row["class1_recall_delta"]) for row in robustness_rows)
    gradcam_foreground = float(candidate_xai_means.get("gradcam_foreground_mass", 0.0))
    gradcam_border = float(candidate_xai_means.get("gradcam_border_mass", 1.0))
    attention_foreground = float(candidate_xai_means.get("attention_foreground_mass", 0.0))
    object_drop = max(
        abs(
            float(
                candidate_xai_means.get(
                    "object_desaturate_original_prediction_drop", 0.0
                )
            )
        ),
        abs(
            float(
                candidate_xai_means.get(
                    "object_desaturate_target_probability_drop", 0.0
                )
            )
        ),
    )
    background_drop = max(
        abs(
            float(
                candidate_xai_means.get(
                    "background_gray_original_prediction_drop", 0.0
                )
            )
        ),
        abs(float(candidate_xai_means.get("background_gray_target_probability_drop", 0.0))),
        abs(
            float(
                candidate_xai_means.get(
                    "background_blur_original_prediction_drop", 0.0
                )
            )
        ),
        abs(float(candidate_xai_means.get("background_blur_target_probability_drop", 0.0))),
    )
    checks = {
        "pair_metric_gate_passed": bool(pair_gate_passed),
        "robustness_three_of_five": robustness_preserved >= 3,
        "robustness_class1_recall_preserved": worst_recall_delta >= -0.05,
        "architecture_trace_passed": bool(trace_passed),
        "native_mhsa_attention_provenance": bool(attention_sources)
        and all(".mhsa_probability" in source for source in attention_sources),
        "grad_rollout_no_fallback": int(grad_rollout_fallback_cases) == 0,
        "attention_foreground_focused": attention_foreground >= 0.60,
        "stem_gradcam_foreground_focused": gradcam_foreground >= 0.65
        and gradcam_foreground >= gradcam_border + 0.20,
        "surface_signal_exceeds_background": object_drop > background_drop,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "stage_b_passed": not failed,
        "five_epoch_permission": not failed,
        "full_train_permission": False,
        "close_method_without_hyperparameter_sweep": bool(failed),
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "robustness_preserved_count": robustness_preserved,
            "robustness_condition_count": len(robustness_rows),
            "worst_class1_recall_delta": worst_recall_delta,
            "candidate_attention_foreground_mass": attention_foreground,
            "candidate_gradcam_foreground_mass": gradcam_foreground,
            "candidate_gradcam_border_mass": gradcam_border,
            "candidate_object_desaturate_causal_drop_abs": object_drop,
            "candidate_background_causal_drop_abs_max": background_drop,
            "candidate_grad_rollout_fallback_cases": int(grad_rollout_fallback_cases),
        },
    }


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
    expected_method: str = "inceptionnext_atto_surface_tokenizer",
    mode: str = "inceptionnext_atto_postsmoke_closure",
) -> Dict[str, object]:
    paths = [
        pair_summary,
        trace_audit,
        paired_xai,
        forensics_control,
        forensics_candidate,
        confusions_control,
        confusions_candidate,
    ]
    (
        pair_summary,
        trace_audit,
        paired_xai,
        forensics_control,
        forensics_candidate,
        confusions_control,
        confusions_candidate,
    ) = [Path(path).resolve() for path in paths]
    robustness_control = Path(robustness_control).resolve()
    robustness_candidate = Path(robustness_candidate).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Post-smoke output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = _read_json(pair_summary)
    trace = _read_json(trace_audit)
    xai = _read_json(paired_xai)
    if pair.get("method") != expected_method:
        raise ValueError("Pair method mismatch")
    if bool(pair.get("sources", {}).get("test_used", True)):
        raise ValueError("Pair summary must declare test_used=false")
    if bool(xai.get("test_data_used", True)):
        raise ValueError("Paired XAI must declare test_data_used=false")
    if not bool(trace.get("checks", {}).get("test_used") is False):
        raise ValueError("Trace audit must declare test_used=false")

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

    robustness_rows = []
    robustness_sources = []
    for condition in EXPECTED_CONDITIONS:
        control_path, control = _condition_summary(robustness_control, condition)
        candidate_path, candidate = _condition_summary(robustness_candidate, condition)
        control_metrics = control.get("metrics", {})
        candidate_metrics = candidate.get("metrics", {})
        if not isinstance(control_metrics, Mapping) or not isinstance(candidate_metrics, Mapping):
            raise ValueError(f"Missing robustness metrics for {condition}")
        control_recall = _focus_recall(control_metrics)
        candidate_recall = _focus_recall(candidate_metrics)
        robustness_rows.append(
            {
                "condition": condition,
                "control_accuracy": float(control_metrics["accuracy"]),
                "candidate_accuracy": float(candidate_metrics["accuracy"]),
                "accuracy_delta": float(candidate_metrics["accuracy"])
                - float(control_metrics["accuracy"]),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "macro_f1_delta": float(candidate_metrics["macro_f1"])
                - float(control_metrics["macro_f1"]),
                "control_class1_recall": control_recall,
                "candidate_class1_recall": candidate_recall,
                "class1_recall_delta": candidate_recall - control_recall,
            }
        )
        robustness_sources.extend(
            (
                {"path": str(control_path), "sha256": _sha256(control_path)},
                {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
            )
        )

    xai_all = xai.get("categories", {}).get("all", {})
    candidate_xai = xai_all.get("candidate", {}) if isinstance(xai_all, Mapping) else {}
    candidate_means = candidate_xai.get("means", {}) if isinstance(candidate_xai, Mapping) else {}
    attention_sources = (
        candidate_xai.get("attention_sources", {}) if isinstance(candidate_xai, Mapping) else {}
    )
    grad_provenance = xai.get("grad_rollout_provenance", {})
    fallback_counts = (
        grad_provenance.get("fallback_case_count", {})
        if isinstance(grad_provenance, Mapping)
        else {}
    )
    candidate_fallback = int(fallback_counts.get("candidate", 0))
    pair_gate = pair.get("gate", {})
    decision = assess_postsmoke(
        pair_gate_passed=bool(pair_gate.get("metric_gate_passed", False)),
        robustness_rows=robustness_rows,
        trace_passed=bool(trace.get("passed", False)),
        candidate_xai_means=candidate_means,
        attention_sources=attention_sources,
        grad_rollout_fallback_cases=candidate_fallback,
    )

    robustness_path = output_dir / "robustness_comparison.csv"
    with robustness_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
        writer.writeheader()
        writer.writerows(robustness_rows)

    control_focus = control_confusions.get("focus_metrics", {})
    candidate_focus = candidate_confusions.get("focus_metrics", {})
    focus_delta = {
        field: float(candidate_focus[field]) - float(control_focus[field])
        for field in ("tp", "fp", "fn", "precision", "recall", "f1")
    }
    summary: Dict[str, object] = {
        "mode": mode,
        "protocol_stage": "C_postsmoke_audit",
        "decision": decision,
        "robustness": {
            "conditions": robustness_rows,
            "comparison_csv": str(robustness_path),
            "comparison_csv_sha256": _sha256(robustness_path),
        },
        "trace": {
            "checks": trace.get("checks", {}),
            "sample_checks": trace.get("sample_checks", []),
        },
        "forensics": {
            "validation_rows": 2606,
            "control_focus": control_focus,
            "candidate_focus": candidate_focus,
            "candidate_minus_control_focus": focus_delta,
            "control_calibration": control_forensics.get("calibration", {}),
            "candidate_calibration": candidate_forensics.get("calibration", {}),
        },
        "xai": {
            "cohort_cases": int(xai.get("cases", 0)),
            "candidate_means": candidate_means,
            "candidate_attention_sources": attention_sources,
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
        description="Close a tokenizer smoke with robustness, trace, and XAI evidence."
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
    parser.add_argument(
        "--expected-method",
        default="inceptionnext_atto_surface_tokenizer",
    )
    parser.add_argument("--mode", default="inceptionnext_atto_postsmoke_closure")
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
        expected_method=args.expected_method,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
