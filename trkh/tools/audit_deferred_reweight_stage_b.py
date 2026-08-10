from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


EXPECTED_SUPPORT = [549, 151, 544, 712, 650]
NONFOCUS_REFERENCE_F1 = {
    0: 0.863850,
    2: 0.824593,
    3: 0.905633,
    4: 0.907618,
}
ROBUSTNESS_CONDITIONS = (
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


def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _confusion_from_predictions(path: Path) -> list[list[int]]:
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    seen: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "target_index", "prediction_index"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV is missing columns {sorted(required)}: {path}")
        for row in reader:
            sample_index = int(row["sample_index"])
            if sample_index in seen:
                raise ValueError(f"Duplicate prediction sample_index={sample_index}")
            seen.add(sample_index)
            target = int(row["target_index"])
            prediction = int(row["prediction_index"])
            if not 0 <= target < 5 or not 0 <= prediction < 5:
                raise ValueError("Prediction class index is outside [0, 5).")
            confusion[target][prediction] += 1
    if len(seen) != sum(EXPECTED_SUPPORT):
        raise ValueError(f"Expected {sum(EXPECTED_SUPPORT)} prediction rows, found {len(seen)}")
    return confusion


def _per_class_from_confusion(confusion: Sequence[Sequence[int]]) -> list[Dict[str, float]]:
    rows: list[Dict[str, float]] = []
    for class_index in range(5):
        tp = int(confusion[class_index][class_index])
        support = int(sum(confusion[class_index]))
        predicted = int(sum(confusion[row][class_index] for row in range(5)))
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "class_index": int(class_index),
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def assess_clean(
    metrics: Mapping[str, object],
    confusion: Sequence[Sequence[int]],
) -> Dict[str, object]:
    reconstructed = _per_class_from_confusion(confusion)
    metric_rows = metrics.get("per_class")
    if not isinstance(metric_rows, list) or len(metric_rows) != 5:
        raise ValueError("Independent metrics must contain five per-class rows.")
    for class_index, (observed, expected) in enumerate(zip(metric_rows, reconstructed)):
        if int(observed["class_index"]) != class_index:
            raise ValueError("Independent per-class rows are not ordered 0..4.")
        for field in ("precision", "recall", "f1"):
            if not math.isclose(
                float(observed[field]), float(expected[field]), rel_tol=0.0, abs_tol=2e-6
            ):
                raise ValueError(f"Independent {field} does not replay for class {class_index}.")
    support = [int(row["support"]) for row in reconstructed]
    if support != EXPECTED_SUPPORT:
        raise ValueError(f"Validation support mismatch: {support} != {EXPECTED_SUPPORT}")
    class1 = reconstructed[1]
    class1_fp = int(sum(confusion[row][1] for row in range(5) if row != 1))
    macro_f1 = float(sum(float(row["f1"]) for row in reconstructed) / 5.0)
    if not math.isclose(macro_f1, float(metrics["macro_f1"]), rel_tol=0.0, abs_tol=2e-6):
        raise ValueError("Independent macro F1 does not replay from predictions.")
    checks: Dict[str, bool] = {
        "full_validation_support_2606": sum(support) == 2606,
        "macro_f1_at_least_0_778520": macro_f1 >= 0.778520,
        "class1_f1_at_least_0_470909": float(class1["f1"]) >= 0.470909,
        "class1_precision_at_least_0_403702": float(class1["precision"]) >= 0.403702,
        "class1_recall_at_least_0_635232": float(class1["recall"]) >= 0.635232,
        "class1_false_positives_at_most_162": class1_fp <= 162,
    }
    for class_index, reference in NONFOCUS_REFERENCE_F1.items():
        checks[f"class{class_index}_f1_loss_at_most_0_04"] = (
            float(reconstructed[class_index]["f1"]) >= float(reference) - 0.04
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "macro_f1": macro_f1,
        "class1": {**class1, "false_positives": class1_fp},
        "per_class": reconstructed,
        "confusion_matrix": [list(map(int, row)) for row in confusion],
    }


def assess_trace(trace: Mapping[str, object]) -> Dict[str, object]:
    samples = trace.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Architecture trace has no sample list.")
    class_ids = sorted(int(row["class_id"]) for row in samples)
    checks = {
        "five_trace_samples": len(samples) == 5,
        "one_trace_per_class": class_ids == [0, 1, 2, 3, 4],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_checks": failed, "class_ids": class_ids}


def assess_robustness(
    candidate: Mapping[str, object],
    keeper: Mapping[str, object],
) -> Dict[str, object]:
    candidate_conditions = candidate.get("conditions")
    keeper_conditions = keeper.get("conditions")
    if not isinstance(candidate_conditions, Mapping) or not isinstance(keeper_conditions, Mapping):
        raise ValueError("Robustness summaries must contain condition mappings.")
    rows: Dict[str, object] = {}
    macro_wins = 0
    tp_retained = True
    support_valid = True
    for condition in ROBUSTNESS_CONDITIONS:
        candidate_metrics = candidate_conditions[condition]["metrics"]
        keeper_metrics = keeper_conditions[condition]["metrics"]
        candidate_confusion = candidate_metrics["confusion_matrix"]
        keeper_confusion = keeper_metrics["confusion_matrix"]
        candidate_support = int(sum(sum(int(value) for value in row) for row in candidate_confusion))
        keeper_support = int(sum(sum(int(value) for value in row) for row in keeper_confusion))
        support_valid = support_valid and candidate_support == 2606 and keeper_support == 2606
        candidate_tp = int(candidate_confusion[1][1])
        keeper_tp = int(keeper_confusion[1][1])
        macro_delta = float(candidate_metrics["macro_f1"]) - float(keeper_metrics["macro_f1"])
        tp_delta = candidate_tp - keeper_tp
        macro_wins += int(macro_delta >= 0.0)
        tp_retained = tp_retained and tp_delta >= -2
        rows[condition] = {
            "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
            "keeper_macro_f1": float(keeper_metrics["macro_f1"]),
            "macro_f1_delta": macro_delta,
            "candidate_class1_tp": candidate_tp,
            "keeper_class1_tp": keeper_tp,
            "class1_tp_delta": tp_delta,
        }
    checks = {
        "all_conditions_full_validation": support_valid,
        "macro_f1_wins_at_least_three_of_five": macro_wins >= 3,
        "class1_tp_loss_at_most_two_each_condition": tp_retained,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "macro_f1_win_count": macro_wins,
        "conditions": rows,
    }


def assess_xai(paired: Mapping[str, object]) -> Dict[str, object]:
    categories = paired.get("categories")
    if not isinstance(categories, Mapping) or not isinstance(categories.get("all"), Mapping):
        raise ValueError("Paired XAI summary has no all-category aggregate.")
    aggregate = categories["all"]
    deltas = aggregate.get("candidate_minus_keeper")
    if not isinstance(deltas, Mapping):
        raise ValueError("Paired XAI summary has no candidate-minus-keeper deltas.")
    sheets = paired.get("contact_sheets")
    if not isinstance(sheets, Mapping):
        raise ValueError("Paired XAI summary has no contact sheets.")
    missing_tiles = int(sum(int(row.get("missing_tiles", 0)) for row in sheets.values()))

    def delta(name: str) -> float:
        value = deltas.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Paired XAI delta is missing or non-finite: {name}")
        return float(value)

    checks = {
        "exact_12_case_cohort": int(paired.get("cases", 0)) == 12,
        "known_attention_provenance": bool(
            paired.get("attention_provenance", {}).get("known_sources_only", False)
        ),
        "contact_sheets_have_no_missing_tiles": missing_tiles == 0,
        "gradcam_foreground_drop_at_most_0_05": delta("gradcam_foreground_mass") >= -0.05,
        "grad_rollout_foreground_drop_at_most_0_05": delta("grad_rollout_foreground_mass") >= -0.05,
        "background_gray_drop_increase_at_most_0_03": delta(
            "background_gray_original_prediction_drop"
        ) <= 0.03,
        "background_blur_drop_increase_at_most_0_03": delta(
            "background_blur_original_prediction_drop"
        ) <= 0.03,
        "center_occlusion_drop_increase_at_most_0_05": delta(
            "center_occlusion_original_prediction_drop"
        ) <= 0.05,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "missing_tiles": missing_tiles,
        "candidate_minus_keeper": dict(deltas),
    }


def _write_gate_table(path: Path, sections: Mapping[str, Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "gate", "passed"))
        writer.writeheader()
        for section, payload in sections.items():
            for gate, passed in payload["checks"].items():
                writer.writerow({"section": section, "gate": gate, "passed": bool(passed)})


def run_audit(
    *,
    metrics_path: Path,
    predictions_path: Path,
    trace_path: Path,
    robustness_path: Path,
    keeper_robustness_path: Path,
    paired_xai_path: Path,
    history_path: Path,
    checkpoint_path: Path,
    cohort_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output exists and is nonempty: {output_dir}")
    metrics = _read_json(metrics_path)
    confusion = _confusion_from_predictions(predictions_path)
    clean = assess_clean(metrics, confusion)
    trace = assess_trace(_read_json(trace_path))
    robustness = assess_robustness(
        _read_json(robustness_path), _read_json(keeper_robustness_path)
    )
    xai = assess_xai(_read_json(paired_xai_path))
    sections = {"clean": clean, "trace": trace, "robustness": robustness, "xai": xai}
    failed_sections = [name for name, payload in sections.items() if not bool(payload["passed"])]
    decision = (
        "authorize_default_off_deferred_reweight_stage_c"
        if not failed_sections
        else "reject_and_close_natural_first_deferred_reweight_recipe"
    )
    inputs = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in {
            "metrics": metrics_path,
            "predictions": predictions_path,
            "trace": trace_path,
            "robustness": robustness_path,
            "keeper_robustness": keeper_robustness_path,
            "paired_xai": paired_xai_path,
            "history": history_path,
            "checkpoint": checkpoint_path,
            "cohort": cohort_path,
        }.items()
    }
    summary: Dict[str, object] = {
        "mode": "deferred_reweight_stage_b_post_smoke_gate",
        "all_gates_passed": not failed_sections,
        "failed_sections": failed_sections,
        "decision": decision,
        "test_used": False,
        "raw_dataset_modified": False,
        "deferred_reweight_used": False,
        "full_train_authorized": False,
        "current_best_command_update_authorized": False,
        "sections": sections,
        "inputs": inputs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "gate_table.csv"
    _write_gate_table(gate_path, sections)
    summary["artifacts"] = {
        "gate_table": gate_path.name,
        "gate_table_sha256": _sha256(gate_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    manifest = {
        "summary_sha256": _sha256(summary_path),
        "gate_table_sha256": _sha256(gate_path),
        "input_sha256": {name: row["sha256"] for name, row in inputs.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deferred-reweight Stage-B smoke.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--keeper-robustness", type=Path, required=True)
    parser.add_argument("--paired-xai", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_audit(
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        trace_path=args.trace,
        robustness_path=args.robustness,
        keeper_robustness_path=args.keeper_robustness,
        paired_xai_path=args.paired_xai,
        history_path=args.history,
        checkpoint_path=args.checkpoint,
        cohort_path=args.cohort,
        output_dir=args.output_dir,
    )
    print(json.dumps({"decision": summary["decision"], "failed_sections": summary["failed_sections"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
