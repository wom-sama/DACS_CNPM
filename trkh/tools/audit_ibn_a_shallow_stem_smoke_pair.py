from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from trkh.tools.audit_visual_contrast_smoke_pair import (
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SUPPORT,
    FOCUS_CLASS,
    FOCUS_FP_CLASSES,
    _calibration,
    _metrics,
    _prepare_output_dir,
    _read_predictions,
    _sha256,
)


METHOD = "ibn_a_shallow_stem"
PROTOCOL_STAGE = "B_matched_keeper_resume_120b_2e_full_validation"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked full-validation audit for the shallow IBN-a smoke pair."
    )
    parser.add_argument("--keeper-predictions", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--locked-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _align_tables(*tables: Mapping[str, object]) -> None:
    rows = [table["rows"] for table in tables]
    if len({len(value) for value in rows}) != 1:
        raise ValueError("Prediction row counts differ")
    for aligned in zip(*rows):
        first = aligned[0]
        for other in aligned[1:]:
            if (
                first["key"] != other["key"]
                or first["image_path"] != other["image_path"]
                or first["target"] != other["target"]
            ):
                raise ValueError("Prediction alignment mismatch")


def _metrics_with_confusion(
    labels: np.ndarray, predictions: np.ndarray
) -> Dict[str, object]:
    result = _metrics(labels, predictions)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    result["confusion_matrix"] = confusion.tolist()
    return result


def _transition_stats(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, int]:
    focus_negative = np.isin(labels, FOCUS_FP_CLASSES)
    focus_positive = labels == FOCUS_CLASS
    corrections = (reference != labels) & (candidate == labels)
    harms = (reference == labels) & (candidate != labels)
    fp_removed = focus_negative & (reference == FOCUS_CLASS) & (
        candidate != FOCUS_CLASS
    )
    fp_created = focus_negative & (reference != FOCUS_CLASS) & (
        candidate == FOCUS_CLASS
    )
    fn_rescued = focus_positive & (reference != FOCUS_CLASS) & (
        candidate == FOCUS_CLASS
    )
    tp_broken = focus_positive & (reference == FOCUS_CLASS) & (
        candidate != FOCUS_CLASS
    )
    return {
        "changed_decisions": int(np.sum(reference != candidate)),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "focus_false_positives_removed": int(fp_removed.sum()),
        "focus_false_positives_created": int(fp_created.sum()),
        "focus_false_positive_net_reduction": int(fp_removed.sum() - fp_created.sum()),
        "focus_false_negatives_rescued": int(fn_rescued.sum()),
        "focus_true_positives_broken": int(tp_broken.sum()),
    }


def _deep_differences(left: object, right: object, path: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_deep_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _deep_differences(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    return [] if left == right else [path]


def _normalized_train_arguments(arguments: Sequence[object]) -> list[str]:
    normalized = [str(value) for value in arguments]
    for name in ("--run-name", "--stem-normalization"):
        try:
            index = normalized.index(name)
        except ValueError as exc:
            raise ValueError(f"Locked arguments omit {name}") from exc
        if index + 1 >= len(normalized) or normalized[index + 1].startswith("--"):
            raise ValueError(f"Locked argument {name} has no value")
        normalized[index + 1] = "<matched-role>"
    return normalized


def _trace_summary(run_dir: Path, expected_normalization: str) -> Dict[str, object]:
    trace_path = run_dir / "architecture_trace" / "trace_summary.json"
    trace = _load_json(trace_path)
    samples = trace.get("samples")
    model_config = trace.get("model_config")
    if not isinstance(samples, list) or not isinstance(model_config, Mapping):
        raise ValueError(f"Malformed architecture trace: {trace_path}")
    class_ids = sorted(int(sample["class_id"]) for sample in samples)
    return {
        "path": str(trace_path.resolve()),
        "sha256": _sha256(trace_path),
        "sample_count": len(samples),
        "class_ids": class_ids,
        "stem_normalization": model_config.get("stem_normalization", "batch"),
        "passed": len(samples) == 5
        and class_ids == list(range(5))
        and model_config.get("stem_normalization", "batch")
        == expected_normalization,
    }


def _contains_test_output(run_dir: Path, run_summary: Mapping[str, object]) -> bool:
    if run_summary.get("test_summary") is not None:
        return True
    forbidden_names = {"final_test", "test", "test_eval", "test_predictions"}
    return any(path.name.casefold() in forbidden_names for path in run_dir.iterdir())


def _validate_provenance(
    *,
    control_run_dir: Path,
    candidate_run_dir: Path,
    control_run: Mapping[str, object],
    candidate_run: Mapping[str, object],
    control_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    control_trace: Mapping[str, object],
    candidate_trace: Mapping[str, object],
    stage_a: Mapping[str, object],
    protocol: Mapping[str, object],
    stage_a_summary_path: Path,
) -> Dict[str, object]:
    gate = stage_a.get("gate")
    sources = stage_a.get("sources")
    if not isinstance(gate, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("Malformed Stage-A summary")
    control_args = protocol.get("control_arguments")
    candidate_args = protocol.get("candidate_arguments")
    if not isinstance(control_args, list) or not isinstance(candidate_args, list):
        raise ValueError("Locked protocol omits matched train arguments")

    config_differences = _deep_differences(control_config, candidate_config)
    allowed_config_differences = {
        "model_config.stem_normalization",
        "run_dir",
        "run_name",
    }
    control_model = control_config.get("model_config")
    candidate_model = candidate_config.get("model_config")
    if not isinstance(control_model, Mapping) or not isinstance(candidate_model, Mapping):
        raise ValueError("Resolved configs omit model_config")

    checks = {
        "stage_a_passed": stage_a.get("method") == METHOD
        and stage_a.get("status") == "passed"
        and bool(gate.get("smoke_permission"))
        and not bool(gate.get("full_train_permission")),
        "stage_a_train_only": not bool(sources.get("validation_loaded"))
        and not bool(sources.get("test_loaded")),
        "stage_a_hash_locked": protocol.get("stage_a_summary_sha256")
        == _sha256(stage_a_summary_path),
        "protocol_identity": protocol.get("method") == METHOD
        and protocol.get("protocol_stage") == PROTOCOL_STAGE
        and not bool(protocol.get("test_allowed")),
        "protocol_budget_locked": int(protocol.get("epochs", -1)) == 2
        and int(protocol.get("max_train_batches", -1)) == 120
        and int(protocol.get("max_val_batches", -1)) == 0
        and int(protocol.get("seed", -1)) == 42
        and int(protocol.get("batch_size", -1)) == 32
        and int(protocol.get("grad_accum_steps", -1)) == 2
        and int(protocol.get("num_workers", -1)) == 4
        and int(protocol.get("eval_num_workers", -1)) == 2,
        "arguments_matched": _normalized_train_arguments(control_args)
        == _normalized_train_arguments(candidate_args),
        "only_resolved_config_difference": set(config_differences)
        == allowed_config_differences,
        "normalization_values_locked": control_model.get("stem_normalization")
        == "batch"
        and candidate_model.get("stem_normalization") == "ibn_a_first",
        "parameter_count_identical": int(control_run.get("parameter_count", -1))
        == int(candidate_run.get("parameter_count", -2)),
        "architecture_traces_complete": bool(control_trace.get("passed"))
        and bool(candidate_trace.get("passed")),
        "no_test_output": not _contains_test_output(control_run_dir, control_run)
        and not _contains_test_output(candidate_run_dir, candidate_run),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not bool(value)],
        "resolved_config_differences": config_differences,
        "control_trace": control_trace,
        "candidate_trace": candidate_trace,
    }


def assess_ibn_a_smoke_pair(
    *,
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions: Mapping[str, int],
    runtime_ratio: float,
    aligned_rows: int,
    probabilities_valid: bool,
    provenance_valid: bool,
) -> Dict[str, object]:
    control_per_class = control_metrics["per_class"]
    candidate_per_class = candidate_metrics["per_class"]
    macro_f1_delta = float(candidate_metrics["macro_f1"]) - float(
        control_metrics["macro_f1"]
    )
    class1_precision_gain = float(candidate_per_class[1]["precision"]) - float(
        control_per_class[1]["precision"]
    )
    class1_recall_delta = float(candidate_per_class[1]["recall"]) - float(
        control_per_class[1]["recall"]
    )
    class1_f1_gain = float(candidate_per_class[1]["f1"]) - float(
        control_per_class[1]["f1"]
    )
    nonfocus_f1_drops = {
        str(index): float(control_per_class[index]["f1"])
        - float(candidate_per_class[index]["f1"])
        for index in (0, 2, 3, 4)
    }
    checks = {
        "full_validation_support": int(aligned_rows) == EXPECTED_VAL_ROWS,
        "finite_normalized_probabilities": bool(probabilities_valid),
        "provenance_and_trace_locked": bool(provenance_valid),
        "macro_f1_preserved": macro_f1_delta >= 0.0,
        "class1_f1_gain": class1_f1_gain >= 0.010,
        "class1_precision_gain": class1_precision_gain >= 0.020,
        "class1_recall_preserved": class1_recall_delta >= -0.015,
        "focus_false_positives_reduced": int(
            transitions["focus_false_positive_net_reduction"]
        )
        >= 4,
        "corrections_cover_harms": int(transitions["corrections"])
        >= int(transitions["harms"]),
        "fn_rescues_cover_tp_breaks": int(
            transitions["focus_false_negatives_rescued"]
        )
        >= int(transitions["focus_true_positives_broken"]),
        "nonfocus_f1_preserved": max(nonfocus_f1_drops.values()) <= 0.015,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio)
        and runtime_ratio <= 1.30,
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    return {
        "metric_gate_passed": not failed,
        "post_smoke_audit_required": not failed,
        "five_epoch_probe_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "route_closed": bool(failed),
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "macro_f1_delta": macro_f1_delta,
            "class1_precision_gain": class1_precision_gain,
            "class1_recall_delta": class1_recall_delta,
            "class1_f1_gain": class1_f1_gain,
            "nonfocus_f1_drops": nonfocus_f1_drops,
            "runtime_ratio": runtime_ratio,
        },
    }


def _source_group_forensics(
    *,
    rows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[Dict[str, object]], Dict[str, int]]:
    grouped: Dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["key"][1]), []).append(index)
    output = []
    for source_stem, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        changed = control[selected] != candidate[selected]
        if not changed.any():
            continue
        source_labels = labels[selected]
        source_control = control[selected]
        source_candidate = candidate[selected]
        transitions = _transition_stats(
            source_labels, source_control, source_candidate
        )
        output.append(
            {
                "source_stem": source_stem,
                "object_count": len(indices),
                "target_classes": ",".join(
                    str(value) for value in sorted(set(source_labels.tolist()))
                ),
                **transitions,
            }
        )
    return output, {
        "changed_source_groups": len(output),
        "changed_multi_object_source_groups": sum(
            int(int(row["object_count"]) > 1) for row in output
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    keeper = _read_predictions(args.keeper_predictions)
    control = _read_predictions(args.control_predictions)
    candidate = _read_predictions(args.candidate_predictions)
    _align_tables(keeper, control, candidate)
    labels = keeper["labels"]
    if np.bincount(labels, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from the locked cohort")
    if not all(np.array_equal(labels, table["labels"]) for table in (control, candidate)):
        raise ValueError("Aligned labels differ")

    probabilities_valid = all(
        bool(np.isfinite(table["probabilities"]).all())
        and bool(
            np.allclose(
                table["probabilities"].sum(axis=1),
                np.ones(len(labels)),
                atol=1e-5,
                rtol=0.0,
            )
        )
        for table in (keeper, control, candidate)
    )
    metrics = {
        name: _metrics_with_confusion(labels, table["predictions"])
        for name, table in (
            ("keeper", keeper),
            ("control", control),
            ("candidate", candidate),
        )
    }
    transitions = {
        "candidate_vs_control": _transition_stats(
            labels, control["predictions"], candidate["predictions"]
        ),
        "candidate_vs_keeper": _transition_stats(
            labels, keeper["predictions"], candidate["predictions"]
        ),
        "control_vs_keeper": _transition_stats(
            labels, keeper["predictions"], control["predictions"]
        ),
    }

    control_run_dir = args.control_run_dir.resolve()
    candidate_run_dir = args.candidate_run_dir.resolve()
    control_run = _load_json(control_run_dir / "summary.json")
    candidate_run = _load_json(candidate_run_dir / "summary.json")
    control_config = _load_json(control_run_dir / "resolved_config.json")
    candidate_config = _load_json(candidate_run_dir / "resolved_config.json")
    control_trace = _trace_summary(control_run_dir, "batch")
    candidate_trace = _trace_summary(candidate_run_dir, "ibn_a_first")
    stage_a = _load_json(args.stage_a_summary)
    protocol = _load_json(args.locked_protocol)
    provenance = _validate_provenance(
        control_run_dir=control_run_dir,
        candidate_run_dir=candidate_run_dir,
        control_run=control_run,
        candidate_run=candidate_run,
        control_config=control_config,
        candidate_config=candidate_config,
        control_trace=control_trace,
        candidate_trace=candidate_trace,
        stage_a=stage_a,
        protocol=protocol,
        stage_a_summary_path=args.stage_a_summary,
    )
    control_seconds = float(control_run["total_seconds"])
    candidate_seconds = float(candidate_run["total_seconds"])
    runtime_ratio = candidate_seconds / max(control_seconds, 1e-12)
    gate = assess_ibn_a_smoke_pair(
        control_metrics=metrics["control"],
        candidate_metrics=metrics["candidate"],
        transitions=transitions["candidate_vs_control"],
        runtime_ratio=runtime_ratio,
        aligned_rows=len(labels),
        probabilities_valid=probabilities_valid,
        provenance_valid=bool(provenance["passed"]),
    )

    changed_path = output_dir / "changed_cases.csv"
    fields = [
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "target",
        "keeper_prediction",
        "control_prediction",
        "candidate_prediction",
        "correction",
        "harm",
        "focus_fp_removed",
        "focus_fp_created",
        "focus_fn_rescue",
        "focus_tp_break",
        *(f"keeper_prob_{index}" for index in range(5)),
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
    ]
    changed_rows = 0
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for keeper_row, control_row, candidate_row in zip(
            keeper["rows"], control["rows"], candidate["rows"]
        ):
            control_prediction = int(control_row["prediction"])
            candidate_prediction = int(candidate_row["prediction"])
            if control_prediction == candidate_prediction:
                continue
            target = int(control_row["target"])
            focus_negative = target in FOCUS_FP_CLASSES
            row = {
                "sample_index": int(control_row["key"][0]),
                "source_stem": control_row["key"][1],
                "object_index": int(control_row["key"][2]),
                "image_path": control_row["image_path"],
                "target": target,
                "keeper_prediction": int(keeper_row["prediction"]),
                "control_prediction": control_prediction,
                "candidate_prediction": candidate_prediction,
                "correction": int(
                    control_prediction != target and candidate_prediction == target
                ),
                "harm": int(
                    control_prediction == target and candidate_prediction != target
                ),
                "focus_fp_removed": int(
                    focus_negative
                    and control_prediction == FOCUS_CLASS
                    and candidate_prediction != FOCUS_CLASS
                ),
                "focus_fp_created": int(
                    focus_negative
                    and control_prediction != FOCUS_CLASS
                    and candidate_prediction == FOCUS_CLASS
                ),
                "focus_fn_rescue": int(
                    target == FOCUS_CLASS
                    and control_prediction != FOCUS_CLASS
                    and candidate_prediction == FOCUS_CLASS
                ),
                "focus_tp_break": int(
                    target == FOCUS_CLASS
                    and control_prediction == FOCUS_CLASS
                    and candidate_prediction != FOCUS_CLASS
                ),
            }
            for class_index in range(5):
                row[f"keeper_prob_{class_index}"] = float(
                    keeper_row["probabilities"][class_index]
                )
                row[f"control_prob_{class_index}"] = float(
                    control_row["probabilities"][class_index]
                )
                row[f"candidate_prob_{class_index}"] = float(
                    candidate_row["probabilities"][class_index]
                )
            writer.writerow(row)
            changed_rows += 1
    if changed_rows != transitions["candidate_vs_control"]["changed_decisions"]:
        raise RuntimeError("Changed-case artifact disagrees with transition audit")

    source_rows, source_summary = _source_group_forensics(
        rows=control["rows"],
        labels=labels,
        control=control["predictions"],
        candidate=candidate["predictions"],
    )
    source_path = output_dir / "source_group_forensics.csv"
    source_fields = [
        "source_stem",
        "object_count",
        "target_classes",
        "changed_decisions",
        "corrections",
        "harms",
        "focus_false_positives_removed",
        "focus_false_positives_created",
        "focus_false_positive_net_reduction",
        "focus_false_negatives_rescued",
        "focus_true_positives_broken",
    ]
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(source_rows)

    summary = {
        "method": METHOD,
        "protocol_stage": PROTOCOL_STAGE,
        "sources": {
            "validation_rows": len(labels),
            "validation_support": np.bincount(labels, minlength=5).tolist(),
            "test_used": False,
            "raw_dataset_modified": False,
            "keeper_predictions_sha256": _sha256(args.keeper_predictions),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "stage_a_summary_sha256": _sha256(args.stage_a_summary),
            "locked_protocol_sha256": _sha256(args.locked_protocol),
        },
        "keeper": {
            "metrics": metrics["keeper"],
            "calibration": _calibration(labels, keeper["probabilities"]),
        },
        "control": {
            "metrics": metrics["control"],
            "calibration": _calibration(labels, control["probabilities"]),
            "runtime_seconds": control_seconds,
        },
        "candidate": {
            "metrics": metrics["candidate"],
            "calibration": _calibration(labels, candidate["probabilities"]),
            "runtime_seconds": candidate_seconds,
        },
        "transitions": transitions,
        "changed_case_rows": changed_rows,
        "source_group_forensics": source_summary,
        "provenance": provenance,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "artifacts": [
            {"path": str(summary_path), "role": "comparison", "sha256": _sha256(summary_path)},
            {"path": str(changed_path), "role": "changed_cases", "sha256": _sha256(changed_path)},
            {"path": str(source_path), "role": "source_groups", "sha256": _sha256(source_path)},
        ],
        "raw_dataset_modified": False,
        "test_used": False,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run_audit(parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
