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


METHOD = "natural_prior_full_model_continuation"
PROTOCOL_STAGE = "matched_60b_2e_full_validation"
KEEPER_MACRO_F1 = 0.8829248547554016
KEEPER_FOCUS_F1 = 0.678260862827301
FOCUS_MILESTONE = 0.70


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked full-validation audit for strict-balanced versus natural-prior "
            "continuation from the completed TRKH scratch checkpoint."
        )
    )
    parser.add_argument("--keeper-predictions", type=Path, required=True)
    parser.add_argument("--scratch-predictions", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--control-run-summary", type=Path, required=True)
    parser.add_argument("--candidate-run-summary", type=Path, required=True)
    parser.add_argument("--control-resolved-config", type=Path, required=True)
    parser.add_argument("--candidate-resolved-config", type=Path, required=True)
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
    changed = reference != candidate
    focus_negative = np.isin(labels, FOCUS_FP_CLASSES)
    focus_positive = labels == FOCUS_CLASS
    reference_correct = reference == labels
    candidate_correct = candidate == labels
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
        "changed_decisions": int(changed.sum()),
        "corrections": int((changed & ~reference_correct & candidate_correct).sum()),
        "harms": int((changed & reference_correct & ~candidate_correct).sum()),
        "neutral_changes": int(
            (changed & (reference_correct == candidate_correct)).sum()
        ),
        "focus_false_positives_removed": int(fp_removed.sum()),
        "focus_false_positives_created": int(fp_created.sum()),
        "focus_false_positive_net_reduction": int(
            fp_removed.sum() - fp_created.sum()
        ),
        "focus_false_negatives_rescued": int(fn_rescued.sum()),
        "focus_true_positives_broken": int(tp_broken.sum()),
        "new_3_to_2_harms": int(
            ((labels == 3) & (reference != 2) & (candidate == 2)).sum()
        ),
    }


def _focus(metrics: Mapping[str, object]) -> Mapping[str, object]:
    return metrics["per_class"][FOCUS_CLASS]


def assess_natural_prior_continuation(
    *,
    keeper_metrics: Mapping[str, object],
    scratch_metrics: Mapping[str, object],
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions_vs_control: Mapping[str, int],
    aligned_rows: int,
    probabilities_valid: bool,
    runtime_ratio: float,
) -> Dict[str, object]:
    scratch_focus = _focus(scratch_metrics)
    control_focus = _focus(control_metrics)
    candidate_focus = _focus(candidate_metrics)
    keeper_focus = _focus(keeper_metrics)
    nonfocus_drops = {
        str(index): float(control_metrics["per_class"][index]["f1"])
        - float(candidate_metrics["per_class"][index]["f1"])
        for index in (0, 2, 3, 4)
    }
    observed = {
        "control_macro_delta_vs_scratch": float(control_metrics["macro_f1"])
        - float(scratch_metrics["macro_f1"]),
        "control_focus_f1_delta_vs_scratch": float(control_focus["f1"])
        - float(scratch_focus["f1"]),
        "control_focus_recall_delta_vs_scratch": float(control_focus["recall"])
        - float(scratch_focus["recall"]),
        "candidate_macro_delta_vs_control": float(candidate_metrics["macro_f1"])
        - float(control_metrics["macro_f1"]),
        "candidate_macro_delta_vs_scratch": float(candidate_metrics["macro_f1"])
        - float(scratch_metrics["macro_f1"]),
        "candidate_focus_precision_delta_vs_control": float(
            candidate_focus["precision"]
        )
        - float(control_focus["precision"]),
        "candidate_focus_precision_delta_vs_scratch": float(
            candidate_focus["precision"]
        )
        - float(scratch_focus["precision"]),
        "candidate_focus_f1_delta_vs_control": float(candidate_focus["f1"])
        - float(control_focus["f1"]),
        "candidate_focus_f1_delta_vs_scratch": float(candidate_focus["f1"])
        - float(scratch_focus["f1"]),
        "candidate_focus_recall_delta_vs_scratch": float(candidate_focus["recall"])
        - float(scratch_focus["recall"]),
        "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
        "candidate_focus_precision": float(candidate_focus["precision"]),
        "candidate_focus_recall": float(candidate_focus["recall"]),
        "candidate_focus_f1": float(candidate_focus["f1"]),
        "keeper_macro_f1": float(keeper_metrics["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "maximum_nonfocus_f1_drop_vs_control": max(nonfocus_drops.values()),
        "nonfocus_f1_drops_vs_control": nonfocus_drops,
        "runtime_ratio": float(runtime_ratio),
    }
    checks = {
        "full_validation_support": int(aligned_rows) == EXPECTED_VAL_ROWS,
        "finite_normalized_probabilities": bool(probabilities_valid),
        "control_macro_preserved_vs_scratch": observed[
            "control_macro_delta_vs_scratch"
        ]
        >= -0.005,
        "control_focus_f1_preserved_vs_scratch": observed[
            "control_focus_f1_delta_vs_scratch"
        ]
        >= -0.010,
        "control_focus_recall_preserved_vs_scratch": observed[
            "control_focus_recall_delta_vs_scratch"
        ]
        >= -0.010,
        "candidate_focus_precision_gain_vs_control": observed[
            "candidate_focus_precision_delta_vs_control"
        ]
        >= 0.020,
        "candidate_focus_precision_gain_vs_scratch": observed[
            "candidate_focus_precision_delta_vs_scratch"
        ]
        >= 0.020,
        "candidate_focus_f1_gain_vs_control": observed[
            "candidate_focus_f1_delta_vs_control"
        ]
        >= 0.010,
        "candidate_focus_f1_gain_vs_scratch": observed[
            "candidate_focus_f1_delta_vs_scratch"
        ]
        >= 0.010,
        "candidate_focus_reaches_0p70": observed["candidate_focus_f1"]
        >= FOCUS_MILESTONE,
        "candidate_focus_recall_preserved_vs_scratch": observed[
            "candidate_focus_recall_delta_vs_scratch"
        ]
        >= -0.010,
        "candidate_macro_not_below_control": observed[
            "candidate_macro_delta_vs_control"
        ]
        >= 0.0,
        "candidate_macro_not_below_scratch": observed[
            "candidate_macro_delta_vs_scratch"
        ]
        >= 0.0,
        "restricted_focus_fp_net_reduction_ge_8": int(
            transitions_vs_control["focus_false_positive_net_reduction"]
        )
        >= 8,
        "corrections_gt_harms": int(transitions_vs_control["corrections"])
        > int(transitions_vs_control["harms"]),
        "focus_fn_rescues_gte_tp_breaks": int(
            transitions_vs_control["focus_false_negatives_rescued"]
        )
        >= int(transitions_vs_control["focus_true_positives_broken"]),
        "nonfocus_f1_preserved": observed[
            "maximum_nonfocus_f1_drop_vs_control"
        ]
        <= 0.010,
        "candidate_reaches_keeper_macro": observed["candidate_macro_f1"]
        >= KEEPER_MACRO_F1,
        "candidate_reaches_keeper_focus": observed["candidate_focus_f1"]
        >= KEEPER_FOCUS_F1,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio)
        and float(runtime_ratio) <= 1.20,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "probe_permission": not failed,
        "full_train_permission": False,
        "test_permission": False,
        "post_smoke_audit_required": True,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")


def _validate_provenance(
    *,
    control_run: Mapping[str, object],
    candidate_run: Mapping[str, object],
    control_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    if (
        protocol.get("method") != METHOD
        or protocol.get("protocol_stage") != PROTOCOL_STAGE
        or bool(protocol.get("test_allowed", True))
        or protocol.get("candidate_only_argument")
        != "--disable-balanced-epoch-sampling"
    ):
        raise ValueError("Locked protocol method/stage/test boundary mismatch")
    locked = {
        "seed": 42,
        "epochs": 2,
        "max_train_batches": 60,
        "max_val_batches": 0,
        "scheduler_total_epochs": 10,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "learning_rate": 8e-5,
        "min_learning_rate": 1e-6,
        "warmup_epochs": 1,
        "patience": 2,
        "weight_decay": 0.05,
    }
    for name, expected in locked.items():
        if protocol.get(name) != expected:
            raise ValueError(
                f"Protocol {name}={protocol.get(name)!r}, expected {expected!r}"
            )
    if control_run.get("test_summary") is not None or candidate_run.get(
        "test_summary"
    ) is not None:
        raise ValueError("Matched smoke contains final-test results")

    if control_config["model_config"] != candidate_config["model_config"]:
        raise ValueError("Control/candidate model configs differ")
    if control_config["augmentation_config"] != candidate_config[
        "augmentation_config"
    ]:
        raise ValueError("Control/candidate augmentation configs differ")

    for config in (control_config, candidate_config):
        train = config["train_config"]
        expected_train = {
            "seed": 42,
            "deterministic": True,
            "batch_size": 32,
            "grad_accum_steps": 2,
            "epochs": 2,
            "scheduler_total_epochs": 10,
            "early_stopping_patience": 2,
            "max_train_batches": 60,
            "max_val_batches": 0,
            "learning_rate": 8e-5,
            "min_learning_rate": 1e-6,
            "warmup_epochs": 1,
            "weight_decay": 0.05,
        }
        for name, expected in expected_train.items():
            if train.get(name) != expected:
                raise ValueError(
                    f"Resolved {name}={train.get(name)!r}, expected {expected!r}"
                )
        resume = config.get("resume", {})
        if not bool(resume.get("loaded", False)):
            raise ValueError("Matched smoke did not load the locked scratch checkpoint")
        for name in ("reset_epoch", "reset_optimizer", "reset_scheduler", "reset_scaler"):
            if not bool(resume.get(name, False)):
                raise ValueError(f"Matched smoke did not reset resume state: {name}")

    control_train = dict(control_config["train_config"])
    candidate_train = dict(candidate_config["train_config"])
    control_balanced = bool(control_train.pop("balanced_epoch_sampling"))
    candidate_balanced = bool(candidate_train.pop("balanced_epoch_sampling"))
    if not control_balanced or candidate_balanced:
        raise ValueError("Resolved sampler roles are not strict-balanced versus natural")
    if control_train != candidate_train:
        raise ValueError("Train configs differ beyond balanced_epoch_sampling")
    if control_config["imbalance_summary"].get("sampler_type") != "strict_balanced":
        raise ValueError("Control sampler telemetry is not strict_balanced")
    if candidate_config["imbalance_summary"].get("sampler_type") != "random":
        raise ValueError("Candidate sampler telemetry is not random")
    if not bool(control_config["data"]["balanced_epoch_exposure"].get("enabled")):
        raise ValueError("Control balanced exposure telemetry is disabled")
    if bool(candidate_config["data"]["balanced_epoch_exposure"].get("enabled")):
        raise ValueError("Candidate unexpectedly reports balanced exposure")


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
    output: list[Dict[str, object]] = []
    for source_stem, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        local_labels = labels[selected]
        local_control = control[selected]
        local_candidate = candidate[selected]
        stats = _transition_stats(local_labels, local_control, local_candidate)
        output.append(
            {
                "source_stem": source_stem,
                "object_count": len(indices),
                "target_classes": ",".join(
                    str(value) for value in sorted(set(local_labels.tolist()))
                ),
                "changed_decisions": stats["changed_decisions"],
                "correct_delta": int((local_candidate == local_labels).sum())
                - int((local_control == local_labels).sum()),
                "focus_fp_removed": stats["focus_false_positives_removed"],
                "focus_fp_created": stats["focus_false_positives_created"],
                "focus_fn_rescued": stats["focus_false_negatives_rescued"],
                "focus_tp_broken": stats["focus_true_positives_broken"],
            }
        )
    changed = [row for row in output if int(row["changed_decisions"]) > 0]
    return output, {
        "source_groups": len(output),
        "changed_source_groups": len(changed),
        "improved_changed_source_groups": sum(
            int(int(row["correct_delta"]) > 0) for row in changed
        ),
        "worsened_changed_source_groups": sum(
            int(int(row["correct_delta"]) < 0) for row in changed
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    keeper = _read_predictions(args.keeper_predictions)
    scratch = _read_predictions(args.scratch_predictions)
    control = _read_predictions(args.control_predictions)
    candidate = _read_predictions(args.candidate_predictions)
    _align_tables(keeper, scratch, control, candidate)
    labels = keeper["labels"]
    if np.bincount(labels, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from locked cohort")
    for table in (scratch, control, candidate):
        if not np.array_equal(labels, table["labels"]):
            raise ValueError("Aligned labels differ")

    protocol = _load_json(args.locked_protocol)
    _assert_file_hash(
        args.keeper_predictions,
        protocol["keeper_predictions_sha256"],
        "keeper predictions",
    )
    _assert_file_hash(
        args.scratch_predictions,
        protocol["scratch_predictions_sha256"],
        "scratch predictions",
    )
    control_run = _load_json(args.control_run_summary)
    candidate_run = _load_json(args.candidate_run_summary)
    control_config = _load_json(args.control_resolved_config)
    candidate_config = _load_json(args.candidate_resolved_config)
    _validate_provenance(
        control_run=control_run,
        candidate_run=candidate_run,
        control_config=control_config,
        candidate_config=candidate_config,
        protocol=protocol,
    )

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
        for table in (keeper, scratch, control, candidate)
    )
    metrics = {
        name: _metrics_with_confusion(labels, table["predictions"])
        for name, table in (
            ("keeper", keeper),
            ("scratch", scratch),
            ("control", control),
            ("candidate", candidate),
        )
    }
    transitions = {
        "candidate_vs_control": _transition_stats(
            labels, control["predictions"], candidate["predictions"]
        ),
        "candidate_vs_scratch": _transition_stats(
            labels, scratch["predictions"], candidate["predictions"]
        ),
        "candidate_vs_keeper": _transition_stats(
            labels, keeper["predictions"], candidate["predictions"]
        ),
        "control_vs_scratch": _transition_stats(
            labels, scratch["predictions"], control["predictions"]
        ),
    }
    runtime_ratio = float(candidate_run["total_seconds"]) / max(
        float(control_run["total_seconds"]), 1e-12
    )
    gate = assess_natural_prior_continuation(
        keeper_metrics=metrics["keeper"],
        scratch_metrics=metrics["scratch"],
        control_metrics=metrics["control"],
        candidate_metrics=metrics["candidate"],
        transitions_vs_control=transitions["candidate_vs_control"],
        aligned_rows=len(labels),
        probabilities_valid=probabilities_valid,
        runtime_ratio=runtime_ratio,
    )

    fields = [
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "target",
        "keeper_prediction",
        "scratch_prediction",
        "control_prediction",
        "candidate_prediction",
        "correction",
        "harm",
        "focus_fp_removed",
        "focus_fp_created",
        "focus_fn_rescue",
        "focus_tp_break",
        *(f"keeper_prob_{index}" for index in range(5)),
        *(f"scratch_prob_{index}" for index in range(5)),
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
    ]
    changed_path = output_dir / "changed_cases.csv"
    written = 0
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for keeper_row, scratch_row, control_row, candidate_row in zip(
            keeper["rows"], scratch["rows"], control["rows"], candidate["rows"]
        ):
            control_prediction = int(control_row["prediction"])
            candidate_prediction = int(candidate_row["prediction"])
            if control_prediction == candidate_prediction:
                continue
            target = int(keeper_row["target"])
            focus_negative = target in FOCUS_FP_CLASSES
            row = {
                "sample_index": int(keeper_row["key"][0]),
                "source_stem": keeper_row["key"][1],
                "object_index": int(keeper_row["key"][2]),
                "image_path": keeper_row["image_path"],
                "target": target,
                "keeper_prediction": int(keeper_row["prediction"]),
                "scratch_prediction": int(scratch_row["prediction"]),
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
                row[f"scratch_prob_{class_index}"] = float(
                    scratch_row["probabilities"][class_index]
                )
                row[f"control_prob_{class_index}"] = float(
                    control_row["probabilities"][class_index]
                )
                row[f"candidate_prob_{class_index}"] = float(
                    candidate_row["probabilities"][class_index]
                )
            writer.writerow(row)
            written += 1
    expected_changed = int(
        transitions["candidate_vs_control"]["changed_decisions"]
    )
    if written != expected_changed:
        raise RuntimeError(f"Changed-case rows {written} != {expected_changed}")

    group_rows, group_summary = _source_group_forensics(
        rows=keeper["rows"],
        labels=labels,
        control=control["predictions"],
        candidate=candidate["predictions"],
    )
    groups_path = output_dir / "source_group_forensics.csv"
    with groups_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    summary = {
        "method": METHOD,
        "protocol_stage": PROTOCOL_STAGE,
        "sources": {
            "validation_rows": len(labels),
            "test_used": False,
            "keeper_predictions_sha256": _sha256(args.keeper_predictions),
            "scratch_predictions_sha256": _sha256(args.scratch_predictions),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "locked_protocol_sha256": _sha256(args.locked_protocol),
        },
        "metrics": metrics,
        "calibration": {
            name: _calibration(labels, table["probabilities"])
            for name, table in (
                ("keeper", keeper),
                ("scratch", scratch),
                ("control", control),
                ("candidate", candidate),
            )
        },
        "runtime_seconds": {
            "control": float(control_run["total_seconds"]),
            "candidate": float(candidate_run["total_seconds"]),
            "ratio": runtime_ratio,
        },
        "transitions": transitions,
        "changed_case_rows": written,
        "source_group_forensics": group_summary,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = [
        "# Natural-Prior Continuation Smoke Audit",
        "",
        f"- Keeper macro/class1 F1: `{metrics['keeper']['macro_f1']:.6f}/{_focus(metrics['keeper'])['f1']:.6f}`",
        f"- Scratch macro/class1 F1: `{metrics['scratch']['macro_f1']:.6f}/{_focus(metrics['scratch'])['f1']:.6f}`",
        f"- Control macro/class1 F1: `{metrics['control']['macro_f1']:.6f}/{_focus(metrics['control'])['f1']:.6f}`",
        f"- Candidate macro/class1 F1: `{metrics['candidate']['macro_f1']:.6f}/{_focus(metrics['candidate'])['f1']:.6f}`",
        f"- Candidate class1 precision/recall: `{_focus(metrics['candidate'])['precision']:.6f}/{_focus(metrics['candidate'])['recall']:.6f}`",
        f"- Changed/corrections/harms: `{written}/{transitions['candidate_vs_control']['corrections']}/{transitions['candidate_vs_control']['harms']}`",
        f"- Restricted FP remove/create: `{transitions['candidate_vs_control']['focus_false_positives_removed']}/{transitions['candidate_vs_control']['focus_false_positives_created']}`",
        f"- FN rescue/TP break: `{transitions['candidate_vs_control']['focus_false_negatives_rescued']}/{transitions['candidate_vs_control']['focus_true_positives_broken']}`",
        f"- Probe permission: `{str(bool(gate['probe_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "Test was not loaded. Post-smoke robustness and paired XAI remain mandatory.",
    ]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "raw_dataset_modified": False,
        "test_used": False,
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path)}
            for path in (summary_path, report_path, changed_path, groups_path)
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run_audit(parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
