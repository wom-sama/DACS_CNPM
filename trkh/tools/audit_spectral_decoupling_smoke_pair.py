from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from trkh.core.utils import json_dump
from trkh.tools.audit_active_gabor_semantic_fusion_smoke_pair import (
    _align_tables,
    _metrics_with_confusion,
    _source_group_forensics,
    _transition_stats,
)
from trkh.tools.audit_spectral_decoupling_preflight import METHOD
from trkh.tools.audit_visual_contrast_smoke_pair import (
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SUPPORT,
    FOCUS_CLASS,
    FOCUS_FP_CLASSES,
    _calibration,
    _prepare_output_dir,
    _read_predictions,
    _sha256,
)


PROTOCOL_STAGE = "B_deterministic_scratch_120b_5e_full_validation"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked full-validation Spectral Decoupling pair audit."
    )
    parser.add_argument("--keeper-predictions", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--control-run-summary", type=Path, required=True)
    parser.add_argument("--candidate-run-summary", type=Path, required=True)
    parser.add_argument("--control-resolved-config", type=Path, required=True)
    parser.add_argument("--candidate-resolved-config", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--locked-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def assess_spectral_decoupling_pair(
    *,
    keeper_metrics: Mapping[str, object],
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions: Mapping[str, int],
    runtime_ratio: float,
    aligned_rows: int,
    probabilities_valid: bool,
) -> Dict[str, object]:
    keeper_class = keeper_metrics["per_class"]
    control_class = control_metrics["per_class"]
    candidate_class = candidate_metrics["per_class"]
    macro_delta = float(candidate_metrics["macro_f1"]) - float(
        control_metrics["macro_f1"]
    )
    precision_delta = float(candidate_class[1]["precision"]) - float(
        control_class[1]["precision"]
    )
    recall_delta = float(candidate_class[1]["recall"]) - float(
        control_class[1]["recall"]
    )
    f1_delta = float(candidate_class[1]["f1"]) - float(control_class[1]["f1"])
    restricted_fp_reduction = int(
        transitions["focus_false_positive_net_reduction"]
    )
    checks = {
        "full_validation_support": aligned_rows == EXPECTED_VAL_ROWS,
        "finite_normalized_probabilities": bool(probabilities_valid),
        "class1_precision_delta_gte_0p005": precision_delta >= 0.005,
        "class1_f1_delta_gte_0p005": f1_delta >= 0.005,
        "macro_f1_delta_gte_neg_0p001": macro_delta >= -0.001,
        "class1_recall_delta_gte_neg_0p010": recall_delta >= -0.010,
        "restricted_class1_fp_reduction_gte_2": restricted_fp_reduction >= 2,
        "corrections_gte_harms": int(transitions["corrections"])
        >= int(transitions["harms"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "metric_gate_passed": not failed,
        "post_smoke_audit_required": not failed,
        "longer_probe_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "macro_f1_delta": macro_delta,
            "class1_precision_delta": precision_delta,
            "class1_recall_delta": recall_delta,
            "class1_f1_delta": f1_delta,
            "restricted_class1_fp_net_reduction": restricted_fp_reduction,
            "runtime_ratio_diagnostic": runtime_ratio,
            "runtime_ratio_finite": math.isfinite(runtime_ratio),
            "keeper_macro_f1": float(keeper_metrics["macro_f1"]),
            "keeper_class1_precision": float(keeper_class[1]["precision"]),
            "keeper_class1_recall": float(keeper_class[1]["recall"]),
            "keeper_class1_f1": float(keeper_class[1]["f1"]),
        },
    }


def _validate_provenance(
    *,
    control_run: Mapping[str, object],
    candidate_run: Mapping[str, object],
    control_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    stage_a: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    if (
        stage_a.get("method") != METHOD
        or not bool(stage_a.get("gate", {}).get("smoke_permission"))
        or bool(stage_a.get("sources", {}).get("validation_loaded", True))
        or bool(stage_a.get("sources", {}).get("test_loaded", True))
    ):
        raise ValueError("Stage-A evidence does not authorize the matched pair")
    if (
        protocol.get("method") != METHOD
        or protocol.get("protocol_stage") != PROTOCOL_STAGE
        or bool(protocol.get("test_allowed", True))
    ):
        raise ValueError("Locked protocol method/stage/test boundary mismatch")
    expected_protocol = {
        "seed": 42,
        "epochs": 5,
        "max_train_batches": 120,
        "max_val_batches": 0,
        "scheduler_total_epochs": 30,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "learning_rate": 2.5e-4,
        "min_learning_rate": 1e-6,
        "warmup_epochs": 1,
        "patience": 3,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "spectral_decoupling_lambda": 0.01,
        "bbox_token_prior_source": "crop_bbox",
        "num_workers": 4,
        "eval_num_workers": 2,
        "deterministic": True,
    }
    for name, expected in expected_protocol.items():
        if protocol.get(name) != expected:
            raise ValueError(
                f"Protocol {name}={protocol.get(name)!r}, expected {expected!r}"
            )
    if control_run.get("test_summary") is not None or candidate_run.get(
        "test_summary"
    ) is not None:
        raise ValueError("Matched pair contains forbidden final-test output")

    control_model = dict(control_config["model_config"])
    candidate_model = dict(candidate_config["model_config"])
    if control_model != candidate_model or bool(control_model.get("pretrained", True)):
        raise ValueError("Model configs are not identical scratch models")

    control_train = dict(control_config["train_config"])
    candidate_train = dict(candidate_config["train_config"])
    if control_train.get("classification_loss") not in {"cross_entropy", "ce"}:
        raise ValueError("Control does not use ordinary cross entropy")
    if candidate_train.get("classification_loss") != "spectral_decoupling":
        raise ValueError("Candidate does not use Spectral Decoupling")
    for train in (control_train, candidate_train):
        train["classification_loss"] = "<matched-loss-role>"
    if control_train != candidate_train:
        raise ValueError("Train configs differ beyond the locked loss role")
    for name, expected in expected_protocol.items():
        if name == "bbox_token_prior_source":
            continue
        if name in control_train and control_train.get(name) != expected:
            raise ValueError(
                f"Resolved train config {name}={control_train.get(name)!r}, expected {expected!r}"
            )


def _write_changed_cases(
    *,
    path: Path,
    keeper: Mapping[str, object],
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> int:
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
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for keeper_row, control_row, candidate_row in zip(
            keeper["rows"], control["rows"], candidate["rows"]
        ):
            control_prediction = int(control_row["prediction"])
            candidate_prediction = int(candidate_row["prediction"])
            if control_prediction == candidate_prediction:
                continue
            target = int(keeper_row["target"])
            keeper_prediction = int(keeper_row["prediction"])
            focus_negative = target in FOCUS_FP_CLASSES
            row = {
                "sample_index": int(keeper_row["key"][0]),
                "source_stem": keeper_row["key"][1],
                "object_index": int(keeper_row["key"][2]),
                "image_path": keeper_row["image_path"],
                "target": target,
                "keeper_prediction": keeper_prediction,
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
            count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = _prepare_output_dir(args.output_dir)
    keeper = _read_predictions(args.keeper_predictions)
    control = _read_predictions(args.control_predictions)
    candidate = _read_predictions(args.candidate_predictions)
    _align_tables(keeper, control, candidate)
    labels = keeper["labels"]
    if tuple(np.bincount(labels, minlength=5).tolist()) != EXPECTED_VAL_SUPPORT:
        raise ValueError("Validation support differs from the locked yolo_f split")
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
    calibration = {
        name: _calibration(labels, table["probabilities"])
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

    control_run = _load_json(args.control_run_summary)
    candidate_run = _load_json(args.candidate_run_summary)
    control_config = _load_json(args.control_resolved_config)
    candidate_config = _load_json(args.candidate_resolved_config)
    stage_a = _load_json(args.stage_a_summary)
    protocol = _load_json(args.locked_protocol)
    _validate_provenance(
        control_run=control_run,
        candidate_run=candidate_run,
        control_config=control_config,
        candidate_config=candidate_config,
        stage_a=stage_a,
        protocol=protocol,
    )
    runtime_ratio = float(candidate_run["total_seconds"]) / max(
        float(control_run["total_seconds"]), 1e-12
    )
    gate = assess_spectral_decoupling_pair(
        keeper_metrics=metrics["keeper"],
        control_metrics=metrics["control"],
        candidate_metrics=metrics["candidate"],
        transitions=transitions["candidate_vs_control"],
        runtime_ratio=runtime_ratio,
        aligned_rows=len(labels),
        probabilities_valid=probabilities_valid,
    )

    changed_path = output_dir / "changed_cases.csv"
    changed_count = _write_changed_cases(
        path=changed_path,
        keeper=keeper,
        control=control,
        candidate=candidate,
    )
    expected_changed = int(
        transitions["candidate_vs_control"]["changed_decisions"]
    )
    if changed_count != expected_changed:
        raise RuntimeError(
            f"Changed-case artifact mismatch: {changed_count} != {expected_changed}"
        )

    source_rows, source_summary = _source_group_forensics(
        rows=keeper["rows"],
        labels=labels,
        control=control["predictions"],
        candidate=candidate["predictions"],
    )
    source_path = output_dir / "source_group_forensics.csv"
    source_fields = list(source_rows[0]) if source_rows else []
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(source_rows)

    summary = {
        "method": METHOD,
        "protocol_stage": PROTOCOL_STAGE,
        "sources": {
            "validation_rows": len(labels),
            "validation_support": list(EXPECTED_VAL_SUPPORT),
            "test_used": False,
            "raw_dataset_modified": False,
            "keeper_predictions_sha256": _sha256(args.keeper_predictions),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "control_resolved_config_sha256": _sha256(args.control_resolved_config),
            "candidate_resolved_config_sha256": _sha256(
                args.candidate_resolved_config
            ),
            "stage_a_summary_sha256": _sha256(args.stage_a_summary),
            "locked_protocol_sha256": _sha256(args.locked_protocol),
        },
        "metrics": metrics,
        "calibration": calibration,
        "transitions": transitions,
        "source_group_forensics": source_summary,
        "gate": gate,
        "artifacts": {
            "changed_cases": str(changed_path.resolve()),
            "changed_cases_sha256": _sha256(changed_path),
            "source_group_forensics": str(source_path.resolve()),
            "source_group_forensics_sha256": _sha256(source_path),
        },
    }
    json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
