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


METHOD = "learnable_gabor_lho_fcm_edge_token_residual"
KEEPER_MACRO_F1 = 0.8829248547554016
KEEPER_CLASS1_PRECISION = 0.6030927896499634
KEEPER_CLASS1_RECALL = 0.7748344540596008
KEEPER_CLASS1_F1 = 0.678260862827301
MIN_MACRO_F1 = 0.882925
MIN_CLASS1_PRECISION = 0.613093
MIN_CLASS1_RECALL = 0.774834
MIN_CLASS1_F1 = 0.683261
MIN_CLASS1_TP = 117


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked keeper/control/candidate audit for Gabor Stage-B smoke."
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
    new_3_to_2 = (labels == 3) & (reference != 2) & (candidate == 2)
    return {
        "changed_decisions": int(np.sum(reference != candidate)),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "focus_false_positives_removed": int(fp_removed.sum()),
        "focus_false_positives_created": int(fp_created.sum()),
        "focus_false_positive_net_reduction": int(fp_removed.sum() - fp_created.sum()),
        "focus_false_negatives_rescued": int(fn_rescued.sum()),
        "focus_true_positives_broken": int(tp_broken.sum()),
        "new_3_to_2_harms": int(new_3_to_2.sum()),
    }


def _metrics_with_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, object]:
    result = _metrics(labels, predictions)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    result["confusion_matrix"] = confusion.tolist()
    return result


def assess_smoke_pair(
    *,
    keeper_metrics: Mapping[str, object],
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions_vs_keeper: Mapping[str, int],
    runtime_ratio: float,
    peak_vram_gib: float,
    aligned_rows: int,
    probabilities_valid: bool,
) -> Dict[str, object]:
    keeper_class = keeper_metrics["per_class"]
    control_class = control_metrics["per_class"]
    candidate_class = candidate_metrics["per_class"]
    candidate_tp = int(candidate_metrics["confusion_matrix"][1][1])
    keeper_tp = int(keeper_metrics["confusion_matrix"][1][1])
    precision_gain = float(candidate_class[1]["precision"]) - float(
        control_class[1]["precision"]
    )
    f1_gain = float(candidate_class[1]["f1"]) - float(control_class[1]["f1"])
    nonfocus_drops = {
        str(index): float(keeper_class[index]["f1"])
        - float(candidate_class[index]["f1"])
        for index in (0, 2, 3, 4)
    }
    checks = {
        "full_validation_support": int(aligned_rows) == EXPECTED_VAL_ROWS,
        "finite_normalized_probabilities": bool(probabilities_valid),
        "macro_f1_absolute": float(candidate_metrics["macro_f1"]) >= MIN_MACRO_F1,
        "macro_f1_vs_control": float(candidate_metrics["macro_f1"])
        >= float(control_metrics["macro_f1"]),
        "class1_precision_absolute": float(candidate_class[1]["precision"])
        >= MIN_CLASS1_PRECISION,
        "class1_precision_vs_control": precision_gain >= 0.010,
        "class1_recall_absolute": float(candidate_class[1]["recall"])
        >= MIN_CLASS1_RECALL,
        "class1_tp_absolute": candidate_tp >= MIN_CLASS1_TP,
        "class1_tp_vs_keeper": candidate_tp >= keeper_tp,
        "class1_f1_absolute": float(candidate_class[1]["f1"]) >= MIN_CLASS1_F1,
        "class1_f1_vs_control": f1_gain >= 0.005,
        "keeper_focus_fp_removed": int(
            transitions_vs_keeper["focus_false_positives_removed"]
        )
        >= 4,
        "corrections_exceed_harms": int(transitions_vs_keeper["corrections"])
        > int(transitions_vs_keeper["harms"]),
        "fn_rescues_cover_tp_breaks": int(
            transitions_vs_keeper["focus_false_negatives_rescued"]
        )
        >= int(transitions_vs_keeper["focus_true_positives_broken"]),
        "new_3_to_2_harms_bounded": int(transitions_vs_keeper["new_3_to_2_harms"])
        <= 3,
        "nonfocus_f1_preserved": max(nonfocus_drops.values()) <= 0.010,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio) and runtime_ratio <= 1.50,
        "stage_a_vram_bounded": math.isfinite(peak_vram_gib)
        and peak_vram_gib < 7.75,
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    return {
        "metric_gate_passed": not failed,
        "post_smoke_audit_required": True,
        "five_epoch_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "candidate_class1_tp": candidate_tp,
            "keeper_class1_tp": keeper_tp,
            "class1_precision_gain_vs_control": precision_gain,
            "class1_f1_gain_vs_control": f1_gain,
            "nonfocus_f1_drops_vs_keeper": nonfocus_drops,
            "runtime_ratio": runtime_ratio,
            "stage_a_peak_vram_gib": peak_vram_gib,
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
    if stage_a.get("method") != METHOD or not bool(stage_a["gate"]["smoke_permission"]):
        raise ValueError("Stage-A method/permission mismatch")
    if bool(stage_a["sources"]["validation_loaded"]) or bool(
        stage_a["sources"]["test_loaded"]
    ):
        raise ValueError("Stage-A is not train-only")
    if protocol.get("method") != METHOD or bool(protocol.get("test_allowed", True)):
        raise ValueError("Locked protocol method/test boundary mismatch")
    if control_run.get("test_summary") is not None or candidate_run.get(
        "test_summary"
    ) is not None:
        raise ValueError("Smoke run contains final-test results")

    control_model = dict(control_config["model_config"])
    candidate_model = dict(candidate_config["model_config"])
    if bool(control_model.get("learnable_gabor_texture_residual", False)):
        raise ValueError("Control unexpectedly enables Gabor residual")
    if not bool(candidate_model.get("learnable_gabor_texture_residual", False)):
        raise ValueError("Candidate does not enable Gabor residual")
    control_model["learnable_gabor_texture_residual"] = False
    candidate_model["learnable_gabor_texture_residual"] = False
    if control_model != candidate_model:
        raise ValueError("Control/candidate model configs differ beyond Gabor flag")

    locked_train = {
        "seed": 42,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "epochs": 2,
        "scheduler_total_epochs": 15,
        "warmup_epochs": 1,
        "max_train_batches": 120,
        "max_val_batches": 0,
        "learning_rate": 8e-5,
        "bbox_token_prior_source": "crop_bbox",
    }
    for config in (control_config, candidate_config):
        train = config["train_config"]
        for name, expected in locked_train.items():
            if train.get(name) != expected:
                raise ValueError(f"Resolved {name}={train.get(name)!r}, expected {expected!r}")
        if not bool(config["model_config"].get("token_pruning")):
            raise ValueError("Token pruning must remain enabled")
        resume = config.get("resume", {})
        if not bool(resume.get("loaded")) or not all(
            bool(resume.get(name))
            for name in ("reset_epoch", "reset_optimizer", "reset_scheduler", "reset_scaler")
        ):
            raise ValueError("Keeper resume/reset protocol mismatch")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    keeper = _read_predictions(args.keeper_predictions)
    control = _read_predictions(args.control_predictions)
    candidate = _read_predictions(args.candidate_predictions)
    _align_tables(keeper, control, candidate)
    labels = keeper["labels"]
    if np.bincount(labels, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from locked cohort")
    for table in (control, candidate):
        if not np.array_equal(labels, table["labels"]):
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
        for name, table in (("keeper", keeper), ("control", control), ("candidate", candidate))
    }
    transitions = {
        "candidate_vs_keeper": _transition_stats(
            labels, keeper["predictions"], candidate["predictions"]
        ),
        "candidate_vs_control": _transition_stats(
            labels, control["predictions"], candidate["predictions"]
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
    peak_vram = float(stage_a["cuda_bf16"]["candidate"]["peak_vram_gib"])
    gate = assess_smoke_pair(
        keeper_metrics=metrics["keeper"],
        control_metrics=metrics["control"],
        candidate_metrics=metrics["candidate"],
        transitions_vs_keeper=transitions["candidate_vs_keeper"],
        runtime_ratio=runtime_ratio,
        peak_vram_gib=peak_vram,
        aligned_rows=len(labels),
        probabilities_valid=probabilities_valid,
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
        "focus_fn_rescue",
        "focus_tp_break",
        "new_3_to_2_harm",
        "candidate_vs_keeper_correction",
        "candidate_vs_keeper_harm",
        *(f"keeper_prob_{index}" for index in range(5)),
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
    ]
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (keeper_row, control_row, candidate_row) in enumerate(
            zip(keeper["rows"], control["rows"], candidate["rows"])
        ):
            keeper_prediction = int(keeper_row["prediction"])
            control_prediction = int(control_row["prediction"])
            candidate_prediction = int(candidate_row["prediction"])
            if (
                keeper_prediction == candidate_prediction
                and control_prediction == candidate_prediction
            ):
                continue
            target = int(keeper_row["target"])
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
                "new_3_to_2_harm": int(
                    target == 3 and control_prediction != 2 and candidate_prediction == 2
                ),
                "candidate_vs_keeper_correction": int(
                    keeper_prediction != target and candidate_prediction == target
                ),
                "candidate_vs_keeper_harm": int(
                    keeper_prediction == target and candidate_prediction != target
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

    summary = {
        "method": METHOD,
        "protocol_stage": "B_matched_keeper_initialized_120b_2e_full_validation",
        "sources": {
            "validation_rows": len(labels),
            "test_used": False,
            "keeper_predictions_sha256": _sha256(args.keeper_predictions),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "control_resolved_config_sha256": _sha256(args.control_resolved_config),
            "candidate_resolved_config_sha256": _sha256(args.candidate_resolved_config),
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
            "runtime_seconds": float(control_run["total_seconds"]),
        },
        "candidate": {
            "metrics": metrics["candidate"],
            "calibration": _calibration(labels, candidate["probabilities"]),
            "runtime_seconds": float(candidate_run["total_seconds"]),
        },
        "transitions": transitions,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "artifacts": [
            {"path": str(summary_path), "sha256": _sha256(summary_path)},
            {"path": str(changed_path), "sha256": _sha256(changed_path)},
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
