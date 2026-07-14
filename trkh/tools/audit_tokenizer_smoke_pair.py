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


DEFAULT_METHOD = "inceptionnext_atto_surface_tokenizer"
DEFAULT_CANDIDATE_STEM = "inceptionnext_atto_tokenizer"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked full-validation comparison for a matched tokenizer smoke."
    )
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--control-run-summary", type=Path, required=True)
    parser.add_argument("--candidate-run-summary", type=Path, required=True)
    parser.add_argument("--control-resolved-config", type=Path, required=True)
    parser.add_argument("--candidate-resolved-config", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--locked-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-method", default=DEFAULT_METHOD)
    parser.add_argument("--candidate-stem", default=DEFAULT_CANDIDATE_STEM)
    return parser.parse_args(argv)


def assess_smoke_pair(
    *,
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions: Mapping[str, int],
    runtime_ratio: float,
    stage_a_peak_vram_gib: float,
    aligned_rows: int,
) -> Dict[str, object]:
    control_per_class = control_metrics["per_class"]
    candidate_per_class = candidate_metrics["per_class"]
    class1_precision_gain = float(candidate_per_class[1]["precision"]) - float(
        control_per_class[1]["precision"]
    )
    class1_recall_delta = float(candidate_per_class[1]["recall"]) - float(
        control_per_class[1]["recall"]
    )
    class1_f1_gain = float(candidate_per_class[1]["f1"]) - float(
        control_per_class[1]["f1"]
    )
    macro_f1_delta = float(candidate_metrics["macro_f1"]) - float(
        control_metrics["macro_f1"]
    )
    nonfocus_f1_drops = {
        str(index): float(control_per_class[index]["f1"])
        - float(candidate_per_class[index]["f1"])
        for index in (0, 2, 3, 4)
    }
    checks = {
        "full_validation_support": int(aligned_rows) == EXPECTED_VAL_ROWS,
        "macro_f1_preserved": macro_f1_delta >= -0.003,
        "class1_f1_gain": class1_f1_gain >= 0.010,
        "class1_precision_gain": class1_precision_gain >= 0.025,
        "class1_recall_preserved": class1_recall_delta >= -0.020,
        "focus_false_positives_reduced": int(transitions["focus_fp_reduction"]) >= 5,
        "focus_tp_breaks_bounded": int(transitions["focus_tp_breaks"])
        <= int(transitions["focus_fn_rescues"]) + 3,
        "net_corrections_positive": int(transitions["corrections"])
        > int(transitions["harms"]),
        "new_3_to_2_harms_bounded": int(transitions["new_3_to_2_harms"]) <= 5,
        "nonfocus_f1_preserved": max(nonfocus_f1_drops.values()) <= 0.020,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio) and runtime_ratio <= 1.50,
        "stage_a_vram_bounded": math.isfinite(stage_a_peak_vram_gib)
        and stage_a_peak_vram_gib <= 7.75,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "metric_gate_passed": not failed,
        "five_epoch_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "macro_f1_delta": macro_f1_delta,
            "class1_precision_gain": class1_precision_gain,
            "class1_recall_delta": class1_recall_delta,
            "class1_f1_gain": class1_f1_gain,
            "nonfocus_f1_drops": nonfocus_f1_drops,
            "runtime_ratio": float(runtime_ratio),
            "stage_a_peak_vram_gib": float(stage_a_peak_vram_gib),
        },
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_provenance(
    *,
    control_run: Mapping[str, object],
    candidate_run: Mapping[str, object],
    control_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    stage_a: Mapping[str, object],
    protocol: Mapping[str, object],
    expected_method: str,
    candidate_stem: str,
) -> None:
    if stage_a.get("method") != expected_method:
        raise ValueError("Stage-A method does not match the locked tokenizer method")
    stage_gate = stage_a.get("gate", {})
    if not isinstance(stage_gate, Mapping) or not bool(stage_gate.get("smoke_permission")):
        raise ValueError("Stage-A did not grant smoke permission")
    stage_sources = stage_a.get("sources", {})
    if not isinstance(stage_sources, Mapping):
        raise ValueError("Stage-A source provenance is missing")
    if bool(stage_sources.get("validation_loaded")) or bool(stage_sources.get("test_loaded")):
        raise ValueError("Stage-A must be train-only")
    if protocol.get("method") != expected_method or bool(protocol.get("test_allowed", True)):
        raise ValueError("Locked protocol method/test boundary mismatch")
    if control_run.get("test_summary") is not None or candidate_run.get("test_summary") is not None:
        raise ValueError("Matched smoke must not contain final-test results")

    expected_stems = ((control_config, "conv_pool"), (candidate_config, candidate_stem))
    for config, expected_stem in expected_stems:
        model = config.get("model_config", {})
        train = config.get("train_config", {})
        if not isinstance(model, Mapping) or not isinstance(train, Mapping):
            raise ValueError("Resolved config is missing model/train configuration")
        observed_stem = str(model.get("stem_architecture", ""))
        if observed_stem != expected_stem:
            raise ValueError(f"Unexpected stem architecture: {observed_stem}")
        locked_train = {
            "seed": 42,
            "batch_size": 32,
            "grad_accum_steps": 2,
            "epochs": 2,
            "scheduler_total_epochs": 15,
            "warmup_epochs": 1,
            "max_train_batches": 120,
            "max_val_batches": 0,
        }
        for name, expected in locked_train.items():
            if train.get(name) != expected:
                raise ValueError(f"Resolved {name}={train.get(name)!r}, expected {expected!r}")
        for name in (
            "attention_view_loss_weight",
            "attention_crop_probability",
            "attention_drop_probability",
        ):
            if float(train.get(name, math.nan)) != 0.0:
                raise ValueError(f"Resolved {name} must be disabled")
        if not bool(model.get("token_pruning")):
            raise ValueError("Source token pruning must remain enabled")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    control = _read_predictions(args.control_predictions)
    candidate = _read_predictions(args.candidate_predictions)
    control_rows = control["rows"]
    candidate_rows = candidate["rows"]
    for left, right in zip(control_rows, candidate_rows):
        if (
            left["key"] != right["key"]
            or left["image_path"] != right["image_path"]
            or left["target"] != right["target"]
        ):
            raise ValueError(f"Prediction alignment mismatch: {left['key']} vs {right['key']}")

    labels = control["labels"]
    if np.bincount(labels, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation class support differs from the locked cohort")
    control_prediction = control["predictions"]
    candidate_prediction = candidate["predictions"]
    control_metrics = _metrics(labels, control_prediction)
    candidate_metrics = _metrics(labels, candidate_prediction)

    focus_negative = np.isin(labels, FOCUS_FP_CLASSES)
    control_focus_fp = int(np.sum(focus_negative & (control_prediction == FOCUS_CLASS)))
    candidate_focus_fp = int(np.sum(focus_negative & (candidate_prediction == FOCUS_CLASS)))
    corrections = (control_prediction != labels) & (candidate_prediction == labels)
    harms = (control_prediction == labels) & (candidate_prediction != labels)
    focus_positive = labels == FOCUS_CLASS
    focus_rescues = focus_positive & (control_prediction != FOCUS_CLASS) & (
        candidate_prediction == FOCUS_CLASS
    )
    focus_breaks = focus_positive & (control_prediction == FOCUS_CLASS) & (
        candidate_prediction != FOCUS_CLASS
    )
    new_3_to_2 = (labels == 3) & (control_prediction != 2) & (candidate_prediction == 2)
    transitions = {
        "changed_decisions": int(np.sum(control_prediction != candidate_prediction)),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "focus_fn_rescues": int(focus_rescues.sum()),
        "focus_tp_breaks": int(focus_breaks.sum()),
        "control_focus_fp_0_2_4_to_1": control_focus_fp,
        "candidate_focus_fp_0_2_4_to_1": candidate_focus_fp,
        "focus_fp_reduction": int(control_focus_fp - candidate_focus_fp),
        "new_3_to_2_harms": int(new_3_to_2.sum()),
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
        expected_method=str(args.expected_method),
        candidate_stem=str(args.candidate_stem),
    )
    control_seconds = float(control_run["total_seconds"])
    candidate_seconds = float(candidate_run["total_seconds"])
    runtime_ratio = candidate_seconds / max(control_seconds, 1e-12)
    stage_a_peak = float(stage_a["cuda_amp"]["candidate"]["peak_vram_gib"])
    gate = assess_smoke_pair(
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        transitions=transitions,
        runtime_ratio=runtime_ratio,
        stage_a_peak_vram_gib=stage_a_peak,
        aligned_rows=len(control_rows),
    )

    changed_path = output_dir / "changed_cases.csv"
    fields = [
        "sample_index",
        "source_stem",
        "object_index",
        "image_path",
        "target",
        "control_prediction",
        "candidate_prediction",
        "correction",
        "harm",
        "focus_fn_rescue",
        "focus_tp_break",
        "new_3_to_2_harm",
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
    ]
    with changed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (left, right) in enumerate(zip(control_rows, candidate_rows)):
            if int(left["prediction"]) == int(right["prediction"]):
                continue
            row = {
                "sample_index": int(left["key"][0]),
                "source_stem": left["key"][1],
                "object_index": int(left["key"][2]),
                "image_path": left["image_path"],
                "target": int(left["target"]),
                "control_prediction": int(left["prediction"]),
                "candidate_prediction": int(right["prediction"]),
                "correction": int(corrections[index]),
                "harm": int(harms[index]),
                "focus_fn_rescue": int(focus_rescues[index]),
                "focus_tp_break": int(focus_breaks[index]),
                "new_3_to_2_harm": int(new_3_to_2[index]),
            }
            for class_index in range(5):
                row[f"control_prob_{class_index}"] = float(left["probabilities"][class_index])
                row[f"candidate_prob_{class_index}"] = float(right["probabilities"][class_index])
            writer.writerow(row)

    summary = {
        "method": str(args.expected_method),
        "protocol_stage": "B_matched_120b_2e_full_validation",
        "sources": {
            "control_predictions": str(Path(args.control_predictions).resolve()),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions": str(Path(args.candidate_predictions).resolve()),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "control_resolved_config_sha256": _sha256(args.control_resolved_config),
            "candidate_resolved_config_sha256": _sha256(args.candidate_resolved_config),
            "stage_a_summary": str(Path(args.stage_a_summary).resolve()),
            "stage_a_summary_sha256": _sha256(args.stage_a_summary),
            "locked_protocol_sha256": _sha256(args.locked_protocol),
            "validation_rows": len(control_rows),
            "test_used": False,
        },
        "control": {
            "metrics": control_metrics,
            "calibration": _calibration(labels, control["probabilities"]),
            "runtime_seconds": control_seconds,
        },
        "candidate": {
            "metrics": candidate_metrics,
            "calibration": _calibration(labels, candidate["probabilities"]),
            "runtime_seconds": candidate_seconds,
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
