from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


EXPECTED_VAL_ROWS = 2606
EXPECTED_VAL_SUPPORT = (549, 151, 544, 712, 650)
FOCUS_CLASS = 1
FOCUS_FP_CLASSES = (0, 2, 4)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked full-validation comparison for the matched dense MHSA/VCA smoke."
    )
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--control-run-summary", type=Path, required=True)
    parser.add_argument("--candidate-run-summary", type=Path, required=True)
    parser.add_argument("--stage-a-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _probability_columns(fieldnames: Sequence[str]) -> list[str]:
    columns = [name for name in fieldnames if str(name).startswith("prob_")]
    columns.sort(key=lambda value: int(re.match(r"prob_(\d+)_", value).group(1)))
    if len(columns) != 5:
        raise ValueError(f"Expected five probability columns, found {columns}")
    return columns


def _read_predictions(path: Path) -> Dict[str, object]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        required = {
            "sample_index",
            "image_path",
            "source_stem",
            "object_index",
            "target_index",
            "prediction_index",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
        probability_columns = _probability_columns(fields)
        for raw in reader:
            probabilities = np.asarray(
                [float(raw[name]) for name in probability_columns],
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise ValueError("Probabilities must be finite and nonnegative")
            if not math.isclose(
                float(probabilities.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("Probability row is not normalized")
            prediction = int(raw["prediction_index"])
            if prediction != int(np.argmax(probabilities)):
                raise ValueError("prediction_index does not equal probability argmax")
            rows.append(
                {
                    "key": (
                        int(raw["sample_index"]),
                        str(raw["source_stem"]).casefold(),
                        int(raw["object_index"]),
                    ),
                    "image_path": str(Path(raw["image_path"]).resolve()),
                    "target": int(raw["target_index"]),
                    "prediction": prediction,
                    "probabilities": probabilities,
                }
            )
    if len(rows) != EXPECTED_VAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_VAL_ROWS} rows, found {len(rows)}")
    return {
        "rows": rows,
        "labels": np.asarray([row["target"] for row in rows], dtype=np.int64),
        "predictions": np.asarray(
            [row["prediction"] for row in rows], dtype=np.int64
        ),
        "probabilities": np.stack([row["probabilities"] for row in rows]),
    }


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(5),
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(5)
        ],
    }


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    epsilon = 1e-12
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == labels).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correctness[selected].mean()) - float(confidence[selected].mean())
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return {
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(labels.size), labels], epsilon, 1.0)).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_10": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


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
        "class1_precision_gain": class1_precision_gain >= 0.020,
        "class1_recall_preserved": class1_recall_delta >= -0.015,
        "focus_false_positives_reduced": int(transitions["focus_fp_reduction"]) >= 5,
        "focus_tp_breaks_bounded": int(transitions["focus_tp_breaks"])
        <= int(transitions["focus_fn_rescues"]) + 3,
        "net_corrections_positive": int(transitions["corrections"])
        > int(transitions["harms"]),
        "new_3_to_2_harms_bounded": int(transitions["new_3_to_2_harms"]) <= 5,
        "nonfocus_f1_preserved": max(nonfocus_f1_drops.values()) <= 0.020,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio) and runtime_ratio <= 1.35,
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
    new_3_to_2 = (labels == 3) & (control_prediction != 2) & (
        candidate_prediction == 2
    )
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

    control_run = json.loads(Path(args.control_run_summary).read_text(encoding="utf-8"))
    candidate_run = json.loads(Path(args.candidate_run_summary).read_text(encoding="utf-8"))
    stage_a = json.loads(Path(args.stage_a_summary).read_text(encoding="utf-8"))
    control_seconds = float(control_run["total_seconds"])
    candidate_seconds = float(candidate_run["total_seconds"])
    runtime_ratio = candidate_seconds / max(control_seconds, 1e-12)
    stage_a_peak = float(stage_a["cuda_amp"]["peak_vram_gib"])
    gate = assess_smoke_pair(
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        transitions=transitions,
        runtime_ratio=runtime_ratio,
        stage_a_peak_vram_gib=stage_a_peak,
        aligned_rows=len(control_rows),
    )

    changed_path = output_dir / "changed_cases.csv"
    probability_fields = [
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
    ]
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
        *probability_fields,
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
                row[f"control_prob_{class_index}"] = float(
                    left["probabilities"][class_index]
                )
                row[f"candidate_prob_{class_index}"] = float(
                    right["probabilities"][class_index]
                )
            writer.writerow(row)

    summary = {
        "method": "visual_contrast_attention",
        "protocol_stage": "B_matched_120b_2e_full_validation",
        "sources": {
            "control_predictions": str(Path(args.control_predictions).resolve()),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions": str(Path(args.candidate_predictions).resolve()),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "stage_a_summary": str(Path(args.stage_a_summary).resolve()),
            "stage_a_summary_sha256": _sha256(args.stage_a_summary),
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
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
