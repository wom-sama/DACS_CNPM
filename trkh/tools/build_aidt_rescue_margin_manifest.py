from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _looks_like_non_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    return "val" in parts or "valid" in parts or "validation" in parts or "test" in parts


def _int_value(row: Dict[str, str], *names: str) -> Optional[int]:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _float_value(row: Dict[str, str], *names: str) -> Optional[float]:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _path_value(row: Dict[str, str]) -> str:
    return (
        str(row.get("image_path", "") or "").strip()
        or str(row.get("path", "") or "").strip()
        or str(row.get("sample_path", "") or "").strip()
    )


def _probability(row: Dict[str, str], class_index: int) -> Optional[float]:
    plain = _float_value(row, f"prob_{int(class_index)}")
    if plain is not None:
        return plain
    prefix = f"prob_{int(class_index)}_"
    for key, value in row.items():
        if str(key).startswith(prefix):
            parsed = _float_value({key: value}, key)
            if parsed is not None:
                return parsed
    return None


def _prediction_from_probs(row: Dict[str, str], num_classes: int) -> Tuple[Optional[int], Optional[float], float]:
    probs: List[float] = []
    for class_index in range(int(num_classes)):
        value = _probability(row, class_index)
        if value is None:
            return None, None, 0.0
        probs.append(float(value))
    if not probs:
        return None, None, 0.0
    ordered = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    top_index, confidence = ordered[0]
    margin = confidence - (ordered[1][1] if len(ordered) > 1 else 0.0)
    return int(top_index), float(confidence), float(margin)


def _load_teacher_by_sample_index(
    teacher_csv: Path,
    *,
    num_classes: int,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    if not teacher_csv.is_file():
        raise FileNotFoundError(f"Missing teacher CSV: {teacher_csv}")
    teacher_by_index: Dict[int, Dict[str, object]] = {}
    duplicate_rows = 0
    invalid_rows = 0
    with teacher_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Teacher CSV must have a header: {teacher_csv}")
        for row in reader:
            sample_index = _int_value(row, "sample_index", "dataset_index")
            if sample_index is None or sample_index < 0:
                invalid_rows += 1
                continue
            pred_from_probs, confidence_from_probs, margin = _prediction_from_probs(row, num_classes)
            prediction = _int_value(row, "prediction_index", "pred_index", "y_pred")
            confidence = _float_value(row, "confidence", "top1_probability", "probability")
            if prediction is None:
                prediction = pred_from_probs
            if confidence is None:
                confidence = confidence_from_probs
            if prediction is None or confidence is None:
                invalid_rows += 1
                continue
            probabilities = [
                float(_probability(row, class_index) or 0.0)
                for class_index in range(int(num_classes))
            ]
            if sample_index in teacher_by_index:
                duplicate_rows += 1
                previous = teacher_by_index[sample_index]
                if float(previous["confidence"]) >= float(confidence):
                    continue
            teacher_by_index[int(sample_index)] = {
                "prediction_index": int(prediction),
                "confidence": float(confidence),
                "margin": float(margin),
                "probabilities": probabilities,
            }
    return teacher_by_index, {
        "teacher_csv": str(teacher_csv.resolve()),
        "teacher_rows": int(len(teacher_by_index)),
        "teacher_duplicate_rows": int(duplicate_rows),
        "teacher_invalid_rows": int(invalid_rows),
    }


def _parse_int_set(text: str) -> set[int]:
    values: set[int] = set()
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.add(int(item))
    return values


def _candidate_weight(
    *,
    confidence: float,
    margin: float,
    base_weight: float,
    confidence_weight: float,
    margin_weight: float,
    max_weight: float,
) -> float:
    confidence_term = max(0.0, min(1.0, confidence))
    margin_term = max(0.0, min(1.0, margin / 0.5))
    weight = float(base_weight) + float(confidence_weight) * confidence_term + float(margin_weight) * margin_term
    return min(float(max_weight), max(1e-6, weight))


def build_manifest(
    *,
    baseline_predictions: Path,
    teacher_csv: Path,
    output: Path,
    num_classes: int,
    focus_class_index: int,
    negative_classes: set[int],
    min_teacher_confidence: float,
    min_teacher_margin: float,
    target_margin: float,
    false_positive_base_weight: float,
    false_negative_base_weight: float,
    confidence_weight: float,
    margin_weight: float,
    max_weight: float,
    max_samples: int,
    max_per_pair: int,
    allow_non_train_paths: bool,
    dry_run: bool,
) -> Dict[str, object]:
    if not baseline_predictions.is_file():
        raise FileNotFoundError(f"Missing baseline predictions CSV: {baseline_predictions}")
    teacher_by_index, teacher_summary = _load_teacher_by_sample_index(
        teacher_csv,
        num_classes=int(num_classes),
    )
    candidates: List[Dict[str, object]] = []
    skipped_missing_teacher = 0
    skipped_non_train = 0
    skipped_invalid = 0
    skipped_base_correct = 0
    skipped_teacher_not_correct = 0
    skipped_teacher_low_confidence = 0
    skipped_outside_scope = 0
    with baseline_predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Baseline CSV must have a header: {baseline_predictions}")
        for row in reader:
            image_path = _path_value(row)
            if not image_path:
                skipped_invalid += 1
                continue
            if _looks_like_non_train(image_path) and not allow_non_train_paths:
                skipped_non_train += 1
                continue
            sample_index = _int_value(row, "sample_index", "dataset_index")
            target_index = _int_value(row, "target_index", "y_true", "label", "label_index")
            baseline_prediction = _int_value(row, "prediction_index", "y_pred", "pred", "pred_index")
            if sample_index is None or target_index is None or baseline_prediction is None:
                skipped_invalid += 1
                continue
            if int(target_index) == int(baseline_prediction):
                skipped_base_correct += 1
                continue
            teacher = teacher_by_index.get(int(sample_index))
            if teacher is None:
                skipped_missing_teacher += 1
                continue
            teacher_prediction = int(teacher["prediction_index"])
            teacher_confidence = float(teacher["confidence"])
            teacher_margin = float(teacher["margin"])
            if teacher_prediction != int(target_index):
                skipped_teacher_not_correct += 1
                continue
            if teacher_confidence < float(min_teacher_confidence) or teacher_margin < float(min_teacher_margin):
                skipped_teacher_low_confidence += 1
                continue
            if int(target_index) == int(focus_class_index) and int(baseline_prediction) in negative_classes:
                reason = "aidt_rescue_false_negative"
                negative_index = int(baseline_prediction)
                base_weight = float(false_negative_base_weight)
            elif int(baseline_prediction) == int(focus_class_index) and int(target_index) in negative_classes:
                reason = "aidt_suppress_false_positive"
                negative_index = int(focus_class_index)
                base_weight = float(false_positive_base_weight)
            else:
                skipped_outside_scope += 1
                continue
            baseline_focus_probability = _probability(row, int(focus_class_index))
            teacher_focus_probability = None
            probabilities = teacher.get("probabilities", [])
            if isinstance(probabilities, list) and 0 <= int(focus_class_index) < len(probabilities):
                teacher_focus_probability = float(probabilities[int(focus_class_index)])
            weight = _candidate_weight(
                confidence=teacher_confidence,
                margin=teacher_margin,
                base_weight=base_weight,
                confidence_weight=float(confidence_weight),
                margin_weight=float(margin_weight),
                max_weight=float(max_weight),
            )
            score = teacher_confidence + 0.5 * teacher_margin + 0.1 * weight
            candidates.append(
                {
                    "score": float(score),
                    "sample_index": int(sample_index),
                    "image_path": image_path,
                    "target_index": int(target_index),
                    "negative_index": int(negative_index),
                    "prediction_index": int(baseline_prediction),
                    "target_name": str(row.get("target_name", "") or ""),
                    "prediction_name": str(row.get("prediction_name", "") or ""),
                    "confidence": str(row.get("confidence", "") or ""),
                    "baseline_focus_probability": (
                        "" if baseline_focus_probability is None else f"{float(baseline_focus_probability):.10g}"
                    ),
                    "teacher_prediction_index": int(teacher_prediction),
                    "teacher_confidence": float(teacher_confidence),
                    "teacher_margin": float(teacher_margin),
                    "teacher_focus_probability": (
                        "" if teacher_focus_probability is None else f"{float(teacher_focus_probability):.10g}"
                    ),
                    "targeted_margin": float(target_margin),
                    "targeted_margin_weight": float(weight),
                    "reason": reason,
                }
            )

    candidates.sort(key=lambda item: (-float(item["score"]), int(item["sample_index"])))
    rows: List[Dict[str, object]] = []
    per_pair: Dict[str, int] = defaultdict(int)
    seen_indices: set[int] = set()
    for candidate in candidates:
        sample_index = int(candidate["sample_index"])
        if sample_index in seen_indices:
            continue
        pair_key = f"{int(candidate['target_index'])}->{int(candidate['negative_index'])}"
        if max_per_pair > 0 and per_pair[pair_key] >= int(max_per_pair):
            continue
        seen_indices.add(sample_index)
        per_pair[pair_key] += 1
        rows.append(candidate)
        if max_samples > 0 and len(rows) >= int(max_samples):
            break

    if not rows:
        raise ValueError("No AIDT rescue targeted-margin rows were selected.")
    by_reason = Counter(str(row["reason"]) for row in rows)
    by_pair = Counter(f"{int(row['target_index'])}->{int(row['negative_index'])}" for row in rows)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_index",
            "image_path",
            "target_index",
            "negative_index",
            "prediction_index",
            "target_name",
            "prediction_name",
            "confidence",
            "baseline_focus_probability",
            "teacher_prediction_index",
            "teacher_confidence",
            "teacher_margin",
            "teacher_focus_probability",
            "targeted_margin",
            "targeted_margin_weight",
            "reason",
            "score",
        ]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            f"{float(row[key]):.10g}"
                            if isinstance(row.get(key), float)
                            else row.get(key, "")
                        )
                        for key in fieldnames
                    }
                )
    return {
        **teacher_summary,
        "baseline_predictions": str(baseline_predictions.resolve()),
        "output": str(output.resolve()),
        "dry_run": bool(dry_run),
        "candidate_rows": int(len(candidates)),
        "rows": int(len(rows)),
        "focus_class_index": int(focus_class_index),
        "negative_classes": sorted(int(value) for value in negative_classes),
        "min_teacher_confidence": float(min_teacher_confidence),
        "min_teacher_margin": float(min_teacher_margin),
        "target_margin": float(target_margin),
        "max_samples": int(max_samples),
        "max_per_pair": int(max_per_pair),
        "by_reason": dict(by_reason),
        "by_target_negative_pair": dict(by_pair),
        "skipped": {
            "missing_teacher": int(skipped_missing_teacher),
            "non_train": int(skipped_non_train),
            "invalid": int(skipped_invalid),
            "base_correct": int(skipped_base_correct),
            "teacher_not_correct": int(skipped_teacher_not_correct),
            "teacher_low_confidence": int(skipped_teacher_low_confidence),
            "outside_scope": int(skipped_outside_scope),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only targeted-margin manifest from cases where baseline TRKH is wrong "
            "and AIDT teacher is correct/confident."
        )
    )
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--negative-classes", type=str, default="0,2,4")
    parser.add_argument("--min-teacher-confidence", type=float, default=0.82)
    parser.add_argument("--min-teacher-margin", type=float, default=0.30)
    parser.add_argument("--target-margin", type=float, default=0.10)
    parser.add_argument("--false-positive-base-weight", type=float, default=0.70)
    parser.add_argument("--false-negative-base-weight", type=float, default=0.55)
    parser.add_argument("--confidence-weight", type=float, default=0.20)
    parser.add_argument("--margin-weight", type=float, default=0.15)
    parser.add_argument("--max-weight", type=float, default=1.25)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-per-pair", type=int, default=256)
    parser.add_argument("--allow-non-train-paths", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        baseline_predictions=args.baseline_predictions,
        teacher_csv=args.teacher_csv,
        output=args.output,
        num_classes=args.num_classes,
        focus_class_index=args.focus_class_index,
        negative_classes=_parse_int_set(args.negative_classes),
        min_teacher_confidence=args.min_teacher_confidence,
        min_teacher_margin=args.min_teacher_margin,
        target_margin=args.target_margin,
        false_positive_base_weight=args.false_positive_base_weight,
        false_negative_base_weight=args.false_negative_base_weight,
        confidence_weight=args.confidence_weight,
        margin_weight=args.margin_weight,
        max_weight=args.max_weight,
        max_samples=args.max_samples,
        max_per_pair=args.max_per_pair,
        allow_non_train_paths=args.allow_non_train_paths,
        dry_run=args.dry_run,
    )
    if args.summary_output is not None and not args.dry_run:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
