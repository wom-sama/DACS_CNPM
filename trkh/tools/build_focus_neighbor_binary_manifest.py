from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _path_value(row: Dict[str, str]) -> str:
    return (
        str(row.get("image_path", "") or "").strip()
        or str(row.get("path", "") or "").strip()
        or str(row.get("sample_path", "") or "").strip()
    )


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


def _probability_columns(fieldnames: Sequence[str]) -> Dict[int, str]:
    columns: Dict[int, str] = {}
    for name in fieldnames:
        normalized = str(name or "").strip().lower()
        if not normalized.startswith("prob_"):
            continue
        parts = normalized.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            class_index = int(parts[1])
        except ValueError:
            continue
        columns.setdefault(class_index, str(name))
    return columns


def _parse_classes(text: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def _top_indices(row: Dict[str, str], probability_columns: Dict[int, str]) -> Tuple[int, int, float]:
    scored: List[Tuple[float, int]] = []
    for class_index, column in probability_columns.items():
        try:
            probability = float(str(row.get(column, "") or "nan"))
        except ValueError:
            continue
        if math.isfinite(probability):
            scored.append((float(probability), int(class_index)))
    scored.sort(reverse=True)
    if not scored:
        prediction = _int_value(row, "prediction_index", "y_pred", "pred_index", "pred")
        return int(prediction if prediction is not None else -1), -1, 0.0
    top1_probability, top1_index = scored[0]
    if len(scored) == 1:
        return int(top1_index), -1, float(top1_probability)
    top2_probability, top2_index = scored[1]
    return int(top1_index), int(top2_index), float(top1_probability - top2_probability)


def _bounded_weight(base: float, confidence: Optional[float], max_weight: float) -> float:
    weight = float(base)
    if confidence is not None and math.isfinite(confidence):
        weight += 0.10 * max(0.0, min(1.0, float(confidence)))
    return min(float(max_weight), max(1e-6, weight))


def build_manifest(
    *,
    predictions: Path,
    output_dir: Path,
    focus_class_index: int,
    neighbor_classes: Sequence[int],
    margin_threshold: float,
    false_positive_weight: float,
    false_negative_weight: float,
    low_margin_positive_weight: float,
    low_margin_negative_weight: float,
    positive_anchor_weight: float,
    include_focus_positive_anchors: bool,
    max_weight: float,
    max_samples: int,
    max_per_bucket: int,
    dry_run: bool,
) -> Dict[str, object]:
    if not predictions.is_file():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions}")
    focus_class_index = int(focus_class_index)
    neighbors = [int(value) for value in neighbor_classes if int(value) != focus_class_index]
    if not neighbors:
        raise ValueError("neighbor_classes must contain at least one non-focus class.")

    rows_out: List[Dict[str, str]] = []
    by_reason: Counter[str] = Counter()
    by_neighbor: Counter[str] = Counter()
    bucket_counts: Dict[str, int] = defaultdict(int)
    skipped_non_train = 0
    skipped_invalid = 0
    seen_paths: set[str] = set()

    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Predictions CSV must have a header: {predictions}")
        probability_columns = _probability_columns(reader.fieldnames)
        for row in reader:
            image_path = _path_value(row)
            if not image_path:
                skipped_invalid += 1
                continue
            if _looks_like_non_train(image_path):
                skipped_non_train += 1
                continue
            key = str(Path(image_path).resolve()).lower()
            if key in seen_paths:
                continue
            target = _int_value(row, "target_index", "y_true", "label", "label_index")
            prediction = _int_value(row, "prediction_index", "y_pred", "pred", "pred_index")
            if target is None or prediction is None:
                skipped_invalid += 1
                continue
            top1, top2, margin = _top_indices(row, probability_columns)
            confidence = _float_value(row, "confidence", "pred_confidence", "probability")
            binary_target: Optional[float] = None
            neighbor_index: Optional[int] = None
            reason = ""
            base_weight = 1.0

            if target == focus_class_index and prediction != focus_class_index:
                if prediction in neighbors:
                    binary_target = 1.0
                    neighbor_index = int(prediction)
                    reason = "focus_false_negative"
                    base_weight = float(false_negative_weight)
            elif prediction == focus_class_index and target != focus_class_index:
                if target in neighbors:
                    binary_target = 0.0
                    neighbor_index = int(target)
                    reason = "focus_false_positive"
                    base_weight = float(false_positive_weight)
            elif target == prediction and margin <= float(margin_threshold):
                if target == focus_class_index and top2 in neighbors:
                    binary_target = 1.0
                    neighbor_index = int(top2)
                    reason = "focus_low_margin_positive"
                    base_weight = float(low_margin_positive_weight)
                elif target in neighbors and top2 == focus_class_index:
                    binary_target = 0.0
                    neighbor_index = int(target)
                    reason = "focus_low_margin_negative"
                    base_weight = float(low_margin_negative_weight)
            elif include_focus_positive_anchors and target == prediction == focus_class_index:
                if top2 in neighbors:
                    binary_target = 1.0
                    neighbor_index = int(top2)
                    reason = "focus_positive_anchor"
                    base_weight = float(positive_anchor_weight)

            if binary_target is None or neighbor_index is None or not reason:
                continue
            bucket = f"{reason}:{neighbor_index}:{int(binary_target >= 0.5)}"
            if max_per_bucket > 0 and bucket_counts[bucket] >= int(max_per_bucket):
                continue
            weight = _bounded_weight(
                base=base_weight,
                confidence=confidence,
                max_weight=max_weight,
            )
            rows_out.append(
                {
                    "image_path": image_path,
                    "target_index": str(int(target)),
                    "prediction_index": str(int(prediction)),
                    "binary_target": f"{float(binary_target):.1f}",
                    "neighbor_index": str(int(neighbor_index)),
                    "binary_weight": f"{float(weight):.10g}",
                    "confidence": "" if confidence is None else f"{float(confidence):.10g}",
                    "top1_index": str(int(top1)),
                    "top2_index": str(int(top2)),
                    "top2_margin": f"{float(margin):.10g}",
                    "reason": reason,
                }
            )
            seen_paths.add(key)
            bucket_counts[bucket] += 1
            by_reason[reason] += 1
            by_neighbor[str(int(neighbor_index))] += 1
            if max_samples > 0 and len(rows_out) >= int(max_samples):
                break

    if not rows_out:
        raise ValueError("No valid focus-neighbor binary rows were selected.")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "focus_neighbor_binary_train_only.csv"
    summary_path = output_dir / "summary.json"
    summary = {
        "predictions": str(predictions.resolve()),
        "manifest": str(manifest_path.resolve()),
        "dry_run": bool(dry_run),
        "rows": int(len(rows_out)),
        "focus_class_index": int(focus_class_index),
        "neighbor_classes": [int(value) for value in neighbors],
        "margin_threshold": float(margin_threshold),
        "include_focus_positive_anchors": bool(include_focus_positive_anchors),
        "max_samples": int(max_samples),
        "max_per_bucket": int(max_per_bucket),
        "by_reason": dict(by_reason),
        "by_neighbor_index": dict(by_neighbor),
        "skipped_non_train": int(skipped_non_train),
        "skipped_invalid": int(skipped_invalid),
        "leakage_guard": "non-train paths are skipped; output is train-only",
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image_path",
            "target_index",
            "prediction_index",
            "binary_target",
            "neighbor_index",
            "binary_weight",
            "confidence",
            "top1_index",
            "top2_index",
            "top2_margin",
            "reason",
        ]
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-only focus-vs-neighbor binary manifest from detailed predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--neighbor-classes", type=str, default="0,2,4")
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--false-positive-weight", type=float, default=1.55)
    parser.add_argument("--false-negative-weight", type=float, default=1.20)
    parser.add_argument("--low-margin-positive-weight", type=float, default=1.05)
    parser.add_argument("--low-margin-negative-weight", type=float, default=1.15)
    parser.add_argument("--positive-anchor-weight", type=float, default=0.70)
    parser.add_argument("--include-focus-positive-anchors", action="store_true")
    parser.add_argument("--max-weight", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=520)
    parser.add_argument("--max-per-bucket", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        predictions=args.predictions,
        output_dir=args.output_dir,
        focus_class_index=args.focus_class_index,
        neighbor_classes=_parse_classes(args.neighbor_classes),
        margin_threshold=max(0.0, float(args.margin_threshold)),
        false_positive_weight=max(0.0, float(args.false_positive_weight)),
        false_negative_weight=max(0.0, float(args.false_negative_weight)),
        low_margin_positive_weight=max(0.0, float(args.low_margin_positive_weight)),
        low_margin_negative_weight=max(0.0, float(args.low_margin_negative_weight)),
        positive_anchor_weight=max(0.0, float(args.positive_anchor_weight)),
        include_focus_positive_anchors=bool(args.include_focus_positive_anchors),
        max_weight=max(1e-6, float(args.max_weight)),
        max_samples=max(0, int(args.max_samples)),
        max_per_bucket=max(0, int(args.max_per_bucket)),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
