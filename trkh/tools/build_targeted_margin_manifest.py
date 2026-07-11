from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def _cartography_row_is_unseen(row: Dict[str, str]) -> bool:
    if "seen_count" not in row:
        return False
    try:
        return float(str(row.get("seen_count", "") or "0").strip()) <= 0.0
    except ValueError:
        return True


def _parse_pairs(text: str) -> set[Tuple[int, int]]:
    pairs: set[Tuple[int, int]] = set()
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair {item!r}; expected A-B.")
        left, right = item.split("-", 1)
        a, b = int(left.strip()), int(right.strip())
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


def _bounded_weight(
    *,
    reason: str,
    confidence: Optional[float],
    false_positive_weight: float,
    false_negative_weight: float,
    low_margin_weight: float,
    max_weight: float,
) -> float:
    if reason == "focus_false_positive":
        base = false_positive_weight
    elif reason == "focus_false_negative":
        base = false_negative_weight
    else:
        base = low_margin_weight
    if confidence is not None and math.isfinite(confidence):
        base += 0.15 * max(0.0, min(1.0, confidence))
    return min(max_weight, max(1e-6, float(base)))


def build_manifest(
    *,
    predictions: Path,
    output: Path,
    focus_class_index: int,
    pairs: set[Tuple[int, int]],
    margin_threshold: float,
    target_margin: float,
    false_positive_weight: float,
    false_negative_weight: float,
    low_margin_weight: float,
    max_weight: float,
    max_samples: int,
    max_per_pair: int,
    include_low_margin_correct: bool,
    allow_non_train_paths: bool,
    dry_run: bool,
) -> Dict[str, object]:
    if not predictions.is_file():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions}")
    rows: List[Dict[str, str]] = []
    skipped_non_train = 0
    skipped_unseen = 0
    skipped_invalid = 0
    by_reason: Counter[str] = Counter()
    by_pair: Counter[str] = Counter()
    per_pair_kept: Dict[str, int] = defaultdict(int)
    seen_paths: set[str] = set()

    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Predictions CSV must have a header: {predictions}")
        for row in reader:
            image_path = _path_value(row)
            if not image_path:
                skipped_invalid += 1
                continue
            if _looks_like_non_train(image_path) and not allow_non_train_paths:
                skipped_non_train += 1
                continue
            if _cartography_row_is_unseen(row):
                skipped_unseen += 1
                continue
            target_index = _int_value(row, "target_index", "y_true", "label", "label_index")
            prediction_index = _int_value(row, "prediction_index", "y_pred", "pred", "pred_index")
            if target_index is None or prediction_index is None:
                skipped_invalid += 1
                continue
            if target_index == prediction_index and not include_low_margin_correct:
                continue
            if pairs and (int(target_index), int(prediction_index)) not in pairs:
                continue

            confidence = _float_value(row, "confidence", "pred_confidence", "probability")
            pred_margin = _float_value(row, "top2_margin", "margin", "probability_margin")
            if target_index != prediction_index:
                if prediction_index == focus_class_index and target_index != focus_class_index:
                    reason = "focus_false_positive"
                    negative_index = int(focus_class_index)
                elif target_index == focus_class_index and prediction_index != focus_class_index:
                    reason = "focus_false_negative"
                    negative_index = int(prediction_index)
                else:
                    continue
            else:
                if not include_low_margin_correct:
                    continue
                if target_index != focus_class_index and prediction_index != focus_class_index:
                    continue
                if pred_margin is None or pred_margin > margin_threshold:
                    continue
                reason = "focus_low_margin_correct"
                negative_index = int(focus_class_index if target_index != focus_class_index else prediction_index)
                if negative_index == target_index:
                    continue

            key = str(Path(image_path).resolve()).lower()
            if key in seen_paths:
                continue
            pair_key = f"{int(target_index)}->{int(negative_index)}"
            if max_per_pair > 0 and per_pair_kept[pair_key] >= max_per_pair:
                continue
            seen_paths.add(key)
            per_pair_kept[pair_key] += 1
            weight = _bounded_weight(
                reason=reason,
                confidence=confidence,
                false_positive_weight=false_positive_weight,
                false_negative_weight=false_negative_weight,
                low_margin_weight=low_margin_weight,
                max_weight=max_weight,
            )
            output_row = {
                "image_path": image_path,
                "target_index": str(int(target_index)),
                "negative_index": str(int(negative_index)),
                "prediction_index": str(int(prediction_index)),
                "target_name": str(row.get("target_name", "") or ""),
                "prediction_name": str(row.get("prediction_name", "") or ""),
                "confidence": "" if confidence is None else f"{confidence:.10g}",
                "prediction_margin": "" if pred_margin is None else f"{pred_margin:.10g}",
                "targeted_margin": f"{float(target_margin):.10g}",
                "targeted_margin_weight": f"{float(weight):.10g}",
                "reason": reason,
            }
            rows.append(output_row)
            by_reason[reason] += 1
            by_pair[pair_key] += 1
            if max_samples > 0 and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError("No valid targeted-margin rows were selected.")
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_path",
                    "target_index",
                    "negative_index",
                    "prediction_index",
                    "target_name",
                    "prediction_name",
                    "confidence",
                    "prediction_margin",
                    "targeted_margin",
                    "targeted_margin_weight",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
    return {
        "predictions": str(predictions.resolve()),
        "output": str(output.resolve()),
        "dry_run": bool(dry_run),
        "rows": int(len(rows)),
        "skipped_non_train": int(skipped_non_train),
        "skipped_unseen": int(skipped_unseen),
        "skipped_invalid": int(skipped_invalid),
        "focus_class_index": int(focus_class_index),
        "target_margin": float(target_margin),
        "max_samples": int(max_samples),
        "max_per_pair": int(max_per_pair),
        "include_low_margin_correct": bool(include_low_margin_correct),
        "by_reason": dict(by_reason),
        "by_target_negative_pair": dict(by_pair),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only directional margin manifest for hard class-boundary cases."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--pairs", type=str, default="0-1,1-2,1-4")
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--target-margin", type=float, default=0.12)
    parser.add_argument("--false-positive-weight", type=float, default=1.35)
    parser.add_argument("--false-negative-weight", type=float, default=1.10)
    parser.add_argument("--low-margin-weight", type=float, default=1.00)
    parser.add_argument("--max-weight", type=float, default=1.6)
    parser.add_argument("--max-samples", type=int, default=320)
    parser.add_argument("--max-per-pair", type=int, default=180)
    parser.add_argument("--include-low-margin-correct", action="store_true")
    parser.add_argument("--allow-non-train-paths", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        predictions=args.predictions,
        output=args.output,
        focus_class_index=args.focus_class_index,
        pairs=_parse_pairs(args.pairs),
        margin_threshold=args.margin_threshold,
        target_margin=args.target_margin,
        false_positive_weight=args.false_positive_weight,
        false_negative_weight=args.false_negative_weight,
        low_margin_weight=args.low_margin_weight,
        max_weight=args.max_weight,
        max_samples=max(0, int(args.max_samples)),
        max_per_pair=max(0, int(args.max_per_pair)),
        include_low_margin_correct=bool(args.include_low_margin_correct),
        allow_non_train_paths=bool(args.allow_non_train_paths),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
