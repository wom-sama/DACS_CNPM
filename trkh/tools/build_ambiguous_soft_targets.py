from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_pairs(text: str) -> set[Tuple[int, int]]:
    pairs: set[Tuple[int, int]] = set()
    for raw_item in str(text or "").replace(";", ",").split(","):
        item = raw_item.strip().replace(":", "-")
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair: {raw_item!r}")
        left, right = [part.strip() for part in item.split("-", 1)]
        pair = tuple(sorted((int(left), int(right))))
        pairs.add(pair)  # type: ignore[arg-type]
    if not pairs:
        raise ValueError("--pairs must contain at least one pair.")
    return pairs


def _probability_columns(fieldnames: Sequence[str]) -> List[str]:
    prob_columns: List[Tuple[int, str]] = []
    for name in fieldnames:
        normalized = str(name or "").strip().lower()
        if not normalized.startswith("prob_"):
            continue
        parts = normalized.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        prob_columns.append((index, name))
    prob_columns.sort(key=lambda item: item[0])
    return [name for _, name in prob_columns]


def _float_value(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "") or "").strip())
    except ValueError:
        return float(default)


def _int_value(row: Dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(str(row.get(key, "") or "").strip()))
    except ValueError:
        return int(default)


def _path_looks_like_train(path_text: str) -> bool:
    normalized = str(path_text or "").replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if "val" in parts or "valid" in parts or "validation" in parts or "test" in parts:
        return False
    return "train" in parts


def _top2_from_probabilities(row: Dict[str, str], prob_columns: Sequence[str]) -> Tuple[int, int, float]:
    probabilities = [_float_value(row, column, 0.0) for column in prob_columns]
    if len(probabilities) < 2:
        return -1, -1, 1.0
    order = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    top1, top2 = int(order[0]), int(order[1])
    margin = float(probabilities[top1] - probabilities[top2])
    return top1, top2, margin


def _candidate_from_row(
    row: Dict[str, str],
    *,
    prob_columns: Sequence[str],
    pairs: set[Tuple[int, int]],
    margin_threshold: float,
    include_errors: bool,
    include_low_margin: bool,
) -> Optional[Tuple[int, int, str, float]]:
    target = _int_value(row, "target_index", _int_value(row, "y_true", -1))
    prediction = _int_value(row, "prediction_index", _int_value(row, "y_pred", -1))
    top1 = _int_value(row, "top1_index", prediction)
    top2 = _int_value(row, "top2_index", -1)
    margin = _float_value(row, "top2_margin", float("nan"))
    if top2 < 0 or not math.isfinite(margin):
        inferred_top1, inferred_top2, inferred_margin = _top2_from_probabilities(row, prob_columns)
        if top1 < 0:
            top1 = inferred_top1
        if top2 < 0:
            top2 = inferred_top2
        if not math.isfinite(margin):
            margin = inferred_margin
    if target < 0:
        return None

    if include_errors and prediction >= 0 and prediction != target:
        pair = tuple(sorted((target, prediction)))
        if pair in pairs:
            return target, prediction, "error_pair", margin

    if include_low_margin and top2 >= 0 and margin <= float(margin_threshold):
        other = top2 if top1 == target else top1
        if other >= 0 and other != target:
            pair = tuple(sorted((target, other)))
            if pair in pairs:
                return target, other, "low_margin_pair", margin
    return None


def build_manifest(
    *,
    predictions: Path,
    output: Path,
    pairs: str,
    alpha: float,
    margin_threshold: float,
    max_samples: int,
    require_train_paths: bool,
    include_errors: bool,
    include_low_margin: bool,
) -> Dict[str, object]:
    pair_set = _parse_pairs(pairs)
    alpha = max(0.0, min(1.0, float(alpha)))
    margin_threshold = max(0.0, float(margin_threshold))
    selected_rows: List[Dict[str, object]] = []
    by_reason: Counter[str] = Counter()
    by_pair: Counter[str] = Counter()
    skipped_non_train = 0
    skipped_invalid = 0

    with Path(predictions).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {predictions}")
        prob_columns = _probability_columns(reader.fieldnames)
        if not prob_columns:
            raise ValueError("Prediction CSV must contain prob_0..prob_N columns.")
        num_classes = len(prob_columns)
        for row in reader:
            image_path = (
                str(row.get("image_path", "") or "").strip()
                or str(row.get("path", "") or "").strip()
                or str(row.get("sample_path", "") or "").strip()
            )
            if not image_path:
                skipped_invalid += 1
                continue
            if require_train_paths and not _path_looks_like_train(image_path):
                skipped_non_train += 1
                continue
            candidate = _candidate_from_row(
                row,
                prob_columns=prob_columns,
                pairs=pair_set,
                margin_threshold=margin_threshold,
                include_errors=include_errors,
                include_low_margin=include_low_margin,
            )
            if candidate is None:
                continue
            target, soft_index, reason, margin = candidate
            if not (0 <= target < num_classes) or not (0 <= soft_index < num_classes):
                skipped_invalid += 1
                continue
            probabilities = [0.0 for _ in range(num_classes)]
            probabilities[target] = 1.0
            if soft_index != target:
                probabilities[target] = 1.0 - alpha
                probabilities[soft_index] = alpha
            pair_key = f"{target}->{soft_index}"
            by_reason[reason] += 1
            by_pair[pair_key] += 1
            selected_rows.append(
                {
                    "image_path": image_path,
                    "target_index": target,
                    "soft_target_index": soft_index,
                    "alpha": alpha,
                    "reason": reason,
                    "top2_margin": margin,
                    "prediction_index": _int_value(row, "prediction_index", _int_value(row, "y_pred", -1)),
                    **{f"soft_{index}": probabilities[index] for index in range(num_classes)},
                }
            )

    selected_rows.sort(
        key=lambda item: (
            0 if item["reason"] == "error_pair" else 1,
            float(item["top2_margin"]) if math.isfinite(float(item["top2_margin"])) else 1.0,
        )
    )
    if max_samples > 0:
        selected_rows = selected_rows[: int(max_samples)]
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "target_index",
        "soft_target_index",
        "alpha",
        "reason",
        "top2_margin",
        "prediction_index",
    ]
    if selected_rows:
        soft_columns = sorted(
            [key for key in selected_rows[0].keys() if re.fullmatch(r"soft_\d+", str(key))],
            key=lambda name: int(str(name).split("_", 1)[1]),
        )
        fieldnames.extend(soft_columns)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)
    return {
        "predictions": str(Path(predictions).resolve()),
        "output": str(output.resolve()),
        "rows": int(len(selected_rows)),
        "pairs": sorted([f"{left}-{right}" for left, right in pair_set]),
        "alpha": float(alpha),
        "margin_threshold": float(margin_threshold),
        "max_samples": int(max_samples),
        "require_train_paths": bool(require_train_paths),
        "skipped_non_train": int(skipped_non_train),
        "skipped_invalid": int(skipped_invalid),
        "by_reason": dict(by_reason),
        "by_pair": dict(by_pair),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only ambiguous soft-target manifest.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="0-1,1-2,2-3")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--allow-non-train-paths", action="store_true", default=False)
    parser.add_argument("--no-errors", action="store_true", default=False)
    parser.add_argument("--no-low-margin", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        predictions=args.predictions,
        output=args.output,
        pairs=args.pairs,
        alpha=args.alpha,
        margin_threshold=args.margin_threshold,
        max_samples=args.max_samples,
        require_train_paths=not args.allow_non_train_paths,
        include_errors=not args.no_errors,
        include_low_margin=not args.no_low_margin,
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
