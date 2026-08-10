from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _path_value(row: Mapping[str, str]) -> str:
    return (
        str(row.get("image_path", "") or "").strip()
        or str(row.get("path", "") or "").strip()
        or str(row.get("sample_path", "") or "").strip()
    )


def _looks_like_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    if "val" in parts or "valid" in parts or "validation" in parts or "test" in parts:
        return False
    return "train" in parts


def _float_value(row: Mapping[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return parsed
    return float(default)


def _int_value(row: Mapping[str, str], *keys: str, default: int = -1) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return int(default)


def _probability_columns(fieldnames: Iterable[str]) -> Dict[int, str]:
    columns: Dict[int, str] = {}
    for name in fieldnames:
        normalized = str(name or "").strip()
        if not normalized.lower().startswith("prob_"):
            continue
        parts = normalized.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            class_index = int(parts[1])
        except ValueError:
            continue
        columns[class_index] = normalized
    return dict(sorted(columns.items(), key=lambda item: item[0]))


def _probability(row: Mapping[str, str], prob_columns: Mapping[int, str], class_index: int) -> float:
    column = prob_columns.get(int(class_index))
    if column is None:
        exact = f"prob_{int(class_index)}"
        if exact in row:
            column = exact
    if column is None:
        return 0.0
    return _float_value(row, column, default=0.0)


def _top2_from_probabilities(row: Mapping[str, str], prob_columns: Mapping[int, str]) -> Tuple[int, float, int, float]:
    probabilities = [
        (int(class_index), _probability(row, prob_columns, int(class_index)))
        for class_index in sorted(prob_columns.keys())
    ]
    if not probabilities:
        return -1, 0.0, -1, 0.0
    probabilities.sort(key=lambda item: item[1], reverse=True)
    top1_index, top1_probability = probabilities[0]
    if len(probabilities) == 1:
        return int(top1_index), float(top1_probability), -1, 0.0
    top2_index, top2_probability = probabilities[1]
    return int(top1_index), float(top1_probability), int(top2_index), float(top2_probability)


def _parse_boundary_pairs(text: str) -> List[Tuple[int, int | str]]:
    pairs: List[Tuple[int, int | str]] = []
    for raw_item in str(text or "").replace(";", ",").split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid boundary pair {raw_item!r}; expected A-B or A-rest.")
        left_text, right_text = [part.strip() for part in item.split("-", 1)]
        left = int(left_text)
        if right_text == "rest":
            pairs.append((left, "rest"))
        else:
            pairs.append((left, int(right_text)))
    if not pairs:
        raise ValueError("--boundary-pairs must contain at least one pair.")
    return pairs


def _pair_name(a: int, b: int, pairs: Sequence[Tuple[int, int | str]]) -> str:
    for left, right in pairs:
        if right == "rest":
            if int(a) == int(left) or int(b) == int(left):
                return f"{left}-rest"
            continue
        if {int(a), int(b)} == {int(left), int(right)}:
            return f"{left}-{right}"
    return ""


def _load_base_weights(path: Optional[Path]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    if path is None or not str(path).strip():
        return {}, {"enabled": False}
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing base sample-weight manifest: {manifest_path}")
    rows = _read_csv(manifest_path)
    weights: Dict[str, Dict[str, object]] = {}
    duplicate_rows = 0
    invalid_rows = 0
    for row in rows:
        image_path = _path_value(row)
        if not image_path:
            invalid_rows += 1
            continue
        weight = _float_value(row, "sample_weight", "weight", default=1.0)
        if not math.isfinite(weight) or weight <= 0.0:
            invalid_rows += 1
            continue
        resolved = str(Path(image_path).resolve())
        key = resolved.lower()
        if key in weights:
            duplicate_rows += 1
            if float(weights[key]["sample_weight"]) >= weight:
                continue
        weights[key] = {
            "image_path": resolved,
            "target_index": row.get("target_index", ""),
            "target_name": row.get("target_name", ""),
            "prediction_index": row.get("prediction_index", ""),
            "prediction_name": row.get("prediction_name", ""),
            "top2_index": row.get("top2_index", ""),
            "top2_name": row.get("top2_name", ""),
            "confidence": row.get("confidence", ""),
            "target_probability": row.get("target_probability", ""),
            "top2_probability": row.get("top2_probability", ""),
            "top2_margin": row.get("top2_margin", ""),
            "boundary_pair": row.get("boundary_pair", ""),
            "base_sample_weight": float(weight),
            "issue_multiplier": 1.0,
            "sample_weight": float(weight),
            "reason": str(row.get("reason", "") or "base_weight"),
            "severity": 0.0,
        }
    summary = {
        "enabled": True,
        "manifest": str(manifest_path.resolve()),
        "rows": int(len(rows)),
        "valid_paths": int(len(weights)),
        "duplicate_rows": int(duplicate_rows),
        "invalid_rows": int(invalid_rows),
    }
    return weights, summary


def _reason_multiplier(reason: str, *, high_confidence: float, low_self: float, ambiguous: float) -> float:
    if reason == "high_confidence_disagreement":
        return float(high_confidence)
    if reason == "low_self_confidence_error":
        return float(low_self)
    return float(ambiguous)


def _issue_candidate(
    row: Mapping[str, str],
    *,
    prob_columns: Mapping[int, str],
    boundary_pairs: Sequence[Tuple[int, int | str]],
    high_confidence_threshold: float,
    low_self_confidence_threshold: float,
    ambiguous_margin_threshold: float,
) -> Optional[Dict[str, object]]:
    image_path = _path_value(row)
    if not image_path:
        return None
    target = _int_value(row, "target_index", "true_index", "y_true", "label", "label_index")
    prediction = _int_value(
        row,
        "prediction_index",
        "suggested_index",
        "pred_index",
        "y_pred",
        "pred",
    )
    if target < 0:
        return None
    inferred_top1, inferred_top1_probability, inferred_top2, inferred_top2_probability = _top2_from_probabilities(
        row, prob_columns
    )
    top1 = _int_value(row, "top1_index", default=inferred_top1)
    top2 = _int_value(row, "top2_index", default=inferred_top2)
    if prediction < 0:
        prediction = top1
    top1_probability = _float_value(
        row,
        "top1_probability",
        "top1_confidence",
        "confidence",
        default=inferred_top1_probability,
    )
    top2_probability = _float_value(row, "top2_probability", default=inferred_top2_probability)
    margin = _float_value(row, "top2_margin", default=top1_probability - top2_probability)
    target_probability = _float_value(row, "self_confidence", default=_probability(row, prob_columns, target))
    if target_probability <= 0.0 and prediction == target:
        target_probability = top1_probability

    prediction_pair = _pair_name(target, prediction, boundary_pairs) if prediction >= 0 else ""
    top2_pair = _pair_name(target, top2, boundary_pairs) if top2 >= 0 else ""
    pair = prediction_pair or top2_pair
    if not pair:
        return None

    correct = int(prediction == target)
    reason = ""
    severity = 0.0
    if not correct and top1_probability >= float(high_confidence_threshold):
        reason = "high_confidence_disagreement"
        severity = 3.0 + float(top1_probability)
    elif not correct and target_probability <= float(low_self_confidence_threshold):
        reason = "low_self_confidence_error"
        severity = 2.0 + (float(low_self_confidence_threshold) - float(target_probability))
    elif margin <= float(ambiguous_margin_threshold):
        reason = "low_margin_boundary"
        severity = 1.0 + (float(ambiguous_margin_threshold) - float(margin))
    if not reason:
        return None

    return {
        "image_path": str(Path(image_path).resolve()),
        "target_index": int(target),
        "target_name": str(row.get("target_name", ""))
        or str(row.get("true_name", ""))
        or str(row.get("label_name", "")),
        "prediction_index": int(prediction),
        "prediction_name": str(row.get("prediction_name", ""))
        or str(row.get("suggested_name", ""))
        or str(row.get("pred_name", ""))
        or str(row.get("teacher_pred_name", "")),
        "top2_index": int(top2),
        "top2_name": str(row.get("top2_name", "")),
        "confidence": float(top1_probability),
        "target_probability": float(target_probability),
        "top2_probability": float(top2_probability),
        "top2_margin": float(margin),
        "boundary_pair": pair,
        "reason": reason,
        "severity": float(severity),
    }


def _copy_review_images(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    max_images_per_reason: int,
) -> Dict[str, int]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reason", "") or "unknown")].append(row)
    copied: Dict[str, int] = {}
    for reason, reason_rows in grouped.items():
        target_dir = output_dir / "review_images" / reason
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for index, row in enumerate(reason_rows, start=1):
            if count >= int(max_images_per_reason):
                break
            source = Path(str(row.get("image_path", "") or ""))
            if not source.is_file():
                continue
            target = target_dir / (
                f"{index:03d}_t{row.get('target_index')}_p{row.get('prediction_index')}_"
                f"w{float(row.get('sample_weight', 1.0)):.2f}_"
                f"m{float(row.get('top2_margin', 0.0)):.3f}{source.suffix.lower()}"
            )
            shutil.copy2(source, target)
            count += 1
        copied[reason] = count
    return copied


def build_manifest(
    *,
    predictions: Path,
    output_dir: Path,
    base_sample_weight_manifest: Optional[Path],
    boundary_pairs: str,
    high_confidence_threshold: float,
    low_self_confidence_threshold: float,
    ambiguous_margin_threshold: float,
    high_confidence_multiplier: float,
    low_self_confidence_multiplier: float,
    ambiguous_multiplier: float,
    min_weight: float,
    max_weight: float,
    max_issues: int,
    max_per_reason: int,
    max_per_pair: int,
    allow_non_train_paths: bool,
    copy_images: bool,
    max_review_images_per_reason: int,
    dry_run: bool,
) -> Dict[str, object]:
    predictions_path = Path(predictions)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_path}")
    rows = _read_csv(predictions_path)
    prob_columns = _probability_columns(rows[0].keys())
    if not prob_columns:
        raise ValueError("Prediction CSV must contain prob_0..prob_N columns.")

    base_records, base_summary = _load_base_weights(base_sample_weight_manifest)
    selected_by_path: Dict[str, Dict[str, object]] = {}
    boundary_pair_specs = _parse_boundary_pairs(boundary_pairs)
    skipped_non_train = 0
    skipped_invalid = 0
    selected_by_reason: Counter[str] = Counter()
    selected_by_pair: Counter[str] = Counter()

    candidates: List[Dict[str, object]] = []
    for row in rows:
        image_path = _path_value(row)
        if not image_path:
            skipped_invalid += 1
            continue
        if not allow_non_train_paths and not _looks_like_train(image_path):
            skipped_non_train += 1
            continue
        candidate = _issue_candidate(
            row,
            prob_columns=prob_columns,
            boundary_pairs=boundary_pair_specs,
            high_confidence_threshold=high_confidence_threshold,
            low_self_confidence_threshold=low_self_confidence_threshold,
            ambiguous_margin_threshold=ambiguous_margin_threshold,
        )
        if candidate is None:
            continue
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -float(item["severity"]),
            float(item["top2_margin"]),
            str(item["image_path"]),
        )
    )
    per_reason_kept: Counter[str] = Counter()
    per_pair_kept: Counter[str] = Counter()
    for candidate in candidates:
        reason = str(candidate["reason"])
        pair = str(candidate["boundary_pair"])
        if max_per_reason > 0 and per_reason_kept[reason] >= int(max_per_reason):
            continue
        if max_per_pair > 0 and per_pair_kept[pair] >= int(max_per_pair):
            continue
        key = str(Path(str(candidate["image_path"])).resolve()).lower()
        if key in selected_by_path:
            continue
        base = float(base_records.get(key, {}).get("sample_weight", 1.0))
        multiplier = _reason_multiplier(
            reason,
            high_confidence=high_confidence_multiplier,
            low_self=low_self_confidence_multiplier,
            ambiguous=ambiguous_multiplier,
        )
        weight = max(float(min_weight), min(float(max_weight), base * float(multiplier)))
        record = dict(candidate)
        record.update(
            {
                "base_sample_weight": float(base),
                "issue_multiplier": float(multiplier),
                "sample_weight": float(weight),
            }
        )
        selected_by_path[key] = record
        per_reason_kept[reason] += 1
        per_pair_kept[pair] += 1
        selected_by_reason[reason] += 1
        selected_by_pair[pair] += 1
        if max_issues > 0 and len(selected_by_path) >= int(max_issues):
            break

    merged = dict(base_records)
    for key, issue_record in selected_by_path.items():
        merged[key] = issue_record

    manifest_rows = list(merged.values())
    manifest_rows.sort(
        key=lambda item: (
            -float(item.get("severity", 0.0) or 0.0),
            float(item.get("sample_weight", 1.0) or 1.0),
            str(item.get("image_path", "")),
        )
    )
    review_rows = list(selected_by_path.values())
    review_rows.sort(
        key=lambda item: (
            -float(item.get("severity", 0.0) or 0.0),
            float(item.get("top2_margin", 1.0) or 1.0),
            str(item.get("image_path", "")),
        )
    )

    output_dir = Path(output_dir)
    manifest_path = output_dir / "sample_weights_train_only.csv"
    review_path = output_dir / "label_issue_review_train_only.csv"
    fieldnames = [
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "top2_index",
        "top2_name",
        "confidence",
        "target_probability",
        "top2_probability",
        "top2_margin",
        "boundary_pair",
        "base_sample_weight",
        "issue_multiplier",
        "sample_weight",
        "reason",
        "severity",
    ]
    copied: Dict[str, int] = {}
    if not dry_run:
        _write_csv(manifest_path, manifest_rows, fieldnames)
        _write_csv(review_path, review_rows, fieldnames)
        if copy_images:
            copied = _copy_review_images(
                review_rows,
                output_dir,
                max_images_per_reason=max_review_images_per_reason,
            )

    weights = [float(row.get("sample_weight", 1.0) or 1.0) for row in manifest_rows]
    summary = {
        "predictions": str(predictions_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "review_csv": str(review_path.resolve()),
        "dry_run": bool(dry_run),
        "base_manifest": base_summary,
        "prediction_rows": int(len(rows)),
        "probability_classes": int(len(prob_columns)),
        "skipped_non_train": int(skipped_non_train),
        "skipped_invalid": int(skipped_invalid),
        "candidate_issues": int(len(candidates)),
        "selected_issues": int(len(review_rows)),
        "merged_manifest_rows": int(len(manifest_rows)),
        "by_reason": dict(selected_by_reason),
        "by_boundary_pair": dict(selected_by_pair),
        "boundary_pairs": [
            f"{left}-{right}" for left, right in boundary_pair_specs
        ],
        "thresholds": {
            "high_confidence": float(high_confidence_threshold),
            "low_self_confidence": float(low_self_confidence_threshold),
            "ambiguous_margin": float(ambiguous_margin_threshold),
        },
        "multipliers": {
            "high_confidence_disagreement": float(high_confidence_multiplier),
            "low_self_confidence_error": float(low_self_confidence_multiplier),
            "low_margin_boundary": float(ambiguous_multiplier),
        },
        "weight_range": {
            "min": float(min(weights)) if weights else 0.0,
            "max": float(max(weights)) if weights else 0.0,
            "mean": float(sum(weights) / max(1, len(weights))) if weights else 0.0,
        },
        "limits": {
            "max_issues": int(max_issues),
            "max_per_reason": int(max_per_reason),
            "max_per_pair": int(max_per_pair),
        },
        "copied_images": copied,
        "leakage_guard": "requires train paths unless --allow-non-train-paths is set",
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only data-centric sample-weight manifest. "
            "The tool downweights likely label-noise/ambiguous boundary samples "
            "and can merge them with an existing hard-boundary upweight manifest."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-sample-weight-manifest", type=Path, default=None)
    parser.add_argument("--boundary-pairs", type=str, default="0-1,1-2,2-3,4-rest")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--low-self-confidence-threshold", type=float, default=0.25)
    parser.add_argument("--ambiguous-margin-threshold", type=float, default=0.06)
    parser.add_argument("--high-confidence-multiplier", type=float, default=0.30)
    parser.add_argument("--low-self-confidence-multiplier", type=float, default=0.45)
    parser.add_argument("--ambiguous-multiplier", type=float, default=0.85)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=2.5)
    parser.add_argument("--max-issues", type=int, default=420)
    parser.add_argument("--max-per-reason", type=int, default=220)
    parser.add_argument("--max-per-pair", type=int, default=180)
    parser.add_argument("--allow-non-train-paths", action="store_true", default=False)
    parser.add_argument("--copy-images", action="store_true", default=False)
    parser.add_argument("--max-review-images-per-reason", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        predictions=args.predictions,
        output_dir=args.output_dir,
        base_sample_weight_manifest=args.base_sample_weight_manifest,
        boundary_pairs=args.boundary_pairs,
        high_confidence_threshold=args.high_confidence_threshold,
        low_self_confidence_threshold=args.low_self_confidence_threshold,
        ambiguous_margin_threshold=args.ambiguous_margin_threshold,
        high_confidence_multiplier=args.high_confidence_multiplier,
        low_self_confidence_multiplier=args.low_self_confidence_multiplier,
        ambiguous_multiplier=args.ambiguous_multiplier,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        max_issues=args.max_issues,
        max_per_reason=args.max_per_reason,
        max_per_pair=args.max_per_pair,
        allow_non_train_paths=args.allow_non_train_paths,
        copy_images=args.copy_images,
        max_review_images_per_reason=args.max_review_images_per_reason,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
