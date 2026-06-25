from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trkh.tools.audit_prediction_forensics import _image_stats


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")
KNOWN_SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return rows, fieldnames


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _path_value(row: Mapping[str, str]) -> str:
    for key in ("image_path", "path", "sample_path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _split_from_path(path_text: str) -> str:
    parts = [part.lower() for part in str(path_text or "").replace("\\", "/").split("/") if part]
    for part in parts:
        if part in KNOWN_SPLITS:
            return KNOWN_SPLITS[part]
    return "unknown"


def _safe_name(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return value.strip("._") or "unknown"


def _int_value(row: Mapping[str, str], keys: Sequence[str], default: int = -1) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            continue
    return int(default)


def _float_value(row: Mapping[str, str], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return float(default)


def _probability_columns(fieldnames: Sequence[str]) -> Tuple[Dict[int, str], List[str]]:
    indexed: Dict[int, Tuple[str, str]] = {}
    for field in fieldnames:
        match = PROBABILITY_COLUMN.match(str(field))
        if match is None:
            continue
        index = int(match.group(1))
        class_name = str(match.group(2) or index)
        indexed[index] = (str(field), class_name)
    if not indexed:
        raise ValueError("Prediction CSV must contain prob_<index> columns.")
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError(f"Probability indices must be contiguous from 0: {sorted(indexed)}")
    columns = {index: indexed[index][0] for index in expected}
    names = [indexed[index][1] for index in expected]
    return columns, names


def _probabilities(row: Mapping[str, str], columns: Mapping[int, str]) -> np.ndarray:
    values = np.asarray(
        [_float_value(row, (column,), default=0.0) for _, column in sorted(columns.items())],
        dtype=np.float64,
    )
    values = np.clip(values, 1e-12, None)
    return values / max(1e-12, float(values.sum()))


def _parse_pairs(text: str) -> List[Tuple[int, int | str, str]]:
    pairs: List[Tuple[int, int | str, str]] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid pair {raw!r}; expected A-B or A-rest.")
        left_text, right_text = [part.strip() for part in item.split("-", 1)]
        left = int(left_text)
        if right_text == "rest":
            pairs.append((left, "rest", f"{left}-rest"))
        else:
            right = int(right_text)
            pairs.append((left, right, f"{left}-{right}"))
    if not pairs:
        raise ValueError("--pairs must contain at least one pair.")
    return pairs


def _match_pair(
    target: int,
    prediction: int,
    top2: int,
    pairs: Sequence[Tuple[int, int | str, str]],
) -> str:
    candidates = {int(value) for value in (target, prediction, top2) if int(value) >= 0}
    for left, right, name in pairs:
        if right == "rest":
            if int(left) in candidates:
                return name
            continue
        pair_values = {int(left), int(right)}
        if target in pair_values and (prediction in pair_values or top2 in pair_values):
            return name
        if prediction in pair_values and top2 in pair_values:
            return name
    return ""


def _quality_buckets(row: Mapping[str, object]) -> List[str]:
    buckets: List[str] = []
    status = str(row.get("image_status", "") or "")
    if status in {"pending", "not_selected", "skipped", "skipped_by_limit"}:
        return buckets
    if status and status != "ok":
        buckets.append(f"image_{status}")
        return buckets
    highlight = float(row.get("highlight_ratio", 0.0) or 0.0)
    shadow = float(row.get("shadow_ratio", 0.0) or 0.0)
    brightness = float(row.get("brightness_mean", 0.0) or 0.0)
    contrast = float(row.get("contrast_proxy", 0.0) or 0.0)
    background = row.get("suspected_background_ratio", "")
    foreground = row.get("foreground_fraction", "")
    center = float(row.get("center_brightness_mean", 0.0) or 0.0)
    border = float(row.get("border_brightness_mean", 0.0) or 0.0)
    if brightness >= 0.74 or highlight >= 0.08:
        buckets.append("over_bright_or_glare")
    if brightness <= 0.24 or shadow >= 0.12:
        buckets.append("under_dark_or_shadow")
    if contrast <= 0.15:
        buckets.append("low_contrast")
    if abs(center - border) >= 0.18:
        buckets.append("center_border_lighting_gap")
    if background != "" and float(background or 0.0) >= 0.55:
        buckets.append("background_heavy")
    if foreground != "" and float(foreground or 0.0) <= 0.18:
        buckets.append("partial_or_small_foreground")
    return buckets


def _select_reason(
    *,
    target: int,
    prediction: int,
    confidence: float,
    margin: float,
    pair: str,
    focus_class_index: int,
    high_confidence_threshold: float,
    low_margin_threshold: float,
    quality_buckets: Sequence[str],
) -> Tuple[str, float]:
    correct = target == prediction
    focus = int(focus_class_index)
    if not correct and target == focus:
        return "focus_false_negative", 5.0 + confidence
    if not correct and prediction == focus:
        return "focus_false_positive", 4.8 + confidence
    if not correct and confidence >= float(high_confidence_threshold):
        return "high_confidence_error", 4.0 + confidence
    if not correct and margin <= float(low_margin_threshold):
        return "low_margin_error", 3.0 + (float(low_margin_threshold) - margin)
    if correct and pair and margin <= float(low_margin_threshold):
        return "low_margin_correct_boundary", 2.0 + (float(low_margin_threshold) - margin)
    if quality_buckets and pair:
        return "quality_boundary_sample", 1.0 + min(0.99, confidence)
    return "", 0.0


def _manual_fields() -> List[str]:
    return [
        "manual_label_status",
        "manual_expected_class",
        "quality_lighting",
        "quality_dirty_obstacle",
        "quality_partial_fruit",
        "quality_background_mask",
        "review_decision",
        "review_notes",
    ]


def _copy_review_images(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    max_per_reason: int,
) -> Dict[str, int]:
    copied: Dict[str, int] = {}
    per_reason: Counter[str] = Counter()
    for row in rows:
        reason = _safe_name(row.get("reason", "unknown"))
        if max_per_reason > 0 and per_reason[reason] >= int(max_per_reason):
            continue
        source = Path(str(row.get("image_path", "") or ""))
        if not source.is_file():
            continue
        split = _safe_name(row.get("split", "unknown"))
        pair = _safe_name(row.get("boundary_pair", "unknown"))
        target_dir = output_dir / "review_images" / split / pair / reason
        target_dir.mkdir(parents=True, exist_ok=True)
        review_id = str(row.get("review_id", "0000"))
        target = target_dir / (
            f"{review_id}_t{row.get('target_index')}_p{row.get('prediction_index')}_"
            f"m{float(row.get('top2_margin', 0.0) or 0.0):.3f}{source.suffix.lower()}"
        )
        shutil.copy2(source, target)
        per_reason[reason] += 1
        copied[reason] = copied.get(reason, 0) + 1
    return copied


def build_manifest(
    *,
    predictions: Path,
    output_dir: Path,
    pairs: str = "0-1,1-2,2-3,4-rest",
    split: str = "auto",
    allow_other_splits: bool = False,
    allow_test: bool = False,
    focus_class_index: int = 1,
    high_confidence_threshold: float = 0.55,
    low_margin_threshold: float = 0.06,
    image_stats_mode: str = "basic",
    foreground_margin: float = 0.08,
    max_image_stats: int = 0,
    quality_scan_limit: int = 0,
    max_total: int = 0,
    max_per_reason: int = 120,
    max_per_pair: int = 120,
    copy_images: bool = False,
    max_copied_per_reason: int = 48,
) -> Dict[str, object]:
    prediction_rows, fieldnames = _read_csv(Path(predictions))
    prob_columns, class_names = _probability_columns(fieldnames)
    parsed_pairs = _parse_pairs(pairs)
    requested_split = str(split or "auto").lower()
    if requested_split == "valid":
        requested_split = "val"
    if requested_split not in {"auto", "train", "val", "test"}:
        raise ValueError("--split must be auto/train/val/test.")

    source_splits = Counter(_split_from_path(_path_value(row)) for row in prediction_rows)
    concrete_splits = {name for name in source_splits if name != "unknown"}
    if requested_split == "auto":
        if len(concrete_splits) != 1:
            raise ValueError(
                "Cannot infer a single split from prediction paths. "
                f"Found {dict(source_splits)}; pass --split explicitly."
            )
        active_split = next(iter(concrete_splits))
    else:
        active_split = requested_split
    if active_split == "test" and not allow_test:
        raise ValueError("Test split is disabled for boundary review by default; pass --allow-test for final audit.")

    selected: List[Dict[str, object]] = []
    skipped_other_split = 0
    skipped_unknown_split = 0
    image_stats_rows = 0
    for row_index, row in enumerate(prediction_rows):
        image_path = _path_value(row)
        row_split = _split_from_path(image_path)
        if row_split == "unknown":
            skipped_unknown_split += 1
            if not allow_other_splits:
                raise ValueError(f"Cannot determine split for row {row_index}: {image_path}")
            continue
        if row_split != active_split:
            skipped_other_split += 1
            if not allow_other_splits:
                raise ValueError(
                    f"Prediction CSV mixes split {row_split!r} into requested split {active_split!r}: {image_path}"
                )
            continue

        probabilities = _probabilities(row, prob_columns)
        sorted_indices = np.argsort(probabilities)[::-1]
        inferred_top1 = int(sorted_indices[0])
        inferred_top2 = int(sorted_indices[1]) if len(sorted_indices) > 1 else -1
        target = _int_value(row, ("target_index", "y_true", "label", "label_index"), default=-1)
        prediction = _int_value(row, ("prediction_index", "y_pred", "pred", "pred_index"), default=inferred_top1)
        top2 = _int_value(row, ("top2_index",), default=inferred_top2)
        if target < 0 or prediction < 0:
            continue
        confidence = _float_value(
            row,
            ("confidence", "top1_probability"),
            default=float(probabilities[prediction]),
        )
        top2_probability = _float_value(row, ("top2_probability",), default=float(probabilities[top2]) if top2 >= 0 else 0.0)
        margin = _float_value(row, ("top2_margin",), default=float(confidence - top2_probability))
        target_probability = float(probabilities[target]) if 0 <= target < len(probabilities) else 0.0
        pair = _match_pair(target, prediction, top2, parsed_pairs)

        reason, severity = _select_reason(
            target=target,
            prediction=prediction,
            confidence=float(confidence),
            margin=float(margin),
            pair=pair,
            focus_class_index=focus_class_index,
            high_confidence_threshold=high_confidence_threshold,
            low_margin_threshold=low_margin_threshold,
            quality_buckets=(),
        )
        scan_quality = not reason and bool(pair) and int(quality_scan_limit) > 0 and row_index < int(quality_scan_limit)
        if reason:
            stats = {"image_status": "pending"}
        elif scan_quality and image_stats_mode != "none" and (max_image_stats <= 0 or image_stats_rows < int(max_image_stats)):
            stats = _image_stats(image_path, float(foreground_margin), image_stats_mode)
            image_stats_rows += 1
        elif scan_quality:
            stats = {"image_status": "skipped_by_limit" if image_stats_mode != "none" else "skipped"}
        else:
            stats = {"image_status": "not_selected"}
        quality = _quality_buckets(stats)
        if not reason and scan_quality:
            reason, severity = _select_reason(
                target=target,
                prediction=prediction,
                confidence=float(confidence),
                margin=float(margin),
                pair=pair,
                focus_class_index=focus_class_index,
                high_confidence_threshold=high_confidence_threshold,
                low_margin_threshold=low_margin_threshold,
                quality_buckets=quality,
            )
        if not reason:
            continue
        boundary_pair = pair or f"{target}-{prediction}"
        record: Dict[str, object] = {
            "review_id": "",
            "source_row_index": int(row_index),
            "split": row_split,
            "image_path": str(Path(image_path).resolve()),
            "target_index": int(target),
            "target_name": row.get("target_name", row.get("true_name", class_names[target] if target < len(class_names) else "")),
            "prediction_index": int(prediction),
            "prediction_name": row.get("prediction_name", class_names[prediction] if prediction < len(class_names) else ""),
            "top2_index": int(top2),
            "top2_name": row.get("top2_name", class_names[top2] if 0 <= top2 < len(class_names) else ""),
            "confidence": float(confidence),
            "target_probability": float(target_probability),
            "top2_probability": float(top2_probability),
            "top2_margin": float(margin),
            "correct": int(target == prediction),
            "boundary_pair": boundary_pair,
            "reason": reason,
            "severity": float(severity),
            "buckets": ";".join(([f"{target}->{prediction}"] if target != prediction else ["correct"]) + quality),
        }
        for class_index, class_name in enumerate(class_names):
            record[f"prob_{class_index}_{class_name}"] = float(probabilities[class_index])
        record.update(stats)
        for field in _manual_fields():
            record[field] = ""
        selected.append(record)

    selected.sort(
        key=lambda item: (
            -float(item.get("severity", 0.0) or 0.0),
            float(item.get("top2_margin", 1.0) or 1.0),
            str(item.get("image_path", "")),
        )
    )
    kept: List[Dict[str, object]] = []
    per_reason: Counter[str] = Counter()
    per_pair: Counter[str] = Counter()
    for row in selected:
        reason = str(row.get("reason", "unknown"))
        pair = str(row.get("boundary_pair", "unknown"))
        if max_per_reason > 0 and per_reason[reason] >= int(max_per_reason):
            continue
        if max_per_pair > 0 and per_pair[pair] >= int(max_per_pair):
            continue
        row = dict(row)
        row["review_id"] = f"{active_split}_{len(kept) + 1:05d}"
        kept.append(row)
        per_reason[reason] += 1
        per_pair[pair] += 1
        if max_total > 0 and len(kept) >= int(max_total):
            break

    for row in kept:
        if row.get("image_status") != "pending":
            continue
        if image_stats_mode == "none":
            stats = {"image_status": "skipped"}
        elif max_image_stats > 0 and image_stats_rows >= int(max_image_stats):
            stats = {"image_status": "skipped_by_limit"}
        else:
            stats = _image_stats(str(row.get("image_path", "") or ""), float(foreground_margin), image_stats_mode)
            image_stats_rows += 1
        row.update(stats)
        quality = _quality_buckets(stats)
        if quality:
            existing = [bucket for bucket in str(row.get("buckets", "")).split(";") if bucket]
            row["buckets"] = ";".join(existing + [bucket for bucket in quality if bucket not in existing])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if kept:
        field_order = [
            "review_id",
            "source_row_index",
            "split",
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
            "correct",
            "boundary_pair",
            "reason",
            "severity",
            "buckets",
        ]
        prob_fields = [f"prob_{index}_{name}" for index, name in enumerate(class_names)]
        stat_fields = [field for field in kept[0].keys() if field not in set(field_order + prob_fields + _manual_fields())]
        fieldnames_out = field_order + prob_fields + sorted(stat_fields) + _manual_fields()
    else:
        fieldnames_out = [
            "review_id",
            "source_row_index",
            "split",
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
            "correct",
            "boundary_pair",
            "reason",
            "severity",
            "buckets",
            *_manual_fields(),
        ]
    _write_csv(output_dir / "boundary_review_manifest.csv", kept, fieldnames_out)
    copied = _copy_review_images(kept, output_dir, max_per_reason=max_copied_per_reason) if copy_images else {}

    bucket_counter: Counter[str] = Counter()
    for row in kept:
        for bucket in str(row.get("buckets", "")).split(";"):
            if bucket:
                bucket_counter[bucket] += 1
    summary = {
        "predictions": str(Path(predictions).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": active_split,
        "source_splits": dict(source_splits),
        "input_rows": int(len(prediction_rows)),
        "selected_before_limits": int(len(selected)),
        "selected_rows": int(len(kept)),
        "skipped_other_split": int(skipped_other_split),
        "skipped_unknown_split": int(skipped_unknown_split),
        "image_stats_rows": int(image_stats_rows),
        "class_names": class_names,
        "pairs": [name for _, _, name in parsed_pairs],
        "focus_class_index": int(focus_class_index),
        "thresholds": {
            "high_confidence": float(high_confidence_threshold),
            "low_margin": float(low_margin_threshold),
            "foreground_margin": float(foreground_margin),
        },
        "image_stats_mode": str(image_stats_mode),
        "quality_scan_limit": int(quality_scan_limit),
        "by_reason": dict(Counter(str(row.get("reason", "")) for row in kept)),
        "by_boundary_pair": dict(Counter(str(row.get("boundary_pair", "")) for row in kept)),
        "by_bucket": dict(bucket_counter),
        "copied_images": copied,
        "manual_columns": _manual_fields(),
        "leakage_guard": (
            "The tool rejects mixed/nonmatching splits by default and rejects test unless --allow-test is explicit. "
            "Use train manifests for data cleaning decisions; use val manifests for diagnosis only."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Boundary Review Manifest",
                "",
                f"- Predictions: `{predictions}`",
                f"- Split: `{active_split}`",
                f"- Selected rows: `{len(kept)}`",
                "",
                "Use `boundary_review_manifest.csv` to label boundary cases manually:",
                "",
                "- `manual_label_status`: `correct`, `ambiguous`, `wrong`, `needs_crop`.",
                "- `quality_lighting`: note over-bright, under-dark, glare, shadow.",
                "- `quality_dirty_obstacle`: note dirt, obstacle, occlusion.",
                "- `quality_partial_fruit`: note partial fruit or object cut off.",
                "- `quality_background_mask`: note whether background/mask is acceptable.",
                "",
                "Do not use validation/test labels to tune train-time sample weights directly.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-guarded train/val boundary review manifests from detailed prediction CSVs."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="0-1,1-2,2-3,4-rest")
    parser.add_argument("--split", choices=("auto", "train", "val", "test"), default="auto")
    parser.add_argument("--allow-other-splits", action="store_true", default=False)
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--low-margin-threshold", type=float, default=0.06)
    parser.add_argument("--image-stats-mode", choices=("none", "basic", "foreground"), default="basic")
    parser.add_argument("--foreground-margin", type=float, default=0.08)
    parser.add_argument("--max-image-stats", type=int, default=0)
    parser.add_argument(
        "--quality-scan-limit",
        type=int,
        default=0,
        help=(
            "Optional number of CSV rows to scan for quality-only boundary samples. "
            "Default 0 avoids slow full-split foreground scans."
        ),
    )
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--max-per-reason", type=int, default=120)
    parser.add_argument("--max-per-pair", type=int, default=120)
    parser.add_argument("--copy-images", action="store_true", default=False)
    parser.add_argument("--max-copied-per-reason", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        predictions=args.predictions,
        output_dir=args.output_dir,
        pairs=args.pairs,
        split=args.split,
        allow_other_splits=args.allow_other_splits,
        allow_test=args.allow_test,
        focus_class_index=args.focus_class_index,
        high_confidence_threshold=args.high_confidence_threshold,
        low_margin_threshold=args.low_margin_threshold,
        image_stats_mode=args.image_stats_mode,
        foreground_margin=args.foreground_margin,
        max_image_stats=args.max_image_stats,
        quality_scan_limit=args.quality_scan_limit,
        max_total=args.max_total,
        max_per_reason=args.max_per_reason,
        max_per_pair=args.max_per_pair,
        copy_images=args.copy_images,
        max_copied_per_reason=args.max_copied_per_reason,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
