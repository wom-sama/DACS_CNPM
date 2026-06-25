from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from trkh.data.dataset import _pseudo_foreground_mask_array


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe prediction forensics from a prediction CSV. "
            "Use train/val for model development; test is final audit only."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        type=str,
        default="0-1,1-2,2-3,3-4,4-rest",
        help="Comma list of pair metrics. Use 4-rest for one-vs-rest.",
    )
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--foreground-margin", type=float, default=0.08)
    parser.add_argument(
        "--image-stats-mode",
        choices=("none", "basic", "foreground"),
        default="basic",
        help=(
            "none skips image reads; basic reads lighting/color stats; foreground also computes "
            "pseudo foreground/background ratios and is slower."
        ),
    )
    parser.add_argument(
        "--max-image-stats",
        type=int,
        default=0,
        help="Optional cap for image-stat rows. 0 means all rows.",
    )
    parser.add_argument("--top-k-images", type=int, default=25)
    parser.add_argument("--copy-images", action="store_true", default=False)
    return parser.parse_args()


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return rows, fieldnames


def _path(row: Mapping[str, str]) -> str:
    for key in ("image_path", "path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _int_value(row: Mapping[str, str], keys: Sequence[str], default: int = 0) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(float(value))
    return int(default)


def _float_value(row: Mapping[str, str], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return float(default)


def _probability_columns(fieldnames: Sequence[str]) -> Tuple[List[str], List[str]]:
    indexed: Dict[int, Tuple[str, str]] = {}
    for field in fieldnames:
        match = PROBABILITY_COLUMN.match(str(field))
        if match is None:
            continue
        index = int(match.group(1))
        name = str(match.group(2) or index)
        indexed[index] = (str(field), name)
    if not indexed:
        raise ValueError("Prediction CSV does not contain prob_<index> columns.")
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise ValueError(f"Probability indices must be contiguous from 0: {sorted(indexed)}")
    return [indexed[index][0] for index in expected], [indexed[index][1] for index in expected]


def _parse_pairs(value: str, class_count: int) -> List[Tuple[int, int | str, str]]:
    pairs: List[Tuple[int, int | str, str]] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Pair must use A-B syntax, got: {item}")
        left_text, right_text = item.split("-", 1)
        left = int(left_text)
        right: int | str = "rest" if right_text.lower() == "rest" else int(right_text)
        if left < 0 or left >= class_count:
            raise ValueError(f"Pair class out of range: {item}")
        if right != "rest" and (int(right) < 0 or int(right) >= class_count):
            raise ValueError(f"Pair class out of range: {item}")
        pairs.append((left, right, item))
    return pairs


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    bins: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    eps = 1e-12
    confidences = probabilities.max(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], eps, 1.0)).mean()
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    rows: List[Dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    bin_count = max(1, int(bins))
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        if index + 1 == bin_count:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count:
            accuracy = float(correctness[mask].mean())
            confidence = float(confidences[mask].mean())
            gap = abs(accuracy - confidence)
            ece += gap * count / max(1, len(targets))
            mce = max(mce, gap)
        else:
            accuracy = confidence = gap = 0.0
        rows.append(
            {
                "bin": int(index),
                "low": float(low),
                "high": float(high),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": float(gap),
            }
        )
    return {"ece": float(ece), "mce": float(mce), "brier": float(brier), "nll": float(nll)}, rows


def _f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return float(precision), float(recall), float(f1)


def _pair_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    pairs: Sequence[Tuple[int, int | str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for left, right, name in pairs:
        if right == "rest":
            target_positive = targets == left
            pred_positive = predictions == left
            support = int(target_positive.sum())
            tp = int(np.logical_and(target_positive, pred_positive).sum())
            fp = int(np.logical_and(~target_positive, pred_positive).sum())
            fn = int(np.logical_and(target_positive, ~pred_positive).sum())
            precision, recall, f1 = _f1_from_counts(tp, fp, fn)
            rows.append(
                {
                    "pair": name,
                    "mode": "one_vs_rest",
                    "support": support,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "a_to_b": "",
                    "b_to_a": "",
                    "outside_predictions": int(np.logical_and(target_positive, ~pred_positive).sum()),
                }
            )
            continue
        right_index = int(right)
        pair_target = np.logical_or(targets == left, targets == right_index)
        support = int(pair_target.sum())
        pair_prediction = np.logical_or(predictions == left, predictions == right_index)
        correct = int(np.logical_and(pair_target, targets == predictions).sum())
        a_to_b = int(np.logical_and(targets == left, predictions == right_index).sum())
        b_to_a = int(np.logical_and(targets == right_index, predictions == left).sum())
        outside = int(np.logical_and(pair_target, ~pair_prediction).sum())
        rows.append(
            {
                "pair": name,
                "mode": "pair_subset",
                "support": support,
                "tp": correct,
                "fp": "",
                "fn": "",
                "precision": "",
                "recall": "",
                "f1": float(correct / max(1, support)),
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "outside_predictions": outside,
            }
        )
    return rows


def _image_stats(path: str, foreground_margin: float, mode: str) -> Dict[str, float | str]:
    if mode == "none":
        return {"image_status": "skipped"}
    image_path = Path(path)
    if not image_path.is_file():
        return {"image_status": "missing"}
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pragma: no cover - corrupt files are data issues
        return {"image_status": f"error:{type(exc).__name__}"}
    array = np.asarray(image, dtype=np.float32) / 255.0
    luminance = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    max_rgb = array.max(axis=2)
    min_rgb = array.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
    if mode == "foreground":
        try:
            foreground = _pseudo_foreground_mask_array(image, margin=float(foreground_margin))
            foreground_fraction = float(np.asarray(foreground, dtype=bool).mean())
        except Exception:
            foreground_fraction = 0.0
    else:
        foreground_fraction = 0.0
    h, w = luminance.shape
    border_width = max(1, int(round(min(h, w) * 0.08)))
    border_mask = np.zeros_like(luminance, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    center_mask = ~border_mask
    return {
        "image_status": "ok",
        "width": int(w),
        "height": int(h),
        "brightness_mean": float(luminance.mean()),
        "brightness_std": float(luminance.std()),
        "contrast_proxy": float(np.quantile(luminance, 0.90) - np.quantile(luminance, 0.10)),
        "highlight_ratio": float((luminance > 0.88).mean()),
        "shadow_ratio": float((luminance < 0.12).mean()),
        "saturation_mean": float(saturation.mean()),
        "foreground_fraction": foreground_fraction if mode == "foreground" else "",
        "suspected_background_ratio": float(1.0 - foreground_fraction) if mode == "foreground" else "",
        "border_brightness_mean": float(luminance[border_mask].mean()),
        "center_brightness_mean": float(luminance[center_mask].mean()) if center_mask.any() else float(luminance.mean()),
    }


def _bucket_for(row: Mapping[str, object]) -> List[str]:
    buckets: List[str] = []
    if int(row["target_index"]) != int(row["prediction_index"]):
        buckets.append(f"{row['target_index']}->{row['prediction_index']}")
        buckets.append("error")
    else:
        buckets.append("correct")
    if float(row.get("top2_margin", 0.0) or 0.0) < 0.05:
        buckets.append("low_margin")
    if float(row.get("confidence", 0.0) or 0.0) > 0.60 and int(row["target_index"]) != int(row["prediction_index"]):
        buckets.append("high_confidence_error")
    background_ratio = row.get("suspected_background_ratio", "")
    if background_ratio != "" and float(background_ratio or 0.0) > 0.55:
        buckets.append("background_heavy")
    if float(row.get("highlight_ratio", 0.0) or 0.0) > 0.08:
        buckets.append("highlight_heavy")
    if float(row.get("shadow_ratio", 0.0) or 0.0) > 0.12:
        buckets.append("shadow_heavy")
    if abs(
        float(row.get("center_brightness_mean", 0.0) or 0.0)
        - float(row.get("border_brightness_mean", 0.0) or 0.0)
    ) > 0.18:
        buckets.append("center_border_lighting_gap")
    return buckets


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _copy_ranked_images(rows: Iterable[Mapping[str, object]], output_dir: Path, limit: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for index, row in enumerate(rows, start=1):
        if copied >= int(limit):
            break
        source = Path(str(row.get("image_path", "") or ""))
        if not source.is_file():
            continue
        target = output_dir / (
            f"{index:03d}_t{row.get('target_index')}_p{row.get('prediction_index')}_"
            f"conf{float(row.get('confidence', 0.0)):.3f}_m{float(row.get('top2_margin', 0.0)):.3f}"
            f"{source.suffix.lower()}"
        )
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    rows, fieldnames = _read_csv(args.predictions)
    probability_columns, class_names = _probability_columns(fieldnames)
    class_count = len(class_names)
    targets = np.asarray(
        [_int_value(row, ("target_index", "y_true")) for row in rows],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [[float(row[column]) for column in probability_columns] for row in rows],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities = probabilities / np.maximum(1e-12, probabilities.sum(axis=1, keepdims=True))
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    top2_margins = sorted_probs[:, -1] - sorted_probs[:, -2] if class_count >= 2 else confidences

    pairs = _parse_pairs(args.pairs, class_count)
    calibration, calibration_rows = _calibration_metrics(
        probabilities,
        targets,
        predictions,
        bins=int(args.ece_bins),
    )
    pair_rows = _pair_metrics(targets, predictions, pairs)

    enriched: List[Dict[str, object]] = []
    max_image_stats = int(args.max_image_stats)
    image_stats_mode = str(args.image_stats_mode)
    for index, source_row in enumerate(rows):
        image_path = _path(source_row)
        target = int(targets[index])
        prediction = int(predictions[index])
        enriched_row: Dict[str, object] = {
            "sample_index": _int_value(source_row, ("sample_index",), index),
            "image_path": image_path,
            "target_index": target,
            "target_name": source_row.get("target_name", source_row.get("true_name", class_names[target])),
            "prediction_index": prediction,
            "prediction_name": class_names[prediction],
            "correct": int(target == prediction),
            "confidence": float(confidences[index]),
            "top2_margin": float(top2_margins[index]),
            "target_probability": float(probabilities[index, target]),
            **{
                f"prob_{class_index}_{class_name}": float(probabilities[index, class_index])
                for class_index, class_name in enumerate(class_names)
            },
        }
        if max_image_stats > 0 and index >= max_image_stats:
            enriched_row.update({"image_status": "skipped_by_limit"})
        else:
            enriched_row.update(_image_stats(image_path, float(args.foreground_margin), image_stats_mode))
        if (index + 1) % 500 == 0:
            print(
                f"processed {index + 1}/{len(rows)} rows forensics",
                file=sys.stderr,
                flush=True,
            )
        buckets = _bucket_for(enriched_row)
        enriched_row["buckets"] = ";".join(buckets)
        enriched.append(enriched_row)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forensics_fields = list(enriched[0].keys()) if enriched else []
    _write_csv(output_dir / "predictions_forensics.csv", enriched, forensics_fields)
    _write_csv(output_dir / "calibration_bins.csv", calibration_rows, list(calibration_rows[0].keys()))
    _write_csv(output_dir / "pair_metrics.csv", pair_rows, list(pair_rows[0].keys()))

    bucket_counter: Counter[str] = Counter()
    error_bucket_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    for row in enriched:
        for bucket in str(row.get("buckets", "")).split(";"):
            if not bucket:
                continue
            bucket_counter[bucket] += 1
            if not int(row["correct"]):
                error_bucket_counter[bucket] += 1
        if not int(row["correct"]):
            pair_counter[f"{row['target_index']}->{row['prediction_index']}"] += 1
    bucket_rows = [
        {
            "bucket": bucket,
            "samples": count,
            "errors": int(error_bucket_counter.get(bucket, 0)),
            "error_rate": float(error_bucket_counter.get(bucket, 0) / max(1, count)),
        }
        for bucket, count in sorted(bucket_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write_csv(output_dir / "bucket_summary.csv", bucket_rows, ["bucket", "samples", "errors", "error_rate"])

    focus = int(args.focus_class_index)
    focus_tp = int(np.logical_and(targets == focus, predictions == focus).sum())
    focus_fp = int(np.logical_and(targets != focus, predictions == focus).sum())
    focus_fn = int(np.logical_and(targets == focus, predictions != focus).sum())
    focus_precision, focus_recall, focus_f1 = _f1_from_counts(focus_tp, focus_fp, focus_fn)
    correct_margins = [float(m) for m, ok in zip(top2_margins, predictions == targets) if bool(ok)]
    error_margins = [float(m) for m, ok in zip(top2_margins, predictions == targets) if not bool(ok)]
    copied = {}
    if args.copy_images:
        error_rows = [row for row in enriched if not int(row["correct"])]
        copied["high_confidence_errors"] = _copy_ranked_images(
            sorted(error_rows, key=lambda row: (-float(row["confidence"]), float(row["top2_margin"]))),
            output_dir / "review_images" / "high_confidence_errors",
            int(args.top_k_images),
        )
        copied["low_margin_errors"] = _copy_ranked_images(
            sorted(error_rows, key=lambda row: float(row["top2_margin"])),
            output_dir / "review_images" / "low_margin_errors",
            int(args.top_k_images),
        )
        copied["background_heavy_errors"] = _copy_ranked_images(
            [row for row in error_rows if "background_heavy" in str(row.get("buckets", ""))],
            output_dir / "review_images" / "background_heavy_errors",
            int(args.top_k_images),
        )
        copied["highlight_shadow_errors"] = _copy_ranked_images(
            [
                row
                for row in error_rows
                if "highlight_heavy" in str(row.get("buckets", ""))
                or "shadow_heavy" in str(row.get("buckets", ""))
            ],
            output_dir / "review_images" / "highlight_shadow_errors",
            int(args.top_k_images),
        )

    summary = {
        "predictions": str(args.predictions),
        "samples": len(rows),
        "class_names": class_names,
        "accuracy": float((predictions == targets).mean()),
        "focus_class": {
            "index": focus,
            "name": class_names[focus] if 0 <= focus < len(class_names) else "",
            "tp": focus_tp,
            "fp": focus_fp,
            "fn": focus_fn,
            "precision": focus_precision,
            "recall": focus_recall,
            "f1": focus_f1,
        },
        "calibration": calibration,
        "margin": {
            "all": _quantiles([float(value) for value in top2_margins]),
            "correct": _quantiles(correct_margins),
            "error": _quantiles(error_margins),
        },
        "top_error_pairs": [
            {"pair": pair, "count": int(count)}
            for pair, count in pair_counter.most_common(20)
        ],
        "top_buckets": bucket_rows[:20],
        "pair_metrics": pair_rows,
        "copied_images": copied,
        "leakage_note": (
            "This tool only summarizes a supplied prediction CSV. Use train/val reports "
            "for development; use test reports only after the model and thresholds are frozen."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Prediction Forensics",
                "",
                f"Predictions: `{args.predictions}`",
                "",
                "Artifacts:",
                "",
                "- `predictions_forensics.csv`: per-sample probabilities, margins, lighting/background features and buckets.",
                "- `pair_metrics.csv`: pair metrics for configured confusable pairs.",
                "- `calibration_bins.csv`: confidence-bin accuracy gaps for ECE.",
                "- `bucket_summary.csv`: sample/error counts per forensic bucket.",
                "- `summary.json`: compact machine-readable summary.",
                "",
                "Do not use test for tuning. Test forensics are final audit only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "accuracy": summary["accuracy"],
                "focus_f1": focus_f1,
                "ece": calibration["ece"],
                "samples": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
