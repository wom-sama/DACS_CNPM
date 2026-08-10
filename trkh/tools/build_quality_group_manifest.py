from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

from trkh.core.config import load_data_spec
from trkh.data.dataset import IMAGE_EXTENSIONS, _pseudo_foreground_mask_array

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def _scan_classification_folder(data_yaml: Path, split: str) -> Tuple[List[Dict[str, object]], List[str]]:
    data_spec = load_data_spec(Path(data_yaml), class_name_mode="raw")
    split_dir = data_spec.split_images_dir(split)
    records: List[Dict[str, object]] = []
    for class_index, class_name in enumerate(data_spec.class_names):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "target_index": int(class_index),
                        "target_name": str(class_name),
                    }
                )
    if not records:
        raise ValueError(f"Khong tim thay anh split={split!r} trong {split_dir}")
    return records, list(data_spec.class_names)


def _fast_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    delta = max_channel - min_channel
    saturation = delta / np.maximum(max_channel, 1e-6)
    height, width = max_channel.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx + 0.5) / max(1.0, float(width)) * 2.0 - 1.0
    yy = (yy + 0.5) / max(1.0, float(height)) * 2.0 - 1.0
    central = ((xx / 0.92) ** 2 + (yy / 0.96) ** 2) <= 1.0
    color_mask = (saturation >= 0.07) & (max_channel >= 0.10)
    dark_detail = (max_channel <= 0.45) & (saturation >= 0.08)
    mask = central & (color_mask | dark_detail)
    if float(mask.mean()) < 0.08:
        mask = central
    return mask.astype(bool)


def _quality_stats(image_path: Path, *, mask_mode: str = "fast") -> Dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    normalized_mask_mode = str(mask_mode or "fast").strip().lower()
    mask = _pseudo_foreground_mask_array(image) if normalized_mask_mode == "pseudo" else _fast_foreground_mask(rgb)
    if mask.shape != luminance.shape:
        mask = np.ones_like(luminance, dtype=bool)
    foreground = mask.astype(bool)
    height, width = luminance.shape
    border_width = max(2, int(round(min(height, width) * 0.06)))
    border = np.zeros_like(foreground, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    fg_pixels = max(1, int(foreground.sum()))
    edge_delta = np.zeros_like(luminance, dtype=np.float32)
    edge_delta[:, 1:] = np.maximum(edge_delta[:, 1:], np.abs(luminance[:, 1:] - luminance[:, :-1]))
    edge_delta[1:, :] = np.maximum(edge_delta[1:, :], np.abs(luminance[1:, :] - luminance[:-1, :]))
    background = ~foreground
    return {
        "brightness_mean": float(luminance.mean()),
        "brightness_std": float(luminance.std()),
        "overexposed_fraction": float((luminance > 0.92).mean()),
        "underexposed_fraction": float((luminance < 0.06).mean()),
        "foreground_fraction": float(foreground.mean()),
        "foreground_border_fraction": float((foreground & border).sum() / float(fg_pixels)),
        "dark_foreground_fraction": float(((luminance < 0.28) & (saturation > 0.10) & foreground).sum() / float(fg_pixels)),
        "foreground_edge_mean": float(edge_delta[foreground].mean()) if bool(foreground.any()) else 0.0,
        "background_edge_mean": float(edge_delta[background].mean()) if bool(background.any()) else 0.0,
    }


def _process_record_quality(record: Mapping[str, object], mask_mode: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    image_path = Path(str(record["image_path"]))
    try:
        stats = _quality_stats(image_path, mask_mode=mask_mode)
    except Exception as exc:
        return None, f"{image_path}: {exc}"
    return {**record, **stats}, None


def _quality_bucket(stats: Mapping[str, float]) -> str:
    brightness = float(stats.get("brightness_mean", 0.0) or 0.0)
    contrast = float(stats.get("brightness_std", 0.0) or 0.0)
    over = float(stats.get("overexposed_fraction", 0.0) or 0.0)
    under = float(stats.get("underexposed_fraction", 0.0) or 0.0)
    foreground = float(stats.get("foreground_fraction", 0.0) or 0.0)
    border = float(stats.get("foreground_border_fraction", 0.0) or 0.0)
    dark_fg = float(stats.get("dark_foreground_fraction", 0.0) or 0.0)
    bg_edge = float(stats.get("background_edge_mean", 0.0) or 0.0)
    fg_edge = float(stats.get("foreground_edge_mean", 0.0) or 0.0)
    if brightness >= 0.72 or over >= 0.08:
        return "overbright"
    if brightness <= 0.28 or under >= 0.18:
        return "underdark"
    if contrast <= 0.105:
        return "low_contrast"
    if foreground <= 0.22 or border >= 0.20:
        return "partial_or_border"
    if foreground >= 0.86:
        return "mask_too_large"
    if dark_fg >= 0.10:
        return "dark_dirty_detail"
    if bg_edge > max(0.08, fg_edge * 1.20):
        return "background_clutter"
    return "normal"


def _prediction_summary(prediction_csvs: Sequence[Path]) -> Dict[str, Dict[str, object]]:
    summaries: Dict[str, Dict[str, object]] = {}
    for csv_path in prediction_csvs:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction CSV: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Prediction CSV phai co header: {path}")
            prob_columns = _probability_columns(reader.fieldnames)
            for row in reader:
                image_path = _path_value(row)
                if not image_path:
                    continue
                key = str(Path(image_path).resolve()).lower()
                target = _int_value(row, "target_index", "y_true", "label", "label_index")
                prediction = _int_value(row, "prediction_index", "y_pred", "pred", "pred_index")
                correct = _int_value(row, "correct", default=int(prediction == target))
                confidence = _float_value(row, "confidence", "top1_probability", default=0.0)
                top2_probability = _float_value(row, "top2_probability", default=0.0)
                margin = _float_value(row, "top2_margin", default=confidence - top2_probability)
                target_probability = 0.0
                target_column = prob_columns.get(int(target))
                if target_column is not None:
                    target_probability = _float_value(row, target_column, default=0.0)
                payload = summaries.setdefault(
                    key,
                    {
                        "count": 0,
                        "correct": 0,
                        "confidence": [],
                        "margin": [],
                        "target_probability": [],
                    },
                )
                payload["count"] = int(payload["count"]) + 1
                payload["correct"] = int(payload["correct"]) + int(correct > 0)
                payload["confidence"].append(float(confidence))
                payload["margin"].append(float(margin))
                payload["target_probability"].append(float(target_probability))
    return summaries


def _mean(values: object) -> float:
    if not isinstance(values, list) or not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _cartography_bucket(summary: Optional[Mapping[str, object]]) -> Tuple[str, Dict[str, float]]:
    if not summary:
        return "unscored", {
            "prediction_count": 0.0,
            "correct_rate": 0.0,
            "mean_confidence": 0.0,
            "mean_margin": 0.0,
            "mean_target_probability": 0.0,
        }
    count = max(1, int(summary.get("count", 0) or 0))
    correct_rate = float(summary.get("correct", 0) or 0) / float(count)
    mean_confidence = _mean(summary.get("confidence"))
    mean_margin = _mean(summary.get("margin"))
    mean_target_probability = _mean(summary.get("target_probability"))
    metrics = {
        "prediction_count": float(count),
        "correct_rate": float(correct_rate),
        "mean_confidence": float(mean_confidence),
        "mean_margin": float(mean_margin),
        "mean_target_probability": float(mean_target_probability),
    }
    if correct_rate >= 0.80 and mean_target_probability >= 0.42:
        return "easy", metrics
    if correct_rate <= 0.20 and mean_confidence >= 0.42:
        return "hard_disagreement", metrics
    if mean_target_probability <= 0.24:
        return "hard_low_self", metrics
    if mean_margin <= 0.075:
        return "ambiguous_boundary", metrics
    return "medium", metrics


def _sample_weight(quality_bucket: str, cartography_bucket: str) -> float:
    weight = 1.0
    if cartography_bucket == "hard_disagreement":
        weight *= 0.45
    elif cartography_bucket == "hard_low_self":
        weight *= 0.65
    elif cartography_bucket == "ambiguous_boundary":
        weight *= 0.78
    if quality_bucket in {"overbright", "underdark", "partial_or_border", "background_clutter"}:
        weight *= 1.12
    elif quality_bucket in {"low_contrast", "dark_dirty_detail"}:
        weight *= 1.06
    return float(max(0.25, min(1.35, weight)))


def _group_name_for(row: Mapping[str, object], group_key: str) -> str:
    label = f"c{int(row.get('target_index', 0) or 0)}"
    quality = str(row.get("quality_bucket", "normal") or "normal")
    cartography = str(row.get("cartography_bucket", "unscored") or "unscored")
    normalized = str(group_key or "class_quality").strip().lower()
    if normalized == "quality":
        return quality
    if normalized == "cartography":
        return cartography
    if normalized == "class_cartography":
        return f"{label}__{cartography}"
    if normalized == "class_quality_cartography":
        return f"{label}__{quality}__{cartography}"
    return f"{label}__{quality}"


def _copy_review_images(rows: Iterable[Mapping[str, object]], output_dir: Path, max_per_group: int) -> Dict[str, int]:
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("quality_group_name", "") or "unknown")].append(row)
    copied: Dict[str, int] = {}
    for group_name, group_rows in grouped.items():
        target_dir = output_dir / "review_images" / group_name.replace("/", "_").replace("\\", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for index, row in enumerate(group_rows[: max(0, int(max_per_group))], start=1):
            source = Path(str(row.get("image_path", "") or ""))
            if not source.is_file():
                continue
            target = target_dir / (
                f"{index:03d}_c{row.get('target_index')}_"
                f"{row.get('quality_bucket')}_{row.get('cartography_bucket')}{source.suffix.lower()}"
            )
            shutil.copy2(source, target)
            count += 1
        copied[group_name] = count
    return copied


def build_manifest(
    *,
    data_yaml: Path,
    output_dir: Path,
    split: str = "train",
    prediction_csvs: Sequence[Path] = (),
    group_key: str = "class_quality",
    max_samples: int = 0,
    workers: int = 0,
    mask_mode: str = "fast",
    progress_interval: int = 500,
    copy_images: bool = False,
    max_review_images_per_group: int = 8,
    dry_run: bool = False,
) -> Dict[str, object]:
    if str(split).strip().lower() != "train":
        raise ValueError("Quality group manifest chi duoc build cho train split de tranh dataleak.")
    records, class_names = _scan_classification_folder(data_yaml, split)
    if max_samples > 0:
        records = records[: int(max_samples)]
    prediction_by_path = _prediction_summary(prediction_csvs)

    rows: List[Dict[str, object]] = []
    group_to_index: Dict[str, int] = {}
    invalid_images = 0
    processed_records: List[Dict[str, object]] = []
    worker_count = max(0, int(workers))
    if worker_count <= 1:
        for index, record in enumerate(records, start=1):
            processed, error = _process_record_quality(record, mask_mode)
            if error:
                invalid_images += 1
                continue
            if processed is not None:
                processed_records.append(processed)
            if progress_interval > 0 and index % int(progress_interval) == 0:
                print(
                    json.dumps(
                        {
                            "progress": "quality_scan",
                            "processed": int(index),
                            "total": int(len(records)),
                            "invalid_images": int(invalid_images),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(_process_record_quality, record, mask_mode): index
                for index, record in enumerate(records, start=1)
            }
            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                processed, error = future.result()
                if error:
                    invalid_images += 1
                elif processed is not None:
                    processed_records.append(processed)
                if progress_interval > 0 and completed % int(progress_interval) == 0:
                    print(
                        json.dumps(
                            {
                                "progress": "quality_scan",
                                "processed": int(completed),
                                "total": int(len(records)),
                                "invalid_images": int(invalid_images),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    processed_records.sort(key=lambda item: str(item["image_path"]))
    for processed in processed_records:
        image_path = Path(str(processed["image_path"]))
        stats = {
            key: float(processed[key])
            for key in (
                "brightness_mean",
                "brightness_std",
                "overexposed_fraction",
                "underexposed_fraction",
                "foreground_fraction",
                "foreground_border_fraction",
                "dark_foreground_fraction",
                "foreground_edge_mean",
                "background_edge_mean",
            )
        }
        quality = _quality_bucket(stats)
        cartography, cartography_metrics = _cartography_bucket(
            prediction_by_path.get(str(image_path.resolve()).lower())
        )
        row: Dict[str, object] = {
            **{
                key: processed[key]
                for key in ("image_path", "target_index", "target_name")
            },
            **stats,
            **cartography_metrics,
            "quality_bucket": quality,
            "cartography_bucket": cartography,
        }
        group_name = _group_name_for(row, group_key)
        if group_name not in group_to_index:
            group_to_index[group_name] = len(group_to_index)
        row["quality_group_name"] = group_name
        row["quality_group_index"] = int(group_to_index[group_name])
        row["sample_weight"] = _sample_weight(quality, cartography)
        rows.append(row)

    if not rows:
        raise ValueError("Khong tao duoc dong quality-group hop le.")
    rows.sort(key=lambda item: (int(item["quality_group_index"]), str(item["image_path"])))

    output_dir = Path(output_dir)
    manifest_path = output_dir / "quality_groups_train_only.csv"
    summary_path = output_dir / "summary.json"
    fieldnames = [
        "image_path",
        "target_index",
        "target_name",
        "quality_group_index",
        "quality_group_name",
        "quality_bucket",
        "cartography_bucket",
        "sample_weight",
        "brightness_mean",
        "brightness_std",
        "overexposed_fraction",
        "underexposed_fraction",
        "foreground_fraction",
        "foreground_border_fraction",
        "dark_foreground_fraction",
        "foreground_edge_mean",
        "background_edge_mean",
        "prediction_count",
        "correct_rate",
        "mean_confidence",
        "mean_margin",
        "mean_target_probability",
    ]

    by_quality = Counter(str(row["quality_bucket"]) for row in rows)
    by_cartography = Counter(str(row["cartography_bucket"]) for row in rows)
    by_group = Counter(str(row["quality_group_name"]) for row in rows)
    copied: Dict[str, int] = {}
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(manifest_path, rows, fieldnames)
        if copy_images:
            copied = _copy_review_images(rows, output_dir, max_review_images_per_group)

    summary = {
        "data_yaml": str(Path(data_yaml).resolve()),
        "split": str(split),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "dry_run": bool(dry_run),
        "class_names": list(class_names),
        "rows": int(len(rows)),
        "invalid_images": int(invalid_images),
        "prediction_csvs": [str(Path(path).resolve()) for path in prediction_csvs],
        "prediction_matched_rows": int(sum(1 for row in rows if float(row["prediction_count"]) > 0.0)),
        "group_key": str(group_key),
        "mask_mode": str(mask_mode),
        "workers": int(worker_count),
        "num_groups": int(len(group_to_index)),
        "by_quality_bucket": dict(by_quality),
        "by_cartography_bucket": dict(by_cartography),
        "by_quality_group": dict(by_group),
        "weight_range": {
            "min": float(min(float(row["sample_weight"]) for row in rows)),
            "max": float(max(float(row["sample_weight"]) for row in rows)),
            "mean": float(sum(float(row["sample_weight"]) for row in rows) / len(rows)),
        },
        "copied_images": copied,
        "leakage_guard": "train split only; val/test are not scanned or required",
    }
    if not dry_run:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only quality group manifest for GroupDRO and sample weights."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prediction-csv", type=Path, action="append", default=[])
    parser.add_argument(
        "--group-key",
        type=str,
        default="class_quality",
        choices=("quality", "cartography", "class_quality", "class_cartography", "class_quality_cartography"),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mask-mode", type=str, choices=("fast", "pseudo"), default="fast")
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--copy-images", action="store_true", default=False)
    parser.add_argument("--max-review-images-per-group", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        data_yaml=args.data,
        output_dir=args.output_dir,
        split=args.split,
        prediction_csvs=args.prediction_csv,
        group_key=args.group_key,
        max_samples=args.max_samples,
        workers=args.workers,
        mask_mode=args.mask_mode,
        progress_interval=args.progress_interval,
        copy_images=args.copy_images,
        max_review_images_per_group=args.max_review_images_per_group,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
