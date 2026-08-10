from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only sample-weight manifest from quality/cartography buckets. "
            "This is intended to reduce boundary-noise learning without oversampling a class."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="sample_weights_quality_guard_train_only.csv")
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--hard-weight", type=float, default=0.55)
    parser.add_argument("--ambiguous-weight", type=float, default=0.75)
    parser.add_argument("--focus-medium-weight", type=float, default=1.25)
    parser.add_argument("--focus-ambiguous-weight", type=float, default=0.90)
    parser.add_argument("--focus-hard-weight", type=float, default=0.75)
    parser.add_argument("--overbright-weight", type=float, default=0.85)
    parser.add_argument("--underdark-weight", type=float, default=0.85)
    parser.add_argument("--low-contrast-weight", type=float, default=0.85)
    parser.add_argument("--partial-weight", type=float, default=0.70)
    parser.add_argument("--min-weight", type=float, default=0.45)
    parser.add_argument("--max-weight", type=float, default=1.50)
    parser.add_argument("--normalize-mean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay input manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest rong hoac khong co header: {path}")
        required = {"image_path", "target_index", "quality_bucket", "cartography_bucket"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest thieu cot bat buoc: {sorted(missing)}")
        return list(reader)


def _quality_cap(row: Dict[str, str], args: argparse.Namespace) -> float:
    quality = str(row.get("quality_bucket", "")).strip().lower()
    caps = []
    if quality == "overbright":
        caps.append(float(args.overbright_weight))
    if quality == "underdark":
        caps.append(float(args.underdark_weight))
    if quality == "low_contrast":
        caps.append(float(args.low_contrast_weight))
    if quality == "partial_or_border":
        caps.append(float(args.partial_weight))
    return min(caps) if caps else float(args.max_weight)


def _base_weight(row: Dict[str, str], args: argparse.Namespace) -> float:
    try:
        target_index = int(float(row.get("target_index", "")))
    except ValueError:
        target_index = -1
    cartography = str(row.get("cartography_bucket", "")).strip().lower()
    if cartography == "hard_low_self":
        weight = float(args.hard_weight)
    elif cartography == "ambiguous_boundary":
        weight = float(args.ambiguous_weight)
    else:
        weight = 1.0

    if target_index == int(args.focus_class_index):
        if cartography == "hard_low_self":
            weight = max(weight, float(args.focus_hard_weight))
        elif cartography == "ambiguous_boundary":
            weight = max(weight, float(args.focus_ambiguous_weight))
        else:
            weight = max(weight, float(args.focus_medium_weight))

    weight = min(weight, _quality_cap(row, args))
    return max(float(args.min_weight), min(float(args.max_weight), weight))


def _guard_train_only(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    split_counts: Counter[str] = Counter()
    for row in rows:
        path_text = str(row.get("image_path", "")).replace("\\", "/").lower()
        if "/train/" in path_text:
            split_counts["train"] += 1
        elif "/val/" in path_text:
            split_counts["val"] += 1
        elif "/test/" in path_text:
            split_counts["test"] += 1
        else:
            split_counts["unknown"] += 1
    if split_counts["val"] or split_counts["test"]:
        raise ValueError(
            "Input manifest co val/test path; sample-weight manifest phai train-only: "
            f"{dict(split_counts)}"
        )
    return dict(split_counts)


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_manifest)
    split_counts = _guard_train_only(rows)
    weighted_rows: List[Dict[str, object]] = []
    for row in rows:
        weight = _base_weight(row, args)
        weighted_rows.append(
            {
                "image_path": row["image_path"],
                "target_index": row.get("target_index", ""),
                "target_name": row.get("target_name", ""),
                "sample_weight": weight,
                "quality_bucket": row.get("quality_bucket", ""),
                "cartography_bucket": row.get("cartography_bucket", ""),
                "quality_group_name": row.get("quality_group_name", ""),
            }
        )

    raw_weights = [float(row["sample_weight"]) for row in weighted_rows]
    raw_mean = sum(raw_weights) / max(1, len(raw_weights))
    if bool(args.normalize_mean) and raw_mean > 0:
        for row in weighted_rows:
            normalized = float(row["sample_weight"]) / raw_mean
            row["sample_weight"] = max(
                float(args.min_weight),
                min(float(args.max_weight), normalized),
            )

    final_weights = [float(row["sample_weight"]) for row in weighted_rows]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / str(args.output_name)
    summary_path = output_dir / "summary.json"
    summary = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(manifest_path),
        "rows": len(weighted_rows),
        "dry_run": bool(args.dry_run),
        "split_counts": split_counts,
        "focus_class_index": int(args.focus_class_index),
        "raw_weight_mean": raw_mean,
        "weight_min": min(final_weights) if final_weights else 0.0,
        "weight_max": max(final_weights) if final_weights else 0.0,
        "weight_mean": sum(final_weights) / max(1, len(final_weights)),
        "by_cartography": dict(Counter(str(row["cartography_bucket"]) for row in weighted_rows)),
        "by_quality": dict(Counter(str(row["quality_bucket"]) for row in weighted_rows)),
    }
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "image_path",
                "target_index",
                "target_name",
                "sample_weight",
                "quality_bucket",
                "cartography_bucket",
                "quality_group_name",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(weighted_rows)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
