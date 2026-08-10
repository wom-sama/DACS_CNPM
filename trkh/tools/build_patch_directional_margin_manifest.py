from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only targeted-margin manifest from directional "
            "patch-verifier gate changed cases."
        )
    )
    parser.add_argument("--changes-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--directions", type=str, default="1->0")
    parser.add_argument("--suppressor-margin", type=float, default=0.06)
    parser.add_argument("--suppressor-weight", type=float, default=1.5)
    parser.add_argument("--protector-margin", type=float, default=0.04)
    parser.add_argument("--protector-weight", type=float, default=0.8)
    parser.add_argument("--include-protectors", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _parse_directions(value: str) -> set[str]:
    directions: set[str] = set()
    for raw in str(value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"direction must look like '1->0', got {item!r}")
        left, right = item.split("->", 1)
        directions.add(f"{int(left)}->{int(right)}")
    if not directions:
        raise ValueError("at least one direction is required")
    return directions


def _as_int(row: Mapping[str, str], key: str) -> int:
    value = str(row.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"missing integer column {key!r}")
    return int(float(value))


def _train_row(row: Mapping[str, str]) -> bool:
    split = str(row.get("split", "") or "").strip().lower()
    image_path = str(row.get("image_path", "") or "").replace("/", "\\").lower()
    return split == "train" and "\\train\\" in image_path


def _manifest_row(
    row: Mapping[str, str],
    *,
    target_index: int,
    negative_index: int,
    margin: float,
    weight: float,
    reason: str,
) -> Dict[str, object]:
    return {
        "sample_index": _as_int(row, "sample_index"),
        "image_path": str(row.get("image_path", "") or "").strip(),
        "target_index": int(target_index),
        "negative_index": int(negative_index),
        "targeted_margin": float(margin),
        "targeted_margin_weight": float(weight),
        "reason": reason,
        "source_direction": str(row.get("direction", "") or "").strip(),
        "source_change_type": str(row.get("change_type", "") or "").strip(),
        "verifier_confidence": str(row.get("verifier_confidence", "") or "").strip(),
        "pair_min_probability": str(row.get("pair_min_probability", "") or "").strip(),
        "pair_margin": str(row.get("pair_margin", "") or "").strip(),
    }


def build_manifest(
    *,
    changes_csv: Path,
    output: Path,
    focus_class: int,
    directions: str,
    suppressor_margin: float,
    suppressor_weight: float,
    protector_margin: float,
    protector_weight: float,
    include_protectors: bool,
) -> Dict[str, object]:
    allowed_directions = _parse_directions(directions)
    focus_class = int(focus_class)
    rows_out: List[Dict[str, object]] = []
    skipped: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_target_negative: Counter[str] = Counter()
    duplicate_sample_indices = 0
    seen_sample_indices: set[int] = set()

    with Path(changes_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty changed-case CSV: {changes_csv}")
        required = {
            "split",
            "sample_index",
            "image_path",
            "target_index",
            "base_prediction",
            "final_prediction",
            "change_type",
            "direction",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{changes_csv} missing required columns: {missing}")
        for row in reader:
            if not _train_row(row):
                skipped["non_train"] += 1
                continue
            direction = str(row.get("direction", "") or "").strip()
            if direction not in allowed_directions:
                skipped["direction"] += 1
                continue
            sample_index = _as_int(row, "sample_index")
            if sample_index in seen_sample_indices:
                duplicate_sample_indices += 1
                continue
            target_index = _as_int(row, "target_index")
            base_prediction = _as_int(row, "base_prediction")
            final_prediction = _as_int(row, "final_prediction")
            change_type = str(row.get("change_type", "") or "").strip()

            if target_index != focus_class:
                negative_index = focus_class
                reason = "patch_gate_false_positive_suppressor"
                manifest_row = _manifest_row(
                    row,
                    target_index=target_index,
                    negative_index=negative_index,
                    margin=float(suppressor_margin),
                    weight=float(suppressor_weight),
                    reason=reason,
                )
            elif include_protectors and change_type == "harm":
                negative_index = final_prediction if final_prediction != target_index else base_prediction
                if negative_index == target_index:
                    skipped["protector_same_negative"] += 1
                    continue
                reason = "patch_gate_class1_recall_protector"
                manifest_row = _manifest_row(
                    row,
                    target_index=target_index,
                    negative_index=negative_index,
                    margin=float(protector_margin),
                    weight=float(protector_weight),
                    reason=reason,
                )
            else:
                skipped["focus_non_harm"] += 1
                continue

            seen_sample_indices.add(sample_index)
            rows_out.append(manifest_row)
            by_reason[str(manifest_row["reason"])] += 1
            by_target_negative[f"{manifest_row['target_index']}->{manifest_row['negative_index']}"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "negative_index",
        "targeted_margin",
        "targeted_margin_weight",
        "reason",
        "source_direction",
        "source_change_type",
        "verifier_confidence",
        "pair_min_probability",
        "pair_margin",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    summary = {
        "changes_csv": str(Path(changes_csv)),
        "output": str(output),
        "rows": len(rows_out),
        "focus_class": focus_class,
        "directions": sorted(allowed_directions),
        "suppressor": {
            "margin": float(suppressor_margin),
            "weight": float(suppressor_weight),
        },
        "protector": {
            "enabled": bool(include_protectors),
            "margin": float(protector_margin),
            "weight": float(protector_weight),
        },
        "by_reason": dict(by_reason),
        "by_target_negative": dict(by_target_negative),
        "skipped": dict(skipped),
        "duplicate_sample_indices": int(duplicate_sample_indices),
        "leakage_guard": "only rows with split=train and image_path containing train are emitted",
    }
    return summary


def run(args: argparse.Namespace) -> Dict[str, object]:
    summary = build_manifest(
        changes_csv=Path(args.changes_csv),
        output=Path(args.output),
        focus_class=int(args.focus_class),
        directions=str(args.directions),
        suppressor_margin=float(args.suppressor_margin),
        suppressor_weight=float(args.suppressor_weight),
        protector_margin=float(args.protector_margin),
        protector_weight=float(args.protector_weight),
        include_protectors=bool(args.include_protectors),
    )
    summary_output = args.summary_output or (Path(args.output).with_suffix(".summary.json"))
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run(_parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
