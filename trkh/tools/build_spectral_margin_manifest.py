from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only targeted-margin manifest from spectral embedding "
            "ambiguity scores. The output is keyed by sample_index for YOLO object crops."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--focus-rival-classes", type=str, default="0,2")
    parser.add_argument("--min-ambiguity-nonfocus", type=float, default=0.78)
    parser.add_argument("--min-ambiguity-focus", type=float, default=0.74)
    parser.add_argument("--targeted-margin", type=float, default=0.08)
    parser.add_argument("--base-weight", type=float, default=0.65)
    parser.add_argument("--weight-scale", type=float, default=0.55)
    parser.add_argument("--max-weight", type=float, default=1.15)
    parser.add_argument("--max-samples", type=int, default=320)
    parser.add_argument("--max-per-pair", type=int, default=160)
    return parser.parse_args(argv)


def _int_value(row: Mapping[str, str], name: str, default: int = -1) -> int:
    try:
        return int(float(str(row.get(name, "") or "").strip()))
    except ValueError:
        return int(default)


def _float_value(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    try:
        value = float(str(row.get(name, "") or "").strip())
    except ValueError:
        return float(default)
    return float(value) if math.isfinite(value) else float(default)


def _parse_int_set(text: str) -> set[int]:
    output: set[int] = set()
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            output.add(int(item))
    return output


def _candidate_from_row(
    row: Mapping[str, str],
    *,
    focus_class_index: int,
    focus_rival_classes: set[int],
    min_ambiguity_nonfocus: float,
    min_ambiguity_focus: float,
    targeted_margin: float,
    base_weight: float,
    weight_scale: float,
    max_weight: float,
) -> Optional[Dict[str, object]]:
    sample_index = _int_value(row, "sample_index", -1)
    target_index = _int_value(row, "target_index", -1)
    rival_index = _int_value(row, "neighbor_top_rival_label", -1)
    base_prediction = _int_value(row, "base_prediction", -1)
    ambiguity = _float_value(row, "ambiguity_score", 0.0)
    if sample_index < 0 or target_index < 0 or rival_index < 0:
        return None

    if target_index != int(focus_class_index) and rival_index == int(focus_class_index):
        if ambiguity < float(min_ambiguity_nonfocus):
            return None
        reason = "spectral_focus_false_positive_risk"
        negative_index = int(focus_class_index)
    elif target_index == int(focus_class_index) and rival_index in focus_rival_classes:
        if ambiguity < float(min_ambiguity_focus):
            return None
        reason = "spectral_focus_false_negative_risk"
        negative_index = int(rival_index)
    else:
        return None

    weight = min(float(max_weight), max(1e-6, float(base_weight) + float(weight_scale) * float(ambiguity)))
    base_error = int(base_prediction >= 0 and base_prediction != target_index)
    pair_key = f"{target_index}->{negative_index}"
    return {
        "sample_index": int(sample_index),
        "image_path": str(row.get("image_path", "") or ""),
        "target_index": int(target_index),
        "negative_index": int(negative_index),
        "prediction_index": int(base_prediction),
        "targeted_margin": float(targeted_margin),
        "targeted_margin_weight": float(weight),
        "reason": reason,
        "ambiguity_score": float(ambiguity),
        "neighbor_same_label_fraction": _float_value(row, "neighbor_same_label_fraction", 0.0),
        "neighbor_top_rival_label": int(rival_index),
        "centroid_margin": _float_value(row, "centroid_margin", 0.0),
        "base_error": int(base_error),
        "pair_key": pair_key,
    }


def build_manifest(
    *,
    scores: Path,
    output: Path,
    focus_class_index: int = 1,
    focus_rival_classes: Optional[set[int]] = None,
    min_ambiguity_nonfocus: float = 0.78,
    min_ambiguity_focus: float = 0.74,
    targeted_margin: float = 0.08,
    base_weight: float = 0.65,
    weight_scale: float = 0.55,
    max_weight: float = 1.15,
    max_samples: int = 320,
    max_per_pair: int = 160,
) -> Dict[str, object]:
    scores = Path(scores)
    if not scores.is_file():
        raise FileNotFoundError(f"Missing spectral scores CSV: {scores}")
    focus_rivals = set(focus_rival_classes or {0, 2})
    candidates: List[Dict[str, object]] = []
    with scores.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Scores CSV must have a header: {scores}")
        for row in reader:
            candidate = _candidate_from_row(
                row,
                focus_class_index=int(focus_class_index),
                focus_rival_classes=focus_rivals,
                min_ambiguity_nonfocus=float(min_ambiguity_nonfocus),
                min_ambiguity_focus=float(min_ambiguity_focus),
                targeted_margin=float(targeted_margin),
                base_weight=float(base_weight),
                weight_scale=float(weight_scale),
                max_weight=float(max_weight),
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("No spectral targeted-margin candidates selected.")

    candidates.sort(
        key=lambda item: (
            -int(item["base_error"]),
            -float(item["ambiguity_score"]),
            int(item["sample_index"]),
        )
    )
    kept: List[Dict[str, object]] = []
    per_pair: Dict[str, int] = defaultdict(int)
    seen_indices: set[int] = set()
    for candidate in candidates:
        sample_index = int(candidate["sample_index"])
        pair_key = str(candidate["pair_key"])
        if sample_index in seen_indices:
            continue
        if int(max_per_pair) > 0 and per_pair[pair_key] >= int(max_per_pair):
            continue
        kept.append(candidate)
        seen_indices.add(sample_index)
        per_pair[pair_key] += 1
        if int(max_samples) > 0 and len(kept) >= int(max_samples):
            break
    if not kept:
        raise ValueError("No spectral targeted-margin candidates survived caps.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "negative_index",
        "prediction_index",
        "targeted_margin",
        "targeted_margin_weight",
        "reason",
        "ambiguity_score",
        "neighbor_same_label_fraction",
        "neighbor_top_rival_label",
        "centroid_margin",
        "base_error",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in kept:
            writer.writerow(
                {
                    "sample_index": int(candidate["sample_index"]),
                    "image_path": str(candidate["image_path"]),
                    "target_index": int(candidate["target_index"]),
                    "negative_index": int(candidate["negative_index"]),
                    "prediction_index": int(candidate["prediction_index"]),
                    "targeted_margin": f"{float(candidate['targeted_margin']):.10g}",
                    "targeted_margin_weight": f"{float(candidate['targeted_margin_weight']):.10g}",
                    "reason": str(candidate["reason"]),
                    "ambiguity_score": f"{float(candidate['ambiguity_score']):.10g}",
                    "neighbor_same_label_fraction": f"{float(candidate['neighbor_same_label_fraction']):.10g}",
                    "neighbor_top_rival_label": int(candidate["neighbor_top_rival_label"]),
                    "centroid_margin": f"{float(candidate['centroid_margin']):.10g}",
                    "base_error": int(candidate["base_error"]),
                }
            )

    by_reason = Counter(str(item["reason"]) for item in kept)
    by_pair = Counter(str(item["pair_key"]) for item in kept)
    summary = {
        "scores": str(scores.resolve()),
        "output": str(output.resolve()),
        "candidates": int(len(candidates)),
        "rows": int(len(kept)),
        "focus_class_index": int(focus_class_index),
        "focus_rival_classes": sorted(int(value) for value in focus_rivals),
        "min_ambiguity_nonfocus": float(min_ambiguity_nonfocus),
        "min_ambiguity_focus": float(min_ambiguity_focus),
        "targeted_margin": float(targeted_margin),
        "base_weight": float(base_weight),
        "weight_scale": float(weight_scale),
        "max_weight": float(max_weight),
        "max_samples": int(max_samples),
        "max_per_pair": int(max_per_pair),
        "by_reason": dict(by_reason),
        "by_target_negative_pair": dict(by_pair),
        "base_error_rows": int(sum(int(item["base_error"]) for item in kept)),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_manifest(
        scores=Path(args.scores),
        output=Path(args.output),
        focus_class_index=int(args.focus_class_index),
        focus_rival_classes=_parse_int_set(str(args.focus_rival_classes)),
        min_ambiguity_nonfocus=float(args.min_ambiguity_nonfocus),
        min_ambiguity_focus=float(args.min_ambiguity_focus),
        targeted_margin=float(args.targeted_margin),
        base_weight=float(args.base_weight),
        weight_scale=float(args.weight_scale),
        max_weight=float(args.max_weight),
        max_samples=int(args.max_samples),
        max_per_pair=int(args.max_per_pair),
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
