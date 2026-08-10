from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows, fieldnames


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
            return float(parsed)
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


def _class_name(row: Mapping[str, str], target_index: int) -> str:
    return (
        str(row.get("target_name", "") or "").strip()
        or str(row.get("true_name", "") or "").strip()
        or f"class_{int(target_index)}"
    )


def _probability_columns(fieldnames: Iterable[str]) -> Dict[int, str]:
    columns: Dict[int, str] = {}
    for name in fieldnames:
        normalized = str(name or "").strip()
        if not normalized.lower().startswith("prob_"):
            continue
        try:
            class_index = int(normalized.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        columns[class_index] = normalized
    return dict(sorted(columns.items()))


def _target_probability(row: Mapping[str, str], prob_columns: Mapping[int, str], target_index: int) -> float:
    column = prob_columns.get(int(target_index), f"prob_{int(target_index)}")
    return _float_value(row, column, "confidence_mean", "self_confidence", default=0.0)


def _anchor_score(
    row: Mapping[str, str],
    *,
    prob_columns: Mapping[int, str],
    target_index: int,
) -> float:
    confidence = _float_value(row, "confidence_mean", "confidence", "top1_confidence", default=0.0)
    target_probability = _target_probability(row, prob_columns, target_index)
    correctness = _float_value(row, "correctness_mean", default=0.0)
    margin = _float_value(row, "top2_margin", default=0.0)
    variability = _float_value(row, "confidence_std", default=0.0)
    ce_loss = _float_value(row, "ce_loss_mean", "loss", default=0.0)
    return float(
        2.0 * correctness
        + 0.8 * confidence
        + 0.6 * target_probability
        + 0.4 * margin
        - 0.25 * variability
        - 0.10 * ce_loss
    )


def build_manifest(
    *,
    cartography_csv: Path,
    output_dir: Path,
    per_class: int,
    min_correctness: float,
    min_confidence: float,
    min_target_probability: float,
    max_confidence_std: float,
    max_ce_loss: float,
    sample_weight: float,
    allow_non_train_paths: bool = False,
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(Path(cartography_csv))
    prob_columns = _probability_columns(fieldnames)
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    skipped_non_train = 0
    skipped_invalid = 0
    skipped_thresholds = 0
    duplicate_paths = 0
    seen_paths: set[str] = set()

    for row in rows:
        image_path = _path_value(row)
        if not image_path:
            skipped_invalid += 1
            continue
        if not allow_non_train_paths and not _looks_like_train(image_path):
            skipped_non_train += 1
            continue
        target_index = _int_value(row, "target_index", "true_index", "label", default=-1)
        if target_index < 0:
            skipped_invalid += 1
            continue
        prediction_index = _int_value(row, "prediction_index", "pred_index", default=target_index)
        correctness = _float_value(row, "correctness_mean", default=1.0 if prediction_index == target_index else 0.0)
        confidence = _float_value(row, "confidence_mean", "confidence", "top1_confidence", default=0.0)
        target_probability = _target_probability(row, prob_columns, target_index)
        confidence_std = _float_value(row, "confidence_std", default=0.0)
        ce_loss = _float_value(row, "ce_loss_mean", "loss", default=0.0)
        if prediction_index != target_index and correctness < 1.0:
            skipped_thresholds += 1
            continue
        if correctness < min_correctness:
            skipped_thresholds += 1
            continue
        if confidence < min_confidence or target_probability < min_target_probability:
            skipped_thresholds += 1
            continue
        if max_confidence_std >= 0.0 and confidence_std > max_confidence_std:
            skipped_thresholds += 1
            continue
        if max_ce_loss >= 0.0 and ce_loss > max_ce_loss:
            skipped_thresholds += 1
            continue
        resolved = str(Path(image_path).resolve())
        key = resolved.lower()
        if key in seen_paths:
            duplicate_paths += 1
            continue
        seen_paths.add(key)
        score = _anchor_score(row, prob_columns=prob_columns, target_index=target_index)
        grouped[int(target_index)].append(
            {
                "image_path": resolved,
                "target_index": int(target_index),
                "target_name": _class_name(row, target_index),
                "prediction_index": int(prediction_index),
                "confidence_mean": f"{confidence:.8f}",
                "target_probability": f"{target_probability:.8f}",
                "correctness_mean": f"{correctness:.8f}",
                "confidence_std": f"{confidence_std:.8f}",
                "ce_loss_mean": f"{ce_loss:.8f}",
                "top2_margin": f"{_float_value(row, 'top2_margin', default=0.0):.8f}",
                "anchor_score": f"{score:.8f}",
                "sample_weight": f"{float(sample_weight):.6f}",
                "reason": "clean_balanced_anchor",
            }
        )

    selected: List[Dict[str, object]] = []
    selected_by_class: Dict[str, int] = {}
    for class_index, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: float(item["anchor_score"]), reverse=True)
        kept = candidates[: max(0, int(per_class))]
        selected.extend(kept)
        selected_by_class[str(class_index)] = int(len(kept))
    selected.sort(key=lambda item: (int(item["target_index"]), -float(item["anchor_score"])))

    output_dir = Path(output_dir)
    manifest_path = output_dir / "clean_anchor_hard_samples_train_only.csv"
    fieldnames_out = [
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "confidence_mean",
        "target_probability",
        "correctness_mean",
        "confidence_std",
        "ce_loss_mean",
        "top2_margin",
        "anchor_score",
        "sample_weight",
        "reason",
    ]
    _write_csv(manifest_path, selected, fieldnames_out)
    sample_weight_path = output_dir / "clean_anchor_sample_weights_train_only.csv"
    _write_csv(sample_weight_path, selected, fieldnames_out)

    by_class_available = {str(key): int(len(value)) for key, value in sorted(grouped.items())}
    summary = {
        "cartography_csv": str(Path(cartography_csv).resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sample_weight_manifest": str(sample_weight_path.resolve()),
        "rows": int(len(rows)),
        "selected": int(len(selected)),
        "per_class": int(per_class),
        "selected_by_class": selected_by_class,
        "available_by_class": by_class_available,
        "skipped_non_train": int(skipped_non_train),
        "skipped_invalid": int(skipped_invalid),
        "skipped_thresholds": int(skipped_thresholds),
        "duplicate_paths": int(duplicate_paths),
        "thresholds": {
            "min_correctness": float(min_correctness),
            "min_confidence": float(min_confidence),
            "min_target_probability": float(min_target_probability),
            "max_confidence_std": float(max_confidence_std),
            "max_ce_loss": float(max_ce_loss),
        },
        "sample_weight": float(sample_weight),
        "by_target_name": dict(Counter(str(row["target_name"]) for row in selected)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-only, class-balanced clean-anchor manifest from data cartography."
    )
    parser.add_argument("--cartography-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=160)
    parser.add_argument("--min-correctness", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-target-probability", type=float, default=0.0)
    parser.add_argument("--max-confidence-std", type=float, default=-1.0)
    parser.add_argument("--max-ce-loss", type=float, default=-1.0)
    parser.add_argument("--sample-weight", type=float, default=1.35)
    parser.add_argument("--allow-non-train-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        cartography_csv=args.cartography_csv,
        output_dir=args.output_dir,
        per_class=args.per_class,
        min_correctness=args.min_correctness,
        min_confidence=args.min_confidence,
        min_target_probability=args.min_target_probability,
        max_confidence_std=args.max_confidence_std,
        max_ce_loss=args.max_ce_loss,
        sample_weight=args.sample_weight,
        allow_non_train_paths=bool(args.allow_non_train_paths),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
