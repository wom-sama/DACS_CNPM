from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


def _read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV must have a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _first(row: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _safe_int(row: Mapping[str, str], *keys: str, default: int = -1) -> int:
    value = _first(row, *keys)
    if not value:
        return int(default)
    try:
        return int(float(value))
    except ValueError:
        return int(default)


def _safe_float(row: Mapping[str, str], *keys: str, default: float = 0.0) -> float:
    value = _first(row, *keys)
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _looks_like_non_train(path_text: str) -> bool:
    parts = [part for part in str(path_text or "").replace("\\", "/").lower().split("/") if part]
    return "val" in parts or "valid" in parts or "validation" in parts or "test" in parts


def _parse_pairs(text: str) -> Set[Tuple[int, int]]:
    parsed: Set[Tuple[int, int]] = set()
    for raw_item in str(text or "").replace(";", ",").split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "->" in item:
            left, right = item.split("->", 1)
            parsed.add((int(left.strip()), int(right.strip())))
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            a = int(left.strip())
            b = int(right.strip())
            parsed.add((a, b))
            parsed.add((b, a))
            continue
        raise ValueError(f"Invalid pair item: {raw_item!r}. Use '0-1' or '0->1'.")
    return parsed


def _infer_num_classes(rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str], requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    max_prob_index = -1
    for name in fieldnames:
        normalized = str(name or "").strip().lower()
        if not normalized.startswith("prob_"):
            continue
        parts = normalized.split("_", 2)
        if len(parts) < 2:
            continue
        try:
            max_prob_index = max(max_prob_index, int(parts[1]))
        except ValueError:
            continue
    max_label = -1
    for row in rows:
        max_label = max(max_label, _safe_int(row, "true_index", "target_index", default=-1))
        max_label = max(max_label, _safe_int(row, "suggested_index", "prediction_index", "pred_index", default=-1))
    return max(max_prob_index, max_label) + 1


def build_oof_cleanlab_strict_soft_targets(
    *,
    source_manifest: Path,
    output_dir: Path,
    pairs: str,
    num_classes: int = 0,
    alpha: float = 0.85,
    min_top1_confidence: float = 0.95,
    max_self_confidence: float = 0.05,
    min_top2_margin: float = 0.80,
    max_per_direction: int = 80,
    require_issue: bool = True,
) -> Dict[str, object]:
    rows, fieldnames = _read_csv(source_manifest)
    selected_pairs = _parse_pairs(pairs)
    if not selected_pairs:
        raise ValueError("At least one pair is required.")
    num_classes = _infer_num_classes(rows, fieldnames, int(num_classes))
    if num_classes <= 1:
        raise ValueError("Could not infer num_classes.")
    alpha = max(0.0, min(1.0, float(alpha)))
    if int(max_per_direction) <= 0:
        raise ValueError("max_per_direction must be positive.")

    candidates: List[Dict[str, str]] = []
    skipped_non_train = 0
    skipped_not_issue = 0
    skipped_threshold = 0
    skipped_invalid = 0
    for row in rows:
        image_path = _first(row, "image_path", "path", "sample_path")
        if not image_path:
            skipped_invalid += 1
            continue
        if _looks_like_non_train(image_path):
            skipped_non_train += 1
            continue
        if require_issue and _safe_int(row, "is_label_issue", default=1) != 1:
            skipped_not_issue += 1
            continue
        target_index = _safe_int(row, "target_index", "true_index", "y_true", default=-1)
        soft_index = _safe_int(row, "soft_target_index", "suggested_index", "prediction_index", "pred_index", "y_pred", default=-1)
        if not (0 <= target_index < num_classes) or not (0 <= soft_index < num_classes) or target_index == soft_index:
            skipped_invalid += 1
            continue
        if (target_index, soft_index) not in selected_pairs:
            continue
        top1_confidence = _safe_float(row, "cleanlab_top1_confidence", "top1_confidence", "confidence", default=0.0)
        self_confidence = _safe_float(
            row,
            "cleanlab_self_confidence",
            "self_confidence",
            "target_probability",
            default=1.0,
        )
        top2_margin = _safe_float(row, "top2_margin", default=0.0)
        if (
            top1_confidence < float(min_top1_confidence)
            or self_confidence > float(max_self_confidence)
            or top2_margin < float(min_top2_margin)
        ):
            skipped_threshold += 1
            continue
        candidates.append(dict(row))

    candidates.sort(
        key=lambda row: (
            _safe_int(row, "cleanlab_issue_rank", "issue_rank", default=10**9),
            _safe_float(row, "cleanlab_self_confidence", "self_confidence", "target_probability", default=1.0),
            -_safe_float(row, "cleanlab_top1_confidence", "top1_confidence", "confidence", default=0.0),
        )
    )

    kept_counts: Counter[str] = Counter()
    output_rows: List[Dict[str, object]] = []
    for row in candidates:
        image_path = _first(row, "image_path", "path", "sample_path")
        target_index = _safe_int(row, "target_index", "true_index", "y_true", default=-1)
        soft_index = _safe_int(row, "soft_target_index", "suggested_index", "prediction_index", "pred_index", "y_pred", default=-1)
        direction = f"{target_index}->{soft_index}"
        if kept_counts[direction] >= int(max_per_direction):
            continue
        kept_counts[direction] += 1
        probabilities = [0.0 for _ in range(num_classes)]
        probabilities[target_index] = 1.0 - alpha
        probabilities[soft_index] += alpha
        reason_rank = _first(row, "cleanlab_issue_rank", "issue_rank") or "na"
        output_rows.append(
            {
                "sample_index": _first(row, "sample_index", "dataset_index"),
                "image_path": image_path,
                "target_index": int(target_index),
                "soft_target_index": int(soft_index),
                "alpha": f"{alpha:.10g}",
                "reason": f"strict_oof_cleanlab_{direction}_rank{reason_rank}",
                "review_id": row.get("review_id", ""),
                "prediction_index": int(soft_index),
                "top2_margin": f"{_safe_float(row, 'top2_margin', default=0.0):.10g}",
                **{f"soft_{index}": f"{probabilities[index]:.10g}" for index in range(num_classes)},
            }
        )

    output_dir = Path(output_dir)
    output_csv = output_dir / "strict_oof_cleanlab_soft_targets_train_only.csv"
    fieldnames_out = [
        "sample_index",
        "image_path",
        "target_index",
        "soft_target_index",
        "alpha",
        "reason",
        "review_id",
        "prediction_index",
        "top2_margin",
        *[f"soft_{index}" for index in range(num_classes)],
    ]
    _write_csv(output_csv, output_rows, fieldnames_out)
    summary = {
        "source_manifest": str(Path(source_manifest).resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows_read": int(len(rows)),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(output_rows)),
        "pairs": sorted(f"{a}->{b}" for a, b in selected_pairs),
        "num_classes": int(num_classes),
        "alpha": float(alpha),
        "min_top1_confidence": float(min_top1_confidence),
        "max_self_confidence": float(max_self_confidence),
        "min_top2_margin": float(min_top2_margin),
        "max_per_direction": int(max_per_direction),
        "skipped_non_train": int(skipped_non_train),
        "skipped_not_issue": int(skipped_not_issue),
        "skipped_threshold": int(skipped_threshold),
        "skipped_invalid": int(skipped_invalid),
        "selected_by_direction": dict(sorted(kept_counts.items())),
        "rows_with_sample_index": int(
            sum(1 for row in output_rows if str(row.get("sample_index", "") or "").strip())
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict soft targets from train-only OOF cleanlab issues.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="0-1,1-2")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--min-top1-confidence", type=float, default=0.95)
    parser.add_argument("--max-self-confidence", type=float, default=0.05)
    parser.add_argument("--min-top2-margin", type=float, default=0.80)
    parser.add_argument("--max-per-direction", type=int, default=80)
    parser.add_argument("--include-non-issues", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_oof_cleanlab_strict_soft_targets(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        pairs=args.pairs,
        num_classes=args.num_classes,
        alpha=args.alpha,
        min_top1_confidence=args.min_top1_confidence,
        max_self_confidence=args.max_self_confidence,
        min_top2_margin=args.min_top2_margin,
        max_per_direction=args.max_per_direction,
        require_issue=not bool(args.include_non_issues),
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
