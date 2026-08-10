from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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


def _safe_int(row: Mapping[str, str], key: str, default: int = -1) -> int:
    value = str(row.get(key, "") or "").strip()
    if not value:
        return int(default)
    try:
        return int(float(value))
    except ValueError:
        return int(default)


def _safe_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "") or "").strip()
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


def _boundary_pair(a: int, b: int) -> str:
    left, right = sorted((int(a), int(b)))
    return f"{left}-{right}"


def _copy_or_link_image(source: Path, destination: Path, mode: str) -> str:
    if mode == "none":
        return "none"
    if not source.is_file():
        return "missing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "exists"
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"copy", "auto"}:
        shutil.copy2(source, destination)
        return "copy"
    raise ValueError(f"Unsupported image mode: {mode}")


def build_oof_cleanlab_review_manifest(
    *,
    cleanlab_manifest: Path,
    output_dir: Path,
    pairs: str,
    max_per_direction: int,
    require_issue: bool = True,
    image_mode: str = "auto",
) -> Dict[str, object]:
    rows, _ = _read_csv(cleanlab_manifest)
    selected_pairs = _parse_pairs(pairs)
    if not selected_pairs:
        raise ValueError("At least one pair is required.")
    if int(max_per_direction) <= 0:
        raise ValueError("max_per_direction must be positive.")
    if image_mode not in {"none", "hardlink", "copy", "auto"}:
        raise ValueError("image_mode must be one of: none, hardlink, copy, auto.")

    candidates: List[Dict[str, str]] = []
    skipped_non_train = 0
    skipped_not_issue = 0
    for row in rows:
        path_text = str(row.get("path", "") or row.get("image_path", "") or "")
        if _looks_like_non_train(path_text):
            skipped_non_train += 1
            continue
        if require_issue and _safe_int(row, "is_label_issue", default=1) != 1:
            skipped_not_issue += 1
            continue
        true_index = _safe_int(row, "true_index", default=-1)
        suggested_index = _safe_int(row, "suggested_index", default=_safe_int(row, "pred_index", default=-1))
        if (true_index, suggested_index) not in selected_pairs:
            continue
        candidates.append(dict(row))

    candidates.sort(
        key=lambda row: (
            _safe_int(row, "issue_rank", default=10**9),
            _safe_float(row, "self_confidence", default=1.0),
            -_safe_float(row, "top1_confidence", default=0.0),
        )
    )

    kept_counts: Counter[str] = Counter()
    output_rows: List[Dict[str, object]] = []
    image_actions: Counter[str] = Counter()
    output_dir = Path(output_dir)
    for row in candidates:
        true_index = _safe_int(row, "true_index", default=-1)
        suggested_index = _safe_int(row, "suggested_index", default=_safe_int(row, "pred_index", default=-1))
        direction = f"{true_index}->{suggested_index}"
        if kept_counts[direction] >= int(max_per_direction):
            continue
        kept_counts[direction] += 1

        review_index = len(output_rows) + 1
        review_id = f"oof_{review_index:05d}"
        source_path = Path(str(row.get("path", "") or ""))
        suffix = source_path.suffix.lower() or ".jpg"
        copied_name = (
            f"{review_id}_t{true_index}_s{suggested_index}_"
            f"rank{_safe_int(row, 'issue_rank', default=0):04d}{suffix}"
        )
        copied_path = output_dir / "review_images" / _boundary_pair(true_index, suggested_index) / direction.replace("->", "_to_") / copied_name
        image_actions[_copy_or_link_image(source_path, copied_path, image_mode)] += 1

        self_confidence = _safe_float(row, "self_confidence", default=0.0)
        top1_confidence = _safe_float(row, "top1_confidence", default=0.0)
        severity = top1_confidence * (1.0 - self_confidence)
        output_rows.append(
            {
                "review_id": review_id,
                "split": "train",
                "image_path": str(source_path),
                "target_index": true_index,
                "target_name": row.get("true_name", ""),
                "prediction_index": suggested_index,
                "prediction_name": row.get("suggested_name", row.get("pred_name", "")),
                "confidence": f"{top1_confidence:.8f}",
                "target_probability": f"{self_confidence:.8f}",
                "top2_margin": f"{_safe_float(row, 'top2_margin', default=0.0):.8f}",
                "boundary_pair": _boundary_pair(true_index, suggested_index),
                "reason": f"oof_cleanlab_{direction}",
                "severity": f"{severity:.8f}",
                "buckets": f"{direction};issue_rank_{_safe_int(row, 'issue_rank', default=0)}",
                "manual_label_status": "",
                "review_notes": "",
                "cleanlab_issue_rank": _safe_int(row, "issue_rank", default=-1),
                "cleanlab_label_quality": row.get("label_quality", ""),
                "cleanlab_self_confidence": row.get("self_confidence", ""),
                "cleanlab_top1_confidence": row.get("top1_confidence", ""),
                "cleanlab_normalized_entropy": row.get("normalized_entropy", ""),
            }
        )

    fieldnames = [
        "review_id",
        "split",
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "confidence",
        "target_probability",
        "top2_margin",
        "boundary_pair",
        "reason",
        "severity",
        "buckets",
        "manual_label_status",
        "review_notes",
        "cleanlab_issue_rank",
        "cleanlab_label_quality",
        "cleanlab_self_confidence",
        "cleanlab_top1_confidence",
        "cleanlab_normalized_entropy",
    ]
    output_csv = output_dir / "boundary_review_manifest.csv"
    _write_csv(output_csv, output_rows, fieldnames)
    summary = {
        "cleanlab_manifest": str(Path(cleanlab_manifest).resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows_read": int(len(rows)),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(output_rows)),
        "pairs": sorted(f"{a}->{b}" for a, b in selected_pairs),
        "max_per_direction": int(max_per_direction),
        "skipped_non_train": int(skipped_non_train),
        "skipped_not_issue": int(skipped_not_issue),
        "selected_by_direction": dict(sorted(kept_counts.items())),
        "image_actions": dict(sorted(image_actions.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only review manifest from OOF cleanlab issues.")
    parser.add_argument("--cleanlab-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="0-1,1-2")
    parser.add_argument("--max-per-direction", type=int, default=80)
    parser.add_argument("--include-non-issues", action="store_true")
    parser.add_argument("--image-mode", choices=["none", "hardlink", "copy", "auto"], default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_oof_cleanlab_review_manifest(
        cleanlab_manifest=args.cleanlab_manifest,
        output_dir=args.output_dir,
        pairs=args.pairs,
        max_per_direction=args.max_per_direction,
        require_issue=not bool(args.include_non_issues),
        image_mode=args.image_mode,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
