from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


Pair = Tuple[int, int]


def _parse_pair(value: str) -> Pair:
    text = str(value or "").strip()
    if "-" not in text:
        raise ValueError(f"pair must use A-B syntax, got {value!r}")
    left, right = text.split("-", 1)
    a = int(left.strip())
    b = int(right.strip())
    if a == b:
        raise ValueError("pair must contain two different classes")
    return min(a, b), max(a, b)


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    text = str(row.get(key, "") or "").strip()
    if not text:
        return float(default)
    return float(text)


def _int(row: Mapping[str, str], key: str) -> Optional[int]:
    text = str(row.get(key, "") or "").strip()
    if not text:
        return None
    return int(float(text))


def build_manifest(
    *,
    teacher_csv: Path,
    output_dir: Path,
    pair: Pair = (0, 1),
    disagreement_confidence_threshold: float = 0.65,
    disagreement_weight: float = 0.35,
) -> Dict[str, object]:
    pair = (int(pair[0]), int(pair[1]))
    threshold = max(0.0, min(1.0, float(disagreement_confidence_threshold)))
    weight = max(1e-6, float(disagreement_weight))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "patch_oof_sample_weights_train_only.csv"

    selected: List[Dict[str, object]] = []
    total_rows = 0
    pair_rows = 0
    by_target_prediction: Counter[str] = Counter()
    with Path(teacher_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"teacher CSV must have a header: {teacher_csv}")
        required = {"sample_index", "target_index", f"prob_{pair[0]}", f"prob_{pair[1]}"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"teacher CSV missing columns {missing}: {teacher_csv}")
        for row in reader:
            total_rows += 1
            sample_index = _int(row, "sample_index")
            target = _int(row, "target_index")
            if sample_index is None or target is None or int(target) not in pair:
                continue
            pair_rows += 1
            left_prob = _float(row, f"prob_{pair[0]}")
            right_prob = _float(row, f"prob_{pair[1]}")
            probabilities = {pair[0]: left_prob, pair[1]: right_prob}
            prediction = pair[0] if left_prob >= right_prob else pair[1]
            confidence = float(max(left_prob, right_prob))
            true_probability = float(probabilities[int(target)])
            if int(prediction) == int(target) or confidence < threshold:
                continue
            by_target_prediction[f"{int(target)}->{int(prediction)}"] += 1
            selected.append(
                {
                    "sample_index": int(sample_index),
                    "image_path": str(row.get("path", "") or row.get("image_path", "") or ""),
                    "target_index": int(target),
                    "prediction_index": int(prediction),
                    "sample_weight": f"{weight:.10g}",
                    "reason": "patch_oof_confident_disagreement",
                    "teacher_confidence": f"{confidence:.10g}",
                    "true_probability": f"{true_probability:.10g}",
                }
            )

    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "prediction_index",
        "sample_weight",
        "reason",
        "teacher_confidence",
        "true_probability",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "teacher_csv": str(Path(teacher_csv).resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "total_rows": int(total_rows),
        "pair": f"{pair[0]}-{pair[1]}",
        "pair_rows": int(pair_rows),
        "selected_rows": int(len(selected)),
        "disagreement_confidence_threshold": float(threshold),
        "disagreement_weight": float(weight),
        "by_target_prediction": dict(by_target_prediction),
        "leakage_guard": "reads train-only OOF teacher CSV; writes sample_index-keyed weights",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not selected:
        raise ValueError("No rows selected; lower threshold or check teacher CSV.")
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only sample weights from a patch-verifier OOF teacher CSV."
    )
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", type=str, default="0-1")
    parser.add_argument("--disagreement-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--disagreement-weight", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_manifest(
        teacher_csv=Path(args.teacher_csv),
        output_dir=Path(args.output_dir),
        pair=_parse_pair(str(args.pair)),
        disagreement_confidence_threshold=float(args.disagreement_confidence_threshold),
        disagreement_weight=float(args.disagreement_weight),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
