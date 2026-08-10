from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only sample-weight manifest for fine-grained boundary "
            "samples from a detailed prediction CSV."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--boundary-pairs",
        type=str,
        default="0-1,1-2,2-3,4-rest",
        help="Comma separated class pairs. Use N-rest for one-vs-rest.",
    )
    parser.add_argument("--close-margin-threshold", type=float, default=0.18)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--close-boundary-weight", type=float, default=1.30)
    parser.add_argument("--misclassification-weight", type=float, default=1.80)
    parser.add_argument("--high-confidence-error-weight", type=float, default=2.00)
    parser.add_argument("--max-weight", type=float, default=2.50)
    parser.add_argument("--top-k-images", type=int, default=40)
    parser.add_argument("--copy-images", action="store_true", default=False)
    parser.add_argument(
        "--allow-non-train-paths",
        action="store_true",
        default=False,
        help="Disable the default guard that all image paths must contain a train directory.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty predictions CSV: {path}")
    return rows


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "") or default))
    except ValueError:
        return float(default)


def _first_non_empty(row: Mapping[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _path(row: Mapping[str, str]) -> str:
    return _first_non_empty(row, ("image_path", "path", "sample_path"))


def _index_from_row(
    row: Mapping[str, str],
    *,
    index_keys: Sequence[str],
    name_keys: Sequence[str],
    class_to_index: Mapping[str, int],
) -> int:
    name = _first_non_empty(row, name_keys)
    if name and name in class_to_index:
        return int(class_to_index[name])
    for key in index_keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(float(value))
    return 0


def _probability_key(row: Mapping[str, str], class_index: int) -> str | None:
    exact = f"prob_{int(class_index)}"
    if exact in row:
        return exact
    prefix = f"prob_{int(class_index)}_"
    for key in row.keys():
        if key.startswith(prefix):
            return key
    return None


def _top2_from_probabilities(row: Mapping[str, str], num_classes: int) -> Tuple[int, float, int, float]:
    probabilities = []
    for class_index in range(num_classes):
        key = _probability_key(row, class_index)
        probabilities.append(_float(row, key, 0.0) if key is not None else 0.0)
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    if not ranked:
        return 0, 0.0, 0, 0.0
    top1_index, top1_probability = ranked[0]
    top2_index, top2_probability = ranked[1] if len(ranked) > 1 else (top1_index, 0.0)
    return int(top1_index), float(top1_probability), int(top2_index), float(top2_probability)


def _parse_boundary_pairs(text: str, num_classes: int) -> List[Tuple[int, int | str]]:
    pairs: List[Tuple[int, int | str]] = []
    for item in str(text or "").split(","):
        value = item.strip().lower()
        if not value or "-" not in value:
            continue
        left, right = [part.strip() for part in value.split("-", 1)]
        if not left:
            continue
        left_index = int(left)
        if right == "rest":
            pairs.append((left_index, "rest"))
        else:
            right_index = int(right)
            if 0 <= left_index < num_classes and 0 <= right_index < num_classes:
                pairs.append((left_index, right_index))
    return pairs


def _pair_matches(a: int, b: int, pair: Tuple[int, int | str]) -> bool:
    left, right = pair
    if right == "rest":
        return int(a) == int(left) or int(b) == int(left)
    return {int(a), int(b)} == {int(left), int(right)}


def _matching_pair(a: int, b: int, pairs: Sequence[Tuple[int, int | str]]) -> str:
    for left, right in pairs:
        if _pair_matches(a, b, (left, right)):
            return f"{left}-{right}"
    return ""


def _is_train_path(path_text: str) -> bool:
    parts = [part.lower() for part in Path(path_text).parts]
    return "train" in parts


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _copy_review_images(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    group_name: str,
    limit: int,
) -> int:
    target_dir = output_dir / "review_images" / group_name
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for index, row in enumerate(rows, start=1):
        if copied >= int(limit):
            break
        source = Path(str(row.get("image_path", "") or ""))
        if not source.is_file():
            continue
        target = target_dir / (
            f"{index:03d}_t{row.get('target_index')}_p{row.get('prediction_index')}_"
            f"w{float(row.get('sample_weight', 1.0)):.2f}_"
            f"margin{float(row.get('top2_margin', 0.0)):.3f}{source.suffix.lower()}"
        )
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.predictions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_index: Dict[str, int] = {}
    class_names: Dict[int, str] = {}
    if args.data is not None:
        data_spec = load_data_spec(args.data)
        class_names = {index: str(name) for index, name in enumerate(data_spec.class_names)}
        class_to_index = {str(name): index for index, name in enumerate(data_spec.class_names)}
        num_classes = len(data_spec.class_names)
    else:
        num_classes = max(
            1,
            len([key for key in rows[0].keys() if key.startswith("prob_")]),
        )
    pairs = _parse_boundary_pairs(args.boundary_pairs, num_classes=num_classes)
    if not pairs:
        raise ValueError("--boundary-pairs khong co pair hop le.")

    selected_by_path: Dict[str, Dict[str, object]] = {}
    reason_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    skipped_non_train = 0
    for row in rows:
        path_text = _path(row)
        if not path_text:
            continue
        if not args.allow_non_train_paths and not _is_train_path(path_text):
            skipped_non_train += 1
            continue
        target = _index_from_row(
            row,
            index_keys=("target_index", "y_true"),
            name_keys=("target_name", "true_name"),
            class_to_index=class_to_index,
        )
        prediction = _index_from_row(
            row,
            index_keys=("prediction_index", "y_pred", "teacher_pred_index"),
            name_keys=("prediction_name", "pred_name", "teacher_pred_name"),
            class_to_index=class_to_index,
        )
        top1, top1_probability, top2, top2_probability = _top2_from_probabilities(row, num_classes)
        if top1_probability <= 0.0:
            top1_probability = _float(row, "confidence", _float(row, "teacher_confidence", 0.0))
        if "top2_probability" in row:
            top2_probability = _float(row, "top2_probability", top2_probability)
        if "top2_index" in row:
            top2 = int(float(str(row.get("top2_index") or top2)))
        margin = top1_probability - top2_probability
        pair_name = _matching_pair(target, prediction, pairs)
        top2_pair_name = _matching_pair(target, top2, pairs)
        correct = int(target == prediction)

        reasons: List[str] = []
        weight = 1.0
        if not correct and pair_name:
            reasons.append("pair_misclassification")
            weight = max(weight, float(args.misclassification_weight))
            if top1_probability >= float(args.high_confidence_threshold):
                reasons.append("high_confidence_pair_error")
                weight = max(weight, float(args.high_confidence_error_weight))
        if top2_pair_name and margin <= float(args.close_margin_threshold):
            reasons.append("close_boundary")
            weight = max(weight, float(args.close_boundary_weight))
            if not pair_name:
                pair_name = top2_pair_name
        if not reasons:
            continue

        weight = min(float(args.max_weight), max(1.0, float(weight)))
        key = str(Path(path_text).resolve()).lower()
        existing = selected_by_path.get(key)
        reason_text = "+".join(sorted(set(reasons)))
        record = {
            "image_path": str(Path(path_text).resolve()),
            "target_index": int(target),
            "target_name": _first_non_empty(row, ("target_name", "true_name")) or class_names.get(target, ""),
            "prediction_index": int(prediction),
            "prediction_name": _first_non_empty(row, ("prediction_name", "pred_name", "teacher_pred_name"))
            or class_names.get(prediction, ""),
            "top2_index": int(top2),
            "top2_name": str(row.get("top2_name", "") or class_names.get(top2, "")),
            "confidence": float(top1_probability),
            "top2_probability": float(top2_probability),
            "top2_margin": float(margin),
            "boundary_pair": pair_name,
            "sample_weight": float(weight),
            "reason": reason_text,
        }
        if existing is None or float(record["sample_weight"]) > float(existing["sample_weight"]):
            selected_by_path[key] = record

    selected = list(selected_by_path.values())
    selected.sort(
        key=lambda item: (
            -float(item["sample_weight"]),
            float(item["top2_margin"]),
            str(item["image_path"]),
        )
    )
    for item in selected:
        reason_counts[str(item["reason"])] += 1
        pair_counts[str(item["boundary_pair"])] += 1

    if not selected:
        raise ValueError("Khong tim thay boundary sample hop le tu prediction CSV.")

    fieldnames = [
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "top2_index",
        "top2_name",
        "confidence",
        "top2_probability",
        "top2_margin",
        "boundary_pair",
        "sample_weight",
        "reason",
    ]
    manifest_path = output_dir / "sample_weights_train_only.csv"
    _write_csv(manifest_path, selected, fieldnames)
    _write_csv(output_dir / "hard_samples_train_only.csv", selected, fieldnames)

    copied: Dict[str, int] = {}
    if args.copy_images:
        by_reason: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        for item in selected:
            by_reason[str(item["reason"])].append(item)
        for reason, reason_rows in by_reason.items():
            copied[reason] = _copy_review_images(
                reason_rows,
                output_dir,
                reason.replace("+", "_"),
                int(args.top_k_images),
            )

    summary = {
        "predictions": str(args.predictions),
        "manifest": str(manifest_path),
        "samples": int(len(selected)),
        "boundary_pairs": [f"{left}-{right}" for left, right in pairs],
        "close_margin_threshold": float(args.close_margin_threshold),
        "high_confidence_threshold": float(args.high_confidence_threshold),
        "weights": {
            "close_boundary": float(args.close_boundary_weight),
            "misclassification": float(args.misclassification_weight),
            "high_confidence_error": float(args.high_confidence_error_weight),
            "max_weight": float(args.max_weight),
        },
        "by_reason": dict(reason_counts),
        "by_boundary_pair": dict(pair_counts),
        "skipped_non_train_rows": int(skipped_non_train),
        "copied_images": copied,
        "leakage_guard": "requires train path unless --allow-non-train-paths is set",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
