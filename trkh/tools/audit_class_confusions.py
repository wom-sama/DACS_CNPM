from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prediction CSV for focus-class and pairwise confusion review."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional data.yaml used to map true_name/pred_name CSVs to class indices.",
    )
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--top-k-images", type=int, default=30)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=False,
        help="Copy representative source images into review folders.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty predictions CSV: {path}")
    return rows


def _int(row: Mapping[str, str], key: str) -> int:
    return int(float(str(row.get(key, "0") or "0")))


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "") or default))
    except ValueError:
        return float(default)


def _path(row: Mapping[str, str]) -> str:
    for key in ("image_path", "path"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _first_non_empty(row: Mapping[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _index_from_row(
    row: Mapping[str, str],
    *,
    index_keys: Sequence[str],
    name_keys: Sequence[str],
    class_to_index: Mapping[str, int],
    require_known_name: bool = False,
) -> int:
    name = _first_non_empty(row, name_keys)
    if name and name in class_to_index:
        return int(class_to_index[name])
    if name and require_known_name:
        raise ValueError(f"Unknown class name in prediction row: {name!r}")
    for key in index_keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(float(value))
    raise ValueError(
        "Prediction row is missing both a recognized class name and numeric class index."
    )


def _margin(row: Mapping[str, str]) -> float:
    if "top1_probability" in row and "top2_probability" in row:
        return _float(row, "top1_probability") - _float(row, "top2_probability")
    return 0.0


def _prob_key(
    row: Mapping[str, str],
    class_index: int,
    class_name: str = "",
) -> str | None:
    normalized_name = str(class_name).strip()
    if normalized_name:
        semantic_matches = []
        for key in row.keys():
            if key == f"prob_{normalized_name}":
                semantic_matches.append(key)
                continue
            parts = key.split("_", 2)
            if len(parts) == 3 and parts[0] == "prob" and parts[1].isdigit():
                if parts[2] == normalized_name:
                    semantic_matches.append(key)
        if len(semantic_matches) > 1:
            raise ValueError(
                f"Multiple probability columns match class {normalized_name!r}: "
                f"{semantic_matches}"
            )
        if semantic_matches:
            return semantic_matches[0]
        # Once an authoritative semantic class name is available, an index-only
        # fallback could silently read a probability from a different class axis.
        return None
    prefix = f"prob_{int(class_index)}_"
    for key in row.keys():
        if key.startswith(prefix):
            return key
    exact = f"prob_{int(class_index)}"
    return exact if exact in row else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
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
    copied = 0
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        if copied >= int(limit):
            break
        source = Path(str(row.get("image_path", "") or ""))
        if not source.is_file():
            continue
        target = group_dir / (
            f"{index:03d}_t{row.get('target_index')}_p{row.get('prediction_index')}_"
            f"conf{float(row.get('confidence', 0.0)):.3f}_margin{float(row.get('top2_margin', 0.0)):.3f}"
            f"{source.suffix.lower()}"
        )
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.predictions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    focus = int(args.focus_class_index)
    class_names: Dict[int, str] = {}
    class_to_index: Dict[str, int] = {}
    if args.data is not None:
        data_spec = load_data_spec(args.data)
        class_names.update({index: str(name) for index, name in enumerate(data_spec.class_names)})
        class_to_index.update({str(name): index for index, name in enumerate(data_spec.class_names)})
    for row in rows:
        target_index = _index_from_row(
            row,
            index_keys=("target_index", "y_true"),
            name_keys=("target_name", "true_name"),
            class_to_index=class_to_index,
            require_known_name=args.data is not None,
        )
        prediction_index = _index_from_row(
            row,
            index_keys=("prediction_index", "y_pred", "teacher_pred_index"),
            name_keys=("prediction_name", "pred_name", "teacher_pred_name"),
            class_to_index=class_to_index,
            require_known_name=args.data is not None,
        )
        target_name = _first_non_empty(row, ("target_name", "true_name"))
        prediction_name = _first_non_empty(row, ("prediction_name", "pred_name", "teacher_pred_name"))
        if target_name:
            class_names[target_index] = target_name
            class_to_index.setdefault(target_name, target_index)
        if prediction_name:
            class_names[prediction_index] = prediction_name
            class_to_index.setdefault(prediction_name, prediction_index)
    focus_prob_key = _prob_key(rows[0], focus, class_names.get(focus, ""))

    enriched: List[Dict[str, object]] = []
    confusion_counter: Counter[tuple[int, int]] = Counter()
    for row in rows:
        target = _index_from_row(
            row,
            index_keys=("target_index", "y_true"),
            name_keys=("target_name", "true_name"),
            class_to_index=class_to_index,
            require_known_name=args.data is not None,
        )
        prediction = _index_from_row(
            row,
            index_keys=("prediction_index", "y_pred", "teacher_pred_index"),
            name_keys=("prediction_name", "pred_name", "teacher_pred_name"),
            class_to_index=class_to_index,
            require_known_name=args.data is not None,
        )
        correct = int(target == prediction)
        confusion_counter[(target, prediction)] += 1
        enriched.append(
            {
                "sample_index": _int(row, "sample_index"),
                "image_path": _path(row),
                "target_index": target,
                "target_name": _first_non_empty(row, ("target_name", "true_name")),
                "prediction_index": prediction,
                "prediction_name": _first_non_empty(row, ("prediction_name", "pred_name", "teacher_pred_name")),
                "correct": correct,
                "confidence": _float(row, "confidence", _float(row, "teacher_confidence")),
                "top2_margin": _margin(row),
                "focus_probability": _float(row, focus_prob_key) if focus_prob_key else "",
                "top1_name": _first_non_empty(row, ("top1_name", "pred_name", "teacher_pred_name")),
                "top2_name": row.get("top2_name", ""),
                "top2_probability": _float(row, "top2_probability"),
            }
        )

    focus_tp = [row for row in enriched if row["target_index"] == focus and row["prediction_index"] == focus]
    focus_fn = [row for row in enriched if row["target_index"] == focus and row["prediction_index"] != focus]
    focus_fp = [row for row in enriched if row["target_index"] != focus and row["prediction_index"] == focus]
    focus_related = [
        row for row in enriched if row["target_index"] == focus or row["prediction_index"] == focus
    ]
    high_conf_errors = sorted(
        [row for row in enriched if not row["correct"]],
        key=lambda item: (-float(item["confidence"]), float(item["top2_margin"])),
    )
    close_focus = sorted(focus_related, key=lambda item: float(item["top2_margin"]))

    confusion_rows = []
    for (target, prediction), count in sorted(confusion_counter.items(), key=lambda item: (-item[1], item[0])):
        if target == prediction:
            continue
        confusion_rows.append(
            {
                "target_index": target,
                "target_name": class_names.get(target, ""),
                "prediction_index": prediction,
                "prediction_name": class_names.get(prediction, ""),
                "count": int(count),
            }
        )

    fieldnames = [
        "sample_index",
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "correct",
        "confidence",
        "top2_margin",
        "focus_probability",
        "top1_name",
        "top2_name",
        "top2_probability",
    ]
    _write_csv(output_dir / "focus_class_false_negative.csv", focus_fn, fieldnames)
    _write_csv(output_dir / "focus_class_false_positive.csv", focus_fp, fieldnames)
    _write_csv(output_dir / "focus_class_true_positive.csv", focus_tp, fieldnames)
    _write_csv(output_dir / "focus_class_close_margin.csv", close_focus[: max(1, int(args.top_k_images) * 2)], fieldnames)
    _write_csv(output_dir / "high_confidence_errors.csv", high_conf_errors[: max(1, int(args.top_k_images) * 2)], fieldnames)
    _write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_rows,
        ["target_index", "target_name", "prediction_index", "prediction_name", "count"],
    )

    copied = {}
    if args.copy_images:
        copied["false_negative"] = _copy_review_images(
            sorted(focus_fn, key=lambda item: -float(item["confidence"])),
            output_dir / "review_images",
            "focus_false_negative",
            int(args.top_k_images),
        )
        copied["false_positive"] = _copy_review_images(
            sorted(focus_fp, key=lambda item: -float(item["confidence"])),
            output_dir / "review_images",
            "focus_false_positive",
            int(args.top_k_images),
        )
        copied["close_margin"] = _copy_review_images(
            close_focus,
            output_dir / "review_images",
            "focus_close_margin",
            int(args.top_k_images),
        )
        copied["high_confidence_errors"] = _copy_review_images(
            high_conf_errors,
            output_dir / "review_images",
            "high_confidence_errors",
            int(args.top_k_images),
        )

    by_target_prediction = defaultdict(int)
    for row in enriched:
        if not row["correct"]:
            by_target_prediction[f"{row['target_index']}->{row['prediction_index']}"] += 1

    precision = len(focus_tp) / max(1, len(focus_tp) + len(focus_fp))
    recall = len(focus_tp) / max(1, len(focus_tp) + len(focus_fn))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    summary = {
        "predictions": str(args.predictions),
        "samples": len(rows),
        "focus_class_index": focus,
        "focus_class_name": class_names.get(focus, ""),
        "focus_probability_column": focus_prob_key,
        "focus_metrics": {
            "tp": len(focus_tp),
            "fp": len(focus_fp),
            "fn": len(focus_fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "top_confusions": confusion_rows[:20],
        "focus_false_negative_by_prediction": dict(Counter(str(row["prediction_index"]) for row in focus_fn)),
        "focus_false_positive_by_target": dict(Counter(str(row["target_index"]) for row in focus_fp)),
        "error_counts": dict(sorted(by_target_prediction.items())),
        "copied_images": copied,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme = [
        "# Class Confusion Audit",
        "",
        f"Predictions: `{args.predictions}`",
        f"Focus class: `{focus}` - `{class_names.get(focus, '')}`",
        "",
        "## Focus Metrics",
        "",
        f"- TP: `{len(focus_tp)}`",
        f"- FP: `{len(focus_fp)}`",
        f"- FN: `{len(focus_fn)}`",
        f"- Precision: `{precision:.6f}`",
        f"- Recall: `{recall:.6f}`",
        f"- F1: `{f1:.6f}`",
        "",
        "## Files",
        "",
        "- `focus_class_false_negative.csv`: class 1 bi du doan thanh class khac.",
        "- `focus_class_false_positive.csv`: class khac bi du doan thanh class 1.",
        "- `focus_class_close_margin.csv`: mau lien quan class 1 co top-2 gan nhau.",
        "- `high_confidence_errors.csv`: loi confidence cao can xem label/boundary.",
        "- `confusion_pairs.csv`: tong hop nham lan theo cap.",
        "- `review_images/`: anh copy de xem nhanh neu bat `--copy-images`.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "focus_f1": f1, "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
