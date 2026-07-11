from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from trkh.core.config import load_data_spec


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank train-only ambiguous/label-issue candidates from class probability CSVs "
            "using cleanlab plus simple confidence diagnostics."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--copy-review-images", type=int, default=120)
    parser.add_argument("--allow-non-train", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    for row in rows:
        if "path" not in row and "image_path" in row:
            row["path"] = str(row.get("image_path", "") or "")
        if "true_name" not in row and "target_name" in row:
            row["true_name"] = str(row.get("target_name", "") or "")
        if "true_index" not in row and "target_index" in row:
            row["true_index"] = str(row.get("target_index", "") or "")
    required = {"path"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing columns {sorted(missing)}: {path}")
    if "true_name" not in rows[0] and "true_index" not in rows[0] and "target_index" not in rows[0]:
        raise ValueError(f"Prediction CSV missing true_name/target_index columns: {path}")
    return rows


def _probability_columns(row: Mapping[str, str], num_classes: int) -> List[str]:
    indexed: Dict[int, str] = {}
    for column in row.keys():
        match = PROBABILITY_COLUMN.match(str(column))
        if match is None:
            continue
        class_index = int(match.group(1))
        if 0 <= class_index < int(num_classes):
            if class_index in indexed:
                raise ValueError(f"Duplicate probability column for class {class_index}.")
            indexed[class_index] = str(column)
    missing = [index for index in range(num_classes) if index not in indexed]
    if missing:
        raise ValueError(f"Prediction row missing probability columns for classes {missing}.")
    return [indexed[index] for index in range(num_classes)]


def _probabilities(
    row: Mapping[str, str],
    num_classes: int,
    probability_columns: Sequence[str],
) -> List[float]:
    values = []
    if len(probability_columns) != int(num_classes):
        raise ValueError("probability_columns length must match num_classes.")
    for key in probability_columns:
        if key not in row:
            raise ValueError(f"Prediction row missing {key}.")
        values.append(max(0.0, float(row[key])))
    total = sum(values)
    if total <= 0.0:
        return [1.0 / float(num_classes) for _ in range(num_classes)]
    return [value / total for value in values]


def _entropy(probabilities: Sequence[float]) -> float:
    return float(-sum(value * math.log(max(value, 1e-12)) for value in probabilities))


def _guard_train_only(rows: Sequence[Mapping[str, str]], *, split: str, allow_non_train: bool) -> Dict[str, object]:
    split_token = f"{Path(split).as_posix()}/"
    bad_paths = []
    for row in rows:
        normalized = str(row["path"]).replace("\\", "/")
        if split_token not in normalized:
            bad_paths.append(str(row["path"]))
            if len(bad_paths) >= 20:
                break
    summary = {
        "split": split,
        "checked_rows": len(rows),
        "non_matching_path_count_sampled": len(bad_paths),
        "sample_non_matching_paths": bad_paths,
    }
    if bad_paths and not allow_non_train:
        raise ValueError(
            "Prediction CSV appears to contain non-train paths. "
            "Pass --allow-non-train only for diagnostic audits, never for train-signal manifests. "
            f"Sample: {bad_paths[:3]}"
        )
    return summary


def _copy_review_images(
    *,
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
    class_names: Sequence[str],
    limit: int,
) -> None:
    if limit <= 0:
        return
    review_dir = output_dir / "review_images"
    review_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for record in records:
        if copied >= limit:
            break
        source = Path(str(record["path"]))
        if not source.is_file():
            continue
        true_index = int(record["true_index"])
        suggested_index = int(record["suggested_index"])
        bucket = review_dir / f"{true_index}_{class_names[true_index]}__to__{suggested_index}_{class_names[suggested_index]}"
        bucket.mkdir(parents=True, exist_ok=True)
        rank = int(record["issue_rank"])
        destination = bucket / f"{rank:04d}_{source.name}"
        if not destination.exists():
            shutil.copy2(source, destination)
        copied += 1


def main() -> None:
    args = parse_args()
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "audit_label_issues_cleanlab requires cleanlab. Install with: python -m pip install cleanlab"
        ) from exc

    data_spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(value) for value in data_spec.class_names]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    rows = _read_rows(args.prediction_csv)
    probability_columns = _probability_columns(rows[0], len(class_names))
    guard_summary = _guard_train_only(rows, split=str(args.split), allow_non_train=bool(args.allow_non_train))

    labels: List[int] = []
    probabilities: List[List[float]] = []
    selected_rows: List[Dict[str, str]] = []
    skipped_unknown = 0
    for row in rows:
        true_name = str(row.get("true_name", "") or "").strip()
        if true_name in class_to_index:
            true_index = int(class_to_index[true_name])
        else:
            true_index = -1
            for key in ("true_index", "target_index", "label"):
                value = str(row.get(key, "") or "").strip()
                if not value:
                    continue
                try:
                    true_index = int(float(value))
                except ValueError:
                    true_index = -1
                break
            if 0 <= true_index < len(class_names):
                true_name = class_names[int(true_index)]
                row["true_name"] = true_name
            else:
                skipped_unknown += 1
                continue
        if not (0 <= true_index < len(class_names)):
            skipped_unknown += 1
            continue
        labels.append(int(true_index))
        probabilities.append(_probabilities(row, len(class_names), probability_columns))
        selected_rows.append(row)
    if not selected_rows:
        raise ValueError("No rows matched data.yaml class names.")

    label_array = np.asarray(labels, dtype=np.int64)
    prob_array = np.asarray(probabilities, dtype=np.float64)
    quality_scores = get_label_quality_scores(label_array, prob_array, method="self_confidence")
    issue_indices = find_label_issues(
        labels=label_array,
        pred_probs=prob_array,
        return_indices_ranked_by="self_confidence",
        filter_by="both",
    )
    issue_rank_by_index = {int(index): rank + 1 for rank, index in enumerate(issue_indices.tolist())}

    records: List[Dict[str, object]] = []
    for row_index, (row, true_index, probs) in enumerate(zip(selected_rows, labels, probabilities)):
        pred_index = int(np.argmax(probs))
        sorted_probs = sorted((float(value) for value in probs), reverse=True)
        self_confidence = float(probs[int(true_index)])
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
        normalized_entropy = _entropy(probs) / math.log(float(len(class_names)))
        records.append(
            {
                "sample_index": str(row.get("sample_index", "") or row.get("dataset_index", "") or ""),
                "path": str(row["path"]),
                "true_index": int(true_index),
                "true_name": class_names[int(true_index)],
                "pred_index": pred_index,
                "pred_name": class_names[pred_index],
                "suggested_index": pred_index,
                "suggested_name": class_names[pred_index],
                "is_label_issue": int(row_index in issue_rank_by_index),
                "issue_rank": int(issue_rank_by_index.get(row_index, len(selected_rows) + row_index + 1)),
                "label_quality": float(quality_scores[row_index]),
                "self_confidence": self_confidence,
                "top1_confidence": float(sorted_probs[0]),
                "top2_margin": margin,
                "normalized_entropy": float(normalized_entropy),
                "given_label_matches_prediction": int(pred_index == int(true_index)),
                **{f"prob_{index}": float(probs[index]) for index in range(len(class_names))},
            }
        )
    records.sort(
        key=lambda item: (
            1 - int(item["is_label_issue"]),
            float(item["label_quality"]),
            float(item["top2_margin"]),
            -float(item["normalized_entropy"]),
            int(item["issue_rank"]),
        )
    )

    top_n = max(1, int(args.top_n))
    output_records = records[:top_n]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "cleanlab_label_issue_manifest.csv"
    fieldnames = list(output_records[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_records)

    issue_counter: Counter[str] = Counter()
    issue_by_true: Counter[str] = Counter()
    issue_by_suggested: Counter[str] = Counter()
    for record in records:
        if int(record["is_label_issue"]) != 1:
            continue
        issue_counter[f"{record['true_name']}->{record['suggested_name']}"] += 1
        issue_by_true[str(record["true_name"])] += 1
        issue_by_suggested[str(record["suggested_name"])] += 1
    low_self_by_true: Dict[str, int] = defaultdict(int)
    for record in records:
        if float(record["self_confidence"]) < 0.45:
            low_self_by_true[str(record["true_name"])] += 1

    summary = {
        "data": str(args.data),
        "prediction_csv": str(args.prediction_csv),
        "output_csv": str(output_csv),
        "classes": class_names,
        "rows_read": len(rows),
        "rows_used": len(selected_rows),
        "skipped_unknown_class": int(skipped_unknown),
        "guard": guard_summary,
        "cleanlab_issue_count": int(len(issue_indices)),
        "top_n_written": int(len(output_records)),
        "issue_pairs": dict(issue_counter.most_common()),
        "issues_by_true_class": dict(issue_by_true.most_common()),
        "issues_by_suggested_class": dict(issue_by_suggested.most_common()),
        "low_self_confidence_lt_0p45_by_true_class": dict(sorted(low_self_by_true.items())),
        "note": (
            "Use this as a review/ambiguity manifest. Do not auto-relabel from model predictions; "
            "in-sample train probabilities can miss memorized label errors."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _copy_review_images(
        records=output_records,
        output_dir=args.output_dir,
        class_names=class_names,
        limit=int(args.copy_review_images),
    )
    print(
        {
            "summary": str(args.output_dir / "summary.json"),
            "manifest": str(output_csv),
            "rows_used": len(selected_rows),
            "cleanlab_issue_count": int(len(issue_indices)),
            "top_issue_pairs": dict(issue_counter.most_common(5)),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
