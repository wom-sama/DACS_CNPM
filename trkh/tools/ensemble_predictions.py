from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine classification prediction CSV files by deterministic majority/weighted vote. "
            "Use validation-selected members for honest model selection; do not tune members on test."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Prediction CSV path with a stable model name. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional comma list NAME:WEIGHT. Missing names use weight 1.0.",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=1,
        help="Minimum number of model predictions required for a sample to be included.",
    )
    parser.add_argument(
        "--tie-break",
        choices=("input_order", "class_prior"),
        default="input_order",
        help="Tie break strategy. input_order is deterministic and should be configured before test.",
    )
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--input must have NAME=CSV format, got: {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--input has empty model name: {value!r}")
    path = Path(path_text.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _parse_weights(value: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Weight item must be NAME:WEIGHT, got: {item!r}")
        name, weight_text = item.split(":", 1)
        weight = float(weight_text)
        if weight <= 0.0:
            raise ValueError(f"Weight must be > 0 for {name!r}, got {weight}")
        weights[name.strip()] = float(weight)
    return weights


def _read_prediction_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"path"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing columns {sorted(missing)}: {path}")
    return rows


def _truth_name(row: Mapping[str, str]) -> str:
    value = str(row.get("true_name", "") or "").strip()
    if value:
        return value
    value = str(row.get("target_name", "") or "").strip()
    if value:
        return value
    return str(row.get("y_true", "") or row.get("target_index", "")).strip()


def _prediction_name(row: Mapping[str, str]) -> str:
    value = str(row.get("pred_name", "") or "").strip()
    if value:
        return value
    value = str(row.get("prediction_name", "") or "").strip()
    if value:
        return value
    return str(row.get("y_pred", "") or row.get("prediction_index", "")).strip()


def _confidence(row: Mapping[str, str]) -> str:
    for key in ("confidence", "prediction_probability", "probability"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _metrics(
    records: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
) -> Dict[str, object]:
    class_names = list(class_names)
    matrix: Dict[str, Counter[str]] = {name: Counter() for name in class_names}
    for record in records:
        target = str(record["true_name"])
        prediction = str(record["ensemble_pred_name"])
        if target not in matrix:
            matrix[target] = Counter()
            class_names.append(target)
        matrix[target][prediction] += 1

    per_class = []
    total = 0
    correct = 0
    for class_name in class_names:
        true_positive = int(matrix[class_name][class_name])
        support = int(sum(matrix[class_name].values()))
        predicted_support = int(sum(matrix[target][class_name] for target in class_names))
        precision = float(true_positive / predicted_support) if predicted_support else 0.0
        recall = float(true_positive / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        correct += true_positive
        total += support
        per_class.append(
            {
                "class_name": class_name,
                "support": support,
                "predicted_support": predicted_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    confusion_matrix = [
        [int(matrix[target][prediction]) for prediction in class_names]
        for target in class_names
    ]
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_precision": float(sum(item["precision"] for item in per_class) / len(per_class))
        if per_class
        else 0.0,
        "macro_recall": float(sum(item["recall"] for item in per_class) / len(per_class))
        if per_class
        else 0.0,
        "macro_f1": float(sum(item["f1"] for item in per_class) / len(per_class)) if per_class else 0.0,
        "weighted_f1": float(
            sum(item["f1"] * item["support"] for item in per_class) / max(1, total)
        ),
        "classes": class_names,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def _vote(
    path: str,
    names: Sequence[str],
    rows_by_name: Mapping[str, Mapping[str, Mapping[str, str]]],
    weights: Mapping[str, float],
    tie_break: str,
    class_prior: Mapping[str, int],
) -> Tuple[str, Dict[str, str], Dict[str, float]]:
    weighted_votes: Dict[str, float] = defaultdict(float)
    raw_votes: Dict[str, str] = {}
    for name in names:
        row = rows_by_name[name].get(path)
        if row is None:
            continue
        prediction = _prediction_name(row)
        raw_votes[name] = prediction
        weighted_votes[prediction] += float(weights.get(name, 1.0))

    if not weighted_votes:
        raise ValueError(f"No votes available for path: {path}")

    max_vote = max(weighted_votes.values())
    candidates = {label for label, score in weighted_votes.items() if score == max_vote}
    if len(candidates) == 1:
        return next(iter(candidates)), raw_votes, dict(weighted_votes)

    if tie_break == "class_prior":
        best_prior = max(int(class_prior.get(label, 0)) for label in candidates)
        prior_candidates = {label for label in candidates if int(class_prior.get(label, 0)) == best_prior}
        if len(prior_candidates) == 1:
            return next(iter(prior_candidates)), raw_votes, dict(weighted_votes)
        candidates = prior_candidates

    for name in names:
        prediction = raw_votes.get(name)
        if prediction in candidates:
            return prediction, raw_votes, dict(weighted_votes)
    return sorted(candidates)[0], raw_votes, dict(weighted_votes)


def main() -> None:
    args = parse_args()
    named_paths = [_parse_named_path(value) for value in args.input]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names are not allowed: {names}")
    if int(args.min_models) < 1:
        raise ValueError("--min-models must be >= 1.")
    weights = _parse_weights(args.weights)

    rows_by_name: Dict[str, Dict[str, Dict[str, str]]] = {}
    for name, path in named_paths:
        rows = _read_prediction_csv(path)
        rows_by_name[name] = {str(row["path"]): row for row in rows}

    path_counts = Counter()
    for rows in rows_by_name.values():
        path_counts.update(rows.keys())
    selected_paths = sorted(path for path, count in path_counts.items() if count >= int(args.min_models))
    if not selected_paths:
        raise ValueError("No common/eligible prediction paths after applying --min-models.")

    reference_name = names[0]
    truth_by_path: Dict[str, str] = {}
    for path in selected_paths:
        truths = [
            _truth_name(rows_by_name[name][path])
            for name in names
            if path in rows_by_name[name]
        ]
        truth_counts = Counter(truths)
        if len(truth_counts) != 1:
            raise ValueError(f"Truth label mismatch for {path}: {truth_counts}")
        truth_by_path[path] = truths[0]
    class_names = sorted(set(truth_by_path.values()))
    class_prior = Counter(truth_by_path.values())

    records: List[Dict[str, object]] = []
    for path in selected_paths:
        present_names = [name for name in names if path in rows_by_name[name]]
        prediction, raw_votes, weighted_votes = _vote(
            path=path,
            names=present_names,
            rows_by_name=rows_by_name,
            weights=weights,
            tie_break=str(args.tie_break),
            class_prior=class_prior,
        )
        records.append(
            {
                "path": path,
                "true_name": truth_by_path[path],
                "ensemble_pred_name": prediction,
                "correct": int(prediction == truth_by_path[path]),
                "model_count": len(present_names),
                "votes_json": json.dumps(raw_votes, ensure_ascii=False, sort_keys=True),
                "weighted_votes_json": json.dumps(weighted_votes, ensure_ascii=False, sort_keys=True),
                **{
                    f"{name}_pred": raw_votes.get(name, "")
                    for name in names
                },
                **{
                    f"{name}_confidence": _confidence(rows_by_name[name][path])
                    if path in rows_by_name[name]
                    else ""
                    for name in names
                },
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(records, class_names)
    payload = {
        "mode": "weighted_majority_vote" if weights else "majority_vote",
        "tie_break": str(args.tie_break),
        "sample_count": len(records),
        "members": [
            {
                "name": name,
                "path": str(path),
                "weight": float(weights.get(name, 1.0)),
                "rows": len(rows_by_name[name]),
            }
            for name, path in named_paths
        ],
        "selection_warning": (
            "For honest reporting, choose members/weights using train/val only. "
            "Do not tune ensemble membership or weights on test metrics."
        ),
        **metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    fieldnames = list(records[0].keys()) if records else []
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *metrics["classes"]])
        for class_name, row in zip(metrics["classes"], metrics["confusion_matrix"]):
            writer.writerow([class_name, *row])

    print(
        {
            "output_dir": str(output_dir),
            "samples": len(records),
            "accuracy": payload["accuracy"],
            "macro_f1": payload["macro_f1"],
            "members": names,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
