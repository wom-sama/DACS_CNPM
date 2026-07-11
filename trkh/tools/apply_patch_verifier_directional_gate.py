from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Pair = Tuple[int, int]


@dataclass(frozen=True)
class GateRow:
    split: str
    sample_index: int
    image_path: str
    target_index: int
    base_prediction: int
    candidate_prediction: int
    verifier_confidence: float
    pair_min_probability: float
    pair_margin: float
    probabilities: Tuple[float, ...]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select direction-specific patch-verifier gates from train OOF "
            "predictions and apply them to evaluation prediction CSVs."
        )
    )
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--eval-predictions", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", type=str, default="0-1")
    parser.add_argument("--directions", type=str, default="1->0,0->1")
    parser.add_argument("--min-pair-probability", type=float, default=0.02)
    parser.add_argument("--max-pair-margin", type=float, default=0.40)
    parser.add_argument("--min-correction-precision", type=float, default=0.60)
    parser.add_argument("--min-net-corrections", type=int, default=1)
    parser.add_argument("--min-corrections", type=int, default=1)
    return parser.parse_args(argv)


def _parse_pair(value: str) -> Pair:
    parts = [part.strip() for part in str(value).split("-")]
    if len(parts) != 2:
        raise ValueError(f"pair must look like '0-1', got {value!r}")
    left, right = int(parts[0]), int(parts[1])
    if left == right:
        raise ValueError("pair classes must be distinct")
    return left, right


def _parse_directions(value: str) -> List[str]:
    directions = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"direction must look like '1->0', got {item!r}")
        left, right = item.split("->", 1)
        directions.append(f"{int(left)}->{int(right)}")
    return directions


def _as_float(row: Mapping[str, str], key: str) -> float:
    value = str(row.get(key, "") or "").strip()
    if value == "":
        raise ValueError(f"missing required numeric column {key!r}")
    return float(value)


def _as_int(row: Mapping[str, str], key: str) -> int:
    value = str(row.get(key, "") or "").strip()
    if value == "":
        raise ValueError(f"missing required integer column {key!r}")
    return int(float(value))


def read_prediction_csv(path: Path, pair: Pair) -> List[GateRow]:
    left, right = pair
    prob_columns: List[str] = []
    rows: List[GateRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty prediction CSV: {path}")
        required = {
            "split",
            "sample_index",
            "image_path",
            "target_index",
            "base_prediction",
            f"prob_{left}",
            f"prob_{right}",
            f"verifier_{left}_{right}_prob_{left}",
            f"verifier_{left}_{right}_prob_{right}",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        prob_columns = sorted(
            [name for name in reader.fieldnames if name.startswith("prob_")],
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for raw in reader:
            probabilities = tuple(float(raw[name]) for name in prob_columns)
            verifier_left = _as_float(raw, f"verifier_{left}_{right}_prob_{left}")
            verifier_right = _as_float(raw, f"verifier_{left}_{right}_prob_{right}")
            if verifier_left >= verifier_right:
                candidate = left
                confidence = verifier_left
            else:
                candidate = right
                confidence = verifier_right
            pair_left_probability = _as_float(raw, f"prob_{left}")
            pair_right_probability = _as_float(raw, f"prob_{right}")
            rows.append(
                GateRow(
                    split=str(raw.get("split", "")),
                    sample_index=_as_int(raw, "sample_index"),
                    image_path=str(raw.get("image_path", "")),
                    target_index=_as_int(raw, "target_index"),
                    base_prediction=_as_int(raw, "base_prediction"),
                    candidate_prediction=int(candidate),
                    verifier_confidence=float(confidence),
                    pair_min_probability=float(min(pair_left_probability, pair_right_probability)),
                    pair_margin=float(abs(pair_left_probability - pair_right_probability)),
                    probabilities=probabilities,
                )
            )
    return rows


def _candidate_direction(row: GateRow, pair: Pair) -> Optional[str]:
    if row.base_prediction not in pair:
        return None
    if row.candidate_prediction == row.base_prediction:
        return None
    return f"{row.base_prediction}->{row.candidate_prediction}"


def _eligible_candidates(
    rows: Sequence[GateRow],
    *,
    pair: Pair,
    directions: Iterable[str],
    min_pair_probability: float,
    max_pair_margin: float,
) -> Dict[str, List[GateRow]]:
    direction_set = set(directions)
    result: Dict[str, List[GateRow]] = {direction: [] for direction in direction_set}
    for row in rows:
        direction = _candidate_direction(row, pair)
        if direction is None or direction not in direction_set:
            continue
        if row.pair_min_probability < float(min_pair_probability):
            continue
        if row.pair_margin > float(max_pair_margin):
            continue
        result.setdefault(direction, []).append(row)
    return result


def _change_stats(rows: Sequence[GateRow], threshold: float) -> Dict[str, float]:
    selected = [row for row in rows if row.verifier_confidence >= float(threshold)]
    corrections = 0
    harms = 0
    neutral = 0
    for row in selected:
        before_correct = row.base_prediction == row.target_index
        after_correct = row.candidate_prediction == row.target_index
        if (not before_correct) and after_correct:
            corrections += 1
        elif before_correct and (not after_correct):
            harms += 1
        else:
            neutral += 1
    precision_denominator = corrections + harms
    correction_precision = corrections / precision_denominator if precision_denominator else 0.0
    return {
        "selected": len(selected),
        "corrections": corrections,
        "harms": harms,
        "neutral": neutral,
        "net_corrections": corrections - harms,
        "correction_precision": correction_precision,
    }


def select_thresholds(
    rows: Sequence[GateRow],
    *,
    pair: Pair,
    directions: Sequence[str],
    min_pair_probability: float,
    max_pair_margin: float,
    min_correction_precision: float,
    min_net_corrections: int,
    min_corrections: int,
) -> Dict[str, Dict[str, object]]:
    candidates_by_direction = _eligible_candidates(
        rows,
        pair=pair,
        directions=directions,
        min_pair_probability=min_pair_probability,
        max_pair_margin=max_pair_margin,
    )
    selected: Dict[str, Dict[str, object]] = {}
    for direction in directions:
        direction_rows = candidates_by_direction.get(direction, [])
        thresholds = sorted({float(row.verifier_confidence) for row in direction_rows})
        best: Optional[Tuple[Tuple[float, ...], float, Dict[str, float]]] = None
        all_threshold_summaries = []
        for threshold in thresholds:
            stats = _change_stats(direction_rows, threshold)
            stats_with_threshold = {"threshold": float(threshold), **stats}
            all_threshold_summaries.append(stats_with_threshold)
            if int(stats["corrections"]) < int(min_corrections):
                continue
            if int(stats["net_corrections"]) < int(min_net_corrections):
                continue
            if float(stats["correction_precision"]) < float(min_correction_precision):
                continue
            key = (
                float(stats["net_corrections"]),
                float(stats["correction_precision"]),
                float(stats["corrections"]),
                -float(stats["harms"]),
                -float(threshold),
            )
            if best is None or key > best[0]:
                best = (key, float(threshold), stats)
        if best is None:
            selected[direction] = {
                "enabled": False,
                "candidate_count": len(direction_rows),
                "reason": "no threshold satisfied train-OOF risk constraints",
                "thresholds_checked": all_threshold_summaries,
            }
        else:
            _key, threshold, stats = best
            selected[direction] = {
                "enabled": True,
                "threshold": float(threshold),
                "candidate_count": len(direction_rows),
                "selection_stats": stats,
                "thresholds_checked": all_threshold_summaries,
            }
    return selected


def apply_thresholds(
    rows: Sequence[GateRow],
    *,
    pair: Pair,
    thresholds: Mapping[str, Mapping[str, object]],
    min_pair_probability: float,
    max_pair_margin: float,
) -> Tuple[List[int], List[Dict[str, object]]]:
    final_predictions: List[int] = []
    changes: List[Dict[str, object]] = []
    for row in rows:
        final = int(row.base_prediction)
        direction = _candidate_direction(row, pair)
        threshold_config = thresholds.get(direction or "", {})
        if (
            direction
            and bool(threshold_config.get("enabled", False))
            and row.pair_min_probability >= float(min_pair_probability)
            and row.pair_margin <= float(max_pair_margin)
            and row.verifier_confidence >= float(threshold_config["threshold"])
        ):
            final = int(row.candidate_prediction)
            before_correct = row.base_prediction == row.target_index
            after_correct = final == row.target_index
            if (not before_correct) and after_correct:
                change_type = "correction"
            elif before_correct and (not after_correct):
                change_type = "harm"
            else:
                change_type = "neutral"
            changes.append(
                {
                    "split": row.split,
                    "sample_index": row.sample_index,
                    "image_path": row.image_path,
                    "target_index": row.target_index,
                    "base_prediction": row.base_prediction,
                    "final_prediction": final,
                    "change_type": change_type,
                    "change_pair": f"{pair[0]}-{pair[1]}",
                    "direction": direction,
                    "verifier_confidence": row.verifier_confidence,
                    "pair_min_probability": row.pair_min_probability,
                    "pair_margin": row.pair_margin,
                }
            )
        final_predictions.append(final)
    return final_predictions, changes


def classification_metrics(
    rows: Sequence[GateRow],
    predictions: Sequence[int],
) -> Dict[str, object]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions length mismatch")
    max_class = 0
    for row, prediction in zip(rows, predictions):
        max_class = max(max_class, int(row.target_index), int(prediction), len(row.probabilities) - 1)
    num_classes = max_class + 1
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for row, prediction in zip(rows, predictions):
        confusion[int(row.target_index)][int(prediction)] += 1
    per_class = []
    f1_values = []
    correct = 0
    total = len(rows)
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[target][class_index] for target in range(num_classes) if target != class_index)
        fn = sum(confusion[class_index][pred] for pred in range(num_classes) if pred != class_index)
        support = sum(confusion[class_index])
        correct += tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": class_index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _change_summary(changes: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = {"changed": len(changes), "corrections": 0, "harms": 0, "neutral_changes": 0}
    for change in changes:
        change_type = str(change.get("change_type", ""))
        if change_type == "correction":
            counts["corrections"] += 1
        elif change_type == "harm":
            counts["harms"] += 1
        else:
            counts["neutral_changes"] += 1
    return counts


def _write_predictions(path: Path, rows: Sequence[GateRow], final_predictions: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_prob_cols = max((len(row.probabilities) for row in rows), default=0)
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "changed",
        "candidate_prediction",
        "verifier_confidence",
        "pair_min_probability",
        "pair_margin",
    ] + [f"prob_{index}" for index in range(max_prob_cols)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, final_prediction in zip(rows, final_predictions):
            payload = {
                "split": row.split,
                "sample_index": row.sample_index,
                "image_path": row.image_path,
                "target_index": row.target_index,
                "base_prediction": row.base_prediction,
                "final_prediction": int(final_prediction),
                "changed": int(int(final_prediction) != row.base_prediction),
                "candidate_prediction": row.candidate_prediction,
                "verifier_confidence": row.verifier_confidence,
                "pair_min_probability": row.pair_min_probability,
                "pair_margin": row.pair_margin,
            }
            for index, probability in enumerate(row.probabilities):
                payload[f"prob_{index}"] = probability
            writer.writerow(payload)


def _write_changes(path: Path, changes: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "change_type",
        "change_pair",
        "direction",
        "verifier_confidence",
        "pair_min_probability",
        "pair_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for change in changes:
            writer.writerow({field: change.get(field, "") for field in fieldnames})


def _evaluate_split(
    rows: Sequence[GateRow],
    *,
    pair: Pair,
    thresholds: Mapping[str, Mapping[str, object]],
    min_pair_probability: float,
    max_pair_margin: float,
    output_dir: Path,
) -> Dict[str, object]:
    base_predictions = [row.base_prediction for row in rows]
    final_predictions, changes = apply_thresholds(
        rows,
        pair=pair,
        thresholds=thresholds,
        min_pair_probability=min_pair_probability,
        max_pair_margin=max_pair_margin,
    )
    metrics = {
        "base": classification_metrics(rows, base_predictions),
        "directional_gated": classification_metrics(rows, final_predictions),
        "changes": _change_summary(changes),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(output_dir / "predictions.csv", rows, final_predictions)
    _write_changes(output_dir / "changed_cases_for_xai.csv", changes)
    return metrics


def run(args: argparse.Namespace) -> Dict[str, object]:
    pair = _parse_pair(args.pair)
    directions = _parse_directions(args.directions)
    train_rows = read_prediction_csv(args.train_predictions, pair)
    output_dir = Path(args.output_dir)
    thresholds = select_thresholds(
        train_rows,
        pair=pair,
        directions=directions,
        min_pair_probability=float(args.min_pair_probability),
        max_pair_margin=float(args.max_pair_margin),
        min_correction_precision=float(args.min_correction_precision),
        min_net_corrections=int(args.min_net_corrections),
        min_corrections=int(args.min_corrections),
    )

    splits: Dict[str, object] = {}
    splits["train"] = {
        "source": str(args.train_predictions),
        "metrics": _evaluate_split(
            train_rows,
            pair=pair,
            thresholds=thresholds,
            min_pair_probability=float(args.min_pair_probability),
            max_pair_margin=float(args.max_pair_margin),
            output_dir=output_dir / "train",
        ),
    }
    for eval_path in args.eval_predictions:
        eval_rows = read_prediction_csv(Path(eval_path), pair)
        split_name = eval_rows[0].split if eval_rows else Path(eval_path).stem
        splits[str(split_name)] = {
            "source": str(eval_path),
            "metrics": _evaluate_split(
                eval_rows,
                pair=pair,
                thresholds=thresholds,
                min_pair_probability=float(args.min_pair_probability),
                max_pair_margin=float(args.max_pair_margin),
                output_dir=output_dir / str(split_name),
            ),
        }

    summary = {
        "pair": f"{pair[0]}-{pair[1]}",
        "settings": {
            "directions": directions,
            "min_pair_probability": float(args.min_pair_probability),
            "max_pair_margin": float(args.max_pair_margin),
            "min_correction_precision": float(args.min_correction_precision),
            "min_net_corrections": int(args.min_net_corrections),
            "min_corrections": int(args.min_corrections),
        },
        "thresholds": thresholds,
        "splits": splits,
        "sources": [
            "https://proceedings.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf",
            "https://doi.org/10.1016/S0893-6080(05)80023-1",
            "https://jair.org/index.php/jair/article/view/10228",
        ],
        "leakage_guard": (
            "direction thresholds are selected from train OOF predictions only; "
            "evaluation CSVs are never used for threshold selection"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run(_parse_args(argv))
    compact = {
        "output_dir": str(summary.get("output_dir", "")),
        "thresholds": summary["thresholds"],
        "splits": {
            key: {
                "base": {
                    "macro_f1": value["metrics"]["base"]["macro_f1"],
                    "class1_f1": value["metrics"]["base"]["per_class"][1]["f1"]
                    if len(value["metrics"]["base"]["per_class"]) > 1
                    else None,
                },
                "directional_gated": {
                    "macro_f1": value["metrics"]["directional_gated"]["macro_f1"],
                    "class1_f1": value["metrics"]["directional_gated"]["per_class"][1]["f1"]
                    if len(value["metrics"]["directional_gated"]["per_class"]) > 1
                    else None,
                },
                "changes": value["metrics"]["changes"],
            }
            for key, value in summary["splits"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
