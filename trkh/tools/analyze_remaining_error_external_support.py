from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROBABILITY_COLUMN_RE = re.compile(r"^prob_(\d+)(?:_.+)?$")


@dataclass(frozen=True)
class PredictionTable:
    name: str
    path: Path
    rows_by_key: Mapping[str, Mapping[str, str]]
    probabilities_by_key: Mapping[str, List[float]]
    predictions_by_key: Mapping[str, int]
    targets_by_key: Mapping[str, int]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only diagnostic for remaining TRKH errors. It measures "
            "whether external/pretrained predictors consistently support a fix, "
            "without selecting deployable thresholds or touching raw data."
        )
    )
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--remaining-errors-csv", type=Path, required=True)
    parser.add_argument(
        "--external",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="External/support prediction CSV with sample_index and prediction/prob columns.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument(
        "--consensus-min-votes",
        type=int,
        default=2,
        help="Minimum external predictors agreeing on one non-base class for diagnostic routing.",
    )
    return parser.parse_args(argv)


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CSV, got {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Empty external name in {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return name, path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _key(row: Mapping[str, str], path: Path) -> str:
    sample_index = str(row.get("sample_index", "") or "").strip()
    if not sample_index:
        raise ValueError(f"CSV row lacks sample_index in {path}")
    try:
        return str(int(sample_index))
    except ValueError as exc:
        raise ValueError(f"Invalid sample_index {sample_index!r} in {path}") from exc


def _int_from_row(row: Mapping[str, str], keys: Sequence[str]) -> int:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return int(value)
    raise ValueError(f"Row lacks any of {keys!r}")


def _probability_columns(row: Mapping[str, str]) -> List[str]:
    preferred: Dict[int, str] = {}
    fallback: Dict[int, str] = {}
    for column in row:
        match = PROBABILITY_COLUMN_RE.match(str(column))
        if match is None:
            continue
        index = int(match.group(1))
        if str(column) == f"prob_{index}":
            preferred[index] = str(column)
        elif index not in fallback:
            fallback[index] = str(column)
    columns_by_index = dict(fallback)
    columns_by_index.update(preferred)
    if not columns_by_index:
        return []
    expected = list(range(max(columns_by_index) + 1))
    if sorted(columns_by_index) != expected:
        raise ValueError(f"Probability columns must be contiguous from 0: {sorted(columns_by_index)}")
    return [columns_by_index[index] for index in expected]


def _row_probabilities(row: Mapping[str, str], columns: Sequence[str], prediction: int) -> List[float]:
    if not columns:
        class_count = max(5, prediction + 1)
        values = [0.0 for _ in range(class_count)]
        values[prediction] = 1.0
        return values
    values = [max(0.0, float(row[column])) for column in columns]
    total = sum(values)
    if total <= 0.0:
        values = [0.0 for _ in columns]
        if 0 <= prediction < len(values):
            values[prediction] = 1.0
        else:
            values[0] = 1.0
        total = 1.0
    return [float(value / total) for value in values]


def _load_prediction_table(name: str, path: Path) -> PredictionTable:
    rows = _read_csv(path)
    columns = _probability_columns(rows[0])
    rows_by_key: Dict[str, Mapping[str, str]] = {}
    probabilities_by_key: Dict[str, List[float]] = {}
    predictions_by_key: Dict[str, int] = {}
    targets_by_key: Dict[str, int] = {}
    duplicate_count = 0
    for row in rows:
        key = _key(row, path)
        if key in rows_by_key:
            duplicate_count += 1
        prediction = _int_from_row(row, ("prediction_index", "y_pred", "pred_index"))
        target = _int_from_row(row, ("target_index", "y_true", "true_index"))
        rows_by_key[key] = dict(row)
        probabilities_by_key[key] = _row_probabilities(row, columns, prediction)
        predictions_by_key[key] = prediction
        targets_by_key[key] = target
    if duplicate_count:
        raise ValueError(f"Duplicate sample_index rows in {path}: {duplicate_count}")
    return PredictionTable(
        name=name,
        path=path,
        rows_by_key=rows_by_key,
        probabilities_by_key=probabilities_by_key,
        predictions_by_key=predictions_by_key,
        targets_by_key=targets_by_key,
    )


def _metrics(targets: Sequence[int], predictions: Sequence[int], class_count: int) -> Dict[str, object]:
    total = len(targets)
    correct = sum(1 for target, pred in zip(targets, predictions) if target == pred)
    per_class: List[Dict[str, float]] = []
    f1_values: List[float] = []
    for cls in range(class_count):
        tp = sum(1 for target, pred in zip(targets, predictions) if target == cls and pred == cls)
        fp = sum(1 for target, pred in zip(targets, predictions) if target != cls and pred == cls)
        fn = sum(1 for target, pred in zip(targets, predictions) if target == cls and pred != cls)
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": cls,
                "support": int(tp + fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return {
        "samples": int(total),
        "accuracy": float(correct / total) if total else 0.0,
        "macro_f1": float(sum(f1_values) / class_count) if class_count else 0.0,
        "per_class": per_class,
    }


def _bucket(target: int, base_pred: int, focus_class: int) -> str:
    if target == focus_class and base_pred != focus_class:
        return "focus_false_negative"
    if target != focus_class and base_pred == focus_class:
        return "focus_false_positive"
    return "other_error"


def _top_vote(predictions: Iterable[int]) -> Tuple[int, int, bool]:
    counts = Counter(int(prediction) for prediction in predictions)
    if not counts:
        return -1, 0, False
    ordered = counts.most_common()
    top_pred, top_count = ordered[0]
    tied = len(ordered) > 1 and ordered[1][1] == top_count
    return int(top_pred), int(top_count), bool(tied)


def _stable_route_stats(counter: Mapping[str, int]) -> Dict[str, int]:
    keys = (
        "changes",
        "corrections",
        "harms",
        "wrong_to_wrong",
        "neutral",
        "suppress_focus_changes",
        "rescue_focus_changes",
    )
    return {key: int(counter.get(key, 0)) for key in keys if key in counter or key in {"changes", "corrections", "harms", "wrong_to_wrong", "neutral"}}


def _probability_for(table: PredictionTable, key: str, cls: int) -> float:
    probs = table.probabilities_by_key[key]
    if 0 <= cls < len(probs):
        return float(probs[cls])
    return 0.0


def _diagnostic_consensus_predictions(
    keys: Sequence[str],
    base: PredictionTable,
    externals: Sequence[PredictionTable],
    min_votes: int,
) -> Tuple[List[int], Dict[str, int]]:
    routed_predictions: List[int] = []
    route_stats = Counter()
    for key in keys:
        base_pred = base.predictions_by_key[key]
        external_preds = [table.predictions_by_key[key] for table in externals]
        consensus_pred, consensus_votes, tied = _top_vote(external_preds)
        route = consensus_votes >= min_votes and not tied and consensus_pred != base_pred
        if route:
            routed_predictions.append(consensus_pred)
            route_stats["changes"] += 1
            target = base.targets_by_key[key]
            if base_pred != target and consensus_pred == target:
                route_stats["corrections"] += 1
            elif base_pred == target and consensus_pred != target:
                route_stats["harms"] += 1
            elif base_pred != target and consensus_pred != target:
                route_stats["wrong_to_wrong"] += 1
            else:
                route_stats["neutral"] += 1
        else:
            routed_predictions.append(base_pred)
    return routed_predictions, _stable_route_stats(route_stats)


def _focus_consensus_predictions(
    keys: Sequence[str],
    base: PredictionTable,
    externals: Sequence[PredictionTable],
    min_votes: int,
    focus_class: int,
) -> Tuple[List[int], Dict[str, int]]:
    routed_predictions: List[int] = []
    route_stats = Counter()
    for key in keys:
        base_pred = base.predictions_by_key[key]
        target = base.targets_by_key[key]
        external_preds = [table.predictions_by_key[key] for table in externals]
        focus_votes = sum(1 for pred in external_preds if pred == focus_class)
        non_focus_preds = [pred for pred in external_preds if pred != focus_class]
        non_focus_pred, non_focus_votes, non_focus_tied = _top_vote(non_focus_preds)
        proposal = base_pred
        if base_pred == focus_class and non_focus_votes >= min_votes and not non_focus_tied:
            proposal = non_focus_pred
            route_stats["suppress_focus_changes"] += 1
        elif base_pred != focus_class and focus_votes >= min_votes:
            proposal = focus_class
            route_stats["rescue_focus_changes"] += 1
        routed_predictions.append(proposal)
        if proposal != base_pred:
            route_stats["changes"] += 1
            if base_pred != target and proposal == target:
                route_stats["corrections"] += 1
            elif base_pred == target and proposal != target:
                route_stats["harms"] += 1
            elif base_pred != target and proposal != target:
                route_stats["wrong_to_wrong"] += 1
            else:
                route_stats["neutral"] += 1
    return routed_predictions, _stable_route_stats(route_stats)


def run(args: argparse.Namespace) -> Dict[str, object]:
    base = _load_prediction_table("base", args.base_csv)
    remaining_rows = _read_csv(args.remaining_errors_csv)
    external_inputs = [_parse_named_path(value) for value in args.external]
    externals = [_load_prediction_table(name, path) for name, path in external_inputs]
    keys = sorted(base.rows_by_key, key=lambda value: int(value))
    class_count = max(max(base.targets_by_key.values()), max(base.predictions_by_key.values())) + 1
    for table in externals:
        missing = [key for key in keys if key not in table.rows_by_key]
        if missing:
            raise ValueError(f"{table.name} missing {len(missing)} base sample_index rows; first={missing[:5]}")
        class_count = max(class_count, max(table.predictions_by_key.values()) + 1)

    targets = [base.targets_by_key[key] for key in keys]
    base_predictions = [base.predictions_by_key[key] for key in keys]
    base_metrics = _metrics(targets, base_predictions, class_count)

    remaining_keys = [_key(row, args.remaining_errors_csv) for row in remaining_rows]
    remaining_support_rows: List[Dict[str, object]] = []
    support_by_bucket: Dict[str, Counter] = {}
    for key, remaining_row in zip(remaining_keys, remaining_rows):
        if key not in base.rows_by_key:
            raise ValueError(f"Remaining error sample_index {key} not present in base CSV.")
        target = base.targets_by_key[key]
        base_pred = base.predictions_by_key[key]
        bucket = _bucket(target, base_pred, args.focus_class_index)
        counter = support_by_bucket.setdefault(bucket, Counter())
        external_preds = [table.predictions_by_key[key] for table in externals]
        consensus_pred, consensus_votes, consensus_tied = _top_vote(external_preds)
        correct_methods = [table.name for table in externals if table.predictions_by_key[key] == target]
        focus_votes = sum(1 for pred in external_preds if pred == args.focus_class_index)
        non_focus_pred, non_focus_votes, non_focus_tied = _top_vote(
            pred for pred in external_preds if pred != args.focus_class_index
        )
        if correct_methods:
            counter["any_external_correct"] += 1
        else:
            counter["no_external_correct"] += 1
        counter[f"external_correct_count_{len(correct_methods)}"] += 1
        if consensus_votes >= args.consensus_min_votes and not consensus_tied:
            counter["consensus_available"] += 1
            if consensus_pred == target:
                counter["consensus_correct"] += 1
            elif consensus_pred != base_pred:
                counter["consensus_wrong_change"] += 1
        if bucket == "focus_false_negative" and focus_votes >= args.consensus_min_votes:
            counter["focus_rescue_votes_available"] += 1
        if (
            bucket == "focus_false_positive"
            and non_focus_votes >= args.consensus_min_votes
            and not non_focus_tied
        ):
            counter["focus_suppress_votes_available"] += 1

        output_row: Dict[str, object] = {
            "sample_index": key,
            "image_path": remaining_row.get("image_path", base.rows_by_key[key].get("image_path", "")),
            "source_stem": remaining_row.get("source_stem", base.rows_by_key[key].get("source_stem", "")),
            "object_index": remaining_row.get("object_index", base.rows_by_key[key].get("object_index", "")),
            "target_index": target,
            "base_prediction_index": base_pred,
            "error_pair": f"{target}->{base_pred}",
            "bucket": bucket,
            "base_confidence": base.rows_by_key[key].get("confidence", ""),
            "base_margin": remaining_row.get("margin", ""),
            "external_correct_count": len(correct_methods),
            "external_correct_methods": "|".join(correct_methods),
            "external_consensus_prediction": consensus_pred,
            "external_consensus_votes": consensus_votes,
            "external_consensus_tied": int(consensus_tied),
            "focus_class_votes": focus_votes,
            "non_focus_consensus_prediction": non_focus_pred,
            "non_focus_consensus_votes": non_focus_votes,
            "non_focus_consensus_tied": int(non_focus_tied),
        }
        for table in externals:
            output_row[f"{table.name}_prediction"] = table.predictions_by_key[key]
            output_row[f"{table.name}_correct"] = int(table.predictions_by_key[key] == target)
            output_row[f"{table.name}_prob_target"] = _probability_for(table, key, target)
            output_row[f"{table.name}_prob_focus"] = _probability_for(table, key, args.focus_class_index)
        remaining_support_rows.append(output_row)

    external_summaries: Dict[str, object] = {}
    for table in externals:
        predictions = [table.predictions_by_key[key] for key in keys]
        metrics = _metrics(targets, predictions, class_count)
        corrections = harms = wrong_to_wrong = neutral = 0
        for key in keys:
            target = base.targets_by_key[key]
            base_pred = base.predictions_by_key[key]
            pred = table.predictions_by_key[key]
            if pred == base_pred:
                continue
            if base_pred != target and pred == target:
                corrections += 1
            elif base_pred == target and pred != target:
                harms += 1
            elif base_pred != target and pred != target:
                wrong_to_wrong += 1
            else:
                neutral += 1
        external_summaries[table.name] = {
            "csv": str(table.path),
            "metrics": metrics,
            "changes_vs_base": {
                "corrections": corrections,
                "harms": harms,
                "wrong_to_wrong": wrong_to_wrong,
                "neutral": neutral,
            },
        }

    consensus_predictions, consensus_stats = _diagnostic_consensus_predictions(
        keys, base, externals, args.consensus_min_votes
    )
    focus_predictions, focus_stats = _focus_consensus_predictions(
        keys,
        base,
        externals,
        args.consensus_min_votes,
        args.focus_class_index,
    )
    consensus_metrics = _metrics(targets, consensus_predictions, class_count)
    focus_metrics = _metrics(targets, focus_predictions, class_count)

    summary: Dict[str, object] = {
        "mode": "remaining_error_external_support",
        "note": (
            "Validation-only diagnostic. External models are support probes, not a "
            "deployable no-pretrain TRKH route."
        ),
        "base_csv": str(args.base_csv),
        "remaining_errors_csv": str(args.remaining_errors_csv),
        "output_dir": str(args.output_dir),
        "focus_class_index": int(args.focus_class_index),
        "consensus_min_votes": int(args.consensus_min_votes),
        "samples": len(keys),
        "remaining_error_count": len(remaining_keys),
        "base_metrics": base_metrics,
        "external": external_summaries,
        "remaining_error_support_by_bucket": {
            bucket: dict(counter) for bucket, counter in sorted(support_by_bucket.items())
        },
        "diagnostic_consensus_route": {
            "metrics": consensus_metrics,
            "stats": consensus_stats,
        },
        "diagnostic_focus_consensus_route": {
            "metrics": focus_metrics,
            "stats": focus_stats,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if remaining_support_rows:
        fieldnames = list(remaining_support_rows[0].keys())
        with (args.output_dir / "remaining_error_external_support.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(remaining_support_rows)
    with (args.output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Remaining Error External Support Diagnostic\n\n")
        handle.write("Validation-only diagnostic; do not use as a deployment router or test tuner.\n\n")
        handle.write(f"- Base CSV: `{args.base_csv}`\n")
        handle.write(f"- Remaining errors CSV: `{args.remaining_errors_csv}`\n")
        handle.write(f"- External predictors: {', '.join(table.name for table in externals)}\n")
        handle.write(f"- Consensus min votes: `{args.consensus_min_votes}`\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
