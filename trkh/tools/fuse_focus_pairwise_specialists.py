from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse a multiclass teacher cache with one-vs-one focus boundary specialists. "
            "Use validation to select a rule, then apply the saved rule unchanged to test."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="TRKH data.yaml or dataset root.")
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument(
        "--specialist",
        action="append",
        default=[],
        metavar="NEGATIVE_CLASS=CSV",
        help=(
            "Boundary specialist prediction CSV. The CSV must contain path and "
            "focus_probability. Repeat for each negative class, e.g. "
            "Xoai_Song_Chua_KhoDap=runs/pair01/predictions_val.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-name", default="Xoai_Song_ChuaNhe_CoNguyCo")
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=None,
        help="Apply the best rule from a previous validation summary instead of sweeping.",
    )
    parser.add_argument("--promote-thresholds", default="0.50:0.96:0.02")
    parser.add_argument("--demote-thresholds", default="-1,0.05:0.46:0.05")
    parser.add_argument("--min-teacher-focus", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--route-max-margins", default="0.08,0.12,0.16,0.20,0.30,1.00")
    parser.add_argument("--focus-weight", type=float, default=0.35)
    parser.add_argument("--focus-precision-weight", type=float, default=0.10)
    return parser.parse_args()


def _parse_float_grid(value: str) -> List[float]:
    output: List[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            output.append(float(parts[0]))
            continue
        if len(parts) != 3:
            raise ValueError(f"Invalid grid item {item!r}; use VALUE or START:STOP:STEP.")
        start, stop, step = (float(part) for part in parts)
        if step <= 0.0 or stop < start:
            raise ValueError(f"Invalid grid range {item!r}.")
        current = start
        while current <= stop + 1e-12:
            output.append(round(float(current), 10))
            current += step
    if not output:
        raise ValueError("At least one grid value is required.")
    return sorted(set(output))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _parse_specialist(value: str) -> Tuple[str, Path]:
    if "=" not in str(value):
        raise ValueError(f"--specialist must be NEGATIVE_CLASS=CSV, got {value!r}")
    name, path_text = str(value).split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        raise ValueError(f"Empty negative class in --specialist {value!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def _metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict[str, object]:
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[int(target), int(pred)] += 1
    per_class: List[Dict[str, object]] = []
    for index, class_name in enumerate(class_names):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append(
            {
                "class_name": str(class_name),
                "support": support,
                "predicted_support": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(sum(float(item["f1"]) for item in per_class) / max(1, class_count)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _objective(metrics: Mapping[str, object], *, focus_index: int, focus_weight: float, precision_weight: float) -> float:
    per_class = list(metrics["per_class"])  # type: ignore[index]
    focus = per_class[int(focus_index)]
    return (
        float(metrics["macro_f1"])
        + float(focus_weight) * float(focus["f1"])
        + float(precision_weight) * float(focus["precision"])
    )


def _teacher_probability(row: Mapping[str, str], class_count: int) -> np.ndarray:
    values = np.asarray([float(row[f"prob_{index}"]) for index in range(class_count)], dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(class_count, 1.0 / max(1, class_count), dtype=np.float64)
    return values / total


def _predict_with_rule(
    teacher_rows: Sequence[Mapping[str, str]],
    specialist_by_negative: Mapping[int, Mapping[str, float]],
    *,
    class_names: Sequence[str],
    focus_index: int,
    promote_threshold: float,
    demote_threshold: float,
    min_teacher_focus: float,
    route_max_margin: float,
) -> Tuple[List[int], List[Dict[str, object]], Counter[str]]:
    predictions: List[int] = []
    audit_rows: List[Dict[str, object]] = []
    counters: Counter[str] = Counter()
    class_count = len(class_names)
    for row in teacher_rows:
        path = str(row["path"])
        probabilities = _teacher_probability(row, class_count)
        base_pred = int(np.argmax(probabilities))
        pred = int(base_pred)
        action = "keep"
        routed_negative = ""
        routed_probability = None
        top2 = set(np.argsort(-probabilities)[:2].astype(int).tolist())
        focus_probability = float(probabilities[int(focus_index)])
        best_promote: Tuple[float, int] | None = None
        best_demote: Tuple[float, int] | None = None
        for negative_index, specialist_rows in specialist_by_negative.items():
            specialist_probability = specialist_rows.get(path)
            if specialist_probability is None:
                continue
            negative_probability = float(probabilities[int(negative_index)])
            margin = abs(focus_probability - negative_probability)
            route = (
                (base_pred in {int(focus_index), int(negative_index)})
                or ({int(focus_index), int(negative_index)}.issubset(top2))
            ) and margin <= float(route_max_margin)
            if not route:
                continue
            if base_pred == int(negative_index) and focus_probability >= float(min_teacher_focus):
                if float(specialist_probability) >= float(promote_threshold):
                    score = float(specialist_probability) - float(promote_threshold)
                    if best_promote is None or score > best_promote[0]:
                        best_promote = (score, int(negative_index))
            if base_pred == int(focus_index) and float(demote_threshold) >= 0.0:
                if float(specialist_probability) <= float(demote_threshold):
                    score = float(demote_threshold) - float(specialist_probability)
                    if best_demote is None or score > best_demote[0]:
                        best_demote = (score, int(negative_index))
        if best_promote is not None:
            pred = int(focus_index)
            action = "promote_focus"
            routed_negative = class_names[int(best_promote[1])]
            routed_probability = float(specialist_by_negative[int(best_promote[1])][path])
            counters[f"promote_from_{routed_negative}"] += 1
        elif best_demote is not None:
            pred = int(best_demote[1])
            action = "demote_focus"
            routed_negative = class_names[int(best_demote[1])]
            routed_probability = float(specialist_by_negative[int(best_demote[1])][path])
            counters[f"demote_to_{routed_negative}"] += 1

        predictions.append(pred)
        audit_rows.append(
            {
                "path": path,
                "true_name": str(row["true_name"]),
                "teacher_pred_index": base_pred,
                "teacher_pred_name": class_names[base_pred],
                "fused_pred_index": pred,
                "fused_pred_name": class_names[pred],
                "action": action,
                "routed_negative_class": routed_negative,
                "routed_specialist_focus_probability": "" if routed_probability is None else routed_probability,
                "teacher_focus_probability": focus_probability,
                "teacher_confidence": float(probabilities[base_pred]),
            }
        )
    return predictions, audit_rows, counters


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "true_name",
        "teacher_pred_index",
        "teacher_pred_name",
        "fused_pred_index",
        "fused_pred_name",
        "action",
        "routed_negative_class",
        "routed_specialist_focus_probability",
        "teacher_focus_probability",
        "teacher_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    data_spec = load_data_spec(args.data)
    class_names = [str(value) for value in data_spec.class_names]
    if args.focus_class_name not in class_names:
        raise ValueError(f"Unknown focus class {args.focus_class_name!r}; classes={class_names}")
    focus_index = int(class_names.index(str(args.focus_class_name)))

    teacher_rows = _read_csv(args.teacher_csv)
    if not teacher_rows:
        raise ValueError(f"Teacher CSV is empty: {args.teacher_csv}")
    y_true = [int(class_names.index(str(row["true_name"]))) for row in teacher_rows]
    teacher_pred = [int(row.get("teacher_pred_index", int(np.argmax(_teacher_probability(row, len(class_names)))))) for row in teacher_rows]
    base_metrics = _metrics(y_true, teacher_pred, class_names)

    specialist_by_negative: Dict[int, Dict[str, float]] = {}
    specialist_sources: Dict[str, str] = {}
    for negative_class, csv_path in [_parse_specialist(value) for value in args.specialist]:
        if negative_class not in class_names:
            raise ValueError(f"Unknown specialist negative class {negative_class!r}; classes={class_names}")
        negative_index = int(class_names.index(negative_class))
        if negative_index == focus_index:
            raise ValueError("Specialist negative class cannot equal focus class.")
        rows = _read_csv(csv_path)
        mapping: Dict[str, float] = {}
        for row in rows:
            if "focus_probability" not in row:
                raise ValueError(f"Specialist CSV missing focus_probability: {csv_path}")
            mapping[str(row["path"])] = float(row["focus_probability"])
        specialist_by_negative[negative_index] = mapping
        specialist_sources[negative_class] = str(csv_path)
    if not specialist_by_negative:
        raise ValueError("At least one --specialist is required.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rules_json is not None:
        payload = json.loads(args.rules_json.read_text(encoding="utf-8"))
        best_rule = dict(payload.get("best_rule", payload.get("best", {})))
        if not best_rule:
            raise ValueError(f"No best_rule/best found in {args.rules_json}")
        rule_rows = [best_rule]
    else:
        rule_rows = []
        for promote_threshold in _parse_float_grid(args.promote_thresholds):
            for demote_threshold in _parse_float_grid(args.demote_thresholds):
                for min_teacher_focus in _parse_float_grid(args.min_teacher_focus):
                    for route_max_margin in _parse_float_grid(args.route_max_margins):
                        rule_rows.append(
                            {
                                "promote_threshold": float(promote_threshold),
                                "demote_threshold": float(demote_threshold),
                                "min_teacher_focus": float(min_teacher_focus),
                                "route_max_margin": float(route_max_margin),
                            }
                        )

    sweep_rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None
    best_predictions: List[int] = []
    best_audit_rows: List[Dict[str, object]] = []
    best_counters: Counter[str] = Counter()
    best_key = (-1.0, -1.0, -1.0)
    for rule in rule_rows:
        predictions, audit_rows, counters = _predict_with_rule(
            teacher_rows,
            specialist_by_negative,
            class_names=class_names,
            focus_index=focus_index,
            promote_threshold=float(rule["promote_threshold"]),
            demote_threshold=float(rule["demote_threshold"]),
            min_teacher_focus=float(rule["min_teacher_focus"]),
            route_max_margin=float(rule["route_max_margin"]),
        )
        metrics = _metrics(y_true, predictions, class_names)
        focus = metrics["per_class"][focus_index]  # type: ignore[index]
        changed = int(sum(int(a != b) for a, b in zip(teacher_pred, predictions)))
        row = {
            **rule,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "focus_f1": float(focus["f1"]),
            "focus_precision": float(focus["precision"]),
            "focus_recall": float(focus["recall"]),
            "predicted_focus": int(focus["predicted_support"]),
            "changed": changed,
            "promoted": int(sum(value for key, value in counters.items() if key.startswith("promote_"))),
            "demoted": int(sum(value for key, value in counters.items() if key.startswith("demote_"))),
            "objective": _objective(
                metrics,
                focus_index=focus_index,
                focus_weight=float(args.focus_weight),
                precision_weight=float(args.focus_precision_weight),
            ),
        }
        sweep_rows.append(row)
        key = (
            float(row["objective"]),
            float(row["macro_f1"]),
            float(row["focus_f1"]),
        )
        if key > best_key:
            best_key = key
            best = row
            best_predictions = predictions
            best_audit_rows = audit_rows
            best_counters = counters

    if best is None:
        raise RuntimeError("No fusion rule was evaluated.")
    best_metrics = _metrics(y_true, best_predictions, class_names)
    _write_predictions(args.output_dir / "predictions_fused.csv", best_audit_rows)
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(sweep_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    summary = {
        "mode": "focus_pairwise_specialist_fusion",
        "classes": class_names,
        "focus_class_name": str(args.focus_class_name),
        "focus_index": int(focus_index),
        "teacher_csv": str(args.teacher_csv),
        "specialists": specialist_sources,
        "rules_json": None if args.rules_json is None else str(args.rules_json),
        "base_metrics": base_metrics,
        "best_rule": best,
        "best_metrics": best_metrics,
        "action_counts": dict(best_counters),
        "samples": len(teacher_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "base_macro_f1": float(base_metrics["macro_f1"]),
                "base_focus_f1": float(base_metrics["per_class"][focus_index]["f1"]),  # type: ignore[index]
                "best_macro_f1": float(best_metrics["macro_f1"]),
                "best_focus_f1": float(best_metrics["per_class"][focus_index]["f1"]),  # type: ignore[index]
                "best_rule": best,
                "summary": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
