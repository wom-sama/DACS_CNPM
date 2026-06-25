from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from trkh.evaluation.metrics import build_metrics
from trkh.tools.calibrate_classification_logits import read_prediction_csv


@dataclass(frozen=True)
class RerouteRule:
    focus_class_index: int
    retain_threshold: Optional[float] = None
    retain_margin: Optional[float] = None
    rescue_threshold: Optional[float] = None
    rescue_margin: Optional[float] = None


def _top2(probabilities: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    values, indices = torch.topk(probabilities.to(torch.float64), k=2, dim=1)
    return indices[:, 0], values[:, 0], indices[:, 1], values[:, 1]


def apply_reroute_rule(probabilities: Tensor, rule: RerouteRule) -> Tuple[Tensor, Dict[str, int]]:
    if probabilities.dim() != 2:
        raise ValueError("probabilities must have shape [N, C].")
    if not 0 <= int(rule.focus_class_index) < probabilities.size(1):
        raise ValueError("focus_class_index is outside the class range.")

    top1, top1_prob, top2, top2_prob = _top2(probabilities)
    predictions = top1.clone()
    focus = int(rule.focus_class_index)
    changed_from_focus = 0
    changed_to_focus = 0

    margins = top1_prob - top2_prob
    focus_prob = probabilities[:, focus].to(torch.float64)
    focus_top1 = top1 == focus
    reject_focus = torch.zeros_like(focus_top1)
    if rule.retain_threshold is not None:
        reject_focus |= focus_top1 & (focus_prob < float(rule.retain_threshold))
    if rule.retain_margin is not None:
        reject_focus |= focus_top1 & (margins < float(rule.retain_margin))
    if reject_focus.any():
        predictions[reject_focus] = top2[reject_focus]
        changed_from_focus = int(reject_focus.sum().item())

    if rule.rescue_threshold is not None:
        close_to_focus = top2 == focus
        rescue = (
            (~focus_top1)
            & close_to_focus
            & (focus_prob >= float(rule.rescue_threshold))
            & ((top1_prob - focus_prob) <= float(rule.rescue_margin or 0.0))
        )
        if rescue.any():
            predictions[rescue] = focus
            changed_to_focus = int(rescue.sum().item())

    changed = predictions != top1
    return predictions.to(torch.int64), {
        "changed": int(changed.sum().item()),
        "changed_from_focus": int(changed_from_focus),
        "changed_to_focus": int(changed_to_focus),
    }


def _metrics(targets: Tensor, predictions: Tensor, class_names: Sequence[str]) -> Dict[str, object]:
    return build_metrics(
        targets=targets.to(torch.int64),
        predictions=predictions.to(torch.int64),
        class_names=class_names,
    )


def _focus_f1(metrics: Mapping[str, object], focus_class_index: int) -> float:
    return float(list(metrics["per_class"])[int(focus_class_index)]["f1"])


def _focus_precision(metrics: Mapping[str, object], focus_class_index: int) -> float:
    return float(list(metrics["per_class"])[int(focus_class_index)]["precision"])


def _focus_recall(metrics: Mapping[str, object], focus_class_index: int) -> float:
    return float(list(metrics["per_class"])[int(focus_class_index)]["recall"])


def objective_score(
    metrics: Mapping[str, object],
    *,
    baseline_macro_f1: float,
    focus_class_index: int,
    focus_weight: float,
    macro_weight: float,
    macro_drop_penalty: float,
    precision_weight: float,
) -> float:
    macro_f1 = float(metrics["macro_f1"])
    focus_f1 = _focus_f1(metrics, focus_class_index)
    focus_precision = _focus_precision(metrics, focus_class_index)
    return (
        float(macro_weight) * macro_f1
        + float(focus_weight) * focus_f1
        + float(precision_weight) * focus_precision
        - max(0.0, float(baseline_macro_f1) - macro_f1) * float(macro_drop_penalty)
    )


def _parse_grid(value: str, *, allow_none: bool = True) -> List[Optional[float]]:
    text = str(value or "").strip()
    if not text:
        return [None] if allow_none else []
    values: List[Optional[float]] = []
    for item in text.replace(";", ",").split(","):
        part = item.strip()
        if not part:
            continue
        if part.lower() in {"none", "off", "null"}:
            if allow_none:
                values.append(None)
            continue
        if ":" in part:
            pieces = [float(piece.strip()) for piece in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid grid range: {part!r}")
            start, stop = pieces[0], pieces[1]
            step = pieces[2] if len(pieces) == 3 else 0.01
            if step <= 0.0:
                raise ValueError(f"Grid step must be positive: {part!r}")
            count = int(math.floor((stop - start) / step + 1e-9)) + 1
            for index in range(max(0, count)):
                values.append(round(start + step * index, 10))
        else:
            values.append(float(part))
    return values


def _candidate_rules(
    *,
    focus_class_index: int,
    retain_thresholds: Sequence[Optional[float]],
    retain_margins: Sequence[Optional[float]],
    rescue_thresholds: Sequence[Optional[float]],
    rescue_margins: Sequence[Optional[float]],
    enable_rescue: bool,
) -> List[RerouteRule]:
    rules: List[RerouteRule] = []
    for retain_threshold in retain_thresholds:
        for retain_margin in retain_margins:
            if retain_threshold is None and retain_margin is None:
                continue
            rules.append(
                RerouteRule(
                    focus_class_index=focus_class_index,
                    retain_threshold=retain_threshold,
                    retain_margin=retain_margin,
                )
            )
            if not enable_rescue:
                continue
            for rescue_threshold in rescue_thresholds:
                if rescue_threshold is None:
                    continue
                for rescue_margin in rescue_margins:
                    if rescue_margin is None:
                        continue
                    rules.append(
                        RerouteRule(
                            focus_class_index=focus_class_index,
                            retain_threshold=retain_threshold,
                            retain_margin=retain_margin,
                            rescue_threshold=rescue_threshold,
                            rescue_margin=rescue_margin,
                        )
                    )
    return rules


def fit_reroute_rule(
    targets: Tensor,
    probabilities: Tensor,
    class_names: Sequence[str],
    *,
    focus_class_index: int = 1,
    retain_thresholds: Sequence[Optional[float]] = (None, 0.24, 0.25, 0.26, 0.27, 0.28),
    retain_margins: Sequence[Optional[float]] = (None, 0.02, 0.03, 0.04, 0.05),
    rescue_thresholds: Sequence[Optional[float]] = (0.30, 0.32, 0.34),
    rescue_margins: Sequence[Optional[float]] = (0.01, 0.02, 0.03),
    enable_rescue: bool = False,
    focus_weight: float = 1.0,
    macro_weight: float = 0.35,
    macro_drop_penalty: float = 3.0,
    precision_weight: float = 0.10,
) -> Tuple[RerouteRule, Dict[str, object], List[Dict[str, object]]]:
    if probabilities.size(0) != targets.numel():
        raise ValueError("targets and probabilities have different lengths.")
    base_predictions = probabilities.argmax(dim=1).to(torch.int64)
    baseline = _metrics(targets, base_predictions, class_names)
    baseline_macro = float(baseline["macro_f1"])
    rules = _candidate_rules(
        focus_class_index=focus_class_index,
        retain_thresholds=retain_thresholds,
        retain_margins=retain_margins,
        rescue_thresholds=rescue_thresholds,
        rescue_margins=rescue_margins,
        enable_rescue=enable_rescue,
    )
    if not rules:
        raise ValueError("No candidate reroute rules were generated.")

    best_rule = rules[0]
    best_metrics: Dict[str, object] = baseline
    best_score = float("-inf")
    trace: List[Dict[str, object]] = []
    for rule in rules:
        predictions, delta = apply_reroute_rule(probabilities, rule)
        metrics = _metrics(targets, predictions, class_names)
        score = objective_score(
            metrics,
            baseline_macro_f1=baseline_macro,
            focus_class_index=focus_class_index,
            focus_weight=focus_weight,
            macro_weight=macro_weight,
            macro_drop_penalty=macro_drop_penalty,
            precision_weight=precision_weight,
        )
        row = {
            **asdict(rule),
            "score": float(score),
            "macro_f1": float(metrics["macro_f1"]),
            "accuracy": float(metrics["accuracy"]),
            "focus_f1": _focus_f1(metrics, focus_class_index),
            "focus_precision": _focus_precision(metrics, focus_class_index),
            "focus_recall": _focus_recall(metrics, focus_class_index),
            **delta,
        }
        trace.append(row)
        key = (
            float(score),
            row["focus_f1"],
            row["macro_f1"],
            row["accuracy"],
            -float(delta["changed"]),
        )
        best_key = (
            best_score,
            _focus_f1(best_metrics, focus_class_index),
            float(best_metrics["macro_f1"]),
            float(best_metrics["accuracy"]),
            0.0,
        )
        if key > best_key:
            best_rule = rule
            best_metrics = metrics
            best_score = float(score)
    return best_rule, best_metrics, trace


def write_rerouted_predictions(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    class_names: Sequence[str],
    probabilities: Tensor,
    predictions: Tensor,
    rule: RerouteRule,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top1, top1_prob, top2, top2_prob = _top2(probabilities)
    fieldnames = list(rows[0].keys()) if rows else []
    extra = [
        "reroute_prediction_index",
        "reroute_prediction_name",
        "reroute_confidence",
        "reroute_correct",
        "reroute_changed",
        "reroute_focus_probability",
        "reroute_top1_top2_margin",
        "reroute_retain_threshold",
        "reroute_retain_margin",
        "reroute_rescue_threshold",
        "reroute_rescue_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra])
        writer.writeheader()
        for index, source in enumerate(rows):
            prediction = int(predictions[index].item())
            target = int(source["target_index"])
            row = dict(source)
            row.update(
                {
                    "reroute_prediction_index": prediction,
                    "reroute_prediction_name": class_names[prediction],
                    "reroute_confidence": float(probabilities[index, prediction].item()),
                    "reroute_correct": int(prediction == target),
                    "reroute_changed": int(prediction != int(top1[index].item())),
                    "reroute_focus_probability": float(
                        probabilities[index, int(rule.focus_class_index)].item()
                    ),
                    "reroute_top1_top2_margin": float((top1_prob[index] - top2_prob[index]).item()),
                    "reroute_retain_threshold": rule.retain_threshold,
                    "reroute_retain_margin": rule.retain_margin,
                    "reroute_rescue_threshold": rule.rescue_threshold,
                    "reroute_rescue_margin": rule.rescue_margin,
                }
            )
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a class-specific reroute rule on one prediction CSV and apply it to another. "
            "Use train or val for fitting, then freeze the rule before any final test audit."
        )
    )
    parser.add_argument("--fit-predictions", type=Path, required=True)
    parser.add_argument("--apply-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--retain-thresholds", type=str, default="none,0.220:0.360:0.005")
    parser.add_argument("--retain-margins", type=str, default="none,0.000:0.080:0.005")
    parser.add_argument("--rescue-thresholds", type=str, default="0.280:0.360:0.020")
    parser.add_argument("--rescue-margins", type=str, default="0.005:0.040:0.005")
    parser.add_argument("--enable-rescue", action="store_true")
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--macro-weight", type=float, default=0.35)
    parser.add_argument("--macro-drop-penalty", type=float, default=3.0)
    parser.add_argument("--precision-weight", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fit_rows, class_names, fit_targets, fit_probabilities = read_prediction_csv(args.fit_predictions)
    apply_rows, apply_class_names, apply_targets, apply_probabilities = read_prediction_csv(
        args.apply_predictions
    )
    if apply_class_names != class_names:
        raise ValueError(
            f"Class columns differ: fit={class_names}, apply={apply_class_names}"
        )
    retain_thresholds = _parse_grid(args.retain_thresholds)
    retain_margins = _parse_grid(args.retain_margins)
    rescue_thresholds = _parse_grid(args.rescue_thresholds, allow_none=False)
    rescue_margins = _parse_grid(args.rescue_margins, allow_none=False)

    base_fit_predictions = fit_probabilities.argmax(dim=1).to(torch.int64)
    base_apply_predictions = apply_probabilities.argmax(dim=1).to(torch.int64)
    fit_baseline = _metrics(fit_targets, base_fit_predictions, class_names)
    apply_baseline = _metrics(apply_targets, base_apply_predictions, class_names)
    rule, fit_adjusted, trace = fit_reroute_rule(
        fit_targets,
        fit_probabilities,
        class_names,
        focus_class_index=args.focus_class_index,
        retain_thresholds=retain_thresholds,
        retain_margins=retain_margins,
        rescue_thresholds=rescue_thresholds,
        rescue_margins=rescue_margins,
        enable_rescue=bool(args.enable_rescue),
        focus_weight=args.focus_weight,
        macro_weight=args.macro_weight,
        macro_drop_penalty=args.macro_drop_penalty,
        precision_weight=args.precision_weight,
    )
    apply_predictions, apply_delta = apply_reroute_rule(apply_probabilities, rule)
    apply_adjusted = _metrics(apply_targets, apply_predictions, class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rerouted_predictions(
        output_dir / "apply_predictions_rerouted.csv",
        apply_rows,
        class_names,
        apply_probabilities,
        apply_predictions,
        rule,
    )
    fit_predictions, fit_delta = apply_reroute_rule(fit_probabilities, rule)
    write_rerouted_predictions(
        output_dir / "fit_predictions_rerouted.csv",
        fit_rows,
        class_names,
        fit_probabilities,
        fit_predictions,
        rule,
    )
    trace_fields = [
        "focus_class_index",
        "retain_threshold",
        "retain_margin",
        "rescue_threshold",
        "rescue_margin",
        "score",
        "macro_f1",
        "accuracy",
        "focus_f1",
        "focus_precision",
        "focus_recall",
        "changed",
        "changed_from_focus",
        "changed_to_focus",
    ]
    with (output_dir / "search_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        for row in trace:
            writer.writerow({name: row.get(name) for name in trace_fields})

    summary = {
        "mode": "class_focus_reroute",
        "fit_predictions": str(Path(args.fit_predictions).resolve()),
        "apply_predictions": str(Path(args.apply_predictions).resolve()),
        "class_names": list(class_names),
        "rule": asdict(rule),
        "fit_delta": fit_delta,
        "apply_delta": apply_delta,
        "fit_baseline": fit_baseline,
        "fit_adjusted": fit_adjusted,
        "apply_baseline": apply_baseline,
        "apply_adjusted": apply_adjusted,
        "leakage_note": (
            "Fit the rule on train or val only. Do not inspect test until model and "
            "reroute rule are frozen."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
