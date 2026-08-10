from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from trkh.core.config import to_serializable
from trkh.core.utils import ensure_dir
from trkh.evaluation.metrics import build_metrics


CLASS1_INDEX = 1


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether clean-vs-hflip decision instability is a useful "
            "signal for current TRKH softboost class-1 errors. This is a "
            "diagnostic-only join over existing validation artifacts; it does "
            "not aggregate TTA predictions, tune thresholds, read test, or "
            "write train manifests."
        )
    )
    parser.add_argument("--hflip-predictions", type=Path, required=True)
    parser.add_argument("--softboost-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class1-index", type=int, default=CLASS1_INDEX)
    parser.add_argument("--top-k-cases", type=int, default=80)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _int(row: Mapping[str, object], key: str, default: int = -1) -> int:
    value = row.get(key, default)
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _indexed(rows: Iterable[Mapping[str, str]], key: str) -> Dict[int, Dict[str, str]]:
    indexed: Dict[int, Dict[str, str]] = {}
    for row in rows:
        index = _int(row, key)
        if index < 0:
            continue
        if index in indexed:
            raise ValueError(f"Duplicate {key}={index} in CSV.")
        indexed[index] = dict(row)
    return indexed


def _confusion_key(target: int, prediction: int) -> str:
    return f"{int(target)}->{int(prediction)}"


def _class1_group(
    *,
    target: int,
    softboost_prediction: int,
    class1_index: int,
) -> str:
    if target == class1_index and softboost_prediction != class1_index:
        return "softboost_class1_fn"
    if target != class1_index and softboost_prediction == class1_index:
        return "softboost_class1_fp"
    if target == class1_index and softboost_prediction == class1_index:
        return "softboost_true_class1_correct"
    if target != softboost_prediction:
        return "softboost_other_error"
    return "softboost_non1_correct"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _summarize_rows(rows: Sequence[Mapping[str, object]], class1_index: int) -> Dict[str, object]:
    count = len(rows)
    changed = sum(int(row.get("hflip_changed", 0) or 0) for row in rows)
    clean_correct = sum(int(row.get("clean_correct", 0) or 0) for row in rows)
    flip_correct = sum(int(row.get("flip_correct", 0) or 0) for row in rows)
    softboost_correct = sum(int(row.get("softboost_correct", 0) or 0) for row in rows)
    hflip_corrections = sum(int(row.get("hflip_correction", 0) or 0) for row in rows)
    hflip_harms = sum(int(row.get("hflip_harm", 0) or 0) for row in rows)
    flip_rescues_softboost = sum(
        int(row.get("flip_corrects_softboost_error", 0) or 0) for row in rows
    )
    flip_hurts_softboost = sum(
        int(row.get("flip_breaks_softboost_correct", 0) or 0) for row in rows
    )
    clean_to_class1 = sum(
        1
        for row in rows
        if int(row.get("clean_pred", -1)) != int(class1_index)
        and int(row.get("flip_pred", -1)) == int(class1_index)
    )
    class1_to_other = sum(
        1
        for row in rows
        if int(row.get("clean_pred", -1)) == int(class1_index)
        and int(row.get("flip_pred", -1)) != int(class1_index)
    )
    softboost_class1_errors = sum(
        1
        for row in rows
        if str(row.get("softboost_group", "")).strip()
        in {"softboost_class1_fn", "softboost_class1_fp"}
    )
    entropy_deltas = [
        float(row.get("flip_entropy", 0.0)) - float(row.get("clean_entropy", 0.0))
        for row in rows
    ]
    confidence_deltas = [
        float(row.get("flip_conf", 0.0)) - float(row.get("clean_conf", 0.0))
        for row in rows
    ]
    return {
        "count": int(count),
        "hflip_changed": int(changed),
        "hflip_changed_rate": _safe_ratio(changed, count),
        "clean_correct": int(clean_correct),
        "clean_accuracy": _safe_ratio(clean_correct, count),
        "flip_correct": int(flip_correct),
        "flip_accuracy": _safe_ratio(flip_correct, count),
        "softboost_correct": int(softboost_correct),
        "softboost_accuracy": _safe_ratio(softboost_correct, count),
        "hflip_corrections_clean_wrong_to_right": int(hflip_corrections),
        "hflip_harms_clean_right_to_wrong": int(hflip_harms),
        "flip_corrects_softboost_errors": int(flip_rescues_softboost),
        "flip_breaks_softboost_correct": int(flip_hurts_softboost),
        "clean_non1_to_flip_class1": int(clean_to_class1),
        "clean_class1_to_flip_non1": int(class1_to_other),
        "softboost_class1_error_count": int(softboost_class1_errors),
        "mean_flip_minus_clean_entropy": float(sum(entropy_deltas) / max(1, count)),
        "mean_flip_minus_clean_confidence": float(sum(confidence_deltas) / max(1, count)),
        "clean_confusion_counts": dict(
            Counter(str(row.get("clean_confusion", "")) for row in rows)
        ),
        "flip_confusion_counts": dict(
            Counter(str(row.get("flip_confusion", "")) for row in rows)
        ),
        "softboost_confusion_counts": dict(
            Counter(str(row.get("softboost_confusion", "")) for row in rows)
        ),
    }


def _metrics_from_rows(
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
    class_names: Sequence[str],
) -> Dict[str, object]:
    if not rows:
        return {}
    targets = torch.tensor([int(row["target"]) for row in rows], dtype=torch.long)
    predictions = torch.tensor([int(row[prediction_key]) for row in rows], dtype=torch.long)
    return build_metrics(targets=targets, predictions=predictions, class_names=class_names)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _select_case_rows(rows: Sequence[Mapping[str, object]], top_k: int) -> List[Dict[str, object]]:
    priority = {
        "softboost_class1_fn": 0,
        "softboost_class1_fp": 1,
        "softboost_other_error": 2,
        "softboost_true_class1_correct": 3,
        "softboost_non1_correct": 4,
    }

    def sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            priority.get(str(row.get("softboost_group", "")), 99),
            -int(row.get("hflip_changed", 0) or 0),
            -int(row.get("flip_corrects_softboost_error", 0) or 0),
            int(row.get("sample_index", 0) or 0),
        )

    selected: List[Dict[str, object]] = []
    seen: set[int] = set()
    for row in sorted(rows, key=sort_key):
        sample_index = int(row.get("sample_index", -1))
        if sample_index in seen:
            continue
        selected.append(dict(row))
        seen.add(sample_index)
        if len(selected) >= max(0, int(top_k)):
            break
    return selected


def audit_instability(
    *,
    hflip_predictions: Path,
    softboost_predictions: Path,
    output_dir: Path,
    class1_index: int = CLASS1_INDEX,
    top_k_cases: int = 80,
) -> Dict[str, object]:
    ensure_dir(output_dir)
    hflip_rows = _read_csv(hflip_predictions)
    softboost_rows = _read_csv(softboost_predictions)
    hflip_by_index = _indexed(hflip_rows, "index")
    softboost_by_index = _indexed(softboost_rows, "sample_index")

    shared_indices = sorted(set(hflip_by_index).intersection(softboost_by_index))
    if not shared_indices:
        raise ValueError("No shared index/sample_index rows between input CSVs.")

    class_count = 0
    for row in softboost_rows:
        for key in row:
            if key.startswith("prob_"):
                parts = key.split("_", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    class_count = max(class_count, int(parts[1]) + 1)
    if class_count <= 0:
        class_count = 5
    class_names = [f"class_{index}" for index in range(class_count)]
    for row in softboost_rows:
        for index in range(class_count):
            prefix = f"prob_{index}_"
            for key in row:
                if key.startswith(prefix):
                    class_names[index] = key[len(prefix) :]
                    break

    joined_rows: List[Dict[str, object]] = []
    target_mismatches: List[int] = []
    for index in shared_indices:
        hflip = hflip_by_index[index]
        softboost = softboost_by_index[index]
        target = _int(softboost, "target_index")
        hflip_target = _int(hflip, "target")
        if target != hflip_target:
            target_mismatches.append(index)
        clean_pred = _int(hflip, "clean_pred")
        flip_pred = _int(hflip, "flip_pred")
        softboost_pred = _int(softboost, "prediction_index")
        clean_correct = int(clean_pred == target)
        flip_correct = int(flip_pred == target)
        softboost_correct = int(softboost_pred == target)
        group = _class1_group(
            target=target,
            softboost_prediction=softboost_pred,
            class1_index=int(class1_index),
        )
        row: Dict[str, object] = {
            "sample_index": int(index),
            "image_path": softboost.get("image_path", ""),
            "target": int(target),
            "softboost_pred": int(softboost_pred),
            "clean_pred": int(clean_pred),
            "flip_pred": int(flip_pred),
            "clean_conf": _float(hflip, "clean_conf"),
            "flip_conf": _float(hflip, "flip_conf"),
            "clean_entropy": _float(hflip, "clean_entropy"),
            "flip_entropy": _float(hflip, "flip_entropy"),
            "softboost_conf": _float(softboost, "confidence"),
            "softboost_group": group,
            "hflip_changed": int(clean_pred != flip_pred),
            "clean_correct": int(clean_correct),
            "flip_correct": int(flip_correct),
            "softboost_correct": int(softboost_correct),
            "hflip_correction": int((not clean_correct) and flip_correct),
            "hflip_harm": int(clean_correct and (not flip_correct)),
            "flip_corrects_softboost_error": int((not softboost_correct) and flip_correct),
            "flip_breaks_softboost_correct": int(softboost_correct and (not flip_correct)),
            "clean_confusion": _confusion_key(target, clean_pred),
            "flip_confusion": _confusion_key(target, flip_pred),
            "softboost_confusion": _confusion_key(target, softboost_pred),
        }
        for class_index in range(class_count):
            key = f"prob_{class_index}_{class_names[class_index]}"
            if key in softboost:
                row[f"softboost_prob_{class_index}"] = _float(softboost, key)
        joined_rows.append(row)

    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in joined_rows:
        groups[str(row["softboost_group"])].append(row)

    group_summary = {
        name: _summarize_rows(rows, int(class1_index))
        for name, rows in sorted(groups.items())
    }
    overall_summary = _summarize_rows(joined_rows, int(class1_index))
    true_class1_correct = group_summary.get("softboost_true_class1_correct", {})
    non1_correct = group_summary.get("softboost_non1_correct", {})
    unsafe_hflip_signal = (
        int(overall_summary.get("flip_breaks_softboost_correct", 0))
        >= int(overall_summary.get("flip_corrects_softboost_errors", 0))
        or int(true_class1_correct.get("flip_breaks_softboost_correct", 0)) > 0
        or int(non1_correct.get("clean_non1_to_flip_class1", 0)) > 0
    )
    if unsafe_hflip_signal:
        route_assessment = "reject_current_hflip_instability_as_training_signal"
        recommendation = (
            "Do not enable hflip TTA aggregation, hflip consistency loss, or "
            "a hflip-instability train policy on the current keeper. The "
            "signal is not recall-safe: it breaks correct softboost rows and "
            "can create non1->1 false positives."
        )
    else:
        route_assessment = "diagnostic_only_needs_followup"
        recommendation = (
            "Use hflip instability only as evidence. Do not enable hflip TTA "
            "aggregation or train a consistency loss unless a separate "
            "train-only/fold-safe audit proves recall-safe class-1 FN rescue "
            "and conservative 0/2/4->1 false-positive control."
        )

    summary: Dict[str, object] = {
        "inputs": {
            "hflip_predictions": str(hflip_predictions),
            "softboost_predictions": str(softboost_predictions),
        },
        "support": int(len(joined_rows)),
        "shared_indices": int(len(shared_indices)),
        "hflip_rows": int(len(hflip_rows)),
        "softboost_rows": int(len(softboost_rows)),
        "target_mismatch_count": int(len(target_mismatches)),
        "target_mismatch_preview": [int(value) for value in target_mismatches[:10]],
        "class1_index": int(class1_index),
        "class_names": list(class_names),
        "metrics": {
            "clean": _metrics_from_rows(joined_rows, "clean_pred", class_names),
            "flip": _metrics_from_rows(joined_rows, "flip_pred", class_names),
            "softboost": _metrics_from_rows(joined_rows, "softboost_pred", class_names),
        },
        "overall": overall_summary,
        "groups": group_summary,
        "hflip_changed_clean_confusions": dict(
            Counter(
                str(row["clean_confusion"])
                for row in joined_rows
                if int(row.get("hflip_changed", 0) or 0)
            )
        ),
        "hflip_changed_softboost_confusions": dict(
            Counter(
                str(row["softboost_confusion"])
                for row in joined_rows
                if int(row.get("hflip_changed", 0) or 0)
            )
        ),
        "decision": {
            "status": route_assessment,
            "training_manifest_written": False,
            "test_split_used": False,
            "raw_dataset_touched": False,
            "recommendation": recommendation,
        },
    }

    case_rows = _select_case_rows(joined_rows, top_k_cases)
    _write_csv(output_dir / "hflip_softboost_instability_cases.csv", case_rows)
    _write_csv(output_dir / "hflip_softboost_instability_all.csv", joined_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme = [
        "# HFlip Decision Instability Audit",
        "",
        "Diagnostic-only audit over existing validation artifacts.",
        "",
        f"- support: {len(joined_rows)}",
        f"- hflip changed rows: {summary['overall']['hflip_changed']}",
        f"- target mismatches: {len(target_mismatches)}",
        "- test split used: false",
        "- raw dataset touched: false",
        "- train manifest written: false",
        "",
        "Decision rule: do not use this artifact as a train policy. It only",
        "checks whether clean-vs-hflip instability contains a recall-safe",
        "class-1 signal worth investigating further.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = audit_instability(
        hflip_predictions=args.hflip_predictions,
        softboost_predictions=args.softboost_predictions,
        output_dir=args.output_dir,
        class1_index=int(args.class1_index),
        top_k_cases=int(args.top_k_cases),
    )
    print(
        json.dumps(
            {
                "support": summary["support"],
                "hflip_changed": summary["overall"]["hflip_changed"],
                "target_mismatch_count": summary["target_mismatch_count"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
