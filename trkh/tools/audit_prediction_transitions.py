from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REQUIRED_COLUMNS = {
    "sample_index",
    "image_path",
    "target_index",
    "prediction_index",
}


def _read_predictions(path: Path) -> tuple[Dict[int, Dict[str, str]], Sequence[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows: Dict[int, Dict[str, str]] = {}
        for row in reader:
            sample_index = int(row["sample_index"])
            if sample_index in rows:
                raise ValueError(f"Duplicate sample_index {sample_index} in {path}")
            rows[sample_index] = dict(row)
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    return rows, fieldnames


def _focus_probability_column(fieldnames: Iterable[str], focus_class: int) -> str | None:
    prefix = f"prob_{int(focus_class)}_"
    matches = [name for name in fieldnames if name.startswith(prefix)]
    if len(matches) > 1:
        raise ValueError(f"Multiple focus probability columns match {prefix}: {matches}")
    if matches:
        return matches[0]
    fallback = f"prob_{int(focus_class)}"
    return fallback if fallback in fieldnames else None


def _transition_outcome(target: int, base: int, candidate: int) -> str:
    if base == candidate:
        return "unchanged"
    if base != target and candidate == target:
        return "correction"
    if base == target and candidate != target:
        return "harm"
    return "wrong_to_wrong"


def audit_prediction_transitions(
    *,
    base_predictions: Path,
    candidate_predictions: Path,
    output_dir: Path,
    focus_class: int = 1,
    base_name: str = "base",
    candidate_name: str = "candidate",
) -> Mapping[str, object]:
    base_rows, base_fields = _read_predictions(base_predictions)
    candidate_rows, candidate_fields = _read_predictions(candidate_predictions)
    if set(base_rows) != set(candidate_rows):
        missing_from_candidate = sorted(set(base_rows).difference(candidate_rows))
        missing_from_base = sorted(set(candidate_rows).difference(base_rows))
        raise ValueError(
            "Prediction sample indices do not align: "
            f"missing_from_candidate={missing_from_candidate[:10]} "
            f"missing_from_base={missing_from_base[:10]}"
        )

    base_focus_column = _focus_probability_column(base_fields, focus_class)
    candidate_focus_column = _focus_probability_column(candidate_fields, focus_class)
    outcome_counts: Counter[str] = Counter()
    prediction_transitions: Counter[str] = Counter()
    changed_rows: List[Dict[str, object]] = []
    focus_counts = Counter(
        {
            "base_tp": 0,
            "base_fp": 0,
            "base_fn": 0,
            "candidate_tp": 0,
            "candidate_fp": 0,
            "candidate_fn": 0,
            "fn_rescued": 0,
            "tp_broken": 0,
            "fp_removed": 0,
            "fp_created": 0,
        }
    )

    for sample_index in sorted(base_rows):
        base_row = base_rows[sample_index]
        candidate_row = candidate_rows[sample_index]
        base_path = str(Path(base_row["image_path"]).resolve()).casefold()
        candidate_path = str(Path(candidate_row["image_path"]).resolve()).casefold()
        base_target = int(base_row["target_index"])
        candidate_target = int(candidate_row["target_index"])
        if base_path != candidate_path or base_target != candidate_target:
            raise ValueError(
                f"Row mismatch at sample_index {sample_index}: "
                f"base_path={base_row['image_path']} candidate_path={candidate_row['image_path']} "
                f"base_target={base_target} candidate_target={candidate_target}"
            )

        base_prediction = int(base_row["prediction_index"])
        candidate_prediction = int(candidate_row["prediction_index"])
        target_is_focus = base_target == int(focus_class)
        base_is_focus = base_prediction == int(focus_class)
        candidate_is_focus = candidate_prediction == int(focus_class)
        focus_counts["base_tp"] += int(target_is_focus and base_is_focus)
        focus_counts["base_fp"] += int((not target_is_focus) and base_is_focus)
        focus_counts["base_fn"] += int(target_is_focus and not base_is_focus)
        focus_counts["candidate_tp"] += int(target_is_focus and candidate_is_focus)
        focus_counts["candidate_fp"] += int((not target_is_focus) and candidate_is_focus)
        focus_counts["candidate_fn"] += int(target_is_focus and not candidate_is_focus)
        focus_counts["fn_rescued"] += int(
            target_is_focus and not base_is_focus and candidate_is_focus
        )
        focus_counts["tp_broken"] += int(
            target_is_focus and base_is_focus and not candidate_is_focus
        )
        focus_counts["fp_removed"] += int(
            (not target_is_focus) and base_is_focus and not candidate_is_focus
        )
        focus_counts["fp_created"] += int(
            (not target_is_focus) and not base_is_focus and candidate_is_focus
        )

        outcome = _transition_outcome(base_target, base_prediction, candidate_prediction)
        outcome_counts[outcome] += 1
        if outcome == "unchanged":
            continue
        transition = f"{base_prediction}->{candidate_prediction}"
        prediction_transitions[transition] += 1
        base_focus_probability = (
            float(base_row[base_focus_column]) if base_focus_column else None
        )
        candidate_focus_probability = (
            float(candidate_row[candidate_focus_column])
            if candidate_focus_column
            else None
        )
        changed_rows.append(
            {
                "sample_index": sample_index,
                "image_path": base_row["image_path"],
                "target_index": base_target,
                "base_prediction_index": base_prediction,
                "candidate_prediction_index": candidate_prediction,
                "outcome": outcome,
                "transition": transition,
                "base_correct": int(base_prediction == base_target),
                "candidate_correct": int(candidate_prediction == base_target),
                "base_focus_probability": base_focus_probability,
                "candidate_focus_probability": candidate_focus_probability,
                "focus_probability_delta": (
                    candidate_focus_probability - base_focus_probability
                    if base_focus_probability is not None
                    and candidate_focus_probability is not None
                    else None
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    changed_path = output_dir / "changed_predictions.csv"
    changed_fields = [
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction_index",
        "candidate_prediction_index",
        "outcome",
        "transition",
        "base_correct",
        "candidate_correct",
        "base_focus_probability",
        "candidate_focus_probability",
        "focus_probability_delta",
    ]
    with changed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=changed_fields)
        writer.writeheader()
        writer.writerows(changed_rows)

    total = len(base_rows)
    summary: Dict[str, object] = {
        "mode": "strict_prediction_transition_audit",
        "base_name": str(base_name),
        "candidate_name": str(candidate_name),
        "base_predictions": str(base_predictions.resolve()),
        "candidate_predictions": str(candidate_predictions.resolve()),
        "focus_class": int(focus_class),
        "rows": total,
        "changed": len(changed_rows),
        "unchanged": int(outcome_counts["unchanged"]),
        "corrections": int(outcome_counts["correction"]),
        "harms": int(outcome_counts["harm"]),
        "wrong_to_wrong": int(outcome_counts["wrong_to_wrong"]),
        "net_corrections": int(outcome_counts["correction"] - outcome_counts["harm"]),
        "focus": dict(focus_counts),
        "prediction_transitions": dict(
            sorted(prediction_transitions.items(), key=lambda item: (-item[1], item[0]))
        ),
        "changed_predictions_csv": str(changed_path.resolve()),
        "raw_dataset_touched": False,
        "test_split_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly align two prediction CSVs and audit changed decisions."
    )
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--candidate-name", default="candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_prediction_transitions(
        base_predictions=args.base_predictions,
        candidate_predictions=args.candidate_predictions,
        output_dir=args.output_dir,
        focus_class=args.focus_class,
        base_name=args.base_name,
        candidate_name=args.candidate_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
