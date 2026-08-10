from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from trkh.data.dataset import PairedViewTrainDataset
from trkh.tools.remap_classification_teacher_to_yolo import (
    _build_yolo_dataset,
    _read_csv,
    _resolve_source_class_names,
    _teacher_probabilities_by_key,
)


def _expert_directories(root: Path, expert_names: str = "") -> List[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Expert root not found: {root}")
    names = [part.strip() for part in str(expert_names or "").split(",") if part.strip()]
    if names:
        directories = [root / name for name in names]
    else:
        directories = [path for path in sorted(root.iterdir()) if path.is_dir()]
    missing = [str(path) for path in directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing expert directories: {missing}")
    if len(directories) < 2:
        raise ValueError("Need at least two expert directories to measure vote disagreement.")
    return directories


def _prediction_columns(num_classes: int) -> List[str]:
    return [f"prob_{index}" for index in range(int(num_classes))]


def _prediction_from_probabilities(probabilities: Sequence[float]) -> int:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return int(max(range(len(probabilities)), key=lambda index: float(probabilities[index])))


def _load_expert_probabilities(
    expert_dir: Path,
    *,
    dataset,
    target_class_names: Sequence[str],
    split: str,
) -> Tuple[List[List[float]], Dict[str, object]]:
    prediction_csv = Path(expert_dir) / f"predictions_{split}.csv"
    metrics_json = Path(expert_dir) / f"metrics_{split}.json"
    rows, fieldnames = _read_csv(prediction_csv)
    source_names = _resolve_source_class_names(
        fieldnames=fieldnames,
        source_class_names="",
        source_metrics_json=metrics_json if metrics_json.is_file() else None,
        class_count=len(target_class_names),
    )
    teacher_by_key, teacher_summary = _teacher_probabilities_by_key(
        rows,
        fieldnames=fieldnames,
        source_class_names=source_names,
        target_class_names=target_class_names,
    )

    probabilities_by_sample: List[List[float]] = []
    missing: List[Dict[str, object]] = []
    for sample_index in range(len(dataset)):
        source_id, object_index, fallback_label = PairedViewTrainDataset._sample_key(
            dataset,
            int(sample_index),
        )
        probabilities = teacher_by_key.get((str(source_id), int(object_index)))
        if probabilities is None:
            missing.append(
                {
                    "sample_index": int(sample_index),
                    "source_stem": str(source_id),
                    "object_index": int(object_index),
                    "fallback_label": int(fallback_label),
                }
            )
            probabilities = [1.0 / float(len(target_class_names)) for _ in target_class_names]
        probabilities_by_sample.append([float(value) for value in probabilities])

    summary = {
        "expert": str(Path(expert_dir).name),
        "prediction_csv": str(prediction_csv.resolve()),
        "metrics_json": str(metrics_json.resolve()) if metrics_json.is_file() else "",
        "mapped_samples": int(len(probabilities_by_sample) - len(missing)),
        "missing_samples": int(len(missing)),
        "missing_examples": missing[:10],
        **teacher_summary,
    }
    return probabilities_by_sample, summary


def _mean_probabilities(probabilities: Sequence[Sequence[float]], sample_index: int) -> List[float]:
    if not probabilities:
        return []
    class_count = len(probabilities[0][sample_index])
    means: List[float] = []
    for class_index in range(class_count):
        means.append(
            float(
                sum(float(expert_probs[sample_index][class_index]) for expert_probs in probabilities)
                / max(1, len(probabilities))
            )
        )
    total = sum(max(0.0, value) for value in means)
    if total > 0.0:
        means = [float(max(0.0, value) / total) for value in means]
    return means


def _majority_vote(
    predictions: Sequence[int],
    *,
    mean_probabilities: Sequence[float],
    num_classes: int,
) -> Tuple[int, int, Tuple[int, ...], Dict[int, int]]:
    counts = Counter(int(prediction) for prediction in predictions)
    if not counts:
        raise ValueError("predictions must not be empty")
    candidates = list(range(int(num_classes)))
    majority = max(
        candidates,
        key=lambda index: (
            int(counts.get(index, 0)),
            float(mean_probabilities[index]) if index < len(mean_probabilities) else 0.0,
            -int(index),
        ),
    )
    pattern = tuple(sorted((int(value) for value in counts.values()), reverse=True))
    return int(majority), int(counts.get(majority, 0)), pattern, dict(counts)


def _weight_for_vote_pattern(
    *,
    target_index: int,
    majority_index: int,
    vote_counts: Mapping[int, int],
    num_experts: int,
    focus_class: int = 1,
    class1_vote_weight: float = 0.82,
    class1_vote_strong_weight: float = 0.72,
    majority_conflict_weight: float = 0.80,
    other_disagreement_weight: float = 0.92,
    focus_disagreement_weight: float = 1.0,
    protect_focus_class: bool = True,
) -> Tuple[float, str]:
    target_index = int(target_index)
    majority_index = int(majority_index)
    focus_class = int(focus_class)
    target_votes = int(vote_counts.get(target_index, 0))
    focus_votes = int(vote_counts.get(focus_class, 0))
    majority_votes = int(vote_counts.get(majority_index, 0))
    if majority_votes >= int(num_experts):
        return 1.0, "unanimous"
    if target_index == focus_class:
        if bool(protect_focus_class):
            return 1.0, "focus_class_recall_protected"
        return float(focus_disagreement_weight), "focus_class_disagreement"
    if focus_votes > 0:
        if majority_index != target_index:
            return (
                float(min(class1_vote_strong_weight, majority_conflict_weight)),
                "nonfocus_majority_conflict_class1_involved",
            )
        if focus_votes >= max(2, int(math.ceil(float(num_experts) * 0.4))):
            return float(class1_vote_strong_weight), "nonfocus_multi_class1_votes"
        return float(class1_vote_weight), "nonfocus_single_class1_vote"
    if majority_index != target_index:
        return float(majority_conflict_weight), "nonfocus_majority_conflict"
    if target_votes < int(num_experts):
        return float(other_disagreement_weight), "nonfocus_other_disagreement"
    return 1.0, "unanimous"


def _format_prob(value: float) -> str:
    return f"{float(value):.10g}"


def build_manifest(
    *,
    data: Path,
    expert_root: Path,
    output_dir: Path,
    split: str = "train",
    expert_names: str = "",
    class_name_mode: str = "raw",
    expected_num_classes: int = 5,
    focus_class: int = 1,
    class1_vote_weight: float = 0.82,
    class1_vote_strong_weight: float = 0.72,
    majority_conflict_weight: float = 0.80,
    other_disagreement_weight: float = 0.92,
    focus_disagreement_weight: float = 1.0,
    protect_focus_class: bool = True,
    default_weight: float = 1.0,
    write_default_rows: bool = False,
    diagnostic_only: bool = False,
) -> Dict[str, object]:
    split_name = str(split or "train").strip().lower()
    if split_name != "train" and not bool(diagnostic_only):
        raise ValueError("Top-5 vote-disagreement sample weights must be built from train split only.")
    dataset, target_class_names = _build_yolo_dataset(
        yolo_data=Path(data),
        split=split_name,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_dirs = _expert_directories(Path(expert_root), expert_names=expert_names)
    expert_probabilities: List[List[List[float]]] = []
    expert_summaries: List[Dict[str, object]] = []
    for expert_dir in expert_dirs:
        probabilities, expert_summary = _load_expert_probabilities(
            expert_dir,
            dataset=dataset,
            target_class_names=target_class_names,
            split=split_name,
        )
        expert_probabilities.append(probabilities)
        expert_summaries.append(expert_summary)

    labels = [int(label) for label in dataset.labels()]
    paths = [Path(path) for path in dataset.sample_paths()]
    expert_names_out = [Path(path).name for path in expert_dirs]
    manifest_path = output_dir / "top5_vote_disagreement_sample_weights_train_only.csv"
    diagnostic_path = output_dir / f"top5_vote_disagreement_{split_name}_diagnostic.csv"

    manifest_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    vote_pattern_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    weighted_target_counter: Counter[str] = Counter()
    weighted_pair_counter: Counter[str] = Counter()
    disagreement_rows = 0
    focus_protected_rows = 0
    target_vote_totals: List[int] = []
    mean_confidences: List[float] = []
    weights_written: List[float] = []

    for sample_index in range(len(dataset)):
        target_index = int(labels[sample_index])
        target_counter[str(target_index)] += 1
        per_expert_probs = [expert_probs[sample_index] for expert_probs in expert_probabilities]
        predictions = [_prediction_from_probabilities(probs) for probs in per_expert_probs]
        confidences = [float(max(probs)) for probs in per_expert_probs]
        mean_probs = _mean_probabilities(expert_probabilities, sample_index)
        majority_index, majority_votes, pattern, vote_counts = _majority_vote(
            predictions,
            mean_probabilities=mean_probs,
            num_classes=len(target_class_names),
        )
        target_votes = int(vote_counts.get(target_index, 0))
        focus_votes = int(vote_counts.get(int(focus_class), 0))
        pattern_text = "-".join(str(value) for value in pattern)
        vote_pattern_counter[pattern_text] += 1
        target_vote_totals.append(target_votes)
        mean_confidences.append(float(sum(confidences) / max(1, len(confidences))))
        weight, reason = _weight_for_vote_pattern(
            target_index=target_index,
            majority_index=majority_index,
            vote_counts=vote_counts,
            num_experts=len(expert_probabilities),
            focus_class=int(focus_class),
            class1_vote_weight=float(class1_vote_weight),
            class1_vote_strong_weight=float(class1_vote_strong_weight),
            majority_conflict_weight=float(majority_conflict_weight),
            other_disagreement_weight=float(other_disagreement_weight),
            focus_disagreement_weight=float(focus_disagreement_weight),
            protect_focus_class=bool(protect_focus_class),
        )
        reason_counter[reason] += 1
        is_disagreement = majority_votes < len(expert_probabilities)
        if is_disagreement:
            disagreement_rows += 1
        if reason == "focus_class_recall_protected":
            focus_protected_rows += 1
        row = {
            "sample_index": int(sample_index),
            "path": str(paths[sample_index]),
            "image_path": str(paths[sample_index]),
            "target_index": int(target_index),
            "target_name": str(target_class_names[target_index]),
            "majority_index": int(majority_index),
            "majority_name": str(target_class_names[majority_index]),
            "majority_votes": int(majority_votes),
            "target_votes": int(target_votes),
            "focus_votes": int(focus_votes),
            "vote_pattern": pattern_text,
            "sample_weight": _format_prob(weight),
            "reason": reason,
            "mean_confidence": _format_prob(float(sum(confidences) / max(1, len(confidences)))),
            "mean_target_probability": _format_prob(mean_probs[target_index]),
            "mean_focus_probability": _format_prob(mean_probs[int(focus_class)]),
            "expert_predictions": "|".join(str(int(value)) for value in predictions),
            "expert_confidences": "|".join(_format_prob(value) for value in confidences),
            "experts": "|".join(expert_names_out),
        }
        for class_index, probability in enumerate(mean_probs):
            row[f"mean_prob_{class_index}"] = _format_prob(probability)
        if is_disagreement:
            diagnostic_rows.append(row)
        if bool(write_default_rows) or abs(float(weight) - float(default_weight)) > 1e-12:
            if not bool(diagnostic_only):
                manifest_rows.append(row)
                weighted_target_counter[str(target_index)] += 1
                weighted_pair_counter[f"{target_index}->{majority_index}"] += 1
                weights_written.append(float(weight))

    diagnostic_fields = [
        "sample_index",
        "path",
        "image_path",
        "target_index",
        "target_name",
        "majority_index",
        "majority_name",
        "majority_votes",
        "target_votes",
        "focus_votes",
        "vote_pattern",
        "sample_weight",
        "reason",
        "mean_confidence",
        "mean_target_probability",
        "mean_focus_probability",
        "expert_predictions",
        "expert_confidences",
        "experts",
    ] + _prediction_columns(len(target_class_names))
    diagnostic_fields = [
        field.replace("prob_", "mean_prob_") if field.startswith("prob_") else field
        for field in diagnostic_fields
    ]
    with diagnostic_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    if not bool(diagnostic_only):
        if not manifest_rows:
            raise ValueError("No sample-weight rows were written; check weights or write-default-rows.")
        manifest_fields = [
            "sample_index",
            "image_path",
            "path",
            "target_index",
            "target_name",
            "prediction_index",
            "prediction_name",
            "sample_weight",
            "reason",
            "majority_votes",
            "target_votes",
            "focus_votes",
            "vote_pattern",
            "mean_confidence",
            "mean_target_probability",
            "mean_focus_probability",
            "expert_predictions",
            "expert_confidences",
            "experts",
        ]
        manifest_output_rows: List[Dict[str, object]] = []
        for row in manifest_rows:
            out = dict(row)
            out["prediction_index"] = int(row["majority_index"])
            out["prediction_name"] = str(row["majority_name"])
            manifest_output_rows.append(out)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows([{key: row.get(key, "") for key in manifest_fields} for row in manifest_output_rows])

    summary = {
        "data_yaml": str(Path(data).resolve()),
        "expert_root": str(Path(expert_root).resolve()),
        "split": split_name,
        "diagnostic_only": bool(diagnostic_only),
        "output_manifest": str(manifest_path.resolve()) if not bool(diagnostic_only) else "",
        "output_diagnostic": str(diagnostic_path.resolve()),
        "samples": int(len(dataset)),
        "experts": expert_names_out,
        "num_experts": int(len(expert_probabilities)),
        "target_class_names": [str(name) for name in target_class_names],
        "focus_class": int(focus_class),
        "policy": {
            "class1_vote_weight": float(class1_vote_weight),
            "class1_vote_strong_weight": float(class1_vote_strong_weight),
            "majority_conflict_weight": float(majority_conflict_weight),
            "other_disagreement_weight": float(other_disagreement_weight),
            "focus_disagreement_weight": float(focus_disagreement_weight),
            "protect_focus_class": bool(protect_focus_class),
            "default_weight": float(default_weight),
            "write_default_rows": bool(write_default_rows),
        },
        "disagreement_rows": int(disagreement_rows),
        "manifest_rows": int(len(manifest_rows)),
        "focus_protected_rows": int(focus_protected_rows),
        "target_counts": dict(sorted(target_counter.items())),
        "weighted_target_counts": dict(sorted(weighted_target_counter.items())),
        "weighted_target_majority_pairs": dict(sorted(weighted_pair_counter.items())),
        "reason_counts": dict(sorted(reason_counter.items())),
        "vote_pattern_counts": dict(sorted(vote_pattern_counter.items())),
        "mean_target_votes": float(sum(target_vote_totals) / max(1, len(target_vote_totals))),
        "mean_expert_confidence": float(sum(mean_confidences) / max(1, len(mean_confidences))),
        "mean_manifest_weight": float(sum(weights_written) / max(1, len(weights_written))) if weights_written else 0.0,
        "min_manifest_weight": float(min(weights_written)) if weights_written else 0.0,
        "max_manifest_weight": float(max(weights_written)) if weights_written else 0.0,
        "expert_summaries": expert_summaries,
        "leakage_guard": (
            "sample weights are written only for train split by sample_index; "
            "non-train splits require diagnostic_only and produce no manifest"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only sample weights from individual top-5 pretrained expert "
            "vote disagreement after strict remap to YOLO sample_index order."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--expert-names", type=str, default="")
    parser.add_argument("--class-name-mode", choices=("raw", "mango", "auto"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--class1-vote-weight", type=float, default=0.82)
    parser.add_argument("--class1-vote-strong-weight", type=float, default=0.72)
    parser.add_argument("--majority-conflict-weight", type=float, default=0.80)
    parser.add_argument("--other-disagreement-weight", type=float, default=0.92)
    parser.add_argument("--focus-disagreement-weight", type=float, default=1.0)
    parser.add_argument("--disable-focus-protection", action="store_true", default=False)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--write-default-rows", action="store_true", default=False)
    parser.add_argument("--diagnostic-only", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    build_manifest(
        data=Path(args.data),
        expert_root=Path(args.expert_root),
        output_dir=Path(args.output_dir),
        split=str(args.split),
        expert_names=str(args.expert_names),
        class_name_mode=str(args.class_name_mode),
        expected_num_classes=int(args.expected_num_classes),
        focus_class=int(args.focus_class),
        class1_vote_weight=float(args.class1_vote_weight),
        class1_vote_strong_weight=float(args.class1_vote_strong_weight),
        majority_conflict_weight=float(args.majority_conflict_weight),
        other_disagreement_weight=float(args.other_disagreement_weight),
        focus_disagreement_weight=float(args.focus_disagreement_weight),
        protect_focus_class=not bool(args.disable_focus_protection),
        default_weight=float(args.default_weight),
        write_default_rows=bool(args.write_default_rows),
        diagnostic_only=bool(args.diagnostic_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
