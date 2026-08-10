from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from trkh.tools.analyze_expert_complementarity import (
    PredictionTable,
    _load_prediction_table,
    _sorted_keys,
)


@dataclass(frozen=True)
class InputSpec:
    family: str
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict validation-only information-ceiling audit across aligned model "
            "prediction CSVs. Oracle outputs use labels and are never deployable metrics."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="FAMILY:NAME=CSV",
        help="Aligned validation prediction source; repeat for every model.",
    )
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=2606)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--target-focus-f1", type=float, default=0.98)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument(
        "--bootstrap-mode",
        choices=("source_group", "stratified_sample"),
        default="source_group",
    )
    return parser.parse_args()


def parse_input_spec(value: str) -> InputSpec:
    if "=" not in value:
        raise ValueError(f"Input must be FAMILY:NAME=CSV, got {value!r}")
    left, path_text = value.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Input must include a family, got {value!r}")
    family, name = (part.strip() for part in left.split(":", 1))
    path = Path(path_text.strip())
    if not family or not name:
        raise ValueError(f"Input family/name cannot be empty: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found for {name}: {path}")
    return InputSpec(family=family, name=name, path=path)


def _split_from_path(path_text: str) -> str | None:
    parts = [part for part in re.split(r"[\\/]", str(path_text).lower()) if part]
    found = [split for split in ("train", "val", "test") if split in parts]
    if len(found) > 1:
        raise ValueError(f"Ambiguous dataset split in path: {path_text}")
    return found[0] if found else None


def _source_group(row: Mapping[str, str], image_path: str, key: str) -> str:
    source_stem = str(row.get("source_stem", "") or "").strip().lower()
    if source_stem:
        return source_stem
    if image_path:
        return Path(str(image_path)).stem.lower()
    return key.lower()


def _metric_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoded = targets.astype(np.int64) * int(class_count) + predictions.astype(np.int64)
    matrix = np.bincount(encoded, minlength=class_count * class_count).reshape(
        class_count, class_count
    )
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_support = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(tp),
        where=denominator > 0,
    )
    total = int(matrix.sum())
    accuracy = float(tp.sum() / total) if total else 0.0
    macro_f1 = float(f1.mean()) if f1.size else 0.0
    return accuracy, macro_f1, precision, recall, f1, matrix


def _metrics_payload(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    accuracy, macro_f1, precision, recall, f1, matrix = _metric_arrays(
        targets, predictions, len(class_names)
    )
    per_class = []
    for index, class_name in enumerate(class_names):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_support = int(matrix[:, index].sum())
        per_class.append(
            {
                "class_index": int(index),
                "class_name": str(class_name),
                "support": support,
                "predicted_support": predicted_support,
                "tp": tp,
                "fp": int(predicted_support - tp),
                "fn": int(support - tp),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    return {
        "samples": int(targets.size),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _focus_payload(metrics: Mapping[str, object], focus_class: int) -> Dict[str, object]:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, list) or not 0 <= focus_class < len(per_class):
        raise ValueError(f"Focus class {focus_class} is absent from metrics")
    return dict(per_class[focus_class])


def _focus_metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, object]:
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _minimum_focus_corrections(
    tp: int,
    fp: int,
    fn: int,
    target_f1: float,
) -> Dict[str, object]:
    candidates: list[tuple[int, int, float]] = []
    for corrected_fn in range(fn + 1):
        for corrected_fp in range(fp + 1):
            metrics = _focus_metrics_from_counts(
                tp + corrected_fn,
                fp - corrected_fp,
                fn - corrected_fn,
            )
            if float(metrics["f1"]) + 1e-12 >= float(target_f1):
                candidates.append((corrected_fn, corrected_fp, float(metrics["f1"])))
    if not candidates:
        return {"reachable": False}
    candidates.sort(key=lambda item: (item[0] + item[1], max(item[0], item[1]), item[0]))
    corrected_fn, corrected_fp, achieved_f1 = candidates[0]
    return {
        "reachable": True,
        "minimum_total_corrections": int(corrected_fn + corrected_fp),
        "corrected_fn": int(corrected_fn),
        "corrected_fp": int(corrected_fp),
        "achieved_f1": achieved_f1,
        "fraction_of_focus_errors": float((corrected_fn + corrected_fp) / max(1, fn + fp)),
    }


def _mcnemar_exact_pvalue(corrections: int, harms: int) -> float:
    discordant = int(corrections + harms)
    if discordant == 0:
        return 1.0
    tail = min(int(corrections), int(harms))
    numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
    return float(min(1.0, (2 * numerator) / (2**discordant)))


def _percentile_summary(values: np.ndarray, point: float) -> Dict[str, float]:
    return {
        "point": float(point),
        "bootstrap_mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def _bootstrap_draws(
    *,
    targets: np.ndarray,
    source_groups: Sequence[str],
    iterations: int,
    seed: int,
    mode: str,
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    if mode == "stratified_sample":
        class_indices = [np.flatnonzero(targets == class_index) for class_index in np.unique(targets)]
        for _ in range(int(iterations)):
            yield np.concatenate(
                [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices]
            )
        return

    groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(source_groups):
        groups[str(group)].append(index)
    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for group, indices in groups.items():
        signature = tuple(sorted(set(int(targets[index]) for index in indices)))
        strata[signature].append(group)
    for _ in range(int(iterations)):
        selected: list[int] = []
        for group_names in strata.values():
            sampled = rng.choice(group_names, size=len(group_names), replace=True)
            for group in sampled.tolist():
                selected.extend(groups[str(group)])
        yield np.asarray(selected, dtype=np.int64)


def _safe_column(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return cleaned or "model"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    def render(value: object) -> str:
        if isinstance(value, float):
            value = f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(field) for field in fieldnames]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(render(row.get(field, "")) for field in headers) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_tables(
    specs: Sequence[InputSpec],
    tables: Mapping[str, PredictionTable],
    base_name: str,
    expected_samples: int,
) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    if base_name not in tables:
        raise ValueError(f"Base model {base_name!r} is not present in --input")
    base = tables[base_name]
    keys = _sorted_keys(base.rows)
    if expected_samples > 0 and len(keys) != int(expected_samples):
        raise ValueError(f"Expected {expected_samples} validation rows, found {len(keys)}")
    if keys and all(key.startswith("sample:") for key in keys):
        actual_indices = [int(key.split(":", 1)[1]) for key in keys]
        expected_indices = list(range(len(keys)))
        if actual_indices != expected_indices:
            raise ValueError("Base sample_index keys are not contiguous from 0")

    class_names = list(base.class_names)
    targets = []
    source_groups = []
    for key in keys:
        if key not in base.targets:
            raise ValueError(f"Base row {key} has no target")
        targets.append(int(base.targets[key]))
        row = base.rows[key]
        image_path = str(base.image_paths.get(key, "") or "")
        split = _split_from_path(image_path)
        if split != "val":
            raise ValueError(
                f"Validation-only guard rejected base row {key}: split={split!r}, path={image_path!r}"
            )
        source_groups.append(_source_group(row, image_path, key))
    target_array = np.asarray(targets, dtype=np.int64)

    base_key_set = set(keys)
    for spec in specs:
        table = tables[spec.name]
        if set(table.rows) != base_key_set:
            missing = _sorted_keys(base_key_set - set(table.rows))[:5]
            extra = _sorted_keys(set(table.rows) - base_key_set)[:5]
            raise ValueError(
                f"Key mismatch for {spec.name}: missing={missing}, extra={extra}, "
                f"rows={len(table.rows)}"
            )
        if list(table.class_names) != class_names:
            raise ValueError(
                f"Class-order mismatch for {spec.name}: {table.class_names} != {class_names}"
            )
        for row_index, key in enumerate(keys):
            if key not in table.targets:
                raise ValueError(f"{spec.name} row {key} has no target")
            if int(table.targets[key]) != int(target_array[row_index]):
                raise ValueError(
                    f"Target mismatch for {spec.name} at {key}: "
                    f"{table.targets[key]} != {target_array[row_index]}"
                )
            image_path = str(table.image_paths.get(key, "") or "")
            split = _split_from_path(image_path)
            if split != "val":
                raise ValueError(
                    f"Validation-only guard rejected {spec.name} row {key}: "
                    f"split={split!r}, path={image_path!r}"
                )
    return keys, target_array, class_names, source_groups


def audit_validation_information_ceiling(
    *,
    input_specs: Sequence[InputSpec],
    base_name: str,
    output_dir: Path,
    expected_samples: int = 2606,
    focus_class_index: int = 1,
    target_focus_f1: float = 0.98,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260711,
    bootstrap_mode: str = "source_group",
) -> Dict[str, object]:
    if len(input_specs) < 2:
        raise ValueError("At least two aligned model inputs are required")
    if not 0.0 < float(target_focus_f1) <= 1.0:
        raise ValueError("target_focus_f1 must be in (0, 1]")
    if int(bootstrap_iterations) < 1:
        raise ValueError("bootstrap_iterations must be positive")
    names = [spec.name for spec in input_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate input model names: {names}")

    first = _load_prediction_table(input_specs[0].name, input_specs[0].path)
    class_count = len(first.class_names)
    tables: dict[str, PredictionTable] = {input_specs[0].name: first}
    for spec in input_specs[1:]:
        tables[spec.name] = _load_prediction_table(spec.name, spec.path, class_count)
    keys, targets, class_names, source_groups = _validate_tables(
        input_specs, tables, base_name, expected_samples
    )
    if not 0 <= int(focus_class_index) < class_count:
        raise ValueError(f"Invalid focus class {focus_class_index} for {class_count} classes")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    family_by_name = {spec.name: spec.family for spec in input_specs}
    path_by_name = {spec.name: str(spec.path.resolve()) for spec in input_specs}
    predictions = {
        name: np.asarray([tables[name].predictions[key] for key in keys], dtype=np.int64)
        for name in names
    }
    base_predictions = predictions[base_name]
    base_correct = base_predictions == targets

    metrics: dict[str, Dict[str, object]] = {
        name: _metrics_payload(targets, predictions[name], class_names) for name in names
    }
    model_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for name in names:
        focus = _focus_payload(metrics[name], int(focus_class_index))
        model_rows.append(
            {
                "family": family_by_name[name],
                "model": name,
                "samples": int(targets.size),
                "accuracy": float(metrics[name]["accuracy"]),
                "macro_f1": float(metrics[name]["macro_f1"]),
                "focus_class_index": int(focus_class_index),
                "focus_f1": float(focus["f1"]),
                "focus_precision": float(focus["precision"]),
                "focus_recall": float(focus["recall"]),
                "focus_tp": int(focus["tp"]),
                "focus_fp": int(focus["fp"]),
                "focus_fn": int(focus["fn"]),
            }
        )
        for item in metrics[name]["per_class"]:
            per_class_rows.append(
                {"family": family_by_name[name], "model": name, **dict(item)}
            )

    pairwise_rows: list[dict[str, object]] = []
    for name in names:
        if name == base_name:
            continue
        model_correct = predictions[name] == targets
        corrections = int(np.sum(~base_correct & model_correct))
        harms = int(np.sum(base_correct & ~model_correct))
        pairwise_rows.append(
            {
                "base_model": base_name,
                "family": family_by_name[name],
                "model": name,
                "corrections": corrections,
                "harms": harms,
                "net_correct": int(corrections - harms),
                "both_correct": int(np.sum(base_correct & model_correct)),
                "both_wrong": int(np.sum(~base_correct & ~model_correct)),
                "prediction_disagreements": int(np.sum(base_predictions != predictions[name])),
                "mcnemar_exact_pvalue": _mcnemar_exact_pvalue(corrections, harms),
            }
        )

    family_members: dict[str, list[str]] = defaultdict(list)
    for spec in input_specs:
        family_members[spec.family].append(spec.name)
    oracle_groups = {family: members for family, members in sorted(family_members.items())}
    oracle_groups["all_models"] = list(names)
    oracle_predictions: dict[str, np.ndarray] = {}
    focus_binary_predictions: dict[str, np.ndarray] = {}
    exact_oracle_details: dict[str, Dict[str, object]] = {}
    oracle_rows: list[dict[str, object]] = []
    oracle_per_class_rows: list[dict[str, object]] = []
    focus_ceiling_rows: list[dict[str, object]] = []
    base_focus = _focus_payload(metrics[base_name], int(focus_class_index))
    base_tp = int(base_focus["tp"])
    base_fp = int(base_focus["fp"])
    base_fn = int(base_focus["fn"])
    focus_fn_mask = (targets == focus_class_index) & (base_predictions != focus_class_index)
    focus_fp_mask = (targets != focus_class_index) & (base_predictions == focus_class_index)

    for group_name, members in oracle_groups.items():
        group_predictions = np.stack([predictions[name] for name in members], axis=0)
        any_exact_correct = np.any(group_predictions == targets[None, :], axis=0)
        exact_predictions = base_predictions.copy()
        exact_predictions[~base_correct & any_exact_correct] = targets[~base_correct & any_exact_correct]
        oracle_predictions[group_name] = exact_predictions
        exact_metrics = _metrics_payload(targets, exact_predictions, class_names)
        exact_oracle_details[group_name] = exact_metrics
        exact_focus = _focus_payload(exact_metrics, int(focus_class_index))
        for item in exact_metrics["per_class"]:
            oracle_per_class_rows.append(
                {"oracle_group": group_name, "members": "|".join(members), **dict(item)}
            )
        oracle_rows.append(
            {
                "oracle_group": group_name,
                "members": "|".join(members),
                "member_count": len(members),
                "correctable_base_errors": int(np.sum(~base_correct & any_exact_correct)),
                "residual_base_errors": int(np.sum(~base_correct & ~any_exact_correct)),
                "accuracy": float(exact_metrics["accuracy"]),
                "macro_f1": float(exact_metrics["macro_f1"]),
                "focus_f1": float(exact_focus["f1"]),
                "focus_precision": float(exact_focus["precision"]),
                "focus_recall": float(exact_focus["recall"]),
                "focus_fp": int(exact_focus["fp"]),
                "focus_fn": int(exact_focus["fn"]),
            }
        )

        predicts_focus = np.any(group_predictions == focus_class_index, axis=0)
        predicts_nonfocus = np.any(group_predictions != focus_class_index, axis=0)
        corrected_fn = int(np.sum(focus_fn_mask & predicts_focus))
        corrected_fp = int(np.sum(focus_fp_mask & predicts_nonfocus))
        binary_correctable = (focus_fn_mask & predicts_focus) | (
            focus_fp_mask & predicts_nonfocus
        )
        binary_predictions = base_predictions.copy()
        binary_predictions[binary_correctable] = targets[binary_correctable]
        focus_binary_predictions[group_name] = binary_predictions
        binary_focus = _focus_metrics_from_counts(
            base_tp + corrected_fn,
            base_fp - corrected_fp,
            base_fn - corrected_fn,
        )
        focus_ceiling_rows.append(
            {
                "oracle_group": group_name,
                "members": "|".join(members),
                "focus_fn_correctable": corrected_fn,
                "focus_fn_residual": int(base_fn - corrected_fn),
                "focus_fp_correctable": corrected_fp,
                "focus_fp_residual": int(base_fp - corrected_fp),
                **binary_focus,
                "target_focus_f1": float(target_focus_f1),
                "target_reached": bool(float(binary_focus["f1"]) >= float(target_focus_f1)),
            }
        )

    all_prediction_matrix = np.stack([predictions[name] for name in names], axis=0)
    correct_matrix = all_prediction_matrix == targets[None, :]
    correct_counts = correct_matrix.sum(axis=0)
    unique_counts = np.asarray(
        [len(set(all_prediction_matrix[:, index].tolist())) for index in range(targets.size)],
        dtype=np.int64,
    )
    unanimous = unique_counts == 1
    all_wrong = correct_counts == 0
    disagreement = ~unanimous
    all_binary_predictions = focus_binary_predictions["all_models"]
    focus_binary_residual = (
        focus_fn_mask & (all_binary_predictions != focus_class_index)
    ) | (focus_fp_mask & (all_binary_predictions == focus_class_index))
    disagreement_summary = {
        "samples": int(targets.size),
        "unanimous_rows": int(unanimous.sum()),
        "unanimous_correct_rows": int(np.sum(unanimous & (correct_counts > 0))),
        "unanimous_wrong_rows": int(np.sum(unanimous & all_wrong)),
        "disagreement_rows": int(disagreement.sum()),
        "disagreement_rate": float(disagreement.mean()),
        "disagreement_any_model_correct": int(np.sum(disagreement & (correct_counts > 0))),
        "disagreement_all_models_wrong": int(np.sum(disagreement & all_wrong)),
        "all_models_wrong_rows": int(all_wrong.sum()),
        "all_models_wrong_rate": float(all_wrong.mean()),
        "base_error_any_model_correct": int(np.sum(~base_correct & (correct_counts > 0))),
        "base_error_all_models_wrong": int(np.sum(~base_correct & all_wrong)),
    }

    transition_rows: list[dict[str, object]] = []
    for target_index in range(class_count):
        for prediction_index in range(class_count):
            if target_index == prediction_index:
                continue
            mask = (targets == target_index) & (base_predictions == prediction_index)
            if not np.any(mask):
                continue
            row: dict[str, object] = {
                "transition": f"{target_index}->{prediction_index}",
                "target_name": class_names[target_index],
                "base_prediction_name": class_names[prediction_index],
                "base_errors": int(mask.sum()),
                "any_model_exact_correct": int(np.sum(mask & (correct_counts > 0))),
                "all_models_wrong": int(np.sum(mask & all_wrong)),
            }
            for family, members in sorted(family_members.items()):
                family_matrix = np.stack([predictions[name] for name in members], axis=0)
                family_correct = np.any(family_matrix == targets[None, :], axis=0)
                row[f"{_safe_column(family)}_exact_correct"] = int(np.sum(mask & family_correct))
            correctors = []
            for name in names:
                count = int(np.sum(mask & (predictions[name] == targets)))
                if count:
                    correctors.append((count, name))
            correctors.sort(reverse=True)
            row["top_correctors"] = "|".join(f"{name}:{count}" for count, name in correctors)
            transition_rows.append(row)
    transition_rows.sort(key=lambda row: (-int(row["base_errors"]), str(row["transition"])))

    base_budget = {
        "target_focus_f1": float(target_focus_f1),
        "base": _focus_metrics_from_counts(base_tp, base_fp, base_fn),
        "minimum_independent_corrections": _minimum_focus_corrections(
            base_tp, base_fp, base_fn, float(target_focus_f1)
        ),
        "family_focus_binary_ceilings": {
            str(row["oracle_group"]): {
                key: row[key]
                for key in (
                    "focus_fn_correctable",
                    "focus_fn_residual",
                    "focus_fp_correctable",
                    "focus_fp_residual",
                    "precision",
                    "recall",
                    "f1",
                    "target_reached",
                )
            }
            for row in focus_ceiling_rows
        },
    }

    system_predictions = {f"model:{name}": predictions[name] for name in names}
    system_predictions.update(
        {f"exact_oracle:{group}": values for group, values in oracle_predictions.items()}
    )
    system_predictions.update(
        {
            f"focus_binary_oracle:{group}": values
            for group, values in focus_binary_predictions.items()
        }
    )
    bootstrap_values = {
        system: {
            "accuracy": [],
            "macro_f1": [],
            "class_f1": [[] for _ in range(class_count)],
        }
        for system in system_predictions
    }
    for draw in _bootstrap_draws(
        targets=targets,
        source_groups=source_groups,
        iterations=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
        mode=str(bootstrap_mode),
    ):
        draw_targets = targets[draw]
        for system, values in system_predictions.items():
            accuracy, macro_f1, _, _, f1, _ = _metric_arrays(
                draw_targets, values[draw], class_count
            )
            bootstrap_values[system]["accuracy"].append(accuracy)
            bootstrap_values[system]["macro_f1"].append(macro_f1)
            for class_index in range(class_count):
                bootstrap_values[system]["class_f1"][class_index].append(float(f1[class_index]))

    bootstrap_rows: list[dict[str, object]] = []
    point_by_system: dict[str, dict[str, object]] = {}
    for system, values in system_predictions.items():
        point_accuracy, point_macro, _, _, point_f1, _ = _metric_arrays(
            targets, values, class_count
        )
        point_by_system[system] = {
            "accuracy": point_accuracy,
            "macro_f1": point_macro,
            "class_f1": point_f1,
        }
        is_focus_binary_oracle = system.startswith("focus_binary_oracle:")
        aggregate_metrics = () if is_focus_binary_oracle else (
            ("accuracy", point_accuracy),
            ("macro_f1", point_macro),
        )
        for metric_name, point in aggregate_metrics:
            summary = _percentile_summary(
                np.asarray(bootstrap_values[system][metric_name], dtype=np.float64), point
            )
            bootstrap_rows.append(
                {
                    "system": system,
                    "metric": metric_name,
                    "class_index": "",
                    "class_name": "",
                    **summary,
                    "iterations": int(bootstrap_iterations),
                    "bootstrap_mode": bootstrap_mode,
                }
            )
        for class_index, class_name in enumerate(class_names):
            if is_focus_binary_oracle and class_index != int(focus_class_index):
                continue
            summary = _percentile_summary(
                np.asarray(bootstrap_values[system]["class_f1"][class_index], dtype=np.float64),
                float(point_f1[class_index]),
            )
            bootstrap_rows.append(
                {
                    "system": system,
                    "metric": "class_f1",
                    "class_index": class_index,
                    "class_name": class_name,
                    **summary,
                    "iterations": int(bootstrap_iterations),
                    "bootstrap_mode": bootstrap_mode,
                }
            )

    delta_rows: list[dict[str, object]] = []
    base_system = f"model:{base_name}"
    for system in system_predictions:
        if system == base_system:
            continue
        delta_metrics = (
            (("class_f1", focus_class_index),)
            if system.startswith("focus_binary_oracle:")
            else (("macro_f1", None), ("class_f1", focus_class_index))
        )
        for metric_name, class_index in delta_metrics:
            if class_index is None:
                system_samples = np.asarray(bootstrap_values[system][metric_name], dtype=np.float64)
                base_samples = np.asarray(bootstrap_values[base_system][metric_name], dtype=np.float64)
                point_delta = float(point_by_system[system][metric_name]) - float(
                    point_by_system[base_system][metric_name]
                )
                output_metric = "macro_f1"
            else:
                system_samples = np.asarray(
                    bootstrap_values[system][metric_name][class_index], dtype=np.float64
                )
                base_samples = np.asarray(
                    bootstrap_values[base_system][metric_name][class_index], dtype=np.float64
                )
                point_delta = float(point_by_system[system][metric_name][class_index]) - float(
                    point_by_system[base_system][metric_name][class_index]
                )
                output_metric = "focus_f1"
            deltas = system_samples - base_samples
            delta_rows.append(
                {
                    "base_system": base_system,
                    "system": system,
                    "metric": output_metric,
                    "point_delta": point_delta,
                    "bootstrap_mean_delta": float(deltas.mean()),
                    "ci_low": float(np.percentile(deltas, 2.5)),
                    "ci_high": float(np.percentile(deltas, 97.5)),
                    "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
                    "iterations": int(bootstrap_iterations),
                    "bootstrap_mode": bootstrap_mode,
                }
            )

    case_rows: list[dict[str, object]] = []
    for index, key in enumerate(keys):
        if not disagreement[index] and not all_wrong[index]:
            continue
        target = int(targets[index])
        base_prediction = int(base_predictions[index])
        row: dict[str, object] = {
            "key": key,
            "sample_index": int(key.split(":", 1)[1]) if key.startswith("sample:") else "",
            "image_path": str(tables[base_name].image_paths.get(key, "")),
            "source_group": source_groups[index],
            "target_index": target,
            "target_name": class_names[target],
            "base_prediction_index": base_prediction,
            "base_prediction_name": class_names[base_prediction],
            "base_correct": int(base_prediction == target),
            "base_transition": "" if base_prediction == target else f"{target}->{base_prediction}",
            "unique_prediction_count": int(unique_counts[index]),
            "correct_model_count": int(correct_counts[index]),
            "unanimous": int(unanimous[index]),
            "all_models_wrong": int(all_wrong[index]),
            "focus_related": int(target == focus_class_index or base_prediction == focus_class_index),
            "focus_binary_residual": int(focus_binary_residual[index]),
            "focus_binary_residual_kind": (
                "fn" if focus_fn_mask[index] and focus_binary_residual[index] else
                "fp" if focus_fp_mask[index] and focus_binary_residual[index] else ""
            ),
        }
        for name in names:
            row[f"pred_{_safe_column(name)}"] = int(predictions[name][index])
        case_rows.append(row)
    case_rows.sort(
        key=lambda row: (
            -int(row["focus_related"]),
            -int(row["all_models_wrong"]),
            int(row["correct_model_count"]),
            int(row["sample_index"]) if row["sample_index"] != "" else 0,
        )
    )
    irreducible_rows = [row for row in case_rows if int(row["all_models_wrong"]) == 1]
    focus_binary_residual_rows = [
        row for row in case_rows if int(row["focus_binary_residual"]) == 1
    ]

    _write_csv(output_dir / "model_metrics.csv", model_rows)
    _write_markdown_table(
        output_dir / "model_metrics.md",
        model_rows,
        (
            "family",
            "model",
            "samples",
            "accuracy",
            "macro_f1",
            "focus_class_index",
            "focus_f1",
            "focus_precision",
            "focus_recall",
            "focus_tp",
            "focus_fp",
            "focus_fn",
        ),
    )
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(output_dir / "pairwise_vs_base.csv", pairwise_rows)
    _write_csv(output_dir / "exact_oracle_metrics.csv", oracle_rows)
    _write_csv(output_dir / "exact_oracle_per_class_metrics.csv", oracle_per_class_rows)
    _write_csv(output_dir / "focus_binary_ceiling.csv", focus_ceiling_rows)
    _write_csv(output_dir / "transition_coverage.csv", transition_rows)
    _write_csv(output_dir / "bootstrap_metrics.csv", bootstrap_rows)
    _write_csv(output_dir / "bootstrap_delta_vs_base.csv", delta_rows)
    _write_csv(output_dir / "disagreement_cases.csv", case_rows)
    _write_csv(output_dir / "all_models_wrong_cases.csv", irreducible_rows)
    _write_csv(output_dir / "focus_binary_residual_cases.csv", focus_binary_residual_rows)

    summary: Dict[str, object] = {
        "audit_mode": "validation_only_label_using_information_ceiling",
        "validation_guard": {
            "required_split": "val",
            "samples": int(targets.size),
            "contiguous_sample_index": bool(all(key.startswith("sample:") for key in keys)),
            "target_alignment": "strict",
            "key_alignment": "strict",
            "test_used": False,
        },
        "base_model": base_name,
        "focus_class_index": int(focus_class_index),
        "focus_class_name": class_names[int(focus_class_index)],
        "class_names": class_names,
        "inputs": [
            {
                "family": spec.family,
                "name": spec.name,
                "path": path_by_name[spec.name],
            }
            for spec in input_specs
        ],
        "model_metrics": {name: metrics[name] for name in names},
        "exact_oracle_metrics": {str(row["oracle_group"]): row for row in oracle_rows},
        "exact_oracle_full_metrics": exact_oracle_details,
        "focus_binary_ceiling": {
            str(row["oracle_group"]): row for row in focus_ceiling_rows
        },
        "focus_target_budget": base_budget,
        "disagreement": disagreement_summary,
        "bootstrap": {
            "mode": bootstrap_mode,
            "iterations": int(bootstrap_iterations),
            "seed": int(bootstrap_seed),
            "confidence_interval": "paired percentile 95%",
            "source_groups": int(len(set(source_groups))),
        },
        "outputs": {
            "model_metrics": "model_metrics.csv",
            "model_metrics_markdown": "model_metrics.md",
            "per_class_metrics": "per_class_metrics.csv",
            "pairwise_vs_base": "pairwise_vs_base.csv",
            "exact_oracle_metrics": "exact_oracle_metrics.csv",
            "exact_oracle_per_class_metrics": "exact_oracle_per_class_metrics.csv",
            "focus_binary_ceiling": "focus_binary_ceiling.csv",
            "transition_coverage": "transition_coverage.csv",
            "bootstrap_metrics": "bootstrap_metrics.csv",
            "bootstrap_delta_vs_base": "bootstrap_delta_vs_base.csv",
            "disagreement_cases": "disagreement_cases.csv",
            "all_models_wrong_cases": "all_models_wrong_cases.csv",
            "focus_binary_residual_cases": "focus_binary_residual_cases.csv",
        },
        "notes": [
            "Every input is strictly aligned by sample_index/key and target on validation only.",
            "Exact oracle and focus-binary ceiling use validation labels and are not deployable metrics.",
            "Focus-binary ceiling asks only whether any member contains the correct class-1-vs-rest decision.",
            "Bootstrap resamples the same rows/groups for every system, so delta intervals are paired.",
            "No test prediction file is accepted or read by this audit.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = audit_validation_information_ceiling(
        input_specs=[parse_input_spec(value) for value in args.input],
        base_name=str(args.base_name),
        output_dir=Path(args.output_dir),
        expected_samples=int(args.expected_samples),
        focus_class_index=int(args.focus_class_index),
        target_focus_f1=float(args.target_focus_f1),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_mode=str(args.bootstrap_mode),
    )
    compact = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "base_model": summary["base_model"],
        "focus_target_budget": summary["focus_target_budget"],
        "disagreement": summary["disagreement"],
        "test_used": summary["validation_guard"]["test_used"],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
