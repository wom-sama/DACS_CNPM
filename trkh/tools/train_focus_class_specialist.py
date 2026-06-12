from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trkh.tools.train_cv_prediction_selector import (
    _build_matrix,
    _load_image_feature_cache,
    _metrics,
    _parse_named_path,
    _read_csv,
    _row_probabilities,
    _save_image_feature_cache,
    _source_class_names,
    _truth_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OOF-selected binary specialist that can rescue or reject one "
            "focus class while preserving an expert aggregate for all other classes."
        ),
    )
    parser.add_argument("--val-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--test-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--focus-class-name",
        default="Xoai_Song_ChuaNhe_CoNguyCo",
    )
    parser.add_argument("--focus-class-weight", type=float, default=0.5)
    parser.add_argument("--max-macro-drop", type=float, default=0.003)
    parser.add_argument(
        "--base-expert-count",
        type=int,
        default=5,
        help="Use the first N inputs for the base aggregate; later inputs remain specialist features.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--image-feature-cache", type=Path, default=None)
    parser.add_argument("--disable-image-features", action="store_true")
    return parser.parse_args()


def _specialist_candidates(seed: int, n_jobs: int):
    return [
        (
            "logreg_balanced_c0p03",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.03,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "logreg_balanced_c0p1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "logreg_balanced_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "extra_trees_balanced_leaf2",
            ExtraTreesClassifier(
                n_estimators=450,
                min_samples_leaf=2,
                max_features=0.7,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_balanced_leaf5",
            ExtraTreesClassifier(
                n_estimators=450,
                min_samples_leaf=5,
                max_features=0.7,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=seed,
            ),
        ),
        (
            "hist_gradient_balanced_leaf15",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                class_weight="balanced",
                early_stopping=True,
                random_state=seed,
            ),
        ),
    ]


def _aggregate_probabilities(
    paths: Sequence[str],
    names: Sequence[str],
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    source_classes: Mapping[str, Sequence[str]],
    class_to_index: Mapping[str, int],
    mode: str,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for path_key in paths:
        expert_probabilities = np.stack(
            [
                _row_probabilities(
                    named_rows[name][path_key],
                    source_classes[name],
                    class_to_index,
                )
                for name in names
            ],
            axis=0,
        )
        if mode == "mean":
            aggregate = expert_probabilities.mean(axis=0)
        elif mode == "geometric":
            aggregate = np.exp(
                np.log(np.clip(expert_probabilities, 1e-8, 1.0)).mean(axis=0)
            )
        elif mode == "majority":
            votes = np.zeros(expert_probabilities.shape[1], dtype=np.float32)
            for prediction in expert_probabilities.argmax(axis=1):
                votes[int(prediction)] += 1.0
            aggregate = votes + expert_probabilities.mean(axis=0) * 1e-3
        else:
            raise ValueError(f"Unsupported aggregate mode: {mode}")
        aggregate = aggregate / max(1e-8, float(aggregate.sum()))
        rows.append(aggregate.astype(np.float32))
    return np.stack(rows, axis=0)


def _apply_specialist(
    base_probabilities: np.ndarray,
    focus_probabilities: np.ndarray,
    focus_index: int,
    rescue_threshold: float,
    retain_threshold: float,
) -> np.ndarray:
    predictions = base_probabilities.argmax(axis=1)
    fallback_probabilities = base_probabilities.copy()
    fallback_probabilities[:, int(focus_index)] = -1.0
    fallback_predictions = fallback_probabilities.argmax(axis=1)
    rescue = (predictions != int(focus_index)) & (
        focus_probabilities >= float(rescue_threshold)
    )
    reject = (predictions == int(focus_index)) & (
        focus_probabilities < float(retain_threshold)
    )
    predictions[rescue] = int(focus_index)
    predictions[reject] = fallback_predictions[reject]
    return predictions


def _search_thresholds(
    targets: np.ndarray,
    class_names: Sequence[str],
    focus_index: int,
    focus_probabilities: np.ndarray,
    base_probabilities_by_mode: Mapping[str, np.ndarray],
    *,
    focus_weight: float,
    max_macro_drop: float,
) -> Dict[str, object]:
    baseline_by_mode = {
        mode: _metrics(
            targets,
            probabilities.argmax(axis=1),
            class_names,
        )
        for mode, probabilities in base_probabilities_by_mode.items()
    }
    baseline_macro = max(
        float(metrics["macro_f1"])
        for metrics in baseline_by_mode.values()
    )
    minimum_macro = baseline_macro - max(0.0, float(max_macro_drop))
    best: Dict[str, object] | None = None
    for mode, metrics in baseline_by_mode.items():
        macro_f1 = float(metrics["macro_f1"])
        if macro_f1 < minimum_macro:
            continue
        focus_f1 = float(metrics["per_class"][focus_index]["f1"])
        objective = macro_f1 + max(0.0, float(focus_weight)) * focus_f1
        candidate = {
            "aggregate_mode": mode,
            "rescue_threshold": 1.01,
            "retain_threshold": 0.0,
            "objective": objective,
            "metrics": metrics,
        }
        if best is None or (
            objective,
            focus_f1,
            macro_f1,
        ) > (
            float(best["objective"]),
            float(best["metrics"]["per_class"][focus_index]["f1"]),
            float(best["metrics"]["macro_f1"]),
        ):
            best = candidate
    for mode, base_probabilities in base_probabilities_by_mode.items():
        for rescue_threshold in np.arange(0.20, 0.951, 0.025):
            for retain_threshold in np.arange(0.05, rescue_threshold + 0.001, 0.025):
                predictions = _apply_specialist(
                    base_probabilities,
                    focus_probabilities,
                    focus_index,
                    float(rescue_threshold),
                    float(retain_threshold),
                )
                metrics = _metrics(targets, predictions, class_names)
                macro_f1 = float(metrics["macro_f1"])
                if macro_f1 < minimum_macro:
                    continue
                focus_f1 = float(metrics["per_class"][focus_index]["f1"])
                objective = macro_f1 + max(0.0, float(focus_weight)) * focus_f1
                candidate = {
                    "aggregate_mode": mode,
                    "rescue_threshold": float(rescue_threshold),
                    "retain_threshold": float(retain_threshold),
                    "objective": objective,
                    "metrics": metrics,
                }
                key = (objective, focus_f1, macro_f1)
                if best is None:
                    best = candidate
                else:
                    best_metrics = best["metrics"]
                    best_key = (
                        float(best["objective"]),
                        float(best_metrics["per_class"][focus_index]["f1"]),
                        float(best_metrics["macro_f1"]),
                    )
                    if key > best_key:
                        best = candidate
    if best is None:
        raise RuntimeError("No base aggregate satisfies the macro-F1 constraint.")
    return best


def main() -> None:
    args = parse_args()
    val_inputs = [_parse_named_path(value) for value in args.val_input]
    test_inputs = [_parse_named_path(value) for value in args.test_input]
    names = [name for name, _ in val_inputs]
    if names != [name for name, _ in test_inputs] or len(set(names)) != len(names):
        raise ValueError("Val/test specialist inputs must match in the same order.")
    base_expert_count = int(args.base_expert_count)
    if not 1 <= base_expert_count <= len(names):
        raise ValueError("--base-expert-count is outside the input range.")
    base_names = names[:base_expert_count]

    val_rows = {name: _read_csv(path) for name, path in val_inputs}
    test_rows = {name: _read_csv(path) for name, path in test_inputs}
    val_paths = sorted(set.intersection(*(set(rows) for rows in val_rows.values())))
    test_paths = sorted(set.intersection(*(set(rows) for rows in test_rows.values())))
    if not val_paths or not test_paths:
        raise ValueError("Specialist inputs do not share aligned paths.")
    class_names = sorted(
        {_truth_name(val_rows[names[0]][path]) for path in val_paths}
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if args.focus_class_name not in class_to_index:
        raise ValueError(f"Unknown focus class: {args.focus_class_name}")
    focus_index = int(class_to_index[args.focus_class_name])
    source_classes = {name: _source_class_names(val_rows[name]) for name in names}
    for name in names:
        if list(_source_class_names(test_rows[name])) != list(source_classes[name]):
            raise ValueError(f"Val/test class order differs for expert {name}.")

    image_cache = _load_image_feature_cache(args.image_feature_cache)
    include_image_features = not bool(args.disable_image_features)
    x_val, y_val, _ = _build_matrix(
        val_paths,
        names,
        val_rows,
        source_classes,
        class_to_index,
        image_cache,
        include_image_features,
    )
    _save_image_feature_cache(args.image_feature_cache, image_cache)
    binary_val = (y_val == focus_index).astype(np.int64)
    splitter = StratifiedKFold(
        n_splits=max(2, min(int(args.folds), int(np.bincount(binary_val).min()))),
        shuffle=True,
        random_state=int(args.seed),
    )
    val_base_probabilities = {
        mode: _aggregate_probabilities(
            val_paths,
            base_names,
            val_rows,
            source_classes,
            class_to_index,
            mode,
        )
        for mode in ("majority", "mean", "geometric")
    }

    candidate_rows: List[Dict[str, object]] = []
    selected_name = ""
    selected_model = None
    selected_rule: Dict[str, object] | None = None
    selected_key = (-math.inf, -math.inf, -math.inf)
    for candidate_name, candidate_model in _specialist_candidates(
        args.seed,
        args.n_jobs,
    ):
        oof_probabilities = cross_val_predict(
            candidate_model,
            x_val,
            binary_val,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        rule = _search_thresholds(
            y_val,
            class_names,
            focus_index,
            oof_probabilities,
            val_base_probabilities,
            focus_weight=args.focus_class_weight,
            max_macro_drop=args.max_macro_drop,
        )
        metrics = rule["metrics"]
        focus_f1 = float(metrics["per_class"][focus_index]["f1"])
        row = {
            "candidate": candidate_name,
            "aggregate_mode": rule["aggregate_mode"],
            "rescue_threshold": rule["rescue_threshold"],
            "retain_threshold": rule["retain_threshold"],
            "objective": rule["objective"],
            "macro_f1": float(metrics["macro_f1"]),
            "focus_f1": focus_f1,
            "accuracy": float(metrics["accuracy"]),
        }
        candidate_rows.append(row)
        print(json.dumps(row), flush=True)
        key = (float(rule["objective"]), focus_f1, float(metrics["macro_f1"]))
        if key > selected_key:
            selected_key = key
            selected_name = candidate_name
            selected_model = clone(candidate_model)
            selected_rule = rule
    if selected_model is None or selected_rule is None:
        raise RuntimeError("No specialist candidate completed.")

    selected_model.fit(x_val, binary_val)
    # Test data is transformed only after candidate and thresholds are frozen.
    x_test, y_test, original_test_paths = _build_matrix(
        test_paths,
        names,
        test_rows,
        source_classes,
        class_to_index,
        image_cache,
        include_image_features,
    )
    _save_image_feature_cache(args.image_feature_cache, image_cache)
    test_focus_probabilities = selected_model.predict_proba(x_test)[:, 1]
    test_base_probabilities = _aggregate_probabilities(
        test_paths,
        base_names,
        test_rows,
        source_classes,
        class_to_index,
        str(selected_rule["aggregate_mode"]),
    )
    test_predictions = _apply_specialist(
        test_base_probabilities,
        test_focus_probabilities,
        focus_index,
        float(selected_rule["rescue_threshold"]),
        float(selected_rule["retain_threshold"]),
    )
    test_metrics = _metrics(y_test, test_predictions, class_names)
    baseline_test_metrics = _metrics(
        y_test,
        test_base_probabilities.argmax(axis=1),
        class_names,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cv_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (output_dir / "specialist.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": selected_model,
                "class_names": class_names,
                "source_classes": source_classes,
                "experts": names,
                "base_experts": base_names,
                "focus_class_name": args.focus_class_name,
                "focus_index": focus_index,
                "aggregate_mode": selected_rule["aggregate_mode"],
                "rescue_threshold": selected_rule["rescue_threshold"],
                "retain_threshold": selected_rule["retain_threshold"],
                "include_image_features": include_image_features,
            },
            handle,
        )
    with (output_dir / "predictions_test.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "path",
            "true_name",
            "base_pred_name",
            "specialist_pred_name",
            "focus_probability",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        base_predictions = test_base_probabilities.argmax(axis=1)
        for row_index, image_path in enumerate(original_test_paths):
            writer.writerow(
                {
                    "path": image_path,
                    "true_name": class_names[int(y_test[row_index])],
                    "base_pred_name": class_names[int(base_predictions[row_index])],
                    "specialist_pred_name": class_names[int(test_predictions[row_index])],
                    "focus_probability": float(test_focus_probabilities[row_index]),
                    "correct": int(test_predictions[row_index] == y_test[row_index]),
                }
            )

    summary = {
        "mode": "oof_focus_class_specialist",
        "experts": names,
        "base_experts": base_names,
        "focus_class_name": args.focus_class_name,
        "focus_index": focus_index,
        "focus_class_weight": float(args.focus_class_weight),
        "max_macro_drop": float(args.max_macro_drop),
        "selected_candidate": selected_name,
        "selected_rule": {
            key: value
            for key, value in selected_rule.items()
            if key != "metrics"
        },
        "selected_oof_metrics": selected_rule["metrics"],
        "baseline_test_metrics": baseline_test_metrics,
        "test_metrics": test_metrics,
        "cv_candidates": candidate_rows,
        "leakage_control": (
            "Binary model, aggregate mode and rescue/retain thresholds are selected "
            "from validation OOF probabilities. Test is evaluated after all choices freeze."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "selected_candidate": selected_name,
                "aggregate_mode": selected_rule["aggregate_mode"],
                "oof_macro_f1": selected_rule["metrics"]["macro_f1"],
                "oof_focus_f1": selected_rule["metrics"]["per_class"][focus_index]["f1"],
                "test_macro_f1_before": baseline_test_metrics["macro_f1"],
                "test_macro_f1_after": test_metrics["macro_f1"],
                "test_focus_f1_before": baseline_test_metrics["per_class"][focus_index]["f1"],
                "test_focus_f1_after": test_metrics["per_class"][focus_index]["f1"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
