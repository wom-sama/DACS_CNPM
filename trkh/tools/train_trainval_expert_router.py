from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trkh.data.dataset import _pseudo_foreground_mask_array


PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_(.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an expert-probability router on train predictions only and evaluate on validation. "
            "This avoids fitting selector logic on test and is intended for fast ensemble research."
        )
    )
    parser.add_argument("--train-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--val-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-class-name", default="Xoai_Song_ChuaNhe_CoNguyCo")
    parser.add_argument("--focus-class-weight", type=float, default=0.25)
    parser.add_argument(
        "--selection-source",
        choices=("train_oof", "val"),
        default="train_oof",
        help=(
            "Select the router by train OOF metrics or by validation metrics. "
            "Use val only for development/model selection, never after looking at test."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--image-feature-cache",
        type=Path,
        default=None,
        help="Optional JSON cache for label-free brightness/background features.",
    )
    parser.add_argument("--disable-image-features", action="store_true")
    parser.add_argument(
        "--write-train-oof",
        action="store_true",
        help="Write train out-of-fold predictions for forensic review.",
    )
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CSV, got: {value!r}")
    name, path_text = value.split("=", 1)
    path = Path(path_text.strip())
    if not name.strip() or not path.is_file():
        raise ValueError(f"Invalid selector input: {value!r}")
    return name.strip(), path


def _read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    output: Dict[str, Dict[str, str]] = {}
    for row in rows:
        image_path = str(row.get("path") or row.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"Prediction CSV is missing path/image_path: {path}")
        row["path"] = image_path
        sample_index = str(row.get("sample_index", "") or "").strip()
        key = f"sample_index:{sample_index}" if sample_index else image_path.lower()
        if key in output:
            raise ValueError(
                "Prediction CSV has duplicate routing key "
                f"{key!r}; include object-level sample_index for YOLO multi-object CSVs: {path}"
            )
        output[key] = row
    return output


def _truth_name(row: Mapping[str, str]) -> str:
    for key in ("true_name", "target_name"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    raise ValueError("Prediction row is missing true_name/target_name.")


def _prediction_name(row: Mapping[str, str]) -> str:
    for key in ("pred_name", "prediction_name", "teacher_pred_name", "selector_pred_name"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    raise ValueError("Prediction row is missing prediction name.")


def _source_class_names(rows: Mapping[str, Mapping[str, str]], metrics_path: Path | None) -> List[str]:
    first = next(iter(rows.values()))
    indexed_columns: Dict[int, str] = {}
    for column in first:
        match = PROBABILITY_COLUMN.match(str(column))
        if match is not None and match.group(2):
            indexed_columns[int(match.group(1))] = str(match.group(2))
    if indexed_columns:
        return [indexed_columns[index] for index in sorted(indexed_columns)]

    if metrics_path is not None and metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("classes", "target_classes"):
            values = payload.get(key)
            if isinstance(values, list) and values:
                return [str(value) for value in values]
        per_class = payload.get("per_class")
        if isinstance(per_class, list) and per_class:
            names = [str(item.get("class_name", "")) for item in per_class]
            if all(names):
                return names

    indexed_names: Dict[int, str] = {}
    for row in rows.values():
        for index_key, name_key in (
            ("y_true", "true_name"),
            ("target_index", "target_name"),
            ("y_pred", "pred_name"),
            ("prediction_index", "prediction_name"),
            ("teacher_pred_index", "teacher_pred_name"),
        ):
            index_text = str(row.get(index_key, "") or "").strip()
            name = str(row.get(name_key, "") or "").strip()
            if index_text and name:
                indexed_names[int(index_text)] = name
    if not indexed_names:
        raise ValueError("Cannot infer class order from prediction rows or metrics JSON.")
    expected = list(range(len(indexed_names)))
    if sorted(indexed_names) != expected:
        raise ValueError(f"Non-contiguous source class indices: {sorted(indexed_names)}")
    return [indexed_names[index] for index in expected]


def _metrics_candidate_path(prediction_csv: Path, split: str) -> Path | None:
    candidates = [
        prediction_csv.with_name(f"metrics_{split}.json"),
        prediction_csv.with_name(f"teacher_probs_{split}_metrics.json"),
        prediction_csv.with_name("metrics.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _probability_columns(row: Mapping[str, str], source_index: int) -> List[str]:
    return [
        key
        for key in row
        if key == f"prob_{source_index}" or key.startswith(f"prob_{source_index}_")
    ]


def _row_probabilities(
    row: Mapping[str, str],
    source_class_names: Sequence[str],
    class_to_index: Mapping[str, int],
) -> np.ndarray:
    probabilities = np.zeros(len(class_to_index), dtype=np.float32)
    found = False
    for source_index, class_name in enumerate(source_class_names):
        candidate_columns = _probability_columns(row, source_index)
        if candidate_columns:
            probabilities[int(class_to_index[class_name])] = float(row[candidate_columns[0]])
            found = True
    if not found:
        probabilities[int(class_to_index[_prediction_name(row)])] = 1.0
    probabilities = np.clip(probabilities, 1e-8, None)
    probabilities /= max(1e-8, float(probabilities.sum()))
    return probabilities


def _prediction_features(
    path_key: str,
    names: Sequence[str],
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    source_classes: Mapping[str, Sequence[str]],
    class_to_index: Mapping[str, int],
) -> List[float]:
    probability_parts: List[np.ndarray] = []
    prediction_indices: List[int] = []
    features: List[float] = []
    for name in names:
        probabilities = _row_probabilities(
            named_rows[name][path_key],
            source_classes[name],
            class_to_index,
        )
        probability_parts.append(probabilities)
        prediction = int(np.argmax(probabilities))
        prediction_indices.append(prediction)
        sorted_probabilities = np.sort(probabilities)
        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))))
        margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])
        features.extend(probabilities.tolist())
        features.extend(np.log(np.clip(probabilities, 1e-8, 1.0)).tolist())
        features.extend([1.0 if index == prediction else 0.0 for index in range(len(class_to_index))])
        features.extend([float(probabilities[prediction]), entropy, margin])

    stacked = np.stack(probability_parts, axis=0)
    stacked_min = stacked.min(axis=0)
    stacked_max = stacked.max(axis=0)
    features.extend(stacked.mean(axis=0).tolist())
    features.extend(stacked.std(axis=0).tolist())
    features.extend(stacked_min.tolist())
    features.extend(stacked_max.tolist())
    features.extend((stacked_max - stacked_min).tolist())
    vote_counts = Counter(prediction_indices)
    vote_fractions = np.asarray(
        [
            float(vote_counts.get(index, 0)) / max(1, len(prediction_indices))
            for index in range(len(class_to_index))
        ],
        dtype=np.float32,
    )
    vote_entropy = -float(np.sum(vote_fractions * np.log(np.clip(vote_fractions, 1e-8, 1.0))))
    features.extend(vote_fractions.tolist())
    features.extend(
        [
            float(max(vote_counts.values())) / max(1, len(prediction_indices)),
            float(len(vote_counts)),
            vote_entropy,
        ]
    )
    return features


def _image_features(image_path: str) -> List[float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((128, 128), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        mask = _pseudo_foreground_mask_array(image, margin=0.08)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)
    gradient_x = np.abs(luminance[:, 1:] - luminance[:, :-1])
    gradient_y = np.abs(luminance[1:, :] - luminance[:-1, :])
    foreground = mask.astype(bool)
    background = ~foreground

    def masked_mean(values: np.ndarray, selection: np.ndarray) -> float:
        if not bool(selection.any()):
            return float(values.mean())
        return float(values[selection].mean())

    foreground_luminance = masked_mean(luminance, foreground)
    background_luminance = masked_mean(luminance, background)
    foreground_saturation = masked_mean(saturation, foreground)
    background_saturation = masked_mean(saturation, background)
    quantiles = np.quantile(luminance, [0.02, 0.10, 0.50, 0.90, 0.98])
    return [
        float(luminance.mean()),
        float(luminance.std()),
        *[float(value) for value in quantiles],
        float((luminance < 0.10).mean()),
        float((luminance > 0.90).mean()),
        float(saturation.mean()),
        float(saturation.std()),
        *[float(value) for value in rgb.mean(axis=(0, 1))],
        *[float(value) for value in rgb.std(axis=(0, 1))],
        float(gradient_x.mean()) if gradient_x.size else 0.0,
        float(gradient_y.mean()) if gradient_y.size else 0.0,
        float(foreground.mean()),
        foreground_luminance,
        background_luminance,
        foreground_luminance - background_luminance,
        foreground_saturation,
        background_saturation,
        foreground_saturation - background_saturation,
        float(rgb.shape[1]) / max(1.0, float(rgb.shape[0])),
    ]


def _load_image_feature_cache(path: Path | None) -> Dict[str, List[float]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key).lower(): [float(value) for value in values] for key, values in dict(payload).items()}


def _save_image_feature_cache(path: Path | None, cache: Mapping[str, Sequence[float]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _build_matrix(
    paths: Sequence[str],
    names: Sequence[str],
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    source_classes: Mapping[str, Sequence[str]],
    class_to_index: Mapping[str, int],
    image_cache: Dict[str, List[float]],
    include_image_features: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_rows: List[List[float]] = []
    targets: List[int] = []
    original_paths: List[str] = []
    truth_source = named_rows[names[0]]
    for path_key in paths:
        source_row = truth_source[path_key]
        image_path = str(source_row["path"])
        features = _prediction_features(path_key, names, named_rows, source_classes, class_to_index)
        if include_image_features:
            if path_key not in image_cache:
                image_cache[path_key] = _image_features(image_path)
            features.extend(image_cache[path_key])
        feature_rows.append(features)
        targets.append(int(class_to_index[_truth_name(source_row)]))
        original_paths.append(image_path)
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(targets, dtype=np.int64), original_paths


def _metrics(targets: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[int(target), int(prediction)] += 1
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {
                "class_index": index,
                "class_name": str(class_names[index]),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(class_names))
        ],
        "confusion_matrix": matrix.tolist(),
    }


class FixedAverageRouter:
    def __init__(self, n_classes: int, n_experts: int, temperature: float = 1.0) -> None:
        self.n_classes = int(n_classes)
        self.n_experts = int(n_experts)
        self.temperature = float(temperature)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {
            "n_classes": self.n_classes,
            "n_experts": self.n_experts,
            "temperature": self.temperature,
        }

    def set_params(self, **params: object) -> "FixedAverageRouter":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FixedAverageRouter":
        return self

    def _extract_probabilities(self, x: np.ndarray) -> np.ndarray:
        width = int(self.n_classes)
        parts = []
        cursor = 0
        for _ in range(int(self.n_experts)):
            probabilities = np.asarray(x[:, cursor : cursor + width], dtype=np.float64)
            probabilities = np.clip(probabilities, 1e-12, 1.0)
            if abs(float(self.temperature) - 1.0) > 1e-8:
                probabilities = probabilities ** (1.0 / max(1e-6, float(self.temperature)))
                probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
            parts.append(probabilities)
            cursor += width * 3 + 3
        return np.stack(parts, axis=1).mean(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = self._extract_probabilities(x)
        probabilities = np.clip(probabilities, 1e-12, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def _candidates(seed: int, n_jobs: int, n_classes: int, n_experts: int):
    candidates = [
        (f"avg_temp_{str(temp).replace('.', 'p')}", FixedAverageRouter(n_classes, n_experts, temp))
        for temp in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    ]
    candidates.extend(
        [
            (
                "logreg_c0p03",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.03, max_iter=3000, random_state=seed),
                ),
            ),
            (
                "logreg_c0p1",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.1, max_iter=3000, random_state=seed),
                ),
            ),
            (
                "logreg_c0p3",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.3, max_iter=3000, random_state=seed),
                ),
            ),
            (
                "logreg_balanced_c0p03",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.03, max_iter=3000, class_weight="balanced", random_state=seed),
                ),
            ),
            (
                "logreg_balanced_c0p1",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=seed),
                ),
            ),
            (
                "extra_trees_leaf10",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=10,
                    max_features=0.6,
                    class_weight=None,
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
            (
                "extra_trees_balanced_leaf10",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=10,
                    max_features=0.6,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
            (
                "hist_gradient_leaf15",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=200,
                    max_leaf_nodes=15,
                    l2_regularization=2.0,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
            (
                "hist_gradient_leaf31",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=260,
                    max_leaf_nodes=31,
                    l2_regularization=4.0,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    return candidates


def main() -> None:
    args = parse_args()
    train_inputs = [_parse_named_path(value) for value in args.train_input]
    val_inputs = [_parse_named_path(value) for value in args.val_input]
    names = [name for name, _ in train_inputs]
    if names != [name for name, _ in val_inputs] or len(set(names)) != len(names):
        raise ValueError("Train/val expert names must match in the same order.")

    train_rows = {name: _read_csv(path) for name, path in train_inputs}
    val_rows = {name: _read_csv(path) for name, path in val_inputs}
    train_paths = sorted(set.intersection(*(set(rows) for rows in train_rows.values())))
    val_paths = sorted(set.intersection(*(set(rows) for rows in val_rows.values())))
    if not train_paths or not val_paths:
        raise ValueError("Router inputs do not share aligned train/val paths.")

    class_names = sorted({_truth_name(train_rows[names[0]][path]) for path in train_paths})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if args.focus_class_name not in class_to_index:
        raise ValueError(f"Unknown focus class: {args.focus_class_name}")
    focus_index = int(class_to_index[args.focus_class_name])
    source_classes = {
        name: _source_class_names(train_rows[name], _metrics_candidate_path(path, "train"))
        for name, path in train_inputs
    }
    for name, path in val_inputs:
        val_source = _source_class_names(val_rows[name], _metrics_candidate_path(path, "val"))
        if list(val_source) != list(source_classes[name]):
            raise ValueError(f"Train/val class order differs for expert {name}.")

    image_cache = _load_image_feature_cache(args.image_feature_cache)
    include_image_features = not bool(args.disable_image_features)
    x_train, y_train, train_original_paths = _build_matrix(
        train_paths,
        names,
        train_rows,
        source_classes,
        class_to_index,
        image_cache,
        include_image_features,
    )
    x_val, y_val, val_original_paths = _build_matrix(
        val_paths,
        names,
        val_rows,
        source_classes,
        class_to_index,
        image_cache,
        include_image_features,
    )
    _save_image_feature_cache(args.image_feature_cache, image_cache)

    folds = min(int(args.folds), int(np.bincount(y_train).min()))
    splitter = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=int(args.seed))
    cv_rows: List[Dict[str, object]] = []
    selected_name = ""
    selected_model = None
    selected_key = (-math.inf, -math.inf, -math.inf)
    selected_oof_predictions: np.ndarray | None = None
    selected_oof_metrics: Dict[str, object] = {}
    for candidate_name, candidate_model in _candidates(args.seed, args.n_jobs, len(class_names), len(names)):
        oof_predictions = cross_val_predict(
            candidate_model,
            x_train,
            y_train,
            cv=splitter,
            method="predict",
            n_jobs=1,
        )
        metrics = _metrics(y_train, oof_predictions, class_names)
        focus_f1 = float(metrics["per_class"][focus_index]["f1"])
        objective = float(metrics["macro_f1"]) + max(0.0, float(args.focus_class_weight)) * focus_f1
        val_candidate_model = clone(candidate_model)
        val_candidate_model.fit(x_train, y_train)
        val_candidate_predictions = val_candidate_model.predict(x_val)
        val_candidate_metrics = _metrics(y_val, val_candidate_predictions, class_names)
        val_focus_f1 = float(val_candidate_metrics["per_class"][focus_index]["f1"])
        val_objective = float(val_candidate_metrics["macro_f1"]) + max(
            0.0,
            float(args.focus_class_weight),
        ) * val_focus_f1
        row = {
            "candidate": candidate_name,
            "objective": objective,
            "oof_macro_f1": float(metrics["macro_f1"]),
            "oof_focus_f1": focus_f1,
            "oof_accuracy": float(metrics["accuracy"]),
            "val_objective": val_objective,
            "val_macro_f1": float(val_candidate_metrics["macro_f1"]),
            "val_focus_f1": val_focus_f1,
            "val_accuracy": float(val_candidate_metrics["accuracy"]),
        }
        cv_rows.append(row)
        print(json.dumps(row), flush=True)
        if str(args.selection_source) == "val":
            key = (val_objective, float(val_candidate_metrics["macro_f1"]), val_focus_f1)
        else:
            key = (objective, float(metrics["macro_f1"]), focus_f1)
        if key > selected_key:
            selected_key = key
            selected_name = candidate_name
            selected_model = clone(candidate_model)
            selected_oof_predictions = np.asarray(oof_predictions, dtype=np.int64)
            selected_oof_metrics = metrics
    if selected_model is None:
        raise RuntimeError("No router candidate completed.")

    selected_model.fit(x_train, y_train)
    val_predictions = selected_model.predict(x_val)
    val_probabilities = selected_model.predict_proba(x_val)
    val_metrics = _metrics(y_val, val_predictions, class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cv_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "candidate",
                "objective",
                "oof_macro_f1",
                "oof_focus_f1",
                "oof_accuracy",
                "val_objective",
                "val_macro_f1",
                "val_focus_f1",
                "val_accuracy",
            ),
        )
        writer.writeheader()
        writer.writerows(cv_rows)
    with (output_dir / "router.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": selected_model,
                "experts": names,
                "source_classes": source_classes,
                "class_names": class_names,
                "include_image_features": include_image_features,
            },
            handle,
        )
    with (output_dir / "predictions_val.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "true_name",
            "router_pred_name",
            "correct",
            "confidence",
            *[f"prob_{index}_{name}" for index, name in enumerate(class_names)],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, image_path in enumerate(val_original_paths):
            prediction = int(val_predictions[row_index])
            row = {
                "path": image_path,
                "true_name": class_names[int(y_val[row_index])],
                "router_pred_name": class_names[prediction],
                "correct": int(prediction == int(y_val[row_index])),
                "confidence": float(val_probabilities[row_index, prediction]),
            }
            row.update(
                {
                    f"prob_{index}_{name}": float(val_probabilities[row_index, index])
                    for index, name in enumerate(class_names)
                }
            )
            writer.writerow(row)

    if args.write_train_oof and selected_oof_predictions is not None:
        with (output_dir / "predictions_train_oof.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("path", "true_name", "router_pred_name", "correct"),
            )
            writer.writeheader()
            for row_index, image_path in enumerate(train_original_paths):
                prediction = int(selected_oof_predictions[row_index])
                writer.writerow(
                    {
                        "path": image_path,
                        "true_name": class_names[int(y_train[row_index])],
                        "router_pred_name": class_names[prediction],
                        "correct": int(prediction == int(y_train[row_index])),
                    }
                )

    summary = {
        "mode": "train_only_expert_router",
        "experts": names,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "feature_dim": int(x_train.shape[1]),
        "image_features": include_image_features,
        "folds": int(splitter.n_splits),
        "focus_class_name": args.focus_class_name,
        "focus_class_index": focus_index,
        "focus_class_weight": float(args.focus_class_weight),
        "selected_candidate": selected_name,
        "selection_source": str(args.selection_source),
        "cv_candidates": cv_rows,
        "selected_oof_metrics": selected_oof_metrics,
        "val_metrics": val_metrics,
        "leakage_control": (
            "Router candidate selection and fit use train prediction CSVs only. "
            "Validation is evaluated after selection. Train predictions are in-sample expert outputs, "
            "so validation metrics are the real signal."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "selected_candidate": selected_name,
                "oof_macro_f1": selected_oof_metrics["macro_f1"],
                "oof_focus_f1": selected_oof_metrics["per_class"][focus_index]["f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_focus_f1": val_metrics["per_class"][focus_index]["f1"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
