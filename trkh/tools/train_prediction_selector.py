from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a validation-only selector over expert prediction CSVs, then apply it to test CSVs. "
            "Use this for stacking without tuning on test."
        )
    )
    parser.add_argument("--val-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--test-input", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="balanced usually helps the rare class selector.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=CSV, got: {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty model name in {value!r}")
    path = Path(path_text.strip())
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV for {name}: {path}")
    return name, path


def _read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    if "path" not in rows[0] and "image_path" not in rows[0]:
        raise ValueError(f"Prediction CSV missing path/image_path column: {path}")
    for row in rows:
        if "path" not in row or not str(row.get("path", "") or "").strip():
            row["path"] = str(row.get("image_path", "") or "").strip()
    return {str(row["path"]): row for row in rows}


def _truth(row: Mapping[str, str]) -> str:
    for key in ("true_name", "target_name", "y_true", "target_index"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    raise ValueError(f"Cannot infer true label from row keys: {sorted(row.keys())}")


def _prediction(row: Mapping[str, str]) -> str:
    for key in (
        "pred_name",
        "prediction_name",
        "teacher_pred_name",
        "selector_pred_name",
        "y_pred",
        "prediction_index",
        "teacher_pred_index",
    ):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    raise ValueError(f"Cannot infer prediction label from row keys: {sorted(row.keys())}")


def _prob_columns(row: Mapping[str, str]) -> List[str]:
    def key_index(key: str):
        suffix = key.split("_", 1)[1] if "_" in key else ""
        first = suffix.split("_", 1)[0]
        return int(first) if first.isdigit() else key

    return sorted(
        [key for key in row.keys() if key.startswith("prob_")],
        key=key_index,
    )


def _aligned_paths(named_rows: Mapping[str, Mapping[str, Mapping[str, str]]]) -> List[str]:
    path_sets = [set(rows.keys()) for rows in named_rows.values()]
    if not path_sets:
        return []
    return sorted(set.intersection(*path_sets))


def _class_names(named_rows: Mapping[str, Mapping[str, Mapping[str, str]]], paths: Sequence[str]) -> List[str]:
    classes = sorted({_truth(next(iter(named_rows.values()))[path]) for path in paths})
    return classes


def _features_for_path(
    path: str,
    names: Sequence[str],
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    class_to_index: Mapping[str, int],
) -> List[float]:
    features: List[float] = []
    for name in names:
        row = named_rows[name][path]
        prob_keys = _prob_columns(row)
        if prob_keys:
            features.extend(float(row[key]) for key in prob_keys)
        else:
            one_hot = [0.0 for _ in class_to_index]
            pred = _prediction(row)
            if pred in class_to_index:
                one_hot[int(class_to_index[pred])] = 1.0
            features.extend(one_hot)
        pred_one_hot = [0.0 for _ in class_to_index]
        pred = _prediction(row)
        if pred in class_to_index:
            pred_one_hot[int(class_to_index[pred])] = 1.0
        features.extend(pred_one_hot)
    return features


def _build_matrix(
    paths: Sequence[str],
    names: Sequence[str],
    named_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    class_to_index: Mapping[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x = np.asarray(
        [_features_for_path(path, names, named_rows, class_to_index) for path in paths],
        dtype=np.float32,
    )
    y_names = [_truth(next(iter(named_rows.values()))[path]) for path in paths]
    y = np.asarray([int(class_to_index[name]) for name in y_names], dtype=np.int64)
    return x, y, y_names


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, prediction in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(target), int(prediction)] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classes": list(classes),
        "per_class": [
            {
                "class_index": index,
                "class_name": str(classes[index]),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(classes))
        ],
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    val_inputs = [_parse_named_path(value) for value in args.val_input]
    test_inputs = [_parse_named_path(value) for value in args.test_input]
    val_names = [name for name, _ in val_inputs]
    test_names = [name for name, _ in test_inputs]
    if val_names != test_names:
        raise ValueError(f"Val/test expert names must match order: val={val_names}, test={test_names}")
    if len(set(val_names)) != len(val_names):
        raise ValueError(f"Duplicate expert names: {val_names}")

    val_rows = {name: _read_csv(path) for name, path in val_inputs}
    test_rows = {name: _read_csv(path) for name, path in test_inputs}
    val_paths = _aligned_paths(val_rows)
    test_paths = _aligned_paths(test_rows)
    if not val_paths or not test_paths:
        raise ValueError("No aligned paths for val/test selector inputs.")

    classes = _class_names(val_rows, val_paths)
    class_to_index = {name: index for index, name in enumerate(classes)}
    x_val, y_val, _ = _build_matrix(val_paths, val_names, val_rows, class_to_index)
    x_test, y_test, y_test_names = _build_matrix(test_paths, test_names, test_rows, class_to_index)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(args.max_iter),
            class_weight=("balanced" if str(args.class_weight) == "balanced" else None),
            random_state=int(args.seed),
        ),
    )
    model.fit(x_val, y_val)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    test_prob = model.predict_proba(x_test)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "validation_trained_logistic_selector",
        "selection_warning": "Selector is fit on validation only; do not change experts/hyperparameters after viewing test.",
        "experts": [{"name": name, "val_rows": len(val_rows[name]), "test_rows": len(test_rows[name])} for name in val_names],
        "val_samples": int(len(val_paths)),
        "test_samples": int(len(test_paths)),
        "feature_dim": int(x_val.shape[1]),
        "class_weight": str(args.class_weight),
        "val_metrics": _metrics(y_val, val_pred, classes),
        "test_metrics": _metrics(y_test, test_pred, classes),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with (output_dir / "predictions_test.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "true_name",
            "selector_pred_name",
            "correct",
            "confidence",
            *[f"prob_{index}" for index in range(len(classes))],
            *[f"{name}_pred" for name in val_names],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, path in enumerate(test_paths):
            prediction = int(test_pred[row_index])
            row = {
                "path": path,
                "true_name": y_test_names[row_index],
                "selector_pred_name": classes[prediction],
                "correct": int(prediction == int(y_test[row_index])),
                "confidence": float(test_prob[row_index, prediction]),
            }
            row.update({f"prob_{index}": float(test_prob[row_index, index]) for index in range(len(classes))})
            row.update({f"{name}_pred": _prediction(test_rows[name][path]) for name in val_names})
            writer.writerow(row)

    print(
        {
            "output_dir": str(output_dir),
            "val_macro_f1": payload["val_metrics"]["macro_f1"],
            "test_macro_f1": payload["test_metrics"]["macro_f1"],
            "test_accuracy": payload["test_metrics"]["accuracy"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
