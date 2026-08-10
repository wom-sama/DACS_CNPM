from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _string_array(values: object, expected: int) -> List[str]:
    if values is None:
        return ["" for _ in range(expected)]
    array = np.asarray(values)
    result = [str(value) for value in array.reshape(-1).tolist()]
    if int(expected) <= 0:
        return result
    if len(result) != int(expected):
        return ["" for _ in range(expected)]
    return result


def _load_feature_npz(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data or "labels" not in data:
            raise ValueError(f"Feature NPZ must contain features and labels: {path}")
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
        sample_index = (
            np.asarray(data["sample_index"], dtype=np.int64).reshape(-1)
            if "sample_index" in data
            else np.arange(labels.shape[0], dtype=np.int64)
        )
        source_stem = _string_array(data["source_stem"] if "source_stem" in data else None, labels.shape[0])
        paths = _string_array(data["paths"] if "paths" in data else None, labels.shape[0])
        classes = _string_array(data["classes"] if "classes" in data else None, 0)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Invalid feature/label shape: features={features.shape}, labels={labels.shape}"
        )
    if sample_index.shape[0] != labels.shape[0]:
        raise ValueError("sample_index length does not match labels")
    if not np.isfinite(features).all():
        raise ValueError(f"Feature NPZ contains non-finite values: {path}")
    return {
        "features": features,
        "labels": labels,
        "sample_index": sample_index,
        "source_stem": source_stem,
        "paths": paths,
        "classes": classes,
    }


def _class_names(train_payload: Mapping[str, object], val_payload: Mapping[str, object]) -> List[str]:
    for payload in (train_payload, val_payload):
        names = [str(value) for value in payload.get("classes", [])]  # type: ignore[arg-type]
        if names:
            return names
    labels = np.asarray(train_payload["labels"], dtype=np.int64)
    class_count = int(labels.max()) + 1 if labels.size else 0
    return [str(index) for index in range(class_count)]


def _candidate_estimators(seed: int, n_jobs: int) -> Dict[str, object]:
    return {
        "logreg_balanced_c0p3": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                class_weight="balanced",
                max_iter=800,
                random_state=int(seed),
                solver="lbfgs",
                n_jobs=int(n_jobs),
            ),
        ),
        "logreg_balanced_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=800,
                random_state=int(seed),
                solver="lbfgs",
                n_jobs=int(n_jobs),
            ),
        ),
        "knn_cosine_k31": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                metric="cosine",
                n_jobs=int(n_jobs),
            ),
        ),
        "extra_trees_leaf5": ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=int(n_jobs),
        ),
    }


def _select_candidates(all_candidates: Mapping[str, object], names: str) -> Dict[str, object]:
    requested = [part.strip() for part in str(names or "").split(",") if part.strip()]
    if not requested:
        return dict(all_candidates)
    missing = [name for name in requested if name not in all_candidates]
    if missing:
        raise ValueError(f"Unknown candidate names: {missing}; available={sorted(all_candidates)}")
    return {name: all_candidates[name] for name in requested}


def _make_splitter(labels: np.ndarray, groups: Sequence[str], folds: int, seed: int):
    fold_count = max(2, min(int(folds), int(np.bincount(labels).min())))
    unique_groups = len(set(str(group) for group in groups if str(group)))
    if unique_groups >= fold_count:
        try:
            return StratifiedGroupKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=int(seed),
            ).split(np.zeros_like(labels), labels, groups)
        except Exception:
            pass
    return StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=int(seed),
    ).split(np.zeros_like(labels), labels)


def _aligned_predict_proba(estimator, features: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    if classes is None:
        if raw.shape[1] != int(class_count):
            raise ValueError(f"Cannot align probabilities with shape {raw.shape}")
        return raw.astype(np.float32)
    aligned = np.zeros((raw.shape[0], int(class_count)), dtype=np.float64)
    for column_index, class_index in enumerate(np.asarray(classes, dtype=np.int64).reshape(-1)):
        if 0 <= int(class_index) < int(class_count):
            aligned[:, int(class_index)] = raw[:, int(column_index)]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, np.maximum(row_sum, 1e-12))
    return aligned.astype(np.float32)


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    focus_class: int,
) -> Dict[str, object]:
    predictions = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    class_count = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(class_count)),
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=list(range(class_count)))
    accuracy = float((predictions == labels).mean()) if labels.size else 0.0
    per_class = []
    for index in range(class_count):
        per_class.append(
            {
                "class_index": int(index),
                "class_name": str(class_names[index]),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)) if precision.size else 0.0,
        "macro_recall": float(np.mean(recall)) if recall.size else 0.0,
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "focus_f1": float(f1[int(focus_class)]) if 0 <= int(focus_class) < len(f1) else 0.0,
        "focus_precision": float(precision[int(focus_class)]) if 0 <= int(focus_class) < len(precision) else 0.0,
        "focus_recall": float(recall[int(focus_class)]) if 0 <= int(focus_class) < len(recall) else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _write_predictions(
    path: Path,
    *,
    sample_index: np.ndarray,
    paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    variant: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    fields = [
        "sample_index",
        "path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "confidence",
        "correct",
        "variant",
    ] + [f"prob_{index}" for index in range(len(class_names))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index in range(labels.shape[0]):
            target = int(labels[row_index])
            prediction = int(predictions[row_index])
            row = {
                "sample_index": int(sample_index[row_index]),
                "path": str(paths[row_index]) if row_index < len(paths) else "",
                "target_index": target,
                "target_name": str(class_names[target]),
                "prediction_index": prediction,
                "prediction_name": str(class_names[prediction]),
                "confidence": f"{float(probabilities[row_index, prediction]):.10g}",
                "correct": int(prediction == target),
                "variant": str(variant),
            }
            for class_index in range(len(class_names)):
                row[f"prob_{class_index}"] = f"{float(probabilities[row_index, class_index]):.10g}"
            writer.writerow(row)


def run_probe(
    *,
    train_npz: Path,
    val_npz: Path,
    output_dir: Path,
    candidates: str = "",
    folds: int = 5,
    seed: int = 42,
    n_jobs: int = 4,
    focus_class: int = 1,
    focus_class_weight: float = 0.50,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _load_feature_npz(Path(train_npz))
    val = _load_feature_npz(Path(val_npz))
    class_names = _class_names(train, val)
    class_count = len(class_names)
    train_features = np.asarray(train["features"], dtype=np.float32)
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_features = np.asarray(val["features"], dtype=np.float32)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Train and val feature dimensions do not match")

    all_candidates = _candidate_estimators(seed=int(seed), n_jobs=int(n_jobs))
    estimators = _select_candidates(all_candidates, candidates)
    groups = [str(value) for value in train["source_stem"]]  # type: ignore[index]
    metrics_rows: List[Dict[str, object]] = []
    variant_payloads: Dict[str, Dict[str, object]] = {}
    for name, estimator in estimators.items():
        splitter = list(_make_splitter(train_labels, groups, folds=folds, seed=seed))
        oof_probabilities = np.asarray(
            cross_val_predict(
                estimator,
                train_features,
                train_labels,
                cv=splitter,
                method="predict_proba",
                n_jobs=1,
            ),
            dtype=np.float32,
        )
        if oof_probabilities.shape[1] != class_count:
            raise ValueError(
                f"OOF probabilities for {name} have {oof_probabilities.shape[1]} classes; expected {class_count}"
            )
        estimator.fit(train_features, train_labels)
        val_probabilities = _aligned_predict_proba(estimator, val_features, class_count)
        oof_metrics = _metrics(
            train_labels,
            oof_probabilities,
            class_names=class_names,
            focus_class=int(focus_class),
        )
        val_metrics = _metrics(
            val_labels,
            val_probabilities,
            class_names=class_names,
            focus_class=int(focus_class),
        )
        selection_score = float(oof_metrics["macro_f1"]) + float(focus_class_weight) * float(
            oof_metrics["focus_f1"]
        )
        metrics_rows.append(
            {
                "variant": name,
                "selection_score": selection_score,
                "oof_macro_f1": float(oof_metrics["macro_f1"]),
                "oof_focus_f1": float(oof_metrics["focus_f1"]),
                "oof_focus_precision": float(oof_metrics["focus_precision"]),
                "oof_focus_recall": float(oof_metrics["focus_recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_focus_f1": float(val_metrics["focus_f1"]),
                "val_focus_precision": float(val_metrics["focus_precision"]),
                "val_focus_recall": float(val_metrics["focus_recall"]),
            }
        )
        variant_payloads[name] = {
            "oof_probabilities": oof_probabilities,
            "val_probabilities": val_probabilities,
            "oof_metrics": oof_metrics,
            "val_metrics": val_metrics,
        }

    metrics_rows.sort(key=lambda row: (float(row["selection_score"]), float(row["oof_macro_f1"])), reverse=True)
    selected = str(metrics_rows[0]["variant"])
    selected_payload = variant_payloads[selected]

    metrics_csv = output_dir / "variant_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(metrics_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_rows)

    _write_predictions(
        output_dir / "train_oof_predictions_selected.csv",
        sample_index=np.asarray(train["sample_index"], dtype=np.int64),
        paths=[str(value) for value in train["paths"]],  # type: ignore[index]
        labels=train_labels,
        probabilities=np.asarray(selected_payload["oof_probabilities"], dtype=np.float32),
        class_names=class_names,
        variant=selected,
    )
    _write_predictions(
        output_dir / "val_predictions_selected.csv",
        sample_index=np.asarray(val["sample_index"], dtype=np.int64),
        paths=[str(value) for value in val["paths"]],  # type: ignore[index]
        labels=val_labels,
        probabilities=np.asarray(selected_payload["val_probabilities"], dtype=np.float32),
        class_names=class_names,
        variant=selected,
    )

    train_counts = Counter(int(value) for value in train_labels)
    val_counts = Counter(int(value) for value in val_labels)
    summary = {
        "train_npz": str(Path(train_npz).resolve()),
        "val_npz": str(Path(val_npz).resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "class_names": [str(name) for name in class_names],
        "train_class_counts": {str(index): int(train_counts[index]) for index in range(class_count)},
        "val_class_counts": {str(index): int(val_counts[index]) for index in range(class_count)},
        "folds": int(folds),
        "seed": int(seed),
        "candidate_order": [str(row["variant"]) for row in metrics_rows],
        "selected_variant": selected,
        "selection_source": "train_oof",
        "focus_class": int(focus_class),
        "focus_class_weight": float(focus_class_weight),
        "variant_metrics": metrics_rows,
        "selected_oof_metrics": selected_payload["oof_metrics"],
        "selected_val_metrics": selected_payload["val_metrics"],
        "leakage_guard": (
            "candidate selection uses train OOF only; validation is reported for diagnosis; "
            "test split is not accepted by this tool"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fold-safe OOF readouts on pretrained feature caches and evaluate validation."
    )
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--focus-class-weight", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    for forbidden in ("test",):
        if forbidden in str(args.train_npz).lower() or forbidden in str(args.val_npz).lower():
            raise ValueError("This diagnostic refuses test feature caches.")
    run_probe(
        train_npz=Path(args.train_npz),
        val_npz=Path(args.val_npz),
        output_dir=Path(args.output_dir),
        candidates=str(args.candidates),
        folds=int(args.folds),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        focus_class=int(args.focus_class),
        focus_class_weight=float(args.focus_class_weight),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
