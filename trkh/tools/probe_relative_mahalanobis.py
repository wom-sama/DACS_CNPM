from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _extract_split_embeddings,
    _resolve_device,
)
from trkh.models.model import build_model_from_checkpoint


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only relative Mahalanobis diagnostic for frozen TRKH embeddings. "
            "The probe fits class means/covariances on train split and applies "
            "threshold-free class-1 conformity guards on validation. It never reads "
            "test unless explicitly requested."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--class1-index", type=int, default=1)
    parser.add_argument("--variance-floor", type=float, default=1e-4)
    parser.add_argument("--class-variance-shrinkage", type=float, default=0.25)
    return parser.parse_args(argv)


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
    return array


def _safe_l2_normalize(values: np.ndarray) -> np.ndarray:
    array = _as_2d_float(values)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norm, 1e-12)).astype(np.float32, copy=False)


def _fit_standardizer(train_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = _as_2d_float(train_features)
    mean = features.mean(axis=0).astype(np.float32, copy=False)
    std = features.std(axis=0).astype(np.float32, copy=False)
    std = np.maximum(std, 1e-6).astype(np.float32, copy=False)
    return mean, std


def _apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((_as_2d_float(features) - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(
        np.float32,
        copy=False,
    )


def _class_means(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    values = _as_2d_float(features)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("labels must have one value per feature row.")
    global_mean = values.mean(axis=0)
    means = np.zeros((class_count, values.shape[1]), dtype=np.float32)
    for class_index in range(class_count):
        mask = labels == int(class_index)
        means[class_index] = values[mask].mean(axis=0) if bool(mask.any()) else global_mean
    return means.astype(np.float32, copy=False)


def _shared_diag_variance(
    features: np.ndarray,
    *,
    variance_floor: float,
) -> np.ndarray:
    values = _as_2d_float(features)
    variance = values.var(axis=0).astype(np.float32, copy=False)
    return np.maximum(variance, float(variance_floor)).astype(np.float32, copy=False)


def _class_diag_variance(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    *,
    variance_floor: float,
    shrinkage: float,
) -> np.ndarray:
    values = _as_2d_float(features)
    labels = np.asarray(labels, dtype=np.int64)
    global_variance = _shared_diag_variance(values, variance_floor=variance_floor)
    shrinkage = min(1.0, max(0.0, float(shrinkage)))
    variances = np.zeros((class_count, values.shape[1]), dtype=np.float32)
    for class_index in range(class_count):
        mask = labels == int(class_index)
        if bool(mask.any()):
            local = values[mask].var(axis=0).astype(np.float32, copy=False)
            local = np.maximum(local, float(variance_floor)).astype(np.float32, copy=False)
        else:
            local = global_variance
        variances[class_index] = (1.0 - shrinkage) * local + shrinkage * global_variance
    return np.maximum(variances, float(variance_floor)).astype(np.float32, copy=False)


def _diag_mahalanobis_scores(
    features: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    values = _as_2d_float(features)
    means = np.asarray(means, dtype=np.float32)
    variances = np.asarray(variances, dtype=np.float32)
    diff = values[:, None, :] - means[None, :, :]
    if variances.ndim == 1:
        scores = (diff * diff / variances.reshape(1, 1, -1)).sum(axis=2)
    elif variances.ndim == 2:
        scores = (diff * diff / variances.reshape(1, variances.shape[0], -1)).sum(axis=2)
    else:
        raise ValueError("variances must be 1D or 2D.")
    return scores.astype(np.float32, copy=False)


def _shared_cov_precision(
    features: np.ndarray,
    *,
    variance_floor: float,
) -> Optional[np.ndarray]:
    values = _as_2d_float(features)
    try:
        from sklearn.covariance import LedoitWolf
    except Exception:
        return None
    try:
        covariance = LedoitWolf().fit(values).covariance_.astype(np.float32, copy=False)
        covariance = covariance + np.eye(covariance.shape[0], dtype=np.float32) * float(variance_floor)
        precision = np.linalg.pinv(covariance).astype(np.float32, copy=False)
        return precision
    except Exception:
        return None


def _full_mahalanobis_scores(
    features: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    values = _as_2d_float(features)
    means = np.asarray(means, dtype=np.float32)
    precision = np.asarray(precision, dtype=np.float32)
    diff = values[:, None, :] - means[None, :, :]
    scores = np.einsum("ncd,df,ncf->nc", diff, precision, diff, optimize=True)
    return scores.astype(np.float32, copy=False)


def _score_margin_to_class1(scores: np.ndarray, class1_index: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    class1_index = int(class1_index)
    if not 0 <= class1_index < values.shape[1]:
        raise ValueError("class1_index is outside score columns.")
    rival = values.copy()
    rival[:, class1_index] = np.inf
    return (values[:, class1_index] - np.min(rival, axis=1)).astype(np.float32, copy=False)


def _guard_class1_predictions(
    base_predictions: np.ndarray,
    nearest_predictions: np.ndarray,
    *,
    class1_index: int = 1,
    allowed_targets: Optional[Sequence[int]] = None,
) -> np.ndarray:
    base = np.asarray(base_predictions, dtype=np.int64)
    nearest = np.asarray(nearest_predictions, dtype=np.int64)
    if base.shape != nearest.shape:
        raise ValueError("base_predictions and nearest_predictions must have the same shape.")
    guarded = base.copy()
    mask = np.logical_and(base == int(class1_index), nearest != int(class1_index))
    if allowed_targets is not None:
        allowed = np.asarray([int(value) for value in allowed_targets], dtype=np.int64)
        mask = np.logical_and(mask, np.isin(nearest, allowed))
    guarded[mask] = nearest[mask]
    return guarded


def _change_summary(
    targets: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    changed = before != after
    corrections = np.logical_and(changed, np.logical_and(before != targets, after == targets))
    harms = np.logical_and(changed, np.logical_and(before == targets, after != targets))
    neutral = np.logical_and(changed, np.logical_and(before != targets, after != targets))
    transitions: Dict[str, int] = {}
    for target, old, new, is_changed in zip(targets, before, after, changed):
        if not bool(is_changed):
            continue
        key = f"{int(target)}:{int(old)}->{int(new)}"
        transitions[key] = int(transitions.get(key, 0) + 1)
    return {
        "changed": int(changed.sum()),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "neutral": int(neutral.sum()),
        "transitions": dict(sorted(transitions.items(), key=lambda item: (-item[1], item[0]))),
    }


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    paths: Sequence[str],
    probabilities: np.ndarray,
    predictions_by_method: Mapping[str, np.ndarray],
    margins_by_method: Mapping[str, np.ndarray],
) -> None:
    prediction_names = list(predictions_by_method.keys())
    margin_names = list(margins_by_method.keys())
    class_count = int(np.asarray(probabilities).shape[1]) if probabilities.size else 0
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        *[f"prob_{index}" for index in range(class_count)],
        *[f"pred_{name}" for name in prediction_names],
        *[f"class1_margin_{name}" for name in margin_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(targets):
            row = {
                "split": split,
                "sample_index": int(index),
                "image_path": str(paths[index]) if index < len(paths) else "",
                "target_index": int(target),
            }
            for class_index in range(class_count):
                row[f"prob_{class_index}"] = f"{float(probabilities[index, class_index]):.9g}"
            for name in prediction_names:
                row[f"pred_{name}"] = int(predictions_by_method[name][index])
            for name in margin_names:
                row[f"class1_margin_{name}"] = f"{float(margins_by_method[name][index]):.9g}"
            writer.writerow(row)


def _write_changed_cases(
    path: Path,
    *,
    split: str,
    method: str,
    targets: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    paths: Sequence[str],
    margins: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "method",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "class1_margin",
        "change_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (target, old, new) in enumerate(zip(targets, before, after)):
            if int(old) == int(new):
                continue
            if int(new) == int(target) and int(old) != int(target):
                change_type = "correction"
            elif int(old) == int(target) and int(new) != int(target):
                change_type = "harm"
            else:
                change_type = "neutral"
            writer.writerow(
                {
                    "split": split,
                    "method": method,
                    "sample_index": int(index),
                    "image_path": str(paths[index]) if index < len(paths) else "",
                    "target_index": int(target),
                    "base_prediction": int(old),
                    "final_prediction": int(new),
                    "class1_margin": f"{float(margins[index]):.9g}",
                    "change_type": change_type,
                }
            )


def _prepare_feature_views(train_embeddings: np.ndarray, embeddings: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    train_raw = _as_2d_float(train_embeddings)
    eval_raw = _as_2d_float(embeddings)
    raw_mean, raw_std = _fit_standardizer(train_raw)
    train_std = _apply_standardizer(train_raw, raw_mean, raw_std)
    eval_std = _apply_standardizer(eval_raw, raw_mean, raw_std)

    train_l2 = _safe_l2_normalize(train_raw)
    eval_l2 = _safe_l2_normalize(eval_raw)
    l2_mean, l2_std = _fit_standardizer(train_l2)
    train_l2std = _apply_standardizer(train_l2, l2_mean, l2_std)
    eval_l2std = _apply_standardizer(eval_l2, l2_mean, l2_std)
    return {
        "std": (train_std, eval_std),
        "l2std": (train_l2std, eval_l2std),
    }


def _build_score_methods(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
    variance_floor: float,
    class_variance_shrinkage: float,
) -> Dict[str, np.ndarray]:
    methods: Dict[str, np.ndarray] = {}
    for view_name, (train_features, eval_features) in _prepare_feature_views(
        train_embeddings,
        eval_embeddings,
    ).items():
        means = _class_means(train_features, train_labels, class_count)
        shared_variance = _shared_diag_variance(
            train_features,
            variance_floor=variance_floor,
        )
        methods[f"diag_shared_{view_name}"] = _diag_mahalanobis_scores(
            eval_features,
            means,
            shared_variance,
        )
        class_variance = _class_diag_variance(
            train_features,
            train_labels,
            class_count,
            variance_floor=variance_floor,
            shrinkage=class_variance_shrinkage,
        )
        methods[f"diag_class_{view_name}"] = _diag_mahalanobis_scores(
            eval_features,
            means,
            class_variance,
        )
        precision = _shared_cov_precision(train_features, variance_floor=variance_floor)
        if precision is not None:
            methods[f"ledoit_shared_{view_name}"] = _full_mahalanobis_scores(
                eval_features,
                means,
                precision,
            )
    return methods


def _method_sort_key(item: Tuple[str, Mapping[str, object]], class1_index: int) -> Tuple[float, float, float, float]:
    metrics = item[1]
    per_class = list(metrics.get("per_class", []))
    class1 = per_class[int(class1_index)] if len(per_class) > int(class1_index) else {}
    return (
        float(metrics.get("macro_f1", 0.0)),
        float(class1.get("f1", 0.0)),
        float(class1.get("precision", 0.0)),
        float(metrics.get("accuracy", 0.0)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    requested_splits = list(args.split or ["train", "val"])
    if "train" not in requested_splits:
        raise ValueError("Relative Mahalanobis probe requires train split statistics.")

    start_time = time.perf_counter()
    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    for split in requested_splits:
        max_samples = int(args.max_train_samples) if split == "train" else int(args.max_eval_samples)
        dataset, class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=str(split),
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples,
        )
        split_payloads[str(split)] = _extract_split_embeddings(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=str(split),
        )

    class_count = len(class_names)
    class1_index = int(args.class1_index)
    if not 0 <= class1_index < class_count:
        raise ValueError("--class1-index is outside class range.")
    train = split_payloads["train"]
    train_embeddings = np.asarray(train["embeddings"], dtype=np.float32)
    train_labels = np.asarray(train["labels"], dtype=np.int64)

    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "class1_index": int(class1_index),
        "variance_floor": float(args.variance_floor),
        "class_variance_shrinkage": float(args.class_variance_shrinkage),
        "leakage_guard": (
            "class means, variances, and covariance precision are fit on train split only; "
            "validation metrics are diagnostic and test is not read unless requested"
        ),
        "splits": {},
    }

    for split, payload in split_payloads.items():
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        base_predictions = np.asarray(payload["base_predictions"], dtype=np.int64)
        score_methods = _build_score_methods(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            eval_embeddings=embeddings,
            class_count=class_count,
            variance_floor=float(args.variance_floor),
            class_variance_shrinkage=float(args.class_variance_shrinkage),
        )
        predictions_by_method: Dict[str, np.ndarray] = {"base_head": base_predictions}
        margins_by_method: Dict[str, np.ndarray] = {}
        change_summaries: Dict[str, object] = {}
        for score_name, scores in score_methods.items():
            nearest = np.argmin(scores, axis=1).astype(np.int64)
            predictions_by_method[f"nearest_{score_name}"] = nearest
            margins = _score_margin_to_class1(scores, class1_index)
            margins_by_method[score_name] = margins
            guarded_all = _guard_class1_predictions(
                base_predictions,
                nearest,
                class1_index=class1_index,
            )
            guarded_to0 = _guard_class1_predictions(
                base_predictions,
                nearest,
                class1_index=class1_index,
                allowed_targets=[0],
            )
            predictions_by_method[f"guard1_any_{score_name}"] = guarded_all
            predictions_by_method[f"guard1_to0_{score_name}"] = guarded_to0
            change_summaries[f"guard1_any_{score_name}"] = _change_summary(
                labels,
                base_predictions,
                guarded_all,
            )
            change_summaries[f"guard1_to0_{score_name}"] = _change_summary(
                labels,
                base_predictions,
                guarded_to0,
            )
            _write_changed_cases(
                output_dir / f"changed_cases_{split}_guard1_any_{score_name}.csv",
                split=str(split),
                method=f"guard1_any_{score_name}",
                targets=labels,
                before=base_predictions,
                after=guarded_all,
                paths=payload.get("paths", []),
                margins=margins,
            )
            _write_changed_cases(
                output_dir / f"changed_cases_{split}_guard1_to0_{score_name}.csv",
                split=str(split),
                method=f"guard1_to0_{score_name}",
                targets=labels,
                before=base_predictions,
                after=guarded_to0,
                paths=payload.get("paths", []),
                margins=margins,
            )

        metrics = {
            method: _classification_metrics(labels, predictions, class_names)
            for method, predictions in predictions_by_method.items()
        }
        guard_metrics = {
            method: metric
            for method, metric in metrics.items()
            if method.startswith("guard1_")
        }
        best_guard = None
        if guard_metrics:
            best_guard = max(
                guard_metrics.items(),
                key=lambda item: _method_sort_key(item, class1_index),
            )[0]
        summary["splits"][str(split)] = {
            "samples": int(labels.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "class_counts": [
                int((labels == class_index_value).sum())
                for class_index_value in range(class_count)
            ],
            "methods": list(predictions_by_method.keys()),
            "metrics": metrics,
            "change_summaries": change_summaries,
            "best_guard_by_macro_then_class1": best_guard,
        }
        _write_predictions(
            output_dir / f"{split}_relative_mahalanobis_predictions.csv",
            split=str(split),
            targets=labels,
            paths=payload.get("paths", []),
            probabilities=probabilities,
            predictions_by_method=predictions_by_method,
            margins_by_method=margins_by_method,
        )

    summary["seconds"] = float(time.perf_counter() - start_time)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compact: Dict[str, object] = {
        "output_dir": str(output_dir),
        "seconds": round(float(summary["seconds"]), 2),
        "splits": {},
    }
    for split, split_summary in summary["splits"].items():
        metrics = split_summary["metrics"]
        best_guard = split_summary.get("best_guard_by_macro_then_class1")
        compact["splits"][split] = {
            "samples": split_summary["samples"],
            "base": {
                "macro_f1": metrics["base_head"]["macro_f1"],
                "class1_f1": metrics["base_head"]["per_class"][class1_index]["f1"],
            },
            "best_guard": best_guard,
            "best_guard_metrics": (
                {
                    "macro_f1": metrics[best_guard]["macro_f1"],
                    "class1_f1": metrics[best_guard]["per_class"][class1_index]["f1"],
                    "class1_precision": metrics[best_guard]["per_class"][class1_index]["precision"],
                    "class1_recall": metrics[best_guard]["per_class"][class1_index]["recall"],
                    "changes": split_summary["change_summaries"].get(best_guard, {}),
                }
                if isinstance(best_guard, str)
                else None
            ),
        }
    print(json.dumps(compact, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
