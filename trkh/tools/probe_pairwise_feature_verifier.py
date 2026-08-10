from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _extract_split_embeddings,
    _resolve_device,
)


Pair = Tuple[int, int]


def parse_pairs(value: str) -> List[Pair]:
    pairs: List[Pair] = []
    for item in str(value or "").split(","):
        text = item.strip()
        if not text:
            continue
        if "-" not in text:
            raise ValueError(f"Pair must use A-B syntax: {text!r}")
        left, right = text.split("-", 1)
        a = int(left.strip())
        b = int(right.strip())
        if a == b:
            raise ValueError(f"Pair must contain two different classes: {text!r}")
        pair = (min(a, b), max(a, b))
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("At least one pair is required.")
    return pairs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only pairwise verifier heads on frozen TRKH embeddings and "
            "apply them to boundary predictions. This is a diagnostic; it does "
            "not edit raw data or read test by default."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="0-1,1-2,1-4,2-3")
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--min-pair-probability", type=float, default=0.02)
    parser.add_argument("--max-pair-margin", type=float, default=0.40)
    parser.add_argument("--verifier-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--verifier-model", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--extra-trees-n-estimators", type=int, default=300)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=5)
    parser.add_argument("--extra-trees-max-depth", type=int, default=0)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_verifier_features(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    logs = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32, copy=False)
    sorted_probs = np.sort(probs, axis=1)
    margins = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    confidence = sorted_probs[:, -1].reshape(-1, 1)
    return np.concatenate(
        [
            np.asarray(embeddings, dtype=np.float32),
            probs,
            logs,
            margins.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
        ],
        axis=1,
    )


def build_verifier_feature_names(*, embedding_dim: int, class_count: int) -> List[str]:
    embedding_dim = max(0, int(embedding_dim))
    class_count = max(0, int(class_count))
    return (
        [f"embedding_{index}" for index in range(embedding_dim)]
        + [f"prob_{class_index}" for class_index in range(class_count)]
        + [f"log_prob_{class_index}" for class_index in range(class_count)]
        + ["prob_margin_top1_top2", "prob_confidence_top1"]
    )


def _make_model(
    *,
    model_type: str = "logistic",
    c_value: float,
    max_iter: int,
    seed: int,
    extra_trees_n_estimators: int = 300,
    extra_trees_min_samples_leaf: int = 5,
    extra_trees_max_depth: int = 0,
):
    if str(model_type) == "extra_trees":
        max_depth = int(extra_trees_max_depth)
        return ExtraTreesClassifier(
            n_estimators=max(10, int(extra_trees_n_estimators)),
            min_samples_leaf=max(1, int(extra_trees_min_samples_leaf)),
            max_depth=max_depth if max_depth > 0 else None,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=max(1e-6, float(c_value)),
            class_weight="balanced",
            max_iter=max(100, int(max_iter)),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )


def _linear_logistic_export(model: object) -> Optional[Dict[str, object]]:
    """Export a sklearn StandardScaler+LogisticRegression pipeline as raw-space weights."""

    named_steps = getattr(model, "named_steps", None)
    if not isinstance(named_steps, Mapping):
        return None
    scaler = named_steps.get("standardscaler")
    classifier = named_steps.get("logisticregression")
    if scaler is None or classifier is None:
        return None
    coef = np.asarray(getattr(classifier, "coef_", None), dtype=np.float64)
    intercept = np.asarray(getattr(classifier, "intercept_", None), dtype=np.float64)
    classes = np.asarray(getattr(classifier, "classes_", []), dtype=np.int64)
    mean = np.asarray(getattr(scaler, "mean_", None), dtype=np.float64)
    scale = np.asarray(getattr(scaler, "scale_", None), dtype=np.float64)
    if coef.ndim != 2 or coef.shape[0] != 1 or intercept.shape != (1,):
        return None
    if mean.ndim != 1 or scale.ndim != 1 or mean.shape[0] != coef.shape[1] or scale.shape[0] != coef.shape[1]:
        return None
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    raw_coef = coef[0] / scale
    raw_intercept = float(intercept[0] - np.sum(coef[0] * mean / scale))
    return {
        "model_type": "standardized_logistic_regression",
        "classes": [int(value) for value in classes.tolist()],
        "feature_dim": int(coef.shape[1]),
        "standardizer_mean": mean.astype(float).tolist(),
        "standardizer_scale": scale.astype(float).tolist(),
        "standardized_coef": coef[0].astype(float).tolist(),
        "standardized_intercept": float(intercept[0]),
        "raw_coef": raw_coef.astype(float).tolist(),
        "raw_intercept": raw_intercept,
        "raw_logit_semantics": "logit(class=1) over local pair labels; sigmoid(raw_coef @ features + raw_intercept)",
    }


def pair_model_export_payload(
    models: Mapping[Pair, object],
    *,
    feature_dim: int,
    metadata: Optional[Mapping[str, object]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    feature_name_list = [str(name) for name in feature_names] if feature_names is not None else []
    if feature_name_list and len(feature_name_list) != int(feature_dim):
        raise ValueError(
            f"feature_names length {len(feature_name_list)} does not match feature_dim {int(feature_dim)}."
        )
    pair_payloads: List[Dict[str, object]] = []
    exportable = 0
    for pair, model in sorted(models.items()):
        item: Dict[str, object] = {
            "pair": f"{int(pair[0])}-{int(pair[1])}",
            "left_class": int(pair[0]),
            "right_class": int(pair[1]),
            "status": "unsupported",
        }
        export = _linear_logistic_export(model)
        if export is not None:
            item.update(export)
            item["status"] = "exported"
            exportable += 1
        pair_payloads.append(item)
    return {
        "feature_dim": int(feature_dim),
        "feature_names": feature_name_list,
        "exportable_pairs": int(exportable),
        "pairs": pair_payloads,
        "metadata": dict(metadata or {}),
        "leakage_guard": (
            "Exports only train-fitted verifier parameters. Use validation/test only "
            "through the calling diagnostic's normal split policy."
        ),
    }


def write_pair_model_export(
    path: Path,
    models: Mapping[Pair, object],
    *,
    feature_dim: int,
    metadata: Optional[Mapping[str, object]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    payload = pair_model_export_payload(
        models,
        feature_dim=int(feature_dim),
        metadata=metadata,
        feature_names=feature_names,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "feature_dim": int(feature_dim),
        "feature_names": int(len(payload.get("feature_names", []))),
        "exportable_pairs": int(payload.get("exportable_pairs", 0)),
        "pairs": [str(item.get("pair", "")) for item in payload.get("pairs", [])],
    }


def _sample_weight_fit_params(model_type: str, weights: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if weights is None:
        return {}
    if str(model_type) == "extra_trees":
        return {"sample_weight": weights}
    return {"logisticregression__sample_weight": weights}


def _local_pair_labels(labels: np.ndarray, pair: Pair) -> Tuple[np.ndarray, np.ndarray]:
    a, b = pair
    mask = np.logical_or(labels == int(a), labels == int(b))
    local = (labels[mask] == int(b)).astype(np.int64)
    return mask, local


def _fit_pair_models(
    features: np.ndarray,
    labels: np.ndarray,
    pairs: Sequence[Pair],
    *,
    model_type: str = "logistic",
    c_value: float,
    max_iter: int,
    seed: int,
    oof_folds: int,
    sample_weights: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    extra_trees_n_estimators: int = 300,
    extra_trees_min_samples_leaf: int = 5,
    extra_trees_max_depth: int = 0,
) -> Tuple[Dict[Pair, object], List[Dict[str, object]], Dict[Pair, np.ndarray]]:
    models: Dict[Pair, object] = {}
    summaries: List[Dict[str, object]] = []
    oof_probabilities: Dict[Pair, np.ndarray] = {}
    weights = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != np.asarray(labels).shape[0]:
            raise ValueError(
                "sample_weights must have one value per feature row: "
                f"got {weights.shape[0]} for {np.asarray(labels).shape[0]} labels"
            )
    source_groups = None
    if groups is not None:
        source_groups = np.asarray(groups, dtype=object).reshape(-1)
        if source_groups.shape[0] != np.asarray(labels).shape[0]:
            raise ValueError(
                "groups must have one value per feature row: "
                f"got {source_groups.shape[0]} for {np.asarray(labels).shape[0]} labels"
            )
        if any(not str(value).strip() for value in source_groups.tolist()):
            raise ValueError("groups must not contain blank source identifiers")
    for pair in pairs:
        mask, local = _local_pair_labels(labels, pair)
        pair_features = features[mask]
        pair_weights = weights[mask] if weights is not None else None
        pair_groups = source_groups[mask] if source_groups is not None else None
        counts = np.bincount(local, minlength=2)
        if pair_features.shape[0] == 0 or int((counts > 0).sum()) < 2:
            summaries.append(
                {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "status": "skipped",
                    "samples": int(pair_features.shape[0]),
                    "class_counts": counts.tolist(),
                }
            )
            continue
        model = _make_model(
            model_type=str(model_type),
            c_value=c_value,
            max_iter=max_iter,
            seed=seed,
            extra_trees_n_estimators=int(extra_trees_n_estimators),
            extra_trees_min_samples_leaf=int(extra_trees_min_samples_leaf),
            extra_trees_max_depth=int(extra_trees_max_depth),
        )
        fit_params = _sample_weight_fit_params(str(model_type), pair_weights)
        model.fit(pair_features, local, **fit_params)
        models[pair] = model

        oof_available = False
        oof = np.full((pair_features.shape[0], 2), np.nan, dtype=np.float32)
        max_folds = min(int(oof_folds), int(counts.min()))
        if pair_groups is not None:
            class_group_counts = [
                int(np.unique(pair_groups[local == class_index]).size)
                for class_index in range(2)
            ]
            max_folds = min(
                max_folds,
                int(np.unique(pair_groups).size),
                *class_group_counts,
            )
        folds = int(max_folds) if int(max_folds) >= 2 else 0
        oof_splitter = "none"
        oof_source_groups = int(np.unique(pair_groups).size) if pair_groups is not None else 0
        oof_fold_source_overlap_max = 0
        if pair_groups is not None and folds < 2:
            raise ValueError(
                "source-group OOF requires at least two source groups for each "
                f"local class in pair {pair[0]}-{pair[1]}"
            )
        if folds >= 2:
            try:
                if pair_groups is not None:
                    splitter = StratifiedGroupKFold(
                        n_splits=folds,
                        shuffle=True,
                        random_state=int(seed),
                    )
                    cv = list(splitter.split(pair_features, local, groups=pair_groups))
                    overlaps = []
                    for fit_indices, held_indices in cv:
                        fit_groups = set(pair_groups[fit_indices].tolist())
                        held_groups = set(pair_groups[held_indices].tolist())
                        overlaps.append(len(fit_groups.intersection(held_groups)))
                    oof_fold_source_overlap_max = max(overlaps, default=0)
                    if oof_fold_source_overlap_max != 0:
                        raise RuntimeError(
                            "StratifiedGroupKFold produced overlapping source groups"
                        )
                    oof_splitter = "stratified_group_kfold"
                else:
                    cv = StratifiedKFold(
                        n_splits=folds,
                        shuffle=True,
                        random_state=int(seed),
                    )
                    oof_splitter = "stratified_kfold"
                oof = cross_val_predict(
                    _make_model(
                        model_type=str(model_type),
                        c_value=c_value,
                        max_iter=max_iter,
                        seed=seed,
                        extra_trees_n_estimators=int(extra_trees_n_estimators),
                        extra_trees_min_samples_leaf=int(extra_trees_min_samples_leaf),
                        extra_trees_max_depth=int(extra_trees_max_depth),
                    ),
                    pair_features,
                    local,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=1,
                    params=fit_params or None,
                ).astype(np.float32, copy=False)
                oof_available = True
            except Exception:
                if pair_groups is not None:
                    raise
                oof_available = False
        if oof_available:
            local_pred = oof.argmax(axis=1)
            local_acc = float((local_pred == local).mean())
        else:
            local_acc = float("nan")
        oof_probabilities[pair] = oof
        summaries.append(
            {
                "pair": f"{pair[0]}-{pair[1]}",
                "status": "fit",
                "model_type": str(model_type),
                "samples": int(pair_features.shape[0]),
                "class_counts": counts.tolist(),
                "oof_folds": int(folds) if oof_available else 0,
                "oof_splitter": str(oof_splitter) if oof_available else "none",
                "oof_source_groups": int(oof_source_groups),
                "oof_fold_source_overlap_max": int(oof_fold_source_overlap_max),
                "oof_local_accuracy": local_acc,
                "sample_weight_min": float(np.min(pair_weights)) if pair_weights is not None else 1.0,
                "sample_weight_max": float(np.max(pair_weights)) if pair_weights is not None else 1.0,
                "sample_weight_mean": float(np.mean(pair_weights)) if pair_weights is not None else 1.0,
            }
        )
    return models, summaries, oof_probabilities


def apply_pairwise_verifiers(
    probabilities: np.ndarray,
    base_predictions: np.ndarray,
    models: Mapping[Pair, object],
    *,
    min_pair_probability: float,
    max_pair_margin: float,
    verifier_confidence_threshold: float,
    features: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    final = np.asarray(base_predictions, dtype=np.int64).copy()
    changes: List[Dict[str, object]] = []
    probs = np.asarray(probabilities, dtype=np.float32)
    for row_index in range(probs.shape[0]):
        current = int(final[row_index])
        best: Optional[Dict[str, object]] = None
        for pair, model in models.items():
            a, b = pair
            if current not in pair and int(np.argmax(probs[row_index])) not in pair:
                continue
            pair_probs = probs[row_index, [a, b]]
            if float(pair_probs.min()) < float(min_pair_probability):
                continue
            if abs(float(pair_probs[0]) - float(pair_probs[1])) > float(max_pair_margin):
                continue
            pair_prediction_prob = model.predict_proba(features[row_index : row_index + 1])[0]
            local_prediction = int(np.argmax(pair_prediction_prob))
            candidate = int(b if local_prediction == 1 else a)
            confidence = float(pair_prediction_prob[local_prediction])
            if candidate == current or confidence < float(verifier_confidence_threshold):
                continue
            score = confidence - float(verifier_confidence_threshold)
            proposal = {
                "sample_index": int(row_index),
                "pair": f"{a}-{b}",
                "from_prediction": int(current),
                "to_prediction": int(candidate),
                "verifier_confidence": confidence,
                "pair_probability_a": float(pair_probs[0]),
                "pair_probability_b": float(pair_probs[1]),
                "pair_probability_margin": abs(float(pair_probs[0]) - float(pair_probs[1])),
                "score": score,
            }
            if best is None or score > float(best["score"]):
                best = proposal
        if best is not None:
            final[row_index] = int(best["to_prediction"])
            changes.append(best)
    return final, changes


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    probabilities: np.ndarray,
    paths: Sequence[str],
    changes: Sequence[Mapping[str, object]],
    pair_probabilities: Optional[Mapping[Pair, np.ndarray]] = None,
) -> None:
    change_by_index = {int(item["sample_index"]): item for item in changes}
    pair_probabilities = dict(pair_probabilities or {})
    pair_fieldnames: List[str] = []
    for pair in sorted(pair_probabilities):
        pair_name = f"{int(pair[0])}_{int(pair[1])}"
        pair_fieldnames.extend(
            [
                f"verifier_{pair_name}_prob_{int(pair[0])}",
                f"verifier_{pair_name}_prob_{int(pair[1])}",
            ]
        )
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "changed",
        "change_pair",
        "verifier_confidence",
    ] + [f"prob_{index}" for index in range(probabilities.shape[1])] + pair_fieldnames
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(targets):
            change = change_by_index.get(int(index), {})
            row = {
                "split": split,
                "sample_index": int(index),
                "image_path": str(paths[index]) if index < len(paths) else "",
                "target_index": int(target),
                "base_prediction": int(base_predictions[index]),
                "final_prediction": int(final_predictions[index]),
                "changed": int(int(base_predictions[index]) != int(final_predictions[index])),
                "change_pair": str(change.get("pair", "")),
                "verifier_confidence": change.get("verifier_confidence", ""),
            }
            for class_index in range(probabilities.shape[1]):
                row[f"prob_{class_index}"] = float(probabilities[index, class_index])
            for pair, pair_probs in sorted(pair_probabilities.items()):
                pair_name = f"{int(pair[0])}_{int(pair[1])}"
                if (
                    index < int(pair_probs.shape[0])
                    and pair_probs.shape[1] >= 2
                    and np.isfinite(pair_probs[index]).all()
                ):
                    row[f"verifier_{pair_name}_prob_{int(pair[0])}"] = float(pair_probs[index, 0])
                    row[f"verifier_{pair_name}_prob_{int(pair[1])}"] = float(pair_probs[index, 1])
                else:
                    row[f"verifier_{pair_name}_prob_{int(pair[0])}"] = ""
                    row[f"verifier_{pair_name}_prob_{int(pair[1])}"] = ""
            writer.writerow(row)


def _pair_verifier_probabilities_for_split(
    *,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    models: Mapping[Pair, object],
    oof_probabilities: Mapping[Pair, np.ndarray],
) -> Dict[Pair, np.ndarray]:
    """Return full-row pair probabilities, replacing train pair rows with OOF values."""

    labels = np.asarray(labels, dtype=np.int64)
    result: Dict[Pair, np.ndarray] = {}
    for pair, model in models.items():
        full = model.predict_proba(features).astype(np.float32, copy=False)
        if full.shape != (labels.shape[0], 2):
            raise ValueError(
                f"Pair verifier {pair} returned probabilities with shape "
                f"{full.shape}; expected {(labels.shape[0], 2)}"
            )
        if split == "train" and pair in oof_probabilities:
            pair_mask, _local = _local_pair_labels(labels, pair)
            pair_indices = np.flatnonzero(pair_mask)
            oof = np.asarray(oof_probabilities[pair], dtype=np.float32)
            if oof.shape == (pair_indices.shape[0], 2):
                full = full.copy()
                full[pair_indices] = oof
        result[pair] = full
    return result


def _change_summary(
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
) -> Dict[str, object]:
    changed = base_predictions.astype(np.int64) != final_predictions.astype(np.int64)
    before_correct = base_predictions.astype(np.int64) == targets.astype(np.int64)
    after_correct = final_predictions.astype(np.int64) == targets.astype(np.int64)
    return {
        "changed": int(changed.sum()),
        "corrections": int(np.logical_and(changed, np.logical_and(~before_correct, after_correct)).sum()),
        "harms": int(np.logical_and(changed, np.logical_and(before_correct, ~after_correct)).sum()),
        "neutral_changes": int(np.logical_and(changed, before_correct == after_correct).sum()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(str(args.pairs))
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    requested_splits = list(args.split or ["train", "val"])
    if "train" not in requested_splits:
        raise ValueError("Pairwise verifier requires train split for fitting.")
    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    start_time = time.perf_counter()
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

    train = split_payloads["train"]
    train_features = build_verifier_features(
        np.asarray(train["embeddings"], dtype=np.float32),
        np.asarray(train["probabilities"], dtype=np.float32),
    )
    feature_names = build_verifier_feature_names(
        embedding_dim=int(np.asarray(train["embeddings"]).shape[1]),
        class_count=int(np.asarray(train["probabilities"]).shape[1]),
    )
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    models, pair_summaries, oof_probabilities = _fit_pair_models(
        train_features,
        train_labels,
        pairs,
        model_type=str(args.verifier_model),
        c_value=float(args.logistic_c),
        max_iter=int(args.logistic_max_iter),
        seed=int(args.seed),
        oof_folds=int(args.oof_folds),
        extra_trees_n_estimators=int(args.extra_trees_n_estimators),
        extra_trees_min_samples_leaf=int(args.extra_trees_min_samples_leaf),
        extra_trees_max_depth=int(args.extra_trees_max_depth),
    )
    model_export_summary = write_pair_model_export(
        output_dir / "pair_verifier_model_params.json",
        models,
        feature_dim=int(train_features.shape[1]),
        feature_names=feature_names,
        metadata={
            "tool": "probe_pairwise_feature_verifier",
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "pairs": [f"{int(a)}-{int(b)}" for a, b in pairs],
            "verifier_model": str(args.verifier_model),
            "logistic_c": float(args.logistic_c),
            "oof_folds": int(args.oof_folds),
        },
    )

    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "pairs": [f"{a}-{b}" for a, b in pairs],
        "settings": {
            "min_pair_probability": float(args.min_pair_probability),
            "max_pair_margin": float(args.max_pair_margin),
            "verifier_confidence_threshold": float(args.verifier_confidence_threshold),
            "verifier_model": str(args.verifier_model),
            "logistic_c": float(args.logistic_c),
            "extra_trees_n_estimators": int(args.extra_trees_n_estimators),
            "extra_trees_min_samples_leaf": int(args.extra_trees_min_samples_leaf),
            "extra_trees_max_depth": int(args.extra_trees_max_depth),
            "oof_folds": int(args.oof_folds),
        },
        "pair_models": pair_summaries,
        "pair_model_export": model_export_summary,
        "splits": {},
        "leakage_guard": "pair verifiers fit on train labels only; val is development diagnostic; test is not read by default",
    }

    for split, payload in split_payloads.items():
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        base_predictions = np.asarray(payload["base_predictions"], dtype=np.int64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        features = build_verifier_features(
            np.asarray(payload["embeddings"], dtype=np.float32),
            probabilities,
        )
        pair_probabilities = _pair_verifier_probabilities_for_split(
            split=str(split),
            features=features,
            labels=labels,
            models=models,
            oof_probabilities=oof_probabilities,
        )
        final_predictions, changes = apply_pairwise_verifiers(
            probabilities,
            base_predictions,
            models,
            min_pair_probability=float(args.min_pair_probability),
            max_pair_margin=float(args.max_pair_margin),
            verifier_confidence_threshold=float(args.verifier_confidence_threshold),
            features=features,
        )
        split_dir = output_dir / str(split)
        _write_predictions(
            split_dir / "predictions.csv",
            split=str(split),
            targets=labels,
            base_predictions=base_predictions,
            final_predictions=final_predictions,
            probabilities=probabilities,
            paths=list(payload.get("paths", [])),
            changes=changes,
            pair_probabilities=pair_probabilities,
        )
        metrics = {
            "base": _classification_metrics(labels, base_predictions, class_names),
            "verified": _classification_metrics(labels, final_predictions, class_names),
            "changes": _change_summary(labels, base_predictions, final_predictions),
        }
        (split_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        summary["splits"][str(split)] = {
            "samples": int(labels.shape[0]),
            "metrics": metrics,
        }

    summary["seconds"] = float(time.perf_counter() - start_time)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
