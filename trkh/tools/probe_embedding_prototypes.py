from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    build_eval_transform,
)
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.core.utils import autocast_context


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether frozen TRKH embeddings are separable with prototype/kNN/"
            "logistic decision layers. Uses train split for fitting and validation "
            "split for development; never reads test unless explicitly requested."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--knn-k", type=int, nargs="*", default=[1, 3, 5, 9])
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--skip-logistic", action="store_true", default=False)
    parser.add_argument("--etf-alpha", type=float, nargs="*", default=[1.0, 10.0])
    parser.add_argument("--skip-etf-ridge", action="store_true", default=False)
    parser.add_argument("--t3a-support-per-class", type=int, nargs="*", default=[])
    parser.add_argument("--t3a-source-weight", type=float, nargs="*", default=[8.0])
    parser.add_argument("--t3a-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--soft-centroid-temperature",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional train-probability weighted centroid temperatures. Fits "
            "centroids from train embeddings and train model probabilities only."
        ),
    )
    parser.add_argument(
        "--soft-centroid-min-confidence",
        type=float,
        nargs="*",
        default=[0.0],
        help="Optional train row confidence filters for soft-centroid fitting.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--save-embedding-cache",
        action="store_true",
        default=False,
        help=(
            "Save per-split embeddings/probabilities/labels beside the diagnostic "
            "summary so later train-to-val audits can reuse the same forward pass."
        ),
    )
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_config_value(
    checkpoint: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, Mapping):
        return model_config.get(key, default)
    return default


def _resolve_augmentation_config_value(
    checkpoint: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    augmentation_config = checkpoint.get("augmentation_config", {})
    if isinstance(augmentation_config, Mapping):
        return augmentation_config.get(key, default)
    return default


def _build_dataset(
    *,
    data_yaml: Path,
    split: str,
    checkpoint: Mapping[str, object],
    class_name_mode: str,
    max_samples: int,
) -> Tuple[Dataset, List[str]]:
    data_spec = load_data_spec(data_yaml, class_name_mode=class_name_mode)
    image_size = int(_resolve_model_config_value(checkpoint, "image_size", 256))
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=str(_resolve_augmentation_config_value(checkpoint, "resize_mode", "pad")),
        illumination_normalization=bool(
            _resolve_augmentation_config_value(
                checkpoint,
                "illumination_normalization",
                False,
            )
        ),
        illumination_normalization_strength=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "illumination_normalization_strength",
                0.0,
            )
        ),
        foreground_crop_mode=str(
            _resolve_augmentation_config_value(checkpoint, "foreground_crop_mode", "none")
        ),
        foreground_crop_margin_ratio=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "foreground_crop_margin_ratio",
                0.08,
            )
        ),
        foreground_crop_min_mask_area_ratio=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "foreground_crop_min_mask_area_ratio",
                0.03,
            )
        ),
        foreground_crop_max_mask_area_ratio=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "foreground_crop_max_mask_area_ratio",
                0.92,
            )
        ),
        foreground_crop_max_crop_area_ratio=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "foreground_crop_max_crop_area_ratio",
                0.98,
            )
        ),
        background_suppression_mode=str(
            _resolve_augmentation_config_value(
                checkpoint,
                "background_suppression_mode",
                "none",
            )
        ),
        background_suppression_margin=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "background_suppression_margin",
                0.08,
            )
        ),
        background_suppression_blur_radius=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "background_suppression_blur_radius",
                7.0,
            )
        ),
        surface_detail_amplification_mode=str(
            _resolve_augmentation_config_value(
                checkpoint,
                "surface_detail_amplification_mode",
                "none",
            )
        ),
        surface_detail_amplification_strength=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "surface_detail_amplification_strength",
                0.0,
            )
        ),
        surface_detail_amplification_blur_radius=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "surface_detail_amplification_blur_radius",
                1.25,
            )
        ),
        surface_detail_amplification_foreground_weight=float(
            _resolve_augmentation_config_value(
                checkpoint,
                "surface_detail_amplification_foreground_weight",
                0.85,
            )
        ),
        eval_surface_detail_amplification=bool(
            _resolve_augmentation_config_value(
                checkpoint,
                "eval_surface_detail_amplification",
                False,
            )
        ),
    )
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        dataset = ClassificationFolderDataset.from_data_spec(
            data_spec,
            split=split,
            transform=transform,
        )
    else:
        dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=transform,
            crop_margin_ratio=float(
                _resolve_augmentation_config_value(checkpoint, "crop_margin_ratio", 0.05)
            ),
            crop_to_primary_object=True,
            classification_target=True,
            classification_object_crops=True,
            classification_bbox_metadata=True,
        )
    if max_samples > 0:
        dataset.samples = dataset.samples[: int(max_samples)]
    return dataset, list(data_spec.class_names)


def _collate_classification(batch):
    images: List[Tensor] = []
    labels: List[int] = []
    paths: List[str] = []
    tensor_metadata: Dict[str, List[Tensor]] = {}
    for item in batch:
        if len(item) == 3:
            image, label, metadata = item
        elif len(item) == 2:
            image, label = item
            metadata = {}
        else:
            raise ValueError("Classification batch item must have 2 or 3 fields.")
        images.append(image)
        labels.append(int(label))
        path = ""
        if isinstance(metadata, Mapping):
            path = str(metadata.get("image_path", "") or metadata.get("path", "") or "")
            for key, value in metadata.items():
                if torch.is_tensor(value):
                    tensor_metadata.setdefault(str(key), []).append(value)
        paths.append(path)
    collated_metadata: Dict[str, object] = {"paths": paths}
    for key, values in tensor_metadata.items():
        if len(values) == len(images):
            collated_metadata[key] = torch.stack(values, dim=0)
    return (
        torch.stack(images, dim=0),
        torch.as_tensor(labels, dtype=torch.long),
        collated_metadata,
    )


def _extract_split_embeddings(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    embedding_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    paths: List[str] = []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in sample_paths_fn()]
        if callable(sample_paths_fn)
        else []
    )
    seen_samples = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"extract-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size_value = int(images.shape[0])
            batch_paths = []
            if isinstance(metadata, Mapping):
                raw_paths = metadata.get("paths", [])
                if isinstance(raw_paths, Sequence):
                    batch_paths = [str(path) for path in raw_paths]
            if (
                dataset_paths
                and (
                    len(batch_paths) != batch_size_value
                    or not any(str(path).strip() for path in batch_paths)
                )
            ):
                batch_paths = dataset_paths[seen_samples : seen_samples + batch_size_value]
            seen_samples += batch_size_value

            bbox_metadata = None
            image_valid_mask = None
            if isinstance(metadata, Mapping):
                bbox_value = metadata.get("bbox")
                if torch.is_tensor(bbox_value):
                    bbox_metadata = bbox_value.to(device=device, dtype=torch.float32, non_blocking=True)
                mask_value = metadata.get("image_mask")
                if torch.is_tensor(mask_value):
                    image_valid_mask = mask_value.to(device=device, dtype=torch.bool, non_blocking=True)
            with autocast_context(device, amp):
                if hasattr(model, "forward_features"):
                    features = model.forward_features(
                        images,
                        image_valid_mask=image_valid_mask,
                        bbox_token_prior=bbox_metadata,
                    )
                    if bbox_metadata is not None:
                        features["bbox"] = bbox_metadata
                    if hasattr(model, "forward_heads"):
                        model_output = model.forward_heads(features)
                    else:
                        model_output = classification_logits_from_features(model, features)
                    logits, _, _ = extract_detection_from_model_output(model_output)
                    embeddings = extract_head_input_from_features(model, features)
                else:
                    logits = model(images)
                    embeddings = logits
                probabilities = logits.float().softmax(dim=1)
            embedding_batches.append(embeddings.detach().float().cpu().numpy())
            probability_batches.append(probabilities.detach().float().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            prediction_batches.append(probabilities.argmax(dim=1).detach().cpu().numpy())
            paths.extend(batch_paths)
    embeddings_np = np.concatenate(embedding_batches, axis=0) if embedding_batches else np.zeros((0, 0), dtype=np.float32)
    probabilities_np = np.concatenate(probability_batches, axis=0) if probability_batches else np.zeros((0, 0), dtype=np.float32)
    labels_np = np.concatenate(label_batches, axis=0) if label_batches else np.zeros((0,), dtype=np.int64)
    predictions_np = np.concatenate(prediction_batches, axis=0) if prediction_batches else np.zeros((0,), dtype=np.int64)
    return {
        "embeddings": embeddings_np.astype(np.float32, copy=False),
        "probabilities": probabilities_np.astype(np.float32, copy=False),
        "labels": labels_np.astype(np.int64, copy=False),
        "base_predictions": predictions_np.astype(np.int64, copy=False),
        "paths": paths,
    }


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets.astype(np.int64), predictions.astype(np.int64)):
        if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
            confusion[int(target), int(prediction)] += 1
    per_class = []
    f1_values = []
    for index, class_name in enumerate(class_names):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(float(f1))
        per_class.append(
            {
                "class_index": int(index),
                "class_name": str(class_name),
                "support": support,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    accuracy = float((targets.astype(np.int64) == predictions.astype(np.int64)).mean()) if targets.size else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def _class_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    *,
    normalize: bool,
) -> np.ndarray:
    source = _normalize_rows(embeddings) if normalize else embeddings
    centroids = np.zeros((class_count, source.shape[1]), dtype=np.float32)
    global_centroid = source.mean(axis=0) if source.size else np.zeros((source.shape[1],), dtype=np.float32)
    for class_index in range(class_count):
        mask = labels == class_index
        centroids[class_index] = source[mask].mean(axis=0) if bool(mask.any()) else global_centroid
    if normalize:
        centroids = _normalize_rows(centroids)
    return centroids.astype(np.float32, copy=False)


def _predict_nearest_centroid_cosine(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
) -> np.ndarray:
    centroids = _class_centroids(train_embeddings, train_labels, class_count, normalize=True)
    scores = _normalize_rows(eval_embeddings).dot(centroids.T)
    return scores.argmax(axis=1).astype(np.int64)


def _predict_nearest_centroid_euclidean(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
) -> np.ndarray:
    centroids = _class_centroids(train_embeddings, train_labels, class_count, normalize=False)
    distances = (
        np.sum(eval_embeddings ** 2, axis=1, keepdims=True)
        - 2.0 * eval_embeddings.dot(centroids.T)
        + np.sum(centroids ** 2, axis=1, keepdims=True).T
    )
    return distances.argmin(axis=1).astype(np.int64)


def _temperature_scale_probabilities(
    probabilities: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    temp = max(1e-6, float(temperature))
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits).astype(np.float32, copy=False)
    return scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)


def _soft_class_centroids(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
    *,
    temperature: float,
    min_confidence: float,
    normalize: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    source = _normalize_rows(embeddings) if normalize else np.asarray(embeddings, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if source.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("soft centroids require 2D embeddings and probabilities.")
    if source.shape[0] != probabilities.shape[0]:
        raise ValueError("soft-centroid embeddings/probabilities row count mismatch.")
    if probabilities.shape[1] < int(class_count):
        raise ValueError("soft-centroid probabilities have fewer columns than classes.")

    confidence = probabilities[:, :class_count].max(axis=1) if probabilities.size else np.zeros((0,), dtype=np.float32)
    keep_mask = confidence >= float(min_confidence)
    if not bool(keep_mask.any()) and source.shape[0] > 0:
        keep_mask = np.ones((source.shape[0],), dtype=bool)

    kept_features = source[keep_mask]
    kept_probabilities = probabilities[keep_mask, :class_count]
    weights = _temperature_scale_probabilities(
        kept_probabilities,
        temperature=float(temperature),
    )
    global_centroid = source.mean(axis=0) if source.size else np.zeros((source.shape[1],), dtype=np.float32)
    centroids = np.zeros((class_count, source.shape[1]), dtype=np.float32)
    class_weight_sums: List[float] = []
    for class_index in range(class_count):
        class_weights = weights[:, class_index].astype(np.float32, copy=False)
        weight_sum = float(class_weights.sum())
        class_weight_sums.append(weight_sum)
        if weight_sum > 1e-8:
            centroids[class_index] = (
                kept_features * class_weights[:, None]
            ).sum(axis=0) / weight_sum
        else:
            centroids[class_index] = global_centroid
    if normalize:
        centroids = _normalize_rows(centroids)
    stats = {
        "temperature": float(temperature),
        "min_confidence": float(min_confidence),
        "kept_rows": int(keep_mask.sum()),
        "total_rows": int(source.shape[0]),
        "mean_kept_confidence": (
            float(confidence[keep_mask].mean()) if bool(keep_mask.any()) else 0.0
        ),
        "class_weight_sums": [float(value) for value in class_weight_sums],
    }
    return centroids.astype(np.float32, copy=False), stats


def _predict_soft_centroid_cosine(
    train_embeddings: np.ndarray,
    train_probabilities: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
    *,
    temperature: float,
    min_confidence: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    centroids, stats = _soft_class_centroids(
        train_embeddings,
        train_probabilities,
        class_count,
        temperature=float(temperature),
        min_confidence=float(min_confidence),
        normalize=True,
    )
    scores = _normalize_rows(eval_embeddings).dot(centroids.T)
    return scores.argmax(axis=1).astype(np.int64), stats


def _predict_knn_cosine(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
    *,
    k: int,
    chunk_size: int = 512,
) -> np.ndarray:
    train_norm = _normalize_rows(train_embeddings)
    eval_norm = _normalize_rows(eval_embeddings)
    k = max(1, min(int(k), int(train_norm.shape[0])))
    predictions: List[np.ndarray] = []
    for start in range(0, eval_norm.shape[0], max(1, int(chunk_size))):
        scores = eval_norm[start : start + chunk_size].dot(train_norm.T)
        neighbor_indices = np.argpartition(scores, kth=train_norm.shape[0] - k, axis=1)[:, -k:]
        neighbor_scores = np.take_along_axis(scores, neighbor_indices, axis=1)
        votes = np.zeros((neighbor_indices.shape[0], class_count), dtype=np.float64)
        for row_index in range(neighbor_indices.shape[0]):
            for neighbor_index, score in zip(neighbor_indices[row_index], neighbor_scores[row_index]):
                votes[row_index, int(train_labels[int(neighbor_index)])] += max(0.0, float(score)) + 1e-6
        predictions.append(votes.argmax(axis=1).astype(np.int64))
    return np.concatenate(predictions, axis=0) if predictions else np.zeros((0,), dtype=np.int64)


def _fit_predict_logistic(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    *,
    class_weight: Optional[str],
    max_iter: int,
) -> Optional[np.ndarray]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max(100, int(max_iter)),
            class_weight=class_weight,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=1,
        ),
    )
    model.fit(train_embeddings, train_labels)
    return model.predict(eval_embeddings).astype(np.int64)


def _simplex_etf_targets(class_count: int) -> np.ndarray:
    count = max(2, int(class_count))
    eye = np.eye(count, dtype=np.float32)
    centered = eye - np.full((count, count), 1.0 / float(count), dtype=np.float32)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _fit_predict_etf_ridge(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    class_count: int,
    *,
    alpha: float,
    class_weight: Optional[str],
) -> Optional[np.ndarray]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if train_labels.size == 0:
        return None
    targets = _simplex_etf_targets(class_count)
    target_vectors = targets[train_labels]
    weights = None
    if class_weight == "balanced":
        counts = np.bincount(train_labels, minlength=class_count).astype(np.float32)
        positive = counts[counts > 0]
        if positive.size:
            base = float(np.mean(positive))
            weights = np.asarray(
                [base / max(1.0, float(counts[int(label)])) for label in train_labels],
                dtype=np.float32,
            )
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=max(1e-8, float(alpha)), random_state=0),
    )
    fit_params = {"ridge__sample_weight": weights} if weights is not None else {}
    model.fit(train_embeddings, target_vectors, **fit_params)
    projected = np.asarray(model.predict(eval_embeddings), dtype=np.float32)
    projected = _normalize_rows(projected)
    scores = projected.dot(targets.T)
    return scores.argmax(axis=1).astype(np.int64)


def _classifier_templates_from_model(
    model: torch.nn.Module,
    *,
    class_count: int,
    embedding_dim: int,
) -> Optional[np.ndarray]:
    for head_name in ("head", "classification_head"):
        head = getattr(model, head_name, None)
        weight = getattr(head, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            if int(weight.shape[0]) >= int(class_count) and int(weight.shape[1]) == int(embedding_dim):
                return _normalize_rows(
                    weight.detach().float().cpu().numpy()[: int(class_count)]
                ).astype(np.float32, copy=False)
    return None


def _predict_t3a_templates(
    eval_embeddings: np.ndarray,
    eval_probabilities: np.ndarray,
    classifier_templates: np.ndarray,
    class_count: int,
    *,
    support_per_class: int,
    source_weight: float,
    min_confidence: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    features = _normalize_rows(np.asarray(eval_embeddings, dtype=np.float32))
    probabilities = np.asarray(eval_probabilities, dtype=np.float32)
    if features.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("T3A requires 2D embeddings and probabilities.")
    if probabilities.shape[0] != features.shape[0]:
        raise ValueError("T3A embeddings/probabilities row count mismatch.")
    templates = _normalize_rows(np.asarray(classifier_templates, dtype=np.float32)[:class_count])
    pseudo_labels = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1)
    keep_per_class = max(1, int(support_per_class))
    source_mass = max(0.0, float(source_weight))
    min_conf = float(min_confidence)
    adjusted = np.zeros_like(templates)
    support_counts: List[int] = []
    mean_confidences: List[float] = []
    for class_index in range(class_count):
        candidate = np.where((pseudo_labels == class_index) & (confidence >= min_conf))[0]
        if candidate.size:
            order = candidate[np.argsort(entropy[candidate], kind="mergesort")]
            selected = order[: min(keep_per_class, int(order.size))]
            selected_features = features[selected]
            selected_sum = selected_features.sum(axis=0)
            denominator = source_mass + float(selected_features.shape[0])
            if denominator > 0.0:
                adjusted[class_index] = (
                    source_mass * templates[class_index] + selected_sum
                ) / denominator
            else:
                adjusted[class_index] = templates[class_index]
            support_counts.append(int(selected_features.shape[0]))
            mean_confidences.append(float(np.mean(confidence[selected])))
        else:
            adjusted[class_index] = templates[class_index]
            support_counts.append(0)
            mean_confidences.append(0.0)
    adjusted = _normalize_rows(adjusted)
    scores = features.dot(adjusted.T)
    stats = {
        "support_per_class": int(keep_per_class),
        "source_weight": float(source_mass),
        "min_confidence": float(min_conf),
        "selected_counts": support_counts,
        "selected_total": int(sum(support_counts)),
        "selected_mean_confidence": mean_confidences,
    }
    return scores.argmax(axis=1).astype(np.int64), stats


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    predictions_by_method: Mapping[str, np.ndarray],
    paths: Sequence[str],
) -> None:
    fieldnames = ["split", "sample_index", "image_path", "target_index"] + [
        f"pred_{name}" for name in predictions_by_method.keys()
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
            for method, predictions in predictions_by_method.items():
                row[f"pred_{method}"] = int(predictions[index])
            writer.writerow(row)


def _write_embedding_cache(path: Path, payload: Mapping[str, object]) -> None:
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    predictions = np.asarray(payload["base_predictions"], dtype=np.int64)
    paths = np.asarray([str(value) for value in payload.get("paths", [])], dtype=object)
    if not (
        embeddings.shape[0]
        == probabilities.shape[0]
        == labels.shape[0]
        == predictions.shape[0]
    ):
        raise ValueError("Embedding cache payload arrays have inconsistent row counts.")
    if paths.shape[0] not in (0, labels.shape[0]):
        raise ValueError("Embedding cache paths do not match the extracted row count.")
    if paths.shape[0] == 0:
        paths = np.asarray([""] * int(labels.shape[0]), dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        probabilities=probabilities,
        labels=labels,
        base_predictions=predictions,
        paths=paths,
        sample_index=np.arange(int(labels.shape[0]), dtype=np.int64),
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
        raise ValueError("Probe requires train split to fit train-only decision layers.")
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
    train_embeddings = np.asarray(train["embeddings"], dtype=np.float32)
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    train_probabilities = np.asarray(train["probabilities"], dtype=np.float32)
    class_count = len(class_names)
    classifier_templates = _classifier_templates_from_model(
        model,
        class_count=class_count,
        embedding_dim=int(train_embeddings.shape[1]) if train_embeddings.ndim == 2 else 0,
    )
    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "splits": {},
        "methods": [
            "base_head",
            "centroid_cosine",
            "centroid_euclidean",
            *[f"knn_cosine_k{k}" for k in args.knn_k],
        ],
        "leakage_guard": "fit prototype/kNN/logistic on train split only; val is used for development metrics",
    }
    if not bool(args.skip_logistic):
        summary["methods"] = list(summary["methods"]) + [
            "logistic",
            "logistic_balanced",
        ]
    if not bool(args.skip_etf_ridge):
        summary["methods"] = list(summary["methods"]) + [
            f"etf_ridge_a{float(alpha):g}" for alpha in args.etf_alpha
        ] + [
            f"etf_ridge_balanced_a{float(alpha):g}" for alpha in args.etf_alpha
        ]
    if classifier_templates is not None and args.t3a_support_per_class:
        summary["methods"] = list(summary["methods"]) + [
            f"t3a_k{int(k)}_sw{float(sw):g}_c{float(args.t3a_min_confidence):g}"
            for k in args.t3a_support_per_class
            for sw in args.t3a_source_weight
        ]
    if args.soft_centroid_temperature:
        summary["methods"] = list(summary["methods"]) + [
            f"soft_centroid_t{float(temp):g}_c{float(min_conf):g}"
            for temp in args.soft_centroid_temperature
            for min_conf in args.soft_centroid_min_confidence
        ]
    summary["classifier_templates_available"] = classifier_templates is not None
    soft_centroid_stats: Dict[str, object] = {}
    for split, payload in split_payloads.items():
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        predictions_by_method: Dict[str, np.ndarray] = {
            "base_head": np.asarray(payload["base_predictions"], dtype=np.int64),
            "centroid_cosine": _predict_nearest_centroid_cosine(
                train_embeddings,
                train_labels,
                embeddings,
                class_count,
            ),
            "centroid_euclidean": _predict_nearest_centroid_euclidean(
                train_embeddings,
                train_labels,
                embeddings,
                class_count,
            ),
        }
        for k_value in args.knn_k:
            method = f"knn_cosine_k{int(k_value)}"
            predictions_by_method[method] = _predict_knn_cosine(
                train_embeddings,
                train_labels,
                embeddings,
                class_count,
                k=int(k_value),
            )
        if not bool(args.skip_logistic):
            logistic_predictions = _fit_predict_logistic(
                train_embeddings,
                train_labels,
                embeddings,
                class_weight=None,
                max_iter=int(args.logistic_max_iter),
            )
            if logistic_predictions is not None:
                predictions_by_method["logistic"] = logistic_predictions
            balanced_predictions = _fit_predict_logistic(
                train_embeddings,
                train_labels,
                embeddings,
                class_weight="balanced",
                max_iter=int(args.logistic_max_iter),
            )
            if balanced_predictions is not None:
                predictions_by_method["logistic_balanced"] = balanced_predictions
        if not bool(args.skip_etf_ridge):
            for alpha_value in args.etf_alpha:
                method = f"etf_ridge_a{float(alpha_value):g}"
                etf_predictions = _fit_predict_etf_ridge(
                    train_embeddings,
                    train_labels,
                    embeddings,
                    class_count,
                    alpha=float(alpha_value),
                    class_weight=None,
                )
                if etf_predictions is not None:
                    predictions_by_method[method] = etf_predictions
                balanced_method = f"etf_ridge_balanced_a{float(alpha_value):g}"
                balanced_etf_predictions = _fit_predict_etf_ridge(
                    train_embeddings,
                    train_labels,
                    embeddings,
                    class_count,
                    alpha=float(alpha_value),
                    class_weight="balanced",
                )
                if balanced_etf_predictions is not None:
                    predictions_by_method[balanced_method] = balanced_etf_predictions
        t3a_stats: Dict[str, object] = {}
        if classifier_templates is not None and args.t3a_support_per_class:
            for support_per_class in args.t3a_support_per_class:
                for source_weight in args.t3a_source_weight:
                    method = (
                        f"t3a_k{int(support_per_class)}_sw{float(source_weight):g}_"
                        f"c{float(args.t3a_min_confidence):g}"
                    )
                    t3a_predictions, stats = _predict_t3a_templates(
                        embeddings,
                        probabilities,
                        classifier_templates,
                        class_count,
                        support_per_class=int(support_per_class),
                        source_weight=float(source_weight),
                        min_confidence=float(args.t3a_min_confidence),
                    )
                    predictions_by_method[method] = t3a_predictions
                    t3a_stats[method] = stats

        split_soft_centroid_stats: Dict[str, object] = {}
        if args.soft_centroid_temperature:
            for temperature in args.soft_centroid_temperature:
                for min_confidence in args.soft_centroid_min_confidence:
                    method = f"soft_centroid_t{float(temperature):g}_c{float(min_confidence):g}"
                    soft_predictions, stats = _predict_soft_centroid_cosine(
                        train_embeddings,
                        train_probabilities,
                        embeddings,
                        class_count,
                        temperature=float(temperature),
                        min_confidence=float(min_confidence),
                    )
                    predictions_by_method[method] = soft_predictions
                    split_soft_centroid_stats[method] = stats
            soft_centroid_stats[str(split)] = split_soft_centroid_stats

        split_summary = {
            "samples": int(labels.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "class_counts": [
                int((labels == class_index).sum()) for class_index in range(class_count)
            ],
            "t3a": t3a_stats,
            "soft_centroid": split_soft_centroid_stats,
            "metrics": {
                method: _classification_metrics(labels, predictions, class_names)
                for method, predictions in predictions_by_method.items()
            },
        }
        if bool(args.save_embedding_cache):
            cache_path = output_dir / f"{split}_embeddings.npz"
            _write_embedding_cache(cache_path, payload)
            split_summary["embedding_cache"] = str(cache_path.resolve())
        summary["splits"][split] = split_summary
        _write_predictions(
            output_dir / f"{split}_prototype_predictions.csv",
            split=split,
            targets=labels,
            predictions_by_method=predictions_by_method,
            paths=payload.get("paths", []),
        )
    summary["seconds"] = float(time.perf_counter() - start_time)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact = {
        "output_dir": str(output_dir),
        "seconds": round(float(summary["seconds"]), 2),
        "val": {
            method: {
                "macro_f1": metrics["macro_f1"],
                "class1_f1": metrics["per_class"][1]["f1"] if len(metrics["per_class"]) > 1 else None,
            }
            for method, metrics in summary["splits"].get("val", {}).get("metrics", {}).items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
