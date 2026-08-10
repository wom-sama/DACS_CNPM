from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _collate_classification,
    _resolve_device,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only diagnostic for part-discovery evidence in frozen TRKH patch "
            "tokens. It clusters train foreground/bbox patch tokens into part "
            "prototypes, fits simple train-only logistic readouts, and evaluates "
            "validation without reading test by default."
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
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-prototypes", type=int, default=32)
    parser.add_argument("--max-fit-patches", type=int, default=120000)
    parser.add_argument("--max-patches-per-sample", type=int, default=14)
    parser.add_argument("--bbox-threshold", type=float, default=0.05)
    parser.add_argument(
        "--bbox-token-prior-source",
        type=str,
        default="crop_bbox",
        choices=("bbox", "crop_bbox"),
    )
    parser.add_argument("--logistic-c", type=float, nargs="*", default=[0.05, 0.1, 0.3, 1.0])
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--focus-class-index", type=int, default=1)
    parser.add_argument("--focus-class-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        default=False,
        help="Allow explicit test split. Keep disabled for development diagnostics.",
    )
    return parser.parse_args(argv)


def _tensor_metadata(
    metadata: Mapping[str, object],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    value = metadata.get(key)
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def _valid_patch_mask(patches: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
    batch_size, token_count = patches.shape[:2]
    if torch.is_tensor(key_padding_mask) and tuple(key_padding_mask.shape[:2]) == (
        batch_size,
        token_count,
    ):
        valid = ~key_padding_mask.to(device=patches.device, dtype=torch.bool)
    else:
        valid = torch.ones((batch_size, token_count), device=patches.device, dtype=torch.bool)
    empty = ~valid.any(dim=1)
    if bool(empty.any().item()):
        valid[empty] = True
    return valid


def _patch_weights_from_features(features: Mapping[str, Tensor], patches: Tensor) -> Tensor:
    prior = features.get("patch_bbox_prior")
    batch_size, token_count = patches.shape[:2]
    if torch.is_tensor(prior) and tuple(prior.shape[:2]) == (batch_size, token_count):
        weights = prior.to(device=patches.device, dtype=torch.float32).clamp(0.0, 1.0)
    else:
        weights = torch.ones((batch_size, token_count), device=patches.device, dtype=torch.float32)
    return weights


def _select_patch_tokens(
    patches: Tensor,
    *,
    valid_mask: Tensor,
    patch_weights: Tensor,
    max_patches_per_sample: int,
    bbox_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected patch rows, owner sample offsets, and weights for one batch."""

    if patches.ndim != 3:
        raise ValueError("patches must be [B,N,D].")
    max_per_sample = max(1, int(max_patches_per_sample))
    threshold = float(max(0.0, bbox_threshold))
    patch_rows: List[np.ndarray] = []
    owner_rows: List[np.ndarray] = []
    weight_rows: List[np.ndarray] = []
    source = patches.detach().float().cpu().numpy()
    valid = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    weights = patch_weights.detach().float().cpu().numpy().astype(np.float32, copy=False)
    for batch_index in range(source.shape[0]):
        sample_valid = valid[batch_index]
        sample_weights = weights[batch_index]
        selected = np.flatnonzero(sample_valid & (sample_weights >= threshold))
        if selected.size == 0:
            selected = np.flatnonzero(sample_valid)
        if selected.size == 0:
            selected = np.arange(source.shape[1], dtype=np.int64)
        if selected.size > max_per_sample:
            order = np.argsort(-sample_weights[selected], kind="mergesort")[:max_per_sample]
            selected = selected[order]
        selected_weights = np.clip(sample_weights[selected], 1e-4, None).astype(np.float32, copy=False)
        patch_rows.append(source[batch_index, selected].astype(np.float32, copy=False))
        owner_rows.append(np.full((selected.size,), int(batch_index), dtype=np.int64))
        weight_rows.append(selected_weights)
    if not patch_rows:
        return (
            np.zeros((0, patches.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.concatenate(patch_rows, axis=0).astype(np.float32, copy=False),
        np.concatenate(owner_rows, axis=0).astype(np.int64, copy=False),
        np.concatenate(weight_rows, axis=0).astype(np.float32, copy=False),
    )


def build_part_histogram_features(
    *,
    patch_tokens: np.ndarray,
    patch_sample_indices: np.ndarray,
    patch_weights: np.ndarray,
    centroids: np.ndarray,
    sample_count: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Build per-image part prototype histograms and confidence summaries."""

    sample_count = int(sample_count)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    tokens = _normalize_rows(np.asarray(patch_tokens, dtype=np.float32))
    centers = _normalize_rows(np.asarray(centroids, dtype=np.float32))
    owners = np.asarray(patch_sample_indices, dtype=np.int64).reshape(-1)
    weights = np.asarray(patch_weights, dtype=np.float32).reshape(-1)
    if tokens.shape[0] != owners.shape[0] or tokens.shape[0] != weights.shape[0]:
        raise ValueError("patch_tokens, patch_sample_indices, and patch_weights must align.")
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise ValueError("centroids must be [K,D] with K > 0.")
    if tokens.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Token dim {tokens.shape[1]} does not match centroid dim {centers.shape[1]}."
        )
    prototype_count = int(centers.shape[0])
    histogram = np.zeros((sample_count, prototype_count), dtype=np.float32)
    max_similarity = np.full((sample_count, prototype_count), -np.inf, dtype=np.float32)
    total_weight = np.zeros((sample_count,), dtype=np.float32)
    max_assigned_similarity = np.zeros((sample_count,), dtype=np.float32)
    selected_count = np.zeros((sample_count,), dtype=np.float32)
    if tokens.shape[0] == 0 or sample_count == 0:
        return np.zeros((sample_count, prototype_count * 2 + 4), dtype=np.float32)
    for start in range(0, tokens.shape[0], max(1, int(chunk_size))):
        end = min(tokens.shape[0], start + max(1, int(chunk_size)))
        chunk = tokens[start:end]
        chunk_owners = owners[start:end]
        chunk_weights = np.clip(weights[start:end], 1e-6, None)
        valid_owner = (chunk_owners >= 0) & (chunk_owners < sample_count)
        if not bool(valid_owner.any()):
            continue
        chunk = chunk[valid_owner]
        chunk_owners = chunk_owners[valid_owner]
        chunk_weights = chunk_weights[valid_owner]
        similarities = chunk.dot(centers.T)
        assignments = similarities.argmax(axis=1).astype(np.int64, copy=False)
        assigned_similarity = similarities[np.arange(similarities.shape[0]), assignments]
        np.add.at(histogram, (chunk_owners, assignments), chunk_weights)
        np.maximum.at(max_similarity, (chunk_owners, assignments), assigned_similarity)
        np.add.at(total_weight, chunk_owners, chunk_weights)
        np.add.at(max_assigned_similarity, chunk_owners, assigned_similarity * chunk_weights)
        np.add.at(selected_count, chunk_owners, 1.0)
    normalized_histogram = histogram / np.maximum(total_weight[:, None], 1e-6)
    max_similarity = np.where(np.isfinite(max_similarity), max_similarity, 0.0).astype(
        np.float32,
        copy=False,
    )
    entropy = -np.sum(
        normalized_histogram * np.log(np.clip(normalized_histogram, 1e-8, 1.0)),
        axis=1,
        keepdims=True,
    )
    active_fraction = (normalized_histogram > 1e-6).mean(axis=1, keepdims=True)
    peak_mass = normalized_histogram.max(axis=1, keepdims=True) if prototype_count else np.zeros((sample_count, 1), dtype=np.float32)
    mean_assigned_similarity = (
        max_assigned_similarity / np.maximum(total_weight, 1e-6)
    ).reshape(-1, 1)
    return np.concatenate(
        [
            normalized_histogram,
            max_similarity,
            entropy.astype(np.float32, copy=False),
            active_fraction.astype(np.float32, copy=False),
            peak_mass.astype(np.float32, copy=False),
            mean_assigned_similarity.astype(np.float32, copy=False),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _extract_split_patch_payload(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
    bbox_token_prior_source: str,
    max_patches_per_sample: int,
    bbox_threshold: float,
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
    patch_token_batches: List[np.ndarray] = []
    patch_owner_batches: List[np.ndarray] = []
    patch_weight_batches: List[np.ndarray] = []
    paths: List[str] = []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    seen_samples = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"part-prototypes-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                metadata = {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batch_size_value = int(images.shape[0])
            batch_paths: List[str] = []
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
            batch_offset = int(seen_samples)
            seen_samples += batch_size_value

            bbox = _tensor_metadata(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _tensor_metadata(metadata, "crop_bbox", device=device, dtype=torch.float32)
            image_mask = _tensor_metadata(metadata, "image_mask", device=device, dtype=torch.bool)
            bbox_prior = (
                crop_bbox
                if str(bbox_token_prior_source) == "crop_bbox" and torch.is_tensor(crop_bbox)
                else bbox
            )

            with autocast_context(device, amp):
                if not hasattr(model, "forward_features"):
                    raise TypeError("Part prototype probe requires a TRKH model with forward_features().")
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=bbox_prior,
                )
                if torch.is_tensor(bbox):
                    features["bbox"] = bbox
                elif torch.is_tensor(bbox_prior):
                    features["bbox"] = bbox_prior
                if hasattr(model, "forward_heads"):
                    model_output = model.forward_heads(features)
                else:
                    model_output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(model_output)
                probabilities = logits.float().softmax(dim=1)
                embeddings = extract_head_input_from_features(model, features)
                patches = features.get("patches")
                if not torch.is_tensor(patches) or patches.ndim != 3:
                    raise ValueError("Model features do not contain patch tokens [B,N,D].")
                valid_mask = _valid_patch_mask(patches, features.get("memory_key_padding_mask"))
                patch_weights = _patch_weights_from_features(features, patches) * valid_mask.to(
                    dtype=torch.float32
                )

            patch_tokens, patch_owners, selected_weights = _select_patch_tokens(
                patches,
                valid_mask=valid_mask,
                patch_weights=patch_weights,
                max_patches_per_sample=int(max_patches_per_sample),
                bbox_threshold=float(bbox_threshold),
            )
            patch_owner_batches.append((patch_owners + batch_offset).astype(np.int64, copy=False))
            patch_token_batches.append(patch_tokens)
            patch_weight_batches.append(selected_weights)
            embedding_batches.append(embeddings.detach().float().cpu().numpy())
            probability_batches.append(probabilities.detach().float().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            prediction_batches.append(probabilities.argmax(dim=1).detach().cpu().numpy())
            paths.extend(batch_paths)

    labels_np = np.concatenate(label_batches, axis=0) if label_batches else np.zeros((0,), dtype=np.int64)
    return {
        "embeddings": (
            np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
            if embedding_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "probabilities": (
            np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)
            if probability_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "labels": labels_np.astype(np.int64, copy=False),
        "base_predictions": (
            np.concatenate(prediction_batches, axis=0).astype(np.int64, copy=False)
            if prediction_batches
            else np.zeros((0,), dtype=np.int64)
        ),
        "patch_tokens": (
            np.concatenate(patch_token_batches, axis=0).astype(np.float32, copy=False)
            if patch_token_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "patch_sample_indices": (
            np.concatenate(patch_owner_batches, axis=0).astype(np.int64, copy=False)
            if patch_owner_batches
            else np.zeros((0,), dtype=np.int64)
        ),
        "patch_weights": (
            np.concatenate(patch_weight_batches, axis=0).astype(np.float32, copy=False)
            if patch_weight_batches
            else np.zeros((0,), dtype=np.float32)
        ),
        "paths": paths,
    }


def _make_logistic_model(c_value: float, max_iter: int, seed: int):
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


def _focus_f1(metrics: Mapping[str, object], focus_class_index: int) -> float:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, Sequence):
        return 0.0
    index = int(focus_class_index)
    if index < 0 or index >= len(per_class):
        return 0.0
    row = per_class[index]
    if not isinstance(row, Mapping):
        return 0.0
    return float(row.get("f1", 0.0))


def _fit_variant_sweep(
    *,
    variant_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    class_names: Sequence[str],
    c_values: Sequence[float],
    max_iter: int,
    oof_folds: int,
    focus_class_index: int,
    focus_class_weight: float,
    seed: int,
) -> Dict[str, object]:
    labels = np.asarray(train_labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=len(class_names))
    non_empty_counts = class_counts[class_counts > 0]
    min_class_count = int(non_empty_counts.min()) if non_empty_counts.size else 0
    folds = min(int(oof_folds), min_class_count) if min_class_count >= 2 else 0
    rows: List[Dict[str, object]] = []
    best_row: Optional[Dict[str, object]] = None
    best_score = -float("inf")
    for c_value in c_values:
        model = _make_logistic_model(float(c_value), int(max_iter), int(seed))
        oof_predictions = None
        oof_metrics = None
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
            oof_predictions = cross_val_predict(model, train_features, labels, cv=cv, n_jobs=1)
            oof_metrics = _classification_metrics(labels, oof_predictions.astype(np.int64), class_names)
        model.fit(train_features, labels)
        val_predictions = model.predict(val_features).astype(np.int64)
        val_metrics = _classification_metrics(val_labels, val_predictions, class_names)
        if oof_metrics is not None:
            selection_score = float(oof_metrics["macro_f1"]) + float(focus_class_weight) * _focus_f1(
                oof_metrics,
                focus_class_index,
            )
        else:
            selection_score = -float("inf")
        row = {
            "variant": str(variant_name),
            "c": float(c_value),
            "selection_score": float(selection_score),
            "train_oof_predictions": oof_predictions.astype(np.int64) if oof_predictions is not None else None,
            "train_oof_metrics": oof_metrics,
            "val_predictions": val_predictions,
            "val_metrics": val_metrics,
        }
        rows.append(row)
        if best_row is None or selection_score > best_score:
            best_score = selection_score
            best_row = row
    if best_row is None:
        raise RuntimeError(f"No logistic candidate completed for {variant_name}.")
    return {
        "variant": str(variant_name),
        "folds": int(folds),
        "candidates": rows,
        "selected": best_row,
    }


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


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    probabilities: np.ndarray,
    paths: Sequence[str],
) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "changed",
    ]
    fieldnames.extend([f"prob_{index}" for index in range(probabilities.shape[1])])
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
                "base_prediction": int(base_predictions[index]),
                "final_prediction": int(final_predictions[index]),
                "changed": int(int(base_predictions[index]) != int(final_predictions[index])),
            }
            for class_index in range(probabilities.shape[1]):
                row[f"prob_{class_index}"] = float(probabilities[index, class_index])
            writer.writerow(row)


def _write_changed_cases(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    paths: Sequence[str],
) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "change_type",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        changed = np.flatnonzero(base_predictions.astype(np.int64) != final_predictions.astype(np.int64))
        for index in changed:
            before_correct = int(base_predictions[index]) == int(targets[index])
            after_correct = int(final_predictions[index]) == int(targets[index])
            if (not before_correct) and after_correct:
                change_type = "correction"
            elif before_correct and (not after_correct):
                change_type = "harm"
            else:
                change_type = "neutral"
            writer.writerow(
                {
                    "split": split,
                    "sample_index": int(index),
                    "image_path": str(paths[index]) if index < len(paths) else "",
                    "target_index": int(targets[index]),
                    "base_prediction": int(base_predictions[index]),
                    "final_prediction": int(final_predictions[index]),
                    "change_type": change_type,
                }
            )


def _serializable_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(metrics))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    requested_splits = list(args.split or ["train", "val"])
    if "train" not in requested_splits:
        raise ValueError("Part-prototype diagnostic requires the train split for fitting.")
    if any(str(split).lower() == "test" for split in requested_splits) and not bool(args.allow_test):
        raise ValueError("Refusing to read test split without --allow-test.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    for split in requested_splits:
        max_samples = int(args.max_train_samples) if str(split) == "train" else int(args.max_eval_samples)
        dataset, class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=str(split),
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples,
        )
        split_payloads[str(split)] = _extract_split_patch_payload(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=str(split),
            bbox_token_prior_source=str(args.bbox_token_prior_source),
            max_patches_per_sample=int(args.max_patches_per_sample),
            bbox_threshold=float(args.bbox_threshold),
        )

    train = split_payloads["train"]
    train_patch_tokens = np.asarray(train["patch_tokens"], dtype=np.float32)
    if train_patch_tokens.shape[0] == 0:
        raise RuntimeError("No train patch tokens were extracted.")
    rng = np.random.default_rng(int(args.seed))
    fit_limit = max(1, int(args.max_fit_patches))
    if train_patch_tokens.shape[0] > fit_limit:
        fit_indices = rng.choice(train_patch_tokens.shape[0], size=fit_limit, replace=False)
        fit_tokens = train_patch_tokens[fit_indices]
    else:
        fit_tokens = train_patch_tokens
    prototype_count = max(2, min(int(args.num_prototypes), int(fit_tokens.shape[0])))
    kmeans = MiniBatchKMeans(
        n_clusters=prototype_count,
        random_state=int(args.seed),
        batch_size=min(4096, max(256, fit_tokens.shape[0])),
        n_init=3,
        max_iter=100,
    )
    kmeans.fit(_normalize_rows(fit_tokens))
    centroids = kmeans.cluster_centers_.astype(np.float32, copy=False)

    for split, payload in split_payloads.items():
        payload["part_features"] = build_part_histogram_features(
            patch_tokens=np.asarray(payload["patch_tokens"], dtype=np.float32),
            patch_sample_indices=np.asarray(payload["patch_sample_indices"], dtype=np.int64),
            patch_weights=np.asarray(payload["patch_weights"], dtype=np.float32),
            centroids=centroids,
            sample_count=int(np.asarray(payload["labels"]).shape[0]),
        )

    val_key = "val" if "val" in split_payloads else requested_splits[-1]
    val = split_payloads[str(val_key)]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    val_base_predictions = np.asarray(val["base_predictions"], dtype=np.int64)
    base_metrics = _classification_metrics(val_labels, val_base_predictions, class_names)

    train_probs = np.asarray(train["probabilities"], dtype=np.float32)
    val_probs = np.asarray(val["probabilities"], dtype=np.float32)
    feature_variants = {
        "part_hist_only": (
            np.asarray(train["part_features"], dtype=np.float32),
            np.asarray(val["part_features"], dtype=np.float32),
        ),
        "head_prob_part_hist": (
            np.concatenate(
                [
                    np.asarray(train["embeddings"], dtype=np.float32),
                    train_probs,
                    np.asarray(train["part_features"], dtype=np.float32),
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    np.asarray(val["embeddings"], dtype=np.float32),
                    val_probs,
                    np.asarray(val["part_features"], dtype=np.float32),
                ],
                axis=1,
            ),
        ),
    }

    sweeps: Dict[str, Dict[str, object]] = {}
    metric_rows: List[Dict[str, object]] = []
    selected_summaries: Dict[str, Dict[str, object]] = {}
    for variant_name, (train_features, val_features) in feature_variants.items():
        sweep = _fit_variant_sweep(
            variant_name=variant_name,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            class_names=class_names,
            c_values=list(args.logistic_c),
            max_iter=int(args.logistic_max_iter),
            oof_folds=int(args.oof_folds),
            focus_class_index=int(args.focus_class_index),
            focus_class_weight=float(args.focus_class_weight),
            seed=int(args.seed),
        )
        sweeps[variant_name] = sweep
        for row in sweep["candidates"]:
            train_oof_metrics = row["train_oof_metrics"] or {}
            val_metrics = row["val_metrics"]
            metric_rows.append(
                {
                    "variant": variant_name,
                    "c": float(row["c"]),
                    "selected": int(row is sweep["selected"]),
                    "selection_score": float(row["selection_score"]),
                    "train_oof_macro_f1": float(train_oof_metrics.get("macro_f1", 0.0)),
                    "train_oof_focus_f1": _focus_f1(train_oof_metrics, int(args.focus_class_index)),
                    "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                    "val_focus_f1": _focus_f1(val_metrics, int(args.focus_class_index)),
                    "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
                }
            )
        selected = sweep["selected"]
        selected_predictions = np.asarray(selected["val_predictions"], dtype=np.int64)
        change = _change_summary(val_labels, val_base_predictions, selected_predictions)
        selected_summaries[variant_name] = {
            "c": float(selected["c"]),
            "selection_score": float(selected["selection_score"]),
            "train_oof_metrics": _serializable_metrics(selected["train_oof_metrics"] or {}),
            "val_metrics": _serializable_metrics(selected["val_metrics"]),
            "val_change_summary": change,
        }
        _write_predictions(
            output_dir / f"predictions_{variant_name}.csv",
            split=str(val_key),
            targets=val_labels,
            base_predictions=val_base_predictions,
            final_predictions=selected_predictions,
            probabilities=val_probs,
            paths=list(val.get("paths", [])),
        )
        _write_changed_cases(
            output_dir / f"changed_{variant_name}.csv",
            split=str(val_key),
            targets=val_labels,
            base_predictions=val_base_predictions,
            final_predictions=selected_predictions,
            paths=list(val.get("paths", [])),
        )

    with (output_dir / "variant_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "variant",
            "c",
            "selected",
            "selection_score",
            "train_oof_macro_f1",
            "train_oof_focus_f1",
            "val_macro_f1",
            "val_focus_f1",
            "val_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "method": "train-only patch-token part prototype diagnostic",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "splits": requested_splits,
        "class_names": list(class_names),
        "num_prototypes": int(prototype_count),
        "fit_patch_rows": int(fit_tokens.shape[0]),
        "train_samples": int(train_labels.shape[0]),
        "val_split": str(val_key),
        "val_samples": int(val_labels.shape[0]),
        "base_val_metrics": _serializable_metrics(base_metrics),
        "selected_variants": selected_summaries,
        "part_feature_dim": int(np.asarray(train["part_features"]).shape[1]),
        "patch_rows": {
            split: int(np.asarray(payload["patch_tokens"]).shape[0])
            for split, payload in split_payloads.items()
        },
        "bbox_threshold": float(args.bbox_threshold),
        "max_patches_per_sample": int(args.max_patches_per_sample),
        "bbox_token_prior_source": str(args.bbox_token_prior_source),
        "research_sources": [
            "https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Saha_PARTICLE_Part_Discovery_and_Contrastive_Learning_for_Fine-Grained_Recognition_ICCVW_2023_paper.pdf",
            "https://arxiv.org/abs/2303.02404",
        ],
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
