from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
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
    _fit_predict_logistic,
    _predict_knn_cosine,
    _predict_nearest_centroid_cosine,
    _resolve_device,
)
from trkh.tools.probe_patch_evidence_mil import _tensor_metadata


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe class separability in bbox foreground tokens versus background "
            "tokens. Fits train-only decision layers and reports validation metrics; "
            "refuses test by default because this is a method-selection diagnostic."
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
    parser.add_argument(
        "--bbox-token-prior-source",
        type=str,
        default="crop_bbox",
        choices=("bbox", "crop_bbox"),
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.20)
    parser.add_argument("--background-threshold", type=float, default=0.20)
    parser.add_argument("--knn-k", type=int, nargs="*", default=[5])
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--skip-logistic", action="store_true", default=False)
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _normalize_rows(values: Tensor, eps: float = 1e-6) -> Tensor:
    denominator = values.sum(dim=1, keepdim=True).clamp_min(float(eps))
    return values / denominator


def _region_weights(
    *,
    bbox_prior: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    foreground_threshold: float = 0.20,
    background_threshold: float = 0.20,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    if bbox_prior.ndim != 2:
        raise ValueError("bbox_prior must be [B, N].")
    prior = bbox_prior.float().clamp(0.0, 1.0)
    if torch.is_tensor(key_padding_mask):
        if tuple(key_padding_mask.shape) != tuple(prior.shape):
            raise ValueError("key_padding_mask shape must match bbox_prior.")
        valid = ~key_padding_mask.to(device=prior.device, dtype=torch.bool)
    else:
        valid = torch.ones_like(prior, dtype=torch.bool)
    valid_float = valid.to(dtype=torch.float32)
    fg_threshold = max(0.0, min(1.0, float(foreground_threshold)))
    bg_threshold = max(0.0, min(1.0, float(background_threshold)))
    foreground = (prior >= fg_threshold).to(dtype=torch.float32) * prior * valid_float
    background = (prior <= bg_threshold).to(dtype=torch.float32) * (1.0 - prior) * valid_float
    empty_foreground = foreground.sum(dim=1) <= 1e-6
    empty_background = background.sum(dim=1) <= 1e-6
    if bool(empty_foreground.any().item()):
        foreground[empty_foreground] = prior[empty_foreground] * valid_float[empty_foreground]
    if bool(empty_background.any().item()):
        background[empty_background] = (1.0 - prior[empty_background]) * valid_float[empty_background]
    still_empty_foreground = foreground.sum(dim=1) <= 1e-6
    still_empty_background = background.sum(dim=1) <= 1e-6
    if bool(still_empty_foreground.any().item()):
        foreground[still_empty_foreground] = valid_float[still_empty_foreground]
    if bool(still_empty_background.any().item()):
        background[still_empty_background] = valid_float[still_empty_background]
    stats = {
        "mean_valid_tokens": float(valid_float.sum(dim=1).mean().detach().cpu().item()),
        "mean_prior_mass": float(
            (prior * valid_float).sum(dim=1).div(valid_float.sum(dim=1).clamp_min(1.0)).mean().detach().cpu().item()
        ),
        "mean_foreground_selected": float((foreground > 0).sum(dim=1).float().mean().detach().cpu().item()),
        "mean_background_selected": float((background > 0).sum(dim=1).float().mean().detach().cpu().item()),
        "empty_foreground_fallbacks": int(empty_foreground.sum().detach().cpu().item()),
        "empty_background_fallbacks": int(empty_background.sum().detach().cpu().item()),
    }
    return _normalize_rows(foreground), _normalize_rows(background), valid, stats


def _weighted_token_mean(tokens: Tensor, weights: Tensor) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, N, D].")
    if tuple(tokens.shape[:2]) != tuple(weights.shape):
        raise ValueError("weights must match token batch and token dimensions.")
    return torch.bmm(weights.to(dtype=tokens.dtype).unsqueeze(1), tokens).squeeze(1)


def _extract_split_regions(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
    bbox_token_prior_source: str,
    foreground_threshold: float,
    background_threshold: float,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    descriptor_batches: Dict[str, List[np.ndarray]] = {
        "head": [],
        "foreground_tokens": [],
        "background_tokens": [],
        "fg_minus_bg": [],
        "fg_bg_concat": [],
    }
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    paths: List[str] = []
    region_stats: List[Dict[str, float]] = []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    seen_samples = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"region-separability-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                metadata = {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batch_size_value = int(images.shape[0])
            raw_paths = metadata.get("paths", [])
            batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
            if dataset_paths and (len(batch_paths) != batch_size_value or not any(str(path).strip() for path in batch_paths)):
                batch_paths = dataset_paths[seen_samples : seen_samples + batch_size_value]
            paths.extend(batch_paths)
            fallback_indices = np.arange(seen_samples, seen_samples + batch_size_value, dtype=np.int64)
            seen_samples += batch_size_value
            sample_index = metadata.get("sample_index")
            if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_size_value:
                sample_index_batches.append(sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1))
            else:
                sample_index_batches.append(fallback_indices)

            bbox = _tensor_metadata(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _tensor_metadata(metadata, "crop_bbox", device=device, dtype=torch.float32)
            image_mask = _tensor_metadata(metadata, "image_mask", device=device, dtype=torch.bool)
            bbox_prior = crop_bbox if str(bbox_token_prior_source) == "crop_bbox" and torch.is_tensor(crop_bbox) else bbox
            with autocast_context(device, amp):
                if not hasattr(model, "forward_features"):
                    raise TypeError("Region separability probe requires forward_features().")
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
                head_descriptor = extract_head_input_from_features(model, features).float()
                patches = features.get("patches")
                patch_prior = features.get("patch_bbox_prior")
                if not torch.is_tensor(patches) or patches.ndim != 3:
                    raise ValueError("Model features do not contain patch tokens.")
                if not torch.is_tensor(patch_prior) or tuple(patch_prior.shape) != tuple(patches.shape[:2]):
                    raise ValueError("Model features do not contain patch_bbox_prior matching patches.")
                fg_weights, bg_weights, _valid, stats = _region_weights(
                    bbox_prior=patch_prior,
                    key_padding_mask=features.get("memory_key_padding_mask"),
                    foreground_threshold=float(foreground_threshold),
                    background_threshold=float(background_threshold),
                )
                foreground_descriptor = _weighted_token_mean(patches.float(), fg_weights)
                background_descriptor = _weighted_token_mean(patches.float(), bg_weights)
                fg_minus_bg = foreground_descriptor - background_descriptor
                fg_bg_concat = torch.cat((foreground_descriptor, background_descriptor), dim=1)
            descriptor_batches["head"].append(head_descriptor.detach().float().cpu().numpy())
            descriptor_batches["foreground_tokens"].append(foreground_descriptor.detach().float().cpu().numpy())
            descriptor_batches["background_tokens"].append(background_descriptor.detach().float().cpu().numpy())
            descriptor_batches["fg_minus_bg"].append(fg_minus_bg.detach().float().cpu().numpy())
            descriptor_batches["fg_bg_concat"].append(fg_bg_concat.detach().float().cpu().numpy())
            probability_batches.append(probabilities.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            prediction_batches.append(probabilities.argmax(dim=1).detach().cpu().numpy())
            region_stats.append(stats)
    descriptors = {
        name: (
            np.concatenate(batches, axis=0).astype(np.float32, copy=False)
            if batches
            else np.zeros((0, 0), dtype=np.float32)
        )
        for name, batches in descriptor_batches.items()
    }
    mean_stats: Dict[str, float] = {}
    if region_stats:
        for key in region_stats[0].keys():
            values = [float(item.get(key, 0.0) or 0.0) for item in region_stats]
            mean_stats[key] = float(np.mean(values))
    return {
        "descriptors": descriptors,
        "probabilities": (
            np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)
            if probability_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "labels": (
            np.concatenate(label_batches, axis=0).astype(np.int64, copy=False)
            if label_batches
            else np.zeros((0,), dtype=np.int64)
        ),
        "base_predictions": (
            np.concatenate(prediction_batches, axis=0).astype(np.int64, copy=False)
            if prediction_batches
            else np.zeros((0,), dtype=np.int64)
        ),
        "sample_indices": (
            np.concatenate(sample_index_batches, axis=0).astype(np.int64, copy=False)
            if sample_index_batches
            else np.zeros((0,), dtype=np.int64)
        ),
        "paths": paths,
        "region_stats": mean_stats,
    }


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "split",
        "descriptor",
        "method",
        "accuracy",
        "macro_f1",
        "class1_f1",
        "class1_precision",
        "class1_recall",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_predictions_csv(
    path: Path,
    *,
    split: str,
    payload: Mapping[str, object],
    predictions_by_name: Mapping[str, np.ndarray],
) -> None:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    sample_indices = np.asarray(payload["sample_indices"], dtype=np.int64)
    paths = list(payload.get("paths", []))
    fieldnames = ["split", "sample_index", "image_path", "target_index", *[f"pred_{name}" for name in predictions_by_name]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, target in enumerate(labels):
            row = {
                "split": split,
                "sample_index": int(sample_indices[row_index]) if row_index < len(sample_indices) else int(row_index),
                "image_path": str(paths[row_index]) if row_index < len(paths) else "",
                "target_index": int(target),
            }
            for name, predictions in predictions_by_name.items():
                row[f"pred_{name}"] = int(predictions[row_index])
            writer.writerow(row)


def _class1(metrics: Mapping[str, object]) -> Mapping[str, object]:
    per_class = metrics.get("per_class", [])
    if isinstance(per_class, Sequence) and len(per_class) > 1 and isinstance(per_class[1], Mapping):
        return per_class[1]
    return {}


def _metric_row(
    *,
    split: str,
    descriptor: str,
    method: str,
    metrics: Mapping[str, object],
) -> Dict[str, object]:
    class1 = _class1(metrics)
    return {
        "split": split,
        "descriptor": descriptor,
        "method": method,
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "class1_f1": float(class1.get("f1", 0.0) or 0.0),
        "class1_precision": float(class1.get("precision", 0.0) or 0.0),
        "class1_recall": float(class1.get("recall", 0.0) or 0.0),
    }


def _fit_predictions(
    *,
    train_descriptors: np.ndarray,
    train_labels: np.ndarray,
    eval_descriptors: np.ndarray,
    class_count: int,
    knn_k: Sequence[int],
    logistic_max_iter: int,
    skip_logistic: bool,
) -> Dict[str, np.ndarray]:
    predictions = {
        "centroid_cosine": _predict_nearest_centroid_cosine(
            train_descriptors,
            train_labels,
            eval_descriptors,
            class_count,
        )
    }
    for k_value in knn_k:
        predictions[f"knn_cosine_k{int(k_value)}"] = _predict_knn_cosine(
            train_descriptors,
            train_labels,
            eval_descriptors,
            class_count,
            k=int(k_value),
        )
    if not bool(skip_logistic):
        logistic = _fit_predict_logistic(
            train_descriptors,
            train_labels,
            eval_descriptors,
            class_weight=None,
            max_iter=int(logistic_max_iter),
        )
        if logistic is not None:
            predictions["logistic"] = logistic
        balanced = _fit_predict_logistic(
            train_descriptors,
            train_labels,
            eval_descriptors,
            class_weight="balanced",
            max_iter=int(logistic_max_iter),
        )
        if balanced is not None:
            predictions["logistic_balanced"] = balanced
    return predictions


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    requested_splits = list(args.split or ["train", "val"])
    if "test" in {str(split).lower() for split in requested_splits} and not bool(args.allow_test):
        raise ValueError("This diagnostic refuses test split unless --allow-test is set for final audit only.")
    if "train" not in requested_splits:
        raise ValueError("Region separability probe requires train split for train-only fitting.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

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
        split_payloads[str(split)] = _extract_split_regions(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=str(split),
            bbox_token_prior_source=str(args.bbox_token_prior_source),
            foreground_threshold=float(args.foreground_threshold),
            background_threshold=float(args.background_threshold),
        )

    train = split_payloads["train"]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    train_descriptors = train["descriptors"]
    class_count = len(class_names)
    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "bbox_token_prior_source": str(args.bbox_token_prior_source),
        "foreground_threshold": float(args.foreground_threshold),
        "background_threshold": float(args.background_threshold),
        "splits": {},
        "region_stats": {
            split: payload.get("region_stats", {}) for split, payload in split_payloads.items()
        },
        "leakage_guard": "decision layers are fit on train descriptors only; test is refused unless --allow-test",
    }
    metric_rows: List[Dict[str, object]] = []
    for split, payload in split_payloads.items():
        labels = np.asarray(payload["labels"], dtype=np.int64)
        split_predictions: Dict[str, np.ndarray] = {
            "base_head": np.asarray(payload["base_predictions"], dtype=np.int64)
        }
        summary["splits"][split] = {
            "support": int(labels.size),
            "base_head": _classification_metrics(labels, split_predictions["base_head"], class_names),
            "descriptors": {},
        }
        metric_rows.append(
            _metric_row(
                split=split,
                descriptor="head",
                method="base_head",
                metrics=summary["splits"][split]["base_head"],
            )
        )
        descriptors = payload["descriptors"]
        for descriptor_name, eval_descriptors in descriptors.items():
            train_descriptor_values = np.asarray(train_descriptors[descriptor_name], dtype=np.float32)
            eval_descriptor_values = np.asarray(eval_descriptors, dtype=np.float32)
            predictions_by_method = _fit_predictions(
                train_descriptors=train_descriptor_values,
                train_labels=train_labels,
                eval_descriptors=eval_descriptor_values,
                class_count=class_count,
                knn_k=list(args.knn_k),
                logistic_max_iter=int(args.logistic_max_iter),
                skip_logistic=bool(args.skip_logistic),
            )
            descriptor_metrics = {}
            for method, predictions in predictions_by_method.items():
                name = f"{descriptor_name}_{method}"
                split_predictions[name] = predictions
                metrics = _classification_metrics(labels, predictions, class_names)
                descriptor_metrics[method] = metrics
                metric_rows.append(
                    _metric_row(
                        split=split,
                        descriptor=descriptor_name,
                        method=method,
                        metrics=metrics,
                    )
                )
            summary["splits"][split]["descriptors"][descriptor_name] = descriptor_metrics
        _write_predictions_csv(
            output_dir / f"predictions_{split}.csv",
            split=split,
            payload=payload,
            predictions_by_name=split_predictions,
        )
    _write_metrics_csv(output_dir / "region_descriptor_metrics.csv", metric_rows)
    summary["elapsed_seconds"] = float(time.perf_counter() - start_time)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    best_val = [
        row for row in metric_rows if str(row.get("split")) == "val"
    ]
    best_val_sorted = sorted(
        best_val,
        key=lambda row: (float(row.get("class1_f1", 0.0)), float(row.get("macro_f1", 0.0))),
        reverse=True,
    )
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# BBox Token Region Separability Probe\n\n")
        handle.write(f"- Data: `{Path(args.data).resolve()}`\n")
        handle.write(f"- Checkpoint: `{Path(args.checkpoint).resolve()}`\n")
        handle.write(f"- Splits: `{', '.join(requested_splits)}`\n")
        handle.write(f"- Test access: `{bool('test' in requested_splits)}`\n\n")
        handle.write("## Top Validation Rows\n\n")
        handle.write("| descriptor | method | macro F1 | class1 F1 | class1 P | class1 R |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in best_val_sorted[:12]:
            handle.write(
                f"| {row['descriptor']} | {row['method']} | "
                f"{float(row['macro_f1']):.4f} | {float(row['class1_f1']):.4f} | "
                f"{float(row['class1_precision']):.4f} | {float(row['class1_recall']):.4f} |\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
