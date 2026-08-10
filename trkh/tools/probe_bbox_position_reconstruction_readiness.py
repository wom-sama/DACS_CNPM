from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.data.dataset import build_train_collate_fn
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _class1_f1,
    _forward_logits,
    _select_bbox_token_prior,
)
from trkh.tools.probe_paired_bbox_aligned_patch_correspondence import (
    _xywh_bounds,
    patch_centers_from_indices,
)


class RidgeAccumulator:
    def __init__(self, feature_dim: int, target_dim: int = 2) -> None:
        if feature_dim < 1 or target_dim < 1:
            raise ValueError("ridge dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)
        self.xtx = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64)
        self.xty = torch.zeros((feature_dim + 1, target_dim), dtype=torch.float64)
        self.sample_count = 0

    def update(self, features: Tensor, targets: Tensor) -> None:
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError("ridge feature shape mismatch")
        if targets.ndim != 2 or targets.shape != (features.size(0), self.target_dim):
            raise ValueError("ridge target shape mismatch")
        x = features.detach().to(device="cpu", dtype=torch.float64)
        y = targets.detach().to(device="cpu", dtype=torch.float64)
        design = torch.cat((x, torch.ones((x.size(0), 1), dtype=x.dtype)), dim=1)
        self.xtx += design.transpose(0, 1) @ design
        self.xty += design.transpose(0, 1) @ y
        self.sample_count += int(x.size(0))

    def solve(self, l2: float = 1e-2) -> Tensor:
        if self.sample_count <= self.feature_dim:
            raise ValueError("not enough ridge samples")
        regularizer = torch.eye(self.feature_dim + 1, dtype=torch.float64) * float(l2)
        regularizer[-1, -1] = 0.0
        return torch.linalg.solve(self.xtx + regularizer, self.xty)


def ridge_predict(features: Tensor, coefficients: Tensor) -> Tensor:
    if features.ndim != 2 or coefficients.ndim != 2:
        raise ValueError("ridge prediction expects matrices")
    if coefficients.size(0) != features.size(1) + 1:
        raise ValueError("ridge coefficient shape mismatch")
    x = features.detach().to(device="cpu", dtype=torch.float64)
    design = torch.cat((x, torch.ones((x.size(0), 1), dtype=x.dtype)), dim=1)
    return design @ coefficients.to(dtype=torch.float64)


def bbox_relative_patch_targets(
    *,
    patch_indices: Tensor,
    grid_size: Sequence[int],
    bbox: Tensor,
    valid_mask: Optional[Tensor] = None,
    max_tokens: int = 32,
) -> Tuple[Tensor, Tensor]:
    centers = patch_centers_from_indices(patch_indices, grid_size)
    device = centers.device
    lower, size = _xywh_bounds(bbox.to(device=device))
    upper = lower + size
    selected = (
        (centers[:, 0] >= lower[0])
        & (centers[:, 0] <= upper[0])
        & (centers[:, 1] >= lower[1])
        & (centers[:, 1] <= upper[1])
    )
    if torch.is_tensor(valid_mask):
        if valid_mask.numel() != selected.numel():
            raise ValueError("valid mask shape mismatch")
        selected &= valid_mask.to(device=device, dtype=torch.bool).flatten()
    local_indices = torch.nonzero(selected, as_tuple=False).flatten()
    if int(max_tokens) > 0 and local_indices.numel() > int(max_tokens):
        positions = torch.linspace(
            0,
            local_indices.numel() - 1,
            steps=int(max_tokens),
            device=device,
        ).round().to(dtype=torch.long)
        local_indices = local_indices.index_select(0, positions)
    targets = (
        centers.index_select(0, local_indices) - lower.unsqueeze(0)
    ) / size.unsqueeze(0)
    return local_indices, targets.clamp(0.0, 1.0)


def scale_patch_position_embeddings(
    position_embeddings: Tensor,
    grid_size: Sequence[int],
    scale: float,
) -> Tensor:
    if position_embeddings.ndim != 3:
        raise ValueError("position embeddings must have shape [1, token, channel]")
    patch_count = int(grid_size[0]) * int(grid_size[1])
    if patch_count <= 0 or position_embeddings.size(1) < patch_count:
        raise ValueError("position embedding patch count mismatch")
    output = position_embeddings.clone()
    output[:, -patch_count:] *= float(scale)
    return output


@contextmanager
def patch_position_scale(model: nn.Module, scale: float) -> Iterator[None]:
    original = getattr(model, "get_interpolated_pos_embed", None)
    if not callable(original):
        raise TypeError("model does not expose get_interpolated_pos_embed")

    def scaled(grid_size):
        return scale_patch_position_embeddings(original(grid_size), grid_size, scale)

    model.get_interpolated_pos_embed = scaled  # type: ignore[method-assign]
    try:
        yield
    finally:
        model.get_interpolated_pos_embed = original  # type: ignore[method-assign]


def position_metrics(predictions: Tensor, targets: Tensor, bins: int = 4) -> Dict[str, object]:
    predictions = predictions.to(dtype=torch.float64).clamp(0.0, 1.0)
    targets = targets.to(dtype=torch.float64).clamp(0.0, 1.0)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.size(1) != 2:
        raise ValueError("position metrics expect matching [N, 2] tensors")
    if targets.size(0) == 0:
        return {
            "token_count": 0,
            "mae_x": 1.0,
            "mae_y": 1.0,
            "mean_mae": 1.0,
            "r2_x": -1.0,
            "r2_y": -1.0,
            "mean_r2": -1.0,
            "coarse_bins": int(bins),
            "coarse_joint_accuracy": 0.0,
            "coarse_random_accuracy": float(1.0 / float(int(bins) ** 2)),
        }
    error = predictions - targets
    abs_error = error.abs()
    target_mean = targets.mean(dim=0)
    residual = error.square().sum(dim=0)
    total = (targets - target_mean.unsqueeze(0)).square().sum(dim=0).clamp(min=1e-12)
    r2 = 1.0 - residual / total
    pred_bins = torch.floor(predictions * int(bins)).clamp(0, int(bins) - 1).to(dtype=torch.long)
    target_bins = torch.floor(targets * int(bins)).clamp(0, int(bins) - 1).to(dtype=torch.long)
    joint_correct = (pred_bins == target_bins).all(dim=1)
    return {
        "token_count": int(targets.size(0)),
        "mae_x": float(abs_error[:, 0].mean().item()),
        "mae_y": float(abs_error[:, 1].mean().item()),
        "mean_mae": float(abs_error.mean().item()),
        "r2_x": float(r2[0].item()),
        "r2_y": float(r2[1].item()),
        "mean_r2": float(r2.mean().item()),
        "coarse_bins": int(bins),
        "coarse_joint_accuracy": float(joint_correct.float().mean().item()),
        "coarse_random_accuracy": float(1.0 / float(int(bins) ** 2)),
    }


def assess_position_readiness(
    *,
    metrics: Mapping[str, object],
    class1_sample_mean_mae: float,
    baseline_macro_f1: float,
    ablated_macro_f1: float,
    changed_predictions: int,
    train_token_count: int,
    val_support_complete: bool,
    min_train_tokens: int = 50000,
    min_val_tokens: int = 50000,
    min_mean_r2: float = 0.10,
    min_coarse_accuracy: float = 0.15,
    max_mean_mae: float = 0.30,
    max_class1_mae_ratio: float = 1.25,
    min_changed_predictions: int = 25,
    max_macro_f1_drop: float = 0.30,
) -> Dict[str, object]:
    def number(values: Mapping[str, object], key: str, default: float) -> float:
        value = values.get(key)
        return float(default if value is None else value)

    mean_mae = number(metrics, "mean_mae", 1.0)
    macro_drop = float(baseline_macro_f1) - float(ablated_macro_f1)
    checks = {
        "val_support_complete": bool(val_support_complete),
        "train_token_count": int(train_token_count) >= int(min_train_tokens),
        "val_token_count": int(metrics.get("token_count", 0) or 0) >= int(min_val_tokens),
        "mean_r2": number(metrics, "mean_r2", -1.0) >= float(min_mean_r2),
        "coarse_accuracy": number(metrics, "coarse_joint_accuracy", 0.0)
        >= float(min_coarse_accuracy),
        "mean_mae": mean_mae <= float(max_mean_mae),
        "class1_mae": float(class1_sample_mean_mae) <= mean_mae * float(max_class1_mae_ratio),
        "classification_position_sensitive": int(changed_predictions) >= int(min_changed_predictions),
        "ablation_not_catastrophic": macro_drop <= float(max_macro_f1_drop),
    }
    failed = [name for name, passed in checks.items() if not passed]
    ready = all(checks.values())
    return {
        "position_target_ready": bool(ready),
        "smoke_ready": bool(ready),
        "training_permission": bool(ready),
        "checks": checks,
        "failed_checks": failed,
        "observed_macro_f1_drop": float(macro_drop),
        "thresholds": {
            "min_train_tokens": int(min_train_tokens),
            "min_val_tokens": int(min_val_tokens),
            "min_mean_r2": float(min_mean_r2),
            "min_coarse_accuracy": float(min_coarse_accuracy),
            "max_mean_mae": float(max_mean_mae),
            "max_class1_mae_ratio": float(max_class1_mae_ratio),
            "min_changed_predictions": int(min_changed_predictions),
            "max_macro_f1_drop": float(max_macro_f1_drop),
        },
    }


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        return
    fieldnames: List[str] = []
    for row in records:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: metrics.get(key)
        for key in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "per_class",
            "confusion_matrix",
        )
        if key in metrics
    }


def _move_metadata(metadata: Mapping[str, object], name: str, device: torch.device, dtype):
    value = metadata.get(name)
    return value.to(device=device, dtype=dtype, non_blocking=True) if torch.is_tensor(value) else None


def _build_loader(dataset, *, batch_size: int, num_workers: int, device: torch.device, context: str):
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=device.type == "cuda",
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **kwargs,
    )
    return loader, summary


def _token_batch(
    *,
    features: Mapping[str, object],
    crop_bbox: Tensor,
    labels: Tensor,
    max_tokens_per_sample: int,
) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, object]]]:
    patches = features.get("patches")
    patch_indices = features.get("patch_indices")
    grid_size = features.get("grid_size")
    key_padding = features.get("memory_key_padding_mask")
    if not torch.is_tensor(patches) or not torch.is_tensor(patch_indices):
        raise TypeError("model features do not contain patches and patch_indices")
    feature_parts: List[Tensor] = []
    target_parts: List[Tensor] = []
    label_parts: List[Tensor] = []
    sample_info: List[Dict[str, object]] = []
    for index in range(patches.size(0)):
        valid = None
        if torch.is_tensor(key_padding):
            valid = ~key_padding[index].to(dtype=torch.bool)
        local_indices, targets = bbox_relative_patch_targets(
            patch_indices=patch_indices[index],
            grid_size=grid_size,
            bbox=crop_bbox[index],
            valid_mask=valid,
            max_tokens=max_tokens_per_sample,
        )
        if local_indices.numel() == 0:
            sample_info.append({"token_count": 0})
            continue
        feature_parts.append(patches[index].index_select(0, local_indices))
        target_parts.append(targets)
        label_parts.append(labels[index].expand(local_indices.numel()))
        sample_info.append(
            {
                "token_count": int(local_indices.numel()),
                "features": patches[index].index_select(0, local_indices),
                "targets": targets,
            }
        )
    if not feature_parts:
        raise ValueError("batch contains no bbox-relative patch targets")
    return (
        torch.cat(feature_parts, dim=0),
        torch.cat(target_parts, dim=0),
        torch.cat(label_parts, dim=0),
        sample_info,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether crop-trained patch tokens can reconstruct bbox-relative positions "
            "after patch positional embeddings are removed."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=120)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-tokens-per-sample", type=int, default=32)
    parser.add_argument("--patch-position-scale", type=float, default=0.0)
    parser.add_argument("--ridge-l2", type=float, default=0.01)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--min-train-tokens", type=int, default=50000)
    parser.add_argument("--min-val-tokens", type=int, default=50000)
    parser.add_argument("--min-mean-r2", type=float, default=0.10)
    parser.add_argument("--min-coarse-accuracy", type=float, default=0.15)
    parser.add_argument("--max-mean-mae", type=float, default=0.30)
    parser.add_argument("--max-class1-mae-ratio", type=float, default=1.25)
    parser.add_argument("--min-changed-predictions", type=int, default=25)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.patch_position_scale <= 1.0:
        raise ValueError("--patch-position-scale must be in [0,1]")
    if args.max_tokens_per_sample < 1:
        raise ValueError("--max-tokens-per-sample must be positive")
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    data_spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint=checkpoint, num_classes=len(class_names))
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 224))
    transform = _build_eval_transform_from_checkpoint(checkpoint, image_size=image_size)
    train_dataset = _build_classification_dataset(
        data_spec=data_spec,
        split="train",
        transform=transform,
        checkpoint=checkpoint,
    )
    val_dataset = _build_classification_dataset(
        data_spec=data_spec,
        split="val",
        transform=transform,
        checkpoint=checkpoint,
    )
    train_loader, train_loader_summary = _build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        context="bbox_position_readiness_train",
    )
    val_loader, val_loader_summary = _build_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        context="bbox_position_readiness_val",
    )

    ridge: Optional[RidgeAccumulator] = None
    train_samples = 0
    train_batches = len(train_loader)
    train_iterator = train_loader
    if args.max_train_batches > 0:
        train_batches = min(train_batches, int(args.max_train_batches))
        train_iterator = islice(train_loader, train_batches)
    with torch.inference_mode(), patch_position_scale(model, args.patch_position_scale):
        for images, labels, metadata in tqdm(
            train_iterator,
            total=train_batches,
            desc="Position ridge train extraction",
            dynamic_ncols=True,
        ):
            metadata = metadata if isinstance(metadata, dict) else {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            bbox = _move_metadata(metadata, "bbox", device, torch.float32)
            crop_bbox = _move_metadata(metadata, "crop_bbox", device, torch.float32)
            image_mask = _move_metadata(metadata, "image_mask", device, torch.bool)
            if crop_bbox is None:
                raise ValueError("crop_bbox metadata is required")
            prior = _select_bbox_token_prior(
                bbox=bbox,
                crop_bbox=crop_bbox,
                source=args.bbox_token_prior_source,
            )
            _, features = _forward_logits(
                model=model,
                images=images,
                image_valid_mask=image_mask,
                bbox_metadata=bbox,
                bbox_token_prior=prior,
                amp=args.amp,
                device=device,
            )
            if not isinstance(features, dict):
                raise TypeError("model did not return feature dictionary")
            feature_batch, target_batch, _, _ = _token_batch(
                features=features,
                crop_bbox=crop_bbox,
                labels=labels,
                max_tokens_per_sample=args.max_tokens_per_sample,
            )
            if ridge is None:
                ridge = RidgeAccumulator(feature_dim=int(feature_batch.size(1)))
            ridge.update(feature_batch, target_batch)
            train_samples += int(images.size(0))
    if ridge is None:
        raise RuntimeError("ridge extraction produced no tokens")
    coefficients = ridge.solve(l2=args.ridge_l2)

    val_batches = len(val_loader)
    val_iterator = val_loader
    if args.max_val_batches > 0:
        val_batches = min(val_batches, int(args.max_val_batches))
        val_iterator = islice(val_loader, val_batches)
    baseline_probability_batches: List[Tensor] = []
    ablated_probability_batches: List[Tensor] = []
    baseline_logit_batches: List[Tensor] = []
    ablated_logit_batches: List[Tensor] = []
    target_batches: List[Tensor] = []
    position_prediction_parts: List[Tensor] = []
    position_target_parts: List[Tensor] = []
    position_label_parts: List[Tensor] = []
    sample_rows: List[Dict[str, object]] = []
    sample_offset = 0
    with torch.inference_mode():
        for images, labels, metadata in tqdm(
            val_iterator,
            total=val_batches,
            desc="Position readiness validation",
            dynamic_ncols=True,
        ):
            metadata = metadata if isinstance(metadata, dict) else {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            bbox = _move_metadata(metadata, "bbox", device, torch.float32)
            crop_bbox = _move_metadata(metadata, "crop_bbox", device, torch.float32)
            image_mask = _move_metadata(metadata, "image_mask", device, torch.bool)
            if crop_bbox is None:
                raise ValueError("crop_bbox metadata is required")
            prior = _select_bbox_token_prior(
                bbox=bbox,
                crop_bbox=crop_bbox,
                source=args.bbox_token_prior_source,
            )
            baseline_logits, _ = _forward_logits(
                model=model,
                images=images,
                image_valid_mask=image_mask,
                bbox_metadata=bbox,
                bbox_token_prior=prior,
                amp=args.amp,
                device=device,
            )
            with patch_position_scale(model, args.patch_position_scale):
                ablated_logits, ablated_features = _forward_logits(
                    model=model,
                    images=images,
                    image_valid_mask=image_mask,
                    bbox_metadata=bbox,
                    bbox_token_prior=prior,
                    amp=args.amp,
                    device=device,
                )
            if not isinstance(ablated_features, dict):
                raise TypeError("model did not return ablated feature dictionary")
            feature_batch, position_targets, token_labels, sample_info = _token_batch(
                features=ablated_features,
                crop_bbox=crop_bbox,
                labels=labels,
                max_tokens_per_sample=args.max_tokens_per_sample,
            )
            predictions = ridge_predict(feature_batch, coefficients)
            position_prediction_parts.append(predictions)
            position_target_parts.append(position_targets.detach().cpu().to(dtype=torch.float64))
            position_label_parts.append(token_labels.detach().cpu().to(dtype=torch.long))

            cursor = 0
            baseline_predictions = baseline_logits.argmax(dim=1)
            ablated_predictions = ablated_logits.argmax(dim=1)
            for local_index, info in enumerate(sample_info):
                token_count = int(info.get("token_count", 0) or 0)
                target = int(labels[local_index].item())
                row: Dict[str, object] = {
                    "sample_index": int(sample_offset + local_index),
                    "target": target,
                    "baseline_prediction": int(baseline_predictions[local_index].item()),
                    "ablated_prediction": int(ablated_predictions[local_index].item()),
                    "prediction_changed": int(
                        baseline_predictions[local_index] != ablated_predictions[local_index]
                    ),
                    "token_count": token_count,
                }
                if token_count > 0:
                    sample_prediction = predictions[cursor : cursor + token_count]
                    sample_target = position_targets[cursor : cursor + token_count].detach().cpu()
                    sample_metric = position_metrics(sample_prediction, sample_target)
                    row.update(
                        {
                            "position_mean_mae": sample_metric["mean_mae"],
                            "position_coarse_accuracy": sample_metric["coarse_joint_accuracy"],
                        }
                    )
                    cursor += token_count
                sample_rows.append(row)
            sample_offset += int(images.size(0))
            baseline_probability_batches.append(torch.softmax(baseline_logits.float(), dim=1).cpu())
            ablated_probability_batches.append(torch.softmax(ablated_logits.float(), dim=1).cpu())
            baseline_logit_batches.append(baseline_logits.detach().float().cpu())
            ablated_logit_batches.append(ablated_logits.detach().float().cpu())
            target_batches.append(labels.detach().cpu())

    targets = torch.cat(target_batches, dim=0)
    baseline_probabilities = torch.cat(baseline_probability_batches, dim=0)
    ablated_probabilities = torch.cat(ablated_probability_batches, dim=0)
    baseline_logits = torch.cat(baseline_logit_batches, dim=0)
    ablated_logits = torch.cat(ablated_logit_batches, dim=0)
    baseline_predictions = baseline_probabilities.argmax(dim=1)
    ablated_predictions = ablated_probabilities.argmax(dim=1)
    baseline_metrics = build_metrics(
        targets=targets,
        predictions=baseline_predictions,
        probabilities=baseline_probabilities,
        class_names=class_names,
    )
    ablated_metrics = build_metrics(
        targets=targets,
        predictions=ablated_predictions,
        probabilities=ablated_probabilities,
        class_names=class_names,
    )
    position_predictions = torch.cat(position_prediction_parts, dim=0)
    position_targets = torch.cat(position_target_parts, dim=0)
    position_labels = torch.cat(position_label_parts, dim=0)
    all_position_metrics = position_metrics(position_predictions, position_targets)
    class1_mask = position_labels == 1
    class1_position_metrics = position_metrics(
        position_predictions[class1_mask],
        position_targets[class1_mask],
    )
    changed_predictions = int((baseline_predictions != ablated_predictions).sum().item())
    corrections = int(
        ((baseline_predictions != targets) & (ablated_predictions == targets)).sum().item()
    )
    harms = int(
        ((baseline_predictions == targets) & (ablated_predictions != targets)).sum().item()
    )
    logit_delta = (baseline_logits - ablated_logits).abs()
    probability_delta = (baseline_probabilities - ablated_probabilities).abs()
    val_support_complete = int(targets.numel()) == int(len(val_dataset))
    gate = assess_position_readiness(
        metrics=all_position_metrics,
        class1_sample_mean_mae=float(class1_position_metrics["mean_mae"]),
        baseline_macro_f1=float(baseline_metrics["macro_f1"]),
        ablated_macro_f1=float(ablated_metrics["macro_f1"]),
        changed_predictions=changed_predictions,
        train_token_count=ridge.sample_count,
        val_support_complete=val_support_complete,
        min_train_tokens=args.min_train_tokens,
        min_val_tokens=args.min_val_tokens,
        min_mean_r2=args.min_mean_r2,
        min_coarse_accuracy=args.min_coarse_accuracy,
        max_mean_mae=args.max_mean_mae,
        max_class1_mae_ratio=args.max_class1_mae_ratio,
        min_changed_predictions=args.min_changed_predictions,
        max_macro_f1_drop=args.max_macro_f1_drop,
    )
    summary = {
        "mode": "bbox_position_reconstruction_readiness_precheck",
        "guardrail": (
            "Train split fits only a frozen ridge position decoder; validation audits a frozen "
            "checkpoint. No model training, test use, raw-data edit, or trainable manifest."
        ),
        "checkpoint": str(args.checkpoint),
        "data": str(data_spec.data_yaml),
        "patch_position_scale": float(args.patch_position_scale),
        "bbox_token_prior_source": args.bbox_token_prior_source,
        "train_samples": int(train_samples),
        "train_dataset_samples": int(len(train_dataset)),
        "train_token_count": int(ridge.sample_count),
        "val_samples": int(targets.numel()),
        "val_dataset_samples": int(len(val_dataset)),
        "val_support_complete": bool(val_support_complete),
        "val_token_count": int(position_targets.size(0)),
        "train_dataloader": train_loader_summary,
        "val_dataloader": val_loader_summary,
        "position_metrics": all_position_metrics,
        "class1_position_metrics": class1_position_metrics,
        "baseline_metrics": _compact_metrics(baseline_metrics),
        "ablated_metrics": _compact_metrics(ablated_metrics),
        "baseline_class1_f1": float(_class1_f1(baseline_metrics, 1)),
        "ablated_class1_f1": float(_class1_f1(ablated_metrics, 1)),
        "classification_changes": {
            "changed_predictions": changed_predictions,
            "corrections": corrections,
            "harms": harms,
            "mean_absolute_logit_delta": float(logit_delta.mean().item()),
            "max_absolute_logit_delta": float(logit_delta.max().item()),
            "mean_absolute_probability_delta": float(probability_delta.mean().item()),
            "max_absolute_probability_delta": float(probability_delta.max().item()),
        },
        "gate": gate,
        "literature": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract.html",
            "https://openaccess.thecvf.com/content/WACV2024/papers/Caron_Location-Aware_Self-Supervised_Transformers_for_Semantic_Segmentation_WACV_2024_paper.pdf",
        ],
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_written": False,
        "trainable_manifest_written": False,
    }
    _write_rows_csv(output_dir / "per_sample_position_readiness.csv", sample_rows)
    json_dump(output_dir / "summary.json", to_serializable(summary))
    readme = [
        "# BBox Position Reconstruction Readiness",
        "",
        f"- Train tokens: `{ridge.sample_count}`",
        f"- Validation support: `{int(targets.numel())}/{len(val_dataset)}`",
        f"- Position mean R2: `{float(all_position_metrics['mean_r2']):.6f}`",
        f"- Position 4x4 accuracy: `{float(all_position_metrics['coarse_joint_accuracy']):.6f}`",
        f"- Baseline/ablated macro F1: `{float(baseline_metrics['macro_f1']):.6f}/{float(ablated_metrics['macro_f1']):.6f}`",
        f"- Baseline/ablated class1 F1: `{float(_class1_f1(baseline_metrics, 1)):.6f}/{float(_class1_f1(ablated_metrics, 1)):.6f}`",
        f"- Smoke ready: `{str(gate['smoke_ready']).lower()}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        "",
        "The ridge decoder is used in memory only; this tool writes no model or training target.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(
        {
            "train_tokens": int(ridge.sample_count),
            "val_samples": int(targets.numel()),
            "mean_r2": float(all_position_metrics["mean_r2"]),
            "coarse_accuracy": float(all_position_metrics["coarse_joint_accuracy"]),
            "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
            "ablated_macro_f1": float(ablated_metrics["macro_f1"]),
            "baseline_class1_f1": float(_class1_f1(baseline_metrics, 1)),
            "ablated_class1_f1": float(_class1_f1(ablated_metrics, 1)),
            "smoke_ready": gate["smoke_ready"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
