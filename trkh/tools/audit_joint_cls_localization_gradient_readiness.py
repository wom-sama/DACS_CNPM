from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.data.dataset import MangoYOLOCropDataset, build_eval_transform
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
)
from trkh.tools.build_detection_classification_curriculum_checkpoint import (
    ENCODER_TRANSFER_PREFIXES,
    is_encoder_transfer_key,
    merge_matching_state,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)
from trkh.training.loss import DETRSetCriterion, unpack_detection_output


SEED = 20260712
FOLDS = 5
SAMPLES_PER_CLASS_PER_FOLD = 12
BATCH_SIZE = 4
HEAD_WARMUP_BATCH_SIZE = 8
HEAD_WARMUP_BATCHES = 120
HEAD_WARMUP_LEARNING_RATE = 2e-4
HEAD_WARMUP_WEIGHT_DECAY = 1e-4
STEP_PARAMETER_RATIO = 1e-4
AUXILIARY_GRADIENT_RATIO = 0.25
FOCUS_CLASS_INDEX = 1
DETECTION_CLASS_WEIGHT = 1.0
DETECTION_OBJECTNESS_WEIGHT = 1.0
DETECTION_BBOX_L1_WEIGHT = 0.20
DETECTION_BBOX_GIOU_WEIGHT = 0.10

DETECTOR_PARAMETER_PREFIXES = (
    "query_embed.",
    "query_content_embed.",
    "memory_adapter.",
    "decoder.",
    "classification_head.",
    "objectness_head.",
    "quality_head.",
    "bbox_head.",
    "count_head.",
)

LITERATURE = (
    {
        "name": "PCGrad",
        "url": "https://arxiv.org/abs/2001.06782",
        "role": "Project an auxiliary task gradient when it conflicts with the primary task.",
    },
    {
        "name": "Official PCGrad implementation",
        "url": "https://github.com/tianheyu927/PCGrad",
        "role": "Primary implementation reference for the projection rule.",
    },
    {
        "name": "GradNorm",
        "url": "https://proceedings.mlr.press/v80/chen18a.html",
        "role": "Reference for task-gradient magnitude imbalance in multitask learning.",
    },
)


@dataclass(frozen=True)
class PairedSource:
    source_stem: str
    label: int
    classification_index: int
    detection_index: int


Gradient = list[Tensor]
GradientMap = Dict[str, Gradient]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only readiness audit for classification-anchored joint full-frame "
            "localization. It never reads validation/test and never writes a model."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--samples-per-class-per-fold",
        type=int,
        default=SAMPLES_PER_CLASS_PER_FOLD,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--head-warmup-batch-size",
        type=int,
        default=HEAD_WARMUP_BATCH_SIZE,
    )
    parser.add_argument(
        "--head-warmup-batches",
        type=int,
        default=HEAD_WARMUP_BATCHES,
    )
    parser.add_argument(
        "--head-warmup-learning-rate",
        type=float,
        default=HEAD_WARMUP_LEARNING_RATE,
    )
    parser.add_argument(
        "--head-warmup-weight-decay",
        type=float,
        default=HEAD_WARMUP_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--step-parameter-ratio",
        type=float,
        default=STEP_PARAMETER_RATIO,
    )
    parser.add_argument(
        "--auxiliary-gradient-ratio",
        type=float,
        default=AUXILIARY_GRADIENT_RATIO,
    )
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-ffn-dim", type=int, default=512)
    parser.add_argument("--decoder-dropout", type=float, default=0.10)
    return parser.parse_args(argv)


def _load_checkpoint(path: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a mapping: {resolved}")
    for key in ("model_state", "model_config", "class_names"):
        if key not in payload:
            raise ValueError(f"Checkpoint is missing {key}: {resolved}")
    if not isinstance(payload["model_state"], Mapping):
        raise ValueError(f"Checkpoint model_state is invalid: {resolved}")
    if not isinstance(payload["model_config"], Mapping):
        raise ValueError(f"Checkpoint model_config is invalid: {resolved}")
    return payload


def _load_resolved_config(checkpoint: Path, explicit: Optional[Path]) -> Dict[str, object]:
    path = Path(explicit) if explicit is not None else Path(checkpoint).resolve().parents[1] / "resolved_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            "The keeper resolved_config.json is required to reproduce the classification view: "
            f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("augmentation_config"), Mapping):
        raise ValueError(f"Invalid resolved configuration: {path}")
    payload["_path"] = str(path.resolve())
    return payload


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "").strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(normalized or ("cuda" if torch.cuda.is_available() else "cpu"))


def _build_datasets(
    *,
    data_yaml: Path,
    model_config: Mapping[str, object],
    resolved_config: Mapping[str, object],
    class_count: int,
) -> Tuple[MangoYOLOCropDataset, MangoYOLOCropDataset, Dict[str, object]]:
    augmentation = dict(resolved_config["augmentation_config"])
    image_size = int(model_config.get("image_size", 256))
    resize_mode = str(augmentation.get("resize_mode", "pad"))
    mean = model_config.get("input_mean", (0.485, 0.456, 0.406))
    std = model_config.get("input_std", (0.229, 0.224, 0.225))
    illumination_normalization = bool(augmentation.get("illumination_normalization", False))
    illumination_strength = float(augmentation.get("illumination_normalization_strength", 0.0))

    classification_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=resize_mode,
        illumination_normalization=illumination_normalization,
        illumination_normalization_strength=illumination_strength,
        foreground_crop_mode=str(augmentation.get("foreground_crop_mode", "none")),
        foreground_crop_margin_ratio=float(augmentation.get("foreground_crop_margin_ratio", 0.08)),
        foreground_crop_min_mask_area_ratio=float(
            augmentation.get("foreground_crop_min_mask_area_ratio", 0.03)
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation.get("foreground_crop_max_mask_area_ratio", 0.92)
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation.get("foreground_crop_max_crop_area_ratio", 0.98)
        ),
        background_suppression_mode=str(
            augmentation.get("background_suppression_mode", "none")
        ),
        background_suppression_margin=float(
            augmentation.get("background_suppression_margin", 0.08)
        ),
        background_suppression_blur_radius=float(
            augmentation.get("background_suppression_blur_radius", 7.0)
        ),
        mean=mean,
        std=std,
    )
    # The auxiliary task keeps the full scene so it can learn object versus background.
    detection_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=resize_mode,
        illumination_normalization=illumination_normalization,
        illumination_normalization_strength=illumination_strength,
        foreground_crop_mode="none",
        background_suppression_mode="none",
        mean=mean,
        std=std,
    )

    classification_dataset = MangoYOLOCropDataset.from_data_yaml(
        data_yaml=Path(data_yaml),
        split="train",
        transform=classification_transform,
        crop_margin_ratio=float(augmentation.get("crop_margin_ratio", 0.05)),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        class_crop_margin_scale_threshold=float(
            augmentation.get("class_crop_margin_scale_threshold", 1.5)
        ),
        class_crop_margin_max_ratio=float(
            augmentation.get("class_crop_margin_max_ratio", 0.05)
        ),
        classification_bbox_metadata=True,
        expected_num_classes=class_count,
    )
    detection_dataset = MangoYOLOCropDataset.from_data_yaml(
        data_yaml=Path(data_yaml),
        split="train",
        transform=detection_transform,
        crop_to_primary_object=False,
        classification_target=False,
        classification_object_crops=False,
        expected_num_classes=class_count,
    )
    protocol = {
        "split": "train",
        "classification_view": "keeper object crop with resolved eval preprocessing",
        "detection_view": "context-preserving full frame with resize/pad and normalization",
        "classification_rows": int(len(classification_dataset)),
        "detection_frames": int(len(detection_dataset)),
        "detection_objects": int(detection_dataset.audit.get("valid_object_count", 0)),
        "classification_background_suppression": str(
            augmentation.get("background_suppression_mode", "none")
        ),
        "detection_background_suppression": "none",
        "image_size": image_size,
        "resize_mode": resize_mode,
        "illumination_normalization": illumination_normalization,
        "illumination_normalization_strength": illumination_strength,
    }
    return classification_dataset, detection_dataset, protocol


def build_paired_single_object_sources(
    classification_dataset: MangoYOLOCropDataset,
    detection_dataset: MangoYOLOCropDataset,
) -> list[PairedSource]:
    classification_by_key: Dict[Tuple[str, int], int] = {}
    for index, sample in enumerate(classification_dataset.samples):
        key = (sample.image_path.stem.casefold(), int(sample.primary_object_index))
        if key in classification_by_key:
            raise ValueError(f"Duplicate classification source/object key: {key}")
        classification_by_key[key] = int(index)

    pairs: list[PairedSource] = []
    for detection_index, sample in enumerate(detection_dataset.samples):
        if len(sample.objects) != 1:
            continue
        obj = sample.objects[0]
        stem = sample.image_path.stem.casefold()
        classification_index = classification_by_key.get((stem, int(obj.object_index)))
        if classification_index is None:
            raise ValueError(f"Missing classification row for single-object source: {stem}")
        pairs.append(
            PairedSource(
                source_stem=stem,
                label=int(obj.label),
                classification_index=int(classification_index),
                detection_index=int(detection_index),
            )
        )
    stems = [row.source_stem for row in pairs]
    if len(stems) != len(set(stems)):
        raise ValueError("Single-object source stems are not unique")
    return pairs


def select_stratified_audit_folds(
    pairs: Sequence[PairedSource],
    *,
    class_count: int,
    folds: int,
    samples_per_class_per_fold: int,
    seed: int,
) -> list[list[PairedSource]]:
    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    if int(samples_per_class_per_fold) <= 0:
        raise ValueError("samples_per_class_per_fold must be positive")
    rng = np.random.default_rng(int(seed))
    selected_by_class: Dict[int, list[PairedSource]] = {}
    required = int(folds) * int(samples_per_class_per_fold)
    for class_index in range(int(class_count)):
        candidates = [row for row in pairs if int(row.label) == class_index]
        if len(candidates) < required:
            raise ValueError(
                f"Class {class_index} has {len(candidates)} single-object sources; "
                f"{required} are required"
            )
        order = rng.permutation(len(candidates))[:required]
        selected_by_class[class_index] = [candidates[int(index)] for index in order]

    output: list[list[PairedSource]] = [[] for _ in range(int(folds))]
    for class_index in range(int(class_count)):
        rows = selected_by_class[class_index]
        for fold_index in range(int(folds)):
            start = fold_index * int(samples_per_class_per_fold)
            stop = start + int(samples_per_class_per_fold)
            output[fold_index].extend(rows[start:stop])
    for fold_index, rows in enumerate(output):
        order = rng.permutation(len(rows))
        output[fold_index] = [rows[int(index)] for index in order]
    all_stems = [row.source_stem for fold in output for row in fold]
    if len(all_stems) != len(set(all_stems)):
        raise ValueError("Audit folds overlap by source")
    return output


def _build_detector(
    checkpoint: Mapping[str, object],
    *,
    num_queries: int,
    decoder_depth: int,
    decoder_ffn_dim: int,
    decoder_dropout: float,
    seed: int,
) -> Tuple[nn.Module, Dict[str, object]]:
    torch.manual_seed(int(seed))
    model_config = dict(checkpoint["model_config"])
    model_config.update(
        {
            "model_type": "detr_vit_registers",
            "num_queries": int(num_queries),
            "decoder_depth": int(decoder_depth),
            "decoder_ffn_dim": int(decoder_ffn_dim),
            "decoder_dropout": float(decoder_dropout),
            "auxiliary_decoder_outputs": False,
            "quality_head": False,
            "count_head": False,
            "query_denoising_noise": 0.0,
        }
    )
    detector = create_model(
        num_classes=len(checkpoint["class_names"]),
        model_config=model_config,
    )
    merged, transfer = merge_matching_state(
        checkpoint["model_state"],
        detector.state_dict(),
        encoder_only=False,
    )
    if transfer["shape_mismatches"]:
        raise ValueError(f"Unsafe classifier-to-detector transfer: {transfer}")
    detector.load_state_dict(merged, strict=True)
    summary = {
        "model_type": "detr_vit_registers",
        "num_queries": int(num_queries),
        "decoder_depth": int(decoder_depth),
        "decoder_ffn_dim": int(decoder_ffn_dim),
        "decoder_dropout": float(decoder_dropout),
        "transfer": transfer,
    }
    return detector, summary


def _shared_parameter_names(classifier: nn.Module, detector: nn.Module) -> list[str]:
    classifier_parameters = dict(classifier.named_parameters())
    detector_parameters = dict(detector.named_parameters())
    names = []
    for name, parameter in classifier_parameters.items():
        if not is_encoder_transfer_key(name):
            continue
        other = detector_parameters.get(name)
        if other is None or tuple(other.shape) != tuple(parameter.shape):
            raise ValueError(f"Shared encoder parameter mismatch: {name}")
        names.append(name)
    if not names:
        raise ValueError("No shared encoder parameters were found")
    return names


def _set_detector_head_warmup_mode(detector: nn.Module, shared_names: Sequence[str]) -> list[nn.Parameter]:
    shared = set(shared_names)
    trainable: list[nn.Parameter] = []
    for name, parameter in detector.named_parameters():
        allowed = name not in shared and any(name.startswith(prefix) for prefix in DETECTOR_PARAMETER_PREFIXES)
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(parameter)
    if not trainable:
        raise ValueError("Detector head warm-up has no trainable parameters")
    detector.eval()
    for module_name in (
        "memory_adapter",
        "decoder",
        "classification_head",
        "objectness_head",
        "quality_head",
        "bbox_head",
        "count_head",
    ):
        module = getattr(detector, module_name, None)
        if isinstance(module, nn.Module):
            module.train()
    return trainable


def _detection_criterion(class_count: int) -> DETRSetCriterion:
    return DETRSetCriterion(
        num_classes=int(class_count),
        cls_weight=DETECTION_CLASS_WEIGHT,
        bbox_l1_weight=DETECTION_BBOX_L1_WEIGHT,
        bbox_giou_weight=DETECTION_BBOX_GIOU_WEIGHT,
        objectness_weight=DETECTION_OBJECTNESS_WEIGHT,
        matcher_class_cost=1.0,
        matcher_objectness_cost=1.0,
    )


def _stack_detection_batch(
    dataset: MangoYOLOCropDataset,
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[Tensor, list[Dict[str, Tensor]], Optional[Tensor]]:
    images: list[Tensor] = []
    targets: list[Dict[str, Tensor]] = []
    masks: list[Tensor] = []
    all_masks = True
    for index in indices:
        image, target = dataset[int(index)]
        if not torch.is_tensor(image):
            raise TypeError("Detection transform did not return a tensor")
        images.append(image)
        targets.append(
            {
                "labels": target["labels"].to(device=device, dtype=torch.long),
                "boxes": target["boxes"].to(device=device, dtype=torch.float32),
            }
        )
        mask = target.get("image_mask")
        if torch.is_tensor(mask):
            masks.append(mask.to(dtype=torch.bool))
        else:
            all_masks = False
    image_batch = torch.stack(images).to(device=device, non_blocking=True)
    mask_batch = (
        torch.stack(masks).to(device=device, non_blocking=True)
        if all_masks and len(masks) == len(images)
        else None
    )
    return image_batch, targets, mask_batch


def _stack_classification_batch(
    dataset: MangoYOLOCropDataset,
    rows: Sequence[PairedSource],
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    images: list[Tensor] = []
    labels: list[int] = []
    metadata_rows: Dict[str, list[Tensor]] = {}
    for row in rows:
        sample = dataset[int(row.classification_index)]
        if len(sample) != 3:
            raise ValueError("Classification audit requires bbox metadata")
        image, label, metadata = sample
        if int(label) != int(row.label):
            raise ValueError(f"Paired label mismatch for {row.source_stem}")
        images.append(image)
        labels.append(int(label))
        for key, value in metadata.items():
            if torch.is_tensor(value):
                metadata_rows.setdefault(key, []).append(value)
    batch_metadata = {
        key: torch.stack(values)
        for key, values in metadata_rows.items()
        if len(values) == len(images)
    }
    return (
        torch.stack(images),
        torch.as_tensor(labels, dtype=torch.long),
        batch_metadata,
    )


def _classification_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Tensor:
    image_mask = metadata.get("image_mask")
    bbox = metadata.get("bbox")
    features = model.forward_features(
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    if torch.is_tensor(bbox):
        features["bbox"] = bbox
    return classification_logits_from_features(model, features)


def _warm_up_detector_heads(
    detector: nn.Module,
    *,
    dataset: MangoYOLOCropDataset,
    allowed_indices: Sequence[int],
    shared_names: Sequence[str],
    class_count: int,
    batch_size: int,
    max_batches: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[list[Dict[str, float]], Dict[str, object]]:
    trainable = _set_detector_head_warmup_mode(detector, shared_names)
    criterion = _detection_criterion(class_count).to(device)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(np.asarray(allowed_indices, dtype=np.int64))
    required = int(batch_size) * int(max_batches)
    if int(order.size) < required:
        raise ValueError(
            f"Head warm-up requires {required} source frames, only {order.size} are available"
        )
    order = order[:required]
    curve: list[Dict[str, float]] = []
    for batch_index in range(int(max_batches)):
        start = batch_index * int(batch_size)
        indices = order[start : start + int(batch_size)].tolist()
        images, targets, masks = _stack_detection_batch(dataset, indices, device)
        output = detector(images, image_valid_mask=masks)
        loss, details = criterion(output, targets, return_details=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {"batch": float(batch_index + 1)}
        for key in (
            "loss",
            "cls_loss",
            "objectness_loss",
            "bbox_l1_loss",
            "bbox_giou_loss",
        ):
            row[key] = float(details[key])
        curve.append(row)
        if (batch_index + 1) % 20 == 0 or batch_index == 0:
            print(
                {
                    "detector_head_warmup": {
                        "batch": batch_index + 1,
                        "loss": row["loss"],
                        "cls_loss": row["cls_loss"],
                        "objectness_loss": row["objectness_loss"],
                    }
                },
                flush=True,
            )

    window = min(20, max(1, len(curve) // 3))
    first_mean = float(np.mean([row["loss"] for row in curve[:window]]))
    last_mean = float(np.mean([row["loss"] for row in curve[-window:]]))
    summary = {
        "batches": int(len(curve)),
        "batch_size": int(batch_size),
        "frames_seen": int(len(curve) * int(batch_size)),
        "first_window": int(window),
        "first_loss_mean": first_mean,
        "last_loss_mean": last_mean,
        "last_to_first_loss_ratio": float(last_mean / max(first_mean, 1e-12)),
        "minimum_loss": float(min(row["loss"] for row in curve)),
        "encoder_updated": False,
    }
    return curve, summary


def _set_gradient_measurement_mode(
    classifier: nn.Module,
    detector: nn.Module,
    shared_names: Sequence[str],
) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
    shared = set(shared_names)
    classifier.eval()
    detector.eval()
    classifier_parameters = dict(classifier.named_parameters())
    detector_parameters = dict(detector.named_parameters())
    for name, parameter in classifier_parameters.items():
        parameter.requires_grad_(name in shared)
    for name, parameter in detector_parameters.items():
        parameter.requires_grad_(name in shared)
    return (
        [classifier_parameters[name] for name in shared_names],
        [detector_parameters[name] for name in shared_names],
    )


def _detection_component_losses(
    criterion: DETRSetCriterion,
    model_output: Mapping[str, Tensor],
    targets: Sequence[Dict[str, Tensor]],
) -> Dict[str, Tensor]:
    logits, pred_boxes, objectness_logits = unpack_detection_output(model_output)
    if objectness_logits is None:
        raise ValueError("Gradient audit requires separate detector objectness")
    matcher_output = {
        "logits": logits,
        "boxes": pred_boxes,
        "objectness_logits": objectness_logits,
    }
    indices = criterion.matcher(matcher_output, targets)
    num_boxes = sum(int(target["labels"].numel()) for target in targets)
    cls_loss = criterion.matched_classification_loss(logits, targets, indices)
    target_objectness = criterion._objectness_targets_from_indices(objectness_logits, indices)
    objectness_loss = criterion.objectness_loss(
        objectness_logits,
        target_objectness,
        num_boxes=num_boxes,
    )
    matched_predictions, matched_targets = criterion._collect_matched_boxes(
        pred_boxes,
        targets,
        indices,
    )
    bbox_l1_loss, bbox_giou_loss = criterion.bbox_loss(
        matched_predictions,
        matched_targets,
        num_boxes=num_boxes,
    )
    return {
        "det_class": cls_loss,
        "det_objectness": objectness_loss,
        "det_bbox_l1": bbox_l1_loss,
        "det_bbox_giou": bbox_giou_loss,
    }


def _zero_gradient_like(parameters: Sequence[nn.Parameter], *, device: torch.device) -> Gradient:
    return [torch.zeros_like(parameter, device=device, dtype=torch.float32) for parameter in parameters]


def _accumulate_gradient(
    accumulator: Gradient,
    gradients: Sequence[Optional[Tensor]],
    *,
    weight: float,
) -> None:
    for target, gradient in zip(accumulator, gradients):
        if gradient is not None:
            target.add_(gradient.detach().to(dtype=torch.float32), alpha=float(weight))


def _gradient_linear_combination(
    gradients: Sequence[Tuple[float, Gradient]],
) -> Gradient:
    if not gradients:
        raise ValueError("At least one gradient is required")
    output = [torch.zeros_like(value) for value in gradients[0][1]]
    for weight, gradient in gradients:
        for target, value in zip(output, gradient):
            target.add_(value, alpha=float(weight))
    return output


def _gradient_scale(gradient: Gradient, scale: float) -> Gradient:
    return [value * float(scale) for value in gradient]


def _gradient_stats(
    primary: Gradient,
    auxiliary: Gradient,
    *,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, float | bool]:
    selected = list(indices) if indices is not None else list(range(len(primary)))
    dot = 0.0
    primary_squared = 0.0
    auxiliary_squared = 0.0
    for index in selected:
        left = primary[int(index)].detach().double()
        right = auxiliary[int(index)].detach().double()
        dot += float(torch.sum(left * right).item())
        primary_squared += float(torch.sum(left.square()).item())
        auxiliary_squared += float(torch.sum(right.square()).item())
    primary_norm = math.sqrt(max(0.0, primary_squared))
    auxiliary_norm = math.sqrt(max(0.0, auxiliary_squared))
    denominator = primary_norm * auxiliary_norm
    cosine = float(dot / denominator) if denominator > 0.0 else 0.0
    return {
        "dot": float(dot),
        "primary_norm": float(primary_norm),
        "auxiliary_norm": float(auxiliary_norm),
        "cosine": float(max(-1.0, min(1.0, cosine))),
        "conflict": bool(dot < 0.0),
    }


def project_conflicting_auxiliary_gradient(
    primary: Gradient,
    auxiliary: Gradient,
) -> Tuple[Gradient, Dict[str, float | bool]]:
    before = _gradient_stats(primary, auxiliary)
    primary_squared = float(before["primary_norm"]) ** 2
    if bool(before["conflict"]) and primary_squared > 0.0:
        coefficient = float(before["dot"]) / primary_squared
        projected = [
            auxiliary_value - coefficient * primary_value
            for primary_value, auxiliary_value in zip(primary, auxiliary)
        ]
    else:
        coefficient = 0.0
        projected = [value.clone() for value in auxiliary]
    after = _gradient_stats(primary, projected)
    retained = float(after["auxiliary_norm"]) / max(float(before["auxiliary_norm"]), 1e-12)
    summary = {
        **{f"raw_{key}": value for key, value in before.items()},
        "projection_coefficient": float(coefficient),
        "projected_dot": float(after["dot"]),
        "projected_cosine": float(after["cosine"]),
        "projected_auxiliary_norm": float(after["auxiliary_norm"]),
        "retained_norm_ratio": float(retained),
    }
    return projected, summary


def _measure_class_gradients(
    classifier: nn.Module,
    detector: nn.Module,
    *,
    classification_dataset: MangoYOLOCropDataset,
    detection_dataset: MangoYOLOCropDataset,
    rows: Sequence[PairedSource],
    classifier_parameters: Sequence[nn.Parameter],
    detector_parameters: Sequence[nn.Parameter],
    criterion: DETRSetCriterion,
    batch_size: int,
    device: torch.device,
) -> Tuple[GradientMap, Dict[str, float]]:
    accumulators = {
        key: _zero_gradient_like(
            classifier_parameters if key == "classification" else detector_parameters,
            device=device,
        )
        for key in (
            "classification",
            "det_class",
            "det_objectness",
            "det_bbox_l1",
            "det_bbox_giou",
        )
    }
    loss_sums = {key: 0.0 for key in accumulators}
    total_rows = 0
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = list(rows[start : start + int(batch_size)])
        row_count = len(batch_rows)
        images, labels, metadata = _stack_classification_batch(
            classification_dataset,
            batch_rows,
        )
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        metadata = {
            key: value.to(device=device, non_blocking=True)
            for key, value in metadata.items()
        }
        logits = _classification_logits(classifier, images, metadata)
        classification_loss = F.cross_entropy(logits, labels)
        classification_gradients = torch.autograd.grad(
            classification_loss,
            classifier_parameters,
            allow_unused=True,
        )
        _accumulate_gradient(
            accumulators["classification"],
            classification_gradients,
            weight=float(row_count),
        )
        loss_sums["classification"] += float(classification_loss.detach().cpu()) * row_count

        detection_indices = [row.detection_index for row in batch_rows]
        full_images, targets, image_masks = _stack_detection_batch(
            detection_dataset,
            detection_indices,
            device,
        )
        detection_output = detector(full_images, image_valid_mask=image_masks)
        component_losses = _detection_component_losses(
            criterion,
            detection_output,
            targets,
        )
        component_names = list(component_losses)
        for component_index, component_name in enumerate(component_names):
            gradients = torch.autograd.grad(
                component_losses[component_name],
                detector_parameters,
                retain_graph=component_index < len(component_names) - 1,
                allow_unused=True,
            )
            _accumulate_gradient(
                accumulators[component_name],
                gradients,
                weight=float(row_count),
            )
            loss_sums[component_name] += (
                float(component_losses[component_name].detach().cpu()) * row_count
            )
        total_rows += row_count

    if total_rows <= 0:
        raise ValueError("Gradient measurement received no rows")
    output: GradientMap = {}
    for key, gradient in accumulators.items():
        averaged = [value.div(float(total_rows)).detach().cpu() for value in gradient]
        output[key] = averaged
    output["det_localization"] = _gradient_linear_combination(
        (
            (DETECTION_BBOX_L1_WEIGHT, output["det_bbox_l1"]),
            (DETECTION_BBOX_GIOU_WEIGHT, output["det_bbox_giou"]),
        )
    )
    output["det_total"] = _gradient_linear_combination(
        (
            (DETECTION_CLASS_WEIGHT, output["det_class"]),
            (DETECTION_OBJECTNESS_WEIGHT, output["det_objectness"]),
            (1.0, output["det_localization"]),
        )
    )
    del output["det_bbox_l1"]
    del output["det_bbox_giou"]
    loss_means = {key: float(value / total_rows) for key, value in loss_sums.items()}
    loss_means["det_localization"] = (
        DETECTION_BBOX_L1_WEIGHT * loss_means["det_bbox_l1"]
        + DETECTION_BBOX_GIOU_WEIGHT * loss_means["det_bbox_giou"]
    )
    loss_means["det_total"] = (
        DETECTION_CLASS_WEIGHT * loss_means["det_class"]
        + DETECTION_OBJECTNESS_WEIGHT * loss_means["det_objectness"]
        + loss_means["det_localization"]
    )
    return output, loss_means


def _combine_class_gradients(
    per_class: Mapping[int, GradientMap],
    *,
    class_weights: Mapping[int, float],
) -> GradientMap:
    task_names = list(next(iter(per_class.values())).keys())
    return {
        task_name: _gradient_linear_combination(
            [
                (float(class_weights[class_index]), per_class[class_index][task_name])
                for class_index in sorted(per_class)
            ]
        )
        for task_name in task_names
    }


def _parameter_stage(name: str) -> str:
    if name.startswith("blocks."):
        block_index = int(name.split(".", 2)[1])
        if block_index < 2:
            return "blocks_1_2"
        if block_index < 5:
            return "blocks_3_5"
        return "blocks_6_8"
    if name.startswith("norm."):
        return "final_norm"
    return "stem_patch_tokens"


def _stage_indices(parameter_names: Sequence[str]) -> Dict[str, list[int]]:
    output: Dict[str, list[int]] = {}
    for index, name in enumerate(parameter_names):
        output.setdefault(_parameter_stage(name), []).append(int(index))
    return output


def _normalized_joint_directions(
    primary: Gradient,
    auxiliary: Gradient,
    *,
    objectness_auxiliary: Gradient,
    localization_auxiliary: Gradient,
    auxiliary_ratio: float,
) -> Tuple[Dict[str, Gradient], Dict[str, object]]:
    raw = _gradient_stats(primary, auxiliary)
    primary_norm = float(raw["primary_norm"])
    auxiliary_norm = float(raw["auxiliary_norm"])
    if primary_norm <= 0.0 or auxiliary_norm <= 0.0:
        raise ValueError("Primary and auxiliary gradients must be nonzero")
    scaled_auxiliary = _gradient_scale(auxiliary, primary_norm / auxiliary_norm)
    projected_auxiliary, projection = project_conflicting_auxiliary_gradient(
        primary,
        scaled_auxiliary,
    )
    naive = _gradient_linear_combination(
        ((1.0, primary), (float(auxiliary_ratio), scaled_auxiliary))
    )
    pcgrad = _gradient_linear_combination(
        ((1.0, primary), (float(auxiliary_ratio), projected_auxiliary))
    )
    component_projections: Dict[str, object] = {}
    projected_spatial_components: list[Gradient] = []
    for component_name, component_gradient in (
        ("objectness", objectness_auxiliary),
        ("localization", localization_auxiliary),
    ):
        component_norm = float(
            _gradient_stats(component_gradient, component_gradient)["primary_norm"]
        )
        if component_norm <= 0.0:
            raise ValueError(f"Zero spatial auxiliary gradient: {component_name}")
        scaled_component = _gradient_scale(
            component_gradient,
            primary_norm / component_norm,
        )
        projected_component, component_summary = project_conflicting_auxiliary_gradient(
            primary,
            scaled_component,
        )
        component_projections[component_name] = component_summary
        projected_spatial_components.append(projected_component)
    per_component_ratio = float(auxiliary_ratio) / float(len(projected_spatial_components))
    pcgrad_spatial = _gradient_linear_combination(
        [
            (1.0, primary),
            *[
                (per_component_ratio, component)
                for component in projected_spatial_components
            ],
        ]
    )
    naive_norm = float(_gradient_stats(naive, naive)["primary_norm"])
    pcgrad_norm = float(_gradient_stats(pcgrad, pcgrad)["primary_norm"])
    pcgrad_spatial_norm = float(
        _gradient_stats(pcgrad_spatial, pcgrad_spatial)["primary_norm"]
    )
    directions = {
        "classification_only": [value.clone() for value in primary],
        "naive_joint": _gradient_scale(naive, primary_norm / max(naive_norm, 1e-12)),
        "pcgrad_joint": _gradient_scale(pcgrad, primary_norm / max(pcgrad_norm, 1e-12)),
        "pcgrad_spatial": _gradient_scale(
            pcgrad_spatial,
            primary_norm / max(pcgrad_spatial_norm, 1e-12),
        ),
    }
    return directions, {
        "raw": raw,
        "projection": projection,
        "spatial_component_projection": component_projections,
        "spatial_candidate": {
            "components": ["objectness", "localization"],
            "component_gradient_norm_balancing": True,
            "per_component_ratio": per_component_ratio,
            "detector_class_gradient_excluded": True,
        },
        "auxiliary_scaled_to_primary_norm": True,
        "auxiliary_ratio": float(auxiliary_ratio),
        "all_direction_norms_equal_primary": True,
    }


def _classification_metrics_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    focus_class_index: int,
) -> Dict[str, float | int]:
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    predictions = logits.argmax(dim=1)
    probabilities = logits.softmax(dim=1)
    rows = int(labels.numel())
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, labels[:, None], float("-inf"))
    true_margin = true_logits - other_logits.max(dim=1).values
    focus = int(focus_class_index)
    positive = labels == focus
    negative = ~positive
    nonfocus_logits = logits.clone()
    nonfocus_logits[:, focus] = float("-inf")
    focus_positive_margin = logits[:, focus] - nonfocus_logits.max(dim=1).values
    focus_negative_margin = true_logits - logits[:, focus]
    return {
        "rows": rows,
        "loss": float(F.cross_entropy(logits, labels).item()),
        "accuracy": float((predictions == labels).float().mean().item()),
        "mean_true_margin": float(true_margin.mean().item()),
        "focus_positive_rows": int(positive.sum().item()),
        "focus_negative_rows": int(negative.sum().item()),
        "focus_positive_margin": (
            float(focus_positive_margin[positive].mean().item()) if positive.any() else float("nan")
        ),
        "focus_negative_margin": (
            float(focus_negative_margin[negative].mean().item()) if negative.any() else float("nan")
        ),
        "focus_recall": (
            float((predictions[positive] == focus).float().mean().item())
            if positive.any()
            else float("nan")
        ),
        "focus_false_positive_rate": (
            float((predictions[negative] == focus).float().mean().item())
            if negative.any()
            else float("nan")
        ),
        "focus_probability_positive": (
            float(probabilities[positive, focus].mean().item())
            if positive.any()
            else float("nan")
        ),
        "focus_probability_negative": (
            float(probabilities[negative, focus].mean().item())
            if negative.any()
            else float("nan")
        ),
    }


def _load_classification_batches(
    dataset: MangoYOLOCropDataset,
    rows: Sequence[PairedSource],
    *,
    batch_size: int,
) -> list[Tuple[Tensor, Tensor, Dict[str, Tensor]]]:
    return [
        _stack_classification_batch(dataset, rows[start : start + int(batch_size)])
        for start in range(0, len(rows), int(batch_size))
    ]


@torch.inference_mode()
def _evaluate_classification_batches(
    classifier: nn.Module,
    batches: Sequence[Tuple[Tensor, Tensor, Dict[str, Tensor]]],
    *,
    focus_class_index: int,
    device: torch.device,
) -> Dict[str, float | int]:
    classifier.eval()
    logits_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    for images, labels, metadata in batches:
        device_metadata = {
            key: value.to(device=device, non_blocking=True)
            for key, value in metadata.items()
        }
        logits_rows.append(
            _classification_logits(
                classifier,
                images.to(device=device, non_blocking=True),
                device_metadata,
            ).cpu()
        )
        label_rows.append(labels.cpu())
    return _classification_metrics_from_logits(
        torch.cat(logits_rows, dim=0),
        torch.cat(label_rows, dim=0),
        focus_class_index=focus_class_index,
    )


def _shared_parameter_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = sum(float(parameter.detach().double().square().sum().item()) for parameter in parameters)
    return math.sqrt(max(0.0, squared))


def _evaluate_local_influence(
    classifier: nn.Module,
    *,
    shared_parameters: Sequence[nn.Parameter],
    directions: Mapping[str, Gradient],
    heldout_batches: Sequence[Tuple[Tensor, Tensor, Dict[str, Tensor]]],
    step_parameter_ratio: float,
    focus_class_index: int,
    device: torch.device,
) -> Dict[str, Dict[str, float | int]]:
    originals = [parameter.detach().clone() for parameter in shared_parameters]
    parameter_norm = _shared_parameter_norm(shared_parameters)
    step_norm = float(step_parameter_ratio) * parameter_norm
    output = {
        "base": _evaluate_classification_batches(
            classifier,
            heldout_batches,
            focus_class_index=focus_class_index,
            device=device,
        )
    }
    try:
        for name, direction in directions.items():
            direction_norm = float(_gradient_stats(direction, direction)["primary_norm"])
            if direction_norm <= 0.0:
                raise ValueError(f"Zero update direction: {name}")
            with torch.no_grad():
                for parameter, original, gradient in zip(
                    shared_parameters,
                    originals,
                    direction,
                ):
                    delta = gradient.to(device=parameter.device, dtype=parameter.dtype)
                    parameter.copy_(original - delta * (step_norm / direction_norm))
            metrics = _evaluate_classification_batches(
                classifier,
                heldout_batches,
                focus_class_index=focus_class_index,
                device=device,
            )
            metrics["parameter_step_norm"] = float(step_norm)
            metrics["parameter_step_ratio"] = float(step_parameter_ratio)
            output[name] = metrics
    finally:
        with torch.no_grad():
            for parameter, original in zip(shared_parameters, originals):
                parameter.copy_(original)
    return output


def _mean_metric(
    fold_rows: Sequence[Mapping[str, object]],
    method: str,
    metric: str,
) -> float:
    return float(
        np.mean(
            [
                float(row["influence"][method][metric])
                for row in fold_rows
            ]
        )
    )


def assess_joint_gradient_readiness(
    *,
    fold_rows: Sequence[Mapping[str, object]],
    warmup_summary: Mapping[str, object],
    source_overlap: int,
    test_split_used: bool,
) -> Dict[str, object]:
    if not fold_rows:
        raise ValueError("fold_rows must not be empty")
    total_cosines = np.asarray(
        [float(row["natural_alignment"]["det_total"]["raw_cosine"]) for row in fold_rows],
        dtype=np.float64,
    )
    total_retained = np.asarray(
        [
            float(row["natural_alignment"]["det_total"]["retained_norm_ratio"])
            for row in fold_rows
        ],
        dtype=np.float64,
    )
    localization_retained = np.asarray(
        [
            float(row["natural_alignment"]["det_localization"]["retained_norm_ratio"])
            for row in fold_rows
        ],
        dtype=np.float64,
    )
    focus_retained = np.asarray(
        [float(row["focus_alignment"]["retained_norm_ratio"]) for row in fold_rows],
        dtype=np.float64,
    )
    pcgrad_loss = _mean_metric(fold_rows, "pcgrad_spatial", "loss")
    classification_loss = _mean_metric(fold_rows, "classification_only", "loss")
    naive_loss = _mean_metric(fold_rows, "naive_joint", "loss")
    total_pcgrad_loss = _mean_metric(fold_rows, "pcgrad_joint", "loss")
    pcgrad_positive_margin = _mean_metric(
        fold_rows,
        "pcgrad_spatial",
        "focus_positive_margin",
    )
    classification_positive_margin = _mean_metric(
        fold_rows,
        "classification_only",
        "focus_positive_margin",
    )
    pcgrad_negative_margin = _mean_metric(
        fold_rows,
        "pcgrad_spatial",
        "focus_negative_margin",
    )
    classification_negative_margin = _mean_metric(
        fold_rows,
        "classification_only",
        "focus_negative_margin",
    )
    pcgrad_true_margin = _mean_metric(
        fold_rows,
        "pcgrad_spatial",
        "mean_true_margin",
    )
    classification_true_margin = _mean_metric(
        fold_rows,
        "classification_only",
        "mean_true_margin",
    )
    nonworse_fold_count = int(
        sum(
            float(row["influence"]["pcgrad_spatial"]["loss"])
            <= float(row["influence"]["classification_only"]["loss"]) + 0.005
            for row in fold_rows
        )
    )
    thresholds = {
        "warmup_last_to_first_loss_ratio_max": 0.90,
        "raw_total_cosine_median_min": -0.35,
        "total_retained_norm_median_min": 0.70,
        "total_retained_norm_fold_min": 0.60,
        "total_retained_norm_required_folds": max(1, len(fold_rows) - 1),
        "localization_retained_norm_median_min": 0.65,
        "focus_retained_norm_median_min": 0.60,
        "pcgrad_loss_delta_vs_classification_max": 0.002,
        "pcgrad_loss_delta_vs_naive_max": 0.001,
        "pcgrad_loss_delta_vs_total_pcgrad_max": 0.001,
        "focus_positive_margin_delta_min": -0.01,
        "focus_negative_margin_delta_min": -0.01,
        "measurable_loss_gain_min": 0.0002,
        "measurable_focus_margin_gain_min": 0.002,
        "measurable_true_margin_gain_min": 0.001,
        "pcgrad_loss_nonworse_required_folds": max(1, len(fold_rows) - 2),
    }
    observed = {
        "warmup_last_to_first_loss_ratio": float(
            warmup_summary["last_to_first_loss_ratio"]
        ),
        "raw_total_cosine_median": float(np.median(total_cosines)),
        "total_retained_norm_median": float(np.median(total_retained)),
        "total_retained_norm_passing_folds": int(
            np.sum(total_retained >= thresholds["total_retained_norm_fold_min"])
        ),
        "localization_retained_norm_median": float(np.median(localization_retained)),
        "focus_retained_norm_median": float(np.median(focus_retained)),
        "pcgrad_loss": pcgrad_loss,
        "classification_only_loss": classification_loss,
        "naive_joint_loss": naive_loss,
        "total_pcgrad_loss": total_pcgrad_loss,
        "pcgrad_loss_delta_vs_classification": float(pcgrad_loss - classification_loss),
        "pcgrad_loss_delta_vs_naive": float(pcgrad_loss - naive_loss),
        "pcgrad_loss_delta_vs_total_pcgrad": float(pcgrad_loss - total_pcgrad_loss),
        "focus_positive_margin_delta_vs_classification": float(
            pcgrad_positive_margin - classification_positive_margin
        ),
        "focus_negative_margin_delta_vs_classification": float(
            pcgrad_negative_margin - classification_negative_margin
        ),
        "true_margin_delta_vs_classification": float(
            pcgrad_true_margin - classification_true_margin
        ),
        "pcgrad_loss_nonworse_folds": nonworse_fold_count,
        "source_overlap": int(source_overlap),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "train_only_no_test": not bool(test_split_used),
        "head_warmup_audit_source_disjoint": int(source_overlap) == 0,
        "head_warmup_loss_ratio_le_0p90": observed[
            "warmup_last_to_first_loss_ratio"
        ]
        <= thresholds["warmup_last_to_first_loss_ratio_max"],
        "raw_total_cosine_median_ge_minus_0p35": observed[
            "raw_total_cosine_median"
        ]
        >= thresholds["raw_total_cosine_median_min"],
        "total_projection_retained_median_ge_0p70": observed[
            "total_retained_norm_median"
        ]
        >= thresholds["total_retained_norm_median_min"],
        "total_projection_retained_4of5_ge_0p60": observed[
            "total_retained_norm_passing_folds"
        ]
        >= thresholds["total_retained_norm_required_folds"],
        "localization_projection_retained_median_ge_0p65": observed[
            "localization_retained_norm_median"
        ]
        >= thresholds["localization_retained_norm_median_min"],
        "focus_projection_retained_median_ge_0p60": observed[
            "focus_retained_norm_median"
        ]
        >= thresholds["focus_retained_norm_median_min"],
        "pcgrad_loss_delta_vs_classification_le_0p002": observed[
            "pcgrad_loss_delta_vs_classification"
        ]
        <= thresholds["pcgrad_loss_delta_vs_classification_max"],
        "pcgrad_loss_delta_vs_naive_le_0p001": observed[
            "pcgrad_loss_delta_vs_naive"
        ]
        <= thresholds["pcgrad_loss_delta_vs_naive_max"],
        "spatial_pcgrad_loss_delta_vs_total_pcgrad_le_0p001": observed[
            "pcgrad_loss_delta_vs_total_pcgrad"
        ]
        <= thresholds["pcgrad_loss_delta_vs_total_pcgrad_max"],
        "focus_positive_margin_delta_ge_minus_0p01": observed[
            "focus_positive_margin_delta_vs_classification"
        ]
        >= thresholds["focus_positive_margin_delta_min"],
        "focus_negative_margin_delta_ge_minus_0p01": observed[
            "focus_negative_margin_delta_vs_classification"
        ]
        >= thresholds["focus_negative_margin_delta_min"],
        "pcgrad_loss_nonworse_3of5_folds": observed[
            "pcgrad_loss_nonworse_folds"
        ]
        >= thresholds["pcgrad_loss_nonworse_required_folds"],
        "spatial_pcgrad_has_measurable_incremental_signal": bool(
            observed["pcgrad_loss_delta_vs_classification"]
            <= -thresholds["measurable_loss_gain_min"]
            or observed["focus_positive_margin_delta_vs_classification"]
            >= thresholds["measurable_focus_margin_gain_min"]
            or observed["focus_negative_margin_delta_vs_classification"]
            >= thresholds["measurable_focus_margin_gain_min"]
            or observed["true_margin_delta_vs_classification"]
            >= thresholds["measurable_true_margin_gain_min"]
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": thresholds,
        "observed": observed,
    }


def _verify_encoder_unchanged(
    reference: Mapping[str, Tensor],
    model: nn.Module,
    shared_names: Sequence[str],
) -> Dict[str, object]:
    state = model.state_dict()
    changed: list[str] = []
    maximum = 0.0
    squared = 0.0
    for name in shared_names:
        current = state[name].detach().cpu().float()
        expected = reference[name].detach().cpu().float()
        delta = current - expected
        local_maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
        if local_maximum > 0.0:
            changed.append(name)
        maximum = max(maximum, local_maximum)
        squared += float(delta.square().sum().item())
    return {
        "passed": not changed,
        "changed_parameter_count": int(len(changed)),
        "changed_parameters": changed,
        "max_absolute_delta": float(maximum),
        "l2_delta": float(math.sqrt(max(0.0, squared))),
    }


def _alignment_summary(primary: Gradient, auxiliary: Gradient) -> Dict[str, object]:
    _, summary = project_conflicting_auxiliary_gradient(primary, auxiliary)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _flatten_audit_rows(
    fold_rows: Sequence[Mapping[str, object]],
) -> Tuple[list[Dict[str, object]], list[Dict[str, object]], list[Dict[str, object]]]:
    alignment_rows: list[Dict[str, object]] = []
    stage_rows: list[Dict[str, object]] = []
    influence_rows: list[Dict[str, object]] = []
    for row in fold_rows:
        fold = int(row["fold"])
        for weighting in ("natural_alignment", "balanced_alignment"):
            for component, values in row[weighting].items():
                alignment_rows.append(
                    {
                        "fold": fold,
                        "weighting": weighting.replace("_alignment", ""),
                        "component": component,
                        **dict(values),
                    }
                )
        alignment_rows.append(
            {
                "fold": fold,
                "weighting": "focus_class",
                "component": "det_total",
                **dict(row["focus_alignment"]),
            }
        )
        for stage, values in row["stage_alignment"].items():
            stage_rows.append({"fold": fold, "stage": stage, **dict(values)})
        for method, values in row["influence"].items():
            influence_rows.append({"fold": fold, "method": method, **dict(values)})
    return alignment_rows, stage_rows, influence_rows


def _plot_warmup(curve: Sequence[Mapping[str, float]], output_path: Path) -> None:
    batches = [int(row["batch"]) for row in curve]
    losses = np.asarray([float(row["loss"]) for row in curve], dtype=np.float64)
    window = min(15, max(1, len(losses) // 4))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smooth = np.convolve(losses, kernel, mode="valid")
        smooth_batches = batches[window - 1 :]
    else:
        smooth = losses
        smooth_batches = batches
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(batches, losses, color="#8a8f98", alpha=0.38, linewidth=1.0, label="batch")
    axis.plot(smooth_batches, smooth, color="#176b87", linewidth=2.0, label="moving mean")
    axis.set_xlabel("Head warm-up batch")
    axis.set_ylabel("Detection loss")
    axis.set_title("Frozen-encoder detector-head warm-up")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_alignment(
    fold_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    components = ("det_class", "det_objectness", "det_localization", "det_total")
    matrix = np.asarray(
        [
            [
                float(row["natural_alignment"][component]["raw_cosine"])
                for component in components
            ]
            for row in fold_rows
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(components)), labels=components, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(fold_rows)), labels=[f"fold {index}" for index in range(len(fold_rows))])
    axis.set_title("Natural-prior cosine: classification vs detection components")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axis.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_influence(
    fold_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    methods = (
        "classification_only",
        "naive_joint",
        "pcgrad_joint",
        "pcgrad_spatial",
    )
    labels = ("CE loss", "class1 positive margin", "class1 negative margin")
    metrics = ("loss", "focus_positive_margin", "focus_negative_margin")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, metric, label in zip(axes, metrics, labels):
        values = np.asarray(
            [
                [float(row["influence"][method][metric]) for row in fold_rows]
                for method in methods
            ],
            dtype=np.float64,
        )
        means = values.mean(axis=1)
        deviations = values.std(axis=1)
        colors = ("#176b87", "#b35c1e", "#7b6a40", "#4f7b40")
        axis.bar(np.arange(len(methods)), means, yerr=deviations, color=colors, alpha=0.88)
        axis.set_xticks(
            np.arange(len(methods)),
            labels=("cls", "naive", "total-PCGrad", "spatial-PCGrad"),
            rotation=15,
        )
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("One-step held-out train influence (mean +/- fold standard deviation)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _aggregate_influence(fold_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    methods = (
        "base",
        "classification_only",
        "naive_joint",
        "pcgrad_joint",
        "pcgrad_spatial",
    )
    metrics = (
        "loss",
        "accuracy",
        "mean_true_margin",
        "focus_positive_margin",
        "focus_negative_margin",
        "focus_recall",
        "focus_false_positive_rate",
        "focus_probability_positive",
        "focus_probability_negative",
    )
    output: Dict[str, object] = {}
    for method in methods:
        output[method] = {
            metric: {
                "mean": _mean_metric(fold_rows, method, metric),
                "std": float(
                    np.std(
                        [
                            float(row["influence"][method][metric])
                            for row in fold_rows
                        ]
                    )
                ),
            }
            for metric in metrics
        }
    return output


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    start_time = time.perf_counter()
    if int(args.folds) < 2:
        raise ValueError("folds must be at least two")
    if int(args.samples_per_class_per_fold) < 4:
        raise ValueError("samples-per-class-per-fold must be at least four")
    if int(args.batch_size) <= 0 or int(args.head_warmup_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    if int(args.head_warmup_batches) < 20:
        raise ValueError("head-warmup-batches must be at least 20")
    if float(args.head_warmup_learning_rate) <= 0.0:
        raise ValueError("head-warmup-learning-rate must be positive")
    if float(args.step_parameter_ratio) <= 0.0:
        raise ValueError("step-parameter-ratio must be positive")
    if not 0.0 < float(args.auxiliary_gradient_ratio) <= 1.0:
        raise ValueError("auxiliary-gradient-ratio must be in (0, 1]")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = _resolve_device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    resolved_config = _load_resolved_config(checkpoint_path, args.resolved_config)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_count = len(class_names)
    focus_class_index = int(args.focus_class_index)
    if not 0 <= focus_class_index < class_count:
        raise ValueError("focus-class-index is outside the class range")

    classification_dataset, detection_dataset, view_protocol = _build_datasets(
        data_yaml=data_yaml,
        model_config=checkpoint["model_config"],
        resolved_config=resolved_config,
        class_count=class_count,
    )
    paired_sources = build_paired_single_object_sources(
        classification_dataset,
        detection_dataset,
    )
    class_counts = {
        class_index: int(sum(row.label == class_index for row in paired_sources))
        for class_index in range(class_count)
    }
    total_single_object = sum(class_counts.values())
    natural_class_weights = {
        class_index: float(class_counts[class_index] / total_single_object)
        for class_index in range(class_count)
    }
    balanced_class_weights = {
        class_index: float(1.0 / class_count) for class_index in range(class_count)
    }
    audit_folds = select_stratified_audit_folds(
        paired_sources,
        class_count=class_count,
        folds=int(args.folds),
        samples_per_class_per_fold=int(args.samples_per_class_per_fold),
        seed=int(args.seed),
    )
    audit_stems = {
        row.source_stem for fold_rows in audit_folds for row in fold_rows
    }
    warmup_indices = [
        index
        for index, sample in enumerate(detection_dataset.samples)
        if sample.image_path.stem.casefold() not in audit_stems
    ]
    warmup_stems = {
        detection_dataset.samples[index].image_path.stem.casefold()
        for index in warmup_indices
    }
    source_overlap = len(audit_stems.intersection(warmup_stems))

    classifier = build_model_from_checkpoint(checkpoint).to(device)
    detector, detector_summary = _build_detector(
        checkpoint,
        num_queries=int(args.num_queries),
        decoder_depth=int(args.decoder_depth),
        decoder_ffn_dim=int(args.decoder_ffn_dim),
        decoder_dropout=float(args.decoder_dropout),
        seed=int(args.seed),
    )
    detector = detector.to(device)
    shared_names = _shared_parameter_names(classifier, detector)
    shared_elements = int(
        sum(dict(classifier.named_parameters())[name].numel() for name in shared_names)
    )
    detector_encoder_reference = {
        name: detector.state_dict()[name].detach().cpu().clone()
        for name in shared_names
    }
    transfer_identity = _verify_encoder_unchanged(
        {
            name: classifier.state_dict()[name].detach().cpu().clone()
            for name in shared_names
        },
        detector,
        shared_names,
    )
    if not transfer_identity["passed"]:
        raise RuntimeError(f"Classifier/detector encoder transfer is not exact: {transfer_identity}")

    warmup_curve, warmup_summary = _warm_up_detector_heads(
        detector,
        dataset=detection_dataset,
        allowed_indices=warmup_indices,
        shared_names=shared_names,
        class_count=class_count,
        batch_size=int(args.head_warmup_batch_size),
        max_batches=int(args.head_warmup_batches),
        learning_rate=float(args.head_warmup_learning_rate),
        weight_decay=float(args.head_warmup_weight_decay),
        seed=int(args.seed) + 1000,
        device=device,
    )
    encoder_integrity = _verify_encoder_unchanged(
        detector_encoder_reference,
        detector,
        shared_names,
    )
    if not encoder_integrity["passed"]:
        raise RuntimeError(f"Detector head warm-up modified the encoder: {encoder_integrity}")

    classifier_parameters, detector_parameters = _set_gradient_measurement_mode(
        classifier,
        detector,
        shared_names,
    )
    criterion = _detection_criterion(class_count).to(device)
    stages = _stage_indices(shared_names)
    fold_summaries: list[Dict[str, object]] = []

    for fold_index, fit_rows in enumerate(audit_folds):
        print(
            {
                "gradient_audit": {
                    "fold": fold_index,
                    "fit_rows": len(fit_rows),
                    "heldout_fold": (fold_index + 1) % len(audit_folds),
                }
            },
            flush=True,
        )
        per_class_gradients: Dict[int, GradientMap] = {}
        per_class_losses: Dict[str, object] = {}
        class_alignment: Dict[str, object] = {}
        for class_index in range(class_count):
            class_rows = [row for row in fit_rows if row.label == class_index]
            gradients, losses = _measure_class_gradients(
                classifier,
                detector,
                classification_dataset=classification_dataset,
                detection_dataset=detection_dataset,
                rows=class_rows,
                classifier_parameters=classifier_parameters,
                detector_parameters=detector_parameters,
                criterion=criterion,
                batch_size=int(args.batch_size),
                device=device,
            )
            per_class_gradients[class_index] = gradients
            per_class_losses[str(class_index)] = losses
            class_alignment[str(class_index)] = _alignment_summary(
                gradients["classification"],
                gradients["det_total"],
            )

        natural_gradients = _combine_class_gradients(
            per_class_gradients,
            class_weights=natural_class_weights,
        )
        balanced_gradients = _combine_class_gradients(
            per_class_gradients,
            class_weights=balanced_class_weights,
        )
        components = (
            "det_class",
            "det_objectness",
            "det_localization",
            "det_total",
        )
        natural_alignment = {
            component: _alignment_summary(
                natural_gradients["classification"],
                natural_gradients[component],
            )
            for component in components
        }
        balanced_alignment = {
            component: _alignment_summary(
                balanced_gradients["classification"],
                balanced_gradients[component],
            )
            for component in components
        }
        stage_alignment = {
            stage: _gradient_stats(
                natural_gradients["classification"],
                natural_gradients["det_total"],
                indices=indices,
            )
            for stage, indices in stages.items()
        }
        directions, direction_summary = _normalized_joint_directions(
            natural_gradients["classification"],
            natural_gradients["det_total"],
            objectness_auxiliary=natural_gradients["det_objectness"],
            localization_auxiliary=natural_gradients["det_localization"],
            auxiliary_ratio=float(args.auxiliary_gradient_ratio),
        )
        heldout_rows = audit_folds[(fold_index + 1) % len(audit_folds)]
        heldout_batches = _load_classification_batches(
            classification_dataset,
            heldout_rows,
            batch_size=int(args.batch_size),
        )
        influence = _evaluate_local_influence(
            classifier,
            shared_parameters=classifier_parameters,
            directions=directions,
            heldout_batches=heldout_batches,
            step_parameter_ratio=float(args.step_parameter_ratio),
            focus_class_index=focus_class_index,
            device=device,
        )
        fold_summary = {
            "fold": int(fold_index),
            "fit_source_count": int(len(fit_rows)),
            "heldout_fold": int((fold_index + 1) % len(audit_folds)),
            "heldout_source_count": int(len(heldout_rows)),
            "fit_heldout_source_overlap": int(
                len(
                    {row.source_stem for row in fit_rows}.intersection(
                        row.source_stem for row in heldout_rows
                    )
                )
            ),
            "per_class_losses": per_class_losses,
            "class_alignment": class_alignment,
            "focus_alignment": class_alignment[str(focus_class_index)],
            "natural_alignment": natural_alignment,
            "balanced_alignment": balanced_alignment,
            "stage_alignment": stage_alignment,
            "direction_summary": direction_summary,
            "influence": influence,
        }
        fold_summaries.append(fold_summary)
        del per_class_gradients, natural_gradients, balanced_gradients, directions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gate = assess_joint_gradient_readiness(
        fold_rows=fold_summaries,
        warmup_summary=warmup_summary,
        source_overlap=source_overlap,
        test_split_used=False,
    )
    alignment_rows, stage_rows, influence_rows = _flatten_audit_rows(fold_summaries)
    pair_rows = [
        {
            "fold": int(fold_index),
            "source_stem": row.source_stem,
            "label": int(row.label),
            "class_name": class_names[int(row.label)],
            "classification_index": int(row.classification_index),
            "detection_index": int(row.detection_index),
        }
        for fold_index, rows in enumerate(audit_folds)
        for row in rows
    ]
    protocol = {
        "method": "classification_anchored_joint_localization_gradient_readiness",
        "seed": int(args.seed),
        "folds": int(args.folds),
        "samples_per_class_per_fold": int(args.samples_per_class_per_fold),
        "gradient_batch_size": int(args.batch_size),
        "head_warmup": {
            "batch_size": int(args.head_warmup_batch_size),
            "batches": int(args.head_warmup_batches),
            "learning_rate": float(args.head_warmup_learning_rate),
            "weight_decay": float(args.head_warmup_weight_decay),
            "encoder_frozen": True,
            "audit_sources_excluded": True,
        },
        "detection_loss_weights": {
            "classification": DETECTION_CLASS_WEIGHT,
            "objectness": DETECTION_OBJECTNESS_WEIGHT,
            "bbox_l1": DETECTION_BBOX_L1_WEIGHT,
            "bbox_giou": DETECTION_BBOX_GIOU_WEIGHT,
        },
        "gradient_weighting": {
            "primary": "natural single-object source class frequency",
            "stress_test": "balanced classes",
            "natural_class_weights": {
                str(key): value for key, value in natural_class_weights.items()
            },
        },
        "pcgrad": {
            "primary_task": "keeper object-crop classification",
            "auxiliary_task": "full-frame DETR class/objectness/localization",
            "primary_gradient_unchanged": True,
            "auxiliary_scaled_to_primary_norm": True,
            "auxiliary_ratio": float(args.auxiliary_gradient_ratio),
            "smoke_candidate": "component-normalized spatial PCGrad",
            "spatial_components": ["objectness", "localization"],
            "spatial_component_ratio": float(args.auxiliary_gradient_ratio) / 2.0,
            "detector_class_gradient_in_candidate": False,
        },
        "local_influence": {
            "heldout_source_fold": "next cyclic audit fold",
            "all_direction_norms_equal": True,
            "parameter_step_ratio": float(args.step_parameter_ratio),
            "updated_parameters": "shared keeper encoder only",
        },
        "view_protocol": view_protocol,
        "validation_read": False,
        "test_read": False,
        "raw_dataset_touched": False,
        "model_written": False,
    }
    summary = {
        "protocol": protocol,
        "checkpoint": str(checkpoint_path),
        "resolved_config": str(resolved_config["_path"]),
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "focus_class_index": focus_class_index,
        "source_audit": {
            "single_object_sources": int(len(paired_sources)),
            "single_object_class_counts": {
                str(key): value for key, value in class_counts.items()
            },
            "audit_sources": int(len(audit_stems)),
            "warmup_eligible_sources": int(len(warmup_stems)),
            "warmup_audit_source_overlap": int(source_overlap),
            "all_fit_heldout_overlaps_zero": all(
                int(row["fit_heldout_source_overlap"]) == 0 for row in fold_summaries
            ),
        },
        "model_audit": {
            "detector": detector_summary,
            "shared_parameter_count": int(len(shared_names)),
            "shared_parameter_elements": shared_elements,
            "shared_prefixes": list(ENCODER_TRANSFER_PREFIXES),
            "stage_parameter_counts": {
                stage: int(len(indices)) for stage, indices in stages.items()
            },
            "classifier_detector_transfer_identity": transfer_identity,
            "encoder_integrity_after_head_warmup": encoder_integrity,
        },
        "head_warmup": warmup_summary,
        "folds": fold_summaries,
        "aggregate_influence": _aggregate_influence(fold_summaries),
        "gate": gate,
        "literature": list(LITERATURE),
        "raw_dataset_touched": False,
        "validation_split_used": False,
        "test_split_used": False,
        "image_model_trained": False,
        "detector_head_warmup_only": True,
        "checkpoint_written": False,
        "model_written": False,
        "trainable_manifest_written": False,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }

    _write_csv(output_dir / "audit_pairs.csv", pair_rows)
    _write_csv(output_dir / "detector_head_warmup_curve.csv", warmup_curve)
    _write_csv(output_dir / "gradient_alignment_by_fold.csv", alignment_rows)
    _write_csv(output_dir / "gradient_alignment_by_stage.csv", stage_rows)
    _write_csv(output_dir / "heldout_train_influence.csv", influence_rows)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_warmup(warmup_curve, output_dir / "detector_head_warmup.png")
    _plot_alignment(fold_summaries, output_dir / "gradient_alignment.png")
    _plot_influence(fold_summaries, output_dir / "heldout_train_influence.png")
    conclusion = (
        "The fixed train-only gate permits a joint-training smoke."
        if gate["smoke_permission"]
        else "The fixed train-only gate rejects joint-training smoke on the current keeper."
    )
    readme = [
        "# Classification-Anchored Joint Localization Gradient Readiness",
        "",
        f"- Keeper: `{checkpoint_path}`",
        f"- Train-only audit sources: `{len(audit_stems)}`",
        f"- Detector-head warm-up frames: `{warmup_summary['frames_seen']}`",
        f"- Warm-up last/first loss ratio: `{float(warmup_summary['last_to_first_loss_ratio']):.6f}`",
        f"- Median raw total-gradient cosine: `{float(gate['observed']['raw_total_cosine_median']):.6f}`",
        f"- Median projected auxiliary norm retained: `{float(gate['observed']['total_retained_norm_median']):.6f}`",
        f"- Spatial-PCGrad CE delta versus classification-only: `{float(gate['observed']['pcgrad_loss_delta_vs_classification']):.6f}`",
        f"- Class-1 positive-margin delta: `{float(gate['observed']['focus_positive_margin_delta_vs_classification']):.6f}`",
        f"- Class-1 negative-margin delta: `{float(gate['observed']['focus_negative_margin_delta_vs_classification']):.6f}`",
        f"- Smoke permission: `{str(bool(gate['smoke_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        conclusion,
        "No validation/test row was read, the raw dataset was not modified, and no model/checkpoint was written.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    summary["artifact_manifest_path"] = str(
        (output_dir / "artifact_manifest.json").resolve()
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    manifest = _write_artifact_manifest(
        output_dir,
        mode="joint_cls_localization_gradient_readiness_evidence_manifest",
    )
    print(
        json.dumps(
            {
                "head_warmup": warmup_summary,
                "gate": gate,
                "artifact_manifest": {
                    "payload_count": manifest["payload_count"],
                    "payload_size_bytes": manifest["payload_size_bytes"],
                    "payload_manifest_sha256": manifest["payload_manifest_sha256"],
                },
                "elapsed_seconds": summary["elapsed_seconds"],
                "cuda_peak_memory_mb": summary["cuda_peak_memory_mb"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
