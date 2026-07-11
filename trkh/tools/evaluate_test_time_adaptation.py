from __future__ import annotations

import argparse
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    maybe_enable_dataset_image_cache,
    save_checkpoint,
    set_seed,
)
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.evaluate import (
    DETECTION_MODEL_TYPES,
    _is_detection_targets,
    _move_targets_to_device,
    _stack_image_masks_from_targets,
    evaluate_model,
    resolve_classification_object_crops,
    resolve_crop_to_primary_object,
    save_evaluation_artifacts,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_detection_from_model_output,
)
from trkh.training.loss import HybridDetectionClassificationLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint after label-free entropy-based test-time "
            "adaptation on the requested split."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument(
        "--bbox-token-prior-source",
        choices=("bbox", "crop_bbox"),
        default="crop_bbox",
    )
    parser.add_argument(
        "--adapt-parameters",
        choices=("layernorm", "layernorm_head_bias", "layernorm_head"),
        default="layernorm_head_bias",
    )
    parser.add_argument("--adapt-lr", type=float, default=1e-5)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapt-steps", type=int, default=1)
    parser.add_argument("--selection-fraction", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--max-selected-per-batch", type=int, default=0)
    parser.add_argument(
        "--save-adapted-checkpoint",
        action="store_true",
        default=False,
        help="Save adapted_checkpoint.pt in output-dir for downstream XAI/audit.",
    )
    return parser.parse_args()


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)


def reliable_entropy_mask(
    logits: torch.Tensor,
    *,
    selection_fraction: float,
    min_confidence: float,
    max_selected: int = 0,
) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    candidate = confidence >= float(min_confidence)
    if not bool(candidate.any()):
        return torch.zeros_like(candidate, dtype=torch.bool)
    candidate_indices = candidate.nonzero(as_tuple=False).flatten()
    candidate_entropy = entropy.index_select(0, candidate_indices)
    keep_count = int(round(float(selection_fraction) * float(candidate_indices.numel())))
    keep_count = max(1, min(int(candidate_indices.numel()), keep_count))
    if int(max_selected) > 0:
        keep_count = min(keep_count, int(max_selected))
    order = torch.argsort(candidate_entropy, descending=False)
    selected = candidate_indices.index_select(0, order[:keep_count])
    mask = torch.zeros_like(candidate, dtype=torch.bool)
    mask[selected] = True
    return mask


def collect_adaptation_parameters(
    model: nn.Module,
    mode: str,
) -> Tuple[List[nn.Parameter], List[str]]:
    mode = str(mode).strip().lower()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    selected: List[Tuple[str, nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for local_name, parameter in module.named_parameters(recurse=False):
                selected.append((f"{module_name}.{local_name}".lstrip("."), parameter))

    if mode in {"layernorm_head_bias", "layernorm_head"}:
        for name, parameter in model.named_parameters():
            normalized = name.lower()
            is_head = normalized == "head.weight" or normalized == "head.bias"
            if not is_head:
                continue
            if mode == "layernorm_head_bias" and not normalized.endswith(".bias"):
                continue
            selected.append((name, parameter))

    unique: Dict[int, Tuple[str, nn.Parameter]] = {}
    for name, parameter in selected:
        unique[id(parameter)] = (name, parameter)

    names = []
    parameters = []
    for name, parameter in unique.values():
        parameter.requires_grad_(True)
        names.append(name)
        parameters.append(parameter)
    if not parameters:
        raise ValueError(f"No adaptation parameters selected for mode={mode!r}.")
    return parameters, names


def _prepare_eval_dataset(
    *,
    checkpoint: Dict[str, object],
    data_spec,
    split: str,
    override_image_size: Optional[int],
    checkpoint_detection_mode: bool,
):
    image_size = int(override_image_size or checkpoint.get("model_config", {}).get("image_size", 224))
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}
    source_context_feature_fusion = bool(model_config.get("source_context_feature_fusion", False))
    source_context_aux_for_eval = bool(
        source_context_feature_fusion
        and not checkpoint_detection_mode
        and bool(augmentation_config.get("classification_source_context_aux", False))
    )
    eval_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=augmentation_config.get("resize_mode", "pad"),
        illumination_normalization=bool(augmentation_config.get("illumination_normalization", False)),
        illumination_normalization_strength=float(
            augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
        ),
        foreground_crop_mode=str(augmentation_config.get("foreground_crop_mode", "none") or "none"),
        foreground_crop_margin_ratio=float(augmentation_config.get("foreground_crop_margin_ratio", 0.08) or 0.08),
        foreground_crop_min_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation_config.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(augmentation_config.get("background_suppression_mode", "none") or "none"),
        background_suppression_margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        background_suppression_blur_radius=float(augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0),
        surface_detail_amplification_mode=str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        surface_detail_amplification_strength=float(
            augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85) or 0.85
        ),
        eval_surface_detail_amplification=bool(augmentation_config.get("eval_surface_detail_amplification", False)),
    )
    if data_spec.data_format == "classification_folder":
        if checkpoint_detection_mode:
            raise ValueError("format=classification_folder does not support detection checkpoints.")
        return ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=eval_transform,
            class_aware_augmentation=False,
        )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=eval_transform,
        crop_margin_ratio=float(augmentation_config.get("crop_margin_ratio", 0.05)),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=not checkpoint_detection_mode,
        classification_object_crops=resolve_classification_object_crops(checkpoint),
        classification_bbox_metadata=not checkpoint_detection_mode,
        classification_source_context_aux=source_context_aux_for_eval,
        classification_source_context_mode=str(
            augmentation_config.get("classification_source_context_mode", "desaturate_blur") or "desaturate_blur"
        ),
        classification_source_context_margin_ratio=float(
            augmentation_config.get("classification_source_context_margin_ratio", 0.12) or 0.12
        ),
        classification_source_context_background_alpha=float(
            augmentation_config.get("classification_source_context_background_alpha", 0.35) or 0.35
        ),
        classification_source_context_blur_radius=float(
            augmentation_config.get("classification_source_context_blur_radius", 7.0) or 7.0
        ),
    )


def _unpack_batch(batch, device: torch.device, bbox_token_prior_source: str):
    bbox_metadata = None
    crop_boxes = None
    detection_targets = None
    targets = None
    if len(batch) == 2 and _is_detection_targets(batch[1]):
        images, batch_targets = batch
        labels = None
        detection_targets = _move_targets_to_device(batch_targets, device)
    elif len(batch) == 3:
        images, labels, targets = batch
        bbox_metadata = targets.get("bbox") if isinstance(targets, dict) else None
        crop_boxes = targets.get("crop_bbox") if isinstance(targets, dict) else None
    elif len(batch) == 2:
        images, labels = batch
    else:
        raise ValueError("Eval dataloader must return 2 or 3 items.")

    images = images.to(device, non_blocking=True)
    if torch.is_tensor(bbox_metadata):
        bbox_metadata = bbox_metadata.to(device, dtype=torch.float32, non_blocking=True)
    else:
        bbox_metadata = None
    if torch.is_tensor(crop_boxes):
        crop_boxes = crop_boxes.to(device, dtype=torch.float32, non_blocking=True)
    else:
        crop_boxes = None
    bbox_prior = crop_boxes if bbox_token_prior_source == "crop_bbox" and torch.is_tensor(crop_boxes) else bbox_metadata
    image_valid_mask = _stack_image_masks_from_targets(targets if len(batch) == 3 else detection_targets)
    if image_valid_mask is not None:
        image_valid_mask = image_valid_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    return images, bbox_metadata, bbox_prior, image_valid_mask


def _forward_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    image_valid_mask: Optional[torch.Tensor],
    bbox_metadata: Optional[torch.Tensor],
    bbox_token_prior: Optional[torch.Tensor],
):
    if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
        features = model.forward_features(
            images,
            image_valid_mask=image_valid_mask,
            bbox_token_prior=bbox_token_prior,
        )
        if bbox_metadata is not None:
            features["bbox"] = bbox_metadata
        output = model.forward_heads(features)
    elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
        features = model.forward_features(
            images,
            image_valid_mask=image_valid_mask,
            bbox_token_prior=bbox_token_prior,
        )
        if bbox_metadata is not None:
            features["bbox"] = bbox_metadata
        output = classification_logits_from_features(model, features)
    else:
        output = model(images)
    logits, _, _ = extract_detection_from_model_output(output)
    return logits


def adapt_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp: bool,
    bbox_token_prior_source: str,
    parameter_mode: str,
    lr: float,
    weight_decay: float,
    steps: int,
    selection_fraction: float,
    min_confidence: float,
    max_batches: Optional[int],
    max_selected_per_batch: int,
) -> Dict[str, object]:
    parameters, parameter_names = collect_adaptation_parameters(model, parameter_mode)
    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    model.eval()
    total_seen = 0
    total_selected = 0
    total_steps = 0
    loss_sum = 0.0
    batch_count = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    iterator = dataloader if max_batches is None else islice(dataloader, batch_count)
    started = time.perf_counter()

    with tqdm(iterator, desc="TTA adapt", leave=False, total=batch_count, dynamic_ncols=True) as pbar:
        for batch in pbar:
            images, bbox_metadata, bbox_prior, image_valid_mask = _unpack_batch(
                batch,
                device,
                bbox_token_prior_source,
            )
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp):
                    logits = _forward_logits(
                        model,
                        images,
                        image_valid_mask=image_valid_mask,
                        bbox_metadata=bbox_metadata,
                        bbox_token_prior=bbox_prior,
                    )
                    mask = reliable_entropy_mask(
                        logits,
                        selection_fraction=selection_fraction,
                        min_confidence=min_confidence,
                        max_selected=max_selected_per_batch,
                    )
                    if not bool(mask.any()):
                        loss = logits.sum() * 0.0
                    else:
                        entropy = prediction_entropy(logits.index_select(0, mask.nonzero(as_tuple=False).flatten()))
                        loss = entropy.mean()
                if bool(mask.any()):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                    optimizer.step()
                    total_steps += 1
                    loss_sum += float(loss.detach().cpu().item())
                selected = int(mask.sum().detach().cpu().item())
                total_seen += int(images.size(0))
                total_selected += selected
                pbar.set_postfix(selected=selected, loss=float(loss.detach().cpu().item()))

    return {
        "parameter_mode": str(parameter_mode),
        "trainable_parameter_names": parameter_names,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "steps_per_batch": int(steps),
        "selection_fraction": float(selection_fraction),
        "min_confidence": float(min_confidence),
        "max_selected_per_batch": int(max_selected_per_batch),
        "seen_samples": int(total_seen),
        "selected_samples": int(total_selected),
        "selected_fraction": float(total_selected / max(1, total_seen)),
        "optimizer_steps": int(total_steps),
        "mean_entropy_loss": float(loss_sum / max(1, total_steps)),
        "adapt_seconds": float(time.perf_counter() - started),
    }


def build_adapted_checkpoint_payload(
    *,
    source_checkpoint: Dict[str, object],
    model: nn.Module,
    adapt_summary: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    payload: Dict[str, object] = dict(source_checkpoint)
    payload["model_state"] = {
        str(name): tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload["test_time_adaptation"] = to_serializable(adapt_summary)
    payload["test_time_adaptation_source_checkpoint"] = str(args.checkpoint.resolve())
    payload["test_time_adaptation_split"] = str(args.split)
    payload["test_time_adaptation_data"] = str(args.data.resolve())
    payload["test_time_adaptation_note"] = (
        "Label-free entropy adaptation artifact for audit; not a training checkpoint."
    )
    for stale_key in (
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "ema_model_state",
        "ema_updates",
        "ema_decay",
        "train_model_state",
    ):
        payload.pop(stale_key, None)
    return payload


def main() -> None:
    args = parse_args()
    if args.adapt_lr <= 0.0:
        raise ValueError("--adapt-lr must be > 0.")
    if args.adapt_weight_decay < 0.0:
        raise ValueError("--adapt-weight-decay must be >= 0.")
    if args.adapt_steps < 1:
        raise ValueError("--adapt-steps must be >= 1.")
    if not 0.0 < args.selection_fraction <= 1.0:
        raise ValueError("--selection-fraction must be in (0, 1].")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1].")

    set_seed(args.seed, deterministic=False)
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        num_classes=len(class_names),
        override_image_size=args.override_image_size,
    )
    model.to(device)
    checkpoint_model_type = str(
        getattr(model, "model_type", checkpoint.get("model_config", {}).get("model_type", ""))
    ).strip().lower()
    checkpoint_detection_mode = checkpoint_model_type in DETECTION_MODEL_TYPES
    dataset = _prepare_eval_dataset(
        checkpoint=checkpoint,
        data_spec=data_spec,
        split=args.split,
        override_image_size=args.override_image_size,
        checkpoint_detection_mode=checkpoint_detection_mode,
    )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"tta_adapt_{args.split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context=f"tta_adapt_{args.split}",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    ensure_dir(args.output_dir)
    json_dump(
        args.output_dir / "setup.json",
        to_serializable(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "data": str(args.data.resolve()),
                "split": args.split,
                "device": str(device),
                "dataloader": dataloader_summary,
                "image_cache": cache_summary,
                "batch_size": int(args.batch_size),
                "max_batches": int(args.max_batches),
            }
        ),
    )
    adapt_summary = adapt_model(
        model=model,
        dataloader=loader,
        device=device,
        amp=bool(args.amp),
        bbox_token_prior_source=str(args.bbox_token_prior_source),
        parameter_mode=str(args.adapt_parameters),
        lr=float(args.adapt_lr),
        weight_decay=float(args.adapt_weight_decay),
        steps=int(args.adapt_steps),
        selection_fraction=float(args.selection_fraction),
        min_confidence=float(args.min_confidence),
        max_batches=(args.max_batches or None),
        max_selected_per_batch=int(args.max_selected_per_batch),
    )
    json_dump(args.output_dir / "adaptation_summary.json", to_serializable(adapt_summary))
    if bool(args.save_adapted_checkpoint):
        save_checkpoint(
            args.output_dir / "adapted_checkpoint.pt",
            build_adapted_checkpoint_payload(
                source_checkpoint=checkpoint,
                model=model,
                adapt_summary=adapt_summary,
                args=args,
            ),
        )
    criterion: nn.Module
    if checkpoint_detection_mode:
        criterion = HybridDetectionClassificationLoss(
            num_classes=data_spec.num_classes,
            label_smoothing=float(checkpoint.get("train_config", {}).get("label_smoothing", 0.0)),
            cls_weight=float(checkpoint.get("train_config", {}).get("cls_loss_weight", 1.0)),
            bbox_l1_weight=float(checkpoint.get("train_config", {}).get("bbox_l1_loss_weight", 1.0)),
            bbox_giou_weight=float(checkpoint.get("train_config", {}).get("bbox_giou_loss_weight", 0.5)),
            background_weight=float(checkpoint.get("train_config", {}).get("background_loss_weight", 0.3)),
            objectness_weight=float(checkpoint.get("train_config", {}).get("objectness_loss_weight", 5.0)),
            objectness_focal_alpha=float(checkpoint.get("train_config", {}).get("objectness_focal_alpha", 0.75)),
            objectness_focal_gamma=float(checkpoint.get("train_config", {}).get("objectness_focal_gamma", 0.5)),
            matcher_class_cost=float(checkpoint.get("train_config", {}).get("matcher_class_cost", 1.0)),
            matcher_objectness_cost=float(checkpoint.get("train_config", {}).get("matcher_objectness_cost", 1.0)),
            cardinality_weight=float(checkpoint.get("train_config", {}).get("cardinality_loss_weight", 0.0)),
            count_weight=float(checkpoint.get("train_config", {}).get("count_loss_weight", 0.0)),
        )
    else:
        criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
        amp=bool(args.amp),
        max_batches=args.max_batches or None,
        collect_artifact_stats=True,
        collect_prediction_records=True,
        bbox_token_prior_source=str(args.bbox_token_prior_source),
    )
    metrics["test_time_adaptation"] = adapt_summary
    save_evaluation_artifacts(metrics, class_names, args.output_dir)


if __name__ == "__main__":
    main()
