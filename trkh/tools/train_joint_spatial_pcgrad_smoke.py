from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.data.dataset import (
    MangoYOLOCropDataset,
    build_eval_transform,
    build_train_transform,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
)
from trkh.tools.audit_joint_cls_localization_gradient_readiness import (
    DETECTION_BBOX_GIOU_WEIGHT,
    DETECTION_BBOX_L1_WEIGHT,
    DETECTION_CLASS_WEIGHT,
    DETECTION_OBJECTNESS_WEIGHT,
    DETECTOR_PARAMETER_PREFIXES,
    PairedSource,
    _build_detector,
    _build_datasets as _build_readiness_datasets,
    _detection_component_losses,
    _detection_criterion,
    _load_checkpoint,
    _load_resolved_config,
    _resolve_device,
    _shared_parameter_names,
    _stack_classification_batch,
    _stack_detection_batch,
    _verify_encoder_unchanged,
    _warm_up_detector_heads,
    build_paired_single_object_sources,
)
from trkh.tools.build_detection_classification_curriculum_checkpoint import (
    fresh_checkpoint_payload,
)
from trkh.training.losses import LDAMFocalLoss
from trkh.training.train import (
    _pairwise_margin_loss_from_features,
    _teacher_focus_binary_loss_from_logits,
)


SEED = 20260712
MAX_TRAIN_BATCHES = 120
BATCH_SIZE = 4
SAMPLES_PER_CLASS = 12
STEP_PARAMETER_RATIO = 1e-4
HEAD_WARMUP_BATCHES = 120
HEAD_WARMUP_BATCH_SIZE = 8
PRIMARY_LEARNING_RATE = 1e-5
DETECTOR_HEAD_LEARNING_RATE = 2e-4
PRIMARY_WEIGHT_DECAY = 0.05
DETECTOR_HEAD_WEIGHT_DECAY = 1e-4
AUXILIARY_GRADIENT_RATIO = 0.25
GRAD_CLIP_NORM = 1.0
FOCUS_CLASS_INDEX = 1
KEEPER_RELOAD_MACRO_F1 = 0.882925
KEEPER_RELOAD_FOCUS_F1 = 0.678261
FOCUS_MILESTONE = 0.70


Gradient = list[Tensor]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-control, train-only joint spatial-PCGrad smoke from a keeper. "
            "It evaluates full validation after training and never reads test."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--update-mode",
        choices=("normalized_macrostep", "adamw_microbatch"),
        default="normalized_macrostep",
    )
    parser.add_argument(
        "--allow-rejected-adamw-ablation",
        action="store_true",
        help=(
            "Explicitly unlock the rejected microbatch-AdamW reproduction path. "
            "It is not a valid promotion protocol."
        ),
    )
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument(
        "--step-parameter-ratio",
        type=float,
        default=STEP_PARAMETER_RATIO,
    )
    parser.add_argument("--max-train-batches", type=int, default=MAX_TRAIN_BATCHES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--head-warmup-batches", type=int, default=HEAD_WARMUP_BATCHES)
    parser.add_argument("--head-warmup-batch-size", type=int, default=HEAD_WARMUP_BATCH_SIZE)
    parser.add_argument("--primary-learning-rate", type=float, default=PRIMARY_LEARNING_RATE)
    parser.add_argument(
        "--detector-head-learning-rate",
        type=float,
        default=DETECTOR_HEAD_LEARNING_RATE,
    )
    parser.add_argument("--primary-weight-decay", type=float, default=PRIMARY_WEIGHT_DECAY)
    parser.add_argument(
        "--detector-head-weight-decay",
        type=float,
        default=DETECTOR_HEAD_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--auxiliary-gradient-ratio",
        type=float,
        default=AUXILIARY_GRADIENT_RATIO,
    )
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-ffn-dim", type=int, default=512)
    parser.add_argument("--decoder-dropout", type=float, default=0.10)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _joint_model_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    num_queries: int,
    decoder_depth: int,
    decoder_ffn_dim: int,
    decoder_dropout: float,
    seed: int,
) -> Tuple[nn.Module, Dict[str, object]]:
    classifier = build_model_from_checkpoint(dict(checkpoint))
    detector, summary = _build_detector(
        checkpoint,
        num_queries=num_queries,
        decoder_depth=decoder_depth,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_dropout=decoder_dropout,
        seed=seed,
    )
    detector.head = copy.deepcopy(classifier.head)
    classifier_state = classifier.state_dict()
    detector_state = detector.state_dict()
    missing = [key for key in classifier_state if key not in detector_state]
    mismatched = [
        key
        for key in classifier_state
        if key in detector_state and tuple(classifier_state[key].shape) != tuple(detector_state[key].shape)
    ]
    if missing or mismatched:
        raise ValueError(
            f"Joint model cannot preserve classifier state: missing={missing} mismatched={mismatched}"
        )
    merged = dict(detector_state)
    for key, value in classifier_state.items():
        merged[key] = value.detach().clone()
    detector.load_state_dict(merged, strict=True)
    identity = _verify_encoder_unchanged(classifier_state, detector, list(classifier_state))
    if not identity["passed"]:
        raise RuntimeError(f"Joint model classifier state is not exact: {identity}")
    summary = {
        **summary,
        "classifier_parameter_tensors_preserved": int(len(classifier_state)),
        "classifier_state_identity": identity,
        "dual_head_runtime_only": True,
        "export_model_type": str(checkpoint["model_config"].get("model_type", "")),
    }
    return detector, summary


def _augmentation_value(config: Mapping[str, object], key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def _build_training_transform(
    augmentation: Mapping[str, object],
    model_config: Mapping[str, object],
    *,
    preserve_full_context: bool,
):
    background_mode = (
        "none"
        if preserve_full_context
        else str(_augmentation_value(augmentation, "background_suppression_mode", "none"))
    )
    foreground_mode = (
        "none"
        if preserve_full_context
        else str(_augmentation_value(augmentation, "foreground_crop_mode", "none"))
    )
    return build_train_transform(
        image_size=int(model_config.get("image_size", 256)),
        resize_mode=str(_augmentation_value(augmentation, "resize_mode", "pad")),
        scale_min=float(_augmentation_value(augmentation, "random_resized_crop_scale_min", 0.8)),
        scale_crop_probability=float(
            _augmentation_value(augmentation, "random_resized_crop_probability", 1.0)
        ),
        brightness=float(_augmentation_value(augmentation, "color_jitter_brightness", 0.2)),
        contrast=float(_augmentation_value(augmentation, "color_jitter_contrast", 0.2)),
        saturation=float(_augmentation_value(augmentation, "color_jitter_saturation", 0.15)),
        hue=float(_augmentation_value(augmentation, "color_jitter_hue", 0.02)),
        random_erasing_probability=float(
            _augmentation_value(augmentation, "random_erasing_probability", 0.2)
        ),
        random_affine_degrees=float(
            _augmentation_value(augmentation, "random_affine_degrees", 8.0)
        ),
        random_affine_translate=float(
            _augmentation_value(augmentation, "random_affine_translate", 0.05)
        ),
        random_affine_scale_min=float(
            _augmentation_value(augmentation, "random_affine_scale_min", 0.9)
        ),
        horizontal_flip_probability=float(
            _augmentation_value(augmentation, "horizontal_flip_probability", 0.5)
        ),
        vertical_flip_probability=float(
            _augmentation_value(augmentation, "vertical_flip_probability", 0.1)
        ),
        rotate90_probability=float(_augmentation_value(augmentation, "rotate90_probability", 0.15)),
        lighting_probability=float(_augmentation_value(augmentation, "lighting_probability", 0.15)),
        randaugment_num_ops=int(_augmentation_value(augmentation, "randaugment_num_ops", 0)),
        randaugment_magnitude=int(_augmentation_value(augmentation, "randaugment_magnitude", 10)),
        illumination_normalization=bool(
            _augmentation_value(augmentation, "illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            _augmentation_value(augmentation, "illumination_normalization_strength", 0.0)
        ),
        foreground_crop_mode=foreground_mode,
        foreground_crop_probability=(
            0.0
            if preserve_full_context
            else float(_augmentation_value(augmentation, "foreground_crop_probability", 0.0))
        ),
        foreground_crop_margin_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_margin_ratio", 0.08)
        ),
        foreground_crop_min_mask_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_min_mask_area_ratio", 0.03)
        ),
        foreground_crop_max_mask_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_max_mask_area_ratio", 0.92)
        ),
        foreground_crop_max_crop_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_max_crop_area_ratio", 0.98)
        ),
        background_suppression_mode=background_mode,
        background_suppression_probability=(
            0.0
            if preserve_full_context
            else float(_augmentation_value(augmentation, "background_suppression_probability", 0.0))
        ),
        background_suppression_margin=float(
            _augmentation_value(augmentation, "background_suppression_margin", 0.08)
        ),
        background_suppression_blur_radius=float(
            _augmentation_value(augmentation, "background_suppression_blur_radius", 7.0)
        ),
        surface_detail_amplification_mode=(
            "none"
            if preserve_full_context
            else str(_augmentation_value(augmentation, "surface_detail_amplification_mode", "none"))
        ),
        surface_detail_amplification_probability=(
            0.0
            if preserve_full_context
            else float(
                _augmentation_value(augmentation, "surface_detail_amplification_probability", 0.0)
            )
        ),
        surface_detail_amplification_strength=float(
            _augmentation_value(augmentation, "surface_detail_amplification_strength", 0.0)
        ),
        surface_detail_amplification_blur_radius=float(
            _augmentation_value(augmentation, "surface_detail_amplification_blur_radius", 1.25)
        ),
        surface_detail_amplification_foreground_weight=float(
            _augmentation_value(
                augmentation,
                "surface_detail_amplification_foreground_weight",
                0.85,
            )
        ),
        local_exposure_probability=float(
            _augmentation_value(augmentation, "local_exposure_probability", 0.0)
        ),
        local_exposure_strength=float(
            _augmentation_value(augmentation, "local_exposure_strength", 0.25)
        ),
        obstacle_probability=float(_augmentation_value(augmentation, "obstacle_probability", 0.0)),
        obstacle_max_area=float(_augmentation_value(augmentation, "obstacle_max_area", 0.12)),
        scale_photometric_with_augmentation=bool(
            _augmentation_value(augmentation, "scale_photometric_with_augmentation", False)
        ),
        mean=model_config.get("input_mean", (0.485, 0.456, 0.406)),
        std=model_config.get("input_std", (0.229, 0.224, 0.225)),
    )


def _build_eval_transform_from_config(
    augmentation: Mapping[str, object],
    model_config: Mapping[str, object],
):
    return build_eval_transform(
        image_size=int(model_config.get("image_size", 256)),
        resize_mode=str(_augmentation_value(augmentation, "resize_mode", "pad")),
        illumination_normalization=bool(
            _augmentation_value(augmentation, "illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            _augmentation_value(augmentation, "illumination_normalization_strength", 0.0)
        ),
        foreground_crop_mode=str(_augmentation_value(augmentation, "foreground_crop_mode", "none")),
        foreground_crop_margin_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_margin_ratio", 0.08)
        ),
        foreground_crop_min_mask_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_min_mask_area_ratio", 0.03)
        ),
        foreground_crop_max_mask_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_max_mask_area_ratio", 0.92)
        ),
        foreground_crop_max_crop_area_ratio=float(
            _augmentation_value(augmentation, "foreground_crop_max_crop_area_ratio", 0.98)
        ),
        background_suppression_mode=str(
            _augmentation_value(augmentation, "background_suppression_mode", "none")
        ),
        background_suppression_margin=float(
            _augmentation_value(augmentation, "background_suppression_margin", 0.08)
        ),
        background_suppression_blur_radius=float(
            _augmentation_value(augmentation, "background_suppression_blur_radius", 7.0)
        ),
        mean=model_config.get("input_mean", (0.485, 0.456, 0.406)),
        std=model_config.get("input_std", (0.229, 0.224, 0.225)),
    )


def _build_datasets(
    *,
    data_yaml: Path,
    checkpoint: Mapping[str, object],
    resolved_config: Mapping[str, object],
) -> Tuple[MangoYOLOCropDataset, MangoYOLOCropDataset, MangoYOLOCropDataset]:
    model_config = checkpoint["model_config"]
    augmentation = resolved_config["augmentation_config"]
    class_count = len(checkpoint["class_names"])
    classification_transform = _build_training_transform(
        augmentation,
        model_config,
        preserve_full_context=False,
    )
    detection_transform = _build_training_transform(
        augmentation,
        model_config,
        preserve_full_context=True,
    )
    eval_transform = _build_eval_transform_from_config(augmentation, model_config)
    common = {
        "crop_margin_ratio": float(_augmentation_value(augmentation, "crop_margin_ratio", 0.05)),
        "class_crop_margin_scale_threshold": float(
            _augmentation_value(augmentation, "class_crop_margin_scale_threshold", 1.5)
        ),
        "class_crop_margin_max_ratio": float(
            _augmentation_value(augmentation, "class_crop_margin_max_ratio", 0.05)
        ),
        "expected_num_classes": class_count,
    }
    classification_train = MangoYOLOCropDataset.from_data_yaml(
        data_yaml=data_yaml,
        split="train",
        transform=classification_transform,
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
        **common,
    )
    detection_train = MangoYOLOCropDataset.from_data_yaml(
        data_yaml=data_yaml,
        split="train",
        transform=detection_transform,
        crop_to_primary_object=False,
        classification_target=False,
        classification_object_crops=False,
        expected_num_classes=class_count,
    )
    classification_val = MangoYOLOCropDataset.from_data_yaml(
        data_yaml=data_yaml,
        split="val",
        transform=eval_transform,
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
        **common,
    )
    return classification_train, detection_train, classification_val


def _build_validation_dataset(
    *,
    data_yaml: Path,
    checkpoint: Mapping[str, object],
    resolved_config: Mapping[str, object],
) -> MangoYOLOCropDataset:
    model_config = checkpoint["model_config"]
    augmentation = resolved_config["augmentation_config"]
    transform = _build_eval_transform_from_config(augmentation, model_config)
    return MangoYOLOCropDataset.from_data_yaml(
        data_yaml=data_yaml,
        split="val",
        transform=transform,
        crop_margin_ratio=float(_augmentation_value(augmentation, "crop_margin_ratio", 0.05)),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        class_crop_margin_scale_threshold=float(
            _augmentation_value(augmentation, "class_crop_margin_scale_threshold", 1.5)
        ),
        class_crop_margin_max_ratio=float(
            _augmentation_value(augmentation, "class_crop_margin_max_ratio", 0.05)
        ),
        classification_bbox_metadata=True,
        expected_num_classes=len(checkpoint["class_names"]),
    )


def _classification_forward(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], Tensor]:
    bbox = metadata.get("bbox")
    features = model.forward_features(
        images,
        image_valid_mask=metadata.get("image_mask"),
        bbox_token_prior=bbox,
    )
    if torch.is_tensor(bbox):
        features["bbox"] = bbox
    logits = classification_logits_from_features(model, features)
    return features, logits


def _classification_objective(
    model: nn.Module,
    teacher: nn.Module,
    *,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    criterion: nn.Module,
    pairwise_weight: float,
    teacher_focus_weight: float,
    teacher_focus_config: Mapping[str, object],
) -> Tuple[Tensor, Dict[str, float]]:
    features, logits = _classification_forward(model, images, metadata)
    primary_loss = criterion(logits, labels)
    pairwise_loss = _pairwise_margin_loss_from_features(
        model=model,
        features=features,
        targets=labels,
    )
    with torch.inference_mode():
        _, teacher_logits = _classification_forward(teacher, images, metadata)
        teacher_probabilities = teacher_logits.float().softmax(dim=1)
    teacher_focus_loss, teacher_focus_fraction, _, _ = _teacher_focus_binary_loss_from_logits(
        logits=logits,
        teacher_probabilities=teacher_probabilities,
        hard_labels=labels,
        targets=labels,
        focus_class=int(teacher_focus_config.get("focus_class", 1)),
        classes=str(teacher_focus_config.get("classes", "0,1,2,4")),
        teacher_min_confidence=float(teacher_focus_config.get("min_confidence", 0.0)),
        error_power=float(teacher_focus_config.get("error_power", 0.0)),
        hard_target_blend=float(teacher_focus_config.get("hard_target_blend", 0.0)),
        require_agreement=bool(teacher_focus_config.get("require_agreement", True)),
    )
    total = (
        primary_loss
        + float(pairwise_weight) * pairwise_loss
        + float(teacher_focus_weight) * teacher_focus_loss
    )
    return total, {
        "classification_loss": float(primary_loss.detach().cpu()),
        "pairwise_loss": float(pairwise_loss.detach().cpu()),
        "teacher_focus_loss": float(teacher_focus_loss.detach().cpu()),
        "teacher_focus_fraction": float(teacher_focus_fraction),
        "total_primary_loss": float(total.detach().cpu()),
    }


def _gradient_dot_and_norms(primary: Gradient, auxiliary: Gradient) -> Tuple[Tensor, Tensor, Tensor]:
    device = primary[0].device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    primary_squared = torch.zeros((), device=device, dtype=torch.float64)
    auxiliary_squared = torch.zeros((), device=device, dtype=torch.float64)
    for left, right in zip(primary, auxiliary):
        left64 = left.detach().double()
        right64 = right.detach().double()
        dot = dot + torch.sum(left64 * right64)
        primary_squared = primary_squared + torch.sum(left64.square())
        auxiliary_squared = auxiliary_squared + torch.sum(right64.square())
    return dot, primary_squared, auxiliary_squared


def componentwise_spatial_pcgrad(
    primary: Gradient,
    objectness: Gradient,
    localization: Gradient,
    *,
    auxiliary_ratio: float,
) -> Tuple[Gradient, Dict[str, float]]:
    _, primary_squared, _ = _gradient_dot_and_norms(primary, primary)
    primary_norm = primary_squared.sqrt().clamp(min=1e-12)
    projected_components: list[Gradient] = []
    telemetry: Dict[str, float] = {"primary_norm": float(primary_norm.cpu())}
    for name, component in (("objectness", objectness), ("localization", localization)):
        raw_dot, _, component_squared = _gradient_dot_and_norms(primary, component)
        component_norm = component_squared.sqrt().clamp(min=1e-12)
        scale = primary_norm / component_norm
        scaled = [value * scale.to(dtype=value.dtype) for value in component]
        scaled_dot, _, scaled_squared = _gradient_dot_and_norms(primary, scaled)
        if bool((scaled_dot < 0.0).item()):
            coefficient = scaled_dot / primary_squared.clamp(min=1e-24)
            projected = [
                auxiliary_value - coefficient.to(dtype=primary_value.dtype) * primary_value
                for primary_value, auxiliary_value in zip(primary, scaled)
            ]
        else:
            projected = scaled
        projected_dot, _, projected_squared = _gradient_dot_and_norms(primary, projected)
        cosine = raw_dot / (primary_norm * component_norm).clamp(min=1e-24)
        retained = projected_squared.sqrt() / scaled_squared.sqrt().clamp(min=1e-12)
        telemetry[f"{name}_raw_cosine"] = float(cosine.clamp(-1.0, 1.0).cpu())
        telemetry[f"{name}_raw_norm"] = float(component_norm.cpu())
        telemetry[f"{name}_projected_dot"] = float(projected_dot.cpu())
        telemetry[f"{name}_retained_norm_ratio"] = float(retained.cpu())
        projected_components.append(projected)

    per_component_ratio = float(auxiliary_ratio) / float(len(projected_components))
    combined = [value.clone() for value in primary]
    for component in projected_components:
        for index, value in enumerate(component):
            combined[index] = combined[index] + per_component_ratio * value
    _, _, combined_squared = _gradient_dot_and_norms(combined, combined)
    combined_norm = combined_squared.sqrt().clamp(min=1e-12)
    scale = primary_norm / combined_norm
    combined = [value * scale.to(dtype=value.dtype) for value in combined]
    telemetry["auxiliary_ratio"] = float(auxiliary_ratio)
    telemetry["per_component_ratio"] = per_component_ratio
    telemetry["combined_pre_normalization_norm"] = float(combined_norm.cpu())
    telemetry["combined_final_norm"] = float(primary_norm.cpu())
    return combined, telemetry


def _materialize_gradients(
    gradients: Sequence[Optional[Tensor]],
    parameters: Sequence[nn.Parameter],
) -> Gradient:
    return [
        (
            gradient.detach().to(dtype=torch.float32)
            if gradient is not None
            else torch.zeros_like(parameter, dtype=torch.float32)
        )
        for gradient, parameter in zip(gradients, parameters)
    ]


def _assign_gradients(
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[Optional[Tensor] | Tensor],
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            parameter.grad = None
        else:
            parameter.grad = gradient.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )


def _partition_parameters(
    model: nn.Module,
    shared_names: Sequence[str],
    *,
    enable_detector_heads: bool,
) -> Dict[str, object]:
    shared_set = set(shared_names)
    shared_parameters: list[nn.Parameter] = []
    primary_nonshared_parameters: list[nn.Parameter] = []
    detector_head_parameters: list[nn.Parameter] = []
    primary_names: list[str] = []
    detector_names: list[str] = []
    for name, parameter in model.named_parameters():
        is_detector_head = any(name.startswith(prefix) for prefix in DETECTOR_PARAMETER_PREFIXES)
        if is_detector_head:
            parameter.requires_grad_(bool(enable_detector_heads))
            if enable_detector_heads:
                detector_head_parameters.append(parameter)
                detector_names.append(name)
            continue
        parameter.requires_grad_(True)
        primary_names.append(name)
        if name in shared_set:
            shared_parameters.append(parameter)
        else:
            primary_nonshared_parameters.append(parameter)
    if len(shared_parameters) != len(shared_names):
        raise ValueError(
            f"Shared parameter partition mismatch: {len(shared_parameters)} != {len(shared_names)}"
        )
    if enable_detector_heads and not detector_head_parameters:
        raise ValueError("Candidate has no detector-head parameters")
    return {
        "shared": shared_parameters,
        "primary_nonshared": primary_nonshared_parameters,
        "primary_all": [*shared_parameters, *primary_nonshared_parameters],
        "detector_head": detector_head_parameters,
        "primary_names": primary_names,
        "detector_names": detector_names,
    }


def _control_step(
    model: nn.Module,
    teacher: nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    primary_parameters: Sequence[nn.Parameter],
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    criterion: nn.Module,
    pairwise_weight: float,
    teacher_focus_weight: float,
    teacher_focus_config: Mapping[str, object],
    grad_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, details = _classification_objective(
        model,
        teacher,
        images=images,
        labels=labels,
        metadata=metadata,
        criterion=criterion,
        pairwise_weight=pairwise_weight,
        teacher_focus_weight=teacher_focus_weight,
        teacher_focus_config=teacher_focus_config,
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        primary_parameters,
        max_norm=float(grad_clip_norm),
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {**details, "gradient_norm": float(gradient_norm.detach().cpu())}


def _candidate_step(
    model: nn.Module,
    teacher: nn.Module,
    *,
    primary_optimizer: torch.optim.Optimizer,
    detector_optimizer: torch.optim.Optimizer,
    parameter_groups: Mapping[str, object],
    crop_images: Tensor,
    labels: Tensor,
    crop_metadata: Mapping[str, Tensor],
    full_images: Tensor,
    detection_targets: Sequence[Dict[str, Tensor]],
    full_image_masks: Optional[Tensor],
    classification_criterion: nn.Module,
    detection_criterion: nn.Module,
    pairwise_weight: float,
    teacher_focus_weight: float,
    teacher_focus_config: Mapping[str, object],
    auxiliary_ratio: float,
    grad_clip_norm: float,
) -> Dict[str, float]:
    shared_parameters = list(parameter_groups["shared"])
    primary_nonshared = list(parameter_groups["primary_nonshared"])
    primary_all = list(parameter_groups["primary_all"])
    detector_head = list(parameter_groups["detector_head"])
    model.train()
    primary_optimizer.zero_grad(set_to_none=True)
    detector_optimizer.zero_grad(set_to_none=True)

    classification_loss, details = _classification_objective(
        model,
        teacher,
        images=crop_images,
        labels=labels,
        metadata=crop_metadata,
        criterion=classification_criterion,
        pairwise_weight=pairwise_weight,
        teacher_focus_weight=teacher_focus_weight,
        teacher_focus_config=teacher_focus_config,
    )
    primary_gradients = torch.autograd.grad(
        classification_loss,
        [*shared_parameters, *primary_nonshared],
        allow_unused=True,
    )
    shared_primary_raw = primary_gradients[: len(shared_parameters)]
    nonshared_primary_raw = primary_gradients[len(shared_parameters) :]
    shared_primary = _materialize_gradients(shared_primary_raw, shared_parameters)

    detection_output = model(full_images, image_valid_mask=full_image_masks)
    component_losses = _detection_component_losses(
        detection_criterion,
        detection_output,
        detection_targets,
    )
    localization_loss = (
        DETECTION_BBOX_L1_WEIGHT * component_losses["det_bbox_l1"]
        + DETECTION_BBOX_GIOU_WEIGHT * component_losses["det_bbox_giou"]
    )
    detection_total = (
        DETECTION_CLASS_WEIGHT * component_losses["det_class"]
        + DETECTION_OBJECTNESS_WEIGHT * component_losses["det_objectness"]
        + localization_loss
    )
    detector_head_gradients = torch.autograd.grad(
        detection_total,
        detector_head,
        retain_graph=True,
        allow_unused=True,
    )
    objectness_raw = torch.autograd.grad(
        component_losses["det_objectness"],
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    localization_raw = torch.autograd.grad(
        localization_loss,
        shared_parameters,
        allow_unused=True,
    )
    objectness_gradients = _materialize_gradients(objectness_raw, shared_parameters)
    localization_gradients = _materialize_gradients(localization_raw, shared_parameters)
    combined_shared, projection = componentwise_spatial_pcgrad(
        shared_primary,
        objectness_gradients,
        localization_gradients,
        auxiliary_ratio=float(auxiliary_ratio),
    )
    _assign_gradients(shared_parameters, combined_shared)
    _assign_gradients(primary_nonshared, nonshared_primary_raw)
    _assign_gradients(detector_head, detector_head_gradients)
    primary_gradient_norm = torch.nn.utils.clip_grad_norm_(
        primary_all,
        max_norm=float(grad_clip_norm),
        error_if_nonfinite=True,
    )
    detector_gradient_norm = torch.nn.utils.clip_grad_norm_(
        detector_head,
        max_norm=float(grad_clip_norm),
        error_if_nonfinite=True,
    )
    primary_optimizer.step()
    detector_optimizer.step()
    return {
        **details,
        **projection,
        "det_class_loss": float(component_losses["det_class"].detach().cpu()),
        "det_objectness_loss": float(component_losses["det_objectness"].detach().cpu()),
        "det_bbox_l1_loss": float(component_losses["det_bbox_l1"].detach().cpu()),
        "det_bbox_giou_loss": float(component_losses["det_bbox_giou"].detach().cpu()),
        "det_localization_loss": float(localization_loss.detach().cpu()),
        "det_total_loss": float(detection_total.detach().cpu()),
        "primary_gradient_norm_before_clip": float(primary_gradient_norm.detach().cpu()),
        "detector_gradient_norm_before_clip": float(detector_gradient_norm.detach().cpu()),
    }


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    class_count = int(probabilities.shape[1])
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class = []
    for class_index in range(class_count):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class.append(
            {
                "class_index": class_index,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(confusion[class_index, :].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return {
        "rows": int(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _validation_rows(dataset: MangoYOLOCropDataset) -> list[PairedSource]:
    return [
        PairedSource(
            source_stem=sample.image_path.stem.casefold(),
            label=int(sample.primary_label),
            classification_index=int(index),
            detection_index=-1,
        )
        for index, sample in enumerate(dataset.samples)
    ]


@torch.inference_mode()
def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    dataset: MangoYOLOCropDataset,
    batch_size: int,
    device: torch.device,
    output_csv: Path,
) -> Dict[str, object]:
    payload = _load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(payload).to(device).eval()
    rows = _validation_rows(dataset)
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    csv_rows: list[Dict[str, object]] = []
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        images, labels, metadata = _stack_classification_batch(dataset, batch_rows)
        metadata = {
            key: value.to(device=device, non_blocking=True)
            for key, value in metadata.items()
        }
        _, logits = _classification_forward(
            model,
            images.to(device=device, non_blocking=True),
            metadata,
        )
        probabilities = logits.float().softmax(dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        all_probabilities.append(probabilities)
        all_labels.append(labels.numpy())
        for local_index, row in enumerate(batch_rows):
            sample = dataset.samples[int(row.classification_index)]
            ordered = np.sort(probabilities[local_index])
            record: Dict[str, object] = {
                "sample_index": int(row.classification_index),
                "path": str(sample.image_path.resolve()),
                "source_stem": row.source_stem,
                "object_index": int(sample.primary_object_index),
                "label": int(labels[local_index]),
                "prediction": int(predictions[local_index]),
                "target_index": int(labels[local_index]),
                "prediction_index": int(predictions[local_index]),
                "y_true": int(labels[local_index]),
                "y_pred": int(predictions[local_index]),
                "confidence": float(ordered[-1]),
                "margin": float(ordered[-1] - ordered[-2]),
            }
            for class_index, probability in enumerate(probabilities[local_index].tolist()):
                record[f"prob_{class_index}"] = float(probability)
            csv_rows.append(record)
    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = _classification_metrics(labels, probabilities)
    fieldnames = list(csv_rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    metrics["checkpoint"] = str(checkpoint_path.resolve())
    metrics["predictions_csv"] = str(output_csv.resolve())
    metrics["probabilities"] = probabilities
    metrics["labels"] = labels
    metrics["predictions"] = probabilities.argmax(axis=1)
    return metrics


@torch.inference_mode()
def _evaluate_checkpoints_shared_batches(
    checkpoints: Mapping[str, Path],
    *,
    dataset: MangoYOLOCropDataset,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Dict[str, object]]:
    models = {
        name: build_model_from_checkpoint(_load_checkpoint(path)).to(device).eval()
        for name, path in checkpoints.items()
    }
    rows = _validation_rows(dataset)
    probability_rows: Dict[str, list[np.ndarray]] = {name: [] for name in models}
    labels_rows: list[np.ndarray] = []
    csv_rows: Dict[str, list[Dict[str, object]]] = {name: [] for name in models}
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        images, labels, metadata = _stack_classification_batch(dataset, batch_rows)
        images = images.to(device=device, non_blocking=True)
        device_metadata = {
            key: value.to(device=device, non_blocking=True)
            for key, value in metadata.items()
        }
        labels_rows.append(labels.numpy())
        for name, model in models.items():
            _, logits = _classification_forward(model, images, device_metadata)
            probabilities = logits.float().softmax(dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            probability_rows[name].append(probabilities)
            for local_index, row in enumerate(batch_rows):
                sample = dataset.samples[int(row.classification_index)]
                ordered = np.sort(probabilities[local_index])
                record: Dict[str, object] = {
                    "sample_index": int(row.classification_index),
                    "path": str(sample.image_path.resolve()),
                    "source_stem": row.source_stem,
                    "object_index": int(sample.primary_object_index),
                    "label": int(labels[local_index]),
                    "prediction": int(predictions[local_index]),
                    "target_index": int(labels[local_index]),
                    "prediction_index": int(predictions[local_index]),
                    "y_true": int(labels[local_index]),
                    "y_pred": int(predictions[local_index]),
                    "confidence": float(ordered[-1]),
                    "margin": float(ordered[-1] - ordered[-2]),
                }
                for class_index, probability in enumerate(probabilities[local_index].tolist()):
                    record[f"prob_{class_index}"] = float(probability)
                csv_rows[name].append(record)

    labels = np.concatenate(labels_rows, axis=0)
    output: Dict[str, Dict[str, object]] = {}
    for name, checkpoint_path in checkpoints.items():
        probabilities = np.concatenate(probability_rows[name], axis=0)
        metrics = _classification_metrics(labels, probabilities)
        prediction_path = output_dir / f"{name}_val_predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[name][0]))
            writer.writeheader()
            writer.writerows(csv_rows[name])
        metrics.update(
            {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "predictions_csv": str(prediction_path.resolve()),
                "probabilities": probabilities,
                "labels": labels,
                "predictions": probabilities.argmax(axis=1),
            }
        )
        output[name] = metrics
    return output


def _transition_summary(
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    focus_class_index: int,
) -> Dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    reference_predictions = np.asarray(reference_predictions, dtype=np.int64)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.int64)
    changed = reference_predictions != candidate_predictions
    reference_correct = reference_predictions == labels
    candidate_correct = candidate_predictions == labels
    focus = int(focus_class_index)
    negative = labels != focus
    positive = labels == focus
    return {
        "changed": int(changed.sum()),
        "corrections": int((changed & ~reference_correct & candidate_correct).sum()),
        "harms": int((changed & reference_correct & ~candidate_correct).sum()),
        "neutral": int((changed & (reference_correct == candidate_correct)).sum()),
        "focus_false_positive_removed": int(
            (negative & (reference_predictions == focus) & (candidate_predictions != focus)).sum()
        ),
        "focus_false_positive_created": int(
            (negative & (reference_predictions != focus) & (candidate_predictions == focus)).sum()
        ),
        "focus_false_negative_rescued": int(
            (positive & (reference_predictions != focus) & (candidate_predictions == focus)).sum()
        ),
        "focus_true_positive_broken": int(
            (positive & (reference_predictions == focus) & (candidate_predictions != focus)).sum()
        ),
    }


def assess_smoke_gate(
    *,
    keeper_metrics: Mapping[str, object],
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    transitions_vs_control: Mapping[str, int],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    keeper_focus = keeper_metrics["per_class"][focus]
    control_focus = control_metrics["per_class"][focus]
    candidate_focus = candidate_metrics["per_class"][focus]
    observed = {
        "validation_rows": int(candidate_metrics["rows"]),
        "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
        "candidate_focus_f1": float(candidate_focus["f1"]),
        "candidate_focus_recall": float(candidate_focus["recall"]),
        "control_macro_f1": float(control_metrics["macro_f1"]),
        "control_focus_f1": float(control_focus["f1"]),
        "control_focus_recall": float(control_focus["recall"]),
        "keeper_macro_f1": float(keeper_metrics["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "candidate_minus_control_macro_f1": float(
            candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
        ),
        "candidate_minus_control_focus_f1": float(
            candidate_focus["f1"] - control_focus["f1"]
        ),
        "candidate_minus_keeper_macro_f1": float(
            candidate_metrics["macro_f1"] - keeper_metrics["macro_f1"]
        ),
        "candidate_minus_keeper_focus_f1": float(candidate_focus["f1"] - keeper_focus["f1"]),
        "transitions_vs_control": dict(transitions_vs_control),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_validation_2606": observed["validation_rows"] == 2606,
        "test_not_used": not bool(test_split_used),
        "candidate_macro_ge_locked_keeper_min": observed["candidate_macro_f1"]
        >= KEEPER_RELOAD_MACRO_F1,
        "candidate_focus_reaches_0p70": observed["candidate_focus_f1"] >= FOCUS_MILESTONE,
        "candidate_focus_gain_vs_control_ge_0p005": observed[
            "candidate_minus_control_focus_f1"
        ]
        >= 0.005,
        "candidate_macro_not_below_control_by_0p001": observed[
            "candidate_minus_control_macro_f1"
        ]
        >= -0.001,
        "candidate_focus_recall_not_below_keeper_by_0p01": observed[
            "candidate_focus_recall"
        ]
        >= observed["keeper_focus_recall"] - 0.01,
        "candidate_corrections_ge_harms_vs_control": int(
            transitions_vs_control["corrections"]
        )
        >= int(transitions_vs_control["harms"]),
        "candidate_focus_fp_removed_ge_created_vs_control": int(
            transitions_vs_control["focus_false_positive_removed"]
        )
        >= int(transitions_vs_control["focus_false_positive_created"]),
        "candidate_focus_fn_rescued_ge_tp_broken_vs_control": int(
            transitions_vs_control["focus_false_negative_rescued"]
        )
        >= int(transitions_vs_control["focus_true_positive_broken"]),
    }
    failed = [key for key, passed in checks.items() if not bool(passed)]
    return {
        "probe_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": {
            "keeper_reload_macro_f1": KEEPER_RELOAD_MACRO_F1,
            "keeper_reload_focus_f1": KEEPER_RELOAD_FOCUS_F1,
            "focus_milestone": FOCUS_MILESTONE,
            "focus_gain_vs_control": 0.005,
            "macro_tolerance_vs_control": -0.001,
            "focus_recall_tolerance_vs_keeper": -0.01,
        },
    }


def _export_classifier_checkpoint(
    joint_model: nn.Module,
    *,
    reference: Mapping[str, object],
    output_path: Path,
    mode: str,
    train_config: Mapping[str, object],
) -> Dict[str, object]:
    classifier = build_model_from_checkpoint(dict(reference))
    target_state = classifier.state_dict()
    joint_state = joint_model.state_dict()
    missing = [key for key in target_state if key not in joint_state]
    mismatched = [
        key
        for key in target_state
        if key in joint_state and tuple(target_state[key].shape) != tuple(joint_state[key].shape)
    ]
    if missing or mismatched:
        raise ValueError(f"Unsafe classifier export: missing={missing} mismatched={mismatched}")
    exported_state = {
        key: joint_state[key].detach().cpu().clone() for key in target_state
    }
    classifier.load_state_dict(exported_state, strict=True)
    payload = fresh_checkpoint_payload(
        reference,
        model_state=exported_state,
        model_config=reference["model_config"],
        transfer_summary={
            "mode": mode,
            "source": "runtime dual-head joint model",
            "detector_heads_exported": False,
            "classifier_key_count": int(len(exported_state)),
        },
        train_config=train_config,
    )
    payload["epoch"] = 1
    payload["joint_spatial_pcgrad"] = {
        "mode": mode,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "detector_heads_exported": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    reloaded = _load_checkpoint(output_path)
    build_model_from_checkpoint(reloaded).load_state_dict(reloaded["model_state"], strict=True)
    return payload


def _compact_evaluation(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"probabilities", "labels", "predictions"}
    }


def _mean_telemetry(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _write_training_csv(
    path: Path,
    control_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
) -> None:
    rows = []
    for mode, values in (("control", control_rows), ("candidate", candidate_rows)):
        for batch_index, row in enumerate(values, start=1):
            rows.append({"mode": mode, "batch": batch_index, **dict(row)})
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_smoke_artifact_manifest(output_dir: Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "artifact_manifest.json"
    files = []
    for path in sorted(output_dir.rglob("*"), key=lambda value: str(value).casefold()):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "name": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    aggregate = hashlib.sha256()
    for row in files:
        aggregate.update(str(row["name"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("ascii"))
    manifest = {
        "mode": "joint_spatial_pcgrad_smoke_evidence_manifest",
        "payload_count": int(len(files)),
        "payload_size_bytes": int(sum(row["size_bytes"] for row in files)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": files,
        "contains_checkpoint": any(row["name"].lower().endswith(".pt") for row in files),
        "contains_model_binary": any(row["name"].lower().endswith(".pt") for row in files),
        "contains_test_payload": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _select_macrostep_rows(
    pairs: Sequence[PairedSource],
    *,
    class_count: int,
    samples_per_class: int,
    seed: int,
) -> list[PairedSource]:
    rng = np.random.default_rng(int(seed))
    output: list[PairedSource] = []
    for class_index in range(int(class_count)):
        candidates = [row for row in pairs if int(row.label) == class_index]
        if len(candidates) < int(samples_per_class):
            raise ValueError(
                f"Class {class_index} has {len(candidates)} rows, "
                f"{samples_per_class} are required"
            )
        indices = rng.permutation(len(candidates))[: int(samples_per_class)]
        output.extend(candidates[int(index)] for index in indices)
    return output


def _set_shared_only_gradients(
    model: nn.Module,
    shared_names: Sequence[str],
) -> list[nn.Parameter]:
    shared = set(shared_names)
    parameters = []
    observed_names = []
    for name, parameter in model.named_parameters():
        enabled = name in shared
        parameter.requires_grad_(enabled)
        if enabled:
            parameters.append(parameter)
            observed_names.append(name)
    if observed_names != list(shared_names):
        raise ValueError("Shared parameter order changed in normalized macro-step")
    model.eval()
    return parameters


def _gradient_average(accumulator: Gradient, rows: int) -> Gradient:
    return [value / float(rows) for value in accumulator]


def _gradient_weighted_sum(
    gradients: Mapping[int, Gradient],
    weights: Mapping[int, float],
) -> Gradient:
    first = gradients[min(gradients)]
    output = [torch.zeros_like(value) for value in first]
    for class_index, gradient in gradients.items():
        for target, value in zip(output, gradient):
            target.add_(value, alpha=float(weights[int(class_index)]))
    return output


def _measure_normalized_macrostep_gradients(
    model: nn.Module,
    *,
    classification_dataset: MangoYOLOCropDataset,
    detection_dataset: MangoYOLOCropDataset,
    rows: Sequence[PairedSource],
    shared_parameters: Sequence[nn.Parameter],
    class_count: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, Gradient], Dict[str, object]]:
    criterion = _detection_criterion(class_count).to(device)
    per_class_primary: Dict[int, Gradient] = {}
    per_class_objectness: Dict[int, Gradient] = {}
    per_class_localization: Dict[int, Gradient] = {}
    per_class_losses: Dict[str, object] = {}
    model.eval()
    for class_index in range(int(class_count)):
        class_rows = [row for row in rows if int(row.label) == class_index]
        accumulators = {
            "primary": [torch.zeros_like(parameter, dtype=torch.float32) for parameter in shared_parameters],
            "objectness": [torch.zeros_like(parameter, dtype=torch.float32) for parameter in shared_parameters],
            "localization": [torch.zeros_like(parameter, dtype=torch.float32) for parameter in shared_parameters],
        }
        loss_sums = {"primary": 0.0, "objectness": 0.0, "localization": 0.0}
        total_rows = 0
        for start in range(0, len(class_rows), int(batch_size)):
            batch_rows = class_rows[start : start + int(batch_size)]
            row_count = len(batch_rows)
            crop_images, labels, metadata = _stack_classification_batch(
                classification_dataset,
                batch_rows,
            )
            crop_images = crop_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            metadata = {
                key: value.to(device=device, non_blocking=True)
                for key, value in metadata.items()
            }
            _, logits = _classification_forward(model, crop_images, metadata)
            primary_loss = F.cross_entropy(logits, labels)
            primary_raw = torch.autograd.grad(
                primary_loss,
                shared_parameters,
                allow_unused=True,
            )
            primary = _materialize_gradients(primary_raw, shared_parameters)

            full_images, targets, masks = _stack_detection_batch(
                detection_dataset,
                [row.detection_index for row in batch_rows],
                device,
            )
            detection_output = model(full_images, image_valid_mask=masks)
            components = _detection_component_losses(criterion, detection_output, targets)
            localization_loss = (
                DETECTION_BBOX_L1_WEIGHT * components["det_bbox_l1"]
                + DETECTION_BBOX_GIOU_WEIGHT * components["det_bbox_giou"]
            )
            objectness_raw = torch.autograd.grad(
                components["det_objectness"],
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            localization_raw = torch.autograd.grad(
                localization_loss,
                shared_parameters,
                allow_unused=True,
            )
            objectness = _materialize_gradients(objectness_raw, shared_parameters)
            localization = _materialize_gradients(localization_raw, shared_parameters)
            for key, gradient in (
                ("primary", primary),
                ("objectness", objectness),
                ("localization", localization),
            ):
                for target, value in zip(accumulators[key], gradient):
                    target.add_(value, alpha=float(row_count))
            loss_sums["primary"] += float(primary_loss.detach().cpu()) * row_count
            loss_sums["objectness"] += (
                float(components["det_objectness"].detach().cpu()) * row_count
            )
            loss_sums["localization"] += float(localization_loss.detach().cpu()) * row_count
            total_rows += row_count
        per_class_primary[class_index] = _gradient_average(accumulators["primary"], total_rows)
        per_class_objectness[class_index] = _gradient_average(
            accumulators["objectness"],
            total_rows,
        )
        per_class_localization[class_index] = _gradient_average(
            accumulators["localization"],
            total_rows,
        )
        per_class_losses[str(class_index)] = {
            key: float(value / total_rows) for key, value in loss_sums.items()
        }

    all_pair_counts = {
        class_index: int(
            sum(int(row.label) == class_index for row in build_paired_single_object_sources(
                classification_dataset,
                detection_dataset,
            ))
        )
        for class_index in range(int(class_count))
    }
    total_pair_count = sum(all_pair_counts.values())
    natural_weights = {
        class_index: float(all_pair_counts[class_index] / total_pair_count)
        for class_index in range(int(class_count))
    }
    return {
        "primary": _gradient_weighted_sum(per_class_primary, natural_weights),
        "objectness": _gradient_weighted_sum(per_class_objectness, natural_weights),
        "localization": _gradient_weighted_sum(per_class_localization, natural_weights),
    }, {
        "per_class_losses": per_class_losses,
        "single_object_class_counts": {
            str(key): value for key, value in all_pair_counts.items()
        },
        "natural_class_weights": {
            str(key): value for key, value in natural_weights.items()
        },
    }


def _apply_normalized_parameter_step(
    parameters: Sequence[nn.Parameter],
    direction: Gradient,
    *,
    parameter_ratio: float,
) -> Dict[str, float]:
    parameter_squared = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    direction_squared = torch.zeros_like(parameter_squared)
    originals = []
    for parameter, gradient in zip(parameters, direction):
        originals.append(parameter.detach().clone())
        parameter_squared = parameter_squared + parameter.detach().double().square().sum()
        direction_squared = direction_squared + gradient.detach().double().square().sum()
    parameter_norm = parameter_squared.sqrt().clamp(min=1e-12)
    direction_norm = direction_squared.sqrt().clamp(min=1e-12)
    requested_step_norm = float(parameter_ratio) * parameter_norm
    scale = requested_step_norm / direction_norm
    with torch.no_grad():
        for parameter, gradient in zip(parameters, direction):
            parameter.add_(
                gradient.to(device=parameter.device, dtype=parameter.dtype),
                alpha=-float(scale.cpu()),
            )
    delta_squared = 0.0
    for parameter, original in zip(parameters, originals):
        delta_squared += float((parameter.detach().double() - original.double()).square().sum().cpu())
    actual_step_norm = math.sqrt(max(0.0, delta_squared))
    return {
        "parameter_norm": float(parameter_norm.cpu()),
        "direction_norm": float(direction_norm.cpu()),
        "requested_step_norm": float(requested_step_norm.cpu()),
        "actual_step_norm": actual_step_norm,
        "actual_parameter_ratio": actual_step_norm / max(float(parameter_norm.cpu()), 1e-12),
    }


def _run_adamw_microbatch_smoke(args: argparse.Namespace) -> Dict[str, object]:
    start_time = time.perf_counter()
    if int(args.max_train_batches) <= 0 or int(args.max_train_batches) > 120:
        raise ValueError("max-train-batches must be in [1, 120] for this smoke")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if int(args.head_warmup_batches) < 20:
        raise ValueError("head-warmup-batches must be at least 20")
    if float(args.primary_learning_rate) <= 0.0 or float(args.detector_head_learning_rate) <= 0.0:
        raise ValueError("learning rates must be positive")
    if not 0.0 < float(args.auxiliary_gradient_ratio) <= 1.0:
        raise ValueError("auxiliary-gradient-ratio must be in (0, 1]")
    if float(args.grad_clip_norm) <= 0.0:
        raise ValueError("grad-clip-norm must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    torch.set_float32_matmul_precision("high")
    device = _resolve_device(str(args.device))
    checkpoint_path = Path(args.checkpoint).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    resolved_config = _load_resolved_config(checkpoint_path, args.resolved_config)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_count = len(class_names)
    focus_class_index = int(args.focus_class_index)
    if not 0 <= focus_class_index < class_count:
        raise ValueError("focus-class-index is outside the class range")

    classification_train, detection_train, classification_val = _build_datasets(
        data_yaml=data_yaml,
        checkpoint=checkpoint,
        resolved_config=resolved_config,
    )
    paired_sources = build_paired_single_object_sources(
        classification_train,
        detection_train,
    )
    required_rows = int(args.max_train_batches) * int(args.batch_size)
    if len(paired_sources) < required_rows:
        raise ValueError(f"Smoke requires {required_rows} paired sources")
    rng = np.random.default_rng(int(args.seed) + 10)
    order = rng.permutation(len(paired_sources))[:required_rows]
    selected_pairs = [paired_sources[int(index)] for index in order]
    selected_stems = {row.source_stem for row in selected_pairs}

    teacher = build_model_from_checkpoint(checkpoint).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    candidate, joint_summary = _joint_model_from_checkpoint(
        checkpoint,
        num_queries=int(args.num_queries),
        decoder_depth=int(args.decoder_depth),
        decoder_ffn_dim=int(args.decoder_ffn_dim),
        decoder_dropout=float(args.decoder_dropout),
        seed=int(args.seed),
    )
    candidate = candidate.to(device)
    shared_names = _shared_parameter_names(teacher, candidate)
    encoder_reference = {
        name: candidate.state_dict()[name].detach().cpu().clone()
        for name in shared_names
    }
    warmup_curve, warmup_summary = _warm_up_detector_heads(
        candidate,
        dataset=detection_train,
        allowed_indices=list(range(len(detection_train))),
        shared_names=shared_names,
        class_count=class_count,
        batch_size=int(args.head_warmup_batch_size),
        max_batches=int(args.head_warmup_batches),
        learning_rate=float(args.detector_head_learning_rate),
        weight_decay=float(args.detector_head_weight_decay),
        seed=int(args.seed) + 20,
        device=device,
    )
    encoder_integrity = _verify_encoder_unchanged(
        encoder_reference,
        candidate,
        shared_names,
    )
    if not encoder_integrity["passed"]:
        raise RuntimeError(f"Head warm-up modified the shared encoder: {encoder_integrity}")
    control = copy.deepcopy(candidate)

    candidate_groups = _partition_parameters(
        candidate,
        shared_names,
        enable_detector_heads=True,
    )
    control_groups = _partition_parameters(
        control,
        shared_names,
        enable_detector_heads=False,
    )
    candidate_primary_optimizer = torch.optim.AdamW(
        candidate_groups["primary_all"],
        lr=float(args.primary_learning_rate),
        weight_decay=float(args.primary_weight_decay),
    )
    candidate_detector_optimizer = torch.optim.AdamW(
        candidate_groups["detector_head"],
        lr=float(args.detector_head_learning_rate),
        weight_decay=float(args.detector_head_weight_decay),
    )
    control_optimizer = torch.optim.AdamW(
        control_groups["primary_all"],
        lr=float(args.primary_learning_rate),
        weight_decay=float(args.primary_weight_decay),
    )

    train_config = dict(resolved_config.get("train_config", {}))
    imbalance = dict(resolved_config.get("imbalance_summary", {}))
    class_counts = [int(value) for value in imbalance.get("class_counts", [])]
    if len(class_counts) != class_count:
        raise ValueError("Resolved imbalance_summary class counts are required")
    classification_criterion = LDAMFocalLoss(
        class_counts=class_counts,
        weight=None,
        gamma=float(imbalance.get("focal_loss_gamma", 1.0)),
        focal_mix=float(imbalance.get("focal_loss_mix", 0.1)),
        label_smoothing=float(train_config.get("label_smoothing", 0.02)),
        max_margin=float(imbalance.get("ldam_max_margin", 0.3)),
        scale=float(train_config.get("ldam_scale", 18.0)),
    ).to(device)
    detection_criterion = _detection_criterion(class_count).to(device)
    pairwise_weight = float(train_config.get("pairwise_margin_loss_weight", 0.04))
    teacher_focus_weight = float(train_config.get("teacher_focus_binary_loss_weight", 0.015))
    teacher_focus_config = {
        "focus_class": int(train_config.get("teacher_focus_binary_focus_class", focus_class_index)),
        "classes": str(train_config.get("teacher_focus_binary_classes", "0,1,2,4")),
        "min_confidence": float(
            train_config.get("teacher_focus_binary_teacher_min_confidence", 0.0)
        ),
        "error_power": float(train_config.get("teacher_focus_binary_error_power", 0.0)),
        "hard_target_blend": float(
            train_config.get("teacher_focus_binary_hard_target_blend", 0.0) or 0.0
        ),
        "require_agreement": bool(
            train_config.get("teacher_focus_binary_require_agreement", True)
        ),
    }

    control_curve: list[Dict[str, float]] = []
    candidate_curve: list[Dict[str, float]] = []
    for batch_index in range(int(args.max_train_batches)):
        batch_rows = selected_pairs[
            batch_index * int(args.batch_size) : (batch_index + 1) * int(args.batch_size)
        ]
        _seed_everything(int(args.seed) + 1000 + batch_index)
        crop_images, labels, crop_metadata = _stack_classification_batch(
            classification_train,
            batch_rows,
        )
        full_images, detection_targets, full_masks = _stack_detection_batch(
            detection_train,
            [row.detection_index for row in batch_rows],
            device,
        )
        crop_images = crop_images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        crop_metadata = {
            key: value.to(device=device, non_blocking=True)
            for key, value in crop_metadata.items()
        }
        forward_seed = int(args.seed) + 100000 + batch_index
        _seed_everything(forward_seed)
        control_row = _control_step(
            control,
            teacher,
            optimizer=control_optimizer,
            primary_parameters=control_groups["primary_all"],
            images=crop_images,
            labels=labels,
            metadata=crop_metadata,
            criterion=classification_criterion,
            pairwise_weight=pairwise_weight,
            teacher_focus_weight=teacher_focus_weight,
            teacher_focus_config=teacher_focus_config,
            grad_clip_norm=float(args.grad_clip_norm),
        )
        _seed_everything(forward_seed)
        candidate_row = _candidate_step(
            candidate,
            teacher,
            primary_optimizer=candidate_primary_optimizer,
            detector_optimizer=candidate_detector_optimizer,
            parameter_groups=candidate_groups,
            crop_images=crop_images,
            labels=labels,
            crop_metadata=crop_metadata,
            full_images=full_images,
            detection_targets=detection_targets,
            full_image_masks=full_masks,
            classification_criterion=classification_criterion,
            detection_criterion=detection_criterion,
            pairwise_weight=pairwise_weight,
            teacher_focus_weight=teacher_focus_weight,
            teacher_focus_config=teacher_focus_config,
            auxiliary_ratio=float(args.auxiliary_gradient_ratio),
            grad_clip_norm=float(args.grad_clip_norm),
        )
        control_curve.append(control_row)
        candidate_curve.append(candidate_row)
        if (batch_index + 1) % 20 == 0 or batch_index == 0:
            print(
                {
                    "joint_spatial_pcgrad_smoke": {
                        "batch": batch_index + 1,
                        "control_primary_loss": control_row["total_primary_loss"],
                        "candidate_primary_loss": candidate_row["total_primary_loss"],
                        "det_total_loss": candidate_row["det_total_loss"],
                        "objectness_cosine": candidate_row["objectness_raw_cosine"],
                        "localization_cosine": candidate_row["localization_raw_cosine"],
                    }
                },
                flush=True,
            )

    export_train_config = {
        "method": "matched_control_joint_spatial_pcgrad_smoke",
        "epochs": 1,
        "max_train_batches": int(args.max_train_batches),
        "batch_size": int(args.batch_size),
        "primary_learning_rate": float(args.primary_learning_rate),
        "detector_head_learning_rate": float(args.detector_head_learning_rate),
        "auxiliary_gradient_ratio": float(args.auxiliary_gradient_ratio),
        "test_split_used": False,
        "raw_dataset_touched": False,
    }
    control_checkpoint = checkpoint_dir / "control_classifier.pt"
    candidate_checkpoint = checkpoint_dir / "candidate_classifier.pt"
    _export_classifier_checkpoint(
        control,
        reference=checkpoint,
        output_path=control_checkpoint,
        mode="matched_classification_only_control",
        train_config=export_train_config,
    )
    _export_classifier_checkpoint(
        candidate,
        reference=checkpoint,
        output_path=candidate_checkpoint,
        mode="component_normalized_spatial_pcgrad_candidate",
        train_config=export_train_config,
    )

    keeper_metrics = _evaluate_checkpoint(
        checkpoint_path,
        dataset=classification_val,
        batch_size=int(args.eval_batch_size),
        device=device,
        output_csv=output_dir / "keeper_val_predictions.csv",
    )
    control_metrics = _evaluate_checkpoint(
        control_checkpoint,
        dataset=classification_val,
        batch_size=int(args.eval_batch_size),
        device=device,
        output_csv=output_dir / "control_val_predictions.csv",
    )
    candidate_metrics = _evaluate_checkpoint(
        candidate_checkpoint,
        dataset=classification_val,
        batch_size=int(args.eval_batch_size),
        device=device,
        output_csv=output_dir / "candidate_val_predictions.csv",
    )
    transitions_vs_control = _transition_summary(
        candidate_metrics["labels"],
        control_metrics["predictions"],
        candidate_metrics["predictions"],
        focus_class_index=focus_class_index,
    )
    transitions_vs_keeper = _transition_summary(
        candidate_metrics["labels"],
        keeper_metrics["predictions"],
        candidate_metrics["predictions"],
        focus_class_index=focus_class_index,
    )
    gate = assess_smoke_gate(
        keeper_metrics=keeper_metrics,
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        transitions_vs_control=transitions_vs_control,
        focus_class_index=focus_class_index,
        test_split_used=False,
    )

    for checkpoint_file, metrics, mode in (
        (control_checkpoint, control_metrics, "control"),
        (candidate_checkpoint, candidate_metrics, "candidate"),
    ):
        payload = _load_checkpoint(checkpoint_file)
        payload["metrics"] = _compact_evaluation(metrics)
        payload["joint_spatial_pcgrad"]["smoke_gate"] = gate
        payload["joint_spatial_pcgrad"]["export_mode"] = mode
        torch.save(payload, checkpoint_file)

    training_summary = {
        "control": {
            "mean_primary_loss": _mean_telemetry(control_curve, "total_primary_loss"),
            "mean_gradient_norm": _mean_telemetry(control_curve, "gradient_norm"),
        },
        "candidate": {
            "mean_primary_loss": _mean_telemetry(candidate_curve, "total_primary_loss"),
            "mean_detection_loss": _mean_telemetry(candidate_curve, "det_total_loss"),
            "mean_objectness_cosine": _mean_telemetry(candidate_curve, "objectness_raw_cosine"),
            "mean_localization_cosine": _mean_telemetry(candidate_curve, "localization_raw_cosine"),
            "mean_objectness_retained": _mean_telemetry(
                candidate_curve,
                "objectness_retained_norm_ratio",
            ),
            "mean_localization_retained": _mean_telemetry(
                candidate_curve,
                "localization_retained_norm_ratio",
            ),
        },
    }
    summary = {
        "protocol": {
            "method": "matched_control_joint_spatial_pcgrad_smoke",
            "source_checkpoint": str(checkpoint_path),
            "data_yaml": str(data_yaml),
            "split_usage": {"train": True, "val": True, "test": False},
            "selected_single_object_sources": int(len(selected_stems)),
            "paired_train_rows": int(len(selected_pairs)),
            "epochs": 1,
            "max_train_batches": int(args.max_train_batches),
            "batch_size": int(args.batch_size),
            "matched_control": True,
            "same_classification_tensors": True,
            "same_classification_dropout_seed": True,
            "classification_objective": {
                "type": "keeper LDAM-focal + pairwise + frozen-keeper focus binary",
                "pairwise_weight": pairwise_weight,
                "teacher_focus_weight": teacher_focus_weight,
            },
            "candidate_auxiliary": {
                "full_frame_context_preserved": True,
                "components": ["objectness", "localization"],
                "detector_class_gradient_shared_with_encoder": False,
                "component_gradient_norm_balancing": True,
                "componentwise_pcgrad": True,
                "auxiliary_ratio": float(args.auxiliary_gradient_ratio),
            },
            "raw_dataset_touched": False,
            "test_split_used": False,
        },
        "joint_model": joint_summary,
        "head_warmup": warmup_summary,
        "encoder_integrity_after_head_warmup": encoder_integrity,
        "training": training_summary,
        "validation": {
            "keeper": _compact_evaluation(keeper_metrics),
            "control": _compact_evaluation(control_metrics),
            "candidate": _compact_evaluation(candidate_metrics),
        },
        "transitions": {
            "candidate_vs_control": transitions_vs_control,
            "candidate_vs_keeper": transitions_vs_keeper,
        },
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "checkpoint_written": True,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "artifact_manifest_path": str(
            (output_dir / "artifact_manifest.json").resolve()
        ),
    }
    _write_training_csv(
        output_dir / "training_telemetry.csv",
        control_curve,
        candidate_curve,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    readme = [
        "# Joint Spatial-PCGrad Matched-Control Smoke",
        "",
        f"- Keeper val macro/class1: `{float(keeper_metrics['macro_f1']):.6f}/{float(keeper_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Control val macro/class1: `{float(control_metrics['macro_f1']):.6f}/{float(control_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Candidate val macro/class1: `{float(candidate_metrics['macro_f1']):.6f}/{float(candidate_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Candidate vs control transitions: `{json.dumps(transitions_vs_control, sort_keys=True)}`",
        f"- Probe permission: `{str(bool(gate['probe_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "No raw dataset file was modified and test was not read.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = write_smoke_artifact_manifest(output_dir)
    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "transitions": summary["transitions"],
                "gate": gate,
                "elapsed_seconds": summary["elapsed_seconds"],
                "cuda_peak_memory_mb": summary["cuda_peak_memory_mb"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def _run_normalized_macrostep_smoke(args: argparse.Namespace) -> Dict[str, object]:
    start_time = time.perf_counter()
    if int(args.samples_per_class) < 4:
        raise ValueError("samples-per-class must be at least four")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if int(args.head_warmup_batches) < 20:
        raise ValueError("head-warmup-batches must be at least 20")
    if not 0.0 < float(args.auxiliary_gradient_ratio) <= 1.0:
        raise ValueError("auxiliary-gradient-ratio must be in (0, 1]")
    if float(args.step_parameter_ratio) <= 0.0:
        raise ValueError("step-parameter-ratio must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    torch.set_float32_matmul_precision("high")
    device = _resolve_device(str(args.device))
    checkpoint_path = Path(args.checkpoint).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    resolved_config = _load_resolved_config(checkpoint_path, args.resolved_config)
    class_names = [str(value) for value in checkpoint["class_names"]]
    class_count = len(class_names)
    focus_class_index = int(args.focus_class_index)
    if not 0 <= focus_class_index < class_count:
        raise ValueError("focus-class-index is outside the class range")

    classification_train, detection_train, view_protocol = _build_readiness_datasets(
        data_yaml=data_yaml,
        model_config=checkpoint["model_config"],
        resolved_config=resolved_config,
        class_count=class_count,
    )
    classification_val = _build_validation_dataset(
        data_yaml=data_yaml,
        checkpoint=checkpoint,
        resolved_config=resolved_config,
    )
    paired_sources = build_paired_single_object_sources(
        classification_train,
        detection_train,
    )
    selected_rows = _select_macrostep_rows(
        paired_sources,
        class_count=class_count,
        samples_per_class=int(args.samples_per_class),
        seed=int(args.seed) + 10,
    )
    selected_stems = {row.source_stem for row in selected_rows}
    warmup_indices = [
        index
        for index, sample in enumerate(detection_train.samples)
        if sample.image_path.stem.casefold() not in selected_stems
    ]

    teacher = build_model_from_checkpoint(checkpoint).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    candidate, joint_summary = _joint_model_from_checkpoint(
        checkpoint,
        num_queries=int(args.num_queries),
        decoder_depth=int(args.decoder_depth),
        decoder_ffn_dim=int(args.decoder_ffn_dim),
        decoder_dropout=float(args.decoder_dropout),
        seed=int(args.seed),
    )
    candidate = candidate.to(device)
    shared_names = _shared_parameter_names(teacher, candidate)
    encoder_reference = {
        name: candidate.state_dict()[name].detach().cpu().clone()
        for name in shared_names
    }
    warmup_curve, warmup_summary = _warm_up_detector_heads(
        candidate,
        dataset=detection_train,
        allowed_indices=warmup_indices,
        shared_names=shared_names,
        class_count=class_count,
        batch_size=int(args.head_warmup_batch_size),
        max_batches=int(args.head_warmup_batches),
        learning_rate=float(args.detector_head_learning_rate),
        weight_decay=float(args.detector_head_weight_decay),
        seed=int(args.seed) + 20,
        device=device,
    )
    encoder_integrity = _verify_encoder_unchanged(
        encoder_reference,
        candidate,
        shared_names,
    )
    if not encoder_integrity["passed"]:
        raise RuntimeError(f"Head warm-up modified the shared encoder: {encoder_integrity}")
    control = copy.deepcopy(candidate)
    candidate_shared = _set_shared_only_gradients(candidate, shared_names)
    control_shared = _set_shared_only_gradients(control, shared_names)

    gradients, gradient_protocol = _measure_normalized_macrostep_gradients(
        candidate,
        classification_dataset=classification_train,
        detection_dataset=detection_train,
        rows=selected_rows,
        shared_parameters=candidate_shared,
        class_count=class_count,
        batch_size=int(args.batch_size),
        device=device,
    )
    candidate_direction, projection = componentwise_spatial_pcgrad(
        gradients["primary"],
        gradients["objectness"],
        gradients["localization"],
        auxiliary_ratio=float(args.auxiliary_gradient_ratio),
    )
    control_update = _apply_normalized_parameter_step(
        control_shared,
        gradients["primary"],
        parameter_ratio=float(args.step_parameter_ratio),
    )
    candidate_update = _apply_normalized_parameter_step(
        candidate_shared,
        candidate_direction,
        parameter_ratio=float(args.step_parameter_ratio),
    )
    step_norm_delta = abs(
        float(candidate_update["actual_step_norm"])
        - float(control_update["actual_step_norm"])
    )
    if step_norm_delta > 1e-5:
        raise RuntimeError(
            f"Matched macro-step norms differ: control={control_update} candidate={candidate_update}"
        )

    export_train_config = {
        "method": "matched_normalized_encoder_macrostep",
        "update_mode": "normalized_macrostep",
        "samples_per_class": int(args.samples_per_class),
        "batch_size": int(args.batch_size),
        "step_parameter_ratio": float(args.step_parameter_ratio),
        "auxiliary_gradient_ratio": float(args.auxiliary_gradient_ratio),
        "test_split_used": False,
        "raw_dataset_touched": False,
    }
    control_checkpoint = checkpoint_dir / "control_classifier.pt"
    candidate_checkpoint = checkpoint_dir / "candidate_classifier.pt"
    _export_classifier_checkpoint(
        control,
        reference=checkpoint,
        output_path=control_checkpoint,
        mode="normalized_classification_gradient_control",
        train_config=export_train_config,
    )
    _export_classifier_checkpoint(
        candidate,
        reference=checkpoint,
        output_path=candidate_checkpoint,
        mode="normalized_component_spatial_pcgrad_candidate",
        train_config=export_train_config,
    )
    del control, candidate, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluations = _evaluate_checkpoints_shared_batches(
        {
            "keeper": checkpoint_path,
            "control": control_checkpoint,
            "candidate": candidate_checkpoint,
        },
        dataset=classification_val,
        batch_size=int(args.eval_batch_size),
        device=device,
        output_dir=output_dir,
    )
    keeper_metrics = evaluations["keeper"]
    control_metrics = evaluations["control"]
    candidate_metrics = evaluations["candidate"]
    transitions_vs_control = _transition_summary(
        candidate_metrics["labels"],
        control_metrics["predictions"],
        candidate_metrics["predictions"],
        focus_class_index=focus_class_index,
    )
    transitions_vs_keeper = _transition_summary(
        candidate_metrics["labels"],
        keeper_metrics["predictions"],
        candidate_metrics["predictions"],
        focus_class_index=focus_class_index,
    )
    gate = assess_smoke_gate(
        keeper_metrics=keeper_metrics,
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        transitions_vs_control=transitions_vs_control,
        focus_class_index=focus_class_index,
        test_split_used=False,
    )
    for checkpoint_file, metrics, mode in (
        (control_checkpoint, control_metrics, "control"),
        (candidate_checkpoint, candidate_metrics, "candidate"),
    ):
        payload = _load_checkpoint(checkpoint_file)
        payload["metrics"] = _compact_evaluation(metrics)
        payload["joint_spatial_pcgrad"]["smoke_gate"] = gate
        payload["joint_spatial_pcgrad"]["export_mode"] = mode
        torch.save(payload, checkpoint_file)

    selected_csv_rows = [
        {
            "source_stem": row.source_stem,
            "label": int(row.label),
            "class_name": class_names[int(row.label)],
            "classification_index": int(row.classification_index),
            "detection_index": int(row.detection_index),
        }
        for row in selected_rows
    ]
    with (output_dir / "macrostep_sources.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_csv_rows[0]))
        writer.writeheader()
        writer.writerows(selected_csv_rows)
    with (output_dir / "detector_head_warmup_curve.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(warmup_curve[0]))
        writer.writeheader()
        writer.writerows(warmup_curve)

    summary = {
        "protocol": {
            "method": "matched_normalized_encoder_macrostep_spatial_pcgrad_smoke",
            "source_checkpoint": str(checkpoint_path),
            "data_yaml": str(data_yaml),
            "split_usage": {"train": True, "val": True, "test": False},
            "view_protocol": view_protocol,
            "samples_per_class": int(args.samples_per_class),
            "macrostep_rows": int(len(selected_rows)),
            "microbatch_size": int(args.batch_size),
            "natural_class_weighted_gradient": True,
            "deterministic_views": True,
            "control_candidate_step_norm_matched": True,
            "step_parameter_ratio": float(args.step_parameter_ratio),
            "candidate_auxiliary": {
                "components": ["objectness", "localization"],
                "detector_class_gradient_shared_with_encoder": False,
                "component_gradient_norm_balancing": True,
                "componentwise_pcgrad": True,
                "auxiliary_ratio": float(args.auxiliary_gradient_ratio),
            },
            "raw_dataset_touched": False,
            "test_split_used": False,
        },
        "joint_model": joint_summary,
        "head_warmup": warmup_summary,
        "encoder_integrity_after_head_warmup": encoder_integrity,
        "gradient_protocol": gradient_protocol,
        "projection": projection,
        "updates": {
            "control": control_update,
            "candidate": candidate_update,
            "step_norm_absolute_delta": step_norm_delta,
        },
        "validation": {
            "keeper": _compact_evaluation(keeper_metrics),
            "control": _compact_evaluation(control_metrics),
            "candidate": _compact_evaluation(candidate_metrics),
        },
        "transitions": {
            "candidate_vs_control": transitions_vs_control,
            "candidate_vs_keeper": transitions_vs_keeper,
        },
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "checkpoint_written": True,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "artifact_manifest_path": str(
            (output_dir / "artifact_manifest.json").resolve()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    readme = [
        "# Normalized Macro-Step Spatial-PCGrad Smoke",
        "",
        f"- Keeper val macro/class1: `{float(keeper_metrics['macro_f1']):.6f}/{float(keeper_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Control val macro/class1: `{float(control_metrics['macro_f1']):.6f}/{float(control_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Candidate val macro/class1: `{float(candidate_metrics['macro_f1']):.6f}/{float(candidate_metrics['per_class'][focus_class_index]['f1']):.6f}`",
        f"- Candidate vs control transitions: `{json.dumps(transitions_vs_control, sort_keys=True)}`",
        f"- Probe permission: `{str(bool(gate['probe_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "The control and candidate update only the shared encoder with equal L2 step norm.",
        "No raw dataset file was modified and test was not read.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_smoke_artifact_manifest(output_dir)
    print(
        json.dumps(
            {
                "projection": projection,
                "updates": summary["updates"],
                "validation": summary["validation"],
                "transitions": summary["transitions"],
                "gate": gate,
                "elapsed_seconds": summary["elapsed_seconds"],
                "cuda_peak_memory_mb": summary["cuda_peak_memory_mb"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    if str(args.update_mode) == "adamw_microbatch":
        if not bool(args.allow_rejected_adamw_ablation):
            raise ValueError(
                "adamw_microbatch is a rejected ablation; pass "
                "--allow-rejected-adamw-ablation only to reproduce its failure"
            )
        return _run_adamw_microbatch_smoke(args)
    return _run_normalized_macrostep_smoke(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
