from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from trkh.core.config import IMAGENET_MEAN, IMAGENET_STD, ModelConfig, load_data_spec, project_dir, to_serializable
from trkh.core.utils import load_checkpoint
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    _amplify_surface_detail_image,
    _normalize_illumination_image,
    _pseudo_foreground_mask_array,
    _surface_detail_foreground_mask_array,
    build_eval_transform,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
)
from trkh.training.train import (
    _attention_crop_single,
    _attention_guided_score_map,
    _bounded_attention_drop_mask,
)


def _supports_trkh_feature_trace(model: torch.nn.Module) -> bool:
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        return False
    try:
        signature = inspect.signature(forward_features)
    except (TypeError, ValueError):
        return False
    return "return_trace" in signature.parameters


def _first_tensor(value: object) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _generic_trace_targets(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    targets: List[Tuple[str, torch.nn.Module]] = []
    patch_embed = getattr(model, "patch_embed", None)
    if isinstance(patch_embed, torch.nn.Module):
        targets.append(("patch_embed", patch_embed))
    for container_name in ("levels", "stages"):
        container = getattr(model, container_name, None)
        if isinstance(container, (torch.nn.ModuleList, torch.nn.Sequential)):
            targets.extend(
                (f"{container_name}_{index + 1:02d}", module)
                for index, module in enumerate(container)
            )
            break
    norm = getattr(model, "norm", None)
    if isinstance(norm, torch.nn.Module):
        targets.append(("norm", norm))
    if not targets:
        convolutions = [
            (name, module)
            for name, module in model.named_modules()
            if name and isinstance(module, torch.nn.Conv2d)
        ]
        if convolutions:
            targets.append(("first_conv", convolutions[0][1]))
            if len(convolutions) > 1:
                targets.append(("last_conv", convolutions[-1][1]))
    return targets


def _forward_generic_feature_trace(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                captured[name] = tensor.detach().float().cpu()

        return hook

    for name, module in _generic_trace_targets(model):
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        output = model(images)
    finally:
        for handle in handles:
            handle.remove()
    logits = _first_tensor(output)
    if logits is None or logits.ndim != 2:
        raise TypeError(
            "Generic architecture trace requires classifier logits shaped [batch, classes]."
        )
    return logits, captured


def _generic_spatial_activation_map(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor.ndim == 4:
        return tensor[0].pow(2).mean(dim=0).sqrt()
    if tensor.ndim == 3:
        token_count = int(tensor.size(1))
        side = int(round(token_count ** 0.5))
        if side * side == token_count:
            return tensor[0].pow(2).mean(dim=-1).sqrt().view(side, side)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace TRKH blocks tren mot anh ngau nhien moi class.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir().parent / "newdataset" / "class_f" / "data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir() / "docs" / "architecture_trace_5class",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default="raw",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=5,
        help="Set 0 to skip class-count validation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint TRKH tuy chon. Neu bo trong, trace dung khoi tao ngau nhien.",
    )
    return parser.parse_args()


def _proposed_model_config(image_size: int) -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=int(image_size),
        patch_size=16,
        stem_channels=32,
        cnn_feature_fusion=True,
        cnn_fusion_dropout=0.10,
        fine_grained_pooling=True,
        fine_grained_pooling_dropout=0.08,
        multi_branch_fusion=True,
        branch_color_tokens=1,
        branch_edge_tokens=1,
        branch_cnn_tokens=0,
        branch_token_dropout=0.08,
        detail_patch_enhancement=True,
        detail_patch_dropout=0.05,
        token_pruning=True,
        token_prune_layers="2,5",
        token_keep_rates="0.75,0.50",
        token_prune_foreground_weight=0.35,
        token_prune_bbox_weight=0.0,
        token_prune_bbox_margin_ratio=0.04,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        num_registers=4,
        dropout=0.12,
        attention_dropout=0.03,
        drop_path_rate=0.12,
        register_positional_embedding=True,
        head_pooling="cls_branch_register_mean",
    )


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    rgb = (tensor.detach().cpu().float() * std + mean).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _heatmap_image(values: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    array = values.detach().cpu().float().squeeze().numpy()
    finite = np.isfinite(array)
    if finite.any():
        minimum = float(array[finite].min())
        maximum = float(array[finite].max())
        array = (array - minimum) / max(maximum - minimum, 1e-8)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    red = np.clip(1.5 * array, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(array - 0.5) * 3.0, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - array), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    image = Image.fromarray((rgb * 255.0).round().astype(np.uint8))
    return image.resize(size, Image.Resampling.NEAREST)


def _overlay_kept_patches(
    image: Image.Image,
    kept_indices: Sequence[int],
    grid_size: Tuple[int, int],
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    grid_h, grid_w = grid_size
    kept = {int(index) for index in kept_indices}
    cell_w = float(canvas.width) / float(grid_w)
    cell_h = float(canvas.height) / float(grid_h)
    for row in range(grid_h):
        for col in range(grid_w):
            index = row * grid_w + col
            box = (
                int(round(col * cell_w)),
                int(round(row * cell_h)),
                int(round((col + 1) * cell_w)),
                int(round((row + 1) * cell_h)),
            )
            if index in kept:
                draw.rectangle(box, outline=(40, 255, 80, 230), width=2)
            else:
                draw.rectangle(box, fill=(0, 0, 0, 145), outline=(255, 80, 60, 120), width=1)
    return canvas.convert("RGB")


def _overlay_foreground_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    canvas = image.convert("RGBA")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 150)
    foreground = Image.new("RGBA", canvas.size, (35, 235, 85, 0))
    foreground.putalpha(mask_image)
    canvas = Image.alpha_composite(canvas, foreground)
    return canvas.convert("RGB")


def _overlay_normalized_box(
    image: Image.Image,
    box: Sequence[float],
    *,
    outline: Tuple[int, int, int] = (255, 190, 30),
    width: int = 3,
) -> Image.Image:
    if len(box) != 4:
        return image.copy()
    x0, y0, x1, y1 = [float(value) for value in box]
    left = int(round(x0 * image.width))
    top = int(round(y0 * image.height))
    right = int(round(x1 * image.width))
    bottom = int(round(y1 * image.height))
    left = max(0, min(image.width - 1, left))
    top = max(0, min(image.height - 1, top))
    right = max(left + 1, min(image.width, right))
    bottom = max(top + 1, min(image.height, bottom))
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((left, top, right, bottom), outline=outline, width=width)
    return canvas


def _scatter_patch_values(
    indices: torch.Tensor,
    values: torch.Tensor,
    grid_size: Tuple[int, int],
) -> torch.Tensor:
    flat = torch.full(
        (grid_size[0] * grid_size[1],),
        float("nan"),
        dtype=torch.float32,
    )
    flat[indices.detach().cpu().long()] = values.detach().cpu().float()
    return flat.view(grid_size)


def _shape_list(value: object) -> List[int]:
    return [int(item) for item in value] if isinstance(value, (tuple, list)) else []


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=int(args.expected_num_classes),
    )
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint is not None else None
    checkpoint = None
    augmentation_config: Dict[str, object] = {}
    train_config: Dict[str, object] = {}
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        raw_augmentation_config = checkpoint.get("augmentation_config", {})
        if isinstance(raw_augmentation_config, dict):
            augmentation_config = raw_augmentation_config
        raw_train_config = checkpoint.get("train_config", {})
        if isinstance(raw_train_config, dict):
            train_config = raw_train_config
    if checkpoint is not None:
        raw_model_config = checkpoint.get("model_config", {})
        model_config_for_dataset = (
            raw_model_config
            if isinstance(raw_model_config, dict)
            else to_serializable(raw_model_config)
        )
        if not isinstance(model_config_for_dataset, dict):
            model_config_for_dataset = {}
    else:
        model_config_for_dataset = to_serializable(
            _proposed_model_config(int(args.image_size))
        )
        if not isinstance(model_config_for_dataset, dict):
            model_config_for_dataset = {}

    transform = build_eval_transform(
        image_size=int(args.image_size),
        resize_mode=str(augmentation_config.get("resize_mode", "pad") or "pad"),
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
        background_suppression_blur_radius=float(
            augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0
        ),
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
        eval_surface_detail_amplification=bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
    )
    raw_transform = build_eval_transform(
        image_size=int(args.image_size),
        resize_mode=str(augmentation_config.get("resize_mode", "pad") or "pad"),
    )
    classification_source_context = bool(augmentation_config.get("classification_source_context", False))
    if data_spec.data_format == "classification_folder":
        dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=transform,
            class_aware_augmentation=False,
        )
    else:
        dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=transform,
            crop_margin_ratio=float(
                augmentation_config.get("crop_margin_ratio", 0.05) or 0.05
            ),
            crop_to_primary_object=not classification_source_context,
            classification_target=True,
            classification_object_crops=True,
            class_aware_augmentation=False,
            classification_source_context=classification_source_context,
            classification_source_context_mode=str(
                augmentation_config.get("classification_source_context_mode", "desaturate_blur")
                or "desaturate_blur"
            ),
            classification_source_context_layout=str(
                augmentation_config.get("classification_source_context_layout", "full")
                or "full"
            ),
            classification_source_context_margin_ratio=float(
                augmentation_config.get("classification_source_context_margin_ratio", 0.12)
                or 0.12
            ),
            classification_source_context_background_alpha=float(
                augmentation_config.get("classification_source_context_background_alpha", 0.35)
                or 0.35
            ),
            classification_source_context_blur_radius=float(
                augmentation_config.get("classification_source_context_blur_radius", 7.0)
                or 7.0
            ),
            classification_source_context_inset_scale=float(
                augmentation_config.get("classification_source_context_inset_scale", 0.34)
                or 0.34
            ),
            classification_bbox_metadata=bool(
                model_config_for_dataset.get("bbox_spatial_fusion", False)
                or train_config.get("bbox_token_prior_source", "bbox") == "crop_bbox"
            ),
        )
    class_to_indices: Dict[int, List[int]] = {index: [] for index in range(data_spec.num_classes)}
    for sample_index, label in enumerate(dataset.labels()):
        class_to_indices[int(label)].append(int(sample_index))
    missing = [index for index, indices in class_to_indices.items() if not indices]
    if missing:
        raise ValueError(f"Khong co train sample cho class: {missing}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if checkpoint_path is not None:
        if checkpoint is None:
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_class_names = list(checkpoint.get("class_names", []))
        if checkpoint_class_names and checkpoint_class_names != list(data_spec.class_names):
            raise ValueError("Class names trong checkpoint khong khop voi data.yaml.")
        model = build_model_from_checkpoint(
            checkpoint,
            num_classes=data_spec.num_classes,
            override_image_size=int(args.image_size),
        )
        model_config = model_config_for_dataset
        weights_description = str(checkpoint_path)
    else:
        model_config = _proposed_model_config(int(args.image_size))
        model = create_model(data_spec.num_classes, model_config=model_config)
        weights_description = "random_initialization_structural_trace"
    model = model.to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for class_index, class_name in enumerate(data_spec.class_names):
        sample_index = random.choice(class_to_indices[class_index])
        sample = dataset.samples[sample_index]
        sample_item = dataset[sample_index]
        if len(sample_item) == 3:
            tensor, label, sample_metadata = sample_item
        else:
            tensor, label = sample_item
            sample_metadata = {}
        input_image = _tensor_to_image(tensor)
        with Image.open(sample.image_path) as source:
            original = source.convert("RGB").copy()
        transform_meta: Dict[str, object] = {}
        dataset_transform = getattr(dataset, "transform", None)
        if dataset_transform is not None:
            try:
                transformed_with_meta = dataset_transform(original, return_meta=True)
                if (
                    isinstance(transformed_with_meta, tuple)
                    and len(transformed_with_meta) >= 2
                    and isinstance(transformed_with_meta[1], dict)
                ):
                    transform_meta = dict(transformed_with_meta[1])
            except TypeError:
                transform_meta = {}
        raw_tensor = raw_transform(original)
        raw_input_image = _tensor_to_image(raw_tensor)
        illumination_image = raw_input_image
        if bool(augmentation_config.get("illumination_normalization", False)):
            illumination_image = _normalize_illumination_image(
                raw_input_image,
                strength=float(
                    augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
                ),
            )
        foreground_mask = _pseudo_foreground_mask_array(
            illumination_image,
            margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        )
        surface_detail_mask = _surface_detail_foreground_mask_array(
            illumination_image,
            margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        )
        surface_detail_image = _amplify_surface_detail_image(
            illumination_image,
            mode=str(augmentation_config.get("surface_detail_amplification_mode", "none") or "none"),
            strength=float(augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0),
            blur_radius=float(
                augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
            ),
            foreground_margin=float(
                augmentation_config.get("background_suppression_margin", 0.08) or 0.08
            ),
            foreground_weight=float(
                augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85) or 0.85
            ),
        )
        model_input = tensor.unsqueeze(0).to(device)
        if not _supports_trkh_feature_trace(model):
            with torch.inference_mode():
                logits, generic_features = _forward_generic_feature_trace(
                    model,
                    model_input,
                )
            probabilities = torch.softmax(logits.float(), dim=-1)
            prediction = int(probabilities.argmax(dim=-1)[0].item())
            class_dir = output_dir / f"class_{class_index}_{class_name}"
            class_dir.mkdir(parents=True, exist_ok=True)
            original.save(class_dir / "00_original.jpg", quality=95)
            raw_input_image.save(class_dir / "01a_resized_before_preprocess.png")
            illumination_image.save(class_dir / "01b_illumination_normalized.png")
            _overlay_foreground_mask(illumination_image, foreground_mask).save(
                class_dir / "01c_foreground_mask_overlay.png"
            )
            _overlay_foreground_mask(illumination_image, surface_detail_mask).save(
                class_dir / "01d_surface_detail_mask_overlay.png"
            )
            surface_detail_image.save(class_dir / "01e_surface_detail_amplified.png")
            input_image.save(class_dir / "01_model_input.png")

            activation_stats: Dict[str, Dict[str, float]] = {}
            for feature_index, (feature_name, feature) in enumerate(
                generic_features.items(),
                start=1,
            ):
                activation_map = _generic_spatial_activation_map(feature)
                if activation_map is None:
                    continue
                activation_stats[feature_name] = {
                    "mean": float(activation_map.mean().item()),
                    "max": float(activation_map.max().item()),
                }
                safe_name = feature_name.replace(".", "_")
                _heatmap_image(activation_map, input_image.size).save(
                    class_dir / f"02_{feature_index:02d}_{safe_name}_activation.png"
                )

            record = {
                "trace_mode": "generic_feature_stages",
                "model_type": str(getattr(model, "model_type", type(model).__name__)),
                "class_id": int(class_index),
                "class_name": class_name,
                "sample_index": int(sample_index),
                "label_from_dataset": int(label),
                "prediction": prediction,
                "prediction_name": str(data_spec.class_names[prediction]),
                "logits": [float(value) for value in logits[0].detach().cpu().tolist()],
                "probabilities": [
                    float(value) for value in probabilities[0].detach().cpu().tolist()
                ],
                "source_image": str(sample.image_path.resolve()),
                "foreground_crop_box": transform_meta.get("foreground_crop_box"),
                "foreground_mask_fraction": float(foreground_mask.mean()),
                "surface_detail_mask_fraction": float(surface_detail_mask.mean()),
                "input_shape": list(model_input.shape),
                "feature_shapes": {
                    name: list(feature.shape) for name, feature in generic_features.items()
                },
                "spatial_activation_stats": activation_stats,
            }
            (class_dir / "shapes.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            records.append(record)
            continue
        with torch.inference_mode():
            bbox_value = sample_metadata.get("bbox") if isinstance(sample_metadata, dict) else None
            crop_bbox_value = sample_metadata.get("crop_bbox") if isinstance(sample_metadata, dict) else None
            bbox_token_prior = (
                bbox_value.unsqueeze(0).to(device=device, dtype=torch.float32)
                if torch.is_tensor(bbox_value)
                else None
            )
            if (
                str(train_config.get("bbox_token_prior_source", "bbox") or "bbox").strip().lower()
                == "crop_bbox"
                and torch.is_tensor(crop_bbox_value)
            ):
                bbox_token_prior = crop_bbox_value.unsqueeze(0).to(device=device, dtype=torch.float32)
            features = model.forward_features(
                model_input,
                bbox_token_prior=bbox_token_prior,
                return_trace=True,
            )
            if torch.is_tensor(bbox_value):
                features["bbox"] = bbox_value.unsqueeze(0).to(device=device, dtype=torch.float32)
            classification_logits_from_features(model, features)
            attention_score = _attention_guided_score_map(
                images=model_input,
                features=features,
                foreground_weight=float(
                    train_config.get("attention_view_foreground_weight", 0.40) or 0.40
                ),
                score_source=str(
                    train_config.get("attention_view_score_source", "learned_attention")
                    or "learned_attention"
                ),
            )
            attention_crop = _attention_crop_single(
                model_input[0],
                attention_score[0],
                threshold=float(train_config.get("attention_crop_threshold", 0.55) or 0.55),
                padding_ratio=float(
                    train_config.get("attention_crop_padding_ratio", 0.08) or 0.08
                ),
                min_area_ratio=float(
                    train_config.get("attention_crop_min_area_ratio", 0.25) or 0.25
                ),
            )
            blur_kernel = int(train_config.get("attention_drop_blur_kernel", 15) or 15)
            blurred = F.avg_pool2d(
                model_input,
                kernel_size=blur_kernel,
                stride=1,
                padding=blur_kernel // 2,
            )
            attention_drop_mask = _bounded_attention_drop_mask(
                attention_score,
                threshold=float(
                    train_config.get("attention_drop_threshold", 0.72) or 0.72
                ),
                dilation_kernel=int(
                    train_config.get("attention_drop_dilation_kernel", 5) or 5
                ),
                min_area_ratio=float(
                    train_config.get("attention_drop_min_area_ratio", 0.06) or 0.06
                ),
                max_area_ratio=float(
                    train_config.get("attention_drop_max_area_ratio", 0.16) or 0.16
                ),
            ).to(dtype=model_input.dtype)
            attention_drop = (
                model_input * (1.0 - attention_drop_mask)
                + blurred * attention_drop_mask
            )
        trace = features["trace"]
        grid_size = tuple(int(value) for value in features["grid_size"])
        class_dir = output_dir / f"class_{class_index}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        original.save(class_dir / "00_original.jpg", quality=95)
        raw_input_image.save(class_dir / "01a_resized_before_preprocess.png")
        illumination_image.save(class_dir / "01b_illumination_normalized.png")
        _overlay_foreground_mask(illumination_image, foreground_mask).save(
            class_dir / "01c_foreground_mask_overlay.png"
        )
        _overlay_foreground_mask(illumination_image, surface_detail_mask).save(
            class_dir / "01d_surface_detail_mask_overlay.png"
        )
        surface_detail_image.save(class_dir / "01e_surface_detail_amplified.png")
        input_image.save(class_dir / "01_model_input.png")
        stem = trace["stem_activation"][0, 0]
        _heatmap_image(stem, input_image.size).save(class_dir / "02_stem_activation.png")
        patch_norm = trace["patch_token_norm"][0].view(grid_size)
        _heatmap_image(patch_norm, input_image.size).save(class_dir / "03_patch_embedding_norm.png")
        shifted_patch_norm = trace.get("shifted_patch_residual_norm")
        if torch.is_tensor(shifted_patch_norm):
            _heatmap_image(shifted_patch_norm[0].view(grid_size), input_image.size).save(
                class_dir / "03b_shifted_patch_residual_norm.png"
            )
        detail_map = trace.get("detail_map")
        if torch.is_tensor(detail_map):
            _heatmap_image(detail_map[0, 0], input_image.size).save(class_dir / "04_detail_map.png")
        foreground_prior = trace["foreground_prior"][0].view(grid_size)
        _heatmap_image(foreground_prior, input_image.size).save(class_dir / "05_foreground_prior.png")
        _heatmap_image(attention_score[0, 0], input_image.size).save(
            class_dir / "06_attention_view_score.png"
        )
        _tensor_to_image(attention_crop).save(class_dir / "07_attention_crop.png")
        _tensor_to_image(attention_drop[0]).save(class_dir / "08_attention_drop.png")
        surface_map_specs = (
            ("09a_foreground_surface_weight.png", "foreground_surface_weight_map"),
            ("09b_foreground_surface_mask.png", "foreground_surface_mask"),
            ("09c_foreground_surface_edge_detail.png", "foreground_surface_edge_detail"),
            ("09d_foreground_surface_dark_spot.png", "foreground_surface_dark_spot"),
            ("09e_foreground_surface_brown_spot.png", "foreground_surface_brown_spot"),
            ("09f_foreground_surface_bright_spot.png", "foreground_surface_bright_spot"),
        )
        for filename, trace_key in surface_map_specs:
            surface_map = trace.get(trace_key)
            if torch.is_tensor(surface_map):
                heatmap_tensor = surface_map[0]
                if heatmap_tensor.ndim == 3:
                    heatmap_tensor = heatmap_tensor[0]
                _heatmap_image(heatmap_tensor, input_image.size).save(class_dir / filename)
        interior_boundary_map_specs = (
            ("09s_interior_boundary_foreground_mask.png", "interior_boundary_foreground_mask"),
            ("09t_interior_boundary_interior_mask.png", "interior_boundary_interior_mask"),
            ("09u_interior_boundary_boundary_mask.png", "interior_boundary_boundary_mask"),
            ("09v0_interior_boundary_bbox_mask.png", "interior_boundary_bbox_mask"),
            (
                "09v_interior_boundary_interior_weight.png",
                "interior_boundary_interior_weight_map",
            ),
            (
                "09w_interior_boundary_boundary_weight.png",
                "interior_boundary_boundary_weight_map",
            ),
            (
                "09x_interior_boundary_boundary_edge.png",
                "interior_boundary_boundary_edge_detail",
            ),
            (
                "09y_interior_boundary_dark_spot.png",
                "interior_boundary_interior_dark_spot",
            ),
            (
                "09z_interior_boundary_boundary_brown.png",
                "interior_boundary_boundary_brown_spot",
            ),
        )
        for filename, trace_key in interior_boundary_map_specs:
            interior_map = trace.get(trace_key)
            if torch.is_tensor(interior_map):
                heatmap_tensor = interior_map[0]
                if heatmap_tensor.ndim == 3:
                    heatmap_tensor = heatmap_tensor[0]
                _heatmap_image(heatmap_tensor, input_image.size).save(class_dir / filename)
        bilinear_attention = trace.get("bilinear_patch_attention")
        if torch.is_tensor(bilinear_attention):
            bilinear_map = _scatter_patch_values(
                features["patch_indices"][0],
                bilinear_attention[0],
                grid_size,
            )
            _heatmap_image(bilinear_map, input_image.size).save(
                class_dir / "09g_bilinear_patch_attention.png"
            )
        frequency_votes = trace.get("frequency_selective_vote_fraction")
        if torch.is_tensor(frequency_votes):
            frequency_vote_map = _scatter_patch_values(
                features["patch_indices"][0],
                frequency_votes[0],
                grid_size,
            )
            _heatmap_image(frequency_vote_map, input_image.size).save(
                class_dir / "09h_frequency_selective_votes.png"
            )
        micro_detail_attention = trace.get("micro_detail_attention")
        if torch.is_tensor(micro_detail_attention):
            micro_detail_map = _scatter_patch_values(
                features["patch_indices"][0],
                micro_detail_attention[0],
                grid_size,
            )
            _heatmap_image(micro_detail_map, input_image.size).save(
                class_dir / "09i_micro_detail_attention.png"
            )
        patch_objectness_attention = trace.get("patch_objectness_attention")
        if torch.is_tensor(patch_objectness_attention):
            patch_objectness_map = _scatter_patch_values(
                features["patch_indices"][0],
                patch_objectness_attention[0],
                grid_size,
            )
            _heatmap_image(patch_objectness_map, input_image.size).save(
                class_dir / "09i1_patch_objectness_attention.png"
            )
        patch_objectness_probability = trace.get("patch_objectness_probability")
        if torch.is_tensor(patch_objectness_probability):
            patch_objectness_probability_map = _scatter_patch_values(
                features["patch_indices"][0],
                patch_objectness_probability[0],
                grid_size,
            )
            _heatmap_image(patch_objectness_probability_map, input_image.size).save(
                class_dir / "09i2_patch_objectness_probability.png"
            )
        bbox_context_object_attention = trace.get("bbox_prior_patch_object_attention")
        if torch.is_tensor(bbox_context_object_attention):
            bbox_context_object_map = _scatter_patch_values(
                features["patch_indices"][0],
                bbox_context_object_attention[0],
                grid_size,
            )
            _heatmap_image(bbox_context_object_map, input_image.size).save(
                class_dir / "09i3_bbox_prior_object_attention.png"
            )
        bbox_context_background_attention = trace.get(
            "bbox_prior_patch_background_attention"
        )
        if torch.is_tensor(bbox_context_background_attention):
            bbox_context_background_map = _scatter_patch_values(
                features["patch_indices"][0],
                bbox_context_background_attention[0],
                grid_size,
            )
            _heatmap_image(bbox_context_background_map, input_image.size).save(
                class_dir / "09i4_bbox_prior_background_attention.png"
            )
        part_token_attention = trace.get("part_token_attention")
        if torch.is_tensor(part_token_attention) and part_token_attention.ndim == 3:
            part_mean_map = _scatter_patch_values(
                features["patch_indices"][0],
                part_token_attention[0].mean(dim=0),
                grid_size,
            )
            _heatmap_image(part_mean_map, input_image.size).save(
                class_dir / "09q_part_token_attention_mean.png"
            )
            part_count = int(part_token_attention.size(1))
            for part_index in range(part_count):
                part_map = _scatter_patch_values(
                    features["patch_indices"][0],
                    part_token_attention[0, part_index],
                    grid_size,
                )
                _heatmap_image(part_map, input_image.size).save(
                    class_dir / f"09q_part_token_{part_index + 1:02d}_attention.png"
                )
        part_token_pairwise_attention = trace.get("part_token_pairwise_attention")
        if torch.is_tensor(part_token_pairwise_attention) and part_token_pairwise_attention.ndim == 3:
            part_pairwise_mean_map = _scatter_patch_values(
                features["patch_indices"][0],
                part_token_pairwise_attention[0].mean(dim=0),
                grid_size,
            )
            _heatmap_image(part_pairwise_mean_map, input_image.size).save(
                class_dir / "09r_part_token_pairwise_attention_mean.png"
            )
            part_pairwise_count = int(part_token_pairwise_attention.size(1))
            for part_index in range(part_pairwise_count):
                part_pairwise_map = _scatter_patch_values(
                    features["patch_indices"][0],
                    part_token_pairwise_attention[0, part_index],
                    grid_size,
                )
                _heatmap_image(part_pairwise_map, input_image.size).save(
                    class_dir
                    / f"09r_part_token_pairwise_{part_index + 1:02d}_attention.png"
                )
        local_zoom_score = trace.get("local_zoom_score_map")
        if torch.is_tensor(local_zoom_score):
            _heatmap_image(local_zoom_score[0, 0], input_image.size).save(
                class_dir / "09j_local_zoom_score.png"
            )
        local_zoom_boxes = trace.get("local_zoom_crop_boxes")
        if torch.is_tensor(local_zoom_boxes):
            _overlay_normalized_box(
                input_image,
                local_zoom_boxes[0].detach().cpu().tolist(),
            ).save(class_dir / "09k_local_zoom_crop_box.png")
        high_frequency_maps = (
            ("09l_high_frequency_high_pass.png", "high_frequency_texture_high_pass"),
            ("09m_high_frequency_gradient.png", "high_frequency_texture_gradient"),
            ("09n_high_frequency_laplacian.png", "high_frequency_texture_laplacian"),
            (
                "09o_high_frequency_foreground_detail.png",
                "high_frequency_texture_foreground_detail",
            ),
            (
                "09p_high_frequency_foreground_weight.png",
                "high_frequency_texture_foreground_weight",
            ),
        )
        for filename, trace_key in high_frequency_maps:
            high_frequency_map = trace.get(trace_key)
            if torch.is_tensor(high_frequency_map):
                _heatmap_image(high_frequency_map[0, 0], input_image.size).save(
                    class_dir / filename
                )
        concurrent_local_activation = trace.get("concurrent_local_activation")
        if torch.is_tensor(concurrent_local_activation):
            _heatmap_image(
                concurrent_local_activation[0, 0],
                input_image.size,
            ).save(class_dir / "03c_concurrent_local_activation.png")

        block_shapes = trace["block_token_shapes"]
        block_indices = trace["block_patch_indices"]
        block_norms = trace["block_patch_norms"]
        for block_index, (indices, norms) in enumerate(zip(block_indices, block_norms), start=1):
            block_map = _scatter_patch_values(indices[0], norms[0], grid_size)
            _heatmap_image(block_map, input_image.size).save(
                class_dir / f"block_{block_index:02d}_token_norm.png"
            )
        relative_position_layers = trace.get("relative_position_attention_layers")
        relative_position_center_maps = trace.get("relative_position_attention_center_map")
        if torch.is_tensor(relative_position_layers) and torch.is_tensor(
            relative_position_center_maps
        ):
            for layer, center_map in zip(
                relative_position_layers.detach().cpu().tolist(),
                relative_position_center_maps,
            ):
                _heatmap_image(center_map.view(grid_size), input_image.size).save(
                    class_dir / f"07b_relative_position_layer_{int(layer):02d}_center.png"
                )

        pruning_records = []
        for prune_index, prune_info in enumerate(trace["pruning"], start=1):
            layer = int(prune_info["layer"].item())
            kept = prune_info["kept_indices"][0].detach().cpu().tolist()
            overlay = _overlay_kept_patches(input_image, kept, grid_size)
            overlay.save(class_dir / f"prune_{prune_index:02d}_after_layer_{layer}.png")
            pruning_records.append(
                {
                    "layer": layer,
                    "before_count": int(prune_info["before_count"].item()),
                    "after_count": int(prune_info["after_count"].item()),
                    "kept_indices": [int(value) for value in kept],
                }
            )

        record = {
            "class_id": int(class_index),
            "class_name": class_name,
            "label_from_dataset": int(label),
            "source_image": str(sample.image_path.resolve()),
            "foreground_crop_box": transform_meta.get("foreground_crop_box"),
            "foreground_mask_fraction": float(foreground_mask.mean()),
            "surface_detail_mask_fraction": float(surface_detail_mask.mean()),
            "attention_drop_area_fraction": float(
                attention_drop_mask.float().mean().item()
            ),
            "input_shape": _shape_list(trace["input_shape"]),
            "stem_shape": _shape_list(trace["stem_shape"]),
            "patch_embedding_shape": _shape_list(trace["patch_embedding_shape"]),
            "branch_token_shape": _shape_list(trace["branch_token_shape"]),
            "block_token_shapes": [_shape_list(shape) for shape in block_shapes],
            "final_patch_shape": list(features["patches"].shape),
            "final_token_shape": list(features["tokens"].shape),
            "grid_size": list(grid_size),
            "pruning": pruning_records,
        }
        shifted_patch_norm = trace.get("shifted_patch_residual_norm")
        if torch.is_tensor(shifted_patch_norm):
            record["shifted_patch_residual_norm_mean"] = float(
                shifted_patch_norm[0].float().mean().item()
            )
            record["shifted_patch_residual_norm_max"] = float(
                shifted_patch_norm[0].float().max().item()
            )
        if torch.is_tensor(relative_position_layers):
            record["relative_position_attention_layers"] = [
                int(value) for value in relative_position_layers.detach().cpu().tolist()
            ]
            for trace_key, record_key in (
                ("relative_position_attention_gate", "relative_position_attention_gate"),
                (
                    "relative_position_attention_raw_gate",
                    "relative_position_attention_raw_gate",
                ),
                (
                    "relative_position_attention_position_distance",
                    "relative_position_attention_position_distance",
                ),
                (
                    "relative_position_attention_content_distance",
                    "relative_position_attention_content_distance",
                ),
                (
                    "relative_position_attention_mixed_distance",
                    "relative_position_attention_mixed_distance",
                ),
                (
                    "relative_position_attention_local_mass",
                    "relative_position_attention_local_mass",
                ),
            ):
                values = trace.get(trace_key)
                if torch.is_tensor(values):
                    record[record_key] = values.detach().cpu().float().tolist()
        concurrent_layers = trace.get("concurrent_local_global_layers")
        if torch.is_tensor(concurrent_layers):
            record["concurrent_local_global_layers"] = [
                int(value) for value in concurrent_layers.detach().cpu().tolist()
            ]
            for trace_key, record_key in (
                ("concurrent_local_state_norm", "concurrent_local_state_norm"),
                ("concurrent_token_residual_norm", "concurrent_token_residual_norm"),
                ("concurrent_token_to_local_scale", "concurrent_token_to_local_scale"),
                ("concurrent_local_update_scale", "concurrent_local_update_scale"),
                ("concurrent_local_to_token_scale", "concurrent_local_to_token_scale"),
            ):
                values = trace.get(trace_key)
                if torch.is_tensor(values):
                    record[record_key] = values[0].detach().cpu().float().tolist()
            if torch.is_tensor(concurrent_local_activation):
                record["concurrent_local_activation_shape"] = list(
                    concurrent_local_activation.shape
                )
        surface_stats = trace.get("foreground_surface_stats")
        if torch.is_tensor(surface_stats):
            record["foreground_surface_stats_shape"] = list(surface_stats.shape)
        surface_pairwise_stats = trace.get("foreground_surface_pairwise_stats")
        if torch.is_tensor(surface_pairwise_stats):
            record["foreground_surface_pairwise_stats_shape"] = list(
                surface_pairwise_stats.shape
            )
        surface_weight_map = trace.get("foreground_surface_weight_map")
        if torch.is_tensor(surface_weight_map):
            record["foreground_surface_weight_map_shape"] = list(surface_weight_map.shape)
        surface_mask = trace.get("foreground_surface_mask")
        if torch.is_tensor(surface_mask):
            record["foreground_surface_mask_shape"] = list(surface_mask.shape)
        interior_boundary_stats = trace.get("interior_boundary_pairwise_stats")
        if torch.is_tensor(interior_boundary_stats):
            record["interior_boundary_pairwise_stats_shape"] = list(
                interior_boundary_stats.shape
            )
        interior_boundary_foreground = trace.get("interior_boundary_foreground_mask")
        if torch.is_tensor(interior_boundary_foreground):
            record["interior_boundary_foreground_mask_shape"] = list(
                interior_boundary_foreground.shape
            )
            record["interior_boundary_foreground_fraction"] = float(
                interior_boundary_foreground[0].detach().cpu().float().mean().item()
            )
        interior_boundary_interior = trace.get("interior_boundary_interior_mask")
        if torch.is_tensor(interior_boundary_interior):
            record["interior_boundary_interior_fraction"] = float(
                interior_boundary_interior[0].detach().cpu().float().mean().item()
            )
        interior_boundary_boundary = trace.get("interior_boundary_boundary_mask")
        if torch.is_tensor(interior_boundary_boundary):
            record["interior_boundary_boundary_fraction"] = float(
                interior_boundary_boundary[0].detach().cpu().float().mean().item()
            )
        interior_boundary_bbox_mask = trace.get("interior_boundary_bbox_mask")
        if torch.is_tensor(interior_boundary_bbox_mask):
            record["interior_boundary_bbox_fraction"] = float(
                interior_boundary_bbox_mask[0].detach().cpu().float().mean().item()
            )
        bilinear_descriptor = trace.get("bilinear_patch_descriptor")
        if torch.is_tensor(bilinear_descriptor):
            record["bilinear_patch_descriptor_shape"] = list(bilinear_descriptor.shape)
        if torch.is_tensor(bilinear_attention):
            record["bilinear_patch_attention_shape"] = list(bilinear_attention.shape)
        if torch.is_tensor(frequency_votes):
            record["frequency_selective_vote_shape"] = list(frequency_votes.shape)
            record["frequency_selective_vote_sum"] = float(
                frequency_votes[0].sum().item()
            )
            kept_foreground_prior = trace["foreground_prior"][0].gather(
                0,
                features["patch_indices"][0],
            )
            record["frequency_selective_foreground_prior_mean"] = float(
                (frequency_votes[0] * kept_foreground_prior).sum().item()
            )
            record["frequency_selective_foreground_vote_mass"] = float(
                frequency_votes[0][kept_foreground_prior >= 0.5].sum().item()
            )
        micro_detail_descriptor = trace.get("micro_detail_descriptor")
        if torch.is_tensor(micro_detail_descriptor):
            record["micro_detail_descriptor_shape"] = list(micro_detail_descriptor.shape)
        if torch.is_tensor(micro_detail_attention):
            record["micro_detail_attention_shape"] = list(micro_detail_attention.shape)
            record["micro_detail_attention_sum"] = float(
                micro_detail_attention[0].sum().item()
            )
        micro_detail_selected = trace.get("micro_detail_selected_indices")
        if torch.is_tensor(micro_detail_selected):
            record["micro_detail_selected_indices"] = [
                int(value)
                for value in micro_detail_selected[0].detach().cpu().tolist()
            ]
        micro_detail_route_weights = trace.get("micro_detail_route_weights")
        if torch.is_tensor(micro_detail_route_weights):
            record["micro_detail_route_weight"] = float(
                micro_detail_route_weights[0].detach().cpu().item()
            )
        patch_objectness_descriptor = trace.get("patch_objectness_descriptor")
        if torch.is_tensor(patch_objectness_descriptor):
            record["patch_objectness_descriptor_shape"] = list(
                patch_objectness_descriptor.shape
            )
        if torch.is_tensor(patch_objectness_attention):
            record["patch_objectness_attention_shape"] = list(
                patch_objectness_attention.shape
            )
            record["patch_objectness_attention_sum"] = float(
                patch_objectness_attention[0].detach().cpu().float().sum().item()
            )
        patch_objectness_logits = trace.get("patch_objectness_logits")
        if torch.is_tensor(patch_objectness_logits):
            record["patch_objectness_logits_shape"] = list(patch_objectness_logits.shape)
            record["patch_objectness_logit_mean"] = float(
                patch_objectness_logits[0].detach().cpu().float().mean().item()
            )
        if torch.is_tensor(patch_objectness_probability):
            patch_objectness_prob_cpu = (
                patch_objectness_probability[0].detach().cpu().float()
            )
            record["patch_objectness_probability_shape"] = list(
                patch_objectness_probability.shape
            )
            record["patch_objectness_probability_max"] = float(
                patch_objectness_prob_cpu.max().item()
            )
            record["patch_objectness_probability_mean"] = float(
                patch_objectness_prob_cpu.mean().item()
            )
        patch_objectness_stats = trace.get("patch_objectness_stats")
        if torch.is_tensor(patch_objectness_stats):
            record["patch_objectness_stats"] = [
                float(value)
                for value in patch_objectness_stats[0].detach().cpu().float().tolist()
            ]
        bbox_context_descriptor = trace.get("bbox_prior_patch_descriptor")
        if torch.is_tensor(bbox_context_descriptor):
            record["bbox_prior_patch_descriptor_shape"] = list(
                bbox_context_descriptor.shape
            )
        bbox_context_object_attention = trace.get("bbox_prior_patch_object_attention")
        if torch.is_tensor(bbox_context_object_attention):
            object_attention_cpu = (
                bbox_context_object_attention[0].detach().cpu().float()
            )
            record["bbox_prior_patch_object_attention_shape"] = list(
                bbox_context_object_attention.shape
            )
            record["bbox_prior_patch_object_attention_sum"] = float(
                object_attention_cpu.sum().item()
            )
            record["bbox_prior_patch_object_attention_max"] = float(
                object_attention_cpu.max().item()
            )
        bbox_context_background_attention = trace.get(
            "bbox_prior_patch_background_attention"
        )
        if torch.is_tensor(bbox_context_background_attention):
            background_attention_cpu = (
                bbox_context_background_attention[0].detach().cpu().float()
            )
            record["bbox_prior_patch_background_attention_shape"] = list(
                bbox_context_background_attention.shape
            )
            record["bbox_prior_patch_background_attention_sum"] = float(
                background_attention_cpu.sum().item()
            )
            record["bbox_prior_patch_background_attention_max"] = float(
                background_attention_cpu.max().item()
            )
        bbox_context_stats = trace.get("bbox_prior_patch_stats")
        if torch.is_tensor(bbox_context_stats):
            record["bbox_prior_patch_stats"] = [
                float(value)
                for value in bbox_context_stats[0].detach().cpu().float().tolist()
            ]
        part_token_descriptor = trace.get("part_token_descriptor")
        if torch.is_tensor(part_token_descriptor):
            record["part_token_descriptor_shape"] = list(part_token_descriptor.shape)
        if torch.is_tensor(part_token_attention):
            record["part_token_attention_shape"] = list(part_token_attention.shape)
            record["part_token_attention_sum_per_part"] = [
                float(value)
                for value in part_token_attention[0].detach().cpu().float().sum(dim=1).tolist()
            ]
        part_token_foreground_mass = trace.get("part_token_foreground_mass")
        if torch.is_tensor(part_token_foreground_mass):
            record["part_token_foreground_mass"] = [
                float(value)
                for value in part_token_foreground_mass[0].detach().cpu().float().tolist()
            ]
        part_token_max_weight = trace.get("part_token_max_weight")
        if torch.is_tensor(part_token_max_weight):
            record["part_token_max_weight"] = [
                float(value)
                for value in part_token_max_weight[0].detach().cpu().float().tolist()
            ]
        part_token_route_weights = trace.get("part_token_route_weights")
        if torch.is_tensor(part_token_route_weights):
            record["part_token_route_weight"] = float(
                part_token_route_weights[0].detach().cpu().item()
            )
        part_token_pairwise_descriptor = trace.get("part_token_pairwise_descriptor")
        if torch.is_tensor(part_token_pairwise_descriptor):
            record["part_token_pairwise_descriptor_shape"] = list(
                part_token_pairwise_descriptor.shape
            )
        if torch.is_tensor(part_token_pairwise_attention):
            record["part_token_pairwise_attention_shape"] = list(
                part_token_pairwise_attention.shape
            )
            record["part_token_pairwise_attention_sum_per_part"] = [
                float(value)
                for value in part_token_pairwise_attention[0]
                .detach()
                .cpu()
                .float()
                .sum(dim=1)
                .tolist()
            ]
        part_token_pairwise_foreground_mass = trace.get(
            "part_token_pairwise_foreground_mass"
        )
        if torch.is_tensor(part_token_pairwise_foreground_mass):
            record["part_token_pairwise_foreground_mass"] = [
                float(value)
                for value in part_token_pairwise_foreground_mass[0]
                .detach()
                .cpu()
                .float()
                .tolist()
            ]
        part_token_pairwise_max_weight = trace.get("part_token_pairwise_max_weight")
        if torch.is_tensor(part_token_pairwise_max_weight):
            record["part_token_pairwise_max_weight"] = [
                float(value)
                for value in part_token_pairwise_max_weight[0]
                .detach()
                .cpu()
                .float()
                .tolist()
            ]
        part_token_pairwise_route_weights = trace.get("part_token_pairwise_route_weights")
        if torch.is_tensor(part_token_pairwise_route_weights):
            record["part_token_pairwise_route_weights"] = [
                float(value)
                for value in part_token_pairwise_route_weights[0]
                .detach()
                .cpu()
                .float()
                .tolist()
            ]
        if torch.is_tensor(local_zoom_score):
            record["local_zoom_score_map_shape"] = list(local_zoom_score.shape)
            record["local_zoom_score_max"] = float(
                local_zoom_score[0].detach().cpu().float().max().item()
            )
            record["local_zoom_score_mean"] = float(
                local_zoom_score[0].detach().cpu().float().mean().item()
            )
        if torch.is_tensor(local_zoom_boxes):
            record["local_zoom_crop_box"] = [
                float(value) for value in local_zoom_boxes[0].detach().cpu().tolist()
            ]
        local_zoom_descriptor = trace.get("local_zoom_descriptor")
        if torch.is_tensor(local_zoom_descriptor):
            record["local_zoom_descriptor_shape"] = list(local_zoom_descriptor.shape)
        local_zoom_logits = trace.get("local_zoom_logits")
        if torch.is_tensor(local_zoom_logits):
            record["local_zoom_logits"] = [
                float(value) for value in local_zoom_logits[0].detach().cpu().tolist()
            ]
        local_zoom_route_weights = trace.get("local_zoom_route_weights")
        if torch.is_tensor(local_zoom_route_weights):
            record["local_zoom_route_weight"] = float(
                local_zoom_route_weights[0].detach().cpu().item()
            )
        high_frequency_descriptor = trace.get("high_frequency_texture_descriptor")
        if torch.is_tensor(high_frequency_descriptor):
            record["high_frequency_texture_descriptor_shape"] = list(
                high_frequency_descriptor.shape
            )
        high_frequency_logits = trace.get("high_frequency_texture_logits")
        if torch.is_tensor(high_frequency_logits):
            record["high_frequency_texture_logits"] = [
                float(value)
                for value in high_frequency_logits[0].detach().cpu().tolist()
            ]
        high_frequency_route_weights = trace.get(
            "high_frequency_texture_route_weights"
        )
        if torch.is_tensor(high_frequency_route_weights):
            record["high_frequency_texture_route_weight"] = float(
                high_frequency_route_weights[0].detach().cpu().item()
            )
        high_frequency_foreground_detail = trace.get(
            "high_frequency_texture_foreground_detail"
        )
        if torch.is_tensor(high_frequency_foreground_detail):
            record["high_frequency_texture_foreground_detail_shape"] = list(
                high_frequency_foreground_detail.shape
            )
            record["high_frequency_texture_foreground_detail_max"] = float(
                high_frequency_foreground_detail[0].detach().cpu().float().max().item()
            )
            record["high_frequency_texture_foreground_detail_mean"] = float(
                high_frequency_foreground_detail[0].detach().cpu().float().mean().item()
            )
        multi_granularity_layers = trace.get("multi_granularity_layers")
        if torch.is_tensor(multi_granularity_layers):
            record["multi_granularity_layers"] = [
                int(value)
                for value in multi_granularity_layers.detach().cpu().tolist()
            ]
        multi_granularity_logits = features.get("multi_granularity_logits")
        if isinstance(multi_granularity_logits, dict) and multi_granularity_logits:
            record["multi_granularity_logits_shapes"] = {
                str(layer): list(logits.shape)
                for layer, logits in multi_granularity_logits.items()
                if torch.is_tensor(logits)
            }
            record["multi_granularity_logits_sample0"] = {
                str(layer): [
                    float(value)
                    for value in logits[0].detach().cpu().tolist()
                ]
                for layer, logits in multi_granularity_logits.items()
                if torch.is_tensor(logits) and logits.ndim == 2 and logits.size(0) > 0
            }
        pairwise_route_weights = trace.get("pairwise_margin_route_weights")
        if torch.is_tensor(pairwise_route_weights):
            record["pairwise_margin_route_weights"] = [
                float(value)
                for value in pairwise_route_weights[0].detach().cpu().tolist()
            ]
        topk_reassessment_logits = features.get("topk_reassessment_logits")
        if torch.is_tensor(topk_reassessment_logits):
            record["topk_reassessment_logits"] = [
                float(value)
                for value in topk_reassessment_logits[0].detach().cpu().tolist()
            ]
        topk_reassessment_route_weights = trace.get("topk_reassessment_route_weights")
        if torch.is_tensor(topk_reassessment_route_weights):
            record["topk_reassessment_route_weight"] = float(
                topk_reassessment_route_weights[0].detach().cpu().item()
            )
        topk_reassessment_adjustment = trace.get("topk_reassessment_adjustment")
        if torch.is_tensor(topk_reassessment_adjustment):
            record["topk_reassessment_adjustment"] = [
                float(value)
                for value in topk_reassessment_adjustment[0].detach().cpu().tolist()
            ]
        cumulative_ordinal_logits = features.get("cumulative_ordinal_logits")
        if torch.is_tensor(cumulative_ordinal_logits):
            record["cumulative_ordinal_logits"] = [
                float(value)
                for value in cumulative_ordinal_logits[0].detach().cpu().tolist()
            ]
        bbox_spatial_logits = features.get("bbox_spatial_logits")
        if torch.is_tensor(bbox_spatial_logits):
            record["bbox_spatial_logits"] = [
                float(value)
                for value in bbox_spatial_logits[0].detach().cpu().tolist()
            ]
        bbox_spatial_stats = trace.get("bbox_spatial_stats")
        if torch.is_tensor(bbox_spatial_stats):
            record["bbox_spatial_stats"] = [
                float(value)
                for value in bbox_spatial_stats[0].detach().cpu().tolist()
            ]
        surface_pairwise_logits = features.get("foreground_surface_pairwise_logits")
        if torch.is_tensor(surface_pairwise_logits):
            record["foreground_surface_pairwise_logits"] = [
                float(value)
                for value in surface_pairwise_logits[0].detach().cpu().tolist()
            ]
        surface_pairwise_route_weights = trace.get(
            "foreground_surface_pairwise_route_weights"
        )
        if torch.is_tensor(surface_pairwise_route_weights):
            record["foreground_surface_pairwise_route_weights"] = [
                float(value)
                for value in surface_pairwise_route_weights[0].detach().cpu().tolist()
            ]
        interior_boundary_pairwise_logits = features.get("interior_boundary_pairwise_logits")
        if torch.is_tensor(interior_boundary_pairwise_logits):
            record["interior_boundary_pairwise_logits"] = [
                float(value)
                for value in interior_boundary_pairwise_logits[0].detach().cpu().tolist()
            ]
        interior_boundary_pairwise_route_weights = trace.get(
            "interior_boundary_pairwise_route_weights"
        )
        if torch.is_tensor(interior_boundary_pairwise_route_weights):
            record["interior_boundary_pairwise_route_weights"] = [
                float(value)
                for value in interior_boundary_pairwise_route_weights[0]
                .detach()
                .cpu()
                .tolist()
            ]
        (class_dir / "shapes.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        records.append(record)

    summary = {
        "data_yaml": str(Path(args.data).resolve()),
        "split_used": "train",
        "selection_seed": int(args.seed),
        "device": str(device),
        "weights": weights_description,
        "model_config": to_serializable(model_config),
        "samples": records,
    }
    (output_dir / "trace_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trace_description = (
        f"Trace dung checkpoint da train: `{checkpoint_path}`."
        if checkpoint_path is not None
        else "Trace dung khoi tao ngau nhien de kiem tra luong du lieu/shape; no khong dai dien cho attention sau huan luyen."
    )
    trkh_readme_lines = [
        "# TRKH 5-Class Architecture Trace",
        "",
        "Moi thu muc class chua mot anh train ngau nhien co seed co dinh va cac anh trung gian.",
        trace_description,
        "",
        "- `00_original.jpg`: anh crop goc.",
        "- `01a_resized_before_preprocess.png`: anh sau resize-pad, truoc normalization/loc nen.",
        "- `01b_illumination_normalized.png`: anh sau chuan hoa sang/toi.",
        "- `01c_foreground_mask_overlay.png`: vung xanh la pseudo foreground duoc giu.",
        "- `01d_surface_detail_mask_overlay.png`: mask chuyen dung cho surface-detail amplification.",
        "- `01e_surface_detail_amplified.png`: anh sau khuech dai residual/high-frequency foreground.",
        "- `01_model_input.png`: anh sau resize-pad va denormalize de xem.",
        "- `02_stem_activation.png`: mean absolute activation cua CNN stem.",
        "- `03_patch_embedding_norm.png`: norm patch token truoc transformer.",
        "- `03b_shifted_patch_residual_norm.png`: norm residual SPT bon huong tai patch embedding.",
        "- `04_detail_map.png`: local color/high-frequency/edge map.",
        "- `05_foreground_prior.png`: prior dung cung attention khi xep hang token.",
        "- `06_attention_view_score.png`: score source theo train config ket hop foreground prior.",
        "- `07_attention_crop.png`: crop salient dung lam view phu khi train.",
        "- `07b_relative_position_layer_XX_center.png`: GPSA-style positional map tu patch gan tam theo tung layer.",
        "- `08_attention_drop.png`: vung salient bi blur de ep model tim dau hieu phu.",
        "- `09a_foreground_surface_weight.png`: mask mem foreground-only cua surface fusion head.",
        "- `09b_foreground_surface_mask.png`: mask cung dung de cat nen cho audit map.",
        "- `09c_foreground_surface_edge_detail.png`: chi tiet cuc bo/edge sau khi mask foreground.",
        "- `09d_foreground_surface_dark_spot.png`: diem vet toi/underexposure/bam sau khi mask foreground.",
        "- `09e_foreground_surface_brown_spot.png`: diem vet nau/hu hong sau khi mask foreground.",
        "- `09f_foreground_surface_bright_spot.png`: diem qua sang/lo sang sau khi mask foreground.",
        "- `09g_bilinear_patch_attention.png`: trong so patch cua compact bilinear fusion sau pruning.",
        "- `09h_frequency_selective_votes.png`: ty le vote patch cua frequency-selective aggregation sau pruning.",
        "- `09i_micro_detail_attention.png`: top-K patch chi tiet foreground cua micro-detail expert.",
        "- `09i1`/`09i2`: attention va probability cua patch-objectness head hoc tu bbox prior.",
        "- `09i3`/`09i4`: object/background patch pooling co dinh theo bbox prior.",
        "- `09q_part_token_attention_mean.png` / `09q_part_token_XX_attention.png`: attention cua learned part-token tren patch foreground/bbox.",
        "- `09r_part_token_pairwise_attention_mean.png` / `09r_part_token_pairwise_XX_attention.png`: attention cua part-token pairwise margin head.",
        "- `09j_local_zoom_score.png` / `09k_local_zoom_crop_box.png`: score map va vung crop cua local-zoom expert.",
        "- `09l`-`09p`: ban do high-pass/gradient/laplacian va foreground-detail cua high-frequency texture expert.",
        "- `09s`-`09z`: bbox/foreground/interior-boundary mask/weight va boundary edge/brown maps cua pairwise head moi.",
        "- `multi_granularity_*` trong `shapes.json`: logits phu o cac transformer layer trung gian.",
        "- `pairwise_margin_route_weights` trong `shapes.json`: trong so router cho tung pairwise specialist.",
        "- `cumulative_ordinal_logits` trong `shapes.json`: threshold logits cho cac bien ordinal 0|1, 1|2, 2|3.",
        "- `block_XX_token_norm.png`: norm token sau tung transformer block; o da prune de trong.",
        "- `prune_XX_after_layer_Y.png`: patch xanh duoc giu, patch toi bi loai.",
        "- `shapes.json`: shape va patch index chi tiet.",
        "",
        f"Dataset: `{Path(args.data).resolve()}`",
        f"Seed: `{int(args.seed)}`",
    ]
    if any(record.get("trace_mode") == "generic_feature_stages" for record in records):
        readme_lines = [
            "# TRKH 5-Class Generic Architecture Trace",
            "",
            "Moi thu muc class chua mot anh train duoc chon bang seed co dinh.",
            trace_description,
            "",
            "- `00_original.jpg`: anh crop goc.",
            "- `01a`-`01e`: resize/preprocess va foreground/surface views.",
            "- `01_model_input.png`: model input da denormalize de xem.",
            "- `02_XX_*_activation.png`: RMS activation theo khong gian cua patch/stage hook.",
            "- `shapes.json`: logits, probabilities, prediction, feature shapes va activation stats.",
            "",
            "Generic stage activation is not causal attention. Use XAI/robustness audits for",
            "localization or shortcut claims.",
            "",
            f"Dataset: `{Path(args.data).resolve()}`",
            f"Seed: `{int(args.seed)}`",
        ]
    else:
        readme_lines = trkh_readme_lines
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "samples": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
