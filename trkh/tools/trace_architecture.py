from __future__ import annotations

import argparse
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
    _normalize_illumination_image,
    _pseudo_foreground_mask_array,
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
    )
    raw_transform = build_eval_transform(
        image_size=int(args.image_size),
        resize_mode=str(augmentation_config.get("resize_mode", "pad") or "pad"),
    )
    dataset = ClassificationFolderDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=transform,
        class_aware_augmentation=False,
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
        model_config = checkpoint.get("model_config", {})
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
        tensor, label = dataset[sample_index]
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
        model_input = tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            features = model.forward_features(
                model_input,
                return_trace=True,
            )
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
        input_image.save(class_dir / "01_model_input.png")
        stem = trace["stem_activation"][0, 0]
        _heatmap_image(stem, input_image.size).save(class_dir / "02_stem_activation.png")
        patch_norm = trace["patch_token_norm"][0].view(grid_size)
        _heatmap_image(patch_norm, input_image.size).save(class_dir / "03_patch_embedding_norm.png")
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
                _heatmap_image(surface_map[0], input_image.size).save(class_dir / filename)
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

        block_shapes = trace["block_token_shapes"]
        block_indices = trace["block_patch_indices"]
        block_norms = trace["block_patch_norms"]
        for block_index, (indices, norms) in enumerate(zip(block_indices, block_norms), start=1):
            block_map = _scatter_patch_values(indices[0], norms[0], grid_size)
            _heatmap_image(block_map, input_image.size).save(
                class_dir / f"block_{block_index:02d}_token_norm.png"
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
        surface_stats = trace.get("foreground_surface_stats")
        if torch.is_tensor(surface_stats):
            record["foreground_surface_stats_shape"] = list(surface_stats.shape)
        surface_weight_map = trace.get("foreground_surface_weight_map")
        if torch.is_tensor(surface_weight_map):
            record["foreground_surface_weight_map_shape"] = list(surface_weight_map.shape)
        surface_mask = trace.get("foreground_surface_mask")
        if torch.is_tensor(surface_mask):
            record["foreground_surface_mask_shape"] = list(surface_mask.shape)
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
        pairwise_route_weights = trace.get("pairwise_margin_route_weights")
        if torch.is_tensor(pairwise_route_weights):
            record["pairwise_margin_route_weights"] = [
                float(value)
                for value in pairwise_route_weights[0].detach().cpu().tolist()
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
    readme_lines = [
        "# TRKH 5-Class Architecture Trace",
        "",
        "Moi thu muc class chua mot anh train ngau nhien co seed co dinh va cac anh trung gian.",
        trace_description,
        "",
        "- `00_original.jpg`: anh crop goc.",
        "- `01a_resized_before_preprocess.png`: anh sau resize-pad, truoc normalization/loc nen.",
        "- `01b_illumination_normalized.png`: anh sau chuan hoa sang/toi.",
        "- `01c_foreground_mask_overlay.png`: vung xanh la pseudo foreground duoc giu.",
        "- `01_model_input.png`: anh sau resize-pad va denormalize de xem.",
        "- `02_stem_activation.png`: mean absolute activation cua CNN stem.",
        "- `03_patch_embedding_norm.png`: norm patch token truoc transformer.",
        "- `04_detail_map.png`: local color/high-frequency/edge map.",
        "- `05_foreground_prior.png`: prior dung cung attention khi xep hang token.",
        "- `06_attention_view_score.png`: score source theo train config ket hop foreground prior.",
        "- `07_attention_crop.png`: crop salient dung lam view phu khi train.",
        "- `08_attention_drop.png`: vung salient bi blur de ep model tim dau hieu phu.",
        "- `09a_foreground_surface_weight.png`: mask mem foreground-only cua surface fusion head.",
        "- `09b_foreground_surface_mask.png`: mask cung dung de cat nen cho audit map.",
        "- `09c_foreground_surface_edge_detail.png`: chi tiet cuc bo/edge sau khi mask foreground.",
        "- `09d_foreground_surface_dark_spot.png`: diem vet toi/underexposure/bam sau khi mask foreground.",
        "- `09e_foreground_surface_brown_spot.png`: diem vet nau/hu hong sau khi mask foreground.",
        "- `09f_foreground_surface_bright_spot.png`: diem qua sang/lo sang sau khi mask foreground.",
        "- `09g_bilinear_patch_attention.png`: trong so patch cua compact bilinear fusion sau pruning.",
        "- `09h_frequency_selective_votes.png`: ty le vote patch cua frequency-selective aggregation sau pruning.",
        "- `pairwise_margin_route_weights` trong `shapes.json`: trong so router cho tung pairwise specialist.",
        "- `block_XX_token_norm.png`: norm token sau tung transformer block; o da prune de trong.",
        "- `prune_XX_after_layer_Y.png`: patch xanh duoc giu, patch toi bi loai.",
        "- `shapes.json`: shape va patch index chi tiet.",
        "",
        f"Dataset: `{Path(args.data).resolve()}`",
        f"Seed: `{int(args.seed)}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "samples": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
