from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trkh.data.dataset import build_eval_transform
from trkh.models.feature_hooks import (
    HookRecorder,
    build_attention_heatmap,
    build_attention_rollout_heatmap,
    build_featuremap_heatmap,
    build_gradient_weighted_attention_rollout_heatmap,
    build_gradcam_heatmap,
    count_attention_layers,
    extract_optional_token_features,
    reconstruct_attention,
    resolve_attention_hook,
    resolve_attention_index,
    resolve_feature_hook,
    summarize_register_attention,
)
from trkh.inference.inference import crop_with_yolo_bbox, load_model
from trkh.models.model import extract_bbox_from_model_output
from trkh.core.utils import ensure_dir, json_dump, summarize_token_norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truc quan attention/feature-map/Grad-CAM cho classifier xoai.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--yolo-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("XC", "YC", "W", "H"),
    )
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--layer", type=int, default=-1, help="Layer attention, ho tro index am.")
    parser.add_argument("--head-reduction", choices=("mean", "max"), default="mean")
    parser.add_argument(
        "--query-tokens",
        choices=("cls", "registers", "cls_register_mean"),
        default="cls_register_mean",
        help="Token query dung de tong hop attention heatmap.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=("attention", "rollout", "grad_rollout", "gradcam", "both", "all"),
        default="both",
    )
    parser.add_argument("--feature-source", choices=("auto", "patch_embed", "stem_last", "last_conv"), default="auto")
    parser.add_argument("--rollout-start-layer", type=int, default=0)
    parser.add_argument("--target-class", type=int, default=None)
    return parser.parse_args()


def resolve_layer_index(layer_index: int, depth: int) -> int:
    resolved = layer_index if layer_index >= 0 else depth + layer_index
    if resolved < 0 or resolved >= depth:
        raise ValueError(f"Layer index khong hop le: {layer_index} voi depth={depth}")
    return resolved


def prepare_image_and_tensor(
    checkpoint: Dict[str, object],
    image_path: Path,
    device: torch.device,
    yolo_bbox: Optional[Tuple[float, float, float, float]] = None,
    crop_margin: float = 0.05,
) -> Tuple[Image.Image, torch.Tensor]:
    image_size = int(checkpoint["model_config"]["image_size"])
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=checkpoint.get("augmentation_config", {}).get("resize_mode", "pad"),
    )

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        if yolo_bbox is not None:
            image = crop_with_yolo_bbox(image, yolo_bbox, crop_margin_ratio=crop_margin)
        crop = image.copy()

    tensor = transform(crop).unsqueeze(0).to(device)
    return crop, tensor


def save_heatmap_visualizations(
    crop_image: Image.Image,
    heatmap,
    output_dir: Path,
    alpha: float,
    prefix: str,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    crop_path = output_dir / "crop.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    heatmap_path = output_dir / f"{prefix}_heatmap.png"

    crop_image.save(crop_path)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(crop_image)
    axis.imshow(heatmap, cmap="jet", alpha=alpha)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(overlay_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(heatmap, cmap="jet")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {
        "crop": str(crop_path.resolve()),
        "overlay": str(overlay_path.resolve()),
        "heatmap": str(heatmap_path.resolve()),
    }


def build_top_predictions(
    probabilities: torch.Tensor,
    class_names: Iterable[str],
    top_k: int,
) -> list[Dict[str, object]]:
    class_names = list(class_names)
    top_k = max(1, min(top_k, probabilities.numel()))
    top_values, top_indices = torch.topk(probabilities, k=top_k)
    predictions = []
    for score, index in zip(top_values.tolist(), top_indices.tolist()):
        predictions.append(
            {
                "class_index": index,
                "class_name": class_names[index],
                "probability": score,
            }
        )
    return predictions


def _pseudo_foreground_mask_from_crop(crop_image: Image.Image, margin: float = 0.08) -> np.ndarray:
    rgb = np.asarray(crop_image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    median_rgb = np.median(rgb.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    median_gray = float(np.median(gray))
    color_delta = np.abs(rgb - median_rgb).mean(axis=2)
    intensity_delta = np.abs(gray - median_gray)
    edge_delta = np.zeros_like(gray)
    edge_delta[:, 1:] = np.maximum(edge_delta[:, 1:], np.abs(gray[:, 1:] - gray[:, :-1]))
    edge_delta[1:, :] = np.maximum(edge_delta[1:, :], np.abs(gray[1:, :] - gray[:-1, :]))
    mask = (color_delta + intensity_delta + 0.5 * edge_delta) > max(0.0, float(margin))

    height, width = gray.shape
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, width)
    central_ellipse = ((xx / 0.82) ** 2 + (yy / 0.92) ** 2) <= 1.0
    mask = np.logical_or(mask, central_ellipse)
    if float(mask.mean()) < 0.08:
        mask = ((xx / 0.78) ** 2 + (yy / 0.90) ** 2) <= 1.0
    return mask


def summarize_heatmap_focus(
    heatmap,
    crop_image: Image.Image,
    *,
    foreground_margin: float = 0.08,
) -> Dict[str, float]:
    heat = np.asarray(heatmap, dtype=np.float64)
    heat = np.maximum(heat, 0.0)
    total = float(heat.sum())
    if total <= 1e-12:
        return {
            "foreground_mass": 0.0,
            "background_mass": 1.0,
            "border_mass": 0.0,
            "entropy": 0.0,
            "peak_x": 0.0,
            "peak_y": 0.0,
        }
    mass = heat / total
    foreground = _pseudo_foreground_mask_from_crop(crop_image, margin=foreground_margin)
    if foreground.shape != heat.shape:
        foreground = np.asarray(
            Image.fromarray((foreground.astype(np.uint8) * 255)).resize(
                (heat.shape[1], heat.shape[0]),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0
    height, width = heat.shape
    border_px = max(1, int(round(min(height, width) * 0.08)))
    border = np.zeros_like(foreground, dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    peak_flat = int(np.argmax(heat))
    peak_y, peak_x = divmod(peak_flat, max(1, width))
    entropy = float(-(mass * np.log(mass + 1e-12)).sum() / np.log(max(2, mass.size)))
    foreground_mass = float(mass[foreground].sum())
    return {
        "foreground_mass": foreground_mass,
        "background_mass": float(max(0.0, 1.0 - foreground_mass)),
        "border_mass": float(mass[border].sum()),
        "entropy": entropy,
        "peak_x": float(peak_x / max(1, width - 1)),
        "peak_y": float(peak_y / max(1, height - 1)),
    }


def _capture_forward(
    model,
    tensor: torch.Tensor,
    crop_image: Image.Image,
    layer_index: int,
    head_reduction: str,
    query_tokens: str,
    method: str,
    target_class: Optional[int],
    feature_source: str = "auto",
    rollout_start_layer: int = 0,
) -> Dict[str, object]:
    feature_spec = resolve_feature_hook(model, feature_source=feature_source)
    attention_spec = resolve_attention_hook(model, layer_index)
    need_grad = method in ("gradcam", "both", "all")
    need_rollout = method in ("rollout", "all")
    need_grad_rollout = method in ("grad_rollout", "all")

    with HookRecorder(feature_spec=feature_spec, attention_spec=attention_spec) as recorder:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad() if need_grad else torch.no_grad():
            logits, _ = extract_bbox_from_model_output(model(tensor))
            probabilities = F.softmax(logits.float(), dim=1)[0].detach().cpu()
            predicted_class = int(logits.argmax(dim=1)[0].item())
            selected_class = predicted_class if target_class is None else int(target_class)
            if need_grad:
                logits[:, selected_class].sum().backward()

    rollout_heatmap = None
    register_attention_summary = None
    if need_rollout and hasattr(model, "forward_features"):
        with torch.no_grad():
            features = model.forward_features(tensor, return_attention=True)
        attentions = features.get("attentions") if isinstance(features, dict) else None
        grid_size = features.get("grid_size") if isinstance(features, dict) else None
        prefix_tokens = attention_spec.prefix_tokens if attention_spec is not None else int(
            1 + int(getattr(model, "num_registers", 0))
        )
        if attentions and grid_size is not None:
            rollout_heatmap = build_attention_rollout_heatmap(
                attentions=attentions,
                grid_size=grid_size,
                prefix_tokens=prefix_tokens,
                output_size=crop_image.size,
                query_tokens=query_tokens,
                start_layer=rollout_start_layer,
            )
            register_attention_summary = summarize_register_attention(
                attentions=attentions,
                prefix_tokens=prefix_tokens,
            )

    grad_rollout_heatmap = None
    if need_grad_rollout and hasattr(model, "forward_features"):
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            features = model.forward_features(tensor, return_attention=True)
            attentions = features.get("attentions") if isinstance(features, dict) else None
            grid_size = features.get("grid_size") if isinstance(features, dict) else None
            prefix_tokens = attention_spec.prefix_tokens if attention_spec is not None else int(
                1 + int(getattr(model, "num_registers", 0))
            )
            if attentions and grid_size is not None:
                for attention in attentions.values() if isinstance(attentions, dict) else attentions:
                    if torch.is_tensor(attention):
                        attention.retain_grad()
                if hasattr(model, "head_input_from_features") and hasattr(model, "head"):
                    logits = model.head(model.head_input_from_features(features))
                    if getattr(model, "cnn_fusion_head", None) is not None and "cnn_pooled" in features:
                        cnn_features = model.cnn_fusion_norm(features["cnn_pooled"])
                        cnn_features = model.cnn_fusion_dropout(cnn_features)
                        logits = logits + model.cnn_fusion_head(cnn_features)
                elif hasattr(model, "forward_heads"):
                    logits, _ = extract_bbox_from_model_output(model.forward_heads(features))
                else:
                    logits, _ = extract_bbox_from_model_output(model(tensor))
                selected_class = int(logits.argmax(dim=1)[0].item()) if target_class is None else int(target_class)
                logits[:, selected_class].sum().backward()
                grad_rollout_heatmap = build_gradient_weighted_attention_rollout_heatmap(
                    attentions=attentions,
                    grid_size=grid_size,
                    prefix_tokens=prefix_tokens,
                    output_size=crop_image.size,
                    query_tokens=query_tokens,
                    start_layer=rollout_start_layer,
                )
                if register_attention_summary is None:
                    register_attention_summary = summarize_register_attention(
                        attentions=attentions,
                        prefix_tokens=prefix_tokens,
                    )

    result: Dict[str, object] = {
        "probabilities": probabilities,
        "predicted_class": predicted_class,
        "selected_class": selected_class,
        "feature_source": feature_spec.source,
        "grid_size": list(recorder.grid_size) if recorder.grid_size is not None else None,
    }
    if rollout_heatmap is not None:
        result["rollout_heatmap"] = rollout_heatmap
    if grad_rollout_heatmap is not None:
        result["grad_rollout_heatmap"] = grad_rollout_heatmap
    if register_attention_summary:
        result["register_attention"] = register_attention_summary

    if recorder.activations is not None and method in ("attention", "both", "all"):
        attention_grid_size = None
        if attention_spec is not None and recorder.qkv_output is not None:
            num_tokens = int(recorder.qkv_output.shape[1])
            patch_tokens = max(0, num_tokens - int(attention_spec.prefix_tokens))
            side = int(round(patch_tokens ** 0.5))
            if side * side == patch_tokens:
                attention_grid_size = (side, side)
        if attention_spec is not None and recorder.qkv_output is not None and attention_grid_size is not None:
            attention = reconstruct_attention(recorder.qkv_output, attention_spec)[0].detach().cpu()
            result["attention_heatmap"] = build_attention_heatmap(
                attention=attention,
                grid_size=attention_grid_size,
                prefix_tokens=attention_spec.prefix_tokens,
                reduction=head_reduction,
                output_size=crop_image.size,
                query_tokens=query_tokens,
            )
            result["attention_source"] = attention_spec.source
            result["attention_grid_size"] = list(attention_grid_size)
        else:
            result["attention_heatmap"] = build_featuremap_heatmap(
                recorder.activations.detach(),
                output_size=crop_image.size,
                reduction=head_reduction,
            )
            result["attention_source"] = "feature_map_fallback"

    if recorder.activations is not None and recorder.gradients is not None and method in ("gradcam", "both", "all"):
        result["gradcam_heatmap"] = build_gradcam_heatmap(
            activations=recorder.activations.detach(),
            gradients=recorder.gradients,
            output_size=crop_image.size,
        )
    return result


def analyze_tensor(
    model,
    class_names,
    crop_image: Image.Image,
    tensor: torch.Tensor,
    layer_index: int,
    head_reduction: str,
    alpha: float,
    top_k: int,
    output_dir: Path,
    method: str = "both",
    target_class: Optional[int] = None,
    query_tokens: str = "cls_register_mean",
    feature_source: str = "auto",
    rollout_start_layer: int = 0,
) -> Dict[str, object]:
    capture = _capture_forward(
        model=model,
        tensor=tensor,
        crop_image=crop_image,
        layer_index=layer_index,
        head_reduction=head_reduction,
        query_tokens=query_tokens,
        method=method,
        target_class=target_class,
        feature_source=feature_source,
        rollout_start_layer=rollout_start_layer,
    )
    optional_token_features = extract_optional_token_features(model, tensor)

    result: Dict[str, object] = {
        "layer_index": layer_index,
        "head_reduction": head_reduction,
        "query_tokens": query_tokens,
        "method": method,
        "feature_source": capture["feature_source"],
        "rollout_start_layer": int(rollout_start_layer),
        "files": {},
        "predictions": build_top_predictions(capture["probabilities"], class_names, top_k),
    }
    if capture.get("grid_size") is not None:
        result["grid_size"] = capture["grid_size"]
    if optional_token_features is not None:
        artifact_stats = summarize_token_norms(optional_token_features)
        if artifact_stats:
            result["artifact_stats"] = artifact_stats

    heatmap_focus: Dict[str, Dict[str, float]] = {}

    if method in ("attention", "both", "all") and "attention_heatmap" in capture:
        result["attention_source"] = capture.get("attention_source")
        result["files"]["attention"] = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=capture["attention_heatmap"],
            output_dir=output_dir,
            alpha=alpha,
            prefix="attention",
        )
        heatmap_focus["attention"] = summarize_heatmap_focus(capture["attention_heatmap"], crop_image)

    if method in ("rollout", "all") and "rollout_heatmap" in capture:
        result["files"]["rollout"] = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=capture["rollout_heatmap"],
            output_dir=output_dir,
            alpha=alpha,
            prefix="rollout",
        )
        heatmap_focus["rollout"] = summarize_heatmap_focus(capture["rollout_heatmap"], crop_image)

    if method in ("grad_rollout", "all") and "grad_rollout_heatmap" in capture:
        result["files"]["grad_rollout"] = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=capture["grad_rollout_heatmap"],
            output_dir=output_dir,
            alpha=alpha,
            prefix="grad_rollout",
        )
        heatmap_focus["grad_rollout"] = summarize_heatmap_focus(capture["grad_rollout_heatmap"], crop_image)

    if method in ("gradcam", "both", "all") and "gradcam_heatmap" in capture:
        result["gradcam"] = {
            "predicted_class": capture["predicted_class"],
            "selected_class": capture["selected_class"],
        }
        result["files"]["gradcam"] = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=capture["gradcam_heatmap"],
            output_dir=output_dir,
            alpha=alpha,
            prefix="gradcam",
        )
        heatmap_focus["gradcam"] = summarize_heatmap_focus(capture["gradcam_heatmap"], crop_image)

    if heatmap_focus:
        result["heatmap_focus"] = heatmap_focus
    if "register_attention" in capture:
        result["register_attention"] = capture["register_attention"]

    if method == "attention" and "attention" in result["files"]:
        result["files"]["crop"] = result["files"]["attention"]["crop"]
    elif method == "rollout" and "rollout" in result["files"]:
        result["files"]["crop"] = result["files"]["rollout"]["crop"]
    elif method == "grad_rollout" and "grad_rollout" in result["files"]:
        result["files"]["crop"] = result["files"]["grad_rollout"]["crop"]
    elif method == "gradcam" and "gradcam" in result["files"]:
        result["files"]["crop"] = result["files"]["gradcam"]["crop"]
    elif "attention" in result["files"]:
        result["files"]["crop"] = result["files"]["attention"]["crop"]
    elif "rollout" in result["files"]:
        result["files"]["crop"] = result["files"]["rollout"]["crop"]
    elif "gradcam" in result["files"]:
        result["files"]["crop"] = result["files"]["gradcam"]["crop"]
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(
        args.checkpoint,
        device,
        override_image_size=args.override_image_size,
    )
    checkpoint["model_config"]["image_size"] = int(
        args.override_image_size or checkpoint["model_config"]["image_size"]
    )

    attention_depth = count_attention_layers(model)
    if attention_depth > 0:
        layer_index = resolve_layer_index(args.layer, attention_depth)
    else:
        layer_index = 0

    crop_image, tensor = prepare_image_and_tensor(
        checkpoint=checkpoint,
        image_path=args.image,
        device=device,
        yolo_bbox=tuple(args.yolo_bbox) if args.yolo_bbox is not None else None,
        crop_margin=args.crop_margin,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.resolve().parent.parent / "attention_viz"
    result = analyze_tensor(
        model=model,
        class_names=class_names,
        crop_image=crop_image,
        tensor=tensor,
        layer_index=layer_index,
        head_reduction=args.head_reduction,
        query_tokens=args.query_tokens,
        alpha=args.alpha,
        top_k=args.top_k,
        output_dir=output_dir,
        method=args.method,
        target_class=args.target_class,
        feature_source=args.feature_source,
        rollout_start_layer=args.rollout_start_layer,
    )
    result["image_path"] = str(args.image.resolve())
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["attention_depth"] = attention_depth

    json_dump(output_dir / "attention_summary.json", result)
    print(result)


if __name__ == "__main__":
    main()
