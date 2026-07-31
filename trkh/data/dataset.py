from __future__ import annotations

import logging
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from trkh.core.config import IMAGENET_MEAN, IMAGENET_STD, DataSpec, load_data_spec

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSIFICATION_CROP_NAME = re.compile(r"^(?P<source>.+)_box(?P<box>\d+)$", re.IGNORECASE)
logger = logging.getLogger(__name__)
PIL_RESAMPLE_MAP = {
    InterpolationMode.NEAREST: Image.Resampling.NEAREST,
    InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
    InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
    InterpolationMode.BOX: Image.Resampling.BOX,
    InterpolationMode.HAMMING: Image.Resampling.HAMMING,
    InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
}


def normalize_class_conditional_augmentation_scales(
    values: Optional[Sequence[float]],
    *,
    num_classes: int,
) -> List[float]:
    if values is None or len(values) == 0:
        return []
    parsed = [float(value) for value in values]
    if len(parsed) != int(num_classes):
        raise ValueError(
            "class_conditional_augmentation_scales phai co dung "
            f"{int(num_classes)} gia tri; nhan duoc {len(parsed)}."
        )
    for index, value in enumerate(parsed):
        if not math.isfinite(value) or not 0.1 <= value <= 1.0:
            raise ValueError(
                "class_conditional_augmentation_scales chi nhan gia tri huu han "
                f"trong [0.1, 1.0]; class {index}={value}."
            )
    return parsed


def _imagenet_fill(mean: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(channel * 255.0)) for channel in mean)


def _pseudo_foreground_mask_array(image: Image.Image, margin: float = 0.08) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        return np.ones((max(1, image.height), max(1, image.width)), dtype=bool)
    height, width = rgb.shape[:2]
    gray = rgb.mean(axis=2)
    fill_rgb = np.asarray(_imagenet_fill(IMAGENET_MEAN), dtype=np.float32).reshape(1, 1, 3) / 255.0
    median_rgb = np.median(rgb.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    median_gray = float(np.median(gray))
    color_delta = np.abs(rgb - median_rgb).mean(axis=2)
    intensity_delta = np.abs(gray - median_gray)
    edge_delta = np.zeros_like(gray, dtype=np.float32)
    edge_delta[:, 1:] = np.maximum(edge_delta[:, 1:], np.abs(gray[:, 1:] - gray[:, :-1]))
    edge_delta[1:, :] = np.maximum(edge_delta[1:, :], np.abs(gray[1:, :] - gray[:-1, :]))
    score = color_delta + intensity_delta + 0.5 * edge_delta

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    delta = max_channel - min_channel
    saturation = np.where(max_channel > 1e-6, delta / np.maximum(max_channel, 1e-6), 0.0)
    hue = np.zeros_like(max_channel, dtype=np.float32)
    non_gray = delta > 1e-6
    red_is_max = (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 0] >= rgb[..., 2])
    green_is_max = (rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] >= rgb[..., 2])
    blue_is_max = ~(red_is_max | green_is_max)
    hue[red_is_max & non_gray] = ((rgb[..., 1] - rgb[..., 2]) / np.maximum(delta, 1e-6))[red_is_max & non_gray] % 6.0
    hue[green_is_max & non_gray] = ((rgb[..., 2] - rgb[..., 0]) / np.maximum(delta, 1e-6) + 2.0)[green_is_max & non_gray]
    hue[blue_is_max & non_gray] = ((rgb[..., 0] - rgb[..., 1]) / np.maximum(delta, 1e-6) + 4.0)[blue_is_max & non_gray]
    hue = (hue / 6.0).astype(np.float32)

    fill_delta = np.abs(rgb - fill_rgb).mean(axis=2)
    padding_like = (fill_delta < 0.035) & (edge_delta < 0.025)
    not_padding = ~padding_like

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx + 0.5) / max(1.0, float(width)) * 2.0 - 1.0
    yy = (yy + 0.5) / max(1.0, float(height)) * 2.0 - 1.0
    central = ((xx / 0.92) ** 2 + (yy / 0.96) ** 2) <= 1.0

    green_yellow = (hue >= 0.08) & (hue <= 0.45) & (saturation >= 0.07) & (max_channel >= 0.12)
    brown_or_orange = (hue >= 0.035) & (hue <= 0.17) & (saturation >= 0.10) & (max_channel >= 0.10)
    dark_defect = (max_channel <= 0.45) & (saturation >= 0.08) & (score > max(0.02, float(margin) * 0.35))
    detail_center = central & (score > max(0.035, float(margin)))
    mask = not_padding & (green_yellow | brown_or_orange | dark_defect | detail_center)
    if float(mask.mean()) > 0.92:
        mask = not_padding & (
            green_yellow
            | brown_or_orange
            | dark_defect
            | (central & (score > max(0.05, float(margin) * 1.25)))
        )
    if float(mask.mean()) < 0.08:
        fallback = not_padding & central
        mask = fallback if float(fallback.mean()) >= 0.03 else not_padding
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
    mask = np.asarray(mask_image, dtype=np.uint8) > 127
    return mask


def _soft_mask_image(mask: np.ndarray, radius: float = 3.0) -> Image.Image:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    image = Image.fromarray(mask_uint8, mode="L")
    if radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return image


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not bool(mask.any()):
        return mask
    try:
        import cv2  # type: ignore
    except Exception:
        return mask
    labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return mask
    counts = np.bincount(labels.reshape(-1), minlength=labels_count)
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    if largest_label <= 0 or int(counts[largest_label]) <= 0:
        return mask
    return labels == largest_label


def _grabcut_foreground_mask_array(
    image: Image.Image,
    margin: float = 0.08,
    iterations: int = 2,
) -> np.ndarray:
    """Refine the pseudo foreground mask with classical GrabCut.

    This stays pretrained-free. If OpenCV is unavailable or GrabCut returns an
    unstable mask, the function falls back to the existing pseudo mask.
    """
    base = image.convert("RGB")
    pseudo_mask = _pseudo_foreground_mask_array(base, margin=margin)
    try:
        import cv2  # type: ignore
    except Exception:
        return pseudo_mask

    rgb = np.asarray(base, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        return pseudo_mask
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return pseudo_mask

    rgb_float = rgb.astype(np.float32) / 255.0
    fill_rgb = np.asarray(_imagenet_fill(IMAGENET_MEAN), dtype=np.float32).reshape(1, 1, 3) / 255.0
    fill_delta = np.abs(rgb_float - fill_rgb).mean(axis=2)
    not_padding = fill_delta >= 0.035

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx_norm = (xx + 0.5) / max(1.0, float(width)) * 2.0 - 1.0
    yy_norm = (yy + 0.5) / max(1.0, float(height)) * 2.0 - 1.0
    central = ((xx_norm / 0.88) ** 2 + (yy_norm / 0.92) ** 2) <= 1.0

    init_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    init_mask[pseudo_mask & not_padding] = cv2.GC_PR_FGD

    kernel_size = max(3, int(round(min(height, width) * 0.025)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    core = cv2.erode((pseudo_mask & central & not_padding).astype(np.uint8), kernel, iterations=1) > 0
    min_core_pixels = max(16, int(round(height * width * 0.005)))
    if int(core.sum()) < min_core_pixels:
        core = pseudo_mask & central & not_padding
    if int(core.sum()) < min_core_pixels:
        return pseudo_mask
    init_mask[core] = cv2.GC_FGD

    border = np.zeros((height, width), dtype=bool)
    border_width = max(2, int(round(min(height, width) * 0.025)))
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    init_mask[(~not_padding) | (border & ~central)] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            init_mask,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except Exception:
        return pseudo_mask

    refined = ((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD)) & not_padding
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) > 0
    refined = _largest_connected_component(refined)

    refined_area = float(refined.mean())
    pseudo_area = max(1e-6, float(pseudo_mask.mean()))
    if refined_area < 0.04 or refined_area > 0.90:
        return pseudo_mask
    if refined_area < pseudo_area * 0.20:
        return pseudo_mask
    return refined


def _normalize_illumination_image(image: Image.Image, strength: float = 0.35) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    low, high = np.percentile(luminance, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or float(high - low) < 1e-4:
        return image
    stretched = np.clip((luminance - low) / max(float(high - low), 1e-4), 0.0, 1.0)
    target = (1.0 - strength) * luminance + strength * (0.08 + 0.84 * stretched)
    ratio = target / np.maximum(luminance, 1e-3)
    ratio = np.clip(ratio, 0.55, 1.65)
    adjusted = np.clip(rgb * ratio[..., None], 0.0, 1.0)
    return Image.fromarray((adjusted * 255.0).round().astype(np.uint8), mode="RGB")


def _suppress_background_image(
    image: Image.Image,
    *,
    mode: str = "none",
    margin: float = 0.08,
    blur_radius: float = 7.0,
) -> Image.Image:
    normalized_mode = str(mode or "none").strip().lower().replace("-", "_")
    if normalized_mode in {"", "none", "off", "false"}:
        return image
    use_grabcut = normalized_mode == "grabcut" or normalized_mode.startswith("grabcut_")
    if use_grabcut:
        mask_mode = normalized_mode[len("grabcut") :].lstrip("_") or "desaturate_blur"
        mask = _grabcut_foreground_mask_array(image, margin=margin)
    else:
        mask_mode = normalized_mode
        mask = _pseudo_foreground_mask_array(image, margin=margin)
    alpha = _soft_mask_image(mask, radius=max(1.0, float(blur_radius) * 0.25))
    base = image.convert("RGB")
    if mask_mode in {"gray", "background_gray"}:
        background = ImageOps.grayscale(base).convert("RGB")
    elif mask_mode in {"blur", "background_blur"}:
        background = base.filter(ImageFilter.GaussianBlur(radius=max(0.1, float(blur_radius))))
    elif mask_mode in {"mean", "background_mean"}:
        rgb = np.asarray(base, dtype=np.float32)
        mean = np.median(rgb.reshape(-1, 3), axis=0).round().astype(np.uint8)
        background = Image.new("RGB", base.size, tuple(int(value) for value in mean.tolist()))
    elif mask_mode in {"desaturate_blur", "blur_gray"}:
        background = ImageOps.grayscale(base).convert("RGB").filter(
            ImageFilter.GaussianBlur(radius=max(0.1, float(blur_radius)))
        )
    else:
        raise ValueError(f"background_suppression_mode khong hop le: {mode}")
    return Image.composite(base, background, alpha)


def _surface_detail_foreground_mask_array(image: Image.Image, margin: float = 0.08) -> np.ndarray:
    base = image.convert("RGB")
    pseudo_mask = _pseudo_foreground_mask_array(base, margin=margin)
    rgb = np.asarray(base, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        return pseudo_mask
    height, width = rgb.shape[:2]
    if height < 4 or width < 4:
        return pseudo_mask

    fill_rgb = np.asarray(_imagenet_fill(IMAGENET_MEAN), dtype=np.float32).reshape(1, 1, 3) / 255.0
    fill_delta = np.abs(rgb - fill_rgb).mean(axis=2)
    not_padding = fill_delta >= 0.035

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    delta = max_channel - min_channel
    saturation = np.where(max_channel > 1e-6, delta / np.maximum(max_channel, 1e-6), 0.0)
    hue = np.zeros_like(max_channel, dtype=np.float32)
    non_gray = delta > 1e-6
    red_is_max = (rgb[..., 0] >= rgb[..., 1]) & (rgb[..., 0] >= rgb[..., 2])
    green_is_max = (rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] >= rgb[..., 2])
    blue_is_max = ~(red_is_max | green_is_max)
    hue[red_is_max & non_gray] = ((rgb[..., 1] - rgb[..., 2]) / np.maximum(delta, 1e-6))[red_is_max & non_gray] % 6.0
    hue[green_is_max & non_gray] = ((rgb[..., 2] - rgb[..., 0]) / np.maximum(delta, 1e-6) + 2.0)[green_is_max & non_gray]
    hue[blue_is_max & non_gray] = ((rgb[..., 0] - rgb[..., 1]) / np.maximum(delta, 1e-6) + 4.0)[blue_is_max & non_gray]
    hue = (hue / 6.0).astype(np.float32)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx + 0.5) / max(1.0, float(width)) * 2.0 - 1.0
    yy = (yy + 0.5) / max(1.0, float(height)) * 2.0 - 1.0
    central = ((xx / 0.84) ** 2 + (yy / 0.92) ** 2) <= 1.0
    broad_central = ((xx / 0.95) ** 2 + (yy / 0.98) ** 2) <= 1.0

    green_yellow = (hue >= 0.08) & (hue <= 0.45) & (saturation >= 0.05) & (max_channel >= 0.12)
    orange_brown = (hue >= 0.03) & (hue <= 0.18) & (saturation >= 0.08) & (max_channel >= 0.08)
    fruit_like = green_yellow | orange_brown
    mask = pseudo_mask & not_padding & ((fruit_like & broad_central) | central)

    mask_fraction = float(mask.mean())
    if mask_fraction > 0.78:
        mask = pseudo_mask & not_padding & central
        mask_fraction = float(mask.mean())
    if mask_fraction < 0.04:
        fallback = pseudo_mask & not_padding & broad_central
        mask = fallback if float(fallback.mean()) >= 0.03 else pseudo_mask

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
    mask = np.asarray(mask_image, dtype=np.uint8) > 127
    component = _largest_connected_component(mask)
    if 0.04 <= float(component.mean()) <= 0.88:
        mask = component
    return mask


def _amplify_surface_detail_image(
    image: Image.Image,
    *,
    mode: str = "none",
    strength: float = 0.0,
    blur_radius: float = 1.25,
    foreground_margin: float = 0.08,
    foreground_weight: float = 0.85,
) -> Image.Image:
    normalized_mode = str(mode or "none").strip().lower().replace("-", "_")
    if normalized_mode in {"", "none", "off", "false"}:
        return image
    strength = max(0.0, float(strength))
    if strength <= 0.0:
        return image
    blur_radius = max(0.1, float(blur_radius))
    foreground_weight = max(0.0, min(1.0, float(foreground_weight)))

    base = image.convert("RGB")
    blurred = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    rgb = np.asarray(base, dtype=np.float32) / 255.0
    low_frequency = np.asarray(blurred, dtype=np.float32) / 255.0
    residual = rgb - low_frequency

    if normalized_mode in {"unsharp", "rgb_unsharp", "foreground_unsharp"}:
        enhanced = rgb + strength * residual
    elif normalized_mode in {"luma", "luma_residual", "foreground_luma"}:
        luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        low_luminance = (
            0.299 * low_frequency[..., 0]
            + 0.587 * low_frequency[..., 1]
            + 0.114 * low_frequency[..., 2]
        ).astype(np.float32)
        target_luminance = np.clip(luminance + strength * (luminance - low_luminance), 0.0, 1.0)
        ratio = np.clip(target_luminance / np.maximum(luminance, 1e-3), 0.55, 1.75)
        enhanced = rgb * ratio[..., None]
    else:
        raise ValueError(f"surface_detail_amplification_mode khong hop le: {mode}")

    use_foreground = normalized_mode.startswith("foreground") or foreground_weight < 1.0
    if use_foreground:
        try:
            mask = _surface_detail_foreground_mask_array(base, margin=float(foreground_margin))
            alpha = np.asarray(_soft_mask_image(mask, radius=max(1.0, blur_radius * 1.5)), dtype=np.float32) / 255.0
            alpha = (1.0 - foreground_weight) + foreground_weight * alpha
            enhanced = rgb + (enhanced - rgb) * alpha[..., None]
        except Exception:
            pass
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return Image.fromarray((enhanced * 255.0).round().astype(np.uint8), mode="RGB")


def _apply_mixup_batch(images: Tensor, targets: Tensor, alpha: float) -> Tuple[Tensor, Tensor]:
    if alpha <= 0.0 or images.size(0) < 2:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    shuffled_indices = torch.randperm(images.size(0))
    mixed_images = lam * images + (1.0 - lam) * images[shuffled_indices]
    mixed_targets = lam * targets + (1.0 - lam) * targets[shuffled_indices]
    return mixed_images, mixed_targets


def _apply_cutmix_batch(
    images: Tensor,
    targets: Tensor,
    alpha: float,
) -> Tuple[Tensor, Tensor]:
    if alpha <= 0.0 or images.size(0) < 2:
        return images, targets

    lam = float(np.random.beta(alpha, alpha))
    shuffled_indices = torch.randperm(images.size(0))

    height = images.size(-2)
    width = images.size(-1)
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = torch.randint(width, (1,)).item()
    center_y = torch.randint(height, (1,)).item()

    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)

    mixed_images = images.clone()
    mixed_images[..., y1:y2, x1:x2] = images[shuffled_indices, ..., y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(max(1, width * height)))
    mixed_targets = lam * targets + (1.0 - lam) * targets[shuffled_indices]
    return mixed_images, mixed_targets


def _pseudo_foreground_mask_from_tensor_images(
    images: Tensor,
    *,
    margin: float = 0.08,
    min_fraction: float = 0.05,
    max_fraction: float = 0.92,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("images phai co shape [B, 3, H, W].")
    device = images.device
    dtype = images.dtype
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    batch_size, _, height, width = rgb.shape
    flat_rgb = rgb.flatten(2)
    median_rgb = flat_rgb.median(dim=2).values.view(batch_size, 3, 1, 1)
    gray = rgb.mean(dim=1, keepdim=True)
    median_gray = gray.flatten(2).median(dim=2).values.view(batch_size, 1, 1, 1)
    color_delta = (rgb - median_rgb).abs().mean(dim=1, keepdim=True)
    intensity_delta = (gray - median_gray).abs()

    edge_delta = torch.zeros_like(gray)
    edge_delta[:, :, :, 1:] = torch.maximum(
        edge_delta[:, :, :, 1:],
        (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(),
    )
    edge_delta[:, :, 1:, :] = torch.maximum(
        edge_delta[:, :, 1:, :],
        (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(),
    )
    score = color_delta + intensity_delta + 0.5 * edge_delta

    max_channel = rgb.max(dim=1, keepdim=True).values
    min_channel = rgb.min(dim=1, keepdim=True).values
    delta = max_channel - min_channel
    saturation = torch.where(
        max_channel > 1e-6,
        delta / torch.clamp(max_channel, min=1e-6),
        torch.zeros_like(delta),
    )

    fill_rgb = torch.tensor(_imagenet_fill(mean), device=device, dtype=dtype).view(1, 3, 1, 1) / 255.0
    fill_delta = (rgb - fill_rgb).abs().mean(dim=1, keepdim=True)
    not_padding = (fill_delta >= 0.035) | (edge_delta >= 0.025)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    central = (((xx / 0.92) ** 2 + (yy / 0.96) ** 2) <= 1.0).view(1, 1, height, width)

    # Keep the tensor variant intentionally conservative. It is used for train-time
    # background recombination, so ambiguous green/yellow regions are only trusted
    # when they differ from the image median or lie inside the central object prior.
    colorful = (saturation >= 0.08) & (max_channel >= 0.10)
    detail_center = central & (score > max(0.035, float(margin)))
    mask = not_padding & (colorful | detail_center)
    mask = F.max_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.5
    mask = -(F.max_pool2d((-mask.float()), kernel_size=5, stride=1, padding=2)) > 0.5

    fractions = mask.float().flatten(1).mean(dim=1)
    fallback = not_padding & central
    fallback_fraction = fallback.float().flatten(1).mean(dim=1)
    too_small = fractions < float(min_fraction)
    if bool(too_small.any().item()):
        use_fallback = too_small & (fallback_fraction >= max(0.03, float(min_fraction) * 0.5))
        mask = torch.where(use_fallback.view(batch_size, 1, 1, 1), fallback, mask)
        fractions = mask.float().flatten(1).mean(dim=1)
    too_large = fractions > float(max_fraction)
    if bool(too_large.any().item()):
        stricter = not_padding & central & (score > max(0.05, float(margin) * 1.25))
        mask = torch.where(too_large.view(batch_size, 1, 1, 1), stricter, mask)
    return mask.to(dtype=torch.bool)


def _apply_foreground_background_mix_batch(
    images: Tensor,
    *,
    probability: float,
    margin: float = 0.08,
    min_foreground_fraction: float = 0.06,
    max_foreground_fraction: float = 0.88,
    softness: float = 5.0,
    mask_source: str = "pseudo",
    bboxes: Optional[Tensor] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    probability = max(0.0, min(1.0, float(probability)))
    if probability <= 0.0 or images.ndim != 4 or images.size(0) < 2:
        return images
    if torch.rand(1).item() >= probability:
        return images

    normalized_mask_source = str(mask_source or "pseudo").strip().lower()
    if normalized_mask_source not in {"pseudo", "bbox", "crop_bbox"}:
        raise ValueError(
            "foreground-background mix mask_source phai la 'pseudo', 'bbox', hoac 'crop_bbox'."
        )

    masks: Optional[Tensor] = None
    if normalized_mask_source in {"bbox", "crop_bbox"} and torch.is_tensor(bboxes):
        boxes = bboxes.detach().to(device=images.device, dtype=torch.float32)
        if boxes.ndim == 2 and boxes.size(0) == images.size(0) and boxes.size(1) == 4:
            height = int(images.size(-2))
            width = int(images.size(-1))
            centers = boxes[:, :2].clamp(0.0, 1.0)
            sizes = boxes[:, 2:].clamp(min=0.0, max=1.0)
            margin_value = float(max(0.0, margin))
            x1 = (centers[:, 0] - sizes[:, 0] * 0.5 - margin_value).clamp(0.0, 1.0)
            y1 = (centers[:, 1] - sizes[:, 1] * 0.5 - margin_value).clamp(0.0, 1.0)
            x2 = (centers[:, 0] + sizes[:, 0] * 0.5 + margin_value).clamp(0.0, 1.0)
            y2 = (centers[:, 1] + sizes[:, 1] * 0.5 + margin_value).clamp(0.0, 1.0)
            ys = (torch.arange(height, device=images.device, dtype=torch.float32) + 0.5) / max(1, height)
            xs = (torch.arange(width, device=images.device, dtype=torch.float32) + 0.5) / max(1, width)
            yy = ys.view(1, height, 1)
            xx = xs.view(1, 1, width)
            masks = (
                (xx >= x1.view(-1, 1, 1))
                & (xx <= x2.view(-1, 1, 1))
                & (yy >= y1.view(-1, 1, 1))
                & (yy <= y2.view(-1, 1, 1))
            ).unsqueeze(1)
    if masks is None:
        masks = _pseudo_foreground_mask_from_tensor_images(
            images,
            margin=margin,
            min_fraction=min_foreground_fraction,
            max_fraction=max_foreground_fraction,
            mean=mean,
            std=std,
        )
    fractions = masks.float().flatten(1).mean(dim=1)
    valid = (fractions >= float(min_foreground_fraction)) & (
        fractions <= float(max_foreground_fraction)
    )
    if not bool(valid.any().item()):
        return images

    batch_size = images.size(0)
    shift = int(torch.randint(1, batch_size, (1,)).item())
    indices = (torch.arange(batch_size, device=images.device) + shift) % batch_size
    alpha = masks.to(dtype=images.dtype)
    if softness > 0.0:
        kernel_size = max(3, int(round(float(softness))))
        if kernel_size % 2 == 0:
            kernel_size += 1
        alpha = F.max_pool2d(alpha, kernel_size=5, stride=1, padding=2)
        alpha = F.avg_pool2d(alpha, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        alpha = alpha.clamp(0.0, 1.0)
    alpha = torch.where(
        valid.view(batch_size, 1, 1, 1),
        alpha,
        torch.ones_like(alpha),
    )
    return images * alpha + images[indices] * (1.0 - alpha)


def _apply_mosaic_batch(
    images: Tensor,
    targets: Tensor,
    min_split: float,
    max_split: float,
) -> Tuple[Tensor, Tensor]:
    batch_size = images.size(0)
    height = images.size(-2)
    width = images.size(-1)
    if batch_size == 0:
        return images, targets

    min_split = float(min(max(min_split, 0.1), 0.9))
    max_split = float(min(max(max_split, min_split), 0.9))
    mosaic_images = images.new_zeros(images.shape)
    mosaic_targets = targets.new_zeros(targets.shape)

    for sample_index in range(batch_size):
        extra_indices = torch.randint(0, batch_size, (3,))
        source_indices = [sample_index] + [int(index.item()) for index in extra_indices]

        split_x = int(round(torch.empty(1).uniform_(min_split, max_split).item() * width))
        split_y = int(round(torch.empty(1).uniform_(min_split, max_split).item() * height))
        split_x = min(max(1, split_x), max(1, width - 1))
        split_y = min(max(1, split_y), max(1, height - 1))

        regions = [
            (0, split_y, 0, split_x),
            (0, split_y, split_x, width),
            (split_y, height, 0, split_x),
            (split_y, height, split_x, width),
        ]

        target = targets.new_zeros(targets.size(1))
        canvas = images.new_zeros(images.shape[1:])
        for source_index, (y0, y1, x0, x1) in zip(source_indices, regions):
            region_height = max(1, y1 - y0)
            region_width = max(1, x1 - x0)
            source = images[source_index]
            if source.ndim == 3:
                resized = F.interpolate(
                    source.unsqueeze(0),
                    size=(region_height, region_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                resized = F.interpolate(
                    source,
                    size=(region_height, region_width),
                    mode="bilinear",
                    align_corners=False,
                )
            canvas[..., y0:y1, x0:x1] = resized
            area_ratio = float(region_height * region_width) / float(max(1, height * width))
            target += targets[source_index] * area_ratio

        mosaic_images[sample_index] = canvas
        mosaic_targets[sample_index] = target

    return mosaic_images, mosaic_targets


def _normalize_detection_target(target: Dict[str, Tensor]) -> Dict[str, Tensor]:
    normalized = {
        "labels": target["labels"].detach().to(dtype=torch.long).cpu(),
        "boxes": target["boxes"].detach().to(dtype=torch.float32).cpu().clamp(0.0, 1.0),
    }
    image_mask = target.get("image_mask")
    if torch.is_tensor(image_mask):
        normalized["image_mask"] = image_mask.detach().to(dtype=torch.bool).cpu()
    return normalized


def _target_augmentation_scale(target: Dict[str, Tensor]) -> float:
    value = target.get("augmentation_scale")
    if value is None:
        return 1.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 1.0
        return float(value.detach().to(dtype=torch.float32).view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _filter_detection_target(
    labels: Tensor,
    boxes: Tensor,
    max_objects: int,
    image_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    def _empty_target() -> Dict[str, Tensor]:
        target = {
            "labels": torch.zeros((0,), dtype=torch.long),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
        }
        if torch.is_tensor(image_mask):
            target["image_mask"] = image_mask.detach().to(dtype=torch.bool).cpu()
        return target

    if labels.numel() == 0 or boxes.numel() == 0:
        return _empty_target()

    boxes = boxes.to(dtype=torch.float32).clamp(0.0, 1.0)
    labels = labels.to(dtype=torch.long)
    valid = (boxes[:, 2] > 1e-5) & (boxes[:, 3] > 1e-5)
    labels = labels[valid]
    boxes = boxes[valid]
    if labels.numel() == 0:
        return _empty_target()

    max_objects = max(1, int(max_objects))
    if labels.numel() > max_objects:
        areas = boxes[:, 2] * boxes[:, 3]
        keep = torch.topk(areas, k=max_objects, largest=True).indices
        labels = labels[keep]
        boxes = boxes[keep]
    target = {"labels": labels.contiguous(), "boxes": boxes.contiguous()}
    if torch.is_tensor(image_mask):
        target["image_mask"] = image_mask.detach().to(dtype=torch.bool).cpu()
    return target


def _xywh_to_xyxy_tensor(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)
    boxes = boxes.to(dtype=torch.float32)
    centers = boxes[:, :2]
    sizes = boxes[:, 2:].clamp(min=0.0)
    top_left = centers - sizes / 2.0
    bottom_right = centers + sizes / 2.0
    return torch.cat((top_left, bottom_right), dim=-1)


def _xyxy_to_xywh_tensor(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)
    boxes = boxes.to(dtype=torch.float32)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    widths = (x2 - x1).clamp(min=0.0)
    heights = (y2 - y1).clamp(min=0.0)
    return torch.stack(
        (
            x1 + widths / 2.0,
            y1 + heights / 2.0,
            widths,
            heights,
        ),
        dim=-1,
    )


def _limit_detection_targets(targets: Sequence[Dict[str, Tensor]], max_objects: int) -> List[Dict[str, Tensor]]:
    return [
        _filter_detection_target(
            target["labels"],
            target["boxes"],
            max_objects=max_objects,
            image_mask=target.get("image_mask"),
        )
        for target in targets
    ]


def _apply_mosaic_detection_batch(
    images: Tensor,
    targets: Sequence[Dict[str, Tensor]],
    min_split: float,
    max_split: float,
    max_objects: int,
    source_weights: Optional[Tensor] = None,
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    batch_size = images.size(0)
    height = int(images.size(-2))
    width = int(images.size(-1))
    if batch_size == 0:
        return images, list(targets)

    min_split = float(min(max(min_split, 0.1), 0.9))
    max_split = float(min(max(max_split, min_split), 0.9))
    mosaic_images = images.new_zeros(images.shape)
    mosaic_targets: List[Dict[str, Tensor]] = []

    normalized_targets = [_normalize_detection_target(target) for target in targets]
    if source_weights is not None and source_weights.numel() == batch_size:
        source_weights = source_weights.detach().cpu().to(dtype=torch.float32).clamp(min=1e-6)
        source_weights = source_weights / source_weights.sum().clamp(min=1e-6)
    else:
        source_weights = None
    for sample_index in range(batch_size):
        if source_weights is None:
            extra_indices = torch.randint(0, batch_size, (3,))
        else:
            extra_indices = torch.multinomial(source_weights, num_samples=3, replacement=True)
        source_indices = [sample_index] + [int(index.item()) for index in extra_indices]

        split_x = int(round(torch.empty(1).uniform_(min_split, max_split).item() * width))
        split_y = int(round(torch.empty(1).uniform_(min_split, max_split).item() * height))
        split_x = min(max(1, split_x), max(1, width - 1))
        split_y = min(max(1, split_y), max(1, height - 1))

        regions = [
            (0, split_y, 0, split_x),
            (0, split_y, split_x, width),
            (split_y, height, 0, split_x),
            (split_y, height, split_x, width),
        ]

        canvas = images.new_zeros(images.shape[1:])
        canvas_mask = torch.zeros((height, width), dtype=torch.bool)
        all_labels: List[Tensor] = []
        all_boxes: List[Tensor] = []
        for source_index, (y0, y1, x0, x1) in zip(source_indices, regions):
            region_height = max(1, int(y1 - y0))
            region_width = max(1, int(x1 - x0))
            source = images[source_index]
            resized = F.interpolate(
                source.unsqueeze(0),
                size=(region_height, region_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            canvas[..., y0:y1, x0:x1] = resized

            source_target = normalized_targets[source_index]
            source_mask = source_target.get("image_mask")
            if torch.is_tensor(source_mask):
                resized_mask = F.interpolate(
                    source_mask.to(dtype=torch.float32).view(1, 1, *source_mask.shape[-2:]),
                    size=(region_height, region_width),
                    mode="nearest",
                ).squeeze(0).squeeze(0) > 0.5
                canvas_mask[y0:y1, x0:x1] = resized_mask
            else:
                canvas_mask[y0:y1, x0:x1] = True

            source_boxes = source_target["boxes"]
            if source_boxes.numel() == 0:
                continue
            xyxy = _xywh_to_xyxy_tensor(source_boxes)
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] * float(region_width) + float(x0)) / float(max(1, width))
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] * float(region_height) + float(y0)) / float(max(1, height))
            xyxy = xyxy.clamp(0.0, 1.0)
            mosaic_boxes = _xyxy_to_xywh_tensor(xyxy)
            all_labels.append(source_target["labels"])
            all_boxes.append(mosaic_boxes)

        mosaic_images[sample_index] = canvas
        if all_boxes:
            mosaic_targets.append(
                _filter_detection_target(
                    labels=torch.cat(all_labels, dim=0),
                    boxes=torch.cat(all_boxes, dim=0),
                    max_objects=max_objects,
                    image_mask=canvas_mask,
                )
            )
        else:
            mosaic_targets.append(
                _filter_detection_target(
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, 4),
                    max_objects,
                    image_mask=canvas_mask,
                )
            )

    return mosaic_images, mosaic_targets


def _apply_cutmix_detection_batch(
    images: Tensor,
    targets: Sequence[Dict[str, Tensor]],
    alpha: float,
    max_objects: int,
    source_weights: Optional[Tensor] = None,
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    if alpha <= 0.0 or images.size(0) < 2:
        return images, _limit_detection_targets(targets, max_objects=max_objects)

    batch_size = int(images.size(0))
    height = int(images.size(-2))
    width = int(images.size(-1))
    if source_weights is not None and source_weights.numel() == batch_size:
        source_weights = source_weights.detach().cpu().to(dtype=torch.float32).clamp(min=1e-6)
        source_weights = source_weights / source_weights.sum().clamp(min=1e-6)
        shuffled_indices = torch.multinomial(source_weights, num_samples=batch_size, replacement=True)
    else:
        shuffled_indices = torch.randperm(batch_size)
    lam = float(np.random.beta(alpha, alpha))
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_width = max(1, int(round(width * cut_ratio)))
    cut_height = max(1, int(round(height * cut_ratio)))

    mixed_images = images.clone()
    mixed_targets: List[Dict[str, Tensor]] = []
    normalized_targets = [_normalize_detection_target(target) for target in targets]

    for batch_index in range(batch_size):
        source_index = int(shuffled_indices[batch_index].item())
        center_x = int(torch.randint(width, (1,)).item())
        center_y = int(torch.randint(height, (1,)).item())
        x1 = max(center_x - cut_width // 2, 0)
        y1 = max(center_y - cut_height // 2, 0)
        x2 = min(center_x + cut_width // 2, width)
        y2 = min(center_y + cut_height // 2, height)
        if x2 <= x1 or y2 <= y1:
            mixed_targets.append(normalized_targets[batch_index])
            continue

        mixed_images[batch_index, ..., y1:y2, x1:x2] = images[source_index, ..., y1:y2, x1:x2]
        base_mask = normalized_targets[batch_index].get("image_mask")
        source_mask = normalized_targets[source_index].get("image_mask")
        if torch.is_tensor(base_mask):
            mixed_mask = base_mask.detach().clone().to(dtype=torch.bool)
        else:
            mixed_mask = torch.ones((height, width), dtype=torch.bool)
        if torch.is_tensor(source_mask):
            mixed_mask[y1:y2, x1:x2] = source_mask.to(dtype=torch.bool)[y1:y2, x1:x2]
        else:
            mixed_mask[y1:y2, x1:x2] = True
        patch = torch.tensor(
            (
                x1 / float(width),
                y1 / float(height),
                x2 / float(width),
                y2 / float(height),
            ),
            dtype=torch.float32,
        )

        base_target = normalized_targets[batch_index]
        base_boxes_xyxy = _xywh_to_xyxy_tensor(base_target["boxes"])
        base_keep = torch.ones((base_boxes_xyxy.shape[0],), dtype=torch.bool)
        if base_boxes_xyxy.numel() > 0:
            inter_x1 = torch.maximum(base_boxes_xyxy[:, 0], patch[0])
            inter_y1 = torch.maximum(base_boxes_xyxy[:, 1], patch[1])
            inter_x2 = torch.minimum(base_boxes_xyxy[:, 2], patch[2])
            inter_y2 = torch.minimum(base_boxes_xyxy[:, 3], patch[3])
            inter_area = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
            box_area = (
                (base_boxes_xyxy[:, 2] - base_boxes_xyxy[:, 0]).clamp(min=1e-6)
                * (base_boxes_xyxy[:, 3] - base_boxes_xyxy[:, 1]).clamp(min=1e-6)
            )
            base_keep = (inter_area / box_area) <= 0.5

        source_target = normalized_targets[source_index]
        source_boxes_xyxy = _xywh_to_xyxy_tensor(source_target["boxes"])
        source_labels = source_target["labels"]
        source_keep = torch.zeros((source_boxes_xyxy.shape[0],), dtype=torch.bool)
        if source_boxes_xyxy.numel() > 0:
            source_boxes_xyxy[:, 0] = torch.maximum(source_boxes_xyxy[:, 0], patch[0])
            source_boxes_xyxy[:, 1] = torch.maximum(source_boxes_xyxy[:, 1], patch[1])
            source_boxes_xyxy[:, 2] = torch.minimum(source_boxes_xyxy[:, 2], patch[2])
            source_boxes_xyxy[:, 3] = torch.minimum(source_boxes_xyxy[:, 3], patch[3])
            source_keep = (
                (source_boxes_xyxy[:, 2] - source_boxes_xyxy[:, 0]) > 1e-5
            ) & (
                (source_boxes_xyxy[:, 3] - source_boxes_xyxy[:, 1]) > 1e-5
            )

        labels_to_merge: List[Tensor] = []
        boxes_to_merge: List[Tensor] = []
        if base_keep.any():
            labels_to_merge.append(base_target["labels"][base_keep])
            boxes_to_merge.append(base_target["boxes"][base_keep])
        if source_keep.any():
            labels_to_merge.append(source_labels[source_keep])
            boxes_to_merge.append(_xyxy_to_xywh_tensor(source_boxes_xyxy[source_keep]))

        if labels_to_merge:
            mixed_targets.append(
                _filter_detection_target(
                    labels=torch.cat(labels_to_merge, dim=0),
                    boxes=torch.cat(boxes_to_merge, dim=0),
                    max_objects=max_objects,
                    image_mask=mixed_mask,
                )
            )
        else:
            mixed_targets.append(
                _filter_detection_target(
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, 4),
                    max_objects,
                    image_mask=mixed_mask,
                )
            )

    return mixed_images, mixed_targets


def _apply_copypaste_detection_batch(
    images: Tensor,
    targets: Sequence[Dict[str, Tensor]],
    max_objects: int,
    max_paste_objects: int = 2,
    source_weights: Optional[Tensor] = None,
    target_class_scales: Optional[Tensor] = None,
    target_scale_threshold: float = 1.5,
    target_probability: float = 1.0,
    padding_ratio: float = 0.06,
    occlusion_threshold: float = 0.6,
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    batch_size = int(images.size(0))
    height = int(images.size(-2))
    width = int(images.size(-1))
    if batch_size < 2 or max_paste_objects <= 0:
        return images, _limit_detection_targets(targets, max_objects=max_objects)

    normalized_targets = [_normalize_detection_target(target) for target in targets]
    if source_weights is not None and source_weights.numel() == batch_size:
        source_weights = source_weights.detach().cpu().to(dtype=torch.float32).clamp(min=1e-6)
        source_weights = source_weights / source_weights.sum().clamp(min=1e-6)
    else:
        source_weights = None
    if target_class_scales is not None and target_class_scales.numel() > 0:
        target_class_scales = target_class_scales.detach().cpu().to(dtype=torch.float32).clamp(min=1.0)
    else:
        target_class_scales = None

    pasted_images = images.clone()
    pasted_targets: List[Dict[str, Tensor]] = []
    max_paste_objects = max(1, int(max_paste_objects))
    padding_ratio = max(0.0, float(padding_ratio))
    occlusion_threshold = min(max(float(occlusion_threshold), 0.0), 1.0)
    target_scale_threshold = max(1.0, float(target_scale_threshold))
    target_probability = min(max(float(target_probability), 0.0), 1.0)

    for batch_index in range(batch_size):
        base_target = normalized_targets[batch_index]
        base_labels = base_target["labels"].clone()
        base_boxes = base_target["boxes"].clone()
        base_mask = base_target.get("image_mask")
        if torch.is_tensor(base_mask):
            mixed_mask = base_mask.detach().clone().to(dtype=torch.bool)
        else:
            mixed_mask = torch.ones((height, width), dtype=torch.bool)

        labels_to_merge: List[Tensor] = [base_labels]
        boxes_to_merge: List[Tensor] = [base_boxes]
        pasted_patches: List[Tensor] = []

        paste_count = int(torch.randint(1, max_paste_objects + 1, (1,)).item())
        for _ in range(paste_count):
            if source_weights is None:
                source_index = int(torch.randint(0, batch_size, (1,)).item())
            else:
                source_index = int(torch.multinomial(source_weights, num_samples=1, replacement=True).item())
            source_target = normalized_targets[source_index]
            source_boxes = source_target["boxes"]
            source_labels = source_target["labels"]
            if source_boxes.numel() == 0:
                continue

            source_areas = (source_boxes[:, 2] * source_boxes[:, 3]).to(dtype=torch.float32)
            valid_object_indices = torch.nonzero(source_areas > 1e-5, as_tuple=False).flatten()
            if valid_object_indices.numel() == 0:
                continue
            selected_indices = valid_object_indices
            if (
                target_class_scales is not None
                and target_scale_threshold > 1.0
                and target_probability > 0.0
                and torch.rand(1).item() < target_probability
            ):
                candidate_labels = source_labels[valid_object_indices].to(dtype=torch.long)
                valid_label_mask = (candidate_labels >= 0) & (candidate_labels < target_class_scales.numel())
                if valid_label_mask.any():
                    candidate_scales = torch.ones_like(candidate_labels, dtype=torch.float32)
                    valid_candidate_labels = candidate_labels[valid_label_mask]
                    candidate_scales[valid_label_mask] = target_class_scales[valid_candidate_labels]
                    rare_mask = candidate_scales >= target_scale_threshold
                    if rare_mask.any():
                        selected_indices = valid_object_indices[rare_mask]
                        selected_weights = candidate_scales[rare_mask].clamp(min=1e-6)
                        selected_weights = selected_weights / selected_weights.sum().clamp(min=1e-6)
                        object_index = int(
                            selected_indices[
                                torch.multinomial(selected_weights, num_samples=1).item()
                            ].item()
                        )
                    else:
                        object_index = int(
                            valid_object_indices[
                                torch.randint(0, valid_object_indices.numel(), (1,)).item()
                            ].item()
                        )
                else:
                    object_index = int(
                        valid_object_indices[torch.randint(0, valid_object_indices.numel(), (1,)).item()].item()
                    )
            else:
                object_index = int(selected_indices[torch.randint(0, selected_indices.numel(), (1,)).item()].item())

            source_xyxy = _xywh_to_xyxy_tensor(source_boxes[object_index : object_index + 1])[0]
            box_width = float((source_xyxy[2] - source_xyxy[0]).clamp(min=0.0).item())
            box_height = float((source_xyxy[3] - source_xyxy[1]).clamp(min=0.0).item())
            if box_width <= 1e-5 or box_height <= 1e-5:
                continue

            pad_x = box_width * padding_ratio
            pad_y = box_height * padding_ratio
            crop_xyxy = torch.tensor(
                (
                    max(0.0, float(source_xyxy[0].item()) - pad_x),
                    max(0.0, float(source_xyxy[1].item()) - pad_y),
                    min(1.0, float(source_xyxy[2].item()) + pad_x),
                    min(1.0, float(source_xyxy[3].item()) + pad_y),
                ),
                dtype=torch.float32,
            )
            crop_x1 = int(math.floor(float(crop_xyxy[0].item()) * width))
            crop_y1 = int(math.floor(float(crop_xyxy[1].item()) * height))
            crop_x2 = int(math.ceil(float(crop_xyxy[2].item()) * width))
            crop_y2 = int(math.ceil(float(crop_xyxy[3].item()) * height))
            crop_x1 = min(max(crop_x1, 0), max(0, width - 1))
            crop_y1 = min(max(crop_y1, 0), max(0, height - 1))
            crop_x2 = min(max(crop_x2, crop_x1 + 1), width)
            crop_y2 = min(max(crop_y2, crop_y1 + 1), height)
            patch_width = int(crop_x2 - crop_x1)
            patch_height = int(crop_y2 - crop_y1)
            if patch_width <= 1 or patch_height <= 1 or patch_width >= width or patch_height >= height:
                continue

            dst_x1 = int(torch.randint(0, max(1, width - patch_width + 1), (1,)).item())
            dst_y1 = int(torch.randint(0, max(1, height - patch_height + 1), (1,)).item())
            dst_x2 = dst_x1 + patch_width
            dst_y2 = dst_y1 + patch_height
            pasted_images[batch_index, :, dst_y1:dst_y2, dst_x1:dst_x2] = images[
                source_index,
                :,
                crop_y1:crop_y2,
                crop_x1:crop_x2,
            ]
            source_mask = source_target.get("image_mask")
            if torch.is_tensor(source_mask):
                mixed_mask[dst_y1:dst_y2, dst_x1:dst_x2] = source_mask.to(dtype=torch.bool)[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2,
                ]
            else:
                mixed_mask[dst_y1:dst_y2, dst_x1:dst_x2] = True

            rel_object_x1 = float(source_xyxy[0].item()) - float(crop_xyxy[0].item())
            rel_object_y1 = float(source_xyxy[1].item()) - float(crop_xyxy[1].item())
            rel_object_x2 = float(source_xyxy[2].item()) - float(crop_xyxy[0].item())
            rel_object_y2 = float(source_xyxy[3].item()) - float(crop_xyxy[1].item())
            crop_width_norm = max(1e-6, float(crop_xyxy[2].item()) - float(crop_xyxy[0].item()))
            crop_height_norm = max(1e-6, float(crop_xyxy[3].item()) - float(crop_xyxy[1].item()))
            pasted_xyxy = torch.tensor(
                (
                    (float(dst_x1) + rel_object_x1 / crop_width_norm * float(patch_width)) / float(width),
                    (float(dst_y1) + rel_object_y1 / crop_height_norm * float(patch_height)) / float(height),
                    (float(dst_x1) + rel_object_x2 / crop_width_norm * float(patch_width)) / float(width),
                    (float(dst_y1) + rel_object_y2 / crop_height_norm * float(patch_height)) / float(height),
                ),
                dtype=torch.float32,
            ).clamp(0.0, 1.0)
            if (
                float((pasted_xyxy[2] - pasted_xyxy[0]).item()) <= 1e-5
                or float((pasted_xyxy[3] - pasted_xyxy[1]).item()) <= 1e-5
            ):
                continue
            pasted_patches.append(
                torch.tensor(
                    (
                        dst_x1 / float(width),
                        dst_y1 / float(height),
                        dst_x2 / float(width),
                        dst_y2 / float(height),
                    ),
                    dtype=torch.float32,
                )
            )
            labels_to_merge.append(source_labels[object_index : object_index + 1])
            boxes_to_merge.append(_xyxy_to_xywh_tensor(pasted_xyxy.view(1, 4)))

        if pasted_patches and base_boxes.numel() > 0:
            base_xyxy = _xywh_to_xyxy_tensor(base_boxes)
            keep_base = torch.ones((base_xyxy.shape[0],), dtype=torch.bool)
            base_area = (
                (base_xyxy[:, 2] - base_xyxy[:, 0]).clamp(min=1e-6)
                * (base_xyxy[:, 3] - base_xyxy[:, 1]).clamp(min=1e-6)
            )
            for patch_xyxy in pasted_patches:
                inter_x1 = torch.maximum(base_xyxy[:, 0], patch_xyxy[0])
                inter_y1 = torch.maximum(base_xyxy[:, 1], patch_xyxy[1])
                inter_x2 = torch.minimum(base_xyxy[:, 2], patch_xyxy[2])
                inter_y2 = torch.minimum(base_xyxy[:, 3], patch_xyxy[3])
                inter_area = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
                keep_base = keep_base & ((inter_area / base_area) <= occlusion_threshold)
            labels_to_merge[0] = base_labels[keep_base]
            boxes_to_merge[0] = base_boxes[keep_base]

        labels_to_merge = [labels for labels in labels_to_merge if labels.numel() > 0]
        boxes_to_merge = [boxes for boxes in boxes_to_merge if boxes.numel() > 0]
        if labels_to_merge and boxes_to_merge:
            pasted_targets.append(
                _filter_detection_target(
                    labels=torch.cat(labels_to_merge, dim=0),
                    boxes=torch.cat(boxes_to_merge, dim=0),
                    max_objects=max_objects,
                    image_mask=mixed_mask,
                )
            )
        else:
            pasted_targets.append(
                _filter_detection_target(
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, 4),
                    max_objects,
                    image_mask=mixed_mask,
                )
            )

    return pasted_images, pasted_targets


@dataclass
class TrainBatchCollator:
    num_classes: int
    batch_mix_probability: float = 0.5
    mosaic_probability: float = 0.25
    mosaic_min_split: float = 0.35
    mosaic_max_split: float = 0.65
    mixup_probability: float = 0.5
    mixup_alpha: float = 0.4
    cutmix_probability: float = 0.5
    cutmix_alpha: float = 1.0
    copy_paste_probability: float = 0.0
    copy_paste_max_objects: int = 2
    max_detection_objects: int = 40
    class_aware_mix_probability_boost: float = 0.0
    class_aware_mix_source_power: float = 1.0
    targeted_copy_paste_class_scales: Optional[Sequence[float]] = None
    targeted_copy_paste_scale_threshold: float = 1.5
    targeted_copy_paste_probability: float = 1.0
    foreground_background_mix_probability: float = 0.0
    foreground_background_mix_margin: float = 0.08
    foreground_background_mix_min_foreground_fraction: float = 0.06
    foreground_background_mix_max_foreground_fraction: float = 0.88
    foreground_background_mix_softness: float = 5.0
    foreground_background_mix_mask_source: str = "pseudo"

    def __post_init__(self) -> None:
        self.num_classes = max(1, int(self.num_classes))
        self.batch_mix_probability = max(0.0, min(1.0, float(self.batch_mix_probability)))
        self.mosaic_probability = max(0.0, float(self.mosaic_probability))
        self.mosaic_min_split = float(self.mosaic_min_split)
        self.mosaic_max_split = float(self.mosaic_max_split)
        self.mixup_probability = max(0.0, float(self.mixup_probability))
        self.mixup_alpha = float(self.mixup_alpha)
        self.cutmix_probability = max(0.0, float(self.cutmix_probability))
        self.cutmix_alpha = float(self.cutmix_alpha)
        self.copy_paste_probability = max(0.0, float(self.copy_paste_probability))
        self.copy_paste_max_objects = max(0, int(self.copy_paste_max_objects))
        self.max_detection_objects = max(1, int(self.max_detection_objects))
        self.class_aware_mix_probability_boost = max(0.0, float(self.class_aware_mix_probability_boost))
        self.class_aware_mix_source_power = max(0.0, float(self.class_aware_mix_source_power))
        self.targeted_copy_paste_scale_threshold = max(
            1.0,
            float(self.targeted_copy_paste_scale_threshold),
        )
        self.targeted_copy_paste_probability = min(
            max(float(self.targeted_copy_paste_probability), 0.0),
            1.0,
        )
        self.foreground_background_mix_probability = min(
            max(float(self.foreground_background_mix_probability), 0.0),
            1.0,
        )
        self.foreground_background_mix_margin = max(0.0, float(self.foreground_background_mix_margin))
        self.foreground_background_mix_min_foreground_fraction = min(
            max(float(self.foreground_background_mix_min_foreground_fraction), 0.0),
            1.0,
        )
        self.foreground_background_mix_max_foreground_fraction = min(
            max(
                float(self.foreground_background_mix_max_foreground_fraction),
                self.foreground_background_mix_min_foreground_fraction,
            ),
            1.0,
        )
        self.foreground_background_mix_softness = max(0.0, float(self.foreground_background_mix_softness))
        self.foreground_background_mix_mask_source = str(
            self.foreground_background_mix_mask_source or "pseudo"
        ).strip().lower()
        if self.foreground_background_mix_mask_source not in {"pseudo", "bbox", "crop_bbox"}:
            raise ValueError(
                "foreground_background_mix_mask_source phai la 'pseudo', 'bbox', hoac 'crop_bbox'."
            )
        if self.targeted_copy_paste_class_scales is not None:
            self.targeted_copy_paste_class_scales = torch.tensor(
                list(self.targeted_copy_paste_class_scales),
                dtype=torch.float32,
            ).clamp(min=1.0)

    def _apply_classification_background_mix(
        self,
        images: Tensor,
        bboxes: Optional[Tensor] = None,
    ) -> Tensor:
        return _apply_foreground_background_mix_batch(
            images,
            probability=self.foreground_background_mix_probability,
            margin=self.foreground_background_mix_margin,
            min_foreground_fraction=self.foreground_background_mix_min_foreground_fraction,
            max_foreground_fraction=self.foreground_background_mix_max_foreground_fraction,
            softness=self.foreground_background_mix_softness,
            mask_source=self.foreground_background_mix_mask_source,
            bboxes=bboxes,
        )

    def __call__(self, batch) -> Tuple[Tensor, Tensor]:
        if not batch:
            raise ValueError("Batch rong khong hop le.")
        sample_size = len(batch[0])
        images = torch.stack([sample[0] for sample in batch], dim=0)
        if sample_size == 2 and isinstance(batch[0][1], dict):
            augmentation_scales = torch.tensor(
                [_target_augmentation_scale(target) for _, target in batch],
                dtype=torch.float32,
            ).clamp(min=1.0)
            targets = []
            for _, target in batch:
                normalized_target = {
                    "labels": target["labels"].to(dtype=torch.long),
                    "boxes": target["boxes"].to(dtype=torch.float32),
                }
                image_mask = target.get("image_mask")
                if torch.is_tensor(image_mask):
                    normalized_target["image_mask"] = image_mask.to(dtype=torch.bool)
                targets.append(normalized_target)
            batch_scale = float(augmentation_scales.max().item()) if augmentation_scales.numel() else 1.0
            effective_mix_probability = min(
                1.0,
                self.batch_mix_probability
                * (1.0 + self.class_aware_mix_probability_boost * max(0.0, batch_scale - 1.0)),
            )
            if effective_mix_probability <= 0.0 or torch.rand(1).item() >= effective_mix_probability:
                return images, _limit_detection_targets(targets, max_objects=self.max_detection_objects)

            detection_choices = []
            if self.mosaic_probability > 0.0:
                detection_choices.append(("mosaic", self.mosaic_probability))
            if self.cutmix_probability > 0.0 and self.cutmix_alpha > 0.0:
                detection_choices.append(("cutmix", self.cutmix_probability))
            if self.copy_paste_probability > 0.0 and self.copy_paste_max_objects > 0:
                detection_choices.append(("copy_paste", self.copy_paste_probability))
            if not detection_choices:
                return images, _limit_detection_targets(targets, max_objects=self.max_detection_objects)

            source_weights = augmentation_scales.pow(self.class_aware_mix_source_power)
            choice_weights = torch.tensor([weight for _, weight in detection_choices], dtype=torch.float32)
            choice_index = int(torch.multinomial(choice_weights, num_samples=1).item())
            choice_name = detection_choices[choice_index][0]
            if choice_name == "mosaic":
                return _apply_mosaic_detection_batch(
                    images,
                    targets,
                    min_split=self.mosaic_min_split,
                    max_split=self.mosaic_max_split,
                    max_objects=self.max_detection_objects,
                    source_weights=source_weights,
                )
            if choice_name == "copy_paste":
                return _apply_copypaste_detection_batch(
                    images,
                    targets,
                    max_objects=self.max_detection_objects,
                    max_paste_objects=self.copy_paste_max_objects,
                    source_weights=source_weights,
                    target_class_scales=(
                        self.targeted_copy_paste_class_scales
                        if torch.is_tensor(self.targeted_copy_paste_class_scales)
                        else None
                    ),
                    target_scale_threshold=self.targeted_copy_paste_scale_threshold,
                    target_probability=self.targeted_copy_paste_probability,
                )
            return _apply_cutmix_detection_batch(
                images,
                targets,
                alpha=self.cutmix_alpha,
                max_objects=self.max_detection_objects,
                source_weights=source_weights,
            )

        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
        if sample_size >= 3 and isinstance(batch[0][2], dict):
            metadata: Dict[str, Tensor] = {}
            if "bbox" in batch[0][2]:
                metadata["bbox"] = torch.stack(
                    [sample[2]["bbox"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "crop_bbox" in batch[0][2]:
                metadata["crop_bbox"] = torch.stack(
                    [sample[2]["crop_bbox"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "source_context_image" in batch[0][2]:
                metadata["source_context_image"] = torch.stack(
                    [sample[2]["source_context_image"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "source_context_image_mask" in batch[0][2]:
                metadata["source_context_image_mask"] = torch.stack(
                    [sample[2]["source_context_image_mask"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.bool)
            if "source_context_bbox" in batch[0][2]:
                metadata["source_context_bbox"] = torch.stack(
                    [sample[2]["source_context_bbox"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "image_mask" in batch[0][2]:
                metadata["image_mask"] = torch.stack(
                    [sample[2]["image_mask"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.bool)
            if "paired_view_image" in batch[0][2]:
                metadata["paired_view_image"] = torch.stack(
                    [sample[2]["paired_view_image"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "paired_view_label" in batch[0][2]:
                metadata["paired_view_label"] = torch.as_tensor(
                    [int(sample[2]["paired_view_label"]) for sample in batch],
                    dtype=torch.long,
                )
            if "paired_view_bbox" in batch[0][2]:
                metadata["paired_view_bbox"] = torch.stack(
                    [sample[2]["paired_view_bbox"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "paired_view_crop_bbox" in batch[0][2]:
                metadata["paired_view_crop_bbox"] = torch.stack(
                    [sample[2]["paired_view_crop_bbox"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "paired_view_image_mask" in batch[0][2]:
                metadata["paired_view_image_mask"] = torch.stack(
                    [sample[2]["paired_view_image_mask"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.bool)
            if "teacher_probs" in batch[0][2]:
                metadata["teacher_probs"] = torch.stack(
                    [sample[2]["teacher_probs"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "patch_router_teacher_probs" in batch[0][2]:
                metadata["patch_router_teacher_probs"] = torch.stack(
                    [sample[2]["patch_router_teacher_probs"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "teacher_features" in batch[0][2]:
                metadata["teacher_features"] = torch.stack(
                    [sample[2]["teacher_features"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "soft_target" in batch[0][2]:
                metadata["soft_target"] = torch.stack(
                    [sample[2]["soft_target"] for sample in batch],
                    dim=0,
                ).to(dtype=torch.float32)
            if "sample_index" in batch[0][2]:
                metadata["sample_index"] = torch.as_tensor(
                    [int(sample[2]["sample_index"]) for sample in batch],
                    dtype=torch.long,
                )
            if "sample_weight" in batch[0][2]:
                metadata["sample_weight"] = torch.as_tensor(
                    [float(sample[2]["sample_weight"]) for sample in batch],
                    dtype=torch.float32,
                )
            if "quality_group_index" in batch[0][2]:
                metadata["quality_group_index"] = torch.as_tensor(
                    [int(sample[2]["quality_group_index"]) for sample in batch],
                    dtype=torch.long,
                )
            if "targeted_margin_negative" in batch[0][2]:
                metadata["targeted_margin_negative"] = torch.as_tensor(
                    [int(sample[2]["targeted_margin_negative"]) for sample in batch],
                    dtype=torch.long,
                )
            if "targeted_margin_weight" in batch[0][2]:
                metadata["targeted_margin_weight"] = torch.as_tensor(
                    [float(sample[2]["targeted_margin_weight"]) for sample in batch],
                    dtype=torch.float32,
                )
            if "targeted_margin_margin" in batch[0][2]:
                metadata["targeted_margin_margin"] = torch.as_tensor(
                    [float(sample[2]["targeted_margin_margin"]) for sample in batch],
                    dtype=torch.float32,
                )
            if metadata:
                bbox_mix_source = None
                if self.foreground_background_mix_mask_source == "crop_bbox":
                    bbox_mix_source = metadata.get("crop_bbox")
                    if bbox_mix_source is None:
                        bbox_mix_source = metadata.get("bbox")
                elif self.foreground_background_mix_mask_source == "bbox":
                    bbox_mix_source = metadata.get("bbox")
                    if bbox_mix_source is None:
                        bbox_mix_source = metadata.get("crop_bbox")
                images = self._apply_classification_background_mix(images, bboxes=bbox_mix_source)
                return images, labels, metadata

        targets = F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.float32)
        images = self._apply_classification_background_mix(images)

        if self.batch_mix_probability <= 0.0:
            return images, targets
        if torch.rand(1).item() >= self.batch_mix_probability:
            return images, targets

        choices = []
        if self.mosaic_probability > 0.0:
            choices.append(("mosaic", self.mosaic_probability))
        if self.mixup_probability > 0.0 and self.mixup_alpha > 0.0:
            choices.append(("mixup", self.mixup_probability))
        if self.cutmix_probability > 0.0 and self.cutmix_alpha > 0.0:
            choices.append(("cutmix", self.cutmix_probability))
        if not choices:
            return images, targets

        choice_weights = torch.tensor([weight for _, weight in choices], dtype=torch.float32)
        choice_index = int(torch.multinomial(choice_weights, num_samples=1).item())
        choice_name = choices[choice_index][0]
        if choice_name == "mosaic":
            return _apply_mosaic_batch(
                images,
                targets,
                min_split=self.mosaic_min_split,
                max_split=self.mosaic_max_split,
            )
        if choice_name == "mixup":
            return _apply_mixup_batch(images, targets, alpha=self.mixup_alpha)
        return _apply_cutmix_batch(images, targets, alpha=self.cutmix_alpha)


def build_train_collate_fn(
    num_classes: int,
    batch_mix_probability: float = 0.5,
    mosaic_probability: float = 0.25,
    mosaic_min_split: float = 0.35,
    mosaic_max_split: float = 0.65,
    mixup_probability: float = 0.5,
    mixup_alpha: float = 0.4,
    cutmix_probability: float = 0.5,
    cutmix_alpha: float = 1.0,
    copy_paste_probability: float = 0.0,
    copy_paste_max_objects: int = 2,
    max_detection_objects: int = 40,
    class_aware_mix_probability_boost: float = 0.0,
    class_aware_mix_source_power: float = 1.0,
    targeted_copy_paste_class_scales: Optional[Sequence[float]] = None,
    targeted_copy_paste_scale_threshold: float = 1.5,
    targeted_copy_paste_probability: float = 1.0,
    foreground_background_mix_probability: float = 0.0,
    foreground_background_mix_margin: float = 0.08,
    foreground_background_mix_min_foreground_fraction: float = 0.06,
    foreground_background_mix_max_foreground_fraction: float = 0.88,
    foreground_background_mix_softness: float = 5.0,
    foreground_background_mix_mask_source: str = "pseudo",
) -> Callable:
    return TrainBatchCollator(
        num_classes=num_classes,
        batch_mix_probability=batch_mix_probability,
        mosaic_probability=mosaic_probability,
        mosaic_min_split=mosaic_min_split,
        mosaic_max_split=mosaic_max_split,
        mixup_probability=mixup_probability,
        mixup_alpha=mixup_alpha,
        cutmix_probability=cutmix_probability,
        cutmix_alpha=cutmix_alpha,
        copy_paste_probability=copy_paste_probability,
        copy_paste_max_objects=copy_paste_max_objects,
        max_detection_objects=max_detection_objects,
        class_aware_mix_probability_boost=class_aware_mix_probability_boost,
        class_aware_mix_source_power=class_aware_mix_source_power,
        targeted_copy_paste_class_scales=targeted_copy_paste_class_scales,
        targeted_copy_paste_scale_threshold=targeted_copy_paste_scale_threshold,
        targeted_copy_paste_probability=targeted_copy_paste_probability,
        foreground_background_mix_probability=foreground_background_mix_probability,
        foreground_background_mix_margin=foreground_background_mix_margin,
        foreground_background_mix_min_foreground_fraction=foreground_background_mix_min_foreground_fraction,
        foreground_background_mix_max_foreground_fraction=foreground_background_mix_max_foreground_fraction,
        foreground_background_mix_softness=foreground_background_mix_softness,
        foreground_background_mix_mask_source=foreground_background_mix_mask_source,
    )


@dataclass
class MangoObject:
    label: int
    bbox: Tuple[float, float, float, float]
    object_index: int


@dataclass
class MangoSample:
    image_path: Path
    label_path: Path
    objects: List[MangoObject]
    primary_label: int
    primary_object_index: int = -1


@dataclass
class ClassificationFolderSample:
    image_path: Path
    label: int
    class_name: str
    yolo_source_image_path: Optional[Path] = None
    yolo_label_path: Optional[Path] = None
    yolo_source_id: str = ""
    yolo_object: Optional[MangoObject] = None
    yolo_object_count: int = 0
    class_label_matches_yolo: Optional[bool] = None


class ClassificationFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        class_names: Sequence[str],
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        class_aware_augmentation: bool = False,
        class_augmentation_power: float = 0.75,
        class_augmentation_max_scale: float = 1.8,
        class_conditional_augmentation_scales: Optional[Sequence[float]] = None,
        class_crop_margin_scales: Optional[Sequence[float]] = None,
        paired_yolo_images_dir: Optional[Path] = None,
        paired_yolo_labels_dir: Optional[Path] = None,
        paired_yolo_class_names: Optional[Sequence[str]] = None,
        classification_source_context: bool = False,
        classification_source_context_mode: str = "desaturate_blur",
        classification_source_context_layout: str = "full",
        classification_source_context_margin_ratio: float = 0.12,
        classification_source_context_background_alpha: float = 0.35,
        classification_source_context_blur_radius: float = 7.0,
        classification_source_context_inset_scale: float = 0.34,
        classification_source_context_aux: bool = False,
        classification_bbox_metadata: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir
        self.labels_dir = self.root_dir
        self.class_names = [str(name) for name in class_names]
        self.num_classes = len(self.class_names)
        self.transform = transform
        self.split = str(split or "unknown")
        self.classification_target = True
        self.classification_object_crops = False
        self.crop_to_primary_object = False
        self.paired_yolo_images_dir = Path(paired_yolo_images_dir) if paired_yolo_images_dir else None
        self.paired_yolo_labels_dir = Path(paired_yolo_labels_dir) if paired_yolo_labels_dir else None
        self.paired_yolo_class_names = (
            [str(name) for name in paired_yolo_class_names]
            if paired_yolo_class_names is not None
            else list(self.class_names)
        )
        self.paired_yolo_enabled = bool(
            self.paired_yolo_images_dir is not None and self.paired_yolo_labels_dir is not None
        )
        self.classification_source_context = bool(
            self.paired_yolo_enabled and classification_source_context
        )
        self.classification_source_context_aux = bool(
            self.paired_yolo_enabled and classification_source_context_aux
        )
        self.classification_source_context_mode = (
            str(classification_source_context_mode or "desaturate_blur")
            .strip()
            .lower()
            .replace("-", "_")
        )
        valid_source_context_modes = {"dim", "gray", "blur", "mean", "desaturate_blur", "blur_gray"}
        if self.classification_source_context_mode not in valid_source_context_modes:
            raise ValueError(
                "classification_source_context_mode khong hop le: "
                f"{classification_source_context_mode}. Hop le: {sorted(valid_source_context_modes)}"
            )
        self.classification_source_context_layout = (
            str(classification_source_context_layout or "full")
            .strip()
            .lower()
            .replace("-", "_")
        )
        valid_source_context_layouts = {"full"}
        if self.classification_source_context_layout not in valid_source_context_layouts:
            raise ValueError(
                "classification_source_context_layout khong hop le cho classification_folder paired YOLO: "
                f"{classification_source_context_layout}. Hop le: {sorted(valid_source_context_layouts)}"
            )
        self.classification_source_context_margin_ratio = max(
            0.0,
            float(classification_source_context_margin_ratio),
        )
        self.classification_source_context_background_alpha = min(
            1.0,
            max(0.0, float(classification_source_context_background_alpha)),
        )
        self.classification_source_context_blur_radius = max(
            0.1,
            float(classification_source_context_blur_radius),
        )
        self.classification_source_context_inset_scale = min(
            0.75,
            max(0.10, float(classification_source_context_inset_scale)),
        )
        self.classification_bbox_metadata = bool(
            self.paired_yolo_enabled and classification_bbox_metadata
        )
        self.class_aware_augmentation = bool(class_aware_augmentation)
        self.class_augmentation_power = max(0.0, float(class_augmentation_power))
        self.class_augmentation_max_scale = max(1.0, float(class_augmentation_max_scale))
        self.class_conditional_augmentation_scales = normalize_class_conditional_augmentation_scales(
            class_conditional_augmentation_scales,
            num_classes=self.num_classes,
        )
        self.class_crop_margin_scales = (
            [max(1.0, float(value)) for value in class_crop_margin_scales]
            if class_crop_margin_scales is not None
            else []
        )
        self._image_cache_enabled = False
        self._image_cache_max_bytes = 0
        self._image_cache_max_items = 0
        self._image_cache_bytes = 0
        self._image_cache_hits = 0
        self._image_cache_misses = 0
        self._image_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self.audit = self._init_audit()
        self.samples = self._index_samples()
        self.class_augmentation_scales = self._build_class_augmentation_scales()
        logger.info(
            "Classification folder dataset initialized: split=%s root=%s "
            "classes=%s selected_samples=%s missing_class_dirs=%s class_aug=%s scales=%s conditional_scales=%s",
            self.split,
            self.root_dir,
            self.num_classes,
            len(self.samples),
            self.audit["missing_class_dir_count"],
            self.class_aware_augmentation,
            self.class_augmentation_scales,
            self.class_conditional_augmentation_scales,
        )

    @classmethod
    def from_data_spec(
        cls,
        data_spec: DataSpec,
        split: str,
        transform: Optional[Callable] = None,
        class_aware_augmentation: bool = False,
        class_augmentation_power: float = 0.75,
        class_augmentation_max_scale: float = 1.8,
        class_conditional_augmentation_scales: Optional[Sequence[float]] = None,
        class_crop_margin_scales: Optional[Sequence[float]] = None,
        paired_yolo_data_spec: Optional[DataSpec] = None,
        classification_source_context: bool = False,
        classification_source_context_mode: str = "desaturate_blur",
        classification_source_context_layout: str = "full",
        classification_source_context_margin_ratio: float = 0.12,
        classification_source_context_background_alpha: float = 0.35,
        classification_source_context_blur_radius: float = 7.0,
        classification_source_context_inset_scale: float = 0.34,
        classification_source_context_aux: bool = False,
        classification_bbox_metadata: bool = False,
    ) -> "ClassificationFolderDataset":
        paired_images_dir = None
        paired_labels_dir = None
        paired_class_names = None
        if paired_yolo_data_spec is not None:
            paired_images_dir = paired_yolo_data_spec.split_images_dir(split)
            paired_labels_dir = paired_yolo_data_spec.split_labels_dir(split)
            paired_class_names = paired_yolo_data_spec.class_names
        return cls(
            root_dir=data_spec.split_images_dir(split),
            class_names=data_spec.class_names,
            transform=transform,
            split=split,
            class_aware_augmentation=class_aware_augmentation,
            class_augmentation_power=class_augmentation_power,
            class_augmentation_max_scale=class_augmentation_max_scale,
            class_conditional_augmentation_scales=class_conditional_augmentation_scales,
            class_crop_margin_scales=class_crop_margin_scales,
            paired_yolo_images_dir=paired_images_dir,
            paired_yolo_labels_dir=paired_labels_dir,
            paired_yolo_class_names=paired_class_names,
            classification_source_context=classification_source_context,
            classification_source_context_mode=classification_source_context_mode,
            classification_source_context_layout=classification_source_context_layout,
            classification_source_context_margin_ratio=classification_source_context_margin_ratio,
            classification_source_context_background_alpha=classification_source_context_background_alpha,
            classification_source_context_blur_radius=classification_source_context_blur_radius,
            classification_source_context_inset_scale=classification_source_context_inset_scale,
            classification_source_context_aux=classification_source_context_aux,
            classification_bbox_metadata=classification_bbox_metadata,
        )

    def _init_audit(self) -> Dict[str, object]:
        return {
            "image_file_count": 0,
            "label_file_count": 0,
            "valid_object_count": 0,
            "selected_sample_count": 0,
            "single_object_image_count": 0,
            "multi_object_image_count": 0,
            "max_objects_per_image": 1,
            "object_count_histogram": {"1": 0},
            "ignored_object_count": 0,
            "missing_image_count": 0,
            "empty_label_count": 0,
            "invalid_line_count": 0,
            "invalid_bbox_count": 0,
            "invalid_class_count": 0,
            "missing_class_dir_count": 0,
            "sample_missing_images": [],
            "sample_empty_labels": [],
            "sample_invalid_lines": [],
            "sample_invalid_bboxes": [],
            "sample_invalid_classes": [],
            "sample_missing_class_dirs": [],
            "paired_yolo_enabled": bool(self.paired_yolo_enabled),
            "paired_yolo_mapped_count": 0,
            "paired_yolo_missing_mapping_count": 0,
            "paired_yolo_label_mismatch_count": 0,
            "paired_yolo_invalid_line_count": 0,
            "paired_yolo_invalid_bbox_count": 0,
            "paired_yolo_invalid_class_count": 0,
            "paired_yolo_multi_object_count": 0,
            "sample_paired_yolo_missing": [],
            "sample_paired_yolo_label_mismatch": [],
            "sample_paired_yolo_invalid_lines": [],
            "sample_paired_yolo_invalid_bboxes": [],
            "sample_paired_yolo_invalid_classes": [],
        }

    def _audit_append(self, key: str, value: object, limit: int = 10) -> None:
        sample_list = self.audit[key]
        if len(sample_list) < limit:
            sample_list.append(value)

    @staticmethod
    def _source_from_crop_path(path: Path) -> Tuple[str, int]:
        match = CLASSIFICATION_CROP_NAME.match(path.stem)
        if match is None:
            raise ValueError(f"Cannot parse classification crop name: {path.name}")
        return str(match.group("source")), int(match.group("box"))

    def _find_paired_yolo_image(self, source_id: str) -> Optional[Path]:
        if self.paired_yolo_images_dir is None:
            return None
        for suffix in IMAGE_EXTENSIONS:
            candidate = self.paired_yolo_images_dir / f"{source_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _load_paired_yolo_objects(self, label_path: Path) -> List[MangoObject]:
        objects: List[MangoObject] = []
        if not label_path.is_file():
            return objects
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for object_index, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                self.audit["paired_yolo_invalid_line_count"] += 1
                self._audit_append(
                    "sample_paired_yolo_invalid_lines",
                    {"file": label_path.name, "line": object_index + 1, "content": line},
                )
                continue
            try:
                label = int(parts[0])
                bbox = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except ValueError:
                self.audit["paired_yolo_invalid_line_count"] += 1
                self._audit_append(
                    "sample_paired_yolo_invalid_lines",
                    {"file": label_path.name, "line": object_index + 1, "content": line},
                )
                continue
            if label < 0 or label >= len(self.paired_yolo_class_names):
                self.audit["paired_yolo_invalid_class_count"] += 1
                self._audit_append(
                    "sample_paired_yolo_invalid_classes",
                    {"file": label_path.name, "line": object_index + 1, "label": label},
                )
                continue
            if any(value < 0.0 or value > 1.0 for value in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
                self.audit["paired_yolo_invalid_bbox_count"] += 1
                self._audit_append(
                    "sample_paired_yolo_invalid_bboxes",
                    {"file": label_path.name, "line": object_index + 1, "bbox": bbox},
                )
                continue
            objects.append(MangoObject(label=label, bbox=bbox, object_index=object_index))
        return objects

    def _paired_yolo_mapping_for_sample(
        self,
        image_path: Path,
        class_label: int,
    ) -> Tuple[Optional[Path], Optional[Path], str, Optional[MangoObject], int, Optional[bool]]:
        if not self.paired_yolo_enabled or self.paired_yolo_labels_dir is None:
            return None, None, "", None, 0, None
        try:
            source_id, box_index = self._source_from_crop_path(image_path)
        except ValueError:
            self.audit["paired_yolo_missing_mapping_count"] += 1
            self._audit_append("sample_paired_yolo_missing", str(image_path))
            return None, None, "", None, 0, None

        source_image_path = self._find_paired_yolo_image(source_id)
        label_path = self.paired_yolo_labels_dir / f"{source_id}.txt"
        objects = self._load_paired_yolo_objects(label_path)
        if source_image_path is None or box_index >= len(objects):
            self.audit["paired_yolo_missing_mapping_count"] += 1
            self._audit_append(
                "sample_paired_yolo_missing",
                {
                    "crop": str(image_path),
                    "source_id": source_id,
                    "box_index": int(box_index),
                    "source_image": str(source_image_path or ""),
                    "label_path": str(label_path),
                    "object_count": len(objects),
                },
            )
            return source_image_path, label_path, source_id, None, len(objects), None

        if len(objects) > 1:
            self.audit["paired_yolo_multi_object_count"] += 1
        selected = objects[int(box_index)]
        label_matches = bool(int(selected.label) == int(class_label))
        if not label_matches:
            self.audit["paired_yolo_label_mismatch_count"] += 1
            self._audit_append(
                "sample_paired_yolo_label_mismatch",
                {
                    "crop": str(image_path),
                    "source_id": source_id,
                    "box_index": int(box_index),
                    "class_f_label": int(class_label),
                    "yolo_label": int(selected.label),
                },
            )
        self.audit["paired_yolo_mapped_count"] += 1
        return source_image_path, label_path, source_id, selected, len(objects), label_matches

    def _apply_paired_yolo_source_context(
        self,
        image: Image.Image,
        yolo_object: MangoObject,
    ) -> Image.Image:
        base = image.convert("RGB")
        width, height = base.size
        if width <= 0 or height <= 0:
            return base

        x1, y1, x2, y2 = bbox_xywh_to_xyxy(
            yolo_object.bbox,
            width=width,
            height=height,
        )
        box_width = max(1.0, float(x2 - x1))
        box_height = max(1.0, float(y2 - y1))
        margin_ratio = max(0.0, float(self.classification_source_context_margin_ratio))
        margin_x = box_width * margin_ratio
        margin_y = box_height * margin_ratio
        left = max(0, int(math.floor(float(x1) - margin_x)))
        top = max(0, int(math.floor(float(y1) - margin_y)))
        right = min(width, int(math.ceil(float(x2) + margin_x)))
        bottom = min(height, int(math.ceil(float(y2) + margin_y)))
        if right <= left or bottom <= top:
            return base

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
        mask = mask.filter(
            ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) * 0.0125))
        )

        mode = self.classification_source_context_mode
        if mode == "dim":
            background = Image.new("RGB", base.size, tuple(_imagenet_fill(IMAGENET_MEAN)))
        elif mode == "gray":
            background = ImageOps.grayscale(base).convert("RGB")
        elif mode == "blur":
            background = base.filter(
                ImageFilter.GaussianBlur(
                    radius=max(0.1, float(self.classification_source_context_blur_radius))
                )
            )
        elif mode == "mean":
            rgb = np.asarray(base, dtype=np.float32)
            mean = np.median(rgb.reshape(-1, 3), axis=0).round().astype(np.uint8)
            background = Image.new("RGB", base.size, tuple(int(value) for value in mean.tolist()))
        elif mode in {"desaturate_blur", "blur_gray"}:
            background = ImageOps.grayscale(base).convert("RGB").filter(
                ImageFilter.GaussianBlur(
                    radius=max(0.1, float(self.classification_source_context_blur_radius))
                )
            )
        else:
            raise ValueError(f"classification_source_context_mode khong hop le: {mode}")

        outside = Image.blend(
            background,
            base,
            float(self.classification_source_context_background_alpha),
        )
        return Image.composite(base, outside, mask)

    def _paired_yolo_source_context_aux_metadata(
        self,
        *,
        source_image: Image.Image,
        yolo_object: MangoObject,
        label: int,
    ) -> Dict[str, Tensor]:
        context_image = self._apply_paired_yolo_source_context(
            image=source_image,
            yolo_object=yolo_object,
        )
        context_target = {
            "labels": torch.tensor([int(label)], dtype=torch.long),
            "boxes": torch.tensor([yolo_object.bbox], dtype=torch.float32),
            "augmentation_scale": torch.tensor(
                [self._augmentation_scale_for_label(int(label))],
                dtype=torch.float32,
            ),
        }
        if self.transform is not None:
            transformed = self.transform(context_image, target=context_target)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                context_tensor, transformed_target = transformed
            else:
                raise TypeError("Transform detection phai tra ve (image_tensor, target).")
        else:
            context_tensor = TF.to_tensor(context_image)
            transformed_target = context_target

        bbox_value = transformed_target.get("boxes") if isinstance(transformed_target, dict) else None
        if torch.is_tensor(bbox_value) and bbox_value.ndim == 2 and bbox_value.size(0) > 0:
            context_bbox = bbox_value[0].to(dtype=torch.float32).clamp(0.0, 1.0)
        else:
            context_bbox = torch.tensor(yolo_object.bbox, dtype=torch.float32).clamp(0.0, 1.0)
        metadata = {
            "source_context_image": context_tensor,
            "source_context_bbox": context_bbox,
        }
        context_mask = (
            transformed_target.get("image_mask")
            if isinstance(transformed_target, dict)
            else None
        )
        if torch.is_tensor(context_mask):
            metadata["source_context_image_mask"] = context_mask.to(dtype=torch.bool)
        return metadata

    def _index_samples(self) -> List[ClassificationFolderSample]:
        samples: List[ClassificationFolderSample] = []
        if not self.root_dir.exists():
            self.audit["missing_image_count"] += 1
            self._audit_append("sample_missing_images", str(self.root_dir))
            return samples

        for class_index, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                self.audit["missing_class_dir_count"] += 1
                self._audit_append("sample_missing_class_dirs", class_name)
                continue
            image_paths = [
                path
                for path in sorted(class_dir.rglob("*"), key=lambda item: str(item).lower())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            self.audit["image_file_count"] += len(image_paths)
            self.audit["valid_object_count"] += len(image_paths)
            self.audit["selected_sample_count"] += len(image_paths)
            self.audit["single_object_image_count"] += len(image_paths)
            histogram = self.audit.get("object_count_histogram")
            if isinstance(histogram, dict):
                histogram["1"] = int(histogram.get("1", 0)) + len(image_paths)
            for image_path in image_paths:
                (
                    yolo_source_image_path,
                    yolo_label_path,
                    yolo_source_id,
                    yolo_object,
                    yolo_object_count,
                    label_matches_yolo,
                ) = self._paired_yolo_mapping_for_sample(image_path, class_index)
                samples.append(
                    ClassificationFolderSample(
                        image_path=image_path,
                        label=int(class_index),
                        class_name=class_name,
                        yolo_source_image_path=yolo_source_image_path,
                        yolo_label_path=yolo_label_path,
                        yolo_source_id=yolo_source_id,
                        yolo_object=yolo_object,
                        yolo_object_count=int(yolo_object_count),
                        class_label_matches_yolo=label_matches_yolo,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(max(1, int(num_classes)))]
        for sample in self.samples:
            if 0 <= int(sample.label) < len(counts):
                counts[int(sample.label)] += 1
        return counts

    def _build_class_augmentation_scales(self) -> List[float]:
        if self.class_crop_margin_scales:
            values = list(self.class_crop_margin_scales[: self.num_classes])
            while len(values) < self.num_classes:
                values.append(1.0)
            return [max(1.0, float(value)) for value in values]
        if not self.class_aware_augmentation or self.num_classes <= 0:
            return [1.0 for _ in range(max(1, self.num_classes))]
        counts = self.class_counts(self.num_classes)
        positive_counts = [max(1, int(count)) for count in counts]
        max_count = max(positive_counts) if positive_counts else 1
        scales = []
        for count in positive_counts:
            scale = (float(max_count) / float(count)) ** self.class_augmentation_power
            scales.append(float(min(self.class_augmentation_max_scale, max(1.0, scale))))
        return scales

    def _augmentation_scale_for_label(self, label: int) -> float:
        label = int(label)
        base_scale = 1.0
        if 0 <= label < len(self.class_augmentation_scales):
            base_scale = float(self.class_augmentation_scales[label])
        conditional_scale = 1.0
        if 0 <= label < len(self.class_conditional_augmentation_scales):
            conditional_scale = float(self.class_conditional_augmentation_scales[label])
        return float(base_scale * conditional_scale)

    def labels(self) -> List[int]:
        return [int(sample.label) for sample in self.samples]

    def bboxes(self) -> List[Tuple[float, float, float, float]]:
        return [(0.5, 0.5, 1.0, 1.0) for _ in self.samples]

    def sample_paths(self) -> List[Path]:
        return [sample.image_path for sample in self.samples]

    def quality_report(self) -> Dict[str, object]:
        report = dict(self.audit)
        report["data_format"] = "classification_folder"
        report["classification_target"] = True
        report["classification_object_crops"] = False
        report["crop_to_primary_object"] = False
        report["class_counts"] = self.class_counts(self.num_classes)
        report["class_augmentation_scales"] = list(self.class_augmentation_scales)
        report["class_conditional_augmentation_scales"] = list(
            self.class_conditional_augmentation_scales
        )
        report["effective_class_augmentation_scales"] = [
            self._augmentation_scale_for_label(class_index)
            for class_index in range(self.num_classes)
        ]
        report["class_aware_augmentation"] = bool(self.class_aware_augmentation)
        report["image_cache"] = self.image_cache_stats()
        report["paired_yolo"] = {
            "enabled": bool(self.paired_yolo_enabled),
            "images_dir": str(self.paired_yolo_images_dir) if self.paired_yolo_images_dir else "",
            "labels_dir": str(self.paired_yolo_labels_dir) if self.paired_yolo_labels_dir else "",
            "mapped_count": int(self.audit.get("paired_yolo_mapped_count", 0)),
            "missing_mapping_count": int(self.audit.get("paired_yolo_missing_mapping_count", 0)),
            "label_mismatch_count": int(self.audit.get("paired_yolo_label_mismatch_count", 0)),
            "multi_object_count": int(self.audit.get("paired_yolo_multi_object_count", 0)),
        }
        report["classification_bbox_metadata"] = bool(self.classification_bbox_metadata)
        report["classification_source_context"] = {
            "enabled": bool(self.classification_source_context),
            "aux_enabled": bool(self.classification_source_context_aux),
            "mode": self.classification_source_context_mode,
            "layout": self.classification_source_context_layout,
            "margin_ratio": self.classification_source_context_margin_ratio,
            "background_alpha": self.classification_source_context_background_alpha,
            "blur_radius": self.classification_source_context_blur_radius,
            "inset_scale": self.classification_source_context_inset_scale,
        }
        return report

    def enable_image_cache(
        self,
        max_megabytes: int = 256,
        max_items: int = 0,
    ) -> None:
        self._image_cache_enabled = int(max_megabytes) > 0 or int(max_items) > 0
        self._image_cache_max_bytes = max(0, int(max_megabytes)) * 1024 * 1024
        self._image_cache_max_items = max(0, int(max_items))
        if not self._image_cache_enabled:
            self.clear_image_cache()
        logger.info(
            "Image cache configured: split=%s enabled=%s max_mb=%s max_items=%s",
            self.split,
            self._image_cache_enabled,
            max_megabytes,
            self._image_cache_max_items or None,
        )

    def clear_image_cache(self) -> None:
        self._image_cache.clear()
        self._image_cache_bytes = 0
        self._image_cache_hits = 0
        self._image_cache_misses = 0

    @staticmethod
    def _estimate_image_bytes(image: Image.Image) -> int:
        bands = max(1, len(image.getbands()))
        width, height = image.size
        return max(1, int(width) * int(height) * bands)

    def _put_image_cache(self, key: str, image: Image.Image) -> None:
        if not self._image_cache_enabled:
            return
        if key in self._image_cache:
            cached = self._image_cache.pop(key)
            self._image_cache_bytes -= self._estimate_image_bytes(cached)

        cached_image = image.copy()
        self._image_cache[key] = cached_image
        self._image_cache_bytes += self._estimate_image_bytes(cached_image)
        self._image_cache.move_to_end(key)

        while self._image_cache and (
            (
                self._image_cache_max_bytes > 0
                and self._image_cache_bytes > self._image_cache_max_bytes
            )
            or (
                self._image_cache_max_items > 0
                and len(self._image_cache) > self._image_cache_max_items
            )
        ):
            _, evicted = self._image_cache.popitem(last=False)
            self._image_cache_bytes -= self._estimate_image_bytes(evicted)

    def image_cache_stats(self) -> Dict[str, object]:
        return {
            "enabled": self._image_cache_enabled,
            "items": len(self._image_cache),
            "bytes": int(self._image_cache_bytes),
            "max_bytes": int(self._image_cache_max_bytes),
            "max_items": int(self._image_cache_max_items),
            "hits": int(self._image_cache_hits),
            "misses": int(self._image_cache_misses),
        }

    def _load_rgb_image(self, image_path: Path) -> Image.Image:
        cache_key = str(image_path)
        if self._image_cache_enabled:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache_hits += 1
                self._image_cache.move_to_end(cache_key)
                return cached.copy()
            self._image_cache_misses += 1

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()
        self._put_image_cache(cache_key, image)
        return image

    def __getitem__(self, index: int):
        sample = self.samples[int(index)]
        image = self._load_rgb_image(sample.image_path)
        augmentation_scale = self._augmentation_scale_for_label(sample.label)
        yolo_object = sample.yolo_object
        source_image: Optional[Image.Image] = None
        if yolo_object is not None and sample.yolo_source_image_path is not None:
            if self.classification_source_context or self.classification_source_context_aux:
                source_image = self._load_rgb_image(sample.yolo_source_image_path)

        main_image = image
        main_bbox = (0.5, 0.5, 1.0, 1.0)
        if self.classification_source_context and yolo_object is not None and source_image is not None:
            main_image = self._apply_paired_yolo_source_context(
                image=source_image,
                yolo_object=yolo_object,
            )
            main_bbox = yolo_object.bbox

        transformed_target: Optional[Mapping[str, object]] = None
        if self.transform is not None:
            target = {
                "labels": torch.tensor([sample.label], dtype=torch.long),
                "boxes": torch.tensor([main_bbox], dtype=torch.float32),
                "augmentation_scale": torch.tensor([augmentation_scale], dtype=torch.float32),
            }
            try:
                transformed = self.transform(main_image, target=target)
            except TypeError:
                transformed = self.transform(main_image)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                image_tensor = transformed[0]
                transformed_target = transformed[1] if isinstance(transformed[1], Mapping) else None
            else:
                image_tensor = transformed
        else:
            image_tensor = main_image
            transformed_target = {
                "labels": torch.tensor([sample.label], dtype=torch.long),
                "boxes": torch.tensor([main_bbox], dtype=torch.float32),
                "augmentation_scale": torch.tensor([augmentation_scale], dtype=torch.float32),
            }

        metadata: Dict[str, Tensor] = {}
        if self.classification_bbox_metadata and yolo_object is not None:
            metadata["bbox"] = torch.tensor(yolo_object.bbox, dtype=torch.float32).clamp(0.0, 1.0)
            transformed_boxes = (
                transformed_target.get("boxes")
                if isinstance(transformed_target, Mapping)
                else None
            )
            if (
                torch.is_tensor(transformed_boxes)
                and transformed_boxes.ndim == 2
                and transformed_boxes.size(0) > 0
            ):
                areas = transformed_boxes[:, 2].clamp(min=0.0) * transformed_boxes[:, 3].clamp(min=0.0)
                crop_bbox = transformed_boxes[int(torch.argmax(areas).item())]
            else:
                crop_bbox = torch.tensor(main_bbox, dtype=torch.float32)
            metadata["crop_bbox"] = crop_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)

        image_mask = (
            transformed_target.get("image_mask")
            if isinstance(transformed_target, Mapping)
            else None
        )
        if torch.is_tensor(image_mask):
            metadata["image_mask"] = image_mask.to(dtype=torch.bool)

        if self.classification_source_context_aux and yolo_object is not None and source_image is not None:
            metadata.update(
                self._paired_yolo_source_context_aux_metadata(
                    source_image=source_image,
                    yolo_object=yolo_object,
                    label=int(sample.label),
                )
            )
        if metadata:
            return image_tensor, int(sample.label), metadata
        return image_tensor, int(sample.label)


class MangoYOLOCropDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transform: Optional[Callable] = None,
        crop_margin_ratio: float = 0.05,
        crop_to_primary_object: bool = True,
        fallback_to_full_image: bool = True,
        num_classes: Optional[int] = None,
        primary_object_strategy: str = "largest",
        split: Optional[str] = None,
        class_aware_augmentation: bool = False,
        class_augmentation_power: float = 0.75,
        class_augmentation_max_scale: float = 1.8,
        class_conditional_augmentation_scales: Optional[Sequence[float]] = None,
        classification_target: bool = False,
        classification_object_crops: bool = True,
        class_crop_margin_scale_threshold: float = 1.5,
        class_crop_margin_max_ratio: Optional[float] = None,
        class_crop_margin_scales: Optional[Sequence[float]] = None,
        classification_source_context: bool = False,
        classification_source_context_mode: str = "desaturate_blur",
        classification_source_context_layout: str = "full",
        classification_source_context_margin_ratio: float = 0.12,
        classification_source_context_background_alpha: float = 0.35,
        classification_source_context_blur_radius: float = 7.0,
        classification_source_context_inset_scale: float = 0.34,
        classification_source_context_aux: bool = False,
        classification_bbox_metadata: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.crop_margin_ratio = max(0.0, float(crop_margin_ratio))
        self.crop_to_primary_object = bool(crop_to_primary_object)
        self.fallback_to_full_image = fallback_to_full_image
        self.num_classes = num_classes
        self.primary_object_strategy = str(primary_object_strategy).strip().lower()
        self.split = str(split or "unknown")
        self.class_aware_augmentation = bool(class_aware_augmentation)
        self.class_augmentation_power = max(0.0, float(class_augmentation_power))
        self.class_augmentation_max_scale = max(1.0, float(class_augmentation_max_scale))
        self.class_conditional_augmentation_scales = normalize_class_conditional_augmentation_scales(
            class_conditional_augmentation_scales,
            num_classes=int(self.num_classes or 0),
        )
        self.classification_target = bool(classification_target)
        self.classification_object_crops = bool(classification_object_crops)
        self.classification_source_context = bool(
            self.classification_target and classification_source_context
        )
        self.classification_source_context_aux = bool(
            self.classification_target and classification_source_context_aux
        )
        self.classification_source_context_mode = (
            str(classification_source_context_mode or "desaturate_blur")
            .strip()
            .lower()
            .replace("-", "_")
        )
        valid_source_context_modes = {"dim", "gray", "blur", "mean", "desaturate_blur", "blur_gray"}
        if self.classification_source_context_mode not in valid_source_context_modes:
            raise ValueError(
                "classification_source_context_mode khong hop le: "
                f"{classification_source_context_mode}. Hop le: {sorted(valid_source_context_modes)}"
            )
        self.classification_source_context_layout = (
            str(classification_source_context_layout or "full")
            .strip()
            .lower()
            .replace("-", "_")
        )
        valid_source_context_layouts = {"full", "crop_inset"}
        if self.classification_source_context_layout not in valid_source_context_layouts:
            raise ValueError(
                "classification_source_context_layout khong hop le: "
                f"{classification_source_context_layout}. Hop le: {sorted(valid_source_context_layouts)}"
            )
        self.classification_source_context_margin_ratio = max(
            0.0,
            float(classification_source_context_margin_ratio),
        )
        self.classification_source_context_background_alpha = min(
            1.0,
            max(0.0, float(classification_source_context_background_alpha)),
        )
        self.classification_source_context_blur_radius = max(
            0.1,
            float(classification_source_context_blur_radius),
        )
        self.classification_source_context_inset_scale = min(
            0.75,
            max(0.10, float(classification_source_context_inset_scale)),
        )
        self.classification_bbox_metadata = bool(classification_bbox_metadata)
        if self.classification_source_context:
            self.crop_to_primary_object = False
        self.class_crop_margin_scale_threshold = max(1.0, float(class_crop_margin_scale_threshold))
        base_crop_margin_ratio = self.crop_margin_ratio
        if class_crop_margin_max_ratio is None:
            self.class_crop_margin_max_ratio = base_crop_margin_ratio
        else:
            self.class_crop_margin_max_ratio = max(
                base_crop_margin_ratio,
                max(0.0, float(class_crop_margin_max_ratio)),
            )
        self.class_crop_margin_scales = (
            [max(1.0, float(value)) for value in class_crop_margin_scales]
            if class_crop_margin_scales is not None
            else []
        )
        self._image_cache_enabled = False
        self._image_cache_max_bytes = 0
        self._image_cache_max_items = 0
        self._image_cache_bytes = 0
        self._image_cache_hits = 0
        self._image_cache_misses = 0
        self._image_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._image_paths_by_stem = self._index_image_paths_by_stem()
        self.audit = self._init_audit()
        self.samples = self._index_samples()
        self.class_augmentation_scales = self._build_class_augmentation_scales()
        logger.info(
            "Dataset initialized: split=%s images_dir=%s labels_dir=%s "
            "image_files=%s label_files=%s selected_samples=%s valid_objects=%s "
            "missing_images=%s invalid_bboxes=%s invalid_classes=%s crop_primary=%s "
            "classification_target=%s object_crops=%s source_context=%s context_layout=%s "
            "source_context_aux=%s class_aug=%s scales=%s conditional_scales=%s",
            self.split,
            self.images_dir,
            self.labels_dir,
            self.audit["image_file_count"],
            self.audit["label_file_count"],
            len(self.samples),
            self.audit["valid_object_count"],
            self.audit["missing_image_count"],
            self.audit["invalid_bbox_count"],
            self.audit["invalid_class_count"],
            self.crop_to_primary_object,
            self.classification_target,
            self.classification_object_crops,
            self.classification_source_context,
            self.classification_source_context_layout,
            self.classification_source_context_aux,
            self.class_aware_augmentation,
            self.class_augmentation_scales,
            self.class_conditional_augmentation_scales,
        )

    @classmethod
    def from_data_spec(
        cls,
        data_spec: DataSpec,
        split: str,
        transform: Optional[transforms.Compose] = None,
        crop_margin_ratio: float = 0.05,
        crop_to_primary_object: bool = True,
        fallback_to_full_image: bool = True,
        class_aware_augmentation: bool = False,
        class_augmentation_power: float = 0.75,
        class_augmentation_max_scale: float = 1.8,
        class_conditional_augmentation_scales: Optional[Sequence[float]] = None,
        classification_target: bool = False,
        classification_object_crops: bool = True,
        class_crop_margin_scale_threshold: float = 1.5,
        class_crop_margin_max_ratio: Optional[float] = None,
        class_crop_margin_scales: Optional[Sequence[float]] = None,
        classification_source_context: bool = False,
        classification_source_context_mode: str = "desaturate_blur",
        classification_source_context_layout: str = "full",
        classification_source_context_margin_ratio: float = 0.12,
        classification_source_context_background_alpha: float = 0.35,
        classification_source_context_blur_radius: float = 7.0,
        classification_source_context_inset_scale: float = 0.34,
        classification_source_context_aux: bool = False,
        classification_bbox_metadata: bool = False,
    ) -> "MangoYOLOCropDataset":
        return cls(
            images_dir=data_spec.split_images_dir(split),
            labels_dir=data_spec.split_labels_dir(split),
            transform=transform,
            crop_margin_ratio=crop_margin_ratio,
            crop_to_primary_object=crop_to_primary_object,
            fallback_to_full_image=fallback_to_full_image,
            num_classes=data_spec.num_classes,
            split=split,
            class_aware_augmentation=class_aware_augmentation,
            class_augmentation_power=class_augmentation_power,
            class_augmentation_max_scale=class_augmentation_max_scale,
            class_conditional_augmentation_scales=class_conditional_augmentation_scales,
            classification_target=classification_target,
            classification_object_crops=classification_object_crops,
            class_crop_margin_scale_threshold=class_crop_margin_scale_threshold,
            class_crop_margin_max_ratio=class_crop_margin_max_ratio,
            class_crop_margin_scales=class_crop_margin_scales,
            classification_source_context=classification_source_context,
            classification_source_context_mode=classification_source_context_mode,
            classification_source_context_layout=classification_source_context_layout,
            classification_source_context_margin_ratio=classification_source_context_margin_ratio,
            classification_source_context_background_alpha=classification_source_context_background_alpha,
            classification_source_context_blur_radius=classification_source_context_blur_radius,
            classification_source_context_inset_scale=classification_source_context_inset_scale,
            classification_source_context_aux=classification_source_context_aux,
            classification_bbox_metadata=classification_bbox_metadata,
        )

    @classmethod
    def from_data_yaml(
        cls,
        data_yaml: Path,
        split: str,
        transform: Optional[transforms.Compose] = None,
        crop_margin_ratio: float = 0.05,
        crop_to_primary_object: bool = True,
        fallback_to_full_image: bool = True,
        class_aware_augmentation: bool = False,
        class_augmentation_power: float = 0.75,
        class_augmentation_max_scale: float = 1.8,
        class_conditional_augmentation_scales: Optional[Sequence[float]] = None,
        classification_target: bool = False,
        classification_object_crops: bool = True,
        class_crop_margin_scale_threshold: float = 1.5,
        class_crop_margin_max_ratio: Optional[float] = None,
        class_crop_margin_scales: Optional[Sequence[float]] = None,
        classification_source_context: bool = False,
        classification_source_context_mode: str = "desaturate_blur",
        classification_source_context_layout: str = "full",
        classification_source_context_margin_ratio: float = 0.12,
        classification_source_context_background_alpha: float = 0.35,
        classification_source_context_blur_radius: float = 7.0,
        classification_source_context_inset_scale: float = 0.34,
        classification_source_context_aux: bool = False,
        classification_bbox_metadata: bool = False,
        class_name_mode: Optional[str] = None,
        expected_num_classes: Optional[int] = None,
    ) -> "MangoYOLOCropDataset":
        data_spec = load_data_spec(
            data_yaml,
            class_name_mode=class_name_mode,
            expected_num_classes=expected_num_classes,
        )
        return cls.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=transform,
            crop_margin_ratio=crop_margin_ratio,
            crop_to_primary_object=crop_to_primary_object,
            fallback_to_full_image=fallback_to_full_image,
            class_aware_augmentation=class_aware_augmentation,
            class_augmentation_power=class_augmentation_power,
            class_augmentation_max_scale=class_augmentation_max_scale,
            class_conditional_augmentation_scales=class_conditional_augmentation_scales,
            classification_target=classification_target,
            classification_object_crops=classification_object_crops,
            class_crop_margin_scale_threshold=class_crop_margin_scale_threshold,
            class_crop_margin_max_ratio=class_crop_margin_max_ratio,
            class_crop_margin_scales=class_crop_margin_scales,
            classification_source_context=classification_source_context,
            classification_source_context_mode=classification_source_context_mode,
            classification_source_context_layout=classification_source_context_layout,
            classification_source_context_margin_ratio=classification_source_context_margin_ratio,
            classification_source_context_background_alpha=classification_source_context_background_alpha,
            classification_source_context_blur_radius=classification_source_context_blur_radius,
            classification_source_context_inset_scale=classification_source_context_inset_scale,
            classification_source_context_aux=classification_source_context_aux,
            classification_bbox_metadata=classification_bbox_metadata,
        )

    def _index_image_paths_by_stem(self) -> Dict[str, Path]:
        if not self.images_dir.exists():
            return {}
        image_paths: Dict[str, Path] = {}
        for path in sorted(self.images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_paths.setdefault(path.stem, path)
        return image_paths

    def _find_image_for_stem(self, stem: str) -> Optional[Path]:
        indexed_path = self._image_paths_by_stem.get(stem)
        if indexed_path is not None:
            return indexed_path
        for extension in IMAGE_EXTENSIONS:
            candidate = self.images_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        return None

    def _init_audit(self) -> Dict[str, object]:
        return {
            "image_file_count": len(
                getattr(self, "_image_paths_by_stem", {})
            ),
            "label_file_count": 0,
            "valid_object_count": 0,
            "selected_sample_count": 0,
            "single_object_image_count": 0,
            "multi_object_image_count": 0,
            "max_objects_per_image": 0,
            "object_count_histogram": {},
            "ignored_object_count": 0,
            "missing_image_count": 0,
            "empty_label_count": 0,
            "invalid_line_count": 0,
            "invalid_bbox_count": 0,
            "invalid_class_count": 0,
            "sample_missing_images": [],
            "sample_empty_labels": [],
            "sample_invalid_lines": [],
            "sample_invalid_bboxes": [],
            "sample_invalid_classes": [],
        }

    def _audit_append(self, key: str, value: object, limit: int = 10) -> None:
        sample_list = self.audit[key]
        if len(sample_list) < limit:
            sample_list.append(value)

    def _select_primary_object(
        self,
        objects: Sequence[MangoObject],
    ) -> MangoObject:
        if not objects:
            raise ValueError("Khong the chon primary object tu danh sach rong.")
        if self.primary_object_strategy != "largest":
            return objects[0]
        return max(
            objects,
            key=lambda item: (
                float(item.bbox[2] * item.bbox[3]),
                -abs(float(item.bbox[0]) - 0.5),
                -abs(float(item.bbox[1]) - 0.5),
            ),
        )

    def _select_sample_primary_object(self, sample: MangoSample) -> MangoObject:
        if sample.primary_object_index >= 0:
            for obj in sample.objects:
                if int(obj.object_index) == int(sample.primary_object_index):
                    return obj
        return self._select_primary_object(sample.objects)

    def _index_samples(self) -> List[MangoSample]:
        samples: List[MangoSample] = []
        for label_path in sorted(self.labels_dir.glob("*.txt")):
            self.audit["label_file_count"] += 1
            image_path = self._find_image_for_stem(label_path.stem)
            if image_path is None:
                self.audit["missing_image_count"] += 1
                self._audit_append("sample_missing_images", label_path.name)
                continue
            lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not lines:
                self.audit["empty_label_count"] += 1
                self._audit_append("sample_empty_labels", label_path.name)
                continue

            objects: List[MangoObject] = []
            for object_index, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    self.audit["invalid_line_count"] += 1
                    self._audit_append(
                        "sample_invalid_lines",
                        {"file": label_path.name, "line": object_index + 1, "content": line},
                    )
                    continue
                try:
                    label = int(parts[0])
                    bbox = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    self.audit["invalid_line_count"] += 1
                    self._audit_append(
                        "sample_invalid_lines",
                        {"file": label_path.name, "line": object_index + 1, "content": line},
                    )
                    continue
                if label < 0 or (self.num_classes is not None and label >= self.num_classes):
                    self.audit["invalid_class_count"] += 1
                    self._audit_append(
                        "sample_invalid_classes",
                        {"file": label_path.name, "line": object_index + 1, "label": label},
                    )
                    continue
                if any(value < 0.0 or value > 1.0 for value in bbox) or bbox[2] <= 0.0 or bbox[3] <= 0.0:
                    self.audit["invalid_bbox_count"] += 1
                    self._audit_append(
                        "sample_invalid_bboxes",
                        {"file": label_path.name, "line": object_index + 1, "bbox": bbox},
                    )
                    continue
                objects.append(
                    MangoObject(
                        label=label,
                        bbox=bbox,
                        object_index=object_index,
                    )
                )
                self.audit["valid_object_count"] += 1
            if not objects:
                continue
            object_count = len(objects)
            self.audit["max_objects_per_image"] = max(
                int(self.audit.get("max_objects_per_image", 0)),
                int(object_count),
            )
            histogram = self.audit.get("object_count_histogram")
            if isinstance(histogram, dict):
                key = str(object_count)
                histogram[key] = int(histogram.get(key, 0)) + 1
            if object_count == 1:
                self.audit["single_object_image_count"] += 1
            if object_count > 1:
                self.audit["multi_object_image_count"] += 1
            if self.classification_target and self.classification_object_crops:
                for obj in objects:
                    samples.append(
                        MangoSample(
                            image_path=image_path,
                            label_path=label_path,
                            objects=objects,
                            primary_label=obj.label,
                            primary_object_index=obj.object_index,
                        )
                    )
                    self.audit["selected_sample_count"] += 1
            else:
                primary_object = self._select_primary_object(objects)
                samples.append(
                    MangoSample(
                        image_path=image_path,
                        label_path=label_path,
                        objects=objects,
                        primary_label=primary_object.label,
                        primary_object_index=primary_object.object_index,
                    )
                )
                self.audit["selected_sample_count"] += 1
                if self.crop_to_primary_object:
                    self.audit["ignored_object_count"] += max(0, object_count - 1)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(num_classes)]
        if self.classification_target:
            for sample in self.samples:
                label = int(sample.primary_label)
                if 0 <= label < num_classes:
                    counts[label] += 1
            return counts
        for sample in self.samples:
            for obj in sample.objects:
                counts[obj.label] += 1
        return counts

    def _build_class_augmentation_scales(self) -> List[float]:
        if not self.class_aware_augmentation or self.num_classes is None:
            return [1.0 for _ in range(max(1, int(self.num_classes or 1)))]
        counts = self.class_counts(int(self.num_classes))
        positive_counts = [max(1, int(count)) for count in counts]
        max_count = max(positive_counts) if positive_counts else 1
        scales = []
        for count in positive_counts:
            scale = (float(max_count) / float(count)) ** self.class_augmentation_power
            scales.append(float(min(self.class_augmentation_max_scale, max(1.0, scale))))
        return scales

    def _augmentation_scale_for_label(self, label: int) -> float:
        label = int(label)
        base_scale = 1.0
        if self.class_aware_augmentation and 0 <= label < len(self.class_augmentation_scales):
            base_scale = float(self.class_augmentation_scales[label])
        conditional_scale = 1.0
        if 0 <= label < len(self.class_conditional_augmentation_scales):
            conditional_scale = float(self.class_conditional_augmentation_scales[label])
        return float(base_scale * conditional_scale)

    def _augmentation_scale_for_labels(self, labels: Tensor) -> float:
        if labels.numel() == 0:
            return 1.0
        scales: List[float] = []
        for label in labels.detach().cpu().to(dtype=torch.long).tolist():
            scales.append(self._augmentation_scale_for_label(int(label)))
        return float(max(scales)) if scales else 1.0

    def _crop_margin_scale_for_label(self, label: int) -> float:
        label = int(label)
        if 0 <= label < len(self.class_crop_margin_scales):
            return float(self.class_crop_margin_scales[label])
        if 0 <= label < len(self.class_augmentation_scales):
            return float(self.class_augmentation_scales[label])
        return 1.0

    def _crop_margin_ratio_for_label(self, label: int) -> float:
        base_margin = max(0.0, float(self.crop_margin_ratio))
        if not self.classification_target:
            return base_margin
        if self.class_crop_margin_max_ratio <= base_margin:
            return base_margin
        class_scale = self._crop_margin_scale_for_label(label)
        if class_scale < self.class_crop_margin_scale_threshold:
            return base_margin
        return float(min(self.class_crop_margin_max_ratio, base_margin * class_scale))

    def _crop_margin_ratio_for_sample(self, sample: MangoSample) -> float:
        primary_object = self._select_sample_primary_object(sample)
        return self._crop_margin_ratio_for_label(int(primary_object.label))

    def _class_crop_margin_report(self) -> Dict[str, object]:
        num_classes = int(self.num_classes or 0)
        scales = [
            self._crop_margin_scale_for_label(class_index)
            for class_index in range(max(0, num_classes))
        ]
        margins = [
            self._crop_margin_ratio_for_label(class_index)
            for class_index in range(max(0, num_classes))
        ]
        target_classes = [
            int(class_index)
            for class_index, scale in enumerate(scales)
            if float(scale) >= float(self.class_crop_margin_scale_threshold)
            and float(margins[class_index]) > float(self.crop_margin_ratio)
        ]
        return {
            "enabled": bool(target_classes),
            "base_ratio": float(self.crop_margin_ratio),
            "max_ratio": float(self.class_crop_margin_max_ratio),
            "scale_threshold": float(self.class_crop_margin_scale_threshold),
            "class_scales": [float(value) for value in scales],
            "effective_ratios": [float(value) for value in margins],
            "target_classes": target_classes,
        }

    def labels(self) -> List[int]:
        return [sample.primary_label for sample in self.samples]

    def bboxes(self) -> List[Tuple[float, float, float, float]]:
        return [obj.bbox for sample in self.samples for obj in sample.objects]

    def sample_paths(self) -> List[Path]:
        return [sample.image_path for sample in self.samples]

    def quality_report(self) -> Dict[str, object]:
        report = dict(self.audit)
        report["crop_to_primary_object"] = bool(self.crop_to_primary_object)
        report["classification_target"] = bool(self.classification_target)
        report["classification_object_crops"] = bool(self.classification_object_crops)
        report["classification_source_context"] = {
            "enabled": bool(self.classification_source_context),
            "aux_enabled": bool(self.classification_source_context_aux),
            "mode": str(self.classification_source_context_mode),
            "layout": str(self.classification_source_context_layout),
            "margin_ratio": float(self.classification_source_context_margin_ratio),
            "background_alpha": float(self.classification_source_context_background_alpha),
            "blur_radius": float(self.classification_source_context_blur_radius),
            "inset_scale": float(self.classification_source_context_inset_scale),
        }
        report["classification_bbox_metadata"] = bool(self.classification_bbox_metadata)
        if self.num_classes is not None:
            report["class_counts"] = self.class_counts(self.num_classes)
            report["class_augmentation_scales"] = list(self.class_augmentation_scales)
            report["class_conditional_augmentation_scales"] = list(
                self.class_conditional_augmentation_scales
            )
            report["effective_class_augmentation_scales"] = [
                self._augmentation_scale_for_label(class_index)
                for class_index in range(int(self.num_classes))
            ]
            report["class_crop_margin"] = self._class_crop_margin_report()
        report["class_aware_augmentation"] = bool(self.class_aware_augmentation)
        report["image_cache"] = self.image_cache_stats()
        return report

    def enable_image_cache(
        self,
        max_megabytes: int = 256,
        max_items: int = 0,
    ) -> None:
        self._image_cache_enabled = int(max_megabytes) > 0 or int(max_items) > 0
        self._image_cache_max_bytes = max(0, int(max_megabytes)) * 1024 * 1024
        self._image_cache_max_items = max(0, int(max_items))
        if not self._image_cache_enabled:
            self.clear_image_cache()
        logger.info(
            "Image cache configured: split=%s enabled=%s max_mb=%s max_items=%s",
            self.split,
            self._image_cache_enabled,
            max_megabytes,
            self._image_cache_max_items or None,
        )

    def clear_image_cache(self) -> None:
        self._image_cache.clear()
        self._image_cache_bytes = 0
        self._image_cache_hits = 0
        self._image_cache_misses = 0

    def image_cache_stats(self) -> Dict[str, object]:
        return {
            "enabled": self._image_cache_enabled,
            "items": len(self._image_cache),
            "bytes": int(self._image_cache_bytes),
            "max_bytes": int(self._image_cache_max_bytes),
            "max_items": int(self._image_cache_max_items),
            "hits": int(self._image_cache_hits),
            "misses": int(self._image_cache_misses),
        }

    @staticmethod
    def _estimate_image_bytes(image: Image.Image) -> int:
        bands = max(1, len(image.getbands()))
        width, height = image.size
        return max(1, int(width) * int(height) * bands)

    def _put_image_cache(self, key: str, image: Image.Image) -> None:
        if not self._image_cache_enabled:
            return
        if key in self._image_cache:
            cached = self._image_cache.pop(key)
            self._image_cache_bytes -= self._estimate_image_bytes(cached)

        cached_image = image.copy()
        self._image_cache[key] = cached_image
        self._image_cache_bytes += self._estimate_image_bytes(cached_image)
        self._image_cache.move_to_end(key)

        while self._image_cache and (
            (
                self._image_cache_max_bytes > 0
                and self._image_cache_bytes > self._image_cache_max_bytes
            )
            or (
                self._image_cache_max_items > 0
                and len(self._image_cache) > self._image_cache_max_items
            )
        ):
            _, evicted = self._image_cache.popitem(last=False)
            self._image_cache_bytes -= self._estimate_image_bytes(evicted)

    def _load_rgb_image(self, image_path: Path) -> Image.Image:
        cache_key = str(image_path)
        if self._image_cache_enabled:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache_hits += 1
                self._image_cache.move_to_end(cache_key)
                return cached.copy()
            self._image_cache_misses += 1

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()
        self._put_image_cache(cache_key, image)
        return image

    def _crop_to_primary_object(
        self,
        image: Image.Image,
        labels: Tensor,
        boxes: Tensor,
        sample: MangoSample,
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        if boxes.numel() == 0 or not sample.objects:
            return image, {"labels": labels, "boxes": boxes}

        width, height = image.size
        primary_object = self._select_sample_primary_object(sample)
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(primary_object.bbox, width=width, height=height)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        crop_margin_ratio = self._crop_margin_ratio_for_sample(sample)
        margin_x = box_width * max(0.0, float(crop_margin_ratio))
        margin_y = box_height * max(0.0, float(crop_margin_ratio))

        left = max(0, int(math.floor(x1 - margin_x)))
        top = max(0, int(math.floor(y1 - margin_y)))
        right = min(width, int(math.ceil(x2 + margin_x)))
        bottom = min(height, int(math.ceil(y2 + margin_y)))
        if right <= left or bottom <= top:
            return image, {"labels": labels, "boxes": boxes}

        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)
        kept_labels: List[int] = []
        kept_boxes: List[Tuple[float, float, float, float]] = []
        for label, box in zip(labels.tolist(), boxes.tolist()):
            obj_x1, obj_y1, obj_x2, obj_y2 = bbox_xywh_to_xyxy(box, width=width, height=height)
            inter_x1 = max(float(left), obj_x1)
            inter_y1 = max(float(top), obj_y1)
            inter_x2 = min(float(right), obj_x2)
            inter_y2 = min(float(bottom), obj_y2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            box_w = inter_x2 - inter_x1
            box_h = inter_y2 - inter_y1
            kept_labels.append(int(label))
            kept_boxes.append(
                (
                    float(((inter_x1 + inter_x2) / 2.0 - left) / crop_width),
                    float(((inter_y1 + inter_y2) / 2.0 - top) / crop_height),
                    float(box_w / crop_width),
                    float(box_h / crop_height),
                )
            )

        if not kept_boxes:
            if self.fallback_to_full_image:
                return image, {"labels": labels, "boxes": boxes}
            cropped_empty = image.crop((left, top, right, bottom))
            return cropped_empty, {
                "labels": torch.zeros((0,), dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }

        cropped = image.crop((left, top, right, bottom))
        return cropped, {
            "labels": torch.tensor(kept_labels, dtype=torch.long),
            "boxes": torch.tensor(kept_boxes, dtype=torch.float32).clamp(0.0, 1.0),
        }

    def _apply_classification_source_context(
        self,
        image: Image.Image,
        primary_object: MangoObject,
    ) -> Image.Image:
        base = image.convert("RGB")
        width, height = base.size
        if width <= 0 or height <= 0:
            return base

        x1, y1, x2, y2 = bbox_xywh_to_xyxy(
            primary_object.bbox,
            width=width,
            height=height,
        )
        box_width = max(1.0, float(x2 - x1))
        box_height = max(1.0, float(y2 - y1))
        margin_ratio = max(0.0, float(self.classification_source_context_margin_ratio))
        margin_x = box_width * margin_ratio
        margin_y = box_height * margin_ratio
        left = max(0, int(math.floor(float(x1) - margin_x)))
        top = max(0, int(math.floor(float(y1) - margin_y)))
        right = min(width, int(math.ceil(float(x2) + margin_x)))
        bottom = min(height, int(math.ceil(float(y2) + margin_y)))
        if right <= left or bottom <= top:
            return base

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
        mask = mask.filter(
            ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) * 0.0125))
        )

        mode = self.classification_source_context_mode
        if mode == "dim":
            background = Image.new("RGB", base.size, tuple(_imagenet_fill(IMAGENET_MEAN)))
        elif mode == "gray":
            background = ImageOps.grayscale(base).convert("RGB")
        elif mode == "blur":
            background = base.filter(
                ImageFilter.GaussianBlur(
                    radius=max(0.1, float(self.classification_source_context_blur_radius))
                )
            )
        elif mode == "mean":
            rgb = np.asarray(base, dtype=np.float32)
            mean = np.median(rgb.reshape(-1, 3), axis=0).round().astype(np.uint8)
            background = Image.new("RGB", base.size, tuple(int(value) for value in mean.tolist()))
        elif mode in {"desaturate_blur", "blur_gray"}:
            background = ImageOps.grayscale(base).convert("RGB").filter(
                ImageFilter.GaussianBlur(
                    radius=max(0.1, float(self.classification_source_context_blur_radius))
                )
            )
        else:
            raise ValueError(f"classification_source_context_mode khong hop le: {mode}")

        outside = Image.blend(
            background,
            base,
            float(self.classification_source_context_background_alpha),
        )
        return Image.composite(base, outside, mask)

    def _compose_classification_crop_inset_context(
        self,
        image: Image.Image,
        labels: Tensor,
        boxes: Tensor,
        sample: MangoSample,
        primary_object: MangoObject,
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        primary_labels = torch.tensor([int(primary_object.label)], dtype=torch.long)
        primary_boxes = torch.tensor([primary_object.bbox], dtype=torch.float32)
        crop_image, crop_target = self._crop_to_primary_object(
            image=image,
            labels=primary_labels,
            boxes=primary_boxes,
            sample=sample,
        )
        canvas = crop_image.convert("RGB")
        width, height = canvas.size
        if width < 8 or height < 8:
            return canvas, crop_target

        context = self._apply_classification_source_context(
            image=image,
            primary_object=primary_object,
        )
        max_inset_width = max(4, int(round(width * self.classification_source_context_inset_scale)))
        max_inset_height = max(4, int(round(height * self.classification_source_context_inset_scale)))
        inset = ImageOps.contain(
            context,
            (max_inset_width, max_inset_height),
            method=Image.Resampling.BILINEAR,
        ).convert("RGB")
        pad = max(1, int(round(min(width, height) * 0.025)))
        left = max(0, width - inset.width - pad)
        top = max(0, height - inset.height - pad)
        border = max(1, int(round(min(width, height) * 0.006)))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (
                max(0, left - border),
                max(0, top - border),
                min(width - 1, left + inset.width + border - 1),
                min(height - 1, top + inset.height + border - 1),
            ),
            fill=(245, 245, 245),
        )
        canvas.paste(inset, (left, top))
        return canvas, crop_target

    def _source_context_aux_metadata(
        self,
        *,
        image: Image.Image,
        primary_object: MangoObject,
    ) -> Dict[str, Tensor]:
        context_image = self._apply_classification_source_context(
            image=image,
            primary_object=primary_object,
        )
        context_target = {
            "labels": torch.tensor([int(primary_object.label)], dtype=torch.long),
            "boxes": torch.tensor([primary_object.bbox], dtype=torch.float32),
            "augmentation_scale": torch.tensor(
                [self._augmentation_scale_for_label(int(primary_object.label))],
                dtype=torch.float32,
            ),
        }
        if self.transform is not None:
            transformed = self.transform(context_image, target=context_target)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                context_tensor, transformed_target = transformed
            else:
                raise TypeError("Transform detection phai tra ve (image_tensor, target).")
        else:
            context_tensor = TF.to_tensor(context_image)
            transformed_target = context_target

        bbox_value = transformed_target.get("boxes") if isinstance(transformed_target, dict) else None
        if torch.is_tensor(bbox_value) and bbox_value.ndim == 2 and bbox_value.size(0) > 0:
            context_bbox = bbox_value[0].to(dtype=torch.float32).clamp(0.0, 1.0)
        else:
            context_bbox = torch.tensor(primary_object.bbox, dtype=torch.float32).clamp(0.0, 1.0)
        metadata = {
            "source_context_image": context_tensor,
            "source_context_bbox": context_bbox,
        }
        context_mask = (
            transformed_target.get("image_mask")
            if isinstance(transformed_target, dict)
            else None
        )
        if torch.is_tensor(context_mask):
            metadata["source_context_image_mask"] = context_mask.to(dtype=torch.bool)
        return metadata

    def __getitem__(self, index: int):
        sample = self.samples[index]
        logger.debug(
            "Loading dataset sample: split=%s index=%s image=%s label=%s objects=%s",
            self.split,
            index,
            sample.image_path,
            sample.label_path,
            len(sample.objects),
        )
        try:
            image = self._load_rgb_image(sample.image_path)
        except Exception:
            logger.exception(
                "Failed to load image: split=%s index=%s image=%s",
                self.split,
                index,
                sample.image_path,
            )
            raise

        labels = torch.tensor([obj.label for obj in sample.objects], dtype=torch.long)
        boxes = torch.tensor([obj.bbox for obj in sample.objects], dtype=torch.float32)
        boxes_have_valid_shape = boxes.ndim == 2 and boxes.shape[-1] == 4
        boxes_in_range = bool(((boxes >= 0.0) & (boxes <= 1.0)).all().item()) if boxes.numel() else True
        boxes_have_positive_size = bool((boxes[:, 2:] > 0.0).all().item()) if boxes.numel() else True
        labels_in_range = bool((labels >= 0).all().item()) if labels.numel() else True
        if self.num_classes is not None and labels.numel():
            labels_in_range = labels_in_range and bool((labels < self.num_classes).all().item())
        logger.debug(
            "Raw target validation: split=%s index=%s labels_valid=%s boxes_valid=%s "
            "box_count=%s label_count=%s",
            self.split,
            index,
            labels_in_range,
            boxes_have_valid_shape and boxes_in_range and boxes_have_positive_size,
            int(boxes.shape[0]) if boxes_have_valid_shape else 0,
            int(labels.numel()),
        )
        target = {
            "labels": labels,
            "boxes": boxes,
        }
        primary_object: Optional[MangoObject] = None
        if self.classification_target and sample.objects:
            primary_object = self._select_sample_primary_object(sample)
        source_image = image.copy() if self.classification_source_context_aux else image
        if self.classification_source_context and primary_object is not None:
            if self.classification_source_context_layout == "crop_inset":
                image, target = self._compose_classification_crop_inset_context(
                    image=image,
                    labels=labels,
                    boxes=boxes,
                    sample=sample,
                    primary_object=primary_object,
                )
            else:
                image = self._apply_classification_source_context(
                    image=image,
                    primary_object=primary_object,
                )
                target = {
                    "labels": torch.tensor([int(primary_object.label)], dtype=torch.long),
                    "boxes": torch.tensor([primary_object.bbox], dtype=torch.float32),
                }
        elif self.crop_to_primary_object:
            image, target = self._crop_to_primary_object(
                image=image,
                labels=labels,
                boxes=boxes,
                sample=sample,
            )
        if self.classification_target and primary_object is not None:
            augmentation_scale = self._augmentation_scale_for_label(int(primary_object.label))
        else:
            augmentation_scale = self._augmentation_scale_for_labels(target["labels"])
        target["augmentation_scale"] = torch.tensor([augmentation_scale], dtype=torch.float32)
        if self.transform is not None:
            transformed = self.transform(image, target=target)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                image_tensor, transformed_target = transformed
            else:
                raise TypeError("Transform detection phai tra ve (image_tensor, target).")
        else:
            image_tensor = image
            transformed_target = target

        if self.classification_target:
            label = int(primary_object.label if primary_object is not None else sample.primary_label)
            metadata: Dict[str, Tensor] = {}
            if self.classification_bbox_metadata:
                bbox = primary_object.bbox if primary_object is not None else (0.5, 0.5, 1.0, 1.0)
                metadata["bbox"] = torch.tensor(bbox, dtype=torch.float32)
                transformed_boxes = (
                    transformed_target.get("boxes")
                    if isinstance(transformed_target, dict)
                    else None
                )
                if (
                    torch.is_tensor(transformed_boxes)
                    and transformed_boxes.ndim == 2
                    and transformed_boxes.size(0) > 0
                ):
                    areas = transformed_boxes[:, 2].clamp(min=0.0) * transformed_boxes[:, 3].clamp(min=0.0)
                    crop_bbox = transformed_boxes[int(torch.argmax(areas).item())]
                else:
                    crop_bbox = torch.tensor((0.5, 0.5, 1.0, 1.0), dtype=torch.float32)
                metadata["crop_bbox"] = crop_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)
            transformed_image_mask = (
                transformed_target.get("image_mask")
                if isinstance(transformed_target, dict)
                else None
            )
            if torch.is_tensor(transformed_image_mask):
                metadata["image_mask"] = transformed_image_mask.to(dtype=torch.bool)
            if self.classification_source_context_aux and primary_object is not None:
                metadata.update(
                    self._source_context_aux_metadata(
                        image=source_image,
                        primary_object=primary_object,
                    )
                )
            if metadata:
                return image_tensor, label, metadata
            return image_tensor, label

        target_metadata = {
            key: transformed_target[key]
            for key in ("augmentation_scale", "image_mask")
            if key in transformed_target
        }
        transformed_target = {
            "labels": transformed_target["labels"].to(dtype=torch.long),
            "boxes": transformed_target["boxes"].to(dtype=torch.float32),
        }
        for key, value in target_metadata.items():
            transformed_target[key] = value
        transformed_boxes = transformed_target["boxes"]
        transformed_labels = transformed_target["labels"]
        transformed_boxes_have_valid_shape = (
            transformed_boxes.ndim == 2 and transformed_boxes.shape[-1] == 4
        )
        transformed_boxes_in_range = (
            bool(((transformed_boxes >= 0.0) & (transformed_boxes <= 1.0)).all().item())
            if transformed_boxes.numel()
            else True
        )
        transformed_boxes_have_positive_size = (
            bool((transformed_boxes[:, 2:] > 0.0).all().item())
            if transformed_boxes.numel()
            else True
        )
        transformed_labels_in_range = (
            bool((transformed_labels >= 0).all().item())
            if transformed_labels.numel()
            else True
        )
        if self.num_classes is not None and transformed_labels.numel():
            transformed_labels_in_range = transformed_labels_in_range and bool(
                (transformed_labels < self.num_classes).all().item()
            )
        if torch.is_tensor(image_tensor):
            image_size = tuple(int(value) for value in image_tensor.shape)
        else:
            image_size = getattr(image_tensor, "size", None)
        logger.debug(
            "Transformed sample: split=%s index=%s image_size=%s labels_valid=%s "
            "boxes_valid=%s box_count=%s label_count=%s",
            self.split,
            index,
            image_size,
            transformed_labels_in_range,
            transformed_boxes_have_valid_shape
            and transformed_boxes_in_range
            and transformed_boxes_have_positive_size,
            int(transformed_boxes.shape[0]) if transformed_boxes_have_valid_shape else 0,
            int(transformed_labels.numel()),
        )
        return image_tensor, transformed_target


class MixedTrainDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        source_names: Optional[Sequence[str]] = None,
        source_weights: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> None:
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        if not self.datasets:
            raise ValueError("MixedTrainDataset yeu cau it nhat 1 dataset khong rong.")
        names = list(source_names or [])
        while len(names) < len(self.datasets):
            names.append(f"source_{len(names)}")
        self.source_names = [str(name or f"source_{index}") for index, name in enumerate(names[: len(self.datasets)])]
        weights = list(source_weights or [])
        while len(weights) < len(self.datasets):
            weights.append(1.0)
        self.source_weights = [max(0.0, float(value)) for value in weights[: len(self.datasets)]]
        self.seed = int(seed)
        self.indices = self._build_indices()
        if not self.indices:
            raise ValueError("MixedTrainDataset khong tao duoc sample nao tu source_weights.")

        primary = self.datasets[0]
        self.split = str(getattr(primary, "split", "train"))
        self.classification_target = bool(getattr(primary, "classification_target", True))
        self.classification_object_crops = bool(getattr(primary, "classification_object_crops", False))
        self.crop_to_primary_object = bool(getattr(primary, "crop_to_primary_object", False))
        self.num_classes = getattr(primary, "num_classes", None)
        self.class_augmentation_scales = list(getattr(primary, "class_augmentation_scales", []))

    def _build_indices(self) -> List[Tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mixed: List[Tuple[int, int]] = []
        for source_index, (dataset, weight) in enumerate(zip(self.datasets, self.source_weights)):
            if weight <= 0.0:
                continue
            dataset_len = len(dataset)
            whole = int(math.floor(weight))
            fraction = float(weight) - float(whole)
            if whole <= 0:
                for sample_index in range(dataset_len):
                    if torch.rand(1, generator=generator).item() < fraction:
                        mixed.append((source_index, sample_index))
                continue
            for sample_index in range(dataset_len):
                mixed.extend([(source_index, sample_index)] * whole)
                if fraction > 0.0 and torch.rand(1, generator=generator).item() < fraction:
                    mixed.append((source_index, sample_index))
        if len(mixed) > 1:
            order = torch.randperm(len(mixed), generator=generator).tolist()
            mixed = [mixed[index] for index in order]
        return mixed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index, sample_index = self.indices[int(index)]
        return self.datasets[int(source_index)][int(sample_index)]

    def labels(self) -> List[int]:
        labels_by_source: List[List[int]] = []
        for dataset in self.datasets:
            labels_fn = getattr(dataset, "labels", None)
            labels_by_source.append([int(value) for value in labels_fn()] if callable(labels_fn) else [])
        labels: List[int] = []
        for source_index, sample_index in self.indices:
            source_labels = labels_by_source[int(source_index)]
            if 0 <= int(sample_index) < len(source_labels):
                labels.append(int(source_labels[int(sample_index)]))
        return labels

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(max(1, int(num_classes)))]
        for label in self.labels():
            if 0 <= int(label) < len(counts):
                counts[int(label)] += 1
        return counts

    def sample_paths(self) -> List[Path]:
        paths_by_source: List[List[Path]] = []
        for dataset in self.datasets:
            sample_paths_fn = getattr(dataset, "sample_paths", None)
            paths_by_source.append([Path(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else [])
        paths: List[Path] = []
        for source_index, sample_index in self.indices:
            source_paths = paths_by_source[int(source_index)]
            if 0 <= int(sample_index) < len(source_paths):
                paths.append(Path(source_paths[int(sample_index)]))
        return paths

    def bboxes(self) -> List[Tuple[float, float, float, float]]:
        bboxes_by_source: List[List[Tuple[float, float, float, float]]] = []
        for dataset in self.datasets:
            bboxes_fn = getattr(dataset, "bboxes", None)
            if callable(bboxes_fn):
                source_bboxes = [tuple(float(value) for value in bbox) for bbox in bboxes_fn()]
            else:
                source_bboxes = [(0.5, 0.5, 1.0, 1.0) for _ in range(len(dataset))]
            bboxes_by_source.append(source_bboxes)
        bboxes: List[Tuple[float, float, float, float]] = []
        for source_index, sample_index in self.indices:
            source_bboxes = bboxes_by_source[int(source_index)]
            if 0 <= int(sample_index) < len(source_bboxes):
                bboxes.append(source_bboxes[int(sample_index)])
        return bboxes

    @staticmethod
    def _sum_report_key(reports: Sequence[Dict[str, object]], key: str) -> int:
        return int(sum(int(report.get(key, 0) or 0) for report in reports))

    def _source_summaries(self) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        effective_counts = Counter(source_index for source_index, _ in self.indices)
        for source_index, dataset in enumerate(self.datasets):
            report_fn = getattr(dataset, "quality_report", None)
            report = report_fn() if callable(report_fn) else {}
            labels_fn = getattr(dataset, "labels", None)
            source_labels = [int(value) for value in labels_fn()] if callable(labels_fn) else []
            summaries.append(
                {
                    "name": self.source_names[source_index],
                    "weight": float(self.source_weights[source_index]),
                    "base_samples": int(len(dataset)),
                    "effective_samples": int(effective_counts.get(source_index, 0)),
                    "class_counts": [
                        int(sum(1 for label in source_labels if int(label) == class_index))
                        for class_index in range(max(1, int(self.num_classes or 1)))
                    ],
                    "report": report,
                }
            )
        return summaries

    def quality_report(self) -> Dict[str, object]:
        source_reports = []
        for dataset in self.datasets:
            report_fn = getattr(dataset, "quality_report", None)
            source_reports.append(report_fn() if callable(report_fn) else {})
        report: Dict[str, object] = {
            "data_format": "mixed_train",
            "classification_target": True,
            "classification_object_crops": any(
                bool(getattr(dataset, "classification_object_crops", False))
                for dataset in self.datasets
            ),
            "crop_to_primary_object": any(
                bool(getattr(dataset, "crop_to_primary_object", False))
                for dataset in self.datasets
            ),
            "image_file_count": self._sum_report_key(source_reports, "image_file_count"),
            "label_file_count": self._sum_report_key(source_reports, "label_file_count"),
            "valid_object_count": self._sum_report_key(source_reports, "valid_object_count"),
            "selected_sample_count": int(len(self)),
            "single_object_image_count": self._sum_report_key(source_reports, "single_object_image_count"),
            "multi_object_image_count": self._sum_report_key(source_reports, "multi_object_image_count"),
            "max_objects_per_image": max(
                [int(report.get("max_objects_per_image", 0) or 0) for report in source_reports],
                default=0,
            ),
            "ignored_object_count": self._sum_report_key(source_reports, "ignored_object_count"),
            "empty_label_count": self._sum_report_key(source_reports, "empty_label_count"),
            "missing_image_count": self._sum_report_key(source_reports, "missing_image_count"),
            "invalid_line_count": self._sum_report_key(source_reports, "invalid_line_count"),
            "invalid_bbox_count": self._sum_report_key(source_reports, "invalid_bbox_count"),
            "invalid_class_count": self._sum_report_key(source_reports, "invalid_class_count"),
            "class_counts": self.class_counts(max(1, int(self.num_classes or 1))),
            "class_augmentation_scales": list(self.class_augmentation_scales),
            "class_aware_augmentation": any(
                bool(getattr(dataset, "class_aware_augmentation", False))
                for dataset in self.datasets
            ),
            "image_cache": self.image_cache_stats(),
            "mixed_train": {
                "enabled": True,
                "seed": int(self.seed),
                "source_count": int(len(self.datasets)),
                "effective_samples": int(len(self)),
                "sources": self._source_summaries(),
            },
        }
        return report

    def enable_image_cache(
        self,
        max_megabytes: int = 256,
        max_items: int = 0,
    ) -> None:
        for dataset in self.datasets:
            enable_fn = getattr(dataset, "enable_image_cache", None)
            if callable(enable_fn):
                enable_fn(max_megabytes=max_megabytes, max_items=max_items)

    def clear_image_cache(self) -> None:
        for dataset in self.datasets:
            clear_fn = getattr(dataset, "clear_image_cache", None)
            if callable(clear_fn):
                clear_fn()

    def image_cache_stats(self) -> Dict[str, object]:
        source_stats = []
        for source_name, dataset in zip(self.source_names, self.datasets):
            stats_fn = getattr(dataset, "image_cache_stats", None)
            stats = stats_fn() if callable(stats_fn) else {}
            stats = dict(stats)
            stats["source"] = str(source_name)
            source_stats.append(stats)
        return {
            "enabled": any(bool(stats.get("enabled", False)) for stats in source_stats),
            "items": int(sum(int(stats.get("items", 0) or 0) for stats in source_stats)),
            "bytes": int(sum(int(stats.get("bytes", 0) or 0) for stats in source_stats)),
            "sources": source_stats,
        }


class PairedViewTrainDataset(Dataset):
    """Attach a second train-only view of the same object to each primary sample."""

    def __init__(
        self,
        primary_dataset: Dataset,
        paired_dataset: Dataset,
        *,
        primary_name: str = "primary",
        paired_name: str = "paired",
        require_all_matched: bool = True,
    ) -> None:
        self.primary_dataset = primary_dataset
        self.paired_dataset = paired_dataset
        self.primary_name = str(primary_name or "primary")
        self.paired_name = str(paired_name or "paired")
        self.require_all_matched = bool(require_all_matched)

        self._paired_index_by_key = self._build_index(paired_dataset)
        self.pairs: List[Tuple[int, int]] = []
        self.missing_primary_keys: List[Tuple[str, int, int]] = []
        for primary_index in range(len(primary_dataset)):
            key = self._sample_key(primary_dataset, primary_index)
            paired_index = self._paired_index_by_key.get(key)
            if paired_index is None:
                self.missing_primary_keys.append(key)
                continue
            self.pairs.append((primary_index, paired_index))
        if not self.pairs:
            raise ValueError("PairedViewTrainDataset khong map duoc cap object nao.")
        if self.require_all_matched and len(self.pairs) != len(primary_dataset):
            preview = self.missing_primary_keys[:5]
            raise ValueError(
                "PairedViewTrainDataset thieu paired view cho primary samples: "
                f"matched={len(self.pairs)}/{len(primary_dataset)} preview={preview}"
            )

        self.split = str(getattr(primary_dataset, "split", "train"))
        self.classification_target = bool(getattr(primary_dataset, "classification_target", True))
        self.classification_object_crops = bool(
            getattr(primary_dataset, "classification_object_crops", False)
        )
        self.crop_to_primary_object = bool(getattr(primary_dataset, "crop_to_primary_object", False))
        self.num_classes = getattr(primary_dataset, "num_classes", None)
        self.class_augmentation_scales = list(
            getattr(primary_dataset, "class_augmentation_scales", [])
        )

    @staticmethod
    def _labels(dataset: Dataset) -> List[int]:
        labels_fn = getattr(dataset, "labels", None)
        return [int(value) for value in labels_fn()] if callable(labels_fn) else []

    @staticmethod
    def _sample_paths(dataset: Dataset) -> List[Path]:
        sample_paths_fn = getattr(dataset, "sample_paths", None)
        return [Path(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []

    @classmethod
    def _sample_key(cls, dataset: Dataset, index: int) -> Tuple[str, int, int]:
        samples = getattr(dataset, "samples", None)
        labels = cls._labels(dataset)
        fallback_label = int(labels[index]) if 0 <= int(index) < len(labels) else -1
        if isinstance(samples, Sequence) and 0 <= int(index) < len(samples):
            sample = samples[int(index)]
            yolo_source_id = getattr(sample, "yolo_source_id", "")
            yolo_object = getattr(sample, "yolo_object", None)
            if yolo_source_id and yolo_object is not None:
                return (
                    str(yolo_source_id),
                    int(getattr(yolo_object, "object_index", 0)),
                    int(fallback_label),
                )
            image_path = getattr(sample, "image_path", None)
            objects = getattr(sample, "objects", None)
            if image_path is not None and objects:
                selector = getattr(dataset, "_select_sample_primary_object", None)
                primary_object = selector(sample) if callable(selector) else objects[0]
                return (
                    Path(image_path).stem,
                    int(getattr(primary_object, "object_index", 0)),
                    int(getattr(primary_object, "label", fallback_label)),
                )
        paths = cls._sample_paths(dataset)
        if 0 <= int(index) < len(paths):
            path = paths[int(index)]
            match = CLASSIFICATION_CROP_NAME.match(path.stem)
            if match is not None:
                return (str(match.group("source")), int(match.group("box")), int(fallback_label))
            return (path.stem, 0, int(fallback_label))
        return (str(index), 0, int(fallback_label))

    @classmethod
    def _build_index(cls, dataset: Dataset) -> Dict[Tuple[str, int, int], int]:
        index_by_key: Dict[Tuple[str, int, int], int] = {}
        duplicate_keys: List[Tuple[str, int, int]] = []
        for sample_index in range(len(dataset)):
            key = cls._sample_key(dataset, sample_index)
            if key in index_by_key:
                duplicate_keys.append(key)
                continue
            index_by_key[key] = int(sample_index)
        if duplicate_keys:
            raise ValueError(
                "PairedViewTrainDataset paired source co duplicate object key, preview="
                f"{duplicate_keys[:5]}"
            )
        return index_by_key

    @staticmethod
    def _split_sample(sample) -> Tuple[object, int, Dict[str, object]]:
        if isinstance(sample, tuple) and len(sample) >= 3 and isinstance(sample[2], dict):
            return sample[0], int(sample[1]), dict(sample[2])
        if isinstance(sample, tuple) and len(sample) >= 2:
            return sample[0], int(sample[1]), {}
        raise TypeError("PairedViewTrainDataset chi ho tro classification samples tuple.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        primary_index, paired_index = self.pairs[int(index)]
        primary_image, primary_label, primary_metadata = self._split_sample(
            self.primary_dataset[int(primary_index)]
        )
        paired_image, paired_label, paired_metadata = self._split_sample(
            self.paired_dataset[int(paired_index)]
        )
        primary_metadata["paired_view_image"] = paired_image
        primary_metadata["paired_view_label"] = int(paired_label)
        if "bbox" in paired_metadata:
            primary_metadata["paired_view_bbox"] = paired_metadata["bbox"]
        if "crop_bbox" in paired_metadata:
            primary_metadata["paired_view_crop_bbox"] = paired_metadata["crop_bbox"]
        if "image_mask" in paired_metadata:
            primary_metadata["paired_view_image_mask"] = paired_metadata["image_mask"]
        return primary_image, int(primary_label), primary_metadata

    def labels(self) -> List[int]:
        labels = self._labels(self.primary_dataset)
        return [int(labels[primary_index]) for primary_index, _ in self.pairs]

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(max(1, int(num_classes)))]
        for label in self.labels():
            if 0 <= int(label) < len(counts):
                counts[int(label)] += 1
        return counts

    def sample_paths(self) -> List[Path]:
        paths = self._sample_paths(self.primary_dataset)
        return [Path(paths[primary_index]) for primary_index, _ in self.pairs]

    def bboxes(self) -> List[Tuple[float, float, float, float]]:
        bboxes_fn = getattr(self.primary_dataset, "bboxes", None)
        if not callable(bboxes_fn):
            return [(0.5, 0.5, 1.0, 1.0) for _ in self.pairs]
        bboxes = [tuple(float(value) for value in bbox) for bbox in bboxes_fn()]
        return [
            bboxes[primary_index]
            if 0 <= int(primary_index) < len(bboxes)
            else (0.5, 0.5, 1.0, 1.0)
            for primary_index, _ in self.pairs
        ]

    def quality_report(self) -> Dict[str, object]:
        report_fn = getattr(self.primary_dataset, "quality_report", None)
        report = dict(report_fn() if callable(report_fn) else {})
        paired_report_fn = getattr(self.paired_dataset, "quality_report", None)
        report["data_format"] = "paired_view_train"
        report["selected_sample_count"] = int(len(self))
        report["class_counts"] = self.class_counts(max(1, int(self.num_classes or 1)))
        report["image_cache"] = self.image_cache_stats()
        report["paired_view_train"] = {
            "enabled": True,
            "primary_name": self.primary_name,
            "paired_name": self.paired_name,
            "primary_base_samples": int(len(self.primary_dataset)),
            "paired_base_samples": int(len(self.paired_dataset)),
            "matched_samples": int(len(self.pairs)),
            "missing_primary_count": int(len(self.missing_primary_keys)),
            "missing_primary_preview": [list(key) for key in self.missing_primary_keys[:5]],
            "paired_report": paired_report_fn() if callable(paired_report_fn) else {},
        }
        return report

    def enable_image_cache(
        self,
        max_megabytes: int = 256,
        max_items: int = 0,
    ) -> None:
        for dataset in (self.primary_dataset, self.paired_dataset):
            enable_fn = getattr(dataset, "enable_image_cache", None)
            if callable(enable_fn):
                enable_fn(max_megabytes=max_megabytes, max_items=max_items)

    def clear_image_cache(self) -> None:
        for dataset in (self.primary_dataset, self.paired_dataset):
            clear_fn = getattr(dataset, "clear_image_cache", None)
            if callable(clear_fn):
                clear_fn()

    def image_cache_stats(self) -> Dict[str, object]:
        source_stats = []
        for source_name, dataset in (
            (self.primary_name, self.primary_dataset),
            (self.paired_name, self.paired_dataset),
        ):
            stats_fn = getattr(dataset, "image_cache_stats", None)
            stats = stats_fn() if callable(stats_fn) else {}
            stats = dict(stats)
            stats["source"] = str(source_name)
            source_stats.append(stats)
        return {
            "enabled": any(bool(stats.get("enabled", False)) for stats in source_stats),
            "items": int(sum(int(stats.get("items", 0) or 0) for stats in source_stats)),
            "bytes": int(sum(int(stats.get("bytes", 0) or 0) for stats in source_stats)),
            "sources": source_stats,
        }


class StrictBalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_classes: int,
        epoch_multiplier: float = 1.0,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size phai >= 1.")
        self.labels = [int(label) for label in labels]
        self.batch_size = int(batch_size)
        self.num_classes = max(1, int(num_classes))
        self.epoch_multiplier = max(1.0, float(epoch_multiplier))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        invalid_labels = sorted({label for label in self.labels if label < 0 or label >= self.num_classes})
        if invalid_labels:
            raise ValueError(
                "StrictBalancedBatchSampler phat hien label nam ngoai khoang [0, num_classes): "
                f"{invalid_labels}"
            )

        self.class_to_indices: Dict[int, List[int]] = {class_index: [] for class_index in range(self.num_classes)}
        for sample_index, label in enumerate(self.labels):
            self.class_to_indices[label].append(sample_index)

        self.active_classes = [
            class_index for class_index in range(self.num_classes) if self.class_to_indices[class_index]
        ]
        if not self.active_classes:
            raise ValueError("StrictBalancedBatchSampler yeu cau it nhat 1 lop co sample.")

        target_samples = int(math.ceil(len(self.labels) * self.epoch_multiplier))
        target_samples = max(self.batch_size, target_samples)
        if self.drop_last:
            self.num_batches = max(1, target_samples // self.batch_size)
        else:
            self.num_batches = max(1, math.ceil(target_samples / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _allocation_for_batch(self, batch_index: int) -> List[Tuple[int, int]]:
        class_count = len(self.active_classes)
        base = self.batch_size // class_count
        remainder = self.batch_size % class_count
        allocations: List[Tuple[int, int]] = [
            (class_index, base) for class_index in self.active_classes
        ]

        if remainder > 0:
            # Rotate every extra slot globally so epoch exposure differs by at most one.
            start = (batch_index * remainder) % class_count
            for extra_index in range(remainder):
                class_index = self.active_classes[(start + extra_index) % class_count]
                for allocation_index, (allocated_class, count) in enumerate(allocations):
                    if allocated_class == class_index:
                        allocations[allocation_index] = (allocated_class, count + 1)
                        break
        return [(class_index, count) for class_index, count in allocations if count > 0]

    def exposure_counts(self, num_batches: Optional[int] = None) -> List[int]:
        effective_num_batches = (
            self.num_batches
            if num_batches is None
            else min(self.num_batches, max(0, int(num_batches)))
        )
        counts = [0 for _ in range(self.num_classes)]
        for batch_index in range(effective_num_batches):
            for class_index, sample_count in self._allocation_for_batch(batch_index):
                counts[int(class_index)] += int(sample_count)
        return counts

    def exposure_summary(self, num_batches: Optional[int] = None) -> Dict[str, object]:
        effective_num_batches = (
            self.num_batches
            if num_batches is None
            else min(self.num_batches, max(0, int(num_batches)))
        )
        counts = self.exposure_counts(num_batches=effective_num_batches)
        positive = [count for count in counts if count > 0]
        minimum = min(positive) if positive else 0
        maximum = max(positive) if positive else 0
        relative_gap = (
            float(maximum - minimum) / float(maximum)
            if maximum > 0
            else 0.0
        )
        return {
            "class_exposure_counts": counts,
            "min_class_exposure": int(minimum),
            "max_class_exposure": int(maximum),
            "relative_gap": float(relative_gap),
            "total_samples": int(sum(counts)),
            "num_batches": int(effective_num_batches),
            "configured_num_batches": int(self.num_batches),
        }

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + max(0, self.epoch))

        class_pools: Dict[int, List[int]] = {}
        class_positions: Dict[int, int] = {}
        for class_index in self.active_classes:
            base_indices = torch.tensor(self.class_to_indices[class_index], dtype=torch.long)
            shuffled = base_indices[torch.randperm(len(base_indices), generator=generator)].tolist()
            class_pools[class_index] = shuffled
            class_positions[class_index] = 0

        def draw_index(class_index: int) -> int:
            position = class_positions[class_index]
            pool = class_pools[class_index]
            if position >= len(pool):
                base_indices = torch.tensor(self.class_to_indices[class_index], dtype=torch.long)
                pool = base_indices[torch.randperm(len(base_indices), generator=generator)].tolist()
                class_pools[class_index] = pool
                class_positions[class_index] = 0
                position = 0
            sampled_index = int(pool[position])
            class_positions[class_index] = position + 1
            return sampled_index

        for batch_index in range(self.num_batches):
            batch_indices: List[int] = []
            for class_index, sample_count in self._allocation_for_batch(batch_index):
                for _ in range(sample_count):
                    batch_indices.append(draw_index(class_index))
            shuffle_order = torch.randperm(len(batch_indices), generator=generator).tolist()
            yield [batch_indices[index] for index in shuffle_order]


class TemperedClassBatchSampler(StrictBalancedBatchSampler):
    """Deterministic class-prior sampler with q_c proportional to n_c ** power."""

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_classes: int,
        power: float,
        epoch_multiplier: float = 1.0,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        power = float(power)
        if not math.isfinite(power) or not 0.0 <= power <= 1.0:
            raise ValueError("TemperedClassBatchSampler power phai huu han va nam trong [0, 1].")
        self.power = power
        super().__init__(
            labels=labels,
            batch_size=batch_size,
            num_classes=num_classes,
            epoch_multiplier=epoch_multiplier,
            seed=seed,
            drop_last=drop_last,
        )
        if self.batch_size < len(self.active_classes):
            raise ValueError(
                "TemperedClassBatchSampler yeu cau batch_size >= so active classes "
                "de bao dam quota deterministic va moi class co mat trong moi batch."
            )

        raw_weights = [
            (
                float(len(self.class_to_indices[class_index])) ** self.power
                if class_index in self.active_classes
                else 0.0
            )
            for class_index in range(self.num_classes)
        ]
        weight_sum = float(sum(raw_weights))
        if weight_sum <= 0.0:
            raise ValueError("TemperedClassBatchSampler khong the tao class prior rong.")
        self.class_probabilities = [
            float(weight / weight_sum) for weight in raw_weights
        ]
        self._batch_allocations = self._build_batch_allocations()

    def _build_batch_allocations(self) -> List[List[Tuple[int, int]]]:
        residual = [0.0 for _ in range(self.num_classes)]
        allocations_by_batch: List[List[Tuple[int, int]]] = []
        for _batch_index in range(self.num_batches):
            raw = [
                float(self.batch_size) * self.class_probabilities[class_index]
                + residual[class_index]
                for class_index in range(self.num_classes)
            ]
            allocated = [
                int(math.floor(value + 1e-12)) if value > 0.0 else 0
                for value in raw
            ]
            remaining = int(self.batch_size - sum(allocated))
            if remaining < 0:
                raise RuntimeError("TemperedClassBatchSampler allocation vuot batch_size.")
            ranked_classes = sorted(
                self.active_classes,
                key=lambda class_index: (
                    -(raw[class_index] - math.floor(raw[class_index])),
                    class_index,
                ),
            )
            if remaining > len(ranked_classes):
                raise RuntimeError("TemperedClassBatchSampler allocation residual khong hop le.")
            for class_index in ranked_classes[:remaining]:
                allocated[class_index] += 1
            residual = [
                raw[class_index] - float(allocated[class_index])
                for class_index in range(self.num_classes)
            ]
            allocations_by_batch.append(
                [
                    (class_index, allocated[class_index])
                    for class_index in self.active_classes
                    if allocated[class_index] > 0
                ]
            )
        return allocations_by_batch

    def _allocation_for_batch(self, batch_index: int) -> List[Tuple[int, int]]:
        return list(self._batch_allocations[int(batch_index)])

    def exposure_summary(self, num_batches: Optional[int] = None) -> Dict[str, object]:
        summary = super().exposure_summary(num_batches=num_batches)
        effective_num_batches = int(summary["num_batches"])
        ideal_counts = [
            float(effective_num_batches * self.batch_size) * probability
            for probability in self.class_probabilities
        ]
        observed_counts = [
            int(value) for value in summary["class_exposure_counts"]
        ]
        max_prefix_error = 0.0
        cumulative = [0 for _ in range(self.num_classes)]
        for batch_index in range(effective_num_batches):
            for class_index, sample_count in self._allocation_for_batch(batch_index):
                cumulative[class_index] += int(sample_count)
            max_prefix_error = max(
                max_prefix_error,
                max(
                    abs(
                        float(cumulative[class_index])
                        - float((batch_index + 1) * self.batch_size)
                        * self.class_probabilities[class_index]
                    )
                    for class_index in range(self.num_classes)
                ),
            )
        summary.update(
            {
                "power": float(self.power),
                "source_class_counts": [
                    int(len(self.class_to_indices[class_index]))
                    for class_index in range(self.num_classes)
                ],
                "class_probabilities": list(self.class_probabilities),
                "ideal_class_exposure_counts": ideal_counts,
                "class_quota_errors": [
                    float(observed - ideal)
                    for observed, ideal in zip(observed_counts, ideal_counts)
                ],
                "max_prefix_absolute_quota_error": float(max_prefix_error),
            }
        )
        return summary


def build_rare_class_repeat_factors(
    class_counts: Sequence[int],
    repeat_power: float = 0.5,
    max_factor: float = 3.0,
    min_ratio: float = 0.35,
) -> List[float]:
    counts = [max(0, int(count)) for count in class_counts]
    if not counts:
        return []
    positive_counts = [count for count in counts if count > 0]
    if not positive_counts:
        return [1.0 for _ in counts]

    max_count = max(positive_counts)
    repeat_power = max(0.0, float(repeat_power))
    max_factor = max(1.0, float(max_factor))
    min_ratio = max(0.0, min(1.0, float(min_ratio)))

    factors: List[float] = []
    for count in counts:
        if count <= 0:
            factors.append(1.0)
            continue
        ratio = float(count) / float(max_count)
        if ratio >= min_ratio or repeat_power <= 0.0:
            factors.append(1.0)
            continue
        factor = (1.0 / max(ratio, 1e-12)) ** repeat_power
        factors.append(float(min(max_factor, max(1.0, factor))))
    return factors


class RareClassRepeatDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        class_repeat_factors: Sequence[float],
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.class_repeat_factors = [max(1.0, float(value)) for value in class_repeat_factors]
        self.seed = int(seed)
        self.indices = self._build_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[int(index)]]

    def _sample_labels(self, sample_index: int) -> List[int]:
        samples = getattr(self.dataset, "samples", None)
        if samples is not None:
            sample = samples[int(sample_index)]
            if bool(getattr(self.dataset, "classification_target", False)):
                if hasattr(sample, "primary_label"):
                    return [int(getattr(sample, "primary_label"))]
                if hasattr(sample, "label"):
                    return [int(getattr(sample, "label"))]
                return []
            objects = getattr(sample, "objects", [])
            return [int(getattr(obj, "label")) for obj in objects]
        labels_fn = getattr(self.dataset, "labels", None)
        if callable(labels_fn):
            labels = labels_fn()
            if 0 <= int(sample_index) < len(labels):
                return [int(labels[int(sample_index)])]
        return []

    def _factor_for_sample(self, sample_index: int) -> float:
        factor = 1.0
        for label in self._sample_labels(sample_index):
            if 0 <= label < len(self.class_repeat_factors):
                factor = max(factor, float(self.class_repeat_factors[label]))
        return factor

    def _build_indices(self) -> List[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        repeated: List[int] = []
        for sample_index in range(len(self.dataset)):
            factor = self._factor_for_sample(sample_index)
            whole = int(math.floor(factor))
            repeated.extend([sample_index] * max(1, whole))
            fractional = factor - float(whole)
            if fractional > 0.0 and torch.rand(1, generator=generator).item() < fractional:
                repeated.append(sample_index)
        if not repeated:
            repeated = list(range(len(self.dataset)))
        order = torch.randperm(len(repeated), generator=generator).tolist()
        return [int(repeated[index]) for index in order]

    def labels(self) -> List[int]:
        base_labels_fn = getattr(self.dataset, "labels", None)
        if not callable(base_labels_fn):
            return []
        base_labels = [int(label) for label in base_labels_fn()]
        return [base_labels[index] for index in self.indices if 0 <= int(index) < len(base_labels)]

    def repeat_summary(self) -> Dict[str, object]:
        base_len = len(self.dataset)
        repeated_len = len(self.indices)
        return {
            "base_samples": int(base_len),
            "effective_samples": int(repeated_len),
            "effective_multiplier": float(repeated_len / max(1, base_len)),
            "class_repeat_factors": [float(value) for value in self.class_repeat_factors],
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["rare_class_repeat"] = self.repeat_summary()
        return report


class HardSampleRepeatDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        hard_sample_paths: Sequence[Path | str],
        repeat_factor: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.repeat_factor = max(1.0, float(repeat_factor))
        self.seed = int(seed)
        self.hard_sample_paths = {
            str(Path(path).resolve()).lower()
            for path in hard_sample_paths
            if str(path).strip()
        }
        self.indices = self._build_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[int(index)]]

    def _sample_paths(self) -> List[str]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return [str(Path(path).resolve()).lower() for path in sample_paths_fn()]
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        paths: List[str] = []
        for sample in samples:
            image_path = getattr(sample, "image_path", None)
            paths.append("" if image_path is None else str(Path(image_path).resolve()).lower())
        return paths

    def _build_indices(self) -> List[int]:
        base_paths = self._sample_paths()
        hard_indices = [
            index
            for index, sample_path in enumerate(base_paths)
            if sample_path in self.hard_sample_paths
        ]
        repeated = list(range(len(self.dataset)))
        if hard_indices:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            extra_factor = max(0.0, self.repeat_factor - 1.0)
            whole = int(math.floor(extra_factor))
            fractional = extra_factor - float(whole)
            for sample_index in hard_indices:
                repeated.extend([sample_index] * whole)
                if fractional > 0.0 and torch.rand(1, generator=generator).item() < fractional:
                    repeated.append(sample_index)
            order = torch.randperm(len(repeated), generator=generator).tolist()
            repeated = [int(repeated[index]) for index in order]
        return repeated

    def labels(self) -> List[int]:
        base_labels_fn = getattr(self.dataset, "labels", None)
        if not callable(base_labels_fn):
            return []
        base_labels = [int(label) for label in base_labels_fn()]
        return [base_labels[index] for index in self.indices if 0 <= int(index) < len(base_labels)]

    def repeat_summary(self) -> Dict[str, object]:
        base_len = len(self.dataset)
        repeated_len = len(self.indices)
        matched_count = max(0, repeated_len - base_len)
        return {
            "base_samples": int(base_len),
            "effective_samples": int(repeated_len),
            "effective_multiplier": float(repeated_len / max(1, base_len)),
            "manifest_paths": int(len(self.hard_sample_paths)),
            "extra_repeats": int(matched_count),
            "repeat_factor": float(self.repeat_factor),
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["hard_sample_repeat"] = self.repeat_summary()
        return report


class SampleWeightDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        sample_weights_by_path: Mapping[str, float],
        *,
        sample_weights_by_sample_index: Optional[Mapping[int, float]] = None,
        default_weight: float = 1.0,
        max_weight: float = 5.0,
    ) -> None:
        self.dataset = dataset
        self.default_weight = max(0.0, float(default_weight))
        self.max_weight = max(self.default_weight, float(max_weight))
        self.sample_weights_by_path = {
            str(Path(path).resolve()).lower(): float(
                min(self.max_weight, max(0.0, weight))
            )
            for path, weight in sample_weights_by_path.items()
            if str(path).strip()
        }
        self.sample_weights_by_sample_index = {
            int(index): float(min(self.max_weight, max(0.0, weight)))
            for index, weight in (sample_weights_by_sample_index or {}).items()
            if int(index) >= 0
        }
        self._sample_paths = self._collect_sample_paths()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def sample_weight_summary(self) -> Dict[str, object]:
        weights = [
            self._weight_for_index(index, path)
            for index, path in enumerate(self._sample_paths)
        ]
        matched = sum(
            1
            for index, path in enumerate(self._sample_paths)
            if int(index) in self.sample_weights_by_sample_index
            or str(Path(path).resolve()).lower() in self.sample_weights_by_path
        )
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else torch.tensor([])
        return {
            "enabled": True,
            "samples": int(len(self._sample_paths)),
            "weighted_paths": int(len(self.sample_weights_by_path)),
            "weighted_sample_indices": int(len(self.sample_weights_by_sample_index)),
            "key_mode": "sample_index" if self.sample_weights_by_sample_index else "path",
            "matched_samples": int(matched),
            "default_weight": float(self.default_weight),
            "max_weight": float(self.max_weight),
            "mean_weight": float(weight_tensor.mean().item()) if weight_tensor.numel() else 0.0,
            "max_observed_weight": float(weight_tensor.max().item()) if weight_tensor.numel() else 0.0,
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["sample_weights"] = self.sample_weight_summary()
        return report

    def _weight_for_index(self, index: int, sample_path: Path) -> float:
        sample_index = int(index)
        if sample_index in self.sample_weights_by_sample_index:
            return float(self.sample_weights_by_sample_index[sample_index])
        key = str(Path(sample_path).resolve()).lower()
        return float(self.sample_weights_by_path.get(key, self.default_weight))

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_path = self._sample_paths[int(index)]
        sample_weight = torch.tensor(
            self._weight_for_index(int(index), sample_path),
            dtype=torch.float32,
        )
        if len(item) == 2:
            image, label = item
            return image, label, {"sample_weight": sample_weight}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata["sample_weight"] = sample_weight
            return item[0], item[1], metadata
        return item


class QualityGroupDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        group_indices_by_path: Mapping[str, int],
        *,
        group_indices_by_sample_index: Optional[Mapping[int, int]] = None,
        default_group_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.default_group_index = max(0, int(default_group_index))
        self.group_indices_by_path = {
            str(Path(path).resolve()).lower(): max(0, int(group_index))
            for path, group_index in group_indices_by_path.items()
            if str(path).strip()
        }
        self.group_indices_by_sample_index = {
            int(index): max(0, int(group_index))
            for index, group_index in (group_indices_by_sample_index or {}).items()
            if int(index) >= 0
        }
        self._sample_paths = self._collect_sample_paths()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def quality_group_summary(self) -> Dict[str, object]:
        group_counts: Dict[int, int] = {}
        matched = 0
        for index, path in enumerate(self._sample_paths):
            group_index = self._group_for_index(int(index), path)
            group_counts[group_index] = int(group_counts.get(group_index, 0)) + 1
            key = str(Path(path).resolve()).lower()
            if int(index) in self.group_indices_by_sample_index or key in self.group_indices_by_path:
                matched += 1
        return {
            "enabled": True,
            "samples": int(len(self._sample_paths)),
            "manifest_paths": int(len(self.group_indices_by_path)),
            "manifest_sample_indices": int(len(self.group_indices_by_sample_index)),
            "key_mode": "sample_index" if self.group_indices_by_sample_index else "path",
            "matched_samples": int(matched),
            "default_group_index": int(self.default_group_index),
            "num_observed_groups": int(len(group_counts)),
            "group_counts": {str(key): int(value) for key, value in sorted(group_counts.items())},
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["quality_groups"] = self.quality_group_summary()
        return report

    def _group_for_index(self, index: int, sample_path: Optional[Path]) -> int:
        if int(index) in self.group_indices_by_sample_index:
            return int(self.group_indices_by_sample_index[int(index)])
        if sample_path is not None:
            key = str(Path(sample_path).resolve()).lower()
            if key in self.group_indices_by_path:
                return int(self.group_indices_by_path[key])
        return int(self.default_group_index)

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_path = self._sample_paths[int(index)] if int(index) < len(self._sample_paths) else None
        group_index = torch.tensor(
            self._group_for_index(int(index), sample_path),
            dtype=torch.long,
        )
        if len(item) == 2:
            image, label = item
            return image, label, {"quality_group_index": group_index}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata["quality_group_index"] = group_index
            return item[0], item[1], metadata
        return item


class AmbiguousSoftTargetDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        soft_targets_by_path: Mapping[str, Sequence[float]],
        *,
        num_classes: int,
        default_alpha: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.default_alpha = max(0.0, min(1.0, float(default_alpha)))
        self.soft_targets_by_path = {
            str(Path(path).resolve()).lower(): torch.tensor(
                list(probabilities),
                dtype=torch.float32,
            )
            for path, probabilities in soft_targets_by_path.items()
            if str(path).strip()
        }
        raw_by_sample_index = getattr(soft_targets_by_path, "by_sample_index", None)
        self.soft_targets_by_sample_index = {
            int(index): torch.tensor(list(probabilities), dtype=torch.float32)
            for index, probabilities in (raw_by_sample_index or {}).items()
            if int(index) >= 0
        }
        self._sample_paths = self._collect_sample_paths()
        self._labels = self._collect_labels()
        invalid_paths = [
            path
            for path, probabilities in self.soft_targets_by_path.items()
            if probabilities.numel() != self.num_classes
            or not torch.isfinite(probabilities).all()
            or float(probabilities.sum().item()) <= 0.0
        ]
        if invalid_paths:
            raise ValueError(
                "Ambiguous soft-target manifest co dong khong hop le: "
                f"count={len(invalid_paths)} preview={invalid_paths[:5]}"
            )
        invalid_indices = [
            index
            for index, probabilities in self.soft_targets_by_sample_index.items()
            if probabilities.numel() != self.num_classes
            or not torch.isfinite(probabilities).all()
            or float(probabilities.sum().item()) <= 0.0
        ]
        if invalid_indices:
            raise ValueError(
                "Ambiguous soft-target manifest co sample_index khong hop le: "
                f"count={len(invalid_indices)} preview={invalid_indices[:5]}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def _collect_labels(self) -> List[int]:
        labels_fn = getattr(self.dataset, "labels", None)
        if callable(labels_fn):
            return [int(value) for value in labels_fn()]
        samples = getattr(self.dataset, "samples", None)
        if samples is not None:
            return [
                int(getattr(sample, "label"))
                for sample in samples
                if getattr(sample, "label", None) is not None
            ]
        return []

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def _hard_target_for_index(self, index: int, label: int) -> Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if 0 <= int(label) < self.num_classes:
            target[int(label)] = 1.0
        return target

    def soft_target_summary(self) -> Dict[str, object]:
        matched = sum(
            1
            for path in self._sample_paths
            if str(Path(path).resolve()).lower() in self.soft_targets_by_path
        )
        matched_by_sample_index = sum(
            1 for index in range(len(self.dataset)) if int(index) in self.soft_targets_by_sample_index
        )
        return {
            "enabled": True,
            "samples": int(len(self._sample_paths)),
            "num_classes": int(self.num_classes),
            "manifest_paths": int(len(self.soft_targets_by_path)),
            "manifest_sample_indices": int(len(self.soft_targets_by_sample_index)),
            "matched_samples": int(matched),
            "matched_sample_indices": int(matched_by_sample_index),
            "default_alpha": float(self.default_alpha),
            "key_mode": "sample_index" if self.soft_targets_by_sample_index else "path",
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["ambiguous_soft_targets"] = self.soft_target_summary()
        return report

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_path = self._sample_paths[int(index)] if int(index) < len(self._sample_paths) else None
        key = str(Path(sample_path).resolve()).lower() if sample_path is not None else ""
        label = int(item[1]) if len(item) >= 2 else 0
        if int(index) in self.soft_targets_by_sample_index:
            soft_target = self.soft_targets_by_sample_index[int(index)].clone()
            soft_target = soft_target / soft_target.sum().clamp(min=1e-12)
        elif key in self.soft_targets_by_path:
            soft_target = self.soft_targets_by_path[key].clone()
            soft_target = soft_target / soft_target.sum().clamp(min=1e-12)
        else:
            soft_target = self._hard_target_for_index(int(index), label)
        if len(item) == 2:
            image, label = item
            return image, label, {"soft_target": soft_target}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata["soft_target"] = soft_target
            return item[0], item[1], metadata
        return item


class TargetedMarginDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        margin_specs_by_path: Mapping[str, Mapping[str, float]],
        *,
        margin_specs_by_sample_index: Optional[Mapping[int, Mapping[str, float]]] = None,
        default_weight: float = 1.0,
        default_margin: float = 0.12,
        max_weight: float = 3.0,
    ) -> None:
        self.dataset = dataset
        self.default_weight = max(0.0, float(default_weight))
        self.default_margin = max(0.0, float(default_margin))
        self.max_weight = max(self.default_weight, float(max_weight))
        self.margin_specs_by_path: Dict[str, Dict[str, float]] = {}
        for path, spec in margin_specs_by_path.items():
            if not str(path).strip():
                continue
            negative_index = int(spec.get("negative_index", -1))
            if negative_index < 0:
                continue
            weight = float(spec.get("weight", self.default_weight))
            margin = float(spec.get("margin", self.default_margin))
            self.margin_specs_by_path[str(Path(path).resolve()).lower()] = {
                "negative_index": float(negative_index),
                "weight": float(min(self.max_weight, max(0.0, weight))),
                "margin": float(max(0.0, margin)),
            }
        self.margin_specs_by_sample_index: Dict[int, Dict[str, float]] = {}
        for index, spec in (margin_specs_by_sample_index or {}).items():
            sample_index = int(index)
            if sample_index < 0:
                continue
            negative_index = int(spec.get("negative_index", -1))
            if negative_index < 0:
                continue
            weight = float(spec.get("weight", self.default_weight))
            margin = float(spec.get("margin", self.default_margin))
            self.margin_specs_by_sample_index[sample_index] = {
                "negative_index": float(negative_index),
                "weight": float(min(self.max_weight, max(0.0, weight))),
                "margin": float(max(0.0, margin)),
            }
        self._sample_paths = self._collect_sample_paths()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def targeted_margin_summary(self) -> Dict[str, object]:
        matched = 0
        weights: List[float] = []
        margins: List[float] = []
        for index, path in enumerate(self._sample_paths):
            spec = self.margin_specs_by_sample_index.get(int(index))
            if spec is None:
                spec = self.margin_specs_by_path.get(str(Path(path).resolve()).lower())
            if spec is None:
                continue
            matched += 1
            weights.append(float(spec["weight"]))
            margins.append(float(spec["margin"]))
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else torch.tensor([])
        margin_tensor = torch.tensor(margins, dtype=torch.float32) if margins else torch.tensor([])
        return {
            "enabled": True,
            "samples": int(len(self._sample_paths)),
            "manifest_paths": int(len(self.margin_specs_by_path)),
            "manifest_sample_indices": int(len(self.margin_specs_by_sample_index)),
            "key_mode": "sample_index" if self.margin_specs_by_sample_index else "path",
            "matched_samples": int(matched),
            "default_weight": float(self.default_weight),
            "default_margin": float(self.default_margin),
            "max_weight": float(self.max_weight),
            "mean_weight": float(weight_tensor.mean().item()) if weight_tensor.numel() else 0.0,
            "mean_margin": float(margin_tensor.mean().item()) if margin_tensor.numel() else 0.0,
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["targeted_margin"] = self.targeted_margin_summary()
        return report

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_path = self._sample_paths[int(index)] if int(index) < len(self._sample_paths) else None
        key = str(Path(sample_path).resolve()).lower() if sample_path is not None else ""
        spec = self.margin_specs_by_sample_index.get(int(index))
        if spec is None:
            spec = self.margin_specs_by_path.get(key)
        if spec is None:
            negative_index = torch.tensor(-1, dtype=torch.long)
            weight = torch.tensor(0.0, dtype=torch.float32)
            margin = torch.tensor(self.default_margin, dtype=torch.float32)
        else:
            negative_index = torch.tensor(int(spec["negative_index"]), dtype=torch.long)
            weight = torch.tensor(float(spec["weight"]), dtype=torch.float32)
            margin = torch.tensor(float(spec["margin"]), dtype=torch.float32)
        metadata_update = {
            "targeted_margin_negative": negative_index,
            "targeted_margin_weight": weight,
            "targeted_margin_margin": margin,
        }
        if len(item) == 2:
            image, label = item
            return image, label, metadata_update
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata.update(metadata_update)
            return item[0], item[1], metadata
        return item


class FocusNeighborBinaryDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        specs_by_path: Mapping[str, Mapping[str, float]],
        *,
        default_weight: float = 1.0,
        max_weight: float = 3.0,
    ) -> None:
        self.dataset = dataset
        self.default_weight = max(0.0, float(default_weight))
        self.max_weight = max(self.default_weight, float(max_weight))
        self.specs_by_path: Dict[str, Dict[str, float]] = {}
        for path, spec in specs_by_path.items():
            if not str(path).strip():
                continue
            target = float(spec.get("binary_target", -1.0))
            if target < 0.0 or target > 1.0:
                continue
            weight = float(spec.get("weight", self.default_weight))
            neighbor_index = int(spec.get("neighbor_index", -1))
            self.specs_by_path[str(Path(path).resolve()).lower()] = {
                "binary_target": float(max(0.0, min(1.0, target))),
                "weight": float(min(self.max_weight, max(0.0, weight))),
                "neighbor_index": float(neighbor_index),
            }
        self._sample_paths = self._collect_sample_paths()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def focus_neighbor_binary_summary(self) -> Dict[str, object]:
        matched = 0
        positive = 0
        negative = 0
        weights: List[float] = []
        neighbor_counts: Dict[int, int] = {}
        for path in self._sample_paths:
            spec = self.specs_by_path.get(str(Path(path).resolve()).lower())
            if spec is None:
                continue
            matched += 1
            target = float(spec["binary_target"])
            if target >= 0.5:
                positive += 1
            else:
                negative += 1
            weights.append(float(spec["weight"]))
            neighbor_index = int(spec.get("neighbor_index", -1))
            if neighbor_index >= 0:
                neighbor_counts[neighbor_index] = int(neighbor_counts.get(neighbor_index, 0)) + 1
        weight_tensor = torch.tensor(weights, dtype=torch.float32) if weights else torch.tensor([])
        return {
            "enabled": True,
            "samples": int(len(self._sample_paths)),
            "manifest_paths": int(len(self.specs_by_path)),
            "matched_samples": int(matched),
            "positive_samples": int(positive),
            "negative_samples": int(negative),
            "default_weight": float(self.default_weight),
            "max_weight": float(self.max_weight),
            "mean_weight": float(weight_tensor.mean().item()) if weight_tensor.numel() else 0.0,
            "max_observed_weight": float(weight_tensor.max().item()) if weight_tensor.numel() else 0.0,
            "neighbor_counts": {str(key): int(value) for key, value in sorted(neighbor_counts.items())},
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["focus_neighbor_binary"] = self.focus_neighbor_binary_summary()
        return report

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_path = self._sample_paths[int(index)] if int(index) < len(self._sample_paths) else None
        key = str(Path(sample_path).resolve()).lower() if sample_path is not None else ""
        spec = self.specs_by_path.get(key)
        if spec is None:
            target = torch.tensor(0.0, dtype=torch.float32)
            weight = torch.tensor(0.0, dtype=torch.float32)
            neighbor_index = torch.tensor(-1, dtype=torch.long)
        else:
            target = torch.tensor(float(spec["binary_target"]), dtype=torch.float32)
            weight = torch.tensor(float(spec["weight"]), dtype=torch.float32)
            neighbor_index = torch.tensor(int(spec.get("neighbor_index", -1)), dtype=torch.long)
        metadata_update = {
            "focus_neighbor_binary_target": target,
            "focus_neighbor_binary_weight": weight,
            "focus_neighbor_binary_neighbor": neighbor_index,
        }
        if len(item) == 2:
            image, label = item
            return image, label, metadata_update
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata.update(metadata_update)
            return item[0], item[1], metadata
        return item


class TeacherProbabilityDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        probabilities_by_path: Mapping[str, Sequence[float]],
        *,
        num_classes: int,
        probabilities_by_sample_index: Optional[Mapping[int, Sequence[float]]] = None,
        metadata_key: str = "teacher_probs",
    ) -> None:
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.metadata_key = str(metadata_key or "teacher_probs").strip()
        if not self.metadata_key:
            raise ValueError("TeacherProbabilityDataset metadata_key khong duoc rong.")
        self.probabilities_by_path = {
            str(Path(path).resolve()).lower(): torch.tensor(
                list(probabilities),
                dtype=torch.float32,
            )
            for path, probabilities in probabilities_by_path.items()
        }
        self.probabilities_by_sample_index = {
            int(index): torch.tensor(
                list(probabilities),
                dtype=torch.float32,
            )
            for index, probabilities in (probabilities_by_sample_index or {}).items()
        }
        self._sample_paths = self._collect_sample_paths()
        sample_paths = self._sample_paths
        self.sample_index_path_overlap_count = 0
        self.sample_index_path_overlap_ratio = 0.0
        if self.probabilities_by_sample_index and self.probabilities_by_path:
            sample_path_keys = {
                str(Path(path).resolve()).lower()
                for path in sample_paths
            }
            self.sample_index_path_overlap_count = sum(
                1 for key in sample_path_keys if key in self.probabilities_by_path
            )
            self.sample_index_path_overlap_ratio = float(
                self.sample_index_path_overlap_count / max(1, len(sample_path_keys))
            )
        if self.probabilities_by_sample_index:
            missing_indices = [
                int(index)
                for index in range(len(self.dataset))
                if int(index) not in self.probabilities_by_sample_index
            ]
            if missing_indices:
                raise ValueError(
                    "Teacher probability cache khong phu het train sample indices: "
                    f"missing={len(missing_indices)}/{len(self.dataset)} "
                    f"preview={missing_indices[:5]}"
                )
        else:
            missing_paths = [
                str(path)
                for path in sample_paths
                if str(Path(path).resolve()).lower() not in self.probabilities_by_path
            ]
            if missing_paths:
                preview = missing_paths[:5]
                raise ValueError(
                    "Teacher probability cache khong phu het train samples: "
                    f"missing={len(missing_paths)}/{len(sample_paths)} preview={preview}"
                )
        invalid_paths = [
            str(path)
            for path, probabilities in self.probabilities_by_path.items()
            if probabilities.numel() != self.num_classes
            or not torch.isfinite(probabilities).all()
            or float(probabilities.sum().item()) <= 0.0
        ]
        invalid_indices = [
            int(index)
            for index, probabilities in self.probabilities_by_sample_index.items()
            if probabilities.numel() != self.num_classes
            or not torch.isfinite(probabilities).all()
            or float(probabilities.sum().item()) <= 0.0
        ]
        if invalid_paths:
            raise ValueError(
                "Teacher probability cache co dong khong hop le: "
                f"count={len(invalid_paths)} preview={invalid_paths[:5]}"
            )
        if invalid_indices:
            raise ValueError(
                "Teacher probability cache co sample index khong hop le: "
                f"count={len(invalid_indices)} preview={invalid_indices[:5]}"
            )
        if self.probabilities_by_sample_index and self.sample_index_path_overlap_ratio < 0.5:
            labels_fn = getattr(self.dataset, "labels", None)
            labels = list(labels_fn()) if callable(labels_fn) else []
            if labels and len(labels) == len(self.dataset):
                probabilities = torch.stack(
                    [
                        self._teacher_probs_for_index(index)
                        for index in range(len(self.dataset))
                    ],
                    dim=0,
                )
                predictions = probabilities.argmax(dim=1)
                label_tensor = torch.tensor(labels, dtype=torch.long)
                agreement = float((predictions.cpu() == label_tensor).float().mean().item())
                if agreement < 0.5:
                    sample_path_keys = {
                        str(Path(path).resolve()).lower()
                        for path in sample_paths
                    }
                    raise ValueError(
                        "Teacher probability CSV co sample_index nhung path gan nhu khong "
                        "khop dataset hien tai va teacher-label agreement rat thap. Co "
                        "the dang dung sample_index cua view khac "
                        f"(path_overlap={self.sample_index_path_overlap_count}/"
                        f"{len(sample_path_keys)}, "
                        f"overlap_ratio={self.sample_index_path_overlap_ratio:.4f}, "
                        f"agreement={agreement:.4f}). Hay remap teacher CSV sang sample "
                        "order cua dataset primary truoc khi train."
                    )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def _teacher_probs_for_index(self, index: int) -> Tensor:
        if self.probabilities_by_sample_index:
            teacher_probs = self.probabilities_by_sample_index[int(index)]
        else:
            sample_path = self._sample_paths[int(index)]
            key = str(Path(sample_path).resolve()).lower()
            teacher_probs = self.probabilities_by_path[key]
        return teacher_probs / teacher_probs.sum().clamp(min=1e-12)

    def teacher_probability_summary(self) -> Dict[str, object]:
        paths = self._sample_paths
        probabilities = torch.stack(
            [
                self._teacher_probs_for_index(index)
                for index in range(len(self.dataset))
            ],
            dim=0,
        )
        predictions = probabilities.argmax(dim=1)
        labels_fn = getattr(self.dataset, "labels", None)
        labels = list(labels_fn()) if callable(labels_fn) else []
        agreement = 0.0
        if labels and len(labels) == len(paths):
            label_tensor = torch.tensor(labels, dtype=torch.long)
            agreement = float((predictions.cpu() == label_tensor).float().mean().item())
        return {
            "enabled": True,
            "samples": len(paths),
            "num_classes": int(self.num_classes),
            "mean_confidence": float(probabilities.max(dim=1).values.mean().item()),
            "teacher_label_agreement": agreement,
            "key_mode": (
                "sample_index"
                if self.probabilities_by_sample_index
                else "path"
            ),
            "sample_index_path_overlap_count": int(self.sample_index_path_overlap_count),
            "sample_index_path_overlap_ratio": float(self.sample_index_path_overlap_ratio),
            "metadata_key": self.metadata_key,
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report_key = (
            "teacher_probabilities"
            if self.metadata_key == "teacher_probs"
            else f"{self.metadata_key}_summary"
        )
        report[report_key] = self.teacher_probability_summary()
        return report

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        teacher_probs = self._teacher_probs_for_index(int(index))
        if len(item) == 2:
            image, label = item
            return image, label, {self.metadata_key: teacher_probs.clone()}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata[self.metadata_key] = teacher_probs.clone()
            return item[0], item[1], metadata
        return item


class TeacherFeatureDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        features_by_path: Mapping[str, Sequence[float]],
        *,
        features_by_sample_index: Optional[Mapping[int, Sequence[float]]] = None,
    ) -> None:
        self.dataset = dataset
        self.features_by_path = {
            str(Path(path).resolve()).lower(): torch.tensor(
                list(features),
                dtype=torch.float32,
            )
            for path, features in features_by_path.items()
        }
        self.features_by_sample_index = {
            int(index): torch.tensor(
                list(features),
                dtype=torch.float32,
            )
            for index, features in (features_by_sample_index or {}).items()
        }
        self._sample_paths = self._collect_sample_paths()

        if self.features_by_sample_index:
            missing_indices = [
                int(index)
                for index in range(len(self.dataset))
                if int(index) not in self.features_by_sample_index
            ]
            if missing_indices:
                raise ValueError(
                    "Teacher feature cache khong phu het train sample indices: "
                    f"missing={len(missing_indices)}/{len(self.dataset)} "
                    f"preview={missing_indices[:5]}"
                )
        else:
            missing_paths = [
                str(path)
                for path in self._sample_paths
                if str(Path(path).resolve()).lower() not in self.features_by_path
            ]
            if missing_paths:
                raise ValueError(
                    "Teacher feature cache khong phu het train samples: "
                    f"missing={len(missing_paths)}/{len(self._sample_paths)} "
                    f"preview={missing_paths[:5]}"
                )

        dimensions = {
            int(feature.numel())
            for feature in list(self.features_by_sample_index.values())
            + list(self.features_by_path.values())
        }
        if len(dimensions) != 1:
            raise ValueError(
                "Teacher feature cache phai co feature_dim dong nhat, "
                f"found={sorted(dimensions)}"
            )
        self.feature_dim = int(next(iter(dimensions))) if dimensions else 0
        invalid_indices = [
            int(index)
            for index, feature in self.features_by_sample_index.items()
            if feature.ndim != 1 or not torch.isfinite(feature).all()
        ]
        invalid_paths = [
            str(path)
            for path, feature in self.features_by_path.items()
            if feature.ndim != 1 or not torch.isfinite(feature).all()
        ]
        if self.feature_dim <= 0 or invalid_indices or invalid_paths:
            raise ValueError(
                "Teacher feature cache co vector khong hop le: "
                f"feature_dim={self.feature_dim} "
                f"invalid_indices={invalid_indices[:5]} invalid_paths={invalid_paths[:5]}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _collect_sample_paths(self) -> List[Path]:
        sample_paths_fn = getattr(self.dataset, "sample_paths", None)
        if callable(sample_paths_fn):
            return list(sample_paths_fn())
        samples = getattr(self.dataset, "samples", None)
        if samples is None:
            return []
        return [
            Path(getattr(sample, "image_path"))
            for sample in samples
            if getattr(sample, "image_path", None) is not None
        ]

    def sample_paths(self) -> List[Path]:
        return list(self._sample_paths)

    def _teacher_features_for_index(self, index: int) -> Tensor:
        if self.features_by_sample_index:
            return self.features_by_sample_index[int(index)]
        sample_path = self._sample_paths[int(index)]
        key = str(Path(sample_path).resolve()).lower()
        return self.features_by_path[key]

    def teacher_feature_summary(self) -> Dict[str, object]:
        features = torch.stack(
            [
                self._teacher_features_for_index(index)
                for index in range(len(self.dataset))
            ],
            dim=0,
        )
        norms = torch.linalg.vector_norm(features.float(), ord=2, dim=1)
        return {
            "enabled": True,
            "samples": int(len(self.dataset)),
            "feature_dim": int(self.feature_dim),
            "path_samples": int(len(self.features_by_path)),
            "sample_index_samples": int(len(self.features_by_sample_index)),
            "key_mode": "sample_index" if self.features_by_sample_index else "path",
            "mean_norm": float(norms.mean().item()) if norms.numel() else 0.0,
            "std_norm": float(norms.std(unbiased=False).item()) if norms.numel() else 0.0,
        }

    def quality_report(self) -> Dict[str, object]:
        quality_fn = getattr(self.dataset, "quality_report", None)
        report = quality_fn() if callable(quality_fn) else {}
        report = dict(report)
        report["teacher_features"] = self.teacher_feature_summary()
        return report

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        teacher_features = self._teacher_features_for_index(int(index))
        if len(item) == 2:
            image, label = item
            return image, label, {"teacher_features": teacher_features.clone()}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata["teacher_features"] = teacher_features.clone()
            return item[0], item[1], metadata
        return item


class IndexedSampleDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, index: int):
        item = self.dataset[int(index)]
        sample_index = torch.tensor(int(index), dtype=torch.long)
        if len(item) == 2:
            image, label = item
            return image, label, {"sample_index": sample_index}
        if len(item) >= 3 and isinstance(item[2], dict):
            metadata = dict(item[2])
            metadata["sample_index"] = sample_index
            return item[0], item[1], metadata
        return item


class ResizePadToSquare:
    def __init__(
        self,
        image_size: int,
        fill: Sequence[int] = (124, 116, 104),
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        self.image_size = image_size
        self.fill = tuple(int(value) for value in fill)
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        resized, _, _ = self.apply_with_meta(image)
        return resized

    def apply_with_meta(
        self,
        image: Image.Image,
        fill: Optional[Sequence[int]] = None,
        interpolation: Optional[InterpolationMode] = None,
        force_rgb: bool = True,
    ) -> Tuple[Image.Image, float, Tuple[int, int, int, int]]:
        if force_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        if width <= 0 or height <= 0:
            empty = Image.new("RGB", (self.image_size, self.image_size), color=self.fill)
            return empty, 1.0, (0, 0, 0, 0)

        scale = min(self.image_size / width, self.image_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=PIL_RESAMPLE_MAP.get(
                interpolation or self.interpolation,
                Image.Resampling.BICUBIC,
            ),
        )

        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        padded = ImageOps.expand(
            resized,
            border=(left, top, right, bottom),
            fill=tuple(fill) if fill is not None else self.fill,
        )
        return padded, float(scale), (left, top, right, bottom)


def _bbox_to_mask(
    image_size: Tuple[int, int],
    bbox: Tuple[float, float, float, float],
) -> Image.Image:
    width, height = image_size
    x_center, y_center, box_width, box_height = bbox
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = max(0, int(round(x_center - box_width / 2.0)))
    y1 = max(0, int(round(y_center - box_height / 2.0)))
    x2 = min(width, int(round(x_center + box_width / 2.0)))
    y2 = min(height, int(round(y_center + box_height / 2.0)))

    mask = Image.new("L", (width, height), color=0)
    if x2 > x1 and y2 > y1:
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x1, y1, x2, y2), fill=255)
    return mask


def _bbox_from_mask(mask: Image.Image) -> Tuple[float, float, float, float]:
    mask_array = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask_array > 0)
    width, height = mask.size
    if xs.size == 0 or ys.size == 0 or width <= 0 or height <= 0:
        return (
            0.5,
            0.5,
            1.0 / float(max(1, width)),
            1.0 / float(max(1, height)),
        )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    return (
        float(x_center / width),
        float(y_center / height),
        float(box_width / width),
        float(box_height / height),
    )


def _bbox_from_mask_or_none(mask: Image.Image) -> Optional[Tuple[float, float, float, float]]:
    mask_array = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask_array > 0)
    width, height = mask.size
    if xs.size == 0 or ys.size == 0 or width <= 0 or height <= 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    return (
        float(x_center / width),
        float(y_center / height),
        float(box_width / width),
        float(box_height / height),
    )


def _boxes_to_masks(
    image_size: Tuple[int, int],
    boxes: Sequence[Sequence[float]],
) -> List[Image.Image]:
    return [
        _bbox_to_mask(
            image_size=image_size,
            bbox=(
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ),
        )
        for box in boxes
    ]


def _target_from_masks(
    masks: Sequence[Image.Image],
    labels: Tensor,
) -> Dict[str, Tensor]:
    kept_labels: List[int] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for label, mask in zip(labels.tolist(), masks):
        bbox = _bbox_from_mask_or_none(mask)
        if bbox is None:
            continue
        kept_labels.append(int(label))
        kept_boxes.append(bbox)

    if not kept_boxes:
        return {
            "labels": torch.zeros((0,), dtype=torch.long),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
        }
    return {
        "labels": torch.tensor(kept_labels, dtype=torch.long),
        "boxes": torch.tensor(kept_boxes, dtype=torch.float32),
    }


def denormalize_bbox_xywh(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = [float(value) for value in bbox]
    return (
        x_center * float(width),
        y_center * float(height),
        box_width * float(width),
        box_height * float(height),
    )


def bbox_xywh_to_xyxy(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = denormalize_bbox_xywh(
        bbox,
        width=width,
        height=height,
    )
    x1 = x_center - box_width / 2.0
    y1 = y_center - box_height / 2.0
    x2 = x_center + box_width / 2.0
    y2 = y_center + box_height / 2.0
    return x1, y1, x2, y2


def _center_crop_square_pair(
    image: Image.Image,
    mask: Optional[Union[Image.Image, Sequence[Image.Image]]] = None,
) -> Tuple[Image.Image, Optional[Union[Image.Image, List[Image.Image]]], Tuple[int, int, int, int]]:
    width, height = image.size
    side = max(1, min(width, height))
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    image = image.crop((left, top, right, bottom))
    if mask is not None:
        if isinstance(mask, Image.Image):
            mask = mask.crop((left, top, right, bottom))
        else:
            mask = [item.crop((left, top, right, bottom)) for item in mask]
    return image, mask, (left, top, right, bottom)


def _bbox_from_bool_mask_or_none(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not bool(mask.any()):
        return None
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _crop_to_foreground_bbox_pair(
    image: Image.Image,
    masks: Optional[List[Image.Image]] = None,
    *,
    mode: str = "none",
    margin_ratio: float = 0.08,
    min_mask_area_ratio: float = 0.03,
    max_mask_area_ratio: float = 0.92,
    max_crop_area_ratio: float = 0.98,
) -> Tuple[Image.Image, Optional[List[Image.Image]], Optional[Tuple[int, int, int, int]]]:
    normalized_mode = str(mode or "none").strip().lower().replace("-", "_")
    if normalized_mode in {"", "none", "off", "false"}:
        return image, masks, None

    width, height = image.size
    if width <= 4 or height <= 4:
        return image, masks, None

    mask_array: Optional[np.ndarray] = None
    if masks:
        target_union = np.zeros((height, width), dtype=bool)
        for mask in masks:
            if mask.size != image.size:
                continue
            target_union |= np.asarray(mask, dtype=np.uint8) > 0
        if bool(target_union.any()):
            mask_array = target_union

    if mask_array is None:
        if normalized_mode == "grabcut":
            mask_array = _grabcut_foreground_mask_array(image)
        elif normalized_mode == "pseudo":
            mask_array = _pseudo_foreground_mask_array(image)
        else:
            raise ValueError(f"foreground_crop_mode khong hop le: {mode}")

    mask_fraction = float(np.asarray(mask_array, dtype=bool).mean())
    if mask_fraction < float(min_mask_area_ratio) or mask_fraction > float(max_mask_area_ratio):
        return image, masks, None

    bbox = _bbox_from_bool_mask_or_none(mask_array)
    if bbox is None:
        return image, masks, None
    left, top, right, bottom = bbox
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    margin_x = int(round(box_width * max(0.0, float(margin_ratio))))
    margin_y = int(round(box_height * max(0.0, float(margin_ratio))))
    left = max(0, left - margin_x)
    top = max(0, top - margin_y)
    right = min(width, right + margin_x)
    bottom = min(height, bottom + margin_y)
    if right <= left or bottom <= top:
        return image, masks, None

    crop_area_ratio = float((right - left) * (bottom - top)) / max(1.0, float(width * height))
    if crop_area_ratio > float(max_crop_area_ratio):
        return image, masks, None
    if (right - left) >= width - 1 and (bottom - top) >= height - 1:
        return image, masks, None

    crop_box = (left, top, right, bottom)
    image = image.crop(crop_box)
    if masks is not None:
        masks = [mask.crop(crop_box) for mask in masks]
    return image, masks, crop_box


class HybridImageTransform:
    def __init__(
        self,
        image_size: int,
        resize_mode: str = "pad",
        train: bool = False,
        random_resized_crop_scale_min: float = 1.0,
        random_resized_crop_probability: float = 1.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
        hue: float = 0.02,
        random_erasing_probability: float = 0.2,
        random_affine_degrees: float = 8.0,
        random_affine_translate: float = 0.05,
        random_affine_scale_min: float = 0.9,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.1,
        rotate90_probability: float = 0.15,
        lighting_probability: float = 0.15,
        illumination_normalization: bool = False,
        illumination_normalization_strength: float = 0.0,
        foreground_crop_mode: str = "none",
        foreground_crop_probability: float = 0.0,
        foreground_crop_margin_ratio: float = 0.08,
        foreground_crop_min_mask_area_ratio: float = 0.03,
        foreground_crop_max_mask_area_ratio: float = 0.92,
        foreground_crop_max_crop_area_ratio: float = 0.98,
        background_suppression_mode: str = "none",
        background_suppression_probability: float = 0.0,
        background_suppression_margin: float = 0.08,
        background_suppression_blur_radius: float = 7.0,
        surface_detail_amplification_mode: str = "none",
        surface_detail_amplification_probability: float = 0.0,
        surface_detail_amplification_strength: float = 0.0,
        surface_detail_amplification_blur_radius: float = 1.25,
        surface_detail_amplification_foreground_weight: float = 0.85,
        local_exposure_probability: float = 0.0,
        local_exposure_strength: float = 0.25,
        obstacle_probability: float = 0.0,
        obstacle_max_area: float = 0.12,
        scale_photometric_with_augmentation: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.image_size = int(image_size)
        self.resize_mode = str(resize_mode)
        self.train = bool(train)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.fill = _imagenet_fill(self.mean)
        self.random_resized_crop_scale_min = float(min(max(random_resized_crop_scale_min, 0.05), 1.0))
        self.random_resized_crop_probability = max(
            0.0,
            min(1.0, float(random_resized_crop_probability)),
        )
        self.color_jitter_brightness = max(0.0, float(brightness))
        self.color_jitter_contrast = max(0.0, float(contrast))
        self.color_jitter_saturation = max(0.0, float(saturation))
        self.color_jitter_hue = max(0.0, float(hue))
        self.lighting_probability = max(0.0, min(1.0, float(lighting_probability)))
        self.random_erasing_probability = float(random_erasing_probability)
        self.random_affine_degrees = float(random_affine_degrees)
        self.random_affine_translate = float(random_affine_translate)
        self.random_affine_scale_min = float(random_affine_scale_min)
        self.horizontal_flip_probability = max(0.0, min(1.0, float(horizontal_flip_probability)))
        self.vertical_flip_probability = max(0.0, min(1.0, float(vertical_flip_probability)))
        self.rotate90_probability = max(0.0, min(1.0, float(rotate90_probability)))
        self.scale_photometric_with_augmentation = bool(scale_photometric_with_augmentation)
        self.illumination_normalization = bool(illumination_normalization)
        self.illumination_normalization_strength = max(0.0, min(1.0, float(illumination_normalization_strength)))
        self.foreground_crop_mode = str(foreground_crop_mode or "none").strip().lower().replace("-", "_")
        self.foreground_crop_probability = max(0.0, min(1.0, float(foreground_crop_probability)))
        self.foreground_crop_margin_ratio = max(0.0, float(foreground_crop_margin_ratio))
        self.foreground_crop_min_mask_area_ratio = max(0.0, min(1.0, float(foreground_crop_min_mask_area_ratio)))
        self.foreground_crop_max_mask_area_ratio = max(
            self.foreground_crop_min_mask_area_ratio,
            min(1.0, float(foreground_crop_max_mask_area_ratio)),
        )
        self.foreground_crop_max_crop_area_ratio = max(0.01, min(1.0, float(foreground_crop_max_crop_area_ratio)))
        self.background_suppression_mode = str(background_suppression_mode or "none").strip().lower()
        self.background_suppression_probability = max(0.0, min(1.0, float(background_suppression_probability)))
        self.background_suppression_margin = max(0.0, float(background_suppression_margin))
        self.background_suppression_blur_radius = max(0.1, float(background_suppression_blur_radius))
        self.surface_detail_amplification_mode = (
            str(surface_detail_amplification_mode or "none").strip().lower().replace("-", "_")
        )
        self.surface_detail_amplification_probability = max(
            0.0,
            min(1.0, float(surface_detail_amplification_probability)),
        )
        self.surface_detail_amplification_strength = max(0.0, float(surface_detail_amplification_strength))
        self.surface_detail_amplification_blur_radius = max(
            0.1,
            float(surface_detail_amplification_blur_radius),
        )
        self.surface_detail_amplification_foreground_weight = max(
            0.0,
            min(1.0, float(surface_detail_amplification_foreground_weight)),
        )
        self.local_exposure_probability = max(0.0, min(1.0, float(local_exposure_probability)))
        self.local_exposure_strength = max(0.0, min(1.0, float(local_exposure_strength)))
        self.obstacle_probability = max(0.0, min(1.0, float(obstacle_probability)))
        self.obstacle_max_area = max(0.0, min(0.5, float(obstacle_max_area)))
        self.resize_pad = ResizePadToSquare(
            image_size=self.image_size,
            fill=self.fill,
            interpolation=InterpolationMode.BICUBIC,
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.lighting = transforms.RandomApply(
            [
                transforms.RandomAutocontrast(p=1.0),
                transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=1.0),
            ],
            p=self.lighting_probability,
        )
        self.random_erasing = transforms.RandomErasing(p=self.random_erasing_probability)

    def _valid_mask_from_meta(self, meta: Dict[str, object]) -> Tensor:
        if meta.get("mode") == "pad":
            left, top, right, bottom = [int(value) for value in meta.get("padding", (0, 0, 0, 0))]
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.bool)
            x1 = min(max(0, left), self.image_size)
            y1 = min(max(0, top), self.image_size)
            x2 = min(max(x1, self.image_size - max(0, right)), self.image_size)
            y2 = min(max(y1, self.image_size - max(0, bottom)), self.image_size)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
            else:
                mask[:, :] = True
            return mask
        return torch.ones((self.image_size, self.image_size), dtype=torch.bool)

    @staticmethod
    def _coerce_augmentation_scale(value: object) -> float:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 1.0
            value = value.detach().to(dtype=torch.float32).view(-1)[0].item()
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(parsed):
            return 1.0
        return max(0.1, parsed)

    @staticmethod
    def _probability_scale(augmentation_scale: float, *, boost_power: float) -> float:
        scale = max(0.1, float(augmentation_scale))
        if scale < 1.0:
            return scale
        return scale ** float(boost_power)

    def _apply_affine(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        if not self.train:
            return image, masks

        augmentation_scale = max(0.1, float(augmentation_scale))
        affine_degrees = self.random_affine_degrees * augmentation_scale
        affine_translate = min(0.2, self.random_affine_translate * augmentation_scale)
        affine_scale_min = max(0.7, 1.0 - (1.0 - self.random_affine_scale_min) * augmentation_scale)
        angle = float(
            torch.empty(1).uniform_(-affine_degrees, affine_degrees).item()
        )
        translate_x = int(
            round(torch.empty(1).uniform_(-affine_translate, affine_translate).item() * image.size[0])
        )
        translate_y = int(
            round(torch.empty(1).uniform_(-affine_translate, affine_translate).item() * image.size[1])
        )
        scale = float(
            torch.empty(1).uniform_(affine_scale_min, 1.0).item()
        )

        image = TF.affine(
            image,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BICUBIC,
            fill=list(self.fill),
        )
        if masks is not None:
            transformed_masks = []
            for mask in masks:
                transformed_masks.append(
                    TF.affine(
                        mask,
                        angle=angle,
                        translate=[translate_x, translate_y],
                        scale=scale,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.NEAREST,
                        fill=0,
                    )
                )
            masks = transformed_masks
        return image, masks

    def _apply_horizontal_flip(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        probability = min(
            1.0,
            self.horizontal_flip_probability
            * self._probability_scale(augmentation_scale, boost_power=0.5),
        )
        if not self.train or torch.rand(1).item() >= probability:
            return image, masks
        image = TF.hflip(image)
        if masks is not None:
            masks = [TF.hflip(mask) for mask in masks]
        return image, masks

    def _apply_vertical_flip(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        probability = min(
            1.0,
            self.vertical_flip_probability
            * self._probability_scale(augmentation_scale, boost_power=1.0),
        )
        if not self.train or torch.rand(1).item() >= probability:
            return image, masks
        image = TF.vflip(image)
        if masks is not None:
            masks = [TF.vflip(mask) for mask in masks]
        return image, masks

    def _apply_rotate90(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        probability = min(
            1.0,
            self.rotate90_probability
            * self._probability_scale(augmentation_scale, boost_power=1.0),
        )
        if not self.train or torch.rand(1).item() >= probability:
            return image, masks
        operations = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )
        operation = operations[int(torch.randint(len(operations), (1,)).item())]
        image = image.transpose(operation)
        if masks is not None:
            masks = [mask.transpose(operation) for mask in masks]
        return image, masks

    def _apply_random_scale_crop(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        if (
            not self.train
            or self.random_resized_crop_scale_min >= 0.999
            or self.random_resized_crop_probability <= 0.0
            or torch.rand(1).item()
            >= min(
                1.0,
                self.random_resized_crop_probability
                * min(1.0, max(0.1, float(augmentation_scale))),
            )
        ):
            return image, masks
        width, height = image.size
        if width <= 1 or height <= 1:
            return image, masks

        min_scale = self.random_resized_crop_scale_min
        augmentation_scale = max(0.1, float(augmentation_scale))
        # Minority-class samples already receive stronger affine/color jitter; keep
        # crop jitter moderate so small/edge mangos are not dropped too often.
        min_scale = max(0.55, 1.0 - (1.0 - min_scale) * min(1.5, augmentation_scale))
        scale = float(torch.empty(1).uniform_(min_scale, 1.0).item())
        crop_width = max(1, min(width, int(round(width * scale))))
        crop_height = max(1, min(height, int(round(height * scale))))
        if crop_width >= width and crop_height >= height:
            return image, masks

        for _ in range(6):
            left = int(torch.randint(0, width - crop_width + 1, (1,)).item())
            top = int(torch.randint(0, height - crop_height + 1, (1,)).item())
            crop_box = (left, top, left + crop_width, top + crop_height)
            cropped_masks = [mask.crop(crop_box) for mask in masks] if masks is not None else None
            if cropped_masks is None or any(np.asarray(mask, dtype=np.uint8).max() > 0 for mask in cropped_masks):
                return image.crop(crop_box), cropped_masks
        return image, masks

    def _apply_input_preprocess(self, image: Image.Image) -> Image.Image:
        if self.illumination_normalization and self.illumination_normalization_strength > 0.0:
            image = _normalize_illumination_image(
                image,
                strength=self.illumination_normalization_strength,
            )
        if self.background_suppression_mode not in {"", "none", "off", "false"}:
            should_apply = (not self.train) or (
                torch.rand(1).item() < self.background_suppression_probability
            )
            if should_apply:
                image = _suppress_background_image(
                    image,
                    mode=self.background_suppression_mode,
                    margin=self.background_suppression_margin,
                    blur_radius=self.background_suppression_blur_radius,
                )
        if self.surface_detail_amplification_mode not in {"", "none", "off", "false"}:
            should_apply = (not self.train) or (
                torch.rand(1).item() < self.surface_detail_amplification_probability
            )
            if should_apply:
                image = _amplify_surface_detail_image(
                    image,
                    mode=self.surface_detail_amplification_mode,
                    strength=self.surface_detail_amplification_strength,
                    blur_radius=self.surface_detail_amplification_blur_radius,
                    foreground_margin=self.background_suppression_margin,
                    foreground_weight=self.surface_detail_amplification_foreground_weight,
                )
        return image

    def _apply_foreground_crop(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]], Optional[Tuple[int, int, int, int]]]:
        if self.foreground_crop_mode in {"", "none", "off", "false"}:
            return image, masks, None
        probability = 1.0
        if self.train:
            probability = min(
                1.0,
                self.foreground_crop_probability
                * self._probability_scale(augmentation_scale, boost_power=0.5),
            )
        if probability <= 0.0 or (self.train and torch.rand(1).item() >= probability):
            return image, masks, None
        return _crop_to_foreground_bbox_pair(
            image,
            masks,
            mode=self.foreground_crop_mode,
            margin_ratio=self.foreground_crop_margin_ratio,
            min_mask_area_ratio=self.foreground_crop_min_mask_area_ratio,
            max_mask_area_ratio=self.foreground_crop_max_mask_area_ratio,
            max_crop_area_ratio=self.foreground_crop_max_crop_area_ratio,
        )

    def _apply_local_exposure_aug(self, image: Image.Image, augmentation_scale: float = 1.0) -> Image.Image:
        if not self.train or self.local_exposure_probability <= 0.0:
            return image
        probability = min(
            1.0,
            self.local_exposure_probability
            * self._probability_scale(augmentation_scale, boost_power=0.5),
        )
        if torch.rand(1).item() >= probability:
            return image
        width, height = image.size
        if width <= 1 or height <= 1:
            return image
        strength = self.local_exposure_strength * float(torch.empty(1).uniform_(0.45, 1.0).item())
        factor = 1.0 - strength if torch.rand(1).item() < 0.55 else 1.0 + strength
        center_x = float(torch.empty(1).uniform_(0.15, 0.85).item()) * width
        center_y = float(torch.empty(1).uniform_(0.15, 0.85).item()) * height
        radius_x = float(torch.empty(1).uniform_(0.20, 0.55).item()) * width
        radius_y = float(torch.empty(1).uniform_(0.18, 0.50).item()) * height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        ellipse = ((xx - center_x) / max(radius_x, 1.0)) ** 2 + ((yy - center_y) / max(radius_y, 1.0)) ** 2
        mask = np.clip(1.0 - ellipse, 0.0, 1.0) ** 1.5
        mask_image = Image.fromarray((mask * 255.0).round().astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(2.0, min(width, height) * 0.025))
        )
        base = image.convert("RGB")
        adjusted = Image.fromarray(
            (np.asarray(base, dtype=np.float32) * factor).clip(0.0, 255.0).round().astype(np.uint8),
            mode="RGB",
        )
        return Image.composite(adjusted, base, mask_image)

    def _apply_obstacle_aug(self, image: Image.Image, augmentation_scale: float = 1.0) -> Image.Image:
        if not self.train or self.obstacle_probability <= 0.0 or self.obstacle_max_area <= 0.0:
            return image
        probability = min(
            1.0,
            self.obstacle_probability
            * self._probability_scale(augmentation_scale, boost_power=0.5),
        )
        if torch.rand(1).item() >= probability:
            return image
        width, height = image.size
        if width <= 1 or height <= 1:
            return image
        max_area = min(0.5, max(0.01, self.obstacle_max_area))
        area = float(torch.empty(1).uniform_(0.015, max_area).item()) * width * height
        aspect = float(torch.empty(1).uniform_(0.35, 2.4).item())
        box_w = min(width, max(4, int(round(math.sqrt(area * aspect)))))
        box_h = min(height, max(4, int(round(math.sqrt(area / max(aspect, 1e-3))))))
        edge_anchor = torch.rand(1).item() < 0.60
        if edge_anchor:
            side = int(torch.randint(4, (1,)).item())
            if side == 0:
                left = 0
                top = int(torch.randint(0, max(1, height - box_h + 1), (1,)).item())
            elif side == 1:
                left = max(0, width - box_w)
                top = int(torch.randint(0, max(1, height - box_h + 1), (1,)).item())
            elif side == 2:
                left = int(torch.randint(0, max(1, width - box_w + 1), (1,)).item())
                top = 0
            else:
                left = int(torch.randint(0, max(1, width - box_w + 1), (1,)).item())
                top = max(0, height - box_h)
        else:
            left = int(torch.randint(0, max(1, width - box_w + 1), (1,)).item())
            top = int(torch.randint(0, max(1, height - box_h + 1), (1,)).item())
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, mode="RGBA")
        color_choices = (
            (36, 28, 24, 130),
            (92, 72, 52, 105),
            (12, 12, 12, 115),
            (205, 190, 160, 80),
        )
        fill = color_choices[int(torch.randint(len(color_choices), (1,)).item())]
        box = (left, top, left + box_w, top + box_h)
        if torch.rand(1).item() < 0.5:
            draw.ellipse(box, fill=fill)
        else:
            draw.rounded_rectangle(box, radius=max(1, min(box_w, box_h) // 6), fill=fill)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.5, min(width, height) * 0.006)))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _apply_resize(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
    ) -> Tuple[Image.Image, Optional[List[Image.Image]], Dict[str, object]]:
        meta: Dict[str, object] = {
            "mode": self.resize_mode,
            "orig_size": tuple(int(value) for value in image.size),
            "output_size": self.image_size,
        }
        if self.resize_mode == "pad":
            image, scale, padding = self.resize_pad.apply_with_meta(image)
            if masks is not None:
                resized_masks = []
                for mask in masks:
                    resized_mask, _, _ = self.resize_pad.apply_with_meta(
                        mask,
                        fill=(0,),
                        interpolation=InterpolationMode.NEAREST,
                        force_rgb=False,
                    )
                    resized_masks.append(resized_mask)
                masks = resized_masks
            meta["scale"] = float(scale)
            meta["padding"] = tuple(int(value) for value in padding)
            meta["crop_box"] = None
            return image, masks, meta

        if self.resize_mode == "stretch":
            width, height = image.size
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if masks is not None:
                masks = [
                    mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                    for mask in masks
                ]
            meta["scale"] = (
                float(self.image_size / max(1, width)),
                float(self.image_size / max(1, height)),
            )
            meta["padding"] = (0, 0, 0, 0)
            meta["crop_box"] = None
            return image, masks, meta

        image, masks, crop_box = _center_crop_square_pair(image, masks)
        crop_side = max(1, int(crop_box[2] - crop_box[0]))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if masks is not None:
            masks = [
                mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                for mask in masks
            ]
        meta["scale"] = float(self.image_size / crop_side)
        meta["padding"] = (0, 0, 0, 0)
        meta["crop_box"] = tuple(int(value) for value in crop_box)
        return image, masks, meta

    def _apply_color(self, image: Image.Image, augmentation_scale: float = 1.0) -> Image.Image:
        if not self.train:
            return image
        augmentation_scale = max(0.1, float(augmentation_scale))
        photometric_scale = augmentation_scale if self.scale_photometric_with_augmentation else 1.0
        image = transforms.ColorJitter(
            brightness=self.color_jitter_brightness * photometric_scale,
            contrast=self.color_jitter_contrast * photometric_scale,
            saturation=self.color_jitter_saturation * photometric_scale,
            hue=min(0.5, self.color_jitter_hue * photometric_scale),
        )(image)
        if torch.rand(1).item() < min(1.0, self.lighting_probability * photometric_scale):
            image = transforms.RandomAutocontrast(p=1.0)(image)
            image = transforms.RandomAdjustSharpness(
                sharpness_factor=1.0 + 0.25 * photometric_scale,
                p=1.0,
            )(image)
        return image

    def _to_tensor(self, image: Image.Image, augmentation_scale: float = 1.0) -> Tensor:
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        if self.train and self.random_erasing_probability > 0.0:
            erasing_scale = (
                max(0.1, float(augmentation_scale))
                if self.scale_photometric_with_augmentation
                else 1.0
            )
            probability = min(1.0, self.random_erasing_probability * erasing_scale)
            tensor = transforms.RandomErasing(p=probability)(tensor)
        return tensor

    def __call__(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target: Optional[Dict[str, Tensor]] = None,
        return_meta: bool = False,
    ):
        image = image.convert("RGB")
        labels: Optional[Tensor] = None
        masks: Optional[List[Image.Image]] = None
        if target is not None:
            labels = target["labels"].detach().to(dtype=torch.long).cpu()
            boxes = target["boxes"].detach().to(dtype=torch.float32).cpu()
            masks = _boxes_to_masks(image.size, boxes.tolist())
            augmentation_scale = self._coerce_augmentation_scale(target.get("augmentation_scale", 1.0))
        elif bbox is not None:
            masks = [_bbox_to_mask(image.size, bbox)]
            augmentation_scale = 1.0
        else:
            augmentation_scale = 1.0

        image, masks = self._apply_affine(image, masks, augmentation_scale=augmentation_scale)
        image, masks = self._apply_horizontal_flip(image, masks, augmentation_scale=augmentation_scale)
        image, masks = self._apply_vertical_flip(image, masks, augmentation_scale=augmentation_scale)
        image, masks = self._apply_rotate90(image, masks, augmentation_scale=augmentation_scale)
        image, masks = self._apply_random_scale_crop(image, masks, augmentation_scale=augmentation_scale)
        image, masks, foreground_crop_box = self._apply_foreground_crop(
            image,
            masks,
            augmentation_scale=augmentation_scale,
        )
        image, masks, meta = self._apply_resize(image, masks)
        meta["augmentation_scale"] = float(augmentation_scale)
        meta["foreground_crop_box"] = (
            tuple(int(value) for value in foreground_crop_box)
            if foreground_crop_box is not None
            else None
        )
        image = self._apply_input_preprocess(image)
        image = self._apply_local_exposure_aug(image, augmentation_scale=augmentation_scale)
        image = self._apply_obstacle_aug(image, augmentation_scale=augmentation_scale)
        image = self._apply_color(image, augmentation_scale=augmentation_scale)
        tensor = self._to_tensor(image, augmentation_scale=augmentation_scale)

        if target is None and bbox is None:
            if return_meta:
                return tensor, meta
            return tensor

        if target is not None:
            if labels is None or masks is None:
                raise ValueError("Transform detection thieu labels hoac masks.")
            transformed_target = _target_from_masks(masks, labels)
            if "augmentation_scale" in target:
                transformed_target["augmentation_scale"] = target["augmentation_scale"]
            transformed_target["image_mask"] = self._valid_mask_from_meta(meta)
            if return_meta:
                return tensor, transformed_target, meta
            return tensor, transformed_target

        if not masks:
            bbox_tensor = torch.tensor((0.5, 0.5, 1.0, 1.0), dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor(_bbox_from_mask(masks[0]), dtype=torch.float32)
        if return_meta:
            return tensor, bbox_tensor, meta
        return tensor, bbox_tensor


class PseudoVideoAugmenter:
    def __init__(
        self,
        frame_transform: Callable[[Image.Image], Tensor],
        temporal_frames: int = 3,
        max_translate_ratio: float = 0.04,
        max_scale_delta: float = 0.05,
        deterministic: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: float = 0.0,
    ) -> None:
        self.frame_transform = frame_transform
        self.temporal_frames = max(1, int(temporal_frames))
        self.max_translate_ratio = max(0.0, float(max_translate_ratio))
        self.max_scale_delta = max(0.0, float(max_scale_delta))
        self.deterministic = bool(deterministic)
        self.interpolation = interpolation
        self.fill = float(fill)

    def _sample_motion(self) -> Tuple[float, float, float]:
        if self.deterministic:
            return (
                0.65 * self.max_translate_ratio,
                -0.45 * self.max_translate_ratio,
                0.5 * self.max_scale_delta,
            )
        return (
            float(torch.empty(1).uniform_(-self.max_translate_ratio, self.max_translate_ratio).item()),
            float(torch.empty(1).uniform_(-self.max_translate_ratio, self.max_translate_ratio).item()),
            float(torch.empty(1).uniform_(-self.max_scale_delta, self.max_scale_delta).item()),
        )

    def _frame_positions(self) -> List[float]:
        if self.temporal_frames <= 1:
            return [0.0]
        return torch.linspace(-1.0, 1.0, steps=self.temporal_frames).tolist()

    def _affine_tensor(
        self,
        tensor: Tensor,
        translate_x_ratio: float,
        translate_y_ratio: float,
        scale_delta: float,
    ) -> Tensor:
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        translate = [
            int(round(translate_x_ratio * width)),
            int(round(translate_y_ratio * height)),
        ]
        scale = max(0.85, 1.0 + float(scale_delta))
        fill = [self.fill for _ in range(int(tensor.shape[-3]))]
        return TF.affine(
            tensor,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=fill,
        )

    def __call__(self, image: Image.Image) -> Tensor:
        anchor = self.frame_transform(image)
        if not isinstance(anchor, torch.Tensor) or anchor.ndim != 3:
            raise TypeError("PseudoVideoAugmenter yeu cau frame_transform tra ve Tensor (C, H, W).")
        if self.temporal_frames == 1:
            return anchor.unsqueeze(0)

        motion_x, motion_y, motion_scale = self._sample_motion()
        frames = []
        for position in self._frame_positions():
            frames.append(
                self._affine_tensor(
                    anchor,
                    translate_x_ratio=motion_x * position,
                    translate_y_ratio=motion_y * position,
                    scale_delta=motion_scale * position,
                )
            )
        return torch.stack(frames, dim=0)


def build_train_transform(
    image_size: int,
    resize_mode: str = "pad",
    scale_min: float = 0.8,
    scale_crop_probability: float = 1.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.15,
    hue: float = 0.02,
    random_erasing_probability: float = 0.2,
    random_affine_degrees: float = 8.0,
    random_affine_translate: float = 0.05,
    random_affine_scale_min: float = 0.9,
    horizontal_flip_probability: float = 0.5,
    vertical_flip_probability: float = 0.1,
    rotate90_probability: float = 0.15,
    lighting_probability: float = 0.15,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 10,
    illumination_normalization: bool = False,
    illumination_normalization_strength: float = 0.0,
    foreground_crop_mode: str = "none",
    foreground_crop_probability: float = 0.0,
    foreground_crop_margin_ratio: float = 0.08,
    foreground_crop_min_mask_area_ratio: float = 0.03,
    foreground_crop_max_mask_area_ratio: float = 0.92,
    foreground_crop_max_crop_area_ratio: float = 0.98,
    background_suppression_mode: str = "none",
    background_suppression_probability: float = 0.0,
    background_suppression_margin: float = 0.08,
    background_suppression_blur_radius: float = 7.0,
    surface_detail_amplification_mode: str = "none",
    surface_detail_amplification_probability: float = 0.0,
    surface_detail_amplification_strength: float = 0.0,
    surface_detail_amplification_blur_radius: float = 1.25,
    surface_detail_amplification_foreground_weight: float = 0.85,
    local_exposure_probability: float = 0.0,
    local_exposure_strength: float = 0.25,
    obstacle_probability: float = 0.0,
    obstacle_max_area: float = 0.12,
    scale_photometric_with_augmentation: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> HybridImageTransform:
    return HybridImageTransform(
        image_size=image_size,
        resize_mode=resize_mode,
        train=True,
        random_resized_crop_scale_min=scale_min,
        random_resized_crop_probability=scale_crop_probability,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        random_erasing_probability=random_erasing_probability,
        random_affine_degrees=random_affine_degrees,
        random_affine_translate=random_affine_translate,
        random_affine_scale_min=random_affine_scale_min,
        horizontal_flip_probability=horizontal_flip_probability,
        vertical_flip_probability=vertical_flip_probability,
        rotate90_probability=rotate90_probability,
        lighting_probability=lighting_probability,
        illumination_normalization=illumination_normalization,
        illumination_normalization_strength=illumination_normalization_strength,
        foreground_crop_mode=foreground_crop_mode,
        foreground_crop_probability=foreground_crop_probability,
        foreground_crop_margin_ratio=foreground_crop_margin_ratio,
        foreground_crop_min_mask_area_ratio=foreground_crop_min_mask_area_ratio,
        foreground_crop_max_mask_area_ratio=foreground_crop_max_mask_area_ratio,
        foreground_crop_max_crop_area_ratio=foreground_crop_max_crop_area_ratio,
        background_suppression_mode=background_suppression_mode,
        background_suppression_probability=background_suppression_probability,
        background_suppression_margin=background_suppression_margin,
        background_suppression_blur_radius=background_suppression_blur_radius,
        surface_detail_amplification_mode=surface_detail_amplification_mode,
        surface_detail_amplification_probability=surface_detail_amplification_probability,
        surface_detail_amplification_strength=surface_detail_amplification_strength,
        surface_detail_amplification_blur_radius=surface_detail_amplification_blur_radius,
        surface_detail_amplification_foreground_weight=surface_detail_amplification_foreground_weight,
        local_exposure_probability=local_exposure_probability,
        local_exposure_strength=local_exposure_strength,
        obstacle_probability=obstacle_probability,
        obstacle_max_area=obstacle_max_area,
        scale_photometric_with_augmentation=scale_photometric_with_augmentation,
        mean=mean,
        std=std,
    )


def build_eval_transform(
    image_size: int,
    resize_mode: str = "pad",
    illumination_normalization: bool = False,
    illumination_normalization_strength: float = 0.0,
    foreground_crop_mode: str = "none",
    foreground_crop_margin_ratio: float = 0.08,
    foreground_crop_min_mask_area_ratio: float = 0.03,
    foreground_crop_max_mask_area_ratio: float = 0.92,
    foreground_crop_max_crop_area_ratio: float = 0.98,
    background_suppression_mode: str = "none",
    background_suppression_margin: float = 0.08,
    background_suppression_blur_radius: float = 7.0,
    surface_detail_amplification_mode: str = "none",
    surface_detail_amplification_strength: float = 0.0,
    surface_detail_amplification_blur_radius: float = 1.25,
    surface_detail_amplification_foreground_weight: float = 0.85,
    eval_surface_detail_amplification: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> HybridImageTransform:
    eval_surface_mode = (
        str(surface_detail_amplification_mode or "none")
        if bool(eval_surface_detail_amplification)
        else "none"
    )
    return HybridImageTransform(
        image_size=image_size,
        resize_mode=resize_mode,
        train=False,
        illumination_normalization=illumination_normalization,
        illumination_normalization_strength=illumination_normalization_strength,
        foreground_crop_mode=foreground_crop_mode,
        foreground_crop_probability=1.0,
        foreground_crop_margin_ratio=foreground_crop_margin_ratio,
        foreground_crop_min_mask_area_ratio=foreground_crop_min_mask_area_ratio,
        foreground_crop_max_mask_area_ratio=foreground_crop_max_mask_area_ratio,
        foreground_crop_max_crop_area_ratio=foreground_crop_max_crop_area_ratio,
        background_suppression_mode=background_suppression_mode,
        background_suppression_probability=1.0,
        background_suppression_margin=background_suppression_margin,
        background_suppression_blur_radius=background_suppression_blur_radius,
        surface_detail_amplification_mode=eval_surface_mode,
        surface_detail_amplification_probability=1.0 if bool(eval_surface_detail_amplification) else 0.0,
        surface_detail_amplification_strength=surface_detail_amplification_strength,
        surface_detail_amplification_blur_radius=surface_detail_amplification_blur_radius,
        surface_detail_amplification_foreground_weight=surface_detail_amplification_foreground_weight,
        mean=mean,
        std=std,
    )
