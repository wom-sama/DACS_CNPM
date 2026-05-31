from __future__ import annotations

import math
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageOps
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from config import IMAGENET_MEAN, IMAGENET_STD, DataSpec, load_data_spec

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
logger = logging.getLogger(__name__)
PIL_RESAMPLE_MAP = {
    InterpolationMode.NEAREST: Image.Resampling.NEAREST,
    InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
    InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
    InterpolationMode.BOX: Image.Resampling.BOX,
    InterpolationMode.HAMMING: Image.Resampling.HAMMING,
    InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
}


def _imagenet_fill(mean: Sequence[float]) -> Tuple[int, int, int]:
    return tuple(int(round(channel * 255.0)) for channel in mean)


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

    pasted_images = images.clone()
    pasted_targets: List[Dict[str, Tensor]] = []
    max_paste_objects = max(1, int(max_paste_objects))
    padding_ratio = max(0.0, float(padding_ratio))
    occlusion_threshold = min(max(float(occlusion_threshold), 0.0), 1.0)

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
            object_index = int(valid_object_indices[torch.randint(0, valid_object_indices.numel(), (1,)).item()].item())

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
            bbox_targets = torch.stack(
                [sample[2]["bbox"] for sample in batch],
                dim=0,
            ).to(dtype=torch.float32)
            return images, labels, {"bbox": bbox_targets}

        targets = F.one_hot(labels, num_classes=self.num_classes).to(dtype=torch.float32)

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
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.crop_margin_ratio = crop_margin_ratio
        self.crop_to_primary_object = bool(crop_to_primary_object)
        self.fallback_to_full_image = fallback_to_full_image
        self.num_classes = num_classes
        self.primary_object_strategy = str(primary_object_strategy).strip().lower()
        self.split = str(split or "unknown")
        self.class_aware_augmentation = bool(class_aware_augmentation)
        self.class_augmentation_power = max(0.0, float(class_augmentation_power))
        self.class_augmentation_max_scale = max(1.0, float(class_augmentation_max_scale))
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
            "missing_images=%s invalid_bboxes=%s invalid_classes=%s crop_primary=%s class_aug=%s scales=%s",
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
            self.class_aware_augmentation,
            self.class_augmentation_scales,
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
            primary_object = self._select_primary_object(objects)
            samples.append(
                MangoSample(
                    image_path=image_path,
                    label_path=label_path,
                    objects=objects,
                    primary_label=primary_object.label,
                )
            )
            self.audit["selected_sample_count"] += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self, num_classes: int) -> List[int]:
        counts = [0 for _ in range(num_classes)]
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

    def _augmentation_scale_for_labels(self, labels: Tensor) -> float:
        if not self.class_aware_augmentation or labels.numel() == 0:
            return 1.0
        if not self.class_augmentation_scales:
            return 1.0
        scale = 1.0
        for label in labels.detach().cpu().to(dtype=torch.long).tolist():
            if 0 <= int(label) < len(self.class_augmentation_scales):
                scale = max(scale, float(self.class_augmentation_scales[int(label)]))
        return float(scale)

    def labels(self) -> List[int]:
        return [sample.primary_label for sample in self.samples]

    def bboxes(self) -> List[Tuple[float, float, float, float]]:
        return [obj.bbox for sample in self.samples for obj in sample.objects]

    def quality_report(self) -> Dict[str, object]:
        report = dict(self.audit)
        report["crop_to_primary_object"] = bool(self.crop_to_primary_object)
        if self.num_classes is not None:
            report["class_counts"] = self.class_counts(self.num_classes)
            report["class_augmentation_scales"] = list(self.class_augmentation_scales)
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
        primary_object = self._select_primary_object(sample.objects)
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(primary_object.bbox, width=width, height=height)
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        margin_x = box_width * max(0.0, float(self.crop_margin_ratio))
        margin_y = box_height * max(0.0, float(self.crop_margin_ratio))

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
        if self.crop_to_primary_object:
            image, target = self._crop_to_primary_object(
                image=image,
                labels=labels,
                boxes=boxes,
                sample=sample,
            )
        target["augmentation_scale"] = torch.tensor(
            [self._augmentation_scale_for_labels(target["labels"])],
            dtype=torch.float32,
        )
        if self.transform is not None:
            transformed = self.transform(image, target=target)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                image_tensor, transformed_target = transformed
            else:
                raise TypeError("Transform detection phai tra ve (image_tensor, target).")
        else:
            image_tensor = image
            transformed_target = target

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
            start = batch_index % class_count
            for extra_index in range(remainder):
                class_index = self.active_classes[(start + extra_index) % class_count]
                for allocation_index, (allocated_class, count) in enumerate(allocations):
                    if allocated_class == class_index:
                        allocations[allocation_index] = (allocated_class, count + 1)
                        break
        return [(class_index, count) for class_index, count in allocations if count > 0]

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


class HybridImageTransform:
    def __init__(
        self,
        image_size: int,
        resize_mode: str = "pad",
        train: bool = False,
        random_resized_crop_scale_min: float = 1.0,
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
            return float(value)
        except (TypeError, ValueError):
            return 1.0

    def _apply_affine(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]],
        augmentation_scale: float = 1.0,
    ) -> Tuple[Image.Image, Optional[List[Image.Image]]]:
        if not self.train:
            return image, masks

        augmentation_scale = max(1.0, float(augmentation_scale))
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
        probability = min(1.0, self.horizontal_flip_probability * math.sqrt(max(1.0, float(augmentation_scale))))
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
        probability = min(1.0, self.vertical_flip_probability * max(1.0, float(augmentation_scale)))
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
        probability = min(1.0, self.rotate90_probability * max(1.0, float(augmentation_scale)))
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
        if not self.train or self.random_resized_crop_scale_min >= 0.999:
            return image, masks
        width, height = image.size
        if width <= 1 or height <= 1:
            return image, masks

        min_scale = self.random_resized_crop_scale_min
        augmentation_scale = max(1.0, float(augmentation_scale))
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
        augmentation_scale = max(1.0, float(augmentation_scale))
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
            erasing_scale = max(1.0, float(augmentation_scale)) if self.scale_photometric_with_augmentation else 1.0
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
        image, masks, meta = self._apply_resize(image, masks)
        meta["augmentation_scale"] = float(augmentation_scale)
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
    scale_photometric_with_augmentation: bool = False,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> HybridImageTransform:
    return HybridImageTransform(
        image_size=image_size,
        resize_mode=resize_mode,
        train=True,
        random_resized_crop_scale_min=scale_min,
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
        scale_photometric_with_augmentation=scale_photometric_with_augmentation,
        mean=mean,
        std=std,
    )


def build_eval_transform(
    image_size: int,
    resize_mode: str = "pad",
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> HybridImageTransform:
    return HybridImageTransform(
        image_size=image_size,
        resize_mode=resize_mode,
        train=False,
        mean=mean,
        std=std,
    )
