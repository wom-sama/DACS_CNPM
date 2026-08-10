from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class FriendlyAttackResult:
    adversarial_images: Tensor
    crossed: Tensor
    crossing_step: Tensor
    delta_rgb: Tensor
    attack_mask: Tensor
    clean_margin: Tensor
    adversarial_margin: Tensor


def _normalization_tensors(
    mean: Sequence[float],
    std: Sequence[float],
    reference: Tensor,
) -> Tuple[Tensor, Tensor]:
    mean_tensor = torch.as_tensor(mean, device=reference.device, dtype=reference.dtype).view(
        1, 3, 1, 1
    )
    std_tensor = torch.as_tensor(std, device=reference.device, dtype=reference.dtype).view(
        1, 3, 1, 1
    )
    if bool((std_tensor <= 0).any().item()):
        raise ValueError("input standard deviations must be positive")
    return mean_tensor, std_tensor


def normalized_to_rgb(
    images: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    mean_tensor, std_tensor = _normalization_tensors(mean, std, images)
    return (images * std_tensor + mean_tensor).clamp(0.0, 1.0)


def rgb_to_normalized(
    images: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    mean_tensor, std_tensor = _normalization_tensors(mean, std, images)
    return (images - mean_tensor) / std_tensor


def build_eroded_bbox_mask(
    crop_bboxes: Tensor,
    image_valid_mask: Tensor,
    *,
    height: int,
    width: int,
    erode_ratio: float,
) -> Tensor:
    if crop_bboxes.ndim == 3:
        crop_bboxes = crop_bboxes[:, 0]
    if crop_bboxes.ndim != 2 or int(crop_bboxes.size(1)) < 4:
        raise ValueError("crop_bboxes must have shape [B,4] or [B,1,4]")
    if image_valid_mask.ndim == 4 and int(image_valid_mask.size(1)) == 1:
        image_valid_mask = image_valid_mask[:, 0]
    if image_valid_mask.ndim != 3:
        raise ValueError("image_valid_mask must have shape [B,H,W]")
    if int(crop_bboxes.size(0)) != int(image_valid_mask.size(0)):
        raise ValueError("bbox and valid-mask batch sizes differ")
    if tuple(image_valid_mask.shape[-2:]) != (int(height), int(width)):
        raise ValueError("valid-mask spatial shape differs from the input")
    ratio = float(erode_ratio)
    if not 0.0 <= ratio < 0.5:
        raise ValueError("erode_ratio must be in [0,0.5)")

    mask = torch.zeros(
        (int(crop_bboxes.size(0)), 1, int(height), int(width)),
        device=crop_bboxes.device,
        dtype=torch.bool,
    )
    boxes = crop_bboxes[:, :4].detach().float().cpu()
    for row_index, values in enumerate(boxes.tolist()):
        center_x, center_y, box_width, box_height = [float(value) for value in values]
        half_width = max(1e-6, box_width * (0.5 - ratio))
        half_height = max(1e-6, box_height * (0.5 - ratio))
        x1 = max(0, min(int(width) - 1, int(math.floor((center_x - half_width) * width))))
        y1 = max(0, min(int(height) - 1, int(math.floor((center_y - half_height) * height))))
        x2 = max(x1 + 1, min(int(width), int(math.ceil((center_x + half_width) * width))))
        y2 = max(y1 + 1, min(int(height), int(math.ceil((center_y + half_height) * height))))
        mask[row_index, :, y1:y2, x1:x2] = True
    return mask & image_valid_mask.to(device=mask.device, dtype=torch.bool).unsqueeze(1)


def class1_attack_margin(
    logits: Tensor,
    labels: Tensor,
    *,
    direction: str,
    focus_class: int,
    negative_classes: Sequence[int],
) -> Tensor:
    if logits.ndim != 2 or labels.ndim != 1 or int(logits.size(0)) != int(labels.numel()):
        raise ValueError("logits/labels must have shapes [B,C] and [B]")
    normalized_direction = str(direction).strip().lower()
    if normalized_direction == "protect":
        negative = torch.as_tensor(
            list(negative_classes), device=logits.device, dtype=torch.long
        )
        rival = logits.index_select(1, negative).max(dim=1).values
        return rival - logits[:, int(focus_class)]
    if normalized_direction == "suppress":
        true_logits = logits.gather(1, labels.long().view(-1, 1)).squeeze(1)
        return logits[:, int(focus_class)] - true_logits
    raise ValueError("direction must be 'protect' or 'suppress'")


def generate_friendly_adversarial_examples(
    *,
    images: Tensor,
    labels: Tensor,
    attack_mask: Tensor,
    forward_logits: Callable[[Tensor], Tensor],
    mean: Sequence[float],
    std: Sequence[float],
    direction: str,
    focus_class: int,
    negative_classes: Sequence[int],
    epsilon: float,
    step_size: float,
    steps: int,
) -> FriendlyAttackResult:
    if images.ndim != 4 or int(images.size(1)) != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if attack_mask.shape != (int(images.size(0)), 1, int(images.size(2)), int(images.size(3))):
        raise ValueError("attack_mask shape differs from images")
    if labels.ndim != 1 or int(labels.numel()) != int(images.size(0)):
        raise ValueError("labels shape differs from images")
    if float(epsilon) <= 0.0 or float(step_size) <= 0.0 or int(steps) <= 0:
        raise ValueError("attack epsilon, step size, and steps must be positive")
    if float(step_size) > float(epsilon):
        raise ValueError("attack step size cannot exceed epsilon")

    original_rgb = normalized_to_rgb(images.detach(), mean=mean, std=std)
    mask = attack_mask.to(device=images.device, dtype=original_rgb.dtype)
    current_rgb = original_rgb.clone()
    crossed = torch.zeros(int(images.size(0)), device=images.device, dtype=torch.bool)
    crossing_step = torch.zeros(int(images.size(0)), device=images.device, dtype=torch.long)

    with torch.no_grad():
        clean_logits = forward_logits(images.detach())
        clean_margin = class1_attack_margin(
            clean_logits,
            labels,
            direction=direction,
            focus_class=focus_class,
            negative_classes=negative_classes,
        )

    for step_index in range(1, int(steps) + 1):
        active = ~crossed
        if not bool(active.any().item()):
            break
        variable_rgb = current_rgb.detach().requires_grad_(True)
        logits = forward_logits(rgb_to_normalized(variable_rgb, mean=mean, std=std))
        margin = class1_attack_margin(
            logits,
            labels,
            direction=direction,
            focus_class=focus_class,
            negative_classes=negative_classes,
        )
        gradient = torch.autograd.grad(margin[active].sum(), variable_rgb, only_inputs=True)[0]
        proposal = variable_rgb.detach() + float(step_size) * gradient.sign() * mask
        delta = (proposal - original_rgb).clamp(min=-float(epsilon), max=float(epsilon))
        proposal = (original_rgb + delta * mask).clamp(0.0, 1.0)
        current_rgb = torch.where(active.view(-1, 1, 1, 1), proposal, current_rgb)
        with torch.no_grad():
            proposal_logits = forward_logits(
                rgb_to_normalized(current_rgb, mean=mean, std=std)
            )
            proposal_margin = class1_attack_margin(
                proposal_logits,
                labels,
                direction=direction,
                focus_class=focus_class,
                negative_classes=negative_classes,
            )
            newly_crossed = active & (proposal_margin >= 0.0)
            crossing_step[newly_crossed] = int(step_index)
            crossed = crossed | newly_crossed

    adversarial_images = rgb_to_normalized(current_rgb.detach(), mean=mean, std=std)
    with torch.no_grad():
        adversarial_margin = class1_attack_margin(
            forward_logits(adversarial_images),
            labels,
            direction=direction,
            focus_class=focus_class,
            negative_classes=negative_classes,
        )
    return FriendlyAttackResult(
        adversarial_images=adversarial_images.detach(),
        crossed=crossed.detach(),
        crossing_step=crossing_step.detach(),
        delta_rgb=(current_rgb.detach() - original_rgb).detach(),
        attack_mask=attack_mask.detach().to(dtype=torch.bool),
        clean_margin=clean_margin.detach(),
        adversarial_margin=adversarial_margin.detach(),
    )
