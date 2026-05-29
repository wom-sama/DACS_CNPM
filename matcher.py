from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _unpack_model_output(model_output) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        objectness_logits = model_output.get("objectness_logits")
    elif isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
        logits, boxes = model_output[0], model_output[1]
        objectness_logits = model_output[2] if len(model_output) >= 3 else None
    else:
        raise TypeError("Model output cho matcher phai chua logits va boxes.")
    if logits is None or boxes is None:
        raise ValueError("Model output cho matcher thieu logits hoac boxes.")
    return logits, boxes, objectness_logits


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    x_center, y_center, box_width, box_height = boxes.unbind(dim=-1)
    x1 = x_center - box_width / 2.0
    y1 = y_center - box_height / 2.0
    x2 = x_center + box_width / 2.0
    y2 = y_center + box_height / 2.0
    return torch.stack((x1, y1, x2, y2), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    return widths * heights


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1 = boxes1.clamp(0.0, 1.0)
    boxes2 = boxes2.clamp(0.0, 1.0)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    union = (area1 + area2 - intersection).clamp(min=1e-6)
    iou = intersection / union

    convex_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    convex_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    convex_wh = (convex_bottom_right - convex_top_left).clamp(min=0.0)
    convex_area = (convex_wh[..., 0] * convex_wh[..., 1]).clamp(min=1e-6)
    return iou - (convex_area - union) / convex_area


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_objectness: float = 0.0,
    ) -> None:
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.cost_objectness = float(cost_objectness)
        self._invalid_cost_warning_count = 0
        if (
            self.cost_class == 0.0
            and self.cost_bbox == 0.0
            and self.cost_giou == 0.0
            and self.cost_objectness == 0.0
        ):
            raise ValueError("HungarianMatcher can it nhat 1 cost khac 0.")

    @torch.no_grad()
    def forward(
        self,
        model_output,
        targets: Sequence[Dict[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        logits, boxes, objectness_logits = _unpack_model_output(model_output)
        safe_logits = torch.nan_to_num(
            logits.float(),
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )
        safe_boxes = torch.nan_to_num(
            boxes.float(),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        probabilities = safe_logits.softmax(dim=-1)
        objectness_probabilities = (
            torch.sigmoid(
                torch.nan_to_num(
                    objectness_logits.float(),
                    nan=0.0,
                    posinf=50.0,
                    neginf=-50.0,
                )
            ).to(dtype=probabilities.dtype)
            if objectness_logits is not None and self.cost_objectness != 0.0
            else None
        )
        indices: List[Tuple[Tensor, Tensor]] = []

        for batch_index, target in enumerate(targets):
            target_labels = target["labels"]
            target_boxes = torch.nan_to_num(
                target["boxes"].float(),
                nan=0.5,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                indices.append((empty, empty))
                continue

            batch_probabilities = probabilities[batch_index]
            batch_boxes = safe_boxes[batch_index]

            cost_class = -batch_probabilities[:, target_labels.to(torch.long)]
            cost_bbox = torch.cdist(batch_boxes, target_boxes.to(batch_boxes.dtype), p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(batch_boxes),
                box_cxcywh_to_xyxy(target_boxes.to(batch_boxes.dtype)),
            )
            if objectness_probabilities is not None:
                cost_objectness = -objectness_probabilities[batch_index].unsqueeze(1).expand_as(cost_class)
            else:
                cost_objectness = torch.zeros_like(cost_class)

            total_cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
                + self.cost_objectness * cost_objectness
            )
            if not torch.isfinite(total_cost).all():
                if self._invalid_cost_warning_count < 10:
                    invalid_count = int((~torch.isfinite(total_cost)).sum().item())
                    logger.warning(
                        "HungarianMatcher sanitized non-finite cost entries: "
                        "batch_index=%s invalid_entries=%s cost_shape=%s",
                        batch_index,
                        invalid_count,
                        tuple(total_cost.shape),
                    )
                    self._invalid_cost_warning_count += 1
                total_cost = torch.nan_to_num(
                    total_cost,
                    nan=1e6,
                    posinf=1e6,
                    neginf=-1e6,
                )
            matched_queries, matched_targets = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(matched_queries, dtype=torch.int64, device=logits.device),
                    torch.as_tensor(matched_targets, dtype=torch.int64, device=logits.device),
                )
            )
        return indices
