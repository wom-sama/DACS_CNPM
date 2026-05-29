from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from debug_and_optimization import BboxDebugger
from matcher import HungarianMatcher

logger = logging.getLogger(__name__)
LOSS_LOGIT_CLAMP = 60.0
LOSS_BOX_EPS = 1e-4


def _finite_loss_logits(tensor: Optional[Tensor]) -> Optional[Tensor]:
    if tensor is None:
        return None
    return torch.nan_to_num(
        tensor.float(),
        nan=0.0,
        posinf=LOSS_LOGIT_CLAMP,
        neginf=-LOSS_LOGIT_CLAMP,
    ).clamp(-LOSS_LOGIT_CLAMP, LOSS_LOGIT_CLAMP)


def _finite_loss_boxes(boxes: Tensor) -> Tensor:
    boxes = torch.nan_to_num(
        boxes.float(),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    if boxes.numel() == 0 or boxes.shape[-1] != 4:
        return boxes
    centers = boxes[..., :2]
    sizes = boxes[..., 2:].clamp(min=LOSS_BOX_EPS, max=1.0)
    return torch.cat((centers, sizes), dim=-1)


def _finite_loss_targets(
    targets: Sequence[Dict[str, Tensor]],
    *,
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    sanitized: List[Dict[str, Tensor]] = []
    for target in targets:
        labels = target["labels"].to(device=device, dtype=torch.long)
        boxes = _finite_loss_boxes(target["boxes"].to(device=device))
        sanitized_target: Dict[str, Tensor] = {"labels": labels, "boxes": boxes}
        image_mask = target.get("image_mask")
        if torch.is_tensor(image_mask):
            sanitized_target["image_mask"] = image_mask.to(device=device, dtype=torch.bool)
        sanitized.append(sanitized_target)
    return sanitized


def unpack_hybrid_output(model_output) -> Tuple[Tensor, Tensor]:
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
    elif isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
        logits, boxes = model_output[0], model_output[1]
    else:
        raise TypeError("Model output phai chua logits va boxes.")

    if logits is None or boxes is None:
        raise ValueError("Model output thieu logits hoac boxes.")
    return logits, boxes


def unpack_detection_output(model_output) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        objectness_logits = model_output.get("objectness_logits")
    elif isinstance(model_output, (tuple, list)) and len(model_output) >= 2:
        logits, boxes = model_output[0], model_output[1]
        objectness_logits = model_output[2] if len(model_output) >= 3 else None
    else:
        raise TypeError("Model output phai chua logits va boxes.")

    if logits is None or boxes is None:
        raise ValueError("Model output thieu logits hoac boxes.")
    return logits, boxes, objectness_logits


def unpack_count_logits(model_output) -> Optional[Tensor]:
    if isinstance(model_output, dict):
        return model_output.get("count_logits")
    if isinstance(model_output, (tuple, list)) and len(model_output) >= 4:
        return model_output[3]
    return None


def unpack_quality_logits(model_output) -> Optional[Tensor]:
    if isinstance(model_output, dict):
        return model_output.get("quality_logits")
    if isinstance(model_output, (tuple, list)) and len(model_output) >= 5:
        return model_output[4]
    return None


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
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


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    union = (area1 + area2 - intersection).clamp(min=1e-6)
    return intersection / union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou = box_iou(boxes1, boxes2)

    convex_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    convex_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    convex_wh = (convex_bottom_right - convex_top_left).clamp(min=0.0)
    convex_area = (convex_wh[..., 0] * convex_wh[..., 1]).clamp(min=1e-6)

    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]
    union = (area1 + area2 - intersection).clamp(min=1e-6)
    return iou - (convex_area - union) / convex_area


def generalized_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1 = xywh_to_xyxy(boxes1).clamp(0.0, 1.0)
    boxes2 = xywh_to_xyxy(boxes2).clamp(0.0, 1.0)
    if boxes1.shape != boxes2.shape:
        raise ValueError("generalized_iou yeu cau hai tensor co cung shape.")

    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = (area1 + area2 - intersection).clamp(min=1e-6)
    iou = intersection / union

    convex_top_left = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    convex_bottom_right = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    convex_wh = (convex_bottom_right - convex_top_left).clamp(min=0.0)
    convex_area = (convex_wh[..., 0] * convex_wh[..., 1]).clamp(min=1e-6)
    return iou - (convex_area - union) / convex_area


def box_iou_from_xywh(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1 = xywh_to_xyxy(boxes1).clamp(0.0, 1.0)
    boxes2 = xywh_to_xyxy(boxes2).clamp(0.0, 1.0)
    if boxes1.shape != boxes2.shape:
        raise ValueError("box_iou_from_xywh yeu cau hai tensor co cung shape.")

    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (bottom_right - top_left).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = (area1 + area2 - intersection).clamp(min=1e-6)
    return intersection / union


class DETRSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: Optional[HungarianMatcher] = None,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        cls_weight: float = 1.0,
        bbox_l1_weight: float = 5.0,
        bbox_giou_weight: float = 2.0,
        background_weight: float = 0.1,
        objectness_weight: float = 5.0,
        objectness_focal_alpha: float = 0.75,
        objectness_focal_gamma: float = 0.5,
        matcher_objectness_cost: float = 1.0,
        cardinality_weight: float = 0.0,
        count_weight: float = 0.0,
        quality_weight: float = 0.0,
        auxiliary_weight: float = 0.0,
        count_objectness_consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_smoothing = float(max(0.0, label_smoothing))
        self.cls_weight = float(max(0.0, cls_weight))
        self.bbox_l1_weight = float(max(0.0, bbox_l1_weight))
        self.bbox_giou_weight = float(max(0.0, bbox_giou_weight))
        self.objectness_weight = float(max(0.0, objectness_weight))
        self.objectness_focal_alpha = float(min(max(objectness_focal_alpha, 0.0), 1.0))
        self.objectness_focal_gamma = float(max(0.0, objectness_focal_gamma))
        self.matcher_objectness_cost = float(max(0.0, matcher_objectness_cost))
        self.cardinality_weight = float(max(0.0, cardinality_weight))
        self.count_weight = float(max(0.0, count_weight))
        self.quality_weight = float(max(0.0, quality_weight))
        self.auxiliary_weight = float(max(0.0, auxiliary_weight))
        self.count_objectness_consistency_weight = float(max(0.0, count_objectness_consistency_weight))
        self.matcher = matcher or HungarianMatcher(
            cost_class=1.0,
            cost_bbox=self.bbox_l1_weight,
            cost_giou=self.bbox_giou_weight,
            cost_objectness=self.matcher_objectness_cost,
        )
        self.base_cls_weight = float(self.cls_weight)
        self.base_bbox_l1_weight = float(self.bbox_l1_weight)
        self.base_bbox_giou_weight = float(self.bbox_giou_weight)
        self.base_objectness_weight = float(self.objectness_weight)
        self.base_cardinality_weight = float(self.cardinality_weight)
        self.base_count_weight = float(self.count_weight)
        self.base_quality_weight = float(self.quality_weight)
        self.base_auxiliary_weight = float(self.auxiliary_weight)
        self.base_count_objectness_consistency_weight = float(self.count_objectness_consistency_weight)
        self.base_matcher_cost_class = float(getattr(self.matcher, "cost_class", 1.0))
        self.base_matcher_cost_bbox = float(getattr(self.matcher, "cost_bbox", self.bbox_l1_weight))
        self.base_matcher_cost_giou = float(getattr(self.matcher, "cost_giou", self.bbox_giou_weight))
        self.base_matcher_cost_objectness = float(
            getattr(self.matcher, "cost_objectness", self.matcher_objectness_cost)
        )

        empty_weight = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.detach().to(torch.float32)
            if class_weights.numel() != self.num_classes:
                raise ValueError("class_weights phai co do dai bang num_classes.")
            empty_weight[:-1] = class_weights
        empty_weight[-1] = float(max(0.0, background_weight))
        self.register_buffer("empty_weight", empty_weight)

    def set_loss_weights(
        self,
        *,
        cls_weight: Optional[float] = None,
        bbox_l1_weight: Optional[float] = None,
        bbox_giou_weight: Optional[float] = None,
        objectness_weight: Optional[float] = None,
        cardinality_weight: Optional[float] = None,
        count_weight: Optional[float] = None,
        quality_weight: Optional[float] = None,
        auxiliary_weight: Optional[float] = None,
        count_objectness_consistency_weight: Optional[float] = None,
        sync_matcher_to_loss: bool = False,
    ) -> None:
        if cls_weight is not None:
            self.cls_weight = float(max(0.0, cls_weight))
        if bbox_l1_weight is not None:
            self.bbox_l1_weight = float(max(0.0, bbox_l1_weight))
        if bbox_giou_weight is not None:
            self.bbox_giou_weight = float(max(0.0, bbox_giou_weight))
        if objectness_weight is not None:
            self.objectness_weight = float(max(0.0, objectness_weight))
        if cardinality_weight is not None:
            self.cardinality_weight = float(max(0.0, cardinality_weight))
        if count_weight is not None:
            self.count_weight = float(max(0.0, count_weight))
        if quality_weight is not None:
            self.quality_weight = float(max(0.0, quality_weight))
        if auxiliary_weight is not None:
            self.auxiliary_weight = float(max(0.0, auxiliary_weight))
        if count_objectness_consistency_weight is not None:
            self.count_objectness_consistency_weight = float(max(0.0, count_objectness_consistency_weight))
        if sync_matcher_to_loss and hasattr(self.matcher, "cost_bbox") and hasattr(self.matcher, "cost_giou"):
            self.matcher.cost_class = float(self.base_matcher_cost_class)
            self.matcher.cost_bbox = float(self.bbox_l1_weight)
            self.matcher.cost_giou = float(self.bbox_giou_weight)
            if hasattr(self.matcher, "cost_objectness"):
                self.matcher.cost_objectness = float(self.base_matcher_cost_objectness)

    def reset_loss_weights(self, sync_matcher_to_loss: bool = False) -> None:
        self.set_loss_weights(
            cls_weight=self.base_cls_weight,
            bbox_l1_weight=self.base_bbox_l1_weight,
            bbox_giou_weight=self.base_bbox_giou_weight,
            objectness_weight=self.base_objectness_weight,
            cardinality_weight=self.base_cardinality_weight,
            count_weight=self.base_count_weight,
            quality_weight=self.base_quality_weight,
            auxiliary_weight=self.base_auxiliary_weight,
            count_objectness_consistency_weight=self.base_count_objectness_consistency_weight,
            sync_matcher_to_loss=sync_matcher_to_loss,
        )

    def get_loss_weights(self) -> Dict[str, float]:
        return {
            "cls_weight": float(self.cls_weight),
            "objectness_weight": float(self.objectness_weight),
            "objectness_focal_alpha": float(self.objectness_focal_alpha),
            "objectness_focal_gamma": float(self.objectness_focal_gamma),
            "matcher_objectness_cost": float(self.matcher_objectness_cost),
            "bbox_l1_weight": float(self.bbox_l1_weight),
            "bbox_giou_weight": float(self.bbox_giou_weight),
            "cardinality_weight": float(self.cardinality_weight),
            "count_weight": float(self.count_weight),
            "quality_weight": float(self.quality_weight),
            "auxiliary_weight": float(self.auxiliary_weight),
            "count_objectness_consistency_weight": float(self.count_objectness_consistency_weight),
        }

    def _target_classes_from_indices(
        self,
        logits: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        batch_size, num_queries = logits.shape[:2]
        target_classes = torch.full(
            (batch_size, num_queries),
            fill_value=self.num_classes,
            dtype=torch.long,
            device=logits.device,
        )
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            target_classes[batch_index, query_indices] = targets[batch_index]["labels"][target_indices].to(
                device=logits.device,
                dtype=torch.long,
            )
        return target_classes

    def classification_loss(self, logits: Tensor, target_classes: Tensor) -> Tensor:
        weights = self.empty_weight.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

    def matched_classification_loss(
        self,
        logits: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        matched_logits: List[Tensor] = []
        matched_labels: List[Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            matched_logits.append(logits[batch_index, query_indices])
            matched_labels.append(
                targets[batch_index]["labels"][target_indices].to(
                    device=logits.device,
                    dtype=torch.long,
                )
            )
        if not matched_logits:
            return logits.sum() * 0.0
        class_logits = torch.cat(matched_logits, dim=0)
        class_labels = torch.cat(matched_labels, dim=0)
        weights = self.empty_weight[:-1].to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            class_logits,
            class_labels,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

    def _objectness_targets_from_indices(
        self,
        objectness_logits: Tensor,
        indices: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        target_objectness = torch.zeros_like(objectness_logits, dtype=objectness_logits.dtype)
        for batch_index, (query_indices, _) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            target_objectness[batch_index, query_indices] = 1.0
        return target_objectness

    def objectness_loss(
        self,
        objectness_logits: Tensor,
        target_objectness: Tensor,
        num_boxes: int,
    ) -> Tensor:
        if objectness_logits.numel() == 0:
            return objectness_logits.sum() * 0.0
        target_objectness = target_objectness.to(device=objectness_logits.device, dtype=objectness_logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            objectness_logits,
            target_objectness,
            reduction="none",
        )
        probabilities = torch.sigmoid(objectness_logits)
        p_t = probabilities * target_objectness + (1.0 - probabilities) * (1.0 - target_objectness)
        focal_factor = (1.0 - p_t).clamp(min=0.0).pow(self.objectness_focal_gamma)
        alpha_t = (
            self.objectness_focal_alpha * target_objectness
            + (1.0 - self.objectness_focal_alpha) * (1.0 - target_objectness)
        )
        weighted_loss = alpha_t * focal_factor * bce_loss
        positive_mask = target_objectness >= 0.5
        negative_mask = ~positive_mask
        zero = objectness_logits.sum() * 0.0
        positive_loss = (
            weighted_loss[positive_mask].sum() / positive_mask.sum().to(dtype=weighted_loss.dtype).clamp(min=1.0)
            if positive_mask.any()
            else zero
        )
        negative_loss = (
            weighted_loss[negative_mask].sum() / negative_mask.sum().to(dtype=weighted_loss.dtype).clamp(min=1.0)
            if negative_mask.any()
            else zero
        )
        return positive_loss + negative_loss

    def bbox_loss(
        self,
        pred_boxes: Tensor,
        target_boxes: Tensor,
        num_boxes: int,
    ) -> Tuple[Tensor, Tensor]:
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        target_boxes = target_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
        pred_boxes = pred_boxes.clamp(0.0, 1.0)
        target_boxes = target_boxes.clamp(0.0, 1.0)

        l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction="none").sum() / float(max(1, num_boxes))
        giou = generalized_iou(pred_boxes, target_boxes)
        giou_loss = (1.0 - giou).sum() / float(max(1, num_boxes))
        return l1_loss, giou_loss

    def cardinality_loss(
        self,
        logits: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        objectness_logits: Optional[Tensor] = None,
    ) -> Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        if objectness_logits is not None:
            foreground_mass = torch.sigmoid(objectness_logits.to(dtype=logits.dtype))
        else:
            probabilities = logits.softmax(dim=-1)
            foreground_mass = 1.0 - probabilities[..., self.num_classes]
        predicted_counts = foreground_mass.sum(dim=1)
        target_counts = torch.as_tensor(
            [int(target["labels"].numel()) for target in targets],
            device=logits.device,
            dtype=predicted_counts.dtype,
        )
        return F.l1_loss(predicted_counts, target_counts, reduction="mean")

    def count_loss(self, count_logits: Optional[Tensor], targets: Sequence[Dict[str, Tensor]], reference: Tensor) -> Tensor:
        if count_logits is None:
            return reference.sum() * 0.0
        raw_counts = count_logits.reshape(-1).to(dtype=reference.dtype, device=reference.device)
        predicted_counts = F.softplus(raw_counts)
        target_counts = torch.as_tensor(
            [int(target["labels"].numel()) for target in targets],
            device=predicted_counts.device,
            dtype=predicted_counts.dtype,
        )
        return F.smooth_l1_loss(predicted_counts, target_counts, reduction="mean")

    def count_objectness_consistency_loss(
        self,
        objectness_logits: Optional[Tensor],
        count_logits: Optional[Tensor],
        reference: Tensor,
    ) -> Tensor:
        if objectness_logits is None or count_logits is None:
            return reference.sum() * 0.0
        objectness_counts = torch.sigmoid(objectness_logits.to(dtype=reference.dtype, device=reference.device)).sum(dim=1)
        predicted_counts = F.softplus(count_logits.reshape(-1).to(dtype=reference.dtype, device=reference.device))
        return F.smooth_l1_loss(objectness_counts, predicted_counts.detach(), reduction="mean")

    def _quality_targets_from_indices(
        self,
        quality_logits: Tensor,
        pred_boxes: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        target_quality = torch.zeros_like(quality_logits, dtype=quality_logits.dtype)
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            matched_predictions = pred_boxes[batch_index, query_indices]
            matched_targets = targets[batch_index]["boxes"][target_indices].to(
                device=pred_boxes.device,
                dtype=pred_boxes.dtype,
            )
            matched_iou = box_iou_from_xywh(matched_predictions, matched_targets).detach().clamp(0.0, 1.0)
            target_quality[batch_index, query_indices] = matched_iou.to(dtype=target_quality.dtype)
        return target_quality

    def quality_loss(
        self,
        quality_logits: Optional[Tensor],
        pred_boxes: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
        reference: Tensor,
    ) -> Tensor:
        if quality_logits is None:
            return reference.sum() * 0.0
        target_quality = self._quality_targets_from_indices(quality_logits, pred_boxes, targets, indices)
        quality_logits = quality_logits.to(dtype=reference.dtype, device=reference.device)
        target_quality = target_quality.to(dtype=quality_logits.dtype, device=quality_logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            quality_logits,
            target_quality,
            reduction="none",
        )
        positive_mask = target_quality > 0.0
        negative_mask = ~positive_mask
        zero = quality_logits.sum() * 0.0
        positive_loss = (
            bce_loss[positive_mask].mean()
            if positive_mask.any()
            else zero
        )
        negative_loss = (
            bce_loss[negative_mask].mean()
            if negative_mask.any()
            else zero
        )
        return positive_loss + 0.25 * negative_loss

    def _collect_matched_boxes(
        self,
        pred_boxes: Tensor,
        targets: Sequence[Dict[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        matched_predictions: List[Tensor] = []
        matched_targets: List[Tensor] = []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            matched_predictions.append(pred_boxes[batch_index, query_indices])
            matched_targets.append(
                targets[batch_index]["boxes"][target_indices].to(
                    device=pred_boxes.device,
                    dtype=pred_boxes.dtype,
                )
            )
        if not matched_predictions:
            empty = pred_boxes.new_zeros((0, 4))
            return empty, empty
        return torch.cat(matched_predictions, dim=0), torch.cat(matched_targets, dim=0)

    def _weighted_detection_loss_without_count(
        self,
        model_output,
        targets: Sequence[Dict[str, Tensor]],
        reference: Tensor,
    ) -> Tensor:
        logits, pred_boxes, objectness_logits = unpack_detection_output(model_output)
        logits = _finite_loss_logits(logits)
        if logits is None:
            raise ValueError("Model output thieu logits.")
        pred_boxes = _finite_loss_boxes(pred_boxes)
        objectness_logits = _finite_loss_logits(objectness_logits)
        quality_logits = unpack_quality_logits(model_output)
        quality_logits = _finite_loss_logits(quality_logits)
        targets = _finite_loss_targets(targets, device=logits.device)
        uses_separate_objectness = objectness_logits is not None
        matcher_output = {"logits": logits, "boxes": pred_boxes}
        if objectness_logits is not None:
            matcher_output["objectness_logits"] = objectness_logits
        indices = self.matcher(matcher_output, targets)
        num_boxes = sum(int(target["labels"].numel()) for target in targets)
        if uses_separate_objectness:
            cls_loss = self.matched_classification_loss(logits, targets, indices)
            target_objectness = self._objectness_targets_from_indices(objectness_logits, indices)
            objectness_loss = self.objectness_loss(
                objectness_logits,
                target_objectness,
                num_boxes=num_boxes,
            )
        else:
            target_classes = self._target_classes_from_indices(logits, targets, indices)
            cls_loss = self.classification_loss(logits, target_classes)
            objectness_loss = logits.sum() * 0.0
        matched_pred_boxes, matched_target_boxes = self._collect_matched_boxes(pred_boxes, targets, indices)
        bbox_l1_loss, bbox_giou_loss = self.bbox_loss(
            matched_pred_boxes,
            matched_target_boxes,
            num_boxes=num_boxes,
        )
        cardinality_loss = self.cardinality_loss(logits, targets, objectness_logits=objectness_logits)
        quality_loss = self.quality_loss(quality_logits, pred_boxes, targets, indices, reference=reference)
        return (
            self.cls_weight * cls_loss
            + self.objectness_weight * objectness_loss
            + self.bbox_l1_weight * bbox_l1_loss
            + self.bbox_giou_weight * bbox_giou_loss
            + self.cardinality_weight * cardinality_loss
            + self.quality_weight * quality_loss
        )

    def auxiliary_loss(
        self,
        aux_outputs,
        targets: Sequence[Dict[str, Tensor]],
        reference: Tensor,
    ) -> Tensor:
        if not aux_outputs:
            return reference.sum() * 0.0
        losses: List[Tensor] = []
        for aux_output in aux_outputs:
            losses.append(
                self._weighted_detection_loss_without_count(
                    aux_output,
                    targets,
                    reference=reference,
                )
            )
        if not losses:
            return reference.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        model_output,
        targets: Sequence[Dict[str, Tensor]],
        return_details: bool = False,
        debug_bbox: bool = False,
    ):
        logits, pred_boxes, objectness_logits = unpack_detection_output(model_output)
        logits = _finite_loss_logits(logits)
        if logits is None:
            raise ValueError("Model output thieu logits.")
        pred_boxes = _finite_loss_boxes(pred_boxes)
        objectness_logits = _finite_loss_logits(objectness_logits)
        count_logits = unpack_count_logits(model_output)
        count_logits = _finite_loss_logits(count_logits)
        quality_logits = unpack_quality_logits(model_output)
        quality_logits = _finite_loss_logits(quality_logits)
        targets = _finite_loss_targets(targets, device=logits.device)
        uses_separate_objectness = objectness_logits is not None
        if uses_separate_objectness and logits.shape[-1] != self.num_classes:
            raise ValueError(
                "Khi dung objectness rieng, logits phai co so kenh bang num_classes."
            )
        if not uses_separate_objectness and logits.shape[-1] != self.num_classes + 1:
            raise ValueError(
                "DETR legacy can logits co num_classes + 1 kenh hoac can objectness_logits rieng."
            )

        if debug_bbox and len(targets) > 0:
            with torch.no_grad():
                logger.info("\n" + "=" * 80)
                logger.info("DEBUG: BBox Coordinates and Format Validation")
                logger.info("=" * 80)

                pred_boxes_for_debug = pred_boxes.detach()
                target_boxes = targets[0]["boxes"].detach().to(
                    device=pred_boxes_for_debug.device,
                    dtype=pred_boxes_for_debug.dtype,
                )

                BboxDebugger.log_bbox_coordinates(
                    predictions=pred_boxes_for_debug,
                    targets=targets[0],
                    batch_idx=0,
                    prefix="LOSS_FORWARD",
                )

                pred_xywh_valid = BboxDebugger.check_bbox_format(
                    pred_boxes_for_debug,
                    expected_format="xywh",
                    name="predictions",
                )
                target_xywh_valid = BboxDebugger.check_bbox_format(
                    target_boxes,
                    expected_format="xywh",
                    name="targets",
                )
                pred_xyxy_valid = BboxDebugger.check_bbox_format(
                    pred_boxes_for_debug,
                    expected_format="xyxy",
                    name="predictions",
                )
                target_xyxy_valid = BboxDebugger.check_bbox_format(
                    target_boxes,
                    expected_format="xyxy",
                    name="targets",
                )
                logger.info(
                    "BBox format check: "
                    "pred_xywh=%s target_xywh=%s pred_xyxy=%s target_xyxy=%s",
                    pred_xywh_valid,
                    target_xywh_valid,
                    pred_xyxy_valid,
                    target_xyxy_valid,
                )

                if pred_boxes_for_debug.numel() > 0 and target_boxes.numel() > 0:
                    BboxDebugger.validate_iou_calculation(
                        pred_boxes=pred_boxes_for_debug[0],
                        target_boxes=target_boxes,
                        format="xywh",
                        iou_threshold=0.5,
                    )

        matcher_output = {"logits": logits, "boxes": pred_boxes}
        if objectness_logits is not None:
            matcher_output["objectness_logits"] = objectness_logits
        indices = self.matcher(matcher_output, targets)
        num_boxes = sum(int(target["labels"].numel()) for target in targets)
        if uses_separate_objectness:
            cls_loss = self.matched_classification_loss(logits, targets, indices)
            target_objectness = self._objectness_targets_from_indices(objectness_logits, indices)
            objectness_loss = self.objectness_loss(
                objectness_logits,
                target_objectness,
                num_boxes=num_boxes,
            )
        else:
            target_classes = self._target_classes_from_indices(logits, targets, indices)
            cls_loss = self.classification_loss(logits, target_classes)
            objectness_loss = logits.sum() * 0.0

        matched_pred_boxes, matched_target_boxes = self._collect_matched_boxes(pred_boxes, targets, indices)
        bbox_l1_loss, bbox_giou_loss = self.bbox_loss(
            matched_pred_boxes,
            matched_target_boxes,
            num_boxes=num_boxes,
        )
        cardinality_loss = self.cardinality_loss(logits, targets, objectness_logits=objectness_logits)
        count_loss = self.count_loss(count_logits, targets, reference=logits)
        quality_loss = self.quality_loss(quality_logits, pred_boxes, targets, indices, reference=logits)
        count_objectness_consistency_loss = self.count_objectness_consistency_loss(
            objectness_logits,
            count_logits,
            reference=logits,
        )
        aux_outputs = model_output.get("aux_outputs") if isinstance(model_output, dict) else None
        auxiliary_loss = (
            self.auxiliary_loss(aux_outputs, targets, reference=logits)
            if self.auxiliary_weight > 0.0
            else logits.sum() * 0.0
        )

        total_loss = (
            self.cls_weight * cls_loss
            + self.objectness_weight * objectness_loss
            + self.bbox_l1_weight * bbox_l1_loss
            + self.bbox_giou_weight * bbox_giou_loss
            + self.cardinality_weight * cardinality_loss
            + self.count_weight * count_loss
            + self.quality_weight * quality_loss
            + self.count_objectness_consistency_weight * count_objectness_consistency_loss
            + self.auxiliary_weight * auxiliary_loss
        )
        if not return_details:
            return total_loss

        details = {
            "loss": float(total_loss.detach().cpu().item()),
            "cls_loss": float(cls_loss.detach().cpu().item()),
            "objectness_loss": float(objectness_loss.detach().cpu().item()),
            "bbox_l1_loss": float(bbox_l1_loss.detach().cpu().item()),
            "bbox_giou_loss": float(bbox_giou_loss.detach().cpu().item()),
            "cardinality_loss": float(cardinality_loss.detach().cpu().item()),
            "count_loss": float(count_loss.detach().cpu().item()),
            "quality_loss": float(quality_loss.detach().cpu().item()),
            "count_objectness_consistency_loss": float(
                count_objectness_consistency_loss.detach().cpu().item()
            ),
            "auxiliary_loss": float(auxiliary_loss.detach().cpu().item()),
            "matched_queries": int(sum(int(query_indices.numel()) for query_indices, _ in indices)),
            "object_count": int(num_boxes),
        }
        if count_logits is not None:
            with torch.no_grad():
                details["count_prediction_mean"] = float(F.softplus(count_logits.reshape(-1)).mean().detach().cpu().item())
        if quality_logits is not None:
            with torch.no_grad():
                details["quality_prediction_mean"] = float(torch.sigmoid(quality_logits).mean().detach().cpu().item())
        details.update(self.get_loss_weights())
        return total_loss, details


HybridDetectionClassificationLoss = DETRSetCriterion
