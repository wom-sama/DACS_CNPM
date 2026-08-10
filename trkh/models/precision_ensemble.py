from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.models.model import (
    classification_logits_from_features,
    extract_detection_from_model_output,
)


def _classification_member_logits(
    model: nn.Module,
    images: Tensor,
    image_valid_mask: Tensor,
    bbox: Tensor,
) -> Tensor:
    if (
        hasattr(model, "forward_features")
        and hasattr(model, "head")
        and hasattr(model, "num_registers")
    ):
        features = model.forward_features(
            images,
            image_valid_mask=image_valid_mask,
            bbox_token_prior=bbox,
        )
        if not isinstance(features, dict):
            raise TypeError("Precision-ensemble member forward_features must return a dict.")
        features["bbox"] = bbox
        logits = classification_logits_from_features(model, features)
    else:
        logits, boxes, objectness_logits = extract_detection_from_model_output(model(images))
        if boxes is not None or objectness_logits is not None:
            raise TypeError("Precision ensemble supports classification members only.")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError(
            "Precision-ensemble member logits must have shape [batch, classes], "
            f"got {type(logits)!r} {getattr(logits, 'shape', None)}."
        )
    return logits


class PrecisionEnsembleClassifier(nn.Module):
    """Inference-only package for the validation-frozen two-member rule."""

    model_type = "precision_ensemble"
    temporal_frames = 1
    requires_spatial_metadata = True
    spatial_metadata_inputs = ("image_valid_mask", "bbox")
    supports_dynamic_batch = False
    certified_batch_sizes = (1,)
    supports_torchscript_deployment = False
    deployment_output_names = (
        "logits",
        "keeper_probabilities",
        "candidate_probabilities",
        "raw_blend",
        "decision_scores",
        "deployment_probabilities",
    )

    def __init__(
        self,
        keeper_model: nn.Module,
        candidate_model: nn.Module,
        *,
        num_classes: int,
        candidate_weight: float,
        focus_class: int,
        focus_margin_offset: float,
        minimum_probability: float = 1e-8,
    ) -> None:
        super().__init__()
        num_classes = int(num_classes)
        candidate_weight = float(candidate_weight)
        focus_class = int(focus_class)
        focus_margin_offset = float(focus_margin_offset)
        minimum_probability = float(minimum_probability)
        if num_classes < 2:
            raise ValueError("Precision ensemble requires at least two classes.")
        if not math.isfinite(candidate_weight) or not 0.0 <= candidate_weight <= 1.0:
            raise ValueError("candidate_weight must be finite and in [0, 1].")
        if not 0 <= focus_class < num_classes:
            raise ValueError(f"focus_class={focus_class} is outside {num_classes} classes.")
        if not math.isfinite(focus_margin_offset) or focus_margin_offset < 0.0:
            raise ValueError("focus_margin_offset must be finite and nonnegative.")
        if not math.isfinite(minimum_probability) or minimum_probability <= 0.0:
            raise ValueError("minimum_probability must be finite and positive.")

        self.keeper_model = keeper_model
        self.candidate_model = candidate_model
        self.num_classes = num_classes
        self.focus_class = focus_class
        self.register_buffer(
            "candidate_weight",
            torch.tensor(candidate_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "focus_margin_offset",
            torch.tensor(focus_margin_offset, dtype=torch.float32),
        )
        focus_mask = torch.zeros((1, num_classes), dtype=torch.float32)
        focus_mask[0, focus_class] = 1.0
        self.register_buffer("focus_mask", focus_mask)
        self.register_buffer(
            "minimum_probability",
            torch.tensor(minimum_probability, dtype=torch.float32),
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def keeper_weight_value(self) -> float:
        return 1.0 - float(self.candidate_weight.detach().cpu().item())

    def member(self, name: str) -> nn.Module:
        normalized = str(name).strip().lower()
        if normalized == "keeper":
            return self.keeper_model
        if normalized in {"candidate", "complement", "scratch"}:
            return self.candidate_model
        raise ValueError(f"Unknown precision-ensemble member: {name!r}")

    def _spatial_metadata(
        self,
        images: Tensor,
        image_valid_mask: Optional[Tensor],
        bbox: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
        batch_size = int(images.shape[0])
        height = int(images.shape[-2])
        width = int(images.shape[-1])
        if image_valid_mask is None:
            resolved_mask = torch.ones(
                (batch_size, height, width),
                device=images.device,
                dtype=torch.bool,
            )
        else:
            if tuple(image_valid_mask.shape) != (batch_size, height, width):
                raise ValueError(
                    "image_valid_mask shape mismatch: "
                    f"expected={(batch_size, height, width)}, "
                    f"got={tuple(image_valid_mask.shape)}"
                )
            resolved_mask = image_valid_mask.to(device=images.device) > 0.5

        if bbox is None:
            resolved_bbox = images.new_tensor((0.5, 0.5, 1.0, 1.0)).view(1, 4)
            resolved_bbox = resolved_bbox.expand(batch_size, -1)
        else:
            if tuple(bbox.shape) != (batch_size, 4):
                raise ValueError(
                    f"bbox shape mismatch: expected={(batch_size, 4)}, got={tuple(bbox.shape)}"
                )
            resolved_bbox = bbox.to(device=images.device, dtype=torch.float32)
        return resolved_mask, resolved_bbox

    def member_logits(
        self,
        images: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        resolved_mask, resolved_bbox = self._spatial_metadata(
            images,
            image_valid_mask,
            bbox,
        )
        keeper_logits = _classification_member_logits(
            self.keeper_model,
            images,
            resolved_mask,
            resolved_bbox,
        )
        candidate_logits = _classification_member_logits(
            self.candidate_model,
            images,
            resolved_mask,
            resolved_bbox,
        )
        if tuple(keeper_logits.shape) != tuple(candidate_logits.shape):
            raise ValueError(
                "Precision-ensemble member shape mismatch: "
                f"keeper={tuple(keeper_logits.shape)}, "
                f"candidate={tuple(candidate_logits.shape)}"
            )
        if int(keeper_logits.shape[1]) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {int(keeper_logits.shape[1])}."
            )
        return keeper_logits, candidate_logits

    def fuse_logits(
        self,
        keeper_logits: Tensor,
        candidate_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        keeper_probabilities = F.softmax(keeper_logits.float(), dim=1)
        candidate_probabilities = F.softmax(candidate_logits.float(), dim=1)
        candidate_weight = self.candidate_weight.to(
            device=keeper_probabilities.device,
            dtype=keeper_probabilities.dtype,
        )
        raw_blend = (
            keeper_probabilities * (1.0 - candidate_weight)
            + candidate_probabilities * candidate_weight
        )
        decision_scores = raw_blend - self.focus_mask.to(
            device=raw_blend.device,
            dtype=raw_blend.dtype,
        ) * self.focus_margin_offset.to(
            device=raw_blend.device,
            dtype=raw_blend.dtype,
        )
        safe_scores = torch.clamp_min(
            decision_scores,
            self.minimum_probability.to(
                device=raw_blend.device,
                dtype=raw_blend.dtype,
            ),
        )
        deployment_probabilities = safe_scores / safe_scores.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        logits = torch.log(deployment_probabilities.clamp_min(1e-8))
        return (
            logits,
            keeper_probabilities,
            candidate_probabilities,
            raw_blend,
            decision_scores,
        )

    def audit_outputs(
        self,
        images: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        keeper_logits, candidate_logits = self.member_logits(
            images,
            image_valid_mask=image_valid_mask,
            bbox=bbox,
        )
        (
            logits,
            keeper_probabilities,
            candidate_probabilities,
            raw_blend,
            decision_scores,
        ) = self.fuse_logits(keeper_logits, candidate_logits)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "keeper_logits": keeper_logits,
            "candidate_logits": candidate_logits,
            "keeper_probabilities": keeper_probabilities,
            "candidate_probabilities": candidate_probabilities,
            "raw_blend": raw_blend,
            "decision_scores": decision_scores,
        }

    def forward(
        self,
        images: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox: Optional[Tensor] = None,
    ) -> Tensor:
        keeper_logits, candidate_logits = self.member_logits(
            images,
            image_valid_mask=image_valid_mask,
            bbox=bbox,
        )
        logits, _, _, _, _ = self.fuse_logits(keeper_logits, candidate_logits)
        return logits

    def deployment_outputs(
        self,
        images: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        outputs = self.audit_outputs(
            images,
            image_valid_mask=image_valid_mask,
            bbox=bbox,
        )
        return (
            outputs["logits"],
            outputs["keeper_probabilities"],
            outputs["candidate_probabilities"],
            outputs["raw_blend"],
            outputs["decision_scores"],
            outputs["probabilities"],
        )

    def validate_rule(
        self,
        *,
        candidate_weight: float,
        focus_class: int,
        focus_margin_offset: float,
        tolerance: float = 1e-8,
    ) -> None:
        observed_weight = float(self.candidate_weight.detach().cpu().item())
        observed_offset = float(self.focus_margin_offset.detach().cpu().item())
        if abs(observed_weight - float(candidate_weight)) > float(tolerance):
            raise ValueError(
                "Checkpoint candidate-weight buffer differs from model_config: "
                f"{observed_weight} vs {candidate_weight}."
            )
        if int(self.focus_class) != int(focus_class):
            raise ValueError(
                "Checkpoint focus class differs from model_config: "
                f"{self.focus_class} vs {focus_class}."
            )
        if abs(observed_offset - float(focus_margin_offset)) > float(tolerance):
            raise ValueError(
                "Checkpoint focus-margin buffer differs from model_config: "
                f"{observed_offset} vs {focus_margin_offset}."
            )
