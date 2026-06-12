from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _SoftTargetLossBase(nn.Module):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.register_buffer("class_weight_multipliers", None)
        self.label_smoothing = float(max(0.0, label_smoothing))

    def _prepare_targets(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        num_classes = logits.size(1)
        if targets.dtype in (torch.int32, torch.int64) and targets.ndim == 1:
            hard_targets = targets.to(torch.int64)
            target_probs = F.one_hot(hard_targets, num_classes=num_classes).to(dtype=logits.dtype)
        else:
            if targets.ndim != 2 or targets.size(1) != num_classes:
                raise ValueError("Soft targets phai co shape [batch, num_classes].")
            target_probs = targets.to(dtype=logits.dtype)
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
            hard_targets = target_probs.argmax(dim=1)

        if self.label_smoothing > 0.0:
            smoothing = float(self.label_smoothing)
            target_probs = (1.0 - smoothing) * target_probs + (smoothing / num_classes)
        return hard_targets, target_probs

    def _soft_cross_entropy(self, logits: Tensor, target_probs: Tensor) -> Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        class_weights = self._effective_class_weights(logits)
        if class_weights is None:
            return -(target_probs * log_probabilities).sum(dim=1)

        weighted_targets = target_probs * class_weights.unsqueeze(0)
        return -(weighted_targets * log_probabilities).sum(dim=1)

    def _effective_class_weights(self, logits: Tensor) -> Optional[Tensor]:
        if self.weight is None and self.class_weight_multipliers is None:
            return None

        if self.weight is None:
            class_weights = torch.ones(logits.size(1), device=logits.device, dtype=logits.dtype)
        else:
            class_weights = self.weight.to(device=logits.device, dtype=logits.dtype)

        if self.class_weight_multipliers is not None:
            multipliers = self.class_weight_multipliers.to(device=logits.device, dtype=logits.dtype)
            if multipliers.numel() != logits.size(1):
                raise ValueError("class weight multipliers phai co do dai bang num_classes.")
            class_weights = class_weights * multipliers.clamp(min=0.0)
        return class_weights

    def set_class_weight_multipliers(self, multipliers: Optional[Tensor]) -> None:
        if multipliers is None:
            self.class_weight_multipliers = None
            return
        self.class_weight_multipliers = multipliers.detach().to(dtype=torch.float32).clone()

    def reset_class_weight_multipliers(self) -> None:
        self.set_class_weight_multipliers(None)


class FocalCrossEntropyLoss(_SoftTargetLossBase):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        gamma: float = 2.0,
        focal_mix: float = 0.35,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.gamma = float(max(0.0, gamma))
        self.focal_mix = float(min(max(focal_mix, 0.0), 1.0))

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        _, target_probs = self._prepare_targets(logits, targets)
        ce_loss = self._soft_cross_entropy(logits, target_probs)
        if self.focal_mix <= 0.0 or self.gamma <= 0.0:
            return ce_loss

        probabilities = torch.softmax(logits, dim=1)
        target_probabilities = (probabilities * target_probs).sum(dim=1).clamp(min=0.0, max=1.0)
        focal_term = (1.0 - target_probabilities).pow(self.gamma)
        focal_loss = focal_term * ce_loss
        return (1.0 - self.focal_mix) * ce_loss + self.focal_mix * focal_loss

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_sample_loss(logits, targets).mean()


class LDAMFocalLoss(_SoftTargetLossBase):
    def __init__(
        self,
        class_counts: Sequence[int],
        weight: Optional[Tensor] = None,
        gamma: float = 2.0,
        focal_mix: float = 0.35,
        label_smoothing: float = 0.0,
        max_margin: float = 0.5,
        scale: float = 30.0,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        if not class_counts:
            raise ValueError("LDAMFocalLoss yeu cau class_counts khong rong.")
        counts = torch.tensor([max(1, int(count)) for count in class_counts], dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (float(max_margin) / margins.max().clamp(min=1e-12))
        self.register_buffer("class_margins", margins)
        self.gamma = float(max(0.0, gamma))
        self.focal_mix = float(min(max(focal_mix, 0.0), 1.0))
        self.scale = float(max(1.0, scale))

    def _apply_ldam_margin(self, logits: Tensor, hard_targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("LDAMFocalLoss yeu cau logits co shape [batch, num_classes].")
        margins = self.class_margins.to(device=logits.device, dtype=logits.dtype)
        target_margins = margins[hard_targets].unsqueeze(1)
        hard_target_mask = F.one_hot(hard_targets, num_classes=logits.size(1)).to(dtype=logits.dtype)
        adjusted_logits = logits - hard_target_mask * target_margins
        return adjusted_logits * self.scale

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        hard_targets, target_probs = self._prepare_targets(logits, targets)
        adjusted_logits = self._apply_ldam_margin(logits, hard_targets)
        ce_loss = self._soft_cross_entropy(adjusted_logits, target_probs)
        if self.focal_mix <= 0.0 or self.gamma <= 0.0:
            return ce_loss

        probabilities = torch.softmax(adjusted_logits, dim=1)
        target_probabilities = (probabilities * target_probs).sum(dim=1).clamp(min=0.0, max=1.0)
        focal_term = (1.0 - target_probabilities).pow(self.gamma)
        focal_loss = focal_term * ce_loss
        return (1.0 - self.focal_mix) * ce_loss + self.focal_mix * focal_loss

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_sample_loss(logits, targets).mean()


class BalancedSoftmaxFocalLoss(_SoftTargetLossBase):
    def __init__(
        self,
        class_counts: Sequence[int],
        weight: Optional[Tensor] = None,
        gamma: float = 2.0,
        focal_mix: float = 0.0,
        label_smoothing: float = 0.0,
        prior_tau: float = 1.0,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        if not class_counts:
            raise ValueError("BalancedSoftmaxFocalLoss yeu cau class_counts khong rong.")
        counts = torch.tensor([max(1, int(count)) for count in class_counts], dtype=torch.float32)
        priors = counts / counts.sum().clamp(min=1.0)
        self.register_buffer("log_priors", torch.log(priors.clamp(min=1e-12)))
        self.gamma = float(max(0.0, gamma))
        self.focal_mix = float(min(max(focal_mix, 0.0), 1.0))
        self.prior_tau = float(max(0.0, prior_tau))

    def _adjust_logits(self, logits: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("BalancedSoftmaxFocalLoss yeu cau logits co shape [batch, num_classes].")
        log_priors = self.log_priors.to(device=logits.device, dtype=logits.dtype)
        if log_priors.numel() != logits.size(1):
            raise ValueError("So class cua BalancedSoftmaxFocalLoss khong khop logits.")
        return logits + float(self.prior_tau) * log_priors.unsqueeze(0)

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        _, target_probs = self._prepare_targets(logits, targets)
        adjusted_logits = self._adjust_logits(logits)
        ce_loss = self._soft_cross_entropy(adjusted_logits, target_probs)
        if self.focal_mix <= 0.0 or self.gamma <= 0.0:
            return ce_loss

        probabilities = torch.softmax(adjusted_logits, dim=1)
        target_probabilities = (probabilities * target_probs).sum(dim=1).clamp(min=0.0, max=1.0)
        focal_term = (1.0 - target_probabilities).pow(self.gamma)
        focal_loss = focal_term * ce_loss
        return (1.0 - self.focal_mix) * ce_loss + self.focal_mix * focal_loss

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_sample_loss(logits, targets).mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.12,
        class_balanced: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(max(1e-6, temperature))
        self.class_balanced = bool(class_balanced)

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("SupervisedContrastiveLoss yeu cau embeddings co shape [batch, dim].")
        if targets.ndim != 1:
            raise ValueError("SupervisedContrastiveLoss yeu cau targets co shape [batch].")
        if embeddings.size(0) != targets.size(0):
            raise ValueError("So sample cua embeddings va targets khong khop.")
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0

        device = embeddings.device
        targets = targets.to(device=device, dtype=torch.long)
        features = F.normalize(embeddings.float(), dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = targets.unsqueeze(0).eq(targets.unsqueeze(1)) & ~self_mask
        valid_anchor_mask = positive_mask.any(dim=1)
        if not bool(valid_anchor_mask.any().item()):
            return embeddings.sum() * 0.0

        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
        positive_count = positive_mask.sum(dim=1).clamp(min=1).to(dtype=log_prob.dtype)
        per_anchor_loss = -(
            positive_mask.to(dtype=log_prob.dtype) * log_prob
        ).sum(dim=1) / positive_count

        valid_losses = per_anchor_loss[valid_anchor_mask]
        if not self.class_balanced:
            return valid_losses.mean()

        class_counts = torch.bincount(targets, minlength=int(targets.max().item()) + 1).to(
            device=device,
            dtype=valid_losses.dtype,
        )
        anchor_weights = 1.0 / torch.sqrt(class_counts[targets[valid_anchor_mask]].clamp(min=1.0))
        anchor_weights = anchor_weights / anchor_weights.mean().clamp(min=1e-12)
        return (valid_losses * anchor_weights).mean()
