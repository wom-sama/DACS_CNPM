from __future__ import annotations

import math
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


class LogitNormCrossEntropyLoss(_SoftTargetLossBase):
    """Cross entropy on L2-normalized logits.

    LogitNorm removes logit magnitude from the classification loss and optimizes
    only the logit direction. This can reduce overconfident boundary decisions
    without changing the model architecture or raw data.
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        temperature: float = 0.04,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.temperature = float(max(float(temperature), 1e-6))
        self.eps = float(max(float(eps), 1e-12))

    def _normalize_logits(self, logits: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("LogitNormCrossEntropyLoss yeu cau logits co shape [batch, num_classes].")
        normalized = F.normalize(logits.float(), p=2.0, dim=1, eps=self.eps)
        return normalized / self.temperature

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        _, target_probs = self._prepare_targets(logits, targets)
        normalized_logits = self._normalize_logits(logits)
        target_probs = target_probs.to(dtype=normalized_logits.dtype)
        return self._soft_cross_entropy(normalized_logits, target_probs).to(dtype=logits.dtype)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_sample_loss(logits, targets).mean()


class SymmetricCrossEntropyLoss(_SoftTargetLossBase):
    """Symmetric cross entropy for noisy boundary labels.

    The CE term keeps ordinary class separation, while the reverse CE term is
    bounded by clipped targets and is less dominated by very hard samples.
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        alpha: float = 0.1,
        beta: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.alpha = float(max(0.0, alpha))
        self.beta = float(max(0.0, beta))
        self.epsilon = float(min(max(float(epsilon), 1e-12), 1.0))

    def _target_sample_weights(self, logits: Tensor, target_probs: Tensor) -> Optional[Tensor]:
        class_weights = self._effective_class_weights(logits)
        if class_weights is None:
            return None
        return (target_probs * class_weights.unsqueeze(0)).sum(dim=1).clamp(min=0.0)

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("SymmetricCrossEntropyLoss yeu cau logits co shape [batch, num_classes].")
        _, target_probs = self._prepare_targets(logits, targets)
        ce_loss = self._soft_cross_entropy(logits, target_probs)

        probabilities = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        clipped_targets = target_probs.clamp(min=self.epsilon, max=1.0)
        reverse_ce_loss = -(probabilities * clipped_targets.log()).sum(dim=1)
        sample_weights = self._target_sample_weights(logits, target_probs)
        if sample_weights is not None:
            reverse_ce_loss = reverse_ce_loss * sample_weights

        return (self.alpha * ce_loss + self.beta * reverse_ce_loss).to(dtype=logits.dtype)

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


class GeneralizedCrossEntropyLoss(_SoftTargetLossBase):
    """Generalized cross entropy for noisy or ambiguous labels.

    GCE caps the contribution of low target-probability samples compared with
    ordinary cross entropy, which is useful when boundary labels are noisy.
    Optional LDAM margins keep the existing long-tail behavior available.
    """

    def __init__(
        self,
        class_counts: Optional[Sequence[int]] = None,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        q: float = 0.7,
        max_margin: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.q = float(min(1.0, max(float(q), 1e-6)))
        self.scale = float(max(1.0, scale))
        self.use_ldam = bool(class_counts) and float(max_margin) > 0.0
        if self.use_ldam:
            counts = torch.tensor([max(1, int(count)) for count in class_counts or []], dtype=torch.float32)
            margins = 1.0 / torch.sqrt(torch.sqrt(counts))
            margins = margins * (float(max_margin) / margins.max().clamp(min=1e-12))
            self.register_buffer("class_margins", margins)
        else:
            self.register_buffer("class_margins", None)

    def _adjust_logits(self, logits: Tensor, hard_targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("GeneralizedCrossEntropyLoss yeu cau logits co shape [batch, num_classes].")
        if self.class_margins is None:
            return logits
        margins = self.class_margins.to(device=logits.device, dtype=logits.dtype)
        if margins.numel() != logits.size(1):
            raise ValueError("So class cua GeneralizedCrossEntropyLoss khong khop logits.")
        target_margins = margins[hard_targets].unsqueeze(1)
        hard_target_mask = F.one_hot(hard_targets, num_classes=logits.size(1)).to(dtype=logits.dtype)
        return (logits - hard_target_mask * target_margins) * self.scale

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        hard_targets, target_probs = self._prepare_targets(logits, targets)
        adjusted_logits = self._adjust_logits(logits, hard_targets)
        probabilities = torch.softmax(adjusted_logits.float(), dim=1).to(dtype=adjusted_logits.dtype)
        target_probability_q = (
            target_probs * probabilities.clamp(min=1e-7, max=1.0).pow(self.q)
        ).sum(dim=1).clamp(min=0.0, max=1.0)
        losses = (1.0 - target_probability_q) / self.q

        class_weights = self._effective_class_weights(adjusted_logits)
        if class_weights is not None:
            sample_weights = (target_probs * class_weights.unsqueeze(0)).sum(dim=1).clamp(min=0.0)
            losses = losses * sample_weights
        return losses.to(dtype=logits.dtype)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_sample_loss(logits, targets).mean()


class LabelDistributionallyRobustLoss(_SoftTargetLossBase):
    """KL-regularized label-distributionally robust multiclass loss.

    This follows the official LDR-KL formulation: positive scores are normalized
    per sample, then a KL-regularized worst-case distribution emphasizes the
    largest target-relative class margins. Unlike fixed label smoothing, the
    adversarial class weights are recomputed from each sample's logits.
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        margin: float = 2.0,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        if float(margin) < 0.0:
            raise ValueError("LabelDistributionallyRobustLoss margin phai >= 0.")
        if float(temperature) <= 0.0:
            raise ValueError("LabelDistributionallyRobustLoss temperature phai > 0.")
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.eps = float(max(float(eps), 1e-12))

    def _target_sample_weights(self, logits: Tensor, target_probs: Tensor) -> Optional[Tensor]:
        class_weights = self._effective_class_weights(logits)
        if class_weights is None:
            return None
        return (target_probs * class_weights.unsqueeze(0)).sum(dim=1).clamp(min=0.0)

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "LabelDistributionallyRobustLoss yeu cau logits co shape [batch, num_classes]."
            )
        if logits.size(1) < 2:
            raise ValueError("LabelDistributionallyRobustLoss can it nhat 2 class.")

        _, target_probs = self._prepare_targets(logits, targets)
        work_logits = logits.float()
        target_probs = target_probs.to(dtype=work_logits.dtype)

        positive_scores = F.softplus(work_logits)
        positive_scores = positive_scores / positive_scores.mean(dim=1, keepdim=True).clamp(
            min=self.eps
        )
        target_scores = (positive_scores * target_probs).sum(dim=1, keepdim=True)
        robust_margins = (
            self.margin * (1.0 - target_probs)
            + positive_scores
            - target_scores
        )
        scaled_margins = robust_margins / self.temperature
        losses = self.temperature * (
            torch.logsumexp(scaled_margins, dim=1) - math.log(float(logits.size(1)))
        )

        sample_weights = self._target_sample_weights(work_logits, target_probs)
        if sample_weights is not None:
            losses = losses * sample_weights
        return losses.to(dtype=logits.dtype)

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


class SeesawCrossEntropyLoss(_SoftTargetLossBase):
    """Seesaw-style cross entropy for long-tail false-positive control.

    The mitigation term can reduce negative pressure on tail classes. Set
    ``mitigation_power=0`` to disable that part and keep only the compensation
    term, which increases the penalty for negative classes whose probability
    exceeds the target-class probability.
    """

    def __init__(
        self,
        class_counts: Sequence[int],
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        mitigation_power: float = 0.8,
        compensation_power: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        if not class_counts:
            raise ValueError("SeesawCrossEntropyLoss yeu cau class_counts khong rong.")
        counts = torch.tensor([max(1, int(count)) for count in class_counts], dtype=torch.float32)
        self.register_buffer("class_counts", counts)
        self.mitigation_power = float(max(0.0, mitigation_power))
        self.compensation_power = float(max(0.0, compensation_power))
        self.eps = float(max(float(eps), 1e-12))

    def _adjust_logits(self, logits: Tensor, hard_targets: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("SeesawCrossEntropyLoss yeu cau logits co shape [batch, num_classes].")
        if hard_targets.ndim != 1 or hard_targets.size(0) != logits.size(0):
            raise ValueError("SeesawCrossEntropyLoss targets phai co shape [batch].")

        num_classes = logits.size(1)
        counts = self.class_counts.to(device=logits.device, dtype=logits.dtype)
        if counts.numel() != num_classes:
            raise ValueError("So class cua SeesawCrossEntropyLoss khong khop logits.")

        target_mask = F.one_hot(hard_targets.to(torch.long), num_classes=num_classes).to(dtype=logits.dtype)
        negative_mask = 1.0 - target_mask
        seesaw_weights = torch.ones_like(logits)

        if self.mitigation_power > 0.0:
            target_counts = counts[hard_targets].unsqueeze(1)
            count_ratio = counts.unsqueeze(0) / target_counts.clamp(min=1.0)
            mitigation = torch.where(
                count_ratio < 1.0,
                count_ratio.clamp(min=self.eps).pow(self.mitigation_power),
                torch.ones_like(count_ratio),
            )
            seesaw_weights = seesaw_weights * mitigation

        if self.compensation_power > 0.0:
            probabilities = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
            target_probabilities = probabilities.gather(1, hard_targets.view(-1, 1)).clamp(min=self.eps)
            probability_ratio = probabilities / target_probabilities
            compensation = torch.where(
                probability_ratio > 1.0,
                probability_ratio.clamp(min=1.0).pow(self.compensation_power),
                torch.ones_like(probability_ratio),
            )
            seesaw_weights = seesaw_weights * compensation

        seesaw_weights = seesaw_weights * negative_mask + target_mask
        return logits + seesaw_weights.clamp(min=self.eps).log() * negative_mask

    def per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        hard_targets, target_probs = self._prepare_targets(logits, targets)
        adjusted_logits = self._adjust_logits(logits, hard_targets)
        return self._soft_cross_entropy(adjusted_logits, target_probs).to(dtype=logits.dtype)

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

    def forward(
        self,
        embeddings: Tensor,
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("SupervisedContrastiveLoss yeu cau embeddings co shape [batch, dim].")
        if targets.ndim != 1:
            raise ValueError("SupervisedContrastiveLoss yeu cau targets co shape [batch].")
        if embeddings.size(0) != targets.size(0):
            raise ValueError("So sample cua embeddings va targets khong khop.")
        if sample_weights is not None:
            if sample_weights.ndim != 1:
                raise ValueError("SupervisedContrastiveLoss sample_weights phai co shape [batch].")
            if sample_weights.size(0) != embeddings.size(0):
                raise ValueError("So sample cua sample_weights va embeddings khong khop.")
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0

        device = embeddings.device
        targets = targets.to(device=device, dtype=torch.long)
        weights: Optional[Tensor] = None
        if sample_weights is not None:
            weights = sample_weights.to(device=device, dtype=torch.float32).view(-1).clamp(min=0.0)
            if not bool((weights > 0).any().item()):
                return embeddings.sum() * 0.0
        features = F.normalize(embeddings.float(), dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = targets.unsqueeze(0).eq(targets.unsqueeze(1)) & ~self_mask
        positive_weight_mask = positive_mask.to(dtype=logits.dtype)
        if weights is not None:
            positive_weight_mask = positive_weight_mask * weights.unsqueeze(0).to(dtype=logits.dtype)
        positive_weight_sum = positive_weight_mask.sum(dim=1)
        valid_anchor_mask = positive_weight_sum > 0
        if weights is not None:
            valid_anchor_mask = valid_anchor_mask & (weights > 0)
        if not bool(valid_anchor_mask.any().item()):
            return embeddings.sum() * 0.0

        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
        per_anchor_loss = -(positive_weight_mask * log_prob).sum(dim=1) / positive_weight_sum.clamp(min=1e-12)

        valid_losses = per_anchor_loss[valid_anchor_mask]
        anchor_weights = (
            weights[valid_anchor_mask].to(dtype=valid_losses.dtype)
            if weights is not None
            else torch.ones_like(valid_losses)
        )
        if not self.class_balanced:
            anchor_weights = anchor_weights / anchor_weights.mean().clamp(min=1e-12)
            return (valid_losses * anchor_weights).mean()

        class_counts = torch.bincount(targets, minlength=int(targets.max().item()) + 1).to(
            device=device,
            dtype=valid_losses.dtype,
        )
        class_anchor_weights = 1.0 / torch.sqrt(class_counts[targets[valid_anchor_mask]].clamp(min=1.0))
        anchor_weights = anchor_weights * class_anchor_weights
        anchor_weights = anchor_weights / anchor_weights.mean().clamp(min=1e-12)
        return (valid_losses * anchor_weights).mean()
