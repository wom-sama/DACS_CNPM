from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from trkh.training.friendly_adversarial import normalized_to_rgb, rgb_to_normalized


LogitFeatureForward = Callable[[Tensor], Tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class HardPositiveResult:
    augmented_images: Tensor
    delta_rgb: Tensor
    attack_mask: Tensor
    feature_distance: Tensor
    true_log_probability_drop: Tensor
    constraint_satisfied: Tensor
    selected_step: Tensor


def _validate_inputs(
    *,
    images: Tensor,
    labels: Tensor,
    attack_mask: Tensor,
    sample_indices: Tensor,
    epsilon: float,
    steps: int,
    label_margin: float,
) -> None:
    if images.ndim != 4 or int(images.size(1)) != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    expected_mask_shape = (
        int(images.size(0)),
        1,
        int(images.size(2)),
        int(images.size(3)),
    )
    if tuple(attack_mask.shape) != expected_mask_shape:
        raise ValueError("attack_mask shape differs from images")
    if labels.ndim != 1 or int(labels.numel()) != int(images.size(0)):
        raise ValueError("labels shape differs from images")
    if sample_indices.ndim != 1 or int(sample_indices.numel()) != int(images.size(0)):
        raise ValueError("sample_indices shape differs from images")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if float(label_margin) < 0.0:
        raise ValueError("label_margin must be non-negative")
    if bool((attack_mask.flatten(1).sum(dim=1) == 0).any().item()):
        raise ValueError("attack masks must be non-empty")


def _sample_noise(
    *,
    sample_indices: Tensor,
    shape: Sequence[int],
    seed: int,
    draw: int,
    normal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if len(shape) != 3:
        raise ValueError("per-sample noise shape must have three dimensions")
    rows = []
    modulus = (1 << 63) - 1
    for sample_index in sample_indices.detach().cpu().tolist():
        row_seed = (
            int(seed)
            + 1_000_003 * int(sample_index)
            + 97_409 * int(draw)
        ) % modulus
        generator = torch.Generator(device="cpu")
        generator.manual_seed(row_seed)
        if normal:
            row = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
        else:
            row = (
                torch.rand(tuple(shape), generator=generator, dtype=torch.float32) * 2.0
                - 1.0
            )
        rows.append(row)
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


def _normalized_features(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("forward features must have shape [B,D]")
    return F.normalize(features.float(), dim=1, eps=1e-6)


def _evaluate_candidate(
    *,
    candidate_rgb: Tensor,
    labels: Tensor,
    clean_log_probability: Tensor,
    clean_features: Tensor,
    forward_logits_features: LogitFeatureForward,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Tensor, Tensor, Tensor]:
    logits, features = forward_logits_features(
        rgb_to_normalized(candidate_rgb, mean=mean, std=std)
    )
    if logits.ndim != 2 or int(logits.size(0)) != int(labels.numel()):
        raise ValueError("forward logits must have shape [B,C]")
    normalized_features = _normalized_features(features)
    feature_distance = torch.linalg.vector_norm(
        normalized_features - clean_features,
        ord=2,
        dim=1,
    )
    log_probability = F.log_softmax(logits.float(), dim=1).gather(
        1, labels.long().view(-1, 1)
    ).squeeze(1)
    return logits.float(), feature_distance, clean_log_probability - log_probability


def _update_best(
    *,
    best_rgb: Tensor,
    best_distance: Tensor,
    best_drop: Tensor,
    best_step: Tensor,
    candidate_rgb: Tensor,
    candidate_distance: Tensor,
    candidate_drop: Tensor,
    step: int,
    label_margin: float,
    require_feasible: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    feasible = candidate_drop <= float(label_margin) + 1e-7
    eligible = feasible if bool(require_feasible) else torch.ones_like(feasible)
    improve = eligible & (candidate_distance > best_distance)
    return (
        torch.where(improve.view(-1, 1, 1, 1), candidate_rgb, best_rgb),
        torch.where(improve, candidate_distance, best_distance),
        torch.where(improve, candidate_drop, best_drop),
        torch.where(
            improve,
            torch.full_like(best_step, int(step)),
            best_step,
        ),
    )


def generate_projected_hard_positives(
    *,
    images: Tensor,
    labels: Tensor,
    attack_mask: Tensor,
    sample_indices: Tensor,
    forward_logits_features: LogitFeatureForward,
    mean: Sequence[float],
    std: Sequence[float],
    epsilon: float,
    steps: int,
    label_margin: float,
    seed: int,
    label_penalty: bool = True,
    require_feasible_selection: bool = True,
    initial_noise_std: float = 0.01,
) -> HardPositiveResult:
    _validate_inputs(
        images=images,
        labels=labels,
        attack_mask=attack_mask,
        sample_indices=sample_indices,
        epsilon=epsilon,
        steps=steps,
        label_margin=label_margin,
    )
    if float(initial_noise_std) < 0.0:
        raise ValueError("initial_noise_std must be non-negative")

    original_rgb = normalized_to_rgb(images.detach(), mean=mean, std=std)
    mask = attack_mask.to(device=images.device, dtype=original_rgb.dtype)
    with torch.no_grad():
        clean_logits, clean_feature_values = forward_logits_features(images.detach())
        clean_features = _normalized_features(clean_feature_values).detach()
        clean_log_probability = F.log_softmax(clean_logits.float(), dim=1).gather(
            1, labels.long().view(-1, 1)
        ).squeeze(1).detach()

    initial_noise = _sample_noise(
        sample_indices=sample_indices,
        shape=original_rgb.shape[1:],
        seed=int(seed),
        draw=0,
        normal=True,
        device=original_rgb.device,
        dtype=original_rgb.dtype,
    )
    initial_delta = (float(initial_noise_std) * initial_noise).clamp(
        min=-float(epsilon),
        max=float(epsilon),
    )
    current_rgb = (original_rgb + initial_delta * mask).clamp(0.0, 1.0)

    batch_size = int(images.size(0))
    best_rgb = original_rgb.clone()
    best_distance = torch.zeros(
        batch_size, device=images.device, dtype=torch.float32
    )
    best_drop = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
    best_step = torch.zeros(batch_size, device=images.device, dtype=torch.long)

    for step_index in range(1, int(steps) + 1):
        variable_rgb = current_rgb.detach().requires_grad_(True)
        _, feature_distance, true_log_probability_drop = _evaluate_candidate(
            candidate_rgb=variable_rgb,
            labels=labels,
            clean_log_probability=clean_log_probability,
            clean_features=clean_features,
            forward_logits_features=forward_logits_features,
            mean=mean,
            std=std,
        )
        lagrange = 10.0 ** (float(step_index) / float(steps))
        penalty = F.relu(true_log_probability_drop - float(label_margin))
        objective = feature_distance
        if bool(label_penalty):
            objective = objective - float(lagrange) * penalty
        gradient = torch.autograd.grad(
            objective.sum(),
            variable_rgb,
            only_inputs=True,
        )[0]
        masked_gradient = gradient * mask
        gradient_norm = torch.linalg.vector_norm(
            masked_gradient.flatten(1),
            ord=2,
            dim=1,
        ).clamp_min(1e-12)
        normalized_gradient = masked_gradient / gradient_norm.view(-1, 1, 1, 1)
        step_size = float(epsilon) * (
            0.1 ** (float(step_index) / float(steps))
        )
        proposal = variable_rgb.detach() + float(step_size) * normalized_gradient
        delta = (proposal - original_rgb).clamp(
            min=-float(epsilon),
            max=float(epsilon),
        )
        current_rgb = (original_rgb + delta * mask).clamp(0.0, 1.0).detach()
        with torch.no_grad():
            _, candidate_distance, candidate_drop = _evaluate_candidate(
                candidate_rgb=current_rgb,
                labels=labels,
                clean_log_probability=clean_log_probability,
                clean_features=clean_features,
                forward_logits_features=forward_logits_features,
                mean=mean,
                std=std,
            )
            best_rgb, best_distance, best_drop, best_step = _update_best(
                best_rgb=best_rgb,
                best_distance=best_distance,
                best_drop=best_drop,
                best_step=best_step,
                candidate_rgb=current_rgb,
                candidate_distance=candidate_distance,
                candidate_drop=candidate_drop,
                step=step_index,
                label_margin=label_margin,
                require_feasible=require_feasible_selection,
            )

    delta_rgb = (best_rgb - original_rgb) * mask
    augmented_images = rgb_to_normalized(
        original_rgb + delta_rgb,
        mean=mean,
        std=std,
    )
    constraint_satisfied = best_drop <= float(label_margin) + 1e-7
    return HardPositiveResult(
        augmented_images=augmented_images.detach(),
        delta_rgb=delta_rgb.detach(),
        attack_mask=attack_mask.detach().to(dtype=torch.bool),
        feature_distance=best_distance.detach(),
        true_log_probability_drop=best_drop.detach(),
        constraint_satisfied=constraint_satisfied.detach(),
        selected_step=best_step.detach(),
    )


def generate_random_feasible_hard_positives(
    *,
    images: Tensor,
    labels: Tensor,
    attack_mask: Tensor,
    sample_indices: Tensor,
    forward_logits_features: LogitFeatureForward,
    mean: Sequence[float],
    std: Sequence[float],
    epsilon: float,
    candidates: int,
    label_margin: float,
    seed: int,
) -> HardPositiveResult:
    _validate_inputs(
        images=images,
        labels=labels,
        attack_mask=attack_mask,
        sample_indices=sample_indices,
        epsilon=epsilon,
        steps=candidates,
        label_margin=label_margin,
    )
    original_rgb = normalized_to_rgb(images.detach(), mean=mean, std=std)
    mask = attack_mask.to(device=images.device, dtype=original_rgb.dtype)
    with torch.no_grad():
        clean_logits, clean_feature_values = forward_logits_features(images.detach())
        clean_features = _normalized_features(clean_feature_values).detach()
        clean_log_probability = F.log_softmax(clean_logits.float(), dim=1).gather(
            1, labels.long().view(-1, 1)
        ).squeeze(1).detach()

        batch_size = int(images.size(0))
        best_rgb = original_rgb.clone()
        best_distance = torch.zeros(
            batch_size, device=images.device, dtype=torch.float32
        )
        best_drop = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
        best_step = torch.zeros(batch_size, device=images.device, dtype=torch.long)
        for candidate_index in range(1, int(candidates) + 1):
            random_delta = _sample_noise(
                sample_indices=sample_indices,
                shape=original_rgb.shape[1:],
                seed=int(seed),
                draw=candidate_index,
                normal=False,
                device=original_rgb.device,
                dtype=original_rgb.dtype,
            )
            candidate_rgb = (
                original_rgb + float(epsilon) * random_delta * mask
            ).clamp(0.0, 1.0)
            _, candidate_distance, candidate_drop = _evaluate_candidate(
                candidate_rgb=candidate_rgb,
                labels=labels,
                clean_log_probability=clean_log_probability,
                clean_features=clean_features,
                forward_logits_features=forward_logits_features,
                mean=mean,
                std=std,
            )
            best_rgb, best_distance, best_drop, best_step = _update_best(
                best_rgb=best_rgb,
                best_distance=best_distance,
                best_drop=best_drop,
                best_step=best_step,
                candidate_rgb=candidate_rgb,
                candidate_distance=candidate_distance,
                candidate_drop=candidate_drop,
                step=candidate_index,
                label_margin=label_margin,
                require_feasible=True,
            )

    delta_rgb = (best_rgb - original_rgb) * mask
    return HardPositiveResult(
        augmented_images=rgb_to_normalized(
            original_rgb + delta_rgb,
            mean=mean,
            std=std,
        ).detach(),
        delta_rgb=delta_rgb.detach(),
        attack_mask=attack_mask.detach().to(dtype=torch.bool),
        feature_distance=best_distance.detach(),
        true_log_probability_drop=best_drop.detach(),
        constraint_satisfied=(
            best_drop <= float(label_margin) + 1e-7
        ).detach(),
        selected_step=best_step.detach(),
    )
