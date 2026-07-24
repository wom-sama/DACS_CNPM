from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_COUNT = 5
FEATURE_DIM = 256
FOCUS_CLASS = 1
ELIGIBLE_CLASSES = (0, 2, 4)
BASE_SEED = 20260724
COVNET_REPEAT_SEED_OFFSET = 100000
EPOCHS = 30
CLEAN_EPOCHS = 10
PARTITION_BATCH_SIZE = 32
META_SAMPLES_PER_CLASS = 6
META_BATCH_SIZE = CLASS_COUNT * META_SAMPLES_PER_CLASS
HEAD_LR = 0.03
HEAD_MOMENTUM = 0.9
COVNET_LR = 0.001
LAMBDA_0 = 5.0
META_INTERVAL = 10

ROLE_NAMES = (
    "ce_control",
    "restricted_classwise_isda",
    "rn_lisda_candidate",
    "rn_lisda_covnet_seed_repeat",
    "restricted_all_rival_meta",
    "full_all_class_meta",
    "rn_lisda_deranged_input",
    "rn_lisda_joint_no_meta",
)

META_ROLES = frozenset(
    {
        "rn_lisda_candidate",
        "rn_lisda_covnet_seed_repeat",
        "restricted_all_rival_meta",
        "full_all_class_meta",
        "rn_lisda_deranged_input",
    }
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(str(name).encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(np.asarray(values[name]))))
    return digest.hexdigest()


def string_sequence_sha256(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    sequence = [str(value) for value in values]
    digest.update(np.asarray([len(sequence)], dtype=np.int64).tobytes())
    for value in sequence:
        encoded = value.encode("utf-8")
        digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
        digest.update(encoded)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        2**63 - 1
    )


class CovariancePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = FEATURE_DIM // 2,
        output_dim: int = FEATURE_DIM,
        *,
        seed: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            hidden_bound = 1.0 / math.sqrt(float(self.hidden.in_features))
            self.hidden.weight.uniform_(
                -hidden_bound, hidden_bound, generator=generator
            )
            self.hidden.bias.uniform_(
                -hidden_bound, hidden_bound, generator=generator
            )
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, features: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(self.output(F.relu(self.hidden(features))))


@dataclass
class LinearHeadState:
    weight: Tensor
    bias: Tensor
    weight_momentum: Optional[Tensor] = None
    bias_momentum: Optional[Tensor] = None

    def clone_cpu(self) -> Dict[str, np.ndarray]:
        values = {
            "weight": self.weight.detach().cpu().numpy().copy(),
            "bias": self.bias.detach().cpu().numpy().copy(),
        }
        if self.weight_momentum is not None:
            values["weight_momentum"] = (
                self.weight_momentum.detach().cpu().numpy().copy()
            )
        if self.bias_momentum is not None:
            values["bias_momentum"] = (
                self.bias_momentum.detach().cpu().numpy().copy()
            )
        return values


def initialize_linear_head(
    *,
    feature_dim: int,
    class_count: int,
    seed: int,
    device: torch.device,
) -> LinearHeadState:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bound = 1.0 / math.sqrt(float(feature_dim))
    weight = torch.empty(class_count, feature_dim, dtype=torch.float32)
    bias = torch.empty(class_count, dtype=torch.float32)
    weight.uniform_(-bound, bound, generator=generator)
    bias.uniform_(-bound, bound, generator=generator)
    return LinearHeadState(
        weight=weight.to(device=device).requires_grad_(True),
        bias=bias.to(device=device).requires_grad_(True),
    )


def population_diagonal_variance(
    features: Tensor,
    labels: Tensor,
    *,
    class_count: int = CLASS_COUNT,
    floor: float = 1e-8,
) -> Tensor:
    rows: List[Tensor] = []
    for class_index in range(int(class_count)):
        selected = features[labels == int(class_index)]
        if selected.shape[0] < 2:
            raise ValueError(
                f"Class {class_index} has fewer than two partition rows"
            )
        rows.append(selected.var(dim=0, correction=0).clamp_min(float(floor)))
    return torch.stack(rows, dim=0)


def predict_covariance(
    predictor: CovariancePredictor,
    covariance_input: Tensor,
    labels: Tensor,
    base_variance: Tensor,
) -> Tuple[Tensor, Tensor]:
    scales = predictor(covariance_input.detach())
    covariance = base_variance.index_select(0, labels) * scales
    return covariance, scales


def isda_logits(
    *,
    features: Tensor,
    labels: Tensor,
    weight: Tensor,
    bias: Tensor,
    covariance: Tensor,
    lambda_strength: float,
    eligible_classes: Sequence[int] = ELIGIBLE_CLASSES,
    rival_mode: str = "focus",
) -> Tuple[Tensor, Tensor, Tensor]:
    clean_logits = F.linear(features, weight, bias)
    if covariance.shape != features.shape:
        raise ValueError(
            f"Covariance shape {tuple(covariance.shape)} "
            f"does not match feature shape {tuple(features.shape)}"
        )
    if rival_mode not in {"focus", "all"}:
        raise ValueError(f"Unsupported rival mode: {rival_mode}")

    eligible = torch.zeros_like(labels, dtype=torch.bool)
    for class_index in eligible_classes:
        eligible |= labels == int(class_index)
    true_weight = weight.index_select(0, labels)

    if rival_mode == "focus":
        difference = weight[FOCUS_CLASS].unsqueeze(0) - true_weight
        values = (
            0.5
            * float(lambda_strength)
            * (difference.square() * covariance).sum(dim=1)
            * eligible.to(dtype=features.dtype)
        )
        focus_basis = F.one_hot(
            torch.full_like(labels, FOCUS_CLASS), num_classes=weight.shape[0]
        ).to(dtype=features.dtype)
        delta = values.unsqueeze(1) * focus_basis
    else:
        difference = weight.unsqueeze(0) - true_weight.unsqueeze(1)
        values = (
            0.5
            * float(lambda_strength)
            * (difference.square() * covariance.unsqueeze(1)).sum(dim=2)
        )
        non_target = ~F.one_hot(
            labels, num_classes=weight.shape[0]
        ).to(dtype=torch.bool)
        delta = (
            values
            * non_target.to(dtype=features.dtype)
            * eligible.unsqueeze(1).to(dtype=features.dtype)
        )

    return clean_logits + delta, clean_logits, delta


def functional_sgd_parameters(
    state: LinearHeadState,
    weight_gradient: Tensor,
    bias_gradient: Tensor,
    *,
    learning_rate: float,
    momentum: float = HEAD_MOMENTUM,
) -> Tuple[Tensor, Tensor]:
    weight_direction = weight_gradient
    bias_direction = bias_gradient
    if state.weight_momentum is not None:
        weight_direction = (
            float(momentum) * state.weight_momentum.detach() + weight_gradient
        )
    if state.bias_momentum is not None:
        bias_direction = (
            float(momentum) * state.bias_momentum.detach() + bias_gradient
        )
    return (
        state.weight - float(learning_rate) * weight_direction,
        state.bias - float(learning_rate) * bias_direction,
    )


def real_sgd_step(
    state: LinearHeadState,
    weight_gradient: Tensor,
    bias_gradient: Tensor,
    *,
    learning_rate: float,
    momentum: float = HEAD_MOMENTUM,
) -> None:
    with torch.no_grad():
        if state.weight_momentum is None:
            state.weight_momentum = weight_gradient.detach().clone()
        else:
            state.weight_momentum.mul_(float(momentum)).add_(
                weight_gradient.detach()
            )
        if state.bias_momentum is None:
            state.bias_momentum = bias_gradient.detach().clone()
        else:
            state.bias_momentum.mul_(float(momentum)).add_(
                bias_gradient.detach()
            )
        state.weight.add_(
            state.weight_momentum, alpha=-float(learning_rate)
        )
        state.bias.add_(state.bias_momentum, alpha=-float(learning_rate))


def paper_learning_rate(
    *,
    initial_lr: float,
    epoch: int,
    step: int,
    steps_per_epoch: int,
    total_epochs: int = EPOCHS,
    warmup_epochs: int = 5,
) -> float:
    if epoch < warmup_epochs:
        current = 1 + int(epoch) * int(steps_per_epoch) + int(step)
        total = 1 + int(warmup_epochs) * int(steps_per_epoch)
        return float(initial_lr) * float(current) / float(total)
    current = (
        (int(epoch) - int(warmup_epochs)) * int(steps_per_epoch) + int(step)
    )
    total = (int(total_epochs) - int(warmup_epochs)) * int(steps_per_epoch)
    return 0.5 * float(initial_lr) * (
        1.0 + math.cos(math.pi * float(current) / float(total))
    )


def semantic_lambda(epoch: int) -> float:
    if int(epoch) < CLEAN_EPOCHS:
        return 0.0
    return float(LAMBDA_0) * float(int(epoch) - CLEAN_EPOCHS + 1) / float(
        EPOCHS - CLEAN_EPOCHS
    )


def deterministic_partition_order(
    indices: np.ndarray,
    *,
    outer_fold: int,
    epoch: int,
    partition_name: str,
) -> np.ndarray:
    generator = np.random.default_rng(
        stable_seed(
            "rn_lisda_batch_order",
            BASE_SEED,
            int(outer_fold),
            int(epoch),
            str(partition_name),
        )
    )
    return np.asarray(indices, dtype=np.int64)[generator.permutation(len(indices))]


def balanced_meta_slot_count(steps_per_epoch: int) -> int:
    semantic_iterations = int(steps_per_epoch) * (EPOCHS - CLEAN_EPOCHS)
    return (semantic_iterations + META_INTERVAL - 1) // META_INTERVAL


def build_balanced_meta_schedule(
    indices: np.ndarray,
    labels: np.ndarray,
    source_stems: np.ndarray,
    *,
    outer_fold: int,
    partition_name: str,
    slot_count: int,
    samples_per_class: int = META_SAMPLES_PER_CLASS,
) -> np.ndarray:
    partition = np.asarray(indices, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    source_array = np.asarray(source_stems).astype(str)
    if source_array.shape[0] != label_array.shape[0]:
        raise ValueError("source_stems and labels must have equal length")
    if int(slot_count) <= 0:
        raise ValueError("Balanced meta schedule requires at least one slot")
    if int(samples_per_class) <= 0:
        raise ValueError("samples_per_class must be positive")

    buckets: Dict[int, np.ndarray] = {}
    for class_index in range(CLASS_COUNT):
        members = partition[label_array[partition] == class_index]
        if len(members) < int(samples_per_class):
            raise ValueError(
                f"Class {class_index} in partition {partition_name} has only "
                f"{len(members)} rows for a {samples_per_class}-row meta draw"
            )
        buckets[class_index] = members

    usage_count = np.zeros(label_array.shape[0], dtype=np.int64)

    schedule = np.empty(
        (int(slot_count), CLASS_COUNT * int(samples_per_class)),
        dtype=np.int64,
    )
    for slot in range(int(slot_count)):
        selected_by_class: Dict[int, List[int]] = {}
        used_sources = set()
        class_priority = [
            (slot + offset) % CLASS_COUNT for offset in range(CLASS_COUNT)
        ]
        for class_index in class_priority:
            members = buckets[class_index]
            generator = np.random.default_rng(
                stable_seed(
                    "rn_lisda_balanced_meta_source_unique",
                    BASE_SEED,
                    int(outer_fold),
                    str(partition_name),
                    int(slot),
                    int(class_index),
                )
            )
            permuted = members[generator.permutation(len(members))]
            ranked = permuted[
                np.argsort(usage_count[permuted], kind="stable")
            ]
            class_selected = []
            for candidate in ranked:
                candidate = int(candidate)
                source = str(source_array[candidate])
                if source in used_sources:
                    continue
                class_selected.append(int(candidate))
                used_sources.add(source)
                if len(class_selected) == int(samples_per_class):
                    break
            if len(class_selected) != int(samples_per_class):
                raise ValueError(
                    "Cannot build source-unique balanced meta batch for "
                    f"fold={outer_fold} partition={partition_name} "
                    f"slot={slot} class={class_index}"
                )
            selected_by_class[class_index] = class_selected
            usage_count[np.asarray(class_selected, dtype=np.int64)] += 1
        selected = [
            index
            for class_index in range(CLASS_COUNT)
            for index in selected_by_class[class_index]
        ]
        schedule[slot] = np.asarray(selected, dtype=np.int64)

    expected = np.full(CLASS_COUNT, int(samples_per_class), dtype=np.int64)
    for row in schedule:
        if len(np.unique(row)) != len(row):
            raise ValueError("Balanced meta batch contains duplicate row indices")
        if len(np.unique(source_array[row])) != len(row):
            raise ValueError("Balanced meta batch contains duplicate sources")
        counts = np.bincount(label_array[row], minlength=CLASS_COUNT)
        if not np.array_equal(counts, expected):
            raise ValueError(
                f"Balanced meta batch has class counts {counts.tolist()}"
            )
    if balanced_meta_max_usage_spread(schedule, partition, label_array) > 1:
        raise ValueError("Balanced meta row-use spread exceeds one")
    return schedule


def balanced_meta_max_usage_spread(
    schedule: np.ndarray,
    partition_indices: np.ndarray,
    labels: np.ndarray,
) -> int:
    rows = np.asarray(schedule, dtype=np.int64)
    partition = np.asarray(partition_indices, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    usage = np.bincount(
        rows.reshape(-1),
        minlength=label_array.shape[0],
    )
    spreads = []
    for class_index in range(CLASS_COUNT):
        members = partition[label_array[partition] == class_index]
        if not len(members):
            raise ValueError(f"Meta partition is missing class {class_index}")
        class_usage = usage[members]
        spreads.append(int(class_usage.max() - class_usage.min()))
    return max(spreads)


def build_same_label_source_derangement(
    indices: np.ndarray,
    labels: np.ndarray,
    source_stems: np.ndarray,
    *,
    outer_fold: int,
    partition_name: str,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(source_stems).astype(str)
    mapping = np.full(labels.shape[0], -1, dtype=np.int64)
    for class_index in range(CLASS_COUNT):
        members = indices[labels[indices] == class_index]
        ranked = sorted(
            members.tolist(),
            key=lambda index: hashlib.sha256(
                (
                    f"rn_lisda_derangement|{BASE_SEED}|{outer_fold}|"
                    f"{partition_name}|{class_index}|{index}|{sources[index]}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < 2:
            raise ValueError(
                f"Class {class_index} cannot be deranged in {partition_name}"
            )
        ranked_array = np.asarray(ranked, dtype=np.int64)
        ranked_sources = sources[ranked_array]
        selected: Optional[np.ndarray] = None
        for shift in range(1, len(ranked)):
            candidate = np.roll(ranked_array, -shift)
            if np.all(ranked_sources != sources[candidate]):
                selected = candidate
                break
        if selected is None:
            raise ValueError(
                f"No source-disjoint cyclic derangement for class "
                f"{class_index} in {partition_name}"
            )
        mapping[ranked_array] = selected

    if np.any(mapping[indices] < 0):
        raise ValueError("Derangement is incomplete")
    if not np.array_equal(labels[indices], labels[mapping[indices]]):
        raise ValueError("Derangement changes labels")
    if np.any(sources[indices] == sources[mapping[indices]]):
        raise ValueError("Derangement reuses a source stem")
    if len(np.unique(mapping[indices])) != len(indices):
        raise ValueError("Derangement does not preserve the partition marginal")
    return mapping


def build_outer_holdout_swap_donors(
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    labels: np.ndarray,
    source_stems: np.ndarray,
    *,
    outer_fold: int,
) -> np.ndarray:
    fit = np.asarray(fit_indices, dtype=np.int64)
    held = np.asarray(holdout_indices, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(source_stems).astype(str)
    ranked_by_class: Dict[int, np.ndarray] = {}
    for class_index in range(CLASS_COUNT):
        members = fit[label_array[fit] == class_index]
        if len(members) == 0:
            raise ValueError(
                f"Outer-fit class {class_index} has no swap donor rows"
            )
        ranked_by_class[class_index] = np.asarray(
            sorted(
                members.tolist(),
                key=lambda index: hashlib.sha256(
                    (
                        "rn_lisda_holdout_donor_rank|"
                        f"{BASE_SEED}|{outer_fold}|{class_index}|"
                        f"{index}|{sources[index]}"
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            dtype=np.int64,
        )

    donors = np.asarray(
        [
            ranked_by_class[int(label_array[index])][
                stable_seed(
                    "rn_lisda_holdout_swap",
                    BASE_SEED,
                    int(outer_fold),
                    int(index),
                )
                % len(ranked_by_class[int(label_array[index])])
            ]
            for index in held
        ],
        dtype=np.int64,
    )
    if not np.array_equal(label_array[held], label_array[donors]):
        raise ValueError("Outer-holdout swap donor changes labels")
    if np.any(sources[held] == sources[donors]):
        raise ValueError("Outer-holdout swap donor reuses the anchor source")
    if np.any(~np.isin(donors, fit)):
        raise ValueError("Outer-holdout swap donor is outside outer fit")
    return donors


def draw_semantic_feature(
    feature: np.ndarray,
    covariance: np.ndarray,
    *,
    outer_fold: int,
    sample_index: int,
    role: str,
    draw_index: int,
    lambda_strength: float = LAMBDA_0,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_array = np.asarray(feature, dtype=np.float32)
    covariance_array = np.asarray(covariance, dtype=np.float32)
    if feature_array.shape != covariance_array.shape:
        raise ValueError("Feature and covariance shapes must match")
    if np.any(covariance_array < 0.0) or not np.all(
        np.isfinite(covariance_array)
    ):
        raise ValueError("Covariance must be finite and non-negative")
    generator = np.random.default_rng(
        stable_seed(
            "rn_lisda_visual",
            BASE_SEED,
            int(outer_fold),
            int(sample_index),
            str(role),
            int(draw_index),
        )
    )
    epsilon = generator.standard_normal(feature_array.shape).astype(np.float32)
    child = feature_array + np.sqrt(
        np.float32(lambda_strength) * covariance_array
    ).astype(np.float32) * epsilon
    return child.astype(np.float32, copy=False), epsilon


def cosine_nearest_proxy(
    query: np.ndarray,
    features: np.ndarray,
    pool_indices: np.ndarray,
    source_stems: np.ndarray,
    *,
    excluded_source: str,
) -> Tuple[int, float]:
    matrix = np.asarray(features, dtype=np.float32)
    pool = np.asarray(pool_indices, dtype=np.int64)
    sources = np.asarray(source_stems).astype(str)
    pool = pool[sources[pool] != str(excluded_source)]
    if len(pool) == 0:
        raise ValueError("No source-disjoint proxy candidate")
    query_array = np.asarray(query, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_array))
    row_norms = np.linalg.norm(matrix[pool], axis=1)
    valid = row_norms > 0.0
    if query_norm <= 0.0 or not np.any(valid):
        raise ValueError("Cosine proxy search requires non-zero embeddings")
    pool = pool[valid]
    normalized_query = query_array / np.float32(query_norm)
    similarities = (
        matrix[pool] / row_norms[valid, None].astype(np.float32)
    ) @ normalized_query
    distances = 1.0 - similarities.astype(np.float64)
    position = int(np.lexsort((pool, distances))[0])
    return int(pool[position]), float(distances[position])


def _gradient_norm(gradients: Iterable[Tensor]) -> float:
    total = 0.0
    for gradient in gradients:
        total += float(gradient.detach().double().square().sum().item())
    return math.sqrt(total)


def meta_covnet_step(
    *,
    predictor: CovariancePredictor,
    head: LinearHeadState,
    train_features: Tensor,
    train_labels: Tensor,
    meta_features: Tensor,
    meta_labels: Tensor,
    covariance_input: Tensor,
    base_variance: Tensor,
    head_learning_rate: float,
    covnet_learning_rate: float,
    lambda_strength: float,
    eligible_classes: Sequence[int],
    rival_mode: str,
) -> Dict[str, float]:
    covariance, _ = predict_covariance(
        predictor, covariance_input, train_labels, base_variance
    )
    augmented, _, _ = isda_logits(
        features=train_features,
        labels=train_labels,
        weight=head.weight,
        bias=head.bias,
        covariance=covariance,
        lambda_strength=lambda_strength,
        eligible_classes=eligible_classes,
        rival_mode=rival_mode,
    )
    pseudo_loss = F.cross_entropy(augmented, train_labels)
    weight_gradient, bias_gradient = torch.autograd.grad(
        pseudo_loss,
        (head.weight, head.bias),
        create_graph=True,
    )
    pseudo_weight, pseudo_bias = functional_sgd_parameters(
        head,
        weight_gradient,
        bias_gradient,
        learning_rate=head_learning_rate,
    )
    meta_logits = F.linear(meta_features, pseudo_weight, pseudo_bias)
    meta_loss = F.cross_entropy(meta_logits, meta_labels)
    parameters = tuple(predictor.parameters())
    gradients = torch.autograd.grad(
        meta_loss,
        parameters,
        allow_unused=False,
    )
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(gradient, alpha=-float(covnet_learning_rate))
    return {
        "pseudo_loss": float(pseudo_loss.detach().item()),
        "meta_loss": float(meta_loss.detach().item()),
        "hidden_gradient_norm": _gradient_norm(gradients[:2]),
        "output_gradient_norm": _gradient_norm(gradients[2:]),
    }


def real_head_step(
    *,
    role: str,
    predictor: Optional[CovariancePredictor],
    head: LinearHeadState,
    features: Tensor,
    labels: Tensor,
    covariance_input: Tensor,
    base_variance: Tensor,
    learning_rate: float,
    covnet_learning_rate: float,
    lambda_strength: float,
) -> Dict[str, float]:
    if role == "ce_control" or float(lambda_strength) == 0.0:
        logits = F.linear(features, head.weight, head.bias)
        loss = F.cross_entropy(logits, labels)
        weight_gradient, bias_gradient = torch.autograd.grad(
            loss, (head.weight, head.bias)
        )
        real_sgd_step(
            head,
            weight_gradient,
            bias_gradient,
            learning_rate=learning_rate,
        )
        return {
            "loss": float(loss.detach().item()),
            "head_gradient_norm": _gradient_norm(
                (weight_gradient, bias_gradient)
            ),
            "covnet_gradient_norm": 0.0,
        }

    rival_mode = (
        "all"
        if role in {"restricted_all_rival_meta", "full_all_class_meta"}
        else "focus"
    )
    eligible_classes: Sequence[int] = (
        tuple(range(CLASS_COUNT))
        if role == "full_all_class_meta"
        else ELIGIBLE_CLASSES
    )
    if role == "restricted_classwise_isda":
        covariance = base_variance.index_select(0, labels)
    else:
        if predictor is None:
            raise ValueError(f"Role {role} requires a covariance predictor")
        covariance, _ = predict_covariance(
            predictor, covariance_input, labels, base_variance
        )
        if role != "rn_lisda_joint_no_meta":
            covariance = covariance.detach()

    augmented, _, _ = isda_logits(
        features=features,
        labels=labels,
        weight=head.weight,
        bias=head.bias,
        covariance=covariance,
        lambda_strength=lambda_strength,
        eligible_classes=eligible_classes,
        rival_mode=rival_mode,
    )
    loss = F.cross_entropy(augmented, labels)

    if role == "rn_lisda_joint_no_meta":
        assert predictor is not None
        parameters = (head.weight, head.bias, *tuple(predictor.parameters()))
        gradients = torch.autograd.grad(loss, parameters)
        weight_gradient, bias_gradient = gradients[:2]
        covnet_gradients = gradients[2:]
        real_sgd_step(
            head,
            weight_gradient,
            bias_gradient,
            learning_rate=learning_rate,
        )
        with torch.no_grad():
            for parameter, gradient in zip(
                predictor.parameters(), covnet_gradients
            ):
                parameter.add_(
                    gradient, alpha=-float(covnet_learning_rate)
                )
        covnet_gradient_norm = _gradient_norm(covnet_gradients)
    else:
        weight_gradient, bias_gradient = torch.autograd.grad(
            loss, (head.weight, head.bias)
        )
        real_sgd_step(
            head,
            weight_gradient,
            bias_gradient,
            learning_rate=learning_rate,
        )
        covnet_gradient_norm = 0.0

    return {
        "loss": float(loss.detach().item()),
        "head_gradient_norm": _gradient_norm(
            (weight_gradient, bias_gradient)
        ),
        "covnet_gradient_norm": covnet_gradient_norm,
    }


def role_settings(role: str) -> Tuple[Sequence[int], str]:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role: {role}")
    eligible: Sequence[int] = (
        tuple(range(CLASS_COUNT))
        if role == "full_all_class_meta"
        else ELIGIBLE_CLASSES
    )
    rival_mode = (
        "all"
        if role in {"restricted_all_rival_meta", "full_all_class_meta"}
        else "focus"
    )
    return eligible, rival_mode


def score_head(
    head: LinearHeadState,
    features: Tensor,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    rows: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            logits = F.linear(
                features[start : start + int(batch_size)],
                head.weight,
                head.bias,
            )
            rows.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(rows, dim=0).numpy()


def covariance_state_arrays(
    predictor: Optional[CovariancePredictor],
) -> Dict[str, np.ndarray]:
    if predictor is None:
        return {}
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in predictor.state_dict().items()
    }


def train_role_fold(
    *,
    role: str,
    features: Tensor,
    labels: Tensor,
    source_stems: np.ndarray,
    outer_fold: int,
    partition_a: np.ndarray,
    partition_b: np.ndarray,
    holdout: np.ndarray,
) -> Dict[str, object]:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role: {role}")
    device = features.device
    covnet_repeat = role == "rn_lisda_covnet_seed_repeat"
    head = initialize_linear_head(
        feature_dim=features.shape[1],
        class_count=CLASS_COUNT,
        seed=stable_seed("rn_lisda_head", BASE_SEED, int(outer_fold)),
        device=device,
    )
    predictor: Optional[CovariancePredictor] = None
    if role in META_ROLES or role == "rn_lisda_joint_no_meta":
        predictor = CovariancePredictor(
            input_dim=features.shape[1],
            hidden_dim=features.shape[1] // 2,
            output_dim=features.shape[1],
            seed=stable_seed(
                "rn_lisda_covnet",
                BASE_SEED
                + (COVNET_REPEAT_SEED_OFFSET if covnet_repeat else 0),
                int(outer_fold),
            ),
        ).to(device=device)

    a = np.asarray(partition_a, dtype=np.int64)
    b = np.asarray(partition_b, dtype=np.int64)
    held = np.asarray(holdout, dtype=np.int64)
    combined = np.concatenate([a, b, held])
    if (
        len(np.unique(a)) != len(a)
        or len(np.unique(b)) != len(b)
        or len(np.unique(held)) != len(held)
        or len(np.unique(combined)) != len(combined)
        or not np.array_equal(
            np.sort(combined), np.arange(features.shape[0], dtype=np.int64)
        )
    ):
        raise ValueError(
            "Inner partitions and outer holdout must be disjoint and complete"
        )
    initial_head_sha256 = state_arrays_sha256(head.clone_cpu())
    initial_covnet_sha256 = state_arrays_sha256(
        covariance_state_arrays(predictor)
    )
    labels_numpy = labels.detach().cpu().numpy()
    base_a = population_diagonal_variance(
        features[torch.as_tensor(a, device=device)],
        labels[torch.as_tensor(a, device=device)],
    )
    base_b = population_diagonal_variance(
        features[torch.as_tensor(b, device=device)],
        labels[torch.as_tensor(b, device=device)],
    )
    deranged_a = build_same_label_source_derangement(
        a,
        labels_numpy,
        source_stems,
        outer_fold=outer_fold,
        partition_name="a",
    )
    deranged_b = build_same_label_source_derangement(
        b,
        labels_numpy,
        source_stems,
        outer_fold=outer_fold,
        partition_name="b",
    )
    steps = min(len(a) // PARTITION_BATCH_SIZE, len(b) // PARTITION_BATCH_SIZE)
    if steps <= 0:
        raise ValueError("Partition does not contain one paired batch")
    meta_slots = balanced_meta_slot_count(steps)
    balanced_meta_a = build_balanced_meta_schedule(
        a,
        labels_numpy,
        source_stems,
        outer_fold=outer_fold,
        partition_name="a",
        slot_count=meta_slots,
    )
    balanced_meta_b = build_balanced_meta_schedule(
        b,
        labels_numpy,
        source_stems,
        outer_fold=outer_fold,
        partition_name="b",
        slot_count=meta_slots,
    )

    trace: List[Dict[str, object]] = []
    meta_trace: List[Dict[str, object]] = []
    semantic_iteration = 0
    balanced_meta_slot = 0
    head_update_count = 0
    meta_update_count = 0
    for epoch in range(EPOCHS):
        order_a = deterministic_partition_order(
            a,
            outer_fold=outer_fold,
            epoch=epoch,
            partition_name="a",
        )
        order_b = deterministic_partition_order(
            b,
            outer_fold=outer_fold,
            epoch=epoch,
            partition_name="b",
        )
        consumed_a = order_a[: steps * PARTITION_BATCH_SIZE]
        consumed_b = order_b[: steps * PARTITION_BATCH_SIZE]
        epoch_losses: List[float] = []
        epoch_meta_losses: List[float] = []
        epoch_head_gradients: List[float] = []
        epoch_covnet_gradients: List[float] = []
        for step in range(steps):
            a_numpy = order_a[
                step
                * PARTITION_BATCH_SIZE : (step + 1)
                * PARTITION_BATCH_SIZE
            ]
            b_numpy = order_b[
                step
                * PARTITION_BATCH_SIZE : (step + 1)
                * PARTITION_BATCH_SIZE
            ]
            a_index = torch.as_tensor(a_numpy, device=device)
            b_index = torch.as_tensor(b_numpy, device=device)
            x_a, y_a = features[a_index], labels[a_index]
            x_b, y_b = features[b_index], labels[b_index]
            head_lr = paper_learning_rate(
                initial_lr=HEAD_LR,
                epoch=epoch,
                step=step,
                steps_per_epoch=steps,
            )
            covnet_lr = paper_learning_rate(
                initial_lr=COVNET_LR,
                epoch=epoch,
                step=step,
                steps_per_epoch=steps,
            )
            strength = semantic_lambda(epoch)
            scheduled_meta = (
                role in META_ROLES
                and epoch >= CLEAN_EPOCHS
                and semantic_iteration % META_INTERVAL == 0
            )

            covariance_input_a = x_a
            covariance_input_b = x_b
            if role == "rn_lisda_deranged_input":
                covariance_input_a = features[
                    torch.as_tensor(deranged_a[a_numpy], device=device)
                ]
                covariance_input_b = features[
                    torch.as_tensor(deranged_b[b_numpy], device=device)
                ]

            eligible, rival_mode = role_settings(role)
            if scheduled_meta:
                assert predictor is not None
                meta_b_numpy = balanced_meta_b[balanced_meta_slot]
                meta_b_index = torch.as_tensor(meta_b_numpy, device=device)
                meta_x_b = features[meta_b_index]
                meta_y_b = labels[meta_b_index]
                source_overlap_a_to_b = len(
                    set(source_stems[a_numpy].tolist())
                    & set(source_stems[meta_b_numpy].tolist())
                )
                if source_overlap_a_to_b:
                    raise ValueError(
                        "A-to-B pseudo/meta sources are not disjoint"
                    )
                telemetry_a = meta_covnet_step(
                    predictor=predictor,
                    head=head,
                    train_features=x_a,
                    train_labels=y_a,
                    meta_features=meta_x_b,
                    meta_labels=meta_y_b,
                    covariance_input=covariance_input_a,
                    base_variance=base_a,
                    head_learning_rate=head_lr,
                    covnet_learning_rate=covnet_lr,
                    lambda_strength=strength,
                    eligible_classes=eligible,
                    rival_mode=rival_mode,
                )
                meta_trace.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "semantic_iteration": semantic_iteration,
                        "direction": "a_to_b",
                        "balanced_meta_slot": balanced_meta_slot,
                        "meta_indices_sha256": array_sha256(meta_b_numpy),
                        "meta_unique_row_count": int(
                            len(np.unique(meta_b_numpy))
                        ),
                        "meta_unique_source_count": int(
                            len(np.unique(source_stems[meta_b_numpy]))
                        ),
                        "meta_class_counts": np.bincount(
                            labels_numpy[meta_b_numpy],
                            minlength=CLASS_COUNT,
                        ).tolist(),
                        "train_meta_source_overlap_count": (
                            source_overlap_a_to_b
                        ),
                        **telemetry_a,
                    }
                )
                epoch_meta_losses.append(telemetry_a["meta_loss"])
                meta_update_count += 1

            update_a = real_head_step(
                role=role,
                predictor=predictor,
                head=head,
                features=x_a,
                labels=y_a,
                covariance_input=covariance_input_a,
                base_variance=base_a,
                learning_rate=head_lr,
                covnet_learning_rate=covnet_lr,
                lambda_strength=strength,
            )
            head_update_count += 1
            epoch_losses.append(update_a["loss"])
            epoch_head_gradients.append(update_a["head_gradient_norm"])
            epoch_covnet_gradients.append(update_a["covnet_gradient_norm"])

            if scheduled_meta:
                assert predictor is not None
                meta_a_numpy = balanced_meta_a[balanced_meta_slot]
                meta_a_index = torch.as_tensor(meta_a_numpy, device=device)
                meta_x_a = features[meta_a_index]
                meta_y_a = labels[meta_a_index]
                source_overlap_b_to_a = len(
                    set(source_stems[b_numpy].tolist())
                    & set(source_stems[meta_a_numpy].tolist())
                )
                if source_overlap_b_to_a:
                    raise ValueError(
                        "B-to-A pseudo/meta sources are not disjoint"
                    )
                telemetry_b = meta_covnet_step(
                    predictor=predictor,
                    head=head,
                    train_features=x_b,
                    train_labels=y_b,
                    meta_features=meta_x_a,
                    meta_labels=meta_y_a,
                    covariance_input=covariance_input_b,
                    base_variance=base_b,
                    head_learning_rate=head_lr,
                    covnet_learning_rate=covnet_lr,
                    lambda_strength=strength,
                    eligible_classes=eligible,
                    rival_mode=rival_mode,
                )
                meta_trace.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "semantic_iteration": semantic_iteration,
                        "direction": "b_to_a",
                        "balanced_meta_slot": balanced_meta_slot,
                        "meta_indices_sha256": array_sha256(meta_a_numpy),
                        "meta_unique_row_count": int(
                            len(np.unique(meta_a_numpy))
                        ),
                        "meta_unique_source_count": int(
                            len(np.unique(source_stems[meta_a_numpy]))
                        ),
                        "meta_class_counts": np.bincount(
                            labels_numpy[meta_a_numpy],
                            minlength=CLASS_COUNT,
                        ).tolist(),
                        "train_meta_source_overlap_count": (
                            source_overlap_b_to_a
                        ),
                        **telemetry_b,
                    }
                )
                epoch_meta_losses.append(telemetry_b["meta_loss"])
                meta_update_count += 1
                balanced_meta_slot += 1

            update_b = real_head_step(
                role=role,
                predictor=predictor,
                head=head,
                features=x_b,
                labels=y_b,
                covariance_input=covariance_input_b,
                base_variance=base_b,
                learning_rate=head_lr,
                covnet_learning_rate=covnet_lr,
                lambda_strength=strength,
            )
            head_update_count += 1
            epoch_losses.append(update_b["loss"])
            epoch_head_gradients.append(update_b["head_gradient_norm"])
            epoch_covnet_gradients.append(update_b["covnet_gradient_norm"])
            if epoch >= CLEAN_EPOCHS:
                semantic_iteration += 1

        trace.append(
            {
                "role": role,
                "outer_fold": int(outer_fold),
                "epoch": int(epoch),
                "steps": int(steps),
                "head_learning_rate_last": float(head_lr),
                "covnet_learning_rate_last": float(covnet_lr),
                "lambda": float(strength),
                "partition_a_consumed_indices_sha256": array_sha256(
                    consumed_a
                ),
                "partition_b_consumed_indices_sha256": array_sha256(
                    consumed_b
                ),
                "mean_loss": float(np.mean(epoch_losses)),
                "mean_meta_loss": (
                    float(np.mean(epoch_meta_losses))
                    if epoch_meta_losses
                    else 0.0
                ),
                "mean_head_gradient_norm": float(
                    np.mean(epoch_head_gradients)
                ),
                "mean_direct_covnet_gradient_norm": float(
                    np.mean(epoch_covnet_gradients)
                ),
            }
        )

    expected_meta_slots = meta_slots if role in META_ROLES else 0
    if balanced_meta_slot != expected_meta_slots:
        raise RuntimeError(
            "Balanced meta schedule consumption mismatch: "
            f"expected={expected_meta_slots} actual={balanced_meta_slot}"
        )
    held_tensor = torch.as_tensor(held, device=device)
    probabilities = score_head(head, features[held_tensor])
    return {
        "role": role,
        "outer_fold": int(outer_fold),
        "holdout_indices": held,
        "probabilities": probabilities,
        "head_state": head.clone_cpu(),
        "covnet_state": covariance_state_arrays(predictor),
        "trace": trace,
        "meta_trace": meta_trace,
        "head_update_count": int(head_update_count),
        "meta_update_count": int(meta_update_count),
        "paired_steps": int(steps),
        "balanced_meta_slot_count": int(balanced_meta_slot),
        "balanced_meta_schedule_a_sha256": array_sha256(balanced_meta_a),
        "balanced_meta_schedule_b_sha256": array_sha256(balanced_meta_b),
        "balanced_meta_schedule_a_max_class_usage_spread": (
            balanced_meta_max_usage_spread(
                balanced_meta_a,
                a,
                labels_numpy,
            )
        ),
        "balanced_meta_schedule_b_max_class_usage_spread": (
            balanced_meta_max_usage_spread(
                balanced_meta_b,
                b,
                labels_numpy,
            )
        ),
        "initial_head_sha256": initial_head_sha256,
        "initial_covnet_sha256": initial_covnet_sha256,
        "partition_a_indices_sha256": array_sha256(a),
        "partition_b_indices_sha256": array_sha256(b),
        "holdout_indices_sha256": array_sha256(held),
        "holdout_score_calls": 1,
        "holdout_score_epoch": EPOCHS,
        "partition_a_base_variance": base_a.detach().cpu().numpy(),
        "partition_b_base_variance": base_b.detach().cpu().numpy(),
        "derangement_a": deranged_a[a],
        "derangement_b": deranged_b[b],
    }
