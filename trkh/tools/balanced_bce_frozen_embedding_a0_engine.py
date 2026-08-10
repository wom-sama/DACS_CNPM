from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_COUNT = 5
FEATURE_DIM = 256
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
BASE_SEED = 20260724
REPEAT_SEED_OFFSET = 100000
EPOCHS = 30
BATCH_SIZE = 64
BASE_LR = 0.03
MOMENTUM = 0.9
WARMUP_EPOCHS = 5
TRACE_EPOCHS = frozenset({1, 5, 10, 20, 30})

ROLE_NAMES = (
    "ce_control",
    "plain_bce_control",
    "balanced_bce_candidate",
    "balanced_bce_seed_repeat",
    "balanced_softmax_tau025_control",
    "reversed_prior_bce_control",
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(str(name).encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(np.asarray(values[name]))))
    return digest.hexdigest()


def initialize_head(*, seed: int, device: torch.device) -> nn.Linear:
    # The lock builder used the stock torch.nn.Linear reset sequence on CPU.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = nn.Linear(FEATURE_DIM, CLASS_COUNT, bias=True)
    return head.to(device=device, dtype=torch.float32)


def initial_head_evidence(*, seed: int) -> Dict[str, object]:
    head = initialize_head(seed=seed, device=torch.device("cpu"))
    weight = head.weight.detach().numpy().astype(np.float32, copy=True)
    bias = head.bias.detach().numpy().astype(np.float32, copy=True)
    evidence: Dict[str, object] = {
        "seed": int(seed),
        "weight_sha256": array_sha256(weight),
        "bias_sha256": array_sha256(bias),
    }
    evidence["combined_sha256"] = json_sha256(evidence)
    return evidence


def build_epoch_orders(
    fit_indices: np.ndarray,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> List[np.ndarray]:
    indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Fit indices must be a non-empty unique sequence")
    rng = np.random.default_rng(int(seed))
    return [
        indices[rng.permutation(indices.size)].astype(np.int64, copy=False)
        for _ in range(int(epochs))
    ]


def learning_rate(epoch_index: int) -> float:
    epoch = int(epoch_index)
    if epoch < 0 or epoch >= EPOCHS:
        raise ValueError(f"Epoch index must be in [0, {EPOCHS}): {epoch}")
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    cosine_steps = EPOCHS - WARMUP_EPOCHS - 1
    progress = float(epoch - WARMUP_EPOCHS) / float(cosine_steps)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def balanced_bce_bias(prior: Tensor) -> Tensor:
    if prior.ndim != 1 or int(prior.numel()) != CLASS_COUNT:
        raise ValueError("Prior must contain exactly five classes")
    if bool(torch.any((prior <= 0.0) | (prior >= 1.0))):
        raise ValueError("Prior values must lie strictly between zero and one")
    return torch.log(prior) - torch.log1p(-prior)


def role_loss(
    role: str,
    raw_logits: Tensor,
    targets: Tensor,
    *,
    prior: Tensor,
) -> Tensor:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown Bal-BCE A0 role: {role}")
    if raw_logits.ndim != 2 or raw_logits.shape[1] != CLASS_COUNT:
        raise ValueError("Raw logits must have shape [N, 5]")
    hard_targets = targets.to(dtype=torch.long)
    if role == "ce_control":
        return F.cross_entropy(raw_logits, hard_targets)
    if role == "balanced_softmax_tau025_control":
        correction = 0.25 * torch.log(prior)
        return F.cross_entropy(raw_logits + correction, hard_targets)
    one_hot = F.one_hot(
        hard_targets, num_classes=CLASS_COUNT
    ).to(dtype=raw_logits.dtype)
    correction = torch.zeros_like(prior)
    if role in {"balanced_bce_candidate", "balanced_bce_seed_repeat"}:
        correction = balanced_bce_bias(prior)
    elif role == "reversed_prior_bce_control":
        correction = -balanced_bce_bias(prior)
    return CLASS_COUNT * F.binary_cross_entropy_with_logits(
        raw_logits + correction,
        one_hot,
        reduction="mean",
    )


def equation_oracles() -> Dict[str, object]:
    logits_np = np.asarray(
        [
            [-4.0, -1.5, 0.0, 2.5, 7.0],
            [12.0, -12.0, 1.0, -2.0, 0.25],
            [-80.0, 80.0, -40.0, 40.0, 0.0],
        ],
        dtype=np.float64,
    )
    targets_np = np.asarray([4, 0, 1], dtype=np.int64)
    prior_np = np.asarray([0.21, 0.058, 0.208, 0.273, 0.251], dtype=np.float64)
    logits = torch.tensor(logits_np, dtype=torch.float64)
    targets = torch.tensor(targets_np, dtype=torch.long)
    prior = torch.tensor(prior_np, dtype=torch.float64)

    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    manual_ce = -np.mean(
        shifted[np.arange(targets_np.size), targets_np]
        - np.log(np.exp(shifted).sum(axis=1))
    )
    torch_ce = float(F.cross_entropy(logits, targets))

    one_hot_np = np.eye(CLASS_COUNT, dtype=np.float64)[targets_np]
    manual_bce_elements = (
        np.maximum(logits_np, 0.0)
        - logits_np * one_hot_np
        + np.log1p(np.exp(-np.abs(logits_np)))
    )
    manual_bce = float(manual_bce_elements.sum(axis=1).mean())
    torch_bce = float(
        role_loss("plain_bce_control", logits, targets, prior=prior)
    )

    expected_bias = np.log(prior_np) - np.log1p(-prior_np)
    observed_bias = balanced_bce_bias(prior).detach().numpy()
    candidate = role_loss(
        "balanced_bce_candidate", logits, targets, prior=prior
    )
    reversed_loss = role_loss(
        "reversed_prior_bce_control", logits, targets, prior=prior
    )
    balanced_softmax = role_loss(
        "balanced_softmax_tau025_control", logits, targets, prior=prior
    )
    checks = {
        "manual_ce_matches_torch": abs(manual_ce - torch_ce) <= 1e-12,
        "manual_bce_matches_torch": abs(manual_bce - torch_bce) <= 1e-12,
        "balanced_bias_exact": bool(
            np.allclose(expected_bias, observed_bias, atol=1e-12, rtol=0.0)
        ),
        "candidate_and_reverse_finite": bool(
            torch.isfinite(candidate) and torch.isfinite(reversed_loss)
        ),
        "balanced_softmax_finite": bool(torch.isfinite(balanced_softmax)),
        "candidate_reverse_bias_opposite": bool(
            np.array_equal(observed_bias, -(-observed_bias))
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "manual_ce": float(manual_ce),
        "torch_ce": torch_ce,
        "manual_plain_bce": manual_bce,
        "torch_plain_bce": torch_bce,
        "balanced_bce_bias": observed_bias.tolist(),
        "balanced_softmax_bias": (0.25 * np.log(prior_np)).tolist(),
        "extreme_candidate_loss": float(candidate),
        "extreme_reversed_loss": float(reversed_loss),
    }


def _head_state(head: nn.Linear, optimizer: torch.optim.Optimizer) -> Dict[str, np.ndarray]:
    state = {
        "weight": head.weight.detach().cpu().numpy().astype(
            np.float32, copy=True
        ),
        "bias": head.bias.detach().cpu().numpy().astype(np.float32, copy=True),
    }
    momentum_weight = optimizer.state.get(head.weight, {}).get(
        "momentum_buffer"
    )
    momentum_bias = optimizer.state.get(head.bias, {}).get("momentum_buffer")
    if momentum_weight is not None:
        state["weight_momentum"] = (
            momentum_weight.detach().cpu().numpy().astype(
                np.float32, copy=True
            )
        )
    if momentum_bias is not None:
        state["bias_momentum"] = (
            momentum_bias.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    return state


def train_role_fold(
    *,
    role: str,
    fold: int,
    features: Tensor,
    targets: Tensor,
    fit_indices: np.ndarray,
    held_indices: np.ndarray,
    prior_float64: Sequence[float],
    device: torch.device,
) -> Dict[str, object]:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role: {role}")
    repeat = role == "balanced_bce_seed_repeat"
    seed = BASE_SEED + int(fold) + (REPEAT_SEED_OFFSET if repeat else 0)
    head = initialize_head(seed=seed, device=device)
    initial = initial_head_evidence(seed=seed)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    )
    prior = torch.as_tensor(
        np.asarray(prior_float64, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    orders = build_epoch_orders(fit_indices, seed=seed)
    trace: List[Dict[str, object]] = []
    update_count = 0

    head.train()
    for epoch_index, order in enumerate(orders):
        lr = learning_rate(epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        row_count = 0
        for start in range(0, int(order.size), BATCH_SIZE):
            batch_np = order[start : start + BATCH_SIZE]
            batch = torch.as_tensor(
                batch_np, dtype=torch.long, device=device
            )
            raw_logits = head(features.index_select(0, batch))
            loss = role_loss(
                role,
                raw_logits,
                targets.index_select(0, batch),
                prior=prior,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite loss role={role} fold={fold} "
                    f"epoch={epoch_index + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = int(batch.numel())
            loss_sum += float(loss.detach()) * batch_rows
            row_count += batch_rows
            update_count += 1
        if epoch_index + 1 in TRACE_EPOCHS:
            trace.append(
                {
                    "role": role,
                    "outer_fold": int(fold),
                    "epoch": int(epoch_index + 1),
                    "learning_rate": float(lr),
                    "mean_loss": float(loss_sum / row_count),
                    "rows": int(row_count),
                    "updates_cumulative": int(update_count),
                }
            )

    head.eval()
    held = torch.as_tensor(
        np.asarray(held_indices, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        raw_logits = head(features.index_select(0, held))
        probabilities = torch.softmax(raw_logits, dim=1)
    state = _head_state(head, optimizer)
    return {
        "role": role,
        "outer_fold": int(fold),
        "seed": int(seed),
        "initial_head": initial,
        "orders": orders,
        "order_hashes": [array_sha256(order) for order in orders],
        "all_orders_sha256": array_sha256(np.concatenate(orders)),
        "raw_logits": raw_logits.detach().cpu().numpy().astype(
            np.float32, copy=True
        ),
        "probabilities": probabilities.detach().cpu().numpy().astype(
            np.float32, copy=True
        ),
        "state": state,
        "state_sha256": state_arrays_sha256(state),
        "trace": trace,
        "update_count": int(update_count),
        "held_score_count": 1,
        "score_epoch": EPOCHS,
    }
