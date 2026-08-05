from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ADAPTER_MARKERS = (".adapter_attn.", ".adapter_mlp.")
DEFAULT_RIVALS = (0, 2, 4)


def is_adapter_parameter(name: str) -> bool:
    qualified = f".{str(name)}"
    return any(marker in qualified for marker in ADAPTER_MARKERS)


def configure_adapter_only(model: nn.Module) -> Dict[str, int]:
    """Freeze the supervised primary path and expose only ConvPass parameters."""

    total = 0
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        enabled = is_adapter_parameter(name)
        parameter.requires_grad_(enabled)
        count = int(parameter.numel())
        total += count
        if enabled:
            trainable += count
        else:
            frozen += count
    if trainable <= 0:
        raise ValueError("B23 found no ConvPass adapter parameters.")
    if total != trainable + frozen:
        raise RuntimeError("B23 parameter accounting is inconsistent.")
    return {"total": total, "trainable": trainable, "frozen": frozen}


def adapter_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
        if is_adapter_parameter(name)
    }


def load_adapter_state_dict(
    model: nn.Module,
    state: Mapping[str, Tensor],
) -> None:
    expected = {
        name for name in model.state_dict() if is_adapter_parameter(name)
    }
    observed = {str(name) for name in state}
    if observed != expected:
        raise ValueError(
            "B23 adapter state keys changed: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"unexpected={sorted(observed - expected)[:5]}."
        )
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"B23 adapter load has unexpected keys: {incompatible.unexpected_keys}."
        )
    missing_non_adapter = {
        name for name in incompatible.missing_keys if not is_adapter_parameter(name)
    }
    expected_non_adapter = {
        name for name in model.state_dict() if not is_adapter_parameter(name)
    }
    if missing_non_adapter != expected_non_adapter:
        raise RuntimeError("B23 adapter-only load did not omit exactly the primary path.")


def class1_margin(logits: Tensor, rivals: Iterable[int] = DEFAULT_RIVALS) -> Tensor:
    if logits.ndim != 2 or int(logits.size(1)) != 5:
        raise ValueError(f"B23 expects five-class [B,5] logits, got {tuple(logits.shape)}.")
    rival_indices = tuple(int(value) for value in rivals)
    if not rival_indices or 1 in rival_indices:
        raise ValueError("B23 rivals must be non-empty and exclude class 1.")
    rival = logits.index_select(
        1,
        torch.as_tensor(rival_indices, dtype=torch.long, device=logits.device),
    ).amax(dim=1)
    return logits[:, 1] - rival


def guarded_adapter_loss(
    logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    task_loss: Callable[[Tensor, Tensor], Tensor],
    *,
    retention_weight: float = 0.25,
    distillation_weight: float = 0.10,
    retention_slack: float = 0.05,
    temperature: float = 2.0,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Task loss plus a true-class-1 margin guard and weak teacher trust region.

    The primary B9 path is frozen.  The guard penalizes an adapter only when it
    lowers a labelled class-1 margin more than ``retention_slack`` relative to
    B9.  It does not prevent the adapter from increasing that margin or from
    suppressing class-1 scores on hard negatives.
    """

    if logits.shape != teacher_logits.shape:
        raise ValueError("B23 student/teacher logit shapes differ.")
    if targets.ndim != 1 or int(targets.numel()) != int(logits.size(0)):
        raise ValueError("B23 targets are not aligned with logits.")
    for name, value in (
        ("retention_weight", retention_weight),
        ("distillation_weight", distillation_weight),
        ("retention_slack", retention_slack),
        ("temperature", temperature),
    ):
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"B23 {name} must be finite.")
    if retention_weight < 0.0 or distillation_weight < 0.0:
        raise ValueError("B23 loss weights must be non-negative.")
    if retention_slack < 0.0 or temperature <= 0.0:
        raise ValueError("B23 slack/temperature are invalid.")

    teacher = teacher_logits.detach()
    task = task_loss(logits, targets)
    positive_mask = targets.eq(1)
    if bool(positive_mask.any().item()):
        teacher_margin = class1_margin(teacher)[positive_mask]
        student_margin = class1_margin(logits)[positive_mask]
        retention = F.relu(
            teacher_margin - student_margin - float(retention_slack)
        ).mean()
    else:
        retention = logits.sum() * 0.0

    t = float(temperature)
    distillation = F.kl_div(
        F.log_softmax(logits / t, dim=1),
        F.softmax(teacher / t, dim=1),
        reduction="batchmean",
    ) * (t * t)
    total = (
        task
        + float(retention_weight) * retention
        + float(distillation_weight) * distillation
    )
    return total, {
        "task": task.detach(),
        "retention": retention.detach(),
        "distillation": distillation.detach(),
        "total": total.detach(),
        "class1_rows": positive_mask.sum().detach(),
    }
