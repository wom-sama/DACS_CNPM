from __future__ import annotations

import csv
from collections import Counter, deque
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_ENV_VALUES


def is_windows_platform() -> bool:
    return platform.system().strip().lower() == "windows"


def build_safe_dataloader_kwargs(
    requested_num_workers: int,
    requested_pin_memory: bool,
    context: str,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    logger: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requested_workers = max(0, int(requested_num_workers))
    effective_workers = requested_workers
    effective_pin_memory = bool(requested_pin_memory)
    windows_safe_mode = False
    notes: List[str] = []

    if is_windows_platform() and requested_workers > 0:
        if _env_flag_enabled("TRKH_ALLOW_WINDOWS_MULTIPROCESSING"):
            notes.append("windows_multiprocessing_allowed_by_env")
        else:
            # Windows PyTorch workers exchange tensors through shared file mappings.
            # Clamping workers prevents error 1455 when the paging file is exhausted.
            effective_workers = 0
            windows_safe_mode = True
            notes.append("windows_num_workers_clamped_to_0")

    if is_windows_platform() and effective_pin_memory:
        if _env_flag_enabled("TRKH_ALLOW_WINDOWS_PIN_MEMORY"):
            notes.append("windows_pin_memory_allowed_by_env")
        else:
            effective_pin_memory = False
            windows_safe_mode = True
            notes.append("windows_pin_memory_disabled")

    kwargs: Dict[str, Any] = {
        "num_workers": effective_workers,
        "pin_memory": effective_pin_memory,
    }
    effective_persistent_workers = bool(persistent_workers)
    if is_windows_platform() and effective_workers > 0 and effective_persistent_workers:
        if _env_flag_enabled("TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS"):
            notes.append("windows_persistent_workers_allowed_by_env")
        else:
            effective_persistent_workers = False
            windows_safe_mode = True
            notes.append("windows_persistent_workers_disabled")

    if effective_workers > 0:
        if effective_persistent_workers:
            kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    summary: Dict[str, Any] = {
        "context": context,
        "platform": platform.system(),
        "requested_num_workers": requested_workers,
        "effective_num_workers": effective_workers,
        "requested_pin_memory": bool(requested_pin_memory),
        "effective_pin_memory": effective_pin_memory,
        "persistent_workers": bool(kwargs.get("persistent_workers", False)),
        "prefetch_factor": kwargs.get("prefetch_factor"),
        "windows_safe_mode": windows_safe_mode,
        "notes": notes,
    }
    if logger is not None and notes:
        logger.warning("DataLoader safety adjustment: %s", summary)
    return kwargs, summary


def resolve_image_cache_megabytes(
    default_megabytes: int = 256,
    context: Optional[str] = None,
) -> int:
    env_names: List[str] = []
    if context:
        context_token = str(context).strip().lower().split("_")[0]
        if context_token:
            env_names.append(f"TRKH_{context_token.upper()}_IMAGE_CACHE_MB")
    env_names.append("TRKH_IMAGE_CACHE_MB")
    for env_name in env_names:
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value.strip() == "":
            continue
        try:
            return max(0, int(float(raw_value)))
        except ValueError:
            continue
    return int(default_megabytes)


def maybe_enable_dataset_image_cache(
    dataset: Any,
    enabled: bool,
    context: str,
    max_megabytes: Optional[int] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    if not enabled:
        return {"context": context, "enabled": False, "reason": "multiprocessing_enabled"}

    cache_mb = (
        resolve_image_cache_megabytes(context=context)
        if max_megabytes is None
        else int(max_megabytes)
    )
    if cache_mb <= 0:
        return {"context": context, "enabled": False, "reason": "cache_size_is_0"}

    enable_cache = getattr(dataset, "enable_image_cache", None)
    if not callable(enable_cache):
        return {"context": context, "enabled": False, "reason": "dataset_has_no_image_cache"}

    enable_cache(max_megabytes=cache_mb)
    stats_fn = getattr(dataset, "image_cache_stats", None)
    stats = stats_fn() if callable(stats_fn) else {}
    summary = {"context": context, "enabled": True, **stats}
    if logger is not None:
        logger.info("Enabled single-process image cache: %s", summary)
    return summary



def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = []
            for existing_row in reader:
                existing_row.pop(None, None)
                existing_rows.append(existing_row)
        merged_fieldnames = list(existing_fieldnames)
        for fieldname in fieldnames:
            if fieldname not in merged_fieldnames:
                merged_fieldnames.append(fieldname)
        if merged_fieldnames != existing_fieldnames:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=merged_fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
            return
        fieldnames = existing_fieldnames

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


DETECTION_HEAD_PREFIXES = (
    "query_embed",
    "query_content_embed",
    "memory_position_embedding",
    "memory_adapter",
    "decoder",
    "classification_head",
    "objectness_head",
    "bbox_head",
)


def build_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float | None = None,
    backbone_lr_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    no_decay_keywords: Iterable[str]
    if hasattr(model, "no_weight_decay_keywords"):
        no_decay_keywords = model.no_weight_decay_keywords()
    else:
        no_decay_keywords = ("bias", "norm")

    lr_scale = float(backbone_lr_scale)
    use_split_lr = bool(getattr(model, "is_detr_model", False)) and abs(lr_scale - 1.0) > 1e-12
    grouped_params: Dict[Tuple[str, bool], List[nn.Parameter]] = {
        ("head", True): [],
        ("head", False): [],
        ("backbone", True): [],
        ("backbone", False): [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = name.startswith(DETECTION_HEAD_PREFIXES) if use_split_lr else True
        group_name = "head" if is_head else "backbone"
        use_decay = not any(keyword in name for keyword in no_decay_keywords)
        grouped_params[(group_name, use_decay)].append(parameter)

    param_groups: List[Dict[str, Any]] = []
    for group_name, use_decay in (
        ("head", True),
        ("head", False),
        ("backbone", True),
        ("backbone", False),
    ):
        params = grouped_params[(group_name, use_decay)]
        if not params:
            continue
        param_group: Dict[str, Any] = {
            "params": params,
            "weight_decay": weight_decay if use_decay else 0.0,
            "name": f"{group_name}_{'decay' if use_decay else 'no_decay'}",
        }
        if learning_rate is not None and use_split_lr and group_name == "backbone":
            param_group["lr"] = float(learning_rate) * lr_scale
        param_groups.append(param_group)
    return param_groups


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"rho phai >= 0, nhan duoc {rho}")
        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho = float(rho)
        self.adaptive = bool(adaptive)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = bool(group.get("adaptive", self.adaptive))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if adaptive:
                    grad = grad * parameter.abs()
                norms.append(torch.norm(grad, p=2).to(shared_device))
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm) or float(grad_norm.item()) == 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        for group in self.param_groups:
            scale = float(group.get("rho", self.rho)) / (grad_norm + 1e-12)
            adaptive = bool(group.get("adaptive", self.adaptive))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                e_w = parameter.grad * scale.to(parameter)
                if adaptive:
                    e_w = e_w * parameter.pow(2)
                parameter.add_(e_w)
                self.state[parameter]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                e_w = self.state[parameter].pop("e_w", None)
                if e_w is not None:
                    parameter.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):  # pragma: no cover
        raise NotImplementedError("SAM yeu cau goi first_step() va second_step() ro rang.")

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = self.base_optimizer.state_dict()
        state_dict["sam_defaults"] = {"rho": self.rho, "adaptive": self.adaptive}
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        sam_defaults = state_dict.pop("sam_defaults", None)
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        if sam_defaults is not None:
            self.rho = float(sam_defaults.get("rho", self.rho))
            self.adaptive = bool(sam_defaults.get("adaptive", self.adaptive))


class WarmupAnnealingScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_factor: float,
        total_epochs: int,
        min_learning_rate: float,
        decay_style: str = "cosine",
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(min(max(warmup_start_factor, 0.0), 1.0))
        self.base_lrs = [float(param_group["lr"]) for param_group in optimizer.param_groups]
        self.min_learning_rate = float(min_learning_rate)
        self.total_epochs = max(self.warmup_epochs + 1, int(total_epochs))
        self.decay_style = str(decay_style).strip().lower()
        if self.decay_style not in {"cosine", "linear"}:
            raise ValueError(f"Khong ho tro decay_style: {decay_style}")
        self.last_lrs = list(self.base_lrs)
        if self.warmup_epochs > 0:
            self._set_warmup_lrs(0.0)
        else:
            self._set_lrs(self.base_lrs)

    def _set_lrs(self, values: Sequence[float]) -> None:
        self.last_lrs = [float(value) for value in values]
        for param_group, value in zip(self.optimizer.param_groups, self.last_lrs):
            param_group["lr"] = float(value)

    def _set_warmup_lrs(self, progress: float) -> None:
        if self.warmup_epochs <= 0:
            self._set_lrs(self.base_lrs)
            return
        clamped_progress = min(max(float(progress), 0.0), float(self.warmup_epochs))
        ratio = clamped_progress / float(max(1, self.warmup_epochs))
        scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * ratio
        self._set_lrs([base_lr * scale for base_lr in self.base_lrs])

    def _set_decay_lrs(self, progress: float) -> None:
        decay_span = float(max(1, self.total_epochs - self.warmup_epochs))
        decay_progress = min(
            max((float(progress) - float(self.warmup_epochs)) / decay_span, 0.0),
            1.0,
        )
        if self.decay_style == "linear":
            decay_factor = 1.0 - decay_progress
        else:
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        values = [
            self.min_learning_rate + (base_lr - self.min_learning_rate) * decay_factor
            for base_lr in self.base_lrs
        ]
        self._set_lrs(values)

    def step(self, epoch_progress: float) -> None:
        progress = float(max(0.0, epoch_progress))
        if self.warmup_epochs > 0 and progress < self.warmup_epochs:
            self._set_warmup_lrs(progress)
            return
        self._set_decay_lrs(progress)

    def get_last_lr(self) -> List[float]:
        return list(self.last_lrs)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_epochs": int(self.warmup_epochs),
            "warmup_start_factor": float(self.warmup_start_factor),
            "base_lrs": list(self.base_lrs),
            "min_learning_rate": float(self.min_learning_rate),
            "total_epochs": int(self.total_epochs),
            "decay_style": str(self.decay_style),
            "last_lrs": list(self.last_lrs),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.warmup_epochs = max(0, int(state_dict.get("warmup_epochs", self.warmup_epochs)))
        self.warmup_start_factor = float(
            min(max(state_dict.get("warmup_start_factor", self.warmup_start_factor), 0.0), 1.0)
        )
        self.base_lrs = [float(value) for value in state_dict.get("base_lrs", self.base_lrs)]
        self.min_learning_rate = float(state_dict.get("min_learning_rate", self.min_learning_rate))
        self.total_epochs = max(
            self.warmup_epochs + 1,
            int(state_dict.get("total_epochs", self.total_epochs)),
        )
        self.decay_style = str(state_dict.get("decay_style", self.decay_style)).strip().lower()
        if self.decay_style not in {"cosine", "linear"}:
            raise ValueError(f"Khong ho tro decay_style: {self.decay_style}")
        last_lrs = [float(value) for value in state_dict.get("last_lrs", self.last_lrs)]
        if len(last_lrs) != len(self.optimizer.param_groups):
            last_lrs = list(self.base_lrs)
        self._set_lrs(last_lrs)


def build_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    warmup_start_factor: float,
    total_epochs: int,
    min_learning_rate: float,
    decay_style: str = "cosine",
) -> WarmupAnnealingScheduler:
    return WarmupAnnealingScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        warmup_start_factor=warmup_start_factor,
        total_epochs=total_epochs,
        min_learning_rate=min_learning_rate,
        decay_style=decay_style,
    )


def compute_class_distribution_skew(class_counts: Sequence[int]) -> Dict[str, float]:
    counts = np.asarray([max(0, int(value)) for value in class_counts], dtype=np.float64)
    total = float(counts.sum())
    num_classes = int(counts.size)
    if num_classes == 0 or total <= 0.0:
        return {
            "num_classes": float(num_classes),
            "total_samples": total,
            "normalized_entropy": 0.0,
            "effective_num_classes": 0.0,
            "max_min_ratio": float("inf"),
            "imbalance_ratio": float("inf"),
            "intervention_strength": 1.0,
        }

    probabilities = counts / total
    non_zero_probabilities = probabilities[probabilities > 0.0]
    entropy = float(-(non_zero_probabilities * np.log(non_zero_probabilities)).sum())
    max_entropy = float(np.log(max(2, num_classes)))
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 1.0
    effective_num_classes = float(np.exp(entropy))

    non_zero_counts = counts[counts > 0.0]
    min_count = float(non_zero_counts.min()) if non_zero_counts.size else 0.0
    max_count = float(non_zero_counts.max()) if non_zero_counts.size else 0.0
    imbalance_ratio = (max_count / min_count) if min_count > 0.0 else float("inf")
    finite_ratio = imbalance_ratio if np.isfinite(imbalance_ratio) else float(num_classes * 4)
    ratio_score = min(1.0, max(0.0, (finite_ratio - 1.0) / max(1.0, num_classes - 1.0)))
    entropy_score = min(1.0, max(0.0, 1.0 - normalized_entropy))
    intervention_strength = float(np.clip(0.65 * entropy_score + 0.35 * ratio_score, 0.0, 1.0))

    return {
        "num_classes": float(num_classes),
        "total_samples": total,
        "normalized_entropy": float(normalized_entropy),
        "effective_num_classes": effective_num_classes,
        "max_min_ratio": float(imbalance_ratio),
        "imbalance_ratio": float(imbalance_ratio),
        "intervention_strength": intervention_strength,
    }


class PredictionDriftMonitor:
    def __init__(
        self,
        reference_class_counts: Sequence[int],
        class_names: Sequence[str],
        window_size: int = 128,
        alert_threshold: float = 0.25,
        log_interval: int = 32,
    ) -> None:
        counts = np.asarray([max(0, int(value)) for value in reference_class_counts], dtype=np.float64)
        total = float(counts.sum())
        self.reference_distribution = (
            counts / total if total > 0.0 else np.full(len(class_names), 1.0 / max(1, len(class_names)))
        )
        self.class_names = list(class_names)
        self.window_size = max(1, int(window_size))
        self.alert_threshold = float(max(0.0, alert_threshold))
        self.log_interval = max(1, int(log_interval))
        self.history: Deque[int] = deque(maxlen=self.window_size)
        self.seen_predictions = 0

    def update(self, prediction_index: Optional[int]) -> Optional[Dict[str, Any]]:
        if prediction_index is None:
            return None
        if prediction_index < 0 or prediction_index >= len(self.class_names):
            return None

        self.history.append(int(prediction_index))
        self.seen_predictions += 1
        if len(self.history) < min(self.window_size, len(self.class_names) * 4):
            return None
        if self.seen_predictions % self.log_interval != 0:
            return None

        counts = Counter(self.history)
        current_distribution = np.asarray(
            [counts.get(index, 0) for index in range(len(self.class_names))],
            dtype=np.float64,
        )
        current_distribution /= current_distribution.sum().clip(min=1.0)
        deltas = np.abs(current_distribution - self.reference_distribution)
        max_shift = float(deltas.max())
        if max_shift < self.alert_threshold:
            return None

        shifted_class_index = int(np.argmax(deltas))
        return {
            "warning": "prediction_distribution_drift",
            "window_size": len(self.history),
            "seen_predictions": self.seen_predictions,
            "alert_threshold": self.alert_threshold,
            "max_abs_shift": max_shift,
            "shifted_class_index": shifted_class_index,
            "shifted_class_name": self.class_names[shifted_class_index],
            "reference_distribution": self.reference_distribution.tolist(),
            "current_distribution": current_distribution.tolist(),
        }


def resolve_amp_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    requested = str(os.getenv("TRKH_AMP_DTYPE", "auto")).strip().lower()
    if requested in {"bf16", "bfloat16"}:
        if device is not None and device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    if requested in {"fp16", "float16", "half"}:
        return torch.float16
    if requested in {"auto"}:
        if device is not None and device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=resolve_amp_dtype(device))


def save_checkpoint(path: Path, checkpoint: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def timestamp_run_name(prefix: str = "vit_registers") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _csv_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _ema_series(values: Sequence[float], alpha: float = 0.35) -> List[float]:
    if not values:
        return []
    ema_values = [float(values[0])]
    for value in values[1:]:
        ema_values.append(alpha * float(value) + (1.0 - alpha) * ema_values[-1])
    return ema_values


def plot_training_history(history_csv: Path, output_path: Path) -> None:
    if not history_csv.exists():
        return

    epochs: List[int] = []
    series: Dict[str, List[float]] = {
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "train_register_patch_ratio": [],
        "val_register_patch_ratio": [],
        "val_high_norm_patch_fraction": [],
        "epoch_seconds": [],
    }

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            for key in series:
                series[key].append(_csv_float(row, key))

    if not epochs:
        return

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    axes[0].plot(epochs, series["train_loss"], label="train_loss", linewidth=2.0)
    axes[0].plot(epochs, series["val_loss"], label="val_loss", linewidth=2.0)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, series["val_accuracy"], label="val_accuracy", linewidth=2.0)
    axes[1].plot(epochs, series["val_macro_f1"], label="val_macro_f1", linewidth=2.0)
    axes[1].plot(
        epochs,
        series["val_weighted_f1"],
        label="val_weighted_f1",
        linewidth=1.8,
        linestyle="--",
    )
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend()

    axes[2].plot(epochs, series["learning_rate"], color="tab:orange", linewidth=2.0)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    if any(value > 0.0 for value in series["learning_rate"]):
        axes[2].set_yscale("log")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[3].plot(
        epochs,
        series["train_register_patch_ratio"],
        label="train_register_patch_ratio",
        linewidth=2.0,
    )
    axes[3].plot(
        epochs,
        series["val_register_patch_ratio"],
        label="val_register_patch_ratio",
        linewidth=2.0,
    )
    axes[3].set_title("Register / Patch Ratio")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[3].legend()

    axes[4].plot(
        epochs,
        series["val_high_norm_patch_fraction"],
        color="tab:red",
        linewidth=2.0,
    )
    axes[4].set_title("High-Norm Patch Fraction")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylim(bottom=0.0)
    axes[4].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[5].plot(epochs, series["epoch_seconds"], color="tab:purple", linewidth=2.0)
    axes[5].set_title("Epoch Duration (s)")
    axes[5].set_xlabel("Epoch")
    axes[5].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _metric_from_final_test(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    calibrated = metrics.get("calibrated")
    if isinstance(calibrated, dict):
        accepted_metrics = calibrated.get("accepted_metrics")
        if isinstance(accepted_metrics, dict):
            value = accepted_metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def plot_train_val_final_test_metrics(
    history_csv: Path,
    final_test_metrics_json: Path,
    output_path: Path,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    if not history_csv.exists():
        return

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    if not epochs:
        return

    def series(name: str) -> List[float]:
        return [_csv_float(row, name) for row in rows]

    val_macro_f1 = series("val_macro_f1")
    best_index = int(np.argmax(val_macro_f1)) if val_macro_f1 else len(epochs) - 1
    best_epoch = epochs[best_index]
    final_test_metrics: Dict[str, Any] = {}
    if final_test_metrics_json.exists():
        try:
            with final_test_metrics_json.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                final_test_metrics = loaded
        except (json.JSONDecodeError, OSError):
            final_test_metrics = {}

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].plot(epochs, series("train_loss"), label="train_loss", linewidth=2.0)
    axes[0].plot(epochs, series("val_loss"), label="val_loss", linewidth=2.0)
    axes[0].set_title("Train/Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    metric_specs = [
        ("val_accuracy", "test accuracy", "accuracy", "tab:blue"),
        ("val_macro_f1", "test macro_f1", "macro_f1", "tab:green"),
        ("val_weighted_f1", "test weighted_f1", "weighted_f1", "tab:orange"),
    ]
    for val_key, test_label, test_key, color in metric_specs:
        values = series(val_key)
        axes[1].plot(epochs, values, label=val_key, linewidth=2.0)
        test_value = _metric_from_final_test(final_test_metrics, test_key)
        if test_value is not None:
            axes[1].scatter([best_epoch], [test_value], marker="*", s=150, color=color, label=test_label)
    axes[1].axvline(best_epoch, color="black", linestyle=":", linewidth=1.2, alpha=0.6, label="best val epoch")
    axes[1].set_title("Validation Curves + Final Test Marker")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend(fontsize=8)

    class_indices = []
    for field_name in rows[0].keys():
        if field_name.startswith("val_class_") and field_name.endswith("_f1"):
            index_text = field_name[len("val_class_") : -len("_f1")]
            if index_text.isdigit():
                class_indices.append(int(index_text))
    class_indices = sorted(set(class_indices))
    final_per_class = final_test_metrics.get("per_class")
    final_class_f1 = {}
    if isinstance(final_per_class, list):
        for item in final_per_class:
            if isinstance(item, dict):
                index = item.get("class_index")
                f1 = item.get("f1")
                if isinstance(index, int) and isinstance(f1, (int, float)):
                    final_class_f1[index] = float(f1)

    for class_index in class_indices:
        label = (
            str(class_names[class_index])
            if class_names is not None and class_index < len(class_names)
            else f"class_{class_index}"
        )
        axes[2].plot(epochs, series(f"val_class_{class_index}_f1"), label=label, linewidth=1.6)
        if class_index in final_class_f1:
            axes[2].scatter([best_epoch], [final_class_f1[class_index]], marker="*", s=90)
    axes[2].axvline(best_epoch, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    axes[2].set_title("Per-Class Validation F1 + Final Test Markers")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[2].legend(fontsize=7)

    axes[3].plot(epochs, series("learning_rate"), color="tab:purple", label="learning_rate", linewidth=2.0)
    if "epoch_seconds" in rows[0]:
        ax_right = axes[3].twinx()
        ax_right.plot(epochs, series("epoch_seconds"), color="tab:red", alpha=0.55, label="epoch_seconds")
        ax_right.set_ylabel("Seconds")
    if any(_csv_float(row, "learning_rate") > 0.0 for row in rows):
        axes[3].set_yscale("log")
    axes[3].set_title("Learning Rate / Epoch Time")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[3].legend(fontsize=8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_all_training_metrics(history_csv: Path, output_path: Path) -> None:
    if not history_csv.exists():
        return

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    if not epochs:
        return

    numeric_series: Dict[str, List[float]] = {}
    ignored_fields = {"epoch", "train_stage"}
    for field_name in rows[0].keys():
        if field_name in ignored_fields:
            continue
        values: List[float] = []
        is_numeric = True
        for row in rows:
            try:
                values.append(_csv_float(row, field_name))
            except (TypeError, ValueError):
                is_numeric = False
                break
        if is_numeric and values:
            numeric_series[field_name] = values

    if not numeric_series:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    max_plots_per_figure = 45
    series_items = list(numeric_series.items())
    for chunk_index in range(0, len(series_items), max_plots_per_figure):
        chunk = series_items[chunk_index : chunk_index + max_plots_per_figure]
        rows_count = int(math.ceil(len(chunk) / columns))
        figure, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(18, max(4.0, rows_count * 3.1)),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for axis, (field_name, values) in zip(axes_flat, chunk):
            axis.plot(epochs, values, linewidth=1.8)
            axis.set_title(field_name)
            axis.set_xlabel("Epoch")
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            if field_name == "learning_rate" and any(value > 0.0 for value in values):
                axis.set_yscale("log")

        for axis in axes_flat[len(chunk):]:
            axis.axis("off")

        figure.tight_layout()
        if chunk_index == 0:
            chunk_output = output_path
        else:
            part_number = int(chunk_index / max_plots_per_figure) + 1
            chunk_output = output_path.with_name(
                f"{output_path.stem}_part{part_number:02d}{output_path.suffix}"
            )
        figure.savefig(chunk_output, dpi=200, bbox_inches="tight")
        plt.close(figure)


def plot_per_class_training_metrics(
    history_csv: Path,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    if not history_csv.exists():
        return

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    if not epochs:
        return

    num_classes = len(class_names)
    has_foreground = any(f"val_class_{index}_f1" in rows[0] for index in range(num_classes))
    has_background_aware = any(f"val_bgaware_class_{index}_f1" in rows[0] for index in range(num_classes))
    if not has_foreground and not has_background_aware:
        return

    columns = 2 if num_classes > 1 else 1
    rows_count = int(math.ceil(num_classes / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(8.2 * columns, 4.8 * rows_count))
    axes = np.atleast_1d(axes).reshape(rows_count, columns)

    for class_index, axis in enumerate(axes.ravel()):
        if class_index >= num_classes:
            axis.axis("off")
            continue

        precision = [_csv_float(row, f"val_class_{class_index}_precision") for row in rows]
        recall = [_csv_float(row, f"val_class_{class_index}_recall") for row in rows]
        f1 = [_csv_float(row, f"val_class_{class_index}_f1") for row in rows]
        bg_f1 = [_csv_float(row, f"val_bgaware_class_{class_index}_f1") for row in rows]

        if has_foreground:
            axis.plot(epochs, precision, label="precision", linewidth=1.5, alpha=0.8)
            axis.plot(epochs, recall, label="recall", linewidth=1.5, alpha=0.8)
            axis.plot(epochs, f1, label="f1", linewidth=2.2)
        if has_background_aware:
            axis.plot(epochs, bg_f1, label="background-aware f1", linewidth=2.0, linestyle="--")

        axis.set_title(str(class_names[class_index]))
        axis.set_xlabel("Epoch")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        axis.legend(fontsize=8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_validation_metric(
    history_csv: Path,
    class_names: Sequence[str],
    output_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
    target: Optional[float] = None,
) -> None:
    if not history_csv.exists():
        return

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    if not epochs:
        return

    metric_key = str(metric).strip().lower()
    field_candidates: Dict[int, List[str]] = {}
    if class_names:
        class_indices = list(range(len(class_names)))
    else:
        class_indices = []
        for field_name in rows[0].keys():
            if not field_name.startswith("val_class_"):
                continue
            pieces = field_name.split("_")
            if len(pieces) >= 4 and pieces[2].isdigit():
                class_indices.append(int(pieces[2]))
        class_indices = sorted(set(class_indices))

    for class_index in class_indices:
        if metric_key in {"accuracy", "acc"}:
            # Per-class validation accuracy for single-label classification is TP / support,
            # which is the same quantity as recall for that class.
            field_candidates[class_index] = [
                f"val_class_{class_index}_accuracy",
                f"val_class_{class_index}_recall",
            ]
        else:
            field_candidates[class_index] = [f"val_class_{class_index}_{metric_key}"]

    plotted = False
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for class_index in class_indices:
        field_name = next(
            (candidate for candidate in field_candidates[class_index] if candidate in rows[0]),
            None,
        )
        if field_name is None:
            continue
        values = [_csv_float(row, field_name) for row in rows]
        if not values:
            continue
        label = (
            str(class_names[class_index])
            if class_names and class_index < len(class_names)
            else f"class_{class_index}"
        )
        axis.plot(epochs, values, label=label, linewidth=2.0)
        plotted = True

    if not plotted:
        plt.close(figure)
        return

    if target is not None:
        axis.axhline(float(target), color="black", linestyle=":", linewidth=1.2, alpha=0.65, label=f"target {target:g}")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_detection_training_metrics(history_csv: Path, output_path: Path) -> None:
    if not history_csv.exists():
        return

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    if not epochs:
        return

    raw_precision = [_csv_float(row, "val_detection_precision_50") for row in rows]
    raw_recall = [_csv_float(row, "val_detection_recall_50") for row in rows]
    raw_f1 = [_csv_float(row, "val_detection_f1_50") for row in rows]
    best_f1 = [_csv_float(row, "val_best_detection_f1_50") for row in rows]
    best_conf = [_csv_float(row, "val_best_detection_confidence") for row in rows]
    bbox_iou = [_csv_float(row, "val_bbox_iou") for row in rows]
    background_rate = [_csv_float(row, "val_background_prediction_rate") for row in rows]

    if not any(max(series, default=0.0) > 0.0 for series in (raw_precision, raw_recall, raw_f1, best_f1, bbox_iou, background_rate)):
        return

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()

    axes[0].plot(epochs, raw_precision, label="raw precision@0.5", linewidth=1.8)
    axes[0].plot(epochs, raw_recall, label="raw recall@0.5", linewidth=1.8)
    axes[0].plot(epochs, raw_f1, label="raw f1@0.5", linewidth=2.2)
    axes[0].set_title("Raw Detection Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, best_f1, label="best threshold f1@0.5", color="tab:green", linewidth=2.2)
    axes[1].plot(epochs, best_conf, label="best confidence", color="tab:orange", linewidth=1.8)
    axes[1].set_title("Calibrated Detection")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend()

    axes[2].plot(epochs, bbox_iou, label="bbox mean IoU", color="tab:blue", linewidth=2.2)
    axes[2].set_title("Localization")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[2].legend()

    axes[3].plot(epochs, background_rate, label="matched background prediction rate", color="tab:red", linewidth=2.0)
    axes[3].set_title("Background Calibration")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylim(0.0, 1.02)
    axes[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[3].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_validation_convergence(history_csv: Path, output_path: Path) -> None:
    if not history_csv.exists():
        return

    epochs: List[int] = []
    val_accuracy: List[float] = []
    val_macro_f1: List[float] = []
    val_weighted_f1: List[float] = []
    train_loss: List[float] = []
    val_loss: List[float] = []

    with history_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(_csv_float(row, "train_loss"))
            val_loss.append(_csv_float(row, "val_loss"))
            val_accuracy.append(_csv_float(row, "val_accuracy"))
            val_macro_f1.append(_csv_float(row, "val_macro_f1"))
            val_weighted_f1.append(_csv_float(row, "val_weighted_f1"))

    if not epochs:
        return

    ema_val_accuracy = _ema_series(val_accuracy)
    ema_val_macro_f1 = _ema_series(val_macro_f1)
    ema_val_weighted_f1 = _ema_series(val_weighted_f1)
    ema_train_loss = _ema_series(train_loss)
    ema_val_loss = _ema_series(val_loss)
    best_accuracy_epoch = epochs[int(np.argmax(np.asarray(val_accuracy, dtype=np.float32)))]
    best_macro_f1_epoch = epochs[int(np.argmax(np.asarray(val_macro_f1, dtype=np.float32)))]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].plot(epochs, val_accuracy, label="val_accuracy", color="tab:blue", alpha=0.35, linewidth=1.2)
    axes[0].plot(epochs, ema_val_accuracy, label="val_accuracy_ema", color="tab:blue", linewidth=2.2)
    axes[0].scatter([best_accuracy_epoch], [max(val_accuracy)], color="tab:blue", s=40)
    axes[0].set_title("Validation Accuracy Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, val_macro_f1, label="val_macro_f1", color="tab:green", alpha=0.35, linewidth=1.2)
    axes[1].plot(epochs, ema_val_macro_f1, label="val_macro_f1_ema", color="tab:green", linewidth=2.2)
    axes[1].scatter([best_macro_f1_epoch], [max(val_macro_f1)], color="tab:green", s=40)
    axes[1].set_title("Validation Macro-F1 Convergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend()

    axes[2].plot(epochs, train_loss, label="train_loss", color="tab:red", alpha=0.3, linewidth=1.2)
    axes[2].plot(epochs, ema_train_loss, label="train_loss_ema", color="tab:red", linewidth=2.1)
    axes[2].plot(epochs, val_loss, label="val_loss", color="tab:orange", alpha=0.3, linewidth=1.2)
    axes[2].plot(epochs, ema_val_loss, label="val_loss_ema", color="tab:orange", linewidth=2.1)
    axes[2].set_title("Loss Convergence")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[2].legend()

    axes[3].plot(
        epochs,
        val_weighted_f1,
        label="val_weighted_f1",
        color="tab:purple",
        alpha=0.35,
        linewidth=1.2,
    )
    axes[3].plot(
        epochs,
        ema_val_weighted_f1,
        label="val_weighted_f1_ema",
        color="tab:purple",
        linewidth=2.2,
    )
    axes[3].set_title("Validation Weighted-F1 Convergence")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylim(0.0, 1.02)
    axes[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[3].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_overview(
    class_names: Sequence[str],
    train_class_counts: Sequence[int],
    val_class_counts: Sequence[int],
    train_bboxes: Sequence[Sequence[float]],
    val_bboxes: Sequence[Sequence[float]],
    train_audit: Dict[str, Any],
    val_audit: Dict[str, Any],
    output_path: Path,
) -> None:
    def _to_bbox_array(values: Sequence[Sequence[float]]) -> np.ndarray:
        if not values:
            return np.zeros((0, 4), dtype=np.float32)
        array = np.asarray(values, dtype=np.float32)
        return array.reshape(-1, 4)

    def _plot_hist(axis, values: np.ndarray, label: str, color: str, bins: int = 30) -> None:
        if values.size == 0:
            return
        axis.hist(values, bins=bins, alpha=0.45, label=label, color=color, edgecolor="none")

    train_boxes = _to_bbox_array(train_bboxes)
    val_boxes = _to_bbox_array(val_bboxes)
    non_empty_boxes = [array for array in (train_boxes, val_boxes) if array.size]
    combined_boxes = (
        np.concatenate(non_empty_boxes, axis=0)
        if non_empty_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )
    class_indices = np.arange(len(class_names), dtype=np.float32)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    width = 0.38
    axes[0].bar(class_indices - width / 2.0, train_class_counts, width=width, label="train")
    axes[0].bar(class_indices + width / 2.0, val_class_counts, width=width, label="val")
    axes[0].set_title("Class Distribution")
    axes[0].set_xticks(class_indices)
    axes[0].set_xticklabels(class_names, rotation=30, ha="right")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    total_keys = ("image_file_count", "label_file_count", "selected_sample_count")
    total_train = [int(train_audit.get(key, 0)) for key in total_keys]
    total_val = [int(val_audit.get(key, 0)) for key in total_keys]
    if total_train[-1] == 0 and total_val[-1] == 0:
        total_train[-1] = int(train_audit.get("valid_object_count", 0))
        total_val[-1] = int(val_audit.get("valid_object_count", 0))
    total_labels = ["images", "labels", "samples"]
    total_indices = np.arange(len(total_labels), dtype=np.float32)
    axes[1].bar(total_indices - width / 2.0, total_train, width=width, label="train")
    axes[1].bar(total_indices + width / 2.0, total_val, width=width, label="val")
    axes[1].set_title("Dataset Totals")
    axes[1].set_xticks(total_indices)
    axes[1].set_xticklabels(total_labels)
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[1].legend()

    issue_keys = (
        "empty_label_count",
        "missing_image_count",
        "invalid_line_count",
        "invalid_bbox_count",
        "invalid_class_count",
    )
    issue_train = [int(train_audit.get(key, 0)) for key in issue_keys]
    issue_val = [int(val_audit.get(key, 0)) for key in issue_keys]
    issue_labels = ["empty", "missing_img", "bad_line", "bad_bbox", "bad_class"]
    issue_indices = np.arange(len(issue_labels), dtype=np.float32)
    axes[2].bar(issue_indices - width / 2.0, issue_train, width=width, label="train")
    axes[2].bar(issue_indices + width / 2.0, issue_val, width=width, label="val")
    axes[2].set_title("Dataset Issues")
    axes[2].set_xticks(issue_indices)
    axes[2].set_xticklabels(issue_labels, rotation=20, ha="right")
    axes[2].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[2].legend()

    if train_boxes.size:
        axes[3].scatter(train_boxes[:, 0], train_boxes[:, 1], s=8, alpha=0.16, label="train")
    if val_boxes.size:
        axes[3].scatter(val_boxes[:, 0], val_boxes[:, 1], s=10, alpha=0.28, label="val")
    axes[3].set_title("BBox Centers")
    axes[3].set_xlabel("x_center")
    axes[3].set_ylabel("y_center")
    axes[3].set_xlim(0.0, 1.0)
    axes[3].set_ylim(1.0, 0.0)
    axes[3].grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    if train_boxes.size or val_boxes.size:
        axes[3].legend()

    if train_boxes.size:
        axes[4].scatter(train_boxes[:, 2], train_boxes[:, 3], s=8, alpha=0.16, label="train")
    if val_boxes.size:
        axes[4].scatter(val_boxes[:, 2], val_boxes[:, 3], s=10, alpha=0.28, label="val")
    axes[4].set_title("BBox Width vs Height")
    axes[4].set_xlabel("width")
    axes[4].set_ylabel("height")
    axes[4].set_xlim(0.0, 1.0)
    axes[4].set_ylim(0.0, 1.0)
    axes[4].grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    if train_boxes.size or val_boxes.size:
        axes[4].legend()

    if combined_boxes.size:
        train_area = train_boxes[:, 2] * train_boxes[:, 3] if train_boxes.size else np.array([])
        val_area = val_boxes[:, 2] * val_boxes[:, 3] if val_boxes.size else np.array([])
        train_aspect = train_boxes[:, 2] / np.clip(train_boxes[:, 3], 1e-6, None) if train_boxes.size else np.array([])
        val_aspect = val_boxes[:, 2] / np.clip(val_boxes[:, 3], 1e-6, None) if val_boxes.size else np.array([])
        train_aspect = np.clip(train_aspect, 0.0, np.percentile(train_aspect, 99.0)) if train_aspect.size else train_aspect
        val_aspect = np.clip(val_aspect, 0.0, np.percentile(val_aspect, 99.0)) if val_aspect.size else val_aspect

        _plot_hist(axes[5], train_area, "train_area", "tab:blue")
        _plot_hist(axes[5], val_area, "val_area", "tab:orange")
        area_axis = axes[5]
        aspect_axis = area_axis.twinx()
        _plot_hist(aspect_axis, train_aspect, "train_aspect", "tab:green")
        _plot_hist(aspect_axis, val_aspect, "val_aspect", "tab:red")
        area_axis.set_title("BBox Area / Aspect Ratio")
        area_axis.set_xlabel("normalized value")
        area_axis.set_ylabel("area count")
        aspect_axis.set_ylabel("aspect count")
        area_handles, area_labels = area_axis.get_legend_handles_labels()
        aspect_handles, aspect_labels = aspect_axis.get_legend_handles_labels()
        area_axis.legend(area_handles + aspect_handles, area_labels + aspect_labels, fontsize=8)
    else:
        axes[5].set_title("BBox Area / Aspect Ratio")
        axes[5].text(0.5, 0.5, "No bbox data", ha="center", va="center")
        axes[5].set_axis_off()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    threshold = 0.008856
    f_xyz = np.where(xyz > threshold, np.cbrt(xyz), (7.787 * xyz) + (16.0 / 63.0))
    lab = np.empty_like(f_xyz, dtype=np.float32)
    lab[:, 0] = 116.0 * f_xyz[:, 1] - 16.0
    lab[:, 1] = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
    lab[:, 2] = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])
    return lab


def _circular_hue_mean(hue: np.ndarray) -> float:
    if hue.size == 0:
        return 0.0
    angles = np.asarray(hue, dtype=np.float32) * (2.0 * math.pi)
    return float((math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean())) / (2.0 * math.pi)) % 1.0)


def plot_dataset_color_audit(
    class_names: Sequence[str],
    train_paths: Sequence[Path],
    train_labels: Sequence[int],
    val_paths: Sequence[Path],
    val_labels: Sequence[int],
    output_path: Path,
    summary_path: Optional[Path] = None,
    max_images_per_class: int = 200,
    max_pixels_per_image: int = 2048,
    thumbnail_size: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    from matplotlib import colors as mcolors
    from PIL import Image

    rng = np.random.default_rng(int(seed))
    num_classes = len(class_names)
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, num_classes)))

    def _sample_by_class(paths: Sequence[Path], labels: Sequence[int]) -> Dict[int, List[Path]]:
        grouped: Dict[int, List[Path]] = {index: [] for index in range(num_classes)}
        for path, label in zip(paths, labels):
            label_int = int(label)
            if 0 <= label_int < num_classes:
                grouped[label_int].append(Path(path))
        sampled: Dict[int, List[Path]] = {}
        for class_index, values in grouped.items():
            if len(values) <= max_images_per_class:
                sampled[class_index] = list(values)
                continue
            indices = rng.choice(len(values), size=max_images_per_class, replace=False)
            sampled[class_index] = [values[int(index)] for index in sorted(indices.tolist())]
        return sampled

    def _collect(paths: Sequence[Path], labels: Sequence[int]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, np.ndarray]]:
        sampled = _sample_by_class(paths, labels)
        split_stats: Dict[int, Dict[str, Any]] = {}
        split_pixels: Dict[int, np.ndarray] = {}
        for class_index in range(num_classes):
            rgb_means: List[np.ndarray] = []
            pixel_chunks: List[np.ndarray] = []
            missing = 0
            for path in sampled.get(class_index, []):
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
                        array = np.asarray(image, dtype=np.float32) / 255.0
                except Exception:
                    missing += 1
                    continue
                pixels = array.reshape(-1, 3)
                if pixels.shape[0] > max_pixels_per_image:
                    indices = rng.choice(pixels.shape[0], size=max_pixels_per_image, replace=False)
                    pixels = pixels[indices]
                rgb_means.append(array.reshape(-1, 3).mean(axis=0))
                pixel_chunks.append(pixels)
            pixels_all = (
                np.concatenate(pixel_chunks, axis=0).astype(np.float32)
                if pixel_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            split_pixels[class_index] = pixels_all
            if pixels_all.size:
                hsv = mcolors.rgb_to_hsv(pixels_all)
                lab = _srgb_to_lab(pixels_all)
                rgb_mean = np.asarray(rgb_means, dtype=np.float32).mean(axis=0) if rgb_means else pixels_all.mean(axis=0)
                split_stats[class_index] = {
                    "sampled_images": len(sampled.get(class_index, [])),
                    "missing_images": int(missing),
                    "sampled_pixels": int(pixels_all.shape[0]),
                    "mean_rgb": [float(value) for value in rgb_mean.tolist()],
                    "mean_hsv": [
                        _circular_hue_mean(hsv[:, 0]),
                        float(hsv[:, 1].mean()),
                        float(hsv[:, 2].mean()),
                    ],
                    "mean_lab": [float(value) for value in lab.mean(axis=0).tolist()],
                    "std_lab": [float(value) for value in lab.std(axis=0).tolist()],
                    "mean_chroma": float(np.sqrt((lab[:, 1] ** 2) + (lab[:, 2] ** 2)).mean()),
                }
            else:
                split_stats[class_index] = {
                    "sampled_images": 0,
                    "missing_images": int(missing),
                    "sampled_pixels": 0,
                    "mean_rgb": [0.0, 0.0, 0.0],
                    "mean_hsv": [0.0, 0.0, 0.0],
                    "mean_lab": [0.0, 0.0, 0.0],
                    "std_lab": [0.0, 0.0, 0.0],
                    "mean_chroma": 0.0,
                }
        return split_stats, split_pixels

    train_stats, train_pixels = _collect(train_paths, train_labels)
    val_stats, val_pixels = _collect(val_paths, val_labels)

    delta_e: Dict[int, float] = {}
    for class_index in range(num_classes):
        train_lab = np.asarray(train_stats[class_index]["mean_lab"], dtype=np.float32)
        val_lab = np.asarray(val_stats[class_index]["mean_lab"], dtype=np.float32)
        delta_e[class_index] = float(np.linalg.norm(train_lab - val_lab))

    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    class_indices = np.arange(num_classes)

    mean_rgb = np.asarray([train_stats[index]["mean_rgb"] for index in range(num_classes)], dtype=np.float32)
    width = 0.25
    axes[0].bar(class_indices - width, mean_rgb[:, 0], width=width, label="R", color="#d95f5f")
    axes[0].bar(class_indices, mean_rgb[:, 1], width=width, label="G", color="#66a65c")
    axes[0].bar(class_indices + width, mean_rgb[:, 2], width=width, label="B", color="#5f7fd9")
    axes[0].set_title("Train Mean RGB By Class")
    axes[0].set_xticks(class_indices)
    axes[0].set_xticklabels(class_names, rotation=30, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    for class_index in range(num_classes):
        pixels = train_pixels[class_index]
        if pixels.size == 0:
            continue
        hsv = mcolors.rgb_to_hsv(pixels)
        axes[1].hist(
            hsv[:, 0],
            bins=36,
            range=(0.0, 1.0),
            histtype="step",
            density=True,
            linewidth=1.5,
            color=palette[class_index],
            label=str(class_names[class_index])[:24],
        )
        axes[2].hist(hsv[:, 1], bins=30, range=(0.0, 1.0), alpha=0.18, density=True, color=palette[class_index])
        axes[3].hist(hsv[:, 2], bins=30, range=(0.0, 1.0), alpha=0.18, density=True, color=palette[class_index])
    axes[1].set_title("Train Hue Histogram")
    axes[1].set_xlabel("Hue [0, 1]")
    axes[1].legend(fontsize=8)
    axes[2].set_title("Train Saturation Distribution")
    axes[2].set_xlabel("Saturation")
    axes[3].set_title("Train Value Distribution")
    axes[3].set_xlabel("Value")
    for axis in (axes[1], axes[2], axes[3]):
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    for class_index in range(num_classes):
        for split_name, stats, marker, alpha in (
            ("train", train_stats, "o", 0.9),
            ("val", val_stats, "x", 0.8),
        ):
            lab = np.asarray(stats[class_index]["mean_lab"], dtype=np.float32)
            axes[4].scatter(
                lab[1],
                lab[2],
                marker=marker,
                s=90,
                color=palette[class_index],
                alpha=alpha,
                label=f"{class_names[class_index][:18]} {split_name}" if split_name == "train" else None,
            )
            axes[4].annotate(str(class_index), (float(lab[1]), float(lab[2])), fontsize=8)
    axes[4].set_title("Lab a*b* Class Centroids")
    axes[4].set_xlabel("a*")
    axes[4].set_ylabel("b*")
    axes[4].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[5].bar(class_indices, [delta_e[index] for index in range(num_classes)], color=palette[:num_classes])
    axes[5].set_title("Train-Val Color Shift (Lab Delta)")
    axes[5].set_xticks(class_indices)
    axes[5].set_xticklabels(class_names, rotation=30, ha="right")
    axes[5].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary: Dict[str, Any] = {
        "max_images_per_class": int(max_images_per_class),
        "max_pixels_per_image": int(max_pixels_per_image),
        "thumbnail_size": int(thumbnail_size),
        "classes": {
            str(index): {
                "name": str(class_names[index]),
                "train": train_stats[index],
                "val": val_stats[index],
                "train_val_lab_delta": float(delta_e[index]),
            }
            for index in range(num_classes)
        },
    }
    if summary_path is not None:
        json_dump(summary_path, summary)
    return summary


def summarize_token_norms(features: Dict[str, torch.Tensor]) -> Dict[str, float]:
    if "patches" not in features or "registers" not in features:
        return {}

    patch_tokens = features["patches"].detach()
    register_tokens = features["registers"].detach()
    if patch_tokens.ndim < 3 or register_tokens.ndim < 3:
        return {}

    patch_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-2], patch_tokens.shape[-1])
    register_tokens = register_tokens.reshape(-1, register_tokens.shape[-2], register_tokens.shape[-1])

    patch_norms = patch_tokens.norm(dim=-1)
    patch_mean = float(patch_norms.mean().item())
    patch_std = float(patch_norms.std(unbiased=False).item())
    patch_max = float(patch_norms.max().item())
    high_threshold = patch_mean + 2.0 * patch_std
    high_fraction = float((patch_norms > high_threshold).float().mean().item())

    if register_tokens.numel() > 0:
        register_norms = register_tokens.norm(dim=-1)
        register_mean = float(register_norms.mean().item())
        register_max = float(register_norms.max().item())
    else:
        register_mean = 0.0
        register_max = 0.0

    ratio = register_mean / max(patch_mean, 1e-6)
    return {
        "patch_norm_mean": patch_mean,
        "patch_norm_std": patch_std,
        "patch_norm_max": patch_max,
        "register_norm_mean": register_mean,
        "register_norm_max": register_max,
        "register_to_patch_ratio": ratio,
        "high_norm_patch_fraction": high_fraction,
    }
