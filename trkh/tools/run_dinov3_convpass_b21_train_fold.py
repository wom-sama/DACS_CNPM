from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from trkh.core.utils import (
    build_optimizer_param_groups,
    build_warmup_decay_scheduler,
)
from trkh.data.dataset import (
    ClassificationFolderDataset,
    TemperedClassBatchSampler,
    build_eval_transform,
    build_train_transform,
)
from trkh.models.dinov3_multidepth_convpass_b21 import (
    ADDED_PARAMETER_COUNT,
    DIRECT_MODE,
    SPATIAL_MODE,
    DinoV3MultiDepthConvPassB21,
)
from trkh.models.pretrained_semantic_branch import (
    load_verified_local_timm_model,
)
from trkh.recipes.pretrained_classf_b0 import (
    DINO_MODEL_NAME,
    DINO_SHA256,
    DINO_SOURCE_LICENSE,
    DINO_SOURCE_REVISION,
    DINO_SOURCE_URL,
)
from trkh.training.losses import LDAMFocalLoss


PROTOCOL_ID = "TRKH_B21_DINOV3_MULTIDEPTH_CONVPASS_FOLD0_20260805"
SEED = 20260805
EXPECTED_DATA_YAML = Path(r"D:\DataAI\AIEx\newdataset\class_f\data.yaml")
EXPECTED_DATA_SHA256 = "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8"
EXPECTED_ASSIGNMENT_SHA256 = "afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f"
EXPECTED_DINO_BYTES = 86_362_376
EXPECTED_ROWS = 8_278
EXPECTED_CLASS_COUNTS = (1_987, 497, 1_326, 2_080, 2_388)
EXPECTED_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
EXPECTED_MODEL_PARAMETERS = 21_759_749
FOLD = 0
EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = 3
WORKERS = 2
TASK_LR = 1.5e-4
BACKBONE_LR_SCALE = 0.1
WEIGHT_DECAY = 0.05
GRAD_CLIP = 0.7
EMA_DECAY = 0.995
WARMUP_EPOCHS = 2
SCHEDULER_HORIZON = 30
MIN_LR = 1e-6
MAX_RESIDUAL_P95 = 0.1
MIN_RESIDUAL_P95 = 1e-4
ARMS = (DIRECT_MODE, SPATIAL_MODE)
RIVALS = (0, 2, 4)


class B21ContractError(RuntimeError):
    """Raised when a prospective B21 input or invariant has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_absolute_path(path: Path) -> Path:
    """Return an absolute path without resolving a format-bearing symlink."""

    logical = Path(path).expanduser()
    if not logical.is_absolute():
        logical = Path.cwd() / logical
    logical = Path(os.path.abspath(os.fspath(logical)))
    if not logical.is_file():
        raise FileNotFoundError(logical)
    return logical


def state_sha256(state: Mapping[str, Tensor], *, exclude_adapters: bool = False) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if exclude_adapters and (".adapter_attn." in name or ".adapter_mlp." in name):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_snapshot(repo: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            encoding="utf-8",
        ).strip()

    branch = git("branch", "--show-current")
    status = git("status", "--short")
    if not branch.startswith("research/pretrained-"):
        raise B21ContractError(f"B21 requires a pretrained research branch, got {branch!r}.")
    if status:
        raise B21ContractError("B21 formal execution requires a clean committed worktree.")
    return {"branch": branch, "head": git("rev-parse", "HEAD"), "clean": True}


def _source_hashes(repo: Path) -> Dict[str, str]:
    relative_paths = {
        "model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "runner": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "model_test": Path("tests/test_dinov3_multidepth_convpass_b21.py"),
        "runner_test": Path("tests/test_run_dinov3_convpass_b21_train_fold.py"),
    }
    return {name: sha256_file(repo / path) for name, path in relative_paths.items()}


def read_accepted_preflight(path: Path, *, repo: Path, git: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("passed") is not True
        or payload.get("train") is not False
        or payload.get("validation") is not False
        or payload.get("test") is not False
        or payload.get("git") != dict(git)
        or payload.get("source_hashes") != _source_hashes(repo)
    ):
        raise B21ContractError("Preflight is not an accepted same-commit B21 artifact.")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seed_worker(_worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _cpu_seed_state(seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator.get_state()


def build_arm_model(dino_weight: Path, mode: str) -> DinoV3MultiDepthConvPassB21:
    """Build one arm without changing the caller's CPU or CUDA RNG."""

    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(_cpu_seed_state(SEED))
        backbone, provenance = load_verified_local_timm_model(
            model_name=DINO_MODEL_NAME,
            checkpoint_path=Path(dino_weight),
            expected_sha256=DINO_SHA256,
            source_repository=DINO_SOURCE_URL,
            source_revision=DINO_SOURCE_REVISION,
            license_id=DINO_SOURCE_LICENSE,
            strict=True,
        )
        backbone.reset_classifier(len(EXPECTED_CLASS_NAMES))
        model = DinoV3MultiDepthConvPassB21(
            backbone,
            num_classes=len(EXPECTED_CLASS_NAMES),
            mode=mode,
            externally_pretrained=True,
            source_provenance=provenance,
        )
    model.model_type = "dinov3_multidepth_convpass_b21"
    model.research_track = "pretrained"
    return model


def read_locked_assignment(path: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if sha256_file(resolved) != EXPECTED_ASSIGNMENT_SHA256:
        raise B21ContractError("TRAIN component assignment SHA-256 changed.")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != [
            "relative_path",
            "label",
            "union_group",
            "fold",
        ]:
            raise B21ContractError("TRAIN component assignment schema changed.")
        rows = list(reader)
    paths = [str(row["relative_path"]).replace("\\", "/") for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([int(row["union_group"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if len(rows) != EXPECTED_ROWS:
        raise B21ContractError(f"TRAIN row count changed: {len(rows)} != {EXPECTED_ROWS}.")
    if tuple(np.bincount(labels, minlength=5).tolist()) != EXPECTED_CLASS_COUNTS:
        raise B21ContractError("TRAIN class counts changed.")
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise B21ContractError("Assignment must contain folds 0..4.")
    for fold in range(5):
        held = folds == fold
        if set(groups[held].tolist()) & set(groups[~held].tolist()):
            raise B21ContractError(f"Fold {fold} is not union-component disjoint.")
        if bool((np.bincount(labels[held], minlength=5) == 0).any()):
            raise B21ContractError(f"Fold {fold} is not class complete.")
    return {
        "path": resolved,
        "relative_paths": paths,
        "labels": labels,
        "groups": groups,
        "folds": folds,
    }


def read_locked_train_root(data_yaml: Path) -> Path:
    resolved = Path(data_yaml).expanduser().resolve(strict=True)
    if resolved != EXPECTED_DATA_YAML.resolve() or sha256_file(resolved) != EXPECTED_DATA_SHA256:
        raise B21ContractError("Canonical class_f data YAML changed.")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, Mapping):
        class_names = tuple(str(names[key]) for key in sorted(names, key=lambda item: int(item)))
    else:
        class_names = tuple(str(value) for value in names or ())
    if str(payload.get("format", "")).strip() != "classification_folder":
        raise B21ContractError("class_f format changed.")
    if class_names != EXPECTED_CLASS_NAMES:
        raise B21ContractError("class_f class order changed.")
    configured_root = Path(str(payload.get("path", ".")))
    root = configured_root if configured_root.is_absolute() else resolved.parent / configured_root
    train_value = Path(str(payload.get("train", "train")))
    train_root = train_value if train_value.is_absolute() else root / train_value
    return train_root.resolve(strict=True)


def build_transforms() -> tuple[Any, Any]:
    train_transform = build_train_transform(
        image_size=256,
        resize_mode="pad",
        scale_min=0.90,
        scale_crop_probability=0.30,
        brightness=0.05,
        contrast=0.05,
        saturation=0.03,
        hue=0.01,
        random_erasing_probability=0.0,
        random_affine_degrees=3.0,
        random_affine_translate=0.02,
        random_affine_scale_min=0.96,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.0,
        rotate90_probability=0.03,
        lighting_probability=0.0,
        randaugment_num_ops=0,
        randaugment_magnitude=0,
        local_exposure_probability=0.15,
        local_exposure_strength=0.20,
        obstacle_probability=0.03,
        obstacle_max_area=0.06,
    )
    return train_transform, build_eval_transform(image_size=256, resize_mode="pad")


def map_dataset_indices(
    dataset: ClassificationFolderDataset,
    *,
    data_root: Path,
    assignment_paths: Sequence[str],
    assignment_labels: np.ndarray,
) -> list[int]:
    index_by_relative: Dict[str, int] = {}
    dataset_labels = dataset.labels()
    for index, path in enumerate(dataset.sample_paths()):
        relative = path.resolve().relative_to(data_root.resolve()).as_posix()
        if relative in index_by_relative:
            raise B21ContractError(f"Duplicate TRAIN relative path: {relative}.")
        index_by_relative[relative] = index
    if set(index_by_relative) != set(assignment_paths):
        missing = sorted(set(assignment_paths) - set(index_by_relative))[:5]
        extra = sorted(set(index_by_relative) - set(assignment_paths))[:5]
        raise B21ContractError(f"Dataset/assignment paths differ; missing={missing}, extra={extra}.")
    indices = [index_by_relative[path] for path in assignment_paths]
    observed_labels = np.asarray([dataset_labels[index] for index in indices], dtype=np.int64)
    if not np.array_equal(observed_labels, assignment_labels):
        raise B21ContractError("Dataset folder labels do not match the locked assignment.")
    return indices


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.is_floating_point():
                target.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                target.copy_(value)


def parameter_role(name: str) -> str:
    if ".adapter_attn." in name or ".adapter_mlp." in name:
        return "adapter"
    if name.startswith("backbone.head."):
        return "head"
    return "backbone"


def parameter_snapshot(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_ratio_by_role(model: nn.Module, before: Mapping[str, Tensor]) -> Dict[str, float]:
    delta_sq = {role: 0.0 for role in ("backbone", "head", "adapter")}
    parameter_sq = {role: 0.0 for role in delta_sq}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        role = parameter_role(name)
        previous = before[name]
        current = parameter.detach().float().cpu()
        delta_sq[role] += float((current - previous).square().sum().item())
        parameter_sq[role] += float(previous.square().sum().item())
    return {
        role: math.sqrt(delta_sq[role]) / max(math.sqrt(parameter_sq[role]), 1e-12)
        for role in delta_sq
    }


def gradient_norm_by_role(model: nn.Module) -> Dict[str, float]:
    totals = {role: 0.0 for role in ("backbone", "head", "adapter")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        totals[parameter_role(name)] += float(parameter.grad.detach().float().square().sum().item())
    return {role: math.sqrt(value) for role, value in totals.items()}


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, Any]:
    true = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if true.shape != pred.shape or true.ndim != 1 or true.size == 0:
        raise ValueError("labels and predictions must be aligned non-empty vectors.")
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    per_class = []
    for class_index in range(5):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class_index": class_index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[class_index, :].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    transitions = {f"{source}->1": int(confusion[source, 1]) for source in range(5) if source != 1}
    return {
        "accuracy": float(np.trace(confusion) / true.size),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "class1": dict(per_class[1]),
        "restricted_fp_into_class1": int(sum(confusion[source, 1] for source in RIVALS)),
        "transitions_into_class1": transitions,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def promotion_gate(
    direct: Mapping[str, Any],
    spatial: Mapping[str, Any],
    *,
    residual_p95: float,
    finite_updates: bool,
) -> Dict[str, Any]:
    direct_c1 = direct["class1"]
    spatial_c1 = spatial["class1"]
    delta_c1 = float(spatial_c1["f1"] - direct_c1["f1"])
    delta_macro = float(spatial["macro_f1"] - direct["macro_f1"])
    direct_fp = int(direct["restricted_fp_into_class1"])
    spatial_fp = int(spatial["restricted_fp_into_class1"])
    fp_reduction = float((direct_fp - spatial_fp) / max(direct_fp, 1))
    tp_retention = float(spatial_c1["tp"] / max(int(direct_c1["tp"]), 1))
    quality_signal = bool(delta_c1 >= 0.005 or (fp_reduction >= 0.05 and tp_retention >= 0.98))
    checks = {
        "quality_signal": quality_signal,
        "macro_noninferiority": bool(delta_macro >= -0.002),
        "two_to_one_nonincrease": bool(
            spatial["transitions_into_class1"]["2->1"]
            <= direct["transitions_into_class1"]["2->1"]
        ),
        "branch_active_bounded": bool(
            math.isfinite(float(residual_p95))
            and MIN_RESIDUAL_P95 <= float(residual_p95) < MAX_RESIDUAL_P95
        ),
        "finite_updates": bool(finite_updates),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "deltas": {
            "class1_f1": delta_c1,
            "macro_f1": delta_macro,
            "restricted_fp_reduction_fraction": fp_reduction,
            "class1_tp_retention": tp_retention,
            "two_to_one": int(
                spatial["transitions_into_class1"]["2->1"]
                - direct["transitions_into_class1"]["2->1"]
            ),
        },
        "meaning": (
            "pass_authorizes_fixed_train_oof_with_dephased_control_only"
            if all(checks.values())
            else "fail_closes_exact_b21_a0_without_validation_or_test"
        ),
    }


def _split_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise B21ContractError("Unexpected classification batch schema.")
    return batch[0], torch.as_tensor(batch[1], dtype=torch.long)


def _build_datasets(
    train_root: Path,
    assignment: Mapping[str, Any],
) -> tuple[Subset, Subset, list[str], list[int]]:
    train_transform, eval_transform = build_transforms()
    train_dataset = ClassificationFolderDataset(
        train_root,
        EXPECTED_CLASS_NAMES,
        transform=train_transform,
        split="train",
    )
    eval_dataset = ClassificationFolderDataset(
        train_root,
        EXPECTED_CLASS_NAMES,
        transform=eval_transform,
        split="train_held",
    )
    data_root = train_root.parent
    assignment_indices = map_dataset_indices(
        train_dataset,
        data_root=data_root,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    eval_assignment_indices = map_dataset_indices(
        eval_dataset,
        data_root=data_root,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    folds = assignment["folds"]
    fit_positions = np.flatnonzero(folds != FOLD).tolist()
    held_positions = np.flatnonzero(folds == FOLD).tolist()
    fit_indices = [assignment_indices[position] for position in fit_positions]
    held_indices = [eval_assignment_indices[position] for position in held_positions]
    fit_labels = [int(assignment["labels"][position]) for position in fit_positions]
    held_paths = [assignment["relative_paths"][position] for position in held_positions]
    return Subset(train_dataset, fit_indices), Subset(eval_dataset, held_indices), held_paths, fit_labels


def _train_arm(
    *,
    mode: str,
    dino_weight: Path,
    fit_dataset: Subset,
    held_dataset: Subset,
    held_paths: Sequence[str],
    fit_labels: Sequence[int],
    output_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    _seed_all(SEED)
    sampler = TemperedClassBatchSampler(
        fit_labels,
        batch_size=BATCH_SIZE,
        num_classes=5,
        power=0.5,
        epoch_multiplier=1.0,
        seed=SEED,
        drop_last=False,
    )
    train_loader = DataLoader(
        fit_dataset,
        batch_sampler=sampler,
        num_workers=WORKERS,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        persistent_workers=WORKERS > 0,
        prefetch_factor=2 if WORKERS > 0 else None,
    )
    held_loader = DataLoader(
        held_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(SEED + 1),
        persistent_workers=False,
        prefetch_factor=2 if WORKERS > 0 else None,
    )
    fit_counts = np.bincount(np.asarray(fit_labels), minlength=5).astype(int).tolist()
    criterion = LDAMFocalLoss(
        class_counts=fit_counts,
        gamma=1.0,
        focal_mix=0.1,
        label_smoothing=0.02,
        max_margin=0.3,
        scale=18.0,
    ).to(device)
    model = build_arm_model(dino_weight, mode)
    model.set_mode(mode)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_MODEL_PARAMETERS or model.added_parameter_count() != ADDED_PARAMETER_COUNT:
        raise B21ContractError("B21 parameter contract changed.")
    initial_primary_sha = state_sha256(model.state_dict(), exclude_adapters=True)
    model.to(device).train()
    optimizer_groups = build_optimizer_param_groups(
        model,
        weight_decay=WEIGHT_DECAY,
        learning_rate=TASK_LR,
        backbone_lr_scale=BACKBONE_LR_SCALE,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=TASK_LR,
        betas=(0.9, 0.999),
    )
    scheduler = build_warmup_decay_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_start_factor=0.1,
        total_epochs=SCHEDULER_HORIZON,
        min_learning_rate=MIN_LR,
        decay_style="cosine",
    )
    ema = ModelEMA(model, EMA_DECAY)
    epoch_rows = []
    total_updates = 0
    finite_updates = True
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        before = parameter_snapshot(model)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        sample_count = 0
        representative_grad = {role: 0.0 for role in ("backbone", "head", "adapter")}
        for batch_index, batch in enumerate(train_loader):
            images, labels = _split_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss).item()):
                finite_updates = False
                raise B21ContractError(f"Non-finite loss: arm={mode} epoch={epoch + 1}.")
            (loss / ACCUMULATION_STEPS).backward()
            should_step = (
                (batch_index + 1) % ACCUMULATION_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                if batch_index + 1 == len(train_loader):
                    representative_grad = gradient_norm_by_role(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if not bool(torch.isfinite(grad_norm).item()):
                    finite_updates = False
                    raise B21ContractError(f"Non-finite gradient: arm={mode} epoch={epoch + 1}.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                total_updates += 1
                progress = epoch + float(batch_index + 1) / max(len(train_loader), 1)
                scheduler.step(progress)
            batch_size = int(labels.numel())
            loss_sum += float(loss.detach().float().item()) * batch_size
            sample_count += batch_size
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(train_loader):
                _atomic_json(
                    output_dir / "progress.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "mode": mode,
                        "epoch": epoch + 1,
                        "epochs": EPOCHS,
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "loss": float(loss.detach().float()),
                        "wall_seconds": time.monotonic() - started,
                    },
                )
                print(
                    f"{mode} epoch={epoch + 1}/{EPOCHS} "
                    f"batch={batch_index + 1}/{len(train_loader)} "
                    f"loss={float(loss.detach().float()):.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "mean_loss": loss_sum / max(sample_count, 1),
                "optimizer_updates_total": total_updates,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "representative_preclip_gradient_norm": representative_grad,
                "update_over_parameter_norm": update_ratio_by_role(model, before),
            }
        )
        del before

    evaluation_model = ema.module.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    logits_all: list[np.ndarray] = []
    first_images: Tensor | None = None
    with torch.inference_mode():
        for batch in held_loader:
            images, labels = _split_batch(batch)
            if first_images is None:
                first_images = images.clone()
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = evaluation_model(images)
            logits_cpu = logits.float().cpu()
            labels_all.extend(int(value) for value in labels.tolist())
            predictions_all.extend(int(value) for value in logits_cpu.argmax(dim=1).tolist())
            logits_all.append(logits_cpu.numpy())
    metrics = classification_metrics(labels_all, predictions_all)
    logits_matrix = np.concatenate(logits_all, axis=0)
    residual_summary: Dict[str, float] = {
        "p50": 0.0,
        "p95": 0.0,
        "max": 0.0,
    }
    if mode == SPATIAL_MODE and first_images is not None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, trace = evaluation_model.forward_with_adapter_trace(first_images.to(device))
        ratios = trace["all_residual_ratio"].float().cpu()
        residual_summary = {
            "p50": float(torch.quantile(ratios, 0.50).item()),
            "p95": float(torch.quantile(ratios, 0.95).item()),
            "max": float(ratios.max().item()),
        }
    checkpoint: Dict[str, Any] | None = None
    first_logits = logits_matrix[: min(EVAL_BATCH_SIZE, len(logits_matrix))]
    if mode == SPATIAL_MODE:
        checkpoint_path = output_dir / "spatial_ema.safetensors"
        cpu_state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in evaluation_model.state_dict().items()
        }
        save_file(
            cpu_state,
            str(checkpoint_path),
            metadata={
                "protocol_id": PROTOCOL_ID,
                "mode": SPATIAL_MODE,
                "research_track": "pretrained",
                "dino_sha256": DINO_SHA256,
            },
        )
        checkpoint = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "state_sha256": state_sha256(cpu_state),
            "mode": SPATIAL_MODE,
        }
        del cpu_state
    result = {
        "mode": mode,
        "metrics": metrics,
        "residual_ratio": residual_summary,
        "epochs": epoch_rows,
        "optimizer_updates": total_updates,
        "finite_updates": finite_updates,
        "initial_primary_state_sha256": initial_primary_sha,
        "parameter_count": parameter_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.monotonic() - started,
        "held_paths": list(held_paths),
        "held_labels": labels_all,
        "held_predictions": predictions_all,
        "held_logits": logits_matrix,
        "first_logits": first_logits,
        "checkpoint": checkpoint,
    }
    del model, ema, evaluation_model, optimizer, scheduler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _verify_spatial_reload(
    *,
    dino_weight: Path,
    checkpoint: Mapping[str, Any],
    first_images: Tensor,
    expected_logits: np.ndarray,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    model = build_arm_model(dino_weight, SPATIAL_MODE)
    model.set_mode(SPATIAL_MODE)
    state = load_file(str(checkpoint["path"]), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise B21ContractError("Spatial EMA strict reload failed.")
    observed_state_sha = state_sha256(model.state_dict())
    if observed_state_sha != checkpoint["state_sha256"]:
        raise B21ContractError("Spatial EMA state hash changed after strict reload.")
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        observed = model(first_images.to(device)).float().cpu().numpy()
    expected = np.asarray(expected_logits, dtype=np.float32)
    exact = bool(np.array_equal(observed, expected))
    max_abs = float(np.max(np.abs(observed - expected)))
    if not exact:
        raise B21ContractError(f"Spatial EMA reload logits changed; max_abs={max_abs}.")
    del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return {"strict": True, "exact_bf16_logits": True, "max_abs": max_abs}


def run_preflight(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    git = _git_snapshot(repo)
    assignment = read_locked_assignment(args.assignment_csv)
    train_root = read_locked_train_root(args.data)
    _, eval_transform = build_transforms()
    dataset = ClassificationFolderDataset(
        train_root,
        EXPECTED_CLASS_NAMES,
        transform=eval_transform,
        split="train_preflight",
    )
    indices = map_dataset_indices(
        dataset,
        data_root=train_root.parent,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=assignment["labels"],
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise B21ContractError("B21 requires a CUDA GPU with BF16 support.")
    torch.cuda.init()
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    model = build_arm_model(args.dino_weight, SPATIAL_MODE)
    rng_preserved = bool(
        torch.equal(cpu_before, torch.random.get_rng_state())
        and all(
            torch.equal(left, right)
            for left, right in zip(cuda_before, torch.cuda.get_rng_state_all())
        )
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(_cpu_seed_state(SEED))
        native, _ = load_verified_local_timm_model(
            model_name=DINO_MODEL_NAME,
            checkpoint_path=args.dino_weight,
            expected_sha256=DINO_SHA256,
            source_repository=DINO_SOURCE_URL,
            source_revision=DINO_SOURCE_REVISION,
            license_id=DINO_SOURCE_LICENSE,
            strict=True,
        )
        native.reset_classifier(5)
    native.eval()
    model.eval()
    probe = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        native_logits = native(probe)
        spatial_logits = model(probe)
        model.set_mode(DIRECT_MODE)
        direct_logits = model(probe)
    identity = {
        "spatial_exact_native": bool(torch.equal(spatial_logits, native_logits)),
        "direct_exact_native": bool(torch.equal(direct_logits, native_logits)),
        "spatial_max_abs": float((spatial_logits - native_logits).abs().max().item()),
        "direct_max_abs": float((direct_logits - native_logits).abs().max().item()),
    }
    del native, native_logits, spatial_logits, direct_logits
    model.set_mode(SPATIAL_MODE)
    model.train().cuda()
    probe = probe.cuda()
    target = torch.tensor([1], device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first_loss = F.cross_entropy(model(probe), target)
    first_loss.backward()
    first_up = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if ".adapter_" in name and name.endswith("up.weight") and parameter.grad is not None
    )
    first_conv = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if ".adapter_" in name and name.endswith("conv.weight") and parameter.grad is not None
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        second_loss = F.cross_entropy(model(probe), target)
    second_loss.backward()
    second_down = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if ".adapter_" in name and name.endswith("down.weight") and parameter.grad is not None
    )
    second_conv = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
        if ".adapter_" in name and name.endswith("conv.weight") and parameter.grad is not None
    )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": git,
        "source_hashes": _source_hashes(repo),
        "mode": "preflight_only",
        "train": False,
        "validation": False,
        "test": False,
        "dataset_rows_mapped": len(indices),
        "rng_preserved_after_construction": rng_preserved,
        "identity": identity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "added_parameter_count": model.added_parameter_count(),
        "gradient_reachability": {
            "first_up_nonzero": first_up,
            "first_conv_nonzero_expected_zero": first_conv,
            "second_down_nonzero": second_down,
            "second_conv_nonzero": second_conv,
        },
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "passed": bool(
            rng_preserved
            and all(identity[key] for key in ("spatial_exact_native", "direct_exact_native"))
            and first_up > 0
            and first_conv == 0
            and second_down > 0
            and second_conv > 0
        ),
    }
    _atomic_json(output / "preflight.json", payload)
    del model, optimizer, probe
    gc.collect()
    torch.cuda.empty_cache()
    if not payload["passed"]:
        raise B21ContractError("B21 preflight failed.")
    return payload


def _sanitize_arm_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"held_logits", "first_logits", "held_paths", "held_labels", "held_predictions"}
    }


def run_training(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    git = _git_snapshot(repo)
    accepted_preflight = read_accepted_preflight(
        args.preflight_artifact,
        repo=repo,
        git=git,
    )
    assignment = read_locked_assignment(args.assignment_csv)
    train_root = read_locked_train_root(args.data)
    fit_dataset, held_dataset, held_paths, fit_labels = _build_datasets(train_root, assignment)
    contract = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": git,
        "accepted_preflight": accepted_preflight,
        "data_yaml": str(Path(args.data).resolve()),
        "data_yaml_sha256": EXPECTED_DATA_SHA256,
        "assignment_csv": str(Path(args.assignment_csv).resolve()),
        "assignment_csv_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "dino_weight": str(logical_absolute_path(args.dino_weight)),
        "dino_weight_sha256": DINO_SHA256,
        "fold": FOLD,
        "fit_rows": len(fit_dataset),
        "held_rows": len(held_dataset),
        "arms": list(ARMS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
        "data_loader_workers": WORKERS,
        "worker_seed_policy": "torch_initial_seed_to_python_and_numpy",
        "scheduler_horizon": SCHEDULER_HORIZON,
        "train": True,
        "validation": False,
        "test": False,
    }
    _atomic_json(output / "contract.json", contract)
    results: Dict[str, Dict[str, Any]] = {}
    for mode in ARMS:
        results[mode] = _train_arm(
            mode=mode,
            dino_weight=args.dino_weight,
            fit_dataset=fit_dataset,
            held_dataset=held_dataset,
            held_paths=held_paths,
            fit_labels=fit_labels,
            output_dir=output,
        )
        _atomic_json(output / f"{mode}.json", _sanitize_arm_result(results[mode]))
    if results[DIRECT_MODE]["initial_primary_state_sha256"] != results[SPATIAL_MODE]["initial_primary_state_sha256"]:
        raise B21ContractError("Direct/spatial primary initial states differ.")
    checkpoint = results[SPATIAL_MODE]["checkpoint"]
    if not isinstance(checkpoint, Mapping):
        raise B21ContractError("Spatial EMA checkpoint was not written.")
    # Recreate the exact first held batch without constructing another split.
    first_images, _ = _split_batch(next(iter(DataLoader(
        held_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    ))))
    reload_verification = _verify_spatial_reload(
        dino_weight=args.dino_weight,
        checkpoint=checkpoint,
        first_images=first_images,
        expected_logits=results[SPATIAL_MODE]["first_logits"],
    )
    gate = promotion_gate(
        results[DIRECT_MODE]["metrics"],
        results[SPATIAL_MODE]["metrics"],
        residual_p95=results[SPATIAL_MODE]["residual_ratio"]["p95"],
        finite_updates=bool(
            results[DIRECT_MODE]["finite_updates"] and results[SPATIAL_MODE]["finite_updates"]
        ),
    )
    predictions_path = output / "held_train_fold0_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "label", "direct_prediction", "spatial_prediction"])
        for row in zip(
            held_paths,
            results[DIRECT_MODE]["held_labels"],
            results[DIRECT_MODE]["held_predictions"],
            results[SPATIAL_MODE]["held_predictions"],
        ):
            writer.writerow(row)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "contract": contract,
        "direct": _sanitize_arm_result(results[DIRECT_MODE]),
        "spatial": _sanitize_arm_result(results[SPATIAL_MODE]),
        "promotion_gate": gate,
        "reload_verification": reload_verification,
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": sha256_file(predictions_path),
        },
        "validation_opened": False,
        "test_opened": False,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked B21 direct-vs-multi-depth ConvPass TRAIN-fold screen."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=EXPECTED_DATA_YAML)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--preflight-artifact", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.data = Path(args.data).expanduser().resolve(strict=True)
    args.assignment_csv = Path(args.assignment_csv).expanduser().resolve(strict=True)
    # Keep the logical `.safetensors` snapshot path. Resolving its symlink to
    # the extensionless Hugging Face blob would select the torch.load parser.
    args.dino_weight = logical_absolute_path(args.dino_weight)
    if not args.preflight_only:
        if args.preflight_artifact is None:
            raise B21ContractError("Formal TRAIN requires --preflight-artifact.")
        args.preflight_artifact = Path(args.preflight_artifact).expanduser().resolve(strict=True)
    if args.dino_weight.stat().st_size != EXPECTED_DINO_BYTES or sha256_file(args.dino_weight) != DINO_SHA256:
        raise B21ContractError("DINOv3 weight file changed.")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _configure_determinism()
    if args.preflight_only:
        payload = run_preflight(args, output)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        summary = run_training(args, output)
        print(json.dumps({"promotion_gate": summary["promotion_gate"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
