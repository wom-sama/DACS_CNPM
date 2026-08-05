from __future__ import annotations

import argparse
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
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import build_warmup_decay_scheduler
from trkh.data.dataset import (
    ClassificationFolderDataset,
    TemperedClassBatchSampler,
)
from trkh.models.dinov3_multidepth_convpass_b21 import (
    ADDED_PARAMETER_COUNT,
    SPATIAL_MODE,
    DinoV3MultiDepthConvPassB21,
)
from trkh.models.model import build_model_from_checkpoint
from trkh.models.supervised_guarded_convpass_b23 import (
    adapter_state_dict,
    configure_adapter_only,
    guarded_adapter_loss,
    is_adapter_parameter,
    load_adapter_state_dict,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SELECTED_STATE_SHA256,
    _state_dict_sha256,
    _validate_b9_checkpoint,
)
from trkh.tools.run_dinov3_convpass_b21_train_fold import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_CLASS_NAMES,
    EXPECTED_DATA_SHA256,
    EXPECTED_DATA_YAML,
    build_transforms,
    classification_metrics,
)
from trkh.training.losses import LDAMFocalLoss


PROTOCOL_ID = "TRKH_B23_B9_GUARDED_CONVPASS_SPATIAL_20260805"
SEED = 20260805
EXPECTED_TRAIN_ROWS = 8_278
EXPECTED_VAL_ROWS = 2_479
EXPECTED_VAL_CLASS_COUNTS = (558, 158, 380, 494, 889)
EXPECTED_B9_CONFUSION = (
    (530, 21, 0, 1, 6),
    (21, 122, 14, 0, 1),
    (0, 41, 328, 8, 3),
    (0, 0, 14, 420, 60),
    (3, 6, 8, 11, 861),
)
EXPECTED_TOTAL_PARAMETERS = 21_759_749

EPOCHS = 6
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = 3
WORKERS = 2
ADAPTER_LR = 1.5e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 0.7
EMA_DECAY = 0.995
WARMUP_EPOCHS = 1
MIN_LR = 1e-6
RETENTION_WEIGHT = 0.25
DISTILLATION_WEIGHT = 0.10
RETENTION_SLACK = 0.05
DISTILLATION_TEMPERATURE = 2.0
RIVALS = (0, 2, 4)

DEFAULT_B9_CHECKPOINT = Path(
    r"D:\DataAI\AIEx\TRKH_pretrained\runs\pretrained_dinov3_classf_b9_b2_reference_completion_full_20260801_b9_b2_ref_full_r1\checkpoints\best.pt"
)


class B23ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
            ["git", *args], cwd=repo, text=True, encoding="utf-8"
        ).strip()

    branch = git("branch", "--show-current")
    status = git("status", "--short")
    if not branch.startswith("research/pretrained-"):
        raise B23ContractError(f"B23 requires a pretrained branch, got {branch!r}.")
    if status:
        raise B23ContractError("B23 requires a clean committed worktree.")
    return {"branch": branch, "head": git("rev-parse", "HEAD"), "clean": True}


def _source_hashes(repo: Path) -> Dict[str, str]:
    paths = {
        "b21_model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "b23_model": Path("trkh/models/supervised_guarded_convpass_b23.py"),
        "runner": Path("trkh/tools/run_b23_b9_guarded_convpass.py"),
        "model_test": Path("tests/test_supervised_guarded_convpass_b23.py"),
        "runner_test": Path("tests/test_run_b23_b9_guarded_convpass.py"),
    }
    return {name: sha256_file(repo / path) for name, path in paths.items()}


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _seed_worker(_worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def _configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _split_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise B23ContractError("Unexpected classification batch schema.")
    return batch[0], torch.as_tensor(batch[1], dtype=torch.long)


def _load_b9(path: Path) -> tuple[Dict[str, Any], nn.Module, Dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise B23ContractError(
            f"B9 checkpoint SHA changed: {digest} != {EXPECTED_CHECKPOINT_SHA256}."
        )
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise B23ContractError("B9 checkpoint is not a dictionary.")
    contract = _validate_b9_checkpoint(checkpoint, digest)
    model = build_model_from_checkpoint(checkpoint)
    return checkpoint, model, {
        "path": str(resolved),
        "sha256": digest,
        "selected_state_sha256": contract["selected_state_sha256"],
    }


def _primary_state(model: DinoV3MultiDepthConvPassB21) -> Dict[str, Tensor]:
    state: Dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        if is_adapter_parameter(name):
            continue
        if not name.startswith("backbone."):
            raise B23ContractError(f"Unexpected B23 primary key: {name}.")
        state[name.removeprefix("backbone.")] = value
    return state


def build_b23_model(backbone: nn.Module) -> DinoV3MultiDepthConvPassB21:
    model = DinoV3MultiDepthConvPassB21(
        backbone,
        num_classes=len(EXPECTED_CLASS_NAMES),
        mode=SPATIAL_MODE,
        externally_pretrained=True,
        source_provenance=getattr(backbone, "pretrained_provenance", {}),
    )
    model.model_type = "b23_b9_guarded_convpass"
    model.research_track = "pretrained"
    counts = configure_adapter_only(model)
    if counts != {
        "total": EXPECTED_TOTAL_PARAMETERS,
        "trainable": ADDED_PARAMETER_COUNT,
        "frozen": EXPECTED_TOTAL_PARAMETERS - ADDED_PARAMETER_COUNT,
    }:
        raise B23ContractError(f"B23 parameter contract changed: {counts}.")
    primary_hash = _state_dict_sha256(_primary_state(model))
    if primary_hash != EXPECTED_SELECTED_STATE_SHA256:
        raise B23ContractError(
            "B23 did not inherit the selected B9 EMA state exactly: "
            f"{primary_hash} != {EXPECTED_SELECTED_STATE_SHA256}."
        )
    return model


def _datasets(
    data_yaml: Path,
    *,
    include_validation: bool,
) -> tuple[ClassificationFolderDataset, ClassificationFolderDataset | None]:
    resolved = Path(data_yaml).expanduser().resolve(strict=True)
    if resolved != EXPECTED_DATA_YAML.resolve() or sha256_file(resolved) != EXPECTED_DATA_SHA256:
        raise B23ContractError("Canonical class_f data contract changed.")
    spec = load_data_spec(resolved, class_name_mode="raw", expected_num_classes=5)
    if tuple(spec.class_names) != EXPECTED_CLASS_NAMES:
        raise B23ContractError("Canonical class order changed.")
    train_transform, eval_transform = build_transforms()
    train = ClassificationFolderDataset.from_data_spec(
        spec, split="train", transform=train_transform
    )
    train_counts = tuple(np.bincount(train.labels(), minlength=5).astype(int).tolist())
    if len(train) != EXPECTED_TRAIN_ROWS or train_counts != EXPECTED_CLASS_COUNTS:
        raise B23ContractError(f"TRAIN contract changed: rows={len(train)}, counts={train_counts}.")
    if not include_validation:
        return train, None
    val = ClassificationFolderDataset.from_data_spec(
        spec, split="val", transform=eval_transform
    )
    val_counts = tuple(np.bincount(val.labels(), minlength=5).astype(int).tolist())
    if len(val) != EXPECTED_VAL_ROWS or val_counts != EXPECTED_VAL_CLASS_COUNTS:
        raise B23ContractError(f"VAL contract changed: rows={len(val)}, counts={val_counts}.")
    return train, val


def _train_loader(dataset: ClassificationFolderDataset) -> tuple[DataLoader, TemperedClassBatchSampler]:
    sampler = TemperedClassBatchSampler(
        dataset.labels(),
        batch_size=BATCH_SIZE,
        num_classes=5,
        power=0.5,
        epoch_multiplier=1.0,
        seed=SEED,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=WORKERS,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        persistent_workers=WORKERS > 0,
        prefetch_factor=2 if WORKERS > 0 else None,
    )
    return loader, sampler


def _eval_loader(dataset: ClassificationFolderDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator(device="cpu").manual_seed(SEED + 1),
        persistent_workers=False,
        prefetch_factor=2 if WORKERS > 0 else None,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    logits_all: list[np.ndarray] = []
    first_images: Tensor | None = None
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            images, labels = _split_batch(batch)
            if first_images is None:
                first_images = images.clone()
            logits = model(images.to(device, non_blocking=True)).float().cpu()
            labels_all.extend(int(value) for value in labels.tolist())
            predictions_all.extend(int(value) for value in logits.argmax(dim=1).tolist())
            logits_all.append(logits.numpy())
    matrix = np.concatenate(logits_all, axis=0)
    return {
        "metrics": classification_metrics(labels_all, predictions_all),
        "labels": labels_all,
        "predictions": predictions_all,
        "logits": matrix,
        "first_images": first_images,
        "wall_seconds": time.monotonic() - started,
    }


class AdapterEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        state = adapter_state_dict(model)
        self.shadow = {
            name: value for name, value in state.items() if value.is_floating_point()
        }
        self.static = {
            name: value for name, value in state.items() if not value.is_floating_point()
        }
        if not self.shadow or not self.static:
            raise B23ContractError("B23 EMA expected floating parameters and integer topology buffers.")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        current = dict(model.state_dict())
        for name, shadow in self.shadow.items():
            value = current[name].detach().cpu()
            shadow.mul_(decay).add_(value, alpha=1.0 - decay)
        for name, expected in self.static.items():
            if not torch.equal(current[name].detach().cpu(), expected):
                raise B23ContractError(f"B23 static adapter buffer changed: {name}.")

    def copy_to(self, model: nn.Module) -> None:
        load_adapter_state_dict(model, {**self.static, **self.shadow})


def _adapter_norm(model: nn.Module) -> float:
    value = sum(
        float(parameter.detach().float().square().sum().item())
        for name, parameter in model.named_parameters()
        if is_adapter_parameter(name)
    )
    return math.sqrt(value)


def _adapter_gradient_norm(model: nn.Module) -> float:
    value = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for name, parameter in model.named_parameters()
        if is_adapter_parameter(name) and parameter.grad is not None
    )
    return math.sqrt(value)


def _promotion_gate(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    base_c1 = base["class1"]
    candidate_c1 = candidate["class1"]
    checks = {
        "class1_f1_gain_at_least_0.005": bool(
            float(candidate_c1["f1"]) >= float(base_c1["f1"]) + 0.005
        ),
        "macro_f1_noninferiority": bool(
            float(candidate["macro_f1"]) >= float(base["macro_f1"]) - 0.002
        ),
        "accuracy_noninferiority": bool(
            float(candidate["accuracy"]) >= float(base["accuracy"]) - 0.005
        ),
        "class1_tp_retention": bool(
            int(candidate_c1["tp"]) >= math.ceil(0.98 * int(base_c1["tp"]))
        ),
        "two_to_one_nonincrease": bool(
            int(candidate["transitions_into_class1"]["2->1"])
            <= int(base["transitions_into_class1"]["2->1"])
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "deltas": {
            "accuracy": float(candidate["accuracy"] - base["accuracy"]),
            "macro_f1": float(candidate["macro_f1"] - base["macro_f1"]),
            "class1_f1": float(candidate_c1["f1"] - base_c1["f1"]),
            "class1_tp": int(candidate_c1["tp"] - base_c1["tp"]),
            "restricted_fp": int(
                candidate["restricted_fp_into_class1"]
                - base["restricted_fp_into_class1"]
            ),
            "two_to_one": int(
                candidate["transitions_into_class1"]["2->1"]
                - base["transitions_into_class1"]["2->1"]
            ),
        },
        "meaning": (
            "pass_authorizes_equal_capacity_dephased_control"
            if all(checks.values())
            else "fail_closes_exact_b23_spatial_recipe"
        ),
    }


def _preflight(args: argparse.Namespace, repo: Path, git: Mapping[str, Any]) -> Dict[str, Any]:
    _checkpoint, teacher, checkpoint_contract = _load_b9(args.b9_checkpoint)
    train, val = _datasets(args.data, include_validation=False)
    if val is not None:
        raise B23ContractError("B23 preflight unexpectedly constructed validation.")
    _seed_all(SEED)
    model = build_b23_model(teacher)
    eval_transform = build_transforms()[1]
    probe_dataset = ClassificationFolderDataset(
        Path(load_data_spec(args.data).split_images_dir("train")),
        EXPECTED_CLASS_NAMES,
        transform=eval_transform,
        split="train_preflight",
    )
    images = torch.stack([probe_dataset[index][0] for index in range(2)], dim=0)
    device = torch.device("cuda")
    model.to(device).eval()
    with torch.inference_mode():
        model.set_mode("direct")
        direct = model(images.to(device)).float()
        model.set_mode(SPATIAL_MODE)
        spatial = model(images.to(device)).float()
    configure_adapter_only(model)
    parity = float((direct - spatial).abs().max().cpu())
    if parity != 0.0:
        raise B23ContractError(f"B23 step-zero branch parity failed: {parity}.")
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    logits = model(images.to(device))
    loss, parts = guarded_adapter_loss(logits, direct.detach(), labels, criterion)
    loss.backward()
    gradient_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    if not gradient_names or any(not is_adapter_parameter(name) for name in gradient_names):
        raise B23ContractError("B23 preflight gradients escaped the adapters.")
    if not bool(torch.isfinite(loss).item()):
        raise B23ContractError("B23 preflight loss is non-finite.")
    ema = AdapterEMA(model, EMA_DECAY)
    ema.update(model)
    ema.copy_to(model)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "git": dict(git),
        "source_hashes": _source_hashes(repo),
        "checkpoint": checkpoint_contract,
        "data": {
            "yaml": str(Path(args.data).resolve()),
            "yaml_sha256": sha256_file(Path(args.data)),
            "train_rows": len(train),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
        },
        "model": {
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "trainable_parameters": ADDED_PARAMETER_COUNT,
            "frozen_parameters": EXPECTED_TOTAL_PARAMETERS - ADDED_PARAMETER_COUNT,
            "primary_state_sha256": _state_dict_sha256(_primary_state(model)),
            "step_zero_max_abs": parity,
            "gradient_parameter_count": len(gradient_names),
            "gradient_adapter_only": True,
            "ema_float_tensors": len(ema.shadow),
            "ema_static_integer_buffers": len(ema.static),
        },
        "loss_probe": {name: float(value.float().cpu()) for name, value in parts.items()},
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "amp": "bf16_train_fp32_eval",
        },
        "train": False,
        "validation": False,
        "test": False,
    }
    return payload


def _read_preflight(
    path: Path, *, repo: Path, git: Mapping[str, Any]
) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("passed") is not True
        or payload.get("git") != dict(git)
        or payload.get("source_hashes") != _source_hashes(repo)
        or payload.get("train") is not False
        or payload.get("validation") is not False
        or payload.get("test") is not False
    ):
        raise B23ContractError("B23 accepted preflight is not same-commit and sealed.")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _run(args: argparse.Namespace, repo: Path, git: Mapping[str, Any]) -> Dict[str, Any]:
    preflight = _read_preflight(args.accepted_preflight, repo=repo, git=git)
    checkpoint, teacher, checkpoint_contract = _load_b9(args.b9_checkpoint)
    train_dataset, val_dataset = _datasets(args.data, include_validation=True)
    if val_dataset is None:
        raise B23ContractError("B23 formal run did not construct validation.")
    train_loader, sampler = _train_loader(train_dataset)
    val_loader = _eval_loader(val_dataset)
    device = torch.device("cuda")
    teacher.to(device).eval().requires_grad_(False)
    base_eval = _evaluate(teacher, val_loader, device=device)
    if tuple(tuple(int(value) for value in row) for row in base_eval["metrics"]["confusion_matrix"]) != EXPECTED_B9_CONFUSION:
        raise B23ContractError(
            "B9 FP32 validation replay changed; refuse candidate training. "
            f"observed={base_eval['metrics']['confusion_matrix']}."
        )

    _seed_all(SEED)
    candidate_backbone = build_model_from_checkpoint(checkpoint)
    model = build_b23_model(candidate_backbone).to(device).eval()
    initial_primary_hash = _state_dict_sha256(_primary_state(model))
    criterion = LDAMFocalLoss(
        class_counts=list(EXPECTED_CLASS_COUNTS),
        gamma=1.0,
        focal_mix=0.1,
        label_smoothing=0.02,
        max_margin=0.3,
        scale=18.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=ADAPTER_LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_warmup_decay_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_start_factor=0.1,
        total_epochs=EPOCHS,
        min_learning_rate=MIN_LR,
        decay_style="cosine",
    )
    ema = AdapterEMA(model, EMA_DECAY)
    initial_adapter_norm = _adapter_norm(model)
    history: list[Dict[str, Any]] = []
    total_updates = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.eval()  # frozen B9 stays in its exact inference regime
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in ("task", "retention", "distillation", "total")}
        sample_count = 0
        final_gradient_norm = 0.0
        for batch_index, batch in enumerate(train_loader):
            images, labels = _split_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = teacher(images)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss, parts = guarded_adapter_loss(
                    logits,
                    teacher_logits,
                    labels,
                    criterion,
                    retention_weight=RETENTION_WEIGHT,
                    distillation_weight=DISTILLATION_WEIGHT,
                    retention_slack=RETENTION_SLACK,
                    temperature=DISTILLATION_TEMPERATURE,
                )
            if not bool(torch.isfinite(loss).item()):
                raise B23ContractError(f"Non-finite B23 loss at epoch {epoch + 1}.")
            (loss / ACCUMULATION_STEPS).backward()
            should_step = (
                (batch_index + 1) % ACCUMULATION_STEPS == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                final_gradient_norm = _adapter_gradient_norm(model)
                clipped = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    GRAD_CLIP,
                )
                if not bool(torch.isfinite(clipped).item()):
                    raise B23ContractError("B23 adapter gradient became non-finite.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                total_updates += 1
                progress = epoch + float(batch_index + 1) / max(len(train_loader), 1)
                scheduler.step(progress)
            count = int(labels.numel())
            sample_count += count
            for name in sums:
                sums[name] += float(parts[name].float().cpu()) * count
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(train_loader):
                _atomic_json(
                    Path(args.output) / "progress.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "epoch": epoch + 1,
                        "epochs": EPOCHS,
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "loss": float(loss.detach().float().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "wall_seconds": time.monotonic() - started,
                    },
                )
                print(
                    f"B23 epoch={epoch + 1}/{EPOCHS} "
                    f"batch={batch_index + 1}/{len(train_loader)} "
                    f"loss={float(loss.detach().float()):.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_updates_total": total_updates,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "mean_losses": {name: value / max(sample_count, 1) for name, value in sums.items()},
                "adapter_gradient_norm": final_gradient_norm,
                "adapter_norm": _adapter_norm(model),
            }
        )

    ema.copy_to(model)
    candidate_eval = _evaluate(model, val_loader, device=device)
    final_primary_hash = _state_dict_sha256(_primary_state(model))
    if final_primary_hash != initial_primary_hash or final_primary_hash != EXPECTED_SELECTED_STATE_SHA256:
        raise B23ContractError("Frozen B9 primary path changed during B23.")

    first_images = candidate_eval["first_images"]
    residual = {"p50": 0.0, "p95": 0.0, "max": 0.0}
    if first_images is not None:
        with torch.inference_mode():
            _, trace = model.forward_with_adapter_trace(first_images.to(device))
        ratios = trace["all_residual_ratio"].float().cpu()
        residual = {
            "p50": float(torch.quantile(ratios, 0.50)),
            "p95": float(torch.quantile(ratios, 0.95)),
            "max": float(ratios.max()),
        }

    adapter_path = Path(args.output) / "b23_spatial_adapter_ema.safetensors"
    state = adapter_state_dict(model)
    save_file(
        state,
        str(adapter_path),
        metadata={
            "protocol_id": PROTOCOL_ID,
            "research_track": "pretrained",
            "base_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "base_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        },
    )
    reloaded_backbone = build_model_from_checkpoint(checkpoint)
    reloaded = build_b23_model(reloaded_backbone)
    load_adapter_state_dict(reloaded, load_file(str(adapter_path), device="cpu"))
    reloaded.to(device).eval()
    reload_max_abs = 0.0
    if first_images is not None:
        with torch.inference_mode():
            observed = reloaded(first_images.to(device)).float().cpu().numpy()
        expected = candidate_eval["logits"][: int(first_images.size(0))]
        reload_max_abs = float(np.max(np.abs(observed - expected)))
        if reload_max_abs != 0.0:
            raise B23ContractError(f"B23 FP32 strict reload parity failed: {reload_max_abs}.")

    prediction_path = Path(args.output) / "validation_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "label", "b9_prediction", "b23_prediction"])
        for path, label, base_pred, candidate_pred in zip(
            val_dataset.sample_paths(),
            candidate_eval["labels"],
            base_eval["predictions"],
            candidate_eval["predictions"],
        ):
            writer.writerow(
                [
                    path.resolve().relative_to(Path(args.data).resolve().parent).as_posix(),
                    label,
                    base_pred,
                    candidate_pred,
                ]
            )

    gate = _promotion_gate(base_eval["metrics"], candidate_eval["metrics"])
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "preflight": preflight,
        "checkpoint": checkpoint_contract,
        "recipe": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "workers": WORKERS,
            "adapter_lr": ADAPTER_LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
            "ema_decay": EMA_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "scheduler_horizon": EPOCHS,
            "min_lr": MIN_LR,
            "retention_weight": RETENTION_WEIGHT,
            "distillation_weight": DISTILLATION_WEIGHT,
            "retention_slack": RETENTION_SLACK,
            "distillation_temperature": DISTILLATION_TEMPERATURE,
            "frozen_primary_eval_mode": True,
            "amp": "bf16_train_fp32_eval",
        },
        "data": {
            "train_rows": len(train_dataset),
            "validation_rows": len(val_dataset),
            "test_rows": 0,
        },
        "parameters": {
            "total": EXPECTED_TOTAL_PARAMETERS,
            "trainable": ADDED_PARAMETER_COUNT,
            "frozen": EXPECTED_TOTAL_PARAMETERS - ADDED_PARAMETER_COUNT,
        },
        "training": {
            "history": history,
            "optimizer_updates": total_updates,
            "initial_adapter_norm": initial_adapter_norm,
            "final_ema_adapter_norm": _adapter_norm(model),
            "primary_state_sha256": final_primary_hash,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.monotonic() - started,
        },
        "b9": base_eval["metrics"],
        "b23": candidate_eval["metrics"],
        "residual_ratio": residual,
        "promotion_gate": gate,
        "artifacts": {
            "adapter": str(adapter_path.resolve()),
            "adapter_sha256": sha256_file(adapter_path),
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256_file(prediction_path),
            "reload_max_abs": reload_max_abs,
        },
        "validation_opened": True,
        "validation_status": "exploratory_design_exposed",
        "test_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B23 supervised-B9 guarded multi-depth ConvPass experiment."
    )
    parser.add_argument("--data", type=Path, default=EXPECTED_DATA_YAML)
    parser.add_argument("--b9-checkpoint", type=Path, default=DEFAULT_B9_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--accepted-preflight", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise B23ContractError("B23 requires CUDA.")
    _configure_determinism()
    repo = Path(__file__).resolve().parents[2]
    git = _git_snapshot(repo)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output = output
    if args.preflight_only:
        if args.accepted_preflight is not None:
            raise B23ContractError("Preflight does not accept --accepted-preflight.")
        payload = _preflight(args, repo, git)
        _atomic_json(output / "preflight.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if args.accepted_preflight is None:
        raise B23ContractError("Formal B23 requires --accepted-preflight.")
    payload = _run(args, repo, git)
    _atomic_json(output / "summary.json", payload)
    print(json.dumps(payload["promotion_gate"], ensure_ascii=False, indent=2), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
