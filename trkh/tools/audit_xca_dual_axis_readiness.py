from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    _forward_classification_with_metadata,
)
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    MultiHeadSelfAttention,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
    _classification_metrics,
    directional_event_masks,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_XCA_LAYERS = (2, 5)
EXPECTED_ADDED_PARAMETERS = 1040
EXPECTED_BLOCKS = 8
EXPECTED_HEADS = 8
EXPECTED_HEAD_DIM = 32
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "5a7a46e28772002089b1daadbfa8e2e468ca16c53559fa669382b401c6c5590f"
LOCKED_XCIT_COMMIT = "82f5291f412604970c39a912586e008ec009cdca"
LOCKED_XCIT_SOURCE_SHA256 = "3e2d4be847dc8c88d98fe0f965bca3119aeeb588975eff0bc62fdfb20e8b3e9e"
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.30
MAX_PEAK_VRAM_GIB = 3.25
STATIC_EXPORT_BATCH_SIZE = 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only readiness audit for shared-projection XCA. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/TRKH_5CLASS_XCA_DUAL_AXIS_READINESS_PROTOCOL_20260715.md"),
    )
    parser.add_argument(
        "--official-xcit-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_next_arch_sources_20260715\xcit"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--xca-layers", type=str, default="2,5")
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().lower():
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _rng_snapshot() -> Dict[str, object]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _rng_summary(snapshot: Mapping[str, object]) -> Dict[str, object]:
    cpu = snapshot.get("cpu")
    cuda = snapshot.get("cuda")
    if not torch.is_tensor(cpu) or not isinstance(cuda, list):
        raise TypeError("Invalid RNG snapshot.")
    return {
        "cpu_sha256": _tensor_sha256(cpu),
        "cuda_sha256": [_tensor_sha256(value) for value in cuda],
    }


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_cpu = left.get("cpu")
    right_cpu = right.get("cpu")
    left_cuda = left.get("cuda")
    right_cuda = right.get("cuda")
    return bool(
        torch.is_tensor(left_cpu)
        and torch.is_tensor(right_cpu)
        and torch.equal(left_cpu, right_cpu)
        and isinstance(left_cuda, list)
        and isinstance(right_cuda, list)
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def _candidate_config(source: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "cross_covariance_attention": True,
            "cross_covariance_attention_layers": str(args.xca_layers),
            "cross_covariance_attention_residual_scale": float(args.residual_scale),
            "pretrained": False,
        }
    )
    return config


def _expected_missing_keys(layers: Sequence[int]) -> set[str]:
    return {
        f"blocks.{int(layer) - 1}.cross_covariance_attention.{suffix}"
        for layer in layers
        for suffix in ("temperature", "norm.weight", "norm.bias")
    }


def _construct_models(
    checkpoint: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[nn.Module, nn.Module, Dict[str, object]]:
    source_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state")
    if not isinstance(source_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Keeper checkpoint lacks model_config/class_names.")
    if not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint lacks model_state.")
    num_classes = len(class_names)

    set_seed(int(args.seed), deterministic=True)
    control = create_model(num_classes=num_classes, model_config=source_config)
    control_constructor_rng = _rng_snapshot()
    load_model_state(control, dict(state), strict=True)

    set_seed(int(args.seed), deterministic=True)
    candidate = create_model(
        num_classes=num_classes,
        model_config=_candidate_config(source_config, args),
    )
    candidate_constructor_rng = _rng_snapshot()
    missing, unexpected = load_model_state(candidate, dict(state), strict=False)
    expected_missing = _expected_missing_keys(EXPECTED_XCA_LAYERS)
    existing_bit_exact = all(
        key in candidate.state_dict()
        and torch.equal(candidate.state_dict()[key].detach().cpu(), value.detach().cpu())
        for key, value in state.items()
    )
    control_parameters = sum(int(value.numel()) for value in control.parameters())
    candidate_parameters = sum(int(value.numel()) for value in candidate.parameters())
    summary = {
        "control_constructor_rng": _rng_summary(control_constructor_rng),
        "candidate_constructor_rng": _rng_summary(candidate_constructor_rng),
        "constructor_rng_equal": _rng_equal(
            control_constructor_rng,
            candidate_constructor_rng,
        ),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "expected_missing_keys": sorted(expected_missing),
        "missing_keys_exact": set(missing) == expected_missing,
        "existing_state_bit_exact": bool(existing_bit_exact),
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "control_parameters": control_parameters,
        "candidate_parameters": candidate_parameters,
        "added_parameters": candidate_parameters - control_parameters,
    }
    return control.eval(), candidate.eval(), summary


def _make_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    context: str,
    seed: int,
) -> tuple[DataLoader, Dict[str, object]]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        indices,
        corruption=IdentityCorruption(),
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            generator=generator,
            **kwargs,
        ),
        summary,
    )


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> Tensor:
    logits, _ = _forward_classification_with_metadata(
        model,
        images,
        metadata,
        device=device,
    )
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("XCA readiness requires classification logits [B,C].")
    return logits


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _predict(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> list[Dict[str, object]]:
    model.eval()
    rows: list[Dict[str, object]] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                logits = _forward_logits(model, images, metadata, device=device)
            probabilities = F.softmax(logits.float(), dim=1)
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Prediction metadata is missing sample_index.")
            for row_index in range(int(targets.numel())):
                row: Dict[str, object] = {
                    "sample_index": int(sample_indices[row_index].item()),
                    "target": int(targets[row_index].item()),
                    "prediction": int(probabilities[row_index].argmax().item()),
                }
                for class_index in range(int(probabilities.size(1))):
                    row[f"prob_{class_index}"] = float(
                        probabilities[row_index, class_index].item()
                    )
                rows.append(row)
    return rows


def _train_variant(
    *,
    name: str,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    started = time.perf_counter()
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device)
    initial_state_sha = _state_sha256(model)
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"xca_{name}_holdout",
        seed=int(args.seed) + 100,
    )
    unadapted_predictions = _predict(
        model=model,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
    )

    set_seed(int(args.seed), deterministic=True)
    train_loader, train_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"xca_{name}_fit",
        seed=int(args.seed) + 200,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    model.train()
    history: list[Dict[str, object]] = []
    ordered_indices: list[int] = []
    post_forward_rng: list[Dict[str, object]] = []
    all_gradients_finite = True
    for batch_index, (images, targets, metadata) in enumerate(train_loader):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        ordered_indices.extend(int(value) for value in sample_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            logits = _forward_logits(model, images, metadata, device=device)
            loss = F.cross_entropy(logits.float(), targets)
        post_forward_rng.append(_rng_summary(_rng_snapshot()))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {name} loss at batch {batch_index}.")
        loss.backward()
        gradient_norm_squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            all_gradients_finite = bool(
                all_gradients_finite and torch.isfinite(parameter.grad).all()
            )
            gradient_norm_squared += float(
                parameter.grad.detach().float().square().sum().item()
            )
        optimizer.step()
        history.append(
            {
                "batch": int(batch_index),
                "rows": int(targets.numel()),
                "loss": float(loss.detach().item()),
                "gradient_norm": math.sqrt(gradient_norm_squared),
            }
        )

    if len(history) != int(args.max_train_batches):
        raise ValueError(
            f"Expected {args.max_train_batches} train batches, observed {len(history)}."
        )
    adapted_predictions = _predict(
        model=model,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    final_state_sha = _state_sha256(model)
    xca_parameter_summary = {
        name: {
            "initial_sha256": _tensor_sha256(prototype.state_dict()[name]),
            "final_sha256": _tensor_sha256(model.state_dict()[name]),
            "changed": not torch.equal(
                prototype.state_dict()[name].detach().cpu(),
                model.state_dict()[name].detach().cpu(),
            ),
        }
        for name in model.state_dict()
        if ".cross_covariance_attention." in name
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(ordered_indices),
        "train_order_sha256": _ordered_index_sha256(ordered_indices),
        "post_forward_rng": post_forward_rng,
        "all_gradients_finite": all_gradients_finite,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "state_changed": initial_state_sha != final_state_sha,
        "xca_parameters": xca_parameter_summary,
        "history": history,
        "unadapted_predictions": unadapted_predictions,
        "adapted_predictions": adapted_predictions,
        "loader": {
            "train": train_loader_summary,
            "holdout": holdout_loader_summary,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def summarize_xca_gradients(model: nn.Module) -> Dict[str, Dict[str, object]]:
    named = dict(model.named_parameters())
    groups: Dict[str, list[str]] = {}
    for layer in EXPECTED_XCA_LAYERS:
        prefix = f"blocks.{int(layer) - 1}"
        groups[f"layer{layer}_temperature"] = [
            f"{prefix}.cross_covariance_attention.temperature"
        ]
        groups[f"layer{layer}_norm"] = [
            f"{prefix}.cross_covariance_attention.norm.weight",
            f"{prefix}.cross_covariance_attention.norm.bias",
        ]
        groups[f"layer{layer}_shared_qkv"] = [
            f"{prefix}.attn.qkv.weight",
            f"{prefix}.attn.qkv.bias",
        ]
        groups[f"layer{layer}_shared_projection"] = [
            f"{prefix}.attn.proj.weight",
            f"{prefix}.attn.proj.bias",
        ]
    output: Dict[str, Dict[str, object]] = {}
    for group, names in groups.items():
        gradients = [named[name].grad for name in names if name in named]
        present = [value for value in gradients if value is not None]
        finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
        nonzero = sum(int(torch.count_nonzero(value).item()) for value in present)
        output[group] = {
            "parameters": names,
            "parameter_tensors": sum(name in named for name in names),
            "gradient_tensors": len(present),
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": bool(
                len(present) == len(names)
                and finite
                and all(int(torch.count_nonzero(value).item()) > 0 for value in present)
            ),
        }
    return output


def _gradient_diagnostics(
    *,
    prototype: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).train()
    count = int(images.size(0))
    input_value = images.detach().clone().to(device).requires_grad_(True)
    target_value = targets[:count].to(device=device, dtype=torch.long)
    model.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        logits = _forward_logits(model, input_value, metadata, device=device)
        loss = F.cross_entropy(logits.float(), target_value)
    loss.backward()
    gradients = summarize_xca_gradients(model)
    input_gradient = input_value.grad
    summary = {
        "batch_size": count,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "float32",
        "loss": float(loss.detach().item()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients": gradients,
        "all_xca_gradient_families": all(
            bool(value["passed"]) for value in gradients.values()
        ),
        "input_gradient_present": input_gradient is not None,
        "input_gradient_finite": bool(
            input_gradient is not None and torch.isfinite(input_gradient).all()
        ),
        "input_gradient_nonzero": bool(
            input_gradient is not None
            and int(torch.count_nonzero(input_gradient).item()) > 0
        ),
    }
    del model, input_value
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _benchmark(
    *,
    prototype: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
    seed: int,
) -> Dict[str, object]:
    set_seed(int(seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device).train()

    def iteration() -> tuple[Tensor, Tensor]:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = _forward_logits(model, images, metadata, device=device)
            loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        return logits, loss

    iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, loss = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "logits_loss_finite": bool(
            torch.isfinite(logits).all() and torch.isfinite(loss)
        ),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _metadata_to_device(
    metadata: Mapping[str, object],
    *,
    device: torch.device,
    count: int,
) -> Dict[str, object]:
    return {
        key: value[:count].to(device=device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in metadata.items()
    }


def _structural_diagnostics(
    *,
    control: nn.Module,
    candidate: nn.Module,
    images_cpu: Tensor,
    targets_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    images = images_cpu.to(device=device, non_blocking=True)
    targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=int(images.size(0)))
    metadata2 = _metadata_to_device(metadata, device=device, count=2)

    control_forward = copy.deepcopy(control).to(device).train()
    candidate_forward = copy.deepcopy(candidate).to(device).train()
    set_seed(int(args.seed), deterministic=True)
    with torch.no_grad():
        _forward_logits(control_forward, images[:2], metadata2, device=device)
    control_forward_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    with torch.no_grad():
        _forward_logits(candidate_forward, images[:2], metadata2, device=device)
    candidate_forward_rng = _rng_snapshot()
    forward_rng_equal = _rng_equal(control_forward_rng, candidate_forward_rng)
    del control_forward, candidate_forward
    gc.collect()
    torch.cuda.empty_cache()

    fp32_count = int(args.fp32_batch_size)
    fp32 = _gradient_diagnostics(
        prototype=candidate,
        images=images[:fp32_count],
        targets=targets[:fp32_count],
        metadata=_metadata_to_device(metadata, device=device, count=fp32_count),
        device=device,
        amp_dtype=None,
    )
    amp = _gradient_diagnostics(
        prototype=candidate,
        images=images,
        targets=targets,
        metadata=metadata,
        device=device,
        amp_dtype=amp_dtype,
    )

    model = copy.deepcopy(candidate).to(device).eval()
    channel_maps: Dict[int, Tensor] = {}
    channel_inputs: Dict[int, Tensor] = {}
    handles = []
    for layer in EXPECTED_XCA_LAYERS:
        module = model.blocks[layer - 1].cross_covariance_attention

        def hook(_, inputs, output, *, layer_number=layer):
            channel_inputs[layer_number] = inputs[0].detach()
            channel_maps[layer_number] = output[1].detach()

        handles.append(module.register_forward_hook(hook))
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    with torch.inference_mode():
        features = model.forward_features(
            images[:1],
            bbox_token_prior=bbox[:1] if torch.is_tensor(bbox) else None,
            image_valid_mask=image_mask[:1] if torch.is_tensor(image_mask) else None,
            return_attention=True,
            return_trace=True,
        )
        pruned_features = model.forward_features(
            images[:1],
            bbox_token_prior=bbox[:1] if torch.is_tensor(bbox) else None,
            image_valid_mask=image_mask[:1] if torch.is_tensor(image_mask) else None,
            return_trace=True,
        )
    for handle in handles:
        handle.remove()
    spatial_attentions = features.get("attentions", {})
    trace = features.get("trace", {})
    channel_rows = []
    residual_squared = 0.0
    patch_squared = 0.0
    for layer in EXPECTED_XCA_LAYERS:
        attention = channel_maps[layer].float()
        patch_value = channel_inputs[layer].float()
        module = model.blocks[layer - 1].cross_covariance_attention
        module_trace = module.trace()
        ratio = float(module_trace["residual_norm_ratio"].item())
        residual_squared += float((patch_value.norm() * ratio).square().item())
        patch_squared += float(patch_value.norm().square().item())
        channel_rows.append(
            {
                "layer": layer,
                "shape": [int(value) for value in attention.shape],
                "finite": bool(torch.isfinite(attention).all()),
                "nonnegative": bool((attention >= 0).all()),
                "max_row_sum_error": float(
                    (attention.sum(dim=-1) - 1.0).abs().amax().item()
                ),
                "normalized_entropy": float(
                    module_trace["normalized_entropy"].item()
                ),
                "diagonal_mass": float(module_trace["diagonal_mass"].item()),
                "query_norm_max_error": float(
                    module_trace["query_norm_max_error"].item()
                ),
                "key_norm_max_error": float(
                    module_trace["key_norm_max_error"].item()
                ),
                "residual_norm_ratio": ratio,
                "temperature": [
                    float(module_trace["temperature_min"].item()),
                    float(module_trace["temperature_mean"].item()),
                    float(module_trace["temperature_max"].item()),
                ],
            }
        )
    combined_residual_ratio = math.sqrt(residual_squared / max(patch_squared, 1e-12))

    prefix_probe = torch.randn(
        2,
        int(model.num_prefix_tokens) + 17,
        int(model.blocks[0].attn.qkv.in_features),
        device=device,
    )
    prefix_reference = prefix_probe[:, : int(model.num_prefix_tokens)].clone()
    with torch.inference_mode():
        prefix_updated = model.blocks[EXPECTED_XCA_LAYERS[0] - 1].apply_cross_covariance_attention(
            prefix_probe,
            prefix_count=int(model.num_prefix_tokens),
        )
    prefix_max_abs_delta = float(
        (prefix_updated[:, : int(model.num_prefix_tokens)] - prefix_reference)
        .abs()
        .amax()
        .item()
    )

    with torch.inference_mode():
        baseline_logits = _forward_logits(
            model,
            images[:2],
            metadata2,
            device=device,
        ).float()
        ablation_rows = []
        for layer in EXPECTED_XCA_LAYERS:
            block = model.blocks[layer - 1]
            scale = block.cross_covariance_attention_residual_scale
            block.cross_covariance_attention_residual_scale = 0.0
            ablated = _forward_logits(
                model,
                images[:2],
                metadata2,
                device=device,
            ).float()
            block.cross_covariance_attention_residual_scale = scale
            ablation_rows.append(
                {
                    "layer": layer,
                    "logit_max_abs_delta": float(
                        (baseline_logits - ablated).abs().amax().item()
                    ),
                }
            )

    control_benchmark = _benchmark(
        prototype=control,
        images=images,
        targets=targets,
        metadata=metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
        seed=int(args.seed),
    )
    candidate_benchmark = _benchmark(
        prototype=candidate,
        images=images,
        targets=targets,
        metadata=metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
        seed=int(args.seed),
    )
    runtime_ratio = float(
        candidate_benchmark["median_seconds"]
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )
    peak_vram_gib = float(candidate_benchmark["peak_vram_gib"])

    export = _export_diagnostics(
        candidate=candidate,
        images=images_cpu[:2],
        metadata=metadata_cpu,
        output_dir=output_dir,
    )
    pruning_layers = [
        int(value["layer"].item())
        for value in pruned_features.get("trace", {}).get("pruning", [])
    ]
    xca_layers = [
        index + 1
        for index, block in enumerate(model.blocks)
        if block.cross_covariance_attention is not None
    ]
    spatial_blocks = [
        index + 1
        for index, block in enumerate(model.blocks)
        if isinstance(block.attn, MultiHeadSelfAttention)
    ]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "control_forward_rng": _rng_summary(control_forward_rng),
        "candidate_forward_rng": _rng_summary(candidate_forward_rng),
        "forward_rng_equal": forward_rng_equal,
        "fp32": fp32,
        "amp": amp,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "xca_layers": xca_layers,
        "spatial_mhsa_layers": spatial_blocks,
        "pruning_layers": pruning_layers,
        "spatial_attention_layers": [int(value) + 1 for value in spatial_attentions],
        "spatial_attention_shapes": {
            str(int(layer) + 1): [int(value) for value in attention.shape]
            for layer, attention in spatial_attentions.items()
        },
        "channel_attention": channel_rows,
        "combined_residual_norm_ratio": combined_residual_ratio,
        "prefix_direct_max_abs_delta": prefix_max_abs_delta,
        "layer_ablation": ablation_rows,
        "control_benchmark": control_benchmark,
        "candidate_benchmark": candidate_benchmark,
        "runtime_ratio": runtime_ratio,
        "peak_vram_gib": peak_vram_gib,
        "export": export,
        "trace_xca_layers": (
            [int(value) for value in trace["cross_covariance_attention_layers"].tolist()]
            if isinstance(trace, Mapping)
            and torch.is_tensor(trace.get("cross_covariance_attention_layers"))
            else []
        ),
    }


class _XCAExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _export_diagnostics(
    *,
    candidate: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    wrapper = _XCAExportWrapper(candidate)
    cpu_images = images[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    if int(cpu_images.size(0)) != STATIC_EXPORT_BATCH_SIZE:
        raise ValueError("XCA ONNX audit requires a static batch of two rows.")
    bbox_value = metadata.get("bbox")
    if not torch.is_tensor(bbox_value):
        bbox_value = torch.zeros(int(cpu_images.size(0)), 8)
    cpu_bbox = bbox_value[: int(cpu_images.size(0))].detach().float().cpu()
    mask_value = metadata.get("image_mask")
    if not torch.is_tensor(mask_value):
        mask_value = torch.ones(
            int(cpu_images.size(0)),
            int(cpu_images.size(2)),
            int(cpu_images.size(3)),
            dtype=torch.bool,
        )
    cpu_mask = mask_value[: int(cpu_images.size(0))].detach().bool().cpu()
    path = output_dir / "xca_candidate.onnx"
    torch.onnx.export(
        wrapper,
        (cpu_images, cpu_bbox, cpu_mask),
        str(path),
        input_names=["images", "bbox", "image_mask"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rows = []
    maximum_error = 0.0
    for batch_size in (STATIC_EXPORT_BATCH_SIZE,):
        batch_images = cpu_images[:batch_size]
        batch_bbox = cpu_bbox[:batch_size]
        batch_mask = cpu_mask[:batch_size]
        with torch.inference_mode():
            expected = wrapper(batch_images, batch_bbox, batch_mask).detach().cpu()
        observed = torch.from_numpy(
            session.run(
                ["logits"],
                {
                    "images": batch_images.numpy(),
                    "bbox": batch_bbox.numpy(),
                    "image_mask": batch_mask.numpy(),
                },
            )[0]
        )
        error = float((expected - observed).abs().amax().item())
        maximum_error = max(maximum_error, error)
        rows.append(
            {
                "batch_size": batch_size,
                "shape": [int(value) for value in observed.shape],
                "finite": bool(torch.isfinite(observed).all()),
                "maximum_absolute_error": error,
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "opset": 17,
        "batch_contract": f"static_batch_{STATIC_EXPORT_BATCH_SIZE}",
        "providers": session.get_providers(),
        "batch_results": rows,
        "maximum_absolute_error": maximum_error,
    }


def _prediction_arrays(
    rows: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    predictions = np.asarray(
        [int(row["prediction"]) for row in rows],
        dtype=np.int64,
    )
    return targets, predictions


def _comparison(
    *,
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    num_classes: int,
    focus_class: int,
) -> Dict[str, object]:
    targets, control = _prediction_arrays(control_rows)
    candidate_targets, candidate = _prediction_arrays(candidate_rows)
    if not np.array_equal(targets, candidate_targets):
        raise ValueError("Control/candidate targets do not match.")
    control_metrics = _classification_metrics(targets, control, num_classes=num_classes)
    candidate_metrics = _classification_metrics(targets, candidate, num_classes=num_classes)
    masks = directional_event_masks(
        targets,
        control,
        candidate,
        focus_class=int(focus_class),
    )
    transitions = {key: int(value.sum()) for key, value in masks.items()}
    restricted_targets = np.isin(targets, np.asarray([0, 2, 4]))
    restricted_control_fp = int((restricted_targets & (control == focus_class)).sum())
    restricted_candidate_fp = int((restricted_targets & (candidate == focus_class)).sum())
    nonfocus = [index for index in range(num_classes) if index != int(focus_class)]
    nonfocus_drops = [
        float(control_metrics["per_class_f1"][index])
        - float(candidate_metrics["per_class_f1"][index])
        for index in nonfocus
    ]
    return {
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "macro_f1": float(candidate_metrics["macro_f1"])
            - float(control_metrics["macro_f1"]),
            "class1_f1": float(candidate_metrics["per_class_f1"][focus_class])
            - float(control_metrics["per_class_f1"][focus_class]),
            "class1_precision": float(
                candidate_metrics["per_class_precision"][focus_class]
            )
            - float(control_metrics["per_class_precision"][focus_class]),
            "class1_recall": float(candidate_metrics["per_class_recall"][focus_class])
            - float(control_metrics["per_class_recall"][focus_class]),
        },
        "transitions": {
            "changed": int((control != candidate).sum()),
            **transitions,
            "restricted_focus_fp_control": restricted_control_fp,
            "restricted_focus_fp_candidate": restricted_candidate_fp,
            "restricted_focus_fp_reduction": (
                restricted_control_fp - restricted_candidate_fp
            ),
            "new_nonfocus_3_to_2_harms": int(
                ((targets == 3) & (control == 3) & (candidate == 2)).sum()
            ),
        },
        "maximum_nonfocus_f1_drop": max(nonfocus_drops),
    }


def assess_xca_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    unadapted: Mapping[str, object],
    adapted: Mapping[str, object],
) -> Dict[str, object]:
    unadapted_delta = unadapted["delta"]
    unadapted_transitions = unadapted["transitions"]
    adapted_delta = adapted["delta"]
    adapted_transitions = adapted["transitions"]
    unadapted_checks = {
        "macro_f1_delta_gte_minus_0p010": float(unadapted_delta["macro_f1"]) >= -0.010,
        "class1_recall_delta_gte_minus_0p020": float(
            unadapted_delta["class1_recall"]
        )
        >= -0.020,
        "net_tp_breaks_at_most_3": (
            int(unadapted_transitions["focus_tp_break"])
            - int(unadapted_transitions["focus_fn_rescue"])
            <= 3
        ),
    }
    adapted_checks = {
        "changed_decisions_gte_2": int(adapted_transitions["changed"]) >= 2,
        "macro_f1_delta_gte_minus_0p002": float(adapted_delta["macro_f1"]) >= -0.002,
        "class1_f1_delta_gte_0p002": float(adapted_delta["class1_f1"]) >= 0.002,
        "class1_precision_delta_gte_0p005": float(
            adapted_delta["class1_precision"]
        )
        >= 0.005,
        "class1_recall_delta_gte_minus_0p010": float(
            adapted_delta["class1_recall"]
        )
        >= -0.010,
        "restricted_focus_fp_reduction_gte_2": int(
            adapted_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "tp_breaks_lte_fn_rescues": int(adapted_transitions["focus_tp_break"])
        <= int(adapted_transitions["focus_fn_rescue"]),
        "corrections_gt_harms": int(adapted_transitions["candidate_correction"])
        > int(adapted_transitions["candidate_harm"]),
        "new_3_to_2_harms_lte_2": int(
            adapted_transitions["new_nonfocus_3_to_2_harms"]
        )
        <= 2,
        "maximum_nonfocus_f1_drop_lte_0p015": float(
            adapted["maximum_nonfocus_f1_drop"]
        )
        <= 0.015,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **unadapted_checks,
        **adapted_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "unadapted_checks": unadapted_checks,
        "adapted_checks": adapted_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_smoke_authorized": not failed,
        "full_train_authorized": False,
    }


def _write_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    source_by_index = {row.sample_index: row for row in rows}
    variants = {
        "control_unadapted": control["unadapted_predictions"],
        "candidate_unadapted": candidate["unadapted_predictions"],
        "control_adapted": control["adapted_predictions"],
        "candidate_adapted": candidate["adapted_predictions"],
    }
    indexed = {
        name: {int(row["sample_index"]): row for row in values}
        for name, values in variants.items()
    }
    ordered_indices = [
        int(row["sample_index"]) for row in variants["control_unadapted"]
    ]
    fields = ["sample_index", "source_stem", "image_path", "fold", "target"]
    for name in variants:
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(len(EXPECTED_CLASS_COUNTS)))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_index in ordered_indices:
            source = source_by_index[sample_index]
            output: Dict[str, object] = {
                "sample_index": sample_index,
                "source_stem": source.source_stem,
                "image_path": str(source.image_path),
                "fold": source.fold,
                "target": source.target,
            }
            for name in variants:
                row = indexed[name][sample_index]
                output[f"{name}_prediction"] = int(row["prediction"])
                for class_index in range(len(EXPECTED_CLASS_COUNTS)):
                    output[f"{name}_prob_{class_index}"] = row[f"prob_{class_index}"]
            writer.writerow(output)


def _compact_variant(payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"unadapted_predictions", "adapted_predictions", "post_forward_rng"}
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked XCA Stage A requires CUDA.")
    locked_config_exact = bool(
        int(args.batch_size) == 32
        and int(args.fp32_batch_size) == 2
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == 20
        and math.isclose(float(args.learning_rate), 8e-5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 0.05, rel_tol=0.0, abs_tol=1e-12)
        and int(args.focus_class) == 1
        and str(args.xca_layers).replace(" ", "") == "2,5"
        and math.isclose(float(args.residual_scale), 0.10, rel_tol=0.0, abs_tol=1e-12)
        and int(args.benchmark_repeats) == 3
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )
    if not locked_config_exact:
        raise ValueError("Stage A arguments differ from the precommitted XCA protocol.")

    output_dir = _prepare_output_dir(args.output_dir)
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
    }
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "protocol": _verify_sha256(paths["protocol"], LOCKED_PROTOCOL_SHA256, "protocol"),
    }
    official_root = Path(args.official_xcit_root).resolve()
    official_source = official_root / "xcit.py"
    official_commit = _git_commit(official_root)
    official_sha = _verify_sha256(
        official_source,
        LOCKED_XCIT_SOURCE_SHA256,
        "official XCiT source",
    )
    if official_commit != LOCKED_XCIT_COMMIT:
        raise ValueError(
            f"Official XCiT commit mismatch: {official_commit} != {LOCKED_XCIT_COMMIT}"
        )

    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use.")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])

    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    amp_dtype = _amp_dtype(device)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Keeper class order is invalid.")
    control, candidate, construction = _construct_models(checkpoint, args)
    source_config = checkpoint["model_config"]
    if bool(source_config.get("visual_contrast_attention", False)):
        raise ValueError("Keeper unexpectedly uses Visual-Contrast Attention.")
    if bool(source_config.get("gated_relative_position_attention", False)):
        raise ValueError("Keeper unexpectedly uses gated relative position attention.")

    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        paths["data"],
        class_name_mode="raw",
        expected_num_classes=len(class_names),
    )
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    sample_paths = [Path(value).resolve() for value in base_dataset.sample_paths()]
    if len(sample_paths) != len(rows):
        raise ValueError("Dataset/CIDT row count mismatch.")
    for row, path in zip(rows, sample_paths):
        if row.image_path != path:
            raise ValueError(f"Dataset/CIDT path mismatch at {row.sample_index}.")
    transform = _build_eval_transform(semantics)

    fold = int(args.fold)
    fit_indices = [row.sample_index for row in rows if row.fold != fold]
    holdout_indices = [row.sample_index for row in rows if row.fold == fold]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    source_overlap = fit_sources.intersection(holdout_sources)
    if source_overlap:
        raise ValueError("Fit/holdout source groups overlap.")
    rng = np.random.default_rng(int(args.seed) + fold)
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    rng.shuffle(shuffled_fit)
    train_rows = int(args.max_train_batches) * int(args.batch_size)
    train_indices = shuffled_fit[:train_rows].tolist()
    if len(train_indices) != 640 or len(holdout_indices) != EXPECTED_FOLD_COUNTS[0]:
        raise ValueError("Locked train/holdout row counts differ.")

    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="xca_stage_a_resource",
        seed=int(args.seed) + 300,
    )
    images_cpu, targets_cpu, metadata_cpu = next(iter(resource_loader))
    structural = _structural_diagnostics(
        control=control,
        candidate=candidate,
        images_cpu=images_cpu,
        targets_cpu=targets_cpu,
        metadata_cpu=metadata_cpu,
        args=args,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )

    control_result = _train_variant(
        name="control",
        prototype=control,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    candidate_result = _train_variant(
        name="candidate",
        prototype=candidate,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )

    expected_holdout_order = holdout_indices
    for result in (control_result, candidate_result):
        for key in ("unadapted_predictions", "adapted_predictions"):
            observed = [int(row["sample_index"]) for row in result[key]]
            if observed != expected_holdout_order:
                raise ValueError(f"{result['name']} {key} holdout order changed.")
    unadapted = _comparison(
        control_rows=control_result["unadapted_predictions"],
        candidate_rows=candidate_result["unadapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    adapted = _comparison(
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )

    channel_rows = structural["channel_attention"]
    structural_checks = {
        "locked_config_exact": locked_config_exact,
        "locked_source_and_inputs": True,
        "train_only_provenance": True,
        "all_rows_and_fold_counts_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "fit_holdout_source_overlap_zero": len(source_overlap) == 0,
        "control_strict_load_bit_exact": bool(construction["existing_state_bit_exact"]),
        "candidate_missing_keys_exact": bool(construction["missing_keys_exact"]),
        "candidate_unexpected_keys_zero": not construction["unexpected_keys"],
        "candidate_existing_state_bit_exact": bool(construction["existing_state_bit_exact"]),
        "added_parameters_exact_1040": int(construction["added_parameters"])
        == EXPECTED_ADDED_PARAMETERS,
        "constructor_rng_equal": bool(construction["constructor_rng_equal"]),
        "forward_rng_equal": bool(structural["forward_rng_equal"]),
        "xca_layers_exact_2_5": structural["xca_layers"] == list(EXPECTED_XCA_LAYERS),
        "trace_xca_layers_exact_2_5": structural["trace_xca_layers"]
        == list(EXPECTED_XCA_LAYERS),
        "all_eight_spatial_mhsa_blocks": structural["spatial_mhsa_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "pruning_after_layers_2_5": structural["pruning_layers"]
        == list(EXPECTED_XCA_LAYERS),
        "spatial_attention_all_eight_layers": structural["spatial_attention_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "channel_map_shapes_exact": all(
            row["shape"][1:] == [EXPECTED_HEADS, EXPECTED_HEAD_DIM, EXPECTED_HEAD_DIM]
            for row in channel_rows
        ),
        "channel_maps_finite_normalized": all(
            bool(row["finite"])
            and bool(row["nonnegative"])
            and float(row["max_row_sum_error"]) <= 1e-5
            for row in channel_rows
        ),
        "qk_token_norms_unit": all(
            float(row["query_norm_max_error"]) <= 1e-5
            and float(row["key_norm_max_error"]) <= 1e-5
            for row in channel_rows
        ),
        "prefix_direct_delta_zero": float(structural["prefix_direct_max_abs_delta"])
        == 0.0,
        "fp32_all_gradients_and_input_finite": bool(
            structural["fp32"]["all_xca_gradient_families"]
            and structural["fp32"]["input_gradient_finite"]
            and structural["fp32"]["input_gradient_nonzero"]
        ),
        "amp_all_gradients_and_input_finite": bool(
            structural["amp"]["all_xca_gradient_families"]
            and structural["amp"]["input_gradient_finite"]
            and structural["amp"]["input_gradient_nonzero"]
        ),
        "both_layer_ablations_material": all(
            float(row["logit_max_abs_delta"]) >= 1e-6
            for row in structural["layer_ablation"]
        ),
        "combined_residual_ratio_in_range": 0.005
        <= float(structural["combined_residual_norm_ratio"])
        <= 0.20,
        "channel_entropy_in_range": all(
            0.20 <= float(row["normalized_entropy"]) <= 0.9995
            for row in channel_rows
        ),
        "channel_diagonal_mass_below_0p95": all(
            float(row["diagonal_mass"]) < 0.95 for row in channel_rows
        ),
        "onnx_error_lte_1e5": float(structural["export"]["maximum_absolute_error"])
        <= MAX_ONNX_ERROR,
        "runtime_ratio_lte_1p30": float(structural["runtime_ratio"])
        <= float(args.max_runtime_ratio),
        "peak_vram_lte_3p25_gib": float(structural["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
        "matched_train_order": control_result["train_order_sha256"]
        == candidate_result["train_order_sha256"]
        == _ordered_index_sha256(train_indices),
        "matched_train_budget": control_result["train_batches"]
        == candidate_result["train_batches"]
        == int(args.max_train_batches),
        "matched_forward_rng_checkpoints": control_result["post_forward_rng"]
        == candidate_result["post_forward_rng"],
        "both_training_gradients_finite": bool(
            control_result["all_gradients_finite"]
            and candidate_result["all_gradients_finite"]
        ),
        "candidate_xca_parameters_changed": bool(candidate_result["xca_parameters"])
        and all(
            bool(value["changed"])
            for value in candidate_result["xca_parameters"].values()
        ),
    }
    gate = assess_xca_stage_a(
        structural_checks=structural_checks,
        unadapted=unadapted,
        adapted=adapted,
    )
    predictions_path = output_dir / "holdout_predictions.csv"
    _write_predictions(
        predictions_path,
        rows=rows,
        control=control_result,
        candidate=candidate_result,
    )
    summary: Dict[str, object] = {
        "method": "shared_projection_xca_dual_axis",
        "protocol_stage": "A_train_only_functional_resource_decision_readiness",
        "test_data_used": False,
        "validation_data_used": False,
        "sources": {
            **{key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_xcit_root": str(official_root),
            "official_xcit_commit": official_commit,
            "official_xcit_source": str(official_source),
            "official_xcit_source_sha256": official_sha,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "config": {
            "batch_size": int(args.batch_size),
            "fp32_batch_size": int(args.fp32_batch_size),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "max_train_batches": int(args.max_train_batches),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "focus_class": int(args.focus_class),
            "xca_layers": str(args.xca_layers),
            "residual_scale": float(args.residual_scale),
        },
        "dataset": {
            "rows": len(rows),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "fit_rows": len(fit_indices),
            "train_budget_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "fit_sources": len(fit_sources),
            "holdout_sources": len(holdout_sources),
            "source_overlap": len(source_overlap),
            "train_order_sha256": _ordered_index_sha256(train_indices),
            "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
            "resource_loader": resource_loader_summary,
        },
        "construction": construction,
        "structural": structural,
        "control_training": _compact_variant(control_result),
        "candidate_training": _compact_variant(candidate_result),
        "unadapted_candidate_vs_control": unadapted,
        "adapted_candidate_vs_control": adapted,
        "gate": gate,
        "artifacts": {
            "holdout_predictions": predictions_path.name,
            "holdout_predictions_sha256": _sha256(predictions_path),
            "onnx": structural["export"]["path"],
        },
        "guardrail": (
            "A pass authorizes one deterministic keeper-initialized no-test validation "
            "smoke only. It never authorizes test access, a full train, or a sweep."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# XCA Dual-Axis Stage-A Readiness",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage-B smoke authorized: `{gate['stage_b_smoke_authorized']}`",
        f"- Failed checks: `{gate['failed_checks']}`",
        f"- Runtime ratio / peak VRAM GiB: `{structural['runtime_ratio']:.6f}/{structural['peak_vram_gib']:.6f}`",
        f"- ONNX max error: `{structural['export']['maximum_absolute_error']:.9g}`",
        f"- Adapted macro/class1 F1 delta: `{adapted['delta']['macro_f1']:.6f}/{adapted['delta']['class1_f1']:.6f}`",
        f"- Adapted class1 precision/recall delta: `{adapted['delta']['class1_precision']:.6f}/{adapted['delta']['class1_recall']:.6f}`",
        f"- Adapted changed/corrections/harms: `{adapted['transitions']['changed']}/{adapted['transitions']['candidate_correction']}/{adapted['transitions']['candidate_harm']}`",
        f"- Adapted restricted FP reduction, FN rescue/TP break: `{adapted['transitions']['restricted_focus_fp_reduction']}/{adapted['transitions']['focus_fn_rescue']}/{adapted['transitions']['focus_tp_break']}`",
        "",
        "Validation and test were not loaded.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "raw_dataset_modified": False,
        "validation_used": False,
        "test_used": False,
        "artifacts": [
            {
                "path": str(summary_path.resolve()),
                "sha256": _sha256(summary_path),
                "role": "stage_a_summary",
            },
            {
                "path": str(predictions_path.resolve()),
                "sha256": _sha256(predictions_path),
                "role": "ordered_train_holdout_predictions",
            },
            {
                "path": structural["export"]["path"],
                "sha256": structural["export"]["sha256"],
                "role": "candidate_onnx",
            },
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(
        json.dumps(
            {
                "output_dir": str(Path(summary["structural"]["export"]["path"]).parent),
                "all_gates_passed": summary["gate"]["all_gates_passed"],
                "stage_b_smoke_authorized": summary["gate"][
                    "stage_b_smoke_authorized"
                ],
                "failed_checks": summary["gate"]["failed_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
