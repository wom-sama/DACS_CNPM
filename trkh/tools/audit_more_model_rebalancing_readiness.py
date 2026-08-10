from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
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
from trkh.models.model import build_model_from_checkpoint
from trkh.models.model_rebalancing import (
    freeze_model_rebalancing_base,
    inject_model_rebalancing_convs,
    merge_model_rebalancing_convs_,
    model_rebalancing_general_only,
    model_rebalancing_modules,
    model_rebalancing_tail_sha256,
    more_discrepancy_loss,
    more_sinusoidal_weight,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
    _classification_metrics,
    directional_event_masks,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_CONVOLUTIONS = (
    "stem.blocks.0.block.conv",
    "stem.blocks.1.block.conv",
    "stem.blocks.2.block.conv",
    "patch_embed.proj",
    "detail_enhancer.proj",
)
EXPECTED_TAIL_PARAMETERS = 57464

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"


@dataclass(frozen=True)
class CleanTrainRow:
    sample_index: int
    source_stem: str
    image_path: Path
    fold: int
    target: int
    keeper_prediction: int
    keeper_probabilities: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only fold-0 functional readiness audit for NeurIPS 2025 MORE. "
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--rank-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--peak-amplitude", type=float, default=10.0)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    return parser.parse_args()


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


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_clean_train_rows(
    path: Path,
    *,
    expected_rows: int = EXPECTED_TRAIN_ROWS,
    expected_class_counts: Sequence[int] = EXPECTED_CLASS_COUNTS,
    expected_fold_counts: Sequence[int] = EXPECTED_FOLD_COUNTS,
) -> list[CleanTrainRow]:
    rows: list[CleanTrainRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target_index",
            "keeper_prediction",
            *{f"keeper_prob_{index}" for index in range(len(expected_class_counts))},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CIDT prediction CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            if str(raw["condition"]).strip().lower() != "clean":
                continue
            probabilities = tuple(
                float(raw[f"keeper_prob_{index}"])
                for index in range(len(expected_class_counts))
            )
            if not all(math.isfinite(value) and value >= 0.0 for value in probabilities):
                raise ValueError("CIDT keeper probabilities must be finite and nonnegative.")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError("CIDT keeper probabilities are not normalized.")
            image_path = Path(str(raw["image_path"])).resolve()
            split_parts = {part.casefold() for part in image_path.parts}
            if "val" in split_parts or "test" in split_parts or "train" not in split_parts:
                raise ValueError(f"Non-train image leaked into MORE readiness: {image_path}")
            row = CleanTrainRow(
                sample_index=int(raw["sample_index"]),
                source_stem=str(raw["source_stem"]).strip().casefold(),
                image_path=image_path,
                fold=int(raw["fold"]),
                target=int(raw["target_index"]),
                keeper_prediction=int(raw["keeper_prediction"]),
                keeper_probabilities=probabilities,
            )
            if row.source_stem != image_path.stem.casefold():
                raise ValueError(f"CIDT source stem does not match image path: {row}")
            rows.append(row)

    if len(rows) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} clean rows, found {len(rows)}.")
    indices = [row.sample_index for row in rows]
    if indices != list(range(int(expected_rows))):
        raise ValueError("CIDT clean rows must be ordered unique sample indices 0..N-1.")
    targets = np.asarray([row.target for row in rows], dtype=np.int64)
    observed_classes = np.bincount(
        targets,
        minlength=len(expected_class_counts),
    ).tolist()
    if observed_classes != [int(value) for value in expected_class_counts]:
        raise ValueError(
            f"CIDT class counts differ: {observed_classes} != {list(expected_class_counts)}"
        )
    folds = np.asarray([row.fold for row in rows], dtype=np.int64)
    observed_folds = np.bincount(folds, minlength=len(expected_fold_counts)).tolist()
    if observed_folds != [int(value) for value in expected_fold_counts]:
        raise ValueError(
            f"CIDT fold counts differ: {observed_folds} != {list(expected_fold_counts)}"
        )
    return rows


def _ordered_index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _base_parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if ".tail_a." in name or ".tail_b." in name:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}\n".encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
        raise ValueError("MORE readiness requires classification logits [B,C].")
    return logits


def _make_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    context: str,
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
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            **kwargs,
        ),
        summary,
    )


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _factor_gradient_status(model: nn.Module) -> Dict[str, bool]:
    status = {
        "finite": True,
        "tail_a_nonzero": False,
        "tail_b_nonzero": False,
    }
    for module in model_rebalancing_modules(model).values():
        for key, parameter in (
            ("tail_a_nonzero", module.tail_a.weight),
            ("tail_b_nonzero", module.tail_b.weight),
        ):
            gradient = parameter.grad
            if gradient is None:
                continue
            status["finite"] = bool(status["finite"] and torch.isfinite(gradient).all())
            status[key] = bool(
                status[key] or int(torch.count_nonzero(gradient).item()) > 0
            )
    return status


def _train_variant(
    *,
    name: str,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    class_counts: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    peak_amplitude: float,
    amp: bool,
) -> Dict[str, object]:
    started = time.perf_counter()
    model = copy.deepcopy(prototype).to(device).eval()
    modules = model_rebalancing_modules(model)
    if tuple(modules) != EXPECTED_CONVOLUTIONS:
        raise ValueError(f"Unexpected MORE module order: {tuple(modules)}")
    parameters = freeze_model_rebalancing_base(model)
    trainable_parameters = sum(int(parameter.numel()) for parameter in parameters)
    if trainable_parameters != EXPECTED_TAIL_PARAMETERS:
        raise ValueError(
            f"Unexpected trainable tail parameter count: {trainable_parameters}"
        )
    base_sha_before = _base_parameter_sha256(model)
    initial_tail_sha = model_rebalancing_tail_sha256(model)
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader, train_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        context=f"more_{name}_train",
    )
    total_steps = len(train_loader)
    if total_steps < 2:
        raise ValueError("MORE preflight requires at least two train batches.")

    history: List[Dict[str, object]] = []
    factor_status = {
        "finite": True,
        "tail_a_nonzero": False,
        "tail_b_nonzero": False,
    }
    first_batch_payload = None
    general_reference = None
    initial_zero_max_abs = None
    for batch_index, (images, targets, metadata) in enumerate(train_loader):
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        if first_batch_payload is None:
            first_batch_payload = (images.detach().clone(), targets.detach().clone(), metadata)
            with torch.inference_mode():
                full_initial = _forward_logits(model, images, metadata, device=device)
                with model_rebalancing_general_only(model):
                    general_reference = _forward_logits(
                        model, images, metadata, device=device
                    ).detach()
            initial_zero_max_abs = float(
                (full_initial.float() - general_reference.float()).abs().max().item()
            )

        optimizer.zero_grad(set_to_none=True)
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            full_logits = _forward_logits(model, images, metadata, device=device)
            if float(peak_amplitude) > 0.0:
                with torch.no_grad(), model_rebalancing_general_only(model):
                    general_logits = _forward_logits(
                        model, images, metadata, device=device
                    )
                more_loss, more_stats = more_discrepancy_loss(
                    full_logits.float(),
                    general_logits.float(),
                    targets,
                    class_counts=class_counts,
                )
            else:
                more_loss = full_logits.float().sum() * 0.0
                more_stats = {
                    "per_sample_discrepancy": torch.zeros_like(
                        targets, dtype=torch.float32
                    )
                }
            ce_loss = F.cross_entropy(full_logits.float(), targets)
            alpha = more_sinusoidal_weight(
                batch_index + 1,
                total_steps,
                peak_amplitude=float(peak_amplitude),
            )
            total_loss = ce_loss + float(alpha) * more_loss
        if not torch.isfinite(total_loss):
            raise ValueError(f"Non-finite MORE loss at batch {batch_index}.")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        current_status = _factor_gradient_status(model)
        for key in factor_status:
            factor_status[key] = bool(
                factor_status[key] and current_status[key]
                if key == "finite"
                else factor_status[key] or current_status[key]
            )
        grad_norm = _gradient_norm(parameters)
        if not math.isfinite(grad_norm):
            raise ValueError("MORE tail gradient norm is non-finite.")
        scaler.step(optimizer)
        scaler.update()
        history.append(
            {
                "batch": int(batch_index),
                "rows": int(targets.numel()),
                "ce_loss": float(ce_loss.detach().cpu().item()),
                "more_loss": float(more_loss.detach().cpu().item()),
                "alpha": float(alpha),
                "weighted_more_loss": float(alpha * more_loss.detach().cpu().item()),
                "mean_discrepancy": float(
                    more_stats["per_sample_discrepancy"].float().mean().detach().cpu().item()
                ),
                "gradient_norm": float(grad_norm),
            }
        )

    if first_batch_payload is None or general_reference is None:
        raise RuntimeError("MORE train loader produced no batches.")
    first_images, _, first_metadata = first_batch_payload
    with torch.inference_mode(), model_rebalancing_general_only(model):
        general_after = _forward_logits(
            model,
            first_images,
            first_metadata,
            device=device,
        )
    general_drift_max_abs = float(
        (general_after.float() - general_reference.float()).abs().max().item()
    )
    base_sha_after = _base_parameter_sha256(model)
    final_tail_sha = model_rebalancing_tail_sha256(model)

    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        context=f"more_{name}_holdout",
    )
    prediction_rows: List[Dict[str, object]] = []
    merge_payload = None
    with torch.inference_mode():
        for images, targets, metadata in holdout_loader:
            images = images.to(device=device, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            full_logits = _forward_logits(model, images, metadata, device=device).float()
            with model_rebalancing_general_only(model):
                general_logits = _forward_logits(
                    model, images, metadata, device=device
                ).float()
            probabilities = F.softmax(full_logits, dim=1)
            discrepancies = torch.linalg.vector_norm(
                full_logits - general_logits,
                ord=2,
                dim=1,
            )
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("MORE holdout metadata is missing sample_index.")
            if merge_payload is None:
                merge_payload = (
                    images.detach().clone(),
                    metadata,
                    full_logits.detach().clone(),
                )
            for row_index in range(int(targets.numel())):
                row = {
                    "sample_index": int(sample_indices[row_index].item()),
                    "target": int(targets[row_index].item()),
                    "prediction": int(probabilities[row_index].argmax().item()),
                    "tail_logit_l2": float(discrepancies[row_index].item()),
                }
                for class_index in range(int(probabilities.shape[1])):
                    row[f"prob_{class_index}"] = float(
                        probabilities[row_index, class_index].item()
                    )
                prediction_rows.append(row)

    if len(prediction_rows) != len(holdout_indices):
        raise ValueError("MORE holdout prediction count is incomplete.")
    if [int(row["sample_index"]) for row in prediction_rows] != [
        int(index) for index in holdout_indices
    ]:
        raise ValueError("MORE holdout sample order changed.")
    if merge_payload is None:
        raise RuntimeError("MORE holdout loader produced no batches.")
    merge_images, merge_metadata, full_before_merge = merge_payload
    merged_names = merge_model_rebalancing_convs_(model)
    with torch.inference_mode():
        full_after_merge = _forward_logits(
            model,
            merge_images,
            merge_metadata,
            device=device,
        ).float()
    merge_max_abs = float(
        (full_before_merge.float() - full_after_merge).abs().max().item()
    )
    return {
        "name": name,
        "peak_amplitude": float(peak_amplitude),
        "train_rows": int(len(train_indices)),
        "train_batches": int(len(history)),
        "holdout_rows": int(len(prediction_rows)),
        "train_index_sha256": _ordered_index_sha256(train_indices),
        "initial_tail_sha256": initial_tail_sha,
        "final_tail_sha256": final_tail_sha,
        "base_parameter_sha256_before": base_sha_before,
        "base_parameter_sha256_after": base_sha_after,
        "base_parameters_unchanged": base_sha_before == base_sha_after,
        "trainable_tail_parameters": int(trainable_parameters),
        "initial_zero_max_abs": float(initial_zero_max_abs or 0.0),
        "general_drift_max_abs": general_drift_max_abs,
        "merge_max_abs": merge_max_abs,
        "merged_names": merged_names,
        "factor_gradient_status": factor_status,
        "history": history,
        "predictions": prediction_rows,
        "loader": {
            "train": train_loader_summary,
            "holdout": holdout_loader_summary,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _write_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    control: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> None:
    row_by_index = {row.sample_index: row for row in rows}
    candidate_by_index = {int(row["sample_index"]): row for row in candidate}
    fields = [
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        "control_prediction",
        "candidate_prediction",
        "control_tail_logit_l2",
        "candidate_tail_logit_l2",
    ]
    for prefix in ("keeper", "control", "candidate"):
        fields.extend(f"{prefix}_prob_{index}" for index in range(len(EXPECTED_CLASS_COUNTS)))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for control_row in control:
            sample_index = int(control_row["sample_index"])
            source = row_by_index[sample_index]
            candidate_row = candidate_by_index[sample_index]
            output = {
                "sample_index": sample_index,
                "source_stem": source.source_stem,
                "image_path": str(source.image_path),
                "fold": source.fold,
                "target": source.target,
                "keeper_prediction": source.keeper_prediction,
                "control_prediction": int(control_row["prediction"]),
                "candidate_prediction": int(candidate_row["prediction"]),
                "control_tail_logit_l2": float(control_row["tail_logit_l2"]),
                "candidate_tail_logit_l2": float(candidate_row["tail_logit_l2"]),
            }
            for class_index, probability in enumerate(source.keeper_probabilities):
                output[f"keeper_prob_{class_index}"] = probability
            for class_index in range(len(EXPECTED_CLASS_COUNTS)):
                output[f"control_prob_{class_index}"] = control_row[f"prob_{class_index}"]
                output[f"candidate_prob_{class_index}"] = candidate_row[f"prob_{class_index}"]
            writer.writerow(output)


def _compact_variant_summary(payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"predictions"}
    }


def main() -> None:
    args = parse_args()
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = args.checkpoint.resolve()
    data_path = args.data.resolve()
    cidt_summary_path = args.cidt_summary.resolve()
    cidt_predictions_path = args.cidt_predictions.resolve()
    keeper_sha = _verify_sha256(
        checkpoint_path,
        LOCKED_KEEPER_SHA256,
        "keeper checkpoint",
    )
    data_sha = _verify_sha256(data_path, LOCKED_DATA_SHA256, "data YAML")
    cidt_summary_sha = _verify_sha256(
        cidt_summary_path,
        LOCKED_CIDT_SUMMARY_SHA256,
        "CIDT summary",
    )
    cidt_predictions_sha = _verify_sha256(
        cidt_predictions_path,
        LOCKED_CIDT_PREDICTIONS_SHA256,
        "CIDT predictions",
    )
    cidt_summary = json.loads(cidt_summary_path.read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use.")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use.")
    rows = _read_clean_train_rows(cidt_predictions_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Keeper checkpoint class order is invalid.")
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("MORE readiness supports temporal_frames=1 only.")
    data_spec = load_data_spec(
        data_path,
        class_name_mode="raw",
        expected_num_classes=len(class_names),
    )
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from the keeper checkpoint.")
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
        raise ValueError("Dataset and CIDT row counts differ.")
    for row, sample_path in zip(rows, sample_paths):
        if row.image_path != sample_path:
            raise ValueError(
                f"Dataset/CIDT path mismatch at sample {row.sample_index}: "
                f"{sample_path} != {row.image_path}"
            )
    transform = _build_eval_transform(semantics)

    fold = int(args.fold)
    if fold != 0:
        raise ValueError("The locked functional preflight uses CIDT fold 0 only.")
    fit_indices = [row.sample_index for row in rows if row.fold != fold]
    holdout_indices = [row.sample_index for row in rows if row.fold == fold]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    source_overlap = fit_sources.intersection(holdout_sources)
    if source_overlap:
        raise ValueError("CIDT fit/holdout source groups overlap.")
    rng = np.random.default_rng(int(args.seed) + fold)
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    rng.shuffle(shuffled_fit)
    requested_train_rows = int(args.max_train_batches) * int(args.batch_size)
    if requested_train_rows > int(shuffled_fit.size):
        raise ValueError("Requested MORE train budget exceeds the fit fold.")
    train_indices = shuffled_fit[:requested_train_rows].tolist()

    locked_config_exact = bool(
        int(args.batch_size) == 64
        and int(args.seed) == 42
        and int(args.max_train_batches) == 20
        and math.isclose(float(args.rank_ratio), 0.1, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.learning_rate), 3e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.momentum), 0.9, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 2e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.peak_amplitude), 10.0, rel_tol=0.0, abs_tol=1e-12)
        and int(args.focus_class) == 1
        and len(holdout_indices) == EXPECTED_FOLD_COUNTS[0]
    )

    prototype = build_model_from_checkpoint(checkpoint).cpu().eval()
    records = inject_model_rebalancing_convs(
        prototype,
        rank_ratio=float(args.rank_ratio),
        seed=int(args.seed),
    )
    freeze_model_rebalancing_base(prototype)
    inventory = [record.__dict__ for record in records]
    inventory_names = tuple(record.name for record in records)
    initial_tail_sha = model_rebalancing_tail_sha256(prototype)
    tail_parameters = sum(int(record.tail_parameters) for record in records)

    algebra_loader, algebra_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=0,
        context="more_algebra",
    )
    algebra_images, _, algebra_metadata = next(iter(algebra_loader))
    algebra_images = algebra_images.to(device)
    unwrapped = build_model_from_checkpoint(checkpoint).to(device).eval()
    wrapped = copy.deepcopy(prototype).to(device).eval()
    with torch.inference_mode():
        unwrapped_logits = _forward_logits(
            unwrapped, algebra_images, algebra_metadata, device=device
        ).float()
        wrapped_logits = _forward_logits(
            wrapped, algebra_images, algebra_metadata, device=device
        ).float()
        with model_rebalancing_general_only(wrapped):
            general_logits = _forward_logits(
                wrapped, algebra_images, algebra_metadata, device=device
            ).float()
    zero_tail_max_abs = float((wrapped_logits - unwrapped_logits).abs().max().item())
    general_keeper_max_abs = float((general_logits - unwrapped_logits).abs().max().item())
    del unwrapped, wrapped, algebra_images, algebra_metadata
    if device.type == "cuda":
        torch.cuda.empty_cache()

    control = _train_variant(
        name="control",
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        class_counts=EXPECTED_CLASS_COUNTS,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        learning_rate=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        peak_amplitude=0.0,
        amp=not bool(args.disable_amp),
    )
    candidate = _train_variant(
        name="more",
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        class_counts=EXPECTED_CLASS_COUNTS,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        learning_rate=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        peak_amplitude=float(args.peak_amplitude),
        amp=not bool(args.disable_amp),
    )
    control_predictions = control["predictions"]
    candidate_predictions = candidate["predictions"]
    targets = np.asarray([rows[index].target for index in holdout_indices], dtype=np.int64)
    keeper_predictions = np.asarray(
        [rows[index].keeper_prediction for index in holdout_indices], dtype=np.int64
    )
    control_classes = np.asarray(
        [int(row["prediction"]) for row in control_predictions], dtype=np.int64
    )
    candidate_classes = np.asarray(
        [int(row["prediction"]) for row in candidate_predictions], dtype=np.int64
    )
    keeper_metrics = _classification_metrics(
        targets, keeper_predictions, num_classes=len(class_names)
    )
    control_metrics = _classification_metrics(
        targets, control_classes, num_classes=len(class_names)
    )
    candidate_metrics = _classification_metrics(
        targets, candidate_classes, num_classes=len(class_names)
    )
    transitions = directional_event_masks(
        targets,
        control_classes,
        candidate_classes,
        focus_class=int(args.focus_class),
    )
    transition_counts = {
        key: int(value.sum()) for key, value in transitions.items()
    }
    changed = int((control_classes != candidate_classes).sum())
    new_nonfocus_3_to_2_harms = int(
        ((targets == 3) & (control_classes == 3) & (candidate_classes == 2)).sum()
    )
    control_discrepancy = np.asarray(
        [float(row["tail_logit_l2"]) for row in control_predictions], dtype=np.float64
    )
    candidate_discrepancy = np.asarray(
        [float(row["tail_logit_l2"]) for row in candidate_predictions], dtype=np.float64
    )
    focus_mask = targets == int(args.focus_class)
    discrepancy_summary = {
        "control_focus_mean": float(control_discrepancy[focus_mask].mean()),
        "control_nonfocus_mean": float(control_discrepancy[~focus_mask].mean()),
        "candidate_focus_mean": float(candidate_discrepancy[focus_mask].mean()),
        "candidate_nonfocus_mean": float(candidate_discrepancy[~focus_mask].mean()),
        "candidate_focus_to_nonfocus_ratio": float(
            candidate_discrepancy[focus_mask].mean()
            / max(1e-12, candidate_discrepancy[~focus_mask].mean())
        ),
    }

    control_history = control["history"]
    candidate_history = candidate["history"]
    structural_gates = {
        "train_only_provenance": True,
        "locked_config_exact": locked_config_exact,
        "all_rows_and_folds_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "fit_holdout_source_overlap_zero": len(source_overlap) == 0,
        "expected_convolution_inventory": inventory_names == EXPECTED_CONVOLUTIONS,
        "expected_tail_parameter_count": tail_parameters == EXPECTED_TAIL_PARAMETERS,
        "zero_tail_keeper_error_1e6": zero_tail_max_abs <= 1e-6,
        "general_keeper_error_1e6": general_keeper_max_abs <= 1e-6,
        "matched_train_order": (
            control["train_index_sha256"] == candidate["train_index_sha256"]
        ),
        "matched_tail_initialization": (
            control["initial_tail_sha256"]
            == candidate["initial_tail_sha256"]
            == initial_tail_sha
        ),
    }
    functional_gates = {
        "control_base_frozen": bool(control["base_parameters_unchanged"]),
        "candidate_base_frozen": bool(candidate["base_parameters_unchanged"]),
        "control_general_drift_1e6": float(control["general_drift_max_abs"]) <= 1e-6,
        "candidate_general_drift_1e6": float(candidate["general_drift_max_abs"]) <= 1e-6,
        "control_merge_error_1e5": float(control["merge_max_abs"]) <= 1e-5,
        "candidate_merge_error_1e5": float(candidate["merge_max_abs"]) <= 1e-5,
        "control_factor_gradients": all(
            bool(value) for value in control["factor_gradient_status"].values()
        ),
        "candidate_factor_gradients": all(
            bool(value) for value in candidate["factor_gradient_status"].values()
        ),
        "candidate_more_loss_nonzero": any(
            float(row["more_loss"]) > 0.0 for row in candidate_history[1:]
        ),
        "candidate_tail_changed": (
            candidate["final_tail_sha256"] != candidate["initial_tail_sha256"]
        ),
        "matched_budget_20_batches": (
            len(control_history) == len(candidate_history) == 20
        ),
    }
    effect_gates = {
        "hard_decision_changes_2": changed >= 2,
        "excess_harms_at_most_5": (
            transition_counts["candidate_harm"]
            - transition_counts["candidate_correction"]
            <= 5
        ),
    }
    all_gates = {
        **structural_gates,
        **functional_gates,
        **effect_gates,
    }
    all_gates_passed = all(bool(value) for value in all_gates.values())

    predictions_path = output_dir / "holdout_predictions.csv"
    _write_predictions(
        predictions_path,
        rows=rows,
        control=control_predictions,
        candidate=candidate_predictions,
    )
    summary = {
        "mode": "more_model_rebalancing_functional_preflight",
        "test_data_used": False,
        "validation_data_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": keeper_sha,
        "data_yaml": str(data_path),
        "data_yaml_sha256": data_sha,
        "cidt_summary": str(cidt_summary_path),
        "cidt_summary_sha256": cidt_summary_sha,
        "cidt_predictions": str(cidt_predictions_path),
        "cidt_predictions_sha256": cidt_predictions_sha,
        "device": str(device),
        "amp": bool(not args.disable_amp and device.type == "cuda"),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_matmul_allow_tf32": (
                torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
            ),
            "cudnn_allow_tf32": (
                torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "config": {
            "fold": fold,
            "batch_size": int(args.batch_size),
            "max_train_batches": int(args.max_train_batches),
            "rank_ratio": float(args.rank_ratio),
            "learning_rate": float(args.learning_rate),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "peak_amplitude": float(args.peak_amplitude),
            "focus_class": int(args.focus_class),
            "seed": int(args.seed),
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
            "ordered_train_index_sha256": _ordered_index_sha256(train_indices),
        },
        "model_rebalancing": {
            "inventory": inventory,
            "tail_parameters": tail_parameters,
            "initial_tail_sha256": initial_tail_sha,
            "zero_tail_max_abs": zero_tail_max_abs,
            "general_keeper_max_abs": general_keeper_max_abs,
            "algebra_loader": algebra_loader_summary,
        },
        "control": _compact_variant_summary(control),
        "candidate": _compact_variant_summary(candidate),
        "metrics": {
            "keeper": keeper_metrics,
            "control": control_metrics,
            "candidate": candidate_metrics,
        },
        "transitions_candidate_vs_control": {
            "changed": changed,
            **transition_counts,
            "new_nonfocus_3_to_2_harms": new_nonfocus_3_to_2_harms,
        },
        "tail_discrepancy": discrepancy_summary,
        "structural_gates": structural_gates,
        "functional_gates": functional_gates,
        "effect_gates": effect_gates,
        "all_gates_passed": all_gates_passed,
        "full_oof_authorized": all_gates_passed,
        "artifacts": {
            "holdout_predictions": predictions_path.name,
        },
        "guardrail": (
            "A pass authorizes the locked five-fold train-only OOF audit only. "
            "It does not authorize validation, test, trainer integration, or a parameter sweep."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# MORE Functional Preflight",
        "",
        f"- All gates passed: `{all_gates_passed}`",
        f"- Full train-only OOF authorized: `{all_gates_passed}`",
        f"- Zero/general keeper max error: `{zero_tail_max_abs:.9g}/{general_keeper_max_abs:.9g}`",
        f"- Control/candidate merge max error: `{float(control['merge_max_abs']):.9g}/{float(candidate['merge_max_abs']):.9g}`",
        f"- Holdout changed/corrections/harms: `{changed}/{transition_counts['candidate_correction']}/{transition_counts['candidate_harm']}`",
        f"- Class1 FP remove/create: `{transition_counts['focus_fp_remove_correct']}/{transition_counts['focus_fp_create']}`",
        f"- Class1 FN rescue/TP break: `{transition_counts['focus_fn_rescue']}/{transition_counts['focus_tp_break']}`",
        "",
        "## Gates",
        "",
    ]
    report.extend(f"- `{key}`: `{value}`" for key, value in all_gates.items())
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_serializable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
