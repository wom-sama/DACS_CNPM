from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import (
    _failed_export,
    _make_lighting_loader,
    _onnx_compare,
    _write_illumination_predictions,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _git_commit,
    _make_loader,
    _prepare_output_dir,
    _rng_snapshot,
    _rng_summary,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics
from trkh.training.representation_self_challenging import locate_protected_rsc


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_TRAIN_BATCHES = 60
EXPECTED_TRAIN_BUDGET_ROWS = 1920
EXPECTED_REPRESENTATION_DIM = 256
EXPECTED_DROP_COUNT = 86
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "9227f5e5122184b89b6b2ee4239f62fc31671343f346e5a9e0b4efc8b9959366"
LOCKED_RSC_PAPER_SHA256 = "fa76aec1fe8aaf430548d938f008adb3936d7ace53061e63412ceac1da0ea48e"
LOCKED_RSC_COMMIT = "bf6d280c5d74910f009ea8963c59167252659666"
LOCKED_RSC_DG_SOURCE_SHA256 = "c2837b190a61be481a7ff7aae31e9459647ec22edc94db88f9d2a0004abf9bf8"
LOCKED_RSC_IMAGENET_SOURCE_SHA256 = "bc898ca8abe7edb3319f8697fabb6dcc552f1c2f6a1badaee006062db5875bb7"
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.35
MAX_PEAK_VRAM_GIB = 3.25


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Stage-A audit for class-1-protected pooled-channel RSC. "
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
        default=Path(
            "docs/TRKH_5CLASS_CLASS1_PROTECTED_RSC_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--rsc-root",
        type=Path,
        default=Path(r"C:\Users\ADMIN\AppData\Local\Temp\trkh_rsc_official_bf6d280"),
    )
    parser.add_argument(
        "--rsc-paper",
        type=Path,
        default=Path(r"C:\Users\ADMIN\AppData\Local\Temp\trkh_rsc_eccv2020_123470120.pdf"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rsc_stage_a"))
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--eligible-classes", type=str, default="0,2,3,4")
    parser.add_argument("--drop-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--batch-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--minimum-drop", type=float, default=1e-4)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    return parser.parse_args(argv)


def _parse_classes(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if parsed != (0, 2, 3, 4):
        raise ValueError("Locked RSC eligible classes must be exactly 0,2,3,4.")
    return parsed


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 32
        and int(args.num_workers) == 4
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == EXPECTED_TRAIN_BATCHES
        and math.isclose(float(args.learning_rate), 1e-5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 0.05, rel_tol=0.0, abs_tol=1e-12)
        and int(args.focus_class) == 1
        and _parse_classes(args.eligible_classes) == (0, 2, 3, 4)
        and math.isclose(float(args.drop_fraction), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.batch_fraction), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.minimum_drop), 1e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = Path(args.rsc_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "rsc_paper": Path(args.rsc_paper).resolve(),
        "rsc_dg_source": root / "Domain_Generalization" / "models" / "resnet.py",
        "rsc_imagenet_source": root / "ImageNet" / "resnet.py",
    }


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Stage-A arguments differ from the precommitted RSC protocol.")
    paths = _source_paths(args)
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
        "rsc_paper": _verify_sha256(
            paths["rsc_paper"], LOCKED_RSC_PAPER_SHA256, "RSC paper"
        ),
        "rsc_dg_source": _verify_sha256(
            paths["rsc_dg_source"], LOCKED_RSC_DG_SOURCE_SHA256, "RSC DG source"
        ),
        "rsc_imagenet_source": _verify_sha256(
            paths["rsc_imagenet_source"],
            LOCKED_RSC_IMAGENET_SOURCE_SHA256,
            "RSC ImageNet source",
        ),
    }
    commit = _git_commit(Path(args.rsc_root).resolve())
    if commit != LOCKED_RSC_COMMIT:
        raise ValueError(f"Official RSC commit differs: {commit} != {LOCKED_RSC_COMMIT}")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "rsc_commit": commit,
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _parameter_group(name: str) -> str:
    if name.startswith("stem."):
        return "stem"
    if name.startswith(("patch_embed.", "cls_token", "register_tokens", "pos_embed", "branch_")):
        return "input_tokens"
    if name.startswith("blocks."):
        return "transformer"
    if name.startswith("norm."):
        return "normalization"
    if name.startswith("fine_grained_pool."):
        return "fine_grained_pool"
    if name.startswith("head."):
        return "classifier"
    if name.startswith(("cnn_fusion_", "cnn_fusion_head.")):
        return "cnn_fusion"
    return "other"


def _parameter_group_hashes(model: nn.Module) -> Dict[str, str]:
    digests: Dict[str, hashlib._Hash] = {}
    for name, value in sorted(model.named_parameters()):
        group = _parameter_group(name)
        digest = digests.setdefault(group, hashlib.sha256())
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {key: value.hexdigest() for key, value in sorted(digests.items())}


def _metadata_values(
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    bbox_value = (
        bbox.to(device=device, dtype=torch.float32, non_blocking=True)
        if torch.is_tensor(bbox)
        else None
    )
    image_mask_value = (
        image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        if torch.is_tensor(image_mask)
        else None
    )
    return bbox_value, image_mask_value


def _features_from_batch(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> Dict[str, Tensor]:
    bbox, image_mask = _metadata_values(metadata, device=device)
    features = model.forward_features(
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    if bbox is not None:
        features["bbox"] = bbox
    return features


def _predict_fp32(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[Dict[str, object]]:
    model.eval()
    rows: list[Dict[str, object]] = []
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            features = _features_from_batch(model, images, metadata, device=device)
            logits = classification_logits_from_features(model, features)
            probabilities = logits.float().softmax(dim=1)
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


def _equation_diagnostics(args: argparse.Namespace) -> Dict[str, object]:
    representation = torch.linspace(
        -1.0,
        1.0,
        steps=4 * EXPECTED_REPRESENTATION_DIM,
        dtype=torch.float32,
    ).reshape(4, EXPECTED_REPRESENTATION_DIM)
    representation.requires_grad_(True)
    weight = torch.linspace(
        -0.7,
        0.9,
        steps=5 * EXPECTED_REPRESENTATION_DIM,
        dtype=torch.float32,
    ).reshape(5, EXPECTED_REPRESENTATION_DIM)
    bias = torch.linspace(-0.2, 0.2, steps=5, dtype=torch.float32)
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)

    def forward(value: Tensor) -> Tensor:
        return value @ weight.T + bias

    clean_logits = forward(representation)
    result = locate_protected_rsc(
        representation=representation,
        clean_logits=clean_logits,
        targets=targets,
        forward_from_representation=forward,
        focus_class=int(args.focus_class),
        eligible_classes=_parse_classes(args.eligible_classes),
        drop_fraction=float(args.drop_fraction),
        batch_fraction=float(args.batch_fraction),
        minimum_drop=float(args.minimum_drop),
    )
    expected_gradient = weight.index_select(0, targets)
    expected_indices = expected_gradient.topk(
        k=EXPECTED_DROP_COUNT,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    expected_mask = torch.ones_like(expected_gradient)
    expected_mask.scatter_(1, expected_indices, 0.0)
    expected_mask[targets.eq(int(args.focus_class))] = 1.0
    selected_mask_counts = (result.final_mask == 0).sum(dim=1)
    return {
        "gradient_max_abs_error": float(
            (result.gradient - expected_gradient).abs().amax().item()
        ),
        "preliminary_mask_bit_exact": bool(
            torch.equal(result.preliminary_mask, expected_mask)
        ),
        "drop_count": int(result.drop_count),
        "focus_masked_channels": int(
            (result.preliminary_mask[targets.eq(int(args.focus_class))] == 0)
            .sum()
            .item()
        ),
        "selected_rows": int(result.selected.sum().item()),
        "selected_mask_counts": selected_mask_counts.tolist(),
        "selected_all_positive_drop": bool(
            result.positive_drop[result.selected].all().item()
            if bool(result.selected.any().item())
            else True
        ),
        "finite": bool(
            torch.isfinite(result.gradient).all().item()
            and torch.isfinite(result.preliminary_logits).all().item()
            and torch.isfinite(result.confidence_drop).all().item()
        ),
    }


def _candidate_forward(
    *,
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Tensor, Dict[str, object]]:
    model.train()
    features = _features_from_batch(model, images, metadata, device=device)
    representation = features.get("pooled")
    if not torch.is_tensor(representation) or representation.ndim != 2:
        raise ValueError("RSC requires the pooled [batch,channels] representation.")
    if int(representation.size(1)) != EXPECTED_REPRESENTATION_DIM:
        raise ValueError("RSC representation dimension differs from the locked protocol.")

    model.eval()

    def forward_from_representation(value: Tensor) -> Tensor:
        challenged_features = dict(features)
        challenged_features["pooled"] = value
        return classification_logits_from_features(model, challenged_features)

    clean_eval_logits = forward_from_representation(representation)
    rsc = locate_protected_rsc(
        representation=representation,
        clean_logits=clean_eval_logits,
        targets=targets,
        forward_from_representation=forward_from_representation,
        focus_class=int(args.focus_class),
        eligible_classes=_parse_classes(args.eligible_classes),
        drop_fraction=float(args.drop_fraction),
        batch_fraction=float(args.batch_fraction),
        minimum_drop=float(args.minimum_drop),
    )

    model.train()
    final_features = dict(features)
    final_features["pooled"] = representation * rsc.final_mask.to(
        dtype=representation.dtype
    )
    logits = classification_logits_from_features(model, final_features)
    selected_mask_counts = (rsc.final_mask == 0).sum(dim=1)
    protected = targets.eq(int(args.focus_class))
    selected_targets = targets[rsc.selected]
    telemetry = {
        "rows": int(targets.numel()),
        "eligible_rows": int(rsc.eligible.sum().item()),
        "positive_drop_rows": int(rsc.positive_drop.sum().item()),
        "selected_rows": int(rsc.selected.sum().item()),
        "selection_limit": int(rsc.selection_limit),
        "drop_count": int(rsc.drop_count),
        "class1_rows": int(protected.sum().item()),
        "class1_masked_channels": int((rsc.final_mask[protected] == 0).sum().item()),
        "selected_mask_count_min": int(
            selected_mask_counts[rsc.selected].min().item()
            if bool(rsc.selected.any().item())
            else 0
        ),
        "selected_mask_count_max": int(
            selected_mask_counts[rsc.selected].max().item()
            if bool(rsc.selected.any().item())
            else 0
        ),
        "selected_contains_focus": bool(
            selected_targets.eq(int(args.focus_class)).any().item()
            if int(selected_targets.numel())
            else False
        ),
        "selected_all_positive_drop": bool(
            rsc.positive_drop[rsc.selected].all().item()
            if bool(rsc.selected.any().item())
            else True
        ),
        "selected_all_eligible": bool(
            rsc.eligible[rsc.selected].all().item()
            if bool(rsc.selected.any().item())
            else True
        ),
        "confidence_drop_mean": float(rsc.confidence_drop.mean().item()),
        "selected_confidence_drop_mean": float(
            rsc.confidence_drop[rsc.selected].mean().item()
            if bool(rsc.selected.any().item())
            else 0.0
        ),
        "gradient_finite": bool(torch.isfinite(rsc.gradient).all().item()),
        "gradient_norm": float(rsc.gradient.float().norm().item()),
        "clean_eval_ce": float(F.cross_entropy(clean_eval_logits.float(), targets).item()),
        "preliminary_ce": float(
            F.cross_entropy(rsc.preliminary_logits.float(), targets).item()
        ),
    }
    return logits, telemetry


def _train_variant(
    *,
    name: str,
    candidate: bool,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, object]]:
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device).train()
    initial_state = _state_sha256(model)
    initial_groups = _parameter_group_hashes(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
    )
    train_loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"class1_protected_rsc_{name}_fit",
        seed=int(args.seed) + 200,
    )
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"class1_protected_rsc_{name}_holdout",
        seed=int(args.seed) + 100,
    )
    ordered_indices: list[int] = []
    history: list[Dict[str, object]] = []
    rng_history: list[Dict[str, object]] = []
    gradient_seen = {key: False for key in initial_groups}
    all_gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (images, targets, metadata) in enumerate(train_loader):
        if batch_index >= int(args.max_train_batches):
            break
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        ordered_indices.extend(int(value) for value in sample_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if candidate:
            logits, telemetry = _candidate_forward(
                model=model,
                images=images,
                targets=targets,
                metadata=metadata,
                device=device,
                args=args,
            )
        else:
            model.train()
            features = _features_from_batch(model, images, metadata, device=device)
            logits = classification_logits_from_features(model, features)
            telemetry = {
                "rows": int(targets.numel()),
                "eligible_rows": 0,
                "positive_drop_rows": 0,
                "selected_rows": 0,
                "selection_limit": 0,
                "drop_count": 0,
                "class1_rows": int(targets.eq(int(args.focus_class)).sum().item()),
                "class1_masked_channels": 0,
                "selected_mask_count_min": 0,
                "selected_mask_count_max": 0,
                "selected_contains_focus": False,
                "selected_all_positive_drop": True,
                "selected_all_eligible": True,
                "confidence_drop_mean": 0.0,
                "selected_confidence_drop_mean": 0.0,
                "gradient_finite": True,
                "gradient_norm": 0.0,
                "clean_eval_ce": 0.0,
                "preliminary_ce": 0.0,
            }
        loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"Non-finite {name} loss at batch {batch_index}.")
        loss.backward()
        group_norm_squared = {key: 0.0 for key in initial_groups}
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            group = _parameter_group(parameter_name)
            finite = bool(torch.isfinite(gradient).all().item())
            nonzero = bool(torch.count_nonzero(gradient).item())
            all_gradients_finite = bool(all_gradients_finite and finite)
            gradient_seen[group] = bool(gradient_seen.get(group, False) or nonzero)
            group_norm_squared[group] = group_norm_squared.get(group, 0.0) + float(
                gradient.detach().float().square().sum().item()
            )
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - started)
        rng_history.append(_rng_summary(_rng_snapshot()))
        history.append(
            {
                "batch": int(batch_index),
                "loss": float(loss.detach().item()),
                "step_seconds": elapsed,
                "gradient_norms": {
                    key: math.sqrt(value) for key, value in group_norm_squared.items()
                },
                "rsc": telemetry,
            }
        )

    if len(history) != int(args.max_train_batches):
        raise ValueError(
            f"Expected {args.max_train_batches} train batches, observed {len(history)}."
        )
    adapted_predictions = _predict_fp32(model=model, loader=holdout_loader, device=device)
    final_state = _state_sha256(model)
    final_groups = _parameter_group_hashes(model)
    movement = {
        key: initial_groups.get(key) != final_groups.get(key)
        for key in sorted(set(initial_groups).union(final_groups))
    }
    measured_steps = [float(row["step_seconds"]) for row in history[5:]]
    peak_vram_gib = float(torch.cuda.max_memory_allocated(device) / (1024.0**3))
    result = {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(ordered_indices),
        "train_order_sha256": _ordered_index_sha256(ordered_indices),
        "trainable_parameter_count": sum(int(value.numel()) for value in model.parameters()),
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "state_changed": initial_state != final_state,
        "parameter_group_hashes_before": initial_groups,
        "parameter_group_hashes_after": final_groups,
        "parameter_group_movement": movement,
        "gradient_seen": gradient_seen,
        "all_gradients_finite": all_gradients_finite,
        "median_step_seconds_excluding_warmup": float(np.median(measured_steps)),
        "peak_vram_gib": peak_vram_gib,
        "history": history,
        "post_forward_rng": rng_history,
        "adapted_predictions": adapted_predictions,
        "loader": {"train": loader_summary, "holdout": holdout_loader_summary},
    }
    model = model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, result


class _FullExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _export_candidate(
    *,
    candidate: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    # ONNX receives CPU inputs below, so normalize the model device explicitly.
    # Several auditors keep a CPU alias to a model that was later moved in place.
    candidate = candidate.cpu().eval()
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    cpu_images = images[:1].detach().float().cpu()
    cpu_bbox = (
        bbox[:1].detach().float().cpu()
        if torch.is_tensor(bbox)
        else torch.zeros(1, 8)
    )
    cpu_mask = (
        image_mask[:1].detach().bool().cpu()
        if torch.is_tensor(image_mask)
        else torch.ones(1, int(cpu_images.size(2)), int(cpu_images.size(3)), dtype=torch.bool)
    )
    path = output_dir / "class1_protected_rsc_candidate.onnx"
    try:
        result = _onnx_compare(
            wrapper=_FullExportWrapper(candidate),
            inputs=(cpu_images, cpu_bbox, cpu_mask),
            input_names=("images", "bbox", "image_mask"),
            path=path,
        )
    except Exception as error:
        result = _failed_export(path, error)
    if path.is_file():
        result["deleted_after_verification"] = True
        result["deleted_size_bytes"] = int(path.stat().st_size)
        path.unlink()
    else:
        result["deleted_after_verification"] = False
        result["deleted_size_bytes"] = 0
    return result


def _write_clean_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    raw: Sequence[Mapping[str, object]],
    control: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> None:
    source = {row.sample_index: row for row in rows}
    variants = {"raw": raw, "control": control, "candidate": candidate}
    indexed = {
        name: {int(row["sample_index"]): row for row in values}
        for name, values in variants.items()
    }
    fields = ["sample_index", "source_stem", "image_path", "fold", "target"]
    for name in variants:
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(5))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw_row in raw:
            sample_index = int(raw_row["sample_index"])
            source_row = source[sample_index]
            output: Dict[str, object] = {
                "sample_index": sample_index,
                "source_stem": source_row.source_stem,
                "image_path": str(source_row.image_path),
                "fold": source_row.fold,
                "target": source_row.target,
            }
            for name in variants:
                value = indexed[name][sample_index]
                output[f"{name}_prediction"] = int(value["prediction"])
                for class_index in range(5):
                    output[f"{name}_prob_{class_index}"] = value[f"prob_{class_index}"]
            writer.writerow(output)


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    clean: Mapping[str, object],
    illumination: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    delta = clean["delta"]
    transitions = clean["transitions"]
    control_support = int(clean["control"]["predicted_support"][1])
    candidate_support = int(clean["candidate"]["predicted_support"][1])
    clean_checks = {
        "macro_f1_delta_nonnegative": float(delta["macro_f1"]) >= 0.0,
        "class1_f1_delta_gte_0p005": float(delta["class1_f1"]) >= 0.005,
        "class1_precision_delta_gte_0p005": float(delta["class1_precision"]) >= 0.005,
        "class1_recall_delta_gte_minus_0p005": float(delta["class1_recall"]) >= -0.005,
        "restricted_focus_fp_reduction_gte_2": int(
            transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "focus_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
        >= int(transitions["focus_tp_break"]),
        "corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "max_nonfocus_f1_drop_lte_0p010": float(clean["maximum_nonfocus_f1_drop"])
        <= 0.010,
        "class1_support_at_least_95pct_control": candidate_support
        >= int(math.ceil(0.95 * control_support)),
    }
    precision_nonnegative = sum(
        float(row["delta"]["class1_precision"]) >= 0.0 for row in illumination
    )
    illumination_checks = {
        "all_class1_f1_deltas_gte_minus_0p010": all(
            float(row["delta"]["class1_f1"]) >= -0.010 for row in illumination
        ),
        "all_class1_recall_deltas_gte_minus_0p015": all(
            float(row["delta"]["class1_recall"]) >= -0.015 for row in illumination
        ),
        "precision_nonnegative_in_at_least_two_conditions": precision_nonnegative >= 2,
        "aggregate_rescues_gte_tp_breaks": sum(
            int(row["transitions"]["focus_fn_rescue"]) for row in illumination
        )
        >= sum(int(row["transitions"]["focus_tp_break"]) for row in illumination),
        "no_condition_increases_restricted_focus_fp": all(
            int(row["transitions"]["restricted_focus_fp_reduction"]) >= 0
            for row in illumination
        ),
    }
    all_checks = {**dict(structural_checks), **clean_checks, **illumination_checks}
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "clean_checks": clean_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_smoke_authorized": not failed,
        "full_train_authorized": False,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked RSC Stage A requires CUDA.")
    provenance = _verify_sources(args)
    equation = _equation_diagnostics(args)
    output_dir = _prepare_output_dir(args.output_dir)
    paths = _source_paths(args)
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
    torch.use_deterministic_algorithms(True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    source_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if len(class_names) != 5 or not isinstance(source_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    set_seed(int(args.seed), deterministic=True)
    prototype = create_model(num_classes=len(class_names), model_config=source_config)
    load_model_state(prototype, dict(state), strict=True)
    prototype = prototype.eval()
    parameter_count = sum(int(value.numel()) for value in prototype.parameters())

    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(paths["data"], class_name_mode="raw", expected_num_classes=5)
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
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    np.random.default_rng(int(args.seed)).shuffle(shuffled_fit)
    train_indices = shuffled_fit[:EXPECTED_TRAIN_BUDGET_ROWS].tolist()
    if (
        len(rows) != EXPECTED_TRAIN_ROWS
        or len(fit_indices) != 7372
        or len(holdout_indices) != EXPECTED_FOLD_COUNTS[0]
        or len(train_indices) != EXPECTED_TRAIN_BUDGET_ROWS
        or source_overlap
    ):
        raise ValueError("Locked train/fold/source-disjoint contract differs.")

    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="class1_protected_rsc_raw_holdout",
        seed=int(args.seed) + 100,
    )
    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="class1_protected_rsc_resource",
        seed=int(args.seed) + 300,
    )
    resource_images, _, resource_metadata = next(iter(resource_loader))
    raw_model = copy.deepcopy(prototype).to(device).eval()
    raw_predictions = _predict_fp32(model=raw_model, loader=holdout_loader, device=device)
    raw_cidt_mismatches = sum(
        int(row["prediction"]) != rows[int(row["sample_index"])].keeper_prediction
        for row in raw_predictions
    )
    del raw_model
    gc.collect()
    torch.cuda.empty_cache()

    control_model, control = _train_variant(
        name="control",
        candidate=False,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
    )
    candidate_model, candidate = _train_variant(
        name="candidate",
        candidate=True,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
    )
    clean = _comparison(
        control_rows=control["adapted_predictions"],
        candidate_rows=candidate["adapted_predictions"],
        num_classes=5,
        focus_class=int(args.focus_class),
    )

    condition_rows: Dict[str, Dict[str, Sequence[Mapping[str, object]]]] = {}
    illumination = []
    control_gpu = control_model.to(device).eval()
    candidate_gpu = candidate_model.to(device).eval()
    for condition_index, (condition, brightness, contrast) in enumerate(LIGHTING_CONDITIONS):
        loader, _ = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"class1_protected_rsc_{condition}",
            seed=int(args.seed) + 500 + condition_index,
        )
        control_rows = _predict_fp32(model=control_gpu, loader=loader, device=device)
        candidate_rows = _predict_fp32(model=candidate_gpu, loader=loader, device=device)
        comparison = _comparison(
            control_rows=control_rows,
            candidate_rows=candidate_rows,
            num_classes=5,
            focus_class=int(args.focus_class),
        )
        comparison["condition"] = condition
        illumination.append(comparison)
        condition_rows[condition] = {
            "control": control_rows,
            "candidate": candidate_rows,
        }
    control_model = control_gpu.cpu().eval()
    candidate_model = candidate_gpu.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()

    export = _export_candidate(
        candidate=candidate_model,
        images=resource_images,
        metadata=resource_metadata,
        output_dir=output_dir,
    )
    runtime_ratio = float(candidate["median_step_seconds_excluding_warmup"]) / max(
        1e-12, float(control["median_step_seconds_excluding_warmup"])
    )
    candidate_history = candidate["history"]
    required_movement_groups = {
        "stem",
        "input_tokens",
        "transformer",
        "normalization",
        "fine_grained_pool",
        "classifier",
        "cnn_fusion",
    }
    structural_checks = {
        "locked_sources_verified": True,
        "train_only_provenance": not provenance["validation_predictions_used"]
        and not provenance["test_data_used"],
        "source_groups_disjoint": not source_overlap,
        "raw_predictions_match_cidt": raw_cidt_mismatches == 0,
        "initial_states_identical": control["initial_state_sha256"]
        == candidate["initial_state_sha256"]
        == _state_sha256(prototype),
        "parameter_count_unchanged": int(control["trainable_parameter_count"])
        == int(candidate["trainable_parameter_count"])
        == parameter_count,
        "equation_gradient_exact": float(equation["gradient_max_abs_error"]) == 0.0,
        "equation_mask_bit_exact": bool(equation["preliminary_mask_bit_exact"]),
        "equation_drop_count_86": int(equation["drop_count"])
        == EXPECTED_DROP_COUNT,
        "equation_focus_protection_exact": int(equation["focus_masked_channels"])
        == 0,
        "equation_finite": bool(equation["finite"]),
        "train_orders_identical": control["train_order_sha256"]
        == candidate["train_order_sha256"],
        "post_forward_rng_identical": control["post_forward_rng"]
        == candidate["post_forward_rng"],
        "all_gradients_finite": bool(control["all_gradients_finite"])
        and bool(candidate["all_gradients_finite"]),
        "all_required_gradient_groups_seen": all(
            bool(candidate["gradient_seen"].get(group, False))
            for group in required_movement_groups
        ),
        "all_required_parameter_groups_moved": all(
            bool(candidate["parameter_group_movement"].get(group, False))
            for group in required_movement_groups
        ),
        "rsc_selected_rows_positive": sum(
            int(row["rsc"]["selected_rows"]) for row in candidate_history
        )
        > 0,
        "class1_never_masked": all(
            int(row["rsc"]["class1_masked_channels"]) == 0
            for row in candidate_history
        ),
        "selected_never_contains_class1": all(
            not bool(row["rsc"]["selected_contains_focus"])
            for row in candidate_history
        ),
        "selected_rows_all_positive_drop": all(
            bool(row["rsc"]["selected_all_positive_drop"])
            for row in candidate_history
        ),
        "selected_rows_all_eligible": all(
            bool(row["rsc"]["selected_all_eligible"])
            for row in candidate_history
        ),
        "selected_rows_mask_exactly_86": all(
            int(row["rsc"]["selected_rows"]) == 0
            or (
                int(row["rsc"]["drop_count"]) == EXPECTED_DROP_COUNT
                and int(row["rsc"]["selected_mask_count_min"])
                == EXPECTED_DROP_COUNT
                and int(row["rsc"]["selected_mask_count_max"])
                == EXPECTED_DROP_COUNT
            )
            for row in candidate_history
        ),
        "all_rsc_gradients_finite": all(
            bool(row["rsc"]["gradient_finite"]) for row in candidate_history
        ),
        "runtime_ratio_lte_1p35": runtime_ratio <= float(args.max_runtime_ratio),
        "peak_vram_lte_3p25_gib": float(candidate["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
        "static_onnx_succeeded": bool(export.get("succeeded", False)),
        "static_onnx_finite": bool(export.get("finite", False)),
        "static_onnx_error_lte_1e5": float(export.get("maximum_absolute_error", 1e9))
        <= MAX_ONNX_ERROR,
        "static_onnx_argmax_match": bool(export.get("argmax_match", False)),
    }
    decision = assess_stage_a(
        structural_checks=structural_checks,
        clean=clean,
        illumination=illumination,
    )

    clean_path = output_dir / "clean_predictions.csv"
    illumination_path = output_dir / "illumination_predictions.csv"
    history_path = output_dir / "training_history.json"
    _write_clean_predictions(
        clean_path,
        rows=rows,
        raw=raw_predictions,
        control=control["adapted_predictions"],
        candidate=candidate["adapted_predictions"],
    )
    _write_illumination_predictions(
        illumination_path,
        rows=rows,
        condition_rows=condition_rows,
    )
    history_path.write_text(
        json.dumps(
            {"control": control["history"], "candidate": candidate["history"]},
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    checkpoint_artifact = None
    if bool(decision["stage_b_smoke_authorized"]):
        checkpoint_path = output_dir / "stage_a_candidate.pt"
        torch.save(
            {
                "model_state": candidate_model.state_dict(),
                "model_config": dict(source_config),
                "class_names": class_names,
                "stage": "class1_protected_rsc_stage_a",
            },
            checkpoint_path,
        )
        checkpoint_artifact = {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
        }

    summary = {
        "protocol": "class1_protected_pooled_channel_rsc_stage_a_v1",
        "provenance": provenance,
        "dataset": {
            "train_rows": len(rows),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fit_rows": len(fit_indices),
            "train_budget_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "source_overlap": len(source_overlap),
            "validation_predictions_used": False,
            "test_data_used": False,
            "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
            "resource_loader": resource_loader_summary,
            "holdout_loader": holdout_loader_summary,
        },
        "model": {
            "parameter_count": parameter_count,
            "state_sha256": _state_sha256(prototype),
            "representation_dim": EXPECTED_REPRESENTATION_DIM,
            "drop_count": EXPECTED_DROP_COUNT,
        },
        "equation": equation,
        "config": to_serializable(vars(args)),
        "raw_cidt_prediction_mismatches": raw_cidt_mismatches,
        "control": {
            key: value
            for key, value in control.items()
            if key not in {"adapted_predictions", "history", "post_forward_rng"}
        },
        "candidate": {
            key: value
            for key, value in candidate.items()
            if key not in {"adapted_predictions", "history", "post_forward_rng"}
        },
        "rsc_aggregate": {
            "selected_rows": sum(
                int(row["rsc"]["selected_rows"]) for row in candidate_history
            ),
            "eligible_rows": sum(
                int(row["rsc"]["eligible_rows"]) for row in candidate_history
            ),
            "positive_drop_rows": sum(
                int(row["rsc"]["positive_drop_rows"]) for row in candidate_history
            ),
            "class1_rows": sum(
                int(row["rsc"]["class1_rows"]) for row in candidate_history
            ),
            "class1_masked_channels": sum(
                int(row["rsc"]["class1_masked_channels"])
                for row in candidate_history
            ),
            "mean_selected_confidence_drop": float(
                np.mean(
                    [
                        float(row["rsc"]["selected_confidence_drop_mean"])
                        for row in candidate_history
                    ]
                )
            ),
        },
        "runtime": {
            "control_median_step_seconds": control[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_median_step_seconds": candidate[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_control_ratio": runtime_ratio,
            "control_peak_vram_gib": control["peak_vram_gib"],
            "candidate_peak_vram_gib": candidate["peak_vram_gib"],
        },
        "clean": clean,
        "illumination": illumination,
        "export": export,
        "decision": decision,
        "stage_a_candidate_checkpoint": checkpoint_artifact,
        "artifacts": {
            "clean_predictions": {
                "path": str(clean_path.resolve()),
                "sha256": _sha256(clean_path),
            },
            "illumination_predictions": {
                "path": str(illumination_path.resolve()),
                "sha256": _sha256(illumination_path),
            },
            "training_history": {
                "path": str(history_path.resolve()),
                "sha256": _sha256(history_path),
            },
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    summary["summary_sha256"] = _sha256(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        result = _verify_sources(args)
        result["preflight_only"] = True
        result["output_dir_created"] = False
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return
    result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
