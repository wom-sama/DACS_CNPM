from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _effective_rank,
    _fit_grouped_readout,
    _l2_normalize,
    _source_stems,
    _transition_summary,
    _write_prediction_audit,
)


LITERATURE = [
    "https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html",
    "https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html",
]

READOUT_VARIANTS = (
    "head",
    "all_second_order",
    "core_first_order",
    "core_second_order",
    "ring_second_order",
    "context_second_order",
    "head_core_second_order",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether fixed compact second-order statistics from an eroded "
            "crop-bbox interior add source-grouped OOF and full-validation signal "
            "beyond the frozen TRKH head. Train and validation only."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--projection-rank", type=int, default=24)
    parser.add_argument("--projection-seed", type=int, default=20260711)
    parser.add_argument("--core-erode-ratio", type=float, default=0.20)
    parser.add_argument("--covariance-epsilon", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--logistic-c", type=float, default=0.3)
    parser.add_argument("--logistic-max-iter", type=int, default=600)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_fixed_orthogonal_projection(
    input_dim: int,
    rank: int,
    *,
    seed: int,
) -> Tensor:
    input_dim = int(input_dim)
    rank = min(input_dim, max(2, int(rank)))
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((input_dim, rank), generator=generator, dtype=torch.float32)
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection.contiguous()


def _valid_patch_mask(
    patch_indices: Tensor,
    key_padding_mask: Optional[Tensor],
    *,
    grid_size: Tuple[int, int],
) -> Tensor:
    grid_height, grid_width = [max(1, int(value)) for value in grid_size]
    valid = (patch_indices >= 0) & (patch_indices < grid_height * grid_width)
    if torch.is_tensor(key_padding_mask):
        if tuple(key_padding_mask.shape) != tuple(patch_indices.shape):
            raise ValueError("key_padding_mask must match patch_indices")
        valid = valid & ~key_padding_mask.to(device=valid.device, dtype=torch.bool)
    empty = ~valid.any(dim=1)
    if bool(empty.any().item()):
        valid = valid.clone()
        valid[empty, 0] = True
    return valid


def token_region_masks(
    *,
    crop_bbox: Tensor,
    patch_indices: Tensor,
    grid_size: Tuple[int, int],
    key_padding_mask: Optional[Tensor] = None,
    core_erode_ratio: float = 0.20,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    if crop_bbox.ndim != 2 or int(crop_bbox.size(1)) < 4:
        raise ValueError("crop_bbox must be [B,4]")
    if patch_indices.ndim != 2 or int(patch_indices.size(0)) != int(crop_bbox.size(0)):
        raise ValueError("patch_indices must be [B,N] and match crop_bbox")
    erode = float(core_erode_ratio)
    if not 0.0 <= erode < 0.5:
        raise ValueError("core_erode_ratio must be in [0, 0.5)")

    grid_height, grid_width = [max(1, int(value)) for value in grid_size]
    indices = patch_indices.to(dtype=torch.long)
    safe_indices = indices.clamp(min=0, max=grid_height * grid_width - 1)
    x = (safe_indices.remainder(grid_width).float() + 0.5) / float(grid_width)
    y = (safe_indices.div(grid_width, rounding_mode="floor").float() + 0.5) / float(grid_height)
    bbox = crop_bbox[:, :4].to(device=x.device, dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = bbox.unbind(dim=1)
    width = width.clamp_min(1.0 / float(grid_width))
    height = height.clamp_min(1.0 / float(grid_height))
    half_width = 0.5 * width
    half_height = 0.5 * height
    inner_half_width = half_width * (1.0 - 2.0 * erode)
    inner_half_height = half_height * (1.0 - 2.0 * erode)
    dx = (x - cx[:, None]).abs()
    dy = (y - cy[:, None]).abs()
    valid = _valid_patch_mask(indices, key_padding_mask, grid_size=grid_size)
    inside = (dx <= half_width[:, None] + 1e-7) & (dy <= half_height[:, None] + 1e-7) & valid
    core = (
        (dx <= inner_half_width[:, None] + 1e-7)
        & (dy <= inner_half_height[:, None] + 1e-7)
        & valid
    )
    ring = inside & ~core
    context = valid & ~inside

    core_fallback = ~core.any(dim=1)
    if bool(core_fallback.any().item()):
        normalized_distance = (dx / half_width[:, None]).square() + (dy / half_height[:, None]).square()
        normalized_distance = normalized_distance.masked_fill(~valid, float("inf"))
        nearest = normalized_distance.argmin(dim=1)
        core = core.clone()
        rows = torch.nonzero(core_fallback, as_tuple=False).flatten()
        core[rows, nearest[rows]] = True
        ring = inside & ~core

    ring_fallback = ~ring.any(dim=1)
    context_fallback = ~context.any(dim=1)
    stats = {
        "valid_count": valid.sum(dim=1),
        "inside_count": inside.sum(dim=1),
        "core_count": core.sum(dim=1),
        "ring_count": ring.sum(dim=1),
        "context_count": context.sum(dim=1),
        "core_fallback": core_fallback,
        "ring_fallback": ring_fallback,
        "context_fallback": context_fallback,
    }
    return core, ring, context, valid, stats


def _normalized_mask_weights(mask: Tensor, valid: Tensor) -> Tensor:
    selected = mask.to(dtype=torch.float32) * valid.to(dtype=torch.float32)
    empty = selected.sum(dim=1) <= 0.0
    if bool(empty.any().item()):
        selected = selected.clone()
        selected[empty] = valid[empty].to(dtype=torch.float32)
    return selected / selected.sum(dim=1, keepdim=True).clamp_min(1.0)


def compact_second_order_descriptor(
    tokens: Tensor,
    weights: Tensor,
    projection: Tensor,
    *,
    epsilon: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    if tokens.ndim != 3 or weights.ndim != 2:
        raise ValueError("tokens/weights must be [B,N,D]/[B,N]")
    if tuple(tokens.shape[:2]) != tuple(weights.shape):
        raise ValueError("weights must match token rows")
    if projection.ndim != 2 or int(projection.size(0)) != int(tokens.size(2)):
        raise ValueError("projection input dimension must match token dimension")
    normalized = F.layer_norm(tokens.float(), (int(tokens.size(2)),))
    projected = normalized @ projection.to(device=tokens.device, dtype=torch.float32)
    normalized_weights = weights.float() / weights.float().sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean = torch.einsum("bn,bnr->br", normalized_weights, projected)
    centered = projected - mean[:, None, :]
    covariance = torch.einsum("bn,bnr,bns->brs", normalized_weights, centered, centered)
    rank = int(projected.size(2))
    identity = torch.eye(rank, device=tokens.device, dtype=torch.float32).unsqueeze(0)
    covariance = 0.5 * (covariance + covariance.transpose(1, 2)) + float(epsilon) * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    square_root = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(float(epsilon)).sqrt()) @ eigenvectors.transpose(1, 2)
    row, column = torch.triu_indices(rank, rank, device=tokens.device)
    upper = square_root[:, row, column]
    off_diagonal = row != column
    if bool(off_diagonal.any().item()):
        upper = upper.clone()
        upper[:, off_diagonal] *= math.sqrt(2.0)
    mean = F.normalize(mean, dim=1, eps=1e-6)
    upper = F.normalize(upper, dim=1, eps=1e-6)
    descriptor = torch.cat((mean, upper), dim=1)
    return descriptor, mean


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(data.mean()) if data.size else 0.0,
        "p01": float(np.quantile(data, 0.01)) if data.size else 0.0,
        "p50": float(np.quantile(data, 0.50)) if data.size else 0.0,
        "p99": float(np.quantile(data, 0.99)) if data.size else 0.0,
    }


def _make_region_overlay(rgb: np.ndarray, region_grid: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    region = Image.fromarray(region_grid.astype(np.int32)).resize(
        (width, height),
        getattr(Image, "Resampling", Image).NEAREST,
    )
    values = np.asarray(region, dtype=np.int16)
    output = rgb.astype(np.float32).copy()
    colors = {
        0: (55.0, 135.0, 245.0),
        1: (250.0, 185.0, 35.0),
        2: (45.0, 220.0, 85.0),
    }
    alphas = {0: 0.16, 1: 0.30, 2: 0.32}
    for key, color in colors.items():
        mask = values == key
        if not bool(mask.any()):
            continue
        alpha = alphas[key]
        output[mask] = (1.0 - alpha) * output[mask] + alpha * np.asarray(color, dtype=np.float32)
    return output.clip(0.0, 255.0).round().astype(np.uint8)


def _write_preview(
    path: Path,
    preview: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
) -> None:
    tile = 192
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index in range(len(class_names)):
        if class_index not in preview:
            continue
        rgb, overlay = preview[class_index]
        row = Image.new("RGB", (tile * 2, tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        row.paste(Image.fromarray(overlay).resize((tile, tile), resampling.BILINEAR), (tile, 0))
        rows.append(row)
        manifest.append(
            {
                "row": len(rows) - 1,
                "class_index": int(class_index),
                "class_name": str(class_names[class_index]),
                "columns": ["object_crop", "kept_context_blue_boundary_amber_core_green"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 2, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _extract_split(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    projection: Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
    core_erode_ratio: float,
    covariance_epsilon: float,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    descriptor_batches: Dict[str, List[np.ndarray]] = {name: [] for name in READOUT_VARIANTS}
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    paths: List[str] = []
    stat_batches: Dict[str, List[np.ndarray]] = {
        "valid_count": [],
        "inside_count": [],
        "core_count": [],
        "ring_count": [],
        "context_count": [],
        "core_fallback": [],
        "ring_fallback": [],
        "context_fallback": [],
    }
    preview: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in dataset_paths_fn()] if callable(dataset_paths_fn) else []
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    seen = 0
    model.eval()
    projection_device = projection.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"interior-second-order-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                raise ValueError("Interior second-order probe requires tensor metadata")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            crop_bbox = metadata.get("crop_bbox")
            if not torch.is_tensor(crop_bbox):
                raise ValueError("crop_bbox metadata is required")
            crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            bbox = metadata.get("bbox")
            bbox = bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(bbox) else None
            image_mask = metadata.get("image_mask")
            image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True) if torch.is_tensor(image_mask) else None

            with autocast_context(device, bool(amp)):
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=crop_bbox,
                )
                if bbox is not None:
                    features["bbox"] = bbox
                elif crop_bbox is not None:
                    features["bbox"] = crop_bbox
                output = (
                    model.forward_heads(features)
                    if hasattr(model, "forward_heads")
                    else classification_logits_from_features(model, features)
                )
                logits, _, _ = extract_detection_from_model_output(output)
                head = extract_head_input_from_features(model, features).float()
            tokens = features.get("patches")
            patch_indices = features.get("patch_indices")
            if not torch.is_tensor(tokens) or tokens.ndim != 3:
                raise ValueError("Model features do not contain patch tokens")
            if not torch.is_tensor(patch_indices) or tuple(patch_indices.shape) != tuple(tokens.shape[:2]):
                raise ValueError("Model features do not contain aligned patch_indices")
            grid_size = tuple(int(value) for value in features.get("grid_size", (0, 0)))
            if len(grid_size) != 2 or min(grid_size) <= 0:
                raise ValueError("Model features do not contain a valid grid_size")
            core, ring, context, valid, region_stats = token_region_masks(
                crop_bbox=crop_bbox,
                patch_indices=patch_indices,
                grid_size=grid_size,
                key_padding_mask=features.get("memory_key_padding_mask"),
                core_erode_ratio=float(core_erode_ratio),
            )
            all_weights = _normalized_mask_weights(valid, valid)
            core_weights = _normalized_mask_weights(core, valid)
            ring_weights = _normalized_mask_weights(ring, valid)
            context_weights = _normalized_mask_weights(context, valid)
            all_second, _ = compact_second_order_descriptor(
                tokens,
                all_weights,
                projection_device,
                epsilon=float(covariance_epsilon),
            )
            core_second, core_first = compact_second_order_descriptor(
                tokens,
                core_weights,
                projection_device,
                epsilon=float(covariance_epsilon),
            )
            ring_second, _ = compact_second_order_descriptor(
                tokens,
                ring_weights,
                projection_device,
                epsilon=float(covariance_epsilon),
            )
            context_second, _ = compact_second_order_descriptor(
                tokens,
                context_weights,
                projection_device,
                epsilon=float(covariance_epsilon),
            )
            head_normalized = F.normalize(head, dim=1, eps=1e-6)
            variants = {
                "head": head_normalized,
                "all_second_order": all_second,
                "core_first_order": core_first,
                "core_second_order": core_second,
                "ring_second_order": ring_second,
                "context_second_order": context_second,
                "head_core_second_order": torch.cat((head_normalized, F.normalize(core_second, dim=1, eps=1e-6)), dim=1),
            }
            for name, value in variants.items():
                descriptor_batches[name].append(value.detach().float().cpu().numpy())
            probability_batches.append(logits.detach().float().softmax(dim=1).cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            for key, value in region_stats.items():
                stat_batches[key].append(value.detach().cpu().numpy())

            batch_count = int(labels.numel())
            raw_paths = metadata.get("paths", [])
            batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
            if dataset_paths and (len(batch_paths) != batch_count or not any(path.strip() for path in batch_paths)):
                batch_paths = dataset_paths[seen : seen + batch_count]
            paths.extend(batch_paths)
            fallback_indices = np.arange(seen, seen + batch_count, dtype=np.int64)
            sample_index = metadata.get("sample_index")
            if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_count:
                sample_index_batches.append(sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1))
            else:
                sample_index_batches.append(fallback_indices)
            seen += batch_count

            for row_index, target in enumerate(labels.detach().cpu().tolist()):
                if int(target) in preview:
                    continue
                rgb = (
                    images[row_index].detach().float().cpu().unsqueeze(0) * std_tensor + mean_tensor
                ).squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
                rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
                region_selected = torch.where(
                    core[row_index],
                    torch.full_like(patch_indices[row_index], 2),
                    torch.where(
                        ring[row_index],
                        torch.full_like(patch_indices[row_index], 1),
                        torch.zeros_like(patch_indices[row_index]),
                    ),
                )
                full_region = torch.full(
                    (grid_size[0] * grid_size[1],),
                    -1,
                    device=patch_indices.device,
                    dtype=torch.long,
                )
                selected_indices = patch_indices[row_index].clamp(0, grid_size[0] * grid_size[1] - 1)
                full_region.scatter_(0, selected_indices, region_selected)
                region_grid = full_region.reshape(grid_size).detach().cpu().numpy()
                preview[int(target)] = (rgb_u8, _make_region_overlay(rgb_u8, region_grid))

    return {
        "descriptors": {
            name: np.concatenate(batches, axis=0).astype(np.float32, copy=False)
            for name, batches in descriptor_batches.items()
        },
        "probabilities": np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate(label_batches, axis=0).astype(np.int64, copy=False),
        "sample_index": np.concatenate(sample_index_batches, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "region_stats": {
            key: np.concatenate(batches, axis=0)
            for key, batches in stat_batches.items()
        },
        "preview": preview,
    }


def assess_interior_second_order_readiness(
    *,
    train_samples: int,
    val_samples: int,
    mean_core_tokens: float,
    core_fallback_fraction: float,
    direct_metrics: Mapping[str, object],
    head_oof_metrics: Mapping[str, object],
    candidate_oof_metrics: Mapping[str, object],
    head_val_metrics: Mapping[str, object],
    candidate_val_metrics: Mapping[str, object],
    all_oof_metrics: Mapping[str, object],
    core_oof_metrics: Mapping[str, object],
    all_val_metrics: Mapping[str, object],
    core_val_metrics: Mapping[str, object],
    direct_transitions: Mapping[str, int],
    class1_error_delta_auc: float,
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_mean_core_tokens": 12.0,
        "max_core_fallback_fraction": 0.01,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_core_vs_all_oof_class1_gain": 0.005,
        "min_core_vs_all_val_class1_gain": 0.010,
        "min_val_class1_f1": 0.70,
        "max_direct_macro_drop": 0.003,
        "min_error_delta_auc": 0.60,
    }
    observed = {
        "oof_macro_gain": float(candidate_oof_metrics["macro_f1"]) - float(head_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(candidate_oof_metrics["focus_f1"]) - float(head_oof_metrics["focus_f1"]),
        "val_macro_gain": float(candidate_val_metrics["macro_f1"]) - float(head_val_metrics["macro_f1"]),
        "val_class1_gain": float(candidate_val_metrics["focus_f1"]) - float(head_val_metrics["focus_f1"]),
        "core_vs_all_oof_class1_gain": float(core_oof_metrics["focus_f1"]) - float(all_oof_metrics["focus_f1"]),
        "core_vs_all_val_class1_gain": float(core_val_metrics["focus_f1"]) - float(all_val_metrics["focus_f1"]),
        "direct_macro_drop": float(direct_metrics["macro_f1"]) - float(candidate_val_metrics["macro_f1"]),
        "class1_error_delta_auc": float(class1_error_delta_auc),
        "mean_core_tokens": float(mean_core_tokens),
        "core_fallback_fraction": float(core_fallback_fraction),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "core_geometry_supported": float(mean_core_tokens) >= float(thresholds["min_mean_core_tokens"]),
        "core_fallback_safe": float(core_fallback_fraction) <= float(thresholds["max_core_fallback_fraction"]),
        "oof_macro_gain": observed["oof_macro_gain"] >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"] >= float(thresholds["min_oof_class1_gain"]),
        "val_macro_gain": observed["val_macro_gain"] >= float(thresholds["min_val_macro_gain"]),
        "val_class1_gain": observed["val_class1_gain"] >= float(thresholds["min_val_class1_gain"]),
        "interior_specific_oof": observed["core_vs_all_oof_class1_gain"] >= float(thresholds["min_core_vs_all_oof_class1_gain"]),
        "interior_specific_val": observed["core_vs_all_val_class1_gain"] >= float(thresholds["min_core_vs_all_val_class1_gain"]),
        "val_class1_milestone": float(candidate_val_metrics["focus_f1"]) >= float(thresholds["min_val_class1_f1"]),
        "direct_macro_preserved": observed["direct_macro_drop"] <= float(thresholds["max_direct_macro_drop"]),
        "net_corrections_nonnegative": int(direct_transitions["corrections"]) >= int(direct_transitions["harms"]),
        "class1_recall_protected": int(direct_transitions["class1_fn_rescued"]) >= int(direct_transitions["class1_tp_broken"]),
        "class1_fp_control": int(direct_transitions["class1_fp_created"]) <= int(direct_transitions["class1_fp_removed"]),
        "class1_error_direction_separable": float(class1_error_delta_auc) >= float(thresholds["min_error_delta_auc"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "interior_second_order_ready": ready,
        "smoke_ready": ready,
        "smoke_permission": ready,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _compact_region_stats(payload: Mapping[str, np.ndarray]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key in ("valid_count", "inside_count", "core_count", "ring_count", "context_count"):
        result[key] = _summary_stats(np.asarray(payload[key], dtype=np.float64))
    for key in ("core_fallback", "ring_fallback", "context_fallback"):
        result[f"{key}_fraction"] = float(np.asarray(payload[key], dtype=np.float32).mean())
    return result


def _save_descriptor_cache(
    path: Path,
    payload: Mapping[str, object],
    *,
    class_names: Sequence[str],
) -> None:
    descriptors = payload["descriptors"]
    np.savez_compressed(
        path,
        **{name: np.asarray(descriptors[name], dtype=np.float32) for name in READOUT_VARIANTS},
        probabilities=np.asarray(payload["probabilities"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
        paths=np.asarray(payload["paths"], dtype=object),
        source_stem=_source_stems(np.asarray(payload["paths"], dtype=object)),
        classes=np.asarray(class_names, dtype=object),
    )


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if not 0.0 <= float(args.core_erode_ratio) < 0.5:
        raise ValueError("core-erode-ratio must be in [0, 0.5)")
    if int(args.folds) < 2:
        raise ValueError("folds must be at least 2")
    if float(args.logistic_c) <= 0.0:
        raise ValueError("logistic-c must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    embed_dim = int(getattr(model, "embed_dim", 0))
    if embed_dim <= 0:
        raise ValueError("Could not resolve model embed_dim")
    projection = build_fixed_orthogonal_projection(
        embed_dim,
        int(args.projection_rank),
        seed=int(args.projection_seed),
    )
    device = _resolve_device(str(args.device or ""))
    model.to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    start = time.perf_counter()

    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    max_samples = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    for split in ("train", "val"):
        dataset, split_class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples[split],
        )
        if class_names and list(split_class_names) != class_names:
            raise ValueError("Train/validation class order mismatch")
        class_names = list(split_class_names)
        split_payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            projection=projection,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=split,
            core_erode_ratio=float(args.core_erode_ratio),
            covariance_epsilon=float(args.covariance_epsilon),
            mean=mean,
            std=std,
        )

    train = split_payloads["train"]
    val = split_payloads["val"]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    train_groups = _source_stems(np.asarray(train["paths"], dtype=object))
    readouts: Dict[str, Dict[str, object]] = {}
    for name in READOUT_VARIANTS:
        print(f"fitting grouped OOF readout: {name}", flush=True)
        result = _fit_grouped_readout(
            train_features=np.asarray(train["descriptors"][name], dtype=np.float32),
            train_labels=train_labels,
            train_groups=train_groups,
            val_features=np.asarray(val["descriptors"][name], dtype=np.float32),
            folds=int(args.folds),
            c_value=float(args.logistic_c),
            max_iter=int(args.logistic_max_iter),
            seed=42,
            class_names=class_names,
        )
        result["val_metrics"] = _classification_metrics(
            val_labels,
            np.asarray(result["val_probabilities"], dtype=np.float32),
            class_names=class_names,
        )
        readouts[name] = result

    direct_val_probabilities = np.asarray(val["probabilities"], dtype=np.float32)
    direct_metrics = _classification_metrics(
        val_labels,
        direct_val_probabilities,
        class_names=class_names,
    )
    candidate_val_probabilities = np.asarray(
        readouts["head_core_second_order"]["val_probabilities"],
        dtype=np.float32,
    )
    direct_transitions = _transition_summary(
        val_labels,
        direct_val_probabilities,
        candidate_val_probabilities,
    )
    head_transitions = _transition_summary(
        val_labels,
        np.asarray(readouts["head"]["val_probabilities"], dtype=np.float32),
        candidate_val_probabilities,
    )
    direct_predictions = direct_val_probabilities.argmax(axis=1)
    recall_mask = (val_labels == 1) & (direct_predictions != 1)
    false_positive_mask = (val_labels != 1) & (direct_predictions == 1)
    error_mask = recall_mask | false_positive_mask
    error_labels = recall_mask[error_mask].astype(np.int64)
    probability_delta = (
        candidate_val_probabilities[:, 1]
        - np.asarray(readouts["head"]["val_probabilities"], dtype=np.float32)[:, 1]
    )
    error_auc = (
        float(roc_auc_score(error_labels, probability_delta[error_mask]))
        if int(np.unique(error_labels).size) == 2
        else 0.5
    )

    compact_region_stats = {
        split: _compact_region_stats(split_payloads[split]["region_stats"])
        for split in ("train", "val")
    }
    gate = assess_interior_second_order_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        mean_core_tokens=float(compact_region_stats["train"]["core_count"]["mean"]),
        core_fallback_fraction=float(compact_region_stats["train"]["core_fallback_fraction"]),
        direct_metrics=direct_metrics,
        head_oof_metrics=readouts["head"]["oof_metrics"],
        candidate_oof_metrics=readouts["head_core_second_order"]["oof_metrics"],
        head_val_metrics=readouts["head"]["val_metrics"],
        candidate_val_metrics=readouts["head_core_second_order"]["val_metrics"],
        all_oof_metrics=readouts["all_second_order"]["oof_metrics"],
        core_oof_metrics=readouts["core_second_order"]["oof_metrics"],
        all_val_metrics=readouts["all_second_order"]["val_metrics"],
        core_val_metrics=readouts["core_second_order"]["val_metrics"],
        direct_transitions=direct_transitions,
        class1_error_delta_auc=error_auc,
    )

    for split in ("train", "val"):
        payload = split_payloads[split]
        _save_descriptor_cache(
            output_dir / f"{split}_interior_second_order_descriptors.npz",
            payload,
            class_names=class_names,
        )
        _write_prediction_audit(
            output_dir / f"{split}_readout_predictions.csv",
            labels=np.asarray(payload["labels"], dtype=np.int64),
            sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
            paths=np.asarray(payload["paths"], dtype=object),
            base_probabilities=np.asarray(payload["probabilities"], dtype=np.float32),
            variants={
                name: np.asarray(
                    readouts[name]["oof_probabilities" if split == "train" else "val_probabilities"],
                    dtype=np.float32,
                )
                for name in READOUT_VARIANTS
            },
        )
    _write_preview(
        output_dir / "interior_region_preview_train.png",
        train["preview"],
        class_names,
    )

    effective_ranks = {
        name: _effective_rank(
            _l2_normalize(np.asarray(train["descriptors"][name], dtype=np.float32)),
            max_rows=2048,
        )
        for name in READOUT_VARIANTS
    }
    compact_readouts = {
        name: {
            "feature_dim": int(np.asarray(train["descriptors"][name]).shape[1]),
            "oof_metrics": readouts[name]["oof_metrics"],
            "val_metrics": readouts[name]["val_metrics"],
            "fold_iterations": readouts[name]["fold_iterations"],
            "final_iterations": readouts[name]["final_iterations"],
        }
        for name in READOUT_VARIANTS
    }
    summary = {
        "mode": "interior_second_order_readiness_precheck",
        "guardrail": (
            "Frozen keeper; one fixed orthogonal projection and geometry; train-only grouped OOF plus "
            "full validation. No test, model training, checkpoint, trainable manifest, or raw-data edit."
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "train_samples": int(train_labels.size),
        "val_samples": int(val_labels.size),
        "source_group_count": int(np.unique(train_groups).size),
        "class_names": class_names,
        "protocol": {
            "projection": "fixed_gaussian_qr_orthogonal",
            "projection_rank": int(projection.shape[1]),
            "projection_seed": int(args.projection_seed),
            "core_erode_ratio": float(args.core_erode_ratio),
            "covariance_normalization": "exact_symmetric_matrix_square_root_upper_triangle",
            "covariance_epsilon": float(args.covariance_epsilon),
            "candidate": "head_core_second_order",
            "controls": list(READOUT_VARIANTS[:-1]),
            "candidate_sweep": False,
        },
        "region_stats": compact_region_stats,
        "effective_ranks": effective_ranks,
        "direct_keeper_val_metrics": direct_metrics,
        "readout": {
            "folds": int(args.folds),
            "logistic_c": float(args.logistic_c),
            "max_iter": int(args.logistic_max_iter),
            "selection": "none; all variants predeclared and only head_core_second_order is candidate",
            "variants": compact_readouts,
        },
        "transitions": {
            "direct_keeper_to_candidate": direct_transitions,
            "head_readout_to_candidate": head_transitions,
            "direct_class1_fn_support": int(recall_mask.sum()),
            "direct_class1_fp_support": int(false_positive_mask.sum()),
            "class1_error_delta_auc": error_auc,
        },
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - start),
        "literature": LITERATURE,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_written": False,
        "trainable_manifest_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    head_metrics = compact_readouts["head"]["val_metrics"]
    candidate_metrics = compact_readouts["head_core_second_order"]["val_metrics"]
    readme = [
        "# Interior-Core Compact Second-Order Readiness Precheck",
        "",
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`",
        f"- Direct keeper macro/class1 F1: `{float(direct_metrics['macro_f1']):.6f}/{float(direct_metrics['focus_f1']):.6f}`",
        f"- Head readout macro/class1 F1: `{float(head_metrics['macro_f1']):.6f}/{float(head_metrics['focus_f1']):.6f}`",
        f"- Head+core-second-order macro/class1 F1: `{float(candidate_metrics['macro_f1']):.6f}/{float(candidate_metrics['focus_f1']):.6f}`",
        f"- Mean core tokens/fallback: `{float(compact_region_stats['train']['core_count']['mean']):.3f}/{float(compact_region_stats['train']['core_fallback_fraction']):.6f}`",
        f"- Class1 FN-vs-FP delta AUC: `{error_auc:.6f}`",
        f"- Smoke ready: `{str(bool(gate['smoke_ready'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "This is a diagnostic-only gate. It does not train or modify TRKH and does not use test.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
