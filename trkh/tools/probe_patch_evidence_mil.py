from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _collate_classification,
    _resolve_device,
)
from trkh.tools.probe_pairwise_feature_verifier import (
    _fit_pair_models,
    apply_pairwise_verifiers,
    build_verifier_feature_names,
    parse_pairs,
    write_pair_model_export,
)


Pair = Tuple[int, int]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether frozen TRKH patch tokens retain spatial class evidence. "
            "This applies the trained classification head to final patch tokens, "
            "fits train-only pair verifiers on summary statistics, and never reads "
            "test unless explicitly requested."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-train-data",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional extra train-only data.yaml files used only to fit the patch "
            "verifier. Validation/test still come from --data."
        ),
    )
    parser.add_argument("--auxiliary-train-weight", type=float, default=0.25)
    parser.add_argument(
        "--source-domain-feature",
        action="store_true",
        default=False,
        help="Append a source-domain indicator to verifier features when auxiliary train data is used.",
    )
    parser.add_argument("--pairs", type=str, default="0-1,1-2,1-4,2-3")
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--spatial-evidence-features",
        action="store_true",
        default=False,
        help="Append interior-vs-border patch evidence features for spatial reliability diagnostics.",
    )
    parser.add_argument("--spatial-interior-erode", type=int, default=1)
    parser.add_argument(
        "--bbox-token-prior-source",
        type=str,
        default="crop_bbox",
        choices=("bbox", "crop_bbox"),
    )
    parser.add_argument("--min-pair-probability", type=float, default=0.02)
    parser.add_argument("--max-pair-margin", type=float, default=0.40)
    parser.add_argument("--verifier-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--verifier-model", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument("--extra-trees-n-estimators", type=int, default=300)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=5)
    parser.add_argument("--extra-trees-max-depth", type=int, default=0)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _valid_patch_mask(
    local_logits: Tensor,
    key_padding_mask: Optional[Tensor],
) -> Tensor:
    batch_size, token_count = local_logits.shape[:2]
    if torch.is_tensor(key_padding_mask) and key_padding_mask.shape[:2] == (batch_size, token_count):
        valid = ~key_padding_mask.to(device=local_logits.device, dtype=torch.bool)
    else:
        valid = torch.ones((batch_size, token_count), device=local_logits.device, dtype=torch.bool)
    empty = ~valid.any(dim=1)
    if bool(empty.any().item()):
        valid[empty] = True
    return valid


def _bbox_patch_weights(
    local_logits: Tensor,
    bbox_prior: Optional[Tensor],
    valid_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    batch_size, token_count = local_logits.shape[:2]
    if torch.is_tensor(bbox_prior) and bbox_prior.shape[:2] == (batch_size, token_count):
        weights = bbox_prior.to(device=local_logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    else:
        weights = torch.ones((batch_size, token_count), device=local_logits.device, dtype=torch.float32)
    weights = weights * valid_mask.to(dtype=torch.float32)
    empty_weight = weights.sum(dim=1) <= 1e-6
    if bool(empty_weight.any().item()):
        weights[empty_weight] = valid_mask[empty_weight].to(dtype=torch.float32)
    bbox_mask = (weights > 0.05) & valid_mask
    empty_mask = ~bbox_mask.any(dim=1)
    if bool(empty_mask.any().item()):
        bbox_mask[empty_mask] = valid_mask[empty_mask]
    return weights, bbox_mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    return (values * weights).sum(dim=1) / denom


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    filled = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    result = filled.max(dim=1).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _masked_min(values: Tensor, mask: Tensor) -> Tensor:
    return -_masked_max(-values, mask)


def _masked_topk_mean(values: Tensor, mask: Tensor, top_k: int) -> Tensor:
    k = max(1, min(int(top_k), int(values.size(1))))
    filled = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    selected = torch.topk(filled, k=k, dim=1).values
    finite = torch.isfinite(selected)
    selected = torch.where(finite, selected, torch.zeros_like(selected))
    denom = finite.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    return selected.sum(dim=1) / denom


def _top1_fraction(local_logits: Tensor, mask: Tensor, class_count: int) -> Tensor:
    top1 = local_logits.argmax(dim=-1)
    one_hot = F.one_hot(top1, num_classes=int(class_count)).to(dtype=local_logits.dtype)
    weights = mask.to(device=local_logits.device, dtype=local_logits.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (one_hot * weights).sum(dim=1) / denom


def _weighted_top1_fraction(local_logits: Tensor, weights: Tensor, class_count: int) -> Tensor:
    top1 = local_logits.argmax(dim=-1)
    one_hot = F.one_hot(top1, num_classes=int(class_count)).to(dtype=local_logits.dtype)
    weights = weights.to(device=local_logits.device, dtype=local_logits.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    return (one_hot * weights).sum(dim=1) / denom


def _ensure_non_empty_mask(mask: Tensor, fallback: Tensor) -> Tensor:
    result = mask.to(dtype=torch.bool).clone()
    fallback = fallback.to(device=result.device, dtype=torch.bool)
    empty = ~result.any(dim=1)
    if bool(empty.any().item()):
        result[empty] = fallback[empty]
    empty = ~result.any(dim=1)
    if bool(empty.any().item()):
        result[empty] = True
    return result


def _spatial_evidence_masks(
    *,
    valid_mask: Tensor,
    bbox_mask: Tensor,
    bbox_weights: Tensor,
    interior_erode: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split patch tokens into object-interior and object/border reliability regions."""

    batch_size, token_count = valid_mask.shape[:2]
    erode = max(0, int(interior_erode))
    side = int(round(math.sqrt(float(token_count))))
    if side * side != token_count or side <= 2 * erode:
        interior_mask = _ensure_non_empty_mask(bbox_mask, valid_mask)
        border_mask = _ensure_non_empty_mask(valid_mask & ~interior_mask, bbox_mask)
    else:
        coords = torch.arange(token_count, device=valid_mask.device)
        rows = coords // side
        cols = coords % side
        frame_keep = (
            (rows >= erode)
            & (rows < side - erode)
            & (cols >= erode)
            & (cols < side - erode)
        ).view(1, token_count)
        bbox_float = bbox_mask.to(dtype=torch.float32).view(batch_size, 1, side, side)
        if erode > 0:
            kernel = 2 * erode + 1
            eroded = 1.0 - F.max_pool2d(1.0 - bbox_float, kernel_size=kernel, stride=1, padding=erode)
            eroded_bbox = eroded.view(batch_size, token_count) > 0.5
        else:
            eroded_bbox = bbox_mask
        interior_mask = _ensure_non_empty_mask(eroded_bbox & frame_keep & valid_mask, bbox_mask & frame_keep)
        border_mask = _ensure_non_empty_mask((bbox_mask & valid_mask) & ~interior_mask, valid_mask & ~frame_keep)
    interior_weights = bbox_weights * interior_mask.to(device=bbox_weights.device, dtype=bbox_weights.dtype)
    border_weights = bbox_weights * border_mask.to(device=bbox_weights.device, dtype=bbox_weights.dtype)
    empty_interior_weights = interior_weights.sum(dim=1) <= 1e-6
    if bool(empty_interior_weights.any().item()):
        interior_weights[empty_interior_weights] = interior_mask[empty_interior_weights].to(dtype=bbox_weights.dtype)
    empty_border_weights = border_weights.sum(dim=1) <= 1e-6
    if bool(empty_border_weights.any().item()):
        border_weights[empty_border_weights] = border_mask[empty_border_weights].to(dtype=bbox_weights.dtype)
    return interior_mask, border_mask, interior_weights, border_weights


def _pair_margin_features(
    local_logits: Tensor,
    *,
    valid_mask: Tensor,
    bbox_mask: Tensor,
    bbox_weights: Tensor,
    pairs: Sequence[Pair],
    top_k: int,
) -> List[Tensor]:
    parts: List[Tensor] = []
    for a, b in pairs:
        margin = (local_logits[:, :, int(b)] - local_logits[:, :, int(a)]).unsqueeze(-1)
        parts.extend(
            [
                _masked_mean(margin, valid_mask),
                _masked_max(margin, valid_mask),
                _masked_min(margin, valid_mask),
                _masked_topk_mean(margin, valid_mask, top_k),
                _weighted_mean(margin, bbox_weights),
                _masked_max(margin, bbox_mask),
                _masked_min(margin, bbox_mask),
                _masked_topk_mean(margin, bbox_mask, top_k),
                _masked_mean((margin > 0).to(dtype=local_logits.dtype), valid_mask),
                _weighted_mean((margin > 0).to(dtype=local_logits.dtype), bbox_weights),
            ]
        )
    return parts


def summarize_patch_evidence(
    local_logits: Tensor,
    *,
    key_padding_mask: Optional[Tensor] = None,
    bbox_prior: Optional[Tensor] = None,
    pairs: Sequence[Pair] = (),
    top_k: int = 4,
    spatial_evidence_features: bool = False,
    spatial_interior_erode: int = 1,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """Summarize dense patch readouts without storing full patch grids."""

    if local_logits.ndim != 3:
        raise ValueError("local_logits must have shape [B, N, C].")
    class_count = int(local_logits.size(-1))
    local_logits = local_logits.float()
    valid_mask = _valid_patch_mask(local_logits, key_padding_mask)
    bbox_weights, bbox_mask = _bbox_patch_weights(local_logits, bbox_prior, valid_mask)

    patch_mean = _masked_mean(local_logits, valid_mask)
    patch_max = _masked_max(local_logits, valid_mask)
    patch_topk_mean = _masked_topk_mean(local_logits, valid_mask, top_k)
    bbox_mean = _weighted_mean(local_logits, bbox_weights)
    bbox_max = _masked_max(local_logits, bbox_mask)
    bbox_topk_mean = _masked_topk_mean(local_logits, bbox_mask, top_k)
    all_top1_fraction = _top1_fraction(local_logits, valid_mask, class_count)
    bbox_top1_fraction = _weighted_top1_fraction(local_logits, bbox_weights, class_count)

    feature_parts: List[Tensor] = [
        patch_mean,
        patch_max,
        patch_topk_mean,
        bbox_mean,
        bbox_max,
        bbox_topk_mean,
        all_top1_fraction,
        bbox_top1_fraction,
    ]
    feature_parts.extend(
        _pair_margin_features(
            local_logits,
            valid_mask=valid_mask,
            bbox_mask=bbox_mask,
            bbox_weights=bbox_weights,
            pairs=pairs,
            top_k=top_k,
        )
    )
    if bool(spatial_evidence_features):
        interior_mask, border_mask, interior_weights, border_weights = _spatial_evidence_masks(
            valid_mask=valid_mask,
            bbox_mask=bbox_mask,
            bbox_weights=bbox_weights,
            interior_erode=int(spatial_interior_erode),
        )
        interior_mean = _masked_mean(local_logits, interior_mask)
        interior_max = _masked_max(local_logits, interior_mask)
        interior_topk_mean = _masked_topk_mean(local_logits, interior_mask, top_k)
        interior_weighted_mean = _weighted_mean(local_logits, interior_weights)
        interior_top1_fraction = _top1_fraction(local_logits, interior_mask, class_count)
        border_mean = _masked_mean(local_logits, border_mask)
        border_max = _masked_max(local_logits, border_mask)
        border_topk_mean = _masked_topk_mean(local_logits, border_mask, top_k)
        border_weighted_mean = _weighted_mean(local_logits, border_weights)
        border_top1_fraction = _top1_fraction(local_logits, border_mask, class_count)
        feature_parts.extend(
            [
                interior_mean,
                interior_max,
                interior_topk_mean,
                interior_weighted_mean,
                interior_top1_fraction,
                border_mean,
                border_max,
                border_topk_mean,
                border_weighted_mean,
                border_top1_fraction,
            ]
        )
        feature_parts.extend(
            _pair_margin_features(
                local_logits,
                valid_mask=interior_mask,
                bbox_mask=interior_mask,
                bbox_weights=interior_weights,
                pairs=pairs,
                top_k=top_k,
            )
        )
        feature_parts.extend(
            _pair_margin_features(
                local_logits,
                valid_mask=border_mask,
                bbox_mask=border_mask,
                bbox_weights=border_weights,
                pairs=pairs,
                top_k=top_k,
            )
        )
    else:
        interior_mask = None
        border_mask = None
    feature_tensor = torch.cat([part.flatten(1) for part in feature_parts], dim=1)
    method_logits = {
        "patch_mean": patch_mean,
        "patch_max": patch_max,
        "patch_topk_mean": patch_topk_mean,
        "bbox_mean": bbox_mean,
        "bbox_max": bbox_max,
        "bbox_topk_mean": bbox_topk_mean,
    }
    if bool(spatial_evidence_features):
        method_logits.update(
            {
                "interior_mean": interior_mean,
                "interior_max": interior_max,
                "interior_topk_mean": interior_topk_mean,
                "border_mean": border_mean,
                "border_max": border_max,
                "border_topk_mean": border_topk_mean,
            }
        )
    diagnostics = {
        "valid_mask": valid_mask,
        "bbox_mask": bbox_mask,
        "bbox_weights": bbox_weights,
        "local_top1": local_logits.argmax(dim=-1),
    }
    if interior_mask is not None and border_mask is not None:
        diagnostics["interior_mask"] = interior_mask
        diagnostics["border_mask"] = border_mask
    return feature_tensor, method_logits, diagnostics


def build_patch_verifier_features(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    patch_features: np.ndarray,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    logs = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32, copy=False)
    sorted_probs = np.sort(probs, axis=1)
    margins = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    confidence = sorted_probs[:, -1].reshape(-1, 1)
    return np.concatenate(
        [
            np.asarray(embeddings, dtype=np.float32),
            probs,
            logs,
            margins.astype(np.float32, copy=False),
            confidence.astype(np.float32, copy=False),
            np.asarray(patch_features, dtype=np.float32),
        ],
        axis=1,
    )


def _class_vector_feature_names(prefix: str, class_count: int) -> List[str]:
    return [f"{prefix}_class_{class_index}" for class_index in range(max(0, int(class_count)))]


def _pair_margin_feature_names(pairs: Sequence[Pair], *, prefix: str = "pair") -> List[str]:
    names: List[str] = []
    suffixes = [
        "margin_mean_all",
        "margin_max_all",
        "margin_min_all",
        "margin_topk_mean_all",
        "margin_weighted_mean_bbox",
        "margin_max_bbox",
        "margin_min_bbox",
        "margin_topk_mean_bbox",
        "positive_fraction_all",
        "positive_fraction_bbox",
    ]
    for a, b in pairs:
        pair_prefix = f"{prefix}_{int(a)}_{int(b)}"
        names.extend([f"{pair_prefix}_{suffix}" for suffix in suffixes])
    return names


def build_patch_evidence_feature_names(
    *,
    class_count: int,
    pairs: Sequence[Pair],
    spatial_evidence_features: bool = False,
) -> List[str]:
    names: List[str] = []
    for prefix in [
        "patch_mean_logit",
        "patch_max_logit",
        "patch_topk_mean_logit",
        "bbox_mean_logit",
        "bbox_max_logit",
        "bbox_topk_mean_logit",
        "all_top1_fraction",
        "bbox_top1_fraction",
    ]:
        names.extend(_class_vector_feature_names(prefix, class_count))
    names.extend(_pair_margin_feature_names(pairs))
    if bool(spatial_evidence_features):
        for prefix in [
            "interior_mean_logit",
            "interior_max_logit",
            "interior_topk_mean_logit",
            "interior_weighted_mean_logit",
            "interior_top1_fraction",
            "border_mean_logit",
            "border_max_logit",
            "border_topk_mean_logit",
            "border_weighted_mean_logit",
            "border_top1_fraction",
        ]:
            names.extend(_class_vector_feature_names(prefix, class_count))
        names.extend(_pair_margin_feature_names(pairs, prefix="interior_pair"))
        names.extend(_pair_margin_feature_names(pairs, prefix="border_pair"))
    return names


def build_patch_verifier_feature_names(
    *,
    embedding_dim: int,
    class_count: int,
    pairs: Sequence[Pair],
    spatial_evidence_features: bool = False,
    source_domain_feature: bool = False,
) -> List[str]:
    names = build_verifier_feature_names(
        embedding_dim=int(embedding_dim),
        class_count=int(class_count),
    )
    names.extend(
        build_patch_evidence_feature_names(
            class_count=int(class_count),
            pairs=pairs,
            spatial_evidence_features=bool(spatial_evidence_features),
        )
    )
    if bool(source_domain_feature):
        names.append("source_domain")
    return names


def _append_source_domain_feature(
    features: np.ndarray,
    *,
    source_value: float,
    enabled: bool,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if not enabled:
        return features
    source = np.full((features.shape[0], 1), float(source_value), dtype=np.float32)
    return np.concatenate([features, source], axis=1)


def _patch_logits_from_head(model: torch.nn.Module, features: Mapping[str, Tensor]) -> Tensor:
    patches = features.get("patches")
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError("Model features do not contain patch tokens [B, N, D].")
    head = getattr(model, "head", None)
    if head is None:
        raise ValueError("Model does not expose a classification head.")
    batch_size, token_count, dim = patches.shape
    logits = head(patches.reshape(batch_size * token_count, dim))
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("Classification head did not return logits [B*N, C].")
    return logits.reshape(batch_size, token_count, int(logits.size(-1)))


def _tensor_metadata(
    metadata: Mapping[str, object],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    value = metadata.get(key)
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _extract_split_patch_evidence(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
    pairs: Sequence[Pair],
    top_k: int,
    bbox_token_prior_source: str,
    spatial_evidence_features: bool,
    spatial_interior_erode: int,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    embedding_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    patch_feature_batches: List[np.ndarray] = []
    recovered_valid_batches: List[np.ndarray] = []
    recovered_bbox_batches: List[np.ndarray] = []
    method_logit_batches: Dict[str, List[np.ndarray]] = {}
    paths: List[str] = []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    seen_samples = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"patch-evidence-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                metadata = {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batch_size_value = int(images.shape[0])
            batch_paths: List[str] = []
            raw_paths = metadata.get("paths", [])
            if isinstance(raw_paths, Sequence):
                batch_paths = [str(path) for path in raw_paths]
            if (
                dataset_paths
                and (
                    len(batch_paths) != batch_size_value
                    or not any(str(path).strip() for path in batch_paths)
                )
            ):
                batch_paths = dataset_paths[seen_samples : seen_samples + batch_size_value]
            seen_samples += batch_size_value

            bbox = _tensor_metadata(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _tensor_metadata(metadata, "crop_bbox", device=device, dtype=torch.float32)
            image_mask = _tensor_metadata(metadata, "image_mask", device=device, dtype=torch.bool)
            bbox_prior = crop_bbox if str(bbox_token_prior_source) == "crop_bbox" and torch.is_tensor(crop_bbox) else bbox

            with autocast_context(device, amp):
                if not hasattr(model, "forward_features"):
                    raise TypeError("Patch evidence probe requires a TRKH model with forward_features().")
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=bbox_prior,
                )
                if torch.is_tensor(bbox):
                    features["bbox"] = bbox
                elif torch.is_tensor(bbox_prior):
                    features["bbox"] = bbox_prior
                if hasattr(model, "forward_heads"):
                    model_output = model.forward_heads(features)
                else:
                    model_output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(model_output)
                probabilities = logits.float().softmax(dim=1)
                embeddings = extract_head_input_from_features(model, features)
                local_logits = _patch_logits_from_head(model, features)
                patch_features, method_logits, diagnostics = summarize_patch_evidence(
                    local_logits,
                    key_padding_mask=features.get("memory_key_padding_mask"),
                    bbox_prior=features.get("patch_bbox_prior"),
                    pairs=pairs,
                    top_k=int(top_k),
                    spatial_evidence_features=bool(spatial_evidence_features),
                    spatial_interior_erode=int(spatial_interior_erode),
                )

            local_top1 = diagnostics["local_top1"]
            valid_mask = diagnostics["valid_mask"]
            bbox_mask = diagnostics["bbox_mask"]
            target = labels.view(-1, 1)
            recovered_valid = ((local_top1 == target) & valid_mask).any(dim=1)
            recovered_bbox = ((local_top1 == target) & bbox_mask).any(dim=1)

            embedding_batches.append(embeddings.detach().float().cpu().numpy())
            probability_batches.append(probabilities.detach().float().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            prediction_batches.append(probabilities.argmax(dim=1).detach().cpu().numpy())
            patch_feature_batches.append(patch_features.detach().float().cpu().numpy())
            recovered_valid_batches.append(recovered_valid.detach().cpu().numpy().astype(np.bool_))
            recovered_bbox_batches.append(recovered_bbox.detach().cpu().numpy().astype(np.bool_))
            for method, method_logits_tensor in method_logits.items():
                method_logit_batches.setdefault(method, []).append(
                    method_logits_tensor.detach().float().cpu().numpy()
                )
            paths.extend(batch_paths)

    labels_np = np.concatenate(label_batches, axis=0) if label_batches else np.zeros((0,), dtype=np.int64)
    probabilities_np = (
        np.concatenate(probability_batches, axis=0)
        if probability_batches
        else np.zeros((0, 0), dtype=np.float32)
    )
    base_predictions_np = (
        np.concatenate(prediction_batches, axis=0)
        if prediction_batches
        else np.zeros((0,), dtype=np.int64)
    )
    method_logits_np = {
        method: np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        for method, batches in method_logit_batches.items()
    }
    return {
        "embeddings": (
            np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
            if embedding_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "probabilities": probabilities_np.astype(np.float32, copy=False),
        "labels": labels_np.astype(np.int64, copy=False),
        "base_predictions": base_predictions_np.astype(np.int64, copy=False),
        "patch_features": (
            np.concatenate(patch_feature_batches, axis=0).astype(np.float32, copy=False)
            if patch_feature_batches
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "method_logits": method_logits_np,
        "target_recovered_valid": (
            np.concatenate(recovered_valid_batches, axis=0)
            if recovered_valid_batches
            else np.zeros((0,), dtype=np.bool_)
        ),
        "target_recovered_bbox": (
            np.concatenate(recovered_bbox_batches, axis=0)
            if recovered_bbox_batches
            else np.zeros((0,), dtype=np.bool_)
        ),
        "paths": paths,
    }


def _change_summary(
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
) -> Dict[str, object]:
    changed = base_predictions.astype(np.int64) != final_predictions.astype(np.int64)
    before_correct = base_predictions.astype(np.int64) == targets.astype(np.int64)
    after_correct = final_predictions.astype(np.int64) == targets.astype(np.int64)
    return {
        "changed": int(changed.sum()),
        "corrections": int(np.logical_and(changed, np.logical_and(~before_correct, after_correct)).sum()),
        "harms": int(np.logical_and(changed, np.logical_and(before_correct, ~after_correct)).sum()),
        "neutral_changes": int(np.logical_and(changed, before_correct == after_correct).sum()),
    }


def _recovery_summary(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    recovered_valid: np.ndarray,
    recovered_bbox: np.ndarray,
) -> Dict[str, object]:
    labels = labels.astype(np.int64)
    base_predictions = base_predictions.astype(np.int64)
    base_wrong = labels != base_predictions
    return {
        "target_recovered_any_valid_fraction": float(np.asarray(recovered_valid, dtype=bool).mean()) if labels.size else 0.0,
        "target_recovered_bbox_fraction": float(np.asarray(recovered_bbox, dtype=bool).mean()) if labels.size else 0.0,
        "base_wrong_target_recovered_any_valid": int(np.logical_and(base_wrong, recovered_valid).sum()),
        "base_wrong_target_recovered_bbox": int(np.logical_and(base_wrong, recovered_bbox).sum()),
        "base_wrong_total": int(base_wrong.sum()),
    }


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    probabilities: np.ndarray,
    method_predictions: Mapping[str, np.ndarray],
    paths: Sequence[str],
    changes: Sequence[Mapping[str, object]],
    pair_probabilities: Optional[Mapping[Pair, np.ndarray]] = None,
) -> None:
    change_by_index = {int(item["sample_index"]): item for item in changes}
    pair_probabilities = dict(pair_probabilities or {})
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "changed",
        "change_pair",
        "verifier_confidence",
    ]
    fieldnames.extend([f"pred_{method}" for method in method_predictions.keys()])
    fieldnames.extend([f"prob_{index}" for index in range(probabilities.shape[1])])
    for pair in pair_probabilities:
        pair_name = f"{int(pair[0])}_{int(pair[1])}"
        fieldnames.extend(
            [
                f"verifier_{pair_name}_prob_{int(pair[0])}",
                f"verifier_{pair_name}_prob_{int(pair[1])}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(targets):
            change = change_by_index.get(int(index), {})
            row = {
                "split": split,
                "sample_index": int(index),
                "image_path": str(paths[index]) if index < len(paths) else "",
                "target_index": int(target),
                "base_prediction": int(base_predictions[index]),
                "final_prediction": int(final_predictions[index]),
                "changed": int(int(base_predictions[index]) != int(final_predictions[index])),
                "change_pair": str(change.get("pair", "")),
                "verifier_confidence": change.get("verifier_confidence", ""),
            }
            for method, predictions in method_predictions.items():
                row[f"pred_{method}"] = int(predictions[index])
            for class_index in range(probabilities.shape[1]):
                row[f"prob_{class_index}"] = float(probabilities[index, class_index])
            for pair, pair_probs in pair_probabilities.items():
                pair_name = f"{int(pair[0])}_{int(pair[1])}"
                if index < int(pair_probs.shape[0]) and np.isfinite(pair_probs[index]).all():
                    row[f"verifier_{pair_name}_prob_{int(pair[0])}"] = float(pair_probs[index, 0])
                    row[f"verifier_{pair_name}_prob_{int(pair[1])}"] = float(pair_probs[index, 1])
                else:
                    row[f"verifier_{pair_name}_prob_{int(pair[0])}"] = ""
                    row[f"verifier_{pair_name}_prob_{int(pair[1])}"] = ""
            writer.writerow(row)


def _pair_verifier_probabilities_for_split(
    *,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    models: Mapping[Pair, object],
    oof_probabilities: Mapping[Pair, np.ndarray],
) -> Dict[Pair, np.ndarray]:
    """Return per-row pair probabilities, using train OOF probabilities when available."""

    labels = np.asarray(labels, dtype=np.int64)
    result: Dict[Pair, np.ndarray] = {}
    for pair, model in models.items():
        full = np.full((labels.shape[0], 2), np.nan, dtype=np.float32)
        pair_mask, _local = _local_pair_labels_for_export(labels, pair)
        pair_indices = np.flatnonzero(pair_mask)
        if labels.shape[0] > 0:
            full[:] = model.predict_proba(features).astype(np.float32, copy=False)
        if split == "train" and pair in oof_probabilities:
            oof = np.asarray(oof_probabilities[pair], dtype=np.float32)
            if oof.shape == (pair_indices.shape[0], 2):
                full[pair_indices] = oof
                result[pair] = full
                continue
        result[pair] = full
    return result


def _local_pair_labels_for_export(labels: np.ndarray, pair: Pair) -> Tuple[np.ndarray, np.ndarray]:
    left, right = pair
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.logical_or(labels == int(left), labels == int(right))
    local = (labels[mask] == int(right)).astype(np.int64)
    return mask, local


def _write_pair_teacher_csv(
    path: Path,
    *,
    labels: np.ndarray,
    paths: Sequence[str],
    pair_probabilities: Mapping[Pair, np.ndarray],
    num_classes: int,
    split: str,
    smoothing: float = 1e-4,
) -> Dict[str, object]:
    """Write a full-coverage teacher CSV from OOF pair verifier probabilities."""

    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(num_classes)
    smoothing = max(0.0, float(smoothing))
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")

    teacher = np.full((labels.shape[0], num_classes), smoothing, dtype=np.float32)
    for row_index, label in enumerate(labels):
        if 0 <= int(label) < num_classes:
            teacher[row_index, int(label)] = 1.0

    pair_rows = 0
    finite_pair_rows = 0
    for pair, pair_probs in pair_probabilities.items():
        left, right = int(pair[0]), int(pair[1])
        if left < 0 or right < 0 or left >= num_classes or right >= num_classes:
            continue
        pair_probs = np.asarray(pair_probs, dtype=np.float32)
        if pair_probs.shape != (labels.shape[0], 2):
            raise ValueError(
                f"pair probabilities for {left}-{right} must have shape "
                f"({labels.shape[0]}, 2), got {pair_probs.shape}"
            )
        label_mask = np.logical_or(labels == left, labels == right)
        finite = np.isfinite(pair_probs).all(axis=1)
        selected = label_mask & finite
        pair_rows += int(label_mask.sum())
        finite_pair_rows += int(selected.sum())
        if selected.any():
            teacher[selected] = smoothing
            teacher[selected, left] = np.clip(pair_probs[selected, 0], 0.0, 1.0)
            teacher[selected, right] = np.clip(pair_probs[selected, 1], 0.0, 1.0)

    teacher = np.clip(teacher, 0.0, None)
    teacher = teacher / np.clip(teacher.sum(axis=1, keepdims=True), 1e-12, None)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sample_index", "path", "target_index"] + [
        f"prob_{index}" for index in range(num_classes)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "split": split,
                    "sample_index": int(index),
                    "path": str(paths[index]) if index < len(paths) else "",
                    "target_index": int(label),
                    **{
                        f"prob_{class_index}": float(teacher[index, class_index])
                        for class_index in range(num_classes)
                    },
                }
            )
    predictions = teacher.argmax(axis=1)
    return {
        "path": str(path),
        "rows": int(labels.shape[0]),
        "pair_rows": int(pair_rows),
        "finite_pair_rows": int(finite_pair_rows),
        "teacher_label_agreement": float((predictions == labels).mean()) if labels.size else 0.0,
        "mean_confidence": float(teacher.max(axis=1).mean()) if labels.size else 0.0,
        "smoothing": float(smoothing),
    }


def _write_changed_cases(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    paths: Sequence[str],
    changes: Sequence[Mapping[str, object]],
) -> None:
    change_by_index = {int(item["sample_index"]): item for item in changes}
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "final_prediction",
        "change_type",
        "change_pair",
        "verifier_confidence",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, change in sorted(change_by_index.items()):
            before_correct = int(base_predictions[index]) == int(targets[index])
            after_correct = int(final_predictions[index]) == int(targets[index])
            if (not before_correct) and after_correct:
                change_type = "correction"
            elif before_correct and (not after_correct):
                change_type = "harm"
            else:
                change_type = "neutral"
            writer.writerow(
                {
                    "split": split,
                    "sample_index": int(index),
                    "image_path": str(paths[index]) if index < len(paths) else "",
                    "target_index": int(targets[index]),
                    "base_prediction": int(base_predictions[index]),
                    "final_prediction": int(final_predictions[index]),
                    "change_type": change_type,
                    "change_pair": str(change.get("pair", "")),
                    "verifier_confidence": change.get("verifier_confidence", ""),
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(str(args.pairs))
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    requested_splits = list(args.split or ["train", "val"])
    if "train" not in requested_splits:
        raise ValueError("Patch-evidence verifier requires train split for fitting.")

    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    start_time = time.perf_counter()
    for split in requested_splits:
        max_samples = int(args.max_train_samples) if split == "train" else int(args.max_eval_samples)
        dataset, class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=str(split),
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples,
        )
        split_payloads[str(split)] = _extract_split_patch_evidence(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=str(split),
            pairs=pairs,
            top_k=int(args.top_k),
            bbox_token_prior_source=str(args.bbox_token_prior_source),
            spatial_evidence_features=bool(args.spatial_evidence_features),
            spatial_interior_erode=int(args.spatial_interior_erode),
        )

    auxiliary_payloads: List[Dict[str, object]] = []
    for aux_index, aux_data in enumerate(list(args.auxiliary_train_data or []), start=1):
        aux_dataset, aux_class_names = _build_dataset(
            data_yaml=Path(aux_data),
            split="train",
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=int(args.max_train_samples),
        )
        if len(aux_class_names) != len(class_names):
            raise ValueError(
                "Auxiliary train data class count does not match primary data: "
                f"{aux_data} has {len(aux_class_names)} classes, primary has {len(class_names)}"
            )
        auxiliary_payloads.append(
            {
                "data": str(Path(aux_data).resolve()),
                "payload": _extract_split_patch_evidence(
                    model=model,
                    dataset=aux_dataset,
                    device=device,
                    batch_size=int(args.batch_size),
                    workers=int(args.workers),
                    amp=bool(args.amp),
                    split=f"aux_train_{aux_index}",
                    pairs=pairs,
                    top_k=int(args.top_k),
                    bbox_token_prior_source=str(args.bbox_token_prior_source),
                    spatial_evidence_features=bool(args.spatial_evidence_features),
                    spatial_interior_erode=int(args.spatial_interior_erode),
                ),
            }
        )

    train = split_payloads["train"]
    primary_train_features = build_patch_verifier_features(
        np.asarray(train["embeddings"], dtype=np.float32),
        np.asarray(train["probabilities"], dtype=np.float32),
        np.asarray(train["patch_features"], dtype=np.float32),
    )
    feature_names = build_patch_verifier_feature_names(
        embedding_dim=int(np.asarray(train["embeddings"]).shape[1]),
        class_count=int(np.asarray(train["probabilities"]).shape[1]),
        pairs=pairs,
        spatial_evidence_features=bool(args.spatial_evidence_features),
        source_domain_feature=bool(args.source_domain_feature),
    )
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    fit_feature_parts = [
        _append_source_domain_feature(
            primary_train_features,
            source_value=0.0,
            enabled=bool(args.source_domain_feature),
        )
    ]
    fit_label_parts = [train_labels]
    fit_weight_parts = [np.ones(train_labels.shape[0], dtype=np.float32)]
    auxiliary_summaries: List[Dict[str, object]] = []
    aux_weight = max(0.0, float(args.auxiliary_train_weight))
    for aux_index, aux_item in enumerate(auxiliary_payloads, start=1):
        aux_payload = aux_item["payload"]
        if not isinstance(aux_payload, Mapping):
            continue
        aux_labels = np.asarray(aux_payload["labels"], dtype=np.int64)
        aux_features = build_patch_verifier_features(
            np.asarray(aux_payload["embeddings"], dtype=np.float32),
            np.asarray(aux_payload["probabilities"], dtype=np.float32),
            np.asarray(aux_payload["patch_features"], dtype=np.float32),
        )
        if aux_labels.shape[0] > 0 and aux_weight > 0.0:
            fit_feature_parts.append(
                _append_source_domain_feature(
                    aux_features,
                    source_value=float(aux_index),
                    enabled=bool(args.source_domain_feature),
                )
            )
            fit_label_parts.append(aux_labels)
            fit_weight_parts.append(np.full(aux_labels.shape[0], aux_weight, dtype=np.float32))
        auxiliary_summaries.append(
            {
                "data": str(aux_item["data"]),
                "samples": int(aux_labels.shape[0]),
                "weight": float(aux_weight),
                "included": bool(aux_labels.shape[0] > 0 and aux_weight > 0.0),
                "patch_feature_dim": int(np.asarray(aux_payload["patch_features"]).shape[1]),
                "verifier_feature_dim": int(aux_features.shape[1] + (1 if bool(args.source_domain_feature) else 0)),
            }
        )
    train_features = np.concatenate(fit_feature_parts, axis=0).astype(np.float32, copy=False)
    fit_labels = np.concatenate(fit_label_parts, axis=0).astype(np.int64, copy=False)
    fit_weights = np.concatenate(fit_weight_parts, axis=0).astype(np.float32, copy=False)
    use_sample_weights = bool(auxiliary_payloads) and not np.allclose(fit_weights, 1.0)
    models, pair_summaries, oof_probabilities = _fit_pair_models(
        train_features,
        fit_labels,
        pairs,
        model_type=str(args.verifier_model),
        c_value=float(args.logistic_c),
        max_iter=int(args.logistic_max_iter),
        seed=int(args.seed),
        oof_folds=int(args.oof_folds),
        sample_weights=fit_weights if use_sample_weights else None,
        extra_trees_n_estimators=int(args.extra_trees_n_estimators),
        extra_trees_min_samples_leaf=int(args.extra_trees_min_samples_leaf),
        extra_trees_max_depth=int(args.extra_trees_max_depth),
    )
    model_export_summary = write_pair_model_export(
        output_dir / "pair_verifier_model_params.json",
        models,
        feature_dim=int(train_features.shape[1]),
        feature_names=feature_names,
        metadata={
            "tool": "probe_patch_evidence_mil",
            "data": str(Path(args.data).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "pairs": [f"{int(a)}-{int(b)}" for a, b in pairs],
            "verifier_model": str(args.verifier_model),
            "logistic_c": float(args.logistic_c),
            "top_k": int(args.top_k),
            "bbox_token_prior_source": str(args.bbox_token_prior_source),
            "spatial_evidence_features": bool(args.spatial_evidence_features),
            "spatial_interior_erode": int(args.spatial_interior_erode),
            "oof_folds": int(args.oof_folds),
            "auxiliary_train_weight": float(args.auxiliary_train_weight),
            "source_domain_feature": bool(args.source_domain_feature),
        },
    )

    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "class_names": list(class_names),
        "pairs": [f"{a}-{b}" for a, b in pairs],
        "settings": {
            "top_k": int(args.top_k),
            "bbox_token_prior_source": str(args.bbox_token_prior_source),
            "spatial_evidence_features": bool(args.spatial_evidence_features),
            "spatial_interior_erode": int(args.spatial_interior_erode),
            "min_pair_probability": float(args.min_pair_probability),
            "max_pair_margin": float(args.max_pair_margin),
            "verifier_confidence_threshold": float(args.verifier_confidence_threshold),
            "verifier_model": str(args.verifier_model),
            "logistic_c": float(args.logistic_c),
            "extra_trees_n_estimators": int(args.extra_trees_n_estimators),
            "extra_trees_min_samples_leaf": int(args.extra_trees_min_samples_leaf),
            "extra_trees_max_depth": int(args.extra_trees_max_depth),
            "oof_folds": int(args.oof_folds),
            "auxiliary_train_weight": float(args.auxiliary_train_weight),
            "source_domain_feature": bool(args.source_domain_feature),
        },
        "pair_models": pair_summaries,
        "pair_model_export": model_export_summary,
        "auxiliary_train": auxiliary_summaries,
        "splits": {},
        "sources": [
            "https://arxiv.org/html/2606.14555v1",
            "https://cdn.aaai.org/ojs/19967/19967-13-23980-1-2-20220628.pdf",
            "https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0685.pdf",
        ],
        "leakage_guard": (
            "pair verifiers fit on train labels only; val is development diagnostic; "
            "test is not read by default"
        ),
    }

    for split, payload in split_payloads.items():
        probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
        base_predictions = np.asarray(payload["base_predictions"], dtype=np.int64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        raw_patch_features = build_patch_verifier_features(
            np.asarray(payload["embeddings"], dtype=np.float32),
            probabilities,
            np.asarray(payload["patch_features"], dtype=np.float32),
        )
        patch_features = _append_source_domain_feature(
            raw_patch_features,
            source_value=0.0,
            enabled=bool(args.source_domain_feature),
        )
        final_predictions, changes = apply_pairwise_verifiers(
            probabilities,
            base_predictions,
            models,
            min_pair_probability=float(args.min_pair_probability),
            max_pair_margin=float(args.max_pair_margin),
            verifier_confidence_threshold=float(args.verifier_confidence_threshold),
            features=patch_features,
        )
        export_oof_probabilities = {} if auxiliary_payloads else oof_probabilities
        pair_probabilities = _pair_verifier_probabilities_for_split(
            split=str(split),
            features=patch_features,
            labels=labels,
            models=models,
            oof_probabilities=export_oof_probabilities,
        )
        method_predictions = {
            method: logits.argmax(axis=1).astype(np.int64)
            for method, logits in dict(payload.get("method_logits", {})).items()
        }
        metrics: Dict[str, object] = {
            "base": _classification_metrics(labels, base_predictions, class_names),
            "patch_verified": _classification_metrics(labels, final_predictions, class_names),
            "changes": _change_summary(labels, base_predictions, final_predictions),
            "target_recovery": _recovery_summary(
                labels,
                base_predictions,
                np.asarray(payload["target_recovered_valid"], dtype=bool),
                np.asarray(payload["target_recovered_bbox"], dtype=bool),
            ),
            "direct_patch_methods": {
                method: _classification_metrics(labels, predictions, class_names)
                for method, predictions in method_predictions.items()
            },
        }
        split_dir = output_dir / str(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_predictions(
            split_dir / "predictions.csv",
            split=str(split),
            targets=labels,
            base_predictions=base_predictions,
            final_predictions=final_predictions,
            probabilities=probabilities,
            method_predictions=method_predictions,
            paths=list(payload.get("paths", [])),
            changes=changes,
            pair_probabilities=pair_probabilities,
        )
        _write_changed_cases(
            split_dir / "changed_cases_for_xai.csv",
            split=str(split),
            targets=labels,
            base_predictions=base_predictions,
            final_predictions=final_predictions,
            paths=list(payload.get("paths", [])),
            changes=changes,
        )
        teacher_csv_summary = None
        if str(split) == "train":
            if auxiliary_payloads:
                teacher_csv_summary = {
                    "status": "skipped",
                    "reason": (
                        "auxiliary train data was used for fitting; combined OOF "
                        "probabilities do not align one-to-one with primary train rows"
                    ),
                }
            else:
                teacher_csv_summary = _write_pair_teacher_csv(
                    split_dir / "pair_verifier_teacher_oof.csv",
                    labels=labels,
                    paths=list(payload.get("paths", [])),
                    pair_probabilities=pair_probabilities,
                    num_classes=int(probabilities.shape[1]),
                    split=str(split),
                )
        summary["splits"][str(split)] = {
            "samples": int(labels.shape[0]),
            "patch_feature_dim": int(np.asarray(payload["patch_features"]).shape[1]),
            "raw_verifier_feature_dim": int(raw_patch_features.shape[1]),
            "verifier_feature_dim": int(patch_features.shape[1]),
            "metrics": metrics,
            "pair_verifier_teacher_csv": teacher_csv_summary,
        }

    summary["seconds"] = float(time.perf_counter() - start_time)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    compact = {
        "output_dir": str(output_dir),
        "seconds": round(float(summary["seconds"]), 2),
        "val": {},
    }
    val_metrics = summary["splits"].get("val", {}).get("metrics", {})
    if isinstance(val_metrics, Mapping):
        for method_name in ("base", "patch_verified"):
            metrics = val_metrics.get(method_name, {})
            if isinstance(metrics, Mapping):
                per_class = metrics.get("per_class", [])
                class1 = per_class[1].get("f1") if isinstance(per_class, list) and len(per_class) > 1 else None
                compact["val"][method_name] = {
                    "macro_f1": metrics.get("macro_f1"),
                    "class1_f1": class1,
                }
    print(json.dumps(compact, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
