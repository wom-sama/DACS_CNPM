from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import _write_artifact_manifest
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification


SEED = 20260712
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
FOCUS_CLASS_INDEX = 1
INPUT_SIZE = 256
CROP_SIZE = 240
CROP_A = (0, 0, CROP_SIZE)
CROP_B = (16, 16, CROP_SIZE)
NUM_MATCHES = 20
LOCATION_MAX_DISTANCE = 0.05
INTERIOR_PRIOR_THRESHOLD = 0.50
XAI_CASES = 12
OFFICIAL_SOURCE_COMMIT = "803ae4c8cd1649a820f03afb4793763e95317620"
LITERATURE = (
    "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
    "39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html",
    "https://github.com/facebookresearch/VICRegL",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-keeper VICRegL crop-location readiness audit. It reads only "
            "train/validation, writes no model, and never edits raw data."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--extract-batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--num-matches", type=int, default=NUM_MATCHES)
    parser.add_argument(
        "--location-max-distance",
        type=float,
        default=LOCATION_MAX_DISTANCE,
    )
    parser.add_argument(
        "--interior-prior-threshold",
        type=float,
        default=INTERIOR_PRIOR_THRESHOLD,
    )
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--xai-cases", type=int, default=XAI_CASES)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_protocol(args: argparse.Namespace) -> None:
    if int(args.extract_batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("Extraction batch size and workers are invalid")
    if int(args.num_matches) != NUM_MATCHES:
        raise ValueError("This locked audit requires the official gamma=20 setting")
    if not math.isclose(
        float(args.location_max_distance),
        LOCATION_MAX_DISTANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("This locked audit requires location max distance 0.05")
    if not math.isclose(
        float(args.interior_prior_threshold),
        INTERIOR_PRIOR_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("This locked audit requires interior prior threshold 0.50")
    if int(args.focus_class_index) != FOCUS_CLASS_INDEX:
        raise ValueError("This locked audit supports class 1 only")
    if int(args.xai_cases) < 4:
        raise ValueError("At least four XAI cases are required")
    if int(args.max_train_samples) < 0 or int(args.max_val_samples) < 0:
        raise ValueError("Sample limits cannot be negative")
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data).resolve().parent
    if _is_relative_to(output_dir, data_root):
        raise ValueError("Output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    device = torch.device(requested) if requested else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_source_group(value: object) -> str:
    return str(value).strip().replace("/", "\\").casefold()


def transform_bbox_for_crop(
    bbox: Tensor,
    *,
    top: int,
    left: int,
    crop_size: int,
    image_size: int,
) -> Tensor:
    if bbox.ndim != 2 or int(bbox.size(1)) < 4:
        raise ValueError("bbox must have shape [B,4+]")
    values = bbox[:, :4].float()
    scale = float(crop_size) / float(image_size)
    left_norm = float(left) / float(image_size)
    top_norm = float(top) / float(image_size)
    x1 = (values[:, 0] - 0.5 * values[:, 2] - left_norm) / scale
    x2 = (values[:, 0] + 0.5 * values[:, 2] - left_norm) / scale
    y1 = (values[:, 1] - 0.5 * values[:, 3] - top_norm) / scale
    y2 = (values[:, 1] + 0.5 * values[:, 3] - top_norm) / scale
    x1, x2 = x1.clamp(0.0, 1.0), x2.clamp(0.0, 1.0)
    y1, y2 = y1.clamp(0.0, 1.0), y2.clamp(0.0, 1.0)
    return torch.stack(
        (
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            (x2 - x1).clamp_min(1e-4),
            (y2 - y1).clamp_min(1e-4),
        ),
        dim=1,
    )


def build_crop_view(
    images: Tensor,
    image_mask: Optional[Tensor],
    bbox: Tensor,
    crop_bbox: Tensor,
    *,
    top: int,
    left: int,
    crop_size: int = CROP_SIZE,
    image_size: int = INPUT_SIZE,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
    if images.ndim != 4 or tuple(images.shape[-2:]) != (image_size, image_size):
        raise ValueError("Crop-pair audit requires square 256px model inputs")
    cropped = images[..., top : top + crop_size, left : left + crop_size]
    cropped = F.interpolate(
        cropped,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    cropped_mask: Optional[Tensor] = None
    if torch.is_tensor(image_mask):
        mask = image_mask
        squeeze = False
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
            squeeze = True
        if mask.ndim != 4:
            raise ValueError("image_mask must have shape [B,H,W] or [B,1,H,W]")
        cropped_mask = F.interpolate(
            mask.float()[..., top : top + crop_size, left : left + crop_size],
            size=(image_size, image_size),
            mode="nearest",
        ) > 0.5
        if squeeze:
            cropped_mask = cropped_mask[:, 0]
    transformed_bbox = transform_bbox_for_crop(
        bbox,
        top=top,
        left=left,
        crop_size=crop_size,
        image_size=image_size,
    )
    transformed_crop_bbox = transform_bbox_for_crop(
        crop_bbox,
        top=top,
        left=left,
        crop_size=crop_size,
        image_size=image_size,
    )
    return cropped, cropped_mask, transformed_bbox, transformed_crop_bbox


def patch_locations(
    patch_indices: Tensor,
    *,
    grid_size: Tuple[int, int],
    top: int,
    left: int,
    crop_size: int = CROP_SIZE,
    image_size: int = INPUT_SIZE,
) -> Tensor:
    if patch_indices.ndim != 2:
        raise ValueError("patch_indices must have shape [B,N]")
    grid_height, grid_width = int(grid_size[0]), int(grid_size[1])
    rows = torch.div(patch_indices, grid_width, rounding_mode="floor").float()
    columns = torch.remainder(patch_indices, grid_width).float()
    y = (
        float(top) + (rows + 0.5) * float(crop_size) / float(grid_height)
    ) / float(image_size)
    x = (
        float(left) + (columns + 0.5) * float(crop_size) / float(grid_width)
    ) / float(image_size)
    return torch.stack((y, x), dim=-1)


def _directional_match_stats(
    source_features: Tensor,
    source_locations: Tensor,
    source_prior: Tensor,
    target_features: Tensor,
    target_locations: Tensor,
    target_prior: Tensor,
    *,
    num_matches: int,
    location_max_distance: float,
    interior_prior_threshold: float,
) -> Dict[str, Tensor]:
    source = F.normalize(source_features.float(), dim=-1, eps=1e-6)
    target = F.normalize(target_features.float(), dim=-1, eps=1e-6)
    location_distances = torch.cdist(source_locations.float(), target_locations.float())
    nearest_location_distance, nearest_location_index = location_distances.min(dim=2)
    matched_target_prior = target_prior.float().gather(1, nearest_location_index)
    eligible = (
        (source_prior.float() >= float(interior_prior_threshold))
        & (matched_target_prior >= float(interior_prior_threshold))
        & (nearest_location_distance <= float(location_max_distance))
    )
    location_only = nearest_location_distance <= float(location_max_distance)
    sparse = eligible.sum(dim=1) < int(num_matches)
    eligible = torch.where(sparse[:, None], location_only, eligible)

    masked_distance = nearest_location_distance.masked_fill(~eligible, torch.inf)
    selected_distance, selected_source_index = torch.topk(
        masked_distance,
        k=min(int(num_matches), int(masked_distance.size(1))),
        dim=1,
        largest=False,
        sorted=True,
    )
    selected_valid = torch.isfinite(selected_distance)
    selected_location_index = nearest_location_index.gather(1, selected_source_index)

    similarities = torch.bmm(source, target.transpose(1, 2))
    valid_target = target_prior.float() >= float(interior_prior_threshold)
    empty_target = ~valid_target.any(dim=1)
    if bool(empty_target.any().item()):
        valid_target = valid_target.clone()
        valid_target[empty_target] = True
    feature_similarities = similarities.masked_fill(~valid_target[:, None, :], -torch.inf)
    feature_best_cosine, feature_nearest_index = feature_similarities.max(dim=2)
    location_cosine = similarities.gather(2, nearest_location_index.unsqueeze(-1)).squeeze(-1)

    selected_location_cosine = location_cosine.gather(1, selected_source_index)
    selected_feature_cosine = feature_best_cosine.gather(1, selected_source_index)
    selected_feature_index = feature_nearest_index.gather(1, selected_source_index)
    selected_feature_location = target_locations.gather(
        1,
        selected_feature_index.unsqueeze(-1).expand(-1, -1, 2),
    )
    selected_source_location = source_locations.gather(
        1,
        selected_source_index.unsqueeze(-1).expand(-1, -1, 2),
    )
    feature_location_distance = (
        selected_feature_location.float() - selected_source_location.float()
    ).norm(dim=2)
    retrieval_correct = selected_feature_index == selected_location_index

    weights = selected_valid.float()
    counts = weights.sum(dim=1).clamp_min(1.0)

    def weighted_mean(values: Tensor) -> Tensor:
        safe = torch.where(selected_valid, values.float(), torch.zeros_like(values.float()))
        return safe.sum(dim=1) / counts

    return {
        "match_count": selected_valid.sum(dim=1),
        "location_distance": weighted_mean(selected_distance),
        "location_match_cosine": weighted_mean(selected_location_cosine),
        "feature_match_cosine": weighted_mean(selected_feature_cosine),
        "retrieval_top1": weighted_mean(retrieval_correct.float()),
        "feature_match_location_distance": weighted_mean(feature_location_distance),
        "all_location_cosine": location_cosine,
        "all_eligible": eligible,
    }


def batch_crop_match_stats(
    patches_a: Tensor,
    indices_a: Tensor,
    prior_a: Tensor,
    patches_b: Tensor,
    indices_b: Tensor,
    prior_b: Tensor,
    *,
    grid_size: Tuple[int, int],
    crop_a: Tuple[int, int, int] = CROP_A,
    crop_b: Tuple[int, int, int] = CROP_B,
    num_matches: int = NUM_MATCHES,
    location_max_distance: float = LOCATION_MAX_DISTANCE,
    interior_prior_threshold: float = INTERIOR_PRIOR_THRESHOLD,
) -> Dict[str, Tensor]:
    if patches_a.ndim != 3 or patches_b.ndim != 3:
        raise ValueError("patch tensors must have shape [B,N,D]")
    if patches_a.size(0) != patches_b.size(0) or patches_a.size(2) != patches_b.size(2):
        raise ValueError("crop-pair patch tensors do not align")
    if tuple(indices_a.shape) != tuple(patches_a.shape[:2]):
        raise ValueError("indices_a does not align with patches_a")
    if tuple(indices_b.shape) != tuple(patches_b.shape[:2]):
        raise ValueError("indices_b does not align with patches_b")
    if tuple(prior_a.shape) != tuple(indices_a.shape):
        raise ValueError("prior_a does not align with indices_a")
    if tuple(prior_b.shape) != tuple(indices_b.shape):
        raise ValueError("prior_b does not align with indices_b")

    locations_a = patch_locations(
        indices_a,
        grid_size=grid_size,
        top=int(crop_a[0]),
        left=int(crop_a[1]),
        crop_size=int(crop_a[2]),
    )
    locations_b = patch_locations(
        indices_b,
        grid_size=grid_size,
        top=int(crop_b[0]),
        left=int(crop_b[1]),
        crop_size=int(crop_b[2]),
    )
    forward = _directional_match_stats(
        patches_a,
        locations_a,
        prior_a,
        patches_b,
        locations_b,
        prior_b,
        num_matches=int(num_matches),
        location_max_distance=float(location_max_distance),
        interior_prior_threshold=float(interior_prior_threshold),
    )
    reverse = _directional_match_stats(
        patches_b,
        locations_b,
        prior_b,
        patches_a,
        locations_a,
        prior_a,
        num_matches=int(num_matches),
        location_max_distance=float(location_max_distance),
        interior_prior_threshold=float(interior_prior_threshold),
    )
    output: Dict[str, Tensor] = {}
    for key in (
        "match_count",
        "location_distance",
        "location_match_cosine",
        "feature_match_cosine",
        "retrieval_top1",
        "feature_match_location_distance",
    ):
        if key == "match_count":
            output[key] = torch.minimum(forward[key], reverse[key])
        else:
            output[key] = 0.5 * (forward[key].float() + reverse[key].float())

    mismatch = (1.0 - forward["all_location_cosine"].float()).clamp_min(0.0)
    mismatch = mismatch * forward["all_eligible"].float()
    grid_height, grid_width = int(grid_size[0]), int(grid_size[1])
    mismatch_grid = mismatch.new_zeros((mismatch.size(0), grid_height * grid_width))
    mismatch_grid.scatter_(1, indices_a.long(), mismatch)
    output["mismatch_map"] = mismatch_grid.view(-1, grid_height, grid_width)
    output["eligible_fraction"] = forward["all_eligible"].float().mean(dim=1)
    return output


def _forward_features_and_probabilities(
    model: nn.Module,
    images: Tensor,
    image_mask: Optional[Tensor],
    bbox: Tensor,
    crop_bbox: Tensor,
    *,
    device: torch.device,
    amp: bool,
) -> Tuple[Dict[str, Tensor], Tensor]:
    with autocast_context(device, bool(amp)):
        features = model.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=crop_bbox,
        )
        features["bbox"] = bbox
        output = classification_logits_from_features(model, features)
        logits, _, _ = extract_detection_from_model_output(output)
    return features, logits.float().softmax(dim=1)


def _extract_split(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    split: str,
    num_matches: int,
    location_max_distance: float,
    interior_prior_threshold: float,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_classification,
        persistent_workers=bool(int(workers) > 0),
    )
    arrays: Dict[str, List[np.ndarray]] = {
        "labels": [],
        "sample_index": [],
        "base_probabilities": [],
        "crop_a_probabilities": [],
        "crop_b_probabilities": [],
        "crop_average_probabilities": [],
        "match_count": [],
        "location_distance": [],
        "location_match_cosine": [],
        "feature_match_cosine": [],
        "retrieval_top1": [],
        "feature_match_location_distance": [],
        "eligible_fraction": [],
    }
    mismatch_maps: List[np.ndarray] = []
    paths: List[str] = []
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in dataset_paths_fn()]
        if callable(dataset_paths_fn)
        else []
    )
    model.eval()
    seen = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"vicregl-crop-match-{split}", dynamic_ncols=True)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                raise ValueError("VICRegL crop audit requires tensor metadata")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            bbox = metadata.get("bbox")
            crop_bbox = metadata.get("crop_bbox")
            if not torch.is_tensor(crop_bbox):
                raise ValueError("crop_bbox metadata is required")
            crop_bbox = crop_bbox.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            bbox = (
                bbox.to(device=device, dtype=torch.float32, non_blocking=True)
                if torch.is_tensor(bbox)
                else crop_bbox
            )
            image_mask = metadata.get("image_mask")
            image_mask = (
                image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
                if torch.is_tensor(image_mask)
                else None
            )
            crop_a_images, crop_a_mask, crop_a_bbox, crop_a_prior_bbox = build_crop_view(
                images,
                image_mask,
                bbox,
                crop_bbox,
                top=CROP_A[0],
                left=CROP_A[1],
                crop_size=CROP_A[2],
            )
            crop_b_images, crop_b_mask, crop_b_bbox, crop_b_prior_bbox = build_crop_view(
                images,
                image_mask,
                bbox,
                crop_bbox,
                top=CROP_B[0],
                left=CROP_B[1],
                crop_size=CROP_B[2],
            )
            base_features, base_probabilities = _forward_features_and_probabilities(
                model,
                images,
                image_mask,
                bbox,
                crop_bbox,
                device=device,
                amp=bool(amp),
            )
            crop_a_features, crop_a_probabilities = _forward_features_and_probabilities(
                model,
                crop_a_images,
                crop_a_mask,
                crop_a_bbox,
                crop_a_prior_bbox,
                device=device,
                amp=bool(amp),
            )
            crop_b_features, crop_b_probabilities = _forward_features_and_probabilities(
                model,
                crop_b_images,
                crop_b_mask,
                crop_b_bbox,
                crop_b_prior_bbox,
                device=device,
                amp=bool(amp),
            )
            del base_features
            required = ("patches", "patch_indices", "patch_bbox_prior", "grid_size")
            if any(key not in crop_a_features or key not in crop_b_features for key in required):
                raise ValueError("Keeper does not expose spatial patch metadata")
            if tuple(crop_a_features["grid_size"]) != tuple(crop_b_features["grid_size"]):
                raise ValueError("Crop-pair patch grids differ")
            match = batch_crop_match_stats(
                crop_a_features["patches"],
                crop_a_features["patch_indices"],
                crop_a_features["patch_bbox_prior"],
                crop_b_features["patches"],
                crop_b_features["patch_indices"],
                crop_b_features["patch_bbox_prior"],
                grid_size=tuple(crop_a_features["grid_size"]),
                num_matches=int(num_matches),
                location_max_distance=float(location_max_distance),
                interior_prior_threshold=float(interior_prior_threshold),
            )
            crop_average = 0.5 * (crop_a_probabilities + crop_b_probabilities)
            arrays["labels"].append(labels.cpu().numpy())
            arrays["base_probabilities"].append(base_probabilities.cpu().numpy())
            arrays["crop_a_probabilities"].append(crop_a_probabilities.cpu().numpy())
            arrays["crop_b_probabilities"].append(crop_b_probabilities.cpu().numpy())
            arrays["crop_average_probabilities"].append(crop_average.cpu().numpy())
            for key in (
                "match_count",
                "location_distance",
                "location_match_cosine",
                "feature_match_cosine",
                "retrieval_top1",
                "feature_match_location_distance",
                "eligible_fraction",
            ):
                arrays[key].append(match[key].detach().float().cpu().numpy())
            mismatch_maps.append(match["mismatch_map"].detach().half().cpu().numpy())

            batch_count = int(labels.numel())
            raw_paths = metadata.get("paths", [])
            batch_paths = (
                [str(path) for path in raw_paths]
                if isinstance(raw_paths, Sequence)
                else []
            )
            if dataset_paths and (
                len(batch_paths) != batch_count
                or not any(path.strip() for path in batch_paths)
            ):
                batch_paths = dataset_paths[seen : seen + batch_count]
            if len(batch_paths) != batch_count:
                raise ValueError("Extracted path count does not match batch size")
            paths.extend(batch_paths)
            fallback_indices = np.arange(seen, seen + batch_count, dtype=np.int64)
            sample_index = metadata.get("sample_index")
            if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_count:
                arrays["sample_index"].append(
                    sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                )
            else:
                arrays["sample_index"].append(fallback_indices)
            seen += batch_count

    payload: Dict[str, object] = {
        key: np.concatenate(values, axis=0)
        for key, values in arrays.items()
    }
    payload["mismatch_map"] = np.concatenate(mismatch_maps, axis=0)
    payload["paths"] = np.asarray(paths, dtype=object)
    payload["source_stem"] = np.asarray(
        [_normalized_source_group(Path(path).stem) for path in paths],
        dtype=object,
    )
    payload["seconds"] = float(time.perf_counter() - start)
    payload["peak_cuda_memory_mib"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    row_count = len(payload["labels"])
    if len(paths) != row_count or any(
        len(payload[key]) != row_count
        for key in payload
        if isinstance(payload[key], np.ndarray)
    ):
        raise ValueError("Crop-match payload arrays have inconsistent rows")
    return payload


def _category_masks(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    *,
    focus_class_index: int,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(base_probabilities).argmax(axis=1)
    focus = int(focus_class_index)
    return {
        "class1_true_positive": (labels == focus) & (predictions == focus),
        "class1_false_negative": (labels == focus) & (predictions != focus),
        "class1_false_positive": (labels != focus) & (predictions == focus),
        "nonclass1_true_negative": (labels != focus) & (predictions != focus),
    }


def summarize_local_signal(
    payload: Mapping[str, object],
    *,
    focus_class_index: int = FOCUS_CLASS_INDEX,
) -> Dict[str, object]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    base = np.asarray(payload["base_probabilities"], dtype=np.float32)
    masks = _category_masks(labels, base, focus_class_index=int(focus_class_index))
    metric_names = (
        "match_count",
        "location_distance",
        "location_match_cosine",
        "feature_match_cosine",
        "retrieval_top1",
        "feature_match_location_distance",
        "eligible_fraction",
    )

    def metric_summary(mask: np.ndarray) -> Dict[str, object]:
        result: Dict[str, object] = {"rows": int(mask.sum())}
        for name in metric_names:
            values = np.asarray(payload[name], dtype=np.float64)[mask]
            result[f"{name}_mean"] = None if values.size == 0 else float(values.mean())
        return result

    categories = {name: metric_summary(mask) for name, mask in masks.items()}
    mismatch = 1.0 - np.asarray(payload["location_match_cosine"], dtype=np.float64)
    fn = masks["class1_false_negative"]
    fp = masks["class1_false_positive"]
    direction_auc: Optional[float] = None
    if int(fn.sum()) > 0 and int(fp.sum()) > 0:
        direction_labels = np.concatenate(
            (np.ones(int(fn.sum()), dtype=np.int64), np.zeros(int(fp.sum()), dtype=np.int64))
        )
        direction_scores = np.concatenate((mismatch[fn], mismatch[fp]))
        direction_auc = float(roc_auc_score(direction_labels, direction_scores))
    tp_mismatch = mismatch[masks["class1_true_positive"]]
    fn_mismatch = mismatch[fn]
    fn_minus_tp = (
        None
        if tp_mismatch.size == 0 or fn_mismatch.size == 0
        else float(fn_mismatch.mean() - tp_mismatch.mean())
    )
    overall = metric_summary(np.ones(len(labels), dtype=bool))
    overall.update(
        {
            "minimum_match_count": int(
                np.asarray(payload["match_count"], dtype=np.int64).min()
            ),
            "fn_vs_fp_mismatch_auc": direction_auc,
            "fn_minus_tp_mismatch": fn_minus_tp,
        }
    )
    return {"overall": overall, "categories": categories}


def _focus(metrics: Mapping[str, object], focus_class_index: int) -> Mapping[str, object]:
    return metrics["per_class"][int(focus_class_index)]


def assess_vicregl_crop_match_readiness(
    *,
    train_rows: int,
    val_rows: int,
    train_val_source_overlap: int,
    features_finite: bool,
    train_signal: Mapping[str, object],
    val_signal: Mapping[str, object],
    base_val: Mapping[str, object],
    crop_average_val: Mapping[str, object],
    transitions_vs_base: Mapping[str, int],
    prediction_agreement: float,
    test_split_used: bool,
    focus_class_index: int = FOCUS_CLASS_INDEX,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    train_overall = train_signal["overall"]
    val_overall = val_signal["overall"]
    base_focus = _focus(base_val, focus)
    crop_focus = _focus(crop_average_val, focus)
    train_auc = train_overall.get("fn_vs_fp_mismatch_auc")
    val_auc = val_overall.get("fn_vs_fp_mismatch_auc")
    train_deficit = train_overall.get("fn_minus_tp_mismatch")
    val_deficit = val_overall.get("fn_minus_tp_mismatch")
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_val_source_overlap": int(train_val_source_overlap),
        "features_finite": bool(features_finite),
        "minimum_match_count": min(
            int(train_overall["minimum_match_count"]),
            int(val_overall["minimum_match_count"]),
        ),
        "train_location_match_cosine": float(
            train_overall["location_match_cosine_mean"]
        ),
        "val_location_match_cosine": float(val_overall["location_match_cosine_mean"]),
        "train_retrieval_top1": float(train_overall["retrieval_top1_mean"]),
        "val_retrieval_top1": float(val_overall["retrieval_top1_mean"]),
        "prediction_agreement": float(prediction_agreement),
        "base_val_macro_f1": float(base_val["macro_f1"]),
        "crop_average_val_macro_f1": float(crop_average_val["macro_f1"]),
        "val_macro_gain": float(crop_average_val["macro_f1"] - base_val["macro_f1"]),
        "base_val_focus_f1": float(base_focus["f1"]),
        "crop_average_val_focus_f1": float(crop_focus["f1"]),
        "val_focus_gain": float(crop_focus["f1"] - base_focus["f1"]),
        "base_val_focus_recall": float(base_focus["recall"]),
        "crop_average_val_focus_recall": float(crop_focus["recall"]),
        "train_fn_minus_tp_mismatch": (
            None if train_deficit is None else float(train_deficit)
        ),
        "val_fn_minus_tp_mismatch": None if val_deficit is None else float(val_deficit),
        "train_fn_vs_fp_mismatch_auc": None if train_auc is None else float(train_auc),
        "val_fn_vs_fp_mismatch_auc": None if val_auc is None else float(val_auc),
        "direction_auc_gap": (
            None
            if train_auc is None or val_auc is None
            else abs(float(train_auc) - float(val_auc))
        ),
        "transitions_vs_base": dict(transitions_vs_base),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "features_finite": bool(features_finite),
        "minimum_match_count_ge_20": observed["minimum_match_count"] >= NUM_MATCHES,
        "train_retrieval_has_headroom_le_0p85": observed["train_retrieval_top1"] <= 0.85,
        "val_retrieval_has_headroom_le_0p85": observed["val_retrieval_top1"] <= 0.85,
        "train_retrieval_is_meaningful_ge_0p25": observed["train_retrieval_top1"] >= 0.25,
        "val_retrieval_is_meaningful_ge_0p25": observed["val_retrieval_top1"] >= 0.25,
        "train_location_cosine_has_headroom_le_0p975": observed[
            "train_location_match_cosine"
        ]
        <= 0.975,
        "val_location_cosine_has_headroom_le_0p975": observed[
            "val_location_match_cosine"
        ]
        <= 0.975,
        "crop_prediction_agreement_ge_0p95": observed["prediction_agreement"] >= 0.95,
        "crop_average_preserves_macro_within_0p002": observed["val_macro_gain"] >= -0.002,
        "crop_average_preserves_focus_within_0p01": observed["val_focus_gain"] >= -0.01,
        "crop_average_preserves_focus_recall_within_0p01": observed[
            "crop_average_val_focus_recall"
        ]
        >= observed["base_val_focus_recall"] - 0.01,
        "crop_corrections_ge_harms": int(transitions_vs_base["corrections"])
        >= int(transitions_vs_base["harms"]),
        "crop_focus_fp_removed_ge_created": int(
            transitions_vs_base["focus_false_positive_removed"]
        )
        >= int(transitions_vs_base["focus_false_positive_created"]),
        "crop_focus_fn_rescued_ge_tp_broken": int(
            transitions_vs_base["focus_false_negative_rescued"]
        )
        >= int(transitions_vs_base["focus_true_positive_broken"]),
        "train_fn_mismatch_exceeds_tp_by_0p005": train_deficit is not None
        and float(train_deficit) >= 0.005,
        "val_fn_mismatch_exceeds_tp_by_0p005": val_deficit is not None
        and float(val_deficit) >= 0.005,
        "train_fn_vs_fp_auc_ge_0p60": train_auc is not None and float(train_auc) >= 0.60,
        "val_fn_vs_fp_auc_ge_0p60": val_auc is not None and float(val_auc) >= 0.60,
        "direction_auc_gap_le_0p15": observed["direction_auc_gap"] is not None
        and float(observed["direction_auc_gap"]) <= 0.15,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "image_smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def select_xai_cases(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    crop_average_probabilities: np.ndarray,
    *,
    focus_class_index: int = FOCUS_CLASS_INDEX,
    maximum_cases: int = XAI_CASES,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base_probabilities).argmax(axis=1)
    crop = np.asarray(crop_average_probabilities).argmax(axis=1)
    focus = int(focus_class_index)
    categories = (
        ("focus_fn_rescued", (labels == focus) & (base != focus) & (crop == focus)),
        ("focus_tp_broken", (labels == focus) & (base == focus) & (crop != focus)),
        ("focus_fp_removed", (labels != focus) & (base == focus) & (crop != focus)),
        ("focus_fp_created", (labels != focus) & (base != focus) & (crop == focus)),
        ("focus_false_negative", (labels == focus) & (base != focus)),
        ("focus_false_positive", (labels != focus) & (base == focus)),
    )
    selected: List[Dict[str, object]] = []
    used: set[int] = set()
    for category, mask in categories:
        for index in np.flatnonzero(mask)[:2].tolist():
            if index not in used:
                selected.append({"row_index": int(index), "category": category})
                used.add(int(index))
            if len(selected) >= int(maximum_cases):
                return selected
    for index in np.argsort(-np.asarray(base_probabilities)[:, focus]).tolist():
        if int(index) not in used:
            selected.append({"row_index": int(index), "category": "high_focus_reference"})
            used.add(int(index))
        if len(selected) >= int(maximum_cases):
            break
    return selected


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    values = np.asarray(heat, dtype=np.float32)
    maximum = max(float(values.max()), 1e-8)
    normalized = np.clip(values / maximum, 0.0, 1.0)
    color = np.zeros((*normalized.shape, 3), dtype=np.float32)
    color[..., 0] = 255.0 * normalized
    color[..., 1] = 60.0 * normalized
    resampling = getattr(Image, "Resampling", Image)
    color_large = np.asarray(
        Image.fromarray(color.astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    strength = np.asarray(
        Image.fromarray((normalized * 255.0).astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )[..., None] / 255.0
    alpha = 0.60 * strength
    return np.clip(
        image.astype(np.float32) * (1.0 - alpha) + color_large * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)


def _tensor_rgb(image: Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    rgb = (
        (image.float().cpu() * std_tensor + mean_tensor)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return (rgb * 255.0).round().astype(np.uint8)


def _mismatch_stats(maps: np.ndarray) -> Dict[str, object]:
    values = np.asarray(maps, dtype=np.float64)
    if values.size == 0:
        return {"cases": 0}
    height, width = values.shape[-2:]
    yy, xx = np.mgrid[0:height, 0:width]
    center = (
        (yy >= height * 0.25)
        & (yy < height * 0.75)
        & (xx >= width * 0.25)
        & (xx < width * 0.75)
    )
    border = (yy == 0) | (yy == height - 1) | (xx == 0) | (xx == width - 1)
    total = values.sum(axis=(1, 2))
    return {
        "cases": int(values.shape[0]),
        "mismatch_mean": float(values.mean()),
        "center_mass_mean": float(
            np.mean(values[:, center].sum(axis=1) / np.maximum(total, 1e-12))
        ),
        "border_mass_mean": float(
            np.mean(values[:, border].sum(axis=1) / np.maximum(total, 1e-12))
        ),
    }


def _write_xai(
    output_dir: Path,
    *,
    dataset: Dataset,
    cases: Sequence[Mapping[str, object]],
    payload: Mapping[str, object],
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    tile = 160
    label_height = 36
    canvas = Image.new(
        "RGB",
        (tile * 4, len(cases) * (tile + label_height)),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image)
    rows: List[Dict[str, object]] = []
    labels = np.asarray(payload["labels"], dtype=np.int64)
    base = np.asarray(payload["base_probabilities"], dtype=np.float32)
    crop_a = np.asarray(payload["crop_a_probabilities"], dtype=np.float32)
    crop_b = np.asarray(payload["crop_b_probabilities"], dtype=np.float32)
    crop_average = np.asarray(payload["crop_average_probabilities"], dtype=np.float32)
    maps = np.asarray(payload["mismatch_map"], dtype=np.float32)
    selected_maps: List[np.ndarray] = []
    for case_index, case in enumerate(cases):
        row_index = int(case["row_index"])
        item = dataset[row_index]
        image = item[0]
        target = int(item[1])
        if target != int(labels[row_index]):
            raise ValueError("XAI dataset order differs from extracted labels")
        original_rgb = _tensor_rgb(image, mean, std)
        crop_a_tensor = F.interpolate(
            image[None, :, CROP_A[0] : CROP_A[0] + CROP_A[2], CROP_A[1] : CROP_A[1] + CROP_A[2]],
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
        crop_b_tensor = F.interpolate(
            image[None, :, CROP_B[0] : CROP_B[0] + CROP_B[2], CROP_B[1] : CROP_B[1] + CROP_B[2]],
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
        crop_a_rgb = _tensor_rgb(crop_a_tensor, mean, std)
        crop_b_rgb = _tensor_rgb(crop_b_tensor, mean, std)
        mismatch = maps[row_index]
        selected_maps.append(mismatch)
        images = (
            original_rgb,
            crop_a_rgb,
            crop_b_rgb,
            _heat_overlay(crop_a_rgb, mismatch),
        )
        y = case_index * (tile + label_height)
        for column, rgb in enumerate(images):
            canvas.paste(
                Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR),
                (column * tile, y),
            )
        predictions = (
            int(base[row_index].argmax()),
            int(crop_a[row_index].argmax()),
            int(crop_b[row_index].argmax()),
            int(crop_average[row_index].argmax()),
        )
        draw.text(
            (4, y + tile + 2),
            (
                f"{case['category']} y={target} "
                f"base/a/b/avg={predictions[0]}/{predictions[1]}/"
                f"{predictions[2]}/{predictions[3]}"
            ),
            fill=(20, 20, 20),
        )
        rows.append(
            {
                "preview_row": int(case_index),
                "row_index": row_index,
                "sample_index": int(np.asarray(payload["sample_index"])[row_index]),
                "category": str(case["category"]),
                "image_path": str(np.asarray(payload["paths"], dtype=object)[row_index]),
                "target_index": target,
                "base_prediction_index": predictions[0],
                "crop_a_prediction_index": predictions[1],
                "crop_b_prediction_index": predictions[2],
                "crop_average_prediction_index": predictions[3],
                "location_match_cosine": float(
                    np.asarray(payload["location_match_cosine"])[row_index]
                ),
                "retrieval_top1": float(
                    np.asarray(payload["retrieval_top1"])[row_index]
                ),
            }
        )
    canvas.save(output_dir / "xai_vicregl_crop_mismatch.png")
    xai = {
        "method": "location_matched_one_minus_cosine_on_crop_a_patch_grid",
        "columns": ["base", "crop_a", "crop_b", "crop_a_mismatch"],
        "red": "large mismatch for the location-matched patch token",
        "selection_uses_labels_for_audit_only": True,
        "stats": _mismatch_stats(np.asarray(selected_maps, dtype=np.float32)),
        "rows": rows,
    }
    (output_dir / "xai_vicregl_crop_mismatch.json").write_text(
        json.dumps(xai, indent=2),
        encoding="utf-8",
    )
    return xai


def _prediction_rows(split: str, payload: Mapping[str, object]) -> List[Dict[str, object]]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    base = np.asarray(payload["base_probabilities"], dtype=np.float32)
    crop_a = np.asarray(payload["crop_a_probabilities"], dtype=np.float32)
    crop_b = np.asarray(payload["crop_b_probabilities"], dtype=np.float32)
    average = np.asarray(payload["crop_average_probabilities"], dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for index in range(len(labels)):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(np.asarray(payload["sample_index"])[index]),
            "source_stem": str(np.asarray(payload["source_stem"], dtype=object)[index]),
            "image_path": str(np.asarray(payload["paths"], dtype=object)[index]),
            "target_index": int(labels[index]),
            "base_prediction_index": int(base[index].argmax()),
            "crop_a_prediction_index": int(crop_a[index].argmax()),
            "crop_b_prediction_index": int(crop_b[index].argmax()),
            "crop_average_prediction_index": int(average[index].argmax()),
            "base_focus_probability": float(base[index, FOCUS_CLASS_INDEX]),
            "crop_a_focus_probability": float(crop_a[index, FOCUS_CLASS_INDEX]),
            "crop_b_focus_probability": float(crop_b[index, FOCUS_CLASS_INDEX]),
            "crop_average_focus_probability": float(
                average[index, FOCUS_CLASS_INDEX]
            ),
        }
        for key in (
            "match_count",
            "location_distance",
            "location_match_cosine",
            "feature_match_cosine",
            "retrieval_top1",
            "feature_match_location_distance",
            "eligible_fraction",
        ):
            row[key] = float(np.asarray(payload[key])[index])
        rows.append(row)
    return rows


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_protocol(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint))
    if not hasattr(model, "forward_features"):
        raise ValueError("VICRegL audit requires TRKH forward_features")
    device = _resolve_device(str(args.device))
    model.to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)

    datasets: Dict[str, Dataset] = {}
    payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    for split, maximum in (
        ("train", int(args.max_train_samples)),
        ("val", int(args.max_val_samples)),
    ):
        dataset, split_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=maximum,
        )
        if class_names and list(split_names) != class_names:
            raise ValueError("Train/validation class order differs")
        class_names = list(split_names)
        datasets[split] = dataset
        payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.extract_batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=split,
            num_matches=int(args.num_matches),
            location_max_distance=float(args.location_max_distance),
            interior_prior_threshold=float(args.interior_prior_threshold),
        )
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if len(class_names) != 5:
        raise ValueError("This locked audit requires five classes")
    train_groups = np.asarray(payloads["train"]["source_stem"], dtype=object)
    val_groups = np.asarray(payloads["val"]["source_stem"], dtype=object)
    train_val_source_overlap = len(
        set(train_groups.tolist()).intersection(set(val_groups.tolist()))
    )
    if train_val_source_overlap:
        raise ValueError(
            f"Train/validation source groups overlap: {train_val_source_overlap}"
        )

    metrics: Dict[str, Dict[str, object]] = {}
    signals: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        labels = np.asarray(payloads[split]["labels"], dtype=np.int64)
        metrics[split] = {
            role: _classification_metrics(
                labels,
                np.asarray(payloads[split][f"{role}_probabilities"], dtype=np.float32),
            )
            for role in ("base", "crop_a", "crop_b", "crop_average")
        }
        signals[split] = summarize_local_signal(
            payloads[split],
            focus_class_index=int(args.focus_class_index),
        )

    val_labels = np.asarray(payloads["val"]["labels"], dtype=np.int64)
    transitions = _transition_stats(
        val_labels,
        np.asarray(payloads["val"]["base_probabilities"], dtype=np.float32),
        np.asarray(payloads["val"]["crop_average_probabilities"], dtype=np.float32),
        focus_class_index=int(args.focus_class_index),
    )
    base_predictions = np.asarray(payloads["val"]["base_probabilities"]).argmax(axis=1)
    crop_predictions = np.asarray(
        payloads["val"]["crop_average_probabilities"]
    ).argmax(axis=1)
    prediction_agreement = float(np.mean(base_predictions == crop_predictions))
    features_finite = all(
        all(
            np.isfinite(np.asarray(payloads[split][key], dtype=np.float64)).all()
            for key in (
                "base_probabilities",
                "crop_a_probabilities",
                "crop_b_probabilities",
                "crop_average_probabilities",
                "location_match_cosine",
                "retrieval_top1",
                "feature_match_location_distance",
            )
        )
        for split in ("train", "val")
    )
    gate = assess_vicregl_crop_match_readiness(
        train_rows=len(payloads["train"]["labels"]),
        val_rows=len(payloads["val"]["labels"]),
        train_val_source_overlap=train_val_source_overlap,
        features_finite=features_finite,
        train_signal=signals["train"],
        val_signal=signals["val"],
        base_val=metrics["val"]["base"],
        crop_average_val=metrics["val"]["crop_average"],
        transitions_vs_base=transitions,
        prediction_agreement=prediction_agreement,
        test_split_used=False,
        focus_class_index=int(args.focus_class_index),
    )

    cases = select_xai_cases(
        val_labels,
        payloads["val"]["base_probabilities"],
        payloads["val"]["crop_average_probabilities"],
        focus_class_index=int(args.focus_class_index),
        maximum_cases=int(args.xai_cases),
    )
    xai = _write_xai(
        output_dir,
        dataset=datasets["val"],
        cases=cases,
        payload=payloads["val"],
        mean=mean,
        std=std,
    )
    _write_csv(
        output_dir / "train_crop_match_rows.csv",
        _prediction_rows("train", payloads["train"]),
    )
    _write_csv(
        output_dir / "val_crop_match_rows.csv",
        _prediction_rows("val", payloads["val"]),
    )

    full_support = (
        len(payloads["train"]["labels"]) == EXPECTED_TRAIN_ROWS
        and len(payloads["val"]["labels"]) == EXPECTED_VAL_ROWS
    )
    protocol = {
        "method": "vicregl_frozen_keeper_overlap_crop_location_readiness",
        "scope": "Representation-readiness audit before any local VICRegL trainer branch.",
        "literature": list(LITERATURE),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split_usage": {"train": True, "val": True, "test": False},
        "support_mode": "full_decision" if full_support else "prefix_preflight_only",
        "train_rows": int(len(payloads["train"]["labels"])),
        "val_rows": int(len(payloads["val"]["labels"])),
        "train_source_groups": int(np.unique(train_groups).size),
        "val_source_groups": int(np.unique(val_groups).size),
        "train_val_source_overlap": int(train_val_source_overlap),
        "base_input_size": INPUT_SIZE,
        "crop_a_top_left_size": list(CROP_A),
        "crop_b_top_left_size": list(CROP_B),
        "crop_overlap_pixels": 224,
        "location_matching": "symmetric_nearest_original_coordinate",
        "feature_matching": "cosine_nearest_interior_token",
        "num_matches": int(args.num_matches),
        "location_max_distance": float(args.location_max_distance),
        "interior_prior_threshold": float(args.interior_prior_threshold),
        "feature_source": "final_post_pruning_patch_tokens",
        "underlying_keeper_train_predictions_are_in_sample": True,
        "validation_hyperparameter_selection": False,
        "runtime": {
            "device": str(device),
            "extract_batch_size": int(args.extract_batch_size),
            "workers": int(args.workers),
            "amp": bool(args.amp),
            "torch_threads": int(args.torch_threads),
        },
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
        "feature_cache_written": False,
        "trainable_manifest_written": False,
    }
    summary = {
        "protocol": protocol,
        "class_names": class_names,
        "extraction": {
            split: {
                "rows": int(len(payloads[split]["labels"])),
                "source_groups": int(np.unique(payloads[split]["source_stem"]).size),
                "seconds": float(payloads[split]["seconds"]),
                "peak_cuda_memory_mib": float(payloads[split]["peak_cuda_memory_mib"]),
            }
            for split in ("train", "val")
        },
        "metrics": metrics,
        "local_signal": signals,
        "transitions_crop_average_vs_base_val": transitions,
        "crop_average_base_prediction_agreement_val": prediction_agreement,
        "xai": {
            "case_count": len(cases),
            "stats": xai["stats"],
            "preview": "xai_vicregl_crop_mismatch.png",
            "manifest": "xai_vicregl_crop_mismatch.json",
        },
        "gate": gate,
        "seconds": float(time.perf_counter() - start),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    observed = gate["observed"]
    readme = [
        "# VICRegL Crop-Match Readiness",
        "",
        (
            f"- Support mode: `{protocol['support_mode']}`; train/validation "
            f"rows `{protocol['train_rows']}/{protocol['val_rows']}`; test is closed."
        ),
        (
            "- Validation base/crop-average macro F1: "
            f"`{observed['base_val_macro_f1']:.6f}/"
            f"{observed['crop_average_val_macro_f1']:.6f}`."
        ),
        (
            "- Validation base/crop-average class1 F1: "
            f"`{observed['base_val_focus_f1']:.6f}/"
            f"{observed['crop_average_val_focus_f1']:.6f}`."
        ),
        (
            "- Train/validation location cosine: "
            f"`{observed['train_location_match_cosine']:.6f}/"
            f"{observed['val_location_match_cosine']:.6f}`."
        ),
        (
            "- Train/validation feature retrieval top1: "
            f"`{observed['train_retrieval_top1']:.6f}/"
            f"{observed['val_retrieval_top1']:.6f}`."
        ),
        f"- Image-smoke permission: `{str(bool(gate['image_smoke_permission'])).lower()}`.",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`.",
        "",
        (
            "Decision: add a local VICRegL trainer branch only if every locked "
            "check passes. Do not sweep crop geometry, match count, matching "
            "threshold, local weight, projector, or optimizer after rejection."
        ),
        "",
        (
            "This diagnostic writes no feature cache, model, checkpoint, test "
            "result, raw-data edit, or trainable manifest."
        ),
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )
    manifest = _write_artifact_manifest(
        output_dir,
        mode="vicregl_crop_match_readiness_evidence_manifest",
    )
    return {
        "gate": gate,
        "artifact_manifest": {
            key: value for key, value in manifest.items() if key != "files"
        },
        "seconds": summary["seconds"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
