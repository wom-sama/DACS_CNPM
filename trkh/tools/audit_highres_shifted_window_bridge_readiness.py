from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_deepten_stem_texture_readiness import build_interior_grid_mask
from trkh.tools.probe_api_pairwise_interaction_readiness import _write_artifact_manifest
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification
from trkh.tools.probe_interior_second_order_readiness import build_fixed_orthogonal_projection
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _effective_rank,
    _transition_summary,
)


SEED = 20260715
DETAIL_PROJECTION_SEED = 20260716
EXPECTED_ROWS = 9215
EXPECTED_SOURCE_GROUPS = 8064
EXPECTED_CLASS_COUNTS = {0: 1941, 1: 541, 2: 1920, 3: 2520, 4: 2293}
EXPECTED_FOLD_COUNTS = {0: 1843, 1: 1830, 2: 1828, 3: 1851, 4: 1863}
EXPECTED_STEM_SHAPE = (256, 32, 32)
EXPECTED_PATCH_GRID = (16, 16)
LOWRES_PROJECTION_DIM = 32
DETAIL_PROJECTION_DIM = 16
SHARED_DESCRIPTOR_DIM = 197
DETAIL_DESCRIPTOR_DIM = 288
TOTAL_DESCRIPTOR_DIM = 485
ERODE_RATIO = 0.08
RESIDUAL_SCALE = 0.10
L2_WEIGHT = 1e-3
LBFGS_MAX_ITER = 200
LBFGS_HISTORY_SIZE = 20
FOCUS_CLASS = 1
RESTRICTED_FOCUS_SOURCES = (0, 2, 4)

DEFAULT_CHECKPOINT = Path(
    "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
    "bboxprior_120b_2e_20260701/checkpoints/best.pt"
)
DEFAULT_DATA = Path("D:/DataAI/AIEx/newdataset/yolo_f/data.yaml")
DEFAULT_FOLD_CSV = Path(
    "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
)
DEFAULT_FOLD_SUMMARY = Path("runs/audit_cidt_readiness_full_train_20260714/summary.json")
LOCKED_SHA256 = {
    "checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "fold_csv": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "fold_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
}
PRIMARY_SOURCES = {
    "swin_paper": "https://arxiv.org/abs/2103.14030",
    "swin_repository": "https://github.com/microsoft/Swin-Transformer",
    "swin_commit": "f82860bfb5225915aca09c3227159ee9e1df874d",
    "swin_source_sha256": (
        "1e91f71a737c8c8ff0874b65e5a5a4d6b50c65e98250a0a634e75ced12a2ff8f"
    ),
    "conv_stem_paper": "https://arxiv.org/abs/2106.14881",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only diagnostic for source-aligned stride-8 Haar detail "
            "beyond the current TRKH 2x2 patch merge."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fold-csv", type=Path, default=DEFAULT_FOLD_CSV)
    parser.add_argument("--fold-summary", type=Path, default=DEFAULT_FOLD_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_source(value: object) -> str:
    return str(value).strip().replace("/", "\\").casefold()


def _normalized_path(value: object) -> str:
    return str(Path(str(value)).resolve()).replace("/", "\\").casefold()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def load_clean_fold_declaration(path: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "fold",
            "target_index",
            *(f"keeper_prob_{index}" for index in range(5)),
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fold CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            if str(raw["condition"]).strip().casefold() != "clean":
                continue
            probabilities = np.asarray(
                [float(raw[f"keeper_prob_{index}"]) for index in range(5)],
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all() or not math.isclose(
                float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=2e-5
            ):
                raise ValueError("Fold CSV contains invalid keeper probabilities")
            rows.append(
                {
                    "sample_index": int(raw["sample_index"]),
                    "source_stem": _normalized_source(raw["source_stem"]),
                    "image_path": _normalized_path(raw["image_path"]),
                    "fold": int(raw["fold"]),
                    "label": int(raw["target_index"]),
                    "probabilities": probabilities.astype(np.float32),
                }
            )
    rows.sort(key=lambda row: int(row["sample_index"]))
    indices = np.asarray([row["sample_index"] for row in rows], dtype=np.int64)
    if indices.size != EXPECTED_ROWS or not np.array_equal(
        indices, np.arange(EXPECTED_ROWS, dtype=np.int64)
    ):
        raise ValueError("Clean fold rows must be exactly sample_index 0..9214")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["source_stem"] for row in rows], dtype=object)
    paths = np.asarray([row["image_path"] for row in rows], dtype=object)
    probabilities = np.stack([row["probabilities"] for row in rows]).astype(np.float32)
    class_counts = {index: int((labels == index).sum()) for index in range(5)}
    fold_counts = {index: int((folds == index).sum()) for index in range(5)}
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Unexpected clean class counts: {class_counts}")
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Unexpected clean fold counts: {fold_counts}")
    source_groups = int(np.unique(groups).size)
    if source_groups != EXPECTED_SOURCE_GROUPS:
        raise ValueError(f"Unexpected source-group count: {source_groups}")
    maximum_overlap = 0
    for fold_index in range(5):
        holdout = folds == fold_index
        overlap = len(set(groups[holdout]).intersection(set(groups[~holdout])))
        maximum_overlap = max(maximum_overlap, overlap)
    if maximum_overlap:
        raise ValueError(f"Fold declaration leaks {maximum_overlap} source groups")
    return {
        "sample_index": indices,
        "labels": labels,
        "folds": folds,
        "groups": groups,
        "paths": paths,
        "probabilities": probabilities,
        "class_counts": class_counts,
        "fold_counts": fold_counts,
        "source_groups": source_groups,
        "maximum_source_overlap": maximum_overlap,
    }


def verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(),
        "data": Path(args.data).resolve(),
        "fold_csv": Path(args.fold_csv).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
    }
    hashes: Dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Locked {key} file is missing: {path}")
        observed = _sha256(path)
        if observed != LOCKED_SHA256[key]:
            raise ValueError(
                f"Locked {key} SHA mismatch: observed={observed}, "
                f"expected={LOCKED_SHA256[key]}"
            )
        hashes[key] = observed
    declaration = load_clean_fold_declaration(paths["fold_csv"])
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Keeper checkpoint is not a mapping")
    model = build_model_from_checkpoint(dict(checkpoint))
    downsample = int(getattr(model.stem, "downsample_factor", -1))
    stem_shape = (int(getattr(model.stem, "out_channels", -1)), 256 // downsample, 256 // downsample)
    patch_grid = tuple(int(value) for value in model.patch_embed.base_grid_size)
    if stem_shape != EXPECTED_STEM_SHAPE:
        raise ValueError(f"Unexpected keeper stem shape contract: {stem_shape}")
    if patch_grid != EXPECTED_PATCH_GRID or int(model.patch_embed.patch_size) != 2:
        raise ValueError(f"Unexpected keeper patch merge: grid={patch_grid}")
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": hashes,
        "rows": int(len(declaration["labels"])),
        "source_groups": declaration["source_groups"],
        "class_counts": declaration["class_counts"],
        "fold_counts": declaration["fold_counts"],
        "maximum_source_overlap": declaration["maximum_source_overlap"],
        "stem_shape": list(stem_shape),
        "patch_grid": list(patch_grid),
        "patch_merge_size": 2,
    }


def haar_bands_from_projected_grid(grid: Tensor) -> Tensor:
    if grid.ndim != 4:
        raise ValueError("Projected stem grid must have shape [B,H,W,D]")
    batch, height, width, channels = [int(value) for value in grid.shape]
    if height % 2 or width % 2:
        raise ValueError("Projected stem grid dimensions must be even")
    cells = grid.contiguous().view(batch, height // 2, 2, width // 2, 2, channels)
    z00, z01 = cells[:, :, 0, :, 0], cells[:, :, 0, :, 1]
    z10, z11 = cells[:, :, 1, :, 0], cells[:, :, 1, :, 1]
    horizontal = (z00 - z01 + z10 - z11) * 0.5
    vertical = (z00 + z01 - z10 - z11) * 0.5
    diagonal = (z00 - z01 - z10 + z11) * 0.5
    return torch.cat((horizontal, vertical, diagonal), dim=-1)


def projected_lowres_map(patch_tokens: Tensor, projection: Tensor) -> Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError("Patch tokens must have shape [B,N,C]")
    side = math.isqrt(int(patch_tokens.size(1)))
    if side * side != int(patch_tokens.size(1)):
        raise ValueError("Patch token count must form a square grid")
    normalized = F.layer_norm(patch_tokens.float(), (int(patch_tokens.size(-1)),))
    projected = normalized @ projection.to(patch_tokens.device, torch.float32)
    return projected.view(int(projected.size(0)), side, side, int(projected.size(-1)))


def projected_haar_detail_map(stem: Tensor, projection: Tensor) -> Tensor:
    if stem.ndim != 4:
        raise ValueError("Stem activation must have shape [B,C,H,W]")
    local = stem.float().permute(0, 2, 3, 1).contiguous()
    local = F.layer_norm(local, (int(local.size(-1)),))
    return haar_bands_from_projected_grid(
        local @ projection.to(stem.device, torch.float32)
    )


def pool_spatial_descriptor(feature_map: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    if feature_map.ndim != 4:
        raise ValueError("Feature map must have shape [B,H,W,D]")
    batch, height, width, channels = [int(value) for value in feature_map.shape]
    flat = feature_map.view(batch, height * width, channels)
    valid = mask.to(feature_map.device, torch.bool)
    if tuple(valid.shape) != (batch, height * width):
        raise ValueError("Spatial mask does not align with feature map")
    rows = torch.arange(height, device=feature_map.device).view(height, 1)
    cols = torch.arange(width, device=feature_map.device).view(1, width)
    quadrants = (
        (rows < height // 2) & (cols < width // 2),
        (rows < height // 2) & (cols >= width // 2),
        (rows >= height // 2) & (cols < width // 2),
        (rows >= height // 2) & (cols >= width // 2),
    )

    def masked_mean(active: Tensor) -> Tuple[Tensor, Tensor]:
        weights = active.to(flat.dtype).unsqueeze(-1)
        count = weights.sum(dim=1)
        mean = (flat * weights).sum(dim=1) / count.clamp_min(1.0)
        return mean, count[:, 0]

    full_mean, full_count = masked_mean(valid)
    full_weights = valid.to(flat.dtype).unsqueeze(-1)
    variance = ((flat - full_mean.unsqueeze(1)).square() * full_weights).sum(dim=1)
    variance = variance / full_count.clamp_min(1.0).unsqueeze(-1)
    pooled = [full_mean, variance.clamp_min(1e-8).sqrt()]
    coverages = [full_count / float(height * width)]
    for quadrant in quadrants:
        active = valid & quadrant.flatten().view(1, height * width)
        mean, count = masked_mean(active)
        pooled.append(mean)
        coverages.append(count / float(int(quadrant.sum())))
    return torch.cat(pooled, dim=1), torch.stack(coverages, dim=1)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(str(value).strip() or "cuda")
    if device.type != "cuda":
        raise ValueError("The locked full A0 extraction requires CUDA FP32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def extract_train_descriptors(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        collate_fn=_collate_classification,
        persistent_workers=bool(int(workers) > 0),
    )
    low_projection = build_fixed_orthogonal_projection(
        256, LOWRES_PROJECTION_DIM, seed=SEED
    ).to(device)
    detail_projection = build_fixed_orthogonal_projection(
        256, DETAIL_PROJECTION_DIM, seed=DETAIL_PROJECTION_SEED
    ).to(device)
    shared_batches: List[np.ndarray] = []
    detail_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    paths: List[str] = []
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in dataset_paths_fn()] if callable(dataset_paths_fn) else []
    captured: List[Tensor] = []

    def capture_stem(_module, _inputs, output) -> None:
        captured.append(output)

    hook = model.stem.register_forward_hook(capture_stem)
    model.eval()
    seen = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for images, labels, metadata in tqdm(
                loader, desc="highres-bridge-a0-train", dynamic_ncols=True
            ):
                if not isinstance(metadata, Mapping):
                    raise ValueError("A0 extraction requires tensor metadata")
                images = images.to(device, torch.float32, non_blocking=True)
                labels = labels.to(device, torch.long, non_blocking=True)
                crop_bbox = metadata.get("crop_bbox")
                if not torch.is_tensor(crop_bbox):
                    raise ValueError("crop_bbox metadata is required")
                crop_bbox = crop_bbox.to(device, torch.float32, non_blocking=True)
                bbox = metadata.get("bbox")
                bbox = (
                    bbox.to(device, torch.float32, non_blocking=True)
                    if torch.is_tensor(bbox)
                    else crop_bbox
                )
                image_mask = metadata.get("image_mask")
                image_mask = (
                    image_mask.to(device, torch.bool, non_blocking=True)
                    if torch.is_tensor(image_mask)
                    else None
                )
                captured.clear()
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=crop_bbox,
                )
                features["bbox"] = bbox
                output = (
                    model.forward_heads(features)
                    if hasattr(model, "forward_heads")
                    else classification_logits_from_features(model, features)
                )
                logits, _, _ = extract_detection_from_model_output(output)
                if len(captured) != 1 or not torch.is_tensor(captured[0]):
                    raise RuntimeError("Expected exactly one CNN stem activation")
                stem = captured[0]
                if tuple(int(value) for value in stem.shape[1:]) != EXPECTED_STEM_SHAPE:
                    raise ValueError(f"Unexpected stem activation shape: {tuple(stem.shape)}")
                low_map = projected_lowres_map(model.patch_embed(stem), low_projection)
                detail_map = projected_haar_detail_map(stem, detail_projection)
                if tuple(low_map.shape[1:3]) != EXPECTED_PATCH_GRID:
                    raise ValueError(f"Unexpected low-resolution grid: {tuple(low_map.shape)}")
                if tuple(detail_map.shape[1:3]) != EXPECTED_PATCH_GRID:
                    raise ValueError(f"Unexpected detail grid: {tuple(detail_map.shape)}")
                spatial_mask = build_interior_grid_mask(
                    crop_bbox,
                    image_mask,
                    grid_size=EXPECTED_PATCH_GRID[0],
                    erode_ratio=ERODE_RATIO,
                )
                low_pooled, coverage = pool_spatial_descriptor(low_map, spatial_mask)
                detail_pooled, detail_coverage = pool_spatial_descriptor(
                    detail_map, spatial_mask
                )
                torch.testing.assert_close(coverage, detail_coverage)
                shared = torch.cat((low_pooled, coverage), dim=1)
                if int(shared.size(1)) != SHARED_DESCRIPTOR_DIM:
                    raise RuntimeError(f"Shared descriptor dim mismatch: {shared.size(1)}")
                if int(detail_pooled.size(1)) != DETAIL_DESCRIPTOR_DIM:
                    raise RuntimeError(f"Detail descriptor dim mismatch: {detail_pooled.size(1)}")
                shared_batches.append(shared.cpu().numpy().astype(np.float32, copy=False))
                detail_batches.append(detail_pooled.cpu().numpy().astype(np.float32, copy=False))
                probability_batches.append(
                    logits.float().softmax(dim=1).cpu().numpy().astype(np.float32, copy=False)
                )
                label_batches.append(labels.cpu().numpy().astype(np.int64, copy=False))

                count = int(labels.numel())
                raw_paths = metadata.get("paths", [])
                batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
                if dataset_paths and (
                    len(batch_paths) != count or not any(path.strip() for path in batch_paths)
                ):
                    batch_paths = dataset_paths[seen : seen + count]
                if len(batch_paths) != count:
                    raise ValueError("Extracted path count does not match batch size")
                paths.extend(batch_paths)
                sample_index = metadata.get("sample_index")
                if torch.is_tensor(sample_index) and int(sample_index.numel()) == count:
                    sample_index_batches.append(
                        sample_index.cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                    )
                else:
                    sample_index_batches.append(np.arange(seen, seen + count, dtype=np.int64))
                seen += count
    finally:
        hook.remove()
    payload = {
        "shared": np.concatenate(shared_batches).astype(np.float32, copy=False),
        "detail": np.concatenate(detail_batches).astype(np.float32, copy=False),
        "labels": np.concatenate(label_batches).astype(np.int64, copy=False),
        "sample_index": np.concatenate(sample_index_batches).astype(np.int64, copy=False),
        "probabilities": np.concatenate(probability_batches).astype(np.float32, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "seconds": float(time.perf_counter() - start),
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated(device) / (1024.0**3)),
    }
    rows = int(payload["labels"].size)
    if not all(
        rows == int(np.asarray(payload[key]).shape[0])
        for key in ("shared", "detail", "sample_index", "probabilities", "paths")
    ):
        raise ValueError("Extracted A0 arrays have inconsistent row counts")
    return payload


def align_extraction(
    payload: Mapping[str, object], declaration: Mapping[str, object]
) -> Dict[str, object]:
    order = np.argsort(np.asarray(payload["sample_index"], dtype=np.int64))
    aligned = {
        key: np.asarray(payload[key])[order]
        for key in ("shared", "detail", "labels", "sample_index", "probabilities", "paths")
    }
    index_match = bool(
        np.array_equal(aligned["sample_index"], np.asarray(declaration["sample_index"]))
    )
    label_match = bool(np.array_equal(aligned["labels"], np.asarray(declaration["labels"])))
    normalized_paths = np.asarray([_normalized_path(value) for value in aligned["paths"]], dtype=object)
    path_match = bool(np.array_equal(normalized_paths, np.asarray(declaration["paths"])))
    sources = np.asarray(
        [_normalized_source(Path(str(value)).stem) for value in aligned["paths"]], dtype=object
    )
    source_match = bool(np.array_equal(sources, np.asarray(declaration["groups"])))
    recomputed = np.asarray(aligned["probabilities"], dtype=np.float32)
    locked = np.asarray(declaration["probabilities"], dtype=np.float32)
    alignment = {
        "sample_index_match": index_match,
        "label_match": label_match,
        "path_match": path_match,
        "source_match": source_match,
        "keeper_probability_max_abs_error": float(np.max(np.abs(recomputed - locked))),
        "keeper_prediction_mismatches": int(
            np.count_nonzero(recomputed.argmax(axis=1) != locked.argmax(axis=1))
        ),
    }
    if not all((index_match, label_match, path_match, source_match)):
        raise ValueError(f"Extraction/fold alignment failed: {alignment}")
    aligned["paths"] = normalized_paths
    aligned["alignment"] = alignment
    return aligned


def source_derangement(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    max_attempts: int = 4096,
) -> np.ndarray:
    subset = np.asarray(indices, dtype=np.int64)
    all_groups = np.asarray(groups, dtype=object)
    if subset.size < 2 or np.unique(all_groups[subset]).size < 2:
        raise ValueError("A source derangement requires at least two source groups")
    generator = np.random.default_rng(int(seed))
    for _ in range(int(max_attempts)):
        permuted = generator.permutation(subset)
        if not np.any(all_groups[permuted] == all_groups[subset]):
            return permuted.astype(np.int64, copy=False)
    raise RuntimeError("Could not construct a zero-same-source detail permutation")


def standardize_fit_holdout(
    fit_features: np.ndarray, holdout_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    fit = np.asarray(fit_features, dtype=np.float64)
    holdout = np.asarray(holdout_features, dtype=np.float64)
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    constant = scale <= 1e-12
    scale[constant] = 1.0
    fit = (fit - mean) / scale
    holdout = (holdout - mean) / scale
    return fit, holdout, {
        "constant_columns": int(constant.sum()),
        "finite": bool(np.isfinite(fit).all() and np.isfinite(holdout).all()),
    }


def fit_residual_readout(
    *,
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    fit_base_probabilities: np.ndarray,
    holdout_features: np.ndarray,
    holdout_base_probabilities: np.ndarray,
    class_count: int,
    max_iter: int = LBFGS_MAX_ITER,
) -> Dict[str, object]:
    fit_x = torch.from_numpy(np.asarray(fit_features, dtype=np.float64))
    holdout_x = torch.from_numpy(np.asarray(holdout_features, dtype=np.float64))
    fit_y = torch.from_numpy(np.asarray(fit_labels, dtype=np.int64))
    fit_offset = torch.from_numpy(
        np.log(np.asarray(fit_base_probabilities, dtype=np.float64).clip(1e-8, 1.0))
    )
    holdout_offset = torch.from_numpy(
        np.log(np.asarray(holdout_base_probabilities, dtype=np.float64).clip(1e-8, 1.0))
    )
    feature_dim = int(fit_x.size(1))
    weight = torch.zeros((feature_dim, class_count), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(class_count, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=int(max_iter),
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def objective() -> Tensor:
        logits = fit_offset + RESIDUAL_SCALE * (fit_x @ weight + bias)
        return F.cross_entropy(logits, fit_y) + L2_WEIGHT * weight.square().mean()

    with torch.no_grad():
        initial = float(objective())

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = float(objective())
        residual = holdout_x @ weight + bias
        probabilities = (holdout_offset + RESIDUAL_SCALE * residual).softmax(dim=1)
        probabilities = probabilities.numpy().astype(np.float32, copy=False)
        weight_norm = float(weight.norm())
        residual_abs_mean = float(residual.abs().mean())
    state = optimizer.state.get(weight, {})
    finite = bool(
        np.isfinite(probabilities).all()
        and all(math.isfinite(value) for value in (initial, final, weight_norm, residual_abs_mean))
    )
    return {
        "probabilities": probabilities,
        "parameter_count": int(weight.numel() + bias.numel()),
        "iterations": int(state.get("n_iter", 0)),
        "function_evaluations": int(state.get("func_evals", 0)),
        "initial_objective": initial,
        "final_objective": final,
        "weight_norm": weight_norm,
        "residual_abs_mean": residual_abs_mean,
        "finite": finite,
    }


def fit_oof_readouts(
    *,
    shared: np.ndarray,
    detail: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    base_probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    row_count, class_count = int(labels.size), len(class_names)
    oof = {
        role: np.zeros((row_count, class_count), dtype=np.float32)
        for role in ("control", "candidate", "placebo")
    }
    placebo_detail_oof = np.zeros_like(detail, dtype=np.float32)
    fold_rows: List[Dict[str, object]] = []
    optimizer_rows: List[Dict[str, object]] = []
    permutation_rows: List[Dict[str, object]] = []
    parameter_count = TOTAL_DESCRIPTOR_DIM * class_count + class_count
    for fold_index in range(5):
        fit_indices = np.flatnonzero(folds != fold_index).astype(np.int64)
        holdout_indices = np.flatnonzero(folds == fold_index).astype(np.int64)
        overlap = len(set(groups[fit_indices]).intersection(set(groups[holdout_indices])))
        fit_permutation = source_derangement(
            fit_indices, groups, seed=SEED + fold_index * 100 + 1
        )
        holdout_permutation = source_derangement(
            holdout_indices, groups, seed=SEED + fold_index * 100 + 2
        )
        placebo_detail_oof[holdout_indices] = detail[holdout_permutation]
        permutation_rows.append(
            {
                "fold": fold_index,
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "source_overlap": int(overlap),
                "fit_same_source_fixed_points": int(
                    np.count_nonzero(groups[fit_indices] == groups[fit_permutation])
                ),
                "holdout_same_source_fixed_points": int(
                    np.count_nonzero(groups[holdout_indices] == groups[holdout_permutation])
                ),
                "fit_permutation_stays_in_fit": bool(
                    set(fit_permutation.tolist()) == set(fit_indices.tolist())
                ),
                "holdout_permutation_stays_in_holdout": bool(
                    set(holdout_permutation.tolist()) == set(holdout_indices.tolist())
                ),
            }
        )
        role_details = {
            "control": (
                np.zeros((fit_indices.size, DETAIL_DESCRIPTOR_DIM), dtype=np.float32),
                np.zeros((holdout_indices.size, DETAIL_DESCRIPTOR_DIM), dtype=np.float32),
            ),
            "candidate": (detail[fit_indices], detail[holdout_indices]),
            "placebo": (detail[fit_permutation], detail[holdout_permutation]),
        }
        role_metrics: Dict[str, Mapping[str, object]] = {}
        for role, (fit_detail, holdout_detail) in role_details.items():
            fit_features = np.concatenate((shared[fit_indices], fit_detail), axis=1)
            holdout_features = np.concatenate(
                (shared[holdout_indices], holdout_detail), axis=1
            )
            if fit_features.shape[1] != TOTAL_DESCRIPTOR_DIM:
                raise RuntimeError("Readout descriptor width is not protocol exact")
            standardized_fit, standardized_holdout, standardization = standardize_fit_holdout(
                fit_features, holdout_features
            )
            result = fit_residual_readout(
                fit_features=standardized_fit,
                fit_labels=labels[fit_indices],
                fit_base_probabilities=base_probabilities[fit_indices],
                holdout_features=standardized_holdout,
                holdout_base_probabilities=base_probabilities[holdout_indices],
                class_count=class_count,
            )
            if int(result["parameter_count"]) != parameter_count:
                raise RuntimeError("Residual readout parameter count mismatch")
            oof[role][holdout_indices] = result["probabilities"]
            role_metrics[role] = _classification_metrics(
                labels[holdout_indices], result["probabilities"], class_names=class_names
            )
            optimizer_rows.append(
                {
                    "fold": fold_index,
                    "role": role,
                    **standardization,
                    **{key: value for key, value in result.items() if key != "probabilities"},
                }
            )
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "source_overlap": int(overlap),
                "control_macro_f1": float(role_metrics["control"]["macro_f1"]),
                "candidate_macro_f1": float(role_metrics["candidate"]["macro_f1"]),
                "placebo_macro_f1": float(role_metrics["placebo"]["macro_f1"]),
                "candidate_control_macro_delta": float(
                    role_metrics["candidate"]["macro_f1"]
                    - role_metrics["control"]["macro_f1"]
                ),
                "control_class1_f1": float(role_metrics["control"]["focus_f1"]),
                "candidate_class1_f1": float(role_metrics["candidate"]["focus_f1"]),
                "placebo_class1_f1": float(role_metrics["placebo"]["focus_f1"]),
                "candidate_control_class1_f1_delta": float(
                    role_metrics["candidate"]["focus_f1"]
                    - role_metrics["control"]["focus_f1"]
                ),
                "control_class1_precision": float(role_metrics["control"]["focus_precision"]),
                "candidate_class1_precision": float(
                    role_metrics["candidate"]["focus_precision"]
                ),
                "placebo_class1_precision": float(role_metrics["placebo"]["focus_precision"]),
                "candidate_control_class1_precision_delta": float(
                    role_metrics["candidate"]["focus_precision"]
                    - role_metrics["control"]["focus_precision"]
                ),
            }
        )
    return {
        "probabilities": oof,
        "metrics": {
            role: _classification_metrics(labels, values, class_names=class_names)
            for role, values in oof.items()
        },
        "fold_rows": fold_rows,
        "optimizer_rows": optimizer_rows,
        "permutation_rows": permutation_rows,
        "placebo_detail_oof": placebo_detail_oof,
        "parameter_count": parameter_count,
    }


def _restricted_focus_fp(labels: np.ndarray, probabilities: np.ndarray) -> int:
    predictions = np.asarray(probabilities).argmax(axis=1)
    restricted = np.isin(labels, np.asarray(RESTRICTED_FOCUS_SOURCES))
    return int(np.count_nonzero(restricted & (predictions == FOCUS_CLASS)))


def assess_highres_bridge_readiness(
    *,
    structural_checks: Mapping[str, bool],
    keeper_metrics: Mapping[str, object],
    metrics: Mapping[str, Mapping[str, object]],
    candidate_control_transitions: Mapping[str, int],
    control_restricted_fp: int,
    candidate_restricted_fp: int,
    fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    control, candidate, placebo = (
        metrics["control"],
        metrics["candidate"],
        metrics["placebo"],
    )
    nonfocus_drops = [
        float(control["per_class"][index]["f1"])
        - float(candidate["per_class"][index]["f1"])
        for index in range(len(control["per_class"]))
        if index != FOCUS_CLASS
    ]
    observed = {
        "candidate_control_macro_delta": float(candidate["macro_f1"])
        - float(control["macro_f1"]),
        "candidate_control_class1_f1_delta": float(candidate["focus_f1"])
        - float(control["focus_f1"]),
        "candidate_control_class1_precision_delta": float(candidate["focus_precision"])
        - float(control["focus_precision"]),
        "candidate_control_class1_recall_delta": float(candidate["focus_recall"])
        - float(control["focus_recall"]),
        "candidate_placebo_macro_delta": float(candidate["macro_f1"])
        - float(placebo["macro_f1"]),
        "candidate_placebo_class1_f1_delta": float(candidate["focus_f1"])
        - float(placebo["focus_f1"]),
        "candidate_placebo_class1_precision_delta": float(candidate["focus_precision"])
        - float(placebo["focus_precision"]),
        "control_keeper_macro_delta": float(control["macro_f1"])
        - float(keeper_metrics["macro_f1"]),
        "control_keeper_class1_f1_delta": float(control["focus_f1"])
        - float(keeper_metrics["focus_f1"]),
        "restricted_focus_fp_reduction": int(control_restricted_fp)
        - int(candidate_restricted_fp),
        "maximum_nonfocus_f1_drop": max(nonfocus_drops, default=0.0),
        "folds_nonnegative_class1_f1": sum(
            float(row["candidate_control_class1_f1_delta"]) >= 0.0 for row in fold_rows
        ),
        "folds_nonnegative_class1_precision": sum(
            float(row["candidate_control_class1_precision_delta"]) >= 0.0
            for row in fold_rows
        ),
    }
    checks: Dict[str, bool] = {
        **{f"structural_{key}": bool(value) for key, value in structural_checks.items()},
        "macro_gain_gte_0p002": observed["candidate_control_macro_delta"] >= 0.002,
        "class1_f1_gain_gte_0p005": observed["candidate_control_class1_f1_delta"] >= 0.005,
        "class1_precision_gain_gte_0p005": (
            observed["candidate_control_class1_precision_delta"] >= 0.005
        ),
        "class1_recall_delta_gte_minus_0p005": (
            observed["candidate_control_class1_recall_delta"] >= -0.005
        ),
        "restricted_focus_fp_reduction_gte_2": observed["restricted_focus_fp_reduction"] >= 2,
        "corrections_gt_harms": int(candidate_control_transitions["corrections"])
        > int(candidate_control_transitions["harms"]),
        "fn_rescues_ge_tp_breaks": int(candidate_control_transitions["class1_fn_rescued"])
        >= int(candidate_control_transitions["class1_tp_broken"]),
        "maximum_nonfocus_f1_drop_lte_0p010": observed["maximum_nonfocus_f1_drop"] <= 0.010,
        "four_of_five_fold_class1_f1_nonnegative": (
            observed["folds_nonnegative_class1_f1"] >= 4
        ),
        "four_of_five_fold_class1_precision_nonnegative": (
            observed["folds_nonnegative_class1_precision"] >= 4
        ),
        "candidate_placebo_macro_gain_gte_0p001": (
            observed["candidate_placebo_macro_delta"] >= 0.001
        ),
        "candidate_placebo_class1_f1_gain_gte_0p003": (
            observed["candidate_placebo_class1_f1_delta"] >= 0.003
        ),
        "candidate_placebo_class1_precision_gain_gte_0p003": (
            observed["candidate_placebo_class1_precision_delta"] >= 0.003
        ),
        "control_macro_within_0p010_of_keeper": observed["control_keeper_macro_delta"] >= -0.010,
        "control_class1_f1_within_0p020_of_keeper": (
            observed["control_keeper_class1_f1_delta"] >= -0.020
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "architecture_implementation_authorized": not failed,
        "stage_a_training_authorized": False,
        "validation_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _prediction_rows(
    declaration: Mapping[str, object], probabilities: Mapping[str, np.ndarray]
) -> List[Dict[str, object]]:
    labels = np.asarray(declaration["labels"])
    folds = np.asarray(declaration["folds"])
    groups = np.asarray(declaration["groups"])
    paths = np.asarray(declaration["paths"])
    keeper = np.asarray(declaration["probabilities"])
    rows: List[Dict[str, object]] = []
    for index in range(labels.size):
        row: Dict[str, object] = {
            "sample_index": index,
            "source_stem": str(groups[index]),
            "image_path": str(paths[index]),
            "fold": int(folds[index]),
            "target_index": int(labels[index]),
            "keeper_prediction": int(keeper[index].argmax()),
        }
        for class_index in range(keeper.shape[1]):
            row[f"keeper_prob_{class_index}"] = float(keeper[index, class_index])
        for role in ("control", "candidate", "placebo"):
            values = probabilities[role][index]
            row[f"{role}_prediction"] = int(values.argmax())
            row[f"{role}_correct"] = bool(int(values.argmax()) == int(labels[index]))
            for class_index in range(values.size):
                row[f"{role}_prob_{class_index}"] = float(values[class_index])
        rows.append(row)
    return rows


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) != 96 or int(args.workers) != 4:
        raise ValueError("Locked A0 requires batch-size=96 and workers=4")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir = Path(args.output_dir).resolve()
    if _is_relative_to(output_dir, Path(args.data).resolve().parent):
        raise ValueError("Output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    provenance = verify_sources(args)
    declaration = load_clean_fold_declaration(Path(args.fold_csv).resolve())
    device = _resolve_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Keeper checkpoint is not a mapping")
    model = build_model_from_checkpoint(dict(checkpoint)).to(device).eval()
    dataset, class_names = _build_dataset(
        data_yaml=Path(args.data).resolve(),
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    if len(dataset) != EXPECTED_ROWS or len(class_names) != 5:
        raise ValueError(f"Unexpected train dataset/classes: {len(dataset)}/{len(class_names)}")

    start = time.perf_counter()
    extracted = extract_train_descriptors(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    aligned = align_extraction(extracted, declaration)
    del model, dataset, checkpoint
    torch.cuda.empty_cache()

    shared = np.asarray(aligned["shared"], dtype=np.float32)
    detail = np.asarray(aligned["detail"], dtype=np.float32)
    labels = np.asarray(declaration["labels"], dtype=np.int64)
    groups = np.asarray(declaration["groups"], dtype=object)
    folds = np.asarray(declaration["folds"], dtype=np.int64)
    base_probabilities = np.asarray(declaration["probabilities"], dtype=np.float32)
    readouts = fit_oof_readouts(
        shared=shared,
        detail=detail,
        labels=labels,
        groups=groups,
        folds=folds,
        base_probabilities=base_probabilities,
        class_names=class_names,
    )
    probabilities = readouts["probabilities"]
    keeper_metrics = _classification_metrics(labels, base_probabilities, class_names=class_names)
    metrics = readouts["metrics"]
    transitions = _transition_summary(
        labels, probabilities["control"], probabilities["candidate"]
    )
    placebo_transitions = _transition_summary(
        labels, probabilities["placebo"], probabilities["candidate"]
    )
    restricted_fp = {
        role: _restricted_focus_fp(labels, probabilities[role])
        for role in ("control", "candidate", "placebo")
    }

    detail_rank = float(_effective_rank(detail, max_rows=4096))
    full_mean = detail[:, : 3 * DETAIL_PROJECTION_DIM]
    band_stds = {
        "horizontal": float(full_mean[:, :DETAIL_PROJECTION_DIM].std()),
        "vertical": float(
            full_mean[:, DETAIL_PROJECTION_DIM : 2 * DETAIL_PROJECTION_DIM].std()
        ),
        "diagonal": float(full_mean[:, 2 * DETAIL_PROJECTION_DIM :].std()),
    }
    placebo_detail = np.asarray(readouts["placebo_detail_oof"], dtype=np.float32)
    optimizer_rows = list(readouts["optimizer_rows"])
    permutation_rows = list(readouts["permutation_rows"])
    structural_checks = {
        "locked_hashes_match": all(
            provenance["sha256"][key] == LOCKED_SHA256[key] for key in LOCKED_SHA256
        ),
        "row_and_fold_contract_exact": (
            labels.size == EXPECTED_ROWS
            and declaration["class_counts"] == EXPECTED_CLASS_COUNTS
            and declaration["fold_counts"] == EXPECTED_FOLD_COUNTS
            and declaration["source_groups"] == EXPECTED_SOURCE_GROUPS
        ),
        "dataset_alignment_exact": all(
            bool(aligned["alignment"][key])
            for key in ("sample_index_match", "label_match", "path_match", "source_match")
        ),
        "keeper_prediction_recompute_exact": (
            int(aligned["alignment"]["keeper_prediction_mismatches"]) == 0
        ),
        "fold_sources_disjoint": declaration["maximum_source_overlap"] == 0
        and all(int(row["source_overlap"]) == 0 for row in permutation_rows),
        "descriptor_dimensions_exact": (
            shared.shape == (EXPECTED_ROWS, SHARED_DESCRIPTOR_DIM)
            and detail.shape == (EXPECTED_ROWS, DETAIL_DESCRIPTOR_DIM)
        ),
        "all_values_finite": bool(
            np.isfinite(shared).all()
            and np.isfinite(detail).all()
            and all(bool(row["finite"]) for row in optimizer_rows)
        ),
        "detail_noncollapsed": detail_rank >= 12.0
        and all(value > 0.0 for value in band_stds.values()),
        "aligned_detail_differs_from_placebo": not np.array_equal(detail, placebo_detail),
        "placebo_derangements_exact": all(
            int(row["fit_same_source_fixed_points"]) == 0
            and int(row["holdout_same_source_fixed_points"]) == 0
            and bool(row["fit_permutation_stays_in_fit"])
            and bool(row["holdout_permutation_stays_in_holdout"])
            for row in permutation_rows
        ),
        "optimizer_contract_exact": all(
            int(row["iterations"]) < LBFGS_MAX_ITER
            and int(row["parameter_count"]) == int(readouts["parameter_count"])
            and float(row["final_objective"]) <= float(row["initial_objective"]) + 1e-10
            for row in optimizer_rows
        ),
        "train_only_no_model_artifact": True,
    }
    decision = assess_highres_bridge_readiness(
        structural_checks=structural_checks,
        keeper_metrics=keeper_metrics,
        metrics=metrics,
        candidate_control_transitions=transitions,
        control_restricted_fp=restricted_fp["control"],
        candidate_restricted_fp=restricted_fp["candidate"],
        fold_rows=readouts["fold_rows"],
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions_oof.csv"
    fold_path = output_dir / "fold_metrics.csv"
    optimizer_path = output_dir / "optimizer_audit.csv"
    permutation_path = output_dir / "placebo_permutation_audit.csv"
    diagnostics_path = output_dir / "descriptor_diagnostics.json"
    _write_csv(prediction_path, _prediction_rows(declaration, probabilities))
    _write_csv(fold_path, readouts["fold_rows"])
    _write_csv(optimizer_path, optimizer_rows)
    _write_csv(permutation_path, permutation_rows)
    diagnostics = {
        "shared_shape": list(shared.shape),
        "detail_shape": list(detail.shape),
        "total_descriptor_dim": TOTAL_DESCRIPTOR_DIM,
        "detail_effective_rank": detail_rank,
        "haar_full_mean_band_std": band_stds,
        "aligned_detail_equals_placebo": bool(np.array_equal(detail, placebo_detail)),
        "shared_abs_mean": float(np.abs(shared).mean()),
        "detail_abs_mean": float(np.abs(detail).mean()),
        "shared_max_abs": float(np.abs(shared).max()),
        "detail_max_abs": float(np.abs(detail).max()),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    summary = {
        "mode": "highres_shifted_window_bridge_train_only_a0",
        "guardrail": (
            "Frozen keeper and yolo_f/train only. Source-group-held-out residual "
            "readouts; no validation, test, checkpoint, ONNX, raw-data edit, or "
            "architecture training."
        ),
        "provenance": provenance,
        "primary_sources": PRIMARY_SOURCES,
        "class_names": list(class_names),
        "protocol": {
            "seed": SEED,
            "detail_projection_seed": DETAIL_PROJECTION_SEED,
            "device": str(device),
            "extraction_dtype": "fp32",
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "lowres_projection_dim": LOWRES_PROJECTION_DIM,
            "detail_projection_dim": DETAIL_PROJECTION_DIM,
            "erode_ratio": ERODE_RATIO,
            "shared_descriptor_dim": SHARED_DESCRIPTOR_DIM,
            "detail_descriptor_dim": DETAIL_DESCRIPTOR_DIM,
            "total_descriptor_dim": TOTAL_DESCRIPTOR_DIM,
            "haar_bands": ["horizontal", "vertical", "diagonal"],
            "readout": "fp64_offset_residual_linear_lbfgs",
            "residual_scale": RESIDUAL_SCALE,
            "l2_weight": L2_WEIGHT,
            "lbfgs_max_iter": LBFGS_MAX_ITER,
            "lbfgs_history_size": LBFGS_HISTORY_SIZE,
            "class_weight": None,
            "selection": "none",
            "sweep": False,
        },
        "extraction": {
            "rows": int(labels.size),
            "seconds": float(extracted["seconds"]),
            "peak_cuda_memory_gib": float(extracted["peak_cuda_memory_gib"]),
            "alignment": aligned["alignment"],
        },
        "descriptor_diagnostics": diagnostics,
        "keeper_metrics": keeper_metrics,
        "readout_metrics": metrics,
        "transitions": {
            "candidate_vs_control": transitions,
            "candidate_vs_placebo": placebo_transitions,
            "restricted_focus_fp": restricted_fp,
        },
        "fold_metrics": readouts["fold_rows"],
        "optimizer_audit": optimizer_rows,
        "permutation_audit": permutation_rows,
        "decision": decision,
        "elapsed_seconds": float(time.perf_counter() - start),
        "raw_dataset_touched": False,
        "validation_split_used": False,
        "test_split_used": False,
        "model_written": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    manifest = _write_artifact_manifest(
        output_dir, mode="highres_shifted_window_bridge_a0_evidence_manifest"
    )
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest_path": str((output_dir / "artifact_manifest.json").resolve()),
        "artifact_manifest_sha256": _sha256(output_dir / "artifact_manifest.json"),
        "payload_manifest_sha256": manifest["payload_manifest_sha256"],
        "decision": decision,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        result = verify_sources(args)
        result["preflight_only"] = True
        result["output_dir_created"] = False
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return
    print(json.dumps(run_audit(args), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
