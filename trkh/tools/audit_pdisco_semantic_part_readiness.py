from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
)


SEED = 20260712
FOCUS_CLASS_INDEX = 1
NUM_PARTS = 2
PART_DROPOUT = 0.30
GUMBEL_TEMPERATURE = 1.0
HEAD_BATCH_SIZE = 64
FULL_EPOCHS = 12
PREFLIGHT_EPOCHS = 2
FULL_FOLDS = 5
PREFLIGHT_FOLDS = 2
HEAD_LR = 0.01414
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
EXPECTED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
EXPECTED_KEEPER_VAL_MACRO_F1 = 0.8829248142
EXPECTED_KEEPER_VAL_CLASS1_F1 = 0.6782608696
OFFICIAL_PDISCO_COMMIT = "1a872e2bed2ea38c4b078fd69294126e9f6b2f33"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train/validation-only PDiscoFormer-style semantic-part "
            "readiness diagnostic on the frozen TRKH keeper. Test is forbidden."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "train"),
        default="all",
        help="Use train to resume from a previously verified cache.",
    )
    parser.add_argument("--preflight", action="store_true", default=False)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rotate_bbox_cw90(bbox: Tensor) -> Tensor:
    if bbox.ndim != 2 or int(bbox.size(1)) < 4:
        raise ValueError("bbox must have shape [B,4+]")
    value = bbox[:, :4].to(dtype=torch.float32)
    cx, cy, width, height = value.unbind(dim=1)
    return torch.stack((1.0 - cy, cx, height, width), dim=1).clamp(0.0, 1.0)


def align_cw90_patch_indices_to_original(
    indices: Tensor,
    *,
    grid_size: Tuple[int, int],
) -> Tensor:
    if indices.ndim != 2:
        raise ValueError("patch indices must have shape [B,N]")
    height, width = [int(value) for value in grid_size]
    if height <= 0 or width <= 0 or height != width:
        raise ValueError("clockwise 90-degree alignment requires one square grid")
    value = indices.to(dtype=torch.long)
    rotated_row = torch.div(value, width, rounding_mode="floor")
    rotated_col = torch.remainder(value, width)
    original_row = height - 1 - rotated_col
    original_col = rotated_row
    return original_row * width + original_col


def _balanced_subset_indices(dataset: Dataset, limit: int) -> List[int]:
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, Sequence) or not samples:
        return list(range(min(int(limit), len(dataset))))
    buckets: Dict[int, List[int]] = {}
    for index, sample in enumerate(samples):
        label = int(getattr(sample, "primary_label", -1))
        buckets.setdefault(label, []).append(index)
    labels = sorted(label for label in buckets if label >= 0)
    if not labels:
        return list(range(min(int(limit), len(dataset))))
    selected: List[int] = []
    offsets = {label: 0 for label in labels}
    while len(selected) < min(int(limit), len(dataset)):
        progressed = False
        for label in labels:
            offset = offsets[label]
            if offset < len(buckets[label]):
                selected.append(int(buckets[label][offset]))
                offsets[label] = offset + 1
                progressed = True
                if len(selected) >= min(int(limit), len(dataset)):
                    break
        if not progressed:
            break
    return selected


def _build_split_dataset(
    *,
    data_yaml: Path,
    split: str,
    checkpoint: Mapping[str, object],
    preflight: bool,
) -> Tuple[Dataset, List[str]]:
    dataset, class_names = _build_dataset(
        data_yaml=Path(data_yaml),
        split=str(split),
        checkpoint=checkpoint,
        class_name_mode="raw",
        max_samples=0,
    )
    if preflight:
        selected = _balanced_subset_indices(dataset, 128)
        dataset = Subset(dataset, selected)
    return dataset, list(class_names)


def dataset_sample_paths(dataset: Dataset) -> List[str]:
    if isinstance(dataset, Subset):
        parent_paths = dataset_sample_paths(dataset.dataset)
        indices = [int(index) for index in dataset.indices]
        if parent_paths and max(indices, default=-1) >= len(parent_paths):
            raise ValueError("subset index exceeds parent sample paths")
        return [parent_paths[index] for index in indices] if parent_paths else []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    if not callable(sample_paths_fn):
        return []
    paths = [str(path) for path in sample_paths_fn()]
    if len(paths) != len(dataset):
        raise ValueError("dataset sample_paths do not match dataset length")
    return paths


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
) -> Dict[str, object]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or int(probs.shape[0]) != int(target.size):
        raise ValueError("labels and probabilities do not align")
    predictions = probs.argmax(axis=1).astype(np.int64, copy=False)
    indices = np.arange(len(class_names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        predictions,
        labels=indices,
        zero_division=0,
    )
    return {
        "rows": int(target.size),
        "accuracy": float(accuracy_score(target, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {
                "class_index": int(index),
                "class_name": str(class_names[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
        ],
        "confusion_matrix": confusion_matrix(
            target,
            predictions,
            labels=indices,
        ).astype(np.int64).tolist(),
    }


def _focus_f1(metrics: Mapping[str, object], focus_class: int = FOCUS_CLASS_INDEX) -> float:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, Sequence) or int(focus_class) >= len(per_class):
        return 0.0
    row = per_class[int(focus_class)]
    return float(row.get("f1", 0.0)) if isinstance(row, Mapping) else 0.0


def transition_counts(
    labels: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class: int = FOCUS_CLASS_INDEX,
) -> Dict[str, int]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    reference = np.asarray(reference_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    changed = reference != candidate
    reference_correct = reference == target
    candidate_correct = candidate == target
    return {
        "changed": int(changed.sum()),
        "corrections": int((changed & ~reference_correct & candidate_correct).sum()),
        "harms": int((changed & reference_correct & ~candidate_correct).sum()),
        "neutral": int((changed & (reference_correct == candidate_correct)).sum()),
        "focus_fn_rescue": int(
            ((target == focus_class) & (reference != focus_class) & (candidate == focus_class)).sum()
        ),
        "focus_tp_break": int(
            ((target == focus_class) & (reference == focus_class) & (candidate != focus_class)).sum()
        ),
        "focus_fp_remove": int(
            ((target != focus_class) & (reference == focus_class) & (candidate != focus_class)).sum()
        ),
        "focus_fp_create": int(
            ((target != focus_class) & (reference != focus_class) & (candidate == focus_class)).sum()
        ),
    }


def focus_direction_auc(
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class: int = FOCUS_CLASS_INDEX,
) -> Dict[str, object]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    keeper = np.asarray(keeper_probabilities, dtype=np.float64)
    keeper_prediction = keeper.argmax(axis=1)
    focus_fn = (target == focus_class) & (keeper_prediction != focus_class)
    focus_fp = (target != focus_class) & (keeper_prediction == focus_class)
    selected = focus_fn | focus_fp
    direction_labels = focus_fn[selected].astype(np.int64)
    delta = (
        np.asarray(candidate_probabilities, dtype=np.float64)[:, focus_class]
        - np.asarray(control_probabilities, dtype=np.float64)[:, focus_class]
    )[selected]
    auc: Optional[float]
    if int(np.unique(direction_labels).size) == 2:
        auc = float(roc_auc_score(direction_labels, delta))
    else:
        auc = None
    return {
        "auc_fn_positive": auc,
        "focus_fn_rows": int(focus_fn.sum()),
        "focus_fp_rows": int(focus_fp.sum()),
        "selected_rows": int(selected.sum()),
        "mean_delta_focus_fn": float(delta[direction_labels == 1].mean())
        if bool((direction_labels == 1).any())
        else None,
        "mean_delta_focus_fp": float(delta[direction_labels == 0].mean())
        if bool((direction_labels == 0).any())
        else None,
    }


class SemanticPartHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        num_parts: int = NUM_PARTS,
        part_dropout: float = PART_DROPOUT,
        temperature: float = GUMBEL_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.num_parts = int(num_parts)
        self.total_parts = self.num_parts + 1
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.empty(self.total_parts, self.feature_dim))
        self.part_norm = nn.LayerNorm(self.feature_dim)
        self.part_dropout = nn.Dropout1d(float(part_dropout))
        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        self.part_norm.reset_parameters()

    def forward(
        self,
        tokens: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        *,
        grid_size: Tuple[int, int],
        stochastic_assignment: bool,
    ) -> Dict[str, Tensor]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,D]")
        if patch_indices.shape != tokens.shape[:2] or token_valid.shape != tokens.shape[:2]:
            raise ValueError("patch indices/valid mask must align with tokens")
        height, width = [int(value) for value in grid_size]
        grid_tokens = int(height * width)
        if grid_tokens <= 0:
            raise ValueError("grid size must be positive")
        indices = patch_indices.to(device=tokens.device, dtype=torch.long)
        if bool((indices < 0).any().item()) or bool((indices >= grid_tokens).any().item()):
            raise ValueError("patch index is outside the source grid")
        valid = token_valid.to(device=tokens.device, dtype=torch.bool)
        features = F.layer_norm(tokens.to(dtype=torch.float32), (self.feature_dim,))
        prototypes = self.prototypes.to(dtype=torch.float32)
        distances = (
            features.square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.einsum("bnd,pd->bnp", features, prototypes)
            + prototypes.square().sum(dim=-1).view(1, 1, -1)
        )
        assignment_logits = -distances
        if stochastic_assignment:
            assignment = F.gumbel_softmax(
                assignment_logits,
                tau=self.temperature,
                hard=False,
                dim=-1,
            )
        else:
            assignment = F.softmax(assignment_logits / self.temperature, dim=-1)
        background = F.one_hot(
            torch.full_like(indices, self.total_parts - 1),
            num_classes=self.total_parts,
        ).to(dtype=assignment.dtype)
        assignment = torch.where(valid.unsqueeze(-1), assignment, background)

        default_map = assignment.new_zeros(
            (tokens.size(0), grid_tokens, self.total_parts)
        )
        default_map[..., -1] = 1.0
        grid_assignment = default_map.scatter(
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.total_parts),
            assignment,
        )
        maps = grid_assignment.transpose(1, 2).reshape(
            tokens.size(0), self.total_parts, height, width
        )

        weighted_assignment = assignment * valid.unsqueeze(-1).to(assignment.dtype)
        all_part_features = torch.einsum(
            "bnp,bnd->bpd",
            weighted_assignment,
            features,
        ) / float(grid_tokens)
        foreground_features = self.part_norm(all_part_features[:, : self.num_parts])
        foreground_features = self.part_dropout(foreground_features)
        part_logits = self.classifier(foreground_features)
        logits = part_logits.mean(dim=1)
        return {
            "logits": logits,
            "maps": maps,
            "all_part_features": all_part_features,
            "part_logits": part_logits,
            "assignment_logits": assignment_logits,
        }


def total_variation_loss(maps: Tensor) -> Tensor:
    if maps.ndim != 4:
        raise ValueError("maps must have shape [B,P,H,W]")
    vertical = (maps[..., 1:, :] - maps[..., :-1, :]).abs().sum()
    horizontal = (maps[..., :, 1:] - maps[..., :, :-1]).abs().sum()
    denominator = max(1, int(maps.size(0) * maps.size(2) * maps.size(3)))
    return (vertical + horizontal) / float(denominator)


def foreground_presence_loss(maps: Tensor) -> Tensor:
    if maps.ndim != 4 or int(maps.size(2)) < 3 or int(maps.size(3)) < 3:
        raise ValueError("foreground maps must be at least 3x3")
    pooled = F.avg_pool2d(maps, kernel_size=3, stride=1)
    maxima = F.adaptive_max_pool2d(pooled, 1).flatten(start_dim=1).max(dim=0).values
    return 1.0 - maxima.mean()


def part_feature_orthogonality_loss(all_part_features: Tensor) -> Tensor:
    if all_part_features.ndim != 3:
        raise ValueError("part features must have shape [B,P,D]")
    normalized = F.normalize(all_part_features, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    identity = torch.eye(
        int(similarity.size(1)),
        device=similarity.device,
        dtype=similarity.dtype,
    ).unsqueeze(0)
    return (similarity - identity).square().mean()


def enforced_edge_background_loss(maps: Tensor) -> Tensor:
    if maps.ndim != 4 or int(maps.size(2)) < 3 or int(maps.size(3)) < 3:
        raise ValueError("assignment maps must be at least 3x3")
    pooled = F.avg_pool2d(maps, kernel_size=3, stride=1)
    height, width = int(pooled.size(2)), int(pooled.size(3))
    y = torch.linspace(-1.0, 1.0, height, device=maps.device, dtype=maps.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=maps.device, dtype=maps.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    edge_mask = (xx.square() + yy.square())
    edge_mask = edge_mask / edge_mask.max().clamp_min(1e-8)
    background = pooled[:, -1] * edge_mask.unsqueeze(0)
    maxima = F.adaptive_max_pool2d(background.unsqueeze(1), 1).flatten()
    return F.binary_cross_entropy(maxima.clamp(1e-6, 1.0 - 1e-6), torch.ones_like(maxima))


def pixel_assignment_entropy_loss(maps: Tensor) -> Tensor:
    probabilities = maps.clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=1).mean()


def part_equivariance_loss(original_maps: Tensor, aligned_rotated_maps: Tensor) -> Tensor:
    if original_maps.shape != aligned_rotated_maps.shape:
        raise ValueError("equivariance maps must have identical shapes")
    original = original_maps[:, :-1].flatten(start_dim=2)
    rotated = aligned_rotated_maps[:, :-1].flatten(start_dim=2)
    similarity = F.cosine_similarity(original, rotated, dim=-1)
    return 1.0 - similarity.mean()


def semantic_part_losses(
    original_output: Mapping[str, Tensor],
    rotated_output: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    maps = original_output["maps"]
    return {
        "presence": foreground_presence_loss(maps[:, :-1]),
        "equivariance": part_equivariance_loss(maps, rotated_output["maps"]),
        "orthogonality": part_feature_orthogonality_loss(
            original_output["all_part_features"]
        ),
        "total_variation": total_variation_loss(maps),
        "enforced_background": enforced_edge_background_loss(maps),
        "pixel_entropy": pixel_assignment_entropy_loss(maps),
    }


def reconstruct_grid_valid(
    image_valid_mask: Tensor,
    *,
    grid_size: Tuple[int, int],
) -> Tensor:
    mask = image_valid_mask
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError("image valid mask must have shape [B,H,W] or [B,1,H,W]")
    fraction = F.interpolate(
        mask.to(dtype=torch.float32),
        size=grid_size,
        mode="area",
    ).flatten(1)
    valid = fraction > 0.05
    all_invalid = ~valid.any(dim=1)
    if bool(all_invalid.any().item()):
        valid[all_invalid] = True
    return valid


def _forward_keeper_features(
    model: nn.Module,
    images: Tensor,
    crop_bbox: Tensor,
    image_mask: Tensor,
    *,
    include_logits: bool,
    head_bbox: Optional[Tensor] = None,
) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
    features = model.forward_features(
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=crop_bbox,
    )
    logits: Optional[Tensor] = None
    if include_logits:
        if torch.is_tensor(head_bbox):
            features["bbox"] = head_bbox[:, :4].to(
                device=images.device,
                dtype=torch.float32,
            )
        output = (
            model.forward_heads(features)
            if hasattr(model, "forward_heads")
            else classification_logits_from_features(model, features)
        )
        logits, _, _ = extract_detection_from_model_output(output)
    patches = features.get("patches")
    patch_indices = features.get("patch_indices")
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError("keeper did not expose final patch tokens")
    if not torch.is_tensor(patch_indices) or patch_indices.shape != patches.shape[:2]:
        raise ValueError("keeper patch indices do not align")
    key_padding = features.get("memory_key_padding_mask")
    token_valid = (
        ~key_padding.to(dtype=torch.bool)
        if torch.is_tensor(key_padding)
        else torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
    )
    features["audit_token_valid"] = token_valid
    return features, logits


def _open_memmap(
    path: Path,
    *,
    dtype: np.dtype,
    shape: Tuple[int, ...],
) -> np.memmap:
    return np.lib.format.open_memmap(
        Path(path),
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _extract_split_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    cache_dir: Path,
    split: str,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    split_dir = Path(cache_dir) / split
    split_dir.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(workers) > 0),
        collate_fn=_collate_classification,
    )
    total_rows = int(len(dataset))
    arrays: Dict[str, np.memmap] = {}
    labels_all: List[np.ndarray] = []
    sample_indices_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    bbox_all: List[np.ndarray] = []
    paths: List[str] = []
    fallback_paths = dataset_sample_paths(dataset)
    if fallback_paths and len(fallback_paths) != total_rows:
        raise ValueError("fallback sample paths do not match cache rows")
    grid_size_seen: Optional[Tuple[int, int]] = None
    token_shape_seen: Optional[Tuple[int, int]] = None
    row_offset = 0
    rotated_grid_valid_mismatch = 0
    started = time.perf_counter()
    model.eval()

    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"pdisco-cache-{split}", dynamic_ncols=True)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                raise ValueError("classification tensor metadata is required")
            crop_bbox = metadata.get("crop_bbox")
            head_bbox = metadata.get("bbox")
            image_mask = metadata.get("image_mask")
            sample_index = metadata.get("sample_index")
            if (
                not torch.is_tensor(crop_bbox)
                or not torch.is_tensor(head_bbox)
                or not torch.is_tensor(image_mask)
            ):
                raise ValueError("bbox, crop_bbox, and image_mask metadata are required")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            head_bbox = head_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
            image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
            original_features, keeper_logits = _forward_keeper_features(
                model,
                images,
                crop_bbox,
                image_mask,
                include_logits=True,
                head_bbox=head_bbox,
            )
            if keeper_logits is None:
                raise RuntimeError("keeper logits were not produced")
            grid_size = tuple(int(value) for value in original_features["grid_size"])
            if len(grid_size) != 2 or min(grid_size) <= 0 or grid_size[0] != grid_size[1]:
                raise ValueError("semantic-part cache requires one square patch grid")

            rotated_images = torch.rot90(images, k=-1, dims=(-2, -1))
            rotated_mask = torch.rot90(image_mask, k=-1, dims=(-2, -1))
            rotated_bbox = rotate_bbox_cw90(crop_bbox)
            rotated_features, _ = _forward_keeper_features(
                model,
                rotated_images,
                rotated_bbox,
                rotated_mask,
                include_logits=False,
            )
            rotated_grid_size = tuple(int(value) for value in rotated_features["grid_size"])
            if rotated_grid_size != grid_size:
                raise ValueError("original and rotated keeper grids differ")

            original_tokens = original_features["patches"].detach().float()
            rotated_tokens = rotated_features["patches"].detach().float()
            original_indices = original_features["patch_indices"].detach().long()
            rotated_indices = align_cw90_patch_indices_to_original(
                rotated_features["patch_indices"].detach(),
                grid_size=grid_size,
            )
            original_valid = original_features["audit_token_valid"].detach().bool()
            rotated_valid = rotated_features["audit_token_valid"].detach().bool()
            if original_tokens.shape[1:] != rotated_tokens.shape[1:]:
                raise ValueError("original and rotated token shapes differ")
            if any(
                torch.unique(row).numel() != row.numel()
                for row in original_indices
            ):
                raise ValueError("keeper emitted duplicate original patch indices")
            if any(
                torch.unique(row).numel() != row.numel()
                for row in rotated_indices
            ):
                raise ValueError("keeper emitted duplicate aligned rotated patch indices")

            original_grid_valid = reconstruct_grid_valid(
                image_mask,
                grid_size=grid_size,
            )
            rotated_grid_valid = reconstruct_grid_valid(
                rotated_mask,
                grid_size=grid_size,
            ).reshape(-1, *grid_size)
            rotated_grid_valid = torch.rot90(
                rotated_grid_valid,
                k=1,
                dims=(-2, -1),
            ).flatten(1)
            rotated_grid_valid_mismatch += int(
                (original_grid_valid != rotated_grid_valid).sum().item()
            )

            batch_rows = int(labels.numel())
            token_shape = (int(original_tokens.size(1)), int(original_tokens.size(2)))
            if grid_size_seen is None:
                grid_size_seen = grid_size
                token_shape_seen = token_shape
                arrays = {
                    "tokens_original": _open_memmap(
                        split_dir / "tokens_original.npy",
                        dtype=np.float16,
                        shape=(total_rows, *token_shape),
                    ),
                    "tokens_rot90": _open_memmap(
                        split_dir / "tokens_rot90.npy",
                        dtype=np.float16,
                        shape=(total_rows, *token_shape),
                    ),
                    "patch_indices_original": _open_memmap(
                        split_dir / "patch_indices_original.npy",
                        dtype=np.int16,
                        shape=(total_rows, token_shape[0]),
                    ),
                    "patch_indices_rot90_aligned": _open_memmap(
                        split_dir / "patch_indices_rot90_aligned.npy",
                        dtype=np.int16,
                        shape=(total_rows, token_shape[0]),
                    ),
                    "token_valid_original": _open_memmap(
                        split_dir / "token_valid_original.npy",
                        dtype=np.uint8,
                        shape=(total_rows, token_shape[0]),
                    ),
                    "token_valid_rot90": _open_memmap(
                        split_dir / "token_valid_rot90.npy",
                        dtype=np.uint8,
                        shape=(total_rows, token_shape[0]),
                    ),
                    "grid_valid": _open_memmap(
                        split_dir / "grid_valid.npy",
                        dtype=np.uint8,
                        shape=(total_rows, int(grid_size[0] * grid_size[1])),
                    ),
                }
            elif grid_size != grid_size_seen or token_shape != token_shape_seen:
                raise ValueError("keeper token geometry changed within a split")

            row_slice = slice(row_offset, row_offset + batch_rows)
            arrays["tokens_original"][row_slice] = original_tokens.cpu().numpy().astype(
                np.float16
            )
            arrays["tokens_rot90"][row_slice] = rotated_tokens.cpu().numpy().astype(
                np.float16
            )
            arrays["patch_indices_original"][row_slice] = original_indices.cpu().numpy().astype(
                np.int16
            )
            arrays["patch_indices_rot90_aligned"][row_slice] = rotated_indices.cpu().numpy().astype(
                np.int16
            )
            arrays["token_valid_original"][row_slice] = original_valid.cpu().numpy().astype(
                np.uint8
            )
            arrays["token_valid_rot90"][row_slice] = rotated_valid.cpu().numpy().astype(
                np.uint8
            )
            arrays["grid_valid"][row_slice] = original_grid_valid.cpu().numpy().astype(
                np.uint8
            )
            labels_all.append(labels.cpu().numpy().astype(np.int64, copy=False))
            probabilities_all.append(
                keeper_logits.detach().float().softmax(dim=1).cpu().numpy()
            )
            bbox_all.append(crop_bbox.detach().cpu().numpy().astype(np.float32, copy=False))
            if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_rows:
                sample_indices_all.append(
                    sample_index.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                )
            else:
                sample_indices_all.append(
                    np.arange(row_offset, row_offset + batch_rows, dtype=np.int64)
                )
            raw_paths = metadata.get("paths", [])
            batch_paths = (
                [str(path) for path in raw_paths]
                if isinstance(raw_paths, Sequence) and len(raw_paths) == batch_rows
                else [""] * batch_rows
            )
            if fallback_paths:
                fallback_batch = fallback_paths[row_offset : row_offset + batch_rows]
                batch_paths = [
                    path if path.strip() else fallback_batch[index]
                    for index, path in enumerate(batch_paths)
                ]
            if len(batch_paths) != batch_rows or any(not path.strip() for path in batch_paths):
                raise ValueError("batch paths are missing or misaligned after fallback")
            paths.extend(batch_paths)
            row_offset += batch_rows

    if row_offset != total_rows or grid_size_seen is None or token_shape_seen is None:
        raise ValueError(f"cache row mismatch: observed={row_offset}, expected={total_rows}")
    for array in arrays.values():
        array.flush()
    del arrays

    labels_array = np.concatenate(labels_all).astype(np.int64, copy=False)
    sample_index_array = np.concatenate(sample_indices_all).astype(np.int64, copy=False)
    probability_array = np.concatenate(probabilities_all).astype(np.float32, copy=False)
    bbox_array = np.concatenate(bbox_all).astype(np.float32, copy=False)
    source_stems = np.asarray([Path(path).stem.casefold() for path in paths], dtype=str)
    path_array = np.asarray(paths, dtype=str)
    np.save(split_dir / "labels.npy", labels_array, allow_pickle=False)
    np.save(split_dir / "sample_index.npy", sample_index_array, allow_pickle=False)
    np.save(split_dir / "keeper_probabilities.npy", probability_array, allow_pickle=False)
    np.save(split_dir / "bbox.npy", bbox_array, allow_pickle=False)
    np.save(split_dir / "paths.npy", path_array, allow_pickle=False)
    np.save(split_dir / "source_stems.npy", source_stems, allow_pickle=False)

    file_records: Dict[str, object] = {}
    for path in sorted(split_dir.glob("*.npy")):
        file_records[path.name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    metrics = classification_metrics(
        labels_array,
        probability_array,
        class_names=class_names,
    )
    summary = {
        "split": str(split),
        "rows": int(total_rows),
        "source_groups": int(np.unique(source_stems).size),
        "class_counts": np.bincount(
            labels_array,
            minlength=len(class_names),
        ).astype(np.int64).tolist(),
        "sample_index_unique": int(np.unique(sample_index_array).size),
        "grid_size": list(grid_size_seen),
        "token_shape": list(token_shape_seen),
        "token_cache_dtype": "float16",
        "keeper_forward_precision": "fp32",
        "rotated_grid_valid_mismatch": int(rotated_grid_valid_mismatch),
        "direct_keeper_metrics": metrics,
        "files": file_records,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (split_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _verify_cache_files(cache_dir: Path) -> Dict[str, object]:
    summary_path = Path(cache_dir) / "cache_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"cache summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        split_summary = summary.get("splits", {}).get(split, {})
        files = split_summary.get("files", {})
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"cache summary has no {split} file records")
        for name, record in files.items():
            path = Path(cache_dir) / split / str(name)
            if not path.is_file():
                raise FileNotFoundError(f"cache payload is missing: {path}")
            if int(path.stat().st_size) != int(record.get("bytes", -1)):
                raise ValueError(f"cache payload size changed: {path}")
            if sha256_file(path) != str(record.get("sha256", "")):
                raise ValueError(f"cache payload hash changed: {path}")
    return summary


@dataclass
class SplitCache:
    root: Path
    tokens_original: np.ndarray
    tokens_rot90: np.ndarray
    patch_indices_original: np.ndarray
    patch_indices_rot90_aligned: np.ndarray
    token_valid_original: np.ndarray
    token_valid_rot90: np.ndarray
    grid_valid: np.ndarray
    labels: np.ndarray
    sample_index: np.ndarray
    keeper_probabilities: np.ndarray
    bbox: np.ndarray
    paths: np.ndarray
    source_stems: np.ndarray
    grid_size: Tuple[int, int]

    @classmethod
    def load(cls, cache_dir: Path, split: str, split_summary: Mapping[str, object]) -> "SplitCache":
        root = Path(cache_dir) / str(split)

        def load(name: str) -> np.ndarray:
            return np.load(root / name, mmap_mode="r", allow_pickle=False)

        grid = tuple(int(value) for value in split_summary.get("grid_size", []))
        if len(grid) != 2:
            raise ValueError(f"invalid grid in {split} cache summary")
        payload = cls(
            root=root,
            tokens_original=load("tokens_original.npy"),
            tokens_rot90=load("tokens_rot90.npy"),
            patch_indices_original=load("patch_indices_original.npy"),
            patch_indices_rot90_aligned=load("patch_indices_rot90_aligned.npy"),
            token_valid_original=load("token_valid_original.npy"),
            token_valid_rot90=load("token_valid_rot90.npy"),
            grid_valid=load("grid_valid.npy"),
            labels=load("labels.npy"),
            sample_index=load("sample_index.npy"),
            keeper_probabilities=load("keeper_probabilities.npy"),
            bbox=load("bbox.npy"),
            paths=load("paths.npy"),
            source_stems=load("source_stems.npy"),
            grid_size=grid,
        )
        rows = int(payload.labels.shape[0])
        for name in (
            "tokens_original",
            "tokens_rot90",
            "patch_indices_original",
            "patch_indices_rot90_aligned",
            "token_valid_original",
            "token_valid_rot90",
            "grid_valid",
            "sample_index",
            "keeper_probabilities",
            "bbox",
            "paths",
            "source_stems",
        ):
            if int(getattr(payload, name).shape[0]) != rows:
                raise ValueError(f"{split} cache row mismatch for {name}")
        return payload

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _tensor_batch(
    array: np.ndarray,
    row_indices: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    values = np.asarray(array[np.asarray(row_indices, dtype=np.int64)])
    return torch.as_tensor(values, device=device, dtype=dtype)


def _head_initial_state(
    *,
    feature_dim: int,
    num_classes: int,
    seed: int,
) -> Tuple[Dict[str, Tensor], str, int]:
    _seed_everything(seed)
    head = SemanticPartHead(feature_dim=feature_dim, num_classes=num_classes)
    state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    parameter_count = int(sum(parameter.numel() for parameter in head.parameters()))
    return state, sha256_state_dict(state), parameter_count


def _train_head(
    *,
    cache: SplitCache,
    train_indices: np.ndarray,
    initial_state: Mapping[str, Tensor],
    semantic_losses_enabled: bool,
    device: torch.device,
    epochs: int,
    seed: int,
    num_classes: int,
) -> Tuple[SemanticPartHead, List[Dict[str, float]], Dict[str, object]]:
    feature_dim = int(cache.tokens_original.shape[2])
    head = SemanticPartHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
    head.load_state_dict(initial_state, strict=True)
    loaded_initial_state_sha256 = sha256_state_dict(head.state_dict())
    optimizer = torch.optim.Adam(head.parameters(), lr=HEAD_LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )
    rows = np.asarray(train_indices, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("head train split is empty")
    curves: List[Dict[str, float]] = []
    nonfinite_steps = 0
    maximum_allocated = 0
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        head.train()
        order = np.random.default_rng(int(seed + epoch * 1009)).permutation(rows)
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "presence": 0.0,
            "equivariance": 0.0,
            "orthogonality": 0.0,
            "total_variation": 0.0,
            "enforced_background": 0.0,
            "pixel_entropy": 0.0,
        }
        samples_seen = 0
        for batch_number, start in enumerate(range(0, int(order.size), HEAD_BATCH_SIZE)):
            batch_rows = order[start : start + HEAD_BATCH_SIZE]
            tokens = _tensor_batch(
                cache.tokens_original,
                batch_rows,
                device=device,
                dtype=torch.float32,
            )
            indices = _tensor_batch(
                cache.patch_indices_original,
                batch_rows,
                device=device,
                dtype=torch.long,
            )
            valid = _tensor_batch(
                cache.token_valid_original,
                batch_rows,
                device=device,
                dtype=torch.bool,
            )
            labels = _tensor_batch(
                cache.labels,
                batch_rows,
                device=device,
                dtype=torch.long,
            )
            original_seed = int(seed + epoch * 1_000_003 + batch_number * 101)
            torch.manual_seed(original_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(original_seed)
            original_output = head(
                tokens,
                indices,
                valid,
                grid_size=cache.grid_size,
                stochastic_assignment=True,
            )
            classification = F.cross_entropy(original_output["logits"], labels)
            component_losses: Dict[str, Tensor] = {
                "presence": classification.new_zeros(()),
                "equivariance": classification.new_zeros(()),
                "orthogonality": classification.new_zeros(()),
                "total_variation": classification.new_zeros(()),
                "enforced_background": classification.new_zeros(()),
                "pixel_entropy": classification.new_zeros(()),
            }
            loss = classification
            if semantic_losses_enabled:
                rotated_tokens = _tensor_batch(
                    cache.tokens_rot90,
                    batch_rows,
                    device=device,
                    dtype=torch.float32,
                )
                rotated_indices = _tensor_batch(
                    cache.patch_indices_rot90_aligned,
                    batch_rows,
                    device=device,
                    dtype=torch.long,
                )
                rotated_valid = _tensor_batch(
                    cache.token_valid_rot90,
                    batch_rows,
                    device=device,
                    dtype=torch.bool,
                )
                rotated_seed = int(original_seed + 53)
                torch.manual_seed(rotated_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(rotated_seed)
                rotated_output = head(
                    rotated_tokens,
                    rotated_indices,
                    rotated_valid,
                    grid_size=cache.grid_size,
                    stochastic_assignment=True,
                )
                component_losses = semantic_part_losses(original_output, rotated_output)
                loss = (
                    classification
                    + component_losses["presence"]
                    + component_losses["equivariance"]
                    + component_losses["orthogonality"]
                    + component_losses["total_variation"]
                    + 2.0 * component_losses["enforced_background"]
                    + component_losses["pixel_entropy"]
                )
            if not bool(torch.isfinite(loss).item()):
                nonfinite_steps += 1
                raise FloatingPointError(
                    f"non-finite semantic-part loss at epoch={epoch + 1}, batch={batch_number}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=2.0)
            if not bool(torch.isfinite(gradient_norm).item()):
                nonfinite_steps += 1
                raise FloatingPointError("non-finite semantic-part gradient norm")
            optimizer.step()

            batch_rows_count = int(labels.numel())
            samples_seen += batch_rows_count
            totals["loss"] += float(loss.detach().item()) * batch_rows_count
            totals["classification"] += float(classification.detach().item()) * batch_rows_count
            for name, value in component_losses.items():
                totals[name] += float(value.detach().item()) * batch_rows_count
            if device.type == "cuda":
                maximum_allocated = max(
                    maximum_allocated,
                    int(torch.cuda.max_memory_allocated(device)),
                )
        scheduler.step()
        curves.append(
            {
                "epoch": float(epoch + 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{
                    name: float(value / max(1, samples_seen))
                    for name, value in totals.items()
                },
            }
        )
    telemetry = {
        "semantic_losses_enabled": bool(semantic_losses_enabled),
        "loaded_initial_state_sha256": loaded_initial_state_sha256,
        "rows": int(rows.size),
        "epochs": int(epochs),
        "batch_size": int(HEAD_BATCH_SIZE),
        "optimizer": "Adam",
        "learning_rate": float(HEAD_LR),
        "weight_decay": 0.0,
        "scheduler": {
            "type": "StepLR",
            "step_size": int(SCHEDULER_STEP_SIZE),
            "gamma": float(SCHEDULER_GAMMA),
        },
        "nonfinite_steps": int(nonfinite_steps),
        "maximum_cuda_memory_allocated_bytes": int(maximum_allocated),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return head, curves, telemetry


def _predict_head(
    *,
    head: SemanticPartHead,
    cache: SplitCache,
    row_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    outputs: List[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, int(rows.size), HEAD_BATCH_SIZE):
            batch_rows = rows[start : start + HEAD_BATCH_SIZE]
            result = head(
                _tensor_batch(
                    cache.tokens_original,
                    batch_rows,
                    device=device,
                    dtype=torch.float32,
                ),
                _tensor_batch(
                    cache.patch_indices_original,
                    batch_rows,
                    device=device,
                    dtype=torch.long,
                ),
                _tensor_batch(
                    cache.token_valid_original,
                    batch_rows,
                    device=device,
                    dtype=torch.bool,
                ),
                grid_size=cache.grid_size,
                stochastic_assignment=False,
            )
            outputs.append(result["logits"].softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _bbox_center_mask(bbox: Tensor, grid_size: Tuple[int, int]) -> Tensor:
    height, width = [int(value) for value in grid_size]
    y = (torch.arange(height, device=bbox.device, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, device=bbox.device, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx, cy, bw, bh = bbox[:, :4].to(dtype=torch.float32).unbind(dim=1)
    inside_x = (xx.unsqueeze(0) - cx[:, None, None]).abs() <= 0.5 * bw[:, None, None]
    inside_y = (yy.unsqueeze(0) - cy[:, None, None]).abs() <= 0.5 * bh[:, None, None]
    return (inside_x & inside_y).flatten(1)


def _evaluate_part_metrics(
    *,
    head: SemanticPartHead,
    cache: SplitCache,
    row_indices: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    rows = np.asarray(row_indices, dtype=np.int64)
    part_mass_sum = np.zeros(NUM_PARTS, dtype=np.float64)
    valid_positions = 0.0
    background_inside_sum = 0.0
    background_inside_count = 0.0
    background_outside_sum = 0.0
    background_outside_count = 0.0
    equivariance_sum = 0.0
    equivariance_count = 0
    tv_sum = 0.0
    entropy_sum = 0.0
    samples_seen = 0
    head.eval()
    with torch.inference_mode():
        for start in range(0, int(rows.size), HEAD_BATCH_SIZE):
            batch_rows = rows[start : start + HEAD_BATCH_SIZE]
            original = head(
                _tensor_batch(cache.tokens_original, batch_rows, device=device, dtype=torch.float32),
                _tensor_batch(
                    cache.patch_indices_original,
                    batch_rows,
                    device=device,
                    dtype=torch.long,
                ),
                _tensor_batch(
                    cache.token_valid_original,
                    batch_rows,
                    device=device,
                    dtype=torch.bool,
                ),
                grid_size=cache.grid_size,
                stochastic_assignment=False,
            )
            rotated = head(
                _tensor_batch(cache.tokens_rot90, batch_rows, device=device, dtype=torch.float32),
                _tensor_batch(
                    cache.patch_indices_rot90_aligned,
                    batch_rows,
                    device=device,
                    dtype=torch.long,
                ),
                _tensor_batch(
                    cache.token_valid_rot90,
                    batch_rows,
                    device=device,
                    dtype=torch.bool,
                ),
                grid_size=cache.grid_size,
                stochastic_assignment=False,
            )
            maps = original["maps"]
            rotated_maps = rotated["maps"]
            grid_valid = _tensor_batch(
                cache.grid_valid,
                batch_rows,
                device=device,
                dtype=torch.bool,
            )
            valid_float = grid_valid.to(dtype=maps.dtype)
            flat_maps = maps.flatten(start_dim=2)
            part_mass_sum += (
                flat_maps[:, :NUM_PARTS] * valid_float.unsqueeze(1)
            ).sum(dim=(0, 2)).cpu().numpy()
            valid_positions += float(valid_float.sum().item())

            bbox = _tensor_batch(cache.bbox, batch_rows, device=device, dtype=torch.float32)
            inside = _bbox_center_mask(bbox, cache.grid_size) & grid_valid
            outside = ~_bbox_center_mask(bbox, cache.grid_size) & grid_valid
            background = flat_maps[:, -1]
            background_inside_sum += float((background * inside).sum().item())
            background_inside_count += float(inside.sum().item())
            background_outside_sum += float((background * outside).sum().item())
            background_outside_count += float(outside.sum().item())

            original_fg = flat_maps[:, :NUM_PARTS] * valid_float.unsqueeze(1)
            rotated_fg = (
                rotated_maps.flatten(start_dim=2)[:, :NUM_PARTS]
                * valid_float.unsqueeze(1)
            )
            cosine = F.cosine_similarity(original_fg, rotated_fg, dim=-1)
            equivariance_sum += float(cosine.sum().item())
            equivariance_count += int(cosine.numel())
            batch_size = int(maps.size(0))
            tv_sum += float(total_variation_loss(maps).item()) * batch_size
            entropy_sum += float(pixel_assignment_entropy_loss(maps).item()) * batch_size
            samples_seen += batch_size

    mean_part_mass = part_mass_sum / max(valid_positions, 1.0)
    normalized_use = mean_part_mass / max(float(mean_part_mass.sum()), 1e-12)
    use_entropy = float(
        -(normalized_use * np.log(np.clip(normalized_use, 1e-12, None))).sum()
        / math.log(float(NUM_PARTS))
    )
    background_inside = background_inside_sum / max(background_inside_count, 1.0)
    background_outside = background_outside_sum / max(background_outside_count, 1.0)
    return {
        "rows": int(rows.size),
        "foreground_part_mean_mass": mean_part_mass.tolist(),
        "foreground_part_normalized_use": normalized_use.tolist(),
        "foreground_part_use_entropy_normalized": use_entropy,
        "background_inside_bbox_mean": float(background_inside),
        "background_outside_bbox_mean": float(background_outside),
        "background_outside_minus_inside": float(background_outside - background_inside),
        "foreground_equivariance_cosine": float(
            equivariance_sum / max(1, equivariance_count)
        ),
        "total_variation": float(tv_sum / max(1, samples_seen)),
        "pixel_entropy": float(entropy_sum / max(1, samples_seen)),
        "valid_grid_positions": int(valid_positions),
    }


def assign_source_grouped_folds(
    labels: np.ndarray,
    source_stems: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_stems).astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments = np.full(target.size, -1, dtype=np.int64)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    overlap_count = 0
    fold_rows: List[Dict[str, object]] = []
    for fold_index, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(target.size), target, groups)
    ):
        fit_sources = set(groups[fit_indices].tolist())
        holdout_sources = set(groups[holdout_indices].tolist())
        overlap = fit_sources.intersection(holdout_sources)
        overlap_count += len(overlap)
        assignments[holdout_indices] = int(fold_index)
        splits.append((fit_indices.astype(np.int64), holdout_indices.astype(np.int64)))
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "fit_sources": int(len(fit_sources)),
                "holdout_sources": int(len(holdout_sources)),
                "source_overlap": int(len(overlap)),
                "holdout_class_counts": np.bincount(
                    target[holdout_indices],
                    minlength=int(target.max(initial=0) + 1),
                ).tolist(),
            }
        )
    if bool((assignments < 0).any()) or not np.array_equal(
        np.bincount(assignments, minlength=int(folds)) > 0,
        np.ones(int(folds), dtype=bool),
    ):
        raise RuntimeError("source-grouped OOF assignment is incomplete")
    return splits, {
        "folds": fold_rows,
        "source_overlap": int(overlap_count),
        "assignment_complete": bool(np.all(assignments >= 0)),
        "assignment_counts": np.bincount(assignments, minlength=int(folds)).tolist(),
    }


def _write_predictions_csv(
    path: Path,
    *,
    cache: SplitCache,
    row_indices: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> None:
    rows = np.asarray(row_indices, dtype=np.int64)
    keeper = np.asarray(cache.keeper_probabilities[rows], dtype=np.float64)
    control = np.asarray(control_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    fieldnames = [
        "row_index",
        "sample_index",
        "image_path",
        "source_stem",
        "target_index",
        "keeper_prediction",
        "control_prediction",
        "candidate_prediction",
        "candidate_corrects_keeper",
        "candidate_harms_keeper",
    ]
    for prefix in ("keeper", "control", "candidate"):
        fieldnames.extend(f"{prefix}_prob_{index}" for index in range(keeper.shape[1]))
    fieldnames.append("candidate_minus_control_focus_probability")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for output_index, row_index in enumerate(rows.tolist()):
            target = int(cache.labels[row_index])
            keeper_prediction = int(keeper[output_index].argmax())
            control_prediction = int(control[output_index].argmax())
            candidate_prediction = int(candidate[output_index].argmax())
            row: Dict[str, object] = {
                "row_index": int(row_index),
                "sample_index": int(cache.sample_index[row_index]),
                "image_path": str(cache.paths[row_index]),
                "source_stem": str(cache.source_stems[row_index]),
                "target_index": target,
                "keeper_prediction": keeper_prediction,
                "control_prediction": control_prediction,
                "candidate_prediction": candidate_prediction,
                "candidate_corrects_keeper": int(
                    keeper_prediction != target and candidate_prediction == target
                ),
                "candidate_harms_keeper": int(
                    keeper_prediction == target and candidate_prediction != target
                ),
                "candidate_minus_control_focus_probability": float(
                    candidate[output_index, FOCUS_CLASS_INDEX]
                    - control[output_index, FOCUS_CLASS_INDEX]
                ),
            }
            for prefix, values in (
                ("keeper", keeper[output_index]),
                ("control", control[output_index]),
                ("candidate", candidate[output_index]),
            ):
                for class_index, value in enumerate(values):
                    row[f"{prefix}_prob_{class_index}"] = float(value)
            writer.writerow(row)


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    image = np.asarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    heat_value = np.asarray(heat, dtype=np.float32).clip(0.0, 1.0)
    heat_image = Image.fromarray((heat_value * 255.0).round().astype(np.uint8)).resize(
        (width, height),
        getattr(Image, "Resampling", Image).BILINEAR,
    )
    alpha = np.asarray(heat_image, dtype=np.float32)[..., None] / 255.0 * 0.68
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = image.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return Image.fromarray(overlay.clip(0.0, 255.0).round().astype(np.uint8))


def _bbox_image(rgb: np.ndarray, bbox: Sequence[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    cx, cy, bw, bh = [float(value) for value in bbox[:4]]
    x1 = int(round((cx - 0.5 * bw) * width))
    y1 = int(round((cy - 0.5 * bh) * height))
    x2 = int(round((cx + 0.5 * bw) * width))
    y2 = int(round((cy + 0.5 * bh) * height))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 35, 35), width=3)
    return image


def _select_review_rows(
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> List[Tuple[str, int]]:
    target = np.asarray(labels, dtype=np.int64)
    keeper = np.asarray(keeper_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    selected: List[Tuple[str, int]] = []
    used: set[int] = set()

    def add(cohort: str, mask: np.ndarray, limit: int = 1) -> None:
        candidates = np.flatnonzero(mask)
        added = 0
        for index in candidates.tolist():
            if int(index) in used:
                continue
            selected.append((str(cohort), int(index)))
            used.add(int(index))
            added += 1
            if added >= int(limit):
                break

    for class_index in range(int(np.max(target)) + 1):
        add(
            f"class_{class_index}_candidate_correct",
            (target == class_index) & (candidate == class_index),
        )
    add("keeper_class1_tp", (target == 1) & (keeper == 1))
    add("keeper_class1_fn", (target == 1) & (keeper != 1), limit=2)
    add("keeper_class1_fp", (target != 1) & (keeper == 1), limit=2)
    add("candidate_correction", (keeper != target) & (candidate == target), limit=2)
    add("candidate_harm", (keeper == target) & (candidate != target), limit=2)
    return selected


def _head_maps_for_rows(
    *,
    head: SemanticPartHead,
    cache: SplitCache,
    rows: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    indices = np.asarray(list(rows), dtype=np.int64)
    head.eval()
    with torch.inference_mode():
        output = head(
            _tensor_batch(cache.tokens_original, indices, device=device, dtype=torch.float32),
            _tensor_batch(
                cache.patch_indices_original,
                indices,
                device=device,
                dtype=torch.long,
            ),
            _tensor_batch(
                cache.token_valid_original,
                indices,
                device=device,
                dtype=torch.bool,
            ),
            grid_size=cache.grid_size,
            stochastic_assignment=False,
        )
    return output["maps"].cpu().numpy().astype(np.float32, copy=False)


def _write_contact_sheet(
    path: Path,
    *,
    dataset: Dataset,
    cache: SplitCache,
    control_head: SemanticPartHead,
    candidate_head: SemanticPartHead,
    candidate_probabilities: np.ndarray,
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    selected = _select_review_rows(
        np.asarray(cache.labels),
        np.asarray(cache.keeper_probabilities),
        candidate_probabilities,
    )
    if not selected:
        raise ValueError("no validation rows were available for semantic-part review")
    cache_rows = [row for _, row in selected]
    control_maps = _head_maps_for_rows(
        head=control_head,
        cache=cache,
        rows=cache_rows,
        device=device,
    )
    candidate_maps = _head_maps_for_rows(
        head=candidate_head,
        cache=cache,
        rows=cache_rows,
        device=device,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tile = 176
    header = 24
    columns = [
        "image",
        "crop_bbox",
        "candidate_part0",
        "candidate_part1",
        "candidate_bg",
        "control_part0",
        "control_part1",
        "control_bg",
    ]
    canvas = Image.new(
        "RGB",
        (tile * len(columns), header + (tile + header) * len(selected)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((column * tile + 4, 5), label, fill=(10, 10, 10))
    manifest_rows: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    colors = ((230, 35, 35), (25, 180, 80), (35, 90, 235))
    for output_row, ((cohort, cache_row), control_map, candidate_map) in enumerate(
        zip(selected, control_maps, candidate_maps)
    ):
        sample_index = int(cache.sample_index[cache_row])
        image, label, metadata = dataset[sample_index]
        rgb = (
            image.detach().float().cpu() * std_tensor + mean_tensor
        ).permute(1, 2, 0).numpy().clip(0.0, 1.0)
        rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
        bbox = np.asarray(cache.bbox[cache_row], dtype=np.float32)
        images = [
            Image.fromarray(rgb_u8),
            _bbox_image(rgb_u8, bbox),
            *[
                _heat_overlay(rgb_u8, candidate_map[index], colors[index])
                for index in range(NUM_PARTS + 1)
            ],
            *[
                _heat_overlay(rgb_u8, control_map[index], colors[index])
                for index in range(NUM_PARTS + 1)
            ],
        ]
        y = header + output_row * (tile + header)
        row_label = (
            f"{cohort} idx={sample_index} y={int(label)} "
            f"k={int(np.asarray(cache.keeper_probabilities[cache_row]).argmax())} "
            f"c={int(candidate_probabilities[cache_row].argmax())}"
        )
        draw.text((4, y + 4), row_label, fill=(10, 10, 10))
        for column, tile_image in enumerate(images):
            canvas.paste(
                tile_image.resize((tile, tile), resampling.BILINEAR),
                (column * tile, y + header),
            )
        manifest_rows.append(
            {
                "row": int(output_row),
                "cohort": str(cohort),
                "cache_row": int(cache_row),
                "sample_index": int(sample_index),
                "image_path": str(cache.paths[cache_row]),
                "target_index": int(label),
                "keeper_prediction": int(
                    np.asarray(cache.keeper_probabilities[cache_row]).argmax()
                ),
                "candidate_prediction": int(candidate_probabilities[cache_row].argmax()),
                "bbox": bbox.tolist(),
            }
        )
    canvas.save(path)
    manifest = {"columns": columns, "rows": manifest_rows}
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def evaluate_gates(
    *,
    preflight: bool,
    checkpoint_sha256: str,
    train_cache: SplitCache,
    val_cache: SplitCache,
    fold_summary: Mapping[str, object],
    oof_metrics: Mapping[str, Mapping[str, object]],
    val_metrics: Mapping[str, Mapping[str, object]],
    fold_results: Sequence[Mapping[str, object]],
    transitions: Mapping[str, Mapping[str, int]],
    direction: Mapping[str, Mapping[str, object]],
    part_metrics: Mapping[str, Mapping[str, object]],
    initialization_equal: bool,
    all_finite: bool,
) -> Dict[str, object]:
    if preflight:
        return {
            "mode": "preflight_contract_only",
            "checks": {},
            "failed_checks": [],
            "all_automated_checks_passed": None,
            "manual_visual_review_required": True,
            "image_smoke_permitted": False,
        }
    train_sources = set(np.asarray(train_cache.source_stems).astype(str).tolist())
    val_sources = set(np.asarray(val_cache.source_stems).astype(str).tolist())
    control_oof = oof_metrics["control"]
    candidate_oof = oof_metrics["candidate"]
    control_val = val_metrics["control"]
    candidate_val = val_metrics["candidate"]
    keeper_val = val_metrics["keeper"]
    fold_wins = sum(
        _focus_f1(row["candidate"]) > _focus_f1(row["control"])
        for row in fold_results
    )
    keeper_transition = transitions["candidate_vs_keeper_val"]
    oof_auc = direction["train_oof_candidate_vs_control"].get("auc_fn_positive")
    val_auc = direction["val_candidate_vs_control"].get("auc_fn_positive")
    candidate_parts = part_metrics["candidate"]
    control_parts = part_metrics["control"]
    part_mass = [float(value) for value in candidate_parts["foreground_part_mean_mass"]]
    checks = {
        "checkpoint_hash_locked": str(checkpoint_sha256) == EXPECTED_KEEPER_SHA256,
        "train_support": len(train_cache) == EXPECTED_TRAIN_ROWS,
        "validation_support": len(val_cache) == EXPECTED_VAL_ROWS,
        "train_sample_index_unique": int(np.unique(train_cache.sample_index).size)
        == len(train_cache),
        "validation_sample_index_unique": int(np.unique(val_cache.sample_index).size)
        == len(val_cache),
        "train_validation_source_disjoint": len(train_sources.intersection(val_sources)) == 0,
        "source_group_folds_disjoint": int(fold_summary.get("source_overlap", -1)) == 0,
        "oof_assignment_complete": bool(fold_summary.get("assignment_complete", False)),
        "matched_initialization": bool(initialization_equal),
        "finite_training": bool(all_finite),
        "keeper_val_macro_locked": abs(
            float(keeper_val["macro_f1"]) - EXPECTED_KEEPER_VAL_MACRO_F1
        )
        <= 1e-6,
        "keeper_val_class1_locked": abs(
            _focus_f1(keeper_val) - EXPECTED_KEEPER_VAL_CLASS1_F1
        )
        <= 1e-6,
        "candidate_oof_macro_not_below_control": float(candidate_oof["macro_f1"])
        >= float(control_oof["macro_f1"]),
        "candidate_oof_class1_not_below_control": _focus_f1(candidate_oof)
        >= _focus_f1(control_oof),
        "candidate_val_macro_not_below_control": float(candidate_val["macro_f1"])
        >= float(control_val["macro_f1"]),
        "candidate_val_class1_not_below_control": _focus_f1(candidate_val)
        >= _focus_f1(control_val),
        "candidate_class1_fold_wins": int(fold_wins) >= 3,
        "candidate_val_macro_at_least_keeper": float(candidate_val["macro_f1"])
        >= EXPECTED_KEEPER_VAL_MACRO_F1,
        "candidate_val_class1_at_least_keeper": _focus_f1(candidate_val)
        >= EXPECTED_KEEPER_VAL_CLASS1_F1,
        "candidate_corrections_at_least_harms": int(keeper_transition["corrections"])
        >= int(keeper_transition["harms"]),
        "candidate_fn_rescue_at_least_tp_break": int(
            keeper_transition["focus_fn_rescue"]
        )
        >= int(keeper_transition["focus_tp_break"]),
        "candidate_fp_remove_at_least_create": int(
            keeper_transition["focus_fp_remove"]
        )
        >= int(keeper_transition["focus_fp_create"]),
        "oof_direction_auc": oof_auc is not None and float(oof_auc) >= 0.55,
        "validation_direction_auc": val_auc is not None and float(val_auc) >= 0.55,
        "direction_auc_transfer_gap": oof_auc is not None
        and val_auc is not None
        and abs(float(oof_auc) - float(val_auc)) <= 0.10,
        "foreground_part_mass_range": len(part_mass) == NUM_PARTS
        and all(0.10 <= value <= 0.75 for value in part_mass),
        "foreground_part_use_entropy": float(
            candidate_parts["foreground_part_use_entropy_normalized"]
        )
        >= 0.75,
        "background_outside_inside_gap": float(
            candidate_parts["background_outside_minus_inside"]
        )
        >= 0.10,
        "candidate_equivariance_absolute": float(
            candidate_parts["foreground_equivariance_cosine"]
        )
        >= 0.75,
        "candidate_equivariance_gain": float(
            candidate_parts["foreground_equivariance_cosine"]
        )
        >= float(control_parts["foreground_equivariance_cosine"]) + 0.02,
        "candidate_total_variation_not_above_control": float(
            candidate_parts["total_variation"]
        )
        <= float(control_parts["total_variation"]),
        "candidate_pixel_entropy_not_above_control": float(
            candidate_parts["pixel_entropy"]
        )
        <= float(control_parts["pixel_entropy"]),
        "test_data_unused": True,
        "raw_data_unmodified": True,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "mode": "full_locked_gate",
        "thresholds": {
            "keeper_val_macro_f1": EXPECTED_KEEPER_VAL_MACRO_F1,
            "keeper_val_class1_f1": EXPECTED_KEEPER_VAL_CLASS1_F1,
            "minimum_class1_fold_wins": 3,
            "minimum_direction_auc": 0.55,
            "maximum_direction_auc_transfer_gap": 0.10,
            "foreground_part_mass_range": [0.10, 0.75],
            "minimum_part_use_entropy": 0.75,
            "minimum_background_gap": 0.10,
            "minimum_equivariance": 0.75,
            "minimum_equivariance_gain": 0.02,
        },
        "observed_class1_fold_wins": int(fold_wins),
        "checks": checks,
        "failed_checks": failed,
        "all_automated_checks_passed": len(failed) == 0,
        "manual_visual_review_required": True,
        "image_smoke_permitted": False,
    }


def _write_markdown_summary(path: Path, summary: Mapping[str, object]) -> None:
    metrics = summary["metrics"]
    oof = metrics["train_oof"]
    val = metrics["validation"]
    gate = summary["gate"]
    lines = [
        "# PDiscoFormer-Style Semantic-Part Readiness",
        "",
        f"- Mode: `{summary['mode']}`.",
        f"- Keeper checkpoint SHA-256: `{summary['checkpoint_sha256']}`.",
        f"- Official code commit: `{OFFICIAL_PDISCO_COMMIT}`.",
        f"- Train/validation rows: `{summary['support']['train_rows']}/{summary['support']['val_rows']}`.",
        f"- Train source overlap with validation: `{summary['support']['train_val_source_overlap']}`.",
        "- Test used: `false`; raw data modified: `false`.",
        "",
        "## Direct metrics",
        "",
        f"- Keeper validation macro/class1: `{float(val['keeper']['macro_f1']):.6f}/{_focus_f1(val['keeper']):.6f}`.",
        f"- Control OOF macro/class1: `{float(oof['control']['macro_f1']):.6f}/{_focus_f1(oof['control']):.6f}`.",
        f"- Candidate OOF macro/class1: `{float(oof['candidate']['macro_f1']):.6f}/{_focus_f1(oof['candidate']):.6f}`.",
        f"- Control validation macro/class1: `{float(val['control']['macro_f1']):.6f}/{_focus_f1(val['control']):.6f}`.",
        f"- Candidate validation macro/class1: `{float(val['candidate']['macro_f1']):.6f}/{_focus_f1(val['candidate']):.6f}`.",
        "",
        "## Gate",
        "",
        f"- Automated pass: `{gate.get('all_automated_checks_passed')}`.",
        f"- Failed checks: `{', '.join(gate.get('failed_checks', [])) or 'none'}`.",
        "- Image smoke permitted: `false` until every automated gate and the manual visual review pass.",
        "",
        "The source-grouped head OOF result is over an in-sample keeper representation; it is not a true OOF keeper backbone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifact_manifest(output_dir: Path) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    for path in sorted(Path(output_dir).iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    retained_cache_summary = Path(output_dir) / "cache_summary_retained.json"
    manifest = {
        "files": records,
        "file_count": int(len(records)),
        "total_bytes": int(sum(int(row["bytes"]) for row in records)),
        "cache_payloads_are_hashed_in": (
            "cache_summary_retained.json"
            if retained_cache_summary.is_file()
            else "cache/cache_summary.json"
        ),
    }
    (Path(output_dir) / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _run_extraction(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    checkpoint_sha256: str,
    device: torch.device,
) -> Dict[str, object]:
    cache_dir = Path(args.output_dir) / "cache"
    if cache_dir.exists():
        raise FileExistsError(f"cache directory already exists: {cache_dir}")
    cache_dir.mkdir(parents=True)
    model = build_model_from_checkpoint(dict(checkpoint)).to(device=device, dtype=torch.float32)
    model.eval()
    split_summaries: Dict[str, object] = {}
    class_names: List[str] = []
    for split in ("train", "val"):
        dataset, split_names = _build_split_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            preflight=bool(args.preflight),
        )
        if class_names and list(split_names) != class_names:
            raise ValueError("train and validation class orders differ")
        class_names = list(split_names)
        split_summaries[split] = _extract_split_cache(
            model=model,
            dataset=dataset,
            cache_dir=cache_dir,
            split=split,
            class_names=class_names,
            device=device,
            batch_size=int(args.extract_batch_size),
            workers=int(args.workers),
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_sources = set(
        np.load(cache_dir / "train" / "source_stems.npy", allow_pickle=False).tolist()
    )
    val_sources = set(
        np.load(cache_dir / "val" / "source_stems.npy", allow_pickle=False).tolist()
    )
    summary = {
        "schema_version": 1,
        "mode": "preflight" if args.preflight else "full",
        "data_yaml": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": str(checkpoint_sha256),
        "official_pdisco_commit": OFFICIAL_PDISCO_COMMIT,
        "class_names": class_names,
        "splits": split_summaries,
        "train_val_source_overlap": int(len(train_sources.intersection(val_sources))),
        "test_data_used": False,
        "raw_data_modified": False,
    }
    (cache_dir / "cache_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_training(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    checkpoint_sha256: str,
    device: torch.device,
) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_summary = _verify_cache_files(cache_dir)
    expected_mode = "preflight" if args.preflight else "full"
    if str(cache_summary.get("mode")) != expected_mode:
        raise ValueError("cache mode does not match requested diagnostic mode")
    if str(cache_summary.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise ValueError("cache checkpoint hash differs from requested checkpoint")
    class_names = [str(value) for value in cache_summary.get("class_names", [])]
    if len(class_names) != 5:
        raise ValueError(f"semantic-part diagnostic requires five classes, got {class_names}")
    train_cache = SplitCache.load(
        cache_dir,
        "train",
        cache_summary["splits"]["train"],
    )
    val_cache = SplitCache.load(
        cache_dir,
        "val",
        cache_summary["splits"]["val"],
    )
    if train_cache.grid_size != val_cache.grid_size:
        raise ValueError("train and validation cache grids differ")
    folds = PREFLIGHT_FOLDS if args.preflight else FULL_FOLDS
    epochs = PREFLIGHT_EPOCHS if args.preflight else FULL_EPOCHS
    fold_splits, fold_summary = assign_source_grouped_folds(
        np.asarray(train_cache.labels),
        np.asarray(train_cache.source_stems),
        folds=folds,
        seed=SEED,
    )
    oof_control = np.full(
        (len(train_cache), len(class_names)),
        np.nan,
        dtype=np.float32,
    )
    oof_candidate = np.full_like(oof_control, np.nan)
    fold_results: List[Dict[str, object]] = []
    training_curves: Dict[str, object] = {"folds": [], "final": {}}
    initialization_hashes: List[Tuple[str, str]] = []
    all_finite = True
    started = time.perf_counter()

    for fold_index, (fit_indices, holdout_indices) in enumerate(fold_splits):
        initial_state, initial_hash, parameter_count = _head_initial_state(
            feature_dim=int(train_cache.tokens_original.shape[2]),
            num_classes=len(class_names),
            seed=SEED + fold_index * 100,
        )
        control_head, control_curve, control_telemetry = _train_head(
            cache=train_cache,
            train_indices=fit_indices,
            initial_state=copy.deepcopy(initial_state),
            semantic_losses_enabled=False,
            device=device,
            epochs=epochs,
            seed=SEED + fold_index * 100,
            num_classes=len(class_names),
        )
        candidate_head, candidate_curve, candidate_telemetry = _train_head(
            cache=train_cache,
            train_indices=fit_indices,
            initial_state=copy.deepcopy(initial_state),
            semantic_losses_enabled=True,
            device=device,
            epochs=epochs,
            seed=SEED + fold_index * 100,
            num_classes=len(class_names),
        )
        initialization_hashes.append(
            (
                str(control_telemetry["loaded_initial_state_sha256"]),
                str(candidate_telemetry["loaded_initial_state_sha256"]),
            )
        )
        if initialization_hashes[-1][0] != initial_hash:
            raise RuntimeError("control head did not load the locked initial state")
        if initialization_hashes[-1][1] != initial_hash:
            raise RuntimeError("candidate head did not load the locked initial state")
        control_probs = _predict_head(
            head=control_head,
            cache=train_cache,
            row_indices=holdout_indices,
            device=device,
        )
        candidate_probs = _predict_head(
            head=candidate_head,
            cache=train_cache,
            row_indices=holdout_indices,
            device=device,
        )
        oof_control[holdout_indices] = control_probs
        oof_candidate[holdout_indices] = candidate_probs
        control_metrics = classification_metrics(
            np.asarray(train_cache.labels[holdout_indices]),
            control_probs,
            class_names=class_names,
        )
        candidate_metrics = classification_metrics(
            np.asarray(train_cache.labels[holdout_indices]),
            candidate_probs,
            class_names=class_names,
        )
        fold_results.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(fit_indices.size),
                "holdout_rows": int(holdout_indices.size),
                "initial_state_sha256": initial_hash,
                "parameter_count": int(parameter_count),
                "control": control_metrics,
                "candidate": candidate_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "semantic_part_fold": int(fold_index),
                    "control_macro_f1": float(control_metrics["macro_f1"]),
                    "control_class1_f1": _focus_f1(control_metrics),
                    "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                    "candidate_class1_f1": _focus_f1(candidate_metrics),
                    "initial_state_sha256": initial_hash,
                }
            ),
            flush=True,
        )
        training_curves["folds"].append(
            {
                "fold": int(fold_index),
                "control_curve": control_curve,
                "candidate_curve": candidate_curve,
                "control_telemetry": control_telemetry,
                "candidate_telemetry": candidate_telemetry,
            }
        )
        all_finite = all_finite and int(control_telemetry["nonfinite_steps"]) == 0
        all_finite = all_finite and int(candidate_telemetry["nonfinite_steps"]) == 0
        del control_head, candidate_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if bool(np.isnan(oof_control).any()) or bool(np.isnan(oof_candidate).any()):
        raise RuntimeError("OOF predictions are incomplete")

    final_state, final_initial_hash, parameter_count = _head_initial_state(
        feature_dim=int(train_cache.tokens_original.shape[2]),
        num_classes=len(class_names),
        seed=SEED + 10_000,
    )
    all_train_indices = np.arange(len(train_cache), dtype=np.int64)
    control_head, control_curve, control_telemetry = _train_head(
        cache=train_cache,
        train_indices=all_train_indices,
        initial_state=copy.deepcopy(final_state),
        semantic_losses_enabled=False,
        device=device,
        epochs=epochs,
        seed=SEED + 10_000,
        num_classes=len(class_names),
    )
    candidate_head, candidate_curve, candidate_telemetry = _train_head(
        cache=train_cache,
        train_indices=all_train_indices,
        initial_state=copy.deepcopy(final_state),
        semantic_losses_enabled=True,
        device=device,
        epochs=epochs,
        seed=SEED + 10_000,
        num_classes=len(class_names),
    )
    initialization_hashes.append(
        (
            str(control_telemetry["loaded_initial_state_sha256"]),
            str(candidate_telemetry["loaded_initial_state_sha256"]),
        )
    )
    if initialization_hashes[-1][0] != final_initial_hash:
        raise RuntimeError("final control head did not load the locked initial state")
    if initialization_hashes[-1][1] != final_initial_hash:
        raise RuntimeError("final candidate head did not load the locked initial state")
    all_finite = all_finite and int(control_telemetry["nonfinite_steps"]) == 0
    all_finite = all_finite and int(candidate_telemetry["nonfinite_steps"]) == 0
    training_curves["final"] = {
        "initial_state_sha256": final_initial_hash,
        "parameter_count": int(parameter_count),
        "control_curve": control_curve,
        "candidate_curve": candidate_curve,
        "control_telemetry": control_telemetry,
        "candidate_telemetry": candidate_telemetry,
    }

    val_indices = np.arange(len(val_cache), dtype=np.int64)
    val_control = _predict_head(
        head=control_head,
        cache=val_cache,
        row_indices=val_indices,
        device=device,
    )
    val_candidate = _predict_head(
        head=candidate_head,
        cache=val_cache,
        row_indices=val_indices,
        device=device,
    )
    print(
        json.dumps(
            {
                "semantic_part_final_heads_complete": True,
                "train_rows": int(len(train_cache)),
                "validation_rows": int(len(val_cache)),
                "epochs": int(epochs),
            }
        ),
        flush=True,
    )
    metrics = {
        "train_oof": {
            "keeper_in_sample_reference": classification_metrics(
                np.asarray(train_cache.labels),
                np.asarray(train_cache.keeper_probabilities),
                class_names=class_names,
            ),
            "control": classification_metrics(
                np.asarray(train_cache.labels),
                oof_control,
                class_names=class_names,
            ),
            "candidate": classification_metrics(
                np.asarray(train_cache.labels),
                oof_candidate,
                class_names=class_names,
            ),
        },
        "validation": {
            "keeper": classification_metrics(
                np.asarray(val_cache.labels),
                np.asarray(val_cache.keeper_probabilities),
                class_names=class_names,
            ),
            "control": classification_metrics(
                np.asarray(val_cache.labels),
                val_control,
                class_names=class_names,
            ),
            "candidate": classification_metrics(
                np.asarray(val_cache.labels),
                val_candidate,
                class_names=class_names,
            ),
        },
    }
    transitions = {
        "candidate_vs_control_oof": transition_counts(
            np.asarray(train_cache.labels),
            oof_control,
            oof_candidate,
        ),
        "candidate_vs_keeper_train_reference": transition_counts(
            np.asarray(train_cache.labels),
            np.asarray(train_cache.keeper_probabilities),
            oof_candidate,
        ),
        "candidate_vs_control_val": transition_counts(
            np.asarray(val_cache.labels),
            val_control,
            val_candidate,
        ),
        "candidate_vs_keeper_val": transition_counts(
            np.asarray(val_cache.labels),
            np.asarray(val_cache.keeper_probabilities),
            val_candidate,
        ),
    }
    direction = {
        "train_oof_candidate_vs_control": focus_direction_auc(
            np.asarray(train_cache.labels),
            np.asarray(train_cache.keeper_probabilities),
            oof_control,
            oof_candidate,
        ),
        "val_candidate_vs_control": focus_direction_auc(
            np.asarray(val_cache.labels),
            np.asarray(val_cache.keeper_probabilities),
            val_control,
            val_candidate,
        ),
    }
    part_metrics = {
        "control": _evaluate_part_metrics(
            head=control_head,
            cache=val_cache,
            row_indices=val_indices,
            device=device,
        ),
        "candidate": _evaluate_part_metrics(
            head=candidate_head,
            cache=val_cache,
            row_indices=val_indices,
            device=device,
        ),
    }
    initialization_equal = all(left == right for left, right in initialization_hashes)
    gate = evaluate_gates(
        preflight=bool(args.preflight),
        checkpoint_sha256=checkpoint_sha256,
        train_cache=train_cache,
        val_cache=val_cache,
        fold_summary=fold_summary,
        oof_metrics=metrics["train_oof"],
        val_metrics=metrics["validation"],
        fold_results=fold_results,
        transitions=transitions,
        direction=direction,
        part_metrics=part_metrics,
        initialization_equal=initialization_equal,
        all_finite=all_finite,
    )
    _write_predictions_csv(
        output_dir / "train_oof_predictions.csv",
        cache=train_cache,
        row_indices=np.arange(len(train_cache), dtype=np.int64),
        control_probabilities=oof_control,
        candidate_probabilities=oof_candidate,
    )
    _write_predictions_csv(
        output_dir / "val_predictions.csv",
        cache=val_cache,
        row_indices=val_indices,
        control_probabilities=val_control,
        candidate_probabilities=val_candidate,
    )
    val_dataset, val_class_names = _build_split_dataset(
        data_yaml=Path(args.data),
        split="val",
        checkpoint=checkpoint,
        preflight=False,
    )
    if list(val_class_names) != class_names:
        raise ValueError("visual-review dataset class order differs from cache")
    visual_manifest = _write_contact_sheet(
        output_dir / "semantic_part_contact_sheet.png",
        dataset=val_dataset,
        cache=val_cache,
        control_head=control_head,
        candidate_head=candidate_head,
        candidate_probabilities=val_candidate,
        checkpoint=checkpoint,
        device=device,
    )
    training_curves_path = output_dir / "training_curves.json"
    training_curves_path.write_text(
        json.dumps(training_curves, indent=2),
        encoding="utf-8",
    )
    train_sources = set(np.asarray(train_cache.source_stems).astype(str).tolist())
    val_sources = set(np.asarray(val_cache.source_stems).astype(str).tolist())
    summary = {
        "schema_version": 1,
        "mode": "preflight" if args.preflight else "full",
        "method": "pdiscoformer_style_semantic_part_readiness",
        "official_pdisco_commit": OFFICIAL_PDISCO_COMMIT,
        "data_yaml": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": str(checkpoint_sha256),
        "class_names": class_names,
        "fixed_configuration": {
            "foreground_parts": NUM_PARTS,
            "background_parts": 1,
            "part_dropout": PART_DROPOUT,
            "gumbel_temperature": GUMBEL_TEMPERATURE,
            "head_batch_size": HEAD_BATCH_SIZE,
            "epochs": int(epochs),
            "folds": int(folds),
            "optimizer": "Adam",
            "learning_rate": HEAD_LR,
            "weight_decay": 0.0,
            "scheduler_step_size": SCHEDULER_STEP_SIZE,
            "scheduler_gamma": SCHEDULER_GAMMA,
            "loss_weights": {
                "classification": 1.0,
                "presence": 1.0,
                "equivariance": 1.0,
                "orthogonality": 1.0,
                "total_variation": 1.0,
                "enforced_background": 2.0,
                "pixel_entropy": 1.0,
            },
            "transform": "exact_clockwise_90_degree",
            "token_cache_dtype": "float16",
            "keeper_forward_precision": "fp32",
        },
        "support": {
            "train_rows": len(train_cache),
            "val_rows": len(val_cache),
            "train_sources": int(len(train_sources)),
            "val_sources": int(len(val_sources)),
            "train_val_source_overlap": int(len(train_sources.intersection(val_sources))),
        },
        "fold_summary": fold_summary,
        "fold_results": fold_results,
        "metrics": metrics,
        "transitions": transitions,
        "direction": direction,
        "part_metrics": part_metrics,
        "initialization_equal": bool(initialization_equal),
        "all_finite": bool(all_finite),
        "gate": gate,
        "visual_review": {
            "contact_sheet": "semantic_part_contact_sheet.png",
            "manifest": "semantic_part_contact_sheet.json",
            "rows": int(len(visual_manifest["rows"])),
            "manual_verdict": "pending",
        },
        "limitations": [
            "The keeper backbone was trained on the full train split; grouped head OOF is not a true OOF keeper representation.",
            "The diagnostic uses final post-pruning keeper tokens quantized to FP16 for cache storage.",
            "PDiscoFormer's reported recipe relies on a frozen pretrained DINOv2 representation; this keeper is trained from scratch.",
        ],
        "test_data_used": False,
        "raw_data_modified": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(output_dir / "README.md", summary)
    _write_artifact_manifest(output_dir)
    return summary


def run(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.extract_batch_size) <= 0:
        raise ValueError("extract-batch-size must be positive")
    if int(args.workers) < 0:
        raise ValueError("workers must be non-negative")
    if "test" in {part.casefold() for part in Path(args.data).parts}:
        raise ValueError("data YAML path must not point at a test directory")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_KEEPER_SHA256:
        raise ValueError(
            "readiness protocol is locked to the keeper checkpoint hash: "
            f"expected={EXPECTED_KEEPER_SHA256}, observed={checkpoint_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    _seed_everything(SEED)
    device = _resolve_device(str(args.device or ""))
    if args.stage in {"all", "extract"}:
        _run_extraction(
            args=args,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
        )
    if args.stage == "extract":
        return {
            "stage": "extract",
            "output_dir": str(output_dir),
            "test_data_used": False,
        }
    summary = _run_training(
        args=args,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )
    print(json.dumps(summary["gate"], indent=2))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
