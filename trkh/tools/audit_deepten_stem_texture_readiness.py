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
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import _write_artifact_manifest
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification
from trkh.tools.probe_interior_second_order_readiness import (
    build_fixed_orthogonal_projection,
)


SEED = 20260712
FOLDS = 5
GRID_SIZE = 12
PROJECTION_DIM = 32
CODEWORDS = 8
HIDDEN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RESIDUAL_SCALE = 0.12
RESIDUAL_L2_WEIGHT = 1e-3
INTERIOR_ERODE_RATIO = 0.12
FOCUS_CLASS_INDEX = 1
FOCUS_MILESTONE = 0.70
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
XAI_CASES = 12
MINIMUM_MASK_TOKENS = 9

LITERATURE = (
    "https://openaccess.thecvf.com/content_cvpr_2017/html/"
    "Zhang_Deep_TEN_Texture_CVPR_2017_paper.html",
    "https://openaccess.thecvf.com/content_cvpr_2018/html/"
    "Xue_Deep_Texture_Manifold_CVPR_2018_paper.html",
    "https://github.com/zhanghang1989/PyTorch-Encoding",
)
OFFICIAL_SOURCE_COMMIT = "ac748410dfc8d7d70a2ce7f5add08050af2fae20"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only Deep-TEN stem-texture readiness audit. It uses a "
            "frozen keeper and never reads test, edits raw data, or writes a model."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--codewords", type=int, default=CODEWORDS)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--extract-batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    parser.add_argument("--residual-l2-weight", type=float, default=RESIDUAL_L2_WEIGHT)
    parser.add_argument("--interior-erode-ratio", type=float, default=INTERIOR_ERODE_RATIO)
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
    if int(args.grid_size) < 4 or int(args.projection_dim) < 4:
        raise ValueError("Texture grid and projection dimensions are too small")
    if int(args.codewords) < 2 or int(args.hidden_dim) < 8:
        raise ValueError("Deep-TEN readout dimensions are invalid")
    if int(args.folds) < 3 or not 1 <= int(args.epochs) <= 30:
        raise ValueError("Source folds or epoch budget are invalid")
    if int(args.batch_size) <= 0 or int(args.extract_batch_size) <= 0:
        raise ValueError("Batch sizes must be positive")
    if int(args.workers) < 0 or float(args.learning_rate) <= 0.0:
        raise ValueError("Worker or optimizer settings are invalid")
    if float(args.weight_decay) < 0.0 or float(args.residual_l2_weight) < 0.0:
        raise ValueError("Regularization settings are invalid")
    if not 0.0 < float(args.residual_scale) <= 0.5:
        raise ValueError("Residual scale must be in (0, 0.5]")
    if not 0.0 <= float(args.interior_erode_ratio) < 0.4:
        raise ValueError("Interior erosion ratio must be in [0, 0.4)")
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


def build_interior_grid_mask(
    crop_bbox: Tensor,
    image_valid_mask: Optional[Tensor],
    *,
    grid_size: int,
    erode_ratio: float,
) -> Tensor:
    if crop_bbox.ndim != 2 or int(crop_bbox.size(1)) < 4:
        raise ValueError("crop_bbox must have shape [B,4+]")
    size = int(grid_size)
    bbox = crop_bbox[:, :4].float().clamp(0.0, 1.0)
    cx, cy, width, height = bbox.unbind(dim=1)
    shrink = max(0.0, 1.0 - 2.0 * float(erode_ratio))
    eroded_width = width * shrink
    eroded_height = height * shrink
    axis = (torch.arange(size, device=bbox.device, dtype=torch.float32) + 0.5) / float(size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x = x.view(1, size, size)
    y = y.view(1, size, size)
    interior = (
        (x - cx[:, None, None]).abs() <= 0.5 * eroded_width[:, None, None]
    ) & ((y - cy[:, None, None]).abs() <= 0.5 * eroded_height[:, None, None])
    full_bbox = (
        (x - cx[:, None, None]).abs() <= 0.5 * width[:, None, None]
    ) & ((y - cy[:, None, None]).abs() <= 0.5 * height[:, None, None])

    if image_valid_mask is None:
        valid = torch.ones_like(interior, dtype=torch.bool)
    else:
        valid = image_valid_mask
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.ndim != 4:
            raise ValueError("image_valid_mask must have shape [B,H,W] or [B,1,H,W]")
        valid = F.interpolate(valid.float(), size=(size, size), mode="nearest")[:, 0] > 0.5

    interior = interior & valid
    full_bbox = full_bbox & valid
    empty = ~interior.flatten(1).any(dim=1)
    if bool(empty.any().item()):
        interior = interior.clone()
        interior[empty] = full_bbox[empty]

    flat_interior = interior.flatten(1)
    flat_valid = valid.flatten(1)
    sparse = flat_interior.sum(dim=1) < MINIMUM_MASK_TOKENS
    if bool(sparse.any().item()):
        flat_interior = flat_interior.clone()
        squared_distance = (
            (x - cx[:, None, None]).square() + (y - cy[:, None, None]).square()
        ).flatten(1)
        for row in torch.nonzero(sparse, as_tuple=False).flatten().tolist():
            valid_indices = torch.nonzero(flat_valid[row], as_tuple=False).flatten()
            if int(valid_indices.numel()) == 0:
                valid_indices = torch.arange(size * size, device=bbox.device)
            flat_interior[row] = False
            nearest = valid_indices[
                torch.argsort(squared_distance[row, valid_indices])[:MINIMUM_MASK_TOKENS]
            ]
            flat_interior[row, nearest] = True
        interior = flat_interior.view_as(interior)
    return interior.flatten(1)


def masked_mean_std(descriptors: Tensor, mask: Tensor) -> Tensor:
    if descriptors.ndim != 3 or mask.ndim != 2:
        raise ValueError("descriptors/mask must be [B,N,D]/[B,N]")
    if tuple(descriptors.shape[:2]) != tuple(mask.shape):
        raise ValueError("descriptors and mask do not align")
    weights = mask.to(device=descriptors.device, dtype=descriptors.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (descriptors * weights).sum(dim=1) / count
    variance = ((descriptors - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
    return torch.cat((mean, variance.clamp_min(1e-8).sqrt()), dim=1)


class MaskedTextureEncoding(nn.Module):
    """Pure-PyTorch masked form of the official Deep-TEN Encoding layer."""

    def __init__(self, feature_dim: int, codewords: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.codeword_count = int(codewords)
        self.codewords = nn.Parameter(torch.empty(self.codeword_count, self.feature_dim))
        self.scale = nn.Parameter(torch.empty(self.codeword_count))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(float(self.codeword_count * self.feature_dim))
        nn.init.uniform_(self.codewords, -bound, bound)
        nn.init.uniform_(self.scale, -1.0, 0.0)

    def forward(
        self,
        descriptors: Tensor,
        mask: Tensor,
        *,
        return_assignments: bool = False,
    ):
        if descriptors.ndim != 3 or int(descriptors.size(2)) != self.feature_dim:
            raise ValueError("Deep-TEN descriptors must have shape [B,N,D]")
        if mask.ndim != 2 or tuple(mask.shape) != tuple(descriptors.shape[:2]):
            raise ValueError("Deep-TEN mask must align with descriptors")
        residuals = descriptors.unsqueeze(2) - self.codewords.view(
            1, 1, self.codeword_count, self.feature_dim
        )
        scaled_l2 = residuals.square().sum(dim=-1) * self.scale.view(1, 1, -1)
        assignments = scaled_l2.softmax(dim=2)
        valid = mask.to(device=descriptors.device, dtype=descriptors.dtype).unsqueeze(-1)
        assignments = assignments * valid
        encoding = (assignments.unsqueeze(-1) * residuals).sum(dim=1)
        if return_assignments:
            return encoding, assignments
        return encoding


class MeanStdResidualControl(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        input_dim = int(feature_dim) * 2
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, int(hidden_dim), bias=False)
        self.output = nn.Linear(int(hidden_dim), int(class_count), bias=False)
        nn.init.zeros_(self.output.weight)

    def residual_logits(self, descriptors: Tensor, mask: Tensor) -> Tensor:
        stats = masked_mean_std(descriptors, mask)
        return self.output(F.gelu(self.project(self.norm(stats))))


class DeepTENResidualHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        codewords: int,
        hidden_dim: int,
        class_count: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(int(feature_dim))
        self.encoding = MaskedTextureEncoding(int(feature_dim), int(codewords))
        encoded_dim = int(feature_dim) * int(codewords)
        self.project = nn.Linear(encoded_dim, int(hidden_dim), bias=False)
        self.output = nn.Linear(int(hidden_dim), int(class_count), bias=False)
        nn.init.zeros_(self.output.weight)

    def encoded_features(
        self,
        descriptors: Tensor,
        mask: Tensor,
        *,
        return_assignments: bool = False,
    ):
        normalized = self.input_norm(descriptors)
        result = self.encoding(
            normalized,
            mask,
            return_assignments=return_assignments,
        )
        if return_assignments:
            encoding, assignments = result
        else:
            encoding = result
            assignments = None
        flattened = F.normalize(encoding.flatten(1), p=2, dim=1, eps=1e-8)
        if return_assignments:
            return flattened, assignments
        return flattened

    def residual_logits(self, descriptors: Tensor, mask: Tensor) -> Tensor:
        encoded = self.encoded_features(descriptors, mask)
        return self.output(F.gelu(self.project(encoded)))


def residual_probabilities(
    model: nn.Module,
    descriptors: Tensor,
    mask: Tensor,
    base_probabilities: Tensor,
    *,
    residual_scale: float,
) -> Tuple[Tensor, Tensor]:
    residual = model.residual_logits(descriptors, mask)
    logits = torch.log(base_probabilities.clamp_min(1e-7)) + float(residual_scale) * residual
    return logits.softmax(dim=1), residual


def _focus(metrics: Mapping[str, object], focus_class_index: int) -> Mapping[str, object]:
    return metrics["per_class"][int(focus_class_index)]


def assess_deepten_stem_texture_readiness(
    *,
    train_rows: int,
    val_rows: int,
    train_val_source_overlap: int,
    fold_source_overlap: int,
    features_finite: bool,
    minimum_mask_tokens: int,
    codeword_utilization_entropy: float,
    scale_positive_fraction: float,
    keeper_oof: Mapping[str, object],
    control_oof: Mapping[str, object],
    candidate_oof: Mapping[str, object],
    keeper_val: Mapping[str, object],
    control_val: Mapping[str, object],
    candidate_val: Mapping[str, object],
    transitions_vs_keeper: Mapping[str, int],
    candidate_vs_control_oof_direction: Mapping[str, object],
    candidate_vs_control_val_direction: Mapping[str, object],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    keeper_focus = _focus(keeper_val, focus)
    control_oof_focus = _focus(control_oof, focus)
    candidate_oof_focus = _focus(candidate_oof, focus)
    control_val_focus = _focus(control_val, focus)
    candidate_val_focus = _focus(candidate_val, focus)
    oof_auc = candidate_vs_control_oof_direction.get("auc_fn_positive")
    val_auc = candidate_vs_control_val_direction.get("auc_fn_positive")
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_val_source_overlap": int(train_val_source_overlap),
        "fold_source_overlap": int(fold_source_overlap),
        "features_finite": bool(features_finite),
        "minimum_mask_tokens": int(minimum_mask_tokens),
        "codeword_utilization_entropy": float(codeword_utilization_entropy),
        "scale_positive_fraction": float(scale_positive_fraction),
        "keeper_oof_macro_f1": float(keeper_oof["macro_f1"]),
        "keeper_oof_focus_f1": float(_focus(keeper_oof, focus)["f1"]),
        "oof_macro_gain_vs_control": float(
            candidate_oof["macro_f1"] - control_oof["macro_f1"]
        ),
        "oof_focus_gain_vs_control": float(
            candidate_oof_focus["f1"] - control_oof_focus["f1"]
        ),
        "val_macro_gain_vs_control": float(
            candidate_val["macro_f1"] - control_val["macro_f1"]
        ),
        "val_focus_gain_vs_control": float(
            candidate_val_focus["f1"] - control_val_focus["f1"]
        ),
        "val_macro_gain_vs_keeper": float(
            candidate_val["macro_f1"] - keeper_val["macro_f1"]
        ),
        "val_focus_gain_vs_keeper": float(
            candidate_val_focus["f1"] - keeper_focus["f1"]
        ),
        "keeper_val_macro_f1": float(keeper_val["macro_f1"]),
        "keeper_val_focus_f1": float(keeper_focus["f1"]),
        "keeper_val_focus_recall": float(keeper_focus["recall"]),
        "control_val_macro_f1": float(control_val["macro_f1"]),
        "control_val_focus_f1": float(control_val_focus["f1"]),
        "candidate_val_macro_f1": float(candidate_val["macro_f1"]),
        "candidate_val_focus_f1": float(candidate_val_focus["f1"]),
        "candidate_val_focus_precision": float(candidate_val_focus["precision"]),
        "candidate_val_focus_recall": float(candidate_val_focus["recall"]),
        "transitions_vs_keeper": dict(transitions_vs_keeper),
        "oof_direction_auc": None if oof_auc is None else float(oof_auc),
        "val_direction_auc": None if val_auc is None else float(val_auc),
        "direction_auc_gap": (
            None
            if oof_auc is None or val_auc is None
            else abs(float(oof_auc) - float(val_auc))
        ),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "features_finite": bool(features_finite),
        "minimum_mask_tokens_ge_9": observed["minimum_mask_tokens"] >= 9,
        "codeword_utilization_entropy_ge_0p55": observed[
            "codeword_utilization_entropy"
        ]
        >= 0.55,
        "all_scales_nonpositive": observed["scale_positive_fraction"] == 0.0,
        "oof_macro_gain_vs_control_ge_0p002": observed[
            "oof_macro_gain_vs_control"
        ]
        >= 0.002,
        "oof_focus_gain_vs_control_ge_0p01": observed[
            "oof_focus_gain_vs_control"
        ]
        >= 0.01,
        "val_macro_gain_vs_control_ge_0p002": observed[
            "val_macro_gain_vs_control"
        ]
        >= 0.002,
        "val_focus_gain_vs_control_ge_0p01": observed[
            "val_focus_gain_vs_control"
        ]
        >= 0.01,
        "val_macro_preserves_keeper_within_0p001": observed[
            "val_macro_gain_vs_keeper"
        ]
        >= -0.001,
        "val_focus_reaches_0p70": observed["candidate_val_focus_f1"]
        >= FOCUS_MILESTONE,
        "val_focus_improves_keeper_by_0p01": observed["val_focus_gain_vs_keeper"]
        >= 0.01,
        "val_focus_recall_preserved_within_0p01": observed[
            "candidate_val_focus_recall"
        ]
        >= observed["keeper_val_focus_recall"] - 0.01,
        "candidate_corrections_ge_harms_vs_keeper": int(
            transitions_vs_keeper["corrections"]
        )
        >= int(transitions_vs_keeper["harms"]),
        "candidate_focus_fp_removed_ge_created_vs_keeper": int(
            transitions_vs_keeper["focus_false_positive_removed"]
        )
        >= int(transitions_vs_keeper["focus_false_positive_created"]),
        "candidate_focus_fn_rescued_ge_tp_broken_vs_keeper": int(
            transitions_vs_keeper["focus_false_negative_rescued"]
        )
        >= int(transitions_vs_keeper["focus_true_positive_broken"]),
        "oof_direction_auc_ge_0p60": oof_auc is not None and float(oof_auc) >= 0.60,
        "val_direction_auc_ge_0p60": val_auc is not None and float(val_auc) >= 0.60,
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


def _extract_split(
    *,
    model: nn.Module,
    dataset: Dataset,
    projection: Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    grid_size: int,
    erode_ratio: float,
    split: str,
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
    descriptor_batches: List[np.ndarray] = []
    mask_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    paths: List[str] = []
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in dataset_paths_fn()]
        if callable(dataset_paths_fn)
        else []
    )
    projection_device = projection.to(device=device, dtype=torch.float32)
    captured: List[Tensor] = []

    def capture_stem(_module, _inputs, output) -> None:
        captured.append(output)

    hook = model.stem.register_forward_hook(capture_stem)
    seen = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc=f"deepten-stem-{split}", dynamic_ncols=True)
            for images, labels, metadata in iterator:
                if not isinstance(metadata, Mapping):
                    raise ValueError("Stem texture extraction requires tensor metadata")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                crop_bbox = metadata.get("crop_bbox")
                if not torch.is_tensor(crop_bbox):
                    raise ValueError("crop_bbox metadata is required")
                crop_bbox = crop_bbox.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                bbox = metadata.get("bbox")
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
                captured.clear()
                with autocast_context(device, bool(amp)):
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
                stem = F.adaptive_avg_pool2d(
                    captured[0].float(),
                    output_size=(int(grid_size), int(grid_size)),
                )
                local = stem.permute(0, 2, 3, 1).contiguous()
                local = F.layer_norm(local, (int(local.size(-1)),))
                local = local @ projection_device
                local = F.layer_norm(local, (int(local.size(-1)),))
                local = local.flatten(1, 2)
                interior_mask = build_interior_grid_mask(
                    crop_bbox,
                    image_mask,
                    grid_size=int(grid_size),
                    erode_ratio=float(erode_ratio),
                )
                descriptor_batches.append(local.detach().half().cpu().numpy())
                mask_batches.append(interior_mask.detach().cpu().numpy())
                probability_batches.append(
                    logits.detach().float().softmax(dim=1).cpu().numpy()
                )
                label_batches.append(labels.detach().cpu().numpy())

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
                    sample_index_batches.append(
                        sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                    )
                else:
                    sample_index_batches.append(fallback_indices)
                seen += batch_count
    finally:
        hook.remove()

    descriptors = np.concatenate(descriptor_batches).astype(np.float16, copy=False)
    masks = np.concatenate(mask_batches).astype(bool, copy=False)
    labels_array = np.concatenate(label_batches).astype(np.int64, copy=False)
    probabilities = np.concatenate(probability_batches).astype(np.float32, copy=False)
    if not (
        len(paths)
        == labels_array.size
        == descriptors.shape[0]
        == masks.shape[0]
        == probabilities.shape[0]
    ):
        raise ValueError("Extracted stem-texture arrays have inconsistent rows")
    return {
        "descriptors": descriptors,
        "mask": masks,
        "probabilities": probabilities,
        "labels": labels_array,
        "sample_index": np.concatenate(sample_index_batches).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "source_stem": np.asarray(
            [_normalized_source_group(Path(path).stem) for path in paths],
            dtype=object,
        ),
        "seconds": float(time.perf_counter() - start),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        ),
    }


def _predict_model(
    model: nn.Module,
    descriptors: np.ndarray,
    mask: np.ndarray,
    base_probabilities: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    residual_scale: float,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch_indices = indices[start : start + int(batch_size)]
            x = torch.from_numpy(descriptors[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            valid = torch.from_numpy(mask[batch_indices]).to(device=device, dtype=torch.bool)
            base = torch.from_numpy(base_probabilities[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            probabilities, _ = residual_probabilities(
                model,
                x,
                valid,
                base,
                residual_scale=float(residual_scale),
            )
            outputs.append(probabilities.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def _train_model_pair(
    descriptors: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    fit_indices: np.ndarray,
    eval_indices: np.ndarray,
    *,
    feature_dim: int,
    codewords: int,
    hidden_dim: int,
    class_count: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    residual_scale: float,
    residual_l2_weight: float,
    seed: int,
    fold_label: str,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    control = MeanStdResidualControl(feature_dim, hidden_dim, class_count).to(device)
    torch.manual_seed(int(seed) + 1)
    candidate = DeepTENResidualHead(feature_dim, codewords, hidden_dim, class_count).to(device)
    models = {"control": control, "candidate": candidate}
    optimizers = {
        role: torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        for role, model in models.items()
    }
    curves: List[Dict[str, object]] = []
    for epoch in range(int(epochs)):
        order = np.random.default_rng(int(seed) + epoch).permutation(fit_indices)
        totals = {role: 0.0 for role in models}
        residual_totals = {role: 0.0 for role in models}
        rows = 0
        for start in range(0, len(order), max(1, int(batch_size))):
            batch_indices = order[start : start + int(batch_size)]
            x = torch.from_numpy(descriptors[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            valid = torch.from_numpy(mask[batch_indices]).to(device=device, dtype=torch.bool)
            y = torch.from_numpy(labels[batch_indices]).to(device=device, dtype=torch.long)
            base = torch.from_numpy(base_probabilities[batch_indices]).to(
                device=device,
                dtype=torch.float32,
            )
            for role, model in models.items():
                model.train()
                optimizer = optimizers[role]
                optimizer.zero_grad(set_to_none=True)
                probabilities, residual = residual_probabilities(
                    model,
                    x,
                    valid,
                    base,
                    residual_scale=float(residual_scale),
                )
                ce = F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), y)
                regularizer = residual.square().mean()
                loss = ce + float(residual_l2_weight) * regularizer
                loss.backward()
                optimizer.step()
                totals[role] += float(ce.detach().cpu()) * int(y.numel())
                residual_totals[role] += float(
                    residual.detach().abs().mean().cpu()
                ) * int(y.numel())
            rows += int(y.numel())
        for role in models:
            curves.append(
                {
                    "fold": fold_label,
                    "role": role,
                    "epoch": int(epoch + 1),
                    "ce": float(totals[role] / max(1, rows)),
                    "residual_abs_mean": float(
                        residual_totals[role] / max(1, rows)
                    ),
                }
            )
    eval_probabilities = {
        role: _predict_model(
            model,
            descriptors,
            mask,
            base_probabilities,
            np.asarray(eval_indices, dtype=np.int64),
            device=device,
            batch_size=int(batch_size),
            residual_scale=float(residual_scale),
        )
        for role, model in models.items()
    }
    return {
        "models": models,
        "probabilities": eval_probabilities,
        "curves": curves,
        "parameter_counts": {
            role: int(sum(parameter.numel() for parameter in model.parameters()))
            for role, model in models.items()
        },
    }


def fit_source_grouped_readouts(
    train_payload: Mapping[str, object],
    val_payload: Mapping[str, object],
    *,
    device: torch.device,
    folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    residual_scale: float,
    residual_l2_weight: float,
    codewords: int,
    hidden_dim: int,
    seed: int,
    focus_class_index: int,
) -> Dict[str, object]:
    train_descriptors = np.asarray(train_payload["descriptors"], dtype=np.float16)
    train_mask = np.asarray(train_payload["mask"], dtype=bool)
    train_labels = np.asarray(train_payload["labels"], dtype=np.int64)
    train_base = np.asarray(train_payload["probabilities"], dtype=np.float32)
    train_groups = np.asarray(train_payload["source_stem"], dtype=object)
    val_descriptors = np.asarray(val_payload["descriptors"], dtype=np.float16)
    val_mask = np.asarray(val_payload["mask"], dtype=bool)
    val_base = np.asarray(val_payload["probabilities"], dtype=np.float32)
    feature_dim = int(train_descriptors.shape[2])
    class_count = int(train_base.shape[1])
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    split_indices = list(
        splitter.split(train_descriptors, train_labels, groups=train_groups)
    )
    oof = {
        "control": np.zeros_like(train_base, dtype=np.float32),
        "candidate": np.zeros_like(train_base, dtype=np.float32),
    }
    fold_assignment = np.full(len(train_labels), -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    maximum_source_overlap = 0
    parameter_counts: Dict[str, int] = {}
    for fold_index, (fit_indices, hold_indices) in enumerate(split_indices):
        fit_sources = set(train_groups[fit_indices].tolist())
        hold_sources = set(train_groups[hold_indices].tolist())
        overlap = len(fit_sources.intersection(hold_sources))
        maximum_source_overlap = max(maximum_source_overlap, overlap)
        result = _train_model_pair(
            train_descriptors,
            train_mask,
            train_labels,
            train_base,
            np.asarray(fit_indices, dtype=np.int64),
            np.asarray(hold_indices, dtype=np.int64),
            feature_dim=feature_dim,
            codewords=int(codewords),
            hidden_dim=int(hidden_dim),
            class_count=class_count,
            device=device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            residual_scale=float(residual_scale),
            residual_l2_weight=float(residual_l2_weight),
            seed=int(seed) + fold_index * 100,
            fold_label=str(fold_index),
        )
        parameter_counts = dict(result["parameter_counts"])
        for role in oof:
            oof[role][hold_indices] = result["probabilities"][role]
        fold_assignment[hold_indices] = int(fold_index)
        curves.extend(result["curves"])
        control_metrics = _classification_metrics(
            train_labels[hold_indices],
            oof["control"][hold_indices],
        )
        candidate_metrics = _classification_metrics(
            train_labels[hold_indices],
            oof["candidate"][hold_indices],
        )
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "control_focus_f1": float(
                    _focus(control_metrics, focus_class_index)["f1"]
                ),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "candidate_focus_f1": float(
                    _focus(candidate_metrics, focus_class_index)["f1"]
                ),
                "macro_gain": float(
                    candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
                ),
                "focus_gain": float(
                    _focus(candidate_metrics, focus_class_index)["f1"]
                    - _focus(control_metrics, focus_class_index)["f1"]
                ),
            }
        )
        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF fold assignment is incomplete")

    combined_descriptors = np.concatenate((train_descriptors, val_descriptors), axis=0)
    combined_mask = np.concatenate((train_mask, val_mask), axis=0)
    combined_base = np.concatenate((train_base, val_base), axis=0)
    fit_indices = np.arange(len(train_labels), dtype=np.int64)
    val_indices = np.arange(
        len(train_labels),
        len(train_labels) + len(val_descriptors),
        dtype=np.int64,
    )
    full = _train_model_pair(
        combined_descriptors,
        combined_mask,
        np.concatenate((train_labels, np.asarray(val_payload["labels"], dtype=np.int64))),
        combined_base,
        fit_indices,
        val_indices,
        feature_dim=feature_dim,
        codewords=int(codewords),
        hidden_dim=int(hidden_dim),
        class_count=class_count,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        residual_scale=float(residual_scale),
        residual_l2_weight=float(residual_l2_weight),
        seed=int(seed) + 10000,
        fold_label="full",
    )
    curves.extend(full["curves"])
    return {
        "oof_probabilities": oof,
        "val_probabilities": full["probabilities"],
        "oof_metrics": {
            role: _classification_metrics(train_labels, probabilities)
            for role, probabilities in oof.items()
        },
        "val_metrics": {
            role: _classification_metrics(
                np.asarray(val_payload["labels"], dtype=np.int64),
                probabilities,
            )
            for role, probabilities in full["probabilities"].items()
        },
        "fold_assignment": fold_assignment,
        "fold_rows": fold_rows,
        "curves": curves,
        "maximum_source_overlap": int(maximum_source_overlap),
        "parameter_counts": parameter_counts,
        "final_models": full["models"],
    }


def _assignment_telemetry(
    model: DeepTENResidualHead,
    descriptors: np.ndarray,
    mask: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    utilization = np.zeros(model.encoding.codeword_count, dtype=np.float64)
    assignment_entropy_sum = 0.0
    valid_count = 0
    encoded_norms: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        indices = np.arange(len(descriptors), dtype=np.int64)
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch = indices[start : start + int(batch_size)]
            x = torch.from_numpy(descriptors[batch]).to(device=device, dtype=torch.float32)
            valid = torch.from_numpy(mask[batch]).to(device=device, dtype=torch.bool)
            encoded, assignments = model.encoded_features(
                x,
                valid,
                return_assignments=True,
            )
            assignment_np = assignments.detach().cpu().numpy().astype(np.float64)
            utilization += assignment_np.sum(axis=(0, 1))
            valid_np = valid.detach().cpu().numpy()
            entropy = -np.sum(
                assignment_np * np.log(np.clip(assignment_np, 1e-12, 1.0)),
                axis=2,
            )
            assignment_entropy_sum += float(entropy[valid_np].sum())
            valid_count += int(valid_np.sum())
            encoded_norms.append(encoded.norm(dim=1).cpu().numpy())
    utilization = utilization / max(float(utilization.sum()), 1e-12)
    utilization_entropy = float(
        -np.sum(utilization * np.log(np.clip(utilization, 1e-12, 1.0)))
        / math.log(max(2, utilization.size))
    )
    descriptor_entropy = float(
        assignment_entropy_sum
        / max(1, valid_count)
        / math.log(max(2, utilization.size))
    )
    codewords = model.encoding.codewords.detach().float()
    normalized = F.normalize(codewords, dim=1)
    cosine = normalized @ normalized.t()
    cosine.fill_diagonal_(-1.0)
    scales = model.encoding.scale.detach().cpu().numpy().astype(np.float64)
    return {
        "utilization": utilization.tolist(),
        "utilization_entropy_normalized": utilization_entropy,
        "descriptor_assignment_entropy_normalized": descriptor_entropy,
        "scale_values": scales.tolist(),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_positive_fraction": float(np.mean(scales > 0.0)),
        "maximum_codeword_cosine": float(cosine.max().cpu()),
        "encoded_feature_norm_mean": float(np.concatenate(encoded_norms).mean()),
    }


def select_xai_cases(
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class_index: int = FOCUS_CLASS_INDEX,
    maximum_cases: int = XAI_CASES,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    keeper = np.asarray(keeper_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    focus = int(focus_class_index)
    categories = (
        ("focus_fn_rescued", (labels == focus) & (keeper != focus) & (candidate == focus)),
        ("focus_tp_broken", (labels == focus) & (keeper == focus) & (candidate != focus)),
        ("focus_fp_removed", (labels != focus) & (keeper == focus) & (candidate != focus)),
        ("focus_fp_created", (labels != focus) & (keeper != focus) & (candidate == focus)),
    )
    selected: List[Dict[str, object]] = []
    used: set[int] = set()
    quota = max(1, int(maximum_cases) // len(categories))
    for category, category_mask in categories:
        for index in np.flatnonzero(category_mask)[:quota].tolist():
            selected.append({"row_index": int(index), "category": category})
            used.add(int(index))
    for index in np.flatnonzero(keeper != candidate).tolist():
        if len(selected) >= int(maximum_cases):
            break
        if int(index) not in used:
            selected.append({"row_index": int(index), "category": "other_changed"})
            used.add(int(index))
    for index in range(len(labels)):
        if len(selected) >= int(maximum_cases):
            break
        if int(index) not in used:
            selected.append({"row_index": int(index), "category": "stable_reference"})
            used.add(int(index))
    return selected[: int(maximum_cases)]


def _residual_saliency(
    model: nn.Module,
    descriptors: np.ndarray,
    mask: np.ndarray,
    indices: Sequence[int],
    *,
    device: torch.device,
    focus_class_index: int,
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(np.asarray(descriptors[list(indices)], dtype=np.float16)).to(
        device=device,
        dtype=torch.float32,
    )
    x.requires_grad_(True)
    valid = torch.from_numpy(np.asarray(mask[list(indices)], dtype=bool)).to(
        device=device,
        dtype=torch.bool,
    )
    residual = model.residual_logits(x, valid)
    gradient = torch.autograd.grad(residual[:, int(focus_class_index)].sum(), x)[0]
    saliency = (gradient * x).sum(dim=2)
    saliency = saliency.masked_fill(~valid, 0.0)
    return saliency.detach().cpu().numpy().astype(np.float32, copy=False)


def _saliency_stats(maps: np.ndarray, *, grid_size: int) -> Dict[str, object]:
    values = np.asarray(maps, dtype=np.float64).reshape(-1, int(grid_size), int(grid_size))
    positive = np.maximum(values, 0.0)
    total = positive.sum(axis=(1, 2))
    center_start = max(0, int(grid_size) // 2 - int(grid_size) // 6)
    center_end = min(int(grid_size), int(grid_size) - center_start)
    center = positive[:, center_start:center_end, center_start:center_end].sum(axis=(1, 2))
    border_width = max(1, int(grid_size) // 6)
    border_mask = np.zeros((int(grid_size), int(grid_size)), dtype=bool)
    border_mask[:border_width] = True
    border_mask[-border_width:] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border = positive[:, border_mask].sum(axis=1)
    distribution = positive / np.maximum(total[:, None, None], 1e-12)
    entropy = -np.sum(
        distribution * np.log(np.clip(distribution, 1e-12, 1.0)),
        axis=(1, 2),
    ) / math.log(max(2, int(grid_size) * int(grid_size)))
    return {
        "cases": int(values.shape[0]),
        "positive_center_mass_mean": float(
            np.mean(center / np.maximum(total, 1e-12))
        ),
        "positive_border_mass_mean": float(
            np.mean(border / np.maximum(total, 1e-12))
        ),
        "positive_entropy_mean": float(entropy.mean()),
        "positive_nonzero_cases": int(np.sum(total > 1e-12)),
    }


def _signed_overlay(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    values = np.asarray(heat, dtype=np.float32)
    maximum = max(float(np.max(np.abs(values))), 1e-8)
    normalized = np.clip(values / maximum, -1.0, 1.0)
    positive = np.maximum(normalized, 0.0)
    negative = np.maximum(-normalized, 0.0)
    color = np.zeros((*normalized.shape, 3), dtype=np.float32)
    color[..., 0] = 255.0 * positive
    color[..., 2] = 255.0 * negative
    color[..., 1] = 70.0 * (positive + negative)
    resampling = getattr(Image, "Resampling", Image)
    color_large = np.asarray(
        Image.fromarray(color.astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    strength = np.asarray(
        Image.fromarray((np.abs(normalized) * 255.0).astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )[..., None] / 255.0
    alpha = 0.55 * strength
    return np.clip(
        image.astype(np.float32) * (1.0 - alpha) + color_large * alpha,
        0,
        255,
    ).astype(np.uint8)


def _dataset_rgb(
    dataset: Dataset,
    row_index: int,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[np.ndarray, int]:
    item = dataset[int(row_index)]
    image = item[0]
    target = int(item[1])
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    rgb = (
        (image.float() * std_tensor + mean_tensor)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return (rgb * 255.0).round().astype(np.uint8), target


def _write_xai_contact_sheet(
    output_dir: Path,
    *,
    dataset: Dataset,
    cases: Sequence[Mapping[str, object]],
    control_maps: np.ndarray,
    candidate_maps: np.ndarray,
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    paths: np.ndarray,
    sample_indices: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    grid_size: int,
) -> Dict[str, object]:
    tile = 168
    label_height = 34
    canvas = Image.new(
        "RGB",
        (tile * 3, len(cases) * (tile + label_height)),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image)
    rows: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        row_index = int(case["row_index"])
        rgb, dataset_target = _dataset_rgb(dataset, row_index, mean=mean, std=std)
        if dataset_target != int(labels[row_index]):
            raise ValueError("XAI dataset order differs from extracted labels")
        images = (
            rgb,
            _signed_overlay(rgb, control_maps[case_index].reshape(grid_size, grid_size)),
            _signed_overlay(rgb, candidate_maps[case_index].reshape(grid_size, grid_size)),
        )
        y = case_index * (tile + label_height)
        for column, image in enumerate(images):
            canvas.paste(
                Image.fromarray(image).resize((tile, tile), resampling.BILINEAR),
                (column * tile, y),
            )
        predictions = (
            int(keeper_probabilities[row_index].argmax()),
            int(control_probabilities[row_index].argmax()),
            int(candidate_probabilities[row_index].argmax()),
        )
        draw.text(
            (4, y + tile + 2),
            (
                f"{case['category']} y={int(labels[row_index])} "
                f"k/m/d={predictions[0]}/{predictions[1]}/{predictions[2]}"
            ),
            fill=(20, 20, 20),
        )
        rows.append(
            {
                "preview_row": int(case_index),
                "row_index": row_index,
                "sample_index": int(sample_indices[row_index]),
                "category": str(case["category"]),
                "image_path": str(paths[row_index]),
                "target_index": int(labels[row_index]),
                "keeper_prediction_index": predictions[0],
                "control_prediction_index": predictions[1],
                "candidate_prediction_index": predictions[2],
                "keeper_focus_probability": float(
                    keeper_probabilities[row_index, FOCUS_CLASS_INDEX]
                ),
                "control_focus_probability": float(
                    control_probabilities[row_index, FOCUS_CLASS_INDEX]
                ),
                "candidate_focus_probability": float(
                    candidate_probabilities[row_index, FOCUS_CLASS_INDEX]
                ),
                "control_residual_saliency": control_maps[case_index].reshape(
                    grid_size, grid_size
                ).tolist(),
                "candidate_residual_saliency": candidate_maps[case_index].reshape(
                    grid_size, grid_size
                ).tolist(),
            }
        )
    canvas.save(output_dir / "xai_deepten_class1_residual.png")
    payload = {
        "method": "gradient_times_projected_stem_descriptor_on_class1_residual",
        "columns": ["keeper_object_crop", "mean_std_control", "deepten_candidate"],
        "red": "positive class1 residual evidence",
        "blue": "negative class1 residual evidence",
        "selection_uses_labels_for_audit_only": True,
        "control_stats": _saliency_stats(control_maps, grid_size=grid_size),
        "candidate_stats": _saliency_stats(candidate_maps, grid_size=grid_size),
        "rows": rows,
    }
    (output_dir / "xai_deepten_class1_residual.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def _prediction_rows(
    *,
    split: str,
    payload: Mapping[str, object],
    fold_assignment: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> List[Dict[str, object]]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    rows: List[Dict[str, object]] = []
    for row_index in range(len(labels)):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(payload["sample_index"][row_index]),
            "fold": int(fold_assignment[row_index]),
            "source_stem": str(payload["source_stem"][row_index]),
            "image_path": str(payload["paths"][row_index]),
            "target_index": int(labels[row_index]),
            "keeper_prediction_index": int(keeper_probabilities[row_index].argmax()),
            "control_prediction_index": int(control_probabilities[row_index].argmax()),
            "candidate_prediction_index": int(candidate_probabilities[row_index].argmax()),
        }
        for class_index in range(int(keeper_probabilities.shape[1])):
            row[f"keeper_prob_{class_index}"] = float(
                keeper_probabilities[row_index, class_index]
            )
            row[f"control_prob_{class_index}"] = float(
                control_probabilities[row_index, class_index]
            )
            row[f"candidate_prob_{class_index}"] = float(
                candidate_probabilities[row_index, class_index]
            )
        rows.append(row)
    return rows


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_protocol(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint))
    if not hasattr(model, "stem") or not hasattr(model, "forward_features"):
        raise ValueError("Deep-TEN audit requires the TRKH CNN stem")
    stem_channels = int(getattr(model.stem, "out_channels", 0))
    if stem_channels <= 0:
        raise ValueError("Could not resolve CNN stem output channels")
    if int(args.projection_dim) > stem_channels:
        raise ValueError("Projection dimension exceeds stem channels")
    projection = build_fixed_orthogonal_projection(
        stem_channels,
        int(args.projection_dim),
        seed=SEED,
    )
    device = _resolve_device(str(args.device))
    model.to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    class_names: List[str] = []
    datasets: Dict[str, Dataset] = {}
    payloads: Dict[str, Dict[str, object]] = {}
    for split, maximum in (
        ("train", int(args.max_train_samples)),
        ("val", int(args.max_val_samples)),
    ):
        dataset, split_class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=maximum,
        )
        if class_names and list(split_class_names) != class_names:
            raise ValueError("Train/validation class order differs")
        class_names = list(split_class_names)
        datasets[split] = dataset
        payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            projection=projection,
            device=device,
            batch_size=int(args.extract_batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            grid_size=int(args.grid_size),
            erode_ratio=float(args.interior_erode_ratio),
            split=split,
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
    readouts = fit_source_grouped_readouts(
        payloads["train"],
        payloads["val"],
        device=device,
        folds=int(args.folds),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        residual_scale=float(args.residual_scale),
        residual_l2_weight=float(args.residual_l2_weight),
        codewords=int(args.codewords),
        hidden_dim=int(args.hidden_dim),
        seed=SEED,
        focus_class_index=int(args.focus_class_index),
    )
    train_labels = np.asarray(payloads["train"]["labels"], dtype=np.int64)
    val_labels = np.asarray(payloads["val"]["labels"], dtype=np.int64)
    keeper_oof_metrics = _classification_metrics(
        train_labels,
        np.asarray(payloads["train"]["probabilities"], dtype=np.float32),
    )
    keeper_val_metrics = _classification_metrics(
        val_labels,
        np.asarray(payloads["val"]["probabilities"], dtype=np.float32),
    )
    transitions = {
        "oof_candidate_vs_control": _transition_stats(
            train_labels,
            readouts["oof_probabilities"]["control"],
            readouts["oof_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _transition_stats(
            val_labels,
            readouts["val_probabilities"]["control"],
            readouts["val_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_control_vs_keeper": _transition_stats(
            val_labels,
            payloads["val"]["probabilities"],
            readouts["val_probabilities"]["control"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_keeper": _transition_stats(
            val_labels,
            payloads["val"]["probabilities"],
            readouts["val_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    directions = {
        "oof_candidate_vs_control": _direction_auc(
            train_labels,
            readouts["oof_probabilities"]["control"],
            readouts["oof_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _direction_auc(
            val_labels,
            readouts["val_probabilities"]["control"],
            readouts["val_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_keeper": _direction_auc(
            val_labels,
            payloads["val"]["probabilities"],
            readouts["val_probabilities"]["candidate"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    candidate_model = readouts["final_models"]["candidate"]
    control_model = readouts["final_models"]["control"]
    telemetry = _assignment_telemetry(
        candidate_model,
        np.asarray(payloads["val"]["descriptors"], dtype=np.float16),
        np.asarray(payloads["val"]["mask"], dtype=bool),
        device=device,
        batch_size=int(args.batch_size),
    )
    minimum_mask_tokens = min(
        int(np.asarray(payloads[split]["mask"], dtype=bool).sum(axis=1).min())
        for split in ("train", "val")
    )
    features_finite = all(
        np.isfinite(np.asarray(payloads[split]["descriptors"], dtype=np.float32)).all()
        and np.isfinite(np.asarray(payloads[split]["probabilities"], dtype=np.float32)).all()
        for split in ("train", "val")
    )
    gate = assess_deepten_stem_texture_readiness(
        train_rows=len(train_labels),
        val_rows=len(val_labels),
        train_val_source_overlap=train_val_source_overlap,
        fold_source_overlap=int(readouts["maximum_source_overlap"]),
        features_finite=features_finite,
        minimum_mask_tokens=minimum_mask_tokens,
        codeword_utilization_entropy=float(
            telemetry["utilization_entropy_normalized"]
        ),
        scale_positive_fraction=float(telemetry["scale_positive_fraction"]),
        keeper_oof=keeper_oof_metrics,
        control_oof=readouts["oof_metrics"]["control"],
        candidate_oof=readouts["oof_metrics"]["candidate"],
        keeper_val=keeper_val_metrics,
        control_val=readouts["val_metrics"]["control"],
        candidate_val=readouts["val_metrics"]["candidate"],
        transitions_vs_keeper=transitions["val_candidate_vs_keeper"],
        candidate_vs_control_oof_direction=directions["oof_candidate_vs_control"],
        candidate_vs_control_val_direction=directions["val_candidate_vs_control"],
        focus_class_index=int(args.focus_class_index),
        test_split_used=False,
    )

    cases = select_xai_cases(
        val_labels,
        payloads["val"]["probabilities"],
        readouts["val_probabilities"]["candidate"],
        focus_class_index=int(args.focus_class_index),
        maximum_cases=int(args.xai_cases),
    )
    case_indices = [int(case["row_index"]) for case in cases]
    control_maps = _residual_saliency(
        control_model,
        payloads["val"]["descriptors"],
        payloads["val"]["mask"],
        case_indices,
        device=device,
        focus_class_index=int(args.focus_class_index),
    )
    candidate_maps = _residual_saliency(
        candidate_model,
        payloads["val"]["descriptors"],
        payloads["val"]["mask"],
        case_indices,
        device=device,
        focus_class_index=int(args.focus_class_index),
    )
    xai = _write_xai_contact_sheet(
        output_dir,
        dataset=datasets["val"],
        cases=cases,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
        labels=val_labels,
        keeper_probabilities=np.asarray(payloads["val"]["probabilities"], dtype=np.float32),
        control_probabilities=readouts["val_probabilities"]["control"],
        candidate_probabilities=readouts["val_probabilities"]["candidate"],
        paths=np.asarray(payloads["val"]["paths"], dtype=object),
        sample_indices=np.asarray(payloads["val"]["sample_index"], dtype=np.int64),
        mean=mean,
        std=std,
        grid_size=int(args.grid_size),
    )

    _write_csv(output_dir / "fold_metrics.csv", readouts["fold_rows"])
    _write_csv(output_dir / "training_curves.csv", readouts["curves"])
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            payload=payloads["train"],
            fold_assignment=readouts["fold_assignment"],
            keeper_probabilities=np.asarray(
                payloads["train"]["probabilities"], dtype=np.float32
            ),
            control_probabilities=readouts["oof_probabilities"]["control"],
            candidate_probabilities=readouts["oof_probabilities"]["candidate"],
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            payload=payloads["val"],
            fold_assignment=np.full(len(val_labels), -1, dtype=np.int64),
            keeper_probabilities=np.asarray(
                payloads["val"]["probabilities"], dtype=np.float32
            ),
            control_probabilities=readouts["val_probabilities"]["control"],
            candidate_probabilities=readouts["val_probabilities"]["candidate"],
        ),
    )
    full_support = len(train_labels) == EXPECTED_TRAIN_ROWS and len(val_labels) == EXPECTED_VAL_ROWS
    protocol = {
        "method": "deepten_frozen_keeper_stem_interior_texture_readiness",
        "scope": "Frozen-keeper representation diagnostic before any core-model branch.",
        "literature": list(LITERATURE),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split_usage": {"train": True, "val": True, "test": False},
        "support_mode": "full_decision" if full_support else "prefix_preflight_only",
        "train_rows": int(len(train_labels)),
        "val_rows": int(len(val_labels)),
        "train_source_groups": int(np.unique(train_groups).size),
        "val_source_groups": int(np.unique(val_groups).size),
        "train_val_source_overlap": int(train_val_source_overlap),
        "stem_channels": stem_channels,
        "grid_size": int(args.grid_size),
        "projection": "fixed_gaussian_qr_orthogonal_after_per_location_layernorm",
        "projection_dim": int(args.projection_dim),
        "projection_seed": SEED,
        "interior_mask": "crop_bbox_eroded_and_intersected_with_image_valid_mask",
        "interior_erode_ratio": float(args.interior_erode_ratio),
        "codewords": int(args.codewords),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "folds": int(args.folds),
        "natural_frequency_batches": True,
        "class_weight": None,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "residual_scale": float(args.residual_scale),
        "residual_l2_weight": float(args.residual_l2_weight),
        "control": "masked_mean_plus_std_zero_init_keeper_log_probability_residual",
        "candidate": (
            "official_formula_masked_deepten_encoding_"
            "zero_init_keeper_log_probability_residual"
        ),
        "matched_source_folds_and_batch_order": True,
        "validation_hyperparameter_selection": False,
        "underlying_keeper_train_predictions_are_in_sample": True,
        "grouped_oof_applies_to_residual_heads_only": True,
        "runtime": {
            "device": str(device),
            "extract_batch_size": int(args.extract_batch_size),
            "readout_batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "amp": bool(args.amp),
            "torch_threads": int(args.torch_threads),
        },
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
        "trainable_manifest_written": False,
        "feature_cache_written": False,
    }
    summary = {
        "protocol": protocol,
        "class_names": class_names,
        "extraction": {
            split: {
                "rows": int(len(payloads[split]["labels"])),
                "source_groups": int(np.unique(payloads[split]["source_stem"]).size),
                "descriptor_shape": list(payloads[split]["descriptors"].shape),
                "mask_tokens": {
                    "min": int(np.asarray(payloads[split]["mask"]).sum(axis=1).min()),
                    "mean": float(np.asarray(payloads[split]["mask"]).sum(axis=1).mean()),
                    "max": int(np.asarray(payloads[split]["mask"]).sum(axis=1).max()),
                },
                "seconds": float(payloads[split]["seconds"]),
                "peak_cuda_memory_mib": float(payloads[split]["peak_cuda_memory_mib"]),
            }
            for split in ("train", "val")
        },
        "parameter_counts": readouts["parameter_counts"],
        "metrics": {
            "train_keeper_in_sample": keeper_oof_metrics,
            "train_residual_oof": readouts["oof_metrics"],
            "val": {
                "keeper": keeper_val_metrics,
                **readouts["val_metrics"],
            },
        },
        "transitions": transitions,
        "directions": directions,
        "codeword_telemetry": telemetry,
        "xai": {
            "case_count": len(cases),
            "control_stats": xai["control_stats"],
            "candidate_stats": xai["candidate_stats"],
            "preview": "xai_deepten_class1_residual.png",
            "manifest": "xai_deepten_class1_residual.json",
        },
        "gate": gate,
        "seconds": float(time.perf_counter() - start),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    observed = gate["observed"]
    readme = [
        "# Deep-TEN Stem-Texture Readiness",
        "",
        (
            f"- Support mode: `{protocol['support_mode']}`; train/validation "
            f"rows `{len(train_labels)}/{len(val_labels)}`; test is closed."
        ),
        (
            "- Keeper validation macro/class1: "
            f"`{observed['keeper_val_macro_f1']:.6f}/"
            f"{observed['keeper_val_focus_f1']:.6f}`."
        ),
        (
            "- Mean/std control validation macro/class1: "
            f"`{observed['control_val_macro_f1']:.6f}/"
            f"{observed['control_val_focus_f1']:.6f}`."
        ),
        (
            "- Deep-TEN candidate validation macro/class1: "
            f"`{observed['candidate_val_macro_f1']:.6f}/"
            f"{observed['candidate_val_focus_f1']:.6f}`."
        ),
        (
            "- Codeword utilization entropy: "
            f"`{observed['codeword_utilization_entropy']:.6f}`."
        ),
        (
            "- Image-smoke permission: "
            f"`{str(bool(gate['image_smoke_permission'])).lower()}`."
        ),
        f"- Failed checks: `{','.join(gate['failed_checks'])}`.",
        "",
        (
            "Decision: implement one core-model Deep-TEN branch only if every "
            "locked check passes. Do not sweep codewords, grid, projection, "
            "residual scale, readout width, optimizer, folds, or epochs after "
            "rejection."
        ),
        "",
        (
            "This diagnostic writes no feature cache, model, checkpoint, test "
            "result, or trainable manifest."
        ),
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    manifest = _write_artifact_manifest(
        output_dir,
        mode="deepten_stem_texture_readiness_evidence_manifest",
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
