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
import statistics
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_augself_color_adapter_readiness import (
    _required_xai_indices,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _build_comparisons,
    _build_dataset,
    _failed_export,
    _full_worktree_clean,
    _heat_overlay,
    _onnx_compare,
    _replay_predictions,
    _rgb_from_tensor,
    _tracked_worktree_clean,
    _write_predictions,
)
from trkh.tools.audit_class1_boundary_cagrad_readiness import CONDITIONS
from trkh.tools.audit_class1_reference_agem_readiness import (
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _git_commit,
    _make_loader,
    _metadata_to_device,
    _prepare_output_dir,
    _sha256,
    _tensor_sha256,
    _verify_sha256,
)


METHOD = "mutual_channel_patch_adapter_a0"
NUM_CLASSES = 5
CHANNELS_PER_CLASS = 3
PROJECTED_CHANNELS = NUM_CLASSES * CHANNELS_PER_CLASS
EMBED_DIM = 256
GRID_HEIGHT = 16
GRID_WIDTH = 16
EXPECTED_PATCHES = 167
EXPECTED_KEEPER_PARAMETERS = 7_245_590
EXPECTED_ADAPTER_PARAMETERS = 3_935
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
FOLD = 0
RESIDUAL_SCALE = 0.05
ALPHA = 1.5
BETA = 20.0
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MAX_RUNTIME_RATIO = 1.10
MAX_PEAK_VRAM_GIB = 0.75
MAX_ONNX_ERROR = 1e-5
STATIC_EXPORT_BATCH_SIZE = 1

LOCKED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
LOCKED_LAUNCHER_ARGS_SHA256 = (
    "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
)
LOCKED_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
LOCKED_CIDT_SUMMARY_SHA256 = (
    "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
)
LOCKED_CIDT_PREDICTIONS_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_PROTOCOL_SHA256 = (
    "5444f20b01fbf1af3b2a36595012a6621912a31d5471e9ae9b94b9287b2680b2"
)
LOCKED_PAPER_SHA256 = (
    "4b1111d23d427e4219cea72763dfaec1f1866418ffb725418bfea95eab82f7d2"
)
LOCKED_OFFICIAL_COMMIT = "befb3692cd0d5382eb32fa4e093226247f609fd9"
LOCKED_OFFICIAL_RESNET_SHA256 = (
    "cf52b28b9a92cb2f2623f69576297f786521232a03f350a0e945360b13a92c3f"
)
LOCKED_OFFICIAL_POOLING_SHA256 = (
    "b2d526efa446a205d272610e06ed2ac117d3d6084042ec6fe38b7f6504bdaa54"
)
LOCKED_OFFICIAL_README_SHA256 = (
    "072043e1663ddfb1ddab5ad355ffb2cf0b351a94ea4225848cd59954cd53b26c"
)
LOCKED_OFFICIAL_LICENSE_SHA256 = (
    "4f8a89636b5e3ce81d19cdfbfdca6c21aa2cf63cb216d93d15f69c61012cd109"
)
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
LOCKED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
LOCKED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_TRAIN_ORDER_SHA256 = (
    "d56903f7806c12c19dddc4dc639054a98ba84a806f9568871157359be31512d4"
)
LOCKED_MASK_OCCURRENCE_SHA256 = (
    "0b15865764091d839872ac840cdc6a37965662a4ef6b9e3e8edc95a7e2c5afb7"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Mutual-Channel final-patch adapter audit. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
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
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_MUTUAL_CHANNEL_PATCH_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\mutual_channel_loss_tip2020.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\mutual_channel_loss"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_mutual_channel_patch_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == 4
        and int(args.epochs) == EPOCHS
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and math.isclose(float(args.learning_rate), LEARNING_RATE, abs_tol=0.0)
        and math.isclose(float(args.momentum), MOMENTUM, abs_tol=0.0)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, abs_tol=0.0)
        and math.isclose(float(args.alpha), ALPHA, abs_tol=0.0)
        and math.isclose(float(args.beta), BETA, abs_tol=0.0)
        and math.isclose(
            float(args.residual_scale), RESIDUAL_SCALE, abs_tol=0.0
        )
        and int(args.benchmark_repeats) >= 3
        and int(args.xai_batch_size) >= 1
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "official_resnet": official / "CUB-200-2011_ResNet18.py",
        "official_pooling": official / "my_pooling.py",
        "official_readme": official / "README.md",
        "official_license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def locked_training_order(
    fit_indices: Sequence[int],
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
) -> list[int]:
    values = np.asarray([int(value) for value in fit_indices], dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    return [
        int(value)
        for _ in range(int(epochs))
        for value in generator.permutation(values).tolist()
    ]


def mcl_drop_index(
    *,
    seed: int,
    epoch: int,
    batch: int,
    sample_index: int,
    class_index: int,
) -> int:
    payload = (
        f"mcl-mask|{int(seed)}|{int(epoch)}|{int(batch)}|"
        f"{int(sample_index)}|{int(class_index)}"
    )
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % 3


def mcl_drop_indices(
    sample_indices: Sequence[int],
    *,
    seed: int,
    epoch: int,
    batch: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    return torch.tensor(
        [
            [
                mcl_drop_index(
                    seed=seed,
                    epoch=epoch,
                    batch=batch,
                    sample_index=int(sample_index),
                    class_index=class_index,
                )
                for class_index in range(NUM_CLASSES)
            ]
            for sample_index in sample_indices
        ],
        dtype=torch.long,
        device=device,
    )


def mask_occurrence_sha256(
    order: Sequence[int],
    *,
    fit_rows: int,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    epochs: int = EPOCHS,
) -> str:
    if len(order) != int(fit_rows) * int(epochs):
        raise ValueError("Mask schedule order length differs from epochs * fit rows.")
    digest = hashlib.sha256()
    offset = 0
    for epoch in range(int(epochs)):
        epoch_order = order[offset : offset + int(fit_rows)]
        offset += int(fit_rows)
        for batch_start in range(0, int(fit_rows), int(batch_size)):
            batch = batch_start // int(batch_size)
            for sample_index in epoch_order[
                batch_start : batch_start + int(batch_size)
            ]:
                for class_index in range(NUM_CLASSES):
                    drop = mcl_drop_index(
                        seed=seed,
                        epoch=epoch,
                        batch=batch,
                        sample_index=int(sample_index),
                        class_index=class_index,
                    )
                    digest.update(
                        (
                            f"{epoch},{batch},{int(sample_index)},"
                            f"{class_index},{drop}\n"
                        ).encode("ascii")
                    )
    return digest.hexdigest()


class MutualChannelPatchAdapter(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = EMBED_DIM,
        num_classes: int = NUM_CLASSES,
        channels_per_class: int = CHANNELS_PER_CLASS,
        residual_scale: float = RESIDUAL_SCALE,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.channels_per_class = int(channels_per_class)
        self.projected_channels = self.num_classes * self.channels_per_class
        self.residual_scale = float(residual_scale)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.projection = nn.Conv2d(
                self.feature_dim,
                self.projected_channels,
                kernel_size=1,
                bias=True,
            )
            self.residual_head = nn.Linear(
                self.projected_channels,
                self.num_classes,
                bias=True,
            )
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def scatter_sparse_tokens(
        patches: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        *,
        grid_height: int = GRID_HEIGHT,
        grid_width: int = GRID_WIDTH,
    ) -> tuple[Tensor, Tensor]:
        if patches.ndim != 3:
            raise ValueError("patches must have shape [B,N,D]")
        if patch_indices.shape != patches.shape[:2]:
            raise ValueError("patch_indices must align with patches")
        if token_valid.shape != patches.shape[:2]:
            raise ValueError("token_valid must align with patches")
        grid_tokens = int(grid_height) * int(grid_width)
        indices = patch_indices.to(device=patches.device, dtype=torch.long)
        valid = token_valid.to(device=patches.device, dtype=torch.bool)
        if bool((indices < 0).any().item()) or bool(
            (indices >= grid_tokens).any().item()
        ):
            raise ValueError("Patch index is outside the native grid.")
        source = patches * valid.unsqueeze(-1).to(dtype=patches.dtype)
        dense_flat = patches.new_zeros(
            (patches.size(0), grid_tokens, patches.size(2))
        ).scatter(
            1,
            indices.unsqueeze(-1).expand(-1, -1, patches.size(2)),
            source,
        )
        valid_flat = torch.zeros(
            (patches.size(0), grid_tokens),
            device=patches.device,
            dtype=torch.bool,
        ).scatter(1, indices, valid)
        dense = dense_flat.transpose(1, 2).reshape(
            patches.size(0),
            patches.size(2),
            int(grid_height),
            int(grid_width),
        )
        mask = valid_flat.reshape(
            patches.size(0), 1, int(grid_height), int(grid_width)
        )
        return dense, mask

    @staticmethod
    def masked_spatial_max(projected: Tensor, valid_mask: Tensor) -> Tensor:
        if projected.ndim != 4 or valid_mask.shape != (
            projected.size(0),
            1,
            projected.size(2),
            projected.size(3),
        ):
            raise ValueError("Projected maps and valid mask do not align.")
        if bool((valid_mask.flatten(1).sum(dim=1) == 0).any().item()):
            raise ValueError("Every sample must retain at least one valid patch.")
        return projected.masked_fill(~valid_mask, -1.0e4).amax(dim=(2, 3))

    def forward_sparse(
        self,
        patches: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        raw_logits: Tensor,
    ) -> Dict[str, Tensor]:
        dense, valid_mask = self.scatter_sparse_tokens(
            patches, patch_indices, token_valid
        )
        projected = self.projection(dense)
        pooled = self.masked_spatial_max(projected, valid_mask)
        residual = self.residual_head(pooled)
        deployed = raw_logits.float() + self.residual_scale * residual.float()
        return {
            "dense": dense,
            "valid_mask": valid_mask,
            "projected": projected,
            "pooled": pooled,
            "residual_logits": residual,
            "logits": deployed,
        }


def spatial_channel_probabilities(projected: Tensor, valid_mask: Tensor) -> Tensor:
    if projected.ndim != 4:
        raise ValueError("projected must have shape [B,C,H,W]")
    flat = projected.flatten(2).float()
    valid = valid_mask.flatten(2).expand(-1, projected.size(1), -1)
    probabilities = F.softmax(flat.masked_fill(~valid, -1.0e4), dim=-1)
    probabilities = probabilities * valid.to(dtype=probabilities.dtype)
    return probabilities.reshape_as(projected)


def mutual_channel_terms(
    projected: Tensor,
    valid_mask: Tensor,
    drop_indices: Tensor,
    targets: Tensor,
) -> Dict[str, Tensor]:
    batch, channels, height, width = projected.shape
    if channels != PROJECTED_CHANNELS:
        raise ValueError("Mutual-Channel projection must have exactly 15 channels.")
    if drop_indices.shape != (batch, NUM_CLASSES):
        raise ValueError("drop_indices must have shape [B,5]")
    grouped = projected.reshape(
        batch, NUM_CLASSES, CHANNELS_PER_CLASS, height, width
    )
    drop = F.one_hot(
        drop_indices.to(device=projected.device, dtype=torch.long),
        num_classes=CHANNELS_PER_CLASS,
    ).to(dtype=projected.dtype)
    kept = grouped * (1.0 - drop[..., None, None])
    discriminative_maps = kept.amax(dim=2)
    valid = valid_mask.to(dtype=projected.dtype)
    valid_count = valid.sum(dim=(2, 3)).clamp_min(1.0)
    discriminative_logits = (
        discriminative_maps * valid
    ).sum(dim=(2, 3)) / valid_count
    per_row_dis = F.cross_entropy(
        discriminative_logits.float(),
        targets.to(dtype=torch.long),
        reduction="none",
    )
    channel_probabilities = spatial_channel_probabilities(projected, valid_mask)
    grouped_probabilities = channel_probabilities.reshape(
        batch, NUM_CLASSES, CHANNELS_PER_CLASS, height, width
    )
    grouped_max = grouped_probabilities.amax(dim=2)
    coverage = (grouped_max * valid).sum(dim=(2, 3))
    per_row_div = 1.0 - coverage.mean(dim=1) / float(CHANNELS_PER_CLASS)
    return {
        "discriminative_logits": discriminative_logits,
        "discriminative_maps": discriminative_maps,
        "channel_probabilities": channel_probabilities,
        "grouped_max_probabilities": grouped_max,
        "coverage": coverage,
        "per_row_dis": per_row_dis,
        "per_row_div": per_row_div,
        "loss_dis": per_row_dis.mean(),
        "loss_div": per_row_div.mean(),
    }


def reference_mutual_channel_terms(
    projected: Tensor,
    valid_mask: Tensor,
    drop_indices: Tensor,
    targets: Tensor,
) -> Dict[str, Tensor]:
    rows = []
    coverages = []
    probabilities = torch.zeros_like(projected, dtype=torch.float32)
    for sample in range(projected.size(0)):
        valid = valid_mask[sample, 0].flatten()
        if not bool(valid.any().item()):
            raise ValueError("Reference equation requires a valid patch.")
        sample_logits = []
        sample_coverage = []
        for class_index in range(NUM_CLASSES):
            group = projected[
                sample,
                class_index * CHANNELS_PER_CLASS : (class_index + 1)
                * CHANNELS_PER_CLASS,
            ].float()
            dropped = group.clone()
            dropped[int(drop_indices[sample, class_index].item())] = 0.0
            sample_logits.append(dropped.amax(dim=0).flatten()[valid].mean())
            channel_maps = []
            for channel_index in range(CHANNELS_PER_CLASS):
                flattened = group[channel_index].flatten()
                values = torch.softmax(flattened[valid], dim=0)
                full = torch.zeros_like(flattened)
                full[valid] = values
                probabilities[
                    sample,
                    class_index * CHANNELS_PER_CLASS + channel_index,
                ] = full.reshape_as(group[channel_index])
                channel_maps.append(values)
            sample_coverage.append(torch.stack(channel_maps).amax(dim=0).sum())
        rows.append(torch.stack(sample_logits))
        coverages.append(torch.stack(sample_coverage))
    logits = torch.stack(rows)
    coverage = torch.stack(coverages)
    per_row_dis = F.cross_entropy(
        logits,
        targets.to(dtype=torch.long),
        reduction="none",
    )
    per_row_div = 1.0 - coverage.mean(dim=1) / float(CHANNELS_PER_CLASS)
    return {
        "discriminative_logits": logits,
        "channel_probabilities": probabilities,
        "coverage": coverage,
        "per_row_dis": per_row_dis,
        "per_row_div": per_row_div,
        "loss_dis": per_row_dis.mean(),
        "loss_div": per_row_div.mean(),
    }


def _parameter_gradient_map(module: nn.Module) -> Dict[str, Optional[Tensor]]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _gradient_norm(values: Mapping[str, Optional[Tensor]]) -> float:
    total = sum(
        float(value.double().square().sum().item())
        for value in values.values()
        if value is not None
    )
    return math.sqrt(total)


def _equation_diagnostics() -> Dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 991)
    projected = torch.randn(
        3, PROJECTED_CHANNELS, 3, 4, generator=generator, dtype=torch.float32
    )
    valid = torch.tensor(
        [
            [[[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]],
            [[[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]],
            [[[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1]]],
        ],
        dtype=torch.bool,
    )
    drops = torch.tensor(
        [[0, 1, 2, 0, 1], [2, 1, 0, 2, 1], [1, 0, 2, 1, 0]],
        dtype=torch.long,
    )
    targets = torch.tensor([0, 1, 4], dtype=torch.long)
    local_input = projected.clone().requires_grad_(True)
    reference_input = projected.clone().requires_grad_(True)
    local = mutual_channel_terms(local_input, valid, drops, targets)
    reference = reference_mutual_channel_terms(
        reference_input, valid, drops, targets
    )
    local_total = ALPHA * local["loss_dis"] + BETA * local["loss_div"]
    reference_total = (
        ALPHA * reference["loss_dis"] + BETA * reference["loss_div"]
    )
    local_total.backward()
    reference_total.backward()
    checks = {
        "discriminative_logits": float(
            (
                local["discriminative_logits"]
                - reference["discriminative_logits"]
            )
            .abs()
            .amax()
            .item()
        ),
        "channel_probabilities": float(
            (
                local["channel_probabilities"]
                - reference["channel_probabilities"]
            )
            .abs()
            .amax()
            .item()
        ),
        "coverage": float(
            (local["coverage"] - reference["coverage"]).abs().amax().item()
        ),
        "loss_dis": float(
            (local["loss_dis"] - reference["loss_dis"]).abs().item()
        ),
        "loss_div": float(
            (local["loss_div"] - reference["loss_div"]).abs().item()
        ),
        "gradient": float(
            (local_input.grad - reference_input.grad).abs().amax().item()
        ),
    }
    return {
        "maximum_absolute_errors": checks,
        "all_equations_match_fp32_lte_1e6": max(checks.values()) <= 1e-6,
        "local_total": float(local_total.detach().item()),
        "reference_total": float(reference_total.detach().item()),
    }


def _cohort_summary(rows: Sequence[CleanTrainRow], *, fold: int) -> Dict[str, object]:
    fit = [row.sample_index for row in rows if int(row.fold) != int(fold)]
    holdout = [row.sample_index for row in rows if int(row.fold) == int(fold)]
    fit_sources = {rows[index].source_stem for index in fit}
    holdout_sources = {rows[index].source_stem for index in holdout}
    return {
        "fit_indices": fit,
        "holdout_indices": holdout,
        "fit_rows": len(fit),
        "holdout_rows": len(holdout),
        "fit_index_sha256": _ordered_index_sha256(fit),
        "holdout_index_sha256": _ordered_index_sha256(holdout),
        "source_overlap": len(fit_sources.intersection(holdout_sources)),
        "fit_class_counts": np.bincount(
            np.asarray([rows[index].target for index in fit], dtype=np.int64),
            minlength=NUM_CLASSES,
        ).tolist(),
        "holdout_class_counts": np.bincount(
            np.asarray([rows[index].target for index in holdout], dtype=np.int64),
            minlength=NUM_CLASSES,
        ).tolist(),
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], Dict[str, object], list[int]]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted MCL protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"
        ),
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
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "MCL protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "MCL paper"),
        "official_resnet": _verify_sha256(
            paths["official_resnet"],
            LOCKED_OFFICIAL_RESNET_SHA256,
            "official ResNet script",
        ),
        "official_pooling": _verify_sha256(
            paths["official_pooling"],
            LOCKED_OFFICIAL_POOLING_SHA256,
            "official pooling script",
        ),
        "official_readme": _verify_sha256(
            paths["official_readme"],
            LOCKED_OFFICIAL_README_SHA256,
            "official README",
        ),
        "official_license": _verify_sha256(
            paths["official_license"],
            LOCKED_OFFICIAL_LICENSE_SHA256,
            "official license",
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best command file",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    official_commit = _git_commit(paths["official_root"])
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official MCL commit differs: {official_commit} != "
            f"{LOCKED_OFFICIAL_COMMIT}"
        )
    official_clean = _full_worktree_clean(paths["official_root"])
    if not official_clean:
        raise ValueError("Official MCL worktree is not clean.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use.")
    if bool(
        cidt_summary.get(
            "validation_predictions_used",
            cidt_summary.get("validation_data_used", True),
        )
    ):
        raise ValueError("CIDT provenance indicates validation use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _cohort_summary(rows, fold=int(args.fold))
    if cohorts["fit_rows"] != 7372 or cohorts["holdout_rows"] != 1843:
        raise ValueError("MCL source-disjoint row counts differ from protocol.")
    if cohorts["fit_index_sha256"] != LOCKED_FIT_INDEX_SHA256:
        raise ValueError("MCL fit-index hash differs from protocol.")
    if cohorts["holdout_index_sha256"] != LOCKED_HOLDOUT_INDEX_SHA256:
        raise ValueError("MCL holdout-index hash differs from protocol.")
    if int(cohorts["source_overlap"]) != 0:
        raise ValueError("MCL fit/holdout source groups overlap.")
    order = locked_training_order(
        cohorts["fit_indices"], seed=int(args.seed), epochs=int(args.epochs)
    )
    order_hash = _ordered_index_sha256(order)
    if order_hash != LOCKED_TRAIN_ORDER_SHA256:
        raise ValueError("MCL train order hash differs from protocol.")
    mask_hash = mask_occurrence_sha256(
        order,
        fit_rows=int(cohorts["fit_rows"]),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        epochs=int(args.epochs),
    )
    if mask_hash != LOCKED_MASK_OCCURRENCE_SHA256:
        raise ValueError("MCL mask occurrence hash differs from protocol.")
    root = Path.cwd().resolve()
    tracked_clean = _tracked_worktree_clean(root)
    if not tracked_clean:
        raise ValueError("Tracked TRKH worktree must be clean for formal MCL A0.")
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_worktree_clean": official_clean,
            "official_license_present": paths["official_license"].is_file(),
            "repository_commit": _git_commit(root),
            "tracked_worktree_clean": tracked_clean,
            "train_order_sha256": order_hash,
            "mask_occurrence_sha256": mask_hash,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
        order,
    )


def _construct_keeper_and_adapters(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, MutualChannelPatchAdapter, MutualChannelPatchAdapter, Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
        raise ValueError("Keeper class order is invalid.")
    set_seed(SEED, deterministic=True)
    keeper = create_model(
        num_classes=NUM_CLASSES, model_config=model_config
    ).eval()
    load_model_state(keeper, dict(model_state), strict=True)
    for parameter in keeper.parameters():
        parameter.requires_grad_(False)
    rng_before = torch.get_rng_state().clone()
    prototype = MutualChannelPatchAdapter()
    rng_after = torch.get_rng_state().clone()
    control = copy.deepcopy(prototype)
    candidate = copy.deepcopy(prototype)
    control_optimizer = torch.optim.SGD(
        control.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    candidate_optimizer = torch.optim.SGD(
        candidate.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    adapter_parameters = sum(parameter.numel() for parameter in prototype.parameters())
    return keeper, control, candidate, {
        "keeper_parameters": sum(parameter.numel() for parameter in keeper.parameters()),
        "adapter_parameters": adapter_parameters,
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "candidate_control_bit_exact": _state_sha256(control)
        == _state_sha256(candidate),
        "candidate_control_initial_optimizer_bit_exact": (
            control_optimizer.state_dict() == candidate_optimizer.state_dict()
        ),
        "parameter_schema": {
            name: [int(value) for value in parameter.shape]
            for name, parameter in prototype.named_parameters()
        },
        "isolated_initialization_rng_restored": torch.equal(
            rng_before, rng_after
        ),
        "residual_head_weight_zero": bool(
            torch.count_nonzero(prototype.residual_head.weight).item() == 0
        ),
        "residual_head_bias_zero": bool(
            torch.count_nonzero(prototype.residual_head.bias).item() == 0
        ),
    }


def _feature_tensors(
    features: Mapping[str, object],
) -> tuple[Tensor, Tensor, Tensor]:
    patches = features.get("patches")
    indices = features.get("patch_indices")
    if not torch.is_tensor(patches) or patches.ndim != 3:
        raise ValueError("Keeper did not expose final normalized patch tokens.")
    if not torch.is_tensor(indices) or indices.shape != patches.shape[:2]:
        raise ValueError("Keeper patch indices do not align.")
    key_padding = features.get("memory_key_padding_mask")
    valid = (
        ~key_padding.to(dtype=torch.bool)
        if torch.is_tensor(key_padding)
        else torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
    )
    return patches, indices.long(), valid


def _extract_fit_cache(
    *,
    keeper: nn.Module,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, Tensor], Dict[str, object], tuple[Tensor, Tensor, Mapping[str, object]]]:
    loader, loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="mcl_fit_feature_cache",
        seed=int(args.seed) + 100,
    )
    keeper = keeper.to(device).eval()
    cache: Optional[Dict[str, Tensor]] = None
    offset = 0
    resource_batch = None
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images_cpu, targets_cpu, metadata_cpu) in enumerate(loader):
            if resource_batch is None:
                resource_batch = (
                    images_cpu.detach().clone(),
                    targets_cpu.detach().clone(),
                    {
                        key: value.detach().clone()
                        if torch.is_tensor(value)
                        else value
                        for key, value in metadata_cpu.items()
                    },
                )
            images = images_cpu.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=True
            ):
                raw_logits, features = _forward_classification_with_metadata(
                    keeper, images, metadata_cpu, device=device
                )
            if not isinstance(features, Mapping):
                raise ValueError("Keeper feature extraction returned no mapping.")
            patches, patch_indices, token_valid = _feature_tensors(features)
            sample_indices = metadata_cpu.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Feature cache batch is missing sample_index.")
            count = int(targets_cpu.numel())
            if cache is None:
                if (
                    int(patches.size(1)) != EXPECTED_PATCHES
                    or int(patches.size(2)) != EMBED_DIM
                ):
                    raise ValueError(
                        f"Keeper patch shape differs: {tuple(patches.shape)}"
                    )
                cache = {
                    "patches": torch.empty(
                        len(indices), EXPECTED_PATCHES, EMBED_DIM, dtype=torch.float32
                    ),
                    "patch_indices": torch.empty(
                        len(indices), EXPECTED_PATCHES, dtype=torch.long
                    ),
                    "token_valid": torch.empty(
                        len(indices), EXPECTED_PATCHES, dtype=torch.bool
                    ),
                    "raw_logits": torch.empty(
                        len(indices), NUM_CLASSES, dtype=torch.float32
                    ),
                    "targets": torch.empty(len(indices), dtype=torch.long),
                    "sample_indices": torch.empty(len(indices), dtype=torch.long),
                }
            row_slice = slice(offset, offset + count)
            cache["patches"][row_slice].copy_(patches.detach().float().cpu())
            cache["patch_indices"][row_slice].copy_(
                patch_indices.detach().long().cpu()
            )
            cache["token_valid"][row_slice].copy_(
                token_valid.detach().bool().cpu()
            )
            cache["raw_logits"][row_slice].copy_(
                raw_logits.detach().float().cpu()
            )
            cache["targets"][row_slice].copy_(targets_cpu.detach().long())
            cache["sample_indices"][row_slice].copy_(
                sample_indices.detach().long()
            )
            offset += count
            if offset % 1024 < count or offset == len(indices):
                print(
                    json.dumps(
                        {
                            "phase": "fit_feature_cache",
                            "processed": offset,
                            "rows": len(indices),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if cache is None or offset != len(indices) or resource_batch is None:
        raise RuntimeError("MCL fit cache extraction is incomplete.")
    expected = [int(value) for value in indices]
    observed = cache["sample_indices"].tolist()
    if observed != expected:
        raise ValueError("MCL fit cache changed sample order.")
    return (
        cache,
        {
            "rows": offset,
            "patch_shape": [EXPECTED_PATCHES, EMBED_DIM],
            "patch_dtype": str(cache["patches"].dtype),
            "sample_index_sha256": _ordered_index_sha256(observed),
            "patch_tensor_sha256": _stream_tensor_sha256(cache["patches"]),
            "patch_index_sha256": _stream_tensor_sha256(
                cache["patch_indices"]
            ),
            "token_valid_sha256": _stream_tensor_sha256(cache["token_valid"]),
            "raw_logit_sha256": _stream_tensor_sha256(cache["raw_logits"]),
            "all_finite": bool(
                torch.isfinite(cache["patches"]).all()
                and torch.isfinite(cache["raw_logits"]).all()
            ),
            "loader": loader_summary,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        resource_batch,
    )


def _cache_positions(cache: Mapping[str, Tensor]) -> Tensor:
    sample_indices = cache["sample_indices"].long()
    size = int(sample_indices.max().item()) + 1
    positions = torch.full((size,), -1, dtype=torch.long)
    positions[sample_indices] = torch.arange(sample_indices.numel())
    return positions


def _batch_from_cache(
    cache: Mapping[str, Tensor],
    positions: Tensor,
    sample_indices: Sequence[int],
    *,
    device: torch.device,
) -> Dict[str, Tensor]:
    index = torch.tensor([int(value) for value in sample_indices], dtype=torch.long)
    if int(index.max().item()) >= positions.numel():
        raise ValueError("Training order contains an out-of-range sample.")
    rows = positions[index]
    if bool((rows < 0).any().item()):
        raise ValueError("Training order contains a sample outside fit cache.")
    return {
        key: value[rows].to(device=device, non_blocking=True)
        for key, value in cache.items()
        if key != "sample_indices"
    }


def _initial_gradient_audit(
    *,
    prototype: MutualChannelPatchAdapter,
    cache: Mapping[str, Tensor],
    order: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    positions = _cache_positions(cache)
    batch_indices = [int(value) for value in order[:BATCH_SIZE]]
    batch = _batch_from_cache(
        cache, positions, batch_indices, device=device
    )
    drops = mcl_drop_indices(
        batch_indices, seed=SEED, epoch=0, batch=0, device=device
    )
    output: Dict[str, object] = {}
    gradient_maps: Dict[str, Dict[str, Optional[Tensor]]] = {}
    for name in ("control", "candidate"):
        adapter = copy.deepcopy(prototype).to(device).train()
        result = adapter.forward_sparse(
            batch["patches"],
            batch["patch_indices"],
            batch["token_valid"],
            batch["raw_logits"],
        )
        terms = mutual_channel_terms(
            result["projected"],
            result["valid_mask"],
            drops,
            batch["targets"],
        )
        ce = F.cross_entropy(result["logits"], batch["targets"])
        if name == "candidate":
            loss = ce + ALPHA * terms["loss_dis"] + BETA * terms["loss_div"]
        else:
            loss = ce + 0.0 * (
                ALPHA * terms["loss_dis"] + BETA * terms["loss_div"]
            ).detach()
        loss.backward()
        gradients = _parameter_gradient_map(adapter)
        gradient_maps[name] = gradients
        output[name] = {
            "loss": float(loss.detach().item()),
            "ce": float(ce.detach().item()),
            "loss_dis": float(terms["loss_dis"].detach().item()),
            "loss_div": float(terms["loss_div"].detach().item()),
            "gradient_norm": _gradient_norm(gradients),
            "gradient_finite": all(
                value is not None and bool(torch.isfinite(value).all())
                for value in gradients.values()
            ),
            "gradient_nonzero": {
                key: bool(value is not None and torch.count_nonzero(value).item())
                for key, value in gradients.items()
            },
        }
        del adapter, result, terms, loss
    ce_only = copy.deepcopy(prototype).to(device).train()
    ce_result = ce_only.forward_sparse(
        batch["patches"],
        batch["patch_indices"],
        batch["token_valid"],
        batch["raw_logits"],
    )
    F.cross_entropy(ce_result["logits"], batch["targets"]).backward()
    ce_only_gradients = _parameter_gradient_map(ce_only)
    control_gradients = gradient_maps["control"]
    detached_control_matches_ce = (
        ce_only_gradients.keys() == control_gradients.keys()
        and all(
            ce_only_gradients[key] is not None
            and control_gradients[key] is not None
            and torch.equal(ce_only_gradients[key], control_gradients[key])
            for key in ce_only_gradients
        )
    )
    candidate = copy.deepcopy(prototype).to(device).train()
    result = candidate.forward_sparse(
        batch["patches"],
        batch["patch_indices"],
        batch["token_valid"],
        batch["raw_logits"],
    )
    terms = mutual_channel_terms(
        result["projected"],
        result["valid_mask"],
        drops,
        batch["targets"],
    )
    mcl_only = ALPHA * terms["loss_dis"] + BETA * terms["loss_div"]
    mcl_only.backward()
    mcl_gradients = _parameter_gradient_map(candidate)
    output["candidate_mcl_only"] = {
        "projection_gradient_norm": _gradient_norm(
            {
                key: value
                for key, value in mcl_gradients.items()
                if key.startswith("projection.")
            }
        ),
        "head_gradient_exact_zero": all(
            value is None or torch.count_nonzero(value).item() == 0
            for key, value in mcl_gradients.items()
            if key.startswith("residual_head.")
        ),
    }
    return {
        **output,
        "rows": len(batch_indices),
        "sample_index_sha256": _ordered_index_sha256(batch_indices),
        "candidate_projection_mcl_gradient_nonzero": float(
            output["candidate_mcl_only"]["projection_gradient_norm"]
        )
        > 0.0,
        "control_diagnostic_detached": True,
        "control_detached_gradient_matches_ce_only_bit_exact": (
            detached_control_matches_ce
        ),
    }


def _train_variant(
    *,
    name: str,
    prototype: MutualChannelPatchAdapter,
    cache: Mapping[str, Tensor],
    order: Sequence[int],
    candidate: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MutualChannelPatchAdapter, Dict[str, object]]:
    adapter = copy.deepcopy(prototype).to(device).train()
    initial = {
        key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()
    }
    optimizer = torch.optim.SGD(
        adapter.parameters(),
        lr=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    positions = _cache_positions(cache)
    fit_rows = int(cache["sample_indices"].numel())
    history = []
    gradient_seen = {name: False for name, _ in adapter.named_parameters()}
    all_gradients_finite = True
    offset = 0
    started = time.perf_counter()
    for epoch in range(int(args.epochs)):
        sums = {
            "loss": 0.0,
            "ce": 0.0,
            "loss_dis": 0.0,
            "loss_div": 0.0,
            "coverage": 0.0,
            "rows": 0,
            "batches": 0,
        }
        epoch_order = order[offset : offset + fit_rows]
        offset += fit_rows
        for batch_start in range(0, fit_rows, int(args.batch_size)):
            batch_number = batch_start // int(args.batch_size)
            sample_indices = epoch_order[
                batch_start : batch_start + int(args.batch_size)
            ]
            batch = _batch_from_cache(
                cache, positions, sample_indices, device=device
            )
            drops = mcl_drop_indices(
                sample_indices,
                seed=int(args.seed),
                epoch=epoch,
                batch=batch_number,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            result = adapter.forward_sparse(
                batch["patches"],
                batch["patch_indices"],
                batch["token_valid"],
                batch["raw_logits"],
            )
            terms = mutual_channel_terms(
                result["projected"],
                result["valid_mask"],
                drops,
                batch["targets"],
            )
            ce = F.cross_entropy(result["logits"], batch["targets"])
            if candidate:
                loss = (
                    ce
                    + float(args.alpha) * terms["loss_dis"]
                    + float(args.beta) * terms["loss_div"]
                )
            else:
                loss = ce + 0.0 * (
                    float(args.alpha) * terms["loss_dis"]
                    + float(args.beta) * terms["loss_div"]
                ).detach()
            loss.backward()
            for parameter_name, parameter in adapter.named_parameters():
                gradient = parameter.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    all_gradients_finite = False
                elif torch.count_nonzero(gradient).item() > 0:
                    gradient_seen[parameter_name] = True
            optimizer.step()
            count = int(batch["targets"].numel())
            sums["loss"] += float(loss.detach().item()) * count
            sums["ce"] += float(ce.detach().item()) * count
            sums["loss_dis"] += float(terms["loss_dis"].detach().item()) * count
            sums["loss_div"] += float(terms["loss_div"].detach().item()) * count
            sums["coverage"] += float(terms["coverage"].detach().mean().item()) * count
            sums["rows"] += count
            sums["batches"] += 1
        history.append(
            {
                "variant": name,
                "epoch": epoch,
                "rows": int(sums["rows"]),
                "batches": int(sums["batches"]),
                "loss": sums["loss"] / sums["rows"],
                "ce": sums["ce"] / sums["rows"],
                "loss_dis": sums["loss_dis"] / sums["rows"],
                "loss_div": sums["loss_div"] / sums["rows"],
                "coverage": sums["coverage"] / sums["rows"],
            }
        )
        print(json.dumps(history[-1]), flush=True)
    final = adapter.state_dict()
    movement = {
        key: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final[key].detach().cpu()),
            "changed": not torch.equal(value, final[key].detach().cpu()),
        }
        for key, value in initial.items()
    }
    adapter = adapter.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return adapter, {
        "name": name,
        "epochs": int(args.epochs),
        "train_rows": len(order),
        "train_batches": sum(int(row["batches"]) for row in history),
        "train_order_sha256": _ordered_index_sha256(order),
        "history": history,
        "initial_state_sha256": _state_dict_sha256(initial),
        "final_state_sha256": _state_sha256(adapter),
        "gradient_seen_nonzero": gradient_seen,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "all_gradients_finite": all_gradients_finite,
        "movement": movement,
        "all_trainable_parameters_changed": all(
            bool(row["changed"]) for row in movement.values()
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _state_dict_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stream_tensor_sha256(value: Tensor, *, rows_per_chunk: int = 64) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        digest.update(memoryview(tensor.numpy()).cast("B"))
        return digest.hexdigest()
    for start in range(0, int(tensor.size(0)), int(rows_per_chunk)):
        chunk = tensor[start : start + int(rows_per_chunk)].contiguous().numpy()
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _mechanism_accumulator() -> Dict[str, object]:
    return {
        "rows": 0,
        "loss_dis_sum": 0.0,
        "loss_div_sum": 0.0,
        "coverage_sum": 0.0,
        "grouped_correct": 0,
        "entropy_sum": 0.0,
        "entropy_count": 0,
        "maximum_within_group_cosine": -math.inf,
        "pooled": [],
        "direction_scores": [],
        "direction_labels": [],
    }


def _update_mechanism(
    accumulator: Dict[str, object],
    *,
    result: Mapping[str, Tensor],
    terms: Mapping[str, Tensor],
    targets: Tensor,
) -> None:
    count = int(targets.numel())
    accumulator["rows"] = int(accumulator["rows"]) + count
    accumulator["loss_dis_sum"] = float(accumulator["loss_dis_sum"]) + float(
        terms["per_row_dis"].detach().sum().item()
    )
    accumulator["loss_div_sum"] = float(accumulator["loss_div_sum"]) + float(
        terms["per_row_div"].detach().sum().item()
    )
    accumulator["coverage_sum"] = float(accumulator["coverage_sum"]) + float(
        terms["coverage"].detach().sum().item()
    )
    accumulator["grouped_correct"] = int(accumulator["grouped_correct"]) + int(
        (
            terms["discriminative_logits"].detach().argmax(dim=1)
            == targets.detach()
        )
        .sum()
        .item()
    )
    probabilities = terms["channel_probabilities"].detach().float()
    valid = result["valid_mask"].detach().flatten(2)
    class1 = probabilities[
        :, FOCUS_CLASS * CHANNELS_PER_CLASS : (FOCUS_CLASS + 1) * CHANNELS_PER_CLASS
    ].flatten(2)
    valid_count = valid.sum(dim=2).clamp_min(2).float()
    entropy = -(
        class1.clamp_min(1e-12).log() * class1
    ).sum(dim=2) / valid_count.log()
    accumulator["entropy_sum"] = float(accumulator["entropy_sum"]) + float(
        entropy.sum().item()
    )
    accumulator["entropy_count"] = int(accumulator["entropy_count"]) + int(
        entropy.numel()
    )
    normalized = F.normalize(class1, p=2, dim=2, eps=1e-12)
    cosine = torch.einsum("bcs,bds->bcd", normalized, normalized)
    off_diagonal = ~torch.eye(
        CHANNELS_PER_CLASS, device=cosine.device, dtype=torch.bool
    ).unsqueeze(0)
    maximum_cosine = float(cosine.masked_select(off_diagonal).max().item())
    accumulator["maximum_within_group_cosine"] = max(
        float(accumulator["maximum_within_group_cosine"]), maximum_cosine
    )
    accumulator["pooled"].append(result["pooled"].detach().float().cpu())
    residual = result["residual_logits"].detach().float()
    restricted = torch.tensor(
        RESTRICTED_NEGATIVE_CLASSES, device=residual.device, dtype=torch.long
    )
    margin = residual[:, FOCUS_CLASS] - residual.index_select(1, restricted).amax(
        dim=1
    )
    direction_mask = (targets == FOCUS_CLASS) | torch.isin(targets, restricted)
    accumulator["direction_scores"].extend(
        margin[direction_mask].detach().cpu().tolist()
    )
    accumulator["direction_labels"].extend(
        (targets[direction_mask] == FOCUS_CLASS).long().detach().cpu().tolist()
    )


def _effective_rank(values: Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())


def _finalize_mechanism(accumulator: Mapping[str, object]) -> Dict[str, object]:
    rows = int(accumulator["rows"])
    pooled = torch.cat(list(accumulator["pooled"]), dim=0)
    labels = np.asarray(accumulator["direction_labels"], dtype=np.int64)
    scores = np.asarray(accumulator["direction_scores"], dtype=np.float64)
    auroc = (
        float(roc_auc_score(labels, scores))
        if labels.size and np.unique(labels).size == 2
        else None
    )
    return {
        "rows": rows,
        "loss_dis": float(accumulator["loss_dis_sum"]) / rows,
        "loss_div": float(accumulator["loss_div_sum"]) / rows,
        "mean_group_coverage": float(accumulator["coverage_sum"])
        / (rows * NUM_CLASSES),
        "grouped_logit_accuracy": int(accumulator["grouped_correct"]) / rows,
        "class1_spatial_entropy": float(accumulator["entropy_sum"])
        / int(accumulator["entropy_count"]),
        "maximum_class1_within_group_cosine": float(
            accumulator["maximum_within_group_cosine"]
        ),
        "pooled_channel_effective_rank": _effective_rank(pooled),
        "class1_residual_margin_auroc_vs_0_2_4": auroc,
        "direction_rows": int(labels.size),
        "all_finite": bool(
            torch.isfinite(pooled).all()
            and np.isfinite(scores).all()
        ),
    }


def _evaluate_cache_mechanism(
    *,
    adapter: MutualChannelPatchAdapter,
    cache: Mapping[str, Tensor],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    adapter = copy.deepcopy(adapter).to(device).eval()
    accumulator = _mechanism_accumulator()
    with torch.inference_mode():
        for start in range(0, int(cache["targets"].numel()), int(batch_size)):
            stop = min(start + int(batch_size), int(cache["targets"].numel()))
            sample_indices = cache["sample_indices"][start:stop].tolist()
            batch = {
                key: value[start:stop].to(device)
                for key, value in cache.items()
                if key != "sample_indices"
            }
            result = adapter.forward_sparse(
                batch["patches"],
                batch["patch_indices"],
                batch["token_valid"],
                batch["raw_logits"],
            )
            drops = mcl_drop_indices(
                sample_indices,
                seed=SEED,
                epoch=EPOCHS,
                batch=start // int(batch_size),
                device=device,
            )
            terms = mutual_channel_terms(
                result["projected"],
                result["valid_mask"],
                drops,
                batch["targets"],
            )
            _update_mechanism(
                accumulator, result=result, terms=terms, targets=batch["targets"]
            )
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return _finalize_mechanism(accumulator)


def _prediction_row(
    *, sample_index: int, target: int, probabilities: Tensor
) -> Dict[str, object]:
    values = probabilities.detach().float().cpu()
    row: Dict[str, object] = {
        "sample_index": int(sample_index),
        "target": int(target),
        "prediction": int(values.argmax().item()),
    }
    for class_index in range(NUM_CLASSES):
        row[f"prob_{class_index}"] = float(values[class_index].item())
    return row


def _condition_loader(
    *,
    name: str,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    args: argparse.Namespace,
) -> tuple[DataLoader, Dict[str, object]]:
    if name == "clean":
        return _make_loader(
            base_dataset=dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context="mcl_holdout_clean",
            seed=int(args.seed) + 200,
        )
    lighting = {
        condition: (brightness, contrast)
        for condition, brightness, contrast in LIGHTING_CONDITIONS
    }
    brightness, contrast = lighting[name]
    return _make_lighting_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        brightness=brightness,
        contrast=contrast,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"mcl_holdout_{name}",
        seed=int(args.seed) + 201 + list(lighting).index(name),
    )


def _evaluate_conditions(
    *,
    keeper: nn.Module,
    control: MutualChannelPatchAdapter,
    candidate: MutualChannelPatchAdapter,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    rows: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, Dict[str, object]],
    Dict[str, object],
]:
    keeper = keeper.to(device).eval()
    control_device = copy.deepcopy(control).to(device).eval()
    candidate_device = copy.deepcopy(candidate).to(device).eval()
    for module in (control_device, candidate_device):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    raw_conditions: Dict[str, list[Dict[str, object]]] = {}
    control_conditions: Dict[str, list[Dict[str, object]]] = {}
    candidate_conditions: Dict[str, list[Dict[str, object]]] = {}
    mechanism: Dict[str, Dict[str, object]] = {}
    loaders: Dict[str, object] = {}
    for condition in CONDITIONS:
        loader, loader_summary = _condition_loader(
            name=condition,
            dataset=dataset,
            transform=transform,
            indices=indices,
            args=args,
        )
        loaders[condition] = loader_summary
        raw_rows: list[Dict[str, object]] = []
        control_rows: list[Dict[str, object]] = []
        candidate_rows: list[Dict[str, object]] = []
        control_accumulator = _mechanism_accumulator()
        candidate_accumulator = _mechanism_accumulator()
        processed = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_number, (images_cpu, targets_cpu, metadata_cpu) in enumerate(
                loader
            ):
                images = images_cpu.to(device=device, non_blocking=True)
                targets = targets_cpu.to(device=device, dtype=torch.long)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=True
                ):
                    raw_logits, features = _forward_classification_with_metadata(
                        keeper, images, metadata_cpu, device=device
                    )
                if not isinstance(features, Mapping):
                    raise ValueError("Holdout keeper forward returned no features.")
                patches, patch_indices, token_valid = _feature_tensors(features)
                patches = patches.float()
                raw_logits = raw_logits.float()
                sample_tensor = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_tensor):
                    raise ValueError("Holdout batch is missing sample_index.")
                sample_indices = [int(value) for value in sample_tensor.tolist()]
                drops = mcl_drop_indices(
                    sample_indices,
                    seed=int(args.seed),
                    epoch=EPOCHS,
                    batch=batch_number,
                    device=device,
                )
                control_result = control_device.forward_sparse(
                    patches, patch_indices, token_valid, raw_logits
                )
                candidate_result = candidate_device.forward_sparse(
                    patches, patch_indices, token_valid, raw_logits
                )
                control_terms = mutual_channel_terms(
                    control_result["projected"],
                    control_result["valid_mask"],
                    drops,
                    targets,
                )
                candidate_terms = mutual_channel_terms(
                    candidate_result["projected"],
                    candidate_result["valid_mask"],
                    drops,
                    targets,
                )
                _update_mechanism(
                    control_accumulator,
                    result=control_result,
                    terms=control_terms,
                    targets=targets,
                )
                _update_mechanism(
                    candidate_accumulator,
                    result=candidate_result,
                    terms=candidate_terms,
                    targets=targets,
                )
                raw_prob = F.softmax(raw_logits, dim=1)
                control_prob = F.softmax(control_result["logits"], dim=1)
                candidate_prob = F.softmax(candidate_result["logits"], dim=1)
                for local, sample_index in enumerate(sample_indices):
                    target = int(targets_cpu[local].item())
                    raw_rows.append(
                        _prediction_row(
                            sample_index=sample_index,
                            target=target,
                            probabilities=raw_prob[local],
                        )
                    )
                    control_rows.append(
                        _prediction_row(
                            sample_index=sample_index,
                            target=target,
                            probabilities=control_prob[local],
                        )
                    )
                    candidate_rows.append(
                        _prediction_row(
                            sample_index=sample_index,
                            target=target,
                            probabilities=candidate_prob[local],
                        )
                    )
                processed += len(sample_indices)
                if processed % 512 < len(sample_indices) or processed == len(indices):
                    print(
                        json.dumps(
                            {
                                "phase": "holdout_evaluation",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(indices),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
        expected = [int(value) for value in indices]
        if [int(row["sample_index"]) for row in raw_rows] != expected:
            raise ValueError(f"Condition {condition} changed holdout order.")
        raw_conditions[condition] = raw_rows
        control_conditions[condition] = control_rows
        candidate_conditions[condition] = candidate_rows
        mechanism[condition] = {
            "control": _finalize_mechanism(control_accumulator),
            "candidate": _finalize_mechanism(candidate_accumulator),
        }
    clean_raw_mismatches = sum(
        int(row["prediction"]) != int(rows[int(row["sample_index"])].keeper_prediction)
        for row in raw_conditions["clean"]
    )
    del control_device, candidate_device
    gc.collect()
    torch.cuda.empty_cache()
    return (
        raw_conditions,
        control_conditions,
        candidate_conditions,
        mechanism,
        {
            "loaders": loaders,
            "clean_raw_cidt_argmax_mismatches": clean_raw_mismatches,
        },
    )


class _IsolatedAdapterExport(nn.Module):
    def __init__(self, adapter: MutualChannelPatchAdapter) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(
        self,
        patches: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        raw_logits: Tensor,
    ) -> Tensor:
        return self.adapter.forward_sparse(
            patches, patch_indices, token_valid, raw_logits
        )["logits"]


class _FullCandidateExport(nn.Module):
    def __init__(
        self, keeper: nn.Module, adapter: MutualChannelPatchAdapter
    ) -> None:
        super().__init__()
        self.keeper = copy.deepcopy(keeper).cpu().eval()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.keeper.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        raw_logits = classification_logits_from_features(self.keeper, features)
        patches, patch_indices, token_valid = _feature_tensors(features)
        return self.adapter.forward_sparse(
            patches, patch_indices, token_valid, raw_logits
        )["logits"]


def _export_diagnostics(
    *,
    keeper: nn.Module,
    candidate: MutualChannelPatchAdapter,
    cache: Mapping[str, Tensor],
    resource_batch: tuple[Tensor, Tensor, Mapping[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    isolated_path = output_dir / "mutual_channel_patch_adapter.onnx"
    try:
        isolated = _onnx_compare(
            wrapper=_IsolatedAdapterExport(candidate),
            inputs=(
                cache["patches"][:1],
                cache["patch_indices"][:1],
                cache["token_valid"][:1],
                cache["raw_logits"][:1],
            ),
            input_names=("patches", "patch_indices", "token_valid", "raw_logits"),
            path=isolated_path,
        )
    except Exception as error:
        isolated = _failed_export(isolated_path, error)
    images, _targets, metadata = resource_batch
    images = images[:STATIC_EXPORT_BATCH_SIZE].float().cpu()
    bbox = metadata.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(STATIC_EXPORT_BATCH_SIZE, 8, dtype=torch.float32)
    bbox = bbox[:STATIC_EXPORT_BATCH_SIZE].float().cpu()
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            STATIC_EXPORT_BATCH_SIZE,
            int(images.size(-2)),
            int(images.size(-1)),
            dtype=torch.bool,
        )
    image_mask = image_mask[:STATIC_EXPORT_BATCH_SIZE].bool().cpu()
    full_path = output_dir / "mutual_channel_patch_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_FullCandidateExport(keeper, candidate),
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as error:
        full = _failed_export(full_path, error)
    return {"isolated": isolated, "full": full}


def _benchmark_inference(
    *,
    keeper: nn.Module,
    adapter: Optional[MutualChannelPatchAdapter],
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
) -> Dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    keeper = keeper.to(device).eval()
    adapter_device = (
        copy.deepcopy(adapter).to(device).eval() if adapter is not None else None
    )
    images = images_cpu[:1].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=1)

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=True
        ):
            raw_logits, features = _forward_classification_with_metadata(
                keeper, images, metadata, device=device
            )
        if adapter_device is None:
            return raw_logits.float()
        if not isinstance(features, Mapping):
            raise ValueError("Benchmark keeper forward returned no features.")
        patches, patch_indices, token_valid = _feature_tensors(features)
        return adapter_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )["logits"]

    for _ in range(3):
        logits = iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "batch_size": 1,
    }
    del adapter_device, images, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _normalize_map(value: Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    return array.astype(np.float32)


def _saliency_subset(
    selected: Sequence[int],
    categories: Mapping[int, Sequence[str]],
    *,
    maximum: int = 12,
) -> list[int]:
    by_category: Dict[str, list[int]] = {}
    for sample_index in selected:
        labels = list(categories.get(int(sample_index), ("diagnostic_boundary",)))
        by_category.setdefault(labels[0], []).append(int(sample_index))
    output: list[int] = []
    while len(output) < min(int(maximum), len(selected)):
        progress = False
        for category in sorted(by_category):
            values = by_category[category]
            if values:
                sample_index = values.pop(0)
                if sample_index not in output:
                    output.append(sample_index)
                    progress = True
                    if len(output) >= min(int(maximum), len(selected)):
                        break
        if not progress:
            break
    return output


def _repeat_metadata(
    metadata: Mapping[str, object], count: int
) -> Dict[str, object]:
    repeated: Dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            if int(value.size(0)) != 1:
                raise ValueError(f"XAI metadata {key} is not batch-one.")
            repeats = (int(count), *(1 for _ in range(value.ndim - 1)))
            repeated[key] = value.repeat(repeats)
        else:
            repeated[key] = value
    return repeated


def _deterministic_occlusion_saliency(
    *,
    keeper: nn.Module,
    candidate: MutualChannelPatchAdapter,
    image: Tensor,
    metadata: Mapping[str, object],
    base_margin: Tensor,
    device: torch.device,
    batch_size: int,
    grid: int = 4,
) -> Tensor:
    if image.shape[0] != 1 or image.ndim != 4:
        raise ValueError("Occlusion saliency requires one BCHW input.")
    height, width = int(image.size(-2)), int(image.size(-1))
    occluded = image.detach().repeat(int(grid * grid), 1, 1, 1)
    for row in range(int(grid)):
        y0 = row * height // int(grid)
        y1 = (row + 1) * height // int(grid)
        for column in range(int(grid)):
            x0 = column * width // int(grid)
            x1 = (column + 1) * width // int(grid)
            occluded[row * int(grid) + column, :, y0:y1, x0:x1] = 0.0
    margins = []
    with torch.inference_mode():
        for start in range(0, int(occluded.size(0)), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), int(occluded.size(0)))
            metadata_batch = _repeat_metadata(metadata, stop - start)
            raw_logits, features = _forward_classification_with_metadata(
                keeper,
                occluded[start:stop],
                metadata_batch,
                device=device,
            )
            if not isinstance(features, Mapping):
                raise ValueError("Occlusion keeper forward returned no features.")
            patches, patch_indices, token_valid = _feature_tensors(features)
            result = candidate.forward_sparse(
                patches.float(), patch_indices, token_valid, raw_logits.float()
            )
            restricted = torch.tensor(
                RESTRICTED_NEGATIVE_CLASSES,
                device=device,
                dtype=torch.long,
            )
            margin = (
                result["logits"][:, FOCUS_CLASS]
                - result["logits"].index_select(1, restricted).amax(dim=1)
            )
            margins.append(margin)
    values = torch.cat(margins)
    coarse = (base_margin.detach().reshape(1) - values).abs().reshape(
        1, 1, int(grid), int(grid)
    )
    return F.interpolate(
        coarse,
        size=(GRID_HEIGHT, GRID_WIDTH),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _render_xai_pages(
    records: Sequence[Mapping[str, object]], output_dir: Path
) -> list[str]:
    pages = []
    columns = (
        "image",
        "valid",
        "ctl_c1_0",
        "ctl_c1_1",
        "ctl_c1_2",
        "ctl_group",
        "cand_c1_0",
        "cand_c1_1",
        "cand_c1_2",
        "cand_group",
        "saliency",
    )
    width, height = 150, 185
    for page_index, start in enumerate(range(0, len(records), 3)):
        page_records = records[start : start + 3]
        canvas = Image.new(
            "RGB", (len(columns) * width, len(page_records) * height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_number, record in enumerate(page_records):
            rgb = np.asarray(record["rgb"], dtype=np.uint8)
            maps = record["maps"]
            for column, name in enumerate(columns):
                if name == "image":
                    panel = Image.fromarray(rgb)
                else:
                    panel = _heat_overlay(rgb, np.asarray(maps[name]), alpha=0.55)
                panel = panel.resize((width, width), Image.Resampling.BILINEAR)
                x = column * width
                y = row_number * height
                canvas.paste(panel, (x, y))
                draw.text((x + 3, y + width + 2), name, fill="black")
            label = (
                f"idx={record['sample_index']} y={record['target']} "
                f"margin ctl/cand={record['control_margin']:.4f}/"
                f"{record['candidate_margin']:.4f}"
            )
            draw.text((3, row_number * height + width + 18), label, fill="black")
        path = output_dir / f"mcl_xai_page_{page_index:03d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages


def _xai_audit(
    *,
    keeper: nn.Module,
    control: MutualChannelPatchAdapter,
    candidate: MutualChannelPatchAdapter,
    dataset: MangoYOLOCropDataset,
    transform,
    raw_clean: Sequence[Mapping[str, object]],
    control_clean: Sequence[Mapping[str, object]],
    candidate_clean: Sequence[Mapping[str, object]],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    selected, categories, required = _required_xai_indices(
        raw_clean, control_clean, candidate_clean, minimum_rows=12
    )
    saliency_indices = set(_saliency_subset(selected, categories, maximum=12))
    keeper = keeper.to(device).eval()
    control_device = copy.deepcopy(control).to(device).eval()
    candidate_device = copy.deepcopy(candidate).to(device).eval()
    for module in (control_device, candidate_device):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    records = []
    arrays: Dict[str, np.ndarray] = {}
    all_maps_finite = True
    all_defined_nonzero = True
    for position, sample_index in enumerate(selected):
        loader, _summary = _make_loader(
            base_dataset=dataset,
            transform=transform,
            indices=[sample_index],
            batch_size=1,
            num_workers=0,
            context=f"mcl_xai_{sample_index}",
            seed=SEED + 500 + position,
        )
        images_cpu, targets_cpu, metadata_cpu = next(iter(loader))
        images = images_cpu.to(device)
        need_saliency = sample_index in saliency_indices
        with torch.inference_mode():
            raw_logits, features = _forward_classification_with_metadata(
                keeper, images, metadata_cpu, device=device
            )
        if not isinstance(features, Mapping):
            raise ValueError("XAI keeper forward returned no features.")
        patches, patch_indices, token_valid = _feature_tensors(features)
        control_result = control_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )
        candidate_result = candidate_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )
        valid = candidate_result["valid_mask"][0, 0].detach().float()
        control_prob = spatial_channel_probabilities(
            control_result["projected"], control_result["valid_mask"]
        )[0]
        candidate_prob = spatial_channel_probabilities(
            candidate_result["projected"], candidate_result["valid_mask"]
        )[0]
        control_class1 = control_prob[
            FOCUS_CLASS * CHANNELS_PER_CLASS : (FOCUS_CLASS + 1)
            * CHANNELS_PER_CLASS
        ]
        candidate_class1 = candidate_prob[
            FOCUS_CLASS * CHANNELS_PER_CLASS : (FOCUS_CLASS + 1)
            * CHANNELS_PER_CLASS
        ]
        restricted = torch.tensor(
            RESTRICTED_NEGATIVE_CLASSES, device=device, dtype=torch.long
        )
        control_margin = (
            control_result["residual_logits"][:, FOCUS_CLASS]
            - control_result["residual_logits"].index_select(1, restricted).amax(
                dim=1
            )
        )
        candidate_margin = (
            candidate_result["residual_logits"][:, FOCUS_CLASS]
            - candidate_result["residual_logits"].index_select(1, restricted).amax(
                dim=1
            )
        )
        if need_saliency:
            deployed_margin = (
                candidate_result["logits"][:, FOCUS_CLASS]
                - candidate_result["logits"].index_select(1, restricted).amax(dim=1)
            )
            saliency = _deterministic_occlusion_saliency(
                keeper=keeper,
                candidate=candidate_device,
                image=images,
                metadata=metadata_cpu,
                base_margin=deployed_margin,
                device=device,
                batch_size=int(args.xai_batch_size),
            )
        else:
            saliency = torch.zeros_like(valid)
        source_maps = {
            "valid": valid,
            "ctl_c1_0": control_class1[0],
            "ctl_c1_1": control_class1[1],
            "ctl_c1_2": control_class1[2],
            "ctl_group": control_class1.amax(dim=0),
            "cand_c1_0": candidate_class1[0],
            "cand_c1_1": candidate_class1[1],
            "cand_c1_2": candidate_class1[2],
            "cand_group": candidate_class1.amax(dim=0),
            "saliency": saliency,
        }
        map_values = {
            key: _normalize_map(value) for key, value in source_maps.items()
        }
        finite = all(np.isfinite(value).all() for value in map_values.values())
        nonzero = all(
            bool(torch.count_nonzero(value).item())
            for key, value in source_maps.items()
            if key != "saliency" or need_saliency
        )
        all_maps_finite = all_maps_finite and finite
        all_defined_nonzero = all_defined_nonzero and nonzero
        rgb = _rgb_from_tensor(images_cpu[0], mean=mean, std=std)
        record = {
            "sample_index": int(sample_index),
            "target": int(targets_cpu[0].item()),
            "categories": list(categories.get(sample_index, ())),
            "control_margin": float(control_margin.detach().item()),
            "candidate_margin": float(candidate_margin.detach().item()),
            "saliency_defined": need_saliency,
            "rgb": rgb,
            "maps": map_values,
        }
        records.append(record)
        arrays[f"sample_{sample_index}_rgb"] = rgb
        for key, value in map_values.items():
            arrays[f"sample_{sample_index}_{key}"] = value
    tensor_path = output_dir / "mutual_channel_xai_tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    pages = _render_xai_pages(records, output_dir)
    serializable_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"rgb", "maps"}
        }
        for record in records
    ]
    manifest = {
        "selected_indices": selected,
        "required_indices": required,
        "categories": {str(key): list(value) for key, value in categories.items()},
        "saliency_indices": sorted(saliency_indices),
        "required_coverage_exact": set(required).issubset(set(selected)),
        "minimum_twelve_when_available": len(selected)
        >= min(12, len(raw_clean)),
        "all_maps_finite": all_maps_finite,
        "all_defined_maps_nonzero": all_defined_nonzero,
        "input_saliency_method": (
            "deterministic_4x4_zero_occlusion_absolute_deployed_margin_drop"
        ),
        "selection_replay_exact": _required_xai_indices(
            raw_clean, control_clean, candidate_clean, minimum_rows=12
        )[0]
        == selected,
        "tensor_path": str(tensor_path.resolve()),
        "tensor_sha256": _sha256(tensor_path),
        "pages": pages,
        "records": serializable_records,
    }
    manifest_path = output_dir / "mutual_channel_xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    del control_device, candidate_device
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **manifest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, Mapping[str, object]]],
    holdout_mechanism: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    control_mechanism = holdout_mechanism["control"]
    candidate_mechanism = holdout_mechanism["candidate"]
    control_auc = control_mechanism["class1_residual_margin_auroc_vs_0_2_4"]
    candidate_auc = candidate_mechanism["class1_residual_margin_auroc_vs_0_2_4"]
    mechanism_checks = {
        "candidate_holdout_dis_lte_0p98x_control": float(
            candidate_mechanism["loss_dis"]
        )
        <= 0.98 * float(control_mechanism["loss_dis"]),
        "candidate_coverage_gte_control": float(
            candidate_mechanism["mean_group_coverage"]
        )
        >= float(control_mechanism["mean_group_coverage"]),
        "candidate_class1_entropy_gte_0p20": float(
            candidate_mechanism["class1_spatial_entropy"]
        )
        >= 0.20,
        "candidate_maximum_within_group_cosine_lte_0p98": float(
            candidate_mechanism["maximum_class1_within_group_cosine"]
        )
        <= 0.98,
        "candidate_direction_auroc_gte_0p70": candidate_auc is not None
        and float(candidate_auc) >= 0.70,
        "candidate_direction_auroc_gte_control": (
            candidate_auc is not None
            and control_auc is not None
            and float(candidate_auc) >= float(control_auc)
        ),
    }
    clean = comparisons["clean"]
    raw = clean["raw_candidate"]
    control = clean["control_candidate"]
    raw_delta = raw["delta"]
    control_delta = control["delta"]
    raw_transition = raw["transitions"]
    control_transition = control["transitions"]
    decision_checks = {
        "macro_f1_delta_vs_raw_gte_minus_0p002": float(raw_delta["macro_f1"])
        >= -0.002,
        "class1_f1_gain_vs_raw_gte_0p005": float(raw_delta["class1_f1"])
        >= 0.005,
        "class1_precision_gain_vs_raw_gte_0p005": float(
            raw_delta["class1_precision"]
        )
        >= 0.005,
        "class1_recall_delta_vs_raw_gte_minus_0p005": float(
            raw_delta["class1_recall"]
        )
        >= -0.005,
        "class1_tp_break_vs_raw_lte_1": int(raw_transition["focus_tp_break"]) <= 1,
        "class1_rescues_gte_breaks_vs_raw": int(
            raw_transition["focus_fn_rescue"]
        )
        >= int(raw_transition["focus_tp_break"]),
        "restricted_fp_reduction_vs_raw_gte_2": int(
            raw_transition["restricted_focus_fp_reduction"]
        )
        >= 2,
        "corrections_gt_harms_vs_raw": int(
            raw_transition["candidate_correction"]
        )
        > int(raw_transition["candidate_harm"]),
        "maximum_nonfocus_f1_drop_vs_raw_lte_0p010": float(
            raw["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "macro_f1_delta_vs_control_nonnegative": float(
            control_delta["macro_f1"]
        )
        >= 0.0,
        "class1_f1_gain_vs_control_gte_0p003": float(
            control_delta["class1_f1"]
        )
        >= 0.003,
        "class1_precision_gain_vs_control_gte_0p003": float(
            control_delta["class1_precision"]
        )
        >= 0.003,
        "class1_recall_delta_vs_control_gte_minus_0p005": float(
            control_delta["class1_recall"]
        )
        >= -0.005,
        "restricted_fp_reduction_vs_control_gte_1": int(
            control_transition["restricted_focus_fp_reduction"]
        )
        >= 1,
        "corrections_gt_harms_vs_control": int(
            control_transition["candidate_correction"]
        )
        > int(control_transition["candidate_harm"]),
    }
    lighting = [comparisons[name] for name, _, _ in LIGHTING_CONDITIONS]
    raw_f1 = [
        float(item["raw_candidate"]["delta"]["class1_f1"]) for item in lighting
    ]
    raw_precision = [
        float(item["raw_candidate"]["delta"]["class1_precision"])
        for item in lighting
    ]
    control_f1 = [
        float(item["control_candidate"]["delta"]["class1_f1"])
        for item in lighting
    ]
    control_precision = [
        float(item["control_candidate"]["delta"]["class1_precision"])
        for item in lighting
    ]
    raw_recall = [
        float(item["raw_candidate"]["delta"]["class1_recall"])
        for item in lighting
    ]
    control_recall = [
        float(item["control_candidate"]["delta"]["class1_recall"])
        for item in lighting
    ]
    illumination_checks = {
        "candidate_raw_class1_f1_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in raw_f1
        )
        >= 2,
        "candidate_raw_class1_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in raw_precision
        )
        >= 2,
        "candidate_control_class1_f1_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_f1
        )
        >= 2,
        "candidate_control_class1_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_precision
        )
        >= 2,
        "worst_class1_recall_delta_vs_raw_gte_minus_0p020": min(raw_recall)
        >= -0.020,
        "worst_class1_recall_delta_vs_control_gte_minus_0p020": min(
            control_recall
        )
        >= -0.020,
        "net_class1_tp_loss_lte_2_each_vs_raw": all(
            int(item["raw_candidate"]["transitions"]["focus_tp_break"])
            - int(item["raw_candidate"]["transitions"]["focus_fn_rescue"])
            <= 2
            for item in lighting
        ),
        "net_class1_tp_loss_lte_2_each_vs_control": all(
            int(item["control_candidate"]["transitions"]["focus_tp_break"])
            - int(item["control_candidate"]["transitions"]["focus_fn_rescue"])
            <= 2
            for item in lighting
        ),
        "aggregate_restricted_fp_reduction_positive_vs_raw": sum(
            int(
                item["raw_candidate"]["transitions"][
                    "restricted_focus_fp_reduction"
                ]
            )
            for item in lighting
        )
        > 0,
        "aggregate_restricted_fp_reduction_positive_vs_control": sum(
            int(
                item["control_candidate"]["transitions"][
                    "restricted_focus_fp_reduction"
                ]
            )
            for item in lighting
        )
        > 0,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **mechanism_checks,
        **decision_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "mechanism_checks": mechanism_checks,
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": not failed,
        "test_authorized": False,
        "full_train_authorized": False,
    }


def _write_training_curve(
    path: Path,
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    rows = [*control["history"], *candidate["history"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "epoch",
                "rows",
                "batches",
                "loss",
                "ce",
                "loss_dis",
                "loss_div",
                "coverage",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_mechanism_csv(
    path: Path,
    *,
    fit: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> None:
    fields = (
        "split",
        "variant",
        "rows",
        "loss_dis",
        "loss_div",
        "mean_group_coverage",
        "grouped_logit_accuracy",
        "class1_spatial_entropy",
        "maximum_class1_within_group_cosine",
        "pooled_channel_effective_rank",
        "class1_residual_margin_auroc_vs_0_2_4",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, variants in (
            ("fit", fit),
            *[(condition, holdout[condition]) for condition in CONDITIONS],
        ):
            for variant in ("control", "candidate"):
                source = variants[variant]
                writer.writerow(
                    {"split": split, "variant": variant, **{
                        key: source.get(key) for key in fields[2:]
                    }}
                )


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    clean = summary["comparisons"]["clean"]
    raw = clean["raw_candidate"]
    control = clean["control_candidate"]
    mechanism = summary["mechanism"]["holdout"]["clean"]
    gate = summary["gate"]
    lines = [
        "# Mutual-Channel Patch Adapter A0 Result",
        "",
        f"- Decision: {'PASS' if gate['all_gates_passed'] else 'REJECT'}.",
        f"- Failed gates: {gate['failed_checks']}.",
        (
            "- Clean candidate versus raw macro/class-1 F1 delta: "
            f"{raw['delta']['macro_f1']:.6f}/"
            f"{raw['delta']['class1_f1']:.6f}."
        ),
        (
            "- Clean candidate versus raw class-1 precision/recall delta: "
            f"{raw['delta']['class1_precision']:.6f}/"
            f"{raw['delta']['class1_recall']:.6f}."
        ),
        (
            "- Clean candidate versus control macro/class-1 F1 delta: "
            f"{control['delta']['macro_f1']:.6f}/"
            f"{control['delta']['class1_f1']:.6f}."
        ),
        (
            "- Holdout control/candidate L_dis: "
            f"{mechanism['control']['loss_dis']:.6f}/"
            f"{mechanism['candidate']['loss_dis']:.6f}."
        ),
        (
            "- Holdout control/candidate class-1 residual-margin AUROC: "
            f"{mechanism['control']['class1_residual_margin_auroc_vs_0_2_4']}/"
            f"{mechanism['candidate']['class1_residual_margin_auroc_vs_0_2_4']}."
        ),
        "- Validation/test data used: false/false.",
        "- Current-best command revision: unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    excluded = {"artifact_manifest.json"}
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in excluded:
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "forbidden_model_binary_count": sum(
            Path(str(row["path"])).suffix.lower()
            in {".pt", ".pth", ".ckpt", ".engine"}
            for row in artifacts
        ),
    }
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance, rows, cohorts, order = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "method": METHOD,
            "provenance": provenance,
            "cohort": {
                key: value
                for key, value in cohorts.items()
                if key not in {"fit_indices", "holdout_indices"}
            },
            "output_directory_created": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Locked MCL A0 requires CUDA.")
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("MCL output cannot be written under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)

    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    amp_dtype = _amp_dtype(device)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    keeper, control_prototype, candidate_prototype, construction = (
        _construct_keeper_and_adapters(checkpoint)
    )
    frozen_keeper_before = _state_sha256(keeper)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, paths["data"]
    )
    cache, cache_summary, resource_batch = _extract_fit_cache(
        keeper=keeper,
        dataset=dataset,
        transform=transform,
        indices=cohorts["fit_indices"],
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    first = _batch_from_cache(
        cache,
        _cache_positions(cache),
        order[:BATCH_SIZE],
        device=device,
    )
    with torch.inference_mode():
        initial_result = candidate_prototype.to(device).forward_sparse(
            first["patches"],
            first["patch_indices"],
            first["token_valid"],
            first["raw_logits"],
        )
    initial_raw_error = float(
        (initial_result["logits"] - first["raw_logits"]).abs().amax().item()
    )
    candidate_prototype = candidate_prototype.cpu()
    initial_gradient = _initial_gradient_audit(
        prototype=candidate_prototype,
        cache=cache,
        order=order,
        device=device,
    )
    control, control_training = _train_variant(
        name="control",
        prototype=control_prototype,
        cache=cache,
        order=order,
        candidate=False,
        args=args,
        device=device,
    )
    candidate, candidate_training = _train_variant(
        name="candidate",
        prototype=candidate_prototype,
        cache=cache,
        order=order,
        candidate=True,
        args=args,
        device=device,
    )
    fit_mechanism = {
        "control": _evaluate_cache_mechanism(
            adapter=control,
            cache=cache,
            device=device,
            batch_size=int(args.batch_size),
        ),
        "candidate": _evaluate_cache_mechanism(
            adapter=candidate,
            cache=cache,
            device=device,
            batch_size=int(args.batch_size),
        ),
    }
    (
        raw_conditions,
        control_conditions,
        candidate_conditions,
        holdout_mechanism,
        evaluation_summary,
    ) = _evaluate_conditions(
        keeper=keeper,
        control=control,
        candidate=candidate,
        dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"],
        rows=rows,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    comparisons = _build_comparisons(
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    raw_benchmark = _benchmark_inference(
        keeper=keeper,
        adapter=None,
        images_cpu=resource_batch[0],
        metadata_cpu=resource_batch[2],
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    candidate_benchmark = _benchmark_inference(
        keeper=keeper,
        adapter=candidate,
        images_cpu=resource_batch[0],
        metadata_cpu=resource_batch[2],
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    runtime_ratio = float(candidate_benchmark["median_seconds"]) / max(
        float(raw_benchmark["median_seconds"]), 1e-12
    )
    export = _export_diagnostics(
        keeper=keeper,
        candidate=candidate,
        cache=cache,
        resource_batch=resource_batch,
        output_dir=output_dir,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    xai = _xai_audit(
        keeper=keeper,
        control=control,
        candidate=candidate,
        dataset=dataset,
        transform=transform,
        raw_clean=raw_conditions["clean"],
        control_clean=control_conditions["clean"],
        candidate_clean=candidate_conditions["clean"],
        mean=mean,
        std=std,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    frozen_keeper_after = _state_sha256(keeper.cpu())
    equation = _equation_diagnostics()
    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_source_and_license_verified": bool(
            provenance["official_worktree_clean"]
            and provenance["official_license_present"]
            and provenance["official_commit"] == LOCKED_OFFICIAL_COMMIT
        ),
        "source_disjoint_split_exact": bool(
            cohorts["fit_rows"] == 7372
            and cohorts["holdout_rows"] == 1843
            and cohorts["source_overlap"] == 0
            and cohorts["fit_index_sha256"] == LOCKED_FIT_INDEX_SHA256
            and cohorts["holdout_index_sha256"] == LOCKED_HOLDOUT_INDEX_SHA256
        ),
        "train_order_and_mask_hash_exact": bool(
            provenance["train_order_sha256"] == LOCKED_TRAIN_ORDER_SHA256
            and provenance["mask_occurrence_sha256"]
            == LOCKED_MASK_OCCURRENCE_SHA256
        ),
        "keeper_parameter_count_exact": int(construction["keeper_parameters"])
        == EXPECTED_KEEPER_PARAMETERS,
        "adapter_parameter_count_exact": int(construction["adapter_parameters"])
        == EXPECTED_ADAPTER_PARAMETERS,
        "candidate_control_initial_state_bit_exact": bool(
            construction["candidate_control_bit_exact"]
        ),
        "candidate_control_initial_optimizer_bit_exact": bool(
            construction["candidate_control_initial_optimizer_bit_exact"]
        ),
        "isolated_initialization_rng_restored": bool(
            construction["isolated_initialization_rng_restored"]
        ),
        "initial_residual_head_zero_exact": bool(
            construction["residual_head_weight_zero"]
            and construction["residual_head_bias_zero"]
        ),
        "initial_raw_equivalence_bit_exact": initial_raw_error == 0.0,
        "equations_and_gradients_match_reference": bool(
            equation["all_equations_match_fp32_lte_1e6"]
        ),
        "candidate_mcl_projection_gradient_nonzero": bool(
            initial_gradient["candidate_projection_mcl_gradient_nonzero"]
        ),
        "control_mcl_diagnostics_detached": bool(
            initial_gradient["control_diagnostic_detached"]
            and initial_gradient[
                "control_detached_gradient_matches_ce_only_bit_exact"
            ]
        ),
        "both_execute_exact_epochs_and_occurrences": bool(
            control_training["epochs"] == EPOCHS
            and candidate_training["epochs"] == EPOCHS
            and control_training["train_rows"] == 73720
            and candidate_training["train_rows"] == 73720
        ),
        "both_all_gradients_finite_nonzero_seen": bool(
            control_training["all_gradients_finite"]
            and candidate_training["all_gradients_finite"]
            and control_training["all_trainable_gradients_seen"]
            and candidate_training["all_trainable_gradients_seen"]
        ),
        "both_all_trainable_parameters_changed": bool(
            control_training["all_trainable_parameters_changed"]
            and candidate_training["all_trainable_parameters_changed"]
        ),
        "keeper_frozen_bit_exact": frozen_keeper_before == frozen_keeper_after,
        "fit_and_holdout_mechanism_finite": bool(
            all(value["all_finite"] for value in fit_mechanism.values())
            and all(
                value["all_finite"]
                for condition in holdout_mechanism.values()
                for value in condition.values()
            )
        ),
        "clean_raw_replays_cidt_argmax": int(
            evaluation_summary["clean_raw_cidt_argmax_mismatches"]
        )
        == 0,
        "runtime_ratio_lte_1p10": runtime_ratio <= MAX_RUNTIME_RATIO,
        "candidate_peak_vram_lte_0p75_gib": float(
            candidate_benchmark["peak_vram_gib"]
        )
        <= MAX_PEAK_VRAM_GIB,
        "isolated_onnx_finite_error_lte_1e5_argmax_exact": bool(
            export["isolated"]["succeeded"]
            and export["isolated"]["finite"]
            and export["isolated"]["argmax_match"]
            and float(export["isolated"]["maximum_absolute_error"])
            <= MAX_ONNX_ERROR
        ),
        "full_onnx_finite_error_lte_1e5_argmax_exact": bool(
            export["full"]["succeeded"]
            and export["full"]["finite"]
            and export["full"]["argmax_match"]
            and float(export["full"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR
        ),
        "xai_required_coverage_exact": bool(
            xai["required_coverage_exact"] and xai["selection_replay_exact"]
        ),
        "xai_minimum_twelve_when_available": bool(
            xai["minimum_twelve_when_available"]
        ),
        "xai_maps_finite_nonzero": bool(
            xai["all_maps_finite"] and xai["all_defined_maps_nonzero"]
        ),
        "current_best_commands_hash_unchanged": provenance["sha256"][
            "current_commands"
        ]
        == LOCKED_CURRENT_COMMAND_SHA256,
        "command_history_hash_unchanged": provenance["sha256"]["command_history"]
        == LOCKED_COMMAND_HISTORY_SHA256,
        "validation_not_used": not bool(provenance["validation_predictions_used"]),
        "test_not_used": not bool(provenance["test_data_used"]),
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        comparisons=comparisons,
        holdout_mechanism=holdout_mechanism["clean"],
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    training_path = output_dir / "training_curve.csv"
    mechanism_path = output_dir / "mechanism_metrics.csv"
    mechanism_json_path = output_dir / "mechanism_metrics.json"
    replay_path = output_dir / "independent_replay.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_predictions(
        predictions_path,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    _write_training_curve(training_path, control_training, candidate_training)
    _write_mechanism_csv(
        mechanism_path, fit=fit_mechanism, holdout=holdout_mechanism
    )
    mechanism_json_path.write_text(
        json.dumps(
            {"fit": fit_mechanism, "holdout": holdout_mechanism},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    replay = _replay_predictions(predictions_path, expected=comparisons)
    if not bool(replay["comparisons_exact"]):
        raise RuntimeError("Independent MCL prediction replay differs.")
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "method": METHOD,
        "status": "passed" if gate["all_gates_passed"] else "rejected",
        "decision": (
            "Advance exact MCL adapter to Stage B."
            if gate["all_gates_passed"]
            else "Reject exact final-patch MCL residual route; no neighboring sweep."
        ),
        "provenance": provenance,
        "cohort": {
            key: value
            for key, value in cohorts.items()
            if key not in {"fit_indices", "holdout_indices"}
        },
        "dataset": dataset_summary,
        "feature_cache": cache_summary,
        "construction": construction,
        "equation_diagnostics": equation,
        "initial_gradient_audit": initial_gradient,
        "training": {
            "control": control_training,
            "candidate": candidate_training,
        },
        "mechanism": {
            "fit": fit_mechanism,
            "holdout": holdout_mechanism,
        },
        "comparisons": comparisons,
        "evaluation": evaluation_summary,
        "benchmark": {
            "raw": raw_benchmark,
            "candidate": candidate_benchmark,
            "runtime_ratio": runtime_ratio,
        },
        "onnx": export,
        "xai": xai,
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
        "current_best_commands_updated": False,
        "model_binary_written": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    manifest = _write_manifest(output_dir)
    return {**summary, "artifact_manifest": manifest}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
