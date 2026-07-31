from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_dinov3_pair_patch_stat_readiness import (
    FOCUS_CLASS,
    RIVAL_CLASSES,
    _assert_output_outside_train,
    _binary_metrics,
    _model_contract,
    _sha256,
    _write_csv,
    normalized_source_group,
)
from trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof import (
    _atomic_json,
    _labels_sha256,
    _path_rows_sha256,
    _source_groups_sha256,
    _validate_fold_assignment_contract,
    expand_pair_rows,
    six_stratum_weights,
)
from trkh.tools.precheck_dinov3_pair_patchstats_train_oof import (
    MODEL_NAME,
    _validate_checkpoint_contract,
    assign_global_source_folds,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B7_PREFIX_RESIDUAL_READINESS_20260731"
EXPECTED_DATA_SHA256 = (
    "fa9581d1a595134bd366105c3999170d3553712099e7222cf67f2339f84da156"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6"
)
EXPECTED_TRAIN_SAMPLES = 8278
EXPECTED_SOURCE_GROUPS = 7751
PREFIX_TOKENS = 5
REGISTER_TOKENS = 4
PATCH_TOKENS = 256
TOKEN_WIDTH = 384
GLOBAL_LOGIT_WIDTH = 5
PAIR_RIVALS = tuple(int(value) for value in RIVAL_CLASSES)
PAIR_COUNT = len(PAIR_RIVALS)
PROJECTION_WIDTH = 16
PAIR_EMBED_WIDTH = 16
DESCRIPTOR_WIDTH = 69
READOUT_HIDDEN = 16
FOLDS = 5
SEED = 20260731
HEAD_EPOCHS = 20
HEAD_BATCH_SIZE = 128
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
HEAD_GRAD_CLIP = 1.0
THRESHOLD = 0.5
SOFT_RECALL_TARGET = 0.95
DUAL_LEARNING_RATE = 0.05
DUAL_MAX = 10.0
AUGMENTED_LAGRANGIAN_RHO = 1.0
EXPECTED_HEAD_PARAMETERS = 7_345
MAX_HEAD_MACS_ALL_PAIRS = 50_000
EXPECTED_HEAD_MACS_ALL_PAIRS = 34_080
CACHE_MANIFEST_SCHEMA_VERSION = 2
PREFIX_CACHE_FILENAME = "prefix_tokens_f16.npy"
PATCH_MEAN_CACHE_FILENAME = "patch_mean_f16.npy"
GLOBAL_LOGITS_FILENAME = "global_logits_f32.npy"
LABELS_FILENAME = "labels_i64.npy"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"
SOURCE_GROUP_NORMALIZATION = r"filename_stem_strip_casefold_remove_regex:_box\d+$"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked B7 train-only OOF readiness audit for pair-conditioned "
            "DINOv3 CLS/register residuals. Validation/test are never built."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


class PairConditionedPrefixResidualReadout(nn.Module):
    """Capacity-matched readout over CLS/register residual summaries."""

    def __init__(
        self,
        *,
        token_width: int = TOKEN_WIDTH,
        prefix_tokens: int = PREFIX_TOKENS,
        register_tokens: int = REGISTER_TOKENS,
        projection_width: int = PROJECTION_WIDTH,
        pair_count: int = PAIR_COUNT,
        pair_embed_width: int = PAIR_EMBED_WIDTH,
        global_logit_width: int = GLOBAL_LOGIT_WIDTH,
        readout_hidden: int = READOUT_HIDDEN,
    ) -> None:
        super().__init__()
        self.token_width = int(token_width)
        self.prefix_tokens = int(prefix_tokens)
        self.register_tokens = int(register_tokens)
        self.projection_width = int(projection_width)
        self.pair_count = int(pair_count)
        self.pair_embed_width = int(pair_embed_width)
        self.global_logit_width = int(global_logit_width)
        self.readout_hidden = int(readout_hidden)
        if self.prefix_tokens != 1 + self.register_tokens:
            raise ValueError("Prefix layout must be CLS followed by registers")
        if self.projection_width != self.pair_embed_width:
            raise ValueError("Locked B7 projection and pair embedding widths differ")
        expected_descriptor = (
            3 * self.projection_width
            + self.global_logit_width
            + self.pair_embed_width
        )
        if expected_descriptor != DESCRIPTOR_WIDTH:
            raise ValueError(
                f"Locked descriptor width changed: {expected_descriptor}"
            )
        self.residual_norm = nn.LayerNorm(
            self.token_width,
            elementwise_affine=False,
        )
        self.shared_projection = nn.Linear(
            self.token_width,
            self.projection_width,
            bias=True,
        )
        self.pair_embedding = nn.Embedding(
            self.pair_count,
            self.pair_embed_width,
        )
        self.output = nn.Sequential(
            nn.Linear(DESCRIPTOR_WIDTH, self.readout_hidden, bias=True),
            nn.GELU(),
            nn.Linear(self.readout_hidden, 1, bias=True),
        )

    def descriptor(
        self,
        prefix_tokens: Tensor,
        patch_mean: Tensor,
        global_logits: Tensor,
        pair_ids: Tensor,
    ) -> Tensor:
        if prefix_tokens.ndim != 3 or tuple(prefix_tokens.shape[1:]) != (
            self.prefix_tokens,
            self.token_width,
        ):
            raise ValueError(
                "prefix_tokens must be [B,5,384], got "
                f"{tuple(prefix_tokens.shape)}"
            )
        if patch_mean.ndim != 2 or int(patch_mean.size(1)) != self.token_width:
            raise ValueError(
                f"patch_mean must be [B,384], got {tuple(patch_mean.shape)}"
            )
        if global_logits.ndim != 2 or int(global_logits.size(1)) != int(
            self.global_logit_width
        ):
            raise ValueError(
                f"global_logits must be [B,5], got {tuple(global_logits.shape)}"
            )
        pair_ids = pair_ids.reshape(-1).to(dtype=torch.long)
        batch = int(prefix_tokens.size(0))
        if not (
            batch
            == int(patch_mean.size(0))
            == int(global_logits.size(0))
            == int(pair_ids.numel())
        ):
            raise ValueError("B7 readout batch dimensions are inconsistent")
        if bool((pair_ids < 0).any()) or bool((pair_ids >= self.pair_count).any()):
            raise ValueError("pair_ids contain an out-of-range value")
        residual = prefix_tokens - patch_mean.unsqueeze(1)
        projected = self.shared_projection(self.residual_norm(residual))
        cls_projected = projected[:, 0]
        register_projected = projected[:, 1:]
        register_mean = register_projected.mean(dim=1)
        register_variance = (
            register_projected - register_mean.unsqueeze(1)
        ).square().mean(dim=1)
        pair_embedding = self.pair_embedding(pair_ids)
        descriptor = torch.cat(
            (
                cls_projected,
                register_mean,
                register_variance,
                global_logits,
                pair_embedding,
            ),
            dim=1,
        )
        if int(descriptor.size(1)) != DESCRIPTOR_WIDTH:
            raise RuntimeError("B7 descriptor width contract changed")
        return descriptor

    def forward_all_pairs(
        self,
        prefix_tokens: Tensor,
        patch_mean: Tensor,
        global_logits: Tensor,
    ) -> Tensor:
        """Evaluate all three pairs while projecting five residuals once."""
        if prefix_tokens.ndim != 3 or tuple(prefix_tokens.shape[1:]) != (
            self.prefix_tokens,
            self.token_width,
        ):
            raise ValueError("prefix_tokens must be [B,5,384]")
        if tuple(patch_mean.shape) != (
            int(prefix_tokens.size(0)),
            self.token_width,
        ):
            raise ValueError("patch_mean must align with prefix_tokens")
        if tuple(global_logits.shape) != (
            int(prefix_tokens.size(0)),
            self.global_logit_width,
        ):
            raise ValueError("global_logits must align with prefix_tokens")
        projected = self.shared_projection(
            self.residual_norm(prefix_tokens - patch_mean.unsqueeze(1))
        )
        cls_projected = projected[:, 0]
        register_projected = projected[:, 1:5]
        register_mean = register_projected.mean(dim=1)
        register_variance = (
            register_projected - register_mean.unsqueeze(1)
        ).square().mean(dim=1)
        batch = int(prefix_tokens.size(0))
        pair_embedding = self.pair_embedding(
            torch.arange(self.pair_count, device=prefix_tokens.device)
        )
        descriptor = torch.cat(
            (
                cls_projected[:, None, :].expand(-1, self.pair_count, -1),
                register_mean[:, None, :].expand(-1, self.pair_count, -1),
                register_variance[:, None, :].expand(-1, self.pair_count, -1),
                global_logits[:, None, :].expand(-1, self.pair_count, -1),
                pair_embedding[None, :, :].expand(batch, -1, -1),
            ),
            dim=2,
        )
        return self.output(descriptor).squeeze(2)

    def forward(
        self,
        prefix_tokens: Tensor,
        patch_mean: Tensor,
        global_logits: Tensor,
        pair_ids: Tensor,
    ) -> Tensor:
        return self.output(
            self.descriptor(prefix_tokens, patch_mean, global_logits, pair_ids)
        ).squeeze(1)


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def estimate_head_macs_all_pairs(
    *,
    prefix_tokens: int = PREFIX_TOKENS,
    token_width: int = TOKEN_WIDTH,
    projection_width: int = PROJECTION_WIDTH,
    pair_count: int = PAIR_COUNT,
    descriptor_width: int = DESCRIPTOR_WIDTH,
    readout_hidden: int = READOUT_HIDDEN,
) -> int:
    shared_projection = (
        int(prefix_tokens) * int(token_width) * int(projection_width)
    )
    pair_readouts = int(pair_count) * (
        int(descriptor_width) * int(readout_hidden) + int(readout_hidden)
    )
    return int(shared_projection + pair_readouts)


def _state_max_abs_error(first: nn.Module, second: nn.Module) -> float:
    errors = [
        float(
            (first.state_dict()[name] - second.state_dict()[name])
            .abs()
            .max()
            .item()
        )
        for name in first.state_dict()
    ]
    return max(errors, default=math.inf)


def architecture_self_check(*, seed: int = SEED) -> Dict[str, object]:
    models: List[PairConditionedPrefixResidualReadout] = []
    for _name in ("candidate", "control", "deranged"):
        torch.manual_seed(int(seed))
        models.append(PairConditionedPrefixResidualReadout().float().eval())
    candidate, control, deranged = models
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    patch_mean = torch.randn(
        6, TOKEN_WIDTH, generator=generator, dtype=torch.float32
    )
    residual = torch.randn(
        6, PREFIX_TOKENS, TOKEN_WIDTH, generator=generator, dtype=torch.float32
    )
    prefixes = patch_mean.unsqueeze(1) + residual
    logits = torch.randn(
        6, GLOBAL_LOGIT_WIDTH, generator=generator, dtype=torch.float32
    )
    pair_ids = torch.arange(6, dtype=torch.long) % PAIR_COUNT
    control_prefixes = patch_mean.unsqueeze(1).expand(-1, PREFIX_TOKENS, -1)
    register_permutation = torch.as_tensor([1, 3, 0, 2], dtype=torch.long)
    register_permuted = prefixes.clone()
    register_permuted[:, 1:5] = prefixes[:, 1:5][:, register_permutation]
    with torch.inference_mode():
        candidate_logits = candidate(prefixes, patch_mean, logits, pair_ids)
        register_permuted_logits = candidate(
            register_permuted, patch_mean, logits, pair_ids
        )
        control_logits = control(
            control_prefixes, patch_mean, logits, pair_ids
        )
        constant_candidate_logits = candidate(
            control_prefixes, patch_mean, logits, pair_ids
        )
        descriptor = candidate.descriptor(
            prefixes, patch_mean, logits, pair_ids
        )
        all_pairs = candidate.forward_all_pairs(prefixes, patch_mean, logits)
        one_pair_at_a_time = torch.stack(
            [
                candidate(
                    prefixes,
                    patch_mean,
                    logits,
                    torch.full(
                        (int(prefixes.size(0)),),
                        pair_index,
                        dtype=torch.long,
                    ),
                )
                for pair_index in range(PAIR_COUNT)
            ],
            dim=1,
        )
    parameters = [count_trainable_parameters(model) for model in models]
    return {
        "candidate_parameters": int(parameters[0]),
        "control_parameters": int(parameters[1]),
        "deranged_parameters": int(parameters[2]),
        "all_parameter_counts_equal": len(set(parameters)) == 1,
        "candidate_control_initial_state_max_abs_error": _state_max_abs_error(
            candidate, control
        ),
        "candidate_deranged_initial_state_max_abs_error": _state_max_abs_error(
            candidate, deranged
        ),
        "zero_residual_candidate_control_max_abs_error": float(
            (constant_candidate_logits - control_logits).abs().max().item()
        ),
        "all_pairs_path_max_abs_error": float(
            (all_pairs - one_pair_at_a_time).abs().max().item()
        ),
        "register_permutation_max_abs_error": float(
            (candidate_logits - register_permuted_logits).abs().max().item()
        ),
        "register_variance_unbiased": False,
        "descriptor_shape": list(descriptor.shape),
        "output_shape": list(candidate_logits.shape),
        "estimated_head_macs_all_pairs": estimate_head_macs_all_pairs(),
        "layer_norm_elementwise_affine": bool(
            candidate.residual_norm.elementwise_affine
        ),
    }


def source_disjoint_block_derangement(
    sample_indices: np.ndarray,
    source_groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    if indices.size < 2 or np.unique(indices).size != indices.size:
        raise ValueError("Derangement indices must be unique and contain >=2 rows")
    if bool((indices < 0).any()) or bool((indices >= groups.size).any()):
        raise IndexError("Derangement indices are outside source-group rows")
    rng = np.random.default_rng(int(seed))
    # Group rows contiguously in a random group order, then rotate by the
    # largest group. Because no group occupies more than half the partition,
    # this is a source-disjoint bijection. Labels are deliberately not input.
    local_groups: Dict[str, List[int]] = {}
    for sample_index in indices.tolist():
        local_groups.setdefault(str(groups[sample_index]), []).append(
            int(sample_index)
        )
    group_keys = list(local_groups)
    rng.shuffle(group_keys)
    ordered: List[int] = []
    maximum_group_size = 0
    for key in group_keys:
        members = np.asarray(local_groups[key], dtype=np.int64)
        rng.shuffle(members)
        ordered.extend(int(value) for value in members.tolist())
        maximum_group_size = max(maximum_group_size, int(members.size))
    if maximum_group_size * 2 > indices.size:
        raise ValueError(
            "A source-group-safe derangement is impossible: one group "
            "occupies more than half of the fold partition"
        )
    ordered_array = np.asarray(ordered, dtype=np.int64)
    source_order = np.roll(ordered_array, -maximum_group_size)
    mapping = {
        int(target): int(source)
        for target, source in zip(ordered_array.tolist(), source_order.tolist())
    }
    sources = np.asarray(
        [mapping[int(target)] for target in indices.tolist()], dtype=np.int64
    )
    if bool(np.any(sources == indices)):
        raise RuntimeError("B7 prefix derangement contains a fixed point")
    allowed = set(indices.tolist())
    if any(int(value) not in allowed for value in sources.tolist()):
        raise RuntimeError("B7 prefix derangement escaped its fold partition")
    if any(
        str(groups[int(source)]) == str(groups[int(target)])
        for target, source in zip(indices.tolist(), sources.tolist())
    ):
        raise RuntimeError("B7 derangement retained a normalized source group")
    if set(sources.tolist()) != allowed:
        raise RuntimeError("B7 block derangement is not bijective")
    return sources


def build_derangement_source_map(
    *,
    sample_count: int,
    fit_indices: np.ndarray,
    hold_indices: np.ndarray,
    source_groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    fit = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    hold = np.asarray(hold_indices, dtype=np.int64).reshape(-1)
    if set(fit.tolist()) & set(hold.tolist()):
        raise ValueError("Fit and hold indices overlap")
    if set(np.concatenate((fit, hold)).tolist()) != set(range(int(sample_count))):
        raise ValueError("Fit/hold indices do not partition the sample rows")
    source_map = np.full(
        (int(sample_count), PREFIX_TOKENS), -1, dtype=np.int64
    )
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    if groups.size != int(sample_count):
        raise ValueError("Source groups do not align with derangement rows")
    fit_donors = source_disjoint_block_derangement(
        fit, groups, seed=int(seed)
    )
    hold_donors = source_disjoint_block_derangement(
        hold, groups, seed=int(seed) + 100_003
    )
    source_map[fit] = fit_donors[:, None]
    source_map[hold] = hold_donors[:, None]
    fixed_points = int(
        np.sum(source_map == np.arange(int(sample_count))[:, None])
    )
    fit_allowed = set(fit.tolist())
    hold_allowed = set(hold.tolist())
    fit_crossings = int(
        sum(int(value) not in fit_allowed for value in source_map[fit].reshape(-1))
    )
    hold_crossings = int(
        sum(
            int(value) not in hold_allowed
            for value in source_map[hold].reshape(-1)
        )
    )
    same_source_group = int(
        sum(
            str(groups[int(source)]) == str(groups[target])
            for target in range(int(sample_count))
            for source in source_map[target].tolist()
        )
    )
    bijective_contracts = int(
        sum(
            set(source_map[partition, 0].tolist()) == set(partition.tolist())
            for partition in (fit, hold)
        )
    )
    block_donor_consistent = bool(
        np.all(source_map == source_map[:, :1])
    )
    report = {
        "fit_rows": int(fit.size),
        "hold_rows": int(hold.size),
        "prefix_positions": PREFIX_TOKENS,
        "fixed_points": fixed_points,
        "fit_to_hold_crossings": fit_crossings,
        "hold_to_fit_crossings": hold_crossings,
        "same_source_group_assignments": same_source_group,
        "bijective_permutation_contracts": bijective_contracts,
        "expected_permutation_contracts": 2,
        "checked_prefix_cells": int(sample_count) * PREFIX_TOKENS,
        "block_donor_consistent": block_donor_consistent,
        "label_blind_construction": True,
    }
    if (
        fixed_points
        or fit_crossings
        or hold_crossings
        or same_source_group
        or bijective_contracts != 2
        or not block_donor_consistent
    ):
        raise RuntimeError("B7 derangement integrity failed")
    return source_map, report


class _PairPrefixDataset(Dataset):
    def __init__(
        self,
        *,
        prefix_cache: np.ndarray,
        patch_mean_cache: np.ndarray,
        global_logits: np.ndarray,
        rows: Mapping[str, np.ndarray],
        mode: str,
        derangement_sources: Optional[np.ndarray] = None,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"candidate", "control", "deranged"}:
            raise ValueError(f"Unsupported B7 readout mode: {mode}")
        self.prefix_cache = prefix_cache
        self.patch_mean_cache = patch_mean_cache
        self.global_logits = global_logits
        self.sample_indices = np.asarray(rows["sample_indices"], dtype=np.int64)
        self.pair_ids = np.asarray(rows["pair_ids"], dtype=np.int64)
        self.targets = np.asarray(rows["targets"], dtype=np.int64)
        self.weights = six_stratum_weights(self.pair_ids, self.targets)
        self.mode = normalized_mode
        self.derangement_sources = (
            None
            if derangement_sources is None
            else np.asarray(derangement_sources, dtype=np.int64)
        )
        sample_count = int(self.prefix_cache.shape[0])
        expected_shapes = (
            tuple(self.prefix_cache.shape[1:]) == (PREFIX_TOKENS, TOKEN_WIDTH),
            tuple(self.patch_mean_cache.shape) == (sample_count, TOKEN_WIDTH),
            tuple(self.global_logits.shape) == (sample_count, GLOBAL_LOGIT_WIDTH),
        )
        if not all(expected_shapes):
            raise ValueError("B7 cache arrays do not satisfy locked shapes")
        if not (
            self.sample_indices.size == self.pair_ids.size == self.targets.size
        ):
            raise ValueError("Expanded B7 rows are not aligned")
        if self.mode == "deranged" and (
            self.derangement_sources is None
            or tuple(self.derangement_sources.shape)
            != (sample_count, PREFIX_TOKENS)
        ):
            raise ValueError("Deranged mode requires a complete source map")

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, index: int):
        sample_index = int(self.sample_indices[index])
        patch_mean = np.asarray(
            self.patch_mean_cache[sample_index], dtype=np.float32
        )
        if self.mode == "candidate":
            prefixes = np.asarray(
                self.prefix_cache[sample_index], dtype=np.float32
            )
        elif self.mode == "control":
            prefixes = np.repeat(
                patch_mean[None, :], PREFIX_TOKENS, axis=0
            )
        else:
            assert self.derangement_sources is not None
            sources = self.derangement_sources[sample_index]
            residuals = np.stack(
                [
                    np.asarray(
                        self.prefix_cache[int(source), prefix_index],
                        dtype=np.float32,
                    )
                    - np.asarray(
                        self.patch_mean_cache[int(source)], dtype=np.float32
                    )
                    for prefix_index, source in enumerate(sources.tolist())
                ],
                axis=0,
            )
            prefixes = patch_mean[None, :] + residuals
        return (
            torch.from_numpy(np.array(prefixes, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(patch_mean, dtype=np.float32, copy=True)),
            torch.from_numpy(
                np.array(
                    self.global_logits[sample_index],
                    dtype=np.float32,
                    copy=True,
                )
            ),
            torch.as_tensor(int(self.pair_ids[index]), dtype=torch.long),
            torch.as_tensor(float(self.targets[index]), dtype=torch.float32),
            torch.as_tensor(float(self.weights[index]), dtype=torch.float32),
        )


def primal_dual_soft_recall_loss(
    *,
    logits: Tensor,
    targets: Tensor,
    pair_ids: Tensor,
    weights: Tensor,
    dual_values: Tensor,
    recall_target: float = SOFT_RECALL_TARGET,
    rho: float = AUGMENTED_LAGRANGIAN_RHO,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(dtype=logits.dtype)
    pair_ids = pair_ids.reshape(-1).to(dtype=torch.long)
    weights = weights.reshape(-1).to(dtype=logits.dtype)
    if not (
        logits.numel()
        == targets.numel()
        == pair_ids.numel()
        == weights.numel()
    ):
        raise ValueError("Primal-dual loss inputs are not aligned")
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weighted_bce = (elementwise * weights).sum() / weights.sum().clamp_min(
        1e-12
    )
    soft_recalls: List[Tensor] = []
    violations: List[Tensor] = []
    active_pairs: List[int] = []
    probabilities = torch.sigmoid(logits)
    for pair_index in range(PAIR_COUNT):
        positive = torch.logical_and(pair_ids == pair_index, targets > 0.5)
        if bool(positive.any()):
            recall = probabilities[positive].mean()
            soft_recalls.append(recall)
            violations.append(
                torch.as_tensor(
                    float(recall_target), device=logits.device, dtype=logits.dtype
                )
                - recall
            )
            active_pairs.append(pair_index)
    if not active_pairs:
        return (
            weighted_bce,
            weighted_bce,
            torch.empty(0, device=logits.device, dtype=logits.dtype),
            torch.empty(0, device=logits.device, dtype=torch.long),
        )
    violation_tensor = torch.stack(violations)
    active_tensor = torch.as_tensor(
        active_pairs, device=logits.device, dtype=torch.long
    )
    multipliers = dual_values[active_tensor].to(dtype=logits.dtype)
    constraint = (
        multipliers * violation_tensor
        + 0.5 * float(rho) * F.relu(violation_tensor).square()
    ).mean()
    return (
        weighted_bce + constraint,
        weighted_bce,
        violation_tensor,
        active_tensor,
    )


def update_dual_values(
    dual_values: Tensor,
    violations: Tensor,
    active_pairs: Tensor,
    *,
    learning_rate: float = DUAL_LEARNING_RATE,
    maximum: float = DUAL_MAX,
) -> Tensor:
    updated = dual_values.detach().clone()
    if int(active_pairs.numel()) > 0:
        indices = active_pairs.to(device=updated.device, dtype=torch.long)
        updated[indices] = torch.clamp(
            updated[indices]
            + float(learning_rate)
            * violations.detach().to(device=updated.device, dtype=updated.dtype),
            min=0.0,
            max=float(maximum),
        )
    return updated


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _train_readout(
    *,
    dataset: Dataset,
    device: torch.device,
    seed: int,
    epochs: int = HEAD_EPOCHS,
    batch_size: int = HEAD_BATCH_SIZE,
) -> Tuple[PairConditionedPrefixResidualReadout, Dict[str, object]]:
    _seed_everything(int(seed))
    model = PairConditionedPrefixResidualReadout().to(
        device=device, dtype=torch.float32
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    dual_values = torch.zeros(PAIR_COUNT, device=device, dtype=torch.float32)
    final_loss = math.nan
    final_bce = math.nan
    updates = 0
    active_constraint_updates = [0 for _ in range(PAIR_COUNT)]
    maximum_violation = -math.inf
    minimum_violation = math.inf
    final_epoch_violation_sum = [0.0 for _ in range(PAIR_COUNT)]
    final_epoch_violation_count = [0 for _ in range(PAIR_COUNT)]
    model.train()
    for _epoch in range(int(epochs)):
        loss_total = 0.0
        bce_total = 0.0
        steps = 0
        epoch_violation_sum = [0.0 for _ in range(PAIR_COUNT)]
        epoch_violation_count = [0 for _ in range(PAIR_COUNT)]
        for prefixes, patch_mean, logits, pair_ids, targets, weights in loader:
            prefixes = prefixes.to(device=device, dtype=torch.float32)
            patch_mean = patch_mean.to(device=device, dtype=torch.float32)
            logits = logits.to(device=device, dtype=torch.float32)
            pair_ids = pair_ids.to(device=device, dtype=torch.long)
            targets = targets.to(device=device, dtype=torch.float32)
            weights = weights.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(prefixes, patch_mean, logits, pair_ids)
            loss, weighted_bce, violations, active_pairs = (
                primal_dual_soft_recall_loss(
                    logits=predictions,
                    targets=targets,
                    pair_ids=pair_ids,
                    weights=weights,
                    dual_values=dual_values,
                )
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("B7 readout produced non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), HEAD_GRAD_CLIP
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("B7 readout produced non-finite gradients")
            optimizer.step()
            dual_values = update_dual_values(
                dual_values, violations, active_pairs
            )
            if int(active_pairs.numel()) > 0:
                for pair_index in active_pairs.detach().cpu().tolist():
                    active_constraint_updates[int(pair_index)] += 1
                for pair_index, violation in zip(
                    active_pairs.detach().cpu().tolist(),
                    violations.detach().cpu().tolist(),
                ):
                    epoch_violation_sum[int(pair_index)] += float(violation)
                    epoch_violation_count[int(pair_index)] += 1
                maximum_violation = max(
                    maximum_violation,
                    float(violations.detach().max().cpu().item()),
                )
                minimum_violation = min(
                    minimum_violation,
                    float(violations.detach().min().cpu().item()),
                )
            if not bool(torch.isfinite(dual_values).all()):
                raise FloatingPointError("B7 dual values became non-finite")
            loss_total += float(loss.detach().item())
            bce_total += float(weighted_bce.detach().item())
            steps += 1
            updates += 1
        final_loss = loss_total / max(1, steps)
        final_bce = bce_total / max(1, steps)
        final_epoch_violation_sum = epoch_violation_sum
        final_epoch_violation_count = epoch_violation_count
    model.eval()
    if updates <= 0 or not math.isfinite(final_loss):
        raise RuntimeError("B7 readout completed without a finite update")
    if (
        any(count <= 0 for count in active_constraint_updates)
        or not math.isfinite(maximum_violation)
        or not math.isfinite(minimum_violation)
    ):
        raise RuntimeError("B7 primal-dual constraints lacked positive support")
    return model, {
        "seed": int(seed),
        "data_order_seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "AdamW",
        "learning_rate": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "gradient_clip": HEAD_GRAD_CLIP,
        "loss_contract": (
            "six_stratum_weighted_bce_plus_augmented_primal_dual_soft_recall"
        ),
        "soft_recall_target": SOFT_RECALL_TARGET,
        "dual_learning_rate": DUAL_LEARNING_RATE,
        "dual_max": DUAL_MAX,
        "augmented_lagrangian_rho": AUGMENTED_LAGRANGIAN_RHO,
        "final_augmented_loss": float(final_loss),
        "final_weighted_bce": float(final_bce),
        "final_dual_values": [float(value) for value in dual_values.cpu()],
        "initial_dual_values": [0.0 for _ in range(PAIR_COUNT)],
        "dual_values_reset_to_zero": True,
        "active_constraint_updates_by_pair": active_constraint_updates,
        "maximum_soft_recall_violation": float(maximum_violation),
        "minimum_soft_recall_violation": float(minimum_violation),
        "dual_values_finite": bool(torch.isfinite(dual_values).all()),
        "final_epoch_mean_soft_recall_by_pair": [
            float(
                SOFT_RECALL_TARGET
                - final_epoch_violation_sum[pair_index]
                / max(1, final_epoch_violation_count[pair_index])
            )
            for pair_index in range(PAIR_COUNT)
        ],
        "final_epoch_mean_violation_by_pair": [
            float(
                final_epoch_violation_sum[pair_index]
                / max(1, final_epoch_violation_count[pair_index])
            )
            for pair_index in range(PAIR_COUNT)
        ],
        "final_epoch_active_batches_by_pair": final_epoch_violation_count,
        "successful_optimizer_updates": int(updates),
        "nonfinite_steps": 0,
    }


def _predict_readout(
    *,
    model: PairConditionedPrefixResidualReadout,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = HEAD_BATCH_SIZE,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    probabilities: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for prefixes, patch_mean, logits, pair_ids, _targets, _weights in loader:
            predictions = model(
                prefixes.to(device=device, dtype=torch.float32),
                patch_mean.to(device=device, dtype=torch.float32),
                logits.to(device=device, dtype=torch.float32),
                pair_ids.to(device=device, dtype=torch.long),
            )
            probabilities.append(torch.sigmoid(predictions).cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if result.size != len(dataset) or not bool(np.isfinite(result).all()):
        raise RuntimeError("B7 prediction coverage is incomplete or non-finite")
    return result


def _cache_paths(output_dir: Path) -> Dict[str, Path]:
    root = Path(output_dir)
    return {
        "prefixes": root / PREFIX_CACHE_FILENAME,
        "patch_mean": root / PATCH_MEAN_CACHE_FILENAME,
        "global_logits": root / GLOBAL_LOGITS_FILENAME,
        "labels": root / LABELS_FILENAME,
        "manifest": root / CACHE_MANIFEST_FILENAME,
    }


def _load_materialized_labels(path: Path) -> np.ndarray:
    """Load the small label cache without retaining a Windows file mapping."""

    return np.array(
        np.load(Path(path), allow_pickle=False),
        dtype=np.int64,
        copy=True,
    )


def _array_all_finite(array: np.ndarray, *, row_chunk: int = 256) -> bool:
    values = np.asarray(array)
    for start in range(0, int(values.shape[0]), max(1, int(row_chunk))):
        if not bool(
            np.isfinite(values[start : start + max(1, int(row_chunk))]).all()
        ):
            return False
    return True


def _cache_manifest_expected(
    *,
    paths: Sequence[str],
    source_groups: np.ndarray,
    expected_labels: np.ndarray,
) -> Dict[str, object]:
    return {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_content_sha256": _labels_sha256(expected_labels),
        "source_groups_sha256": _source_groups_sha256(
            source_groups.tolist()
        ),
        "source_group_count": int(np.unique(source_groups).size),
        "source_group_rows": int(source_groups.size),
        "source_group_normalization": SOURCE_GROUP_NORMALIZATION,
        "prefix_shape": [EXPECTED_TRAIN_SAMPLES, PREFIX_TOKENS, TOKEN_WIDTH],
        "prefix_dtype": "float16",
        "patch_mean_shape": [EXPECTED_TRAIN_SAMPLES, TOKEN_WIDTH],
        "patch_mean_dtype": "float16",
        "global_logits_shape": [EXPECTED_TRAIN_SAMPLES, GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [EXPECTED_TRAIN_SAMPLES],
        "labels_dtype": "int64",
        "prefix_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "prefix_tokens": PREFIX_TOKENS,
        "register_tokens": REGISTER_TOKENS,
        "patch_tokens": PATCH_TOKENS,
        "token_width": TOKEN_WIDTH,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }


def _validate_cache_manifest(
    *,
    manifest: Mapping[str, object],
    cache_paths: Mapping[str, Path],
    paths: Sequence[str],
    source_groups: np.ndarray,
    expected_labels: np.ndarray,
    verify_file_hashes: bool = True,
) -> None:
    expected = _cache_manifest_expected(
        paths=paths,
        source_groups=np.asarray(source_groups, dtype=object),
        expected_labels=np.asarray(expected_labels, dtype=np.int64),
    )
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B7 cache manifest mismatch: {mismatches}")
    required_hashes = {
        "prefixes_sha256": "prefixes",
        "patch_mean_sha256": "patch_mean",
        "global_logits_sha256": "global_logits",
        "labels_sha256": "labels",
    }
    for manifest_key, path_key in required_hashes.items():
        value = str(manifest.get(manifest_key, ""))
        if len(value) != 64:
            raise ValueError(f"B7 cache manifest lacks {manifest_key}")
        path = Path(cache_paths[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_file_hashes and _sha256(path) != value:
            raise ValueError(f"B7 cache hash mismatch for {path_key}")
    for numeric_key in (
        "maximum_pool_parity_error",
        "maximum_prefix_float16_quantization_error",
        "maximum_patch_mean_float16_quantization_error",
    ):
        value = float(manifest.get(numeric_key, math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"B7 cache manifest has invalid {numeric_key}")
    if float(manifest["maximum_pool_parity_error"]) > 1e-6:
        raise ValueError("B7 deployed-pool parity exceeds 1e-6")


def _load_existing_cache(
    *,
    output_dir: Path,
    paths: Sequence[str],
    source_groups: np.ndarray,
    expected_labels: np.ndarray,
) -> Optional[Dict[str, object]]:
    cache_paths = _cache_paths(output_dir)
    existing = [path.exists() for path in cache_paths.values()]
    if not any(existing):
        return None
    if not all(existing):
        raise RuntimeError(
            "Partial B7 cache exists; use a fresh output directory"
        )
    manifest = json.loads(cache_paths["manifest"].read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("B7 cache manifest must be a JSON object")
    _validate_cache_manifest(
        manifest=manifest,
        cache_paths=cache_paths,
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
    )
    arrays = {
        "prefixes": np.load(
            cache_paths["prefixes"], mmap_mode="r", allow_pickle=False
        ),
        "patch_mean": np.load(
            cache_paths["patch_mean"], mmap_mode="r", allow_pickle=False
        ),
        "global_logits": np.load(
            cache_paths["global_logits"], mmap_mode="r", allow_pickle=False
        ),
        "labels": np.load(
            cache_paths["labels"], mmap_mode="r", allow_pickle=False
        ),
    }
    array_contracts = {
        "prefixes": ((EXPECTED_TRAIN_SAMPLES, PREFIX_TOKENS, TOKEN_WIDTH), np.dtype(np.float16)),
        "patch_mean": ((EXPECTED_TRAIN_SAMPLES, TOKEN_WIDTH), np.dtype(np.float16)),
        "global_logits": ((EXPECTED_TRAIN_SAMPLES, GLOBAL_LOGIT_WIDTH), np.dtype(np.float32)),
        "labels": ((EXPECTED_TRAIN_SAMPLES,), np.dtype(np.int64)),
    }
    actual_mismatches = {
        key: {
            "expected_shape": list(expected_shape),
            "observed_shape": list(np.asarray(arrays[key]).shape),
            "expected_dtype": str(expected_dtype),
            "observed_dtype": str(np.asarray(arrays[key]).dtype),
        }
        for key, (expected_shape, expected_dtype) in array_contracts.items()
        if tuple(np.asarray(arrays[key]).shape) != tuple(expected_shape)
        or np.asarray(arrays[key]).dtype != expected_dtype
    }
    if actual_mismatches:
        raise ValueError(f"B7 cache array contract mismatch: {actual_mismatches}")
    if not np.array_equal(
        np.asarray(arrays["labels"], dtype=np.int64),
        np.asarray(expected_labels, dtype=np.int64),
    ):
        raise ValueError("B7 cached labels differ from canonical labels")
    return {**arrays, "manifest": dict(manifest)}


def _extract_train_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    paths: Sequence[str],
    source_groups: np.ndarray,
    expected_labels: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    cache_paths = _cache_paths(output_dir)
    partial = {
        key: path.with_suffix(path.suffix + ".partial")
        for key, path in cache_paths.items()
        if key != "manifest"
    }
    for path in partial.values():
        if path.exists():
            raise RuntimeError(
                f"A partial B7 cache exists; use a new output directory: {path}"
            )
    prefix_cache = np.lib.format.open_memmap(
        partial["prefixes"],
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), PREFIX_TOKENS, TOKEN_WIDTH),
    )
    patch_mean_cache = np.lib.format.open_memmap(
        partial["patch_mean"],
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), TOKEN_WIDTH),
    )
    logit_cache = np.lib.format.open_memmap(
        partial["global_logits"],
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), GLOBAL_LOGIT_WIDTH),
    )
    label_cache = np.lib.format.open_memmap(
        partial["labels"],
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    observed_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
    if observed_prefix_tokens != PREFIX_TOKENS:
        raise ValueError(
            f"B7 requires five prefix tokens, got {observed_prefix_tokens}"
        )
    cursor = 0
    maximum_pool_parity_error = 0.0
    maximum_prefix_quantization_error = 0.0
    maximum_patch_mean_quantization_error = 0.0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(
            loader, desc="b7-dinov3-prefix-cache-fp32", dynamic_ncols=True
        )
        for images, labels, _metadata in iterator:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            raw_tokens = model.forward_features(images)
            if not torch.is_tensor(raw_tokens) or raw_tokens.ndim != 3:
                raise TypeError("DINOv3 forward_features must return [B,N,D]")
            prefixes = raw_tokens.float()[:, :PREFIX_TOKENS]
            patches = raw_tokens.float()[:, PREFIX_TOKENS:]
            if tuple(prefixes.shape[1:]) != (PREFIX_TOKENS, TOKEN_WIDTH):
                raise ValueError(
                    f"Locked B7 prefix shape mismatch: {prefixes.shape}"
                )
            if tuple(patches.shape[1:]) != (PATCH_TOKENS, TOKEN_WIDTH):
                raise ValueError(
                    f"Locked B7 patch shape mismatch: {patches.shape}"
                )
            patch_mean = patches.mean(dim=1)
            pooled = model.forward_head(raw_tokens, pre_logits=True).float()
            parity_error = float(
                (pooled - patch_mean).abs().max().cpu().item()
            )
            maximum_pool_parity_error = max(
                maximum_pool_parity_error, parity_error
            )
            if parity_error > 1e-6:
                raise RuntimeError(
                    "B7 raw patch mean does not match deployed average pool: "
                    f"max_abs={parity_error}"
                )
            global_logits = model.head(pooled).float()
            prefix_numpy = prefixes.cpu().numpy().astype(np.float32, copy=False)
            mean_numpy = patch_mean.cpu().numpy().astype(np.float32, copy=False)
            logit_numpy = global_logits.cpu().numpy().astype(np.float32, copy=False)
            if not (
                np.isfinite(prefix_numpy).all()
                and np.isfinite(mean_numpy).all()
                and np.isfinite(logit_numpy).all()
            ):
                raise FloatingPointError("B7 extracted features are non-finite")
            prefix_quantized = prefix_numpy.astype(np.float16)
            mean_quantized = mean_numpy.astype(np.float16)
            maximum_prefix_quantization_error = max(
                maximum_prefix_quantization_error,
                float(
                    np.max(
                        np.abs(prefix_numpy - prefix_quantized.astype(np.float32))
                    )
                ),
            )
            maximum_patch_mean_quantization_error = max(
                maximum_patch_mean_quantization_error,
                float(
                    np.max(
                        np.abs(mean_numpy - mean_quantized.astype(np.float32))
                    )
                ),
            )
            count = int(labels.numel())
            end = cursor + count
            prefix_cache[cursor:end] = prefix_quantized
            patch_mean_cache[cursor:end] = mean_quantized
            logit_cache[cursor:end] = logit_numpy
            label_cache[cursor:end] = labels.cpu().numpy().astype(np.int64)
            cursor = end
    if cursor != len(dataset) or cursor != len(paths):
        raise RuntimeError(
            f"B7 cache coverage mismatch: cached={cursor}, dataset={len(dataset)}"
        )
    for array in (prefix_cache, patch_mean_cache, logit_cache, label_cache):
        array.flush()
    del prefix_cache, patch_mean_cache, logit_cache, label_cache
    # Labels are tiny, so materialize them before atomic promotion.  Keeping a
    # read-only memmap alive here prevents Path.replace() on Windows even after
    # ``del`` because NumPy may still own the underlying file mapping.
    extracted_labels = _load_materialized_labels(partial["labels"])
    if not np.array_equal(
        np.asarray(extracted_labels, dtype=np.int64),
        np.asarray(expected_labels, dtype=np.int64),
    ):
        raise ValueError("B7 extracted labels differ from canonical labels")
    del extracted_labels
    for key, path in partial.items():
        path.replace(cache_paths[key])
    manifest: Dict[str, object] = {
        **_cache_manifest_expected(
            paths=paths,
            source_groups=source_groups,
            expected_labels=expected_labels,
        ),
        "maximum_pool_parity_error": float(maximum_pool_parity_error),
        "maximum_prefix_float16_quantization_error": float(
            maximum_prefix_quantization_error
        ),
        "maximum_patch_mean_float16_quantization_error": float(
            maximum_patch_mean_quantization_error
        ),
        "prefixes_sha256": _sha256(cache_paths["prefixes"]),
        "patch_mean_sha256": _sha256(cache_paths["patch_mean"]),
        "global_logits_sha256": _sha256(cache_paths["global_logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
    }
    _atomic_json(cache_paths["manifest"], manifest)
    _validate_cache_manifest(
        manifest=manifest,
        cache_paths=cache_paths,
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
    )
    loaded = _load_existing_cache(
        output_dir=output_dir,
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
    )
    if loaded is None:
        raise RuntimeError("B7 cache disappeared after extraction")
    return loaded


def _load_or_extract_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    paths: Sequence[str],
    source_groups: np.ndarray,
    expected_labels: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    existing = _load_existing_cache(
        output_dir=output_dir,
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
    )
    if existing is not None:
        return existing
    return _extract_train_cache(
        model=model,
        dataset=dataset,
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )


def _runtime_representation_contract(model: nn.Module) -> Dict[str, object]:
    import timm

    prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
    token_width = int(
        getattr(model, "num_features", getattr(model, "embed_dim", 0)) or 0
    )
    patch_embed = getattr(model, "patch_embed", None)
    patch_tokens = int(getattr(patch_embed, "num_patches", 0) or 0)
    global_pool = str(getattr(model, "global_pool", ""))
    cls_token = getattr(model, "cls_token", None)
    reg_token = getattr(model, "reg_token", None)
    cls_shape = list(cls_token.shape) if torch.is_tensor(cls_token) else []
    reg_shape = list(reg_token.shape) if torch.is_tensor(reg_token) else []
    model.eval()
    with torch.inference_mode():
        synthetic = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        raw_tokens = model.forward_features(synthetic)
        if not torch.is_tensor(raw_tokens):
            raise TypeError("B7 synthetic forward_features did not return a tensor")
        pooled = model.forward_head(raw_tokens, pre_logits=True).float()
        synthetic_patch_mean = raw_tokens.float()[:, PREFIX_TOKENS:].mean(dim=1)
        synthetic_pool_parity_error = float(
            (pooled - synthetic_patch_mean).abs().max().cpu().item()
        )
        synthetic_logits = model.head(pooled).float()
    observed = {
        "runtime_class_module": type(model).__module__,
        "runtime_class_name": type(model).__name__,
        "timm_version": str(timm.__version__),
        "num_prefix_tokens": prefix_tokens,
        "cls_token_shape": cls_shape,
        "reg_token_shape": reg_shape,
        "cls_slice": [0, 1],
        "register_slice": [1, 5],
        "register_tokens": max(0, prefix_tokens - 1),
        "patch_tokens": patch_tokens,
        "token_width": token_width,
        "global_pool": global_pool,
        "synthetic_forward_features_shape": list(raw_tokens.shape),
        "synthetic_pre_logits_shape": list(pooled.shape),
        "synthetic_global_logits_shape": list(synthetic_logits.shape),
        "synthetic_pool_parity_error": synthetic_pool_parity_error,
        "synthetic_tensors_finite": bool(
            torch.isfinite(raw_tokens).all()
            and torch.isfinite(pooled).all()
            and torch.isfinite(synthetic_logits).all()
        ),
        "register_variance_unbiased": False,
    }
    expected = {
        "runtime_class_module": "timm.models.eva",
        "runtime_class_name": "Eva",
        "timm_version": "1.0.27",
        "num_prefix_tokens": PREFIX_TOKENS,
        "cls_token_shape": [1, 1, TOKEN_WIDTH],
        "reg_token_shape": [1, REGISTER_TOKENS, TOKEN_WIDTH],
        "register_tokens": REGISTER_TOKENS,
        "patch_tokens": PATCH_TOKENS,
        "token_width": TOKEN_WIDTH,
        "global_pool": "avg",
        "synthetic_forward_features_shape": [
            1,
            PREFIX_TOKENS + PATCH_TOKENS,
            TOKEN_WIDTH,
        ],
        "synthetic_pre_logits_shape": [1, TOKEN_WIDTH],
        "synthetic_global_logits_shape": [1, GLOBAL_LOGIT_WIDTH],
        "synthetic_tensors_finite": True,
        "register_variance_unbiased": False,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f"Locked B7 representation mismatch: {mismatches}")
    if synthetic_pool_parity_error > 1e-6:
        raise ValueError(
            "Locked B7 synthetic deployed-pool parity exceeds 1e-6: "
            f"{synthetic_pool_parity_error}"
        )
    return {"expected": expected, "observed": observed, "mismatches": mismatches}


def _prefixed_metrics(
    row: Dict[str, object], prefix: str, metrics: Mapping[str, float]
) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = float(value)


def _validate_oof_prediction_coverage(
    *,
    prediction_rows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    source_groups: np.ndarray,
    fold_assignments: np.ndarray,
    paths: Sequence[str],
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    assignments = np.asarray(fold_assignments, dtype=np.int64).reshape(-1)
    expected_keys = set()
    for sample_index, target in enumerate(labels.tolist()):
        rivals = (
            PAIR_RIVALS
            if int(target) == int(FOCUS_CLASS)
            else (int(target),)
            if int(target) in PAIR_RIVALS
            else ()
        )
        for rival in rivals:
            expected_keys.add((sample_index, f"{int(rival)}-1"))
    observed_keys = []
    finite = True
    aligned = True
    for row in prediction_rows:
        sample_index = int(row["sample_index"])
        key = (sample_index, str(row["pair"]))
        observed_keys.append(key)
        if sample_index < 0 or sample_index >= labels.size:
            aligned = False
            continue
        aligned = aligned and (
            int(row["fold"]) == int(assignments[sample_index])
            and int(row["target_index"]) == int(labels[sample_index])
            and str(row["source_group"]) == str(groups[sample_index])
            and str(row["image_path"]) == str(paths[sample_index])
        )
        probabilities = [
            float(row[f"{name}_probability_class1"])
            for name in ("control", "candidate", "deranged")
        ]
        finite = finite and bool(np.isfinite(probabilities).all())
    unique_keys = set(observed_keys)
    if len(observed_keys) != len(unique_keys):
        raise RuntimeError("B7 OOF predictions contain duplicated keys")
    if unique_keys != expected_keys:
        raise RuntimeError("B7 OOF prediction coverage differs from contract")
    if not aligned or not finite:
        raise RuntimeError("B7 OOF predictions are misaligned or non-finite")
    return {
        "expected_prediction_rows": int(len(expected_keys)),
        "observed_prediction_rows": int(len(observed_keys)),
        "unique_prediction_keys": int(len(unique_keys)),
        "prediction_alignment_exact": bool(aligned),
        "probabilities_finite": bool(finite),
    }


def run_prefix_residual_oof(
    *,
    prefix_cache: np.ndarray,
    patch_mean_cache: np.ndarray,
    global_logits: np.ndarray,
    labels: np.ndarray,
    source_groups: np.ndarray,
    fold_assignments: np.ndarray,
    paths: Sequence[str],
    output_dir: Path,
    device: torch.device,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    assignments = np.asarray(fold_assignments, dtype=np.int64).reshape(-1)
    if not (
        int(prefix_cache.shape[0])
        == int(patch_mean_cache.shape[0])
        == int(global_logits.shape[0])
        == labels.size
        == groups.size
        == assignments.size
        == len(paths)
    ):
        raise ValueError("B7 OOF arrays have inconsistent row counts")
    fold_assignment_integrity = _validate_fold_assignment_contract(
        groups, assignments
    )
    model_dir = Path(output_dir) / "fold_readouts"
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    derangement_reports: List[Dict[str, object]] = []
    training_telemetry_reports: List[Dict[str, object]] = []
    for fold_index in range(FOLDS):
        fit_indices = np.flatnonzero(assignments != int(fold_index))
        hold_indices = np.flatnonzero(assignments == int(fold_index))
        fit_groups = set(groups[fit_indices].tolist())
        hold_groups = set(groups[hold_indices].tolist())
        overlap = fit_groups & hold_groups
        if overlap:
            raise RuntimeError(f"B7 fold {fold_index} has source overlap")
        fold_seed = SEED + int(fold_index)
        derangement_sources, derangement_report = build_derangement_source_map(
            sample_count=labels.size,
            fit_indices=fit_indices,
            hold_indices=hold_indices,
            source_groups=groups,
            seed=fold_seed,
        )
        derangement_path = model_dir / f"fold_{fold_index:02d}_derangement.npy"
        np.save(derangement_path, derangement_sources, allow_pickle=False)
        derangement_report = {
            **derangement_report,
            "fold": int(fold_index),
            "seed_fit": int(fold_seed),
            "seed_hold": int(fold_seed + 100_003),
            "source_map_sha256": _sha256(derangement_path),
            "source_map_path": str(derangement_path),
        }
        derangement_reports.append(derangement_report)
        fit_rows = expand_pair_rows(labels, fit_indices)
        hold_rows = expand_pair_rows(labels, hold_indices)
        datasets_fit = {
            mode: _PairPrefixDataset(
                prefix_cache=prefix_cache,
                patch_mean_cache=patch_mean_cache,
                global_logits=global_logits,
                rows=fit_rows,
                mode=mode,
                derangement_sources=(
                    derangement_sources if mode == "deranged" else None
                ),
            )
            for mode in ("candidate", "control", "deranged")
        }
        trained: Dict[str, PairConditionedPrefixResidualReadout] = {}
        telemetry: Dict[str, Dict[str, object]] = {}
        for mode in ("candidate", "control", "deranged"):
            trained[mode], telemetry[mode] = _train_readout(
                dataset=datasets_fit[mode],
                device=device,
                seed=fold_seed,
            )
        training_telemetry_reports.append(
            {
                "fold": int(fold_index),
                "seed": int(fold_seed),
                "readouts": telemetry,
            }
        )
        parameter_counts = {
            mode: count_trainable_parameters(model)
            for mode, model in trained.items()
        }
        if set(parameter_counts.values()) != {EXPECTED_HEAD_PARAMETERS}:
            raise RuntimeError(
                f"B7 matched parameter contract failed: {parameter_counts}"
            )
        datasets_hold = {
            mode: _PairPrefixDataset(
                prefix_cache=prefix_cache,
                patch_mean_cache=patch_mean_cache,
                global_logits=global_logits,
                rows=hold_rows,
                mode=mode,
                derangement_sources=(
                    derangement_sources if mode == "deranged" else None
                ),
            )
            for mode in ("candidate", "control", "deranged")
        }
        probabilities = {
            mode: _predict_readout(
                model=trained[mode], dataset=datasets_hold[mode], device=device
            )
            for mode in ("candidate", "control", "deranged")
        }
        checkpoint_payload = {
            "protocol_id": PROTOCOL_ID,
            "fold": int(fold_index),
            "seed": int(fold_seed),
            "parameter_counts": parameter_counts,
            "telemetry": telemetry,
            "derangement": derangement_report,
            **{
                f"{mode}_state_dict": trained[mode].state_dict()
                for mode in ("candidate", "control", "deranged")
            },
        }
        torch.save(checkpoint_payload, model_dir / f"fold_{fold_index:02d}.pt")
        hold_pair_ids = np.asarray(hold_rows["pair_ids"], dtype=np.int64)
        hold_targets = np.asarray(hold_rows["targets"], dtype=np.int64)
        hold_sample_indices = np.asarray(
            hold_rows["sample_indices"], dtype=np.int64
        )
        for pair_index, rival in enumerate(PAIR_RIVALS):
            local = np.flatnonzero(hold_pair_ids == int(pair_index))
            pair_targets = hold_targets[local]
            metrics = {
                mode: _binary_metrics(pair_targets, probabilities[mode][local])
                for mode in ("candidate", "control", "deranged")
            }
            fold_row: Dict[str, object] = {
                "pair": f"{int(rival)}-1",
                "rival_class": int(rival),
                "fold": int(fold_index),
                "fit_samples": int(fit_indices.size),
                "holdout_samples": int(hold_indices.size),
                "pair_holdout_samples": int(local.size),
                "source_overlap": int(len(overlap)),
                "derangement_fixed_points": int(
                    derangement_report["fixed_points"]
                ),
                "derangement_same_source_group_assignments": int(
                    derangement_report["same_source_group_assignments"]
                ),
                "derangement_block_donor_consistent": bool(
                    derangement_report["block_donor_consistent"]
                ),
            }
            for mode in ("candidate", "control", "deranged"):
                _prefixed_metrics(fold_row, mode, metrics[mode])
                fold_row[f"{mode}_final_augmented_loss"] = float(
                    telemetry[mode]["final_augmented_loss"]
                )
            fold_row["candidate_control_delta_auroc"] = float(
                metrics["candidate"]["auroc"] - metrics["control"]["auroc"]
            )
            fold_row["candidate_deranged_delta_auroc"] = float(
                metrics["candidate"]["auroc"] - metrics["deranged"]["auroc"]
            )
            fold_metric_rows.append(fold_row)
            for offset in local.tolist():
                sample_index = int(hold_sample_indices[offset])
                prediction_rows.append(
                    {
                        "sample_index": sample_index,
                        "image_path": str(paths[sample_index]),
                        "source_group": str(groups[sample_index]),
                        "fold": int(fold_index),
                        "target_index": int(labels[sample_index]),
                        "pair": f"{int(rival)}-1",
                        "binary_target_class1": int(hold_targets[offset]),
                        **{
                            f"{mode}_probability_class1": float(
                                probabilities[mode][offset]
                            )
                            for mode in ("control", "candidate", "deranged")
                        },
                    }
                )
        del trained, datasets_fit, datasets_hold
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prediction_integrity = _validate_oof_prediction_coverage(
        prediction_rows=prediction_rows,
        labels=labels,
        source_groups=groups,
        fold_assignments=assignments,
        paths=paths,
    )
    pair_rows: List[Dict[str, object]] = []
    for rival in PAIR_RIVALS:
        selected = [
            row for row in prediction_rows if row["pair"] == f"{int(rival)}-1"
        ]
        targets = np.asarray(
            [row["binary_target_class1"] for row in selected], dtype=np.int64
        )
        metrics: Dict[str, Dict[str, float]] = {}
        for mode in ("control", "candidate", "deranged"):
            values = np.asarray(
                [row[f"{mode}_probability_class1"] for row in selected],
                dtype=np.float64,
            )
            metrics[mode] = _binary_metrics(targets, values)
        row: Dict[str, object] = {
            "pair": f"{int(rival)}-1",
            "rival_class": int(rival),
            "samples": int(targets.size),
            "class1_samples": int(targets.sum()),
            "rival_samples": int((targets == 0).sum()),
        }
        for mode in ("control", "candidate", "deranged"):
            _prefixed_metrics(row, mode, metrics[mode])
        for key in (
            "balanced_accuracy",
            "auroc",
            "precision_class1",
            "recall_class1",
            "specificity_rival",
            "f1_class1",
        ):
            row[f"delta_{key}"] = float(
                metrics["candidate"][key] - metrics["control"][key]
            )
        row["candidate_deranged_delta_auroc"] = float(
            metrics["candidate"]["auroc"] - metrics["deranged"]["auroc"]
        )
        row["rival_fp_reduction"] = float(
            (metrics["control"]["fp"] - metrics["candidate"]["fp"])
            / max(1.0, metrics["control"]["fp"])
        )
        pair_rows.append(row)
    total_contracts = int(
        sum(int(row["bijective_permutation_contracts"]) for row in derangement_reports)
    )
    derangement_integrity = {
        "fold_reports": derangement_reports,
        "expected_bijective_permutation_contracts": 2 * FOLDS,
        "observed_bijective_permutation_contracts": total_contracts,
        "fixed_points": int(
            sum(int(row["fixed_points"]) for row in derangement_reports)
        ),
        "same_source_group_assignments": int(
            sum(
                int(row["same_source_group_assignments"])
                for row in derangement_reports
            )
        ),
        "fold_boundary_crossings": int(
            sum(
                int(row["fit_to_hold_crossings"])
                + int(row["hold_to_fit_crossings"])
                for row in derangement_reports
            )
        ),
        "all_block_donors_consistent": all(
            bool(row["block_donor_consistent"])
            for row in derangement_reports
        ),
        "label_blind_construction": all(
            bool(row["label_blind_construction"])
            for row in derangement_reports
        ),
    }
    return (
        pair_rows,
        fold_metric_rows,
        prediction_rows,
        {
            **fold_assignment_integrity,
            **prediction_integrity,
            "derangement": derangement_integrity,
            "training_telemetry": training_telemetry_reports,
        },
    )


def assess_prefix_residual_readiness(
    *,
    pair_rows: Sequence[Mapping[str, object]],
    fold_metric_rows: Sequence[Mapping[str, object]],
    global_fold_rows: Sequence[Mapping[str, object]],
    train_samples: int,
    source_groups: int,
    architecture: Mapping[str, object],
    runtime_contract: Mapping[str, object],
    cache_manifest: Mapping[str, object],
    cache_finite: bool,
    oof_integrity: Mapping[str, object],
) -> Dict[str, object]:
    pairs = [dict(row) for row in pair_rows]
    folds = [dict(row) for row in fold_metric_rows]
    global_folds = [dict(row) for row in global_fold_rows]
    thresholds: Dict[str, object] = {
        "required_train_samples": EXPECTED_TRAIN_SAMPLES,
        "required_source_groups": EXPECTED_SOURCE_GROUPS,
        "required_global_folds": FOLDS,
        "required_pair_folds": FOLDS * PAIR_COUNT,
        "required_derangement_contracts": 2 * FOLDS,
        "max_source_overlap": 0,
        "max_invariance_error": 1e-6,
        "expected_head_parameters": EXPECTED_HEAD_PARAMETERS,
        "max_head_macs_all_pairs": MAX_HEAD_MACS_ALL_PAIRS,
        "min_mean_auroc_gain": 0.010,
        "min_positive_pair_fold_gains": 10,
        "min_pairs_with_auroc_gain": 2,
        "min_pair_auroc_gain": 0.010,
        "max_pair_auroc_loss": 0.005,
        "min_pairs_with_f1_gain": 2,
        "min_pair_f1_gain": 0.010,
        "min_class1_tp_retention": 0.990,
        "max_pair_recall_loss": 0.005,
        "min_rival_fp_reduction": 0.100,
        "critical_rival_class": 2,
        "min_critical_auroc_gain": 0.010,
        "min_critical_fp_reduction": 0.100,
        "min_candidate_deranged_mean_auroc_gain": 0.008,
        "min_candidate_deranged_pair_fold_wins": 10,
    }
    expected_rivals = set(PAIR_RIVALS)
    observed_rivals = [int(row.get("rival_class", -1)) for row in pairs]
    complete_pairs = (
        len(pairs) == PAIR_COUNT
        and set(observed_rivals) == expected_rivals
        and len(set(observed_rivals)) == PAIR_COUNT
    )
    pair_labels_consistent = all(
        str(row.get("pair", "")) == f"{int(row.get('rival_class', -1))}-1"
        for row in pairs
    )
    expected_fold_keys = {
        (int(rival), int(fold))
        for rival in PAIR_RIVALS
        for fold in range(FOLDS)
    }
    observed_fold_keys = [
        (int(row.get("rival_class", -1)), int(row.get("fold", -1)))
        for row in folds
    ]
    fold_fields = {
        "candidate_auroc",
        "control_auroc",
        "deranged_auroc",
        "candidate_control_delta_auroc",
        "candidate_deranged_delta_auroc",
    }
    complete_pair_folds = (
        len(folds) == FOLDS * PAIR_COUNT
        and set(observed_fold_keys) == expected_fold_keys
        and len(set(observed_fold_keys)) == len(observed_fold_keys)
        and all(fold_fields.issubset(row) for row in folds)
    )
    observed_global_folds = [int(row.get("fold", -1)) for row in global_folds]
    complete_global_folds = (
        len(global_folds) == FOLDS
        and set(observed_global_folds) == set(range(FOLDS))
        and len(set(observed_global_folds)) == FOLDS
    )
    numeric_values = [
        float(value)
        for row in [*pairs, *folds, *global_folds]
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    metrics_finite = bool(numeric_values) and bool(
        np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all()
    )
    mean_auroc_gain = float(
        np.mean([float(row["delta_auroc"]) for row in pairs])
    ) if complete_pairs else -math.inf
    positive_pair_fold_gains = int(
        sum(
            float(row.get("candidate_control_delta_auroc", -math.inf)) > 0.0
            for row in folds
        )
    )
    pairs_with_auroc_gain = int(
        sum(
            float(row.get("delta_auroc", -math.inf))
            >= float(thresholds["min_pair_auroc_gain"])
            for row in pairs
        )
    )
    minimum_pair_auroc_gain = float(
        min((float(row.get("delta_auroc", -math.inf)) for row in pairs), default=-math.inf)
    )
    pairs_with_f1_gain = int(
        sum(
            float(row.get("delta_f1_class1", -math.inf))
            >= float(thresholds["min_pair_f1_gain"])
            for row in pairs
        )
    )
    minimum_pair_recall_delta = float(
        min(
            (float(row.get("delta_recall_class1", -math.inf)) for row in pairs),
            default=-math.inf,
        )
    )
    control_tp = float(sum(float(row.get("control_tp", 0.0)) for row in pairs))
    candidate_tp = float(
        sum(float(row.get("candidate_tp", 0.0)) for row in pairs)
    )
    control_fp = float(sum(float(row.get("control_fp", 0.0)) for row in pairs))
    candidate_fp = float(
        sum(float(row.get("candidate_fp", 0.0)) for row in pairs)
    )
    tp_retention = candidate_tp / max(1.0, control_tp)
    fp_reduction = (control_fp - candidate_fp) / max(1.0, control_fp)
    candidate_deranged_mean_gain = float(
        np.mean(
            [float(row.get("candidate_deranged_delta_auroc", -math.inf)) for row in pairs]
        )
    ) if complete_pairs else -math.inf
    candidate_deranged_fold_wins = int(
        sum(
            float(row.get("candidate_deranged_delta_auroc", -math.inf)) > 0.0
            for row in folds
        )
    )
    critical = next(
        (row for row in pairs if int(row.get("rival_class", -1)) == 2), None
    )
    critical_auroc = float(critical.get("delta_auroc", -math.inf)) if critical else -math.inf
    critical_fp_reduction = (
        float(critical.get("rival_fp_reduction", -math.inf))
        if critical
        else -math.inf
    )
    critical_recall = (
        float(critical.get("delta_recall_class1", -math.inf))
        if critical
        else -math.inf
    )
    maximum_source_overlap = int(
        max(
            [int(row.get("source_overlap", 0)) for row in [*folds, *global_folds]],
            default=0,
        )
    )
    derangement = oof_integrity.get("derangement", {})
    if not isinstance(derangement, Mapping):
        derangement = {}
    training_telemetry = oof_integrity.get("training_telemetry", [])
    if not isinstance(training_telemetry, Sequence) or isinstance(
        training_telemetry, (str, bytes)
    ):
        training_telemetry = []
    telemetry_complete = len(training_telemetry) == FOLDS
    telemetry_finite = True
    telemetry_fold_ids: List[int] = []
    for fold_telemetry in training_telemetry:
        if not isinstance(fold_telemetry, Mapping):
            telemetry_complete = False
            telemetry_finite = False
            continue
        fold_index = int(fold_telemetry.get("fold", -1))
        telemetry_fold_ids.append(fold_index)
        readouts = fold_telemetry.get("readouts", {})
        if not isinstance(readouts, Mapping) or set(readouts) != {
            "candidate",
            "control",
            "deranged",
        }:
            telemetry_complete = False
            continue
        for payload in readouts.values():
            if not isinstance(payload, Mapping):
                telemetry_complete = False
                telemetry_finite = False
                continue
            expected_seed = SEED + fold_index
            vectors = [
                payload.get("initial_dual_values", []),
                payload.get("final_dual_values", []),
                payload.get("active_constraint_updates_by_pair", []),
                payload.get("final_epoch_mean_soft_recall_by_pair", []),
                payload.get("final_epoch_mean_violation_by_pair", []),
                payload.get("final_epoch_active_batches_by_pair", []),
            ]
            flat_values = [
                float(value)
                for vector in vectors
                if isinstance(vector, Sequence)
                for value in vector
            ]
            telemetry_finite = telemetry_finite and bool(
                flat_values
                and np.isfinite(np.asarray(flat_values, dtype=np.float64)).all()
            )
            telemetry_complete = telemetry_complete and (
                int(payload.get("seed", -1)) == expected_seed
                and int(payload.get("data_order_seed", -1)) == expected_seed
                and payload.get("initial_dual_values") == [0.0] * PAIR_COUNT
                and all(
                    len(vector) == PAIR_COUNT
                    for vector in vectors
                    if isinstance(vector, Sequence)
                )
                and all(
                    int(value) > 0
                    for value in payload.get(
                        "active_constraint_updates_by_pair", []
                    )
                )
                and all(
                    int(value) > 0
                    for value in payload.get(
                        "final_epoch_active_batches_by_pair", []
                    )
                )
                and int(payload.get("successful_optimizer_updates", 0)) > 0
                and int(payload.get("nonfinite_steps", -1)) == 0
                and bool(payload.get("dual_values_finite", False))
            )
    telemetry_complete = telemetry_complete and (
        set(telemetry_fold_ids) == set(range(FOLDS))
        and len(set(telemetry_fold_ids)) == FOLDS
    )
    observed_contracts = int(
        derangement.get("observed_bijective_permutation_contracts", -1)
    )
    provenance_exact = (
        int(cache_manifest.get("schema_version", -1))
        == CACHE_MANIFEST_SCHEMA_VERSION
        and str(cache_manifest.get("protocol_id", "")) == PROTOCOL_ID
        and str(cache_manifest.get("data_sha256", "")) == EXPECTED_DATA_SHA256
        and str(cache_manifest.get("checkpoint_sha256", ""))
        == EXPECTED_CHECKPOINT_SHA256
        and int(cache_manifest.get("source_group_count", -1))
        == EXPECTED_SOURCE_GROUPS
        and bool(cache_manifest.get("train_split_used", False))
        and not bool(cache_manifest.get("validation_split_used", True))
        and not bool(cache_manifest.get("test_split_used", True))
        and not bool(cache_manifest.get("validation_dataset_constructed", True))
        and not bool(cache_manifest.get("test_dataset_constructed", True))
    )
    parameter_counts = [
        int(architecture.get(f"{mode}_parameters", -1))
        for mode in ("candidate", "control", "deranged")
    ]
    observed: Dict[str, object] = {
        "mean_auroc_gain": mean_auroc_gain,
        "positive_pair_fold_gains": positive_pair_fold_gains,
        "pairs_with_auroc_gain": pairs_with_auroc_gain,
        "minimum_pair_auroc_gain": minimum_pair_auroc_gain,
        "pairs_with_f1_gain": pairs_with_f1_gain,
        "minimum_pair_recall_delta": minimum_pair_recall_delta,
        "class1_tp_retention": float(tp_retention),
        "rival_fp_reduction": float(fp_reduction),
        "control_tp": int(control_tp),
        "candidate_tp": int(candidate_tp),
        "control_fp": int(control_fp),
        "candidate_fp": int(candidate_fp),
        "candidate_deranged_mean_auroc_gain": candidate_deranged_mean_gain,
        "candidate_deranged_pair_fold_wins": candidate_deranged_fold_wins,
        "critical_2_1_auroc_gain": critical_auroc,
        "critical_2_1_fp_reduction": critical_fp_reduction,
        "critical_2_1_recall_delta": critical_recall,
        "maximum_source_overlap": maximum_source_overlap,
        "derangement_contracts": observed_contracts,
        "parameter_counts": parameter_counts,
        "head_macs_all_pairs": int(
            architecture.get("estimated_head_macs_all_pairs", -1)
        ),
        "metrics_finite": metrics_finite,
        "provenance_exact": provenance_exact,
        "training_telemetry_complete": telemetry_complete,
        "training_telemetry_finite": telemetry_finite,
    }
    checks = {
        "canonical_train_support": int(train_samples) == EXPECTED_TRAIN_SAMPLES,
        "canonical_source_groups": int(source_groups) == EXPECTED_SOURCE_GROUPS,
        "complete_global_folds": complete_global_folds,
        "complete_pair_coverage": complete_pairs and pair_labels_consistent,
        "complete_pair_fold_coverage": complete_pair_folds,
        "source_group_folds_disjoint": maximum_source_overlap == 0,
        "cache_and_metrics_finite": bool(cache_finite) and metrics_finite,
        "provenance_exact": provenance_exact,
        "runtime_representation_exact": not bool(
            runtime_contract.get("mismatches", {"missing": True})
        )
        and float(
            dict(runtime_contract.get("observed", {})).get(
                "synthetic_pool_parity_error", math.inf
            )
        )
        <= 1e-6,
        "parameter_count_matched": set(parameter_counts)
        == {EXPECTED_HEAD_PARAMETERS},
        "head_mac_budget": int(
            architecture.get("estimated_head_macs_all_pairs", -1)
        )
        <= MAX_HEAD_MACS_ALL_PAIRS,
        "zero_residual_control_parity": float(
            architecture.get(
                "zero_residual_candidate_control_max_abs_error", math.inf
            )
        )
        <= 1e-6,
        "optimized_all_pairs_path_parity": float(
            architecture.get("all_pairs_path_max_abs_error", math.inf)
        )
        <= 1e-6,
        "register_permutation_invariance": float(
            architecture.get("register_permutation_max_abs_error", math.inf)
        )
        <= 1e-6,
        "non_affine_layer_norm": not bool(
            architecture.get("layer_norm_elementwise_affine", True)
        ),
        "population_register_variance": architecture.get(
            "register_variance_unbiased", None
        )
        is False,
        "initial_states_matched": float(
            architecture.get(
                "candidate_control_initial_state_max_abs_error", math.inf
            )
        )
        == 0.0
        and float(
            architecture.get(
                "candidate_deranged_initial_state_max_abs_error", math.inf
            )
        )
        == 0.0,
        "derangement_bijections_complete": observed_contracts == 2 * FOLDS,
        "derangement_source_disjoint": int(
            derangement.get("same_source_group_assignments", -1)
        )
        == 0,
        "derangement_fold_safe": int(
            derangement.get("fold_boundary_crossings", -1)
        )
        == 0,
        "derangement_zero_fixed_points": int(
            derangement.get("fixed_points", -1)
        )
        == 0,
        "derangement_block_coherent": bool(
            derangement.get("all_block_donors_consistent", False)
        ),
        "derangement_label_blind": bool(
            derangement.get("label_blind_construction", False)
        ),
        "primal_dual_telemetry_complete": telemetry_complete
        and telemetry_finite,
        "mean_auroc_gain": mean_auroc_gain >= 0.010,
        "fold_direction_stability": positive_pair_fold_gains >= 10,
        "pair_auroc_support": pairs_with_auroc_gain >= 2,
        "no_pair_auroc_collapse": minimum_pair_auroc_gain >= -0.005,
        "pair_f1_support": pairs_with_f1_gain >= 2,
        "class1_tp_retained": tp_retention >= 0.990,
        "class1_recall_protected": minimum_pair_recall_delta >= -0.005,
        "rival_false_positives_reduced": fp_reduction >= 0.100,
        "critical_2_1_auroc_gain": critical_auroc >= 0.010,
        "critical_2_1_fp_reduction": critical_fp_reduction >= 0.100,
        "critical_2_1_recall_protected": critical_recall >= -0.005,
        "candidate_beats_deranged_mean": candidate_deranged_mean_gain >= 0.008,
        "candidate_beats_deranged_folds": candidate_deranged_fold_wins >= 10,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "prefix_residual_alignment_ready": bool(ready),
        "implementation_permission": bool(ready),
        "validation_permission": False,
        "smoke_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _locked_protocol_payload() -> Dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "focus_class": int(FOCUS_CLASS),
        "rival_classes": list(PAIR_RIVALS),
        "folds": FOLDS,
        "seed": SEED,
        "candidate": "aligned_cls_register_residual_block",
        "control": "all_five_prefixes_equal_aligned_patch_mean",
        "falsification": (
            "source_disjoint_label_blind_whole_residual_block_derangement;"
            "fit_hold_independent;global_logits_and_patch_mean_aligned"
        ),
        "prefix_layout": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "prefix_tokens": PREFIX_TOKENS,
        "patch_tokens": PATCH_TOKENS,
        "token_width": TOKEN_WIDTH,
        "projection": [TOKEN_WIDTH, PROJECTION_WIDTH],
        "projection_shared_across_prefixes_and_pairs": True,
        "register_aggregates": ["mean", "population_variance"],
        "global_logit_width": GLOBAL_LOGIT_WIDTH,
        "pair_embedding_width": PAIR_EMBED_WIDTH,
        "descriptor_width": DESCRIPTOR_WIDTH,
        "readout_hidden": READOUT_HIDDEN,
        "trainable_parameters_each": EXPECTED_HEAD_PARAMETERS,
        "head_macs_all_pairs": EXPECTED_HEAD_MACS_ALL_PAIRS,
        "max_head_macs_all_pairs": MAX_HEAD_MACS_ALL_PAIRS,
        "head_epochs": HEAD_EPOCHS,
        "head_batch_size": HEAD_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "gradient_clip": HEAD_GRAD_CLIP,
        "threshold": THRESHOLD,
        "loss": "six_stratum_weighted_bce_plus_augmented_primal_dual_soft_recall",
        "soft_recall_target": SOFT_RECALL_TARGET,
        "dual_learning_rate": DUAL_LEARNING_RATE,
        "dual_max": DUAL_MAX,
        "augmented_lagrangian_rho": AUGMENTED_LAGRANGIAN_RHO,
        "dual_reset_per_fold_and_readout": True,
        "head_fp32": True,
        "early_stopping": False,
        "architecture_or_threshold_sweep": False,
        "candidate_selection_uses_validation": False,
    }


def _validate_architecture_preflight(
    architecture: Mapping[str, object],
) -> None:
    counts = {
        int(architecture.get(f"{mode}_parameters", -1))
        for mode in ("candidate", "control", "deranged")
    }
    checks = {
        "parameter_count": counts == {EXPECTED_HEAD_PARAMETERS},
        "mac_count": int(architecture["estimated_head_macs_all_pairs"])
        == EXPECTED_HEAD_MACS_ALL_PAIRS
        and int(architecture["estimated_head_macs_all_pairs"])
        <= MAX_HEAD_MACS_ALL_PAIRS,
        "initial_state_candidate_control": float(
            architecture["candidate_control_initial_state_max_abs_error"]
        )
        == 0.0,
        "initial_state_candidate_deranged": float(
            architecture["candidate_deranged_initial_state_max_abs_error"]
        )
        == 0.0,
        "zero_residual_parity": float(
            architecture["zero_residual_candidate_control_max_abs_error"]
        )
        <= 1e-6,
        "all_pairs_parity": float(
            architecture["all_pairs_path_max_abs_error"]
        )
        <= 1e-6,
        "register_permutation_invariance": float(
            architecture["register_permutation_max_abs_error"]
        )
        <= 1e-6,
        "non_affine_layer_norm": not bool(
            architecture["layer_norm_elementwise_affine"]
        ),
        "population_variance": architecture["register_variance_unbiased"]
        is False,
        "descriptor_shape": architecture["descriptor_shape"]
        == [6, DESCRIPTOR_WIDTH],
        "output_shape": architecture["output_shape"] == [6],
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"Locked B7 architecture preflight failed: {failed}")


def _preflight_derangement_contract(
    *,
    labels: np.ndarray,
    source_groups: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    assignments, global_fold_rows = assign_global_source_folds(
        labels, source_groups, folds=FOLDS, seed=SEED
    )
    _validate_fold_assignment_contract(source_groups, assignments)
    reports: List[Dict[str, object]] = []
    for fold_index in range(FOLDS):
        fit = np.flatnonzero(assignments != fold_index)
        hold = np.flatnonzero(assignments == fold_index)
        _map, report = build_derangement_source_map(
            sample_count=labels.size,
            fit_indices=fit,
            hold_indices=hold,
            source_groups=source_groups,
            seed=SEED + fold_index,
        )
        reports.append({"fold": fold_index, **report})
    summary = {
        "folds": FOLDS,
        "fit_hold_maps": 2 * FOLDS,
        "bijective_permutation_contracts": int(
            sum(int(row["bijective_permutation_contracts"]) for row in reports)
        ),
        "checked_prefix_cells": int(
            sum(int(row["checked_prefix_cells"]) for row in reports)
        ),
        "fixed_points": int(sum(int(row["fixed_points"]) for row in reports)),
        "same_source_group_assignments": int(
            sum(int(row["same_source_group_assignments"]) for row in reports)
        ),
        "fold_boundary_crossings": int(
            sum(
                int(row["fit_to_hold_crossings"])
                + int(row["hold_to_fit_crossings"])
                for row in reports
            )
        ),
        "all_block_donors_consistent": all(
            bool(row["block_donor_consistent"]) for row in reports
        ),
        "label_blind_construction": all(
            bool(row["label_blind_construction"]) for row in reports
        ),
    }
    if not (
        summary["bijective_permutation_contracts"] == 2 * FOLDS
        and summary["fixed_points"] == 0
        and summary["same_source_group_assignments"] == 0
        and summary["fold_boundary_crossings"] == 0
        and summary["all_block_donors_consistent"]
        and summary["label_blind_construction"]
    ):
        raise RuntimeError("Locked B7 derangement preflight failed")
    return assignments, global_fold_rows, {"summary": summary, "folds": reports}


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (data_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    data_sha256 = _sha256(data_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(
            f"Locked B7 data hash mismatch: {data_sha256} != {EXPECTED_DATA_SHA256}"
        )
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Locked B7 checkpoint hash mismatch: "
            f"{checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Invalid B7 checkpoint payload: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model_contract = _model_contract(model)
    checkpoint_contract = _validate_checkpoint_contract(
        checkpoint, model_contract
    )
    runtime_contract = _runtime_representation_contract(model)
    dataset, class_names = _build_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    if len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"Locked B7 train support mismatch: {len(dataset)} != "
            f"{EXPECTED_TRAIN_SAMPLES}"
        )
    if len(class_names) != GLOBAL_LOGIT_WIDTH:
        raise ValueError(f"Locked B7 requires five classes, got {len(class_names)}")
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    labels_fn = getattr(dataset, "labels", None)
    if not callable(sample_paths_fn) or not callable(labels_fn):
        raise TypeError("B7 train dataset must expose sample_paths() and labels()")
    paths = [str(path) for path in sample_paths_fn()]
    canonical_labels = np.asarray(labels_fn(), dtype=np.int64).reshape(-1)
    if len(paths) != len(dataset) or canonical_labels.size != len(dataset):
        raise RuntimeError("B7 canonical paths/labels are misaligned")
    if set(np.unique(canonical_labels).tolist()) != set(
        range(GLOBAL_LOGIT_WIDTH)
    ):
        raise ValueError("B7 canonical labels must contain classes 0..4")
    source_groups = np.asarray(
        [normalized_source_group(path) for path in paths], dtype=object
    )
    if any(not str(value).strip() for value in source_groups.tolist()):
        raise ValueError("B7 source grouping produced an empty identifier")
    source_group_count = int(np.unique(source_groups).size)
    if source_group_count != EXPECTED_SOURCE_GROUPS:
        raise ValueError(
            f"Locked B7 source groups mismatch: {source_group_count} != "
            f"{EXPECTED_SOURCE_GROUPS}"
        )
    _assert_output_outside_train(output_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture = architecture_self_check()
    _validate_architecture_preflight(architecture)
    (
        fold_assignments,
        global_fold_rows,
        derangement_preflight,
    ) = _preflight_derangement_contract(
        labels=canonical_labels, source_groups=source_groups
    )
    preflight: Dict[str, object] = {
        "schema_version": 1,
        "mode": "dinov3_prefix_residual_train_only_oof",
        "protocol_id": PROTOCOL_ID,
        "data": str(data_path),
        "data_sha256": data_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(output_dir),
        "class_names": [str(name) for name in class_names],
        "train_samples": int(len(dataset)),
        "source_groups": source_group_count,
        "paths_sha256": _path_rows_sha256(paths),
        "source_groups_sha256": _source_groups_sha256(
            source_groups.tolist()
        ),
        "canonical_labels_sha256": _labels_sha256(canonical_labels),
        "fold_assignments_sha256": hashlib.sha256(
            np.asarray(fold_assignments, dtype="<i8").tobytes()
        ).hexdigest(),
        "source_group_normalization": SOURCE_GROUP_NORMALIZATION,
        "model_name": MODEL_NAME,
        "model_contract": model_contract,
        "checkpoint_contract": checkpoint_contract,
        "runtime_representation_contract": runtime_contract,
        "architecture_contract": architecture,
        "derangement_preflight": derangement_preflight,
        "global_folds": global_fold_rows,
        "locked_protocol": _locked_protocol_payload(),
        "estimated_cache_bytes": int(
            EXPECTED_TRAIN_SAMPLES
            * (
                PREFIX_TOKENS * TOKEN_WIDTH * np.dtype(np.float16).itemsize
                + TOKEN_WIDTH * np.dtype(np.float16).itemsize
                + GLOBAL_LOGIT_WIDTH * np.dtype(np.float32).itemsize
                + np.dtype(np.int64).itemsize
            )
        ),
        "representation_scope": (
            "matched_incremental_alignment_readiness_on_full_train_fitted_b2;"
            "not_causal_and_not_unbiased_new_source_generalization"
        ),
        "synthetic_model_probe_used": True,
        "dataset_feature_extraction_performed": False,
        "head_training_performed": False,
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "raw_dataset_modified": False,
    }
    _atomic_json(output_dir / "preflight.json", preflight)
    if bool(args.preflight_only):
        return preflight
    device = _resolve_device(str(args.device or ""))
    model.to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    cache = _load_or_extract_cache(
        model=model,
        dataset=dataset,
        paths=paths,
        source_groups=source_groups,
        expected_labels=canonical_labels,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    labels = np.asarray(cache["labels"], dtype=np.int64)
    if not np.array_equal(labels, canonical_labels):
        raise ValueError("B7 cached labels differ from canonical labels")
    (
        pair_rows,
        fold_metric_rows,
        prediction_rows,
        oof_integrity,
    ) = run_prefix_residual_oof(
        prefix_cache=np.asarray(cache["prefixes"]),
        patch_mean_cache=np.asarray(cache["patch_mean"]),
        global_logits=np.asarray(cache["global_logits"]),
        labels=labels,
        source_groups=source_groups,
        fold_assignments=fold_assignments,
        paths=paths,
        output_dir=output_dir,
        device=device,
    )
    assignment_rows = [
        {
            "sample_index": int(index),
            "image_path": str(paths[index]),
            "source_group": str(source_groups[index]),
            "target_index": int(labels[index]),
            "fold": int(fold_assignments[index]),
        }
        for index in range(labels.size)
    ]
    _write_csv(output_dir / "global_fold_assignments.csv", assignment_rows)
    _write_csv(output_dir / "pair_fold_metrics.csv", fold_metric_rows)
    _write_csv(output_dir / "pair_metrics.csv", pair_rows)
    _write_csv(
        output_dir / "train_oof_pair_predictions.csv", prediction_rows
    )
    cache_finite = bool(
        _array_all_finite(np.asarray(cache["prefixes"]))
        and _array_all_finite(np.asarray(cache["patch_mean"]))
        and _array_all_finite(np.asarray(cache["global_logits"]))
    )
    readiness = assess_prefix_residual_readiness(
        pair_rows=pair_rows,
        fold_metric_rows=fold_metric_rows,
        global_fold_rows=global_fold_rows,
        train_samples=int(labels.size),
        source_groups=source_group_count,
        architecture=architecture,
        runtime_contract=runtime_contract,
        cache_manifest=dict(cache["manifest"]),
        cache_finite=cache_finite,
        oof_integrity=oof_integrity,
    )
    summary: Dict[str, object] = {
        **preflight,
        "dataset_feature_extraction_performed": True,
        "head_training_performed": True,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "cache_manifest": dict(cache["manifest"]),
        "pair_results": pair_rows,
        "oof_integrity": oof_integrity,
        "readiness": readiness,
        "artifacts": {
            "preflight": str(output_dir / "preflight.json"),
            "cache_manifest": str(output_dir / CACHE_MANIFEST_FILENAME),
            "global_fold_assignments": str(
                output_dir / "global_fold_assignments.csv"
            ),
            "pair_fold_metrics": str(output_dir / "pair_fold_metrics.csv"),
            "pair_metrics": str(output_dir / "pair_metrics.csv"),
            "oof_predictions": str(
                output_dir / "train_oof_pair_predictions.csv"
            ),
            "fold_readouts": str(output_dir / "fold_readouts"),
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_precheck(args)
    readiness = summary.get("readiness")
    if isinstance(readiness, Mapping):
        output = {
            "prefix_residual_alignment_ready": bool(
                readiness.get("prefix_residual_alignment_ready", False)
            ),
            "failed_checks": list(readiness.get("failed_checks", [])),
            "observed": dict(readiness.get("observed", {})),
            "validation_split_used": False,
            "test_split_used": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    else:
        output = {
            "preflight_only": bool(args.preflight_only),
            "train_samples": int(summary.get("train_samples", 0)),
            "source_groups": int(summary.get("source_groups", 0)),
            "dataset_feature_extraction_performed": bool(
                summary.get("dataset_feature_extraction_performed", False)
            ),
            "head_training_performed": bool(
                summary.get("head_training_performed", False)
            ),
            "validation_split_used": False,
            "test_split_used": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
