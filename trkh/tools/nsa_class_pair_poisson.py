from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


SEED = 20260724
GAMMA_SHAPE = 2.0
GAMMA_SCALE = 0.05
GAMMA_OFFSET = 0.03
HALF_SIZE_MIN = 0.03
HALF_SIZE_MAX = 0.35
RESIZE_MEAN = 1.0
RESIZE_STD = 0.5
RESIZE_MIN = 0.7
RESIZE_MAX = 1.3
INTENSITY_K = 1.0 / 12.0
INTENSITY_X0 = 24.0
MIN_CLONE_PIXELS = 50


@dataclass(frozen=True)
class TargetGeometry:
    seed: int
    top: int
    left: int
    height: int
    width: int
    shape_attempt: int
    target_candidate_count: int
    half_height_ratio: float
    half_width_ratio: float


@dataclass(frozen=True)
class SourceGeometry:
    seed: int
    top: int
    left: int
    height: int
    width: int
    resize_scale: float
    source_candidate_count: int
    source_attempt: int


@dataclass(frozen=True)
class PoissonGeometry:
    target: TargetGeometry
    source: SourceGeometry


@dataclass(frozen=True)
class PoissonView:
    composite_rgb: np.ndarray
    intensity_target: np.ndarray
    changed_mask: np.ndarray
    clone_mask: np.ndarray
    geometry: PoissonGeometry
    maximum_outside_target_delta: int


def local_seed(
    sample_index: int,
    epoch: int,
    role_id: int,
    *,
    base_seed: int = SEED,
) -> int:
    value = (
        int(base_seed)
        + 1_000_003 * int(sample_index)
        + 1_009 * int(epoch)
        + 37 * int(role_id)
    )
    return int(value % (2**63 - 1))


def rng_snapshot() -> Tuple[object, tuple, Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def rng_snapshot_equal(
    left: Tuple[object, tuple, Tensor],
    right: Tuple[object, tuple, Tensor],
) -> bool:
    random_equal = left[0] == right[0]
    numpy_equal = (
        left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1])
        and left[1][2:] == right[1][2:]
    )
    return bool(random_equal and numpy_equal and torch.equal(left[2], right[2]))


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def geometry_sha256(geometry: PoissonGeometry) -> str:
    payload = json.dumps(
        asdict(geometry),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not bool(value.any()):
        raise ValueError("support must be a nonempty 2D mask")
    return value


def support_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    value = _as_bool_mask(mask)
    ys, xs = np.nonzero(value)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def valid_rectangle_positions(
    support: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    mask = _as_bool_mask(support)
    patch_height = int(height)
    patch_width = int(width)
    if patch_height < 1 or patch_width < 1:
        raise ValueError("rectangle dimensions must be positive")
    image_height, image_width = mask.shape
    if patch_height > image_height or patch_width > image_width:
        return np.empty((0, 2), dtype=np.int32)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    covered = (
        integral[patch_height:, patch_width:]
        - integral[:-patch_height, patch_width:]
        - integral[patch_height:, :-patch_width]
        + integral[:-patch_height, :-patch_width]
    )
    return np.argwhere(covered == patch_height * patch_width).astype(
        np.int32, copy=False
    )


def sample_target_geometry(
    support: np.ndarray,
    *,
    sample_index: int,
    epoch: int,
    role_id: int,
    max_attempts: int = 32,
) -> TargetGeometry:
    mask = _as_bool_mask(support)
    x0, y0, x1, y1 = support_bounds(mask)
    support_height = y1 - y0
    support_width = x1 - x0
    seed = local_seed(sample_index, epoch, role_id)
    generator = np.random.Generator(np.random.PCG64(seed))
    for attempt in range(1, int(max_attempts) + 1):
        half_height_ratio = float(
            np.clip(
                GAMMA_OFFSET + generator.gamma(GAMMA_SHAPE, GAMMA_SCALE),
                HALF_SIZE_MIN,
                HALF_SIZE_MAX,
            )
        )
        half_width_ratio = float(
            np.clip(
                GAMMA_OFFSET + generator.gamma(GAMMA_SHAPE, GAMMA_SCALE),
                HALF_SIZE_MIN,
                HALF_SIZE_MAX,
            )
        )
        height = max(4, 2 * int(round(half_height_ratio * support_height)))
        width = max(4, 2 * int(round(half_width_ratio * support_width)))
        if (height - 2) * (width - 2) < MIN_CLONE_PIXELS:
            continue
        candidates = valid_rectangle_positions(mask, height=height, width=width)
        if int(candidates.shape[0]) == 0:
            continue
        selected = int(generator.integers(0, int(candidates.shape[0])))
        top, left = [int(value) for value in candidates[selected]]
        return TargetGeometry(
            seed=seed,
            top=top,
            left=left,
            height=height,
            width=width,
            shape_attempt=attempt,
            target_candidate_count=int(candidates.shape[0]),
            half_height_ratio=half_height_ratio,
            half_width_ratio=half_width_ratio,
        )
    raise RuntimeError("no valid target Poisson geometry after locked attempts")


def sample_source_geometry(
    support: np.ndarray,
    target: TargetGeometry,
    *,
    sample_index: int,
    epoch: int,
    role_id: int,
    max_attempts: int = 32,
) -> SourceGeometry:
    mask = _as_bool_mask(support)
    seed = local_seed(sample_index, epoch, role_id)
    generator = np.random.Generator(np.random.PCG64(seed))
    for attempt in range(1, int(max_attempts) + 1):
        resize_scale = float(
            np.clip(
                generator.normal(RESIZE_MEAN, RESIZE_STD),
                RESIZE_MIN,
                RESIZE_MAX,
            )
        )
        source_height = max(3, int(round(target.height / resize_scale)))
        source_width = max(3, int(round(target.width / resize_scale)))
        candidates = valid_rectangle_positions(
            mask,
            height=source_height,
            width=source_width,
        )
        if int(candidates.shape[0]) == 0:
            continue
        selected = int(generator.integers(0, int(candidates.shape[0])))
        top, left = [int(value) for value in candidates[selected]]
        return SourceGeometry(
            seed=seed,
            top=top,
            left=left,
            height=source_height,
            width=source_width,
            resize_scale=resize_scale,
            source_candidate_count=int(candidates.shape[0]),
            source_attempt=attempt,
        )
    raise RuntimeError("no valid source Poisson geometry after locked attempts")


def sample_poisson_geometry(
    target_support: np.ndarray,
    source_support: np.ndarray,
    *,
    sample_index: int,
    epoch: int,
    target_role_id: int,
    source_role_id: int,
    matched_target: Optional[TargetGeometry] = None,
) -> PoissonGeometry:
    target = (
        matched_target
        if matched_target is not None
        else sample_target_geometry(
            target_support,
            sample_index=sample_index,
            epoch=epoch,
            role_id=target_role_id,
        )
    )
    source = sample_source_geometry(
        source_support,
        target,
        sample_index=sample_index,
        epoch=epoch,
        role_id=source_role_id,
    )
    return PoissonGeometry(target=target, source=source)


def _validate_rgb(array: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{label} must be uint8 RGB [H,W,3]")
    return np.ascontiguousarray(value)


def make_poisson_view(
    target_rgb: np.ndarray,
    source_rgb: np.ndarray,
    target_support: np.ndarray,
    source_support: np.ndarray,
    geometry: PoissonGeometry,
) -> PoissonView:
    target_image = _validate_rgb(target_rgb, "target_rgb")
    source_image = _validate_rgb(source_rgb, "source_rgb")
    target_mask = _as_bool_mask(target_support)
    source_mask = _as_bool_mask(source_support)
    if target_image.shape[:2] != target_mask.shape:
        raise ValueError("target support shape differs from target RGB")
    if source_image.shape[:2] != source_mask.shape:
        raise ValueError("source support shape differs from source RGB")

    target = geometry.target
    source = geometry.source
    target_slice = (
        slice(target.top, target.top + target.height),
        slice(target.left, target.left + target.width),
    )
    source_slice = (
        slice(source.top, source.top + source.height),
        slice(source.left, source.left + source.width),
    )
    if not bool(target_mask[target_slice].all()):
        raise ValueError("target rectangle leaves locked support")
    if not bool(source_mask[source_slice].all()):
        raise ValueError("source rectangle leaves locked support")

    patch = source_image[source_slice].copy()
    patch = cv2.resize(
        patch,
        (target.width, target.height),
        interpolation=cv2.INTER_LINEAR,
    )
    clone_mask = np.full((target.height, target.width), 255, dtype=np.uint8)
    clone_mask[0, :] = 0
    clone_mask[-1, :] = 0
    clone_mask[:, 0] = 0
    clone_mask[:, -1] = 0
    if int(np.count_nonzero(clone_mask)) < MIN_CLONE_PIXELS:
        raise ValueError("clone mask is smaller than the locked minimum")
    center = (
        int(target.left + target.width // 2),
        int(target.top + target.height // 2),
    )
    composite = cv2.seamlessClone(
        patch,
        target_image,
        clone_mask.copy(),
        center,
        cv2.NORMAL_CLONE,
    )
    composite = np.ascontiguousarray(composite, dtype=np.uint8)
    delta_rgb = np.abs(
        composite.astype(np.int16) - target_image.astype(np.int16)
    ).astype(np.uint8)
    mean_delta = delta_rgb.mean(axis=2).astype(np.float32)
    changed = (mean_delta > 1.0).astype(np.uint8)
    changed = cv2.medianBlur(changed, 5).astype(bool)
    intensity = cv2.medianBlur(
        np.rint(mean_delta).clip(0, 255).astype(np.uint8),
        5,
    ).astype(np.float32)
    intensity_target = changed.astype(np.float32) / (
        1.0 + np.exp(-INTENSITY_K * (intensity - INTENSITY_X0))
    )

    placed = np.zeros(target_mask.shape, dtype=bool)
    placed[target_slice] = clone_mask > 0
    changed &= placed & target_mask
    intensity_target *= changed.astype(np.float32)
    if not bool(changed.any()) or float(intensity_target.max()) <= 0.0:
        raise RuntimeError("Poisson blend produced an empty intensity target")
    outside = ~placed
    maximum_outside_delta = int(delta_rgb[outside].max()) if bool(outside.any()) else 0
    if maximum_outside_delta != 0:
        raise ValueError("Poisson blend changed pixels outside the placed rectangle")

    full_clone_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    full_clone_mask[target_slice] = clone_mask
    return PoissonView(
        composite_rgb=composite,
        intensity_target=np.ascontiguousarray(intensity_target, dtype=np.float32),
        changed_mask=np.ascontiguousarray(changed),
        clone_mask=full_clone_mask,
        geometry=geometry,
        maximum_outside_target_delta=maximum_outside_delta,
    )


def downsample_support(mask: Tensor, size: Tuple[int, int] = (32, 32)) -> Tensor:
    value = mask
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError("support mask must resolve to [B,1,H,W]")
    pooled = F.interpolate(value.float(), size=size, mode="area")
    return pooled >= 0.5


def downsample_target(target: Tensor, size: Tuple[int, int] = (32, 32)) -> Tensor:
    value = target
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError("target must resolve to [B,1,H,W]")
    return F.interpolate(value.float(), size=size, mode="area").clamp(0.0, 1.0)


class ClassQueryMismatchAdapter(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 256,
        hidden_channels: int = 64,
        num_queries: int = 5,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_queries = int(num_queries)
        self.input_norm = nn.GroupNorm(16, self.input_channels)
        self.projection = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=1,
        )
        self.projection_norm = nn.GroupNorm(8, self.hidden_channels)
        self.query_embedding = nn.Embedding(self.num_queries, self.hidden_channels)
        self.fusion = nn.Conv2d(
            self.hidden_channels * 4,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.fusion_norm = nn.GroupNorm(8, self.hidden_channels)
        self.output = nn.Conv2d(self.hidden_channels, 1, kernel_size=1)

    def forward(self, stem: Tensor, query: Tensor) -> Tensor:
        if stem.ndim != 4 or int(stem.shape[1]) != self.input_channels:
            raise ValueError("stem must be [B,input_channels,H,W]")
        query_ids = query.reshape(-1).long()
        if int(query_ids.numel()) != int(stem.shape[0]):
            raise ValueError("query count differs from stem batch")
        if bool(((query_ids < 0) | (query_ids >= self.num_queries)).any()):
            raise ValueError("query ID is outside the adapter vocabulary")
        feature = F.gelu(self.projection_norm(self.projection(self.input_norm(stem))))
        query_feature = self.query_embedding(query_ids).to(dtype=feature.dtype)
        query_map = query_feature[:, :, None, None].expand_as(feature)
        fused = torch.cat(
            (
                feature,
                query_map,
                feature * query_map,
                (feature - query_map).abs(),
            ),
            dim=1,
        )
        return self.output(F.gelu(self.fusion_norm(self.fusion(fused))))


def masked_bce_per_sample(
    logits: Tensor,
    target: Tensor,
    support: Tensor,
) -> Tensor:
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError("logits must be [B,1,H,W]")
    if target.shape != logits.shape:
        raise ValueError("target shape differs from logits")
    if support.shape != logits.shape:
        raise ValueError("support shape differs from logits")
    mask = support.to(dtype=logits.dtype)
    denominator = mask.flatten(1).sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("masked BCE received an empty support")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).flatten(1).sum(dim=1) / denominator


def map_summaries(probabilities: Tensor, support: Tensor) -> Dict[str, Tensor]:
    if probabilities.ndim != 4 or int(probabilities.shape[1]) != 1:
        raise ValueError("probabilities must be [B,1,H,W]")
    if support.shape != probabilities.shape:
        raise ValueError("support shape differs from probabilities")
    means = []
    top_means = []
    quantiles = []
    for batch_index in range(int(probabilities.shape[0])):
        values = probabilities[batch_index, 0][support[batch_index, 0].bool()]
        if int(values.numel()) == 0:
            raise ValueError("map summary received an empty support")
        top_count = max(1, int(math.ceil(0.20 * int(values.numel()))))
        means.append(values.mean())
        top_means.append(torch.topk(values, k=top_count, largest=True).values.mean())
        quantiles.append(torch.quantile(values, 0.90, interpolation="linear"))
    mean = torch.stack(means)
    top_quintile_mean = torch.stack(top_means)
    percentile_90 = torch.stack(quantiles)
    compatibility = 1.0 - 0.5 * mean - 0.5 * top_quintile_mean
    return {
        "support_mean": mean,
        "top_quintile_mean": top_quintile_mean,
        "percentile_90": percentile_90,
        "compatibility": compatibility,
    }


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def deterministic_deranged_queries(
    labels: Tensor,
    *,
    sample_indices: Tensor,
    epoch: int,
    namespace: int,
    num_queries: int = 5,
) -> Tensor:
    label_values = labels.detach().cpu().long().reshape(-1)
    indices = sample_indices.detach().cpu().long().reshape(-1)
    if int(label_values.numel()) != int(indices.numel()):
        raise ValueError("labels and sample indices differ")
    output = torch.empty_like(label_values)
    for position, (label, sample_index) in enumerate(
        zip(label_values.tolist(), indices.tolist())
    ):
        seed = local_seed(sample_index, epoch, namespace)
        generator = np.random.Generator(np.random.PCG64(seed))
        candidates = [query for query in range(int(num_queries)) if query != int(label)]
        output[position] = int(candidates[int(generator.integers(len(candidates)))])
    return output.to(device=labels.device)


def validate_role_parameter_parity(
    roles: Mapping[str, nn.Module],
) -> Tuple[int, Dict[str, int]]:
    counts = {name: parameter_count(module) for name, module in roles.items()}
    unique = set(counts.values())
    if len(unique) != 1:
        raise ValueError(f"adapter roles are not capacity matched: {counts}")
    return next(iter(unique)), counts
