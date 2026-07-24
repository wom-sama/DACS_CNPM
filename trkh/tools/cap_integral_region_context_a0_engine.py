from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CHANNELS = 256
FEATURE_SIZE = 16
UPSAMPLE_SIZE = 42
GRID_SIZE = 3
MIN_GRID_SIZE = 2
REGION_COUNT = 27
REGION_POOL_SIZE = 7
PIXEL_QUERY_KEY_CHANNELS = 32
REGION_QUERY_KEY_DIM = 32
LSTM_HIDDEN_DIM = 128
NETVLAD_CLUSTERS = 32
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)

PRIMARY_SEED = 20260724
REPEAT_SEED_OFFSET = 100000
SPATIAL_DERANGEMENT_SEED = 20265724
EPOCHS = 30
BATCH_SIZE = 32
BASE_LR = 3e-4
WARMUP_EPOCHS = 3
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
TRACE_EPOCHS = frozenset({1, 5, 10, 20, 30})

ROLE_NAMES = (
    "keeper_margin_control",
    "global_gap_linear_control",
    "integral_self_only_control",
    "cap_context_candidate",
    "cap_context_seed_repeat",
    "cap_spatial_deranged_control",
    "cap_cross_sample_context_control",
)
TRAINABLE_ROLES = ROLE_NAMES[1:]
CAP_ROLES = ROLE_NAMES[2:]
SAME_BUDGET_CAP_ROLES = (
    "cap_context_candidate",
    "cap_context_seed_repeat",
    "cap_spatial_deranged_control",
    "cap_cross_sample_context_control",
)

_ROLE_TO_CONTEXT_MODE = {
    "integral_self_only_control": "self_only",
    "cap_context_candidate": "context",
    "cap_context_seed_repeat": "context",
    "cap_spatial_deranged_control": "spatial_deranged",
    "cap_cross_sample_context_control": "cross_sample_context",
}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def string_sequence_sha256(values: Iterable[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def state_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(str(name).encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(np.asarray(values[name]))))
    return digest.hexdigest()


def enumerate_region_boxes() -> np.ndarray:
    step = UPSAMPLE_SIZE / GRID_SIZE
    boxes: List[List[int]] = []
    for column1 in range(GRID_SIZE + 1):
        for column2 in range(GRID_SIZE + 1):
            for row1 in range(GRID_SIZE + 1):
                for row2 in range(GRID_SIZE + 1):
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    width = x1 - x0
                    height = y1 - y0
                    valid_size = (
                        width >= step * MIN_GRID_SIZE
                        or height >= step * MIN_GRID_SIZE
                    )
                    if (
                        x1 > x0
                        and y1 > y0
                        and valid_size
                        and not (
                            x0 == y0 == 0
                            and x1 == y1 == UPSAMPLE_SIZE
                        )
                    ):
                        boxes.append([x0, y0, width, height])
    boxes.append([0, 0, UPSAMPLE_SIZE, UPSAMPLE_SIZE])
    result = np.asarray(boxes, dtype=np.int64)
    if result.shape != (REGION_COUNT, 4):
        raise RuntimeError(f"CAP region geometry differs: {result.shape}")
    return result


REGION_BOXES = enumerate_region_boxes()


def valid_support_boxes(valid_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(valid_masks, dtype=np.bool_)
    if masks.ndim != 3 or masks.shape[1:] != (FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError("Valid masks must have shape [N, 16, 16]")
    boxes: List[List[int]] = []
    for mask in masks:
        rows, columns = np.where(mask)
        if rows.size == 0:
            raise ValueError("A valid support is empty")
        y0 = int(rows.min())
        y1 = int(rows.max()) + 1
        x0 = int(columns.min())
        x1 = int(columns.max()) + 1
        if not bool(mask[y0:y1, x0:x1].all()):
            raise ValueError("Valid support is not rectangular")
        if int(mask.sum()) != (y1 - y0) * (x1 - x0):
            raise ValueError("Valid support has pixels outside its rectangle")
        boxes.append([x0, y0, x1, y1])
    return np.asarray(boxes, dtype=np.int64)


def _normalized_support_boxes(
    support_boxes: np.ndarray | Tensor | Sequence[Sequence[int]],
    batch_size: int,
) -> np.ndarray:
    if isinstance(support_boxes, Tensor):
        boxes = support_boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(support_boxes)
    boxes = np.asarray(boxes, dtype=np.int64)
    if boxes.shape != (batch_size, 4):
        raise ValueError(
            f"Support boxes must have shape [{batch_size}, 4], got {boxes.shape}"
        )
    if bool(
        (boxes[:, 0] < 0).any()
        or (boxes[:, 1] < 0).any()
        or (boxes[:, 2] > FEATURE_SIZE).any()
        or (boxes[:, 3] > FEATURE_SIZE).any()
        or (boxes[:, 2] <= boxes[:, 0]).any()
        or (boxes[:, 3] <= boxes[:, 1]).any()
    ):
        raise ValueError("Support boxes leave the 16x16 feature map")
    return boxes


def canonicalize_valid_support(
    features: Tensor,
    support_boxes: np.ndarray | Tensor | Sequence[Sequence[int]],
) -> Tensor:
    if features.ndim != 4 or tuple(features.shape[1:]) != (
        CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError("Features must have shape [B, 256, 16, 16]")
    boxes = _normalized_support_boxes(support_boxes, int(features.size(0)))
    restored: List[Tensor] = []
    for feature, (x0, y0, x1, y1) in zip(features, boxes.tolist()):
        crop = feature[:, y0:y1, x0:x1].unsqueeze(0)
        restored.append(
            F.interpolate(
                crop,
                size=(FEATURE_SIZE, FEATURE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        )
    return torch.cat(restored, dim=0)


def valid_support_global_average(
    features: Tensor,
    valid_masks: Tensor,
) -> Tensor:
    if features.ndim != 4 or tuple(features.shape[1:]) != (
        CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError("Features must have shape [B, 256, 16, 16]")
    if tuple(valid_masks.shape) != (
        int(features.size(0)),
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError("Valid masks do not align with features")
    weights = valid_masks.to(device=features.device, dtype=features.dtype)
    denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
    return (features * weights.unsqueeze(1)).sum(dim=(2, 3)) / denominator[:, None]


def sattolo_permutation(
    size: int,
    seed: np.random.SeedSequence,
) -> np.ndarray:
    if int(size) < 2:
        raise ValueError("Sattolo requires at least two entries")
    values = np.arange(int(size), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for upper in range(int(size) - 1, 0, -1):
        lower = int(rng.integers(0, upper))
        values[upper], values[lower] = values[lower], values[upper]
    if bool((values == np.arange(int(size), dtype=np.int64)).any()):
        raise RuntimeError("Sattolo permutation contains a fixed point")
    return values


def build_spatial_permutations(sample_indices: np.ndarray) -> np.ndarray:
    samples = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    permutations = np.stack(
        [
            sattolo_permutation(
                REGION_COUNT,
                np.random.SeedSequence(
                    [SPATIAL_DERANGEMENT_SEED, int(sample_index)]
                ),
            )
            for sample_index in samples.tolist()
        ],
        axis=0,
    )
    expected = np.arange(REGION_COUNT, dtype=np.int64)
    if not all(
        np.array_equal(np.sort(permutation), expected)
        for permutation in permutations
    ):
        raise RuntimeError("A spatial row is not a permutation")
    if bool((permutations == expected[None, :]).any()):
        raise RuntimeError("A spatial permutation contains a fixed point")
    return permutations


def _different_source_derangement(
    indices: np.ndarray,
    sample_indices: np.ndarray,
    sources: Sequence[str],
    seed: int,
) -> np.ndarray:
    partition = np.asarray(indices, dtype=np.int64).reshape(-1)
    ordered = partition[
        np.argsort(sample_indices[partition], kind="stable")
    ]
    if ordered.size < 2:
        raise ValueError("Cross-sample partition requires at least two rows")
    rng = np.random.default_rng(int(seed))
    offsets = rng.permutation(np.arange(1, ordered.size, dtype=np.int64))
    for offset in offsets:
        partners = np.roll(ordered, -int(offset))
        if all(
            sources[int(left)] != sources[int(right)]
            for left, right in zip(ordered.tolist(), partners.tolist())
        ):
            mapping = np.full(sample_indices.size, -1, dtype=np.int64)
            mapping[ordered] = partners
            return mapping
    raise ValueError("No label-blind different-source derangement exists")


def build_partitioned_cross_sample_mapping(
    partitions: Mapping[str, np.ndarray],
    sample_indices: np.ndarray,
    sources: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    samples = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if len(sources) != samples.size:
        raise ValueError("Source count differs from sample count")
    mapping = np.full(samples.size, -1, dtype=np.int64)
    details: Dict[str, object] = {}
    seen: set[int] = set()
    for offset, (name, indices) in enumerate(partitions.items()):
        partition = np.asarray(indices, dtype=np.int64).reshape(-1)
        if any(int(index) in seen for index in partition.tolist()):
            raise ValueError("Cross-sample partitions overlap")
        seen.update(int(index) for index in partition.tolist())
        local = _different_source_derangement(
            partition,
            samples,
            sources,
            int(seed) + offset,
        )
        partners = local[partition]
        allowed = set(partition.tolist())
        outside = sum(int(partner) not in allowed for partner in partners)
        source_overlap = sum(
            sources[int(left)] == sources[int(right)]
            for left, right in zip(partition.tolist(), partners.tolist())
        )
        self_pairs = int((partition == partners).sum())
        unique_partners = len(set(partners.tolist()))
        if (
            outside
            or source_overlap
            or self_pairs
            or unique_partners != int(partition.size)
        ):
            raise RuntimeError(f"Invalid {name} cross-sample derangement")
        mapping[partition] = partners
        details[str(name)] = {
            "rows": int(partition.size),
            "mapping_sha256": array_sha256(partners),
            "source_overlap": int(source_overlap),
            "outside_partition": int(outside),
            "self_pairs": int(self_pairs),
            "unique_partners": int(unique_partners),
        }
    if len(seen) != samples.size or bool((mapping < 0).any()):
        raise ValueError("Partitioned cross-sample derangement is incomplete")
    return mapping, details


def build_epoch_orders(
    fit_indices: np.ndarray,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> List[np.ndarray]:
    indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Fit indices must be a non-empty unique sequence")
    rng = np.random.default_rng(int(seed))
    return [
        indices[rng.permutation(indices.size)].astype(np.int64, copy=False)
        for _ in range(int(epochs))
    ]


def learning_rate(epoch_index: int) -> float:
    epoch = int(epoch_index)
    if epoch < 0 or epoch >= EPOCHS:
        raise ValueError(f"Epoch index must be in [0, {EPOCHS}): {epoch}")
    if epoch < WARMUP_EPOCHS:
        return BASE_LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    cosine_steps = EPOCHS - WARMUP_EPOCHS - 1
    progress = float(epoch - WARMUP_EPOCHS) / float(cosine_steps)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


class PixelContext(nn.Module):
    def __init__(
        self,
        channels: int = CHANNELS,
        query_key_channels: int = PIXEL_QUERY_KEY_CHANNELS,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.query = nn.Conv2d(
            self.channels, int(query_key_channels), kernel_size=1, bias=True
        )
        self.key = nn.Conv2d(
            self.channels, int(query_key_channels), kernel_size=1, bias=True
        )
        self.value = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=True
        )
        self.gamma = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        if inputs.ndim != 4 or int(inputs.size(1)) != self.channels:
            raise ValueError("Pixel context input has an invalid shape")
        batch, _, height, width = inputs.shape
        query = self.query(inputs).flatten(2).transpose(1, 2)
        key = self.key(inputs).flatten(2)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(inputs).flatten(2).transpose(1, 2)
        context = torch.bmm(attention, value)
        context = context.transpose(1, 2).reshape(
            batch, self.channels, height, width
        )
        return inputs + self.gamma * context, attention


class IntegralRegionExtractor(nn.Module):
    def __init__(
        self,
        *,
        upsample_size: int = UPSAMPLE_SIZE,
        pool_size: int = REGION_POOL_SIZE,
        boxes: np.ndarray = REGION_BOXES,
    ) -> None:
        super().__init__()
        value = np.asarray(boxes, dtype=np.int64)
        if value.shape != (REGION_COUNT, 4):
            raise ValueError("Integral-region boxes must have shape [27, 4]")
        self.upsample_size = int(upsample_size)
        self.pool_size = int(pool_size)
        self.register_buffer(
            "boxes",
            torch.as_tensor(value, dtype=torch.int64),
            persistent=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or int(inputs.size(1)) != CHANNELS:
            raise ValueError("Integral-region input has an invalid shape")
        upsampled = F.interpolate(
            inputs,
            size=(self.upsample_size, self.upsample_size),
            mode="bilinear",
            align_corners=False,
        )
        regions: List[Tensor] = []
        for x0, y0, width, height in self.boxes.detach().cpu().tolist():
            crop = upsampled[
                :, :, y0 : y0 + height, x0 : x0 + width
            ]
            regions.append(
                F.interpolate(
                    crop,
                    size=(self.pool_size, self.pool_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.stack(regions, dim=1)


class RegionContextAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = CHANNELS * REGION_POOL_SIZE * REGION_POOL_SIZE,
        hidden_dim: int = REGION_QUERY_KEY_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.beta_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        self.score = nn.Linear(self.hidden_dim, 1, bias=True)

    def forward(
        self,
        query_regions: Tensor,
        *,
        key_regions: Tensor | None = None,
        value_regions: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        keys = query_regions if key_regions is None else key_regions
        values = query_regions if value_regions is None else value_regions
        for name, tensor in (
            ("query", query_regions),
            ("key", keys),
            ("value", values),
        ):
            if (
                tensor.ndim != 3
                or int(tensor.size(1)) != REGION_COUNT
                or int(tensor.size(2)) != self.input_dim
            ):
                raise ValueError(f"{name} regions have an invalid shape")
        if not (
            int(query_regions.size(0))
            == int(keys.size(0))
            == int(values.size(0))
        ):
            raise ValueError("Region query/key/value batches differ")
        query = self.query(query_regions).unsqueeze(2)
        key = self.key(keys).unsqueeze(1)
        beta = torch.tanh(query + key + self.beta_bias)
        logits = self.score(beta).squeeze(-1)
        attention = torch.softmax(logits, dim=-1)

        channels = self.input_dim // (
            REGION_POOL_SIZE * REGION_POOL_SIZE
        )
        if channels * REGION_POOL_SIZE * REGION_POOL_SIZE != self.input_dim:
            raise ValueError("Region feature dimension is not a 7x7 map")
        region_means = values.reshape(
            values.size(0),
            REGION_COUNT,
            channels,
            REGION_POOL_SIZE,
            REGION_POOL_SIZE,
        ).mean(dim=(-1, -2))
        context_vectors = torch.bmm(attention, region_means)
        return context_vectors, attention


class ResidualLessNetVLAD(nn.Module):
    def __init__(
        self,
        feature_dim: int = LSTM_HIDDEN_DIM,
        cluster_count: int = NETVLAD_CLUSTERS,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.cluster_count = int(cluster_count)
        self.assignment = nn.Linear(
            self.feature_dim, self.cluster_count, bias=True
        )

    @property
    def output_dim(self) -> int:
        return self.feature_dim * self.cluster_count

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or int(inputs.size(2)) != self.feature_dim:
            raise ValueError("Residual-less NetVLAD input has an invalid shape")
        assignments = torch.softmax(self.assignment(inputs), dim=-1)
        responses = torch.einsum("btk,btd->bdk", assignments, inputs)
        responses = F.normalize(responses, p=2.0, dim=1, eps=1e-12)
        descriptor = responses.flatten(1)
        descriptor = F.normalize(descriptor, p=2.0, dim=1, eps=1e-12)
        return descriptor, assignments


class CAPIntegralRegionBinaryHead(nn.Module):
    def __init__(self, *, context_mode: str = "context") -> None:
        super().__init__()
        if context_mode not in {
            "context",
            "self_only",
            "spatial_deranged",
            "cross_sample_context",
        }:
            raise ValueError(f"Unknown CAP context mode: {context_mode}")
        self.context_mode = context_mode
        self.pixel_context = PixelContext()
        self.region_extractor = IntegralRegionExtractor()
        self.region_context = RegionContextAttention()
        self.lstm = nn.LSTM(
            input_size=CHANNELS,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.netvlad = ResidualLessNetVLAD()
        self.classifier = nn.Linear(self.netvlad.output_dim, 1, bias=True)

    @staticmethod
    def _flatten_regions(regions: Tensor) -> Tensor:
        if tuple(regions.shape[1:]) != (
            REGION_COUNT,
            CHANNELS,
            REGION_POOL_SIZE,
            REGION_POOL_SIZE,
        ):
            raise ValueError("CAP regions have an invalid shape")
        return regions.flatten(2)

    @staticmethod
    def _permute_regions(regions: Tensor, permutations: Tensor) -> Tensor:
        if tuple(permutations.shape) != (
            int(regions.size(0)),
            REGION_COUNT,
        ):
            raise ValueError("Spatial permutations do not align with regions")
        expected = torch.arange(
            REGION_COUNT, device=permutations.device, dtype=torch.long
        )
        if bool(
            torch.any(
                torch.sort(permutations.to(dtype=torch.long), dim=1).values
                != expected.unsqueeze(0)
            )
        ):
            raise ValueError("A spatial row is not a permutation")
        if bool(torch.any(permutations == expected.unsqueeze(0))):
            raise ValueError("A spatial permutation contains a fixed point")
        gather = permutations.to(
            device=regions.device, dtype=torch.long
        )[:, :, None, None, None]
        gather = gather.expand(
            -1,
            -1,
            int(regions.size(2)),
            int(regions.size(3)),
            int(regions.size(4)),
        )
        return torch.gather(regions, dim=1, index=gather)

    def forward(
        self,
        features: Tensor,
        support_boxes: np.ndarray | Tensor | Sequence[Sequence[int]],
        *,
        spatial_permutations: Tensor | None = None,
        partner_features: Tensor | None = None,
        partner_support_boxes: (
            np.ndarray | Tensor | Sequence[Sequence[int]] | None
        ) = None,
        return_auxiliary: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        own = canonicalize_valid_support(features, support_boxes)
        partner: Tensor | None = None
        if self.context_mode == "cross_sample_context":
            if partner_features is None or partner_support_boxes is None:
                raise ValueError("Cross-sample context requires partner inputs")
            partner = canonicalize_valid_support(
                partner_features, partner_support_boxes
            )
        elif partner_features is not None or partner_support_boxes is not None:
            raise ValueError("Partner inputs are only valid for cross-sample context")

        if partner is None:
            pixel_outputs, pixel_attention = self.pixel_context(own)
            regions = self.region_extractor(pixel_outputs)
            partner_regions = None
        else:
            combined, combined_attention = self.pixel_context(
                torch.cat([own, partner], dim=0)
            )
            own_count = int(own.size(0))
            pixel_outputs = combined[:own_count]
            pixel_attention = combined_attention[:own_count]
            regions_all = self.region_extractor(combined)
            regions = regions_all[:own_count]
            partner_regions = regions_all[own_count:]

        if self.context_mode == "spatial_deranged":
            if spatial_permutations is None:
                raise ValueError("Spatial-deranged CAP requires permutations")
            regions = self._permute_regions(regions, spatial_permutations)
        elif spatial_permutations is not None:
            raise ValueError("Spatial permutations are only valid for their control")

        flattened = self._flatten_regions(regions)
        if self.context_mode == "self_only":
            region_vectors = regions.mean(dim=(-1, -2))
            attention = torch.eye(
                REGION_COUNT,
                dtype=regions.dtype,
                device=regions.device,
            ).unsqueeze(0).expand(int(regions.size(0)), -1, -1)
        elif self.context_mode == "cross_sample_context":
            if partner_regions is None:
                raise RuntimeError("Cross-sample partner regions are absent")
            region_vectors, attention = self.region_context(
                flattened,
                key_regions=self._flatten_regions(partner_regions),
                value_regions=flattened,
            )
        else:
            region_vectors, attention = self.region_context(flattened)

        hidden, _ = self.lstm(region_vectors)
        descriptor, assignments = self.netvlad(hidden)
        logits = self.classifier(descriptor).squeeze(-1)
        if not return_auxiliary:
            return logits
        return logits, {
            "canonical_features": own,
            "pixel_outputs": pixel_outputs,
            "pixel_attention": pixel_attention,
            "region_attention": attention,
            "region_vectors": region_vectors,
            "lstm_hidden": hidden,
            "netvlad_assignments": assignments,
            "descriptor": descriptor,
        }


class GlobalGAPBinaryHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(CHANNELS, 1, bias=True)

    def forward(self, gap_features: Tensor) -> Tensor:
        if gap_features.ndim != 2 or int(gap_features.size(1)) != CHANNELS:
            raise ValueError("Global-GAP features must have shape [B, 256]")
        return self.classifier(gap_features).squeeze(-1)


def role_seed(role: str, fold: int) -> int:
    if role not in TRAINABLE_ROLES:
        raise ValueError(f"Role is not trainable: {role}")
    repeat = role == "cap_context_seed_repeat"
    return (
        PRIMARY_SEED
        + int(fold)
        + (REPEAT_SEED_OFFSET if repeat else 0)
    )


def initialize_role_model(
    role: str,
    *,
    fold: int,
    device: torch.device,
) -> nn.Module:
    seed = role_seed(role, fold)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if role == "global_gap_linear_control":
            model: nn.Module = GlobalGAPBinaryHead()
        else:
            model = CAPIntegralRegionBinaryHead(
                context_mode=_ROLE_TO_CONTEXT_MODE[role]
            )
    return model.to(device=device, dtype=torch.float32)


def model_state_arrays(model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        f"model.{name}": tensor.detach()
        .cpu()
        .numpy()
        .copy()
        for name, tensor in model.state_dict().items()
    }


def optimizer_state_arrays(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, np.ndarray]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    arrays: Dict[str, np.ndarray] = {}
    for parameter, state in optimizer.state.items():
        name = names.get(parameter)
        if name is None:
            raise RuntimeError("Optimizer contains an unknown parameter")
        for key, value in state.items():
            if isinstance(value, Tensor):
                arrays[f"optimizer.{name}.{key}"] = (
                    value.detach().cpu().numpy().copy()
                )
            elif isinstance(value, (int, float, bool)):
                arrays[f"optimizer.{name}.{key}"] = np.asarray(value)
            else:
                raise TypeError(
                    f"Unsupported optimizer state {name}.{key}: {type(value)}"
                )
    return arrays


def parameter_contract(model: nn.Module) -> Dict[str, object]:
    shapes = {
        name: [int(value) for value in parameter.shape]
        for name, parameter in model.named_parameters()
    }
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        "parameter_shapes": shapes,
        "parameter_shapes_sha256": json_sha256(shapes),
        "trainable_parameter_count": int(sum(trainable.values())),
        "trainable_tensor_count": int(len(trainable)),
    }


def keeper_margin(keeper_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(keeper_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Keeper probabilities must have shape [N, 5]")
    if not bool(np.isfinite(probabilities).all()):
        raise ValueError("Keeper probabilities are non-finite")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    rival = np.max(clipped[:, [0, 2, 3, 4]], axis=1)
    return np.log(clipped[:, FOCUS_CLASS]) - np.log(rival)


def calibrate_tp_retention_threshold(
    scores: np.ndarray,
    keeper_true_positive_mask: np.ndarray,
    *,
    maximum_break_fraction: float = 0.03,
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = np.asarray(
        keeper_true_positive_mask, dtype=np.bool_
    ).reshape(-1)
    if values.shape != positives.shape or not bool(np.isfinite(values).all()):
        raise ValueError("Calibration scores and TP mask do not align")
    positive_scores = np.sort(values[positives], kind="stable")
    if positive_scores.size == 0:
        raise ValueError("Calibration partition has no keeper class-1 TP")
    allowed = int(
        math.floor(float(maximum_break_fraction) * positive_scores.size)
    )
    if allowed <= 0:
        threshold = float(
            np.nextafter(positive_scores[0], -np.inf, dtype=np.float64)
        )
        boundary = "below_minimum"
    else:
        lower = float(positive_scores[allowed - 1])
        upper = float(positive_scores[allowed])
        if lower < upper:
            threshold = lower + (upper - lower) * 0.5
            boundary = "midpoint"
        else:
            threshold = float(
                np.nextafter(upper, -np.inf, dtype=np.float64)
            )
            boundary = "tie_next_lower"
    broken = int((positive_scores < threshold).sum())
    retained = float((positive_scores >= threshold).mean())
    if broken > allowed:
        raise RuntimeError("Calibration threshold exceeds the TP-break budget")
    return {
        "threshold": float(threshold),
        "boundary": boundary,
        "keeper_tp_rows": int(positive_scores.size),
        "allowed_tp_breaks": int(allowed),
        "observed_tp_breaks": int(broken),
        "keeper_tp_retention": retained,
    }


def apply_keeper_suppression(
    keeper_probabilities: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, np.ndarray]:
    probabilities = np.asarray(keeper_probabilities, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if probabilities.shape != (values.size, 5):
        raise ValueError("Keeper probabilities and scores do not align")
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    rival_probabilities = probabilities.copy()
    rival_probabilities[:, FOCUS_CLASS] = -np.inf
    replacements = rival_probabilities.argmax(axis=1).astype(np.int64)
    suppressed = (predictions == FOCUS_CLASS) & (
        values < float(threshold)
    )
    updated = predictions.copy()
    updated[suppressed] = replacements[suppressed]
    return {
        "keeper_predictions": predictions,
        "predictions": updated,
        "replacements": replacements,
        "suppressed": suppressed,
        "retained": ~suppressed,
    }


def _batch_boxes(boxes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(boxes, dtype=np.int64)[
        np.asarray(indices, dtype=np.int64)
    ]


def _role_logits(
    model: nn.Module,
    role: str,
    *,
    features: Tensor,
    gap_features: Tensor,
    support_boxes: np.ndarray,
    indices: np.ndarray,
    spatial_permutations: np.ndarray | None,
    cross_sample_mapping: np.ndarray | None,
) -> Tensor:
    batch_indices = np.asarray(indices, dtype=np.int64)
    device = features.device
    batch = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
    if role == "global_gap_linear_control":
        return model(gap_features.index_select(0, batch))
    kwargs: Dict[str, object] = {}
    if role == "cap_spatial_deranged_control":
        if spatial_permutations is None:
            raise ValueError("Spatial permutations are absent")
        kwargs["spatial_permutations"] = torch.as_tensor(
            spatial_permutations[batch_indices],
            dtype=torch.long,
            device=device,
        )
    if role == "cap_cross_sample_context_control":
        if cross_sample_mapping is None:
            raise ValueError("Cross-sample mapping is absent")
        partner_indices = np.asarray(
            cross_sample_mapping[batch_indices], dtype=np.int64
        )
        if bool((partner_indices < 0).any()):
            raise ValueError("Cross-sample mapping is incomplete")
        partners = torch.as_tensor(
            partner_indices, dtype=torch.long, device=device
        )
        kwargs["partner_features"] = features.index_select(0, partners)
        kwargs["partner_support_boxes"] = _batch_boxes(
            support_boxes, partner_indices
        )
    return model(
        features.index_select(0, batch),
        _batch_boxes(support_boxes, batch_indices),
        **kwargs,
    )


def _score_partition(
    model: nn.Module,
    role: str,
    indices: np.ndarray,
    *,
    features: Tensor,
    gap_features: Tensor,
    support_boxes: np.ndarray,
    spatial_permutations: np.ndarray | None,
    cross_sample_mapping: np.ndarray | None,
) -> np.ndarray:
    partition = np.asarray(indices, dtype=np.int64).reshape(-1)
    logits: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(partition.size), BATCH_SIZE):
            batch = partition[start : start + BATCH_SIZE]
            output = _role_logits(
                model,
                role,
                features=features,
                gap_features=gap_features,
                support_boxes=support_boxes,
                indices=batch,
                spatial_permutations=spatial_permutations,
                cross_sample_mapping=cross_sample_mapping,
            )
            logits.append(
                output.detach().cpu().numpy().astype(np.float32, copy=True)
            )
    return np.concatenate(logits).astype(np.float32, copy=False)


@dataclass
class TrainedRoleFold:
    model: nn.Module
    evidence: Dict[str, object]


def train_role_fold(
    *,
    role: str,
    fold: int,
    features: Tensor,
    gap_features: Tensor,
    targets: Tensor,
    support_boxes: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    held_indices: np.ndarray,
    spatial_permutations: np.ndarray | None,
    cross_sample_mapping: np.ndarray | None,
    device: torch.device,
) -> TrainedRoleFold:
    if role not in TRAINABLE_ROLES:
        raise ValueError(f"Unknown trainable CAP role: {role}")
    resolved_device = features.device
    requested_index = (
        torch.cuda.current_device()
        if device.type == "cuda" and device.index is None
        else device.index
    )
    if (
        gap_features.device != resolved_device
        or resolved_device.type != device.type
        or (
            device.type == "cuda"
            and resolved_device.index != requested_index
        )
    ):
        raise ValueError("Feature tensors are not on the requested device")
    seed = role_seed(role, fold)
    model = initialize_role_model(
        role, fold=fold, device=resolved_device
    )
    initial_state = model_state_arrays(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    orders = build_epoch_orders(fit_indices, seed=seed)
    trace: List[Dict[str, object]] = []
    update_count = 0
    model.train()
    for epoch_index, order in enumerate(orders):
        lr = learning_rate(epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        row_count = 0
        gradient_norm_max = 0.0
        for start in range(0, int(order.size), BATCH_SIZE):
            batch_indices = order[start : start + BATCH_SIZE]
            batch = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=resolved_device,
            )
            logits = _role_logits(
                model,
                role,
                features=features,
                gap_features=gap_features,
                support_boxes=support_boxes,
                indices=batch_indices,
                spatial_permutations=spatial_permutations,
                cross_sample_mapping=cross_sample_mapping,
            )
            binary_targets = (
                targets.index_select(0, batch) == FOCUS_CLASS
            ).to(dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(
                logits, binary_targets, reduction="mean"
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite CAP loss role={role} fold={fold} "
                    f"epoch={epoch_index + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()
            batch_rows = int(batch.size(0))
            loss_sum += float(loss.detach()) * batch_rows
            row_count += batch_rows
            update_count += 1
            gradient_norm_max = max(
                gradient_norm_max, float(gradient_norm.detach())
            )
        if epoch_index + 1 in TRACE_EPOCHS:
            trace.append(
                {
                    "role": role,
                    "outer_fold": int(fold),
                    "epoch": int(epoch_index + 1),
                    "learning_rate": float(lr),
                    "mean_loss": float(loss_sum / row_count),
                    "rows": int(row_count),
                    "updates_cumulative": int(update_count),
                    "maximum_preclip_gradient_norm": float(
                        gradient_norm_max
                    ),
                }
            )
    calibration_logits = _score_partition(
        model,
        role,
        calibration_indices,
        features=features,
        gap_features=gap_features,
        support_boxes=support_boxes,
        spatial_permutations=spatial_permutations,
        cross_sample_mapping=cross_sample_mapping,
    )
    held_logits = _score_partition(
        model,
        role,
        held_indices,
        features=features,
        gap_features=gap_features,
        support_boxes=support_boxes,
        spatial_permutations=spatial_permutations,
        cross_sample_mapping=cross_sample_mapping,
    )
    final_state = model_state_arrays(model)
    final_state.update(optimizer_state_arrays(model, optimizer))
    return TrainedRoleFold(
        model=model,
        evidence={
            "role": role,
            "outer_fold": int(fold),
            "seed": int(seed),
            "parameter_contract": parameter_contract(model),
            "initial_state_sha256": state_arrays_sha256(initial_state),
            "orders": orders,
            "order_hashes": [array_sha256(order) for order in orders],
            "all_orders_sha256": array_sha256(np.concatenate(orders)),
            "calibration_logits": calibration_logits,
            "calibration_scores": (
                1.0 / (1.0 + np.exp(-calibration_logits.astype(np.float64)))
            ).astype(np.float32),
            "held_logits": held_logits,
            "held_scores": (
                1.0 / (1.0 + np.exp(-held_logits.astype(np.float64)))
            ).astype(np.float32),
            "state": final_state,
            "state_sha256": state_arrays_sha256(final_state),
            "trace": trace,
            "update_count": int(update_count),
            "calibration_score_count": 1,
            "held_score_count": 1,
            "score_epoch": EPOCHS,
        },
    )


def _numpy_softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def _numpy_bilinear_resize_nchw(
    value: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    inputs = np.asarray(value, dtype=np.float64)
    if inputs.ndim != 4:
        raise ValueError("Bilinear oracle expects NCHW")
    _, _, input_height, input_width = inputs.shape

    def coordinates(input_size: int, output_size: int) -> Tuple[np.ndarray, ...]:
        source = (
            (np.arange(output_size, dtype=np.float64) + 0.5)
            * input_size
            / output_size
            - 0.5
        )
        source = np.clip(source, 0.0, float(input_size - 1))
        lower = np.floor(source).astype(np.int64)
        upper = np.minimum(lower + 1, input_size - 1)
        fraction = source - lower
        return lower, upper, fraction

    y0, y1, wy = coordinates(input_height, int(output_height))
    x0, x1, wx = coordinates(input_width, int(output_width))
    top = (
        inputs[:, :, y0[:, None], x0[None, :]]
        * (1.0 - wx)[None, None, None, :]
        + inputs[:, :, y0[:, None], x1[None, :]]
        * wx[None, None, None, :]
    )
    bottom = (
        inputs[:, :, y1[:, None], x0[None, :]]
        * (1.0 - wx)[None, None, None, :]
        + inputs[:, :, y1[:, None], x1[None, :]]
        * wx[None, None, None, :]
    )
    return (
        top * (1.0 - wy)[None, None, :, None]
        + bottom * wy[None, None, :, None]
    )


def _numpy_lstm(
    inputs: np.ndarray,
    *,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> np.ndarray:
    sequence = np.asarray(inputs, dtype=np.float64)
    batch, steps, _ = sequence.shape
    hidden_size = weight_hh.shape[1]
    hidden = np.zeros((batch, hidden_size), dtype=np.float64)
    cell = np.zeros_like(hidden)
    outputs: List[np.ndarray] = []
    for step in range(steps):
        gates = (
            sequence[:, step] @ weight_ih.T
            + hidden @ weight_hh.T
            + bias_ih
            + bias_hh
        )
        input_gate, forget_gate, candidate, output_gate = np.split(
            gates, 4, axis=1
        )
        input_gate = 1.0 / (1.0 + np.exp(-input_gate))
        forget_gate = 1.0 / (1.0 + np.exp(-forget_gate))
        candidate = np.tanh(candidate)
        output_gate = 1.0 / (1.0 + np.exp(-output_gate))
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden.copy())
    return np.stack(outputs, axis=1)


def engineering_oracles() -> Dict[str, object]:
    generator = np.random.default_rng(20260724)
    errors: Dict[str, float] = {}
    checks: Dict[str, bool] = {}

    pixel = PixelContext(channels=3, query_key_channels=2).to(
        dtype=torch.float64
    )
    with torch.no_grad():
        for parameter in pixel.parameters():
            parameter.copy_(
                torch.as_tensor(
                    generator.normal(0.0, 0.2, size=parameter.shape),
                    dtype=torch.float64,
                )
            )
        pixel.gamma.copy_(torch.tensor(0.3, dtype=torch.float64))
    pixel_input = generator.normal(size=(2, 3, 3, 2))
    pixel_torch, attention_torch = pixel(
        torch.as_tensor(pixel_input, dtype=torch.float64)
    )
    query = np.einsum(
        "oc,bchw->bohw", pixel.query.weight.detach().numpy()[:, :, 0, 0], pixel_input
    ) + pixel.query.bias.detach().numpy()[None, :, None, None]
    key = np.einsum(
        "oc,bchw->bohw", pixel.key.weight.detach().numpy()[:, :, 0, 0], pixel_input
    ) + pixel.key.bias.detach().numpy()[None, :, None, None]
    value = np.einsum(
        "oc,bchw->bohw", pixel.value.weight.detach().numpy()[:, :, 0, 0], pixel_input
    ) + pixel.value.bias.detach().numpy()[None, :, None, None]
    query_flat = query.reshape(2, 2, 6).transpose(0, 2, 1)
    key_flat = key.reshape(2, 2, 6)
    attention_numpy = _numpy_softmax(
        np.matmul(query_flat, key_flat), axis=-1
    )
    context_numpy = np.matmul(
        attention_numpy, value.reshape(2, 3, 6).transpose(0, 2, 1)
    )
    context_numpy = context_numpy.transpose(0, 2, 1).reshape(2, 3, 3, 2)
    pixel_numpy = pixel_input + 0.3 * context_numpy
    errors["pixel_attention"] = float(
        np.max(np.abs(attention_torch.detach().numpy() - attention_numpy))
    )
    errors["pixel_output"] = float(
        np.max(np.abs(pixel_torch.detach().numpy() - pixel_numpy))
    )

    canonical_input = generator.normal(
        size=(1, CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    )
    canonical_box = np.asarray([[3, 2, 13, 15]], dtype=np.int64)
    canonical_torch = canonicalize_valid_support(
        torch.as_tensor(canonical_input, dtype=torch.float64),
        canonical_box,
    )
    canonical_numpy = _numpy_bilinear_resize_nchw(
        canonical_input[:, :, 2:15, 3:13],
        FEATURE_SIZE,
        FEATURE_SIZE,
    )
    errors["valid_support_resize"] = float(
        np.max(np.abs(canonical_torch.detach().numpy() - canonical_numpy))
    )

    extractor = IntegralRegionExtractor().to(dtype=torch.float64)
    region_input = generator.normal(
        size=(1, CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
    )
    regions_torch = extractor(
        torch.as_tensor(region_input, dtype=torch.float64)
    ).detach().numpy()
    upsampled_numpy = _numpy_bilinear_resize_nchw(
        region_input, UPSAMPLE_SIZE, UPSAMPLE_SIZE
    )
    first = REGION_BOXES[0]
    x0, y0, width, height = [int(value) for value in first]
    first_numpy = _numpy_bilinear_resize_nchw(
        upsampled_numpy[:, :, y0 : y0 + height, x0 : x0 + width],
        REGION_POOL_SIZE,
        REGION_POOL_SIZE,
    )
    errors["first_region_pool"] = float(
        np.max(np.abs(regions_torch[:, 0] - first_numpy))
    )

    value_regions = generator.normal(
        size=(2, REGION_COUNT, REGION_POOL_SIZE * REGION_POOL_SIZE)
    )
    compact_attention = RegionContextAttention(
        input_dim=REGION_POOL_SIZE * REGION_POOL_SIZE,
        hidden_dim=3,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        compact_attention.query.weight.copy_(
            torch.as_tensor(
                generator.normal(
                    0.0, 0.15, size=compact_attention.query.weight.shape
                ),
                dtype=torch.float64,
            )
        )
        compact_attention.key.weight.copy_(
            torch.as_tensor(
                generator.normal(
                    0.0, 0.15, size=compact_attention.key.weight.shape
                ),
                dtype=torch.float64,
            )
        )
        compact_attention.beta_bias.copy_(
            torch.as_tensor(
                generator.normal(
                    0.0, 0.15, size=compact_attention.beta_bias.shape
                ),
                dtype=torch.float64,
            )
        )
        compact_attention.score.weight.copy_(
            torch.as_tensor(
                generator.normal(
                    0.0, 0.15, size=compact_attention.score.weight.shape
                ),
                dtype=torch.float64,
            )
        )
        compact_attention.score.bias.copy_(
            torch.as_tensor(
                generator.normal(
                    0.0, 0.15, size=compact_attention.score.bias.shape
                ),
                dtype=torch.float64,
            )
        )
    context_torch, region_alpha_torch = compact_attention(
        torch.as_tensor(value_regions, dtype=torch.float64),
        key_regions=torch.as_tensor(
            np.flip(value_regions, axis=0).copy(), dtype=torch.float64
        ),
        value_regions=torch.as_tensor(value_regions, dtype=torch.float64),
    )
    q_numpy = value_regions @ compact_attention.query.weight.detach().numpy().T
    paired = np.flip(value_regions, axis=0).copy()
    k_numpy = paired @ compact_attention.key.weight.detach().numpy().T
    beta_numpy = np.tanh(
        q_numpy[:, :, None, :]
        + k_numpy[:, None, :, :]
        + compact_attention.beta_bias.detach().numpy()
    )
    logits_numpy = (
        beta_numpy
        @ compact_attention.score.weight.detach().numpy().T
        + compact_attention.score.bias.detach().numpy()
    ).squeeze(-1)
    alpha_numpy = _numpy_softmax(logits_numpy, axis=-1)
    means_numpy = value_regions.reshape(
        2, REGION_COUNT, 1, REGION_POOL_SIZE, REGION_POOL_SIZE
    ).mean(axis=(-1, -2))
    context_numpy = np.matmul(alpha_numpy, means_numpy)
    errors["region_attention"] = float(
        np.max(
            np.abs(region_alpha_torch.detach().numpy() - alpha_numpy)
        )
    )
    errors["cross_sample_qkv"] = float(
        np.max(np.abs(context_torch.detach().numpy() - context_numpy))
    )
    full_context = np.matmul(alpha_numpy, value_regions)
    full_context_gap = full_context.reshape(
        2, REGION_COUNT, 1, REGION_POOL_SIZE, REGION_POOL_SIZE
    ).mean(axis=(-1, -2))
    errors["gap_context_equivalence"] = float(
        np.max(np.abs(full_context_gap - context_numpy))
    )

    lstm = nn.LSTM(4, 3, num_layers=1, batch_first=True).to(
        dtype=torch.float64
    )
    with torch.no_grad():
        for parameter in lstm.parameters():
            parameter.copy_(
                torch.as_tensor(
                    generator.normal(0.0, 0.2, size=parameter.shape),
                    dtype=torch.float64,
                )
            )
    lstm_input = generator.normal(size=(2, 5, 4))
    lstm_torch, _ = lstm(torch.as_tensor(lstm_input, dtype=torch.float64))
    lstm_numpy = _numpy_lstm(
        lstm_input,
        weight_ih=lstm.weight_ih_l0.detach().numpy(),
        weight_hh=lstm.weight_hh_l0.detach().numpy(),
        bias_ih=lstm.bias_ih_l0.detach().numpy(),
        bias_hh=lstm.bias_hh_l0.detach().numpy(),
    )
    errors["lstm_order"] = float(
        np.max(np.abs(lstm_torch.detach().numpy() - lstm_numpy))
    )
    reversed_torch, _ = lstm(
        torch.as_tensor(lstm_input[:, ::-1].copy(), dtype=torch.float64)
    )
    checks["lstm_order_changes_output"] = bool(
        not np.allclose(
            lstm_torch.detach().numpy(),
            reversed_torch.detach().numpy(),
            atol=1e-10,
            rtol=0.0,
        )
    )

    netvlad = ResidualLessNetVLAD(
        feature_dim=3, cluster_count=4
    ).to(dtype=torch.float64)
    with torch.no_grad():
        for parameter in netvlad.parameters():
            parameter.copy_(
                torch.as_tensor(
                    generator.normal(0.0, 0.2, size=parameter.shape),
                    dtype=torch.float64,
                )
            )
    vlad_input = generator.normal(size=(2, 5, 3))
    descriptor_torch, assignment_torch = netvlad(
        torch.as_tensor(vlad_input, dtype=torch.float64)
    )
    assignment_numpy = _numpy_softmax(
        vlad_input @ netvlad.assignment.weight.detach().numpy().T
        + netvlad.assignment.bias.detach().numpy(),
        axis=-1,
    )
    responses = np.einsum("btk,btd->bdk", assignment_numpy, vlad_input)
    responses /= np.maximum(
        np.linalg.norm(responses, axis=1, keepdims=True), 1e-12
    )
    descriptor_numpy = responses.reshape(2, -1)
    descriptor_numpy /= np.maximum(
        np.linalg.norm(descriptor_numpy, axis=1, keepdims=True), 1e-12
    )
    errors["netvlad_assignments"] = float(
        np.max(
            np.abs(assignment_torch.detach().numpy() - assignment_numpy)
        )
    )
    errors["netvlad_descriptor"] = float(
        np.max(
            np.abs(descriptor_torch.detach().numpy() - descriptor_numpy)
        )
    )

    spatial = build_spatial_permutations(
        np.asarray([2, 58, 163], dtype=np.int64)
    )
    checks["region_geometry_count"] = REGION_BOXES.shape == (27, 4)
    checks["region_geometry_full_last"] = bool(
        np.array_equal(
            REGION_BOXES[-1],
            np.asarray([0, 0, 42, 42], dtype=np.int64),
        )
    )
    checks["sattolo_zero_fixed_points"] = bool(
        not (
            spatial
            == np.arange(REGION_COUNT, dtype=np.int64)[None, :]
        ).any()
    )
    threshold = calibrate_tp_retention_threshold(
        np.asarray([0.1, 0.2, 0.2, 0.3, 0.9], dtype=np.float64),
        np.ones(5, dtype=np.bool_),
        maximum_break_fraction=0.4,
    )
    checks["threshold_tie_retains_boundary"] = bool(
        threshold["boundary"] == "tie_next_lower"
        and int(threshold["observed_tp_breaks"]) == 1
    )
    action = apply_keeper_suppression(
        np.asarray(
            [
                [0.1, 0.7, 0.05, 0.1, 0.05],
                [0.6, 0.2, 0.1, 0.05, 0.05],
            ],
            dtype=np.float64,
        ),
        np.asarray([0.1, 0.1], dtype=np.float64),
        0.2,
    )
    checks["action_only_suppresses_keeper_class1"] = bool(
        action["predictions"].tolist() == [0, 0]
        and action["suppressed"].tolist() == [True, False]
    )
    for name, error in errors.items():
        checks[f"{name}_matches_numpy"] = float(error) <= 2e-10
    return {
        "checks": checks,
        "errors": errors,
        "passed": all(checks.values()),
    }
