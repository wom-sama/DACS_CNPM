from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROWS = 763
FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
FOCUS_CLASS = 1
RIVALS = (0, 2, 4)
SCORE_NAMES = (
    "union",
    "class1_vs_0",
    "class1_vs_2",
    "class1_vs_4",
)
ROLE_NAMES = (
    "ddf_full",
    "static_matched",
    "ddf_spatial_only",
    "ddf_channel_only",
    "ddf_full_repeat",
)
PRIMARY_SEED = 20260729
REPEAT_OFFSET = 100000
BASE_LR = 0.003
WEIGHT_DECAY = 0.0001
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
PAIR_LOSS_WEIGHT = 0.5
BBOX_ATTENTION_WEIGHT = 0.05
BBOX_ATTENTION_CLAMP_MIN = 1e-8
FILTER_EPSILON = 1e-10
FILTER_GAIN = math.sqrt(2.0) / 3.0
MASKED_BN_EPSILON = 1e-5
INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.229, 0.224, 0.225)
INPUT_MEAN_FP32_SHA256 = (
    "64adba3379585c990c33ac8a768652797811c460965970493bacbc150b4709e3"
)
INPUT_STD_FP32_SHA256 = (
    "c917c2b282ce5e9dafe8456bfca484a079a72b4eff1a4587f571bc0e2f4f2bfa"
)

FULL_MODE = "full"
SPATIAL_ONLY_MODE = "spatial_only"
CHANNEL_ONLY_MODE = "channel_only"
DDF_MODES = (FULL_MODE, SPATIAL_ONLY_MODE, CHANNEL_ONLY_MODE)

CAUSAL_CLEAN = "clean"
CAUSAL_SPATIAL_NEUTRAL = "spatial_neutral"
CAUSAL_CHANNEL_NEUTRAL = "channel_neutral"
CAUSAL_MODES = (
    CAUSAL_CLEAN,
    CAUSAL_SPATIAL_NEUTRAL,
    CAUSAL_CHANNEL_NEUTRAL,
)

LOCKED_NONWRAP_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

REQUIRED_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def input_normalization_contract() -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(INPUT_MEAN, dtype=np.float32)
    std = np.asarray(INPUT_STD, dtype=np.float32)
    if array_sha256(mean) != INPUT_MEAN_FP32_SHA256:
        raise RuntimeError("ImageNet FP32 mean bytes changed")
    if array_sha256(std) != INPUT_STD_FP32_SHA256:
        raise RuntimeError("ImageNet FP32 std bytes changed")
    return mean, std


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(str(name).encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(np.asarray(values[name]))))
    return digest.hexdigest()


def assert_required_thread_environment() -> None:
    mismatches = {
        name: {"required": required, "observed": os.environ.get(name)}
        for name, required in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != required
    }
    if mismatches:
        raise RuntimeError(
            "The v2 launcher thread environment is not locked: "
            + json.dumps(mismatches, sort_keys=True)
        )


def configure_torch_determinism() -> None:
    """Apply the PyTorch half of the prospective deterministic launcher lock.

    Thread environment variables must have been set before importing scientific
    libraries.  This helper deliberately does not mutate them after import.
    """

    assert_required_thread_environment()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _positive_zero_like(value: Tensor) -> Tensor:
    return torch.zeros((), dtype=value.dtype, device=value.device)


def _canonical_mask(
    mask: Tensor,
    *,
    reference: Tensor,
    name: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tensor:
    value = mask if isinstance(mask, Tensor) else torch.as_tensor(mask)
    value = value.to(device=reference.device)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if not torch.jit.is_tracing():
        height, width = (
            (int(reference.size(2)), int(reference.size(3)))
            if spatial_shape is None
            else spatial_shape
        )
        expected = (int(reference.size(0)), 1, height, width)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape [B, 1, H, W] or [B, H, W]"
            )
    return value.to(dtype=torch.bool)


def literal_zero_mask(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim != 4:
        raise ValueError("Masked values must have shape [B, C, H, W]")
    valid = _canonical_mask(
        mask,
        reference=values,
        name="valid mask",
    )
    return torch.where(valid, values, _positive_zero_like(values))


def validity_weighted_channel_pool(inputs: Tensor, mask: Tensor) -> Tensor:
    if inputs.ndim != 4:
        raise ValueError("Channel pooling input must have shape [B, C, H, W]")
    valid = _canonical_mask(
        mask,
        reference=inputs,
        name="channel-pooling mask",
    )
    counts = valid.sum(dim=(2, 3), keepdim=True)
    if not torch.jit.is_tracing() and bool((counts <= 0).any()):
        raise ValueError("Every sample needs at least one valid pooling cell")
    numerator = literal_zero_mask(inputs, valid).sum(
        dim=(2, 3), keepdim=True
    )
    return numerator / counts.to(dtype=inputs.dtype)


def masked_spatial_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 4:
        raise ValueError("Attention logits must have shape [B, T, H, W]")
    valid = _canonical_mask(
        mask,
        reference=logits,
        name="attention mask",
    )
    counts = valid.sum(dim=(2, 3), keepdim=True)
    if not torch.jit.is_tracing() and bool((counts <= 0).any()):
        raise ValueError("Masked softmax cannot consume an empty validity map")
    negative = torch.full(
        (),
        torch.finfo(logits.dtype).min,
        dtype=logits.dtype,
        device=logits.device,
    )
    selected = torch.where(valid, logits, negative)
    maximum = selected.amax(dim=(2, 3), keepdim=True)
    exponentials = torch.where(
        valid,
        torch.exp(logits - maximum),
        _positive_zero_like(logits),
    )
    denominator = exponentials.sum(dim=(2, 3), keepdim=True)
    return torch.where(
        valid,
        exponentials / denominator,
        _positive_zero_like(logits),
    )


class MaskedBatchNorm2d(nn.Module):
    """BatchNorm2d with statistics and outputs restricted to valid cells.

    Optimizer-step forwards use population batch variance and increment only
    ``num_batches_tracked``.  Running buffers are updated exclusively by the
    explicit, fit-only sequential recalibration pass.
    """

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = MASKED_BN_EPSILON,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))
        self.register_buffer(
            "num_batches_tracked",
            torch.zeros((), dtype=torch.long),
        )

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        if inputs.ndim != 4 or (
            not torch.jit.is_tracing()
            and int(inputs.size(1)) != self.num_features
        ):
            raise ValueError("MaskedBatchNorm2d input has the wrong shape")
        valid = _canonical_mask(
            mask,
            reference=inputs,
            name="masked-BN mask",
        )
        if self.training:
            count = valid.sum()
            if not torch.jit.is_tracing() and int(count) <= 1:
                raise ValueError("MaskedBatchNorm2d requires M > 1")
            selected = literal_zero_mask(inputs, valid)
            divisor = count.to(dtype=inputs.dtype)
            mean = selected.sum(dim=(0, 2, 3)) / divisor
            centered = torch.where(
                valid,
                inputs - mean.view(1, -1, 1, 1),
                _positive_zero_like(inputs),
            )
            variance = centered.square().sum(dim=(0, 2, 3)) / divisor
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean
            variance = self.running_var
        normalized = (
            inputs - mean.view(1, -1, 1, 1)
        ) / torch.sqrt(variance.view(1, -1, 1, 1) + self.eps)
        affine = (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )
        return literal_zero_mask(affine, valid)


class MaskedPopulationAccumulator:
    """CPU FP64 sufficient statistics for one masked normalization site."""

    def __init__(self, channels: int) -> None:
        self.channels = int(channels)
        self.sum = torch.zeros(self.channels, dtype=torch.float64)
        self.sumsq = torch.zeros(self.channels, dtype=torch.float64)
        self.count = 0

    def update(self, inputs: Tensor, mask: Tensor) -> None:
        values = inputs.detach().to(device="cpu", dtype=torch.float64)
        valid = _canonical_mask(
            mask.detach().to(device="cpu"),
            reference=values,
            name="recalibration mask",
        )
        if int(values.size(1)) != self.channels:
            raise ValueError("Recalibration channel count changed")
        selected = literal_zero_mask(values, valid)
        self.sum += selected.sum(dim=(0, 2, 3))
        self.sumsq += selected.square().sum(dim=(0, 2, 3))
        self.count += int(valid.sum())

    def finalize(self) -> Tuple[Tensor, Tensor]:
        if self.count <= 1:
            raise ValueError("Masked population statistics require M > 1")
        mean = self.sum / float(self.count)
        variance = torch.clamp(
            self.sumsq / float(self.count) - mean.square(),
            min=0.0,
        )
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()):
            raise FloatingPointError("Non-finite masked population statistics")
        return mean, variance


@dataclass(frozen=True)
class PreparedSurfaceInputs:
    values: Tensor
    valid64: Tensor
    valid32: Tensor
    valid16: Tensor

    def take(self, indices: Tensor | np.ndarray | Sequence[int]) -> "PreparedSurfaceInputs":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.values.device)
        return PreparedSurfaceInputs(
            values=self.values.index_select(0, index),
            valid64=self.valid64.index_select(0, index),
            valid32=self.valid32.index_select(0, index),
            valid16=self.valid16.index_select(0, index),
        )


def unpack_valid_masks_little(packed: np.ndarray) -> np.ndarray:
    """Decode the retained cache with its mandatory little-endian bit order."""

    values = np.asarray(packed)
    if values.ndim != 2 or values.shape[1] != 8192 or values.dtype != np.uint8:
        raise ValueError(
            "Packed validity masks must be uint8 with shape [N, 8192]"
        )
    unpacked = np.unpackbits(
        values,
        axis=1,
        count=256 * 256,
        bitorder="little",
    )
    return unpacked.reshape(values.shape[0], 256, 256).astype(
        np.bool_,
        copy=False,
    )


def prepare_cached_model_inputs(
    model_srgb_uint8: Tensor,
    valid256: Tensor,
) -> PreparedSurfaceInputs:
    if model_srgb_uint8.ndim != 4 or model_srgb_uint8.shape[1:] != (
        3,
        256,
        256,
    ):
        raise ValueError("Cached sRGB tensor must have shape [B, 3, 256, 256]")
    if model_srgb_uint8.dtype != torch.uint8:
        raise ValueError("Cached sRGB tensor must be uint8")
    valid = _canonical_mask(
        valid256,
        reference=model_srgb_uint8,
        name="valid256",
    )
    values = model_srgb_uint8.to(dtype=torch.float32) / 255.0
    mean_array, std_array = input_normalization_contract()
    mean = values.new_tensor(mean_array).view(1, 3, 1, 1)
    std = values.new_tensor(std_array).view(1, 3, 1, 1)
    normalized = (values - mean) / std
    normalized = literal_zero_mask(normalized, valid)
    denominator = F.avg_pool2d(
        valid.to(dtype=normalized.dtype),
        kernel_size=4,
        stride=4,
    )
    numerator = F.avg_pool2d(normalized, kernel_size=4, stride=4)
    valid64 = denominator > 0
    prepared = torch.where(
        valid64,
        numerator / denominator.clamp_min(1.0 / 16.0),
        _positive_zero_like(numerator),
    )
    valid32 = F.max_pool2d(
        valid64.to(dtype=prepared.dtype),
        kernel_size=3,
        stride=2,
        padding=1,
    ) > 0
    valid16 = F.max_pool2d(
        valid32.to(dtype=prepared.dtype),
        kernel_size=3,
        stride=2,
        padding=1,
    ) > 0
    if bool((valid64.sum(dim=(1, 2, 3)) <= 0).any()):
        raise ValueError("Every row must contain valid surface input")
    return PreparedSurfaceInputs(
        values=literal_zero_mask(prepared, valid64),
        valid64=valid64,
        valid32=valid32,
        valid16=valid16,
    )


def filter_normalize(
    values: Tensor,
    *,
    tap_dimension: int,
    scale: Tensor | float,
) -> Tensor:
    if not torch.jit.is_tracing() and int(values.size(tap_dimension)) != 9:
        raise ValueError("DDF filter normalization requires nine taps")
    centered = values - values.mean(dim=tap_dimension, keepdim=True)
    standard_deviation = values.std(
        dim=tap_dimension,
        correction=1,
        keepdim=True,
    )
    return centered / (standard_deviation + FILTER_EPSILON) * scale


def numpy_filter_normalize(
    values: np.ndarray,
    *,
    tap_axis: int,
    scale: np.ndarray | float,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[tap_axis] != 9:
        raise ValueError("DDF filter normalization requires nine taps")
    centered = array - array.mean(axis=tap_axis, keepdims=True)
    standard_deviation = array.std(axis=tap_axis, ddof=1, keepdims=True)
    return centered / (standard_deviation + FILTER_EPSILON) * scale


def neighborhood_stack(inputs: Tensor) -> Tensor:
    if not torch.jit.is_tracing() and inputs.ndim != 4:
        raise ValueError("DDF input must have shape [B, C, H, W]")
    padded = F.pad(inputs, (1, 1, 1, 1), mode="constant", value=0.0)
    return torch.stack(
        (
            padded[:, :, 0:-2, 0:-2],
            padded[:, :, 0:-2, 1:-1],
            padded[:, :, 0:-2, 2:],
            padded[:, :, 1:-1, 0:-2],
            padded[:, :, 1:-1, 1:-1],
            padded[:, :, 1:-1, 2:],
            padded[:, :, 2:, 0:-2],
            padded[:, :, 2:, 1:-1],
            padded[:, :, 2:, 2:],
        ),
        dim=2,
    )


def apply_ddf_standard(
    inputs: Tensor,
    channel_filter: Tensor,
    spatial_filter: Tensor,
) -> Tensor:
    if not torch.jit.is_tracing():
        if inputs.ndim != 4:
            raise ValueError("DDF input must have shape [B, C, H, W]")
        batch, channels, height, width = inputs.shape
        if channel_filter.shape != (batch, channels, 9):
            raise ValueError("Channel filter does not align with DDF input")
        if spatial_filter.shape != (batch, 9, height, width):
            raise ValueError("Spatial filter does not align with DDF input")
    neighborhoods = neighborhood_stack(inputs)
    return (
        neighborhoods
        * channel_filter.unsqueeze(-1).unsqueeze(-1)
        * spatial_filter.unsqueeze(1)
    ).sum(dim=2)


def numpy_ddf(
    inputs: np.ndarray,
    channel_filter: np.ndarray,
    spatial_filter: np.ndarray,
) -> np.ndarray:
    features = np.asarray(inputs)
    channel = np.asarray(channel_filter)
    spatial = np.asarray(spatial_filter)
    if features.ndim != 4:
        raise ValueError("DDF input must have shape [B, C, H, W]")
    batch, channels, height, width = features.shape
    if channel.shape != (batch, channels, 9):
        raise ValueError("Channel filter does not align with DDF input")
    if spatial.shape != (batch, 9, height, width):
        raise ValueError("Spatial filter does not align with DDF input")
    output = np.zeros_like(features)
    for sample in range(batch):
        for feature_channel in range(channels):
            for output_y in range(height):
                for output_x in range(width):
                    value = 0.0
                    for kernel_y in range(3):
                        input_y = output_y + kernel_y - 1
                        if input_y < 0 or input_y >= height:
                            continue
                        for kernel_x in range(3):
                            input_x = output_x + kernel_x - 1
                            if input_x < 0 or input_x >= width:
                                continue
                            tap = kernel_y * 3 + kernel_x
                            value += (
                                features[
                                    sample,
                                    feature_channel,
                                    input_y,
                                    input_x,
                                ]
                                * channel[sample, feature_channel, tap]
                                * spatial[sample, tap, output_y, output_x]
                            )
                    output[
                        sample,
                        feature_channel,
                        output_y,
                        output_x,
                    ] = value
    return output


def neutralize_spatial_outside_support(
    spatial_filter: Tensor,
    victim_valid: Tensor,
    source_valid: Tensor,
) -> Tensor:
    if spatial_filter.ndim != 4 or int(spatial_filter.size(1)) != 9:
        raise ValueError("Spatial filters must have shape [B, 9, H, W]")
    batch, _, height, width = spatial_filter.shape
    victim = _canonical_mask(
        victim_valid,
        reference=spatial_filter,
        name="victim filter support",
    )
    source = _canonical_mask(
        source_valid,
        reference=spatial_filter,
        name="source filter support",
    )
    return torch.where(
        victim & source,
        spatial_filter,
        torch.ones((), dtype=spatial_filter.dtype, device=spatial_filter.device),
    )


def displace_spatial_filters_nonwrap(
    spatial_filter: Tensor,
    offsets: Tensor | np.ndarray | Sequence[Sequence[int]],
    *,
    direction: str,
) -> Tensor:
    if spatial_filter.ndim != 4 or int(spatial_filter.size(1)) != 9:
        raise ValueError("Spatial filters must have shape [B, 9, H, W]")
    shifts = torch.as_tensor(offsets, dtype=torch.long, device="cpu")
    if shifts.shape != (int(spatial_filter.size(0)), 2):
        raise ValueError("Spatial offsets must have shape [B, 2]")
    allowed = set(LOCKED_NONWRAP_OFFSETS)
    observed = [tuple(int(value) for value in row) for row in shifts.tolist()]
    if any(offset not in allowed for offset in observed):
        raise ValueError("Every displacement must be a locked nonzero offset")
    if direction != "source_to_destination":
        raise ValueError(
            "V2 displacement requires the locked source_to_destination rule"
        )
    output_rows: List[Tensor] = []
    height, width = int(spatial_filter.size(2)), int(spatial_filter.size(3))
    for index, (dy, dx) in enumerate(observed):
        source = spatial_filter[index]
        destination = torch.ones_like(source)
        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        source_x0 = max(0, -dx)
        source_x1 = min(width, width - dx)
        destination_y0 = source_y0 + dy
        destination_y1 = source_y1 + dy
        destination_x0 = source_x0 + dx
        destination_x1 = source_x1 + dx
        destination[
            :,
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] = source[:, source_y0:source_y1, source_x0:source_x1]
        output_rows.append(destination)
    return torch.stack(output_rows, dim=0)


class MultiplicativeDDFBlock(nn.Module):
    def __init__(self, channels: int, squeeze: int, *, mode: str) -> None:
        super().__init__()
        if mode not in DDF_MODES:
            raise ValueError(f"Unknown DDF mode: {mode}")
        self.channels = int(channels)
        self.squeeze = int(squeeze)
        self.mode = mode
        self.spatial_projection = nn.Conv2d(
            self.channels,
            9,
            kernel_size=1,
            bias=True,
        )
        self.channel_reduce = nn.Conv2d(
            self.channels,
            self.squeeze,
            kernel_size=1,
            bias=True,
        )
        self.channel_expand = nn.Conv2d(
            self.squeeze,
            self.channels * 9,
            kernel_size=1,
            bias=True,
        )
        self.channel_scale = nn.Parameter(torch.empty(self.channels, 9))
        if mode == SPATIAL_ONLY_MODE:
            for parameter in (
                *self.channel_reduce.parameters(),
                *self.channel_expand.parameters(),
                self.channel_scale,
            ):
                parameter.requires_grad_(False)
        elif mode == CHANNEL_ONLY_MODE:
            for parameter in self.spatial_projection.parameters():
                parameter.requires_grad_(False)

    def generate_filters(
        self,
        inputs: Tensor,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch, _, height, width = inputs.shape
        valid = _canonical_mask(
            valid_mask,
            reference=inputs,
            name="DDF validity mask",
        )
        masked_inputs = literal_zero_mask(inputs, valid)
        if self.mode == CHANNEL_ONLY_MODE:
            spatial = inputs.new_ones((batch, 9, height, width))
        else:
            spatial_logits = literal_zero_mask(
                self.spatial_projection(masked_inputs),
                valid,
            )
            spatial = filter_normalize(
                spatial_logits,
                tap_dimension=1,
                scale=FILTER_GAIN,
            )
            spatial = torch.where(
                valid,
                spatial,
                torch.ones((), dtype=spatial.dtype, device=spatial.device),
            )
        if self.mode == SPATIAL_ONLY_MODE:
            channel = inputs.new_ones((batch, self.channels, 9))
        else:
            pooled = validity_weighted_channel_pool(masked_inputs, valid)
            reduced = F.relu(self.channel_reduce(pooled))
            channel_logits = self.channel_expand(reduced).reshape(
                batch,
                self.channels,
                9,
            )
            channel = filter_normalize(
                channel_logits,
                tap_dimension=2,
                scale=self.channel_scale.unsqueeze(0),
            )
        return spatial, channel

    def forward_with_filters(
        self,
        inputs: Tensor,
        valid_mask: Tensor,
        *,
        override: Optional[Tuple[Tensor, Tensor]] = None,
        override_valid: Optional[Tensor] = None,
        spatial_offsets: Optional[Tensor] = None,
        spatial_offset_direction: Optional[str] = None,
        causal_mode: str = CAUSAL_CLEAN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if causal_mode not in CAUSAL_MODES:
            raise ValueError(f"Unknown causal mode: {causal_mode}")
        batch, _, height, width = inputs.shape
        valid = _canonical_mask(
            valid_mask,
            reference=inputs,
            name="DDF validity mask",
        )
        masked_inputs = literal_zero_mask(inputs, valid)
        if override is None:
            spatial, channel = self.generate_filters(masked_inputs, valid)
        else:
            spatial, channel = override
            spatial = spatial.to(device=inputs.device, dtype=inputs.dtype)
            channel = channel.to(device=inputs.device, dtype=inputs.dtype)
            source_valid = valid if override_valid is None else override_valid
            spatial = neutralize_spatial_outside_support(
                spatial,
                valid,
                source_valid,
            )
        if spatial_offsets is not None:
            if spatial_offset_direction is None:
                raise ValueError(
                    "A prospectively locked non-wrap direction is required"
                )
            spatial = displace_spatial_filters_nonwrap(
                spatial,
                spatial_offsets,
                direction=spatial_offset_direction,
            )
            spatial = neutralize_spatial_outside_support(
                spatial,
                valid,
                valid,
            )
        if causal_mode == CAUSAL_SPATIAL_NEUTRAL:
            spatial = torch.ones_like(spatial)
        elif causal_mode == CAUSAL_CHANNEL_NEUTRAL:
            channel = torch.ones_like(channel)
        output = masked_inputs + apply_ddf_standard(
            masked_inputs,
            channel,
            spatial,
        )
        return literal_zero_mask(output, valid), spatial, channel

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        output, _, _ = self.forward_with_filters(inputs, valid_mask)
        return output


class StaticMatchedBlock(nn.Module):
    def __init__(self, channels: int, intermediate: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise_in = nn.Conv2d(
            channels,
            intermediate,
            kernel_size=1,
            bias=True,
        )
        self.pointwise_out = nn.Conv2d(
            intermediate,
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        masked_inputs = literal_zero_mask(inputs, valid_mask)
        depthwise = literal_zero_mask(self.depthwise(masked_inputs), valid_mask)
        projected = literal_zero_mask(self.pointwise_in(depthwise), valid_mask)
        activated = literal_zero_mask(F.silu(projected), valid_mask)
        residual = literal_zero_mask(self.pointwise_out(activated), valid_mask)
        return literal_zero_mask(masked_inputs + residual, valid_mask)


class PairSurfaceDDFV2Sidecar(nn.Module):
    NORMALIZATION_NAMES = ("bn1_pre", "bn1_post", "bn2_pre", "bn2_post")

    def __init__(self, role: str = "ddf_full") -> None:
        super().__init__()
        if role not in ROLE_NAMES:
            raise ValueError(f"Unknown Pair-Surface DDF v2 role: {role}")
        self.role = role
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1_pre = MaskedBatchNorm2d(16)
        if role == "static_matched":
            self.block1: nn.Module = StaticMatchedBlock(16, 28)
        else:
            mode = {
                "ddf_spatial_only": SPATIAL_ONLY_MODE,
                "ddf_channel_only": CHANNEL_ONLY_MODE,
            }.get(role, FULL_MODE)
            self.block1 = MultiplicativeDDFBlock(16, 4, mode=mode)
        self.bn1_post = MaskedBatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2_pre = MaskedBatchNorm2d(32)
        if role == "static_matched":
            self.block2: nn.Module = StaticMatchedBlock(32, 39)
        else:
            mode = {
                "ddf_spatial_only": SPATIAL_ONLY_MODE,
                "ddf_channel_only": CHANNEL_ONLY_MODE,
            }.get(role, FULL_MODE)
            self.block2 = MultiplicativeDDFBlock(32, 6, mode=mode)
        self.bn2_post = MaskedBatchNorm2d(32)
        self.evidence = nn.Conv2d(32, 4, kernel_size=1, bias=True)
        self.attention = nn.Conv2d(32, 4, kernel_size=1, bias=True)

    def masked_batch_norms(self) -> Tuple[MaskedBatchNorm2d, ...]:
        return tuple(
            getattr(self, name) for name in self.NORMALIZATION_NAMES
        )

    def _dynamic_block(
        self,
        block: nn.Module,
        inputs: Tensor,
        valid_mask: Tensor,
        *,
        override: Optional[Tuple[Tensor, Tensor]],
        override_valid: Optional[Tensor],
        spatial_offsets: Optional[Tensor],
        spatial_offset_direction: Optional[str],
        causal_mode: str,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if isinstance(block, MultiplicativeDDFBlock):
            output, spatial, channel = block.forward_with_filters(
                inputs,
                valid_mask,
                override=override,
                override_valid=override_valid,
                spatial_offsets=spatial_offsets,
                spatial_offset_direction=spatial_offset_direction,
                causal_mode=causal_mode,
            )
            return output, spatial, channel
        if (
            override is not None
            or override_valid is not None
            or spatial_offsets is not None
            or spatial_offset_direction is not None
            or causal_mode != CAUSAL_CLEAN
        ):
            raise ValueError("Static role does not accept DDF causal overrides")
        return block(inputs, valid_mask), None, None

    def _validate_masks(
        self,
        inputs: Tensor,
        valid64: Tensor,
        valid32: Tensor,
        valid16: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not torch.jit.is_tracing() and inputs.shape[1:] != (3, 64, 64):
            raise ValueError("Pair-Surface DDF v2 input must be [B, 3, 64, 64]")
        mask64 = _canonical_mask(
            valid64,
            reference=inputs,
            name="valid64",
        )
        mask32 = _canonical_mask(
            valid32,
            reference=inputs,
            name="valid32",
            spatial_shape=(32, 32),
        )
        mask16 = _canonical_mask(
            valid16,
            reference=inputs,
            name="valid16",
            spatial_shape=(16, 16),
        )
        if not torch.jit.is_tracing():
            expected32 = F.max_pool2d(
                mask64.to(dtype=torch.float32),
                kernel_size=3,
                stride=2,
                padding=1,
            ) > 0
            expected16 = F.max_pool2d(
                mask32.to(dtype=torch.float32),
                kernel_size=3,
                stride=2,
                padding=1,
            ) > 0
            if not torch.equal(mask32, expected32) or not torch.equal(
                mask16,
                expected16,
            ):
                raise ValueError("valid32/valid16 do not match locked propagation")
            if bool((mask16.sum(dim=(1, 2, 3)) <= 0).any()):
                raise ValueError("Every row must have non-empty valid16")
        return mask64, mask32, mask16

    def forward_with_trace(
        self,
        inputs: Tensor,
        valid64: Tensor,
        valid32: Tensor,
        valid16: Tensor,
        *,
        filter_overrides: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
        override_valid_masks: Optional[Sequence[Tensor]] = None,
        spatial_offsets: Optional[Tensor] = None,
        spatial_offset_direction: Optional[str] = None,
        causal_mode: str = CAUSAL_CLEAN,
    ) -> Dict[str, Tensor | List[Tensor]]:
        mask64, mask32, mask16 = self._validate_masks(
            inputs,
            valid64,
            valid32,
            valid16,
        )
        if filter_overrides is not None and len(filter_overrides) != 2:
            raise ValueError("Filter overrides must contain two DDF blocks")
        if override_valid_masks is not None and len(override_valid_masks) != 2:
            raise ValueError("Override validity must contain two DDF masks")
        if spatial_offsets is not None and spatial_offsets.shape != (
            int(inputs.size(0)),
            2,
            2,
        ):
            raise ValueError("Spatial offsets must have shape [B, 2, 2]")

        masked_input = literal_zero_mask(inputs, mask64)
        bn_input_1 = literal_zero_mask(self.conv1(masked_input), mask32)
        normalized_1 = self.bn1_pre(bn_input_1, mask32)
        hidden1 = literal_zero_mask(F.silu(normalized_1), mask32)
        block1, spatial1, channel1 = self._dynamic_block(
            self.block1,
            hidden1,
            mask32,
            override=None if filter_overrides is None else filter_overrides[0],
            override_valid=(
                None
                if override_valid_masks is None
                else override_valid_masks[0]
            ),
            spatial_offsets=(
                None if spatial_offsets is None else spatial_offsets[:, 0]
            ),
            spatial_offset_direction=spatial_offset_direction,
            causal_mode=causal_mode,
        )
        bn_input_2 = literal_zero_mask(block1, mask32)
        normalized_2 = self.bn1_post(bn_input_2, mask32)
        hidden1_post = literal_zero_mask(F.silu(normalized_2), mask32)
        bn_input_3 = literal_zero_mask(self.conv2(hidden1_post), mask16)
        normalized_3 = self.bn2_pre(bn_input_3, mask16)
        hidden2 = literal_zero_mask(F.silu(normalized_3), mask16)
        block2, spatial2, channel2 = self._dynamic_block(
            self.block2,
            hidden2,
            mask16,
            override=None if filter_overrides is None else filter_overrides[1],
            override_valid=(
                None
                if override_valid_masks is None
                else override_valid_masks[1]
            ),
            spatial_offsets=(
                None if spatial_offsets is None else spatial_offsets[:, 1]
            ),
            spatial_offset_direction=spatial_offset_direction,
            causal_mode=causal_mode,
        )
        bn_input_4 = literal_zero_mask(block2, mask16)
        normalized_4 = self.bn2_post(bn_input_4, mask16)
        features = literal_zero_mask(F.silu(normalized_4), mask16)
        evidence_maps = literal_zero_mask(self.evidence(features), mask16)
        attention_logits = literal_zero_mask(self.attention(features), mask16)
        attention_maps = masked_spatial_softmax(attention_logits, mask16)
        scores = (evidence_maps * attention_maps).sum(dim=(2, 3))
        trace: Dict[str, Tensor | List[Tensor]] = {
            "scores": scores,
            "evidence_maps": evidence_maps,
            "attention_maps": attention_maps,
            "features": features,
            "normalization_inputs": [
                bn_input_1,
                bn_input_2,
                bn_input_3,
                bn_input_4,
            ],
            "normalization_masks": [mask32, mask32, mask16, mask16],
        }
        if spatial1 is not None:
            trace["spatial_filters"] = [spatial1, spatial2]
            trace["channel_filters"] = [channel1, channel2]
        return trace

    def forward(
        self,
        inputs: Tensor,
        valid64: Tensor,
        valid32: Tensor,
        valid16: Tensor,
    ) -> Tensor:
        trace = self.forward_with_trace(inputs, valid64, valid32, valid16)
        return trace["scores"]  # type: ignore[return-value]


def tensor_initialization_seed(base_seed: int, canonical_name: str) -> int:
    payload = f"{int(base_seed)}:{canonical_name}".encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big") & ((1 << 63) - 1)


def _generator(base_seed: int, canonical_name: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(tensor_initialization_seed(base_seed, canonical_name))
    return generator


def initialize_sidecar_parameters(
    model: PairSurfaceDDFV2Sidecar,
    *,
    seed: int,
) -> None:
    handled = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, nn.Conv2d):
            weight_name = f"{prefix}weight"
            nn.init.kaiming_uniform_(
                module.weight,
                a=math.sqrt(5.0),
                generator=_generator(seed, weight_name),
            )
            handled.add(weight_name)
            if module.bias is not None:
                bias_name = f"{prefix}bias"
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    module.weight
                )
                bound = 1.0 / math.sqrt(float(fan_in))
                nn.init.uniform_(
                    module.bias,
                    -bound,
                    bound,
                    generator=_generator(seed, bias_name),
                )
                handled.add(bias_name)
        elif isinstance(module, MaskedBatchNorm2d):
            weight_name = f"{prefix}weight"
            bias_name = f"{prefix}bias"
            module.reset_parameters()
            handled.update((weight_name, bias_name))
        elif isinstance(module, MultiplicativeDDFBlock):
            scale_name = f"{prefix}channel_scale"
            nn.init.normal_(
                module.channel_scale,
                mean=0.0,
                std=FILTER_GAIN,
                generator=_generator(seed, scale_name),
            )
            handled.add(scale_name)
    parameter_names = {name for name, _ in model.named_parameters()}
    if handled != parameter_names:
        raise RuntimeError(
            "Initialization did not cover every parameter: "
            f"missing={sorted(parameter_names - handled)}, "
            f"extra={sorted(handled - parameter_names)}"
        )


def role_initialization_seed(role: str, fold: int) -> int:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown Pair-Surface DDF v2 role: {role}")
    if not 0 <= int(fold) < FOLDS:
        raise ValueError(f"Fold must lie in [0, {FOLDS})")
    offset = REPEAT_OFFSET if role == "ddf_full_repeat" else 0
    return PRIMARY_SEED + offset + 100 * int(fold)


def role_order_seed(role: str, fold: int) -> int:
    return role_initialization_seed(role, fold) + 1


def initialize_sidecar(
    role: str,
    *,
    fold: int,
    device: torch.device | str = "cpu",
) -> PairSurfaceDDFV2Sidecar:
    # Module constructors perform default initialization before the canonical
    # per-tensor initializer replaces it.  Preserve the caller's global RNG so
    # those irrelevant constructor draws cannot perturb a locked experiment.
    with torch.random.fork_rng(devices=[], enabled=True):
        model = PairSurfaceDDFV2Sidecar(role)
        initialize_sidecar_parameters(
            model,
            seed=role_initialization_seed(role, fold),
        )
    return model.to(device=device, dtype=torch.float32)


def model_state_arrays(model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        f"model.{name}": value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }


def parameter_contract(model: nn.Module) -> Dict[str, object]:
    shapes = {
        name: [int(dimension) for dimension in parameter.shape]
        for name, parameter in model.named_parameters()
    }
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    all_parameters = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
    }
    return {
        "parameter_shapes": shapes,
        "parameter_shapes_sha256": json_sha256(shapes),
        "parameter_count": int(sum(all_parameters.values())),
        "trainable_parameter_count": int(sum(trainable.values())),
        "trainable_tensor_count": int(len(trainable)),
        "frozen_parameter_count": int(
            sum(all_parameters.values()) - sum(trainable.values())
        ),
    }


def learning_rate(epoch_index: int) -> float:
    epoch = int(epoch_index)
    if epoch < 0 or epoch > EPOCHS:
        raise ValueError(f"Epoch index must be in [0, {EPOCHS}]: {epoch}")
    return BASE_LR * 0.5 * (
        1.0 + math.cos(math.pi * float(epoch) / float(EPOCHS))
    )


def build_epoch_orders(
    fit_indices: np.ndarray,
    *,
    seed: int,
) -> List[np.ndarray]:
    indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Fit indices must be a non-empty unique sequence")
    if not np.array_equal(indices, np.sort(indices)):
        raise ValueError("Fit indices must be in ascending numeric order")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return [
        np.asarray(rng.permutation(indices), dtype=np.int64)
        for _ in range(EPOCHS)
    ]


def build_adamw(
    model: nn.Module,
    *,
    lr: float,
) -> torch.optim.AdamW:
    parameters = [
        parameter
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The v2 optimizer requires trainable parameters")
    return torch.optim.AdamW(
        [{"params": parameters, "weight_decay": WEIGHT_DECAY}],
        lr=float(lr),
        betas=BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


@dataclass(frozen=True)
class TaskSupervision:
    labels: Tensor
    active: Tensor
    classification_weights: Tensor
    bbox_weights: Tensor
    bbox_usable: Tensor

    def take(self, indices: Tensor | np.ndarray | Sequence[int]) -> "TaskSupervision":
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.labels.device,
        )
        return TaskSupervision(
            labels=self.labels.index_select(0, index),
            active=self.active.index_select(0, index),
            classification_weights=self.classification_weights.index_select(
                0,
                index,
            ),
            bbox_weights=self.bbox_weights.index_select(0, index),
            bbox_usable=self.bbox_usable.index_select(0, index),
        )


def build_task_supervision(
    targets: Tensor,
    bbox_usable: Tensor,
) -> TaskSupervision:
    values = torch.as_tensor(targets, dtype=torch.long)
    usable = torch.as_tensor(
        bbox_usable,
        dtype=torch.bool,
        device=values.device,
    )
    if values.ndim != 1 or usable.shape != values.shape:
        raise ValueError("Targets and bbox usability must be aligned vectors")
    allowed = values.new_tensor((0, 1, 2, 4))
    if not bool(torch.isin(values, allowed).all()):
        raise ValueError("Targets may contain only 0/1/2/4")
    rows = int(values.numel())
    labels = torch.zeros((rows, 4), dtype=torch.float32, device=values.device)
    active = torch.zeros((rows, 4), dtype=torch.bool, device=values.device)
    class_weights = torch.zeros_like(labels)
    bbox_weights = torch.zeros_like(labels)
    task_rivals: Tuple[Optional[int], ...] = (None, *RIVALS)
    for task_index, rival in enumerate(task_rivals):
        positive = values == FOCUS_CLASS
        negative = (
            values != FOCUS_CLASS
            if rival is None
            else values == int(rival)
        )
        task_active = positive | negative
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"Task {SCORE_NAMES[task_index]} lacks support")
        labels[positive, task_index] = 1.0
        active[:, task_index] = task_active
        class_weights[positive, task_index] = rows / (2.0 * positive_count)
        class_weights[negative, task_index] = rows / (2.0 * negative_count)

        usable_positive = positive & usable
        usable_negative = negative & usable
        usable_positive_count = int(usable_positive.sum())
        usable_negative_count = int(usable_negative.sum())
        if usable_positive_count == 0 or usable_negative_count == 0:
            raise ValueError(
                f"Task {SCORE_NAMES[task_index]} lacks usable bbox support"
            )
        bbox_weights[usable_positive, task_index] = rows / (
            2.0 * usable_positive_count
        )
        bbox_weights[usable_negative, task_index] = rows / (
            2.0 * usable_negative_count
        )
    return TaskSupervision(
        labels=labels,
        active=active,
        classification_weights=class_weights,
        bbox_weights=bbox_weights,
        bbox_usable=usable,
    )


def rasterize_model_boxes_16(model_boxes: Tensor) -> Tensor:
    boxes = torch.as_tensor(model_boxes, dtype=torch.float64, device="cpu")
    if boxes.ndim != 2 or boxes.size(1) != 4:
        raise ValueError("Model boxes must have shape [B, 4] as cx/cy/w/h")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("Model boxes must be finite")
    output = torch.zeros((int(boxes.size(0)), 16, 16), dtype=torch.bool)
    for index, row in enumerate(boxes.tolist()):
        center_x, center_y, width, height = row
        left = min(max(center_x - width / 2.0, 0.0), 1.0)
        right = min(max(center_x + width / 2.0, 0.0), 1.0)
        top = min(max(center_y - height / 2.0, 0.0), 1.0)
        bottom = min(max(center_y + height / 2.0, 0.0), 1.0)
        x0 = min(max(int(math.floor(16.0 * left)), 0), 15)
        y0 = min(max(int(math.floor(16.0 * top)), 0), 15)
        x1 = min(max(int(math.ceil(16.0 * right)), x0 + 1), 16)
        y1 = min(max(int(math.ceil(16.0 * bottom)), y0 + 1), 16)
        output[index, y0:y1, x0:x1] = True
    return output


def pair_surface_loss(
    scores: Tensor,
    attention_maps: Tensor,
    supervision: TaskSupervision,
    bbox_valid_masks: Tensor,
    valid16: Tensor,
) -> Dict[str, Tensor]:
    if scores.ndim != 2 or int(scores.size(1)) != 4:
        raise ValueError("Scores must have shape [B, 4]")
    batch = int(scores.size(0))
    if (
        supervision.labels.shape != scores.shape
        or supervision.active.shape != scores.shape
        or supervision.classification_weights.shape != scores.shape
        or supervision.bbox_weights.shape != scores.shape
        or supervision.bbox_usable.shape != (batch,)
    ):
        raise ValueError("Task supervision does not align with scores")
    if attention_maps.shape != (batch, 4, 16, 16):
        raise ValueError("Attention maps must have shape [B, 4, 16, 16]")
    bbox = torch.as_tensor(
        bbox_valid_masks,
        dtype=torch.bool,
        device=attention_maps.device,
    )
    if bbox.shape != (batch, 16, 16):
        raise ValueError("BBox-valid masks must have shape [B, 16, 16]")
    valid = _canonical_mask(
        valid16,
        reference=attention_maps,
        name="loss valid16",
    )
    if bool((bbox & ~valid[:, 0]).any()):
        raise ValueError("BBox supervision must already be intersected with valid16")
    observed_usable = bbox.flatten(start_dim=1).any(dim=1)
    if not torch.equal(
        observed_usable,
        supervision.bbox_usable.to(device=observed_usable.device),
    ):
        raise ValueError("BBox usability changed after fit weights were frozen")
    if bool(
        (
            supervision.classification_weights[~supervision.active]
            != 0
        ).any()
    ):
        raise ValueError("Inactive classification weights must be exact zero")
    per_row_bce = F.binary_cross_entropy_with_logits(
        scores,
        supervision.labels.to(device=scores.device, dtype=scores.dtype),
        reduction="none",
    )
    classification_by_task = (
        per_row_bce
        * supervision.classification_weights.to(
            device=scores.device,
            dtype=scores.dtype,
        )
    ).sum(dim=0) / float(BATCH_SIZE)
    bbox_mass = (
        attention_maps
        * bbox.to(dtype=attention_maps.dtype).unsqueeze(1)
    ).sum(dim=(2, 3))
    per_row_bbox = -torch.log(
        bbox_mass.clamp_min(BBOX_ATTENTION_CLAMP_MIN)
    )
    bbox_by_task = (
        per_row_bbox
        * supervision.bbox_weights.to(
            device=scores.device,
            dtype=scores.dtype,
        )
    ).sum(dim=0) / float(BATCH_SIZE)
    classification_pair_mean = classification_by_task[1:].mean()
    bbox_pair_mean = bbox_by_task[1:].mean()
    classification_total = (
        classification_by_task[0]
        + PAIR_LOSS_WEIGHT * classification_pair_mean
    )
    bbox_total = (
        bbox_by_task[0] + PAIR_LOSS_WEIGHT * bbox_pair_mean
    ) / 1.5
    total = classification_total + BBOX_ATTENTION_WEIGHT * bbox_total
    return {
        "total": total,
        "classification_total": classification_total,
        "classification_by_task": classification_by_task,
        "classification_pair_mean": classification_pair_mean,
        "bbox_total": bbox_total,
        "bbox_by_task": bbox_by_task,
        "bbox_pair_mean": bbox_pair_mean,
        "bbox_mass": bbox_mass,
    }


def _streaming_array_digest(
    dtype: np.dtype,
    shape: Sequence[int],
) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(str(np.dtype(dtype)).encode("ascii"))
    digest.update(np.asarray(tuple(shape), dtype=np.dtype("<i8")).tobytes())
    return digest


def recalibrate_masked_batch_norms(
    model: PairSurfaceDDFV2Sidecar,
    prepared: PreparedSurfaceInputs,
    *,
    sample_indices: np.ndarray,
    batch_size: int = BATCH_SIZE,
    expected_training_calls: int = EPOCHS * 8,
) -> List[Dict[str, object]]:
    """Sequentially freeze fit-only FP64 population statistics.

    ``prepared`` must already be ordered by ascending ``sample_indices``.  The
    method performs four complete no-grad passes, freezing one site before the
    next site is observed.  It leaves the model in evaluation mode.
    """

    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    rows = int(prepared.values.size(0))
    if indices.shape != (rows,) or not np.array_equal(indices, np.sort(indices)):
        raise ValueError("Recalibration rows must be unique ascending sample indices")
    if np.unique(indices).size != indices.size:
        raise ValueError("Recalibration sample indices must be unique")
    if batch_size <= 0:
        raise ValueError("Recalibration batch size must be positive")
    if not (
        prepared.valid64.size(0)
        == prepared.valid32.size(0)
        == prepared.valid16.size(0)
        == rows
    ):
        raise ValueError("Prepared recalibration tensors are not row-aligned")

    counters_before = [
        int(module.num_batches_tracked) for module in model.masked_batch_norms()
    ]
    if counters_before != [int(expected_training_calls)] * 4:
        raise RuntimeError(
            "Masked-BN optimizer-forward counters are not locked: "
            f"observed={counters_before}, expected={expected_training_calls}"
        )
    parameters_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    records: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for site_index, (site_name, module) in enumerate(
            zip(model.NORMALIZATION_NAMES, model.masked_batch_norms())
        ):
            accumulator = MaskedPopulationAccumulator(module.num_features)
            value_digest: Optional["hashlib._Hash"] = None
            mask_digest: Optional["hashlib._Hash"] = None
            for start in range(0, rows, batch_size):
                stop = min(start + batch_size, rows)
                positions = torch.arange(
                    start,
                    stop,
                    dtype=torch.long,
                    device=prepared.values.device,
                )
                batch = prepared.take(positions)
                trace = model.forward_with_trace(
                    batch.values,
                    batch.valid64,
                    batch.valid32,
                    batch.valid16,
                )
                normalization_inputs = trace["normalization_inputs"]
                normalization_masks = trace["normalization_masks"]
                assert isinstance(normalization_inputs, list)
                assert isinstance(normalization_masks, list)
                values = normalization_inputs[site_index]
                mask = normalization_masks[site_index]
                accumulator.update(values, mask)
                values_numpy = np.ascontiguousarray(
                    values.detach().cpu().numpy()
                )
                mask_numpy = np.ascontiguousarray(mask.detach().cpu().numpy())
                if value_digest is None:
                    value_digest = _streaming_array_digest(
                        values_numpy.dtype,
                        (rows, *values_numpy.shape[1:]),
                    )
                    mask_digest = _streaming_array_digest(
                        mask_numpy.dtype,
                        (rows, *mask_numpy.shape[1:]),
                    )
                value_digest.update(values_numpy.tobytes())
                assert mask_digest is not None
                mask_digest.update(mask_numpy.tobytes())
            mean, variance = accumulator.finalize()
            module.running_mean.copy_(
                mean.to(device=module.running_mean.device, dtype=module.running_mean.dtype)
            )
            module.running_var.copy_(
                variance.to(device=module.running_var.device, dtype=module.running_var.dtype)
            )
            assert value_digest is not None and mask_digest is not None
            records.append(
                {
                    "site_index": site_index,
                    "site_name": site_name,
                    "sample_indices_sha256": array_sha256(indices),
                    "pre_normalization_sha256": value_digest.hexdigest(),
                    "mask_sha256": mask_digest.hexdigest(),
                    "valid_count": accumulator.count,
                    "sum_sha256": array_sha256(accumulator.sum.numpy()),
                    "sumsq_sha256": array_sha256(accumulator.sumsq.numpy()),
                    "running_mean_sha256": array_sha256(
                        module.running_mean.detach().cpu().numpy()
                    ),
                    "running_var_sha256": array_sha256(
                        module.running_var.detach().cpu().numpy()
                    ),
                }
            )
    counters_after = [
        int(module.num_batches_tracked) for module in model.masked_batch_norms()
    ]
    if counters_after != counters_before:
        raise RuntimeError("Recalibration changed optimizer-forward counters")
    if any(
        not torch.equal(parameters_before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("Recalibration changed a frozen trainable tensor")
    return records
