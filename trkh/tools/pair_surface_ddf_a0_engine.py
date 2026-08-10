from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
PRIMARY_SEED = 20260725
REPEAT_OFFSET = 100000
CALIBRATION_FOLDS = (2, 3, 4, 2, 3)
BASE_LR = 0.003
WEIGHT_DECAY = 0.0001
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
BBOX_ATTENTION_WEIGHT = 0.05
PAIR_LOSS_WEIGHT = 0.5
BBOX_ATTENTION_CLAMP_MIN = 1e-8
FILTER_EPSILON = 1e-10
FILTER_GAIN = math.sqrt(2.0) / 3.0
INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.229, 0.224, 0.225)

FULL_MODE = "full"
SPATIAL_ONLY_MODE = "spatial_only"
CHANNEL_ONLY_MODE = "channel_only"
DDF_MODES = (FULL_MODE, SPATIAL_ONLY_MODE, CHANNEL_ONLY_MODE)

CAUSAL_CLEAN = "clean"
CAUSAL_SPATIAL_NEUTRAL = "spatial_fixed_neutral"
CAUSAL_CHANNEL_NEUTRAL = "channel_fixed_neutral"
CAUSAL_MODES = (
    CAUSAL_CLEAN,
    CAUSAL_SPATIAL_NEUTRAL,
    CAUSAL_CHANNEL_NEUTRAL,
)


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


def state_arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(str(name).encode("utf-8"))
        digest.update(bytes.fromhex(array_sha256(np.asarray(values[name]))))
    return digest.hexdigest()


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
    normalized = centered / (standard_deviation + FILTER_EPSILON)
    return normalized * scale


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
    standard_deviation = array.std(
        axis=tap_axis,
        ddof=1,
        keepdims=True,
    )
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

    def generate_filters(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        batch, _, height, width = inputs.shape
        if self.mode == CHANNEL_ONLY_MODE:
            spatial = inputs.new_ones((batch, 9, height, width))
        else:
            spatial_logits = self.spatial_projection(inputs)
            spatial = filter_normalize(
                spatial_logits,
                tap_dimension=1,
                scale=FILTER_GAIN,
            )
        if self.mode == SPATIAL_ONLY_MODE:
            channel = inputs.new_ones((batch, self.channels, 9))
        else:
            pooled = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
            channel_logits = self.channel_expand(
                F.relu(self.channel_reduce(pooled))
            ).reshape(batch, self.channels, 9)
            channel = filter_normalize(
                channel_logits,
                tap_dimension=2,
                scale=self.channel_scale.unsqueeze(0),
            )
        return spatial, channel

    def forward_with_filters(
        self,
        inputs: Tensor,
        *,
        override: Optional[Tuple[Tensor, Tensor]] = None,
        spatial_rolls: Optional[Tensor] = None,
        causal_mode: str = CAUSAL_CLEAN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if causal_mode not in CAUSAL_MODES:
            raise ValueError(f"Unknown causal mode: {causal_mode}")
        if override is None:
            spatial, channel = self.generate_filters(inputs)
        else:
            spatial, channel = override
            spatial = spatial.to(device=inputs.device, dtype=inputs.dtype)
            channel = channel.to(device=inputs.device, dtype=inputs.dtype)
        if spatial_rolls is not None:
            spatial = roll_spatial_filters(spatial, spatial_rolls)
        if causal_mode == CAUSAL_SPATIAL_NEUTRAL:
            spatial = torch.ones_like(spatial)
        elif causal_mode == CAUSAL_CHANNEL_NEUTRAL:
            channel = torch.ones_like(channel)
        output = inputs + apply_ddf_standard(inputs, channel, spatial)
        return output, spatial, channel

    def forward(self, inputs: Tensor) -> Tensor:
        output, _, _ = self.forward_with_filters(inputs)
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

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.pointwise_out(
            F.silu(self.pointwise_in(self.depthwise(inputs)))
        )
        return inputs + residual


def roll_spatial_filters(filters: Tensor, offsets: Tensor) -> Tensor:
    if filters.ndim != 4 or int(filters.size(1)) != 9:
        raise ValueError("Spatial filters must have shape [B, 9, H, W]")
    shifts = torch.as_tensor(offsets, dtype=torch.long, device="cpu")
    if shifts.shape != (int(filters.size(0)), 2):
        raise ValueError("Spatial roll offsets must have shape [B, 2]")
    rows = [
        torch.roll(
            filters[index],
            shifts=(int(shifts[index, 0]), int(shifts[index, 1])),
            dims=(-2, -1),
        )
        for index in range(int(filters.size(0)))
    ]
    return torch.stack(rows, dim=0)


class PairSurfaceDDFSidecar(nn.Module):
    def __init__(self, role: str = "ddf_full") -> None:
        super().__init__()
        if role not in ROLE_NAMES:
            raise ValueError(f"Unknown Pair-Surface DDF role: {role}")
        self.role = role
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1_pre = nn.BatchNorm2d(16)
        if role == "static_matched":
            self.block1: nn.Module = StaticMatchedBlock(16, 28)
        else:
            mode = {
                "ddf_spatial_only": SPATIAL_ONLY_MODE,
                "ddf_channel_only": CHANNEL_ONLY_MODE,
            }.get(role, FULL_MODE)
            self.block1 = MultiplicativeDDFBlock(16, 4, mode=mode)
        self.bn1_post = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2_pre = nn.BatchNorm2d(32)
        if role == "static_matched":
            self.block2: nn.Module = StaticMatchedBlock(32, 39)
        else:
            mode = {
                "ddf_spatial_only": SPATIAL_ONLY_MODE,
                "ddf_channel_only": CHANNEL_ONLY_MODE,
            }.get(role, FULL_MODE)
            self.block2 = MultiplicativeDDFBlock(32, 6, mode=mode)
        self.bn2_post = nn.BatchNorm2d(32)
        self.evidence = nn.Conv2d(32, 4, kernel_size=1, bias=True)
        self.attention = nn.Conv2d(32, 4, kernel_size=1, bias=True)

    def _dynamic_block(
        self,
        block: nn.Module,
        inputs: Tensor,
        *,
        override: Optional[Tuple[Tensor, Tensor]],
        roll: Optional[Tensor],
        causal_mode: str,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if isinstance(block, MultiplicativeDDFBlock):
            output, spatial, channel = block.forward_with_filters(
                inputs,
                override=override,
                spatial_rolls=roll,
                causal_mode=causal_mode,
            )
            return output, spatial, channel
        if override is not None or roll is not None or causal_mode != CAUSAL_CLEAN:
            raise ValueError("Static role does not accept DDF causal overrides")
        return block(inputs), None, None

    def forward_with_trace(
        self,
        inputs: Tensor,
        *,
        filter_overrides: Optional[
            Sequence[Tuple[Tensor, Tensor]]
        ] = None,
        spatial_rolls: Optional[Tensor] = None,
        causal_mode: str = CAUSAL_CLEAN,
    ) -> Dict[str, Tensor | List[Tensor]]:
        if not torch.jit.is_tracing() and inputs.shape[1:] != (3, 64, 64):
            raise ValueError("Pair-Surface DDF input must be [B, 3, 64, 64]")
        if filter_overrides is not None and len(filter_overrides) != 2:
            raise ValueError("Filter overrides must contain two DDF blocks")
        if spatial_rolls is not None and spatial_rolls.shape != (
            int(inputs.size(0)),
            2,
            2,
        ):
            raise ValueError("Spatial rolls must have shape [B, 2, 2]")

        hidden1 = F.silu(self.bn1_pre(self.conv1(inputs)))
        block1, spatial1, channel1 = self._dynamic_block(
            self.block1,
            hidden1,
            override=None if filter_overrides is None else filter_overrides[0],
            roll=None if spatial_rolls is None else spatial_rolls[:, 0],
            causal_mode=causal_mode,
        )
        hidden1_post = F.silu(self.bn1_post(block1))
        hidden2 = F.silu(self.bn2_pre(self.conv2(hidden1_post)))
        block2, spatial2, channel2 = self._dynamic_block(
            self.block2,
            hidden2,
            override=None if filter_overrides is None else filter_overrides[1],
            roll=None if spatial_rolls is None else spatial_rolls[:, 1],
            causal_mode=causal_mode,
        )
        features = F.silu(self.bn2_post(block2))
        evidence_maps = self.evidence(features)
        attention_logits = self.attention(features)
        attention_maps = torch.softmax(
            attention_logits.flatten(start_dim=2),
            dim=2,
        ).reshape_as(attention_logits)
        scores = (evidence_maps * attention_maps).sum(dim=(2, 3))
        trace: Dict[str, Tensor | List[Tensor]] = {
            "scores": scores,
            "evidence_maps": evidence_maps,
            "attention_maps": attention_maps,
            "features": features,
        }
        if spatial1 is not None:
            trace["spatial_filters"] = [spatial1, spatial2]
            trace["channel_filters"] = [channel1, channel2]
        return trace

    def forward(self, inputs: Tensor) -> Tensor:
        trace = self.forward_with_trace(inputs)
        return trace["scores"]  # type: ignore[return-value]


def _tensor_seed(base_seed: int, canonical_name: str) -> int:
    payload = f"{int(base_seed)}:{canonical_name}".encode("utf-8")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big") & ((1 << 63) - 1)


def _generator(base_seed: int, canonical_name: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_tensor_seed(base_seed, canonical_name))
    return generator


def initialize_sidecar_parameters(
    model: PairSurfaceDDFSidecar,
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
        elif isinstance(module, nn.BatchNorm2d):
            weight_name = f"{prefix}weight"
            bias_name = f"{prefix}bias"
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            module.running_mean.zero_()
            module.running_var.fill_(1.0)
            module.num_batches_tracked.zero_()
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
        raise ValueError(f"Unknown Pair-Surface DDF role: {role}")
    if not 0 <= int(fold) < FOLDS:
        raise ValueError(f"Fold must lie in [0, {FOLDS})")
    offset = REPEAT_OFFSET if role == "ddf_full_repeat" else 0
    return PRIMARY_SEED + offset + 100 * int(fold)


def initialize_sidecar(
    role: str,
    *,
    fold: int,
    device: torch.device | str = "cpu",
) -> PairSurfaceDDFSidecar:
    model = PairSurfaceDDFSidecar(role)
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


def prepare_cached_model_inputs(model_srgb_uint8: Tensor) -> Tensor:
    if model_srgb_uint8.ndim != 4 or int(model_srgb_uint8.size(1)) != 3:
        raise ValueError("Cached sRGB tensor must have shape [B, 3, H, W]")
    if model_srgb_uint8.dtype != torch.uint8:
        raise ValueError("Cached sRGB tensor must be uint8")
    values = model_srgb_uint8.to(dtype=torch.float32) / 255.0
    mean = values.new_tensor(INPUT_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(INPUT_STD).view(1, 3, 1, 1)
    normalized = (values - mean) / std
    return F.interpolate(
        normalized,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def task_targets_and_weights(targets: Tensor) -> Tuple[Tensor, Tensor]:
    values = torch.as_tensor(targets, dtype=torch.long)
    if values.ndim != 1 or not bool(
        torch.isin(values, values.new_tensor((0, 1, 2, 4))).all()
    ):
        raise ValueError("Targets must be a vector containing only 0/1/2/4")
    rows = int(values.numel())
    labels = values.new_zeros((rows, 4), dtype=torch.float32)
    weights = labels.clone()
    task_rivals: Tuple[Optional[int], ...] = (None, *RIVALS)
    for task_index, rival in enumerate(task_rivals):
        positive = values == FOCUS_CLASS
        negative = (
            values != FOCUS_CLASS
            if rival is None
            else values == int(rival)
        )
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"Task {task_index} lacks binary support")
        labels[positive, task_index] = 1.0
        weights[positive, task_index] = rows / (2.0 * positive_count)
        weights[negative, task_index] = rows / (2.0 * negative_count)
    return labels, weights


def pair_surface_loss(
    scores: Tensor,
    attention_maps: Tensor,
    labels: Tensor,
    row_weights: Tensor,
    bbox_masks: Tensor,
) -> Dict[str, Tensor]:
    if scores.ndim != 2 or int(scores.size(1)) != 4:
        raise ValueError("Scores must have shape [B, 4]")
    if labels.shape != scores.shape or row_weights.shape != scores.shape:
        raise ValueError("Labels and row weights must align with scores")
    if attention_maps.shape != (
        int(scores.size(0)),
        4,
        16,
        16,
    ):
        raise ValueError("Attention maps must have shape [B, 4, 16, 16]")
    if bbox_masks.shape != (int(scores.size(0)), 16, 16):
        raise ValueError("BBox masks must have shape [B, 16, 16]")
    per_row = F.binary_cross_entropy_with_logits(
        scores,
        labels.to(dtype=scores.dtype),
        reduction="none",
    )
    task_losses = (
        per_row * row_weights.to(dtype=scores.dtype)
    ).sum(dim=0) / float(scores.size(0))
    bbox_mass = (
        attention_maps
        * bbox_masks.to(dtype=attention_maps.dtype).unsqueeze(1)
    ).sum(dim=(2, 3))
    bbox_loss = -torch.log(
        bbox_mass.clamp_min(BBOX_ATTENTION_CLAMP_MIN)
    ).mean()
    total = (
        task_losses[0]
        + PAIR_LOSS_WEIGHT * task_losses[1:].mean()
        + BBOX_ATTENTION_WEIGHT * bbox_loss
    )
    return {
        "total": total,
        "union": task_losses[0],
        "pair_mean": task_losses[1:].mean(),
        "bbox_attention": bbox_loss,
        "bbox_mass_mean": bbox_mass.mean(),
    }


def learning_rate(epoch_index: int) -> float:
    epoch = int(epoch_index)
    if epoch < 0 or epoch >= EPOCHS:
        raise ValueError(f"Epoch index must be in [0, {EPOCHS}): {epoch}")
    return BASE_LR * 0.5 * (
        1.0 + math.cos(math.pi * float(epoch) / float(EPOCHS - 1))
    )


def build_epoch_orders(
    fit_indices: np.ndarray,
    *,
    seed: int,
) -> List[np.ndarray]:
    indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Fit indices must be a non-empty unique sequence")
    rng = np.random.default_rng(int(seed))
    return [
        indices[rng.permutation(indices.size)].astype(np.int64, copy=False)
        for _ in range(EPOCHS)
    ]
