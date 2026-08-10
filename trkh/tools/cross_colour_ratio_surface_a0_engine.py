from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_IMAGE_SIZE = 256
VIEW_SIZE = 96
INPUT_CHANNELS = 6
CLASS_ORDER = (0, 1, 2, 4)
PAIR_RIVALS = (0, 2, 4)
FOCUS_CLASS = 1

PRIMARY_SEED = 20260725
REPEAT_SEED_OFFSET = 100000
DEPHASE_SEED = 20265725
EPOCHS = 20
BATCH_SIZE = 64
BASE_LR = 1e-3
WARMUP_EPOCHS = 2
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

GAUSSIAN_SIGMA = 1.0
GAUSSIAN_RADIUS = 3
LOG_EPSILON = 1.0 / 255.0
SATURATION_LOW = 1.0 / 255.0
SATURATION_HIGH = 254.0 / 255.0
SIGNED_RESPONSE_LIMIT = 8.0

INPUT_MEAN = (0.485, 0.456, 0.406)
INPUT_STD = (0.229, 0.224, 0.225)

COLOUR_RATIO_MODE = "colour_ratio"
CROSS_COLOUR_RATIO_MODE = "cross_colour_ratio"
DESCRIPTOR_MODES = (COLOUR_RATIO_MODE, CROSS_COLOUR_RATIO_MODE)
DESCRIPTOR_CHANNEL_NAMES = (
    "rg_x",
    "rg_y",
    "rb_x",
    "rb_y",
    "gb_x",
    "gb_y",
)

ROLE_NAMES = (
    "colour_ratio_control",
    "cross_colour_ratio_candidate",
    "cross_colour_ratio_seed_repeat",
    "cross_colour_ratio_spatial_dephased_control",
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


def area_resize_weights(input_size: int, output_size: int) -> np.ndarray:
    input_length = int(input_size)
    output_length = int(output_size)
    if input_length <= 0 or output_length <= 0 or output_length > input_length:
        raise ValueError("Area resize requires 0 < output_size <= input_size")
    weights = np.zeros((output_length, input_length), dtype=np.float64)
    for output_index in range(output_length):
        start = int(math.floor(output_index * input_length / output_length))
        end = int(
            math.ceil((output_index + 1) * input_length / output_length)
        )
        if end <= start:
            raise RuntimeError("Area resize produced an empty pooling bin")
        weights[output_index, start:end] = 1.0 / float(end - start)
    return weights


class FixedAreaResize2d(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        weights = area_resize_weights(self.input_size, self.output_size)
        self.register_buffer(
            "weights",
            torch.from_numpy(weights).to(dtype=torch.float32),
            persistent=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weights = self.weights
        horizontal = torch.matmul(inputs, weights.transpose(0, 1))
        return torch.matmul(weights, horizontal)


def gaussian_derivative_kernels() -> np.ndarray:
    coordinates = np.arange(
        -GAUSSIAN_RADIUS,
        GAUSSIAN_RADIUS + 1,
        dtype=np.float64,
    )
    gaussian = np.exp(
        -(coordinates**2) / (2.0 * GAUSSIAN_SIGMA**2)
    )
    gaussian /= gaussian.sum()
    correlation_derivative = (
        coordinates * gaussian / (GAUSSIAN_SIGMA**2)
    )
    horizontal = np.outer(gaussian, correlation_derivative)
    vertical = np.outer(correlation_derivative, gaussian)
    kernels = []
    for _ in range(3):
        kernels.extend((horizontal, vertical))
    return np.asarray(kernels, dtype=np.float64)[:, None, :, :]


def srgb_to_linear(srgb: Tensor) -> Tensor:
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(linear_rgb: Tensor) -> Tensor:
    return torch.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055,
    )


def _log_planes(linear_rgb: Tensor, mode: str) -> Tensor:
    logs = torch.log(torch.clamp(linear_rgb, min=LOG_EPSILON))
    if mode == COLOUR_RATIO_MODE:
        return logs
    if mode != CROSS_COLOUR_RATIO_MODE:
        raise ValueError(f"Unknown descriptor mode: {mode}")
    log_r, log_g, log_b = logs.unbind(dim=1)
    return torch.stack(
        (log_r - log_g, log_r - log_b, log_g - log_b),
        dim=1,
    )


def signed_colour_derivatives(
    linear_rgb: Tensor,
    output_reliability: Tensor,
    *,
    mode: str,
    kernels: Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    if mode not in DESCRIPTOR_MODES:
        raise ValueError(f"Unknown descriptor mode: {mode}")
    if not torch.jit.is_tracing():
        if linear_rgb.ndim != 4 or int(linear_rgb.size(1)) != 3:
            raise ValueError("Linear RGB must have shape [B, 3, H, W]")
        if output_reliability.shape != (
            int(linear_rgb.size(0)),
            1,
            int(linear_rgb.size(2)),
            int(linear_rgb.size(3)),
        ):
            raise ValueError("Output reliability does not align with linear RGB")
    if kernels is None:
        kernels = torch.from_numpy(gaussian_derivative_kernels())
    kernels = kernels.to(device=linear_rgb.device, dtype=linear_rgb.dtype)
    raw = F.conv2d(
        _log_planes(linear_rgb, mode),
        kernels,
        padding=GAUSSIAN_RADIUS,
        groups=3,
    )
    finite = torch.isfinite(raw)
    clipped = (~finite) | (raw.abs() > SIGNED_RESPONSE_LIMIT)
    bounded = torch.nan_to_num(
        raw,
        nan=0.0,
        posinf=SIGNED_RESPONSE_LIMIT,
        neginf=-SIGNED_RESPONSE_LIMIT,
    ).clamp(-SIGNED_RESPONSE_LIMIT, SIGNED_RESPONSE_LIMIT)
    reliability = output_reliability.to(dtype=bounded.dtype)
    descriptors = bounded * reliability
    denominator = (
        reliability.sum(dim=(1, 2, 3)) * float(INPUT_CHANNELS)
    ).clamp_min(1.0)
    clip_fraction = (
        clipped.to(dtype=bounded.dtype)
        * reliability.expand_as(bounded)
    ).sum(dim=(1, 2, 3)) / denominator
    return descriptors, clip_fraction


class ColourDerivativeDescriptorExtractor(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        input_mean: Sequence[float] = INPUT_MEAN,
        input_std: Sequence[float] = INPUT_STD,
    ) -> None:
        super().__init__()
        if mode not in DESCRIPTOR_MODES:
            raise ValueError(f"Unknown descriptor mode: {mode}")
        if len(input_mean) != 3 or len(input_std) != 3:
            raise ValueError("Input mean/std must each contain three values")
        self.mode = mode
        self.resize = FixedAreaResize2d(MODEL_IMAGE_SIZE, VIEW_SIZE)
        self.register_buffer(
            "input_mean",
            torch.tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "support_kernel",
            torch.ones(
                (1, 1, 2 * GAUSSIAN_RADIUS + 1, 2 * GAUSSIAN_RADIUS + 1),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "derivative_kernels",
            torch.from_numpy(gaussian_derivative_kernels()).to(
                dtype=torch.float32
            ),
        )

    def forward(
        self,
        model_input: Tensor,
        image_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        srgb = (
            model_input * self.input_std + self.input_mean
        ).clamp(0.0, 1.0)
        view_srgb = self.resize(srgb)
        view_valid_fraction = self.resize(
            image_valid_mask.to(dtype=srgb.dtype)
        )
        view_valid = view_valid_fraction >= (1.0 - 1e-6)
        unsaturated = (
            (view_srgb > SATURATION_LOW)
            & (view_srgb < SATURATION_HIGH)
        ).all(dim=1, keepdim=True)
        base_reliability = (view_valid & unsaturated).to(dtype=srgb.dtype)
        support_count = F.conv2d(
            base_reliability,
            self.support_kernel.to(dtype=srgb.dtype),
            padding=GAUSSIAN_RADIUS,
        )
        support_area = float((2 * GAUSSIAN_RADIUS + 1) ** 2)
        reliability = (support_count >= (support_area - 1e-6)).to(
            dtype=srgb.dtype
        )
        descriptors, clip_fraction = signed_colour_derivatives(
            srgb_to_linear(view_srgb),
            reliability,
            mode=self.mode,
            kernels=self.derivative_kernels,
        )
        return descriptors, reliability, clip_fraction


class CrossColourRatioEvidenceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS,
            24,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(6, 24)
        self.depthwise = nn.Conv2d(
            24,
            24,
            kernel_size=3,
            padding=1,
            groups=24,
            bias=False,
        )
        self.pointwise = nn.Conv2d(24, 48, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 48)
        self.evidence = nn.Conv2d(48, 4, kernel_size=1, bias=True)

    def forward(
        self,
        descriptors: Tensor,
        reliability: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = F.avg_pool2d(
            F.gelu(self.norm1(self.conv1(descriptors))),
            kernel_size=2,
            stride=2,
        )
        hidden = self.depthwise(hidden)
        hidden = self.pointwise(hidden)
        hidden = F.avg_pool2d(
            F.gelu(self.norm2(hidden)),
            kernel_size=2,
            stride=2,
        )
        evidence_maps = self.evidence(hidden)
        evidence_reliability = F.avg_pool2d(
            F.avg_pool2d(reliability, kernel_size=2, stride=2),
            kernel_size=2,
            stride=2,
        )
        denominator = evidence_reliability.sum(dim=(2, 3)).clamp_min(1e-12)
        logits = (
            evidence_maps * evidence_reliability
        ).sum(dim=(2, 3)) / denominator
        pair_maps = torch.stack(
            (
                evidence_maps[:, 1] - evidence_maps[:, 0],
                evidence_maps[:, 1] - evidence_maps[:, 2],
                evidence_maps[:, 1] - evidence_maps[:, 3],
            ),
            dim=1,
        )
        return logits, evidence_maps, pair_maps, evidence_reliability


class CrossColourRatioSurfacePipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = ColourDerivativeDescriptorExtractor(
            mode=CROSS_COLOUR_RATIO_MODE
        )
        self.head = CrossColourRatioEvidenceHead()

    def forward(
        self,
        model_input: Tensor,
        image_valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        descriptors, reliability, _ = self.extractor(
            model_input,
            image_valid_mask,
        )
        logits, _, pair_maps, _ = self.head(descriptors, reliability)
        return torch.softmax(logits, dim=1), pair_maps


def direct_cross_colour_log_ratio(
    first: Tensor,
    second: Tensor,
) -> Tensor:
    first_safe = torch.clamp(first, min=LOG_EPSILON)
    second_safe = torch.clamp(second, min=LOG_EPSILON)
    r1, g1, b1 = first_safe.unbind(dim=-1)
    r2, g2, b2 = second_safe.unbind(dim=-1)
    return torch.stack(
        (
            torch.log((r1 * g2) / (r2 * g1)),
            torch.log((r1 * b2) / (r2 * b1)),
            torch.log((g1 * b2) / (g2 * b1)),
        ),
        dim=-1,
    )


def log_difference_cross_colour_ratio(
    first: Tensor,
    second: Tensor,
) -> Tensor:
    first_logs = torch.log(torch.clamp(first, min=LOG_EPSILON))
    second_logs = torch.log(torch.clamp(second, min=LOG_EPSILON))
    first_r, first_g, first_b = first_logs.unbind(dim=-1)
    second_r, second_g, second_b = second_logs.unbind(dim=-1)
    return torch.stack(
        (
            (first_r - first_g) - (second_r - second_g),
            (first_r - first_b) - (second_r - second_b),
            (first_g - first_b) - (second_g - second_b),
        ),
        dim=-1,
    )


def build_dephase_offsets(
    sample_indices: np.ndarray,
    *,
    seed: int = DEPHASE_SEED,
    view_size: int = VIEW_SIZE,
) -> np.ndarray:
    samples = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if samples.size == 0 or np.unique(samples).size != samples.size:
        raise ValueError("Sample indices must be a non-empty unique sequence")
    offsets = np.empty((samples.size, INPUT_CHANNELS, 2), dtype=np.int64)
    for position, sample_index in enumerate(samples.tolist()):
        for channel in range(INPUT_CHANNELS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(seed), int(sample_index), int(channel)]
                )
            )
            offsets[position, channel, 0] = int(rng.integers(1, view_size))
            offsets[position, channel, 1] = int(rng.integers(1, view_size))
    if bool((offsets <= 0).any()) or bool((offsets >= view_size).any()):
        raise RuntimeError("A dephase offset is zero or outside the view")
    return offsets


def spatially_dephase(
    descriptors: Tensor,
    offsets: np.ndarray | Tensor,
) -> Tensor:
    if descriptors.ndim != 4 or int(descriptors.size(1)) != INPUT_CHANNELS:
        raise ValueError("Descriptors must have shape [B, 6, H, W]")
    if isinstance(offsets, Tensor):
        shifts = offsets.detach().cpu().numpy()
    else:
        shifts = np.asarray(offsets)
    expected = (int(descriptors.size(0)), INPUT_CHANNELS, 2)
    if shifts.shape != expected:
        raise ValueError(f"Offsets must have shape {expected}, got {shifts.shape}")
    output = torch.empty_like(descriptors)
    for row in range(expected[0]):
        for channel in range(INPUT_CHANNELS):
            output[row, channel] = torch.roll(
                descriptors[row, channel],
                shifts=(
                    int(shifts[row, channel, 0]),
                    int(shifts[row, channel, 1]),
                ),
                dims=(-2, -1),
            )
    return output


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


def role_seed(role: str, fold: int) -> int:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown CCR role: {role}")
    repeat_offset = (
        REPEAT_SEED_OFFSET
        if role == "cross_colour_ratio_seed_repeat"
        else 0
    )
    return PRIMARY_SEED + int(fold) + repeat_offset


def initialize_role_head(
    role: str,
    *,
    fold: int,
    device: torch.device,
) -> CrossColourRatioEvidenceHead:
    seed = role_seed(role, fold)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CrossColourRatioEvidenceHead()
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
    return {
        "parameter_shapes": shapes,
        "parameter_shapes_sha256": json_sha256(shapes),
        "trainable_parameter_count": int(sum(trainable.values())),
        "trainable_tensor_count": int(len(trainable)),
    }


def map_targets_to_head_indices(targets: np.ndarray | Tensor) -> Tensor:
    values = torch.as_tensor(targets, dtype=torch.long)
    mapped = torch.full_like(values, -1)
    for head_index, target in enumerate(CLASS_ORDER):
        mapped[values == target] = head_index
    if bool((mapped < 0).any()):
        raise ValueError("Targets contain a class outside [0, 1, 2, 4]")
    return mapped


def class1_evidence_score(logits: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(logits, Tensor):
        if logits.ndim != 2 or int(logits.size(1)) != 4:
            raise ValueError("Head logits must have shape [N, 4]")
        return logits[:, 1] - torch.logsumexp(logits[:, [0, 2, 3]], dim=1)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Head logits must have shape [N, 4]")
    rivals = values[:, [0, 2, 3]]
    maximum = rivals.max(axis=1)
    logsumexp = maximum + np.log(
        np.exp(rivals - maximum[:, None]).sum(axis=1)
    )
    return values[:, 1] - logsumexp


def keeper_margin(keeper_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(keeper_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Keeper probabilities must have shape [N, 5]")
    if not bool(np.isfinite(probabilities).all()):
        raise ValueError("Keeper probabilities are non-finite")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    rival = np.max(clipped[:, [0, 2, 4]], axis=1)
    return np.log(clipped[:, FOCUS_CLASS]) - np.log(rival)


def calibrate_class1_retention_threshold(
    scores: np.ndarray,
    class1_mask: np.ndarray,
    *,
    minimum_retention: float = 0.97,
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = np.asarray(class1_mask, dtype=np.bool_).reshape(-1)
    if values.shape != positives.shape or not bool(np.isfinite(values).all()):
        raise ValueError("Calibration scores and class-1 mask do not align")
    if not 0.0 < float(minimum_retention) <= 1.0:
        raise ValueError("Minimum retention must lie in (0, 1]")
    positive_scores = np.sort(values[positives], kind="stable")
    if positive_scores.size == 0:
        raise ValueError("Calibration partition has no class-1 row")
    allowed_breaks = int(
        math.floor(
            (1.0 - float(minimum_retention)) * positive_scores.size + 1e-12
        )
    )
    unique_scores = np.unique(positive_scores)
    break_counts = np.searchsorted(
        positive_scores,
        unique_scores,
        side="left",
    )
    eligible = unique_scores[break_counts <= allowed_breaks]
    if eligible.size == 0:
        raise RuntimeError("No threshold satisfies the retention budget")
    threshold = float(eligible[-1])
    observed_breaks = int((positive_scores < threshold).sum())
    retention = float((positive_scores >= threshold).mean())
    if observed_breaks > allowed_breaks or retention < minimum_retention:
        raise RuntimeError("Selected threshold violates class-1 retention")
    return {
        "threshold": threshold,
        "boundary": "highest_observed_retention_safe",
        "class1_rows": int(positive_scores.size),
        "allowed_class1_breaks": allowed_breaks,
        "observed_class1_breaks": observed_breaks,
        "class1_retention": retention,
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


def margin_crop_box(
    image_width: int,
    image_height: int,
    normalized_bbox_xywh: Sequence[float],
    *,
    margin_ratio: float = 0.05,
) -> Tuple[int, int, int, int]:
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0 or len(normalized_bbox_xywh) != 4:
        raise ValueError("Image size and normalized bbox must be valid")
    x_center, y_center, box_width, box_height = (
        float(value) for value in normalized_bbox_xywh
    )
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    pixel_width = max(1.0, x2 - x1)
    pixel_height = max(1.0, y2 - y1)
    margin_x = pixel_width * max(0.0, float(margin_ratio))
    margin_y = pixel_height * max(0.0, float(margin_ratio))
    left = max(0, int(math.floor(x1 - margin_x)))
    top = max(0, int(math.floor(y1 - margin_y)))
    right = min(width, int(math.ceil(x2 + margin_x)))
    bottom = min(height, int(math.ceil(y2 + margin_y)))
    if right <= left or bottom <= top:
        raise ValueError("Normalized bbox produces an empty margin crop")
    return left, top, right, bottom
