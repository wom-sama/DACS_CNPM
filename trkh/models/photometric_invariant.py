from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ColorInvariantW(nn.Module):
    """Physics-based illumination-intensity invariant edge response.

    The equations follow the W invariant and CIConv construction from Lengyel
    et al., ICCV 2021. The implementation is device-safe and keeps filter
    construction differentiable with respect to scale inside a fixed support.
    Reference implementation: https://github.com/Attila94/CIConv (MIT).
    """

    def __init__(
        self,
        *,
        scale: float = 0.0,
        truncate: float = 3.0,
        trainable_scale: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if float(truncate) <= 0.0:
            raise ValueError("truncate must be positive")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        scale_tensor = torch.tensor(float(scale), dtype=torch.float32)
        if trainable_scale:
            self.scale = nn.Parameter(scale_tensor)
        else:
            self.register_buffer("scale", scale_tensor)
        self.truncate = float(truncate)
        self.eps = float(eps)
        self.register_buffer(
            "gaussian_color_model",
            torch.tensor(
                [
                    [0.06, 0.63, 0.27],
                    [0.30, 0.04, -0.35],
                    [0.34, -0.60, 0.17],
                ],
                dtype=torch.float32,
            ),
        )

    def _basis_filters(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        scale = self.scale.to(device=device, dtype=torch.float32).clamp(-2.5, 2.5)
        sigma = torch.pow(scale.new_tensor(2.0), scale)
        radius = int(torch.ceil(self.truncate * sigma.detach() + 0.5).item())
        coordinate = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        gaussian = torch.exp(-((xx.square() + yy.square()) / (2.0 * sigma.square())))
        gaussian = gaussian / gaussian.sum().clamp_min(self.eps)

        coefficient = -1.0 / (sigma.pow(3) * (2.0 * math.pi))
        derivative_x = coefficient * xx * torch.exp(
            -((xx.square() + yy.square()) / (2.0 * sigma.square()))
        )
        derivative_y = coefficient * yy * torch.exp(
            -((xx.square() + yy.square()) / (2.0 * sigma.square()))
        )
        derivative_x = derivative_x / derivative_x.abs().sum().clamp_min(self.eps)
        derivative_y = derivative_y / derivative_y.abs().sum().clamp_min(self.eps)
        return torch.stack((gaussian, derivative_x, derivative_y), dim=0).unsqueeze(1).to(dtype=dtype)

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or int(rgb.size(1)) != 3:
            raise ValueError("ColorInvariantW expects RGB input shaped [B, 3, H, W]")
        if not torch.is_floating_point(rgb):
            raise TypeError("ColorInvariantW expects floating-point RGB input")
        source_dtype = rgb.dtype
        rgb_float = rgb.to(dtype=torch.float32)
        color_model = self.gaussian_color_model.to(device=rgb.device, dtype=torch.float32)
        channels = torch.einsum("oc,bchw->bohw", color_model, rgb_float)
        energy, spectral_first, spectral_second = channels.split(1, dim=1)
        basis = self._basis_filters(device=rgb.device, dtype=torch.float32)
        padding = int(basis.shape[-1] // 2)

        energy_blur = F.conv2d(energy, basis, padding=padding)
        first_blur = F.conv2d(spectral_first, basis, padding=padding)
        second_blur = F.conv2d(spectral_second, basis, padding=padding)
        energy_value, energy_x, energy_y = energy_blur.split(1, dim=1)
        _, first_x, first_y = first_blur.split(1, dim=1)
        _, second_x, second_y = second_blur.split(1, dim=1)

        denominator = energy_value + self.eps
        response = (
            (energy_x / denominator).square()
            + (energy_y / denominator).square()
            + (first_x / denominator).square()
            + (first_y / denominator).square()
            + (second_x / denominator).square()
            + (second_y / denominator).square()
        )
        log_response = torch.log(response.clamp_min(self.eps))
        mean = log_response.mean(dim=(-2, -1), keepdim=True)
        variance = (log_response - mean).square().mean(dim=(-2, -1), keepdim=True)
        normalized = (log_response - mean) * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=source_dtype)


def suppress_invalid_boundary(
    response: Tensor,
    valid_mask: Tensor | None,
    *,
    boundary_width: int = 2,
) -> Tensor:
    """Remove padding and a narrow derivative halo around invalid pixels."""

    if valid_mask is None:
        return response
    if response.ndim != 4 or int(response.size(1)) != 1:
        raise ValueError("response must have shape [B, 1, H, W]")
    mask = valid_mask
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or int(mask.size(1)) != 1:
        raise ValueError("valid_mask must have shape [B, H, W] or [B, 1, H, W]")
    mask = mask.to(device=response.device, dtype=torch.float32)
    if tuple(mask.shape[-2:]) != tuple(response.shape[-2:]):
        mask = F.interpolate(mask, size=response.shape[-2:], mode="nearest")
    width = max(0, int(boundary_width))
    if width > 0:
        kernel = 2 * width + 1
        invalid = 1.0 - mask.clamp(0.0, 1.0)
        mask = 1.0 - F.max_pool2d(invalid, kernel_size=kernel, stride=1, padding=width)
    return response * mask.to(dtype=response.dtype).clamp(0.0, 1.0)


def invariant_response_to_model_view(response: Tensor, *, clip: float = 3.0) -> Tensor:
    """Replicate a standardized invariant response for a frozen RGB trunk probe."""

    limit = float(clip)
    if limit <= 0.0:
        raise ValueError("clip must be positive")
    if response.ndim != 4 or int(response.size(1)) != 1:
        raise ValueError("response must have shape [B, 1, H, W]")
    return response.clamp(-limit, limit).repeat(1, 3, 1, 1)


def _gray_edge_derivative_filters(
    *,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma_value)))
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    gaussian = torch.exp(-((xx.square() + yy.square()) / (2.0 * sigma_value * sigma_value)))
    derivative_x = -(xx / (sigma_value * sigma_value)) * gaussian
    derivative_y = -(yy / (sigma_value * sigma_value)) * gaussian
    derivative_x = derivative_x / derivative_x.abs().sum().clamp_min(1e-8)
    derivative_y = derivative_y / derivative_y.abs().sum().clamp_min(1e-8)
    return derivative_x.to(dtype=dtype), derivative_y.to(dtype=dtype)


def _coerce_spatial_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(
            (reference.size(0), 1, reference.size(-2), reference.size(-1)),
            device=reference.device,
            dtype=torch.float32,
        )
    value = mask
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or int(value.size(1)) != 1:
        raise ValueError("spatial mask must have shape [B, H, W] or [B, 1, H, W]")
    value = value.to(device=reference.device, dtype=torch.float32)
    if tuple(value.shape[-2:]) != tuple(reference.shape[-2:]):
        value = F.interpolate(value, size=reference.shape[-2:], mode="nearest")
    return value.clamp(0.0, 1.0)


def _exclude_normalized_bboxes(
    mask: Tensor,
    bbox: Tensor | None,
    *,
    margin_ratio: float,
) -> Tensor:
    if bbox is None:
        return mask
    boxes = bbox
    if boxes.ndim == 3:
        boxes = boxes[:, 0]
    if boxes.ndim != 2 or int(boxes.size(1)) < 4:
        raise ValueError("bbox must have shape [B, 4]")
    boxes = boxes[:, :4].to(device=mask.device, dtype=torch.float32).clamp(0.0, 1.0)
    x_center, y_center, width, height = boxes.unbind(dim=1)
    margin = max(0.0, float(margin_ratio))
    half_width = width * (0.5 + margin)
    half_height = height * (0.5 + margin)
    x_min = (x_center - half_width).clamp(0.0, 1.0).view(-1, 1, 1)
    x_max = (x_center + half_width).clamp(0.0, 1.0).view(-1, 1, 1)
    y_min = (y_center - half_height).clamp(0.0, 1.0).view(-1, 1, 1)
    y_max = (y_center + half_height).clamp(0.0, 1.0).view(-1, 1, 1)
    image_height, image_width = int(mask.size(-2)), int(mask.size(-1))
    x_grid = (
        (torch.arange(image_width, device=mask.device, dtype=torch.float32) + 0.5)
        / float(max(1, image_width))
    ).view(1, 1, image_width)
    y_grid = (
        (torch.arange(image_height, device=mask.device, dtype=torch.float32) + 0.5)
        / float(max(1, image_height))
    ).view(1, image_height, 1)
    inside = (x_grid >= x_min) & (x_grid <= x_max) & (y_grid >= y_min) & (y_grid <= y_max)
    return mask * (~inside).unsqueeze(1).to(dtype=mask.dtype)


def estimate_gray_edge_illuminant(
    rgb: Tensor,
    *,
    valid_mask: Tensor | None = None,
    excluded_bbox: Tensor | None = None,
    bbox_margin_ratio: float = 0.08,
    sigma: float = 2.0,
    minkowski_p: float = 6.0,
    saturation_low: float = 0.02,
    saturation_high: float = 0.98,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Estimate scene illuminant with first-order Gray-Edge statistics.

    The target bbox may be excluded so broad context estimates the nuisance
    illuminant without normalizing the object's class-bearing color itself.
    Returns an L2-normalized RGB illuminant and the supported-pixel fraction.
    """

    if rgb.ndim != 4 or int(rgb.size(1)) != 3:
        raise ValueError("estimate_gray_edge_illuminant expects [B, 3, H, W]")
    if float(minkowski_p) <= 0.0:
        raise ValueError("minkowski_p must be positive")
    if not 0.0 <= float(saturation_low) < float(saturation_high) <= 1.0:
        raise ValueError("saturation limits must satisfy 0 <= low < high <= 1")
    source_dtype = rgb.dtype
    image = rgb.to(dtype=torch.float32).clamp(0.0, 1.0)
    derivative_x, derivative_y = _gray_edge_derivative_filters(
        sigma=float(sigma),
        device=image.device,
        dtype=torch.float32,
    )
    kernel_size = int(derivative_x.size(0))
    padding = kernel_size // 2
    weight_x = derivative_x.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    weight_y = derivative_y.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    gradient_x = F.conv2d(image, weight_x, padding=padding, groups=3)
    gradient_y = F.conv2d(image, weight_y, padding=padding, groups=3)
    magnitude = torch.sqrt((gradient_x.square() + gradient_y.square()).clamp_min(0.0)).clamp_min(
        float(eps)
    )

    mask = _coerce_spatial_mask(valid_mask, image)
    if padding > 0:
        invalid = 1.0 - mask
        mask = 1.0 - F.max_pool2d(
            invalid,
            kernel_size=2 * padding + 1,
            stride=1,
            padding=padding,
        )
        mask[..., :padding, :] = 0.0
        mask[..., -padding:, :] = 0.0
        mask[..., :, :padding] = 0.0
        mask[..., :, -padding:] = 0.0
    mask = _exclude_normalized_bboxes(
        mask.clamp(0.0, 1.0),
        excluded_bbox,
        margin_ratio=float(bbox_margin_ratio),
    )
    unsaturated = (
        (image.amin(dim=1, keepdim=True) > float(saturation_low))
        & (image.amax(dim=1, keepdim=True) < float(saturation_high))
    ).to(dtype=torch.float32)
    mask = mask * unsaturated
    supported = mask.flatten(1).sum(dim=1)
    support_fraction = supported / float(max(1, image.size(-2) * image.size(-1)))
    p_value = float(minkowski_p)
    channel_moment = (
        magnitude.pow(p_value) * mask
    ).flatten(2).sum(dim=2) / supported.view(-1, 1).clamp_min(1.0)
    illuminant = channel_moment.clamp_min(float(eps) ** p_value).pow(1.0 / p_value)
    fallback = image.new_full((image.size(0), 3), 1.0 / math.sqrt(3.0))
    illuminant = illuminant / illuminant.norm(dim=1, keepdim=True).clamp_min(float(eps))
    valid_estimate = (supported >= 16.0).view(-1, 1)
    illuminant = torch.where(valid_estimate, illuminant, fallback)
    return illuminant.to(dtype=source_dtype), support_fraction.to(dtype=source_dtype)


def apply_diagonal_color_constancy(
    rgb: Tensor,
    illuminant: Tensor,
    *,
    valid_mask: Tensor | None = None,
    min_gain: float = 0.5,
    max_gain: float = 2.0,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Apply a diagonal von Kries correction while preserving padded pixels."""

    if rgb.ndim != 4 or int(rgb.size(1)) != 3:
        raise ValueError("apply_diagonal_color_constancy expects [B, 3, H, W]")
    if illuminant.ndim != 2 or tuple(illuminant.shape) != (int(rgb.size(0)), 3):
        raise ValueError("illuminant must have shape [B, 3]")
    if float(min_gain) <= 0.0 or float(max_gain) < float(min_gain):
        raise ValueError("gain bounds must satisfy 0 < min_gain <= max_gain")
    light = illuminant.to(device=rgb.device, dtype=torch.float32).clamp_min(float(eps))
    light = light / light.mean(dim=1, keepdim=True).clamp_min(float(eps))
    gains = light.reciprocal().clamp(float(min_gain), float(max_gain))
    source = rgb.to(dtype=torch.float32)
    corrected = (source * gains.view(-1, 3, 1, 1)).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = _coerce_spatial_mask(valid_mask, source)
        corrected = corrected * mask + source * (1.0 - mask)
    return corrected.to(dtype=rgb.dtype), gains.to(dtype=rgb.dtype)
