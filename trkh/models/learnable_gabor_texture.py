from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _inverse_sigmoid(value: Tensor, eps: float = 1e-5) -> Tensor:
    value = value.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(value) - torch.log1p(-value)


class CompactSelfAttention(nn.Module):
    """Small export-friendly multi-head self-attention without dropout."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if int(dim) <= 0 or int(num_heads) <= 0 or int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be positive and divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or int(inputs.size(-1)) != self.dim:
            raise ValueError("CompactSelfAttention expects [B,N,dim].")
        batch_size, token_count, _ = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        attention = torch.matmul(query, key.transpose(-2, -1)) * float(self.scale)
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch_size, token_count, self.dim)
        return self.proj(output), attention


class LearnableGaborTextureResidual(nn.Module):
    """Constrained Gabor + LHO/FCM texture descriptor for one TRKH edge token."""

    filter_count = 32
    low_filter_count = 16
    histogram_levels = 8
    analysis_size = 64
    response_size = 16
    kernel_size = 11
    hidden_dim = 64
    attention_heads = 4
    interior_margin_ratio = 0.06
    gate_scale = 0.10

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if int(embed_dim) <= 0:
            raise ValueError("embed_dim must be positive.")
        self.embed_dim = int(embed_dim)

        kernel_size = float(self.kernel_size)
        maximum_frequency = (2.0 * math.pi * kernel_size - 25.0) / (
            4.0 * math.pi * kernel_size
        )
        frequency_lower = torch.zeros(self.filter_count, dtype=torch.float32)
        frequency_upper = torch.full_like(frequency_lower, maximum_frequency / 2.0)
        frequency_lower[self.low_filter_count :] = maximum_frequency / 2.0
        frequency_upper[self.low_filter_count :] = maximum_frequency
        self.register_buffer("frequency_lower", frequency_lower)
        self.register_buffer("frequency_upper", frequency_upper)
        self.register_buffer(
            "sigma_y_bounds",
            torch.tensor(
                [5.0 / (2.0 * math.pi), kernel_size / 5.0],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

        coordinates = torch.arange(self.kernel_size, dtype=torch.float32)
        coordinates = coordinates - float(self.kernel_size - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("kernel_grid_x", grid_x.clone())
        self.register_buffer("kernel_grid_y", grid_y.clone())

        response_axis = (
            torch.arange(self.response_size, dtype=torch.float32) + 0.5
        ) / float(self.response_size)
        position_y, position_x = torch.meshgrid(response_axis, response_axis, indexing="ij")
        position_features = torch.stack(
            (
                torch.sin(math.pi * position_x),
                torch.cos(math.pi * position_x),
                torch.sin(math.pi * position_y),
                torch.cos(math.pi * position_y),
            ),
            dim=-1,
        ).reshape(self.response_size * self.response_size, 4)
        self.register_buffer("position_features", position_features)
        self.register_buffer(
            "histogram_level_fractions",
            torch.linspace(0.0, 1.0, self.histogram_levels, dtype=torch.float32),
        )

        orientation_fraction = (
            torch.arange(self.filter_count, dtype=torch.float32) + 0.5
        ) / float(self.filter_count)
        group_fraction = (
            torch.arange(self.low_filter_count, dtype=torch.float32) + 0.5
        ) / float(self.low_filter_count)
        frequency_fraction = torch.cat((group_fraction, group_fraction), dim=0)
        self.raw_theta = nn.Parameter(_inverse_sigmoid(orientation_fraction))
        self.raw_frequency = nn.Parameter(_inverse_sigmoid(frequency_fraction))
        self.raw_sigma_x = nn.Parameter(torch.zeros(self.filter_count, dtype=torch.float32))
        self.raw_sigma_y = nn.Parameter(torch.zeros(self.filter_count, dtype=torch.float32))

        # The optional branch must not shift the base model's later random
        # initialization or the train-time dropout/augmentation RNG stream.
        # fork_rng keeps initialization seed-dependent while restoring the
        # caller's CPU RNG state after these randomized layers are created.
        with torch.random.fork_rng(devices=[]):
            self.level_projection = nn.Linear(2, self.hidden_dim, bias=True)
            self.position_projection = nn.Linear(4, self.hidden_dim, bias=False)
            self.lho_attention = CompactSelfAttention(
                self.hidden_dim,
                self.attention_heads,
            )
            self.lho_norm = nn.LayerNorm(self.hidden_dim)
            self.filter_parameter_projection = nn.Linear(
                4,
                self.hidden_dim,
                bias=True,
            )
            self.fcm_attention = CompactSelfAttention(
                self.hidden_dim,
                self.attention_heads,
            )
            self.fcm_norm = nn.LayerNorm(self.hidden_dim)
            self.output_norm = nn.LayerNorm(self.hidden_dim)
            self.output_projection = nn.Linear(
                self.hidden_dim,
                self.embed_dim,
                bias=True,
            )
            self.token_norm = nn.LayerNorm(self.embed_dim)
        self.raw_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def high_filter_count(self) -> int:
        return self.filter_count - self.low_filter_count

    @property
    def maximum_frequency(self) -> float:
        return float(self.frequency_upper[-1].item())

    def effective_gate(self) -> Tensor:
        return torch.tanh(self.raw_gate) * float(self.gate_scale)

    def constrained_parameters(self) -> Dict[str, Tensor]:
        theta = math.pi * torch.sigmoid(self.raw_theta)
        frequency_fraction = torch.sigmoid(self.raw_frequency)
        frequency = self.frequency_lower + (
            self.frequency_upper - self.frequency_lower
        ) * frequency_fraction
        sigma_x_lower = 5.0 / (2.0 * math.pi * (1.0 - 2.0 * frequency))
        sigma_x_upper = torch.full_like(sigma_x_lower, float(self.kernel_size) / 5.0)
        sigma_x = sigma_x_lower + (
            sigma_x_upper - sigma_x_lower
        ) * torch.sigmoid(self.raw_sigma_x)
        sigma_y_lower = self.sigma_y_bounds[0]
        sigma_y_upper = self.sigma_y_bounds[1]
        sigma_y = sigma_y_lower + (sigma_y_upper - sigma_y_lower) * torch.sigmoid(
            self.raw_sigma_y
        )
        normalized = torch.stack(
            (
                theta / math.pi,
                (sigma_x - sigma_x_lower)
                / (sigma_x_upper - sigma_x_lower).clamp_min(1e-6),
                (sigma_y - sigma_y_lower) / (sigma_y_upper - sigma_y_lower),
                frequency_fraction,
            ),
            dim=1,
        )
        return {
            "theta": theta,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "frequency": frequency,
            "sigma_x_lower": sigma_x_lower,
            "sigma_x_upper": sigma_x_upper,
            "normalized": normalized,
        }

    @staticmethod
    def _normalize_kernel(kernel: Tensor) -> Tensor:
        kernel = kernel - kernel.mean(dim=(-2, -1), keepdim=True)
        norm = kernel.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return kernel / norm

    def build_kernels(self) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        parameters = self.constrained_parameters()
        theta = parameters["theta"].view(-1, 1, 1)
        sigma_x = parameters["sigma_x"].view(-1, 1, 1)
        sigma_y = parameters["sigma_y"].view(-1, 1, 1)
        frequency = parameters["frequency"].view(-1, 1, 1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        rotated_x = self.kernel_grid_x * cosine + self.kernel_grid_y * sine
        rotated_y = -self.kernel_grid_x * sine + self.kernel_grid_y * cosine
        envelope = torch.exp(
            -0.5
            * (
                rotated_x.square() / sigma_x.square().clamp_min(1e-6)
                + rotated_y.square() / sigma_y.square().clamp_min(1e-6)
            )
        )
        phase = 2.0 * math.pi * frequency * rotated_x
        real_kernel = self._normalize_kernel(envelope * torch.cos(phase)).unsqueeze(1)
        imaginary_kernel = self._normalize_kernel(envelope * torch.sin(phase)).unsqueeze(1)
        return real_kernel, imaginary_kernel, parameters

    def _resolve_bbox(self, bbox: Optional[Tensor], batch_size, device: torch.device) -> Tuple[Tensor, Tensor]:
        full_bbox = torch.tensor(
            [0.5, 0.5, 1.0, 1.0],
            device=device,
            dtype=torch.float32,
        ).view(1, 4).expand(batch_size, -1)
        if bbox is None or not torch.is_tensor(bbox):
            return full_bbox, torch.ones(batch_size, device=device, dtype=torch.bool)
        bbox_value = bbox
        if bbox_value.ndim == 3:
            bbox_value = bbox_value[:, 0]
        if bbox_value.ndim != 2 or int(bbox_value.size(1)) < 4:
            return full_bbox, torch.ones(batch_size, device=device, dtype=torch.bool)
        bbox_value = bbox_value[:, :4].to(device=device, dtype=torch.float32)
        valid = torch.isfinite(bbox_value).all(dim=1)
        valid = valid & (bbox_value[:, 2] > 1e-3) & (bbox_value[:, 3] > 1e-3)
        bbox_value = bbox_value.clamp(0.0, 1.0)
        bbox_value = torch.where(valid[:, None], bbox_value, full_bbox)
        return bbox_value, ~valid

    def _texture_input(self, image: Tensor, bbox: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if image.ndim != 4 or int(image.size(1)) != 3:
            raise ValueError("Learnable Gabor texture input must have shape [B,3,H,W].")
        image_float = image.to(dtype=torch.float32)
        rgb = (image_float * self.rgb_std + self.rgb_mean).clamp(0.0, 1.0)
        gray = rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
        gray = F.interpolate(
            gray,
            size=(self.analysis_size, self.analysis_size),
            mode="bilinear",
            align_corners=False,
        )

        bbox_value, fallback = self._resolve_bbox(bbox, image.shape[0], image.device)
        cx, cy, width, height = bbox_value.unbind(dim=1)
        margin = float(self.interior_margin_ratio)
        half_width = (0.5 - margin) * width
        half_height = (0.5 - margin) * height
        x1 = (cx - half_width).clamp(0.0, 1.0)
        x2 = (cx + half_width).clamp(0.0, 1.0)
        y1 = (cy - half_height).clamp(0.0, 1.0)
        y2 = (cy + half_height).clamp(0.0, 1.0)
        axis = (
            torch.arange(self.analysis_size, device=image.device, dtype=torch.float32) + 0.5
        ) / float(self.analysis_size)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        softness = 1.0 / float(self.analysis_size)
        mask = (
            torch.sigmoid((grid_x[None] - x1[:, None, None]) / softness)
            * torch.sigmoid((x2[:, None, None] - grid_x[None]) / softness)
            * torch.sigmoid((grid_y[None] - y1[:, None, None]) / softness)
            * torch.sigmoid((y2[:, None, None] - grid_y[None]) / softness)
        ).unsqueeze(1)
        mask = torch.where(fallback[:, None, None, None], torch.ones_like(mask), mask)
        denominator = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mean = (gray * mask).sum(dim=(-2, -1), keepdim=True) / denominator
        variance = ((gray - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denominator
        normalized = (gray - mean) / torch.sqrt(variance + 1e-6)
        return normalized * mask, mask, fallback

    def _filter_slice(self, use_low: bool, use_high: bool) -> slice:
        if bool(use_low) and bool(use_high):
            return slice(0, self.filter_count)
        if bool(use_low):
            return slice(0, self.low_filter_count)
        if bool(use_high):
            return slice(self.low_filter_count, self.filter_count)
        raise ValueError("At least one Gabor frequency group must remain active.")

    def forward_components(
        self,
        image: Tensor,
        bbox: Optional[Tensor] = None,
        *,
        kernel_override: Optional[Tuple[Tensor, Tensor]] = None,
        use_low: bool = True,
        use_high: bool = True,
        use_lho_position: bool = True,
        use_filter_parameter_encoding: bool = True,
    ) -> Dict[str, Tensor]:
        device_type = str(image.device.type)
        with torch.autocast(device_type=device_type, enabled=False):
            texture_input, mask, fallback = self._texture_input(image, bbox)
            dynamic_real, dynamic_imaginary, parameters = self.build_kernels()
            if kernel_override is None:
                real_kernel = dynamic_real
                imaginary_kernel = dynamic_imaginary
            else:
                real_kernel, imaginary_kernel = kernel_override
                real_kernel = real_kernel.to(device=image.device, dtype=torch.float32)
                imaginary_kernel = imaginary_kernel.to(device=image.device, dtype=torch.float32)
                expected = (self.filter_count, 1, self.kernel_size, self.kernel_size)
                if tuple(real_kernel.shape) != expected or tuple(imaginary_kernel.shape) != expected:
                    raise ValueError(f"Materialized Gabor kernels must have shape {expected}.")

            real_response = F.conv2d(
                texture_input,
                real_kernel,
                padding=self.kernel_size // 2,
            )
            imaginary_response = F.conv2d(
                texture_input,
                imaginary_kernel,
                padding=self.kernel_size // 2,
            )
            magnitude = torch.sqrt(
                real_response.square() + imaginary_response.square() + 1e-8
            )
            magnitude = F.avg_pool2d(
                magnitude,
                kernel_size=self.analysis_size // self.response_size,
                stride=self.analysis_size // self.response_size,
            )
            unmasked_magnitude = magnitude
            response_mask = F.avg_pool2d(
                mask,
                kernel_size=self.analysis_size // self.response_size,
                stride=self.analysis_size // self.response_size,
            )
            magnitude = magnitude * response_mask

            selected = self._filter_slice(use_low=use_low, use_high=use_high)
            selected_magnitude = magnitude[:, selected]
            selected_parameters = parameters["normalized"][selected]
            valid = response_mask > 0.05
            valid = valid.expand(-1, int(selected_magnitude.size(1)), -1, -1)
            upper_fill = selected_magnitude.amax(dim=(-2, -1), keepdim=True) + 1.0
            response_min = torch.where(valid, selected_magnitude, upper_fill).amin(
                dim=(-2, -1)
            )
            response_max = torch.where(
                valid,
                selected_magnitude,
                torch.zeros_like(selected_magnitude),
            ).amax(dim=(-2, -1))
            response_range = (response_max - response_min).clamp_min(1e-4)
            levels = response_min.unsqueeze(-1) + response_range.unsqueeze(-1) * (
                self.histogram_level_fractions.view(1, 1, -1)
            )
            bin_width = (response_range / float(self.histogram_levels - 1)).clamp_min(1e-4)
            assignment = F.relu(
                1.0
                - (
                    selected_magnitude.unsqueeze(2)
                    - levels.unsqueeze(-1).unsqueeze(-1)
                ).abs()
                / bin_width.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            assignment = assignment * response_mask.unsqueeze(2)
            counts = assignment.sum(dim=(-2, -1))
            counts = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            global_level_scale = response_max.amax(dim=1, keepdim=True).clamp_min(1e-6)
            normalized_levels = levels / global_level_scale.unsqueeze(-1)
            level_inputs = torch.stack((counts, normalized_levels), dim=-1)
            level_embeddings = self.level_projection(level_inputs)

            assignment_flat = assignment.flatten(-2)
            position_descriptor = torch.matmul(
                assignment_flat,
                self.position_features,
            ) / float(self.response_size * self.response_size)
            if bool(use_lho_position):
                level_embeddings = level_embeddings + self.position_projection(
                    position_descriptor
                )
            batch_size, selected_count, level_count, _ = level_embeddings.shape
            lho_input = level_embeddings.reshape(
                batch_size * selected_count,
                level_count,
                self.hidden_dim,
            )
            lho_attention_output, lho_attention = self.lho_attention(lho_input)
            lho_output = self.lho_norm(lho_input + lho_attention_output)
            lho_features = lho_output.mean(dim=1).reshape(
                batch_size,
                selected_count,
                self.hidden_dim,
            )

            filter_features = lho_features
            parameter_encoding = self.filter_parameter_projection(selected_parameters)
            if bool(use_filter_parameter_encoding):
                filter_features = filter_features + parameter_encoding.unsqueeze(0)
            fcm_attention_output, fcm_attention = self.fcm_attention(filter_features)
            fcm_output = self.fcm_norm(filter_features + fcm_attention_output)
            descriptor = self.output_norm(fcm_output.mean(dim=1))
            texture_token = self.token_norm(self.output_projection(descriptor))

        return {
            "texture_input": texture_input,
            "mask": mask,
            "bbox_fallback": fallback,
            "real_kernel": dynamic_real,
            "imaginary_kernel": dynamic_imaginary,
            "real_response": real_response,
            "imaginary_response": imaginary_response,
            "unmasked_magnitude": unmasked_magnitude,
            "magnitude": magnitude,
            "response_mask": response_mask,
            "levels": levels,
            "assignment": assignment,
            "counts": counts,
            "normalized_levels": normalized_levels,
            "level_embeddings": level_embeddings,
            "position_descriptor": position_descriptor,
            "lho_attention": lho_attention,
            "lho_output": lho_output,
            "lho_features": lho_features,
            "parameter_encoding": parameter_encoding,
            "filter_features": filter_features,
            "fcm_attention": fcm_attention,
            "fcm_output": fcm_output,
            "texture_descriptor": descriptor,
            "texture_token": texture_token,
            "theta": parameters["theta"],
            "sigma_x": parameters["sigma_x"],
            "sigma_y": parameters["sigma_y"],
            "frequency": parameters["frequency"],
            "normalized_parameters": parameters["normalized"],
        }

    def materialized_kernels(self) -> Tuple[Tensor, Tensor]:
        real_kernel, imaginary_kernel, _ = self.build_kernels()
        return real_kernel.detach().clone(), imaginary_kernel.detach().clone()

    def forward(
        self,
        image: Tensor,
        bbox: Optional[Tensor] = None,
        *,
        return_trace: bool = False,
        kernel_override: Optional[Tuple[Tensor, Tensor]] = None,
        use_low: bool = True,
        use_high: bool = True,
        use_lho_position: bool = True,
        use_filter_parameter_encoding: bool = True,
    ):
        components = self.forward_components(
            image,
            bbox,
            kernel_override=kernel_override,
            use_low=use_low,
            use_high=use_high,
            use_lho_position=use_lho_position,
            use_filter_parameter_encoding=use_filter_parameter_encoding,
        )
        gate = self.effective_gate()
        residual = components["texture_token"].to(dtype=image.dtype) * gate.to(
            device=image.device,
            dtype=image.dtype,
        )
        if not return_trace:
            return residual

        magnitude = components["magnitude"]
        unmasked_magnitude = components["unmasked_magnitude"]
        response_mask = components["response_mask"]
        foreground_mass = (unmasked_magnitude * response_mask).sum(
            dim=(-2, -1)
        ) / unmasked_magnitude.sum(dim=(-2, -1)).clamp_min(1e-6)
        counts = components["counts"]
        entropy = -(counts * counts.clamp_min(1e-8).log()).sum(dim=-1)
        trace = {
            "mask": components["mask"].detach(),
            "bbox_fallback": components["bbox_fallback"].detach(),
            "theta": components["theta"].detach(),
            "sigma_x": components["sigma_x"].detach(),
            "sigma_y": components["sigma_y"].detach(),
            "frequency": components["frequency"].detach(),
            "low_response_map": magnitude[:, : self.low_filter_count].mean(
                dim=1,
                keepdim=True,
            ).detach(),
            "high_response_map": magnitude[:, self.low_filter_count :].mean(
                dim=1,
                keepdim=True,
            ).detach(),
            "response_foreground_mass": foreground_mass.detach(),
            "mask_area_fraction": response_mask.mean(dim=(-2, -1)).detach(),
            "counts": counts.detach(),
            "entropy": entropy.detach(),
            "filter_features": components["filter_features"].detach(),
            "texture_descriptor": components["texture_descriptor"].detach(),
            "texture_token": components["texture_token"].detach(),
            "gate": gate.detach(),
            "residual": residual.detach(),
        }
        return residual, trace
