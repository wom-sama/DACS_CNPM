from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torchvision import models as tv_models

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


def _parse_pairwise_margin_pairs(pairs: str, num_classes: int) -> List[Tuple[int, int]]:
    parsed: List[Tuple[int, int]] = []
    for raw_item in str(pairs or "").replace(";", ",").split(","):
        item = raw_item.strip().lower().replace(":", "-")
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"pairwise margin pair khong hop le: {raw_item!r}")
        left_text, right_text = [part.strip() for part in item.split("-", 1)]
        if right_text in {"rest", "others", "other"}:
            pair = (-1, int(left_text))
        elif left_text in {"rest", "others", "other"}:
            pair = (-1, int(right_text))
        else:
            pair = (int(left_text), int(right_text))
        for class_index in pair:
            if class_index >= 0 and class_index >= int(num_classes):
                raise ValueError(
                    "pairwise margin pair nam ngoai khoang class: "
                    f"pair={raw_item!r}, num_classes={num_classes}"
                )
        if pair not in parsed:
            parsed.append(pair)
    return parsed


def _parse_ordered_class_indices(classes: str, num_classes: int) -> List[int]:
    parsed: List[int] = []
    for raw_item in str(classes or "").replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        class_index = int(item)
        if class_index < 0 or class_index >= int(num_classes):
            raise ValueError(
                "ordinal maturity class nam ngoai khoang: "
                f"class={class_index}, num_classes={num_classes}"
            )
        if class_index not in parsed:
            parsed.append(class_index)
    if parsed and len(parsed) < 2:
        raise ValueError("ordinal maturity head can it nhat 2 class co thu tu.")
    return parsed


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        dual_patch_norm: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size phai chia het cho patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.base_grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.dual_patch_norm = bool(dual_patch_norm)
        self.patch_vector_dim = int(in_channels) * int(patch_size) * int(patch_size)
        if self.dual_patch_norm:
            self.pre_patch_norm = nn.LayerNorm(self.patch_vector_dim)
            self.proj = nn.Linear(self.patch_vector_dim, embed_dim)
            self.post_patch_norm = nn.LayerNorm(embed_dim)
        else:
            self.pre_patch_norm = None
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.post_patch_norm = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dual_patch_norm:
            patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
            patches = patches.transpose(1, 2)
            patches = self.pre_patch_norm(patches)
            patches = self.proj(patches)
            return self.post_patch_norm(patches)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvStemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("norm", nn.BatchNorm2d(out_channels)),
                    ("act", nn.GELU()),
                    ("pool", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class HybridConvStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        mid_channels = stem_channels * 2
        self.blocks = nn.Sequential(
            ConvStemBlock(in_channels, stem_channels),
            ConvStemBlock(stem_channels, mid_channels),
            ConvStemBlock(mid_channels, embed_dim),
        )
        self.downsample_factor = 8
        self.out_channels = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class FineGrainedPatchPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        hidden_dim = max(32, int(dim))
        self.score = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(dim, dim),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.fusion[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def attention_weights(
        self,
        global_feature: Tensor,
        patch_tokens: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            return patch_tokens.new_zeros((patch_tokens.size(0), patch_tokens.size(1)))
        normalized_patches = self.patch_norm(patch_tokens)
        context = self.context_norm(global_feature).unsqueeze(1).expand_as(normalized_patches)
        scores = self.score(torch.cat((normalized_patches, context), dim=-1)).squeeze(-1)
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == tuple(scores.shape):
            valid = valid_mask.to(device=scores.device, dtype=torch.bool)
            all_invalid = ~valid.any(dim=1)
            if all_invalid.any():
                valid = valid.clone()
                valid[all_invalid] = True
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)

    def forward(
        self,
        global_feature: Tensor,
        patch_tokens: Tensor,
        attention: Optional[Tensor] = None,
    ) -> Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            return global_feature
        if attention is None:
            attention = self.attention_weights(global_feature, patch_tokens)
        patch_feature = torch.bmm(attention.unsqueeze(1), patch_tokens).squeeze(1)
        residual = self.fusion(torch.cat((global_feature, patch_feature), dim=-1))
        return global_feature + residual


class FrequencySelectivePatchPooling(nn.Module):
    """Aggregate feature dimensions from frequency-stable patch tokens."""

    def __init__(
        self,
        dim: int,
        top_k: int = 1,
        foreground_threshold: float = 0.35,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_k = max(1, int(top_k))
        self.foreground_threshold = float(
            min(max(foreground_threshold, 0.0), 1.0)
        )
        self.eps = float(max(eps, 1e-12))
        positions = torch.arange(
            -(self.dim // 2) + 1,
            (self.dim // 2) + 1,
            dtype=torch.float32,
        )
        if int(positions.numel()) != self.dim:
            positions = torch.linspace(
                -(self.dim - 1) / 2.0,
                (self.dim - 1) / 2.0,
                steps=self.dim,
                dtype=torch.float32,
            )
        sigma = math.sqrt(float(max(1, self.dim)))
        kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
        kernel = kernel / kernel.max().clamp(min=1e-8)
        self.register_buffer(
            "gaussian_kernel",
            kernel.view(1, 1, self.dim),
            persistent=False,
        )

    def forward(
        self,
        patch_tokens: Tensor,
        valid_mask: Optional[Tensor] = None,
        foreground_prior: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            selected = patch_tokens.new_zeros((patch_tokens.size(0), self.dim))
            votes = patch_tokens.new_zeros(patch_tokens.shape[:2], dtype=torch.float32)
            if not return_trace:
                return selected
            return selected, {"vote_fraction": votes}

        source = patch_tokens.to(dtype=torch.float32)
        frequency = torch.fft.fft(source, dim=-1)
        frequency = torch.fft.fftshift(frequency, dim=-1)
        frequency = frequency * self.gaussian_kernel.to(device=source.device)
        filtered = torch.fft.ifftshift(frequency, dim=-1)
        filtered = torch.fft.ifft(filtered, dim=-1).real
        stability = source / (filtered - source).abs().clamp(min=self.eps)

        valid = torch.ones(
            source.shape[:2],
            device=source.device,
            dtype=torch.bool,
        )
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == tuple(source.shape[:2]):
            valid = valid & valid_mask.to(device=source.device, dtype=torch.bool)
        candidate_mask = valid
        if (
            torch.is_tensor(foreground_prior)
            and tuple(foreground_prior.shape) == tuple(source.shape[:2])
            and self.foreground_threshold > 0.0
        ):
            foreground_candidate = foreground_prior.to(
                device=source.device,
                dtype=torch.float32,
            ) >= self.foreground_threshold
            gated = valid & foreground_candidate
            enough_candidates = gated.sum(dim=1) >= min(self.top_k, int(source.size(1)))
            candidate_mask = torch.where(
                enough_candidates[:, None],
                gated,
                valid,
            )
        all_invalid = ~candidate_mask.any(dim=1)
        if all_invalid.any():
            candidate_mask = candidate_mask.clone()
            candidate_mask[all_invalid] = True
        stability = stability.masked_fill(
            ~candidate_mask.unsqueeze(-1),
            torch.finfo(stability.dtype).min,
        )

        selected_count = min(self.top_k, int(source.size(1)))
        selected_indices = torch.topk(
            stability,
            k=selected_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        selected = torch.gather(source, dim=1, index=selected_indices).mean(dim=1)

        vote_counts = source.new_zeros(source.shape[:2])
        flattened_indices = selected_indices.reshape(source.size(0), -1)
        vote_counts.scatter_add_(
            1,
            flattened_indices,
            torch.ones_like(flattened_indices, dtype=vote_counts.dtype),
        )
        vote_fraction = vote_counts / float(max(1, self.dim * selected_count))
        selected = selected.to(dtype=patch_tokens.dtype)
        if not return_trace:
            return selected
        return selected, {
            "vote_fraction": vote_fraction.detach(),
            "selected_indices": selected_indices.detach(),
            "candidate_mask": candidate_mask.detach(),
        }


class CompactBilinearPatchFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        rank: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = max(8, int(rank))
        self.patch_norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, self.rank, bias=False)
        self.right_proj = nn.Linear(dim, self.rank, bias=False)
        self.descriptor_dim = (self.rank * self.rank) + (2 * self.rank)
        hidden = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _normalized_weights(
        patch_tokens: Tensor,
        attention: Optional[Tensor],
        valid_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, token_count = patch_tokens.shape[:2]
        if torch.is_tensor(attention) and tuple(attention.shape) == (batch_size, token_count):
            weights = attention.to(device=patch_tokens.device, dtype=torch.float32).clamp(min=0.0)
        else:
            weights = torch.ones(
                (batch_size, token_count),
                device=patch_tokens.device,
                dtype=torch.float32,
            )
        valid = None
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=patch_tokens.device, dtype=torch.bool)
            weights = weights * valid.to(dtype=weights.dtype)
        totals = weights.sum(dim=1, keepdim=True)
        empty = totals.squeeze(1) <= 1e-8
        if empty.any():
            fallback = (
                valid.to(dtype=weights.dtype)
                if valid is not None
                else torch.ones_like(weights)
            )
            fallback_totals = fallback.sum(dim=1, keepdim=True)
            fallback_empty = fallback_totals.squeeze(1) <= 1e-8
            if fallback_empty.any():
                fallback = fallback.clone()
                fallback[fallback_empty] = 1.0
            weights = torch.where(empty[:, None], fallback, weights)
            totals = weights.sum(dim=1, keepdim=True)
        return weights / totals.clamp(min=1e-8)

    def forward(
        self,
        patch_tokens: Tensor,
        attention: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            logits = patch_tokens.new_zeros((patch_tokens.size(0), self.net[-1].out_features))
            if not return_trace:
                return logits
            return logits, {
                "attention": patch_tokens.new_zeros(patch_tokens.shape[:2]),
                "descriptor": patch_tokens.new_zeros((patch_tokens.size(0), self.descriptor_dim)),
            }

        normalized = self.patch_norm(patch_tokens)
        left = F.normalize(F.gelu(self.left_proj(normalized)).float(), dim=-1, eps=1e-6)
        right = F.normalize(F.gelu(self.right_proj(normalized)).float(), dim=-1, eps=1e-6)
        weights = self._normalized_weights(patch_tokens, attention, valid_mask)
        left_mean = torch.einsum("bn,bnr->br", weights, left)
        right_mean = torch.einsum("bn,bnr->br", weights, right)
        bilinear = torch.einsum("bn,bnr,bns->brs", weights, left, right).flatten(1)
        bilinear = torch.sign(bilinear) * torch.sqrt(bilinear.abs() + 1e-8)
        bilinear = F.normalize(bilinear, dim=1, eps=1e-6)
        descriptor = torch.cat((left_mean, right_mean, bilinear), dim=1)
        logits = self.net(descriptor.to(dtype=patch_tokens.dtype))
        if not return_trace:
            return logits
        return logits, {
            "attention": weights.detach(),
            "descriptor": descriptor.detach(),
        }


class ColorStatisticFusion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hue_bins = 12
        self.color_bins = 8
        self.stats_dim = 47 + self.hue_bins + (self.color_bins * 6)
        self.max_stats_size = 56
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        hidden = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def _spatial_weights(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        y = torch.linspace(-1.0, 1.0, steps=max(1, height), device=device, dtype=dtype).view(1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, steps=max(1, width), device=device, dtype=dtype).view(1, 1, 1, width)
        center_raw = torch.exp(-((x * x) + (y * y)) / (2.0 * 0.45 * 0.45))
        center = center_raw / center_raw.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        border = (1.0 - center_raw).clamp(min=0.0)
        border = border / border.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return center, border

    def _weighted_mean_std(self, image: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
        mean = (image * weights).sum(dim=(-2, -1))
        variance = (((image - mean[:, :, None, None]) ** 2) * weights).sum(dim=(-2, -1))
        return mean, variance.clamp(min=1e-8).sqrt()

    def _rgb_to_hsv_maps(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]
        max_channel, _ = image.max(dim=1)
        min_channel, _ = image.min(dim=1)
        delta = max_channel - min_channel
        eps = 1e-6
        hue_red = ((green - blue) / delta.clamp(min=eps)) % 6.0
        hue_green = ((blue - red) / delta.clamp(min=eps)) + 2.0
        hue_blue = ((red - green) / delta.clamp(min=eps)) + 4.0
        hue = torch.where(
            max_channel == red,
            hue_red,
            torch.where(max_channel == green, hue_green, hue_blue),
        )
        hue = torch.where(delta > eps, hue / 6.0, torch.zeros_like(hue))
        saturation = torch.where(max_channel > eps, delta / max_channel.clamp(min=eps), torch.zeros_like(max_channel))
        value = max_channel
        return hue, saturation, value

    def _rgb_to_hsv_summary(self, image: Tensor, weights: Tensor) -> Tensor:
        hue, saturation, value = self._rgb_to_hsv_maps(image)
        hue_angle = hue * (2.0 * math.pi)
        sin_hue = torch.sin(hue_angle)
        cos_hue = torch.cos(hue_angle)
        return torch.stack(
            (
                (sin_hue[:, None] * weights).sum(dim=(-2, -1)).squeeze(1),
                (cos_hue[:, None] * weights).sum(dim=(-2, -1)).squeeze(1),
                (saturation[:, None] * weights).sum(dim=(-2, -1)).squeeze(1),
                (value[:, None] * weights).sum(dim=(-2, -1)).squeeze(1),
            ),
            dim=1,
        )

    def _soft_histogram(
        self,
        values: Tensor,
        weights: Tensor,
        *,
        bins: int,
        min_value: float,
        max_value: float,
    ) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Color histogram expects values shaped (B, H, W).")
        bin_count = max(2, int(bins))
        value_range = float(max_value) - float(min_value)
        centers = torch.linspace(
            float(min_value),
            float(max_value),
            steps=bin_count,
            device=values.device,
            dtype=values.dtype,
        ).view(1, bin_count, 1, 1)
        sigma = max(value_range / float(max(1, bin_count - 1)), 1e-6)
        activations = torch.exp(-0.5 * ((values[:, None] - centers) / sigma) ** 2)
        spatial_weights = weights.squeeze(1)[:, None]
        histogram = (activations * spatial_weights).sum(dim=(-2, -1))
        return histogram / histogram.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def _rgb_to_lab(self, image: Tensor) -> Tensor:
        rgb = image.permute(0, 2, 3, 1)
        linear = torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))
        matrix = image.new_tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
        )
        xyz = torch.matmul(linear, matrix.transpose(0, 1))
        white = image.new_tensor([0.95047, 1.0, 1.08883])
        xyz = xyz / white
        f_xyz = torch.where(xyz > 0.008856, torch.pow(xyz.clamp(min=1e-8), 1.0 / 3.0), 7.787 * xyz + 16.0 / 63.0)
        lab = torch.empty_like(f_xyz)
        lab[..., 0] = 116.0 * f_xyz[..., 1] - 16.0
        lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
        lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
        return lab.permute(0, 3, 1, 2)

    def extract_stats(self, image: Tensor) -> Tensor:
        image_float = (image.to(dtype=torch.float32) * self.rgb_std) + self.rgb_mean
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.max_stats_size:
            image_float = F.interpolate(
                image_float,
                size=(self.max_stats_size, self.max_stats_size),
                mode="area",
            )
        flat = image_float.flatten(2)
        global_mean = flat.mean(dim=-1)
        global_std = flat.std(dim=-1, unbiased=False)
        center_weights, border_weights = self._spatial_weights(
            height=int(image_float.shape[-2]),
            width=int(image_float.shape[-1]),
            device=image_float.device,
            dtype=image_float.dtype,
        )
        center_mean, center_std = self._weighted_mean_std(image_float, center_weights)
        border_mean, _ = self._weighted_mean_std(image_float, border_weights)
        center_minus_border = center_mean - border_mean
        global_weights = torch.full_like(center_weights, 1.0 / max(1, int(image_float.shape[-2]) * int(image_float.shape[-1])))
        global_hsv = self._rgb_to_hsv_summary(image_float, global_weights)
        center_hsv = self._rgb_to_hsv_summary(image_float, center_weights)
        hue, saturation, value = self._rgb_to_hsv_maps(image_float)
        lab = self._rgb_to_lab(image_float)
        lab_global_mean = lab.flatten(2).mean(dim=-1)
        lab_global_std = lab.flatten(2).std(dim=-1, unbiased=False)
        lab_center_mean, _ = self._weighted_mean_std(lab, center_weights)
        lab_border_mean, _ = self._weighted_mean_std(lab, border_weights)
        lab_center_minus_border = lab_center_mean - lab_border_mean
        lab_chroma = torch.sqrt((lab[:, 1] ** 2 + lab[:, 2] ** 2).clamp(min=0.0))
        color_histograms = torch.cat(
            (
                self._soft_histogram(hue, global_weights, bins=self.hue_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(saturation, global_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(value, global_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(lab[:, 0], global_weights, bins=self.color_bins, min_value=0.0, max_value=100.0),
                self._soft_histogram(lab[:, 1], global_weights, bins=self.color_bins, min_value=-128.0, max_value=128.0),
                self._soft_histogram(lab[:, 2], global_weights, bins=self.color_bins, min_value=-128.0, max_value=128.0),
                self._soft_histogram(lab_chroma, global_weights, bins=self.color_bins, min_value=0.0, max_value=180.0),
            ),
            dim=1,
        )
        channel_deltas = torch.stack(
            (
                global_mean[:, 0] - global_mean[:, 1],
                global_mean[:, 1] - global_mean[:, 2],
                global_mean[:, 0] - global_mean[:, 2],
            ),
            dim=1,
        )
        center_deltas = torch.stack(
            (
                center_mean[:, 0] - center_mean[:, 1],
                center_mean[:, 1] - center_mean[:, 2],
                center_mean[:, 0] - center_mean[:, 2],
            ),
            dim=1,
        )
        stats = torch.cat(
            (
                global_mean,
                global_std,
                center_mean,
                center_std,
                border_mean,
                center_minus_border,
                channel_deltas,
                center_deltas,
                global_hsv,
                center_hsv,
                lab_global_mean,
                lab_global_std,
                lab_center_mean,
                lab_border_mean,
                lab_center_minus_border,
                color_histograms,
            ),
            dim=1,
        )
        return stats

    def forward(self, image: Tensor) -> Tensor:
        stats = self.extract_stats(image)
        return self.net(stats.to(dtype=image.dtype))


class DefectStatisticFusion(ColorStatisticFusion):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.hist_bins = 8
        self.stats_dim = 90
        self.max_stats_size = 64
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        hidden = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )

    @staticmethod
    def _circular_hue_distance(hue: Tensor, center: float) -> Tensor:
        distance = (hue - float(center)).abs()
        return torch.minimum(distance, 1.0 - distance)

    @staticmethod
    def _normalize_map(values: Tensor) -> Tensor:
        flat = values.flatten(1)
        minimum = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
        maximum = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
        return (values - minimum) / (maximum - minimum).clamp(min=1e-6)

    def _weighted_scalar_stats(self, values: Tensor, center_weights: Tensor, border_weights: Tensor) -> Tensor:
        global_mean = values.flatten(1).mean(dim=1)
        center_mean = (values[:, None] * center_weights).sum(dim=(-2, -1)).squeeze(1)
        border_mean = (values[:, None] * border_weights).sum(dim=(-2, -1)).squeeze(1)
        maximum = values.flatten(1).amax(dim=1)
        flat = values.flatten(1)
        top_k = max(1, int(math.ceil(float(flat.size(1)) * 0.05)))
        top_mean = torch.topk(flat, k=top_k, dim=1, largest=True, sorted=False).values.mean(dim=1)
        return torch.stack((global_mean, center_mean, border_mean, maximum, top_mean), dim=1)

    def _weighted_fraction_stats(self, values: Tensor, center_weights: Tensor, border_weights: Tensor, threshold: float = 0.5) -> Tensor:
        mask = (values > float(threshold)).to(dtype=values.dtype)
        global_fraction = mask.flatten(1).mean(dim=1)
        center_fraction = (mask[:, None] * center_weights).sum(dim=(-2, -1)).squeeze(1)
        border_fraction = (mask[:, None] * border_weights).sum(dim=(-2, -1)).squeeze(1)
        return torch.stack((global_fraction, center_fraction, border_fraction), dim=1)

    def extract_stats(self, image: Tensor) -> Tensor:
        image_float = (image.to(dtype=torch.float32) * self.rgb_std) + self.rgb_mean
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.max_stats_size:
            image_float = F.interpolate(
                image_float,
                size=(self.max_stats_size, self.max_stats_size),
                mode="area",
            )

        hue, saturation, value = self._rgb_to_hsv_maps(image_float)
        luminance = (
            0.299 * image_float[:, 0]
            + 0.587 * image_float[:, 1]
            + 0.114 * image_float[:, 2]
        )
        local_mean = F.avg_pool2d(luminance[:, None], kernel_size=5, stride=1, padding=2).squeeze(1)
        local_contrast = self._normalize_map((luminance - local_mean).abs())
        grad_x = F.pad((luminance[:, :, 1:] - luminance[:, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((luminance[:, 1:, :] - luminance[:, :-1, :]).abs(), (0, 0, 0, 1))
        edge_detail = self._normalize_map(local_contrast + 0.5 * (grad_x + grad_y))

        dark_spot = torch.sigmoid((0.38 - value) * 16.0) * torch.sigmoid((saturation - 0.10) * 12.0)
        brown_hue = torch.exp(-0.5 * (self._circular_hue_distance(hue, 0.10) / 0.09).pow(2))
        brown_spot = brown_hue * torch.sigmoid((0.65 - value) * 10.0) * torch.sigmoid((saturation - 0.12) * 10.0)
        bright_spot = torch.sigmoid((value - 0.88) * 14.0) * torch.sigmoid((0.24 - saturation) * 10.0)

        center_weights, border_weights = self._spatial_weights(
            height=int(image_float.shape[-2]),
            width=int(image_float.shape[-1]),
            device=image_float.device,
            dtype=image_float.dtype,
        )
        global_weights = torch.full_like(
            center_weights,
            1.0 / max(1, int(image_float.shape[-2]) * int(image_float.shape[-1])),
        )

        score_stats = torch.cat(
            [
                self._weighted_scalar_stats(score, center_weights, border_weights)
                for score in (dark_spot, brown_spot, bright_spot, local_contrast, edge_detail)
            ],
            dim=1,
        )
        fraction_stats = torch.cat(
            [
                self._weighted_fraction_stats(score, center_weights, border_weights, threshold=0.5)
                for score in (dark_spot, brown_spot, bright_spot)
            ],
            dim=1,
        )
        hsv_stats = torch.cat(
            (
                self._rgb_to_hsv_summary(image_float, global_weights),
                self._rgb_to_hsv_summary(image_float, center_weights),
            ),
            dim=1,
        )
        histograms = torch.cat(
            (
                self._soft_histogram(dark_spot, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(brown_spot, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(bright_spot, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(local_contrast, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(value, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(saturation, global_weights, bins=self.hist_bins, min_value=0.0, max_value=1.0),
            ),
            dim=1,
        )
        stats = torch.cat((score_stats, fraction_stats, hsv_stats, histograms), dim=1)
        if int(stats.size(1)) != self.stats_dim:
            raise RuntimeError(f"DefectStatisticFusion stats_dim mismatch: {int(stats.size(1))} != {self.stats_dim}")
        return stats


class ForegroundSurfaceStatisticFusion(DefectStatisticFusion):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.hue_bins = 12
        self.color_bins = 8
        self.stats_dim = 129
        self.max_stats_size = 64
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        hidden = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )

    @staticmethod
    def _normalize_weights(weights: Tensor) -> Tensor:
        return weights / weights.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

    def _foreground_weight_maps(
        self,
        image_float: Tensor,
        hue: Tensor,
        saturation: Tensor,
        value: Tensor,
        edge_detail: Tensor,
        dark_spot: Tensor,
    ) -> Dict[str, Tensor]:
        fill_distance = (image_float - self.rgb_mean.to(device=image_float.device, dtype=image_float.dtype)).abs().mean(dim=1)
        non_padding = (fill_distance > 0.035).to(dtype=image_float.dtype)
        green_yellow = (
            (hue >= 0.08)
            & (hue <= 0.45)
            & (saturation >= 0.07)
            & (value >= 0.12)
        ).to(dtype=image_float.dtype)
        yellow_band = (
            (hue >= 0.10)
            & (hue <= 0.24)
            & (saturation >= 0.08)
            & (value >= 0.16)
        ).to(dtype=image_float.dtype)
        brown_or_orange = (
            (hue >= 0.035)
            & (hue <= 0.17)
            & (saturation >= 0.10)
            & (value >= 0.10)
        ).to(dtype=image_float.dtype)
        dark_defect = (dark_spot > 0.35).to(dtype=image_float.dtype)
        fruit_color = torch.maximum(torch.maximum(green_yellow, yellow_band), brown_or_orange)
        fruit_support = F.max_pool2d(
            fruit_color.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)

        height, width = int(image_float.shape[-2]), int(image_float.shape[-1])
        y = torch.linspace(-1.0, 1.0, height, device=image_float.device, dtype=image_float.dtype).view(1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=image_float.device, dtype=image_float.dtype).view(1, 1, width)
        center_prior = torch.exp(-((x * x) + (y * y)) / (2.0 * 0.70 * 0.70))
        foreground_raw = (
            fruit_color
            + 0.30 * edge_detail * fruit_support
            + 0.15 * dark_defect * fruit_support
            + 0.08 * center_prior * fruit_support
        ) * non_padding
        fallback_raw = (non_padding + 0.05 * center_prior * non_padding).clamp(min=0.0)
        use_fallback = foreground_raw.flatten(1).sum(dim=1).view(-1, 1, 1) <= 1e-6
        foreground_raw = torch.where(use_fallback, fallback_raw, foreground_raw)
        foreground_weights = self._normalize_weights(foreground_raw.unsqueeze(1))
        center_weights = self._normalize_weights(foreground_weights * center_prior.unsqueeze(1))
        border_weights = self._normalize_weights(foreground_weights * (1.0 - center_prior).clamp(min=0.0).unsqueeze(1))
        return {
            "foreground_raw": foreground_raw,
            "foreground_weights": foreground_weights,
            "center_weights": center_weights,
            "border_weights": border_weights,
            "center_prior": center_prior,
            "non_padding": non_padding,
            "green_yellow": green_yellow,
            "yellow_band": yellow_band,
            "brown_or_orange": brown_or_orange,
            "dark_defect": dark_defect,
        }

    def _weighted_channel_mean_std(self, image: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
        mean = (image * weights).sum(dim=(-2, -1))
        variance = (((image - mean[:, :, None, None]) ** 2) * weights).sum(dim=(-2, -1))
        return mean, variance.clamp(min=1e-8).sqrt()

    def _weighted_scalar_pack(self, values: Tensor, weights: Tensor, threshold: float = 0.5) -> Tensor:
        weight_map = weights.squeeze(1)
        weighted_mean = (values[:, None] * weights).sum(dim=(-2, -1)).squeeze(1)
        masked_values = torch.where(weight_map > 0.0, values, torch.zeros_like(values))
        maximum = masked_values.flatten(1).amax(dim=1)
        flat = masked_values.flatten(1)
        top_k = max(1, int(math.ceil(float(flat.size(1)) * 0.05)))
        top_mean = torch.topk(flat, k=top_k, dim=1, largest=True, sorted=False).values.mean(dim=1)
        fraction = (((values > float(threshold)).to(dtype=values.dtype))[:, None] * weights).sum(dim=(-2, -1)).squeeze(1)
        return torch.stack((weighted_mean, maximum, top_mean, fraction), dim=1)

    def _weighted_mask_fraction(self, mask: Tensor, weights: Tensor) -> Tensor:
        return (mask.to(dtype=weights.dtype)[:, None] * weights).sum(dim=(-2, -1)).squeeze(1)

    def _gray_world_normalize(self, image_float: Tensor, weights: Tensor) -> Tensor:
        mean_rgb, _ = self._weighted_channel_mean_std(image_float, weights)
        gray_level = mean_rgb.mean(dim=1, keepdim=True).clamp(min=0.08)
        scale = gray_level / mean_rgb.clamp(min=0.08)
        return (image_float * scale[:, :, None, None]).clamp(0.0, 1.0)

    def _extract_stats_and_maps(self, image: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_float = (image.to(dtype=torch.float32) * self.rgb_std) + self.rgb_mean
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.max_stats_size:
            image_float = F.interpolate(
                image_float,
                size=(self.max_stats_size, self.max_stats_size),
                mode="area",
            )

        hue, saturation, value = self._rgb_to_hsv_maps(image_float)
        luminance = (
            0.299 * image_float[:, 0]
            + 0.587 * image_float[:, 1]
            + 0.114 * image_float[:, 2]
        )
        local_mean = F.avg_pool2d(luminance[:, None], kernel_size=5, stride=1, padding=2).squeeze(1)
        local_contrast = self._normalize_map((luminance - local_mean).abs())
        grad_x = F.pad((luminance[:, :, 1:] - luminance[:, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((luminance[:, 1:, :] - luminance[:, :-1, :]).abs(), (0, 0, 0, 1))
        edge_detail = self._normalize_map(local_contrast + 0.5 * (grad_x + grad_y))

        dark_spot = torch.sigmoid((0.38 - value) * 16.0) * torch.sigmoid((saturation - 0.10) * 12.0)
        brown_hue = torch.exp(-0.5 * (self._circular_hue_distance(hue, 0.10) / 0.09).pow(2))
        brown_spot = brown_hue * torch.sigmoid((0.65 - value) * 10.0) * torch.sigmoid((saturation - 0.12) * 10.0)
        bright_spot = torch.sigmoid((value - 0.88) * 14.0) * torch.sigmoid((0.24 - saturation) * 10.0)
        maps = self._foreground_weight_maps(
            image_float=image_float,
            hue=hue,
            saturation=saturation,
            value=value,
            edge_detail=edge_detail,
            dark_spot=dark_spot,
        )
        foreground_weights = maps["foreground_weights"]
        center_weights = maps["center_weights"]
        border_weights = maps["border_weights"]
        normalized_image = self._gray_world_normalize(image_float, foreground_weights)
        hue_norm, saturation_norm, value_norm = self._rgb_to_hsv_maps(normalized_image)
        lab = self._rgb_to_lab(image_float)
        lab_norm = self._rgb_to_lab(normalized_image)

        raw_rgb_mean, raw_rgb_std = self._weighted_channel_mean_std(image_float, foreground_weights)
        norm_rgb_mean, norm_rgb_std = self._weighted_channel_mean_std(normalized_image, foreground_weights)
        raw_lab_mean, raw_lab_std = self._weighted_channel_mean_std(lab, foreground_weights)
        norm_lab_mean, norm_lab_std = self._weighted_channel_mean_std(lab_norm, foreground_weights)
        center_rgb_mean, _ = self._weighted_channel_mean_std(image_float, center_weights)
        border_rgb_mean, _ = self._weighted_channel_mean_std(image_float, border_weights)
        center_lab_mean, _ = self._weighted_channel_mean_std(lab, center_weights)
        border_lab_mean, _ = self._weighted_channel_mean_std(lab, border_weights)
        center_damage = torch.cat(
            [
                self._weighted_scalar_pack(score, center_weights, threshold=0.5)[:, :1]
                - self._weighted_scalar_pack(score, border_weights, threshold=0.5)[:, :1]
                for score in (dark_spot, brown_spot, bright_spot)
            ],
            dim=1,
        )

        foreground_area = (maps["foreground_raw"] > 0.05).to(dtype=image_float.dtype).flatten(1).mean(dim=1)
        area_stats = torch.stack(
            (
                maps["non_padding"].flatten(1).mean(dim=1),
                maps["foreground_raw"].flatten(1).mean(dim=1),
                foreground_area,
                (foreground_weights.squeeze(1) * maps["center_prior"]).sum(dim=(-2, -1)),
                (foreground_weights.squeeze(1) * (1.0 - maps["center_prior"]).clamp(min=0.0)).sum(dim=(-2, -1)),
            ),
            dim=1,
        )
        color_ratios = torch.stack(
            (
                self._weighted_mask_fraction(maps["green_yellow"], foreground_weights),
                self._weighted_mask_fraction(maps["yellow_band"], foreground_weights),
                self._weighted_mask_fraction(maps["brown_or_orange"], foreground_weights),
                self._weighted_mask_fraction(maps["dark_defect"], foreground_weights),
                self._weighted_mask_fraction(bright_spot > 0.5, foreground_weights),
                self._weighted_mask_fraction(value < 0.18, foreground_weights),
                self._weighted_mask_fraction(value > 0.90, foreground_weights),
            ),
            dim=1,
        )
        score_packs = torch.cat(
            [
                self._weighted_scalar_pack(score, foreground_weights, threshold=0.5)
                for score in (dark_spot, brown_spot, bright_spot, local_contrast, edge_detail)
            ],
            dim=1,
        )
        histograms = torch.cat(
            (
                self._soft_histogram(hue, foreground_weights, bins=self.hue_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(saturation, foreground_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(value, foreground_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(hue_norm, foreground_weights, bins=self.hue_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(saturation_norm, foreground_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
                self._soft_histogram(value_norm, foreground_weights, bins=self.color_bins, min_value=0.0, max_value=1.0),
            ),
            dim=1,
        )
        center_border_stats = torch.cat(
            (
                center_rgb_mean - border_rgb_mean,
                center_lab_mean - border_lab_mean,
                center_damage,
            ),
            dim=1,
        )
        stats = torch.cat(
            (
                area_stats,
                raw_rgb_mean,
                raw_rgb_std,
                norm_rgb_mean,
                norm_rgb_std,
                raw_lab_mean,
                raw_lab_std,
                norm_lab_mean,
                norm_lab_std,
                self._rgb_to_hsv_summary(image_float, foreground_weights),
                self._rgb_to_hsv_summary(normalized_image, foreground_weights),
                color_ratios,
                score_packs,
                histograms,
                center_border_stats,
            ),
            dim=1,
        )
        if int(stats.size(1)) != self.stats_dim:
            raise RuntimeError(
                f"ForegroundSurfaceStatisticFusion stats_dim mismatch: {int(stats.size(1))} != {self.stats_dim}"
            )
        maps.update(
            {
                "edge_detail": edge_detail,
                "dark_spot": dark_spot,
                "brown_spot": brown_spot,
                "bright_spot": bright_spot,
            }
        )
        return stats, maps

    def extract_stats(self, image: Tensor) -> Tensor:
        stats, _ = self._extract_stats_and_maps(image)
        return stats

    def forward(self, image: Tensor, return_trace: bool = False):
        stats, maps = self._extract_stats_and_maps(image)
        logits = self.net(stats.to(dtype=image.dtype))
        if not return_trace:
            return logits
        foreground_surface_mask = (maps["foreground_raw"] > 0.05).to(dtype=stats.dtype)
        return logits, {
            "stats": stats.detach(),
            "foreground_weight_map": maps["foreground_weights"].detach(),
            "foreground_mask": foreground_surface_mask.detach(),
            "edge_detail": (maps["edge_detail"] * foreground_surface_mask).detach(),
            "dark_spot": (maps["dark_spot"] * foreground_surface_mask).detach(),
            "brown_spot": (maps["brown_spot"] * foreground_surface_mask).detach(),
            "bright_spot": (maps["bright_spot"] * foreground_surface_mask).detach(),
        }


class ColorStatisticTokenBranch(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = max(0, int(num_tokens))
        self.stats = ColorStatisticFusion(num_classes=1, hidden_dim=hidden_dim, dropout=dropout)
        hidden = max(32, int(hidden_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(self.stats.stats_dim),
            nn.Linear(self.stats.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, self.num_tokens * int(embed_dim)),
        )
        self.embed_dim = int(embed_dim)

    def forward(self, image: Tensor) -> Tensor:
        if self.num_tokens <= 0:
            return image.new_zeros((image.shape[0], 0, self.embed_dim))
        stats = self.stats.extract_stats(image)
        tokens = self.proj(stats.to(dtype=image.dtype))
        return tokens.view(image.shape[0], self.num_tokens, self.embed_dim)


class EdgeStatisticTokenBranch(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 1,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = max(0, int(num_tokens))
        self.embed_dim = int(embed_dim)
        self.max_stats_size = 56
        self.stats_dim = 24
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        hidden = max(32, int(hidden_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, self.num_tokens * self.embed_dim),
        )

    def _spatial_weights(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        y = torch.linspace(-1.0, 1.0, steps=max(1, height), device=device, dtype=dtype).view(1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, steps=max(1, width), device=device, dtype=dtype).view(1, 1, 1, width)
        center_raw = torch.exp(-((x * x) + (y * y)) / (2.0 * 0.45 * 0.45))
        center = center_raw / center_raw.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        border = (1.0 - center_raw).clamp(min=0.0)
        border = border / border.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return center, border

    def forward(self, image: Tensor) -> Tensor:
        if self.num_tokens <= 0:
            return image.new_zeros((image.shape[0], 0, self.embed_dim))
        image_float = (image.to(dtype=torch.float32) * self.rgb_std) + self.rgb_mean
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.max_stats_size:
            image_float = F.interpolate(image_float, size=(self.max_stats_size, self.max_stats_size), mode="area")
        gray = (
            image_float[:, 0:1] * 0.299
            + image_float[:, 1:2] * 0.587
            + image_float[:, 2:3] * 0.114
        )
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt((edge_x * edge_x + edge_y * edge_y).clamp(min=1e-8))
        center, border = self._spatial_weights(
            height=int(magnitude.shape[-2]),
            width=int(magnitude.shape[-1]),
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        flat_edge = magnitude.flatten(1)
        flat_gray = gray.flatten(1)
        pooled = F.adaptive_avg_pool2d(magnitude, output_size=(4, 4)).flatten(1)
        stats = torch.cat(
            (
                flat_edge.mean(dim=1, keepdim=True),
                flat_edge.std(dim=1, unbiased=False, keepdim=True),
                flat_edge.amax(dim=1, keepdim=True),
                (magnitude * center).sum(dim=(-2, -1)).flatten(1),
                (magnitude * border).sum(dim=(-2, -1)).flatten(1),
                ((magnitude * center).sum(dim=(-2, -1)) - (magnitude * border).sum(dim=(-2, -1))).flatten(1),
                flat_gray.mean(dim=1, keepdim=True),
                flat_gray.std(dim=1, unbiased=False, keepdim=True),
                pooled,
            ),
            dim=1,
        )
        tokens = self.proj(stats.to(dtype=image.dtype))
        return tokens.view(image.shape[0], self.num_tokens, self.embed_dim)


class StemTokenBranch(nn.Module):
    def __init__(
        self,
        stem_dim: int,
        embed_dim: int,
        num_tokens: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = max(0, int(num_tokens))
        self.embed_dim = int(embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(int(stem_dim)),
            nn.Linear(int(stem_dim), int(embed_dim)),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(int(embed_dim), self.num_tokens * int(embed_dim)),
        )

    def forward(self, stem_features: Optional[Tensor], image: Tensor) -> Tensor:
        if self.num_tokens <= 0:
            return image.new_zeros((image.shape[0], 0, self.embed_dim))
        if stem_features is None or stem_features.ndim != 4:
            return image.new_zeros((image.shape[0], self.num_tokens, self.embed_dim))
        pooled = F.adaptive_avg_pool2d(stem_features, output_size=1).flatten(1)
        tokens = self.proj(pooled)
        return tokens.view(stem_features.shape[0], self.num_tokens, self.embed_dim)


class MultiBranchTokenFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        stem_dim: int,
        color_tokens: int = 1,
        edge_tokens: int = 1,
        cnn_tokens: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.color_tokens = max(0, int(color_tokens))
        self.edge_tokens = max(0, int(edge_tokens))
        self.cnn_tokens = max(0, int(cnn_tokens))
        self.token_count = self.color_tokens + self.edge_tokens + self.cnn_tokens
        self.color_branch = (
            ColorStatisticTokenBranch(embed_dim=embed_dim, num_tokens=self.color_tokens, dropout=dropout)
            if self.color_tokens > 0
            else None
        )
        self.edge_branch = (
            EdgeStatisticTokenBranch(embed_dim=embed_dim, num_tokens=self.edge_tokens, dropout=dropout)
            if self.edge_tokens > 0
            else None
        )
        self.cnn_branch = (
            StemTokenBranch(stem_dim=stem_dim, embed_dim=embed_dim, num_tokens=self.cnn_tokens, dropout=dropout)
            if self.cnn_tokens > 0
            else None
        )
        self.branch_type_embed = nn.Parameter(torch.zeros(1, self.token_count, self.embed_dim))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.token_count > 0:
            nn.init.trunc_normal_(self.branch_type_embed, std=0.02)

    def forward(self, image: Tensor, stem_features: Optional[Tensor]) -> Tensor:
        if self.token_count <= 0:
            return image.new_zeros((image.shape[0], 0, self.embed_dim))
        chunks: List[Tensor] = []
        if self.color_branch is not None:
            chunks.append(self.color_branch(image))
        if self.edge_branch is not None:
            chunks.append(self.edge_branch(image))
        if self.cnn_branch is not None:
            chunks.append(self.cnn_branch(stem_features, image))
        if not chunks:
            return image.new_zeros((image.shape[0], 0, self.embed_dim))
        tokens = torch.cat(chunks, dim=1)
        tokens = tokens + self.branch_type_embed.to(dtype=tokens.dtype)
        return self.dropout(self.norm(tokens))


class PatchDetailEnhancer(nn.Module):
    """Inject low-cost local color/edge residuals into spatial patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.proj = nn.Conv2d(7, int(embed_dim), kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(int(embed_dim))
        self.dropout = nn.Dropout(float(max(0.0, dropout)))
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        image: Tensor,
        grid_size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        rgb = image.to(dtype=torch.float32) * self.rgb_std + self.rgb_mean
        rgb = rgb.clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2)
        high_frequency = rgb - local_mean
        gray = rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
        grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        edge_magnitude = torch.sqrt((grad_x * grad_x + grad_y * grad_y).clamp(min=1e-8))
        detail_input = torch.cat(
            (
                high_frequency,
                high_frequency.abs(),
                edge_magnitude,
            ),
            dim=1,
        )
        detail_grid = F.adaptive_avg_pool2d(detail_input, output_size=grid_size)
        detail_tokens = self.proj(detail_grid).flatten(2).transpose(1, 2)
        detail_tokens = self.dropout(self.norm(detail_tokens))
        scale = self.residual_scale.to(dtype=detail_tokens.dtype).clamp(0.0, 1.0)
        detail_map = F.adaptive_avg_pool2d(
            high_frequency.abs().mean(dim=1, keepdim=True) + edge_magnitude,
            output_size=grid_size,
        )
        return detail_tokens * scale, detail_map


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim phai chia het cho num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor, return_attention: bool = False):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        out = attention @ value
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        out = self.projection_dropout(out)
        if return_attention:
            return out, attention
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attention = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            return x, attention
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


TransformerBlock = CustomTransformerEncoderLayer


class VisionTransformerWithRegisters(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cnn_stem: bool = True,
        stem_channels: int = 32,
        dual_patch_norm: bool = False,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        color_stat_fusion: bool = False,
        color_stat_fusion_dropout: float = 0.1,
        defect_stat_fusion: bool = False,
        defect_stat_fusion_dropout: float = 0.1,
        foreground_surface_fusion: bool = False,
        foreground_surface_fusion_dropout: float = 0.1,
        bilinear_patch_fusion: bool = False,
        bilinear_patch_rank: int = 32,
        bilinear_patch_dropout: float = 0.1,
        frequency_selective_pooling: bool = False,
        frequency_selective_top_k: int = 1,
        frequency_selective_blend: float = 1.0,
        frequency_selective_foreground_threshold: float = 0.35,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        register_positional_embedding: bool = False,
        fine_grained_pooling: bool = False,
        fine_grained_pooling_dropout: float = 0.1,
        multi_branch_fusion: bool = False,
        branch_color_tokens: int = 1,
        branch_edge_tokens: int = 1,
        branch_cnn_tokens: int = 1,
        branch_token_dropout: float = 0.1,
        detail_patch_enhancement: bool = False,
        detail_patch_dropout: float = 0.05,
        token_pruning: bool = False,
        token_prune_layers: str = "2,5",
        token_keep_rates: str = "0.75,0.50",
        token_prune_foreground_weight: float = 0.35,
        pairwise_margin_head: bool = False,
        pairwise_margin_pairs: str = "0-1,2-3,4-rest",
        pairwise_margin_logit_scale: float = 0.35,
        pairwise_margin_dropout: float = 0.05,
        pairwise_margin_routing: bool = False,
        pairwise_margin_route_max_probability_margin: float = 0.20,
        ordinal_maturity_head: bool = False,
        ordinal_maturity_classes: str = "0,1,2,3",
        ordinal_maturity_logit_scale: float = 0.20,
        ordinal_maturity_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        head_pooling: str = "cls_register_mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.register_positional_embedding = register_positional_embedding
        self.use_cnn_stem = use_cnn_stem
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed_dim = int(embed_dim)
        self.head_pooling = str(head_pooling).strip().lower()
        self.cnn_feature_fusion = bool(cnn_feature_fusion and use_cnn_stem)
        self.fine_grained_pooling = bool(fine_grained_pooling)
        self.color_stat_fusion = bool(color_stat_fusion)
        self.defect_stat_fusion = bool(defect_stat_fusion)
        self.foreground_surface_fusion = bool(foreground_surface_fusion)
        self.bilinear_patch_fusion = bool(bilinear_patch_fusion)
        self.frequency_selective_pooling = bool(frequency_selective_pooling)
        self.frequency_selective_blend = float(
            min(max(frequency_selective_blend, 0.0), 1.0)
        )
        self.detail_patch_enhancement = bool(detail_patch_enhancement)
        self.pairwise_margin_head_enabled = bool(pairwise_margin_head)
        self.pairwise_margin_logit_scale = float(max(0.0, pairwise_margin_logit_scale))
        self.pairwise_margin_routing = bool(pairwise_margin_routing)
        self.pairwise_margin_route_max_probability_margin = float(
            max(0.0, pairwise_margin_route_max_probability_margin)
        )
        self.pairwise_margin_pairs = (
            _parse_pairwise_margin_pairs(pairwise_margin_pairs, num_classes)
            if self.pairwise_margin_head_enabled
            else []
        )
        self.ordinal_maturity_head_enabled = bool(ordinal_maturity_head)
        self.ordinal_maturity_logit_scale = float(max(0.0, ordinal_maturity_logit_scale))
        self.ordinal_maturity_classes = (
            _parse_ordered_class_indices(ordinal_maturity_classes, num_classes)
            if self.ordinal_maturity_head_enabled
            else []
        )
        self.token_pruning = bool(token_pruning)
        self.token_prune_foreground_weight = float(max(0.0, token_prune_foreground_weight))
        self.token_prune_schedule = (
            self._parse_token_prune_schedule(
                depth=int(depth),
                layers=token_prune_layers,
                keep_rates=token_keep_rates,
            )
            if self.token_pruning
            else {}
        )
        if not self.token_prune_schedule:
            self.token_pruning = False
        if self.head_pooling not in {"cls", "cls_register_mean", "cls_branch_register_mean"}:
            raise ValueError(f"Khong ho tro head_pooling={head_pooling!r}.")

        if use_cnn_stem:
            self.stem = HybridConvStem(
                in_channels=in_channels,
                stem_channels=stem_channels,
                embed_dim=embed_dim,
            )
            stem_stride = self.stem.downsample_factor
            if image_size % stem_stride != 0:
                raise ValueError("image_size phai chia het cho downsample factor cua CNN stem")
            if patch_size % stem_stride != 0:
                raise ValueError("patch_size phai chia het cho downsample factor cua CNN stem")
            patch_embed_image_size = image_size // stem_stride
            patch_embed_patch_size = patch_size // stem_stride
            patch_embed_channels = self.stem.out_channels
        else:
            self.stem = nn.Identity()
            patch_embed_image_size = image_size
            patch_embed_patch_size = patch_size
            patch_embed_channels = in_channels

        self.patch_embed = PatchEmbedding(
            image_size=patch_embed_image_size,
            patch_size=patch_embed_patch_size,
            in_channels=patch_embed_channels,
            embed_dim=embed_dim,
            dual_patch_norm=dual_patch_norm,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        effective_branch_cnn_tokens = int(branch_cnn_tokens) if use_cnn_stem else 0
        self.branch_token_fusion = (
            MultiBranchTokenFusion(
                embed_dim=embed_dim,
                stem_dim=patch_embed_channels,
                color_tokens=int(branch_color_tokens),
                edge_tokens=int(branch_edge_tokens),
                cnn_tokens=effective_branch_cnn_tokens,
                dropout=branch_token_dropout,
            )
            if bool(multi_branch_fusion)
            and (
                int(branch_color_tokens) > 0
                or int(branch_edge_tokens) > 0
                or effective_branch_cnn_tokens > 0
            )
            else None
        )
        self.num_branch_tokens = int(getattr(self.branch_token_fusion, "token_count", 0))
        self.num_prefix_tokens = 1 + int(self.num_registers) + int(self.num_branch_tokens)
        self.detail_enhancer = (
            PatchDetailEnhancer(
                embed_dim=embed_dim,
                dropout=detail_patch_dropout,
            )
            if self.detail_patch_enhancement
            else None
        )

        pos_token_count = 1 + self.patch_embed.num_patches
        if register_positional_embedding:
            pos_token_count += num_registers
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_token_count, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_values[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if self.fine_grained_pooling:
            self.fine_grained_pool = FineGrainedPatchPooling(
                dim=embed_dim,
                dropout=fine_grained_pooling_dropout,
            )
        else:
            self.fine_grained_pool = None
        self.frequency_selective_pool = (
            FrequencySelectivePatchPooling(
                dim=embed_dim,
                top_k=frequency_selective_top_k,
                foreground_threshold=frequency_selective_foreground_threshold,
            )
            if self.frequency_selective_pooling
            else None
        )
        self.head = nn.Linear(embed_dim, num_classes)
        if self.pairwise_margin_pairs:
            self.pairwise_margin_norm = nn.LayerNorm(embed_dim)
            self.pairwise_margin_dropout = nn.Dropout(pairwise_margin_dropout)
            self.pairwise_margin_head = nn.Linear(embed_dim, len(self.pairwise_margin_pairs))
        else:
            self.pairwise_margin_norm = None
            self.pairwise_margin_dropout = None
            self.pairwise_margin_head = None
        if self.ordinal_maturity_classes:
            self.ordinal_maturity_norm = nn.LayerNorm(embed_dim)
            self.ordinal_maturity_dropout = nn.Dropout(ordinal_maturity_dropout)
            self.ordinal_maturity_head = nn.Linear(embed_dim, 1)
        else:
            self.ordinal_maturity_norm = None
            self.ordinal_maturity_dropout = None
            self.ordinal_maturity_head = None
        if self.cnn_feature_fusion:
            self.cnn_fusion_norm = nn.LayerNorm(embed_dim)
            self.cnn_fusion_dropout = nn.Dropout(float(max(0.0, cnn_fusion_dropout)))
            self.cnn_fusion_head = nn.Linear(embed_dim, num_classes)
        else:
            self.cnn_fusion_norm = nn.Identity()
            self.cnn_fusion_dropout = nn.Identity()
            self.cnn_fusion_head = None
        if self.color_stat_fusion:
            self.color_fusion_head = ColorStatisticFusion(
                num_classes=num_classes,
                hidden_dim=max(32, embed_dim // 2),
                dropout=color_stat_fusion_dropout,
            )
        else:
            self.color_fusion_head = None
        if self.defect_stat_fusion:
            self.defect_fusion_head = DefectStatisticFusion(
                num_classes=num_classes,
                hidden_dim=max(32, embed_dim // 2),
                dropout=defect_stat_fusion_dropout,
            )
        else:
            self.defect_fusion_head = None
        if self.foreground_surface_fusion:
            self.foreground_surface_fusion_head = ForegroundSurfaceStatisticFusion(
                num_classes=num_classes,
                hidden_dim=max(32, embed_dim // 2),
                dropout=foreground_surface_fusion_dropout,
            )
        else:
            self.foreground_surface_fusion_head = None
        if self.bilinear_patch_fusion:
            self.bilinear_patch_fusion_head = CompactBilinearPatchFusion(
                dim=embed_dim,
                num_classes=num_classes,
                rank=bilinear_patch_rank,
                hidden_dim=max(64, embed_dim // 2),
                dropout=bilinear_patch_dropout,
            )
        else:
            self.bilinear_patch_fusion_head = None

        self.apply(self._init_weights)
        self._init_parameter_tensors()
        if self.fine_grained_pool is not None:
            self.fine_grained_pool.zero_init_residual()
        if self.cnn_fusion_head is not None:
            nn.init.zeros_(self.cnn_fusion_head.weight)
            if self.cnn_fusion_head.bias is not None:
                nn.init.zeros_(self.cnn_fusion_head.bias)
        if self.color_fusion_head is not None:
            self.color_fusion_head.zero_init_residual()
        if self.defect_fusion_head is not None:
            self.defect_fusion_head.zero_init_residual()
        if self.foreground_surface_fusion_head is not None:
            self.foreground_surface_fusion_head.zero_init_residual()
        if self.bilinear_patch_fusion_head is not None:
            self.bilinear_patch_fusion_head.zero_init_residual()
        if self.pairwise_margin_head is not None:
            nn.init.zeros_(self.pairwise_margin_head.weight)
            if self.pairwise_margin_head.bias is not None:
                nn.init.zeros_(self.pairwise_margin_head.bias)
        if self.ordinal_maturity_head is not None:
            nn.init.zeros_(self.ordinal_maturity_head.weight)
            if self.ordinal_maturity_head.bias is not None:
                nn.init.zeros_(self.ordinal_maturity_head.bias)

    def _init_parameter_tensors(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _parse_csv_numbers(value: Any, cast) -> List[Any]:
        if isinstance(value, str):
            items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        elif isinstance(value, Sequence):
            items = list(value)
        else:
            items = [value]
        return [cast(item) for item in items]

    @classmethod
    def _parse_token_prune_schedule(
        cls,
        *,
        depth: int,
        layers: Any,
        keep_rates: Any,
    ) -> Dict[int, float]:
        parsed_layers = cls._parse_csv_numbers(layers, int)
        parsed_rates = cls._parse_csv_numbers(keep_rates, float)
        if len(parsed_layers) != len(parsed_rates):
            raise ValueError("token_prune_layers va token_keep_rates phai co cung so phan tu.")
        schedule: Dict[int, float] = {}
        previous_layer = 0
        previous_rate = 1.0
        for one_based_layer, keep_rate in zip(parsed_layers, parsed_rates):
            if one_based_layer < 1 or one_based_layer >= int(depth):
                raise ValueError("Moi token prune layer phai nam trong [1, depth - 1].")
            if one_based_layer <= previous_layer:
                raise ValueError("token_prune_layers phai tang dan va khong trung lap.")
            if not 0.0 < float(keep_rate) <= 1.0:
                raise ValueError("Moi token keep rate phai nam trong (0, 1].")
            if float(keep_rate) > previous_rate:
                raise ValueError("token_keep_rates phai giam dan theo chieu sau model.")
            schedule[int(one_based_layer) - 1] = float(keep_rate)
            previous_layer = int(one_based_layer)
            previous_rate = float(keep_rate)
        return schedule

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        return (
            "bias",
            "norm",
            "cls_token",
            "register_tokens",
            "pos_embed",
            "branch_type_embed",
            "residual_scale",
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _split_pos_embed(
        self,
        pos_embed: Tensor,
        has_register_positional_embedding: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cls_pos = pos_embed[:, :1]
        if has_register_positional_embedding:
            reg_pos = pos_embed[:, 1 : 1 + self.num_registers]
            patch_pos = pos_embed[:, 1 + self.num_registers :]
        else:
            reg_pos = pos_embed.new_zeros((pos_embed.shape[0], self.num_registers, pos_embed.shape[-1]))
            patch_pos = pos_embed[:, 1:]
        return cls_pos, reg_pos, patch_pos

    def _infer_source_register_mode(self, pos_embed: Tensor) -> Tuple[bool, int]:
        total_tokens = pos_embed.shape[1]
        patch_only_tokens = total_tokens - 1
        patch_with_register_tokens = total_tokens - 1 - self.num_registers

        patch_only_grid = int(round(math.sqrt(max(0, patch_only_tokens))))
        if patch_only_grid * patch_only_grid == patch_only_tokens:
            return False, patch_only_grid

        patch_with_register_grid = int(round(math.sqrt(max(0, patch_with_register_tokens))))
        if patch_with_register_grid * patch_with_register_grid == patch_with_register_tokens:
            return True, patch_with_register_grid

        raise ValueError("Khong the suy ra grid size tu pos_embed checkpoint.")

    def interpolate_external_pos_embed(self, pos_embed: Tensor) -> Tensor:
        source_has_register_pos, source_grid = self._infer_source_register_mode(pos_embed)
        cls_pos, reg_pos, patch_pos = self._split_pos_embed(
            pos_embed=pos_embed,
            has_register_positional_embedding=source_has_register_pos,
        )

        target_grid = self.patch_embed.base_grid_size
        if (source_grid, source_grid) != target_grid:
            patch_pos = patch_pos.reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos,
                size=target_grid,
                mode="bicubic",
                align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
                1,
                target_grid[0] * target_grid[1],
                -1,
            )

        if self.register_positional_embedding:
            if source_has_register_pos:
                target_reg_pos = reg_pos
            else:
                target_reg_pos = self.pos_embed[:, 1 : 1 + self.num_registers].detach().clone()
            return torch.cat((cls_pos, target_reg_pos, patch_pos), dim=1)

        return torch.cat((cls_pos, patch_pos), dim=1)

    def get_interpolated_pos_embed(self, grid_size: Tuple[int, int]) -> Tensor:
        cls_pos, reg_pos, patch_pos = self._split_pos_embed(
            pos_embed=self.pos_embed,
            has_register_positional_embedding=self.register_positional_embedding,
        )

        patch_pos = patch_pos.reshape(
            1,
            self.patch_embed.base_grid_size[0],
            self.patch_embed.base_grid_size[1],
            -1,
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1,
            grid_size[0] * grid_size[1],
            -1,
        )

        if self.register_positional_embedding:
            return torch.cat((cls_pos, reg_pos, patch_pos), dim=1)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def load_flexible_state_dict(self, state_dict, strict: bool = True):
        adapted_state = dict(state_dict)
        if "pos_embed" in adapted_state and adapted_state["pos_embed"].shape != self.pos_embed.shape:
            adapted_state["pos_embed"] = self.interpolate_external_pos_embed(adapted_state["pos_embed"])
        missing_keys, unexpected_keys = self.load_state_dict(adapted_state, strict=False)
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Loi load state_dict. missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return missing_keys, unexpected_keys

    @staticmethod
    def _normalize_token_scores(scores: Tensor) -> Tensor:
        minimum = scores.amin(dim=1, keepdim=True)
        maximum = scores.amax(dim=1, keepdim=True)
        return (scores - minimum) / (maximum - minimum).clamp(min=1e-6)

    def _build_patch_foreground_prior(
        self,
        image: Tensor,
        grid_size: Tuple[int, int],
        image_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        image_float = image.to(dtype=torch.float32)
        local_mean = F.avg_pool2d(image_float, kernel_size=7, stride=1, padding=3)
        local_contrast = (image_float - local_mean).abs().mean(dim=1, keepdim=True)
        gray = image_float.mean(dim=1, keepdim=True)
        grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        detail = local_contrast + 0.5 * (grad_x + grad_y)
        detail_prior = F.adaptive_avg_pool2d(detail, output_size=grid_size).flatten(1)

        rgb_mean = image_float.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        rgb_std = image_float.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (image_float * rgb_std + rgb_mean).clamp(0.0, 1.0)
        red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_channel, max_index = rgb.max(dim=1)
        min_channel, _ = rgb.min(dim=1)
        delta = max_channel - min_channel
        eps = 1e-6
        hue_red = torch.remainder((green - blue) / delta.clamp(min=eps), 6.0)
        hue_green = ((blue - red) / delta.clamp(min=eps)) + 2.0
        hue_blue = ((red - green) / delta.clamp(min=eps)) + 4.0
        hue = torch.where(
            max_index == 0,
            hue_red,
            torch.where(max_index == 1, hue_green, hue_blue),
        )
        hue = torch.where(delta > eps, hue / 6.0, torch.zeros_like(hue))
        saturation = torch.where(max_channel > eps, delta / max_channel.clamp(min=eps), torch.zeros_like(max_channel))
        fill_distance = (rgb - rgb_mean).abs().mean(dim=1)
        non_padding = (fill_distance > 0.035).to(dtype=image_float.dtype)
        green_yellow = (
            (hue >= 0.08)
            & (hue <= 0.45)
            & (saturation >= 0.07)
            & (max_channel >= 0.12)
        ).to(dtype=image_float.dtype)
        brown_or_orange = (
            (hue >= 0.035)
            & (hue <= 0.17)
            & (saturation >= 0.10)
            & (max_channel >= 0.10)
        ).to(dtype=image_float.dtype)
        dark_defect = (
            (max_channel <= 0.45)
            & (saturation >= 0.08)
            & (F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1).squeeze(1) > 0.025)
        ).to(dtype=image_float.dtype)
        color_prior_map = torch.maximum(torch.maximum(green_yellow, brown_or_orange), dark_defect) * non_padding
        color_prior = F.adaptive_avg_pool2d(
            color_prior_map.unsqueeze(1),
            output_size=grid_size,
        ).flatten(1)
        non_padding_prior = F.adaptive_avg_pool2d(
            non_padding.unsqueeze(1),
            output_size=grid_size,
        ).flatten(1)

        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        y = torch.linspace(-1.0, 1.0, grid_height, device=image.device, dtype=detail_prior.dtype)
        x = torch.linspace(-1.0, 1.0, grid_width, device=image.device, dtype=detail_prior.dtype)
        center_prior = torch.exp(
            -(
                y.view(grid_height, 1).pow(2)
                + x.view(1, grid_width).pow(2)
            )
            / (2.0 * 0.75 * 0.75)
        ).flatten()
        prior = (
            self._normalize_token_scores(detail_prior) * non_padding_prior
            + 0.35 * color_prior
            + 0.12 * center_prior.unsqueeze(0) * non_padding_prior
        )

        if image_valid_mask is not None:
            if image_valid_mask.ndim == 3:
                valid_mask = image_valid_mask.unsqueeze(1)
            elif image_valid_mask.ndim == 4:
                valid_mask = image_valid_mask
            else:
                raise ValueError("image_valid_mask phai co shape [B,H,W] hoac [B,1,H,W].")
            valid_fraction = F.interpolate(
                valid_mask.to(device=image.device, dtype=torch.float32),
                size=grid_size,
                mode="area",
            ).flatten(1)
            prior = prior * valid_fraction
        return self._normalize_token_scores(prior)

    def _prune_patch_tokens(
        self,
        *,
        tokens: Tensor,
        attention: Tensor,
        patch_indices: Tensor,
        foreground_prior: Tensor,
        original_patch_count: int,
        keep_rate: float,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        prefix_count = int(self.num_prefix_tokens)
        patch_tokens = tokens[:, prefix_count:]
        current_patch_count = int(patch_tokens.size(1))
        target_patch_count = max(1, int(math.ceil(float(original_patch_count) * float(keep_rate))))
        target_patch_count = min(current_patch_count, target_patch_count)
        if target_patch_count >= current_patch_count:
            empty = {
                "kept_indices": patch_indices,
                "scores": foreground_prior.gather(1, patch_indices),
            }
            return tokens, patch_indices, empty

        query_count = max(1, prefix_count)
        attention_score = attention[:, :, :query_count, prefix_count:].mean(dim=(1, 2))
        gathered_prior = foreground_prior.gather(1, patch_indices)
        score = self._normalize_token_scores(attention_score)
        if self.token_prune_foreground_weight > 0.0:
            score = score + self.token_prune_foreground_weight * self._normalize_token_scores(gathered_prior)

        selected_local = torch.topk(score, k=target_patch_count, dim=1, largest=True, sorted=False).indices
        selected_original = patch_indices.gather(1, selected_local)
        spatial_order = selected_original.argsort(dim=1)
        selected_local = selected_local.gather(1, spatial_order)
        selected_original = selected_original.gather(1, spatial_order)
        selected_tokens = patch_tokens.gather(
            1,
            selected_local.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1)),
        )
        selected_scores = score.gather(1, selected_local)
        pruned_tokens = torch.cat((tokens[:, :prefix_count], selected_tokens), dim=1)
        trace = {
            "kept_indices": selected_original,
            "scores": selected_scores,
        }
        return pruned_tokens, selected_original, trace

    def forward_features(
        self,
        x: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        attention_layers: Optional[Sequence[int]] = None,
        return_trace: bool = False,
    ) -> Dict[str, Tensor]:
        input_spatial_size = tuple(int(value) for value in x.shape[-2:])
        input_image = x
        x = self.stem(x)
        stem_features = x
        batch_size = x.shape[0]
        grid_size = (
            x.shape[-2] // self.patch_embed.patch_size,
            x.shape[-1] // self.patch_embed.patch_size,
        )
        patch_tokens = self.patch_embed(x)
        detail_map = None
        if self.detail_enhancer is not None:
            detail_tokens, detail_map = self.detail_enhancer(input_image, grid_size)
            patch_tokens = patch_tokens + detail_tokens.to(dtype=patch_tokens.dtype)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        branch_tokens = (
            self.branch_token_fusion(input_image, stem_features)
            if self.branch_token_fusion is not None
            else patch_tokens.new_zeros((batch_size, 0, patch_tokens.shape[-1]))
        )
        pos_embed = self.get_interpolated_pos_embed(grid_size).to(device=patch_tokens.device)

        if self.register_positional_embedding:
            positioned_tokens = torch.cat((cls_tokens, register_tokens, patch_tokens), dim=1)
            positioned_tokens = positioned_tokens + pos_embed
            tokens = torch.cat(
                (
                    positioned_tokens[:, : 1 + self.num_registers],
                    branch_tokens,
                    positioned_tokens[:, 1 + self.num_registers :],
                ),
                dim=1,
            )
        else:
            cls_and_patches = torch.cat((cls_tokens, patch_tokens), dim=1)
            cls_and_patches = cls_and_patches + pos_embed
            tokens = torch.cat(
                (cls_and_patches[:, :1], register_tokens, branch_tokens, cls_and_patches[:, 1:]),
                dim=1,
            )

        tokens = self.pos_drop(tokens)
        original_patch_count = int(patch_tokens.size(1))
        patch_indices = torch.arange(
            original_patch_count,
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        pruning_enabled = bool(self.token_pruning and not return_attention)
        foreground_prior = (
            self._build_patch_foreground_prior(
                input_image,
                grid_size=grid_size,
                image_valid_mask=image_valid_mask,
            )
            if pruning_enabled or return_trace or self.frequency_selective_pool is not None
            else patch_tokens.new_zeros((batch_size, original_patch_count), dtype=torch.float32)
        )
        collect_all_attentions = return_attention and attention_layers is None
        attention_layer_set = set(attention_layers or [])
        attention_maps = {}
        pruning_trace: List[Dict[str, Tensor]] = []
        block_token_shapes: List[Tuple[int, ...]] = []
        block_patch_indices: List[Tensor] = []
        block_patch_norms: List[Tensor] = []
        for block_index, block in enumerate(self.blocks):
            should_prune = pruning_enabled and block_index in self.token_prune_schedule
            if collect_all_attentions or block_index in attention_layer_set or should_prune:
                tokens, attention = block(tokens, return_attention=True)
                if collect_all_attentions or block_index in attention_layer_set:
                    attention_maps[block_index] = attention
                if should_prune:
                    before_count = int(tokens.size(1) - self.num_prefix_tokens)
                    tokens, patch_indices, prune_info = self._prune_patch_tokens(
                        tokens=tokens,
                        attention=attention,
                        patch_indices=patch_indices,
                        foreground_prior=foreground_prior,
                        original_patch_count=original_patch_count,
                        keep_rate=self.token_prune_schedule[block_index],
                    )
                    prune_info["layer"] = torch.tensor(
                        int(block_index + 1),
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    prune_info["before_count"] = torch.tensor(
                        before_count,
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    prune_info["after_count"] = torch.tensor(
                        int(tokens.size(1) - self.num_prefix_tokens),
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    pruning_trace.append(prune_info)
            else:
                if self.gradient_checkpointing and self.training:
                    tokens = gradient_checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = block(tokens)
            if return_trace:
                block_token_shapes.append(tuple(int(value) for value in tokens.shape))
                block_patch_indices.append(patch_indices.detach().clone())
                block_patch_norms.append(
                    tokens[:, self.num_prefix_tokens :].detach().float().norm(dim=-1)
                )
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        register_end = 1 + self.num_registers
        branch_end = register_end + self.num_branch_tokens
        reg_out = tokens[:, 1:register_end]
        branch_out = tokens[:, register_end:branch_end]
        patch_out = tokens[:, branch_end:]
        features = {
            "cls": cls_out,
            "registers": reg_out,
            "branch_tokens": branch_out,
            "patches": patch_out,
            "tokens": tokens,
            "grid_size": grid_size,
            "patch_indices": patch_indices,
            "pooled": self.pool_tokens_for_head(cls_out, reg_out, branch_out),
        }
        if pruning_enabled or return_trace:
            features["patch_keep_mask"] = F.one_hot(
                patch_indices,
                num_classes=original_patch_count,
            ).sum(dim=1).clamp(max=1).to(dtype=torch.bool)
        if self.frequency_selective_pool is not None:
            features["patch_foreground_prior"] = foreground_prior.gather(
                1,
                patch_indices,
            )
        if self.cnn_feature_fusion:
            features["cnn_pooled"] = F.adaptive_avg_pool2d(stem_features, output_size=1).flatten(1)
        if detail_map is not None:
            features["detail_map"] = detail_map
        if image_valid_mask is not None:
            key_padding_mask = self._build_patch_key_padding_mask(
                image_valid_mask=image_valid_mask,
                input_spatial_size=input_spatial_size,
                grid_size=grid_size,
                device=patch_out.device,
            )
            if patch_indices.size(1) != key_padding_mask.size(1):
                key_padding_mask = key_padding_mask.gather(1, patch_indices)
            features["memory_key_padding_mask"] = key_padding_mask
        if attention_maps:
            features["attentions"] = attention_maps
        if self.color_fusion_head is not None:
            features["color_logits"] = self.color_fusion_head(input_image)
        if self.defect_fusion_head is not None:
            features["defect_logits"] = self.defect_fusion_head(input_image)
        foreground_surface_trace = None
        if self.foreground_surface_fusion_head is not None:
            if return_trace:
                foreground_surface_logits, foreground_surface_trace = self.foreground_surface_fusion_head(
                    input_image,
                    return_trace=True,
                )
                features["foreground_surface_logits"] = foreground_surface_logits
                features["foreground_surface_trace"] = foreground_surface_trace
            else:
                features["foreground_surface_logits"] = self.foreground_surface_fusion_head(input_image)
        if return_trace:
            features["trace"] = {
                "input_shape": tuple(int(value) for value in input_image.shape),
                "stem_shape": tuple(int(value) for value in stem_features.shape),
                "patch_embedding_shape": (
                    int(batch_size),
                    int(original_patch_count),
                    int(patch_tokens.size(-1)),
                ),
                "branch_token_shape": tuple(int(value) for value in branch_tokens.shape),
                "block_token_shapes": block_token_shapes,
                "block_patch_indices": block_patch_indices,
                "block_patch_norms": block_patch_norms,
                "stem_activation": stem_features.detach().float().abs().mean(dim=1, keepdim=True),
                "patch_token_norm": patch_tokens.detach().float().norm(dim=-1),
                "detail_map": detail_map,
                "foreground_prior": foreground_prior,
                "pruning": pruning_trace,
            }
            if foreground_surface_trace is not None:
                features["trace"]["foreground_surface_stats"] = foreground_surface_trace["stats"]
                features["trace"]["foreground_surface_weight_map"] = foreground_surface_trace["foreground_weight_map"]
                features["trace"]["foreground_surface_mask"] = foreground_surface_trace["foreground_mask"]
                features["trace"]["foreground_surface_edge_detail"] = foreground_surface_trace["edge_detail"]
                features["trace"]["foreground_surface_dark_spot"] = foreground_surface_trace["dark_spot"]
                features["trace"]["foreground_surface_brown_spot"] = foreground_surface_trace["brown_spot"]
                features["trace"]["foreground_surface_bright_spot"] = foreground_surface_trace["bright_spot"]
        return features

    def _build_patch_key_padding_mask(
        self,
        image_valid_mask: Tensor,
        input_spatial_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        if image_valid_mask.ndim == 3:
            mask = image_valid_mask.unsqueeze(1)
        elif image_valid_mask.ndim == 4:
            mask = image_valid_mask
        else:
            raise ValueError("image_valid_mask phai co shape [B,H,W] hoac [B,1,H,W].")
        mask = mask.to(device=device, dtype=torch.float32)
        if tuple(mask.shape[-2:]) != tuple(input_spatial_size):
            mask = F.interpolate(mask, size=input_spatial_size, mode="nearest")
        valid_fraction = F.interpolate(
            mask,
            size=grid_size,
            mode="area",
        ).flatten(1)
        key_padding_mask = valid_fraction <= 0.05
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked] = False
        return key_padding_mask

    def pool_tokens_for_head(
        self,
        cls_tokens: Tensor,
        register_tokens: Tensor,
        branch_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        if self.head_pooling == "cls" or register_tokens.numel() == 0:
            return cls_tokens
        pooled_parts = [cls_tokens.unsqueeze(1), register_tokens]
        if (
            self.head_pooling == "cls_branch_register_mean"
            and branch_tokens is not None
            and branch_tokens.numel() > 0
        ):
            pooled_parts.append(branch_tokens)
        pooled_tokens = torch.cat(pooled_parts, dim=1)
        return pooled_tokens.mean(dim=1)

    def head_input_from_features(self, features: Dict[str, Tensor]) -> Tensor:
        if "pooled" in features:
            pooled = features["pooled"]
        else:
            pooled = self.pool_tokens_for_head(
                features["cls"],
                features["registers"],
                features.get("branch_tokens"),
            )
        valid_mask = None
        key_padding_mask = features.get("memory_key_padding_mask")
        if torch.is_tensor(key_padding_mask):
            valid_mask = ~key_padding_mask.to(dtype=torch.bool)
        if self.frequency_selective_pool is not None and "patches" in features:
            return_trace = isinstance(features.get("trace"), dict)
            selected_output = self.frequency_selective_pool(
                features["patches"],
                valid_mask=valid_mask,
                foreground_prior=features.get("patch_foreground_prior"),
                return_trace=return_trace,
            )
            if return_trace:
                selected_feature, selective_trace = selected_output
                features["trace"]["frequency_selective_vote_fraction"] = (
                    selective_trace["vote_fraction"]
                )
                features["trace"]["frequency_selective_selected_indices"] = (
                    selective_trace["selected_indices"]
                )
                features["trace"]["frequency_selective_candidate_mask"] = (
                    selective_trace["candidate_mask"]
                )
            else:
                selected_feature = selected_output
            features["frequency_selective_feature"] = selected_feature
            blend = float(self.frequency_selective_blend)
            pooled = pooled * (1.0 - blend) + selected_feature * blend
        if self.fine_grained_pool is not None and "patches" in features:
            patch_attention = self.fine_grained_pool.attention_weights(
                pooled,
                features["patches"],
                valid_mask=valid_mask,
            )
            features["fine_grained_attention"] = patch_attention
            pooled = self.fine_grained_pool(
                pooled,
                features["patches"],
                attention=patch_attention,
            )
        return pooled

    def pairwise_margin_logits_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.pairwise_margin_head is None:
            return None
        pairwise_input = self.pairwise_margin_norm(head_input)
        pairwise_input = self.pairwise_margin_dropout(pairwise_input)
        return self.pairwise_margin_head(pairwise_input)

    def pairwise_margin_route_weights(self, like_logits: Tensor) -> Tensor:
        pair_count = len(self.pairwise_margin_pairs)
        if like_logits.ndim != 2:
            batch_size = 0
        else:
            batch_size = int(like_logits.size(0))
        route_weights = like_logits.new_zeros((batch_size, pair_count))
        if pair_count == 0:
            return route_weights
        if not self.pairwise_margin_routing:
            return route_weights.fill_(1.0)
        if like_logits.ndim != 2 or like_logits.size(1) < 2:
            return route_weights

        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.pairwise_margin_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            for pair_index, (left_class, right_class) in enumerate(
                self.pairwise_margin_pairs
            ):
                if int(left_class) < 0:
                    route_match = (top1 == int(right_class)) | (
                        top2 == int(right_class)
                    )
                else:
                    route_match = (
                        (top1 == int(left_class)) & (top2 == int(right_class))
                    ) | (
                        (top1 == int(right_class)) & (top2 == int(left_class))
                    )
                route_weights[:, pair_index] = (
                    route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
                ).to(dtype=route_weights.dtype)
        return route_weights

    def pairwise_margin_adjustment(
        self,
        pairwise_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(pairwise_logits) or pairwise_logits.ndim != 2:
            if return_route_weights:
                return adjustment, like_logits.new_zeros(
                    (like_logits.size(0), len(self.pairwise_margin_pairs))
                )
            return adjustment
        scale = float(self.pairwise_margin_logit_scale)
        if scale <= 0.0:
            if return_route_weights:
                return adjustment, like_logits.new_zeros(
                    (like_logits.size(0), len(self.pairwise_margin_pairs))
                )
            return adjustment
        route_weights = self.pairwise_margin_route_weights(like_logits)
        class_count = int(like_logits.size(1))
        for pair_index, (left_class, right_class) in enumerate(self.pairwise_margin_pairs):
            if pair_index >= int(pairwise_logits.size(1)):
                break
            score = pairwise_logits[:, pair_index] * scale
            if pair_index < int(route_weights.size(1)):
                score = score * route_weights[:, pair_index]
            if int(left_class) < 0:
                positive_class = int(right_class)
                if 0 <= positive_class < class_count:
                    adjustment[:, positive_class] = adjustment[:, positive_class] + score
                    if class_count > 1:
                        rest_delta = score / float(class_count - 1)
                        for class_index in range(class_count):
                            if class_index != positive_class:
                                adjustment[:, class_index] = adjustment[:, class_index] - rest_delta
                continue
            if 0 <= int(left_class) < class_count:
                adjustment[:, int(left_class)] = adjustment[:, int(left_class)] - score
            if 0 <= int(right_class) < class_count:
                adjustment[:, int(right_class)] = adjustment[:, int(right_class)] + score
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def ordinal_maturity_score_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.ordinal_maturity_head is None:
            return None
        ordinal_input = self.ordinal_maturity_norm(head_input)
        ordinal_input = self.ordinal_maturity_dropout(ordinal_input)
        return self.ordinal_maturity_head(ordinal_input)

    def ordinal_maturity_adjustment(self, maturity_score: Tensor, like_logits: Tensor) -> Tensor:
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(maturity_score) or maturity_score.ndim != 2:
            return adjustment
        scale = float(self.ordinal_maturity_logit_scale)
        class_count = len(self.ordinal_maturity_classes)
        if scale <= 0.0 or class_count < 2:
            return adjustment
        centered_ranks = torch.arange(
            class_count,
            device=like_logits.device,
            dtype=like_logits.dtype,
        )
        centered_ranks = centered_ranks - centered_ranks.mean()
        score = maturity_score[:, 0].to(dtype=like_logits.dtype)
        for rank, class_index in enumerate(self.ordinal_maturity_classes):
            if 0 <= int(class_index) < int(like_logits.size(1)):
                adjustment[:, int(class_index)] = (
                    adjustment[:, int(class_index)]
                    + scale * score * centered_ranks[rank]
                )
        return adjustment

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return classification_logits_from_features(self, features)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP yeu cau num_layers >= 1.")
        layers = []
        in_dim = int(input_dim)
        for layer_index in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SinePositionEmbedding2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2.0 * math.pi,
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim cho positional embedding 2D phai chia het cho 4.")
        self.embed_dim = int(embed_dim)
        self.num_pos_feats = self.embed_dim // 2
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale)

    def forward(
        self,
        grid_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        y_embed = torch.arange(1, grid_height + 1, device=device, dtype=dtype).unsqueeze(1).repeat(1, grid_width)
        x_embed = torch.arange(1, grid_width + 1, device=device, dtype=dtype).unsqueeze(0).repeat(grid_height, 1)
        if self.normalize:
            y_embed = y_embed / max(1, grid_height) * self.scale
            x_embed = x_embed / max(1, grid_width) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2.0 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        position = torch.cat((pos_y, pos_x), dim=-1)
        return position.reshape(1, grid_height * grid_width, self.embed_dim)


class DETRDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        normalized_target = self.norm1(target)
        query = normalized_target + query_pos
        target2 = self.self_attn(query, query, value=normalized_target, need_weights=False)[0]
        target = target + self.dropout1(target2)

        normalized_target = self.norm2(target)
        target2 = self.cross_attn(
            query=normalized_target + query_pos,
            key=memory + memory_pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        target = target + self.dropout2(target2)

        normalized_target = self.norm3(target)
        target2 = self.linear2(self.dropout(F.gelu(self.linear1(normalized_target))))
        target = target + self.dropout3(target2)
        return target


class DETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DETRDecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_intermediate: bool = False,
    ):
        output = target
        intermediate_outputs: List[Tensor] = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                query_pos,
                memory_pos,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if return_intermediate:
                intermediate_outputs.append(self.norm(output))
        final_output = self.norm(output)
        if return_intermediate:
            if intermediate_outputs:
                intermediate_outputs[-1] = final_output
            return final_output, intermediate_outputs
        return final_output


class PatchMemoryAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.depthwise = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(max(0.0, dropout)))

    def forward(self, tokens: Tensor, grid_size: Tuple[int, int]) -> Tensor:
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        batch_size, token_count, embed_dim = tokens.shape
        if token_count != grid_height * grid_width:
            return tokens
        normalized = self.norm(tokens)
        grid = normalized.transpose(1, 2).reshape(batch_size, embed_dim, grid_height, grid_width)
        adapted = self.depthwise(grid)
        adapted = self.pointwise(self.act(adapted))
        adapted = adapted.flatten(2).transpose(1, 2)
        return tokens + self.dropout(adapted)


class DETRVisionTransformerWithRegisters(VisionTransformerWithRegisters):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cnn_stem: bool = True,
        stem_channels: int = 32,
        dual_patch_norm: bool = False,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        color_stat_fusion: bool = False,
        color_stat_fusion_dropout: float = 0.1,
        defect_stat_fusion: bool = False,
        defect_stat_fusion_dropout: float = 0.1,
        foreground_surface_fusion: bool = False,
        foreground_surface_fusion_dropout: float = 0.1,
        bilinear_patch_fusion: bool = False,
        bilinear_patch_rank: int = 32,
        bilinear_patch_dropout: float = 0.1,
        frequency_selective_pooling: bool = False,
        frequency_selective_top_k: int = 1,
        frequency_selective_blend: float = 1.0,
        frequency_selective_foreground_threshold: float = 0.35,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        register_positional_embedding: bool = False,
        fine_grained_pooling: bool = False,
        fine_grained_pooling_dropout: float = 0.1,
        multi_branch_fusion: bool = False,
        branch_color_tokens: int = 1,
        branch_edge_tokens: int = 1,
        branch_cnn_tokens: int = 1,
        branch_token_dropout: float = 0.1,
        detail_patch_enhancement: bool = False,
        detail_patch_dropout: float = 0.05,
        token_pruning: bool = False,
        token_prune_layers: str = "2,5",
        token_keep_rates: str = "0.75,0.50",
        token_prune_foreground_weight: float = 0.35,
        pairwise_margin_head: bool = False,
        pairwise_margin_pairs: str = "0-1,2-3,4-rest",
        pairwise_margin_logit_scale: float = 0.35,
        pairwise_margin_dropout: float = 0.05,
        pairwise_margin_routing: bool = False,
        pairwise_margin_route_max_probability_margin: float = 0.20,
        ordinal_maturity_head: bool = False,
        ordinal_maturity_classes: str = "0,1,2,3",
        ordinal_maturity_logit_scale: float = 0.20,
        ordinal_maturity_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        head_pooling: str = "cls_register_mean",
        bbox_head_hidden_dim: Optional[int] = None,
        num_queries: int = 40,
        decoder_depth: int = 4,
        decoder_num_heads: Optional[int] = None,
        decoder_ffn_dim: Optional[int] = None,
        decoder_dropout: Optional[float] = None,
        decoder_memory_adapter: bool = False,
        decoder_memory_adapter_dropout: float = 0.0,
        learned_query_content: bool = True,
        separate_objectness: bool = True,
        objectness_prior_prob: float = 0.125,
        quality_head: bool = False,
        quality_prior_prob: float = 0.125,
        auxiliary_decoder_outputs: bool = False,
        query_denoising_noise: float = 0.0,
        count_head: bool = False,
        count_head_hidden_dim: Optional[int] = None,
        count_head_dropout: float = 0.05,
        count_head_prior: float = 1.2,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            use_cnn_stem=use_cnn_stem,
            stem_channels=stem_channels,
            dual_patch_norm=dual_patch_norm,
            cnn_feature_fusion=cnn_feature_fusion,
            cnn_fusion_dropout=cnn_fusion_dropout,
            color_stat_fusion=color_stat_fusion,
            color_stat_fusion_dropout=color_stat_fusion_dropout,
            defect_stat_fusion=defect_stat_fusion,
            defect_stat_fusion_dropout=defect_stat_fusion_dropout,
            foreground_surface_fusion=foreground_surface_fusion,
            foreground_surface_fusion_dropout=foreground_surface_fusion_dropout,
            bilinear_patch_fusion=bilinear_patch_fusion,
            bilinear_patch_rank=bilinear_patch_rank,
            bilinear_patch_dropout=bilinear_patch_dropout,
            frequency_selective_pooling=frequency_selective_pooling,
            frequency_selective_top_k=frequency_selective_top_k,
            frequency_selective_blend=frequency_selective_blend,
            frequency_selective_foreground_threshold=frequency_selective_foreground_threshold,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_registers=num_registers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            register_positional_embedding=register_positional_embedding,
            fine_grained_pooling=fine_grained_pooling,
            fine_grained_pooling_dropout=fine_grained_pooling_dropout,
            multi_branch_fusion=multi_branch_fusion,
            branch_color_tokens=branch_color_tokens,
            branch_edge_tokens=branch_edge_tokens,
            branch_cnn_tokens=branch_cnn_tokens,
            branch_token_dropout=branch_token_dropout,
            detail_patch_enhancement=detail_patch_enhancement,
            detail_patch_dropout=detail_patch_dropout,
            token_pruning=token_pruning,
            token_prune_layers=token_prune_layers,
            token_keep_rates=token_keep_rates,
            token_prune_foreground_weight=token_prune_foreground_weight,
            pairwise_margin_head=pairwise_margin_head,
            pairwise_margin_pairs=pairwise_margin_pairs,
            pairwise_margin_logit_scale=pairwise_margin_logit_scale,
            pairwise_margin_dropout=pairwise_margin_dropout,
            pairwise_margin_routing=pairwise_margin_routing,
            pairwise_margin_route_max_probability_margin=(
                pairwise_margin_route_max_probability_margin
            ),
            ordinal_maturity_head=ordinal_maturity_head,
            ordinal_maturity_classes=ordinal_maturity_classes,
            ordinal_maturity_logit_scale=ordinal_maturity_logit_scale,
            ordinal_maturity_dropout=ordinal_maturity_dropout,
            gradient_checkpointing=gradient_checkpointing,
            head_pooling=head_pooling,
        )
        hidden_dim = int(bbox_head_hidden_dim or max(64, embed_dim * 2))
        self.num_queries = max(1, int(num_queries))
        self.decoder_depth = max(1, int(decoder_depth))
        self.decoder_num_heads = max(1, int(decoder_num_heads or num_heads))
        self.decoder_ffn_dim = int(decoder_ffn_dim or max(embed_dim * 4, 256))
        self.decoder_dropout = float(dropout if decoder_dropout is None else decoder_dropout)
        self.decoder_memory_adapter_enabled = bool(decoder_memory_adapter)
        self.learned_query_content = bool(learned_query_content)
        self.separate_objectness = bool(separate_objectness)
        self.objectness_prior_prob = float(min(max(objectness_prior_prob, 1e-4), 1.0 - 1e-4))
        self.quality_head_enabled = bool(quality_head)
        self.quality_prior_prob = float(min(max(quality_prior_prob, 1e-4), 1.0 - 1e-4))
        self.auxiliary_decoder_outputs = bool(auxiliary_decoder_outputs)
        self.query_denoising_noise = float(max(0.0, query_denoising_noise))
        self.count_head_enabled = bool(count_head)
        self.count_head_prior = float(max(1e-4, count_head_prior))
        self.is_hybrid_model = True
        self.is_detr_model = True
        self.head = nn.Identity()
        self.query_embed = nn.Embedding(self.num_queries, embed_dim)
        self.query_content_embed = nn.Embedding(self.num_queries, embed_dim) if self.learned_query_content else None
        self.memory_position_embedding = SinePositionEmbedding2D(embed_dim)
        self.memory_adapter = (
            PatchMemoryAdapter(embed_dim=embed_dim, dropout=decoder_memory_adapter_dropout)
            if self.decoder_memory_adapter_enabled
            else nn.Identity()
        )
        self.decoder = DETRTransformerDecoder(
            embed_dim=embed_dim,
            num_layers=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            ffn_dim=self.decoder_ffn_dim,
            dropout=self.decoder_dropout,
        )
        self.classification_head = nn.Linear(
            embed_dim,
            num_classes if self.separate_objectness else num_classes + 1,
        )
        self.objectness_head = nn.Linear(embed_dim, 1) if self.separate_objectness else None
        self.quality_head = nn.Linear(embed_dim, 1) if self.quality_head_enabled else None
        self.bbox_head = MLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            dropout=self.decoder_dropout,
        )
        count_hidden_dim = int(count_head_hidden_dim or max(64, embed_dim))
        self.count_head = (
            MLP(
                input_dim=embed_dim,
                hidden_dim=count_hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=float(max(0.0, count_head_dropout)),
            )
            if self.count_head_enabled
            else None
        )
        self.memory_adapter.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.classification_head.apply(self._init_weights)
        if self.objectness_head is not None:
            self.objectness_head.apply(self._init_weights)
            prior_logit = math.log(self.objectness_prior_prob / (1.0 - self.objectness_prior_prob))
            nn.init.constant_(self.objectness_head.bias, prior_logit)
        if self.quality_head is not None:
            self.quality_head.apply(self._init_weights)
            quality_prior_logit = math.log(self.quality_prior_prob / (1.0 - self.quality_prior_prob))
            nn.init.constant_(self.quality_head.bias, quality_prior_logit)
        self.bbox_head.apply(self._init_weights)
        if self.count_head is not None:
            self.count_head.apply(self._init_weights)
            last_linear = next(
                (module for module in reversed(list(self.count_head.modules())) if isinstance(module, nn.Linear)),
                None,
            )
            if last_linear is not None:
                prior_raw = math.log(math.expm1(self.count_head_prior)) if self.count_head_prior < 20.0 else self.count_head_prior
                nn.init.constant_(last_linear.bias, prior_raw)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        if self.query_content_embed is not None:
            nn.init.trunc_normal_(self.query_content_embed.weight, std=0.02)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        return super().no_weight_decay_keywords() + ("query_embed", "query_content_embed")

    def _prediction_heads_from_decoder_output(self, decoder_output: Tensor) -> Dict[str, Tensor]:
        output = {
            "logits": self.classification_head(decoder_output),
            "boxes": self.bbox_head(decoder_output).sigmoid(),
            "decoder_output": decoder_output,
        }
        if self.objectness_head is not None:
            output["objectness_logits"] = self.objectness_head(decoder_output).squeeze(-1)
        if self.quality_head is not None:
            output["quality_logits"] = self.quality_head(decoder_output).squeeze(-1)
        return output

    def forward_heads(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        memory = self.memory_adapter(features["patches"], features["grid_size"]) if self.decoder_memory_adapter_enabled else features["patches"]
        batch_size = memory.shape[0]
        memory_pos = self.memory_position_embedding(
            grid_size=features["grid_size"],
            device=memory.device,
            dtype=memory.dtype,
        ).expand(batch_size, -1, -1)
        patch_indices = features.get("patch_indices")
        if (
            torch.is_tensor(patch_indices)
            and patch_indices.ndim == 2
            and patch_indices.size(1) != memory_pos.size(1)
        ):
            memory_pos = memory_pos.gather(
                1,
                patch_indices.unsqueeze(-1).expand(-1, -1, memory_pos.size(-1)),
            )
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if self.query_content_embed is not None:
            decoder_input = self.query_content_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            decoder_input = torch.zeros_like(query_pos)
        if self.training and self.query_denoising_noise > 0.0:
            noise_scale = float(self.query_denoising_noise)
            query_pos = query_pos + torch.randn_like(query_pos) * noise_scale
            decoder_input = decoder_input + torch.randn_like(decoder_input) * noise_scale
        decoder_result = self.decoder(
            target=decoder_input,
            memory=memory,
            query_pos=query_pos,
            memory_pos=memory_pos,
            memory_key_padding_mask=features.get("memory_key_padding_mask"),
            return_intermediate=self.auxiliary_decoder_outputs,
        )
        if self.auxiliary_decoder_outputs:
            decoder_output, intermediate_outputs = decoder_result
        else:
            decoder_output = decoder_result
            intermediate_outputs = []
        output = self._prediction_heads_from_decoder_output(decoder_output)
        if self.auxiliary_decoder_outputs and len(intermediate_outputs) > 1:
            output["aux_outputs"] = [
                self._prediction_heads_from_decoder_output(aux_output)
                for aux_output in intermediate_outputs[:-1]
            ]
        if self.count_head is not None:
            count_input = self.head_input_from_features(features)
            output["count_logits"] = self.count_head(count_input).squeeze(-1)
        return output

    def forward(self, x: Tensor, image_valid_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        features = self.forward_features(x, image_valid_mask=image_valid_mask)
        return self.forward_heads(features)


def extract_head_input_from_features(model: nn.Module, features: Dict[str, Tensor]) -> Tensor:
    if hasattr(model, "head_input_from_features"):
        return model.head_input_from_features(features)
    if "pooled" in features:
        return features["pooled"]
    if "cls" in features:
        return features["cls"]
    raise KeyError("Khong tim thay feature dau vao cho classification head.")


def classification_logits_from_features(model: nn.Module, features: Dict[str, Tensor]) -> Tensor:
    head_input = extract_head_input_from_features(model, features)
    logits = model.head(head_input)
    fusion_head = getattr(model, "cnn_fusion_head", None)
    if fusion_head is not None and "cnn_pooled" in features:
        cnn_features = model.cnn_fusion_norm(features["cnn_pooled"])
        cnn_features = model.cnn_fusion_dropout(cnn_features)
        logits = logits + fusion_head(cnn_features)
    color_logits = features.get("color_logits")
    if torch.is_tensor(color_logits):
        logits = logits + color_logits
    defect_logits = features.get("defect_logits")
    if torch.is_tensor(defect_logits):
        logits = logits + defect_logits
    foreground_surface_logits = features.get("foreground_surface_logits")
    if torch.is_tensor(foreground_surface_logits):
        logits = logits + foreground_surface_logits
    bilinear_head = getattr(model, "bilinear_patch_fusion_head", None)
    if bilinear_head is not None and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        bilinear_output = bilinear_head(
            features["patches"],
            attention=features.get("fine_grained_attention"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            bilinear_logits, bilinear_trace = bilinear_output
            features["trace"]["bilinear_patch_attention"] = bilinear_trace["attention"]
            features["trace"]["bilinear_patch_descriptor"] = bilinear_trace["descriptor"]
        else:
            bilinear_logits = bilinear_output
        features["bilinear_patch_logits"] = bilinear_logits
        logits = logits + bilinear_logits
    pairwise_fn = getattr(model, "pairwise_margin_logits_from_head_input", None)
    adjust_fn = getattr(model, "pairwise_margin_adjustment", None)
    if callable(pairwise_fn) and callable(adjust_fn):
        pairwise_logits = pairwise_fn(head_input)
        if torch.is_tensor(pairwise_logits):
            features["pairwise_margin_logits"] = pairwise_logits
            if isinstance(features.get("trace"), dict):
                pairwise_adjustment, route_weights = adjust_fn(
                    pairwise_logits,
                    logits,
                    return_route_weights=True,
                )
                features["trace"]["pairwise_margin_route_weights"] = route_weights
            else:
                pairwise_adjustment = adjust_fn(pairwise_logits, logits)
            logits = logits + pairwise_adjustment
    ordinal_fn = getattr(model, "ordinal_maturity_score_from_head_input", None)
    ordinal_adjust_fn = getattr(model, "ordinal_maturity_adjustment", None)
    if callable(ordinal_fn) and callable(ordinal_adjust_fn):
        maturity_score = ordinal_fn(head_input)
        if torch.is_tensor(maturity_score):
            features["ordinal_maturity_score"] = maturity_score
            logits = logits + ordinal_adjust_fn(maturity_score, logits)
    return logits


def extract_bbox_from_model_output(model_output):
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        return logits, boxes
    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model output tuple/list rong.")
        if len(model_output) == 1:
            return model_output[0], None
        return model_output[0], model_output[1]
    return model_output, None


def extract_detection_from_model_output(model_output):
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        objectness_logits = model_output.get("objectness_logits")
        return logits, boxes, objectness_logits
    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model output tuple/list rong.")
        logits = model_output[0]
        boxes = model_output[1] if len(model_output) >= 2 else None
        objectness_logits = model_output[2] if len(model_output) >= 3 else None
        return logits, boxes, objectness_logits
    return model_output, None, None


def _infer_feature_dim(model: nn.Module) -> int:
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return int(model.head.in_features)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return int(model.fc.in_features)
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        return int(model.heads.head.in_features)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for module in reversed(model.classifier):
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Linear):
            return int(classifier.in_features)
    raise TypeError("Khong the suy ra kich thuoc embedding tu backbone.")


def _strip_classifier_for_temporal(model: nn.Module) -> Tuple[nn.Module, int]:
    feature_dim = _infer_feature_dim(model)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Identity()
        return model, feature_dim
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Identity()
        return model, feature_dim
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        model.heads.head = nn.Identity()
        return model, feature_dim
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for index in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[index], nn.Linear):
                model.classifier[index] = nn.Identity()
                return model, feature_dim
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
        return model, feature_dim
    raise TypeError("Khong the strip classifier de dung cho temporal wrapper.")


def _quantize_kv_tensor(tensor: Tensor, num_bits: int) -> Tuple[Tensor, Tensor]:
    quant_max = float((1 << (int(num_bits) - 1)) - 1)
    scale = tensor.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / quant_max
    quantized = torch.clamp(torch.round(tensor / scale), min=-quant_max, max=quant_max).to(torch.int8)
    return quantized, scale


def _dequantize_kv_tensor(quantized: Tensor, scale: Tensor) -> Tensor:
    return quantized.to(dtype=scale.dtype) * scale


class TemporalAttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        kv_quant_bits: int = 8,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Temporal embed_dim phai chia het cho temporal_num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_quant_bits = int(kv_quant_bits)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, tensor: Tensor) -> Tensor:
        batch_size, time_steps, dim = tensor.shape
        tensor = tensor.reshape(batch_size, time_steps, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def _maybe_quantize_cache(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        cache_info: Dict[str, object] = {
            "enabled": False,
            "quant_bits": int(self.kv_quant_bits),
            "cached_frames": max(0, key.shape[1] - 1),
            "mode": "disabled",
        }
        if self.training:
            cache_info["mode"] = "train_bypass"
            return key, value, cache_info
        if self.kv_quant_bits not in (4, 8) or key.shape[1] <= 1:
            cache_info["mode"] = "fp_cache"
            return key, value, cache_info

        past_key = key[:, :-1]
        past_value = value[:, :-1]
        quant_key, key_scale = _quantize_kv_tensor(past_key, self.kv_quant_bits)
        quant_value, value_scale = _quantize_kv_tensor(past_value, self.kv_quant_bits)
        cache_info.update(
            {
                "enabled": True,
                "mode": f"int{int(self.kv_quant_bits)}_kv_cache",
                "key_dtype": str(quant_key.dtype),
                "value_dtype": str(quant_value.dtype),
            }
        )
        restored_key = _dequantize_kv_tensor(quant_key, key_scale)
        restored_value = _dequantize_kv_tensor(quant_value, value_scale)
        key = torch.cat((restored_key, key[:, -1:]), dim=1)
        value = torch.cat((restored_value, value[:, -1:]), dim=1)
        return key, value, cache_info

    def forward(self, embeddings: Tensor) -> Dict[str, Tensor]:
        normalized = self.norm(embeddings)
        query = self.q_proj(normalized[:, -1:])
        key = self.k_proj(normalized)
        value = self.v_proj(normalized)
        key, value, cache_info = self._maybe_quantize_cache(key, value)

        query = self._reshape_heads(query)
        key = self._reshape_heads(key)
        value = self._reshape_heads(value)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        pooled = attention @ value
        pooled = pooled.transpose(1, 2).reshape(embeddings.size(0), 1, embeddings.size(-1))
        pooled = self.out_proj(pooled[:, 0])
        pooled = pooled + embeddings[:, -1]
        return {
            "pooled": pooled,
            "attention": attention[:, :, 0],
            "cache_enabled": torch.tensor(1 if cache_info["enabled"] else 0, device=embeddings.device),
            "cache_quant_bits": torch.tensor(int(cache_info["quant_bits"]), device=embeddings.device),
        }


class StreamingViT(nn.Module):
    def __init__(
        self,
        frame_model: nn.Module,
        num_classes: int,
        temporal_frames: int = 3,
        temporal_num_heads: int = 4,
        temporal_dropout: float = 0.1,
        temporal_kv_quant_bits: int = 8,
    ) -> None:
        super().__init__()
        frame_model, embed_dim = _strip_classifier_for_temporal(frame_model)
        self.frame_model = frame_model
        self.temporal_frames = max(1, int(temporal_frames))
        self.num_registers = int(getattr(frame_model, "num_registers", 0))
        self.temporal_pool = TemporalAttentionPool(
            dim=embed_dim,
            num_heads=temporal_num_heads,
            dropout=temporal_dropout,
            kv_quant_bits=temporal_kv_quant_bits,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.temporal_pool.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        if hasattr(self.frame_model, "no_weight_decay_keywords"):
            return tuple(self.frame_model.no_weight_decay_keywords()) + ("temporal_pool.norm", "bias")
        return ("bias", "norm")

    def _ensure_temporal_input(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x.unsqueeze(1).repeat(1, self.temporal_frames, 1, 1, 1)
        if x.ndim != 5:
            raise ValueError("StreamingViT yeu cau input (B, C, H, W) hoac (B, T, C, H, W).")
        if x.shape[1] == self.temporal_frames:
            return x
        if x.shape[1] > self.temporal_frames:
            return x[:, -self.temporal_frames :]
        pad_count = self.temporal_frames - x.shape[1]
        pad = x[:, :1].expand(-1, pad_count, -1, -1, -1)
        return torch.cat((pad, x), dim=1)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        x = self._ensure_temporal_input(x)
        batch_size, time_steps, channels, height, width = x.shape
        flat = x.reshape(batch_size * time_steps, channels, height, width)

        base_features: Optional[Dict[str, Tensor]] = None
        if hasattr(self.frame_model, "forward_features"):
            maybe_features = self.frame_model.forward_features(flat)
            if isinstance(maybe_features, dict) and "cls" in maybe_features:
                base_features = maybe_features
                if hasattr(self.frame_model, "head_input_from_features"):
                    frame_embeddings = self.frame_model.head_input_from_features(maybe_features)
                else:
                    frame_embeddings = maybe_features.get("pooled", maybe_features["cls"])
            else:
                frame_embeddings = self.frame_model(flat)
        else:
            frame_embeddings = self.frame_model(flat)

        frame_embeddings = frame_embeddings.reshape(batch_size, time_steps, -1)
        temporal_features = self.temporal_pool(frame_embeddings)
        features: Dict[str, Tensor] = {
            "cls": temporal_features["pooled"],
            "pooled": temporal_features["pooled"],
            "frame_cls": frame_embeddings,
            "temporal_attention": temporal_features["attention"],
            "temporal_cache_enabled": temporal_features["cache_enabled"],
            "temporal_cache_quant_bits": temporal_features["cache_quant_bits"],
        }

        if base_features is not None:
            if "patches" in base_features:
                patch_shape = base_features["patches"].shape[1:]
                features["patches"] = base_features["patches"].reshape(batch_size, time_steps, *patch_shape)
            if "registers" in base_features:
                register_shape = base_features["registers"].shape[1:]
                features["registers"] = base_features["registers"].reshape(
                    batch_size,
                    time_steps,
                    *register_shape,
                )
            if "tokens" in base_features:
                token_shape = base_features["tokens"].shape[1:]
                features["tokens"] = base_features["tokens"].reshape(batch_size, time_steps, *token_shape)
            if "grid_size" in base_features:
                features["grid_size"] = base_features["grid_size"]
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.head(features["pooled"])


def _config_to_dict(model_config: Any) -> Dict[str, Any]:
    if model_config is None:
        return {}
    if isinstance(model_config, dict):
        return dict(model_config)
    if hasattr(model_config, "__dict__"):
        return dict(vars(model_config))
    raise TypeError(f"Khong the chuyen model_config sang dict: {type(model_config)!r}")


def _replace_resnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("ResNet baseline khong co fc Linear nhu mong doi.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _replace_mobilenet_head(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Sequential):
        raise TypeError("MobileNet baseline khong co classifier nhu mong doi.")
    if not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("MobileNet classifier[-1] khong phai Linear.")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _replace_torchvision_vit_head(model: nn.Module, num_classes: int) -> nn.Module:
    heads = getattr(model, "heads", None)
    if not isinstance(heads, nn.Sequential):
        raise TypeError("Torchvision ViT baseline khong co heads Sequential nhu mong doi.")
    if "head" in heads._modules and isinstance(heads._modules["head"], nn.Linear):
        in_features = heads._modules["head"].in_features
        heads._modules["head"] = nn.Linear(in_features, num_classes)
        return model
    if len(heads) > 0 and isinstance(heads[-1], nn.Linear):
        in_features = heads[-1].in_features
        heads[-1] = nn.Linear(in_features, num_classes)
        return model
    raise TypeError("Torchvision ViT heads khong co Linear head nhu mong doi.")


def _torchvision_default_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    weights_cls = getattr(tv_models, name, None)
    if weights_cls is None or not hasattr(weights_cls, "DEFAULT"):
        raise ValueError(f"torchvision hien tai khong ho tro weights DEFAULT cho {name}.")
    return weights_cls.DEFAULT


def _build_torchvision_vit(
    num_classes: int,
    image_size: int,
    dropout: float,
    pretrained: bool = False,
) -> nn.Module:
    if pretrained and int(image_size) != 224:
        raise ValueError(
            "Pretrained torchvision vit_b_16 chi duoc ho tro voi --image-size 224 "
            "de tranh mismatch positional embedding."
        )
    weights = _torchvision_default_weights("ViT_B_16_Weights", pretrained)
    if weights is None:
        return tv_models.vit_b_16(
            weights=None,
            image_size=image_size,
            num_classes=num_classes,
            dropout=dropout,
        )
    model = tv_models.vit_b_16(
        weights=weights,
        image_size=image_size,
        dropout=dropout,
    )
    return _replace_torchvision_vit_head(model, num_classes)


def _build_resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    model = tv_models.resnet50(
        weights=_torchvision_default_weights("ResNet50_Weights", pretrained)
    )
    return _replace_resnet_head(model, num_classes)


def _build_mobilenet_v3_large(num_classes: int, pretrained: bool = False) -> nn.Module:
    model = tv_models.mobilenet_v3_large(
        weights=_torchvision_default_weights("MobileNet_V3_Large_Weights", pretrained)
    )
    return _replace_mobilenet_head(model, num_classes)


def _pop_pretraining_option(config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        return None
    return config.pop(key)


def _pretraining_option_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    return True


def _extract_pretrained_flag(config: Dict[str, Any]) -> bool:
    flag_keys = (
        "pretrained",
        "pretrain",
        "use_pretrained",
        "timm_pretrained",
    )
    flag_values = [
        _pop_pretraining_option(config, key)
        for key in flag_keys
        if key in config
    ]
    pretrained = any(_pretraining_option_enabled(value) for value in flag_values)
    blocked_keys = (
        "weights",
        "pretrained_weights",
        "weights_path",
        "weight_path",
        "pretrained_path",
        "init_checkpoint",
        "checkpoint_path",
        "external_checkpoint",
    )
    blocked_values = {
        key: _pop_pretraining_option(config, key)
        for key in blocked_keys
        if key in config
    }
    enabled_keys = [
        key for key, value in blocked_values.items() if _pretraining_option_enabled(value)
    ]
    if enabled_keys:
        raise ValueError(
            "pretrained/external weights: TRKH chi cho phep pretrained qua flag "
            "--pretrained cho backbone torchvision noi bo; "
            "khong nap external checkpoint/weights path de tranh leak/so sanh khong ro nguon. "
            f"Tham so bi chan: {', '.join(enabled_keys)}."
        )
    return pretrained


def create_model(
    num_classes: int,
    model_config: Optional[Any] = None,
    **overrides: Any,
) -> nn.Module:
    config = _config_to_dict(model_config)
    config.update(overrides)
    pretrained = _extract_pretrained_flag(config)

    temporal_frames = max(1, int(config.pop("temporal_frames", 1)))
    temporal_num_heads = max(1, int(config.pop("temporal_num_heads", 4)))
    temporal_dropout = float(config.pop("temporal_dropout", config.get("dropout", 0.1)))
    temporal_kv_quant_bits = int(config.pop("temporal_kv_quant_bits", 8))
    learned_query_content = bool(config.pop("learned_query_content", True))
    separate_objectness = bool(config.pop("separate_objectness", True))
    objectness_prior_prob = float(config.pop("objectness_prior_prob", 0.125))
    model_type = str(config.pop("model_type", "vit_registers_hybrid")).strip().lower()
    config.pop("timm_model_name", None)
    if pretrained and model_type not in {"resnet50", "mobilenet_v3_large", "vit_b_16"}:
        raise ValueError(
            "pretrained/external weights: --pretrained hien chi ho tro model_type "
            "resnet50, mobilenet_v3_large, vit_b_16. "
            "Cac kien truc TRKH custom vit_registers/vit_registers_hybrid van train tu dau; "
            "neu can pretrained cho custom TRKH thi phai them adapter/teacher distillation rieng."
        )

    if model_type == "vit_registers":
        for detector_only_key in (
            "bbox_head_hidden_dim",
            "num_queries",
            "decoder_depth",
            "decoder_num_heads",
            "decoder_ffn_dim",
            "decoder_dropout",
            "decoder_memory_adapter",
            "decoder_memory_adapter_dropout",
            "quality_head",
            "quality_prior_prob",
            "auxiliary_decoder_outputs",
            "query_denoising_noise",
            "count_head",
            "count_head_hidden_dim",
            "count_head_dropout",
            "count_head_prior",
        ):
            config.pop(detector_only_key, None)
        model = VisionTransformerWithRegisters(num_classes=num_classes, **config)
    elif model_type in {"detr_vit_registers", "vit_registers_hybrid"}:
        model = DETRVisionTransformerWithRegisters(
            num_classes=num_classes,
            learned_query_content=learned_query_content,
            separate_objectness=separate_objectness,
            objectness_prior_prob=objectness_prior_prob,
            **config,
        )
    elif model_type == "resnet50":
        model = _build_resnet50(num_classes, pretrained=pretrained)
    elif model_type == "mobilenet_v3_large":
        model = _build_mobilenet_v3_large(num_classes, pretrained=pretrained)
    elif model_type == "vit_b_16":
        image_size = int(config.get("image_size", 224))
        dropout = float(config.get("dropout", 0.0))
        model = _build_torchvision_vit(
            num_classes=num_classes,
            image_size=image_size,
            dropout=dropout,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Khong ho tro model_type: {model_type}.")

    model.model_type = model_type
    if temporal_frames > 1:
        if model_type in {"detr_vit_registers", "vit_registers_hybrid"}:
            raise ValueError(
                "DETR-ViT-Registers hien chi ho tro temporal_frames=1. "
                "Hay dung temporal smoothing o stream_infer.py cho video."
            )
        model = StreamingViT(
            frame_model=model,
            num_classes=num_classes,
            temporal_frames=temporal_frames,
            temporal_num_heads=temporal_num_heads,
            temporal_dropout=temporal_dropout,
            temporal_kv_quant_bits=temporal_kv_quant_bits,
        )
        model.model_type = f"streaming_{model_type}"
    model.temporal_frames = temporal_frames
    return model


def load_model_state(model: nn.Module, state_dict: Dict[str, Tensor], strict: bool = True):
    if hasattr(model, "load_flexible_state_dict"):
        return model.load_flexible_state_dict(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    num_classes: Optional[int] = None,
    override_image_size: Optional[int] = None,
) -> nn.Module:
    class_names = checkpoint.get("class_names", [])
    resolved_num_classes = int(num_classes or len(class_names))
    model_config = dict(checkpoint.get("model_config", {}))
    if "model_type" not in model_config:
        state_keys = checkpoint.get("model_state", {}).keys()
        model_config["model_type"] = (
            "vit_registers_hybrid"
            if any(
                str(key).startswith(prefix)
                for key in state_keys
                for prefix in ("bbox_head", "query_embed", "decoder", "classification_head")
            )
            else "vit_registers"
        )
    state_dict = checkpoint.get("model_state", {})
    state_keys = {str(key) for key in state_dict.keys()}
    legacy_classifier_hybrid = (
        model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"}
        and any(key.startswith("head.") for key in state_keys)
        and any(key.startswith("bbox_head.") for key in state_keys)
        and not any(
            key.startswith(prefix)
            for key in state_keys
            for prefix in (
                "query_embed.",
                "query_content_embed.",
                "decoder.",
                "objectness_head.",
            )
        )
    )
    if legacy_classifier_hybrid:
        model_config["model_type"] = "vit_registers"
    classification_weight = state_dict.get("classification_head.weight")
    has_objectness_head = any(str(key).startswith("objectness_head.") for key in state_dict.keys())
    has_query_content = any(str(key).startswith("query_content_embed.") for key in state_dict.keys())
    has_count_head = any(str(key).startswith("count_head.") for key in state_dict.keys())
    has_quality_head = any(str(key).startswith("quality_head.") for key in state_dict.keys())
    if (
        model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"}
        and torch.is_tensor(classification_weight)
        and int(classification_weight.shape[0]) == resolved_num_classes + 1
        and not has_objectness_head
    ):
        model_config["separate_objectness"] = False
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and not has_query_content:
        model_config["learned_query_content"] = False
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and has_count_head:
        model_config["count_head"] = True
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and has_quality_head:
        model_config["quality_head"] = True
    if override_image_size is not None:
        model_config["image_size"] = int(override_image_size)
    model = create_model(num_classes=resolved_num_classes, model_config=model_config)
    load_state = state_dict
    if legacy_classifier_hybrid:
        load_state = {
            key: value
            for key, value in state_dict.items()
            if not (
                str(key).startswith("bbox_head.")
                or str(key).startswith("classification_head.")
            )
        }
    load_model_state(model, load_state, strict=True)
    return model
