from __future__ import annotations

import copy
from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torchvision import models as tv_models

from trkh.models.inceptionnext_atto_tokenizer import InceptionNeXtAttoTokenizer
from trkh.models.learnable_gabor_texture import (
    LearnableGaborTextureEncoder,
    LearnableGaborTextureResidual,
)
from trkh.models.moga_surface_tokenizer import MogaXTTokenizer
from trkh.models.octave_conv_stem import OctaveConvStem
from trkh.models.patch_style_recalibration import PatchStyleRecalibration
from trkh.models.starnet_s2_tokenizer import StarNetS2Tokenizer
from trkh.models.visual_contrast_attention import VisualContrastAttention
from trkh.models.foveal_aggregated_attention import FovealAggregatedAttention
from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.models.bi_level_routing_attention import BiLevelRoutingAttention
from trkh.models.cropr_token_selector import CroprTokenSelector
from trkh.models.diverse_branch_stem import DiverseBranchConvStem
from trkh.models.cross_covariance_attention import (
    SharedProjectionCrossCovarianceAttention,
)
from trkh.models.dynamic_graph_mixer import MaxRelativeDynamicGraphMixer
from trkh.models.deep_class_prompt import DeepClassPrompt
from trkh.models.soft_moe_patch_adapter import SoftMoEPatchAdapter


def _adaptive_average_matrix(
    input_size: int,
    output_size: int,
) -> List[List[float]]:
    """Return the exact bin weights used by adaptive average pooling."""
    input_size = int(input_size)
    output_size = int(output_size)
    if input_size <= 0 or output_size <= 0:
        raise ValueError("Adaptive-pooling dimensions must be positive.")
    rows: List[List[float]] = []
    for output_index in range(output_size):
        start = math.floor(output_index * input_size / output_size)
        end = math.ceil((output_index + 1) * input_size / output_size)
        scale = 1.0 / float(end - start)
        row = [0.0] * input_size
        for input_index in range(start, end):
            row[input_index] = scale
        rows.append(row)
    return rows


def _onnx_exact_adaptive_avg_pool2d(
    inputs: Tensor,
    output_size: int | Tuple[int, int],
) -> Tensor:
    """Decompose exact fixed-shape adaptive pooling into ONNX MatMul ops."""
    if isinstance(output_size, int):
        output_height = output_width = int(output_size)
    else:
        output_height, output_width = [int(value) for value in output_size]
    input_height = int(inputs.shape[-2])
    input_width = int(inputs.shape[-1])
    height_weights = inputs.new_tensor(
        _adaptive_average_matrix(input_height, output_height)
    )
    width_weights = inputs.new_tensor(
        _adaptive_average_matrix(input_width, output_width)
    )
    pooled_width = torch.matmul(inputs, width_weights.transpose(0, 1))
    return torch.matmul(
        pooled_width.transpose(-2, -1),
        height_weights.transpose(0, 1),
    ).transpose(-2, -1)


def _exportable_adaptive_avg_pool2d(
    inputs: Tensor,
    output_size: int | Tuple[int, int],
) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        return _onnx_exact_adaptive_avg_pool2d(inputs, output_size)
    return F.adaptive_avg_pool2d(inputs, output_size=output_size)


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


def _parse_auxiliary_layer_indices(layers: str, depth: int) -> List[int]:
    parsed: List[int] = []
    for raw_item in str(layers or "").replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            layer_number = int(item)
        except ValueError as exc:
            raise ValueError(f"auxiliary layer khong hop le: {raw_item!r}") from exc
        if layer_number < 1 or layer_number > int(depth):
            raise ValueError(
                "auxiliary layer nam ngoai khoang transformer depth: "
                f"layer={layer_number}, depth={int(depth)}"
            )
        parsed.append(layer_number)
    return sorted(set(parsed))


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


class ShiftedPatchTokenResidual(nn.Module):
    """Checkpoint-safe SPT residual over a stem feature map."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        shift: int = 1,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.shift = int(shift)
        self.residual_scale = float(residual_scale)
        if self.shift <= 0:
            raise ValueError("shifted_patch_shift must be positive.")
        if self.residual_scale < 0.0:
            raise ValueError("shifted_patch_residual_scale must be non-negative.")
        self.proj = nn.Conv2d(
            int(in_channels) * 4,
            int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
            bias=True,
        )

    def zero_init_residual(self) -> None:
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def shifted_views(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError("ShiftedPatchTokenResidual expects [B, C, H, W].")
        shift = self.shift
        height, width = int(x.size(-2)), int(x.size(-1))
        padded = F.pad(x, (shift, shift, shift, shift))
        return (
            padded[:, :, :height, :width],
            padded[:, :, :height, 2 * shift : 2 * shift + width],
            padded[:, :, 2 * shift : 2 * shift + height, :width],
            padded[
                :,
                :,
                2 * shift : 2 * shift + height,
                2 * shift : 2 * shift + width,
            ],
        )

    def forward(self, x: Tensor) -> Tensor:
        shifted = torch.cat(self.shifted_views(x), dim=1)
        residual = self.proj(shifted).flatten(2).transpose(1, 2)
        return residual * self.residual_scale


class SoftPool2d(nn.Module):
    """Parameter-free exponentially weighted activation downsampling."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        if self.kernel_size <= 0 or self.stride <= 0:
            raise ValueError("SoftPool2d kernel_size/stride must be positive.")

    def forward(self, x: Tensor) -> Tensor:
        source_dtype = x.dtype
        values = x.float()
        weights = values.clamp(min=-20.0, max=20.0).exp()
        numerator = F.avg_pool2d(
            values * weights,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        denominator = F.avg_pool2d(
            weights,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        return (numerator / denominator.clamp_min(1e-12)).to(dtype=source_dtype)


class MaxSoftPool2d(nn.Module):
    """Conservative detail-preserving blend with a legacy max-pool anchor."""

    def __init__(self, kernel_size: int = 2, stride: int = 2, soft_blend: float = 0.15) -> None:
        super().__init__()
        self.soft_blend = float(soft_blend)
        if not 0.0 <= self.soft_blend <= 1.0:
            raise ValueError("stem_softpool_blend must be in [0, 1].")
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.soft_pool = SoftPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        max_values = self.max_pool(x)
        if self.soft_blend <= 0.0:
            return max_values
        soft_values = self.soft_pool(x)
        return torch.lerp(max_values, soft_values, self.soft_blend)


class InstanceBatchNorm2d(nn.BatchNorm2d):
    """IBN-a split with the legacy BatchNorm2d state-dict schema."""

    def __init__(self, num_features: int, ratio: float = 0.5, **kwargs: Any) -> None:
        super().__init__(num_features, **kwargs)
        self.instance_channels = int(num_features * float(ratio))
        if not 0 < self.instance_channels < int(num_features):
            raise ValueError("IBN-a requires non-empty instance and batch channel groups.")

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        batch_output = super().forward(x)
        instance_input = x[:, : self.instance_channels].contiguous()
        instance_output = F.instance_norm(
            instance_input,
            running_mean=None,
            running_var=None,
            weight=(
                self.weight[: self.instance_channels]
                if self.affine and self.weight is not None
                else None
            ),
            bias=(
                self.bias[: self.instance_channels]
                if self.affine and self.bias is not None
                else None
            ),
            use_input_stats=True,
            momentum=0.0,
            eps=self.eps,
        )
        return torch.cat(
            (instance_output, batch_output[:, self.instance_channels :]),
            dim=1,
        )


class ConvStemBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_mode: str = "max",
        softpool_blend: float = 0.15,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        normalized_pooling_mode = str(pooling_mode).strip().lower()
        if normalized_pooling_mode == "max":
            pooling: nn.Module = nn.MaxPool2d(kernel_size=2, stride=2)
        elif normalized_pooling_mode == "soft":
            pooling = SoftPool2d(kernel_size=2, stride=2)
        elif normalized_pooling_mode == "max_soft":
            pooling = MaxSoftPool2d(
                kernel_size=2,
                stride=2,
                soft_blend=softpool_blend,
            )
        else:
            raise ValueError(
                "stem_pooling_mode must be one of: max, soft, max_soft; "
                f"got {pooling_mode!r}."
            )
        normalized_normalization = str(normalization).strip().lower()
        if normalized_normalization == "batch":
            norm: nn.Module = nn.BatchNorm2d(out_channels)
        elif normalized_normalization == "ibn_a":
            norm = InstanceBatchNorm2d(out_channels, ratio=0.5)
        else:
            raise ValueError(
                "stem block normalization must be one of: batch, ibn_a; "
                f"got {normalization!r}."
            )
        self.pooling_mode = normalized_pooling_mode
        self.normalization = normalized_normalization
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
                    ("norm", norm),
                    ("act", nn.GELU()),
                    ("pool", pooling),
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
        pooling_mode: str = "max",
        softpool_blend: float = 0.15,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        normalized_normalization = str(normalization).strip().lower()
        if normalized_normalization not in {"batch", "ibn_a_first"}:
            raise ValueError(
                "stem_normalization must be one of: batch, ibn_a_first; "
                f"got {normalization!r}."
            )
        mid_channels = stem_channels * 2
        self.blocks = nn.Sequential(
            ConvStemBlock(
                in_channels,
                stem_channels,
                pooling_mode=pooling_mode,
                softpool_blend=softpool_blend,
                normalization=(
                    "ibn_a" if normalized_normalization == "ibn_a_first" else "batch"
                ),
            ),
            ConvStemBlock(
                stem_channels,
                mid_channels,
                pooling_mode=pooling_mode,
                softpool_blend=softpool_blend,
            ),
            ConvStemBlock(
                mid_channels,
                embed_dim,
                pooling_mode=pooling_mode,
                softpool_blend=softpool_blend,
            ),
        )
        self.pooling_mode = str(pooling_mode).strip().lower()
        self.softpool_blend = float(softpool_blend)
        self.normalization = normalized_normalization
        self.downsample_factor = 8
        self.out_channels = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class CoAtNetMBConvStem(nn.Module):
    """CoAtNet-Nano convolutional stages used as a scratch TRKH stem."""

    model_name = "coatnet_nano_rw_224"

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "stem_architecture=coatnet_mbconv requires the timm package."
            ) from exc

        source = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
        )
        if not hasattr(source, "stem") or not hasattr(source, "stages"):
            raise RuntimeError(
                f"Unexpected timm structure for {self.model_name!r}; stem/stages are required."
            )
        if len(source.stages) < 2:
            raise RuntimeError(
                f"Unexpected timm structure for {self.model_name!r}; two MBConv stages are required."
            )

        self.coatnet_stem = source.stem
        self.mbconv_stages = nn.Sequential(source.stages[0], source.stages[1])
        source_channels = int(source.feature_info[2]["num_chs"])
        self.projection = nn.Sequential(
            nn.Conv2d(source_channels, int(embed_dim), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(embed_dim)),
            nn.SiLU(inplace=True),
        )
        self.downsample_factor = 8
        self.out_channels = int(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.coatnet_stem(x)
        x = self.mbconv_stages(x)
        return self.projection(x)


class MixStyle(nn.Module):
    """Training-only feature-statistic mixing for domain/style robustness."""

    def __init__(
        self,
        probability: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.probability = float(min(max(probability, 0.0), 1.0))
        self.alpha = float(max(alpha, 1e-6))
        self.eps = float(max(eps, 1e-12))

    def forward(self, x: Tensor) -> Tensor:
        if (
            not self.training
            or self.probability <= 0.0
            or x.ndim != 4
            or int(x.size(0)) < 2
            or torch.rand(1, device=x.device).item() >= self.probability
        ):
            return x
        mean = x.mean(dim=(-2, -1), keepdim=True).detach()
        std = x.var(dim=(-2, -1), keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        normalized = (x - mean) / std

        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((int(x.size(0)), 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(int(x.size(0)), device=x.device)
        mixed_mean = lam * mean + (1.0 - lam) * mean[perm]
        mixed_std = lam * std + (1.0 - lam) * std[perm]
        return normalized * mixed_std + mixed_mean


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


class ResidualMLPClassificationHead(nn.Module):
    """Linear checkpoint-compatible classifier plus a zero-init residual MLP."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.08,
        residual_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.num_classes))
        hidden = max(self.num_classes, int(hidden_dim))
        drop = float(max(0.0, dropout))
        self.residual_norm = nn.LayerNorm(self.in_features)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.in_features, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.num_classes),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(max(0.0, residual_scale)), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def zero_init_residual(self) -> None:
        final = self.residual_mlp[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def forward(self, x: Tensor) -> Tensor:
        base_logits = F.linear(x, self.weight, self.bias)
        residual_logits = self.residual_mlp(self.residual_norm(x))
        return base_logits + residual_logits * self.residual_scale.to(dtype=x.dtype)


class SubCenterProxyHead(nn.Module):
    """Learnable class proxies with multiple sub-centers per class."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        subcenters: int = 3,
        dropout: float = 0.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.subcenters = max(1, int(subcenters))
        self.init_std = float(max(0.0, init_std))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))
        self.proxies = nn.Parameter(
            torch.empty(self.num_classes, self.subcenters, self.embed_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.proxies, std=max(1e-6, self.init_std))
        if self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if self.norm.bias is not None:
            nn.init.zeros_(self.norm.bias)

    def initialize_from_classifier(
        self,
        classifier: Optional[nn.Module],
        noise_std: float = 0.002,
    ) -> bool:
        weight = getattr(classifier, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            return False
        if tuple(weight.shape) != (self.num_classes, self.embed_dim):
            return False
        with torch.no_grad():
            base = weight.detach().to(device=self.proxies.device, dtype=self.proxies.dtype)
            expanded = base[:, None, :].expand(-1, self.subcenters, -1).clone()
            std = float(max(0.0, noise_std))
            if std > 0.0 and self.subcenters > 1:
                expanded[:, 1:, :] = expanded[:, 1:, :] + torch.randn_like(
                    expanded[:, 1:, :]
                ) * std
            self.proxies.copy_(expanded)
        return True

    def forward(self, head_input: Tensor) -> Tuple[Tensor, Tensor]:
        if head_input.ndim != 2 or int(head_input.size(1)) != self.embed_dim:
            raise ValueError("SubCenterProxyHead expects head_input [B, embed_dim].")
        features = F.normalize(self.dropout(self.norm(head_input)).float(), dim=1, eps=1e-6)
        proxies = F.normalize(self.proxies.float(), dim=2, eps=1e-6)
        all_scores = torch.einsum("bd,ckd->bck", features, proxies).to(dtype=head_input.dtype)
        logits = all_scores.amax(dim=2)
        return logits, all_scores


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


class ComplementaryPatchSuppressionHead(nn.Module):
    """Residual classifier over patch evidence left after suppressing top salient tokens."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        top_k: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        temperature: float = 0.20,
        suppression_strength: float = 0.85,
        bbox_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.top_k = max(1, int(top_k))
        self.temperature = max(1e-4, float(temperature))
        self.suppression_strength = float(min(1.0, max(0.0, suppression_strength)))
        self.bbox_weight = float(min(1.0, max(0.0, bbox_weight)))
        self.patch_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.descriptor_dim = (5 * int(dim)) + 5
        hidden = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, self.num_classes),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _valid_mask(patch_tokens: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        batch_size, token_count = patch_tokens.shape[:2]
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=patch_tokens.device, dtype=torch.bool).clone()
        else:
            valid = torch.ones(
                (batch_size, token_count),
                device=patch_tokens.device,
                dtype=torch.bool,
            )
        empty = ~valid.any(dim=1)
        if empty.any():
            valid[empty] = True
        return valid

    @staticmethod
    def _coerce_prior(prior: Optional[Tensor], patch_tokens: Tensor) -> Optional[Tensor]:
        if not torch.is_tensor(prior) or tuple(prior.shape) != tuple(patch_tokens.shape[:2]):
            return None
        prior_value = prior.to(device=patch_tokens.device, dtype=torch.float32)
        return prior_value.nan_to_num(0.0).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_weights(weights: Tensor, valid: Tensor) -> Tensor:
        weights = weights.to(dtype=torch.float32).masked_fill(~valid, 0.0)
        totals = weights.sum(dim=1, keepdim=True)
        empty = totals.squeeze(1) <= 1e-8
        if empty.any():
            fallback = valid.to(dtype=weights.dtype)
            weights = torch.where(empty[:, None], fallback, weights)
            totals = weights.sum(dim=1, keepdim=True)
        return weights / totals.clamp(min=1e-8)

    def forward(
        self,
        head_input: Tensor,
        patch_tokens: Tensor,
        bbox_prior: Optional[Tensor] = None,
        foreground_prior: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            logits = patch_tokens.new_zeros((head_input.size(0), self.num_classes))
            if not return_trace:
                return logits
            empty_attention = patch_tokens.new_zeros(patch_tokens.shape[:2])
            return logits, {
                "saliency": empty_attention,
                "suppressed_mask": empty_attention.to(dtype=torch.bool),
                "complement_attention": empty_attention,
                "salient_attention": empty_attention,
                "descriptor": patch_tokens.new_zeros((head_input.size(0), self.descriptor_dim)),
            }

        valid = self._valid_mask(patch_tokens, valid_mask)
        normalized_patches = self.patch_norm(patch_tokens)
        normalized_query = self.query_norm(head_input)
        saliency = torch.einsum(
            "bd,bnd->bn",
            F.normalize(normalized_query.float(), dim=-1, eps=1e-6),
            F.normalize(normalized_patches.float(), dim=-1, eps=1e-6),
        )
        bbox_value = self._coerce_prior(bbox_prior, patch_tokens)
        foreground_value = self._coerce_prior(foreground_prior, patch_tokens)
        if bbox_value is not None and self.bbox_weight > 0.0:
            saliency = saliency * (1.0 - self.bbox_weight) + bbox_value * self.bbox_weight
        saliency = saliency.masked_fill(~valid, torch.finfo(saliency.dtype).min)

        selected_count = min(self.top_k, int(patch_tokens.size(1)))
        selected_indices = torch.topk(
            saliency,
            k=selected_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        suppressed_mask = torch.zeros_like(valid)
        suppressed_mask.scatter_(1, selected_indices, True)
        suppressed_mask = suppressed_mask & valid

        base_weights = valid.to(dtype=torch.float32)
        if bbox_value is not None and self.bbox_weight > 0.0:
            base_weights = base_weights * (
                (1.0 - self.bbox_weight) + bbox_value * self.bbox_weight
            )
        if foreground_value is not None:
            base_weights = base_weights * (0.75 + 0.25 * foreground_value)
        global_weights = self._normalize_weights(base_weights, valid)
        salient_weights = self._normalize_weights(base_weights * suppressed_mask, suppressed_mask)
        complement_weights = base_weights * (
            1.0 - self.suppression_strength * suppressed_mask.to(dtype=torch.float32)
        )
        complement_weights = self._normalize_weights(complement_weights, valid)

        normalized_patches = normalized_patches.to(dtype=head_input.dtype)
        global_feature = torch.einsum("bn,bnd->bd", global_weights.to(normalized_patches.dtype), normalized_patches)
        salient_feature = torch.einsum("bn,bnd->bd", salient_weights.to(normalized_patches.dtype), normalized_patches)
        complement_feature = torch.einsum(
            "bn,bnd->bd",
            complement_weights.to(normalized_patches.dtype),
            normalized_patches,
        )
        contrast_feature = complement_feature - salient_feature

        safe_saliency = saliency.masked_fill(~valid, 0.0)
        valid_count = valid.sum(dim=1).to(dtype=torch.float32).clamp(min=1.0)
        max_saliency = saliency.masked_fill(~valid, -1e4).amax(dim=1)
        mean_saliency = safe_saliency.sum(dim=1) / valid_count
        suppressed_fraction = suppressed_mask.sum(dim=1).to(dtype=torch.float32) / valid_count
        suppressed_mass = (global_weights * suppressed_mask.to(dtype=global_weights.dtype)).sum(dim=1)
        entropy = -(
            complement_weights * (complement_weights.clamp(min=1e-8).log())
        ).sum(dim=1) / math.log(float(max(2, patch_tokens.size(1))))
        stats = torch.stack(
            (
                max_saliency,
                mean_saliency,
                suppressed_fraction,
                suppressed_mass,
                entropy,
            ),
            dim=1,
        ).to(dtype=head_input.dtype)
        descriptor = torch.cat(
            (
                normalized_query.to(dtype=head_input.dtype),
                global_feature,
                salient_feature,
                complement_feature,
                contrast_feature,
                stats,
            ),
            dim=1,
        )
        logits = self.net(descriptor)
        if not return_trace:
            return logits
        return logits, {
            "saliency": safe_saliency.detach(),
            "suppressed_mask": suppressed_mask.detach(),
            "suppressed_indices": selected_indices.detach(),
            "global_attention": global_weights.detach(),
            "salient_attention": salient_weights.detach(),
            "complement_attention": complement_weights.detach(),
            "descriptor": descriptor.detach(),
        }


class MicroDetailPatchExpert(nn.Module):
    """Class residual head that pools the most detailed foreground patch tokens."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        top_k: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.12,
        foreground_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_k = max(1, int(top_k))
        self.temperature = max(1e-4, float(temperature))
        self.foreground_power = max(0.0, float(foreground_power))
        self.patch_norm = nn.LayerNorm(dim)
        self.descriptor_dim = (3 * int(dim)) + 4
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
    def _normalize_per_sample(values: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        values = values.float()
        if torch.is_tensor(valid_mask) and valid_mask.shape == values.shape:
            valid = valid_mask.to(device=values.device, dtype=torch.bool)
        else:
            valid = torch.ones_like(values, dtype=torch.bool)
        masked_min = values.masked_fill(~valid, torch.inf).amin(dim=1, keepdim=True)
        masked_max = values.masked_fill(~valid, -torch.inf).amax(dim=1, keepdim=True)
        empty = ~torch.isfinite(masked_min.squeeze(1)) | ~torch.isfinite(masked_max.squeeze(1))
        masked_min = torch.where(empty[:, None], torch.zeros_like(masked_min), masked_min)
        masked_max = torch.where(empty[:, None], torch.ones_like(masked_max), masked_max)
        normalized = (values - masked_min) / (masked_max - masked_min).clamp(min=1e-6)
        return normalized.masked_fill(~valid, 0.0)

    @staticmethod
    def _gather_detail_scores(
        detail_map: Optional[Tensor],
        patch_indices: Optional[Tensor],
        token_count: int,
    ) -> Optional[Tensor]:
        if not torch.is_tensor(detail_map) or not torch.is_tensor(patch_indices):
            return None
        if detail_map.ndim == 4:
            flat = detail_map.detach().float().flatten(2).squeeze(1)
        elif detail_map.ndim == 3:
            flat = detail_map.detach().float().flatten(1)
        else:
            return None
        if flat.ndim != 2 or patch_indices.ndim != 2:
            return None
        if flat.size(0) != patch_indices.size(0) or patch_indices.size(1) != token_count:
            return None
        if int(patch_indices.max().detach().cpu().item()) >= int(flat.size(1)):
            return None
        return flat.gather(1, patch_indices.to(device=flat.device, dtype=torch.long))

    def _score_patches(
        self,
        patch_tokens: Tensor,
        detail_map: Optional[Tensor],
        patch_indices: Optional[Tensor],
        foreground_prior: Optional[Tensor],
        valid_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        batch_size, token_count = patch_tokens.shape[:2]
        device = patch_tokens.device
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=device, dtype=torch.bool)
        else:
            valid = torch.ones((batch_size, token_count), device=device, dtype=torch.bool)

        token_energy = patch_tokens.detach().float().norm(dim=-1)
        token_score = self._normalize_per_sample(token_energy, valid)
        detail_score = self._gather_detail_scores(detail_map, patch_indices, token_count)
        if torch.is_tensor(detail_score):
            detail_score = self._normalize_per_sample(detail_score.to(device=device), valid)
        else:
            detail_score = token_score
        if torch.is_tensor(foreground_prior) and tuple(foreground_prior.shape) == (batch_size, token_count):
            foreground = foreground_prior.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            foreground = torch.ones((batch_size, token_count), device=device, dtype=torch.float32)
        if self.foreground_power > 0.0:
            foreground_gate = foreground.clamp(min=0.0).pow(self.foreground_power)
        else:
            foreground_gate = torch.ones_like(foreground)
        score = (0.70 * detail_score + 0.30 * token_score) * (0.15 + 0.85 * foreground_gate)
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        return score, foreground

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        detail_map: Optional[Tensor] = None,
        patch_indices: Optional[Tensor] = None,
        foreground_prior: Optional[Tensor] = None,
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
                "selected_indices": patch_tokens.new_zeros(patch_tokens.shape[:2], dtype=torch.long),
            }

        normalized = self.patch_norm(patch_tokens)
        score, foreground = self._score_patches(
            patch_tokens,
            detail_map=detail_map,
            patch_indices=patch_indices,
            foreground_prior=foreground_prior,
            valid_mask=valid_mask,
        )
        selected_count = min(self.top_k, int(patch_tokens.size(1)))
        selected_indices = torch.topk(score, k=selected_count, dim=1, largest=True, sorted=False).indices
        gathered_tokens = normalized.gather(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, normalized.size(-1)),
        )
        selected_score = score.gather(1, selected_indices).float()
        empty = selected_score <= torch.finfo(selected_score.dtype).min / 2
        if empty.any():
            selected_score = selected_score.masked_fill(empty, 0.0)
        selected_weights = F.softmax(selected_score / self.temperature, dim=1)
        weighted_feature = torch.einsum("bk,bkd->bd", selected_weights, gathered_tokens.float())
        mean_feature = gathered_tokens.float().mean(dim=1)
        max_local = selected_score.argmax(dim=1)
        max_feature = gathered_tokens[
            torch.arange(gathered_tokens.size(0), device=gathered_tokens.device),
            max_local,
        ].float()
        selected_foreground = foreground.gather(1, selected_indices)
        stats = torch.stack(
            (
                selected_score.max(dim=1).values,
                selected_score.mean(dim=1),
                selected_foreground.mean(dim=1),
                selected_weights.max(dim=1).values,
            ),
            dim=1,
        )
        descriptor = torch.cat((weighted_feature, mean_feature, max_feature, stats), dim=1)
        logits = self.net(descriptor.to(dtype=patch_tokens.dtype))
        if not return_trace:
            return logits

        attention = patch_tokens.new_zeros(patch_tokens.shape[:2], dtype=torch.float32)
        attention.scatter_add_(1, selected_indices, selected_weights.detach().float())
        return logits, {
            "attention": attention.detach(),
            "descriptor": descriptor.detach(),
            "selected_indices": selected_indices.detach(),
        }


class PatchObjectnessGuidedHead(nn.Module):
    """Residual classifier guided by token-level objectness learned from bbox priors."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        temperature: float = 0.75,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = max(1e-4, float(temperature))
        hidden = max(32, int(hidden_dim))
        self.patch_norm = nn.LayerNorm(dim)
        self.objectness_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, 1),
        )
        self.descriptor_dim = (3 * int(dim)) + 4
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
    def _resolve_valid_mask(
        patch_tokens: Tensor,
        valid_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, token_count = patch_tokens.shape[:2]
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=patch_tokens.device, dtype=torch.bool)
        else:
            valid = torch.ones((batch_size, token_count), device=patch_tokens.device, dtype=torch.bool)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid[empty] = True
        return valid

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        batch_size = int(patch_tokens.size(0))
        token_count = int(patch_tokens.size(1)) if patch_tokens.ndim >= 2 else 0
        if patch_tokens.ndim != 3 or token_count == 0:
            logits = patch_tokens.new_zeros((batch_size, self.net[-1].out_features))
            empty_tokens = patch_tokens.new_zeros((batch_size, token_count), dtype=torch.float32)
            if not return_trace:
                return logits
            return logits, {
                "attention": empty_tokens,
                "objectness_logits": empty_tokens,
                "objectness_probability": empty_tokens,
                "descriptor": patch_tokens.new_zeros((batch_size, self.descriptor_dim)),
                "stats": patch_tokens.new_zeros((batch_size, 4)),
            }

        normalized = self.patch_norm(patch_tokens)
        valid = self._resolve_valid_mask(patch_tokens, valid_mask)
        objectness_logits = self.objectness_head(normalized).squeeze(-1)
        objectness_probability = torch.sigmoid(objectness_logits.float()).masked_fill(~valid, 0.0)
        weight_scores = torch.sigmoid(objectness_logits.float() / self.temperature).masked_fill(~valid, 0.0)
        weight_sum = weight_scores.sum(dim=1, keepdim=True)
        uniform = valid.to(dtype=weight_scores.dtype) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        weights = torch.where(weight_sum > 1e-6, weight_scores / weight_sum.clamp(min=1e-6), uniform)

        weighted_feature = torch.einsum("bn,bnd->bd", weights, normalized.float())
        valid_float = valid.to(dtype=normalized.dtype).unsqueeze(-1)
        mean_feature = (
            (normalized * valid_float).float().sum(dim=1)
            / valid_float.float().sum(dim=1).clamp(min=1.0)
        )
        max_index = objectness_probability.masked_fill(~valid, -1.0).argmax(dim=1)
        max_feature = normalized[
            torch.arange(batch_size, device=normalized.device),
            max_index,
        ].float()
        entropy = -(weights * weights.clamp(min=1e-8).log()).sum(dim=1)
        entropy = entropy / valid.sum(dim=1).float().clamp(min=2.0).log()
        stats = torch.stack(
            (
                objectness_probability.max(dim=1).values,
                (
                    objectness_probability.sum(dim=1)
                    / valid.sum(dim=1).to(dtype=objectness_probability.dtype).clamp(min=1.0)
                ),
                weights.max(dim=1).values,
                entropy.clamp(min=0.0, max=1.0),
            ),
            dim=1,
        )
        descriptor = torch.cat((weighted_feature, mean_feature, max_feature, stats), dim=1)
        logits = self.net(descriptor.to(dtype=patch_tokens.dtype))
        if not return_trace:
            return logits
        return logits, {
            "attention": weights.detach(),
            "objectness_logits": objectness_logits.detach(),
            "objectness_probability": objectness_probability.detach(),
            "descriptor": descriptor.detach(),
            "stats": stats.detach(),
        }


class BBoxPriorPatchContextHead(nn.Module):
    """Residual classifier from deterministic bbox/object and outside-context patch pools."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        temperature: float = 0.50,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = max(1e-4, float(temperature))
        hidden = max(32, int(hidden_dim))
        self.patch_norm = nn.LayerNorm(dim)
        self.descriptor_dim = (4 * int(dim)) + 6
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
    def _resolve_valid_mask(
        patch_tokens: Tensor,
        valid_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, token_count = patch_tokens.shape[:2]
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=patch_tokens.device, dtype=torch.bool)
        else:
            valid = torch.ones((batch_size, token_count), device=patch_tokens.device, dtype=torch.bool)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty] = True
        return valid

    @staticmethod
    def _normalize_weights(scores: Tensor, valid: Tensor) -> Tensor:
        masked = scores.float().clamp(min=0.0) * valid.to(dtype=torch.float32)
        totals = masked.sum(dim=1, keepdim=True)
        empty = totals.squeeze(1) <= 1e-8
        if empty.any():
            fallback = valid.to(dtype=torch.float32)
            fallback_totals = fallback.sum(dim=1, keepdim=True)
            fallback_empty = fallback_totals.squeeze(1) <= 1e-8
            if fallback_empty.any():
                fallback = fallback.clone()
                fallback[fallback_empty] = 1.0
                fallback_totals = fallback.sum(dim=1, keepdim=True)
            masked = torch.where(empty[:, None], fallback, masked)
            totals = torch.where(empty[:, None], fallback_totals, totals)
        return masked / totals.clamp(min=1e-8)

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        bbox_prior: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        batch_size = int(patch_tokens.size(0))
        token_count = int(patch_tokens.size(1)) if patch_tokens.ndim >= 2 else 0
        if patch_tokens.ndim != 3 or token_count == 0:
            logits = patch_tokens.new_zeros((batch_size, self.net[-1].out_features))
            empty_attention = patch_tokens.new_zeros((batch_size, token_count), dtype=torch.float32)
            if not return_trace:
                return logits
            return logits, {
                "object_attention": empty_attention,
                "background_attention": empty_attention,
                "descriptor": patch_tokens.new_zeros((batch_size, self.descriptor_dim)),
                "stats": patch_tokens.new_zeros((batch_size, 6)),
            }

        normalized = self.patch_norm(patch_tokens)
        valid = self._resolve_valid_mask(patch_tokens, valid_mask)
        if torch.is_tensor(bbox_prior) and tuple(bbox_prior.shape) == (batch_size, token_count):
            prior = bbox_prior.to(device=patch_tokens.device, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            prior = torch.ones((batch_size, token_count), device=patch_tokens.device, dtype=torch.float32)
        prior = prior.masked_fill(~valid, 0.0)
        if self.temperature != 1.0:
            prior_for_object = prior.clamp(min=0.0).pow(1.0 / self.temperature)
        else:
            prior_for_object = prior
        object_weights = self._normalize_weights(prior_for_object, valid)
        background_scores = (1.0 - prior).clamp(min=0.0) * valid.to(dtype=torch.float32)
        background_weights = self._normalize_weights(background_scores, valid)
        global_weights = valid.to(dtype=torch.float32)
        global_weights = global_weights / global_weights.sum(dim=1, keepdim=True).clamp(min=1.0)

        object_feature = torch.einsum("bn,bnd->bd", object_weights, normalized.float())
        background_feature = torch.einsum("bn,bnd->bd", background_weights, normalized.float())
        global_feature = torch.einsum("bn,bnd->bd", global_weights, normalized.float())
        max_index = prior.masked_fill(~valid, -1.0).argmax(dim=1)
        max_feature = normalized[
            torch.arange(batch_size, device=normalized.device),
            max_index,
        ].float()
        object_entropy = -(object_weights * object_weights.clamp(min=1e-8).log()).sum(dim=1)
        object_entropy = object_entropy / valid.sum(dim=1).float().clamp(min=2.0).log()
        stats = torch.stack(
            (
                prior.max(dim=1).values,
                (prior.sum(dim=1) / valid.sum(dim=1).float().clamp(min=1.0)),
                object_weights.max(dim=1).values,
                background_weights.max(dim=1).values,
                object_entropy.clamp(min=0.0, max=1.0),
                (object_feature - background_feature).float().norm(dim=1)
                / math.sqrt(float(max(1, self.dim))),
            ),
            dim=1,
        )
        descriptor = torch.cat(
            (
                object_feature,
                background_feature,
                object_feature - background_feature,
                max_feature + 0.25 * global_feature,
                stats,
            ),
            dim=1,
        )
        logits = self.net(descriptor.to(dtype=patch_tokens.dtype))
        if not return_trace:
            return logits
        return logits, {
            "object_attention": object_weights.detach(),
            "background_attention": background_weights.detach(),
            "descriptor": descriptor.detach(),
            "stats": stats.detach(),
        }


class LateClassAttentionPooling(nn.Module):
    """CaiT-style late class query over patch tokens before the classifier head."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.05,
        mlp_ratio: float = 2.0,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        requested_heads = max(1, int(num_heads))
        self.num_heads = math.gcd(self.dim, requested_heads) or 1
        self.head_dim = self.dim // self.num_heads
        hidden_dim = max(self.dim, int(round(float(max(0.25, mlp_ratio)) * self.dim)))
        self.query_norm = nn.LayerNorm(self.dim)
        self.patch_norm = nn.LayerNorm(self.dim)
        self.query_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.key_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_dropout = nn.Dropout(float(max(0.0, dropout)))
        self.residual_norm = nn.LayerNorm(self.dim)
        self.residual_projection = nn.Linear(self.dim, self.dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden_dim, self.dim),
        )
        self.output_dropout = nn.Dropout(float(max(0.0, dropout)))
        self.residual_scale = nn.Parameter(
            torch.tensor(float(max(0.0, residual_scale)), dtype=torch.float32)
        )

    def zero_init_residual(self) -> None:
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            if projection.weight.shape[0] == projection.weight.shape[1]:
                nn.init.eye_(projection.weight)
        nn.init.zeros_(self.residual_projection.weight)
        if self.residual_projection.bias is not None:
            nn.init.zeros_(self.residual_projection.bias)
        final_linear = self.mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _resolve_valid_mask(patch_tokens: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        batch_size, token_count = patch_tokens.shape[:2]
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=patch_tokens.device, dtype=torch.bool)
        else:
            valid = torch.ones((batch_size, token_count), device=patch_tokens.device, dtype=torch.bool)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty] = True
        return valid

    def forward(
        self,
        head_input: Tensor,
        patch_tokens: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            if not return_trace:
                return head_input
            empty_attention = patch_tokens.new_zeros((head_input.size(0), 0), dtype=torch.float32)
            return head_input, {
                "attention": empty_attention,
                "context_norm": head_input.new_zeros((head_input.size(0),), dtype=torch.float32),
                "residual_norm": head_input.new_zeros((head_input.size(0),), dtype=torch.float32),
            }

        batch_size, token_count = patch_tokens.shape[:2]
        valid = self._resolve_valid_mask(patch_tokens, valid_mask)
        query = self.query_proj(self.query_norm(head_input))
        keys = self.key_proj(self.patch_norm(patch_tokens))
        values = self.value_proj(self.patch_norm(patch_tokens))
        query = query.view(batch_size, self.num_heads, 1, self.head_dim)
        keys = keys.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query.float(), keys.float().transpose(-2, -1))
        scores = scores / math.sqrt(float(max(1, self.head_dim)))
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        attention = F.softmax(scores, dim=-1).to(dtype=values.dtype)
        attention = self.attn_dropout(attention)
        context = torch.matmul(attention, values).transpose(1, 2).reshape(batch_size, self.dim)
        residual = self.residual_projection(self.residual_norm(context))
        residual = residual + self.mlp(context)
        pooled = head_input + self.output_dropout(residual) * self.residual_scale.to(
            device=head_input.device,
            dtype=head_input.dtype,
        )
        if not return_trace:
            return pooled
        attention_trace = attention.detach().float().mean(dim=1).squeeze(1)
        return pooled, {
            "attention": attention_trace,
            "context_norm": context.detach().float().norm(dim=1),
            "residual_norm": residual.detach().float().norm(dim=1),
        }


class AdaptivePartTokenLearner(nn.Module):
    """Learn a compact set of foreground-biased part tokens from patch tokens."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        part_count: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.08,
        temperature: float = 0.70,
        foreground_power: float = 1.0,
        bbox_weight: float = 0.75,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.part_count = max(1, int(part_count))
        self.temperature = max(1e-4, float(temperature))
        self.foreground_power = max(0.0, float(foreground_power))
        self.bbox_weight = min(1.0, max(0.0, float(bbox_weight)))
        self.patch_norm = nn.LayerNorm(dim)
        self.part_queries = nn.Parameter(torch.empty(self.part_count, int(dim)))
        self.attention_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(32, int(hidden_dim))),
            nn.GELU(),
            nn.Linear(max(32, int(hidden_dim)), self.part_count),
        )
        self.descriptor_dim = (self.part_count + 1) * int(dim) + (3 * self.part_count)
        hidden = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Linear(self.descriptor_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.part_queries, std=0.02)

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def _combined_prior(
        self,
        *,
        foreground_prior: Optional[Tensor],
        bbox_prior: Optional[Tensor],
        shape: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        batch_size, token_count = shape
        prior = torch.ones((batch_size, token_count), device=device, dtype=torch.float32)
        has_prior = False
        if torch.is_tensor(foreground_prior) and tuple(foreground_prior.shape) == shape:
            prior = foreground_prior.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            has_prior = True
        if torch.is_tensor(bbox_prior) and tuple(bbox_prior.shape) == shape:
            bbox = bbox_prior.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if has_prior:
                prior = (1.0 - self.bbox_weight) * prior + self.bbox_weight * bbox
            else:
                prior = bbox
        if self.foreground_power > 0.0:
            prior = prior.clamp(min=0.0).pow(self.foreground_power)
        return prior.clamp(0.0, 1.0)

    def forward(
        self,
        patch_tokens: Tensor,
        *,
        foreground_prior: Optional[Tensor] = None,
        bbox_prior: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            logits = patch_tokens.new_zeros((patch_tokens.size(0), self.net[-1].out_features))
            if not return_trace:
                return logits
            return logits, {
                "attention": patch_tokens.new_zeros(
                    (patch_tokens.size(0), self.part_count, patch_tokens.size(1)),
                    dtype=torch.float32,
                ),
                "descriptor": patch_tokens.new_zeros((patch_tokens.size(0), self.descriptor_dim)),
                "foreground_mass": patch_tokens.new_zeros((patch_tokens.size(0), self.part_count)),
                "max_weight": patch_tokens.new_zeros((patch_tokens.size(0), self.part_count)),
            }

        batch_size, token_count = patch_tokens.shape[:2]
        device = patch_tokens.device
        if torch.is_tensor(valid_mask) and tuple(valid_mask.shape) == (batch_size, token_count):
            valid = valid_mask.to(device=device, dtype=torch.bool)
        else:
            valid = torch.ones((batch_size, token_count), device=device, dtype=torch.bool)
        normalized = self.patch_norm(patch_tokens)
        query_logits = torch.einsum(
            "bnd,pd->bnp",
            normalized.float(),
            self.part_queries.float(),
        ) / math.sqrt(float(max(1, self.dim)))
        learned_logits = self.attention_head(normalized).float()
        attention_logits = query_logits + learned_logits
        prior = self._combined_prior(
            foreground_prior=foreground_prior,
            bbox_prior=bbox_prior,
            shape=(batch_size, token_count),
            device=device,
        )
        attention_logits = attention_logits + torch.log(prior.clamp(min=1e-4)).unsqueeze(-1)
        attention_logits = attention_logits.masked_fill(~valid.unsqueeze(-1), -1e4)
        empty = ~valid.any(dim=1)
        if empty.any():
            attention_logits[empty] = 0.0
        attention = F.softmax(attention_logits.transpose(1, 2) / self.temperature, dim=-1)
        part_tokens = torch.einsum("bpn,bnd->bpd", attention, normalized.float())

        valid_float = valid.to(dtype=normalized.dtype).unsqueeze(-1)
        mean_feature = (normalized * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1.0)
        foreground_mass = torch.einsum("bpn,bn->bp", attention.float(), prior.float())
        entropy = -(attention.float() * attention.float().clamp(min=1e-6).log()).sum(dim=-1)
        entropy = entropy / math.log(float(max(2, token_count)))
        max_weight = attention.float().max(dim=-1).values
        stats = torch.stack((foreground_mass, max_weight, entropy), dim=-1).flatten(1)
        descriptor = torch.cat((part_tokens.flatten(1), mean_feature.float(), stats), dim=1)
        logits = self.net(descriptor.to(dtype=patch_tokens.dtype))
        if not return_trace:
            return logits
        return logits, {
            "attention": attention.detach(),
            "descriptor": descriptor.detach(),
            "foreground_mass": foreground_mass.detach(),
            "max_weight": max_weight.detach(),
        }


class PatchEvidenceRouterHead(nn.Module):
    """Gated-attention MIL router over patch tokens for one-vs-one evidence."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        top_k: int = 4,
        bbox_weight: float = 0.75,
        margin_prior_mode: str = "none",
        margin_prior_scale: float = 0.0,
        summary_stats: bool = False,
        gate_bias: float = -1.5,
    ) -> None:
        super().__init__()
        hidden = max(16, int(hidden_dim))
        self.top_k = max(1, int(top_k))
        self.bbox_weight = float(max(0.0, bbox_weight))
        self.summary_stats = bool(summary_stats)
        self.margin_prior_mode = str(margin_prior_mode or "none").strip().lower()
        if self.margin_prior_mode not in {
            "none",
            "weighted",
            "max",
            "topk_mean",
            "weighted_minus_mean",
        }:
            raise ValueError(
                "margin_prior_mode phai la none, weighted, max, topk_mean, "
                "hoac weighted_minus_mean."
            )
        self.margin_prior_scale = float(max(0.0, margin_prior_scale))
        self.patch_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.patch_value = nn.Linear(dim, hidden)
        self.patch_gate = nn.Linear(dim, hidden)
        self.context_value = nn.Linear(dim, hidden, bias=False)
        self.attention = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(float(max(0.0, dropout)))
        summary_dim = 13 if self.summary_stats else 0
        descriptor_dim = (2 * dim) + 3 + summary_dim
        self.output = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.patch_gate.bias, float(gate_bias))

    def zero_init_residual(self) -> None:
        final_linear = self.output[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        patches: Tensor,
        context: Tensor,
        patch_pair_margin: Tensor,
        *,
        bbox_prior: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        base_pair_stats: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if patches.ndim != 3:
            raise ValueError("PatchEvidenceRouterHead expects patches [B,N,D].")
        batch_size, patch_count, _ = patches.shape
        if patch_pair_margin.shape[:2] != (batch_size, patch_count):
            raise ValueError("patch_pair_margin phai co shape [B,N].")

        patch_norm = self.patch_norm(patches)
        context_norm = self.context_norm(context).unsqueeze(1)
        gated = torch.tanh(
            self.patch_value(patch_norm) + self.context_value(context_norm)
        ) * torch.sigmoid(self.patch_gate(patch_norm))
        attention_logits = self.attention(self.dropout(gated)).squeeze(-1)
        if torch.is_tensor(bbox_prior) and bbox_prior.shape[:2] == attention_logits.shape:
            bbox_values = bbox_prior.to(device=patches.device, dtype=torch.float32).clamp(0.0, 1.0)
            bbox_logit = torch.logit(bbox_values.clamp(1e-4, 1.0 - 1e-4))
            attention_logits = attention_logits + float(self.bbox_weight) * bbox_logit.to(
                dtype=attention_logits.dtype
            )
        else:
            bbox_values = None
        if torch.is_tensor(valid_mask):
            valid = valid_mask.to(device=patches.device, dtype=torch.bool)
            if valid.shape != attention_logits.shape:
                valid = None
        else:
            valid = None
        if valid is not None:
            all_invalid = ~valid.any(dim=1)
            if bool(all_invalid.any().item()):
                valid = valid.clone()
                valid[all_invalid] = True
            attention_logits = attention_logits.masked_fill(~valid, torch.finfo(attention_logits.dtype).min)

        k = min(self.top_k, patch_count)
        if k < patch_count:
            top_values, top_indices = attention_logits.topk(k=k, dim=1)
            top_attention = F.softmax(top_values.float(), dim=1).to(dtype=patches.dtype)
            attention = patches.new_zeros((batch_size, patch_count))
            attention.scatter_(1, top_indices, top_attention)
            selected_margin = patch_pair_margin.gather(1, top_indices).to(dtype=torch.float32)
            max_margin = selected_margin.max(dim=1).values
            topk_mean_margin = selected_margin.mean(dim=1)
        else:
            attention = F.softmax(attention_logits.float(), dim=1).to(dtype=patches.dtype)
            top_indices = torch.arange(patch_count, device=patches.device, dtype=torch.long).view(1, -1)
            top_indices = top_indices.expand(batch_size, -1)
            selected_margin = patch_pair_margin.float()
            max_margin = selected_margin.max(dim=1).values
            topk_mean_margin = selected_margin.mean(dim=1)

        weighted_patch = torch.bmm(attention.unsqueeze(1), patches).squeeze(1)
        weighted_margin = (attention.float() * patch_pair_margin.float()).sum(dim=1)
        mean_margin = self._masked_mean(
            patch_pair_margin.float(),
            valid if valid is not None else None,
        )
        descriptor_parts = [
            context,
            weighted_patch,
            weighted_margin.to(dtype=context.dtype).unsqueeze(1),
            max_margin.to(dtype=context.dtype).unsqueeze(1),
            mean_margin.to(dtype=context.dtype).unsqueeze(1),
        ]
        summary_values: Optional[Tensor] = None
        if self.summary_stats:
            margin_values = patch_pair_margin.float()
            if valid is not None:
                valid_float = valid.to(device=patches.device, dtype=torch.float32)
                safe_mask = valid
            else:
                valid_float = torch.ones(
                    (batch_size, patch_count),
                    device=patches.device,
                    dtype=torch.float32,
                )
                safe_mask = torch.ones(
                    (batch_size, patch_count),
                    device=patches.device,
                    dtype=torch.bool,
                )
            valid_denom = valid_float.sum(dim=1).clamp_min(1.0)
            min_margin = margin_values.masked_fill(~safe_mask, float("inf")).min(dim=1).values
            min_margin = torch.where(torch.isfinite(min_margin), min_margin, mean_margin)
            positive_fraction = (
                ((margin_values > 0.0).to(dtype=torch.float32) * valid_float).sum(dim=1)
                / valid_denom
            )
            if bbox_values is not None:
                bbox_weights = bbox_values.to(device=patches.device, dtype=torch.float32) * valid_float
            else:
                bbox_weights = valid_float
            bbox_empty = bbox_weights.sum(dim=1) <= 1e-6
            if bool(bbox_empty.any().item()):
                bbox_weights = bbox_weights.clone()
                bbox_weights[bbox_empty] = valid_float[bbox_empty]
            bbox_denom = bbox_weights.sum(dim=1).clamp_min(1e-6)
            bbox_mask = bbox_weights > 0.05
            bbox_empty_mask = ~bbox_mask.any(dim=1)
            if bool(bbox_empty_mask.any().item()):
                bbox_mask = bbox_mask.clone()
                bbox_mask[bbox_empty_mask] = safe_mask[bbox_empty_mask]
            bbox_weighted_margin = (margin_values * bbox_weights).sum(dim=1) / bbox_denom
            bbox_max_margin = margin_values.masked_fill(~bbox_mask, float("-inf")).max(dim=1).values
            bbox_max_margin = torch.where(
                torch.isfinite(bbox_max_margin),
                bbox_max_margin,
                bbox_weighted_margin,
            )
            bbox_min_margin = margin_values.masked_fill(~bbox_mask, float("inf")).min(dim=1).values
            bbox_min_margin = torch.where(
                torch.isfinite(bbox_min_margin),
                bbox_min_margin,
                bbox_weighted_margin,
            )
            bbox_k = min(self.top_k, patch_count)
            bbox_top_values = margin_values.masked_fill(~bbox_mask, float("-inf")).topk(
                k=bbox_k,
                dim=1,
            ).values
            bbox_top_finite = torch.isfinite(bbox_top_values)
            bbox_top_values = torch.where(
                bbox_top_finite,
                bbox_top_values,
                torch.zeros_like(bbox_top_values),
            )
            bbox_topk_mean_margin = (
                bbox_top_values.sum(dim=1)
                / bbox_top_finite.to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
            )
            bbox_positive_fraction = (
                ((margin_values > 0.0).to(dtype=torch.float32) * bbox_weights).sum(dim=1)
                / bbox_denom
            )
            if torch.is_tensor(base_pair_stats) and base_pair_stats.shape == (batch_size, 5):
                base_stats = base_pair_stats.to(device=patches.device, dtype=torch.float32)
            else:
                base_stats = margin_values.new_zeros((batch_size, 5))
            summary_values = torch.cat(
                (
                    min_margin.unsqueeze(1),
                    topk_mean_margin.unsqueeze(1),
                    bbox_weighted_margin.unsqueeze(1),
                    bbox_max_margin.unsqueeze(1),
                    bbox_min_margin.unsqueeze(1),
                    bbox_topk_mean_margin.unsqueeze(1),
                    positive_fraction.unsqueeze(1),
                    bbox_positive_fraction.unsqueeze(1),
                    base_stats,
                ),
                dim=1,
            )
            descriptor_parts.append(summary_values.to(dtype=context.dtype))
        descriptor = torch.cat(tuple(descriptor_parts), dim=1)
        logits = self.output(descriptor).squeeze(1)
        prior = logits.new_zeros((batch_size,))
        if self.margin_prior_scale > 0.0 and self.margin_prior_mode != "none":
            if self.margin_prior_mode == "weighted":
                prior = weighted_margin.to(device=logits.device, dtype=logits.dtype)
            elif self.margin_prior_mode == "max":
                prior = max_margin.to(device=logits.device, dtype=logits.dtype)
            elif self.margin_prior_mode == "topk_mean":
                prior = topk_mean_margin.to(device=logits.device, dtype=logits.dtype)
            elif self.margin_prior_mode == "weighted_minus_mean":
                prior = (weighted_margin - mean_margin).to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
            logits = logits + float(self.margin_prior_scale) * prior
        if not return_trace:
            return logits
        trace = {
            "attention": attention.detach(),
            "descriptor": descriptor.detach(),
            "weighted_margin": weighted_margin.detach(),
            "max_margin": max_margin.detach(),
            "mean_margin": mean_margin.detach(),
            "topk_mean_margin": topk_mean_margin.detach(),
            "margin_prior": prior.detach(),
            "selected_indices": top_indices.detach(),
        }
        if summary_values is not None:
            trace["summary_stats"] = summary_values.detach()
        if bbox_values is not None:
            trace["bbox_mass"] = (attention.float() * bbox_values.float()).sum(dim=1).detach()
        return logits, trace


def _patch_evidence_valid_mask(local_logits: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
    batch_size, token_count = local_logits.shape[:2]
    if torch.is_tensor(key_padding_mask) and key_padding_mask.shape[:2] == (batch_size, token_count):
        valid = ~key_padding_mask.to(device=local_logits.device, dtype=torch.bool)
    else:
        valid = torch.ones((batch_size, token_count), device=local_logits.device, dtype=torch.bool)
    empty = ~valid.any(dim=1)
    if bool(empty.any().item()):
        valid = valid.clone()
        valid[empty] = True
    return valid


def _patch_evidence_bbox_weights(
    local_logits: Tensor,
    bbox_prior: Optional[Tensor],
    valid_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    batch_size, token_count = local_logits.shape[:2]
    if torch.is_tensor(bbox_prior) and bbox_prior.shape[:2] == (batch_size, token_count):
        weights = bbox_prior.to(device=local_logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    else:
        weights = torch.ones((batch_size, token_count), device=local_logits.device, dtype=torch.float32)
    weights = weights * valid_mask.to(dtype=torch.float32)
    empty_weight = weights.sum(dim=1) <= 1e-6
    if bool(empty_weight.any().item()):
        weights = weights.clone()
        weights[empty_weight] = valid_mask[empty_weight].to(dtype=torch.float32)
    bbox_mask = (weights > 0.05) & valid_mask
    empty_mask = ~bbox_mask.any(dim=1)
    if bool(empty_mask.any().item()):
        bbox_mask = bbox_mask.clone()
        bbox_mask[empty_mask] = valid_mask[empty_mask]
    return weights, bbox_mask


def _patch_evidence_masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def _patch_evidence_weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    return (values * weights).sum(dim=1) / denom


def _patch_evidence_masked_max(values: Tensor, mask: Tensor) -> Tensor:
    filled = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    result = filled.max(dim=1).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


def _patch_evidence_masked_min(values: Tensor, mask: Tensor) -> Tensor:
    return -_patch_evidence_masked_max(-values, mask)


def _patch_evidence_masked_topk_mean(values: Tensor, mask: Tensor, top_k: int) -> Tensor:
    k = max(1, min(int(top_k), int(values.size(1))))
    filled = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    selected = torch.topk(filled, k=k, dim=1).values
    finite = torch.isfinite(selected)
    selected = torch.where(finite, selected, torch.zeros_like(selected))
    denom = finite.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    return selected.sum(dim=1) / denom


def _patch_evidence_top1_fraction(local_logits: Tensor, mask: Tensor, class_count: int) -> Tensor:
    top1 = local_logits.argmax(dim=-1)
    one_hot = F.one_hot(top1, num_classes=int(class_count)).to(dtype=local_logits.dtype)
    weights = mask.to(device=local_logits.device, dtype=local_logits.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (one_hot * weights).sum(dim=1) / denom


def _patch_evidence_weighted_top1_fraction(
    local_logits: Tensor,
    weights: Tensor,
    class_count: int,
) -> Tensor:
    top1 = local_logits.argmax(dim=-1)
    one_hot = F.one_hot(top1, num_classes=int(class_count)).to(dtype=local_logits.dtype)
    weights = weights.to(device=local_logits.device, dtype=local_logits.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-6)
    return (one_hot * weights).sum(dim=1) / denom


def _patch_evidence_pair_margin_features(
    local_logits: Tensor,
    *,
    valid_mask: Tensor,
    bbox_mask: Tensor,
    bbox_weights: Tensor,
    pairs: Sequence[Tuple[int, int]],
    top_k: int,
) -> List[Tensor]:
    parts: List[Tensor] = []
    for left_class, right_class in pairs:
        margin = (
            local_logits[:, :, int(right_class)] - local_logits[:, :, int(left_class)]
        ).unsqueeze(-1)
        positive = (margin > 0).to(dtype=local_logits.dtype)
        parts.extend(
            [
                _patch_evidence_masked_mean(margin, valid_mask),
                _patch_evidence_masked_max(margin, valid_mask),
                _patch_evidence_masked_min(margin, valid_mask),
                _patch_evidence_masked_topk_mean(margin, valid_mask, top_k),
                _patch_evidence_weighted_mean(margin, bbox_weights),
                _patch_evidence_masked_max(margin, bbox_mask),
                _patch_evidence_masked_min(margin, bbox_mask),
                _patch_evidence_masked_topk_mean(margin, bbox_mask, top_k),
                _patch_evidence_masked_mean(positive, valid_mask),
                _patch_evidence_weighted_mean(positive, bbox_weights),
            ]
        )
    return parts


def patch_evidence_summary_features(
    local_logits: Tensor,
    *,
    key_padding_mask: Optional[Tensor] = None,
    bbox_prior: Optional[Tensor] = None,
    pairs: Sequence[Tuple[int, int]] = (),
    top_k: int = 4,
) -> Tensor:
    if local_logits.ndim != 3:
        raise ValueError("local_logits must have shape [B, N, C].")
    class_count = int(local_logits.size(-1))
    local_logits = local_logits.float()
    valid_mask = _patch_evidence_valid_mask(local_logits, key_padding_mask)
    bbox_weights, bbox_mask = _patch_evidence_bbox_weights(
        local_logits,
        bbox_prior,
        valid_mask,
    )
    feature_parts: List[Tensor] = [
        _patch_evidence_masked_mean(local_logits, valid_mask),
        _patch_evidence_masked_max(local_logits, valid_mask),
        _patch_evidence_masked_topk_mean(local_logits, valid_mask, top_k),
        _patch_evidence_weighted_mean(local_logits, bbox_weights),
        _patch_evidence_masked_max(local_logits, bbox_mask),
        _patch_evidence_masked_topk_mean(local_logits, bbox_mask, top_k),
        _patch_evidence_top1_fraction(local_logits, valid_mask, class_count),
        _patch_evidence_weighted_top1_fraction(local_logits, bbox_weights, class_count),
    ]
    feature_parts.extend(
        _patch_evidence_pair_margin_features(
            local_logits,
            valid_mask=valid_mask,
            bbox_mask=bbox_mask,
            bbox_weights=bbox_weights,
            pairs=pairs,
            top_k=top_k,
        )
    )
    return torch.cat([part.flatten(1) for part in feature_parts], dim=1)


class PatchEvidenceLinearVerifier(nn.Module):
    """Frozen full-feature 0-1 patch verifier exported from the diagnostic probe."""

    def __init__(
        self,
        *,
        raw_coef: Tensor,
        raw_intercept: float,
        pair: Tuple[int, int] = (0, 1),
        top_k: int = 4,
        min_pair_probability: float = 0.02,
        max_pair_margin: float = 0.40,
        confidence_threshold: float = 0.60,
        logit_boost: float = 0.01,
        protect_right_min_probability: float = 0.0,
        training_soft_adjustment: bool = False,
        training_soft_logit_scale: float = 0.05,
        training_soft_gate_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        coef = torch.as_tensor(raw_coef, dtype=torch.float32).flatten()
        if int(coef.numel()) <= 0:
            raise ValueError("PatchEvidenceLinearVerifier raw_coef must be non-empty.")
        self.register_buffer("raw_coef", coef, persistent=False)
        self.register_buffer(
            "raw_intercept",
            torch.tensor(float(raw_intercept), dtype=torch.float32),
            persistent=False,
        )
        self.pair = (int(pair[0]), int(pair[1]))
        self.top_k = max(1, int(top_k))
        self.min_pair_probability = float(min(1.0, max(0.0, min_pair_probability)))
        self.max_pair_margin = float(max(0.0, max_pair_margin))
        self.confidence_threshold = float(min(1.0, max(0.0, confidence_threshold)))
        self.logit_boost = float(max(0.0, logit_boost))
        self.protect_right_min_probability = float(
            min(1.0, max(0.0, protect_right_min_probability))
        )
        self.training_soft_adjustment = bool(training_soft_adjustment)
        self.training_soft_logit_scale = float(max(0.0, training_soft_logit_scale))
        self.training_soft_gate_temperature = float(
            max(1e-4, training_soft_gate_temperature)
        )

    @property
    def feature_dim(self) -> int:
        return int(self.raw_coef.numel())

    @classmethod
    def from_export_payload(
        cls,
        payload: Dict[str, Any],
        *,
        pair: str = "0-1",
        min_pair_probability: float = 0.02,
        max_pair_margin: float = 0.40,
        confidence_threshold: float = 0.60,
        logit_boost: float = 0.01,
        protect_right_min_probability: float = 0.0,
        training_soft_adjustment: bool = False,
        training_soft_logit_scale: float = 0.05,
        training_soft_gate_temperature: float = 0.05,
    ) -> "PatchEvidenceLinearVerifier":
        pair_text = str(pair or "").strip()
        pair_items = list(payload.get("pairs", []))
        selected: Optional[Dict[str, Any]] = None
        for item in pair_items:
            if not isinstance(item, dict):
                continue
            if pair_text and str(item.get("pair", "")) != pair_text:
                continue
            selected = item
            break
        if selected is None:
            for item in pair_items:
                if isinstance(item, dict) and str(item.get("status", "")) == "exported":
                    selected = item
                    break
        if selected is None or str(selected.get("status", "")) != "exported":
            raise ValueError("No exported patch-evidence verifier pair found in payload.")

        pair_name = str(selected.get("pair", pair_text or "0-1"))
        left_text, right_text = pair_name.replace(":", "-").split("-", 1)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if bool(metadata.get("spatial_evidence_features", False)):
            raise ValueError("Spatial patch-evidence exports are not supported by the in-model verifier.")
        if bool(metadata.get("source_domain_feature", False)):
            raise ValueError("Source-domain patch-evidence exports are not supported by the in-model verifier.")

        top_k = int(metadata.get("top_k", 4))
        raw_coef = torch.tensor(selected.get("raw_coef", []), dtype=torch.float32)
        feature_dim = int(payload.get("feature_dim", selected.get("feature_dim", raw_coef.numel())))
        feature_names = payload.get("feature_names", [])
        if isinstance(feature_names, list) and feature_names and len(feature_names) != feature_dim:
            raise ValueError("feature_names length does not match feature_dim in verifier export.")
        if int(raw_coef.numel()) != feature_dim:
            raise ValueError(
                f"raw_coef length {int(raw_coef.numel())} does not match feature_dim {feature_dim}."
            )
        return cls(
            raw_coef=raw_coef,
            raw_intercept=float(selected.get("raw_intercept", 0.0)),
            pair=(int(left_text), int(right_text)),
            top_k=top_k,
            min_pair_probability=min_pair_probability,
            max_pair_margin=max_pair_margin,
            confidence_threshold=confidence_threshold,
            logit_boost=logit_boost,
            protect_right_min_probability=protect_right_min_probability,
            training_soft_adjustment=training_soft_adjustment,
            training_soft_logit_scale=training_soft_logit_scale,
            training_soft_gate_temperature=training_soft_gate_temperature,
        )

    @classmethod
    def from_export_path(
        cls,
        path: Path,
        **kwargs: Any,
    ) -> "PatchEvidenceLinearVerifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid verifier export payload: {path}")
        return cls.from_export_payload(payload, **kwargs)

    def build_feature_vector(
        self,
        *,
        head_input: Tensor,
        logits: Tensor,
        local_logits: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        bbox_prior: Optional[Tensor] = None,
    ) -> Tensor:
        probabilities = logits.float().softmax(dim=1)
        log_probabilities = torch.log(probabilities.clamp(1e-8, 1.0))
        sorted_probabilities = torch.sort(probabilities, dim=1).values
        margins = (sorted_probabilities[:, -1] - sorted_probabilities[:, -2]).unsqueeze(1)
        confidence = sorted_probabilities[:, -1].unsqueeze(1)
        patch_features = patch_evidence_summary_features(
            local_logits,
            key_padding_mask=key_padding_mask,
            bbox_prior=bbox_prior,
            pairs=[self.pair],
            top_k=self.top_k,
        )
        features = torch.cat(
            (
                head_input.float(),
                probabilities,
                log_probabilities,
                margins,
                confidence,
                patch_features,
            ),
            dim=1,
        )
        if int(features.size(1)) != self.feature_dim:
            raise ValueError(
                f"Patch verifier feature dim mismatch: got {int(features.size(1))}, "
                f"expected {self.feature_dim}."
            )
        return features

    def verifier_probability(self, feature_vector: Tensor) -> Tensor:
        coef = self.raw_coef.to(device=feature_vector.device, dtype=feature_vector.dtype)
        intercept = self.raw_intercept.to(device=feature_vector.device, dtype=feature_vector.dtype)
        return torch.sigmoid(feature_vector @ coef + intercept)

    def forward(
        self,
        logits: Tensor,
        feature_vector: Tensor,
        *,
        return_trace: bool = False,
    ):
        probabilities = logits.float().softmax(dim=1)
        current = probabilities.argmax(dim=1)
        top1 = current
        left_class, right_class = self.pair
        if (
            left_class < 0
            or right_class < 0
            or left_class >= int(probabilities.size(1))
            or right_class >= int(probabilities.size(1))
        ):
            adjusted = logits
            route_mask = torch.zeros((int(logits.size(0)),), device=logits.device, dtype=torch.bool)
            trace = {
                "probability": logits.new_zeros((int(logits.size(0)),)),
                "confidence": logits.new_zeros((int(logits.size(0)),)),
                "candidate": current.detach(),
                "route_mask": route_mask.detach(),
            }
            return (adjusted, trace) if return_trace else adjusted

        pair_probabilities = probabilities[:, [left_class, right_class]]
        verifier_right_probability = self.verifier_probability(feature_vector.float())
        if self.training and self.training_soft_adjustment:
            pair_mass = pair_probabilities.min(dim=1).values
            pair_margin = (pair_probabilities[:, 0] - pair_probabilities[:, 1]).abs()
            temperature = float(self.training_soft_gate_temperature)
            pair_mass_weight = torch.sigmoid(
                (pair_mass - float(self.min_pair_probability)) / temperature
            )
            pair_margin_weight = torch.sigmoid(
                (float(self.max_pair_margin) - pair_margin) / temperature
            )
            confidence = torch.where(
                verifier_right_probability >= 0.5,
                verifier_right_probability,
                1.0 - verifier_right_probability,
            )
            if self.confidence_threshold > 0.5:
                confidence_weight = torch.sigmoid(
                    (confidence - float(self.confidence_threshold)) / temperature
                )
            else:
                confidence_weight = torch.ones_like(confidence)
            soft_gate = pair_mass_weight * pair_margin_weight * confidence_weight
            scale = (
                float(self.training_soft_logit_scale)
                if self.training_soft_logit_scale > 0.0
                else float(self.logit_boost)
            )
            delta = scale * soft_gate * (2.0 * verifier_right_probability - 1.0)
            adjusted = logits.clone()
            adjusted[:, int(right_class)] = adjusted[:, int(right_class)] + delta.to(
                dtype=adjusted.dtype
            )
            adjusted[:, int(left_class)] = adjusted[:, int(left_class)] - delta.to(
                dtype=adjusted.dtype
            )
            if not return_trace:
                return adjusted
            trace = {
                "probability": verifier_right_probability.detach(),
                "confidence": confidence.detach(),
                "candidate": torch.where(
                    verifier_right_probability >= 0.5,
                    torch.full_like(current, int(right_class)),
                    torch.full_like(current, int(left_class)),
                ).detach(),
                "route_mask": (soft_gate.detach() > 0.5),
                "feature_vector": feature_vector.detach(),
                "soft_gate": soft_gate.detach(),
                "soft_delta": delta.detach(),
            }
            return adjusted, trace

        candidate = torch.where(
            verifier_right_probability >= 0.5,
            torch.full_like(current, int(right_class)),
            torch.full_like(current, int(left_class)),
        )
        confidence = torch.where(
            verifier_right_probability >= 0.5,
            verifier_right_probability,
            1.0 - verifier_right_probability,
        )
        in_pair = (
            (current == int(left_class))
            | (current == int(right_class))
            | (top1 == int(left_class))
            | (top1 == int(right_class))
        )
        pair_mass_ok = pair_probabilities.min(dim=1).values >= float(self.min_pair_probability)
        pair_margin_ok = (
            (pair_probabilities[:, 0] - pair_probabilities[:, 1]).abs()
            <= float(self.max_pair_margin)
        )
        confidence_ok = confidence >= float(self.confidence_threshold)
        changed = candidate != current
        route_mask = in_pair & pair_mass_ok & pair_margin_ok & confidence_ok & changed
        if self.protect_right_min_probability > 0.0:
            suppress_right = (current == int(right_class)) & (candidate == int(left_class))
            protected = probabilities[:, int(right_class)] >= float(
                self.protect_right_min_probability
            )
            route_mask = route_mask & ~(suppress_right & protected)

        adjusted = logits.clone()
        if bool(route_mask.any().item()) and self.logit_boost > 0.0:
            rows = route_mask.nonzero(as_tuple=False).flatten()
            row_high = adjusted.max(dim=1).values + float(self.logit_boost)
            row_low = adjusted.min(dim=1).values - float(self.logit_boost)
            adjusted[rows, candidate[rows]] = row_high[rows].to(dtype=adjusted.dtype)
            adjusted[rows, current[rows]] = row_low[rows].to(dtype=adjusted.dtype)
        if not return_trace:
            return adjusted
        trace = {
            "probability": verifier_right_probability.detach(),
            "confidence": confidence.detach(),
            "candidate": candidate.detach(),
            "route_mask": route_mask.detach(),
            "feature_vector": feature_vector.detach(),
            "soft_gate": torch.zeros_like(verifier_right_probability).detach(),
            "soft_delta": torch.zeros_like(verifier_right_probability).detach(),
        }
        return adjusted, trace


class LocalZoomImageExpert(nn.Module):
    """Lightweight raw-image expert over the most salient foreground/detail crop."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        crop_size: int = 128,
        crop_scale: float = 0.48,
        score_mode: str = "foreground_detail",
        dropout: float = 0.08,
        score_size: int = 64,
    ) -> None:
        super().__init__()
        self.crop_size = max(32, int(crop_size))
        self.crop_scale = min(0.95, max(0.20, float(crop_scale)))
        self.score_mode = str(score_mode or "foreground_detail").strip().lower().replace("-", "_")
        self.score_size = max(16, int(score_size))
        hidden = max(32, int(hidden_dim))
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        self.descriptor_dim = hidden + 6
        self.head = nn.Sequential(
            nn.LayerNorm(self.descriptor_dim),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(self.descriptor_dim, int(num_classes)),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _normalize_map(values: Tensor) -> Tensor:
        flat = values.flatten(1)
        minimum = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
        maximum = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
        return (values - minimum) / (maximum - minimum).clamp(min=1e-6)

    @staticmethod
    def _soft_erode_map(values: Tensor, kernel_size: int) -> Tensor:
        kernel = max(1, int(kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        return -F.max_pool2d(-values[:, None], kernel_size=kernel, stride=1, padding=padding).squeeze(1)

    @staticmethod
    def _rgb_to_hsv_maps(image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]
        max_channel, max_index = image.max(dim=1)
        min_channel, _ = image.min(dim=1)
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
        saturation = torch.where(
            max_channel > eps,
            delta / max_channel.clamp(min=eps),
            torch.zeros_like(max_channel),
        )
        return hue, saturation, max_channel

    @staticmethod
    def _circular_hue_distance(hue: Tensor, center: float) -> Tensor:
        distance = (hue - float(center)).abs()
        return torch.minimum(distance, 1.0 - distance)

    def _build_score_map(self, image: Tensor) -> Tensor:
        image_float = (image.to(dtype=torch.float32) * self.rgb_std.to(image.device)) + self.rgb_mean.to(image.device)
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.score_size:
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.score_size, self.score_size),
            )
        hue, saturation, value = self._rgb_to_hsv_maps(image_float)
        luminance = 0.299 * image_float[:, 0] + 0.587 * image_float[:, 1] + 0.114 * image_float[:, 2]
        local_mean = F.avg_pool2d(luminance[:, None], kernel_size=5, stride=1, padding=2).squeeze(1)
        local_contrast = self._normalize_map((luminance - local_mean).abs())
        grad_x = F.pad((luminance[:, :, 1:] - luminance[:, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((luminance[:, 1:, :] - luminance[:, :-1, :]).abs(), (0, 0, 0, 1))
        edge_detail = self._normalize_map(local_contrast + 0.5 * (grad_x + grad_y))
        fill_distance = (
            image_float - self.rgb_mean.to(device=image_float.device, dtype=image_float.dtype)
        ).abs().mean(dim=1)
        non_padding = (fill_distance > 0.035).to(dtype=image_float.dtype)
        green_yellow = (
            (hue >= 0.08)
            & (hue <= 0.45)
            & (saturation >= 0.07)
            & (value >= 0.12)
        ).to(dtype=image_float.dtype)
        brown_or_orange = (
            (hue >= 0.035)
            & (hue <= 0.17)
            & (saturation >= 0.10)
            & (value >= 0.10)
        ).to(dtype=image_float.dtype)
        dark_detail = torch.sigmoid((0.42 - value) * 14.0) * torch.sigmoid((saturation - 0.08) * 10.0)
        brown_hue = torch.exp(-0.5 * (self._circular_hue_distance(hue, 0.10) / 0.09).pow(2))
        brown_spot = brown_hue * torch.sigmoid((0.68 - value) * 10.0) * torch.sigmoid((saturation - 0.10) * 10.0)
        fruit_support = torch.maximum(green_yellow, brown_or_orange)
        height, width = int(image_float.shape[-2]), int(image_float.shape[-1])
        y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image_float.dtype).view(1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image_float.dtype).view(1, 1, width)
        center_prior = torch.exp(-((x * x) + (y * y)) / (2.0 * 0.75 * 0.75))
        score = (
            0.55 * edge_detail
            + 0.30 * fruit_support
            + 0.20 * dark_detail * torch.maximum(fruit_support, non_padding)
            + 0.05 * center_prior
        ) * non_padding
        fallback = (edge_detail + 0.05 * center_prior) * non_padding
        interior_defect_modes = {"defect_spot_interior", "defect_interior", "interior_defect"}
        if self.score_mode in {"defect", "defect_spot", "spot", "dark_brown"} | interior_defect_modes:
            support = torch.maximum(fruit_support, non_padding * center_prior)
            if self.score_mode in interior_defect_modes:
                chroma_support = (
                    torch.sigmoid((saturation - 0.055) * 18.0)
                    * torch.sigmoid((value - 0.08) * 12.0)
                    * non_padding
                )
                support_soft = torch.maximum(fruit_support, 0.65 * chroma_support)
                support_smooth = F.avg_pool2d(support_soft[:, None], kernel_size=5, stride=1, padding=2).squeeze(1)
                interior_prior = self._soft_erode_map(support_smooth.clamp(0.0, 1.0), kernel_size=7)
                interior_prior = (0.75 * interior_prior + 0.25 * support_smooth).clamp(0.0, 1.0)
                frame_margin = torch.minimum(
                    torch.minimum((x + 1.0) * 0.5, (1.0 - x) * 0.5),
                    torch.minimum((y + 1.0) * 0.5, (1.0 - y) * 0.5),
                )
                frame_prior = torch.sigmoid((frame_margin - 0.055) * 38.0)
                interior_prior = interior_prior * frame_prior * (0.45 + 0.55 * center_prior)
                defect_core = torch.maximum(brown_spot, dark_detail * torch.maximum(brown_hue, fruit_support))
                defect_score = (
                    0.68 * brown_spot
                    + 0.48 * dark_detail * torch.maximum(brown_hue, fruit_support)
                    + 0.20 * local_contrast * defect_core
                    + 0.06 * center_prior
                ) * interior_prior * non_padding
            else:
                defect_score = (
                    0.62 * dark_detail
                    + 0.52 * brown_spot
                    + 0.18 * edge_detail * torch.maximum(dark_detail, brown_spot)
                    + 0.04 * center_prior
                ) * support * non_padding
            has_defect = defect_score.flatten(1).amax(dim=1) > 0.08
            score = torch.where(has_defect[:, None, None], defect_score, score)
        empty = score.flatten(1).amax(dim=1) <= 1e-6
        score = torch.where(empty[:, None, None], fallback, score)
        return self._normalize_map(score)

    def _crop_images(self, image: Tensor, score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, _, height, width = image.shape
        score_height, score_width = int(score.shape[-2]), int(score.shape[-1])
        flat_index = score.flatten(1).argmax(dim=1)
        score_y = torch.div(flat_index, score_width, rounding_mode="floor")
        score_x = flat_index.remainder(score_width)
        center_y = ((score_y.float() + 0.5) / float(score_height) * float(height)).round().long()
        center_x = ((score_x.float() + 0.5) / float(score_width) * float(width)).round().long()
        side = max(8, int(round(float(min(height, width)) * self.crop_scale)))
        half = max(4, side // 2)
        crops: List[Tensor] = []
        boxes: List[Tensor] = []
        stats: List[Tensor] = []
        for batch_index in range(batch_size):
            cy = int(center_y[batch_index].detach().cpu().item())
            cx = int(center_x[batch_index].detach().cpu().item())
            y0 = max(0, min(height - 1, cy - half))
            x0 = max(0, min(width - 1, cx - half))
            y1 = min(height, y0 + side)
            x1 = min(width, x0 + side)
            y0 = max(0, y1 - side)
            x0 = max(0, x1 - side)
            crop = image[batch_index : batch_index + 1, :, y0:y1, x0:x1]
            crops.append(
                F.interpolate(
                    crop,
                    size=(self.crop_size, self.crop_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            boxes.append(
                image.new_tensor(
                    [
                        float(x0) / float(max(1, width)),
                        float(y0) / float(max(1, height)),
                        float(x1) / float(max(1, width)),
                        float(y1) / float(max(1, height)),
                    ]
                )
            )
            stats.append(
                image.new_tensor(
                    [
                        float(cx) / float(max(1, width)),
                        float(cy) / float(max(1, height)),
                        float(x1 - x0) / float(max(1, width)),
                        float(y1 - y0) / float(max(1, height)),
                        float(score[batch_index].max().detach().cpu().item()),
                        float(score[batch_index].mean().detach().cpu().item()),
                    ]
                )
            )
        return torch.cat(crops, dim=0), torch.stack(boxes, dim=0), torch.stack(stats, dim=0)

    def forward(self, image: Tensor, *, return_trace: bool = False):
        score = self._build_score_map(image)
        crops, boxes, stats = self._crop_images(image, score)
        crop_descriptor = self.encoder(crops)
        descriptor = torch.cat(
            (crop_descriptor, stats.to(device=crop_descriptor.device, dtype=crop_descriptor.dtype)),
            dim=1,
        )
        logits = self.head(descriptor)
        if not return_trace:
            return logits
        return logits, {
            "score_map": score[:, None].detach(),
            "crop_boxes": boxes.detach(),
            "descriptor": descriptor.detach(),
        }


class HighFrequencyTextureExpert(nn.Module):
    """Raw-image expert for tiny texture/edge differences before patchification."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.08,
        analysis_size: int = 96,
    ) -> None:
        super().__init__()
        self.analysis_size = max(32, int(analysis_size))
        hidden = max(32, int(hidden_dim))
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        self.stats_dim = 34
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + self.stats_dim),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden + self.stats_dim, int(num_classes)),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _normalize_map(values: Tensor) -> Tensor:
        flat = values.flatten(1)
        minimum = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
        maximum = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
        return (values - minimum) / (maximum - minimum).clamp(min=1e-6)

    @staticmethod
    def _rgb_to_hsv_maps(image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]
        max_channel, max_index = image.max(dim=1)
        min_channel, _ = image.min(dim=1)
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
        saturation = torch.where(
            max_channel > eps,
            delta / max_channel.clamp(min=eps),
            torch.zeros_like(max_channel),
        )
        return hue, saturation, max_channel

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
        return (values * weights).flatten(1).sum(dim=1) / weights.flatten(1).sum(dim=1).clamp(min=1e-6)

    @staticmethod
    def _top_fraction_mean(values: Tensor, fraction: float = 0.05) -> Tensor:
        flat = values.flatten(1)
        k = max(1, int(math.ceil(float(flat.size(1)) * float(fraction))))
        return torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)

    def _prepare_image(self, image: Tensor) -> Tensor:
        image_float = (image.to(dtype=torch.float32) * self.rgb_std.to(image.device)) + self.rgb_mean.to(image.device)
        image_float = image_float.clamp(0.0, 1.0)
        if max(int(image_float.shape[-2]), int(image_float.shape[-1])) > self.analysis_size:
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.analysis_size, self.analysis_size),
            )
        return image_float

    def _texture_maps(self, image: Tensor) -> Dict[str, Tensor]:
        image_float = self._prepare_image(image)
        hue, saturation, value = self._rgb_to_hsv_maps(image_float)
        luminance = (
            0.299 * image_float[:, 0:1]
            + 0.587 * image_float[:, 1:2]
            + 0.114 * image_float[:, 2:3]
        )
        local_mean = F.avg_pool2d(luminance, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d((luminance - local_mean).pow(2), kernel_size=5, stride=1, padding=2)
        local_std = torch.sqrt(local_var.clamp(min=1e-8))
        high_pass = (luminance - local_mean).abs()
        grad_x = F.pad((luminance[:, :, :, 1:] - luminance[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((luminance[:, :, 1:, :] - luminance[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        gradient = torch.sqrt((grad_x * grad_x + grad_y * grad_y).clamp(min=1e-8))
        laplacian = (
            4.0 * luminance
            - F.pad(luminance[:, :, :, 1:], (0, 1, 0, 0))
            - F.pad(luminance[:, :, :, :-1], (1, 0, 0, 0))
            - F.pad(luminance[:, :, 1:, :], (0, 0, 0, 1))
            - F.pad(luminance[:, :, :-1, :], (0, 0, 1, 0))
        ).abs()

        fill_distance = (
            image_float - self.rgb_mean.to(device=image_float.device, dtype=image_float.dtype)
        ).abs().mean(dim=1)
        non_padding = (fill_distance > 0.035).to(dtype=image_float.dtype)
        green_yellow = (
            (hue >= 0.08)
            & (hue <= 0.45)
            & (saturation >= 0.07)
            & (value >= 0.12)
        ).to(dtype=image_float.dtype)
        brown_or_orange = (
            (hue >= 0.035)
            & (hue <= 0.17)
            & (saturation >= 0.10)
            & (value >= 0.10)
        ).to(dtype=image_float.dtype)
        dark_detail = torch.sigmoid((0.42 - value) * 14.0) * torch.sigmoid((saturation - 0.08) * 10.0)
        bright_detail = torch.sigmoid((value - 0.88) * 14.0) * torch.sigmoid((0.24 - saturation) * 10.0)
        fruit_support = torch.maximum(torch.maximum(green_yellow, brown_or_orange), dark_detail)
        height, width = int(image_float.shape[-2]), int(image_float.shape[-1])
        y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image_float.dtype).view(1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image_float.dtype).view(1, 1, width)
        center_prior = torch.exp(-((x * x) + (y * y)) / (2.0 * 0.72 * 0.72))
        foreground = (0.65 * fruit_support + 0.25 * non_padding + 0.10 * center_prior).clamp(0.0, 1.0)
        foreground = foreground.unsqueeze(1)

        normalized_high = self._normalize_map(high_pass.squeeze(1)).unsqueeze(1)
        normalized_gradient = self._normalize_map(gradient.squeeze(1)).unsqueeze(1)
        normalized_lap = self._normalize_map(laplacian.squeeze(1)).unsqueeze(1)
        normalized_std = self._normalize_map(local_std.squeeze(1)).unsqueeze(1)
        dark_map = dark_detail.unsqueeze(1)
        bright_map = bright_detail.unsqueeze(1)
        foreground_detail = (
            0.35 * normalized_high
            + 0.25 * normalized_gradient
            + 0.20 * normalized_lap
            + 0.20 * normalized_std
        ) * (0.20 + 0.80 * foreground)
        return {
            "input": torch.cat(
                (
                    normalized_high,
                    normalized_gradient,
                    normalized_lap,
                    normalized_std,
                    dark_map,
                    bright_map,
                    foreground_detail,
                ),
                dim=1,
            ),
            "high_pass": normalized_high,
            "gradient": normalized_gradient,
            "laplacian": normalized_lap,
            "local_std": normalized_std,
            "dark_detail": dark_map,
            "bright_detail": bright_map,
            "foreground_weight": foreground,
            "foreground_detail": foreground_detail,
        }

    def _stats_from_maps(self, maps: Dict[str, Tensor]) -> Tensor:
        foreground = maps["foreground_weight"].clamp(min=0.0)
        global_weight = torch.full_like(foreground, 1.0 / max(1, foreground.shape[-2] * foreground.shape[-1]))
        foreground_weight = foreground / foreground.flatten(1).sum(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-6)
        stats: List[Tensor] = []
        for key in ("high_pass", "gradient", "laplacian", "local_std", "dark_detail", "bright_detail", "foreground_detail"):
            values = maps[key]
            stats.extend(
                [
                    self._weighted_mean(values, global_weight),
                    self._weighted_mean(values, foreground_weight),
                    values.flatten(1).amax(dim=1),
                    self._top_fraction_mean(values),
                ]
            )
        stats.extend(
            [
                foreground.flatten(1).mean(dim=1),
                foreground.flatten(1).amax(dim=1),
                self._weighted_mean(maps["dark_detail"], foreground_weight),
                self._weighted_mean(maps["bright_detail"], foreground_weight),
                self._weighted_mean(maps["gradient"], foreground_weight)
                - self._weighted_mean(maps["high_pass"], foreground_weight),
                self._weighted_mean(maps["laplacian"], foreground_weight)
                - self._weighted_mean(maps["local_std"], foreground_weight),
            ]
        )
        return torch.stack(stats, dim=1)

    def forward(self, image: Tensor, *, return_trace: bool = False):
        maps = self._texture_maps(image)
        encoded = self.encoder(maps["input"].to(dtype=image.dtype))
        stats = self._stats_from_maps(maps).to(device=encoded.device, dtype=encoded.dtype)
        descriptor = torch.cat((encoded, stats), dim=1)
        logits = self.head(descriptor)
        if not return_trace:
            return logits
        return logits, {
            "descriptor": descriptor.detach(),
            "high_pass": maps["high_pass"].detach(),
            "gradient": maps["gradient"].detach(),
            "laplacian": maps["laplacian"].detach(),
            "foreground_weight": maps["foreground_weight"].detach(),
            "foreground_detail": maps["foreground_detail"].detach(),
        }


class BBoxSpatialPriorFusion(nn.Module):
    """Residual logits from YOLO object position/scale metadata."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.stats_dim = 18
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

    def extract_stats(self, bbox: Tensor) -> Tensor:
        if bbox.ndim == 3:
            bbox = bbox[:, 0]
        if bbox.ndim != 2 or int(bbox.size(1)) < 4:
            raise ValueError("BBoxSpatialPriorFusion expects bbox shaped [B, 4].")
        bbox_float = bbox[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
        x_center, y_center, box_width, box_height = bbox_float.unbind(dim=1)
        half_width = box_width * 0.5
        half_height = box_height * 0.5
        left = (x_center - half_width).clamp(0.0, 1.0)
        top = (y_center - half_height).clamp(0.0, 1.0)
        right = (x_center + half_width).clamp(0.0, 1.0)
        bottom = (y_center + half_height).clamp(0.0, 1.0)
        area = (box_width * box_height).clamp(0.0, 1.0)
        sqrt_area = torch.sqrt(area.clamp(min=1e-8))
        aspect_log = torch.log((box_width + 1e-6) / (box_height + 1e-6)).clamp(-4.0, 4.0) / 4.0
        x_offset = x_center - 0.5
        y_offset = y_center - 0.5
        center_distance = torch.sqrt((x_offset * x_offset + y_offset * y_offset).clamp(min=0.0)) / math.sqrt(0.5)
        border_min = torch.stack((left, top, 1.0 - right, 1.0 - bottom), dim=1).amin(dim=1)
        border_max = torch.stack((left, top, 1.0 - right, 1.0 - bottom), dim=1).amax(dim=1)
        angle = 2.0 * math.pi
        stats = torch.stack(
            (
                x_center,
                y_center,
                box_width,
                box_height,
                left,
                top,
                right,
                bottom,
                area,
                sqrt_area,
                aspect_log,
                x_offset,
                y_offset,
                center_distance.clamp(0.0, 1.0),
                border_min.clamp(0.0, 1.0),
                border_max.clamp(0.0, 1.0),
                torch.sin(x_center * angle),
                torch.cos(y_center * angle),
            ),
            dim=1,
        )
        return stats

    def forward(self, bbox: Tensor, *, dtype: torch.dtype, return_trace: bool = False):
        stats = self.extract_stats(bbox)
        logits = self.net(stats.to(device=bbox.device, dtype=dtype))
        if not return_trace:
            return logits
        return logits, {"stats": stats.detach()}


class SourceContextFeatureFusion(nn.Module):
    """Residual logits from paired object-crop and source-context features."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_dim = int(feature_dim) * 4 + int(num_classes) * 2
        hidden = max(16, int(hidden_dim))
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.delta = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, int(num_classes)),
        )
        self.gate_bias = float(gate_bias)

    def zero_init_residual(self) -> None:
        delta_final = self.delta[-1]
        if isinstance(delta_final, nn.Linear):
            nn.init.zeros_(delta_final.weight)
            if delta_final.bias is not None:
                nn.init.zeros_(delta_final.bias)
        gate_final = self.gate[-1]
        if isinstance(gate_final, nn.Linear):
            nn.init.zeros_(gate_final.weight)
            if gate_final.bias is not None:
                nn.init.constant_(gate_final.bias, self.gate_bias)

    def forward(
        self,
        primary_feature: Tensor,
        context_feature: Tensor,
        primary_logits: Tensor,
        context_logits: Tensor,
        *,
        return_trace: bool = False,
    ):
        if primary_feature.ndim != 2 or context_feature.ndim != 2:
            raise ValueError("SourceContextFeatureFusion expects [B, D] features.")
        if primary_feature.shape != context_feature.shape:
            raise ValueError(
                "Primary/context feature shape mismatch: "
                f"{tuple(primary_feature.shape)} vs {tuple(context_feature.shape)}"
            )
        if primary_logits.shape != context_logits.shape:
            raise ValueError(
                "Primary/context logits shape mismatch: "
                f"{tuple(primary_logits.shape)} vs {tuple(context_logits.shape)}"
            )
        dtype = primary_feature.dtype
        fused = torch.cat(
            (
                primary_feature,
                context_feature,
                primary_feature - context_feature,
                primary_feature * context_feature,
                primary_logits.to(dtype=dtype),
                context_logits.to(dtype=dtype),
            ),
            dim=1,
        )
        fused = self.input_norm(fused)
        delta = self.delta(fused)
        gate = torch.sigmoid(self.gate(fused))
        adjustment = delta * gate
        if not return_trace:
            return adjustment
        return adjustment, {
            "delta": delta.detach(),
            "gate": gate.detach(),
            "adjustment": adjustment.detach(),
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
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.max_stats_size, self.max_stats_size),
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
        lab_chroma = torch.sqrt(
            (lab[:, 1] ** 2 + lab[:, 2] ** 2).clamp(min=1e-8)
        )
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
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.max_stats_size, self.max_stats_size),
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
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.max_stats_size, self.max_stats_size),
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
                "image_float": image_float,
                "normalized_image": normalized_image,
                "hue": hue,
                "saturation": saturation,
                "value": value,
                "hue_norm": hue_norm,
                "saturation_norm": saturation_norm,
                "value_norm": value_norm,
                "local_contrast": local_contrast,
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


class ForegroundSurfacePairwiseHead(nn.Module):
    """Pairwise boundary logits from foreground-only color/defect statistics."""

    def __init__(
        self,
        pair_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.pair_count = max(0, int(pair_count))
        self.stats_extractor = ForegroundSurfaceStatisticFusion(
            num_classes=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        hidden = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_extractor.stats_dim),
            nn.Linear(self.stats_extractor.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, self.pair_count),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def forward_from_stats(self, stats: Tensor, dtype: torch.dtype) -> Tensor:
        return self.net(stats.to(dtype=dtype))

    def forward(self, image: Tensor, return_trace: bool = False):
        stats, maps = self.stats_extractor._extract_stats_and_maps(image)
        logits = self.forward_from_stats(stats, image.dtype)
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


class InteriorBoundaryPairwiseHead(nn.Module):
    """Pairwise logits from object interior versus boundary-ring surface stats."""

    def __init__(
        self,
        pair_count: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        erode_kernel: int = 9,
        foreground_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.pair_count = max(0, int(pair_count))
        self.erode_kernel = max(3, int(erode_kernel) | 1)
        self.foreground_threshold = float(max(0.0, foreground_threshold))
        self.stats_extractor = ForegroundSurfaceStatisticFusion(
            num_classes=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.hist_bins = 8
        self.stats_dim = 234
        hidden = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, self.pair_count),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.net[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    @staticmethod
    def _normalize_weights(weights: Tensor) -> Tensor:
        return weights / weights.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _bbox_mask(
        bbox: Optional[Tensor],
        *,
        reference: Tensor,
        margin_ratio: float = 0.03,
    ) -> Optional[Tensor]:
        if not torch.is_tensor(bbox) or bbox.ndim != 2 or int(bbox.size(1)) < 4:
            return None
        height, width = [int(value) for value in reference.shape[-2:]]
        if height <= 0 or width <= 0:
            return None
        bbox_value = bbox[:, :4].to(device=reference.device, dtype=reference.dtype)
        center_x = bbox_value[:, 0].clamp(0.0, 1.0)
        center_y = bbox_value[:, 1].clamp(0.0, 1.0)
        box_width = bbox_value[:, 2].clamp(0.0, 1.0)
        box_height = bbox_value[:, 3].clamp(0.0, 1.0)
        margin = float(max(0.0, margin_ratio))
        left = (center_x - 0.5 * box_width - margin).clamp(0.0, 1.0)
        right = (center_x + 0.5 * box_width + margin).clamp(0.0, 1.0)
        top = (center_y - 0.5 * box_height - margin).clamp(0.0, 1.0)
        bottom = (center_y + 0.5 * box_height + margin).clamp(0.0, 1.0)
        ys = torch.linspace(
            0.5 / float(height),
            1.0 - 0.5 / float(height),
            steps=height,
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, height, 1)
        xs = torch.linspace(
            0.5 / float(width),
            1.0 - 0.5 / float(width),
            steps=width,
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, 1, width)
        x_mask = (xs >= left[:, None, None]) & (xs <= right[:, None, None])
        y_mask = (ys >= top[:, None, None]) & (ys <= bottom[:, None, None])
        mask = (x_mask & y_mask).to(dtype=reference.dtype)
        valid = (box_width > 1e-4) & (box_height > 1e-4)
        return mask * valid.to(dtype=reference.dtype).view(-1, 1, 1)

    def _interior_boundary_weights(
        self,
        maps: Dict[str, Tensor],
        bbox: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        foreground_raw = maps["foreground_raw"].to(dtype=torch.float32).clamp(min=0.0)
        non_padding = maps["non_padding"].to(dtype=foreground_raw.dtype)
        center_prior = maps["center_prior"].to(dtype=foreground_raw.dtype)
        saturation = maps["saturation"].to(dtype=foreground_raw.dtype).clamp(0.0, 1.0)
        dark_spot = maps["dark_spot"].to(dtype=foreground_raw.dtype).clamp(0.0, 1.0)
        brown_spot = maps["brown_spot"].to(dtype=foreground_raw.dtype).clamp(0.0, 1.0)
        saturation_gate = torch.sigmoid((saturation - 0.28) * 18.0)
        objectness_gate = torch.maximum(saturation_gate, 0.35 * center_prior)
        objectness_gate = torch.maximum(objectness_gate, 0.80 * dark_spot)
        objectness_gate = torch.maximum(objectness_gate, 0.80 * brown_spot)
        foreground_raw = foreground_raw * objectness_gate
        bbox_mask = self._bbox_mask(bbox, reference=foreground_raw)
        if bbox_mask is not None:
            bbox_valid = bbox_mask.flatten(1).sum(dim=1).view(-1, 1, 1) > 1e-6
            foreground_raw = torch.where(bbox_valid, foreground_raw * bbox_mask, foreground_raw)
            non_padding = torch.where(bbox_valid, non_padding * bbox_mask, non_padding)
            center_prior = torch.where(bbox_valid, center_prior * bbox_mask, center_prior)
        foreground_mask = (foreground_raw > self.foreground_threshold).to(dtype=foreground_raw.dtype)
        foreground_mask = foreground_mask * non_padding
        empty = foreground_mask.flatten(1).sum(dim=1).view(-1, 1, 1) <= 1e-6
        fallback_mask = non_padding
        foreground_mask = torch.where(empty, fallback_mask, foreground_mask)
        kernel = int(self.erode_kernel)
        padding = kernel // 2
        eroded = 1.0 - F.max_pool2d(
            (1.0 - foreground_mask).unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ).squeeze(1)
        eroded = eroded.clamp(min=0.0, max=1.0) * foreground_mask
        boundary = (foreground_mask - eroded).clamp(min=0.0)

        interior_raw = foreground_raw * eroded
        boundary_raw = foreground_raw * boundary
        interior_empty = interior_raw.flatten(1).sum(dim=1).view(-1, 1, 1) <= 1e-6
        boundary_empty = boundary_raw.flatten(1).sum(dim=1).view(-1, 1, 1) <= 1e-6
        interior_fallback = foreground_raw * center_prior
        boundary_fallback = foreground_raw * (1.0 - center_prior).clamp(min=0.0)
        interior_raw = torch.where(interior_empty, interior_fallback, interior_raw)
        boundary_raw = torch.where(boundary_empty, boundary_fallback, boundary_raw)
        interior_weights = self._normalize_weights(interior_raw.unsqueeze(1))
        boundary_weights = self._normalize_weights(boundary_raw.unsqueeze(1))
        trace = {
            "foreground_mask": foreground_mask,
            "interior_mask": (interior_raw > 0.0).to(dtype=foreground_raw.dtype),
            "boundary_mask": (boundary_raw > 0.0).to(dtype=foreground_raw.dtype),
            "interior_weight_map": interior_weights,
            "boundary_weight_map": boundary_weights,
        }
        if bbox_mask is not None:
            trace["bbox_mask"] = bbox_mask
        return interior_weights, boundary_weights, trace

    def _score_pack(self, values: Tensor, weights: Tensor) -> Tensor:
        return self.stats_extractor._weighted_scalar_pack(values, weights, threshold=0.5)

    def _mask_fraction(self, mask: Tensor, weights: Tensor) -> Tensor:
        return self.stats_extractor._weighted_mask_fraction(mask, weights)

    def _extract_stats_and_maps(
        self,
        image: Tensor,
        bbox: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, maps = self.stats_extractor._extract_stats_and_maps(image)
        interior_weights, boundary_weights, trace_maps = self._interior_boundary_weights(
            maps,
            bbox=bbox,
        )
        image_float = maps["image_float"]
        normalized_image = maps["normalized_image"]
        lab = self.stats_extractor._rgb_to_lab(image_float)
        lab_norm = self.stats_extractor._rgb_to_lab(normalized_image)

        foreground_raw = maps["foreground_raw"].to(dtype=image_float.dtype)
        non_padding = maps["non_padding"].to(dtype=image_float.dtype)
        foreground_area = (foreground_raw > self.foreground_threshold).to(dtype=image_float.dtype).flatten(1).mean(dim=1)
        interior_area = trace_maps["interior_mask"].flatten(1).mean(dim=1)
        boundary_area = trace_maps["boundary_mask"].flatten(1).mean(dim=1)
        non_padding_area = non_padding.flatten(1).mean(dim=1)
        area_stats = torch.stack(
            (
                non_padding_area,
                foreground_area,
                interior_area,
                boundary_area,
                interior_area / foreground_area.clamp(min=1e-6),
                boundary_area / foreground_area.clamp(min=1e-6),
            ),
            dim=1,
        )

        interior_rgb_mean, interior_rgb_std = self.stats_extractor._weighted_channel_mean_std(
            image_float,
            interior_weights,
        )
        boundary_rgb_mean, boundary_rgb_std = self.stats_extractor._weighted_channel_mean_std(
            image_float,
            boundary_weights,
        )
        rgb_stats = torch.cat(
            (
                interior_rgb_mean,
                boundary_rgb_mean,
                interior_rgb_mean - boundary_rgb_mean,
                interior_rgb_std,
                boundary_rgb_std,
                interior_rgb_std - boundary_rgb_std,
            ),
            dim=1,
        )

        interior_lab_mean, interior_lab_std = self.stats_extractor._weighted_channel_mean_std(
            lab,
            interior_weights,
        )
        boundary_lab_mean, boundary_lab_std = self.stats_extractor._weighted_channel_mean_std(
            lab,
            boundary_weights,
        )
        lab_stats = torch.cat(
            (
                interior_lab_mean,
                boundary_lab_mean,
                interior_lab_mean - boundary_lab_mean,
                interior_lab_std,
                boundary_lab_std,
                interior_lab_std - boundary_lab_std,
            ),
            dim=1,
        )

        raw_hsv_interior = self.stats_extractor._rgb_to_hsv_summary(image_float, interior_weights)
        raw_hsv_boundary = self.stats_extractor._rgb_to_hsv_summary(image_float, boundary_weights)
        norm_hsv_interior = self.stats_extractor._rgb_to_hsv_summary(
            normalized_image,
            interior_weights,
        )
        norm_hsv_boundary = self.stats_extractor._rgb_to_hsv_summary(
            normalized_image,
            boundary_weights,
        )
        hsv_stats = torch.cat(
            (
                raw_hsv_interior,
                raw_hsv_boundary,
                raw_hsv_interior - raw_hsv_boundary,
                norm_hsv_interior,
                norm_hsv_boundary,
                norm_hsv_interior - norm_hsv_boundary,
            ),
            dim=1,
        )

        score_stats = []
        score_sources = (
            maps["edge_detail"],
            maps["dark_spot"],
            maps["brown_spot"],
            maps["bright_spot"],
            maps["local_contrast"],
        )
        for score in score_sources:
            interior_pack = self._score_pack(score, interior_weights)
            boundary_pack = self._score_pack(score, boundary_weights)
            score_stats.extend((interior_pack, boundary_pack, interior_pack - boundary_pack))

        color_ratios = []
        for mask in (
            maps["green_yellow"],
            maps["yellow_band"],
            maps["brown_or_orange"],
            maps["dark_defect"],
        ):
            interior_fraction = self._mask_fraction(mask, interior_weights)
            boundary_fraction = self._mask_fraction(mask, boundary_weights)
            color_ratios.extend(
                (
                    interior_fraction[:, None],
                    boundary_fraction[:, None],
                    (interior_fraction - boundary_fraction)[:, None],
                )
            )

        histograms = []
        for values in (
            maps["value"],
            maps["saturation"],
            maps["edge_detail"],
            maps["dark_spot"],
            maps["brown_spot"],
            maps["bright_spot"],
        ):
            histograms.append(
                self.stats_extractor._soft_histogram(
                    values,
                    interior_weights,
                    bins=self.hist_bins,
                    min_value=0.0,
                    max_value=1.0,
                )
            )
            histograms.append(
                self.stats_extractor._soft_histogram(
                    values,
                    boundary_weights,
                    bins=self.hist_bins,
                    min_value=0.0,
                    max_value=1.0,
                )
            )

        stats = torch.cat(
            (
                area_stats,
                rgb_stats,
                lab_stats,
                hsv_stats,
                *score_stats,
                *color_ratios,
                *histograms,
            ),
            dim=1,
        )
        if int(stats.size(1)) != self.stats_dim:
            raise RuntimeError(
                f"InteriorBoundaryPairwiseHead stats_dim mismatch: {int(stats.size(1))} != {self.stats_dim}"
            )
        trace_maps.update(
            {
                "stats": stats,
                "edge_detail": maps["edge_detail"] * trace_maps["foreground_mask"],
                "boundary_edge_detail": maps["edge_detail"] * trace_maps["boundary_mask"],
                "interior_dark_spot": maps["dark_spot"] * trace_maps["interior_mask"],
                "boundary_brown_spot": maps["brown_spot"] * trace_maps["boundary_mask"],
            }
        )
        return stats, trace_maps

    def forward(self, image: Tensor, bbox: Optional[Tensor] = None, return_trace: bool = False):
        stats, trace_maps = self._extract_stats_and_maps(image, bbox=bbox)
        logits = self.net(stats.to(dtype=image.dtype))
        if not return_trace:
            return logits
        return logits, {key: value.detach() for key, value in trace_maps.items()}


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
            image_float = _exportable_adaptive_avg_pool2d(
                image_float,
                output_size=(self.max_stats_size, self.max_stats_size),
            )
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
        pooled = _exportable_adaptive_avg_pool2d(magnitude, output_size=(4, 4)).flatten(1)
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
        pooled = _exportable_adaptive_avg_pool2d(stem_features, output_size=1).flatten(1)
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
        detail_grid = _exportable_adaptive_avg_pool2d(detail_input, output_size=grid_size)
        detail_tokens = self.proj(detail_grid).flatten(2).transpose(1, 2)
        detail_tokens = self.dropout(self.norm(detail_tokens))
        scale = self.residual_scale.to(dtype=detail_tokens.dtype).clamp(0.0, 1.0)
        detail_map = _exportable_adaptive_avg_pool2d(
            high_frequency.abs().mean(dim=1, keepdim=True) + edge_magnitude,
            output_size=grid_size,
        )
        return detail_tokens * scale, detail_map


class GatedPatchRelativePositionAttention(nn.Module):
    """GPSA-style positional mixing restricted to patch-to-patch attention."""

    def __init__(
        self,
        num_heads: int,
        max_mix: float = 0.25,
        locality_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be positive.")
        if not 0.0 < float(max_mix) <= 1.0:
            raise ValueError("max_mix must be in (0, 1].")
        if float(locality_strength) <= 0.0:
            raise ValueError("locality_strength must be positive.")
        self.num_heads = int(num_heads)
        self.max_mix = float(max_mix)
        self.locality_strength = float(locality_strength)
        self.position_projection = nn.Linear(3, self.num_heads, bias=False)
        self.gate = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self._last_trace: Dict[str, Tensor] = {}
        self.reset_locality_parameters()

    def reset_locality_parameters(self) -> None:
        angles = torch.arange(self.num_heads, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / float(self.num_heads))
        centers = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        strength = float(self.locality_strength)
        with torch.no_grad():
            self.position_projection.weight.zero_()
            self.position_projection.weight[:, 0].copy_(2.0 * strength * centers[:, 0])
            self.position_projection.weight[:, 1].copy_(2.0 * strength * centers[:, 1])
            self.position_projection.weight[:, 2].fill_(-strength)
            self.gate.zero_()

    def effective_gate(self) -> Tensor:
        clamped = self.gate.clamp(min=0.0, max=self.max_mix)
        if self.training:
            # Straight-through projection keeps the forward attention a valid
            # convex mixture while avoiding a one-batch dead gate below zero.
            return self.gate + (clamped - self.gate).detach()
        return clamped

    @staticmethod
    def _patch_coordinates(
        *,
        attention: Tensor,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor],
    ) -> Optional[Tensor]:
        if grid_size is None:
            return None
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        patch_count = int(attention.size(-1)) - int(prefix_count)
        if patch_count <= 0:
            return None
        batch_size = int(attention.size(0))
        grid_patch_count = grid_height * grid_width
        if torch.is_tensor(patch_indices):
            if (
                patch_indices.ndim != 2
                or int(patch_indices.size(0)) != batch_size
                or int(patch_indices.size(1)) != patch_count
            ):
                return None
            indices = patch_indices.to(device=attention.device, dtype=torch.long)
        elif patch_count == grid_patch_count:
            indices = torch.arange(
                grid_patch_count,
                device=attention.device,
                dtype=torch.long,
            ).view(1, -1).expand(batch_size, -1)
        else:
            return None
        if bool(((indices < 0) | (indices >= grid_patch_count)).any().item()):
            return None
        x = torch.remainder(indices, grid_width).to(dtype=torch.float32)
        y = torch.div(indices, grid_width, rounding_mode="floor").to(dtype=torch.float32)
        return torch.stack((x, y), dim=-1)

    def forward(
        self,
        attention: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor] = None,
    ) -> Tensor:
        if attention.ndim != 4 or int(attention.size(1)) != self.num_heads:
            return attention
        token_count = int(attention.size(-1))
        if int(attention.size(-2)) != token_count:
            return attention
        prefix_count = max(0, min(int(prefix_count), token_count))
        coordinates = self._patch_coordinates(
            attention=attention,
            grid_size=grid_size,
            prefix_count=prefix_count,
            patch_indices=patch_indices,
        )
        if coordinates is None:
            self._last_trace = {}
            return attention

        delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        distance_squared = delta.square().sum(dim=-1)
        relative_features = torch.stack(
            (delta[..., 0], delta[..., 1], distance_squared),
            dim=-1,
        ).to(dtype=self.position_projection.weight.dtype)
        positional_logits = self.position_projection(relative_features).permute(0, 3, 1, 2)
        positional_attention = positional_logits.float().softmax(dim=-1).to(
            dtype=attention.dtype
        )

        patch_attention = attention[:, :, prefix_count:, prefix_count:]
        patch_mass = patch_attention.sum(dim=-1, keepdim=True)
        positional_with_content_mass = positional_attention * patch_mass
        gate = self.effective_gate().to(
            device=attention.device,
            dtype=attention.dtype,
        ).view(1, self.num_heads, 1, 1)
        mixed_patch_attention = patch_attention + gate * (
            positional_with_content_mass - patch_attention
        )
        if prefix_count > 0:
            patch_rows = torch.cat(
                (attention[:, :, prefix_count:, :prefix_count], mixed_patch_attention),
                dim=-1,
            )
            mixed_attention = torch.cat(
                (attention[:, :, :prefix_count, :], patch_rows),
                dim=-2,
            )
        else:
            mixed_attention = mixed_patch_attention

        distance = distance_squared.clamp_min(0.0).sqrt().unsqueeze(1)
        content_distribution = patch_attention / patch_mass.clamp_min(1e-8)
        mixed_distribution = mixed_patch_attention / patch_mass.clamp_min(1e-8)
        expected_position_distance = (positional_attention.float() * distance).sum(dim=-1)
        expected_content_distance = (content_distribution.float() * distance).sum(dim=-1)
        expected_mixed_distance = (mixed_distribution.float() * distance).sum(dim=-1)
        local_mask = (delta.abs().amax(dim=-1) <= 1.0).unsqueeze(1)
        positional_local_mass = (
            positional_attention.float() * local_mask.to(dtype=torch.float32)
        ).sum(dim=-1)

        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        center = torch.tensor(
            [(grid_width - 1) * 0.5, (grid_height - 1) * 0.5],
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        center_query = (coordinates[0] - center).square().sum(dim=-1).argmin()
        center_values = positional_attention[0, :, center_query].float().mean(dim=0)
        full_center_map = center_values.new_zeros(grid_height * grid_width)
        if torch.is_tensor(patch_indices):
            center_indices = patch_indices[0].to(device=center_values.device, dtype=torch.long)
        else:
            center_indices = torch.arange(
                center_values.numel(),
                device=center_values.device,
                dtype=torch.long,
            )
        full_center_map.scatter_(0, center_indices, center_values)
        self._last_trace = {
            "gate": self.effective_gate().detach().float(),
            "raw_gate": self.gate.detach().float(),
            "position_distance": expected_position_distance.detach().mean(dim=(0, 2)),
            "content_distance": expected_content_distance.detach().mean(dim=(0, 2)),
            "mixed_distance": expected_mixed_distance.detach().mean(dim=(0, 2)),
            "position_local_mass": positional_local_mass.detach().mean(dim=(0, 2)),
            "center_map": full_center_map.detach(),
        }
        return mixed_attention

    def trace(self) -> Dict[str, Tensor]:
        return dict(self._last_trace)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        gated_relative_position_attention: bool = False,
        relative_position_max_mix: float = 0.25,
        relative_position_locality_strength: float = 1.0,
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
        self.relative_position_attention = (
            GatedPatchRelativePositionAttention(
                num_heads=num_heads,
                max_mix=relative_position_max_mix,
                locality_strength=relative_position_locality_strength,
            )
            if bool(gated_relative_position_attention)
            else None
        )

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
    ):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        if self.relative_position_attention is not None:
            attention = self.relative_position_attention(
                attention,
                grid_size=grid_size,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        attention = self.attention_dropout(attention)

        out = attention @ value
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        out = self.projection_dropout(out)
        if return_attention:
            return out, attention
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        patch_style_recalibration: bool = False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.style_recalibration = (
            PatchStyleRecalibration(hidden_dim)
            if bool(patch_style_recalibration)
            else None
        )

    def forward(self, x: Tensor, *, prefix_count: int = 0) -> Tensor:
        if self.style_recalibration is None:
            return self.net(x)
        hidden = self.net[0](x)
        hidden = self.net[1](hidden)
        hidden = self.style_recalibration(
            hidden,
            prefix_count=prefix_count,
        )
        hidden = self.net[2](hidden)
        hidden = self.net[3](hidden)
        return self.net[4](hidden)


class LocallyEnhancedFeedForward(nn.Module):
    """CeiT/LocalViT-style FFN with spatial depthwise mixing for patch tokens."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("LocallyEnhancedFeedForward kernel_size must be odd and >= 3.")
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.pre_activation = nn.GELU()
        self.pre_dropout = nn.Dropout(dropout)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.spatial_norm = nn.BatchNorm2d(hidden_dim)
        self.spatial_activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor] = None,
    ) -> Tensor:
        hidden = self.pre_dropout(self.pre_activation(self.fc1(tokens)))
        if tokens.ndim != 3 or grid_size is None:
            return self.output_dropout(self.fc2(hidden))

        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        grid_patch_count = grid_height * grid_width
        prefix_count = max(0, min(int(prefix_count), int(tokens.size(1))))
        prefix_hidden = hidden[:, :prefix_count]
        patch_hidden = hidden[:, prefix_count:]
        batch_size, patch_count, hidden_dim = patch_hidden.shape
        if patch_count <= 0:
            return self.output_dropout(self.fc2(hidden))

        gather_indices: Optional[Tensor] = None
        gather_valid: Optional[Tensor] = None
        if patch_count == grid_patch_count:
            grid_hidden = patch_hidden
        elif (
            torch.is_tensor(patch_indices)
            and patch_indices.ndim == 2
            and int(patch_indices.size(0)) == batch_size
            and int(patch_indices.size(1)) == patch_count
        ):
            raw_indices = patch_indices.to(device=tokens.device, dtype=torch.long)
            gather_valid = (raw_indices >= 0) & (raw_indices < grid_patch_count)
            gather_indices = raw_indices.clamp(min=0, max=max(0, grid_patch_count - 1))
            grid_hidden = patch_hidden.new_zeros(
                (batch_size, grid_patch_count, hidden_dim)
            )
            grid_hidden = grid_hidden.scatter(
                1,
                gather_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
                patch_hidden * gather_valid.unsqueeze(-1).to(dtype=patch_hidden.dtype),
            )
        else:
            return self.output_dropout(self.fc2(hidden))

        spatial = grid_hidden.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_height,
            grid_width,
        )
        spatial = self.spatial_activation(self.spatial_norm(self.depthwise(spatial)))
        spatial = spatial.flatten(2).transpose(1, 2)
        if gather_indices is not None:
            spatial = spatial.gather(
                1,
                gather_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            if gather_valid is not None:
                spatial = spatial * gather_valid.unsqueeze(-1).to(dtype=spatial.dtype)
        combined = torch.cat((prefix_hidden, spatial), dim=1) if prefix_count else spatial
        return self.output_dropout(self.fc2(combined))


class ConcurrentLocalInitializer(nn.Module):
    """Initialize a persistent local feature map at the transformer patch grid."""

    def __init__(self, in_channels: int, local_dim: int) -> None:
        super().__init__()
        if int(local_dim) <= 0:
            raise ValueError("ConcurrentLocalInitializer local_dim must be positive.")
        self.projection = nn.Sequential(
            nn.Conv2d(int(in_channels), int(local_dim), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(local_dim)),
            nn.GELU(),
        )

    def forward(self, stem_features: Tensor, grid_size: Tuple[int, int]) -> Tensor:
        if stem_features.ndim != 4:
            raise ValueError("ConcurrentLocalInitializer expects [B, C, H, W].")
        height, width = [max(1, int(value)) for value in grid_size]
        pooled = _exportable_adaptive_avg_pool2d(
            stem_features,
            output_size=(height, width),
        )
        return self.projection(pooled)


class ConcurrentLocalGlobalCoupling(nn.Module):
    """Conformer-style bidirectional coupling between local maps and patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        local_dim: int = 64,
        kernel_size: int = 3,
        expansion_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        local_dim = int(local_dim)
        kernel_size = int(kernel_size)
        if embed_dim <= 0 or local_dim <= 0:
            raise ValueError("Concurrent coupling dimensions must be positive.")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Concurrent coupling kernel_size must be odd and >= 3.")
        hidden_dim = max(local_dim, int(round(local_dim * float(expansion_ratio))))
        self.embed_dim = embed_dim
        self.local_dim = local_dim
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_to_local = nn.Linear(embed_dim, local_dim, bias=False)
        self.local_update = nn.Sequential(
            nn.Conv2d(
                local_dim,
                local_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=local_dim,
                bias=False,
            ),
            nn.BatchNorm2d(local_dim),
            nn.GELU(),
            nn.Conv2d(local_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(local_dim),
        )
        self.local_to_token = nn.Sequential(
            nn.Conv2d(local_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.token_to_local_scale = nn.Parameter(torch.tensor(0.50))
        self.local_update_scale = nn.Parameter(torch.tensor(0.50))
        self.local_to_token_scale = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _validated_indices(
        patch_indices: Optional[Tensor],
        *,
        batch_size: int,
        patch_count: int,
        grid_patch_count: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if patch_count == grid_patch_count:
            return None, None
        if (
            not torch.is_tensor(patch_indices)
            or patch_indices.ndim != 2
            or int(patch_indices.size(0)) != batch_size
            or int(patch_indices.size(1)) != patch_count
        ):
            raise ValueError(
                "Pruned concurrent coupling requires patch_indices [B, kept_patches]."
            )
        raw = patch_indices.to(device=device, dtype=torch.long)
        valid = (raw >= 0) & (raw < grid_patch_count)
        return raw.clamp(min=0, max=max(0, grid_patch_count - 1)), valid

    def forward(
        self,
        tokens: Tensor,
        local_state: Tensor,
        *,
        grid_size: Tuple[int, int],
        prefix_count: int,
        patch_indices: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        if tokens.ndim != 3 or local_state.ndim != 4:
            raise ValueError("Concurrent coupling expects tokens [B, N, D] and map [B, C, H, W].")
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        grid_patch_count = grid_height * grid_width
        batch_size = int(tokens.size(0))
        if tuple(int(value) for value in local_state.shape) != (
            batch_size,
            self.local_dim,
            grid_height,
            grid_width,
        ):
            raise ValueError("Concurrent local state shape does not match the patch grid.")
        prefix_count = max(0, min(int(prefix_count), int(tokens.size(1))))
        prefix_tokens = tokens[:, :prefix_count]
        patch_tokens = tokens[:, prefix_count:]
        patch_count = int(patch_tokens.size(1))
        if patch_count <= 0:
            if return_trace:
                empty = tokens.new_zeros((batch_size,), dtype=torch.float32)
                return tokens, local_state, {
                    "local_state_norm": empty,
                    "token_residual_norm": empty,
                }
            return tokens, local_state

        indices, valid = self._validated_indices(
            patch_indices,
            batch_size=batch_size,
            patch_count=patch_count,
            grid_patch_count=grid_patch_count,
            device=tokens.device,
        )
        projected_tokens = self.token_to_local(self.token_norm(patch_tokens))
        if indices is None:
            token_map = projected_tokens.transpose(1, 2).reshape(
                batch_size,
                self.local_dim,
                grid_height,
                grid_width,
            )
        else:
            dense_tokens = projected_tokens.new_zeros(
                (batch_size, grid_patch_count, self.local_dim)
            )
            dense_tokens = dense_tokens.scatter(
                1,
                indices.unsqueeze(-1).expand(-1, -1, self.local_dim),
                projected_tokens * valid.unsqueeze(-1).to(dtype=projected_tokens.dtype),
            )
            token_map = dense_tokens.transpose(1, 2).reshape(
                batch_size,
                self.local_dim,
                grid_height,
                grid_width,
            )

        local_input = local_state + token_map * self.token_to_local_scale.to(
            device=local_state.device,
            dtype=local_state.dtype,
        )
        local_delta = self.local_update(local_input)
        updated_local = local_state + local_delta * self.local_update_scale.to(
            device=local_state.device,
            dtype=local_state.dtype,
        )
        token_residual_map = self.local_to_token(updated_local)
        dense_residual = token_residual_map.flatten(2).transpose(1, 2)
        if indices is None:
            token_residual = dense_residual
        else:
            token_residual = dense_residual.gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
            token_residual = token_residual * valid.unsqueeze(-1).to(
                dtype=token_residual.dtype
            )
        updated_patches = patch_tokens + token_residual * self.local_to_token_scale.to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        updated_tokens = (
            torch.cat((prefix_tokens, updated_patches), dim=1)
            if prefix_count
            else updated_patches
        )
        if not return_trace:
            return updated_tokens, updated_local
        trace = {
            "local_state_norm": updated_local.detach().float().flatten(2).norm(dim=1).mean(dim=1),
            "token_residual_norm": token_residual.detach().float().norm(dim=-1).mean(dim=1),
            "token_to_local_scale": self.token_to_local_scale.detach().float().expand(batch_size),
            "local_update_scale": self.local_update_scale.detach().float().expand(batch_size),
            "local_to_token_scale": self.local_to_token_scale.detach().float().expand(batch_size),
        }
        return updated_tokens, updated_local, trace


class BlockLocalPatchMixer(nn.Module):
    """CMT/LeFF-style local residual mixer for patch tokens inside a block."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        residual_scale: float = 0.10,
        zero_init: bool = True,
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
        self.residual_scale = float(max(0.0, residual_scale))
        if bool(zero_init):
            self.zero_init_residual()

    def zero_init_residual(self) -> None:
        nn.init.zeros_(self.pointwise.weight)
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(
        self,
        tokens: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor] = None,
    ) -> Tensor:
        if tokens.ndim != 3 or grid_size is None:
            return tokens
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        grid_patch_count = grid_height * grid_width
        prefix_count = max(0, min(int(prefix_count), int(tokens.size(1))))
        patch_tokens = tokens[:, prefix_count:]
        batch_size, patch_count, embed_dim = patch_tokens.shape
        if patch_count <= 0:
            return tokens
        normalized = self.norm(patch_tokens)

        gather_indices: Optional[Tensor] = None
        gather_valid: Optional[Tensor] = None
        if patch_count == grid_patch_count:
            grid_tokens = normalized
        elif (
            torch.is_tensor(patch_indices)
            and patch_indices.ndim == 2
            and int(patch_indices.size(0)) == batch_size
            and int(patch_indices.size(1)) == patch_count
        ):
            raw_indices = patch_indices.to(device=tokens.device, dtype=torch.long)
            gather_valid = (raw_indices >= 0) & (raw_indices < grid_patch_count)
            gather_indices = raw_indices.clamp(min=0, max=max(0, grid_patch_count - 1))
            grid_tokens = normalized.new_zeros((batch_size, grid_patch_count, embed_dim))
            grid_tokens = grid_tokens.scatter(
                1,
                gather_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
                normalized * gather_valid.unsqueeze(-1).to(dtype=normalized.dtype),
            )
        else:
            return tokens

        grid = grid_tokens.transpose(1, 2).reshape(
            batch_size,
            embed_dim,
            grid_height,
            grid_width,
        )
        mixed = self.depthwise(grid)
        mixed = self.pointwise(self.act(mixed))
        mixed = mixed.flatten(2).transpose(1, 2)
        if gather_indices is not None:
            mixed = mixed.gather(
                1,
                gather_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
            )
            if gather_valid is not None:
                mixed = mixed * gather_valid.unsqueeze(-1).to(dtype=mixed.dtype)
        mixed = self.dropout(mixed) * self.residual_scale
        if prefix_count <= 0:
            return patch_tokens + mixed
        return torch.cat((tokens[:, :prefix_count], patch_tokens + mixed), dim=1)


class MaskedPatchReconstructionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, patch_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(patch_dim)),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("MaskedPatchReconstructionHead expects patch tokens [B, N, D].")
        return self.net(patch_tokens)


class TopKReassessmentHead(nn.Module):
    """Small residual classifier that reassesses low-margin top-k decisions."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_classes: int,
        top_k: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.top_k = max(1, int(top_k))
        input_dim = int(embed_dim) + 4 * self.num_classes
        hidden_dim = int(max(16, hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def zero_init_residual(self) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def forward(self, head_input: Tensor, base_logits: Tensor) -> Tensor:
        if head_input.ndim != 2:
            raise ValueError("TopKReassessmentHead expects head_input [B, D].")
        if base_logits.ndim != 2 or int(base_logits.size(1)) != self.num_classes:
            raise ValueError("TopKReassessmentHead expects base logits [B, num_classes].")
        base_float = base_logits.detach().float()
        centered = base_float - base_float.mean(dim=1, keepdim=True)
        normalized_logits = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(
            1e-4
        )
        probabilities = base_float.softmax(dim=1)
        selected_count = min(self.top_k, self.num_classes)
        top_values, top_indices = torch.topk(
            base_float,
            k=selected_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        top_logits = torch.zeros_like(base_float)
        top_logits.scatter_(1, top_indices, top_values)
        top_probabilities = torch.zeros_like(base_float)
        top_probabilities.scatter_(1, top_indices, probabilities.gather(1, top_indices))
        descriptor = torch.cat(
            [
                head_input,
                normalized_logits.to(device=head_input.device, dtype=head_input.dtype),
                probabilities.to(device=head_input.device, dtype=head_input.dtype),
                top_logits.to(device=head_input.device, dtype=head_input.dtype),
                top_probabilities.to(device=head_input.device, dtype=head_input.dtype),
            ],
            dim=1,
        )
        return self.net(descriptor)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        local_patch_mixer: bool = False,
        local_patch_mixer_dropout: float = 0.0,
        local_patch_mixer_scale: float = 0.10,
        locally_enhanced_ffn: bool = False,
        locally_enhanced_ffn_kernel_size: int = 3,
        gated_relative_position_attention: bool = False,
        relative_position_max_mix: float = 0.25,
        relative_position_locality_strength: float = 1.0,
        visual_contrast_attention: bool = False,
        visual_contrast_tokens: int = 64,
        foveal_aggregated_attention: bool = False,
        foveal_aggregated_attention_input_resolution: Tuple[int, int] = (16, 16),
        foveal_aggregated_attention_window_size: int = 3,
        foveal_aggregated_attention_pool_size: int = 4,
        deformable_spatial_attention: bool = False,
        deformable_spatial_attention_input_resolution: Tuple[int, int] = (16, 16),
        deformable_spatial_attention_groups: int = 2,
        deformable_spatial_attention_kernel_size: int = 5,
        deformable_spatial_attention_offset_range: float = 2.0,
        bi_level_routing_attention: bool = False,
        bi_level_routing_attention_input_resolution: Tuple[int, int] = (16, 16),
        bi_level_routing_attention_regions_per_axis: int = 4,
        bi_level_routing_attention_topk: int = 4,
        bi_level_routing_attention_local_context_kernel_size: int = 5,
        cross_covariance_attention: bool = False,
        cross_covariance_attention_residual_scale: float = 0.10,
        dynamic_graph_mixer: bool = False,
        dynamic_graph_mixer_bottleneck_dim: int = 64,
        dynamic_graph_mixer_k: int = 9,
        soft_moe_patch_adapter: bool = False,
        soft_moe_hidden_dim: int = 64,
        soft_moe_num_experts: int = 4,
        soft_moe_residual_scale: float = 0.10,
        soft_moe_router_scale_init: float = 10.0,
        soft_moe_init_seed: int = 20260715,
        patch_style_recalibration: bool = False,
        block_depth: int = 0,
    ) -> None:
        super().__init__()
        if bool(visual_contrast_attention) and bool(gated_relative_position_attention):
            raise ValueError(
                "Visual-Contrast Attention cannot share a block with gated relative "
                "position attention."
            )
        if bool(foveal_aggregated_attention) and bool(visual_contrast_attention):
            raise ValueError(
                "Foveal Aggregated Attention cannot share a block with "
                "Visual-Contrast Attention."
            )
        if bool(foveal_aggregated_attention) and bool(gated_relative_position_attention):
            raise ValueError(
                "Foveal Aggregated Attention cannot share a block with gated "
                "relative position attention."
            )
        if bool(deformable_spatial_attention) and bool(visual_contrast_attention):
            raise ValueError(
                "Deformable Spatial Attention cannot share a block with "
                "Visual-Contrast Attention."
            )
        if bool(deformable_spatial_attention) and bool(foveal_aggregated_attention):
            raise ValueError(
                "Deformable Spatial Attention cannot share a block with "
                "Foveal Aggregated Attention."
            )
        if bool(deformable_spatial_attention) and bool(gated_relative_position_attention):
            raise ValueError(
                "Deformable Spatial Attention cannot share a block with gated "
                "relative position attention."
            )
        if bool(deformable_spatial_attention) and bool(cross_covariance_attention):
            raise ValueError(
                "Deformable Spatial Attention cannot share a block with "
                "cross-covariance attention."
            )
        if bool(deformable_spatial_attention) and bool(dynamic_graph_mixer):
            raise ValueError(
                "Deformable Spatial Attention cannot share a block with "
                "dynamic graph mixing."
            )
        if bool(bi_level_routing_attention) and bool(visual_contrast_attention):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with "
                "Visual-Contrast Attention."
            )
        if bool(bi_level_routing_attention) and bool(foveal_aggregated_attention):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with "
                "Foveal Aggregated Attention."
            )
        if bool(bi_level_routing_attention) and bool(deformable_spatial_attention):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with "
                "Deformable Spatial Attention."
            )
        if bool(bi_level_routing_attention) and bool(gated_relative_position_attention):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with gated "
                "relative position attention."
            )
        if bool(bi_level_routing_attention) and bool(cross_covariance_attention):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with "
                "cross-covariance attention."
            )
        if bool(bi_level_routing_attention) and bool(dynamic_graph_mixer):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with "
                "dynamic graph mixing."
            )
        if bool(bi_level_routing_attention) and bool(soft_moe_patch_adapter):
            raise ValueError(
                "Bi-Level Routing Attention cannot share a block with a "
                "Soft-MoE patch adapter."
            )
        if bool(cross_covariance_attention) and bool(visual_contrast_attention):
            raise ValueError(
                "Cross-covariance attention requires the standard spatial MHSA block."
            )
        if bool(dynamic_graph_mixer) and bool(visual_contrast_attention):
            raise ValueError(
                "Dynamic graph mixing requires the standard spatial MHSA block."
            )
        if bool(dynamic_graph_mixer) and bool(cross_covariance_attention):
            raise ValueError(
                "Dynamic graph mixing cannot share a block with cross-covariance attention."
            )
        if bool(dynamic_graph_mixer) and bool(local_patch_mixer):
            raise ValueError(
                "Dynamic graph mixing cannot share a block with the local patch mixer."
            )
        if float(cross_covariance_attention_residual_scale) < 0.0:
            raise ValueError("cross_covariance_attention_residual_scale must be >= 0.")
        if bool(patch_style_recalibration) and bool(locally_enhanced_ffn):
            raise ValueError(
                "Patch-style recalibration requires the standard FeedForward path."
            )
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        if bool(visual_contrast_attention):
            self.attn = VisualContrastAttention(
                dim=dim,
                num_heads=num_heads,
                visual_contrast_tokens=visual_contrast_tokens,
                block_depth=block_depth,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
            )
        elif bool(foveal_aggregated_attention):
            standard_attention = MultiHeadSelfAttention(
                dim=dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
                gated_relative_position_attention=gated_relative_position_attention,
                relative_position_max_mix=relative_position_max_mix,
                relative_position_locality_strength=relative_position_locality_strength,
            )
            with torch.random.fork_rng(devices=[]):
                foveal_attention = FovealAggregatedAttention(
                    dim=dim,
                    input_resolution=foveal_aggregated_attention_input_resolution,
                    num_heads=num_heads,
                    window_size=foveal_aggregated_attention_window_size,
                    fixed_pool_size=foveal_aggregated_attention_pool_size,
                    attention_dropout=attention_dropout,
                    projection_dropout=dropout,
                )
            foveal_attention.copy_shared_projections_from(standard_attention)
            self.attn = foveal_attention
        elif bool(deformable_spatial_attention):
            standard_attention = MultiHeadSelfAttention(
                dim=dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
                gated_relative_position_attention=False,
            )
            with torch.random.fork_rng(devices=[]):
                deformable_attention = DeformableSpatialAttention(
                    dim=dim,
                    input_resolution=deformable_spatial_attention_input_resolution,
                    num_heads=num_heads,
                    offset_groups=deformable_spatial_attention_groups,
                    offset_kernel_size=deformable_spatial_attention_kernel_size,
                    offset_stride=1,
                    offset_range_factor=deformable_spatial_attention_offset_range,
                    attention_dropout=attention_dropout,
                    projection_dropout=dropout,
                )
            deformable_attention.copy_shared_projections_from(standard_attention)
            self.attn = deformable_attention
        elif bool(bi_level_routing_attention):
            standard_attention = MultiHeadSelfAttention(
                dim=dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
                gated_relative_position_attention=False,
            )
            with torch.random.fork_rng(devices=[]):
                routing_attention = BiLevelRoutingAttention(
                    dim=dim,
                    input_resolution=bi_level_routing_attention_input_resolution,
                    num_heads=num_heads,
                    regions_per_axis=(
                        bi_level_routing_attention_regions_per_axis
                    ),
                    topk=bi_level_routing_attention_topk,
                    local_context_kernel_size=(
                        bi_level_routing_attention_local_context_kernel_size
                    ),
                    attention_dropout=attention_dropout,
                    projection_dropout=dropout,
                )
            routing_attention.copy_shared_projections_from(standard_attention)
            self.attn = routing_attention
        else:
            self.attn = MultiHeadSelfAttention(
                dim=dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=dropout,
                gated_relative_position_attention=gated_relative_position_attention,
                relative_position_max_mix=relative_position_max_mix,
                relative_position_locality_strength=relative_position_locality_strength,
            )
        self.visual_contrast_attention_enabled = bool(visual_contrast_attention)
        self.foveal_aggregated_attention_enabled = bool(foveal_aggregated_attention)
        self.deformable_spatial_attention_enabled = bool(
            deformable_spatial_attention
        )
        self.bi_level_routing_attention_enabled = bool(
            bi_level_routing_attention
        )
        self.cross_covariance_attention = (
            SharedProjectionCrossCovarianceAttention(dim=dim, num_heads=num_heads)
            if bool(cross_covariance_attention)
            else None
        )
        self.cross_covariance_attention_residual_scale = float(
            cross_covariance_attention_residual_scale
        )
        self.dynamic_graph_mixer = (
            MaxRelativeDynamicGraphMixer(
                dim=dim,
                bottleneck_dim=dynamic_graph_mixer_bottleneck_dim,
                k=dynamic_graph_mixer_k,
            )
            if bool(dynamic_graph_mixer)
            else None
        )
        self.soft_moe_patch_adapter = (
            SoftMoEPatchAdapter(
                dim=dim,
                hidden_dim=soft_moe_hidden_dim,
                num_experts=soft_moe_num_experts,
                residual_scale=soft_moe_residual_scale,
                router_scale_init=soft_moe_router_scale_init,
                init_seed=int(soft_moe_init_seed) + int(block_depth),
            )
            if bool(soft_moe_patch_adapter)
            else None
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = (
            LocallyEnhancedFeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=locally_enhanced_ffn_kernel_size,
            )
            if bool(locally_enhanced_ffn)
            else FeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                patch_style_recalibration=patch_style_recalibration,
            )
        )
        self.locally_enhanced_ffn_enabled = bool(locally_enhanced_ffn)
        self.patch_style_recalibration_enabled = bool(patch_style_recalibration)
        self.drop_path2 = DropPath(drop_path_rate)
        self.local_patch_mixer = (
            BlockLocalPatchMixer(
                embed_dim=dim,
                dropout=local_patch_mixer_dropout,
                residual_scale=local_patch_mixer_scale,
                zero_init=True,
            )
            if bool(local_patch_mixer)
            else None
        )

    def _forward_mlp(
        self,
        x: Tensor,
        *,
        grid_size: Optional[Tuple[int, int]],
        prefix_count: int,
        patch_indices: Optional[Tensor],
    ) -> Tensor:
        normalized = self.norm2(x)
        if self.locally_enhanced_ffn_enabled:
            return self.mlp(
                normalized,
                grid_size=grid_size,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        return self.mlp(normalized, prefix_count=prefix_count)

    def apply_cross_covariance_attention(
        self,
        x: Tensor,
        *,
        prefix_count: int,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        module = self.cross_covariance_attention
        if module is None:
            if return_attention:
                raise RuntimeError("Cross-covariance attention is not enabled in this block.")
            return x
        resolved_prefix_count = int(prefix_count)
        if not 0 <= resolved_prefix_count < int(x.size(1)):
            raise ValueError("prefix_count must leave at least one patch token.")
        patch_tokens = x[:, resolved_prefix_count:]
        residual, channel_attention = module(
            patch_tokens,
            qkv_projection=self.attn.qkv,
            output_projection=self.attn.proj,
            return_attention=True,
        )
        residual = residual * self.cross_covariance_attention_residual_scale
        denominator = patch_tokens.detach().float().norm().clamp_min(1e-12)
        module.record_residual_norm_ratio(residual.detach().float().norm() / denominator)
        updated_patches = patch_tokens + residual
        if resolved_prefix_count:
            updated = torch.cat((x[:, :resolved_prefix_count], updated_patches), dim=1)
        else:
            updated = updated_patches
        if return_attention:
            return updated, channel_attention
        return updated

    def apply_dynamic_graph_mixer(
        self,
        x: Tensor,
        *,
        prefix_count: int,
        grid_size: Optional[Tuple[int, int]],
        patch_indices: Optional[Tensor],
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        module = self.dynamic_graph_mixer
        if module is None:
            if return_details:
                raise RuntimeError("Dynamic graph mixing is not enabled in this block.")
            return x
        resolved_prefix_count = int(prefix_count)
        if not 0 <= resolved_prefix_count < int(x.size(1)):
            raise ValueError("prefix_count must leave at least one patch token.")
        patch_tokens = x[:, resolved_prefix_count:]
        result = module(
            patch_tokens,
            patch_indices=patch_indices,
            grid_size=grid_size,
            return_details=return_details,
        )
        if return_details:
            residual, details = result
        else:
            residual = result
        updated_patches = patch_tokens + residual
        updated = (
            torch.cat((x[:, :resolved_prefix_count], updated_patches), dim=1)
            if resolved_prefix_count
            else updated_patches
        )
        if return_details:
            return updated, details
        return updated

    def apply_soft_moe_patch_adapter(
        self,
        x: Tensor,
        *,
        prefix_count: int,
        patch_indices: Optional[Tensor] = None,
        return_details: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        module = self.soft_moe_patch_adapter
        if module is None:
            if return_details:
                raise RuntimeError("Soft-MoE patch adapter is not enabled in this block.")
            return x
        return module(
            x,
            prefix_count=prefix_count,
            patch_indices=patch_indices,
            return_details=return_details,
        )

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
        collect_attention_trace: bool = False,
    ):
        if return_attention:
            if self.bi_level_routing_attention_enabled:
                attn_out, attention = self.attn(
                    self.norm1(x),
                    return_attention=True,
                    grid_size=grid_size,
                    prefix_count=prefix_count,
                    patch_indices=patch_indices,
                    collect_trace=bool(collect_attention_trace),
                )
            else:
                attn_out, attention = self.attn(
                    self.norm1(x),
                    return_attention=True,
                    grid_size=grid_size,
                    prefix_count=prefix_count,
                    patch_indices=patch_indices,
                )
            x = x + self.drop_path1(attn_out)
            if self.cross_covariance_attention is not None:
                x = self.apply_cross_covariance_attention(
                    x,
                    prefix_count=prefix_count,
                )
            if self.dynamic_graph_mixer is not None:
                x = self.apply_dynamic_graph_mixer(
                    x,
                    prefix_count=prefix_count,
                    grid_size=grid_size,
                    patch_indices=patch_indices,
                )
            if self.local_patch_mixer is not None:
                x = self.local_patch_mixer(
                    x,
                    grid_size=grid_size,
                    prefix_count=prefix_count,
                    patch_indices=patch_indices,
                )
            if self.soft_moe_patch_adapter is not None:
                x = self.apply_soft_moe_patch_adapter(
                    x,
                    prefix_count=prefix_count,
                    patch_indices=patch_indices,
                )
            x = x + self.drop_path2(
                self._forward_mlp(
                    x,
                    grid_size=grid_size,
                    prefix_count=prefix_count,
                    patch_indices=patch_indices,
                )
            )
            return x, attention
        x = x + self.drop_path1(
            self.attn(
                self.norm1(x),
                grid_size=grid_size,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        )
        if self.cross_covariance_attention is not None:
            x = self.apply_cross_covariance_attention(
                x,
                prefix_count=prefix_count,
            )
        if self.dynamic_graph_mixer is not None:
            x = self.apply_dynamic_graph_mixer(
                x,
                prefix_count=prefix_count,
                grid_size=grid_size,
                patch_indices=patch_indices,
            )
        if self.local_patch_mixer is not None:
            x = self.local_patch_mixer(
                x,
                grid_size=grid_size,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        if self.soft_moe_patch_adapter is not None:
            x = self.apply_soft_moe_patch_adapter(
                x,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        x = x + self.drop_path2(
            self._forward_mlp(
                x,
                grid_size=grid_size,
                prefix_count=prefix_count,
                patch_indices=patch_indices,
            )
        )
        return x


TransformerBlock = CustomTransformerEncoderLayer


class TeacherFeatureProjectionAdapter(nn.Module):
    """Train-time projector for aligning student and teacher feature spaces."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        projection_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(student_dim) <= 0:
            raise ValueError("student_dim phai > 0.")
        if int(teacher_dim) <= 0:
            raise ValueError("teacher_dim phai > 0.")
        if int(projection_dim) <= 0:
            raise ValueError("projection_dim phai > 0.")
        drop = float(max(0.0, dropout))
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.projection_dim = int(projection_dim)
        self.student_projector = nn.Sequential(
            nn.LayerNorm(self.student_dim),
            nn.Dropout(drop),
            nn.Linear(self.student_dim, self.projection_dim, bias=False),
        )
        self.teacher_projector = nn.Sequential(
            nn.LayerNorm(self.teacher_dim),
            nn.Dropout(drop),
            nn.Linear(self.teacher_dim, self.projection_dim, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.student_projector[-1], self.teacher_projector[-1]):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, student: Tensor, teacher: Tensor) -> Tuple[Tensor, Tensor]:
        if student.ndim != 2 or teacher.ndim != 2:
            raise ValueError("TeacherFeatureProjectionAdapter expects [B, D] tensors.")
        if int(student.size(1)) != self.student_dim:
            raise ValueError(
                f"Student feature dim mismatch: got {int(student.size(1))}, expected {self.student_dim}."
            )
        if int(teacher.size(1)) != self.teacher_dim:
            raise ValueError(
                f"Teacher feature dim mismatch: got {int(teacher.size(1))}, expected {self.teacher_dim}."
            )
        return self.student_projector(student.float()), self.teacher_projector(teacher.float())


class VisionTransformerWithRegisters(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cnn_stem: bool = True,
        stem_channels: int = 32,
        stem_architecture: str = "conv_pool",
        stem_normalization: str = "batch",
        stem_pooling_mode: str = "max",
        stem_softpool_blend: float = 0.15,
        shifted_patch_tokenization: bool = False,
        shifted_patch_shift: int = 1,
        shifted_patch_residual_scale: float = 0.10,
        dual_patch_norm: bool = False,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        color_stat_fusion: bool = False,
        color_stat_fusion_dropout: float = 0.1,
        defect_stat_fusion: bool = False,
        defect_stat_fusion_dropout: float = 0.1,
        foreground_surface_fusion: bool = False,
        foreground_surface_fusion_dropout: float = 0.1,
        foreground_surface_pairwise_head: bool = False,
        foreground_surface_pairwise_pairs: str = "0-1,1-2,1-4,2-3",
        foreground_surface_pairwise_logit_scale: float = 0.18,
        foreground_surface_pairwise_dropout: float = 0.05,
        foreground_surface_pairwise_routing: bool = True,
        foreground_surface_pairwise_route_max_probability_margin: float = 0.22,
        interior_boundary_pairwise_head: bool = False,
        interior_boundary_pairwise_pairs: str = "0-1,4-1",
        interior_boundary_pairwise_logit_scale: float = 0.14,
        interior_boundary_pairwise_dropout: float = 0.05,
        interior_boundary_pairwise_hidden_dim: int = 128,
        interior_boundary_pairwise_erode_kernel: int = 9,
        interior_boundary_pairwise_routing: bool = True,
        interior_boundary_pairwise_route_max_probability_margin: float = 0.22,
        bbox_spatial_fusion: bool = False,
        bbox_spatial_fusion_hidden_dim: int = 64,
        bbox_spatial_fusion_dropout: float = 0.05,
        bbox_spatial_fusion_logit_scale: float = 0.20,
        patch_objectness_guided_head: bool = False,
        patch_objectness_hidden_dim: int = 128,
        patch_objectness_dropout: float = 0.05,
        patch_objectness_logit_scale: float = 0.12,
        patch_objectness_temperature: float = 0.75,
        bbox_prior_patch_context_head: bool = False,
        bbox_prior_patch_context_hidden_dim: int = 128,
        bbox_prior_patch_context_dropout: float = 0.05,
        bbox_prior_patch_context_logit_scale: float = 0.12,
        bbox_prior_patch_context_temperature: float = 0.50,
        source_context_feature_fusion: bool = False,
        source_context_fusion_hidden_dim: int = 128,
        source_context_fusion_dropout: float = 0.05,
        source_context_fusion_logit_scale: float = 0.20,
        source_context_fusion_gate_bias: float = -2.0,
        paired_view_feature_fusion: bool = False,
        paired_view_fusion_hidden_dim: int = 128,
        paired_view_fusion_dropout: float = 0.05,
        paired_view_fusion_logit_scale: float = 0.12,
        paired_view_fusion_gate_bias: float = -2.0,
        bilinear_patch_fusion: bool = False,
        bilinear_patch_rank: int = 32,
        bilinear_patch_dropout: float = 0.1,
        complementary_patch_suppression_head: bool = False,
        complementary_patch_suppression_top_k: int = 6,
        complementary_patch_suppression_hidden_dim: int = 128,
        complementary_patch_suppression_dropout: float = 0.05,
        complementary_patch_suppression_temperature: float = 0.20,
        complementary_patch_suppression_strength: float = 0.85,
        complementary_patch_suppression_bbox_weight: float = 0.35,
        complementary_patch_suppression_logit_scale: float = 0.12,
        micro_detail_patch_expert: bool = False,
        micro_detail_top_k: int = 8,
        micro_detail_hidden_dim: int = 128,
        micro_detail_dropout: float = 0.08,
        micro_detail_temperature: float = 0.12,
        micro_detail_foreground_power: float = 1.0,
        micro_detail_logit_scale: float = 0.18,
        micro_detail_routing: bool = True,
        micro_detail_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        micro_detail_route_max_probability_margin: float = 0.25,
        part_token_learner: bool = False,
        part_token_count: int = 4,
        part_token_hidden_dim: int = 128,
        part_token_dropout: float = 0.08,
        part_token_temperature: float = 0.70,
        part_token_foreground_power: float = 1.0,
        part_token_bbox_weight: float = 0.75,
        part_token_logit_scale: float = 0.16,
        part_token_routing: bool = True,
        part_token_route_pairs: str = "0-1,1-2,4-1,2-3",
        part_token_route_max_probability_margin: float = 0.25,
        part_token_pairwise_head: bool = False,
        part_token_pairwise_pairs: str = "0-1,1-2,4-1,2-3",
        part_token_pairwise_logit_scale: float = 0.18,
        part_token_pairwise_dropout: float = 0.08,
        part_token_pairwise_routing: bool = True,
        part_token_pairwise_route_max_probability_margin: float = 0.22,
        patch_evidence_router_head: bool = False,
        patch_evidence_router_pair: str = "0-1",
        patch_evidence_router_hidden_dim: int = 128,
        patch_evidence_router_top_k: int = 4,
        patch_evidence_router_bbox_weight: float = 0.75,
        patch_evidence_router_dropout: float = 0.05,
        patch_evidence_router_logit_scale: float = 0.12,
        patch_evidence_router_margin_prior_mode: str = "none",
        patch_evidence_router_margin_prior_scale: float = 0.0,
        patch_evidence_router_summary_stats: bool = False,
        patch_evidence_router_routing: bool = True,
        patch_evidence_router_route_max_probability_margin: float = 0.25,
        patch_evidence_router_route_min_pair_probability: float = 0.02,
        local_zoom_image_expert: bool = False,
        local_zoom_crop_size: int = 128,
        local_zoom_crop_scale: float = 0.48,
        local_zoom_score_mode: str = "foreground_detail",
        local_zoom_hidden_dim: int = 128,
        local_zoom_dropout: float = 0.08,
        local_zoom_logit_scale: float = 0.16,
        local_zoom_routing: bool = True,
        local_zoom_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        local_zoom_route_max_probability_margin: float = 0.25,
        high_frequency_texture_expert: bool = False,
        high_frequency_texture_hidden_dim: int = 128,
        high_frequency_texture_dropout: float = 0.08,
        high_frequency_texture_analysis_size: int = 96,
        high_frequency_texture_logit_scale: float = 0.16,
        high_frequency_texture_routing: bool = True,
        high_frequency_texture_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        high_frequency_texture_route_max_probability_margin: float = 0.25,
        multi_granularity_aux_heads: bool = False,
        multi_granularity_aux_layers: str = "2,5,8",
        multi_granularity_aux_dropout: float = 0.08,
        self_boosting_attention_head: bool = False,
        block_local_patch_mixer: bool = False,
        block_local_patch_mixer_layers: str = "6,7,8",
        block_local_patch_mixer_dropout: float = 0.0,
        block_local_patch_mixer_scale: float = 0.10,
        locally_enhanced_ffn: bool = False,
        locally_enhanced_ffn_layers: str = "1,2,3,4",
        locally_enhanced_ffn_kernel_size: int = 3,
        concurrent_local_global_coupling: bool = False,
        concurrent_local_global_layers: str = "1,2,3,4,5,6,7,8",
        concurrent_local_global_dim: int = 64,
        concurrent_local_global_kernel_size: int = 3,
        gated_relative_position_attention: bool = False,
        gated_relative_position_attention_layers: str = "1,2,3,4",
        gated_relative_position_attention_max_mix: float = 0.25,
        gated_relative_position_attention_locality_strength: float = 1.0,
        visual_contrast_attention: bool = False,
        visual_contrast_attention_layers: str = "1,2,3,4,5,6,7,8",
        visual_contrast_tokens: int = 64,
        foveal_aggregated_attention: bool = False,
        foveal_aggregated_attention_layers: str = "1",
        foveal_aggregated_attention_window_size: int = 3,
        foveal_aggregated_attention_pool_size: int = 4,
        deformable_spatial_attention: bool = False,
        deformable_spatial_attention_layers: str = "2",
        deformable_spatial_attention_groups: int = 2,
        deformable_spatial_attention_kernel_size: int = 5,
        deformable_spatial_attention_offset_range: float = 2.0,
        bi_level_routing_attention: bool = False,
        bi_level_routing_attention_layers: str = "2",
        bi_level_routing_attention_regions_per_axis: int = 4,
        bi_level_routing_attention_topk: int = 4,
        bi_level_routing_attention_local_context_kernel_size: int = 5,
        cross_covariance_attention: bool = False,
        cross_covariance_attention_layers: str = "2,5",
        cross_covariance_attention_residual_scale: float = 0.10,
        dynamic_graph_mixer: bool = False,
        dynamic_graph_mixer_layers: str = "2,5",
        dynamic_graph_mixer_bottleneck_dim: int = 64,
        dynamic_graph_mixer_k: int = 9,
        soft_moe_patch_adapter: bool = False,
        soft_moe_patch_adapter_layers: str = "2,5",
        soft_moe_hidden_dim: int = 64,
        soft_moe_num_experts: int = 4,
        soft_moe_residual_scale: float = 0.10,
        soft_moe_router_scale_init: float = 10.0,
        soft_moe_init_seed: int = 20260715,
        deep_class_prompt: bool = False,
        deep_class_prompt_logit_scale: float = 0.10,
        deep_class_prompt_init_seed: int = 20260715,
        patch_style_recalibration: bool = False,
        patch_style_recalibration_layers: str = "2,5",
        layer_token_fusion: bool = False,
        layer_token_fusion_layers: str = "2,4,6",
        layer_token_fusion_top_k: int = 4,
        layer_token_fusion_blend: float = 0.12,
        layer_token_fusion_attention_temperature: float = 0.20,
        layer_token_fusion_bbox_weight: float = 0.20,
        layer_token_fusion_foreground_weight: float = 0.10,
        masked_reconstruction_head: bool = False,
        masked_reconstruction_hidden_dim: int = 192,
        frequency_selective_pooling: bool = False,
        frequency_selective_top_k: int = 1,
        frequency_selective_blend: float = 1.0,
        frequency_selective_foreground_threshold: float = 0.35,
        patch_memory_adapter: bool = False,
        patch_memory_adapter_dropout: float = 0.0,
        late_class_attention_pooling: bool = False,
        late_class_attention_heads: int = 4,
        late_class_attention_dropout: float = 0.05,
        late_class_attention_mlp_ratio: float = 2.0,
        late_class_attention_residual_scale: float = 0.10,
        late_member_branch: bool = False,
        late_member_fork_after_block: int = 6,
        late_member_candidate_weight: float = 0.40,
        late_member_focus_class: int = 1,
        late_member_focus_margin_offset: float = 0.0,
        mixstyle: bool = False,
        mixstyle_probability: float = 0.5,
        mixstyle_alpha: float = 0.1,
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
        learnable_gabor_texture_residual: bool = False,
        learnable_gabor_texture_semantic_fusion: bool = False,
        detail_patch_enhancement: bool = False,
        detail_patch_dropout: float = 0.05,
        token_pruning: bool = False,
        token_prune_layers: str = "2,5",
        token_keep_rates: str = "0.75,0.50",
        token_prune_foreground_weight: float = 0.35,
        token_prune_bbox_weight: float = 0.0,
        token_prune_bbox_margin_ratio: float = 0.04,
        early_token_mask_keep_rate: float = 1.0,
        inattentive_token_fusion: bool = False,
        cropr_token_selector: bool = False,
        cropr_token_selector_routing: bool = True,
        pairwise_margin_head: bool = False,
        pairwise_margin_pairs: str = "0-1,2-3,4-rest",
        pairwise_margin_logit_scale: float = 0.35,
        pairwise_margin_dropout: float = 0.05,
        pairwise_margin_routing: bool = False,
        pairwise_margin_route_max_probability_margin: float = 0.20,
        topk_reassessment_head: bool = False,
        topk_reassessment_top_k: int = 2,
        topk_reassessment_hidden_dim: int = 128,
        topk_reassessment_dropout: float = 0.05,
        topk_reassessment_logit_scale: float = 0.15,
        topk_reassessment_routing: bool = True,
        topk_reassessment_route_pairs: str = "0-1,1-2,2-3,4-rest",
        topk_reassessment_route_max_probability_margin: float = 0.30,
        focus_class_head: bool = False,
        focus_class_index: int = 1,
        focus_class_logit_scale: float = 0.20,
        focus_class_dropout: float = 0.05,
        focus_class_routing: bool = True,
        focus_class_route_max_probability_margin: float = 0.35,
        focus_class_route_min_probability: float = 0.08,
        class_independent_head: bool = False,
        class_independent_dropout: float = 0.05,
        ordinal_maturity_head: bool = False,
        ordinal_maturity_classes: str = "0,1,2,3",
        ordinal_maturity_logit_scale: float = 0.20,
        ordinal_maturity_dropout: float = 0.05,
        cumulative_ordinal_head: bool = False,
        cumulative_ordinal_classes: str = "0,1,2,3",
        cumulative_ordinal_logit_scale: float = 0.25,
        cumulative_ordinal_dropout: float = 0.05,
        classification_mlp_head: bool = False,
        classification_mlp_hidden_dim: int = 512,
        classification_mlp_dropout: float = 0.08,
        classification_mlp_residual_scale: float = 0.20,
        subcenter_proxy_head: bool = False,
        subcenter_proxy_subcenters: int = 3,
        subcenter_proxy_dropout: float = 0.0,
        subcenter_proxy_init_std: float = 0.01,
        deep_abstention_head: bool = False,
        deep_abstention_dropout: float = 0.05,
        deep_abstention_initial_probability: float = 0.01,
        teacher_feature_projection_adapter: bool = False,
        teacher_feature_projection_teacher_dim: int = 0,
        teacher_feature_projection_dim: int = 128,
        teacher_feature_projection_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        head_pooling: str = "cls_register_mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.register_positional_embedding = register_positional_embedding
        self.use_cnn_stem = use_cnn_stem
        self.stem_architecture = str(stem_architecture).strip().lower()
        if self.stem_architecture not in {
            "conv_pool",
            "coatnet_mbconv",
            "dbb_conv_pool",
            "inceptionnext_atto_tokenizer",
            "moganet_xt_tokenizer",
            "octave_conv",
            "starnet_s2_tokenizer",
        }:
            raise ValueError(
                "stem_architecture must be one of: conv_pool, coatnet_mbconv, "
                "dbb_conv_pool, "
                "inceptionnext_atto_tokenizer, moganet_xt_tokenizer, "
                "octave_conv, starnet_s2_tokenizer; "
                f"got {stem_architecture!r}."
            )
        self.stem_normalization = str(stem_normalization).strip().lower()
        if self.stem_normalization not in {"batch", "ibn_a_first"}:
            raise ValueError(
                "stem_normalization must be one of: batch, ibn_a_first; "
                f"got {stem_normalization!r}."
            )
        if self.stem_normalization != "batch" and self.stem_architecture != "conv_pool":
            raise ValueError(
                "stem_normalization=ibn_a_first is only supported by stem_architecture=conv_pool."
            )
        self.stem_pooling_mode = str(stem_pooling_mode).strip().lower()
        self.stem_softpool_blend = float(stem_softpool_blend)
        self.shifted_patch_tokenization_enabled = bool(shifted_patch_tokenization)
        self.shifted_patch_shift = int(shifted_patch_shift)
        self.shifted_patch_residual_scale = float(shifted_patch_residual_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed_dim = int(embed_dim)
        self.deep_class_prompt_enabled = bool(deep_class_prompt)
        self.deep_class_prompt_logit_scale = float(deep_class_prompt_logit_scale)
        self.deep_class_prompt_init_seed = int(deep_class_prompt_init_seed)
        if self.deep_class_prompt_logit_scale < 0.0:
            raise ValueError("deep_class_prompt_logit_scale must be >= 0.")
        self.head_pooling = str(head_pooling).strip().lower()
        self.classification_mlp_head_enabled = bool(classification_mlp_head)
        self.subcenter_proxy_head_enabled = bool(subcenter_proxy_head)
        self.deep_abstention_head_enabled = bool(deep_abstention_head)
        self.cnn_feature_fusion = bool(cnn_feature_fusion and use_cnn_stem)
        self.fine_grained_pooling = bool(fine_grained_pooling)
        self.color_stat_fusion = bool(color_stat_fusion)
        self.defect_stat_fusion = bool(defect_stat_fusion)
        self.foreground_surface_fusion = bool(foreground_surface_fusion)
        self.foreground_surface_pairwise_head_enabled = bool(foreground_surface_pairwise_head)
        self.foreground_surface_pairwise_logit_scale = float(
            max(0.0, foreground_surface_pairwise_logit_scale)
        )
        self.foreground_surface_pairwise_routing = bool(foreground_surface_pairwise_routing)
        self.foreground_surface_pairwise_route_max_probability_margin = float(
            max(0.0, foreground_surface_pairwise_route_max_probability_margin)
        )
        self.foreground_surface_pairwise_pairs = (
            _parse_pairwise_margin_pairs(foreground_surface_pairwise_pairs, num_classes)
            if self.foreground_surface_pairwise_head_enabled
            else []
        )
        self.interior_boundary_pairwise_head_enabled = bool(interior_boundary_pairwise_head)
        self.interior_boundary_pairwise_logit_scale = float(
            max(0.0, interior_boundary_pairwise_logit_scale)
        )
        self.interior_boundary_pairwise_routing = bool(interior_boundary_pairwise_routing)
        self.interior_boundary_pairwise_route_max_probability_margin = float(
            max(0.0, interior_boundary_pairwise_route_max_probability_margin)
        )
        self.interior_boundary_pairwise_pairs = (
            _parse_pairwise_margin_pairs(interior_boundary_pairwise_pairs, num_classes)
            if self.interior_boundary_pairwise_head_enabled
            else []
        )
        self.bbox_spatial_fusion_enabled = bool(bbox_spatial_fusion)
        self.bbox_spatial_fusion_logit_scale = float(max(0.0, bbox_spatial_fusion_logit_scale))
        self.patch_objectness_guided_head_enabled = bool(patch_objectness_guided_head)
        self.patch_objectness_logit_scale = float(max(0.0, patch_objectness_logit_scale))
        self.bbox_prior_patch_context_head_enabled = bool(bbox_prior_patch_context_head)
        self.bbox_prior_patch_context_logit_scale = float(
            max(0.0, bbox_prior_patch_context_logit_scale)
        )
        self.source_context_feature_fusion_enabled = bool(source_context_feature_fusion)
        self.source_context_fusion_logit_scale = float(
            max(0.0, source_context_fusion_logit_scale)
        )
        self.paired_view_feature_fusion_enabled = bool(paired_view_feature_fusion)
        self.paired_view_fusion_logit_scale = float(
            max(0.0, paired_view_fusion_logit_scale)
        )
        self.bilinear_patch_fusion = bool(bilinear_patch_fusion)
        self.complementary_patch_suppression_head_enabled = bool(
            complementary_patch_suppression_head
        )
        self.complementary_patch_suppression_logit_scale = float(
            max(0.0, complementary_patch_suppression_logit_scale)
        )
        self.micro_detail_patch_expert_enabled = bool(micro_detail_patch_expert)
        self.micro_detail_logit_scale = float(max(0.0, micro_detail_logit_scale))
        self.micro_detail_routing = bool(micro_detail_routing)
        self.micro_detail_route_max_probability_margin = float(
            max(0.0, micro_detail_route_max_probability_margin)
        )
        self.micro_detail_route_pairs = (
            _parse_pairwise_margin_pairs(micro_detail_route_pairs, num_classes)
            if self.micro_detail_patch_expert_enabled
            else []
        )
        self.part_token_learner_enabled = bool(part_token_learner)
        self.part_token_logit_scale = float(max(0.0, part_token_logit_scale))
        self.part_token_routing = bool(part_token_routing)
        self.part_token_route_max_probability_margin = float(
            max(0.0, part_token_route_max_probability_margin)
        )
        self.part_token_route_pairs = (
            _parse_pairwise_margin_pairs(part_token_route_pairs, num_classes)
            if self.part_token_learner_enabled
            else []
        )
        self.part_token_pairwise_head_enabled = bool(part_token_pairwise_head)
        self.part_token_pairwise_logit_scale = float(max(0.0, part_token_pairwise_logit_scale))
        self.part_token_pairwise_routing = bool(part_token_pairwise_routing)
        self.part_token_pairwise_route_max_probability_margin = float(
            max(0.0, part_token_pairwise_route_max_probability_margin)
        )
        self.part_token_pairwise_pairs = (
            _parse_pairwise_margin_pairs(part_token_pairwise_pairs, num_classes)
            if self.part_token_pairwise_head_enabled
            else []
        )
        self.patch_evidence_router_head_enabled = bool(patch_evidence_router_head)
        self.patch_evidence_router_logit_scale = float(max(0.0, patch_evidence_router_logit_scale))
        self.patch_evidence_router_margin_prior_mode = str(
            patch_evidence_router_margin_prior_mode or "none"
        ).strip().lower()
        self.patch_evidence_router_margin_prior_scale = float(
            max(0.0, patch_evidence_router_margin_prior_scale)
        )
        self.patch_evidence_router_summary_stats = bool(patch_evidence_router_summary_stats)
        self.patch_evidence_router_routing = bool(patch_evidence_router_routing)
        self.patch_evidence_router_route_max_probability_margin = float(
            max(0.0, patch_evidence_router_route_max_probability_margin)
        )
        self.patch_evidence_router_route_min_pair_probability = float(
            min(1.0, max(0.0, patch_evidence_router_route_min_pair_probability))
        )
        self.patch_evidence_router_pairs = (
            _parse_pairwise_margin_pairs(patch_evidence_router_pair, num_classes)
            if self.patch_evidence_router_head_enabled
            else []
        )
        self.local_zoom_image_expert_enabled = bool(local_zoom_image_expert)
        self.local_zoom_logit_scale = float(max(0.0, local_zoom_logit_scale))
        self.local_zoom_routing = bool(local_zoom_routing)
        self.local_zoom_route_max_probability_margin = float(
            max(0.0, local_zoom_route_max_probability_margin)
        )
        self.local_zoom_route_pairs = (
            _parse_pairwise_margin_pairs(local_zoom_route_pairs, num_classes)
            if self.local_zoom_image_expert_enabled
            else []
        )
        self.high_frequency_texture_expert_enabled = bool(high_frequency_texture_expert)
        self.high_frequency_texture_logit_scale = float(
            max(0.0, high_frequency_texture_logit_scale)
        )
        self.high_frequency_texture_routing = bool(high_frequency_texture_routing)
        self.high_frequency_texture_route_max_probability_margin = float(
            max(0.0, high_frequency_texture_route_max_probability_margin)
        )
        self.high_frequency_texture_route_pairs = (
            _parse_pairwise_margin_pairs(high_frequency_texture_route_pairs, num_classes)
            if self.high_frequency_texture_expert_enabled
            else []
        )
        self.multi_granularity_aux_layer_numbers = (
            _parse_auxiliary_layer_indices(multi_granularity_aux_layers, int(depth))
            if bool(multi_granularity_aux_heads)
            else []
        )
        self.self_boosting_attention_head_enabled = bool(self_boosting_attention_head)
        self.block_local_patch_mixer_layer_numbers = (
            _parse_auxiliary_layer_indices(block_local_patch_mixer_layers, int(depth))
            if bool(block_local_patch_mixer)
            else []
        )
        self.block_local_patch_mixer_enabled = bool(
            self.block_local_patch_mixer_layer_numbers
        )
        self.locally_enhanced_ffn_layer_numbers = (
            _parse_auxiliary_layer_indices(locally_enhanced_ffn_layers, int(depth))
            if bool(locally_enhanced_ffn)
            else []
        )
        self.locally_enhanced_ffn_enabled = bool(
            self.locally_enhanced_ffn_layer_numbers
        )
        self.concurrent_local_global_layer_numbers = (
            _parse_auxiliary_layer_indices(concurrent_local_global_layers, int(depth))
            if bool(concurrent_local_global_coupling)
            else []
        )
        self.concurrent_local_global_enabled = bool(
            self.concurrent_local_global_layer_numbers
        )
        self.concurrent_local_global_dim = int(concurrent_local_global_dim)
        self.concurrent_local_global_kernel_size = int(
            concurrent_local_global_kernel_size
        )
        if self.concurrent_local_global_enabled:
            if self.concurrent_local_global_dim <= 0:
                raise ValueError("concurrent_local_global_dim must be positive.")
            if (
                self.concurrent_local_global_kernel_size < 3
                or self.concurrent_local_global_kernel_size % 2 == 0
            ):
                raise ValueError(
                    "concurrent_local_global_kernel_size must be odd and >= 3."
                )
        self.gated_relative_position_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                gated_relative_position_attention_layers,
                int(depth),
            )
            if bool(gated_relative_position_attention)
            else []
        )
        self.gated_relative_position_attention_enabled = bool(
            self.gated_relative_position_attention_layer_numbers
        )
        self.visual_contrast_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                visual_contrast_attention_layers,
                int(depth),
            )
            if bool(visual_contrast_attention)
            else []
        )
        self.visual_contrast_attention_enabled = bool(
            self.visual_contrast_attention_layer_numbers
        )
        self.visual_contrast_tokens = int(visual_contrast_tokens)
        visual_contrast_side = math.isqrt(self.visual_contrast_tokens)
        if self.visual_contrast_attention_enabled:
            if (
                self.visual_contrast_tokens <= 0
                or visual_contrast_side * visual_contrast_side
                != self.visual_contrast_tokens
            ):
                raise ValueError("visual_contrast_tokens must be a positive perfect square.")
            if bool(token_pruning) or float(early_token_mask_keep_rate) < 1.0:
                raise ValueError(
                    "Visual-Contrast Attention requires a complete dense patch grid; "
                    "disable token pruning and early token masking."
                )
            if self.gated_relative_position_attention_enabled:
                raise ValueError(
                    "Visual-Contrast Attention and gated relative position attention "
                    "cannot be enabled in the same model."
                )
        self.foveal_aggregated_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                foveal_aggregated_attention_layers,
                int(depth),
            )
            if bool(foveal_aggregated_attention)
            else []
        )
        self.foveal_aggregated_attention_enabled = bool(
            self.foveal_aggregated_attention_layer_numbers
        )
        self.foveal_aggregated_attention_window_size = int(
            foveal_aggregated_attention_window_size
        )
        self.foveal_aggregated_attention_pool_size = int(
            foveal_aggregated_attention_pool_size
        )
        if self.foveal_aggregated_attention_enabled:
            if self.foveal_aggregated_attention_layer_numbers != [1]:
                raise ValueError(
                    "Foveal Aggregated Attention is locked to transformer layer 1."
                )
            if float(early_token_mask_keep_rate) < 1.0:
                raise ValueError(
                    "Foveal Aggregated Attention requires the complete dense patch "
                    "grid before block 1; disable early token masking."
                )
            if self.foveal_aggregated_attention_window_size < 3 or (
                self.foveal_aggregated_attention_window_size % 2 == 0
            ):
                raise ValueError(
                    "foveal_aggregated_attention_window_size must be odd and >= 3."
                )
            if (
                self.foveal_aggregated_attention_pool_size <= 0
                or self.foveal_aggregated_attention_pool_size
                >= int(image_size) // int(patch_size)
            ):
                raise ValueError(
                    "foveal_aggregated_attention_pool_size must be positive and "
                    "smaller than the base patch grid."
                )
            if self.visual_contrast_attention_enabled:
                raise ValueError(
                    "Foveal Aggregated Attention cannot be combined with "
                    "Visual-Contrast Attention."
                )
            if self.gated_relative_position_attention_enabled:
                raise ValueError(
                    "Foveal Aggregated Attention cannot be combined with gated "
                    "relative position attention."
                )
        self.deformable_spatial_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                deformable_spatial_attention_layers,
                int(depth),
            )
            if bool(deformable_spatial_attention)
            else []
        )
        self.deformable_spatial_attention_enabled = bool(
            self.deformable_spatial_attention_layer_numbers
        )
        self.deformable_spatial_attention_groups = int(
            deformable_spatial_attention_groups
        )
        self.deformable_spatial_attention_kernel_size = int(
            deformable_spatial_attention_kernel_size
        )
        self.deformable_spatial_attention_offset_range = float(
            deformable_spatial_attention_offset_range
        )
        if self.deformable_spatial_attention_enabled:
            if self.deformable_spatial_attention_layer_numbers != [2]:
                raise ValueError(
                    "Deformable Spatial Attention is locked to transformer layer 2."
                )
            if float(early_token_mask_keep_rate) < 1.0:
                raise ValueError(
                    "Deformable Spatial Attention requires the complete dense patch "
                    "grid before block 2; disable early token masking."
                )
            if self.deformable_spatial_attention_groups != 2:
                raise ValueError(
                    "The locked Deformable Spatial Attention route requires 2 groups."
                )
            if self.deformable_spatial_attention_kernel_size != 5:
                raise ValueError(
                    "The locked Deformable Spatial Attention route requires kernel 5."
                )
            if self.deformable_spatial_attention_offset_range != 2.0:
                raise ValueError(
                    "The locked Deformable Spatial Attention route requires range 2.0."
                )
            if self.foveal_aggregated_attention_enabled:
                raise ValueError(
                    "Deformable Spatial Attention cannot be combined with "
                    "Foveal Aggregated Attention."
                )
            if self.visual_contrast_attention_enabled:
                raise ValueError(
                    "Deformable Spatial Attention cannot be combined with "
                    "Visual-Contrast Attention."
                )
            if self.gated_relative_position_attention_enabled:
                raise ValueError(
                    "Deformable Spatial Attention cannot be combined with gated "
                    "relative position attention."
                )
        self.bi_level_routing_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                bi_level_routing_attention_layers,
                int(depth),
            )
            if bool(bi_level_routing_attention)
            else []
        )
        self.bi_level_routing_attention_enabled = bool(
            self.bi_level_routing_attention_layer_numbers
        )
        self.bi_level_routing_attention_regions_per_axis = int(
            bi_level_routing_attention_regions_per_axis
        )
        self.bi_level_routing_attention_topk = int(
            bi_level_routing_attention_topk
        )
        self.bi_level_routing_attention_local_context_kernel_size = int(
            bi_level_routing_attention_local_context_kernel_size
        )
        if self.bi_level_routing_attention_enabled:
            if self.bi_level_routing_attention_layer_numbers != [2]:
                raise ValueError(
                    "Bi-Level Routing Attention is locked to transformer layer 2."
                )
            if float(early_token_mask_keep_rate) < 1.0:
                raise ValueError(
                    "Bi-Level Routing Attention requires the complete dense patch "
                    "grid before block 2; disable early token masking."
                )
            if self.bi_level_routing_attention_regions_per_axis != 4:
                raise ValueError(
                    "The locked Bi-Level Routing Attention route requires S=4."
                )
            if self.bi_level_routing_attention_topk not in {4, 16}:
                raise ValueError(
                    "The locked Bi-Level Routing Attention roles require topk 4 or 16."
                )
            if self.bi_level_routing_attention_local_context_kernel_size != 5:
                raise ValueError(
                    "The locked Bi-Level Routing Attention route requires LCE kernel 5."
                )
            if self.foveal_aggregated_attention_enabled:
                raise ValueError(
                    "Bi-Level Routing Attention cannot be combined with "
                    "Foveal Aggregated Attention."
                )
            if self.deformable_spatial_attention_enabled:
                raise ValueError(
                    "Bi-Level Routing Attention cannot be combined with "
                    "Deformable Spatial Attention."
                )
            if self.visual_contrast_attention_enabled:
                raise ValueError(
                    "Bi-Level Routing Attention cannot be combined with "
                    "Visual-Contrast Attention."
                )
            if self.gated_relative_position_attention_enabled:
                raise ValueError(
                    "Bi-Level Routing Attention cannot be combined with gated "
                    "relative position attention."
                )
            if bool(soft_moe_patch_adapter):
                raise ValueError(
                    "Bi-Level Routing Attention cannot be combined with a "
                    "Soft-MoE patch adapter."
                )
            if bool(deep_class_prompt):
                raise ValueError(
                    "Bi-Level Routing Attention requires the locked seven-prefix "
                    "layout and cannot use deep class prompts."
                )
        self.cross_covariance_attention_layer_numbers = (
            _parse_auxiliary_layer_indices(
                cross_covariance_attention_layers,
                int(depth),
            )
            if bool(cross_covariance_attention)
            else []
        )
        self.cross_covariance_attention_enabled = bool(
            self.cross_covariance_attention_layer_numbers
        )
        self.cross_covariance_attention_residual_scale = float(
            cross_covariance_attention_residual_scale
        )
        if self.cross_covariance_attention_residual_scale < 0.0:
            raise ValueError("cross_covariance_attention_residual_scale must be >= 0.")
        if (
            self.cross_covariance_attention_enabled
            and self.visual_contrast_attention_enabled
        ):
            raise ValueError(
                "Cross-covariance attention cannot be combined with Visual-Contrast Attention."
            )
        if (
            self.cross_covariance_attention_enabled
            and self.foveal_aggregated_attention_enabled
        ):
            raise ValueError(
                "Cross-covariance attention cannot be combined with Foveal "
                "Aggregated Attention in the locked route."
            )
        if (
            self.cross_covariance_attention_enabled
            and self.deformable_spatial_attention_enabled
        ):
            raise ValueError(
                "Cross-covariance attention cannot be combined with Deformable "
                "Spatial Attention in the locked route."
            )
        if (
            self.cross_covariance_attention_enabled
            and self.bi_level_routing_attention_enabled
        ):
            raise ValueError(
                "Cross-covariance attention cannot be combined with Bi-Level "
                "Routing Attention in the locked route."
            )
        self.dynamic_graph_mixer_layer_numbers = (
            _parse_auxiliary_layer_indices(
                dynamic_graph_mixer_layers,
                int(depth),
            )
            if bool(dynamic_graph_mixer)
            else []
        )
        self.dynamic_graph_mixer_enabled = bool(
            self.dynamic_graph_mixer_layer_numbers
        )
        self.dynamic_graph_mixer_bottleneck_dim = int(
            dynamic_graph_mixer_bottleneck_dim
        )
        self.dynamic_graph_mixer_k = int(dynamic_graph_mixer_k)
        if self.dynamic_graph_mixer_enabled:
            if self.dynamic_graph_mixer_bottleneck_dim <= 0:
                raise ValueError("dynamic_graph_mixer_bottleneck_dim must be positive.")
            if int(embed_dim) % self.dynamic_graph_mixer_bottleneck_dim != 0:
                raise ValueError("dynamic_graph_mixer_bottleneck_dim must divide embed_dim.")
            if self.dynamic_graph_mixer_k <= 0:
                raise ValueError("dynamic_graph_mixer_k must be positive.")
            if self.visual_contrast_attention_enabled:
                raise ValueError(
                    "Dynamic graph mixing cannot be combined with Visual-Contrast Attention."
                )
            if self.foveal_aggregated_attention_enabled:
                raise ValueError(
                    "Dynamic graph mixing cannot be combined with Foveal "
                    "Aggregated Attention in the locked route."
                )
            if self.deformable_spatial_attention_enabled:
                raise ValueError(
                    "Dynamic graph mixing cannot be combined with Deformable "
                    "Spatial Attention in the locked route."
                )
            if self.bi_level_routing_attention_enabled:
                raise ValueError(
                    "Dynamic graph mixing cannot be combined with Bi-Level "
                    "Routing Attention in the locked route."
                )
            if self.cross_covariance_attention_enabled:
                raise ValueError(
                    "Dynamic graph mixing cannot be combined with cross-covariance attention."
                )
            if any(
                layer in self.block_local_patch_mixer_layer_numbers
                for layer in self.dynamic_graph_mixer_layer_numbers
            ):
                raise ValueError(
                    "Dynamic graph mixing cannot overlap block-local patch mixer layers."
                )
        self.soft_moe_patch_adapter_layer_numbers = (
            _parse_auxiliary_layer_indices(
                soft_moe_patch_adapter_layers,
                int(depth),
            )
            if bool(soft_moe_patch_adapter)
            else []
        )
        self.soft_moe_patch_adapter_enabled = bool(
            self.soft_moe_patch_adapter_layer_numbers
        )
        self.soft_moe_hidden_dim = int(soft_moe_hidden_dim)
        self.soft_moe_num_experts = int(soft_moe_num_experts)
        self.soft_moe_residual_scale = float(soft_moe_residual_scale)
        self.soft_moe_router_scale_init = float(soft_moe_router_scale_init)
        self.soft_moe_init_seed = int(soft_moe_init_seed)
        if self.soft_moe_patch_adapter_enabled:
            if self.soft_moe_hidden_dim <= 0:
                raise ValueError("soft_moe_hidden_dim must be positive.")
            if self.soft_moe_num_experts <= 1:
                raise ValueError("soft_moe_num_experts must be at least two.")
            if self.soft_moe_residual_scale < 0.0:
                raise ValueError("soft_moe_residual_scale must be >= 0.")
            if self.soft_moe_router_scale_init <= 0.0:
                raise ValueError("soft_moe_router_scale_init must be positive.")
        self.patch_style_recalibration_layer_numbers = (
            _parse_auxiliary_layer_indices(
                patch_style_recalibration_layers,
                int(depth),
            )
            if bool(patch_style_recalibration)
            else []
        )
        self.patch_style_recalibration_enabled = bool(
            self.patch_style_recalibration_layer_numbers
        )
        if self.patch_style_recalibration_enabled and bool(locally_enhanced_ffn):
            raise ValueError(
                "Patch-style recalibration cannot be combined with locally enhanced FFN."
            )
        self.layer_token_fusion_enabled = bool(layer_token_fusion)
        self.layer_token_fusion_layer_numbers = (
            _parse_auxiliary_layer_indices(layer_token_fusion_layers, int(depth))
            if self.layer_token_fusion_enabled
            else []
        )
        self.layer_token_fusion_top_k = max(1, int(layer_token_fusion_top_k))
        self.layer_token_fusion_blend = float(
            min(1.0, max(0.0, layer_token_fusion_blend))
        )
        self.layer_token_fusion_attention_temperature = float(
            max(1e-4, layer_token_fusion_attention_temperature)
        )
        self.layer_token_fusion_bbox_weight = float(max(0.0, layer_token_fusion_bbox_weight))
        self.layer_token_fusion_foreground_weight = float(
            max(0.0, layer_token_fusion_foreground_weight)
        )
        if not self.layer_token_fusion_layer_numbers or self.layer_token_fusion_blend <= 0.0:
            self.layer_token_fusion_enabled = False
        self.masked_reconstruction_head_enabled = bool(masked_reconstruction_head)
        self.frequency_selective_pooling = bool(frequency_selective_pooling)
        self.frequency_selective_blend = float(
            min(max(frequency_selective_blend, 0.0), 1.0)
        )
        self.late_class_attention_pooling_enabled = bool(late_class_attention_pooling)
        self.late_member_enabled = bool(late_member_branch)
        self.late_member_fork_after_block = int(late_member_fork_after_block)
        if self.late_member_enabled and not 1 <= self.late_member_fork_after_block < int(depth):
            raise ValueError(
                "late_member_fork_after_block must leave at least one shared and one "
                f"member-specific block; got fork={self.late_member_fork_after_block}, "
                f"depth={int(depth)}."
            )
        self.late_member_candidate_weight = float(late_member_candidate_weight)
        if not 0.0 <= self.late_member_candidate_weight <= 1.0:
            raise ValueError("late_member_candidate_weight must be in [0, 1].")
        self.late_member_focus_class = int(late_member_focus_class)
        if not 0 <= self.late_member_focus_class < int(num_classes):
            raise ValueError("late_member_focus_class is outside the class range.")
        self.late_member_focus_margin_offset = float(late_member_focus_margin_offset)
        if not 0.0 <= self.late_member_focus_margin_offset < 1.0:
            raise ValueError("late_member_focus_margin_offset must be in [0, 1).")
        self.mixstyle_enabled = bool(mixstyle and use_cnn_stem)
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
        self.topk_reassessment_head_enabled = bool(topk_reassessment_head)
        self.topk_reassessment_logit_scale = float(max(0.0, topk_reassessment_logit_scale))
        self.topk_reassessment_routing = bool(topk_reassessment_routing)
        self.topk_reassessment_route_max_probability_margin = float(
            max(0.0, topk_reassessment_route_max_probability_margin)
        )
        self.topk_reassessment_route_pairs = (
            _parse_pairwise_margin_pairs(topk_reassessment_route_pairs, num_classes)
            if self.topk_reassessment_head_enabled and str(topk_reassessment_route_pairs).strip()
            else []
        )
        self.focus_class_head_enabled = bool(focus_class_head)
        self.focus_class_index = int(focus_class_index)
        if not 0 <= self.focus_class_index < int(num_classes):
            self.focus_class_head_enabled = False
        self.focus_class_logit_scale = float(max(0.0, focus_class_logit_scale))
        self.focus_class_routing = bool(focus_class_routing)
        self.focus_class_route_max_probability_margin = float(
            max(0.0, focus_class_route_max_probability_margin)
        )
        self.focus_class_route_min_probability = float(
            min(1.0, max(0.0, focus_class_route_min_probability))
        )
        self.class_independent_head_enabled = bool(class_independent_head)
        self.ordinal_maturity_head_enabled = bool(ordinal_maturity_head)
        self.ordinal_maturity_logit_scale = float(max(0.0, ordinal_maturity_logit_scale))
        self.ordinal_maturity_classes = (
            _parse_ordered_class_indices(ordinal_maturity_classes, num_classes)
            if self.ordinal_maturity_head_enabled
            else []
        )
        self.cumulative_ordinal_head_enabled = bool(cumulative_ordinal_head)
        self.cumulative_ordinal_logit_scale = float(max(0.0, cumulative_ordinal_logit_scale))
        self.cumulative_ordinal_classes = (
            _parse_ordered_class_indices(cumulative_ordinal_classes, num_classes)
            if self.cumulative_ordinal_head_enabled
            else []
        )
        self.token_pruning = bool(token_pruning)
        self.token_prune_foreground_weight = float(max(0.0, token_prune_foreground_weight))
        self.token_prune_bbox_weight = float(max(0.0, token_prune_bbox_weight))
        self.token_prune_bbox_margin_ratio = float(max(0.0, token_prune_bbox_margin_ratio))
        self.early_token_mask_keep_rate = float(
            min(1.0, max(0.0, early_token_mask_keep_rate))
        )
        self.inattentive_token_fusion_enabled = bool(inattentive_token_fusion)
        self.cropr_token_selector_enabled = bool(cropr_token_selector)
        self.cropr_token_selector_routing = bool(cropr_token_selector_routing)
        self.token_prune_schedule = (
            self._parse_token_prune_schedule(
                depth=int(depth),
                layers=token_prune_layers,
                keep_rates=token_keep_rates,
            )
            if self.token_pruning
            else {}
        )
        if self.deformable_spatial_attention_enabled and any(
            int(layer_index) < 1 for layer_index in self.token_prune_schedule
        ):
            raise ValueError(
                "Deformable Spatial Attention requires no pruning before block 2."
            )
        if self.bi_level_routing_attention_enabled and any(
            int(layer_index) < 1 for layer_index in self.token_prune_schedule
        ):
            raise ValueError(
                "Bi-Level Routing Attention requires no pruning before block 2."
            )
        if not self.token_prune_schedule and self.early_token_mask_keep_rate >= 1.0:
            self.token_pruning = False
        if self.inattentive_token_fusion_enabled:
            if not self.token_pruning or not self.token_prune_schedule:
                raise ValueError(
                    "inattentive_token_fusion requires attention-based token pruning."
                )
            if self.early_token_mask_keep_rate < 1.0:
                raise ValueError(
                    "inattentive_token_fusion does not support early token masking."
                )
            incompatible = []
            if self.deep_class_prompt_enabled:
                incompatible.append("deep_class_prompt")
            if self.concurrent_local_global_enabled:
                incompatible.append("concurrent_local_global_coupling")
            if self.late_member_enabled:
                incompatible.append("late_member_branch")
            if incompatible:
                raise ValueError(
                    "inattentive_token_fusion is incompatible with dynamic-prefix "
                    f"consumers: {', '.join(incompatible)}."
                )
        if self.cropr_token_selector_enabled:
            if not self.token_pruning or not self.token_prune_schedule:
                raise ValueError(
                    "cropr_token_selector requires attention-based token pruning."
                )
            expected_schedule = {1: 0.85, 4: 0.65}
            if set(self.token_prune_schedule) != set(expected_schedule) or any(
                not math.isclose(
                    float(self.token_prune_schedule[layer]),
                    expected_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for layer, expected_rate in expected_schedule.items()
            ):
                raise ValueError(
                    "cropr_token_selector A0 is locked to prune layers 2,5 and "
                    "keep rates 0.85,0.65."
                )
            if not math.isclose(
                self.token_prune_foreground_weight,
                0.35,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "cropr_token_selector A0 requires foreground weight 0.35."
                )
            if self.early_token_mask_keep_rate < 1.0:
                raise ValueError(
                    "cropr_token_selector A0 does not support early token masking."
                )
            if self.inattentive_token_fusion_enabled:
                raise ValueError(
                    "cropr_token_selector cannot be combined with inattentive token fusion."
                )
            incompatible = []
            if self.deep_class_prompt_enabled:
                incompatible.append("deep_class_prompt")
            if self.concurrent_local_global_enabled:
                incompatible.append("concurrent_local_global_coupling")
            if self.late_member_enabled:
                incompatible.append("late_member_branch")
            if incompatible:
                raise ValueError(
                    "cropr_token_selector is incompatible with dynamic-prefix "
                    f"consumers: {', '.join(incompatible)}."
                )
        if self.head_pooling not in {"cls", "cls_register_mean", "cls_branch_register_mean"}:
            raise ValueError(f"Khong ho tro head_pooling={head_pooling!r}.")

        if use_cnn_stem:
            if self.stem_architecture == "inceptionnext_atto_tokenizer":
                self.stem = InceptionNeXtAttoTokenizer(in_channels=in_channels)
            elif self.stem_architecture == "moganet_xt_tokenizer":
                self.stem = MogaXTTokenizer(in_channels=in_channels)
            elif self.stem_architecture == "starnet_s2_tokenizer":
                self.stem = StarNetS2Tokenizer(in_channels=in_channels)
            elif self.stem_architecture == "octave_conv":
                # Match legacy construction RNG so all downstream scratch
                # parameters remain paired under the same seed.
                with torch.random.fork_rng(devices=[]):
                    self.stem = OctaveConvStem(
                        in_channels=in_channels,
                        stem_channels=stem_channels,
                        embed_dim=embed_dim,
                    )
                constructor_rng_proxy = HybridConvStem(
                    in_channels=in_channels,
                    stem_channels=stem_channels,
                    embed_dim=embed_dim,
                    pooling_mode=self.stem_pooling_mode,
                    softpool_blend=self.stem_softpool_blend,
                )
                del constructor_rng_proxy
            elif self.stem_architecture == "dbb_conv_pool":
                if self.stem_pooling_mode != "max":
                    raise ValueError(
                        "dbb_conv_pool requires the locked stem_pooling_mode=max."
                    )
                with torch.random.fork_rng(devices=[]):
                    self.stem = DiverseBranchConvStem(
                        in_channels=in_channels,
                        stem_channels=stem_channels,
                        embed_dim=embed_dim,
                    )
                constructor_rng_proxy = HybridConvStem(
                    in_channels=in_channels,
                    stem_channels=stem_channels,
                    embed_dim=embed_dim,
                    pooling_mode=self.stem_pooling_mode,
                    softpool_blend=self.stem_softpool_blend,
                )
                del constructor_rng_proxy
            elif self.stem_architecture == "coatnet_mbconv":
                if int(in_channels) != 3:
                    raise ValueError("coatnet_mbconv stem requires three-channel RGB input.")
                self.stem = CoAtNetMBConvStem(embed_dim=embed_dim)
            else:
                self.stem = HybridConvStem(
                    in_channels=in_channels,
                    stem_channels=stem_channels,
                    embed_dim=embed_dim,
                    pooling_mode=self.stem_pooling_mode,
                    softpool_blend=self.stem_softpool_blend,
                    normalization=self.stem_normalization,
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
        self.mixstyle = (
            MixStyle(
                probability=mixstyle_probability,
                alpha=mixstyle_alpha,
            )
            if self.mixstyle_enabled
            else nn.Identity()
        )

        self.patch_embed = PatchEmbedding(
            image_size=patch_embed_image_size,
            patch_size=patch_embed_patch_size,
            in_channels=patch_embed_channels,
            embed_dim=embed_dim,
            dual_patch_norm=dual_patch_norm,
        )
        self.shifted_patch_token_residual = (
            ShiftedPatchTokenResidual(
                in_channels=patch_embed_channels,
                embed_dim=embed_dim,
                patch_size=patch_embed_patch_size,
                shift=self.shifted_patch_shift,
                residual_scale=self.shifted_patch_residual_scale,
            )
            if self.shifted_patch_tokenization_enabled
            else None
        )
        self.input_patch_size = int(patch_size)
        self.input_patch_dim = int(in_channels) * self.input_patch_size * self.input_patch_size
        self.masked_reconstruction_head = (
            MaskedPatchReconstructionHead(
                input_dim=embed_dim,
                hidden_dim=int(max(16, masked_reconstruction_hidden_dim)),
                patch_dim=self.input_patch_dim,
            )
            if self.masked_reconstruction_head_enabled
            else None
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
        if bool(learnable_gabor_texture_residual) and bool(
            learnable_gabor_texture_semantic_fusion
        ):
            raise ValueError(
                "learnable Gabor edge-token residual and semantic fusion are "
                "mutually exclusive."
            )
        if bool(learnable_gabor_texture_residual):
            if self.branch_token_fusion is None or self.branch_token_fusion.edge_tokens <= 0:
                raise ValueError(
                    "learnable_gabor_texture_residual requires multi-branch fusion "
                    "with at least one edge token."
                )
            self.gabor_texture_edge_token_index = int(
                self.branch_token_fusion.color_tokens
            )
            self.gabor_texture_residual = LearnableGaborTextureResidual(
                embed_dim=embed_dim
            )
        else:
            self.gabor_texture_edge_token_index = -1
            self.gabor_texture_residual = None
        self.gabor_texture_semantic_encoder = (
            LearnableGaborTextureEncoder(embed_dim=embed_dim)
            if bool(learnable_gabor_texture_semantic_fusion)
            else None
        )
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
                    local_patch_mixer=(
                        int(index + 1) in self.block_local_patch_mixer_layer_numbers
                    ),
                    local_patch_mixer_dropout=block_local_patch_mixer_dropout,
                    local_patch_mixer_scale=block_local_patch_mixer_scale,
                    locally_enhanced_ffn=(
                        int(index + 1) in self.locally_enhanced_ffn_layer_numbers
                    ),
                    locally_enhanced_ffn_kernel_size=(
                        locally_enhanced_ffn_kernel_size
                    ),
                    gated_relative_position_attention=(
                        int(index + 1)
                        in self.gated_relative_position_attention_layer_numbers
                    ),
                    relative_position_max_mix=(
                        gated_relative_position_attention_max_mix
                    ),
                    relative_position_locality_strength=(
                        gated_relative_position_attention_locality_strength
                    ),
                    visual_contrast_attention=(
                        int(index + 1)
                        in self.visual_contrast_attention_layer_numbers
                    ),
                    visual_contrast_tokens=self.visual_contrast_tokens,
                    foveal_aggregated_attention=(
                        int(index + 1)
                        in self.foveal_aggregated_attention_layer_numbers
                    ),
                    foveal_aggregated_attention_input_resolution=(
                        self.patch_embed.base_grid_size
                    ),
                    foveal_aggregated_attention_window_size=(
                        self.foveal_aggregated_attention_window_size
                    ),
                    foveal_aggregated_attention_pool_size=(
                        self.foveal_aggregated_attention_pool_size
                    ),
                    deformable_spatial_attention=(
                        int(index + 1)
                        in self.deformable_spatial_attention_layer_numbers
                    ),
                    deformable_spatial_attention_input_resolution=(
                        self.patch_embed.base_grid_size
                    ),
                    deformable_spatial_attention_groups=(
                        self.deformable_spatial_attention_groups
                    ),
                    deformable_spatial_attention_kernel_size=(
                        self.deformable_spatial_attention_kernel_size
                    ),
                    deformable_spatial_attention_offset_range=(
                        self.deformable_spatial_attention_offset_range
                    ),
                    bi_level_routing_attention=(
                        int(index + 1)
                        in self.bi_level_routing_attention_layer_numbers
                    ),
                    bi_level_routing_attention_input_resolution=(
                        self.patch_embed.base_grid_size
                    ),
                    bi_level_routing_attention_regions_per_axis=(
                        self.bi_level_routing_attention_regions_per_axis
                    ),
                    bi_level_routing_attention_topk=(
                        self.bi_level_routing_attention_topk
                    ),
                    bi_level_routing_attention_local_context_kernel_size=(
                        self.bi_level_routing_attention_local_context_kernel_size
                    ),
                    cross_covariance_attention=(
                        int(index + 1)
                        in self.cross_covariance_attention_layer_numbers
                    ),
                    cross_covariance_attention_residual_scale=(
                        self.cross_covariance_attention_residual_scale
                    ),
                    dynamic_graph_mixer=(
                        int(index + 1) in self.dynamic_graph_mixer_layer_numbers
                    ),
                    dynamic_graph_mixer_bottleneck_dim=(
                        self.dynamic_graph_mixer_bottleneck_dim
                    ),
                    dynamic_graph_mixer_k=self.dynamic_graph_mixer_k,
                    soft_moe_patch_adapter=(
                        int(index + 1)
                        in self.soft_moe_patch_adapter_layer_numbers
                    ),
                    soft_moe_hidden_dim=self.soft_moe_hidden_dim,
                    soft_moe_num_experts=self.soft_moe_num_experts,
                    soft_moe_residual_scale=self.soft_moe_residual_scale,
                    soft_moe_router_scale_init=self.soft_moe_router_scale_init,
                    soft_moe_init_seed=self.soft_moe_init_seed,
                    patch_style_recalibration=(
                        int(index + 1)
                        in self.patch_style_recalibration_layer_numbers
                    ),
                    block_depth=index,
                )
                for index in range(depth)
            ]
        )
        self.concurrent_local_initializer = (
            ConcurrentLocalInitializer(
                in_channels=patch_embed_channels,
                local_dim=self.concurrent_local_global_dim,
            )
            if self.concurrent_local_global_enabled
            else None
        )
        self.concurrent_local_couplings = nn.ModuleDict(
            {
                str(layer_number): ConcurrentLocalGlobalCoupling(
                    embed_dim=embed_dim,
                    local_dim=self.concurrent_local_global_dim,
                    kernel_size=self.concurrent_local_global_kernel_size,
                )
                for layer_number in self.concurrent_local_global_layer_numbers
            }
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.multi_granularity_aux_heads = nn.ModuleDict(
            {
                str(layer_number): nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(float(max(0.0, multi_granularity_aux_dropout))),
                    nn.Linear(embed_dim, num_classes),
                )
                for layer_number in self.multi_granularity_aux_layer_numbers
            }
        )
        self.self_boosting_attention_projection = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1),
            )
            if self.self_boosting_attention_head_enabled
            else None
        )
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
        self.patch_memory_adapter = (
            PatchMemoryAdapter(
                embed_dim=embed_dim,
                dropout=patch_memory_adapter_dropout,
                zero_init=True,
            )
            if bool(patch_memory_adapter)
            else None
        )
        self.late_class_attention_pool = (
            LateClassAttentionPooling(
                dim=embed_dim,
                num_heads=late_class_attention_heads,
                dropout=late_class_attention_dropout,
                mlp_ratio=late_class_attention_mlp_ratio,
                residual_scale=late_class_attention_residual_scale,
            )
            if self.late_class_attention_pooling_enabled
            else None
        )
        self.head = (
            ResidualMLPClassificationHead(
                in_features=embed_dim,
                num_classes=num_classes,
                hidden_dim=classification_mlp_hidden_dim,
                dropout=classification_mlp_dropout,
                residual_scale=classification_mlp_residual_scale,
            )
            if self.classification_mlp_head_enabled
            else nn.Linear(embed_dim, num_classes)
        )
        self.teacher_feature_projection_adapter = (
            TeacherFeatureProjectionAdapter(
                student_dim=int(embed_dim),
                teacher_dim=int(teacher_feature_projection_teacher_dim),
                projection_dim=int(teacher_feature_projection_dim),
                dropout=float(teacher_feature_projection_dropout),
            )
            if bool(teacher_feature_projection_adapter)
            and int(teacher_feature_projection_teacher_dim) > 0
            and int(teacher_feature_projection_dim) > 0
            else None
        )
        self.subcenter_proxy_head = (
            SubCenterProxyHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                subcenters=subcenter_proxy_subcenters,
                dropout=subcenter_proxy_dropout,
                init_std=subcenter_proxy_init_std,
            )
            if self.subcenter_proxy_head_enabled
            else None
        )
        if self.deep_abstention_head_enabled:
            initial_probability = min(
                max(float(deep_abstention_initial_probability), 1e-5),
                1.0 - 1e-5,
            )
            self.deep_abstention_norm = nn.LayerNorm(embed_dim)
            self.deep_abstention_dropout = nn.Dropout(
                float(max(0.0, deep_abstention_dropout))
            )
            self.deep_abstention_head = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.deep_abstention_head.weight)
            nn.init.constant_(
                self.deep_abstention_head.bias,
                math.log(initial_probability / (1.0 - initial_probability)),
            )
        else:
            self.deep_abstention_norm = None
            self.deep_abstention_dropout = None
            self.deep_abstention_head = None
        if self.pairwise_margin_pairs:
            self.pairwise_margin_norm = nn.LayerNorm(embed_dim)
            self.pairwise_margin_dropout = nn.Dropout(pairwise_margin_dropout)
            self.pairwise_margin_head = nn.Linear(embed_dim, len(self.pairwise_margin_pairs))
        else:
            self.pairwise_margin_norm = None
            self.pairwise_margin_dropout = None
            self.pairwise_margin_head = None
        if self.topk_reassessment_head_enabled:
            self.topk_reassessment_head = TopKReassessmentHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                top_k=topk_reassessment_top_k,
                hidden_dim=topk_reassessment_hidden_dim,
                dropout=topk_reassessment_dropout,
            )
        else:
            self.topk_reassessment_head = None
        if self.focus_class_head_enabled:
            self.focus_class_norm = nn.LayerNorm(embed_dim)
            self.focus_class_dropout = nn.Dropout(focus_class_dropout)
            self.focus_class_head = nn.Linear(embed_dim, 1)
        else:
            self.focus_class_norm = None
            self.focus_class_dropout = None
            self.focus_class_head = None
        if self.class_independent_head_enabled:
            self.class_independent_norm = nn.LayerNorm(embed_dim)
            self.class_independent_dropout = nn.Dropout(class_independent_dropout)
            self.class_independent_head = nn.Linear(embed_dim, num_classes)
        else:
            self.class_independent_norm = None
            self.class_independent_dropout = None
            self.class_independent_head = None
        if self.ordinal_maturity_classes:
            self.ordinal_maturity_norm = nn.LayerNorm(embed_dim)
            self.ordinal_maturity_dropout = nn.Dropout(ordinal_maturity_dropout)
            self.ordinal_maturity_head = nn.Linear(embed_dim, 1)
        else:
            self.ordinal_maturity_norm = None
            self.ordinal_maturity_dropout = None
            self.ordinal_maturity_head = None
        if len(self.cumulative_ordinal_classes) >= 2:
            self.cumulative_ordinal_norm = nn.LayerNorm(embed_dim)
            self.cumulative_ordinal_dropout = nn.Dropout(cumulative_ordinal_dropout)
            self.cumulative_ordinal_head = nn.Linear(
                embed_dim,
                len(self.cumulative_ordinal_classes) - 1,
            )
        else:
            self.cumulative_ordinal_norm = None
            self.cumulative_ordinal_dropout = None
            self.cumulative_ordinal_head = None
        if self.cnn_feature_fusion:
            self.cnn_fusion_norm = nn.LayerNorm(patch_embed_channels)
            self.cnn_fusion_dropout = nn.Dropout(float(max(0.0, cnn_fusion_dropout)))
            self.cnn_fusion_head = nn.Linear(patch_embed_channels, num_classes)
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
        if self.foreground_surface_pairwise_pairs:
            self.foreground_surface_pairwise_head = ForegroundSurfacePairwiseHead(
                pair_count=len(self.foreground_surface_pairwise_pairs),
                hidden_dim=max(32, embed_dim // 2),
                dropout=foreground_surface_pairwise_dropout,
            )
        else:
            self.foreground_surface_pairwise_head = None
        if self.interior_boundary_pairwise_pairs:
            self.interior_boundary_pairwise_head = InteriorBoundaryPairwiseHead(
                pair_count=len(self.interior_boundary_pairwise_pairs),
                hidden_dim=interior_boundary_pairwise_hidden_dim,
                dropout=interior_boundary_pairwise_dropout,
                erode_kernel=interior_boundary_pairwise_erode_kernel,
            )
        else:
            self.interior_boundary_pairwise_head = None
        if self.bbox_spatial_fusion_enabled:
            self.bbox_spatial_fusion_head = BBoxSpatialPriorFusion(
                num_classes=num_classes,
                hidden_dim=bbox_spatial_fusion_hidden_dim,
                dropout=bbox_spatial_fusion_dropout,
            )
        else:
            self.bbox_spatial_fusion_head = None
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
        if self.complementary_patch_suppression_head_enabled:
            self.complementary_patch_suppression_head = ComplementaryPatchSuppressionHead(
                dim=embed_dim,
                num_classes=num_classes,
                top_k=complementary_patch_suppression_top_k,
                hidden_dim=complementary_patch_suppression_hidden_dim,
                dropout=complementary_patch_suppression_dropout,
                temperature=complementary_patch_suppression_temperature,
                suppression_strength=complementary_patch_suppression_strength,
                bbox_weight=complementary_patch_suppression_bbox_weight,
            )
        else:
            self.complementary_patch_suppression_head = None
        if self.patch_objectness_guided_head_enabled:
            self.patch_objectness_guided_head = PatchObjectnessGuidedHead(
                dim=embed_dim,
                num_classes=num_classes,
                hidden_dim=patch_objectness_hidden_dim,
                dropout=patch_objectness_dropout,
                temperature=patch_objectness_temperature,
            )
        else:
            self.patch_objectness_guided_head = None
        if self.bbox_prior_patch_context_head_enabled:
            self.bbox_prior_patch_context_head = BBoxPriorPatchContextHead(
                dim=embed_dim,
                num_classes=num_classes,
                hidden_dim=bbox_prior_patch_context_hidden_dim,
                dropout=bbox_prior_patch_context_dropout,
                temperature=bbox_prior_patch_context_temperature,
            )
        else:
            self.bbox_prior_patch_context_head = None
        if self.source_context_feature_fusion_enabled:
            self.source_context_fusion_head = SourceContextFeatureFusion(
                feature_dim=embed_dim,
                num_classes=num_classes,
                hidden_dim=source_context_fusion_hidden_dim,
                dropout=source_context_fusion_dropout,
                gate_bias=source_context_fusion_gate_bias,
            )
        else:
            self.source_context_fusion_head = None
        if self.paired_view_feature_fusion_enabled:
            self.paired_view_fusion_head = SourceContextFeatureFusion(
                feature_dim=embed_dim,
                num_classes=num_classes,
                hidden_dim=paired_view_fusion_hidden_dim,
                dropout=paired_view_fusion_dropout,
                gate_bias=paired_view_fusion_gate_bias,
            )
        else:
            self.paired_view_fusion_head = None
        if self.micro_detail_patch_expert_enabled:
            self.micro_detail_patch_expert = MicroDetailPatchExpert(
                dim=embed_dim,
                num_classes=num_classes,
                top_k=micro_detail_top_k,
                hidden_dim=micro_detail_hidden_dim,
                dropout=micro_detail_dropout,
                temperature=micro_detail_temperature,
                foreground_power=micro_detail_foreground_power,
            )
        else:
            self.micro_detail_patch_expert = None
        if self.part_token_learner_enabled:
            self.part_token_learner = AdaptivePartTokenLearner(
                dim=embed_dim,
                num_classes=num_classes,
                part_count=part_token_count,
                hidden_dim=part_token_hidden_dim,
                dropout=part_token_dropout,
                temperature=part_token_temperature,
                foreground_power=part_token_foreground_power,
                bbox_weight=part_token_bbox_weight,
            )
        else:
            self.part_token_learner = None
        if self.part_token_pairwise_pairs:
            self.part_token_pairwise_learner = AdaptivePartTokenLearner(
                dim=embed_dim,
                num_classes=len(self.part_token_pairwise_pairs),
                part_count=part_token_count,
                hidden_dim=part_token_hidden_dim,
                dropout=part_token_pairwise_dropout,
                temperature=part_token_temperature,
                foreground_power=part_token_foreground_power,
                bbox_weight=part_token_bbox_weight,
            )
        else:
            self.part_token_pairwise_learner = None
        if self.patch_evidence_router_pairs:
            self.patch_evidence_router_head = PatchEvidenceRouterHead(
                dim=embed_dim,
                hidden_dim=patch_evidence_router_hidden_dim,
                dropout=patch_evidence_router_dropout,
                top_k=patch_evidence_router_top_k,
                bbox_weight=patch_evidence_router_bbox_weight,
                margin_prior_mode=self.patch_evidence_router_margin_prior_mode,
                margin_prior_scale=self.patch_evidence_router_margin_prior_scale,
                summary_stats=self.patch_evidence_router_summary_stats,
            )
        else:
            self.patch_evidence_router_head = None
        self.patch_evidence_linear_verifier: Optional[PatchEvidenceLinearVerifier] = None
        if self.local_zoom_image_expert_enabled:
            self.local_zoom_image_expert = LocalZoomImageExpert(
                num_classes=num_classes,
                hidden_dim=local_zoom_hidden_dim,
                crop_size=local_zoom_crop_size,
                crop_scale=local_zoom_crop_scale,
                score_mode=local_zoom_score_mode,
                dropout=local_zoom_dropout,
            )
        else:
            self.local_zoom_image_expert = None
        if self.high_frequency_texture_expert_enabled:
            self.high_frequency_texture_expert = HighFrequencyTextureExpert(
                num_classes=num_classes,
                hidden_dim=high_frequency_texture_hidden_dim,
                dropout=high_frequency_texture_dropout,
                analysis_size=high_frequency_texture_analysis_size,
            )
        else:
            self.high_frequency_texture_expert = None

        gabor_modules = [
            module
            for module in (
                self.gabor_texture_residual,
                self.gabor_texture_semantic_encoder,
            )
            if module is not None
        ]
        soft_moe_modules = [
            block.soft_moe_patch_adapter
            for block in self.blocks
            if block.soft_moe_patch_adapter is not None
        ]
        foveal_extra_modules = []
        deformable_extra_modules = []
        for block in self.blocks:
            attention_module = block.attn
            if isinstance(attention_module, FovealAggregatedAttention):
                foveal_extra_modules.extend(
                    (
                        attention_module.sr,
                        attention_module.pool_norm,
                        attention_module.cpb_fc1,
                        attention_module.cpb_fc2,
                    )
                )
            if isinstance(attention_module, DeformableSpatialAttention):
                deformable_extra_modules.append(attention_module.conv_offset)
        if isinstance(self.stem, (OctaveConvStem, DiverseBranchConvStem)):
            isolated_stem_module_ids = {id(child) for child in self.stem.modules()}
            gabor_module_ids = {
                id(child)
                for gabor_module in gabor_modules
                for child in gabor_module.modules()
            }
            soft_moe_module_ids = {
                id(child)
                for soft_moe_module in soft_moe_modules
                for child in soft_moe_module.modules()
            }
            foveal_extra_module_ids = {
                id(child)
                for foveal_extra_module in foveal_extra_modules
                for child in foveal_extra_module.modules()
            }
            deformable_extra_module_ids = {
                id(child)
                for deformable_extra_module in deformable_extra_modules
                for child in deformable_extra_module.modules()
            }
            excluded_module_ids = (
                isolated_stem_module_ids
                | gabor_module_ids
                | soft_moe_module_ids
                | foveal_extra_module_ids
                | deformable_extra_module_ids
            )
            with torch.random.fork_rng(devices=[]):
                legacy_init_proxy = HybridConvStem(
                    in_channels=in_channels,
                    stem_channels=stem_channels,
                    embed_dim=embed_dim,
                    pooling_mode=self.stem_pooling_mode,
                    softpool_blend=self.stem_softpool_blend,
                )
            isolated_stem_init_rng_state = torch.get_rng_state().clone()
            legacy_init_proxy.apply(self._init_weights)
            if isinstance(self.stem, DiverseBranchConvStem):
                self.stem.reset_origin_from_control(legacy_init_proxy)
            del legacy_init_proxy

            def init_non_isolated_stem_module(module: nn.Module) -> None:
                if id(module) not in excluded_module_ids:
                    self._init_weights(module)

            self.apply(init_non_isolated_stem_module)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(isolated_stem_init_rng_state)
                if isinstance(self.stem, OctaveConvStem):
                    self.stem.reset_from_virtual_vanilla_kernels()
                else:
                    self.stem.reset_candidate_only_parameters(self._init_weights)
            if gabor_modules:
                with torch.random.fork_rng(devices=[]):
                    for gabor_module in gabor_modules:
                        gabor_module.apply(self._init_weights)
        elif (
            not gabor_modules
            and not soft_moe_modules
            and not foveal_extra_modules
            and not deformable_extra_modules
        ):
            self.apply(self._init_weights)
        else:
            isolated_module_ids = {
                id(child)
                for isolated_module in (
                    *gabor_modules,
                    *soft_moe_modules,
                    *foveal_extra_modules,
                    *deformable_extra_modules,
                )
                for child in isolated_module.modules()
            }

            def init_base_module(module: nn.Module) -> None:
                if id(module) not in isolated_module_ids:
                    self._init_weights(module)

            self.apply(init_base_module)
            # Keep the optional branch on the repository's standard
            # initialization while preserving the base model's RNG stream.
            if gabor_modules:
                with torch.random.fork_rng(devices=[]):
                    for gabor_module in gabor_modules:
                        gabor_module.apply(self._init_weights)
        if isinstance(self.stem, (InceptionNeXtAttoTokenizer, StarNetS2Tokenizer)):
            self.stem.reset_parameters()
        self._init_parameter_tensors()
        if self.deep_abstention_head is not None:
            initial_probability = min(
                max(float(deep_abstention_initial_probability), 1e-5),
                1.0 - 1e-5,
            )
            nn.init.zeros_(self.deep_abstention_head.weight)
            nn.init.constant_(
                self.deep_abstention_head.bias,
                math.log(initial_probability / (1.0 - initial_probability)),
            )
        if self.shifted_patch_token_residual is not None:
            self.shifted_patch_token_residual.zero_init_residual()
        if isinstance(self.head, ResidualMLPClassificationHead):
            self.head.zero_init_residual()
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
        if self.foreground_surface_pairwise_head is not None:
            self.foreground_surface_pairwise_head.zero_init_residual()
        if self.interior_boundary_pairwise_head is not None:
            self.interior_boundary_pairwise_head.zero_init_residual()
        if self.bbox_spatial_fusion_head is not None:
            self.bbox_spatial_fusion_head.zero_init_residual()
        if self.bilinear_patch_fusion_head is not None:
            self.bilinear_patch_fusion_head.zero_init_residual()
        if self.complementary_patch_suppression_head is not None:
            self.complementary_patch_suppression_head.zero_init_residual()
        if self.patch_objectness_guided_head is not None:
            self.patch_objectness_guided_head.zero_init_residual()
        if self.bbox_prior_patch_context_head is not None:
            self.bbox_prior_patch_context_head.zero_init_residual()
        if self.source_context_fusion_head is not None:
            self.source_context_fusion_head.zero_init_residual()
        if self.paired_view_fusion_head is not None:
            self.paired_view_fusion_head.zero_init_residual()
        if self.micro_detail_patch_expert is not None:
            self.micro_detail_patch_expert.zero_init_residual()
        if self.part_token_learner is not None:
            self.part_token_learner.zero_init_residual()
        if self.part_token_pairwise_learner is not None:
            self.part_token_pairwise_learner.zero_init_residual()
        if self.patch_evidence_router_head is not None:
            self.patch_evidence_router_head.zero_init_residual()
        if self.local_zoom_image_expert is not None:
            self.local_zoom_image_expert.zero_init_residual()
        if self.high_frequency_texture_expert is not None:
            self.high_frequency_texture_expert.zero_init_residual()
        for block in self.blocks:
            local_patch_mixer = getattr(block, "local_patch_mixer", None)
            if local_patch_mixer is not None:
                local_patch_mixer.zero_init_residual()
            relative_position_attention = getattr(
                getattr(block, "attn", None),
                "relative_position_attention",
                None,
            )
            if relative_position_attention is not None:
                relative_position_attention.reset_locality_parameters()
        if self.late_class_attention_pool is not None:
            self.late_class_attention_pool.zero_init_residual()
        if self.pairwise_margin_head is not None:
            nn.init.zeros_(self.pairwise_margin_head.weight)
            if self.pairwise_margin_head.bias is not None:
                nn.init.zeros_(self.pairwise_margin_head.bias)
        if self.topk_reassessment_head is not None:
            self.topk_reassessment_head.zero_init_residual()
        if self.focus_class_head is not None:
            nn.init.zeros_(self.focus_class_head.weight)
            if self.focus_class_head.bias is not None:
                nn.init.zeros_(self.focus_class_head.bias)
        if self.class_independent_head is not None:
            nn.init.zeros_(self.class_independent_head.weight)
            if self.class_independent_head.bias is not None:
                nn.init.zeros_(self.class_independent_head.bias)
        if self.ordinal_maturity_head is not None:
            nn.init.zeros_(self.ordinal_maturity_head.weight)
            if self.ordinal_maturity_head.bias is not None:
                nn.init.zeros_(self.ordinal_maturity_head.bias)
        if self.cumulative_ordinal_head is not None:
            nn.init.zeros_(self.cumulative_ordinal_head.weight)
            if self.cumulative_ordinal_head.bias is not None:
                nn.init.zeros_(self.cumulative_ordinal_head.bias)
        self.cropr_token_selectors = nn.ModuleDict(
            {
                str(int(layer_index + 1)): CroprTokenSelector(
                    dim=int(embed_dim),
                    num_classes=int(num_classes),
                    mlp_ratio=4.0,
                )
                for layer_index in self.token_prune_schedule
            }
            if self.cropr_token_selector_enabled
            else {}
        )
        self.deep_class_prompt = (
            DeepClassPrompt(
                depth=int(depth),
                num_classes=int(num_classes),
                embed_dim=int(embed_dim),
                init_seed=self.deep_class_prompt_init_seed,
            )
            if self.deep_class_prompt_enabled
            else None
        )
        self._initialize_late_member_modules()

    def _init_parameter_tensors(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _initialize_late_member_modules(self) -> None:
        self.late_member_blocks = nn.ModuleList()
        self.late_member_norm = None
        self.late_member_head = None
        self.late_member_fine_grained_pool = None
        self.late_member_pairwise_margin_norm = None
        self.late_member_pairwise_margin_dropout = None
        self.late_member_pairwise_margin_head = None
        self.late_member_cnn_fusion_norm = None
        self.late_member_cnn_fusion_dropout = None
        self.late_member_cnn_fusion_head = None
        self.late_member_bbox_spatial_fusion_head = None
        if not self.late_member_enabled:
            return

        prune_layers = tuple(int(index) + 1 for index in self.token_prune_schedule)
        if prune_layers and max(prune_layers) > self.late_member_fork_after_block:
            raise ValueError(
                "Late-member branching requires token pruning to finish at or before "
                f"the fork; prune_layers={prune_layers}, "
                f"fork_after_block={self.late_member_fork_after_block}."
            )
        unsupported_modules = {
            "frequency_selective_pool": self.frequency_selective_pool,
            "patch_memory_adapter": self.patch_memory_adapter,
            "late_class_attention_pool": self.late_class_attention_pool,
            "color_fusion_head": self.color_fusion_head,
            "defect_fusion_head": self.defect_fusion_head,
            "foreground_surface_fusion_head": self.foreground_surface_fusion_head,
            "foreground_surface_pairwise_head": self.foreground_surface_pairwise_head,
            "interior_boundary_pairwise_head": self.interior_boundary_pairwise_head,
            "bilinear_patch_fusion_head": self.bilinear_patch_fusion_head,
            "complementary_patch_suppression_head": self.complementary_patch_suppression_head,
            "patch_objectness_guided_head": self.patch_objectness_guided_head,
            "bbox_prior_patch_context_head": self.bbox_prior_patch_context_head,
            "micro_detail_patch_expert": self.micro_detail_patch_expert,
            "part_token_learner": self.part_token_learner,
            "part_token_pairwise_learner": self.part_token_pairwise_learner,
            "patch_evidence_router_head": self.patch_evidence_router_head,
            "local_zoom_image_expert": self.local_zoom_image_expert,
            "high_frequency_texture_expert": self.high_frequency_texture_expert,
            "focus_class_head": self.focus_class_head,
            "topk_reassessment_head": self.topk_reassessment_head,
            "ordinal_maturity_head": self.ordinal_maturity_head,
            "cumulative_ordinal_head": self.cumulative_ordinal_head,
            "deep_class_prompt": self.deep_class_prompt,
        }
        unsupported_enabled = sorted(
            name for name, module in unsupported_modules.items() if module is not None
        )
        if self.layer_token_fusion_enabled:
            unsupported_enabled.append("layer_token_fusion")
        if self.soft_moe_patch_adapter_enabled:
            unsupported_enabled.append("soft_moe_patch_adapter")
        if unsupported_enabled:
            raise ValueError(
                "Late-member branch does not yet duplicate these active readouts: "
                + ", ".join(unsupported_enabled)
            )

        self.late_member_blocks = copy.deepcopy(
            self.blocks[self.late_member_fork_after_block :]
        )
        self.late_member_norm = copy.deepcopy(self.norm)
        self.late_member_head = copy.deepcopy(self.head)
        self.late_member_fine_grained_pool = copy.deepcopy(self.fine_grained_pool)
        self.late_member_pairwise_margin_norm = copy.deepcopy(self.pairwise_margin_norm)
        self.late_member_pairwise_margin_dropout = copy.deepcopy(
            self.pairwise_margin_dropout
        )
        self.late_member_pairwise_margin_head = copy.deepcopy(self.pairwise_margin_head)
        if self.cnn_fusion_head is not None:
            self.late_member_cnn_fusion_norm = copy.deepcopy(self.cnn_fusion_norm)
            self.late_member_cnn_fusion_dropout = copy.deepcopy(self.cnn_fusion_dropout)
            self.late_member_cnn_fusion_head = copy.deepcopy(self.cnn_fusion_head)
        self.late_member_bbox_spatial_fusion_head = copy.deepcopy(
            self.bbox_spatial_fusion_head
        )

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
            "positive_embedding",
            "negative_embedding",
            "lambda_q",
            "lambda_k",
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
        if tuple(int(value) for value in grid_size) != tuple(
            int(value) for value in self.patch_embed.base_grid_size
        ):
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
        branch_key = "branch_token_fusion.branch_type_embed"
        if (
            branch_key in adapted_state
            and self.branch_token_fusion is not None
            and adapted_state[branch_key].shape
            != self.branch_token_fusion.branch_type_embed.shape
        ):
            source = adapted_state[branch_key]
            target = self.branch_token_fusion.branch_type_embed.detach().clone()
            if (
                source.ndim == target.ndim == 3
                and int(source.shape[0]) == int(target.shape[0])
                and int(source.shape[2]) == int(target.shape[2])
                and int(source.shape[1]) <= int(target.shape[1])
            ):
                target[:, : int(source.shape[1]), :] = source.to(
                    device=target.device,
                    dtype=target.dtype,
                )
                adapted_state[branch_key] = target
        if self.late_member_enabled:
            current_state = self.state_dict()
            late_prefix_sources = {
                "late_member_norm.": "norm.",
                "late_member_head.": "head.",
                "late_member_fine_grained_pool.": "fine_grained_pool.",
                "late_member_pairwise_margin_norm.": "pairwise_margin_norm.",
                "late_member_pairwise_margin_head.": "pairwise_margin_head.",
                "late_member_cnn_fusion_norm.": "cnn_fusion_norm.",
                "late_member_cnn_fusion_head.": "cnn_fusion_head.",
                "late_member_bbox_spatial_fusion_head.": "bbox_spatial_fusion_head.",
            }
            for target_key, target_tensor in current_state.items():
                if target_key in adapted_state:
                    continue
                source_key = None
                if target_key.startswith("late_member_blocks."):
                    suffix = target_key[len("late_member_blocks.") :]
                    block_text, separator, remainder = suffix.partition(".")
                    if separator and block_text.isdigit():
                        source_block = (
                            self.late_member_fork_after_block + int(block_text)
                        )
                        source_key = f"blocks.{source_block}.{remainder}"
                else:
                    for target_prefix, source_prefix in late_prefix_sources.items():
                        if target_key.startswith(target_prefix):
                            source_key = source_prefix + target_key[len(target_prefix) :]
                            break
                if source_key is None or source_key not in adapted_state:
                    continue
                source_tensor = adapted_state[source_key]
                if not torch.is_tensor(source_tensor):
                    raise ValueError(f"Late-member source is not a tensor: {source_key}")
                if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                    raise ValueError(
                        "Late-member initialization shape mismatch: "
                        f"target={target_key}{tuple(target_tensor.shape)}, "
                        f"source={source_key}{tuple(source_tensor.shape)}"
                    )
                adapted_state[target_key] = source_tensor.detach().clone()
        missing_keys, unexpected_keys = self.load_state_dict(adapted_state, strict=False)
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Loi load state_dict. missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return missing_keys, unexpected_keys

    def load_patch_evidence_linear_verifier_export(
        self,
        path: Path,
        *,
        pair: str = "0-1",
        min_pair_probability: float = 0.02,
        max_pair_margin: float = 0.40,
        confidence_threshold: float = 0.60,
        logit_boost: float = 0.01,
        protect_right_min_probability: float = 0.0,
        training_soft_adjustment: bool = False,
        training_soft_logit_scale: float = 0.05,
        training_soft_gate_temperature: float = 0.05,
    ) -> Dict[str, object]:
        verifier = PatchEvidenceLinearVerifier.from_export_path(
            Path(path),
            pair=pair,
            min_pair_probability=min_pair_probability,
            max_pair_margin=max_pair_margin,
            confidence_threshold=confidence_threshold,
            logit_boost=logit_boost,
            protect_right_min_probability=protect_right_min_probability,
            training_soft_adjustment=training_soft_adjustment,
            training_soft_logit_scale=training_soft_logit_scale,
            training_soft_gate_temperature=training_soft_gate_temperature,
        )
        self.patch_evidence_linear_verifier = verifier
        return {
            "enabled": True,
            "path": str(Path(path)),
            "pair": [int(verifier.pair[0]), int(verifier.pair[1])],
            "feature_dim": int(verifier.feature_dim),
            "top_k": int(verifier.top_k),
            "min_pair_probability": float(verifier.min_pair_probability),
            "max_pair_margin": float(verifier.max_pair_margin),
            "confidence_threshold": float(verifier.confidence_threshold),
            "logit_boost": float(verifier.logit_boost),
            "protect_right_min_probability": float(verifier.protect_right_min_probability),
            "training_soft_adjustment": bool(verifier.training_soft_adjustment),
            "training_soft_logit_scale": float(verifier.training_soft_logit_scale),
            "training_soft_gate_temperature": float(
                verifier.training_soft_gate_temperature
            ),
        }

    @staticmethod
    def _normalize_token_scores(scores: Tensor) -> Tensor:
        minimum = scores.amin(dim=1, keepdim=True)
        maximum = scores.amax(dim=1, keepdim=True)
        return (scores - minimum) / (maximum - minimum).clamp(min=1e-6)

    def _select_layer_token_fusion_feature(
        self,
        tokens: Tensor,
        attention: Tensor,
        patch_indices: Tensor,
        foreground_prior: Optional[Tensor],
        bbox_patch_prior: Optional[Tensor],
        prefix_count: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        resolved_prefix_count = int(
            self.num_prefix_tokens if prefix_count is None else prefix_count
        )
        patch_tokens = tokens[:, resolved_prefix_count:]
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            batch_size = int(tokens.size(0))
            empty_feature = tokens.new_zeros((batch_size, self.embed_dim))
            empty_scores = tokens.new_zeros((batch_size, 0), dtype=torch.float32)
            empty_indices = tokens.new_zeros((batch_size, 0), dtype=torch.long)
            return empty_feature, empty_scores, empty_indices
        if attention.ndim != 4:
            raise ValueError("layer token fusion expects attention [B,H,N,N].")
        scores = attention[:, :, 0, resolved_prefix_count:].detach().float().mean(dim=1)
        if tuple(scores.shape) != tuple(patch_tokens.shape[:2]):
            scores = patch_tokens.detach().float().norm(dim=-1)
        if (
            self.layer_token_fusion_foreground_weight > 0.0
            and torch.is_tensor(foreground_prior)
            and foreground_prior.ndim == 2
            and patch_indices.ndim == 2
            and foreground_prior.size(0) == patch_indices.size(0)
            and foreground_prior.size(1) > int(patch_indices.max().detach().item())
        ):
            gathered_prior = foreground_prior.to(device=tokens.device, dtype=torch.float32).gather(
                1,
                patch_indices.to(device=tokens.device, dtype=torch.long),
            )
            scores = scores + self.layer_token_fusion_foreground_weight * self._normalize_token_scores(
                gathered_prior
            )
        if (
            self.layer_token_fusion_bbox_weight > 0.0
            and torch.is_tensor(bbox_patch_prior)
            and bbox_patch_prior.ndim == 2
            and patch_indices.ndim == 2
            and bbox_patch_prior.size(0) == patch_indices.size(0)
            and bbox_patch_prior.size(1) > int(patch_indices.max().detach().item())
        ):
            gathered_bbox = bbox_patch_prior.to(device=tokens.device, dtype=torch.float32).gather(
                1,
                patch_indices.to(device=tokens.device, dtype=torch.long),
            )
            scores = scores + self.layer_token_fusion_bbox_weight * self._normalize_token_scores(
                gathered_bbox
            )
        selected_count = min(self.layer_token_fusion_top_k, int(patch_tokens.size(1)))
        selected_scores, selected_indices = torch.topk(
            scores,
            k=selected_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        normalized_tokens = self.norm(patch_tokens)
        selected_tokens = normalized_tokens.gather(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, normalized_tokens.size(-1)),
        )
        weights = F.softmax(
            selected_scores / self.layer_token_fusion_attention_temperature,
            dim=1,
        ).to(dtype=selected_tokens.dtype)
        selected_feature = torch.einsum("bk,bkd->bd", weights, selected_tokens)
        selected_original_indices = patch_indices.gather(
            1,
            selected_indices.to(device=patch_indices.device, dtype=torch.long),
        )
        return selected_feature, selected_scores.detach(), selected_original_indices.detach()

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
        detail_prior = _exportable_adaptive_avg_pool2d(
            detail,
            output_size=grid_size,
        ).flatten(1)

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
        color_prior = _exportable_adaptive_avg_pool2d(
            color_prior_map.unsqueeze(1),
            output_size=grid_size,
        ).flatten(1)
        non_padding_prior = _exportable_adaptive_avg_pool2d(
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
            valid_fraction = _exportable_adaptive_avg_pool2d(
                valid_mask.to(device=image.device, dtype=torch.float32),
                output_size=grid_size,
            ).flatten(1)
            prior = prior * valid_fraction
        return self._normalize_token_scores(prior)

    def _build_patch_bbox_prior(
        self,
        bbox_token_prior: Optional[Tensor],
        grid_size: Tuple[int, int],
    ) -> Optional[Tensor]:
        if bbox_token_prior is None or not torch.is_tensor(bbox_token_prior):
            return None
        if bbox_token_prior.ndim == 3:
            bbox_token_prior = bbox_token_prior[:, 0]
        if bbox_token_prior.ndim != 2 or int(bbox_token_prior.size(1)) < 4:
            return None
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        bbox = bbox_token_prior[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
        cx, cy, bw, bh = bbox.unbind(dim=1)
        margin = float(max(0.0, self.token_prune_bbox_margin_ratio))
        x1 = (cx - 0.5 * bw - bw * margin).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * bh - bh * margin).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * bw + bw * margin).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * bh + bh * margin).clamp(0.0, 1.0)

        x_edges = torch.linspace(0.0, 1.0, grid_width + 1, device=bbox.device, dtype=torch.float32)
        y_edges = torch.linspace(0.0, 1.0, grid_height + 1, device=bbox.device, dtype=torch.float32)
        patch_x1 = x_edges[:-1].view(1, 1, grid_width)
        patch_x2 = x_edges[1:].view(1, 1, grid_width)
        patch_y1 = y_edges[:-1].view(1, grid_height, 1)
        patch_y2 = y_edges[1:].view(1, grid_height, 1)

        inter_w = (
            torch.minimum(patch_x2, x2.view(-1, 1, 1))
            - torch.maximum(patch_x1, x1.view(-1, 1, 1))
        ).clamp(min=0.0)
        inter_h = (
            torch.minimum(patch_y2, y2.view(-1, 1, 1))
            - torch.maximum(patch_y1, y1.view(-1, 1, 1))
        ).clamp(min=0.0)
        patch_area = (1.0 / float(grid_width)) * (1.0 / float(grid_height))
        overlap = (inter_w * inter_h / max(patch_area, 1e-6)).flatten(1)
        return self._normalize_token_scores(overlap)

    @staticmethod
    def _fuse_inattentive_context(
        *,
        patch_tokens: Tensor,
        cls_attention: Tensor,
        selected_local: Tensor,
        previous_context: Optional[Tensor] = None,
        previous_context_attention: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply the parameter-free EViT weighted sum to rejected patches.

        The first-stage equation follows the official Apache-2.0 EViT source
        (`youweiliang/evit`, commit 97e58f6). A later stage also refolds the
        existing non-spatial context with its current CLS attention.
        """

        if patch_tokens.ndim != 3 or cls_attention.ndim != 2:
            raise ValueError("Inattentive fusion expects [B,N,C] tokens and [B,N] attention.")
        if tuple(patch_tokens.shape[:2]) != tuple(cls_attention.shape):
            raise ValueError("Patch-token and CLS-attention shapes must agree.")
        if selected_local.ndim != 2 or selected_local.size(0) != patch_tokens.size(0):
            raise ValueError("Selected patch indices must have shape [B,K].")

        patch_count = int(patch_tokens.size(1))
        dropped_count = patch_count - int(selected_local.size(1))
        if dropped_count <= 0:
            raise ValueError("Inattentive fusion requires at least one rejected patch.")

        selected_mask = cls_attention.new_zeros(cls_attention.shape).scatter(
            1,
            selected_local,
            torch.ones_like(selected_local, dtype=cls_attention.dtype),
        )
        dropped_mask = (1.0 - selected_mask).clamp(min=0.0, max=1.0)
        dropped_weight_map = cls_attention * dropped_mask
        context = torch.sum(
            patch_tokens * dropped_weight_map.to(dtype=patch_tokens.dtype).unsqueeze(-1),
            dim=1,
            keepdim=True,
        )

        previous_mass = cls_attention.new_zeros((patch_tokens.size(0), 1))
        if previous_context is not None:
            if previous_context_attention is None:
                raise ValueError("Previous context requires its current CLS attention.")
            if tuple(previous_context.shape) != (
                int(patch_tokens.size(0)),
                1,
                int(patch_tokens.size(2)),
            ):
                raise ValueError("Previous context must have shape [B,1,C].")
            if tuple(previous_context_attention.shape) != (
                int(patch_tokens.size(0)),
                1,
            ):
                raise ValueError("Previous context attention must have shape [B,1].")
            previous_mass = previous_context_attention
            context = context + previous_context * previous_mass.to(
                dtype=previous_context.dtype
            ).unsqueeze(-1)
        elif previous_context_attention is not None:
            raise ValueError("Previous context attention was provided without a context token.")

        dropped_local = torch.topk(
            dropped_mask,
            k=dropped_count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        dropped_weights = cls_attention.gather(1, dropped_local)
        total_mass = dropped_weights.sum(dim=1, keepdim=True) + previous_mass
        return context, dropped_local, dropped_weights, total_mass, previous_mass

    def _prune_patch_tokens(
        self,
        *,
        tokens: Tensor,
        attention: Tensor,
        patch_indices: Tensor,
        foreground_prior: Tensor,
        original_patch_count: int,
        keep_rate: float,
        prefix_count: Optional[int] = None,
        cropr_scores: Optional[Tensor] = None,
        cropr_trace: Optional[Dict[str, Tensor]] = None,
        cropr_routing: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        resolved_prefix_count = int(
            self.num_prefix_tokens if prefix_count is None else prefix_count
        )
        context_count = resolved_prefix_count - int(self.num_prefix_tokens)
        if context_count < 0:
            raise ValueError("Resolved prefix count cannot be smaller than base prefixes.")
        if self.inattentive_token_fusion_enabled and context_count not in {0, 1}:
            raise ValueError("Inattentive fusion supports zero or one active context token.")
        patch_tokens = tokens[:, resolved_prefix_count:]
        current_patch_count = int(patch_tokens.size(1))
        target_patch_count = max(1, int(math.ceil(float(original_patch_count) * float(keep_rate))))
        target_patch_count = min(current_patch_count, target_patch_count)
        if target_patch_count >= current_patch_count and cropr_scores is None:
            empty = {
                "kept_indices": patch_indices,
                "scores": foreground_prior.gather(1, patch_indices),
            }
            return tokens, patch_indices, empty

        query_count = max(1, int(self.num_prefix_tokens))
        attention_score = attention[
            :, :, :query_count, resolved_prefix_count:
        ].mean(dim=(1, 2))
        gathered_prior = foreground_prior.gather(1, patch_indices)
        native_score = self._normalize_token_scores(attention_score)
        if cropr_scores is not None and tuple(cropr_scores.shape) != tuple(native_score.shape):
            raise ValueError("Cropr scores must match the active spatial patch layout.")
        if cropr_routing and cropr_scores is None:
            raise ValueError("Cropr routing requires learned selector scores.")
        learned_score = (
            self._normalize_token_scores(cropr_scores)
            if cropr_scores is not None
            else None
        )
        score = learned_score if cropr_routing and learned_score is not None else native_score
        if self.token_prune_foreground_weight > 0.0:
            score = score + self.token_prune_foreground_weight * self._normalize_token_scores(gathered_prior)

        if target_patch_count >= current_patch_count:
            unpruned = {
                "kept_indices": patch_indices,
                "scores": score,
            }
            if learned_score is not None:
                unpruned["native_scores"] = native_score
                unpruned["cropr_scores"] = learned_score
                unpruned["cropr_routing"] = torch.tensor(
                    bool(cropr_routing), device=score.device, dtype=torch.bool
                )
            if cropr_trace:
                unpruned.update(cropr_trace)
            return tokens, patch_indices, unpruned

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
        trace = {
            "kept_indices": selected_original,
            "scores": selected_scores,
        }
        if learned_score is not None:
            trace["native_scores"] = native_score
            trace["cropr_scores"] = learned_score
            trace["cropr_routing"] = torch.tensor(
                bool(cropr_routing), device=score.device, dtype=torch.bool
            )
        if cropr_trace:
            trace.update(cropr_trace)
            active_local = torch.arange(
                current_patch_count,
                device=score.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(int(score.size(0)), -1)
            kept_mask = torch.zeros_like(score, dtype=torch.bool).scatter(
                1, selected_local, True
            )
            dropped_local_for_trace = active_local.masked_select(
                ~kept_mask
            ).reshape(int(score.size(0)), current_patch_count - target_patch_count)
            trace["dropped_indices"] = patch_indices.gather(
                1, dropped_local_for_trace
            )
        if self.inattentive_token_fusion_enabled:
            cls_patch_attention = attention[
                :, :, 0, resolved_prefix_count:
            ].mean(dim=1)
            previous_context = (
                tokens[:, self.num_prefix_tokens : resolved_prefix_count]
                if context_count == 1
                else None
            )
            previous_context_attention = (
                attention[
                    :, :, 0, self.num_prefix_tokens : resolved_prefix_count
                ].mean(dim=1)
                if context_count == 1
                else None
            )
            (
                context_token,
                dropped_local,
                dropped_weights,
                context_attention_mass,
                previous_context_mass,
            ) = self._fuse_inattentive_context(
                patch_tokens=patch_tokens,
                cls_attention=cls_patch_attention,
                selected_local=selected_local,
                previous_context=previous_context,
                previous_context_attention=previous_context_attention,
            )
            dropped_original = patch_indices.gather(1, dropped_local)
            pruned_tokens = torch.cat(
                (
                    tokens[:, : self.num_prefix_tokens],
                    context_token,
                    selected_tokens,
                ),
                dim=1,
            )
            trace.update(
                {
                    "dropped_indices": dropped_original,
                    "fusion_weights": dropped_weights,
                    "context_attention_mass": context_attention_mass,
                    "previous_context_attention_mass": previous_context_mass,
                    "context_token": context_token,
                }
            )
        else:
            pruned_tokens = torch.cat(
                (tokens[:, :resolved_prefix_count], selected_tokens), dim=1
            )
        return pruned_tokens, selected_original, trace

    def _early_mask_patch_tokens(
        self,
        *,
        tokens: Tensor,
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
        gathered_prior = foreground_prior.gather(1, patch_indices)
        if target_patch_count >= current_patch_count:
            empty = {
                "kept_indices": patch_indices,
                "scores": self._normalize_token_scores(gathered_prior),
            }
            return tokens, patch_indices, empty

        score = self._normalize_token_scores(gathered_prior)
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
        masked_tokens = torch.cat((tokens[:, :prefix_count], selected_tokens), dim=1)
        trace = {
            "kept_indices": selected_original,
            "scores": selected_scores,
        }
        return masked_tokens, selected_original, trace

    def forward_features(
        self,
        x: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox_token_prior: Optional[Tensor] = None,
        return_attention: bool = False,
        attention_layers: Optional[Sequence[int]] = None,
        return_trace: bool = False,
    ) -> Dict[str, Tensor]:
        input_spatial_size = tuple(int(value) for value in x.shape[-2:])
        input_image = x
        x = self.stem(x)
        x = self.mixstyle(x)
        stem_features = x
        batch_size = x.shape[0]
        # Deployment fixes spatial dimensions at export time and keeps only the
        # batch axis dynamic. Materializing these values also gives the legacy
        # ONNX exporter constant adaptive-pooling output sizes.
        grid_size = (
            int(x.shape[-2] // self.patch_embed.patch_size),
            int(x.shape[-1] // self.patch_embed.patch_size),
        )
        patch_tokens = self.patch_embed(x)
        shifted_patch_residual = None
        if self.shifted_patch_token_residual is not None:
            shifted_patch_residual = self.shifted_patch_token_residual(x)
            patch_tokens = patch_tokens + shifted_patch_residual.to(dtype=patch_tokens.dtype)
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
        gabor_texture_trace = None
        gabor_texture_semantic_feature = None
        gabor_texture_semantic_trace = None
        if self.gabor_texture_residual is not None:
            if return_trace:
                gabor_texture_value, gabor_texture_trace = self.gabor_texture_residual(
                    input_image,
                    bbox_token_prior,
                    return_trace=True,
                )
            else:
                gabor_texture_value = self.gabor_texture_residual(
                    input_image,
                    bbox_token_prior,
                )
            edge_index = int(self.gabor_texture_edge_token_index)
            if not 0 <= edge_index < int(branch_tokens.size(1)):
                raise RuntimeError("Gabor texture edge-token index is outside branch tokens.")
            gabor_texture_value = gabor_texture_value.to(dtype=branch_tokens.dtype)
            branch_tokens = torch.cat(
                (
                    branch_tokens[:, :edge_index],
                    branch_tokens[:, edge_index : edge_index + 1]
                    + gabor_texture_value.unsqueeze(1),
                    branch_tokens[:, edge_index + 1 :],
                ),
                dim=1,
            )
        if self.gabor_texture_semantic_encoder is not None:
            if return_trace:
                (
                    gabor_texture_semantic_feature,
                    gabor_texture_semantic_trace,
                ) = self.gabor_texture_semantic_encoder(
                    input_image,
                    bbox_token_prior,
                    return_trace=True,
                )
            else:
                gabor_texture_semantic_feature = (
                    self.gabor_texture_semantic_encoder(
                        input_image,
                        bbox_token_prior,
                    )
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
        concurrent_local_state = (
            self.concurrent_local_initializer(stem_features, grid_size)
            if self.concurrent_local_initializer is not None
            else None
        )
        pruning_enabled = bool(self.token_pruning and not return_attention)
        foreground_prior = (
            self._build_patch_foreground_prior(
                input_image,
                grid_size=grid_size,
                image_valid_mask=image_valid_mask,
            )
            if (
                pruning_enabled
                or return_trace
                or self.frequency_selective_pool is not None
                or self.micro_detail_patch_expert is not None
                or self.part_token_learner is not None
                or self.part_token_pairwise_learner is not None
                or self.layer_token_fusion_enabled
            )
            else patch_tokens.new_zeros((batch_size, original_patch_count), dtype=torch.float32)
        )
        bbox_patch_prior = self._build_patch_bbox_prior(
            bbox_token_prior,
            grid_size=grid_size,
        )
        if bbox_patch_prior is not None and self.token_prune_bbox_weight > 0.0:
            foreground_prior = self._normalize_token_scores(
                foreground_prior
                + float(self.token_prune_bbox_weight)
                * bbox_patch_prior.to(device=foreground_prior.device, dtype=foreground_prior.dtype)
            )
        collect_all_attentions = return_attention and attention_layers is None
        attention_layer_set = set(attention_layers or [])
        layer_token_fusion_layer_set = {
            int(layer_number) - 1 for layer_number in self.layer_token_fusion_layer_numbers
        }
        attention_maps = {}
        pruning_trace: List[Dict[str, Tensor]] = []
        cropr_auxiliary_logits: Dict[str, Tensor] = {}
        active_prefix_count = int(self.num_prefix_tokens)
        if pruning_enabled and self.early_token_mask_keep_rate < 1.0:
            before_count = int(tokens.size(1) - self.num_prefix_tokens)
            tokens, patch_indices, prune_info = self._early_mask_patch_tokens(
                tokens=tokens,
                patch_indices=patch_indices,
                foreground_prior=foreground_prior,
                original_patch_count=original_patch_count,
                keep_rate=self.early_token_mask_keep_rate,
            )
            prune_info["layer"] = torch.tensor(
                0,
                device=tokens.device,
                dtype=torch.long,
            )
            prune_info["before_count"] = torch.tensor(
                before_count,
                device=tokens.device,
                dtype=torch.long,
            )
            prune_info["after_count"] = torch.tensor(
                int(patch_indices.size(1)),
                device=tokens.device,
                dtype=torch.long,
            )
            pruning_trace.append(prune_info)
        multi_granularity_logits: Dict[str, Tensor] = {}
        multi_granularity_features: Dict[str, Tensor] = {}
        block_token_shapes: List[Tuple[int, ...]] = []
        block_patch_indices: List[Tensor] = []
        block_patch_norms: List[Tensor] = []
        layer_token_fusion_features: List[Tensor] = []
        layer_token_fusion_layers: List[int] = []
        layer_token_fusion_scores: List[Tensor] = []
        layer_token_fusion_indices: List[Tensor] = []
        concurrent_local_traces: List[Dict[str, Tensor]] = []
        deep_class_prompt_tokens: List[Tensor] = []
        deep_class_prompt_attention_maps: Dict[int, Tensor] = {}
        collect_deep_class_prompt_attention = bool(
            self.deep_class_prompt is not None and (return_attention or return_trace)
        )
        late_member_seed_tokens: Optional[Tensor] = None
        late_member_patch_indices: Optional[Tensor] = None
        late_member_attention_maps: Dict[int, Tensor] = {}
        for block_index, block in enumerate(self.blocks):
            layer_key = str(int(block_index + 1))
            should_prune = pruning_enabled and block_index in self.token_prune_schedule
            should_collect_layer_tokens = block_index in layer_token_fusion_layer_set
            block_tokens = tokens
            block_prefix_count = int(active_prefix_count)
            if self.deep_class_prompt is not None:
                block_tokens = self.deep_class_prompt.insert(
                    block_tokens,
                    block_index=block_index,
                    base_prefix_count=self.num_prefix_tokens,
                )
                block_prefix_count += int(self.deep_class_prompt.num_classes)
            if (
                collect_all_attentions
                or block_index in attention_layer_set
                or should_prune
                or should_collect_layer_tokens
                or collect_deep_class_prompt_attention
            ):
                block_tokens, attention = block(
                    block_tokens,
                    return_attention=True,
                    grid_size=grid_size,
                    prefix_count=block_prefix_count,
                    patch_indices=patch_indices,
                    collect_attention_trace=return_trace,
                )
                if collect_deep_class_prompt_attention:
                    deep_class_prompt_attention_maps[block_index] = (
                        self.deep_class_prompt.class_to_patch_attention(
                            attention,
                            base_prefix_count=self.num_prefix_tokens,
                            patch_indices=patch_indices,
                            original_patch_count=original_patch_count,
                        )
                    )
                if collect_all_attentions or block_index in attention_layer_set:
                    attention_maps[block_index] = (
                        self.deep_class_prompt.sanitize_native_attention(
                            attention,
                            base_prefix_count=self.num_prefix_tokens,
                        )
                        if self.deep_class_prompt is not None
                        else attention
                    )
                if should_collect_layer_tokens:
                    (
                        layer_feature,
                        layer_scores,
                        layer_indices,
                    ) = self._select_layer_token_fusion_feature(
                        tokens=block_tokens,
                        attention=attention,
                        patch_indices=patch_indices,
                        foreground_prior=foreground_prior,
                        bbox_patch_prior=bbox_patch_prior,
                        prefix_count=block_prefix_count,
                    )
                    layer_token_fusion_features.append(layer_feature)
                    layer_token_fusion_layers.append(int(block_index + 1))
                    layer_token_fusion_scores.append(layer_scores)
                    layer_token_fusion_indices.append(layer_indices)
                if should_prune:
                    before_count = int(block_tokens.size(1) - block_prefix_count)
                    cropr_scores = None
                    cropr_trace: Dict[str, Tensor] = {}
                    cropr_selector = (
                        self.cropr_token_selectors[layer_key]
                        if layer_key in self.cropr_token_selectors
                        else None
                    )
                    if cropr_selector is not None:
                        (
                            cropr_scores,
                            cropr_logits,
                            cropr_trace,
                        ) = cropr_selector(
                            block_tokens[:, block_prefix_count:],
                            collect_auxiliary=bool(self.training or return_trace),
                            return_trace=return_trace,
                        )
                        if cropr_logits is not None:
                            cropr_auxiliary_logits[layer_key] = cropr_logits
                    block_tokens, patch_indices, prune_info = self._prune_patch_tokens(
                        tokens=block_tokens,
                        attention=attention,
                        patch_indices=patch_indices,
                        foreground_prior=foreground_prior,
                        original_patch_count=original_patch_count,
                        keep_rate=self.token_prune_schedule[block_index],
                        prefix_count=block_prefix_count,
                        cropr_scores=cropr_scores,
                        cropr_trace=cropr_trace,
                        cropr_routing=bool(
                            cropr_selector is not None
                            and self.cropr_token_selector_routing
                        ),
                    )
                    prune_info["layer"] = torch.tensor(
                        int(block_index + 1),
                        device=block_tokens.device,
                        dtype=torch.long,
                    )
                    prune_info["before_count"] = torch.tensor(
                        before_count,
                        device=block_tokens.device,
                        dtype=torch.long,
                    )
                    prune_info["after_count"] = torch.tensor(
                        int(patch_indices.size(1)),
                        device=block_tokens.device,
                        dtype=torch.long,
                    )
                    pruning_trace.append(prune_info)
                    active_prefix_count = int(
                        block_tokens.size(1) - patch_indices.size(1)
                    )
            else:
                if self.gradient_checkpointing and self.training:
                    block_tokens = gradient_checkpoint(
                        lambda current_tokens: block(
                            current_tokens,
                            grid_size=grid_size,
                            prefix_count=block_prefix_count,
                            patch_indices=patch_indices,
                        ),
                        block_tokens,
                        use_reentrant=False,
                    )
                else:
                    block_tokens = block(
                        block_tokens,
                        grid_size=grid_size,
                        prefix_count=block_prefix_count,
                        patch_indices=patch_indices,
                    )
            if self.deep_class_prompt is not None:
                tokens, prompt_tokens = self.deep_class_prompt.extract_and_remove(
                    block_tokens,
                    base_prefix_count=self.num_prefix_tokens,
                )
                deep_class_prompt_tokens.append(prompt_tokens)
            else:
                tokens = block_tokens
            active_prefix_count = int(tokens.size(1) - patch_indices.size(1))
            coupling = (
                self.concurrent_local_couplings[layer_key]
                if layer_key in self.concurrent_local_couplings
                else None
            )
            if coupling is not None and concurrent_local_state is not None:
                if return_trace:
                    tokens, concurrent_local_state, coupling_trace = coupling(
                        tokens,
                        concurrent_local_state,
                        grid_size=grid_size,
                        prefix_count=active_prefix_count,
                        patch_indices=patch_indices,
                        return_trace=True,
                    )
                    concurrent_local_traces.append(coupling_trace)
                else:
                    tokens, concurrent_local_state = coupling(
                        tokens,
                        concurrent_local_state,
                        grid_size=grid_size,
                        prefix_count=active_prefix_count,
                        patch_indices=patch_indices,
                    )
            if (
                self.late_member_enabled
                and int(block_index + 1) == self.late_member_fork_after_block
            ):
                late_member_seed_tokens = tokens
                late_member_patch_indices = patch_indices
            if return_trace:
                block_token_shapes.append(tuple(int(value) for value in tokens.shape))
                block_patch_indices.append(patch_indices.detach().clone())
                block_patch_norms.append(
                    tokens[:, active_prefix_count:].detach().float().norm(dim=-1)
                )
            if layer_key in self.multi_granularity_aux_heads:
                register_end_for_aux = 1 + self.num_registers
                branch_end_for_aux = register_end_for_aux + self.num_branch_tokens
                pooled_aux = self.pool_tokens_for_head(
                    tokens[:, 0],
                    tokens[:, 1:register_end_for_aux],
                    tokens[:, register_end_for_aux:branch_end_for_aux],
                )
                multi_granularity_features[layer_key] = pooled_aux
                multi_granularity_logits[layer_key] = self.multi_granularity_aux_heads[
                    layer_key
                ](pooled_aux)
        late_member_tokens = None
        if self.late_member_enabled:
            if late_member_seed_tokens is None or late_member_patch_indices is None:
                raise RuntimeError("Late-member fork was not reached during forward_features.")
            late_member_tokens = late_member_seed_tokens
            for late_block_index, late_block in enumerate(self.late_member_blocks):
                if return_attention:
                    late_member_tokens, late_member_attention = late_block(
                        late_member_tokens,
                        return_attention=True,
                        grid_size=grid_size,
                        prefix_count=self.num_prefix_tokens,
                        patch_indices=late_member_patch_indices,
                    )
                    late_member_attention_maps[
                        self.late_member_fork_after_block + int(late_block_index)
                    ] = late_member_attention
                elif self.gradient_checkpointing and self.training:
                    late_member_tokens = gradient_checkpoint(
                        lambda current_tokens, current_block=late_block: current_block(
                            current_tokens,
                            grid_size=grid_size,
                            prefix_count=self.num_prefix_tokens,
                            patch_indices=late_member_patch_indices,
                        ),
                        late_member_tokens,
                        use_reentrant=False,
                    )
                else:
                    late_member_tokens = late_block(
                        late_member_tokens,
                        grid_size=grid_size,
                        prefix_count=self.num_prefix_tokens,
                        patch_indices=late_member_patch_indices,
                    )
            if self.late_member_norm is None:
                raise RuntimeError("Late-member norm is missing.")
            late_member_tokens = self.late_member_norm(late_member_tokens)
        tokens = self.norm(tokens)

        active_prefix_count = int(tokens.size(1) - patch_indices.size(1))
        if active_prefix_count < int(self.num_prefix_tokens):
            raise RuntimeError("Active prefix count is smaller than the base layout.")
        context_out = tokens[:, self.num_prefix_tokens : active_prefix_count]
        public_tokens = (
            torch.cat(
                (
                    tokens[:, : self.num_prefix_tokens],
                    tokens[:, active_prefix_count:],
                ),
                dim=1,
            )
            if active_prefix_count > int(self.num_prefix_tokens)
            else tokens
        )

        deep_class_prompt_logits = None
        if self.deep_class_prompt is not None:
            if len(deep_class_prompt_tokens) != len(self.blocks):
                raise RuntimeError("Deep class prompts did not traverse every block.")
            deep_class_prompt_logits = self.deep_class_prompt.logits(
                deep_class_prompt_tokens[-1]
            )

        cls_out = public_tokens[:, 0]
        register_end = 1 + self.num_registers
        branch_end = register_end + self.num_branch_tokens
        reg_out = public_tokens[:, 1:register_end]
        branch_out = public_tokens[:, register_end:branch_end]
        patch_out = public_tokens[:, branch_end:]
        layer_token_fusion_feature = None
        if layer_token_fusion_features:
            layer_token_fusion_feature = torch.stack(layer_token_fusion_features, dim=1).mean(dim=1)
        features = {
            "cls": cls_out,
            "registers": reg_out,
            "branch_tokens": branch_out,
            "patches": patch_out,
            "tokens": public_tokens,
            "grid_size": grid_size,
            "patch_indices": patch_indices,
            "pooled": self.pool_tokens_for_head(cls_out, reg_out, branch_out),
        }
        if self.inattentive_token_fusion_enabled:
            features["inattentive_context"] = context_out
        if deep_class_prompt_logits is not None:
            features["deep_class_prompt_logits"] = deep_class_prompt_logits
            features["deep_class_prompt_tokens"] = deep_class_prompt_tokens[-1]
        if deep_class_prompt_attention_maps:
            features["deep_class_prompt_attentions"] = (
                deep_class_prompt_attention_maps
            )
        if gabor_texture_semantic_feature is not None:
            features["gabor_texture_semantic_feature"] = (
                gabor_texture_semantic_feature
            )
        if attention_maps:
            attention_representations = {
                int(layer_index): (
                    "vca_effective_positive"
                    if int(layer_index + 1)
                    in self.visual_contrast_attention_layer_numbers
                    else (
                        "faa_local_pool_proxy"
                        if int(layer_index + 1)
                        in self.foveal_aggregated_attention_layer_numbers
                        else (
                            "deformable_bilinear_sample_proxy"
                            if int(layer_index + 1)
                            in self.deformable_spatial_attention_layer_numbers
                            else (
                                "bi_level_routing_sparse_probability"
                                if int(layer_index + 1)
                                in self.bi_level_routing_attention_layer_numbers
                                else "mhsa_probability"
                            )
                        )
                    )
                )
                for layer_index in attention_maps
            }
            features["attention_representations"] = attention_representations
            unique_attention_representations = set(attention_representations.values())
            features["attention_representation"] = (
                next(iter(unique_attention_representations))
                if len(unique_attention_representations) == 1
                else "mixed"
            )
        if late_member_tokens is not None and late_member_patch_indices is not None:
            late_register_end = 1 + self.num_registers
            late_branch_end = late_register_end + self.num_branch_tokens
            late_cls = late_member_tokens[:, 0]
            late_registers = late_member_tokens[:, 1:late_register_end]
            late_branches = late_member_tokens[:, late_register_end:late_branch_end]
            late_patches = late_member_tokens[:, late_branch_end:]
            features["late_member_cls"] = late_cls
            features["late_member_registers"] = late_registers
            features["late_member_branch_tokens"] = late_branches
            features["late_member_patches"] = late_patches
            features["late_member_tokens"] = late_member_tokens
            features["late_member_patch_indices"] = late_member_patch_indices
            features["late_member_pooled"] = self.pool_tokens_for_head(
                late_cls,
                late_registers,
                late_branches,
            )
        if layer_token_fusion_feature is not None:
            features["layer_token_fusion_feature"] = layer_token_fusion_feature
        if concurrent_local_state is not None:
            features["concurrent_local_map"] = concurrent_local_state
        if torch.is_tensor(bbox_token_prior):
            bbox_value = bbox_token_prior
            if bbox_value.ndim == 3:
                bbox_value = bbox_value[:, 0]
            if bbox_value.ndim == 2 and int(bbox_value.size(1)) >= 4:
                features["bbox"] = bbox_value[:, :4].to(
                    device=patch_out.device,
                    dtype=torch.float32,
                )
        if multi_granularity_logits:
            features["multi_granularity_logits"] = multi_granularity_logits
        if multi_granularity_features:
            features["multi_granularity_features"] = multi_granularity_features
        if cropr_auxiliary_logits:
            features["cropr_auxiliary_logits"] = cropr_auxiliary_logits
        if self.training:
            features["stem_features"] = stem_features
        if pruning_enabled or return_trace:
            features["patch_keep_mask"] = F.one_hot(
                patch_indices,
                num_classes=original_patch_count,
            ).sum(dim=1).clamp(max=1).to(dtype=torch.bool)
        if (
            self.frequency_selective_pool is not None
            or self.micro_detail_patch_expert is not None
            or self.part_token_learner is not None
            or self.part_token_pairwise_learner is not None
        ):
            features["patch_foreground_prior"] = foreground_prior.gather(
                1,
                patch_indices,
            )
        if bbox_patch_prior is not None:
            features["patch_bbox_prior"] = bbox_patch_prior.to(
                device=patch_indices.device,
                dtype=torch.float32,
            ).gather(1, patch_indices)
        local_zoom_trace = None
        if self.local_zoom_image_expert is not None:
            if return_trace:
                local_zoom_logits, local_zoom_trace = self.local_zoom_image_expert(
                    input_image,
                    return_trace=True,
                )
                features["local_zoom_logits"] = local_zoom_logits
            else:
                features["local_zoom_logits"] = self.local_zoom_image_expert(input_image)
        high_frequency_texture_trace = None
        if self.high_frequency_texture_expert is not None:
            if return_trace:
                (
                    high_frequency_texture_logits,
                    high_frequency_texture_trace,
                ) = self.high_frequency_texture_expert(
                    input_image,
                    return_trace=True,
                )
                features["high_frequency_texture_logits"] = (
                    high_frequency_texture_logits
                )
            else:
                features["high_frequency_texture_logits"] = (
                    self.high_frequency_texture_expert(input_image)
                )
        if self.cnn_feature_fusion:
            features["cnn_pooled"] = _exportable_adaptive_avg_pool2d(
                stem_features,
                output_size=1,
            ).flatten(1)
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
            if late_member_attention_maps:
                late_path_attention_maps = {
                    int(layer_index): attention
                    for layer_index, attention in attention_maps.items()
                    if int(layer_index) < self.late_member_fork_after_block
                }
                late_path_attention_maps.update(late_member_attention_maps)
                features["primary_attentions"] = attention_maps
                features["late_member_attentions"] = late_path_attention_maps
                # XAI on a late-member model should explain the newly trained path.
                features["attentions"] = late_path_attention_maps
                features["attention_member"] = "late_member"
            else:
                features["attentions"] = attention_maps
                features["attention_member"] = "primary"
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
        foreground_surface_pairwise_trace = None
        if self.foreground_surface_pairwise_head is not None:
            if (
                foreground_surface_trace is not None
                and torch.is_tensor(foreground_surface_trace.get("stats"))
            ):
                features["foreground_surface_pairwise_logits"] = (
                    self.foreground_surface_pairwise_head.forward_from_stats(
                        foreground_surface_trace["stats"],
                        input_image.dtype,
                    )
                )
                foreground_surface_pairwise_trace = foreground_surface_trace
            elif return_trace:
                (
                    foreground_surface_pairwise_logits,
                    foreground_surface_pairwise_trace,
                ) = self.foreground_surface_pairwise_head(
                    input_image,
                    return_trace=True,
                )
                features["foreground_surface_pairwise_logits"] = (
                    foreground_surface_pairwise_logits
                )
            else:
                features["foreground_surface_pairwise_logits"] = (
                    self.foreground_surface_pairwise_head(input_image)
                )
        interior_boundary_pairwise_trace = None
        if self.interior_boundary_pairwise_head is not None:
            interior_boundary_bbox = features.get("bbox")
            if return_trace:
                (
                    interior_boundary_pairwise_logits,
                    interior_boundary_pairwise_trace,
                ) = self.interior_boundary_pairwise_head(
                    input_image,
                    bbox=interior_boundary_bbox,
                    return_trace=True,
                )
                features["interior_boundary_pairwise_logits"] = (
                    interior_boundary_pairwise_logits
                )
            else:
                features["interior_boundary_pairwise_logits"] = (
                    self.interior_boundary_pairwise_head(
                        input_image,
                        bbox=interior_boundary_bbox,
                    )
                )
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
                "shifted_patch_residual_norm": (
                    shifted_patch_residual.detach().float().norm(dim=-1)
                    if shifted_patch_residual is not None
                    else patch_tokens.new_zeros((batch_size, original_patch_count))
                ),
                "detail_map": detail_map,
                "foreground_prior": foreground_prior,
                "pruning": pruning_trace,
            }
            if self.inattentive_token_fusion_enabled:
                features["trace"]["inattentive_context"] = context_out
                features["trace"]["active_prefix_count"] = torch.tensor(
                    active_prefix_count,
                    device=tokens.device,
                    dtype=torch.long,
                )
            if self.cropr_token_selector_enabled:
                features["trace"]["cropr_token_selector_layers"] = torch.tensor(
                    [int(layer) for layer in self.cropr_token_selectors.keys()],
                    device=tokens.device,
                    dtype=torch.long,
                )
                features["trace"]["cropr_token_selector_routing"] = torch.tensor(
                    self.cropr_token_selector_routing,
                    device=tokens.device,
                    dtype=torch.bool,
                )
            if deep_class_prompt_logits is not None:
                features["trace"]["deep_class_prompt_layers"] = torch.arange(
                    1,
                    len(deep_class_prompt_tokens) + 1,
                    device=tokens.device,
                    dtype=torch.long,
                )
                features["trace"]["deep_class_prompt_token_norms"] = torch.stack(
                    [
                        prompt.detach().float().norm(dim=-1)
                        for prompt in deep_class_prompt_tokens
                    ],
                    dim=0,
                )
                features["trace"]["deep_class_prompt_logits"] = (
                    deep_class_prompt_logits.detach()
                )
                features["trace"]["deep_class_prompt_attentions"] = {
                    int(layer_index): value.detach()
                    for layer_index, value in deep_class_prompt_attention_maps.items()
                }
            if bbox_patch_prior is not None:
                features["trace"]["bbox_patch_prior"] = bbox_patch_prior.detach()
            if gabor_texture_trace is not None:
                for trace_key, trace_value in gabor_texture_trace.items():
                    features["trace"][f"gabor_texture_{trace_key}"] = trace_value
            if gabor_texture_semantic_trace is not None:
                for trace_key, trace_value in gabor_texture_semantic_trace.items():
                    features["trace"][
                        f"gabor_texture_semantic_{trace_key}"
                    ] = trace_value
            if multi_granularity_logits:
                features["trace"]["multi_granularity_layers"] = torch.tensor(
                    [int(layer) for layer in multi_granularity_logits.keys()],
                    device=tokens.device,
                    dtype=torch.long,
                )
                features["trace"]["multi_granularity_feature_shapes"] = [
                    tuple(int(value) for value in feature.shape)
                    for feature in multi_granularity_features.values()
                ]
            if self.block_local_patch_mixer_enabled:
                features["trace"]["block_local_patch_mixer_layers"] = torch.tensor(
                    self.block_local_patch_mixer_layer_numbers,
                    device=tokens.device,
                    dtype=torch.long,
                )
            if self.locally_enhanced_ffn_enabled:
                features["trace"]["locally_enhanced_ffn_layers"] = torch.tensor(
                    self.locally_enhanced_ffn_layer_numbers,
                    device=tokens.device,
                    dtype=torch.long,
                )
            if self.concurrent_local_global_enabled and concurrent_local_state is not None:
                features["trace"]["concurrent_local_global_layers"] = torch.tensor(
                    self.concurrent_local_global_layer_numbers,
                    device=tokens.device,
                    dtype=torch.long,
                )
                features["trace"]["concurrent_local_activation"] = (
                    concurrent_local_state.detach().float().abs().mean(dim=1, keepdim=True)
                )
                if concurrent_local_traces:
                    for trace_key in (
                        "local_state_norm",
                        "token_residual_norm",
                        "token_to_local_scale",
                        "local_update_scale",
                        "local_to_token_scale",
                    ):
                        features["trace"][f"concurrent_{trace_key}"] = torch.stack(
                            [entry[trace_key] for entry in concurrent_local_traces],
                            dim=1,
                        )
            if self.gated_relative_position_attention_enabled:
                relative_position_entries = []
                for layer_number in self.gated_relative_position_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].attn.relative_position_attention
                    module_trace = module.trace() if module is not None else {}
                    if module_trace:
                        relative_position_entries.append((int(layer_number), module_trace))
                if relative_position_entries:
                    features["trace"]["relative_position_attention_layers"] = torch.tensor(
                        [layer for layer, _ in relative_position_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key, output_key in (
                        ("gate", "relative_position_attention_gate"),
                        ("raw_gate", "relative_position_attention_raw_gate"),
                        ("position_distance", "relative_position_attention_position_distance"),
                        ("content_distance", "relative_position_attention_content_distance"),
                        ("mixed_distance", "relative_position_attention_mixed_distance"),
                        ("position_local_mass", "relative_position_attention_local_mass"),
                        ("center_map", "relative_position_attention_center_map"),
                    ):
                        features["trace"][output_key] = torch.stack(
                            [entry[trace_key] for _, entry in relative_position_entries],
                            dim=0,
                        )
            if self.visual_contrast_attention_enabled:
                visual_contrast_entries = []
                for layer_number in self.visual_contrast_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].attn
                    module_trace = module.trace() if hasattr(module, "trace") else {}
                    if module_trace:
                        visual_contrast_entries.append((int(layer_number), module_trace))
                if visual_contrast_entries:
                    features["trace"]["visual_contrast_attention_layers"] = torch.tensor(
                        [layer for layer, _ in visual_contrast_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key in (
                        "lambda_stage1",
                        "lambda_stage2",
                        "stage1_positive_mass",
                        "stage1_negative_mass",
                        "stage2_positive_mass",
                        "stage2_negative_mass",
                        "stage1_contrast_norm",
                        "stage2_contrast_norm",
                    ):
                        features["trace"][f"visual_contrast_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in visual_contrast_entries],
                            dim=0,
                        )
            if self.foveal_aggregated_attention_enabled:
                foveal_entries = []
                for layer_number in self.foveal_aggregated_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].attn
                    module_trace = module.trace() if hasattr(module, "trace") else {}
                    if module_trace:
                        foveal_entries.append((int(layer_number), module_trace))
                if foveal_entries:
                    features["trace"]["foveal_aggregated_attention_layers"] = torch.tensor(
                        [layer for layer, _ in foveal_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key in (
                        "local_mass_mean",
                        "local_mass_min",
                        "local_mass_max",
                        "pooled_mass_mean",
                        "pooled_mass_min",
                        "pooled_mass_max",
                        "dual_route_fraction",
                        "temperature_mean",
                        "patch_count",
                    ):
                        features["trace"][f"foveal_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in foveal_entries],
                            dim=0,
                        )
            if self.deformable_spatial_attention_enabled:
                deformable_entries = []
                for layer_number in self.deformable_spatial_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].attn
                    module_trace = module.trace() if hasattr(module, "trace") else {}
                    if module_trace:
                        deformable_entries.append((int(layer_number), module_trace))
                if deformable_entries:
                    features["trace"]["deformable_spatial_attention_layers"] = (
                        torch.tensor(
                            [layer for layer, _ in deformable_entries],
                            device=tokens.device,
                            dtype=torch.long,
                        )
                    )
                    for trace_key in (
                        "offset_rms",
                        "offset_rms_per_group",
                        "inter_group_position_rms",
                        "valid_interpolation_mass_mean",
                        "valid_interpolation_mass_min",
                        "position_min",
                        "position_max",
                    ):
                        features["trace"][f"deformable_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in deformable_entries],
                            dim=0,
                        )
            if self.bi_level_routing_attention_enabled:
                routing_entries = []
                for layer_number in self.bi_level_routing_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].attn
                    module_trace = module.trace() if hasattr(module, "trace") else {}
                    if module_trace:
                        routing_entries.append((int(layer_number), module_trace))
                if routing_entries:
                    features["trace"]["bi_level_routing_attention_layers"] = (
                        torch.tensor(
                            [layer for layer, _ in routing_entries],
                            device=tokens.device,
                            dtype=torch.long,
                        )
                    )
                    for trace_key in (
                        "topk",
                        "region_count",
                        "patch_count",
                        "selected_affinity_margin_mean",
                        "route_distance_mean",
                        "nonlocal_route_fraction",
                        "distinct_route_sets_mean",
                        "pairwise_route_jaccard_mean",
                        "local_context_norm_ratio",
                    ):
                        features["trace"][f"bi_level_routing_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in routing_entries],
                            dim=0,
                        )
            if self.cross_covariance_attention_enabled:
                cross_covariance_entries = []
                for layer_number in self.cross_covariance_attention_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].cross_covariance_attention
                    module_trace = module.trace() if module is not None else {}
                    if module_trace:
                        cross_covariance_entries.append((int(layer_number), module_trace))
                if cross_covariance_entries:
                    features["trace"]["cross_covariance_attention_layers"] = torch.tensor(
                        [layer for layer, _ in cross_covariance_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key in (
                        "temperature_min",
                        "temperature_mean",
                        "temperature_max",
                        "normalized_entropy",
                        "diagonal_mass",
                        "query_norm_max_error",
                        "key_norm_max_error",
                        "residual_norm_ratio",
                        "patch_count",
                    ):
                        features["trace"][f"cross_covariance_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in cross_covariance_entries],
                            dim=0,
                        )
            if self.soft_moe_patch_adapter_enabled:
                soft_moe_entries = []
                for layer_number in self.soft_moe_patch_adapter_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].soft_moe_patch_adapter
                    module_trace = module.trace() if module is not None else {}
                    if module_trace:
                        soft_moe_entries.append((int(layer_number), module_trace))
                if soft_moe_entries:
                    features["trace"]["soft_moe_patch_adapter_layers"] = torch.tensor(
                        [layer for layer, _ in soft_moe_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    features["trace"]["soft_moe_router_logits"] = {
                        layer: entry["router_logits"]
                        for layer, entry in soft_moe_entries
                    }
                    features["trace"]["soft_moe_dispatch_weights"] = {
                        layer: entry["dispatch_weights"]
                        for layer, entry in soft_moe_entries
                    }
                    features["trace"]["soft_moe_combine_weights"] = {
                        layer: entry["combine_weights"]
                        for layer, entry in soft_moe_entries
                    }
                    features["trace"]["soft_moe_residual_norm"] = {
                        layer: entry["residual_norm"]
                        for layer, entry in soft_moe_entries
                    }
                    features["trace"]["soft_moe_patch_indices"] = {
                        layer: entry["patch_indices"]
                        for layer, entry in soft_moe_entries
                    }
                    for trace_key in (
                        "residual_norm_ratio",
                        "combine_mass",
                        "combine_entropy",
                        "dispatch_similarity_off_diagonal",
                    ):
                        features["trace"][f"soft_moe_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in soft_moe_entries],
                            dim=0,
                        )
            if self.patch_style_recalibration_enabled:
                patch_style_entries = []
                for layer_number in self.patch_style_recalibration_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].mlp.style_recalibration
                    module_trace = module.trace() if module is not None else {}
                    if module_trace:
                        patch_style_entries.append((int(layer_number), module_trace))
                if patch_style_entries:
                    features["trace"]["patch_style_recalibration_layers"] = torch.tensor(
                        [layer for layer, _ in patch_style_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key in (
                        "gate_min",
                        "gate_mean",
                        "gate_max",
                        "gate_channel_std",
                        "gate_sample_std",
                        "style_mean_abs",
                        "style_std_mean",
                        "cfc_l2_norm",
                        "patch_hidden_norm_ratio",
                        "patch_count",
                    ):
                        features["trace"][f"patch_style_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in patch_style_entries],
                            dim=0,
                        )
            if self.dynamic_graph_mixer_enabled:
                graph_entries = []
                for layer_number in self.dynamic_graph_mixer_layer_numbers:
                    module = self.blocks[int(layer_number) - 1].dynamic_graph_mixer
                    module_trace = module.trace() if module is not None else {}
                    if module_trace:
                        graph_entries.append((int(layer_number), module_trace))
                if graph_entries:
                    features["trace"]["dynamic_graph_mixer_layers"] = torch.tensor(
                        [layer for layer, _ in graph_entries],
                        device=tokens.device,
                        dtype=torch.long,
                    )
                    for trace_key in (
                        "residual_norm_ratio",
                        "projected_node_norm",
                        "neighbor_distance_mean",
                        "nonself_neighbor_count",
                        "nonlocal_neighbor_fraction",
                        "neighbor_selection_entropy",
                        "patch_count",
                        "neighbor_count",
                    ):
                        features["trace"][f"dynamic_graph_{trace_key}"] = torch.stack(
                            [entry[trace_key] for _, entry in graph_entries],
                            dim=0,
                        )
            if layer_token_fusion_feature is not None:
                features["trace"]["layer_token_fusion_layers"] = torch.tensor(
                    layer_token_fusion_layers,
                    device=tokens.device,
                    dtype=torch.long,
                )
                features["trace"]["layer_token_fusion_feature"] = (
                    layer_token_fusion_feature.detach()
                )
                features["trace"]["layer_token_fusion_scores"] = layer_token_fusion_scores
                features["trace"]["layer_token_fusion_indices"] = layer_token_fusion_indices
            if foreground_surface_trace is not None:
                features["trace"]["foreground_surface_stats"] = foreground_surface_trace["stats"]
                features["trace"]["foreground_surface_weight_map"] = foreground_surface_trace["foreground_weight_map"]
                features["trace"]["foreground_surface_mask"] = foreground_surface_trace["foreground_mask"]
                features["trace"]["foreground_surface_edge_detail"] = foreground_surface_trace["edge_detail"]
                features["trace"]["foreground_surface_dark_spot"] = foreground_surface_trace["dark_spot"]
                features["trace"]["foreground_surface_brown_spot"] = foreground_surface_trace["brown_spot"]
                features["trace"]["foreground_surface_bright_spot"] = foreground_surface_trace["bright_spot"]
            if foreground_surface_pairwise_trace is not None:
                features["trace"]["foreground_surface_pairwise_stats"] = (
                    foreground_surface_pairwise_trace["stats"]
                )
            if interior_boundary_pairwise_trace is not None:
                features["trace"]["interior_boundary_pairwise_stats"] = (
                    interior_boundary_pairwise_trace["stats"]
                )
                features["trace"]["interior_boundary_foreground_mask"] = (
                    interior_boundary_pairwise_trace["foreground_mask"]
                )
                features["trace"]["interior_boundary_interior_mask"] = (
                    interior_boundary_pairwise_trace["interior_mask"]
                )
                features["trace"]["interior_boundary_boundary_mask"] = (
                    interior_boundary_pairwise_trace["boundary_mask"]
                )
                features["trace"]["interior_boundary_interior_weight_map"] = (
                    interior_boundary_pairwise_trace["interior_weight_map"]
                )
                features["trace"]["interior_boundary_boundary_weight_map"] = (
                    interior_boundary_pairwise_trace["boundary_weight_map"]
                )
                if "bbox_mask" in interior_boundary_pairwise_trace:
                    features["trace"]["interior_boundary_bbox_mask"] = (
                        interior_boundary_pairwise_trace["bbox_mask"]
                    )
                features["trace"]["interior_boundary_edge_detail"] = (
                    interior_boundary_pairwise_trace["edge_detail"]
                )
                features["trace"]["interior_boundary_boundary_edge_detail"] = (
                    interior_boundary_pairwise_trace["boundary_edge_detail"]
                )
                features["trace"]["interior_boundary_interior_dark_spot"] = (
                    interior_boundary_pairwise_trace["interior_dark_spot"]
                )
                features["trace"]["interior_boundary_boundary_brown_spot"] = (
                    interior_boundary_pairwise_trace["boundary_brown_spot"]
                )
            if local_zoom_trace is not None:
                features["trace"]["local_zoom_score_map"] = local_zoom_trace["score_map"]
                features["trace"]["local_zoom_crop_boxes"] = local_zoom_trace["crop_boxes"]
                features["trace"]["local_zoom_descriptor"] = local_zoom_trace["descriptor"]
            if high_frequency_texture_trace is not None:
                features["trace"]["high_frequency_texture_descriptor"] = (
                    high_frequency_texture_trace["descriptor"]
                )
                features["trace"]["high_frequency_texture_high_pass"] = (
                    high_frequency_texture_trace["high_pass"]
                )
                features["trace"]["high_frequency_texture_gradient"] = (
                    high_frequency_texture_trace["gradient"]
                )
                features["trace"]["high_frequency_texture_laplacian"] = (
                    high_frequency_texture_trace["laplacian"]
                )
                features["trace"]["high_frequency_texture_foreground_weight"] = (
                    high_frequency_texture_trace["foreground_weight"]
                )
                features["trace"]["high_frequency_texture_foreground_detail"] = (
                    high_frequency_texture_trace["foreground_detail"]
                )
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
        valid_fraction = _exportable_adaptive_avg_pool2d(
            mask,
            output_size=grid_size,
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
        if self.patch_memory_adapter is not None and "patches" in features:
            grid_size = features.get("grid_size", (1, int(features["patches"].size(1))))
            features["patches"] = self.patch_memory_adapter(features["patches"], grid_size)
            if isinstance(features.get("trace"), dict):
                features["trace"]["patch_memory_adapter_enabled"] = True
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
        layer_token_feature = features.get("layer_token_fusion_feature")
        if torch.is_tensor(layer_token_feature):
            blend = float(self.layer_token_fusion_blend)
            if blend > 0.0:
                pooled = pooled * (1.0 - blend) + layer_token_feature.to(
                    dtype=pooled.dtype,
                    device=pooled.device,
                ) * blend
        if self.late_class_attention_pool is not None and "patches" in features:
            return_trace = isinstance(features.get("trace"), dict)
            late_output = self.late_class_attention_pool(
                pooled,
                features["patches"],
                valid_mask=valid_mask,
                return_trace=return_trace,
            )
            if return_trace:
                pooled, late_trace = late_output
                features["trace"]["late_class_attention"] = late_trace["attention"]
                features["trace"]["late_class_attention_context_norm"] = late_trace[
                    "context_norm"
                ]
                features["trace"]["late_class_attention_residual_norm"] = late_trace[
                    "residual_norm"
                ]
            else:
                pooled = late_output
            features["late_class_attention_feature"] = pooled
        texture_feature = features.get("gabor_texture_semantic_feature")
        if torch.is_tensor(texture_feature):
            texture_feature = texture_feature.to(
                device=pooled.device,
                dtype=pooled.dtype,
            )
            semantic_feature = pooled
            pooled = semantic_feature + texture_feature
            features["gabor_texture_semantic_head_input"] = pooled
            if isinstance(features.get("trace"), dict):
                semantic_float = semantic_feature.detach().float()
                texture_float = texture_feature.detach().float()
                fused_float = pooled.detach().float()
                features["trace"]["gabor_texture_semantic_semantic_feature"] = (
                    semantic_float
                )
                features["trace"]["gabor_texture_semantic_fused_feature"] = (
                    fused_float
                )
                features["trace"]["gabor_texture_semantic_semantic_norm"] = (
                    semantic_float.norm(dim=-1)
                )
                features["trace"]["gabor_texture_semantic_texture_norm"] = (
                    texture_float.norm(dim=-1)
                )
                features["trace"]["gabor_texture_semantic_fused_norm"] = (
                    fused_float.norm(dim=-1)
                )
                features["trace"]["gabor_texture_semantic_cosine"] = (
                    F.cosine_similarity(semantic_float, texture_float, dim=-1)
                )
        return pooled

    def late_member_logits_from_features(self, features: Dict[str, Tensor]) -> Tensor:
        if not self.late_member_enabled:
            raise RuntimeError("Late-member branch is disabled.")
        if self.late_member_head is None:
            raise RuntimeError("Late-member classification head is missing.")
        if self.patch_evidence_linear_verifier is not None:
            raise ValueError(
                "Late-member fusion cannot be combined with the post-hoc patch verifier."
            )
        pooled = features.get("late_member_pooled")
        patches = features.get("late_member_patches")
        if not torch.is_tensor(pooled) or not torch.is_tensor(patches):
            raise KeyError("Late-member features are missing from forward_features output.")

        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        if self.late_member_fine_grained_pool is not None:
            patch_attention = self.late_member_fine_grained_pool.attention_weights(
                pooled,
                patches,
                valid_mask=valid_mask,
            )
            features["late_member_fine_grained_attention"] = patch_attention
            pooled = self.late_member_fine_grained_pool(
                pooled,
                patches,
                attention=patch_attention,
            )
        features["late_member_head_input"] = pooled
        logits = self.late_member_head(pooled)

        if self.late_member_cnn_fusion_head is not None and "cnn_pooled" in features:
            if (
                self.late_member_cnn_fusion_norm is None
                or self.late_member_cnn_fusion_dropout is None
            ):
                raise RuntimeError("Late-member CNN fusion modules are incomplete.")
            cnn_features = self.late_member_cnn_fusion_norm(features["cnn_pooled"])
            cnn_features = self.late_member_cnn_fusion_dropout(cnn_features)
            logits = logits + self.late_member_cnn_fusion_head(cnn_features)

        bbox_value = features.get("bbox")
        if self.late_member_bbox_spatial_fusion_head is not None and torch.is_tensor(
            bbox_value
        ):
            bbox_logits = self.late_member_bbox_spatial_fusion_head(
                bbox_value,
                dtype=logits.dtype,
                return_trace=False,
            )
            features["late_member_bbox_spatial_logits"] = bbox_logits
            if self.bbox_spatial_fusion_logit_scale > 0.0:
                logits = logits + bbox_logits.to(dtype=logits.dtype) * float(
                    self.bbox_spatial_fusion_logit_scale
                )

        if self.late_member_pairwise_margin_head is not None:
            if (
                self.late_member_pairwise_margin_norm is None
                or self.late_member_pairwise_margin_dropout is None
            ):
                raise RuntimeError("Late-member pairwise modules are incomplete.")
            pairwise_input = self.late_member_pairwise_margin_norm(pooled)
            pairwise_input = self.late_member_pairwise_margin_dropout(pairwise_input)
            pairwise_logits = self.late_member_pairwise_margin_head(pairwise_input)
            features["late_member_pairwise_margin_logits"] = pairwise_logits
            logits = logits + self.pairwise_margin_adjustment(pairwise_logits, logits)
        return logits

    def fuse_late_member_logits(
        self,
        primary_logits: Tensor,
        candidate_logits: Tensor,
    ) -> Tensor:
        if tuple(primary_logits.shape) != tuple(candidate_logits.shape):
            raise ValueError(
                "Late-member logits shape mismatch: "
                f"primary={tuple(primary_logits.shape)}, "
                f"candidate={tuple(candidate_logits.shape)}"
            )
        candidate_weight = float(self.late_member_candidate_weight)
        primary_probabilities = F.softmax(primary_logits.float(), dim=1)
        candidate_probabilities = F.softmax(candidate_logits.float(), dim=1)
        probabilities = (
            primary_probabilities * (1.0 - candidate_weight)
            + candidate_probabilities * candidate_weight
        )
        if self.late_member_focus_margin_offset > 0.0:
            probabilities = probabilities.clone()
            probabilities[:, self.late_member_focus_class] = (
                probabilities[:, self.late_member_focus_class]
                - float(self.late_member_focus_margin_offset)
            ).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        return probabilities.clamp_min(1e-8).log().to(dtype=primary_logits.dtype)

    def pairwise_margin_logits_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.pairwise_margin_head is None:
            return None
        pairwise_input = self.pairwise_margin_norm(head_input)
        pairwise_input = self.pairwise_margin_dropout(pairwise_input)
        return self.pairwise_margin_head(pairwise_input)

    def focus_class_logit_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.focus_class_head is None:
            return None
        focus_input = self.focus_class_norm(head_input)
        focus_input = self.focus_class_dropout(focus_input)
        return self.focus_class_head(focus_input).squeeze(1)

    def class_independent_logits_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.class_independent_head is None:
            return None
        ova_input = self.class_independent_norm(head_input)
        ova_input = self.class_independent_dropout(ova_input)
        return self.class_independent_head(ova_input)

    def subcenter_proxy_logits_from_head_input(
        self,
        head_input: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if self.subcenter_proxy_head is None:
            return None
        return self.subcenter_proxy_head(head_input)

    def deep_abstention_logit_from_head_input(
        self,
        head_input: Tensor,
        class_logits: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self.deep_abstention_head is None:
            return None
        abstention_input = self.deep_abstention_norm(head_input)
        abstention_input = self.deep_abstention_dropout(abstention_input)
        abstention_log_odds = self.deep_abstention_head(abstention_input).squeeze(1)
        if torch.is_tensor(class_logits):
            class_log_normalizer = torch.logsumexp(
                class_logits.detach().float(),
                dim=1,
            ).to(dtype=abstention_log_odds.dtype)
            return abstention_log_odds + class_log_normalizer
        return abstention_log_odds

    def focus_class_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.focus_class_routing:
            return like_logits.new_ones((batch_size,))
        class_index = int(self.focus_class_index)
        if class_index < 0 or class_index >= int(like_logits.size(1)):
            return like_logits.new_zeros((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=min(2, int(probabilities.size(1))), dim=1)
            if int(top_indices.size(1)) == 1:
                top1 = top_indices[:, 0]
                top2 = top1
                probability_margin = torch.zeros_like(top_probabilities[:, 0])
            else:
                top1 = top_indices[:, 0]
                top2 = top_indices[:, 1]
                probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.focus_class_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            focus_probability = probabilities[:, class_index]
            route_match = (
                (top1 == class_index)
                | (top2 == class_index)
                | (focus_probability >= float(self.focus_class_route_min_probability))
            )
            route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
        return route_weights.to(dtype=like_logits.dtype)

    def focus_class_adjustment(
        self,
        focus_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(focus_logits) or focus_logits.ndim != 1:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        class_index = int(self.focus_class_index)
        if class_index < 0 or class_index >= int(like_logits.size(1)):
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.focus_class_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.focus_class_route_weights(like_logits)
        score = focus_logits.to(dtype=like_logits.dtype) * route_weights * scale
        adjustment[:, class_index] = adjustment[:, class_index] + score
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def foreground_surface_pairwise_route_weights(self, like_logits: Tensor) -> Tensor:
        pair_count = len(self.foreground_surface_pairwise_pairs)
        if like_logits.ndim != 2:
            batch_size = 0
        else:
            batch_size = int(like_logits.size(0))
        route_weights = like_logits.new_zeros((batch_size, pair_count))
        if pair_count == 0:
            return route_weights
        if not self.foreground_surface_pairwise_routing:
            return route_weights.fill_(1.0)
        if like_logits.ndim != 2 or like_logits.size(1) < 2:
            return route_weights

        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.foreground_surface_pairwise_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            for pair_index, (left_class, right_class) in enumerate(
                self.foreground_surface_pairwise_pairs
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

    def foreground_surface_pairwise_adjustment(
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
                    (like_logits.size(0), len(self.foreground_surface_pairwise_pairs))
                )
            return adjustment
        scale = float(self.foreground_surface_pairwise_logit_scale)
        if scale <= 0.0:
            if return_route_weights:
                return adjustment, like_logits.new_zeros(
                    (like_logits.size(0), len(self.foreground_surface_pairwise_pairs))
                )
            return adjustment
        route_weights = self.foreground_surface_pairwise_route_weights(like_logits)
        class_count = int(like_logits.size(1))
        for pair_index, (left_class, right_class) in enumerate(
            self.foreground_surface_pairwise_pairs
        ):
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

    def interior_boundary_pairwise_route_weights(self, like_logits: Tensor) -> Tensor:
        pair_count = len(self.interior_boundary_pairwise_pairs)
        if like_logits.ndim != 2:
            batch_size = 0
        else:
            batch_size = int(like_logits.size(0))
        route_weights = like_logits.new_zeros((batch_size, pair_count))
        if pair_count == 0:
            return route_weights
        if not self.interior_boundary_pairwise_routing:
            return route_weights.fill_(1.0)
        if like_logits.ndim != 2 or like_logits.size(1) < 2:
            return route_weights

        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.interior_boundary_pairwise_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            for pair_index, (left_class, right_class) in enumerate(
                self.interior_boundary_pairwise_pairs
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

    def interior_boundary_pairwise_adjustment(
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
                    (like_logits.size(0), len(self.interior_boundary_pairwise_pairs))
                )
            return adjustment
        scale = float(self.interior_boundary_pairwise_logit_scale)
        if scale <= 0.0:
            if return_route_weights:
                return adjustment, like_logits.new_zeros(
                    (like_logits.size(0), len(self.interior_boundary_pairwise_pairs))
                )
            return adjustment
        route_weights = self.interior_boundary_pairwise_route_weights(like_logits)
        class_count = int(like_logits.size(1))
        for pair_index, (left_class, right_class) in enumerate(
            self.interior_boundary_pairwise_pairs
        ):
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

    def micro_detail_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.micro_detail_routing:
            return like_logits.new_ones((batch_size,))
        if like_logits.size(1) < 2:
            return like_logits.new_zeros((batch_size,))
        pairs = list(self.micro_detail_route_pairs)
        if not pairs:
            return like_logits.new_ones((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.micro_detail_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            route_match = torch.zeros_like(ambiguity_weight, dtype=torch.bool)
            for left_class, right_class in pairs:
                if int(left_class) < 0:
                    route_match = route_match | (top1 == int(right_class)) | (top2 == int(right_class))
                else:
                    route_match = route_match | (
                        ((top1 == int(left_class)) & (top2 == int(right_class)))
                        | ((top1 == int(right_class)) & (top2 == int(left_class)))
                    )
            route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
        return route_weights.to(dtype=like_logits.dtype)

    def micro_detail_adjustment(
        self,
        micro_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(micro_logits) or micro_logits.ndim != 2:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.micro_detail_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.micro_detail_route_weights(like_logits)
        adjustment = micro_logits.to(dtype=like_logits.dtype) * scale * route_weights[:, None]
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def part_token_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.part_token_routing:
            return like_logits.new_ones((batch_size,))
        if like_logits.size(1) < 2:
            return like_logits.new_zeros((batch_size,))
        pairs = list(self.part_token_route_pairs)
        if not pairs:
            return like_logits.new_ones((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.part_token_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            route_match = torch.zeros_like(ambiguity_weight, dtype=torch.bool)
            for left_class, right_class in pairs:
                if int(left_class) < 0:
                    route_match = route_match | (top1 == int(right_class)) | (top2 == int(right_class))
                else:
                    route_match = route_match | (
                        ((top1 == int(left_class)) & (top2 == int(right_class)))
                        | ((top1 == int(right_class)) & (top2 == int(left_class)))
                    )
            route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
        return route_weights.to(dtype=like_logits.dtype)

    def part_token_adjustment(
        self,
        part_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(part_logits) or part_logits.ndim != 2:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.part_token_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.part_token_route_weights(like_logits)
        adjustment = part_logits.to(dtype=like_logits.dtype) * scale * route_weights[:, None]
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def part_token_pairwise_route_weights(self, like_logits: Tensor) -> Tensor:
        pair_count = len(self.part_token_pairwise_pairs)
        if like_logits.ndim != 2:
            batch_size = 0
        else:
            batch_size = int(like_logits.size(0))
        route_weights = like_logits.new_zeros((batch_size, pair_count))
        if pair_count == 0:
            return route_weights
        if not self.part_token_pairwise_routing:
            return route_weights.fill_(1.0)
        if like_logits.ndim != 2 or like_logits.size(1) < 2:
            return route_weights

        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.part_token_pairwise_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            for pair_index, (left_class, right_class) in enumerate(
                self.part_token_pairwise_pairs
            ):
                if int(left_class) < 0:
                    route_match = (top1 == int(right_class)) | (top2 == int(right_class))
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

    def part_token_pairwise_adjustment(
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
                    (like_logits.size(0), len(self.part_token_pairwise_pairs))
                )
            return adjustment
        scale = float(self.part_token_pairwise_logit_scale)
        if scale <= 0.0:
            if return_route_weights:
                return adjustment, like_logits.new_zeros(
                    (like_logits.size(0), len(self.part_token_pairwise_pairs))
                )
            return adjustment
        route_weights = self.part_token_pairwise_route_weights(like_logits)
        class_count = int(like_logits.size(1))
        for pair_index, (left_class, right_class) in enumerate(self.part_token_pairwise_pairs):
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

    def patch_evidence_router_route_weights(self, like_logits: Tensor) -> Tensor:
        pair_count = len(self.patch_evidence_router_pairs)
        if like_logits.ndim != 2:
            batch_size = 0
        else:
            batch_size = int(like_logits.size(0))
        route_weights = like_logits.new_zeros((batch_size, pair_count))
        if pair_count == 0:
            return route_weights
        if not self.patch_evidence_router_routing:
            return route_weights.fill_(1.0)
        if like_logits.ndim != 2 or like_logits.size(1) < 2:
            return route_weights
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.patch_evidence_router_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            min_pair_probability = float(self.patch_evidence_router_route_min_pair_probability)
            for pair_index, (left_class, right_class) in enumerate(
                self.patch_evidence_router_pairs
            ):
                left_class = int(left_class)
                right_class = int(right_class)
                if (
                    left_class < 0
                    or right_class < 0
                    or left_class >= int(probabilities.size(1))
                    or right_class >= int(probabilities.size(1))
                ):
                    continue
                pair_probability = probabilities[:, left_class] + probabilities[:, right_class]
                route_match = (
                    ((top1 == left_class) & (top2 == right_class))
                    | ((top1 == right_class) & (top2 == left_class))
                ) & (pair_probability >= min_pair_probability)
                route_weights[:, pair_index] = (
                    route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
                ).to(dtype=route_weights.dtype)
        return route_weights

    def patch_evidence_router_adjustment(
        self,
        pairwise_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        pair_count = len(self.patch_evidence_router_pairs)
        if not torch.is_tensor(pairwise_logits) or pairwise_logits.ndim != 2:
            route_weights = like_logits.new_zeros((like_logits.size(0), pair_count))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.patch_evidence_router_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0), pair_count))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.patch_evidence_router_route_weights(like_logits)
        class_count = int(like_logits.size(1))
        for pair_index, (left_class, right_class) in enumerate(
            self.patch_evidence_router_pairs
        ):
            if pair_index >= int(pairwise_logits.size(1)):
                break
            score = pairwise_logits[:, pair_index].to(dtype=like_logits.dtype) * scale
            if pair_index < int(route_weights.size(1)):
                score = score * route_weights[:, pair_index]
            if 0 <= int(left_class) < class_count:
                adjustment[:, int(left_class)] = adjustment[:, int(left_class)] - score
            if 0 <= int(right_class) < class_count:
                adjustment[:, int(right_class)] = adjustment[:, int(right_class)] + score
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def local_zoom_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.local_zoom_routing:
            return like_logits.new_ones((batch_size,))
        if like_logits.size(1) < 2:
            return like_logits.new_zeros((batch_size,))
        pairs = list(self.local_zoom_route_pairs)
        if not pairs:
            return like_logits.new_ones((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.local_zoom_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            route_match = torch.zeros_like(ambiguity_weight, dtype=torch.bool)
            for left_class, right_class in pairs:
                if int(left_class) < 0:
                    route_match = route_match | (top1 == int(right_class)) | (top2 == int(right_class))
                else:
                    route_match = route_match | (
                        ((top1 == int(left_class)) & (top2 == int(right_class)))
                        | ((top1 == int(right_class)) & (top2 == int(left_class)))
                    )
            route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
        return route_weights.to(dtype=like_logits.dtype)

    def local_zoom_adjustment(
        self,
        local_zoom_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(local_zoom_logits) or local_zoom_logits.ndim != 2:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.local_zoom_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.local_zoom_route_weights(like_logits)
        adjustment = local_zoom_logits.to(dtype=like_logits.dtype) * scale * route_weights[:, None]
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

    def high_frequency_texture_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.high_frequency_texture_routing:
            return like_logits.new_ones((batch_size,))
        if like_logits.size(1) < 2:
            return like_logits.new_zeros((batch_size,))
        pairs = list(self.high_frequency_texture_route_pairs)
        if not pairs:
            return like_logits.new_ones((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.high_frequency_texture_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (
                    1.0 - probability_margin / max_margin
                ).clamp(min=0.0, max=1.0)
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            top1 = top_indices[:, 0]
            top2 = top_indices[:, 1]
            route_match = torch.zeros_like(ambiguity_weight, dtype=torch.bool)
            for left_class, right_class in pairs:
                if int(left_class) < 0:
                    route_match = route_match | (top1 == int(right_class)) | (top2 == int(right_class))
                else:
                    route_match = route_match | (
                        ((top1 == int(left_class)) & (top2 == int(right_class)))
                        | ((top1 == int(right_class)) & (top2 == int(left_class)))
                    )
            route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
        return route_weights.to(dtype=like_logits.dtype)

    def high_frequency_texture_adjustment(
        self,
        high_frequency_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if not torch.is_tensor(high_frequency_logits) or high_frequency_logits.ndim != 2:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.high_frequency_texture_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.high_frequency_texture_route_weights(like_logits)
        adjustment = high_frequency_logits.to(dtype=like_logits.dtype) * scale * route_weights[:, None]
        if return_route_weights:
            return adjustment, route_weights.detach()
        return adjustment

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

    def topk_reassessment_logits_from_head_input(
        self,
        head_input: Tensor,
        like_logits: Tensor,
    ) -> Optional[Tensor]:
        if self.topk_reassessment_head is None:
            return None
        return self.topk_reassessment_head(head_input, like_logits.detach())

    def topk_reassessment_route_weights(self, like_logits: Tensor) -> Tensor:
        if like_logits.ndim != 2:
            return like_logits.new_zeros((0,))
        batch_size = int(like_logits.size(0))
        if not self.topk_reassessment_routing:
            return like_logits.new_ones((batch_size,))
        if like_logits.size(1) < 2:
            return like_logits.new_zeros((batch_size,))
        with torch.no_grad():
            probabilities = like_logits.detach().float().softmax(dim=1)
            top_probabilities, top_indices = probabilities.topk(k=2, dim=1)
            probability_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
            max_margin = float(self.topk_reassessment_route_max_probability_margin)
            if max_margin > 0.0:
                ambiguity_weight = (1.0 - probability_margin / max_margin).clamp(
                    min=0.0,
                    max=1.0,
                )
            else:
                ambiguity_weight = torch.ones_like(probability_margin)
            if self.topk_reassessment_route_pairs:
                top1 = top_indices[:, 0]
                top2 = top_indices[:, 1]
                route_match = torch.zeros_like(ambiguity_weight, dtype=torch.bool)
                for left_class, right_class in self.topk_reassessment_route_pairs:
                    if int(left_class) < 0:
                        route_match = (
                            route_match
                            | (top1 == int(right_class))
                            | (top2 == int(right_class))
                        )
                    else:
                        route_match = route_match | (
                            ((top1 == int(left_class)) & (top2 == int(right_class)))
                            | ((top1 == int(right_class)) & (top2 == int(left_class)))
                        )
                route_weights = route_match.to(dtype=ambiguity_weight.dtype) * ambiguity_weight
            else:
                route_weights = ambiguity_weight
        return route_weights.to(device=like_logits.device, dtype=like_logits.dtype)

    def topk_reassessment_adjustment(
        self,
        reassessment_logits: Tensor,
        like_logits: Tensor,
        *,
        return_route_weights: bool = False,
    ):
        adjustment = like_logits.new_zeros(like_logits.shape)
        if (
            not torch.is_tensor(reassessment_logits)
            or reassessment_logits.ndim != 2
            or reassessment_logits.shape != like_logits.shape
        ):
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        scale = float(self.topk_reassessment_logit_scale)
        if scale <= 0.0:
            route_weights = like_logits.new_zeros((like_logits.size(0),))
            return (adjustment, route_weights) if return_route_weights else adjustment
        route_weights = self.topk_reassessment_route_weights(like_logits)
        adjustment = (
            reassessment_logits.to(dtype=like_logits.dtype)
            * scale
            * route_weights[:, None]
        )
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

    def cumulative_ordinal_logits_from_head_input(self, head_input: Tensor) -> Optional[Tensor]:
        if self.cumulative_ordinal_head is None:
            return None
        ordinal_input = self.cumulative_ordinal_norm(head_input)
        ordinal_input = self.cumulative_ordinal_dropout(ordinal_input)
        return self.cumulative_ordinal_head(ordinal_input)

    @staticmethod
    def _cumulative_ordinal_probs(threshold_logits: Tensor) -> Tensor:
        q = threshold_logits.float().sigmoid().clamp(min=1e-5, max=1.0 - 1e-5)
        class_probs: List[Tensor] = []
        survival = torch.ones((q.size(0),), device=q.device, dtype=q.dtype)
        for threshold_index in range(int(q.size(1))):
            threshold_prob = q[:, threshold_index]
            class_probs.append(survival * (1.0 - threshold_prob))
            survival = survival * threshold_prob
        class_probs.append(survival)
        probs = torch.stack(class_probs, dim=1)
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def cumulative_ordinal_adjustment(
        self,
        threshold_logits: Tensor,
        like_logits: Tensor,
    ) -> Tensor:
        adjustment = like_logits.new_zeros(like_logits.shape)
        if (
            not torch.is_tensor(threshold_logits)
            or threshold_logits.ndim != 2
            or len(self.cumulative_ordinal_classes) < 2
        ):
            return adjustment
        class_count = len(self.cumulative_ordinal_classes)
        if int(threshold_logits.size(1)) != class_count - 1:
            return adjustment
        scale = float(self.cumulative_ordinal_logit_scale)
        if scale <= 0.0:
            return adjustment

        probs = self._cumulative_ordinal_probs(threshold_logits)
        baseline_values = []
        survival = 1.0
        for _ in range(class_count - 1):
            baseline_values.append(survival * 0.5)
            survival *= 0.5
        baseline_values.append(survival)
        baseline = torch.tensor(
            baseline_values,
            device=like_logits.device,
            dtype=torch.float32,
        ).view(1, class_count)
        baseline = baseline / baseline.sum(dim=1, keepdim=True).clamp(min=1e-8)
        ordinal_logits = (
            probs.clamp(min=1e-6).log()
            - baseline.clamp(min=1e-6).log()
        ).to(dtype=like_logits.dtype)
        for rank, class_index in enumerate(self.cumulative_ordinal_classes):
            if 0 <= int(class_index) < int(like_logits.size(1)):
                adjustment[:, int(class_index)] = (
                    adjustment[:, int(class_index)]
                    + scale * ordinal_logits[:, rank]
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
        zero_init: bool = False,
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
        if bool(zero_init):
            nn.init.zeros_(self.pointwise.weight)
            if self.pointwise.bias is not None:
                nn.init.zeros_(self.pointwise.bias)

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
        stem_architecture: str = "conv_pool",
        stem_normalization: str = "batch",
        stem_pooling_mode: str = "max",
        stem_softpool_blend: float = 0.15,
        shifted_patch_tokenization: bool = False,
        shifted_patch_shift: int = 1,
        shifted_patch_residual_scale: float = 0.10,
        dual_patch_norm: bool = False,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        color_stat_fusion: bool = False,
        color_stat_fusion_dropout: float = 0.1,
        defect_stat_fusion: bool = False,
        defect_stat_fusion_dropout: float = 0.1,
        foreground_surface_fusion: bool = False,
        foreground_surface_fusion_dropout: float = 0.1,
        foreground_surface_pairwise_head: bool = False,
        foreground_surface_pairwise_pairs: str = "0-1,1-2,1-4,2-3",
        foreground_surface_pairwise_logit_scale: float = 0.18,
        foreground_surface_pairwise_dropout: float = 0.05,
        foreground_surface_pairwise_routing: bool = True,
        foreground_surface_pairwise_route_max_probability_margin: float = 0.22,
        interior_boundary_pairwise_head: bool = False,
        interior_boundary_pairwise_pairs: str = "0-1,4-1",
        interior_boundary_pairwise_logit_scale: float = 0.14,
        interior_boundary_pairwise_dropout: float = 0.05,
        interior_boundary_pairwise_hidden_dim: int = 128,
        interior_boundary_pairwise_erode_kernel: int = 9,
        interior_boundary_pairwise_routing: bool = True,
        interior_boundary_pairwise_route_max_probability_margin: float = 0.22,
        bbox_spatial_fusion: bool = False,
        bbox_spatial_fusion_hidden_dim: int = 64,
        bbox_spatial_fusion_dropout: float = 0.05,
        bbox_spatial_fusion_logit_scale: float = 0.20,
        patch_objectness_guided_head: bool = False,
        patch_objectness_hidden_dim: int = 128,
        patch_objectness_dropout: float = 0.05,
        patch_objectness_logit_scale: float = 0.12,
        patch_objectness_temperature: float = 0.75,
        bbox_prior_patch_context_head: bool = False,
        bbox_prior_patch_context_hidden_dim: int = 128,
        bbox_prior_patch_context_dropout: float = 0.05,
        bbox_prior_patch_context_logit_scale: float = 0.12,
        bbox_prior_patch_context_temperature: float = 0.50,
        source_context_feature_fusion: bool = False,
        source_context_fusion_hidden_dim: int = 128,
        source_context_fusion_dropout: float = 0.05,
        source_context_fusion_logit_scale: float = 0.20,
        source_context_fusion_gate_bias: float = -2.0,
        paired_view_feature_fusion: bool = False,
        paired_view_fusion_hidden_dim: int = 128,
        paired_view_fusion_dropout: float = 0.05,
        paired_view_fusion_logit_scale: float = 0.12,
        paired_view_fusion_gate_bias: float = -2.0,
        bilinear_patch_fusion: bool = False,
        bilinear_patch_rank: int = 32,
        bilinear_patch_dropout: float = 0.1,
        complementary_patch_suppression_head: bool = False,
        complementary_patch_suppression_top_k: int = 6,
        complementary_patch_suppression_hidden_dim: int = 128,
        complementary_patch_suppression_dropout: float = 0.05,
        complementary_patch_suppression_temperature: float = 0.20,
        complementary_patch_suppression_strength: float = 0.85,
        complementary_patch_suppression_bbox_weight: float = 0.35,
        complementary_patch_suppression_logit_scale: float = 0.12,
        micro_detail_patch_expert: bool = False,
        micro_detail_top_k: int = 8,
        micro_detail_hidden_dim: int = 128,
        micro_detail_dropout: float = 0.08,
        micro_detail_temperature: float = 0.12,
        micro_detail_foreground_power: float = 1.0,
        micro_detail_logit_scale: float = 0.18,
        micro_detail_routing: bool = True,
        micro_detail_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        micro_detail_route_max_probability_margin: float = 0.25,
        part_token_learner: bool = False,
        part_token_count: int = 4,
        part_token_hidden_dim: int = 128,
        part_token_dropout: float = 0.08,
        part_token_temperature: float = 0.70,
        part_token_foreground_power: float = 1.0,
        part_token_bbox_weight: float = 0.75,
        part_token_logit_scale: float = 0.16,
        part_token_routing: bool = True,
        part_token_route_pairs: str = "0-1,1-2,4-1,2-3",
        part_token_route_max_probability_margin: float = 0.25,
        part_token_pairwise_head: bool = False,
        part_token_pairwise_pairs: str = "0-1,1-2,4-1,2-3",
        part_token_pairwise_logit_scale: float = 0.18,
        part_token_pairwise_dropout: float = 0.08,
        part_token_pairwise_routing: bool = True,
        part_token_pairwise_route_max_probability_margin: float = 0.22,
        patch_evidence_router_head: bool = False,
        patch_evidence_router_pair: str = "0-1",
        patch_evidence_router_hidden_dim: int = 128,
        patch_evidence_router_top_k: int = 4,
        patch_evidence_router_bbox_weight: float = 0.75,
        patch_evidence_router_dropout: float = 0.05,
        patch_evidence_router_logit_scale: float = 0.12,
        patch_evidence_router_margin_prior_mode: str = "none",
        patch_evidence_router_margin_prior_scale: float = 0.0,
        patch_evidence_router_summary_stats: bool = False,
        patch_evidence_router_routing: bool = True,
        patch_evidence_router_route_max_probability_margin: float = 0.25,
        patch_evidence_router_route_min_pair_probability: float = 0.02,
        local_zoom_image_expert: bool = False,
        local_zoom_crop_size: int = 128,
        local_zoom_crop_scale: float = 0.48,
        local_zoom_score_mode: str = "foreground_detail",
        local_zoom_hidden_dim: int = 128,
        local_zoom_dropout: float = 0.08,
        local_zoom_logit_scale: float = 0.16,
        local_zoom_routing: bool = True,
        local_zoom_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        local_zoom_route_max_probability_margin: float = 0.25,
        high_frequency_texture_expert: bool = False,
        high_frequency_texture_hidden_dim: int = 128,
        high_frequency_texture_dropout: float = 0.08,
        high_frequency_texture_analysis_size: int = 96,
        high_frequency_texture_logit_scale: float = 0.16,
        high_frequency_texture_routing: bool = True,
        high_frequency_texture_route_pairs: str = "0-1,1-2,2-3,1-4,4-rest",
        high_frequency_texture_route_max_probability_margin: float = 0.25,
        multi_granularity_aux_heads: bool = False,
        multi_granularity_aux_layers: str = "2,5,8",
        multi_granularity_aux_dropout: float = 0.08,
        self_boosting_attention_head: bool = False,
        block_local_patch_mixer: bool = False,
        block_local_patch_mixer_layers: str = "6,7,8",
        block_local_patch_mixer_dropout: float = 0.0,
        block_local_patch_mixer_scale: float = 0.10,
        locally_enhanced_ffn: bool = False,
        locally_enhanced_ffn_layers: str = "1,2,3,4",
        locally_enhanced_ffn_kernel_size: int = 3,
        concurrent_local_global_coupling: bool = False,
        concurrent_local_global_layers: str = "1,2,3,4,5,6,7,8",
        concurrent_local_global_dim: int = 64,
        concurrent_local_global_kernel_size: int = 3,
        gated_relative_position_attention: bool = False,
        gated_relative_position_attention_layers: str = "1,2,3,4",
        gated_relative_position_attention_max_mix: float = 0.25,
        gated_relative_position_attention_locality_strength: float = 1.0,
        visual_contrast_attention: bool = False,
        visual_contrast_attention_layers: str = "1,2,3,4,5,6,7,8",
        visual_contrast_tokens: int = 64,
        foveal_aggregated_attention: bool = False,
        foveal_aggregated_attention_layers: str = "1",
        foveal_aggregated_attention_window_size: int = 3,
        foveal_aggregated_attention_pool_size: int = 4,
        deformable_spatial_attention: bool = False,
        deformable_spatial_attention_layers: str = "2",
        deformable_spatial_attention_groups: int = 2,
        deformable_spatial_attention_kernel_size: int = 5,
        deformable_spatial_attention_offset_range: float = 2.0,
        bi_level_routing_attention: bool = False,
        bi_level_routing_attention_layers: str = "2",
        bi_level_routing_attention_regions_per_axis: int = 4,
        bi_level_routing_attention_topk: int = 4,
        bi_level_routing_attention_local_context_kernel_size: int = 5,
        cross_covariance_attention: bool = False,
        cross_covariance_attention_layers: str = "2,5",
        cross_covariance_attention_residual_scale: float = 0.10,
        dynamic_graph_mixer: bool = False,
        dynamic_graph_mixer_layers: str = "2,5",
        dynamic_graph_mixer_bottleneck_dim: int = 64,
        dynamic_graph_mixer_k: int = 9,
        soft_moe_patch_adapter: bool = False,
        soft_moe_patch_adapter_layers: str = "2,5",
        soft_moe_hidden_dim: int = 64,
        soft_moe_num_experts: int = 4,
        soft_moe_residual_scale: float = 0.10,
        soft_moe_router_scale_init: float = 10.0,
        soft_moe_init_seed: int = 20260715,
        deep_class_prompt: bool = False,
        deep_class_prompt_logit_scale: float = 0.10,
        deep_class_prompt_init_seed: int = 20260715,
        patch_style_recalibration: bool = False,
        patch_style_recalibration_layers: str = "2,5",
        layer_token_fusion: bool = False,
        layer_token_fusion_layers: str = "2,4,6",
        layer_token_fusion_top_k: int = 4,
        layer_token_fusion_blend: float = 0.12,
        layer_token_fusion_attention_temperature: float = 0.20,
        layer_token_fusion_bbox_weight: float = 0.20,
        layer_token_fusion_foreground_weight: float = 0.10,
        masked_reconstruction_head: bool = False,
        masked_reconstruction_hidden_dim: int = 192,
        frequency_selective_pooling: bool = False,
        frequency_selective_top_k: int = 1,
        frequency_selective_blend: float = 1.0,
        frequency_selective_foreground_threshold: float = 0.35,
        patch_memory_adapter: bool = False,
        patch_memory_adapter_dropout: float = 0.0,
        late_class_attention_pooling: bool = False,
        late_class_attention_heads: int = 4,
        late_class_attention_dropout: float = 0.05,
        late_class_attention_mlp_ratio: float = 2.0,
        late_class_attention_residual_scale: float = 0.10,
        late_member_branch: bool = False,
        late_member_fork_after_block: int = 6,
        late_member_candidate_weight: float = 0.40,
        late_member_focus_class: int = 1,
        late_member_focus_margin_offset: float = 0.0,
        mixstyle: bool = False,
        mixstyle_probability: float = 0.5,
        mixstyle_alpha: float = 0.1,
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
        learnable_gabor_texture_residual: bool = False,
        learnable_gabor_texture_semantic_fusion: bool = False,
        detail_patch_enhancement: bool = False,
        detail_patch_dropout: float = 0.05,
        token_pruning: bool = False,
        token_prune_layers: str = "2,5",
        token_keep_rates: str = "0.75,0.50",
        token_prune_foreground_weight: float = 0.35,
        token_prune_bbox_weight: float = 0.0,
        token_prune_bbox_margin_ratio: float = 0.04,
        early_token_mask_keep_rate: float = 1.0,
        inattentive_token_fusion: bool = False,
        cropr_token_selector: bool = False,
        cropr_token_selector_routing: bool = True,
        pairwise_margin_head: bool = False,
        pairwise_margin_pairs: str = "0-1,2-3,4-rest",
        pairwise_margin_logit_scale: float = 0.35,
        pairwise_margin_dropout: float = 0.05,
        pairwise_margin_routing: bool = False,
        pairwise_margin_route_max_probability_margin: float = 0.20,
        topk_reassessment_head: bool = False,
        topk_reassessment_top_k: int = 2,
        topk_reassessment_hidden_dim: int = 128,
        topk_reassessment_dropout: float = 0.05,
        topk_reassessment_logit_scale: float = 0.15,
        topk_reassessment_routing: bool = True,
        topk_reassessment_route_pairs: str = "0-1,1-2,2-3,4-rest",
        topk_reassessment_route_max_probability_margin: float = 0.30,
        focus_class_head: bool = False,
        focus_class_index: int = 1,
        focus_class_logit_scale: float = 0.20,
        focus_class_dropout: float = 0.05,
        focus_class_routing: bool = True,
        focus_class_route_max_probability_margin: float = 0.35,
        focus_class_route_min_probability: float = 0.08,
        class_independent_head: bool = False,
        class_independent_dropout: float = 0.05,
        ordinal_maturity_head: bool = False,
        ordinal_maturity_classes: str = "0,1,2,3",
        ordinal_maturity_logit_scale: float = 0.20,
        ordinal_maturity_dropout: float = 0.05,
        cumulative_ordinal_head: bool = False,
        cumulative_ordinal_classes: str = "0,1,2,3",
        cumulative_ordinal_logit_scale: float = 0.25,
        cumulative_ordinal_dropout: float = 0.05,
        classification_mlp_head: bool = False,
        classification_mlp_hidden_dim: int = 512,
        classification_mlp_dropout: float = 0.08,
        classification_mlp_residual_scale: float = 0.20,
        subcenter_proxy_head: bool = False,
        subcenter_proxy_subcenters: int = 3,
        subcenter_proxy_dropout: float = 0.0,
        subcenter_proxy_init_std: float = 0.01,
        deep_abstention_head: bool = False,
        deep_abstention_dropout: float = 0.05,
        deep_abstention_initial_probability: float = 0.01,
        teacher_feature_projection_adapter: bool = False,
        teacher_feature_projection_teacher_dim: int = 0,
        teacher_feature_projection_dim: int = 128,
        teacher_feature_projection_dropout: float = 0.0,
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
            stem_architecture=stem_architecture,
            stem_normalization=stem_normalization,
            stem_pooling_mode=stem_pooling_mode,
            stem_softpool_blend=stem_softpool_blend,
            shifted_patch_tokenization=shifted_patch_tokenization,
            shifted_patch_shift=shifted_patch_shift,
            shifted_patch_residual_scale=shifted_patch_residual_scale,
            dual_patch_norm=dual_patch_norm,
            cnn_feature_fusion=cnn_feature_fusion,
            cnn_fusion_dropout=cnn_fusion_dropout,
            color_stat_fusion=color_stat_fusion,
            color_stat_fusion_dropout=color_stat_fusion_dropout,
            defect_stat_fusion=defect_stat_fusion,
            defect_stat_fusion_dropout=defect_stat_fusion_dropout,
            foreground_surface_fusion=foreground_surface_fusion,
            foreground_surface_fusion_dropout=foreground_surface_fusion_dropout,
            foreground_surface_pairwise_head=foreground_surface_pairwise_head,
            foreground_surface_pairwise_pairs=foreground_surface_pairwise_pairs,
            foreground_surface_pairwise_logit_scale=foreground_surface_pairwise_logit_scale,
            foreground_surface_pairwise_dropout=foreground_surface_pairwise_dropout,
            foreground_surface_pairwise_routing=foreground_surface_pairwise_routing,
            foreground_surface_pairwise_route_max_probability_margin=(
                foreground_surface_pairwise_route_max_probability_margin
            ),
            interior_boundary_pairwise_head=interior_boundary_pairwise_head,
            interior_boundary_pairwise_pairs=interior_boundary_pairwise_pairs,
            interior_boundary_pairwise_logit_scale=interior_boundary_pairwise_logit_scale,
            interior_boundary_pairwise_dropout=interior_boundary_pairwise_dropout,
            interior_boundary_pairwise_hidden_dim=interior_boundary_pairwise_hidden_dim,
            interior_boundary_pairwise_erode_kernel=interior_boundary_pairwise_erode_kernel,
            interior_boundary_pairwise_routing=interior_boundary_pairwise_routing,
            interior_boundary_pairwise_route_max_probability_margin=(
                interior_boundary_pairwise_route_max_probability_margin
            ),
            bbox_spatial_fusion=bbox_spatial_fusion,
            bbox_spatial_fusion_hidden_dim=bbox_spatial_fusion_hidden_dim,
            bbox_spatial_fusion_dropout=bbox_spatial_fusion_dropout,
            bbox_spatial_fusion_logit_scale=bbox_spatial_fusion_logit_scale,
            patch_objectness_guided_head=patch_objectness_guided_head,
            patch_objectness_hidden_dim=patch_objectness_hidden_dim,
            patch_objectness_dropout=patch_objectness_dropout,
            patch_objectness_logit_scale=patch_objectness_logit_scale,
            patch_objectness_temperature=patch_objectness_temperature,
            bbox_prior_patch_context_head=bbox_prior_patch_context_head,
            bbox_prior_patch_context_hidden_dim=bbox_prior_patch_context_hidden_dim,
            bbox_prior_patch_context_dropout=bbox_prior_patch_context_dropout,
            bbox_prior_patch_context_logit_scale=bbox_prior_patch_context_logit_scale,
            bbox_prior_patch_context_temperature=bbox_prior_patch_context_temperature,
            source_context_feature_fusion=source_context_feature_fusion,
            source_context_fusion_hidden_dim=source_context_fusion_hidden_dim,
            source_context_fusion_dropout=source_context_fusion_dropout,
            source_context_fusion_logit_scale=source_context_fusion_logit_scale,
            source_context_fusion_gate_bias=source_context_fusion_gate_bias,
            paired_view_feature_fusion=paired_view_feature_fusion,
            paired_view_fusion_hidden_dim=paired_view_fusion_hidden_dim,
            paired_view_fusion_dropout=paired_view_fusion_dropout,
            paired_view_fusion_logit_scale=paired_view_fusion_logit_scale,
            paired_view_fusion_gate_bias=paired_view_fusion_gate_bias,
            bilinear_patch_fusion=bilinear_patch_fusion,
            bilinear_patch_rank=bilinear_patch_rank,
            bilinear_patch_dropout=bilinear_patch_dropout,
            complementary_patch_suppression_head=complementary_patch_suppression_head,
            complementary_patch_suppression_top_k=complementary_patch_suppression_top_k,
            complementary_patch_suppression_hidden_dim=(
                complementary_patch_suppression_hidden_dim
            ),
            complementary_patch_suppression_dropout=complementary_patch_suppression_dropout,
            complementary_patch_suppression_temperature=(
                complementary_patch_suppression_temperature
            ),
            complementary_patch_suppression_strength=(
                complementary_patch_suppression_strength
            ),
            complementary_patch_suppression_bbox_weight=(
                complementary_patch_suppression_bbox_weight
            ),
            complementary_patch_suppression_logit_scale=(
                complementary_patch_suppression_logit_scale
            ),
            micro_detail_patch_expert=micro_detail_patch_expert,
            micro_detail_top_k=micro_detail_top_k,
            micro_detail_hidden_dim=micro_detail_hidden_dim,
            micro_detail_dropout=micro_detail_dropout,
            micro_detail_temperature=micro_detail_temperature,
            micro_detail_foreground_power=micro_detail_foreground_power,
            micro_detail_logit_scale=micro_detail_logit_scale,
            micro_detail_routing=micro_detail_routing,
            micro_detail_route_pairs=micro_detail_route_pairs,
            micro_detail_route_max_probability_margin=micro_detail_route_max_probability_margin,
            part_token_learner=part_token_learner,
            part_token_count=part_token_count,
            part_token_hidden_dim=part_token_hidden_dim,
            part_token_dropout=part_token_dropout,
            part_token_temperature=part_token_temperature,
            part_token_foreground_power=part_token_foreground_power,
            part_token_bbox_weight=part_token_bbox_weight,
            part_token_logit_scale=part_token_logit_scale,
            part_token_routing=part_token_routing,
            part_token_route_pairs=part_token_route_pairs,
            part_token_route_max_probability_margin=part_token_route_max_probability_margin,
            part_token_pairwise_head=part_token_pairwise_head,
            part_token_pairwise_pairs=part_token_pairwise_pairs,
            part_token_pairwise_logit_scale=part_token_pairwise_logit_scale,
            part_token_pairwise_dropout=part_token_pairwise_dropout,
            part_token_pairwise_routing=part_token_pairwise_routing,
            part_token_pairwise_route_max_probability_margin=(
                part_token_pairwise_route_max_probability_margin
            ),
            patch_evidence_router_head=patch_evidence_router_head,
            patch_evidence_router_pair=patch_evidence_router_pair,
            patch_evidence_router_hidden_dim=patch_evidence_router_hidden_dim,
            patch_evidence_router_top_k=patch_evidence_router_top_k,
            patch_evidence_router_bbox_weight=patch_evidence_router_bbox_weight,
            patch_evidence_router_dropout=patch_evidence_router_dropout,
            patch_evidence_router_logit_scale=patch_evidence_router_logit_scale,
            patch_evidence_router_margin_prior_mode=patch_evidence_router_margin_prior_mode,
            patch_evidence_router_margin_prior_scale=patch_evidence_router_margin_prior_scale,
            patch_evidence_router_summary_stats=patch_evidence_router_summary_stats,
            patch_evidence_router_routing=patch_evidence_router_routing,
            patch_evidence_router_route_max_probability_margin=(
                patch_evidence_router_route_max_probability_margin
            ),
            patch_evidence_router_route_min_pair_probability=(
                patch_evidence_router_route_min_pair_probability
            ),
            local_zoom_image_expert=local_zoom_image_expert,
            local_zoom_crop_size=local_zoom_crop_size,
            local_zoom_crop_scale=local_zoom_crop_scale,
            local_zoom_score_mode=local_zoom_score_mode,
            local_zoom_hidden_dim=local_zoom_hidden_dim,
            local_zoom_dropout=local_zoom_dropout,
            local_zoom_logit_scale=local_zoom_logit_scale,
            local_zoom_routing=local_zoom_routing,
            local_zoom_route_pairs=local_zoom_route_pairs,
            local_zoom_route_max_probability_margin=local_zoom_route_max_probability_margin,
            high_frequency_texture_expert=high_frequency_texture_expert,
            high_frequency_texture_hidden_dim=high_frequency_texture_hidden_dim,
            high_frequency_texture_dropout=high_frequency_texture_dropout,
            high_frequency_texture_analysis_size=high_frequency_texture_analysis_size,
            high_frequency_texture_logit_scale=high_frequency_texture_logit_scale,
            high_frequency_texture_routing=high_frequency_texture_routing,
            high_frequency_texture_route_pairs=high_frequency_texture_route_pairs,
            high_frequency_texture_route_max_probability_margin=(
                high_frequency_texture_route_max_probability_margin
            ),
            multi_granularity_aux_heads=multi_granularity_aux_heads,
            multi_granularity_aux_layers=multi_granularity_aux_layers,
            multi_granularity_aux_dropout=multi_granularity_aux_dropout,
            self_boosting_attention_head=self_boosting_attention_head,
            block_local_patch_mixer=block_local_patch_mixer,
            block_local_patch_mixer_layers=block_local_patch_mixer_layers,
            block_local_patch_mixer_dropout=block_local_patch_mixer_dropout,
            block_local_patch_mixer_scale=block_local_patch_mixer_scale,
            locally_enhanced_ffn=locally_enhanced_ffn,
            locally_enhanced_ffn_layers=locally_enhanced_ffn_layers,
            locally_enhanced_ffn_kernel_size=locally_enhanced_ffn_kernel_size,
            concurrent_local_global_coupling=concurrent_local_global_coupling,
            concurrent_local_global_layers=concurrent_local_global_layers,
            concurrent_local_global_dim=concurrent_local_global_dim,
            concurrent_local_global_kernel_size=(
                concurrent_local_global_kernel_size
            ),
            gated_relative_position_attention=gated_relative_position_attention,
            gated_relative_position_attention_layers=(
                gated_relative_position_attention_layers
            ),
            gated_relative_position_attention_max_mix=(
                gated_relative_position_attention_max_mix
            ),
            gated_relative_position_attention_locality_strength=(
                gated_relative_position_attention_locality_strength
            ),
            visual_contrast_attention=visual_contrast_attention,
            visual_contrast_attention_layers=visual_contrast_attention_layers,
            visual_contrast_tokens=visual_contrast_tokens,
            foveal_aggregated_attention=foveal_aggregated_attention,
            foveal_aggregated_attention_layers=(
                foveal_aggregated_attention_layers
            ),
            foveal_aggregated_attention_window_size=(
                foveal_aggregated_attention_window_size
            ),
            foveal_aggregated_attention_pool_size=(
                foveal_aggregated_attention_pool_size
            ),
            deformable_spatial_attention=deformable_spatial_attention,
            deformable_spatial_attention_layers=(
                deformable_spatial_attention_layers
            ),
            deformable_spatial_attention_groups=(
                deformable_spatial_attention_groups
            ),
            deformable_spatial_attention_kernel_size=(
                deformable_spatial_attention_kernel_size
            ),
            deformable_spatial_attention_offset_range=(
                deformable_spatial_attention_offset_range
            ),
            bi_level_routing_attention=bi_level_routing_attention,
            bi_level_routing_attention_layers=(
                bi_level_routing_attention_layers
            ),
            bi_level_routing_attention_regions_per_axis=(
                bi_level_routing_attention_regions_per_axis
            ),
            bi_level_routing_attention_topk=(
                bi_level_routing_attention_topk
            ),
            bi_level_routing_attention_local_context_kernel_size=(
                bi_level_routing_attention_local_context_kernel_size
            ),
            cross_covariance_attention=cross_covariance_attention,
            cross_covariance_attention_layers=cross_covariance_attention_layers,
            cross_covariance_attention_residual_scale=(
                cross_covariance_attention_residual_scale
            ),
            dynamic_graph_mixer=dynamic_graph_mixer,
            dynamic_graph_mixer_layers=dynamic_graph_mixer_layers,
            dynamic_graph_mixer_bottleneck_dim=(
                dynamic_graph_mixer_bottleneck_dim
            ),
            dynamic_graph_mixer_k=dynamic_graph_mixer_k,
            soft_moe_patch_adapter=soft_moe_patch_adapter,
            soft_moe_patch_adapter_layers=soft_moe_patch_adapter_layers,
            soft_moe_hidden_dim=soft_moe_hidden_dim,
            soft_moe_num_experts=soft_moe_num_experts,
            soft_moe_residual_scale=soft_moe_residual_scale,
            soft_moe_router_scale_init=soft_moe_router_scale_init,
            soft_moe_init_seed=soft_moe_init_seed,
            deep_class_prompt=deep_class_prompt,
            deep_class_prompt_logit_scale=deep_class_prompt_logit_scale,
            deep_class_prompt_init_seed=deep_class_prompt_init_seed,
            patch_style_recalibration=patch_style_recalibration,
            patch_style_recalibration_layers=patch_style_recalibration_layers,
            layer_token_fusion=layer_token_fusion,
            layer_token_fusion_layers=layer_token_fusion_layers,
            layer_token_fusion_top_k=layer_token_fusion_top_k,
            layer_token_fusion_blend=layer_token_fusion_blend,
            layer_token_fusion_attention_temperature=(
                layer_token_fusion_attention_temperature
            ),
            layer_token_fusion_bbox_weight=layer_token_fusion_bbox_weight,
            layer_token_fusion_foreground_weight=layer_token_fusion_foreground_weight,
            masked_reconstruction_head=masked_reconstruction_head,
            masked_reconstruction_hidden_dim=masked_reconstruction_hidden_dim,
            frequency_selective_pooling=frequency_selective_pooling,
            frequency_selective_top_k=frequency_selective_top_k,
            frequency_selective_blend=frequency_selective_blend,
            frequency_selective_foreground_threshold=frequency_selective_foreground_threshold,
            patch_memory_adapter=patch_memory_adapter,
            patch_memory_adapter_dropout=patch_memory_adapter_dropout,
            late_class_attention_pooling=late_class_attention_pooling,
            late_class_attention_heads=late_class_attention_heads,
            late_class_attention_dropout=late_class_attention_dropout,
            late_class_attention_mlp_ratio=late_class_attention_mlp_ratio,
            late_class_attention_residual_scale=late_class_attention_residual_scale,
            late_member_branch=late_member_branch,
            late_member_fork_after_block=late_member_fork_after_block,
            late_member_candidate_weight=late_member_candidate_weight,
            late_member_focus_class=late_member_focus_class,
            late_member_focus_margin_offset=late_member_focus_margin_offset,
            mixstyle=mixstyle,
            mixstyle_probability=mixstyle_probability,
            mixstyle_alpha=mixstyle_alpha,
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
            learnable_gabor_texture_residual=learnable_gabor_texture_residual,
            learnable_gabor_texture_semantic_fusion=(
                learnable_gabor_texture_semantic_fusion
            ),
            detail_patch_enhancement=detail_patch_enhancement,
            detail_patch_dropout=detail_patch_dropout,
            token_pruning=token_pruning,
            token_prune_layers=token_prune_layers,
            token_keep_rates=token_keep_rates,
            token_prune_foreground_weight=token_prune_foreground_weight,
            token_prune_bbox_weight=token_prune_bbox_weight,
            token_prune_bbox_margin_ratio=token_prune_bbox_margin_ratio,
            early_token_mask_keep_rate=early_token_mask_keep_rate,
            inattentive_token_fusion=inattentive_token_fusion,
            cropr_token_selector=cropr_token_selector,
            cropr_token_selector_routing=cropr_token_selector_routing,
            pairwise_margin_head=pairwise_margin_head,
            pairwise_margin_pairs=pairwise_margin_pairs,
            pairwise_margin_logit_scale=pairwise_margin_logit_scale,
            pairwise_margin_dropout=pairwise_margin_dropout,
            pairwise_margin_routing=pairwise_margin_routing,
            pairwise_margin_route_max_probability_margin=(
                pairwise_margin_route_max_probability_margin
            ),
            topk_reassessment_head=topk_reassessment_head,
            topk_reassessment_top_k=topk_reassessment_top_k,
            topk_reassessment_hidden_dim=topk_reassessment_hidden_dim,
            topk_reassessment_dropout=topk_reassessment_dropout,
            topk_reassessment_logit_scale=topk_reassessment_logit_scale,
            topk_reassessment_routing=topk_reassessment_routing,
            topk_reassessment_route_pairs=topk_reassessment_route_pairs,
            topk_reassessment_route_max_probability_margin=(
                topk_reassessment_route_max_probability_margin
            ),
            focus_class_head=focus_class_head,
            focus_class_index=focus_class_index,
            focus_class_logit_scale=focus_class_logit_scale,
            focus_class_dropout=focus_class_dropout,
            focus_class_routing=focus_class_routing,
            focus_class_route_max_probability_margin=focus_class_route_max_probability_margin,
            focus_class_route_min_probability=focus_class_route_min_probability,
            class_independent_head=class_independent_head,
            class_independent_dropout=class_independent_dropout,
            ordinal_maturity_head=ordinal_maturity_head,
            ordinal_maturity_classes=ordinal_maturity_classes,
            ordinal_maturity_logit_scale=ordinal_maturity_logit_scale,
            ordinal_maturity_dropout=ordinal_maturity_dropout,
            cumulative_ordinal_head=cumulative_ordinal_head,
            cumulative_ordinal_classes=cumulative_ordinal_classes,
            cumulative_ordinal_logit_scale=cumulative_ordinal_logit_scale,
            cumulative_ordinal_dropout=cumulative_ordinal_dropout,
            classification_mlp_head=classification_mlp_head,
            classification_mlp_hidden_dim=classification_mlp_hidden_dim,
            classification_mlp_dropout=classification_mlp_dropout,
            classification_mlp_residual_scale=classification_mlp_residual_scale,
            subcenter_proxy_head=subcenter_proxy_head,
            subcenter_proxy_subcenters=subcenter_proxy_subcenters,
            subcenter_proxy_dropout=subcenter_proxy_dropout,
            subcenter_proxy_init_std=subcenter_proxy_init_std,
            deep_abstention_head=deep_abstention_head,
            deep_abstention_dropout=deep_abstention_dropout,
            deep_abstention_initial_probability=deep_abstention_initial_probability,
            teacher_feature_projection_adapter=teacher_feature_projection_adapter,
            teacher_feature_projection_teacher_dim=teacher_feature_projection_teacher_dim,
            teacher_feature_projection_dim=teacher_feature_projection_dim,
            teacher_feature_projection_dropout=teacher_feature_projection_dropout,
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

    def forward(
        self,
        x: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        bbox_token_prior: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        features = self.forward_features(
            x,
            image_valid_mask=image_valid_mask,
            bbox_token_prior=bbox_token_prior,
        )
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
    deep_class_prompt_logits = features.get("deep_class_prompt_logits")
    if torch.is_tensor(deep_class_prompt_logits):
        if tuple(deep_class_prompt_logits.shape) != tuple(logits.shape):
            raise RuntimeError(
                "Deep class-prompt logits must match the base classification logits."
            )
        deep_class_prompt_scale = float(
            getattr(model, "deep_class_prompt_logit_scale", 0.0)
        )
        deep_class_prompt_adjustment = deep_class_prompt_logits.to(
            dtype=logits.dtype
        ) * deep_class_prompt_scale
        features["deep_class_prompt_adjustment"] = deep_class_prompt_adjustment
        if isinstance(features.get("trace"), dict):
            features["trace"]["deep_class_prompt_adjustment"] = (
                deep_class_prompt_adjustment.detach()
            )
        logits = logits + deep_class_prompt_adjustment
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
    bbox_spatial_head = getattr(model, "bbox_spatial_fusion_head", None)
    bbox_value = features.get("bbox")
    if bbox_spatial_head is not None and torch.is_tensor(bbox_value):
        return_trace = isinstance(features.get("trace"), dict)
        bbox_output = bbox_spatial_head(
            bbox_value,
            dtype=logits.dtype,
            return_trace=return_trace,
        )
        if return_trace:
            bbox_spatial_logits, bbox_trace = bbox_output
            features["trace"]["bbox_spatial_stats"] = bbox_trace["stats"]
            features["trace"]["bbox_spatial_logits"] = bbox_spatial_logits.detach()
        else:
            bbox_spatial_logits = bbox_output
        features["bbox_spatial_logits"] = bbox_spatial_logits
        scale = float(getattr(model, "bbox_spatial_fusion_logit_scale", 1.0))
        if scale > 0.0:
            logits = logits + bbox_spatial_logits.to(dtype=logits.dtype) * scale
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
    complementary_head = getattr(model, "complementary_patch_suppression_head", None)
    if complementary_head is not None and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        complementary_output = complementary_head(
            head_input,
            features["patches"],
            bbox_prior=features.get("patch_bbox_prior"),
            foreground_prior=features.get("patch_foreground_prior"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            complementary_logits, complementary_trace = complementary_output
            features["trace"]["complementary_patch_saliency"] = complementary_trace[
                "saliency"
            ]
            features["trace"]["complementary_patch_suppressed_mask"] = complementary_trace[
                "suppressed_mask"
            ]
            features["trace"]["complementary_patch_suppressed_indices"] = complementary_trace[
                "suppressed_indices"
            ]
            features["trace"]["complementary_patch_attention"] = complementary_trace[
                "complement_attention"
            ]
            features["trace"]["complementary_patch_descriptor"] = complementary_trace[
                "descriptor"
            ]
        else:
            complementary_logits = complementary_output
        features["complementary_patch_suppression_logits"] = complementary_logits
        scale = float(getattr(model, "complementary_patch_suppression_logit_scale", 1.0))
        if scale > 0.0:
            logits = logits + complementary_logits.to(dtype=logits.dtype) * scale
    patch_objectness_head = getattr(model, "patch_objectness_guided_head", None)
    if patch_objectness_head is not None and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        trace_enabled = isinstance(features.get("trace"), dict)
        return_trace = trace_enabled or bool(getattr(model, "training", False))
        patch_objectness_output = patch_objectness_head(
            features["patches"],
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            patch_objectness_class_logits, patch_objectness_trace = patch_objectness_output
            features["patch_objectness_logits"] = patch_objectness_trace["objectness_logits"]
            features["patch_objectness_probability"] = patch_objectness_trace[
                "objectness_probability"
            ]
            if trace_enabled:
                features["trace"]["patch_objectness_attention"] = patch_objectness_trace[
                    "attention"
                ]
                features["trace"]["patch_objectness_logits"] = patch_objectness_trace[
                    "objectness_logits"
                ]
                features["trace"]["patch_objectness_probability"] = patch_objectness_trace[
                    "objectness_probability"
                ]
                features["trace"]["patch_objectness_descriptor"] = patch_objectness_trace[
                    "descriptor"
                ]
                features["trace"]["patch_objectness_stats"] = patch_objectness_trace["stats"]
        else:
            patch_objectness_class_logits = patch_objectness_output
        features["patch_objectness_class_logits"] = patch_objectness_class_logits
        scale = float(getattr(model, "patch_objectness_logit_scale", 1.0))
        if scale > 0.0:
            logits = logits + patch_objectness_class_logits.to(dtype=logits.dtype) * scale
    bbox_context_head = getattr(model, "bbox_prior_patch_context_head", None)
    if bbox_context_head is not None and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        bbox_context_output = bbox_context_head(
            features["patches"],
            bbox_prior=features.get("patch_bbox_prior"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            bbox_context_logits, bbox_context_trace = bbox_context_output
            features["trace"]["bbox_prior_patch_object_attention"] = bbox_context_trace[
                "object_attention"
            ]
            features["trace"]["bbox_prior_patch_background_attention"] = bbox_context_trace[
                "background_attention"
            ]
            features["trace"]["bbox_prior_patch_descriptor"] = bbox_context_trace[
                "descriptor"
            ]
            features["trace"]["bbox_prior_patch_stats"] = bbox_context_trace["stats"]
        else:
            bbox_context_logits = bbox_context_output
        features["bbox_prior_patch_context_logits"] = bbox_context_logits
        scale = float(getattr(model, "bbox_prior_patch_context_logit_scale", 1.0))
        if scale > 0.0:
            logits = logits + bbox_context_logits.to(dtype=logits.dtype) * scale
    micro_detail_head = getattr(model, "micro_detail_patch_expert", None)
    micro_detail_adjust_fn = getattr(model, "micro_detail_adjustment", None)
    if micro_detail_head is not None and callable(micro_detail_adjust_fn) and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        micro_output = micro_detail_head(
            features["patches"],
            detail_map=features.get("detail_map"),
            patch_indices=features.get("patch_indices"),
            foreground_prior=features.get("patch_foreground_prior"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            micro_logits, micro_trace = micro_output
            features["trace"]["micro_detail_attention"] = micro_trace["attention"]
            features["trace"]["micro_detail_descriptor"] = micro_trace["descriptor"]
            features["trace"]["micro_detail_selected_indices"] = micro_trace["selected_indices"]
            micro_adjustment, micro_route_weights = micro_detail_adjust_fn(
                micro_logits,
                logits,
                return_route_weights=True,
            )
            features["trace"]["micro_detail_route_weights"] = micro_route_weights
        else:
            micro_logits = micro_output
            micro_adjustment = micro_detail_adjust_fn(micro_logits, logits)
        features["micro_detail_logits"] = micro_logits
        logits = logits + micro_adjustment
    part_token_head = getattr(model, "part_token_learner", None)
    part_token_adjust_fn = getattr(model, "part_token_adjustment", None)
    if part_token_head is not None and callable(part_token_adjust_fn) and "patches" in features:
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        part_output = part_token_head(
            features["patches"],
            foreground_prior=features.get("patch_foreground_prior"),
            bbox_prior=features.get("patch_bbox_prior"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            part_logits, part_trace = part_output
            features["trace"]["part_token_attention"] = part_trace["attention"]
            features["trace"]["part_token_descriptor"] = part_trace["descriptor"]
            features["trace"]["part_token_foreground_mass"] = part_trace["foreground_mass"]
            features["trace"]["part_token_max_weight"] = part_trace["max_weight"]
            part_adjustment, part_route_weights = part_token_adjust_fn(
                part_logits,
                logits,
                return_route_weights=True,
            )
            features["trace"]["part_token_route_weights"] = part_route_weights
        else:
            part_logits = part_output
            part_adjustment = part_token_adjust_fn(part_logits, logits)
        features["part_token_logits"] = part_logits
        logits = logits + part_adjustment
    part_token_pairwise_head = getattr(model, "part_token_pairwise_learner", None)
    part_token_pairwise_adjust_fn = getattr(model, "part_token_pairwise_adjustment", None)
    if (
        part_token_pairwise_head is not None
        and callable(part_token_pairwise_adjust_fn)
        and "patches" in features
    ):
        key_padding_mask = features.get("memory_key_padding_mask")
        valid_mask = (
            ~key_padding_mask.to(dtype=torch.bool)
            if torch.is_tensor(key_padding_mask)
            else None
        )
        return_trace = isinstance(features.get("trace"), dict)
        pairwise_output = part_token_pairwise_head(
            features["patches"],
            foreground_prior=features.get("patch_foreground_prior"),
            bbox_prior=features.get("patch_bbox_prior"),
            valid_mask=valid_mask,
            return_trace=return_trace,
        )
        if return_trace:
            part_pairwise_logits, part_pairwise_trace = pairwise_output
            features["trace"]["part_token_pairwise_attention"] = part_pairwise_trace["attention"]
            features["trace"]["part_token_pairwise_descriptor"] = part_pairwise_trace[
                "descriptor"
            ]
            features["trace"]["part_token_pairwise_foreground_mass"] = part_pairwise_trace[
                "foreground_mass"
            ]
            features["trace"]["part_token_pairwise_max_weight"] = part_pairwise_trace[
                "max_weight"
            ]
            part_pairwise_adjustment, part_pairwise_route_weights = (
                part_token_pairwise_adjust_fn(
                    part_pairwise_logits,
                    logits,
                    return_route_weights=True,
                )
            )
            features["trace"]["part_token_pairwise_route_weights"] = (
                part_pairwise_route_weights
            )
        else:
            part_pairwise_logits = pairwise_output
            part_pairwise_adjustment = part_token_pairwise_adjust_fn(
                part_pairwise_logits,
                logits,
            )
        features["part_token_pairwise_logits"] = part_pairwise_logits
        logits = logits + part_pairwise_adjustment
    patch_router_head = getattr(model, "patch_evidence_router_head", None)
    patch_router_adjust_fn = getattr(model, "patch_evidence_router_adjustment", None)
    patch_router_pairs = list(getattr(model, "patch_evidence_router_pairs", []))
    if (
        patch_router_head is not None
        and callable(patch_router_adjust_fn)
        and patch_router_pairs
        and "patches" in features
    ):
        left_class, right_class = patch_router_pairs[0]
        patch_logits = model.head(features["patches"])
        if (
            torch.is_tensor(patch_logits)
            and patch_logits.ndim == 3
            and 0 <= int(left_class) < int(patch_logits.size(-1))
            and 0 <= int(right_class) < int(patch_logits.size(-1))
        ):
            patch_pair_margin = (
                patch_logits[..., int(right_class)] - patch_logits[..., int(left_class)]
            )
            base_pair_stats = None
            if bool(getattr(patch_router_head, "summary_stats", False)):
                probabilities = logits.float().softmax(dim=1)
                if (
                    0 <= int(left_class) < int(probabilities.size(1))
                    and 0 <= int(right_class) < int(probabilities.size(1))
                ):
                    top_probabilities = probabilities.topk(k=2, dim=1).values
                    pair_left = probabilities[:, int(left_class)]
                    pair_right = probabilities[:, int(right_class)]
                    base_pair_stats = torch.stack(
                        (
                            pair_left,
                            pair_right,
                            pair_left + pair_right,
                            top_probabilities[:, 0],
                            top_probabilities[:, 0] - top_probabilities[:, 1],
                        ),
                        dim=1,
                    ).to(dtype=logits.dtype)
            key_padding_mask = features.get("memory_key_padding_mask")
            valid_mask = (
                ~key_padding_mask.to(dtype=torch.bool)
                if torch.is_tensor(key_padding_mask)
                else None
            )
            return_trace = isinstance(features.get("trace"), dict)
            router_output = patch_router_head(
                features["patches"],
                head_input,
                patch_pair_margin,
                bbox_prior=features.get("patch_bbox_prior"),
                valid_mask=valid_mask,
                base_pair_stats=base_pair_stats,
                return_trace=return_trace,
            )
            if return_trace:
                router_logits_1d, router_trace = router_output
                features["trace"]["patch_evidence_router_attention"] = router_trace[
                    "attention"
                ]
                features["trace"]["patch_evidence_router_descriptor"] = router_trace[
                    "descriptor"
                ]
                features["trace"]["patch_evidence_router_weighted_margin"] = router_trace[
                    "weighted_margin"
                ]
                features["trace"]["patch_evidence_router_max_margin"] = router_trace[
                    "max_margin"
                ]
                features["trace"]["patch_evidence_router_mean_margin"] = router_trace[
                    "mean_margin"
                ]
                features["trace"]["patch_evidence_router_topk_mean_margin"] = router_trace[
                    "topk_mean_margin"
                ]
                if "summary_stats" in router_trace:
                    features["trace"]["patch_evidence_router_summary_stats"] = (
                        router_trace["summary_stats"]
                    )
                features["trace"]["patch_evidence_router_margin_prior"] = router_trace[
                    "margin_prior"
                ]
                features["trace"]["patch_evidence_router_selected_indices"] = router_trace[
                    "selected_indices"
                ]
                if "bbox_mass" in router_trace:
                    features["trace"]["patch_evidence_router_bbox_mass"] = router_trace[
                        "bbox_mass"
                    ]
            else:
                router_logits_1d = router_output
            router_logits = router_logits_1d.unsqueeze(1)
            if return_trace:
                router_adjustment, router_route_weights = patch_router_adjust_fn(
                    router_logits,
                    logits,
                    return_route_weights=True,
                )
                features["trace"]["patch_evidence_router_route_weights"] = (
                    router_route_weights
                )
                features["trace"]["patch_evidence_router_logits"] = router_logits.detach()
            else:
                router_adjustment = patch_router_adjust_fn(router_logits, logits)
            features["patch_evidence_router_logits"] = router_logits
            logits = logits + router_adjustment
    local_zoom_logits = features.get("local_zoom_logits")
    local_zoom_adjust_fn = getattr(model, "local_zoom_adjustment", None)
    if torch.is_tensor(local_zoom_logits) and callable(local_zoom_adjust_fn):
        if isinstance(features.get("trace"), dict):
            local_zoom_adjustment, local_zoom_route_weights = local_zoom_adjust_fn(
                local_zoom_logits,
                logits,
                return_route_weights=True,
            )
            features["trace"]["local_zoom_logits"] = local_zoom_logits.detach()
            features["trace"]["local_zoom_route_weights"] = local_zoom_route_weights
        else:
            local_zoom_adjustment = local_zoom_adjust_fn(local_zoom_logits, logits)
        logits = logits + local_zoom_adjustment
    high_frequency_texture_logits = features.get("high_frequency_texture_logits")
    high_frequency_adjust_fn = getattr(model, "high_frequency_texture_adjustment", None)
    if torch.is_tensor(high_frequency_texture_logits) and callable(high_frequency_adjust_fn):
        if isinstance(features.get("trace"), dict):
            high_frequency_adjustment, high_frequency_route_weights = (
                high_frequency_adjust_fn(
                    high_frequency_texture_logits,
                    logits,
                    return_route_weights=True,
                )
            )
            features["trace"]["high_frequency_texture_logits"] = (
                high_frequency_texture_logits.detach()
            )
            features["trace"]["high_frequency_texture_route_weights"] = (
                high_frequency_route_weights
            )
        else:
            high_frequency_adjustment = high_frequency_adjust_fn(
                high_frequency_texture_logits,
                logits,
            )
        logits = logits + high_frequency_adjustment
    surface_pairwise_logits = features.get("foreground_surface_pairwise_logits")
    surface_pairwise_adjust_fn = getattr(
        model,
        "foreground_surface_pairwise_adjustment",
        None,
    )
    if torch.is_tensor(surface_pairwise_logits) and callable(surface_pairwise_adjust_fn):
        if isinstance(features.get("trace"), dict):
            surface_pairwise_adjustment, surface_pairwise_route_weights = (
                surface_pairwise_adjust_fn(
                    surface_pairwise_logits,
                    logits,
                    return_route_weights=True,
                )
            )
            features["trace"]["foreground_surface_pairwise_route_weights"] = (
                surface_pairwise_route_weights
            )
        else:
            surface_pairwise_adjustment = surface_pairwise_adjust_fn(
                surface_pairwise_logits,
                logits,
            )
        logits = logits + surface_pairwise_adjustment
    interior_boundary_pairwise_logits = features.get("interior_boundary_pairwise_logits")
    interior_boundary_pairwise_adjust_fn = getattr(
        model,
        "interior_boundary_pairwise_adjustment",
        None,
    )
    if (
        torch.is_tensor(interior_boundary_pairwise_logits)
        and callable(interior_boundary_pairwise_adjust_fn)
    ):
        if isinstance(features.get("trace"), dict):
            (
                interior_boundary_pairwise_adjustment,
                interior_boundary_pairwise_route_weights,
            ) = interior_boundary_pairwise_adjust_fn(
                interior_boundary_pairwise_logits,
                logits,
                return_route_weights=True,
            )
            features["trace"]["interior_boundary_pairwise_logits"] = (
                interior_boundary_pairwise_logits.detach()
            )
            features["trace"]["interior_boundary_pairwise_route_weights"] = (
                interior_boundary_pairwise_route_weights
            )
        else:
            interior_boundary_pairwise_adjustment = interior_boundary_pairwise_adjust_fn(
                interior_boundary_pairwise_logits,
                logits,
            )
        logits = logits + interior_boundary_pairwise_adjustment
    focus_class_fn = getattr(model, "focus_class_logit_from_head_input", None)
    focus_adjust_fn = getattr(model, "focus_class_adjustment", None)
    if callable(focus_class_fn) and callable(focus_adjust_fn):
        focus_logits = focus_class_fn(head_input)
        if torch.is_tensor(focus_logits):
            features["focus_class_logits"] = focus_logits
            if isinstance(features.get("trace"), dict):
                focus_adjustment, focus_route_weights = focus_adjust_fn(
                    focus_logits,
                    logits,
                    return_route_weights=True,
                )
                features["trace"]["focus_class_logits"] = focus_logits.detach()
                features["trace"]["focus_class_route_weights"] = focus_route_weights
            else:
                focus_adjustment = focus_adjust_fn(focus_logits, logits)
            logits = logits + focus_adjustment
    class_independent_fn = getattr(model, "class_independent_logits_from_head_input", None)
    if callable(class_independent_fn):
        class_independent_logits = class_independent_fn(head_input)
        if torch.is_tensor(class_independent_logits):
            features["class_independent_logits"] = class_independent_logits
            if isinstance(features.get("trace"), dict):
                features["trace"]["class_independent_logits"] = (
                    class_independent_logits.detach()
                )
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
    reassess_fn = getattr(model, "topk_reassessment_logits_from_head_input", None)
    reassess_adjust_fn = getattr(model, "topk_reassessment_adjustment", None)
    if callable(reassess_fn) and callable(reassess_adjust_fn):
        reassessment_base_logits = logits
        reassessment_logits = reassess_fn(head_input, logits)
        if torch.is_tensor(reassessment_logits):
            features["topk_reassessment_logits"] = reassessment_logits
            features["topk_reassessment_base_logits"] = reassessment_base_logits
            if isinstance(features.get("trace"), dict):
                reassessment_adjustment, reassessment_route_weights = reassess_adjust_fn(
                    reassessment_logits,
                    reassessment_base_logits,
                    return_route_weights=True,
                )
                features["trace"]["topk_reassessment_logits"] = reassessment_logits.detach()
                features["trace"]["topk_reassessment_route_weights"] = (
                    reassessment_route_weights
                )
                features["trace"]["topk_reassessment_adjustment"] = (
                    reassessment_adjustment.detach()
                )
            else:
                reassessment_adjustment, reassessment_route_weights = reassess_adjust_fn(
                    reassessment_logits,
                    reassessment_base_logits,
                    return_route_weights=True,
                )
            features["topk_reassessment_route_weights"] = reassessment_route_weights
            features["topk_reassessment_adjustment"] = reassessment_adjustment
            logits = logits + reassessment_adjustment
    ordinal_fn = getattr(model, "ordinal_maturity_score_from_head_input", None)
    ordinal_adjust_fn = getattr(model, "ordinal_maturity_adjustment", None)
    if callable(ordinal_fn) and callable(ordinal_adjust_fn):
        maturity_score = ordinal_fn(head_input)
        if torch.is_tensor(maturity_score):
            features["ordinal_maturity_score"] = maturity_score
            logits = logits + ordinal_adjust_fn(maturity_score, logits)
    cumulative_fn = getattr(model, "cumulative_ordinal_logits_from_head_input", None)
    cumulative_adjust_fn = getattr(model, "cumulative_ordinal_adjustment", None)
    if callable(cumulative_fn) and callable(cumulative_adjust_fn):
        cumulative_logits = cumulative_fn(head_input)
        if torch.is_tensor(cumulative_logits):
            features["cumulative_ordinal_logits"] = cumulative_logits
            if isinstance(features.get("trace"), dict):
                features["trace"]["cumulative_ordinal_logits"] = cumulative_logits.detach()
            logits = logits + cumulative_adjust_fn(cumulative_logits, logits)
    late_member_fn = getattr(model, "late_member_logits_from_features", None)
    late_member_fuse_fn = getattr(model, "fuse_late_member_logits", None)
    if bool(getattr(model, "late_member_enabled", False)):
        if not callable(late_member_fn) or not callable(late_member_fuse_fn):
            raise RuntimeError("Late-member model is missing its readout or fusion method.")
        primary_logits = logits
        candidate_logits = late_member_fn(features)
        features["late_member_primary_logits"] = primary_logits
        features["late_member_logits"] = candidate_logits
        logits = late_member_fuse_fn(primary_logits, candidate_logits)
        features["late_member_fused_logits"] = logits
    linear_verifier = getattr(model, "patch_evidence_linear_verifier", None)
    patches = features.get("patches")
    if linear_verifier is not None and torch.is_tensor(patches) and patches.ndim == 3:
        batch_size, token_count, embed_dim = patches.shape
        patch_logits = model.head(patches.reshape(batch_size * token_count, embed_dim)).reshape(
            batch_size,
            token_count,
            -1,
        )
        return_trace = isinstance(features.get("trace"), dict)
        feature_vector = linear_verifier.build_feature_vector(
            head_input=head_input,
            logits=logits,
            local_logits=patch_logits,
            key_padding_mask=features.get("memory_key_padding_mask"),
            bbox_prior=features.get("patch_bbox_prior"),
        )
        verifier_output = linear_verifier(
            logits,
            feature_vector,
            return_trace=return_trace,
        )
        if return_trace:
            logits, verifier_trace = verifier_output
            features["trace"]["patch_evidence_linear_verifier_probability"] = (
                verifier_trace["probability"]
            )
            features["trace"]["patch_evidence_linear_verifier_confidence"] = (
                verifier_trace["confidence"]
            )
            features["trace"]["patch_evidence_linear_verifier_candidate"] = (
                verifier_trace["candidate"]
            )
            features["trace"]["patch_evidence_linear_verifier_route_mask"] = (
                verifier_trace["route_mask"]
            )
            if "soft_gate" in verifier_trace:
                features["trace"]["patch_evidence_linear_verifier_soft_gate"] = (
                    verifier_trace["soft_gate"]
                )
            if "soft_delta" in verifier_trace:
                features["trace"]["patch_evidence_linear_verifier_soft_delta"] = (
                    verifier_trace["soft_delta"]
                )
            features["trace"]["patch_evidence_linear_verifier_feature_vector"] = (
                verifier_trace["feature_vector"]
            )
        else:
            logits = verifier_output
    abstention_fn = getattr(model, "deep_abstention_logit_from_head_input", None)
    if callable(abstention_fn):
        abstention_logit = abstention_fn(head_input, class_logits=logits)
        if torch.is_tensor(abstention_logit):
            features["deep_abstention_logit"] = abstention_logit
            if isinstance(features.get("trace"), dict):
                features["trace"]["deep_abstention_logit"] = abstention_logit.detach()
    return logits


def source_context_fused_logits_from_features(
    model: nn.Module,
    primary_features: Dict[str, Tensor],
    context_features: Dict[str, Tensor],
    primary_logits: Tensor,
    context_logits: Tensor,
    *,
    return_trace: bool = False,
):
    fusion_head = getattr(model, "source_context_fusion_head", None)
    if fusion_head is None:
        return None
    primary_head_input = extract_head_input_from_features(model, primary_features)
    context_head_input = extract_head_input_from_features(model, context_features)
    fusion_output = fusion_head(
        primary_head_input,
        context_head_input,
        primary_logits,
        context_logits,
        return_trace=return_trace,
    )
    if return_trace:
        raw_adjustment, trace = fusion_output
    else:
        raw_adjustment = fusion_output
        trace = None
    scale = float(getattr(model, "source_context_fusion_logit_scale", 1.0))
    adjustment = raw_adjustment.to(dtype=primary_logits.dtype) * scale
    fused_logits = primary_logits + adjustment
    primary_features["source_context_logits"] = context_logits
    primary_features["source_context_fusion_adjustment"] = adjustment
    if isinstance(primary_features.get("trace"), dict) and trace is not None:
        primary_features["trace"]["source_context_fusion_delta"] = trace["delta"]
        primary_features["trace"]["source_context_fusion_gate"] = trace["gate"]
        primary_features["trace"]["source_context_fusion_adjustment"] = adjustment.detach()
    if return_trace:
        return fused_logits, {
            "adjustment": adjustment.detach(),
            "raw_adjustment": raw_adjustment.detach(),
            "gate": trace["gate"] if trace is not None else None,
        }
    return fused_logits


def paired_view_fused_logits_from_features(
    model: nn.Module,
    primary_features: Dict[str, Tensor],
    paired_features: Dict[str, Tensor],
    primary_logits: Tensor,
    paired_logits: Tensor,
    *,
    return_trace: bool = False,
):
    fusion_head = getattr(model, "paired_view_fusion_head", None)
    if fusion_head is None:
        return None
    primary_head_input = extract_head_input_from_features(model, primary_features)
    paired_head_input = extract_head_input_from_features(model, paired_features)
    fusion_output = fusion_head(
        primary_head_input,
        paired_head_input,
        primary_logits,
        paired_logits,
        return_trace=return_trace,
    )
    if return_trace:
        raw_adjustment, trace = fusion_output
    else:
        raw_adjustment = fusion_output
        trace = None
    scale = float(getattr(model, "paired_view_fusion_logit_scale", 1.0))
    adjustment = raw_adjustment.to(dtype=primary_logits.dtype) * scale
    fused_logits = primary_logits + adjustment
    primary_features["paired_view_logits"] = paired_logits
    primary_features["paired_view_fusion_adjustment"] = adjustment
    if isinstance(primary_features.get("trace"), dict) and trace is not None:
        primary_features["trace"]["paired_view_fusion_delta"] = trace["delta"]
        primary_features["trace"]["paired_view_fusion_gate"] = trace["gate"]
        primary_features["trace"]["paired_view_fusion_adjustment"] = adjustment.detach()
    if return_trace:
        return fused_logits, {
            "adjustment": adjustment.detach(),
            "raw_adjustment": raw_adjustment.detach(),
            "gate": trace["gate"] if trace is not None else None,
        }
    return fused_logits


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


def _build_timm_classifier(
    num_classes: int,
    model_name: str,
    pretrained: bool = False,
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on optional env package
        raise ImportError(
            "model_type=timm_classifier yeu cau package timm. "
            "Hay cai timm hoac dung model_type torchvision/no-pretrain khac."
        ) from exc

    name = str(model_name or "").strip()
    if not name:
        raise ValueError("timm_classifier yeu cau timm_model_name khong rong.")
    return timm.create_model(name, pretrained=bool(pretrained), num_classes=num_classes)


MAMBAVISION_NANO_SPEC = {
    "factory": "mamba_vision_T",
    "resolution": 256,
    "dim": 48,
    "in_dim": 32,
    "depths": (1, 2, 4, 2),
    "num_heads": (2, 4, 8, 16),
    "window_size": (8, 8, 16, 8),
    "drop_path_rate": 0.10,
}


def _build_mambavision_nano(
    num_classes: int,
    image_size: int = 256,
    pretrained: bool = False,
) -> nn.Module:
    if pretrained:
        raise ValueError(
            "mambavision_nano is a locked scratch-only TRKH candidate; "
            "pretrained weights are not allowed."
        )
    if int(image_size) != int(MAMBAVISION_NANO_SPEC["resolution"]):
        raise ValueError(
            "mambavision_nano has a locked input size of "
            f"{MAMBAVISION_NANO_SPEC['resolution']}; got {image_size}."
        )
    try:
        import mambavision
    except ImportError as exc:  # pragma: no cover - depends on optional env package
        raise ImportError(
            "model_type=mambavision_nano requires mambavision==1.2.0 and "
            "mamba-ssm==2.2.4."
        ) from exc

    spec = MAMBAVISION_NANO_SPEC
    model = mambavision.create_model(
        str(spec["factory"]),
        pretrained=False,
        num_classes=int(num_classes),
        resolution=int(spec["resolution"]),
        dim=int(spec["dim"]),
        in_dim=int(spec["in_dim"]),
        depths=list(spec["depths"]),
        num_heads=list(spec["num_heads"]),
        window_size=list(spec["window_size"]),
        drop_path_rate=float(spec["drop_path_rate"]),
    )

    observed_depths = tuple(len(level.blocks) for level in model.levels)
    observed_mixers = tuple(
        tuple(
            type(block.mixer).__name__
            if getattr(level, "transformer_block", False)
            else type(block).__name__
            for block in level.blocks
        )
        for level in model.levels
    )
    expected_mixers = (
        ("ConvBlock",),
        ("ConvBlock", "ConvBlock"),
        ("MambaVisionMixer", "MambaVisionMixer", "Attention", "Attention"),
        ("MambaVisionMixer", "Attention"),
    )
    if observed_depths != tuple(spec["depths"]) or observed_mixers != expected_mixers:
        raise RuntimeError(
            "Installed MambaVision implementation does not match the locked Nano "
            f"architecture: depths={observed_depths}, mixers={observed_mixers}."
        )

    feature_dim = int(spec["dim"]) * 8
    expected_parameters = 6_170_272 + (feature_dim + 1) * int(num_classes)
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != expected_parameters:
        raise RuntimeError(
            "Installed MambaVision parameterization drifted from the locked Nano "
            f"architecture: expected={expected_parameters}, observed={observed_parameters}."
        )
    model.trkh_architecture_spec = dict(spec)
    return model


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
    timm_model_name = str(
        config.pop("timm_model_name", "mobilenetv3_large_100.ra_in1k")
    ).strip()
    # Input normalization is consumed by the data transforms, not by model
    # constructors. Keep checkpoints/configs with these fields loadable.
    config.pop("input_mean", None)
    config.pop("input_std", None)
    if pretrained and model_type not in {
        "resnet50",
        "mobilenet_v3_large",
        "vit_b_16",
        "timm_classifier",
    }:
        raise ValueError(
            "pretrained/external weights: --pretrained hien chi ho tro model_type "
            "resnet50, mobilenet_v3_large, vit_b_16, timm_classifier. "
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
    elif model_type == "timm_classifier":
        model = _build_timm_classifier(
            num_classes=num_classes,
            model_name=timm_model_name,
            pretrained=pretrained,
        )
    elif model_type == "mambavision_nano":
        if temporal_frames != 1:
            raise ValueError("mambavision_nano is locked to temporal_frames=1.")
        model = _build_mambavision_nano(
            num_classes=num_classes,
            image_size=int(config.get("image_size", MAMBAVISION_NANO_SPEC["resolution"])),
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
    override_stem_pooling_mode: Optional[str] = None,
    override_stem_softpool_blend: Optional[float] = None,
) -> nn.Module:
    class_names = checkpoint.get("class_names", [])
    resolved_num_classes = int(num_classes or len(class_names))
    model_config = dict(checkpoint.get("model_config", {}))
    checkpoint_model_type = str(model_config.get("model_type", "")).strip().lower()
    if checkpoint_model_type == "precision_ensemble":
        if resolved_num_classes < 2:
            raise ValueError("Precision-ensemble checkpoint has no valid class order.")
        ensemble_config = model_config.get("precision_ensemble")
        if not isinstance(ensemble_config, dict):
            raise ValueError("Precision-ensemble checkpoint is missing precision_ensemble config.")
        member_configs = ensemble_config.get("member_model_configs")
        if not isinstance(member_configs, dict):
            raise ValueError("Precision-ensemble checkpoint is missing member model configs.")
        keeper_config = member_configs.get("keeper")
        candidate_config = member_configs.get("candidate")
        if not isinstance(keeper_config, dict) or not isinstance(candidate_config, dict):
            raise ValueError("Precision-ensemble member configs must contain keeper and candidate.")
        keeper_config = dict(keeper_config)
        candidate_config = dict(candidate_config)
        if override_image_size is not None:
            keeper_config["image_size"] = int(override_image_size)
            candidate_config["image_size"] = int(override_image_size)
        if override_stem_pooling_mode is not None:
            keeper_config["stem_pooling_mode"] = str(override_stem_pooling_mode)
            candidate_config["stem_pooling_mode"] = str(override_stem_pooling_mode)
        if override_stem_softpool_blend is not None:
            keeper_config["stem_softpool_blend"] = float(override_stem_softpool_blend)
            candidate_config["stem_softpool_blend"] = float(override_stem_softpool_blend)

        keeper_model = create_model(
            num_classes=resolved_num_classes,
            model_config=keeper_config,
        )
        candidate_model = create_model(
            num_classes=resolved_num_classes,
            model_config=candidate_config,
        )
        from trkh.models.precision_ensemble import PrecisionEnsembleClassifier

        model = PrecisionEnsembleClassifier(
            keeper_model,
            candidate_model,
            num_classes=resolved_num_classes,
            candidate_weight=float(ensemble_config["candidate_weight"]),
            focus_class=int(ensemble_config["focus_class"]),
            focus_margin_offset=float(ensemble_config["focus_margin_offset"]),
            minimum_probability=float(
                ensemble_config.get("minimum_probability", 1e-8)
            ),
        )
        load_model_state(model, checkpoint.get("model_state", {}), strict=True)
        model.validate_rule(
            candidate_weight=float(ensemble_config["candidate_weight"]),
            focus_class=int(ensemble_config["focus_class"]),
            focus_margin_offset=float(ensemble_config["focus_margin_offset"]),
        )
        return model
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
    if override_stem_pooling_mode is not None:
        model_config["stem_pooling_mode"] = str(override_stem_pooling_mode)
    if override_stem_softpool_blend is not None:
        model_config["stem_softpool_blend"] = float(override_stem_softpool_blend)
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
