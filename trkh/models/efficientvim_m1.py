# Copyright (c) 2024 MLVLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Minimal, deployment-oriented EfficientViM-M1 with verified provenance.

Adapted from MLVLab/EfficientViM at commit
``304340cb9c339b61669250d058525c9cdadd5e93``. This module intentionally keeps
only the non-distilled M1 classifier and removes the upstream registry and
FLOP-analysis dependency. Its state-dict names remain strictly compatible with
the official non-distilled 1,000-class M1 checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OFFICIAL_REPOSITORY = "https://github.com/mlvlab/EfficientViM"
OFFICIAL_REVISION = "304340cb9c339b61669250d058525c9cdadd5e93"
OFFICIAL_M1_E450_URL = (
    "https://drive.google.com/file/d/1ztpSVnWccoGEEobpmk107US3MjX4LEgz/view"
)
OFFICIAL_M1_E450_SHA256 = (
    "c04c83b982a9a136cec8dca98c5397540bb7d8b18acaa22e559edca042c937a6"
)
OFFICIAL_STATE_KEY = "model_ema"


class _LayerNorm2D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = channels
        self.eps = eps
        self.affine = True
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return ((x - mean) / torch.sqrt(variance + self.eps)) * self.weight + self.bias


class _LayerNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = channels
        self.eps = eps
        self.affine = True
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return ((x - mean) / torch.sqrt(variance + self.eps)) * self.weight + self.bias


class _ConvLayer2D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        norm: type[nn.Module] | None = nn.BatchNorm2d,
        act_layer: type[nn.Module] | None = nn.ReLU,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            groups=groups,
            bias=False,
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _ConvLayer1D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        norm: type[nn.Module] | None = nn.BatchNorm1d,
        act_layer: type[nn.Module] | None = nn.ReLU,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _SqueezeExcite(nn.Module):
    """Exact M1 form of timm's SqueezeExcite, without a timm dependency."""

    def __init__(self, channels: int, reduction_ratio: float = 0.25) -> None:
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * self.gate(scale)


class _FFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = _ConvLayer2D(in_dim, hidden_dim, kernel_size=1)
        self.fc2 = _ConvLayer2D(
            hidden_dim, in_dim, kernel_size=1, act_layer=None, bn_weight_init=0
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x))


class _Stem(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ConvLayer2D(in_dim, dim // 8, stride=2, padding=1),
            _ConvLayer2D(dim // 8, dim // 4, stride=2, padding=1),
            _ConvLayer2D(dim // 4, dim // 2, stride=2, padding=1),
            _ConvLayer2D(dim // 2, dim, stride=2, padding=1, act_layer=None),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden_dim = out_dim * 4
        self.conv = nn.Sequential(
            _ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            _ConvLayer2D(
                hidden_dim,
                hidden_dim,
                stride=2,
                padding=1,
                groups=hidden_dim,
            ),
            _SqueezeExcite(hidden_dim),
            _ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None),
        )
        self.dwconv1 = _ConvLayer2D(
            in_dim, in_dim, padding=1, groups=in_dim, act_layer=None
        )
        self.dwconv2 = _ConvLayer2D(
            out_dim, out_dim, padding=1, groups=out_dim, act_layer=None
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dwconv1(x)
        x = self.conv(x)
        return x + self.dwconv2(x)


class _HSMSSD(nn.Module):
    def __init__(self, dim: int, state_dim: int) -> None:
        super().__init__()
        self.ssd_expand = 1.0
        self.d_inner = dim
        self.state_dim = state_dim
        self.BCdt_proj = _ConvLayer1D(
            dim, 3 * state_dim, kernel_size=1, norm=None, act_layer=None
        )
        conv_dim = 3 * state_dim
        self.dw = _ConvLayer2D(
            conv_dim,
            conv_dim,
            stride=1,
            padding=1,
            groups=conv_dim,
            norm=None,
            act_layer=None,
            bn_weight_init=0,
        )
        self.hz_proj = _ConvLayer1D(
            dim, 2 * dim, kernel_size=1, norm=None, act_layer=None
        )
        self.out_proj = _ConvLayer1D(
            dim, dim, kernel_size=1, norm=None, act_layer=None, bn_weight_init=0
        )
        self.A = nn.Parameter(torch.empty(state_dim).uniform_(1, 16))
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, length = x.shape
        side = math.isqrt(int(length))
        if side * side != length:
            raise ValueError(
                f"EfficientViM hidden length must be a perfect square, got {length}."
            )

        projected = self.BCdt_proj(x).view(batch, -1, side, side)
        b_state, c_state, delta = torch.split(
            self.dw(projected).flatten(2),
            [self.state_dim, self.state_dim, self.state_dim],
            dim=1,
        )
        transition = (delta + self.A.view(1, -1, 1)).softmax(-1)
        hidden = x @ (transition * b_state).transpose(-2, -1)
        hidden, gate = torch.split(self.hz_proj(hidden), [self.d_inner] * 2, dim=1)
        hidden = self.out_proj(hidden * self.act(gate) + hidden * self.D)
        output = (hidden @ c_state).view(batch, -1, side, side).contiguous()
        return output, hidden


class _EfficientViMBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp_ratio = 4.0
        self.mixer = _HSMSSD(dim, state_dim)
        self.norm = _LayerNorm1D(dim)
        self.dwconv1 = _ConvLayer2D(
            dim, dim, padding=1, groups=dim, bn_weight_init=0, act_layer=None
        )
        self.dwconv2 = _ConvLayer2D(
            dim, dim, padding=1, groups=dim, bn_weight_init=0, act_layer=None
        )
        self.ffn = _FFN(dim, dim * 4)
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)
        residual = x
        x, hidden = self.mixer(self.norm(x.flatten(2)))
        x = (1 - alpha[1]) * residual + alpha[1] * x
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, hidden


class _EfficientViMStage(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int | None, depth: int, state_dim: int
    ) -> None:
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList(
            [_EfficientViMBlock(in_dim, state_dim) for _ in range(depth)]
        )
        self.downsample = _PatchMerging(in_dim, out_dim) if out_dim else None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden: Tensor | None = None
        for block in self.blocks:
            x, hidden = block(x)
        if hidden is None:  # Depth is fixed to two, but keep failure explicit.
            raise RuntimeError("EfficientViM stage must contain at least one block.")
        stage_output = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, stage_output, hidden


class EfficientViMM1(nn.Module):
    """Official non-distilled EfficientViM-M1, fixed to square 224px input."""

    input_size = 224
    feature_dim = 320
    multistage_dims = (128, 192, 320, 320)

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        self.num_layers = 3
        self.num_classes = num_classes
        self.distillation = False
        dims = self.multistage_dims[:3]
        state_dims = (49, 25, 9)
        self.patch_embed = _Stem(3, dims[0])
        self.stages = nn.ModuleList(
            [
                _EfficientViMStage(
                    dims[index],
                    dims[index + 1] if index < self.num_layers - 1 else None,
                    depth=2,
                    state_dim=state_dims[index],
                )
                for index in range(self.num_layers)
            ]
        )
        self.weights = nn.Parameter(torch.ones(4))
        self.norm = nn.ModuleList(
            [
                _LayerNorm1D(dims[0]),
                _LayerNorm1D(dims[1]),
                _LayerNorm1D(dims[2]),
                _LayerNorm2D(dims[2]),
            ]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(dim, num_classes) for dim in self.multistage_dims]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (_LayerNorm1D, _LayerNorm2D)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor) or x.ndim != 4:
            raise ValueError("EfficientViM-M1 input must have shape [B, 3, 224, 224].")
        _, channels, height, width = x.shape
        if channels != 3 or height != width or height != self.input_size:
            raise ValueError(
                "EfficientViM-M1 is locked to square [B, 3, 224, 224] input; "
                f"got {tuple(x.shape)}."
            )

    def forward_multistage_features(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return three pooled hidden states plus the final pooled spatial state."""

        self._validate_input(x)
        x = self.patch_embed(x)
        features: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            x, _, hidden = stage(x)
            hidden = self.norm[index](hidden)
            features.append(F.adaptive_avg_pool1d(hidden, 1).flatten(1))
        x = self.norm[3](x)
        features.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return features[0], features[1], features[2], features[3]

    def forward_final_features(self, x: Tensor) -> Tensor:
        """Return the 320-dimensional input used by the fourth official head."""

        return self.forward_multistage_features(x)[-1]

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_multistage_features(x)
        fusion_weights = self.weights.softmax(-1)
        logits = torch.zeros(
            (features[0].shape[0], self.num_classes), device=features[0].device
        )
        for index, feature in enumerate(features):
            logits = logits + fusion_weights[index] * self.heads[index](feature)
        return logits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_efficientvim_m1_weights(
    model: nn.Module,
    checkpoint_path: str | PathLike[str],
    *,
    expected_sha256: str = OFFICIAL_M1_E450_SHA256,
) -> str:
    """Verify and strictly load the official EMA state without unsafe pickle.

    The downloaded checkpoint embeds a ``yacs.config.CfgNode`` and training
    state. A temporary inert class is allow-listed only for the duration of
    ``weights_only=True`` deserialization. It is never installed in
    ``sys.modules`` and the safe-global context is restored on exit.

    Returns the verified lowercase SHA-256 digest.
    """

    path = Path(checkpoint_path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "EfficientViM checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}."
        )

    cfg_node = type("CfgNode", (dict,), {"__module__": "yacs.config"})
    with torch.serialization.safe_globals([cfg_node]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("EfficientViM checkpoint root must be a mapping.")
    if OFFICIAL_STATE_KEY not in checkpoint:
        raise KeyError(
            f"EfficientViM checkpoint is missing required '{OFFICIAL_STATE_KEY}' state."
        )
    state = checkpoint[OFFICIAL_STATE_KEY]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise TypeError("EfficientViM model_ema must be a string-to-tensor mapping.")
    model.load_state_dict(state, strict=True)
    return actual_sha256


__all__ = [
    "EfficientViMM1",
    "OFFICIAL_M1_E450_SHA256",
    "OFFICIAL_M1_E450_URL",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_REVISION",
    "OFFICIAL_STATE_KEY",
    "load_official_efficientvim_m1_weights",
]
