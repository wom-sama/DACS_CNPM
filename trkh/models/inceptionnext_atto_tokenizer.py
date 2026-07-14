from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class InceptionNeXtDWConv2d(nn.Module):
    """Official Inception depthwise mixer with four channel branches."""

    def __init__(
        self,
        channels: int,
        *,
        square_kernel_size: int = 3,
        band_kernel_size: int = 9,
        branch_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        channels = int(channels)
        branch_channels = int(channels * float(branch_ratio))
        identity_channels = channels - 3 * branch_channels
        if channels <= 0 or branch_channels <= 0 or identity_channels <= 0:
            raise ValueError(
                "InceptionNeXt branch split requires positive channel groups."
            )
        if int(square_kernel_size) % 2 != 1 or int(band_kernel_size) % 2 != 1:
            raise ValueError("InceptionNeXt kernels must be odd.")
        self.channels = channels
        self.square_kernel_size = int(square_kernel_size)
        self.band_kernel_size = int(band_kernel_size)
        self.branch_ratio = float(branch_ratio)
        self.split_channels = (
            identity_channels,
            branch_channels,
            branch_channels,
            branch_channels,
        )
        self.dwconv_hw = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=self.square_kernel_size,
            padding=self.square_kernel_size // 2,
            groups=branch_channels,
        )
        self.dwconv_w = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, self.band_kernel_size),
            padding=(0, self.band_kernel_size // 2),
            groups=branch_channels,
        )
        self.dwconv_h = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(self.band_kernel_size, 1),
            padding=(self.band_kernel_size // 2, 0),
            groups=branch_channels,
        )

    def branch_features(self, x: Tensor) -> Dict[str, Tensor]:
        identity, square, horizontal, vertical = torch.split(
            x,
            self.split_channels,
            dim=1,
        )
        return {
            "identity": identity,
            "square": self.dwconv_hw(square),
            "horizontal": self.dwconv_w(horizontal),
            "vertical": self.dwconv_h(vertical),
        }

    def forward(self, x: Tensor) -> Tensor:
        branches = self.branch_features(x)
        return torch.cat(tuple(branches.values()), dim=1)


class InceptionNeXtConvMlp(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(int(channels), int(hidden_channels), kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(int(hidden_channels), int(channels), kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class InceptionNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        mlp_ratio: float = 4.0,
        square_kernel_size: int = 3,
        band_kernel_size: int = 9,
        branch_ratio: float = 0.25,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        channels = int(channels)
        self.token_mixer = InceptionNeXtDWConv2d(
            channels,
            square_kernel_size=int(square_kernel_size),
            band_kernel_size=int(band_kernel_size),
            branch_ratio=float(branch_ratio),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.mlp = InceptionNeXtConvMlp(
            channels,
            hidden_channels=int(channels * float(mlp_ratio)),
        )
        self.gamma = nn.Parameter(
            torch.full((channels,), float(layer_scale_init))
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.mlp(self.norm(self.token_mixer(x)))
        return x + residual * self.gamma.reshape(1, -1, 1, 1)


class InceptionNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(int(in_channels)),
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=2,
                    stride=2,
                ),
            )
        elif int(in_channels) == int(out_channels):
            self.downsample = nn.Identity()
        else:
            raise ValueError("A width-changing InceptionNeXt stage must downsample.")
        self.blocks = nn.ModuleList(
            InceptionNeXtBlock(int(out_channels)) for _ in range(int(depth))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class InceptionNeXtAttoTokenizer(nn.Module):
    """Official InceptionNeXt-Atto stages 1-3 as a TRKH tokenizer."""

    source_commit = "3f9769c6b3fcf903d1dc2f436eddc6963cacb535"
    source_sha256 = "aed1b0a9ac410d5d9db042b85afe23de2b799bc70110c349acf851688e9f7b88"
    stage_channels = (40, 80, 160)
    stage_depths = (2, 2, 6)
    square_kernel_size = 3
    band_kernel_size = 9
    branch_ratio = 0.25
    mlp_ratio = 4.0
    layer_scale_init = 1e-6
    downsample_factor = 16
    out_channels = 160

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError(
                "inceptionnext_atto_tokenizer requires three-channel RGB input."
            )
        self.stem = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                self.stage_channels[0],
                kernel_size=4,
                stride=4,
            ),
            nn.BatchNorm2d(self.stage_channels[0]),
        )
        self.stages = nn.ModuleList(
            (
                InceptionNeXtStage(
                    self.stage_channels[0],
                    self.stage_channels[0],
                    depth=self.stage_depths[0],
                    downsample=False,
                ),
                InceptionNeXtStage(
                    self.stage_channels[0],
                    self.stage_channels[1],
                    depth=self.stage_depths[1],
                    downsample=True,
                ),
                InceptionNeXtStage(
                    self.stage_channels[1],
                    self.stage_channels[2],
                    depth=self.stage_depths[2],
                    downsample=True,
                ),
            )
        )
        self.reset_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)
        for stage in self.stages:
            for block in stage.blocks:
                nn.init.constant_(block.gamma, self.layer_scale_init)

    @property
    def block_count(self) -> int:
        return sum(len(stage.blocks) for stage in self.stages)

    @property
    def stage_splits(self) -> Tuple[Tuple[int, int, int, int], ...]:
        return tuple(
            stage.blocks[0].token_mixer.split_channels for stage in self.stages
        )

    def forward_with_trace(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        trace: Dict[str, Tensor] = {}
        x = self.stem(x)
        for stage_index, stage in enumerate(self.stages, start=1):
            x = stage.downsample(x)
            trace[f"stage{stage_index}_embedded"] = x
            for block in stage.blocks:
                x = block(x)
            trace[f"stage{stage_index}_output"] = x
        return x, trace

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_trace(x)[0]
