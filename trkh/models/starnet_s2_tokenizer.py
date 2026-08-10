"""StarNet-S2 tokenizer adapted from the Apache-2.0 official implementation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class StarNetConvBN(nn.Module):
    """Bias-enabled Conv2d followed by the optional official BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        *,
        with_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            groups=int(groups),
            bias=True,
        )
        self.bn = nn.BatchNorm2d(int(out_channels)) if with_bn else nn.Identity()
        self.with_bn = bool(with_bn)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))


class StarNetBlock(nn.Module):
    """Official local star block with a same-weight sum audit path."""

    def __init__(self, channels: int, *, expansion: int = 4) -> None:
        super().__init__()
        channels = int(channels)
        hidden_channels = channels * int(expansion)
        if channels <= 0 or int(expansion) <= 0:
            raise ValueError("StarNet channels and expansion must be positive.")
        self.channels = channels
        self.expansion = int(expansion)
        self.dwconv = StarNetConvBN(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=channels,
            with_bn=True,
        )
        self.f1 = StarNetConvBN(
            channels,
            hidden_channels,
            with_bn=False,
        )
        self.f2 = StarNetConvBN(
            channels,
            hidden_channels,
            with_bn=False,
        )
        self.g = StarNetConvBN(
            hidden_channels,
            channels,
            with_bn=True,
        )
        self.dwconv2 = StarNetConvBN(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=channels,
            with_bn=False,
        )
        self.act = nn.ReLU6()
        self.drop_path = nn.Identity()

    def interaction_features(self, x: Tensor) -> Dict[str, Tensor]:
        spatial = self.dwconv(x)
        f1 = self.f1(spatial)
        f2 = self.f2(spatial)
        activated_f1 = self.act(f1)
        return {
            "spatial": spatial,
            "f1": f1,
            "f2": f2,
            "activated_f1": activated_f1,
            "star": activated_f1 * f2,
            "sum": activated_f1 + f2,
        }

    def forward_with_interaction(self, x: Tensor, *, interaction: str) -> Tensor:
        features = self.interaction_features(x)
        interaction = str(interaction).strip().lower()
        if interaction not in {"star", "sum"}:
            raise ValueError("StarNet interaction must be 'star' or 'sum'.")
        residual = self.dwconv2(self.g(features[interaction]))
        return x + self.drop_path(residual)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_interaction(x, interaction="star")


class StarNetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if int(depth) <= 0:
            raise ValueError("StarNet stage depth must be positive.")
        self.downsample = StarNetConvBN(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=2,
            padding=1,
            with_bn=True,
        )
        self.blocks = nn.ModuleList(
            StarNetBlock(int(out_channels), expansion=int(expansion))
            for _ in range(int(depth))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class StarNetS2Tokenizer(nn.Module):
    """Official StarNet-S2 stages 1-3 as a local TRKH tokenizer."""

    source_commit = "c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e"
    source_sha256 = "4e9eb1f58ea51427eebe17baf9891a3e1d73d000597c032a700176c954c3031a"
    stem_channels = 32
    stage_channels = (32, 64, 128)
    stage_depths = (1, 2, 6)
    expansion = 4
    depthwise_kernel_size = 7
    downsample_factor = 16
    out_channels = 128

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("starnet_s2_tokenizer requires three-channel RGB input.")
        self.stem = nn.Sequential(
            StarNetConvBN(
                int(in_channels),
                self.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                with_bn=True,
            ),
            nn.ReLU6(),
        )
        self.stages = nn.ModuleList(
            (
                StarNetStage(
                    self.stem_channels,
                    self.stage_channels[0],
                    depth=self.stage_depths[0],
                    expansion=self.expansion,
                ),
                StarNetStage(
                    self.stage_channels[0],
                    self.stage_channels[1],
                    depth=self.stage_depths[1],
                    expansion=self.expansion,
                ),
                StarNetStage(
                    self.stage_channels[1],
                    self.stage_channels[2],
                    depth=self.stage_depths[2],
                    expansion=self.expansion,
                ),
            )
        )

    def reset_parameters(self) -> None:
        # The official source leaves Conv2d at PyTorch defaults and resets BN.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.reset_parameters()
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_parameters()

    @property
    def block_count(self) -> int:
        return sum(len(stage.blocks) for stage in self.stages)

    @property
    def stage_width_depth_pairs(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(zip(self.stage_channels, self.stage_depths))

    def forward_with_trace(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        trace: Dict[str, Tensor] = {}
        x = self.stem(x)
        trace["stem_output"] = x
        for stage_index, stage in enumerate(self.stages, start=1):
            x = stage.downsample(x)
            trace[f"stage{stage_index}_embedded"] = x
            for block in stage.blocks:
                x = block(x)
            trace[f"stage{stage_index}_output"] = x
        return x, trace

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_trace(x)[0]
