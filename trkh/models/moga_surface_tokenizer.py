from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MogaElementScale(nn.Module):
    def __init__(self, channels: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((1, int(channels), 1, 1), float(init_value))
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class MogaChannelAggregationFFN(nn.Module):
    """Official MogaNet channel mixer with feature decomposition."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act = nn.GELU()
        self.decompose = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.decompose_act = nn.GELU()
        self.sigma = MogaElementScale(hidden_channels, init_value=1e-5)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.dwconv(self.fc1(x)))
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return self.fc2(x)


class MogaMultiOrderDWConv(nn.Module):
    """The locked 1:3:4 low/middle/high-order MogaNet context mixer."""

    def __init__(
        self,
        channels: int,
        dilations: Tuple[int, int, int] = (1, 2, 3),
        channel_split: Tuple[int, int, int] = (1, 3, 4),
    ) -> None:
        super().__init__()
        channels = int(channels)
        if len(dilations) != 3 or len(channel_split) != 3:
            raise ValueError("Moga multi-order settings require exactly three branches.")
        if min(dilations) < 1 or max(dilations) > 3:
            raise ValueError("Moga dilations must remain in [1, 3].")
        split_total = int(sum(channel_split))
        if channels <= 0 or channels % split_total != 0:
            raise ValueError(
                f"Moga channels must be positive and divisible by {split_total}: {channels}"
            )
        middle_channels = channels * int(channel_split[1]) // split_total
        high_channels = channels * int(channel_split[2]) // split_total
        low_channels = channels - middle_channels - high_channels
        self.channels = channels
        self.low_channels = low_channels
        self.middle_channels = middle_channels
        self.high_channels = high_channels
        self.dw_conv_low = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=(1 + 4 * int(dilations[0])) // 2,
            dilation=int(dilations[0]),
            groups=channels,
        )
        self.dw_conv_middle = nn.Conv2d(
            middle_channels,
            middle_channels,
            kernel_size=5,
            stride=1,
            padding=(1 + 4 * int(dilations[1])) // 2,
            dilation=int(dilations[1]),
            groups=middle_channels,
        )
        self.dw_conv_high = nn.Conv2d(
            high_channels,
            high_channels,
            kernel_size=7,
            stride=1,
            padding=(1 + 6 * int(dilations[2])) // 2,
            dilation=int(dilations[2]),
            groups=high_channels,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

    def branch_features(self, x: Tensor) -> Dict[str, Tensor]:
        low_all = self.dw_conv_low(x)
        middle_start = self.low_channels
        middle_end = middle_start + self.middle_channels
        middle = self.dw_conv_middle(low_all[:, middle_start:middle_end])
        high = self.dw_conv_high(low_all[:, -self.high_channels :])
        return {
            "low": low_all[:, : self.low_channels],
            "middle": middle,
            "high": high,
        }

    def forward(self, x: Tensor) -> Tensor:
        branches = self.branch_features(x)
        return self.project(
            torch.cat(
                (branches["low"], branches["middle"], branches["high"]),
                dim=1,
            )
        )


class MogaMultiOrderGatedAggregation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.proj_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigma = MogaElementScale(channels, init_value=1e-5)
        self.value_act = nn.SiLU()
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = MogaMultiOrderDWConv(channels)
        self.gate_act = nn.SiLU()
        self.proj_2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        projected = self.proj_1(x)
        global_component = F.adaptive_avg_pool2d(projected, output_size=1)
        decomposed = self.value_act(
            projected + self.sigma(projected - global_component)
        )
        gate = self.gate_act(self.gate(decomposed))
        value = self.gate_act(self.value(decomposed))
        return shortcut + self.proj_2(gate * value)


class MogaBlock(nn.Module):
    def __init__(self, channels: int, ffn_ratio: float) -> None:
        super().__init__()
        channels = int(channels)
        self.norm1 = nn.BatchNorm2d(channels, eps=1e-5)
        self.spatial = MogaMultiOrderGatedAggregation(channels)
        self.layer_scale_1 = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-5)
        )
        self.norm2 = nn.BatchNorm2d(channels, eps=1e-5)
        self.channel = MogaChannelAggregationFFN(
            channels=channels,
            hidden_channels=int(channels * float(ffn_ratio)),
        )
        self.layer_scale_2 = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-5)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.layer_scale_1 * self.spatial(self.norm1(x))
        return x + self.layer_scale_2 * self.channel(self.norm2(x))


class MogaConvPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.norm = nn.BatchNorm2d(int(out_channels), eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.projection(x))


class MogaStackedPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden_channels = int(out_channels) // 2
        self.projection = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels, eps=1e-5),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                int(out_channels),
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(out_channels), eps=1e-5),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class MogaXTTokenizer(nn.Module):
    """MogaNet-XT stages 1-3 used as a 16x16 TRKH surface tokenizer."""

    source_commit = "c83e328b513289fddd2921e17a9143b0f966c0a1"
    source_sha256 = "1ac13dfb57db813ab310b581c434d3240d875f0997bf955550c92abf49c79797"
    stage_channels = (32, 64, 96)
    stage_depths = (3, 3, 10)
    stage_ffn_ratios = (8.0, 8.0, 4.0)
    downsample_factor = 16
    out_channels = 96

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("moganet_xt_tokenizer requires three-channel RGB input.")
        self.patch_embed1 = MogaStackedPatchEmbed(in_channels, self.stage_channels[0])
        self.blocks1 = nn.ModuleList(
            MogaBlock(self.stage_channels[0], self.stage_ffn_ratios[0])
            for _ in range(self.stage_depths[0])
        )
        self.norm1 = nn.BatchNorm2d(self.stage_channels[0], eps=1e-5)
        self.patch_embed2 = MogaConvPatchEmbed(
            self.stage_channels[0], self.stage_channels[1]
        )
        self.blocks2 = nn.ModuleList(
            MogaBlock(self.stage_channels[1], self.stage_ffn_ratios[1])
            for _ in range(self.stage_depths[1])
        )
        self.norm2 = nn.BatchNorm2d(self.stage_channels[1], eps=1e-5)
        self.patch_embed3 = MogaConvPatchEmbed(
            self.stage_channels[1], self.stage_channels[2]
        )
        self.blocks3 = nn.ModuleList(
            MogaBlock(self.stage_channels[2], self.stage_ffn_ratios[2])
            for _ in range(self.stage_depths[2])
        )
        self.norm3 = nn.BatchNorm2d(self.stage_channels[2], eps=1e-5)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = (
                int(module.kernel_size[0])
                * int(module.kernel_size[1])
                * int(module.out_channels)
            )
            fan_out //= int(module.groups)
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def block_count(self) -> int:
        return len(self.blocks1) + len(self.blocks2) + len(self.blocks3)

    def forward_with_trace(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        trace: Dict[str, Tensor] = {}
        for stage_index in range(1, 4):
            patch_embed = getattr(self, f"patch_embed{stage_index}")
            blocks = getattr(self, f"blocks{stage_index}")
            norm = getattr(self, f"norm{stage_index}")
            x = patch_embed(x)
            trace[f"stage{stage_index}_embedded"] = x
            for block in blocks:
                x = block(x)
            x = norm(x)
            trace[f"stage{stage_index}_output"] = x
        return x, trace

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_trace(x)[0]
