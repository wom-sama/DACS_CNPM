"""Parameter-matched Octave Convolution stem for TRKH."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _frequency_channels(channels: int, alpha: float) -> Tuple[int, int]:
    channels = int(channels)
    alpha = float(alpha)
    if channels <= 0:
        raise ValueError("OctConv channels must be positive.")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("OctConv alpha must be in [0, 1).")
    low = int(round(channels * alpha))
    high = channels - low
    if high <= 0:
        raise ValueError("OctConv must retain at least one high-frequency channel.")
    return high, low


class OctaveConv2d(nn.Module):
    """ICCV-2019 OctConv with explicit high/low communication paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        alpha_in: float,
        alpha_out: float,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("OctConv kernel_size must be a positive odd integer.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.alpha_in = float(alpha_in)
        self.alpha_out = float(alpha_out)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.in_high_channels, self.in_low_channels = _frequency_channels(
            self.in_channels, self.alpha_in
        )
        self.out_high_channels, self.out_low_channels = _frequency_channels(
            self.out_channels, self.alpha_out
        )

        def make_conv(input_channels: int, output_channels: int) -> Optional[nn.Conv2d]:
            if input_channels <= 0 or output_channels <= 0:
                return None
            return nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                bias=False,
            )

        self.conv_hh = make_conv(self.in_high_channels, self.out_high_channels)
        self.conv_hl = make_conv(self.in_high_channels, self.out_low_channels)
        self.conv_ll = make_conv(self.in_low_channels, self.out_low_channels)
        self.conv_lh = make_conv(self.in_low_channels, self.out_high_channels)

    @property
    def available_paths(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, module in (
                ("hh", self.conv_hh),
                ("hl", self.conv_hl),
                ("ll", self.conv_ll),
                ("lh", self.conv_lh),
            )
            if module is not None
        )

    @property
    def kernel_parameter_count(self) -> int:
        return sum(
            int(module.weight.numel())
            for module in (self.conv_hh, self.conv_hl, self.conv_ll, self.conv_lh)
            if module is not None
        )

    def _compute_paths(
        self,
        high: Tensor,
        low: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        if high.ndim != 4 or int(high.shape[1]) != self.in_high_channels:
            raise ValueError(
                "OctConv high input must be BCHW with the configured high channels."
            )
        if self.in_low_channels > 0:
            if low is None or low.ndim != 4 or int(low.shape[1]) != self.in_low_channels:
                raise ValueError(
                    "OctConv low input must be BCHW with the configured low channels."
                )
            if tuple(high.shape[-2:]) != (
                int(low.shape[-2]) * 2,
                int(low.shape[-1]) * 2,
            ):
                raise ValueError("OctConv low input must be one spatial octave below high.")
        elif low is not None:
            raise ValueError("OctConv received a low input while alpha_in is zero.")

        paths: Dict[str, Tensor] = {}
        if self.conv_hh is not None:
            paths["hh"] = self.conv_hh(high)
        if self.conv_hl is not None:
            paths["hl"] = self.conv_hl(F.avg_pool2d(high, kernel_size=2, stride=2))
        if self.conv_ll is not None and low is not None:
            paths["ll"] = self.conv_ll(low)
        if self.conv_lh is not None and low is not None:
            paths["lh"] = F.interpolate(
                self.conv_lh(low),
                size=tuple(int(value) for value in high.shape[-2:]),
                mode="nearest",
            )
        return paths

    def forward_with_paths(
        self,
        high: Tensor,
        low: Optional[Tensor] = None,
        *,
        disabled_paths: Iterable[str] = (),
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
        disabled = {str(value).strip().lower() for value in disabled_paths}
        unknown = disabled.difference(self.available_paths)
        if unknown:
            raise ValueError(f"Cannot disable unavailable OctConv paths: {sorted(unknown)}")
        paths = self._compute_paths(high, low)

        high_terms = [
            paths[name]
            for name in ("hh", "lh")
            if name in paths and name not in disabled
        ]
        if not high_terms:
            raise ValueError("OctConv ablation removed every high-output path.")
        high_output = high_terms[0]
        for term in high_terms[1:]:
            high_output = high_output + term

        low_output: Optional[Tensor] = None
        if self.out_low_channels > 0:
            low_terms = [
                paths[name]
                for name in ("ll", "hl")
                if name in paths and name not in disabled
            ]
            if not low_terms:
                raise ValueError("OctConv ablation removed every low-output path.")
            low_output = low_terms[0]
            for term in low_terms[1:]:
                low_output = low_output + term
        return high_output, low_output, paths

    def forward(
        self,
        high: Tensor,
        low: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        high_output, low_output, _ = self.forward_with_paths(high, low)
        return high_output, low_output

    def reset_from_virtual_vanilla_kernel(self) -> None:
        reference = next(module for module in self.modules() if isinstance(module, nn.Conv2d))
        full_weight = torch.empty(
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            device=reference.weight.device,
            dtype=reference.weight.dtype,
        )
        nn.init.kaiming_normal_(full_weight, mode="fan_out", nonlinearity="relu")
        ih = self.in_high_channels
        oh = self.out_high_channels
        with torch.no_grad():
            if self.conv_hh is not None:
                self.conv_hh.weight.copy_(full_weight[:oh, :ih])
            if self.conv_hl is not None:
                self.conv_hl.weight.copy_(full_weight[oh:, :ih])
            if self.conv_ll is not None:
                self.conv_ll.weight.copy_(full_weight[oh:, ih:])
            if self.conv_lh is not None:
                self.conv_lh.weight.copy_(full_weight[:oh, ih:])

    def reconstructed_vanilla_kernel(self) -> Tensor:
        reference = next(module for module in self.modules() if isinstance(module, nn.Conv2d))
        full_weight = reference.weight.new_zeros(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        ih = self.in_high_channels
        oh = self.out_high_channels
        with torch.no_grad():
            if self.conv_hh is not None:
                full_weight[:oh, :ih].copy_(self.conv_hh.weight)
            if self.conv_hl is not None:
                full_weight[oh:, :ih].copy_(self.conv_hl.weight)
            if self.conv_ll is not None:
                full_weight[oh:, ih:].copy_(self.conv_ll.weight)
            if self.conv_lh is not None:
                full_weight[:oh, ih:].copy_(self.conv_lh.weight)
        return full_weight


class OctaveConvStemBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        alpha_in: float,
        alpha_out: float,
    ) -> None:
        super().__init__()
        self.octave = OctaveConv2d(
            in_channels,
            out_channels,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )
        self.high_norm = nn.BatchNorm2d(self.octave.out_high_channels)
        self.low_norm = (
            nn.BatchNorm2d(self.octave.out_low_channels)
            if self.octave.out_low_channels > 0
            else None
        )
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_with_trace(
        self,
        high: Tensor,
        low: Optional[Tensor] = None,
        *,
        disabled_paths: Iterable[str] = (),
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
        high_output, low_output, paths = self.octave.forward_with_paths(
            high,
            low,
            disabled_paths=disabled_paths,
        )
        high_output = self.pool(self.activation(self.high_norm(high_output)))
        if low_output is not None:
            if self.low_norm is None:
                raise RuntimeError("OctConv low output lacks its normalization module.")
            low_output = self.pool(self.activation(self.low_norm(low_output)))
        trace = {f"path_{name}": value for name, value in paths.items()}
        trace["high_output"] = high_output
        if low_output is not None:
            trace["low_output"] = low_output
        return high_output, low_output, trace

    def forward(
        self,
        high: Tensor,
        low: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        high_output, low_output, _ = self.forward_with_trace(high, low)
        return high_output, low_output

    def reset_from_virtual_vanilla_kernel(self) -> None:
        self.octave.reset_from_virtual_vanilla_kernel()
        for norm in (self.high_norm, self.low_norm):
            if isinstance(norm, nn.BatchNorm2d):
                norm.reset_running_stats()
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)


class OctaveConvStem(nn.Module):
    """Three-block, stride-8 OctConv replacement for HybridConvStem."""

    source_commit = "87c44f79162f3a2ef316bf924ad3697b8957a463"
    source_sha256 = "06e60037b8fd5d4cd9e063ab1d439b25ddc98c4a109f770cbc0adcb9d7295675"
    alpha = 0.125
    downsample_factor = 8

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("octave_conv stem requires three-channel RGB input.")
        mid_channels = int(stem_channels) * 2
        self.out_channels = int(embed_dim)
        self.blocks = nn.ModuleList(
            (
                OctaveConvStemBlock(
                    int(in_channels),
                    int(stem_channels),
                    alpha_in=0.0,
                    alpha_out=self.alpha,
                ),
                OctaveConvStemBlock(
                    int(stem_channels),
                    mid_channels,
                    alpha_in=self.alpha,
                    alpha_out=self.alpha,
                ),
                OctaveConvStemBlock(
                    mid_channels,
                    self.out_channels,
                    alpha_in=self.alpha,
                    alpha_out=0.0,
                ),
            )
        )

    @property
    def kernel_parameter_count(self) -> int:
        return sum(block.octave.kernel_parameter_count for block in self.blocks)

    def reset_from_virtual_vanilla_kernels(self) -> None:
        for block in self.blocks:
            block.reset_from_virtual_vanilla_kernel()

    def reconstructed_vanilla_kernels(self) -> Tuple[Tensor, ...]:
        return tuple(block.octave.reconstructed_vanilla_kernel() for block in self.blocks)

    def forward_with_trace(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        high = x
        low: Optional[Tensor] = None
        trace: Dict[str, Tensor] = {}
        for index, block in enumerate(self.blocks, start=1):
            high, low, block_trace = block.forward_with_trace(high, low)
            for name, value in block_trace.items():
                trace[f"block{index}_{name}"] = value
        if low is not None:
            raise RuntimeError("Final OctConv stem block must merge the low-frequency group.")
        trace["stem_output"] = high
        return high, trace

    def forward(self, x: Tensor) -> Tensor:
        high = x
        low: Optional[Tensor] = None
        for block in self.blocks:
            high, low = block(high, low)
        if low is not None:
            raise RuntimeError("Final OctConv stem block must merge the low-frequency group.")
        return high
