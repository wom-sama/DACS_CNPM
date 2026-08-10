"""CVPR-2021 Diverse Branch Block stem for scratch TRKH training.

The structural transformations are adapted from Ding et al.'s official
Apache-2.0 implementation at commit 8d2b16b6aee45a33236b2d11685be6857f9ba929.
The training-time branches are converted once into ordinary 3x3 convolutions
for deployment.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Callable, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def fuse_conv_batch_norm(kernel: Tensor, norm: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]:
    """Fold an affine BatchNorm2d into a bias-free convolution kernel."""
    if norm.weight is None or norm.bias is None:
        raise ValueError("DBB requires affine BatchNorm2d modules.")
    scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
    fused_bias = norm.bias - norm.running_mean * scale
    return fused_kernel, fused_bias


def add_equivalent_branches(
    kernels: Tuple[Tensor, ...],
    biases: Tuple[Tensor, ...],
) -> Tuple[Tensor, Tensor]:
    if not kernels or len(kernels) != len(biases):
        raise ValueError("DBB branch kernels and biases must be nonempty and aligned.")
    return torch.stack(kernels, dim=0).sum(dim=0), torch.stack(biases, dim=0).sum(dim=0)


def compose_1x1_kxk(
    first_kernel: Tensor,
    first_bias: Tensor,
    second_kernel: Tensor,
    second_bias: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Fuse a dense 1x1 convolution followed by a KxK convolution."""
    if first_kernel.shape[-2:] != (1, 1):
        raise ValueError("The first DBB sequential kernel must be 1x1.")
    if second_kernel.is_cuda:
        with torch.backends.cudnn.flags(allow_tf32=False):
            fused_kernel = F.conv2d(
                second_kernel,
                first_kernel.permute(1, 0, 2, 3),
            )
    else:
        fused_kernel = F.conv2d(
            second_kernel,
            first_kernel.permute(1, 0, 2, 3),
        )
    propagated_bias = (
        second_kernel * first_bias.reshape(1, -1, 1, 1)
    ).sum(dim=(1, 2, 3))
    return fused_kernel, propagated_bias + second_bias


def average_pool_kernel(
    channels: int,
    kernel_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    channels = int(channels)
    kernel_size = int(kernel_size)
    if channels <= 0 or kernel_size <= 0:
        raise ValueError("Average-pool conversion dimensions must be positive.")
    kernel = torch.zeros(
        channels,
        channels,
        kernel_size,
        kernel_size,
        device=device,
        dtype=dtype,
    )
    diagonal = torch.arange(channels, device=device)
    kernel[diagonal, diagonal] = 1.0 / float(kernel_size * kernel_size)
    return kernel


def pad_kernel_to(kernel: Tensor, target_kernel_size: int) -> Tensor:
    target_kernel_size = int(target_kernel_size)
    height_padding = target_kernel_size - int(kernel.shape[-2])
    width_padding = target_kernel_size - int(kernel.shape[-1])
    if height_padding < 0 or width_padding < 0:
        raise ValueError("Cannot shrink a DBB branch kernel during fusion.")
    if height_padding % 2 or width_padding % 2:
        raise ValueError("DBB fusion requires symmetric integer kernel padding.")
    return F.pad(
        kernel,
        (
            width_padding // 2,
            width_padding // 2,
            height_padding // 2,
            height_padding // 2,
        ),
    )


class IdentityBasedConv1x1(nn.Conv2d):
    """Trainable residual around an exact grouped 1x1 identity kernel."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            in_channels=int(channels),
            out_channels=int(channels),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        identity = torch.eye(int(channels)).reshape(int(channels), int(channels), 1, 1)
        self.register_buffer("identity_kernel", identity, persistent=False)
        nn.init.zeros_(self.weight)

    def actual_kernel(self) -> Tensor:
        return self.weight + self.identity_kernel.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.actual_kernel())


class BatchNormAndPad2d(nn.Module):
    """Official BN-aware border padding for a fusable sequential branch."""

    def __init__(self, channels: int, pad_pixels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(int(channels))
        self.pad_pixels = int(pad_pixels)
        if self.pad_pixels < 0:
            raise ValueError("DBB pad_pixels must be non-negative.")

    @property
    def weight(self) -> Tensor:
        if self.norm.weight is None:
            raise RuntimeError("DBB BatchNormAndPad2d requires affine normalization.")
        return self.norm.weight

    @property
    def bias(self) -> Tensor:
        if self.norm.bias is None:
            raise RuntimeError("DBB BatchNormAndPad2d requires affine normalization.")
        return self.norm.bias

    @property
    def running_mean(self) -> Tensor:
        return self.norm.running_mean

    @property
    def running_var(self) -> Tensor:
        return self.norm.running_var

    @property
    def eps(self) -> float:
        return float(self.norm.eps)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.norm(inputs)
        if self.pad_pixels == 0:
            return outputs
        scale = self.weight.detach() / torch.sqrt(
            self.running_var + self.norm.eps
        )
        pad_values = self.bias.detach() - self.running_mean * scale
        outputs = F.pad(outputs, [self.pad_pixels] * 4)
        pad_values = pad_values.reshape(1, -1, 1, 1)
        outputs[:, :, : self.pad_pixels, :] = pad_values
        outputs[:, :, -self.pad_pixels :, :] = pad_values
        outputs[:, :, :, : self.pad_pixels] = pad_values
        outputs[:, :, :, -self.pad_pixels :] = pad_values
        return outputs


def _conv_norm(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int,
    padding: int,
) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            (
                (
                    "conv",
                    nn.Conv2d(
                        int(in_channels),
                        int(out_channels),
                        kernel_size=int(kernel_size),
                        stride=int(stride),
                        padding=int(padding),
                        bias=False,
                    ),
                ),
                ("norm", nn.BatchNorm2d(int(out_channels))),
            )
        )
    )


class DiverseBranchConvBN(nn.Module):
    """Locked dense four-path DBB that deploys as one 3x3 convolution."""

    branch_names = ("origin", "pointwise", "sequential", "average")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.deploy = bool(deploy)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("DBB channels must be positive.")

        if self.deploy:
            self.reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            return

        self.dbb_origin = _conv_norm(
            self.in_channels,
            self.out_channels,
            3,
            stride=1,
            padding=1,
        )
        self.dbb_1x1 = _conv_norm(
            self.in_channels,
            self.out_channels,
            1,
            stride=1,
            padding=0,
        )
        self.dbb_1x1_kxk = nn.Sequential(
            OrderedDict(
                (
                    ("idconv1", IdentityBasedConv1x1(self.in_channels)),
                    ("norm1", BatchNormAndPad2d(self.in_channels, pad_pixels=1)),
                    (
                        "conv2",
                        nn.Conv2d(
                            self.in_channels,
                            self.out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("norm2", nn.BatchNorm2d(self.out_channels)),
                )
            )
        )
        self.dbb_avg = nn.Sequential(
            OrderedDict(
                (
                    (
                        "conv",
                        nn.Conv2d(
                            self.in_channels,
                            self.out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("norm", BatchNormAndPad2d(self.out_channels, pad_pixels=1)),
                    ("avg", nn.AvgPool2d(kernel_size=3, stride=1, padding=0)),
                    ("avg_norm", nn.BatchNorm2d(self.out_channels)),
                )
            )
        )

    def branch_outputs(self, inputs: Tensor) -> Dict[str, Tensor]:
        if hasattr(self, "reparam"):
            return {"deployed": self.reparam(inputs)}
        return {
            "origin": self.dbb_origin(inputs),
            "pointwise": self.dbb_1x1(inputs),
            "sequential": self.dbb_1x1_kxk(inputs),
            "average": self.dbb_avg(inputs),
        }

    def forward_with_branches(self, inputs: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        branches = self.branch_outputs(inputs)
        outputs = next(iter(branches.values()))
        for name in tuple(branches)[1:]:
            outputs = outputs + branches[name]
        return outputs, branches

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, "reparam"):
            return self.reparam(inputs)
        outputs, _ = self.forward_with_branches(inputs)
        return outputs

    def equivalent_path_kernels(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        if hasattr(self, "reparam"):
            if self.reparam.bias is None:
                raise RuntimeError("Deployed DBB convolution must have a bias.")
            return {"deployed": (self.reparam.weight, self.reparam.bias)}

        origin = fuse_conv_batch_norm(
            self.dbb_origin.conv.weight,
            self.dbb_origin.norm,
        )
        pointwise_kernel, pointwise_bias = fuse_conv_batch_norm(
            self.dbb_1x1.conv.weight,
            self.dbb_1x1.norm,
        )
        pointwise = (
            pad_kernel_to(pointwise_kernel, self.kernel_size),
            pointwise_bias,
        )

        first_kernel, first_bias = fuse_conv_batch_norm(
            self.dbb_1x1_kxk.idconv1.actual_kernel(),
            self.dbb_1x1_kxk.norm1,
        )
        second_kernel, second_bias = fuse_conv_batch_norm(
            self.dbb_1x1_kxk.conv2.weight,
            self.dbb_1x1_kxk.norm2,
        )
        sequential = compose_1x1_kxk(
            first_kernel,
            first_bias,
            second_kernel,
            second_bias,
        )

        avg_kernel = average_pool_kernel(
            self.out_channels,
            self.kernel_size,
            device=self.dbb_avg.avg_norm.weight.device,
            dtype=self.dbb_avg.avg_norm.weight.dtype,
        )
        avg_kernel, avg_bias = fuse_conv_batch_norm(
            avg_kernel,
            self.dbb_avg.avg_norm,
        )
        avg_first_kernel, avg_first_bias = fuse_conv_batch_norm(
            self.dbb_avg.conv.weight,
            self.dbb_avg.norm,
        )
        average = compose_1x1_kxk(
            avg_first_kernel,
            avg_first_bias,
            avg_kernel,
            avg_bias,
        )
        return {
            "origin": origin,
            "pointwise": pointwise,
            "sequential": sequential,
            "average": average,
        }

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        paths = self.equivalent_path_kernels()
        return add_equivalent_branches(
            tuple(paths[name][0] for name in paths),
            tuple(paths[name][1] for name in paths),
        )

    def switch_to_deploy(self) -> None:
        if hasattr(self, "reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.dbb_origin
        del self.dbb_1x1
        del self.dbb_1x1_kxk
        del self.dbb_avg
        self.deploy = True

    def reset_candidate_only_parameters(
        self,
        initializer: Callable[[nn.Module], None],
    ) -> None:
        if hasattr(self, "reparam"):
            raise RuntimeError("Cannot initialize candidate paths after DBB deployment.")
        self.dbb_1x1.apply(initializer)
        self.dbb_1x1_kxk.apply(initializer)
        self.dbb_avg.apply(initializer)
        nn.init.zeros_(self.dbb_1x1_kxk.idconv1.weight)

    def terminal_norms(self) -> Mapping[str, nn.BatchNorm2d]:
        if hasattr(self, "reparam"):
            return {}
        return {
            "origin": self.dbb_origin.norm,
            "pointwise": self.dbb_1x1.norm,
            "sequential": self.dbb_1x1_kxk.norm2,
            "average": self.dbb_avg.avg_norm,
        }


class DiverseBranchStemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dbb = DiverseBranchConvBN(in_channels, out_channels)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_with_trace(self, inputs: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        pre_activation, branches = self.dbb.forward_with_branches(inputs)
        outputs = self.pool(self.activation(pre_activation))
        trace = {f"path_{name}": value for name, value in branches.items()}
        trace["pre_activation"] = pre_activation
        trace["output"] = outputs
        return outputs, trace

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pool(self.activation(self.dbb(inputs)))


class DiverseBranchConvStem(nn.Module):
    """Three-block DBB replacement for the legacy stride-8 conv-pool stem."""

    source_commit = "8d2b16b6aee45a33236b2d11685be6857f9ba929"
    source_tree = "b038d8e014664312f8979d0b7917466e70877b30"
    source_sha256 = "5f67f20081ffef0a9f7ad6f9f5cc1a7bb4e59902dc31c86a8e59c37fd477ccb0"
    downsample_factor = 8

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        mid_channels = int(stem_channels) * 2
        self.out_channels = int(embed_dim)
        self.blocks = nn.ModuleList(
            (
                DiverseBranchStemBlock(int(in_channels), int(stem_channels)),
                DiverseBranchStemBlock(int(stem_channels), mid_channels),
                DiverseBranchStemBlock(mid_channels, self.out_channels),
            )
        )

    def forward_with_trace(self, inputs: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        outputs = inputs
        trace: Dict[str, Tensor] = {}
        for index, block in enumerate(self.blocks, start=1):
            outputs, block_trace = block.forward_with_trace(outputs)
            for name, value in block_trace.items():
                trace[f"block{index}_{name}"] = value
        trace["stem_output"] = outputs
        return outputs, trace

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return outputs

    def reset_origin_from_control(self, control_stem: nn.Module) -> None:
        control_blocks = getattr(control_stem, "blocks", None)
        if control_blocks is None or len(control_blocks) != len(self.blocks):
            raise ValueError("DBB origin reset requires a three-block control stem.")
        for candidate_block, control_block in zip(self.blocks, control_blocks):
            control_sequence = getattr(control_block, "block", None)
            if control_sequence is None:
                raise ValueError("DBB origin reset requires legacy ConvStemBlock inputs.")
            candidate_block.dbb.dbb_origin.conv.load_state_dict(
                control_sequence.conv.state_dict(), strict=True
            )
            candidate_block.dbb.dbb_origin.norm.load_state_dict(
                control_sequence.norm.state_dict(), strict=True
            )

    def reset_candidate_only_parameters(
        self,
        initializer: Callable[[nn.Module], None],
    ) -> None:
        for block in self.blocks:
            block.dbb.reset_candidate_only_parameters(initializer)

    def switch_to_deploy(self) -> None:
        for block in self.blocks:
            block.dbb.switch_to_deploy()

    @property
    def is_deployed(self) -> bool:
        return all(hasattr(block.dbb, "reparam") for block in self.blocks)


def convert_dbb_modules_to_deploy(module: nn.Module, *, inplace: bool = False) -> nn.Module:
    converted = module if inplace else copy.deepcopy(module)
    for child in tuple(converted.modules()):
        if isinstance(child, DiverseBranchConvBN):
            child.switch_to_deploy()
    return converted
