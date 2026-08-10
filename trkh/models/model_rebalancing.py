from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelRebalancingConvRecord:
    name: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    rank: int
    base_parameters: int
    tail_parameters: int


class ModelRebalancingConv2d(nn.Conv2d):
    """Conv2d with a mergeable low-rank MORE tail component."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        tail_rank: int,
        device=None,
        dtype=None,
    ) -> None:
        if int(groups) != 1:
            raise ValueError("MORE convolution currently requires groups=1 for exact merging.")
        if int(tail_rank) < 1:
            raise ValueError("tail_rank must be positive.")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.tail_rank = int(tail_rank)
        self.tail_a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.tail_rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=False,
            padding_mode=self.padding_mode,
            device=device,
            dtype=dtype,
        )
        self.tail_b = nn.Conv2d(
            in_channels=self.tail_rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self.tail_enabled = True

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        *,
        tail_rank: int,
        generator: Optional[torch.Generator] = None,
    ) -> "ModelRebalancingConv2d":
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("from_conv expects nn.Conv2d.")
        wrapped = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            tail_rank=int(tail_rank),
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            wrapped.weight.copy_(conv.weight)
            if conv.bias is not None and wrapped.bias is not None:
                wrapped.bias.copy_(conv.bias)
            nn.init.kaiming_uniform_(
                wrapped.tail_a.weight,
                a=math.sqrt(5.0),
                generator=generator,
            )
            wrapped.tail_b.weight.zero_()
        wrapped.train(conv.training)
        return wrapped

    def forward(self, inputs: Tensor) -> Tensor:
        general = self._conv_forward(inputs, self.weight, self.bias)
        if not self.tail_enabled:
            return general
        return general + self.tail_b(self.tail_a(inputs))

    def effective_tail_weight(self) -> Tensor:
        tail_b = self.tail_b.weight[:, :, 0, 0]
        return torch.einsum("or,rihw->oihw", tail_b, self.tail_a.weight)

    def merged_weight(self) -> Tensor:
        return self.weight + self.effective_tail_weight().to(dtype=self.weight.dtype)

    def to_plain_conv(self) -> nn.Conv2d:
        plain = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            plain.weight.copy_(self.merged_weight())
            if self.bias is not None and plain.bias is not None:
                plain.bias.copy_(self.bias)
        plain.train(self.training)
        return plain


def _resolve_parent_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = str(qualified_name).split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid module name: {qualified_name!r}")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(f"Parent path is not a module: {qualified_name!r}")
    return parent, parts[-1]


def _set_child_module(root: nn.Module, qualified_name: str, module: nn.Module) -> None:
    parent, child_name = _resolve_parent_module(root, qualified_name)
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _tail_rank(conv: nn.Conv2d, rank_ratio: float) -> int:
    ratio = float(rank_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("rank_ratio must be finite and in (0, 1].")
    flattened_input = int(conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1])
    return max(1, int(round(ratio * min(int(conv.out_channels), flattened_input))))


def model_rebalancing_modules(
    model: nn.Module,
) -> Dict[str, ModelRebalancingConv2d]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ModelRebalancingConv2d)
    }


def inject_model_rebalancing_convs(
    model: nn.Module,
    *,
    rank_ratio: float = 0.1,
    seed: int = 42,
    module_names: Optional[Sequence[str]] = None,
) -> list[ModelRebalancingConvRecord]:
    if model_rebalancing_modules(model):
        raise ValueError("Model already contains MORE convolutions.")
    requested = None if module_names is None else {str(name) for name in module_names}
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if type(module) is nn.Conv2d and (requested is None or name in requested)
    ]
    if requested is not None:
        found = {name for name, _ in candidates}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Requested MORE convolutions were not found: {missing}")
    if not candidates:
        raise ValueError("No Conv2d modules are available for MORE injection.")

    records: list[ModelRebalancingConvRecord] = []
    for module_index, (name, conv) in enumerate(candidates):
        if int(conv.groups) != 1:
            raise ValueError(f"Grouped convolution is not supported for exact MORE merge: {name}")
        rank = _tail_rank(conv, rank_ratio)
        generator_device = conv.weight.device if conv.weight.is_cuda else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed) + int(module_index))
        wrapped = ModelRebalancingConv2d.from_conv(
            conv,
            tail_rank=rank,
            generator=generator,
        )
        _set_child_module(model, name, wrapped)
        records.append(
            ModelRebalancingConvRecord(
                name=name,
                in_channels=int(conv.in_channels),
                out_channels=int(conv.out_channels),
                kernel_size=(int(conv.kernel_size[0]), int(conv.kernel_size[1])),
                rank=rank,
                base_parameters=int(conv.weight.numel())
                + (int(conv.bias.numel()) if conv.bias is not None else 0),
                tail_parameters=int(wrapped.tail_a.weight.numel())
                + int(wrapped.tail_b.weight.numel()),
            )
        )
    return records


@contextmanager
def model_rebalancing_general_only(model: nn.Module) -> Iterator[None]:
    modules = list(model_rebalancing_modules(model).values())
    previous = [bool(module.tail_enabled) for module in modules]
    try:
        for module in modules:
            module.tail_enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.tail_enabled = enabled


def freeze_model_rebalancing_base(model: nn.Module) -> list[nn.Parameter]:
    modules = model_rebalancing_modules(model)
    if not modules:
        raise ValueError("Model has no MORE convolutions.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tail_parameters: list[nn.Parameter] = []
    for module in modules.values():
        for parameter in module.tail_a.parameters():
            parameter.requires_grad_(True)
            tail_parameters.append(parameter)
        for parameter in module.tail_b.parameters():
            parameter.requires_grad_(True)
            tail_parameters.append(parameter)
    return tail_parameters


def merge_model_rebalancing_convs_(model: nn.Module) -> list[str]:
    modules = list(model_rebalancing_modules(model).items())
    if not modules:
        raise ValueError("Model has no MORE convolutions to merge.")
    for name, module in modules:
        _set_child_module(model, name, module.to_plain_conv())
    return [name for name, _ in modules]


def model_rebalancing_tail_sha256(model: nn.Module) -> str:
    modules = model_rebalancing_modules(model)
    if not modules:
        raise ValueError("Model has no MORE convolutions.")
    digest = hashlib.sha256()
    for name, module in sorted(modules.items()):
        for suffix, tensor in (
            ("tail_a.weight", module.tail_a.weight),
            ("tail_b.weight", module.tail_b.weight),
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(f"{name}.{suffix}:{tuple(contiguous.shape)}\n".encode("utf-8"))
            digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def more_sinusoidal_weight(
    step: int,
    total_steps: int,
    *,
    peak_amplitude: float,
) -> float:
    if int(total_steps) < 1:
        raise ValueError("total_steps must be positive.")
    if not 0 <= int(step) <= int(total_steps):
        raise ValueError("step must be in [0, total_steps].")
    peak = float(peak_amplitude)
    if not math.isfinite(peak) or peak < 0.0:
        raise ValueError("peak_amplitude must be finite and nonnegative.")
    return peak * math.sin(math.pi * float(step) / float(total_steps))


def more_discrepancy_loss(
    full_logits: Tensor,
    general_logits: Tensor,
    targets: Tensor,
    *,
    class_counts: Sequence[int] | Tensor,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    if full_logits.ndim != 2 or tuple(full_logits.shape) != tuple(general_logits.shape):
        raise ValueError("full_logits and general_logits must share shape [B, C].")
    if targets.ndim != 1 or int(targets.numel()) != int(full_logits.shape[0]):
        raise ValueError("targets must have shape [B].")
    if not torch.isfinite(full_logits).all() or not torch.isfinite(general_logits).all():
        raise ValueError("MORE logits must be finite.")
    counts = torch.as_tensor(
        class_counts,
        device=full_logits.device,
        dtype=full_logits.dtype,
    )
    if counts.ndim != 1 or int(counts.numel()) != int(full_logits.shape[1]):
        raise ValueError("class_counts must match the logit class dimension.")
    if not torch.isfinite(counts).all() or bool((counts <= 0).any().item()):
        raise ValueError("class_counts must contain finite positive values.")
    targets = targets.to(device=full_logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= int(counts.numel()))).any().item()):
        raise ValueError("targets contain an out-of-range class index.")
    class_priors = counts / counts.sum()
    per_sample_discrepancy = (full_logits - general_logits).square().sum(dim=1)
    sample_weights = class_priors.index_select(0, targets)
    loss = (sample_weights * per_sample_discrepancy).mean()
    return loss, {
        "class_priors": class_priors,
        "sample_weights": sample_weights,
        "per_sample_discrepancy": per_sample_discrepancy,
    }
