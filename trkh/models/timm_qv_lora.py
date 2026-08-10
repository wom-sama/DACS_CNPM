from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SUPPORTED_TIMM_QV_LORA_MODELS = frozenset(
    {"vit_small_patch16_dinov3.lvd1689m"}
)


class LowRankUpdate(nn.Module):
    """Bias-free low-rank branch with the standard LoRA zero-up start."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        dropout: float,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if int(in_features) <= 0 or int(out_features) <= 0 or int(rank) <= 0:
            raise ValueError("LoRA feature dimensions and rank must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.Linear(
            int(in_features),
            int(rank),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up = nn.Linear(
            int(rank),
            int(out_features),
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.up(self.down(self.dropout(inputs)))

    def merged_weight(self) -> Tensor:
        return self.up.weight @ self.down.weight


class TimmQKVLoRALinear(nn.Linear):
    """Fused QKV linear with mergeable rank-r updates on Q and V only."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if int(out_features) != 3 * int(in_features):
            raise ValueError(
                "TimmQKVLoRALinear requires fused QKV width 3 * in_features"
            )
        if float(alpha) <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        super().__init__(
            int(in_features),
            int(out_features),
            bias=bool(bias),
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.q_lora = LowRankUpdate(
            int(in_features),
            int(in_features),
            int(rank),
            float(dropout),
            device=device,
            dtype=dtype,
        )
        self.v_lora = LowRankUpdate(
            int(in_features),
            int(in_features),
            int(rank),
            float(dropout),
            device=device,
            dtype=dtype,
        )
        # This deliberately stays a Python boolean. Reading a CUDA tensor via
        # .item() in every attention forward would synchronize the GPU and
        # would also leak control flow into graph export.
        self._lora_merged = False

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "TimmQKVLoRALinear":
        target = cls(
            int(source.in_features),
            int(source.out_features),
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            target.weight.copy_(source.weight)
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(source.bias)
        target.train(source.training)
        return target

    @property
    def lora_merged(self) -> bool:
        return bool(self._lora_merged)

    def forward(self, inputs: Tensor) -> Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        if self.lora_merged:
            return base
        width = int(self.in_features)
        query, key, value = base.split(width, dim=-1)
        query = query + self.scaling * self.q_lora(inputs)
        value = value + self.scaling * self.v_lora(inputs)
        return torch.cat((query, key, value), dim=-1)

    @torch.no_grad()
    def merge_lora_(self) -> "TimmQKVLoRALinear":
        if self.lora_merged:
            return self
        if self.training:
            raise RuntimeError("Set the model to eval() before merging LoRA weights")
        width = int(self.in_features)
        self.weight[:width].add_(
            self.q_lora.merged_weight().to(
                device=self.weight.device, dtype=self.weight.dtype
            ),
            alpha=float(self.scaling),
        )
        self.weight[2 * width :].add_(
            self.v_lora.merged_weight().to(
                device=self.weight.device, dtype=self.weight.dtype
            ),
            alpha=float(self.scaling),
        )
        self._lora_merged = True
        return self

    @torch.no_grad()
    def unmerge_lora_(self) -> "TimmQKVLoRALinear":
        if not self.lora_merged:
            return self
        width = int(self.in_features)
        self.weight[:width].sub_(
            self.q_lora.merged_weight().to(
                device=self.weight.device, dtype=self.weight.dtype
            ),
            alpha=float(self.scaling),
        )
        self.weight[2 * width :].sub_(
            self.v_lora.merged_weight().to(
                device=self.weight.device, dtype=self.weight.dtype
            ),
            alpha=float(self.scaling),
        )
        self._lora_merged = False
        return self

    def train(self, mode: bool = True):
        if bool(mode) and self.lora_merged:
            self.unmerge_lora_()
        return super().train(mode)


def parse_lora_layers(value: object) -> List[int]:
    if isinstance(value, str):
        raw_values: Iterable[object] = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        raw_values = (value,)
    layers: List[int] = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        index = int(text)
        if index not in layers:
            layers.append(index)
    if not layers:
        raise ValueError("TIMM Q/V LoRA requires at least one block index")
    return layers


def timm_qv_lora_parameter_prefixes(layers: object) -> List[str]:
    return [
        f"blocks.{index}.attn.qkv.{branch}_lora"
        for index in parse_lora_layers(layers)
        for branch in ("q", "v")
    ]


def inject_timm_qv_lora(
    model: nn.Module,
    *,
    layers: object,
    rank: int,
    alpha: float,
    dropout: float,
) -> Dict[str, object]:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)):
        raise TypeError("TIMM Q/V LoRA requires a model.blocks sequence")
    layer_indices = parse_lora_layers(layers)
    invalid = [index for index in layer_indices if index < 0 or index >= len(blocks)]
    if invalid:
        raise ValueError(
            f"TIMM Q/V LoRA block indices are outside [0,{len(blocks) - 1}]: {invalid}"
        )
    if int(rank) <= 0 or float(alpha) <= 0.0:
        raise ValueError("TIMM Q/V LoRA rank and alpha must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("TIMM Q/V LoRA dropout must be in [0, 1)")

    parameter_count = 0
    wrapped: List[str] = []
    for index in layer_indices:
        attention = getattr(blocks[index], "attn", None)
        qkv = getattr(attention, "qkv", None)
        separate_bias_names = [
            name
            for name in ("q_bias", "k_bias", "v_bias")
            if getattr(attention, name, None) is not None
        ]
        if separate_bias_names:
            raise TypeError(
                "TIMM Q/V LoRA refuses attention modules with separate Q/K/V "
                "bias tensors because some EVA implementations bypass "
                f"qkv.forward: block={index}, biases={separate_bias_names}"
            )
        if isinstance(qkv, TimmQKVLoRALinear):
            if (
                int(qkv.rank) != int(rank)
                or float(qkv.alpha) != float(alpha)
                or float(qkv.q_lora.dropout.p) != float(dropout)
            ):
                raise ValueError(
                    f"Block {index} already has a different Q/V LoRA configuration"
                )
            wrapped.append(f"blocks.{index}.attn.qkv")
            parameter_count += sum(
                int(parameter.numel())
                for name, parameter in qkv.named_parameters()
                if name.startswith(("q_lora.", "v_lora."))
            )
            continue
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Block {index} attention does not expose a linear qkv")
        replacement = TimmQKVLoRALinear.from_linear(
            qkv,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
        )
        attention.qkv = replacement
        wrapped.append(f"blocks.{index}.attn.qkv")
        parameter_count += sum(
            int(parameter.numel())
            for name, parameter in replacement.named_parameters()
            if name.startswith(("q_lora.", "v_lora."))
        )

    summary = {
        "enabled": True,
        "layers": layer_indices,
        "rank": int(rank),
        "alpha": float(alpha),
        "scaling": float(alpha) / float(rank),
        "dropout": float(dropout),
        "wrapped_modules": wrapped,
        "lora_parameter_count": int(parameter_count),
        "inference_mergeable": True,
    }
    model.timm_qv_lora_enabled = True
    model.timm_qv_lora_summary = dict(summary)
    return summary


def iter_timm_qv_lora_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, TimmQKVLoRALinear):
            yield module


def _materialize_lora_children_(parent: nn.Module) -> List[str]:
    replaced: List[str] = []
    for child_name, child in list(parent.named_children()):
        if isinstance(child, TimmQKVLoRALinear):
            child.merge_lora_()
            replacement = nn.Linear(
                int(child.in_features),
                int(child.out_features),
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.weight.requires_grad_(child.weight.requires_grad)
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
                    replacement.bias.requires_grad_(child.bias.requires_grad)
            replacement.train(child.training)
            setattr(parent, child_name, replacement)
            replaced.append(child_name)
            continue
        for descendant_name in _materialize_lora_children_(child):
            replaced.append(f"{child_name}.{descendant_name}")
    return replaced


@torch.no_grad()
def merge_timm_qv_lora_(model: nn.Module) -> Dict[str, object]:
    model.eval()
    modules = list(iter_timm_qv_lora_modules(model))
    for module in modules:
        module.merge_lora_()
    return {
        "merged_modules": int(len(modules)),
        "all_merged": bool(modules) and all(module.lora_merged for module in modules),
    }


@torch.no_grad()
def materialize_timm_qv_lora_(model: nn.Module) -> Dict[str, object]:
    """Irreversibly replace merged LoRA wrappers with plain linear layers."""

    model.eval()
    source_module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, TimmQKVLoRALinear)
    ]
    declared_enabled = bool(getattr(model, "timm_qv_lora_enabled", False))
    declared_summary = getattr(model, "timm_qv_lora_summary", {})
    declared_module_names = (
        [
            str(name)
            for name in declared_summary.get("wrapped_modules", [])
        ]
        if isinstance(declared_summary, dict)
        else []
    )
    if declared_enabled and not source_module_names:
        raise RuntimeError(
            "Model declares TIMM Q/V LoRA but exposes no adapter modules."
        )
    if declared_module_names and source_module_names != declared_module_names:
        raise RuntimeError(
            "TIMM Q/V LoRA materialization module contract mismatch: "
            f"declared={declared_module_names}, actual={source_module_names}"
        )
    replaced_modules = _materialize_lora_children_(model)
    remaining_modules = list(iter_timm_qv_lora_modules(model))
    if replaced_modules != source_module_names or remaining_modules:
        raise RuntimeError(
            "TIMM Q/V LoRA materialization did not replace the exact source set: "
            f"source={source_module_names}, replaced={replaced_modules}, "
            f"remaining={len(remaining_modules)}"
        )
    if declared_enabled:
        model.timm_qv_lora_enabled = False
        model.timm_qv_lora_summary = {
            **dict(declared_summary),
            "enabled": False,
            "materialized": True,
            "source_modules": source_module_names,
            "wrapped_modules": [],
            "lora_parameter_count": 0,
            "deployment_checkpoint_roundtrippable": False,
        }
    return {
        "materialized_modules": int(len(replaced_modules)),
        "module_names": replaced_modules,
        "remaining_lora_modules": int(len(remaining_modules)),
        "source_dtype": (
            str(next(model.parameters()).dtype).replace("torch.", "")
            if any(True for _ in model.parameters())
            else None
        ),
    }


@torch.no_grad()
def unmerge_timm_qv_lora_(model: nn.Module) -> Dict[str, object]:
    modules = list(iter_timm_qv_lora_modules(model))
    for module in modules:
        module.unmerge_lora_()
    return {
        "unmerged_modules": int(len(modules)),
        "all_unmerged": all(not module.lora_merged for module in modules),
    }
