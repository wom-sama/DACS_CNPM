from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FeatureHookSpec:
    module: nn.Module
    source: str


@dataclass
class AttentionHookSpec:
    module: nn.Module
    num_heads: int
    scale: float
    prefix_tokens: int
    source: str


class HookRecorder:
    def __init__(
        self,
        feature_spec: FeatureHookSpec,
        attention_spec: Optional[AttentionHookSpec] = None,
    ) -> None:
        self.feature_spec = feature_spec
        self.attention_spec = attention_spec
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self.qkv_output: Optional[Tensor] = None
        self.grid_size: Optional[Tuple[int, int]] = None
        self._handles = []

    def __enter__(self) -> "HookRecorder":
        self._handles.append(self.feature_spec.module.register_forward_hook(self._forward_hook))
        self._handles.append(self.feature_spec.module.register_full_backward_hook(self._backward_hook))
        if self.attention_spec is not None:
            self._handles.append(self.attention_spec.module.register_forward_hook(self._attention_hook))
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _forward_hook(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output):
            self.activations = output
            if output.ndim >= 4:
                self.grid_size = (int(output.shape[-2]), int(output.shape[-1]))

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        if grad_output and torch.is_tensor(grad_output[0]):
            self.gradients = grad_output[0].detach()

    def _attention_hook(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output):
            self.qkv_output = output.detach()


def count_attention_layers(model: nn.Module) -> int:
    backbone = getattr(model, "frame_model", model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return 0
    return sum(1 for block in blocks if hasattr(block, "attn") and hasattr(block.attn, "qkv"))


def resolve_attention_index(model: nn.Module, layer_index: int) -> Optional[int]:
    depth = count_attention_layers(model)
    if depth == 0:
        return None
    resolved = layer_index if layer_index >= 0 else depth + layer_index
    if resolved < 0 or resolved >= depth:
        raise ValueError(f"Layer index khong hop le: {layer_index} voi depth={depth}")
    return resolved


def resolve_feature_hook(model: nn.Module) -> FeatureHookSpec:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        return FeatureHookSpec(module=model.patch_embed.proj, source="patch_embed.proj")

    if hasattr(model, "layer4"):
        layer4 = model.layer4
        if len(layer4) > 0 and hasattr(layer4[-1], "conv3"):
            return FeatureHookSpec(module=layer4[-1].conv3, source="layer4[-1].conv3")

    last_conv = None
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name
    if last_conv is None or last_name is None:
        raise TypeError("Khong tim thay Conv2d phu hop de trich xuat feature map.")
    return FeatureHookSpec(module=last_conv, source=last_name)


def resolve_attention_hook(model: nn.Module, layer_index: int) -> Optional[AttentionHookSpec]:
    backbone = getattr(model, "frame_model", model)
    resolved_index = resolve_attention_index(model, layer_index)
    if resolved_index is None:
        return None

    attn = backbone.blocks[resolved_index].attn
    qkv_module = getattr(attn, "qkv", None)
    if qkv_module is None or not hasattr(attn, "num_heads"):
        return None

    head_dim = getattr(attn, "head_dim", None)
    scale = getattr(attn, "scale", None)
    if scale is None:
        if head_dim is None:
            raise TypeError("Khong the suy ra attention scale tu module attn.")
        scale = float(head_dim) ** -0.5
    prefix_tokens = int(
        getattr(backbone, "num_prefix_tokens", 1 + int(getattr(backbone, "num_registers", 0)))
    )
    return AttentionHookSpec(
        module=qkv_module,
        num_heads=int(attn.num_heads),
        scale=float(scale),
        prefix_tokens=prefix_tokens,
        source=f"blocks[{resolved_index}].attn.qkv",
    )


def reconstruct_attention(qkv_output: Tensor, spec: AttentionHookSpec) -> Tensor:
    if qkv_output.ndim != 3:
        raise ValueError("Tensor qkv hook phai co shape (B, N, 3 * D).")
    batch_size, num_tokens, total_dim = qkv_output.shape
    head_dim = total_dim // (3 * spec.num_heads)
    qkv = qkv_output.reshape(batch_size, num_tokens, 3, spec.num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key = qkv[0], qkv[1]
    attention = (query @ key.transpose(-2, -1)) * spec.scale
    return attention.softmax(dim=-1)


def _normalize_heatmap(heatmap: Tensor) -> np.ndarray:
    heatmap = heatmap.clamp(min=0)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max().clamp(min=1e-8)
    return heatmap.detach().cpu().numpy()


def build_attention_heatmap(
    attention: Tensor,
    grid_size: Tuple[int, int],
    prefix_tokens: int,
    reduction: str,
    output_size: Tuple[int, int],
    query_tokens: str = "cls",
) -> np.ndarray:
    query_tokens = str(query_tokens or "cls").strip().lower()
    if query_tokens == "cls":
        query_indices = [0]
    elif query_tokens in {"register", "registers"}:
        query_indices = list(range(1, max(1, int(prefix_tokens))))
        if not query_indices:
            query_indices = [0]
    elif query_tokens in {"cls_register_mean", "cls_registers", "head"}:
        query_indices = list(range(0, max(1, int(prefix_tokens))))
    else:
        raise ValueError(
            "query_tokens chi ho tro cls, registers, hoac cls_register_mean."
        )

    query_to_patch = attention[:, query_indices, prefix_tokens:]
    if reduction == "max":
        patch_attention = query_to_patch.flatten(0, 1).max(dim=0).values
    else:
        patch_attention = query_to_patch.mean(dim=(0, 1))

    heatmap = patch_attention.reshape(grid_size[0], grid_size[1]).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=(output_size[1], output_size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalize_heatmap(heatmap)


def build_featuremap_heatmap(
    activations: Tensor,
    output_size: Tuple[int, int],
    reduction: str = "mean",
) -> np.ndarray:
    if activations.ndim != 4:
        raise ValueError("Feature-map heatmap yeu cau activation 4D (B, C, H, W).")
    feature_map = activations[0].detach()
    if reduction == "max":
        heatmap = feature_map.max(dim=0).values
    else:
        heatmap = feature_map.mean(dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=(output_size[1], output_size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalize_heatmap(heatmap)


def build_gradcam_heatmap(
    activations: Tensor,
    gradients: Tensor,
    output_size: Tuple[int, int],
) -> np.ndarray:
    if activations.ndim != 4 or gradients.ndim != 4:
        raise ValueError("Grad-CAM yeu cau activation/gradient 4D.")
    weights = gradients[0].mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=0, keepdim=True))
    cam = cam.unsqueeze(0)
    cam = F.interpolate(
        cam,
        size=(output_size[1], output_size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalize_heatmap(cam)


def extract_optional_token_features(model: nn.Module, tensor: Tensor) -> Optional[Dict[str, Tensor]]:
    if not hasattr(model, "forward_features"):
        return None
    with torch.no_grad():
        features = model.forward_features(tensor)
    if isinstance(features, dict) and "patches" in features and "registers" in features:
        return features
    return None
