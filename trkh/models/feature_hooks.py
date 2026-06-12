from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

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


def _last_conv_in_module(module: nn.Module) -> Tuple[Optional[nn.Module], Optional[str]]:
    last_conv = None
    last_name = None
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
            last_name = name
    return last_conv, last_name


def resolve_feature_hook(model: nn.Module, feature_source: str = "auto") -> FeatureHookSpec:
    feature_source = str(feature_source or "auto").strip().lower()
    backbone = getattr(model, "frame_model", model)
    if feature_source not in {"auto", "patch_embed", "patch_embed.proj", "stem_last", "last_conv"}:
        raise ValueError("feature_source chi ho tro auto, patch_embed, stem_last, last_conv.")

    if feature_source in {"auto", "patch_embed", "patch_embed.proj"}:
        if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "proj"):
            return FeatureHookSpec(module=backbone.patch_embed.proj, source="patch_embed.proj")
        if feature_source in {"patch_embed", "patch_embed.proj"}:
            raise TypeError("Khong tim thay patch_embed.proj de hook Grad-CAM.")

    if feature_source == "stem_last":
        stem = getattr(backbone, "stem", None)
        if stem is None:
            raise TypeError("Model khong co CNN stem de hook stem_last.")
        stem_conv, stem_name = _last_conv_in_module(stem)
        if stem_conv is None or stem_name is None:
            raise TypeError("Khong tim thay Conv2d trong CNN stem.")
        return FeatureHookSpec(module=stem_conv, source=f"stem.{stem_name}")

    if hasattr(model, "layer4"):
        layer4 = model.layer4
        if len(layer4) > 0 and hasattr(layer4[-1], "conv3"):
            return FeatureHookSpec(module=layer4[-1].conv3, source="layer4[-1].conv3")

    last_conv, last_name = _last_conv_in_module(backbone)
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


def _resolve_query_indices(query_tokens: str, prefix_tokens: int) -> list[int]:
    query_tokens = str(query_tokens or "cls").strip().lower()
    if query_tokens == "cls":
        return [0]
    if query_tokens in {"register", "registers"}:
        return list(range(1, max(1, int(prefix_tokens)))) or [0]
    if query_tokens in {"cls_register_mean", "cls_registers", "head"}:
        return list(range(0, max(1, int(prefix_tokens))))
    raise ValueError("query_tokens chi ho tro cls, registers, hoac cls_register_mean.")


def build_attention_heatmap(
    attention: Tensor,
    grid_size: Tuple[int, int],
    prefix_tokens: int,
    reduction: str,
    output_size: Tuple[int, int],
    query_tokens: str = "cls",
) -> np.ndarray:
    query_indices = _resolve_query_indices(query_tokens, prefix_tokens)

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


def build_attention_rollout_heatmap(
    attentions: Union[Dict[int, Tensor], Sequence[Tensor]],
    grid_size: Tuple[int, int],
    prefix_tokens: int,
    output_size: Tuple[int, int],
    query_tokens: str = "cls_register_mean",
    start_layer: int = 0,
) -> np.ndarray:
    if isinstance(attentions, dict):
        ordered = [attentions[index] for index in sorted(attentions)]
    else:
        ordered = list(attentions)
    if not ordered:
        raise ValueError("Attention rollout yeu cau it nhat mot attention map.")

    start_layer = max(0, int(start_layer))
    selected = ordered[start_layer:] or ordered
    device = selected[0].device
    num_tokens = int(selected[0].shape[-1])
    rollout = torch.eye(num_tokens, device=device, dtype=selected[0].dtype)
    for attention in selected:
        if attention.ndim == 4:
            attention = attention[0]
        if attention.ndim != 3:
            raise ValueError("Attention map phai co shape [heads, tokens, tokens] hoac [B, heads, tokens, tokens].")
        fused = attention.mean(dim=0)
        fused = fused + torch.eye(num_tokens, device=fused.device, dtype=fused.dtype)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = fused @ rollout

    query_indices = _resolve_query_indices(query_tokens, prefix_tokens)

    patch_relevance = rollout[query_indices, prefix_tokens:].mean(dim=0)
    heatmap = patch_relevance.reshape(grid_size[0], grid_size[1]).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=(output_size[1], output_size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalize_heatmap(heatmap)


def build_gradient_weighted_attention_rollout_heatmap(
    attentions: Union[Dict[int, Tensor], Sequence[Tensor]],
    grid_size: Tuple[int, int],
    prefix_tokens: int,
    output_size: Tuple[int, int],
    query_tokens: str = "cls_register_mean",
    start_layer: int = 0,
) -> np.ndarray:
    if isinstance(attentions, dict):
        ordered = [attentions[index] for index in sorted(attentions)]
    else:
        ordered = list(attentions)
    if not ordered:
        raise ValueError("Gradient-weighted rollout yeu cau it nhat mot attention map.")

    start_layer = max(0, int(start_layer))
    selected = ordered[start_layer:] or ordered
    device = selected[0].device
    num_tokens = int(selected[0].shape[-1])
    rollout = torch.eye(num_tokens, device=device, dtype=selected[0].dtype)
    for attention in selected:
        if attention.ndim == 4:
            attention_for_grad = attention[0]
        elif attention.ndim == 3:
            attention_for_grad = attention
        else:
            raise ValueError("Attention map phai co shape [heads,tokens,tokens] hoac [B,heads,tokens,tokens].")

        gradient = attention.grad
        if gradient is not None and gradient.ndim == 4:
            gradient = gradient[0]
        if gradient is None:
            weighted = attention_for_grad
        else:
            weighted = torch.relu(gradient) * attention_for_grad
            if float(weighted.detach().sum().abs().item()) <= 1e-12:
                weighted = attention_for_grad

        fused = weighted.mean(dim=0).clamp(min=0)
        fused = fused + torch.eye(num_tokens, device=fused.device, dtype=fused.dtype)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = fused @ rollout

    query_indices = _resolve_query_indices(query_tokens, prefix_tokens)
    patch_relevance = rollout[query_indices, prefix_tokens:].mean(dim=0)
    heatmap = patch_relevance.reshape(grid_size[0], grid_size[1]).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=(output_size[1], output_size[0]),
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    return _normalize_heatmap(heatmap)


def _entropy_1d(values: Tensor) -> float:
    values = values.detach().float().clamp(min=0)
    total = values.sum().clamp(min=1e-12)
    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp(min=1e-12))).sum()
    return float((entropy / torch.log(torch.tensor(float(max(2, values.numel()))))).item())


def summarize_register_attention(
    attentions: Union[Dict[int, Tensor], Sequence[Tensor]],
    prefix_tokens: int,
    register_prefix_tokens: Optional[int] = None,
) -> Dict[str, float]:
    if isinstance(attentions, dict):
        ordered = [attentions[index] for index in sorted(attentions)]
    else:
        ordered = list(attentions)
    if not ordered:
        return {}
    attention = ordered[-1]
    if attention.ndim == 4:
        attention = attention[0]
    if attention.ndim != 3:
        return {}

    prefix_tokens = int(prefix_tokens)
    register_prefix_tokens = int(register_prefix_tokens or prefix_tokens)
    register_prefix_tokens = max(1, min(register_prefix_tokens, prefix_tokens))
    patch_attention = attention[:, :, prefix_tokens:]
    if patch_attention.numel() == 0:
        return {}

    cls_patch = patch_attention[:, 0, :].mean(dim=0)
    result: Dict[str, float] = {
        "cls_to_patch_attention_mean": float(cls_patch.mean().item()),
        "cls_attention_entropy": _entropy_1d(cls_patch),
    }
    if register_prefix_tokens > 1:
        register_patch = patch_attention[:, 1:register_prefix_tokens, :].mean(dim=(0, 1))
        similarity = F.cosine_similarity(
            cls_patch.flatten().unsqueeze(0),
            register_patch.flatten().unsqueeze(0),
            dim=1,
        )[0]
        result.update(
            {
                "register_to_patch_attention_mean": float(register_patch.mean().item()),
                "register_attention_entropy": _entropy_1d(register_patch),
                "cls_register_heatmap_similarity": float(similarity.item()),
                "register_to_cls_attention_mean": float(attention[:, 1:register_prefix_tokens, 0].mean().item()),
                "register_to_register_attention_mean": float(
                    attention[:, 1:register_prefix_tokens, 1:register_prefix_tokens].mean().item()
                ),
            }
        )
    else:
        result.update(
            {
                "register_to_patch_attention_mean": 0.0,
                "register_attention_entropy": 0.0,
                "cls_register_heatmap_similarity": 0.0,
                "register_to_cls_attention_mean": 0.0,
                "register_to_register_attention_mean": 0.0,
            }
        )
    return result


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
