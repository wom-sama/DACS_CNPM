from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torchvision import models as tv_models

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size phai chia het cho patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.base_grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvStemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("norm", nn.BatchNorm2d(out_channels)),
                    ("act", nn.GELU()),
                    ("pool", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class HybridConvStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        mid_channels = stem_channels * 2
        self.blocks = nn.Sequential(
            ConvStemBlock(in_channels, stem_channels),
            ConvStemBlock(stem_channels, mid_channels),
            ConvStemBlock(mid_channels, embed_dim),
        )
        self.downsample_factor = 8
        self.out_channels = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class FineGrainedPatchPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        hidden_dim = max(32, int(dim))
        self.score = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(dim, dim),
        )

    def zero_init_residual(self) -> None:
        final_linear = self.fusion[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def forward(self, global_feature: Tensor, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.size(1) == 0:
            return global_feature
        normalized_patches = self.patch_norm(patch_tokens)
        context = self.context_norm(global_feature).unsqueeze(1).expand_as(normalized_patches)
        scores = self.score(torch.cat((normalized_patches, context), dim=-1)).squeeze(-1)
        attention = torch.softmax(scores, dim=1)
        patch_feature = torch.bmm(attention.unsqueeze(1), patch_tokens).squeeze(1)
        residual = self.fusion(torch.cat((global_feature, patch_feature), dim=-1))
        return global_feature + residual


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim phai chia het cho num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: Tensor, return_attention: bool = False):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        out = attention @ value
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        out = self.projection_dropout(out)
        if return_attention:
            return out, attention
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attention = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            return x, attention
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


TransformerBlock = CustomTransformerEncoderLayer


class VisionTransformerWithRegisters(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cnn_stem: bool = True,
        stem_channels: int = 32,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        register_positional_embedding: bool = False,
        fine_grained_pooling: bool = False,
        fine_grained_pooling_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        head_pooling: str = "cls_register_mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.register_positional_embedding = register_positional_embedding
        self.use_cnn_stem = use_cnn_stem
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.embed_dim = int(embed_dim)
        self.head_pooling = str(head_pooling).strip().lower()
        self.cnn_feature_fusion = bool(cnn_feature_fusion and use_cnn_stem)
        self.fine_grained_pooling = bool(fine_grained_pooling)
        if self.head_pooling not in {"cls", "cls_register_mean"}:
            raise ValueError(f"Khong ho tro head_pooling={head_pooling!r}.")

        if use_cnn_stem:
            self.stem = HybridConvStem(
                in_channels=in_channels,
                stem_channels=stem_channels,
                embed_dim=embed_dim,
            )
            stem_stride = self.stem.downsample_factor
            if image_size % stem_stride != 0:
                raise ValueError("image_size phai chia het cho downsample factor cua CNN stem")
            if patch_size % stem_stride != 0:
                raise ValueError("patch_size phai chia het cho downsample factor cua CNN stem")
            patch_embed_image_size = image_size // stem_stride
            patch_embed_patch_size = patch_size // stem_stride
            patch_embed_channels = self.stem.out_channels
        else:
            self.stem = nn.Identity()
            patch_embed_image_size = image_size
            patch_embed_patch_size = patch_size
            patch_embed_channels = in_channels

        self.patch_embed = PatchEmbedding(
            image_size=patch_embed_image_size,
            patch_size=patch_embed_patch_size,
            in_channels=patch_embed_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))

        pos_token_count = 1 + self.patch_embed.num_patches
        if register_positional_embedding:
            pos_token_count += num_registers
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_token_count, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_values[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if self.fine_grained_pooling:
            self.fine_grained_pool = FineGrainedPatchPooling(
                dim=embed_dim,
                dropout=fine_grained_pooling_dropout,
            )
        else:
            self.fine_grained_pool = None
        self.head = nn.Linear(embed_dim, num_classes)
        if self.cnn_feature_fusion:
            self.cnn_fusion_norm = nn.LayerNorm(embed_dim)
            self.cnn_fusion_dropout = nn.Dropout(float(max(0.0, cnn_fusion_dropout)))
            self.cnn_fusion_head = nn.Linear(embed_dim, num_classes)
        else:
            self.cnn_fusion_norm = nn.Identity()
            self.cnn_fusion_dropout = nn.Identity()
            self.cnn_fusion_head = None

        self.apply(self._init_weights)
        self._init_parameter_tensors()
        if self.fine_grained_pool is not None:
            self.fine_grained_pool.zero_init_residual()
        if self.cnn_fusion_head is not None:
            nn.init.zeros_(self.cnn_fusion_head.weight)
            if self.cnn_fusion_head.bias is not None:
                nn.init.zeros_(self.cnn_fusion_head.bias)

    def _init_parameter_tensors(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        return ("bias", "norm", "cls_token", "register_tokens", "pos_embed")

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _split_pos_embed(
        self,
        pos_embed: Tensor,
        has_register_positional_embedding: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cls_pos = pos_embed[:, :1]
        if has_register_positional_embedding:
            reg_pos = pos_embed[:, 1 : 1 + self.num_registers]
            patch_pos = pos_embed[:, 1 + self.num_registers :]
        else:
            reg_pos = pos_embed.new_zeros((pos_embed.shape[0], self.num_registers, pos_embed.shape[-1]))
            patch_pos = pos_embed[:, 1:]
        return cls_pos, reg_pos, patch_pos

    def _infer_source_register_mode(self, pos_embed: Tensor) -> Tuple[bool, int]:
        total_tokens = pos_embed.shape[1]
        patch_only_tokens = total_tokens - 1
        patch_with_register_tokens = total_tokens - 1 - self.num_registers

        patch_only_grid = int(round(math.sqrt(max(0, patch_only_tokens))))
        if patch_only_grid * patch_only_grid == patch_only_tokens:
            return False, patch_only_grid

        patch_with_register_grid = int(round(math.sqrt(max(0, patch_with_register_tokens))))
        if patch_with_register_grid * patch_with_register_grid == patch_with_register_tokens:
            return True, patch_with_register_grid

        raise ValueError("Khong the suy ra grid size tu pos_embed checkpoint.")

    def interpolate_external_pos_embed(self, pos_embed: Tensor) -> Tensor:
        source_has_register_pos, source_grid = self._infer_source_register_mode(pos_embed)
        cls_pos, reg_pos, patch_pos = self._split_pos_embed(
            pos_embed=pos_embed,
            has_register_positional_embedding=source_has_register_pos,
        )

        target_grid = self.patch_embed.base_grid_size
        if (source_grid, source_grid) != target_grid:
            patch_pos = patch_pos.reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos,
                size=target_grid,
                mode="bicubic",
                align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
                1,
                target_grid[0] * target_grid[1],
                -1,
            )

        if self.register_positional_embedding:
            if source_has_register_pos:
                target_reg_pos = reg_pos
            else:
                target_reg_pos = self.pos_embed[:, 1 : 1 + self.num_registers].detach().clone()
            return torch.cat((cls_pos, target_reg_pos, patch_pos), dim=1)

        return torch.cat((cls_pos, patch_pos), dim=1)

    def get_interpolated_pos_embed(self, grid_size: Tuple[int, int]) -> Tensor:
        cls_pos, reg_pos, patch_pos = self._split_pos_embed(
            pos_embed=self.pos_embed,
            has_register_positional_embedding=self.register_positional_embedding,
        )

        patch_pos = patch_pos.reshape(
            1,
            self.patch_embed.base_grid_size[0],
            self.patch_embed.base_grid_size[1],
            -1,
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1,
            grid_size[0] * grid_size[1],
            -1,
        )

        if self.register_positional_embedding:
            return torch.cat((cls_pos, reg_pos, patch_pos), dim=1)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def load_flexible_state_dict(self, state_dict, strict: bool = True):
        adapted_state = dict(state_dict)
        if "pos_embed" in adapted_state and adapted_state["pos_embed"].shape != self.pos_embed.shape:
            adapted_state["pos_embed"] = self.interpolate_external_pos_embed(adapted_state["pos_embed"])
        missing_keys, unexpected_keys = self.load_state_dict(adapted_state, strict=False)
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Loi load state_dict. missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return missing_keys, unexpected_keys

    def forward_features(
        self,
        x: Tensor,
        image_valid_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        attention_layers: Optional[Sequence[int]] = None,
    ) -> Dict[str, Tensor]:
        input_spatial_size = tuple(int(value) for value in x.shape[-2:])
        x = self.stem(x)
        stem_features = x
        batch_size = x.shape[0]
        grid_size = (
            x.shape[-2] // self.patch_embed.patch_size,
            x.shape[-1] // self.patch_embed.patch_size,
        )
        patch_tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        pos_embed = self.get_interpolated_pos_embed(grid_size).to(device=patch_tokens.device)

        if self.register_positional_embedding:
            tokens = torch.cat((cls_tokens, register_tokens, patch_tokens), dim=1)
            tokens = tokens + pos_embed
        else:
            cls_and_patches = torch.cat((cls_tokens, patch_tokens), dim=1)
            cls_and_patches = cls_and_patches + pos_embed
            tokens = torch.cat(
                (cls_and_patches[:, :1], register_tokens, cls_and_patches[:, 1:]),
                dim=1,
            )

        tokens = self.pos_drop(tokens)
        collect_all_attentions = return_attention and attention_layers is None
        attention_layer_set = set(attention_layers or [])
        attention_maps = {}
        for block_index, block in enumerate(self.blocks):
            if collect_all_attentions or block_index in attention_layer_set:
                tokens, attention = block(tokens, return_attention=True)
                attention_maps[block_index] = attention
            else:
                if self.gradient_checkpointing and self.training:
                    tokens = gradient_checkpoint(block, tokens, use_reentrant=False)
                else:
                    tokens = block(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        reg_out = tokens[:, 1 : 1 + self.num_registers]
        patch_out = tokens[:, 1 + self.num_registers :]
        features = {
            "cls": cls_out,
            "registers": reg_out,
            "patches": patch_out,
            "tokens": tokens,
            "grid_size": grid_size,
            "pooled": self.pool_tokens_for_head(cls_out, reg_out),
        }
        if self.cnn_feature_fusion:
            features["cnn_pooled"] = F.adaptive_avg_pool2d(stem_features, output_size=1).flatten(1)
        if image_valid_mask is not None:
            key_padding_mask = self._build_patch_key_padding_mask(
                image_valid_mask=image_valid_mask,
                input_spatial_size=input_spatial_size,
                grid_size=grid_size,
                device=patch_out.device,
            )
            features["memory_key_padding_mask"] = key_padding_mask
        if attention_maps:
            features["attentions"] = attention_maps
        return features

    def _build_patch_key_padding_mask(
        self,
        image_valid_mask: Tensor,
        input_spatial_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        if image_valid_mask.ndim == 3:
            mask = image_valid_mask.unsqueeze(1)
        elif image_valid_mask.ndim == 4:
            mask = image_valid_mask
        else:
            raise ValueError("image_valid_mask phai co shape [B,H,W] hoac [B,1,H,W].")
        mask = mask.to(device=device, dtype=torch.float32)
        if tuple(mask.shape[-2:]) != tuple(input_spatial_size):
            mask = F.interpolate(mask, size=input_spatial_size, mode="nearest")
        valid_fraction = F.interpolate(
            mask,
            size=grid_size,
            mode="area",
        ).flatten(1)
        key_padding_mask = valid_fraction <= 0.05
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked] = False
        return key_padding_mask

    def pool_tokens_for_head(self, cls_tokens: Tensor, register_tokens: Tensor) -> Tensor:
        if self.head_pooling == "cls" or register_tokens.numel() == 0:
            return cls_tokens
        pooled_tokens = torch.cat((cls_tokens.unsqueeze(1), register_tokens), dim=1)
        return pooled_tokens.mean(dim=1)

    def head_input_from_features(self, features: Dict[str, Tensor]) -> Tensor:
        if "pooled" in features:
            pooled = features["pooled"]
        else:
            pooled = self.pool_tokens_for_head(features["cls"], features["registers"])
        if self.fine_grained_pool is not None and "patches" in features:
            pooled = self.fine_grained_pool(pooled, features["patches"])
        return pooled

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        logits = self.head(self.head_input_from_features(features))
        if self.cnn_fusion_head is not None and "cnn_pooled" in features:
            cnn_features = self.cnn_fusion_norm(features["cnn_pooled"])
            cnn_features = self.cnn_fusion_dropout(cnn_features)
            logits = logits + self.cnn_fusion_head(cnn_features)
        return logits


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP yeu cau num_layers >= 1.")
        layers = []
        in_dim = int(input_dim)
        for layer_index in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SinePositionEmbedding2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2.0 * math.pi,
    ) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim cho positional embedding 2D phai chia het cho 4.")
        self.embed_dim = int(embed_dim)
        self.num_pos_feats = self.embed_dim // 2
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale)

    def forward(
        self,
        grid_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        y_embed = torch.arange(1, grid_height + 1, device=device, dtype=dtype).unsqueeze(1).repeat(1, grid_width)
        x_embed = torch.arange(1, grid_width + 1, device=device, dtype=dtype).unsqueeze(0).repeat(grid_height, 1)
        if self.normalize:
            y_embed = y_embed / max(1, grid_height) * self.scale
            x_embed = x_embed / max(1, grid_width) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2.0 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        position = torch.cat((pos_y, pos_x), dim=-1)
        return position.reshape(1, grid_height * grid_width, self.embed_dim)


class DETRDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        normalized_target = self.norm1(target)
        query = normalized_target + query_pos
        target2 = self.self_attn(query, query, value=normalized_target, need_weights=False)[0]
        target = target + self.dropout1(target2)

        normalized_target = self.norm2(target)
        target2 = self.cross_attn(
            query=normalized_target + query_pos,
            key=memory + memory_pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        target = target + self.dropout2(target2)

        normalized_target = self.norm3(target)
        target2 = self.linear2(self.dropout(F.gelu(self.linear1(normalized_target))))
        target = target + self.dropout3(target2)
        return target


class DETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DETRDecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_pos: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_intermediate: bool = False,
    ):
        output = target
        intermediate_outputs: List[Tensor] = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                query_pos,
                memory_pos,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if return_intermediate:
                intermediate_outputs.append(self.norm(output))
        final_output = self.norm(output)
        if return_intermediate:
            if intermediate_outputs:
                intermediate_outputs[-1] = final_output
            return final_output, intermediate_outputs
        return final_output


class PatchMemoryAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.depthwise = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(max(0.0, dropout)))

    def forward(self, tokens: Tensor, grid_size: Tuple[int, int]) -> Tensor:
        grid_height, grid_width = [max(1, int(value)) for value in grid_size]
        batch_size, token_count, embed_dim = tokens.shape
        if token_count != grid_height * grid_width:
            return tokens
        normalized = self.norm(tokens)
        grid = normalized.transpose(1, 2).reshape(batch_size, embed_dim, grid_height, grid_width)
        adapted = self.depthwise(grid)
        adapted = self.pointwise(self.act(adapted))
        adapted = adapted.flatten(2).transpose(1, 2)
        return tokens + self.dropout(adapted)


class DETRVisionTransformerWithRegisters(VisionTransformerWithRegisters):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cnn_stem: bool = True,
        stem_channels: int = 32,
        cnn_feature_fusion: bool = False,
        cnn_fusion_dropout: float = 0.1,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_registers: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        register_positional_embedding: bool = False,
        gradient_checkpointing: bool = False,
        head_pooling: str = "cls_register_mean",
        bbox_head_hidden_dim: Optional[int] = None,
        num_queries: int = 40,
        decoder_depth: int = 4,
        decoder_num_heads: Optional[int] = None,
        decoder_ffn_dim: Optional[int] = None,
        decoder_dropout: Optional[float] = None,
        decoder_memory_adapter: bool = False,
        decoder_memory_adapter_dropout: float = 0.0,
        learned_query_content: bool = True,
        separate_objectness: bool = True,
        objectness_prior_prob: float = 0.125,
        quality_head: bool = False,
        quality_prior_prob: float = 0.125,
        auxiliary_decoder_outputs: bool = False,
        query_denoising_noise: float = 0.0,
        count_head: bool = False,
        count_head_hidden_dim: Optional[int] = None,
        count_head_dropout: float = 0.05,
        count_head_prior: float = 1.2,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            use_cnn_stem=use_cnn_stem,
            stem_channels=stem_channels,
            cnn_feature_fusion=cnn_feature_fusion,
            cnn_fusion_dropout=cnn_fusion_dropout,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_registers=num_registers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            register_positional_embedding=register_positional_embedding,
            gradient_checkpointing=gradient_checkpointing,
            head_pooling=head_pooling,
        )
        hidden_dim = int(bbox_head_hidden_dim or max(64, embed_dim * 2))
        self.num_queries = max(1, int(num_queries))
        self.decoder_depth = max(1, int(decoder_depth))
        self.decoder_num_heads = max(1, int(decoder_num_heads or num_heads))
        self.decoder_ffn_dim = int(decoder_ffn_dim or max(embed_dim * 4, 256))
        self.decoder_dropout = float(dropout if decoder_dropout is None else decoder_dropout)
        self.decoder_memory_adapter_enabled = bool(decoder_memory_adapter)
        self.learned_query_content = bool(learned_query_content)
        self.separate_objectness = bool(separate_objectness)
        self.objectness_prior_prob = float(min(max(objectness_prior_prob, 1e-4), 1.0 - 1e-4))
        self.quality_head_enabled = bool(quality_head)
        self.quality_prior_prob = float(min(max(quality_prior_prob, 1e-4), 1.0 - 1e-4))
        self.auxiliary_decoder_outputs = bool(auxiliary_decoder_outputs)
        self.query_denoising_noise = float(max(0.0, query_denoising_noise))
        self.count_head_enabled = bool(count_head)
        self.count_head_prior = float(max(1e-4, count_head_prior))
        self.is_hybrid_model = True
        self.is_detr_model = True
        self.head = nn.Identity()
        self.query_embed = nn.Embedding(self.num_queries, embed_dim)
        self.query_content_embed = nn.Embedding(self.num_queries, embed_dim) if self.learned_query_content else None
        self.memory_position_embedding = SinePositionEmbedding2D(embed_dim)
        self.memory_adapter = (
            PatchMemoryAdapter(embed_dim=embed_dim, dropout=decoder_memory_adapter_dropout)
            if self.decoder_memory_adapter_enabled
            else nn.Identity()
        )
        self.decoder = DETRTransformerDecoder(
            embed_dim=embed_dim,
            num_layers=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            ffn_dim=self.decoder_ffn_dim,
            dropout=self.decoder_dropout,
        )
        self.classification_head = nn.Linear(
            embed_dim,
            num_classes if self.separate_objectness else num_classes + 1,
        )
        self.objectness_head = nn.Linear(embed_dim, 1) if self.separate_objectness else None
        self.quality_head = nn.Linear(embed_dim, 1) if self.quality_head_enabled else None
        self.bbox_head = MLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=4,
            num_layers=3,
            dropout=self.decoder_dropout,
        )
        count_hidden_dim = int(count_head_hidden_dim or max(64, embed_dim))
        self.count_head = (
            MLP(
                input_dim=embed_dim,
                hidden_dim=count_hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=float(max(0.0, count_head_dropout)),
            )
            if self.count_head_enabled
            else None
        )
        self.memory_adapter.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.classification_head.apply(self._init_weights)
        if self.objectness_head is not None:
            self.objectness_head.apply(self._init_weights)
            prior_logit = math.log(self.objectness_prior_prob / (1.0 - self.objectness_prior_prob))
            nn.init.constant_(self.objectness_head.bias, prior_logit)
        if self.quality_head is not None:
            self.quality_head.apply(self._init_weights)
            quality_prior_logit = math.log(self.quality_prior_prob / (1.0 - self.quality_prior_prob))
            nn.init.constant_(self.quality_head.bias, quality_prior_logit)
        self.bbox_head.apply(self._init_weights)
        if self.count_head is not None:
            self.count_head.apply(self._init_weights)
            last_linear = next(
                (module for module in reversed(list(self.count_head.modules())) if isinstance(module, nn.Linear)),
                None,
            )
            if last_linear is not None:
                prior_raw = math.log(math.expm1(self.count_head_prior)) if self.count_head_prior < 20.0 else self.count_head_prior
                nn.init.constant_(last_linear.bias, prior_raw)
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        if self.query_content_embed is not None:
            nn.init.trunc_normal_(self.query_content_embed.weight, std=0.02)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        return super().no_weight_decay_keywords() + ("query_embed", "query_content_embed")

    def _prediction_heads_from_decoder_output(self, decoder_output: Tensor) -> Dict[str, Tensor]:
        output = {
            "logits": self.classification_head(decoder_output),
            "boxes": self.bbox_head(decoder_output).sigmoid(),
            "decoder_output": decoder_output,
        }
        if self.objectness_head is not None:
            output["objectness_logits"] = self.objectness_head(decoder_output).squeeze(-1)
        if self.quality_head is not None:
            output["quality_logits"] = self.quality_head(decoder_output).squeeze(-1)
        return output

    def forward_heads(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        memory = self.memory_adapter(features["patches"], features["grid_size"]) if self.decoder_memory_adapter_enabled else features["patches"]
        batch_size = memory.shape[0]
        memory_pos = self.memory_position_embedding(
            grid_size=features["grid_size"],
            device=memory.device,
            dtype=memory.dtype,
        ).expand(batch_size, -1, -1)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if self.query_content_embed is not None:
            decoder_input = self.query_content_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            decoder_input = torch.zeros_like(query_pos)
        if self.training and self.query_denoising_noise > 0.0:
            noise_scale = float(self.query_denoising_noise)
            query_pos = query_pos + torch.randn_like(query_pos) * noise_scale
            decoder_input = decoder_input + torch.randn_like(decoder_input) * noise_scale
        decoder_result = self.decoder(
            target=decoder_input,
            memory=memory,
            query_pos=query_pos,
            memory_pos=memory_pos,
            memory_key_padding_mask=features.get("memory_key_padding_mask"),
            return_intermediate=self.auxiliary_decoder_outputs,
        )
        if self.auxiliary_decoder_outputs:
            decoder_output, intermediate_outputs = decoder_result
        else:
            decoder_output = decoder_result
            intermediate_outputs = []
        output = self._prediction_heads_from_decoder_output(decoder_output)
        if self.auxiliary_decoder_outputs and len(intermediate_outputs) > 1:
            output["aux_outputs"] = [
                self._prediction_heads_from_decoder_output(aux_output)
                for aux_output in intermediate_outputs[:-1]
            ]
        if self.count_head is not None:
            count_input = self.head_input_from_features(features)
            output["count_logits"] = self.count_head(count_input).squeeze(-1)
        return output

    def forward(self, x: Tensor, image_valid_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        features = self.forward_features(x, image_valid_mask=image_valid_mask)
        return self.forward_heads(features)


def extract_head_input_from_features(model: nn.Module, features: Dict[str, Tensor]) -> Tensor:
    if hasattr(model, "head_input_from_features"):
        return model.head_input_from_features(features)
    if "pooled" in features:
        return features["pooled"]
    if "cls" in features:
        return features["cls"]
    raise KeyError("Khong tim thay feature dau vao cho classification head.")


def extract_bbox_from_model_output(model_output):
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        return logits, boxes
    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model output tuple/list rong.")
        if len(model_output) == 1:
            return model_output[0], None
        return model_output[0], model_output[1]
    return model_output, None


def extract_detection_from_model_output(model_output):
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        boxes = model_output.get("boxes")
        objectness_logits = model_output.get("objectness_logits")
        return logits, boxes, objectness_logits
    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model output tuple/list rong.")
        logits = model_output[0]
        boxes = model_output[1] if len(model_output) >= 2 else None
        objectness_logits = model_output[2] if len(model_output) >= 3 else None
        return logits, boxes, objectness_logits
    return model_output, None, None


def _infer_feature_dim(model: nn.Module) -> int:
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return int(model.head.in_features)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return int(model.fc.in_features)
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        return int(model.heads.head.in_features)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for module in reversed(model.classifier):
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Linear):
            return int(classifier.in_features)
    raise TypeError("Khong the suy ra kich thuoc embedding tu backbone.")


def _strip_classifier_for_temporal(model: nn.Module) -> Tuple[nn.Module, int]:
    feature_dim = _infer_feature_dim(model)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Identity()
        return model, feature_dim
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Identity()
        return model, feature_dim
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        model.heads.head = nn.Identity()
        return model, feature_dim
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for index in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[index], nn.Linear):
                model.classifier[index] = nn.Identity()
                return model, feature_dim
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
        return model, feature_dim
    raise TypeError("Khong the strip classifier de dung cho temporal wrapper.")


def _quantize_kv_tensor(tensor: Tensor, num_bits: int) -> Tuple[Tensor, Tensor]:
    quant_max = float((1 << (int(num_bits) - 1)) - 1)
    scale = tensor.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / quant_max
    quantized = torch.clamp(torch.round(tensor / scale), min=-quant_max, max=quant_max).to(torch.int8)
    return quantized, scale


def _dequantize_kv_tensor(quantized: Tensor, scale: Tensor) -> Tensor:
    return quantized.to(dtype=scale.dtype) * scale


class TemporalAttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        kv_quant_bits: int = 8,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Temporal embed_dim phai chia het cho temporal_num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_quant_bits = int(kv_quant_bits)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, tensor: Tensor) -> Tensor:
        batch_size, time_steps, dim = tensor.shape
        tensor = tensor.reshape(batch_size, time_steps, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def _maybe_quantize_cache(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        cache_info: Dict[str, object] = {
            "enabled": False,
            "quant_bits": int(self.kv_quant_bits),
            "cached_frames": max(0, key.shape[1] - 1),
            "mode": "disabled",
        }
        if self.training:
            cache_info["mode"] = "train_bypass"
            return key, value, cache_info
        if self.kv_quant_bits not in (4, 8) or key.shape[1] <= 1:
            cache_info["mode"] = "fp_cache"
            return key, value, cache_info

        past_key = key[:, :-1]
        past_value = value[:, :-1]
        quant_key, key_scale = _quantize_kv_tensor(past_key, self.kv_quant_bits)
        quant_value, value_scale = _quantize_kv_tensor(past_value, self.kv_quant_bits)
        cache_info.update(
            {
                "enabled": True,
                "mode": f"int{int(self.kv_quant_bits)}_kv_cache",
                "key_dtype": str(quant_key.dtype),
                "value_dtype": str(quant_value.dtype),
            }
        )
        restored_key = _dequantize_kv_tensor(quant_key, key_scale)
        restored_value = _dequantize_kv_tensor(quant_value, value_scale)
        key = torch.cat((restored_key, key[:, -1:]), dim=1)
        value = torch.cat((restored_value, value[:, -1:]), dim=1)
        return key, value, cache_info

    def forward(self, embeddings: Tensor) -> Dict[str, Tensor]:
        normalized = self.norm(embeddings)
        query = self.q_proj(normalized[:, -1:])
        key = self.k_proj(normalized)
        value = self.v_proj(normalized)
        key, value, cache_info = self._maybe_quantize_cache(key, value)

        query = self._reshape_heads(query)
        key = self._reshape_heads(key)
        value = self._reshape_heads(value)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        pooled = attention @ value
        pooled = pooled.transpose(1, 2).reshape(embeddings.size(0), 1, embeddings.size(-1))
        pooled = self.out_proj(pooled[:, 0])
        pooled = pooled + embeddings[:, -1]
        return {
            "pooled": pooled,
            "attention": attention[:, :, 0],
            "cache_enabled": torch.tensor(1 if cache_info["enabled"] else 0, device=embeddings.device),
            "cache_quant_bits": torch.tensor(int(cache_info["quant_bits"]), device=embeddings.device),
        }


class StreamingViT(nn.Module):
    def __init__(
        self,
        frame_model: nn.Module,
        num_classes: int,
        temporal_frames: int = 3,
        temporal_num_heads: int = 4,
        temporal_dropout: float = 0.1,
        temporal_kv_quant_bits: int = 8,
    ) -> None:
        super().__init__()
        frame_model, embed_dim = _strip_classifier_for_temporal(frame_model)
        self.frame_model = frame_model
        self.temporal_frames = max(1, int(temporal_frames))
        self.num_registers = int(getattr(frame_model, "num_registers", 0))
        self.temporal_pool = TemporalAttentionPool(
            dim=embed_dim,
            num_heads=temporal_num_heads,
            dropout=temporal_dropout,
            kv_quant_bits=temporal_kv_quant_bits,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.temporal_pool.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def no_weight_decay_keywords(self) -> Tuple[str, ...]:
        if hasattr(self.frame_model, "no_weight_decay_keywords"):
            return tuple(self.frame_model.no_weight_decay_keywords()) + ("temporal_pool.norm", "bias")
        return ("bias", "norm")

    def _ensure_temporal_input(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return x.unsqueeze(1).repeat(1, self.temporal_frames, 1, 1, 1)
        if x.ndim != 5:
            raise ValueError("StreamingViT yeu cau input (B, C, H, W) hoac (B, T, C, H, W).")
        if x.shape[1] == self.temporal_frames:
            return x
        if x.shape[1] > self.temporal_frames:
            return x[:, -self.temporal_frames :]
        pad_count = self.temporal_frames - x.shape[1]
        pad = x[:, :1].expand(-1, pad_count, -1, -1, -1)
        return torch.cat((pad, x), dim=1)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        x = self._ensure_temporal_input(x)
        batch_size, time_steps, channels, height, width = x.shape
        flat = x.reshape(batch_size * time_steps, channels, height, width)

        base_features: Optional[Dict[str, Tensor]] = None
        if hasattr(self.frame_model, "forward_features"):
            maybe_features = self.frame_model.forward_features(flat)
            if isinstance(maybe_features, dict) and "cls" in maybe_features:
                base_features = maybe_features
                if hasattr(self.frame_model, "head_input_from_features"):
                    frame_embeddings = self.frame_model.head_input_from_features(maybe_features)
                else:
                    frame_embeddings = maybe_features.get("pooled", maybe_features["cls"])
            else:
                frame_embeddings = self.frame_model(flat)
        else:
            frame_embeddings = self.frame_model(flat)

        frame_embeddings = frame_embeddings.reshape(batch_size, time_steps, -1)
        temporal_features = self.temporal_pool(frame_embeddings)
        features: Dict[str, Tensor] = {
            "cls": temporal_features["pooled"],
            "pooled": temporal_features["pooled"],
            "frame_cls": frame_embeddings,
            "temporal_attention": temporal_features["attention"],
            "temporal_cache_enabled": temporal_features["cache_enabled"],
            "temporal_cache_quant_bits": temporal_features["cache_quant_bits"],
        }

        if base_features is not None:
            if "patches" in base_features:
                patch_shape = base_features["patches"].shape[1:]
                features["patches"] = base_features["patches"].reshape(batch_size, time_steps, *patch_shape)
            if "registers" in base_features:
                register_shape = base_features["registers"].shape[1:]
                features["registers"] = base_features["registers"].reshape(
                    batch_size,
                    time_steps,
                    *register_shape,
                )
            if "tokens" in base_features:
                token_shape = base_features["tokens"].shape[1:]
                features["tokens"] = base_features["tokens"].reshape(batch_size, time_steps, *token_shape)
            if "grid_size" in base_features:
                features["grid_size"] = base_features["grid_size"]
        return features

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return self.head(features["pooled"])


def _config_to_dict(model_config: Any) -> Dict[str, Any]:
    if model_config is None:
        return {}
    if isinstance(model_config, dict):
        return dict(model_config)
    if hasattr(model_config, "__dict__"):
        return dict(vars(model_config))
    raise TypeError(f"Khong the chuyen model_config sang dict: {type(model_config)!r}")


def _replace_resnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise TypeError("ResNet baseline khong co fc Linear nhu mong doi.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _replace_mobilenet_head(model: nn.Module, num_classes: int) -> nn.Module:
    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Sequential):
        raise TypeError("MobileNet baseline khong co classifier nhu mong doi.")
    if not isinstance(model.classifier[-1], nn.Linear):
        raise TypeError("MobileNet classifier[-1] khong phai Linear.")
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_torchvision_vit(
    num_classes: int,
    image_size: int,
    dropout: float,
) -> nn.Module:
    return tv_models.vit_b_16(
        weights=None,
        image_size=image_size,
        num_classes=num_classes,
        dropout=dropout,
    )


def _build_resnet50(num_classes: int) -> nn.Module:
    model = tv_models.resnet50(weights=None)
    return _replace_resnet_head(model, num_classes)


def _build_mobilenet_v3_large(num_classes: int) -> nn.Module:
    model = tv_models.mobilenet_v3_large(weights=None)
    return _replace_mobilenet_head(model, num_classes)


def _pop_pretraining_option(config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        return None
    return config.pop(key)


def _pretraining_option_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    return True


def _reject_pretraining_options(config: Dict[str, Any]) -> None:
    blocked_keys = (
        "pretrained",
        "pretrain",
        "use_pretrained",
        "timm_pretrained",
        "weights",
        "pretrained_weights",
        "weights_path",
        "weight_path",
        "pretrained_path",
        "init_checkpoint",
        "checkpoint_path",
        "external_checkpoint",
    )
    blocked_values = {
        key: _pop_pretraining_option(config, key)
        for key in blocked_keys
        if key in config
    }
    enabled_keys = [
        key for key, value in blocked_values.items() if _pretraining_option_enabled(value)
    ]
    if enabled_keys:
        raise ValueError(
            "Project TRKH khong cho phep bat ky hinh thuc pretrained/external weights nao. "
            f"Tham so bi chan: {', '.join(enabled_keys)}. Hay train model tu dau voi weights=None."
        )


def create_model(
    num_classes: int,
    model_config: Optional[Any] = None,
    **overrides: Any,
) -> nn.Module:
    config = _config_to_dict(model_config)
    config.update(overrides)
    _reject_pretraining_options(config)

    temporal_frames = max(1, int(config.pop("temporal_frames", 1)))
    temporal_num_heads = max(1, int(config.pop("temporal_num_heads", 4)))
    temporal_dropout = float(config.pop("temporal_dropout", config.get("dropout", 0.1)))
    temporal_kv_quant_bits = int(config.pop("temporal_kv_quant_bits", 8))
    learned_query_content = bool(config.pop("learned_query_content", True))
    separate_objectness = bool(config.pop("separate_objectness", True))
    objectness_prior_prob = float(config.pop("objectness_prior_prob", 0.125))
    model_type = str(config.pop("model_type", "vit_registers_hybrid")).strip().lower()
    config.pop("timm_model_name", None)

    if model_type == "vit_registers":
        for detector_only_key in (
            "bbox_head_hidden_dim",
            "num_queries",
            "decoder_depth",
            "decoder_num_heads",
            "decoder_ffn_dim",
            "decoder_dropout",
            "decoder_memory_adapter",
            "decoder_memory_adapter_dropout",
            "quality_head",
            "quality_prior_prob",
            "auxiliary_decoder_outputs",
            "query_denoising_noise",
            "count_head",
            "count_head_hidden_dim",
            "count_head_dropout",
            "count_head_prior",
        ):
            config.pop(detector_only_key, None)
        model = VisionTransformerWithRegisters(num_classes=num_classes, **config)
    elif model_type in {"detr_vit_registers", "vit_registers_hybrid"}:
        model = DETRVisionTransformerWithRegisters(
            num_classes=num_classes,
            learned_query_content=learned_query_content,
            separate_objectness=separate_objectness,
            objectness_prior_prob=objectness_prior_prob,
            **config,
        )
    elif model_type == "resnet50":
        model = _build_resnet50(num_classes)
    elif model_type == "mobilenet_v3_large":
        model = _build_mobilenet_v3_large(num_classes)
    elif model_type == "vit_b_16":
        image_size = int(config.get("image_size", 224))
        dropout = float(config.get("dropout", 0.0))
        model = _build_torchvision_vit(
            num_classes=num_classes,
            image_size=image_size,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Khong ho tro model_type: {model_type}.")

    model.model_type = model_type
    if temporal_frames > 1:
        if model_type in {"detr_vit_registers", "vit_registers_hybrid"}:
            raise ValueError(
                "DETR-ViT-Registers hien chi ho tro temporal_frames=1. "
                "Hay dung temporal smoothing o stream_infer.py cho video."
            )
        model = StreamingViT(
            frame_model=model,
            num_classes=num_classes,
            temporal_frames=temporal_frames,
            temporal_num_heads=temporal_num_heads,
            temporal_dropout=temporal_dropout,
            temporal_kv_quant_bits=temporal_kv_quant_bits,
        )
        model.model_type = f"streaming_{model_type}"
    model.temporal_frames = temporal_frames
    return model


def load_model_state(model: nn.Module, state_dict: Dict[str, Tensor], strict: bool = True):
    if hasattr(model, "load_flexible_state_dict"):
        return model.load_flexible_state_dict(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    num_classes: Optional[int] = None,
    override_image_size: Optional[int] = None,
) -> nn.Module:
    class_names = checkpoint.get("class_names", [])
    resolved_num_classes = int(num_classes or len(class_names))
    model_config = dict(checkpoint.get("model_config", {}))
    if "model_type" not in model_config:
        state_keys = checkpoint.get("model_state", {}).keys()
        model_config["model_type"] = (
            "vit_registers_hybrid"
            if any(
                str(key).startswith(prefix)
                for key in state_keys
                for prefix in ("bbox_head", "query_embed", "decoder", "classification_head")
            )
            else "vit_registers"
        )
    state_dict = checkpoint.get("model_state", {})
    classification_weight = state_dict.get("classification_head.weight")
    has_objectness_head = any(str(key).startswith("objectness_head.") for key in state_dict.keys())
    has_query_content = any(str(key).startswith("query_content_embed.") for key in state_dict.keys())
    has_count_head = any(str(key).startswith("count_head.") for key in state_dict.keys())
    has_quality_head = any(str(key).startswith("quality_head.") for key in state_dict.keys())
    if (
        model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"}
        and torch.is_tensor(classification_weight)
        and int(classification_weight.shape[0]) == resolved_num_classes + 1
        and not has_objectness_head
    ):
        model_config["separate_objectness"] = False
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and not has_query_content:
        model_config["learned_query_content"] = False
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and has_count_head:
        model_config["count_head"] = True
    if model_config.get("model_type") in {"detr_vit_registers", "vit_registers_hybrid"} and has_quality_head:
        model_config["quality_head"] = True
    if override_image_size is not None:
        model_config["image_size"] = int(override_image_size)
    model = create_model(num_classes=resolved_num_classes, model_config=model_config)
    load_model_state(model, checkpoint["model_state"], strict=True)
    return model
