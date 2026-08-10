from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import ModelConfig, load_data_spec, to_serializable
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    ensure_dir,
    format_seconds,
    json_dump,
    load_checkpoint,
    resolve_amp_dtype,
    save_checkpoint,
    set_seed,
)
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    StrictBalancedBatchSampler,
    _pseudo_foreground_mask_from_tensor_images,
    build_train_transform,
)
from trkh.models.model import create_model, extract_head_input_from_features
from trkh.training.losses import SupervisedContrastiveLoss


def load_model_init_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    expected_class_names: Sequence[str],
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Khong tim thay init checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Init checkpoint khong co model_state hop le: {checkpoint_path}")
    checkpoint_classes = list(checkpoint.get("class_names", []))
    if checkpoint_classes and checkpoint_classes != list(expected_class_names):
        raise ValueError("Class names trong init checkpoint khong khop data.yaml hien tai.")
    if hasattr(model, "load_flexible_state_dict"):
        missing_keys, unexpected_keys = model.load_flexible_state_dict(model_state, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    return {
        "enabled": True,
        "path": str(checkpoint_path),
        "checkpoint_kind": str(checkpoint.get("checkpoint_kind", "")),
        "epoch": int(checkpoint.get("epoch", checkpoint.get("completed_epoch", 0)) or 0),
        "best_epoch": int(checkpoint.get("best_epoch", 0) or 0),
        "missing_key_count": int(len(missing_keys)),
        "unexpected_key_count": int(len(unexpected_keys)),
        "missing_key_prefixes": sorted({str(key).split(".", 1)[0] for key in missing_keys})[:12],
        "unexpected_key_prefixes": sorted({str(key).split(".", 1)[0] for key in unexpected_keys})[:12],
        "note": "Loaded model weights only; optimizer/epoch are not resumed.",
    }


class TwoViewDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform_a=None,
        transform_b=None,
        dataset_b: Optional[Dataset] = None,
    ) -> None:
        self.dataset = dataset
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.dataset_b = dataset_b

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _unpack_classification_item(item) -> Tuple[Tensor, int]:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("TwoViewDataset expects classification samples as (image, label[, metadata]).")
        image = item[0]
        label = item[1]
        if not torch.is_tensor(image):
            raise TypeError("TwoViewDataset expects transformed image tensors.")
        return image, int(label)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        if self.dataset_b is not None:
            view_a, label_a = self._unpack_classification_item(self.dataset[int(index)])
            view_b, label_b = self._unpack_classification_item(self.dataset_b[int(index)])
            if label_a != label_b:
                raise ValueError(
                    f"Two SSL views have mismatched labels at index {index}: {label_a} vs {label_b}."
                )
            return view_a, view_b, int(label_a)

        if self.transform_a is None:
            view_a, label = self._unpack_classification_item(self.dataset[int(index)])
            return view_a, view_a, int(label)

        sample = self.dataset.samples[int(index)]
        image = self.dataset._load_rgb_image(sample.image_path)
        view_a = self.transform_a(image)
        view_b = self.transform_b(image) if self.transform_b is not None else view_a
        return view_a, view_b, int(sample.label)


class BarlowProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DinoProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaskedPatchDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, patch_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_dim),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("MaskedPatchDecoder expects patch tokens [B, N, D].")
        return self.net(patch_tokens)


def patchify_images(images: Tensor, patch_size: int) -> Tensor:
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("patchify_images expects images [B, 3, H, W].")
    patch = int(patch_size)
    if patch <= 0:
        raise ValueError("patch_size must be > 0.")
    height = int(images.size(-2))
    width = int(images.size(-1))
    if height % patch != 0 or width % patch != 0:
        raise ValueError("image height/width must be divisible by patch_size.")
    grid_h = height // patch
    grid_w = width // patch
    patches = images.reshape(images.size(0), 3, grid_h, patch, grid_w, patch)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(images.size(0), grid_h * grid_w, 3 * patch * patch)
    return patches


def _patch_detail_scores(images: Tensor, patch_size: int) -> Tensor:
    gray = images.float().mean(dim=1, keepdim=True)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, :, :, 1:] = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
    dy[:, :, 1:, :] = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
    detail = dx + dy
    pooled = F.avg_pool2d(detail, kernel_size=int(patch_size), stride=int(patch_size))
    flat = pooled.flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def build_mae_patch_mask(
    images: Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    foreground_weight: float = 0.75,
    detail_weight: float = 0.25,
) -> Tensor:
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("build_mae_patch_mask expects images [B, 3, H, W].")
    patch = int(patch_size)
    if patch <= 0:
        raise ValueError("patch_size must be > 0.")
    height = int(images.size(-2))
    width = int(images.size(-1))
    if height % patch != 0 or width % patch != 0:
        raise ValueError("image height/width must be divisible by patch_size.")
    batch_size = int(images.size(0))
    patch_count = int((height // patch) * (width // patch))
    ratio = max(0.0, min(0.95, float(mask_ratio)))
    mask_count = max(1, min(patch_count, int(round(float(patch_count) * ratio))))
    random_scores = torch.rand((batch_size, patch_count), device=images.device, dtype=torch.float32)
    scores = random_scores
    if float(foreground_weight) > 0.0:
        foreground = _pseudo_foreground_mask_from_tensor_images(images.detach()).to(dtype=torch.float32)
        patch_fg = F.avg_pool2d(foreground, kernel_size=int(patch_size), stride=int(patch_size)).flatten(1)
        scores = scores + float(foreground_weight) * patch_fg
    if float(detail_weight) > 0.0:
        scores = scores + float(detail_weight) * _patch_detail_scores(images.detach(), int(patch_size))
    selected = torch.topk(scores, k=mask_count, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros((batch_size, patch_count), device=images.device, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask


def apply_patch_mask(images: Tensor, patch_mask: Tensor, patch_size: int, mask_value: float = 0.0) -> Tensor:
    if patch_mask.ndim != 2:
        raise ValueError("patch_mask must have shape [B, N].")
    patch = int(patch_size)
    batch_size, _, height, width = images.shape
    if height % patch != 0 or width % patch != 0:
        raise ValueError("image height/width must be divisible by patch_size.")
    grid_h = height // patch
    grid_w = width // patch
    if patch_mask.shape != (batch_size, grid_h * grid_w):
        raise ValueError("patch_mask shape does not match image grid.")
    pixel_mask = patch_mask.view(batch_size, 1, grid_h, grid_w)
    pixel_mask = pixel_mask.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
    masked = images.clone()
    masked = torch.where(pixel_mask.expand_as(masked), masked.new_tensor(float(mask_value)), masked)
    return masked


def mae_reconstruction_loss(
    model: nn.Module,
    decoder: nn.Module,
    images: Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    foreground_weight: float,
    detail_weight: float,
) -> Tuple[Tensor, Dict[str, float]]:
    patch_mask = build_mae_patch_mask(
        images,
        patch_size=int(patch_size),
        mask_ratio=float(mask_ratio),
        foreground_weight=float(foreground_weight),
        detail_weight=float(detail_weight),
    )
    masked_images = apply_patch_mask(images, patch_mask, int(patch_size), mask_value=0.0)
    if not hasattr(model, "forward_features"):
        raise ValueError("MAE pretrain requires a model with forward_features.")
    features = model.forward_features(masked_images)
    if not isinstance(features, dict) or "patches" not in features:
        raise ValueError("MAE pretrain requires features['patches'].")
    patch_tokens = features["patches"]
    if patch_tokens.ndim != 3:
        raise ValueError("features['patches'] must have shape [B, N, D].")
    target = patchify_images(images.float(), int(patch_size))
    active_mask = patch_mask
    patch_indices = features.get("patch_indices")
    if torch.is_tensor(patch_indices) and patch_indices.ndim == 2 and patch_indices.shape[:1] == patch_mask.shape[:1]:
        patch_indices = patch_indices.to(device=target.device, dtype=torch.long)
        if patch_indices.size(1) != target.size(1):
            target = target.gather(1, patch_indices.unsqueeze(-1).expand(-1, -1, target.size(-1)))
            active_mask = active_mask.gather(1, patch_indices.to(device=active_mask.device))
    if patch_tokens.size(1) != target.size(1):
        raise ValueError("Patch token count does not match target patch count.")
    prediction = decoder(patch_tokens).float()
    per_patch = (prediction - target.to(device=prediction.device, dtype=prediction.dtype)).pow(2).mean(dim=-1)
    active_mask = active_mask.to(device=per_patch.device, dtype=torch.bool)
    if not bool(active_mask.any().item()):
        loss = per_patch.mean()
    else:
        loss = per_patch.masked_select(active_mask).mean()
    return loss, {
        "mask_fraction": float(active_mask.float().mean().detach().cpu().item()),
        "patch_count": float(active_mask.size(1)),
    }


def off_diagonal(x: Tensor) -> Tensor:
    if x.ndim != 2 or x.size(0) != x.size(1):
        raise ValueError("off_diagonal expects a square matrix.")
    n = x.size(0)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z_a: Tensor, z_b: Tensor, *, lambd: float = 0.005) -> Tensor:
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("Barlow Twins projections must have matching shape [B, D].")
    if z_a.size(0) < 2:
        return z_a.sum() * 0.0
    z_a = z_a.float()
    z_b = z_b.float()
    z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0, unbiased=False).clamp_min(1e-4)
    z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0, unbiased=False).clamp_min(1e-4)
    correlation = torch.matmul(z_a.T, z_b) / float(z_a.size(0))
    on_diag = torch.diagonal(correlation).add(-1.0).pow(2).sum()
    off_diag = off_diagonal(correlation).pow(2).sum()
    return on_diag + float(lambd) * off_diag


def vicreg_loss(
    z_a: Tensor,
    z_b: Tensor,
    *,
    invariance_coeff: float = 25.0,
    variance_coeff: float = 25.0,
    covariance_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape != z_b.shape:
        raise ValueError("VICReg projections must have matching shape [B, D].")
    z_a = z_a.float()
    z_b = z_b.float()
    invariance_loss = F.mse_loss(z_a, z_b)
    std_a = torch.sqrt(z_a.var(dim=0, unbiased=False) + float(eps))
    std_b = torch.sqrt(z_b.var(dim=0, unbiased=False) + float(eps))
    variance_loss = (
        F.relu(float(gamma) - std_a).mean()
        + F.relu(float(gamma) - std_b).mean()
    )
    if z_a.size(0) < 2:
        covariance_loss = z_a.sum() * 0.0
    else:
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_a = torch.matmul(z_a.T, z_a) / float(z_a.size(0) - 1)
        cov_b = torch.matmul(z_b.T, z_b) / float(z_b.size(0) - 1)
        dim = max(1, z_a.size(1))
        covariance_loss = (
            off_diagonal(cov_a).pow(2).sum() / float(dim)
            + off_diagonal(cov_b).pow(2).sum() / float(dim)
        )
    loss = (
        float(invariance_coeff) * invariance_loss
        + float(variance_coeff) * variance_loss
        + float(covariance_coeff) * covariance_loss
    )
    return loss, {
        "invariance": invariance_loss,
        "variance": variance_loss,
        "covariance": covariance_loss,
    }


def dino_self_distillation_loss(
    student_a: Tensor,
    student_b: Tensor,
    teacher_a: Tensor,
    teacher_b: Tensor,
    *,
    center: Tensor,
    student_temperature: float = 0.10,
    teacher_temperature: float = 0.07,
) -> Tensor:
    if student_a.ndim != 2 or student_b.ndim != 2 or teacher_a.ndim != 2 or teacher_b.ndim != 2:
        raise ValueError("DINO logits must be rank-2 tensors [B, D].")
    if student_a.shape != student_b.shape or student_a.shape != teacher_a.shape or student_a.shape != teacher_b.shape:
        raise ValueError("DINO student/teacher logits must share shape.")
    if center.ndim != 1 or center.numel() != student_a.size(1):
        raise ValueError("DINO center must have shape [D].")
    student_temp = max(float(student_temperature), 1e-6)
    teacher_temp = max(float(teacher_temperature), 1e-6)
    teacher_center = center.to(device=teacher_a.device, dtype=teacher_a.dtype).view(1, -1)
    with torch.no_grad():
        teacher_prob_a = F.softmax((teacher_a.float() - teacher_center.float()) / teacher_temp, dim=-1)
        teacher_prob_b = F.softmax((teacher_b.float() - teacher_center.float()) / teacher_temp, dim=-1)
    student_logprob_a = F.log_softmax(student_a.float() / student_temp, dim=-1)
    student_logprob_b = F.log_softmax(student_b.float() / student_temp, dim=-1)
    loss_ab = -(teacher_prob_a * student_logprob_b).sum(dim=-1).mean()
    loss_ba = -(teacher_prob_b * student_logprob_a).sum(dim=-1).mean()
    return 0.5 * (loss_ab + loss_ba)


@torch.no_grad()
def update_ema_model(source: nn.Module, target: nn.Module, momentum: float) -> None:
    m = max(0.0, min(1.0, float(momentum)))
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(m).add_(source_param.data, alpha=1.0 - m)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        if torch.is_floating_point(target_buffer):
            target_buffer.data.mul_(m).add_(source_buffer.data, alpha=1.0 - m)
        else:
            target_buffer.data.copy_(source_buffer.data)


def build_v8_model_config(
    image_size: int,
    *,
    bbox_spatial_fusion: bool = False,
    bbox_spatial_fusion_hidden_dim: int = 64,
    bbox_spatial_fusion_dropout: float = 0.05,
    bbox_spatial_fusion_logit_scale: float = 0.20,
) -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        pretrained=False,
        image_size=int(image_size),
        patch_size=16,
        stem_channels=32,
        cnn_feature_fusion=True,
        cnn_fusion_dropout=0.10,
        fine_grained_pooling=True,
        fine_grained_pooling_dropout=0.08,
        multi_branch_fusion=True,
        branch_color_tokens=1,
        branch_edge_tokens=1,
        branch_cnn_tokens=0,
        branch_token_dropout=0.08,
        detail_patch_enhancement=True,
        detail_patch_dropout=0.05,
        token_pruning=True,
        token_prune_layers="2,5",
        token_keep_rates="0.85,0.65",
        token_prune_foreground_weight=0.45,
        pairwise_margin_head=True,
        pairwise_margin_pairs="0-1,1-2,2-3,4-rest",
        pairwise_margin_logit_scale=0.25,
        pairwise_margin_dropout=0.05,
        pairwise_margin_routing=True,
        pairwise_margin_route_max_probability_margin=0.20,
        bbox_spatial_fusion=bool(bbox_spatial_fusion),
        bbox_spatial_fusion_hidden_dim=int(bbox_spatial_fusion_hidden_dim),
        bbox_spatial_fusion_dropout=float(bbox_spatial_fusion_dropout),
        bbox_spatial_fusion_logit_scale=float(bbox_spatial_fusion_logit_scale),
        embed_dim=256,
        depth=8,
        num_heads=8,
        num_registers=4,
        register_positional_embedding=True,
        head_pooling="cls_branch_register_mean",
        dropout=0.12,
        attention_dropout=0.03,
        drop_path_rate=0.10,
    )


def build_ssl_transform(args: argparse.Namespace):
    return build_train_transform(
        image_size=int(args.image_size),
        resize_mode="pad",
        scale_min=float(args.scale_min),
        scale_crop_probability=float(args.scale_crop_probability),
        brightness=float(args.brightness),
        contrast=float(args.contrast),
        saturation=float(args.saturation),
        hue=float(args.hue),
        random_erasing_probability=float(args.random_erasing_probability),
        random_affine_degrees=float(args.random_affine_degrees),
        random_affine_translate=float(args.random_affine_translate),
        random_affine_scale_min=float(args.random_affine_scale_min),
        horizontal_flip_probability=float(args.horizontal_flip_probability),
        vertical_flip_probability=float(args.vertical_flip_probability),
        rotate90_probability=float(args.rotate90_probability),
        lighting_probability=float(args.lighting_probability),
        illumination_normalization=True,
        illumination_normalization_strength=float(args.illumination_normalization_strength),
        background_suppression_mode=str(args.background_suppression_mode),
        background_suppression_probability=float(args.background_suppression_probability),
        background_suppression_margin=float(args.background_suppression_margin),
        background_suppression_blur_radius=float(args.background_suppression_blur_radius),
        surface_detail_amplification_mode=str(args.surface_detail_amplification_mode),
        surface_detail_amplification_probability=float(args.surface_detail_amplification_probability),
        surface_detail_amplification_strength=float(args.surface_detail_amplification_strength),
        surface_detail_amplification_blur_radius=float(args.surface_detail_amplification_blur_radius),
        surface_detail_amplification_foreground_weight=float(args.surface_detail_amplification_foreground_weight),
        local_exposure_probability=float(args.local_exposure_probability),
        local_exposure_strength=float(args.local_exposure_strength),
        obstacle_probability=float(args.obstacle_probability),
        obstacle_max_area=float(args.obstacle_max_area),
        randaugment_num_ops=int(args.randaugment_num_ops),
        randaugment_magnitude=int(args.randaugment_magnitude),
    )


def build_ssl_dataset(data_spec, *, split: str, transform, args: argparse.Namespace) -> Dataset:
    data_format = str(data_spec.data_format or "yolo").strip().lower()
    if data_format == "classification_folder":
        return ClassificationFolderDataset.from_data_spec(
            data_spec,
            split=split,
            transform=transform,
            class_aware_augmentation=False,
        )
    if data_format == "yolo":
        return MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=transform,
            crop_margin_ratio=float(args.crop_margin_ratio),
            crop_to_primary_object=True,
            classification_target=True,
            classification_object_crops=True,
            class_aware_augmentation=False,
            classification_bbox_metadata=False,
        )
    raise ValueError(f"Unsupported internal SSL data format: {data_spec.data_format}")


def encode(model: nn.Module, images: Tensor) -> Tensor:
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
        if isinstance(features, dict):
            return extract_head_input_from_features(model, features)
    output = model(images)
    if isinstance(output, dict):
        output = output.get("logits")
    if not torch.is_tensor(output) or output.ndim != 2:
        raise ValueError("Cannot extract SSL embedding from model output.")
    return output


def infer_embedding_dim(model: nn.Module, image_size: int, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, int(image_size), int(image_size), device=device)
        embedding = encode(model, dummy)
    if embedding.ndim != 2:
        raise ValueError("SSL embedding must have shape [B, D].")
    return int(embedding.size(1))


def save_ssl_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    projector: Optional[nn.Module],
    decoder: Optional[nn.Module],
    optimizer: optim.Optimizer,
    model_config: ModelConfig,
    class_names: Sequence[str],
    data_yaml: Path,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    ssl_config: Dict[str, object],
) -> None:
    ssl_method = str(ssl_config.get("ssl_method", "barlow")).strip().lower() or "barlow"
    payload = {
        "checkpoint_kind": f"internal_{ssl_method}_pretrain",
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_ssl_loss": float(best_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": to_serializable(model_config),
        "class_names": list(class_names),
        "data_yaml": str(Path(data_yaml).resolve()),
        "ssl_config": ssl_config,
        "note": (
            f"Train-only internal {ssl_method.upper()} pretrain; "
            "no val/test images and no external pretrained weights."
        ),
    }
    if projector is not None:
        payload["projector_state"] = projector.state_dict()
    if decoder is not None:
        payload["decoder_state"] = decoder.state_dict()
    save_checkpoint(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only internal SSL pretrain for TRKH.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="internal_barlow_pretrain")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--ssl-method", choices=("barlow", "vicreg", "mae", "dino"), default="barlow")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional supervised/internal checkpoint used only to initialize model weights before "
            "train-only SSL; does not load optimizer or epoch state."
        ),
    )
    parser.add_argument("--resume-reset-optimizer", action="store_true", default=False)
    parser.add_argument("--resume-reset-epoch", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--projector-output-dim", type=int, default=512)
    parser.add_argument("--barlow-lambda", type=float, default=0.005)
    parser.add_argument("--vicreg-invariance-coeff", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-coeff", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-coeff", type=float, default=1.0)
    parser.add_argument("--vicreg-gamma", type=float, default=1.0)
    parser.add_argument("--vicreg-eps", type=float, default=1e-4)
    parser.add_argument("--mae-mask-ratio", type=float, default=0.55)
    parser.add_argument("--mae-foreground-weight", type=float, default=0.75)
    parser.add_argument("--mae-detail-weight", type=float, default=0.25)
    parser.add_argument("--mae-decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--dino-output-dim", type=int, default=1024)
    parser.add_argument("--dino-student-temperature", type=float, default=0.10)
    parser.add_argument("--dino-teacher-temperature", type=float, default=0.07)
    parser.add_argument("--dino-center-momentum", type=float, default=0.90)
    parser.add_argument("--dino-teacher-momentum", type=float, default=0.996)
    parser.add_argument(
        "--bbox-spatial-fusion",
        action="store_true",
        default=False,
        help=(
            "Build the SSL model with the same bbox_spatial_fusion_head extension "
            "used by the supervised V8 keeper. The SSL forward path does not need "
            "bbox metadata, but preserving this module avoids reinitializing it "
            "during the follow-up supervised smoke."
        ),
    )
    parser.add_argument("--bbox-spatial-fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--bbox-spatial-fusion-dropout", type=float, default=0.05)
    parser.add_argument("--bbox-spatial-fusion-logit-scale", type=float, default=0.20)
    parser.add_argument("--supcon-loss-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.16)
    parser.add_argument("--disable-supcon-class-balanced", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--disable-balanced-sampler", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.05,
        help="YOLO-format SSL object-crop margin; ignored for classification_folder data.",
    )
    parser.add_argument("--scale-min", type=float, default=0.78)
    parser.add_argument("--scale-crop-probability", type=float, default=0.75)
    parser.add_argument("--brightness", type=float, default=0.12)
    parser.add_argument("--contrast", type=float, default=0.12)
    parser.add_argument("--saturation", type=float, default=0.08)
    parser.add_argument("--hue", type=float, default=0.015)
    parser.add_argument("--random-erasing-probability", type=float, default=0.08)
    parser.add_argument("--random-affine-degrees", type=float, default=6.0)
    parser.add_argument("--random-affine-translate", type=float, default=0.04)
    parser.add_argument("--random-affine-scale-min", type=float, default=0.92)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.05)
    parser.add_argument("--rotate90-probability", type=float, default=0.10)
    parser.add_argument("--lighting-probability", type=float, default=0.15)
    parser.add_argument("--illumination-normalization-strength", type=float, default=0.18)
    parser.add_argument(
        "--background-suppression-mode",
        choices=("none", "desaturate_blur", "gray", "blur", "mean"),
        default="desaturate_blur",
    )
    parser.add_argument("--background-suppression-probability", type=float, default=0.40)
    parser.add_argument("--background-suppression-margin", type=float, default=0.08)
    parser.add_argument("--background-suppression-blur-radius", type=float, default=7.0)
    parser.add_argument(
        "--surface-detail-amplification-mode",
        choices=("none", "unsharp", "rgb_unsharp", "foreground_unsharp", "luma", "luma_residual", "foreground_luma"),
        default="none",
    )
    parser.add_argument("--surface-detail-amplification-probability", type=float, default=0.0)
    parser.add_argument("--surface-detail-amplification-strength", type=float, default=0.0)
    parser.add_argument("--surface-detail-amplification-blur-radius", type=float, default=1.25)
    parser.add_argument("--surface-detail-amplification-foreground-weight", type=float, default=0.85)
    parser.add_argument("--local-exposure-probability", type=float, default=0.12)
    parser.add_argument("--local-exposure-strength", type=float, default=0.25)
    parser.add_argument("--obstacle-probability", type=float, default=0.02)
    parser.add_argument("--obstacle-max-area", type=float, default=0.06)
    parser.add_argument("--randaugment-num-ops", type=int, default=0)
    parser.add_argument("--randaugment-magnitude", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    ssl_method = str(args.ssl_method).strip().lower()
    if args.batch_size < 2 and ssl_method != "mae":
        raise ValueError("--batch-size must be >= 2 for Barlow/VICReg/DINO.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume va --init-checkpoint khong duoc dung dong thoi.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if args.vicreg_invariance_coeff < 0.0:
        raise ValueError("--vicreg-invariance-coeff must be >= 0.")
    if args.vicreg_variance_coeff < 0.0:
        raise ValueError("--vicreg-variance-coeff must be >= 0.")
    if args.vicreg_covariance_coeff < 0.0:
        raise ValueError("--vicreg-covariance-coeff must be >= 0.")
    if args.vicreg_gamma <= 0.0:
        raise ValueError("--vicreg-gamma must be > 0.")
    if args.vicreg_eps <= 0.0:
        raise ValueError("--vicreg-eps must be > 0.")
    if not (0.0 < float(args.mae_mask_ratio) < 0.95):
        raise ValueError("--mae-mask-ratio must be in (0, 0.95).")
    if args.mae_decoder_hidden_dim < 16:
        raise ValueError("--mae-decoder-hidden-dim must be >= 16.")
    if args.dino_output_dim < 2:
        raise ValueError("--dino-output-dim must be >= 2.")
    if args.dino_student_temperature <= 0.0:
        raise ValueError("--dino-student-temperature must be > 0.")
    if args.dino_teacher_temperature <= 0.0:
        raise ValueError("--dino-teacher-temperature must be > 0.")
    if not (0.0 <= float(args.dino_center_momentum) < 1.0):
        raise ValueError("--dino-center-momentum must be in [0, 1).")
    if not (0.0 <= float(args.dino_teacher_momentum) <= 1.0):
        raise ValueError("--dino-teacher-momentum must be in [0, 1].")
    if args.bbox_spatial_fusion_hidden_dim < 1:
        raise ValueError("--bbox-spatial-fusion-hidden-dim must be >= 1.")
    if args.bbox_spatial_fusion_dropout < 0.0:
        raise ValueError("--bbox-spatial-fusion-dropout must be >= 0.")
    if args.bbox_spatial_fusion_logit_scale < 0.0:
        raise ValueError("--bbox-spatial-fusion-logit-scale must be >= 0.")
    if args.supcon_loss_weight < 0.0:
        raise ValueError("--supcon-loss-weight must be >= 0.")
    if args.supcon_temperature <= 0.0:
        raise ValueError("--supcon-temperature must be > 0.")
    if ssl_method == "mae" and float(args.supcon_loss_weight) > 0.0:
        raise ValueError("MAE pretrain khong dung --supcon-loss-weight; hay dat ve 0.")

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ensure_dir(Path(args.output_dir) / str(args.run_name))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    data_spec = load_data_spec(Path(args.data))
    if str(data_spec.data_format).strip().lower() not in {"classification_folder", "yolo"}:
        raise ValueError("Internal SSL pretrain expects classification_folder or YOLO-format data.")

    transform_a = build_ssl_transform(args)
    transform_b = None if ssl_method == "mae" else build_ssl_transform(args)
    base_dataset = build_ssl_dataset(
        data_spec,
        split="train",
        transform=transform_a,
        args=args,
    )
    paired_dataset = None
    if transform_b is not None:
        paired_dataset = build_ssl_dataset(
            data_spec,
            split="train",
            transform=transform_b,
            args=args,
        )
    dataset = TwoViewDataset(base_dataset, dataset_b=paired_dataset)
    labels = base_dataset.labels()
    sampler = None
    if not bool(args.disable_balanced_sampler):
        sampler = StrictBalancedBatchSampler(
            labels,
            batch_size=int(args.batch_size),
            num_classes=int(data_spec.num_classes),
            epoch_multiplier=1.0,
            seed=int(args.seed),
            drop_last=False,
        )

    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=device.type == "cuda",
        context="internal_ssl_train",
        prefetch_factor=2,
        persistent_workers=False,
    )
    if sampler is not None:
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **dataloader_kwargs,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            **dataloader_kwargs,
        )

    model_config = build_v8_model_config(
        int(args.image_size),
        bbox_spatial_fusion=bool(args.bbox_spatial_fusion),
        bbox_spatial_fusion_hidden_dim=int(args.bbox_spatial_fusion_hidden_dim),
        bbox_spatial_fusion_dropout=float(args.bbox_spatial_fusion_dropout),
        bbox_spatial_fusion_logit_scale=float(args.bbox_spatial_fusion_logit_scale),
    )
    if ssl_method == "mae":
        # Reconstruction needs stable one-to-one patch targets. Token pruning has no
        # learned weights, so disabling it here still yields a compatible fine-tune checkpoint.
        model_config.token_pruning = False
    model = create_model(num_classes=int(data_spec.num_classes), model_config=to_serializable(model_config)).to(device)
    embedding_dim = infer_embedding_dim(model, int(args.image_size), device)
    projector: Optional[nn.Module] = None
    decoder: Optional[nn.Module] = None
    teacher_model: Optional[nn.Module] = None
    teacher_projector: Optional[nn.Module] = None
    dino_center: Optional[Tensor] = None
    if ssl_method == "mae":
        decoder = MaskedPatchDecoder(
            input_dim=int(model_config.embed_dim),
            hidden_dim=int(args.mae_decoder_hidden_dim),
            patch_dim=int(3 * int(model_config.patch_size) * int(model_config.patch_size)),
        ).to(device)
    elif ssl_method == "dino":
        projector = DinoProjector(
            input_dim=embedding_dim,
            hidden_dim=int(args.projector_hidden_dim),
            output_dim=int(args.dino_output_dim),
        ).to(device)
        teacher_model = create_model(
            num_classes=int(data_spec.num_classes),
            model_config=to_serializable(model_config),
        ).to(device)
        teacher_model.load_state_dict(model.state_dict(), strict=True)
        teacher_projector = DinoProjector(
            input_dim=embedding_dim,
            hidden_dim=int(args.projector_hidden_dim),
            output_dim=int(args.dino_output_dim),
        ).to(device)
        teacher_projector.load_state_dict(projector.state_dict(), strict=True)
        teacher_model.eval()
        teacher_projector.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        for parameter in teacher_projector.parameters():
            parameter.requires_grad_(False)
        dino_center = torch.zeros(int(args.dino_output_dim), device=device, dtype=torch.float32)
    else:
        projector = BarlowProjector(
            input_dim=embedding_dim,
            hidden_dim=int(args.projector_hidden_dim),
            output_dim=int(args.projector_output_dim),
        ).to(device)
    init_summary: Optional[Dict[str, object]] = None
    if args.init_checkpoint is not None:
        init_summary = load_model_init_checkpoint(
            model,
            Path(args.init_checkpoint),
            expected_class_names=data_spec.class_names,
        )
        if ssl_method == "dino":
            if teacher_model is None or teacher_projector is None or projector is None:
                raise RuntimeError("DINO teacher was not initialized.")
            teacher_model.load_state_dict(model.state_dict(), strict=True)
            teacher_projector.load_state_dict(projector.state_dict(), strict=True)
            init_summary["dino_teacher_synced_after_init"] = True
    trainable_parameters = list(model.parameters()) + (
        list(decoder.parameters()) if decoder is not None else list(projector.parameters())
    )
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    supcon_criterion = SupervisedContrastiveLoss(
        temperature=float(args.supcon_temperature),
        class_balanced=not bool(args.disable_supcon_class_balanced),
    )

    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    resume_summary: Optional[Dict[str, object]] = None
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Khong tim thay internal SSL checkpoint: {resume_path}")
        resume_checkpoint = load_checkpoint(resume_path, map_location="cpu")
        resume_kind = str(resume_checkpoint.get("checkpoint_kind", ""))
        allowed_resume_kinds = {
            "internal_barlow_pretrain",
            "internal_vicreg_pretrain",
            "internal_mae_pretrain",
            "internal_dino_pretrain",
            "internal_ssl_pretrain",
        }
        if resume_kind not in allowed_resume_kinds:
            raise ValueError(f"Checkpoint khong phai internal SSL pretrain: {resume_path}")
        checkpoint_classes = list(resume_checkpoint.get("class_names", []))
        if checkpoint_classes and checkpoint_classes != list(data_spec.class_names):
            raise ValueError("Class names trong resume checkpoint khong khop data.yaml hien tai.")
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
        if projector is not None and isinstance(resume_checkpoint.get("projector_state"), dict):
            projector.load_state_dict(resume_checkpoint["projector_state"], strict=True)
        if decoder is not None:
            if not isinstance(resume_checkpoint.get("decoder_state"), dict):
                raise ValueError("MAE resume checkpoint thieu decoder_state.")
            decoder.load_state_dict(resume_checkpoint["decoder_state"], strict=True)
        loaded_epoch = int(resume_checkpoint.get("epoch", 0) or 0)
        best_epoch = int(resume_checkpoint.get("best_epoch", loaded_epoch) or loaded_epoch)
        best_loss = float(resume_checkpoint.get("best_ssl_loss", float("inf")) or float("inf"))
        resume_ssl_config = resume_checkpoint.get("ssl_config", {})
        resume_method = (
            str(resume_ssl_config.get("ssl_method", "barlow")).strip().lower()
            if isinstance(resume_ssl_config, dict)
            else "barlow"
        )
        if resume_kind == "internal_vicreg_pretrain":
            resume_method = "vicreg"
        if resume_kind == "internal_mae_pretrain":
            resume_method = "mae"
        if resume_kind == "internal_dino_pretrain":
            resume_method = "dino"
        if ssl_method == "dino":
            if teacher_model is None or teacher_projector is None or projector is None:
                raise RuntimeError("DINO teacher was not initialized.")
            teacher_model.load_state_dict(model.state_dict(), strict=True)
            teacher_projector.load_state_dict(projector.state_dict(), strict=True)
        optimizer_loaded = False
        if (
            resume_method == ssl_method
            and (not bool(args.resume_reset_optimizer))
            and isinstance(resume_checkpoint.get("optimizer_state"), dict)
        ):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            optimizer_loaded = True
        if resume_method != ssl_method:
            best_epoch = 0
            best_loss = float("inf")
        if not bool(args.resume_reset_epoch):
            start_epoch = loaded_epoch + 1
        resume_summary = {
            "path": str(resume_path),
            "checkpoint_kind": resume_kind,
            "resume_method": resume_method,
            "target_method": ssl_method,
            "loaded_epoch": int(loaded_epoch),
            "start_epoch": int(start_epoch),
            "best_epoch": int(best_epoch),
            "best_ssl_loss": float(best_loss),
            "best_loss_reset_for_method_change": bool(resume_method != ssl_method),
            "optimizer_loaded": bool(optimizer_loaded),
            "reset_epoch": bool(args.resume_reset_epoch),
        }

    ssl_config: Dict[str, object] = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": str(Path(args.data).resolve()),
        "data_format": str(data_spec.data_format),
        "train_only": True,
        "class_counts": base_dataset.class_counts(int(data_spec.num_classes)),
        "dataset_report": (
            base_dataset.quality_report()
            if hasattr(base_dataset, "quality_report")
            else None
        ),
        "balanced_sampler": not bool(args.disable_balanced_sampler),
        "balanced_exposure": sampler.exposure_summary() if sampler is not None else None,
        "dataloader": dataloader_summary,
        "device": str(device),
        "embedding_dim": int(embedding_dim),
        "mae_decoder": (
            {
                "input_dim": int(model_config.embed_dim),
                "hidden_dim": int(args.mae_decoder_hidden_dim),
                "patch_dim": int(3 * int(model_config.patch_size) * int(model_config.patch_size)),
            }
            if decoder is not None
            else None
        ),
        "dino_teacher": (
            {
                "output_dim": int(args.dino_output_dim),
                "student_temperature": float(args.dino_student_temperature),
                "teacher_temperature": float(args.dino_teacher_temperature),
                "center_momentum": float(args.dino_center_momentum),
                "teacher_momentum": float(args.dino_teacher_momentum),
            }
            if ssl_method == "dino"
            else None
        ),
        "mae_token_pruning_disabled_for_pretrain": bool(ssl_method == "mae"),
        "init_checkpoint": init_summary or {"enabled": False},
        "resume": resume_summary,
    }
    json_dump(run_dir / "ssl_config.json", ssl_config)
    print(json.dumps({"internal_ssl_preflight": ssl_config}, ensure_ascii=False), flush=True)
    if bool(args.dry_run):
        print(json.dumps({"status": "dry_run_ok", "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
        return 0

    train_amp = not bool(args.disable_amp)
    amp_dtype = resolve_amp_dtype(device) if train_amp and device.type == "cuda" else None
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and train_amp and amp_dtype != torch.bfloat16,
    )

    history = []
    started = time.time()
    end_epoch = (start_epoch + int(args.epochs) - 1) if not bool(args.resume_reset_epoch) else int(args.epochs)
    for epoch in range(start_epoch, end_epoch + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        if projector is not None:
            projector.train()
        if decoder is not None:
            decoder.train()
        total_loss = 0.0
        total_barlow_loss = 0.0
        total_vicreg_loss = 0.0
        total_vicreg_invariance_loss = 0.0
        total_vicreg_variance_loss = 0.0
        total_vicreg_covariance_loss = 0.0
        total_mae_loss = 0.0
        total_mae_mask_fraction = 0.0
        total_mae_patch_count = 0.0
        total_dino_loss = 0.0
        total_dino_teacher_entropy = 0.0
        total_dino_center_norm = 0.0
        total_supcon_loss = 0.0
        total_batches = len(dataloader)
        if int(args.max_train_batches) > 0:
            total_batches = min(total_batches, int(args.max_train_batches))
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(dataloader, total=total_batches, desc=f"SSL epoch {epoch}", dynamic_ncols=True)
        batch_count = 0
        for batch_index, batch in enumerate(iterator):
            if int(args.max_train_batches) > 0 and batch_index >= int(args.max_train_batches):
                break
            view_a, view_b, _ = batch
            labels = batch[2].to(device, non_blocking=True)
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            with autocast_context(device, train_amp):
                if ssl_method == "mae":
                    if decoder is None:
                        raise RuntimeError("MAE decoder was not initialized.")
                    ssl_base_loss, mae_stats = mae_reconstruction_loss(
                        model,
                        decoder,
                        view_a,
                        patch_size=int(model_config.patch_size),
                        mask_ratio=float(args.mae_mask_ratio),
                        foreground_weight=float(args.mae_foreground_weight),
                        detail_weight=float(args.mae_detail_weight),
                    )
                    barlow_loss = ssl_base_loss * 0.0
                    vicreg_base_loss = ssl_base_loss * 0.0
                    vicreg_invariance_loss = ssl_base_loss * 0.0
                    vicreg_variance_loss = ssl_base_loss * 0.0
                    vicreg_covariance_loss = ssl_base_loss * 0.0
                    mae_loss = ssl_base_loss
                    mae_mask_fraction = float(mae_stats["mask_fraction"])
                    mae_patch_count = float(mae_stats["patch_count"])
                    dino_loss = ssl_base_loss * 0.0
                    teacher_entropy = ssl_base_loss * 0.0
                    emb_a = None
                    emb_b = None
                else:
                    if projector is None:
                        raise RuntimeError("SSL projector was not initialized.")
                    emb_a = encode(model, view_a)
                    emb_b = encode(model, view_b)
                    proj_a = projector(emb_a)
                    proj_b = projector(emb_b)
                    if ssl_method == "dino":
                        if (
                            teacher_model is None
                            or teacher_projector is None
                            or dino_center is None
                        ):
                            raise RuntimeError("DINO teacher/center was not initialized.")
                        with torch.no_grad():
                            teacher_emb_a = encode(teacher_model, view_a)
                            teacher_emb_b = encode(teacher_model, view_b)
                            teacher_proj_a = teacher_projector(teacher_emb_a)
                            teacher_proj_b = teacher_projector(teacher_emb_b)
                        ssl_base_loss = dino_self_distillation_loss(
                            proj_a,
                            proj_b,
                            teacher_proj_a,
                            teacher_proj_b,
                            center=dino_center,
                            student_temperature=float(args.dino_student_temperature),
                            teacher_temperature=float(args.dino_teacher_temperature),
                        )
                        teacher_logits = torch.cat(
                            [teacher_proj_a.detach().float(), teacher_proj_b.detach().float()],
                            dim=0,
                        )
                        teacher_center = dino_center.view(1, -1)
                        teacher_prob = F.softmax(
                            (teacher_logits - teacher_center) / max(float(args.dino_teacher_temperature), 1e-6),
                            dim=-1,
                        )
                        teacher_entropy = -(teacher_prob * teacher_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
                        batch_center = teacher_logits.mean(dim=0)
                        dino_center = (
                            float(args.dino_center_momentum) * dino_center
                            + (1.0 - float(args.dino_center_momentum)) * batch_center.to(device=dino_center.device)
                        ).detach()
                        barlow_loss = ssl_base_loss * 0.0
                        vicreg_base_loss = ssl_base_loss * 0.0
                        vicreg_invariance_loss = ssl_base_loss * 0.0
                        vicreg_variance_loss = ssl_base_loss * 0.0
                        vicreg_covariance_loss = ssl_base_loss * 0.0
                        dino_loss = ssl_base_loss
                    elif ssl_method == "vicreg":
                        ssl_base_loss, vicreg_parts = vicreg_loss(
                            proj_a,
                            proj_b,
                            invariance_coeff=float(args.vicreg_invariance_coeff),
                            variance_coeff=float(args.vicreg_variance_coeff),
                            covariance_coeff=float(args.vicreg_covariance_coeff),
                            gamma=float(args.vicreg_gamma),
                            eps=float(args.vicreg_eps),
                        )
                        barlow_loss = ssl_base_loss * 0.0
                        vicreg_base_loss = ssl_base_loss
                        vicreg_invariance_loss = vicreg_parts["invariance"]
                        vicreg_variance_loss = vicreg_parts["variance"]
                        vicreg_covariance_loss = vicreg_parts["covariance"]
                        dino_loss = ssl_base_loss * 0.0
                        teacher_entropy = ssl_base_loss * 0.0
                    else:
                        ssl_base_loss = barlow_twins_loss(
                            proj_a,
                            proj_b,
                            lambd=float(args.barlow_lambda),
                        )
                        barlow_loss = ssl_base_loss
                        vicreg_base_loss = ssl_base_loss * 0.0
                        vicreg_invariance_loss = ssl_base_loss * 0.0
                        vicreg_variance_loss = ssl_base_loss * 0.0
                        vicreg_covariance_loss = ssl_base_loss * 0.0
                        dino_loss = ssl_base_loss * 0.0
                        teacher_entropy = ssl_base_loss * 0.0
                    mae_loss = ssl_base_loss * 0.0
                    mae_mask_fraction = 0.0
                    mae_patch_count = 0.0
                supcon_loss = ssl_base_loss * 0.0
                loss = ssl_base_loss
                if float(args.supcon_loss_weight) > 0.0:
                    supcon_embeddings = torch.cat([emb_a, emb_b], dim=0)
                    supcon_targets = torch.cat([labels, labels], dim=0)
                    supcon_loss = supcon_criterion(supcon_embeddings, supcon_targets)
                    loss = loss + float(args.supcon_loss_weight) * supcon_loss
                scaled_loss = loss / float(args.grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % int(args.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu().item())
            barlow_value = float(barlow_loss.detach().cpu().item())
            vicreg_value = float(vicreg_base_loss.detach().cpu().item())
            vicreg_invariance_value = float(vicreg_invariance_loss.detach().cpu().item())
            vicreg_variance_value = float(vicreg_variance_loss.detach().cpu().item())
            vicreg_covariance_value = float(vicreg_covariance_loss.detach().cpu().item())
            supcon_value = float(supcon_loss.detach().cpu().item())
            total_loss += loss_value
            total_barlow_loss += barlow_value
            total_vicreg_loss += vicreg_value
            total_vicreg_invariance_loss += vicreg_invariance_value
            total_vicreg_variance_loss += vicreg_variance_value
            total_vicreg_covariance_loss += vicreg_covariance_value
            total_mae_loss += float(mae_loss.detach().cpu().item())
            total_mae_mask_fraction += float(mae_mask_fraction)
            total_mae_patch_count += float(mae_patch_count)
            total_dino_loss += float(dino_loss.detach().cpu().item())
            total_dino_teacher_entropy += float(teacher_entropy.detach().cpu().item())
            total_dino_center_norm += float(
                dino_center.norm().detach().cpu().item() if dino_center is not None else 0.0
            )
            total_supcon_loss += supcon_value
            batch_count += 1
            iterator.set_postfix(
                method=ssl_method,
                loss=f"{loss_value:.4f}",
                mae_mask=f"{mae_mask_fraction:.3f}",
                dino_entropy=f"{float(teacher_entropy.detach().cpu().item()):.3f}",
                supcon=f"{supcon_value:.4f}",
            )
            if ssl_method == "dino":
                if teacher_model is None or teacher_projector is None:
                    raise RuntimeError("DINO teacher was not initialized.")
                update_ema_model(model, teacher_model, float(args.dino_teacher_momentum))
                update_ema_model(projector, teacher_projector, float(args.dino_teacher_momentum))
        if batch_count % int(args.grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        epoch_loss = total_loss / max(1, batch_count)
        row = {
            "epoch": int(epoch),
            "ssl_method": ssl_method,
            "ssl_loss": float(epoch_loss),
            "barlow_loss": float(total_barlow_loss / max(1, batch_count)),
            "vicreg_loss": float(total_vicreg_loss / max(1, batch_count)),
            "vicreg_invariance_loss": float(
                total_vicreg_invariance_loss / max(1, batch_count)
            ),
            "vicreg_variance_loss": float(total_vicreg_variance_loss / max(1, batch_count)),
            "vicreg_covariance_loss": float(
                total_vicreg_covariance_loss / max(1, batch_count)
            ),
            "mae_loss": float(total_mae_loss / max(1, batch_count)),
            "mae_mask_fraction": float(total_mae_mask_fraction / max(1, batch_count)),
            "mae_patch_count": float(total_mae_patch_count / max(1, batch_count)),
            "dino_loss": float(total_dino_loss / max(1, batch_count)),
            "dino_teacher_entropy": float(total_dino_teacher_entropy / max(1, batch_count)),
            "dino_center_norm": float(total_dino_center_norm / max(1, batch_count)),
            "supcon_loss": float(total_supcon_loss / max(1, batch_count)),
            "supcon_loss_weight": float(args.supcon_loss_weight),
            "batches": int(batch_count),
            "elapsed": format_seconds(time.time() - started),
        }
        history.append(row)
        json_dump(run_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch_loss < best_loss:
            best_loss = float(epoch_loss)
            best_epoch = int(epoch)
            save_ssl_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=model,
                projector=projector,
                decoder=decoder,
                optimizer=optimizer,
                model_config=model_config,
                class_names=data_spec.class_names,
                data_yaml=Path(args.data),
                epoch=epoch,
                best_epoch=best_epoch,
                best_loss=best_loss,
                ssl_config=ssl_config,
            )
        save_ssl_checkpoint(
            path=checkpoints_dir / "last.pt",
            model=model,
            projector=projector,
            decoder=decoder,
            optimizer=optimizer,
            model_config=model_config,
            class_names=data_spec.class_names,
            data_yaml=Path(args.data),
            epoch=epoch,
            best_epoch=best_epoch,
            best_loss=best_loss,
            ssl_config=ssl_config,
        )

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_ssl_loss": float(best_loss),
        "checkpoint": str((checkpoints_dir / "best.pt").resolve()),
        "total_seconds": float(time.time() - started),
    }
    json_dump(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
