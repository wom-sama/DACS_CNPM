"""
================================================================================
DEBUG & OPTIMIZATION MODULE
================================================================================
Comprehensive debugging và optimization fixes cho DETR ViT-Registers model.

Bao gồm:
1. Fix Detection F1 = 0.0 (Bbox format & IoU debugging)
2. Training Scheduler & Progressive Freezing Optimization
3. Gradient Checkpointing Memory Optimization
4. Multi-scale Training & Test-Time Augmentation (TTA)

Author: ML Engineer
Date: May 2026
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from trkh.core.utils import build_safe_dataloader_kwargs, maybe_enable_dataset_image_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================================================================
# TASK 1: FIX DETECTION F1 = 0.0 (Bbox Format & IoU Debugging)
# ================================================================================

class BboxDebugger:
    """
    Debugging utilities để fix F1 = 0.0 issue.
    Logs tọa độ predictions vs ground truth, kiểm tra format consistency.
    """

    @staticmethod
    def log_bbox_coordinates(
        predictions: Tensor,
        targets: Tensor,
        batch_idx: int = 0,
        query_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log chi tiết tọa độ bounding boxes (predictions vs ground truth).

        Args:
            predictions: Tensor shape [batch, num_queries, 4] (xywh hoặc xyxy)
            targets: Dict with 'boxes' key, shape [num_targets, 4]
            batch_idx: Chỉ số batch cần log
            query_idx: Nếu None, log tất cả queries; nếu chỉ định, log 1 query
            prefix: Prefix cho log message
        """
        prefix = f"[{prefix}] " if prefix else ""

        # Extract boxes từ predictions (giả sử shape [batch, num_queries, 4])
        if predictions.dim() == 3:
            pred_boxes_batch = predictions[batch_idx]  # [num_queries, 4]
        else:
            pred_boxes_batch = predictions

        if isinstance(targets, dict):
            target_boxes = targets.get('boxes', targets.get('labels'))
        else:
            target_boxes = targets

        if target_boxes.dim() == 3:
            target_boxes_batch = target_boxes[batch_idx]
        else:
            target_boxes_batch = target_boxes

        logger.info(f"\n{prefix}BBOX DEBUG INFO (Batch {batch_idx}):")
        logger.info(f"{prefix}  Pred boxes shape: {pred_boxes_batch.shape}")
        logger.info(f"{prefix}  Target boxes shape: {target_boxes_batch.shape}")

        # Log predictions
        if query_idx is not None:
            logger.info(f"{prefix}  Pred Query {query_idx}: {pred_boxes_batch[query_idx].tolist()}")
        else:
            for q_idx, pred_box in enumerate(pred_boxes_batch[:5]):  # Log first 5 queries
                logger.info(f"{prefix}  Pred Query {q_idx}: {pred_box.tolist()}")

        # Log targets
        if target_boxes_batch.numel() > 0:
            for t_idx, target_box in enumerate(target_boxes_batch[:5]):  # Log first 5 targets
                logger.info(f"{prefix}  Target {t_idx}: {target_box.tolist()}")
        else:
            logger.info(f"{prefix}  No targets in this batch!")

        # Log statistics
        logger.info(f"{prefix}  Pred min/max: {pred_boxes_batch.min():.4f} / {pred_boxes_batch.max():.4f}")
        if target_boxes_batch.numel() > 0:
            logger.info(f"{prefix}  Target min/max: {target_boxes_batch.min():.4f} / {target_boxes_batch.max():.4f}")

    @staticmethod
    def check_bbox_format(
        boxes: Tensor,
        expected_format: str = "xywh",
        name: str = "boxes",
    ) -> bool:
        """
        Kiểm tra xem boxes có trong định dạng xywh hay xyxy.

        Args:
            boxes: Tensor [N, 4] containing bounding boxes
            expected_format: "xywh" hoặc "xyxy"
            name: Tên của boxes (cho logging)

        Returns:
            True nếu format khớp expected_format
        """
        if boxes.numel() == 0:
            return True

        # Clamp để kiểm tra
        if expected_format == "xywh":
            # xywh: center coordinates + width/height, thường trong [0, 1]
            # width và height phải dương
            is_valid = (boxes[..., 2] > 0) & (boxes[..., 3] > 0)
            logger.info(f"{name} (xywh): Valid format = {is_valid.float().mean().item():.2%}")
        elif expected_format == "xyxy":
            # xyxy: x1 < x2 và y1 < y2
            is_valid = (boxes[..., 0] < boxes[..., 2]) & (boxes[..., 1] < boxes[..., 3])
            logger.info(f"{name} (xyxy): Valid format = {is_valid.float().mean().item():.2%}")
        else:
            raise ValueError(f"Unknown format: {expected_format}")

        return is_valid.all().item()

    @staticmethod
    def convert_bbox_format(
        boxes: Tensor,
        from_format: str,
        to_format: str,
    ) -> Tensor:
        """
        Chuyển đổi bbox từ format này sang format khác.

        Args:
            boxes: [N, 4] bounding boxes
            from_format: "xywh" hoặc "xyxy"
            to_format: "xywh" hoặc "xyxy"

        Returns:
            Converted boxes
        """
        if from_format == to_format:
            return boxes

        if from_format == "xywh" and to_format == "xyxy":
            # xywh -> xyxy
            x_center, y_center, width, height = boxes.unbind(dim=-1)
            x1 = x_center - width / 2.0
            y1 = y_center - height / 2.0
            x2 = x_center + width / 2.0
            y2 = y_center + height / 2.0
            return torch.stack((x1, y1, x2, y2), dim=-1)

        elif from_format == "xyxy" and to_format == "xywh":
            # xyxy -> xywh
            x1, y1, x2, y2 = boxes.unbind(dim=-1)
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            return torch.stack((x_center, y_center, width, height), dim=-1)

        else:
            raise ValueError(f"Unknown conversion: {from_format} -> {to_format}")

    @staticmethod
    def validate_iou_calculation(
        pred_boxes: Tensor,
        target_boxes: Tensor,
        format: str = "xywh",
        iou_threshold: float = 0.5,
    ) -> Dict[str, object]:
        """
        Validate IoU calculation và check nếu có boxes vượt ngưỡng.

        Args:
            pred_boxes: [N, 4] predicted boxes
            target_boxes: [M, 4] target boxes
            format: "xywh" hoặc "xyxy"
            iou_threshold: Ngưỡng IoU

        Returns:
            Dict chứa statistics về IoU
        """
        # Convert to xyxy nếu cần
        if format == "xywh":
            pred_xyxy = BboxDebugger.convert_bbox_format(pred_boxes, "xywh", "xyxy")
            target_xyxy = BboxDebugger.convert_bbox_format(target_boxes, "xywh", "xyxy")
        else:
            pred_xyxy = pred_boxes
            target_xyxy = target_boxes

        # Clamp to [0, 1]
        pred_xyxy = pred_xyxy.clamp(0, 1)
        target_xyxy = target_xyxy.clamp(0, 1)

        # Calculate IoU
        iou = _calculate_iou_matrix(pred_xyxy, target_xyxy)

        # Statistics
        max_iou = iou.max(dim=1)[0] if iou.numel() > 0 else torch.tensor([])

        stats = {
            "num_predictions": pred_boxes.shape[0],
            "num_targets": target_boxes.shape[0],
            "iou_matrix_shape": iou.shape,
            "max_iou_mean": float(max_iou.mean().item()) if max_iou.numel() > 0 else 0.0,
            "max_iou_min": float(max_iou.min().item()) if max_iou.numel() > 0 else 0.0,
            "matches_above_threshold": int((max_iou >= iou_threshold).sum().item()) if max_iou.numel() > 0 else 0,
        }

        logger.info(f"\nIoU Validation:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats


def _calculate_iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Tính IoU matrix giữa 2 tập boxes (xyxy format).

    boxes1: [N, 4]
    boxes2: [M, 4]
    return: [N, M] IoU matrix
    """
    # boxes1: [N, 4] -> [N, 1, 4]
    # boxes2: [M, 4] -> [1, M, 4]

    x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(dim=-1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(dim=-1)

    # Calculate intersection
    inter_x1 = torch.maximum(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
    inter_y1 = torch.maximum(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
    inter_x2 = torch.minimum(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
    inter_y2 = torch.minimum(y2_1.unsqueeze(1), y2_2.unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Calculate union
    area1 = ((x2_1 - x1_1) * (y2_1 - y1_1)).unsqueeze(1)
    area2 = ((x2_2 - x1_2) * (y2_2 - y1_2)).unsqueeze(0)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou


# ================================================================================
# TASK 2: TRAINING SCHEDULER & PROGRESSIVE FREEZING
# ================================================================================

class TrainingScheduler:
    """
    Quản lý training schedule với 2 stages:
    - Stage 1: Classification chỉ (5-8 epochs) - ngắn hơn vì hội tụ nhanh
    - Stage 2: Progressive unfreezing backbone (35-40 epochs)
    """

    def __init__(
        self,
        model: nn.Module,
        stage1_epochs: int = 5,
        stage2_epochs: int = 40,
        total_epochs: int = 45,
    ):
        self.model = model
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch."""
        self.current_epoch = epoch

    def get_current_stage(self) -> int:
        """Return current training stage (1 or 2)."""
        if self.current_epoch < self.stage1_epochs:
            return 1
        return 2

    def get_frozen_params_count(self) -> int:
        """Get number of frozen parameters."""
        count = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                count += param.numel()
        return count

    def get_trainable_params_count(self) -> int:
        """Get number of trainable parameters."""
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                count += param.numel()
        return count

    def apply_progressive_unfreezing(self) -> str:
        """
        Áp dụng progressive unfreezing trong Stage 2.

        Stage 2 schedule:
        - Epochs 0-20 (từ start của Stage 2): Freeze backbone, train detection head only
        - Epochs 20-40: Unfreeze 2 transformer layers cuối cùng
        - Epochs 40+: Unfreeze toàn bộ model

        Returns:
            Status message describing current freezing state
        """
        if self.current_epoch < self.stage1_epochs:
            # Stage 1: Freeze mọi thứ ngoài detection head
            return self._freeze_stage1()
        else:
            # Stage 2: Progressive unfreezing
            stage2_epoch = self.current_epoch - self.stage1_epochs
            return self._unfreeze_stage2(stage2_epoch)

    def _freeze_stage1(self) -> str:
        """Freeze backbone, train detection head only."""
        self._freeze_backbone()
        self._set_head_trainable(True)

        frozen = self.get_frozen_params_count()
        trainable = self.get_trainable_params_count()
        return f"Stage 1 - Frozen: {frozen:,} | Trainable: {trainable:,}"

    def _unfreeze_stage2(self, stage2_epoch: int) -> str:
        """Progressive unfreezing in Stage 2."""
        backbone = self._get_backbone()

        if backbone is None:
            return "Could not find backbone for unfreezing"

        if stage2_epoch < 20:
            # Epochs 0-19 of Stage 2: Freeze backbone, train detection head
            self._freeze_backbone()
            status = "Stage 2.1 (0-20) - Backbone frozen, head trainable"

        elif stage2_epoch < 40:
            # Epochs 20-39 of Stage 2: Unfreeze last 2 transformer layers
            self._freeze_backbone()
            self._unfreeze_last_n_layers(backbone, n=2)
            status = "Stage 2.2 (20-40) - Last 2 layers unfrozen"

        else:
            # Epochs 40+ of Stage 2: Unfreeze toàn bộ backbone
            self._unfreeze_backbone()
            status = "Stage 2.3 (40+) - Full backbone unfrozen"

        self._set_head_trainable(True)
        frozen = self.get_frozen_params_count()
        trainable = self.get_trainable_params_count()
        return f"{status} | Frozen: {frozen:,} | Trainable: {trainable:,}"

    def _get_backbone(self) -> Optional[nn.Module]:
        """Get backbone module."""
        if hasattr(self.model, 'vit_backbone'):
            return self.model.vit_backbone
        elif hasattr(self.model, 'backbone'):
            return self.model.backbone
        elif hasattr(self.model, 'blocks'):
            return self.model
        return None

    def _backbone_attribute_names(self) -> Tuple[str, ...]:
        if hasattr(self.model, 'vit_backbone') or hasattr(self.model, 'backbone'):
            return ()
        return ('stem', 'patch_embed', 'blocks', 'norm')

    def _freeze_backbone(self) -> None:
        """Freeze feature-extractor parameters for supported model layouts."""
        if hasattr(self.model, 'vit_backbone'):
            self._freeze_module(self.model.vit_backbone)
            return
        if hasattr(self.model, 'backbone'):
            self._freeze_module(self.model.backbone)
            return

        for module_name in self._backbone_attribute_names():
            module = getattr(self.model, module_name, None)
            if module is not None:
                self._freeze_module(module)
        for parameter_name in ('cls_token', 'register_tokens', 'pos_embed'):
            parameter = getattr(self.model, parameter_name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze feature-extractor parameters for supported model layouts."""
        if hasattr(self.model, 'vit_backbone'):
            self._unfreeze_module(self.model.vit_backbone)
            return
        if hasattr(self.model, 'backbone'):
            self._unfreeze_module(self.model.backbone)
            return

        for module_name in self._backbone_attribute_names():
            module = getattr(self.model, module_name, None)
            if module is not None:
                self._unfreeze_module(module)
        for parameter_name in ('cls_token', 'register_tokens', 'pos_embed'):
            parameter = getattr(self.model, parameter_name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad = True

    def _set_head_trainable(self, trainable: bool) -> None:
        """Keep classifier and detector heads trainable while stages change."""
        head_names = (
            'head',
            'detection_head',
            'classification_head',
            'bbox_head',
            'decoder',
            'query_embed',
        )
        for module_name in head_names:
            module = getattr(self.model, module_name, None)
            if module is None or not hasattr(module, 'parameters'):
                continue
            for param in module.parameters():
                param.requires_grad = bool(trainable)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        """Freeze all parameters in module."""
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _unfreeze_module(module: nn.Module) -> None:
        """Unfreeze all parameters in module."""
        for param in module.parameters():
            param.requires_grad = True

    @staticmethod
    def _unfreeze_last_n_layers(module: nn.Module, n: int = 2) -> None:
        """
        Unfreeze last N transformer layers.
        Giả sử module có attribute 'blocks' hoặc 'layers' (ViT structure).
        """
        # Try to find transformer blocks
        blocks = None
        if hasattr(module, 'blocks'):
            blocks = module.blocks
        elif hasattr(module, 'layers'):
            blocks = module.layers

        if blocks is None or not hasattr(blocks, '__len__'):
            return

        # Unfreeze last n blocks
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True


# ================================================================================
# TASK 3: GRADIENT CHECKPOINTING OPTIMIZATION
# ================================================================================

class GradientCheckpointingEnabler:
    """
    Enable gradient checkpointing để giảm memory usage (~30%) với cost tốc độ (5-10%).
    """

    @staticmethod
    def enable_gradient_checkpointing(
        model: nn.Module,
        enable_for_layers: Optional[List[str]] = None,
    ) -> None:
        """
        Enable gradient checkpointing cho các layers chỉ định.

        Args:
            model: PyTorch model
            enable_for_layers: List tên layers (nếu None, enable cho transformer blocks)
        """
        if not enable_for_layers:
            enable_for_layers = ['blocks', 'transformer_layers', 'decoder']

        if hasattr(model, 'set_gradient_checkpointing'):
            model.set_gradient_checkpointing(True)
            logger.info("Enabled model gradient checkpointing flag")
        elif hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = True
            logger.info("Enabled model gradient checkpointing flag")

        for module_name, module in model.named_modules():
            # Check if this is a layer cần enable gradient checkpointing
            for layer_key in enable_for_layers:
                if layer_key in module_name.lower():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                        logger.info(f"Enabled gradient checkpointing for: {module_name}")
                    break

    @staticmethod
    def enable_activation_checkpointing(
        model: nn.Module,
        checkpoint_segments: int = 1,
    ) -> nn.Module:
        """
        Wrap transformer blocks với activation checkpointing.

        Args:
            model: Original model
            checkpoint_segments: Số segments để checkpoint

        Returns:
            Model với activation checkpointing enabled
        """
        from torch.utils.checkpoint import checkpoint

        # Tìm transformer blocks
        if hasattr(model, 'vit_backbone') and hasattr(model.vit_backbone, 'blocks'):
            blocks = model.vit_backbone.blocks
            _wrap_blocks_with_checkpoint(blocks)
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
            blocks = model.backbone.blocks
            _wrap_blocks_with_checkpoint(blocks)

        return model


def _wrap_blocks_with_checkpoint(blocks: nn.ModuleList) -> None:
    """Wrap each block with checkpoint function."""
    from torch.utils.checkpoint import checkpoint

    for idx, block in enumerate(blocks):
        # Store original forward
        original_forward = block.forward

        # Create wrapper
        def make_checkpoint_forward(orig_forward):
            def checkpoint_forward(x, *args, **kwargs):
                return checkpoint(orig_forward, x, *args, use_reentrant=False, **kwargs)
            return checkpoint_forward

        # Replace forward with checkpoint version
        block.forward = make_checkpoint_forward(original_forward)
        logger.info(f"Wrapped block {idx} with checkpoint")


# ================================================================================
# TASK 4: MULTI-SCALE TRAINING & TEST-TIME AUGMENTATION
# ================================================================================

class MultiScaleTransform:
    """
    Multi-scale training transform: tính chọn ngẫu nhiên kích thước ảnh
    từ tập (224, 320, 416, 512, 640) và rotate mỗi 10 epochs.
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (224, 320, 416, 512, 640),
        update_interval: int = 10,
        scale_select_prob: Optional[Tuple[float, ...]] = None,
        base_transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            scales: Tuple của các kích thước ảnh để rotate
            update_interval: Số epochs giữa mỗi lần thay đổi scale
            scale_select_prob: Probability weights cho mỗi scale
            base_transform: Base augmentation transforms
        """
        self.scales = scales
        self.update_interval = update_interval
        self.current_epoch = 0
        self.current_scale_idx = 0

        # Default probabilities: more weight on smaller scales
        if scale_select_prob is None:
            self.scale_select_prob = tuple(1.0 / len(scales) for _ in scales)
        else:
            if len(scale_select_prob) != len(scales):
                raise ValueError("scale_select_prob phải có độ dài bằng scales")
            total = sum(scale_select_prob)
            self.scale_select_prob = tuple(p / total for p in scale_select_prob)

        self.base_transform = base_transform

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch và rotate scale nếu cần."""
        self.current_epoch = epoch
        # Rotate scale mỗi update_interval epochs
        self.current_scale_idx = (epoch // self.update_interval) % len(self.scales)

    def get_current_scale(self) -> int:
        """Get current image size."""
        return self.scales[self.current_scale_idx]

    def __call__(self, image: Tensor) -> Tensor:
        """
        Apply transform với current scale.

        Args:
            image: PIL Image hoặc Tensor

        Returns:
            Transformed image
        """
        current_size = self.get_current_scale()

        # Convert to PIL if needed
        if isinstance(image, Tensor):
            from torchvision.transforms import ToPILImage
            image = ToPILImage()(image)

        # Resize to current scale
        import torchvision.transforms.functional as F_trans
        image = F_trans.resize(image, current_size, interpolation=2)

        # Apply base transforms
        if self.base_transform:
            image = self.base_transform(image)
        else:
            # Default to_tensor
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)

        return image


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) cho inference.
    Chạy inference trên 3-4 augmentation, sau đó average predictions.
    """

    def __init__(
        self,
        brightness_delta: float = 0.08,
        contrast_delta: float = 0.08,
        saturation_delta: float = 0.08,
        num_aug: int = 4,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            brightness_delta: Brightness augmentation delta
            contrast_delta: Contrast augmentation delta
            saturation_delta: Saturation augmentation delta
            num_aug: Số augmentations khác nhau
        """
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.saturation_delta = saturation_delta
        self.num_aug = num_aug
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def get_augmented_images(self, images: Tensor) -> List[Tensor]:
        """
        Generate list của augmented images.

        Args:
            images: [B, C, H, W] input images

        Returns:
            List[Tensor] của B * num_aug images
        """
        channel_dim = images.ndim - 3
        view_shape = [1] * images.ndim
        view_shape[channel_dim] = len(self.mean)
        mean = images.new_tensor(self.mean).view(*view_shape)
        std = images.new_tensor(self.std).view(*view_shape)

        def normalize(raw_images: Tensor) -> Tensor:
            return (raw_images - mean) / std

        raw_images = torch.clamp(images * std + mean, 0.0, 1.0)
        augmented = [images]

        if self.brightness_delta > 0.0:
            augmented.append(
                normalize(
                    transforms.functional.adjust_brightness(
                        raw_images,
                        brightness_factor=1.0 + float(self.brightness_delta),
                    )
                )
            )
        if self.contrast_delta > 0.0:
            augmented.append(
                normalize(
                    transforms.functional.adjust_contrast(
                        raw_images,
                        contrast_factor=1.0 + float(self.contrast_delta),
                    )
                )
            )
        if self.saturation_delta > 0.0:
            augmented.append(
                normalize(
                    transforms.functional.adjust_saturation(
                        raw_images,
                        saturation_factor=1.0 + float(self.saturation_delta),
                    )
                )
            )
        if len(augmented) < self.num_aug:
            combined = transforms.functional.adjust_brightness(
                raw_images,
                brightness_factor=1.0 + float(self.brightness_delta) * 0.5,
            )
            combined = transforms.functional.adjust_contrast(
                combined,
                contrast_factor=1.0 + float(self.contrast_delta) * 0.5,
            )
            combined = transforms.functional.adjust_saturation(
                combined,
                saturation_factor=1.0 + float(self.saturation_delta) * 0.5,
            )
            augmented.append(normalize(combined))

        return augmented[:self.num_aug]

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        images: Tensor,
        return_logits: bool = True,
        model_kwargs: Optional[Dict[str, Tensor]] = None,
        forward_fn: Optional[Callable[[Tensor], object]] = None,
    ) -> Dict[str, Tensor]:
        """
        Perform TTA inference.

        Args:
            model: Detection model
            images: [B, C, H, W] input images
            return_logits: If True, return logits; else return softmax probabilities

        Returns:
            Dict với 'logits' và 'boxes' (averaged)
        """
        aug_images = self.get_augmented_images(images)

        all_logits = []
        all_boxes = []
        all_objectness_logits = []
        all_quality_logits = []
        all_count_logits = []

        forward_kwargs = dict(model_kwargs or {})
        for aug_img in aug_images:
            output = forward_fn(aug_img) if forward_fn is not None else model(aug_img, **forward_kwargs)

            # Extract logits and boxes
            if isinstance(output, dict):
                logits = output.get('logits')
                boxes = output.get('boxes')
                objectness_logits = output.get('objectness_logits')
                quality_logits = output.get('quality_logits')
                count_logits = output.get('count_logits')
            elif isinstance(output, (tuple, list)):
                logits, boxes = output[0], output[1]
                objectness_logits = output[2] if len(output) >= 3 else None
                count_logits = output[3] if len(output) >= 4 else None
                quality_logits = output[4] if len(output) >= 5 else None
            else:
                logits = output
                boxes = None
                objectness_logits = None
                quality_logits = None
                count_logits = None

            if logits is None:
                raise ValueError("TTA output is missing logits")

            all_logits.append(logits)
            if boxes is not None:
                all_boxes.append(boxes)
            if objectness_logits is not None:
                all_objectness_logits.append(objectness_logits)
            if quality_logits is not None:
                all_quality_logits.append(quality_logits)
            if count_logits is not None:
                all_count_logits.append(count_logits)

        # Average predictions
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        avg_boxes = torch.stack(all_boxes, dim=0).mean(dim=0) if all_boxes else None

        result = {
            'logits': avg_logits,
            'boxes': avg_boxes,
        }
        if all_objectness_logits:
            result['objectness_logits'] = torch.stack(all_objectness_logits, dim=0).mean(dim=0)
        if all_quality_logits:
            result['quality_logits'] = torch.stack(all_quality_logits, dim=0).mean(dim=0)
        if all_count_logits:
            result['count_logits'] = torch.stack(all_count_logits, dim=0).mean(dim=0)
        return result


# ================================================================================
# HELPER UTILITIES
# ================================================================================

def create_multi_scale_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    scales: Tuple[int, ...] = (224, 320, 416, 512, 640),
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader với multi-scale transform.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        scales: Tuple của image sizes
        num_workers: Number of workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    # Create multi-scale transform
    multi_scale_tf = MultiScaleTransform(scales=scales)

    # Wrap dataset
    class MultiScaleDatasetWrapper:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return getattr(self.dataset, name)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Apply multi-scale transform nếu có image
            if isinstance(item, (tuple, list)) and len(item) > 0:
                image = item[0]
                image = self.transform(image)
                return (image,) + tuple(item[1:])
            return item

    wrapped_dataset = MultiScaleDatasetWrapper(dataset, multi_scale_tf)

    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=torch.cuda.is_available(),
        context="debug_multi_scale",
        prefetch_factor=2,
        persistent_workers=True,
        logger=logger,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        wrapped_dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context="debug_multi_scale",
        logger=logger,
    )
    logger.info(
        "Creating multi-scale DataLoader: batch_size=%s requested_workers=%s "
        "effective_workers=%s pin_memory=%s cache=%s",
        batch_size,
        num_workers,
        dataloader_summary["effective_num_workers"],
        dataloader_summary["effective_pin_memory"],
        cache_summary.get("enabled", False),
    )
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


# ================================================================================
# LOGGING & MONITORING
# ================================================================================

class OptimizationMonitor:
    """Monitor training metrics và optimization effects."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("./debug_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'epochs': [],
            'frozen_params': [],
            'trainable_params': [],
            'memory_usage': [],
            'bbox_iou': [],
        }

    def log_freezing_state(self, epoch: int, scheduler: TrainingScheduler) -> None:
        """Log current freezing state."""
        frozen = scheduler.get_frozen_params_count()
        trainable = scheduler.get_trainable_params_count()

        self.history['epochs'].append(epoch)
        self.history['frozen_params'].append(frozen)
        self.history['trainable_params'].append(trainable)

        logger.info(f"Epoch {epoch}: Frozen={frozen:,} | Trainable={trainable:,}")

    def log_memory_usage(self, epoch: int) -> None:
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            self.history['memory_usage'].append(memory)
            logger.info(f"Epoch {epoch}: GPU Memory = {memory:.2f} GB")

    def save_report(self) -> None:
        """Save optimization report."""
        import json
        report_path = self.log_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    logger.info("Debug & Optimization Module Loaded Successfully")
