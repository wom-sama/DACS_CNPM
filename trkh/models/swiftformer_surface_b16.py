from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SURFACE_SPATIAL_B16_MODE = "surface_spatial_b16"
SURFACE_MEAN_CONTROL_B16_MODE = "surface_mean_control_b16"
SURFACE_OFF_B16_MODE = "surface_off_b16"
SUPPORTED_SURFACE_B16_MODES = (
    SURFACE_SPATIAL_B16_MODE,
    SURFACE_MEAN_CONTROL_B16_MODE,
    SURFACE_OFF_B16_MODE,
)

SWIFTFORMER_XS_S2_CHANNELS = 112
SWIFTFORMER_XS_S3_CHANNELS = 220
SWIFTFORMER_XS_NUM_CLASSES = 5
SURFACE_BRANCH_LAYER_SCALE_INIT = 1.0e-3
RELATION_EPS = 1.0e-6
RELATION_SMOOTH_L1_BETA = 0.1


class SwiftFormerSurfaceB16ContractError(ValueError):
    """Raised when a backbone or tensor violates the isolated B16 contract."""


class SurfaceResidualBranchB16(nn.Module):
    """Lightweight S2-to-S3 residual shared by the candidate and control."""

    def __init__(self) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            SWIFTFORMER_XS_S2_CHANNELS,
            SWIFTFORMER_XS_S2_CHANNELS,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=SWIFTFORMER_XS_S2_CHANNELS,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(SWIFTFORMER_XS_S2_CHANNELS)
        self.activation = nn.SiLU()
        self.pointwise = nn.Conv2d(
            SWIFTFORMER_XS_S2_CHANNELS,
            SWIFTFORMER_XS_S3_CHANNELS,
            kernel_size=1,
            bias=False,
        )
        self.layer_scale = nn.Parameter(
            torch.full(
                (1, SWIFTFORMER_XS_S3_CHANNELS, 1, 1),
                SURFACE_BRANCH_LAYER_SCALE_INIT,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # ``fan_in`` is intentional for the grouped depthwise kernel: its
        # effective receptive fan is 3x3, not 112x3x3. LayerScale supplies the
        # small identity-preserving insertion magnitude.
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_normal_(
            self.pointwise.weight, mode="fan_in", nonlinearity="relu"
        )
        with torch.no_grad():
            self.layer_scale.fill_(SURFACE_BRANCH_LAYER_SCALE_INIT)

    def forward(self, s2: Tensor, *, mean_broadcast: bool = False) -> Tensor:
        if s2.ndim != 4 or int(s2.size(1)) != SWIFTFORMER_XS_S2_CHANNELS:
            raise SwiftFormerSurfaceB16ContractError(
                "s2 must have shape [B,112,H,W]."
            )
        residual = self.pointwise(
            self.activation(self.norm(self.depthwise(s2)))
        )
        residual = residual * self.layer_scale
        if mean_broadcast:
            residual = residual.mean(dim=(2, 3), keepdim=True).expand_as(residual)
        return residual


class SwiftFormerSurfaceB16(nn.Module):
    """SwiftFormer-XS with an isolated, default-off S2 surface residual.

    The supplied backbone owns all stock SwiftFormer parameters. B16 manually
    traverses the stem and stages so its residual can be added after the stock
    stage-3 downsample and before the stage-3 blocks. No teacher is registered;
    a raw teacher feature map enters only the standalone relation-loss function.
    """

    def __init__(self, backbone: nn.Module, mode: str = SURFACE_OFF_B16_MODE) -> None:
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in SUPPORTED_SURFACE_B16_MODES:
            raise SwiftFormerSurfaceB16ContractError(
                f"mode must be one of {SUPPORTED_SURFACE_B16_MODES!r}; "
                f"got {mode!r}."
            )
        self._validate_backbone(backbone)
        self.backbone = backbone
        self.surface_branch = SurfaceResidualBranchB16()
        self.mode = resolved_mode

    @staticmethod
    def _validate_backbone(backbone: nn.Module) -> None:
        required = ("stem", "stages", "norm", "head_drop", "head", "head_dist")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise SwiftFormerSurfaceB16ContractError(
                f"backbone misses required SwiftFormer attributes: {missing!r}."
            )
        stages = backbone.stages
        if not hasattr(stages, "__len__") or len(stages) != 4:
            raise SwiftFormerSurfaceB16ContractError(
                "backbone.stages must contain exactly four SwiftFormer stages."
            )
        stage3 = stages[3]
        if not hasattr(stage3, "downsample") or not hasattr(stage3, "blocks"):
            raise SwiftFormerSurfaceB16ContractError(
                "SwiftFormer stage 3 must expose downsample and blocks."
            )
        projection = getattr(stage3.downsample, "proj", None)
        if not isinstance(projection, nn.Conv2d):
            raise SwiftFormerSurfaceB16ContractError(
                "SwiftFormer stage-3 downsample must expose a Conv2d proj."
            )
        if (
            projection.in_channels != SWIFTFORMER_XS_S2_CHANNELS
            or projection.out_channels != SWIFTFORMER_XS_S3_CHANNELS
            or projection.stride != (2, 2)
        ):
            raise SwiftFormerSurfaceB16ContractError(
                "backbone must use the SwiftFormer-XS 112->220 stride-2 stage 3."
            )
        if getattr(backbone, "global_pool", None) != "avg":
            raise SwiftFormerSurfaceB16ContractError(
                "B16 requires the stock SwiftFormer average-pooling head."
            )
        head = backbone.head
        head_dist = backbone.head_dist
        if not isinstance(head, nn.Linear) or not isinstance(head_dist, nn.Linear):
            raise SwiftFormerSurfaceB16ContractError(
                "B16 requires both stock SwiftFormer linear heads."
            )
        if (
            head.in_features != SWIFTFORMER_XS_S3_CHANNELS
            or head_dist.in_features != SWIFTFORMER_XS_S3_CHANNELS
            or head.out_features != SWIFTFORMER_XS_NUM_CLASSES
            or head_dist.out_features != SWIFTFORMER_XS_NUM_CLASSES
        ):
            raise SwiftFormerSurfaceB16ContractError(
                "B16 requires SwiftFormer-XS heads with geometry 220->5."
            )

    def forward_features(
        self, x: Tensor, *, return_s2: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)
        x = self.backbone.stages[1](x)
        s2 = self.backbone.stages[2](x)

        stage3 = self.backbone.stages[3]
        x = stage3.downsample(s2)
        if self.mode != SURFACE_OFF_B16_MODE:
            residual = self.surface_branch(
                s2,
                mean_broadcast=self.mode == SURFACE_MEAN_CONTROL_B16_MODE,
            )
            if residual.shape != x.shape:
                raise SwiftFormerSurfaceB16ContractError(
                    "surface residual must match the stage-3 downsample shape."
                )
            x = x + residual
        x = stage3.blocks(x)
        x = self.backbone.norm(x)
        if return_s2:
            return x, s2
        return x

    def forward_head(self, features: Tensor) -> Tensor:
        pooled = features.mean(dim=(2, 3))
        pooled = self.backbone.head_drop(pooled)
        main_logits = self.backbone.head(pooled)
        distilled_logits = self.backbone.head_dist(pooled)
        return (main_logits + distilled_logits) / 2

    def forward(
        self, x: Tensor, *, return_s2: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if return_s2:
            features, s2 = self.forward_features(x, return_s2=True)
            return self.forward_head(features), s2
        features = self.forward_features(x)
        return self.forward_head(features)


def cosine_neighbor_relation_field(
    features: Tensor, *, eps: float = RELATION_EPS
) -> Tensor:
    """Return every horizontal and vertical channel-cosine neighbour edge.

    Horizontal ``[B,H,W-1]`` edges precede vertical ``[B,H-1,W]`` edges in a
    flat ``[B, H*(W-1) + (H-1)*W]`` vector. Channel count is deliberately
    absent, allowing direct comparison of DINO and SwiftFormer feature maps
    without dropping border-adjacent relations.
    """

    if features.ndim != 4:
        raise SwiftFormerSurfaceB16ContractError(
            "relation features must have shape [B,C,H,W]."
        )
    if int(features.size(1)) < 1 or int(features.size(2)) < 2 or int(
        features.size(3)
    ) < 2:
        raise SwiftFormerSurfaceB16ContractError(
            "relation features require C>=1 and H,W>=2."
        )
    if not features.is_floating_point():
        raise SwiftFormerSurfaceB16ContractError(
            "relation features must use a floating-point dtype."
        )
    epsilon = float(eps)
    if not 0.0 < epsilon < 1.0:
        raise SwiftFormerSurfaceB16ContractError("eps must be in (0,1).")
    work = features.float()
    if not bool(torch.isfinite(work).all()):
        raise SwiftFormerSurfaceB16ContractError(
            "relation features must be finite."
        )
    normalized = F.normalize(work, p=2.0, dim=1, eps=epsilon)
    right = (normalized[:, :, :, :-1] * normalized[:, :, :, 1:]).sum(dim=1)
    down = (normalized[:, :, :-1, :] * normalized[:, :, 1:, :]).sum(dim=1)
    return torch.cat((right.flatten(1), down.flatten(1)), dim=1)


def surface_relation_distillation_loss(
    student_s2: Tensor,
    raw_teacher_map: Tensor,
    *,
    eps: float = RELATION_EPS,
    beta: float = RELATION_SMOOTH_L1_BETA,
) -> Tensor:
    """Match local right/down relations without matching teacher channels.

    ``raw_teacher_map`` is an ordinary tensor, never a registered submodule.
    In the locked research path it is ``[B,384,16,16]`` and the student S2 map
    is ``[B,112,14,14]``. The teacher is detached, converted to FP32, and
    resized to the student's grid using bilinear interpolation with
    ``align_corners=False``. The loss is an FP32 SmoothL1 difference between
    the two relation fields; the relation geometry itself remains
    channel-agnostic and spatial-size agnostic.
    """

    if student_s2.ndim != 4 or raw_teacher_map.ndim != 4:
        raise SwiftFormerSurfaceB16ContractError(
            "student and teacher maps must both have shape [B,C,H,W]."
        )
    if int(student_s2.size(0)) != int(raw_teacher_map.size(0)):
        raise SwiftFormerSurfaceB16ContractError(
            "student and teacher batch sizes must match."
        )
    if student_s2.device != raw_teacher_map.device:
        raise SwiftFormerSurfaceB16ContractError(
            "student and teacher maps must be on the same device."
        )
    if not student_s2.is_floating_point() or not raw_teacher_map.is_floating_point():
        raise SwiftFormerSurfaceB16ContractError(
            "student and teacher maps must use floating-point dtypes."
        )
    resolved_beta = float(beta)
    if not math.isfinite(resolved_beta) or resolved_beta <= 0.0:
        raise SwiftFormerSurfaceB16ContractError(
            "SmoothL1 beta must be finite and greater than zero."
        )
    teacher = F.interpolate(
        raw_teacher_map.detach().float(),
        size=tuple(int(value) for value in student_s2.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    student_relation = cosine_neighbor_relation_field(student_s2, eps=eps)
    teacher_relation = cosine_neighbor_relation_field(teacher, eps=eps)
    return F.smooth_l1_loss(
        student_relation,
        teacher_relation,
        beta=resolved_beta,
        reduction="mean",
    )
