from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trkh.models.swiftformer_surface_b16 import cosine_neighbor_relation_field


SURFACEFOLD_SPATIAL_B17_MODE = "surfacefold_spatial_b17"
SURFACEFOLD_MEAN_CONTROL_B17_MODE = "surfacefold_mean_control_b17"
SURFACEFOLD_OFF_B17_MODE = "surfacefold_off_b17"
SUPPORTED_SURFACEFOLD_B17_MODES = (
    SURFACEFOLD_SPATIAL_B17_MODE,
    SURFACEFOLD_MEAN_CONTROL_B17_MODE,
    SURFACEFOLD_OFF_B17_MODE,
)
_SURFACEFOLD_MODE_IDS = {
    SURFACEFOLD_OFF_B17_MODE: 0,
    SURFACEFOLD_MEAN_CONTROL_B17_MODE: 1,
    SURFACEFOLD_SPATIAL_B17_MODE: 2,
}

SWIFTFORMER_XS_S2_CHANNELS = 112
SWIFTFORMER_XS_S3_CHANNELS = 220
SWIFTFORMER_XS_NUM_CLASSES = 5
SWIFTFORMER_XS_INPUT_SIZE = 224
SWIFTFORMER_XS_S3_SIZE = 7
DINO_SMALL_CHANNELS = 384
DINO_SMALL_PATCH_GRID = 16
SURFACEFOLD_FACTOR_SEED = 17_042
SURFACEFOLD_INITIAL_DELTA_RATIO = 0.01
SURFACEFOLD_D_INITIAL_VALUE = 1.0 / 9.0
SURFACEFOLD_RELATION_EDGES = 84
SURFACEFOLD_RELATION_WEIGHT = 0.1
SURFACEFOLD_RELATION_BETA = 0.1


class SwiftFormerSurfaceFoldB17ContractError(ValueError):
    """Raised when a tensor or backbone violates the locked B17 contract."""


def _validate_mode(mode: str) -> str:
    resolved = str(mode).strip().lower()
    if resolved not in SUPPORTED_SURFACEFOLD_B17_MODES:
        raise SwiftFormerSurfaceFoldB17ContractError(
            f"mode must be one of {SUPPORTED_SURFACEFOLD_B17_MODES!r}; "
            f"got {mode!r}."
        )
    return resolved


def _factor_initial_state(projection: nn.Conv2d, seed: int) -> Tuple[Tensor, Tensor]:
    """Create active, matched factors without consuming the global RNG.

    D begins as an exact 3x3 box average. P is non-zero and scaled per output
    channel so every effective delta row has exactly one percent of the
    corresponding frozen pretrained row's Frobenius norm. Consequently the
    spatial candidate and mean-D control start identically, while both factors
    can receive gradients on the first backward pass.
    """

    weight = projection.weight.detach()
    if not weight.is_floating_point() or not bool(torch.isfinite(weight).all()):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "stage-3 projection weights must be finite floating-point values."
        )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    raw = torch.randn(
        (SWIFTFORMER_XS_S3_CHANNELS, SWIFTFORMER_XS_S2_CHANNELS),
        generator=cpu_generator,
        dtype=torch.float64,
        device="cpu",
    ).to(device=weight.device)
    raw_norm = raw.norm(dim=1, keepdim=True)
    base_norm = weight.double().flatten(1).norm(dim=1, keepdim=True)
    if not bool((raw_norm > 0).all()) or not bool((base_norm > 0).all()):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "stage-3 projection rows and deterministic P rows must be non-zero."
        )

    # ||D_i||_F == 1/3 when all nine coefficients are 1/9. Therefore
    # ||P_o * D||_F == ||P_o||_2 / 3.
    target_p_norm = (
        3.0 * SURFACEFOLD_INITIAL_DELTA_RATIO * base_norm
    )
    factor_p = (raw / raw_norm * target_p_norm).to(dtype=weight.dtype)
    factor_d = torch.full(
        (SWIFTFORMER_XS_S2_CHANNELS, 3, 3),
        SURFACEFOLD_D_INITIAL_VALUE,
        dtype=weight.dtype,
        device=weight.device,
    )
    return factor_p, factor_d


class SwiftFormerSurfaceFoldB17(nn.Module):
    """SwiftFormer-XS with a foldable factorized stage-3 weight delta.

    The pretrained stage-3 3x3 projection and its bias are immutable. During
    training the spatial arm adds ``P[o,i] * D[i,h,w]`` to that kernel. The
    equal-state control replaces each D kernel by its scalar spatial mean. The
    off arm owns no factors. Candidate/control factors can be folded into the
    existing projection for a stock-topology deployment model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mode: str = SURFACEFOLD_OFF_B17_MODE,
        *,
        factor_seed: int = SURFACEFOLD_FACTOR_SEED,
    ) -> None:
        super().__init__()
        self.mode = _validate_mode(mode)
        self.register_buffer(
            "_surfacefold_mode_id",
            torch.tensor(_SURFACEFOLD_MODE_IDS[self.mode], dtype=torch.int64),
            persistent=True,
        )
        self._validate_backbone(backbone)
        self.backbone = backbone

        projection = self._projection()
        projection.weight.requires_grad_(False)
        if projection.bias is None:
            raise SwiftFormerSurfaceFoldB17ContractError(
                "the locked SwiftFormer-XS projection must have a bias."
            )
        projection.bias.requires_grad_(False)

        if self.mode == SURFACEFOLD_OFF_B17_MODE:
            self.register_parameter("factor_p", None)
            self.register_parameter("factor_d", None)
        else:
            factor_p, factor_d = _factor_initial_state(projection, factor_seed)
            self.factor_p = nn.Parameter(factor_p)
            self.factor_d = nn.Parameter(factor_d)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        mode_key = prefix + "_surfacefold_mode_id"
        incoming = state_dict.get(mode_key)
        expected = self._surfacefold_mode_id
        valid = (
            isinstance(incoming, Tensor)
            and incoming.dtype == torch.int64
            and incoming.shape == expected.shape
            and int(incoming.detach().cpu().item()) == int(expected.item())
        )
        if not valid:
            observed = None
            if isinstance(incoming, Tensor) and incoming.numel() == 1:
                observed = int(incoming.detach().cpu().item())
            error_msgs.append(
                f"{mode_key} violates the immutable B17 mode contract: "
                f"expected={int(expected.item())}, observed={observed}."
            )
            state_dict = state_dict.copy()
            state_dict[mode_key] = expected.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _assert_mode_contract(self) -> None:
        expected = _SURFACEFOLD_MODE_IDS[self.mode]
        if (
            self._surfacefold_mode_id.dtype != torch.int64
            or self._surfacefold_mode_id.shape != torch.Size([])
            or int(self._surfacefold_mode_id.detach().cpu().item()) != expected
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "persistent B17 mode metadata disagrees with runtime semantics."
            )

    @staticmethod
    def _validate_backbone(backbone: nn.Module) -> None:
        required = (
            "stem",
            "stages",
            "norm",
            "head_drop",
            "head",
            "head_dist",
            "forward_head",
        )
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise SwiftFormerSurfaceFoldB17ContractError(
                f"backbone misses required SwiftFormer attributes: {missing!r}."
            )
        stages = backbone.stages
        if not hasattr(stages, "__len__") or len(stages) != 4:
            raise SwiftFormerSurfaceFoldB17ContractError(
                "backbone.stages must contain exactly four SwiftFormer stages."
            )
        if any(bool(getattr(stage, "grad_checkpointing", False)) for stage in stages):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "B17 manual traversal requires gradient checkpointing to be disabled."
            )
        stage3 = stages[3]
        downsample = getattr(stage3, "downsample", None)
        projection = getattr(downsample, "proj", None)
        downsample_norm = getattr(downsample, "norm", None)
        if not isinstance(projection, nn.Conv2d) or not isinstance(
            downsample_norm, nn.BatchNorm2d
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "stage 3 must expose the stock Conv2d projection and BatchNorm2d."
            )
        expected_geometry = (
            projection.in_channels == SWIFTFORMER_XS_S2_CHANNELS
            and projection.out_channels == SWIFTFORMER_XS_S3_CHANNELS
            and projection.kernel_size == (3, 3)
            and projection.stride == (2, 2)
            and projection.padding == (1, 1)
            and projection.dilation == (1, 1)
            and projection.groups == 1
            and projection.bias is not None
            and downsample_norm.num_features == SWIFTFORMER_XS_S3_CHANNELS
        )
        if not expected_geometry:
            raise SwiftFormerSurfaceFoldB17ContractError(
                "backbone must use the stock SwiftFormer-XS 112->220 k3/s2/p1 "
                "stage-3 downsample with bias and BatchNorm."
            )
        if getattr(backbone, "global_pool", None) != "avg":
            raise SwiftFormerSurfaceFoldB17ContractError(
                "B17 requires the stock SwiftFormer average-pooling head."
            )
        if not isinstance(backbone.head, nn.Linear) or not isinstance(
            backbone.head_dist, nn.Linear
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "B17 requires both stock SwiftFormer linear heads."
            )
        for head in (backbone.head, backbone.head_dist):
            if (
                head.in_features != SWIFTFORMER_XS_S3_CHANNELS
                or head.out_features != SWIFTFORMER_XS_NUM_CLASSES
            ):
                raise SwiftFormerSurfaceFoldB17ContractError(
                    "B17 requires two SwiftFormer-XS 220->5 heads."
                )

    def _projection(self) -> nn.Conv2d:
        return self.backbone.stages[3].downsample.proj

    def effective_delta_weight(self) -> Tensor:
        self._assert_mode_contract()
        projection = self._projection()
        if self.mode == SURFACEFOLD_OFF_B17_MODE:
            return torch.zeros_like(projection.weight)
        if self.factor_p is None or self.factor_d is None:
            raise SwiftFormerSurfaceFoldB17ContractError(
                "an active B17 mode must own both factors."
            )
        factor_d = self.factor_d
        if self.mode == SURFACEFOLD_MEAN_CONTROL_B17_MODE:
            factor_d = factor_d.mean(dim=(1, 2), keepdim=True).expand_as(factor_d)
        delta = self.factor_p[:, :, None, None] * factor_d[None, :, :, :]
        if delta.shape != projection.weight.shape:
            raise SwiftFormerSurfaceFoldB17ContractError(
                "factorized delta does not match the stock projection."
            )
        return delta

    def effective_projection_weight(self) -> Tensor:
        return self._projection().weight + self.effective_delta_weight()

    def forward_stage3_downsample(self, s2: Tensor) -> Tensor:
        if (
            s2.ndim != 4
            or int(s2.size(1)) != SWIFTFORMER_XS_S2_CHANNELS
            or tuple(int(value) for value in s2.shape[-2:]) != (14, 14)
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "S2 must have shape [B,112,14,14]."
            )
        stage3 = self.backbone.stages[3]
        projection = stage3.downsample.proj
        x = F.conv2d(
            s2,
            self.effective_projection_weight(),
            projection.bias,
            stride=projection.stride,
            padding=projection.padding,
            dilation=projection.dilation,
            groups=projection.groups,
        )
        s3 = stage3.downsample.norm(x)
        if tuple(int(value) for value in s3.shape[1:]) != (
            SWIFTFORMER_XS_S3_CHANNELS,
            SWIFTFORMER_XS_S3_SIZE,
            SWIFTFORMER_XS_S3_SIZE,
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "post-BN S3 must have shape [B,220,7,7]."
            )
        return s3

    def forward_features(
        self, x: Tensor, *, return_s3: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if x.ndim != 4 or tuple(int(value) for value in x.shape[-2:]) != (
            SWIFTFORMER_XS_INPUT_SIZE,
            SWIFTFORMER_XS_INPUT_SIZE,
        ):
            raise SwiftFormerSurfaceFoldB17ContractError(
                "B17 requires image tensors with spatial size 224x224."
            )
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)
        x = self.backbone.stages[1](x)
        s2 = self.backbone.stages[2](x)
        s3 = self.forward_stage3_downsample(s2)
        x = self.backbone.stages[3].blocks(s3)
        x = self.backbone.norm(x)
        if return_s3:
            return x, s3
        return x

    def forward(
        self, x: Tensor, *, return_s3: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if return_s3:
            features, s3 = self.forward_features(x, return_s3=True)
            return self.backbone.forward_head(features), s3
        return self.backbone.forward_head(self.forward_features(x))

    def fold_to_deploy(self) -> nn.Module:
        """Return a stock-topology backbone with the trained delta folded in."""

        deployed = copy.deepcopy(self.backbone)
        folded_weight = self.effective_projection_weight().detach()
        deployed_projection = deployed.stages[3].downsample.proj
        with torch.no_grad():
            deployed_projection.weight.copy_(folded_weight)
        return deployed


def dino_s3_relation_target_b17(raw_teacher_map: Tensor) -> Tensor:
    """Build one detached FP32 84-edge DINO target shared by all arms."""

    if (
        raw_teacher_map.ndim != 4
        or int(raw_teacher_map.size(1)) != DINO_SMALL_CHANNELS
        or tuple(int(value) for value in raw_teacher_map.shape[-2:])
        != (DINO_SMALL_PATCH_GRID, DINO_SMALL_PATCH_GRID)
    ):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "raw DINO-S map must have shape [B,384,16,16]."
        )
    if not raw_teacher_map.is_floating_point():
        raise SwiftFormerSurfaceFoldB17ContractError(
            "raw DINO-S map must use a floating-point dtype."
        )
    teacher = F.interpolate(
        raw_teacher_map.detach().float(),
        size=(SWIFTFORMER_XS_S3_SIZE, SWIFTFORMER_XS_S3_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    target = cosine_neighbor_relation_field(teacher).detach()
    if target.shape != (int(raw_teacher_map.size(0)), SURFACEFOLD_RELATION_EDGES):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "the DINO target must contain all 84 right/down S3 edges."
        )
    return target


def surfacefold_s3_relation_loss_b17(student_s3: Tensor, target: Tensor) -> Tensor:
    """Match post-BN/pre-block S3 relations to a shared DINO target."""

    if (
        student_s3.ndim != 4
        or tuple(int(value) for value in student_s3.shape[1:])
        != (
            SWIFTFORMER_XS_S3_CHANNELS,
            SWIFTFORMER_XS_S3_SIZE,
            SWIFTFORMER_XS_S3_SIZE,
        )
    ):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "student S3 must have shape [B,220,7,7]."
        )
    if (
        target.ndim != 2
        or tuple(int(value) for value in target.shape)
        != (int(student_s3.size(0)), SURFACEFOLD_RELATION_EDGES)
        or target.requires_grad
        or target.dtype != torch.float32
        or target.device != student_s3.device
        or not bool(torch.isfinite(target).all())
    ):
        raise SwiftFormerSurfaceFoldB17ContractError(
            "shared target must be detached finite FP32 [B,84] on the S3 device."
        )
    relation = cosine_neighbor_relation_field(student_s3)
    return F.smooth_l1_loss(
        relation,
        target.float(),
        beta=SURFACEFOLD_RELATION_BETA,
        reduction="mean",
    )


def surfacefold_relation_distillation_loss_b17(
    student_s3: Tensor, raw_teacher_map: Tensor
) -> Tensor:
    """Convenience composition for a single arm; runners should share target."""

    if student_s3.device != raw_teacher_map.device:
        raise SwiftFormerSurfaceFoldB17ContractError(
            "student and raw teacher maps must be on the same device."
        )
    return surfacefold_s3_relation_loss_b17(
        student_s3, dino_s3_relation_target_b17(raw_teacher_map)
    )


def factorized_delta_parameter_count_b17() -> int:
    return (
        SWIFTFORMER_XS_S3_CHANNELS * SWIFTFORMER_XS_S2_CHANNELS
        + SWIFTFORMER_XS_S2_CHANNELS * 3 * 3
    )


if factorized_delta_parameter_count_b17() != 25_648:
    raise RuntimeError("B17 factorized parameter count changed")
