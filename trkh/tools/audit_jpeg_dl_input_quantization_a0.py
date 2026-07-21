from __future__ import annotations

"""Prospectively locked train-only JPEG-DL input-quantization A0."""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import subprocess
import time
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.config import to_serializable
from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_model
from trkh.tools.audit_attention_maxsep_prototype_a0 import (
    _dataset_stat_snapshot,
    _gpu_snapshot,
    _state_mapping_sha256,
)
from trkh.tools.audit_boxinst_foreground_mask_a0 import (
    locked_cohort,
    locked_training_rows,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _classification_metrics,
    directional_event_masks,
)
from trkh.tools.audit_hamburger_nmf_surface_a0 import (
    _build_dataset,
    _global_rng_equal,
    _global_rng_snapshot,
    _make_condition_loader,
    _model_state_sha256,
    _prepare_output_dir,
    _protected_untracked_state,
    _repo_state,
    _sha256,
    _verify_hash,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)


METHOD = "jpeg_dl_input_quantization_a0"
MODE = "jpeg_dl_input_quantization_a0_train_information_gate"
SEED = 42
FOCUS_CLASS = 1
NUM_CLASSES = 5
FOLDS = (0, 1, 2, 3, 4)
RESTRICTED_CLASSES = (0, 2, 4)
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 2
JPEG_LEARNING_RATE = 0.005
JPEG_ALPHA = 5.0
FP_MARGIN = 0.15
TP_MARGIN = 0.10
FP_WEIGHT = 0.35
TP_WEIGHT = 0.20
MIN_Q = 1e-5
MAX_Q = 255.0
BLOCK_SIZE = 8
PHASE_SHIFT = (4, 4)
ROLE_CHUNK_SIZE = 3
TRAIN_ROLE_MICROBATCH_SIZE = 1
MAX_XAI_BATCH_SHAPE_PROBABILITY_ERROR = 3e-3
ENGINEERING_CIDT_PROBABILITY_TOLERANCE = 1e-4
KEEPER_COMPATIBILITY_BATCH_SIZE = 64
KEEPER_COMPATIBILITY_PROBABILITY_TOLERANCE = 1e-6
FORMAL_BATCH32_CIDT_TELEMETRY_TOLERANCE = 0.015
MICROBATCH_GRADIENT_MAX_ABS_ERROR = 2e-3
MICROBATCH_GRADIENT_MIN_COSINE = 0.9999

KEEPER_ROLE = "keeper_raw"
FIXED_ROLE = "fixed_q1"
PAPER_ROLE = "paper_ce_full"
SCALAR_ROLE = "precision_scalar"
CANDIDATE_ROLE = "precision_full"
PERMUTED_ROLE = "precision_full_frequency_permuted"
PHASE_ROLE = "precision_full_phase_shifted"
TRAINED_ROLES = (PAPER_ROLE, SCALAR_ROLE, CANDIDATE_ROLE)
EVALUATED_ROLES = (
    KEEPER_ROLE,
    FIXED_ROLE,
    PAPER_ROLE,
    SCALAR_ROLE,
    CANDIDATE_ROLE,
    PERMUTED_ROLE,
    PHASE_ROLE,
)
CONDITIONS = (
    ("dim", 0.70, 0.90),
    ("bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)

EXPECTED_TRAIN_ROWS = 9215
EXPECTED_TRAIN_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_HOLDOUT_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_TRAIN_INDEX_SHA256 = (
    "d41d14b04dfb70b60b5058f30d1c4cfe0189aef3b0efc0a22741f5672c1473c8"
)
EXPECTED_COHORT_ROWS = 763
EXPECTED_KEEPER_TP = 528
EXPECTED_KEEPER_FN = 13
EXPECTED_RESTRICTED_FP = 222
EXPECTED_COHORT_INDEX_SHA256 = (
    "59d2146617f2642f99c081f49a9aa5812eae4206b27887a1fa5476f6f61c85b2"
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "160dc501daf6375ce6f94c4a02fcf47e02639111cb74220a953eca1fd4763642"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "09ed1ad8c87f22f28ad2ef3efd259b4a867ebebdc6e93ea7ce0ea791ccc07b5c"
LOCKED_OFFICIAL_COMMIT = "894fdb651326b7c68f13967886fea2ee04831b3a"
LOCKED_OFFICIAL_TREE = "c33f70fcb321ea9bc7431f4b6f9b998b5e107e84"
LOCKED_OFFICIAL_LICENSE_SHA256 = "7838cf441bbc8e87a1c903ad9ccc03476868334c306ee70a3f31f0b31fc3552c"
LOCKED_OFFICIAL_JPEG_SHA256 = "53c65c0fa52794c7e914f02cecca2674641f204a927522aceba3f86496fa954b"
LOCKED_OFFICIAL_TRAIN_SHA256 = "dcad3877e9c335b4f2a558d177d979dcc7a13af9cc91d7e0cd71c5001d98cb4a"
LOCKED_OFFICIAL_TRANSFORMER_SHA256 = "60fe58066495010d3c3f6e0f60f05c5405ae7fb5ae4fe0955b3fb9b1d8b399c1"
LOCKED_OFFICIAL_FINEGRAINED_SHA256 = "3b9f8d6593a45aeba2d07eab4917f8225722cae1810c4d80463a6f183e498659"
LOCKED_OFFICIAL_README_SHA256 = "f24ab0a9245c07959e20bdf3c1a4788357ea475dfefbc95b770243151f1f1173"

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\JPEG-Inspired-DL")
PAPER_PATH = Path(
    r"D:\DataAI\external_sources\papers\Salamah_JPEG_Inspired_DL_ICLR_2025.pdf"
)
IMPLEMENTATION_PATHS = (
    "trkh/tools/audit_jpeg_dl_input_quantization_a0.py",
    "tests/test_audit_jpeg_dl_input_quantization_a0.py",
    "scripts/run_trkh_jpeg_dl_input_quantization_a0.ps1",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only JPEG-DL input-quantization A0 gate."
    )
    parser.add_argument("--checkpoint", type=Path, default=KEEPER_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--launcher-args", type=Path, default=KEEPER_ROOT / "launcher_args.json")
    parser.add_argument("--resolved-config", type=Path, default=KEEPER_ROOT / "resolved_config.json")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714" / "summary.json",
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_JPEG_DL_INPUT_QUANTIZATION_A0_PROTOCOL_20260721.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_jpeg_dl_input_quantization_a0_20260721",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--engineering-forward", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("JPEG-DL A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError("Batch size or workers differ from prospective lock")
    if int(args.seed) != SEED:
        raise ValueError("Seed differs from prospective lock")


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def implementation_tracking_state() -> Dict[str, object]:
    tracked = {
        line.strip().replace("\\", "/")
        for line in _git_value(REPO_ROOT, "ls-files").splitlines()
        if line.strip()
    }
    status = _git_value(
        REPO_ROOT, "status", "--porcelain=v1", "--untracked-files=all"
    )
    dirty = []
    for line in status.splitlines():
        path = line[3:].strip().strip('"').replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty.append(path)
    return {
        "paths": list(IMPLEMENTATION_PATHS),
        "all_tracked": all(path in tracked for path in IMPLEMENTATION_PATHS),
        "dirty_implementation_paths": sorted(
            path for path in dirty if path in IMPLEMENTATION_PATHS
        ),
        "head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "passed": all(path in tracked for path in IMPLEMENTATION_PATHS)
        and not any(path in IMPLEMENTATION_PATHS for path in dirty),
    }


def verify_locked_inputs(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    files = {
        "keeper": _verify_hash(args.checkpoint, LOCKED_KEEPER_SHA256, "keeper"),
        "launcher_args": _verify_hash(
            args.launcher_args, LOCKED_LAUNCHER_ARGS_SHA256, "keeper launcher args"
        ),
        "resolved_config": _verify_hash(
            args.resolved_config, LOCKED_RESOLVED_CONFIG_SHA256, "keeper resolved config"
        ),
        "data_yaml": _verify_hash(args.data, LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_hash(
            args.cidt_summary, LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            args.cidt_predictions, LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"
        ),
        "protocol": _verify_hash(args.protocol, LOCKED_PROTOCOL_SHA256, "protocol"),
        "current_commands": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
            LOCKED_CURRENT_COMMAND_SHA256,
            "current commands",
        ),
        "command_history": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
            LOCKED_COMMAND_HISTORY_SHA256,
            "command history",
        ),
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "JPEG-DL paper"),
        "official_license": _verify_hash(
            OFFICIAL_ROOT / "LICENSE",
            LOCKED_OFFICIAL_LICENSE_SHA256,
            "JPEG-DL license",
        ),
        "official_jpeg_layer": _verify_hash(
            OFFICIAL_ROOT / "ImageNet" / "JPEG_layer.py",
            LOCKED_OFFICIAL_JPEG_SHA256,
            "JPEG-DL layer source",
        ),
        "official_train": _verify_hash(
            OFFICIAL_ROOT / "ImageNet" / "train.py",
            LOCKED_OFFICIAL_TRAIN_SHA256,
            "JPEG-DL ImageNet training source",
        ),
        "official_transformer": _verify_hash(
            OFFICIAL_ROOT / "TransformerBased" / "main_JPEG.py",
            LOCKED_OFFICIAL_TRANSFORMER_SHA256,
            "JPEG-DL transformer source",
        ),
        "official_finegrained": _verify_hash(
            OFFICIAL_ROOT / "train_teacher_JPEG.py",
            LOCKED_OFFICIAL_FINEGRAINED_SHA256,
            "JPEG-DL fine-grained source",
        ),
        "official_readme": _verify_hash(
            OFFICIAL_ROOT / "README.md",
            LOCKED_OFFICIAL_README_SHA256,
            "JPEG-DL README",
        ),
    }
    commit = _git_value(OFFICIAL_ROOT, "rev-parse", "HEAD")
    tree = _git_value(OFFICIAL_ROOT, "rev-parse", "HEAD^{tree}")
    status = _git_value(OFFICIAL_ROOT, "status", "--porcelain=v1")
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE or status:
        raise ValueError(
            "Pinned JPEG-DL repository differs: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "official_repository": {
            "path": str(OFFICIAL_ROOT.resolve()),
            "remote": _git_value(OFFICIAL_ROOT, "remote", "get-url", "origin"),
            "commit": commit,
            "tree": tree,
            "tracked_worktree_clean": status == "",
            "source_imported": False,
        },
        "protected_untracked": protected,
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def _dct_matrix_numpy(size: int = BLOCK_SIZE) -> np.ndarray:
    matrix = np.empty((size, size), dtype=np.float64)
    for frequency in range(size):
        scale = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
        for spatial in range(size):
            matrix[frequency, spatial] = scale * math.cos(
                math.pi * (2 * spatial + 1) * frequency / (2 * size)
            )
    return matrix


def _frequency_permutation() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    permutation = np.arange(BLOCK_SIZE * BLOCK_SIZE, dtype=np.int64)
    permutation[1:] = generator.permutation(permutation[1:])
    return permutation


class JpegDlInputLayer(nn.Module):
    """Independent 4:4:4 JPEG-DL layer with a five-point masked CPMF."""

    def __init__(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        scalar_tables: bool = False,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_tables = bool(scalar_tables)
        shape = (2, 1, 1) if self.scalar_tables else (2, BLOCK_SIZE, BLOCK_SIZE)
        self.q = nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=trainable)
        self.register_buffer(
            "mean", torch.tensor(tuple(mean), dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(tuple(std), dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "dct_matrix", torch.from_numpy(_dct_matrix_numpy())
        )
        self.register_buffer(
            "candidate_offsets", torch.arange(5, dtype=torch.float32)
        )
        self.register_buffer(
            "frequency_permutation",
            torch.from_numpy(_frequency_permutation()).to(torch.long),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def project_(self) -> None:
        with torch.no_grad():
            self.q.clamp_(MIN_Q, MAX_Q)

    def q_tables(self, *, frequency_permuted: bool = False) -> Tensor:
        q = self.q.clamp(MIN_Q, MAX_Q)
        if self.scalar_tables:
            q = q.expand(2, BLOCK_SIZE, BLOCK_SIZE)
        if frequency_permuted:
            flat = q.reshape(2, -1)
            q = flat[:, self.frequency_permutation].reshape_as(q)
        return q

    @staticmethod
    def rgb_to_ycbcr(rgb: Tensor) -> Tensor:
        red, green, blue = rgb.unbind(dim=1)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = (blue - luminance) / (2.0 * (1.0 - 0.114)) + 0.5
        cr = (red - luminance) / (2.0 * (1.0 - 0.299)) + 0.5
        return torch.stack((luminance, cb, cr), dim=1)

    @staticmethod
    def ycbcr_to_rgb(ycbcr: Tensor) -> Tensor:
        luminance, cb, cr = ycbcr.unbind(dim=1)
        cb = cb - 0.5
        cr = cr - 0.5
        red = luminance + 2.0 * (1.0 - 0.299) * cr
        green = (
            luminance
            - 2.0 * (1.0 - 0.299) * 0.299 / 0.587 * cr
            - 2.0 * (1.0 - 0.114) * 0.114 / 0.587 * cb
        )
        blue = luminance + 2.0 * (1.0 - 0.114) * cb
        return torch.stack((red, green, blue), dim=1)

    @staticmethod
    def blockify(value: Tensor) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        if value.ndim != 4:
            raise ValueError("JPEG-DL input must be BCHW")
        batch, channels, height, width = value.shape
        pad_h = (-height) % BLOCK_SIZE
        pad_w = (-width) % BLOCK_SIZE
        padded = F.pad(value, (0, pad_w, 0, pad_h))
        padded_h, padded_w = int(padded.shape[-2]), int(padded.shape[-1])
        blocks = padded.reshape(
            batch,
            channels,
            padded_h // BLOCK_SIZE,
            BLOCK_SIZE,
            padded_w // BLOCK_SIZE,
            BLOCK_SIZE,
        ).permute(0, 1, 2, 4, 3, 5)
        return blocks.contiguous(), (height, width, pad_h, pad_w)

    @staticmethod
    def deblockify(blocks: Tensor, geometry: Tuple[int, int, int, int]) -> Tensor:
        height, width, pad_h, pad_w = geometry
        batch, channels, rows, columns, _, _ = blocks.shape
        value = blocks.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(
            batch,
            channels,
            rows * BLOCK_SIZE,
            columns * BLOCK_SIZE,
        )
        if pad_h or pad_w:
            value = value[:, :, :height, :width]
        return value

    def dct(self, blocks: Tensor) -> Tensor:
        matrix = self.dct_matrix.to(device=blocks.device, dtype=blocks.dtype)
        return torch.matmul(torch.matmul(matrix, blocks), matrix.t())

    def idct(self, coefficients: Tensor) -> Tensor:
        matrix = self.dct_matrix.to(device=coefficients.device, dtype=coefficients.dtype)
        return torch.matmul(torch.matmul(matrix.t(), coefficients), matrix)

    def soft_quantize(self, coefficients: Tensor, q_tables: Tensor) -> Tensor:
        channel_q = torch.stack((q_tables[0], q_tables[1], q_tables[1]), dim=0)
        q = channel_q.view(1, 3, 1, 1, BLOCK_SIZE, BLOCK_SIZE, 1)
        expanded = coefficients.unsqueeze(-1)
        center = torch.round(expanded / q).detach()
        base = (center - 2.0).clamp(-127.0, 123.0)
        indices = base + self.candidate_offsets.to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        reconstruction = indices * q
        weights = torch.softmax(
            -JPEG_ALPHA * torch.square(expanded - reconstruction), dim=-1
        )
        return torch.sum(weights * reconstruction, dim=-1)

    def _jpeg_rgb255(
        self,
        rgb255: Tensor,
        *,
        frequency_permuted: bool = False,
        phase_shifted: bool = False,
        quantize: bool = True,
    ) -> Tensor:
        if not torch.is_floating_point(rgb255):
            rgb255 = rgb255.float()
        if phase_shifted:
            rgb255 = torch.roll(rgb255, shifts=PHASE_SHIFT, dims=(-2, -1))
        centered = rgb255 - 128.0
        ycbcr = self.rgb_to_ycbcr(centered)
        blocks, geometry = self.blockify(ycbcr)
        coefficients = self.dct(blocks)
        if quantize:
            coefficients = self.soft_quantize(
                coefficients,
                self.q_tables(frequency_permuted=frequency_permuted).to(
                    device=coefficients.device, dtype=coefficients.dtype
                ),
            )
        reconstructed = self.deblockify(self.idct(coefficients), geometry)
        output = self.ycbcr_to_rgb(reconstructed) + 128.0
        if phase_shifted:
            output = torch.roll(
                output,
                shifts=(-PHASE_SHIFT[0], -PHASE_SHIFT[1]),
                dims=(-2, -1),
            )
        return output

    def rgb01_from_normalized(self, images: Tensor) -> Tensor:
        if not torch.is_floating_point(images):
            images = images.float()
        return images * self.std + self.mean

    def normalize_rgb01(self, rgb01: Tensor) -> Tensor:
        return (rgb01 - self.mean) / self.std

    def forward(
        self,
        images: Tensor,
        *,
        frequency_permuted: bool = False,
        phase_shifted: bool = False,
        return_rgb: bool = False,
    ):
        rgb255 = self.rgb01_from_normalized(images) * 255.0
        reconstructed_rgb = self._jpeg_rgb255(
            rgb255,
            frequency_permuted=frequency_permuted,
            phase_shifted=phase_shifted,
            quantize=True,
        ) / 255.0
        normalized = self.normalize_rgb01(reconstructed_rgb)
        if return_rgb:
            return normalized, reconstructed_rgb
        return normalized

    def bypass(self, images: Tensor, *, phase_shifted: bool = False) -> Tensor:
        rgb255 = self.rgb01_from_normalized(images) * 255.0
        reconstructed = self._jpeg_rgb255(
            rgb255, phase_shifted=phase_shifted, quantize=False
        ) / 255.0
        return self.normalize_rgb01(reconstructed)


class _JpegExport(nn.Module):
    def __init__(self, layer: JpegDlInputLayer) -> None:
        super().__init__()
        self.layer = copy.deepcopy(layer).cpu().eval()

    def forward(self, images: Tensor) -> Tensor:
        return self.layer(images)


def _numpy_rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = (blue - luminance) / (2.0 * (1.0 - 0.114)) + 0.5
    cr = (red - luminance) / (2.0 * (1.0 - 0.299)) + 0.5
    return np.stack((luminance, cb, cr), axis=1)


def _numpy_ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    luminance, cb, cr = ycbcr[:, 0], ycbcr[:, 1] - 0.5, ycbcr[:, 2] - 0.5
    red = luminance + 2.0 * (1.0 - 0.299) * cr
    green = (
        luminance
        - 2.0 * (1.0 - 0.299) * 0.299 / 0.587 * cr
        - 2.0 * (1.0 - 0.114) * 0.114 / 0.587 * cb
    )
    blue = luminance + 2.0 * (1.0 - 0.114) * cb
    return np.stack((red, green, blue), axis=1)


def _numpy_soft_quantize(z: np.ndarray, q: np.ndarray) -> np.ndarray:
    center = np.rint(z / q)
    base = np.clip(center - 2.0, -127.0, 123.0)
    indices = base[..., None] + np.arange(5, dtype=np.float64)
    reconstruction = indices * q[..., None]
    logits = -JPEG_ALPHA * np.square(z[..., None] - reconstruction)
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.sum(weights * reconstruction, axis=-1)


def _full_space_soft_quantize(z: float, q: float, alpha: float) -> float:
    indices = np.arange(-256, 257, dtype=np.float64)
    reconstruction = indices * float(q)
    logits = -float(alpha) * np.square(float(z) - reconstruction)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return float(np.sum(weights * reconstruction))


def engineering_checks() -> Dict[str, object]:
    before = _global_rng_snapshot()
    generator = np.random.default_rng(731)
    matrix = _dct_matrix_numpy()
    blocks = generator.normal(size=(2, 3, 2, 2, 8, 8)).astype(np.float64)
    layer = JpegDlInputLayer(mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4)).double()
    torch_blocks = torch.from_numpy(blocks)
    torch_dct = layer.dct(torch_blocks).detach().numpy()
    numpy_dct = matrix @ blocks @ matrix.T
    dct_error = float(np.max(np.abs(torch_dct - numpy_dct)))
    torch_roundtrip = layer.idct(torch.from_numpy(torch_dct)).detach().numpy()
    dct_roundtrip_error = float(np.max(np.abs(torch_roundtrip - blocks)))

    rgb = generator.normal(size=(2, 3, 7, 9)).astype(np.float64)
    torch_ycbcr = layer.rgb_to_ycbcr(torch.from_numpy(rgb)).detach().numpy()
    numpy_ycbcr = _numpy_rgb_to_ycbcr(rgb)
    color_forward_error = float(np.max(np.abs(torch_ycbcr - numpy_ycbcr)))
    torch_color_roundtrip = layer.ycbcr_to_rgb(
        torch.from_numpy(torch_ycbcr)
    ).detach().numpy()
    numpy_color_roundtrip = _numpy_ycbcr_to_rgb(numpy_ycbcr)
    color_roundtrip_error = float(
        max(
            np.max(np.abs(torch_color_roundtrip - rgb)),
            np.max(np.abs(numpy_color_roundtrip - rgb)),
        )
    )

    z = generator.normal(size=(2, 3, 1, 1, 8, 8)).astype(np.float64)
    q = generator.uniform(0.6, 2.0, size=(1, 3, 1, 1, 8, 8)).astype(np.float64)
    q[:, 2] = q[:, 1]
    torch_quant = layer.soft_quantize(
        torch.from_numpy(z),
        torch.from_numpy(np.stack((q[0, 0, 0, 0], q[0, 1, 0, 0]))),
    ).detach().numpy()
    numpy_quant = _numpy_soft_quantize(z, q)
    quantizer_error = float(np.max(np.abs(torch_quant - numpy_quant)))

    scalar_q = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    scalar_z = torch.tensor(0.37, dtype=torch.float64)

    def scalar_quantizer(q_value: Tensor) -> Tensor:
        center = torch.round(scalar_z / q_value).detach()
        indices = (center - 2.0).clamp(-127.0, 123.0) + torch.arange(
            5, dtype=torch.float64
        )
        reconstruction = indices * q_value
        weights = torch.softmax(
            -JPEG_ALPHA * torch.square(scalar_z - reconstruction), dim=0
        )
        return torch.sum(weights * reconstruction)

    scalar_quantizer(scalar_q).backward()
    analytic = float(scalar_q.grad)
    epsilon = 1e-5
    with torch.no_grad():
        finite_difference = float(
            (
                scalar_quantizer(torch.tensor(1.2 + epsilon, dtype=torch.float64))
                - scalar_quantizer(torch.tensor(1.2 - epsilon, dtype=torch.float64))
            )
            / (2.0 * epsilon)
        )
    gradient_relative_error = abs(analytic - finite_difference) / max(
        1e-12, abs(analytic), abs(finite_difference)
    )

    images_fp64 = torch.from_numpy(
        generator.uniform(-2.0, 2.0, size=(2, 3, 16, 24))
    ).double()
    fp64_bypass_error = float((layer.bypass(images_fp64) - images_fp64).abs().max())
    fp32_layer = JpegDlInputLayer(
        mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4)
    ).float()
    images_fp32 = images_fp64.float()
    bypass_error = float((fp32_layer.bypass(images_fp32) - images_fp32).abs().max())
    phase_bypass_error = float(
        (fp32_layer.bypass(images_fp32, phase_shifted=True) - images_fp32)
        .abs()
        .max()
    )
    q_tables = layer.q_tables().detach().numpy()
    permuted = layer.q_tables(frequency_permuted=True).detach().numpy()
    permutation_multiset_exact = all(
        np.array_equal(np.sort(q_tables[index].reshape(-1)), np.sort(permuted[index].reshape(-1)))
        for index in range(2)
    )
    masked_reference = float(
        _numpy_soft_quantize(np.asarray(0.5), np.asarray(1.0))
    )
    full_reference = _full_space_soft_quantize(0.5, 1.0, JPEG_ALPHA)
    after = _global_rng_snapshot()
    checks = {
        "dct_numpy_error_lte_1e10": dct_error <= 1e-10,
        "dct_roundtrip_error_lte_1e10": dct_roundtrip_error <= 1e-10,
        "color_numpy_error_lte_1e10": color_forward_error <= 1e-10,
        "color_roundtrip_error_lte_1e10": color_roundtrip_error <= 1e-10,
        "quantizer_numpy_error_lte_1e10": quantizer_error <= 1e-10,
        "gradient_relative_error_lte_5e4": gradient_relative_error <= 5e-4,
        "gradient_finite_nonzero": math.isfinite(analytic) and abs(analytic) > 0.0,
        "bypass_error_lte_1e6": bypass_error <= 1e-6,
        "phase_bypass_error_lte_1e6": phase_bypass_error <= 1e-6,
        "masked_full_reference_error_lte_1e6": abs(masked_reference - full_reference)
        <= 1e-6,
        "frequency_permutation_multiset_exact": permutation_multiset_exact,
        "full_parameter_count_128": layer.q.numel() == 128,
        "global_rng_unchanged": _global_rng_equal(before, after),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": {
            "dct_numpy_max_abs": dct_error,
            "dct_roundtrip_max_abs": dct_roundtrip_error,
            "color_numpy_max_abs": color_forward_error,
            "color_roundtrip_max_abs": color_roundtrip_error,
            "quantizer_numpy_max_abs": quantizer_error,
            "gradient_relative": gradient_relative_error,
            "bypass_max_abs": bypass_error,
            "phase_bypass_max_abs": phase_bypass_error,
            "fp64_bypass_max_abs_telemetry": fp64_bypass_error,
            "masked_full_reference_abs": abs(masked_reference - full_reference),
        },
        "gradient": {
            "autograd": analytic,
            "finite_difference": finite_difference,
        },
        "reference": {
            "masked_z0p5_q1_alpha5": masked_reference,
            "full_z0p5_q1_alpha5": full_reference,
        },
    }


def precision_loss(
    logits: Tensor,
    targets: Tensor,
    keeper_predictions: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if logits.ndim != 2 or int(logits.shape[1]) != NUM_CLASSES:
        raise ValueError("JPEG-DL precision loss requires Bx5 logits")
    targets = targets.long().reshape(-1)
    keeper_predictions = keeper_predictions.long().reshape(-1)
    other = torch.cat((logits[:, :FOCUS_CLASS], logits[:, FOCUS_CLASS + 1 :]), dim=1)
    margin = logits[:, FOCUS_CLASS] - other.max(dim=1).values
    restricted = (
        ((targets == 0) | (targets == 2) | (targets == 4))
        & (keeper_predictions == FOCUS_CLASS)
    )
    protected = (targets == FOCUS_CLASS) & (keeper_predictions == FOCUS_CLASS)
    zero = logits.sum() * 0.0
    fp_term = F.softplus(margin[restricted] + FP_MARGIN).mean() if restricted.any() else zero
    tp_term = F.softplus(TP_MARGIN - margin[protected]).mean() if protected.any() else zero
    cross_entropy = F.cross_entropy(logits, targets)
    total = cross_entropy + FP_WEIGHT * fp_term + TP_WEIGHT * tp_term
    return total, {
        "cross_entropy": cross_entropy,
        "restricted_fp": fp_term,
        "protected_tp": tp_term,
        "restricted_rows": restricted.sum(),
        "protected_rows": protected.sum(),
    }


def _index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{int(index)}\n".encode("ascii"))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        to_serializable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_runtime_state() -> Dict[str, bool]:
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _numeric_runtime_matches_cidt() -> bool:
    return _numeric_runtime_state() == {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(dict(payload)), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _read_prediction_csv(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: Dict[str, object] = {
                "condition": str(raw["condition"]),
                "sample_index": int(raw["sample_index"]),
                "source_stem": str(raw["source_stem"]),
                "fold": int(raw["fold"]),
                "target": int(raw["target"]),
                "psnr": float(raw["psnr"]),
                "mean_absolute_rgb_change": float(raw["mean_absolute_rgb_change"]),
                "max_abs_mean_channel_shift": float(raw["max_abs_mean_channel_shift"]),
                "out_of_range_fraction": float(raw["out_of_range_fraction"]),
                "minimum_channel_variance": float(raw["minimum_channel_variance"]),
                "candidate_constant": str(raw["candidate_constant"]).casefold() == "true",
            }
            for role in EVALUATED_ROLES:
                for class_index in range(NUM_CLASSES):
                    row[f"prob_{role}_{class_index}"] = float(
                        raw[f"prob_{role}_{class_index}"]
                    )
            records.append(row)
    return records


def _epoch_orders(fit_positions: np.ndarray, *, fold: int) -> Tuple[List[np.ndarray], str]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    orders: List[np.ndarray] = []
    digest = hashlib.sha256()
    for epoch in range(EPOCHS):
        generator = np.random.default_rng(SEED + 100_003 * int(fold) + 997 * epoch)
        order = generator.permutation(positions)
        orders.append(order)
        digest.update(f"fold={fold};epoch={epoch}\n".encode("ascii"))
        digest.update(np.ascontiguousarray(order).tobytes())
    return orders, digest.hexdigest()


def _repeat_metadata(
    metadata: Mapping[str, object], repeats: int
) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for key in ("bbox", "image_mask"):
        value = metadata.get(key)
        if torch.is_tensor(value):
            result[key] = torch.cat([value] * int(repeats), dim=0)
    return result


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
    amp: bool,
) -> Tensor:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        logits, _ = _forward_classification_with_metadata(
            model, images, metadata, device=device
        )
    return logits.float()


def _cpu_tree(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _new_training_layers(
    *, mean: Sequence[float], std: Sequence[float], device: torch.device
) -> Dict[str, JpegDlInputLayer]:
    return {
        PAPER_ROLE: JpegDlInputLayer(mean=mean, std=std).to(device),
        SCALAR_ROLE: JpegDlInputLayer(
            mean=mean, std=std, scalar_tables=True
        ).to(device),
        CANDIDATE_ROLE: JpegDlInputLayer(mean=mean, std=std).to(device),
    }


def _training_loss_for_role(
    role: str,
    logits: Tensor,
    targets: Tensor,
    keeper_predictions: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if role == PAPER_ROLE:
        cross_entropy = F.cross_entropy(logits, targets)
        zero = cross_entropy * 0.0
        return cross_entropy, {
            "cross_entropy": cross_entropy,
            "restricted_fp": zero,
            "protected_tp": zero,
        }
    if role not in (SCALAR_ROLE, CANDIDATE_ROLE):
        raise ValueError(f"Unsupported JPEG-DL training role: {role}")
    total, parts = precision_loss(logits, targets, keeper_predictions)
    return total, {
        "cross_entropy": parts["cross_entropy"],
        "restricted_fp": parts["restricted_fp"],
        "protected_tp": parts["protected_tp"],
    }


def _backward_role_microbatches(
    *,
    model: nn.Module,
    layers: Mapping[str, JpegDlInputLayer],
    images: Tensor,
    targets: Tensor,
    keeper_predictions: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
) -> Tuple[Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]:
    if TRAIN_ROLE_MICROBATCH_SIZE != 1:
        raise RuntimeError("JPEG-DL A0 training role microbatch lock differs")
    losses: Dict[str, Tensor] = {}
    parts: Dict[str, Dict[str, Tensor]] = {}
    for role in TRAINED_ROLES:
        transformed = layers[role](images)
        logits = _forward_logits(model, transformed, metadata, device=device, amp=True)
        total, role_parts = _training_loss_for_role(
            role, logits, targets, keeper_predictions
        )
        total.backward()
        losses[role] = total.detach()
        parts[role] = {key: value.detach() for key, value in role_parts.items()}
        del transformed, logits, total, role_parts
    return losses, parts


def train_fold(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    rows: Sequence[CleanTrainRow],
    fold: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    num_workers: int,
) -> Dict[str, object]:
    fold = int(fold)
    fit_positions = np.asarray(
        [row.sample_index for row in rows if int(row.fold) != fold], dtype=np.int64
    )
    held_positions = np.asarray(
        [row.sample_index for row in rows if int(row.fold) == fold], dtype=np.int64
    )
    fit_sources = {rows[int(index)].source_stem for index in fit_positions}
    held_sources = {rows[int(index)].source_stem for index in held_positions}
    overlap = sorted(fit_sources.intersection(held_sources))
    if overlap:
        raise ValueError(f"JPEG-DL fold {fold} has source overlap")
    layers = _new_training_layers(mean=mean, std=std, device=device)
    optimizer = torch.optim.Adam(
        [parameter for layer in layers.values() for parameter in layer.parameters()],
        lr=JPEG_LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    initial_hashes = {
        role: _state_mapping_sha256(layer.state_dict()) for role, layer in layers.items()
    }
    if len({initial_hashes[role] for role in (PAPER_ROLE, CANDIDATE_ROLE)}) != 1:
        raise RuntimeError("Full JPEG-DL roles did not start byte-identically")
    orders, occurrence_sha = _epoch_orders(fit_positions, fold=fold)
    epoch_records = []
    gradient_records: Dict[str, Dict[str, object]] = {}
    started = time.perf_counter()
    for epoch, order in enumerate(orders):
        loader, loader_summary = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=order.tolist(),
            brightness=1.0,
            contrast=1.0,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            context=f"jpeg_dl_fold_{fold}_epoch_{epoch}",
        )
        totals = {role: 0.0 for role in TRAINED_ROLES}
        component_totals = {
            role: {"cross_entropy": 0.0, "restricted_fp": 0.0, "protected_tp": 0.0}
            for role in TRAINED_ROLES
        }
        batches = 0
        rows_seen = 0
        for images_cpu, targets_cpu, metadata in loader:
            if not isinstance(metadata, Mapping) or not torch.is_tensor(
                metadata.get("sample_index")
            ):
                raise ValueError("JPEG-DL training requires sample-index metadata")
            sample_indices = metadata["sample_index"].long().tolist()
            expected_targets = [rows[index].target for index in sample_indices]
            if targets_cpu.long().tolist() != expected_targets:
                raise ValueError("JPEG-DL training target order differs")
            images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
            keeper_predictions = torch.tensor(
                [rows[index].keeper_prediction for index in sample_indices],
                device=device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            losses, parts = _backward_role_microbatches(
                model=model,
                layers=layers,
                images=images,
                targets=targets,
                keeper_predictions=keeper_predictions,
                metadata=metadata,
                device=device,
            )
            for role, layer in layers.items():
                gradient = layer.q.grad
                gradient_records[role] = {
                    "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                    "nonzero": bool(gradient is not None and float(gradient.abs().max()) > 0.0),
                    "maximum_absolute": float(gradient.abs().max()) if gradient is not None else 0.0,
                }
            optimizer.step()
            for layer in layers.values():
                layer.project_()
            batch_rows = int(targets.numel())
            rows_seen += batch_rows
            batches += 1
            for role in TRAINED_ROLES:
                totals[role] += float(losses[role].detach()) * batch_rows
                for key in component_totals[role]:
                    component_totals[role][key] += float(parts[role][key].detach()) * batch_rows
        if rows_seen != int(fit_positions.size):
            raise ValueError(
                f"JPEG-DL fold {fold} epoch {epoch} saw {rows_seen} rows, expected {fit_positions.size}"
            )
        epoch_records.append(
            {
                "epoch": epoch,
                "rows": rows_seen,
                "batches": batches,
                "loader": loader_summary,
                "mean_losses": {
                    role: totals[role] / max(1, rows_seen) for role in TRAINED_ROLES
                },
                "mean_components": {
                    role: {
                        key: value / max(1, rows_seen)
                        for key, value in component_totals[role].items()
                    }
                    for role in TRAINED_ROLES
                },
                "q_hashes": {
                    role: _state_mapping_sha256(layer.state_dict())
                    for role, layer in layers.items()
                },
            }
        )
    elapsed = time.perf_counter() - started
    states = {role: _cpu_tree(layer.state_dict()) for role, layer in layers.items()}
    result = {
        "fold": fold,
        "fit_rows": int(fit_positions.size),
        "held_rows": int(held_positions.size),
        "fit_sources": len(fit_sources),
        "held_sources": len(held_sources),
        "source_overlap": overlap,
        "occurrence_sha256": occurrence_sha,
        "initial_hashes": initial_hashes,
        "final_hashes": {
            role: _state_mapping_sha256(state) for role, state in states.items()
        },
        "epochs": epoch_records,
        "gradient_records": gradient_records,
        "all_gradient_roles_live": all(
            value["finite"] and value["nonzero"] for value in gradient_records.values()
        ),
        "seconds": elapsed,
        "images_per_second_role_equivalent": (
            float(EPOCHS * fit_positions.size * len(TRAINED_ROLES) / elapsed)
            if elapsed > 0.0
            else 0.0
        ),
        "training_role_microbatch_size": TRAIN_ROLE_MICROBATCH_SIZE,
        "states": states,
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
    }
    for layer in layers.values():
        layer.cpu()
    del layers, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_fold_layers(
    fold_training: Mapping[str, object],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> Dict[str, JpegDlInputLayer]:
    layers = _new_training_layers(mean=mean, std=std, device=device)
    states = fold_training["states"]
    if not isinstance(states, Mapping):
        raise TypeError("JPEG-DL fold states must be a mapping")
    for role, layer in layers.items():
        layer.load_state_dict(states[role], strict=True)
        layer.eval()
    return layers


def _role_probabilities(
    *,
    model: nn.Module,
    role_inputs: Mapping[str, Tensor],
    metadata: Mapping[str, object],
    device: torch.device,
    amp: bool = False,
) -> Dict[str, Tensor]:
    names = list(role_inputs)
    result: Dict[str, Tensor] = {}
    for start in range(0, len(names), ROLE_CHUNK_SIZE):
        chunk_names = names[start : start + ROLE_CHUNK_SIZE]
        joined = torch.cat([role_inputs[name] for name in chunk_names], dim=0)
        repeated = _repeat_metadata(metadata, len(chunk_names))
        logits = _forward_logits(model, joined, repeated, device=device, amp=amp)
        for name, value in zip(chunk_names, logits.chunk(len(chunk_names), dim=0)):
            result[name] = value.softmax(dim=1)
    return result


def _reconstruction_rows(
    original: Tensor,
    candidate: Tensor,
    valid_mask: Optional[Tensor],
) -> List[Dict[str, object]]:
    original = original.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if valid_mask is None:
        valid = torch.ones(
            (int(original.shape[0]), 1, int(original.shape[-2]), int(original.shape[-1])),
            dtype=torch.bool,
        )
    else:
        valid = valid_mask.detach().cpu().bool()
        if valid.ndim == 3:
            valid = valid[:, None]
    valid_rgb = valid.expand(-1, 3, -1, -1)
    rows = []
    for index in range(int(original.shape[0])):
        selected = valid_rgb[index]
        source = original[index][selected]
        output = candidate[index][selected]
        difference = output - source
        mse = float(torch.square(difference).mean()) if output.numel() else float("inf")
        psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        channel_shifts = []
        channel_variances = []
        for channel in range(3):
            channel_valid = valid[index, 0]
            channel_delta = candidate[index, channel][channel_valid] - original[index, channel][
                channel_valid
            ]
            channel_output = candidate[index, channel][channel_valid]
            channel_shifts.append(float(channel_delta.mean()) if channel_delta.numel() else 0.0)
            channel_variances.append(
                float(channel_output.var(unbiased=False)) if channel_output.numel() else 0.0
            )
        out_of_range = ((output < 0.0) | (output > 1.0)).float()
        rows.append(
            {
                "psnr": psnr,
                "mean_absolute_rgb_change": float(difference.abs().mean()),
                "max_abs_mean_channel_shift": max(abs(value) for value in channel_shifts),
                "out_of_range_fraction": float(out_of_range.mean()),
                "minimum_channel_variance": min(channel_variances),
                "candidate_constant": min(channel_variances) <= 1e-12,
            }
        )
    return rows


def evaluate_fold(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    rows: Sequence[CleanTrainRow],
    fold_training: Mapping[str, object],
    fold: int,
    condition: str,
    brightness: float,
    contrast: float,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    num_workers: int,
    cohort_only: bool = False,
) -> Dict[str, object]:
    fold = int(fold)
    if cohort_only:
        selected_rows = [row for row in locked_cohort(rows) if int(row.fold) == fold]
    else:
        selected_rows = [row for row in rows if int(row.fold) == fold]
    selected_indices = [row.sample_index for row in selected_rows]
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=selected_indices,
        brightness=brightness,
        contrast=contrast,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        context=f"jpeg_dl_{condition}_fold_{fold}",
    )
    layers = _load_fold_layers(
        fold_training, mean=mean, std=std, device=device
    )
    fixed = JpegDlInputLayer(mean=mean, std=std, trainable=False).to(device).eval()
    candidate = layers[CANDIDATE_ROLE]
    records: List[Dict[str, object]] = []
    position = 0
    cidt_error = 0.0
    cidt_argmax_exact = True
    state_before = _model_state_sha256(model)
    started = time.perf_counter()
    with torch.inference_mode():
        for images_cpu, targets_cpu, metadata in loader:
            if not isinstance(metadata, Mapping) or not torch.is_tensor(
                metadata.get("sample_index")
            ):
                raise ValueError("JPEG-DL evaluation requires sample-index metadata")
            sample_indices = metadata["sample_index"].long().tolist()
            batch = len(sample_indices)
            expected = selected_rows[position : position + batch]
            if sample_indices != [row.sample_index for row in expected]:
                raise ValueError("JPEG-DL evaluation order differs")
            if targets_cpu.long().tolist() != [row.target for row in expected]:
                raise ValueError("JPEG-DL evaluation targets differ")
            images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            raw_probabilities = _forward_logits(
                model, images, metadata, device=device, amp=False
            ).softmax(dim=1)
            candidate_normalized, candidate_rgb = candidate(images, return_rgb=True)
            role_inputs = {
                FIXED_ROLE: fixed(images),
                PAPER_ROLE: layers[PAPER_ROLE](images),
                SCALAR_ROLE: layers[SCALAR_ROLE](images),
                CANDIDATE_ROLE: candidate_normalized,
                PERMUTED_ROLE: candidate(images, frequency_permuted=True),
                PHASE_ROLE: candidate(images, phase_shifted=True),
            }
            role_probabilities = _role_probabilities(
                model=model,
                role_inputs=role_inputs,
                metadata=metadata,
                device=device,
            )
            role_probabilities[KEEPER_ROLE] = raw_probabilities
            image_mask = metadata.get("image_mask")
            original_rgb = candidate.rgb01_from_normalized(images)
            reconstruction = _reconstruction_rows(
                original_rgb,
                candidate_rgb,
                image_mask if torch.is_tensor(image_mask) else None,
            )
            for batch_index, row in enumerate(expected):
                cidt = np.asarray(row.keeper_probabilities, dtype=np.float64)
                live = raw_probabilities[batch_index].detach().cpu().numpy().astype(np.float64)
                cidt_error = max(cidt_error, float(np.max(np.abs(cidt - live))))
                cidt_argmax_exact = cidt_argmax_exact and int(np.argmax(cidt)) == int(
                    np.argmax(live)
                )
                record: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": row.sample_index,
                    "source_stem": row.source_stem,
                    "fold": row.fold,
                    "target": row.target,
                    **reconstruction[batch_index],
                }
                for role in EVALUATED_ROLES:
                    probabilities = role_probabilities[role][batch_index].detach().cpu().tolist()
                    for class_index, value in enumerate(probabilities):
                        record[f"prob_{role}_{class_index}"] = float(value)
                records.append(record)
            position += batch
    elapsed = time.perf_counter() - started
    state_after = _model_state_sha256(model)
    if position != len(selected_rows):
        raise ValueError("JPEG-DL evaluation did not consume every selected row")
    for layer in layers.values():
        layer.cpu()
    fixed.cpu()
    del layers, fixed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checks = {
        "row_count_exact": len(records) == len(selected_rows),
        "cidt_argmax_exact": cidt_argmax_exact,
        "cidt_probability_error_lte_0p015": cidt_error
        <= FORMAL_BATCH32_CIDT_TELEMETRY_TOLERANCE,
        "keeper_state_exact": state_before == state_after,
        "all_probabilities_finite": all(
            math.isfinite(float(row[f"prob_{role}_{class_index}"]))
            for row in records
            for role in EVALUATED_ROLES
            for class_index in range(NUM_CLASSES)
        ),
    }
    return {
        "records": records,
        "loader": loader_summary,
        "seconds": elapsed,
        "images_per_second": len(records) / elapsed if elapsed > 0.0 else 0.0,
        "cidt_probability_max_abs_error": cidt_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _probability_matrix(records: Sequence[Mapping[str, object]], role: str) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"prob_{role}_{class_index}"]) for class_index in range(NUM_CLASSES)]
            for row in records
        ],
        dtype=np.float64,
    )


def _role_analysis(
    records: Sequence[Mapping[str, object]], role: str
) -> Dict[str, object]:
    targets = np.asarray([int(row["target"]) for row in records], dtype=np.int64)
    raw_probabilities = _probability_matrix(records, KEEPER_ROLE)
    probabilities = _probability_matrix(records, role)
    raw_predictions = raw_probabilities.argmax(axis=1)
    predictions = probabilities.argmax(axis=1)
    classification = _classification_metrics(
        targets, predictions, num_classes=NUM_CLASSES
    )
    decision = (targets == FOCUS_CLASS) | (
        np.isin(targets, RESTRICTED_CLASSES) & (raw_predictions == FOCUS_CLASS)
    )
    labels = (targets[decision] == FOCUS_CLASS).astype(np.int64)
    p1 = probabilities[decision, FOCUS_CLASS]
    events = directional_event_masks(
        targets, raw_predictions, predictions, focus_class=FOCUS_CLASS
    )
    keeper_tp = (targets == FOCUS_CLASS) & (raw_predictions == FOCUS_CLASS)
    keeper_fn = (targets == FOCUS_CLASS) & (raw_predictions != FOCUS_CLASS)
    restricted_fp = np.isin(targets, RESTRICTED_CLASSES) & (
        raw_predictions == FOCUS_CLASS
    )
    fp_removed = restricted_fp & (predictions != FOCUS_CLASS)
    fp_removed_correct = restricted_fp & (predictions == targets)
    fp_created = np.isin(targets, RESTRICTED_CLASSES) & (
        raw_predictions != FOCUS_CLASS
    ) & (predictions == FOCUS_CLASS)
    tp_break = keeper_tp & (predictions != FOCUS_CLASS)
    fn_support = keeper_fn & (predictions == FOCUS_CLASS)
    source_net: MutableMapping[str, int] = {}
    for index, row in enumerate(records):
        source = str(row["source_stem"])
        delta = int(events["candidate_correction"][index]) - int(
            events["candidate_harm"][index]
        )
        source_net[source] = source_net.get(source, 0) + delta
    positive_source = [value for value in source_net.values() if value > 0]
    source_fraction = (
        max(positive_source) / sum(positive_source) if positive_source else 1.0
    )
    return {
        "rows": len(records),
        "classification": classification,
        "class1_precision": float(classification["per_class_precision"][FOCUS_CLASS]),
        "class1_recall": float(classification["per_class_recall"][FOCUS_CLASS]),
        "class1_f1": float(classification["per_class_f1"][FOCUS_CLASS]),
        "decision_rows": int(decision.sum()),
        "p1_auroc": float(roc_auc_score(labels, p1)),
        "p1_auprc": float(average_precision_score(labels, p1)),
        "keeper_tp": int(keeper_tp.sum()),
        "keeper_fn": int(keeper_fn.sum()),
        "restricted_fp": int(restricted_fp.sum()),
        "tp_retained": int((keeper_tp & (predictions == FOCUS_CLASS)).sum()),
        "tp_broken": int(tp_break.sum()),
        "tp_retention": float(
            1.0 - tp_break.sum() / max(1, int(keeper_tp.sum()))
        ),
        "fn_supported": int(fn_support.sum()),
        "fp_removed": int(fp_removed.sum()),
        "fp_removed_correct": int(fp_removed_correct.sum()),
        "fp_created": int(fp_created.sum()),
        "fp_rejection": float(fp_removed.sum() / max(1, int(restricted_fp.sum()))),
        "corrections": int(events["candidate_correction"].sum()),
        "harms": int(events["candidate_harm"].sum()),
        "net_corrections": int(
            events["candidate_correction"].sum() - events["candidate_harm"].sum()
        ),
        "maximum_positive_source_fraction": float(source_fraction),
    }


def analyze_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    values = list(records)
    if not values:
        raise ValueError("JPEG-DL analysis requires records")
    condition_values = sorted({str(row["condition"]) for row in values})
    conditions: Dict[str, object] = {}
    for condition in condition_values:
        selected = [row for row in values if str(row["condition"]) == condition]
        folds = sorted({int(row["fold"]) for row in selected})
        conditions[condition] = {
            "rows": len(selected),
            "folds": folds,
            "roles": {
                role: _role_analysis(selected, role) for role in EVALUATED_ROLES
            },
            "per_fold": {
                str(fold): {
                    role: _role_analysis(
                        [row for row in selected if int(row["fold"]) == fold], role
                    )
                    for role in EVALUATED_ROLES
                }
                for fold in folds
            },
        }
    return {"conditions": conditions}


def reconstruction_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    rows = list(records)
    values = {
        "mean_psnr": float(np.mean([float(row["psnr"]) for row in rows])),
        "mean_absolute_rgb_change": float(
            np.mean([float(row["mean_absolute_rgb_change"]) for row in rows])
        ),
        "maximum_abs_mean_channel_shift": float(
            np.max([float(row["max_abs_mean_channel_shift"]) for row in rows])
        ),
        "mean_out_of_range_fraction": float(
            np.mean([float(row["out_of_range_fraction"]) for row in rows])
        ),
        "minimum_channel_variance": float(
            np.min([float(row["minimum_channel_variance"]) for row in rows])
        ),
        "constant_images": int(sum(bool(row["candidate_constant"]) for row in rows)),
    }
    checks = {
        "mean_psnr_gte_28": values["mean_psnr"] >= 28.0,
        "mean_absolute_change_lte_0p05": values["mean_absolute_rgb_change"] <= 0.05,
        "channel_shift_lte_0p02": values["maximum_abs_mean_channel_shift"] <= 0.02,
        "out_of_range_lte_0p005": values["mean_out_of_range_fraction"] <= 0.005,
        "all_channels_nonconstant": values["minimum_channel_variance"] > 1e-12,
        "no_constant_images": values["constant_images"] == 0,
    }
    return {**values, "checks": checks, "passed": all(checks.values())}


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(left_rank.std()) <= 1e-12 or float(right_rank.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def q_diagnostics(fold_trainings: Mapping[int, Mapping[str, object]]) -> Dict[str, object]:
    fold_values: Dict[str, object] = {}
    candidate_deltas = []
    for fold in sorted(fold_trainings):
        training = fold_trainings[fold]
        states = training["states"]
        role_values = {}
        for role in TRAINED_ROLES:
            q = np.asarray(states[role]["q"], dtype=np.float64)
            expanded = (
                np.broadcast_to(q, (2, 8, 8)).copy()
                if q.shape == (2, 1, 1)
                else q.copy()
            )
            delta = expanded - 1.0
            role_values[role] = {
                "shape": list(q.shape),
                "minimum": float(expanded.min()),
                "maximum": float(expanded.max()),
                "mean": float(expanded.mean()),
                "standard_deviation": float(expanded.std()),
                "entries_changed_ge_0p01": int((np.abs(delta) >= 0.01).sum()),
                "near_bound_fraction": float(
                    ((expanded <= MIN_Q + 1e-4) | (expanded >= MAX_Q - 1e-4)).mean()
                ),
                "y_ac_standard_deviation": float(expanded[0].reshape(-1)[1:].std()),
                "cbcr_ac_standard_deviation": float(expanded[1].reshape(-1)[1:].std()),
                "values": expanded.tolist(),
            }
            if role == CANDIDATE_ROLE:
                candidate_deltas.append(delta.reshape(-1))
        fold_values[str(fold)] = role_values
    pairwise_spearman = []
    pairwise_sign_cosine = []
    for left in range(len(candidate_deltas)):
        for right in range(left + 1, len(candidate_deltas)):
            first = candidate_deltas[left]
            second = candidate_deltas[right]
            pairwise_spearman.append(_spearman(first, second))
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            pairwise_sign_cosine.append(
                float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0
            )
    candidate_checks = []
    for fold in fold_values.values():
        candidate = fold[CANDIDATE_ROLE]
        candidate_checks.append(
            bool(
                candidate["near_bound_fraction"] <= 0.05
                and candidate["entries_changed_ge_0p01"] >= 16
                and candidate["y_ac_standard_deviation"] >= 0.01
                and candidate["cbcr_ac_standard_deviation"] >= 0.01
            )
        )
    consistency_required = len(candidate_deltas) > 1
    median_spearman = float(np.median(pairwise_spearman)) if pairwise_spearman else 1.0
    median_cosine = float(np.median(pairwise_sign_cosine)) if pairwise_sign_cosine else 1.0
    checks = {
        "all_candidate_tables_active_noncollapsed": all(candidate_checks),
        "median_delta_spearman_gte_0p40": (not consistency_required)
        or median_spearman >= 0.40,
        "median_delta_sign_cosine_gte_0p50": (not consistency_required)
        or median_cosine >= 0.50,
    }
    return {
        "folds": fold_values,
        "pairwise_delta_spearman": pairwise_spearman,
        "pairwise_delta_sign_cosine": pairwise_sign_cosine,
        "median_delta_spearman": median_spearman,
        "median_delta_sign_cosine": median_cosine,
        "checks": checks,
        "passed": all(checks.values()),
    }


def stage_a_gates(
    analysis: Mapping[str, object],
    reconstruction: Mapping[str, object],
    quantizers: Mapping[str, object],
    fold_training: Mapping[str, object],
) -> Dict[str, bool]:
    clean = analysis["conditions"]["clean"]
    raw = clean["roles"][KEEPER_ROLE]
    candidate = clean["roles"][CANDIDATE_ROLE]
    paper = clean["roles"][PAPER_ROLE]
    scalar = clean["roles"][SCALAR_ROLE]
    return {
        "candidate_auroc_gain_gte_0p005": candidate["p1_auroc"]
        >= raw["p1_auroc"] + 0.005,
        "candidate_precision_gain_gte_0p005": candidate["class1_precision"]
        >= raw["class1_precision"] + 0.005,
        "candidate_recall_drop_lte_0p02": candidate["class1_recall"]
        >= raw["class1_recall"] - 0.02,
        "candidate_removes_at_least_5_fp": candidate["fp_removed"] >= 5,
        "candidate_breaks_at_most_2_tp": candidate["tp_broken"] <= 2,
        "candidate_macro_f1_drop_lte_0p005": candidate["classification"]["macro_f1"]
        >= raw["classification"]["macro_f1"] - 0.005,
        "candidate_beats_paper_auroc_by_0p002": candidate["p1_auroc"]
        >= paper["p1_auroc"] + 0.002,
        "candidate_beats_scalar_auroc_by_0p002": candidate["p1_auroc"]
        >= scalar["p1_auroc"] + 0.002,
        "reconstruction_anti_collapse": bool(reconstruction["passed"]),
        "quantizer_anti_collapse": bool(quantizers["passed"]),
        "all_gradient_roles_live": bool(fold_training["all_gradient_roles_live"]),
    }


def clean_oof_gates(
    analysis: Mapping[str, object],
    reconstruction: Mapping[str, object],
    quantizers: Mapping[str, object],
) -> Dict[str, bool]:
    clean = analysis["conditions"]["clean"]
    raw = clean["roles"][KEEPER_ROLE]
    candidate = clean["roles"][CANDIDATE_ROLE]
    paper = clean["roles"][PAPER_ROLE]
    scalar = clean["roles"][SCALAR_ROLE]
    fixed = clean["roles"][FIXED_ROLE]
    permuted = clean["roles"][PERMUTED_ROLE]
    phase = clean["roles"][PHASE_ROLE]
    raw_per_class = raw["classification"]["per_class_f1"]
    candidate_per_class = candidate["classification"]["per_class_f1"]
    candidate_auc_gain = candidate["p1_auroc"] - raw["p1_auroc"]
    candidate_precision_gain = candidate["class1_precision"] - raw["class1_precision"]
    fixed_auc_gain = fixed["p1_auroc"] - raw["p1_auroc"]
    fixed_precision_gain = fixed["class1_precision"] - raw["class1_precision"]
    per_fold = clean["per_fold"]
    fold_precision = []
    fold_f1 = []
    fold_tp = []
    fold_fp = []
    for fold in FOLDS:
        fold_raw = per_fold[str(fold)][KEEPER_ROLE]
        fold_candidate = per_fold[str(fold)][CANDIDATE_ROLE]
        fold_precision.append(
            fold_candidate["class1_precision"] >= fold_raw["class1_precision"]
        )
        fold_f1.append(fold_candidate["class1_f1"] >= fold_raw["class1_f1"])
        fold_tp.append(fold_candidate["tp_retention"] >= 0.95)
        fold_fp.append(fold_candidate["fp_removed"] >= fold_candidate["fp_created"])
    controls = {}
    for name, control in ((PERMUTED_ROLE, permuted), (PHASE_ROLE, phase)):
        controls[name] = (
            candidate["p1_auroc"] >= control["p1_auroc"] + 0.005
            or candidate["fp_removed"] >= control["fp_removed"] + 5
        )
    return {
        "precision_gain_gte_0p010": candidate["class1_precision"]
        >= raw["class1_precision"] + 0.010,
        "precision_absolute_gte_0p72": candidate["class1_precision"] >= 0.72,
        "class1_f1_gain_gte_0p005": candidate["class1_f1"]
        >= raw["class1_f1"] + 0.005,
        "recall_drop_lte_0p010": candidate["class1_recall"]
        >= raw["class1_recall"] - 0.010,
        "tp_retention_gte_0p98": candidate["tp_retention"] >= 0.98,
        "fp_rejection_gte_0p15": candidate["fp_rejection"] >= 0.15,
        "net_corrections_gte_20": candidate["net_corrections"] >= 20,
        "auroc_gain_gte_0p010": candidate["p1_auroc"] >= raw["p1_auroc"] + 0.010,
        "auprc_non_decrease": candidate["p1_auprc"] >= raw["p1_auprc"],
        "macro_f1_drop_lte_0p002": candidate["classification"]["macro_f1"]
        >= raw["classification"]["macro_f1"] - 0.002,
        "nonfocus_f1_drop_lte_0p010": all(
            candidate_per_class[index] >= raw_per_class[index] - 0.010
            for index in range(NUM_CLASSES)
            if index != FOCUS_CLASS
        ),
        "four_folds_precision_non_decrease": sum(fold_precision) >= 4,
        "three_folds_f1_non_decrease": sum(fold_f1) >= 3,
        "all_folds_tp_retention_gte_0p95": all(fold_tp),
        "all_folds_no_net_fp_creation": all(fold_fp),
        "beats_paper_auroc_by_0p005": candidate["p1_auroc"]
        >= paper["p1_auroc"] + 0.005,
        "beats_scalar_auroc_by_0p005": candidate["p1_auroc"]
        >= scalar["p1_auroc"] + 0.005,
        "beats_paper_fp_by_5_without_more_tp_harm": candidate["fp_removed"]
        >= paper["fp_removed"] + 5
        and candidate["tp_broken"] <= paper["tp_broken"],
        "beats_scalar_fp_by_5_without_more_tp_harm": candidate["fp_removed"]
        >= scalar["fp_removed"] + 5
        and candidate["tp_broken"] <= scalar["tp_broken"],
        "fixed_q1_not_half_auc_gain": fixed_auc_gain <= 0.5 * candidate_auc_gain,
        "fixed_q1_not_half_precision_gain": fixed_precision_gain
        <= 0.5 * candidate_precision_gain,
        "frequency_permutation_causal": controls[PERMUTED_ROLE],
        "block_phase_causal": controls[PHASE_ROLE],
        "source_concentration_lte_0p35": candidate["maximum_positive_source_fraction"]
        <= 0.35,
        "reconstruction_anti_collapse": bool(reconstruction["passed"]),
        "quantizer_anti_collapse_and_consistency": bool(quantizers["passed"]),
    }


def shifted_condition_gates(analysis: Mapping[str, object]) -> Dict[str, bool]:
    conditions = analysis["conditions"]
    gates: Dict[str, bool] = {}
    f1_improvements = 0
    for name, _, _ in CONDITIONS:
        condition = conditions[name]
        raw = condition["roles"][KEEPER_ROLE]
        candidate = condition["roles"][CANDIDATE_ROLE]
        gates[f"{name}_precision_non_decrease"] = (
            candidate["class1_precision"] >= raw["class1_precision"]
        )
        gates[f"{name}_tp_retention_gte_0p95"] = candidate["tp_retention"] >= 0.95
        gates[f"{name}_restricted_fp_not_increased"] = (
            candidate["fp_removed"] >= candidate["fp_created"]
        )
        gates[f"{name}_auroc_drop_lte_0p01"] = (
            candidate["p1_auroc"] >= raw["p1_auroc"] - 0.01
        )
        if candidate["class1_f1"] > raw["class1_f1"]:
            f1_improvements += 1
    gates["at_least_two_conditions_improve_class1_f1"] = f1_improvements >= 2
    return gates


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    provenance = verify_locked_inputs(args)
    equations = engineering_checks()
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
    cohort = locked_cohort(rows)
    checks = {
        "equations_pass": bool(equations["passed"]),
        "train_rows_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "train_class_counts_exact": tuple(
            sum(row.target == class_index for row in rows)
            for class_index in range(NUM_CLASSES)
        )
        == EXPECTED_TRAIN_CLASS_COUNTS,
        "holdout_counts_exact": tuple(
            sum(row.fold == fold for row in rows) for fold in FOLDS
        )
        == EXPECTED_HOLDOUT_COUNTS,
        "train_index_hash_exact": _index_sha256([row.sample_index for row in rows])
        == EXPECTED_TRAIN_INDEX_SHA256,
        "cohort_rows_exact": len(cohort) == EXPECTED_COHORT_ROWS,
        "cohort_tp_exact": sum(
            row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
            for row in cohort
        )
        == EXPECTED_KEEPER_TP,
        "cohort_fn_exact": sum(
            row.target == FOCUS_CLASS and row.keeper_prediction != FOCUS_CLASS
            for row in cohort
        )
        == EXPECTED_KEEPER_FN,
        "cohort_fp_exact": sum(row.target != FOCUS_CLASS for row in cohort)
        == EXPECTED_RESTRICTED_FP,
        "cohort_index_hash_exact": _index_sha256(
            [row.sample_index for row in cohort]
        )
        == EXPECTED_COHORT_INDEX_SHA256,
        "protected_user_payloads_exact": bool(provenance["protected_untracked"]["passed"]),
    }
    return {
        "mode": f"{METHOD}_preflight",
        "passed": all(checks.values()),
        "checks": checks,
        "validation_data_used": False,
        "test_data_used": False,
        "candidate_pixels_used": False,
        "candidate_metrics_used": False,
        "provenance": provenance,
        "equations": equations,
        "implementation_observed_not_gated_until_formal": implementation_tracking_state(),
        "repo_state_observed_not_gated_until_formal": _repo_state(),
    }


def _first_engineering_rows(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = locked_cohort(rows)
    return [
        next(
            row
            for row in cohort
            if row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS
        ),
        next(row for row in cohort if row.target != FOCUS_CLASS),
    ]


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for JPEG-DL engineering forward")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model, checkpoint, _ = load_model(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
    selected = _first_engineering_rows(rows)
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, args.data)
    semantics = dataset_summary["semantics"]
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    loader, loader_summary = _make_condition_loader(
        base_dataset=dataset,
        transform=transform,
        indices=[row.sample_index for row in selected],
        brightness=1.0,
        contrast=1.0,
        batch_size=2,
        num_workers=0,
        context="jpeg_dl_engineering_forward",
    )
    images_cpu, targets_cpu, metadata = next(iter(loader))
    images = images_cpu.to(device=device, dtype=torch.float32)
    targets = targets_cpu.to(device=device, dtype=torch.long)
    keeper_predictions = torch.tensor(
        [row.keeper_prediction for row in selected], device=device, dtype=torch.long
    )
    model_state_before = _model_state_sha256(model)
    raw_probabilities = _forward_logits(
        model, images, metadata, device=device, amp=False
    ).softmax(dim=1)
    cidt_probabilities = torch.tensor(
        [row.keeper_probabilities for row in selected],
        device=device,
        dtype=torch.float32,
    )
    cidt_error = float((raw_probabilities - cidt_probabilities).abs().max())

    concatenated_layers = _new_training_layers(mean=mean, std=std, device=device)
    concatenated_transformed = [
        concatenated_layers[role](images) for role in TRAINED_ROLES
    ]
    concatenated_logits = _forward_logits(
        model,
        torch.cat(concatenated_transformed, dim=0),
        _repeat_metadata(metadata, len(TRAINED_ROLES)),
        device=device,
        amp=True,
    )
    concatenated_role_logits = dict(
        zip(
            TRAINED_ROLES,
            concatenated_logits.chunk(len(TRAINED_ROLES), dim=0),
        )
    )
    concatenated_losses = [
        _training_loss_for_role(
            role,
            concatenated_role_logits[role],
            targets,
            keeper_predictions,
        )[0]
        for role in TRAINED_ROLES
    ]
    sum(concatenated_losses).backward()
    concatenated_gradient_tensors = {
        role: layer.q.grad.detach().clone()
        for role, layer in concatenated_layers.items()
        if layer.q.grad is not None
    }
    concatenated_gradients = {
        role: {
            "finite": bool(layer.q.grad is not None and torch.isfinite(layer.q.grad).all()),
            "nonzero": bool(layer.q.grad is not None and float(layer.q.grad.abs().max()) > 0.0),
            "maximum_absolute": float(layer.q.grad.abs().max())
            if layer.q.grad is not None
            else 0.0,
        }
        for role, layer in concatenated_layers.items()
    }

    microbatch_layers = _new_training_layers(mean=mean, std=std, device=device)
    for layer in microbatch_layers.values():
        layer.zero_grad(set_to_none=True)
    microbatch_losses, _ = _backward_role_microbatches(
        model=model,
        layers=microbatch_layers,
        images=images,
        targets=targets,
        keeper_predictions=keeper_predictions,
        metadata=metadata,
        device=device,
    )
    microbatch_gradient_tensors = {
        role: layer.q.grad.detach().clone()
        for role, layer in microbatch_layers.items()
        if layer.q.grad is not None
    }
    microbatch_gradients = {
        role: {
            "finite": bool(layer.q.grad is not None and torch.isfinite(layer.q.grad).all()),
            "nonzero": bool(layer.q.grad is not None and float(layer.q.grad.abs().max()) > 0.0),
            "maximum_absolute": float(layer.q.grad.abs().max())
            if layer.q.grad is not None
            else 0.0,
        }
        for role, layer in microbatch_layers.items()
    }
    gradient_equivalence: Dict[str, Dict[str, float]] = {}
    for role in TRAINED_ROLES:
        concatenated_gradient = concatenated_gradient_tensors[role].double().flatten()
        microbatch_gradient = microbatch_gradient_tensors[role].double().flatten()
        gradient_equivalence[role] = {
            "maximum_absolute_error": float(
                (concatenated_gradient - microbatch_gradient).abs().max()
            ),
            "cosine_similarity": float(
                F.cosine_similarity(
                    concatenated_gradient.unsqueeze(0),
                    microbatch_gradient.unsqueeze(0),
                    dim=1,
                    eps=1e-12,
                )[0]
            ),
        }

    del concatenated_transformed, concatenated_role_logits, concatenated_losses
    del concatenated_logits, concatenated_gradient_tensors
    for layer in concatenated_layers.values():
        layer.cpu()
    del concatenated_layers
    gc.collect()
    torch.cuda.empty_cache()

    physical_cuda_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    safe_cuda_bytes = int(0.90 * physical_cuda_bytes)
    repeats = BATCH_SIZE // int(images.shape[0])
    large_images = images.repeat(repeats, 1, 1, 1)
    large_targets = targets.repeat(repeats)
    large_keeper_predictions = keeper_predictions.repeat(repeats)
    large_metadata = {
        key: value.repeat((repeats,) + (1,) * (value.ndim - 1))
        for key, value in metadata.items()
        if torch.is_tensor(value) and key in {"bbox", "image_mask"}
    }

    concatenated_batch_supported = True
    concatenated_peak_bytes = 0
    concatenated_error = None
    try:
        for layer in microbatch_layers.values():
            layer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        large_transformed = [
            microbatch_layers[role](large_images) for role in TRAINED_ROLES
        ]
        large_logits = _forward_logits(
            model,
            torch.cat(large_transformed, dim=0),
            _repeat_metadata(large_metadata, len(TRAINED_ROLES)),
            device=device,
            amp=True,
        )
        large_logits.float().square().mean().backward()
        torch.cuda.synchronize(device)
        concatenated_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        del large_transformed, large_logits
    except RuntimeError as error:
        if "out of memory" not in str(error).casefold():
            raise
        concatenated_batch_supported = False
        concatenated_error = str(error)
    finally:
        for layer in microbatch_layers.values():
            layer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    microbatch_batch_supported = True
    microbatch_peak_bytes = 0
    microbatch_error = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        _backward_role_microbatches(
            model=model,
            layers=microbatch_layers,
            images=large_images,
            targets=large_targets,
            keeper_predictions=large_keeper_predictions,
            metadata=large_metadata,
            device=device,
        )
        torch.cuda.synchronize(device)
        microbatch_peak_bytes = int(torch.cuda.max_memory_allocated(device))
    except RuntimeError as error:
        if "out of memory" not in str(error).casefold():
            raise
        microbatch_batch_supported = False
        microbatch_error = str(error)
    finally:
        for layer in microbatch_layers.values():
            layer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    gradient_equivalence_pass = all(
        value["maximum_absolute_error"] <= MICROBATCH_GRADIENT_MAX_ABS_ERROR
        and value["cosine_similarity"] >= MICROBATCH_GRADIENT_MIN_COSINE
        for value in gradient_equivalence.values()
    )
    concatenated_fits_safe_budget = bool(
        concatenated_batch_supported
        and concatenated_peak_bytes > 0
        and concatenated_peak_bytes <= safe_cuda_bytes
    )
    microbatch_fits_safe_budget = bool(
        microbatch_batch_supported
        and microbatch_peak_bytes > 0
        and microbatch_peak_bytes <= safe_cuda_bytes
    )
    model_state_after = _model_state_sha256(model)
    checks = {
        "provenance_verified": bool(provenance),
        "numeric_runtime_matches_cidt": _numeric_runtime_matches_cidt(),
        "two_real_rows_exact": int(targets.numel()) == 2,
        "cidt_argmax_exact": torch.equal(
            raw_probabilities.argmax(dim=1), cidt_probabilities.argmax(dim=1)
        ),
        "batch2_cidt_probability_error_lte_1e4": cidt_error
        <= ENGINEERING_CIDT_PROBABILITY_TOLERANCE,
        "all_concatenated_role_gradients_live": all(
            value["finite"] and value["nonzero"]
            for value in concatenated_gradients.values()
        ),
        "all_microbatch_role_gradients_live": all(
            value["finite"] and value["nonzero"]
            for value in microbatch_gradients.values()
        ),
        "microbatch_gradients_match_concatenated": gradient_equivalence_pass,
        "selected_batch32_role_microbatch_fits_90pct_vram": microbatch_fits_safe_budget,
        "keeper_state_exact": model_state_before == model_state_after,
        "all_outputs_finite": bool(torch.isfinite(raw_probabilities).all())
        and all(torch.isfinite(value) for value in microbatch_losses.values()),
    }
    result = {
        "mode": f"{METHOD}_engineering_forward",
        "passed": all(checks.values()),
        "output_created": False,
        "validation_data_used": False,
        "test_data_used": False,
        "candidate_metrics_used": False,
        "rows": len(selected),
        "checks": checks,
        "training_role_microbatch_size": TRAIN_ROLE_MICROBATCH_SIZE,
        "microbatch_gradients": microbatch_gradients,
        "concatenated_gradients": concatenated_gradients,
        "gradient_equivalence": gradient_equivalence,
        "cidt_probability_max_abs_error": cidt_error,
        "cidt_probability_keeper_compatibility_tolerance": (
            KEEPER_COMPATIBILITY_PROBABILITY_TOLERANCE
        ),
        "cidt_probability_batch32_telemetry_tolerance": (
            FORMAL_BATCH32_CIDT_TELEMETRY_TOLERANCE
        ),
        "physical_cuda_bytes": physical_cuda_bytes,
        "safe_cuda_bytes_90pct": safe_cuda_bytes,
        "concatenated_batch96": {
            "supported_without_oom": concatenated_batch_supported,
            "peak_cuda_bytes": concatenated_peak_bytes,
            "peak_cuda_gib": float(concatenated_peak_bytes / 2**30),
            "fits_safe_budget": concatenated_fits_safe_budget,
            "error": concatenated_error,
        },
        "selected_microbatch_batch32": {
            "supported_without_oom": microbatch_batch_supported,
            "peak_cuda_bytes": microbatch_peak_bytes,
            "peak_cuda_gib": float(microbatch_peak_bytes / 2**30),
            "fits_safe_budget": microbatch_fits_safe_budget,
            "error": microbatch_error,
        },
        "dataset": dataset_summary,
        "loader": loader_summary,
        "numeric_runtime": _numeric_runtime_state(),
    }
    for layer in microbatch_layers.values():
        layer.cpu()
    del microbatch_layers, model, images, large_images
    gc.collect()
    torch.cuda.empty_cache()
    return result


def keeper_compatibility_audit(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    rows: Sequence[CleanTrainRow],
    device: torch.device,
    num_workers: int,
) -> Dict[str, object]:
    indices = [row.sample_index for row in rows]
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=KEEPER_COMPATIBILITY_BATCH_SIZE,
        num_workers=num_workers,
        context="jpeg_dl_keeper_compatibility_pre_candidate",
    )
    state_before = _model_state_sha256(model)
    maximum_error = 0.0
    argmax_exact = True
    all_finite = True
    position = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images_cpu, targets_cpu, metadata in loader:
            if not isinstance(metadata, Mapping) or not torch.is_tensor(
                metadata.get("sample_index")
            ):
                raise ValueError("JPEG-DL keeper compatibility requires sample indices")
            sample_indices = metadata["sample_index"].long().tolist()
            batch = len(sample_indices)
            expected = rows[position : position + batch]
            if sample_indices != [row.sample_index for row in expected]:
                raise ValueError("JPEG-DL keeper compatibility order differs")
            if targets_cpu.long().tolist() != [row.target for row in expected]:
                raise ValueError("JPEG-DL keeper compatibility targets differ")
            images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            live = _forward_logits(
                model, images, metadata, device=device, amp=False
            ).softmax(dim=1)
            cached = torch.tensor(
                [row.keeper_probabilities for row in expected],
                device=device,
                dtype=torch.float32,
            )
            maximum_error = max(maximum_error, float((live - cached).abs().max()))
            argmax_exact = argmax_exact and torch.equal(
                live.argmax(dim=1), cached.argmax(dim=1)
            )
            all_finite = all_finite and bool(torch.isfinite(live).all())
            position += batch
    elapsed = time.perf_counter() - started
    checks = {
        "all_train_rows_exact": position == len(rows) == EXPECTED_TRAIN_ROWS,
        "cidt_argmax_exact": argmax_exact,
        "cidt_probability_error_lte_1e6": maximum_error
        <= KEEPER_COMPATIBILITY_PROBABILITY_TOLERANCE,
        "all_probabilities_finite": all_finite,
        "keeper_state_exact": state_before == _model_state_sha256(model),
        "numeric_runtime_matches_cidt": _numeric_runtime_matches_cidt(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rows": position,
        "batch_size": KEEPER_COMPATIBILITY_BATCH_SIZE,
        "cidt_probability_max_abs_error": maximum_error,
        "seconds": elapsed,
        "images_per_second": position / elapsed if elapsed > 0.0 else 0.0,
        "candidate_pixels_used": False,
        "candidate_metrics_used": False,
        "loader": loader_summary,
        "numeric_runtime": _numeric_runtime_state(),
    }


def _onnx_export_diagnostic(
    layer: JpegDlInputLayer,
    images_cpu: Tensor,
    path: Path,
) -> Dict[str, object]:
    try:
        import onnx
        import onnxruntime as ort

        wrapper = _JpegExport(layer).eval()
        inputs = images_cpu[:1].detach().float().cpu()
        torch.onnx.export(
            wrapper,
            (inputs,),
            str(path),
            input_names=["images"],
            output_names=["reconstructed"],
            opset_version=17,
            do_constant_folding=True,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        with torch.inference_mode():
            expected = wrapper(inputs).detach().cpu()
        observed = torch.from_numpy(
            session.run(["reconstructed"], {"images": inputs.numpy()})[0]
        )
        domains = sorted(
            {str(node.domain) for node in graph.graph.node if str(node.domain)}
        )
        return {
            "succeeded": True,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "maximum_absolute_error": float((expected - observed).abs().max()),
            "finite": bool(torch.isfinite(observed).all()),
            "custom_domains": domains,
            "providers": session.get_providers(),
        }
    except Exception as error:
        return {
            "succeeded": False,
            "path": str(path.resolve()),
            "sha256": _sha256(path) if path.is_file() else None,
            "maximum_absolute_error": 1e9,
            "finite": False,
            "custom_domains": ["export_failed"],
            "providers": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }


def resource_and_numeric_diagnostics(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    rows: Sequence[CleanTrainRow],
    fold_training: Mapping[str, object],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    indices = [
        row.sample_index for row in rows if int(row.fold) == 0
    ][:BATCH_SIZE]
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=0,
        context="jpeg_dl_resource_numeric",
    )
    images_cpu, _, metadata = next(iter(loader))
    images = images_cpu.to(device=device, dtype=torch.float32)
    layers = _load_fold_layers(
        fold_training, mean=mean, std=std, device=device
    )
    candidate = layers[CANDIDATE_ROLE]
    with torch.inference_mode():
        transformed_fp32 = candidate(images)
        probabilities_fp32 = _forward_logits(
            model, transformed_fp32, metadata, device=device, amp=False
        ).softmax(dim=1)
        probabilities_bf16 = _forward_logits(
            model, transformed_fp32, metadata, device=device, amp=True
        ).softmax(dim=1)
        candidate_fp64 = copy.deepcopy(candidate).cpu().double().eval()
        transformed_fp64_cpu = candidate_fp64(images_cpu.double()).float()
        transformed_fp64 = transformed_fp64_cpu.to(device)
        probabilities_fp64 = _forward_logits(
            model, transformed_fp64, metadata, device=device, amp=False
        ).softmax(dim=1)
    fp64_probability_error = float(
        (probabilities_fp32 - probabilities_fp64).abs().max()
    )
    bf16_probability_error = float(
        (probabilities_fp32 - probabilities_bf16).abs().max()
    )
    fp64_transform_error = float(
        (transformed_fp32.detach().cpu() - transformed_fp64_cpu).abs().max()
    )

    def benchmark(candidate_path: bool, iterations: int = 20) -> float:
        for _ in range(5):
            with torch.inference_mode():
                current = candidate(images) if candidate_path else images
                _forward_logits(model, current, metadata, device=device, amp=False)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            with torch.inference_mode():
                current = candidate(images) if candidate_path else images
                _forward_logits(model, current, metadata, device=device, amp=False)
        torch.cuda.synchronize(device)
        return (time.perf_counter() - started) / iterations

    ordinary_seconds = benchmark(False)
    candidate_seconds = benchmark(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _forward_logits(model, images, metadata, device=device, amp=False)
    torch.cuda.synchronize(device)
    ordinary_peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        current = candidate(images)
        _forward_logits(model, current, metadata, device=device, amp=False)
    torch.cuda.synchronize(device)
    candidate_peak = int(torch.cuda.max_memory_allocated(device))
    onnx_result = _onnx_export_diagnostic(
        candidate, images_cpu, output_dir / "jpeg_dl_input_layer.onnx"
    )
    runtime_ratio = candidate_seconds / max(ordinary_seconds, 1e-12)
    additional_peak = max(0, candidate_peak - ordinary_peak)
    checks = {
        "fp32_fp64_probability_error_lte_1e4": fp64_probability_error <= 1e-4,
        "fp32_bf16_probability_error_lte_5e3": bf16_probability_error <= 5e-3,
        "bf16_argmax_agreement_gte_0p995": float(
            (probabilities_fp32.argmax(dim=1) == probabilities_bf16.argmax(dim=1))
            .float()
            .mean()
        )
        >= 0.995,
        "fp64_argmax_exact": torch.equal(
            probabilities_fp32.argmax(dim=1), probabilities_fp64.argmax(dim=1)
        ),
        "onnx_succeeded": bool(onnx_result["succeeded"]),
        "onnx_error_lte_1e5": float(onnx_result["maximum_absolute_error"]) <= 1e-5,
        "onnx_no_custom_domain": not onnx_result["custom_domains"],
        "runtime_ratio_lte_1p45": runtime_ratio <= 1.45,
        "additional_peak_cuda_lte_2gib": additional_peak <= 2 * 2**30,
        "total_peak_cuda_fits_8gib": candidate_peak <= 8 * 2**30,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "fp32_fp64_probability_max_abs_error": fp64_probability_error,
        "fp32_bf16_probability_max_abs_error": bf16_probability_error,
        "fp32_fp64_transform_max_abs_error": fp64_transform_error,
        "bf16_argmax_agreement": float(
            (probabilities_fp32.argmax(dim=1) == probabilities_bf16.argmax(dim=1))
            .float()
            .mean()
        ),
        "ordinary_batch32_seconds": ordinary_seconds,
        "candidate_batch32_seconds": candidate_seconds,
        "runtime_ratio": runtime_ratio,
        "ordinary_peak_cuda_bytes": ordinary_peak,
        "candidate_peak_cuda_bytes": candidate_peak,
        "additional_peak_cuda_bytes": additional_peak,
        "loader": loader_summary,
        "onnx": onnx_result,
    }
    for layer in layers.values():
        layer.cpu()
    del layers, candidate_fp64
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _persist_fold_training(
    output_dir: Path, training: Mapping[str, object]
) -> Tuple[Dict[str, object], Dict[str, object]]:
    fold = int(training["fold"])
    path = output_dir / f"jpeg_q_state_fold_{fold}.pt"
    payload = {
        "schema": "trkh_jpeg_dl_input_quantization_a0_fold_state_v1",
        "fold": fold,
        "states": training["states"],
        "optimizer_state": training["optimizer_state"],
        "occurrence_sha256": training["occurrence_sha256"],
        "final_hashes": training["final_hashes"],
    }
    torch.save(payload, path)
    public = {
        key: value
        for key, value in training.items()
        if key not in {"states", "optimizer_state"}
    }
    artifact = {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    return public, artifact


def _load_persisted_fold(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "trkh_jpeg_dl_input_quantization_a0_fold_state_v1"
    ):
        raise ValueError(f"Invalid JPEG-DL fold state: {path}")
    states = payload.get("states")
    if not isinstance(states, Mapping):
        raise ValueError(f"JPEG-DL fold state lacks role states: {path}")
    observed = {
        role: _state_mapping_sha256(states[role]) for role in TRAINED_ROLES
    }
    if observed != dict(payload["final_hashes"]):
        raise ValueError(f"JPEG-DL fold state hashes differ: {path}")
    return dict(payload)


def _write_manifest(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    path = root / "artifact_manifest.json"
    files = []
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact != path:
            files.append(
                {
                    "path": artifact.relative_to(root).as_posix(),
                    "bytes": int(artifact.stat().st_size),
                    "sha256": _sha256(artifact),
                }
            )
    _write_json(
        path,
        {
            "schema": "trkh_jpeg_dl_input_quantization_a0_manifest_v1",
            "files": files,
        },
    )
    return path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    path = root / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): row for row in payload["files"]}
    observed = {
        artifact.relative_to(root).as_posix(): artifact
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != path
    }
    missing = sorted(set(expected).difference(observed))
    unexpected = sorted(set(observed).difference(expected))
    mismatched = []
    for relative in sorted(set(expected).intersection(observed)):
        artifact = observed[relative]
        row = expected[relative]
        if int(row["bytes"]) != int(artifact.stat().st_size) or str(
            row["sha256"]
        ) != _sha256(artifact):
            mismatched.append(relative)
    return {
        "passed": not missing and not unexpected and not mismatched,
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "files": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def fixed_xai_rows(
    rows: Sequence[CleanTrainRow], executed_folds: Sequence[int]
) -> List[CleanTrainRow]:
    cohort = locked_cohort(rows)
    selected: List[CleanTrainRow] = []
    for fold in executed_folds:
        categories = (
            lambda row: row.target == FOCUS_CLASS
            and row.keeper_prediction == FOCUS_CLASS,
            lambda row: row.target == FOCUS_CLASS
            and row.keeper_prediction != FOCUS_CLASS,
            lambda row: row.target != FOCUS_CLASS
            and row.keeper_prediction == FOCUS_CLASS,
        )
        for predicate in categories:
            candidates = sorted(
                (
                    row
                    for row in cohort
                    if int(row.fold) == int(fold) and predicate(row)
                ),
                key=lambda row: row.sample_index,
            )
            if not candidates:
                raise ValueError(f"JPEG-DL XAI lacks fold {fold} category")
            selected.append(candidates[0])
    return selected


def _normalize_heat(value: Tensor) -> Tensor:
    value = value.detach().float()
    minimum = value.flatten(1).min(dim=1).values.view(-1, 1, 1)
    maximum = value.flatten(1).max(dim=1).values.view(-1, 1, 1)
    return (value - minimum) / (maximum - minimum).clamp_min(1e-9)


def _gradcam_and_input_gradient(
    *,
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    value = images.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
    captured: Dict[str, Tensor] = {}

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError("JPEG-DL XAI stem output must be a tensor")
        output.retain_grad()
        captured["activation"] = output

    handle = model.stem.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logits = _forward_logits(model, value, metadata, device=device, amp=False)
    logits[:, FOCUS_CLASS].sum().backward()
    handle.remove()
    activation = captured.get("activation")
    if activation is None or activation.grad is None or value.grad is None:
        raise RuntimeError("JPEG-DL XAI failed to capture gradients")
    weights = activation.grad.float().mean(dim=(2, 3), keepdim=True)
    gradcam = torch.relu((weights * activation.float()).sum(dim=1))
    gradcam = F.interpolate(
        _normalize_heat(gradcam).unsqueeze(1),
        size=(int(value.shape[-2]), int(value.shape[-1])),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    input_gradient = _normalize_heat(value.grad.detach().float().abs().mean(dim=1))
    image_mask = metadata.get("image_mask")
    padding_leakage = 0.0
    if torch.is_tensor(image_mask):
        valid = image_mask.to(device=device, dtype=torch.bool)
        if valid.ndim == 4:
            valid = valid[:, 0]
        invalid = ~valid
        if invalid.any():
            padding_leakage = max(
                float(gradcam[invalid].abs().max()),
                float(input_gradient[invalid].abs().max()),
            )
        gradcam = gradcam * valid.float()
        input_gradient = input_gradient * valid.float()
    return {
        "gradcam": gradcam.detach().cpu().numpy(),
        "input_gradient_fallback": input_gradient.detach().cpu().numpy(),
        "probabilities": logits.detach().softmax(dim=1).cpu().numpy(),
        "padding_pre_mask_max_abs": padding_leakage,
    }


def _uint8_rgb(value: Tensor) -> np.ndarray:
    return (
        value.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray) -> Image.Image:
    source = np.asarray(rgb, dtype=np.float32)
    values = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    tint = np.stack(
        (
            np.full_like(values, 255.0),
            190.0 * values,
            35.0 * (1.0 - values),
        ),
        axis=-1,
    )
    strength = (0.48 * values)[..., None]
    output = source * (1.0 - strength) + tint * strength
    return Image.fromarray(output.clip(0.0, 255.0).round().astype(np.uint8))


def _signed_difference_image(original: Tensor, candidate: Tensor) -> Image.Image:
    delta = (candidate.detach().float().cpu() - original.detach().float().cpu())[0]
    display = (0.5 + 5.0 * delta).clamp(0.0, 1.0)
    array = (
        display.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    )
    return Image.fromarray(array)


def _absolute_difference_image(original: Tensor, candidate: Tensor) -> Image.Image:
    delta = (candidate.detach().float().cpu() - original.detach().float().cpu())[0]
    values = delta.abs().mean(dim=0)
    values = values / values.max().clamp_min(1e-9)
    array = torch.stack((values, values * 0.6, torch.zeros_like(values)), dim=-1)
    return Image.fromarray(
        array.mul(255.0).round().clamp(0, 255).to(torch.uint8).numpy()
    )


def _fit_panel(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def _render_contact_sheet(
    records: Sequence[Mapping[str, object]], path: Path
) -> Dict[str, object]:
    columns = (
        "original",
        FIXED_ROLE,
        PAPER_ROLE,
        SCALAR_ROLE,
        CANDIDATE_ROLE,
        "signed_delta_x5",
        "absolute_delta",
        "raw_gradcam",
        "candidate_gradcam",
        "raw_input_gradient_fallback",
        "candidate_input_gradient_fallback",
        "permuted_gradcam",
        "phase_gradcam",
    )
    panel = 176
    header = 54
    row_text = 48
    width = len(columns) * panel
    height = header + len(records) * (panel + row_text)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(columns):
        draw.text((column * panel + 4, 8), label, fill="black", font=font)
    for row_index, record in enumerate(records):
        top = header + row_index * (panel + row_text)
        for column, label in enumerate(columns):
            image = record["images"][label]
            canvas.paste(_fit_panel(image, panel), (column * panel, top))
        candidate_probabilities = record["candidate_probabilities"]
        raw_probabilities = record["raw_probabilities"]
        text_value = (
            f"idx={record['sample_index']} fold={record['fold']} target={record['target']} "
            f"raw={int(np.argmax(raw_probabilities))} cand={int(np.argmax(candidate_probabilities))} "
            f"p1={raw_probabilities[1]:.4f}->{candidate_probabilities[1]:.4f} "
            f"psnr={record['psnr']:.2f}"
        )
        draw.text((4, top + panel + 4), text_value, fill="black", font=font)
    canvas.save(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "width": width,
        "height": height,
        "rows": len(records),
        "columns": list(columns),
    }


def _q_color_image(values: np.ndarray, *, signed: bool) -> Image.Image:
    array = np.asarray(values, dtype=np.float64)
    if signed:
        scale = max(1e-9, float(np.abs(array).max()))
        normalized = np.clip(array / scale, -1.0, 1.0)
        red = np.where(normalized >= 0.0, normalized, 0.0)
        blue = np.where(normalized < 0.0, -normalized, 0.0)
        green = 1.0 - np.abs(normalized)
    else:
        minimum = float(array.min())
        maximum = float(array.max())
        normalized = (array - minimum) / max(1e-9, maximum - minimum)
        red = normalized
        blue = 1.0 - normalized
        green = 0.35 + 0.3 * (1.0 - np.abs(2.0 * normalized - 1.0))
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(
        (rgb.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    ).resize((192, 192), Image.Resampling.NEAREST)


def _render_q_heatmaps(
    fold_trainings: Mapping[int, Mapping[str, object]], path: Path
) -> Dict[str, object]:
    panel = 192
    title = 28
    columns = 6
    rows_count = len(fold_trainings)
    canvas = Image.new("RGB", (columns * panel, rows_count * (panel + title)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ("Y initial", "Y final", "Y delta", "CbCr initial", "CbCr final", "CbCr delta")
    for row_index, fold in enumerate(sorted(fold_trainings)):
        state = fold_trainings[fold]["states"][CANDIDATE_ROLE]
        q = np.asarray(state["q"], dtype=np.float64).reshape(2, 8, 8)
        initial = np.ones_like(q)
        images = (
            _q_color_image(initial[0], signed=False),
            _q_color_image(q[0], signed=False),
            _q_color_image(q[0] - 1.0, signed=True),
            _q_color_image(initial[1], signed=False),
            _q_color_image(q[1], signed=False),
            _q_color_image(q[1] - 1.0, signed=True),
        )
        top = row_index * (panel + title)
        for column, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (column * panel, top))
            draw.text(
                (column * panel + 4, top + panel + 4),
                f"fold={fold} {label}",
                fill="black",
                font=font,
            )
    canvas.save(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "width": int(canvas.width),
        "height": int(canvas.height),
        "folds": sorted(fold_trainings),
    }


def xai_audit(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    rows: Sequence[CleanTrainRow],
    fold_trainings: Mapping[int, Mapping[str, object]],
    clean_records: Sequence[Mapping[str, object]],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    executed_folds = sorted(fold_trainings)
    selected = fixed_xai_rows(rows, executed_folds)
    formal_by_index = {
        int(row["sample_index"]): row
        for row in clean_records
        if str(row["condition"]) == "clean"
    }
    model_state_before = _model_state_sha256(model)
    xai_records: List[Dict[str, object]] = []
    repeat_error = 0.0
    padding_after_mask = 0.0
    padding_pre_mask = 0.0
    batch_shape_error = 0.0
    argmax_exact = True
    xai_maps_finite = True
    for selected_row in selected:
        loader, _ = _make_condition_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=[selected_row.sample_index],
            brightness=1.0,
            contrast=1.0,
            batch_size=1,
            num_workers=0,
            context=f"jpeg_dl_xai_{selected_row.sample_index}",
        )
        images_cpu, _, metadata = next(iter(loader))
        images = images_cpu.to(device=device, dtype=torch.float32)
        layers = _load_fold_layers(
            fold_trainings[int(selected_row.fold)],
            mean=mean,
            std=std,
            device=device,
        )
        fixed = JpegDlInputLayer(mean=mean, std=std, trainable=False).to(device).eval()
        candidate = layers[CANDIDATE_ROLE]
        with torch.inference_mode():
            original_rgb = candidate.rgb01_from_normalized(images)
            fixed_input, fixed_rgb = fixed(images, return_rgb=True)
            paper_input, paper_rgb = layers[PAPER_ROLE](images, return_rgb=True)
            scalar_input, scalar_rgb = layers[SCALAR_ROLE](images, return_rgb=True)
            candidate_input, candidate_rgb = candidate(images, return_rgb=True)
            permuted_input, permuted_rgb = candidate(
                images, frequency_permuted=True, return_rgb=True
            )
            phase_input, phase_rgb = candidate(
                images, phase_shifted=True, return_rgb=True
            )
        maps = {
            "raw": _gradcam_and_input_gradient(
                model=model, images=images, metadata=metadata, device=device
            ),
            "candidate": _gradcam_and_input_gradient(
                model=model,
                images=candidate_input,
                metadata=metadata,
                device=device,
            ),
            "permuted": _gradcam_and_input_gradient(
                model=model,
                images=permuted_input,
                metadata=metadata,
                device=device,
            ),
            "phase": _gradcam_and_input_gradient(
                model=model,
                images=phase_input,
                metadata=metadata,
                device=device,
            ),
        }
        xai_maps_finite = xai_maps_finite and all(
            np.isfinite(np.asarray(value[map_name])).all()
            for value in maps.values()
            for map_name in ("gradcam", "input_gradient_fallback", "probabilities")
        )
        if not xai_records:
            repeated = _gradcam_and_input_gradient(
                model=model,
                images=candidate_input,
                metadata=metadata,
                device=device,
            )
            repeat_error = max(
                float(
                    np.max(
                        np.abs(
                            maps["candidate"]["gradcam"] - repeated["gradcam"]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            maps["candidate"]["input_gradient_fallback"]
                            - repeated["input_gradient_fallback"]
                        )
                    )
                ),
            )
        formal = formal_by_index[selected_row.sample_index]
        formal_raw = np.asarray(
            [formal[f"prob_{KEEPER_ROLE}_{index}"] for index in range(NUM_CLASSES)],
            dtype=np.float64,
        )
        formal_candidate = np.asarray(
            [formal[f"prob_{CANDIDATE_ROLE}_{index}"] for index in range(NUM_CLASSES)],
            dtype=np.float64,
        )
        xai_raw = np.asarray(maps["raw"]["probabilities"][0], dtype=np.float64)
        xai_candidate = np.asarray(
            maps["candidate"]["probabilities"][0], dtype=np.float64
        )
        batch_shape_error = max(
            batch_shape_error,
            float(np.max(np.abs(formal_raw - xai_raw))),
            float(np.max(np.abs(formal_candidate - xai_candidate))),
        )
        argmax_exact = argmax_exact and int(np.argmax(formal_raw)) == int(
            np.argmax(xai_raw)
        ) and int(np.argmax(formal_candidate)) == int(np.argmax(xai_candidate))
        padding_pre_mask = max(
            padding_pre_mask,
            *(float(value["padding_pre_mask_max_abs"]) for value in maps.values()),
        )
        mask = metadata.get("image_mask")
        if torch.is_tensor(mask):
            valid = mask.detach().cpu().bool()
            if valid.ndim == 4:
                valid = valid[:, 0]
            invalid = ~valid.numpy()[0]
            if invalid.any():
                for value in maps.values():
                    padding_after_mask = max(
                        padding_after_mask,
                        float(np.max(np.abs(value["gradcam"][0][invalid]))),
                        float(
                            np.max(
                                np.abs(
                                    value["input_gradient_fallback"][0][invalid]
                                )
                            )
                        ),
                    )
        original_array = _uint8_rgb(original_rgb)[0]
        images_payload = {
            "original": Image.fromarray(original_array),
            FIXED_ROLE: Image.fromarray(_uint8_rgb(fixed_rgb)[0]),
            PAPER_ROLE: Image.fromarray(_uint8_rgb(paper_rgb)[0]),
            SCALAR_ROLE: Image.fromarray(_uint8_rgb(scalar_rgb)[0]),
            CANDIDATE_ROLE: Image.fromarray(_uint8_rgb(candidate_rgb)[0]),
            "signed_delta_x5": _signed_difference_image(original_rgb, candidate_rgb),
            "absolute_delta": _absolute_difference_image(original_rgb, candidate_rgb),
            "raw_gradcam": _heat_overlay(original_array, maps["raw"]["gradcam"][0]),
            "candidate_gradcam": _heat_overlay(
                _uint8_rgb(candidate_rgb)[0], maps["candidate"]["gradcam"][0]
            ),
            "raw_input_gradient_fallback": _heat_overlay(
                original_array, maps["raw"]["input_gradient_fallback"][0]
            ),
            "candidate_input_gradient_fallback": _heat_overlay(
                _uint8_rgb(candidate_rgb)[0],
                maps["candidate"]["input_gradient_fallback"][0],
            ),
            "permuted_gradcam": _heat_overlay(
                _uint8_rgb(permuted_rgb)[0], maps["permuted"]["gradcam"][0]
            ),
            "phase_gradcam": _heat_overlay(
                _uint8_rgb(phase_rgb)[0], maps["phase"]["gradcam"][0]
            ),
        }
        reconstruction = _reconstruction_rows(
            original_rgb,
            candidate_rgb,
            mask if torch.is_tensor(mask) else None,
        )[0]
        xai_records.append(
            {
                "sample_index": selected_row.sample_index,
                "fold": selected_row.fold,
                "target": selected_row.target,
                "source_stem": selected_row.source_stem,
                "raw_probabilities": xai_raw.tolist(),
                "candidate_probabilities": xai_candidate.tolist(),
                "psnr": reconstruction["psnr"],
                "images": images_payload,
            }
        )
        for layer in layers.values():
            layer.cpu()
        fixed.cpu()
        del layers, fixed, fixed_input, paper_input, scalar_input
        gc.collect()
        torch.cuda.empty_cache()
    contact = _render_contact_sheet(
        xai_records, output_dir / "jpeg_dl_xai_contact_sheet.png"
    )
    heatmaps = _render_q_heatmaps(
        fold_trainings, output_dir / "jpeg_dl_q_heatmaps.png"
    )
    model_state_after = _model_state_sha256(model)
    selected_indices = [row.sample_index for row in selected]
    checks = {
        "selection_exact": selected_indices
        == [row.sample_index for row in fixed_xai_rows(rows, executed_folds)],
        "three_rows_per_fold": len(selected_indices) == 3 * len(executed_folds),
        "maps_finite": xai_maps_finite,
        "repeat_error_zero": repeat_error == 0.0,
        "padding_after_mask_zero": padding_after_mask == 0.0,
        "batch_shape_probability_error_lte_3e3": batch_shape_error
        <= MAX_XAI_BATCH_SHAPE_PROBABILITY_ERROR,
        "argmax_exact": argmax_exact,
        "keeper_state_exact": model_state_before == model_state_after,
    }
    serializable_records = [
        {key: value for key, value in record.items() if key != "images"}
        for record in xai_records
    ]
    result = {
        "passed_automatic": all(checks.values()),
        "checks": checks,
        "manual_review": "pending",
        "selected_rows": serializable_records,
        "selected_index_sha256": _index_sha256(selected_indices),
        "gradient_repeat_max_abs_error": repeat_error,
        "padding_pre_mask_max_abs": padding_pre_mask,
        "padding_after_mask_max_abs": padding_after_mask,
        "batch_shape_probability_max_abs_error": batch_shape_error,
        "contact_sheet": contact,
        "q_heatmaps": heatmaps,
        "fallback_label": "input-gradient fallback; not native attention",
    }
    _write_json(output_dir / "jpeg_dl_xai_summary.json", result)
    return result


def _current_command_hashes() -> Dict[str, str]:
    return {
        "current_commands": _sha256(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ),
        "command_history": _sha256(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ),
    }


def _public_training(training: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in training.items()
        if key not in {"states", "optimizer_state"}
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_JPEG_DL_A0_PREFLIGHT") != "passed":
        raise RuntimeError("Formal JPEG-DL A0 must use the locked PowerShell preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal JPEG-DL A0")
    preflight_result = preflight(args)
    if not bool(preflight_result["passed"]):
        raise RuntimeError("JPEG-DL preflight failed before formal execution")
    repo_state = _repo_state()
    implementation = implementation_tracking_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal JPEG-DL A0 requires clean pushed tracked state: {repo_state}"
        )
    if not bool(implementation["passed"]):
        raise ValueError(
            f"Formal JPEG-DL A0 requires committed implementation: {implementation}"
        )
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    gpu_before = _gpu_snapshot()
    dataset_before = _dataset_stat_snapshot(args.data.parent)
    command_hashes_before = _current_command_hashes()
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    keeper_state_before = _model_state_sha256(model)
    rows = locked_training_rows(_read_clean_train_rows(args.cidt_predictions))
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, args.data)
    semantics = dataset_summary["semantics"]
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    keeper_compatibility = keeper_compatibility_audit(
        model=model,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        device=device,
        num_workers=NUM_WORKERS,
    )
    if not bool(keeper_compatibility["passed"]):
        raise RuntimeError(
            "JPEG-DL keeper compatibility failed before q fitting or candidate metrics: "
            f"{keeper_compatibility}"
        )
    output_dir = _prepare_output_dir(args.output_dir)

    fold_trainings: Dict[int, Dict[str, object]] = {}
    public_trainings: Dict[str, object] = {}
    state_artifacts: Dict[str, object] = {}
    clean_records: List[Dict[str, object]] = []
    evaluation_summaries: Dict[str, object] = {}

    fold_zero = train_fold(
        model=model,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        fold=0,
        mean=mean,
        std=std,
        device=device,
        num_workers=NUM_WORKERS,
    )
    fold_trainings[0] = fold_zero
    public, artifact = _persist_fold_training(output_dir, fold_zero)
    public_trainings["0"] = public
    state_artifacts["0"] = artifact
    fold_zero_eval = evaluate_fold(
        model=model,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        fold_training=fold_zero,
        fold=0,
        condition="clean",
        brightness=1.0,
        contrast=1.0,
        mean=mean,
        std=std,
        device=device,
        num_workers=NUM_WORKERS,
    )
    clean_records.extend(fold_zero_eval["records"])
    evaluation_summaries["clean_fold_0"] = {
        key: value for key, value in fold_zero_eval.items() if key != "records"
    }
    stage_analysis = analyze_records(clean_records)
    stage_reconstruction = reconstruction_summary(clean_records)
    stage_quantizers = q_diagnostics(fold_trainings)
    resources = resource_and_numeric_diagnostics(
        model=model,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        fold_training=fold_zero,
        mean=mean,
        std=std,
        device=device,
        output_dir=output_dir,
    )
    stage_gates = stage_a_gates(
        stage_analysis, stage_reconstruction, stage_quantizers, fold_zero
    )
    stage_structural_gates = {
        "preflight_pass": bool(preflight_result["passed"]),
        "fold_zero_evaluation_pass": bool(fold_zero_eval["passed"]),
        "resource_numeric_pass": bool(resources["passed"]),
        "pre_candidate_keeper_compatibility_pass": bool(
            keeper_compatibility["passed"]
        ),
        "keeper_state_exact_after_stage_a": keeper_state_before
        == _model_state_sha256(model),
    }
    stage_a_continued = all(stage_gates.values()) and all(
        stage_structural_gates.values()
    )

    if stage_a_continued:
        for fold in FOLDS[1:]:
            training = train_fold(
                model=model,
                base_dataset=dataset,
                transform=transform,
                rows=rows,
                fold=fold,
                mean=mean,
                std=std,
                device=device,
                num_workers=NUM_WORKERS,
            )
            fold_trainings[fold] = training
            public, artifact = _persist_fold_training(output_dir, training)
            public_trainings[str(fold)] = public
            state_artifacts[str(fold)] = artifact
            evaluation = evaluate_fold(
                model=model,
                base_dataset=dataset,
                transform=transform,
                rows=rows,
                fold_training=training,
                fold=fold,
                condition="clean",
                brightness=1.0,
                contrast=1.0,
                mean=mean,
                std=std,
                device=device,
                num_workers=NUM_WORKERS,
            )
            clean_records.extend(evaluation["records"])
            evaluation_summaries[f"clean_fold_{fold}"] = {
                key: value for key, value in evaluation.items() if key != "records"
            }

    clean_analysis = analyze_records(clean_records)
    clean_reconstruction = reconstruction_summary(clean_records)
    quantizer_analysis = q_diagnostics(fold_trainings)
    full_clean_gates: Dict[str, bool] = {}
    if stage_a_continued and len(fold_trainings) == len(FOLDS):
        full_clean_gates = clean_oof_gates(
            clean_analysis, clean_reconstruction, quantizer_analysis
        )
    clean_automated_pass = bool(full_clean_gates) and all(full_clean_gates.values())
    clean_structural_pass = (
        all(bool(value["passed"]) for value in evaluation_summaries.values())
        and all(
            bool(training["all_gradient_roles_live"])
            for training in fold_trainings.values()
        )
        and bool(resources["passed"])
        and bool(quantizer_analysis["passed"])
        and bool(clean_reconstruction["passed"])
    )

    all_records = list(clean_records)
    shifted_evaluation_summaries: Dict[str, object] = {}
    robustness_opened = clean_automated_pass and clean_structural_pass
    if robustness_opened:
        for condition, brightness, contrast in CONDITIONS:
            for fold in FOLDS:
                evaluation = evaluate_fold(
                    model=model,
                    base_dataset=dataset,
                    transform=transform,
                    rows=rows,
                    fold_training=fold_trainings[fold],
                    fold=fold,
                    condition=condition,
                    brightness=brightness,
                    contrast=contrast,
                    mean=mean,
                    std=std,
                    device=device,
                    num_workers=NUM_WORKERS,
                    cohort_only=True,
                )
                all_records.extend(evaluation["records"])
                shifted_evaluation_summaries[f"{condition}_fold_{fold}"] = {
                    key: value for key, value in evaluation.items() if key != "records"
                }

    full_analysis = analyze_records(all_records)
    robustness_gates: Dict[str, bool] = {}
    robustness_reconstruction: Dict[str, object] = {}
    if robustness_opened:
        robustness_gates = shifted_condition_gates(full_analysis)
        for condition, _, _ in CONDITIONS:
            selected = [
                row for row in all_records if str(row["condition"]) == condition
            ]
            robustness_reconstruction[condition] = reconstruction_summary(selected)
            robustness_gates[f"{condition}_reconstruction_anti_collapse"] = bool(
                robustness_reconstruction[condition]["passed"]
            )

    xai = xai_audit(
        model=model,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        fold_trainings=fold_trainings,
        clean_records=clean_records,
        mean=mean,
        std=std,
        device=device,
        output_dir=output_dir,
    )
    predictions_path = output_dir / "predictions_all_opened_conditions.csv"
    _write_csv(predictions_path, all_records)
    replayed_records = _read_prediction_csv(predictions_path)
    replayed_analysis = analyze_records(replayed_records)
    analysis_roundtrip_exact = _canonical_sha256(full_analysis) == _canonical_sha256(
        replayed_analysis
    )
    full_analysis = replayed_analysis

    dataset_after = _dataset_stat_snapshot(args.data.parent)
    command_hashes_after = _current_command_hashes()
    keeper_state_after = _model_state_sha256(model)
    gpu_after = _gpu_snapshot()
    repo_state_after = _repo_state()
    implementation_after = implementation_tracking_state()
    structural_gates = {
        "repo_tracked_clean_and_pushed": bool(repo_state_after["tracked_worktree_clean"])
        and bool(repo_state_after["head_matches_upstream"]),
        "implementation_tracked_clean": bool(implementation_after["passed"]),
        "dataset_stat_snapshot_exact": dataset_before == dataset_after,
        "current_commands_unchanged": command_hashes_before == command_hashes_after
        and command_hashes_after["current_commands"] == LOCKED_CURRENT_COMMAND_SHA256
        and command_hashes_after["command_history"] == LOCKED_COMMAND_HISTORY_SHA256,
        "keeper_state_exact": keeper_state_before == keeper_state_after,
        "all_fold_gradients_live": all(
            bool(training["all_gradient_roles_live"])
            for training in fold_trainings.values()
        ),
        "all_opened_evaluations_pass": all(
            bool(value["passed"])
            for value in (
                list(evaluation_summaries.values())
                + list(shifted_evaluation_summaries.values())
            )
        ),
        "resource_numeric_pass": bool(resources["passed"]),
        "pre_candidate_keeper_compatibility_pass": bool(
            keeper_compatibility["passed"]
        ),
        "analysis_csv_roundtrip_exact": analysis_roundtrip_exact,
        "xai_automatic_pass": bool(xai["passed_automatic"]),
        "state_artifacts_exist": all(Path(value["path"]).is_file() for value in state_artifacts.values()),
    }
    automated_pass = bool(
        stage_a_continued
        and len(fold_trainings) == len(FOLDS)
        and full_clean_gates
        and all(full_clean_gates.values())
        and robustness_opened
        and robustness_gates
        and all(robustness_gates.values())
        and all(structural_gates.values())
    )
    status = (
        "pending_manual_visual_review"
        if automated_pass
        else (
            "rejected_stage_a_pending_manual_visual_review"
            if not stage_a_continued
            else "rejected_automated_gate_pending_manual_visual_review"
        )
    )
    summary_path = output_dir / "summary.json"
    summary = {
        "schema": "trkh_jpeg_dl_input_quantization_a0_summary_v1",
        "method": METHOD,
        "mode": MODE,
        "status": status,
        "automated_pass": automated_pass,
        "manual_visual_review": "pending",
        "validation_data_used": False,
        "test_data_used": False,
        "raw_dataset_modified": False,
        "production_model_modified": False,
        "current_best_command_updated": False,
        "class_names": class_names,
        "executed_folds": sorted(fold_trainings),
        "stage_a_continued": stage_a_continued,
        "robustness_opened": robustness_opened,
        "stage_a_gates": stage_gates,
        "stage_a_structural_gates": stage_structural_gates,
        "clean_oof_gates": full_clean_gates,
        "robustness_gates": robustness_gates,
        "structural_gates": structural_gates,
        "analysis": full_analysis,
        "analysis_sha256": _canonical_sha256(full_analysis),
        "stage_a_analysis": stage_analysis,
        "clean_reconstruction": clean_reconstruction,
        "robustness_reconstruction": robustness_reconstruction,
        "quantizers": quantizer_analysis,
        "resources_and_numeric": resources,
        "pre_candidate_keeper_compatibility": keeper_compatibility,
        "xai": xai,
        "training": public_trainings,
        "evaluation": {
            **evaluation_summaries,
            **shifted_evaluation_summaries,
        },
        "provenance": preflight_result["provenance"],
        "equations": preflight_result["equations"],
        "dataset": dataset_summary,
        "dataset_before": dataset_before,
        "dataset_after": dataset_after,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "repo_state": repo_state_after,
        "implementation": implementation_after,
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "fold_states": state_artifacts,
            "manifest": str((output_dir / "artifact_manifest.json").resolve()),
            "xai_contact_sheet": xai["contact_sheet"]["path"],
            "q_heatmaps": xai["q_heatmaps"]["path"],
            "onnx": resources["onnx"]["path"],
        },
        "decision": (
            "Authorize one production integration only after manual visual pass."
            if automated_pass
            else "Reject this locked JPEG-DL A0; no trainer integration or full train."
        ),
    }
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    print(json.dumps(to_serializable(summary), indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(summary_path)}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    return summary


def replay_summary(summary_path: Path, *, write: bool = True) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = summary_path.parent
    manifest = _verify_manifest(output_dir)
    predictions_path = Path(summary["artifacts"]["predictions"])
    predictions_hash_exact = _sha256(predictions_path) == str(
        summary["artifacts"]["predictions_sha256"]
    )
    records = _read_prediction_csv(predictions_path)
    analysis = analyze_records(records)
    analysis_hash = _canonical_sha256(analysis)
    analysis_exact = analysis_hash == str(summary["analysis_sha256"])
    fold_trainings: Dict[int, Dict[str, object]] = {}
    state_checks = {}
    for fold_text, artifact in summary["artifacts"]["fold_states"].items():
        path = Path(artifact["path"])
        hash_exact = _sha256(path) == str(artifact["sha256"])
        payload = _load_persisted_fold(path)
        fold = int(fold_text)
        fold_trainings[fold] = {"states": payload["states"]}
        state_checks[fold_text] = {
            "hash_exact": hash_exact,
            "state_hashes_exact": {
                role: _state_mapping_sha256(payload["states"][role])
                == str(payload["final_hashes"][role])
                for role in TRAINED_ROLES
            },
        }
    quantizers = q_diagnostics(fold_trainings)
    clean_records = [row for row in records if str(row["condition"]) == "clean"]
    reconstruction = reconstruction_summary(clean_records)
    if bool(summary["stage_a_continued"]) and len(fold_trainings) == len(FOLDS):
        clean_gates = clean_oof_gates(analysis, reconstruction, quantizers)
    else:
        stage_training = summary["training"]["0"]
        clean_gates = stage_a_gates(
            analyze_records(clean_records), reconstruction, quantizers, stage_training
        )
    expected_gates = (
        summary["clean_oof_gates"]
        if bool(summary["stage_a_continued"])
        else summary["stage_a_gates"]
    )
    gate_replay_exact = clean_gates == expected_gates
    robustness_gates = {}
    if bool(summary["robustness_opened"]):
        robustness_gates = shifted_condition_gates(analysis)
        for condition, _, _ in CONDITIONS:
            condition_records = [
                row for row in records if str(row["condition"]) == condition
            ]
            robustness_gates[f"{condition}_reconstruction_anti_collapse"] = bool(
                reconstruction_summary(condition_records)["passed"]
            )
    robustness_exact = robustness_gates == summary["robustness_gates"]
    checks = {
        "manifest_exact_before_replay_write": bool(manifest["passed"]),
        "predictions_hash_exact": predictions_hash_exact,
        "analysis_exact": analysis_exact,
        "fold_state_files_exact": all(
            value["hash_exact"] and all(value["state_hashes_exact"].values())
            for value in state_checks.values()
        ),
        "quantizer_analysis_exact": _canonical_sha256(quantizers)
        == _canonical_sha256(summary["quantizers"]),
        "clean_reconstruction_exact": _canonical_sha256(reconstruction)
        == _canonical_sha256(summary["clean_reconstruction"]),
        "clean_gate_replay_exact": gate_replay_exact,
        "robustness_gate_replay_exact": robustness_exact,
    }
    result = {
        "schema": "trkh_jpeg_dl_input_quantization_a0_replay_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "analysis_sha256": analysis_hash,
        "state_checks": state_checks,
        "manifest_before_write": manifest,
        "summary_sha256_before_write": _sha256(summary_path),
    }
    if write:
        replay_path = output_dir / "replay.json"
        _write_json(replay_path, result)
        summary["replay"] = {
            "path": str(replay_path.resolve()),
            "sha256": _sha256(replay_path),
            "passed": bool(result["passed"]),
        }
        _write_json(summary_path, summary)
        manifest_path = _write_manifest(output_dir)
        result["summary_sha256_after_write"] = _sha256(summary_path)
        result["manifest_sha256_after_write"] = _sha256(manifest_path)
    return result


def finalize_visual_review(
    summary_path: Path,
    *,
    decision: str,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    observed = _sha256(summary_path)
    if observed != str(expected_summary_sha256).strip().casefold():
        raise ValueError(
            f"JPEG-DL visual finalization summary hash differs: {observed}"
        )
    replay = replay_summary(summary_path, write=False)
    if not bool(replay["passed"]):
        raise RuntimeError("JPEG-DL replay failed before visual finalization")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = str(decision).casefold()
    summary["manual_visual_review"] = decision
    summary["xai"]["manual_review"] = decision
    if bool(summary["automated_pass"]) and decision == "pass":
        summary["status"] = "passed_all_gates"
        summary["decision"] = (
            "Authorize one production JPEG-DL integration, then smoke/probe gates."
        )
    elif bool(summary["automated_pass"]):
        summary["status"] = "rejected_manual_visual_gate"
        summary["decision"] = "Reject JPEG-DL A0 after manual visual failure."
    else:
        summary["status"] = "rejected_automated_gate_visual_review_recorded"
        summary["decision"] = (
            "Reject this locked JPEG-DL A0; no trainer integration or full train."
        )
    summary["visual_finalization"] = {
        "decision": decision,
        "input_summary_sha256": observed,
        "replay_passed": True,
        "scientific_metrics_changed": False,
    }
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(summary_path.parent)
    result = {
        "status": summary["status"],
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.finalize_visual_review:
        if args.replay_summary is None or not args.expected_summary_sha256:
            raise ValueError(
                "Visual finalization requires --replay-summary and --expected-summary-sha256"
            )
        result = finalize_visual_review(
            args.replay_summary,
            decision=args.finalize_visual_review,
            expected_summary_sha256=args.expected_summary_sha256,
        )
    elif args.replay_summary is not None:
        result = replay_summary(args.replay_summary, write=True)
    elif args.preflight_only:
        result = preflight(args)
    elif args.engineering_forward:
        result = engineering_forward(args)
    else:
        result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True))
    return 0 if bool(result.get("passed", result.get("passed_automatic", True))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
