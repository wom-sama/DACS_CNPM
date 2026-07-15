from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.config import load_data_spec
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    ConvStemBlock,
    HybridConvStem,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_class1_boundary_cagrad_readiness import (
    CONDITIONS,
    _read_prior_predictions,
)
from trkh.tools.audit_class1_reference_agem_readiness import (
    EXPECTED_HARD_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_REFERENCE_ROWS,
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    _cohort_serializable,
    _locked_cohort_summary,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _comparison,
    _forward_logits,
    _git_commit,
    _make_loader,
    _metadata_to_device,
    _prepare_output_dir,
    _rng_snapshot,
    _rng_summary,
    _sha256,
    _tensor_sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "ceconv_stem_residual_a0"
ROTATIONS = 3
BATCH_SIZE = 32
HALF_BATCH = 16
TRAIN_STEPS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
SEED = 42
FOLD = 0
EXPECTED_TRAINABLE_PARAMETERS = 960
MAX_GATE_ABS = 0.25
MAX_RUNTIME_RATIO = 1.35
MAX_PEAK_VRAM_GIB = 3.25
MAX_ONNX_ERROR = 1e-5
STATIC_EXPORT_BATCH_SIZE = 1

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_CAGRAD_SUMMARY_SHA256 = "dab24c6dd7dcf873e523915aadbdd1f8ef5337801278d9ffc55ce8c867a37750"
LOCKED_CAGRAD_PREDICTIONS_SHA256 = "92dc296345e34d3f0ac026db2dcb2f982e40af79e1179136dfbb17f1cc2b9d9f"
LOCKED_CAGRAD_MANIFEST_SHA256 = "7f1f2a18846cd54fb7a7eafe2964a17681b99d009a70aadfaa09be54d43d1ced"
LOCKED_PROTOCOL_SHA256 = "9c5af4e9886ec2068f5e88899f25f54c9b06e870538b31f6158cce370da57082"
LOCKED_PAPER_SHA256 = "1c41da4c99ffc46ef0c318e7145ab749efb14b6ef63927bba9a8071937ad8f83"
LOCKED_OFFICIAL_COMMIT = "8f46c78c3a7cf91ad905d0255b756d13cd1e0c94"
LOCKED_CECONV_SOURCE_SHA256 = "c9a730691e3186179b64b40f966dbd60d21add3040b0911786408cefc75e6469"
LOCKED_POOLING_SOURCE_SHA256 = "882425da67ed9700f8bd2c12d2e708e8d8ae857a816f39cebae389d68ab9351c"
LOCKED_LICENSE_SHA256 = "4d846d51cf0de71d223d0ba9ef4a2ddf41088e0b0eb92af4a6d17ca98cc3b6df"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only CEConv first-stem residual gate. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--cagrad-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_cagrad_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--cagrad-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--cagrad-manifest",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CECONV_STEM_RESIDUAL_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\ceconv_neurips2023.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\ceconv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_ceconv_stem_residual_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--focus-class", type=int, default=FOCUS_CLASS)
    parser.add_argument("--rotations", type=int, default=ROTATIONS)
    parser.add_argument("--max-train-batches", type=int, default=TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == 4
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and int(args.focus_class) == FOCUS_CLASS
        and int(args.rotations) == ROTATIONS
        and int(args.max_train_batches) == TRAIN_STEPS
        and math.isclose(float(args.learning_rate), LEARNING_RATE, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, rel_tol=0.0, abs_tol=0.0)
        and int(args.benchmark_repeats) == 3
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO, rel_tol=0.0, abs_tol=0.0)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB, rel_tol=0.0, abs_tol=0.0)
        and int(args.xai_batch_size) == 4
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "cagrad_summary": Path(args.cagrad_summary).resolve(),
        "cagrad_predictions": Path(args.cagrad_predictions).resolve(),
        "cagrad_manifest": Path(args.cagrad_manifest).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "ceconv_source": official / "ceconv" / "ceconv2d.py",
        "pooling_source": official / "ceconv" / "pooling.py",
        "license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _tracked_worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(Path(root).resolve()),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _full_worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def hue_rotation_matrix(rotations: int = ROTATIONS) -> Tensor:
    if int(rotations) <= 0:
        raise ValueError("Number of hue rotations must be positive.")
    cosine = math.cos(2.0 * math.pi / int(rotations))
    sine = math.sin(2.0 * math.pi / int(rotations))
    const_a = (1.0 - cosine) / 3.0
    const_b = math.sqrt(1.0 / 3.0) * sine
    return torch.tensor(
        [
            [cosine + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cosine + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cosine + const_a],
        ],
        dtype=torch.float32,
    )


def hue_rotation_powers(
    rotations: int = ROTATIONS,
    *,
    identity_control: bool = False,
) -> Tensor:
    matrix = hue_rotation_matrix(rotations)
    if identity_control:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(int(rotations), 1, 1)
    return torch.stack(
        [torch.matrix_power(matrix, index) for index in range(int(rotations))],
        dim=0,
    )


def transform_input_filter(weights: Tensor, rotation_powers: Tensor) -> Tensor:
    if weights.ndim != 5 or int(weights.size(1)) != 3 or int(weights.size(2)) != 1:
        raise ValueError("CEConv input weights must be [Cout,3,1,K,K].")
    if rotation_powers.ndim != 3 or tuple(rotation_powers.shape[1:]) != (3, 3):
        raise ValueError("Rotation powers must be [G,3,3].")
    flat = weights.permute(2, 1, 0, 3, 4)
    flat_shape = flat.shape
    flat = flat.reshape(1, 3, -1)
    transformed = torch.matmul(rotation_powers.to(weights), flat)
    transformed = transformed.view((int(rotation_powers.size(0)),) + flat_shape)
    return transformed.permute(3, 0, 2, 1, 4, 5).contiguous()


def official_equation_replay(weights: Tensor, rotations: int = ROTATIONS) -> Tensor:
    matrix = hue_rotation_matrix(rotations).to(weights)
    powers = torch.stack(
        [torch.matrix_power(matrix, index) for index in range(int(rotations))], dim=0
    )
    flat = weights.permute(2, 1, 0, 3, 4)
    flat_shape = flat.shape
    transformed = torch.matmul(powers, flat.reshape(1, 3, -1))
    transformed = transformed.view((int(rotations),) + flat_shape)
    return transformed.permute(3, 0, 2, 1, 4, 5).contiguous()


class CEConvStemResidual(nn.Module):
    def __init__(
        self,
        native: ConvStemBlock,
        *,
        rotations: int = ROTATIONS,
        identity_control: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(native, ConvStemBlock):
            raise TypeError("CEConv residual requires a ConvStemBlock.")
        conv = native.block.conv
        if not isinstance(conv, nn.Conv2d) or int(conv.in_channels) != 3:
            raise TypeError("CEConv residual requires the first RGB Conv2d.")
        if conv.bias is not None or int(conv.groups) != 1:
            raise ValueError("Locked first stem convolution must be bias-free and dense.")
        self.native = copy.deepcopy(native)
        self.rotations = int(rotations)
        self.identity_control = bool(identity_control)
        self.branch_weight = nn.Parameter(conv.weight.detach().clone().unsqueeze(2))
        self.branch_norm = nn.BatchNorm3d(int(conv.out_channels))
        self.branch_act = nn.GELU()
        self.branch_pool = copy.deepcopy(native.block.pool)
        self.gate = nn.Parameter(torch.zeros(int(conv.out_channels)))
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.register_buffer(
            "rotation_powers",
            hue_rotation_powers(
                self.rotations,
                identity_control=self.identity_control,
            ),
            persistent=True,
        )

    def transformed_filters(self) -> Tensor:
        return transform_input_filter(self.branch_weight, self.rotation_powers)

    def pre_norm_groups(self, x: Tensor) -> Tensor:
        transformed = self.transformed_filters()
        out_channels, rotations, in_channels, singleton, kh, kw = transformed.shape
        if singleton != 1:
            raise RuntimeError("CEConv transformed singleton axis differs.")
        kernels = transformed.reshape(out_channels * rotations, in_channels, kh, kw)
        value = F.conv2d(
            x,
            weight=kernels,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
        )
        return value.reshape(
            int(x.size(0)),
            out_channels,
            rotations,
            int(value.size(-2)),
            int(value.size(-1)),
        )

    def branch_groups(self, x: Tensor) -> Tensor:
        groups = self.branch_act(self.branch_norm(self.pre_norm_groups(x)))
        batch, channels, rotations, height, width = groups.shape
        pooled = self.branch_pool(groups.reshape(batch, channels * rotations, height, width))
        return pooled.reshape(
            batch,
            channels,
            rotations,
            int(pooled.size(-2)),
            int(pooled.size(-1)),
        )

    def forward_with_trace(self, x: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        native = self.native(x)
        groups = self.branch_groups(x)
        pooled, winner = groups.max(dim=2)
        gate = torch.tanh(self.gate).view(1, -1, 1, 1)
        output = native + gate * pooled
        return output, {
            "native": native,
            "groups": groups,
            "pooled": pooled,
            "winner": winner,
            "gate": gate,
        }

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.forward_with_trace(x)
        return output


def _adapter(model: nn.Module) -> CEConvStemResidual:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem):
        raise TypeError("Keeper does not expose the locked HybridConvStem.")
    module = stem.blocks[0]
    if not isinstance(module, CEConvStemResidual):
        raise TypeError("Model does not contain CEConvStemResidual at stem block 0.")
    return module


def _wrap_model(prototype: nn.Module, *, identity_control: bool) -> nn.Module:
    model = copy.deepcopy(prototype)
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem) or not isinstance(stem.blocks[0], ConvStemBlock):
        raise TypeError("Locked keeper stem schema differs.")
    stem.blocks[0] = CEConvStemResidual(
        stem.blocks[0], rotations=ROTATIONS, identity_control=identity_control
    )
    return model.eval()


def _trainable_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if name
        in {
            "stem.blocks.0.branch_weight",
            "stem.blocks.0.branch_norm.weight",
            "stem.blocks.0.branch_norm.bias",
            "stem.blocks.0.gate",
        }
    }


def _frozen_state_sha256(model: nn.Module) -> str:
    excluded = (
        "stem.blocks.0.branch_weight",
        "stem.blocks.0.branch_norm.",
        "stem.blocks.0.gate",
    )
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name == excluded[0] or name == excluded[2] or name.startswith(excluded[1]):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _configure_trainability(model: nn.Module) -> list[str]:
    allowed = {
        "stem.blocks.0.branch_weight",
        "stem.blocks.0.branch_norm.weight",
        "stem.blocks.0.branch_norm.bias",
        "stem.blocks.0.gate",
    }
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if parameter.requires_grad:
            trainable.append(name)
    if set(trainable) != allowed:
        raise ValueError(f"CEConv trainable schema differs: {trainable}")
    model.eval()
    _adapter(model).branch_norm.train()
    return trainable


def balanced_boundary_order(
    reference_indices: Sequence[int],
    hard_indices: Sequence[int],
    *,
    steps: int = TRAIN_STEPS,
    half_batch: int = HALF_BATCH,
    seed: int = SEED,
) -> list[int]:
    positive = np.asarray([int(value) for value in reference_indices], dtype=np.int64)
    negative = np.asarray([int(value) for value in hard_indices], dtype=np.int64)
    if positive.size != EXPECTED_REFERENCE_ROWS or negative.size != EXPECTED_HARD_ROWS:
        raise ValueError("Locked class1/hard-negative cohort sizes differ.")
    positive_rng = np.random.default_rng(int(seed) + 11)
    negative_rng = np.random.default_rng(int(seed) + 29)
    batch_rng = np.random.default_rng(int(seed) + 47)
    positive_pool = positive_rng.permutation(positive)
    negative_pool = negative_rng.permutation(negative)
    positive_at = 0
    negative_at = 0

    def take(
        pool: np.ndarray,
        cursor: int,
        source: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        values = []
        remaining = int(half_batch)
        while remaining > 0:
            available = int(pool.size - cursor)
            count = min(remaining, available)
            values.extend(pool[cursor : cursor + count].tolist())
            cursor += count
            remaining -= count
            if cursor == int(pool.size):
                pool = rng.permutation(source)
                cursor = 0
        return np.asarray(values, dtype=np.int64), cursor, pool

    order: list[int] = []
    for _ in range(int(steps)):
        positives, positive_at, positive_pool = take(
            positive_pool, positive_at, positive, positive_rng
        )
        negatives, negative_at, negative_pool = take(
            negative_pool, negative_at, negative, negative_rng
        )
        batch = np.concatenate((positives, negatives))
        batch = batch[batch_rng.permutation(batch.size)]
        order.extend(int(value) for value in batch.tolist())
    return order


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, object],
    list[CleanTrainRow],
    Dict[str, object],
    Dict[str, Dict[str, list[Dict[str, object]]]],
]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted CEConv protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "cagrad_summary": _verify_sha256(
            paths["cagrad_summary"], LOCKED_CAGRAD_SUMMARY_SHA256, "CAGrad summary"
        ),
        "cagrad_predictions": _verify_sha256(
            paths["cagrad_predictions"],
            LOCKED_CAGRAD_PREDICTIONS_SHA256,
            "CAGrad predictions",
        ),
        "cagrad_manifest": _verify_sha256(
            paths["cagrad_manifest"], LOCKED_CAGRAD_MANIFEST_SHA256, "CAGrad manifest"
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "CEConv protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "CEConv paper"),
        "ceconv_source": _verify_sha256(
            paths["ceconv_source"], LOCKED_CECONV_SOURCE_SHA256, "official CEConv source"
        ),
        "pooling_source": _verify_sha256(
            paths["pooling_source"],
            LOCKED_POOLING_SOURCE_SHA256,
            "official pooling source",
        ),
        "license": _verify_sha256(paths["license"], LOCKED_LICENSE_SHA256, "MIT license"),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best command file",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    official_commit = _git_commit(paths["official_root"])
    official_clean = _full_worktree_clean(paths["official_root"])
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official CEConv commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if not official_clean:
        raise ValueError("Official CEConv worktree is not clean.")
    for name in ("cidt_summary", "cagrad_summary"):
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        if bool(payload.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(
            payload.get(
                "validation_predictions_used",
                payload.get("validation_data_used", True),
            )
        ):
            raise ValueError(f"{name} provenance indicates validation use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    prior, argmax_mismatches = _read_prior_predictions(
        paths["cagrad_predictions"],
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
    )
    repo_root = Path.cwd().resolve()
    tracked_clean = _tracked_worktree_clean(repo_root)
    if not tracked_clean:
        raise ValueError("Tracked TRKH worktree must be clean for the formal CEConv gate.")
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_worktree_clean": official_clean,
            "repository_commit": _git_commit(repo_root),
            "tracked_worktree_clean": tracked_clean,
            "prior_argmax_mismatches": argmax_mismatches,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
        prior,
    )


def _construct_models(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, nn.Module, nn.Module, Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if not isinstance(class_names, list) or len(class_names) != 5:
        raise ValueError("Keeper class order is invalid.")
    set_seed(SEED, deterministic=True)
    raw = create_model(num_classes=5, model_config=model_config).eval()
    load_model_state(raw, dict(model_state), strict=True)
    control = _wrap_model(raw, identity_control=True)
    candidate = _wrap_model(raw, identity_control=False)
    control_trainable = _trainable_state(control)
    candidate_trainable = _trainable_state(candidate)
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    differing = [
        name
        for name in control_state
        if name not in candidate_state
        or not torch.equal(control_state[name], candidate_state[name])
    ]
    return raw, control, candidate, {
        "raw_state_sha256": _state_sha256(raw),
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "candidate_control_differing_state": differing,
        "candidate_control_only_rotation_buffer_differs": differing
        == ["stem.blocks.0.rotation_powers"],
        "trainable_state_keys": sorted(candidate_trainable),
        "candidate_control_trainable_bit_exact": control_trainable.keys()
        == candidate_trainable.keys()
        and all(
            torch.equal(value, candidate_trainable[name])
            for name, value in control_trainable.items()
        ),
        "trainable_parameters": sum(value.numel() for value in candidate_trainable.values()),
    }


def _build_dataset(
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    data_path: Path,
) -> tuple[MangoYOLOCropDataset, object, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    paths = [Path(value).resolve() for value in dataset.sample_paths()]
    exact = len(paths) == len(rows) and all(
        source.image_path == observed for source, observed in zip(rows, paths)
    )
    train_only = all(
        "train" in {part.casefold() for part in path.parts}
        and "val" not in {part.casefold() for part in path.parts}
        and "test" not in {part.casefold() for part in path.parts}
        for path in paths
    )
    if not exact or not train_only:
        raise ValueError("Dataset path/order violates the locked train-only contract.")
    return dataset, _build_eval_transform(semantics), {
        "rows": len(paths),
        "paths_exact": exact,
        "train_paths_only": train_only,
        "semantics": dict(semantics),
    }


def _movement_summary(initial: Mapping[str, Tensor], model: nn.Module) -> Dict[str, object]:
    final = model.state_dict()
    return {
        name: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final[name]),
            "changed": not torch.equal(value, final[name].detach().cpu()),
        }
        for name, value in initial.items()
    }


def _train_variant(
    *,
    name: str,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[nn.Module, Dict[str, object]]:
    started = time.perf_counter()
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device)
    trainable_names = _configure_trainability(model)
    initial_state_sha = _state_sha256(model)
    initial_frozen_sha = _frozen_state_sha256(model)
    tracked_names = {
        *trainable_names,
        "stem.blocks.0.branch_norm.running_mean",
        "stem.blocks.0.branch_norm.running_var",
        "stem.blocks.0.branch_norm.num_batches_tracked",
    }
    initial_tracked = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key in tracked_names
    }
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"ceconv_{name}_fit",
        seed=int(args.seed) + 200,
    )
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[Dict[str, object]] = []
    observed_indices: list[int] = []
    gradient_seen = {key: False for key in trainable_names}
    all_gradients_finite = True
    rng_checkpoints = []
    named_parameters = dict(model.named_parameters())
    for batch_index, (images, targets, metadata) in enumerate(loader):
        if batch_index >= int(args.max_train_batches):
            break
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        observed_indices.extend(int(value) for value in sample_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = _forward_logits(model, images, metadata, device=device)
            loss = F.cross_entropy(logits.float(), targets)
        rng_checkpoints.append(_rng_summary(_rng_snapshot()))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {name} loss at batch {batch_index}.")
        loss.backward()
        gradient_squared = 0.0
        for parameter_name in trainable_names:
            gradient = named_parameters[parameter_name].grad
            if gradient is None:
                continue
            all_gradients_finite = bool(
                all_gradients_finite and torch.isfinite(gradient).all()
            )
            gradient_seen[parameter_name] = bool(
                gradient_seen[parameter_name]
                or int(torch.count_nonzero(gradient).item()) > 0
            )
            gradient_squared += float(gradient.detach().float().square().sum().item())
        optimizer.step()
        history.append(
            {
                "variant": name,
                "step": batch_index + 1,
                "loss": float(loss.detach().item()),
                "gradient_norm": math.sqrt(gradient_squared),
            }
        )
    if len(history) != TRAIN_STEPS or observed_indices != list(train_indices):
        raise ValueError("CEConv train budget/order differs from the locked protocol.")
    final_state_sha = _state_sha256(model)
    final_frozen_sha = _frozen_state_sha256(model)
    gate = torch.tanh(_adapter(model).gate.detach().float()).cpu()
    movement = _movement_summary(initial_tracked, model)
    model = model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(observed_indices),
        "train_order_sha256": _ordered_index_sha256(observed_indices),
        "trainable_parameters": trainable_names,
        "trainable_parameter_count": sum(
            int(named_parameters[key].numel()) for key in trainable_names
        ),
        "gradient_seen_nonzero": gradient_seen,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "all_gradients_finite": all_gradients_finite,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "state_changed": initial_state_sha != final_state_sha,
        "frozen_state_sha256_before": initial_frozen_sha,
        "frozen_state_sha256_after": final_frozen_sha,
        "frozen_state_bit_exact": initial_frozen_sha == final_frozen_sha,
        "state_movement": movement,
        "gate": {
            "minimum": float(gate.min().item()),
            "mean": float(gate.mean().item()),
            "maximum": float(gate.max().item()),
            "maximum_absolute": float(gate.abs().max().item()),
            "nonzero_channels": int(torch.count_nonzero(gate).item()),
            "finite": bool(torch.isfinite(gate).all()),
        },
        "rng_checkpoints": rng_checkpoints,
        "history": history,
        "loader": loader_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _equation_diagnostics(
    raw: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
) -> Dict[str, object]:
    control_adapter = copy.deepcopy(_adapter(control)).cpu().eval()
    candidate_adapter = copy.deepcopy(_adapter(candidate)).cpu().eval()
    rotation = hue_rotation_matrix(ROTATIONS)
    identity = torch.eye(3)
    diagonal = torch.ones(3) / math.sqrt(3.0)
    orthogonality_error = float((rotation.T @ rotation - identity).abs().amax().item())
    diagonal_error = float((rotation @ diagonal - diagonal).abs().amax().item())
    cycle_error = float(
        (torch.matrix_power(rotation, ROTATIONS) - identity).abs().amax().item()
    )
    determinant = float(torch.linalg.det(rotation).item())

    transformed = candidate_adapter.transformed_filters()
    replay = official_equation_replay(candidate_adapter.branch_weight, ROTATIONS)
    equation_error = float((transformed - replay).abs().amax().item())
    group_zero_exact = bool(
        torch.equal(
            transformed[:, 0, :, 0],
            candidate_adapter.branch_weight[:, :, 0],
        )
    )
    control_transformed = control_adapter.transformed_filters()
    candidate_control_filter_delta = float(
        (transformed - control_transformed).abs().amax().item()
    )

    values = torch.linspace(-1.0, 1.0, steps=2 * 3 * 13 * 11).reshape(2, 3, 13, 11)
    rotated_values = torch.einsum("ij,bjhw->bihw", rotation, values)
    with torch.inference_mode():
        raw_stem = copy.deepcopy(raw.stem).cpu().eval()(values)
        control_stem = copy.deepcopy(control.stem).cpu().eval()(values)
        candidate_stem = copy.deepcopy(candidate.stem).cpu().eval()(values)
        candidate_groups = candidate_adapter.pre_norm_groups(values)
        candidate_rotated = candidate_adapter.pre_norm_groups(rotated_values)
        control_groups = control_adapter.pre_norm_groups(values)
        control_rotated = control_adapter.pre_norm_groups(rotated_values)
    candidate_shift_errors = {
        str(shift): float(
            (candidate_rotated - torch.roll(candidate_groups, shifts=shift, dims=2))
            .abs()
            .amax()
            .item()
        )
        for shift in (-1, 1)
    }
    control_shift_errors = {
        str(shift): float(
            (control_rotated - torch.roll(control_groups, shifts=shift, dims=2))
            .abs()
            .amax()
            .item()
        )
        for shift in (-1, 1)
    }
    return {
        "rotation": {
            "matrix": rotation.tolist(),
            "orthogonality_max_abs_error": orthogonality_error,
            "diagonal_preservation_max_abs_error": diagonal_error,
            "cycle_max_abs_error": cycle_error,
            "determinant": determinant,
        },
        "filter_equation": {
            "maximum_absolute_error": equation_error,
            "group_zero_bit_exact": group_zero_exact,
            "candidate_control_max_abs_delta": candidate_control_filter_delta,
            "shape": [int(value) for value in transformed.shape],
        },
        "hue_cycle": {
            "candidate_shift_errors": candidate_shift_errors,
            "candidate_best_error": min(candidate_shift_errors.values()),
            "control_shift_errors": control_shift_errors,
            "control_best_error": min(control_shift_errors.values()),
        },
        "zero_gate": {
            "raw_control_stem_bit_exact": torch.equal(raw_stem, control_stem),
            "raw_candidate_stem_bit_exact": torch.equal(raw_stem, candidate_stem),
        },
    }


def _initial_exactness(
    *,
    raw: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    prior_clean: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    count = min(5, int(images_cpu.size(0)))
    images = images_cpu[:count].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=count)
    outputs: Dict[str, Tensor] = {}
    rng = {}
    for name, prototype in (
        ("raw", raw),
        ("control", control),
        ("candidate", candidate),
    ):
        set_seed(SEED, deterministic=True)
        model = copy.deepcopy(prototype).to(device).eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            outputs[name] = _forward_logits(
                model,
                images,
                metadata,
                device=device,
            ).detach().cpu()
        rng[name] = _rng_summary(_rng_snapshot())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    sample_indices = metadata_cpu.get("sample_index")
    if not torch.is_tensor(sample_indices):
        raise ValueError("Initial exactness metadata is missing sample_index.")
    prior_by_index = {int(row["sample_index"]): row for row in prior_clean}
    prior_probabilities = torch.tensor(
        [
            [
                float(prior_by_index[int(index)][f"prob_{class_index}"])
                for class_index in range(5)
            ]
            for index in sample_indices[:count].tolist()
        ],
        dtype=torch.float32,
    )
    raw_probabilities = outputs["raw"].softmax(dim=1)
    return {
        "rows": count,
        "raw_control_logits_bit_exact": torch.equal(outputs["raw"], outputs["control"]),
        "raw_candidate_logits_bit_exact": torch.equal(
            outputs["raw"], outputs["candidate"]
        ),
        "control_candidate_logits_bit_exact": torch.equal(
            outputs["control"], outputs["candidate"]
        ),
        "raw_prior_argmax_match": torch.equal(
            raw_probabilities.argmax(dim=1), prior_probabilities.argmax(dim=1)
        ),
        "raw_prior_probability_max_abs_error": float(
            (raw_probabilities - prior_probabilities).abs().amax().item()
        ),
        "rng_equal": rng["raw"] == rng["control"] == rng["candidate"],
        "rng": rng,
    }


def _predict_paired_conditions(
    *,
    control_prototype: nn.Module,
    candidate_prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, object],
]:
    control = copy.deepcopy(control_prototype).to(device).eval()
    candidate = copy.deepcopy(candidate_prototype).to(device).eval()
    control_output: Dict[str, list[Dict[str, object]]] = {}
    candidate_output: Dict[str, list[Dict[str, object]]] = {}
    loader_summaries: Dict[str, object] = {}
    condition_specs = [("clean", None, None), *LIGHTING_CONDITIONS]
    for condition_index, (condition, brightness, contrast) in enumerate(condition_specs):
        if condition == "clean":
            loader, summary = _make_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=holdout_indices,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context="ceconv_paired_clean_holdout",
                seed=int(args.seed) + 500,
            )
        else:
            loader, summary = _make_lighting_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=holdout_indices,
                brightness=float(brightness),
                contrast=float(contrast),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"ceconv_paired_{condition}_holdout",
                seed=int(args.seed) + 510 + condition_index,
            )
        control_rows: list[Dict[str, object]] = []
        candidate_rows: list[Dict[str, object]] = []
        with torch.inference_mode():
            for images, targets, metadata in loader:
                images = images.to(device=device, non_blocking=True)
                targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
                sample_indices = metadata.get("sample_index")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Paired prediction metadata is missing sample_index.")
                probabilities = []
                for model in (control, candidate):
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = _forward_logits(model, images, metadata, device=device)
                    probabilities.append(logits.float().softmax(dim=1))
                for position, sample_index in enumerate(sample_indices.tolist()):
                    for destination, values in (
                        (control_rows, probabilities[0]),
                        (candidate_rows, probabilities[1]),
                    ):
                        row: Dict[str, object] = {
                            "sample_index": int(sample_index),
                            "target": int(targets[position].item()),
                            "prediction": int(values[position].argmax().item()),
                        }
                        for class_index in range(5):
                            row[f"prob_{class_index}"] = float(
                                values[position, class_index].item()
                            )
                        destination.append(row)
        control_output[str(condition)] = control_rows
        candidate_output[str(condition)] = candidate_rows
        loader_summaries[str(condition)] = summary
    del control, candidate
    gc.collect()
    torch.cuda.empty_cache()
    expected = list(holdout_indices)
    for name, result in (("control", control_output), ("candidate", candidate_output)):
        for condition, values in result.items():
            observed = [int(row["sample_index"]) for row in values]
            if observed != expected:
                raise ValueError(f"{name}/{condition} paired prediction order differs.")
    return control_output, candidate_output, loader_summaries


def _benchmark(
    *,
    prototype: nn.Module,
    candidate: bool,
    images_cpu: Tensor,
    targets_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
) -> Dict[str, object]:
    set_seed(SEED, deterministic=True)
    model = copy.deepcopy(prototype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if candidate:
        _configure_trainability(model)
    named_parameters = dict(model.named_parameters())
    for name in ("head.weight", "head.bias"):
        parameter = named_parameters.get(name)
        if parameter is None:
            raise ValueError(f"Benchmark classifier parameter is missing: {name}")
        parameter.requires_grad_(True)
    benchmark_trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    metadata = _metadata_to_device(
        metadata_cpu,
        device=device,
        count=int(images_cpu.size(0)),
    )
    targets = targets_cpu.to(device=device, dtype=torch.long)

    def iteration() -> tuple[Tensor, Tensor]:
        model.zero_grad(set_to_none=True)
        images = images_cpu.detach().to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = _forward_logits(model, images, metadata, device=device)
            loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        return logits, loss

    iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, loss = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "logits_loss_finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
        "trainable_parameters": benchmark_trainable,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


class _IsolatedAdapterExport(nn.Module):
    def __init__(self, adapter: CEConvStemResidual) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(self, images: Tensor) -> Tensor:
        return self.adapter(images)


class _FullCandidateExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _onnx_compare(
    *,
    wrapper: nn.Module,
    inputs: tuple[Tensor, ...],
    input_names: Sequence[str],
    path: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    wrapper.eval()
    torch.onnx.export(
        wrapper,
        inputs,
        str(path),
        input_names=list(input_names),
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = wrapper(*inputs).detach().cpu()
    observed = torch.from_numpy(
        session.run(
            ["output"],
            {
                name: value.detach().cpu().numpy()
                for name, value in zip(input_names, inputs)
            },
        )[0]
    )
    return {
        "succeeded": True,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "opset": 17,
        "batch_contract": "static_batch_1",
        "providers": session.get_providers(),
        "shape": [int(value) for value in observed.shape],
        "finite": bool(torch.isfinite(observed).all()),
        "maximum_absolute_error": float((expected - observed).abs().amax().item()),
        "argmax_match": bool(
            expected.ndim < 2
            or torch.equal(expected.argmax(dim=-1), observed.argmax(dim=-1))
        ),
    }


def _failed_export(path: Path, error: Exception) -> Dict[str, object]:
    return {
        "succeeded": False,
        "path": str(path.resolve()),
        "sha256": _sha256(path) if path.is_file() else None,
        "opset": 17,
        "batch_contract": "static_batch_1",
        "providers": [],
        "shape": [],
        "finite": False,
        "maximum_absolute_error": 1e9,
        "argmax_match": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _export_diagnostics(
    *,
    candidate: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    images = images_cpu[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    isolated_path = output_dir / "ceconv_stem_residual_adapter.onnx"
    try:
        isolated = _onnx_compare(
            wrapper=_IsolatedAdapterExport(_adapter(candidate)),
            inputs=(images,),
            input_names=("images",),
            path=isolated_path,
        )
    except Exception as error:
        isolated = _failed_export(isolated_path, error)

    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(STATIC_EXPORT_BATCH_SIZE, 8)
    bbox = bbox[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            STATIC_EXPORT_BATCH_SIZE,
            int(images.size(-2)),
            int(images.size(-1)),
            dtype=torch.bool,
        )
    image_mask = image_mask[:STATIC_EXPORT_BATCH_SIZE].detach().bool().cpu()
    full_path = output_dir / "ceconv_stem_residual_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_FullCandidateExport(candidate),
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as error:
        full = _failed_export(full_path, error)
    return {"isolated": isolated, "full": full}


def _normalize_maps(values: Tensor) -> Tensor:
    values = values.detach().float()
    flat = values.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1)
    return (values - minimum) / (maximum - minimum).clamp_min(1e-9)


def _batched_gradcam(activations: Tensor, gradients: Tensor, size: tuple[int, int]) -> Tensor:
    if activations.ndim != 4 or gradients.shape != activations.shape:
        raise ValueError("Stem Grad-CAM tensors must be matching BCHW values.")
    weights = gradients.float().mean(dim=(2, 3), keepdim=True)
    heat = torch.relu((weights * activations.float()).sum(dim=1))
    heat = _normalize_maps(heat)
    return F.interpolate(
        heat.unsqueeze(1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _rgb_from_tensor(
    value: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    tensor = value.detach().float().cpu()
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor * std_tensor + mean_tensor
    return (
        tensor.clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _collect_xai_maps(
    *,
    name: str,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    selected_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    collect_groups: bool,
) -> Dict[str, object]:
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=selected_indices,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        context=f"ceconv_{name}_xai",
        seed=int(args.seed) + 700,
    )
    model = copy.deepcopy(prototype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records: Dict[int, Dict[str, object]] = {}
    for images_cpu, _, metadata in loader:
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("XAI metadata is missing sample_index.")
        images = images_cpu.to(device).requires_grad_(True)
        captured: Dict[str, Tensor] = {}

        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError("Stem hook output must be a tensor.")
            output.retain_grad()
            captured["activation"] = output

        handle = model.stem.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        logits = _forward_logits(model, images, metadata, device=device)
        logits[:, FOCUS_CLASS].sum().backward()
        handle.remove()
        activation = captured.get("activation")
        if activation is None or activation.grad is None:
            raise RuntimeError("Stem Grad-CAM activation/gradient was not captured.")
        heat = _batched_gradcam(
            activation,
            activation.grad,
            size=(int(images.size(-2)), int(images.size(-1))),
        ).cpu()
        group_energy = None
        winner = None
        if collect_groups:
            with torch.inference_mode():
                groups = _adapter(model).branch_groups(images.detach())
                group_energy = groups.detach().float().abs().mean(dim=1)
                winner = group_energy.argmax(dim=1)
                normalized = []
                for group_index in range(ROTATIONS):
                    normalized.append(_normalize_maps(group_energy[:, group_index]))
                group_energy = torch.stack(normalized, dim=1).cpu()
                winner = winner.cpu()
        probabilities = logits.detach().float().softmax(dim=1).cpu()
        for position, sample_index in enumerate(sample_indices.tolist()):
            row: Dict[str, object] = {
                "rgb": _rgb_from_tensor(images_cpu[position], mean=mean, std=std),
                "gradcam": heat[position].numpy(),
                "prediction": int(probabilities[position].argmax().item()),
                "probabilities": probabilities[position].tolist(),
            }
            if group_energy is not None and winner is not None:
                row["group_energy"] = group_energy[position].numpy()
                row["winner"] = winner[position].numpy()
            records[int(sample_index)] = row
    del model
    gc.collect()
    torch.cuda.empty_cache()
    observed = list(records)
    if observed != list(selected_indices):
        raise ValueError(f"{name} XAI order differs from selected indices.")
    finite = all(
        np.isfinite(np.asarray(row["gradcam"])).all()
        and (
            not collect_groups
            or (
                np.isfinite(np.asarray(row["group_energy"])).all()
                and np.isfinite(np.asarray(row["winner"])).all()
            )
        )
        for row in records.values()
    )
    return {"records": records, "loader": loader_summary, "finite": bool(finite)}


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray, *, alpha: float = 0.48) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    heat_image = Image.fromarray(
        np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0) * 255.0
    ).convert("L")
    heat_image = heat_image.resize(image.size, Image.Resampling.BILINEAR)
    values = np.asarray(heat_image, dtype=np.float32) / 255.0
    tint = np.stack(
        (
            np.full_like(values, 255.0),
            190.0 * values,
            35.0 * (1.0 - values),
        ),
        axis=-1,
    )
    source = np.asarray(image, dtype=np.float32)
    strength = (float(alpha) * values)[..., None]
    output = source * (1.0 - strength) + tint * strength
    return Image.fromarray(output.clip(0.0, 255.0).round().astype(np.uint8))


def _winner_overlay(rgb: np.ndarray, winner: np.ndarray) -> Image.Image:
    image = np.asarray(rgb, dtype=np.float32)
    winner_image = Image.fromarray(np.asarray(winner, dtype=np.uint8)).resize(
        (int(image.shape[1]), int(image.shape[0])),
        Image.Resampling.NEAREST,
    )
    winner_values = np.asarray(winner_image, dtype=np.int64)
    palette = np.asarray(
        [[225.0, 45.0, 45.0], [40.0, 190.0, 80.0], [50.0, 105.0, 235.0]],
        dtype=np.float32,
    )
    output = 0.58 * image + 0.42 * palette[winner_values]
    return Image.fromarray(output.clip(0.0, 255.0).round().astype(np.uint8))


def _required_xai_indices(
    raw_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[list[int], Dict[int, list[str]]]:
    categories: Dict[int, list[str]] = {}
    ranking = []
    for raw, control, candidate in zip(raw_rows, control_rows, candidate_rows):
        index = int(raw["sample_index"])
        target = int(raw["target"])
        raw_prediction = int(raw["prediction"])
        candidate_prediction = int(candidate["prediction"])
        labels = []
        if raw_prediction != target and candidate_prediction == target:
            labels.append("correction")
        if raw_prediction == target and candidate_prediction != target:
            labels.append("harm")
        if target == FOCUS_CLASS and raw_prediction == FOCUS_CLASS and candidate_prediction != FOCUS_CLASS:
            labels.append("class1_tp_break")
        if labels:
            categories[index] = labels
        restricted_fp = (
            target in RESTRICTED_NEGATIVE_CLASSES and raw_prediction == FOCUS_CLASS
        )
        delta = abs(
            float(candidate[f"prob_{FOCUS_CLASS}"])
            - float(control[f"prob_{FOCUS_CLASS}"])
        )
        ranking.append((not restricted_fp, -delta, index, target, raw_prediction))
    required = list(categories)
    selected = list(required)
    for _, _, index, target, raw_prediction in sorted(ranking):
        if len(selected) >= max(8, len(required)):
            break
        if index in categories:
            continue
        categories[index] = [
            "diagnostic_restricted_fp"
            if target in RESTRICTED_NEGATIVE_CLASSES and raw_prediction == FOCUS_CLASS
            else "diagnostic_boundary"
        ]
        selected.append(index)
    order = {int(row["sample_index"]): position for position, row in enumerate(raw_rows)}
    selected.sort(key=order.__getitem__)
    return selected, categories


def _render_xai_pages(
    *,
    output_dir: Path,
    selected_indices: Sequence[int],
    categories: Mapping[int, Sequence[str]],
    rows: Sequence[CleanTrainRow],
    raw_clean: Sequence[Mapping[str, object]],
    control_clean: Sequence[Mapping[str, object]],
    candidate_clean: Sequence[Mapping[str, object]],
    candidate_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    raw_maps: Mapping[int, Mapping[str, object]],
    control_maps: Mapping[int, Mapping[str, object]],
    candidate_maps: Mapping[int, Mapping[str, object]],
    gate: Mapping[str, object],
) -> list[str]:
    raw_by_index = {int(row["sample_index"]): row for row in raw_clean}
    control_by_index = {int(row["sample_index"]): row for row in control_clean}
    candidate_by_index = {int(row["sample_index"]): row for row in candidate_clean}
    condition_by_index = {
        condition: {int(row["sample_index"]): row for row in values}
        for condition, values in candidate_conditions.items()
    }
    columns = (
        "input",
        "raw Grad-CAM",
        "identity Grad-CAM",
        "CEConv Grad-CAM",
        "group 0",
        "group 1",
        "group 2",
        "winner",
    )
    tile = 144
    header = 28
    label_height = 48
    page_size = 12
    pages = []
    for page_index, start in enumerate(range(0, len(selected_indices), page_size), start=1):
        page_indices = list(selected_indices[start : start + page_size])
        canvas = Image.new(
            "RGB",
            (len(columns) * tile, header + len(page_indices) * (tile + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(columns):
            draw.text((column * tile + 4, 7), title, fill="black")
        for row_position, sample_index in enumerate(page_indices):
            y = header + row_position * (tile + label_height)
            rgb = np.asarray(raw_maps[sample_index]["rgb"], dtype=np.uint8)
            candidate_group = np.asarray(
                candidate_maps[sample_index]["group_energy"], dtype=np.float32
            )
            views = [
                Image.fromarray(rgb),
                _heat_overlay(rgb, np.asarray(raw_maps[sample_index]["gradcam"])),
                _heat_overlay(rgb, np.asarray(control_maps[sample_index]["gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate_maps[sample_index]["gradcam"])),
                *[
                    _heat_overlay(rgb, candidate_group[group_index])
                    for group_index in range(ROTATIONS)
                ],
                _winner_overlay(rgb, np.asarray(candidate_maps[sample_index]["winner"])),
            ]
            for column, view in enumerate(views):
                canvas.paste(
                    view.resize((tile, tile), Image.Resampling.BILINEAR),
                    (column * tile, y),
                )
            source = rows[sample_index]
            raw_row = raw_by_index[sample_index]
            control_row = control_by_index[sample_index]
            candidate_row = candidate_by_index[sample_index]
            lighting = "/".join(
                str(condition_by_index[name][sample_index]["prediction"])
                for name, _, _ in LIGHTING_CONDITIONS
            )
            line_one = (
                f"idx={sample_index} y={source.target} raw/id/ce="
                f"{raw_row['prediction']}/{control_row['prediction']}/"
                f"{candidate_row['prediction']} [{','.join(categories[sample_index])}]"
            )
            line_two = (
                f"lighting ce={lighting}; gate min/mean/max="
                f"{float(gate['minimum']):+.4f}/{float(gate['mean']):+.4f}/"
                f"{float(gate['maximum']):+.4f}"
            )
            draw.text((4, y + tile + 3), line_one, fill="black")
            draw.text((4, y + tile + 22), line_two, fill="black")
        path = output_dir / f"ceconv_xai_contact_sheet_{page_index:02d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages


def _xai_audit(
    *,
    output_dir: Path,
    raw: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    checkpoint: Mapping[str, object],
    base_dataset: MangoYOLOCropDataset,
    transform,
    rows: Sequence[CleanTrainRow],
    raw_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    control_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_training: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    selected, categories = _required_xai_indices(
        raw_conditions["clean"],
        control_conditions["clean"],
        candidate_conditions["clean"],
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    raw_result = _collect_xai_maps(
        name="raw",
        prototype=raw,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=device,
        mean=mean,
        std=std,
        collect_groups=False,
    )
    control_result = _collect_xai_maps(
        name="identity_control",
        prototype=control,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=device,
        mean=mean,
        std=std,
        collect_groups=True,
    )
    candidate_result = _collect_xai_maps(
        name="ceconv_candidate",
        prototype=candidate,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=device,
        mean=mean,
        std=std,
        collect_groups=True,
    )
    pages = _render_xai_pages(
        output_dir=output_dir,
        selected_indices=selected,
        categories=categories,
        rows=rows,
        raw_clean=raw_conditions["clean"],
        control_clean=control_conditions["clean"],
        candidate_clean=candidate_conditions["clean"],
        candidate_conditions=candidate_conditions,
        raw_maps=raw_result["records"],
        control_maps=control_result["records"],
        candidate_maps=candidate_result["records"],
        gate=candidate_training["gate"],
    )
    required = [
        index
        for index, values in categories.items()
        if any(value in {"correction", "harm", "class1_tp_break"} for value in values)
    ]
    manifest = {
        "selected_indices": selected,
        "categories": {str(key): list(value) for key, value in categories.items()},
        "required_indices": required,
        "required_coverage_exact": set(required).issubset(set(selected)),
        "raw_maps_finite": bool(raw_result["finite"]),
        "control_maps_finite": bool(control_result["finite"]),
        "candidate_maps_finite": bool(candidate_result["finite"]),
        "page_count": len(pages),
        "pages": pages,
        "loaders": {
            "raw": raw_result["loader"],
            "control": control_result["loader"],
            "candidate": candidate_result["loader"],
        },
        "target_class": FOCUS_CLASS,
        "gradcam_source": "stem_output",
        "group_energy_source": "candidate_first_block_pre_coset_abs_channel_mean",
    }
    manifest_path = output_dir / "ceconv_xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path.resolve())}


def _build_comparisons(
    *,
    raw_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    control_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_conditions: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, object]]:
    output: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        raw = raw_conditions[condition]
        control = control_conditions[condition]
        candidate = candidate_conditions[condition]
        expected = [int(row["sample_index"]) for row in raw]
        if [int(row["sample_index"]) for row in control] != expected:
            raise ValueError(f"Control {condition} order differs from raw.")
        if [int(row["sample_index"]) for row in candidate] != expected:
            raise ValueError(f"Candidate {condition} order differs from raw.")
        output[condition] = {
            "raw_control": _comparison(
                control_rows=raw,
                candidate_rows=control,
                num_classes=5,
                focus_class=FOCUS_CLASS,
            ),
            "raw_candidate": _comparison(
                control_rows=raw,
                candidate_rows=candidate,
                num_classes=5,
                focus_class=FOCUS_CLASS,
            ),
            "control_candidate": _comparison(
                control_rows=control,
                candidate_rows=candidate,
                num_classes=5,
                focus_class=FOCUS_CLASS,
            ),
        }
    return output


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    clean = comparisons["clean"]
    raw_candidate = clean["raw_candidate"]
    control_candidate = clean["control_candidate"]
    raw_delta = raw_candidate["delta"]
    control_delta = control_candidate["delta"]
    raw_transitions = raw_candidate["transitions"]
    control_transitions = control_candidate["transitions"]
    decision_checks = {
        "macro_f1_gain_vs_raw_gte_0p001": float(raw_delta["macro_f1"]) >= 0.001,
        "class1_f1_gain_vs_raw_gte_0p003": float(raw_delta["class1_f1"]) >= 0.003,
        "class1_precision_gain_vs_raw_gte_0p005": float(
            raw_delta["class1_precision"]
        )
        >= 0.005,
        "macro_f1_delta_vs_control_nonnegative": float(control_delta["macro_f1"])
        >= 0.0,
        "class1_f1_gain_vs_control_gte_0p002": float(control_delta["class1_f1"])
        >= 0.002,
        "class1_precision_gain_vs_control_gte_0p003": float(
            control_delta["class1_precision"]
        )
        >= 0.003,
        "class1_recall_delta_vs_raw_gte_minus_0p005": float(
            raw_delta["class1_recall"]
        )
        >= -0.005,
        "class1_recall_delta_vs_control_gte_minus_0p005": float(
            control_delta["class1_recall"]
        )
        >= -0.005,
        "zero_net_raw_class1_tp_break": int(raw_transitions["focus_tp_break"])
        <= int(raw_transitions["focus_fn_rescue"]),
        "restricted_fp_reduction_vs_raw_gte_2": int(
            raw_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "restricted_fp_reduction_vs_control_gte_1": int(
            control_transitions["restricted_focus_fp_reduction"]
        )
        >= 1,
        "corrections_gt_harms_vs_raw": int(raw_transitions["candidate_correction"])
        > int(raw_transitions["candidate_harm"]),
        "corrections_gt_harms_vs_control": int(
            control_transitions["candidate_correction"]
        )
        > int(control_transitions["candidate_harm"]),
        "maximum_nonfocus_f1_drop_vs_raw_lte_0p010": float(
            raw_candidate["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "new_3_to_2_harms_lte_2": int(
            raw_transitions["new_nonfocus_3_to_2_harms"]
        )
        <= 2,
    }
    lighting = [comparisons[name] for name, _, _ in LIGHTING_CONDITIONS]
    control_macro = [
        float(row["control_candidate"]["delta"]["macro_f1"]) for row in lighting
    ]
    control_precision = [
        float(row["control_candidate"]["delta"]["class1_precision"])
        for row in lighting
    ]
    raw_recall = [
        float(row["raw_candidate"]["delta"]["class1_recall"]) for row in lighting
    ]
    raw_removals = sum(
        int(row["raw_candidate"]["transitions"]["focus_fp_remove_correct"])
        for row in lighting
    )
    raw_creations = sum(
        int(row["raw_candidate"]["transitions"]["focus_fp_create"])
        for row in lighting
    )
    control_removals = sum(
        int(row["control_candidate"]["transitions"]["focus_fp_remove_correct"])
        for row in lighting
    )
    control_creations = sum(
        int(row["control_candidate"]["transitions"]["focus_fp_create"])
        for row in lighting
    )
    illumination_checks = {
        "three_lighting_conditions_exact": len(lighting) == 3,
        "candidate_control_macro_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_macro
        )
        >= 2,
        "candidate_control_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_precision
        )
        >= 2,
        "worst_class1_recall_delta_vs_raw_gte_minus_0p020": min(
            raw_recall, default=-math.inf
        )
        >= -0.020,
        "net_class1_tp_loss_lte_2_each": all(
            int(row["raw_candidate"]["transitions"]["focus_tp_break"])
            - int(row["raw_candidate"]["transitions"]["focus_fn_rescue"])
            <= 2
            for row in lighting
        ),
        "aggregate_restricted_fp_removals_gt_creations_vs_raw": raw_removals
        > raw_creations,
        "aggregate_restricted_fp_removals_gt_creations_vs_control": control_removals
        > control_creations,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **decision_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": not failed,
        "test_authorized": False,
        "full_train_authorized": False,
    }


def _write_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    raw_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    control_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_conditions: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for name in ("raw", "control", "candidate"):
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(5))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            for raw, control, candidate in zip(
                raw_conditions[condition],
                control_conditions[condition],
                candidate_conditions[condition],
            ):
                sample_index = int(raw["sample_index"])
                source = rows[sample_index]
                row: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source.source_stem,
                    "image_path": str(source.image_path),
                    "fold": source.fold,
                    "target": source.target,
                }
                for name, values in (
                    ("raw", raw),
                    ("control", control),
                    ("candidate", candidate),
                ):
                    row[f"{name}_prediction"] = int(values["prediction"])
                    for class_index in range(5):
                        row[f"{name}_prob_{class_index}"] = float(
                            values[f"prob_{class_index}"]
                        )
                writer.writerow(row)


def _write_training_curve(
    path: Path,
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    rows = [*control["history"], *candidate["history"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "step", "loss", "gradient_norm"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _replay_predictions(
    path: Path,
    *,
    expected: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    grouped = {
        condition: {name: [] for name in ("raw", "control", "candidate")}
        for condition in CONDITIONS
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            condition = str(source["condition"])
            if condition not in grouped:
                raise ValueError(f"Replay found unexpected condition: {condition}")
            for name in grouped[condition]:
                row: Dict[str, object] = {
                    "sample_index": int(source["sample_index"]),
                    "target": int(source["target"]),
                    "prediction": int(source[f"{name}_prediction"]),
                }
                for class_index in range(5):
                    row[f"prob_{class_index}"] = float(
                        source[f"{name}_prob_{class_index}"]
                    )
                grouped[condition][name].append(row)
    replay = _build_comparisons(
        raw_conditions={key: value["raw"] for key, value in grouped.items()},
        control_conditions={key: value["control"] for key, value in grouped.items()},
        candidate_conditions={key: value["candidate"] for key, value in grouped.items()},
    )
    matches = replay == expected
    return {
        "rows": sum(len(value["raw"]) for value in grouped.values()),
        "condition_rows": {
            condition: len(values["raw"]) for condition, values in grouped.items()
        },
        "comparisons_exact": matches,
        "prediction_sha256": _sha256(path),
        "replayed_comparisons": replay,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    clean = summary["comparisons"]["clean"]
    raw = clean["raw_candidate"]
    control = clean["control_candidate"]
    gate = summary["gate"]
    lines = [
        "# CEConv Stem Residual A0 Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Clean macro/class1 F1 delta vs raw: `{raw['delta']['macro_f1']:+.6f}/{raw['delta']['class1_f1']:+.6f}`",
        f"- Clean class1 precision/recall delta vs raw: `{raw['delta']['class1_precision']:+.6f}/{raw['delta']['class1_recall']:+.6f}`",
        f"- Clean class1 F1/precision delta vs identity control: `{control['delta']['class1_f1']:+.6f}/{control['delta']['class1_precision']:+.6f}`",
        f"- Restricted FP reduction / TP breaks vs raw: `{raw['transitions']['restricted_focus_fp_reduction']}/{raw['transitions']['focus_tp_break']}`",
        f"- Candidate runtime ratio / peak VRAM GiB: `{summary['resources']['runtime_ratio']:.6f}/{summary['resources']['candidate_benchmark']['peak_vram_gib']:.6f}`",
        f"- XAI rows/pages: `{len(summary['xai']['selected_indices'])}/{summary['xai']['page_count']}`",
        "",
        "Only source-disjoint yolo_f/train rows were used. Validation and test were not accessed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    forbidden = {".pt", ".pth", ".ckpt", ".engine", ".trt"}
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden trainable binary artifact: {path}")
        artifacts.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    manifest = {
        "method": METHOD,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked CEConv A0 requires CUDA.")
    provenance, rows, cohorts, prior = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "method": METHOD,
            "provenance": provenance,
            "cohort": _cohort_serializable(cohorts),
            "output_directory_created": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }

    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("CEConv output cannot be written under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)

    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    amp_dtype = _amp_dtype(device)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    raw, control, candidate, construction = _construct_models(checkpoint)
    equation = _equation_diagnostics(raw, control, candidate)
    dataset, transform, dataset_summary = _build_dataset(checkpoint, rows, paths["data"])
    train_indices = balanced_boundary_order(
        cohorts["reference_indices"],
        cohorts["hard_indices"],
        steps=TRAIN_STEPS,
        half_batch=HALF_BATCH,
        seed=SEED,
    )
    reference_set = set(int(value) for value in cohorts["reference_indices"])
    hard_set = set(int(value) for value in cohorts["hard_indices"])
    balanced_batches_exact = all(
        sum(index in reference_set for index in train_indices[start : start + BATCH_SIZE])
        == HALF_BATCH
        and sum(index in hard_set for index in train_indices[start : start + BATCH_SIZE])
        == HALF_BATCH
        for start in range(0, len(train_indices), BATCH_SIZE)
    )

    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"][:BATCH_SIZE],
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="ceconv_resource_batch",
        seed=SEED + 400,
    )
    images_cpu, targets_cpu, metadata_cpu = next(iter(resource_loader))
    initial = _initial_exactness(
        raw=raw,
        control=control,
        candidate=candidate,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        prior_clean=prior["clean"]["raw"],
        device=device,
    )

    control_model, control_training = _train_variant(
        name="identity_control",
        prototype=control,
        base_dataset=dataset,
        transform=transform,
        train_indices=train_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    candidate_model, candidate_training = _train_variant(
        name="ceconv_candidate",
        prototype=candidate,
        base_dataset=dataset,
        transform=transform,
        train_indices=train_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )

    control_conditions, candidate_conditions, paired_loaders = _predict_paired_conditions(
        control_prototype=control_model,
        candidate_prototype=candidate_model,
        base_dataset=dataset,
        transform=transform,
        holdout_indices=cohorts["holdout_indices"],
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    raw_conditions = {condition: prior[condition]["raw"] for condition in CONDITIONS}
    comparisons = _build_comparisons(
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )

    raw_benchmark = _benchmark(
        prototype=raw,
        candidate=False,
        images_cpu=images_cpu,
        targets_cpu=targets_cpu,
        metadata_cpu=metadata_cpu,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    candidate_benchmark = _benchmark(
        prototype=candidate_model,
        candidate=True,
        images_cpu=images_cpu,
        targets_cpu=targets_cpu,
        metadata_cpu=metadata_cpu,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    runtime_ratio = float(
        candidate_benchmark["median_seconds"]
        / max(float(raw_benchmark["median_seconds"]), 1e-12)
    )
    export = _export_diagnostics(
        candidate=candidate_model,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        output_dir=output_dir,
    )
    xai = _xai_audit(
        output_dir=output_dir,
        raw=raw,
        control=control_model,
        candidate=candidate_model,
        checkpoint=checkpoint,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
        candidate_training=candidate_training,
        args=args,
        device=device,
    )

    candidate_movement = candidate_training["state_movement"]
    control_movement = control_training["state_movement"]
    trainable_names = set(candidate_training["trainable_parameters"])
    running_names = {
        "stem.blocks.0.branch_norm.running_mean",
        "stem.blocks.0.branch_norm.running_var",
        "stem.blocks.0.branch_norm.num_batches_tracked",
    }
    clean_probability_delta = max(
        abs(float(candidate[f"prob_{class_index}"]) - float(control[f"prob_{class_index}"]))
        for control, candidate in zip(
            control_conditions["clean"], candidate_conditions["clean"]
        )
        for class_index in range(5)
    )
    structural_checks = {
        "locked_sources_verified": True,
        "tracked_worktree_clean": bool(provenance["tracked_worktree_clean"]),
        "official_worktree_clean": bool(provenance["official_worktree_clean"]),
        "train_only_dataset_contract_exact": bool(
            dataset_summary["paths_exact"] and dataset_summary["train_paths_only"]
        ),
        "source_disjoint_7372_1843_exact": len(cohorts["fit_indices"]) == 7372
        and len(cohorts["holdout_indices"]) == EXPECTED_HOLDOUT_ROWS
        and not cohorts["source_overlap"],
        "fit_cohort_432_186_exact": len(cohorts["reference_indices"])
        == EXPECTED_REFERENCE_ROWS
        and len(cohorts["hard_indices"]) == EXPECTED_HARD_ROWS,
        "balanced_60x32_order_exact": len(train_indices) == TRAIN_STEPS * BATCH_SIZE
        and balanced_batches_exact,
        "candidate_control_only_rotation_buffer_differs": bool(
            construction["candidate_control_only_rotation_buffer_differs"]
        ),
        "candidate_control_trainable_initial_state_bit_exact": bool(
            construction["candidate_control_trainable_bit_exact"]
        ),
        "trainable_parameter_count_960": int(construction["trainable_parameters"])
        == EXPECTED_TRAINABLE_PARAMETERS,
        "rotation_orthogonal_lte_1e6": float(
            equation["rotation"]["orthogonality_max_abs_error"]
        )
        <= 1e-6,
        "rotation_preserves_rgb_diagonal_lte_1e6": float(
            equation["rotation"]["diagonal_preservation_max_abs_error"]
        )
        <= 1e-6,
        "rotation_determinant_plus_one_lte_1e6": abs(
            float(equation["rotation"]["determinant"]) - 1.0
        )
        <= 1e-6,
        "rotation_cycle_lte_1e6": float(equation["rotation"]["cycle_max_abs_error"])
        <= 1e-6,
        "official_filter_equation_lte_1e7": float(
            equation["filter_equation"]["maximum_absolute_error"]
        )
        <= 1e-7,
        "filter_group_zero_bit_exact": bool(
            equation["filter_equation"]["group_zero_bit_exact"]
        ),
        "candidate_control_filter_distinct": float(
            equation["filter_equation"]["candidate_control_max_abs_delta"]
        )
        >= 1e-4,
        "candidate_hue_cycle_error_lte_1e5": float(
            equation["hue_cycle"]["candidate_best_error"]
        )
        <= 1e-5,
        "identity_control_fails_hue_cycle_distinction": float(
            equation["hue_cycle"]["control_best_error"]
        )
        >= 1e-4,
        "zero_gate_stem_bit_exact": bool(
            equation["zero_gate"]["raw_control_stem_bit_exact"]
            and equation["zero_gate"]["raw_candidate_stem_bit_exact"]
        ),
        "zero_gate_logits_bit_exact": bool(
            initial["raw_control_logits_bit_exact"]
            and initial["raw_candidate_logits_bit_exact"]
            and initial["control_candidate_logits_bit_exact"]
        ),
        "initial_raw_prior_argmax_match": bool(initial["raw_prior_argmax_match"]),
        "initial_forward_rng_equal": bool(initial["rng_equal"]),
        "matched_train_order_and_rng": control_training["train_order_sha256"]
        == candidate_training["train_order_sha256"]
        == _ordered_index_sha256(train_indices)
        and control_training["rng_checkpoints"] == candidate_training["rng_checkpoints"],
        "both_gradients_finite_nonzero": bool(
            control_training["all_gradients_finite"]
            and candidate_training["all_gradients_finite"]
            and control_training["all_trainable_gradients_seen"]
            and candidate_training["all_trainable_gradients_seen"]
        ),
        "both_frozen_states_bit_exact": bool(
            control_training["frozen_state_bit_exact"]
            and candidate_training["frozen_state_bit_exact"]
        ),
        "all_trainable_tensors_moved": all(
            bool(candidate_movement[name]["changed"])
            and bool(control_movement[name]["changed"])
            for name in trainable_names
        ),
        "both_bn_running_stats_updated": all(
            bool(candidate_movement[name]["changed"])
            and bool(control_movement[name]["changed"])
            for name in running_names
        ),
        "candidate_gate_finite_bounded_nonzero_24": bool(
            candidate_training["gate"]["finite"]
            and float(candidate_training["gate"]["maximum_absolute"]) <= MAX_GATE_ABS
            and int(candidate_training["gate"]["nonzero_channels"]) >= 24
        ),
        "control_gate_finite_bounded_nonzero_24": bool(
            control_training["gate"]["finite"]
            and float(control_training["gate"]["maximum_absolute"]) <= MAX_GATE_ABS
            and int(control_training["gate"]["nonzero_channels"]) >= 24
        ),
        "candidate_control_adapted_outputs_distinct": clean_probability_delta >= 1e-5,
        "isolated_onnx_error_lte_1e5": bool(export["isolated"]["succeeded"])
        and float(export["isolated"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR,
        "full_onnx_error_lte_1e5_argmax_match": bool(export["full"]["succeeded"])
        and float(export["full"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR
        and bool(export["full"]["argmax_match"]),
        "runtime_ratio_lte_1p35": runtime_ratio <= float(args.max_runtime_ratio),
        "peak_vram_lte_3p25_gib": float(candidate_benchmark["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
        "xai_required_coverage_exact": bool(xai["required_coverage_exact"]),
        "xai_all_maps_finite": bool(
            xai["raw_maps_finite"]
            and xai["control_maps_finite"]
            and xai["candidate_maps_finite"]
        ),
        "xai_contact_sheet_written": int(xai["page_count"]) >= 1,
        "current_best_commands_hash_unchanged": provenance["sha256"]["current_commands"]
        == LOCKED_CURRENT_COMMAND_SHA256,
        "command_history_hash_unchanged": provenance["sha256"]["command_history"]
        == LOCKED_COMMAND_HISTORY_SHA256,
        "validation_not_used": not bool(provenance["validation_predictions_used"]),
        "test_not_used": not bool(provenance["test_data_used"]),
    }
    gate = assess_stage_a(structural_checks=structural_checks, comparisons=comparisons)

    predictions_path = output_dir / "predictions_all_conditions.csv"
    training_path = output_dir / "training_curve.csv"
    replay_path = output_dir / "independent_replay.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_predictions(
        predictions_path,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    _write_training_curve(training_path, control_training, candidate_training)
    replay = _replay_predictions(predictions_path, expected=comparisons)
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if not bool(replay["comparisons_exact"]):
        raise RuntimeError("Independent CSV replay differs from in-memory comparisons.")

    summary: Dict[str, object] = {
        "status": "authorized" if gate["stage_b_authorized"] else "rejected",
        "method": METHOD,
        "protocol_stage": "A_train_only_matched_identity_control",
        "protocol": {
            "rotations": ROTATIONS,
            "seed": SEED,
            "fold": FOLD,
            "batch_size": BATCH_SIZE,
            "half_batch": HALF_BATCH,
            "train_steps": TRAIN_STEPS,
            "train_rows": len(train_indices),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "selection": "single_final_step_60_no_sweep",
        },
        "provenance": provenance,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "cohort": {
            **_cohort_serializable(cohorts),
            "balanced_train_order_sha256": _ordered_index_sha256(train_indices),
            "balanced_batches_exact": balanced_batches_exact,
        },
        "dataset": dataset_summary,
        "construction": construction,
        "equation": equation,
        "initial_exactness": initial,
        "training": {
            "control": control_training,
            "candidate": candidate_training,
        },
        "comparisons": comparisons,
        "resources": {
            "raw_benchmark": raw_benchmark,
            "candidate_benchmark": candidate_benchmark,
            "runtime_ratio": runtime_ratio,
            "export": export,
        },
        "xai": xai,
        "gate": gate,
        "replay": {
            key: value for key, value in replay.items() if key != "replayed_comparisons"
        },
        "loaders": {
            "resource": resource_loader_summary,
            "paired_conditions": paired_loaders,
        },
        "validation_predictions_used": False,
        "test_data_used": False,
        "raw_dataset_touched": False,
        "shared_model_code_touched": False,
        "trainable_checkpoint_written": False,
        "onnx_export_written": True,
        "current_best_command_revision_created": False,
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "training_curve": str(training_path.resolve()),
            "independent_replay": str(replay_path.resolve()),
            "xai_manifest": xai["manifest"],
            "manifest": str(manifest_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(json.dumps(result["gate"], indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
