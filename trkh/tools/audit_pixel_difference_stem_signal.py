from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Dict, Mapping, Optional, Sequence
import warnings

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import IdentityCorruption, LightingShift
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    ConvStemBlock,
    HybridConvStem,
    create_model,
    load_model_state,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import _amp_dtype, _predict
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "pixel_difference_stem_signal_a0"
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42
BENCHMARK_REPEATS = 3
MAX_RUNTIME_RATIO = 1.50
MAX_MEMORY_RATIO = 1.25
MAX_ONNX_ERROR = 1e-5
MAX_EQUATION_ERROR = 1e-6
MAX_FINITE_DIFFERENCE_ERROR = 1e-4
MAX_BF16_ERROR = 0.02
STATIC_EXPORT_BATCH_SIZE = 1
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)
EXPECTED_COHORT_ROWS = 607
EXPECTED_POSITIVES = 421
EXPECTED_NEGATIVES = 186
EXPECTED_FOLD_COUNTS = {
    1: {"tp": 112, "fp": 45},
    2: {"tp": 100, "fp": 48},
    3: {"tp": 101, "fp": 52},
    4: {"tp": 108, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd"
)
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
OPERATOR_ORDER = ("cd", "ad", "rd")
AD_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)
RD_POSITIVE = (0, 2, 4, 10, 14, 20, 22, 24)
RD_NEGATIVE = (6, 7, 8, 11, 13, 16, 17, 18)
FEATURE_SUFFIXES = (
    "core_mean_abs",
    "core_rms",
    "boundary_mean_abs",
    "boundary_rms",
    "outside_mean_abs",
    "outside_rms",
    "boundary_core_contrast",
    "core_outside_contrast",
)
FEATURE_NAMES = tuple(
    f"block{block}_{suffix}"
    for block in range(1, 4)
    for suffix in FEATURE_SUFFIXES
)
OBJECT_FEATURE_INDICES = tuple(
    block * len(FEATURE_SUFFIXES) + offset
    for block in range(3)
    for offset in (0, 1, 2, 3, 6)
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "af2a2ba84cf0aca09873ea6070872d48917bd10a4eb251ecfd5669f7457adcaa"
LOCKED_PAPER_SHA256 = "7ac637512d852cee0ba6cf2d475c3bb06ded4c34431be4b1d93cea74e1da4e09"
LOCKED_OFFICIAL_COMMIT = "d21aa881ed9c628571636fad39acfe1fad517ebd"
LOCKED_OFFICIAL_TREE = "b57c16070137773a76d09356df70dd973632887f"
LOCKED_OFFICIAL_HASHES = {
    "ops": "b71294df26463b3caf075f1cc596e9849b2859f35db230346ed7f02f35a4ba34",
    "ops_theta": "8672cc4732c50bb1b8245f53d47a9535d59246f9ec206628022f36a8184ca2cb",
    "config": "5790cb9469694f7c3f2e690b8f9964b0fbc5f8e82ac3e18b9d28ad208f01a752",
    "convert": "72cd01fe9e8e1ba4ffb599e0b1c1c5cf83e626971029b9be37b4aefb9d9ad519",
    "pidinet": "ee4243d1537f95a8a9ab795e47380115886e4984415c1586c2abcd913bfb8c7b",
    "license": "81ae4ff9ec8a220b015473aba1adbe1ee1c2e9cae844f8dc57fd5848023ba2c1",
}
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked fit-only PiDiNet PDC frozen-stem signal gate. "
            "Holdout, validation, test, and training are forbidden."
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
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_PIXEL_DIFFERENCE_STEM_SIGNAL_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\pidinet_iccv2021.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\pidinet-iccv2021"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_pixel_difference_stem_signal_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=BENCHMARK_REPEATS
    )
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.seed) == SEED
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().casefold():
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tracked_worktree_clean(root: Path) -> bool:
    return not _git_value(root, "status", "--porcelain", "--untracked-files=no")


def _full_worktree_clean(root: Path) -> bool:
    return not _git_value(root, "status", "--porcelain")


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "ops": official / "models" / "ops.py",
        "ops_theta": official / "models" / "ops_theta.py",
        "config": official / "models" / "config.py",
        "convert": official / "models" / "convert_pidinet.py",
        "pidinet": official / "models" / "pidinet.py",
        "license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _cohort_label(row: CleanTrainRow) -> Optional[str]:
    if row.fold not in FIT_FOLDS or row.keeper_prediction != FOCUS_CLASS:
        return None
    if row.target == FOCUS_CLASS:
        return "tp"
    if row.target in RESTRICTED_NEGATIVE_CLASSES:
        return "fp"
    return None


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], list[CleanTrainRow]]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted PDC A0 protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
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
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "PDC protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "PiDiNet paper"
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    for name, expected in LOCKED_OFFICIAL_HASHES.items():
        hashes[name] = _verify_sha256(
            paths[name], expected, f"official PiDiNet {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(paths["official_root"], "rev-parse", "HEAD^{tree}")
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official PiDiNet commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(
            f"Official PiDiNet tree differs: {official_tree} != {LOCKED_OFFICIAL_TREE}"
        )
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official PiDiNet worktree must be clean.")

    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test-data use.")
    if bool(
        cidt_summary.get(
            "validation_predictions_used",
            cidt_summary.get("validation_data_used", True),
        )
    ):
        raise ValueError("CIDT provenance indicates validation-data use.")

    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohort = [row for row in rows if _cohort_label(row) is not None]
    if len(cohort) != EXPECTED_COHORT_ROWS:
        raise ValueError(
            f"Locked PDC cohort differs: {len(cohort)} != {EXPECTED_COHORT_ROWS}"
        )
    positives = sum(_cohort_label(row) == "tp" for row in cohort)
    negatives = sum(_cohort_label(row) == "fp" for row in cohort)
    if (positives, negatives) != (EXPECTED_POSITIVES, EXPECTED_NEGATIVES):
        raise ValueError(
            "Locked PDC class counts differ: "
            f"{positives}/{negatives} != {EXPECTED_POSITIVES}/{EXPECTED_NEGATIVES}"
        )
    fold_counts: Dict[int, Dict[str, int]] = {
        fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS
    }
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked PDC fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(
            f"Locked cohort index hash differs: {ordered_hash}"
        )

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for formal PDC A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal PDC A0 requires the exact pushed repository commit.")

    return (
        {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_worktree_clean": True,
            "repository_commit": repository_commit,
            "upstream_commit": upstream_commit,
            "tracked_worktree_clean": True,
            "ordered_cohort_index_sha256": ordered_hash,
            "cohort_rows": len(cohort),
            "positive_rows": positives,
            "negative_rows": negatives,
            "fold_counts": fold_counts,
            "validation_predictions_used": False,
            "holdout_data_used": False,
            "test_data_used": False,
            "training_used": False,
        },
        rows,
        cohort,
    )


def convert_cd_weight(weight: Tensor) -> Tensor:
    if tuple(weight.shape[-2:]) != (3, 3):
        raise ValueError("CD requires a 3x3 source kernel.")
    center = weight.new_zeros((1, 1, 3, 3))
    center[..., 1, 1] = 1
    return weight - weight.sum(dim=(-2, -1), keepdim=True) * center


def convert_ad_weight(weight: Tensor) -> Tensor:
    if tuple(weight.shape[-2:]) != (3, 3):
        raise ValueError("AD requires a 3x3 source kernel.")
    flat = weight.flatten(start_dim=-2)
    permutation = torch.tensor(
        AD_PERMUTATION, dtype=torch.long, device=weight.device
    )
    return (flat - flat.index_select(-1, permutation)).reshape_as(weight)


def convert_rd_weight(weight: Tensor) -> Tensor:
    if tuple(weight.shape[-2:]) != (3, 3):
        raise ValueError("RD requires a 3x3 source kernel.")
    source = weight.flatten(start_dim=-2)[..., 1:]
    output = weight.new_zeros((*weight.shape[:-2], 25))
    positive = torch.tensor(RD_POSITIVE, dtype=torch.long, device=weight.device)
    negative = torch.tensor(RD_NEGATIVE, dtype=torch.long, device=weight.device)
    shape = (*source.shape[:-1], len(RD_POSITIVE))
    output = output.scatter(-1, positive.expand(shape), source)
    output = output.scatter_add(-1, negative.expand(shape), -source)
    return output.reshape(*weight.shape[:-2], 5, 5)


def convert_pdc_weight(weight: Tensor, operator: str) -> Tensor:
    normalized = str(operator).strip().casefold()
    if normalized == "cv":
        return weight
    if normalized == "cd":
        return convert_cd_weight(weight)
    if normalized == "ad":
        return convert_ad_weight(weight)
    if normalized == "rd":
        return convert_rd_weight(weight)
    raise ValueError(f"Unknown PDC operator: {operator!r}")


def independent_conversion_oracle(weight: Tensor, operator: str) -> Tensor:
    normalized = str(operator).strip().casefold()
    source = weight.flatten(start_dim=-2)
    if normalized == "cv":
        return weight.clone()
    if normalized == "cd":
        values = [source[..., index] for index in range(9)]
        values[4] = values[4] - sum(values)
        return torch.stack(values, dim=-1).reshape_as(weight)
    if normalized == "ad":
        return torch.stack(
            [
                source[..., index] - source[..., AD_PERMUTATION[index]]
                for index in range(9)
            ],
            dim=-1,
        ).reshape_as(weight)
    if normalized == "rd":
        values = [torch.zeros_like(source[..., 0]) for _ in range(25)]
        for source_index, (positive, negative) in enumerate(
            zip(RD_POSITIVE, RD_NEGATIVE), start=1
        ):
            values[positive] = values[positive] + source[..., source_index]
            values[negative] = values[negative] - source[..., source_index]
        return torch.stack(values, dim=-1).reshape(*weight.shape[:-2], 5, 5)
    raise ValueError(f"Unknown PDC operator: {operator!r}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load locked source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_modules(root: Path):
    package_name = "_trkh_locked_pidinet"
    models_name = f"{package_name}.models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(root).resolve())]
    models = types.ModuleType(models_name)
    models.__path__ = [str(Path(root).resolve() / "models")]
    sys.modules[package_name] = package
    sys.modules[models_name] = models
    ops = _load_module(f"{models_name}.ops", Path(root) / "models" / "ops.py")
    _load_module(f"{models_name}.config", Path(root) / "models" / "config.py")
    convert = _load_module(
        f"{models_name}.convert_pidinet",
        Path(root) / "models" / "convert_pidinet.py",
    )
    return ops, convert


class FrozenStemTrace(nn.Module):
    def __init__(
        self,
        source: HybridConvStem,
        operators: Sequence[str],
    ) -> None:
        super().__init__()
        if len(source.blocks) != 3 or len(operators) != 3:
            raise ValueError("PDC A0 requires exactly three stem blocks/operators.")
        blocks: list[nn.Module] = []
        self.operators = tuple(str(value).casefold() for value in operators)
        for source_block, operator in zip(source.blocks, self.operators):
            if not isinstance(source_block, ConvStemBlock):
                raise TypeError("PDC A0 requires native ConvStemBlock instances.")
            block = copy.deepcopy(source_block)
            source_conv = source_block.block.conv
            if not isinstance(source_conv, nn.Conv2d):
                raise TypeError("Stem block convolution is not Conv2d.")
            if tuple(source_conv.kernel_size) != (3, 3):
                raise ValueError("PDC source stem convolution must be 3x3.")
            converted = convert_pdc_weight(source_conv.weight.detach(), operator)
            kernel_size = int(converted.shape[-1])
            conv = nn.Conv2d(
                source_conv.in_channels,
                source_conv.out_channels,
                kernel_size=kernel_size,
                stride=source_conv.stride,
                padding=kernel_size // 2,
                dilation=source_conv.dilation,
                groups=source_conv.groups,
                bias=source_conv.bias is not None,
                padding_mode=source_conv.padding_mode,
            )
            with torch.no_grad():
                conv.weight.copy_(converted)
                if source_conv.bias is not None and conv.bias is not None:
                    conv.bias.copy_(source_conv.bias)
            block.block.conv = conv
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs: list[Tensor] = []
        hidden = images
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return outputs[0], outputs[1], outputs[2]


class _StemFinal(nn.Module):
    def __init__(self, stem: FrozenStemTrace) -> None:
        super().__init__()
        self.stem = stem

    def forward(self, images: Tensor) -> Tensor:
        return self.stem(images)[-1]


def _load_keeper_model(checkpoint: Mapping[str, object]) -> nn.Module:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper checkpoint lacks valid model config/state.")
    if not isinstance(class_names, list) or len(class_names) != 5:
        raise ValueError("Keeper checkpoint class order is invalid.")
    set_seed(SEED, deterministic=True)
    model = create_model(num_classes=5, model_config=model_config).eval()
    load_model_state(model, dict(model_state), strict=True)
    return model


def _construct_stems(model: nn.Module) -> tuple[FrozenStemTrace, FrozenStemTrace, Dict[str, object]]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem):
        raise TypeError("Keeper does not expose the locked HybridConvStem.")
    native = FrozenStemTrace(stem, ("cv", "cv", "cv"))
    candidate = FrozenStemTrace(stem, OPERATOR_ORDER)
    native_kernels = [
        list(block.block.conv.weight.shape[-2:]) for block in native.blocks
    ]
    candidate_kernels = [
        list(block.block.conv.weight.shape[-2:]) for block in candidate.blocks
    ]
    return native, candidate, {
        "operator_order": list(OPERATOR_ORDER),
        "native_kernel_shapes": native_kernels,
        "candidate_kernel_shapes": candidate_kernels,
        "native_state_sha256": _state_sha256(native),
        "candidate_state_sha256": _state_sha256(candidate),
        "source_parameter_count": sum(value.numel() for value in stem.parameters()),
        "native_parameter_count": sum(value.numel() for value in native.parameters()),
        "candidate_parameter_count": sum(
            value.numel() for value in candidate.parameters()
        ),
        "all_parameters_frozen": all(
            not parameter.requires_grad
            for module in (native, candidate)
            for parameter in module.parameters()
        ),
    }


def _operator_case(
    *,
    operator: str,
    weight: Tensor,
    official_ops,
    official_convert,
) -> Dict[str, object]:
    weight = weight.detach().cpu().float().contiguous()
    out_channels = min(4, int(weight.size(0)))
    in_channels = min(4, int(weight.size(1)))
    weight = weight[:out_channels, :in_channels].clone()
    generator = torch.Generator().manual_seed(SEED + len(operator) + weight.numel())
    images = torch.randn(2, in_channels, 11, 13, generator=generator) * 0.25
    official_function = official_ops.createConvFunc(operator)

    official_weight = official_convert.convert_pdc(operator, weight.clone())
    local_weight = convert_pdc_weight(weight, operator)
    oracle_weight = independent_conversion_oracle(weight, operator)
    padding = int(local_weight.shape[-1] // 2)
    official_output = official_function(
        images, weight, None, 1, 1, 1, 1
    )
    converted_output = F.conv2d(images, local_weight, padding=padding)
    vanilla_output = F.conv2d(images, weight, padding=1)

    images_direct = images.clone().requires_grad_(True)
    weights_direct = weight.clone().requires_grad_(True)
    direct = official_function(images_direct, weights_direct, None, 1, 1, 1, 1)
    upstream = torch.linspace(-0.3, 0.4, steps=direct.numel()).reshape_as(direct)
    direct_loss = (direct * upstream).sum()
    direct_input_grad, direct_weight_grad = torch.autograd.grad(
        direct_loss, (images_direct, weights_direct)
    )

    images_converted = images.clone().requires_grad_(True)
    weights_converted = weight.clone().requires_grad_(True)
    converted = F.conv2d(
        images_converted,
        convert_pdc_weight(weights_converted, operator),
        padding=padding,
    )
    converted_loss = (converted * upstream).sum()
    converted_input_grad, converted_weight_grad = torch.autograd.grad(
        converted_loss, (images_converted, weights_converted)
    )

    return {
        "operator": operator,
        "source_shape": list(weight.shape),
        "converted_shape": list(local_weight.shape),
        "official_conversion_max_abs_error": float(
            (official_weight - local_weight).abs().amax().item()
        ),
        "oracle_conversion_max_abs_error": float(
            (oracle_weight - local_weight).abs().amax().item()
        ),
        "output_max_abs_error": float(
            (official_output - converted_output).abs().amax().item()
        ),
        "input_gradient_max_abs_error": float(
            (direct_input_grad - converted_input_grad).abs().amax().item()
        ),
        "weight_gradient_max_abs_error": float(
            (direct_weight_grad - converted_weight_grad).abs().amax().item()
        ),
        "candidate_vanilla_max_abs_delta": float(
            (converted_output - vanilla_output).abs().amax().item()
        ),
        "candidate_output_rms": float(converted_output.square().mean().sqrt().item()),
        "all_finite": bool(
            torch.isfinite(converted_output).all()
            and torch.isfinite(converted_input_grad).all()
            and torch.isfinite(converted_weight_grad).all()
        ),
    }


def _finite_difference_error(operator: str) -> float:
    generator = torch.Generator().manual_seed(SEED + 701 + len(operator))
    images = (
        torch.randn(1, 2, 7, 8, generator=generator, dtype=torch.float64) * 0.2
    )
    weight = (
        torch.randn(2, 2, 3, 3, generator=generator, dtype=torch.float64) * 0.1
    ).requires_grad_(True)
    converted = convert_pdc_weight(weight, operator)
    output = F.conv2d(images, converted, padding=converted.shape[-1] // 2)
    upstream = torch.linspace(
        -0.2, 0.3, steps=output.numel(), dtype=torch.float64
    ).reshape_as(output)
    loss = (output * upstream).sum()
    gradient = torch.autograd.grad(loss, weight)[0]
    source_index = 1 if operator == "rd" else (4 if operator == "cd" else 2)
    coordinate = (0, 0, source_index // 3, source_index % 3)
    epsilon = 1e-5
    base = weight.detach()
    plus = base.clone()
    minus = base.clone()
    plus[coordinate] += epsilon
    minus[coordinate] -= epsilon

    def value(candidate: Tensor) -> Tensor:
        converted_candidate = convert_pdc_weight(candidate, operator)
        return (
            F.conv2d(
                images,
                converted_candidate,
                padding=converted_candidate.shape[-1] // 2,
            )
            * upstream
        ).sum()

    estimate = (value(plus) - value(minus)) / (2.0 * epsilon)
    return float(abs(estimate.item() - gradient[coordinate].item()))


def _bf16_case(operator: str, weight: Tensor, device: torch.device) -> Dict[str, object]:
    weight = weight.detach().cpu().float()
    weight = weight[: min(4, weight.size(0)), : min(4, weight.size(1))]
    generator = torch.Generator().manual_seed(SEED + 1701 + weight.numel())
    images = torch.randn(
        2, int(weight.size(1)), 13, 15, generator=generator
    ) * 0.25
    weight32 = weight.to(device=device)
    images32 = images.to(device=device)
    converted32 = convert_pdc_weight(weight32, operator)
    reference = F.conv2d(
        images32, converted32, padding=converted32.shape[-1] // 2
    )
    weight_bf16 = weight32.to(dtype=torch.bfloat16).detach().requires_grad_(True)
    images_bf16 = images32.to(dtype=torch.bfloat16).detach().requires_grad_(True)
    converted_bf16 = convert_pdc_weight(weight_bf16, operator)
    output_bf16 = F.conv2d(
        images_bf16,
        converted_bf16,
        padding=converted_bf16.shape[-1] // 2,
    )
    output_bf16.float().square().mean().backward()
    return {
        "max_abs_error": float(
            (reference - output_bf16.float()).abs().amax().item()
        ),
        "output_finite": bool(torch.isfinite(output_bf16).all()),
        "input_gradient_finite": bool(
            images_bf16.grad is not None and torch.isfinite(images_bf16.grad).all()
        ),
        "weight_gradient_finite": bool(
            weight_bf16.grad is not None and torch.isfinite(weight_bf16.grad).all()
        ),
        "weight_gradient_nonzero": bool(
            weight_bf16.grad is not None
            and int(torch.count_nonzero(weight_bf16.grad).item()) > 0
        ),
    }


def _equation_diagnostics(
    stem: HybridConvStem,
    *,
    official_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    official_ops, official_convert = _load_official_modules(official_root)
    generator = torch.Generator().manual_seed(SEED + 501)
    random_weights = {
        operator: torch.randn(4, 3, 3, 3, generator=generator) * 0.08
        for operator in OPERATOR_ORDER
    }
    real_weights = {
        operator: stem.blocks[index].block.conv.weight.detach()
        for index, operator in enumerate(OPERATOR_ORDER)
    }
    cases: list[Dict[str, object]] = []
    for source_name, source_weights in (
        ("random", random_weights),
        ("keeper", real_weights),
    ):
        for operator in OPERATOR_ORDER:
            result = _operator_case(
                operator=operator,
                weight=source_weights[operator],
                official_ops=official_ops,
                official_convert=official_convert,
            )
            result["source"] = source_name
            cases.append(result)
    finite_difference = {
        operator: _finite_difference_error(operator) for operator in OPERATOR_ORDER
    }
    bf16_supported = bool(
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    bf16: Dict[str, object] = {}
    if bf16_supported:
        for operator in OPERATOR_ORDER:
            bf16[operator] = _bf16_case(operator, real_weights[operator], device)
    max_output_error = max(float(row["output_max_abs_error"]) for row in cases)
    max_input_gradient_error = max(
        float(row["input_gradient_max_abs_error"]) for row in cases
    )
    max_weight_gradient_error = max(
        float(row["weight_gradient_max_abs_error"]) for row in cases
    )
    max_official_conversion_error = max(
        float(row["official_conversion_max_abs_error"]) for row in cases
    )
    max_oracle_conversion_error = max(
        float(row["oracle_conversion_max_abs_error"]) for row in cases
    )
    max_finite_difference_error = max(finite_difference.values())
    max_bf16_error = (
        max(float(row["max_abs_error"]) for row in bf16.values())
        if bf16
        else float("inf")
    )
    return {
        "cases": cases,
        "finite_difference_error": finite_difference,
        "bf16_supported": bf16_supported,
        "bf16": bf16,
        "max_output_error": max_output_error,
        "max_input_gradient_error": max_input_gradient_error,
        "max_weight_gradient_error": max_weight_gradient_error,
        "max_official_conversion_error": max_official_conversion_error,
        "max_oracle_conversion_error": max_oracle_conversion_error,
        "max_finite_difference_error": max_finite_difference_error,
        "max_bf16_error": max_bf16_error,
        "all_cases_finite": all(bool(row["all_finite"]) for row in cases),
        "all_operators_nondegenerate": all(
            float(row["candidate_vanilla_max_abs_delta"]) > 1e-4
            and float(row["candidate_output_rms"]) > 1e-6
            for row in cases
        ),
        "all_bf16_gradients_finite": bool(bf16)
        and all(
            bool(row["output_finite"])
            and bool(row["input_gradient_finite"])
            and bool(row["weight_gradient_finite"])
            and bool(row["weight_gradient_nonzero"])
            for row in bf16.values()
        ),
    }


def _build_dataset(
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    data_path: Path,
) -> tuple[MangoYOLOCropDataset, object, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        data_path, class_name_mode="raw", expected_num_classes=5
    )
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper checkpoint.")
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
        raise ValueError("Dataset mapping violates the locked train-only declaration.")
    return dataset, _build_eval_transform(semantics), {
        "rows": len(paths),
        "paths_exact": exact,
        "train_paths_only": train_only,
        "semantics": dict(semantics),
    }


def _make_condition_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    brightness: float,
    contrast: float,
    batch_size: int,
    num_workers: int,
    context: str,
) -> tuple[DataLoader, Dict[str, object]]:
    corruption = (
        IdentityCorruption()
        if math.isclose(brightness, 1.0) and math.isclose(contrast, 1.0)
        else LightingShift(brightness=brightness, contrast=contrast)
    )
    dataset = _SelectedConditionDataset(
        base_dataset,
        indices,
        corruption=corruption,
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            generator=generator,
            **kwargs,
        ),
        summary,
    )


def _replay_keeper_declarations(
    *,
    model: nn.Module,
    loader: DataLoader,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
) -> Dict[str, object]:
    model = model.to(device).eval()
    predicted = _predict(
        model=model,
        loader=loader,
        device=device,
        amp_dtype=_amp_dtype(device),
    )
    mismatches: list[Dict[str, int]] = []
    if len(predicted) != len(cohort):
        raise ValueError("Keeper declaration replay row count differs.")
    for source, observed in zip(cohort, predicted):
        mismatch = bool(
            int(observed["sample_index"]) != source.sample_index
            or int(observed["target"]) != source.target
            or int(observed["prediction"]) != source.keeper_prediction
        )
        if mismatch:
            mismatches.append(
                {
                    "expected_sample_index": source.sample_index,
                    "observed_sample_index": int(observed["sample_index"]),
                    "expected_target": source.target,
                    "observed_target": int(observed["target"]),
                    "expected_prediction": source.keeper_prediction,
                    "observed_prediction": int(observed["prediction"]),
                }
            )
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "rows": len(predicted),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "exact": not mismatches,
        "batch_size": BATCH_SIZE,
    }


def region_masks(bboxes: Tensor, height: int, width: int) -> tuple[Tensor, Tensor, Tensor]:
    if bboxes.ndim != 2 or bboxes.size(1) != 4:
        raise ValueError("Bboxes must be normalized xywh tensors [B,4].")
    boxes = bboxes.to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, box_width, box_height = boxes.unbind(dim=1)
    x1 = (cx - box_width / 2.0).clamp(0.0, 1.0)
    x2 = (cx + box_width / 2.0).clamp(0.0, 1.0)
    y1 = (cy - box_height / 2.0).clamp(0.0, 1.0)
    y2 = (cy + box_height / 2.0).clamp(0.0, 1.0)
    core_x1 = x1 + 0.20 * (x2 - x1)
    core_x2 = x2 - 0.20 * (x2 - x1)
    core_y1 = y1 + 0.20 * (y2 - y1)
    core_y2 = y2 - 0.20 * (y2 - y1)
    x_coordinates = (
        (torch.arange(width, device=boxes.device, dtype=torch.float32) + 0.5)
        / float(width)
    ).view(1, 1, width)
    y_coordinates = (
        (torch.arange(height, device=boxes.device, dtype=torch.float32) + 0.5)
        / float(height)
    ).view(1, height, 1)
    bbox_mask = (
        (x_coordinates >= x1[:, None, None])
        & (x_coordinates < x2[:, None, None])
        & (y_coordinates >= y1[:, None, None])
        & (y_coordinates < y2[:, None, None])
    )
    core_mask = (
        (x_coordinates >= core_x1[:, None, None])
        & (x_coordinates < core_x2[:, None, None])
        & (y_coordinates >= core_y1[:, None, None])
        & (y_coordinates < core_y2[:, None, None])
    )
    boundary_mask = bbox_mask & ~core_mask
    outside_mask = ~bbox_mask
    valid = (
        core_mask.flatten(1).any(dim=1)
        & boundary_mask.flatten(1).any(dim=1)
        & outside_mask.flatten(1).any(dim=1)
    )
    if not bool(valid.all()):
        invalid = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Empty core/boundary/outside region at {height}x{width}: {invalid[:20]}"
        )
    return core_mask, boundary_mask, outside_mask


def _masked_stats(activation: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    expanded = mask[:, None].to(dtype=activation.dtype)
    denominator = expanded.sum(dim=(-2, -1)).squeeze(1) * float(activation.size(1))
    mean_abs = (activation.abs() * expanded).sum(dim=(1, 2, 3)) / denominator
    rms = (
        (activation.square() * expanded).sum(dim=(1, 2, 3)) / denominator
    ).clamp_min(0.0).sqrt()
    return mean_abs, rms


def activation_descriptors(activation: Tensor, bboxes: Tensor) -> Tensor:
    core, boundary, outside = region_masks(
        bboxes, int(activation.size(-2)), int(activation.size(-1))
    )
    core_abs, core_rms = _masked_stats(activation, core)
    boundary_abs, boundary_rms = _masked_stats(activation, boundary)
    outside_abs, outside_rms = _masked_stats(activation, outside)
    epsilon = torch.finfo(activation.dtype).eps
    boundary_core = (boundary_abs - core_abs) / (
        boundary_abs + core_abs + epsilon
    )
    core_outside = (core_abs - outside_abs) / (
        core_abs + outside_abs + epsilon
    )
    return torch.stack(
        (
            core_abs,
            core_rms,
            boundary_abs,
            boundary_rms,
            outside_abs,
            outside_rms,
            boundary_core,
            core_outside,
        ),
        dim=1,
    )


def _extract_features(
    *,
    native: FrozenStemTrace,
    candidate: FrozenStemTrace,
    base_dataset: MangoYOLOCropDataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[int, np.ndarray],
    Dict[int, tuple[float, float, float, float]],
    Dict[str, object],
]:
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    native = native.to(device).eval()
    candidate = candidate.to(device).eval()
    features: Dict[str, Dict[str, np.ndarray]] = {}
    clean_energy: Dict[int, np.ndarray] = {}
    original_bboxes: Dict[int, tuple[float, float, float, float]] = {}
    loader_summaries: Dict[str, object] = {}
    with torch.inference_mode():
        for condition, brightness, contrast in CONDITIONS:
            loader, loader_summary = _make_condition_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=indices,
                brightness=brightness,
                contrast=contrast,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"pdc_a0_{condition}",
            )
            loader_summaries[condition] = loader_summary
            role_batches: Dict[str, list[np.ndarray]] = {"native": [], "pdc": []}
            observed_indices: list[int] = []
            observed_targets: list[int] = []
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                bboxes = metadata.get("bbox")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Feature loader lacks sample_index metadata.")
                if not torch.is_tensor(crop_bboxes) or not torch.is_tensor(bboxes):
                    raise ValueError("Feature loader lacks bbox/crop_bbox metadata.")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                crop_bboxes = crop_bboxes.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                native_outputs = native(images)
                candidate_outputs = candidate(images)
                for role, outputs in (
                    ("native", native_outputs),
                    ("pdc", candidate_outputs),
                ):
                    descriptors = torch.cat(
                        [
                            activation_descriptors(output.float(), crop_bboxes)
                            for output in outputs
                        ],
                        dim=1,
                    )
                    role_batches[role].append(descriptors.cpu().numpy())
                batch_indices = [int(value) for value in sample_indices.tolist()]
                observed_indices.extend(batch_indices)
                observed_targets.extend(int(value) for value in targets.tolist())
                if condition == "clean":
                    energy = candidate_outputs[-1].float().square().mean(dim=1).sqrt()
                    for local_index, sample_index in enumerate(batch_indices):
                        clean_energy[sample_index] = (
                            energy[local_index].cpu().numpy().astype(np.float32)
                        )
                        original_bboxes[sample_index] = tuple(
                            float(value) for value in bboxes[local_index].tolist()
                        )
            if observed_indices != indices or observed_targets != expected_targets:
                raise ValueError(f"{condition} feature mapping differs from cohort.")
            features[condition] = {
                role: np.concatenate(values, axis=0).astype(np.float64)
                for role, values in role_batches.items()
            }
            for role in ("native", "pdc"):
                matrix = features[condition][role]
                if matrix.shape != (EXPECTED_COHORT_ROWS, len(FEATURE_NAMES)):
                    raise ValueError(
                        f"Unexpected {condition}/{role} feature shape: {matrix.shape}"
                    )
                if not bool(np.isfinite(matrix).all()):
                    raise ValueError(f"Non-finite {condition}/{role} descriptors.")
    native.cpu()
    candidate.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    if len(clean_energy) != EXPECTED_COHORT_ROWS:
        raise ValueError("Clean PDC energy map coverage is incomplete.")
    return features, clean_energy, original_bboxes, loader_summaries


def _view_indices(view: str) -> tuple[int, ...]:
    if view == "object":
        return OBJECT_FEATURE_INDICES
    if view == "context":
        return tuple(range(len(FEATURE_NAMES)))
    raise ValueError(f"Unknown descriptor view: {view}")


def _fit_oof_scores(
    *,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    view: str,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    columns = _view_indices(view)
    scores = {
        condition: np.full(labels.size, np.nan, dtype=np.float64)
        for condition, _, _ in CONDITIONS
    }
    diagnostics: list[Dict[str, object]] = []
    for fold in FIT_FOLDS:
        fit_mask = folds != fold
        holdout_mask = folds == fold
        if set(labels[fit_mask].tolist()) != {0, 1} or set(
            labels[holdout_mask].tolist()
        ) != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both binary classes.")
        scaler = StandardScaler()
        clean_fit = scaler.fit_transform(features["clean"][fit_mask][:, columns])
        classifier = LogisticRegression(
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            fit_intercept=True,
            max_iter=1000,
            tol=1e-6,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(clean_fit, labels[fit_mask])
        convergence_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        converged = bool(
            not convergence_warnings and int(classifier.n_iter_[0]) < 1000
        )
        if not converged:
            raise ValueError(f"Locked logistic readout did not converge in fold {fold}.")
        for condition, _, _ in CONDITIONS:
            transformed = scaler.transform(
                features[condition][holdout_mask][:, columns]
            )
            scores[condition][holdout_mask] = classifier.predict_proba(transformed)[
                :, 1
            ]
        diagnostics.append(
            {
                "fold": fold,
                "fit_rows": int(fit_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "iterations": int(classifier.n_iter_[0]),
                "converged": converged,
                "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    if not all(bool(np.isfinite(values).all()) for values in scores.values()):
        raise ValueError(f"OOF scores are incomplete for {view} view.")
    return scores, {
        "view": view,
        "feature_count": len(columns),
        "feature_names": [FEATURE_NAMES[index] for index in columns],
        "folds": diagnostics,
    }


def _metrics_from_scores(
    scores: Mapping[str, np.ndarray], labels: np.ndarray
) -> Dict[str, object]:
    clean_tp_scores = np.asarray(scores["clean"])[labels == 1]
    threshold = float(np.quantile(clean_tp_scores, 0.03, method="lower"))
    conditions: Dict[str, object] = {}
    for condition, _, _ in CONDITIONS:
        values = np.asarray(scores[condition], dtype=np.float64)
        tp_values = values[labels == 1]
        fp_values = values[labels == 0]
        conditions[condition] = {
            "auroc": float(roc_auc_score(labels, values)),
            "tp_retention": float(np.mean(tp_values >= threshold)),
            "fp_rejection": float(np.mean(fp_values < threshold)),
            "tp_median_score": float(np.median(tp_values)),
            "fp_median_score": float(np.median(fp_values)),
            "tp_min_score": float(np.min(tp_values)),
            "tp_max_score": float(np.max(tp_values)),
            "fp_min_score": float(np.min(fp_values)),
            "fp_max_score": float(np.max(fp_values)),
        }
    return {"threshold": threshold, "conditions": conditions}


def _run_readouts(
    *,
    features: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    folds: np.ndarray,
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, object],
    Dict[str, object],
]:
    all_scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    metrics: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {}
    for role in ("native", "pdc"):
        role_features = {
            condition: features[condition][role] for condition, _, _ in CONDITIONS
        }
        all_scores[role] = {}
        metrics[role] = {}
        diagnostics[role] = {}
        for view in ("object", "context"):
            scores, readout_diagnostics = _fit_oof_scores(
                features=role_features,
                labels=labels,
                folds=folds,
                view=view,
            )
            all_scores[role][view] = scores
            metrics[role][view] = _metrics_from_scores(scores, labels)
            diagnostics[role][view] = readout_diagnostics
    return all_scores, metrics, diagnostics


def _response_comparisons(
    pdc_clean_features: np.ndarray, labels: np.ndarray
) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    passing_blocks = 0
    for block in range(3):
        offset = block * len(FEATURE_SUFFIXES)
        boundary = pdc_clean_features[:, offset + 2]
        outside = pdc_clean_features[:, offset + 4]
        cohort_rows: Dict[str, object] = {}
        both_pass = True
        for cohort_name, label in (("tp", 1), ("fp", 0)):
            selected = labels == label
            boundary_median = float(np.median(boundary[selected]))
            outside_median = float(np.median(outside[selected]))
            passed = boundary_median > outside_median
            both_pass = bool(both_pass and passed)
            cohort_rows[cohort_name] = {
                "boundary_median": boundary_median,
                "outside_median": outside_median,
                "boundary_gt_outside": passed,
            }
        passing_blocks += int(both_pass)
        rows[f"block{block + 1}"] = {
            "cohorts": cohort_rows,
            "both_cohorts_pass": both_pass,
        }
    return {"blocks": rows, "passing_block_count": passing_blocks}


def assess_information_gate(
    *,
    metrics: Mapping[str, object],
    response_comparisons: Mapping[str, object],
) -> Dict[str, object]:
    native_object = metrics["native"]["object"]["conditions"]
    pdc_object = metrics["pdc"]["object"]["conditions"]
    pdc_context = metrics["pdc"]["context"]["conditions"]
    clean_delta = float(pdc_object["clean"]["auroc"]) - float(
        native_object["clean"]["auroc"]
    )
    shifts = [condition for condition, _, _ in CONDITIONS if condition != "clean"]
    shifted_deltas = {
        condition: float(pdc_object[condition]["auroc"])
        - float(native_object[condition]["auroc"])
        for condition in shifts
    }
    checks = {
        "object_clean_auroc_gte_0p64": float(pdc_object["clean"]["auroc"])
        >= 0.64,
        "object_clean_auroc_delta_gte_0p025": clean_delta >= 0.025,
        "minimum_shifted_auroc_gte_0p58": min(
            float(pdc_object[condition]["auroc"]) for condition in shifts
        )
        >= 0.58,
        "pdc_no_worse_in_two_of_three_shifts": sum(
            delta >= 0.0 for delta in shifted_deltas.values()
        )
        >= 2,
        "worst_shifted_delta_gte_minus_0p015": min(shifted_deltas.values())
        >= -0.015,
        "clean_tp_retention_gte_0p97": float(
            pdc_object["clean"]["tp_retention"]
        )
        >= 0.97,
        "clean_fp_rejection_gte_0p20": float(
            pdc_object["clean"]["fp_rejection"]
        )
        >= 0.20,
        "clean_fp_rejection_delta_gte_0p08": float(
            pdc_object["clean"]["fp_rejection"]
        )
        - float(native_object["clean"]["fp_rejection"])
        >= 0.08,
        "all_shifted_tp_retention_gte_0p90": all(
            float(pdc_object[condition]["tp_retention"]) >= 0.90
            for condition in shifts
        ),
        "all_shifted_fp_rejection_gte_0p10": all(
            float(pdc_object[condition]["fp_rejection"]) >= 0.10
            for condition in shifts
        ),
        "aggregate_shifted_fp_rejection_gt_native": statistics.mean(
            float(pdc_object[condition]["fp_rejection"]) for condition in shifts
        )
        > statistics.mean(
            float(native_object[condition]["fp_rejection"]) for condition in shifts
        ),
        "tp_median_gt_fp_every_condition": all(
            float(pdc_object[condition]["tp_median_score"])
            > float(pdc_object[condition]["fp_median_score"])
            for condition, _, _ in CONDITIONS
        ),
        "object_clean_within_0p02_of_context": float(
            pdc_object["clean"]["auroc"]
        )
        >= float(pdc_context["clean"]["auroc"]) - 0.02,
        "object_pdc_beats_object_native_by_margin": clean_delta >= 0.025,
        "two_blocks_boundary_gt_outside_both_cohorts": int(
            response_comparisons["passing_block_count"]
        )
        >= 2,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "clean_object_auroc_delta": clean_delta,
        "shifted_object_auroc_deltas": shifted_deltas,
    }


def _benchmark_one(
    stem: FrozenStemTrace,
    *,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    model = _StemFinal(copy.deepcopy(stem)).to(device).eval()
    generator = torch.Generator().manual_seed(SEED + 2401)
    images = torch.randn(BATCH_SIZE, 3, 256, 256, generator=generator).to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(2):
            model(images)
        torch.cuda.synchronize(device)
        elapsed: list[float] = []
        for _ in range(int(repeats)):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            model(images)
            finished.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(started.elapsed_time(finished)))
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    model.cpu()
    del model, images
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "repeats": int(repeats),
        "elapsed_ms": elapsed,
        "median_ms": float(statistics.median(elapsed)),
        "peak_allocated_bytes": peak_bytes,
    }


def _benchmark(
    native: FrozenStemTrace,
    candidate: FrozenStemTrace,
    *,
    device: torch.device,
    repeats: int,
) -> Dict[str, object]:
    native_result = _benchmark_one(native, device=device, repeats=repeats)
    candidate_result = _benchmark_one(candidate, device=device, repeats=repeats)
    runtime_ratio = float(candidate_result["median_ms"]) / float(
        native_result["median_ms"]
    )
    memory_ratio = float(candidate_result["peak_allocated_bytes"]) / float(
        native_result["peak_allocated_bytes"]
    )
    return {
        "native": native_result,
        "pdc": candidate_result,
        "runtime_ratio": runtime_ratio,
        "peak_memory_ratio": memory_ratio,
    }


def _onnx_export(
    candidate: FrozenStemTrace,
    *,
    output_path: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    model = copy.deepcopy(candidate).cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 3101)
    example = torch.randn(
        STATIC_EXPORT_BATCH_SIZE, 3, 256, 256, generator=generator
    )
    with torch.inference_mode():
        reference = tuple(value.numpy() for value in model(example))
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["images"],
        output_names=["block1", "block2", "block3"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {"images": example.numpy()})
    errors = [
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(reference, observed)
    ]
    shape_match = [list(value.shape) for value in reference] == [
        list(value.shape) for value in observed
    ]
    domains = sorted({str(node.domain) for node in graph.graph.node})
    operators = sorted({str(node.op_type) for node in graph.graph.node})
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "opset": 17,
        "domains": domains,
        "operators": operators,
        "standard_domains_only": all(domain in {"", "ai.onnx"} for domain in domains),
        "output_shapes_match": shape_match,
        "per_output_max_abs_error": errors,
        "max_abs_error": max(errors),
    }


def _write_cohort_csv(
    *,
    path: Path,
    cohort: Sequence[CleanTrainRow],
    bboxes: Mapping[int, tuple[float, float, float, float]],
    features: Mapping[str, Mapping[str, np.ndarray]],
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    fieldnames = [
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        "cohort",
        "condition",
        "role",
        "bbox_cx",
        "bbox_cy",
        "bbox_width",
        "bbox_height",
        *FEATURE_NAMES,
        "object_score",
        "context_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for condition, _, _ in CONDITIONS:
            for role in ("native", "pdc"):
                matrix = features[condition][role]
                for position, source in enumerate(cohort):
                    bbox = bboxes[source.sample_index]
                    row: Dict[str, object] = {
                        "sample_index": source.sample_index,
                        "source_stem": source.source_stem,
                        "image_path": str(source.image_path),
                        "fold": source.fold,
                        "target": source.target,
                        "keeper_prediction": source.keeper_prediction,
                        "cohort": str(_cohort_label(source)),
                        "condition": condition,
                        "role": role,
                        "bbox_cx": bbox[0],
                        "bbox_cy": bbox[1],
                        "bbox_width": bbox[2],
                        "bbox_height": bbox[3],
                        "object_score": float(scores[role]["object"][condition][position]),
                        "context_score": float(scores[role]["context"][condition][position]),
                    }
                    row.update(
                        {
                            name: float(matrix[position, index])
                            for index, name in enumerate(FEATURE_NAMES)
                        }
                    )
                    writer.writerow(row)


def _replay_csv(path: Path) -> Dict[str, object]:
    records: list[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_index",
            "fold",
            "cohort",
            "condition",
            "role",
            "object_score",
            "context_score",
            *FEATURE_NAMES,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PDC replay CSV misses columns: {sorted(missing)}")
        records.extend(dict(row) for row in reader)
    expected_rows = EXPECTED_COHORT_ROWS * len(CONDITIONS) * 2
    if len(records) != expected_rows:
        raise ValueError(f"PDC replay CSV rows differ: {len(records)} != {expected_rows}")
    clean_native = [
        row
        for row in records
        if row["condition"] == "clean" and row["role"] == "native"
    ]
    ordered_indices = [int(row["sample_index"]) for row in clean_native]
    if _ordered_index_sha256(ordered_indices) != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError("Replay CSV ordered cohort index hash differs.")
    labels = np.asarray(
        [1 if row["cohort"] == "tp" else 0 for row in clean_native],
        dtype=np.int64,
    )
    metrics: Dict[str, object] = {}
    for role in ("native", "pdc"):
        metrics[role] = {}
        for view in ("object", "context"):
            score_key = f"{view}_score"
            score_table: Dict[str, np.ndarray] = {}
            for condition, _, _ in CONDITIONS:
                selected = [
                    row
                    for row in records
                    if row["role"] == role and row["condition"] == condition
                ]
                if [int(row["sample_index"]) for row in selected] != ordered_indices:
                    raise ValueError(f"Replay row order differs for {role}/{condition}.")
                score_table[condition] = np.asarray(
                    [float(row[score_key]) for row in selected], dtype=np.float64
                )
            metrics[role][view] = _metrics_from_scores(score_table, labels)
    clean_pdc = [
        row
        for row in records
        if row["condition"] == "clean" and row["role"] == "pdc"
    ]
    clean_features = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in clean_pdc],
        dtype=np.float64,
    )
    responses = _response_comparisons(clean_features, labels)
    information = assess_information_gate(
        metrics=metrics, response_comparisons=responses
    )
    return {
        "rows": len(records),
        "metrics": metrics,
        "response_comparisons": responses,
        "information_gate": information,
    }


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _bbox_overlay(
    image: Image.Image, bbox: tuple[float, float, float, float]
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    cx, cy, width, height = bbox
    x1 = (cx - width / 2.0) * output.width
    x2 = (cx + width / 2.0) * output.width
    y1 = (cy - height / 2.0) * output.height
    y2 = (cy + height / 2.0) * output.height
    draw.rectangle((x1, y1, x2, y2), outline=(220, 45, 45), width=4)
    core = (
        x1 + 0.20 * (x2 - x1),
        y1 + 0.20 * (y2 - y1),
        x2 - 0.20 * (x2 - x1),
        y2 - 0.20 * (y2 - y1),
    )
    draw.rectangle(core, outline=(30, 190, 80), width=4)
    return output


def _energy_image(energy: np.ndarray) -> Image.Image:
    values = np.asarray(energy, dtype=np.float32)
    lower, upper = np.percentile(values, (2.0, 98.0))
    normalized = np.clip((values - lower) / max(float(upper - lower), 1e-8), 0, 1)
    red = np.clip(1.8 * normalized, 0, 1)
    green = np.clip(1.8 - np.abs(normalized - 0.55) * 3.2, 0, 1)
    blue = np.clip(1.4 * (1.0 - normalized), 0, 1)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB")


def _render_contact_sheets(
    *,
    output_dir: Path,
    cohort: Sequence[CleanTrainRow],
    bboxes: Mapping[int, tuple[float, float, float, float]],
    energy: Mapping[int, np.ndarray],
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> Dict[str, object]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    clean_scores = scores["pdc"]["object"]["clean"]
    selected_positions: list[int] = []
    selection_labels: list[str] = []
    for cohort_name, label in (("tp", 1), ("fp", 0)):
        positions = np.flatnonzero(labels == label)
        ordered = positions[np.argsort(clean_scores[positions], kind="mergesort")]
        for rank_name, values in (("lowest", ordered[:4]), ("highest", ordered[-4:])):
            selected_positions.extend(int(value) for value in values)
            selection_labels.extend(
                f"{cohort_name}_{rank_name}_{rank + 1}"
                for rank in range(len(values))
            )
    pages: list[Dict[str, object]] = []
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(selected_positions), 4), start=1):
        page_positions = selected_positions[start : start + 4]
        page_labels = selection_labels[start : start + 4]
        page = Image.new("RGB", (1280, 1080), (245, 245, 245))
        draw = ImageDraw.Draw(page)
        for panel_index, (position, selection) in enumerate(
            zip(page_positions, page_labels)
        ):
            source = cohort[position]
            top = panel_index * 270
            with Image.open(source.image_path) as handle:
                source_image = ImageOps.exif_transpose(handle).convert("RGB")
            overlay = _fit_image(
                _bbox_overlay(source_image, bboxes[source.sample_index]), (300, 240)
            )
            heat = _fit_image(_energy_image(energy[source.sample_index]), (300, 240))
            page.paste(overlay, (10, top + 20))
            page.paste(heat, (320, top + 20))
            lines = [
                f"{selection} | sample={source.sample_index} fold={source.fold}",
                f"target={source.target} keeper={source.keeper_prediction} source={source.source_stem}",
                "red=bbox green=core | right=block3 PDC RMS energy",
            ]
            for condition, _, _ in CONDITIONS:
                pdc_score = float(scores["pdc"]["object"][condition][position])
                native_score = float(
                    scores["native"]["object"][condition][position]
                )
                lines.append(
                    f"{condition}: pdc={pdc_score:.6f} native={native_score:.6f}"
                )
            lines.append(str(source.image_path))
            draw.multiline_text(
                (635, top + 25),
                "\n".join(lines),
                fill=(20, 20, 20),
                font=font,
                spacing=8,
            )
        path = output_dir / f"pdc_signal_contact_sheet_{page_index:02d}.png"
        page.save(path)
        pages.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "selections": page_labels,
                "sample_indices": [cohort[value].sample_index for value in page_positions],
            }
        )
    return {
        "page_count": len(pages),
        "selected_rows": len(selected_positions),
        "pages": pages,
    }


def _canonical_equal(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    metrics = summary["readout_metrics"]
    native = metrics["native"]["object"]["conditions"]
    pdc = metrics["pdc"]["object"]["conditions"]
    lines = [
        "# Pixel-Difference Stem Signal A0 Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Matched 5e pair authorized: `{summary['matched_5e_pair_authorized']}`",
        f"- Structural failed checks: `{summary['structural_gate']['failed_checks']}`",
        f"- Information failed checks: `{summary['information_gate']['failed_checks']}`",
        "",
        "## Object-Only OOF Metrics",
        "",
        "| Condition | Native AUROC | PDC AUROC | Native FP reject | PDC FP reject | PDC TP retain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, _, _ in CONDITIONS:
        lines.append(
            "| {condition} | {native_auc:.6f} | {pdc_auc:.6f} | "
            "{native_fp:.6f} | {pdc_fp:.6f} | {pdc_tp:.6f} |".format(
                condition=condition,
                native_auc=float(native[condition]["auroc"]),
                pdc_auc=float(pdc[condition]["auroc"]),
                native_fp=float(native[condition]["fp_rejection"]),
                pdc_fp=float(pdc[condition]["fp_rejection"]),
                pdc_tp=float(pdc[condition]["tp_retention"]),
            )
        )
    lines.extend(
        [
            "",
            "## Deployment",
            "",
            f"- Runtime ratio: `{summary['resource_audit']['runtime_ratio']:.6f}`",
            f"- Peak-memory ratio: `{summary['resource_audit']['peak_memory_ratio']:.6f}`",
            f"- ONNX max error: `{summary['onnx_audit']['max_abs_error']:.8g}`",
            "",
            "No holdout, validation, test, image training, or current-command update was used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    artifacts = []
    forbidden = {".pt", ".pth", ".ckpt", ".engine"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden model artifact in PDC A0 output: {path}")
        artifacts.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    provenance, rows, cohort = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
        }

    output_dir = _prepare_output_dir(args.output_dir)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Locked PDC A0 requires CUDA.")
    set_seed(int(args.seed), deterministic=True)
    checkpoint = load_checkpoint(
        Path(provenance["paths"]["checkpoint"]), map_location="cpu"
    )
    model = _load_keeper_model(checkpoint)
    source_stem = getattr(model, "stem", None)
    if not isinstance(source_stem, HybridConvStem):
        raise TypeError("Keeper stem differs from locked HybridConvStem.")
    equation = _equation_diagnostics(
        source_stem,
        official_root=Path(provenance["paths"]["official_root"]),
        device=device,
    )
    native, candidate, stem_structure = _construct_stems(model)
    resource = _benchmark(
        native,
        candidate,
        device=device,
        repeats=int(args.benchmark_repeats),
    )
    onnx_path = output_dir / "pixel_difference_converted_stem.onnx"
    onnx_audit = _onnx_export(candidate, output_path=onnx_path)

    base_dataset, transform, dataset_mapping = _build_dataset(
        checkpoint,
        rows,
        Path(provenance["paths"]["data"]),
    )
    cohort_indices = [row.sample_index for row in cohort]
    clean_loader, declaration_loader = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=cohort_indices,
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="pdc_a0_keeper_declaration_replay",
    )
    declaration_replay = _replay_keeper_declarations(
        model=model,
        loader=clean_loader,
        cohort=cohort,
        device=device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    features, energy, bboxes, feature_loaders = _extract_features(
        native=native,
        candidate=candidate,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    scores, readout_metrics, readout_diagnostics = _run_readouts(
        features=features,
        labels=labels,
        folds=folds,
    )
    response_comparisons = _response_comparisons(features["clean"]["pdc"], labels)
    information_gate = assess_information_gate(
        metrics=readout_metrics,
        response_comparisons=response_comparisons,
    )

    cohort_csv = output_dir / "cohort_scores_and_features.csv"
    _write_cohort_csv(
        path=cohort_csv,
        cohort=cohort,
        bboxes=bboxes,
        features=features,
        scores=scores,
    )
    replay = _replay_csv(cohort_csv)
    replay_metrics_exact = _canonical_equal(
        readout_metrics, replay["metrics"]
    )
    replay_responses_exact = _canonical_equal(
        response_comparisons, replay["response_comparisons"]
    )
    replay_gate_exact = _canonical_equal(
        information_gate, replay["information_gate"]
    )
    contact_sheets = _render_contact_sheets(
        output_dir=output_dir,
        cohort=cohort,
        bboxes=bboxes,
        energy=energy,
        scores=scores,
    )

    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_and_hashes_exact": True,
        "official_and_trkh_worktrees_clean": bool(
            provenance["official_worktree_clean"]
            and provenance["tracked_worktree_clean"]
        ),
        "repository_commit_pushed": provenance["repository_commit"]
        == provenance["upstream_commit"],
        "cohort_count_fold_and_order_exact": bool(
            provenance["cohort_rows"] == EXPECTED_COHORT_ROWS
            and provenance["positive_rows"] == EXPECTED_POSITIVES
            and provenance["negative_rows"] == EXPECTED_NEGATIVES
            and provenance["ordered_cohort_index_sha256"]
            == EXPECTED_ORDERED_INDEX_SHA256
        ),
        "keeper_declarations_exact_batch64": bool(declaration_replay["exact"]),
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"] and dataset_mapping["train_paths_only"]
        ),
        "official_and_oracle_conversion_error_lte_1e6": max(
            float(equation["max_official_conversion_error"]),
            float(equation["max_oracle_conversion_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "forward_error_lte_1e6": float(equation["max_output_error"])
        <= MAX_EQUATION_ERROR,
        "input_gradient_error_lte_1e6": float(
            equation["max_input_gradient_error"]
        )
        <= MAX_EQUATION_ERROR,
        "weight_gradient_error_lte_1e6": float(
            equation["max_weight_gradient_error"]
        )
        <= MAX_EQUATION_ERROR,
        "finite_difference_error_lte_1e4": float(
            equation["max_finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "equations_finite_and_nondegenerate": bool(
            equation["all_cases_finite"]
            and equation["all_operators_nondegenerate"]
        ),
        "bf16_error_lte_0p02_and_gradients_finite": bool(
            equation["bf16_supported"]
            and equation["all_bf16_gradients_finite"]
            and float(equation["max_bf16_error"]) <= MAX_BF16_ERROR
        ),
        "stem_structure_and_freeze_exact": bool(
            stem_structure["operator_order"] == list(OPERATOR_ORDER)
            and stem_structure["native_kernel_shapes"] == [[3, 3], [3, 3], [3, 3]]
            and stem_structure["candidate_kernel_shapes"]
            == [[3, 3], [3, 3], [5, 5]]
            and stem_structure["all_parameters_frozen"]
        ),
        "onnx_standard_ops_shape_and_error": bool(
            onnx_audit["standard_domains_only"]
            and onnx_audit["output_shapes_match"]
            and float(onnx_audit["max_abs_error"]) <= MAX_ONNX_ERROR
        ),
        "runtime_ratio_lte_1p50": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_lte_1p25": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "descriptor_and_readout_replay_exact": bool(
            replay_metrics_exact and replay_responses_exact and replay_gate_exact
        ),
        "contact_sheet_complete": int(contact_sheets["selected_rows"]) == 16
        and int(contact_sheets["page_count"]) >= 1,
        "no_holdout_validation_test_or_training": bool(
            not provenance["holdout_data_used"]
            and not provenance["validation_predictions_used"]
            and not provenance["test_data_used"]
            and not provenance["training_used"]
        ),
    }
    structural_gate = {
        "checks": structural_checks,
        "failed_checks": [
            name for name, passed in structural_checks.items() if not passed
        ],
        "passed": all(structural_checks.values()),
    }
    authorized = bool(structural_gate["passed"] and information_gate["passed"])
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "artifact_manifest.json"
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": (
            "authorized_for_matched_5e_pair" if authorized else "rejected_at_a0"
        ),
        "matched_5e_pair_authorized": authorized,
        "provenance": provenance,
        "stem_structure": stem_structure,
        "equation_audit": equation,
        "dataset_mapping": dataset_mapping,
        "declaration_replay": declaration_replay,
        "loader_audit": {
            "declaration": declaration_loader,
            "features": feature_loaders,
        },
        "readout_diagnostics": readout_diagnostics,
        "readout_metrics": readout_metrics,
        "response_comparisons": response_comparisons,
        "information_gate": information_gate,
        "resource_audit": resource,
        "onnx_audit": onnx_audit,
        "replay_audit": {
            "rows": replay["rows"],
            "metrics_exact": replay_metrics_exact,
            "responses_exact": replay_responses_exact,
            "gate_exact": replay_gate_exact,
        },
        "contact_sheet_audit": contact_sheets,
        "structural_gate": structural_gate,
        "validation_predictions_used": False,
        "holdout_data_used": False,
        "test_data_used": False,
        "training_used": False,
        "current_best_commands_updated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "cohort_csv": str(cohort_csv.resolve()),
            "onnx": str(onnx_path.resolve()),
            "report": str(report_path.resolve()),
            "summary": str(summary_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    manifest = _write_manifest(output_dir)
    summary["artifact_manifest"] = manifest
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(
        json.dumps(
            {
                "status": result["status"],
                "matched_5e_pair_authorized": result[
                    "matched_5e_pair_authorized"
                ],
                "structural_failed_checks": result["structural_gate"][
                    "failed_checks"
                ],
                "information_failed_checks": result["information_gate"][
                    "failed_checks"
                ],
                "summary": result["artifacts"]["summary"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
