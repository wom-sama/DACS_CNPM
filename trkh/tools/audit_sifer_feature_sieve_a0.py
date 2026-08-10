from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    HybridConvStem,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_class1_protected_rsc_readiness import (
    _export_candidate,
    _features_from_batch,
    _parameter_group,
    _parameter_group_hashes,
    _predict_fp32,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_efficienttrain_low_frequency_curriculum_a0 import (
    _write_manifest,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import (
    _make_lighting_loader,
    _write_illumination_predictions,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _comparison,
    _make_loader,
    _prepare_output_dir,
    _rng_snapshot,
    _rng_summary,
    _sha256,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_FIT_ROWS = 7372
EXPECTED_TRAIN_BATCHES = 60
EXPECTED_TRAIN_BUDGET_ROWS = 1920
EXPECTED_HOLDOUT_CLASS_COUNTS = (380, 109, 393, 503, 458)
EXPECTED_STEM_CHANNELS = 256
EXPECTED_STEM_GRID = 32
EXPECTED_AUXILIARY_PARAMETERS = 2_362_629
EXPECTED_FORGET_STEPS = tuple(range(0, EXPECTED_TRAIN_BATCHES, 5))

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "90588283fd200ad0f39b2622433091dd4db0c900d77d479eab2d7c8292c235bc"
LOCKED_SIFER_PAPER_SHA256 = "49a64b977dcbabfed14473639a211e1385ea71da6ffb75cf19c8065b46ec7959"
LOCKED_SIFER_COMMIT = "76b0612b5b48acfa53eba6b433e98764805286f4"
LOCKED_SIFER_TREE = "890e37473fb42a9272e6b5478c23e70e6d3c0a57"
LOCKED_SIFER_ALGORITHM_BLOB = "43595d17c2ddbca3d50f95bde0189228b92da1aa"
LOCKED_SIFER_NETWORK_BLOB = "cf59f5414e1168edd7c45214545eb91c6c52e3cf"
LOCKED_SIFER_PARAMS_BLOB = "5d3aefd418843cb6ef4d61905e26933427170a96"
LOCKED_LICENSE_BLOB = "d645695673349e3947e8e5ae42332d0ac3164cd7"

MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.60
MAX_PEAK_VRAM_GIB = 6.50


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only A0 audit for an ICML-2023 SIFER feature sieve. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/launcher_args.json"
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
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_SIFER_FEATURE_SIEVE_A0_PROTOCOL_20260719.md"
        ),
    )
    parser.add_argument(
        "--sifer-git-dir",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_sifer_official_76b0612_bare.git"
        ),
    )
    parser.add_argument(
        "--sifer-paper",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_sifer_icml2023_tiwari23a.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_sifer_feature_sieve_a0_20260719"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument(
        "--finalize-visual-review", action="store_true", default=False
    )
    parser.add_argument(
        "--visual-review-result", choices=("pass", "fail"), default="fail"
    )
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--main-learning-rate", type=float, default=1e-5)
    parser.add_argument("--main-weight-decay", type=float, default=0.05)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-2)
    parser.add_argument("--aux-weight-decay", type=float, default=1e-4)
    parser.add_argument("--forget-learning-rate", type=float, default=1e-4)
    parser.add_argument("--forget-interval", type=int, default=5)
    parser.add_argument("--aux-seed", type=int, default=42042)
    parser.add_argument("--aux-depth", type=int, default=2)
    parser.add_argument("--aux-width", type=int, default=256)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--contact-rows", type=int, default=8)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 32
        and int(args.num_workers) == 4
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == EXPECTED_TRAIN_BATCHES
        and math.isclose(float(args.main_learning_rate), 1e-5, abs_tol=1e-12)
        and math.isclose(float(args.main_weight_decay), 0.05, abs_tol=1e-12)
        and math.isclose(float(args.aux_learning_rate), 1e-2, abs_tol=1e-12)
        and math.isclose(float(args.aux_weight_decay), 1e-4, abs_tol=1e-12)
        and math.isclose(float(args.forget_learning_rate), 1e-4, abs_tol=1e-12)
        and int(args.forget_interval) == 5
        and int(args.aux_seed) == 42042
        and int(args.aux_depth) == 2
        and int(args.aux_width) == 256
        and int(args.focus_class) == 1
        and int(args.contact_rows) == 8
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "sifer_git_dir": Path(args.sifer_git_dir).resolve(),
        "sifer_paper": Path(args.sifer_paper).resolve(),
    }


def _git_object(git_dir: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", str(revision)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Unexpected Git rev-parse output for {revision!r}: {lines}")
    return lines[0].casefold()


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("A0 arguments differ from the precommitted SIFER protocol.")
    paths = _source_paths(args)
    if not paths["sifer_git_dir"].is_dir():
        raise FileNotFoundError(
            f"Official SIFER bare repository is missing: {paths['sifer_git_dir']}"
        )
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "SIFER protocol"
        ),
        "sifer_paper": _verify_sha256(
            paths["sifer_paper"], LOCKED_SIFER_PAPER_SHA256, "SIFER paper"
        ),
    }
    objects = {
        "commit": _git_object(paths["sifer_git_dir"], LOCKED_SIFER_COMMIT),
        "tree": _git_object(
            paths["sifer_git_dir"], f"{LOCKED_SIFER_COMMIT}:sifer"
        ),
        "algorithm": _git_object(
            paths["sifer_git_dir"],
            f"{LOCKED_SIFER_COMMIT}:sifer/learning/algorithms.py",
        ),
        "network": _git_object(
            paths["sifer_git_dir"],
            f"{LOCKED_SIFER_COMMIT}:sifer/models/networks.py",
        ),
        "params": _git_object(
            paths["sifer_git_dir"], f"{LOCKED_SIFER_COMMIT}:sifer/params.py"
        ),
        "license": _git_object(
            paths["sifer_git_dir"], f"{LOCKED_SIFER_COMMIT}:LICENSE"
        ),
    }
    expected = {
        "commit": LOCKED_SIFER_COMMIT,
        "tree": LOCKED_SIFER_TREE,
        "algorithm": LOCKED_SIFER_ALGORITHM_BLOB,
        "network": LOCKED_SIFER_NETWORK_BLOB,
        "params": LOCKED_SIFER_PARAMS_BLOB,
        "license": LOCKED_LICENSE_BLOB,
    }
    if objects != expected:
        raise ValueError(f"Official SIFER Git objects differ: {objects} != {expected}")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "git_objects": objects,
        "license": "Apache-2.0",
        "validation_predictions_used": False,
        "test_data_used": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


class SiferBasicBlock(nn.Module):
    """Exact SIFER auxiliary BasicBlock topology, adapted to fixed width."""

    def __init__(self, in_channels: int, channels: int, *, apply_skip: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            int(in_channels), int(channels), kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(int(channels))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            int(channels), int(channels), kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(channels))
        self.apply_skip = bool(apply_skip)

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.apply_skip:
            outputs = outputs + identity
        return self.relu(outputs)


class SiferAuxiliary(nn.Module):
    """Training-only SIFER auxiliary classifier; never part of TRKH inference."""

    def __init__(
        self,
        *,
        in_channels: int = EXPECTED_STEM_CHANNELS,
        width: int = 256,
        depth: int = 2,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        if int(depth) != 2:
            raise ValueError("Locked SIFER auxiliary depth must be two.")
        self.layers = nn.Sequential(
            SiferBasicBlock(int(in_channels), int(width), apply_skip=False),
            SiferBasicBlock(int(width), int(width), apply_skip=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(width), int(num_classes))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.layers(inputs)
        pooled = self.avgpool(features)
        return self.fc(torch.flatten(pooled, 1))


def uniform_cross_entropy(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or int(logits.size(1)) <= 1:
        raise ValueError("Uniform cross entropy expects [B,C] logits with C > 1.")
    values = logits.float()
    targets = torch.full_like(values, 1.0 / float(values.size(1)))
    return F.cross_entropy(values, targets)


def _entropy(logits: Tensor) -> Tensor:
    probabilities = F.softmax(logits.float(), dim=1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def _module_parameter_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_buffer_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_buffers()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_state_finite(module: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(value.detach()).all().item())
        for value in (*module.parameters(), *module.buffers())
    )


@contextmanager
def _isolated_torch_seed(seed: int) -> Iterator[None]:
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        yield


def _equation_diagnostics(args: argparse.Namespace) -> Dict[str, object]:
    logits = torch.tensor(
        [
            [0.2, -0.3, 1.1, 0.4, -0.8],
            [-1.0, 0.5, 0.2, 1.3, -0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([2, 3, 1], dtype=torch.long)
    uniform_targets = torch.full_like(logits, 1.0 / float(logits.size(1)))
    implemented_uniform = uniform_cross_entropy(logits)
    direct_uniform = F.cross_entropy(logits, uniform_targets)
    implemented_hard = F.cross_entropy(logits, targets)
    direct_hard = -F.log_softmax(logits, dim=1).gather(
        1, targets[:, None]
    ).mean()
    total = implemented_uniform + implemented_hard
    gradient = torch.autograd.grad(total, logits)[0]

    with _isolated_torch_seed(int(args.aux_seed)):
        auxiliary = SiferAuxiliary(
            width=int(args.aux_width),
            depth=int(args.aux_depth),
        )
    auxiliary_parameters = sum(int(value.numel()) for value in auxiliary.parameters())
    synthetic = torch.linspace(
        -1.0,
        1.0,
        steps=2 * EXPECTED_STEM_CHANNELS * 7 * 7,
        dtype=torch.float32,
    ).reshape(2, EXPECTED_STEM_CHANNELS, 7, 7)
    auxiliary.eval()
    output = auxiliary(synthetic)
    return {
        "uniform_ce": float(implemented_uniform.item()),
        "uniform_ce_direct": float(direct_uniform.item()),
        "uniform_ce_max_abs_error": float(
            (implemented_uniform - direct_uniform).abs().item()
        ),
        "hard_ce": float(implemented_hard.item()),
        "hard_ce_direct": float(direct_hard.item()),
        "hard_ce_max_abs_error": float(
            (implemented_hard - direct_hard).abs().item()
        ),
        "gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "auxiliary_parameters": auxiliary_parameters,
        "auxiliary_output_shape": list(output.shape),
        "auxiliary_output_finite": bool(torch.isfinite(output).all().item()),
        "forget_steps": list(range(0, int(args.max_train_batches), int(args.forget_interval))),
    }


@contextmanager
def _batchnorm_eval(module: nn.Module) -> Iterator[None]:
    states = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            states.append((child, bool(child.training)))
            child.eval()
    try:
        yield
    finally:
        for child, training in states:
            child.train(training)


def _set_main_forward_rng(seed: int, step: int) -> None:
    role_seed = int(seed) * 1_000_003 + int(step) * 9_973 + 71
    role_seed %= 2**31 - 1
    torch.manual_seed(role_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(role_seed)


def _stem_features(model: nn.Module, images: Tensor) -> Tensor:
    features = model.stem(images)
    if features.ndim != 4 or int(features.size(1)) != EXPECTED_STEM_CHANNELS:
        raise ValueError(
            f"Locked SIFER stem feature differs: {tuple(features.shape)}"
        )
    if tuple(int(value) for value in features.shape[-2:]) != (
        EXPECTED_STEM_GRID,
        EXPECTED_STEM_GRID,
    ):
        raise ValueError(
            f"Locked SIFER stem grid differs: {tuple(features.shape[-2:])}"
        )
    return features


def _make_auxiliary(args: argparse.Namespace, device: torch.device) -> nn.Module:
    with _isolated_torch_seed(int(args.aux_seed)):
        auxiliary = SiferAuxiliary(
            width=int(args.aux_width),
            depth=int(args.aux_depth),
        )
    return auxiliary.to(device)


def _gradient_telemetry(model: nn.Module) -> tuple[Dict[str, float], Dict[str, bool], bool]:
    squared: Dict[str, float] = {}
    seen: Dict[str, bool] = {}
    finite = True
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        gradient = parameter.grad
        if gradient is None:
            seen.setdefault(group, False)
            squared.setdefault(group, 0.0)
            continue
        is_finite = bool(torch.isfinite(gradient).all().item())
        is_nonzero = bool(torch.count_nonzero(gradient).item())
        finite = bool(finite and is_finite)
        seen[group] = bool(seen.get(group, False) or is_nonzero)
        squared[group] = squared.get(group, 0.0) + float(
            gradient.detach().float().square().sum().item()
        )
    return (
        {key: math.sqrt(value) for key, value in sorted(squared.items())},
        seen,
        finite,
    )


def _evaluate_auxiliary(
    *,
    model: nn.Module,
    auxiliary: nn.Module,
    images: Tensor,
    targets: Tensor,
) -> Dict[str, object]:
    model_training = bool(model.training)
    auxiliary_training = bool(auxiliary.training)
    model.train()
    auxiliary.eval()
    with _batchnorm_eval(model.stem), torch.no_grad():
        logits = auxiliary(_stem_features(model, images))
        loss = F.cross_entropy(logits.float(), targets)
        uniform = uniform_cross_entropy(logits)
        entropy = _entropy(logits)
        accuracy = logits.argmax(dim=1).eq(targets).float().mean()
    model.train(model_training)
    auxiliary.train(auxiliary_training)
    return {
        "supervised_ce": float(loss.item()),
        "uniform_ce": float(uniform.item()),
        "entropy_mean": float(entropy.mean().item()),
        "accuracy": float(accuracy.item()),
        "finite": bool(
            torch.isfinite(logits).all().item()
            and torch.isfinite(loss).item()
            and torch.isfinite(uniform).item()
            and torch.isfinite(entropy).all().item()
        ),
    }


def _train_variant(
    *,
    name: str,
    candidate: bool,
    prototype: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, Optional[nn.Module], Dict[str, object]]:
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial_state = _state_sha256(model)
    initial_groups = _parameter_group_hashes(model)
    main_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.main_learning_rate),
        betas=(0.9, 0.999),
        weight_decay=float(args.main_weight_decay),
    )
    auxiliary: Optional[nn.Module] = None
    auxiliary_optimizer: Optional[torch.optim.Optimizer] = None
    forget_optimizer: Optional[torch.optim.Optimizer] = None
    auxiliary_initial_parameter_sha256 = None
    auxiliary_initial_buffer_sha256 = None
    auxiliary_initialization_rng_before = None
    auxiliary_initialization_rng_after = None
    if candidate:
        auxiliary_initialization_rng_before = _rng_summary(_rng_snapshot())
        auxiliary = _make_auxiliary(args, device).train()
        auxiliary_initialization_rng_after = _rng_summary(_rng_snapshot())
        auxiliary_initial_parameter_sha256 = _module_parameter_sha256(auxiliary)
        auxiliary_initial_buffer_sha256 = _module_buffer_sha256(auxiliary)
        auxiliary_optimizer = torch.optim.SGD(
            auxiliary.parameters(),
            lr=float(args.aux_learning_rate),
            momentum=0.0,
            weight_decay=float(args.aux_weight_decay),
        )
        forget_optimizer = torch.optim.SGD(
            model.stem.parameters(),
            lr=float(args.forget_learning_rate),
            momentum=0.0,
            weight_decay=0.0,
        )

    train_loader, train_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"sifer_{name}_fit",
        seed=int(args.seed) + 200,
    )
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"sifer_{name}_holdout",
        seed=int(args.seed) + 100,
    )
    ordered_indices: list[int] = []
    history: list[Dict[str, object]] = []
    main_rng_before: list[Dict[str, object]] = []
    main_rng_after: list[Dict[str, object]] = []
    gradient_seen = {key: False for key in initial_groups}
    all_gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (images, targets, metadata) in enumerate(train_loader):
        if batch_index >= int(args.max_train_batches):
            break
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        ordered_indices.extend(int(value) for value in sample_indices.tolist())
        step_seconds = 0.0
        identify: Optional[Dict[str, object]] = None
        forget: Optional[Dict[str, object]] = None

        if candidate:
            assert auxiliary is not None
            assert auxiliary_optimizer is not None
            assert forget_optimizer is not None
            identify_before = _evaluate_auxiliary(
                model=model,
                auxiliary=auxiliary,
                images=images,
                targets=targets,
            )
            identify_stem_before = _module_parameter_sha256(model.stem)
            identify_auxiliary_before = _module_parameter_sha256(auxiliary)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with _batchnorm_eval(model.stem), torch.no_grad():
                detached_features = _stem_features(model, images).detach()
            auxiliary.train()
            auxiliary_optimizer.zero_grad(set_to_none=True)
            identify_logits = auxiliary(detached_features)
            identify_loss = F.cross_entropy(identify_logits.float(), targets)
            identify_loss.backward()
            auxiliary_gradient_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in auxiliary.parameters()
            )
            auxiliary_gradient_nonzero = any(
                parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in auxiliary.parameters()
            )
            auxiliary_optimizer.step()
            torch.cuda.synchronize(device)
            step_seconds += float(time.perf_counter() - started)
            identify_stem_after = _module_parameter_sha256(model.stem)
            identify_auxiliary_after = _module_parameter_sha256(auxiliary)
            identify_after = _evaluate_auxiliary(
                model=model,
                auxiliary=auxiliary,
                images=images,
                targets=targets,
            )
            identify = {
                "loss": float(identify_loss.detach().item()),
                "before": identify_before,
                "after": identify_after,
                "loss_decreased": float(identify_after["supervised_ce"])
                <= float(identify_before["supervised_ce"]),
                "gradient_finite": auxiliary_gradient_finite,
                "gradient_nonzero": auxiliary_gradient_nonzero,
                "stem_parameters_unchanged": identify_stem_before
                == identify_stem_after,
                "auxiliary_parameters_changed": identify_auxiliary_before
                != identify_auxiliary_after,
            }

            if batch_index % int(args.forget_interval) == 0:
                forget_before = _evaluate_auxiliary(
                    model=model,
                    auxiliary=auxiliary,
                    images=images,
                    targets=targets,
                )
                stem_before = _parameter_group_hashes(model)["stem"]
                nonstem_before = {
                    key: value
                    for key, value in _parameter_group_hashes(model).items()
                    if key != "stem"
                }
                auxiliary_parameter_before = _module_parameter_sha256(auxiliary)
                auxiliary_buffer_before = _module_buffer_sha256(auxiliary)
                model.zero_grad(set_to_none=True)
                auxiliary.zero_grad(set_to_none=True)
                for parameter in auxiliary.parameters():
                    parameter.requires_grad_(False)
                auxiliary.train()
                forget_optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _batchnorm_eval(model.stem):
                    forget_logits = auxiliary(_stem_features(model, images))
                forget_loss = uniform_cross_entropy(forget_logits)
                forget_loss.backward()
                forget_gradient_squared = 0.0
                forget_gradient_finite = True
                forget_gradient_nonzero = False
                for parameter in model.stem.parameters():
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    forget_gradient_finite = bool(
                        forget_gradient_finite
                        and torch.isfinite(gradient).all().item()
                    )
                    forget_gradient_nonzero = bool(
                        forget_gradient_nonzero
                        or torch.count_nonzero(gradient).item()
                    )
                    forget_gradient_squared += float(
                        gradient.detach().float().square().sum().item()
                    )
                nonstem_gradient_absent = all(
                    parameter.grad is None
                    or not bool(torch.count_nonzero(parameter.grad).item())
                    for parameter_name, parameter in model.named_parameters()
                    if not parameter_name.startswith("stem.")
                )
                forget_optimizer.step()
                torch.cuda.synchronize(device)
                step_seconds += float(time.perf_counter() - started)
                for parameter in auxiliary.parameters():
                    parameter.requires_grad_(True)
                forget_after = _evaluate_auxiliary(
                    model=model,
                    auxiliary=auxiliary,
                    images=images,
                    targets=targets,
                )
                stem_after = _parameter_group_hashes(model)["stem"]
                nonstem_after = {
                    key: value
                    for key, value in _parameter_group_hashes(model).items()
                    if key != "stem"
                }
                auxiliary_parameter_after = _module_parameter_sha256(auxiliary)
                auxiliary_buffer_after = _module_buffer_sha256(auxiliary)
                forget = {
                    "step": int(batch_index),
                    "loss": float(forget_loss.detach().item()),
                    "before": forget_before,
                    "after": forget_after,
                    "uniform_ce_nonincreasing": float(forget_after["uniform_ce"])
                    <= float(forget_before["uniform_ce"]) + 1e-12,
                    "entropy_nondecreasing": float(forget_after["entropy_mean"])
                    >= float(forget_before["entropy_mean"]) - 1e-12,
                    "gradient_norm": math.sqrt(forget_gradient_squared),
                    "gradient_finite": forget_gradient_finite,
                    "gradient_nonzero": forget_gradient_nonzero,
                    "stem_parameters_changed": stem_before != stem_after,
                    "nonstem_parameters_unchanged": nonstem_before == nonstem_after,
                    "nonstem_gradient_absent": nonstem_gradient_absent,
                    "auxiliary_parameters_unchanged": (
                        auxiliary_parameter_before == auxiliary_parameter_after
                    ),
                    "auxiliary_buffers_changed": (
                        auxiliary_buffer_before != auxiliary_buffer_after
                    ),
                }

        model.train()
        main_optimizer.zero_grad(set_to_none=True)
        _set_main_forward_rng(int(args.seed), int(batch_index))
        rng_before = _rng_summary(_rng_snapshot())
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        features = _features_from_batch(model, images, metadata, device=device)
        logits = classification_logits_from_features(model, features)
        rng_after = _rng_summary(_rng_snapshot())
        main_loss = F.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(main_loss).item()):
            raise ValueError(f"Non-finite {name} main loss at batch {batch_index}.")
        main_loss.backward()
        gradient_norms, batch_gradient_seen, gradients_finite = _gradient_telemetry(
            model
        )
        for group, seen in batch_gradient_seen.items():
            gradient_seen[group] = bool(gradient_seen.get(group, False) or seen)
        all_gradients_finite = bool(all_gradients_finite and gradients_finite)
        main_optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds += float(time.perf_counter() - started)
        main_rng_before.append(rng_before)
        main_rng_after.append(rng_after)
        history.append(
            {
                "batch": int(batch_index),
                "rows": int(targets.numel()),
                "main_loss": float(main_loss.detach().item()),
                "method_step_seconds": step_seconds,
                "main_gradient_norms": gradient_norms,
                "identify": identify,
                "forget": forget,
            }
        )

    if len(history) != int(args.max_train_batches):
        raise ValueError(
            f"Expected {args.max_train_batches} train batches, observed {len(history)}."
        )
    if len(ordered_indices) != EXPECTED_TRAIN_BUDGET_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_BUDGET_ROWS} train rows, observed {len(ordered_indices)}."
        )
    adapted_predictions = _predict_fp32(
        model=model, loader=holdout_loader, device=device
    )
    final_groups = _parameter_group_hashes(model)
    measured_steps = [float(row["method_step_seconds"]) for row in history[5:]]
    result = {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(ordered_indices),
        "train_order_sha256": _ordered_index_sha256(ordered_indices),
        "main_parameter_count": sum(int(value.numel()) for value in model.parameters()),
        "auxiliary_parameter_count": (
            sum(int(value.numel()) for value in auxiliary.parameters())
            if auxiliary is not None
            else 0
        ),
        "initial_state_sha256": initial_state,
        "final_state_sha256": _state_sha256(model),
        "parameter_group_hashes_before": initial_groups,
        "parameter_group_hashes_after": final_groups,
        "parameter_group_movement": {
            key: initial_groups.get(key) != final_groups.get(key)
            for key in sorted(set(initial_groups).union(final_groups))
        },
        "gradient_seen": gradient_seen,
        "all_gradients_finite": all_gradients_finite,
        "median_step_seconds_excluding_warmup": float(np.median(measured_steps)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024.0**3)),
        "main_rng_before": main_rng_before,
        "main_rng_after": main_rng_after,
        "history": history,
        "adapted_predictions": adapted_predictions,
        "loader": {"train": train_loader_summary, "holdout": holdout_loader_summary},
        "auxiliary_initial_parameter_sha256": auxiliary_initial_parameter_sha256,
        "auxiliary_final_parameter_sha256": (
            _module_parameter_sha256(auxiliary) if auxiliary is not None else None
        ),
        "auxiliary_initial_buffer_sha256": auxiliary_initial_buffer_sha256,
        "auxiliary_final_buffer_sha256": (
            _module_buffer_sha256(auxiliary) if auxiliary is not None else None
        ),
        "auxiliary_initialization_rng_before": auxiliary_initialization_rng_before,
        "auxiliary_initialization_rng_after": auxiliary_initialization_rng_after,
        "all_main_state_finite": _module_state_finite(model),
        "all_auxiliary_state_finite": (
            _module_state_finite(auxiliary) if auxiliary is not None else True
        ),
    }
    model = model.cpu().eval()
    if auxiliary is not None:
        auxiliary = auxiliary.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, auxiliary, result


def _predict_auxiliary_pair(
    *,
    control: nn.Module,
    candidate: nn.Module,
    auxiliary: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[Dict[str, object]], float]:
    control.eval()
    candidate.eval()
    auxiliary.eval()
    rows: list[Dict[str, object]] = []
    absolute_difference = 0.0
    element_count = 0
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            control_stem = _stem_features(control, images)
            candidate_stem = _stem_features(candidate, images)
            control_logits = auxiliary(control_stem)
            candidate_logits = auxiliary(candidate_stem)
            control_probabilities = F.softmax(control_logits.float(), dim=1)
            candidate_probabilities = F.softmax(candidate_logits.float(), dim=1)
            control_entropy = _entropy(control_logits)
            candidate_entropy = _entropy(candidate_logits)
            row_difference = (candidate_stem - control_stem).float().abs().flatten(1).mean(1)
            absolute_difference += float(
                (candidate_stem - control_stem).float().abs().sum().item()
            )
            element_count += int(candidate_stem.numel())
            sample_indices = metadata.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Auxiliary prediction metadata lacks sample_index.")
            for row_index in range(int(targets.numel())):
                rows.append(
                    {
                        "sample_index": int(sample_indices[row_index].item()),
                        "target": int(targets[row_index].item()),
                        "control_aux_prediction": int(
                            control_probabilities[row_index].argmax().item()
                        ),
                        "candidate_aux_prediction": int(
                            candidate_probabilities[row_index].argmax().item()
                        ),
                        "control_aux_prob_1": float(
                            control_probabilities[row_index, 1].item()
                        ),
                        "candidate_aux_prob_1": float(
                            candidate_probabilities[row_index, 1].item()
                        ),
                        "control_aux_entropy": float(control_entropy[row_index].item()),
                        "candidate_aux_entropy": float(
                            candidate_entropy[row_index].item()
                        ),
                        "stem_mean_absolute_difference": float(
                            row_difference[row_index].item()
                        ),
                    }
                )
    return rows, absolute_difference / max(1, element_count)


def _write_clean_predictions(
    path: Path,
    *,
    source_rows: Sequence[CleanTrainRow],
    raw_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    auxiliary_rows: Sequence[Mapping[str, object]],
) -> None:
    source = {row.sample_index: row for row in source_rows}
    variants = {
        "raw": {int(row["sample_index"]): row for row in raw_rows},
        "control": {int(row["sample_index"]): row for row in control_rows},
        "candidate": {int(row["sample_index"]): row for row in candidate_rows},
    }
    auxiliary = {int(row["sample_index"]): row for row in auxiliary_rows}
    fields = ["sample_index", "source_stem", "image_path", "fold", "target"]
    for name in variants:
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(5))
    fields.extend(
        [
            "control_aux_prediction",
            "candidate_aux_prediction",
            "control_aux_prob_1",
            "candidate_aux_prob_1",
            "control_aux_entropy",
            "candidate_aux_entropy",
            "stem_mean_absolute_difference",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw_row in raw_rows:
            sample_index = int(raw_row["sample_index"])
            source_row = source[sample_index]
            output: Dict[str, object] = {
                "sample_index": sample_index,
                "source_stem": source_row.source_stem,
                "image_path": str(source_row.image_path),
                "fold": source_row.fold,
                "target": source_row.target,
            }
            for name, indexed in variants.items():
                value = indexed[sample_index]
                output[f"{name}_prediction"] = int(value["prediction"])
                for class_index in range(5):
                    output[f"{name}_prob_{class_index}"] = float(
                        value[f"prob_{class_index}"]
                    )
            for field in fields[-7:]:
                output[field] = auxiliary[sample_index][field]
            writer.writerow(output)


def _mean_probability_difference(
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> float:
    control = {
        int(row["sample_index"]): np.asarray(
            [float(row[f"prob_{index}"]) for index in range(5)], dtype=np.float64
        )
        for row in control_rows
    }
    candidate = {
        int(row["sample_index"]): np.asarray(
            [float(row[f"prob_{index}"]) for index in range(5)], dtype=np.float64
        )
        for row in candidate_rows
    }
    if set(control) != set(candidate):
        raise ValueError("Control/candidate probability sample sets differ.")
    return float(
        np.mean(
            [
                np.abs(candidate[index] - control[index]).mean()
                for index in sorted(control)
            ]
        )
    )


def _visual_category(target: int, control: int, candidate: int) -> str:
    if target in {0, 2, 4} and control == 1 and candidate != 1:
        return "restricted_fp_removed"
    if target == 1 and control == 1 and candidate != 1:
        return "class1_tp_broken"
    if target == 1 and control != 1 and candidate == 1:
        return "class1_fn_rescued"
    if target in {0, 2, 4} and control != 1 and candidate == 1:
        return "restricted_fp_created"
    if control != candidate:
        return "other_changed"
    return "largest_p1_shift"


def _select_visual_rows(
    *,
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    limit: int,
) -> list[Dict[str, object]]:
    candidate = {int(row["sample_index"]): row for row in candidate_rows}
    priority = {
        "restricted_fp_removed": 0,
        "class1_tp_broken": 1,
        "class1_fn_rescued": 2,
        "restricted_fp_created": 3,
        "other_changed": 4,
        "largest_p1_shift": 5,
    }
    rows = []
    for control_row in control_rows:
        sample_index = int(control_row["sample_index"])
        candidate_row = candidate[sample_index]
        target = int(control_row["target"])
        control_prediction = int(control_row["prediction"])
        candidate_prediction = int(candidate_row["prediction"])
        category = _visual_category(target, control_prediction, candidate_prediction)
        rows.append(
            {
                "sample_index": sample_index,
                "target": target,
                "control_prediction": control_prediction,
                "candidate_prediction": candidate_prediction,
                "control_prob_1": float(control_row["prob_1"]),
                "candidate_prob_1": float(candidate_row["prob_1"]),
                "category": category,
                "p1_absolute_delta": abs(
                    float(candidate_row["prob_1"]) - float(control_row["prob_1"])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            priority[str(row["category"])],
            -float(row["p1_absolute_delta"]),
            int(row["sample_index"]),
        )
    )
    selected = []
    selected_indices = set()
    safety_categories = (
        "restricted_fp_removed",
        "class1_tp_broken",
        "class1_fn_rescued",
        "restricted_fp_created",
    )
    for category in safety_categories:
        match = next((row for row in rows if row["category"] == category), None)
        if match is not None and len(selected) < int(limit):
            selected.append(match)
            selected_indices.add(int(match["sample_index"]))
    for row in rows:
        if len(selected) >= int(limit):
            break
        if int(row["sample_index"]) not in selected_indices:
            selected.append(row)
            selected_indices.add(int(row["sample_index"]))
    return selected[: int(limit)]


def _auxiliary_cam(
    *,
    model: nn.Module,
    auxiliary: nn.Module,
    images: Tensor,
    focus_class: int,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    auxiliary.eval()
    with torch.enable_grad():
        features = _stem_features(model, images)
        logits = auxiliary(features)
        gradient = torch.autograd.grad(
            logits[:, int(focus_class)].sum(),
            features,
            retain_graph=False,
            create_graph=False,
        )[0]
        weights = gradient.float().mean(dim=(2, 3), keepdim=True)
        cam = F.relu((features.float() * weights).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=tuple(int(value) for value in images.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        minimum = cam.flatten(1).amin(dim=1).view(-1, 1, 1, 1)
        maximum = cam.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
        probabilities = F.softmax(logits.float(), dim=1)
        entropy = _entropy(logits)
    return cam.detach(), probabilities.detach(), entropy.detach()


def _image_from_tensor(
    tensor: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> Image.Image:
    value = tensor.detach().float().cpu().clone()
    mean_tensor = torch.tensor(mean, dtype=value.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=value.dtype).view(3, 1, 1)
    value = (value * std_tensor + mean_tensor).clamp(0.0, 1.0)
    array = (value.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _overlay_cam(image: Image.Image, cam: Tensor) -> Image.Image:
    heat = cam.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    red = (255.0 * heat).astype(np.uint8)
    green = (220.0 * np.sqrt(heat)).astype(np.uint8)
    blue = (32.0 * (1.0 - heat)).astype(np.uint8)
    heat_image = Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")
    heat_image = heat_image.resize(image.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(image, heat_image, alpha=0.45)


def _render_contact_sheets(
    *,
    output_dir: Path,
    base_dataset: MangoYOLOCropDataset,
    transform,
    source_rows: Sequence[CleanTrainRow],
    condition_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    control: nn.Module,
    candidate: nn.Module,
    auxiliary: nn.Module,
    semantics: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    condition_spec = {
        "clean": (1.0, 1.0),
        **{
            str(name): (float(brightness), float(contrast))
            for name, brightness, contrast in LIGHTING_CONDITIONS
        },
    }
    source = {row.sample_index: row for row in source_rows}
    sheets = []
    for condition_index, (condition, (brightness, contrast)) in enumerate(
        condition_spec.items()
    ):
        predictions = condition_rows[condition]
        selected = _select_visual_rows(
            control_rows=predictions["control"],
            candidate_rows=predictions["candidate"],
            limit=int(args.contact_rows),
        )
        panel_width = 256
        header_height = 58
        row_height = 302
        canvas = Image.new(
            "RGB",
            (panel_width * 3, header_height + row_height * len(selected)),
            color=(248, 248, 248),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7),
            f"SIFER A0 {condition}: image | control aux-CAM | candidate aux-CAM",
            fill=(0, 0, 0),
        )
        draw.text(
            (8, 29),
            "CAM target is auxiliary class 1; rows are prospectively prioritized transitions.",
            fill=(45, 45, 45),
        )
        manifest_rows = []
        for position, selected_row in enumerate(selected):
            sample_index = int(selected_row["sample_index"])
            loader, _ = _make_lighting_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=[sample_index],
                brightness=brightness,
                contrast=contrast,
                batch_size=1,
                num_workers=0,
                context=f"sifer_xai_{condition}_{sample_index}",
                seed=int(args.seed) + 900 + condition_index,
            )
            images, targets, _ = next(iter(loader))
            images = images.to(device=device, dtype=torch.float32)
            control_cam, control_aux_probabilities, control_aux_entropy = (
                _auxiliary_cam(
                    model=control,
                    auxiliary=auxiliary,
                    images=images,
                    focus_class=int(args.focus_class),
                )
            )
            candidate_cam, candidate_aux_probabilities, candidate_aux_entropy = (
                _auxiliary_cam(
                    model=candidate,
                    auxiliary=auxiliary,
                    images=images,
                    focus_class=int(args.focus_class),
                )
            )
            image = _image_from_tensor(
                images[0],
                mean=semantics["input_mean"],
                std=semantics["input_std"],
            )
            control_overlay = _overlay_cam(image, control_cam[0])
            candidate_overlay = _overlay_cam(image, candidate_cam[0])
            top = header_height + position * row_height
            canvas.paste(image, (0, top))
            canvas.paste(control_overlay, (panel_width, top))
            canvas.paste(candidate_overlay, (panel_width * 2, top))
            label_y = top + panel_width + 4
            source_row = source[sample_index]
            line1 = (
                f"idx={sample_index} target={int(targets[0].item())} "
                f"{selected_row['category']}"
            )
            line2 = (
                f"pred C/S={selected_row['control_prediction']}/"
                f"{selected_row['candidate_prediction']} p1="
                f"{selected_row['control_prob_1']:.4f}/"
                f"{selected_row['candidate_prob_1']:.4f}"
            )
            line3 = (
                f"aux p1={float(control_aux_probabilities[0,1]):.4f}/"
                f"{float(candidate_aux_probabilities[0,1]):.4f} H="
                f"{float(control_aux_entropy[0]):.3f}/"
                f"{float(candidate_aux_entropy[0]):.3f}"
            )
            draw.text((6, label_y), line1, fill=(0, 0, 0))
            draw.text((6, label_y + 14), line2, fill=(0, 0, 0))
            draw.text((6, label_y + 28), line3, fill=(0, 0, 0))
            manifest_rows.append(
                {
                    **selected_row,
                    "source_stem": source_row.source_stem,
                    "image_path": str(source_row.image_path),
                    "control_aux_prob_1": float(control_aux_probabilities[0, 1]),
                    "candidate_aux_prob_1": float(candidate_aux_probabilities[0, 1]),
                    "control_aux_entropy": float(control_aux_entropy[0]),
                    "candidate_aux_entropy": float(candidate_aux_entropy[0]),
                    "control_cam_nonzero": bool(torch.count_nonzero(control_cam).item()),
                    "candidate_cam_nonzero": bool(
                        torch.count_nonzero(candidate_cam).item()
                    ),
                }
            )
        path = output_dir / f"contact_sheet_{condition}.png"
        canvas.save(path, format="PNG", optimize=True)
        sheets.append(
            {
                "condition": condition,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": manifest_rows,
                "row_count": len(manifest_rows),
            }
        )
    manifest = {
        "method": "sifer_feature_sieve_a0",
        "required_sheet_count": 4,
        "sheet_count": len(sheets),
        "manual_review_status": "pending",
        "sheets": sheets,
    }
    path = output_dir / "contact_sheet_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "path": str(path.resolve()), "sha256": _sha256(path)}


def assess_a0(
    *,
    structural_checks: Mapping[str, bool],
    mechanism: Mapping[str, object],
    clean: Mapping[str, object],
    illumination: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    clean_delta = clean["delta"]
    clean_transitions = clean["transitions"]
    control_support = int(clean["control"]["predicted_support"][1])
    candidate_support = int(clean["candidate"]["predicted_support"][1])
    mechanism_checks = {
        "identify_decreases_at_least_8_of_12": int(
            mechanism["identify_decrease_count"]
        )
        >= 8,
        "uniform_ce_nonincrease_at_least_8_of_12": int(
            mechanism["uniform_ce_nonincrease_count"]
        )
        >= 8,
        "entropy_nondecrease_at_least_8_of_12": int(
            mechanism["entropy_nondecrease_count"]
        )
        >= 8,
        "stem_feature_difference_gte_1e4": float(
            mechanism["stem_mean_absolute_difference"]
        )
        >= 1e-4,
        "probability_difference_gte_1e4": float(
            mechanism["probability_mean_absolute_difference"]
        )
        >= 1e-4,
    }
    clean_checks = {
        "macro_f1_delta_nonnegative": float(clean_delta["macro_f1"]) >= 0.0,
        "class1_f1_delta_gte_0p005": float(clean_delta["class1_f1"]) >= 0.005,
        "class1_precision_delta_gte_0p005": float(
            clean_delta["class1_precision"]
        )
        >= 0.005,
        "class1_recall_delta_gte_minus_0p005": float(
            clean_delta["class1_recall"]
        )
        >= -0.005,
        "restricted_focus_fp_reduction_gte_2": int(
            clean_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "focus_rescues_gte_tp_breaks": int(clean_transitions["focus_fn_rescue"])
        >= int(clean_transitions["focus_tp_break"]),
        "corrections_gt_harms": int(clean_transitions["candidate_correction"])
        > int(clean_transitions["candidate_harm"]),
        "maximum_nonfocus_f1_drop_lte_0p010": float(
            clean["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "class1_support_at_least_95pct_control": candidate_support
        >= int(math.ceil(0.95 * control_support)),
    }
    illumination_checks = {
        "all_class1_f1_deltas_gte_minus_0p010": all(
            float(row["delta"]["class1_f1"]) >= -0.010 for row in illumination
        ),
        "all_class1_recall_deltas_gte_minus_0p015": all(
            float(row["delta"]["class1_recall"]) >= -0.015
            for row in illumination
        ),
        "precision_nonnegative_in_at_least_two_conditions": sum(
            float(row["delta"]["class1_precision"]) >= 0.0
            for row in illumination
        )
        >= 2,
        "aggregate_rescues_gte_tp_breaks": sum(
            int(row["transitions"]["focus_fn_rescue"]) for row in illumination
        )
        >= sum(int(row["transitions"]["focus_tp_break"]) for row in illumination),
        "no_condition_increases_restricted_focus_fp": all(
            int(row["transitions"]["restricted_focus_fp_reduction"]) >= 0
            for row in illumination
        ),
    }
    groups = {
        "structural": dict(structural_checks),
        "mechanism": mechanism_checks,
        "clean": clean_checks,
        "illumination": illumination_checks,
    }
    automatic_failed = [
        f"{group}.{name}"
        for group, checks in groups.items()
        for name, passed in checks.items()
        if not bool(passed)
    ]
    return {
        "groups": groups,
        "automatic_failed_checks": automatic_failed,
        "automatic_gates_passed": not automatic_failed,
        "manual_visual_review_passed": None,
        "all_gates_passed": False,
        "stage_b_smoke_authorized": False,
        "validation_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "current_best_update_authorized": False,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    clean = summary.get("clean", {})
    delta = clean.get("delta", {}) if isinstance(clean, Mapping) else {}
    transitions = clean.get("transitions", {}) if isinstance(clean, Mapping) else {}
    mechanism = summary.get("mechanism", {})
    gate = summary.get("gate", {})
    visual = summary.get("visual_review")
    lines = [
        "# SIFER Feature-Sieve A0 Audit",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "## Scope",
        "",
        "- Split: `yolo_f/train` only.",
        "- Validation predictions used: `false`.",
        "- Test data used: `false`.",
        "- Current-best commands updated: `false`.",
        "",
        "## Clean Holdout",
        "",
        f"- Macro F1 delta: `{float(delta.get('macro_f1', 0.0)):.6f}`.",
        f"- Class-1 F1 delta: `{float(delta.get('class1_f1', 0.0)):.6f}`.",
        f"- Class-1 precision delta: `{float(delta.get('class1_precision', 0.0)):.6f}`.",
        f"- Class-1 recall delta: `{float(delta.get('class1_recall', 0.0)):.6f}`.",
        f"- Restricted FP reduction: `{int(transitions.get('restricted_focus_fp_reduction', 0))}`.",
        f"- FN rescues / TP breaks: `{int(transitions.get('focus_fn_rescue', 0))}` / `{int(transitions.get('focus_tp_break', 0))}`.",
        "",
        "## Mechanism",
        "",
        f"- Identify decreases: `{mechanism.get('identify_decrease_count')}/12`.",
        f"- Uniform CE non-increases: `{mechanism.get('uniform_ce_nonincrease_count')}/12`.",
        f"- Entropy non-decreases: `{mechanism.get('entropy_nondecrease_count')}/12`.",
        f"- Stem mean absolute difference: `{float(mechanism.get('stem_mean_absolute_difference', 0.0)):.8f}`.",
        f"- Probability mean absolute difference: `{float(mechanism.get('probability_mean_absolute_difference', 0.0)):.8f}`.",
        "",
        "## Gate",
        "",
        f"- Automatic gates passed: `{bool(gate.get('automatic_gates_passed', False))}`.",
        f"- Failed checks: `{gate.get('automatic_failed_checks', [])}`.",
    ]
    if isinstance(visual, Mapping):
        lines.extend(
            [
                f"- Manual visual review: `{visual.get('result')}`.",
                f"- Review note: {visual.get('note')}",
            ]
        )
    lines.extend(
        [
            "",
            "A pass authorizes only one separately locked two-epoch validation smoke. ",
            "It does not authorize test, full training, or a current-best command update.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked SIFER A0 requires CUDA.")
    provenance = _verify_sources(args)
    equation = _equation_diagnostics(args)
    output_dir = _prepare_output_dir(args.output_dir)
    paths = _source_paths(args)
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)):
        raise ValueError("CIDT provenance indicates test data use.")
    if bool(cidt_summary.get("validation_predictions_used", True)):
        raise ValueError("CIDT provenance indicates validation prediction use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])

    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    source_config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if (
        len(class_names) != 5
        or not isinstance(source_config, Mapping)
        or not isinstance(state, Mapping)
    ):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    set_seed(int(args.seed), deterministic=True)
    prototype = create_model(num_classes=len(class_names), model_config=source_config)
    load_model_state(prototype, dict(state), strict=True)
    prototype = prototype.eval()
    if not isinstance(prototype.stem, HybridConvStem):
        raise ValueError(
            f"Locked SIFER requires HybridConvStem, found {type(prototype.stem).__name__}."
        )
    if (
        len(prototype.stem.blocks) != 3
        or int(prototype.stem.out_channels) != EXPECTED_STEM_CHANNELS
        or int(prototype.stem.downsample_factor) != 8
    ):
        raise ValueError("Keeper stem contract differs from locked SIFER protocol.")
    parameter_count = sum(int(value.numel()) for value in prototype.parameters())
    state_keys = tuple(sorted(prototype.state_dict()))

    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        paths["data"], class_name_mode="raw", expected_num_classes=5
    )
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    sample_paths = [Path(value).resolve() for value in base_dataset.sample_paths()]
    if len(sample_paths) != len(rows):
        raise ValueError("Dataset/CIDT row count mismatch.")
    for expected_index, (row, path) in enumerate(zip(rows, sample_paths)):
        if int(row.sample_index) < 0 or int(row.sample_index) >= len(rows):
            raise ValueError("CIDT sample index is outside the train dataset.")
        if int(row.sample_index) != expected_index:
            raise ValueError("CIDT sample indices are not in exact dataset order.")
        if row.image_path != path:
            raise ValueError(f"Dataset/CIDT path mismatch at {row.sample_index}.")
    transform = _build_eval_transform(semantics)
    fold = int(args.fold)
    fit_indices = [row.sample_index for row in rows if row.fold != fold]
    holdout_indices = [row.sample_index for row in rows if row.fold == fold]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    source_overlap = fit_sources.intersection(holdout_sources)
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    np.random.default_rng(int(args.seed)).shuffle(shuffled_fit)
    train_indices = shuffled_fit[:EXPECTED_TRAIN_BUDGET_ROWS].tolist()
    holdout_counts = tuple(
        sum(int(rows[index].target) == class_index for index in holdout_indices)
        for class_index in range(5)
    )
    if (
        len(rows) != EXPECTED_TRAIN_ROWS
        or len(fit_indices) != EXPECTED_FIT_ROWS
        or len(holdout_indices) != EXPECTED_FOLD_COUNTS[0]
        or len(train_indices) != EXPECTED_TRAIN_BUDGET_ROWS
        or holdout_counts != EXPECTED_HOLDOUT_CLASS_COUNTS
        or source_overlap
    ):
        raise ValueError("Locked train/fold/source-disjoint contract differs.")

    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="sifer_raw_holdout",
        seed=int(args.seed) + 100,
    )
    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="sifer_resource",
        seed=int(args.seed) + 300,
    )
    resource_images, _, resource_metadata = next(iter(resource_loader))
    raw_model = copy.deepcopy(prototype).to(device).eval()
    raw_predictions = _predict_fp32(
        model=raw_model, loader=holdout_loader, device=device
    )
    raw_cidt_mismatches = sum(
        int(row["prediction"])
        != rows[int(row["sample_index"])].keeper_prediction
        for row in raw_predictions
    )
    del raw_model
    gc.collect()
    torch.cuda.empty_cache()

    control_model, control_auxiliary, control_result = _train_variant(
        name="control",
        candidate=False,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
    )
    candidate_model, candidate_auxiliary, candidate_result = _train_variant(
        name="candidate",
        candidate=True,
        prototype=prototype,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
    )
    if control_auxiliary is not None or candidate_auxiliary is None:
        raise RuntimeError("SIFER auxiliary ownership contract failed.")

    clean = _comparison(
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=5,
        focus_class=int(args.focus_class),
    )
    control_gpu = control_model.to(device).eval()
    candidate_gpu = candidate_model.to(device).eval()
    auxiliary_gpu = candidate_auxiliary.to(device).eval()
    clean_auxiliary_rows, stem_mean_absolute_difference = _predict_auxiliary_pair(
        control=control_gpu,
        candidate=candidate_gpu,
        auxiliary=auxiliary_gpu,
        loader=holdout_loader,
        device=device,
    )

    condition_rows: Dict[str, Dict[str, Sequence[Mapping[str, object]]]] = {
        "clean": {
            "control": control_result["adapted_predictions"],
            "candidate": candidate_result["adapted_predictions"],
        }
    }
    illumination = []
    illumination_loader_summaries = {}
    for condition_index, (condition, brightness, contrast) in enumerate(
        LIGHTING_CONDITIONS
    ):
        loader, loader_summary = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"sifer_{condition}",
            seed=int(args.seed) + 500 + condition_index,
        )
        control_rows = _predict_fp32(
            model=control_gpu, loader=loader, device=device
        )
        candidate_rows = _predict_fp32(
            model=candidate_gpu, loader=loader, device=device
        )
        comparison = _comparison(
            control_rows=control_rows,
            candidate_rows=candidate_rows,
            num_classes=5,
            focus_class=int(args.focus_class),
        )
        comparison["condition"] = condition
        comparison["brightness"] = float(brightness)
        comparison["contrast"] = float(contrast)
        illumination.append(comparison)
        condition_rows[condition] = {
            "control": control_rows,
            "candidate": candidate_rows,
        }
        illumination_loader_summaries[condition] = loader_summary

    clean_path = output_dir / "clean_predictions.csv"
    illumination_path = output_dir / "illumination_predictions.csv"
    history_path = output_dir / "training_history.json"
    _write_clean_predictions(
        clean_path,
        source_rows=rows,
        raw_rows=raw_predictions,
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        auxiliary_rows=clean_auxiliary_rows,
    )
    _write_illumination_predictions(
        illumination_path,
        rows=rows,
        condition_rows={
            key: value for key, value in condition_rows.items() if key != "clean"
        },
    )
    history_path.write_text(
        json.dumps(
            {
                "control": control_result["history"],
                "candidate": candidate_result["history"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contact_sheets = _render_contact_sheets(
        output_dir=output_dir,
        base_dataset=base_dataset,
        transform=transform,
        source_rows=rows,
        condition_rows=condition_rows,
        control=control_gpu,
        candidate=candidate_gpu,
        auxiliary=auxiliary_gpu,
        semantics=semantics,
        args=args,
        device=device,
    )
    control_model = control_gpu.cpu().eval()
    candidate_model = candidate_gpu.cpu().eval()
    candidate_auxiliary = auxiliary_gpu.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()

    export = _export_candidate(
        candidate=candidate_model,
        images=resource_images,
        metadata=resource_metadata,
        output_dir=output_dir,
    )
    forget_rows = [
        row["forget"]
        for row in candidate_result["history"]
        if row.get("forget") is not None
    ]
    identify_rows = [
        row["identify"]
        for row in candidate_result["history"]
        if row.get("identify") is not None
    ]
    mechanism = {
        "identify_steps": len(identify_rows),
        "forget_steps": [int(row["step"]) for row in forget_rows],
        "identify_decrease_count": sum(
            bool(candidate_result["history"][step]["identify"]["loss_decreased"])
            for step in EXPECTED_FORGET_STEPS
        ),
        "uniform_ce_nonincrease_count": sum(
            bool(row["uniform_ce_nonincreasing"]) for row in forget_rows
        ),
        "entropy_nondecrease_count": sum(
            bool(row["entropy_nondecreasing"]) for row in forget_rows
        ),
        "stem_mean_absolute_difference": stem_mean_absolute_difference,
        "probability_mean_absolute_difference": _mean_probability_difference(
            control_result["adapted_predictions"],
            candidate_result["adapted_predictions"],
        ),
        "auxiliary_parameter_state_changed": candidate_result[
            "auxiliary_initial_parameter_sha256"
        ]
        != candidate_result["auxiliary_final_parameter_sha256"],
        "auxiliary_buffer_state_changed": candidate_result[
            "auxiliary_initial_buffer_sha256"
        ]
        != candidate_result["auxiliary_final_buffer_sha256"],
        "forget_uniform_ce_deltas": [
            float(row["after"]["uniform_ce"])
            - float(row["before"]["uniform_ce"])
            for row in forget_rows
        ],
        "forget_entropy_deltas": [
            float(row["after"]["entropy_mean"])
            - float(row["before"]["entropy_mean"])
            for row in forget_rows
        ],
    }
    runtime_ratio = float(
        candidate_result["median_step_seconds_excluding_warmup"]
    ) / max(1e-12, float(control_result["median_step_seconds_excluding_warmup"]))
    required_groups = {
        "stem",
        "input_tokens",
        "transformer",
        "normalization",
        "fine_grained_pool",
        "classifier",
        "cnn_fusion",
    }
    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "locked_sources_verified": True,
        "train_only_provenance": not provenance["validation_predictions_used"]
        and not provenance["test_data_used"],
        "source_groups_disjoint": not source_overlap,
        "row_and_class_contract_exact": len(rows) == EXPECTED_TRAIN_ROWS
        and holdout_counts == EXPECTED_HOLDOUT_CLASS_COUNTS,
        "raw_predictions_match_cidt": raw_cidt_mismatches == 0,
        "stem_contract_exact": isinstance(prototype.stem, HybridConvStem)
        and len(prototype.stem.blocks) == 3
        and int(prototype.stem.out_channels) == EXPECTED_STEM_CHANNELS,
        "initial_main_states_identical": control_result["initial_state_sha256"]
        == candidate_result["initial_state_sha256"]
        == _state_sha256(prototype),
        "inference_parameter_count_unchanged": int(
            control_result["main_parameter_count"]
        )
        == int(candidate_result["main_parameter_count"])
        == parameter_count,
        "inference_state_schema_unchanged": tuple(sorted(control_model.state_dict()))
        == tuple(sorted(candidate_model.state_dict()))
        == state_keys,
        "auxiliary_external_to_inference": not any(
            "aux" in key.casefold() for key in candidate_model.state_dict()
        ),
        "auxiliary_parameter_count_exact": int(
            candidate_result["auxiliary_parameter_count"]
        )
        == EXPECTED_AUXILIARY_PARAMETERS,
        "equation_uniform_ce_exact": float(equation["uniform_ce_max_abs_error"])
        <= 1e-7,
        "equation_hard_ce_exact": float(equation["hard_ce_max_abs_error"])
        <= 1e-7,
        "equation_gradient_finite": bool(equation["gradient_finite"]),
        "equation_auxiliary_shape_exact": equation["auxiliary_output_shape"] == [2, 5],
        "equation_auxiliary_parameter_count_exact": int(
            equation["auxiliary_parameters"]
        )
        == EXPECTED_AUXILIARY_PARAMETERS,
        "train_orders_identical": control_result["train_order_sha256"]
        == candidate_result["train_order_sha256"],
        "main_rng_before_identical": control_result["main_rng_before"]
        == candidate_result["main_rng_before"],
        "main_rng_after_identical": control_result["main_rng_after"]
        == candidate_result["main_rng_after"],
        "auxiliary_initialization_rng_isolated": candidate_result[
            "auxiliary_initialization_rng_before"
        ]
        == candidate_result["auxiliary_initialization_rng_after"],
        "identify_step_count_exact": len(identify_rows) == EXPECTED_TRAIN_BATCHES,
        "forget_step_sequence_exact": tuple(int(row["step"]) for row in forget_rows)
        == EXPECTED_FORGET_STEPS,
        "identify_updates_auxiliary_only": all(
            bool(row["stem_parameters_unchanged"])
            and bool(row["auxiliary_parameters_changed"])
            and bool(row["gradient_finite"])
            and bool(row["gradient_nonzero"])
            for row in identify_rows
        ),
        "forget_updates_stem_only": all(
            bool(row["stem_parameters_changed"])
            and bool(row["nonstem_parameters_unchanged"])
            and bool(row["nonstem_gradient_absent"])
            and bool(row["auxiliary_parameters_unchanged"])
            and bool(row["gradient_finite"])
            and bool(row["gradient_nonzero"])
            for row in forget_rows
        ),
        "all_losses_and_auxiliary_values_finite": all(
            math.isfinite(float(row["main_loss"]))
            and (
                row["identify"] is None
                or (
                    bool(row["identify"]["before"]["finite"])
                    and bool(row["identify"]["after"]["finite"])
                )
            )
            and (
                row["forget"] is None
                or (
                    bool(row["forget"]["before"]["finite"])
                    and bool(row["forget"]["after"]["finite"])
                )
            )
            for row in candidate_result["history"]
        ),
        "all_main_gradients_finite": bool(control_result["all_gradients_finite"])
        and bool(candidate_result["all_gradients_finite"]),
        "all_final_model_states_finite": bool(control_result["all_main_state_finite"])
        and bool(candidate_result["all_main_state_finite"])
        and bool(candidate_result["all_auxiliary_state_finite"]),
        "all_required_main_gradient_groups_seen": all(
            bool(candidate_result["gradient_seen"].get(group, False))
            for group in required_groups
        ),
        "all_required_main_parameter_groups_moved": all(
            bool(candidate_result["parameter_group_movement"].get(group, False))
            for group in required_groups
        ),
        "runtime_ratio_lte_1p60": runtime_ratio <= float(args.max_runtime_ratio),
        "peak_vram_lte_6p50_gib": float(candidate_result["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
        "static_onnx_succeeded": bool(export.get("succeeded", False)),
        "static_onnx_finite": bool(export.get("finite", False)),
        "static_onnx_error_lte_1e5": float(
            export.get("maximum_absolute_error", 1e9)
        )
        <= MAX_ONNX_ERROR,
        "static_onnx_argmax_match": bool(export.get("argmax_match", False)),
        "contact_sheet_count_exact": int(contact_sheets["sheet_count"]) == 4,
        "contact_sheet_files_verified": all(
            Path(str(sheet["path"])).is_file()
            and _sha256(Path(str(sheet["path"]))) == str(sheet["sha256"])
            for sheet in contact_sheets["sheets"]
        ),
    }
    gate = assess_a0(
        structural_checks=structural_checks,
        mechanism=mechanism,
        clean=clean,
        illumination=illumination,
    )

    def compact_result(result: Mapping[str, object]) -> Dict[str, object]:
        return {
            key: value
            for key, value in result.items()
            if key
            not in {
                "adapted_predictions",
                "history",
                "main_rng_before",
                "main_rng_after",
            }
        }

    summary: Dict[str, object] = {
        "method": "sifer_feature_sieve_a0",
        "status": "awaiting_visual_review",
        "auto_gate_eligible": bool(gate["automatic_gates_passed"]),
        "provenance": provenance,
        "dataset": {
            "split": "train",
            "rows": len(rows),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fit_rows": len(fit_indices),
            "train_budget_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "holdout_class_counts": list(holdout_counts),
            "source_overlap": len(source_overlap),
            "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
            "validation_predictions_used": False,
            "test_data_used": False,
            "holdout_loader": holdout_loader_summary,
            "resource_loader": resource_loader_summary,
            "illumination_loaders": illumination_loader_summaries,
        },
        "model": {
            "main_parameter_count": parameter_count,
            "auxiliary_parameter_count": EXPECTED_AUXILIARY_PARAMETERS,
            "initial_state_sha256": _state_sha256(prototype),
            "stem_type": type(prototype.stem).__name__,
            "stem_channels": EXPECTED_STEM_CHANNELS,
            "stem_grid": [EXPECTED_STEM_GRID, EXPECTED_STEM_GRID],
            "inference_graph_changed": False,
        },
        "config": to_serializable(vars(args)),
        "equation": equation,
        "raw_cidt_prediction_mismatches": raw_cidt_mismatches,
        "control": compact_result(control_result),
        "candidate": compact_result(candidate_result),
        "mechanism": mechanism,
        "runtime": {
            "control_median_step_seconds": control_result[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_median_step_seconds": candidate_result[
                "median_step_seconds_excluding_warmup"
            ],
            "candidate_control_ratio": runtime_ratio,
            "control_peak_vram_gib": control_result["peak_vram_gib"],
            "candidate_peak_vram_gib": candidate_result["peak_vram_gib"],
        },
        "clean": clean,
        "illumination": illumination,
        "export": export,
        "contact_sheet_audit": contact_sheets,
        "visual_review": None,
        "gate": gate,
        "current_best_commands_updated": False,
        "artifacts": {
            "clean_predictions": {
                "path": str(clean_path.resolve()),
                "sha256": _sha256(clean_path),
            },
            "illumination_predictions": {
                "path": str(illumination_path.resolve()),
                "sha256": _sha256(illumination_path),
            },
            "training_history": {
                "path": str(history_path.resolve()),
                "sha256": _sha256(history_path),
            },
            "contact_sheet_manifest": {
                "path": str(Path(str(contact_sheets["path"])).resolve()),
                "sha256": str(contact_sheets["sha256"]),
            },
        },
    }
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    _write_report(report_path, summary)
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_artifact_count"]) != 0:
        raise RuntimeError("SIFER A0 retained a forbidden model/export artifact.")
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    provenance = _verify_sources(args)
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Formal SIFER summary is missing: {summary_path}")
    observed_sha = _sha256(summary_path)
    expected_sha = str(args.expected_summary_sha256).strip().casefold()
    if not expected_sha or observed_sha != expected_sha:
        raise ValueError(
            f"Pre-review summary SHA differs: {observed_sha} != {expected_sha}"
        )
    note = str(args.visual_review_note).strip()
    if not note:
        raise ValueError("Visual finalization requires a non-empty review note.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("SIFER summary is not awaiting visual review.")
    if summary["provenance"]["git_objects"] != provenance["git_objects"]:
        raise ValueError("Official SIFER provenance changed before finalization.")
    sheets = summary["contact_sheet_audit"]["sheets"]
    if len(sheets) != 4:
        raise ValueError("Exactly four SIFER contact sheets must be reviewed.")
    for sheet in sheets:
        path = Path(str(sheet["path"]))
        if not path.is_file() or _sha256(path) != str(sheet["sha256"]):
            raise ValueError(f"Contact sheet changed before review: {path}")

    visual_passed = str(args.visual_review_result) == "pass"
    review_manifest = {
        "result": str(args.visual_review_result),
        "passed": visual_passed,
        "note": note,
        "pre_review_summary_sha256": observed_sha,
        "reviewed_sheet_count": len(sheets),
        "sheets": sheets,
    }
    review_path = output_dir / "contact_sheet_review_manifest.json"
    review_path.write_text(
        json.dumps(review_manifest, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    summary["contact_sheet_audit"]["manual_review_status"] = (
        "passed" if visual_passed else "failed"
    )
    summary["contact_sheet_audit"]["review_manifest_sha256"] = _sha256(
        review_path
    )
    summary["visual_review"] = review_manifest
    gate = dict(summary["gate"])
    failed = list(gate["automatic_failed_checks"])
    if not visual_passed:
        failed.append("manual_visual_review_passed")
    gate["manual_visual_review_passed"] = visual_passed
    gate["failed_checks"] = failed
    gate["all_gates_passed"] = not failed
    gate["stage_b_smoke_authorized"] = not failed
    gate["validation_authorized"] = False
    gate["test_authorized"] = False
    gate["full_train_authorized"] = False
    gate["current_best_update_authorized"] = False
    summary["gate"] = gate
    summary["status"] = "passed" if not failed else "rejected"
    summary["current_best_commands_updated"] = False
    _write_report(output_dir / "report.md", summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_artifact_count"]) != 0:
        raise RuntimeError("Finalized SIFER A0 contains a forbidden artifact.")
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if bool(args.finalize_visual_review):
        result = _finalize_visual_review(args)
    elif bool(args.preflight_only):
        provenance = _verify_sources(args)
        equation = _equation_diagnostics(args)
        result = {
            "method": "sifer_feature_sieve_a0",
            "status": "preflight_passed",
            "preflight_only": True,
            "provenance": provenance,
            "equation": equation,
            "output_created": False,
        }
    else:
        result = run_audit(args)
    print(
        json.dumps(
            {
                "method": result["method"],
                "status": result["status"],
                "summary_path": result.get("summary_path"),
                "summary_sha256": result.get("summary_sha256"),
                "auto_gate_eligible": result.get("auto_gate_eligible", False),
                "stage_b_smoke_authorized": result.get("gate", {}).get(
                    "stage_b_smoke_authorized", False
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
