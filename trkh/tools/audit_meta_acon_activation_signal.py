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
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.inference.inference import load_checkpoint
from trkh.models.model import HybridConvStem
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _build_dataset,
    _load_keeper_model,
    _make_condition_loader,
    _replay_keeper_declarations,
    region_masks,
)


METHOD = "meta_acon_activation_signal_a0"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 12
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
MIN_FIT_TP_RETENTION = 0.90
BENCHMARK_REPEATS = 3
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
ROLES = ("identity", "acon_static", "meta_acon")
META_OBJECT_BETA_ROLE = "meta_acon_object_beta"
WIDTH = 256
REDUCTION = 16
FEATURE_HEIGHT = 32
FEATURE_WIDTH = 32
MAX_BF16_ERROR = 0.02
MAX_EQUATION_ERROR = 1e-6
MAX_FINITE_DIFFERENCE_ERROR = 1e-4
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.25
MAX_MEMORY_RATIO = 1.25

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "c1fb84b33532285c81691952e0dff7d08dd0d8541dd3706cae02be6f8792e436"
LOCKED_PAPER_SHA256 = "acae6d8f9100250f9d21c2bbfc8a6f5ae4fbd7d886643e0cbd4a1405a02f9218"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "99fd67928a6ffb0543b54614303caada96c756f5"
LOCKED_OFFICIAL_TREE = "f269fee6201f0945fc154134f310704b59175837"
LOCKED_OFFICIAL_HASHES = {
    "acon_source": "670d4d70e44009fb857bc4e056798dd60fe66ac2ed02c1c849f04c0b44ee4087",
    "license": "e09e657f606b61bf1ab24f59db9b68d61b0532943ff22db2ff8dd0cfc84ece60",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Meta-ACON class-1 TP-versus-restricted-FP gate. "
            "Validation, test, trainer integration, and raw-data edits are forbidden."
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
            "docs/TRKH_5CLASS_META_ACON_ACTIVATION_SIGNAL_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\acon_cvpr2021.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\acon-cvpr2021"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_meta_acon_activation_signal_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--benchmark-repeats", type=int, default=BENCHMARK_REPEATS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == NUM_WORKERS
        and int(args.epochs) == EPOCHS
        and math.isclose(float(args.learning_rate), LEARNING_RATE)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY)
        and int(args.benchmark_repeats) == BENCHMARK_REPEATS
        and int(args.seed) == SEED
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
        "acon_source": official / "acon.py",
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
        raise ValueError("Arguments differ from the locked Meta-ACON A0 protocol.")
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "Meta-ACON protocol"
        ),
        "paper": _verify_sha256(
            paths["paper"], LOCKED_PAPER_SHA256, "accepted ACON paper"
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
            paths[name], expected, f"official ACON {name}"
        )

    official_commit = _git_value(paths["official_root"], "rev-parse", "HEAD")
    official_tree = _git_value(
        paths["official_root"], "rev-parse", "HEAD^{tree}"
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official ACON commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError(
            f"Official ACON tree differs: {official_tree} != {LOCKED_OFFICIAL_TREE}"
        )
    if not _full_worktree_clean(paths["official_root"]):
        raise ValueError("Official ACON worktree must be clean.")

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
            f"Locked Meta-ACON cohort differs: {len(cohort)} != {EXPECTED_COHORT_ROWS}"
        )
    positives = sum(_cohort_label(row) == "tp" for row in cohort)
    negatives = sum(_cohort_label(row) == "fp" for row in cohort)
    if (positives, negatives) != (EXPECTED_POSITIVES, EXPECTED_NEGATIVES):
        raise ValueError(
            "Locked Meta-ACON class counts differ: "
            f"{positives}/{negatives} != {EXPECTED_POSITIVES}/{EXPECTED_NEGATIVES}"
        )
    fold_counts: Dict[int, Dict[str, int]] = {
        fold: {"tp": 0, "fp": 0} for fold in FIT_FOLDS
    }
    for row in cohort:
        fold_counts[row.fold][str(_cohort_label(row))] += 1
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Locked Meta-ACON fold counts differ: {fold_counts}")
    ordered_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Locked cohort index hash differs: {ordered_hash}")

    repo_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repo_root):
        raise ValueError("Tracked TRKH worktree must be clean for Meta-ACON A0.")
    repository_commit = _git_value(repo_root, "rev-parse", "HEAD")
    upstream_commit = _git_value(
        repo_root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal Meta-ACON A0 requires the pushed repository commit.")

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
            "validation_data_used": False,
            "test_data_used": False,
            "image_model_training_used": False,
            "activation_readout_training_used": True,
        },
        rows,
        cohort,
    )


class AconC(nn.Module):
    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, int(width), 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, int(width), 1, 1))
        self.beta = nn.Parameter(torch.ones(1, int(width), 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        delta = self.p1 * x - self.p2 * x
        return delta * torch.sigmoid(self.beta * delta) + self.p2 * x


class MetaAconC(nn.Module):
    def __init__(self, width: int = WIDTH, reduction: int = REDUCTION) -> None:
        super().__init__()
        hidden = max(int(reduction), int(width) // int(reduction))
        self.fc1 = nn.Conv2d(int(width), hidden, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, int(width), kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(width))
        self.p1 = nn.Parameter(torch.ones(1, int(width), 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, int(width), 1, 1))

    @staticmethod
    def _masked_gap(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return x.mean(dim=(2, 3), keepdim=True)
        if mask.ndim != 3 or tuple(mask.shape) != (
            int(x.size(0)),
            int(x.size(2)),
            int(x.size(3)),
        ):
            raise ValueError("Meta-ACON beta mask shape differs from feature map.")
        expanded = mask[:, None].to(dtype=x.dtype)
        denominator = expanded.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (x * expanded).sum(dim=(2, 3), keepdim=True) / denominator

    def compute_beta(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pooled = self._masked_gap(x, mask)
        return torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(pooled)))))

    def forward_with_beta(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        beta = self.compute_beta(x, mask)
        delta = self.p1 * x - self.p2 * x
        output = delta * torch.sigmoid(beta * delta) + self.p2 * x
        return output, beta

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_beta(x)[0]


def object_mask_from_bbox(bboxes: Tensor, height: int, width: int) -> Tensor:
    core, boundary, _ = region_masks(bboxes, height, width)
    return core | boundary


class ObjectActivationReadout(nn.Module):
    def __init__(self, role: str, width: int = WIDTH) -> None:
        super().__init__()
        normalized = str(role).strip().casefold()
        if normalized not in ROLES:
            raise ValueError(f"Unknown activation role: {role}")
        self.role = normalized
        if normalized == "acon_static":
            self.activation: Optional[nn.Module] = AconC(width)
        elif normalized == "meta_acon":
            self.activation = MetaAconC(width)
        else:
            self.activation = None
        self.head = nn.Linear(int(width), 1)

    def forward(
        self,
        x: Tensor,
        bboxes: Tensor,
        *,
        object_only_beta: bool = False,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        return self._forward_with_mask(
            x, bboxes, object_only_beta=object_only_beta
        )

    def _forward_with_mask(
        self,
        x: Tensor,
        bboxes: Tensor,
        *,
        object_only_beta: bool = False,
        precomputed_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        mask = (
            precomputed_mask
            if precomputed_mask is not None
            else object_mask_from_bbox(bboxes, int(x.size(2)), int(x.size(3)))
        )
        beta: Optional[Tensor] = None
        if self.role == "identity":
            activated = x
        elif self.role == "acon_static":
            assert isinstance(self.activation, AconC)
            activated = self.activation(x)
            beta = self.activation.beta.expand(int(x.size(0)), -1, -1, -1)
        else:
            assert isinstance(self.activation, MetaAconC)
            beta_mask = mask if object_only_beta else None
            activated, beta = self.activation.forward_with_beta(x, beta_mask)
        expanded = mask[:, None].to(dtype=activated.dtype)
        denominator = expanded.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (activated * expanded).sum(dim=(2, 3)) / denominator
        logits = self.head(pooled).squeeze(1)
        delta_rms = (
            (activated.float() - x.float()).square().mean(dim=(1, 2, 3)).sqrt()
        )
        return logits, beta, delta_rms


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("locked_official_acon", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official ACON source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _independent_meta_acon(module: MetaAconC, x: Tensor) -> tuple[Tensor, Tensor]:
    pooled = x.mean(dim=(2, 3), keepdim=True)
    first = F.conv2d(pooled, module.fc1.weight, module.fc1.bias)
    first = F.batch_norm(
        first,
        module.bn1.running_mean,
        module.bn1.running_var,
        module.bn1.weight,
        module.bn1.bias,
        training=False,
        momentum=0.0,
        eps=module.bn1.eps,
    )
    second = F.conv2d(first, module.fc2.weight, module.fc2.bias)
    second = F.batch_norm(
        second,
        module.bn2.running_mean,
        module.bn2.running_var,
        module.bn2.weight,
        module.bn2.bias,
        training=False,
        momentum=0.0,
        eps=module.bn2.eps,
    )
    beta = torch.sigmoid(second)
    delta = module.p1 * x - module.p2 * x
    return delta * torch.sigmoid(beta * delta) + module.p2 * x, beta


def _equation_diagnostics(official_source: Path, device: torch.device) -> Dict[str, object]:
    official_module = _load_module(official_source)
    if not hasattr(official_module, "MetaAconC"):
        raise AttributeError("Official ACON source lacks MetaAconC.")

    torch.manual_seed(SEED + 101)
    official = official_module.MetaAconC(16, r=16).double().eval()
    local = MetaAconC(16, reduction=16).double().eval()
    local.load_state_dict(official.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(SEED + 102)
    x_local = torch.randn(4, 16, 7, 9, generator=generator, dtype=torch.float64)
    x_official = x_local.detach().clone()
    x_local.requires_grad_(True)
    x_official.requires_grad_(True)
    local_output, local_beta = local.forward_with_beta(x_local)
    official_output = official(x_official)
    oracle_output, oracle_beta = _independent_meta_acon(local, x_local)
    probe = torch.randn(local_output.shape, generator=generator, dtype=torch.float64)
    local_loss = (local_output * probe).sum()
    official_loss = (official_output * probe).sum()
    local_parameters = dict(local.named_parameters())
    official_parameters = dict(official.named_parameters())
    local_gradients = torch.autograd.grad(
        local_loss, [x_local, *local_parameters.values()], retain_graph=True
    )
    official_gradients = torch.autograd.grad(
        official_loss, [x_official, *official_parameters.values()]
    )
    gradient_errors = {
        "input": float((local_gradients[0] - official_gradients[0]).abs().max())
    }
    for index, name in enumerate(local_parameters, start=1):
        gradient_errors[name] = float(
            (local_gradients[index] - official_gradients[index]).abs().max()
        )

    finite_module = copy.deepcopy(local)
    finite_x = x_local.detach()
    coordinate = (0, 0, 0, 0)
    epsilon = 1e-5
    with torch.no_grad():
        original = float(finite_module.p1[coordinate])
        finite_module.p1[coordinate] = original + epsilon
        plus = float((finite_module(finite_x) * probe).sum())
        finite_module.p1[coordinate] = original - epsilon
        minus = float((finite_module(finite_x) * probe).sum())
        finite_module.p1[coordinate] = original
    numerical = (plus - minus) / (2.0 * epsilon)
    analytic = float(local_gradients[1][coordinate])

    paper_init = MetaAconC(16, reduction=16)
    source_text = official_source.read_text(encoding="utf-8")
    source_uses_random_p = bool(
        "self.p1 = nn.Parameter(torch.randn" in source_text
        and "self.p2 = nn.Parameter(torch.randn" in source_text
    )

    bf16_model = MetaAconC(16, reduction=16).to(device).eval()
    bf16_model.load_state_dict(paper_init.state_dict(), strict=True)
    bf16_x = torch.randn(4, 16, 7, 9, device=device, dtype=torch.float32)
    fp32_x = bf16_x.detach().clone().requires_grad_(True)
    fp32_output = bf16_model(fp32_x)
    fp32_loss = fp32_output.square().mean()
    fp32_gradients = torch.autograd.grad(
        fp32_loss, [fp32_x, *bf16_model.parameters()]
    )
    fp32_finite = all(
        bool(torch.isfinite(value).all()) for value in fp32_gradients
    )
    fp32_nonzero = all(
        int(torch.count_nonzero(value)) > 0 for value in fp32_gradients
    )
    bf16_copy = copy.deepcopy(bf16_model).to(dtype=torch.bfloat16)
    bf16_input = bf16_x.to(dtype=torch.bfloat16).requires_grad_(True)
    bf16_output = bf16_copy(bf16_input)
    bf16_loss = bf16_output.float().square().mean()
    bf16_gradients = torch.autograd.grad(
        bf16_loss, [bf16_input, *bf16_copy.parameters()]
    )
    bf16_error = float(
        (fp32_output.detach() - bf16_output.detach().float()).abs().max()
    )
    bf16_finite = all(bool(torch.isfinite(value).all()) for value in bf16_gradients)
    bf16_nonzero = all(int(torch.count_nonzero(value)) > 0 for value in bf16_gradients)
    bf16_model.cpu()
    bf16_copy.cpu()
    torch.cuda.empty_cache()

    return {
        "official_output_max_abs_error": float(
            (local_output.detach() - official_output.detach()).abs().max()
        ),
        "oracle_output_max_abs_error": float(
            (local_output.detach() - oracle_output.detach()).abs().max()
        ),
        "oracle_beta_max_abs_error": float(
            (local_beta.detach() - oracle_beta.detach()).abs().max()
        ),
        "gradient_max_abs_errors": gradient_errors,
        "maximum_gradient_error": max(gradient_errors.values()),
        "finite_difference_error": abs(numerical - analytic),
        "finite_difference_analytic": analytic,
        "finite_difference_numerical": numerical,
        "paper_initialization_exact": bool(
            torch.equal(paper_init.p1, torch.ones_like(paper_init.p1))
            and torch.equal(paper_init.p2, torch.zeros_like(paper_init.p2))
        ),
        "official_source_random_p_initialization_detected": source_uses_random_p,
        "initialization_discrepancy_disclosed": source_uses_random_p,
        "bf16_max_abs_error": bf16_error,
        "fp32_gradients_finite": fp32_finite,
        "fp32_gradient_families_nonzero": fp32_nonzero,
        "bf16_gradients_finite": bf16_finite,
        "bf16_gradient_families_nonzero": bf16_nonzero,
        "all_outputs_finite": bool(
            torch.isfinite(local_output).all()
            and torch.isfinite(official_output).all()
            and torch.isfinite(oracle_output).all()
            and torch.isfinite(bf16_output).all()
        ),
    }


def _extract_stem_cache(
    *,
    model: nn.Module,
    base_dataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, Tensor], Tensor, Dict[str, object]]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem):
        raise TypeError("Meta-ACON A0 requires the locked HybridConvStem.")
    stem = stem.to(device).eval()
    indices = [row.sample_index for row in cohort]
    expected_targets = [row.target for row in cohort]
    caches: Dict[str, Tensor] = {}
    locked_bboxes: Optional[Tensor] = None
    loader_summaries: Dict[str, object] = {}
    with torch.inference_mode():
        for condition, brightness, contrast in CONDITIONS:
            loader, summary = _make_condition_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=indices,
                brightness=brightness,
                contrast=contrast,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                context=f"meta_acon_a0_{condition}",
            )
            loader_summaries[condition] = summary
            feature_batches: list[Tensor] = []
            bbox_batches: list[Tensor] = []
            observed_indices: list[int] = []
            observed_targets: list[int] = []
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Meta-ACON cache loader lacks sample_index.")
                if not torch.is_tensor(crop_bboxes):
                    raise ValueError("Meta-ACON cache loader lacks crop_bbox.")
                images = images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                features = stem(images)
                if tuple(features.shape[1:]) != (
                    WIDTH,
                    FEATURE_HEIGHT,
                    FEATURE_WIDTH,
                ):
                    raise ValueError(
                        f"Locked stem shape differs: {tuple(features.shape)}"
                    )
                feature_batches.append(features.to(dtype=torch.float16).cpu())
                bbox_batches.append(crop_bboxes.to(dtype=torch.float32).cpu())
                observed_indices.extend(int(value) for value in sample_indices.tolist())
                observed_targets.extend(int(value) for value in targets.tolist())
            if observed_indices != indices or observed_targets != expected_targets:
                raise ValueError(f"{condition} Meta-ACON cache order differs.")
            cache = torch.cat(feature_batches, dim=0).contiguous()
            condition_bboxes = torch.cat(bbox_batches, dim=0).contiguous()
            if cache.shape != (
                EXPECTED_COHORT_ROWS,
                WIDTH,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            ):
                raise ValueError(f"Unexpected {condition} cache shape: {cache.shape}")
            if not bool(torch.isfinite(cache).all()):
                raise ValueError(f"Non-finite {condition} stem cache.")
            if locked_bboxes is None:
                locked_bboxes = condition_bboxes
            elif not torch.equal(locked_bboxes, condition_bboxes):
                raise ValueError("Transformed bbox geometry differs across conditions.")
            caches[condition] = cache
            del feature_batches, bbox_batches
            gc.collect()
    stem.cpu()
    torch.cuda.empty_cache()
    if locked_bboxes is None:
        raise RuntimeError("Meta-ACON bbox cache was not constructed.")
    return caches, locked_bboxes, {
        "conditions": loader_summaries,
        "cache_dtype": "float16_cpu_ephemeral",
        "cache_shapes": {
            name: list(value.shape) for name, value in caches.items()
        },
        "cache_written_to_disk": False,
    }


def _head_state(seed: int) -> Dict[str, Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = nn.Linear(WIDTH, 1)
    return {name: value.detach().clone() for name, value in head.state_dict().items()}


def _build_role(role: str, head_state: Mapping[str, Tensor], seed: int) -> ObjectActivationReadout:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ObjectActivationReadout(role, WIDTH)
    model.head.load_state_dict(head_state, strict=True)
    return model


def select_recall_constrained_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    min_tp_retention: float = MIN_FIT_TP_RETENTION,
) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size != labels.size or not scores.size:
        raise ValueError("Threshold scores/labels do not align.")
    positives = scores[labels == 1]
    if positives.size == 0:
        raise ValueError("Threshold fit contains no positives.")
    candidates = np.unique(scores)
    valid = [
        float(threshold)
        for threshold in candidates
        if float(np.mean(positives >= threshold)) >= float(min_tp_retention)
    ]
    if not valid:
        raise RuntimeError("No threshold satisfies the fit TP-retention constraint.")
    return max(valid)


def _predict_cached(
    *,
    model: ObjectActivationReadout,
    features: Tensor,
    bboxes: Tensor,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    object_only_beta: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    model.eval()
    score_batches: list[np.ndarray] = []
    beta_batches: list[np.ndarray] = []
    delta_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(indices.size), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            x = features[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            boxes = bboxes[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits, beta, delta = model(
                x, boxes, object_only_beta=object_only_beta
            )
            score_batches.append(torch.sigmoid(logits).cpu().numpy())
            delta_batches.append(delta.cpu().numpy())
            if beta is not None:
                beta_batches.append(beta.flatten(1).cpu().numpy())
    scores = np.concatenate(score_batches).astype(np.float64)
    deltas = np.concatenate(delta_batches).astype(np.float64)
    beta_values = (
        np.concatenate(beta_batches).astype(np.float64) if beta_batches else None
    )
    return scores, beta_values, deltas


def _train_role(
    *,
    role: str,
    head_state: Mapping[str, Tensor],
    clean_features: Tensor,
    bboxes: Tensor,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    permutations: Sequence[np.ndarray],
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ObjectActivationReadout, Dict[str, object]]:
    model = _build_role(
        role,
        head_state,
        seed=int(args.seed) + int(fold) * 1000 + 17,
    ).to(device)
    initial_head = {
        name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    epoch_rows: list[Dict[str, float]] = []
    for epoch, permutation in enumerate(permutations, start=1):
        model.train()
        losses: list[float] = []
        for start in range(0, int(permutation.size), int(args.batch_size)):
            local = permutation[start : start + int(args.batch_size)]
            batch_indices = fit_indices[local]
            x = clean_features[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            boxes = bboxes[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = torch.as_tensor(
                labels[batch_indices], device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(x, boxes)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {role} loss in fold {fold}.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_rows.append(
            {
                "epoch": float(epoch),
                "mean_loss": float(statistics.mean(losses)),
            }
        )
    head_start_exact = all(
        torch.equal(initial_head[name], head_state[name]) for name in head_state
    )
    return model, {
        "role": role,
        "fold": int(fold),
        "fit_rows": int(fit_indices.size),
        "epochs": int(args.epochs),
        "head_initialization_exact": head_start_exact,
        "train_curve": epoch_rows,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


def _binary_metrics(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> Dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.shape != labels.shape or scores.shape != thresholds.shape:
        raise ValueError("OOF score/label/threshold arrays do not align.")
    if not bool(np.isfinite(scores).all() and np.isfinite(thresholds).all()):
        raise ValueError("OOF scores or thresholds are non-finite.")
    predicted = scores >= thresholds
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tp_retention": float(tp / max(1, tp + fn)),
        "fp_rejection": float(tn / max(1, fp + tn)),
        "precision": float(tp / max(1, tp + fp)),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "tp_median_score": float(np.median(scores[positive])),
        "fp_median_score": float(np.median(scores[negative])),
        "score_mean": float(np.mean(scores)),
        "threshold_mean": float(np.mean(thresholds)),
    }


def _run_oof(
    *,
    caches: Mapping[str, Tensor],
    bboxes: Tensor,
    cohort: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, object],
    Dict[str, object],
]:
    labels = np.asarray(
        [1 if _cohort_label(row) == "tp" else 0 for row in cohort], dtype=np.int64
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    all_roles = (*ROLES, META_OBJECT_BETA_ROLE)
    scores = {
        role: {
            condition: np.full(labels.size, np.nan, dtype=np.float64)
            for condition, _, _ in CONDITIONS
        }
        for role in all_roles
    }
    thresholds = {
        role: np.full(labels.size, np.nan, dtype=np.float64) for role in all_roles
    }
    beta_rows = np.full((labels.size, WIDTH), np.nan, dtype=np.float64)
    delta_rows = np.full(labels.size, np.nan, dtype=np.float64)
    fold_diagnostics: list[Dict[str, object]] = []

    for fold in FIT_FOLDS:
        fit_indices = np.flatnonzero(folds != int(fold)).astype(np.int64)
        hold_indices = np.flatnonzero(folds == int(fold)).astype(np.int64)
        fit_sources = {cohort[index].source_stem for index in fit_indices}
        hold_sources = {cohort[index].source_stem for index in hold_indices}
        if fit_sources.intersection(hold_sources):
            raise ValueError(f"Source overlap in Meta-ACON fold {fold}.")
        rng = np.random.default_rng(int(args.seed) + int(fold) * 100)
        permutations = [rng.permutation(fit_indices.size) for _ in range(EPOCHS)]
        shared_head = _head_state(int(args.seed) + int(fold) * 1000 + 11)
        fold_row: Dict[str, object] = {
            "fold": int(fold),
            "fit_rows": int(fit_indices.size),
            "holdout_rows": int(hold_indices.size),
            "fit_sources": len(fit_sources),
            "holdout_sources": len(hold_sources),
            "source_overlap": 0,
            "roles": {},
        }

        for role in ROLES:
            model, train_diagnostic = _train_role(
                role=role,
                head_state=shared_head,
                clean_features=caches["clean"],
                bboxes=bboxes,
                labels=labels,
                fit_indices=fit_indices,
                permutations=permutations,
                fold=fold,
                args=args,
                device=device,
            )
            fit_scores, _, _ = _predict_cached(
                model=model,
                features=caches["clean"],
                bboxes=bboxes,
                indices=fit_indices,
                device=device,
                batch_size=int(args.batch_size),
            )
            threshold = select_recall_constrained_threshold(
                fit_scores, labels[fit_indices]
            )
            thresholds[role][hold_indices] = threshold
            if role == "meta_acon":
                thresholds[META_OBJECT_BETA_ROLE][hold_indices] = threshold

            condition_metrics: Dict[str, object] = {}
            for condition, _, _ in CONDITIONS:
                hold_scores, beta, delta = _predict_cached(
                    model=model,
                    features=caches[condition],
                    bboxes=bboxes,
                    indices=hold_indices,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                scores[role][condition][hold_indices] = hold_scores
                fold_thresholds = np.full(hold_indices.size, threshold)
                condition_metrics[condition] = _binary_metrics(
                    hold_scores, labels[hold_indices], fold_thresholds
                )
                if role == "meta_acon" and condition == "clean":
                    if beta is None:
                        raise RuntimeError("Meta-ACON did not expose beta.")
                    beta_rows[hold_indices] = beta
                    delta_rows[hold_indices] = delta

            if role == "meta_acon":
                for condition, _, _ in CONDITIONS:
                    object_scores, _, _ = _predict_cached(
                        model=model,
                        features=caches[condition],
                        bboxes=bboxes,
                        indices=hold_indices,
                        device=device,
                        batch_size=int(args.batch_size),
                        object_only_beta=True,
                    )
                    scores[META_OBJECT_BETA_ROLE][condition][hold_indices] = (
                        object_scores
                    )

            train_diagnostic["fit_threshold"] = threshold
            train_diagnostic["conditions"] = condition_metrics
            fold_row["roles"][role] = train_diagnostic
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        fold_row["clean_auroc_delta_meta_vs_static"] = float(
            fold_row["roles"]["meta_acon"]["conditions"]["clean"]["auroc"]
            - fold_row["roles"]["acon_static"]["conditions"]["clean"]["auroc"]
        )
        fold_diagnostics.append(fold_row)

    for role in all_roles:
        if not bool(np.isfinite(thresholds[role]).all()):
            raise ValueError(f"Incomplete OOF thresholds for {role}.")
        for condition, _, _ in CONDITIONS:
            if not bool(np.isfinite(scores[role][condition]).all()):
                raise ValueError(f"Incomplete OOF scores for {role}/{condition}.")
    if not bool(np.isfinite(beta_rows).all() and np.isfinite(delta_rows).all()):
        raise ValueError("Meta-ACON activity telemetry is incomplete.")

    metrics = {
        role: {
            condition: _binary_metrics(
                scores[role][condition], labels, thresholds[role]
            )
            for condition, _, _ in CONDITIONS
        }
        for role in all_roles
    }
    beta_centered = beta_rows - np.mean(beta_rows, axis=0, keepdims=True)
    activity = {
        "between_sample_beta_rms": float(np.sqrt(np.mean(beta_centered**2))),
        "activation_delta_rms_mean": float(np.mean(delta_rows)),
        "activation_delta_rms_median": float(np.median(delta_rows)),
        "beta_mean": float(np.mean(beta_rows)),
        "beta_std": float(np.std(beta_rows)),
        "tp_beta_mean": float(np.mean(beta_rows[labels == 1])),
        "fp_beta_mean": float(np.mean(beta_rows[labels == 0])),
        "all_finite": True,
    }
    diagnostics = {
        "folds": fold_diagnostics,
        "activity": activity,
        "head_initialization_shared_every_role": all(
            bool(role["head_initialization_exact"])
            for fold in fold_diagnostics
            for role in fold["roles"].values()
        ),
        "source_overlap_zero": all(
            int(fold["source_overlap"]) == 0 for fold in fold_diagnostics
        ),
    }
    return scores, thresholds, metrics, diagnostics


def assess_information_gate(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    meta = metrics["meta_acon"]
    static = metrics["acon_static"]
    identity = metrics["identity"]
    object_beta = metrics[META_OBJECT_BETA_ROLE]
    fold_deltas = [
        float(row["clean_auroc_delta_meta_vs_static"])
        for row in diagnostics["folds"]
    ]
    shifts = [name for name, _, _ in CONDITIONS if name != "clean"]
    activity = diagnostics["activity"]
    checks = {
        "clean_meta_auroc_gte_0p65": float(meta["clean"]["auroc"]) >= 0.65,
        "clean_delta_vs_identity_gte_0p03": float(meta["clean"]["auroc"])
        - float(identity["clean"]["auroc"])
        >= 0.03,
        "clean_delta_vs_static_gte_0p03": float(meta["clean"]["auroc"])
        - float(static["clean"]["auroc"])
        >= 0.03,
        "three_of_four_fold_deltas_positive": sum(value > 0.0 for value in fold_deltas)
        >= 3,
        "no_fold_delta_below_minus_0p02": min(fold_deltas) >= -0.02,
        "all_shift_meta_auroc_gte_0p60": all(
            float(meta[name]["auroc"]) >= 0.60 for name in shifts
        ),
        "all_shift_meta_not_below_static": all(
            float(meta[name]["auroc"]) >= float(static[name]["auroc"])
            for name in shifts
        ),
        "clean_tp_retention_gte_0p90": float(meta["clean"]["tp_retention"])
        >= 0.90,
        "clean_fp_rejection_gte_0p20": float(meta["clean"]["fp_rejection"])
        >= 0.20,
        "all_shift_tp_retention_gte_0p85": all(
            float(meta[name]["tp_retention"]) >= 0.85 for name in shifts
        ),
        "all_shift_fp_rejection_gte_0p10": all(
            float(meta[name]["fp_rejection"]) >= 0.10 for name in shifts
        ),
        "tp_median_gt_fp_every_condition": all(
            float(meta[name]["tp_median_score"])
            > float(meta[name]["fp_median_score"])
            for name, _, _ in CONDITIONS
        ),
        "object_beta_delta_vs_static_gte_0p02": float(
            object_beta["clean"]["auroc"]
        )
        - float(static["clean"]["auroc"])
        >= 0.02,
        "object_beta_within_0p02_of_full": float(object_beta["clean"]["auroc"])
        >= float(meta["clean"]["auroc"]) - 0.02,
        "between_sample_beta_rms_gte_0p005": float(
            activity["between_sample_beta_rms"]
        )
        >= 0.005,
        "activation_delta_rms_gte_0p01": float(
            activity["activation_delta_rms_mean"]
        )
        >= 0.01,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
        "clean_meta_auroc": float(meta["clean"]["auroc"]),
        "clean_delta_vs_identity": float(meta["clean"]["auroc"])
        - float(identity["clean"]["auroc"]),
        "clean_delta_vs_static": float(meta["clean"]["auroc"])
        - float(static["clean"]["auroc"]),
        "fold_deltas_meta_vs_static": fold_deltas,
    }


class _StemOnly(nn.Module):
    def __init__(self, stem: HybridConvStem, activation: Optional[nn.Module]) -> None:
        super().__init__()
        self.stem = stem
        self.activation = activation

    def forward(self, images: Tensor) -> Tensor:
        output = self.stem(images)
        return output if self.activation is None else self.activation(output)


def _benchmark_one(
    model: nn.Module, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    model = copy.deepcopy(model).to(device=device, dtype=torch.float16).eval()
    generator = torch.Generator().manual_seed(SEED + 501)
    images = torch.randn(BATCH_SIZE, 3, 256, 256, generator=generator).to(
        device=device, dtype=torch.float16
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    with torch.inference_mode():
        for _ in range(2):
            model(images)
        torch.cuda.synchronize(device)
        for _ in range(int(repeats)):
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            finish.record()
            torch.cuda.synchronize(device)
            elapsed.append(float(start.elapsed_time(finish)))
    peak = int(torch.cuda.max_memory_allocated(device))
    model.cpu()
    del model, images
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "elapsed_ms": elapsed,
        "median_ms": float(statistics.median(elapsed)),
        "peak_allocated_bytes": peak,
        "batch_size": BATCH_SIZE,
        "dtype": "float16",
    }


def _resource_audit(
    stem: HybridConvStem, *, device: torch.device, repeats: int
) -> Dict[str, object]:
    native = _StemOnly(copy.deepcopy(stem), None)
    candidate = _StemOnly(copy.deepcopy(stem), MetaAconC(WIDTH, REDUCTION))
    native_result = _benchmark_one(native, device=device, repeats=repeats)
    candidate_result = _benchmark_one(candidate, device=device, repeats=repeats)
    return {
        "native": native_result,
        "meta_acon": candidate_result,
        "runtime_ratio": float(candidate_result["median_ms"])
        / float(native_result["median_ms"]),
        "peak_memory_ratio": float(candidate_result["peak_allocated_bytes"])
        / float(native_result["peak_allocated_bytes"]),
        "added_parameters": int(
            sum(value.numel() for value in candidate.activation.parameters())
        ),
    }


def _onnx_audit() -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    model = MetaAconC(WIDTH, REDUCTION).cpu().eval()
    generator = torch.Generator().manual_seed(SEED + 601)
    example = torch.randn(1, WIDTH, FEATURE_HEIGHT, FEATURE_WIDTH, generator=generator)
    with torch.inference_mode():
        reference = model(example).numpy()
    with tempfile.TemporaryDirectory(prefix="trkh_meta_acon_onnx_") as directory:
        path = Path(directory) / "meta_acon.onnx"
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["features"],
            output_names=["activated"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        observed = session.run(None, {"features": example.numpy()})[0]
        domains = sorted({str(node.domain) for node in graph.graph.node})
        operators = sorted({str(node.op_type) for node in graph.graph.node})
        payload_sha = _sha256(path)
        payload_bytes = int(path.stat().st_size)
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            parse_ok = bool(parser.parse(path.read_bytes()))
            parser_errors = [
                str(parser.get_error(index)) for index in range(parser.num_errors)
            ]
            engine_ok = False
            engine_bytes = 0
            if parse_ok:
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                engine = builder.build_serialized_network(network, config)
                engine_ok = engine is not None
                if engine is not None:
                    engine_bytes = len(bytes(engine))
        except Exception as exc:  # pragma: no cover - environment dependent
            parse_ok = False
            engine_ok = False
            engine_bytes = 0
            parser_errors = [repr(exc)]
    return {
        "opset": 17,
        "domains": domains,
        "operators": operators,
        "standard_domains_only": all(value in {"", "ai.onnx"} for value in domains),
        "output_shape_match": list(reference.shape) == list(observed.shape),
        "max_abs_error": float(np.max(np.abs(reference - observed))),
        "onnx_sha256_ephemeral": payload_sha,
        "onnx_bytes_ephemeral": payload_bytes,
        "onnx_retained": False,
        "tensorrt_parse": parse_ok,
        "tensorrt_errors": parser_errors,
        "tensorrt_engine_build": engine_ok,
        "tensorrt_engine_bytes_ephemeral": engine_bytes,
        "tensorrt_engine_retained": False,
    }


def _write_oof_csv(
    *,
    path: Path,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Mapping[str, np.ndarray],
) -> None:
    fields = [
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
        "keeper_prediction",
        "cohort",
        "condition",
        *[f"score_{role}" for role in scores],
        *[f"threshold_{role}" for role in scores],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(cohort):
            for condition, _, _ in CONDITIONS:
                payload: Dict[str, object] = {
                    "sample_index": row.sample_index,
                    "source_stem": row.source_stem,
                    "image_path": str(row.image_path),
                    "fold": row.fold,
                    "target": row.target,
                    "keeper_prediction": row.keeper_prediction,
                    "cohort": _cohort_label(row),
                    "condition": condition,
                }
                for role in scores:
                    payload[f"score_{role}"] = float(scores[role][condition][position])
                    payload[f"threshold_{role}"] = float(thresholds[role][position])
                writer.writerow(payload)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    metrics = summary["oof_metrics"]
    meta = metrics["meta_acon"]
    lines = [
        "# Meta-ACON Activation Signal A0 Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Five-epoch pair authorized: `{str(summary['matched_5e_pair_authorized']).lower()}`",
        f"- Structural failures: `{', '.join(summary['structural_gate']['failed_checks']) or 'none'}`",
        f"- Information failures: `{', '.join(summary['information_gate']['failed_checks']) or 'none'}`",
        "",
        "| Condition | Identity AUROC | Static ACON AUROC | Meta-ACON AUROC | Meta TP retention | Meta FP rejection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, _, _ in CONDITIONS:
        lines.append(
            "| {condition} | {identity:.6f} | {static:.6f} | {meta:.6f} | "
            "{tp:.6f} | {fp:.6f} |".format(
                condition=condition,
                identity=float(metrics["identity"][condition]["auroc"]),
                static=float(metrics["acon_static"][condition]["auroc"]),
                meta=float(meta[condition]["auroc"]),
                tp=float(meta[condition]["tp_retention"]),
                fp=float(meta[condition]["fp_rejection"]),
            )
        )
    lines.extend(
        [
            "",
            "## Activity and deployment",
            "",
            f"- Between-sample beta RMS: `{summary['oof_diagnostics']['activity']['between_sample_beta_rms']:.6f}`",
            f"- Activation delta RMS mean: `{summary['oof_diagnostics']['activity']['activation_delta_rms_mean']:.6f}`",
            f"- Runtime ratio: `{summary['resource_audit']['runtime_ratio']:.6f}`",
            f"- Peak-memory ratio: `{summary['resource_audit']['peak_memory_ratio']:.6f}`",
            f"- ONNX Runtime max error: `{summary['onnx_audit']['max_abs_error']:.9g}`",
            f"- TensorRT parse/build: `{summary['onnx_audit']['tensorrt_parse']}/{summary['onnx_audit']['tensorrt_engine_build']}`",
            "",
            "No validation/test data, image-model training, raw-data edit, checkpoint, engine, or reusable feature cache was used or retained.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    forbidden = {".pt", ".pth", ".ckpt", ".engine", ".onnx", ".npy", ".npz"}
    rows = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden retained Meta-ACON artifact: {path}")
        rows.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    payload = {
        "method": METHOD,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "forbidden_binary_count": 0,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **payload,
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    provenance, rows, cohort = _load_locked_inputs(args)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Meta-ACON A0 requires CUDA.")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    equation = _equation_diagnostics(
        Path(provenance["paths"]["acon_source"]), device
    )
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "preflight_created_output": False,
            "provenance": provenance,
            "equation_audit": equation,
        }

    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint = load_checkpoint(
        Path(provenance["paths"]["checkpoint"]), map_location="cpu"
    )
    model = _load_keeper_model(checkpoint)
    stem = getattr(model, "stem", None)
    if not isinstance(stem, HybridConvStem):
        raise TypeError("Keeper stem differs from locked HybridConvStem.")
    resource = _resource_audit(
        stem, device=device, repeats=int(args.benchmark_repeats)
    )
    onnx_audit = _onnx_audit()
    base_dataset, transform, dataset_mapping = _build_dataset(
        checkpoint, rows, Path(provenance["paths"]["data"])
    )
    declaration_loader, declaration_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="meta_acon_a0_keeper_declaration",
    )
    declaration = _replay_keeper_declarations(
        model=model,
        loader=declaration_loader,
        cohort=cohort,
        device=device,
    )
    caches, bboxes, cache_audit = _extract_stem_cache(
        model=model,
        base_dataset=base_dataset,
        transform=transform,
        cohort=cohort,
        args=args,
        device=device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    scores, thresholds, metrics, diagnostics = _run_oof(
        caches=caches,
        bboxes=bboxes,
        cohort=cohort,
        args=args,
        device=device,
    )
    information_gate = assess_information_gate(metrics, diagnostics)

    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_commit_tree_hashes_and_license_exact": True,
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
        "dataset_mapping_train_only_exact": bool(
            dataset_mapping["paths_exact"] and dataset_mapping["train_paths_only"]
        ),
        "keeper_declaration_replay_exact": bool(declaration["exact"]),
        "official_output_error_lte_1e6": float(
            equation["official_output_max_abs_error"]
        )
        <= MAX_EQUATION_ERROR,
        "oracle_output_beta_error_lte_1e6": max(
            float(equation["oracle_output_max_abs_error"]),
            float(equation["oracle_beta_max_abs_error"]),
        )
        <= MAX_EQUATION_ERROR,
        "gradient_error_lte_1e6": float(equation["maximum_gradient_error"])
        <= MAX_EQUATION_ERROR,
        "finite_difference_error_lte_1e4": float(
            equation["finite_difference_error"]
        )
        <= MAX_FINITE_DIFFERENCE_ERROR,
        "paper_init_and_source_discrepancy_exact": bool(
            equation["paper_initialization_exact"]
            and equation["official_source_random_p_initialization_detected"]
            and equation["initialization_discrepancy_disclosed"]
        ),
        "bf16_error_and_gradients_pass": bool(
            float(equation["bf16_max_abs_error"]) <= MAX_BF16_ERROR
            and equation["fp32_gradients_finite"]
            and equation["fp32_gradient_families_nonzero"]
            and equation["bf16_gradients_finite"]
            and equation["bf16_gradient_families_nonzero"]
        ),
        "oof_source_overlap_zero_and_shared_heads": bool(
            diagnostics["source_overlap_zero"]
            and diagnostics["head_initialization_shared_every_role"]
        ),
        "runtime_ratio_lte_1p25": float(resource["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "memory_ratio_lte_1p25": float(resource["peak_memory_ratio"])
        <= MAX_MEMORY_RATIO,
        "onnx_standard_shape_error_pass": bool(
            onnx_audit["standard_domains_only"]
            and onnx_audit["output_shape_match"]
            and float(onnx_audit["max_abs_error"]) <= MAX_ONNX_ERROR
        ),
        "tensorrt_parse_and_build_pass": bool(
            onnx_audit["tensorrt_parse"]
            and onnx_audit["tensorrt_engine_build"]
        ),
        "ephemeral_cache_and_binaries_not_retained": bool(
            not cache_audit["cache_written_to_disk"]
            and not onnx_audit["onnx_retained"]
            and not onnx_audit["tensorrt_engine_retained"]
        ),
        "no_validation_test_or_image_model_training": bool(
            not provenance["validation_data_used"]
            and not provenance["test_data_used"]
            and not provenance["image_model_training_used"]
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

    oof_path = output_dir / "oof_scores.csv"
    _write_oof_csv(
        path=oof_path,
        cohort=cohort,
        scores=scores,
        thresholds=thresholds,
    )
    folds_path = output_dir / "fold_diagnostics.json"
    folds_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": (
            "authorized_for_matched_5e_pair" if authorized else "rejected_at_a0"
        ),
        "matched_5e_pair_authorized": authorized,
        "provenance": provenance,
        "equation_audit": equation,
        "dataset_mapping": dataset_mapping,
        "declaration_replay": declaration,
        "loader_audit": {
            "declaration": declaration_loader_summary,
            "cache": cache_audit,
        },
        "oof_metrics": metrics,
        "oof_diagnostics": diagnostics,
        "information_gate": information_gate,
        "resource_audit": resource,
        "onnx_audit": onnx_audit,
        "structural_gate": structural_gate,
        "validation_data_used": False,
        "test_data_used": False,
        "image_model_training_used": False,
        "activation_readout_training_used": True,
        "raw_dataset_touched": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary)
    del caches, bboxes
    gc.collect()
    torch.cuda.empty_cache()
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_audit(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
