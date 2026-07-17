from __future__ import annotations

import argparse
import ast
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
import sys
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _build_comparisons,
    _build_dataset,
    _failed_export,
    _full_worktree_clean,
    _heat_overlay,
    _onnx_compare,
    _rgb_from_tensor,
    _tracked_worktree_clean,
)
from trkh.tools.audit_class1_boundary_cagrad_readiness import CONDITIONS
from trkh.tools.audit_class1_reference_agem_readiness import (
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
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
    _git_commit,
    _make_loader,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)


METHOD = "wildcat_dense_block2_negative_evidence_a0"
NUM_CLASSES = 5
MAPS_PER_CLASS = 4
PROJECTED_CHANNELS = NUM_CLASSES * MAPS_PER_CLASS
EMBED_DIM = 256
GRID_HEIGHT = 16
GRID_WIDTH = 16
SPATIAL_CELLS = GRID_HEIGHT * GRID_WIDTH
PREFIX_TOKENS = 7
BLOCK_INDEX = 1
K_FRACTION = 0.2
SELECTED_CELLS = 51
ALPHA = 0.7
EXPECTED_KEEPER_PARAMETERS = 7_245_590
EXPECTED_HEAD_PARAMETERS = 5_140
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
FOLD = 0
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MAX_ONNX_ERROR = 1e-5
MAX_TRT_ERROR = 2e-3
MAX_RUNTIME_RATIO = 1.10
MAX_EXTRA_VRAM_GIB = 0.25

LOCKED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
LOCKED_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
LOCKED_FOLD_SUMMARY_SHA256 = (
    "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
)
LOCKED_CIDT_PREDICTIONS_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_PROTOCOL_SHA256 = (
    "0914b04beb8b27496cf980d08a753e74be330e8b25210278f7eb23187433fb0c"
)
LOCKED_PAPER_SHA256 = (
    "2209957a67f669aef37911294ea8bcf52da5c93fbecdaac20798af8297323995"
)
LOCKED_OFFICIAL_COMMIT = "c7d355049a8fe34e98f49d1ff4ae91c996bd3825"
LOCKED_OFFICIAL_TREE = "2b777ae5058f1e71812f7201ddda2dc64cbb127a"
LOCKED_OFFICIAL_POOLING_SHA256 = (
    "d4a5fa61638acd5e39bc9cddf7b67aa67140c28b8c91505f18f7781fdbf504d5"
)
LOCKED_OFFICIAL_MODELS_SHA256 = (
    "be9c5bba1244175a21844fec15543e6f4b674c56a9b4f86b7880f71ba23cdbc5"
)
LOCKED_OFFICIAL_README_SHA256 = (
    "ec817152e3199287f7ca2bbf23f03fb60d8423468c4de034844a80f053d68fd5"
)
LOCKED_OFFICIAL_LICENSE_SHA256 = (
    "90a9749c681f6d7f840925c81f8d04aba7f86f05f00bbccb543b443ad206064a"
)
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
LOCKED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
LOCKED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_TRAIN_ORDER_SHA256 = (
    "bc02b9a8e0a36c96a7d407b654208a87d558beb907a8c7fa48174d1b5b2a7cb5"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only WILDCAT dense-block2 negative-evidence audit. "
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
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
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
            "docs/TRKH_5CLASS_WILDCAT_NEGATIVE_EVIDENCE_READINESS_"
            "PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\Durand_WILDCAT_CVPR_2017.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\wildcat.pytorch"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_wildcat_negative_evidence_a0_20260717"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--finalize-visual-review", action="store_true", default=False)
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", default="")
    parser.add_argument("--expected-summary-sha256", default="")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--maps-per-class", type=int, default=MAPS_PER_CLASS)
    parser.add_argument("--k-fraction", type=float, default=K_FRACTION)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == 4
        and int(args.epochs) == EPOCHS
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and math.isclose(float(args.learning_rate), LEARNING_RATE, abs_tol=0.0)
        and math.isclose(float(args.momentum), MOMENTUM, abs_tol=0.0)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, abs_tol=0.0)
        and int(args.maps_per_class) == MAPS_PER_CLASS
        and math.isclose(float(args.k_fraction), K_FRACTION, abs_tol=0.0)
        and math.isclose(float(args.alpha), ALPHA, abs_tol=0.0)
        and int(args.benchmark_repeats) >= 3
        and int(args.xai_batch_size) >= 1
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "data": Path(args.data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "official_pooling": official / "wildcat" / "pooling.py",
        "official_models": official / "wildcat" / "models.py",
        "official_readme": official / "README.md",
        "official_license": official / "LICENSE",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def locked_training_order(
    fit_indices: Sequence[int], *, seed: int, epochs: int
) -> list[int]:
    generator = np.random.default_rng(int(seed))
    source = np.asarray([int(value) for value in fit_indices], dtype=np.int64)
    order: list[int] = []
    for _ in range(int(epochs)):
        order.extend(int(value) for value in generator.permutation(source).tolist())
    return order


def _cohort_summary(rows: Sequence[CleanTrainRow], *, fold: int) -> Dict[str, object]:
    fit = [row.sample_index for row in rows if int(row.fold) != int(fold)]
    holdout = [row.sample_index for row in rows if int(row.fold) == int(fold)]
    fit_sources = {rows[index].source_stem for index in fit}
    holdout_sources = {rows[index].source_stem for index in holdout}
    return {
        "fit_indices": fit,
        "holdout_indices": holdout,
        "fit_rows": len(fit),
        "holdout_rows": len(holdout),
        "fit_index_sha256": _ordered_index_sha256(fit),
        "holdout_index_sha256": _ordered_index_sha256(holdout),
        "source_overlap": len(fit_sources.intersection(holdout_sources)),
        "fit_class_counts": np.bincount(
            np.asarray([rows[index].target for index in fit], dtype=np.int64),
            minlength=NUM_CLASSES,
        ).tolist(),
        "holdout_class_counts": np.bincount(
            np.asarray([rows[index].target for index in holdout], dtype=np.int64),
            minlength=NUM_CLASSES,
        ).tolist(),
    }


def _git_tree(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=str(root), text=True
    ).strip()


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_committed_implementation(root: Path) -> Dict[str, object]:
    root = Path(root).resolve()
    paths = {
        "auditor": Path(__file__).resolve(),
        "focused_test": root
        / "tests"
        / "test_audit_wildcat_negative_evidence_readiness.py",
        "launcher": root / "scripts" / "run_trkh_wildcat_negative_evidence_a0.ps1",
    }
    relative_paths: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"WILDCAT {name} is missing: {path}")
        relative = path.relative_to(root).as_posix()
        _git_value(root, "ls-files", "--error-unmatch", relative)
        relative_paths[name] = relative
    status = _git_value(root, "status", "--porcelain", "--", *relative_paths.values())
    if status:
        raise ValueError(f"WILDCAT implementation is not commit-clean: {status}")
    return {
        name: {"path": relative_paths[name], "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], Dict[str, object], list[int]]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted WILDCAT protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "fold_summary": _verify_sha256(
            paths["fold_summary"], LOCKED_FOLD_SUMMARY_SHA256, "fold summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "WILDCAT protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "paper"),
        "official_pooling": _verify_sha256(
            paths["official_pooling"],
            LOCKED_OFFICIAL_POOLING_SHA256,
            "official pooling",
        ),
        "official_models": _verify_sha256(
            paths["official_models"], LOCKED_OFFICIAL_MODELS_SHA256, "official models"
        ),
        "official_readme": _verify_sha256(
            paths["official_readme"], LOCKED_OFFICIAL_README_SHA256, "README"
        ),
        "official_license": _verify_sha256(
            paths["official_license"], LOCKED_OFFICIAL_LICENSE_SHA256, "license"
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
    official_commit = _git_commit(paths["official_root"])
    official_tree = _git_tree(paths["official_root"])
    official_clean = _full_worktree_clean(paths["official_root"])
    if official_commit != LOCKED_OFFICIAL_COMMIT or official_tree != LOCKED_OFFICIAL_TREE:
        raise ValueError("Official WILDCAT commit/tree differs from the protocol.")
    if not official_clean:
        raise ValueError("Official WILDCAT worktree is not clean.")
    fold_summary = json.loads(paths["fold_summary"].read_text(encoding="utf-8"))
    if int(fold_summary.get("held_out_fold", -1)) != FOLD:
        raise ValueError("Fold summary does not describe the locked fold zero.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _cohort_summary(rows, fold=int(args.fold))
    if cohorts["fit_rows"] != 7372 or cohorts["holdout_rows"] != 1843:
        raise ValueError("WILDCAT source-disjoint row counts differ.")
    if cohorts["fit_index_sha256"] != LOCKED_FIT_INDEX_SHA256:
        raise ValueError("WILDCAT fit-index hash differs.")
    if cohorts["holdout_index_sha256"] != LOCKED_HOLDOUT_INDEX_SHA256:
        raise ValueError("WILDCAT holdout-index hash differs.")
    if int(cohorts["source_overlap"]) != 0:
        raise ValueError("WILDCAT fit and holdout source groups overlap.")
    order = locked_training_order(
        cohorts["fit_indices"], seed=int(args.seed), epochs=int(args.epochs)
    )
    order_hash = _ordered_index_sha256(order)
    if len(order) != 147_440 or order_hash != LOCKED_TRAIN_ORDER_SHA256:
        raise ValueError("WILDCAT training order differs from the protocol.")
    cohorts["training_order_rows"] = len(order)
    cohorts["training_order_sha256"] = order_hash
    cohorts["training_epochs"] = int(args.epochs)
    root = Path(__file__).resolve().parents[2]
    if not _tracked_worktree_clean(root):
        raise ValueError("Tracked TRKH worktree must be clean for formal WILDCAT A0.")
    implementation = _verify_committed_implementation(root)
    repository_commit = _git_commit(root)
    repository_tree = _git_tree(root)
    upstream_commit = _git_value(
        root, "rev-parse", "origin/classification-only-research"
    )
    if repository_commit != upstream_commit:
        raise ValueError("Formal WILDCAT A0 requires the pushed repository commit.")
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "official_worktree_clean": official_clean,
            "tracked_trkh_worktree_clean": True,
            "implementation_committed_clean": True,
            "implementation": implementation,
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "upstream_commit": upstream_commit,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
        order,
    )


def positive_k(k: float, cells: int) -> int:
    if float(k) <= 0.0:
        return 0
    if float(k) < 1.0:
        return int(round(float(k) * int(cells)))
    return min(int(k), int(cells))


def class_wise_pool(projected: Tensor, maps_per_class: int = MAPS_PER_CLASS) -> Tensor:
    if projected.ndim != 4:
        raise ValueError("Class-wise pooling requires [B,C,H,W].")
    batch, channels, height, width = projected.shape
    if int(maps_per_class) <= 0 or channels % int(maps_per_class) != 0:
        raise ValueError("Projected channels must be divisible by maps_per_class.")
    classes = channels // int(maps_per_class)
    return projected.reshape(
        batch, classes, int(maps_per_class), height, width
    ).mean(dim=2)


def spatial_pool_components(
    class_maps: Tensor, *, k_fraction: float = K_FRACTION
) -> Dict[str, Tensor]:
    if class_maps.ndim != 4:
        raise ValueError("Spatial pooling requires [B,C,H,W].")
    flat = class_maps.flatten(2)
    selected = positive_k(float(k_fraction), int(flat.size(2)))
    if selected <= 0:
        raise ValueError("WILDCAT requires at least one selected spatial cell.")
    top_values, top_indices = torch.topk(flat, k=selected, dim=2, largest=True)
    bottom_values, bottom_indices = torch.topk(
        flat, k=selected, dim=2, largest=False
    )
    return {
        "top_mean": top_values.mean(dim=2),
        "bottom_mean": bottom_values.mean(dim=2),
        "top_indices": top_indices,
        "bottom_indices": bottom_indices,
    }


def wildcat_pool(
    class_maps: Tensor,
    *,
    k_fraction: float = K_FRACTION,
    alpha: float = ALPHA,
) -> Tensor:
    components = spatial_pool_components(class_maps, k_fraction=k_fraction)
    if float(alpha) == 0.0:
        return components["top_mean"]
    return (
        components["top_mean"] + float(alpha) * components["bottom_mean"]
    ) / 2.0


class WildcatSpatialHead(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"gap", "top_only", "wildcat"}:
            raise ValueError(f"Unsupported WILDCAT head mode: {mode}")
        self.mode = str(mode)
        self.projection = nn.Conv2d(
            EMBED_DIM, PROJECTED_CHANNELS, kernel_size=1, bias=True
        )

    def components(self, features: Tensor) -> Dict[str, Tensor]:
        projected = self.projection(features)
        class_maps = class_wise_pool(projected)
        spatial = spatial_pool_components(class_maps)
        if self.mode == "gap":
            logits = class_maps.mean(dim=(2, 3))
        elif self.mode == "top_only":
            logits = spatial["top_mean"]
        else:
            logits = (spatial["top_mean"] + ALPHA * spatial["bottom_mean"]) / 2.0
        return {"projected": projected, "class_maps": class_maps, "logits": logits, **spatial}

    def forward(self, features: Tensor) -> Tensor:
        return self.components(features)["logits"]

    def no_bottom_logits(self, features: Tensor) -> Tensor:
        return self.components(features)["top_mean"] / 2.0


def _independent_pool_oracle(
    projected: Tensor, *, alpha: float
) -> tuple[Tensor, Tensor, Tensor]:
    if projected.size(1) != PROJECTED_CHANNELS:
        raise ValueError("Independent oracle requires the locked channel count.")
    class_rows = []
    top_rows = []
    bottom_rows = []
    for sample in projected:
        sample_classes = []
        sample_top = []
        sample_bottom = []
        for class_index in range(NUM_CLASSES):
            start = class_index * MAPS_PER_CLASS
            class_map = sample[start : start + MAPS_PER_CLASS].mean(dim=0)
            ordered = torch.sort(class_map.flatten(), descending=True).values
            top = ordered[:SELECTED_CELLS].mean()
            bottom = ordered[-SELECTED_CELLS:].mean()
            score = top if float(alpha) == 0.0 else (top + float(alpha) * bottom) / 2.0
            sample_classes.append(score)
            sample_top.append(top)
            sample_bottom.append(bottom)
        class_rows.append(torch.stack(sample_classes))
        top_rows.append(torch.stack(sample_top))
        bottom_rows.append(torch.stack(sample_bottom))
    return torch.stack(class_rows), torch.stack(top_rows), torch.stack(bottom_rows)


def _official_source_classes(source_path: Path):
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"WildcatPool2dFunction", "ClassWisePoolFunction"}
    }
    if set(found) != {"WildcatPool2dFunction", "ClassWisePoolFunction"}:
        raise ValueError("Official WILDCAT source classes are missing.")

    def build(source_name: str, target_name: str):
        source_class = found[source_name]
        methods = [
            copy.deepcopy(node)
            for node in source_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"get_positive_k", "forward"}
        ]
        noop = ast.parse(
            "def save_for_backward(self, *values):\n    self.saved = values\n"
        ).body[0]
        generated = ast.ClassDef(
            name=target_name,
            bases=[],
            keywords=[],
            body=[noop, *methods],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[generated], type_ignores=[]))
        namespace = {"torch": torch, "sys": sys}
        exec(compile(module, str(source_path), "exec"), namespace)
        return namespace[target_name]

    return build("WildcatPool2dFunction", "OfficialWildcat"), build(
        "ClassWisePoolFunction", "OfficialClassWise"
    )


def _finite_difference_error() -> float:
    generator = torch.Generator().manual_seed(SEED + 17)
    value = torch.randn(
        1,
        PROJECTED_CHANNELS,
        GRID_HEIGHT,
        GRID_WIDTH,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    output = wildcat_pool(class_wise_pool(value)).square().mean()
    analytic = torch.autograd.grad(output, value)[0]
    index = (0, 3, 7, 9)
    epsilon = 1e-5
    plus = value.detach().clone()
    minus = value.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    plus_value = wildcat_pool(class_wise_pool(plus)).square().mean()
    minus_value = wildcat_pool(class_wise_pool(minus)).square().mean()
    numeric = (plus_value - minus_value) / (2.0 * epsilon)
    return float((analytic[index] - numeric).abs().item())


def _equation_diagnostics(source_path: Path, device: torch.device) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 7)
    projected = torch.randn(
        2,
        PROJECTED_CHANNELS,
        GRID_HEIGHT,
        GRID_WIDTH,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    local_maps = class_wise_pool(projected)
    local = wildcat_pool(local_maps)
    reference, reference_top, reference_bottom = _independent_pool_oracle(
        projected, alpha=ALPHA
    )
    local_gradient = torch.autograd.grad(local.square().mean(), projected, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(
        reference.square().mean(), projected, retain_graph=True
    )[0]

    official_wildcat, official_classwise = _official_source_classes(source_path)
    official_classwise_instance = official_classwise()
    official_classwise_instance.num_maps = MAPS_PER_CLASS
    official_maps = official_classwise_instance.forward(projected.detach().float())
    official_wildcat_instance = official_wildcat()
    official_wildcat_instance.kmax = K_FRACTION
    official_wildcat_instance.kmin = K_FRACTION
    official_wildcat_instance.alpha = ALPHA
    official = official_wildcat_instance.forward(official_maps)
    source_text = Path(source_path).read_text(encoding="utf-8")

    bf16_input = projected.detach().to(device=device, dtype=torch.bfloat16)
    bf16 = wildcat_pool(class_wise_pool(bf16_input)).float().cpu()
    fp32 = wildcat_pool(class_wise_pool(projected.detach().float())).cpu()
    ties = torch.ones(2, PROJECTED_CHANNELS, 5, 7)
    non_square = wildcat_pool(
        class_wise_pool(ties), k_fraction=K_FRACTION, alpha=ALPHA
    )
    invalid_rejected = False
    try:
        class_wise_pool(torch.zeros(1, 19, 4, 4))
    except ValueError:
        invalid_rejected = True
    invalid_k_rejected = False
    try:
        spatial_pool_components(torch.zeros(1, NUM_CLASSES, 4, 4), k_fraction=0.0)
    except ValueError:
        invalid_k_rejected = True
    invalid_mode_rejected = False
    try:
        WildcatSpatialHead("unsupported")
    except ValueError:
        invalid_mode_rejected = True
    return {
        "selected_cells": positive_k(K_FRACTION, SPATIAL_CELLS),
        "classwise_oracle_max_abs_error": float(
            (local_maps - projected.reshape(2, 5, 4, 16, 16).mean(dim=2))
            .abs()
            .amax()
            .item()
        ),
        "pool_oracle_max_abs_error": float((local - reference).abs().amax().item()),
        "top_oracle_max_abs_error": float(
            (spatial_pool_components(local_maps)["top_mean"] - reference_top)
            .abs()
            .amax()
            .item()
        ),
        "bottom_oracle_max_abs_error": float(
            (spatial_pool_components(local_maps)["bottom_mean"] - reference_bottom)
            .abs()
            .amax()
            .item()
        ),
        "gradient_oracle_max_abs_error": float(
            (local_gradient - reference_gradient).abs().amax().item()
        ),
        "official_source_max_abs_error": float(
            (official.double() - local.detach()).abs().amax().item()
        ),
        "finite_difference_max_abs_error": _finite_difference_error(),
        "bf16_vs_fp32_max_abs_error": float((bf16 - fp32).abs().amax().item()),
        "ties_finite": bool(torch.isfinite(non_square).all()),
        "non_square_shape": [int(value) for value in non_square.shape],
        "invalid_channel_count_rejected": invalid_rejected,
        "invalid_k_rejected": invalid_k_rejected,
        "invalid_mode_rejected": invalid_mode_rejected,
        "legacy_identity_predicate_disclosed": "self.alpha is not 0" in source_text,
        "official_divide_by_two_present": ".div_(2)" in source_text,
    }


class DenseBlock2Capture:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.value: Optional[Tensor] = None
        self.output_shape: Optional[tuple[int, ...]] = None
        self.call_count = 0
        self._handle = None

    def _hook(self, _module, _inputs, output) -> None:
        tokens = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            raise TypeError("Block-2 hook did not receive token tensor output.")
        self.output_shape = tuple(int(value) for value in tokens.shape)
        prefix = int(getattr(self.model, "num_prefix_tokens", -1))
        patches = tokens[:, prefix:]
        if prefix != PREFIX_TOKENS or tuple(patches.shape[1:]) != (
            SPATIAL_CELLS,
            EMBED_DIM,
        ):
            raise ValueError(
                f"Dense block-2 contract differs: prefix={prefix}, shape={tuple(patches.shape)}"
            )
        self.value = patches.transpose(1, 2).reshape(
            int(patches.size(0)), EMBED_DIM, GRID_HEIGHT, GRID_WIDTH
        )
        self.call_count += 1

    def take(self) -> Tensor:
        if self.value is None:
            raise RuntimeError("Dense block-2 hook did not capture a value.")
        value = self.value
        self.value = None
        return value

    def __enter__(self):
        self._handle = self.model.blocks[BLOCK_INDEX].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _construct_keeper_and_heads(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, Dict[str, WildcatSpatialHead], Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
        raise ValueError("Keeper class order is invalid.")
    set_seed(SEED, deterministic=True)
    keeper = create_model(num_classes=NUM_CLASSES, model_config=model_config).eval()
    load_model_state(keeper, dict(model_state), strict=True)
    for parameter in keeper.parameters():
        parameter.requires_grad_(False)

    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        prototype = WildcatSpatialHead("gap")
    rng_after = torch.get_rng_state().clone()
    heads = {
        "gap": copy.deepcopy(prototype),
        "top_only": copy.deepcopy(prototype),
        "wildcat": copy.deepcopy(prototype),
    }
    heads["top_only"].mode = "top_only"
    heads["wildcat"].mode = "wildcat"
    hashes = {name: _state_sha256(head) for name, head in heads.items()}
    parameter_count = sum(value.numel() for value in prototype.parameters())
    return keeper, heads, {
        "keeper_parameters": sum(value.numel() for value in keeper.parameters()),
        "head_parameters": parameter_count,
        "parameter_schema": {
            name: [int(value) for value in parameter.shape]
            for name, parameter in prototype.named_parameters()
        },
        "head_initial_state_sha256": hashes,
        "heads_initially_bit_exact": len(set(hashes.values())) == 1,
        "isolated_initialization_rng_restored": torch.equal(rng_before, rng_after),
        "keeper_state_sha256": _state_sha256(keeper),
    }


def _structural_shape_replay(keeper: nn.Module, device: torch.device) -> Dict[str, object]:
    keeper = keeper.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED + 29)
    images = torch.randn(1, 3, 256, 256, generator=generator).to(device)
    with DenseBlock2Capture(keeper) as capture:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            logits = keeper(images)
        dense = capture.take()
    return {
        "block_output_shape": list(capture.output_shape or ()),
        "prefix_tokens": int(getattr(keeper, "num_prefix_tokens", -1)),
        "dense_feature_shape": [int(value) for value in dense.shape],
        "logit_shape": [int(value) for value in logits.shape],
        "hook_calls": capture.call_count,
        "finite": bool(torch.isfinite(dense).all() and torch.isfinite(logits).all()),
    }


def _stream_file_sha256(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_fit_cache(
    *,
    keeper: nn.Module,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    cache_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[np.memmap, np.ndarray, np.ndarray, Dict[str, object], tuple]:
    if cache_path.exists():
        raise FileExistsError(f"Temporary WILDCAT cache already exists: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader, loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="wildcat_fit_dense_block2_cache",
        seed=int(args.seed) + 100,
    )
    cache = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(indices), EMBED_DIM, GRID_HEIGHT, GRID_WIDTH),
    )
    targets = np.empty(len(indices), dtype=np.int64)
    sample_indices = np.empty(len(indices), dtype=np.int64)
    keeper = keeper.to(device).eval()
    resource_batch = None
    offset = 0
    started = time.perf_counter()
    with DenseBlock2Capture(keeper) as capture, torch.inference_mode():
        for images_cpu, targets_cpu, metadata_cpu in loader:
            if resource_batch is None:
                resource_batch = (
                    images_cpu.detach().clone(),
                    targets_cpu.detach().clone(),
                    {
                        key: value.detach().clone()
                        if torch.is_tensor(value)
                        else value
                        for key, value in metadata_cpu.items()
                    },
                )
            images = images_cpu.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=True
            ):
                logits, _ = _forward_classification_with_metadata(
                    keeper, images, metadata_cpu, device=device
                )
            dense = capture.take()
            sample_tensor = metadata_cpu.get("sample_index")
            if not torch.is_tensor(sample_tensor):
                raise ValueError("Fit feature cache batch lacks sample_index.")
            count = int(targets_cpu.numel())
            row_slice = slice(offset, offset + count)
            cache[row_slice] = (
                dense.detach().to(dtype=torch.float16).cpu().contiguous().numpy()
            )
            targets[row_slice] = targets_cpu.detach().long().numpy()
            sample_indices[row_slice] = sample_tensor.detach().long().numpy()
            if not torch.isfinite(logits).all() or not torch.isfinite(dense).all():
                raise ValueError("Fit feature extraction produced non-finite values.")
            offset += count
            if offset % 1024 < count or offset == len(indices):
                print(
                    json.dumps(
                        {
                            "phase": "wildcat_fit_feature_cache",
                            "processed": offset,
                            "rows": len(indices),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    cache.flush()
    if resource_batch is None or offset != len(indices):
        raise RuntimeError("WILDCAT fit cache extraction is incomplete.")
    expected = [int(value) for value in indices]
    if sample_indices.tolist() != expected:
        raise ValueError("WILDCAT fit cache changed sample order.")
    info = {
        "rows": offset,
        "shape": [len(indices), EMBED_DIM, GRID_HEIGHT, GRID_WIDTH],
        "dtype": "float16",
        "bytes": int(cache_path.stat().st_size),
        "sha256": _stream_file_sha256(cache_path),
        "sample_index_sha256": _ordered_index_sha256(sample_indices.tolist()),
        "hook_calls": int(capture.call_count),
        "loader": loader_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
        "temporary_path": str(cache_path.resolve()),
        "temporary_cache_removed": False,
    }
    return cache, targets, sample_indices, info, resource_batch


def _gradient_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _train_heads(
    *,
    prototypes: Mapping[str, WildcatSpatialHead],
    cache: np.memmap,
    targets: np.ndarray,
    sample_indices: np.ndarray,
    order: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, WildcatSpatialHead], Dict[str, object]]:
    names = ("gap", "top_only", "wildcat")
    heads = {name: copy.deepcopy(prototypes[name]).to(device).train() for name in names}
    optimizers = {
        name: torch.optim.SGD(
            heads[name].parameters(),
            lr=float(args.learning_rate),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
        for name in names
    }
    optimizer_configuration = {
        name: {
            "name": type(optimizer).__name__,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "momentum": float(optimizer.param_groups[0]["momentum"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "dampening": float(optimizer.param_groups[0]["dampening"]),
            "nesterov": bool(optimizer.param_groups[0]["nesterov"]),
            "parameter_groups": len(optimizer.param_groups),
            "parameter_tensors": sum(
                len(group["params"]) for group in optimizer.param_groups
            ),
        }
        for name, optimizer in optimizers.items()
    }
    initial_states = {
        name: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        for name, head in heads.items()
    }
    positions = np.full(int(sample_indices.max()) + 1, -1, dtype=np.int64)
    positions[sample_indices] = np.arange(sample_indices.size, dtype=np.int64)
    if any(int(index) >= positions.size or positions[int(index)] < 0 for index in order):
        raise ValueError("Training order references a sample outside the fit cache.")

    history: Dict[str, list[Dict[str, object]]] = {name: [] for name in names}
    updates = {name: 0 for name in names}
    gradient_seen_nonzero = {
        name: {parameter_name: False for parameter_name, _ in head.named_parameters()}
        for name, head in heads.items()
    }
    first_gradients: Optional[Dict[str, object]] = None
    started = time.perf_counter()
    fit_rows = int(sample_indices.size)
    for epoch in range(int(args.epochs)):
        epoch_order = order[epoch * fit_rows : (epoch + 1) * fit_rows]
        loss_sums = {name: 0.0 for name in names}
        gradient_sums = {name: 0.0 for name in names}
        logit_abs_sums = {name: 0.0 for name in names}
        logit_square_sums = {name: 0.0 for name in names}
        logit_maxima = {name: 0.0 for name in names}
        logit_elements = {name: 0 for name in names}
        occurrences = 0
        for batch_start in range(0, fit_rows, int(args.batch_size)):
            batch_indices = np.asarray(
                epoch_order[batch_start : batch_start + int(args.batch_size)],
                dtype=np.int64,
            )
            batch_positions = positions[batch_indices]
            feature_array = np.array(
                cache[batch_positions], dtype=np.float32, copy=True
            )
            target_array = np.array(targets[batch_positions], dtype=np.int64, copy=True)
            features = torch.from_numpy(feature_array).to(device=device, non_blocking=True)
            target_tensor = torch.from_numpy(target_array).to(device=device, non_blocking=True)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            logits = {name: heads[name](features) for name in names}
            losses = {
                name: F.cross_entropy(logits[name], target_tensor) for name in names
            }
            for name in names:
                if not torch.isfinite(logits[name]).all() or not torch.isfinite(
                    losses[name]
                ):
                    raise ValueError(
                        f"WILDCAT {name} produced non-finite training values."
                    )
            sum(losses.values()).backward()
            for name in names:
                for parameter_name, parameter in heads[name].named_parameters():
                    gradient = parameter.grad
                    if gradient is None or not torch.isfinite(gradient).all():
                        raise ValueError(
                            f"WILDCAT {name}/{parameter_name} gradient is absent or non-finite."
                        )
                    gradient_seen_nonzero[name][parameter_name] = bool(
                        gradient_seen_nonzero[name][parameter_name]
                        or torch.count_nonzero(gradient).item() > 0
                    )
            gradients = {name: _gradient_norm(heads[name]) for name in names}
            if first_gradients is None:
                first_gradients = {
                    name: {
                        parameter_name: {
                            "present": parameter.grad is not None,
                            "finite": bool(
                                parameter.grad is not None
                                and torch.isfinite(parameter.grad).all()
                            ),
                            "nonzero": bool(
                                parameter.grad is not None
                                and torch.count_nonzero(parameter.grad).item() > 0
                            ),
                            "norm": 0.0
                            if parameter.grad is None
                            else float(parameter.grad.detach().float().norm().item()),
                        }
                        for parameter_name, parameter in heads[name].named_parameters()
                    }
                    for name in names
                }
            for name in names:
                optimizers[name].step()
                updates[name] += 1
                count = int(target_tensor.numel())
                loss_sums[name] += float(losses[name].detach().item()) * count
                gradient_sums[name] += gradients[name]
                detached_logits = logits[name].detach().float()
                logit_abs_sums[name] += float(detached_logits.abs().sum().item())
                logit_square_sums[name] += float(
                    detached_logits.square().sum().item()
                )
                logit_maxima[name] = max(
                    logit_maxima[name], float(detached_logits.abs().amax().item())
                )
                logit_elements[name] += int(detached_logits.numel())
            occurrences += int(target_tensor.numel())
        for name in names:
            history[name].append(
                {
                    "variant": name,
                    "epoch": epoch + 1,
                    "occurrences": occurrences,
                    "updates": updates[name],
                    "mean_loss": loss_sums[name] / occurrences,
                    "mean_gradient_norm": gradient_sums[name]
                    / math.ceil(fit_rows / int(args.batch_size)),
                    "mean_absolute_logit": logit_abs_sums[name]
                    / max(1, logit_elements[name]),
                    "rms_logit": math.sqrt(
                        logit_square_sums[name] / max(1, logit_elements[name])
                    ),
                    "maximum_absolute_logit": logit_maxima[name],
                }
            )
        print(
            json.dumps(
                {
                    "phase": "wildcat_head_training",
                    "epoch": epoch + 1,
                    "epochs": int(args.epochs),
                    "loss": {
                        name: history[name][-1]["mean_loss"] for name in names
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    if first_gradients is None:
        raise RuntimeError("WILDCAT head training executed no batch.")
    trained = {name: copy.deepcopy(heads[name]).cpu().eval() for name in names}
    final_parameters_finite = {
        name: all(torch.isfinite(parameter).all() for parameter in head.parameters())
        for name, head in trained.items()
    }
    movement = {
        name: {
            key: {
                "changed": not torch.equal(initial_states[name][key], value.detach().cpu()),
                "initial_sha256": hashlib.sha256(
                    initial_states[name][key].contiguous().numpy().tobytes()
                ).hexdigest(),
                "final_sha256": hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
            for key, value in trained[name].state_dict().items()
        }
        for name in names
    }
    summary = {
        "history": history,
        "updates": updates,
        "occurrences": {name: int(args.epochs) * fit_rows for name in names},
        "training_order_sha256": _ordered_index_sha256(order),
        "optimizer_configuration": optimizer_configuration,
        "optimizer_configurations_equal": len(
            {
                json.dumps(value, sort_keys=True)
                for value in optimizer_configuration.values()
            }
        )
        == 1,
        "optimizer_state_parameter_counts": {
            name: len(optimizer.state) for name, optimizer in optimizers.items()
        },
        "first_batch_gradients": first_gradients,
        "gradient_seen_nonzero": gradient_seen_nonzero,
        "all_gradients_seen_nonzero": all(
            all(values.values()) for values in gradient_seen_nonzero.values()
        ),
        "all_training_values_finite": True,
        "all_final_parameters_finite": all(final_parameters_finite.values()),
        "final_parameters_finite": final_parameters_finite,
        "movement": movement,
        "final_state_sha256": {
            name: _state_sha256(trained[name]) for name in names
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "simultaneous_matched_batch_training": True,
    }
    del heads, optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return trained, summary


def _prediction_row(
    *, sample_index: int, target: int, probabilities: Tensor
) -> Dict[str, object]:
    values = probabilities.detach().float().cpu()
    row: Dict[str, object] = {
        "sample_index": int(sample_index),
        "target": int(target),
        "prediction": int(values.argmax().item()),
    }
    for class_index in range(NUM_CLASSES):
        row[f"prob_{class_index}"] = float(values[class_index].item())
    return row


def _condition_loader(
    *,
    name: str,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    args: argparse.Namespace,
):
    if name == "clean":
        return _make_loader(
            base_dataset=dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context="wildcat_holdout_clean",
            seed=int(args.seed) + 200,
        )
    lighting = {
        condition: (brightness, contrast)
        for condition, brightness, contrast in LIGHTING_CONDITIONS
    }
    brightness, contrast = lighting[name]
    return _make_lighting_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        brightness=brightness,
        contrast=contrast,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"wildcat_holdout_{name}",
        seed=int(args.seed) + 201 + list(lighting).index(name),
    )


def _bbox_mask(bboxes: Tensor, *, height: int, width: int) -> Tensor:
    boxes = bboxes.detach().float().clamp(0.0, 1.0)
    x_center, y_center, box_width, box_height = boxes.unbind(dim=1)
    left = x_center - box_width / 2.0
    right = x_center + box_width / 2.0
    top = y_center - box_height / 2.0
    bottom = y_center + box_height / 2.0
    xs = (torch.arange(width, device=boxes.device).float() + 0.5) / width
    ys = (torch.arange(height, device=boxes.device).float() + 0.5) / height
    mask = (
        (xs[None, None, :] >= left[:, None, None])
        & (xs[None, None, :] <= right[:, None, None])
        & (ys[None, :, None] >= top[:, None, None])
        & (ys[None, :, None] <= bottom[:, None, None])
    )
    empty = ~mask.flatten(1).any(dim=1)
    if bool(empty.any()):
        x_index = (x_center[empty] * width).long().clamp(0, width - 1)
        y_index = (y_center[empty] * height).long().clamp(0, height - 1)
        mask[empty] = False
        mask[empty, y_index, x_index] = True
    return mask


def _direction_margin(logits: Tensor) -> Tensor:
    restricted = torch.tensor(
        RESTRICTED_NEGATIVE_CLASSES, device=logits.device, dtype=torch.long
    )
    return logits[:, FOCUS_CLASS] - logits.index_select(1, restricted).amax(dim=1)


def _safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.size == 0 or np.unique(label_array).size != 2:
        return None
    return float(roc_auc_score(label_array, score_array))


def _summarize_mechanism_rows(
    values: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    def mean_or_none(rows: Sequence[Mapping[str, object]], key: str) -> Optional[float]:
        if not rows:
            return None
        return float(np.mean([float(row[key]) for row in rows]))

    output: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        rows = [row for row in values if str(row["condition"]) == condition]
        direction = [
            row
            for row in rows
            if int(row["target"]) == FOCUS_CLASS
            or int(row["target"]) in RESTRICTED_NEGATIVE_CLASSES
        ]
        labels = [int(int(row["target"]) == FOCUS_CLASS) for row in direction]
        tp_rows = [
            row
            for row in rows
            if int(row["target"]) == FOCUS_CLASS
            and int(row["wildcat_prediction"]) == FOCUS_CLASS
        ]
        bottom_direction = [
            row
            for row in rows
            if (
                int(row["target"]) == FOCUS_CLASS
                and int(row["wildcat_prediction"]) == FOCUS_CLASS
            )
            or int(row["target"]) in RESTRICTED_NEGATIVE_CLASSES
        ]
        bottom_labels = [
            int(int(row["target"]) == FOCUS_CLASS) for row in bottom_direction
        ]
        output[condition] = {
            "rows": len(rows),
            "direction_rows": len(direction),
            "bottom_direction_rows": len(bottom_direction),
            "candidate_class1_tp_rows": len(tp_rows),
            "direction_auroc": {
                name: _safe_auroc(
                    labels, [float(row[f"{name}_direction_margin"]) for row in direction]
                )
                for name in ("gap", "top_only", "wildcat", "no_bottom")
            },
            "bottom_margin_auroc": _safe_auroc(
                bottom_labels,
                [float(row["wildcat_bottom_margin"]) for row in bottom_direction],
            ),
            "tp_top_bbox_fraction": mean_or_none(tp_rows, "class1_top_bbox_fraction"),
            "tp_bottom_bbox_fraction": mean_or_none(
                tp_rows, "class1_bottom_bbox_fraction"
            ),
            "tp_bbox_chance": mean_or_none(tp_rows, "bbox_grid_fraction"),
            "tp_top_alignment_above_chance": None
            if not tp_rows
            else float(
                mean_or_none(tp_rows, "class1_top_bbox_fraction")
                - mean_or_none(tp_rows, "bbox_grid_fraction")
            ),
            "mean_class1_map_entropy": mean_or_none(rows, "class1_map_entropy"),
            "mean_class_map_effective_rank": mean_or_none(
                rows, "class_map_effective_rank"
            ),
            "mean_top_by_class": [
                mean_or_none(rows, f"class_{class_index}_top_mean")
                for class_index in range(NUM_CLASSES)
            ],
            "mean_bottom_by_class": [
                mean_or_none(rows, f"class_{class_index}_bottom_mean")
                for class_index in range(NUM_CLASSES)
            ],
            "all_finite": bool(
                all(
                    math.isfinite(float(value))
                    for row in rows
                    for key, value in row.items()
                    if key not in {"condition"}
                    and key not in {"sample_index", "target"}
                )
            ),
        }
    return output


def _evaluate_conditions(
    *,
    keeper: nn.Module,
    heads: Mapping[str, WildcatSpatialHead],
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    rows: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, Dict[str, list[Dict[str, object]]]], list[Dict[str, object]], Dict[str, object]]:
    keeper = keeper.to(device).eval()
    device_heads = {
        name: copy.deepcopy(head).to(device).eval() for name, head in heads.items()
    }
    for head in device_heads.values():
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    predictions: Dict[str, Dict[str, list[Dict[str, object]]]] = {
        name: {} for name in ("raw", "gap", "top_only", "wildcat", "no_bottom")
    }
    mechanism_rows: list[Dict[str, object]] = []
    loaders: Dict[str, object] = {}
    hook_calls: Dict[str, int] = {}
    for condition in CONDITIONS:
        loader, loader_summary = _condition_loader(
            name=condition,
            dataset=dataset,
            transform=transform,
            indices=indices,
            args=args,
        )
        loaders[condition] = loader_summary
        condition_rows = {name: [] for name in predictions}
        processed = 0
        started = time.perf_counter()
        with DenseBlock2Capture(keeper) as capture, torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=True
                ):
                    raw_logits, _ = _forward_classification_with_metadata(
                        keeper, images, metadata_cpu, device=device
                    )
                dense = capture.take().float()
                gap_logits = device_heads["gap"](dense)
                top_logits = device_heads["top_only"](dense)
                candidate_components = device_heads["wildcat"].components(dense)
                wildcat_logits = candidate_components["logits"]
                no_bottom_logits = candidate_components["top_mean"] / 2.0
                logits_by_name = {
                    "raw": raw_logits.float(),
                    "gap": gap_logits,
                    "top_only": top_logits,
                    "wildcat": wildcat_logits,
                    "no_bottom": no_bottom_logits,
                }
                probabilities = {
                    name: F.softmax(logits.float(), dim=1)
                    for name, logits in logits_by_name.items()
                }
                sample_tensor = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_tensor):
                    raise ValueError("Holdout metadata lacks sample_index.")
                sample_values = [int(value) for value in sample_tensor.tolist()]
                bbox_cpu = metadata_cpu.get("crop_bbox")
                if not torch.is_tensor(bbox_cpu):
                    bbox_cpu = metadata_cpu.get("bbox")
                if not torch.is_tensor(bbox_cpu):
                    raise ValueError("Holdout metadata lacks bbox/crop_bbox.")
                bbox = bbox_cpu.to(device=device, dtype=torch.float32)
                bbox_grid = _bbox_mask(
                    bbox, height=GRID_HEIGHT, width=GRID_WIDTH
                ).flatten(1)
                top_indices = candidate_components["top_indices"][:, FOCUS_CLASS]
                bottom_indices = candidate_components["bottom_indices"][:, FOCUS_CLASS]
                top_in_bbox = bbox_grid.gather(1, top_indices).float().mean(dim=1)
                bottom_in_bbox = bbox_grid.gather(1, bottom_indices).float().mean(dim=1)
                bbox_fraction = bbox_grid.float().mean(dim=1)
                class1_map = candidate_components["class_maps"][:, FOCUS_CLASS].flatten(1)
                spatial_probability = F.softmax(class1_map, dim=1)
                entropy = -(
                    spatial_probability
                    * spatial_probability.clamp_min(1e-12).log()
                ).sum(dim=1) / math.log(SPATIAL_CELLS)
                singular = torch.linalg.svdvals(
                    candidate_components["class_maps"].flatten(2).float()
                )
                singular_probability = singular / singular.sum(dim=1, keepdim=True).clamp_min(1e-12)
                effective_rank = torch.exp(
                    -(
                        singular_probability
                        * singular_probability.clamp_min(1e-12).log()
                    ).sum(dim=1)
                )
                direction = {
                    name: _direction_margin(logits)
                    for name, logits in logits_by_name.items()
                    if name != "raw"
                }
                bottom_margin = _direction_margin(candidate_components["bottom_mean"])
                for local, sample_index in enumerate(sample_values):
                    target = int(targets_cpu[local].item())
                    for name in predictions:
                        condition_rows[name].append(
                            _prediction_row(
                                sample_index=sample_index,
                                target=target,
                                probabilities=probabilities[name][local],
                            )
                        )
                    mechanism_rows.append(
                        {
                            "condition": condition,
                            "sample_index": sample_index,
                            "target": target,
                            "wildcat_prediction": int(
                                wildcat_logits[local].argmax().item()
                            ),
                            "gap_direction_margin": float(direction["gap"][local].item()),
                            "top_only_direction_margin": float(
                                direction["top_only"][local].item()
                            ),
                            "wildcat_direction_margin": float(
                                direction["wildcat"][local].item()
                            ),
                            "no_bottom_direction_margin": float(
                                direction["no_bottom"][local].item()
                            ),
                            "wildcat_bottom_margin": float(bottom_margin[local].item()),
                            "class1_top_mean": float(
                                candidate_components["top_mean"][local, FOCUS_CLASS].item()
                            ),
                            "class1_bottom_mean": float(
                                candidate_components["bottom_mean"][local, FOCUS_CLASS].item()
                            ),
                            "class1_top_bbox_fraction": float(top_in_bbox[local].item()),
                            "class1_bottom_bbox_fraction": float(
                                bottom_in_bbox[local].item()
                            ),
                            "bbox_grid_fraction": float(bbox_fraction[local].item()),
                            "class1_map_entropy": float(entropy[local].item()),
                            "class_map_effective_rank": float(
                                effective_rank[local].item()
                            ),
                            **{
                                f"class_{class_index}_top_mean": float(
                                    candidate_components["top_mean"][
                                        local, class_index
                                    ].item()
                                )
                                for class_index in range(NUM_CLASSES)
                            },
                            **{
                                f"class_{class_index}_bottom_mean": float(
                                    candidate_components["bottom_mean"][
                                        local, class_index
                                    ].item()
                                )
                                for class_index in range(NUM_CLASSES)
                            },
                        }
                    )
                processed += len(sample_values)
                if processed % 512 < len(sample_values) or processed == len(indices):
                    print(
                        json.dumps(
                            {
                                "phase": "wildcat_holdout_evaluation",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(indices),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
        hook_calls[condition] = capture.call_count
        expected = [int(value) for value in indices]
        for name in predictions:
            observed = [int(row["sample_index"]) for row in condition_rows[name]]
            if observed != expected:
                raise ValueError(f"{condition}/{name} prediction order differs.")
            predictions[name][condition] = condition_rows[name]
    clean_raw_mismatches = sum(
        int(row["prediction"])
        != int(rows[int(row["sample_index"])].keeper_prediction)
        for row in predictions["raw"]["clean"]
    )
    summary = {
        "loaders": loaders,
        "hook_calls": hook_calls,
        "clean_raw_cidt_argmax_mismatches": clean_raw_mismatches,
        "mechanism": _summarize_mechanism_rows(mechanism_rows),
    }
    del device_heads
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, mechanism_rows, summary


def _comparison_sets(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> Dict[str, object]:
    return {
        "controls": _build_comparisons(
            raw_conditions=predictions["gap"],
            control_conditions=predictions["top_only"],
            candidate_conditions=predictions["wildcat"],
        ),
        "same_weight_ablation": _build_comparisons(
            raw_conditions=predictions["no_bottom"],
            control_conditions=predictions["top_only"],
            candidate_conditions=predictions["wildcat"],
        ),
        "raw_context": _build_comparisons(
            raw_conditions=predictions["raw"],
            control_conditions=predictions["gap"],
            candidate_conditions=predictions["wildcat"],
        ),
    }


def _write_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    variants = ("raw", "gap", "top_only", "wildcat", "no_bottom")
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for name in variants:
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(NUM_CLASSES))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            for position, raw in enumerate(predictions["raw"][condition]):
                sample_index = int(raw["sample_index"])
                source = rows[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source.source_stem,
                    "image_path": str(source.image_path),
                    "fold": source.fold,
                    "target": source.target,
                }
                for name in variants:
                    values = predictions[name][condition][position]
                    output[f"{name}_prediction"] = int(values["prediction"])
                    for class_index in range(NUM_CLASSES):
                        output[f"{name}_prob_{class_index}"] = float(
                            values[f"prob_{class_index}"]
                        )
                writer.writerow(output)


def _write_mechanism(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    if not values:
        raise ValueError("No WILDCAT mechanism rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def _write_training(path: Path, training: Mapping[str, object]) -> None:
    rows = []
    for name in ("gap", "top_only", "wildcat"):
        rows.extend(training["history"][name])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _replay_predictions(path: Path, expected: Mapping[str, object]) -> Dict[str, object]:
    variants = ("raw", "gap", "top_only", "wildcat", "no_bottom")
    grouped = {
        name: {condition: [] for condition in CONDITIONS} for name in variants
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            condition = str(source["condition"])
            if condition not in CONDITIONS:
                raise ValueError(f"Unexpected replay condition: {condition}")
            for name in variants:
                row: Dict[str, object] = {
                    "sample_index": int(source["sample_index"]),
                    "target": int(source["target"]),
                    "prediction": int(source[f"{name}_prediction"]),
                }
                for class_index in range(NUM_CLASSES):
                    row[f"prob_{class_index}"] = float(
                        source[f"{name}_prob_{class_index}"]
                    )
                grouped[name][condition].append(row)
    replay = _comparison_sets(grouped)
    return {
        "rows": sum(len(grouped["raw"][condition]) for condition in CONDITIONS),
        "condition_rows": {
            condition: len(grouped["raw"][condition]) for condition in CONDITIONS
        },
        "comparisons_exact": replay == expected,
        "sha256": _sha256(path),
    }


def _replay_mechanism(
    path: Path, expected: Mapping[str, object]
) -> Dict[str, object]:
    rows: list[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            row: Dict[str, object] = {
                "condition": source["condition"],
                "sample_index": int(source["sample_index"]),
                "target": int(source["target"]),
                "wildcat_prediction": int(source["wildcat_prediction"]),
            }
            for key, value in source.items():
                if key not in row:
                    row[key] = float(value)
            rows.append(row)
    replay = _summarize_mechanism_rows(rows)
    max_error = 0.0
    for condition in CONDITIONS:
        for key in (
            "tp_top_bbox_fraction",
            "tp_bottom_bbox_fraction",
            "tp_bbox_chance",
            "tp_top_alignment_above_chance",
            "mean_class1_map_entropy",
            "mean_class_map_effective_rank",
            "bottom_margin_auroc",
        ):
            left = replay[condition][key]
            right = expected[condition][key]
            if left is not None and right is not None:
                max_error = max(max_error, abs(float(left) - float(right)))
        for name in ("gap", "top_only", "wildcat", "no_bottom"):
            left = replay[condition]["direction_auroc"][name]
            right = expected[condition]["direction_auroc"][name]
            if left is not None and right is not None:
                max_error = max(max_error, abs(float(left) - float(right)))
        for key in ("mean_top_by_class", "mean_bottom_by_class"):
            for left, right in zip(replay[condition][key], expected[condition][key]):
                if left is not None and right is not None:
                    max_error = max(max_error, abs(float(left) - float(right)))
    return {
        "rows": len(rows),
        "maximum_absolute_error": max_error,
        "exact_within_1e12": max_error <= 1e-12,
        "sha256": _sha256(path),
    }


def _benchmark_resources(
    *,
    keeper: nn.Module,
    head: WildcatSpatialHead,
    resource_batch: tuple,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, object], Tensor]:
    images_cpu, _targets_cpu, metadata_cpu = resource_batch
    images_cpu = images_cpu[: int(args.batch_size)]
    metadata_cpu = {
        key: value[: int(images_cpu.size(0))] if torch.is_tensor(value) else value
        for key, value in metadata_cpu.items()
    }
    keeper = keeper.to(device).eval()
    candidate = copy.deepcopy(head).to(device).eval()
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    images = images_cpu.to(device=device, non_blocking=True)

    def raw_iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=True
        ):
            logits, _ = _forward_classification_with_metadata(
                keeper, images, metadata_cpu, device=device
            )
        return logits.float()

    captured_dense: Optional[Tensor] = None

    def candidate_iteration() -> tuple[Tensor, Tensor]:
        with DenseBlock2Capture(keeper) as capture:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=True
            ):
                logits, _ = _forward_classification_with_metadata(
                    keeper, images, metadata_cpu, device=device
                )
            dense = capture.take().float()
            branch = candidate(dense)
        return logits.float(), branch

    for _ in range(3):
        raw_iteration()
        _, branch = candidate_iteration()
        if not torch.isfinite(branch).all():
            raise ValueError("Candidate benchmark produced non-finite logits.")
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    raw_seconds = []
    raw_logits = None
    for _ in range(int(args.benchmark_repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        raw_logits = raw_iteration()
        torch.cuda.synchronize(device)
        raw_seconds.append(float(time.perf_counter() - started))
    raw_peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))

    torch.cuda.reset_peak_memory_stats(device)
    candidate_seconds = []
    captured_logits = None
    branch_logits = None
    for _ in range(int(args.benchmark_repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        captured_logits, branch_logits = candidate_iteration()
        torch.cuda.synchronize(device)
        candidate_seconds.append(float(time.perf_counter() - started))
    candidate_peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    with DenseBlock2Capture(keeper) as capture:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=True
        ):
            identity_logits, _ = _forward_classification_with_metadata(
                keeper, images, metadata_cpu, device=device
            )
        captured_dense = capture.take().detach().float().cpu()
    if raw_logits is None or captured_logits is None or branch_logits is None:
        raise RuntimeError("Resource benchmark did not execute.")
    raw_median = float(statistics.median(raw_seconds))
    candidate_median = float(statistics.median(candidate_seconds))
    summary = {
        "batch_size": int(images.size(0)),
        "repeats": int(args.benchmark_repeats),
        "raw_seconds": raw_seconds,
        "candidate_seconds": candidate_seconds,
        "raw_median_seconds": raw_median,
        "candidate_median_seconds": candidate_median,
        "runtime_ratio": candidate_median / max(raw_median, 1e-12),
        "raw_peak_vram_gib": raw_peak,
        "candidate_peak_vram_gib": candidate_peak,
        "peak_vram_ratio": candidate_peak / max(raw_peak, 1e-12),
        "extra_peak_vram_gib": candidate_peak - raw_peak,
        "capture_raw_logit_max_abs_error": float(
            (raw_logits - identity_logits.float()).abs().amax().item()
        ),
        "capture_raw_prediction_mismatches": int(
            (raw_logits.argmax(dim=1) != identity_logits.argmax(dim=1)).sum().item()
        ),
        "branch_finite": bool(torch.isfinite(branch_logits).all()),
        "branch_shape": [int(value) for value in branch_logits.shape],
    }
    gc.collect()
    torch.cuda.empty_cache()
    return summary, captured_dense


class _IsolatedHeadExport(nn.Module):
    def __init__(self, head: WildcatSpatialHead) -> None:
        super().__init__()
        self.head = copy.deepcopy(head).cpu().eval()

    def forward(self, dense_features: Tensor) -> Tensor:
        return self.head(dense_features)


class _FullWildcatExport(nn.Module):
    def __init__(self, keeper: nn.Module, head: WildcatSpatialHead) -> None:
        super().__init__()
        self.keeper = copy.deepcopy(keeper).cpu().eval()
        self.head = copy.deepcopy(head).cpu().eval()
        self._dense: Optional[Tensor] = None
        self._handle = self.keeper.blocks[BLOCK_INDEX].register_forward_hook(
            self._capture
        )

    def _capture(self, _module, _inputs, output) -> None:
        tokens = output[0] if isinstance(output, tuple) else output
        patches = tokens[:, PREFIX_TOKENS:]
        self._dense = patches.transpose(1, 2).reshape(
            int(patches.size(0)), EMBED_DIM, GRID_HEIGHT, GRID_WIDTH
        )

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        self._dense = None
        features = self.keeper.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=bbox,
        )
        features["bbox"] = bbox
        raw = classification_logits_from_features(self.keeper, features)
        if self._dense is None:
            raise RuntimeError("Full WILDCAT export did not capture block-2 features.")
        branch = self.head(self._dense.float())
        return torch.cat((raw.float(), branch.float()), dim=1)


def _torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    normalized = np.dtype(dtype)
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
    }
    if normalized not in mapping:
        raise TypeError(f"Unsupported TensorRT IO dtype: {normalized}")
    return mapping[normalized]


def _tensorrt_audit(
    *, path: Path, wrapper: nn.Module, inputs: tuple[Tensor, ...]
) -> Dict[str, object]:
    try:
        import tensorrt as trt
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "succeeded": False,
            "error": repr(exc),
            "parse": False,
            "build": False,
            "parity": False,
        }
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        parse_ok = bool(parser.parse(path.read_bytes()))
        parser_errors = [
            str(parser.get_error(index)) for index in range(parser.num_errors)
        ]
        if not parse_ok:
            return {
                "available": True,
                "succeeded": False,
                "parse": False,
                "build": False,
                "parity": False,
                "errors": parser_errors,
            }
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        precision = "fp32"
        bf16_flag = getattr(trt.BuilderFlag, "BF16", None)
        bf16_support = getattr(builder, "platform_has_fast_bf16", False)
        if callable(bf16_support):
            bf16_support = bf16_support()
        if bf16_flag is not None and bool(bf16_support):
            config.set_flag(bf16_flag)
            precision = "bf16"
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            return {
                "available": True,
                "succeeded": False,
                "parse": True,
                "build": False,
                "parity": False,
                "errors": parser_errors,
                "precision": precision,
            }
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("TensorRT deserialization returned no engine.")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT engine returned no execution context.")
        if not hasattr(engine, "num_io_tensors") or not hasattr(
            context, "execute_async_v3"
        ):
            raise RuntimeError("TensorRT runtime lacks the locked named-IO API.")
        input_by_name = {
            str(network.get_input(index).name): inputs[index]
            for index in range(network.num_inputs)
        }
        device_values: Dict[str, Tensor] = {}
        output_names = []
        for index in range(int(engine.num_io_tensors)):
            name = str(engine.get_tensor_name(index))
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                source = input_by_name[name].detach().contiguous()
                dtype = _torch_dtype_from_numpy(np.dtype(trt.nptype(engine.get_tensor_dtype(name))))
                value = source.to(device="cuda", dtype=dtype).contiguous()
                context.set_input_shape(name, tuple(int(v) for v in value.shape))
                device_values[name] = value
            else:
                output_names.append(name)
        for name in output_names:
            shape = tuple(int(value) for value in context.get_tensor_shape(name))
            dtype = _torch_dtype_from_numpy(np.dtype(trt.nptype(engine.get_tensor_dtype(name))))
            device_values[name] = torch.empty(shape, device="cuda", dtype=dtype)
        for name, value in device_values.items():
            context.set_tensor_address(name, int(value.data_ptr()))
        stream = torch.cuda.current_stream()
        executed = bool(context.execute_async_v3(stream.cuda_stream))
        stream.synchronize()
        if not executed or len(output_names) != 1:
            raise RuntimeError("TensorRT execution failed or output count differs.")
        observed = device_values[output_names[0]].detach().float().cpu()
        with torch.inference_mode():
            expected = wrapper(*inputs).detach().float().cpu()
        error = float((expected - observed).abs().amax().item())
        return {
            "available": True,
            "succeeded": True,
            "parse": True,
            "build": True,
            "parity": bool(
                error <= MAX_TRT_ERROR
                and torch.equal(expected.argmax(dim=1), observed.argmax(dim=1))
            ),
            "precision": precision,
            "engine_bytes_ephemeral": len(bytes(serialized)),
            "engine_retained": False,
            "maximum_absolute_error": error,
            "argmax_match": bool(
                torch.equal(expected.argmax(dim=1), observed.argmax(dim=1))
            ),
            "finite": bool(torch.isfinite(observed).all()),
            "errors": parser_errors,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": True,
            "succeeded": False,
            "parse": False,
            "build": False,
            "parity": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _prepare_export_inputs(resource_batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
    images_cpu, _targets_cpu, metadata_cpu = resource_batch
    images = images_cpu[:1].detach().float().cpu()
    bbox_value = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox_value):
        raise ValueError("Export resource batch lacks normal-inference bbox metadata.")
    bbox = bbox_value[:1].detach().float().cpu()
    mask_value = metadata_cpu.get("image_mask")
    if not torch.is_tensor(mask_value):
        mask_value = metadata_cpu.get("image_valid_mask")
    if torch.is_tensor(mask_value):
        image_mask = mask_value[:1].detach().bool().cpu()
    else:
        image_mask = torch.ones(
            1, images.size(-2), images.size(-1), dtype=torch.bool
        )
    return images, bbox, image_mask


def _export_diagnostics(
    *,
    keeper: nn.Module,
    head: WildcatSpatialHead,
    dense_example: Tensor,
    resource_batch: tuple,
    output_dir: Path,
) -> Dict[str, object]:
    images, bbox, image_mask = _prepare_export_inputs(resource_batch)
    isolated_path = output_dir / "wildcat_head.onnx"
    full_path = output_dir / "wildcat_full_wrapper.onnx"
    isolated_wrapper = _IsolatedHeadExport(head)
    full_wrapper = _FullWildcatExport(keeper, head)
    try:
        isolated = _onnx_compare(
            wrapper=isolated_wrapper,
            inputs=(dense_example[:1].float().cpu(),),
            input_names=("dense_features",),
            path=isolated_path,
        )
    except Exception as exc:
        isolated = _failed_export(isolated_path, exc)
    try:
        full = _onnx_compare(
            wrapper=full_wrapper,
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as exc:
        full = _failed_export(full_path, exc)
    tensorrt = (
        _tensorrt_audit(
            path=full_path,
            wrapper=full_wrapper,
            inputs=(images, bbox, image_mask),
        )
        if bool(full.get("succeeded"))
        else {
            "available": None,
            "succeeded": False,
            "parse": False,
            "build": False,
            "parity": False,
            "error": "Full ONNX export failed before TensorRT.",
        }
    )
    return {"isolated_onnx": isolated, "full_onnx": full, "tensorrt": tensorrt}


def _xai_selection(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> tuple[list[int], Dict[int, list[str]], list[int]]:
    top_rows = predictions["top_only"]["clean"]
    candidate_rows = predictions["wildcat"]["clean"]
    ablation_rows = predictions["no_bottom"]["clean"]
    required: Dict[int, list[str]] = {}
    secondary: Dict[int, list[str]] = {}
    ranking = []

    def add(container: Dict[int, list[str]], index: int, label: str) -> None:
        container.setdefault(int(index), [])
        if label not in container[int(index)]:
            container[int(index)].append(label)

    for top, candidate, ablation in zip(top_rows, candidate_rows, ablation_rows):
        index = int(candidate["sample_index"])
        target = int(candidate["target"])
        candidate_prediction = int(candidate["prediction"])
        for prefix, comparator in (("vs_top", top), ("vs_no_bottom", ablation)):
            comparator_prediction = int(comparator["prediction"])
            if target == FOCUS_CLASS:
                if comparator_prediction == FOCUS_CLASS and candidate_prediction != FOCUS_CLASS:
                    add(required, index, f"{prefix}_class1_tp_break")
                if comparator_prediction != FOCUS_CLASS and candidate_prediction == FOCUS_CLASS:
                    add(required, index, f"{prefix}_class1_fn_rescue")
            if target in RESTRICTED_NEGATIVE_CLASSES:
                if comparator_prediction == FOCUS_CLASS and candidate_prediction != FOCUS_CLASS:
                    add(required, index, f"{prefix}_restricted_fp_remove")
                if comparator_prediction != FOCUS_CLASS and candidate_prediction == FOCUS_CLASS:
                    add(required, index, f"{prefix}_restricted_fp_create")
            if comparator_prediction != target and candidate_prediction == target:
                add(secondary, index, f"{prefix}_correction")
            if comparator_prediction == target and candidate_prediction != target:
                add(secondary, index, f"{prefix}_harm")
        delta = abs(
            float(candidate[f"prob_{FOCUS_CLASS}"])
            - float(top[f"prob_{FOCUS_CLASS}"])
        )
        ranking.append((-delta, index))
    order = {
        int(row["sample_index"]): position for position, row in enumerate(candidate_rows)
    }
    required_indices = sorted(required, key=order.__getitem__)
    selected = list(required_indices)
    categories = {index: list(required[index]) for index in required_indices}
    for index in required_indices:
        for label in secondary.get(index, []):
            add(categories, index, label)
    for index in sorted(secondary, key=order.__getitem__):
        if len(selected) >= 16:
            break
        if index not in categories:
            categories[index] = list(secondary[index])
            selected.append(index)
    for _, index in sorted(ranking):
        if len(selected) >= 16:
            break
        if index not in categories:
            categories[index] = ["deterministic_boundary"]
            selected.append(index)
    selected.sort(key=order.__getitem__)
    return selected, categories, required_indices


def _normalize_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def _signed_overlay(
    rgb: np.ndarray, signed_map: np.ndarray, *, alpha: float = 0.58
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    values = np.asarray(signed_map, dtype=np.float32)
    scale = float(np.max(np.abs(values)))
    normalized = np.zeros_like(values) if scale <= 1e-12 else values / scale
    resized = Image.fromarray(normalized.astype(np.float32)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    normalized = np.asarray(resized, dtype=np.float32).clip(-1.0, 1.0)
    positive = normalized >= 0.0
    tint = np.empty((*normalized.shape, 3), dtype=np.float32)
    tint[positive] = np.asarray((255.0, 55.0, 40.0), dtype=np.float32)
    tint[~positive] = np.asarray((35.0, 95.0, 255.0), dtype=np.float32)
    source = np.asarray(image, dtype=np.float32)
    strength = (float(alpha) * np.abs(normalized))[..., None]
    output = source * (1.0 - strength) + tint * strength
    return Image.fromarray(output.clip(0.0, 255.0).round().astype(np.uint8))


def _draw_bbox(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    x, y, width, height = [float(value) for value in bbox]
    left = int(round((x - width / 2.0) * output.width))
    right = int(round((x + width / 2.0) * output.width))
    top = int(round((y - height / 2.0) * output.height))
    bottom = int(round((y + height / 2.0) * output.height))
    draw.rectangle((left, top, right, bottom), outline=(255, 40, 40), width=3)
    return output


def _xai_audit(
    *,
    keeper: nn.Module,
    head: WildcatSpatialHead,
    dataset: MangoYOLOCropDataset,
    transform,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    rows: Sequence[CleanTrainRow],
    checkpoint: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    selected, categories, required_indices = _xai_selection(predictions)
    loader, loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=selected,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        context="wildcat_xai_selected",
        seed=int(args.seed) + 401,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    keeper = keeper.to(device).eval()
    candidate = copy.deepcopy(head).to(device).eval()
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    maps: Dict[int, Dict[str, object]] = {}
    with DenseBlock2Capture(keeper) as capture:
        for images_cpu, _targets_cpu, metadata_cpu in loader:
            images = images_cpu.to(device=device, dtype=torch.float32).requires_grad_(True)
            keeper.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
            _raw_logits, _ = _forward_classification_with_metadata(
                keeper, images, metadata_cpu, device=device
            )
            dense = capture.take().float()
            components = candidate.components(dense)
            saliency = torch.autograd.grad(
                components["logits"][:, FOCUS_CLASS].sum(), images
            )[0].detach().abs().mean(dim=1)
            sample_tensor = metadata_cpu.get("sample_index")
            bbox_cpu = metadata_cpu.get("crop_bbox")
            if not torch.is_tensor(bbox_cpu):
                bbox_cpu = metadata_cpu.get("bbox")
            if not torch.is_tensor(sample_tensor) or not torch.is_tensor(bbox_cpu):
                raise ValueError("XAI metadata lacks sample_index or bbox.")
            class1_maps = components["class_maps"][:, FOCUS_CLASS].detach()
            top_indices = components["top_indices"][:, FOCUS_CLASS].detach()
            bottom_indices = components["bottom_indices"][:, FOCUS_CLASS].detach()
            top_masks = torch.zeros_like(class1_maps.flatten(1))
            bottom_masks = torch.zeros_like(top_masks)
            top_masks.scatter_(1, top_indices, 1.0)
            bottom_masks.scatter_(1, bottom_indices, 1.0)
            top_masks = top_masks.reshape_as(class1_maps)
            bottom_masks = bottom_masks.reshape_as(class1_maps)
            bottom_contribution = (
                class1_maps * bottom_masks * (ALPHA / (2.0 * SELECTED_CELLS))
            )
            for local, sample_index in enumerate(sample_tensor.tolist()):
                index = int(sample_index)
                maps[index] = {
                    "rgb": _rgb_from_tensor(images_cpu[local], mean=mean, std=std),
                    "bbox": bbox_cpu[local].detach().float().tolist(),
                    "class1_map": class1_maps[local].float().cpu().numpy(),
                    "top_mask": top_masks[local].float().cpu().numpy(),
                    "bottom_mask": bottom_masks[local].float().cpu().numpy(),
                    "bottom_contribution": bottom_contribution[local]
                    .float()
                    .cpu()
                    .numpy(),
                    "saliency": saliency[local].float().cpu().numpy(),
                }
    expected = selected
    if list(maps) != expected:
        raise ValueError("XAI selected sample order differs.")
    npz_values = {}
    for sample_index in selected:
        for key in (
            "class1_map",
            "top_mask",
            "bottom_mask",
            "bottom_contribution",
            "saliency",
        ):
            npz_values[f"sample_{sample_index}_{key}"] = np.asarray(
                maps[sample_index][key], dtype=np.float32
            )
    npz_path = output_dir / "xai_maps.npz"
    np.savez_compressed(npz_path, **npz_values)
    with np.load(npz_path, allow_pickle=False) as replayed:
        npz_keys_exact = set(replayed.files) == set(npz_values)
        npz_replay_max_abs_error = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(replayed[key], dtype=np.float32)
                            - np.asarray(value, dtype=np.float32)
                        )
                    )
                )
                for key, value in npz_values.items()
            ),
            default=0.0,
        )
        npz_replay_finite = all(
            np.isfinite(np.asarray(replayed[key])).all() for key in replayed.files
        )

    candidate_by_index = {
        int(row["sample_index"]): row for row in predictions["wildcat"]["clean"]
    }
    top_by_index = {
        int(row["sample_index"]): row for row in predictions["top_only"]["clean"]
    }
    no_bottom_by_index = {
        int(row["sample_index"]): row for row in predictions["no_bottom"]["clean"]
    }
    columns = (
        "input+bbox",
        "class1 map",
        "top51",
        "bottom51",
        "signed bottom term",
        "input saliency",
    )
    tile = 144
    header = 28
    label_height = 64
    page_size = 8
    pages = []
    for page_number, start in enumerate(range(0, len(selected), page_size), start=1):
        page_indices = selected[start : start + page_size]
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
            item = maps[sample_index]
            rgb = np.asarray(item["rgb"], dtype=np.uint8)
            base = _draw_bbox(Image.fromarray(rgb), item["bbox"])
            views = [
                base,
                _heat_overlay(rgb, _normalize_map(item["class1_map"])),
                _heat_overlay(rgb, np.asarray(item["top_mask"], dtype=np.float32)),
                _heat_overlay(rgb, np.asarray(item["bottom_mask"], dtype=np.float32)),
                _signed_overlay(rgb, np.asarray(item["bottom_contribution"])),
                _heat_overlay(rgb, _normalize_map(item["saliency"])),
            ]
            for column, view in enumerate(views):
                canvas.paste(
                    view.resize((tile, tile), Image.Resampling.BILINEAR),
                    (column * tile, y),
                )
            source = rows[sample_index]
            line_one = (
                f"idx={sample_index} y={source.target} top/full/nobottom="
                f"{top_by_index[sample_index]['prediction']}/"
                f"{candidate_by_index[sample_index]['prediction']}/"
                f"{no_bottom_by_index[sample_index]['prediction']}"
            )
            line_two = ",".join(categories[sample_index])
            bottom = np.asarray(item["bottom_contribution"], dtype=np.float32)
            line_three = (
                f"signed bottom range=[{float(bottom.min()):+.3e},"
                f"{float(bottom.max()):+.3e}] red=positive blue=negative"
            )
            draw.text((4, y + tile + 3), line_one, fill="black")
            draw.text((4, y + tile + 22), line_two[:150], fill="black")
            draw.text((4, y + tile + 41), line_three, fill="black")
        page_path = output_dir / f"xai_contact_sheet_{page_number:03d}.png"
        canvas.save(page_path)
        pages.append(str(page_path.resolve()))
    finite = all(
        np.isfinite(np.asarray(item[key])).all()
        for item in maps.values()
        for key in (
            "class1_map",
            "top_mask",
            "bottom_mask",
            "bottom_contribution",
            "saliency",
        )
    )
    nonzero = all(
        np.count_nonzero(np.asarray(item[key])) > 0
        for item in maps.values()
        for key in (
            "class1_map",
            "top_mask",
            "bottom_mask",
            "bottom_contribution",
            "saliency",
        )
    )
    manifest = {
        "selected_indices": selected,
        "required_event_indices": required_indices,
        "categories": {str(key): value for key, value in categories.items()},
        "required_event_rows": len(required_indices),
        "required_event_coverage_exact": all(
            index in selected for index in required_indices
        ),
        "selected_rows": len(selected),
        "pages": pages,
        "page_count": len(pages),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256(npz_path),
        "npz_key_count": len(npz_values),
        "npz_keys_exact": bool(npz_keys_exact),
        "npz_replay_finite": bool(npz_replay_finite),
        "npz_replay_max_abs_error": float(npz_replay_max_abs_error),
        "loader": loader_summary,
        "hook_calls": capture.call_count,
        "selection_order_exact": list(maps) == selected,
        "all_finite": bool(finite),
        "defined_maps_nonzero": bool(nonzero),
        "manual_review_required": True,
    }
    manifest_path = output_dir / "xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def _class_metric(comparison: Mapping[str, object], metric: str) -> float:
    candidate = comparison["candidate"]
    if metric == "macro_f1":
        return float(candidate["macro_f1"])
    lookup = {
        "precision": "per_class_precision",
        "recall": "per_class_recall",
        "f1": "per_class_f1",
    }
    return float(candidate[lookup[metric]][FOCUS_CLASS])


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    comparisons: Mapping[str, object],
    mechanism: Mapping[str, Mapping[str, object]],
    resources: Mapping[str, object],
    exports: Mapping[str, object],
    xai: Mapping[str, object],
    visual_review_passed: Optional[bool] = None,
) -> Dict[str, object]:
    controls = comparisons["controls"]
    ablations = comparisons["same_weight_ablation"]
    clean_gap = controls["clean"]["raw_candidate"]
    clean_top = controls["clean"]["control_candidate"]
    clean_ablation = ablations["clean"]["raw_candidate"]
    clean_mechanism = mechanism["clean"]
    direction = clean_mechanism["direction_auroc"]
    wildcat_direction = direction["wildcat"]
    gap_direction = direction["gap"]
    top_direction = direction["top_only"]
    bottom_margin_auroc = clean_mechanism["bottom_margin_auroc"]
    top_alignment = clean_mechanism["tp_top_alignment_above_chance"]
    information_checks = {
        "clean_direction_auroc_gte_0p70": wildcat_direction is not None
        and float(wildcat_direction) >= 0.70,
        "clean_direction_auroc_gain_vs_gap_gte_0p005": wildcat_direction
        is not None
        and gap_direction is not None
        and float(wildcat_direction) - float(gap_direction) >= 0.005,
        "clean_direction_auroc_gain_vs_top_gte_0p005": wildcat_direction
        is not None
        and top_direction is not None
        and float(wildcat_direction) - float(top_direction) >= 0.005,
        "bottom_margin_auroc_gte_0p65": bottom_margin_auroc is not None
        and float(bottom_margin_auroc) >= 0.65,
        "bottom_restricted_fp_reduction_gte_2": int(
            clean_ablation["transitions"]["restricted_focus_fp_reduction"]
        )
        >= 2,
        "bottom_class1_tp_break_lte_1": int(
            clean_ablation["transitions"]["focus_tp_break"]
        )
        <= 1,
        "bottom_corrections_gt_harms": int(
            clean_ablation["transitions"]["candidate_correction"]
        )
        > int(clean_ablation["transitions"]["candidate_harm"]),
        "class1_top_bbox_alignment_above_chance_gte_0p10": top_alignment
        is not None
        and float(top_alignment) >= 0.10,
    }
    decision_checks: Dict[str, bool] = {
        "candidate_class1_precision_gte_0p75": _class_metric(
            clean_top, "precision"
        )
        >= 0.75,
        "candidate_class1_recall_gte_0p70": _class_metric(clean_top, "recall")
        >= 0.70,
        "candidate_class1_f1_gte_0p70": _class_metric(clean_top, "f1") >= 0.70,
    }
    for label, comparison in (("gap", clean_gap), ("top", clean_top)):
        delta = comparison["delta"]
        transitions = comparison["transitions"]
        decision_checks.update(
            {
                f"macro_f1_delta_vs_{label}_gte_minus_0p002": float(
                    delta["macro_f1"]
                )
                >= -0.002,
                f"class1_precision_gain_vs_{label}_gte_0p005": float(
                    delta["class1_precision"]
                )
                >= 0.005,
                f"class1_f1_gain_vs_{label}_gte_0p003": float(
                    delta["class1_f1"]
                )
                >= 0.003,
                f"class1_recall_delta_vs_{label}_gte_minus_0p020": float(
                    delta["class1_recall"]
                )
                >= -0.020,
                f"class1_tp_break_vs_{label}_lte_2": int(
                    transitions["focus_tp_break"]
                )
                <= 2,
                f"restricted_fp_reduction_vs_{label}_gte_2": int(
                    transitions["restricted_focus_fp_reduction"]
                )
                >= 2,
                f"corrections_gt_harms_vs_{label}": int(
                    transitions["candidate_correction"]
                )
                > int(transitions["candidate_harm"]),
                f"maximum_nonfocus_f1_drop_vs_{label}_lte_0p010": float(
                    comparison["maximum_nonfocus_f1_drop"]
                )
                <= 0.010,
            }
        )

    shifted = [name for name in CONDITIONS if name != "clean"]
    illumination_checks: Dict[str, bool] = {}
    for label, key in (("gap", "raw_candidate"), ("top", "control_candidate")):
        rows = [controls[name][key] for name in shifted]
        illumination_checks.update(
            {
                f"precision_nonnegative_vs_{label}_at_least_2_of_3": sum(
                    float(row["delta"]["class1_precision"]) >= 0.0 for row in rows
                )
                >= 2,
                f"f1_nonnegative_vs_{label}_at_least_2_of_3": sum(
                    float(row["delta"]["class1_f1"]) >= 0.0 for row in rows
                )
                >= 2,
                f"direction_auroc_nonnegative_vs_{label}_at_least_2_of_3": sum(
                    float(mechanism[name]["direction_auroc"]["wildcat"])
                    >= float(mechanism[name]["direction_auroc"][
                        "gap" if label == "gap" else "top_only"
                    ])
                    for name in shifted
                )
                >= 2,
                f"worst_recall_delta_vs_{label}_gte_minus_0p030": min(
                    float(row["delta"]["class1_recall"]) for row in rows
                )
                >= -0.030,
                f"aggregate_fp_removals_gt_creations_vs_{label}": sum(
                    int(row["transitions"]["focus_fp_remove_correct"])
                    for row in rows
                )
                > sum(
                    int(row["transitions"]["focus_fp_create"]) for row in rows
                ),
            }
        )
    illumination_checks["same_weight_bottom_tp_safe_at_least_2_of_3"] = sum(
        int(ablations[name]["raw_candidate"]["transitions"]["focus_tp_break"])
        <= int(
            ablations[name]["raw_candidate"]["transitions"]["focus_fn_rescue"]
        )
        for name in shifted
    ) >= 2
    illumination_checks["all_condition_mechanism_finite"] = all(
        bool(mechanism[name]["all_finite"]) for name in CONDITIONS
    )

    isolated = exports["isolated_onnx"]
    full = exports["full_onnx"]
    trt = exports["tensorrt"]
    deployment_checks = {
        "runtime_ratio_lte_1p10": float(resources["runtime_ratio"])
        <= MAX_RUNTIME_RATIO,
        "peak_memory_within_ratio_or_delta": bool(
            float(resources["peak_vram_ratio"]) <= 1.10
            or float(resources["extra_peak_vram_gib"]) <= MAX_EXTRA_VRAM_GIB
        ),
        "capture_preserves_raw_logits": float(
            resources["capture_raw_logit_max_abs_error"]
        )
        == 0.0
        and int(resources["capture_raw_prediction_mismatches"]) == 0,
        "branch_runtime_output_finite": bool(resources["branch_finite"]),
        "isolated_onnx_pass": bool(isolated.get("succeeded"))
        and bool(isolated.get("finite"))
        and bool(isolated.get("argmax_match"))
        and float(isolated.get("maximum_absolute_error", math.inf))
        <= MAX_ONNX_ERROR,
        "full_onnx_pass": bool(full.get("succeeded"))
        and bool(full.get("finite"))
        and bool(full.get("argmax_match"))
        and float(full.get("maximum_absolute_error", math.inf)) <= MAX_ONNX_ERROR,
        "tensorrt_unavailable_or_pass": bool(trt.get("available") is False)
        or bool(
            trt.get("available") is True
            and trt.get("succeeded")
            and trt.get("parse")
            and trt.get("build")
            and trt.get("parity")
            and trt.get("finite")
            and float(trt.get("maximum_absolute_error", math.inf)) <= MAX_TRT_ERROR
        ),
        "xai_selection_order_exact": bool(xai["selection_order_exact"]),
        "xai_required_event_coverage_exact": bool(
            xai["required_event_coverage_exact"]
        ),
        "xai_all_finite": bool(xai["all_finite"]),
        "xai_defined_maps_nonzero": bool(xai["defined_maps_nonzero"]),
        "xai_npz_replay_exact": bool(xai["npz_keys_exact"])
        and bool(xai["npz_replay_finite"])
        and int(xai["npz_key_count"]) == 5 * int(xai["selected_rows"])
        and float(xai["npz_replay_max_abs_error"]) == 0.0,
        "xai_at_least_16_rows": int(xai["selected_rows"]) >= 16,
        "xai_pages_cover_all_rows": int(xai["page_count"])
        == math.ceil(int(xai["selected_rows"]) / 8)
        and all(Path(path).is_file() for path in xai["pages"]),
    }
    automatic_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **information_checks,
        **decision_checks,
        **illumination_checks,
        **deployment_checks,
    }
    automatic_failed = [key for key, passed in automatic_checks.items() if not passed]
    manual_check = visual_review_passed is True
    failed = list(automatic_failed)
    if visual_review_passed is not None and not manual_check:
        failed.append("manual_visual_review_passed")
    all_passed = not automatic_failed and manual_check
    return {
        "automatic_checks": automatic_checks,
        "information_checks": information_checks,
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "deployment_checks": deployment_checks,
        "automatic_failed_checks": automatic_failed,
        "automatic_gates_passed": not automatic_failed,
        "manual_visual_review_passed": visual_review_passed,
        "failed_checks": failed,
        "all_gates_passed": all_passed,
        "stage_b_authorized": all_passed,
        "validation_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "current_best_update_authorized": False,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    def format_optional(value: object) -> str:
        if value is None:
            return "n/a"
        numeric = float(value)
        return "n/a" if not math.isfinite(numeric) else f"{numeric:.6f}"

    controls = summary["comparisons"]["controls"]["clean"]
    gap = controls["raw_candidate"]
    top = controls["control_candidate"]
    ablation = summary["comparisons"]["same_weight_ablation"]["clean"][
        "raw_candidate"
    ]
    mechanism = summary["evaluation"]["mechanism"]["clean"]
    gate = summary["gate"]
    lines = [
        "# WILDCAT Dense Block-2 Negative-Evidence A0 Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Automatic gates passed: `{gate['automatic_gates_passed']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Clean class1 P/R/F1: `{_class_metric(top, 'precision'):.6f}/{_class_metric(top, 'recall'):.6f}/{_class_metric(top, 'f1'):.6f}`",
        f"- Clean macro/class1-F1 delta vs GAP: `{gap['delta']['macro_f1']:+.6f}/{gap['delta']['class1_f1']:+.6f}`",
        f"- Clean macro/class1-F1 delta vs top-only: `{top['delta']['macro_f1']:+.6f}/{top['delta']['class1_f1']:+.6f}`",
        f"- Same-weight bottom FP reduction / TP breaks: `{ablation['transitions']['restricted_focus_fp_reduction']}/{ablation['transitions']['focus_tp_break']}`",
        "- Direction AUROC GAP/top/WILDCAT: `"
        f"{format_optional(mechanism['direction_auroc']['gap'])}/"
        f"{format_optional(mechanism['direction_auroc']['top_only'])}/"
        f"{format_optional(mechanism['direction_auroc']['wildcat'])}`",
        f"- Bottom-margin AUROC: `{format_optional(mechanism['bottom_margin_auroc'])}`",
        f"- Runtime / peak-memory ratio: `{summary['resources']['runtime_ratio']:.6f}/{summary['resources']['peak_vram_ratio']:.6f}`",
        f"- XAI selected/pages: `{summary['xai']['selected_rows']}/{summary['xai']['page_count']}`",
        "- Validation/test/raw-data modification: `false/false/false`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "forbidden_checkpoint_count": sum(
            Path(str(row["path"])).suffix.lower() in {".pt", ".pth", ".ckpt"}
            for row in artifacts
        ),
        "temporary_feature_cache_count": sum(
            "wildcat_dense_block2" in str(row["path"]).casefold()
            and Path(str(row["path"])).suffix.lower() in {".f16", ".mmap", ".memmap"}
            for row in artifacts
        ),
    }
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def _formal_structural_checks(
    *,
    provenance: Mapping[str, object],
    cohorts: Mapping[str, object],
    equation: Mapping[str, object],
    shape: Mapping[str, object],
    construction: Mapping[str, object],
    dataset_summary: Mapping[str, object],
    cache_info: Mapping[str, object],
    training: Mapping[str, object],
    evaluation: Mapping[str, object],
    replay: Mapping[str, object],
    keeper_state_unchanged: bool,
    paths: Mapping[str, Path],
) -> Dict[str, bool]:
    gradient_checks = [
        value
        for head in training["first_batch_gradients"].values()
        for value in head.values()
    ]
    movement_checks = [
        value
        for head in training["movement"].values()
        for value in head.values()
    ]
    history_rows = [
        row
        for name in ("gap", "top_only", "wildcat")
        for row in training["history"][name]
    ]
    implementation = provenance["implementation"]
    repository_root = Path(__file__).resolve().parents[2]
    implementation_unchanged = all(
        _sha256(repository_root / str(value["path"])) == str(value["sha256"])
        for value in implementation.values()
    )
    return {
        "repository_pushed_and_implementation_committed": bool(
            provenance["tracked_trkh_worktree_clean"]
        )
        and bool(provenance["implementation_committed_clean"])
        and provenance["repository_commit"] == provenance["upstream_commit"]
        and len(implementation) == 3
        and all(
            len(str(value["sha256"])) == 64 for value in implementation.values()
        ),
        "official_source_provenance_exact": bool(
            provenance["official_worktree_clean"]
        )
        and provenance["official_commit"] == LOCKED_OFFICIAL_COMMIT
        and provenance["official_tree"] == LOCKED_OFFICIAL_TREE,
        "cohort_and_order_contract_exact": int(cohorts["fit_rows"]) == 7372
        and int(cohorts["holdout_rows"]) == 1843
        and int(cohorts["source_overlap"]) == 0
        and cohorts["fit_class_counts"] == [1561, 432, 1527, 2017, 1835]
        and cohorts["holdout_class_counts"] == [380, 109, 393, 503, 458]
        and cohorts["fit_index_sha256"] == LOCKED_FIT_INDEX_SHA256
        and cohorts["holdout_index_sha256"] == LOCKED_HOLDOUT_INDEX_SHA256
        and int(cohorts["training_order_rows"]) == 147_440
        and cohorts["training_order_sha256"] == LOCKED_TRAIN_ORDER_SHA256
        and int(cohorts["training_epochs"]) == EPOCHS,
        "equation_selected_cells_exact": int(equation["selected_cells"])
        == SELECTED_CELLS,
        "equation_oracles_match_lte_1e6": max(
            float(equation["classwise_oracle_max_abs_error"]),
            float(equation["pool_oracle_max_abs_error"]),
            float(equation["top_oracle_max_abs_error"]),
            float(equation["bottom_oracle_max_abs_error"]),
            float(equation["gradient_oracle_max_abs_error"]),
            float(equation["official_source_max_abs_error"]),
        )
        <= 1e-6,
        "equation_finite_difference_lte_1e4": float(
            equation["finite_difference_max_abs_error"]
        )
        <= 1e-4,
        "equation_bf16_error_lte_0p02": float(equation["bf16_vs_fp32_max_abs_error"])
        <= 0.02,
        "equation_edge_cases_pass": bool(equation["ties_finite"])
        and equation["non_square_shape"] == [2, 5]
        and bool(equation["invalid_channel_count_rejected"])
        and bool(equation["invalid_k_rejected"])
        and bool(equation["invalid_mode_rejected"])
        and bool(equation["legacy_identity_predicate_disclosed"])
        and bool(equation["official_divide_by_two_present"]),
        "dense_shape_contract_exact": shape["block_output_shape"]
        == [1, PREFIX_TOKENS + SPATIAL_CELLS, EMBED_DIM]
        and shape["dense_feature_shape"]
        == [1, EMBED_DIM, GRID_HEIGHT, GRID_WIDTH]
        and int(shape["prefix_tokens"]) == PREFIX_TOKENS
        and shape["logit_shape"] == [1, NUM_CLASSES]
        and int(shape["hook_calls"]) == 1
        and bool(shape["finite"]),
        "keeper_parameter_count_exact": int(construction["keeper_parameters"])
        == EXPECTED_KEEPER_PARAMETERS,
        "head_parameter_count_exact": int(construction["head_parameters"])
        == EXPECTED_HEAD_PARAMETERS,
        "head_parameter_schema_exact": construction["parameter_schema"]
        == {
            "projection.weight": [PROJECTED_CHANNELS, EMBED_DIM, 1, 1],
            "projection.bias": [PROJECTED_CHANNELS],
        },
        "head_initialization_exact": bool(construction["heads_initially_bit_exact"])
        and bool(construction["isolated_initialization_rng_restored"]),
        "dataset_train_only_exact": int(dataset_summary["rows"]) == 9215
        and bool(dataset_summary["paths_exact"])
        and bool(dataset_summary["train_paths_only"]),
        "cache_contract_exact": int(cache_info["rows"]) == 7372
        and cache_info["shape"] == [7372, EMBED_DIM, GRID_HEIGHT, GRID_WIDTH]
        and int(cache_info["hook_calls"]) == math.ceil(7372 / BATCH_SIZE)
        and cache_info["sample_index_sha256"] == LOCKED_FIT_INDEX_SHA256
        and bool(cache_info["temporary_cache_removed"]),
        "training_updates_exact": all(
            int(value) == 4620 for value in training["updates"].values()
        ),
        "training_occurrences_exact": all(
            int(value) == 147_440 for value in training["occurrences"].values()
        )
        and training["training_order_sha256"] == LOCKED_TRAIN_ORDER_SHA256,
        "optimizer_parity_exact": bool(training["optimizer_configurations_equal"])
        and all(
            int(value["parameter_groups"]) == 1
            and int(value["parameter_tensors"]) == 2
            and math.isclose(
                float(value["learning_rate"]), LEARNING_RATE, abs_tol=0.0
            )
            and math.isclose(float(value["momentum"]), MOMENTUM, abs_tol=0.0)
            and math.isclose(
                float(value["weight_decay"]), WEIGHT_DECAY, abs_tol=0.0
            )
            for value in training["optimizer_configuration"].values()
        )
        and all(
            int(value) == 2
            for value in training["optimizer_state_parameter_counts"].values()
        ),
        "training_logit_scale_recorded_finite": len(history_rows) == 3 * EPOCHS
        and bool(training["all_training_values_finite"])
        and bool(training["all_final_parameters_finite"])
        and all(
            math.isfinite(float(row[key]))
            for row in history_rows
            for key in (
                "mean_loss",
                "mean_gradient_norm",
                "mean_absolute_logit",
                "rms_logit",
                "maximum_absolute_logit",
            )
        ),
        "all_head_gradients_finite_nonzero": all(
            bool(value["present"] and value["finite"] and value["nonzero"])
            for value in gradient_checks
        )
        and bool(training["all_gradients_seen_nonzero"]),
        "every_head_parameter_changed": all(
            bool(value["changed"]) for value in movement_checks
        ),
        "keeper_state_bit_exact": bool(keeper_state_unchanged),
        "evaluation_rows_and_hooks_exact": all(
            int(evaluation["mechanism"][condition]["rows"]) == 1843
            and int(evaluation["hook_calls"][condition])
            == math.ceil(1843 / BATCH_SIZE)
            for condition in CONDITIONS
        ),
        "mechanism_per_class_telemetry_complete": all(
            len(evaluation["mechanism"][condition]["mean_top_by_class"])
            == NUM_CLASSES
            and len(evaluation["mechanism"][condition]["mean_bottom_by_class"])
            == NUM_CLASSES
            and all(
                value is not None and math.isfinite(float(value))
                for value in (
                    *evaluation["mechanism"][condition]["mean_top_by_class"],
                    *evaluation["mechanism"][condition]["mean_bottom_by_class"],
                )
            )
            for condition in CONDITIONS
        ),
        "raw_cidt_near_tie_mismatches_lte_1": int(
            evaluation["clean_raw_cidt_argmax_mismatches"]
        )
        <= 1,
        "prediction_replay_exact": bool(replay["predictions"]["comparisons_exact"])
        and replay["predictions"]["condition_rows"]
        == {condition: 1843 for condition in CONDITIONS},
        "mechanism_replay_exact": bool(replay["mechanism"]["exact_within_1e12"])
        and int(replay["mechanism"]["rows"]) == 4 * 1843,
        "current_commands_unchanged_at_end": _sha256(paths["current_commands"])
        == LOCKED_CURRENT_COMMAND_SHA256
        and _sha256(paths["command_history"]) == LOCKED_COMMAND_HISTORY_SHA256,
        "raw_data_yaml_unchanged_at_end": _sha256(paths["data"])
        == LOCKED_DATA_SHA256,
        "repository_and_implementation_unchanged_at_end": _tracked_worktree_clean(
            repository_root
        )
        and implementation_unchanged
        and _git_commit(repository_root) == provenance["repository_commit"]
        and _git_value(
            repository_root,
            "rev-parse",
            "origin/classification-only-research",
        )
        == provenance["upstream_commit"],
        "official_worktree_clean_at_end": _full_worktree_clean(paths["official_root"])
        and _git_commit(paths["official_root"]) == LOCKED_OFFICIAL_COMMIT
        and _git_tree(paths["official_root"]) == LOCKED_OFFICIAL_TREE,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance, rows, cohorts, order = _load_locked_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked WILDCAT A0 requires CUDA.")
    device = torch.device("cuda")
    paths = _source_paths(args)
    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    keeper, prototypes, construction = _construct_keeper_and_heads(checkpoint)
    equation = _equation_diagnostics(paths["official_pooling"], device)
    shape = _structural_shape_replay(keeper, device)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, paths["data"]
    )
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "method": METHOD,
            "provenance": provenance,
            "cohort": {
                key: value
                for key, value in cohorts.items()
                if key not in {"fit_indices", "holdout_indices"}
            },
            "equation": equation,
            "shape": shape,
            "construction": construction,
            "dataset": dataset_summary,
            "output_directory_created": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }

    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("WILDCAT output cannot be under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)
    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    amp_dtype = _amp_dtype(device)
    keeper_state_before = _state_sha256(keeper)

    cache_path = output_dir.parent / (
        f".wildcat_dense_block2_{os.getpid()}_{int(time.time())}.f16"
    )
    cache = None
    cache_info: Dict[str, object] = {}
    resource_batch = None
    try:
        cache, fit_targets, fit_sample_indices, cache_info, resource_batch = (
            _extract_fit_cache(
                keeper=keeper,
                dataset=dataset,
                transform=transform,
                indices=cohorts["fit_indices"],
                cache_path=cache_path,
                args=args,
                device=device,
                amp_dtype=amp_dtype,
            )
        )
        trained, training = _train_heads(
            prototypes=prototypes,
            cache=cache,
            targets=fit_targets,
            sample_indices=fit_sample_indices,
            order=order,
            args=args,
            device=device,
        )
    finally:
        if cache is not None:
            cache.flush()
            del cache
        gc.collect()
        if cache_path.exists():
            cache_path.unlink()
        if cache_info:
            cache_info["temporary_cache_removed"] = not cache_path.exists()
    if resource_batch is None:
        raise RuntimeError("WILDCAT resource batch was not captured.")

    predictions, mechanism_rows, evaluation = _evaluate_conditions(
        keeper=keeper,
        heads=trained,
        dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"],
        rows=rows,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    comparisons = _comparison_sets(predictions)
    prediction_path = output_dir / "predictions_all_conditions.csv"
    mechanism_path = output_dir / "mechanism_all_conditions.csv"
    training_path = output_dir / "training_curve.csv"
    _write_predictions(prediction_path, rows=rows, predictions=predictions)
    _write_mechanism(mechanism_path, mechanism_rows)
    _write_training(training_path, training)
    replay = {
        "predictions": _replay_predictions(prediction_path, comparisons),
        "mechanism": _replay_mechanism(
            mechanism_path, evaluation["mechanism"]
        ),
    }
    resources, dense_example = _benchmark_resources(
        keeper=keeper,
        head=trained["wildcat"],
        resource_batch=resource_batch,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    exports = _export_diagnostics(
        keeper=keeper,
        head=trained["wildcat"],
        dense_example=dense_example,
        resource_batch=resource_batch,
        output_dir=output_dir,
    )
    xai = _xai_audit(
        keeper=keeper,
        head=trained["wildcat"],
        dataset=dataset,
        transform=transform,
        predictions=predictions,
        rows=rows,
        checkpoint=checkpoint,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    keeper_state_after = _state_sha256(keeper)
    structural_checks = _formal_structural_checks(
        provenance=provenance,
        cohorts=cohorts,
        equation=equation,
        shape=shape,
        construction=construction,
        dataset_summary=dataset_summary,
        cache_info=cache_info,
        training=training,
        evaluation=evaluation,
        replay=replay,
        keeper_state_unchanged=keeper_state_before == keeper_state_after,
        paths=paths,
    )
    gate = assess_stage_a(
        structural_checks=structural_checks,
        comparisons=comparisons,
        mechanism=evaluation["mechanism"],
        resources=resources,
        exports=exports,
        xai=xai,
        visual_review_passed=None,
    )
    summary: Dict[str, object] = {
        "status": "awaiting_visual_review",
        "method": METHOD,
        "provenance": provenance,
        "cohort": {
            key: value
            for key, value in cohorts.items()
            if key not in {"fit_indices", "holdout_indices"}
        },
        "equation": equation,
        "shape": shape,
        "construction": construction,
        "dataset": dataset_summary,
        "feature_cache": cache_info,
        "training": training,
        "evaluation": evaluation,
        "comparisons": comparisons,
        "replay": replay,
        "resources": resources,
        "exports": exports,
        "xai": xai,
        "structural_checks": structural_checks,
        "gate": gate,
        "visual_review": None,
        "validation_predictions_used": False,
        "test_data_used": False,
        "raw_data_modified": False,
        "image_epochs_run": 0,
        "current_best_commands_updated": False,
    }
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    _write_report(report_path, summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_checkpoint_count"]) != 0:
        raise RuntimeError("WILDCAT formal output contains a forbidden checkpoint.")
    if int(manifest["temporary_feature_cache_count"]) != 0:
        raise RuntimeError("WILDCAT formal output retained a feature cache.")
    return {
        **summary,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Formal WILDCAT summary is missing: {summary_path}")
    expected = str(args.expected_summary_sha256).strip().casefold()
    observed = _sha256(summary_path)
    if not expected or expected != observed:
        raise ValueError(
            f"Pre-review summary SHA differs: expected={expected}, observed={observed}"
        )
    if not str(args.visual_review_note).strip():
        raise ValueError("Visual review finalization requires a non-empty note.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "awaiting_visual_review":
        raise ValueError("WILDCAT summary is not awaiting visual review.")
    repository_root = Path(__file__).resolve().parents[2]
    provenance = summary["provenance"]
    if not _tracked_worktree_clean(repository_root):
        raise ValueError("Tracked TRKH worktree changed before visual finalization.")
    if _git_commit(repository_root) != provenance["repository_commit"]:
        raise ValueError("TRKH commit changed before visual finalization.")
    if (
        _git_value(
            repository_root, "rev-parse", "origin/classification-only-research"
        )
        != provenance["upstream_commit"]
    ):
        raise ValueError("Pushed TRKH commit changed before visual finalization.")
    for value in provenance["implementation"].values():
        implementation_path = repository_root / str(value["path"])
        if _sha256(implementation_path) != str(value["sha256"]):
            raise ValueError(
                f"WILDCAT implementation changed before visual finalization: "
                f"{implementation_path}"
            )
    locked_paths = _source_paths(args)
    if _sha256(locked_paths["data"]) != LOCKED_DATA_SHA256:
        raise ValueError("Raw-data YAML changed before visual finalization.")
    if (
        _sha256(locked_paths["current_commands"]) != LOCKED_CURRENT_COMMAND_SHA256
        or _sha256(locked_paths["command_history"])
        != LOCKED_COMMAND_HISTORY_SHA256
    ):
        raise ValueError("Current-best command records changed before finalization.")
    if (
        not _full_worktree_clean(locked_paths["official_root"])
        or _git_commit(locked_paths["official_root"]) != LOCKED_OFFICIAL_COMMIT
        or _git_tree(locked_paths["official_root"]) != LOCKED_OFFICIAL_TREE
    ):
        raise ValueError("Official WILDCAT source changed before finalization.")
    passed = str(args.visual_review_result) == "pass"
    summary["visual_review"] = {
        "result": str(args.visual_review_result),
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "reviewed_page_count": int(summary["xai"]["page_count"]),
        "pre_review_summary_sha256": observed,
    }
    gate = dict(summary["gate"])
    failed = list(gate["automatic_failed_checks"])
    if not passed:
        failed.append("manual_visual_review_passed")
    gate["manual_visual_review_passed"] = passed
    gate["failed_checks"] = failed
    gate["all_gates_passed"] = not failed
    gate["stage_b_authorized"] = not failed
    gate["validation_authorized"] = False
    gate["test_authorized"] = False
    gate["full_train_authorized"] = False
    gate["current_best_update_authorized"] = False
    summary["gate"] = gate
    summary["status"] = "passed" if not failed else "rejected"
    _write_report(output_dir / "report.md", summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(output_dir)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "artifact_manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if bool(args.finalize_visual_review):
        result = _finalize_visual_review(args)
    else:
        result = run_audit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method": result["method"],
                "summary_path": result.get("summary_path"),
                "summary_sha256": result.get("summary_sha256"),
                "stage_b_authorized": result.get("gate", {}).get(
                    "stage_b_authorized", False
                ),
                "validation_predictions_used": result.get(
                    "validation_predictions_used", False
                ),
                "test_data_used": result.get("test_data_used", False),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
