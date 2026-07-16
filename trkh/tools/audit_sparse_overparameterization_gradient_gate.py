from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    _forward_classification_with_metadata,
    _transform_classification_image,
    _unpack_classification_sample,
)
from trkh.inference.inference import load_model
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "sparse_overparameterization_a0_fit_gradient_gate"
NUM_CLASSES = 5
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLD = 0
SEED = 42
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_CONFUSION = [
    [1_416, 135, 5, 0, 5],
    [9, 422, 0, 0, 1],
    [4, 41, 1_455, 22, 5],
    [0, 4, 33, 1_978, 2],
    [13, 9, 2, 0, 1_811],
]
EXPECTED_METRICS = {
    "macro_f1": 0.937980,
    "class1_f1": 0.809204,
    "class1_precision": 0.690671,
    "class1_recall": 0.976852,
}
EXPECTED_CLASS1_TP = 422
EXPECTED_CLASS1_FP = 189
EXPECTED_RESTRICTED_FP = 185

LOCKED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
LOCKED_RAW_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
LOCKED_DECLARATION_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_FOLD_DATA_SHA256 = (
    "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e"
)
LOCKED_FOLD_SUMMARY_SHA256 = (
    "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
)
LOCKED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
LOCKED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_PROTOCOL_SHA256 = (
    "d2b94048da9d2a3ed261fda4f97c333c51bada3b008bd2303629aefd96b687c8"
)
LOCKED_PAPER_SHA256 = (
    "087e1c05425bf9682351a27097ce25cb2ecc94a912b715bbebc97589506bac12"
)
LOCKED_OFFICIAL_COMMIT = "4d991cedf1fafec98f858a213ccc31e52318a77f"
LOCKED_OFFICIAL_TREE = "dadbf1b288599c67d7fcced4db2cbdc3db10c6cd"
LOCKED_OFFICIAL_LOSS_SHA256 = (
    "66b252e62b1e0e1d8ed50a896f1eaaed9506a35a68a96f1b0cf0966ba744af48"
)
LOCKED_OFFICIAL_TRAINER_SHA256 = (
    "8c53028d7e4b220b0ddaac7010a1eedf34a41a35427cdb7120e3b8ee6392a8db"
)
LOCKED_OFFICIAL_LICENSE_SHA256 = (
    "05d6838d881952883e6da340405ba12c3c94372f7f30298beec0f1d902cb8886"
)
LOCKED_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
LOCKED_NBDT_CLOSURE_SHA256 = (
    "d1747782630eaa57f38239025e4925f1b1d393562bc65028e7b5dc3231a9dd3b"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the precommitted fit-only SOP A0 logit-gradient compatibility gate. "
            "No holdout, validation, test, image epoch, or trainer integration is used."
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
        "--fold-data",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--declaration",
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
            "docs/TRKH_5CLASS_SPARSE_OVERPARAMETERIZATION_A0_"
            "READINESS_PROTOCOL_20260717.md"
        ),
    )
    parser.add_argument(
        "--nbdt-closure",
        type=Path,
        default=Path("docs/TRKH_5CLASS_DOMAIN_NBDT_A0_CLOSURE_20260717.md"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\sop-icml2022"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\liu2022_sop_icml.pdf"),
    )
    parser.add_argument(
        "--current-commands",
        type=Path,
        default=Path("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"),
    )
    parser.add_argument(
        "--command-history",
        type=Path,
        default=Path("docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_sparse_overparameterization_a0_20260717"),
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--noise-batch-size", type=int, default=128)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--u-lr", type=float, default=0.1)
    parser.add_argument("--v-lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


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
    if observed != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(root).resolve()), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def _tracked_worktree_clean(root: Path) -> bool:
    return not bool(
        _git_value(root, "status", "--short", "--untracked-files=no").strip()
    )


def _prepare_output(path: Path, *, raw_data: Path) -> Path:
    resolved = Path(path).resolve()
    raw_root = Path(raw_data).resolve().parent
    if resolved == raw_root or raw_root in resolved.parents:
        raise ValueError("SOP audit output may not be inside the raw dataset tree.")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"SOP audit output must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sop_official_equation(
    logits: Tensor,
    labels_one_hot: Tensor,
    u_rows: Tensor,
    v_rows: Tensor,
    *,
    eps: float = 1e-4,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Reproduce the licensed official SOP forward equation without extras."""
    if logits.ndim != 2 or labels_one_hot.shape != logits.shape:
        raise ValueError("SOP logits and one-hot labels must share shape [B,C].")
    if u_rows.shape != (logits.shape[0], 1) or v_rows.shape != logits.shape:
        raise ValueError("SOP u/v rows have invalid shapes.")
    # The official implementation keeps one-hot labels and u/v in FP32 even
    # when network logits come from a reduced-precision autocast forward.
    labels = labels_one_hot.to(device=logits.device)
    u_effective = torch.clamp(u_rows.square() * labels, 0.0, 1.0)
    v_effective = torch.clamp(v_rows.square() * (1.0 - labels), 0.0, 1.0)
    original_probability = F.softmax(logits, dim=1)
    corrected_raw = torch.clamp(
        original_probability + u_effective - v_effective.detach(),
        min=float(eps),
    )
    corrected_probability = F.normalize(
        corrected_raw,
        p=1,
        dim=1,
        eps=float(eps),
    )
    corrected_probability = torch.clamp(
        corrected_probability,
        min=float(eps),
        max=1.0,
    )
    hard = torch.zeros_like(labels).scatter_(
        1,
        logits.detach().argmax(dim=1, keepdim=True),
        1.0,
    )
    mse = F.mse_loss(
        hard + u_effective - v_effective,
        labels,
        reduction="sum",
    ) / logits.shape[0]
    cross_entropy = torch.mean(
        -torch.sum(labels * torch.log(corrected_probability), dim=-1)
    )
    loss = cross_entropy + mse
    return loss, {
        "original_probability": original_probability,
        "corrected_probability": corrected_probability,
        "u_effective": u_effective,
        "v_effective": v_effective,
        "hard": hard,
        "cross_entropy": cross_entropy,
        "mse": mse,
    }


def sop_independent_equation(
    logits: Tensor,
    labels_one_hot: Tensor,
    u_rows: Tensor,
    v_rows: Tensor,
    *,
    eps: float = 1e-4,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Independent expression used as a second implementation oracle."""
    labels = labels_one_hot.to(device=logits.device)
    shifted = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    network_probability = shifted.exp()
    zeros = torch.zeros((), device=logits.device, dtype=u_rows.dtype)
    ones = torch.ones((), device=logits.device, dtype=u_rows.dtype)
    positive_noise = torch.minimum(
        torch.maximum(u_rows.pow(2) * labels, zeros), ones
    )
    negative_noise = torch.minimum(
        torch.maximum(v_rows.pow(2) * (ones - labels), zeros), ones
    )
    lower = torch.full((), float(eps), device=logits.device, dtype=u_rows.dtype)
    unnormalized = torch.maximum(
        network_probability + positive_noise - negative_noise.detach(), lower
    )
    normalized = unnormalized / unnormalized.abs().sum(dim=1, keepdim=True).clamp_min(
        float(eps)
    )
    corrected = torch.minimum(torch.maximum(normalized, lower), ones)
    hard_index = logits.detach().argmax(dim=1)
    hard = F.one_hot(hard_index, num_classes=logits.shape[1]).to(labels.dtype)
    ce = -(labels * corrected.log()).sum(dim=1).mean()
    mse = ((hard + positive_noise - negative_noise - labels).pow(2)).sum()
    mse = mse / logits.shape[0]
    return ce + mse, {
        "original_probability": network_probability,
        "corrected_probability": corrected,
        "u_effective": positive_noise,
        "v_effective": negative_noise,
        "hard": hard,
        "cross_entropy": ce,
        "mse": mse,
    }


def _maximum_error(left: Tensor, right: Tensor) -> float:
    return float((left.detach().double() - right.detach().double()).abs().amax().item())


def _finite_difference_diagnostics() -> Dict[str, object]:
    dtype = torch.float64
    logits = torch.tensor(
        [[0.7, -0.2, 0.1, -0.4, 0.3], [-0.1, 0.8, 0.2, 0.0, -0.3]],
        dtype=dtype,
        requires_grad=True,
    )
    targets = F.one_hot(torch.tensor([0, 1]), NUM_CLASSES).to(dtype)
    u = torch.tensor([[0.07], [-0.09]], dtype=dtype, requires_grad=True)
    v = torch.tensor(
        [[0.03, -0.04, 0.05, 0.02, -0.06], [0.04, 0.01, -0.03, 0.05, 0.02]],
        dtype=dtype,
        requires_grad=True,
    )
    loss, _ = sop_independent_equation(logits, targets, u, v)
    gradients = torch.autograd.grad(loss, (logits, u, v))
    with torch.no_grad():
        fixed_v_for_ce = torch.clamp(v.detach().square() * (1.0 - targets), 0.0, 1.0)

    def v_stop_gradient_surrogate(v_value: Tensor) -> Tensor:
        network_probability = torch.softmax(logits.detach(), dim=1)
        positive_noise = torch.clamp(u.detach().square() * targets, 0.0, 1.0)
        varying_v = torch.clamp(v_value.square() * (1.0 - targets), 0.0, 1.0)
        corrected_raw = torch.clamp(
            network_probability + positive_noise - fixed_v_for_ce,
            min=1e-4,
        )
        corrected = corrected_raw / corrected_raw.sum(dim=1, keepdim=True)
        corrected = torch.clamp(corrected, min=1e-4, max=1.0)
        hard = F.one_hot(
            logits.detach().argmax(dim=1), num_classes=NUM_CLASSES
        ).to(dtype)
        ce = -(targets * corrected.log()).sum(dim=1).mean()
        mse = ((hard + positive_noise - varying_v - targets).square()).sum()
        return ce + mse / logits.shape[0]

    epsilon = 1e-5
    probes = ((0, (0, 1)), (1, (1, 0)), (2, (0, 2)))
    maximum = 0.0
    rows: list[Dict[str, object]] = []
    bases = [logits.detach(), u.detach(), v.detach()]
    for tensor_index, position in probes:
        plus = [value.clone() for value in bases]
        minus = [value.clone() for value in bases]
        plus[tensor_index][position] += epsilon
        minus[tensor_index][position] -= epsilon
        if tensor_index == 2:
            # The official CE path intentionally stop-grads V. Hold that CE value
            # fixed while finite-differencing the trainable MSE path for v.
            plus_loss = v_stop_gradient_surrogate(plus[2])
            minus_loss = v_stop_gradient_surrogate(minus[2])
        else:
            plus_loss = sop_independent_equation(plus[0], targets, plus[1], plus[2])[0]
            minus_loss = sop_independent_equation(minus[0], targets, minus[1], minus[2])[0]
        numerical = float(((plus_loss - minus_loss) / (2.0 * epsilon)).item())
        analytic = float(gradients[tensor_index][position].item())
        error = abs(numerical - analytic)
        maximum = max(maximum, error)
        rows.append(
            {
                "tensor": ("logits", "u", "v")[tensor_index],
                "position": list(position),
                "analytic": analytic,
                "numerical": numerical,
                "absolute_error": error,
            }
        )
    return {
        "rows": rows,
        "v_finite_difference_holds_ce_stop_gradient_value_fixed": True,
        "maximum_absolute_error": maximum,
        "passed": maximum <= 1e-6,
    }


def _load_official_loss_class(loss_path: Path):
    source = Path(loss_path).read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(loss_path))
    class_names = {
        node.name for node in parsed.body if isinstance(node, ast.ClassDef)
    }
    if "overparametrization_loss" not in class_names:
        raise ValueError("Locked official SOP AST lacks overparametrization_loss.")
    root = Path(loss_path).resolve().parents[1]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = "trkh_locked_official_sop_loss"
    specification = importlib.util.spec_from_file_location(module_name, loss_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import official SOP loss: {loss_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.overparametrization_loss, ast.dump(parsed, include_attributes=False)


def _make_official_instance(loss_class, u: Tensor, v: Tensor) -> nn.Module:
    instance = object.__new__(loss_class)
    nn.Module.__init__(instance)
    instance.num_classes = NUM_CLASSES
    instance.config = {"num_classes": NUM_CLASSES}
    instance.USE_CUDA = True
    instance.num_examp = int(u.shape[0])
    instance.ratio_consistency = 0
    instance.ratio_balance = 0
    instance.u = nn.Parameter(u.detach().clone())
    instance.v = nn.Parameter(v.detach().clone())
    instance.E = None
    return instance


def official_source_replay(loss_path: Path, *, device: torch.device) -> Dict[str, object]:
    if device.type != "cuda":
        raise ValueError("The locked official SOP implementation hardcodes CUDA.")
    loss_class, ast_dump = _load_official_loss_class(loss_path)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    batch = 6
    logits_base = torch.randn(batch, NUM_CLASSES, generator=generator) * 0.7
    target_indices = torch.tensor([0, 1, 2, 3, 4, 1], dtype=torch.long)
    targets = F.one_hot(target_indices, NUM_CLASSES).float().to(device)
    u_base = (torch.randn(batch, 1, generator=generator) * 0.08).to(device)
    v_base = (torch.randn(batch, NUM_CLASSES, generator=generator) * 0.08).to(device)
    indices = torch.arange(batch, device=device)

    official = _make_official_instance(loss_class, u_base, v_base).to(device)
    official_logits = logits_base.to(device).requires_grad_(True)
    official_loss = official(indices, official_logits, targets)
    official_gradients = torch.autograd.grad(
        official_loss, (official_logits, official.u, official.v)
    )

    local_logits = logits_base.to(device).requires_grad_(True)
    local_u = u_base.detach().clone().requires_grad_(True)
    local_v = v_base.detach().clone().requires_grad_(True)
    local_loss, local_terms = sop_official_equation(local_logits, targets, local_u, local_v)
    local_gradients = torch.autograd.grad(local_loss, (local_logits, local_u, local_v))

    independent_logits = logits_base.to(device).requires_grad_(True)
    independent_u = u_base.detach().clone().requires_grad_(True)
    independent_v = v_base.detach().clone().requires_grad_(True)
    independent_loss, independent_terms = sop_independent_equation(
        independent_logits, targets, independent_u, independent_v
    )
    independent_gradients = torch.autograd.grad(
        independent_loss, (independent_logits, independent_u, independent_v)
    )

    official_local_gradient_error = max(
        _maximum_error(left, right)
        for left, right in zip(official_gradients, local_gradients)
    )
    local_independent_gradient_error = max(
        _maximum_error(left, right)
        for left, right in zip(local_gradients, independent_gradients)
    )
    probability_error = _maximum_error(
        local_terms["corrected_probability"],
        independent_terms["corrected_probability"],
    )

    official_step = _make_official_instance(loss_class, u_base, v_base).to(device)
    local_step_u = nn.Parameter(u_base.detach().clone())
    local_step_v = nn.Parameter(v_base.detach().clone())
    official_optimizer = torch.optim.SGD(
        [
            {"params": [official_step.u], "lr": 0.1},
            {"params": [official_step.v], "lr": 1.0},
        ],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
    )
    local_optimizer = torch.optim.SGD(
        [
            {"params": [local_step_u], "lr": 0.1},
            {"params": [local_step_v], "lr": 1.0},
        ],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
    )
    official_optimizer.zero_grad(set_to_none=True)
    official_step_loss = official_step(indices, logits_base.to(device), targets)
    official_step_loss.backward()
    official_optimizer.step()
    local_optimizer.zero_grad(set_to_none=True)
    local_step_loss = sop_official_equation(
        logits_base.to(device), targets, local_step_u, local_step_v
    )[0]
    local_step_loss.backward()
    local_optimizer.step()
    step_error = max(
        _maximum_error(official_step.u, local_step_u),
        _maximum_error(official_step.v, local_step_v),
    )

    bf16_logits = logits_base.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    bf16_u = u_base.detach().clone().requires_grad_(True)
    bf16_v = v_base.detach().clone().requires_grad_(True)
    bf16_loss, bf16_terms = sop_official_equation(bf16_logits, targets, bf16_u, bf16_v)
    bf16_gradients = torch.autograd.grad(bf16_loss, (bf16_logits, bf16_u, bf16_v))
    bf16_probability_error = _maximum_error(
        bf16_terms["corrected_probability"], local_terms["corrected_probability"]
    )
    finite_difference = _finite_difference_diagnostics()
    checks = {
        "official_ast_parsed": bool(ast_dump),
        "official_loss_match": abs(float(official_loss.item()) - float(local_loss.item()))
        <= 1e-6,
        "independent_loss_match": abs(
            float(independent_loss.item()) - float(local_loss.item())
        )
        <= 1e-6,
        "official_gradient_match": official_local_gradient_error <= 1e-6,
        "independent_gradient_match": local_independent_gradient_error <= 1e-6,
        "independent_probability_match": probability_error <= 1e-6,
        "official_optimizer_step_match": step_error <= 1e-6,
        "finite_difference_match": bool(finite_difference["passed"]),
        "bf16_finite": bool(
            torch.isfinite(bf16_loss).all().item()
            and torch.isfinite(bf16_terms["corrected_probability"]).all().item()
            and all(torch.isfinite(value).all().item() for value in bf16_gradients)
        ),
        "bf16_probability_error": bf16_probability_error <= 2e-3,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "official_loss": float(official_loss.item()),
        "local_loss": float(local_loss.item()),
        "independent_loss": float(independent_loss.item()),
        "official_local_gradient_max_abs_error": official_local_gradient_error,
        "local_independent_gradient_max_abs_error": local_independent_gradient_error,
        "local_independent_probability_max_abs_error": probability_error,
        "optimizer_step_max_abs_error": step_error,
        "bf16_probability_max_abs_error": bf16_probability_error,
        "finite_difference": finite_difference,
        "official_ast_sha256": hashlib.sha256(ast_dump.encode("utf-8")).hexdigest(),
    }


class _FitDataset(Dataset):
    def __init__(
        self,
        base_dataset: MangoYOLOCropDataset,
        transform,
        original_indices: Sequence[int],
    ) -> None:
        if len(base_dataset) != len(original_indices):
            raise ValueError("Fold train rows and original fit indices differ.")
        self.base_dataset = base_dataset
        self.transform = transform
        self.original_indices = [int(value) for value in original_indices]
        self.corruption = IdentityCorruption()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, position: int):
        image, label, metadata, bbox = _unpack_classification_sample(
            self.base_dataset[int(position)]
        )
        image = self.corruption(image)
        tensor, transformed_metadata = _transform_classification_image(
            image,
            label=label,
            metadata=metadata,
            bbox=bbox,
            transform=self.transform,
        )
        transformed_metadata["sample_index"] = torch.tensor(
            self.original_indices[int(position)], dtype=torch.long
        )
        return tensor, label, transformed_metadata


def _fit_rows(rows: Sequence[CleanTrainRow]) -> tuple[list[CleanTrainRow], list[int]]:
    selected = [row for row in rows if int(row.fold) != FOLD]
    indices = [int(row.sample_index) for row in selected]
    if len(selected) != EXPECTED_FIT_ROWS:
        raise ValueError(f"SOP fit rows differ: {len(selected)} != {EXPECTED_FIT_ROWS}")
    if _ordered_index_sha256(indices) != LOCKED_FIT_INDEX_SHA256:
        raise ValueError("SOP ordered fit-index hash differs from the protocol.")
    holdout_indices = [int(row.sample_index) for row in rows if int(row.fold) == FOLD]
    if len(holdout_indices) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("SOP holdout declaration count differs from the protocol.")
    if _ordered_index_sha256(holdout_indices) != LOCKED_HOLDOUT_INDEX_SHA256:
        raise ValueError("SOP ordered holdout-index hash differs from the protocol.")
    return selected, indices


def _validate_fold_summary(summary: Mapping[str, object]) -> Dict[str, bool]:
    checks = {
        "fit_rows": int(summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "holdout_rows": int(summary.get("holdout_rows", -1)) == EXPECTED_HOLDOUT_ROWS,
        "fit_counts": summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "holdout_counts": summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fit_index_hash": summary.get("fit_index_sha256")
        == LOCKED_FIT_INDEX_SHA256,
        "holdout_index_hash": summary.get("holdout_index_sha256")
        == LOCKED_HOLDOUT_INDEX_SHA256,
        "source_overlap_zero": int(summary.get("source_overlap", -1)) == 0,
        "test_mirrors_holdout_declared": bool(summary.get("test_mirrors_holdout")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"SOP fold summary differs: {failed}")
    return checks


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official_root = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "nbdt_closure": Path(args.nbdt_closure).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_loss": official_root / "model" / "loss.py",
        "official_trainer": official_root / "trainer" / "trainer.py",
        "official_license": official_root / "LICENSE",
        "current_commands": Path(args.current_commands).resolve(),
        "command_history": Path(args.command_history).resolve(),
    }


def _verify_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], list[int]]:
    if (
        int(args.batch_size) != 64
        or int(args.noise_batch_size) != 128
        or int(args.passes) != 10
        or int(args.seed) != SEED
        or not math.isclose(float(args.u_lr), 0.1)
        or not math.isclose(float(args.v_lr), 1.0)
    ):
        raise ValueError("SOP A0 schedule differs from the precommitted protocol.")
    paths = _source_paths(args)
    expected_hashes = {
        "checkpoint": LOCKED_KEEPER_SHA256,
        "fold_data": LOCKED_FOLD_DATA_SHA256,
        "fold_summary": LOCKED_FOLD_SUMMARY_SHA256,
        "raw_data": LOCKED_RAW_DATA_SHA256,
        "declaration": LOCKED_DECLARATION_SHA256,
        "protocol": LOCKED_PROTOCOL_SHA256,
        "nbdt_closure": LOCKED_NBDT_CLOSURE_SHA256,
        "paper": LOCKED_PAPER_SHA256,
        "official_loss": LOCKED_OFFICIAL_LOSS_SHA256,
        "official_trainer": LOCKED_OFFICIAL_TRAINER_SHA256,
        "official_license": LOCKED_OFFICIAL_LICENSE_SHA256,
        "current_commands": LOCKED_COMMAND_SHA256,
        "command_history": LOCKED_HISTORY_SHA256,
    }
    hashes = {
        name: _verify_sha256(paths[name], expected, name)
        for name, expected in expected_hashes.items()
    }
    fold_summary = _load_json(paths["fold_summary"])
    fold_checks = _validate_fold_summary(fold_summary)
    rows = _read_clean_train_rows(paths["declaration"])
    selected_rows, fit_indices = _fit_rows(rows)
    fit_counts = np.bincount(
        np.asarray([row.target for row in selected_rows], dtype=np.int64),
        minlength=NUM_CLASSES,
    ).tolist()
    if fit_counts != EXPECTED_FIT_COUNTS:
        raise ValueError(f"SOP declaration fit class counts differ: {fit_counts}")

    repository_root = Path.cwd().resolve()
    official_root = Path(args.official_root).resolve()
    repository_clean = _tracked_worktree_clean(repository_root)
    head = _git_value(repository_root, "rev-parse", "HEAD")
    upstream = _git_value(repository_root, "rev-parse", "@{upstream}")
    official_commit = _git_value(official_root, "rev-parse", "HEAD")
    official_tree = _git_value(official_root, "rev-parse", "HEAD^{tree}")
    official_clean = _tracked_worktree_clean(official_root)
    checks = {
        "repository_tracked_clean": repository_clean,
        "repository_head_pushed": head == upstream,
        "official_commit": official_commit == LOCKED_OFFICIAL_COMMIT,
        "official_tree": official_tree == LOCKED_OFFICIAL_TREE,
        "official_tracked_clean": official_clean,
        **{f"fold_{name}": passed for name, passed in fold_checks.items()},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"SOP source/preflight checks failed: {failed}")
    return (
        {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": hashes,
            "checks": checks,
            "repository_head": head,
            "repository_upstream": upstream,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "fit_rows": EXPECTED_FIT_ROWS,
            "fit_index_sha256": _ordered_index_sha256(fit_indices),
            "holdout_loader_constructed": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        selected_rows,
        fit_indices,
    )


def _build_fit_dataset(
    checkpoint: Mapping[str, object],
    class_names: Sequence[str],
    fold_data: Path,
    selected_rows: Sequence[CleanTrainRow],
    fit_indices: Sequence[int],
) -> tuple[_FitDataset, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(fold_data, class_name_mode="raw", expected_num_classes=5)
    if list(data_spec.class_names) != list(class_names):
        raise ValueError("SOP fold class order differs from the keeper.")
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
    if len(base_dataset) != EXPECTED_FIT_ROWS:
        raise ValueError(f"SOP fold train rows differ: {len(base_dataset)}")
    if base_dataset.class_counts(NUM_CLASSES) != EXPECTED_FIT_COUNTS:
        raise ValueError("SOP fold train class counts differ from the protocol.")
    for position, (sample, declared) in enumerate(zip(base_dataset.samples, selected_rows)):
        path = Path(sample.image_path).resolve()
        parts = {part.casefold() for part in path.parts}
        if "train" not in parts or "val" in parts or "test" in parts:
            raise ValueError(f"Non-fit path entered SOP at row {position}: {path}")
        if path.stem.casefold() != declared.source_stem:
            raise ValueError(f"SOP source order differs at fit row {position}.")
        if int(sample.primary_label) != int(declared.target):
            raise ValueError(f"SOP target order differs at fit row {position}.")
    transform = _build_eval_transform(semantics)
    return _FitDataset(base_dataset, transform, fit_indices), {
        "rows": len(base_dataset),
        "class_counts": base_dataset.class_counts(NUM_CLASSES),
        "source_target_order_exact": True,
        "split_constructed": "train",
        "holdout_split_constructed": False,
        "semantics": dict(semantics),
    }


def _classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(1, support.sum())),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "support": support.tolist(),
        "predicted_support": predicted_support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _extract_fit_logits(
    model: nn.Module,
    dataset: Dataset,
    *,
    fit_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[Tensor, Tensor, Tensor, Dict[str, object]]:
    kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context="sop_a0_fit_logits",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    logits_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    index_rows: list[Tensor] = []
    processed = 0
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            logits, features = _forward_classification_with_metadata(
                model, images, metadata, device=device
            )
            del features
            logits_rows.append(logits.detach().float().cpu())
            target_rows.append(targets.detach().long().cpu())
            index_rows.append(metadata["sample_index"].detach().long().cpu())
            processed += int(targets.numel())
            if processed % 1024 < int(targets.numel()) or processed == len(dataset):
                print(
                    json.dumps(
                        {
                            "phase": "fit_logits",
                            "processed": processed,
                            "rows": len(dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    logits = torch.cat(logits_rows)
    targets = torch.cat(target_rows)
    indices = torch.cat(index_rows)
    if indices.tolist() != [int(value) for value in fit_indices]:
        raise ValueError("SOP fit loader changed ordered original sample identity.")
    if logits.shape != (EXPECTED_FIT_ROWS, NUM_CLASSES):
        raise ValueError(f"SOP fit logits have invalid shape: {tuple(logits.shape)}")
    return logits, targets, indices, {
        "loader": loader_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "rows": int(logits.shape[0]),
    }


def _initialize_noise(rows: int, *, seed: int, device: torch.device) -> tuple[nn.Parameter, nn.Parameter]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    u_cpu = torch.empty(rows, 1, dtype=torch.float32)
    v_cpu = torch.empty(rows, NUM_CLASSES, dtype=torch.float32)
    nn.init.normal_(u_cpu, mean=0.0, std=1e-8, generator=generator)
    nn.init.normal_(v_cpu, mean=0.0, std=1e-8, generator=generator)
    return nn.Parameter(u_cpu.to(device)), nn.Parameter(v_cpu.to(device))


def optimize_noise_buffers(
    logits_cpu: Tensor,
    targets_cpu: Tensor,
    sample_indices_cpu: Tensor,
    *,
    device: torch.device,
    passes: int = 10,
    batch_size: int = 128,
    u_lr: float = 0.1,
    v_lr: float = 1.0,
    seed: int = SEED,
) -> tuple[Tensor, Tensor, Dict[str, object]]:
    logits = logits_cpu.to(device=device, dtype=torch.float32)
    targets = targets_cpu.to(device=device, dtype=torch.long)
    u, v = _initialize_noise(len(targets_cpu), seed=seed, device=device)
    optimizer = torch.optim.SGD(
        [
            {"params": [u], "lr": float(u_lr)},
            {"params": [v], "lr": float(v_lr)},
        ],
        lr=float(u_lr),
        momentum=0.0,
        weight_decay=0.0,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(seed))
    occurrence = hashlib.sha256()
    pass_rows: list[Dict[str, object]] = []
    started = time.perf_counter()
    for pass_index in range(int(passes)):
        permutation = torch.randperm(len(targets_cpu), generator=shuffle_generator)
        loss_sum = 0.0
        ce_sum = 0.0
        mse_sum = 0.0
        seen = 0
        for start in range(0, len(targets_cpu), int(batch_size)):
            local_cpu = permutation[start : start + int(batch_size)]
            for local_index in local_cpu.tolist():
                occurrence.update(
                    f"{pass_index}:{int(sample_indices_cpu[local_index].item())}\n".encode(
                        "ascii"
                    )
                )
            local = local_cpu.to(device=device)
            one_hot = F.one_hot(targets[local], NUM_CLASSES).to(torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = sop_official_equation(
                logits[local], one_hot, u[local], v[local]
            )
            loss.backward()
            optimizer.step()
            count = int(local.numel())
            loss_sum += float(loss.detach().item()) * count
            ce_sum += float(terms["cross_entropy"].detach().item()) * count
            mse_sum += float(terms["mse"].detach().item()) * count
            seen += count
        with torch.no_grad():
            labels_all = F.one_hot(targets, NUM_CLASSES).to(torch.float32)
            v_effective = torch.clamp(v.square() * (1.0 - labels_all), 0.0, 1.0)
            pass_rows.append(
                {
                    "pass": pass_index + 1,
                    "mean_loss": loss_sum / seen,
                    "mean_cross_entropy": ce_sum / seen,
                    "mean_mse": mse_sum / seen,
                    "u_abs_max": float(u.abs().amax().item()),
                    "v_abs_max": float(v.abs().amax().item()),
                    "active_v_row_fraction": float(
                        (v_effective.amax(dim=1) > 1e-6).float().mean().item()
                    ),
                }
            )
        print(json.dumps({"phase": "noise_pass", **pass_rows[-1]}), flush=True)
    return u.detach().cpu(), v.detach().cpu(), {
        "passes": pass_rows,
        "occurrence_sha256": occurrence.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer": {
            "name": "SGD",
            "u_lr": float(u_lr),
            "v_lr": float(v_lr),
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    }


def _final_row_diagnostics(
    logits_cpu: Tensor,
    targets_cpu: Tensor,
    u_cpu: Tensor,
    v_cpu: Tensor,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, Tensor]:
    collected: Dict[str, list[Tensor]] = {
        "original_probability": [],
        "corrected_probability": [],
        "u_effective": [],
        "v_effective": [],
        "sop_gradient": [],
    }
    for start in range(0, len(targets_cpu), int(batch_size)):
        stop = min(start + int(batch_size), len(targets_cpu))
        logits = logits_cpu[start:stop].to(device=device).detach().requires_grad_(True)
        targets = targets_cpu[start:stop].to(device=device)
        one_hot = F.one_hot(targets, NUM_CLASSES).to(torch.float32)
        u = u_cpu[start:stop].to(device=device)
        v = v_cpu[start:stop].to(device=device)
        loss, terms = sop_official_equation(logits, one_hot, u, v)
        gradient = torch.autograd.grad(loss, logits)[0] * logits.shape[0]
        for name in (
            "original_probability",
            "corrected_probability",
            "u_effective",
            "v_effective",
        ):
            collected[name].append(terms[name].detach().float().cpu())
        collected["sop_gradient"].append(gradient.detach().float().cpu())
    result = {name: torch.cat(values) for name, values in collected.items()}
    one_hot_cpu = F.one_hot(targets_cpu.long(), NUM_CLASSES).float()
    result["ce_gradient"] = result["original_probability"] - one_hot_cpu
    result["noise_energy"] = (
        result["u_effective"].square() + result["v_effective"].square()
    ).sum(dim=1)
    return result


def assess_gradient_gate(
    *,
    targets: Tensor,
    raw_predictions: Tensor,
    corrected_probability: Tensor,
    ce_gradient: Tensor,
    sop_gradient: Tensor,
    u: Tensor,
    v: Tensor,
    u_effective: Tensor,
    v_effective: Tensor,
    noise_energy: Tensor,
    replay_passed: bool,
    source_replay_passed: bool,
) -> Dict[str, object]:
    restricted_mask = (
        (raw_predictions == FOCUS_CLASS)
        & torch.isin(targets, torch.tensor(RESTRICTED_NEGATIVE_CLASSES))
    )
    true_positive_mask = (targets == FOCUS_CLASS) & (raw_predictions == FOCUS_CLASS)
    raw_error_mask = raw_predictions != targets
    other_error_mask = raw_error_mask & ~restricted_mask
    ce_focus = ce_gradient[:, FOCUS_CLASS]
    sop_focus = sop_gradient[:, FOCUS_CLASS]
    ratio = sop_focus.abs() / ce_focus.abs().clamp_min(1e-12)

    def cohort(mask: Tensor, *, expected_sign: str) -> Dict[str, object]:
        selected = sop_focus[mask]
        selected_ratio = ratio[mask]
        sign = selected > 0 if expected_sign == "positive" else selected < 0
        return {
            "rows": int(mask.sum().item()),
            "expected_sign": expected_sign,
            "sign_retained_rows": int(sign.sum().item()),
            "all_signs_retained": bool(sign.all().item()),
            "median_magnitude_ratio": float(selected_ratio.median().item()),
            "lost_more_than_half_fraction": float((selected_ratio < 0.5).float().mean().item()),
            "minimum_magnitude_ratio": float(selected_ratio.amin().item()),
        }

    restricted = cohort(restricted_mask, expected_sign="positive")
    true_positive = cohort(true_positive_mask, expected_sign="negative")
    active_v_fraction = float(
        (v_effective.amax(dim=1) > 1e-6).float().mean().item()
    )
    class1_clamped = corrected_probability[restricted_mask, FOCUS_CLASS] <= 1.00001e-4
    class1_zero_gradient = sop_focus[restricted_mask].abs() <= 1e-12
    clamped_zero_count = int((class1_clamped & class1_zero_gradient).sum().item())
    restricted_energy = float(noise_energy[restricted_mask].mean().item())
    other_error_energy = float(noise_energy[other_error_mask].mean().item())
    corrected_prediction = corrected_probability.argmax(dim=1)
    probability_sum_error = float(
        (corrected_probability.sum(dim=1) - 1.0).abs().amax().item()
    )
    finite = bool(
        all(
            torch.isfinite(value).all().item()
            for value in (
                corrected_probability,
                ce_gradient,
                sop_gradient,
                u,
                v,
                u_effective,
                v_effective,
                noise_energy,
            )
        )
    )
    checks = {
        "source_equation_replay": bool(source_replay_passed),
        "keeper_fit_replay": bool(replay_passed),
        "all_rows_and_tensors_finite": finite and len(targets) == EXPECTED_FIT_ROWS,
        "effective_v_active_at_least_1pct": active_v_fraction >= 0.01,
        "noise_parameters_within_unit_domain": bool(
            u.abs().amax().item() <= 1.0 and v.abs().amax().item() <= 1.0
        ),
        "restricted_fp_count_exact": int(restricted_mask.sum().item())
        == EXPECTED_RESTRICTED_FP,
        "restricted_fp_corrective_sign_retained": bool(restricted["all_signs_retained"]),
        "restricted_fp_median_ratio_at_least_0p95": float(
            restricted["median_magnitude_ratio"]
        )
        >= 0.95,
        "restricted_fp_half_loss_at_most_5pct": float(
            restricted["lost_more_than_half_fraction"]
        )
        <= 0.05,
        "class1_tp_count_exact": int(true_positive_mask.sum().item())
        == EXPECTED_CLASS1_TP,
        "class1_tp_support_sign_retained": bool(true_positive["all_signs_retained"]),
        "class1_tp_median_ratio_at_least_0p95": float(
            true_positive["median_magnitude_ratio"]
        )
        >= 0.95,
        "class1_tp_half_loss_at_most_5pct": float(
            true_positive["lost_more_than_half_fraction"]
        )
        <= 0.05,
        "no_restricted_fp_class1_clamp_zero_gradient": clamped_zero_count == 0,
        "restricted_fp_noise_not_preferential": restricted_energy
        <= other_error_energy,
        "corrected_probability_normalized": probability_sum_error <= 1e-6,
        "corrected_probability_not_single_class": int(corrected_prediction.unique().numel())
        > 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "trainer_integration_authorized": not failed,
        "image_epoch_authorized": not failed,
        "validation_authorized": False,
        "test_authorized": False,
        "restricted_false_positive": restricted,
        "class1_true_positive": true_positive,
        "active_v_row_fraction": active_v_fraction,
        "u_abs_max": float(u.abs().amax().item()),
        "v_abs_max": float(v.abs().amax().item()),
        "clamped_zero_gradient_restricted_fp": clamped_zero_count,
        "restricted_fp_mean_noise_energy": restricted_energy,
        "other_raw_error_mean_noise_energy": other_error_energy,
        "corrected_probability_max_sum_error": probability_sum_error,
        "corrected_prediction_support": torch.bincount(
            corrected_prediction, minlength=NUM_CLASSES
        ).tolist(),
    }


def _write_gradient_rows(
    path: Path,
    *,
    sample_indices: Tensor,
    targets: Tensor,
    raw_predictions: Tensor,
    diagnostics: Mapping[str, Tensor],
    u: Tensor,
    v: Tensor,
) -> None:
    fields = [
        "sample_index",
        "target",
        "raw_prediction",
        "raw_correct",
        "restricted_fp",
        "class1_tp",
        *[f"raw_prob_{index}" for index in range(NUM_CLASSES)],
        *[f"sop_prob_{index}" for index in range(NUM_CLASSES)],
        "u_parameter",
        "v_parameter_abs_max",
        "u_effective_target",
        "v_effective_max",
        "noise_energy",
        "ce_gradient_class1",
        "sop_gradient_class1",
        "class1_gradient_magnitude_ratio",
        "sop_prediction",
    ]
    probabilities = diagnostics["original_probability"]
    corrected = diagnostics["corrected_probability"]
    u_effective = diagnostics["u_effective"]
    v_effective = diagnostics["v_effective"]
    ce_focus = diagnostics["ce_gradient"][:, FOCUS_CLASS]
    sop_focus = diagnostics["sop_gradient"][:, FOCUS_CLASS]
    ratios = sop_focus.abs() / ce_focus.abs().clamp_min(1e-12)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position in range(len(targets)):
            target = int(targets[position].item())
            prediction = int(raw_predictions[position].item())
            row: Dict[str, object] = {
                "sample_index": int(sample_indices[position].item()),
                "target": target,
                "raw_prediction": prediction,
                "raw_correct": prediction == target,
                "restricted_fp": prediction == FOCUS_CLASS
                and target in RESTRICTED_NEGATIVE_CLASSES,
                "class1_tp": prediction == FOCUS_CLASS and target == FOCUS_CLASS,
                "u_parameter": float(u[position, 0].item()),
                "v_parameter_abs_max": float(v[position].abs().amax().item()),
                "u_effective_target": float(u_effective[position, target].item()),
                "v_effective_max": float(v_effective[position].amax().item()),
                "noise_energy": float(diagnostics["noise_energy"][position].item()),
                "ce_gradient_class1": float(ce_focus[position].item()),
                "sop_gradient_class1": float(sop_focus[position].item()),
                "class1_gradient_magnitude_ratio": float(ratios[position].item()),
                "sop_prediction": int(corrected[position].argmax().item()),
            }
            for class_index in range(NUM_CLASSES):
                row[f"raw_prob_{class_index}"] = float(
                    probabilities[position, class_index].item()
                )
                row[f"sop_prob_{class_index}"] = float(
                    corrected[position, class_index].item()
                )
            writer.writerow(row)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    gate = summary["gate"]
    replay = summary["keeper_fit_replay"]
    lines = [
        "# SOP A0 Fit-Gradient Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Phase-A pass: `{gate['all_gates_passed']}`",
        f"- Fit rows: `{replay['rows']}`",
        f"- Macro/class-1 F1: `{replay['metrics']['macro_f1']:.6f}` / "
        f"`{replay['metrics']['per_class_f1'][1]:.6f}`",
        f"- Restricted false positives: `{gate['restricted_false_positive']['rows']}`",
        f"- Class-1 true positives: `{gate['class1_true_positive']['rows']}`",
        f"- Effective-V active fraction: `{gate['active_v_row_fraction']:.8f}`",
        f"- Failed gates: `{', '.join(gate['failed_checks']) or 'none'}`",
        "- Holdout/validation/test used: `false/false/false`",
        "",
        "A failure closes this exact SOP A0 route before trainer integration. It does "
        "not authorize a learning-rate, buffer, loss, epoch, seed, fold, SGN, or "
        "Label-Wave rescue sweep.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {"method": METHOD, "artifacts": artifacts}
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance, selected_rows, fit_indices = _verify_inputs(args)
    if bool(args.preflight_only):
        return {
            "method": METHOD,
            "status": "preflight_passed",
            "provenance": provenance,
            "output_directory_created": False,
            "image_data_opened": False,
            "holdout_loader_constructed": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("The locked SOP A0 formal gate requires CUDA.")
    device = torch.device("cuda")
    set_seed(SEED, deterministic=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    source_replay = official_source_replay(
        Path(provenance["paths"]["official_loss"]), device=device
    )
    if not bool(source_replay["all_checks_passed"]):
        raise ValueError("Official/independent SOP equation replay failed.")

    model, checkpoint, class_names = load_model(Path(args.checkpoint), device)
    model_state_before = _model_state_sha256(model)
    dataset, dataset_summary = _build_fit_dataset(
        checkpoint,
        class_names,
        Path(args.fold_data),
        selected_rows,
        fit_indices,
    )
    logits, targets, sample_indices, inference = _extract_fit_logits(
        model,
        dataset,
        fit_indices=fit_indices,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    model_state_after = _model_state_sha256(model)
    if model_state_before != model_state_after:
        raise RuntimeError("SOP fit inference modified the keeper state.")
    probabilities = F.softmax(logits, dim=1)
    raw_predictions = probabilities.argmax(dim=1)
    targets_numpy = targets.numpy()
    predictions_numpy = raw_predictions.numpy()
    metrics = _classification_metrics(targets_numpy, predictions_numpy)
    prediction_replay_rows = sum(
        int(predictions_numpy[position]) == int(row.keeper_prediction)
        for position, row in enumerate(selected_rows)
    )
    class1_tp = int(
        ((targets == FOCUS_CLASS) & (raw_predictions == FOCUS_CLASS)).sum().item()
    )
    class1_fp = int(
        ((targets != FOCUS_CLASS) & (raw_predictions == FOCUS_CLASS)).sum().item()
    )
    restricted_fp = int(
        (
            (raw_predictions == FOCUS_CLASS)
            & torch.isin(targets, torch.tensor(RESTRICTED_NEGATIVE_CLASSES))
        ).sum().item()
    )
    replay_checks = {
        "rows_exact": len(targets) == EXPECTED_FIT_ROWS,
        "targets_exact": targets.tolist() == [int(row.target) for row in selected_rows],
        "declaration_predictions_exact": prediction_replay_rows == EXPECTED_FIT_ROWS,
        "confusion_exact": metrics["confusion_matrix"] == EXPECTED_CONFUSION,
        "macro_f1": abs(float(metrics["macro_f1"]) - EXPECTED_METRICS["macro_f1"])
        <= 1e-6,
        "class1_f1": abs(
            float(metrics["per_class_f1"][FOCUS_CLASS])
            - EXPECTED_METRICS["class1_f1"]
        )
        <= 1e-6,
        "class1_precision": abs(
            float(metrics["per_class_precision"][FOCUS_CLASS])
            - EXPECTED_METRICS["class1_precision"]
        )
        <= 1e-6,
        "class1_recall": abs(
            float(metrics["per_class_recall"][FOCUS_CLASS])
            - EXPECTED_METRICS["class1_recall"]
        )
        <= 1e-6,
        "class1_tp_exact": class1_tp == EXPECTED_CLASS1_TP,
        "class1_fp_exact": class1_fp == EXPECTED_CLASS1_FP,
        "restricted_fp_exact": restricted_fp == EXPECTED_RESTRICTED_FP,
    }
    if not all(replay_checks.values()):
        raise ValueError(
            f"Keeper fit replay differs from the protocol: "
            f"{[name for name, passed in replay_checks.items() if not passed]}"
        )

    u, v, optimization = optimize_noise_buffers(
        logits,
        targets,
        sample_indices,
        device=device,
        passes=int(args.passes),
        batch_size=int(args.noise_batch_size),
        u_lr=float(args.u_lr),
        v_lr=float(args.v_lr),
        seed=int(args.seed),
    )
    diagnostics = _final_row_diagnostics(
        logits,
        targets,
        u,
        v,
        device=device,
        batch_size=int(args.noise_batch_size),
    )
    gate = assess_gradient_gate(
        targets=targets,
        raw_predictions=raw_predictions,
        corrected_probability=diagnostics["corrected_probability"],
        ce_gradient=diagnostics["ce_gradient"],
        sop_gradient=diagnostics["sop_gradient"],
        u=u,
        v=v,
        u_effective=diagnostics["u_effective"],
        v_effective=diagnostics["v_effective"],
        noise_energy=diagnostics["noise_energy"],
        replay_passed=all(replay_checks.values()),
        source_replay_passed=bool(source_replay["all_checks_passed"]),
    )
    status = "phase_a_passed" if gate["all_gates_passed"] else "rejected_phase_a"
    summary: Dict[str, object] = {
        "method": METHOD,
        "status": status,
        "provenance": provenance,
        "dataset": dataset_summary,
        "inference": inference,
        "source_equation_replay": source_replay,
        "keeper_fit_replay": {
            "rows": len(targets),
            "metrics": metrics,
            "class1_tp": class1_tp,
            "class1_fp": class1_fp,
            "restricted_fp": restricted_fp,
            "declaration_prediction_rows_matched": prediction_replay_rows,
            "checks": replay_checks,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
        },
        "noise_optimization": optimization,
        "gate": gate,
        "trainer_modified": False,
        "model_modified": False,
        "checkpoint_created": False,
        "image_epoch_started": False,
        "holdout_loader_constructed": False,
        "validation_predictions_used": False,
        "test_data_used": False,
    }
    output_dir = _prepare_output(Path(args.output_dir), raw_data=Path(args.raw_data))
    _write_gradient_rows(
        output_dir / "gradient_rows.csv",
        sample_indices=sample_indices,
        targets=targets,
        raw_predictions=raw_predictions,
        diagnostics=diagnostics,
        u=u,
        v=v,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    return {**summary, "artifact_manifest": manifest}


def main(argv: Optional[Sequence[str]] = None) -> None:
    result = run_audit(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
