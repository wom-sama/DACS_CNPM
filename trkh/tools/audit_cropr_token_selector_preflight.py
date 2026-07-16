from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.cropr_token_selector import CroprTokenSelector
from trkh.models.model import classification_logits_from_features, create_model
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
)
from trkh.tools.audit_foveal_aggregated_attention_preflight import (
    _build_dataset,
    _forward,
    _git_value,
    _load_json,
    _prepare_output,
    _sha256,
    _tracked_worktree_clean,
    _unpack_batch,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.training.train import _cropr_auxiliary_loss_from_features


METHOD = "cropr_token_selector_a0_preflight"
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_OFFICIAL_COMMIT = "fa259e9030f5fddf4721ac75cdd18561524de6f9"
LOCKED_OFFICIAL_TREE = "4a83993890026c85cfd81b559b08c341fd86851d"
LOCKED_HASHES = {
    "protocol": "42e912a6bdc320d98a9acbb79b6d48c1e372a5df861f279af1c6988b3ff6a847",
    "fit_only_data": "5306a58516ecbc85cfb775c734413bdd24561e3ddbfbe0a7245c90461a00ff79",
    "resolved_config": "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "declaration_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "fold_data": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "current_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    "v8_launcher": "56e39d8f98004025cab6bdb6d9603c89c07d91bfea3e0f5fb06eaea237ea6b74",
    "selector_runtime": "0d9debc76f957cc638d11d442d915badeadcbaa0d400c049a018fb04c67192b3",
    "model_runtime": "57942c7e0457e947d21b77d20ad789dc6694b31e5d457a2b65dc247efe430f7f",
    "train_runtime": "17e82fbb26f6919c6e62bd49ba6d3015ad9e782fbdaf926819e055c84b93d3a7",
    "config_runtime": "1f63ec7934e5f1bfca607079f1a67c3164d0ddb83f6fb99775899b0a1d569e90",
    "selector_tests": "47209a4722d134e541c87f771f96e769d09941a2f71b76e0be5549279fe0f8bc",
    "official_cropr": "3f47ab92dc2509eafcf3a5e247b317637daf5d6fae177a20ba6de83b563402bd",
    "official_vit": "6ba20db35c770f95d9fcbd57c0ea2f7c57be1ae3a0407097f452bb316c176355",
    "official_engine": "e23c80bc52aeb83e1f9494db05f5f57d0301a28ce7b097b474d91abf6d3aef42",
    "official_recipe": "7e160dc61a1a611df3a512902d211214eb2a0d161313996a4fcb8cca29e5fb1b",
    "official_license": "49bae98540619fac9cf9546aee90df20d77ff754f2fddfb38059fe404d9b2596",
    "paper": "7525a21f05f1ddd982fea1d7d5777b1e87175aeb52d60729eedcb0e448f1a97d",
}
MAX_INFERENCE_RUNTIME_RATIO = 1.08
MAX_TRAIN_RUNTIME_RATIO = 1.20
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.15


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Cropr token-selector A0 engineering preflight. "
            "Official validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--keeper-checkpoint",
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
        "--fit-only-data",
        type=Path,
        default=Path("configs/trkh_cropr_a0_fitonly_20260716.yaml"),
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
        "--declaration-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\cropr-cvpr2025"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\cropr-cvpr2025-paper.pdf"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    return parser.parse_args(argv)


def _rng_sha256(state: Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_snapshot() -> Dict[str, object]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return bool(
        torch.equal(left["cpu"], right["cpu"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(
            torch.equal(a, b)
            for a, b in zip(left["cuda"], right["cuda"])
        )
    )


def _rng_summary(state: Mapping[str, object]) -> Dict[str, object]:
    return {
        "cpu_sha256": _rng_sha256(state["cpu"]),
        "cuda_sha256": [_rng_sha256(value) for value in state["cuda"]],
    }


def _selector_config(
    source: Mapping[str, object],
    *,
    enabled: bool,
    routing: bool,
) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "token_pruning": True,
            "token_prune_layers": "2,5",
            "token_keep_rates": "0.85,0.65",
            "token_prune_foreground_weight": 0.35,
            "early_token_mask_keep_rate": 1.0,
            "inattentive_token_fusion": False,
            "cropr_token_selector": bool(enabled),
            "cropr_token_selector_routing": bool(routing),
            "gated_relative_position_attention": False,
            "visual_contrast_attention": False,
            "foveal_aggregated_attention": False,
            "deformable_spatial_attention": False,
            "bi_level_routing_attention": False,
            "cross_covariance_attention": False,
            "dynamic_graph_mixer": False,
            "soft_moe_patch_adapter": False,
            "deep_class_prompt": False,
            "layer_token_fusion": False,
            "patch_style_recalibration": False,
            "block_local_patch_mixer": False,
            "locally_enhanced_ffn": False,
            "concurrent_local_global_coupling": False,
            "late_member_branch": False,
        }
    )
    return config


def _state_summary(left: nn.Module, right: nn.Module) -> Dict[str, object]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    left_keys = list(left_state)
    right_keys = list(right_state)
    differing = [
        name
        for name in sorted(set(left_state).intersection(right_state))
        if not torch.equal(left_state[name], right_state[name])
    ]
    finite = all(bool(torch.isfinite(value).all()) for value in left_state.values())
    finite = finite and all(
        bool(torch.isfinite(value).all()) for value in right_state.values()
    )
    return {
        "left_key_count": len(left_keys),
        "right_key_count": len(right_keys),
        "same_inventory": left_keys == right_keys,
        "differing_keys": differing,
        "bit_exact": left_keys == right_keys and not differing,
        "finite": finite,
        "left_parameter_count": sum(p.numel() for p in left.parameters()),
        "right_parameter_count": sum(p.numel() for p in right.parameters()),
    }


def _common_state_summary(left: nn.Module, right: nn.Module) -> Dict[str, object]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    common = sorted(set(left_state).intersection(right_state))
    differing = [
        name for name in common if not torch.equal(left_state[name], right_state[name])
    ]
    return {
        "common_key_count": len(common),
        "differing_common_keys": differing,
        "right_only_keys": sorted(set(right_state).difference(left_state)),
        "all_common_bit_exact": not differing,
    }


def _tensor_tree_comparison(left: object, right: object) -> Dict[str, object]:
    maximum_error = 0.0
    mismatches: list[str] = []

    def visit(a: object, b: object, path: str) -> None:
        nonlocal maximum_error
        if torch.is_tensor(a) and torch.is_tensor(b):
            if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
                mismatches.append(path)
                return
            if a.is_floating_point() or a.is_complex():
                error = float((a.detach().float() - b.detach().float()).abs().amax().item())
                maximum_error = max(maximum_error, error)
                if error != 0.0:
                    mismatches.append(path)
            elif not torch.equal(a, b):
                mismatches.append(path)
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                mismatches.append(f"{path}.__keys__")
            for key in sorted(set(a).intersection(b), key=str):
                visit(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(f"{path}.__len__")
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f"{path}[{index}]")
            return
        if a != b:
            mismatches.append(path)

    visit(left, right, "root")
    return {
        "maximum_absolute_error": maximum_error,
        "mismatches": mismatches,
        "bit_exact": not mismatches,
    }


def _load_official_cropr(path: Path):
    spec = importlib.util.spec_from_file_location("locked_cvpr2025_cropr_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked official Cropr source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Cropr


def _copy_candidate_to_official(
    candidate: CroprTokenSelector,
    official: nn.Module,
) -> None:
    with torch.no_grad():
        official.cross_attn.queries.copy_(candidate.query)
        official.cross_attn.mlp_norm.load_state_dict(candidate.mlp_norm.state_dict())
        official.cross_attn.mlp.fc1.load_state_dict(candidate.mlp[0].state_dict())
        official.cross_attn.mlp.fc2.load_state_dict(candidate.mlp[2].state_dict())
        official.head.norm.load_state_dict(candidate.head_norm.state_dict())
        official.head.head.load_state_dict(candidate.head.state_dict())


def _official_equation_replay(official_path: Path) -> Dict[str, object]:
    official_class = _load_official_cropr(official_path)
    torch.manual_seed(41)
    candidate = CroprTokenSelector(dim=16, num_classes=5, mlp_ratio=4.0)
    official = official_class(
        pruning_rate=3,
        num_queries=1,
        num_classes=5,
        embed_dim=16,
        num_heads=1,
        pre_attn_norm=False,
        q_proj=False,
        k_proj=False,
        v_proj=False,
        mlp=True,
        mlp_ratio=4.0,
        training=True,
    )
    _copy_candidate_to_official(candidate, official)
    torch.manual_seed(19)
    candidate_input = torch.randn(3, 9, 16, requires_grad=True)
    official_input = candidate_input.detach().clone().requires_grad_(True)
    scores, logits, trace = candidate(
        candidate_input,
        collect_auxiliary=True,
        return_trace=True,
    )
    official_aggregate, official_scores = official.cross_attn(official_input.detach())
    official_logits = official.head(official_aggregate)
    if logits is None:
        raise RuntimeError("Candidate Cropr auxiliary logits are absent.")
    official_attention = (official_scores * (16**-0.5)).softmax(dim=-1)
    selected_equal = torch.equal(
        torch.topk(scores, k=6, dim=1).indices,
        torch.topk(official_scores, k=6, dim=1).indices,
    )
    targets = torch.tensor([0, 2, 4])
    F.cross_entropy(logits.float(), targets).backward()
    F.cross_entropy(official_logits.float(), targets).backward()
    mappings = {
        "query": "cross_attn.queries",
        "mlp_norm.weight": "cross_attn.mlp_norm.weight",
        "mlp_norm.bias": "cross_attn.mlp_norm.bias",
        "mlp.0.weight": "cross_attn.mlp.fc1.weight",
        "mlp.0.bias": "cross_attn.mlp.fc1.bias",
        "mlp.2.weight": "cross_attn.mlp.fc2.weight",
        "mlp.2.bias": "cross_attn.mlp.fc2.bias",
        "head_norm.weight": "head.norm.weight",
        "head_norm.bias": "head.norm.bias",
        "head.weight": "head.head.weight",
        "head.bias": "head.head.bias",
    }
    candidate_parameters = dict(candidate.named_parameters())
    official_parameters = dict(official.named_parameters())
    gradient_errors: Dict[str, float] = {}
    gradients_present = True
    for candidate_name, official_name in mappings.items():
        observed = candidate_parameters[candidate_name].grad
        expected = official_parameters[official_name].grad
        if observed is None or expected is None:
            gradients_present = False
            gradient_errors[candidate_name] = math.inf
        else:
            gradient_errors[candidate_name] = float(
                (observed - expected).abs().amax().item()
            )
    errors = {
        "raw_scores": float((scores - official_scores).abs().amax().item()),
        "scaled_attention": float(
            (trace["attention"] - official_attention).abs().amax().item()
        ),
        "residual_mlp_output": float(
            (trace["aggregated"] - official_aggregate).abs().amax().item()
        ),
        "auxiliary_logits": float((logits - official_logits).abs().amax().item()),
        "parameter_gradients": max(gradient_errors.values()),
    }
    passed = bool(
        gradients_present
        and selected_equal
        and candidate_input.grad is None
        and official_input.grad is None
        and max(errors.values()) <= 1e-6
    )
    return {
        "errors": errors,
        "gradient_errors": gradient_errors,
        "selected_indices_equal": selected_equal,
        "candidate_input_gradient_absent": candidate_input.grad is None,
        "official_input_gradient_absent": official_input.grad is None,
        "passed": passed,
    }


def _direct_equation_replay() -> Dict[str, object]:
    torch.manual_seed(7)
    selector = CroprTokenSelector(dim=12, num_classes=5, mlp_ratio=4.0)
    patches = torch.randn(4, 7, 12, requires_grad=True)
    scores, logits, trace = selector(
        patches,
        collect_auxiliary=True,
        return_trace=True,
    )
    if logits is None:
        raise RuntimeError("Direct replay lacks auxiliary logits.")
    clones = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in selector.named_parameters()
    }
    values = patches.detach()
    manual_scores = (
        clones["query"].expand(4, -1, -1) @ values.transpose(1, 2)
    ).squeeze(1)
    manual_attention = (manual_scores * (12**-0.5)).softmax(dim=-1)
    manual_aggregate = (manual_attention.unsqueeze(1) @ values).squeeze(1)
    normalized = F.layer_norm(
        manual_aggregate,
        (12,),
        clones["mlp_norm.weight"],
        clones["mlp_norm.bias"],
        1e-6,
    )
    hidden = F.gelu(
        F.linear(normalized, clones["mlp.0.weight"], clones["mlp.0.bias"])
    )
    manual_aggregate = manual_aggregate + F.linear(
        hidden,
        clones["mlp.2.weight"],
        clones["mlp.2.bias"],
    )
    normalized_head = F.layer_norm(
        manual_aggregate,
        (12,),
        clones["head_norm.weight"],
        clones["head_norm.bias"],
        1e-5,
    )
    manual_logits = F.linear(
        normalized_head,
        clones["head.weight"],
        clones["head.bias"],
    )
    targets = torch.tensor([0, 1, 3, 4])
    F.cross_entropy(logits.float(), targets).backward()
    F.cross_entropy(manual_logits.float(), targets).backward()
    gradient_errors = {
        name: float((parameter.grad - clones[name].grad).abs().amax().item())
        for name, parameter in selector.named_parameters()
        if parameter.grad is not None and clones[name].grad is not None
    }
    all_gradients = len(gradient_errors) == len(dict(selector.named_parameters()))
    errors = {
        "raw_scores": float((scores - manual_scores).abs().amax().item()),
        "scaled_attention": float(
            (trace["attention"] - manual_attention).abs().amax().item()
        ),
        "residual_mlp_output": float(
            (trace["aggregated"] - manual_aggregate).abs().amax().item()
        ),
        "auxiliary_logits": float((logits - manual_logits).abs().amax().item()),
        "parameter_gradients": max(gradient_errors.values()),
    }

    score_input = patches.detach().clone().requires_grad_(True)
    manual_input = patches.detach().clone().requires_grad_(True)
    query_clone = selector.query.detach().clone().requires_grad_(True)
    scorer_output = selector.forward_scorer(score_input)
    manual_output = (
        query_clone.expand(4, -1, -1) @ manual_input.transpose(1, 2)
    ).squeeze(1)
    scorer_output.square().mean().backward()
    manual_output.square().mean().backward()
    score_input_gradient_error = float(
        (score_input.grad - manual_input.grad).abs().amax().item()
    )
    passed = bool(
        all_gradients
        and patches.grad is None
        and max(errors.values()) <= 1e-6
        and score_input_gradient_error <= 1e-6
    )
    return {
        "errors": errors,
        "gradient_errors": gradient_errors,
        "scorer_input_gradient_error": score_input_gradient_error,
        "detached_full_input_gradient_absent": patches.grad is None,
        "passed": passed,
    }


def _selector_gradient_summary(model: nn.Module) -> Dict[str, object]:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("cropr_token_selectors.")
    }
    missing = [name for name, parameter in selected.items() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in selected.items()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    zero = [
        name
        for name, parameter in selected.items()
        if parameter.grad is not None
        and int(torch.count_nonzero(parameter.grad).item()) == 0
    ]
    norms = {
        name: float(parameter.grad.detach().float().norm().item())
        for name, parameter in selected.items()
        if parameter.grad is not None
    }
    return {
        "parameter_tensors": len(selected),
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
        "zero_gradients": zero,
        "gradient_norms": norms,
        "passed": bool(selected and not missing and not nonfinite and not zero),
    }


def _detach_check() -> Dict[str, object]:
    torch.manual_seed(23)
    selector = CroprTokenSelector(dim=16, num_classes=5)
    patches = torch.randn(6, 11, 16, requires_grad=True)
    _, logits, _ = selector(patches, collect_auxiliary=True)
    if logits is None:
        raise RuntimeError("Detach check lacks auxiliary logits.")
    F.cross_entropy(logits.float(), torch.tensor([0, 1, 2, 3, 4, 0])).backward()
    gradients = _selector_gradient_summary(
        nn.ModuleDict({"cropr_token_selectors": nn.ModuleDict({"x": selector})})
    )
    return {
        "input_gradient_absent": patches.grad is None,
        "selector_gradients": gradients,
        "passed": bool(patches.grad is None and gradients["passed"]),
    }


def _synthetic_learnability() -> Dict[str, object]:
    torch.manual_seed(101)
    batch_size, token_count, dim = 160, 8, 16
    labels = torch.arange(batch_size) % 5
    relevant = torch.arange(batch_size) % token_count
    patches = 0.03 * torch.randn(batch_size, token_count, dim)
    patches[:, :, 0] -= 1.0
    rows = torch.arange(batch_size)
    patches[rows, relevant, 0] = 5.0
    patches[rows, relevant, 1 + labels] = 3.0
    selector = CroprTokenSelector(dim=dim, num_classes=5)
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.03)
    with torch.no_grad():
        _, initial_logits, _ = selector(patches, collect_auxiliary=True)
        if initial_logits is None:
            raise RuntimeError("Synthetic initial logits are absent.")
        initial_loss = float(F.cross_entropy(initial_logits.float(), labels).item())
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        _, logits, _ = selector(patches, collect_auxiliary=True)
        if logits is None:
            raise RuntimeError("Synthetic training logits are absent.")
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores, final_logits, _ = selector(patches, collect_auxiliary=True)
        if final_logits is None:
            raise RuntimeError("Synthetic final logits are absent.")
        final_loss = float(F.cross_entropy(final_logits.float(), labels).item())
        hit_rate = float((scores.argmax(dim=1) == relevant).float().mean().item())
    reduction = 1.0 - final_loss / max(initial_loss, 1e-12)
    return {
        "repository_data_used": False,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_fraction": reduction,
        "relevant_token_top1_hit_rate": hit_rate,
        "passed": bool(reduction >= 0.50 and hit_rate >= 0.95),
    }


def _metadata_to_device(
    metadata: Mapping[str, Tensor], device: torch.device, count: Optional[int] = None
) -> Dict[str, Tensor]:
    return {
        key: value[:count].to(device=device, non_blocking=True)
        for key, value in metadata.items()
    }


def _default_off_parity(
    source_config: Mapping[str, object],
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    implicit_config = dict(source_config)
    implicit_config.pop("cropr_token_selector", None)
    implicit_config.pop("cropr_token_selector_routing", None)
    explicit_config = dict(implicit_config)
    explicit_config.update(
        {"cropr_token_selector": False, "cropr_token_selector_routing": True}
    )
    set_seed(seed, deterministic=True)
    implicit = create_model(num_classes=5, model_config=implicit_config)
    implicit_rng = _rng_snapshot()
    set_seed(seed, deterministic=True)
    explicit = create_model(num_classes=5, model_config=explicit_config)
    explicit_rng = _rng_snapshot()
    state = _state_summary(implicit, explicit)
    implicit = implicit.to(device).eval()
    explicit = explicit.to(device).eval()
    with torch.inference_mode():
        implicit_logits, implicit_features = _forward(
            implicit, images, metadata, return_trace=True
        )
        explicit_logits, explicit_features = _forward(
            explicit, images, metadata, return_trace=True
        )
    logits_error = float((implicit_logits - explicit_logits).abs().amax().item())
    trace = _tensor_tree_comparison(
        implicit_features["trace"], explicit_features["trace"]
    )
    public_shapes_equal = all(
        tuple(implicit_features[key].shape) == tuple(explicit_features[key].shape)
        for key in ("tokens", "patches", "patch_indices", "pooled")
    )
    result = {
        "state": state,
        "rng_equal": _rng_equal(implicit_rng, explicit_rng),
        "logit_maximum_absolute_error": logits_error,
        "argmax_equal": torch.equal(
            implicit_logits.argmax(dim=1), explicit_logits.argmax(dim=1)
        ),
        "trace": trace,
        "selected_indices_equal": torch.equal(
            implicit_features["patch_indices"], explicit_features["patch_indices"]
        ),
        "public_shapes_equal": public_shapes_equal,
    }
    result["passed"] = bool(
        state["bit_exact"]
        and result["rng_equal"]
        and logits_error == 0.0
        and result["argmax_equal"]
        and trace["bit_exact"]
        and result["selected_indices_equal"]
        and public_shapes_equal
    )
    del implicit, explicit
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _control_native_parity(
    source_config: Mapping[str, object],
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    native_config = _selector_config(source_config, enabled=False, routing=False)
    control_config = _selector_config(source_config, enabled=True, routing=False)
    set_seed(seed, deterministic=True)
    native = create_model(num_classes=5, model_config=native_config)
    set_seed(seed, deterministic=True)
    control = create_model(num_classes=5, model_config=control_config)
    common_state = _common_state_summary(native, control)
    native = native.to(device).eval()
    control = control.to(device).eval()
    with torch.inference_mode():
        native_logits, native_features = _forward(
            native, images, metadata, return_trace=True
        )
        control_logits, control_features = _forward(
            control, images, metadata, return_trace=True
        )
    stage_equal = []
    for native_stage, control_stage in zip(
        native_features["trace"]["pruning"],
        control_features["trace"]["pruning"],
    ):
        stage_equal.append(
            bool(
                torch.equal(
                    native_stage["kept_indices"], control_stage["kept_indices"]
                )
                and torch.equal(native_stage["scores"], control_stage["scores"])
                and not bool(control_stage["cropr_routing"].item())
            )
        )
    result = {
        "common_state": common_state,
        "logit_maximum_absolute_error": float(
            (native_logits - control_logits).abs().amax().item()
        ),
        "argmax_equal": torch.equal(
            native_logits.argmax(dim=1), control_logits.argmax(dim=1)
        ),
        "selected_indices_equal": torch.equal(
            native_features["patch_indices"], control_features["patch_indices"]
        ),
        "stage_equal": stage_equal,
        "auxiliary_layers": sorted(
            int(value) for value in control_features["cropr_auxiliary_logits"]
        ),
    }
    result["passed"] = bool(
        common_state["all_common_bit_exact"]
        and all(
            name.startswith("cropr_token_selectors.")
            for name in common_state["right_only_keys"]
        )
        and result["logit_maximum_absolute_error"] == 0.0
        and result["argmax_equal"]
        and result["selected_indices_equal"]
        and all(stage_equal)
        and result["auxiliary_layers"] == [2, 5]
    )
    del native, control
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _partition_summary(features: Mapping[str, object]) -> Dict[str, object]:
    previous = torch.arange(256, device=features["patch_indices"].device)
    previous = previous.unsqueeze(0).expand(int(features["patch_indices"].size(0)), -1)
    expected_counts = [218, 167]
    stages: list[Dict[str, object]] = []
    all_valid = True
    for stage, expected_count in zip(features["trace"]["pruning"], expected_counts):
        kept = stage["kept_indices"]
        dropped = stage["dropped_indices"]
        sorted_spatially = torch.equal(kept, kept.sort(dim=1).values)
        row_valid = []
        for row in range(int(kept.size(0))):
            kept_set = set(int(value) for value in kept[row].tolist())
            dropped_set = set(int(value) for value in dropped[row].tolist())
            previous_set = set(int(value) for value in previous[row].tolist())
            row_valid.append(
                not bool(kept_set.intersection(dropped_set))
                and kept_set.union(dropped_set) == previous_set
            )
        valid = bool(
            int(kept.size(1)) == expected_count
            and sorted_spatially
            and all(row_valid)
            and bool(stage["cropr_routing"].item())
        )
        all_valid = all_valid and valid
        stages.append(
            {
                "layer": int(stage["layer"].item()),
                "before_count": int(stage["before_count"].item()),
                "after_count": int(stage["after_count"].item()),
                "dropped_count": int(dropped.size(1)),
                "spatially_sorted": sorted_spatially,
                "all_rows_complete_disjoint": all(row_valid),
                "raw_score_shape": list(stage["raw_scores"].shape),
                "attention_shape": list(stage["attention"].shape),
                "native_score_shape": list(stage["native_scores"].shape),
                "normalized_entropy_mean": float(
                    stage["normalized_entropy"].float().mean().item()
                ),
                "score_variance_mean": float(
                    stage["score_variance"].float().mean().item()
                ),
                "valid": valid,
            }
        )
        previous = kept
    return {"stages": stages, "passed": all_valid}


def _forward_parity(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    bf16: bool,
) -> Tuple[Dict[str, object], Mapping[str, object]]:
    model.eval()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with torch.inference_mode(), context:
        standard_logits, standard_features = _forward(model, images, metadata)
        trace_logits, trace_features = _forward(
            model, images, metadata, return_trace=True
        )
    result = {
        "maximum_logit_absolute_error": float(
            (standard_logits.float() - trace_logits.float()).abs().amax().item()
        ),
        "argmax_mismatch_count": int(
            (standard_logits.argmax(dim=1) != trace_logits.argmax(dim=1)).sum().item()
        ),
        "selected_indices_equal": torch.equal(
            standard_features["patch_indices"], trace_features["patch_indices"]
        ),
        "finite": bool(
            torch.isfinite(standard_logits).all() and torch.isfinite(trace_logits).all()
        ),
    }
    result["passed"] = bool(
        result["maximum_logit_absolute_error"] <= 1e-6
        and result["argmax_mismatch_count"] == 0
        and result["selected_indices_equal"]
        and result["finite"]
    )
    return result, trace_features


def _mean_stage_jaccard(
    fp32_features: Mapping[str, object], bf16_features: Mapping[str, object]
) -> Dict[str, object]:
    stage_values = []
    for fp32_stage, bf16_stage in zip(
        fp32_features["trace"]["pruning"],
        bf16_features["trace"]["pruning"],
    ):
        left = fp32_stage["kept_indices"]
        right = bf16_stage["kept_indices"]
        rows = []
        for row in range(int(left.size(0))):
            left_set = set(int(value) for value in left[row].tolist())
            right_set = set(int(value) for value in right[row].tolist())
            rows.append(len(left_set.intersection(right_set)) / len(left_set.union(right_set)))
        stage_values.append(
            {"layer": int(fp32_stage["layer"].item()), "mean_jaccard": float(sum(rows) / len(rows))}
        )
    minimum = min(value["mean_jaccard"] for value in stage_values)
    return {"stages": stage_values, "minimum": minimum, "passed": minimum >= 0.98}


def _backward_check(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    bf16: bool,
    seed: int,
) -> Dict[str, object]:
    set_seed(seed, deterministic=True)
    model.train()
    model.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        logits, features = _forward(model, images, metadata)
        auxiliary_loss, auxiliary_stats = _cropr_auxiliary_loss_from_features(
            features=features, targets=labels
        )
        loss = F.cross_entropy(logits.float(), labels) + auxiliary_loss
    loss.backward()
    gradients = _selector_gradient_summary(model)
    result = {
        "loss": float(loss.detach().item()),
        "auxiliary_loss": float(auxiliary_loss.detach().item()),
        "auxiliary_stats": auxiliary_stats,
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss).all()),
        "selector_gradients": gradients,
    }
    result["passed"] = bool(
        result["logits_finite"] and result["loss_finite"] and gradients["passed"]
    )
    model.zero_grad(set_to_none=True)
    return result


def _timed_resource(
    *,
    config: Mapping[str, object],
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    training: bool,
    seed: int,
    warmup: int,
    iterations: int,
) -> Dict[str, object]:
    set_seed(seed, deterministic=True)
    model = create_model(num_classes=5, model_config=config).to(images.device)
    model.train(training)

    def step() -> float:
        if training:
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, features = _forward(model, images, metadata)
                auxiliary_loss, _ = _cropr_auxiliary_loss_from_features(
                    features=features, targets=labels
                )
                loss = F.cross_entropy(logits.float(), labels) + auxiliary_loss
            loss.backward()
            return float(loss.detach().item())
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits, _ = _forward(model, images, metadata)
        return float(logits.float().abs().mean().item())

    for _ in range(max(0, warmup)):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    values = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        values.append(step())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    result = {
        "mode": "training" if training else "inference",
        "batch_size": int(images.size(0)),
        "median_seconds": float(statistics.median(times)),
        "iterations": len(times),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "finite": bool(all(math.isfinite(value) for value in values + times)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


class _CroprExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _onnx_audit(
    *,
    config: Mapping[str, object],
    images: Tensor,
    metadata: Mapping[str, Tensor],
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    import onnx

    set_seed(seed, deterministic=True)
    model = create_model(num_classes=5, model_config=config).cpu().eval()
    export_images = images[:1].detach().float().cpu()
    bbox = metadata.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 8)
    bbox = bbox[:1].detach().float().cpu()
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(1, 256, 256, dtype=torch.bool)
    image_mask = image_mask[:1].detach().bool().cpu()
    path = output_dir / "cropr_token_selector_candidate.onnx"
    try:
        result = _onnx_compare(
            wrapper=_CroprExport(model),
            inputs=(export_images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=path,
        )
        graph = onnx.load(str(path)).graph
        op_types = [node.op_type for node in graph.node]
        graph_names = [initializer.name.lower() for initializer in graph.initializer]
        graph_names.extend(node.name.lower() for node in graph.node if node.name)
        forbidden = [
            name
            for name in graph_names
            if "cropr_token_selectors" in name
            and any(marker in name for marker in ("mlp", "head_norm", ".head"))
        ]
        result.update(
            {
                "op_type_counts": {
                    name: op_types.count(name) for name in sorted(set(op_types))
                },
                "topk_present": "TopK" in op_types,
                "gather_present": any(name.startswith("Gather") for name in op_types),
                "forbidden_auxiliary_graph_names": forbidden,
                "auxiliary_path_absent": not forbidden,
            }
        )
    except Exception as error:
        result = _failed_export(path, error)
        result.update(
            {
                "op_type_counts": {},
                "topk_present": False,
                "gather_present": False,
                "forbidden_auxiliary_graph_names": [],
                "auxiliary_path_absent": False,
            }
        )
    result["passed"] = bool(
        result.get("succeeded", False)
        and result.get("finite", False)
        and float(result.get("maximum_absolute_error", math.inf)) <= 1e-4
        and result.get("argmax_match", False)
        and result["topk_present"]
        and result["gather_present"]
        and result["auxiliary_path_absent"]
    )
    return result


def _architecture_trace(
    *,
    dataset: object,
    model: nn.Module,
    device: torch.device,
    num_workers: int,
    output_dir: Path,
) -> Dict[str, object]:
    selected: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        label = int(sample.primary_label)
        if label not in selected:
            selected[label] = index
        if len(selected) == 5:
            break
    if sorted(selected) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Could not select one fit example per class.")
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=True,
        context="cropr_architecture_trace",
        persistent_workers=False,
    )
    loader = DataLoader(
        Subset(dataset, [selected[label] for label in range(5)]),
        batch_size=5,
        shuffle=False,
        **loader_kwargs,
    )
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(next(iter(loader)))
    images = images_cpu.to(device)
    metadata = _metadata_to_device(metadata_cpu, device)
    model.eval()
    with torch.inference_mode():
        logits, features = _forward(model, images, metadata, return_trace=True)
    partitions = _partition_summary(features)
    samples = []
    for label in range(5):
        sample = dataset.samples[selected[label]]
        samples.append(
            {
                "dataset_index": selected[label],
                "label": label,
                "image_name": sample.image_path.name,
                "object_index": int(sample.primary_object_index),
            }
        )
    result = {
        "samples": samples,
        "observed_labels": [int(value) for value in labels_cpu.tolist()],
        "predictions": [int(value) for value in logits.argmax(dim=1).cpu().tolist()],
        "num_prefix_tokens": int(model.num_prefix_tokens),
        "public_token_shape": list(features["tokens"].shape),
        "public_patch_shape": list(features["patches"].shape),
        "block_token_shapes": [
            list(shape) for shape in features["trace"]["block_token_shapes"]
        ],
        "partitions": partitions,
        "loader": loader_summary,
    }
    result["passed"] = bool(
        result["observed_labels"] == [0, 1, 2, 3, 4]
        and result["num_prefix_tokens"] == 7
        and [stage["after_count"] for stage in partitions["stages"]] == [218, 167]
        and partitions["passed"]
        and int(features["patch_indices"].size(1)) == 167
    )
    path = output_dir / "architecture_trace.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    result["path"] = str(path.resolve())
    result["sha256"] = _sha256(path)
    return result


def _write_manifest(output_dir: Path) -> Path:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    manifest = output_dir / "artifact_manifest.json"
    manifest.write_text(
        json.dumps({"files": rows}, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Cropr token-selector preflight requires CUDA.")
    if (
        int(args.seed) != 42
        or int(args.batch_size) != 32
        or int(args.fp32_batch_size) != 2
        or int(args.warmup_iterations) != 2
        or int(args.timed_iterations) != 5
    ):
        raise ValueError(
            "Cropr preflight is locked to seed42, batch32, FP32 batch2, "
            "warmup2, timed5."
        )
    output_was_absent = not Path(args.output_dir).resolve().exists()
    output_dir = _prepare_output(Path(args.output_dir))
    os.environ.setdefault("TRKH_ALLOW_WINDOWS_MULTIPROCESSING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    official_root = Path(args.official_root).resolve()
    paths = {
        "protocol": Path(args.protocol).resolve(),
        "resolved_config": Path(args.resolved_config).resolve(),
        "fit_only_data": Path(args.fit_only_data).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "declaration_summary": Path(args.declaration_summary).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
        "v8_launcher": Path(
            "scripts/run_trkh_5class_attention_views_v8.ps1"
        ).resolve(),
        "selector_runtime": Path("trkh/models/cropr_token_selector.py").resolve(),
        "model_runtime": Path("trkh/models/model.py").resolve(),
        "train_runtime": Path("trkh/training/train.py").resolve(),
        "config_runtime": Path("trkh/core/config.py").resolve(),
        "selector_tests": Path("tests/test_cropr_token_selector.py").resolve(),
        "official_cropr": official_root / "cls" / "cropr.py",
        "official_vit": official_root / "cls" / "vision_transformer.py",
        "official_engine": official_root / "cls" / "engine.py",
        "official_recipe": official_root / "cls" / "CLASSIFICATION.md",
        "official_license": official_root / "LICENSE",
        "paper": Path(args.paper).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Locked Cropr inputs are missing: {missing}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    hash_checks = {
        name: hashes.get(name) == expected for name, expected in LOCKED_HASHES.items()
    }
    resolved_config = _load_json(paths["resolved_config"])
    source_model_config = resolved_config.get("model_config")
    if not isinstance(source_model_config, Mapping):
        raise ValueError("Resolved config lacks model_config.")
    fold_summary = _load_json(paths["fold_summary"])
    declaration_rows = _read_clean_train_rows(paths["declaration"])
    fit_rows = [row for row in declaration_rows if int(row.fold) != 0]
    holdout_rows = [row for row in declaration_rows if int(row.fold) == 0]
    fit_counts = [sum(int(row.target) == label for row in fit_rows) for label in range(5)]
    holdout_counts = [
        sum(int(row.target) == label for row in holdout_rows) for label in range(5)
    ]
    fit_sources = len({row.source_stem for row in fit_rows})
    holdout_sources = len({row.source_stem for row in holdout_rows})
    fold_checks = {
        "fit_rows": len(fit_rows) == EXPECTED_FIT_ROWS,
        "holdout_rows": len(holdout_rows) == EXPECTED_HOLDOUT_ROWS,
        "fit_sources": fit_sources == EXPECTED_FIT_SOURCES,
        "holdout_sources": holdout_sources == EXPECTED_HOLDOUT_SOURCES,
        "source_overlap_zero": not bool(
            {row.source_stem for row in fit_rows}.intersection(
                {row.source_stem for row in holdout_rows}
            )
        ),
        "fit_counts": fit_counts == EXPECTED_FIT_COUNTS,
        "holdout_counts": holdout_counts == EXPECTED_HOLDOUT_COUNTS,
        "fit_index_sha256": _ordered_index_sha256(
            [row.sample_index for row in fit_rows]
        )
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_sha256": _ordered_index_sha256(
            [row.sample_index for row in holdout_rows]
        )
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "summary_source_overlap_zero": int(fold_summary.get("source_overlap", -1)) == 0,
        "test_mirror_not_used": bool(fold_summary.get("test_mirrors_holdout", False)),
    }

    head = _git_value("rev-parse", "HEAD")
    upstream = _git_value("rev-parse", "@{upstream}")
    full_status = _git_value("status", "--short")
    status_lines = [line for line in full_status.splitlines() if line.strip()]
    protected_scope = all(
        any(
            marker in line
            for marker in (
                "BaoCao/",
                "deep-research-report (9).md",
                "deep-research-report (10).md",
            )
        )
        for line in status_lines
    )
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    official_tree = subprocess.check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD^{tree}"],
        text=True,
        encoding="utf-8",
    ).strip()
    official_status = subprocess.check_output(
        ["git", "-C", str(official_root), "status", "--short"],
        text=True,
        encoding="utf-8",
    ).strip()

    fit_dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=_selector_config(source_model_config, enabled=True, routing=True),
        resolved_config=resolved_config,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="cropr_preflight_fit_only",
        persistent_workers=False,
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(next(iter(fit_loader)))
    device = torch.device("cuda")
    images = images_cpu.to(device=device, non_blocking=True)
    labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = _metadata_to_device(metadata_cpu, device)

    official_equation = _official_equation_replay(paths["official_cropr"])
    direct_equation = _direct_equation_replay()
    detach = _detach_check()
    synthetic = _synthetic_learnability()
    small_images = images[: int(args.fp32_batch_size)]
    small_metadata = {
        key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()
    }
    default_off = _default_off_parity(
        source_model_config,
        small_images,
        small_metadata,
        device=device,
        seed=int(args.seed),
    )
    control_native = _control_native_parity(
        source_model_config,
        small_images,
        small_metadata,
        device=device,
        seed=int(args.seed),
    )

    control_config = _selector_config(source_model_config, enabled=True, routing=False)
    candidate_config = _selector_config(source_model_config, enabled=True, routing=True)
    set_seed(int(args.seed), deterministic=True)
    scratch_control = create_model(num_classes=5, model_config=control_config)
    control_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    scratch_candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = _rng_snapshot()
    matched_state = _state_summary(scratch_control, scratch_candidate)
    routing_state_keys = [
        name for name in scratch_candidate.state_dict() if "routing" in name.lower()
    ]
    scratch_control = scratch_control.to(device)
    scratch_candidate = scratch_candidate.to(device)
    fp32_backward = _backward_check(
        scratch_candidate,
        small_images,
        labels[: int(args.fp32_batch_size)],
        small_metadata,
        bf16=False,
        seed=int(args.seed),
    )
    bf16_backward = _backward_check(
        scratch_candidate,
        images,
        labels,
        metadata,
        bf16=True,
        seed=int(args.seed),
    )
    fp32_parity, fp32_features = _forward_parity(
        scratch_candidate, images, metadata, bf16=False
    )
    bf16_parity, bf16_features = _forward_parity(
        scratch_candidate, images, metadata, bf16=True
    )
    partitions = _partition_summary(fp32_features)
    precision_stability = _mean_stage_jaccard(fp32_features, bf16_features)
    architecture = _architecture_trace(
        dataset=fit_dataset,
        model=scratch_candidate,
        device=device,
        num_workers=int(args.num_workers),
        output_dir=output_dir,
    )
    del scratch_control, scratch_candidate
    gc.collect()
    torch.cuda.empty_cache()

    control_inference = _timed_resource(
        config=control_config,
        images=images,
        labels=labels,
        metadata=metadata,
        training=False,
        seed=int(args.seed),
        warmup=int(args.warmup_iterations),
        iterations=int(args.timed_iterations),
    )
    candidate_inference = _timed_resource(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        training=False,
        seed=int(args.seed),
        warmup=int(args.warmup_iterations),
        iterations=int(args.timed_iterations),
    )
    control_training = _timed_resource(
        config=control_config,
        images=images,
        labels=labels,
        metadata=metadata,
        training=True,
        seed=int(args.seed),
        warmup=int(args.warmup_iterations),
        iterations=int(args.timed_iterations),
    )
    candidate_training = _timed_resource(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        training=True,
        seed=int(args.seed),
        warmup=int(args.warmup_iterations),
        iterations=int(args.timed_iterations),
    )
    inference_ratio = float(
        candidate_inference["median_seconds"] / control_inference["median_seconds"]
    )
    train_ratio = float(
        candidate_training["median_seconds"] / control_training["median_seconds"]
    )
    inference_vram_ratio = float(
        candidate_inference["peak_vram_gib"]
        / max(control_inference["peak_vram_gib"], 1e-12)
    )
    train_vram_ratio = float(
        candidate_training["peak_vram_gib"]
        / max(control_training["peak_vram_gib"], 1e-12)
    )
    onnx = _onnx_audit(
        config=candidate_config,
        images=images_cpu,
        metadata=metadata_cpu,
        output_dir=output_dir,
        seed=int(args.seed),
    )

    checks = {
        **{f"hash_{name}": value for name, value in hash_checks.items()},
        **{f"fold_{name}": value for name, value in fold_checks.items()},
        "git_tracked_clean": _tracked_worktree_clean(),
        "git_head_pushed": head == upstream,
        "worktree_scope_protected_only": protected_scope,
        "official_commit": official_commit == LOCKED_OFFICIAL_COMMIT,
        "official_tree": official_tree == LOCKED_OFFICIAL_TREE,
        "official_worktree_clean": not bool(official_status),
        "fit_dataset_rows": len(fit_dataset) == EXPECTED_FIT_ROWS,
        "official_equation": bool(official_equation["passed"]),
        "direct_equation": bool(direct_equation["passed"]),
        "detached_auxiliary": bool(detach["passed"]),
        "synthetic_learnability": bool(synthetic["passed"]),
        "default_off_bit_exact": bool(default_off["passed"]),
        "matched_role_state": bool(
            matched_state["bit_exact"]
            and matched_state["finite"]
            and matched_state["left_parameter_count"]
            == matched_state["right_parameter_count"]
        ),
        "matched_role_rng": _rng_equal(control_rng, candidate_rng),
        "routing_not_in_state": not routing_state_keys,
        "control_native_parity": bool(control_native["passed"]),
        "candidate_partitions": bool(partitions["passed"]),
        "fp32_standard_trace_parity": bool(fp32_parity["passed"]),
        "bf16_standard_trace_parity": bool(bf16_parity["passed"]),
        "bf16_fp32_selected_jaccard": bool(precision_stability["passed"]),
        "fp32_backward": bool(fp32_backward["passed"]),
        "bf16_backward": bool(bf16_backward["passed"]),
        "onnx": bool(onnx["passed"]),
        "inference_runtime": math.isfinite(inference_ratio)
        and inference_ratio <= MAX_INFERENCE_RUNTIME_RATIO,
        "training_runtime": math.isfinite(train_ratio)
        and train_ratio <= MAX_TRAIN_RUNTIME_RATIO,
        "inference_vram": bool(
            candidate_inference["peak_vram_gib"] <= MAX_VRAM_GIB
            and inference_vram_ratio <= MAX_VRAM_RATIO
        ),
        "training_vram": bool(
            candidate_training["peak_vram_gib"] <= MAX_VRAM_GIB
            and train_vram_ratio <= MAX_VRAM_RATIO
        ),
        "architecture_trace": bool(architecture["passed"]),
        "output_absent_before_run": output_was_absent,
        "validation_not_loaded": True,
        "test_not_loaded": True,
        "current_best_unchanged": bool(
            hash_checks["keeper_checkpoint"]
            and hash_checks["current_commands"]
            and hash_checks["command_history"]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": head,
            "upstream": upstream,
            "tracked_clean": _tracked_worktree_clean(),
            "full_status": status_lines,
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "official_repository": {
            "root": str(official_root),
            "commit": official_commit,
            "tree": official_tree,
            "clean": not bool(official_status),
            "license": "MIT",
        },
        "fold": {
            "summary": fold_summary,
            "checks": fold_checks,
            "fit_counts": fit_counts,
            "holdout_counts": holdout_counts,
            "fit_index_sha256": _ordered_index_sha256(
                [row.sample_index for row in fit_rows]
            ),
            "holdout_index_sha256": _ordered_index_sha256(
                [row.sample_index for row in holdout_rows]
            ),
            "dataset_rows": len(fit_dataset),
            "loader": loader_summary,
        },
        "official_equation": official_equation,
        "direct_equation": direct_equation,
        "detach": detach,
        "synthetic": synthetic,
        "default_off": default_off,
        "matched_roles": {
            "state": matched_state,
            "rng": {
                "control": _rng_summary(control_rng),
                "candidate": _rng_summary(candidate_rng),
                "equal": _rng_equal(control_rng, candidate_rng),
            },
            "routing_state_keys": routing_state_keys,
        },
        "control_native_parity": control_native,
        "candidate_partitions": partitions,
        "trace_parity": {"fp32": fp32_parity, "bf16": bf16_parity},
        "precision_stability": precision_stability,
        "backward": {"fp32": fp32_backward, "bf16": bf16_backward},
        "architecture": architecture,
        "resource": {
            "control_inference": control_inference,
            "candidate_inference": candidate_inference,
            "control_training": control_training,
            "candidate_training": candidate_training,
            "inference_runtime_ratio": inference_ratio,
            "training_runtime_ratio": train_ratio,
            "inference_vram_ratio": inference_vram_ratio,
            "training_vram_ratio": train_vram_ratio,
        },
        "onnx": onnx,
        "validation_used": False,
        "test_used": False,
        "gate": {
            "checks": checks,
            "failed_checks": failed,
            "automated_pass": not failed,
            "formal_pair_permission": not failed,
            "validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_preflight(args)
    print(
        json.dumps(
            {
                "method": summary["method"],
                "failed_checks": summary["gate"]["failed_checks"],
                "formal_pair_permission": summary["gate"]["formal_pair_permission"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not bool(summary["gate"]["formal_pair_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
