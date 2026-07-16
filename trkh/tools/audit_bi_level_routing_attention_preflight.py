from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.bi_level_routing_attention import BiLevelRoutingAttention
from trkh.models.model import (
    MultiHeadSelfAttention,
    create_model,
    load_model_state,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
    _rgb_from_tensor,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
)
from trkh.tools.audit_deformable_spatial_attention_preflight import (
    _inference_benchmark,
)
from trkh.tools.audit_foveal_aggregated_attention_pair import (
    CONDITIONS,
    _build_holdout_dataset,
    _condition_corruption,
    _make_loader,
    _metadata_to_device,
)
from trkh.tools.audit_foveal_aggregated_attention_preflight import (
    _FullExport,
    _benchmark,
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
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)


METHOD = "bi_level_routing_attention_a1_preflight"
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLD = 0
LOCKED_HASHES = {
    "protocol": "e7ce7cf3ba44361f29760ed2b1ed1f06538db50b44613ce03c1373a35cc3ede3",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "declaration_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "resolved_config": "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    "scratch_launcher": "dac9977b13249ffb71ffd74339d3c8980e504f2652788c45e677c5aeeca587fe",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "fold_data": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "dbb_closure": "c24367649e6786b3d05af9454691d610f6c01ea506c830da4a469c5973d31504",
    "official_attention": "5b7b35316bb3b2d3f8494785bafc381200285ecb08086b95d5a61f4a5eab388e",
    "official_gather": "93d291582f7eb087a6091e247053e6425bbaacc3603383187c2be3669ff22ff4",
    "official_architecture": "97a253c8e06bac251be79ec0e298796169f31c49fb99028f64ffe18551f34456",
    "official_recipe": "e583aaeccb1949ef55b96b11859be2df07ca5311ca02a381a20b644e3940b0d2",
    "official_license": "63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97",
    "paper": "d7415feb19a0818b39b9250b2a048aae2fb311079cfbabf14705362a8a353a5c",
    "current_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
LOCKED_OFFICIAL_BRANCH = "public_release"
LOCKED_OFFICIAL_COMMIT = "1697bbbeafb8680524898f1dcaac10defd0604be"
LOCKED_OFFICIAL_TREE = "313af0f24b31141cdde68fd775e75105278f8e52"
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_INDEX_SHA256 = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
EXPECTED_HOLDOUT_INDEX_SHA256 = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
MAX_ADDED_PARAMETERS = 7_000
MAX_INFERENCE_RUNTIME_RATIO = 1.35
MAX_TRAIN_RUNTIME_RATIO = 1.50
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.25


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only engineering and selectivity preflight for block-2 "
            "CVPR-2023 BiFormer Bi-Level Routing Attention."
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
        "--scratch-launcher",
        type=Path,
        default=Path("scripts/run_trkh_5class_surface_detail_v9.ps1"),
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
            "docs/TRKH_5CLASS_BILEVEL_ROUTING_ATTENTION_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--dbb-closure",
        type=Path,
        default=Path("docs/TRKH_5CLASS_DIVERSE_BRANCH_STEM_CLOSURE_20260716.md"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\biformer-cvpr2023"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\biformer-cvpr2023\BiFormer_CVPR2023_paper.pdf"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--selectivity-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    return parser.parse_args(argv)


def _bra_config(source: Mapping[str, object], *, topk: int) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "early_token_mask_keep_rate": 1.0,
            "gated_relative_position_attention": False,
            "visual_contrast_attention": False,
            "foveal_aggregated_attention": False,
            "deformable_spatial_attention": False,
            "cross_covariance_attention": False,
            "dynamic_graph_mixer": False,
            "soft_moe_patch_adapter": False,
            "deep_class_prompt": False,
            "layer_token_fusion": False,
            "patch_style_recalibration": False,
            "block_local_patch_mixer": False,
            "locally_enhanced_ffn": False,
            "concurrent_local_global_coupling": False,
            "bi_level_routing_attention": True,
            "bi_level_routing_attention_layers": "2",
            "bi_level_routing_attention_regions_per_axis": 4,
            "bi_level_routing_attention_topk": int(topk),
            "bi_level_routing_attention_local_context_kernel_size": 5,
        }
    )
    return config


def _baseline_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _bra_config(source, topk=4)
    config["bi_level_routing_attention"] = False
    return config


def _rng_sha256(state: Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_snapshot() -> Dict[str, object]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _rng_summary(state: Mapping[str, object]) -> Dict[str, object]:
    return {
        "cpu_sha256": _rng_sha256(state["cpu"]),
        "cuda_sha256": [_rng_sha256(value) for value in state["cuda"]],
    }


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_cuda = left["cuda"]
    right_cuda = right["cuda"]
    return bool(
        torch.equal(left["cpu"], right["cpu"])
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def _repository_state(root: Path) -> Dict[str, str]:
    def value(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "branch": value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": value("rev-parse", "HEAD"),
        "tree": value("rev-parse", "HEAD^{tree}"),
    }


def _state_common_summary(left: nn.Module, right: nn.Module) -> Dict[str, object]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    left_keys = set(left_state)
    right_keys = set(right_state)
    common = sorted(left_keys.intersection(right_keys))
    differing = [key for key in common if not torch.equal(left_state[key], right_state[key])]
    return {
        "left_keys": len(left_keys),
        "right_keys": len(right_keys),
        "inventory_equal": left_keys == right_keys,
        "common_keys": len(common),
        "differing_common_keys": differing,
        "all_common_bit_exact": not differing,
        "left_only": sorted(left_keys.difference(right_keys)),
        "right_only": sorted(right_keys.difference(left_keys)),
    }


def _copy_candidate_to_official(
    candidate: BiLevelRoutingAttention, official: nn.Module
) -> None:
    with torch.no_grad():
        official.qkv_linear.weight.copy_(candidate.qkv.weight[:, :, None, None])
        official.qkv_linear.bias.copy_(candidate.qkv.bias)
        official.output_linear.weight.copy_(candidate.proj.weight[:, :, None, None])
        official.output_linear.bias.copy_(candidate.proj.bias)
        official.lepe.weight.copy_(candidate.local_context.weight)
        official.lepe.bias.copy_(candidate.local_context.bias)


def _official_equation_replay(official_root: Path) -> Dict[str, object]:
    root = str(Path(official_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ops.bra_nchw import nchwBRA

    torch.manual_seed(41)
    candidate = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=2,
        topk=2,
        local_context_kernel_size=3,
    ).eval()
    official = nchwBRA(
        dim=32,
        num_heads=4,
        n_win=2,
        qk_scale=candidate.scale,
        topk=2,
        side_dwconv=3,
    ).eval()
    _copy_candidate_to_official(candidate, official)

    torch.manual_seed(19)
    official_input = torch.randn(2, 32, 4, 4, requires_grad=True)
    candidate_input = (
        official_input.detach().flatten(2).transpose(1, 2).clone().requires_grad_(True)
    )
    official_output, official_attention = official(
        official_input, ret_attn_mask=True
    )
    candidate_output, _ = candidate(
        candidate_input,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=0,
        patch_indices=torch.arange(16).unsqueeze(0).expand(2, -1),
        collect_trace=True,
    )
    candidate_map = candidate_output.transpose(1, 2).reshape_as(official_output)
    trace = candidate.trace()

    with torch.no_grad():
        projected = official.qkv_linear(official_input)
        query, key, _ = projected.chunk(3, dim=1)
        query_region = F.avg_pool2d(query.detach(), kernel_size=(2, 2))
        key_region = F.avg_pool2d(key.detach(), kernel_size=(2, 2))
        affinity = query_region.permute(0, 2, 3, 1).flatten(1, 2) @ key_region.flatten(2)
        official_routes = torch.topk(affinity, k=2, dim=-1).indices

    candidate_map.square().mean().backward()
    official_output.square().mean().backward()
    parameter_mappings = {
        "qkv.weight": ("qkv_linear.weight", lambda value: value[:, :, None, None]),
        "qkv.bias": ("qkv_linear.bias", lambda value: value),
        "proj.weight": ("output_linear.weight", lambda value: value[:, :, None, None]),
        "proj.bias": ("output_linear.bias", lambda value: value),
        "local_context.weight": ("lepe.weight", lambda value: value),
        "local_context.bias": ("lepe.bias", lambda value: value),
    }
    candidate_parameters = dict(candidate.named_parameters())
    official_parameters = dict(official.named_parameters())
    parameter_errors: Dict[str, float] = {}
    for candidate_name, (official_name, transform) in parameter_mappings.items():
        candidate_gradient = candidate_parameters[candidate_name].grad
        official_gradient = official_parameters[official_name].grad
        if candidate_gradient is None or official_gradient is None:
            parameter_errors[candidate_name] = math.inf
        else:
            parameter_errors[candidate_name] = float(
                (transform(candidate_gradient) - official_gradient).abs().amax().item()
            )

    errors = {
        "output_maximum_absolute_error": float(
            (candidate_map - official_output).abs().amax().item()
        ),
        "attention_maximum_absolute_error": float(
            (trace["native_sparse_attention"] - official_attention).abs().amax().item()
        ),
        "input_gradient_maximum_absolute_error": float(
            (
                candidate_input.grad.transpose(1, 2).reshape_as(official_input)
                - official_input.grad
            )
            .abs()
            .amax()
            .item()
        ),
        "parameter_gradient_errors": parameter_errors,
        "parameter_gradient_maximum_absolute_error": max(parameter_errors.values()),
        "route_indices_exact": bool(
            torch.equal(trace["route_indices"], official_routes)
        ),
    }
    errors["passed"] = bool(
        errors["route_indices_exact"]
        and max(
            errors["output_maximum_absolute_error"],
            errors["attention_maximum_absolute_error"],
            errors["input_gradient_maximum_absolute_error"],
            errors["parameter_gradient_maximum_absolute_error"],
        )
        <= 1e-6
    )
    return errors


def _dense_mhsa_parity() -> Dict[str, object]:
    torch.manual_seed(23)
    standard = MultiHeadSelfAttention(dim=32, num_heads=4).eval()
    torch.manual_seed(99)
    control = BiLevelRoutingAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        regions_per_axis=4,
        topk=16,
        local_context_kernel_size=5,
    ).eval()
    control.copy_shared_projections_from(standard)
    with torch.no_grad():
        control.local_context.weight.zero_()
        control.local_context.bias.zero_()
    torch.manual_seed(7)
    standard_input = torch.randn(2, 21, 32, requires_grad=True)
    control_input = standard_input.detach().clone().requires_grad_(True)
    standard_output, standard_attention = standard(
        standard_input, return_attention=True
    )
    control_output, control_attention = control(
        control_input,
        return_attention=True,
        grid_size=(4, 4),
        prefix_count=5,
        patch_indices=torch.arange(16).unsqueeze(0).expand(2, -1),
        collect_trace=True,
    )
    torch.manual_seed(31)
    projection = torch.randn_like(standard_output)
    (standard_output * projection).sum().backward()
    (control_output * projection).sum().backward()
    standard_parameters = dict(standard.named_parameters())
    control_parameters = dict(control.named_parameters())
    parameter_errors = {
        name: float(
            (standard_parameters[name].grad - control_parameters[name].grad)
            .abs()
            .amax()
            .item()
        )
        for name in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias")
    }
    result = {
        "output_maximum_absolute_error": float(
            (standard_output - control_output).abs().amax().item()
        ),
        "attention_maximum_absolute_error": float(
            (standard_attention - control_attention).abs().amax().item()
        ),
        "input_gradient_maximum_absolute_error": float(
            (standard_input.grad - control_input.grad).abs().amax().item()
        ),
        "parameter_gradient_errors": parameter_errors,
        "parameter_gradient_maximum_absolute_error": max(parameter_errors.values()),
    }
    result["passed"] = max(
        result["output_maximum_absolute_error"],
        result["attention_maximum_absolute_error"],
        result["input_gradient_maximum_absolute_error"],
        result["parameter_gradient_maximum_absolute_error"],
    ) <= 1e-6
    return result


def _gradient_summary(model: nn.Module) -> Dict[str, Dict[str, object]]:
    module = model.blocks[1].attn
    if not isinstance(module, BiLevelRoutingAttention):
        raise TypeError("Block 2 is not BRA during gradient inspection.")
    qkv_weight = module.qkv.weight.grad
    qkv_bias = module.qkv.bias.grad
    dim = module.dim
    families: Dict[str, Sequence[Optional[Tensor]]] = {
        "query": (
            None if qkv_weight is None else qkv_weight[:dim],
            None if qkv_bias is None else qkv_bias[:dim],
        ),
        "key": (
            None if qkv_weight is None else qkv_weight[dim : 2 * dim],
            None if qkv_bias is None else qkv_bias[dim : 2 * dim],
        ),
        "value": (
            None if qkv_weight is None else qkv_weight[2 * dim :],
            None if qkv_bias is None else qkv_bias[2 * dim :],
        ),
        "projection": (module.proj.weight.grad, module.proj.bias.grad),
        "local_context": (
            module.local_context.weight.grad,
            module.local_context.bias.grad,
        ),
    }
    summary: Dict[str, Dict[str, object]] = {}
    for name, values in families.items():
        present = [value for value in values if torch.is_tensor(value)]
        finite = len(present) == len(values) and all(
            bool(torch.isfinite(value).all()) for value in present
        )
        nonzero = sum(int(torch.count_nonzero(value).item()) for value in present)
        summary[name] = {
            "tensor_count": len(values),
            "gradient_tensor_count": len(present),
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": finite and nonzero > 0,
        }
    route_indices = module.trace().get("route_indices")
    summary["routing"] = {
        "indices_present": torch.is_tensor(route_indices),
        "indices_require_grad": bool(
            torch.is_tensor(route_indices) and route_indices.requires_grad
        ),
        "passed": bool(
            torch.is_tensor(route_indices) and not route_indices.requires_grad
        ),
    }
    return summary


def _route_set_comparison(left: Tensor, right: Tensor) -> Dict[str, object]:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Route comparison expects equal [B,R,K] tensors.")
    left_sorted = left.sort(dim=-1).values
    right_sorted = right.sort(dim=-1).values
    exact = left_sorted.eq(right_sorted).all(dim=-1)
    equality = left_sorted[..., :, None].eq(right_sorted[..., None, :])
    intersection = equality.any(dim=-1).sum(dim=-1).float()
    union = float(2 * left.size(-1)) - intersection
    jaccard = intersection / union.clamp_min(1.0)
    disagreements = (~exact).nonzero(as_tuple=False)
    return {
        "mean_jaccard": float(jaccard.mean().item()),
        "exact_fraction": float(exact.float().mean().item()),
        "disagreement_count": int(disagreements.size(0)),
        "disagreement_indices": disagreements[:256].cpu().tolist(),
    }


def _forward_capture(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    return_attention: bool,
    return_trace: bool,
) -> Tuple[Tensor, Tensor, Mapping[str, object]]:
    module = model.blocks[1].attn
    captured: list[Tensor] = []

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach().clone())

    handle = module.register_forward_hook(hook)
    try:
        logits, features = _forward(
            model,
            images,
            metadata,
            return_attention=return_attention,
            return_trace=return_trace,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one BRA invocation, observed {len(captured)}.")
    return logits, captured[0], features


def _trace_parity(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Dict[str, object]:
    model.eval()
    count = min(2, int(images.size(0)))
    sample_images = images[:count]
    sample_metadata = {key: value[:count] for key, value in metadata.items()}

    def compare(*, bf16: bool) -> Dict[str, object]:
        context = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=bf16
        )
        with torch.inference_mode(), context:
            standard_logits, standard_native, _ = _forward_capture(
                model,
                sample_images,
                sample_metadata,
                return_attention=False,
                return_trace=False,
            )
            trace_logits, trace_native, _ = _forward_capture(
                model,
                sample_images,
                sample_metadata,
                return_attention=True,
                return_trace=True,
            )
        return {
            "logit_maximum_absolute_error": float(
                (standard_logits.float() - trace_logits.float()).abs().amax().item()
            ),
            "native_output_maximum_absolute_error": float(
                (standard_native.float() - trace_native.float()).abs().amax().item()
            ),
            "argmax_match": bool(
                torch.equal(standard_logits.argmax(dim=1), trace_logits.argmax(dim=1))
            ),
            "finite": bool(
                torch.isfinite(standard_logits).all()
                and torch.isfinite(trace_logits).all()
                and torch.isfinite(standard_native).all()
                and torch.isfinite(trace_native).all()
            ),
        }

    fp32 = compare(bf16=False)
    bf16 = compare(bf16=True)
    return {
        "fp32": fp32,
        "bf16": bf16,
        "passed": all(
            bool(value["finite"])
            and bool(value["argmax_match"])
            and float(value["logit_maximum_absolute_error"]) <= 1e-6
            and float(value["native_output_maximum_absolute_error"]) <= 1e-6
            for value in (fp32, bf16)
        ),
    }


def _independent_sparse_scatter(
    native_attention: Tensor,
    route_indices: Tensor,
    region_token_indices: Tensor,
    *,
    prefix_count: int,
    patch_count: int,
) -> Tensor:
    batch_size, heads, regions, query_tokens, native_columns = native_attention.shape
    selected = region_token_indices.to(route_indices.device)[route_indices].flatten(2, 3)
    expected_columns = prefix_count + int(selected.size(-1))
    if native_columns != expected_columns:
        raise ValueError("Native BRA attention has an unexpected key count.")
    result = native_attention.new_zeros(
        batch_size, heads, patch_count, prefix_count + patch_count
    )
    query_indices = region_token_indices.to(route_indices.device)
    for region in range(regions):
        rows = query_indices[region]
        if prefix_count:
            result[:, :, rows, :prefix_count] = native_attention[
                :, :, region, :, :prefix_count
            ]
        columns = selected[:, region]
        values = native_attention[:, :, region, :, prefix_count:]
        scatter_index = columns[:, None, None, :].expand(
            batch_size, heads, query_tokens, -1
        )
        destination = result[:, :, rows, prefix_count:]
        destination.scatter_add_(dim=-1, index=scatter_index, src=values)
        result[:, :, rows, prefix_count:] = destination
    return result


def _attention_integrity(module: BiLevelRoutingAttention) -> Dict[str, object]:
    trace = module.trace()
    dense = trace["dense_attention"].float()
    native = trace["native_sparse_attention"].float()
    routes = trace["route_indices"]
    prefix_count = int(dense.size(-1)) - module.patch_count
    replay = _independent_sparse_scatter(
        native,
        routes,
        module.region_token_indices,
        prefix_count=prefix_count,
        patch_count=module.patch_count,
    )
    observed_patch = dense[:, :, prefix_count:]
    return {
        "dense_shape": [int(value) for value in dense.shape],
        "native_shape": [int(value) for value in native.shape],
        "finite": bool(torch.isfinite(dense).all() and torch.isfinite(native).all()),
        "nonnegative": bool((dense >= 0).all() and (native >= 0).all()),
        "dense_row_sum_maximum_error": float(
            (dense.sum(dim=-1) - 1.0).abs().amax().item()
        ),
        "native_row_sum_maximum_error": float(
            (native.sum(dim=-1) - 1.0).abs().amax().item()
        ),
        "scatter_replay_maximum_absolute_error": float(
            (observed_patch - replay).abs().amax().item()
        ),
    }


def _bbox_masks(bbox: Tensor, *, grid_size: int = 16) -> Tuple[Tensor, Tensor]:
    values = bbox.detach().float().cpu().flatten()
    if values.numel() < 4 or not bool(torch.isfinite(values[:4]).all()):
        raise ValueError("Selectivity audit requires a finite normalized bbox.")
    center_x, center_y, width, height = [float(value) for value in values[:4]]
    x1 = max(0.0, center_x - width / 2.0)
    y1 = max(0.0, center_y - height / 2.0)
    x2 = min(1.0, center_x + width / 2.0)
    y2 = min(1.0, center_y + height / 2.0)
    centers = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    object_mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    patch_margin = 1.0 / grid_size
    expanded_x1 = max(0.0, x1 - patch_margin)
    expanded_y1 = max(0.0, y1 - patch_margin)
    expanded_x2 = min(1.0, x2 + patch_margin)
    expanded_y2 = min(1.0, y2 + patch_margin)
    near_or_object = (
        (xx >= expanded_x1)
        & (xx <= expanded_x2)
        & (yy >= expanded_y1)
        & (yy <= expanded_y2)
    )
    return object_mask.flatten(), (~near_or_object).flatten()


def _single_route_geometry(
    routes: Tensor,
    bbox: Tensor,
    region_token_indices: Tensor,
) -> Dict[str, object]:
    routes = routes.detach().long().cpu()
    region_token_indices = region_token_indices.detach().long().cpu()
    if tuple(routes.shape) != (16, 4) or tuple(region_token_indices.shape) != (16, 16):
        raise ValueError("Locked BRA geometry expects routes [16,4] and regions [16,16].")
    object_mask, far_mask = _bbox_masks(bbox)
    selected_patch_indices = region_token_indices[routes].flatten(1, 2)
    query_weights = object_mask[region_token_indices].sum(dim=1).float()
    weight_sum = float(query_weights.sum().item())
    valid = weight_sum > 0.0
    if valid:
        routed_foreground_by_query = object_mask[selected_patch_indices].float().mean(dim=1)
        routed_far_by_query = far_mask[selected_patch_indices].float().mean(dim=1)
        routed_foreground = float(
            (routed_foreground_by_query * query_weights).sum().item() / weight_sum
        )
        routed_far = float((routed_far_by_query * query_weights).sum().item() / weight_sum)
    else:
        routed_foreground = math.nan
        routed_far = math.nan
    all_region_foreground = float(object_mask.float().mean().item())
    all_region_far = float(far_mask.float().mean().item())

    sorted_routes = routes.sort(dim=-1).values
    distinct_sets = int(torch.unique(sorted_routes, dim=0).size(0))
    equality = sorted_routes[:, None, :, None].eq(sorted_routes[None, :, None, :])
    intersection = equality.any(dim=-1).sum(dim=-1).float()
    union = 8.0 - intersection
    upper = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    pairwise_jaccard = float((intersection / union.clamp_min(1.0))[upper].mean().item())

    rows = torch.arange(4).repeat_interleave(4)
    columns = torch.arange(4).repeat(4)
    coordinates = torch.stack((rows, columns), dim=1)
    query_coordinates = coordinates[:, None, :]
    selected_coordinates = coordinates[routes]
    chebyshev_distance = (selected_coordinates - query_coordinates).abs().amax(dim=-1)
    nonlocal_fraction = float(chebyshev_distance.gt(1).float().mean().item())
    values = bbox.detach().float().cpu().flatten()
    center_x, center_y, width, height = [float(value) for value in values[:4]]
    edge_gap = min(
        center_x - width / 2.0,
        center_y - height / 2.0,
        1.0 - center_x - width / 2.0,
        1.0 - center_y - height / 2.0,
    )
    return {
        "valid_object_query": valid,
        "object_token_count": int(object_mask.sum().item()),
        "routed_foreground_fraction": routed_foreground,
        "all_region_foreground_fraction": all_region_foreground,
        "foreground_gain": routed_foreground - all_region_foreground,
        "routed_far_background_fraction": routed_far,
        "all_region_far_background_fraction": all_region_far,
        "far_background_reduction": all_region_far - routed_far,
        "distinct_route_sets": distinct_sets,
        "pairwise_route_jaccard": pairwise_jaccard,
        "nonlocal_route_fraction": nonlocal_fraction,
        "bbox_area": max(0.0, width) * max(0.0, height),
        "bbox_edge_gap": edge_gap,
        "bbox_center_edge_distance": max(abs(center_x - 0.5), abs(center_y - 0.5)),
    }


def _single_route_jaccard(left: Tensor, right: Tensor) -> float:
    result = _route_set_comparison(left.unsqueeze(0), right.unsqueeze(0))
    return float(result["mean_jaccard"])


def _construct_keeper_models(
    checkpoint: Mapping[str, object], *, seed: int
) -> Tuple[nn.Module, nn.Module, Dict[str, object]]:
    source_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state")
    if not isinstance(source_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Keeper checkpoint lacks model_config/class_names.")
    if not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint lacks model_state.")

    set_seed(seed, deterministic=True)
    control = create_model(
        num_classes=len(class_names), model_config=_bra_config(source_config, topk=16)
    )
    control_rng = _rng_snapshot()
    control_missing, control_unexpected = load_model_state(
        control, dict(state), strict=False
    )
    set_seed(seed, deterministic=True)
    candidate = create_model(
        num_classes=len(class_names), model_config=_bra_config(source_config, topk=4)
    )
    candidate_rng = _rng_snapshot()
    candidate_missing, candidate_unexpected = load_model_state(
        candidate, dict(state), strict=False
    )
    expected_additions = {
        "blocks.1.attn.local_context.weight",
        "blocks.1.attn.local_context.bias",
    }
    control_additions = set(control.state_dict()).difference(state)
    candidate_additions = set(candidate.state_dict()).difference(state)
    common_state = _state_common_summary(control, candidate)
    checkpoint_state_exact = all(
        key in control.state_dict()
        and key in candidate.state_dict()
        and torch.equal(control.state_dict()[key].cpu(), value.detach().cpu())
        and torch.equal(candidate.state_dict()[key].cpu(), value.detach().cpu())
        for key, value in state.items()
    )
    return control.eval(), candidate.eval(), {
        "control_constructor_rng": _rng_summary(control_rng),
        "candidate_constructor_rng": _rng_summary(candidate_rng),
        "constructor_rng_equal": _rng_equal(control_rng, candidate_rng),
        "control_missing_keys": sorted(control_missing),
        "candidate_missing_keys": sorted(candidate_missing),
        "control_unexpected_keys": sorted(control_unexpected),
        "candidate_unexpected_keys": sorted(candidate_unexpected),
        "expected_state_additions": sorted(expected_additions),
        "control_state_additions": sorted(control_additions),
        "candidate_state_additions": sorted(candidate_additions),
        "state_additions_exact": (
            control_additions == expected_additions
            and candidate_additions == expected_additions
            and set(control_missing) == expected_additions
            and set(candidate_missing) == expected_additions
        ),
        "checkpoint_state_bit_exact": checkpoint_state_exact,
        "matched_state": common_state,
    }


def _condition_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("No BRA selectivity rows were collected.")

    def mean(name: str) -> float:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        return float(values.mean()) if np.isfinite(values).all() else math.nan

    foreground_gains = np.asarray(
        [float(row["foreground_gain"]) for row in rows], dtype=np.float64
    )
    route_shift_values = [
        float(row["clean_route_jaccard"])
        for row in rows
        if row.get("clean_route_jaccard") not in (None, "")
    ]
    return {
        "rows": len(rows),
        "valid_object_query_rows": sum(
            int(bool(row["valid_object_query"])) for row in rows
        ),
        "finite_rows": sum(
            int(
                all(
                    math.isfinite(float(row[key]))
                    for key in (
                        "foreground_gain",
                        "far_background_reduction",
                        "distinct_route_sets",
                        "pairwise_route_jaccard",
                        "nonlocal_route_fraction",
                        "probability_mae",
                    )
                )
            )
            for row in rows
        ),
        "foreground_gain_mean": mean("foreground_gain"),
        "positive_foreground_gain_fraction": float(
            np.mean(foreground_gains > 0.0)
        ),
        "routed_foreground_fraction_mean": mean("routed_foreground_fraction"),
        "all_region_foreground_fraction_mean": mean(
            "all_region_foreground_fraction"
        ),
        "far_background_reduction_mean": mean("far_background_reduction"),
        "routed_far_background_fraction_mean": mean(
            "routed_far_background_fraction"
        ),
        "all_region_far_background_fraction_mean": mean(
            "all_region_far_background_fraction"
        ),
        "distinct_route_sets_mean": mean("distinct_route_sets"),
        "pairwise_route_jaccard_mean": mean("pairwise_route_jaccard"),
        "nonlocal_route_fraction_mean": mean("nonlocal_route_fraction"),
        "probability_mae_mean": mean("probability_mae"),
        "clean_route_jaccard_mean": (
            float(np.mean(route_shift_values)) if route_shift_values else None
        ),
        "selected_affinity_margin_mean": mean("selected_affinity_margin"),
    }


def _write_selectivity_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty selectivity CSV.")
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selectivity_audit(
    *,
    control: nn.Module,
    candidate: nn.Module,
    checkpoint: Mapping[str, object],
    fold_data: Path,
    holdout_rows: Sequence[CleanTrainRow],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, object], object, object, Sequence[Mapping[str, object]]]:
    base_dataset, transform, dataset_summary = _build_holdout_dataset(
        checkpoint,
        fold_data=fold_data,
        holdout_rows=holdout_rows,
    )
    control = control.to(device).eval()
    candidate = candidate.to(device).eval()
    module = candidate.blocks[1].attn
    if not isinstance(module, BiLevelRoutingAttention):
        raise TypeError("Keeper candidate block 2 is not BRA.")
    clean_routes: Dict[int, Tensor] = {}
    all_rows: list[Dict[str, object]] = []
    condition_summaries: Dict[str, object] = {}
    loader_summaries: Dict[str, object] = {}
    outputs_finite = True
    for condition_index, (condition, brightness, contrast) in enumerate(CONDITIONS):
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            list(range(len(base_dataset))),
            corruption=_condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = _make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"bra_preflight_{condition}_holdout",
            seed=seed + 100 + condition_index,
        )
        loader_summaries[condition] = loader_summary
        condition_rows: list[Dict[str, object]] = []
        with torch.inference_mode():
            for images_cpu, _targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = _metadata_to_device(metadata_cpu, device)
                control_logits, _ = _forward(control, images, metadata)
                candidate_logits, _ = _forward(candidate, images, metadata)
                trace_logits, _ = _forward(
                    candidate,
                    images,
                    metadata,
                    return_attention=True,
                    return_trace=True,
                )
                outputs_finite = bool(
                    outputs_finite
                    and torch.isfinite(control_logits).all()
                    and torch.isfinite(candidate_logits).all()
                    and torch.isfinite(trace_logits).all()
                )
                routes = module.trace()["route_indices"].detach().cpu()
                affinity_margin = (
                    module.trace()["selected_affinity_margin"].detach().float().cpu()
                )
                control_probabilities = control_logits.float().softmax(dim=1).cpu()
                candidate_probabilities = candidate_logits.float().softmax(dim=1).cpu()
                sample_indices = metadata_cpu.get("sample_index")
                bboxes = metadata_cpu.get("bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
                    raise ValueError("BRA selectivity metadata lacks sample_index/bbox.")
                for position, local_index_value in enumerate(sample_indices.tolist()):
                    local_index = int(local_index_value)
                    source = holdout_rows[local_index]
                    route = routes[position]
                    geometry = _single_route_geometry(
                        route,
                        bboxes[position],
                        module.region_token_indices,
                    )
                    if condition == "clean":
                        clean_routes[local_index] = route.clone()
                        clean_jaccard: Optional[float] = None
                    else:
                        clean_jaccard = _single_route_jaccard(
                            clean_routes[local_index], route
                        )
                    row: Dict[str, object] = {
                        "condition": condition,
                        "sample_index": source.sample_index,
                        "local_holdout_index": local_index,
                        "source_stem": source.source_stem,
                        "target": source.target,
                        "keeper_prediction": source.keeper_prediction,
                        **geometry,
                        "probability_mae": float(
                            (
                                control_probabilities[position]
                                - candidate_probabilities[position]
                            )
                            .abs()
                            .mean()
                            .item()
                        ),
                        "clean_route_jaccard": (
                            "" if clean_jaccard is None else clean_jaccard
                        ),
                        "selected_affinity_margin": float(
                            affinity_margin[position].mean().item()
                        ),
                    }
                    condition_rows.append(row)
                    all_rows.append(row)
        condition_summaries[condition] = _condition_summary(condition_rows)

    csv_path = output_dir / "route_selectivity_rows.csv"
    _write_selectivity_csv(csv_path, all_rows)
    clean = condition_summaries["clean"]
    shifted = [name for name, _, _ in CONDITIONS if name != "clean"]
    checks = {
        "all_holdout_rows_all_conditions": len(all_rows)
        == EXPECTED_HOLDOUT_ROWS * len(CONDITIONS),
        "all_object_queries_valid": all(
            int(summary["valid_object_query_rows"]) == EXPECTED_HOLDOUT_ROWS
            for summary in condition_summaries.values()
        ),
        "all_rows_finite": outputs_finite
        and all(
            int(summary["finite_rows"]) == EXPECTED_HOLDOUT_ROWS
            for summary in condition_summaries.values()
        ),
        "clean_foreground_gain": float(clean["foreground_gain_mean"]) >= 0.020,
        "clean_positive_gain_fraction": float(
            clean["positive_foreground_gain_fraction"]
        )
        >= 0.55,
        "clean_far_background_reduction": float(
            clean["far_background_reduction_mean"]
        )
        >= 0.020,
        "shift_foreground_gain_positive": all(
            float(condition_summaries[name]["foreground_gain_mean"]) > 0.0
            for name in shifted
        ),
        "shift_far_background_not_above_control": all(
            float(condition_summaries[name]["far_background_reduction_mean"]) >= 0.0
            for name in shifted
        ),
        "query_adaptive_distinct_sets": float(clean["distinct_route_sets_mean"])
        >= 4.0,
        "query_adaptive_pairwise_jaccard": float(
            clean["pairwise_route_jaccard_mean"]
        )
        <= 0.85,
        "nonlocal_route_fraction": float(clean["nonlocal_route_fraction_mean"])
        >= 0.10,
        "condition_route_stability": all(
            float(condition_summaries[name]["clean_route_jaccard_mean"]) >= 0.65
            for name in shifted
        ),
        "routing_changes_probability": float(clean["probability_mae_mean"]) >= 1e-4,
    }
    return (
        {
            "dataset": dataset_summary,
            "loader_summaries": loader_summaries,
            "conditions": condition_summaries,
            "checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            "passed": all(checks.values()),
            "rows_csv": str(csv_path.resolve()),
            "rows_csv_sha256": _sha256(csv_path),
        },
        base_dataset,
        transform,
        all_rows,
    )


def _event_category(row: CleanTrainRow) -> Optional[str]:
    if row.target == FOCUS_CLASS and row.keeper_prediction == FOCUS_CLASS:
        return "class1_keeper_tp"
    if row.target == FOCUS_CLASS and row.keeper_prediction != FOCUS_CLASS:
        return "class1_keeper_fn"
    if (
        row.target in RESTRICTED_NEGATIVE_CLASSES
        and row.keeper_prediction == FOCUS_CLASS
    ):
        return "restricted_keeper_fp"
    return None


def _select_visual_rows(
    clean_rows: Sequence[Mapping[str, object]],
    holdout_rows: Sequence[CleanTrainRow],
) -> Sequence[Dict[str, object]]:
    indexed = {int(row["local_holdout_index"]): dict(row) for row in clean_rows}
    selected: Dict[int, set[str]] = {}

    def add(local_index: int, tag: str) -> None:
        selected.setdefault(int(local_index), set()).add(tag)

    for category in (
        "class1_keeper_tp",
        "class1_keeper_fn",
        "restricted_keeper_fp",
    ):
        candidates = [
            local_index
            for local_index, source in enumerate(holdout_rows)
            if _event_category(source) == category
        ]
        if candidates:
            add(candidates[0], category)

    valid_indices = sorted(indexed)
    if not valid_indices:
        raise ValueError("No clean selectivity rows are available for visual review.")
    add(max(valid_indices, key=lambda index: indexed[index]["bbox_area"]), "close")
    add(min(valid_indices, key=lambda index: indexed[index]["bbox_area"]), "wide")
    add(min(valid_indices, key=lambda index: indexed[index]["bbox_area"]), "tiny")
    add(min(valid_indices, key=lambda index: indexed[index]["bbox_edge_gap"]), "partial")
    add(
        max(
            valid_indices,
            key=lambda index: indexed[index]["bbox_center_edge_distance"],
        ),
        "edge",
    )
    result = []
    for local_index in sorted(selected):
        source = holdout_rows[local_index]
        result.append(
            {
                "local_holdout_index": local_index,
                "sample_index": source.sample_index,
                "source_stem": source.source_stem,
                "target": source.target,
                "keeper_prediction": source.keeper_prediction,
                "event_category": _event_category(source) or "geometry_representative",
                "tags": sorted(selected[local_index]),
            }
        )
    return result


def _draw_route_overlay(
    image_tensor: Tensor,
    bbox: Tensor,
    routes: Tensor,
    region_token_indices: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    title: str,
) -> Image.Image:
    rgb = _rgb_from_tensor(image_tensor, mean=mean, std=std)
    base = Image.fromarray(rgb).convert("RGBA")
    width, height = base.size
    object_mask, _ = _bbox_masks(bbox)
    query_weights = object_mask[region_token_indices.cpu()].sum(dim=1)
    query_region = int(query_weights.argmax().item())
    selected_regions = [int(value) for value in routes[query_region].tolist()]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    region_width = width / 4.0
    region_height = height / 4.0
    for region in selected_regions:
        row, column = divmod(region, 4)
        rectangle = (
            int(round(column * region_width)),
            int(round(row * region_height)),
            int(round((column + 1) * region_width)),
            int(round((row + 1) * region_height)),
        )
        draw_overlay.rectangle(rectangle, fill=(40, 120, 255, 58), outline=(20, 90, 220, 210), width=2)
    query_row, query_column = divmod(query_region, 4)
    query_rectangle = (
        int(round(query_column * region_width)),
        int(round(query_row * region_height)),
        int(round((query_column + 1) * region_width)),
        int(round((query_row + 1) * region_height)),
    )
    draw_overlay.rectangle(query_rectangle, outline=(230, 35, 35, 255), width=4)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    for index in range(17):
        x = int(round(index * width / 16.0))
        y = int(round(index * height / 16.0))
        draw.line((x, 0, x, height), fill=(255, 255, 255), width=1)
        draw.line((0, y, width, y), fill=(255, 255, 255), width=1)
    values = bbox.detach().float().cpu().flatten()
    center_x, center_y, bbox_width, bbox_height = [float(value) for value in values[:4]]
    bbox_rectangle = (
        int(round((center_x - bbox_width / 2.0) * width)),
        int(round((center_y - bbox_height / 2.0) * height)),
        int(round((center_x + bbox_width / 2.0) * width)),
        int(round((center_y + bbox_height / 2.0) * height)),
    )
    draw.rectangle(bbox_rectangle, outline=(20, 220, 80), width=4)
    canvas = Image.new("RGB", (width, height + 42), (250, 250, 250))
    canvas.paste(composed, (0, 42))
    ImageDraw.Draw(canvas).text((6, 6), title[:100], fill=(10, 10, 10))
    return canvas


def _render_route_overlays(
    *,
    candidate: nn.Module,
    base_dataset,
    transform,
    selected_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    candidate.eval()
    module = candidate.blocks[1].attn
    if not isinstance(module, BiLevelRoutingAttention):
        raise TypeError("Visual-review candidate block 2 is not BRA.")
    cells: list[Tuple[int, str, Image.Image]] = []
    for row_index, selected in enumerate(selected_rows):
        local_index = int(selected["local_holdout_index"])
        for condition, brightness, contrast in CONDITIONS:
            dataset = _SelectedConditionDataset(
                base_dataset,
                [local_index],
                corruption=_condition_corruption(condition, brightness, contrast),
                transform=transform,
            )
            image_tensor, _label, metadata = dataset[0]
            batch_images = image_tensor.unsqueeze(0).to(device)
            batch_metadata = {
                key: value.unsqueeze(0).to(device)
                for key, value in metadata.items()
                if torch.is_tensor(value) and key != "sample_index"
            }
            with torch.inference_mode():
                _forward(
                    candidate,
                    batch_images,
                    batch_metadata,
                    return_attention=True,
                    return_trace=True,
                )
            routes = module.trace()["route_indices"][0].detach().cpu()
            title = (
                f"{condition} | idx={selected['sample_index']} "
                f"y={selected['target']} pred={selected['keeper_prediction']} | "
                f"{','.join(selected['tags'])}"
            )
            cells.append(
                (
                    row_index,
                    condition,
                    _draw_route_overlay(
                        image_tensor,
                        metadata["bbox"],
                        routes,
                        module.region_token_indices,
                        mean=mean,
                        std=std,
                        title=title,
                    ),
                )
            )

    rows_per_page = 3
    page_paths: list[str] = []
    page_hashes: Dict[str, str] = {}
    for page_index, row_start in enumerate(range(0, len(selected_rows), rows_per_page), start=1):
        page_rows = min(rows_per_page, len(selected_rows) - row_start)
        cell_width, cell_height = cells[0][2].size
        page = Image.new(
            "RGB",
            (cell_width * len(CONDITIONS), cell_height * page_rows),
            (235, 235, 235),
        )
        for row_offset in range(page_rows):
            global_row = row_start + row_offset
            row_cells = [cell for index, _condition, cell in cells if index == global_row]
            if len(row_cells) != len(CONDITIONS):
                raise RuntimeError("Visual route page has incomplete condition coverage.")
            for column, cell in enumerate(row_cells):
                page.paste(cell, (column * cell_width, row_offset * cell_height))
        path = output_dir / f"route_overlay_page_{page_index:02d}.png"
        page.save(path, format="PNG", optimize=True)
        resolved = str(path.resolve())
        page_paths.append(resolved)
        page_hashes[resolved] = _sha256(path)

    represented_tags = sorted(
        {str(tag) for selected in selected_rows for tag in selected["tags"]}
    )
    required_tags = {
        "class1_keeper_tp",
        "class1_keeper_fn",
        "restricted_keeper_fp",
        "close",
        "wide",
        "partial",
        "edge",
        "tiny",
    }
    return {
        "selected_rows": list(selected_rows),
        "selected_row_count": len(selected_rows),
        "conditions": [name for name, _, _ in CONDITIONS],
        "represented_tags": represented_tags,
        "required_tags": sorted(required_tags),
        "coverage_complete": required_tags.issubset(represented_tags),
        "pages": page_paths,
        "page_sha256": page_hashes,
        "page_count": len(page_paths),
    }


def _write_manifest(output_dir: Path) -> Path:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(
            {
                "method": METHOD,
                "artifacts": artifacts,
                "raw_data_modified": False,
                "validation_used": False,
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"BRA preflight summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual review finalization requires a result and nonempty note.")
    review_path = output_dir / "visual_review.json"
    if review_path.exists():
        raise FileExistsError(f"BRA visual review is already finalized: {review_path}")
    summary = _load_json(summary_path)
    visual = summary.get("visual_review_artifacts")
    if not isinstance(visual, Mapping):
        raise ValueError("BRA summary lacks visual-review artifacts.")
    pages = visual.get("pages")
    page_hashes = visual.get("page_sha256")
    if not isinstance(pages, list) or not isinstance(page_hashes, Mapping) or not pages:
        raise ValueError("BRA visual-review pages are incomplete.")
    observed_hashes = {}
    for raw_path in pages:
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BRA visual-review page is missing: {path}")
        observed_hashes[str(path)] = _sha256(path)
    if observed_hashes != {str(key): str(value) for key, value in page_hashes.items()}:
        raise ValueError("BRA visual-review page hashes changed after the automated audit.")
    passed = args.visual_review_result == "pass"
    review = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": args.visual_review_result,
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "page_count": len(pages),
        "pages_sha256": observed_hashes,
    }
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    gate = summary["gate"]
    gate["visual_review_completed"] = True
    gate["visual_review_passed"] = passed
    gate["formal_pair_permission"] = bool(gate["automated_pass"] and passed)
    summary["visual_review"] = review
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("BRA preflight requires CUDA.")
    if (
        int(args.seed) != 42
        or int(args.batch_size) != 32
        or int(args.fp32_batch_size) != 2
        or int(args.selectivity_batch_size) != 16
    ):
        raise ValueError(
            "BRA preflight is locked to seed42, batch32, FP32 batch2, selectivity batch16."
        )
    output_dir = _prepare_output(Path(args.output_dir))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    official_root = Path(args.official_root).resolve()
    paths = {
        "resolved_config": Path(args.resolved_config).resolve(),
        "scratch_launcher": Path(args.scratch_launcher).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "declaration_summary": Path(args.declaration_summary).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "dbb_closure": Path(args.dbb_closure).resolve(),
        "official_attention": official_root / "ops" / "bra_nchw.py",
        "official_gather": official_root / "ops" / "torch" / "rrsda.py",
        "official_architecture": official_root / "models" / "biformer_stl_nchw.py",
        "official_recipe": official_root / "configs" / "train_args.yaml",
        "official_license": official_root / "LICENSE",
        "paper": Path(args.paper).resolve(),
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    resolved_config = _load_json(paths["resolved_config"])
    source_model_config = resolved_config.get("model_config")
    if not isinstance(source_model_config, Mapping):
        raise ValueError("Resolved config lacks model_config.")
    fold_summary = _load_json(paths["fold_summary"])
    checkpoint = torch.load(
        paths["keeper_checkpoint"], map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Keeper checkpoint must be a mapping.")
    declaration_rows = _read_clean_train_rows(paths["declaration"])
    fit_rows = [row for row in declaration_rows if row.fold != FOLD]
    holdout_rows = [row for row in declaration_rows if row.fold == FOLD]

    baseline_config = _baseline_config(source_model_config)
    control_config = _bra_config(source_model_config, topk=16)
    candidate_config = _bra_config(source_model_config, topk=4)
    set_seed(int(args.seed), deterministic=True)
    baseline = create_model(num_classes=5, model_config=baseline_config)
    baseline_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    control = create_model(num_classes=5, model_config=control_config)
    control_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_rng = _rng_snapshot()
    matched_state = _state_common_summary(control, candidate)
    baseline_parameters = sum(int(value.numel()) for value in baseline.parameters())
    control_parameters = sum(int(value.numel()) for value in control.parameters())
    candidate_parameters = sum(int(value.numel()) for value in candidate.parameters())
    added_parameters = candidate_parameters - baseline_parameters
    bra_layers = [
        index + 1
        for index, block in enumerate(candidate.blocks)
        if isinstance(block.attn, BiLevelRoutingAttention)
    ]
    candidate_prefix_count = int(candidate.num_prefix_tokens)

    fit_dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    loader_kwargs, fit_loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="bra_a1_train_only_preflight",
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
    metadata = {
        key: value.to(device=device, non_blocking=True)
        for key, value in metadata_cpu.items()
    }

    del baseline, control
    candidate = candidate.to(device).train()
    fp32_count = int(args.fp32_batch_size)
    candidate.zero_grad(set_to_none=True)
    logits_fp32, _ = _forward(
        candidate,
        images[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    gradients_fp32 = _gradient_summary(candidate)
    fp32_finite = bool(torch.isfinite(logits_fp32).all() and torch.isfinite(loss_fp32))

    candidate.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_bf16, _ = _forward(candidate, images, metadata)
        loss_bf16 = F.cross_entropy(logits_bf16.float(), labels)
    loss_bf16.backward()
    gradients_bf16 = _gradient_summary(candidate)
    bf16_finite = bool(torch.isfinite(logits_bf16).all() and torch.isfinite(loss_bf16))

    candidate.eval()
    with torch.inference_mode():
        _forward(
            candidate,
            images,
            metadata,
            return_attention=True,
            return_trace=True,
        )
        module = candidate.blocks[1].attn
        if not isinstance(module, BiLevelRoutingAttention):
            raise TypeError("Scratch candidate block 2 is not BRA.")
        routes_fp32 = module.trace()["route_indices"].detach().clone()
        margin_fp32 = module.trace()["selected_affinity_margin"].detach().float().clone()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _forward(
                candidate,
                images,
                metadata,
                return_attention=True,
                return_trace=True,
            )
        routes_bf16 = module.trace()["route_indices"].detach().clone()
        margin_bf16 = module.trace()["selected_affinity_margin"].detach().float().clone()
    route_stability = _route_set_comparison(routes_fp32, routes_bf16)
    disagreement_mask = routes_fp32.sort(dim=-1).values.ne(
        routes_bf16.sort(dim=-1).values
    ).any(dim=-1)
    route_stability["fp32_margin_mean"] = float(margin_fp32.mean().item())
    route_stability["fp32_margin_min"] = float(margin_fp32.min().item())
    route_stability["bf16_margin_mean"] = float(margin_bf16.mean().item())
    route_stability["disagreement_fp32_margin_mean"] = (
        float(margin_fp32[disagreement_mask].mean().item())
        if bool(disagreement_mask.any())
        else None
    )
    attention_integrity = _attention_integrity(module)
    trace_parity = _trace_parity(candidate, images, metadata)
    del candidate
    gc.collect()
    torch.cuda.empty_cache()

    control_train_resource = _benchmark(
        config=baseline_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    candidate_train_resource = _benchmark(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    baseline_inference = _inference_benchmark(
        config=baseline_config,
        images=images,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    candidate_inference = _inference_benchmark(
        config=candidate_config,
        images=images,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    train_runtime_ratio = float(candidate_train_resource["median_seconds"]) / max(
        float(control_train_resource["median_seconds"]), 1e-12
    )
    inference_runtime_ratio = float(candidate_inference["median_seconds"]) / max(
        float(baseline_inference["median_seconds"]), 1e-12
    )
    train_vram_ratio = float(candidate_train_resource["peak_vram_gib"]) / max(
        float(control_train_resource["peak_vram_gib"]), 1e-12
    )

    set_seed(int(args.seed), deterministic=True)
    export_candidate = create_model(
        num_classes=5, model_config=candidate_config
    ).eval()
    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 4)
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            1, images_cpu.size(-2), images_cpu.size(-1), dtype=torch.bool
        )
    onnx_path = output_dir / "bra_candidate_static_batch1.onnx"
    try:
        onnx = _onnx_compare(
            wrapper=_FullExport(export_candidate),
            inputs=(
                images_cpu[:1].float(),
                bbox[:1].float(),
                image_mask[:1].bool(),
            ),
            input_names=("images", "bbox", "image_mask"),
            path=onnx_path,
        )
        import onnx as onnx_package

        graph = onnx_package.load(str(onnx_path)).graph
        operation_counts: Dict[str, int] = {}
        for node in graph.node:
            operation_counts[node.op_type] = operation_counts.get(node.op_type, 0) + 1
        onnx["operation_counts"] = operation_counts
        onnx["contains_topk"] = int(operation_counts.get("TopK", 0)) > 0
        onnx["contains_gather"] = any(
            int(operation_counts.get(name, 0)) > 0
            for name in ("Gather", "GatherElements", "GatherND")
        )
    except Exception as error:
        onnx = _failed_export(onnx_path, error)
        onnx["operation_counts"] = {}
        onnx["contains_topk"] = False
        onnx["contains_gather"] = False

    official_equation = _official_equation_replay(official_root)
    dense_parity = _dense_mhsa_parity()
    keeper_control, keeper_candidate, keeper_model_summary = _construct_keeper_models(
        checkpoint, seed=int(args.seed)
    )
    selectivity, base_dataset, eval_transform, selectivity_rows = _selectivity_audit(
        control=keeper_control,
        candidate=keeper_candidate,
        checkpoint=checkpoint,
        fold_data=paths["fold_data"],
        holdout_rows=holdout_rows,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.selectivity_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    clean_rows = [row for row in selectivity_rows if row["condition"] == "clean"]
    selected_visual_rows = _select_visual_rows(clean_rows, holdout_rows)
    semantics = selectivity["dataset"]["semantics"]
    visual_review_artifacts = _render_route_overlays(
        candidate=keeper_candidate,
        base_dataset=base_dataset,
        transform=eval_transform,
        selected_rows=selected_visual_rows,
        output_dir=output_dir,
        device=device,
        mean=semantics["input_mean"],
        std=semantics["input_std"],
    )
    del keeper_control, keeper_candidate
    gc.collect()
    torch.cuda.empty_cache()

    official_repository = _repository_state(official_root)
    gradient_pass = all(
        bool(value["passed"])
        for summary in (gradients_fp32, gradients_bf16)
        for value in summary.values()
    )
    source_hash_checks = {
        f"{name}_hash": hashes[name] == expected
        for name, expected in LOCKED_HASHES.items()
    }
    checks = {
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "head_pushed": _git_value("rev-parse", "HEAD")
        == _git_value("rev-parse", "@{upstream}"),
        **source_hash_checks,
        "official_branch": official_repository["branch"] == LOCKED_OFFICIAL_BRANCH,
        "official_commit": official_repository["commit"] == LOCKED_OFFICIAL_COMMIT,
        "official_tree": official_repository["tree"] == LOCKED_OFFICIAL_TREE,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_sources": int(fold_summary.get("fit_sources", -1))
        == EXPECTED_FIT_SOURCES,
        "fold_holdout_sources": int(fold_summary.get("holdout_sources", -1))
        == EXPECTED_HOLDOUT_SOURCES,
        "fold_source_disjoint": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fit_index_hash": _ordered_index_sha256(
            [row.sample_index for row in fit_rows]
        )
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_hash": _ordered_index_sha256(
            [row.sample_index for row in holdout_rows]
        )
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "fit_dataset_rows": len(fit_dataset) == EXPECTED_FIT_ROWS,
        "locked_architecture": bool(
            bra_layers == [2]
            and candidate_prefix_count == 7
            and candidate_config["bi_level_routing_attention_regions_per_axis"] == 4
            and candidate_config["bi_level_routing_attention_topk"] == 4
            and candidate_config[
                "bi_level_routing_attention_local_context_kernel_size"
            ]
            == 5
        ),
        "matched_parameter_inventory": bool(
            matched_state["inventory_equal"]
            and matched_state["all_common_bit_exact"]
            and control_parameters == candidate_parameters
        ),
        "constructor_rng_equal": _rng_equal(control_rng, candidate_rng)
        and _rng_equal(baseline_rng, control_rng),
        "added_parameter_budget": 0 < added_parameters <= MAX_ADDED_PARAMETERS,
        "official_equation_replay": bool(official_equation["passed"]),
        "dense_mhsa_parity": bool(dense_parity["passed"]),
        "fp32_finite": fp32_finite,
        "bf16_finite": bf16_finite,
        "all_gradient_families": gradient_pass,
        "route_stability_jaccard": float(route_stability["mean_jaccard"]) >= 0.98,
        "route_stability_exact": float(route_stability["exact_fraction"]) >= 0.95,
        "standard_trace_parity": bool(trace_parity["passed"]),
        "attention_finite_nonnegative": bool(
            attention_integrity["finite"] and attention_integrity["nonnegative"]
        ),
        "attention_rows_normalized": max(
            float(attention_integrity["dense_row_sum_maximum_error"]),
            float(attention_integrity["native_row_sum_maximum_error"]),
        )
        <= 1e-6,
        "attention_scatter_replay": float(
            attention_integrity["scatter_replay_maximum_absolute_error"]
        )
        <= 1e-6,
        "onnx_succeeded": bool(onnx.get("succeeded", False)),
        "onnx_finite": bool(onnx.get("finite", False)),
        "onnx_error": float(onnx.get("maximum_absolute_error", 1e9)) <= 1e-4,
        "onnx_argmax": bool(onnx.get("argmax_match", False)),
        "onnx_topk_gather": bool(
            onnx.get("contains_topk", False) and onnx.get("contains_gather", False)
        ),
        "inference_runtime_ratio": math.isfinite(inference_runtime_ratio)
        and inference_runtime_ratio <= MAX_INFERENCE_RUNTIME_RATIO,
        "train_runtime_ratio": math.isfinite(train_runtime_ratio)
        and train_runtime_ratio <= MAX_TRAIN_RUNTIME_RATIO,
        "candidate_vram_budget": float(candidate_train_resource["peak_vram_gib"])
        <= MAX_VRAM_GIB,
        "candidate_vram_ratio": math.isfinite(train_vram_ratio)
        and train_vram_ratio <= MAX_VRAM_RATIO,
        "resource_finite": all(
            bool(value["finite"])
            for value in (
                control_train_resource,
                candidate_train_resource,
                baseline_inference,
                candidate_inference,
            )
        ),
        "keeper_state_extensions_exact": bool(
            keeper_model_summary["state_additions_exact"]
            and keeper_model_summary["checkpoint_state_bit_exact"]
            and keeper_model_summary["constructor_rng_equal"]
            and keeper_model_summary["matched_state"]["all_common_bit_exact"]
        ),
        **{
            f"selectivity_{name}": bool(value)
            for name, value in selectivity["checks"].items()
        },
        "visual_artifact_coverage": bool(
            visual_review_artifacts["coverage_complete"]
            and visual_review_artifacts["page_count"] > 0
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "upstream": _git_value("rev-parse", "@{upstream}"),
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "official_repository": official_repository,
        "fold": fold_summary,
        "model": {
            "baseline_parameters": baseline_parameters,
            "control_parameters": control_parameters,
            "candidate_parameters": candidate_parameters,
            "added_parameters": added_parameters,
            "bra_layers": bra_layers,
            "matched_state": matched_state,
            "rng": {
                "baseline": _rng_summary(baseline_rng),
                "control": _rng_summary(control_rng),
                "candidate": _rng_summary(candidate_rng),
            },
            "keeper_models": keeper_model_summary,
        },
        "data": {
            "fit_rows": len(fit_dataset),
            "holdout_rows": len(holdout_rows),
            "fit_batch_size": int(images.size(0)),
            "fit_batch_labels": [int(value) for value in labels.cpu().tolist()],
            "fit_loader": fit_loader_summary,
        },
        "official_equation": official_equation,
        "dense_mhsa_parity": dense_parity,
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "gradients": gradients_fp32,
        },
        "bf16": {
            "batch_size": int(images.size(0)),
            "loss": float(loss_bf16.detach().item()),
            "gradients": gradients_bf16,
        },
        "route_stability": route_stability,
        "trace_parity": trace_parity,
        "attention_integrity": attention_integrity,
        "resource": {
            "baseline_train_step": control_train_resource,
            "candidate_train_step": candidate_train_resource,
            "baseline_inference": baseline_inference,
            "candidate_inference": candidate_inference,
            "train_runtime_ratio": train_runtime_ratio,
            "inference_runtime_ratio": inference_runtime_ratio,
            "train_peak_vram_ratio": train_vram_ratio,
        },
        "onnx": onnx,
        "selectivity": selectivity,
        "visual_review_artifacts": visual_review_artifacts,
        "validation_used": False,
        "test_used": False,
        "gate": {
            "checks": checks,
            "failed_checks": failed,
            "automated_pass": not failed,
            "visual_review_completed": False,
            "visual_review_passed": False,
            "formal_pair_permission": False,
            "validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "visual_review_required.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "required": True,
                "page_count": visual_review_artifacts["page_count"],
                "finalize_command": (
                    "python -m trkh.tools.audit_bi_level_routing_attention_preflight "
                    f"--output-dir \"{output_dir}\" --finalize-visual-review "
                    "--visual-review-result pass|fail --visual-review-note \"...\""
                ),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = (
        _finalize_visual_review(args)
        if bool(args.finalize_visual_review)
        else run_preflight(args)
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True), flush=True)
    if not bool(args.finalize_visual_review) and not bool(
        summary["gate"]["automated_pass"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
