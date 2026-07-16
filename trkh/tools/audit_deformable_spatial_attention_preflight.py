from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.models.model import MultiHeadSelfAttention, create_model
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
)
from trkh.tools.audit_foveal_aggregated_attention_preflight import (
    _FullExport,
    _benchmark,
    _build_dataset,
    _common_state_summary,
    _forward,
    _git_value,
    _load_json,
    _prepare_output,
    _sha256,
    _tracked_worktree_clean,
    _unpack_batch,
)


LOCKED_HASHES = {
    "protocol": "6abbd94ca08436225b65e4d460dbcb85e73098841094a3edd6a109f213f73cbc",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "declaration_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "resolved_config": "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    "launcher_args": "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "fold_data": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "faa_closure": "3cb722008f59e2558b66eee47efdee86def884225f882ab3e2be40079070a896",
    "official_attention": "f73185275dfddc6d5f1530ab5e10d3b5a85102b72ae7c73a9d85f64d8b7470ea",
    "official_architecture": "cea9a503e5b9c456802011cf84e585e3642b30140f4f2aa889b47272312d8be5",
    "official_config": "9427c5621e3636f9a6781dd1f8988451c8e6348498fdb1f3efd6841a9e119316",
    "official_license": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
    "paper": "92c3f6bba2aac7c1039ee4ed386743d345db8088e88279311eb7397630cb8a60",
    "current_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
LOCKED_OFFICIAL_COMMIT = "566a593daf96efc3df58a1542e60b847b8b6f4ff"
LOCKED_OFFICIAL_TREE = "4346f051456b3270e9161839410123260db82508"
LOCKED_OFFICIAL_TAG = "CVPR2022"
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_INDEX_SHA256 = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
EXPECTED_HOLDOUT_INDEX_SHA256 = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
MAX_ADDED_PARAMETERS = 50_000
MAX_RUNTIME_RATIO = 1.35
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.25


DAT_GRADIENT_MARKERS = {
    "offset_depthwise": ("blocks.1.attn.conv_offset.0.",),
    "offset_normalization": ("blocks.1.attn.conv_offset.1.norm.",),
    "offset_pointwise": ("blocks.1.attn.conv_offset.3.",),
    "relative_position_bias": (
        "blocks.1.attn.relative_position_bias_table",
    ),
    "qkv": ("blocks.1.attn.qkv.",),
    "projection": ("blocks.1.attn.proj.",),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only engineering preflight for block-2 CVPR-2022 "
            "Deformable Spatial Attention. Validation and test are forbidden."
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
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json"
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
            "docs/TRKH_5CLASS_DEFORMABLE_ATTENTION_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--faa-closure",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_FOVEAL_AGGREGATED_ATTENTION_CLOSURE_20260716.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\dat-cvpr2022"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\xia2022_dat.pdf"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    return parser.parse_args(argv)


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "early_token_mask_keep_rate": 1.0,
            "gated_relative_position_attention": False,
            "visual_contrast_attention": False,
            "foveal_aggregated_attention": False,
            "cross_covariance_attention": False,
            "dynamic_graph_mixer": False,
            "soft_moe_patch_adapter": False,
            "locally_enhanced_ffn": False,
            "concurrent_local_global_coupling": False,
            "deformable_spatial_attention": True,
            "deformable_spatial_attention_layers": "2",
            "deformable_spatial_attention_groups": 2,
            "deformable_spatial_attention_kernel_size": 5,
            "deformable_spatial_attention_offset_range": 2.0,
        }
    )
    return config


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _candidate_config(source)
    config["deformable_spatial_attention"] = False
    return config


def _gradient_summary(model: nn.Module) -> Dict[str, Dict[str, object]]:
    parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in DAT_GRADIENT_MARKERS.items():
        selected = {
            name: value
            for name, value in parameters.items()
            if any(marker in name for marker in markers)
        }
        gradients = [value.grad for value in selected.values()]
        present = [value for value in gradients if value is not None]
        finite = bool(present) and all(
            bool(torch.isfinite(value).all()) for value in present
        )
        nonzero = sum(int(torch.count_nonzero(value).item()) for value in present)
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": bool(
                selected and len(present) == len(selected) and finite and nonzero > 0
            ),
        }
    return summary


def _load_official_attention(path: Path):
    spec = importlib.util.spec_from_file_location(
        "locked_cvpr2022_dat_blocks_preflight",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked official DAT attention source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DAttentionBaseline


def _copy_candidate_to_official(
    candidate: DeformableSpatialAttention,
    official: nn.Module,
) -> None:
    dim = candidate.dim
    with torch.no_grad():
        official.proj_q.weight.copy_(
            candidate.qkv.weight[:dim].reshape(dim, dim, 1, 1)
        )
        official.proj_q.bias.copy_(candidate.qkv.bias[:dim])
        official.proj_k.weight.copy_(
            candidate.qkv.weight[dim : 2 * dim].reshape(dim, dim, 1, 1)
        )
        official.proj_k.bias.copy_(candidate.qkv.bias[dim : 2 * dim])
        official.proj_v.weight.copy_(
            candidate.qkv.weight[2 * dim :].reshape(dim, dim, 1, 1)
        )
        official.proj_v.bias.copy_(candidate.qkv.bias[2 * dim :])
        official.proj_out.weight.copy_(
            candidate.proj.weight.reshape(dim, dim, 1, 1)
        )
        official.proj_out.bias.copy_(candidate.proj.bias)
        official.conv_offset.load_state_dict(candidate.conv_offset.state_dict())
        official.rpe_table.copy_(candidate.relative_position_bias_table)


def _candidate_gradient_slice(
    candidate: DeformableSpatialAttention,
    official_name: str,
) -> Tensor:
    dim = candidate.dim
    named = dict(candidate.named_parameters())
    mapped_name = {
        "rpe_table": "relative_position_bias_table",
        "conv_offset.0.weight": "conv_offset.0.weight",
        "conv_offset.0.bias": "conv_offset.0.bias",
        "conv_offset.1.norm.weight": "conv_offset.1.norm.weight",
        "conv_offset.1.norm.bias": "conv_offset.1.norm.bias",
        "conv_offset.3.weight": "conv_offset.3.weight",
        "proj_q.weight": "qkv.weight",
        "proj_q.bias": "qkv.bias",
        "proj_k.weight": "qkv.weight",
        "proj_k.bias": "qkv.bias",
        "proj_v.weight": "qkv.weight",
        "proj_v.bias": "qkv.bias",
        "proj_out.weight": "proj.weight",
        "proj_out.bias": "proj.bias",
    }[official_name]
    gradient = named[mapped_name].grad
    if gradient is None:
        raise RuntimeError(f"Missing candidate gradient for {official_name}.")
    if official_name == "proj_q.weight":
        return gradient[:dim].reshape(dim, dim, 1, 1)
    if official_name == "proj_q.bias":
        return gradient[:dim]
    if official_name == "proj_k.weight":
        return gradient[dim : 2 * dim].reshape(dim, dim, 1, 1)
    if official_name == "proj_k.bias":
        return gradient[dim : 2 * dim]
    if official_name == "proj_v.weight":
        return gradient[2 * dim :].reshape(dim, dim, 1, 1)
    if official_name == "proj_v.bias":
        return gradient[2 * dim :]
    if official_name == "proj_out.weight":
        return gradient.reshape(dim, dim, 1, 1)
    return gradient


def _official_equation_replay(official_attention_path: Path) -> Dict[str, object]:
    official_class = _load_official_attention(official_attention_path)
    torch.manual_seed(41)
    candidate = DeformableSpatialAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        offset_groups=2,
        offset_kernel_size=5,
        offset_range_factor=2.0,
    ).eval()
    official = official_class(
        (4, 4),
        (4, 4),
        4,
        8,
        2,
        0.0,
        0.0,
        1,
        2.0,
        True,
        False,
        False,
        False,
        2,
    ).eval()
    _copy_candidate_to_official(candidate, official)
    torch.manual_seed(19)
    official_input = torch.randn(2, 32, 4, 4, requires_grad=True)
    candidate_input = (
        official_input.detach()
        .permute(0, 2, 3, 1)
        .reshape(2, 16, 32)
        .clone()
        .requires_grad_(True)
    )
    official_output, official_positions, official_reference = official(official_input)
    candidate_output = candidate(
        candidate_input,
        grid_size=(4, 4),
        prefix_count=0,
        patch_indices=torch.arange(16).unsqueeze(0).expand(2, -1),
    )
    candidate_map = candidate_output.reshape(2, 4, 4, 32).permute(0, 3, 1, 2)
    candidate_trace = candidate.trace()
    candidate_map.square().mean().backward()
    official_output.square().mean().backward()
    candidate_input_gradient = candidate_input.grad.reshape(2, 4, 4, 32).permute(
        0, 3, 1, 2
    )
    parameter_errors: Dict[str, float] = {}
    for name, parameter in official.named_parameters():
        if parameter.grad is None:
            parameter_errors[name] = math.inf
            continue
        parameter_errors[name] = float(
            (_candidate_gradient_slice(candidate, name) - parameter.grad)
            .abs()
            .amax()
            .item()
        )
    output_error = float((candidate_map - official_output).abs().amax().item())
    position_error = float(
        (candidate_trace["positions"] - official_positions).abs().amax().item()
    )
    reference_error = float(
        (candidate_trace["reference_positions"] - official_reference)
        .abs()
        .amax()
        .item()
    )
    input_gradient_error = float(
        (candidate_input_gradient - official_input.grad).abs().amax().item()
    )
    maximum_parameter_gradient_error = max(parameter_errors.values())
    passed = max(
        output_error,
        position_error,
        reference_error,
        input_gradient_error,
        maximum_parameter_gradient_error,
    ) <= 1e-6
    return {
        "output_maximum_absolute_error": output_error,
        "position_maximum_absolute_error": position_error,
        "reference_maximum_absolute_error": reference_error,
        "input_gradient_maximum_absolute_error": input_gradient_error,
        "parameter_gradient_maximum_absolute_error": maximum_parameter_gradient_error,
        "parameter_gradient_errors": parameter_errors,
        "passed": passed,
    }


def _prefix_parity(control: nn.Module, candidate: nn.Module) -> Dict[str, object]:
    control_attention = control.blocks[1].attn.eval()
    candidate_attention = candidate.blocks[1].attn.eval()
    if not isinstance(control_attention, MultiHeadSelfAttention):
        raise TypeError("DAT control block 2 is not standard MHSA.")
    if not isinstance(candidate_attention, DeformableSpatialAttention):
        raise TypeError("DAT candidate block 2 is not deformable attention.")
    prefix_count = int(candidate.num_prefix_tokens)
    patch_count = candidate_attention.patch_count
    torch.manual_seed(73)
    inputs = torch.randn(2, prefix_count + patch_count, candidate_attention.dim)
    indices = torch.arange(patch_count).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        control_output, control_map = control_attention(inputs, return_attention=True)
        candidate_output, candidate_map = candidate_attention(
            inputs,
            return_attention=True,
            grid_size=candidate_attention.input_resolution,
            prefix_count=prefix_count,
            patch_indices=indices,
        )
    output_error = float(
        (control_output[:, :prefix_count] - candidate_output[:, :prefix_count])
        .abs()
        .amax()
        .item()
    )
    attention_error = float(
        (control_map[:, :, :prefix_count] - candidate_map[:, :, :prefix_count])
        .abs()
        .amax()
        .item()
    )
    return {
        "prefix_count": prefix_count,
        "output_maximum_absolute_error": output_error,
        "attention_maximum_absolute_error": attention_error,
        "passed": max(output_error, attention_error) <= 1e-6,
    }


def _independent_bilinear_scatter(
    sample_attention: Tensor,
    positions: Tensor,
) -> Tuple[Tensor, Tensor]:
    if sample_attention.ndim != 4 or positions.ndim != 5:
        raise ValueError("Independent DAT proxy replay received invalid tensor ranks.")
    batch_size, num_heads, _, sample_count = sample_attention.shape
    _, groups, height, width, coordinates = positions.shape
    if coordinates != 2 or sample_count != height * width or num_heads % groups:
        raise ValueError("Independent DAT proxy replay received inconsistent shapes.")
    patch_count = height * width
    basis = torch.eye(
        patch_count,
        dtype=sample_attention.dtype,
        device=sample_attention.device,
    ).reshape(patch_count, 1, height, width)
    batch_bases = []
    for batch_index in range(batch_size):
        group_bases = []
        for group_index in range(groups):
            grid = positions[batch_index, group_index][..., (1, 0)]
            sampled = F.grid_sample(
                basis,
                grid.unsqueeze(0).expand(patch_count, -1, -1, -1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            group_bases.append(sampled[:, 0].flatten(1).transpose(0, 1))
        batch_bases.append(torch.stack(group_bases, dim=0))
    sampled_basis = torch.stack(batch_bases, dim=0)
    heads_per_group = num_heads // groups
    sampled_basis_heads = sampled_basis.unsqueeze(2).expand(
        -1, -1, heads_per_group, -1, -1
    ).reshape(batch_size, num_heads, sample_count, patch_count)
    dense = sample_attention @ sampled_basis_heads
    valid_mass = dense.sum(dim=-1)
    dense = dense / valid_mass.clamp_min(torch.finfo(dense.dtype).eps).unsqueeze(-1)
    return dense, valid_mass


def _rng_sha256(state: Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _inference_benchmark(
    *,
    config: Mapping[str, object],
    images: Tensor,
    metadata: Mapping[str, Tensor],
    seed: int,
    warmup_iterations: int,
    timed_iterations: int,
) -> Dict[str, object]:
    set_seed(seed)
    model = create_model(num_classes=5, model_config=config).to(images.device).eval()

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits, _ = _forward(model, images, metadata)
        return logits

    for _ in range(int(warmup_iterations)):
        iteration()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(images.device)
    elapsed = []
    logits = None
    for _ in range(int(timed_iterations)):
        torch.cuda.synchronize(images.device)
        started = time.perf_counter()
        logits = iteration()
        torch.cuda.synchronize(images.device)
        elapsed.append(float(time.perf_counter() - started))
    if logits is None:
        raise RuntimeError("DAT inference benchmark executed no timed iteration.")
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(
            torch.cuda.max_memory_allocated(images.device) / (1024**3)
        ),
        "finite": bool(torch.isfinite(logits).all()),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _git_repository_state(root: Path) -> Dict[str, str]:
    def value(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "commit": value("rev-parse", "HEAD"),
        "tree": value("rev-parse", "HEAD^{tree}"),
        "tag": value("describe", "--tags", "--exact-match", "HEAD"),
    }


def _trace_finite(trace: Mapping[str, Tensor]) -> bool:
    keys = ("positions", "reference_positions", "offsets")
    return all(
        torch.is_tensor(trace.get(key)) and bool(torch.isfinite(trace[key]).all())
        for key in keys
    )


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("DAT preflight requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("DAT preflight is locked to batch32 and FP32 batch2.")
    output_dir = _prepare_output(Path(args.output_dir))
    official_root = Path(args.official_root).resolve()
    paths = {
        "resolved_config": Path(args.resolved_config).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "declaration_summary": Path(args.declaration_summary).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "faa_closure": Path(args.faa_closure).resolve(),
        "official_attention": official_root / "models" / "dat_blocks.py",
        "official_architecture": official_root / "models" / "dat.py",
        "official_config": official_root / "configs" / "dat_tiny.yaml",
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
    candidate_config = _candidate_config(source_model_config)
    control_config = _control_config(source_model_config)

    set_seed(int(args.seed))
    control = create_model(num_classes=5, model_config=control_config)
    control_cpu_rng = torch.random.get_rng_state().clone()
    control_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    set_seed(int(args.seed))
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_cpu_rng = torch.random.get_rng_state().clone()
    candidate_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    common_state = _common_state_summary(control, candidate)
    common_state["candidate_only_finite"] = all(
        bool(torch.isfinite(value).all())
        for key, value in candidate.state_dict().items()
        if key in set(common_state["candidate_only_keys"])
        and (value.is_floating_point() or value.is_complex())
    )
    rng = {
        "control_cpu_sha256": _rng_sha256(control_cpu_rng),
        "candidate_cpu_sha256": _rng_sha256(candidate_cpu_rng),
        "cpu_equal": bool(torch.equal(control_cpu_rng, candidate_cpu_rng)),
        "control_cuda_sha256": [_rng_sha256(value) for value in control_cuda_rng],
        "candidate_cuda_sha256": [_rng_sha256(value) for value in candidate_cuda_rng],
        "cuda_equal": len(control_cuda_rng) == len(candidate_cuda_rng)
        and all(
            torch.equal(left, right)
            for left, right in zip(control_cuda_rng, candidate_cuda_rng)
        ),
    }
    prefix_parity = _prefix_parity(control, candidate)
    control_parameters = sum(value.numel() for value in control.parameters())
    candidate_parameters = sum(value.numel() for value in candidate.parameters())
    added_parameters = candidate_parameters - control_parameters
    dat_layers = [
        index + 1
        for index, block in enumerate(candidate.blocks)
        if isinstance(block.attn, DeformableSpatialAttention)
    ]

    dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="dat_a1_train_only_preflight",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(next(iter(loader)))
    device = torch.device("cuda")
    images = images_cpu.to(device=device, non_blocking=True)
    labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = {
        key: value.to(device=device, non_blocking=True)
        for key, value in metadata_cpu.items()
    }

    del control
    candidate = candidate.to(device).train()
    fp32_count = int(args.fp32_batch_size)
    candidate.zero_grad(set_to_none=True)
    logits_fp32, features_fp32 = _forward(
        candidate,
        images[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
        return_attention=True,
        return_trace=True,
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    fp32_gradients = _gradient_summary(candidate)
    fp32_attention = features_fp32["attentions"][1]
    fp32_trace = candidate.blocks[1].attn.trace()
    fp32_finite = bool(
        torch.isfinite(logits_fp32).all()
        and torch.isfinite(loss_fp32)
        and torch.isfinite(fp32_attention).all()
        and _trace_finite(fp32_trace)
    )

    candidate.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_bf16, _ = _forward(candidate, images, metadata)
        loss_bf16 = F.cross_entropy(logits_bf16.float(), labels)
    loss_bf16.backward()
    bf16_gradients = _gradient_summary(candidate)
    bf16_backward_trace = candidate.blocks[1].attn.trace()
    candidate.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        _, features_bf16 = _forward(
            candidate,
            images[:fp32_count],
            {key: value[:fp32_count] for key, value in metadata.items()},
            return_attention=True,
            return_trace=True,
        )
    bf16_attention = features_bf16["attentions"][1]
    bf16_attention_trace = candidate.blocks[1].attn.trace()
    bf16_finite = bool(
        torch.isfinite(logits_bf16).all()
        and torch.isfinite(loss_bf16)
        and torch.isfinite(bf16_attention).all()
        and _trace_finite(bf16_backward_trace)
        and _trace_finite(bf16_attention_trace)
    )

    with torch.inference_mode():
        attention_logits, attention_features = _forward(
            candidate,
            images[:1],
            {key: value[:1] for key, value in metadata.items()},
            return_attention=True,
            return_trace=True,
        )
    block_attention = attention_features["attentions"][1]
    module_trace = candidate.blocks[1].attn.trace()
    prefix_count = int(candidate.num_prefix_tokens)
    token_count = int(block_attention.size(-1))
    attention_row_error = float(
        (block_attention.sum(dim=-1) - 1.0).abs().amax().item()
    )
    patch_prefix_nonzero = int(
        torch.count_nonzero(
            block_attention[:, :, prefix_count:, :prefix_count]
        ).item()
    )
    independent_proxy, independent_valid_mass = _independent_bilinear_scatter(
        module_trace["sample_attention"].float(),
        module_trace["positions"].float(),
    )
    returned_patch_proxy = block_attention[
        :, :, prefix_count:, prefix_count:
    ].float()
    proxy_error = float(
        (returned_patch_proxy - independent_proxy).abs().amax().item()
    )
    offset_rms_per_group = module_trace["offset_rms_per_group"].float()
    inter_group_position_rms = float(
        module_trace["inter_group_position_rms"].item()
    )
    valid_interpolation_mass_mean = float(
        module_trace["valid_interpolation_mass_mean"].item()
    )
    valid_interpolation_mass_min = float(
        module_trace["valid_interpolation_mass_min"].item()
    )

    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    control_train_resource = _benchmark(
        config=control_config,
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
    control_inference = _inference_benchmark(
        config=control_config,
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
    runtime_ratio = float(candidate_inference["median_seconds"]) / max(
        float(control_inference["median_seconds"]), 1e-12
    )
    vram_ratio = float(candidate_train_resource["peak_vram_gib"]) / max(
        float(control_train_resource["peak_vram_gib"]), 1e-12
    )

    set_seed(int(args.seed))
    export_candidate = create_model(num_classes=5, model_config=candidate_config).eval()
    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 8)
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            1,
            images_cpu.size(-2),
            images_cpu.size(-1),
            dtype=torch.bool,
        )
    onnx_path = output_dir / "dat_candidate_static_batch1.onnx"
    try:
        onnx = _onnx_compare(
            wrapper=_FullExport(export_candidate),
            inputs=(images_cpu[:1].float(), bbox[:1].float(), image_mask[:1].bool()),
            input_names=("images", "bbox", "image_mask"),
            path=onnx_path,
        )
    except Exception as error:
        onnx = _failed_export(onnx_path, error)
    equation = _official_equation_replay(paths["official_attention"])
    official_repository = _git_repository_state(official_root)

    gradient_pass = all(
        bool(value["passed"]) for value in fp32_gradients.values()
    ) and all(bool(value["passed"]) for value in bf16_gradients.values())
    hash_checks = {
        f"{name}_hash": hashes[name] == expected
        for name, expected in LOCKED_HASHES.items()
    }
    checks = {
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "head_pushed": _git_value("rev-parse", "HEAD")
        == _git_value("rev-parse", "@{upstream}"),
        **hash_checks,
        "official_commit": official_repository["commit"] == LOCKED_OFFICIAL_COMMIT,
        "official_tree": official_repository["tree"] == LOCKED_OFFICIAL_TREE,
        "official_tag": official_repository["tag"] == LOCKED_OFFICIAL_TAG,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_sources": int(fold_summary.get("fit_sources", -1))
        == EXPECTED_FIT_SOURCES,
        "fold_holdout_sources": int(fold_summary.get("holdout_sources", -1))
        == EXPECTED_HOLDOUT_SOURCES,
        "fold_zero_source_overlap": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fold_fit_index_hash": fold_summary.get("fit_index_sha256")
        == EXPECTED_FIT_INDEX_SHA256,
        "fold_holdout_index_hash": fold_summary.get("holdout_index_sha256")
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "dataset_fit_rows": len(dataset) == EXPECTED_FIT_ROWS,
        "common_state_bit_exact": bool(common_state["all_common_bit_exact"]),
        "candidate_only_state_finite": bool(common_state["candidate_only_finite"]),
        "constructor_cpu_rng_equal": bool(rng["cpu_equal"]),
        "constructor_cuda_rng_equal": bool(rng["cuda_equal"]),
        "candidate_layer_exact": dat_layers == [2],
        "candidate_settings_exact": all(
            (
                candidate_config["deformable_spatial_attention"] is True,
                candidate_config["deformable_spatial_attention_layers"] == "2",
                candidate_config["deformable_spatial_attention_groups"] == 2,
                candidate_config["deformable_spatial_attention_kernel_size"] == 5,
                candidate_config["deformable_spatial_attention_offset_range"] == 2.0,
            )
        ),
        "added_parameters_budget": 0 < added_parameters <= MAX_ADDED_PARAMETERS,
        "official_equation_replay": bool(equation["passed"]),
        "prefix_parity": bool(prefix_parity["passed"]),
        "fp32_finite": fp32_finite,
        "bf16_finite": bf16_finite,
        "all_gradient_families": gradient_pass,
        "proxy_shape": tuple(block_attention.shape)
        == (1, 8, token_count, token_count),
        "proxy_finite_nonnegative": bool(
            torch.isfinite(block_attention).all() and (block_attention >= 0).all()
        ),
        "proxy_rows_normalized": attention_row_error <= 1e-5,
        "proxy_patch_to_prefix_zero": patch_prefix_nonzero == 0,
        "proxy_independent_replay": proxy_error <= 1e-6,
        "proxy_representation_labeled": attention_features.get(
            "attention_representations", {}
        ).get(1)
        == "deformable_bilinear_sample_proxy",
        "positions_offsets_finite": _trace_finite(module_trace),
        "every_group_offset_nonzero": bool((offset_rms_per_group > 0).all()),
        "inter_group_positions_nonidentical": inter_group_position_rms > 0.0,
        "valid_interpolation_mass_nonzero": valid_interpolation_mass_min > 0.0
        and bool((independent_valid_mass > 0).all()),
        "runtime_ratio": math.isfinite(runtime_ratio)
        and runtime_ratio <= MAX_RUNTIME_RATIO,
        "candidate_vram_budget": float(candidate_train_resource["peak_vram_gib"])
        <= MAX_VRAM_GIB,
        "candidate_vram_ratio": math.isfinite(vram_ratio)
        and vram_ratio <= MAX_VRAM_RATIO,
        "resource_finite": bool(
            control_train_resource["finite"]
            and candidate_train_resource["finite"]
            and control_inference["finite"]
            and candidate_inference["finite"]
        ),
        "onnx_succeeded": bool(onnx.get("succeeded", False)),
        "onnx_finite": bool(onnx.get("finite", False)),
        "onnx_error": float(onnx.get("maximum_absolute_error", 1e9)) <= 1e-4,
        "onnx_argmax": bool(onnx.get("argmax_match", False)),
        "attention_logits_finite": bool(torch.isfinite(attention_logits).all()),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    summary: Dict[str, object] = {
        "method": "deformable_spatial_attention_a1_preflight",
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
            "control_parameters": control_parameters,
            "candidate_parameters": candidate_parameters,
            "added_parameters": added_parameters,
            "dat_layers": dat_layers,
            "common_state": common_state,
            "rng": rng,
            "prefix_parity": prefix_parity,
        },
        "data": {
            "fit_rows": len(dataset),
            "batch_size": int(images.size(0)),
            "labels": [int(value) for value in labels.cpu().tolist()],
            "dataloader": loader_summary,
        },
        "official_equation": equation,
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "gradients": fp32_gradients,
        },
        "bf16": {
            "batch_size": int(images.size(0)),
            "loss": float(loss_bf16.detach().item()),
            "gradients": bf16_gradients,
        },
        "attention": {
            "shape": [int(value) for value in block_attention.shape],
            "row_sum_max_error": attention_row_error,
            "patch_prefix_nonzero": patch_prefix_nonzero,
            "independent_proxy_maximum_absolute_error": proxy_error,
            "representation": attention_features.get(
                "attention_representations", {}
            ).get(1),
            "offset_rms": float(module_trace["offset_rms"].item()),
            "offset_rms_per_group": [
                float(value) for value in offset_rms_per_group.cpu().tolist()
            ],
            "inter_group_position_rms": inter_group_position_rms,
            "valid_interpolation_mass_mean": valid_interpolation_mass_mean,
            "valid_interpolation_mass_min": valid_interpolation_mass_min,
            "position_min": float(module_trace["position_min"].item()),
            "position_max": float(module_trace["position_max"].item()),
        },
        "resource": {
            "control_train_step": control_train_resource,
            "candidate_train_step": candidate_train_resource,
            "control_inference": control_inference,
            "candidate_inference": candidate_inference,
            "inference_runtime_ratio": runtime_ratio,
            "train_peak_vram_ratio": vram_ratio,
        },
        "onnx": onnx,
        "gate": {
            "formal_pair_permission": not failed,
            "checks": checks,
            "failed_checks": failed,
            "validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
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
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": "deformable_spatial_attention_a1_preflight",
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
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run_preflight(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    if not bool(summary["gate"]["formal_pair_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
