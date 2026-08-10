from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.models.diverse_branch_stem import (
    BatchNormAndPad2d,
    DiverseBranchConvBN,
    DiverseBranchConvStem,
    IdentityBasedConv1x1,
    convert_dbb_modules_to_deploy,
    fuse_conv_batch_norm,
)
from trkh.models.model import HybridConvStem, create_model
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
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


LOCKED_HASHES = {
    "protocol": "578b208f2f50da69b3b3ac2113705750302ffc5b6f205fc20bd60133045d22ff",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "declaration_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "resolved_config": "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97",
    "launcher_args": "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6",
    "keeper_checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "scratch_checkpoint": "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549",
    "fold_data": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "official_block": "5f67f20081ffef0a9f7ad6f9f5cc1a7bb4e59902dc31c86a8e59c37fd477ccb0",
    "official_transforms": "1424673513c4883ffb2e88b24f81a41cb69569f2621aa18f76a8fe341c92b11f",
    "official_readme": "a8966d48d0eb0f37923bb9e5863f1409bd6f4daaeceefbbb07316e9589456267",
    "official_license": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
    "paper": "ba1c0f907f69dea3a6e84634f4198ff4cb9799569633078a9ca5a013e3b0d991",
    "current_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
LOCKED_OFFICIAL_COMMIT = "8d2b16b6aee45a33236b2d11685be6857f9ba929"
LOCKED_OFFICIAL_TREE = "b038d8e014664312f8979d0b7917466e70877b30"
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_INDEX_SHA256 = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
EXPECTED_HOLDOUT_INDEX_SHA256 = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
OFFICIAL_TOLERANCE = 1e-6
FP32_DEPLOY_TOLERANCE = 1e-5
BF16_DEPLOY_TOLERANCE = 2e-3
MAX_TRAIN_RUNTIME_RATIO = 1.50
MAX_INFERENCE_RUNTIME_RATIO = 1.05
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.25


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only engineering preflight for the CVPR-2021 "
            "Diverse Branch Block stem. Validation and test are forbidden."
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
        "--scratch-checkpoint",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt"
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
            "docs/TRKH_5CLASS_DIVERSE_BRANCH_STEM_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\dbb-cvpr2021"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\dbb_cvpr2021.pdf"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=7)
    return parser.parse_args(argv)


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "stem_architecture": "dbb_conv_pool",
            "stem_pooling_mode": "max",
        }
    )
    return config


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _candidate_config(source)
    config["stem_architecture"] = "conv_pool"
    return config


def _rng_sha256(state: Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


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
    }


def _state_pairing_summary(control: nn.Module, candidate: nn.Module) -> Dict[str, object]:
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    control_nonstem = {
        key: value for key, value in control_state.items() if not key.startswith("stem.")
    }
    candidate_nonstem = {
        key: value
        for key, value in candidate_state.items()
        if not key.startswith("stem.")
    }
    nonstem_keys_equal = control_nonstem.keys() == candidate_nonstem.keys()
    differing_nonstem = (
        [
            key
            for key in control_nonstem
            if not torch.equal(control_nonstem[key], candidate_nonstem[key])
        ]
        if nonstem_keys_equal
        else sorted(set(control_nonstem).symmetric_difference(candidate_nonstem))
    )

    origin_pairs = []
    differing_origin = []
    for block_index in range(3):
        control_prefix = f"stem.blocks.{block_index}.block."
        candidate_prefix = f"stem.blocks.{block_index}.dbb.dbb_origin."
        for suffix in (
            "conv.weight",
            "norm.weight",
            "norm.bias",
            "norm.running_mean",
            "norm.running_var",
            "norm.num_batches_tracked",
        ):
            control_key = control_prefix + suffix
            candidate_key = candidate_prefix + suffix
            equal = bool(
                control_key in control_state
                and candidate_key in candidate_state
                and torch.equal(control_state[control_key], candidate_state[candidate_key])
            )
            origin_pairs.append(
                {
                    "control": control_key,
                    "candidate": candidate_key,
                    "bit_exact": equal,
                }
            )
            if not equal:
                differing_origin.append(f"{control_key}->{candidate_key}")

    origin_candidate_keys = {item["candidate"] for item in origin_pairs}
    candidate_only = {
        key: value
        for key, value in candidate_state.items()
        if key.startswith("stem.") and key not in origin_candidate_keys
    }
    candidate_only_finite = all(
        bool(torch.isfinite(value).all())
        for value in candidate_only.values()
        if value.is_floating_point() or value.is_complex()
    )
    terminal_gamma_one = all(
        bool(torch.equal(norm.weight, torch.ones_like(norm.weight)))
        for block in candidate.stem.blocks
        for norm in block.dbb.terminal_norms().values()
    )
    return {
        "nonstem_keys_equal": nonstem_keys_equal,
        "differing_nonstem_keys": differing_nonstem,
        "all_nonstem_bit_exact": nonstem_keys_equal and not differing_nonstem,
        "origin_pairs": origin_pairs,
        "differing_origin_pairs": differing_origin,
        "all_origin_bit_exact": not differing_origin,
        "candidate_only_keys": sorted(candidate_only),
        "candidate_only_finite": candidate_only_finite,
        "terminal_gamma_one": terminal_gamma_one,
    }


def _path_gradient_summary(model: nn.Module) -> Dict[str, Dict[str, object]]:
    parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    path_modules = {
        "origin": "dbb_origin.",
        "pointwise": "dbb_1x1.",
        "sequential": "dbb_1x1_kxk.",
        "average": "dbb_avg.",
    }
    for block_index in range(3):
        for path_name, module_name in path_modules.items():
            prefix = f"stem.blocks.{block_index}.dbb.{module_name}"
            selected = {
                name: parameter
                for name, parameter in parameters.items()
                if name.startswith(prefix)
            }
            present = {
                name: parameter.grad
                for name, parameter in selected.items()
                if parameter.grad is not None
            }
            finite = bool(present) and all(
                bool(torch.isfinite(gradient).all())
                for gradient in present.values()
            )
            nonzero_by_tensor = {
                name: int(torch.count_nonzero(gradient).item())
                for name, gradient in present.items()
            }
            key = f"block{block_index + 1}.{path_name}"
            summary[key] = {
                "parameter_tensors": len(selected),
                "gradient_tensors": len(present),
                "finite": finite,
                "nonzero_elements": int(sum(nonzero_by_tensor.values())),
                "zero_gradient_tensors": sorted(
                    name for name, count in nonzero_by_tensor.items() if count == 0
                ),
                "passed": bool(
                    selected
                    and len(present) == len(selected)
                    and finite
                    and all(count > 0 for count in nonzero_by_tensor.values())
                ),
            }
    return summary


def _load_official_block(path: Path):
    source_root = str(path.parent)
    sys.path.insert(0, source_root)
    try:
        spec = importlib.util.spec_from_file_location(
            "locked_cvpr2021_diverse_branch_block_preflight",
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import locked official DBB source.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DiverseBranchBlock
    finally:
        if sys.path and sys.path[0] == source_root:
            sys.path.pop(0)


def _copy_local_to_official(local: DiverseBranchConvBN, official: nn.Module) -> None:
    official.dbb_origin.conv.load_state_dict(local.dbb_origin.conv.state_dict())
    official.dbb_origin.bn.load_state_dict(local.dbb_origin.norm.state_dict())
    official.dbb_1x1.conv.load_state_dict(local.dbb_1x1.conv.state_dict())
    official.dbb_1x1.bn.load_state_dict(local.dbb_1x1.norm.state_dict())
    official.dbb_1x1_kxk.idconv1.load_state_dict(
        local.dbb_1x1_kxk.idconv1.state_dict()
    )
    official.dbb_1x1_kxk.bn1.bn.load_state_dict(
        local.dbb_1x1_kxk.norm1.norm.state_dict()
    )
    official.dbb_1x1_kxk.conv2.load_state_dict(
        local.dbb_1x1_kxk.conv2.state_dict()
    )
    official.dbb_1x1_kxk.bn2.load_state_dict(
        local.dbb_1x1_kxk.norm2.state_dict()
    )
    official.dbb_avg.conv.load_state_dict(local.dbb_avg.conv.state_dict())
    official.dbb_avg.bn.bn.load_state_dict(local.dbb_avg.norm.norm.state_dict())
    official.dbb_avg.avgbn.load_state_dict(local.dbb_avg.avg_norm.state_dict())


def _randomize_replay_block(block: DiverseBranchConvBN, seed: int) -> None:
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            if any(
                name.endswith(suffix)
                for suffix in ("norm.weight", "norm2.weight", "avg_norm.weight")
            ):
                parameter.uniform_(0.55, 1.35, generator=generator)
            elif any(
                name.endswith(suffix)
                for suffix in ("norm.bias", "norm2.bias", "avg_norm.bias")
            ):
                parameter.uniform_(-0.15, 0.15, generator=generator)
            else:
                parameter.normal_(mean=0.0, std=0.08, generator=generator)
        for module in block.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.normal_(mean=0.0, std=0.2, generator=generator)
                module.running_var.uniform_(0.55, 1.45, generator=generator)


def _official_parameter_pairs(
    local: DiverseBranchConvBN,
    official: nn.Module,
) -> Dict[str, Tuple[nn.Parameter, nn.Parameter]]:
    return {
        "origin.conv": (local.dbb_origin.conv.weight, official.dbb_origin.conv.weight),
        "origin.norm.weight": (local.dbb_origin.norm.weight, official.dbb_origin.bn.weight),
        "origin.norm.bias": (local.dbb_origin.norm.bias, official.dbb_origin.bn.bias),
        "pointwise.conv": (local.dbb_1x1.conv.weight, official.dbb_1x1.conv.weight),
        "pointwise.norm.weight": (local.dbb_1x1.norm.weight, official.dbb_1x1.bn.weight),
        "pointwise.norm.bias": (local.dbb_1x1.norm.bias, official.dbb_1x1.bn.bias),
        "sequential.idconv": (
            local.dbb_1x1_kxk.idconv1.weight,
            official.dbb_1x1_kxk.idconv1.weight,
        ),
        "sequential.norm1.weight": (
            local.dbb_1x1_kxk.norm1.norm.weight,
            official.dbb_1x1_kxk.bn1.bn.weight,
        ),
        "sequential.norm1.bias": (
            local.dbb_1x1_kxk.norm1.norm.bias,
            official.dbb_1x1_kxk.bn1.bn.bias,
        ),
        "sequential.conv2": (
            local.dbb_1x1_kxk.conv2.weight,
            official.dbb_1x1_kxk.conv2.weight,
        ),
        "sequential.norm2.weight": (
            local.dbb_1x1_kxk.norm2.weight,
            official.dbb_1x1_kxk.bn2.weight,
        ),
        "sequential.norm2.bias": (
            local.dbb_1x1_kxk.norm2.bias,
            official.dbb_1x1_kxk.bn2.bias,
        ),
        "average.conv": (local.dbb_avg.conv.weight, official.dbb_avg.conv.weight),
        "average.norm.weight": (
            local.dbb_avg.norm.norm.weight,
            official.dbb_avg.bn.bn.weight,
        ),
        "average.norm.bias": (
            local.dbb_avg.norm.norm.bias,
            official.dbb_avg.bn.bn.bias,
        ),
        "average.final_norm.weight": (
            local.dbb_avg.avg_norm.weight,
            official.dbb_avg.avgbn.weight,
        ),
        "average.final_norm.bias": (
            local.dbb_avg.avg_norm.bias,
            official.dbb_avg.avgbn.bias,
        ),
    }


def _official_replay_case(
    official_class: type,
    inputs: Tensor,
    *,
    training: bool,
    seed: int,
) -> Dict[str, object]:
    local = DiverseBranchConvBN(3, 32)
    _randomize_replay_block(local, seed)
    official = official_class(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        deploy=False,
        nonlinear=nn.Identity(),
    )
    _copy_local_to_official(local, official)
    local.train(training)
    official.train(training)
    local_input = inputs.detach().float().clone().requires_grad_(True)
    official_input = inputs.detach().float().clone().requires_grad_(True)
    local_output = local(local_input)
    official_output = official(official_input)
    local_output.square().mean().backward()
    official_output.square().mean().backward()
    parameter_errors = {}
    for name, (local_parameter, official_parameter) in _official_parameter_pairs(
        local, official
    ).items():
        if local_parameter.grad is None or official_parameter.grad is None:
            parameter_errors[name] = math.inf
        else:
            parameter_errors[name] = float(
                (local_parameter.grad - official_parameter.grad).abs().amax().item()
            )
    local_kernel, local_bias = local.get_equivalent_kernel_bias()
    official_kernel, official_bias = official.get_equivalent_kernel_bias()
    result = {
        "training": bool(training),
        "shape": [int(value) for value in inputs.shape],
        "output_maximum_absolute_error": float(
            (local_output - official_output).abs().amax().item()
        ),
        "input_gradient_maximum_absolute_error": float(
            (local_input.grad - official_input.grad).abs().amax().item()
        ),
        "kernel_maximum_absolute_error": float(
            (local_kernel - official_kernel).abs().amax().item()
        ),
        "bias_maximum_absolute_error": float(
            (local_bias - official_bias).abs().amax().item()
        ),
        "parameter_gradient_errors": parameter_errors,
        "parameter_gradient_maximum_absolute_error": max(parameter_errors.values()),
    }
    result["passed"] = max(
        float(result["output_maximum_absolute_error"]),
        float(result["input_gradient_maximum_absolute_error"]),
        float(result["kernel_maximum_absolute_error"]),
        float(result["bias_maximum_absolute_error"]),
        float(result["parameter_gradient_maximum_absolute_error"]),
    ) <= OFFICIAL_TOLERANCE
    return result


def _official_replay(
    official_path: Path,
    real_inputs: Tensor,
) -> Dict[str, object]:
    official_class = _load_official_block(official_path)
    torch.manual_seed(19)
    random_inputs = torch.randn(2, 3, 17, 17)
    real_inputs = real_inputs[:2].detach().float().cpu()
    cases = {
        f"{source}_{mode}": _official_replay_case(
            official_class,
            inputs,
            training=mode == "train",
            seed=41,
        )
        for source, inputs in (("random", random_inputs), ("real_fit", real_inputs))
        for mode in ("train", "eval")
    }
    return {
        "cases": cases,
        "maximum_error": max(
            max(
                float(case["output_maximum_absolute_error"]),
                float(case["input_gradient_maximum_absolute_error"]),
                float(case["kernel_maximum_absolute_error"]),
                float(case["bias_maximum_absolute_error"]),
                float(case["parameter_gradient_maximum_absolute_error"]),
            )
            for case in cases.values()
        ),
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def _independent_fuse(kernel: Tensor, norm: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]:
    scale = norm.weight / (norm.running_var + norm.eps).sqrt()
    return (
        torch.einsum("oihw,o->oihw", kernel, scale),
        norm.bias - norm.running_mean * scale,
    )


def _independent_compose(
    first_kernel: Tensor,
    first_bias: Tensor,
    second_kernel: Tensor,
    second_bias: Tensor,
) -> Tuple[Tensor, Tensor]:
    kernel = torch.einsum(
        "omhw,mi->oihw",
        second_kernel,
        first_kernel[:, :, 0, 0],
    )
    bias = torch.einsum("omhw,m->o", second_kernel, first_bias) + second_bias
    return kernel, bias


def _independent_equation_replay() -> Dict[str, object]:
    block = DiverseBranchConvBN(3, 7).eval()
    _randomize_replay_block(block, 71)
    torch.manual_seed(73)
    inputs = torch.randn(2, 3, 13, 13)
    with torch.inference_mode():
        observed_total, observed_paths = block.forward_with_branches(inputs)

        origin = _independent_fuse(block.dbb_origin.conv.weight, block.dbb_origin.norm)
        pointwise_small = _independent_fuse(
            block.dbb_1x1.conv.weight,
            block.dbb_1x1.norm,
        )
        pointwise = (F.pad(pointwise_small[0], (1, 1, 1, 1)), pointwise_small[1])

        sequential_first = _independent_fuse(
            block.dbb_1x1_kxk.idconv1.actual_kernel(),
            block.dbb_1x1_kxk.norm1.norm,
        )
        sequential_second = _independent_fuse(
            block.dbb_1x1_kxk.conv2.weight,
            block.dbb_1x1_kxk.norm2,
        )
        sequential = _independent_compose(
            *sequential_first,
            *sequential_second,
        )

        avg_kernel = torch.zeros(7, 7, 3, 3)
        diagonal = torch.arange(7)
        avg_kernel[diagonal, diagonal] = 1.0 / 9.0
        average_second = _independent_fuse(avg_kernel, block.dbb_avg.avg_norm)
        average_first = _independent_fuse(
            block.dbb_avg.conv.weight,
            block.dbb_avg.norm.norm,
        )
        average = _independent_compose(*average_first, *average_second)

        independent_paths = {
            "origin": origin,
            "pointwise": pointwise,
            "sequential": sequential,
            "average": average,
        }
        replayed_paths = {
            name: F.conv2d(inputs, kernel, bias, padding=1)
            for name, (kernel, bias) in independent_paths.items()
        }
        independent_kernel = torch.stack(
            [independent_paths[name][0] for name in block.branch_names]
        ).sum(dim=0)
        independent_bias = torch.stack(
            [independent_paths[name][1] for name in block.branch_names]
        ).sum(dim=0)
        replayed_total = F.conv2d(inputs, independent_kernel, independent_bias, padding=1)
        local_paths = block.equivalent_path_kernels()
        local_kernel, local_bias = block.get_equivalent_kernel_bias()

        sequential_input = block.dbb_1x1_kxk.idconv1(inputs)
        observed_padded = block.dbb_1x1_kxk.norm1(sequential_input)
        norm = block.dbb_1x1_kxk.norm1.norm
        normalized = F.batch_norm(
            sequential_input,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=norm.momentum,
            eps=norm.eps,
        )
        expected_padded = F.pad(normalized, (1, 1, 1, 1))
        scale = norm.weight / (norm.running_var + norm.eps).sqrt()
        pad_values = (norm.bias - norm.running_mean * scale).reshape(1, -1, 1, 1)
        expected_padded[:, :, :1, :] = pad_values
        expected_padded[:, :, -1:, :] = pad_values
        expected_padded[:, :, :, :1] = pad_values
        expected_padded[:, :, :, -1:] = pad_values

    path_output_errors = {
        name: float((observed_paths[name] - replayed_paths[name]).abs().amax().item())
        for name in block.branch_names
    }
    path_kernel_errors = {
        name: float(
            (local_paths[name][0] - independent_paths[name][0]).abs().amax().item()
        )
        for name in block.branch_names
    }
    path_bias_errors = {
        name: float(
            (local_paths[name][1] - independent_paths[name][1]).abs().amax().item()
        )
        for name in block.branch_names
    }
    errors = {
        "path_output": path_output_errors,
        "path_kernel": path_kernel_errors,
        "path_bias": path_bias_errors,
        "branch_addition": float(
            (
                observed_total
                - torch.stack([observed_paths[name] for name in block.branch_names]).sum(0)
            )
            .abs()
            .amax()
            .item()
        ),
        "fused_output": float((observed_total - replayed_total).abs().amax().item()),
        "fused_kernel": float((local_kernel - independent_kernel).abs().amax().item()),
        "fused_bias": float((local_bias - independent_bias).abs().amax().item()),
        "bn_aware_border_padding": float(
            (observed_padded - expected_padded).abs().amax().item()
        ),
    }
    flat_errors = [
        *path_output_errors.values(),
        *path_kernel_errors.values(),
        *path_bias_errors.values(),
        *[
            float(value)
            for key, value in errors.items()
            if key not in {"path_output", "path_kernel", "path_bias"}
        ],
    ]
    return {
        "errors": errors,
        "maximum_error": max(flat_errors),
        "passed": max(flat_errors) <= OFFICIAL_TOLERANCE,
    }


def _fold_control_stem(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    folded = model if inplace else copy.deepcopy(model)
    if not isinstance(folded.stem, HybridConvStem):
        raise TypeError("Control stem folding requires HybridConvStem.")
    for stem_block in folded.stem.blocks:
        sequence = stem_block.block
        conv = sequence.conv
        norm = sequence.norm
        kernel, bias = fuse_conv_batch_norm(conv.weight, norm)
        replacement = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(kernel)
            replacement.bias.copy_(bias)
        sequence.conv = replacement
        sequence.norm = nn.Identity()
    return folded


def _deployment_structure(stem: nn.Module) -> Dict[str, object]:
    counts = {
        "conv2d": sum(isinstance(module, nn.Conv2d) for module in stem.modules()),
        "batch_norm": sum(
            isinstance(module, nn.BatchNorm2d) for module in stem.modules()
        ),
        "average_pool": sum(
            isinstance(module, nn.AvgPool2d) for module in stem.modules()
        ),
        "identity_conv": sum(
            isinstance(module, IdentityBasedConv1x1) for module in stem.modules()
        ),
        "bn_and_pad": sum(
            isinstance(module, BatchNormAndPad2d) for module in stem.modules()
        ),
    }
    counts["parameters"] = sum(parameter.numel() for parameter in stem.parameters())
    counts["passed"] = bool(
        counts["conv2d"] == 3
        and counts["batch_norm"] == 0
        and counts["average_pool"] == 0
        and counts["identity_conv"] == 0
        and counts["bn_and_pad"] == 0
    )
    return counts


def _branch_activation_summary(
    candidate: nn.Module,
    images: Tensor,
) -> Dict[str, object]:
    with torch.inference_mode():
        _, trace = candidate.stem.forward_with_trace(images)
    blocks: Dict[str, object] = {}
    all_active = True
    all_distinct = True
    for block_index in range(1, 4):
        paths = {
            name: trace[f"block{block_index}_path_{name}"].float()
            for name in DiverseBranchConvBN.branch_names
        }
        rms = {
            name: float(value.square().mean().sqrt().item())
            for name, value in paths.items()
        }
        pairwise_rms = {}
        names = tuple(paths)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                pairwise_rms[f"{left}__{right}"] = float(
                    (paths[left] - paths[right]).square().mean().sqrt().item()
                )
        active = all(math.isfinite(value) and value > 0.0 for value in rms.values())
        distinct = all(
            math.isfinite(value) and value > 1e-8 for value in pairwise_rms.values()
        )
        all_active = all_active and active
        all_distinct = all_distinct and distinct
        blocks[f"block{block_index}"] = {
            "path_rms": rms,
            "pairwise_difference_rms": pairwise_rms,
            "all_active": active,
            "all_distinct": distinct,
        }
    return {
        "blocks": blocks,
        "all_twelve_paths_active": all_active,
        "all_within_block_paths_distinct": all_distinct,
        "passed": all_active and all_distinct,
    }


def _gradient_diagnostics(
    *,
    config: Mapping[str, object],
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    seed: int,
    use_bf16: bool,
) -> Dict[str, object]:
    set_seed(seed)
    model = create_model(num_classes=5, model_config=config).to(images.device).train()
    model.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        logits, _ = _forward(model, images, metadata)
        loss = F.cross_entropy(logits.float(), labels)
    loss.backward()
    gradients = _path_gradient_summary(model)
    result = {
        "batch_size": int(images.size(0)),
        "loss": float(loss.detach().item()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients": gradients,
        "all_path_gradients": all(bool(value["passed"]) for value in gradients.values()),
    }
    result["passed"] = bool(
        result["logits_finite"]
        and result["loss_finite"]
        and result["all_path_gradients"]
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _deployment_parity(
    *,
    control: nn.Module,
    candidate: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Tuple[Dict[str, object], nn.Module, nn.Module]:
    control = control.cpu().eval()
    candidate = candidate.cpu().eval()
    folded_control = _fold_control_stem(control, inplace=False).eval()
    deployed_candidate = convert_dbb_modules_to_deploy(candidate, inplace=False).eval()
    control = control.to(images.device)
    candidate = candidate.to(images.device)
    folded_control = folded_control.to(images.device)
    deployed_candidate = deployed_candidate.to(images.device)

    def compare(
        source: nn.Module,
        converted: nn.Module,
        *,
        use_bf16: bool,
    ) -> Dict[str, object]:
        previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
        previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        if not use_bf16:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else torch.autocast(device_type="cuda", enabled=False)
        )
        try:
            with torch.inference_mode(), context:
                source_stem = source.stem(images)
                converted_stem = converted.stem(images)
                source_logits, _ = _forward(source, images, metadata)
                converted_logits, _ = _forward(converted, images, metadata)
            return {
                "stem_maximum_absolute_error": float(
                    (source_stem - converted_stem).abs().amax().item()
                ),
                "logit_maximum_absolute_error": float(
                    (source_logits - converted_logits).abs().amax().item()
                ),
                "argmax_mismatches": int(
                    torch.count_nonzero(
                        source_logits.argmax(dim=1) != converted_logits.argmax(dim=1)
                    ).item()
                ),
                "finite": bool(
                    torch.isfinite(source_logits).all()
                    and torch.isfinite(converted_logits).all()
                ),
                "autocast_bf16": bool(use_bf16),
                "tf32_disabled": not use_bf16,
            }
        finally:
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32

    candidate_fp32 = compare(candidate, deployed_candidate, use_bf16=False)
    candidate_bf16 = compare(candidate, deployed_candidate, use_bf16=True)
    control_fp32 = compare(control, folded_control, use_bf16=False)
    control_structure = _deployment_structure(folded_control.stem)
    candidate_structure = _deployment_structure(deployed_candidate.stem)
    parameter_match = (
        int(control_structure["parameters"]) == int(candidate_structure["parameters"])
    )
    result = {
        "candidate_fp32": candidate_fp32,
        "candidate_bf16": candidate_bf16,
        "control_fp32": control_fp32,
        "folded_control_structure": control_structure,
        "deployed_candidate_structure": candidate_structure,
        "stem_parameter_count_match": parameter_match,
        "conversion_device": "cpu_fp32",
    }
    result["passed"] = bool(
        candidate_fp32["stem_maximum_absolute_error"] <= FP32_DEPLOY_TOLERANCE
        and candidate_fp32["logit_maximum_absolute_error"] <= FP32_DEPLOY_TOLERANCE
        and candidate_fp32["argmax_mismatches"] == 0
        and candidate_fp32["finite"]
        and candidate_bf16["stem_maximum_absolute_error"] <= BF16_DEPLOY_TOLERANCE
        and candidate_bf16["logit_maximum_absolute_error"] <= BF16_DEPLOY_TOLERANCE
        and candidate_bf16["argmax_mismatches"] == 0
        and candidate_bf16["finite"]
        and control_fp32["stem_maximum_absolute_error"] <= FP32_DEPLOY_TOLERANCE
        and control_fp32["logit_maximum_absolute_error"] <= FP32_DEPLOY_TOLERANCE
        and control_fp32["argmax_mismatches"] == 0
        and control_fp32["finite"]
        and control_structure["passed"]
        and candidate_structure["passed"]
        and parameter_match
    )
    return result, folded_control, deployed_candidate


def _paired_inference_benchmark(
    *,
    control: nn.Module,
    candidate: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    warmup_iterations: int,
    timed_iterations: int,
) -> Dict[str, object]:
    control = control.to(images.device).eval()
    candidate = candidate.to(images.device).eval()

    def iteration(model: nn.Module) -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits, _ = _forward(model, images, metadata)
        return logits

    for _ in range(int(warmup_iterations)):
        iteration(control)
        iteration(candidate)
    timings = {"control": [], "candidate": []}
    outputs: Dict[str, Tensor] = {}
    for iteration_index in range(int(timed_iterations)):
        order = (
            (("control", control), ("candidate", candidate))
            if iteration_index % 2 == 0
            else (("candidate", candidate), ("control", control))
        )
        for name, model in order:
            torch.cuda.synchronize(images.device)
            started = time.perf_counter()
            outputs[name] = iteration(model)
            torch.cuda.synchronize(images.device)
            timings[name].append(float(time.perf_counter() - started))
    medians = {
        name: float(torch.tensor(values, dtype=torch.float64).median().item())
        for name, values in timings.items()
    }
    return {
        "seconds": timings,
        "median_seconds": medians,
        "runtime_ratio": medians["candidate"] / max(medians["control"], 1e-12),
        "finite": all(bool(torch.isfinite(value).all()) for value in outputs.values()),
    }


def _formal_gate(checks: Mapping[str, bool]) -> Dict[str, object]:
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    return {
        "formal_pair_permission": not failed,
        "checks": dict(checks),
        "failed_checks": failed,
        "validation_permission": False,
        "test_permission": False,
    }


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("DBB preflight requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("DBB preflight is locked to batch32 and FP32 batch2.")
    if int(args.timed_iterations) < 5:
        raise ValueError("DBB preflight requires at least five timed iterations.")
    output_dir = _prepare_output(Path(args.output_dir))
    official_root = Path(args.official_root).resolve()
    paths = {
        "resolved_config": Path(args.resolved_config).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "scratch_checkpoint": Path(args.scratch_checkpoint).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "declaration_summary": Path(args.declaration_summary).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "official_block": official_root / "diversebranchblock.py",
        "official_transforms": official_root / "dbb_transforms.py",
        "official_readme": official_root / "README.md",
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
    control_config = _control_config(source_model_config)
    candidate_config = _candidate_config(source_model_config)
    config_differences = sorted(
        key
        for key in set(control_config).union(candidate_config)
        if control_config.get(key) != candidate_config.get(key)
    )

    set_seed(int(args.seed))
    control = create_model(num_classes=5, model_config=control_config)
    control_cpu_rng = torch.random.get_rng_state().clone()
    control_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    set_seed(int(args.seed))
    candidate = create_model(num_classes=5, model_config=candidate_config)
    candidate_cpu_rng = torch.random.get_rng_state().clone()
    candidate_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    state_pairing = _state_pairing_summary(control, candidate)
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

    dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="dbb_a0_train_only_preflight",
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
    fp32_count = int(args.fp32_batch_size)
    fp32_images = images[:fp32_count]
    fp32_labels = labels[:fp32_count]
    fp32_metadata = {
        key: value[:fp32_count] for key, value in metadata.items()
    }

    official_replay = _official_replay(paths["official_block"], images_cpu)
    independent_equations = _independent_equation_replay()
    candidate = candidate.to(device).eval()
    branch_activation = _branch_activation_summary(candidate, fp32_images)
    candidate = candidate.cpu()
    torch.cuda.empty_cache()

    fp32_diagnostics = _gradient_diagnostics(
        config=candidate_config,
        images=fp32_images,
        labels=fp32_labels,
        metadata=fp32_metadata,
        seed=int(args.seed),
        use_bf16=False,
    )
    bf16_diagnostics = _gradient_diagnostics(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        use_bf16=True,
    )

    set_seed(int(args.seed))
    parity_control = create_model(num_classes=5, model_config=control_config)
    set_seed(int(args.seed))
    parity_candidate = create_model(num_classes=5, model_config=candidate_config)
    deployment, folded_control, deployed_candidate = _deployment_parity(
        control=parity_control,
        candidate=parity_candidate,
        images=fp32_images,
        metadata=fp32_metadata,
    )
    folded_control = folded_control.cpu()
    deployed_candidate = deployed_candidate.cpu()
    del parity_control, parity_candidate
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
    train_runtime_ratio = float(candidate_train_resource["median_seconds"]) / max(
        float(control_train_resource["median_seconds"]), 1e-12
    )
    vram_ratio = float(candidate_train_resource["peak_vram_gib"]) / max(
        float(control_train_resource["peak_vram_gib"]), 1e-12
    )
    inference_resource = _paired_inference_benchmark(
        control=folded_control,
        candidate=deployed_candidate,
        images=images,
        metadata=metadata,
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    folded_control = folded_control.cpu()
    deployed_candidate = deployed_candidate.cpu()
    torch.cuda.empty_cache()

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
    onnx_path = output_dir / "dbb_candidate_deployed_static_batch1.onnx"
    try:
        onnx = _onnx_compare(
            wrapper=_FullExport(deployed_candidate),
            inputs=(images_cpu[:1].float(), bbox[:1].float(), image_mask[:1].bool()),
            input_names=("images", "bbox", "image_mask"),
            path=onnx_path,
        )
        import onnx as onnx_library

        onnx_model = onnx_library.load(str(onnx_path))
        onnx["operators"] = sorted(
            {node.op_type for node in onnx_model.graph.node}
        )
    except Exception as error:
        onnx = _failed_export(onnx_path, error)
        onnx["operators"] = []

    official_repository = _git_repository_state(official_root)
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
        "config_only_changes_stem": config_differences == ["stem_architecture"],
        "candidate_stem_type": isinstance(candidate.stem, DiverseBranchConvStem),
        "control_stem_type": isinstance(control.stem, HybridConvStem),
        "nonstem_state_bit_exact": bool(state_pairing["all_nonstem_bit_exact"]),
        "origin_state_bit_exact": bool(state_pairing["all_origin_bit_exact"]),
        "candidate_only_state_present": bool(state_pairing["candidate_only_keys"]),
        "candidate_only_state_finite": bool(state_pairing["candidate_only_finite"]),
        "terminal_gamma_initialized_one": bool(state_pairing["terminal_gamma_one"]),
        "constructor_cpu_rng_equal": bool(rng["cpu_equal"]),
        "constructor_cuda_rng_equal": bool(rng["cuda_equal"]),
        "official_random_real_train_eval_replay": bool(official_replay["passed"]),
        "independent_equation_replay": bool(independent_equations["passed"]),
        "all_twelve_paths_active_distinct": bool(branch_activation["passed"]),
        "fp32_forward_backward_gradients": bool(fp32_diagnostics["passed"]),
        "bf16_forward_backward_gradients": bool(bf16_diagnostics["passed"]),
        "deployment_parity_structure": bool(deployment["passed"]),
        "train_runtime_ratio": math.isfinite(train_runtime_ratio)
        and train_runtime_ratio <= MAX_TRAIN_RUNTIME_RATIO,
        "candidate_vram_budget": float(candidate_train_resource["peak_vram_gib"])
        <= MAX_VRAM_GIB,
        "candidate_vram_ratio": math.isfinite(vram_ratio)
        and vram_ratio <= MAX_VRAM_RATIO,
        "inference_runtime_ratio": math.isfinite(
            float(inference_resource["runtime_ratio"])
        )
        and float(inference_resource["runtime_ratio"])
        <= MAX_INFERENCE_RUNTIME_RATIO,
        "resource_finite": bool(
            control_train_resource["finite"]
            and candidate_train_resource["finite"]
            and inference_resource["finite"]
        ),
        "onnx_succeeded": bool(onnx.get("succeeded", False)),
        "onnx_finite": bool(onnx.get("finite", False)),
        "onnx_error": float(onnx.get("maximum_absolute_error", 1e9)) <= 1e-4,
        "onnx_argmax": bool(onnx.get("argmax_match", False)),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = _formal_gate(checks)
    summary: Dict[str, object] = {
        "method": "diverse_branch_stem_a0_preflight",
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
            "control_parameters": sum(value.numel() for value in control.parameters()),
            "candidate_parameters": sum(value.numel() for value in candidate.parameters()),
            "config_differences": config_differences,
            "state_pairing": state_pairing,
            "rng": rng,
        },
        "data": {
            "fit_rows": len(dataset),
            "batch_size": int(images.size(0)),
            "labels": [int(value) for value in labels.cpu().tolist()],
            "dataloader": loader_summary,
        },
        "official_replay": official_replay,
        "independent_equations": independent_equations,
        "branch_activation": branch_activation,
        "fp32": fp32_diagnostics,
        "bf16": bf16_diagnostics,
        "deployment": deployment,
        "resource": {
            "control_train_step": control_train_resource,
            "candidate_train_step": candidate_train_resource,
            "train_runtime_ratio": train_runtime_ratio,
            "train_peak_vram_ratio": vram_ratio,
            "converted_inference": inference_resource,
        },
        "onnx": onnx,
        "gate": gate,
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
                "method": "diverse_branch_stem_a0_preflight",
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
