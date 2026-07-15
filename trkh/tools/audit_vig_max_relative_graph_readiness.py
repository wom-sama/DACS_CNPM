from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.inference.inference import load_checkpoint
from trkh.models.dynamic_graph_mixer import MaxRelativeDynamicGraphMixer
from trkh.models.model import (
    MultiHeadSelfAttention,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import (
    LIGHTING_CONDITIONS,
    _failed_export,
    _make_lighting_loader,
    _onnx_compare,
    _write_illumination_predictions,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _comparison,
    _forward_logits,
    _git_commit,
    _make_loader,
    _metadata_to_device,
    _predict,
    _prepare_output_dir,
    _rng_equal,
    _rng_snapshot,
    _rng_summary,
    _sha256,
    _tensor_sha256,
    _verify_sha256,
    _write_predictions,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_GRAPH_LAYERS = (2, 5)
EXPECTED_ADDED_PARAMETERS = 99_840
EXPECTED_BLOCKS = 8
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "9a0ad9f928e1915417c198b12dfeaaf07327734eb10d532566aa34f76382dd76"
LOCKED_VIG_COMMIT = "f90e129b645c3b1684fe07cd361cd557d0ad71f7"
LOCKED_VIG_SOURCE_SHA256 = {
    "torch_vertex.py": "c0b198b0f21fe20947de3317b3df500ffc76a224fe5fa46bc3c5585b5a465ef5",
    "torch_nn.py": "28a9b32f630b0774823c5ff315bdccad4ffd7e1affdb1dadd068e8e39388cff9",
    "torch_edge.py": "88bcd6b8bdeb628f2f9fb21e1dfe3affe31d833167402e9f12063544634cde53",
    "pyramid_vig.py": "ae596df75b019fad2e59080cf3a884276f65d695b3062dedad8cf3095fdd1089",
}
STATIC_EXPORT_BATCH_SIZE = 1
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.30
MAX_PEAK_VRAM_GIB = 3.25


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only readiness audit for ViG max-relative graph mixing. "
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
            "docs/TRKH_5CLASS_VIG_MAX_RELATIVE_GRAPH_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--official-vig-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_next_arch_sources_20260715\vision_gnn"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--graph-layers", type=str, default="2,5")
    parser.add_argument("--graph-bottleneck-dim", type=int, default=64)
    parser.add_argument("--graph-k", type=int, default=9)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    return parser.parse_args(argv)


def _candidate_config(
    source: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "dynamic_graph_mixer": True,
            "dynamic_graph_mixer_layers": str(args.graph_layers),
            "dynamic_graph_mixer_bottleneck_dim": int(args.graph_bottleneck_dim),
            "dynamic_graph_mixer_k": int(args.graph_k),
            "pretrained": False,
        }
    )
    return config


def _expected_state_additions(layers: Sequence[int]) -> set[str]:
    suffixes = (
        "norm.weight",
        "norm.bias",
        "input_projection.weight",
        "output_projection.weight",
        "output_projection.bias",
    )
    return {
        f"blocks.{int(layer) - 1}.dynamic_graph_mixer.{suffix}"
        for layer in layers
        for suffix in suffixes
    }


def _construct_models(
    checkpoint: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[nn.Module, nn.Module, Dict[str, object]]:
    source_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state")
    if not isinstance(source_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Keeper checkpoint lacks model_config/class_names.")
    if not isinstance(state, Mapping):
        raise ValueError("Keeper checkpoint lacks model_state.")
    num_classes = len(class_names)

    set_seed(int(args.seed), deterministic=True)
    control = create_model(num_classes=num_classes, model_config=source_config)
    control_constructor_rng = _rng_snapshot()
    load_model_state(control, dict(state), strict=True)

    set_seed(int(args.seed), deterministic=True)
    candidate = create_model(
        num_classes=num_classes,
        model_config=_candidate_config(source_config, args),
    )
    candidate_constructor_rng = _rng_snapshot()
    missing, unexpected = load_model_state(candidate, dict(state), strict=False)
    expected_additions = _expected_state_additions(EXPECTED_GRAPH_LAYERS)
    state_additions = set(candidate.state_dict()).difference(state)
    candidate_state = candidate.state_dict()
    existing_bit_exact = all(
        key in candidate_state
        and torch.equal(candidate_state[key].detach().cpu(), value.detach().cpu())
        for key, value in state.items()
    )
    control_parameters = sum(int(value.numel()) for value in control.parameters())
    candidate_parameters = sum(int(value.numel()) for value in candidate.parameters())
    return control.eval(), candidate.eval(), {
        "control_constructor_rng": _rng_summary(control_constructor_rng),
        "candidate_constructor_rng": _rng_summary(candidate_constructor_rng),
        "constructor_rng_equal": _rng_equal(
            control_constructor_rng,
            candidate_constructor_rng,
        ),
        "reported_missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "state_additions": sorted(state_additions),
        "expected_state_additions": sorted(expected_additions),
        "state_additions_exact": state_additions == expected_additions,
        "reported_missing_keys_exact": set(missing) == expected_additions,
        "existing_state_bit_exact": bool(existing_bit_exact),
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "control_parameters": control_parameters,
        "candidate_parameters": candidate_parameters,
        "added_parameters": candidate_parameters - control_parameters,
    }


def _configure_trainability(model: nn.Module, *, candidate: bool) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name in {"head.weight", "head.bias"}
        if candidate and ".dynamic_graph_mixer." in name:
            enabled = True
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    model.eval()
    return trainable


def _subset_state_sha256(model: nn.Module, *, candidate: bool) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("head."):
            continue
        if candidate and ".dynamic_graph_mixer." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tracked_state(model: nn.Module, *, candidate: bool) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("head.")
        or (candidate and ".dynamic_graph_mixer." in name)
    }


def _train_variant(
    *,
    name: str,
    prototype: nn.Module,
    candidate: bool,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[nn.Module, Dict[str, object]]:
    started = time.perf_counter()
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device)
    initial_state_sha = _state_sha256(model)
    initial_frozen_sha = _subset_state_sha256(model, candidate=candidate)
    initial_tracked = _tracked_state(model, candidate=candidate)
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"vig_graph_{name}_holdout",
        seed=int(args.seed) + 100,
    )
    unadapted_predictions = _predict(
        model=model,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
    )

    set_seed(int(args.seed), deterministic=True)
    train_loader, train_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"vig_graph_{name}_fit",
        seed=int(args.seed) + 200,
    )
    trainable_names = _configure_trainability(model, candidate=candidate)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[Dict[str, object]] = []
    ordered_indices: list[int] = []
    post_forward_rng: list[Dict[str, object]] = []
    gradient_seen = {key: False for key in trainable_names}
    all_gradients_finite = True
    named_parameters = dict(model.named_parameters())
    for batch_index, (images, targets, metadata) in enumerate(train_loader):
        if batch_index >= int(args.max_train_batches):
            break
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        sample_indices = metadata.get("sample_index")
        if not torch.is_tensor(sample_indices):
            raise ValueError("Training metadata is missing sample_index.")
        ordered_indices.extend(int(value) for value in sample_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = _forward_logits(model, images, metadata, device=device)
            loss = F.cross_entropy(logits.float(), targets)
        post_forward_rng.append(_rng_summary(_rng_snapshot()))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {name} loss at batch {batch_index}.")
        loss.backward()
        gradient_norm_squared = 0.0
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
            gradient_norm_squared += float(
                gradient.detach().float().square().sum().item()
            )
        optimizer.step()
        history.append(
            {
                "batch": int(batch_index),
                "rows": int(targets.numel()),
                "loss": float(loss.detach().item()),
                "gradient_norm": math.sqrt(gradient_norm_squared),
            }
        )
    if len(history) != int(args.max_train_batches):
        raise ValueError(
            f"Expected {args.max_train_batches} train batches, observed {len(history)}."
        )
    adapted_predictions = _predict(
        model=model,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    final_state_sha = _state_sha256(model)
    final_frozen_sha = _subset_state_sha256(model, candidate=candidate)
    final_state = model.state_dict()
    movement = {
        key: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final_state[key]),
            "changed": not torch.equal(value, final_state[key].detach().cpu()),
        }
        for key, value in initial_tracked.items()
    }
    model = model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, {
        "name": name,
        "train_batches": len(history),
        "train_rows": len(ordered_indices),
        "train_order_sha256": _ordered_index_sha256(ordered_indices),
        "post_forward_rng": post_forward_rng,
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
        "tracked_state_movement": movement,
        "history": history,
        "unadapted_predictions": unadapted_predictions,
        "adapted_predictions": adapted_predictions,
        "loader": {
            "train": train_loader_summary,
            "holdout": holdout_loader_summary,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _activate_equation_module(module: MaxRelativeDynamicGraphMixer) -> None:
    with torch.no_grad():
        module.output_projection.weight.copy_(
            torch.linspace(
                -0.03,
                0.03,
                steps=module.output_projection.weight.numel(),
            ).reshape_as(module.output_projection.weight)
        )
        module.output_projection.bias.copy_(
            torch.linspace(-0.01, 0.01, steps=module.dim)
        )


def _equation_check(
    module: MaxRelativeDynamicGraphMixer,
    tokens: Tensor,
) -> Dict[str, object]:
    residual, details = module(tokens, grid_size=(4, 4), return_details=True)
    projected = F.linear(module.norm(tokens), module.input_projection.weight)
    normalized = F.normalize(projected.detach().float(), p=2.0, dim=-1)
    squared_norm = normalized.square().sum(dim=-1, keepdim=True)
    distance = (
        squared_norm
        - 2.0 * normalized @ normalized.transpose(1, 2)
        + squared_norm.transpose(1, 2)
    ).clamp_min(0.0)
    expected_distance, expected_indices = torch.topk(
        -distance,
        k=module.k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    neighbors = projected.gather(
        1,
        expected_indices.reshape(int(tokens.size(0)), -1)
        .unsqueeze(-1)
        .expand(-1, -1, module.bottleneck_dim),
    ).reshape(
        int(tokens.size(0)),
        int(tokens.size(1)),
        module.k,
        module.bottleneck_dim,
    )
    message = (neighbors - projected.unsqueeze(2)).amax(dim=2)
    expected_residual = F.gelu(
        F.linear(
            torch.cat((projected, message), dim=-1),
            module.output_projection.weight,
            module.output_projection.bias,
        )
    )
    return {
        "residual_max_abs_error": float((residual - expected_residual).abs().amax()),
        "projected_max_abs_error": float(
            (details["projected_nodes"] - projected).abs().amax()
        ),
        "indices_exact": bool(
            torch.equal(details["neighbor_indices"], expected_indices)
        ),
        "distance_max_abs_error": float(
            (details["neighbor_distances"] + expected_distance).abs().amax()
        ),
        "message_max_abs_error": float(
            (details["max_relative_message"] - message).abs().amax()
        ),
        "finite": bool(torch.isfinite(residual).all()),
    }


def _equation_diagnostics(device: torch.device) -> Dict[str, object]:
    module = MaxRelativeDynamicGraphMixer(dim=32, bottleneck_dim=8, k=9).eval()
    _activate_equation_module(module)
    tokens = torch.linspace(-1.2, 1.3, steps=2 * 16 * 32).reshape(2, 16, 32)
    fp32 = _equation_check(module, tokens)
    bf16_module = copy.deepcopy(module).to(device=device, dtype=torch.bfloat16)
    bf16 = _equation_check(
        bf16_module,
        tokens.to(device=device, dtype=torch.bfloat16),
    )
    return {"fp32": fp32, "bf16": bf16}


def _gradient_groups() -> Dict[str, list[str]]:
    groups: Dict[str, list[str]] = {"classifier": ["head.weight", "head.bias"]}
    for layer in EXPECTED_GRAPH_LAYERS:
        prefix = f"blocks.{int(layer) - 1}.dynamic_graph_mixer"
        groups[f"layer{layer}_norm"] = [
            f"{prefix}.norm.weight",
            f"{prefix}.norm.bias",
        ]
        groups[f"layer{layer}_input_projection"] = [
            f"{prefix}.input_projection.weight"
        ]
        groups[f"layer{layer}_output_projection"] = [
            f"{prefix}.output_projection.weight",
            f"{prefix}.output_projection.bias",
        ]
    return groups


def _gradient_diagnostics(
    *,
    prototype: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    count = int(images.size(0))
    value = images.detach().clone().to(device).requires_grad_(True)
    targets = torch.arange(count, device=device, dtype=torch.long) % 5
    metadata_value = _metadata_to_device(metadata, device=device, count=count)
    model.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with context:
        logits = _forward_logits(model, value, metadata_value, device=device)
        loss = F.cross_entropy(logits.float(), targets)
    loss.backward()
    named = dict(model.named_parameters())
    groups = {}
    for group, names in _gradient_groups().items():
        gradients = [named[name].grad for name in names if name in named]
        groups[group] = {
            "parameters": names,
            "finite": bool(gradients)
            and all(
                value is not None and bool(torch.isfinite(value).all())
                for value in gradients
            ),
            "nonzero_elements": sum(
                int(torch.count_nonzero(value).item())
                for value in gradients
                if value is not None
            ),
            "passed": len(gradients) == len(names)
            and all(
                value is not None
                and bool(torch.isfinite(value).all())
                and int(torch.count_nonzero(value).item()) > 0
                for value in gradients
            ),
        }
    input_gradient = value.grad
    output = {
        "batch_size": count,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "float32",
        "loss": float(loss.detach().item()),
        "logits_loss_finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
        "groups": groups,
        "all_parameter_families": all(bool(row["passed"]) for row in groups.values()),
        "input_gradient_finite_nonzero": bool(
            input_gradient is not None
            and torch.isfinite(input_gradient).all()
            and int(torch.count_nonzero(input_gradient).item()) > 0
        ),
    }
    del model, value
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _benchmark(
    *,
    prototype: nn.Module,
    candidate: bool,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
    seed: int,
) -> Dict[str, object]:
    set_seed(int(seed), deterministic=True)
    model = copy.deepcopy(prototype).to(device)
    _configure_trainability(model, candidate=candidate)

    def iteration() -> tuple[Tensor, Tensor]:
        model.zero_grad(set_to_none=True)
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
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _graph_snapshot(
    *,
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
) -> list[Dict[str, object]]:
    model.eval()
    metadata_value = _metadata_to_device(metadata, device=device, count=int(images.size(0)))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _forward_logits(model, images.to(device), metadata_value, device=device)
    rows = []
    for layer in EXPECTED_GRAPH_LAYERS:
        module = model.blocks[layer - 1].dynamic_graph_mixer
        if module is None:
            raise ValueError(f"Missing graph module in layer {layer}.")
        trace = module.trace()
        rows.append(
            {
                "layer": layer,
                **{
                    key: float(value.detach().float().item())
                    for key, value in trace.items()
                },
            }
        )
    return rows


class _GraphEquationExportWrapper(nn.Module):
    def __init__(self, module: MaxRelativeDynamicGraphMixer) -> None:
        super().__init__()
        self.module = copy.deepcopy(module).cpu().eval()
        self.eval()

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.module(patch_tokens, grid_size=(4, 4))


class _GraphFullExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()
        self.eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def _export_diagnostics(
    *,
    candidate: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    module = candidate.blocks[EXPECTED_GRAPH_LAYERS[0] - 1].dynamic_graph_mixer
    patch_tokens = torch.linspace(-1.0, 1.0, steps=1 * 16 * module.dim).reshape(
        1, 16, module.dim
    )
    isolated_path = output_dir / "vig_max_relative_graph_equation.onnx"
    try:
        isolated = _onnx_compare(
            wrapper=_GraphEquationExportWrapper(module),
            inputs=(patch_tokens,),
            input_names=("patch_tokens",),
            path=isolated_path,
        )
    except Exception as error:
        isolated = _failed_export(isolated_path, error)

    cpu_images = images[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    bbox_value = metadata.get("bbox")
    if not torch.is_tensor(bbox_value):
        bbox_value = torch.zeros(STATIC_EXPORT_BATCH_SIZE, 8)
    cpu_bbox = bbox_value[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    mask_value = metadata.get("image_mask")
    if not torch.is_tensor(mask_value):
        mask_value = torch.ones(
            STATIC_EXPORT_BATCH_SIZE,
            int(cpu_images.size(2)),
            int(cpu_images.size(3)),
            dtype=torch.bool,
        )
    cpu_mask = mask_value[:STATIC_EXPORT_BATCH_SIZE].detach().bool().cpu()
    full_path = output_dir / "vig_max_relative_graph_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_GraphFullExportWrapper(candidate),
            inputs=(cpu_images, cpu_bbox, cpu_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as error:
        full = _failed_export(full_path, error)
    return {"isolated": isolated, "full": full}


def _pre_adaptation_diagnostics(
    *,
    control: nn.Module,
    candidate: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    count = 2
    images = images_cpu[:count].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=count)
    control_forward = copy.deepcopy(control).to(device).eval()
    candidate_forward = copy.deepcopy(candidate).to(device).eval()
    set_seed(int(args.seed), deterministic=True)
    with torch.inference_mode():
        control_logits = _forward_logits(control_forward, images, metadata, device=device)
    control_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    with torch.inference_mode():
        candidate_logits = _forward_logits(candidate_forward, images, metadata, device=device)
    candidate_rng = _rng_snapshot()
    exact_logit_delta = float((control_logits - candidate_logits).abs().amax().item())
    del control_forward, candidate_forward
    gc.collect()
    torch.cuda.empty_cache()

    model = copy.deepcopy(candidate).to(device).eval()
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    with torch.inference_mode():
        features = model.forward_features(
            images[:1],
            bbox_token_prior=bbox[:1] if torch.is_tensor(bbox) else None,
            image_valid_mask=image_mask[:1] if torch.is_tensor(image_mask) else None,
            return_attention=True,
            return_trace=True,
        )
        pruned = model.forward_features(
            images[:1],
            bbox_token_prior=bbox[:1] if torch.is_tensor(bbox) else None,
            image_valid_mask=image_mask[:1] if torch.is_tensor(image_mask) else None,
            return_trace=True,
        )
    trace = features.get("trace", {})
    first_block = model.blocks[EXPECTED_GRAPH_LAYERS[0] - 1]
    prefix_count = int(model.num_prefix_tokens)
    token_probe = torch.linspace(
        -1.0,
        1.0,
        steps=2 * (prefix_count + 16) * int(model.embed_dim),
        device=device,
    ).reshape(2, prefix_count + 16, int(model.embed_dim))
    with torch.inference_mode():
        probe_output = first_block.apply_dynamic_graph_mixer(
            token_probe,
            prefix_count=prefix_count,
            grid_size=(4, 4),
            patch_indices=None,
        )
    spatial_attentions = features.get("attentions", {})
    pruning_layers = [
        int(value["layer"].item())
        for value in pruned.get("trace", {}).get("pruning", [])
    ]
    output = {
        "control_forward_rng": _rng_summary(control_rng),
        "candidate_forward_rng": _rng_summary(candidate_rng),
        "forward_rng_equal": _rng_equal(control_rng, candidate_rng),
        "initial_logit_max_abs_delta": exact_logit_delta,
        "equation": _equation_diagnostics(device),
        "graph_layers": [
            index + 1
            for index, block in enumerate(model.blocks)
            if block.dynamic_graph_mixer is not None
        ],
        "trace_graph_layers": (
            [int(value) for value in trace["dynamic_graph_mixer_layers"].tolist()]
            if torch.is_tensor(trace.get("dynamic_graph_mixer_layers"))
            else []
        ),
        "initial_residual_ratios": [
            float(value)
            for value in trace.get("dynamic_graph_residual_norm_ratio", []).tolist()
        ]
        if torch.is_tensor(trace.get("dynamic_graph_residual_norm_ratio"))
        else [],
        "spatial_mhsa_layers": [
            index + 1
            for index, block in enumerate(model.blocks)
            if isinstance(block.attn, MultiHeadSelfAttention)
        ],
        "spatial_attention_layers": [int(value) + 1 for value in spatial_attentions],
        "pruning_layers": pruning_layers,
        "prefix_direct_max_abs_delta": float(
            (probe_output[:, :prefix_count] - token_probe[:, :prefix_count])
            .abs()
            .amax()
            .item()
        ),
        "probe_total_max_abs_delta": float(
            (probe_output - token_probe).abs().amax().item()
        ),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _post_adaptation_diagnostics(
    *,
    control: nn.Module,
    candidate: nn.Module,
    candidate_training: Mapping[str, object],
    images_cpu: Tensor,
    targets_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    fp32_count = int(args.fp32_batch_size)
    fp32 = _gradient_diagnostics(
        prototype=candidate,
        images=images_cpu[:fp32_count],
        metadata=metadata_cpu,
        device=device,
        amp_dtype=None,
    )
    bf16 = _gradient_diagnostics(
        prototype=candidate,
        images=images_cpu,
        metadata=metadata_cpu,
        device=device,
        amp_dtype=amp_dtype,
    )
    images = images_cpu.to(device)
    targets = targets_cpu.to(device=device, dtype=torch.long)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=int(images.size(0)))
    control_benchmark = _benchmark(
        prototype=control,
        candidate=False,
        images=images,
        targets=targets,
        metadata=metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
        seed=int(args.seed),
    )
    candidate_benchmark = _benchmark(
        prototype=candidate,
        candidate=True,
        images=images,
        targets=targets,
        metadata=metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
        seed=int(args.seed),
    )
    runtime_ratio = float(
        candidate_benchmark["median_seconds"]
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )

    model = copy.deepcopy(candidate).to(device).eval()
    graph_trace = _graph_snapshot(
        model=model,
        images=images_cpu,
        metadata=metadata_cpu,
        device=device,
    )
    metadata_device = _metadata_to_device(
        metadata_cpu,
        device=device,
        count=int(images_cpu.size(0)),
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        baseline_logits = _forward_logits(
            model,
            images_cpu.to(device),
            metadata_device,
            device=device,
        ).float()
        ablations = []
        for layer in EXPECTED_GRAPH_LAYERS:
            module = model.blocks[layer - 1].dynamic_graph_mixer

            def bypass(_, __, output):
                return torch.zeros_like(output)

            handle = module.register_forward_hook(bypass)
            ablated = _forward_logits(
                model,
                images_cpu.to(device),
                metadata_device,
                device=device,
            ).float()
            handle.remove()
            ablations.append(
                {
                    "layer": layer,
                    "logit_max_abs_delta": float(
                        (baseline_logits - ablated).abs().amax().item()
                    ),
                }
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    movement = {
        key: value
        for key, value in candidate_training["tracked_state_movement"].items()
        if ".dynamic_graph_mixer." in key
    }
    export = _export_diagnostics(
        candidate=candidate,
        images=images_cpu,
        metadata=metadata_cpu,
        output_dir=output_dir,
    )
    return {
        "fp32_gradients": fp32,
        "bf16_gradients": bf16,
        "control_benchmark": control_benchmark,
        "candidate_benchmark": candidate_benchmark,
        "runtime_ratio": runtime_ratio,
        "peak_vram_gib": float(candidate_benchmark["peak_vram_gib"]),
        "graph_trace": graph_trace,
        "layer_ablation": ablations,
        "graph_state_movement": movement,
        "export": export,
    }


def assess_vig_graph_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    unadapted: Mapping[str, object],
    adapted: Mapping[str, object],
    illumination: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    unadapted_delta = unadapted["delta"]
    unadapted_transitions = unadapted["transitions"]
    adapted_delta = adapted["delta"]
    adapted_transitions = adapted["transitions"]
    unadapted_checks = {
        "macro_f1_delta_gte_minus_0p010": float(unadapted_delta["macro_f1"]) >= -0.010,
        "class1_recall_delta_gte_minus_0p020": float(
            unadapted_delta["class1_recall"]
        )
        >= -0.020,
        "net_tp_breaks_at_most_3": (
            int(unadapted_transitions["focus_tp_break"])
            - int(unadapted_transitions["focus_fn_rescue"])
            <= 3
        ),
    }
    adapted_checks = {
        "changed_decisions_gte_2": int(adapted_transitions["changed"]) >= 2,
        "macro_f1_delta_gte_minus_0p002": float(adapted_delta["macro_f1"]) >= -0.002,
        "class1_f1_delta_gte_0p002": float(adapted_delta["class1_f1"]) >= 0.002,
        "class1_precision_delta_gte_0p005": float(
            adapted_delta["class1_precision"]
        )
        >= 0.005,
        "class1_recall_delta_gte_minus_0p010": float(
            adapted_delta["class1_recall"]
        )
        >= -0.010,
        "restricted_focus_fp_reduction_gte_2": int(
            adapted_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "tp_breaks_lte_fn_rescues": int(adapted_transitions["focus_tp_break"])
        <= int(adapted_transitions["focus_fn_rescue"]),
        "corrections_gt_harms": int(adapted_transitions["candidate_correction"])
        > int(adapted_transitions["candidate_harm"]),
        "new_3_to_2_harms_lte_2": int(
            adapted_transitions["new_nonfocus_3_to_2_harms"]
        )
        <= 2,
        "maximum_nonfocus_f1_drop_lte_0p015": float(
            adapted["maximum_nonfocus_f1_drop"]
        )
        <= 0.015,
    }
    illumination = list(illumination)
    precision_deltas = [float(row["delta"]["class1_precision"]) for row in illumination]
    recall_deltas = [float(row["delta"]["class1_recall"]) for row in illumination]
    macro_deltas = [float(row["delta"]["macro_f1"]) for row in illumination]
    total_removals = sum(
        int(row["transitions"]["focus_fp_remove_correct"]) for row in illumination
    )
    total_creations = sum(
        int(row["transitions"]["focus_fp_create"]) for row in illumination
    )
    illumination_checks = {
        "three_lighting_conditions_exact": len(illumination) == 3,
        "lighting_macro_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in macro_deltas
        )
        >= 2,
        "lighting_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in precision_deltas
        )
        >= 2,
        "worst_lighting_precision_delta_gte_minus_0p010": min(
            precision_deltas, default=-math.inf
        )
        >= -0.010,
        "worst_lighting_recall_delta_gte_minus_0p030": min(
            recall_deltas, default=-math.inf
        )
        >= -0.030,
        "aggregate_focus_fp_removals_gt_creations": total_removals > total_creations,
        "lighting_net_tp_breaks_lte_3_each": all(
            int(row["transitions"]["focus_tp_break"])
            - int(row["transitions"]["focus_fn_rescue"])
            <= 3
            for row in illumination
        ),
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **unadapted_checks,
        **adapted_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "unadapted_checks": unadapted_checks,
        "adapted_checks": adapted_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_smoke_authorized": not failed,
        "full_train_authorized": False,
    }


def _compact_variant(payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"unadapted_predictions", "adapted_predictions", "post_forward_rng"}
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked ViG graph Stage A requires CUDA.")
    locked_config_exact = bool(
        int(args.batch_size) == 32
        and int(args.fp32_batch_size) == 5
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == 30
        and math.isclose(float(args.learning_rate), 3e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 0.01, rel_tol=0.0, abs_tol=1e-12)
        and int(args.focus_class) == 1
        and str(args.graph_layers).replace(" ", "") == "2,5"
        and int(args.graph_bottleneck_dim) == 64
        and int(args.graph_k) == 9
        and int(args.benchmark_repeats) == 3
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )
    if not locked_config_exact:
        raise ValueError("Stage A arguments differ from the precommitted ViG protocol.")

    output_dir = _prepare_output_dir(args.output_dir)
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
    }
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
        "protocol": _verify_sha256(paths["protocol"], LOCKED_PROTOCOL_SHA256, "protocol"),
    }
    official_root = Path(args.official_vig_root).resolve()
    official_files = {
        "torch_vertex.py": official_root / "vig_pytorch" / "gcn_lib" / "torch_vertex.py",
        "torch_nn.py": official_root / "vig_pytorch" / "gcn_lib" / "torch_nn.py",
        "torch_edge.py": official_root / "vig_pytorch" / "gcn_lib" / "torch_edge.py",
        "pyramid_vig.py": official_root / "vig_pytorch" / "pyramid_vig.py",
    }
    official_commit = _git_commit(official_root)
    if official_commit != LOCKED_VIG_COMMIT:
        raise ValueError(
            f"Official ViG commit mismatch: {official_commit} != {LOCKED_VIG_COMMIT}"
        )
    official_hashes = {
        name: _verify_sha256(path, LOCKED_VIG_SOURCE_SHA256[name], name)
        for name, path in official_files.items()
    }

    repo_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "auditor": Path(__file__).resolve(),
        "module": repo_root / "trkh" / "models" / "dynamic_graph_mixer.py",
        "model": repo_root / "trkh" / "models" / "model.py",
        "config": repo_root / "trkh" / "core" / "config.py",
        "trainer": repo_root / "trkh" / "training" / "train.py",
        "launcher": repo_root / "scripts" / "run_trkh_5class_attention_views_v8.ps1",
        "audit_launcher": repo_root / "scripts" / "run_trkh_vig_graph_readiness.ps1",
    }
    implementation_hashes = {
        key: _sha256(value) for key, value in implementation_paths.items()
    }

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
    amp_dtype = _amp_dtype(device)
    if amp_dtype != torch.bfloat16:
        raise RuntimeError("Locked ViG graph Stage A requires BF16-capable CUDA.")

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Keeper class order is invalid.")
    source_config = checkpoint.get("model_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("Keeper model config is invalid.")
    if bool(source_config.get("dynamic_graph_mixer", False)):
        raise ValueError("Keeper unexpectedly already uses dynamic graph mixing.")
    for conflict in (
        "visual_contrast_attention",
        "cross_covariance_attention",
    ):
        if bool(source_config.get(conflict, False)):
            raise ValueError(f"Keeper {conflict} conflicts with ViG graph mixing.")
    control, candidate, construction = _construct_models(checkpoint, args)

    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        paths["data"],
        class_name_mode="raw",
        expected_num_classes=len(class_names),
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
    for row, path in zip(rows, sample_paths):
        if row.image_path != path:
            raise ValueError(f"Dataset/CIDT path mismatch at {row.sample_index}.")
    transform = _build_eval_transform(semantics)

    fold = int(args.fold)
    fit_indices = [row.sample_index for row in rows if row.fold != fold]
    holdout_indices = [row.sample_index for row in rows if row.fold == fold]
    fit_sources = {rows[index].source_stem for index in fit_indices}
    holdout_sources = {rows[index].source_stem for index in holdout_indices}
    source_overlap = fit_sources.intersection(holdout_sources)
    if source_overlap:
        raise ValueError("Fit/holdout source groups overlap.")
    shuffled_fit = np.asarray(fit_indices, dtype=np.int64)
    np.random.default_rng(int(args.seed)).shuffle(shuffled_fit)
    train_rows = int(args.max_train_batches) * int(args.batch_size)
    train_indices = shuffled_fit[:train_rows].tolist()
    if len(train_indices) != 960 or len(holdout_indices) != EXPECTED_FOLD_COUNTS[0]:
        raise ValueError("Locked train/holdout row counts differ.")

    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="vig_graph_stage_a_resource",
        seed=int(args.seed) + 300,
    )
    images_cpu, targets_cpu, metadata_cpu = next(iter(resource_loader))
    pre = _pre_adaptation_diagnostics(
        control=control,
        candidate=candidate,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        args=args,
        device=device,
    )

    control_model, control_result = _train_variant(
        name="control",
        prototype=control,
        candidate=False,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    candidate_model, candidate_result = _train_variant(
        name="candidate",
        prototype=candidate,
        candidate=True,
        base_dataset=base_dataset,
        transform=transform,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    for result in (control_result, candidate_result):
        for key in ("unadapted_predictions", "adapted_predictions"):
            observed = [int(row["sample_index"]) for row in result[key]]
            if observed != holdout_indices:
                raise ValueError(f"{result['name']} {key} holdout order changed.")
    unadapted = _comparison(
        control_rows=control_result["unadapted_predictions"],
        candidate_rows=candidate_result["unadapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    adapted = _comparison(
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )

    condition_predictions: Dict[str, Dict[str, Sequence[Mapping[str, object]]]] = {}
    illumination = []
    control_model = control_model.to(device).eval()
    candidate_model = candidate_model.to(device).eval()
    for condition_index, (condition, brightness, contrast) in enumerate(
        LIGHTING_CONDITIONS
    ):
        loader, _ = _make_lighting_loader(
            base_dataset=base_dataset,
            transform=transform,
            indices=holdout_indices,
            brightness=brightness,
            contrast=contrast,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=f"vig_graph_{condition}",
            seed=int(args.seed) + 500 + condition_index,
        )
        control_rows = _predict(
            model=control_model,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        candidate_rows = _predict(
            model=candidate_model,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        if [int(row["sample_index"]) for row in control_rows] != holdout_indices:
            raise ValueError(f"{condition} control order changed.")
        if [int(row["sample_index"]) for row in candidate_rows] != holdout_indices:
            raise ValueError(f"{condition} candidate order changed.")
        condition_predictions[condition] = {
            "control": control_rows,
            "candidate": candidate_rows,
        }
        comparison = _comparison(
            control_rows=control_rows,
            candidate_rows=candidate_rows,
            num_classes=len(class_names),
            focus_class=int(args.focus_class),
        )
        comparison["condition"] = condition
        comparison["brightness"] = brightness
        comparison["contrast"] = contrast
        illumination.append(comparison)
    control_model = control_model.cpu().eval()
    candidate_model = candidate_model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()

    post = _post_adaptation_diagnostics(
        control=control_model,
        candidate=candidate_model,
        candidate_training=candidate_result,
        images_cpu=images_cpu,
        targets_cpu=targets_cpu,
        metadata_cpu=metadata_cpu,
        args=args,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )
    movement = post["graph_state_movement"]
    equation = pre["equation"]
    graph_trace = post["graph_trace"]
    expected_trainable = {
        "head.weight",
        "head.bias",
        *{
            key
            for layer in EXPECTED_GRAPH_LAYERS
            for key in (
                f"blocks.{layer - 1}.dynamic_graph_mixer.norm.weight",
                f"blocks.{layer - 1}.dynamic_graph_mixer.norm.bias",
                f"blocks.{layer - 1}.dynamic_graph_mixer.input_projection.weight",
                f"blocks.{layer - 1}.dynamic_graph_mixer.output_projection.weight",
                f"blocks.{layer - 1}.dynamic_graph_mixer.output_projection.bias",
            )
        },
    }
    equation_errors = [
        float(equation[precision][key])
        for precision in ("fp32", "bf16")
        for key in (
            "residual_max_abs_error",
            "projected_max_abs_error",
            "distance_max_abs_error",
            "message_max_abs_error",
        )
    ]
    structural_checks = {
        "locked_config_exact": locked_config_exact,
        "locked_source_and_inputs": True,
        "train_only_provenance": True,
        "all_rows_and_fold_counts_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "fit_holdout_source_overlap_zero": len(source_overlap) == 0,
        "candidate_state_additions_exact_10": bool(construction["state_additions_exact"]),
        "candidate_reported_missing_keys_exact_10": bool(
            construction["reported_missing_keys_exact"]
        ),
        "candidate_unexpected_keys_zero": not construction["unexpected_keys"],
        "candidate_existing_state_bit_exact": bool(construction["existing_state_bit_exact"]),
        "added_parameters_exact_99840": int(construction["added_parameters"])
        == EXPECTED_ADDED_PARAMETERS,
        "constructor_rng_equal": bool(construction["constructor_rng_equal"]),
        "forward_rng_equal": bool(pre["forward_rng_equal"]),
        "initial_logits_exact": float(pre["initial_logit_max_abs_delta"]) == 0.0,
        "graph_layers_exact_2_5": pre["graph_layers"] == list(EXPECTED_GRAPH_LAYERS),
        "trace_graph_layers_exact_2_5": pre["trace_graph_layers"]
        == list(EXPECTED_GRAPH_LAYERS),
        "initial_residual_zero_each": pre["initial_residual_ratios"] == [0.0, 0.0],
        "all_eight_spatial_mhsa_blocks": pre["spatial_mhsa_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "spatial_attention_all_eight_layers": pre["spatial_attention_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "pruning_after_layers_2_5": pre["pruning_layers"]
        == list(EXPECTED_GRAPH_LAYERS),
        "prefix_direct_delta_zero": float(pre["prefix_direct_max_abs_delta"]) == 0.0,
        "zero_init_probe_exact": float(pre["probe_total_max_abs_delta"]) == 0.0,
        "fp32_bf16_equation_indices_exact": bool(
            equation["fp32"]["indices_exact"] and equation["bf16"]["indices_exact"]
        ),
        "equation_errors_lte_0p02": max(equation_errors) <= 0.02,
        "fp32_all_gradients_finite_nonzero": bool(
            post["fp32_gradients"]["all_parameter_families"]
            and post["fp32_gradients"]["input_gradient_finite_nonzero"]
        ),
        "bf16_all_gradients_finite_nonzero": bool(
            post["bf16_gradients"]["all_parameter_families"]
            and post["bf16_gradients"]["input_gradient_finite_nonzero"]
        ),
        "matched_train_order": control_result["train_order_sha256"]
        == candidate_result["train_order_sha256"]
        == _ordered_index_sha256(train_indices),
        "matched_train_budget_30x32": control_result["train_batches"]
        == candidate_result["train_batches"]
        == 30
        and control_result["train_rows"] == candidate_result["train_rows"] == 960,
        "matched_forward_rng_checkpoints": control_result["post_forward_rng"]
        == candidate_result["post_forward_rng"],
        "control_trainable_head_only": control_result["trainable_parameters"]
        == ["head.weight", "head.bias"],
        "candidate_trainable_head_plus_graph_only": set(
            candidate_result["trainable_parameters"]
        )
        == expected_trainable,
        "both_training_gradients_finite_seen": bool(
            control_result["all_gradients_finite"]
            and candidate_result["all_gradients_finite"]
            and control_result["all_trainable_gradients_seen"]
            and candidate_result["all_trainable_gradients_seen"]
        ),
        "frozen_existing_state_bit_exact": bool(
            control_result["frozen_state_bit_exact"]
            and candidate_result["frozen_state_bit_exact"]
        ),
        "all_graph_tensors_moved": len(movement) == 10
        and all(bool(value["changed"]) for value in movement.values()),
        "residual_ratio_each_in_0p005_0p20": len(graph_trace) == 2
        and all(
            0.005 <= float(row["residual_norm_ratio"]) <= 0.20
            for row in graph_trace
        ),
        "nonself_neighbors_gte_4_each": len(graph_trace) == 2
        and all(float(row["nonself_neighbor_count"]) >= 4.0 for row in graph_trace),
        "nonlocal_neighbor_fraction_gte_0p20_each": len(graph_trace) == 2
        and all(
            float(row["nonlocal_neighbor_fraction"]) >= 0.20
            for row in graph_trace
        ),
        "neighbor_selection_entropy_gte_0p20_each": len(graph_trace) == 2
        and all(
            float(row["neighbor_selection_entropy"]) >= 0.20
            for row in graph_trace
        ),
        "both_layer_ablations_material_1e4": all(
            float(row["logit_max_abs_delta"]) >= 1e-4
            for row in post["layer_ablation"]
        ),
        "isolated_onnx_error_lte_1e5": bool(post["export"]["isolated"]["succeeded"])
        and float(post["export"]["isolated"]["maximum_absolute_error"])
        <= MAX_ONNX_ERROR,
        "full_onnx_error_lte_1e5_argmax_match": bool(
            post["export"]["full"]["succeeded"]
        )
        and float(post["export"]["full"]["maximum_absolute_error"])
        <= MAX_ONNX_ERROR
        and bool(post["export"]["full"]["argmax_match"]),
        "runtime_ratio_lte_1p30": float(post["runtime_ratio"])
        <= float(args.max_runtime_ratio),
        "peak_vram_lte_3p25_gib": float(post["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
    }
    gate = assess_vig_graph_stage_a(
        structural_checks=structural_checks,
        unadapted=unadapted,
        adapted=adapted,
        illumination=illumination,
    )
    predictions_path = output_dir / "holdout_predictions.csv"
    _write_predictions(
        predictions_path,
        rows=rows,
        control=control_result,
        candidate=candidate_result,
    )
    illumination_path = output_dir / "illumination_predictions.csv"
    _write_illumination_predictions(
        illumination_path,
        rows=rows,
        condition_rows=condition_predictions,
    )
    summary: Dict[str, object] = {
        "method": "vig_max_relative_graph",
        "protocol_stage": "A_train_only_functional_resource_decision_readiness",
        "test_data_used": False,
        "validation_data_used": False,
        "sources": {
            **{key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_vig_root": str(official_root),
            "official_vig_commit": official_commit,
            "official_files": {key: str(value) for key, value in official_files.items()},
            "official_sha256": official_hashes,
            "repository_commit": _git_commit(repo_root),
            "implementation": {
                key: str(value) for key, value in implementation_paths.items()
            },
            "implementation_sha256": implementation_hashes,
        },
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
        "config": {
            "batch_size": int(args.batch_size),
            "fp32_batch_size": int(args.fp32_batch_size),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "max_train_batches": int(args.max_train_batches),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "focus_class": int(args.focus_class),
            "graph_layers": str(args.graph_layers),
            "graph_bottleneck_dim": int(args.graph_bottleneck_dim),
            "graph_k": int(args.graph_k),
        },
        "dataset": {
            "rows": len(rows),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "fold_counts": list(EXPECTED_FOLD_COUNTS),
            "fit_rows": len(fit_indices),
            "train_budget_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "fit_sources": len(fit_sources),
            "holdout_sources": len(holdout_sources),
            "source_overlap": len(source_overlap),
            "train_order_sha256": _ordered_index_sha256(train_indices),
            "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
            "resource_loader": resource_loader_summary,
        },
        "construction": construction,
        "pre_adaptation": pre,
        "post_adaptation": post,
        "control_training": _compact_variant(control_result),
        "candidate_training": _compact_variant(candidate_result),
        "unadapted_candidate_vs_control": unadapted,
        "adapted_candidate_vs_control": adapted,
        "illumination_candidate_vs_control": illumination,
        "gate": gate,
        "artifacts": {
            "holdout_predictions": predictions_path.name,
            "holdout_predictions_sha256": _sha256(predictions_path),
            "illumination_predictions": illumination_path.name,
            "illumination_predictions_sha256": _sha256(illumination_path),
            "isolated_onnx": post["export"]["isolated"]["path"],
            "full_onnx": post["export"]["full"]["path"],
        },
        "guardrail": (
            "A pass authorizes one keeper-initialized no-test validation smoke only. "
            "It never authorizes test access, a full train, or a sweep."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# ViG Max-Relative Graph Stage-A Readiness",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage-B smoke authorized: `{gate['stage_b_smoke_authorized']}`",
        f"- Failed checks: `{gate['failed_checks']}`",
        f"- Runtime ratio / peak VRAM GiB: `{post['runtime_ratio']:.6f}/{post['peak_vram_gib']:.6f}`",
        f"- Full ONNX max error: `{post['export']['full']['maximum_absolute_error']:.9g}`",
        f"- Adapted macro/class1 F1 delta: `{adapted['delta']['macro_f1']:.6f}/{adapted['delta']['class1_f1']:.6f}`",
        f"- Adapted class1 precision/recall delta: `{adapted['delta']['class1_precision']:.6f}/{adapted['delta']['class1_recall']:.6f}`",
        f"- Adapted changed/corrections/harms: `{adapted['transitions']['changed']}/{adapted['transitions']['candidate_correction']}/{adapted['transitions']['candidate_harm']}`",
        f"- Adapted restricted FP reduction, FN rescue/TP break: `{adapted['transitions']['restricted_focus_fp_reduction']}/{adapted['transitions']['focus_fn_rescue']}/{adapted['transitions']['focus_tp_break']}`",
        "",
        "Validation and test were not loaded.",
    ]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_artifacts = [
        {
            "path": str(summary_path.resolve()),
            "sha256": _sha256(summary_path),
            "role": "stage_a_summary",
        },
        {
            "path": str(predictions_path.resolve()),
            "sha256": _sha256(predictions_path),
            "role": "ordered_train_holdout_predictions",
        },
        {
            "path": str(illumination_path.resolve()),
            "sha256": _sha256(illumination_path),
            "role": "ordered_train_holdout_illumination_predictions",
        },
    ]
    for export_key, role in (
        ("isolated", "isolated_graph_onnx"),
        ("full", "candidate_full_onnx"),
    ):
        export_row = post["export"][export_key]
        if export_row.get("sha256"):
            manifest_artifacts.append(
                {
                    "path": export_row["path"],
                    "sha256": export_row["sha256"],
                    "role": role,
                    "export_succeeded": bool(export_row["succeeded"]),
                }
            )
    manifest = {
        "raw_dataset_modified": False,
        "validation_used": False,
        "test_used": False,
        "artifacts": manifest_artifacts,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(
        json.dumps(
            {
                "output_dir": str(Path(summary["artifacts"]["full_onnx"]).parent),
                "all_gates_passed": summary["gate"]["all_gates_passed"],
                "stage_b_smoke_authorized": summary["gate"]["stage_b_smoke_authorized"],
                "failed_checks": summary["gate"]["failed_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
