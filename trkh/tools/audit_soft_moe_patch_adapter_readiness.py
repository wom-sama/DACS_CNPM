from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
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
    MultiHeadSelfAttention,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.models.soft_moe_patch_adapter import SoftMoEPatchAdapter
from trkh.tools.audit_deep_class_prompt_readiness import (
    LIGHTING_CONDITIONS,
    _balanced_indices,
    _benchmark_inference,
    _compact_variant,
    _deep_comparison,
    _forward_features_and_logits,
    _keeper_state_sha256,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_patch_style_srm_readiness import (
    _failed_export,
    _make_lighting_loader,
    _onnx_compare,
    _write_illumination_predictions,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
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
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_BLOCKS = 8
EXPECTED_LAYERS = (2, 5)
EXPECTED_EXPERTS = 4
EXPECTED_HIDDEN_DIM = 64
EXPECTED_ADDED_PARAMETERS = 266_754
EXPECTED_TRAIN_BATCHES = 60
EXPECTED_TRAIN_BUDGET_ROWS = 1920
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "d46319582c8fba1c3f061e987b3f9bbdc4442d188c5f39bd20e0dfe1ca57110b"
LOCKED_SOFT_MOE_PAPER_SHA256 = "0d35f584a24f81b9e2736ca32548630747e9859a8e754d8e448899bbf5eae549"
LOCKED_SWEET_SPOT_PAPER_SHA256 = "73a5c6bd80e1818cd2e159399e2e8b909ee1a4535f4e32a25f0ea418b36a2bcb"
LOCKED_VMOE_COMMIT = "1030c713e7c0db5e12bb1a8cd1164d2c63b7b401"
LOCKED_VMOE_ROUTER_SHA256 = "d50d3894886e259c2adc274e4c852afb676ffa8ff0712982f25378970c75c720"
LOCKED_VMOE_COMMON_SHA256 = "9c1f9a56660432e43d0a2742b09106ed247942afd148b512801dc423b3137531"
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.25
MAX_PEAK_VRAM_GIB = 3.25
STATIC_EXPORT_BATCH_SIZE = 1


def _expected_state_keys() -> set[str]:
    suffixes = {"router_slots", "router_scale"}
    for expert_index in range(EXPECTED_EXPERTS):
        suffixes.update(
            {
                f"experts.{expert_index}.input_projection.weight",
                f"experts.{expert_index}.input_projection.bias",
                f"experts.{expert_index}.output_projection.weight",
                f"experts.{expert_index}.output_projection.bias",
            }
        )
    return {
        f"blocks.{layer - 1}.soft_moe_patch_adapter.{suffix}"
        for layer in EXPECTED_LAYERS
        for suffix in suffixes
    }


EXPECTED_STATE_KEYS = _expected_state_keys()
CONTROL_TRAINABLE = {
    key
    for key in EXPECTED_STATE_KEYS
    if ".output_projection." in key
}
CANDIDATE_TRAINABLE = set(EXPECTED_STATE_KEYS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Stage-A audit for the TRKH Soft-MoE patch adapter. "
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
            "docs/TRKH_5CLASS_SOFT_MOE_PATCH_ADAPTER_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--vmoe-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\vmoe"
        ),
    )
    parser.add_argument(
        "--soft-moe-paper",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\soft_moe_arxiv_2308.00951.pdf"
        ),
    )
    parser.add_argument(
        "--sweet-spot-paper",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_soft_moe_primary_20260715\vision_moe_sweet_spot_arxiv_2411.18322v2.pdf"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=5)
    parser.add_argument("--balanced-per-class", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--layers", type=str, default="2,5")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--router-scale-init", type=float, default=10.0)
    parser.add_argument("--init-seed", type=int, default=20260715)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
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
            "soft_moe_patch_adapter": True,
            "soft_moe_patch_adapter_layers": str(args.layers),
            "soft_moe_hidden_dim": int(args.hidden_dim),
            "soft_moe_num_experts": int(args.num_experts),
            "soft_moe_residual_scale": float(args.residual_scale),
            "soft_moe_router_scale_init": float(args.router_scale_init),
            "soft_moe_init_seed": int(args.init_seed),
            "pretrained": False,
        }
    )
    return config


def _extension_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if ".soft_moe_patch_adapter." in name
    }


def _extension_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(_extension_state(model).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


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

    set_seed(int(args.seed), deterministic=True)
    keeper = create_model(num_classes=len(class_names), model_config=source_config)
    keeper_constructor_rng = _rng_snapshot()
    load_model_state(keeper, dict(state), strict=True)

    set_seed(int(args.seed), deterministic=True)
    prototype = create_model(
        num_classes=len(class_names),
        model_config=_candidate_config(source_config, args),
    )
    prototype_constructor_rng = _rng_snapshot()
    missing, unexpected = load_model_state(prototype, dict(state), strict=False)
    prototype_state = prototype.state_dict()
    state_additions = set(prototype_state).difference(state)
    existing_bit_exact = all(
        key in prototype_state
        and torch.equal(prototype_state[key].detach().cpu(), value.detach().cpu())
        for key, value in state.items()
    )
    keeper_parameters = sum(int(value.numel()) for value in keeper.parameters())
    prototype_parameters = sum(int(value.numel()) for value in prototype.parameters())
    return keeper.eval(), prototype.eval(), {
        "keeper_constructor_rng": _rng_summary(keeper_constructor_rng),
        "prototype_constructor_rng": _rng_summary(prototype_constructor_rng),
        "constructor_rng_equal": _rng_equal(
            keeper_constructor_rng,
            prototype_constructor_rng,
        ),
        "reported_missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "state_additions": sorted(state_additions),
        "expected_state_additions": sorted(EXPECTED_STATE_KEYS),
        "state_additions_exact": state_additions == EXPECTED_STATE_KEYS,
        "reported_missing_keys_exact": set(missing) == EXPECTED_STATE_KEYS,
        "existing_state_bit_exact": bool(existing_bit_exact),
        "keeper_state_sha256": _state_sha256(keeper),
        "prototype_state_sha256": _state_sha256(prototype),
        "extension_state_sha256": _extension_state_sha256(prototype),
        "keeper_parameters": keeper_parameters,
        "prototype_parameters": prototype_parameters,
        "added_parameters": prototype_parameters - keeper_parameters,
    }


def _configure_trainability(model: nn.Module, *, candidate: bool) -> list[str]:
    expected = CANDIDATE_TRAINABLE if candidate else CONTROL_TRAINABLE
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name in expected
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if set(trainable) != expected:
        raise ValueError(
            f"Soft-MoE trainable set mismatch: {set(trainable)} != {expected}."
        )
    model.eval()
    return trainable


def _movement_groups(movement: Mapping[str, Mapping[str, object]]) -> Dict[str, bool]:
    groups = {
        "router": [key for key in movement if "router_" in key],
        "input_projection": [key for key in movement if ".input_projection." in key],
        "output_projection": [key for key in movement if ".output_projection." in key],
    }
    return {
        group: bool(keys) and any(bool(movement[key]["changed"]) for key in keys)
        for group, keys in groups.items()
    }


def _train_variant(
    *,
    name: str,
    prototype: nn.Module,
    candidate: bool,
    keeper_keys: Sequence[str],
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
    model = copy.deepcopy(prototype).to(device).eval()
    initial_full_sha = _state_sha256(model)
    initial_keeper_sha = _keeper_state_sha256(model, keeper_keys)
    initial_extension = _extension_state(model)
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"soft_moe_{name}_holdout",
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
        context=f"soft_moe_{name}_fit",
        seed=int(args.seed) + 200,
    )
    trainable_names = _configure_trainability(model, candidate=candidate)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    named_parameters = dict(model.named_parameters())
    ordered_indices: list[int] = []
    post_forward_rng: list[Dict[str, object]] = []
    history: list[Dict[str, object]] = []
    gradient_seen = {key: False for key in trainable_names}
    all_gradients_finite = True
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
            finite = bool(torch.isfinite(gradient).all())
            nonzero = int(torch.count_nonzero(gradient).item()) > 0
            all_gradients_finite = bool(all_gradients_finite and finite)
            gradient_seen[parameter_name] = bool(
                gradient_seen[parameter_name] or nonzero
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
    final_full_sha = _state_sha256(model)
    final_keeper_sha = _keeper_state_sha256(model, keeper_keys)
    final_state = model.state_dict()
    movement = {
        key: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final_state[key]),
            "changed": not torch.equal(value, final_state[key].detach().cpu()),
        }
        for key, value in initial_extension.items()
    }
    result = {
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
        "initial_state_sha256": initial_full_sha,
        "final_state_sha256": final_full_sha,
        "state_changed": initial_full_sha != final_full_sha,
        "keeper_state_sha256_before": initial_keeper_sha,
        "keeper_state_sha256_after": final_keeper_sha,
        "keeper_state_bit_exact": initial_keeper_sha == final_keeper_sha,
        "extension_state_movement": movement,
        "movement_groups": _movement_groups(movement),
        "history": history,
        "unadapted_predictions": unadapted_predictions,
        "adapted_predictions": adapted_predictions,
        "loader": {
            "train": train_loader_summary,
            "holdout": holdout_loader_summary,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    model = model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, result


def _activate_audit_outputs(module: SoftMoEPatchAdapter) -> None:
    with torch.no_grad():
        for expert_index, expert in enumerate(module.experts):
            values = torch.linspace(
                -0.02,
                0.02,
                expert.output_projection.weight.numel(),
                device=expert.output_projection.weight.device,
            ).reshape_as(expert.output_projection.weight)
            expert.output_projection.weight.copy_(
                values * float(expert_index + 1)
            )
            expert.output_projection.bias.copy_(
                torch.linspace(
                    -0.01,
                    0.01,
                    module.dim,
                    device=expert.output_projection.bias.device,
                )
            )


def _equation_diagnostics(
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for label, dtype in (("fp32", None), ("bf16", amp_dtype)):
        module = SoftMoEPatchAdapter(
            dim=32,
            hidden_dim=8,
            num_experts=4,
            residual_scale=0.10,
            router_scale_init=10.0,
            init_seed=20260715,
        ).to(device).eval()
        _activate_audit_outputs(module)
        tokens = torch.linspace(
            -1.0,
            1.0,
            2 * 15 * 32,
            device=device,
        ).reshape(2, 15, 32)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=dtype is not None,
        ):
            updated, details = module(
                tokens,
                prefix_count=3,
                return_details=True,
            )
        patches = tokens[:, 3:]
        dispatch = details["dispatch_weights"]
        combine = details["combine_weights"]
        # Recompose under the same autocast contract as the module. Comparing a
        # BF16 einsum against a separately recomposed FP32 einsum is not an
        # equation error and previously produced a false structural failure.
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=dtype is not None,
        ):
            expected_slots = torch.einsum(
                "bne,bnd->bed",
                dispatch.to(dtype=patches.dtype),
                patches,
            )
            expected_residual = torch.einsum(
                "bne,bed->bnd",
                combine.to(dtype=details["expert_outputs"].dtype),
                details["expert_outputs"],
            )
            expected_patches = (
                patches + 0.10 * expected_residual.to(dtype=patches.dtype)
            )
        output[label] = {
            "dispatch_sum_max_error": float(
                (dispatch.sum(dim=1) - 1.0).abs().amax().item()
            ),
            "combine_sum_max_error": float(
                (combine.sum(dim=2) - 1.0).abs().amax().item()
            ),
            "slot_max_error": float(
                (details["slots"] - expected_slots).abs().amax().item()
            ),
            "residual_max_error": float(
                (details["residual"] - expected_residual).abs().amax().item()
            ),
            "update_max_error": float(
                (updated[:, 3:] - expected_patches).abs().amax().item()
            ),
            "prefix_bit_exact": bool(torch.equal(updated[:, :3], tokens[:, :3])),
            "finite": bool(
                torch.isfinite(updated).all()
                and torch.isfinite(dispatch).all()
                and torch.isfinite(combine).all()
            ),
        }
    return output


def _gradient_diagnostics(
    *,
    prototype: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    trainable_names = _configure_trainability(model, candidate=True)
    images_value = images.to(device)
    targets_value = targets.to(device=device, dtype=torch.long)
    metadata_value = _metadata_for_batch(
        metadata,
        device=device,
        count=int(images_value.size(0)),
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_dtype is not None,
    ):
        logits = _forward_logits(model, images_value, metadata_value, device=device)
        loss = F.cross_entropy(logits.float(), targets_value)
    loss.backward()
    named_parameters = dict(model.named_parameters())
    gradients = {}
    for name in trainable_names:
        gradient = named_parameters[name].grad
        gradients[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero": bool(
                gradient is not None
                and int(torch.count_nonzero(gradient).item()) > 0
            ),
            "norm": float(
                gradient.detach().float().norm().item()
                if gradient is not None
                else 0.0
            ),
        }
    result = {
        "loss": float(loss.detach().item()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "gradients": gradients,
        "all_parameter_families": all(
            row["present"] and row["finite"] and row["nonzero"]
            for row in gradients.values()
        ),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _metadata_for_batch(
    metadata: Mapping[str, object],
    *,
    device: torch.device,
    count: int,
) -> Dict[str, object]:
    resolved_count = int(count)
    if resolved_count <= 0:
        raise ValueError("Metadata batch count must be positive.")
    output = _metadata_to_device(
        metadata,
        device=device,
        count=resolved_count,
    )
    mismatched = {
        key: int(value.size(0))
        for key, value in output.items()
        if torch.is_tensor(value)
        and value.ndim > 0
        and int(value.size(0)) != resolved_count
    }
    if mismatched:
        raise ValueError(
            f"Metadata tensors do not match batch size {resolved_count}: {mismatched}."
        )
    return output


def _pre_adaptation_diagnostics(
    *,
    keeper: nn.Module,
    prototype: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    keeper_model = copy.deepcopy(keeper).to(device).eval()
    candidate_model = copy.deepcopy(prototype).to(device).eval()
    images = images_cpu.to(device)

    set_seed(20260717, deterministic=True)
    with torch.inference_mode():
        keeper_features, keeper_logits = _forward_features_and_logits(
            keeper_model,
            images,
            metadata_cpu,
            device=device,
            return_attention=True,
            return_trace=True,
        )
    keeper_forward_rng = _rng_snapshot()
    set_seed(20260717, deterministic=True)
    with torch.inference_mode():
        candidate_features, candidate_logits = _forward_features_and_logits(
            candidate_model,
            images,
            metadata_cpu,
            device=device,
            return_attention=True,
            return_trace=True,
        )
    candidate_forward_rng = _rng_snapshot()
    with torch.inference_mode():
        pruned_features, pruned_logits = _forward_features_and_logits(
            candidate_model,
            images,
            metadata_cpu,
            device=device,
            return_trace=True,
        )

    keeper_attention_shapes = {
        int(layer): list(attention.shape)
        for layer, attention in keeper_features["attentions"].items()
    }
    candidate_attention_shapes = {
        int(layer): list(attention.shape)
        for layer, attention in candidate_features["attentions"].items()
    }
    dense_trace = candidate_features["trace"]
    pruned_trace = pruned_features["trace"]
    output_projection_zero = all(
        int(torch.count_nonzero(expert.output_projection.weight).item()) == 0
        and int(torch.count_nonzero(expert.output_projection.bias).item()) == 0
        for block in candidate_model.blocks
        if block.soft_moe_patch_adapter is not None
        for expert in block.soft_moe_patch_adapter.experts
    )
    result = {
        "initial_dense_logit_max_error": float(
            (candidate_logits - keeper_logits).abs().amax().item()
        ),
        "initial_dense_argmax_match": bool(
            torch.equal(candidate_logits.argmax(dim=1), keeper_logits.argmax(dim=1))
        ),
        "initial_pruned_logits_finite": bool(torch.isfinite(pruned_logits).all()),
        "forward_rng_equal": _rng_equal(keeper_forward_rng, candidate_forward_rng),
        "keeper_forward_rng": _rng_summary(keeper_forward_rng),
        "candidate_forward_rng": _rng_summary(candidate_forward_rng),
        "spatial_mhsa_layers": [
            index + 1
            for index, block in enumerate(candidate_model.blocks)
            if isinstance(block.attn, MultiHeadSelfAttention)
        ],
        "soft_moe_layers": [
            index + 1
            for index, block in enumerate(candidate_model.blocks)
            if block.soft_moe_patch_adapter is not None
        ],
        "keeper_attention_shapes": keeper_attention_shapes,
        "candidate_attention_shapes": candidate_attention_shapes,
        "attention_shapes_exact": keeper_attention_shapes
        == candidate_attention_shapes,
        "dense_trace_layers": dense_trace[
            "soft_moe_patch_adapter_layers"
        ].tolist(),
        "pruned_trace_layers": pruned_trace[
            "soft_moe_patch_adapter_layers"
        ].tolist(),
        "pruning_layers": [
            int(entry["layer"]) for entry in pruned_trace["pruning"]
        ],
        "output_projections_zero": output_projection_zero,
        "initial_dispatch_sum_max_error": max(
            float((value.sum(dim=1) - 1.0).abs().amax().item())
            for value in dense_trace["soft_moe_dispatch_weights"].values()
        ),
        "initial_combine_sum_max_error": max(
            float((value.sum(dim=2) - 1.0).abs().amax().item())
            for value in dense_trace["soft_moe_combine_weights"].values()
        ),
        "equation": _equation_diagnostics(device, amp_dtype),
    }
    del keeper_model, candidate_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _routing_snapshot(
    *,
    prototype: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    focus_class: int,
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    layer_values: Dict[int, Dict[str, list[Tensor]]] = {
        layer: {
            "combine_mass": [],
            "combine_entropy": [],
            "bbox_mass": [],
            "context_mass": [],
        }
        for layer in EXPECTED_LAYERS
    }
    targets_all: list[Tensor] = []
    predictions_all: list[Tensor] = []
    dispatch_sum_max_error = 0.0
    combine_sum_max_error = 0.0
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                features, logits = _forward_features_and_logits(
                    model,
                    images,
                    metadata,
                    device=device,
                    return_trace=True,
                )
            predictions = logits.float().argmax(dim=1)
            trace = features["trace"]
            bbox_prior = trace.get("bbox_patch_prior")
            if not torch.is_tensor(bbox_prior):
                raise ValueError("Soft-MoE routing XAI requires bbox_patch_prior.")
            for layer in EXPECTED_LAYERS:
                dispatch = trace["soft_moe_dispatch_weights"][layer].float()
                combine = trace["soft_moe_combine_weights"][layer].float()
                residual_norm = trace["soft_moe_residual_norm"][layer].float()
                patch_indices = trace["soft_moe_patch_indices"][layer].long()
                prior = bbox_prior.float().gather(1, patch_indices)
                normalized_residual = residual_norm / residual_norm.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)
                layer_values[layer]["combine_mass"].append(
                    combine.mean(dim=1).cpu()
                )
                entropy = -(
                    combine.clamp_min(1e-12) * combine.clamp_min(1e-12).log()
                ).sum(dim=2) / math.log(float(EXPECTED_EXPERTS))
                layer_values[layer]["combine_entropy"].append(
                    entropy.mean(dim=1).cpu()
                )
                layer_values[layer]["bbox_mass"].append(
                    (normalized_residual * (prior > 0.0).float()).sum(dim=1).cpu()
                )
                layer_values[layer]["context_mass"].append(
                    (normalized_residual * (prior <= 0.0).float()).sum(dim=1).cpu()
                )
                dispatch_sum_max_error = max(
                    dispatch_sum_max_error,
                    float((dispatch.sum(dim=1) - 1.0).abs().amax().item()),
                )
                combine_sum_max_error = max(
                    combine_sum_max_error,
                    float((combine.sum(dim=2) - 1.0).abs().amax().item()),
                )
            targets_all.append(targets.long().cpu())
            predictions_all.append(predictions.cpu())
    targets_value = torch.cat(targets_all)
    predictions_value = torch.cat(predictions_all)
    true_positive = (targets_value == int(focus_class)) & (
        predictions_value == int(focus_class)
    )
    restricted_false_positive = (
        torch.isin(targets_value, torch.tensor([0, 2, 4]))
        & (predictions_value == int(focus_class))
    )
    layers: Dict[str, object] = {}
    signature_separations = []
    for layer in EXPECTED_LAYERS:
        values = {
            key: torch.cat(rows, dim=0)
            for key, rows in layer_values[layer].items()
        }
        signature = values["combine_mass"]
        if bool(true_positive.any()) and bool(restricted_false_positive.any()):
            separation = float(
                (
                    signature[true_positive].mean(dim=0)
                    - signature[restricted_false_positive].mean(dim=0)
                )
                .abs()
                .sum()
                .item()
            )
        else:
            separation = 0.0
        signature_separations.append(separation)
        expert_mass = signature.mean(dim=0)
        layers[str(layer)] = {
            "mean_expert_combine_mass": expert_mass.tolist(),
            "minimum_expert_combine_mass": float(expert_mass.min().item()),
            "maximum_expert_combine_mass": float(expert_mass.max().item()),
            "mean_normalized_combine_entropy": float(
                values["combine_entropy"].mean().item()
            ),
            "mean_residual_bbox_mass": float(values["bbox_mass"].mean().item()),
            "mean_residual_context_mass": float(
                values["context_mass"].mean().item()
            ),
            "tp_vs_restricted_fp_signature_l1": separation,
        }
    result = {
        "rows": int(targets_value.numel()),
        "class1_true_positive_rows": int(true_positive.sum().item()),
        "restricted_focus_false_positive_rows": int(
            restricted_false_positive.sum().item()
        ),
        "dispatch_sum_max_error": dispatch_sum_max_error,
        "combine_sum_max_error": combine_sum_max_error,
        "maximum_signature_l1": max(signature_separations, default=0.0),
        "layers": layers,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _ablation_diagnostics(
    *,
    prototype: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    images_value = images.to(device)

    def logits() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            return _forward_logits(model, images_value, metadata, device=device).float()

    baseline = logits()
    layer_deltas: Dict[str, float] = {}
    expert_deltas: Dict[str, Dict[str, float]] = {}
    for layer in EXPECTED_LAYERS:
        module = model.blocks[layer - 1].soft_moe_patch_adapter
        if module is None:
            raise RuntimeError(f"Soft-MoE layer {layer} is missing.")
        original_scale = float(module.residual_scale)
        module.residual_scale = 0.0
        ablated = logits()
        module.residual_scale = original_scale
        layer_deltas[str(layer)] = float((baseline - ablated).abs().amax().item())
        expert_deltas[str(layer)] = {}
        for expert_index, expert in enumerate(module.experts):
            weight = expert.output_projection.weight.detach().clone()
            bias = expert.output_projection.bias.detach().clone()
            with torch.no_grad():
                expert.output_projection.weight.zero_()
                expert.output_projection.bias.zero_()
            expert_ablated = logits()
            with torch.no_grad():
                expert.output_projection.weight.copy_(weight)
                expert.output_projection.bias.copy_(bias)
            expert_deltas[str(layer)][str(expert_index)] = float(
                (baseline - expert_ablated).abs().amax().item()
            )
    result = {
        "layer_max_logit_delta": layer_deltas,
        "expert_max_logit_delta": expert_deltas,
        "minimum_layer_delta": min(layer_deltas.values(), default=0.0),
        "minimum_expert_delta": min(
            (
                value
                for rows in expert_deltas.values()
                for value in rows.values()
            ),
            default=0.0,
        ),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _peak_training_allocation(
    *,
    prototype: nn.Module,
    images: Tensor,
    targets: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    _configure_trainability(model, candidate=True)
    images_value = images.to(device)
    targets_value = targets.to(device=device, dtype=torch.long)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = _forward_logits(model, images_value, metadata, device=device)
        loss = F.cross_entropy(logits.float(), targets_value)
    loss.backward()
    torch.cuda.synchronize(device)
    result = {
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "loss": float(loss.detach().item()),
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


class _SoftMoEExportWrapper(nn.Module):
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
    wrapper = _SoftMoEExportWrapper(candidate)
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
    path = output_dir / "soft_moe_candidate.onnx"
    try:
        return _onnx_compare(
            wrapper=wrapper,
            inputs=(cpu_images, cpu_bbox, cpu_mask),
            input_names=("images", "bbox", "image_mask"),
            path=path,
        )
    except Exception as error:
        return _failed_export(path, error)


def _post_adaptation_diagnostics(
    *,
    keeper: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    resource_images: Tensor,
    resource_targets: Tensor,
    resource_metadata: Mapping[str, object],
    holdout_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, object]:
    fp32_count = int(args.fp32_batch_size)
    fp32_gradients = _gradient_diagnostics(
        prototype=candidate,
        images=resource_images[:fp32_count],
        targets=resource_targets[:fp32_count],
        metadata=resource_metadata,
        device=device,
        amp_dtype=None,
    )
    bf16_gradients = _gradient_diagnostics(
        prototype=candidate,
        images=resource_images,
        targets=resource_targets,
        metadata=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
    )
    control_routing = _routing_snapshot(
        prototype=control,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
        focus_class=int(args.focus_class),
    )
    candidate_routing = _routing_snapshot(
        prototype=candidate,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
        focus_class=int(args.focus_class),
    )
    spatial_delta = {}
    for layer in EXPECTED_LAYERS:
        key = str(layer)
        spatial_delta[key] = {
            "residual_bbox_mass": float(
                candidate_routing["layers"][key]["mean_residual_bbox_mass"]
            )
            - float(control_routing["layers"][key]["mean_residual_bbox_mass"]),
            "residual_context_mass": float(
                candidate_routing["layers"][key]["mean_residual_context_mass"]
            )
            - float(control_routing["layers"][key]["mean_residual_context_mass"]),
        }
    ablation = _ablation_diagnostics(
        prototype=candidate,
        images=resource_images,
        metadata=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
    )
    keeper_benchmark = _benchmark_inference(
        prototype=keeper,
        images=resource_images,
        metadata=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    candidate_benchmark = _benchmark_inference(
        prototype=candidate,
        images=resource_images,
        metadata=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    runtime_ratio = float(
        candidate_benchmark["median_seconds"]
        / max(float(keeper_benchmark["median_seconds"]), 1e-12)
    )
    training_allocation = _peak_training_allocation(
        prototype=candidate,
        images=resource_images,
        targets=resource_targets,
        metadata=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
    )
    export = _export_diagnostics(
        candidate=candidate,
        images=resource_images,
        metadata=resource_metadata,
        output_dir=output_dir,
    )
    return {
        "fp32_gradients": fp32_gradients,
        "bf16_gradients": bf16_gradients,
        "control_routing_xai": control_routing,
        "candidate_routing_xai": candidate_routing,
        "routing_spatial_delta": spatial_delta,
        "ablation": ablation,
        "keeper_inference_benchmark": keeper_benchmark,
        "candidate_inference_benchmark": candidate_benchmark,
        "runtime_ratio": runtime_ratio,
        "training_allocation": training_allocation,
        "peak_vram_gib": float(training_allocation["peak_vram_gib"]),
        "export": export,
    }


def _decision_checks(
    comparison: Mapping[str, object],
    *,
    prefix: str,
) -> Dict[str, bool]:
    delta = comparison["delta"]
    transitions = comparison["transitions"]
    return {
        f"{prefix}_macro_f1_delta_gte_0": float(delta["macro_f1"]) >= 0.0,
        f"{prefix}_class1_f1_delta_gte_0p005": float(delta["class1_f1"])
        >= 0.005,
        f"{prefix}_class1_precision_delta_gte_0p005": float(
            delta["class1_precision"]
        )
        >= 0.005,
        f"{prefix}_class1_recall_delta_gte_minus_0p005": float(
            delta["class1_recall"]
        )
        >= -0.005,
        f"{prefix}_restricted_focus_fp_corrected_gte_2": int(
            transitions["restricted_focus_fp_corrected_to_target"]
        )
        >= 2,
        f"{prefix}_focus_fn_rescues_gte_tp_breaks": int(
            transitions["focus_fn_rescue"]
        )
        >= int(transitions["focus_tp_break"]),
        f"{prefix}_corrections_gt_harms": int(
            transitions["candidate_correction"]
        )
        > int(transitions["candidate_harm"]),
        f"{prefix}_maximum_nonfocus_f1_drop_lte_0p005": float(
            comparison["maximum_nonfocus_f1_drop"]
        )
        <= 0.005,
    }


def _illumination_checks(
    comparisons: Sequence[Mapping[str, object]],
    *,
    prefix: str,
) -> Dict[str, bool]:
    values = list(comparisons)
    f1 = [float(row["delta"]["class1_f1"]) for row in values]
    precision = [float(row["delta"]["class1_precision"]) for row in values]
    recall = [float(row["delta"]["class1_recall"]) for row in values]
    return {
        f"{prefix}_three_conditions_exact": len(values) == 3,
        f"{prefix}_all_class1_f1_deltas_gte_minus_0p005": all(
            value >= -0.005 for value in f1
        ),
        f"{prefix}_worst_class1_recall_delta_gte_minus_0p010": min(
            recall,
            default=-math.inf,
        )
        >= -0.010,
        f"{prefix}_class1_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in precision
        )
        >= 2,
    }


def assess_soft_moe_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    candidate_vs_raw: Mapping[str, object],
    candidate_vs_control: Mapping[str, object],
    illumination_vs_raw: Sequence[Mapping[str, object]],
    illumination_vs_control: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    decision_checks = {
        **_decision_checks(candidate_vs_raw, prefix="candidate_vs_raw"),
        **_decision_checks(candidate_vs_control, prefix="candidate_vs_control"),
    }
    illumination_checks = {
        **_illumination_checks(illumination_vs_raw, prefix="candidate_vs_raw"),
        **_illumination_checks(
            illumination_vs_control,
            prefix="candidate_vs_control",
        ),
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **decision_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_smoke_authorized": not failed,
        "full_train_authorized": False,
    }


def _write_holdout_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    raw: Sequence[Mapping[str, object]],
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    variants = {
        "raw": raw,
        "control_unadapted": control["unadapted_predictions"],
        "control_adapted": control["adapted_predictions"],
        "candidate_unadapted": candidate["unadapted_predictions"],
        "candidate_adapted": candidate["adapted_predictions"],
    }
    row_by_index = {int(row.sample_index): row for row in rows}
    fieldnames = ["sample_index", "path", "target"]
    for variant in variants:
        fieldnames.append(f"{variant}_prediction")
        fieldnames.extend(f"{variant}_prob_{index}" for index in range(5))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, raw_row in enumerate(raw):
            sample_index = int(raw_row["sample_index"])
            source = row_by_index[sample_index]
            output: Dict[str, object] = {
                "sample_index": sample_index,
                "path": str(source.image_path),
                "target": int(raw_row["target"]),
            }
            for variant, values in variants.items():
                value = values[position]
                if int(value["sample_index"]) != sample_index:
                    raise ValueError(f"{variant} holdout order changed.")
                output[f"{variant}_prediction"] = int(value["prediction"])
                for class_index in range(5):
                    output[f"{variant}_prob_{class_index}"] = float(
                        value[f"prob_{class_index}"]
                    )
            writer.writerow(output)


def _write_illumination_audit(
    path: Path,
    *,
    condition_rows: Mapping[
        str, Mapping[str, Sequence[Mapping[str, object]]]
    ],
) -> None:
    fieldnames = ["condition", "sample_index", "target"]
    for variant in ("raw", "control", "candidate"):
        fieldnames.append(f"{variant}_prediction")
        fieldnames.extend(f"{variant}_prob_{index}" for index in range(5))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, variants in condition_rows.items():
            raw_rows = variants["raw"]
            for position, raw_row in enumerate(raw_rows):
                sample_index = int(raw_row["sample_index"])
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "target": int(raw_row["target"]),
                }
                for variant in ("raw", "control", "candidate"):
                    row = variants[variant][position]
                    if int(row["sample_index"]) != sample_index:
                        raise ValueError(f"{condition}/{variant} order changed.")
                    output[f"{variant}_prediction"] = int(row["prediction"])
                    for class_index in range(5):
                        output[f"{variant}_prob_{class_index}"] = float(
                            row[f"prob_{class_index}"]
                        )
                writer.writerow(output)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked Soft-MoE Stage A requires CUDA.")
    locked_config_exact = bool(
        int(args.batch_size) == 32
        and int(args.fp32_batch_size) == 5
        and int(args.balanced_per_class) == 5
        and int(args.seed) == 42
        and int(args.fold) == 0
        and int(args.max_train_batches) == EXPECTED_TRAIN_BATCHES
        and math.isclose(float(args.learning_rate), 0.005, abs_tol=1e-12)
        and math.isclose(float(args.momentum), 0.9, abs_tol=1e-12)
        and math.isclose(float(args.weight_decay), 0.001, abs_tol=1e-12)
        and int(args.focus_class) == 1
        and str(args.layers) == "2,5"
        and int(args.hidden_dim) == EXPECTED_HIDDEN_DIM
        and int(args.num_experts) == EXPECTED_EXPERTS
        and math.isclose(float(args.residual_scale), 0.10, abs_tol=1e-12)
        and math.isclose(float(args.router_scale_init), 10.0, abs_tol=1e-12)
        and int(args.init_seed) == 20260715
        and int(args.benchmark_repeats) == 5
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )
    if not locked_config_exact:
        raise ValueError(
            "Stage A arguments differ from the precommitted Soft-MoE protocol."
        )

    output_dir = _prepare_output_dir(args.output_dir)
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "soft_moe_paper": Path(args.soft_moe_paper).resolve(),
        "sweet_spot_paper": Path(args.sweet_spot_paper).resolve(),
    }
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"],
            LOCKED_LAUNCHER_ARGS_SHA256,
            "launcher args",
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
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "protocol"
        ),
        "soft_moe_paper": _verify_sha256(
            paths["soft_moe_paper"],
            LOCKED_SOFT_MOE_PAPER_SHA256,
            "Soft-MoE paper",
        ),
        "sweet_spot_paper": _verify_sha256(
            paths["sweet_spot_paper"],
            LOCKED_SWEET_SPOT_PAPER_SHA256,
            "vision MoE sweet-spot paper",
        ),
    }
    vmoe_root = Path(args.vmoe_root).resolve()
    vmoe_commit = _git_commit(vmoe_root)
    if vmoe_commit != LOCKED_VMOE_COMMIT:
        raise ValueError("Official V-MoE commit differs from the locked source.")
    vmoe_router = vmoe_root / "vmoe" / "projects" / "soft_moe" / "router.py"
    vmoe_common = (
        vmoe_root
        / "vmoe"
        / "projects"
        / "soft_moe"
        / "configs"
        / "common.py"
    )
    _verify_sha256(vmoe_router, LOCKED_VMOE_ROUTER_SHA256, "V-MoE SoftRouter")
    _verify_sha256(vmoe_common, LOCKED_VMOE_COMMON_SHA256, "V-MoE common config")

    repo_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "auditor": Path(__file__).resolve(),
        "module": repo_root / "trkh" / "models" / "soft_moe_patch_adapter.py",
        "model": repo_root / "trkh" / "models" / "model.py",
        "config": repo_root / "trkh" / "core" / "config.py",
        "trainer": repo_root / "trkh" / "training" / "train.py",
        "v8_launcher": repo_root / "scripts" / "run_trkh_5class_attention_views_v8.ps1",
        "audit_launcher": repo_root / "scripts" / "run_trkh_soft_moe_patch_adapter_readiness.ps1",
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
    class_counts = tuple(
        sum(row.target == class_index for row in rows)
        for class_index in range(len(EXPECTED_CLASS_COUNTS))
    )
    fold_counts = tuple(
        sum(row.fold == fold_index for row in rows)
        for fold_index in range(len(EXPECTED_FOLD_COUNTS))
    )
    if len(rows) != EXPECTED_TRAIN_ROWS or class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError("Locked train rows/class counts differ.")
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError("Locked source-fold counts differ.")

    device = torch.device("cuda")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    amp_dtype = _amp_dtype(device)
    if amp_dtype != torch.bfloat16:
        raise RuntimeError("Locked Soft-MoE Stage A requires BF16-capable CUDA.")

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    source_config = checkpoint.get("model_config")
    keeper_state = checkpoint.get("model_state")
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Keeper class order is invalid.")
    if not isinstance(source_config, Mapping) or not isinstance(keeper_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if bool(source_config.get("soft_moe_patch_adapter", False)):
        raise ValueError("Keeper unexpectedly already contains Soft MoE.")
    if int(source_config.get("depth", 0)) != EXPECTED_BLOCKS:
        raise ValueError("Keeper Transformer depth differs from locked protocol.")
    keeper, prototype, construction = _construct_models(checkpoint, args)

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
    from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
        _build_eval_transform,
    )

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
    if (
        len(fit_indices) != 7372
        or len(train_indices) != EXPECTED_TRAIN_BUDGET_ROWS
        or len(holdout_indices) != EXPECTED_FOLD_COUNTS[0]
    ):
        raise ValueError("Locked fit/train-budget/holdout row counts differ.")

    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices[: int(args.batch_size)],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="soft_moe_stage_a_resource",
        seed=int(args.seed) + 300,
    )
    resource_images, resource_targets, resource_metadata = next(iter(resource_loader))
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="soft_moe_stage_a_holdout_xai",
        seed=int(args.seed) + 301,
    )
    balanced_indices = _balanced_indices(
        rows,
        holdout_indices,
        per_class=int(args.balanced_per_class),
    )
    pre = _pre_adaptation_diagnostics(
        keeper=keeper,
        prototype=prototype,
        images_cpu=resource_images,
        metadata_cpu=resource_metadata,
        device=device,
        amp_dtype=amp_dtype,
    )

    keeper_gpu = keeper.to(device).eval()
    raw_predictions = _predict(
        model=keeper_gpu,
        loader=holdout_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    keeper = keeper_gpu.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()

    control_model, control_result = _train_variant(
        name="control",
        prototype=prototype,
        candidate=False,
        keeper_keys=list(keeper_state),
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
        prototype=prototype,
        candidate=True,
        keeper_keys=list(keeper_state),
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
    if [int(row["sample_index"]) for row in raw_predictions] != holdout_indices:
        raise ValueError("Raw keeper holdout order changed.")

    unadapted_control_vs_raw = _deep_comparison(
        control_rows=raw_predictions,
        candidate_rows=control_result["unadapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    unadapted_candidate_vs_raw = _deep_comparison(
        control_rows=raw_predictions,
        candidate_rows=candidate_result["unadapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    candidate_vs_raw = _deep_comparison(
        control_rows=raw_predictions,
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    candidate_vs_control = _deep_comparison(
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )

    condition_predictions: Dict[
        str, Dict[str, Sequence[Mapping[str, object]]]
    ] = {}
    illumination_vs_raw = []
    illumination_vs_control = []
    keeper_gpu = keeper.to(device).eval()
    control_gpu = control_model.to(device).eval()
    candidate_gpu = candidate_model.to(device).eval()
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
            context=f"soft_moe_{condition}",
            seed=int(args.seed) + 500 + condition_index,
        )
        raw_rows = _predict(
            model=keeper_gpu,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        control_rows = _predict(
            model=control_gpu,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        candidate_rows = _predict(
            model=candidate_gpu,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        condition_predictions[condition] = {
            "raw": raw_rows,
            "control": control_rows,
            "candidate": candidate_rows,
        }
        raw_comparison = _deep_comparison(
            control_rows=raw_rows,
            candidate_rows=candidate_rows,
            num_classes=len(class_names),
            focus_class=int(args.focus_class),
        )
        control_comparison = _deep_comparison(
            control_rows=control_rows,
            candidate_rows=candidate_rows,
            num_classes=len(class_names),
            focus_class=int(args.focus_class),
        )
        for comparison in (raw_comparison, control_comparison):
            comparison["condition"] = condition
            comparison["brightness"] = brightness
            comparison["contrast"] = contrast
        illumination_vs_raw.append(raw_comparison)
        illumination_vs_control.append(control_comparison)
    keeper = keeper_gpu.cpu().eval()
    control_model = control_gpu.cpu().eval()
    candidate_model = candidate_gpu.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()

    post = _post_adaptation_diagnostics(
        keeper=keeper,
        control=control_model,
        candidate=candidate_model,
        resource_images=resource_images,
        resource_targets=resource_targets,
        resource_metadata=resource_metadata,
        holdout_loader=holdout_loader,
        args=args,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )
    equation_rows = [
        value
        for precision in pre["equation"].values()
        for key, value in precision.items()
        if key.endswith("_error")
    ]
    candidate_routing = post["candidate_routing_xai"]
    routing_layers = candidate_routing["layers"]
    routing_noncollapse = all(
        float(row["minimum_expert_combine_mass"]) >= 0.05
        and float(row["maximum_expert_combine_mass"]) <= 0.70
        and 0.35 <= float(row["mean_normalized_combine_entropy"]) < 0.995
        for row in routing_layers.values()
    )
    spatial_pass = all(
        float(row["residual_bbox_mass"]) >= -0.01
        and float(row["residual_context_mass"]) <= 0.02
        for row in post["routing_spatial_delta"].values()
    )
    control_movement = control_result["extension_state_movement"]
    candidate_movement = candidate_result["extension_state_movement"]
    structural_checks = {
        "locked_config_exact": locked_config_exact,
        "locked_sources_and_inputs": True,
        "train_only_provenance": True,
        "all_rows_class_and_fold_counts_exact": True,
        "fit_holdout_source_overlap_zero": len(source_overlap) == 0,
        "state_additions_exact_36": bool(construction["state_additions_exact"]),
        "reported_missing_keys_exact_36": bool(
            construction["reported_missing_keys_exact"]
        ),
        "unexpected_keys_zero": not construction["unexpected_keys"],
        "keeper_state_bit_exact_after_load": bool(
            construction["existing_state_bit_exact"]
        ),
        "added_parameters_exact_266754": int(construction["added_parameters"])
        == EXPECTED_ADDED_PARAMETERS,
        "constructor_rng_equal": bool(construction["constructor_rng_equal"]),
        "forward_rng_equal": bool(pre["forward_rng_equal"]),
        "initial_dense_logits_bit_exact": float(
            pre["initial_dense_logit_max_error"]
        )
        == 0.0
        and bool(pre["initial_dense_argmax_match"]),
        "zero_output_projections": bool(pre["output_projections_zero"]),
        "all_eight_native_spatial_mhsa_blocks": pre["spatial_mhsa_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "soft_moe_layers_exact_2_5": pre["soft_moe_layers"]
        == list(EXPECTED_LAYERS)
        and pre["dense_trace_layers"] == list(EXPECTED_LAYERS)
        and pre["pruned_trace_layers"] == list(EXPECTED_LAYERS),
        "pruning_after_layers_2_5": pre["pruning_layers"]
        == list(EXPECTED_LAYERS),
        "native_attention_schema_exact": bool(pre["attention_shapes_exact"]),
        "fp32_bf16_equation_and_normalization": max(
            [float(value) for value in equation_rows]
            + [
                float(pre["initial_dispatch_sum_max_error"]),
                float(pre["initial_combine_sum_max_error"]),
            ],
            default=math.inf,
        )
        <= 1e-5
        and all(
            bool(value["prefix_bit_exact"]) and bool(value["finite"])
            for value in pre["equation"].values()
        ),
        "matched_train_order": control_result["train_order_sha256"]
        == candidate_result["train_order_sha256"]
        == _ordered_index_sha256(train_indices),
        "matched_train_budget_60x32": control_result["train_batches"]
        == candidate_result["train_batches"]
        == EXPECTED_TRAIN_BATCHES
        and control_result["train_rows"]
        == candidate_result["train_rows"]
        == EXPECTED_TRAIN_BUDGET_ROWS,
        "matched_forward_rng_checkpoints": control_result["post_forward_rng"]
        == candidate_result["post_forward_rng"],
        "control_trainable_output_projection_only": set(
            control_result["trainable_parameters"]
        )
        == CONTROL_TRAINABLE,
        "candidate_trainable_all_soft_moe_only": set(
            candidate_result["trainable_parameters"]
        )
        == CANDIDATE_TRAINABLE,
        "both_training_gradients_finite_seen": bool(
            control_result["all_gradients_finite"]
            and candidate_result["all_gradients_finite"]
            and control_result["all_trainable_gradients_seen"]
            and candidate_result["all_trainable_gradients_seen"]
        ),
        "post_warm_fp32_all_gradients_finite_nonzero": bool(
            post["fp32_gradients"]["all_parameter_families"]
        ),
        "post_warm_bf16_all_gradients_finite_nonzero": bool(
            post["bf16_gradients"]["all_parameter_families"]
        ),
        "keeper_state_bit_exact_after_adaptation": bool(
            control_result["keeper_state_bit_exact"]
            and candidate_result["keeper_state_bit_exact"]
        ),
        "control_only_output_group_moved": bool(
            control_result["movement_groups"]["output_projection"]
            and not control_result["movement_groups"]["router"]
            and not control_result["movement_groups"]["input_projection"]
        ),
        "candidate_all_groups_moved": all(
            bool(value) for value in candidate_result["movement_groups"].values()
        ),
        "unadapted_control_and_candidate_exact_raw": int(
            unadapted_control_vs_raw["transitions"]["changed"]
        )
        == 0
        and int(unadapted_candidate_vs_raw["transitions"]["changed"]) == 0,
        "adapted_dispatch_combine_normalized": float(
            candidate_routing["dispatch_sum_max_error"]
        )
        <= 1e-5
        and float(candidate_routing["combine_sum_max_error"]) <= 1e-5,
        "routing_noncollapse": routing_noncollapse,
        "routing_signature_separation_gte_0p01": int(
            candidate_routing["class1_true_positive_rows"]
        )
        > 0
        and int(candidate_routing["restricted_focus_false_positive_rows"]) > 0
        and float(candidate_routing["maximum_signature_l1"]) >= 0.01,
        "residual_bbox_context_xai_preserved": spatial_pass,
        "both_layers_ablation_delta_gte_1e4": float(
            post["ablation"]["minimum_layer_delta"]
        )
        >= 1e-4,
        "onnx_error_lte_1e5_argmax_match": bool(post["export"]["succeeded"])
        and float(post["export"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR
        and bool(post["export"]["argmax_match"]),
        "runtime_ratio_lte_1p25": float(post["runtime_ratio"])
        <= float(args.max_runtime_ratio),
        "peak_training_vram_lte_3p25_gib": float(post["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
    }
    gate = assess_soft_moe_stage_a(
        structural_checks=structural_checks,
        candidate_vs_raw=candidate_vs_raw,
        candidate_vs_control=candidate_vs_control,
        illumination_vs_raw=illumination_vs_raw,
        illumination_vs_control=illumination_vs_control,
    )

    predictions_path = output_dir / "holdout_predictions.csv"
    _write_holdout_predictions(
        predictions_path,
        rows=rows,
        raw=raw_predictions,
        control=control_result,
        candidate=candidate_result,
    )
    illumination_path = output_dir / "illumination_predictions.csv"
    _write_illumination_audit(
        illumination_path,
        condition_rows=condition_predictions,
    )
    extension_path = output_dir / "adapted_soft_moe_extensions.pt"
    torch.save(
        {
            "method": "soft_moe_patch_adapter",
            "protocol_sha256": LOCKED_PROTOCOL_SHA256,
            "keeper_sha256": LOCKED_KEEPER_SHA256,
            "control_state": _extension_state(control_model),
            "candidate_state": _extension_state(candidate_model),
            "control_train_order_sha256": control_result["train_order_sha256"],
            "candidate_train_order_sha256": candidate_result[
                "train_order_sha256"
            ],
        },
        extension_path,
    )

    summary: Dict[str, object] = {
        "method": "soft_moe_patch_adapter",
        "protocol_stage": "A_train_only_functional_xai_resource_decision_readiness",
        "test_data_used": False,
        "validation_data_used": False,
        "sources": {
            **{key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "vmoe_root": str(vmoe_root),
            "vmoe_commit": vmoe_commit,
            "vmoe_router": str(vmoe_router),
            "vmoe_router_sha256": LOCKED_VMOE_ROUTER_SHA256,
            "vmoe_common": str(vmoe_common),
            "vmoe_common_sha256": LOCKED_VMOE_COMMON_SHA256,
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
            "balanced_per_class": int(args.balanced_per_class),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "max_train_batches": int(args.max_train_batches),
            "learning_rate": float(args.learning_rate),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "focus_class": int(args.focus_class),
            "layers": str(args.layers),
            "hidden_dim": int(args.hidden_dim),
            "num_experts": int(args.num_experts),
            "residual_scale": float(args.residual_scale),
            "router_scale_init": float(args.router_scale_init),
            "init_seed": int(args.init_seed),
        },
        "dataset": {
            "split_constructed": "train",
            "rows": len(rows),
            "class_counts": list(class_counts),
            "fold_counts": list(fold_counts),
            "fit_rows": len(fit_indices),
            "train_budget_rows": len(train_indices),
            "holdout_rows": len(holdout_indices),
            "fit_sources": len(fit_sources),
            "holdout_sources": len(holdout_sources),
            "source_overlap": len(source_overlap),
            "train_order_sha256": _ordered_index_sha256(train_indices),
            "holdout_order_sha256": _ordered_index_sha256(holdout_indices),
            "balanced_indices": balanced_indices,
            "resource_loader": resource_loader_summary,
            "holdout_loader": holdout_loader_summary,
        },
        "construction": construction,
        "pre_adaptation": pre,
        "post_adaptation": post,
        "control_training": _compact_variant(control_result),
        "candidate_training": _compact_variant(candidate_result),
        "unadapted_control_vs_raw": unadapted_control_vs_raw,
        "unadapted_candidate_vs_raw": unadapted_candidate_vs_raw,
        "adapted_candidate_vs_raw": candidate_vs_raw,
        "adapted_candidate_vs_control": candidate_vs_control,
        "illumination_candidate_vs_raw": illumination_vs_raw,
        "illumination_candidate_vs_control": illumination_vs_control,
        "gate": gate,
        "artifacts": {
            "holdout_predictions": predictions_path.name,
            "holdout_predictions_sha256": _sha256(predictions_path),
            "illumination_predictions": illumination_path.name,
            "illumination_predictions_sha256": _sha256(illumination_path),
            "adapted_soft_moe_extensions": extension_path.name,
            "adapted_soft_moe_extensions_sha256": _sha256(extension_path),
            "candidate_onnx": post["export"]["path"],
        },
        "guardrail": (
            "A complete pass authorizes one keeper-initialized no-test validation "
            "smoke only. It never authorizes test access, a full train, or a sweep."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Soft-MoE Patch Adapter Stage-A Readiness",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage-B smoke authorized: `{gate['stage_b_smoke_authorized']}`",
        f"- Failed checks: `{gate['failed_checks']}`",
        f"- Runtime ratio / peak VRAM GiB: `{post['runtime_ratio']:.6f}/{post['peak_vram_gib']:.6f}`",
        f"- ONNX max error: `{post['export']['maximum_absolute_error']:.9g}`",
        f"- Candidate-vs-raw macro/class1 F1 delta: `{candidate_vs_raw['delta']['macro_f1']:.6f}/{candidate_vs_raw['delta']['class1_f1']:.6f}`",
        f"- Candidate-vs-control macro/class1 F1 delta: `{candidate_vs_control['delta']['macro_f1']:.6f}/{candidate_vs_control['delta']['class1_f1']:.6f}`",
        f"- Candidate-vs-raw class1 precision/recall delta: `{candidate_vs_raw['delta']['class1_precision']:.6f}/{candidate_vs_raw['delta']['class1_recall']:.6f}`",
        f"- Candidate-vs-control class1 precision/recall delta: `{candidate_vs_control['delta']['class1_precision']:.6f}/{candidate_vs_control['delta']['class1_recall']:.6f}`",
        f"- Routing TP/restricted-FP/signature L1: `{candidate_routing['class1_true_positive_rows']}/{candidate_routing['restricted_focus_false_positive_rows']}/{candidate_routing['maximum_signature_l1']:.6f}`",
        f"- Minimum layer/expert ablation delta: `{post['ablation']['minimum_layer_delta']:.6f}/{post['ablation']['minimum_expert_delta']:.6f}`",
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
        {
            "path": str(extension_path.resolve()),
            "sha256": _sha256(extension_path),
            "role": "adapted_soft_moe_extensions",
        },
    ]
    if post["export"].get("sha256"):
        manifest_artifacts.append(
            {
                "path": post["export"]["path"],
                "sha256": post["export"]["sha256"],
                "role": "candidate_static_batch1_onnx",
                "export_succeeded": bool(post["export"]["succeeded"]),
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
                "all_gates_passed": summary["gate"]["all_gates_passed"],
                "stage_b_smoke_authorized": summary["gate"][
                    "stage_b_smoke_authorized"
                ],
                "failed_checks": summary["gate"]["failed_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
