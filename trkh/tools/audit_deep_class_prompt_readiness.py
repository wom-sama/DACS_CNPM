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
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import LightingShift
from trkh.inference.inference import load_checkpoint
from trkh.models.deep_class_prompt import DeepClassPrompt
from trkh.models.model import (
    MultiHeadSelfAttention,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
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
EXPECTED_BLOCKS = 8
EXPECTED_ADDED_PARAMETERS = 11_009
EXPECTED_TRAIN_BATCHES = 60
EXPECTED_TRAIN_BUDGET_ROWS = 1920
EXPECTED_PROMPT_STATE_KEYS = {
    "deep_class_prompt.prompt_embeddings",
    "deep_class_prompt.norm.weight",
    "deep_class_prompt.norm.bias",
    "deep_class_prompt.classifier.weight",
    "deep_class_prompt.classifier.bias",
}
CONTROL_TRAINABLE = {
    "deep_class_prompt.norm.weight",
    "deep_class_prompt.norm.bias",
    "deep_class_prompt.classifier.weight",
    "deep_class_prompt.classifier.bias",
}
CANDIDATE_TRAINABLE = CONTROL_TRAINABLE | {
    "deep_class_prompt.prompt_embeddings"
}
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "15703848b396beb2adbcc88f0f07f58e56c69ec449276d7fb76e1b833a865f9d"
LOCKED_PROMPT_CAM_COMMIT = "4d35f3fb2eb99a63859465fcc1c7c3f4879f3ac5"
LOCKED_PROMPT_CAM_VIT_SHA256 = "f409d3e101e01dffd05342b1cff798fb739046ced730d7451ddf5e0f49aa647d"
LOCKED_PROMPT_CAM_BUILD_SHA256 = "7b1e59dd88dfd269cf9c494426f9f2d546cc5903023f56f2aeff1cddcbfa3a62"
LOCKED_MCTFORMER_COMMIT = "0acc27ada87a5582053efb14648442d8644168aa"
LOCKED_MCTFORMER_SOURCE_SHA256 = "39fd884077bfb56d5bb21635ab0fc7d80148813d920a7c0e9329051fc6a5e00e"
LIGHTING_CONDITIONS = (
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
MAX_ONNX_ERROR = 1e-5
MAX_RUNTIME_RATIO = 1.25
MAX_PEAK_VRAM_GIB = 3.25
STATIC_EXPORT_BATCH_SIZE = 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only Stage-A audit for TRKH deep class prompts. "
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
            "docs/TRKH_5CLASS_DEEP_CLASS_PROMPT_READINESS_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--prompt-cam-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_promptcam_primary_20260715"
        ),
    )
    parser.add_argument(
        "--mctformer-root",
        type=Path,
        default=Path(
            r"C:\Users\ADMIN\AppData\Local\Temp\trkh_mctformer_primary_20260715"
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
    parser.add_argument("--logit-scale", type=float, default=0.10)
    parser.add_argument("--prompt-init-seed", type=int, default=20260715)
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
            "deep_class_prompt": True,
            "deep_class_prompt_logit_scale": float(args.logit_scale),
            "deep_class_prompt_init_seed": int(args.prompt_init_seed),
            "pretrained": False,
        }
    )
    return config


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
    prompt_model = create_model(
        num_classes=len(class_names),
        model_config=_candidate_config(source_config, args),
    )
    prompt_constructor_rng = _rng_snapshot()
    missing, unexpected = load_model_state(prompt_model, dict(state), strict=False)
    prompt_state = prompt_model.state_dict()
    state_additions = set(prompt_state).difference(state)
    existing_bit_exact = all(
        key in prompt_state
        and torch.equal(prompt_state[key].detach().cpu(), value.detach().cpu())
        for key, value in state.items()
    )
    keeper_parameters = sum(int(value.numel()) for value in keeper.parameters())
    prompt_parameters = sum(int(value.numel()) for value in prompt_model.parameters())
    prompt_extension_sha = _prompt_state_sha256(prompt_model)
    return keeper.eval(), prompt_model.eval(), {
        "keeper_constructor_rng": _rng_summary(keeper_constructor_rng),
        "prompt_constructor_rng": _rng_summary(prompt_constructor_rng),
        "constructor_rng_equal": _rng_equal(
            keeper_constructor_rng,
            prompt_constructor_rng,
        ),
        "reported_missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "state_additions": sorted(state_additions),
        "expected_state_additions": sorted(EXPECTED_PROMPT_STATE_KEYS),
        "state_additions_exact": state_additions == EXPECTED_PROMPT_STATE_KEYS,
        "reported_missing_keys_exact": set(missing) == EXPECTED_PROMPT_STATE_KEYS,
        "existing_state_bit_exact": bool(existing_bit_exact),
        "keeper_state_sha256": _state_sha256(keeper),
        "prompt_model_state_sha256": _state_sha256(prompt_model),
        "prompt_extension_sha256": prompt_extension_sha,
        "control_candidate_initial_extension_equal": True,
        "keeper_parameters": keeper_parameters,
        "prompt_parameters": prompt_parameters,
        "added_parameters": prompt_parameters - keeper_parameters,
    }


def _prompt_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("deep_class_prompt."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _keeper_state_sha256(model: nn.Module, keeper_keys: Sequence[str]) -> str:
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(str(value) for value in keeper_keys):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tracked_prompt_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("deep_class_prompt.")
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
            f"Deep prompt trainable set mismatch: {set(trainable)} != {expected}."
        )
    model.eval()
    return trainable


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
    initial_prompt_state = _tracked_prompt_state(model)
    holdout_loader, holdout_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=holdout_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"deep_class_prompt_{name}_holdout",
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
        context=f"deep_class_prompt_{name}_fit",
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
        for key, value in initial_prompt_state.items()
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
        "initial_state_sha256": initial_full_sha,
        "final_state_sha256": final_full_sha,
        "state_changed": initial_full_sha != final_full_sha,
        "keeper_state_sha256_before": initial_keeper_sha,
        "keeper_state_sha256_after": final_keeper_sha,
        "keeper_state_bit_exact": initial_keeper_sha == final_keeper_sha,
        "prompt_state_movement": movement,
        "history": history,
        "unadapted_predictions": unadapted_predictions,
        "adapted_predictions": adapted_predictions,
        "loader": {
            "train": train_loader_summary,
            "holdout": holdout_loader_summary,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _forward_features_and_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
    return_attention: bool = False,
    return_trace: bool = False,
) -> tuple[Dict[str, Tensor], Tensor]:
    metadata_value = _metadata_to_device(
        metadata,
        device=device,
        count=int(images.size(0)),
    )
    bbox = metadata_value.get("bbox")
    image_mask = metadata_value.get("image_mask")
    features = model.forward_features(
        images.to(device),
        bbox_token_prior=bbox if torch.is_tensor(bbox) else None,
        image_valid_mask=image_mask if torch.is_tensor(image_mask) else None,
        return_attention=return_attention,
        return_trace=return_trace,
    )
    if torch.is_tensor(bbox):
        features["bbox"] = bbox
    logits = classification_logits_from_features(model, features)
    return features, logits


def _equation_diagnostics(device: torch.device) -> Dict[str, object]:
    def run(dtype: torch.dtype) -> Dict[str, object]:
        module = DeepClassPrompt(
            depth=3,
            num_classes=5,
            embed_dim=16,
            init_seed=20260715,
        ).to(device).eval()
        tokens = torch.linspace(
            -1.0,
            1.0,
            steps=2 * 13 * 16,
            device=device,
            dtype=torch.float32,
        ).reshape(2, 13, 16)
        if dtype == torch.bfloat16:
            tokens = tokens.to(dtype=dtype)
        prefix_count = 3
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16,
        ):
            inserted = module.insert(
                tokens,
                block_index=1,
                base_prefix_count=prefix_count,
            )
            expected_inserted = torch.cat(
                (
                    tokens[:, :prefix_count],
                    module.prompt_embeddings[1]
                    .unsqueeze(0)
                    .expand(2, -1, -1)
                    .to(dtype=tokens.dtype),
                    tokens[:, prefix_count:],
                ),
                dim=1,
            )
            restored, prompts = module.extract_and_remove(
                inserted,
                base_prefix_count=prefix_count,
            )
            logits = module.logits(prompts)
            explicit_logits = F.linear(
                F.layer_norm(
                    prompts,
                    (16,),
                    module.norm.weight,
                    module.norm.bias,
                    module.norm.eps,
                ),
                module.classifier.weight,
                module.classifier.bias,
            ).squeeze(-1)
        return {
            "insert_max_abs_error": float(
                (inserted.float() - expected_inserted.float()).abs().amax().item()
            ),
            "restore_max_abs_error": float(
                (restored.float() - tokens.float()).abs().amax().item()
            ),
            "logit_max_abs_error": float(
                (logits.float() - explicit_logits.float()).abs().amax().item()
            ),
        }

    return {
        "fp32": run(torch.float32),
        "bf16": run(torch.bfloat16),
    }


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
    trainable = _configure_trainability(model, candidate=True)
    count = int(images.size(0))
    value = images.to(device).detach().requires_grad_(True)
    targets_value = targets[:count].to(device=device, dtype=torch.long)
    metadata_value = _metadata_to_device(metadata, device=device, count=count)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_dtype is not None,
    ):
        logits = _forward_logits(model, value, metadata_value, device=device)
        loss = F.cross_entropy(logits.float(), targets_value)
    loss.backward()
    named = dict(model.named_parameters())
    rows: Dict[str, Dict[str, object]] = {}
    for name in trainable:
        gradient = named[name].grad
        rows[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero": bool(
                gradient is not None and int(torch.count_nonzero(gradient).item()) > 0
            ),
            "l2_norm": float(
                gradient.detach().float().norm().item()
                if gradient is not None
                else 0.0
            ),
        }
    input_gradient = value.grad
    output = {
        "loss": float(loss.detach().item()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "parameters": rows,
        "all_parameter_families": all(
            bool(row["present"] and row["finite"] and row["nonzero"])
            for row in rows.values()
        ),
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


def _pre_adaptation_diagnostics(
    *,
    keeper: nn.Module,
    prompt_model: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    count = 2
    images = images_cpu[:count].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=count)
    keeper_forward = copy.deepcopy(keeper).to(device).eval()
    prompt_forward = copy.deepcopy(prompt_model).to(device).eval()
    set_seed(int(args.seed), deterministic=True)
    with torch.inference_mode():
        _forward_logits(keeper_forward, images, metadata, device=device)
    keeper_rng = _rng_snapshot()
    set_seed(int(args.seed), deterministic=True)
    with torch.inference_mode():
        _forward_logits(prompt_forward, images, metadata, device=device)
    prompt_rng = _rng_snapshot()
    del keeper_forward
    gc.collect()
    torch.cuda.empty_cache()

    model = prompt_forward
    block_input_lengths: list[int] = []
    handles = []
    for block in model.blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, inputs: block_input_lengths.append(
                    int(inputs[0].size(1))
                )
            )
        )
    with torch.inference_mode():
        pruned, _ = _forward_features_and_logits(
            model,
            images[:1],
            metadata,
            device=device,
            return_trace=True,
        )
    for handle in handles:
        handle.remove()
    with torch.inference_mode():
        dense, _ = _forward_features_and_logits(
            model,
            images[:1],
            metadata,
            device=device,
            return_attention=True,
            return_trace=True,
        )
    trace = pruned.get("trace", {})
    block_patch_indices = trace.get("block_patch_indices", [])
    block_token_shapes = trace.get("block_token_shapes", [])
    base_prefix_count = int(model.num_prefix_tokens)
    prompt_count = int(model.deep_class_prompt.num_classes)
    insertion_rows = []
    previous_patch_count = int(dense["patch_indices"].size(1))
    for index in range(len(model.blocks)):
        patch_count_after = int(block_patch_indices[index].size(1))
        base_tokens_after = int(block_token_shapes[index][1])
        insertion_rows.append(
            {
                "layer": index + 1,
                "observed_block_input_tokens": block_input_lengths[index],
                "expected_block_input_tokens": (
                    base_prefix_count + prompt_count + previous_patch_count
                ),
                "observed_base_tokens_after_removal": base_tokens_after,
                "expected_base_tokens_after_removal": (
                    base_prefix_count + patch_count_after
                ),
                "patches_before": previous_patch_count,
                "patches_after": patch_count_after,
            }
        )
        previous_patch_count = patch_count_after
    native_attentions = dense.get("attentions", {})
    dense_prompt_maps = dense.get("deep_class_prompt_attentions", {})
    pruned_prompt_maps = pruned.get("deep_class_prompt_attentions", {})
    original_patch_count = int(dense["patch_indices"].size(1))
    output = {
        "keeper_forward_rng": _rng_summary(keeper_rng),
        "prompt_forward_rng": _rng_summary(prompt_rng),
        "forward_rng_equal": _rng_equal(keeper_rng, prompt_rng),
        "equation": _equation_diagnostics(device),
        "prompt_layers": len(model.blocks),
        "spatial_mhsa_layers": [
            index + 1
            for index, block in enumerate(model.blocks)
            if isinstance(block.attn, MultiHeadSelfAttention)
        ],
        "native_attention_layers": [int(value) + 1 for value in native_attentions],
        "native_attention_shapes": {
            str(int(layer) + 1): [int(value) for value in attention.shape]
            for layer, attention in native_attentions.items()
        },
        "dense_prompt_map_layers": [
            int(value) + 1 for value in dense_prompt_maps
        ],
        "pruned_prompt_map_layers": [
            int(value) + 1 for value in pruned_prompt_maps
        ],
        "dense_prompt_map_shapes": {
            str(int(layer) + 1): [int(value) for value in attention.shape]
            for layer, attention in dense_prompt_maps.items()
        },
        "pruned_prompt_map_shapes": {
            str(int(layer) + 1): [int(value) for value in attention.shape]
            for layer, attention in pruned_prompt_maps.items()
        },
        "original_patch_count": original_patch_count,
        "pruning_layers": [
            int(value["layer"].item()) for value in trace.get("pruning", [])
        ],
        "insertion_removal": insertion_rows,
        "base_prefix_count": base_prefix_count,
        "prompt_count": prompt_count,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _balanced_indices(
    rows: Sequence[CleanTrainRow],
    holdout_indices: Sequence[int],
    *,
    per_class: int,
) -> list[int]:
    selected: list[int] = []
    holdout_set = set(int(value) for value in holdout_indices)
    for class_index in range(len(EXPECTED_CLASS_COUNTS)):
        class_rows = [
            row.sample_index
            for row in rows
            if row.sample_index in holdout_set and row.target == class_index
        ]
        if len(class_rows) < int(per_class):
            raise ValueError(f"Not enough fold rows for balanced class {class_index}.")
        selected.extend(class_rows[: int(per_class)])
    return selected


def _prompt_map_snapshot(
    *,
    prototype: nn.Module,
    loader: DataLoader,
    device: torch.device,
    focus_class: int,
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    images, targets, metadata = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device=device, dtype=torch.long)
    metadata_value = _metadata_to_device(
        metadata,
        device=device,
        count=int(images.size(0)),
    )
    with torch.inference_mode():
        features, _ = _forward_features_and_logits(
            model,
            images,
            metadata_value,
            device=device,
            return_attention=True,
            return_trace=True,
        )
    prompt_maps_by_layer = features.get("deep_class_prompt_attentions", {})
    if set(prompt_maps_by_layer) != set(range(EXPECTED_BLOCKS)):
        raise ValueError("Deep prompt snapshot is missing layer maps.")
    prompt_maps = prompt_maps_by_layer[EXPECTED_BLOCKS - 1].float()
    prompt_logits = features["deep_class_prompt_logits"].float()
    normalized_vectors = F.normalize(prompt_maps, dim=-1, eps=1e-8)
    cosine = normalized_vectors @ normalized_vectors.transpose(1, 2)
    off_diagonal = ~torch.eye(
        int(prompt_maps.size(1)),
        device=device,
        dtype=torch.bool,
    )
    mean_off_diagonal_cosine = float(cosine[:, off_diagonal].mean().item())
    effective_ranks = []
    for sample_maps in prompt_maps:
        singular_values = torch.linalg.svdvals(sample_maps)
        probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
        effective_ranks.append(
            torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )
        )
    logit_variance = prompt_logits.var(dim=0, correction=0)

    bbox_prior = features.get("patch_bbox_prior")
    if not torch.is_tensor(bbox_prior):
        raise ValueError("Prompt XAI snapshot requires patch_bbox_prior.")
    normalized_maps = prompt_maps / prompt_maps.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    inside = (bbox_prior > 0.0).float()
    core = (bbox_prior >= 0.75).float()
    ring = ((bbox_prior > 0.0) & (bbox_prior < 0.75)).float()
    context = (bbox_prior <= 0.0).float()
    focus_rows = targets == int(focus_class)
    if int(focus_rows.sum().item()) == 0:
        raise ValueError("Balanced prompt snapshot has no focus-class rows.")
    focus_maps = normalized_maps[focus_rows]
    focus_inside = inside[focus_rows]
    focus_core = core[focus_rows]
    focus_ring = ring[focus_rows]
    focus_context = context[focus_rows]
    class_mass_inside = (focus_maps * focus_inside.unsqueeze(1)).sum(dim=-1)
    target_inside = class_mass_inside[:, int(focus_class)]
    confuser_classes = [0, 2, 4]
    confuser_inside = class_mass_inside[:, confuser_classes].amax(dim=1)
    target_map = focus_maps[:, int(focus_class)]
    spatial = {
        "target_bbox_mass": float(target_inside.mean().item()),
        "strongest_confuser_bbox_mass": float(confuser_inside.mean().item()),
        "target_vs_confuser_separation": float(
            (target_inside - confuser_inside).mean().item()
        ),
        "target_core_mass": float((target_map * focus_core).sum(dim=1).mean().item()),
        "target_ring_mass": float((target_map * focus_ring).sum(dim=1).mean().item()),
        "target_context_mass": float(
            (target_map * focus_context).sum(dim=1).mean().item()
        ),
        "focus_rows": int(focus_rows.sum().item()),
    }
    output = {
        "mean_off_diagonal_cosine": mean_off_diagonal_cosine,
        "effective_rank_mean": float(torch.stack(effective_ranks).mean().item()),
        "effective_rank_minimum": float(torch.stack(effective_ranks).amin().item()),
        "prompt_logit_variance_per_class": [
            float(value) for value in logit_variance.tolist()
        ],
        "prompt_logit_minimum_variance": float(logit_variance.amin().item()),
        "spatial": spatial,
        "sample_indices": [
            int(value) for value in metadata["sample_index"].tolist()
        ],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _benchmark_inference(
    *,
    prototype: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
) -> Dict[str, object]:
    model = copy.deepcopy(prototype).to(device).eval()
    images_value = images.to(device)
    metadata_value = _metadata_to_device(
        metadata,
        device=device,
        count=int(images_value.size(0)),
    )

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            return _forward_logits(
                model,
                images_value,
                metadata_value,
                device=device,
            )

    for _ in range(3):
        iteration()
    torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "logits_finite": bool(torch.isfinite(logits).all()),
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
    metadata_value = _metadata_to_device(
        metadata,
        device=device,
        count=int(images_value.size(0)),
    )
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = _forward_logits(
            model,
            images_value,
            metadata_value,
            device=device,
        )
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


class _DeepClassPromptExportWrapper(nn.Module):
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
    wrapper = _DeepClassPromptExportWrapper(candidate)
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
    path = output_dir / "deep_class_prompt_candidate.onnx"
    try:
        return _onnx_compare(
            wrapper=wrapper,
            inputs=(cpu_images, cpu_bbox, cpu_mask),
            input_names=("images", "bbox", "image_mask"),
            path=path,
        )
    except Exception as error:
        return _failed_export(path, error)


def _deep_comparison(
    *,
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    num_classes: int,
    focus_class: int,
) -> Dict[str, object]:
    result = _comparison(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        num_classes=num_classes,
        focus_class=focus_class,
    )
    targets = np.asarray([int(row["target"]) for row in control_rows], dtype=np.int64)
    control = np.asarray(
        [int(row["prediction"]) for row in control_rows], dtype=np.int64
    )
    candidate = np.asarray(
        [int(row["prediction"]) for row in candidate_rows], dtype=np.int64
    )
    restricted = np.isin(targets, np.asarray([0, 2, 4], dtype=np.int64))
    result["transitions"]["restricted_focus_fp_corrected_to_target"] = int(
        (
            restricted
            & (control == int(focus_class))
            & (candidate == targets)
        ).sum()
    )
    return result


def _post_adaptation_diagnostics(
    *,
    keeper: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    resource_images: Tensor,
    resource_targets: Tensor,
    resource_metadata: Mapping[str, object],
    balanced_loader: DataLoader,
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
    control_snapshot = _prompt_map_snapshot(
        prototype=control,
        loader=balanced_loader,
        device=device,
        focus_class=int(args.focus_class),
    )
    candidate_snapshot = _prompt_map_snapshot(
        prototype=candidate,
        loader=balanced_loader,
        device=device,
        focus_class=int(args.focus_class),
    )
    control_spatial = control_snapshot["spatial"]
    candidate_spatial = candidate_snapshot["spatial"]
    spatial_delta = {
        key: float(candidate_spatial[key]) - float(control_spatial[key])
        for key in (
            "target_bbox_mass",
            "strongest_confuser_bbox_mass",
            "target_vs_confuser_separation",
            "target_core_mass",
            "target_ring_mass",
            "target_context_mass",
        )
    }
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
        "control_prompt_xai": control_snapshot,
        "candidate_prompt_xai": candidate_snapshot,
        "prompt_xai_delta": spatial_delta,
        "keeper_inference_benchmark": keeper_benchmark,
        "candidate_inference_benchmark": candidate_benchmark,
        "runtime_ratio": runtime_ratio,
        "training_allocation": training_allocation,
        "peak_vram_gib": float(training_allocation["peak_vram_gib"]),
        "export": export,
    }


def assess_deep_class_prompt_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    adapted: Mapping[str, object],
    illumination: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    delta = adapted["delta"]
    transitions = adapted["transitions"]
    adapted_checks = {
        "macro_f1_delta_gte_0": float(delta["macro_f1"]) >= 0.0,
        "class1_f1_delta_gte_0p005": float(delta["class1_f1"]) >= 0.005,
        "class1_precision_delta_gte_0p005": float(delta["class1_precision"])
        >= 0.005,
        "class1_recall_delta_gte_minus_0p005": float(delta["class1_recall"])
        >= -0.005,
        "restricted_focus_fp_corrected_to_target_gte_2": int(
            transitions["restricted_focus_fp_corrected_to_target"]
        )
        >= 2,
        "focus_fn_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
        >= int(transitions["focus_tp_break"]),
        "corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "maximum_nonfocus_f1_drop_lte_0p005": float(
            adapted["maximum_nonfocus_f1_drop"]
        )
        <= 0.005,
    }
    illumination = list(illumination)
    f1_deltas = [float(row["delta"]["class1_f1"]) for row in illumination]
    precision_deltas = [
        float(row["delta"]["class1_precision"]) for row in illumination
    ]
    recall_deltas = [
        float(row["delta"]["class1_recall"]) for row in illumination
    ]
    illumination_checks = {
        "three_lighting_conditions_exact": len(illumination) == 3,
        "all_lighting_class1_f1_deltas_gte_minus_0p005": all(
            value >= -0.005 for value in f1_deltas
        ),
        "worst_lighting_class1_recall_delta_gte_minus_0p010": min(
            recall_deltas,
            default=-math.inf,
        )
        >= -0.010,
        "lighting_class1_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in precision_deltas
        )
        >= 2,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **adapted_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
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
        if key not in {
            "unadapted_predictions",
            "adapted_predictions",
            "post_forward_rng",
        }
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Locked deep class-prompt Stage A requires CUDA.")
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
        and math.isclose(float(args.logit_scale), 0.10, abs_tol=1e-12)
        and int(args.prompt_init_seed) == 20260715
        and int(args.benchmark_repeats) == 5
        and math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO)
        and math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB)
    )
    if not locked_config_exact:
        raise ValueError(
            "Stage A arguments differ from the precommitted deep prompt protocol."
        )

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
    }
    prompt_cam_root = Path(args.prompt_cam_root).resolve()
    prompt_cam_vit = prompt_cam_root / "model" / "vision_transformer.py"
    prompt_cam_build = prompt_cam_root / "experiment" / "build_model.py"
    prompt_cam_commit = _git_commit(prompt_cam_root)
    if prompt_cam_commit != LOCKED_PROMPT_CAM_COMMIT:
        raise ValueError("Prompt-CAM commit differs from locked source.")
    _verify_sha256(
        prompt_cam_vit,
        LOCKED_PROMPT_CAM_VIT_SHA256,
        "Prompt-CAM vision transformer",
    )
    _verify_sha256(
        prompt_cam_build,
        LOCKED_PROMPT_CAM_BUILD_SHA256,
        "Prompt-CAM model builder",
    )
    mctformer_root = Path(args.mctformer_root).resolve()
    mctformer_source = mctformer_root / "models.py"
    mctformer_commit = _git_commit(mctformer_root)
    if mctformer_commit != LOCKED_MCTFORMER_COMMIT:
        raise ValueError("MCTformer commit differs from locked source.")
    _verify_sha256(
        mctformer_source,
        LOCKED_MCTFORMER_SOURCE_SHA256,
        "MCTformer source",
    )

    repo_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "auditor": Path(__file__).resolve(),
        "module": repo_root / "trkh" / "models" / "deep_class_prompt.py",
        "model": repo_root / "trkh" / "models" / "model.py",
        "config": repo_root / "trkh" / "core" / "config.py",
        "trainer": repo_root / "trkh" / "training" / "train.py",
        "v8_launcher": repo_root / "scripts" / "run_trkh_5class_attention_views_v8.ps1",
        "audit_launcher": repo_root / "scripts" / "run_trkh_deep_class_prompt_readiness.ps1",
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
        raise RuntimeError("Locked deep prompt Stage A requires BF16-capable CUDA.")

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    source_config = checkpoint.get("model_config")
    keeper_state = checkpoint.get("model_state")
    if len(class_names) != len(EXPECTED_CLASS_COUNTS):
        raise ValueError("Keeper class order is invalid.")
    if not isinstance(source_config, Mapping) or not isinstance(keeper_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if bool(source_config.get("deep_class_prompt", False)):
        raise ValueError("Keeper unexpectedly already contains deep class prompts.")
    if int(source_config.get("depth", 0)) != EXPECTED_BLOCKS:
        raise ValueError("Keeper Transformer depth differs from locked protocol.")
    keeper, prompt_prototype, construction = _construct_models(checkpoint, args)

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
        context="deep_class_prompt_stage_a_resource",
        seed=int(args.seed) + 300,
    )
    resource_images, resource_targets, resource_metadata = next(
        iter(resource_loader)
    )
    balanced_indices = _balanced_indices(
        rows,
        holdout_indices,
        per_class=int(args.balanced_per_class),
    )
    balanced_loader, balanced_loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=balanced_indices,
        batch_size=len(balanced_indices),
        num_workers=int(args.num_workers),
        context="deep_class_prompt_stage_a_balanced_xai",
        seed=int(args.seed) + 301,
    )
    pre = _pre_adaptation_diagnostics(
        keeper=keeper,
        prompt_model=prompt_prototype,
        images_cpu=resource_images,
        metadata_cpu=resource_metadata,
        args=args,
        device=device,
    )

    control_model, control_result = _train_variant(
        name="control",
        prototype=prompt_prototype,
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
        prototype=prompt_prototype,
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
    unadapted = _deep_comparison(
        control_rows=control_result["unadapted_predictions"],
        candidate_rows=candidate_result["unadapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )
    adapted = _deep_comparison(
        control_rows=control_result["adapted_predictions"],
        candidate_rows=candidate_result["adapted_predictions"],
        num_classes=len(class_names),
        focus_class=int(args.focus_class),
    )

    condition_predictions: Dict[
        str, Dict[str, Sequence[Mapping[str, object]]]
    ] = {}
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
            context=f"deep_class_prompt_{condition}",
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
        comparison = _deep_comparison(
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
        keeper=keeper,
        control=control_model,
        candidate=candidate_model,
        resource_images=resource_images,
        resource_targets=resource_targets,
        resource_metadata=resource_metadata,
        balanced_loader=balanced_loader,
        args=args,
        output_dir=output_dir,
        device=device,
        amp_dtype=amp_dtype,
    )
    control_movement = control_result["prompt_state_movement"]
    candidate_movement = candidate_result["prompt_state_movement"]
    insertion_removal_exact = all(
        int(row["observed_block_input_tokens"])
        == int(row["expected_block_input_tokens"])
        and int(row["observed_base_tokens_after_removal"])
        == int(row["expected_base_tokens_after_removal"])
        for row in pre["insertion_removal"]
    )
    original_patch_count = int(pre["original_patch_count"])
    native_schema_exact = all(
        int(shape[-1]) == int(pre["base_prefix_count"]) + original_patch_count
        and int(shape[-2]) == int(shape[-1])
        for shape in pre["native_attention_shapes"].values()
    )
    prompt_map_schema_exact = all(
        shape[-2:] == [len(class_names), original_patch_count]
        for shapes in (
            pre["dense_prompt_map_shapes"],
            pre["pruned_prompt_map_shapes"],
        )
        for shape in shapes.values()
    )
    equation_errors = [
        float(value)
        for precision in pre["equation"].values()
        for value in precision.values()
    ]
    candidate_xai = post["candidate_prompt_xai"]
    xai_delta = post["prompt_xai_delta"]
    structural_checks = {
        "locked_config_exact": locked_config_exact,
        "locked_source_and_inputs": True,
        "train_only_provenance": True,
        "all_rows_class_and_fold_counts_exact": True,
        "fit_holdout_source_overlap_zero": len(source_overlap) == 0,
        "state_additions_exact_5": bool(construction["state_additions_exact"]),
        "reported_missing_keys_exact_5": bool(
            construction["reported_missing_keys_exact"]
        ),
        "unexpected_keys_zero": not construction["unexpected_keys"],
        "keeper_state_bit_exact_after_load": bool(
            construction["existing_state_bit_exact"]
        ),
        "added_parameters_exact_11009": int(construction["added_parameters"])
        == EXPECTED_ADDED_PARAMETERS,
        "constructor_rng_equal": bool(construction["constructor_rng_equal"]),
        "forward_rng_equal": bool(pre["forward_rng_equal"]),
        "all_eight_spatial_mhsa_blocks": pre["spatial_mhsa_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1)),
        "prompt_insert_remove_all_eight_blocks": len(pre["insertion_removal"])
        == EXPECTED_BLOCKS
        and insertion_removal_exact,
        "pruning_after_layers_2_5": pre["pruning_layers"] == [2, 5],
        "native_attention_all_eight_legacy_schema": pre["native_attention_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1))
        and native_schema_exact,
        "prompt_maps_all_eight_original_grid": pre["dense_prompt_map_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1))
        and pre["pruned_prompt_map_layers"]
        == list(range(1, EXPECTED_BLOCKS + 1))
        and prompt_map_schema_exact,
        "fp32_bf16_equation_exact": max(equation_errors, default=math.inf)
        <= 1e-6,
        "fp32_all_prompt_gradients_finite_nonzero": bool(
            post["fp32_gradients"]["all_parameter_families"]
        ),
        "bf16_all_prompt_gradients_finite_nonzero": bool(
            post["bf16_gradients"]["all_parameter_families"]
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
        "control_trainable_norm_scalar_only": set(
            control_result["trainable_parameters"]
        )
        == CONTROL_TRAINABLE,
        "candidate_trainable_prompts_norm_scalar_only": set(
            candidate_result["trainable_parameters"]
        )
        == CANDIDATE_TRAINABLE,
        "both_training_gradients_finite_seen": bool(
            control_result["all_gradients_finite"]
            and candidate_result["all_gradients_finite"]
            and control_result["all_trainable_gradients_seen"]
            and candidate_result["all_trainable_gradients_seen"]
        ),
        "keeper_state_bit_exact_after_adaptation": bool(
            control_result["keeper_state_bit_exact"]
            and candidate_result["keeper_state_bit_exact"]
        ),
        "control_prompt_embeddings_exact": not bool(
            control_movement["deep_class_prompt.prompt_embeddings"]["changed"]
        ),
        "candidate_prompt_embeddings_moved": bool(
            candidate_movement["deep_class_prompt.prompt_embeddings"]["changed"]
        ),
        "unadapted_matched_predictions_exact": int(
            unadapted["transitions"]["changed"]
        )
        == 0,
        "candidate_map_offdiag_cosine_lt_0p995": float(
            candidate_xai["mean_off_diagonal_cosine"]
        )
        < 0.995,
        "candidate_map_effective_rank_gt_1p5": float(
            candidate_xai["effective_rank_mean"]
        )
        > 1.5,
        "every_prompt_class_logit_has_variance": float(
            candidate_xai["prompt_logit_minimum_variance"]
        )
        > 0.0,
        "class1_spatial_separation_delta_gte_0p005": float(
            xai_delta["target_vs_confuser_separation"]
        )
        >= 0.005,
        "class1_context_mass_delta_lte_0p02": float(
            xai_delta["target_context_mass"]
        )
        <= 0.02,
        "onnx_error_lte_1e5_argmax_match": bool(post["export"]["succeeded"])
        and float(post["export"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR
        and bool(post["export"]["argmax_match"]),
        "runtime_ratio_lte_1p25": float(post["runtime_ratio"])
        <= float(args.max_runtime_ratio),
        "peak_training_vram_lte_3p25_gib": float(post["peak_vram_gib"])
        <= float(args.max_peak_vram_gib),
    }
    gate = assess_deep_class_prompt_stage_a(
        structural_checks=structural_checks,
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
    extension_path = output_dir / "adapted_prompt_extensions.pt"
    torch.save(
        {
            "method": "deep_class_prompt",
            "protocol_sha256": LOCKED_PROTOCOL_SHA256,
            "keeper_sha256": LOCKED_KEEPER_SHA256,
            "control_state": control_model.deep_class_prompt.state_dict(),
            "candidate_state": candidate_model.deep_class_prompt.state_dict(),
            "control_train_order_sha256": control_result["train_order_sha256"],
            "candidate_train_order_sha256": candidate_result["train_order_sha256"],
        },
        extension_path,
    )

    summary: Dict[str, object] = {
        "method": "deep_class_prompt",
        "protocol_stage": "A_train_only_functional_xai_resource_decision_readiness",
        "test_data_used": False,
        "validation_data_used": False,
        "sources": {
            **{key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "prompt_cam_root": str(prompt_cam_root),
            "prompt_cam_commit": prompt_cam_commit,
            "prompt_cam_sources": {
                "vision_transformer": str(prompt_cam_vit),
                "vision_transformer_sha256": LOCKED_PROMPT_CAM_VIT_SHA256,
                "build_model": str(prompt_cam_build),
                "build_model_sha256": LOCKED_PROMPT_CAM_BUILD_SHA256,
            },
            "mctformer_root": str(mctformer_root),
            "mctformer_commit": mctformer_commit,
            "mctformer_source": str(mctformer_source),
            "mctformer_source_sha256": LOCKED_MCTFORMER_SOURCE_SHA256,
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
            "logit_scale": float(args.logit_scale),
            "prompt_init_seed": int(args.prompt_init_seed),
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
            "balanced_loader": balanced_loader_summary,
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
            "adapted_prompt_extensions": extension_path.name,
            "adapted_prompt_extensions_sha256": _sha256(extension_path),
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
        "# Deep Class-Prompt Stage-A Readiness",
        "",
        f"- All gates passed: `{gate['all_gates_passed']}`",
        f"- Stage-B smoke authorized: `{gate['stage_b_smoke_authorized']}`",
        f"- Failed checks: `{gate['failed_checks']}`",
        f"- Runtime ratio / peak VRAM GiB: `{post['runtime_ratio']:.6f}/{post['peak_vram_gib']:.6f}`",
        f"- ONNX max error: `{post['export']['maximum_absolute_error']:.9g}`",
        f"- Adapted macro/class1 F1 delta: `{adapted['delta']['macro_f1']:.6f}/{adapted['delta']['class1_f1']:.6f}`",
        f"- Adapted class1 precision/recall delta: `{adapted['delta']['class1_precision']:.6f}/{adapted['delta']['class1_recall']:.6f}`",
        f"- Corrections/harms: `{adapted['transitions']['candidate_correction']}/{adapted['transitions']['candidate_harm']}`",
        f"- Restricted FP corrected / FN rescue / TP break: `{adapted['transitions']['restricted_focus_fp_corrected_to_target']}/{adapted['transitions']['focus_fn_rescue']}/{adapted['transitions']['focus_tp_break']}`",
        f"- Prompt map cosine/effective rank: `{candidate_xai['mean_off_diagonal_cosine']:.6f}/{candidate_xai['effective_rank_mean']:.6f}`",
        f"- Class1 separation/context delta: `{xai_delta['target_vs_confuser_separation']:.6f}/{xai_delta['target_context_mass']:.6f}`",
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
            "role": "adapted_prompt_extensions",
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
                "output_dir": str(
                    Path(summary["artifacts"]["candidate_onnx"]).parent
                ),
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
