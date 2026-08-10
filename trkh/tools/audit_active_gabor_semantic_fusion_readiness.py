from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.inference.inference import load_checkpoint
from trkh.models.learnable_gabor_texture import LearnableGaborTextureEncoder
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
)
from trkh.tools.audit_learnable_gabor_texture_readiness import (
    _MaterializedGaborExport,
    _balanced_sensitivity,
    _bbox_from_metadata,
    _benchmark,
    _export_diagnostics,
    _forward_logits,
    _illumination_diagnostics,
    _module_diagnostics,
)
from trkh.tools.audit_moga_tokenizer_readiness import _move_batch
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _load_json_mapping,
    _prepare_output_dir,
    _sha256,
    _state_sha256,
)


EXPECTED_TRAIN_ROWS = 9215
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_LAUNCHER_SHA256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
LOCKED_RESOLVED_CONFIG_SHA256 = (
    "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97"
)
LOCKED_SCRATCH_SHA256 = "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"

MAX_ADDED_PARAMETERS = 100_000
MAX_PEAK_VRAM_GIB = 7.75
MAX_RUNTIME_RATIO = 1.50
MIN_ACTIVE_LOGIT_DELTA = 1e-6
MIN_NORM_RATIO = 0.50
MAX_NORM_RATIO = 3.00
MIN_FUSED_NORM_RATIO = 0.75
MAX_FUSED_NORM_RATIO = 3.00
MIN_LOSS_REDUCTION = 0.10
MIN_DIVERSITY_IMPROVEMENT = 1e-4
MICRO_STEPS = 16
MICRO_LR = 2.5e-4
MICRO_WEIGHT_DECAY = 0.05
BENCHMARK_REPEATS = 3
MAX_MATERIALIZED_ERROR = 1e-6
MAX_ONNX_ERROR = 1e-5


ACTIVE_GRADIENT_FAMILIES = {
    "gabor_parameters": (
        "raw_theta",
        "raw_frequency",
        "raw_sigma_x",
        "raw_sigma_y",
    ),
    "lho_projection": ("level_projection.", "position_projection."),
    "lho_attention": ("lho_attention.", "lho_norm."),
    "fcm_parameter_encoding": ("filter_parameter_projection.",),
    "fcm_attention": ("fcm_attention.", "fcm_norm."),
    "output_projection": ("output_norm.", "output_projection.", "token_norm."),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only readiness audit for active Gabor texture plus TRKH "
            "semantic feature fusion. Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--scratch-checkpoint",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/"
            "checkpoints/best.pt"
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
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        default=Path("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_ACTIVE_GABOR_SEMANTIC_FUSION_PROTOCOL_20260715.md"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "learnable_gabor_texture_residual": False,
            "learnable_gabor_texture_semantic_fusion": False,
            "pretrained": False,
        }
    )
    return config


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _control_config(source)
    config["learnable_gabor_texture_semantic_fusion"] = True
    return config


def _rng_state_sha256(state: Tensor) -> str:
    payload = state.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def summarize_active_gradients(
    module: LearnableGaborTextureEncoder,
) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(module.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in ACTIVE_GRADIENT_FAMILIES.items():
        selected = {
            name: parameter
            for name, parameter in named_parameters.items()
            if any(name == marker or name.startswith(marker) for marker in markers)
        }
        gradients = {
            name: parameter.grad
            for name, parameter in selected.items()
            if parameter.grad is not None
        }
        finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients.values()
        )
        nonzero = sum(
            int(torch.count_nonzero(gradient.detach()).item())
            for gradient in gradients.values()
        )
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(gradients),
            "finite": finite,
            "nonzero_elements": int(nonzero),
            "l2_norm": float(
                math.sqrt(
                    sum(
                        float(gradient.detach().float().square().sum().item())
                        for gradient in gradients.values()
                    )
                )
            ),
            "passed": bool(
                selected
                and len(gradients) == len(selected)
                and finite
                and nonzero > 0
            ),
        }
    return summary


def _family_parameter_movement(
    before: Mapping[str, Tensor],
    module: LearnableGaborTextureEncoder,
) -> Dict[str, Dict[str, object]]:
    current = dict(module.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in ACTIVE_GRADIENT_FAMILIES.items():
        selected = [
            name
            for name in before
            if any(name == marker or name.startswith(marker) for marker in markers)
        ]
        absolute_sum = 0.0
        maximum = 0.0
        changed_tensors = 0
        for name in selected:
            delta = current[name].detach().float().cpu() - before[name].float()
            absolute_sum += float(delta.abs().sum().item())
            maximum = max(maximum, float(delta.abs().max().item()))
            changed_tensors += int(torch.count_nonzero(delta).item() > 0)
        summary[family] = {
            "parameter_tensors": len(selected),
            "changed_tensors": int(changed_tensors),
            "absolute_sum": float(absolute_sum),
            "maximum_absolute": float(maximum),
            "passed": bool(selected and changed_tensors == len(selected) and absolute_sum > 0.0),
        }
    return summary


def _mean_off_diagonal_cosine(features: Tensor) -> float:
    values = features.detach().float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or int(values.size(1)) < 2:
        raise ValueError("Expected [B,N,D] or [N,D] features with N >= 2")
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.transpose(-2, -1)
    mask = ~torch.eye(
        int(values.size(1)),
        device=values.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    selected = similarity.masked_select(mask.expand_as(similarity))
    return float(selected.mean().item())


def _active_feature_diagnostics(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        features = model.forward_features(
            images,
            image_valid_mask=metadata.get("image_mask"),
            bbox_token_prior=_bbox_from_metadata(metadata),
            return_trace=True,
        )
        texture = features.get("gabor_texture_semantic_feature")
        if not torch.is_tensor(texture):
            raise RuntimeError("Candidate did not emit the active texture feature")
        semantic_features = dict(features)
        semantic_features.pop("gabor_texture_semantic_feature", None)
        semantic_features["trace"] = {}
        semantic = model.head_input_from_features(semantic_features)
        fused = model.head_input_from_features(features)
        texture = texture.to(device=semantic.device, dtype=semantic.dtype)
        expected = semantic + texture
        semantic_logits = model.head(semantic)
        fused_logits = model.head(fused)

    semantic_float = semantic.float()
    texture_float = texture.float()
    fused_float = fused.float()
    semantic_norm = semantic_float.norm(dim=-1).clamp_min(1e-8)
    texture_norm = texture_float.norm(dim=-1)
    fused_norm = fused_float.norm(dim=-1)
    return {
        "semantic": semantic.detach(),
        "texture": texture.detach(),
        "fused": fused.detach(),
        "direct_sum_max_error": float((fused - expected).abs().max().item()),
        "main_logit_mean_absolute_delta": float(
            (fused_logits - semantic_logits).abs().mean().item()
        ),
        "main_logit_max_absolute_delta": float(
            (fused_logits - semantic_logits).abs().max().item()
        ),
        "texture_semantic_norm_ratio_mean": float(
            (texture_norm / semantic_norm).mean().item()
        ),
        "texture_semantic_norm_ratio_minimum": float(
            (texture_norm / semantic_norm).min().item()
        ),
        "texture_semantic_norm_ratio_maximum": float(
            (texture_norm / semantic_norm).max().item()
        ),
        "fused_semantic_norm_ratio_mean": float(
            (fused_norm / semantic_norm).mean().item()
        ),
        "semantic_texture_cosine_mean": float(
            F.cosine_similarity(semantic_float, texture_float, dim=-1).mean().item()
        ),
        "texture_sample_mean_off_diagonal_cosine": _mean_off_diagonal_cosine(
            texture_float.unsqueeze(0)
        ),
        "semantic_finite": bool(torch.isfinite(semantic).all()),
        "texture_finite": bool(torch.isfinite(texture).all()),
        "fused_finite": bool(torch.isfinite(fused).all()),
    }


def _response_gradient_diagnostics(
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    bbox: Tensor,
) -> Dict[str, Dict[str, object]]:
    module.zero_grad(set_to_none=True)
    probe = images.detach().clone().requires_grad_(True)
    components = module.forward_components(probe, bbox)
    components["real_response"].retain_grad()
    components["imaginary_response"].retain_grad()
    components["texture_token"].square().mean().backward()
    summary: Dict[str, Dict[str, object]] = {}
    for name in ("real_response", "imaginary_response"):
        gradient = components[name].grad
        summary[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero_elements": int(
                torch.count_nonzero(gradient.detach()).item()
                if gradient is not None
                else 0
            ),
        }
    return summary


def _head_and_semantic_gradient_diagnostics(model: nn.Module) -> Dict[str, object]:
    head_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("head.")
    }
    semantic_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("gabor_texture_semantic_encoder.")
        and not name.startswith("head.")
    }

    def summarize(parameters: Mapping[str, nn.Parameter]) -> Dict[str, object]:
        gradients = [parameter.grad for parameter in parameters.values() if parameter.grad is not None]
        return {
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(gradients),
            "finite": bool(gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            ),
            "nonzero_elements": int(
                sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients)
            ),
        }

    return {
        "main_head": summarize(head_parameters),
        "shared_semantic_path": summarize(semantic_parameters),
    }


def _micro_train_diagnostics(
    *,
    model_config: Mapping[str, object],
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(int(seed), deterministic=True)
    model = create_model(num_classes=5, model_config=model_config).to(device)
    model.eval()
    module = getattr(model, "gabor_texture_semantic_encoder", None)
    if not isinstance(module, LearnableGaborTextureEncoder):
        raise TypeError("Micro-train candidate lacks LearnableGaborTextureEncoder")
    bbox = _bbox_from_metadata(metadata)
    if bbox is None:
        raise RuntimeError("Micro-train batch lacks bbox metadata")

    before_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in module.named_parameters()
    }
    with torch.no_grad():
        before_components = module.forward_components(images, bbox)
        before_active = _active_feature_diagnostics(model, images, metadata)
        before_logits, _ = _forward_logits(model, images, metadata)
        before_loss = F.cross_entropy(before_logits.float(), labels)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MICRO_LR,
        weight_decay=MICRO_WEIGHT_DECAY,
    )
    losses = []
    first_step_gradients = None
    for step in range(MICRO_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _forward_logits(model, images, metadata)
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        if step == 0:
            first_step_gradients = summarize_active_gradients(module)
        optimizer.step()
        losses.append(float(loss.detach().item()))

    with torch.no_grad():
        after_components = module.forward_components(images, bbox)
        after_active = _active_feature_diagnostics(model, images, metadata)
        after_logits, _ = _forward_logits(model, images, metadata)
        after_loss = F.cross_entropy(after_logits.float(), labels)

    before_texture_cosine = _mean_off_diagonal_cosine(
        before_components["texture_token"].unsqueeze(0)
    )
    after_texture_cosine = _mean_off_diagonal_cosine(
        after_components["texture_token"].unsqueeze(0)
    )
    before_filter_cosine = _mean_off_diagonal_cosine(
        before_components["filter_features"]
    )
    after_filter_cosine = _mean_off_diagonal_cosine(
        after_components["filter_features"]
    )
    loss_reduction = float(
        (float(before_loss.item()) - float(after_loss.item()))
        / max(float(before_loss.item()), 1e-8)
    )
    movement = _family_parameter_movement(before_parameters, module)
    raw_names = ("raw_theta", "raw_frequency", "raw_sigma_x", "raw_sigma_y")
    raw_movement = {
        name: float(
            (
                dict(module.named_parameters())[name].detach().float().cpu()
                - before_parameters[name]
            ).abs().sum().item()
        )
        for name in raw_names
    }
    result = {
        "mode": "eval_deterministic_full_candidate_updates",
        "steps": MICRO_STEPS,
        "optimizer": "AdamW",
        "learning_rate": MICRO_LR,
        "weight_decay": MICRO_WEIGHT_DECAY,
        "loss_before": float(before_loss.item()),
        "step_losses": losses,
        "loss_after": float(after_loss.item()),
        "loss_reduction_fraction": loss_reduction,
        "first_step_gradients": first_step_gradients,
        "parameter_movement": movement,
        "raw_parameter_absolute_movement": raw_movement,
        "texture_sample_cosine_before": before_texture_cosine,
        "texture_sample_cosine_after": after_texture_cosine,
        "texture_sample_cosine_decrease": float(
            before_texture_cosine - after_texture_cosine
        ),
        "filter_feature_cosine_before": before_filter_cosine,
        "filter_feature_cosine_after": after_filter_cosine,
        "filter_feature_cosine_decrease": float(
            before_filter_cosine - after_filter_cosine
        ),
        "active_feature_before": {
            key: value
            for key, value in before_active.items()
            if not torch.is_tensor(value)
        },
        "active_feature_after": {
            key: value
            for key, value in after_active.items()
            if not torch.is_tensor(value)
        },
        "finite": bool(
            math.isfinite(float(before_loss.item()))
            and math.isfinite(float(after_loss.item()))
            and all(math.isfinite(value) for value in losses)
        ),
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


class _MaterializedActiveFusionExport(nn.Module):
    def __init__(self, module: LearnableGaborTextureEncoder) -> None:
        super().__init__()
        self.texture = _MaterializedGaborExport(module)

    def forward(self, image: Tensor, bbox: Tensor, semantic: Tensor) -> Tensor:
        return semantic + self.texture(image, bbox).to(dtype=semantic.dtype)


def _active_sum_export_diagnostics(
    *,
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    bbox: Tensor,
    semantic: Tensor,
    output_dir: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    wrapper = _MaterializedActiveFusionExport(module).cpu().eval()
    dynamic_module = copy.deepcopy(module).cpu().eval()
    cpu_images = images[:2].detach().float().cpu()
    cpu_bbox = bbox[:2].detach().float().cpu()
    cpu_semantic = semantic[:2].detach().float().cpu()
    with torch.no_grad():
        expected = wrapper(cpu_images, cpu_bbox, cpu_semantic)
        dynamic = cpu_semantic + dynamic_module(cpu_images, cpu_bbox)
    dynamic_materialized_error = float((dynamic - expected).abs().max().item())

    path = output_dir / "active_gabor_semantic_sum_materialized.onnx"
    torch.onnx.export(
        wrapper,
        (cpu_images, cpu_bbox, cpu_semantic),
        str(path),
        input_names=["image", "bbox", "semantic"],
        output_names=["fused"],
        dynamic_axes={
            "image": {0: "batch"},
            "bbox": {0: "batch"},
            "semantic": {0: "batch"},
            "fused": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    results = []
    maximum_error = 0.0
    for batch_size in (1, 2):
        image_value = cpu_images[:batch_size]
        bbox_value = cpu_bbox[:batch_size]
        semantic_value = cpu_semantic[:batch_size]
        with torch.no_grad():
            reference = wrapper(image_value, bbox_value, semantic_value).numpy()
        observed = session.run(
            ["fused"],
            {
                "image": image_value.numpy(),
                "bbox": bbox_value.numpy(),
                "semantic": semantic_value.numpy(),
            },
        )[0]
        error = float(np.max(np.abs(reference - observed)))
        maximum_error = max(maximum_error, error)
        results.append(
            {
                "batch_size": int(batch_size),
                "maximum_absolute_error": error,
                "finite": bool(np.isfinite(observed).all()),
                "shape": [int(value) for value in observed.shape],
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "opset": 17,
        "dynamic_materialized_max_error": dynamic_materialized_error,
        "onnx_max_error": maximum_error,
        "batch_results": results,
    }


def assess_active_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    runtime_ratio: float,
) -> Dict[str, object]:
    resolved = {str(key): bool(value) for key, value in checks.items()}
    resolved["peak_vram_within_budget"] = bool(
        math.isfinite(float(peak_vram_gib))
        and float(peak_vram_gib) <= MAX_PEAK_VRAM_GIB
    )
    resolved["runtime_ratio_within_budget"] = bool(
        math.isfinite(float(runtime_ratio))
        and float(runtime_ratio) <= MAX_RUNTIME_RATIO
    )
    failed = [name for name, passed in resolved.items() if not passed]
    return {
        "smoke_permission": not failed,
        "probe_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": resolved,
        "failed_checks": failed,
        "thresholds": {
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
            "max_added_parameters": MAX_ADDED_PARAMETERS,
            "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            "max_runtime_ratio": MAX_RUNTIME_RATIO,
            "min_active_logit_delta": MIN_ACTIVE_LOGIT_DELTA,
            "texture_semantic_norm_ratio": [MIN_NORM_RATIO, MAX_NORM_RATIO],
            "fused_semantic_norm_ratio": [
                MIN_FUSED_NORM_RATIO,
                MAX_FUSED_NORM_RATIO,
            ],
            "micro_steps": MICRO_STEPS,
            "micro_loss_reduction": MIN_LOSS_REDUCTION,
            "micro_diversity_improvement": MIN_DIVERSITY_IMPROVEMENT,
            "maximum_materialized_error": MAX_MATERIALIZED_ERROR,
            "maximum_onnx_error": MAX_ONNX_ERROR,
        },
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    paths = {
        "scratch_checkpoint": Path(args.scratch_checkpoint).resolve(),
        "keeper_checkpoint": Path(args.keeper_checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "resolved_config": Path(args.resolved_config).resolve(),
        "current_command": Path(args.command_file).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "data_yaml": Path(args.data).resolve(),
    }
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Active Gabor Stage A requires CUDA")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked Stage-A batch sizes are AMP=32 and FP32=2")
    if int(args.seed) != 42:
        raise ValueError("Locked Stage-A seed is 42")
    if int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError(f"Locked benchmark repeats are {BENCHMARK_REPEATS}")

    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    checkpoint = load_checkpoint(paths["scratch_checkpoint"], map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(source_model_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Scratch checkpoint lacks model_config/class_names")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}")
    resolved_config = _load_json_mapping(paths["resolved_config"])
    launcher_args = _load_json_mapping(paths["launcher_args"])
    control_config = _control_config(source_model_config)
    candidate_config = _candidate_config(source_model_config)

    implicit_config = dict(source_model_config)
    implicit_config.pop("learnable_gabor_texture_residual", None)
    implicit_config.pop("learnable_gabor_texture_semantic_fusion", None)
    implicit_config["pretrained"] = False

    set_seed(int(args.seed), deterministic=True)
    implicit_control = create_model(5, implicit_config)
    implicit_rng = torch.get_rng_state().clone()
    implicit_state_sha = _state_sha256(implicit_control)
    set_seed(int(args.seed), deterministic=True)
    control_model = create_model(5, control_config)
    control_rng = torch.get_rng_state().clone()
    control_state_sha = _state_sha256(control_model)
    set_seed(int(args.seed), deterministic=True)
    candidate_model = create_model(5, candidate_config)
    candidate_rng = torch.get_rng_state().clone()
    candidate_state_sha = _state_sha256(candidate_model)
    set_seed(int(args.seed), deterministic=True)
    repeated_candidate = create_model(5, candidate_config)
    repeated_candidate_rng = torch.get_rng_state().clone()
    repeated_candidate_state_sha = _state_sha256(repeated_candidate)

    control_state = control_model.state_dict()
    candidate_state = candidate_model.state_dict()
    shared_names = [name for name in candidate_state if name in control_state]
    candidate_only_names = [name for name in candidate_state if name not in control_state]
    control_only_names = [name for name in control_state if name not in candidate_state]
    shared_exact = bool(shared_names) and all(
        torch.equal(candidate_state[name], control_state[name])
        for name in shared_names
    )
    roundtrip = build_model_from_checkpoint(
        {
            "class_names": list(class_names),
            "model_config": dict(candidate_config),
            "model_state": candidate_state,
        }
    )
    roundtrip_state_sha = _state_sha256(roundtrip)
    del repeated_candidate, roundtrip

    dataset = _build_train_only_dataset(
        data_yaml=paths["data_yaml"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    dataset.classification_bbox_metadata = True
    data_spec = load_data_spec(paths["data_yaml"])
    train_root = Path(data_spec.split_images_dir("train")).resolve()
    balanced_indices: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        label = int(sample.primary_label)
        if label not in balanced_indices:
            balanced_indices[label] = int(index)
        if len(balanced_indices) == 5:
            break
    if set(balanced_indices) != set(range(5)):
        raise RuntimeError("Could not construct one train object per class")
    ordered_indices = [balanced_indices[index] for index in range(5)]
    balanced_batch = default_collate([dataset[index] for index in ordered_indices])
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="active_gabor_semantic_fusion_stage_a_train_only",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    resource_batch = next(iter(loader))
    device = torch.device("cuda")
    images, labels, metadata = _move_batch(resource_batch, device=device)
    balanced_images, balanced_labels, balanced_metadata = _move_batch(
        balanced_batch,
        device=device,
    )
    bbox = _bbox_from_metadata(balanced_metadata)
    if bbox is None:
        raise RuntimeError("Stage-A train objects lack transformed bbox metadata")

    implicit_control = implicit_control.to(device).eval()
    control_model = control_model.to(device).eval()
    candidate_model = candidate_model.to(device).eval()
    with torch.no_grad():
        implicit_logits, _ = _forward_logits(
            implicit_control,
            balanced_images,
            balanced_metadata,
        )
        control_features = control_model.forward_features(
            balanced_images,
            image_valid_mask=balanced_metadata.get("image_mask"),
            bbox_token_prior=bbox,
        )
        control_logits = classification_logits_from_features(
            control_model,
            control_features,
        )
        candidate_features = candidate_model.forward_features(
            balanced_images,
            image_valid_mask=balanced_metadata.get("image_mask"),
            bbox_token_prior=bbox,
        )
        semantic_candidate_features = dict(candidate_features)
        semantic_candidate_features.pop("gabor_texture_semantic_feature", None)
        semantic_candidate_logits = classification_logits_from_features(
            candidate_model,
            semantic_candidate_features,
        )
        active_candidate_logits = classification_logits_from_features(
            candidate_model,
            candidate_features,
        )
    default_control_error = float((implicit_logits - control_logits).abs().max().item())
    shared_semantic_error = float(
        (control_logits - semantic_candidate_logits).abs().max().item()
    )
    active_full_logit_delta = float(
        (active_candidate_logits - control_logits).abs().mean().item()
    )
    active_features = _active_feature_diagnostics(
        candidate_model,
        balanced_images,
        balanced_metadata,
    )
    del implicit_control
    gc.collect()
    torch.cuda.empty_cache()

    module = getattr(candidate_model, "gabor_texture_semantic_encoder", None)
    if not isinstance(module, LearnableGaborTextureEncoder):
        raise TypeError("Candidate did not construct LearnableGaborTextureEncoder")
    if hasattr(module, "raw_gate"):
        raise RuntimeError("Active texture encoder must not contain raw_gate")
    control_parameter_count = sum(parameter.numel() for parameter in control_model.parameters())
    candidate_parameter_count = sum(parameter.numel() for parameter in candidate_model.parameters())
    added_parameters = int(candidate_parameter_count - control_parameter_count)
    module_parameters = int(sum(parameter.numel() for parameter in module.parameters()))
    constrained = module.constrained_parameters()
    midpoint = float(module.maximum_frequency / 2.0)
    module_diagnostics = _module_diagnostics(module, balanced_images, bbox)
    illumination = _illumination_diagnostics(
        module,
        balanced_images,
        bbox,
        balanced_labels,
    )
    sensitivity = _balanced_sensitivity(
        model=candidate_model,
        module=module,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
    )
    response_gradients = _response_gradient_diagnostics(
        module,
        balanced_images[:2],
        bbox[:2],
    )

    candidate_model.eval()
    candidate_model.zero_grad(set_to_none=True)
    fp32_logits, _ = _forward_logits(
        candidate_model,
        images[: int(args.fp32_batch_size)],
        {key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()},
    )
    fp32_loss = F.cross_entropy(
        fp32_logits.float(),
        labels[: int(args.fp32_batch_size)],
    )
    fp32_loss.backward()
    fp32_gradients = summarize_active_gradients(module)
    fp32_head_semantic_gradients = _head_and_semantic_gradient_diagnostics(
        candidate_model
    )

    micro = _micro_train_diagnostics(
        model_config=candidate_config,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
        seed=int(args.seed),
        device=device,
    )

    control_benchmark = _benchmark(
        model=control_model,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    del control_model
    gc.collect()
    torch.cuda.empty_cache()
    candidate_benchmark = _benchmark(
        model=candidate_model,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    bf16_gradients = summarize_active_gradients(module)
    runtime_ratio = float(
        float(candidate_benchmark["median_seconds"])
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )

    active_export = _active_sum_export_diagnostics(
        module=module,
        images=balanced_images,
        bbox=bbox,
        semantic=active_features["semantic"],
        output_dir=output_dir,
    )
    gabor_export = _export_diagnostics(
        module=module,
        images=balanced_images,
        bbox=bbox,
        output_dir=output_dir,
    )

    balanced_paths = [
        str(dataset.samples[index].image_path.resolve()) for index in ordered_indices
    ]
    train_only_paths = all(
        Path(path).is_relative_to(train_root) for path in balanced_paths
    )
    parameter_constraints_valid = bool(
        torch.isfinite(constrained["normalized"]).all()
        and (constrained["theta"] >= 0.0).all()
        and (constrained["theta"] <= math.pi).all()
        and (constrained["sigma_x"] >= constrained["sigma_x_lower"]).all()
        and (constrained["sigma_x"] <= constrained["sigma_x_upper"] + 1e-6).all()
        and (constrained["sigma_y"] >= module.sigma_y_bounds[0]).all()
        and (constrained["sigma_y"] <= module.sigma_y_bounds[1]).all()
    )
    frequency_split_valid = bool(
        (constrained["frequency"][: module.low_filter_count] <= midpoint).all()
        and (constrained["frequency"][module.low_filter_count :] >= midpoint).all()
        and module.low_filter_count == module.high_filter_count == 16
    )
    micro_diversity_improvement = max(
        float(micro["texture_sample_cosine_decrease"]),
        float(micro["filter_feature_cosine_decrease"]),
    )
    norm_ratio = float(active_features["texture_semantic_norm_ratio_mean"])
    fused_ratio = float(active_features["fused_semantic_norm_ratio_mean"])
    checks = {
        "locked_data_sha256": source_hashes["data_yaml"] == LOCKED_DATA_SHA256,
        "locked_launcher_sha256": source_hashes["launcher_args"] == LOCKED_LAUNCHER_SHA256,
        "locked_resolved_config_sha256": (
            source_hashes["resolved_config"] == LOCKED_RESOLVED_CONFIG_SHA256
        ),
        "locked_scratch_sha256": (
            source_hashes["scratch_checkpoint"] == LOCKED_SCRATCH_SHA256
        ),
        "locked_keeper_sha256": (
            source_hashes["keeper_checkpoint"] == LOCKED_KEEPER_SHA256
        ),
        "locked_current_command_sha256": (
            source_hashes["current_command"] == LOCKED_COMMAND_SHA256
        ),
        "launcher_declares_no_resume": not bool(launcher_args.get("resume_checkpoint")),
        "candidate_pretrained_disabled": not bool(candidate_config.get("pretrained", False)),
        "train_split_only": bool(train_only_paths),
        "train_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "balanced_real_train_batch": (
            [int(value) for value in balanced_labels.detach().cpu().tolist()]
            == list(range(5))
        ),
        "transformed_bbox_metadata_present": torch.is_tensor(
            balanced_metadata.get("crop_bbox")
        ),
        "default_off_state_exact": implicit_state_sha == control_state_sha,
        "default_off_rng_exact": torch.equal(implicit_rng, control_rng),
        "default_off_logits_exact": default_control_error == 0.0,
        "candidate_shared_schema_exact": (
            not control_only_names
            and bool(candidate_only_names)
            and len(shared_names) == len(control_state)
        ),
        "candidate_shared_tensors_exact": shared_exact,
        "candidate_only_schema": all(
            name.startswith("gabor_texture_semantic_encoder.")
            for name in candidate_only_names
        ),
        "candidate_rng_neutral": torch.equal(control_rng, candidate_rng),
        "candidate_repeat_rng_exact": torch.equal(candidate_rng, repeated_candidate_rng),
        "candidate_repeat_state_exact": candidate_state_sha == repeated_candidate_state_sha,
        "candidate_checkpoint_roundtrip": candidate_state_sha == roundtrip_state_sha,
        "shared_semantic_logits_exact": shared_semantic_error == 0.0,
        "active_encoder_has_no_gate": not hasattr(module, "raw_gate"),
        "active_direct_sum_exact": float(active_features["direct_sum_max_error"]) == 0.0,
        "active_main_logits_material": (
            float(active_features["main_logit_mean_absolute_delta"])
            > MIN_ACTIVE_LOGIT_DELTA
            and active_full_logit_delta > MIN_ACTIVE_LOGIT_DELTA
        ),
        "active_features_finite": bool(
            active_features["semantic_finite"]
            and active_features["texture_finite"]
            and active_features["fused_finite"]
        ),
        "texture_semantic_norm_ratio": MIN_NORM_RATIO <= norm_ratio <= MAX_NORM_RATIO,
        "fused_semantic_norm_ratio": (
            MIN_FUSED_NORM_RATIO <= fused_ratio <= MAX_FUSED_NORM_RATIO
        ),
        "added_parameters_within_budget": 0 < added_parameters <= MAX_ADDED_PARAMETERS,
        "module_parameter_accounting": added_parameters == module_parameters,
        "gabor_parameter_constraints": parameter_constraints_valid,
        "exact_low_high_split": frequency_split_valid,
        "kernels_finite_normalized": bool(
            module_diagnostics["kernels"]["real"]["finite"]
            and module_diagnostics["kernels"]["imaginary"]["finite"]
            and module_diagnostics["kernels"]["maximum_absolute_mean"] <= 1e-5
            and module_diagnostics["kernels"]["maximum_l2_error"] <= 1e-5
        ),
        "kernels_diverse": (
            module_diagnostics["kernels"]["maximum_absolute_pair_cosine"] < 0.9999
        ),
        "responses_live": bool(
            module_diagnostics["responses"]["real"]["finite"]
            and module_diagnostics["responses"]["imaginary"]["finite"]
            and module_diagnostics["responses"]["real"]["rms"] > 1e-4
            and module_diagnostics["responses"]["imaginary"]["rms"] > 1e-4
            and module_diagnostics["responses"]["magnitude"]["std"] > 1e-4
        ),
        "response_gradients_live": all(
            bool(value["finite"]) and int(value["nonzero_elements"]) > 0
            for value in response_gradients.values()
        ),
        "lho_counts_normalized": (
            module_diagnostics["histogram"]["maximum_count_sum_error"] <= 1e-5
        ),
        "lho_entropy_noncollapsed": (
            module_diagnostics["histogram"]["minimum_entropy"] >= 0.25
        ),
        "mechanism_ablations_material": all(
            math.isfinite(float(value)) and float(value) > 1e-6
            for value in module_diagnostics["ablation_deltas"].values()
        ),
        "bbox_response_concentration": (
            module_diagnostics["responses"]["foreground_mass_minus_area"] > 0.0
        ),
        "bbox_descriptor_sensitivity": (
            module_diagnostics["bbox_full_frame_descriptor_delta"] > 1e-6
        ),
        "illumination_mean_cosine": illumination["mean_cosine"] >= 0.90,
        "illumination_per_class_cosine": (
            illumination["minimum_class_condition_cosine"] >= 0.85
        ),
        "balanced_five_class_sensitivity": bool(sensitivity["all_classes_live"]),
        "fp32_logits_loss_finite": bool(
            torch.isfinite(fp32_logits).all() and torch.isfinite(fp32_loss)
        ),
        "fp32_all_active_gradient_families": all(
            bool(value["passed"]) for value in fp32_gradients.values()
        ),
        "fp32_main_head_gradient_live": bool(
            fp32_head_semantic_gradients["main_head"]["finite"]
            and fp32_head_semantic_gradients["main_head"]["nonzero_elements"] > 0
        ),
        "fp32_shared_semantic_gradient_live": bool(
            fp32_head_semantic_gradients["shared_semantic_path"]["finite"]
            and fp32_head_semantic_gradients["shared_semantic_path"]["nonzero_elements"] > 0
        ),
        "bf16_control_logits_loss_finite": bool(control_benchmark["logits_loss_finite"]),
        "bf16_candidate_logits_loss_finite": bool(candidate_benchmark["logits_loss_finite"]),
        "bf16_all_active_gradient_families": all(
            bool(value["passed"]) for value in bf16_gradients.values()
        ),
        "micro_updates_finite": bool(micro["finite"]),
        "micro_loss_reduction": (
            float(micro["loss_reduction_fraction"]) >= MIN_LOSS_REDUCTION
        ),
        "micro_all_gradient_families": all(
            bool(value["passed"])
            for value in micro["first_step_gradients"].values()
        ),
        "micro_all_parameter_families_move": all(
            bool(value["passed"]) for value in micro["parameter_movement"].values()
        ),
        "micro_all_raw_parameters_move": all(
            float(value) > 0.0
            for value in micro["raw_parameter_absolute_movement"].values()
        ),
        "micro_representation_diversifies": (
            micro_diversity_improvement >= MIN_DIVERSITY_IMPROVEMENT
        ),
        "micro_active_contribution_material": (
            float(
                micro["active_feature_after"]["main_logit_mean_absolute_delta"]
            )
            > MIN_ACTIVE_LOGIT_DELTA
        ),
        "materialized_kernel_equivalence": bool(
            module_diagnostics["materialized_kernel_max_error"]
            <= MAX_MATERIALIZED_ERROR
            and gabor_export["dynamic_materialized_max_error"]
            <= MAX_MATERIALIZED_ERROR
            and active_export["dynamic_materialized_max_error"]
            <= MAX_MATERIALIZED_ERROR
        ),
        "gabor_onnx_cpu_equivalence": bool(
            gabor_export["onnx_max_error"] <= MAX_ONNX_ERROR
            and all(value["finite"] for value in gabor_export["batch_results"])
        ),
        "active_sum_onnx_cpu_equivalence": bool(
            active_export["onnx_max_error"] <= MAX_ONNX_ERROR
            and all(value["finite"] for value in active_export["batch_results"])
        ),
        "onnx_dynamic_batch_1_2": (
            [value["batch_size"] for value in active_export["batch_results"]]
            == [1, 2]
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_active_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        runtime_ratio=runtime_ratio,
    )

    summary: Dict[str, object] = {
        "method": "active_gabor_lho_fcm_semantic_sum",
        "protocol_stage": "A_train_only_active_fusion_readiness",
        "sources": {
            **{name: str(path) for name, path in paths.items()},
            "hashes": source_hashes,
            "split_loaded": "train",
            "validation_loaded": False,
            "test_loaded": False,
            "raw_dataset_modified": False,
        },
        "model": {
            "class_names": [str(value) for value in class_names],
            "control_parameters": int(control_parameter_count),
            "candidate_parameters": int(candidate_parameter_count),
            "candidate_added_parameters": added_parameters,
            "gabor_module_parameters": module_parameters,
            "implicit_control_state_sha256": implicit_state_sha,
            "explicit_control_state_sha256": control_state_sha,
            "candidate_state_sha256": candidate_state_sha,
            "repeated_candidate_state_sha256": repeated_candidate_state_sha,
            "roundtrip_state_sha256": roundtrip_state_sha,
            "candidate_only_keys": candidate_only_names,
            "control_only_keys": control_only_names,
            "shared_key_count": len(shared_names),
            "post_constructor_rng": {
                "implicit_control_sha256": _rng_state_sha256(implicit_rng),
                "explicit_control_sha256": _rng_state_sha256(control_rng),
                "candidate_sha256": _rng_state_sha256(candidate_rng),
                "repeated_candidate_sha256": _rng_state_sha256(
                    repeated_candidate_rng
                ),
                "control_candidate_differing_bytes": int(
                    torch.count_nonzero(control_rng != candidate_rng).item()
                ),
            },
            "default_control_max_logit_error": default_control_error,
            "shared_semantic_max_logit_error": shared_semantic_error,
            "active_full_logit_mean_absolute_delta": active_full_logit_delta,
        },
        "data": {
            "train_rows": len(dataset),
            "train_root": str(train_root),
            "balanced_indices": ordered_indices,
            "balanced_labels": [
                int(value) for value in balanced_labels.detach().cpu().tolist()
            ],
            "balanced_paths": balanced_paths,
            "resource_batch_size": int(images.size(0)),
            "resource_labels": [int(value) for value in labels.detach().cpu().tolist()],
            "bbox_source": "crop_bbox",
            "dataloader": loader_summary,
        },
        "active_feature": {
            key: value
            for key, value in active_features.items()
            if not torch.is_tensor(value)
        },
        "module_diagnostics": module_diagnostics,
        "illumination": illumination,
        "balanced_sensitivity": sensitivity,
        "gradients": {
            "response_intermediates": response_gradients,
            "fp32_active": fp32_gradients,
            "fp32_head_and_semantic": fp32_head_semantic_gradients,
            "bf16_active": bf16_gradients,
        },
        "micro_train": micro,
        "cuda_bf16": {
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_gib": float(
                torch.cuda.get_device_properties(device).total_memory / (1024**3)
            ),
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "candidate_control_runtime_ratio": runtime_ratio,
        },
        "export": {
            "gabor": gabor_export,
            "active_sum": active_export,
        },
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "artifacts": [
            {
                "path": str(summary_path.resolve()),
                "sha256": _sha256(summary_path),
                "role": "stage_a_summary",
            },
            {
                "path": gabor_export["path"],
                "sha256": gabor_export["sha256"],
                "role": "materialized_gabor_onnx",
            },
            {
                "path": gabor_export["intermediate_path"],
                "sha256": gabor_export["intermediate_sha256"],
                "role": "gabor_intermediate_onnx",
            },
            {
                "path": active_export["path"],
                "sha256": active_export["sha256"],
                "role": "active_semantic_sum_onnx",
            },
        ],
        "raw_dataset_modified": False,
        "validation_used": False,
        "test_used": False,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not bool(summary["gate"]["smoke_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
