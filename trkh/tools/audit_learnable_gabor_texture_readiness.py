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
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    resolve_amp_dtype,
    set_seed,
)
from trkh.inference.inference import load_checkpoint
from trkh.models.learnable_gabor_texture import (
    LearnableGaborTextureEncoder,
    LearnableGaborTextureResidual,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_moga_tokenizer_readiness import _move_batch, _tensor_summary
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _load_json_mapping,
    _prepare_output_dir,
    _sha256,
    _state_sha256,
)


EXPECTED_TRAIN_ROWS = 9215
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_SCRATCH_SHA256 = "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
LOCKED_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
MAX_ADDED_PARAMETERS = 100_000
MAX_PEAK_VRAM_GIB = 7.75
MAX_RUNTIME_RATIO = 1.50
BENCHMARK_REPEATS = 3
MIN_MECHANISM_DELTA = 1e-6
MIN_HISTOGRAM_ENTROPY = 0.25
MIN_ILLUMINATION_MEAN_COSINE = 0.90
MIN_ILLUMINATION_CLASS_COSINE = 0.85
MAX_MATERIALIZED_ERROR = 1e-6
MAX_ONNX_ERROR = 1e-5


GABOR_GRADIENT_FAMILIES = {
    "gate": ("raw_gate",),
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


def _rng_state_sha256(state: Tensor) -> str:
    payload = state.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only readiness audit for the locked compact learnable-Gabor "
            "LHO/FCM residual. Validation and test access are forbidden."
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
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--scratch-reference",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt"
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
            "docs/TRKH_5CLASS_LEARNABLE_GABOR_TEXTURE_READINESS_PROTOCOL_20260715.md"
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
            "pretrained": False,
        }
    )
    return config


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _control_config(source)
    config["learnable_gabor_texture_residual"] = True
    return config


def _bbox_from_metadata(metadata: Mapping[str, Tensor]) -> Optional[Tensor]:
    crop_bbox = metadata.get("crop_bbox")
    if torch.is_tensor(crop_bbox):
        return crop_bbox
    bbox = metadata.get("bbox")
    return bbox if torch.is_tensor(bbox) else None


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    return_trace: bool = False,
) -> Tuple[Tensor, Mapping[str, object]]:
    features = model.forward_features(
        images,
        image_valid_mask=metadata.get("image_mask"),
        bbox_token_prior=_bbox_from_metadata(metadata),
        return_trace=return_trace,
    )
    return classification_logits_from_features(model, features), features


def summarize_gabor_gradients(
    module: LearnableGaborTextureResidual,
) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(module.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in GABOR_GRADIENT_FAMILIES.items():
        selected = {
            name: parameter
            for name, parameter in named_parameters.items()
            if any(name == marker or name.startswith(marker) for marker in markers)
        }
        present = {
            name: parameter.grad
            for name, parameter in selected.items()
            if parameter.grad is not None
        }
        finite = bool(present) and all(
            bool(torch.isfinite(gradient).all()) for gradient in present.values()
        )
        live_tensors = sum(
            int(torch.count_nonzero(gradient.detach()).item()) > 0
            for gradient in present.values()
        )
        nonzero = sum(
            int(torch.count_nonzero(gradient.detach()).item())
            for gradient in present.values()
        )
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "live_gradient_tensors": int(live_tensors),
            "finite": finite,
            "nonzero_elements": int(nonzero),
            "passed": bool(
                selected
                and len(present) == len(selected)
                and finite
                and live_tensors > 0
                and nonzero > 0
            ),
        }
    return summary


def assess_readiness(
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
            "minimum_mechanism_delta": MIN_MECHANISM_DELTA,
            "minimum_histogram_entropy": MIN_HISTOGRAM_ENTROPY,
            "minimum_illumination_mean_cosine": MIN_ILLUMINATION_MEAN_COSINE,
            "minimum_illumination_class_cosine": MIN_ILLUMINATION_CLASS_COSINE,
            "maximum_materialized_error": MAX_MATERIALIZED_ERROR,
            "maximum_onnx_error": MAX_ONNX_ERROR,
        },
    }


def _benchmark(
    *,
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    repeats: int,
    device: torch.device,
) -> Dict[str, object]:
    model.train()
    amp_dtype = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype != torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def iteration() -> Tuple[Tensor, Tensor]:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _ = _forward_logits(model, images, metadata)
            loss = F.cross_entropy(logits.float(), labels)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            scaler.update()
        return logits, loss

    iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, loss = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - start))
    return {
        "batch_size": int(images.size(0)),
        "repeats": int(repeats),
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "minimum_seconds": float(min(elapsed)),
        "maximum_seconds": float(max(elapsed)),
        "loss": float(loss.detach().item()),
        "logits_shape": [int(value) for value in logits.shape],
        "logits_loss_finite": bool(
            torch.isfinite(logits).all() and torch.isfinite(loss)
        ),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "grad_scaler_enabled": bool(scaler.is_enabled()),
    }


def _module_diagnostics(
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    bbox: Tensor,
) -> Dict[str, object]:
    module.eval()
    full_bbox = images.new_tensor([0.5, 0.5, 1.0, 1.0]).view(1, 4).expand(
        int(images.size(0)),
        -1,
    )
    with torch.no_grad():
        full = module.forward_components(images, bbox)
        low = module.forward_components(images, bbox, use_high=False)
        high = module.forward_components(images, bbox, use_low=False)
        no_position = module.forward_components(
            images,
            bbox,
            use_lho_position=False,
        )
        no_parameter = module.forward_components(
            images,
            bbox,
            use_filter_parameter_encoding=False,
        )
        full_frame = module.forward_components(images, full_bbox)
        materialized = module.forward_components(
            images,
            bbox,
            kernel_override=module.materialized_kernels(),
        )

    real_kernel = full["real_kernel"]
    imaginary_kernel = full["imaginary_kernel"]
    kernel_vectors = torch.cat(
        (real_kernel.flatten(1), imaginary_kernel.flatten(1)),
        dim=1,
    )
    kernel_vectors = F.normalize(kernel_vectors, dim=1)
    similarity = kernel_vectors @ kernel_vectors.transpose(0, 1)
    similarity = similarity.abs() - torch.eye(
        int(similarity.size(0)),
        device=similarity.device,
    )
    counts = full["counts"]
    entropy = -(counts * counts.clamp_min(1e-8).log()).sum(dim=-1)
    foreground_mass = (
        full["unmasked_magnitude"] * full["response_mask"]
    ).sum(dim=(-2, -1)) / full["unmasked_magnitude"].sum(
        dim=(-2, -1)
    ).clamp_min(1e-6)
    mask_area = full["response_mask"].mean(dim=(-2, -1))
    materialized_error = float(
        (full["texture_token"] - materialized["texture_token"]).abs().max().item()
    )
    ablation_deltas = {
        "low_only": float(
            (full["texture_token"] - low["texture_token"]).abs().mean().item()
        ),
        "high_only": float(
            (full["texture_token"] - high["texture_token"]).abs().mean().item()
        ),
        "without_lho_position": float(
            (full["texture_token"] - no_position["texture_token"]).abs().mean().item()
        ),
        "without_filter_parameter_encoding": float(
            (full["texture_token"] - no_parameter["texture_token"]).abs().mean().item()
        ),
    }
    return {
        "filter_count": int(module.filter_count),
        "low_filter_count": int(module.low_filter_count),
        "high_filter_count": int(module.high_filter_count),
        "histogram_levels": int(module.histogram_levels),
        "analysis_size": int(module.analysis_size),
        "response_size": int(module.response_size),
        "kernel_size": int(module.kernel_size),
        "hidden_dim": int(module.hidden_dim),
        "attention_heads": int(module.attention_heads),
        "gate_scale": (
            float(module.gate_scale)
            if hasattr(module, "gate_scale")
            else None
        ),
        "effective_gate": (
            float(module.effective_gate().detach().item())
            if hasattr(module, "effective_gate")
            else None
        ),
        "parameters": {
            "theta": _tensor_summary(full["theta"]),
            "sigma_x": _tensor_summary(full["sigma_x"]),
            "sigma_y": _tensor_summary(full["sigma_y"]),
            "frequency": _tensor_summary(full["frequency"]),
        },
        "kernels": {
            "real": _tensor_summary(real_kernel),
            "imaginary": _tensor_summary(imaginary_kernel),
            "maximum_absolute_pair_cosine": float(similarity.max().item()),
            "maximum_absolute_mean": float(
                max(
                    real_kernel.mean(dim=(-2, -1)).abs().max().item(),
                    imaginary_kernel.mean(dim=(-2, -1)).abs().max().item(),
                )
            ),
            "maximum_l2_error": float(
                max(
                    (real_kernel.flatten(1).norm(dim=1) - 1.0).abs().max().item(),
                    (imaginary_kernel.flatten(1).norm(dim=1) - 1.0).abs().max().item(),
                )
            ),
        },
        "responses": {
            "real": _tensor_summary(full["real_response"]),
            "imaginary": _tensor_summary(full["imaginary_response"]),
            "magnitude": _tensor_summary(full["magnitude"]),
            "mean_foreground_mass": float(foreground_mass.mean().item()),
            "minimum_foreground_mass": float(foreground_mass.min().item()),
            "mean_mask_area_fraction": float(mask_area.mean().item()),
            "foreground_mass_minus_area": float(
                (foreground_mass - mask_area).mean().item()
            ),
        },
        "histogram": {
            "maximum_count_sum_error": float(
                (counts.sum(dim=-1) - 1.0).abs().max().item()
            ),
            "minimum_entropy": float(entropy.min().item()),
            "mean_entropy": float(entropy.mean().item()),
            "counts": _tensor_summary(counts),
        },
        "texture_token": _tensor_summary(full["texture_token"]),
        "ablation_deltas": ablation_deltas,
        "bbox_full_frame_descriptor_delta": float(
            (full["texture_token"] - full_frame["texture_token"]).abs().mean().item()
        ),
        "materialized_kernel_max_error": materialized_error,
    }


def _illumination_diagnostics(
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    bbox: Tensor,
    labels: Tensor,
) -> Dict[str, object]:
    mean = module.rgb_mean.to(device=images.device)
    std = module.rgb_std.to(device=images.device)
    rgb = (images.float() * std + mean).clamp(0.0, 1.0)
    image_mean = rgb.mean(dim=(-2, -1), keepdim=True)
    variants = {
        "dim": rgb * 0.75,
        "bright": (rgb * 1.15 + 0.05).clamp(0.0, 1.0),
        "low_contrast": ((rgb - image_mean) * 0.70 + image_mean).clamp(0.0, 1.0),
    }
    module.eval()
    with torch.no_grad():
        clean = module.forward_components(images, bbox)["texture_token"]
        conditions = {}
        all_values = []
        for name, rgb_variant in variants.items():
            normalized = (rgb_variant - mean) / std
            token = module.forward_components(normalized, bbox)["texture_token"]
            cosine = F.cosine_similarity(clean, token, dim=1)
            values = [float(value) for value in cosine.detach().cpu().tolist()]
            all_values.extend(values)
            conditions[name] = {
                "mean_cosine": float(cosine.mean().item()),
                "minimum_cosine": float(cosine.min().item()),
                "per_class": [
                    {
                        "class_index": int(label),
                        "cosine": float(value),
                    }
                    for label, value in zip(labels.detach().cpu().tolist(), values)
                ],
            }
    return {
        "transforms": {
            "dim_factor": 0.75,
            "bright_scale": 1.15,
            "bright_offset": 0.05,
            "low_contrast_factor": 0.70,
        },
        "conditions": conditions,
        "mean_cosine": float(sum(all_values) / max(1, len(all_values))),
        "minimum_class_condition_cosine": float(min(all_values)),
    }


def _balanced_sensitivity(
    *,
    model: nn.Module,
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
) -> Dict[str, object]:
    bbox = _bbox_from_metadata(metadata)
    if bbox is None:
        raise RuntimeError("Balanced train batch lacks bbox metadata")
    module.zero_grad(set_to_none=True)
    descriptor_probe = images.detach().clone().requires_grad_(True)
    descriptor = module.forward_components(descriptor_probe, bbox)["texture_token"]
    descriptor_gradient = torch.autograd.grad(
        descriptor.square().mean(),
        descriptor_probe,
        retain_graph=False,
    )[0]

    model.eval()
    model.zero_grad(set_to_none=True)
    logit_probe = images.detach().clone().requires_grad_(True)
    logits, _ = _forward_logits(model, logit_probe, metadata)
    selected = logits.gather(1, labels.view(-1, 1)).sum()
    logit_gradient = torch.autograd.grad(selected, logit_probe, retain_graph=False)[0]
    rows = []
    for row_index, label in enumerate(labels.detach().cpu().tolist()):
        descriptor_row = descriptor_gradient[row_index].detach().float()
        logit_row = logit_gradient[row_index].detach().float()
        rows.append(
            {
                "class_index": int(label),
                "descriptor_norm": float(descriptor[row_index].detach().float().norm().item()),
                "descriptor_gradient_finite": bool(torch.isfinite(descriptor_row).all()),
                "descriptor_gradient_nonzero": int(torch.count_nonzero(descriptor_row).item()),
                "true_logit_gradient_finite": bool(torch.isfinite(logit_row).all()),
                "true_logit_gradient_nonzero": int(torch.count_nonzero(logit_row).item()),
            }
        )
    return {
        "labels": [int(value) for value in labels.detach().cpu().tolist()],
        "rows": rows,
        "all_classes_live": bool(
            len(rows) == 5
            and {int(row["class_index"]) for row in rows} == set(range(5))
            and all(
                float(row["descriptor_norm"]) > 0.0
                and bool(row["descriptor_gradient_finite"])
                and int(row["descriptor_gradient_nonzero"]) > 0
                and bool(row["true_logit_gradient_finite"])
                and int(row["true_logit_gradient_nonzero"]) > 0
                for row in rows
            )
        ),
    }


class _MaterializedGaborExport(nn.Module):
    def __init__(self, module: LearnableGaborTextureEncoder) -> None:
        super().__init__()
        # Materialize on the deployment backend. CPU/GPU trigonometric kernels
        # can differ by a few ulps, which the triangular LHO bins may amplify.
        self.module = copy.deepcopy(module).cpu().eval()
        real_kernel, imaginary_kernel = self.module.materialized_kernels()
        self.register_buffer("real_kernel", real_kernel)
        self.register_buffer("imaginary_kernel", imaginary_kernel)

    def forward(self, image: Tensor, bbox: Tensor) -> Tensor:
        return self.module.forward_components(
            image,
            bbox,
            kernel_override=(self.real_kernel, self.imaginary_kernel),
        )["texture_token"]


class _MaterializedGaborIntermediateExport(nn.Module):
    component_names = (
        "texture_input",
        "mask",
        "real_response",
        "imaginary_response",
        "unmasked_magnitude",
        "magnitude",
        "response_mask",
        "levels",
        "assignment",
        "counts",
        "normalized_levels",
        "level_embeddings",
        "position_descriptor",
        "lho_features",
        "filter_features",
        "fcm_output",
        "texture_descriptor",
        "texture_token",
    )

    def __init__(self, wrapper: _MaterializedGaborExport) -> None:
        super().__init__()
        self.wrapper = wrapper

    def forward(self, image: Tensor, bbox: Tensor):
        components = self.wrapper.module.forward_components(
            image,
            bbox,
            kernel_override=(self.wrapper.real_kernel, self.wrapper.imaginary_kernel),
        )
        return tuple(components[name] for name in self.component_names)


def _export_diagnostics(
    *,
    module: LearnableGaborTextureEncoder,
    images: Tensor,
    bbox: Tensor,
    output_dir: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    wrapper = _MaterializedGaborExport(module).cpu().eval()
    cpu_images = images[:2].detach().float().cpu()
    cpu_bbox = bbox[:2].detach().float().cpu()
    onnx_path = output_dir / "learnable_gabor_texture_materialized.onnx"
    intermediate_path = output_dir / "learnable_gabor_texture_intermediates.onnx"
    with torch.no_grad():
        expected = wrapper(cpu_images, cpu_bbox)
        dynamic = module.cpu().eval().forward_components(cpu_images, cpu_bbox)[
            "texture_token"
        ]
    dynamic_materialized_error = float((dynamic - expected).abs().max().item())
    torch.onnx.export(
        wrapper,
        (cpu_images, cpu_bbox),
        str(onnx_path),
        input_names=["image", "bbox"],
        output_names=["texture_token"],
        dynamic_axes={
            "image": {0: "batch"},
            "bbox": {0: "batch"},
            "texture_token": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    batch_results = []
    maximum_error = 0.0
    for batch_size in (1, 2):
        image_value = cpu_images[:batch_size]
        bbox_value = cpu_bbox[:batch_size]
        with torch.no_grad():
            reference = wrapper(image_value, bbox_value).cpu().numpy()
        observed = session.run(
            ["texture_token"],
            {
                "image": image_value.numpy(),
                "bbox": bbox_value.numpy(),
            },
        )[0]
        error = float(np.max(np.abs(reference - observed)))
        maximum_error = max(maximum_error, error)
        batch_results.append(
            {
                "batch_size": batch_size,
                "maximum_absolute_error": error,
                "finite": bool(np.isfinite(observed).all()),
                "shape": [int(value) for value in observed.shape],
            }
        )

    intermediate_wrapper = _MaterializedGaborIntermediateExport(wrapper).cpu().eval()
    intermediate_names = list(intermediate_wrapper.component_names)
    intermediate_dynamic_axes = {
        "image": {0: "batch"},
        "bbox": {0: "batch"},
    }
    intermediate_dynamic_axes.update(
        {name: {0: "batch"} for name in intermediate_names}
    )
    torch.onnx.export(
        intermediate_wrapper,
        (cpu_images, cpu_bbox),
        str(intermediate_path),
        input_names=["image", "bbox"],
        output_names=intermediate_names,
        dynamic_axes=intermediate_dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    intermediate_model = onnx.load(str(intermediate_path))
    onnx.checker.check_model(intermediate_model)
    intermediate_session = ort.InferenceSession(
        str(intermediate_path),
        providers=["CPUExecutionProvider"],
    )
    intermediate_results = []
    for batch_size in (1, 2):
        image_value = cpu_images[:batch_size]
        bbox_value = cpu_bbox[:batch_size]
        with torch.no_grad():
            references = intermediate_wrapper(image_value, bbox_value)
        observations = intermediate_session.run(
            intermediate_names,
            {
                "image": image_value.numpy(),
                "bbox": bbox_value.numpy(),
            },
        )
        component_results = []
        for name, reference, observed in zip(
            intermediate_names,
            references,
            observations,
        ):
            reference_array = reference.detach().cpu().numpy()
            absolute_error = np.abs(reference_array - observed)
            component_results.append(
                {
                    "name": name,
                    "maximum_absolute_error": float(absolute_error.max()),
                    "mean_absolute_error": float(absolute_error.mean()),
                    "reference_maximum_absolute_value": float(
                        np.abs(reference_array).max()
                    ),
                    "finite": bool(np.isfinite(observed).all()),
                    "shape": [int(value) for value in observed.shape],
                }
            )
        intermediate_results.append(
            {
                "batch_size": batch_size,
                "components": component_results,
            }
        )
    return {
        "path": str(onnx_path.resolve()),
        "sha256": _sha256(onnx_path),
        "opset": 17,
        "dynamic_materialized_max_error": dynamic_materialized_error,
        "onnx_max_error": maximum_error,
        "batch_results": batch_results,
        "intermediate_path": str(intermediate_path.resolve()),
        "intermediate_sha256": _sha256(intermediate_path),
        "intermediate_results": intermediate_results,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    scratch_path = Path(args.scratch_reference).resolve()
    command_path = Path(args.command_file).resolve()
    protocol_path = Path(args.protocol).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Learnable-Gabor Stage A requires CUDA")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked Stage-A batch sizes are AMP=32 and FP32=2")
    if int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError(f"Locked benchmark repeats are {BENCHMARK_REPEATS}")

    source_hashes = {
        "data_yaml": _sha256(data_path),
        "keeper_checkpoint": _sha256(checkpoint_path),
        "scratch_reference": _sha256(scratch_path),
        "current_command": _sha256(command_path),
        "protocol": _sha256(protocol_path),
        "resolved_config": _sha256(resolved_config_path),
    }
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    state_dict = checkpoint.get("model_state")
    if (
        not isinstance(source_model_config, Mapping)
        or not isinstance(class_names, list)
        or not isinstance(state_dict, Mapping)
    ):
        raise ValueError("Keeper checkpoint lacks model_config/class_names/model_state")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}")
    resolved_config = _load_json_mapping(resolved_config_path)
    control_config = _control_config(source_model_config)
    candidate_config = _candidate_config(source_model_config)

    implicit_control_config = dict(source_model_config)
    implicit_control_config.pop("learnable_gabor_texture_residual", None)
    set_seed(int(args.seed))
    implicit_control = create_model(len(class_names), implicit_control_config)
    implicit_post_constructor_rng = torch.get_rng_state().clone()
    load_model_state(implicit_control, dict(state_dict), strict=True)
    implicit_names = list(implicit_control.state_dict())
    implicit_state_sha = _state_sha256(implicit_control)
    set_seed(int(args.seed))
    control_model = create_model(len(class_names), control_config)
    control_post_constructor_rng = torch.get_rng_state().clone()
    load_model_state(control_model, dict(state_dict), strict=True)
    control_names = list(control_model.state_dict())
    control_state_sha = _state_sha256(control_model)
    control_parameter_count = int(
        sum(parameter.numel() for parameter in control_model.parameters())
    )

    set_seed(int(args.seed))
    candidate_model = create_model(len(class_names), candidate_config)
    candidate_post_constructor_rng = torch.get_rng_state().clone()
    missing_keys, unexpected_keys = load_model_state(
        candidate_model,
        dict(state_dict),
        strict=False,
    )
    candidate_initial_state_sha = _state_sha256(candidate_model)
    candidate_parameter_count = int(
        sum(parameter.numel() for parameter in candidate_model.parameters())
    )
    set_seed(int(args.seed))
    same_seed_candidate = create_model(len(class_names), candidate_config)
    same_seed_candidate_post_constructor_rng = torch.get_rng_state().clone()
    same_missing, same_unexpected = load_model_state(
        same_seed_candidate,
        dict(state_dict),
        strict=False,
    )
    same_seed_state_sha = _state_sha256(same_seed_candidate)
    del same_seed_candidate

    roundtrip_checkpoint = {
        "class_names": list(class_names),
        "model_config": dict(candidate_config),
        "model_state": candidate_model.state_dict(),
    }
    roundtrip_model = build_model_from_checkpoint(roundtrip_checkpoint)
    roundtrip_state_sha = _state_sha256(roundtrip_model)
    del roundtrip_model

    dataset = _build_train_only_dataset(
        data_yaml=data_path,
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    dataset.classification_bbox_metadata = True
    data_spec = load_data_spec(data_path)
    train_root = Path(data_spec.split_images_dir("train")).resolve()
    balanced_indices: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        label = int(sample.primary_label)
        if label not in balanced_indices:
            balanced_indices[label] = int(index)
        if len(balanced_indices) == len(class_names):
            break
    if set(balanced_indices) != set(range(len(class_names))):
        raise RuntimeError("Could not construct one real train sample per class")
    ordered_balanced_indices = [balanced_indices[index] for index in range(len(class_names))]
    balanced_batch = default_collate([dataset[index] for index in ordered_balanced_indices])
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="learnable_gabor_stage_a_train_only",
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
        raise RuntimeError("Stage-A dataset did not emit transformed bbox metadata")

    implicit_control = implicit_control.to(device).eval()
    control_model = control_model.to(device).eval()
    candidate_model = candidate_model.to(device).eval()
    with torch.no_grad():
        implicit_logits, _ = _forward_logits(
            implicit_control,
            balanced_images,
            balanced_metadata,
        )
        control_logits, _ = _forward_logits(
            control_model,
            balanced_images,
            balanced_metadata,
        )
        candidate_logits, _ = _forward_logits(
            candidate_model,
            balanced_images,
            balanced_metadata,
        )
    default_control_max_error = float(
        (implicit_logits - control_logits).abs().max().item()
    )
    zero_gate_max_error = float((control_logits - candidate_logits).abs().max().item())
    default_control_exact = torch.equal(implicit_logits, control_logits)
    zero_gate_exact = torch.equal(control_logits, candidate_logits)
    del implicit_control
    gc.collect()
    torch.cuda.empty_cache()

    module = getattr(candidate_model, "gabor_texture_residual", None)
    if not isinstance(module, LearnableGaborTextureResidual):
        raise TypeError("Candidate did not construct LearnableGaborTextureResidual")
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

    module.raw_gate.data.zero_()
    candidate_model.eval()
    candidate_model.zero_grad(set_to_none=True)
    zero_logits, _ = _forward_logits(
        candidate_model,
        images[: int(args.fp32_batch_size)],
        {key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()},
    )
    zero_loss = F.cross_entropy(zero_logits.float(), labels[: int(args.fp32_batch_size)])
    zero_loss.backward()
    zero_gate_gradient = module.raw_gate.grad
    zero_gate_gradient_summary = {
        "present": zero_gate_gradient is not None,
        "finite": bool(
            zero_gate_gradient is not None and torch.isfinite(zero_gate_gradient)
        ),
        "absolute_value": float(
            zero_gate_gradient.detach().abs().item()
            if zero_gate_gradient is not None
            else 0.0
        ),
    }

    candidate_model.zero_grad(set_to_none=True)
    response_probe = balanced_images[:2].detach().clone().requires_grad_(True)
    response_components = module.forward_components(response_probe, bbox[:2])
    response_components["real_response"].retain_grad()
    response_components["imaginary_response"].retain_grad()
    response_components["texture_token"].square().mean().backward()
    response_gradient_summary = {}
    for name in ("real_response", "imaginary_response"):
        gradient = response_components[name].grad
        response_gradient_summary[name] = {
            "present": gradient is not None,
            "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
            "nonzero_elements": int(
                torch.count_nonzero(gradient.detach()).item()
                if gradient is not None
                else 0
            ),
        }

    candidate_model.zero_grad(set_to_none=True)
    module.raw_gate.data.fill_(torch.atanh(torch.tensor(0.2, device=device)))
    candidate_model.train()
    fp32_logits, _ = _forward_logits(
        candidate_model,
        images[: int(args.fp32_batch_size)],
        {key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()},
    )
    fp32_loss = F.cross_entropy(fp32_logits.float(), labels[: int(args.fp32_batch_size)])
    fp32_loss.backward()
    fp32_gradients = summarize_gabor_gradients(module)
    fp32_finite = bool(torch.isfinite(fp32_logits).all() and torch.isfinite(fp32_loss))

    sensitivity = _balanced_sensitivity(
        model=candidate_model,
        module=module,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
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
    bf16_gradients = summarize_gabor_gradients(module)
    runtime_ratio = float(
        float(candidate_benchmark["median_seconds"])
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )

    export = _export_diagnostics(
        module=module,
        images=balanced_images,
        bbox=bbox,
        output_dir=output_dir,
    )

    paths = [
        str(dataset.samples[index].image_path.resolve())
        for index in ordered_balanced_indices
    ]
    train_only_paths = all(Path(path).is_relative_to(train_root) for path in paths)
    missing_allowed = bool(missing_keys) and all(
        str(key).startswith("gabor_texture_residual.") for key in missing_keys
    )
    same_missing_allowed = list(missing_keys) == list(same_missing)
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
    mechanism_deltas = module_diagnostics["ablation_deltas"]
    checks = {
        "locked_data_sha256": source_hashes["data_yaml"] == LOCKED_DATA_SHA256,
        "locked_keeper_sha256": source_hashes["keeper_checkpoint"] == LOCKED_KEEPER_SHA256,
        "locked_scratch_reference_sha256": source_hashes["scratch_reference"] == LOCKED_SCRATCH_SHA256,
        "locked_current_command_sha256": source_hashes["current_command"] == LOCKED_COMMAND_SHA256,
        "candidate_pretrained_disabled": not bool(candidate_config.get("pretrained", False)),
        "train_split_only": bool(train_only_paths),
        "train_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "balanced_real_train_batch": (
            [int(value) for value in balanced_labels.detach().cpu().tolist()]
            == list(range(5))
        ),
        "transformed_bbox_metadata_present": bool(
            torch.is_tensor(balanced_metadata.get("crop_bbox"))
        ),
        "default_off_state_schema": implicit_names == control_names,
        "default_off_state_identical": implicit_state_sha == control_state_sha,
        "default_off_post_constructor_rng_exact": torch.equal(
            implicit_post_constructor_rng,
            control_post_constructor_rng,
        ),
        "default_off_logits_exact": default_control_exact and default_control_max_error == 0.0,
        "candidate_missing_only_gabor": missing_allowed and not unexpected_keys,
        "candidate_same_seed_missing_identical": same_missing_allowed and not same_unexpected,
        "candidate_same_seed_deterministic": candidate_initial_state_sha == same_seed_state_sha,
        "candidate_same_seed_post_constructor_rng_exact": torch.equal(
            candidate_post_constructor_rng,
            same_seed_candidate_post_constructor_rng,
        ),
        "candidate_post_constructor_rng_neutral": torch.equal(
            control_post_constructor_rng,
            candidate_post_constructor_rng,
        ),
        "candidate_checkpoint_roundtrip": candidate_initial_state_sha == roundtrip_state_sha,
        "zero_gate_exact_keeper_logits": zero_gate_exact and zero_gate_max_error <= 1e-7,
        "zero_gate_first_gradient_live": bool(
            zero_gate_gradient_summary["finite"]
            and zero_gate_gradient_summary["absolute_value"] > 0.0
        ),
        "locked_module_dimensions": bool(
            module.filter_count == 32
            and module.histogram_levels == 8
            and module.analysis_size == 64
            and module.response_size == 16
            and module.kernel_size == 11
            and module.hidden_dim == 64
            and module.attention_heads == 4
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
        "kernels_diverse": bool(
            module_diagnostics["kernels"]["maximum_absolute_pair_cosine"] < 0.9999
        ),
        "real_imaginary_responses_live": bool(
            module_diagnostics["responses"]["real"]["finite"]
            and module_diagnostics["responses"]["imaginary"]["finite"]
            and module_diagnostics["responses"]["real"]["rms"] > 1e-4
            and module_diagnostics["responses"]["imaginary"]["rms"] > 1e-4
            and module_diagnostics["responses"]["magnitude"]["std"] > 1e-4
        ),
        "real_imaginary_response_gradients": all(
            bool(value["finite"]) and int(value["nonzero_elements"]) > 0
            for value in response_gradient_summary.values()
        ),
        "lho_counts_normalized": (
            module_diagnostics["histogram"]["maximum_count_sum_error"] <= 1e-5
        ),
        "lho_entropy_noncollapsed": (
            module_diagnostics["histogram"]["minimum_entropy"]
            >= MIN_HISTOGRAM_ENTROPY
        ),
        "mechanism_ablations_material": all(
            math.isfinite(float(value)) and float(value) > MIN_MECHANISM_DELTA
            for value in mechanism_deltas.values()
        ),
        "bbox_response_concentration": (
            module_diagnostics["responses"]["foreground_mass_minus_area"] > 0.0
        ),
        "bbox_descriptor_sensitivity": (
            module_diagnostics["bbox_full_frame_descriptor_delta"] > MIN_MECHANISM_DELTA
        ),
        "materialized_kernel_equivalence": (
            module_diagnostics["materialized_kernel_max_error"] <= MAX_MATERIALIZED_ERROR
            and export["dynamic_materialized_max_error"] <= MAX_MATERIALIZED_ERROR
        ),
        "illumination_mean_cosine": (
            illumination["mean_cosine"] >= MIN_ILLUMINATION_MEAN_COSINE
        ),
        "illumination_per_class_cosine": (
            illumination["minimum_class_condition_cosine"]
            >= MIN_ILLUMINATION_CLASS_COSINE
        ),
        "balanced_five_class_sensitivity": bool(sensitivity["all_classes_live"]),
        "fp32_logits_loss_finite": fp32_finite,
        "fp32_all_gradient_families": all(
            bool(value["passed"]) for value in fp32_gradients.values()
        ),
        "bf16_control_logits_loss_finite": bool(control_benchmark["logits_loss_finite"]),
        "bf16_candidate_logits_loss_finite": bool(candidate_benchmark["logits_loss_finite"]),
        "bf16_all_gradient_families": all(
            bool(value["passed"]) for value in bf16_gradients.values()
        ),
        "onnx_cpu_equivalence": bool(
            export["onnx_max_error"] <= MAX_ONNX_ERROR
            and all(bool(value["finite"]) for value in export["batch_results"])
        ),
        "onnx_dynamic_batch_1_2": [
            int(value["batch_size"]) for value in export["batch_results"]
        ]
        == [1, 2],
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        runtime_ratio=runtime_ratio,
    )
    summary: Dict[str, object] = {
        "method": "learnable_gabor_lho_fcm_edge_token_residual",
        "protocol_stage": "A_train_only_functional_resource_export_preflight",
        "sources": {
            "checkpoint": str(checkpoint_path),
            "resolved_config": str(resolved_config_path),
            "scratch_reference": str(scratch_path),
            "current_command": str(command_path),
            "protocol": str(protocol_path),
            "data_yaml": str(data_path),
            "hashes": source_hashes,
            "split_loaded": "train",
            "validation_loaded": False,
            "test_loaded": False,
            "raw_dataset_modified": False,
        },
        "model": {
            "class_names": [str(value) for value in class_names],
            "control_parameters": control_parameter_count,
            "candidate_parameters": candidate_parameter_count,
            "candidate_added_parameters": added_parameters,
            "gabor_module_parameters": module_parameters,
            "implicit_control_state_sha256": implicit_state_sha,
            "explicit_control_state_sha256": control_state_sha,
            "candidate_initial_state_sha256": candidate_initial_state_sha,
            "candidate_same_seed_state_sha256": same_seed_state_sha,
            "candidate_roundtrip_state_sha256": roundtrip_state_sha,
            "post_constructor_rng": {
                "implicit_control_sha256": _rng_state_sha256(
                    implicit_post_constructor_rng
                ),
                "explicit_control_sha256": _rng_state_sha256(
                    control_post_constructor_rng
                ),
                "candidate_sha256": _rng_state_sha256(
                    candidate_post_constructor_rng
                ),
                "same_seed_candidate_sha256": _rng_state_sha256(
                    same_seed_candidate_post_constructor_rng
                ),
                "control_candidate_differing_bytes": int(
                    torch.count_nonzero(
                        control_post_constructor_rng
                        != candidate_post_constructor_rng
                    ).item()
                ),
            },
            "missing_keys": [str(value) for value in missing_keys],
            "unexpected_keys": [str(value) for value in unexpected_keys],
            "default_control_max_logit_error": default_control_max_error,
            "zero_gate_max_logit_error": zero_gate_max_error,
        },
        "data": {
            "train_rows": len(dataset),
            "train_root": str(train_root),
            "balanced_indices": ordered_balanced_indices,
            "balanced_labels": [
                int(value) for value in balanced_labels.detach().cpu().tolist()
            ],
            "balanced_paths": paths,
            "resource_batch_size": int(images.size(0)),
            "resource_labels": [int(value) for value in labels.detach().cpu().tolist()],
            "bbox_source": "crop_bbox",
            "dataloader": loader_summary,
        },
        "module_diagnostics": module_diagnostics,
        "illumination": illumination,
        "balanced_sensitivity": sensitivity,
        "gradients": {
            "zero_gate_first_step": zero_gate_gradient_summary,
            "response_intermediates": response_gradient_summary,
            "fp32": fp32_gradients,
            "bf16": bf16_gradients,
        },
        "fp32": {
            "batch_size": int(args.fp32_batch_size),
            "loss": float(fp32_loss.detach().item()),
            "logits_shape": [int(value) for value in fp32_logits.shape],
        },
        "cuda_bf16": {
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_gib": float(
                torch.cuda.get_device_properties(device).total_memory / (1024**3)
            ),
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "candidate_control_runtime_ratio": runtime_ratio,
        },
        "export": export,
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
                "path": export["path"],
                "sha256": export["sha256"],
                "role": "materialized_gabor_onnx",
            },
            {
                "path": export["intermediate_path"],
                "sha256": export["intermediate_sha256"],
                "role": "gabor_onnx_intermediate_trace",
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
