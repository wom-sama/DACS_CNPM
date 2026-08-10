from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    resolve_amp_dtype,
    set_seed,
)
from trkh.inference.inference import load_checkpoint
from trkh.models.moga_surface_tokenizer import MogaXTTokenizer
from trkh.models.model import create_model
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _forward_logits,
    _load_json_mapping,
    _prepare_output_dir,
    _sha256,
    _state_sha256,
    _unpack_batch,
)


LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_SCRATCH_CHECKPOINT_SHA256 = (
    "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
)
LOCKED_SOURCE_LAUNCHER_SHA256 = (
    "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
)
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_STAGE_DEPTHS = (3, 3, 10)
EXPECTED_BLOCKS = 16
EXPECTED_PATCH_TOKENS = 256
MAX_CANDIDATE_PARAMETERS = 10_000_000
MAX_ADDED_PARAMETERS = 3_000_000
MAX_PEAK_VRAM_GIB = 7.75
MAX_RUNTIME_RATIO = 1.75
BENCHMARK_REPEATS = 3

MOGA_GRADIENT_MARKERS = {
    "stage_embeddings": (
        "stem.patch_embed1.",
        "stem.patch_embed2.",
        "stem.patch_embed3.",
    ),
    "low_order": (".spatial.value.dw_conv_low.",),
    "middle_order": (".spatial.value.dw_conv_middle.",),
    "high_order": (".spatial.value.dw_conv_high.",),
    "gate": (".spatial.gate.",),
    "spatial_projection": (
        ".spatial.proj_1.",
        ".spatial.value.project.",
        ".spatial.proj_2.",
    ),
    "spatial_decomposition": (".spatial.sigma.scale",),
    "channel_ffn": (
        ".channel.fc1.",
        ".channel.dwconv.",
        ".channel.fc2.",
    ),
    "channel_decomposition": (
        ".channel.decompose.",
        ".channel.sigma.scale",
    ),
    "layer_scales": (".layer_scale_",),
    "patch_projection": ("patch_embed.proj.",),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only functional/resource readiness audit for the scratch "
            "MogaNet-XT surface tokenizer. Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt"
        ),
        help="Scratch checkpoint used only as the locked model-config source.",
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--source-launcher-args",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/launcher_args.json"
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
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    parser.add_argument("--max-runtime-ratio", type=float, default=MAX_RUNTIME_RATIO)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "stem_architecture": "conv_pool",
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "stem_architecture": "moganet_xt_tokenizer",
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def summarize_moga_gradients(model: nn.Module) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in MOGA_GRADIENT_MARKERS.items():
        selected = {
            name: parameter
            for name, parameter in named_parameters.items()
            if any(marker in name for marker in markers)
        }
        gradients = [parameter.grad for parameter in selected.values()]
        present = [gradient for gradient in gradients if gradient is not None]
        finite = bool(present) and all(
            bool(torch.isfinite(gradient).all()) for gradient in present
        )
        nonzero = sum(
            int(torch.count_nonzero(gradient.detach()).item())
            for gradient in present
        )
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "finite": finite,
            "nonzero_elements": int(nonzero),
            "passed": bool(
                selected
                and len(present) == len(selected)
                and finite
                and nonzero > 0
            ),
        }
    return summary


def assess_moga_tokenizer_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    max_peak_vram_gib: float,
    runtime_ratio: float,
    max_runtime_ratio: float,
) -> Dict[str, object]:
    resolved_checks = {str(key): bool(value) for key, value in checks.items()}
    resolved_checks["peak_vram_within_budget"] = bool(
        math.isfinite(float(peak_vram_gib))
        and float(peak_vram_gib) <= float(max_peak_vram_gib)
    )
    resolved_checks["runtime_ratio_within_budget"] = bool(
        math.isfinite(float(runtime_ratio))
        and float(runtime_ratio) <= float(max_runtime_ratio)
    )
    failed = [name for name, passed in resolved_checks.items() if not passed]
    return {
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": resolved_checks,
        "failed_checks": failed,
        "thresholds": {
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
            "expected_stage_depths": list(EXPECTED_STAGE_DEPTHS),
            "expected_blocks": EXPECTED_BLOCKS,
            "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
            "max_candidate_parameters": MAX_CANDIDATE_PARAMETERS,
            "max_added_parameters": MAX_ADDED_PARAMETERS,
            "max_peak_vram_gib": float(max_peak_vram_gib),
            "max_runtime_ratio": float(max_runtime_ratio),
        },
    }


def _tensor_summary(value: Tensor) -> Dict[str, object]:
    detached = value.detach().float()
    return {
        "shape": [int(size) for size in detached.shape],
        "finite": bool(torch.isfinite(detached).all()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "rms": float(detached.square().mean().sqrt().item()),
        "minimum": float(detached.min().item()),
        "maximum": float(detached.max().item()),
    }


def _move_batch(
    batch: object,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
    images, labels, metadata = _unpack_batch(batch)
    return (
        images.to(device=device, non_blocking=True),
        labels.to(device=device, dtype=torch.long, non_blocking=True),
        {
            key: value.to(device=device, non_blocking=True)
            for key, value in metadata.items()
        },
    )


def _amp_forward_backward_benchmark(
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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype != torch.bfloat16,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def run_iteration() -> tuple[Tensor, Tensor]:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _ = _forward_logits(model, images, metadata)
            loss = F.cross_entropy(logits.float(), labels)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            scaler.update()
        return logits, loss

    run_iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    logits: Tensor
    loss: Tensor
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, loss = run_iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    peak_vram_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return {
        "batch_size": int(images.size(0)),
        "repeats": int(repeats),
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "minimum_seconds": float(min(elapsed)),
        "maximum_seconds": float(max(elapsed)),
        "loss": float(loss.detach().item()),
        "logits_shape": [int(size) for size in logits.shape],
        "logits_loss_finite": bool(
            torch.isfinite(logits).all() and torch.isfinite(loss)
        ),
        "peak_vram_gib": peak_vram_gib,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "grad_scaler_enabled": bool(scaler.is_enabled()),
    }


def _activation_diagnostics(
    *,
    model: nn.Module,
    images: Tensor,
) -> Dict[str, object]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, MogaXTTokenizer):
        raise TypeError("Candidate model does not contain MogaXTTokenizer")
    model.eval()
    with torch.inference_mode():
        stem_output, trace = stem.forward_with_trace(images)
        embedded = stem.patch_embed1(images)
        first_block = stem.blocks1[0]
        normalized = first_block.norm1(embedded)
        projected = first_block.spatial.proj_1(normalized)
        global_component = F.adaptive_avg_pool2d(projected, output_size=1)
        decomposed = first_block.spatial.value_act(
            projected
            + first_block.spatial.sigma(projected - global_component)
        )
        branch_features = first_block.spatial.value.branch_features(decomposed)

    stage_summaries = {
        name: _tensor_summary(value)
        for name, value in trace.items()
    }
    branch_summaries = {
        name: _tensor_summary(value)
        for name, value in branch_features.items()
    }
    signatures = {
        (
            round(float(item["mean"]), 8),
            round(float(item["std"]), 8),
            round(float(item["rms"]), 8),
        )
        for item in branch_summaries.values()
    }
    return {
        "stem_output": _tensor_summary(stem_output),
        "stages": stage_summaries,
        "branches": branch_summaries,
        "branches_distinct": len(signatures) == 3,
    }


def _balanced_input_sensitivity(
    *,
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
) -> Dict[str, object]:
    model.eval()
    model.zero_grad(set_to_none=True)
    probe = images.detach().clone().requires_grad_(True)
    logits, _ = _forward_logits(model, probe, metadata)
    selected = logits.gather(1, labels.view(-1, 1)).sum()
    gradients = torch.autograd.grad(selected, probe, retain_graph=False)[0]
    flattened = gradients.detach().float().flatten(1)
    rows = []
    for row_index, label in enumerate(labels.detach().cpu().tolist()):
        row = flattened[row_index]
        rows.append(
            {
                "class_index": int(label),
                "finite": bool(torch.isfinite(row).all()),
                "nonzero_elements": int(torch.count_nonzero(row).item()),
                "mean_absolute_gradient": float(row.abs().mean().item()),
                "maximum_absolute_gradient": float(row.abs().max().item()),
            }
        )
    return {
        "labels": [int(value) for value in labels.detach().cpu().tolist()],
        "logits_shape": [int(size) for size in logits.shape],
        "logits_finite": bool(torch.isfinite(logits).all()),
        "rows": rows,
        "all_classes_live": bool(
            len(rows) == 5
            and {int(row["class_index"]) for row in rows} == set(range(5))
            and all(
                bool(row["finite"]) and int(row["nonzero_elements"]) > 0
                for row in rows
            )
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    launcher_path = Path(args.source_launcher_args).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-A Moga readiness requires an available CUDA device")
    if int(args.batch_size) != 32:
        raise ValueError("Locked Stage-A resource batch size is 32")
    if int(args.fp32_batch_size) != 2:
        raise ValueError("Locked Stage-A FP32 batch size is 2")
    if int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError(f"Locked benchmark repeat count is {BENCHMARK_REPEATS}")
    if not math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB):
        raise ValueError(f"Locked peak VRAM threshold is {MAX_PEAK_VRAM_GIB} GiB")
    if not math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO):
        raise ValueError(f"Locked runtime-ratio threshold is {MAX_RUNTIME_RATIO}")

    data_sha = _sha256(data_path)
    checkpoint_sha = _sha256(checkpoint_path)
    resolved_config_sha = _sha256(resolved_config_path)
    launcher_sha = _sha256(launcher_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(source_model_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Scratch checkpoint lacks model_config/class_names")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}")
    resolved_config = _load_json_mapping(resolved_config_path)
    launcher_args = _load_json_mapping(launcher_path)
    control_config = _control_config(source_model_config)
    candidate_config = _candidate_config(source_model_config)
    num_classes = len(class_names)

    set_seed(int(args.seed))
    control_model = create_model(num_classes=num_classes, model_config=control_config)
    control_state_sha = _state_sha256(control_model)
    control_state_names = list(control_model.state_dict())
    control_parameters = sum(parameter.numel() for parameter in control_model.parameters())

    default_control_config = dict(control_config)
    default_control_config.pop("stem_architecture", None)
    set_seed(int(args.seed))
    default_control_model = create_model(
        num_classes=num_classes,
        model_config=default_control_config,
    )
    default_control_state_sha = _state_sha256(default_control_model)
    default_control_state_names = list(default_control_model.state_dict())
    del default_control_model

    set_seed(int(args.seed))
    model = create_model(num_classes=num_classes, model_config=candidate_config)
    candidate_state_sha = _state_sha256(model)
    set_seed(int(args.seed))
    same_seed_model = create_model(num_classes=num_classes, model_config=candidate_config)
    same_seed_state_sha = _state_sha256(same_seed_model)
    same_seed_model.load_state_dict(model.state_dict(), strict=True)
    roundtrip_state_sha = _state_sha256(same_seed_model)
    del same_seed_model

    candidate_parameters = sum(parameter.numel() for parameter in model.parameters())
    added_parameters = int(candidate_parameters - control_parameters)
    stem = getattr(model, "stem", None)
    if not isinstance(stem, MogaXTTokenizer):
        raise TypeError("stem_architecture did not construct MogaXTTokenizer")
    stage_depths = (
        len(stem.blocks1),
        len(stem.blocks2),
        len(stem.blocks3),
    )

    dataset = _build_train_only_dataset(
        data_yaml=data_path,
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    if len(dataset) < int(args.batch_size):
        raise RuntimeError("Train dataset is smaller than the locked resource batch")
    data_spec = load_data_spec(data_path)
    train_root = Path(data_spec.split_images_dir("train")).resolve()
    selected_paths = [
        str(dataset.samples[index].image_path.resolve())
        for index in range(int(args.batch_size))
    ]
    balanced_indices: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        label = int(sample.primary_label)
        if label not in balanced_indices:
            balanced_indices[label] = int(index)
        if len(balanced_indices) == num_classes:
            break
    if set(balanced_indices) != set(range(num_classes)):
        raise RuntimeError("Could not construct a one-example-per-class train batch")
    ordered_balanced_indices = [balanced_indices[index] for index in range(num_classes)]
    balanced_paths = [
        str(dataset.samples[index].image_path.resolve())
        for index in ordered_balanced_indices
    ]
    train_only_paths = all(
        Path(path).is_relative_to(train_root)
        for path in selected_paths + balanced_paths
    )

    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="moga_tokenizer_stage_a_train_only",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **dataloader_kwargs,
    )
    resource_batch = next(iter(loader))
    balanced_batch = default_collate(
        [dataset[index] for index in ordered_balanced_indices]
    )

    device = torch.device("cuda")
    images, labels, metadata = _move_batch(resource_batch, device=device)
    balanced_images, balanced_labels, balanced_metadata = _move_batch(
        balanced_batch,
        device=device,
    )

    control_model = control_model.to(device)
    control_benchmark = _amp_forward_backward_benchmark(
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

    model = model.to(device)
    candidate_benchmark = _amp_forward_backward_benchmark(
        model=model,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    amp_gradients = summarize_moga_gradients(model)
    candidate_benchmark["gradients"] = amp_gradients
    runtime_ratio = float(
        float(candidate_benchmark["median_seconds"])
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )

    model.train()
    model.zero_grad(set_to_none=True)
    fp32_count = int(args.fp32_batch_size)
    logits_fp32, _ = _forward_logits(
        model,
        images[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    fp32_gradients = summarize_moga_gradients(model)
    fp32_finite = bool(
        torch.isfinite(logits_fp32).all() and torch.isfinite(loss_fp32)
    )

    activations = _activation_diagnostics(model=model, images=balanced_images)
    sensitivity = _balanced_input_sensitivity(
        model=model,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
    )

    expected_stage_shapes = {
        "stage1_output": [num_classes, 32, 64, 64],
        "stage2_output": [num_classes, 64, 32, 32],
        "stage3_output": [num_classes, 96, 16, 16],
    }
    stage_shapes_valid = all(
        activations["stages"][name]["shape"] == shape
        for name, shape in expected_stage_shapes.items()
    )
    stage_variance_valid = all(
        bool(activations["stages"][name]["finite"])
        and float(activations["stages"][name]["std"]) > 1e-4
        for name in expected_stage_shapes
    )
    branch_energy_valid = all(
        bool(item["finite"]) and float(item["rms"]) > 0.0
        for item in activations["branches"].values()
    )
    fp32_gradient_pass = all(
        bool(value["passed"]) for value in fp32_gradients.values()
    )
    amp_gradient_pass = all(
        bool(value["passed"]) for value in amp_gradients.values()
    )
    checks = {
        "locked_data_sha256": data_sha == LOCKED_DATA_SHA256,
        "locked_scratch_checkpoint_sha256": (
            checkpoint_sha == LOCKED_SCRATCH_CHECKPOINT_SHA256
        ),
        "locked_source_launcher_sha256": launcher_sha == LOCKED_SOURCE_LAUNCHER_SHA256,
        "source_protocol_scratch": bool(launcher_args.get("no_pretrained", False)),
        "candidate_pretrained_disabled": not bool(candidate_config.get("pretrained", False)),
        "train_split_only": bool(train_only_paths),
        "train_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "balanced_real_train_batch": (
            [int(value) for value in balanced_labels.detach().cpu().tolist()]
            == list(range(num_classes))
        ),
        "default_control_state_schema_unchanged": (
            control_state_names == default_control_state_names
        ),
        "default_control_bit_identical": control_state_sha == default_control_state_sha,
        "candidate_same_seed_deterministic": candidate_state_sha == same_seed_state_sha,
        "candidate_strict_state_roundtrip": candidate_state_sha == roundtrip_state_sha,
        "moga_tokenizer_present": isinstance(stem, MogaXTTokenizer),
        "locked_stage_depths": stage_depths == EXPECTED_STAGE_DEPTHS,
        "all_sixteen_moga_blocks": stem.block_count == EXPECTED_BLOCKS,
        "candidate_parameters_within_budget": (
            0 < candidate_parameters <= MAX_CANDIDATE_PARAMETERS
        ),
        "added_parameters_within_budget": (
            0 < added_parameters <= MAX_ADDED_PARAMETERS
        ),
        "stage_output_shapes": bool(stage_shapes_valid),
        "stem_output_shape": activations["stem_output"]["shape"] == [5, 96, 16, 16],
        "exact_patch_token_count": (
            int(getattr(model.patch_embed, "num_patches", -1))
            == EXPECTED_PATCH_TOKENS
        ),
        "fp32_logits_loss_finite": fp32_finite,
        "fp32_all_moga_gradient_families": fp32_gradient_pass,
        "amp_control_logits_loss_finite": bool(
            control_benchmark["logits_loss_finite"]
        ),
        "amp_candidate_logits_loss_finite": bool(
            candidate_benchmark["logits_loss_finite"]
        ),
        "amp_all_moga_gradient_families": amp_gradient_pass,
        "stage_activation_variance": bool(stage_variance_valid),
        "multi_order_branch_energy": bool(branch_energy_valid),
        "multi_order_branches_distinct": bool(activations["branches_distinct"]),
        "balanced_class_input_logit_sensitivity": bool(
            sensitivity["all_classes_live"]
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_moga_tokenizer_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        max_peak_vram_gib=float(args.max_peak_vram_gib),
        runtime_ratio=runtime_ratio,
        max_runtime_ratio=float(args.max_runtime_ratio),
    )

    summary: Dict[str, object] = {
        "method": "moganet_xt_surface_tokenizer",
        "protocol_stage": "A_train_only_functional_resource_preflight",
        "sources": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_weights_loaded": False,
            "resolved_config": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha,
            "source_launcher_args": str(launcher_path),
            "source_launcher_args_sha256": launcher_sha,
            "data_yaml": str(data_path),
            "data_yaml_sha256": data_sha,
            "split_loaded": "train",
            "validation_loaded": False,
            "test_loaded": False,
            "moganet_source_commit": MogaXTTokenizer.source_commit,
            "moganet_source_file_sha256": MogaXTTokenizer.source_sha256,
        },
        "model": {
            "class_names": [str(value) for value in class_names],
            "control_parameters": int(control_parameters),
            "candidate_parameters": int(candidate_parameters),
            "added_parameters": added_parameters,
            "stage_depths": list(stage_depths),
            "moga_blocks": int(stem.block_count),
            "patch_tokens": int(getattr(model.patch_embed, "num_patches", -1)),
            "control_state_sha256": control_state_sha,
            "default_control_state_sha256": default_control_state_sha,
            "candidate_state_sha256": candidate_state_sha,
            "candidate_same_seed_state_sha256": same_seed_state_sha,
            "candidate_roundtrip_state_sha256": roundtrip_state_sha,
        },
        "data": {
            "train_rows": len(dataset),
            "train_root": str(train_root),
            "resource_batch_size": int(images.size(0)),
            "resource_labels": [
                int(value) for value in labels.detach().cpu().tolist()
            ],
            "resource_paths": selected_paths,
            "balanced_indices": ordered_balanced_indices,
            "balanced_labels": [
                int(value) for value in balanced_labels.detach().cpu().tolist()
            ],
            "balanced_paths": balanced_paths,
            "dataloader": dataloader_summary,
        },
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "logits_shape": [int(size) for size in logits_fp32.shape],
            "gradients": fp32_gradients,
        },
        "cuda_amp": {
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_gib": float(
                torch.cuda.get_device_properties(device).total_memory / (1024**3)
            ),
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "candidate_control_runtime_ratio": runtime_ratio,
        },
        "activations": activations,
        "balanced_input_sensitivity": sensitivity,
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
            }
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
