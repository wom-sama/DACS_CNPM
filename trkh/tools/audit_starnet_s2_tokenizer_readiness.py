from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.inference.inference import load_checkpoint
from trkh.models.model import create_model
from trkh.models.starnet_s2_tokenizer import (
    StarNetConvBN,
    StarNetS2Tokenizer,
)
from trkh.tools.audit_moga_tokenizer_readiness import (
    BENCHMARK_REPEATS,
    LOCKED_DATA_SHA256,
    LOCKED_SCRATCH_CHECKPOINT_SHA256,
    LOCKED_SOURCE_LAUNCHER_SHA256,
    MAX_PEAK_VRAM_GIB,
    MAX_RUNTIME_RATIO,
    _amp_forward_backward_benchmark,
    _balanced_input_sensitivity,
    _control_config,
    _move_batch,
    _tensor_summary,
)
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _forward_logits,
    _load_json_mapping,
    _prepare_output_dir,
    _sha256,
    _state_sha256,
)


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_STAGE_DEPTHS = (1, 2, 6)
EXPECTED_STAGE_CHANNELS = (32, 64, 128)
EXPECTED_BLOCKS = 9
EXPECTED_PATCH_TOKENS = 256
MAX_CANDIDATE_PARAMETERS = 10_000_000
MAX_ADDED_PARAMETERS = 3_000_000
MIN_MECHANISM_DELTA = 1e-7

STARNET_GRADIENT_MARKERS = {
    "stem_embedding": ("stem.stem.0.",),
    "stage_downsampling": (".stages.0.downsample.", ".stages.1.downsample.", ".stages.2.downsample."),
    "first_depthwise": (".dwconv.",),
    "f1_projection": (".f1.",),
    "f2_projection": (".f2.",),
    "g_projection": (".g.",),
    "second_depthwise": (".dwconv2.",),
    "patch_projection": ("patch_embed.proj.",),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only functional/resource readiness audit for the scratch "
            "StarNet-S2 stage-1-to-3 TRKH tokenizer."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/checkpoints/best.pt"
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


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "stem_architecture": "starnet_s2_tokenizer",
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def summarize_starnet_gradients(
    model: nn.Module,
) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in STARNET_GRADIENT_MARKERS.items():
        selected = {
            name: parameter
            for name, parameter in named_parameters.items()
            if any(marker in name for marker in markers)
        }
        present = {
            name: parameter.grad
            for name, parameter in selected.items()
            if parameter.grad is not None
        }
        finite = bool(present) and all(
            bool(torch.isfinite(gradient).all()) for gradient in present.values()
        )
        required_weights = {
            name: gradient
            for name, gradient in present.items()
            if named_parameters[name].ndim >= 2
        }
        live_weights = {
            name: gradient
            for name, gradient in required_weights.items()
            if int(torch.count_nonzero(gradient.detach()).item()) > 0
        }
        nonzero = sum(
            int(torch.count_nonzero(gradient.detach()).item())
            for gradient in present.values()
        )
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "required_weight_tensors": len(required_weights),
            "live_weight_tensors": len(live_weights),
            "finite": finite,
            "nonzero_elements": int(nonzero),
            "passed": bool(
                selected
                and len(present) == len(selected)
                and required_weights
                and len(live_weights) == len(required_weights)
                and finite
                and nonzero > 0
            ),
        }
    return summary


def assess_starnet_readiness(
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
            "expected_stage_channels": list(EXPECTED_STAGE_CHANNELS),
            "expected_blocks": EXPECTED_BLOCKS,
            "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
            "max_candidate_parameters": MAX_CANDIDATE_PARAMETERS,
            "max_added_parameters": MAX_ADDED_PARAMETERS,
            "minimum_mechanism_delta": MIN_MECHANISM_DELTA,
            "max_peak_vram_gib": float(max_peak_vram_gib),
            "max_runtime_ratio": float(max_runtime_ratio),
        },
    }


def _conv_spec(module: StarNetConvBN) -> Dict[str, object]:
    conv = module.conv
    kernel = tuple(int(value) for value in conv.kernel_size)
    stride = tuple(int(value) for value in conv.stride)
    padding = tuple(int(value) for value in conv.padding)
    fan_in = int(conv.in_channels // conv.groups) * kernel[0] * kernel[1]
    bound = 1.0 / math.sqrt(float(fan_in))
    weight_max = float(conv.weight.detach().abs().max().item())
    bias_max = (
        float(conv.bias.detach().abs().max().item())
        if conv.bias is not None
        else math.inf
    )
    bn_initial = True
    if module.with_bn:
        if not isinstance(module.bn, nn.BatchNorm2d):
            bn_initial = False
        else:
            bn_initial = bool(
                torch.equal(module.bn.weight.detach(), torch.ones_like(module.bn.weight))
                and torch.equal(module.bn.bias.detach(), torch.zeros_like(module.bn.bias))
            )
    return {
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "kernel_size": list(kernel),
        "stride": list(stride),
        "padding": list(padding),
        "groups": int(conv.groups),
        "bias": conv.bias is not None,
        "with_bn": bool(module.with_bn),
        "weight_max_abs": weight_max,
        "bias_max_abs": bias_max,
        "default_initialization_bound": bound,
        "default_initialization_valid": bool(
            math.isfinite(weight_max)
            and math.isfinite(bias_max)
            and weight_max <= bound + 1e-7
            and bias_max <= bound + 1e-7
            and bn_initial
        ),
    }


def _architecture_summary(stem: StarNetS2Tokenizer) -> Dict[str, object]:
    stem_spec = _conv_spec(stem.stem[0])
    downsampling = []
    blocks = []
    locked = bool(
        stem_spec["in_channels"] == 3
        and stem_spec["out_channels"] == 32
        and stem_spec["kernel_size"] == [3, 3]
        and stem_spec["stride"] == [2, 2]
        and stem_spec["padding"] == [1, 1]
        and stem_spec["with_bn"]
        and isinstance(stem.stem[1], nn.ReLU6)
    )
    expected_inputs = (32, 32, 64)
    for stage_index, stage in enumerate(stem.stages, start=1):
        expected_channels = EXPECTED_STAGE_CHANNELS[stage_index - 1]
        downsample = _conv_spec(stage.downsample)
        downsampling.append(downsample)
        locked = locked and bool(
            downsample["in_channels"] == expected_inputs[stage_index - 1]
            and downsample["out_channels"] == expected_channels
            and downsample["kernel_size"] == [3, 3]
            and downsample["stride"] == [2, 2]
            and downsample["padding"] == [1, 1]
            and downsample["with_bn"]
        )
        for block_index, block in enumerate(stage.blocks, start=1):
            dwconv = _conv_spec(block.dwconv)
            f1 = _conv_spec(block.f1)
            f2 = _conv_spec(block.f2)
            g = _conv_spec(block.g)
            dwconv2 = _conv_spec(block.dwconv2)
            item = {
                "stage": stage_index,
                "block": block_index,
                "channels": int(block.channels),
                "expansion": int(block.expansion),
                "dwconv": dwconv,
                "f1": f1,
                "f2": f2,
                "g": g,
                "dwconv2": dwconv2,
                "activation": type(block.act).__name__,
                "drop_path": type(block.drop_path).__name__,
            }
            blocks.append(item)
            locked = locked and bool(
                item["channels"] == expected_channels
                and item["expansion"] == 4
                and dwconv["kernel_size"] == [7, 7]
                and dwconv["groups"] == expected_channels
                and dwconv["with_bn"]
                and f1["kernel_size"] == [1, 1]
                and f1["out_channels"] == 4 * expected_channels
                and not f1["with_bn"]
                and f2["kernel_size"] == [1, 1]
                and f2["out_channels"] == 4 * expected_channels
                and not f2["with_bn"]
                and g["kernel_size"] == [1, 1]
                and g["in_channels"] == 4 * expected_channels
                and g["out_channels"] == expected_channels
                and g["with_bn"]
                and dwconv2["kernel_size"] == [7, 7]
                and dwconv2["groups"] == expected_channels
                and not dwconv2["with_bn"]
                and all(spec["bias"] for spec in (dwconv, f1, f2, g, dwconv2))
                and item["activation"] == "ReLU6"
                and item["drop_path"] == "Identity"
            )
    initialization_valid = all(
        bool(spec["default_initialization_valid"])
        for spec in [stem_spec, *downsampling]
    ) and all(
        bool(item[name]["default_initialization_valid"])
        for item in blocks
        for name in ("dwconv", "f1", "f2", "g", "dwconv2")
    )
    return {
        "stem": stem_spec,
        "stage_depths": [len(stage.blocks) for stage in stem.stages],
        "stage_channels": list(stem.stage_channels),
        "stage_width_depth_pairs": [list(value) for value in stem.stage_width_depth_pairs],
        "block_count": int(stem.block_count),
        "downsampling": downsampling,
        "blocks": blocks,
        "actual_official_initialization_valid": bool(initialization_valid),
        "all_official_s2_settings_locked": bool(locked),
    }


def _activation_diagnostics(*, model: nn.Module, images: Tensor) -> Dict[str, object]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, StarNetS2Tokenizer):
        raise TypeError("Candidate model lacks StarNetS2Tokenizer")
    model.eval()
    stage_summaries: Dict[str, Dict[str, object]] = {}
    interactions = []
    with torch.inference_mode():
        x = stem.stem(images)
        stem_embedding = x
        for stage_index, stage in enumerate(stem.stages, start=1):
            x = stage.downsample(x)
            stage_summaries[f"stage{stage_index}_embedded"] = _tensor_summary(x)
            for block_index, block in enumerate(stage.blocks, start=1):
                features = block.interaction_features(x)
                star_output = block.forward_with_interaction(x, interaction="star")
                sum_output = block.forward_with_interaction(x, interaction="sum")
                delta = float((star_output - sum_output).abs().mean().item())
                interactions.append(
                    {
                        "stage": stage_index,
                        "block": block_index,
                        "f1": _tensor_summary(features["f1"]),
                        "f2": _tensor_summary(features["f2"]),
                        "activated_f1": _tensor_summary(features["activated_f1"]),
                        "star": _tensor_summary(features["star"]),
                        "same_weight_sum_output_delta": delta,
                        "product_differs_from_f1": not torch.equal(
                            features["star"], features["f1"]
                        ),
                        "product_differs_from_f2": not torch.equal(
                            features["star"], features["f2"]
                        ),
                    }
                )
                x = star_output
            stage_summaries[f"stage{stage_index}_output"] = _tensor_summary(x)
        stem_output = x
    return {
        "stem_embedding": _tensor_summary(stem_embedding),
        "stem_output": _tensor_summary(stem_output),
        "stages": stage_summaries,
        "interactions": interactions,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    launcher_path = Path(args.source_launcher_args).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-A StarNet-S2 readiness requires CUDA")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked batch sizes are AMP=32 and FP32=2")
    if int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError(f"Locked benchmark repeats are {BENCHMARK_REPEATS}")
    if not math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB):
        raise ValueError(f"Locked peak VRAM threshold is {MAX_PEAK_VRAM_GIB}")
    if not math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO):
        raise ValueError(f"Locked runtime threshold is {MAX_RUNTIME_RATIO}")

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
    schema_probe = torch.linspace(-1.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    control_model.eval()
    default_control_model.eval()
    with torch.inference_mode():
        control_probe_logits, _ = _forward_logits(control_model, schema_probe, {})
        default_probe_logits, _ = _forward_logits(
            default_control_model,
            schema_probe,
            {},
        )
    default_control_logits_identical = torch.equal(
        control_probe_logits,
        default_probe_logits,
    )
    del default_control_model, control_probe_logits, default_probe_logits, schema_probe

    set_seed(int(args.seed))
    model = create_model(num_classes=num_classes, model_config=candidate_config)
    candidate_state_sha = _state_sha256(model)
    set_seed(int(args.seed))
    same_seed_model = create_model(num_classes=num_classes, model_config=candidate_config)
    same_seed_state_sha = _state_sha256(same_seed_model)
    same_seed_model.load_state_dict(model.state_dict(), strict=True)
    roundtrip_state_sha = _state_sha256(same_seed_model)
    del same_seed_model

    stem = getattr(model, "stem", None)
    if not isinstance(stem, StarNetS2Tokenizer):
        raise TypeError("stem_architecture did not construct the candidate")
    architecture = _architecture_summary(stem)
    candidate_parameters = sum(parameter.numel() for parameter in model.parameters())
    added_parameters = int(candidate_parameters - control_parameters)

    dataset = _build_train_only_dataset(
        data_yaml=data_path,
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
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
        raise RuntimeError("Could not construct a balanced real-train batch")
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
        context="starnet_s2_stage_a_train_only",
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
    amp_gradients = summarize_starnet_gradients(model)
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
    fp32_gradients = summarize_starnet_gradients(model)
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
        "stage1_output": [5, 32, 64, 64],
        "stage2_output": [5, 64, 32, 32],
        "stage3_output": [5, 128, 16, 16],
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
    interaction_energy_valid = all(
        all(
            bool(item[name]["finite"]) and float(item[name]["rms"]) > 0.0
            for name in ("f1", "f2", "activated_f1", "star")
        )
        for item in activations["interactions"]
    )
    interaction_distinct_valid = all(
        bool(item["product_differs_from_f1"])
        and bool(item["product_differs_from_f2"])
        for item in activations["interactions"]
    )
    mechanism_delta_valid = all(
        math.isfinite(float(item["same_weight_sum_output_delta"]))
        and float(item["same_weight_sum_output_delta"]) > MIN_MECHANISM_DELTA
        for item in activations["interactions"]
    )
    stage3_gradient_live = bool(
        fp32_gradients["f1_projection"]["passed"]
        and fp32_gradients["f2_projection"]["passed"]
        and amp_gradients["f1_projection"]["passed"]
        and amp_gradients["f2_projection"]["passed"]
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
        "default_control_bit_identical": (
            control_state_sha == default_control_state_sha
            and default_control_logits_identical
        ),
        "candidate_same_seed_deterministic": candidate_state_sha == same_seed_state_sha,
        "candidate_strict_state_roundtrip": candidate_state_sha == roundtrip_state_sha,
        "starnet_tokenizer_present": isinstance(stem, StarNetS2Tokenizer),
        "official_s2_settings_locked": bool(
            architecture["all_official_s2_settings_locked"]
        ),
        "official_initialization_locked": bool(
            architecture["actual_official_initialization_valid"]
        ),
        "locked_stage_depths": tuple(architecture["stage_depths"]) == EXPECTED_STAGE_DEPTHS,
        "locked_stage_channels": tuple(architecture["stage_channels"]) == EXPECTED_STAGE_CHANNELS,
        "all_nine_blocks": int(architecture["block_count"]) == EXPECTED_BLOCKS,
        "candidate_parameters_within_budget": (
            0 < candidate_parameters <= MAX_CANDIDATE_PARAMETERS
        ),
        "added_parameters_within_budget": (
            0 < added_parameters <= MAX_ADDED_PARAMETERS
        ),
        "stage_output_shapes": bool(stage_shapes_valid),
        "stem_output_shape": activations["stem_output"]["shape"] == [5, 128, 16, 16],
        "exact_patch_token_count": (
            int(getattr(model.patch_embed, "num_patches", -1)) == EXPECTED_PATCH_TOKENS
        ),
        "fp32_logits_loss_finite": fp32_finite,
        "fp32_all_gradient_families": all(
            bool(value["passed"]) for value in fp32_gradients.values()
        ),
        "amp_control_logits_loss_finite": bool(control_benchmark["logits_loss_finite"]),
        "amp_candidate_logits_loss_finite": bool(candidate_benchmark["logits_loss_finite"]),
        "amp_all_gradient_families": all(
            bool(value["passed"]) for value in amp_gradients.values()
        ),
        "stage_activation_variance": bool(stage_variance_valid),
        "all_star_interactions_live": bool(interaction_energy_valid),
        "star_products_distinct": bool(interaction_distinct_valid),
        "same_weight_star_sum_mechanism": bool(mechanism_delta_valid),
        "stage3_f1_f2_gradients_live": bool(stage3_gradient_live),
        "balanced_class_input_logit_sensitivity": bool(
            sensitivity["all_classes_live"]
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_starnet_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        max_peak_vram_gib=float(args.max_peak_vram_gib),
        runtime_ratio=runtime_ratio,
        max_runtime_ratio=float(args.max_runtime_ratio),
    )
    summary: Dict[str, object] = {
        "method": "starnet_s2_local_multiplicative_tokenizer",
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
            "official_source_commit": StarNetS2Tokenizer.source_commit,
            "official_source_file_sha256": StarNetS2Tokenizer.source_sha256,
        },
        "model": {
            "class_names": [str(value) for value in class_names],
            "control_parameters": int(control_parameters),
            "candidate_parameters": int(candidate_parameters),
            "added_parameters": added_parameters,
            "patch_tokens": int(getattr(model.patch_embed, "num_patches", -1)),
            "architecture": architecture,
            "control_state_sha256": control_state_sha,
            "default_control_state_sha256": default_control_state_sha,
            "default_control_logits_identical": bool(default_control_logits_identical),
            "candidate_state_sha256": candidate_state_sha,
            "candidate_same_seed_state_sha256": same_seed_state_sha,
            "candidate_roundtrip_state_sha256": roundtrip_state_sha,
        },
        "data": {
            "train_rows": len(dataset),
            "train_root": str(train_root),
            "resource_batch_size": int(images.size(0)),
            "resource_labels": [int(value) for value in labels.detach().cpu().tolist()],
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
