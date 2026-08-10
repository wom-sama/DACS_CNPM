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
from trkh.models.inceptionnext_atto_tokenizer import InceptionNeXtAttoTokenizer
from trkh.models.model import create_model
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
EXPECTED_STAGE_DEPTHS = (2, 2, 6)
EXPECTED_STAGE_CHANNELS = (40, 80, 160)
EXPECTED_STAGE_SPLITS = (
    (10, 10, 10, 10),
    (20, 20, 20, 20),
    (40, 40, 40, 40),
)
EXPECTED_BLOCKS = 10
EXPECTED_PATCH_TOKENS = 256
MAX_CANDIDATE_PARAMETERS = 10_000_000
MAX_ADDED_PARAMETERS = 3_000_000

INCEPTIONNEXT_GRADIENT_MARKERS = {
    "stem_embedding": ("stem.stem.",),
    "stage_downsampling": (
        "stem.stages.1.downsample.",
        "stem.stages.2.downsample.",
    ),
    "square_branch": (".token_mixer.dwconv_hw.",),
    "horizontal_branch": (".token_mixer.dwconv_w.",),
    "vertical_branch": (".token_mixer.dwconv_h.",),
    "block_normalization": (".norm.",),
    "mlp_input_projection": (".mlp.fc1.",),
    "mlp_output_projection": (".mlp.fc2.",),
    "layer_scales": (".gamma",),
    "patch_projection": ("patch_embed.proj.",),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only functional/resource readiness audit for the scratch "
            "InceptionNeXt-Atto stage-1-to-3 TRKH tokenizer."
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
            "stem_architecture": "inceptionnext_atto_tokenizer",
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def summarize_inceptionnext_gradients(
    model: nn.Module,
) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in INCEPTIONNEXT_GRADIENT_MARKERS.items():
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


def assess_inceptionnext_readiness(
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
            "expected_stage_splits": [list(value) for value in EXPECTED_STAGE_SPLITS],
            "expected_blocks": EXPECTED_BLOCKS,
            "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
            "max_candidate_parameters": MAX_CANDIDATE_PARAMETERS,
            "max_added_parameters": MAX_ADDED_PARAMETERS,
            "max_peak_vram_gib": float(max_peak_vram_gib),
            "max_runtime_ratio": float(max_runtime_ratio),
        },
    }


def _architecture_summary(stem: InceptionNeXtAttoTokenizer) -> Dict[str, object]:
    blocks = []
    locked = True
    for stage_index, stage in enumerate(stem.stages, start=1):
        expected_channels = EXPECTED_STAGE_CHANNELS[stage_index - 1]
        for block_index, block in enumerate(stage.blocks, start=1):
            mixer = block.token_mixer
            gamma = block.gamma.detach()
            item = {
                "stage": stage_index,
                "block": block_index,
                "channels": int(mixer.channels),
                "split_channels": [int(value) for value in mixer.split_channels],
                "square_kernel_size": int(mixer.square_kernel_size),
                "band_kernel_size": int(mixer.band_kernel_size),
                "branch_ratio": float(mixer.branch_ratio),
                "mlp_hidden_channels": int(block.mlp.fc1.out_channels),
                "gamma_minimum": float(gamma.min().item()),
                "gamma_maximum": float(gamma.max().item()),
            }
            blocks.append(item)
            locked = locked and bool(
                item["channels"] == expected_channels
                and tuple(item["split_channels"])
                == EXPECTED_STAGE_SPLITS[stage_index - 1]
                and item["square_kernel_size"] == 3
                and item["band_kernel_size"] == 9
                and math.isclose(item["branch_ratio"], 0.25)
                and item["mlp_hidden_channels"] == 4 * expected_channels
                and math.isclose(item["gamma_minimum"], 1e-6, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(item["gamma_maximum"], 1e-6, rel_tol=0.0, abs_tol=1e-12)
            )
    return {
        "stage_depths": [len(stage.blocks) for stage in stem.stages],
        "stage_channels": list(stem.stage_channels),
        "stage_splits": [list(value) for value in stem.stage_splits],
        "block_count": int(stem.block_count),
        "blocks": blocks,
        "all_official_atto_settings_locked": bool(locked),
    }


def _activation_diagnostics(
    *,
    model: nn.Module,
    images: Tensor,
) -> Dict[str, object]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, InceptionNeXtAttoTokenizer):
        raise TypeError("Candidate model lacks InceptionNeXtAttoTokenizer")
    model.eval()
    with torch.inference_mode():
        stem_output, trace = stem.forward_with_trace(images)
        embedded = stem.stem(images)
        branch_features = stem.stages[0].blocks[0].token_mixer.branch_features(
            embedded
        )
    stage_summaries = {
        name: _tensor_summary(value) for name, value in trace.items()
    }
    branch_summaries = {
        name: _tensor_summary(value) for name, value in branch_features.items()
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
        "branches_distinct": len(signatures) == 4,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    launcher_path = Path(args.source_launcher_args).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-A InceptionNeXt readiness requires CUDA")
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

    stem = getattr(model, "stem", None)
    if not isinstance(stem, InceptionNeXtAttoTokenizer):
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
        context="inceptionnext_atto_stage_a_train_only",
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
    amp_gradients = summarize_inceptionnext_gradients(model)
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
    fp32_gradients = summarize_inceptionnext_gradients(model)
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
        "stage1_output": [5, 40, 64, 64],
        "stage2_output": [5, 80, 32, 32],
        "stage3_output": [5, 160, 16, 16],
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
        "inceptionnext_tokenizer_present": isinstance(stem, InceptionNeXtAttoTokenizer),
        "official_atto_settings_locked": bool(
            architecture["all_official_atto_settings_locked"]
        ),
        "locked_stage_depths": tuple(architecture["stage_depths"]) == EXPECTED_STAGE_DEPTHS,
        "locked_stage_channels": tuple(architecture["stage_channels"]) == EXPECTED_STAGE_CHANNELS,
        "locked_stage_splits": tuple(
            tuple(value) for value in architecture["stage_splits"]
        ) == EXPECTED_STAGE_SPLITS,
        "all_ten_blocks": int(architecture["block_count"]) == EXPECTED_BLOCKS,
        "candidate_parameters_within_budget": (
            0 < candidate_parameters <= MAX_CANDIDATE_PARAMETERS
        ),
        "added_parameters_within_budget": (
            0 < added_parameters <= MAX_ADDED_PARAMETERS
        ),
        "stage_output_shapes": bool(stage_shapes_valid),
        "stem_output_shape": activations["stem_output"]["shape"] == [5, 160, 16, 16],
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
        "four_branch_energy": bool(branch_energy_valid),
        "four_branches_distinct": bool(activations["branches_distinct"]),
        "balanced_class_input_logit_sensitivity": bool(
            sensitivity["all_classes_live"]
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_inceptionnext_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        max_peak_vram_gib=float(args.max_peak_vram_gib),
        runtime_ratio=runtime_ratio,
        max_runtime_ratio=float(args.max_runtime_ratio),
    )
    summary: Dict[str, object] = {
        "method": "inceptionnext_atto_surface_tokenizer",
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
            "official_source_commit": InceptionNeXtAttoTokenizer.source_commit,
            "official_source_file_sha256": InceptionNeXtAttoTokenizer.source_sha256,
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
