from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
from trkh.models.model import HybridConvStem, create_model
from trkh.models.octave_conv_stem import OctaveConvStem
from trkh.tools.audit_moga_tokenizer_readiness import (
    BENCHMARK_REPEATS,
    LOCKED_DATA_SHA256,
    LOCKED_SCRATCH_CHECKPOINT_SHA256,
    LOCKED_SOURCE_LAUNCHER_SHA256,
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
EXPECTED_PATCH_TOKENS = 256
LOCKED_ALPHA = 0.125
MAX_PEAK_VRAM_GIB = 3.25
MAX_RUNTIME_RATIO = 1.50
MAX_CONTRIBUTION_COSINE = 0.9995
MIN_PATH_ABLATION_DELTA = 1e-6
MAX_ONNX_ERROR = 1e-5
EXPECTED_SOURCE_COMMIT = "87c44f79162f3a2ef316bf924ad3697b8957a463"
EXPECTED_SOURCE_SHA256 = "06e60037b8fd5d4cd9e063ab1d439b25ddc98c4a109f770cbc0adcb9d7295675"

PATH_GRADIENT_PARAMETERS = {
    "block1_hh": "stem.blocks.0.octave.conv_hh.weight",
    "block1_hl": "stem.blocks.0.octave.conv_hl.weight",
    "block2_hh": "stem.blocks.1.octave.conv_hh.weight",
    "block2_hl": "stem.blocks.1.octave.conv_hl.weight",
    "block2_ll": "stem.blocks.1.octave.conv_ll.weight",
    "block2_lh": "stem.blocks.1.octave.conv_lh.weight",
    "block3_hh": "stem.blocks.2.octave.conv_hh.weight",
    "block3_lh": "stem.blocks.2.octave.conv_lh.weight",
    "patch_projection": "patch_embed.proj.weight",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only functional/resource readiness audit for OctConv TRKH."
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
            "stem_architecture": "octave_conv",
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def _tensor_sha256(value: Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _rng_snapshot() -> Dict[str, object]:
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in cuda_states],
    }


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_cpu = left["cpu"]
    right_cpu = right["cpu"]
    left_cuda = left["cuda"]
    right_cuda = right["cuda"]
    return bool(
        isinstance(left_cpu, Tensor)
        and isinstance(right_cpu, Tensor)
        and torch.equal(left_cpu, right_cpu)
        and isinstance(left_cuda, list)
        and isinstance(right_cuda, list)
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def _rng_summary(snapshot: Mapping[str, object]) -> Dict[str, object]:
    cpu = snapshot["cpu"]
    cuda = snapshot["cuda"]
    if not isinstance(cpu, Tensor) or not isinstance(cuda, list):
        raise TypeError("Invalid RNG snapshot.")
    return {
        "cpu_sha256": _tensor_sha256(cpu),
        "cuda_sha256": [_tensor_sha256(value) for value in cuda],
    }


def summarize_octave_gradients(model: nn.Module) -> Dict[str, Dict[str, object]]:
    named = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, name in PATH_GRADIENT_PARAMETERS.items():
        parameter = named.get(name)
        gradient = parameter.grad if parameter is not None else None
        finite = bool(gradient is not None and torch.isfinite(gradient).all())
        nonzero = (
            int(torch.count_nonzero(gradient.detach()).item())
            if gradient is not None
            else 0
        )
        summary[family] = {
            "parameter": name,
            "present": parameter is not None,
            "gradient_present": gradient is not None,
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": bool(parameter is not None and finite and nonzero > 0),
        }
    return summary


def _flatten_cosine(left: Tensor, right: Tensor) -> Dict[str, object]:
    if left.shape != right.shape:
        raise ValueError("Contribution cosine requires matching tensor shapes.")
    values = F.cosine_similarity(
        left.detach().float().flatten(1),
        right.detach().float().flatten(1),
        dim=1,
        eps=1e-8,
    )
    return {
        "values": [float(value) for value in values.cpu().tolist()],
        "mean": float(values.mean().item()),
        "maximum_absolute": float(values.abs().max().item()),
        "finite": bool(torch.isfinite(values).all()),
    }


def _normalized_total_variation(value: Tensor) -> Tensor:
    value = value.detach().float()
    horizontal = (
        (value[..., 1:] - value[..., :-1]).abs().flatten(1).mean(dim=1)
    )
    vertical = (
        (value[..., 1:, :] - value[..., :-1, :]).abs().flatten(1).mean(dim=1)
    )
    scale = value.abs().flatten(1).mean(dim=1).clamp_min(1e-8)
    return (horizontal + vertical) / scale


def _activation_diagnostics(model: nn.Module, images: Tensor) -> Dict[str, object]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, OctaveConvStem):
        raise TypeError("Candidate model lacks OctaveConvStem.")
    stem.eval()
    with torch.inference_mode():
        output, trace = stem.forward_with_trace(images)
    high_energy = trace["block2_high_output"].abs().mean(dim=1, keepdim=True)
    low_energy = trace["block2_low_output"].abs().mean(dim=1, keepdim=True)
    low_energy_up = F.interpolate(low_energy, size=high_energy.shape[-2:], mode="nearest")
    high_tv = _normalized_total_variation(high_energy)
    low_tv = _normalized_total_variation(low_energy_up)
    hh_lh = _flatten_cosine(trace["block2_path_hh"], trace["block2_path_lh"])
    ll_hl = _flatten_cosine(trace["block2_path_ll"], trace["block2_path_hl"])
    energy_cosine = _flatten_cosine(high_energy, low_energy_up)
    return {
        "stem_output": _tensor_summary(output),
        "trace": {name: _tensor_summary(value) for name, value in trace.items()},
        "contribution_cosines": {
            "middle_hh_lh": hh_lh,
            "middle_ll_hl": ll_hl,
            "middle_high_low_energy": energy_cosine,
        },
        "normalized_total_variation": {
            "high": [float(value) for value in high_tv.cpu().tolist()],
            "low_upsampled": [float(value) for value in low_tv.cpu().tolist()],
            "high_mean": float(high_tv.mean().item()),
            "low_mean": float(low_tv.mean().item()),
            "low_below_high": bool(low_tv.mean() < high_tv.mean()),
        },
    }


def _path_ablation_diagnostics(
    *,
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Dict[str, object]:
    stem = getattr(model, "stem", None)
    if not isinstance(stem, OctaveConvStem):
        raise TypeError("Candidate model lacks OctaveConvStem.")
    model.eval()
    with torch.inference_mode():
        baseline_stem = stem(images)
        baseline_logits, _ = _forward_logits(model, images, metadata)
    targets = {
        "block1_hl": stem.blocks[0].octave.conv_hl,
        "block2_hl": stem.blocks[1].octave.conv_hl,
        "block2_lh": stem.blocks[1].octave.conv_lh,
        "block3_lh": stem.blocks[2].octave.conv_lh,
    }
    rows = []
    for name, module in targets.items():
        if module is None:
            raise RuntimeError(f"Locked OctConv path is missing: {name}")
        saved = module.weight.detach().clone()
        try:
            with torch.no_grad():
                module.weight.zero_()
                changed_stem = stem(images)
                changed_logits, _ = _forward_logits(model, images, metadata)
        finally:
            with torch.no_grad():
                module.weight.copy_(saved)
        rows.append(
            {
                "path": name,
                "stem_max_abs_delta": float(
                    (changed_stem - baseline_stem).abs().max().item()
                ),
                "logit_max_abs_delta": float(
                    (changed_logits - baseline_logits).abs().max().item()
                ),
            }
        )
    return {
        "rows": rows,
        "all_paths_material": all(
            float(row["stem_max_abs_delta"]) >= MIN_PATH_ABLATION_DELTA
            and float(row["logit_max_abs_delta"]) >= MIN_PATH_ABLATION_DELTA
            for row in rows
        ),
    }


def _export_diagnostics(
    *,
    stem: OctaveConvStem,
    images: Tensor,
    output_dir: Path,
) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    export_stem = copy.deepcopy(stem).cpu().eval()
    path = output_dir / "octave_conv_stem.onnx"
    probe = images[:1].detach().cpu()
    torch.onnx.export(
        export_stem,
        (probe,),
        str(path),
        input_names=["images"],
        output_names=["stem_features"],
        dynamic_axes={"images": {0: "batch"}, "stem_features": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rows = []
    maximum_error = 0.0
    for batch_size in (1, 2):
        batch = images[:batch_size].detach().cpu()
        with torch.inference_mode():
            expected = export_stem(batch).detach().cpu()
        observed = torch.from_numpy(
            session.run(None, {"images": batch.numpy()})[0]
        )
        error = float((expected - observed).abs().max().item())
        maximum_error = max(maximum_error, error)
        rows.append(
            {
                "batch_size": batch_size,
                "shape": [int(value) for value in observed.shape],
                "finite": bool(torch.isfinite(observed).all()),
                "maximum_absolute_error": error,
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "opset": 17,
        "providers": session.get_providers(),
        "batch_results": rows,
        "maximum_absolute_error": maximum_error,
    }


def assess_octave_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    runtime_ratio: float,
) -> Dict[str, object]:
    resolved = {str(name): bool(value) for name, value in checks.items()}
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
        "full_train_permission": False,
        "checks": resolved,
        "failed_checks": failed,
        "thresholds": {
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
            "expected_patch_tokens": EXPECTED_PATCH_TOKENS,
            "locked_alpha": LOCKED_ALPHA,
            "maximum_contribution_cosine": MAX_CONTRIBUTION_COSINE,
            "minimum_path_ablation_delta": MIN_PATH_ABLATION_DELTA,
            "maximum_onnx_error": MAX_ONNX_ERROR,
            "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            "max_runtime_ratio": MAX_RUNTIME_RATIO,
        },
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    launcher_path = Path(args.source_launcher_args).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-A OctConv readiness requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked Stage-A batch sizes are AMP=32 and FP32=2.")
    if int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError(f"Locked benchmark repeats are {BENCHMARK_REPEATS}.")
    if not math.isclose(float(args.max_peak_vram_gib), MAX_PEAK_VRAM_GIB):
        raise ValueError(f"Locked peak VRAM threshold is {MAX_PEAK_VRAM_GIB}.")
    if not math.isclose(float(args.max_runtime_ratio), MAX_RUNTIME_RATIO):
        raise ValueError(f"Locked runtime threshold is {MAX_RUNTIME_RATIO}.")

    data_sha = _sha256(data_path)
    checkpoint_sha = _sha256(checkpoint_path)
    resolved_config_sha = _sha256(resolved_config_path)
    launcher_sha = _sha256(launcher_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(source_model_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Scratch checkpoint lacks model_config/class_names.")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}.")
    resolved_config = _load_json_mapping(resolved_config_path)
    launcher_args = _load_json_mapping(launcher_path)
    control_config = _control_config(source_model_config)
    candidate_config = _candidate_config(source_model_config)
    num_classes = len(class_names)

    set_seed(int(args.seed))
    control_model = create_model(num_classes=num_classes, model_config=control_config)
    control_rng = _rng_snapshot()
    control_state_sha = _state_sha256(control_model)
    control_parameters = sum(parameter.numel() for parameter in control_model.parameters())
    control_stem_parameters = sum(
        parameter.numel() for parameter in control_model.stem.parameters()
    )

    default_control_config = dict(control_config)
    default_control_config.pop("stem_architecture", None)
    set_seed(int(args.seed))
    default_control_model = create_model(
        num_classes=num_classes,
        model_config=default_control_config,
    )
    default_control_rng = _rng_snapshot()
    default_control_state_sha = _state_sha256(default_control_model)
    schema_probe = torch.linspace(-1.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    control_model.eval()
    default_control_model.eval()
    with torch.inference_mode():
        control_probe_logits, _ = _forward_logits(control_model, schema_probe, {})
        default_probe_logits, _ = _forward_logits(default_control_model, schema_probe, {})
    default_control_logits_identical = torch.equal(
        control_probe_logits, default_probe_logits
    )
    del default_control_model, control_probe_logits, default_probe_logits, schema_probe

    set_seed(int(args.seed))
    model = create_model(num_classes=num_classes, model_config=candidate_config)
    candidate_rng = _rng_snapshot()
    candidate_state_sha = _state_sha256(model)
    set_seed(int(args.seed))
    repeated_model = create_model(num_classes=num_classes, model_config=candidate_config)
    repeated_rng = _rng_snapshot()
    repeated_state_sha = _state_sha256(repeated_model)
    repeated_model.load_state_dict(model.state_dict(), strict=True)
    roundtrip_state_sha = _state_sha256(repeated_model)
    del repeated_model

    stem = getattr(model, "stem", None)
    if not isinstance(stem, OctaveConvStem):
        raise TypeError("stem_architecture did not construct OctaveConvStem.")
    if not isinstance(control_model.stem, HybridConvStem):
        raise TypeError("Matched control did not construct HybridConvStem.")
    candidate_parameters = sum(parameter.numel() for parameter in model.parameters())
    candidate_stem_parameters = sum(parameter.numel() for parameter in stem.parameters())
    control_nonstem = {
        name: value
        for name, value in control_model.state_dict().items()
        if not name.startswith("stem.")
    }
    candidate_nonstem = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("stem.")
    }
    nonstem_bit_identical = bool(
        control_nonstem.keys() == candidate_nonstem.keys()
        and all(
            torch.equal(value, candidate_nonstem[name])
            for name, value in control_nonstem.items()
        )
    )
    control_kernels = [
        block.block.conv.weight.detach()
        for block in control_model.stem.blocks
    ]
    candidate_kernels = stem.reconstructed_vanilla_kernels()
    virtual_kernels_identical = bool(
        len(control_kernels) == len(candidate_kernels)
        and all(torch.equal(a, b) for a, b in zip(control_kernels, candidate_kernels))
    )
    control_kernel_parameters = sum(int(value.numel()) for value in control_kernels)

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
        balanced_indices.setdefault(label, int(index))
        if len(balanced_indices) == num_classes:
            break
    if set(balanced_indices) != set(range(num_classes)):
        raise RuntimeError("Could not construct a five-class real-train cohort.")
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
        context="octave_conv_stage_a_train_only",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **dataloader_kwargs,
    )
    resource_batch_cpu = next(iter(loader))
    balanced_batch_cpu = default_collate(
        [dataset[index] for index in ordered_balanced_indices]
    )
    export = _export_diagnostics(
        stem=stem,
        images=resource_batch_cpu[0],
        output_dir=output_dir,
    )

    device = torch.device("cuda")
    images, labels, metadata = _move_batch(resource_batch_cpu, device=device)
    balanced_images, balanced_labels, balanced_metadata = _move_batch(
        balanced_batch_cpu,
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
    amp_gradients = summarize_octave_gradients(model)
    runtime_ratio = float(
        float(candidate_benchmark["median_seconds"])
        / max(float(control_benchmark["median_seconds"]), 1e-12)
    )

    model.train()
    model.zero_grad(set_to_none=True)
    fp32_count = int(args.fp32_batch_size)
    fp32_probe = images[:fp32_count].detach().clone().requires_grad_(True)
    logits_fp32, _ = _forward_logits(
        model,
        fp32_probe,
        {key: value[:fp32_count] for key, value in metadata.items()},
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    fp32_gradients = summarize_octave_gradients(model)
    input_gradient = fp32_probe.grad
    fp32_finite = bool(
        torch.isfinite(logits_fp32).all()
        and torch.isfinite(loss_fp32)
        and input_gradient is not None
        and torch.isfinite(input_gradient).all()
        and int(torch.count_nonzero(input_gradient)) > 0
    )
    activations = _activation_diagnostics(model, balanced_images)
    ablations = _path_ablation_diagnostics(
        model=model,
        images=balanced_images,
        metadata=balanced_metadata,
    )
    sensitivity = _balanced_input_sensitivity(
        model=model,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
    )

    contribution_cosines = activations["contribution_cosines"]
    cosine_noncollapse = all(
        bool(value["finite"])
        and float(value["maximum_absolute"]) < MAX_CONTRIBUTION_COSINE
        for value in contribution_cosines.values()
    )
    trace_shapes = {
        name: value["shape"] for name, value in activations["trace"].items()
    }
    expected_shapes = {
        "block1_high_output": [5, 28, 128, 128],
        "block1_low_output": [5, 4, 64, 64],
        "block2_high_output": [5, 56, 64, 64],
        "block2_low_output": [5, 8, 32, 32],
        "block3_high_output": [5, 256, 32, 32],
        "stem_output": [5, 256, 32, 32],
    }
    shapes_valid = all(trace_shapes.get(name) == shape for name, shape in expected_shapes.items())
    path_layouts = [block.octave.available_paths for block in stem.blocks]
    checks = {
        "locked_data_sha256": data_sha == LOCKED_DATA_SHA256,
        "locked_scratch_checkpoint_sha256": checkpoint_sha == LOCKED_SCRATCH_CHECKPOINT_SHA256,
        "locked_source_launcher_sha256": launcher_sha == LOCKED_SOURCE_LAUNCHER_SHA256,
        "source_protocol_scratch": bool(launcher_args.get("no_pretrained", False)),
        "candidate_pretrained_disabled": not bool(candidate_config.get("pretrained", False)),
        "train_split_only": bool(train_only_paths),
        "train_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "balanced_real_train_batch": [int(value) for value in balanced_labels.cpu().tolist()] == list(range(5)),
        "default_control_bit_identical": bool(
            control_state_sha == default_control_state_sha
            and default_control_logits_identical
            and _rng_equal(control_rng, default_control_rng)
        ),
        "candidate_same_seed_deterministic": bool(
            candidate_state_sha == repeated_state_sha
            and _rng_equal(candidate_rng, repeated_rng)
        ),
        "candidate_strict_state_roundtrip": candidate_state_sha == roundtrip_state_sha,
        "control_candidate_rng_paired": _rng_equal(control_rng, candidate_rng),
        "control_candidate_nonstem_bit_identical": nonstem_bit_identical,
        "control_candidate_parameter_count_equal": candidate_parameters == control_parameters,
        "control_candidate_stem_parameter_count_equal": candidate_stem_parameters == control_stem_parameters,
        "control_candidate_kernel_parameter_count_equal": stem.kernel_parameter_count == control_kernel_parameters,
        "virtual_vanilla_kernels_bit_identical": virtual_kernels_identical,
        "official_source_locked": bool(
            stem.source_commit == EXPECTED_SOURCE_COMMIT
            and stem.source_sha256 == EXPECTED_SOURCE_SHA256
        ),
        "locked_alpha": math.isclose(float(stem.alpha), LOCKED_ALPHA),
        "locked_path_layouts": path_layouts == [
            ("hh", "hl"),
            ("hh", "hl", "ll", "lh"),
            ("hh", "lh"),
        ],
        "stem_and_frequency_shapes": bool(shapes_valid),
        "exact_patch_token_count": int(getattr(model.patch_embed, "num_patches", -1)) == EXPECTED_PATCH_TOKENS,
        "fp32_logits_loss_input_gradient_finite": fp32_finite,
        "fp32_all_path_gradients": all(bool(value["passed"]) for value in fp32_gradients.values()),
        "amp_control_logits_loss_finite": bool(control_benchmark["logits_loss_finite"]),
        "amp_candidate_logits_loss_finite": bool(candidate_benchmark["logits_loss_finite"]),
        "amp_all_path_gradients": all(bool(value["passed"]) for value in amp_gradients.values()),
        "all_cross_frequency_paths_material": bool(ablations["all_paths_material"]),
        "path_and_energy_noncollapse": bool(cosine_noncollapse),
        "low_frequency_total_variation_lower": bool(
            activations["normalized_total_variation"]["low_below_high"]
        ),
        "balanced_class_input_logit_sensitivity": bool(sensitivity["all_classes_live"]),
        "onnx_cpu_equivalence": bool(
            float(export["maximum_absolute_error"]) <= MAX_ONNX_ERROR
            and all(bool(row["finite"]) for row in export["batch_results"])
        ),
        "onnx_dynamic_batch_1_2": [int(row["batch_size"]) for row in export["batch_results"]] == [1, 2],
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_octave_readiness(
        checks=checks,
        peak_vram_gib=float(candidate_benchmark["peak_vram_gib"]),
        runtime_ratio=runtime_ratio,
    )
    summary: Dict[str, object] = {
        "method": "octave_conv_parameter_matched_frequency_stem",
        "protocol_stage": "A_train_only_functional_resource_export_preflight",
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
            "official_source_commit": stem.source_commit,
            "official_source_file_sha256": stem.source_sha256,
        },
        "model": {
            "class_names": [str(value) for value in class_names],
            "control_parameters": int(control_parameters),
            "candidate_parameters": int(candidate_parameters),
            "control_stem_parameters": int(control_stem_parameters),
            "candidate_stem_parameters": int(candidate_stem_parameters),
            "control_kernel_parameters": int(control_kernel_parameters),
            "candidate_kernel_parameters": int(stem.kernel_parameter_count),
            "patch_tokens": int(getattr(model.patch_embed, "num_patches", -1)),
            "path_layouts": [list(value) for value in path_layouts],
            "control_state_sha256": control_state_sha,
            "default_control_state_sha256": default_control_state_sha,
            "candidate_state_sha256": candidate_state_sha,
            "repeated_candidate_state_sha256": repeated_state_sha,
            "roundtrip_state_sha256": roundtrip_state_sha,
            "control_rng": _rng_summary(control_rng),
            "candidate_rng": _rng_summary(candidate_rng),
            "nonstem_bit_identical": nonstem_bit_identical,
            "virtual_vanilla_kernels_bit_identical": virtual_kernels_identical,
        },
        "data": {
            "train_rows": len(dataset),
            "train_root": str(train_root),
            "resource_batch_size": int(images.size(0)),
            "resource_labels": [int(value) for value in labels.cpu().tolist()],
            "resource_paths": selected_paths,
            "balanced_indices": ordered_balanced_indices,
            "balanced_labels": [int(value) for value in balanced_labels.cpu().tolist()],
            "balanced_paths": balanced_paths,
            "dataloader": dataloader_summary,
        },
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "logits_shape": [int(value) for value in logits_fp32.shape],
            "gradients": fp32_gradients,
        },
        "cuda_amp": {
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_gib": float(torch.cuda.get_device_properties(device).total_memory / (1024**3)),
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "candidate_control_runtime_ratio": runtime_ratio,
        },
        "activations": activations,
        "cross_frequency_ablations": ablations,
        "balanced_input_sensitivity": sensitivity,
        "export": export,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
                "role": "octave_conv_stem_onnx",
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
