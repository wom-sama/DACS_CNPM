from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
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
from trkh.models.model import (
    HybridConvStem,
    InstanceBatchNorm2d,
    create_model,
    load_model_state,
)
from trkh.tools.audit_moga_tokenizer_readiness import (
    _amp_forward_backward_benchmark,
    _move_batch,
)
from trkh.tools.audit_visual_contrast_attention_readiness import (
    _build_train_only_dataset,
    _forward_logits,
    _prepare_output_dir,
    _state_sha256,
)


EXPECTED_TRAIN_ROWS = 9215
EXPECTED_STEM_SHAPE = (32, 256, 32, 32)
EXPECTED_INSTANCE_CHANNELS = 16
MAX_EQUATION_ERROR = 1e-6
MAX_STANDARDIZED_MEAN = 1e-5
MAX_STANDARDIZED_VARIANCE_ERROR = 2e-4
MAX_STYLE_ERROR_RATIO = 0.05
MAX_ONNX_ERROR = 5e-5
MAX_RUNTIME_RATIO = 1.30
MAX_PEAK_VRAM_GIB = 7.5
BENCHMARK_REPEATS = 3

LOCKED_HASHES = {
    "checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "resolved_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "launcher_args": "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    "data_yaml": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "protocol": "7e329a7cf1bceea3d2dd8958fbdb1d5d34dd422b423c740032322511d167e576",
    "paper": "411fd8d8afd08e0d57f295821919f286c7206e7fe1ce982cedc0f06045de1b36",
    "official_modules": "1578ca15ba7fbe6349d3e590c684c16236a2c0ccabf737a4762f6fed8a6b499c",
    "official_resnet": "32af52f5f638bb0528b537b2e352da4ed65c02ccbf39b84fc0c5e06b35edd0f9",
    "official_license": "4e2e849faed41630d067a8789edd12e4da452e8cd52b326c44e54b911b064532",
    "current_best_command": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "current_best_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
EXPECTED_OFFICIAL_COMMIT = "d1673389b36c1180cf9bc35ea8260d84046da915"
EXPECTED_OFFICIAL_TREE = "e113673c2aa64dc761a67f1585fd168a95d6ab4b"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only engineering readiness audit for shallow IBN-a TRKH."
    )
    keeper = Path(
        "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
        "bboxprior_120b_2e_20260701"
    )
    parser.add_argument("--checkpoint", type=Path, default=keeper / "checkpoints/best.pt")
    parser.add_argument("--resolved-config", type=Path, default=keeper / "resolved_config.json")
    parser.add_argument("--source-launcher-args", type=Path, default=keeper / "launcher_args.json")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/TRKH_5CLASS_IBN_A_SHALLOW_STEM_PROTOCOL_20260720.md"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\ibn_net_eccv2018.pdf"),
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\repositories\IBN-Net"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _git_value(repo: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", expression],
        text=True,
        encoding="utf-8",
    ).strip()


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_repo).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "resolved_config": Path(args.resolved_config).resolve(),
        "launcher_args": Path(args.source_launcher_args).resolve(),
        "data_yaml": Path(args.data).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_modules": official / "ibnnet/modules.py",
        "official_resnet": official / "ibnnet/resnet_ibn.py",
        "official_license": official / "LICENSE",
        "current_best_command": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "current_best_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _model_configs(source: Mapping[str, object]) -> tuple[Dict[str, object], Dict[str, object]]:
    control = dict(source)
    control.update(
        {
            "stem_architecture": "conv_pool",
            "stem_normalization": "batch",
            "pretrained": False,
        }
    )
    candidate = dict(control)
    candidate["stem_normalization"] = "ibn_a_first"
    return control, candidate


def _config_differences(
    left: Mapping[str, object], right: Mapping[str, object]
) -> list[str]:
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )


def _state_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return bool(
        list(left_state) == list(right_state)
        and all(torch.equal(value, right_state[name]) for name, value in left_state.items())
    )


def _oracle() -> Dict[str, object]:
    set_seed(20260720)
    layer = InstanceBatchNorm2d(32, ratio=0.5).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(0.7, 1.3, 32))
        layer.bias.copy_(torch.linspace(-0.2, 0.2, 32))
        layer.running_mean.copy_(torch.linspace(-0.4, 0.4, 32))
        layer.running_var.copy_(torch.linspace(0.6, 1.4, 32))
    values = torch.randn(4, 32, 17, 13)
    actual = layer(values)
    expected_instance = F.instance_norm(
        values[:, :16].contiguous(),
        weight=layer.weight[:16],
        bias=layer.bias[:16],
        use_input_stats=True,
        momentum=0.0,
        eps=layer.eps,
    )
    expected_batch = F.batch_norm(
        values[:, 16:],
        layer.running_mean[16:],
        layer.running_var[16:],
        layer.weight[16:],
        layer.bias[16:],
        training=False,
        momentum=layer.momentum,
        eps=layer.eps,
    )
    expected = torch.cat((expected_instance, expected_batch), dim=1)
    standardized = (
        actual[:, :16] - layer.bias[:16].view(1, -1, 1, 1)
    ) / layer.weight[:16].view(1, -1, 1, 1)

    control = nn.BatchNorm2d(32).eval()
    candidate = InstanceBatchNorm2d(32).eval()
    candidate.load_state_dict(control.state_dict(), strict=True)
    base = torch.randn(4, 32, 19, 15)
    scale = torch.tensor([0.55, 0.85, 1.25, 1.80]).view(4, 1, 1, 1)
    shift = torch.tensor([-0.7, -0.2, 0.35, 1.1]).view(4, 1, 1, 1)
    transformed = base * scale + shift
    base_control = control(base)
    transformed_control = control(transformed)
    base_candidate = candidate(base)
    transformed_candidate = candidate(transformed)
    control_error = float(
        (transformed_control[:, :16] - base_control[:, :16]).abs().mean().item()
    )
    candidate_error = float(
        (transformed_candidate[:, :16] - base_candidate[:, :16]).abs().mean().item()
    )
    style_ratio = candidate_error / max(control_error, 1e-12)
    return {
        "equation_max_abs_error": float((actual - expected).abs().max().item()),
        "standardized_mean_max_abs": float(
            standardized.mean(dim=(-2, -1)).abs().max().item()
        ),
        "standardized_variance_max_error": float(
            (
                standardized.var(dim=(-2, -1), unbiased=False)
                - torch.ones(1, 16)
            )
            .abs()
            .max()
            .item()
        ),
        "control_style_error": control_error,
        "candidate_style_error": candidate_error,
        "style_error_ratio": float(style_ratio),
        "batch_half_bit_identical": bool(
            torch.equal(candidate(base)[:, 16:], control(base)[:, 16:])
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _export_norm(layer: InstanceBatchNorm2d, output_dir: Path) -> Dict[str, object]:
    import onnx
    import onnxruntime as ort

    export_path = output_dir / "ibn_a_first_norm.onnx"
    probe = torch.linspace(-1.0, 1.0, 2 * 32 * 16 * 16).reshape(2, 32, 16, 16)
    export_layer = InstanceBatchNorm2d(32, ratio=0.5).eval()
    export_layer.load_state_dict(layer.state_dict(), strict=True)
    with torch.inference_mode():
        expected = export_layer(probe).cpu().numpy()
    torch.onnx.export(
        export_layer,
        probe,
        export_path,
        input_names=["features"],
        output_names=["normalized"],
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(str(export_path))
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    observed = session.run(None, {"features": probe.numpy()})[0]
    maximum_error = float(np.max(np.abs(expected - observed)))
    custom_operators = [
        operator
        for operator in operators
        if operator.startswith("ATen") or operator.startswith("Python")
    ]
    return {
        "path": str(export_path.resolve()),
        "sha256": _sha256(export_path),
        "operators": operators,
        "custom_operators": custom_operators,
        "maximum_abs_error": maximum_error,
        "passed": bool(not custom_operators and maximum_error <= MAX_ONNX_ERROR),
    }


def _gradient_summary(model: nn.Module) -> Dict[str, object]:
    norm = model.stem.blocks[0].block.norm
    if not isinstance(norm, InstanceBatchNorm2d) or norm.weight.grad is None:
        return {"passed": False, "reason": "missing IBN-a affine gradient"}
    patch = dict(model.named_parameters()).get("patch_embed.proj.weight")
    patch_gradient = patch.grad if patch is not None else None

    def summarize(value: Optional[Tensor]) -> Dict[str, object]:
        return {
            "present": value is not None,
            "finite": bool(value is not None and torch.isfinite(value).all()),
            "nonzero": int(torch.count_nonzero(value).item()) if value is not None else 0,
        }

    instance = summarize(norm.weight.grad[: norm.instance_channels])
    batch = summarize(norm.weight.grad[norm.instance_channels :])
    patch_summary = summarize(patch_gradient)
    passed = all(
        item["present"] and item["finite"] and int(item["nonzero"]) > 0
        for item in (instance, batch, patch_summary)
    )
    return {
        "instance_half": instance,
        "batch_half": batch,
        "patch_projection": patch_summary,
        "passed": bool(passed),
    }


def assess_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    runtime_ratio: float,
) -> Dict[str, object]:
    resolved = {str(name): bool(value) for name, value in checks.items()}
    resolved["peak_vram_within_budget"] = bool(
        math.isfinite(peak_vram_gib) and peak_vram_gib <= MAX_PEAK_VRAM_GIB
    )
    resolved["runtime_ratio_within_budget"] = bool(
        math.isfinite(runtime_ratio) and runtime_ratio <= MAX_RUNTIME_RATIO
    )
    failed = [name for name, passed in resolved.items() if not passed]
    return {
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": resolved,
        "failed_checks": failed,
        "thresholds": {
            "maximum_equation_error": MAX_EQUATION_ERROR,
            "maximum_standardized_mean": MAX_STANDARDIZED_MEAN,
            "maximum_standardized_variance_error": MAX_STANDARDIZED_VARIANCE_ERROR,
            "maximum_style_error_ratio": MAX_STYLE_ERROR_RATIO,
            "maximum_onnx_error": MAX_ONNX_ERROR,
            "maximum_runtime_ratio": MAX_RUNTIME_RATIO,
            "maximum_peak_vram_gib": MAX_PEAK_VRAM_GIB,
        },
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if str(args.device).strip().lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal shallow IBN-a Stage A requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("Locked batch sizes are AMP=32 and FP32=2.")
    if int(args.num_workers) != 4 or int(args.benchmark_repeats) != BENCHMARK_REPEATS:
        raise ValueError("Locked worker/repeat settings are workers=4 and repeats=3.")

    output_dir = _prepare_output_dir(Path(args.output_dir))
    paths = _source_paths(args)
    observed_hashes = {name: _sha256(path) for name, path in paths.items()}
    hash_checks = {
        name: observed_hashes.get(name) == expected
        for name, expected in LOCKED_HASHES.items()
    }
    official_repo = Path(args.official_repo).resolve()
    official_commit = _git_value(official_repo, "HEAD")
    official_tree = _git_value(official_repo, "HEAD^{tree}")
    official_git_locked = bool(
        official_commit == EXPECTED_OFFICIAL_COMMIT
        and official_tree == EXPECTED_OFFICIAL_TREE
    )

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    model_state = checkpoint.get("model_state")
    if (
        not isinstance(source_model_config, Mapping)
        or not isinstance(class_names, list)
        or not isinstance(model_state, Mapping)
    ):
        raise ValueError("Keeper lacks model_config, class_names, or model_state.")
    if len(class_names) != 5:
        raise ValueError(f"Expected five classes, got {len(class_names)}.")
    resolved_config = _load_json(paths["resolved_config"])
    control_config, candidate_config = _model_configs(source_model_config)
    differences = _config_differences(control_config, candidate_config)

    set_seed(int(args.seed))
    control = create_model(num_classes=5, model_config=control_config)
    set_seed(int(args.seed))
    candidate = create_model(num_classes=5, model_config=candidate_config)
    initial_state_equal = _state_equal(control, candidate)
    initial_control_sha = _state_sha256(control)
    initial_candidate_sha = _state_sha256(candidate)
    parameter_count_control = sum(parameter.numel() for parameter in control.parameters())
    parameter_count_candidate = sum(parameter.numel() for parameter in candidate.parameters())

    default_config = dict(control_config)
    default_config.pop("stem_normalization", None)
    set_seed(int(args.seed))
    default_model = create_model(num_classes=5, model_config=default_config).eval()
    set_seed(int(args.seed))
    explicit_model = create_model(num_classes=5, model_config=control_config).eval()
    schema_probe = torch.linspace(-1.0, 1.0, 3 * 256 * 256).reshape(1, 3, 256, 256)
    with torch.inference_mode():
        default_logits, _ = _forward_logits(default_model, schema_probe, {})
        explicit_logits, _ = _forward_logits(explicit_model, schema_probe, {})
    default_state_equal = _state_equal(default_model, explicit_model)
    default_logits_equal = bool(torch.equal(default_logits, explicit_logits))
    del default_model, explicit_model, default_logits, explicit_logits, schema_probe

    control_load = load_model_state(control, dict(model_state), strict=True)
    candidate_load = load_model_state(candidate, dict(model_state), strict=True)
    loaded_state_equal = _state_equal(control, candidate)
    loaded_control_sha = _state_sha256(control)
    loaded_candidate_sha = _state_sha256(candidate)

    if not isinstance(control.stem, HybridConvStem) or not isinstance(
        candidate.stem, HybridConvStem
    ):
        raise TypeError("Locked candidate requires the legacy HybridConvStem.")
    control_norms = [block.block.norm for block in control.stem.blocks]
    candidate_norms = [block.block.norm for block in candidate.stem.blocks]
    placement = {
        "control_types": [type(norm).__name__ for norm in control_norms],
        "candidate_types": [type(norm).__name__ for norm in candidate_norms],
        "candidate_instance_channels": int(
            candidate_norms[0].instance_channels
            if isinstance(candidate_norms[0], InstanceBatchNorm2d)
            else -1
        ),
    }
    placement["passed"] = bool(
        all(type(norm) is nn.BatchNorm2d for norm in control_norms)
        and isinstance(candidate_norms[0], InstanceBatchNorm2d)
        and all(type(norm) is nn.BatchNorm2d for norm in candidate_norms[1:])
        and placement["candidate_instance_channels"] == EXPECTED_INSTANCE_CHANNELS
    )

    oracle = _oracle()
    export = _export_norm(candidate_norms[0], output_dir)

    dataset = _build_train_only_dataset(
        data_yaml=paths["data_yaml"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    data_spec = load_data_spec(paths["data_yaml"])
    train_root = Path(data_spec.split_images_dir("train")).resolve()
    balanced_indices: Dict[int, int] = {}
    for index, sample in enumerate(dataset.samples):
        balanced_indices.setdefault(int(sample.primary_label), int(index))
        if len(balanced_indices) == 5:
            break
    if set(balanced_indices) != set(range(5)):
        raise RuntimeError("Could not create a deterministic five-class train cohort.")
    ordered_indices = [balanced_indices[index] for index in range(5)]
    balanced_paths = [
        str(dataset.samples[index].image_path.resolve()) for index in ordered_indices
    ]
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="ibn_a_shallow_stage_a_train_only",
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
    balanced_batch_cpu = default_collate([dataset[index] for index in ordered_indices])
    resource_paths = [
        str(dataset.samples[index].image_path.resolve())
        for index in range(int(args.batch_size))
    ]
    train_only_paths = bool(
        all(Path(path).is_relative_to(train_root) for path in resource_paths + balanced_paths)
    )

    device = torch.device("cuda")
    images, labels, metadata = _move_batch(resource_batch_cpu, device=device)
    balanced_images, _, balanced_metadata = _move_batch(
        balanced_batch_cpu, device=device
    )
    control = control.to(device).eval()
    candidate = candidate.to(device).eval()
    with torch.inference_mode():
        pre_norm = candidate.stem.blocks[0].block.conv(images)
        control_normalized = control.stem.blocks[0].block.norm(pre_norm)
        candidate_normalized = candidate.stem.blocks[0].block.norm(pre_norm)
        direct_instance = F.instance_norm(
            pre_norm[:, :16].contiguous(),
            weight=candidate_norms[0].weight[:16],
            bias=candidate_norms[0].bias[:16],
            use_input_stats=True,
            momentum=0.0,
            eps=candidate_norms[0].eps,
        )
        direct_batch = F.batch_norm(
            pre_norm[:, 16:],
            candidate_norms[0].running_mean[16:],
            candidate_norms[0].running_var[16:],
            candidate_norms[0].weight[16:],
            candidate_norms[0].bias[16:],
            training=False,
            momentum=candidate_norms[0].momentum,
            eps=candidate_norms[0].eps,
        )
        direct_real = torch.cat((direct_instance, direct_batch), dim=1)
        scale = torch.linspace(0.65, 1.35, images.size(0), device=device).view(-1, 1, 1, 1)
        shift = torch.linspace(-0.45, 0.45, images.size(0), device=device).view(-1, 1, 1, 1)
        shifted_pre_norm = pre_norm * scale + shift
        shifted_control = control.stem.blocks[0].block.norm(shifted_pre_norm)
        shifted_candidate = candidate.stem.blocks[0].block.norm(shifted_pre_norm)
        real_control_style_error = float(
            (shifted_control[:, :16] - control_normalized[:, :16]).abs().mean().item()
        )
        real_candidate_style_error = float(
            (shifted_candidate[:, :16] - candidate_normalized[:, :16]).abs().mean().item()
        )
        real_style_ratio = real_candidate_style_error / max(real_control_style_error, 1e-12)
        stem_output = candidate.stem(images)
        balanced_logits, _ = _forward_logits(candidate, balanced_images, balanced_metadata)
    real_diagnostics = {
        "direct_equation_max_abs_error": float(
            (candidate_normalized - direct_real).abs().max().item()
        ),
        "batch_half_bit_identical": bool(
            torch.equal(candidate_normalized[:, 16:], control_normalized[:, 16:])
        ),
        "control_style_error": real_control_style_error,
        "candidate_style_error": real_candidate_style_error,
        "style_error_ratio": float(real_style_ratio),
        "stem_shape": [int(value) for value in stem_output.shape],
        "stem_finite": bool(torch.isfinite(stem_output).all()),
        "balanced_logits_shape": [int(value) for value in balanced_logits.shape],
        "balanced_logits_finite": bool(torch.isfinite(balanced_logits).all()),
    }

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    fp32_logits, _ = _forward_logits(
        candidate,
        images[: int(args.fp32_batch_size)],
        {key: value[: int(args.fp32_batch_size)] for key, value in metadata.items()},
    )
    fp32_loss = F.cross_entropy(fp32_logits, labels[: int(args.fp32_batch_size)])
    fp32_loss.backward()
    fp32_gradient = _gradient_summary(candidate)
    fp32_peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    fp32_summary = {
        "batch_size": int(args.fp32_batch_size),
        "logits_finite": bool(torch.isfinite(fp32_logits).all()),
        "loss": float(fp32_loss.detach().item()),
        "loss_finite": bool(torch.isfinite(fp32_loss)),
        "gradient": fp32_gradient,
        "peak_vram_gib": fp32_peak,
    }

    load_model_state(candidate, dict(model_state), strict=True)
    candidate.zero_grad(set_to_none=True)
    control = control.train()
    control_benchmark = _amp_forward_backward_benchmark(
        model=control,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    control = control.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    candidate = candidate.train()
    candidate_benchmark = _amp_forward_backward_benchmark(
        model=candidate,
        images=images,
        labels=labels,
        metadata=metadata,
        repeats=int(args.benchmark_repeats),
        device=device,
    )
    amp_gradient = _gradient_summary(candidate)
    runtime_ratio = float(candidate_benchmark["median_seconds"]) / max(
        float(control_benchmark["median_seconds"]), 1e-12
    )
    peak_vram_gib = max(
        fp32_peak,
        float(control_benchmark["peak_vram_gib"]),
        float(candidate_benchmark["peak_vram_gib"]),
    )

    load_results_empty = bool(
        tuple(control_load) == ([], []) and tuple(candidate_load) == ([], [])
    )
    checks = {
        "all_locked_hashes": all(hash_checks.values()),
        "official_git_locked": official_git_locked,
        "only_config_difference": differences == ["stem_normalization"],
        "default_state_bit_identical": default_state_equal,
        "default_logits_bit_identical": default_logits_equal,
        "initial_state_bit_identical": initial_state_equal,
        "initial_state_hash_identical": initial_control_sha == initial_candidate_sha,
        "parameter_count_identical": parameter_count_control == parameter_count_candidate,
        "strict_keeper_load": load_results_empty,
        "loaded_state_bit_identical": loaded_state_equal,
        "loaded_state_hash_identical": loaded_control_sha == loaded_candidate_sha,
        "placement_exact": bool(placement["passed"]),
        "oracle_equation": float(oracle["equation_max_abs_error"]) <= MAX_EQUATION_ERROR,
        "oracle_standardized_mean": float(oracle["standardized_mean_max_abs"])
        <= MAX_STANDARDIZED_MEAN,
        "oracle_standardized_variance": float(
            oracle["standardized_variance_max_error"]
        )
        <= MAX_STANDARDIZED_VARIANCE_ERROR,
        "oracle_style_invariance": float(oracle["style_error_ratio"])
        <= MAX_STYLE_ERROR_RATIO,
        "oracle_batch_half_preserved": bool(oracle["batch_half_bit_identical"]),
        "real_equation": float(real_diagnostics["direct_equation_max_abs_error"])
        <= MAX_EQUATION_ERROR,
        "real_style_invariance": float(real_diagnostics["style_error_ratio"])
        <= MAX_STYLE_ERROR_RATIO,
        "real_batch_half_preserved": bool(real_diagnostics["batch_half_bit_identical"]),
        "real_shapes_finite": bool(
            tuple(real_diagnostics["stem_shape"]) == EXPECTED_STEM_SHAPE
            and real_diagnostics["stem_finite"]
            and real_diagnostics["balanced_logits_shape"] == [5, 5]
            and real_diagnostics["balanced_logits_finite"]
        ),
        "fp32_forward_backward": bool(
            fp32_summary["logits_finite"]
            and fp32_summary["loss_finite"]
            and fp32_gradient["passed"]
        ),
        "amp_forward_backward": bool(
            candidate_benchmark["logits_loss_finite"] and amp_gradient["passed"]
        ),
        "onnx_export": bool(export["passed"]),
        "dataset_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "train_only_paths": train_only_paths,
        "five_class_cohort": len(balanced_paths) == 5,
    }
    gate = assess_readiness(
        checks=checks,
        peak_vram_gib=peak_vram_gib,
        runtime_ratio=runtime_ratio,
    )
    current_head = _git_value(Path.cwd(), "HEAD")
    summary = {
        "method": "ibn_a_shallow_stem",
        "status": "passed" if gate["smoke_permission"] else "rejected",
        "repository_head": current_head,
        "sources": {
            "paths": {name: str(path) for name, path in paths.items()},
            "observed_sha256": observed_hashes,
            "expected_sha256": LOCKED_HASHES,
            "hash_checks": hash_checks,
            "official_commit": official_commit,
            "official_tree": official_tree,
            "validation_loaded": False,
            "test_loaded": False,
            "raw_dataset_modified": False,
        },
        "configuration": {
            "differences": differences,
            "control_stem_normalization": control_config["stem_normalization"],
            "candidate_stem_normalization": candidate_config["stem_normalization"],
            "seed": int(args.seed),
        },
        "state_schema": {
            "initial_control_sha256": initial_control_sha,
            "initial_candidate_sha256": initial_candidate_sha,
            "loaded_control_sha256": loaded_control_sha,
            "loaded_candidate_sha256": loaded_candidate_sha,
            "initial_state_equal": initial_state_equal,
            "loaded_state_equal": loaded_state_equal,
            "parameter_count_control": parameter_count_control,
            "parameter_count_candidate": parameter_count_candidate,
            "strict_load_results_empty": load_results_empty,
            "default_state_equal": default_state_equal,
            "default_logits_equal": default_logits_equal,
        },
        "placement": placement,
        "oracle": oracle,
        "real_train_diagnostics": real_diagnostics,
        "selected_train_paths": {
            "train_root": str(train_root),
            "resource": resource_paths,
            "balanced": balanced_paths,
            "train_only": train_only_paths,
        },
        "dataloader": dataloader_summary,
        "fp32": fp32_summary,
        "cuda_amp": {
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "candidate_gradient": amp_gradient,
            "runtime_ratio": runtime_ratio,
            "peak_vram_gib": peak_vram_gib,
        },
        "export": export,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
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
                "role": "ibn_a_norm_onnx",
            },
        ],
        "raw_dataset_modified": False,
        "validation_used": False,
        "test_used": False,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not bool(summary["gate"]["smoke_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
