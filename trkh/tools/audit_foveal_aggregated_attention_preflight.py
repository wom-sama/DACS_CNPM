from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset, build_eval_transform
from trkh.models.foveal_aggregated_attention import FovealAggregatedAttention
from trkh.models.model import classification_logits_from_features, create_model
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _failed_export,
    _onnx_compare,
)


LOCKED_PROTOCOL_SHA256 = "1676ef25fde2d90ca0786730d0dd91ab9bba551976111b19912da452759bfec8"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_DECLARATION_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_RESOLVED_CONFIG_SHA256 = "c549d911bfbe46c0a63ce06c10e4b9455982753469ba8905cdf4533e88e84a97"
LOCKED_OFFICIAL_ATTENTION_SHA256 = "f70bc20818c9b0456d6eaaf58669561f0ede8fa5aba3c3a557cb5094557d86fe"
LOCKED_OFFICIAL_LICENSE_SHA256 = "68341bcf5aea46bf8f6c63f5c382e74dc960feaa31cd391e01a4d764d6121824"
LOCKED_PAPER_SHA256 = "0d1a5d07b747eb7c02ed66a840f2f0242252697d883077325862509ad28e1c5e"
LOCKED_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "c8a99743b60ac94ac8d2bf66ffe164a440dcfe21"
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_SOURCES = 6_452
EXPECTED_HOLDOUT_SOURCES = 1_612
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
MAX_ADDED_PARAMETERS = 100_000
MAX_RUNTIME_RATIO = 1.35
MAX_VRAM_GIB = 7.5
MAX_VRAM_RATIO = 1.25


FAA_GRADIENT_MARKERS = {
    "qkv": ("blocks.0.attn.qkv.",),
    "projection": ("blocks.0.attn.proj.",),
    "temperature": ("blocks.0.attn.temperature",),
    "query_embedding": ("blocks.0.attn.query_embedding",),
    "pooled_projection": ("blocks.0.attn.sr.",),
    "pooled_normalization": ("blocks.0.attn.pool_norm.",),
    "continuous_position_bias": (
        "blocks.0.attn.cpb_fc1.",
        "blocks.0.attn.cpb_fc2.",
    ),
    "local_position_bias": ("blocks.0.attn.relative_position_bias_local",),
    "local_tokens": (
        "blocks.0.attn.learnable_tokens",
        "blocks.0.attn.learnable_bias",
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only functional/resource preflight for block-1 "
            "Foveal Aggregated Attention. Validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
    )
    parser.add_argument(
        "--fold-data",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_FOVEAL_AGGREGATED_ATTENTION_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\transnext-cvpr2024"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\shi2024_transnext.pdf"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--timed-iterations", type=int, default=5)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _prepare_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Preflight output must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        encoding="utf-8",
    ).strip()


def _tracked_worktree_clean() -> bool:
    output = _git_value("status", "--short", "--untracked-files=no")
    return not bool(output.strip())


def _official_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()


def _candidate_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "pretrained": False,
            "early_token_mask_keep_rate": 1.0,
            "gated_relative_position_attention": False,
            "visual_contrast_attention": False,
            "cross_covariance_attention": False,
            "dynamic_graph_mixer": False,
            "soft_moe_patch_adapter": False,
            "locally_enhanced_ffn": False,
            "concurrent_local_global_coupling": False,
            "foveal_aggregated_attention": True,
            "foveal_aggregated_attention_layers": "1",
            "foveal_aggregated_attention_window_size": 3,
            "foveal_aggregated_attention_pool_size": 4,
        }
    )
    return config


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = _candidate_config(source)
    config["foveal_aggregated_attention"] = False
    return config


def _build_dataset(
    *,
    fold_data: Path,
    model_config: Mapping[str, object],
    resolved_config: Mapping[str, object],
) -> MangoYOLOCropDataset:
    data_spec = load_data_spec(
        fold_data,
        class_name_mode="raw",
        expected_num_classes=5,
    )
    augmentation = resolved_config.get("augmentation_config")
    if not isinstance(augmentation, Mapping):
        raise ValueError("Resolved config lacks augmentation_config.")
    transform = build_eval_transform(
        image_size=int(model_config.get("image_size", 256)),
        resize_mode=str(augmentation.get("resize_mode", "pad")),
        illumination_normalization=bool(
            augmentation.get("illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            augmentation.get("illumination_normalization_strength", 0.0)
        ),
        foreground_crop_mode=str(augmentation.get("foreground_crop_mode", "none")),
        foreground_crop_margin_ratio=float(
            augmentation.get("foreground_crop_margin_ratio", 0.08)
        ),
        foreground_crop_min_mask_area_ratio=float(
            augmentation.get("foreground_crop_min_mask_area_ratio", 0.03)
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation.get("foreground_crop_max_mask_area_ratio", 0.92)
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation.get("foreground_crop_max_crop_area_ratio", 0.98)
        ),
        background_suppression_mode=str(
            augmentation.get("background_suppression_mode", "none")
        ),
        background_suppression_margin=float(
            augmentation.get("background_suppression_margin", 0.08)
        ),
        background_suppression_blur_radius=float(
            augmentation.get("background_suppression_blur_radius", 7.0)
        ),
        mean=model_config.get("input_mean", (0.485, 0.456, 0.406)),
        std=model_config.get("input_std", (0.229, 0.224, 0.225)),
    )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=transform,
        crop_margin_ratio=float(augmentation.get("crop_margin_ratio", 0.05)),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
        class_aware_augmentation=False,
        class_crop_margin_scale_threshold=float(
            augmentation.get("class_crop_margin_scale_threshold", 1.5)
        ),
        class_crop_margin_max_ratio=float(
            augmentation.get("class_crop_margin_max_ratio", 0.16)
        ),
        classification_source_context=False,
    )


def _unpack_batch(batch: object) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
        raise TypeError("Expected a two- or three-item classification batch.")
    images, labels = batch[0], batch[1]
    metadata = batch[2] if len(batch) == 3 and isinstance(batch[2], Mapping) else {}
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise TypeError("Classification batch images/labels must be tensors.")
    return images, labels, {
        str(key): value for key, value in metadata.items() if torch.is_tensor(value)
    }


def _forward(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    return_attention: bool = False,
    return_trace: bool = False,
) -> Tuple[Tensor, Mapping[str, object]]:
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    features = model.forward_features(
        images,
        bbox_token_prior=bbox,
        image_valid_mask=image_mask,
        return_attention=return_attention,
        return_trace=return_trace,
    )
    if torch.is_tensor(bbox):
        features["bbox"] = bbox
    return classification_logits_from_features(model, features), features


def _common_state_summary(control: nn.Module, candidate: nn.Module) -> Dict[str, object]:
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    common = sorted(set(control_state).intersection(candidate_state))
    differing = [
        key for key in common if not torch.equal(control_state[key], candidate_state[key])
    ]
    candidate_only = sorted(set(candidate_state).difference(control_state))
    return {
        "common_keys": len(common),
        "differing_common_keys": differing,
        "candidate_only_keys": candidate_only,
        "all_common_bit_exact": not differing,
    }


def _gradient_summary(model: nn.Module) -> Dict[str, Dict[str, object]]:
    parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in FAA_GRADIENT_MARKERS.items():
        selected = {
            name: value
            for name, value in parameters.items()
            if any(marker in name for marker in markers)
        }
        gradients = [value.grad for value in selected.values()]
        present = [value for value in gradients if value is not None]
        finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
        nonzero = sum(int(torch.count_nonzero(value).item()) for value in present)
        summary[family] = {
            "parameter_tensors": len(selected),
            "gradient_tensors": len(present),
            "finite": finite,
            "nonzero_elements": nonzero,
            "passed": bool(selected and len(present) == len(selected) and finite and nonzero > 0),
        }
    return summary


def _official_equation_replay(official_attention_path: Path) -> Dict[str, object]:
    spec = importlib.util.spec_from_file_location(
        "locked_transnext_attention_native_preflight",
        official_attention_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import locked official attention source.")
    official_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_module)
    torch.manual_seed(41)
    candidate = FovealAggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    ).eval()
    official = official_module.AggregatedAttention(
        dim=32,
        input_resolution=(4, 4),
        num_heads=4,
        window_size=3,
        fixed_pool_size=2,
    ).eval()
    dim = candidate.dim
    with torch.no_grad():
        official.q.weight.copy_(candidate.qkv.weight[:dim])
        official.q.bias.copy_(candidate.qkv.bias[:dim])
        official.kv.weight.copy_(candidate.qkv.weight[dim:])
        official.kv.bias.copy_(candidate.qkv.bias[dim:])
        official.proj.load_state_dict(candidate.proj.state_dict())
        official.temperature.copy_(candidate.temperature)
        official.query_embedding.copy_(candidate.query_embedding)
        official.sr.load_state_dict(candidate.sr.state_dict())
        official.norm.load_state_dict(candidate.pool_norm.state_dict())
        official.cpb_fc1.load_state_dict(candidate.cpb_fc1.state_dict())
        official.cpb_fc2.load_state_dict(candidate.cpb_fc2.state_dict())
        official.relative_pos_bias_local.copy_(candidate.relative_position_bias_local)
        official.learnable_tokens.copy_(candidate.learnable_tokens)
        official.learnable_bias.copy_(candidate.learnable_bias)
    torch.manual_seed(19)
    candidate_input = torch.randn(2, 16, 32, requires_grad=True)
    official_input = candidate_input.detach().clone().requires_grad_(True)
    candidate_output = candidate(
        candidate_input,
        grid_size=(4, 4),
        prefix_count=0,
    )
    official_output = official(
        official_input,
        4,
        4,
        candidate.relative_position_index,
        candidate.relative_coords_table,
    )
    candidate_output.square().mean().backward()
    official_output.square().mean().backward()
    output_error = float((candidate_output - official_output).abs().amax().item())
    gradient_error = float(
        (candidate_input.grad - official_input.grad).abs().amax().item()
    )
    return {
        "output_maximum_absolute_error": output_error,
        "input_gradient_maximum_absolute_error": gradient_error,
        "passed": output_error <= 1e-6 and gradient_error <= 1e-6,
    }


def _benchmark(
    *,
    config: Mapping[str, object],
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    seed: int,
    warmup_iterations: int,
    timed_iterations: int,
) -> Dict[str, object]:
    set_seed(seed)
    model = create_model(num_classes=5, model_config=config).to(images.device).train()

    def iteration() -> Tuple[Tensor, Tensor]:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = _forward(model, images, metadata)
            loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        return logits, loss

    for _ in range(int(warmup_iterations)):
        iteration()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(images.device)
    elapsed = []
    logits = loss = None
    for _ in range(int(timed_iterations)):
        torch.cuda.synchronize(images.device)
        started = time.perf_counter()
        logits, loss = iteration()
        torch.cuda.synchronize(images.device)
        elapsed.append(float(time.perf_counter() - started))
    if logits is None or loss is None:
        raise RuntimeError("FAA benchmark did not execute a timed iteration.")
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(images.device) / (1024**3)),
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


class _FullExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.cpu().eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.model.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        return classification_logits_from_features(self.model, features)


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAA preflight requires CUDA.")
    if int(args.batch_size) != 32 or int(args.fp32_batch_size) != 2:
        raise ValueError("FAA preflight is locked to batch32 and FP32 batch2.")
    output_dir = _prepare_output(Path(args.output_dir))
    paths = {
        "resolved_config": Path(args.resolved_config).resolve(),
        "fold_data": Path(args.fold_data).resolve(),
        "fold_summary": Path(args.fold_summary).resolve(),
        "raw_data": Path(args.raw_data).resolve(),
        "declaration": Path(args.declaration).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "official_attention": (
            Path(args.official_root).resolve() / "classification" / "attention_native.py"
        ),
        "official_license": Path(args.official_root).resolve() / "LICENSE",
        "paper": Path(args.paper).resolve(),
        "current_commands": Path("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt").resolve(),
        "command_history": Path("docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt").resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    resolved_config = _load_json(paths["resolved_config"])
    source_model_config = resolved_config.get("model_config")
    if not isinstance(source_model_config, Mapping):
        raise ValueError("Resolved config lacks model_config.")
    fold_summary = _load_json(paths["fold_summary"])
    candidate_config = _candidate_config(source_model_config)
    control_config = _control_config(source_model_config)

    set_seed(int(args.seed))
    control = create_model(num_classes=5, model_config=control_config)
    set_seed(int(args.seed))
    candidate = create_model(num_classes=5, model_config=candidate_config)
    common_state = _common_state_summary(control, candidate)
    control_parameters = sum(value.numel() for value in control.parameters())
    candidate_parameters = sum(value.numel() for value in candidate.parameters())
    added_parameters = candidate_parameters - control_parameters
    faa_layers = [
        index + 1
        for index, block in enumerate(candidate.blocks)
        if isinstance(block.attn, FovealAggregatedAttention)
    ]

    dataset = _build_dataset(
        fold_data=paths["fold_data"],
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="faa_a1_train_only_preflight",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(next(iter(loader)))
    device = torch.device("cuda")
    images = images_cpu.to(device=device, non_blocking=True)
    labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = {
        key: value.to(device=device, non_blocking=True)
        for key, value in metadata_cpu.items()
    }

    del control
    candidate = candidate.to(device).train()
    candidate.zero_grad(set_to_none=True)
    fp32_count = int(args.fp32_batch_size)
    logits_fp32, _ = _forward(
        candidate,
        images[:fp32_count],
        {key: value[:fp32_count] for key, value in metadata.items()},
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    fp32_gradients = _gradient_summary(candidate)
    fp32_finite = bool(torch.isfinite(logits_fp32).all() and torch.isfinite(loss_fp32))

    candidate.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_bf16, _ = _forward(candidate, images, metadata)
        loss_bf16 = F.cross_entropy(logits_bf16.float(), labels)
    loss_bf16.backward()
    bf16_gradients = _gradient_summary(candidate)
    bf16_finite = bool(torch.isfinite(logits_bf16).all() and torch.isfinite(loss_bf16))

    candidate.eval()
    with torch.inference_mode():
        attention_logits, attention_features = _forward(
            candidate,
            images[:1],
            {key: value[:1] for key, value in metadata.items()},
            return_attention=True,
            return_trace=True,
        )
    first_attention = attention_features["attentions"][0]
    attention_row_error = float(
        (first_attention.sum(dim=-1) - 1.0).abs().amax().item()
    )
    prefix_count = int(candidate.num_prefix_tokens)
    patch_prefix_nonzero = int(
        torch.count_nonzero(first_attention[:, :, prefix_count:, :prefix_count]).item()
    )
    trace = attention_features.get("trace", {})
    dual_route_fraction = float(trace["foveal_dual_route_fraction"][0].item())
    local_mass = float(trace["foveal_local_mass_mean"][0].item())
    pooled_mass = float(trace["foveal_pooled_mass_mean"][0].item())

    del candidate
    gc.collect()
    torch.cuda.empty_cache()
    control_benchmark = _benchmark(
        config=control_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    candidate_benchmark = _benchmark(
        config=candidate_config,
        images=images,
        labels=labels,
        metadata=metadata,
        seed=int(args.seed),
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    runtime_ratio = (
        float(candidate_benchmark["median_seconds"])
        / float(control_benchmark["median_seconds"])
    )
    vram_ratio = (
        float(candidate_benchmark["peak_vram_gib"])
        / max(float(control_benchmark["peak_vram_gib"]), 1e-12)
    )

    set_seed(int(args.seed))
    export_candidate = create_model(num_classes=5, model_config=candidate_config).eval()
    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 8)
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(1, images_cpu.size(-2), images_cpu.size(-1), dtype=torch.bool)
    onnx_path = output_dir / "faa_candidate_static_batch1.onnx"
    try:
        onnx = _onnx_compare(
            wrapper=_FullExport(export_candidate),
            inputs=(images_cpu[:1].float(), bbox[:1].float(), image_mask[:1].bool()),
            input_names=("images", "bbox", "image_mask"),
            path=onnx_path,
        )
    except Exception as error:
        onnx = _failed_export(onnx_path, error)
    equation = _official_equation_replay(paths["official_attention"])

    focused_gradient_pass = all(
        bool(value["passed"]) for value in fp32_gradients.values()
    ) and all(bool(value["passed"]) for value in bf16_gradients.values())
    checks = {
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "head_pushed": _git_value("rev-parse", "HEAD")
        == _git_value("rev-parse", "@{upstream}"),
        "protocol_hash": hashes["protocol"] == LOCKED_PROTOCOL_SHA256,
        "raw_data_hash": hashes["raw_data"] == LOCKED_DATA_SHA256,
        "declaration_hash": hashes["declaration"] == LOCKED_DECLARATION_SHA256,
        "resolved_config_hash": hashes["resolved_config"] == LOCKED_RESOLVED_CONFIG_SHA256,
        "official_attention_hash": hashes["official_attention"]
        == LOCKED_OFFICIAL_ATTENTION_SHA256,
        "official_license_hash": hashes["official_license"]
        == LOCKED_OFFICIAL_LICENSE_SHA256,
        "paper_hash": hashes["paper"] == LOCKED_PAPER_SHA256,
        "official_commit": _official_commit(Path(args.official_root).resolve())
        == LOCKED_OFFICIAL_COMMIT,
        "current_command_hash": hashes["current_commands"] == LOCKED_COMMAND_SHA256,
        "command_history_hash": hashes["command_history"] == LOCKED_HISTORY_SHA256,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_sources": int(fold_summary.get("fit_sources", -1))
        == EXPECTED_FIT_SOURCES,
        "fold_holdout_sources": int(fold_summary.get("holdout_sources", -1))
        == EXPECTED_HOLDOUT_SOURCES,
        "fold_zero_source_overlap": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "dataset_fit_rows": len(dataset) == EXPECTED_FIT_ROWS,
        "common_state_bit_exact": bool(common_state["all_common_bit_exact"]),
        "candidate_layer_exact": faa_layers == [1],
        "added_parameters_budget": 0 < added_parameters <= MAX_ADDED_PARAMETERS,
        "official_equation_replay": bool(equation["passed"]),
        "fp32_finite": fp32_finite,
        "bf16_finite": bf16_finite,
        "all_gradient_families": focused_gradient_pass,
        "attention_finite_nonnegative": bool(
            torch.isfinite(first_attention).all() and (first_attention >= 0).all()
        ),
        "attention_rows_normalized": attention_row_error <= 1e-5,
        "patch_to_prefix_mass_zero": patch_prefix_nonzero == 0,
        "local_pooled_mass_nonzero": 0.0 < local_mass < 1.0 and 0.0 < pooled_mass < 1.0,
        "dual_route_fraction": dual_route_fraction >= 0.95,
        "runtime_ratio": math.isfinite(runtime_ratio) and runtime_ratio <= MAX_RUNTIME_RATIO,
        "candidate_vram_budget": float(candidate_benchmark["peak_vram_gib"])
        <= MAX_VRAM_GIB,
        "candidate_vram_ratio": math.isfinite(vram_ratio) and vram_ratio <= MAX_VRAM_RATIO,
        "onnx_succeeded": bool(onnx.get("succeeded", False)),
        "onnx_finite": bool(onnx.get("finite", False)),
        "onnx_error": float(onnx.get("maximum_absolute_error", 1e9)) <= 1e-4,
        "onnx_argmax": bool(onnx.get("argmax_match", False)),
        "attention_logits_finite": bool(torch.isfinite(attention_logits).all()),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    summary: Dict[str, object] = {
        "method": "foveal_aggregated_attention_a1_preflight",
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "upstream": _git_value("rev-parse", "@{upstream}"),
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "fold": fold_summary,
        "model": {
            "control_parameters": control_parameters,
            "candidate_parameters": candidate_parameters,
            "added_parameters": added_parameters,
            "faa_layers": faa_layers,
            "common_state": common_state,
        },
        "data": {
            "fit_rows": len(dataset),
            "batch_size": int(images.size(0)),
            "labels": [int(value) for value in labels.cpu().tolist()],
            "dataloader": loader_summary,
        },
        "official_equation": equation,
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "gradients": fp32_gradients,
        },
        "bf16": {
            "batch_size": int(images.size(0)),
            "loss": float(loss_bf16.detach().item()),
            "gradients": bf16_gradients,
        },
        "attention": {
            "shape": [int(value) for value in first_attention.shape],
            "row_sum_max_error": attention_row_error,
            "patch_prefix_nonzero": patch_prefix_nonzero,
            "local_mass_mean": local_mass,
            "pooled_mass_mean": pooled_mass,
            "dual_route_fraction": dual_route_fraction,
            "representation": attention_features.get("attention_representations", {}).get(0),
        },
        "resource": {
            "control": control_benchmark,
            "candidate": candidate_benchmark,
            "runtime_ratio": runtime_ratio,
            "vram_ratio": vram_ratio,
        },
        "onnx": onnx,
        "gate": {
            "formal_pair_permission": not failed,
            "checks": checks,
            "failed_checks": failed,
            "validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": "foveal_aggregated_attention_a1_preflight",
                "artifacts": artifacts,
                "raw_data_modified": False,
                "validation_used": False,
                "test_used": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run_preflight(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True), flush=True)
    if not bool(summary["gate"]["formal_pair_permission"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
