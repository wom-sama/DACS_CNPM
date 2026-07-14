from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset, build_eval_transform
from trkh.inference.inference import load_checkpoint
from trkh.models.model import classification_logits_from_features, create_model
from trkh.models.visual_contrast_attention import VisualContrastAttention


LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_SCRATCH_CHECKPOINT_SHA256 = (
    "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
)
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_BLOCKS = 8
MAX_ADDED_PARAMETERS = 350_000
MAX_PEAK_VRAM_GIB = 7.75

VCA_GRADIENT_MARKERS = {
    "qkv": (".attn.qkv.",),
    "positive_embedding": (".attn.positive_embedding",),
    "negative_embedding": (".attn.negative_embedding",),
    "stage1_lambda": (".attn.stage1_lambda_",),
    "stage2_lambda": (".attn.stage2_lambda_",),
    "stage1_norm": (".attn.stage1_norm.",),
    "stage2_norm": (".attn.stage2_norm.",),
    "depthwise_value": (".attn.depthwise_value.",),
    "projection": (".attn.proj.",),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only CUDA functional/resource readiness audit for TRKH "
            "Visual-Contrast Attention. Validation and test access are forbidden."
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
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp32-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-contrast-tokens", type=int, default=64)
    parser.add_argument("--max-peak-vram-gib", type=float, default=MAX_PEAK_VRAM_GIB)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(
            f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode("utf-8")
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_json_mapping(path: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def _vca_config(
    source: Mapping[str, object],
    *,
    visual_contrast_tokens: int,
) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "token_pruning": False,
            "early_token_mask_keep_rate": 1.0,
            "gated_relative_position_attention": False,
            "visual_contrast_attention": True,
            "visual_contrast_attention_layers": "1,2,3,4,5,6,7,8",
            "visual_contrast_tokens": int(visual_contrast_tokens),
            "pretrained": False,
        }
    )
    return config


def _control_config(source: Mapping[str, object]) -> Dict[str, object]:
    config = dict(source)
    config.update(
        {
            "token_pruning": False,
            "early_token_mask_keep_rate": 1.0,
            "gated_relative_position_attention": False,
            "visual_contrast_attention": False,
            "pretrained": False,
        }
    )
    return config


def _build_train_only_dataset(
    *,
    data_yaml: Path,
    model_config: Mapping[str, object],
    resolved_config: Mapping[str, object],
) -> MangoYOLOCropDataset:
    data_spec = load_data_spec(Path(data_yaml))
    augmentation = resolved_config.get("augmentation_config")
    if not isinstance(augmentation, Mapping):
        raise ValueError("resolved_config is missing augmentation_config")
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
        surface_detail_amplification_mode=str(
            augmentation.get("surface_detail_amplification_mode", "none")
        ),
        surface_detail_amplification_strength=float(
            augmentation.get("surface_detail_amplification_strength", 0.0)
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation.get("surface_detail_amplification_blur_radius", 1.25)
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation.get("surface_detail_amplification_foreground_weight", 0.85)
        ),
        eval_surface_detail_amplification=bool(
            augmentation.get("eval_surface_detail_amplification", False)
        ),
        mean=model_config.get("input_mean", (0.485, 0.456, 0.406)),
        std=model_config.get("input_std", (0.229, 0.224, 0.225)),
    )
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=transform,
        crop_margin_ratio=float(augmentation.get("crop_margin_ratio", 0.05)),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        class_aware_augmentation=False,
        class_crop_margin_scale_threshold=float(
            augmentation.get("class_crop_margin_scale_threshold", 1.5)
        ),
        class_crop_margin_max_ratio=float(
            augmentation.get("class_crop_margin_max_ratio", 0.16)
        ),
        classification_source_context=False,
        classification_bbox_metadata=False,
    )
    return dataset


def _unpack_batch(batch: object) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
        raise TypeError("Expected a classification batch with two or three items")
    images = batch[0]
    labels = batch[1]
    metadata = batch[2] if len(batch) == 3 else {}
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise TypeError("Classification images and labels must be tensors")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return images, labels, {
        str(key): value for key, value in metadata.items() if torch.is_tensor(value)
    }


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    return_attention: bool = False,
    return_trace: bool = False,
) -> tuple[Tensor, Mapping[str, object]]:
    image_valid_mask = metadata.get("image_mask")
    features = model.forward_features(
        images,
        image_valid_mask=image_valid_mask,
        return_attention=return_attention,
        return_trace=return_trace,
    )
    logits = classification_logits_from_features(model, features)
    return logits, features


def summarize_vca_gradients(model: nn.Module) -> Dict[str, Dict[str, object]]:
    named_parameters = dict(model.named_parameters())
    summary: Dict[str, Dict[str, object]] = {}
    for family, markers in VCA_GRADIENT_MARKERS.items():
        selected = {
            name: parameter
            for name, parameter in named_parameters.items()
            if any(marker in name for marker in markers)
        }
        gradients = [parameter.grad for parameter in selected.values()]
        present = [gradient for gradient in gradients if gradient is not None]
        finite = bool(present) and all(bool(torch.isfinite(value).all()) for value in present)
        nonzero = sum(
            int(torch.count_nonzero(value.detach()).item()) for value in present
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


def assess_visual_contrast_readiness(
    *,
    checks: Mapping[str, bool],
    peak_vram_gib: float,
    max_peak_vram_gib: float,
) -> Dict[str, object]:
    resolved_checks = {str(key): bool(value) for key, value in checks.items()}
    resolved_checks["peak_vram_within_budget"] = bool(
        math.isfinite(float(peak_vram_gib))
        and float(peak_vram_gib) <= float(max_peak_vram_gib)
    )
    failed = [name for name, passed in resolved_checks.items() if not passed]
    return {
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": resolved_checks,
        "failed_checks": failed,
        "thresholds": {
            "max_added_parameters": MAX_ADDED_PARAMETERS,
            "max_peak_vram_gib": float(max_peak_vram_gib),
            "expected_blocks": EXPECTED_BLOCKS,
            "expected_train_rows": EXPECTED_TRAIN_ROWS,
        },
    }


def _attention_summary(features: Mapping[str, object]) -> Dict[str, object]:
    attentions = features.get("attentions")
    if not isinstance(attentions, Mapping) or not attentions:
        raise RuntimeError("VCA forward did not return attention maps")
    shapes = []
    finite = True
    nonnegative = True
    max_row_sum_error = 0.0
    for layer, attention in sorted(attentions.items()):
        if not torch.is_tensor(attention):
            raise TypeError(f"Attention at layer {layer} is not a tensor")
        shapes.append([int(layer), *[int(value) for value in attention.shape]])
        finite = finite and bool(torch.isfinite(attention).all())
        nonnegative = nonnegative and bool((attention >= 0).all())
        error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
        max_row_sum_error = max(max_row_sum_error, error)
    return {
        "layers": [int(value) for value in sorted(attentions)],
        "shapes": shapes,
        "finite": finite,
        "nonnegative": nonnegative,
        "max_row_sum_error": max_row_sum_error,
        "representation": features.get("attention_representation"),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    resolved_config_path = Path(args.resolved_config).resolve()
    data_path = Path(args.data).resolve()
    if str(args.device).lower() != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-A VCA readiness requires an available CUDA device")
    if int(args.batch_size) != 32:
        raise ValueError("Locked Stage-A resource batch size is 32")
    if int(args.fp32_batch_size) <= 0 or int(args.fp32_batch_size) > int(args.batch_size):
        raise ValueError("fp32-batch-size must be in [1, batch-size]")

    data_sha = _sha256(data_path)
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(source_model_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Scratch checkpoint lacks model_config/class_names")
    resolved_config = _load_json_mapping(resolved_config_path)
    candidate_config = _vca_config(
        source_model_config,
        visual_contrast_tokens=int(args.visual_contrast_tokens),
    )
    control_config = _control_config(source_model_config)
    num_classes = len(class_names)

    set_seed(int(args.seed))
    control_model = create_model(num_classes=num_classes, model_config=control_config)
    control_parameters = sum(parameter.numel() for parameter in control_model.parameters())
    del control_model

    set_seed(int(args.seed))
    model = create_model(num_classes=num_classes, model_config=candidate_config)
    first_state_sha = _state_sha256(model)
    set_seed(int(args.seed))
    roundtrip_model = create_model(num_classes=num_classes, model_config=candidate_config)
    second_state_sha = _state_sha256(roundtrip_model)
    roundtrip_model.load_state_dict(model.state_dict(), strict=True)
    roundtrip_state_sha = _state_sha256(roundtrip_model)
    del roundtrip_model

    candidate_parameters = sum(parameter.numel() for parameter in model.parameters())
    added_parameters = int(candidate_parameters - control_parameters)
    vca_blocks = [
        index + 1
        for index, block in enumerate(model.blocks)
        if isinstance(block.attn, VisualContrastAttention)
    ]

    dataset = _build_train_only_dataset(
        data_yaml=data_path,
        model_config=candidate_config,
        resolved_config=resolved_config,
    )
    if len(dataset) < int(args.batch_size):
        raise RuntimeError("Train dataset is smaller than the locked resource batch")
    selected_paths = [
        str(dataset.samples[index].image_path.resolve())
        for index in range(int(args.batch_size))
    ]
    train_root = Path(load_data_spec(data_path).split_images_dir("train")).resolve()
    train_only_paths = all(Path(path).is_relative_to(train_root) for path in selected_paths)
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(args.num_workers),
        requested_pin_memory=True,
        context="vca_stage_a_train_only",
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=True,
        **dataloader_kwargs,
    )
    batch = next(iter(loader))
    images_cpu, labels_cpu, metadata_cpu = _unpack_batch(batch)
    device = torch.device("cuda")
    images = images_cpu.to(device=device, non_blocking=True)
    labels = labels_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    metadata = {
        key: value.to(device=device, non_blocking=True)
        for key, value in metadata_cpu.items()
    }
    model = model.to(device)
    model.train()

    fp32_count = int(args.fp32_batch_size)
    fp32_metadata = {key: value[:fp32_count] for key, value in metadata.items()}
    model.zero_grad(set_to_none=True)
    logits_fp32, _ = _forward_logits(
        model,
        images[:fp32_count],
        fp32_metadata,
    )
    loss_fp32 = F.cross_entropy(logits_fp32.float(), labels[:fp32_count])
    loss_fp32.backward()
    fp32_gradients = summarize_vca_gradients(model)
    fp32_finite = bool(torch.isfinite(logits_fp32).all() and torch.isfinite(loss_fp32))

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits_amp, _ = _forward_logits(model, images, metadata)
        loss_amp = F.cross_entropy(logits_amp.float(), labels)
    loss_amp.backward()
    torch.cuda.synchronize(device)
    amp_seconds = float(time.perf_counter() - started)
    peak_vram_gib = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    amp_gradients = summarize_vca_gradients(model)
    amp_finite = bool(torch.isfinite(logits_amp).all() and torch.isfinite(loss_amp))

    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        attention_logits, attention_features = _forward_logits(
            model,
            images[:1],
            {key: value[:1] for key, value in metadata.items()},
            return_attention=True,
            return_trace=True,
        )
    attention = _attention_summary(attention_features)
    trace = attention_features.get("trace", {})
    trace_layers = (
        trace.get("visual_contrast_attention_layers")
        if isinstance(trace, Mapping)
        else None
    )
    traced_blocks = (
        [int(value) for value in trace_layers.detach().cpu().tolist()]
        if torch.is_tensor(trace_layers)
        else []
    )
    expected_tokens = int(model.num_prefix_tokens) + int(
        attention_features["grid_size"][0] * attention_features["grid_size"][1]
    )
    attention_shapes_valid = all(
        shape[2:] == [int(model.blocks[0].attn.num_heads), expected_tokens, expected_tokens]
        for shape in attention["shapes"]
    )

    fp32_gradient_pass = all(
        bool(value["passed"]) for value in fp32_gradients.values()
    )
    amp_gradient_pass = all(bool(value["passed"]) for value in amp_gradients.values())
    checks = {
        "locked_data_sha256": data_sha == LOCKED_DATA_SHA256,
        "locked_scratch_checkpoint_sha256": (
            checkpoint_sha == LOCKED_SCRATCH_CHECKPOINT_SHA256
        ),
        "train_split_only": bool(train_only_paths),
        "train_row_count": len(dataset) == EXPECTED_TRAIN_ROWS,
        "deterministic_same_seed_initialization": first_state_sha == second_state_sha,
        "strict_state_roundtrip": first_state_sha == roundtrip_state_sha,
        "all_eight_vca_blocks": vca_blocks == list(range(1, EXPECTED_BLOCKS + 1)),
        "trace_all_eight_vca_blocks": traced_blocks == vca_blocks,
        "added_parameters_within_budget": (
            0 < added_parameters <= MAX_ADDED_PARAMETERS
        ),
        "dense_config_locked": (
            not bool(candidate_config["token_pruning"])
            and float(candidate_config["early_token_mask_keep_rate"]) == 1.0
        ),
        "fp32_logits_loss_finite": fp32_finite,
        "fp32_all_vca_gradient_families": fp32_gradient_pass,
        "amp_logits_loss_finite": amp_finite,
        "amp_all_vca_gradient_families": amp_gradient_pass,
        "attention_logits_finite": bool(torch.isfinite(attention_logits).all()),
        "attention_all_layers": attention["layers"] == list(range(EXPECTED_BLOCKS)),
        "attention_shapes_full_grid": bool(attention_shapes_valid),
        "attention_finite": bool(attention["finite"]),
        "attention_nonnegative": bool(attention["nonnegative"]),
        "attention_rows_normalized": float(attention["max_row_sum_error"]) <= 1e-5,
        "attention_provenance": (
            attention["representation"] == "vca_effective_positive"
        ),
        "validation_not_loaded": True,
        "test_not_loaded": True,
    }
    gate = assess_visual_contrast_readiness(
        checks=checks,
        peak_vram_gib=peak_vram_gib,
        max_peak_vram_gib=float(args.max_peak_vram_gib),
    )
    summary: Dict[str, object] = {
        "method": "visual_contrast_attention",
        "protocol_stage": "A_train_only_functional_resource_preflight",
        "sources": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config": str(resolved_config_path),
            "resolved_config_sha256": _sha256(resolved_config_path),
            "data_yaml": str(data_path),
            "data_yaml_sha256": data_sha,
            "split_loaded": "train",
            "validation_loaded": False,
            "test_loaded": False,
        },
        "model": {
            "control_parameters": int(control_parameters),
            "candidate_parameters": int(candidate_parameters),
            "added_parameters": added_parameters,
            "vca_blocks": vca_blocks,
            "visual_contrast_tokens": int(args.visual_contrast_tokens),
            "state_sha256": first_state_sha,
            "same_seed_state_sha256": second_state_sha,
            "roundtrip_state_sha256": roundtrip_state_sha,
        },
        "data": {
            "train_rows": len(dataset),
            "batch_size": int(images.size(0)),
            "labels": [int(value) for value in labels.detach().cpu().tolist()],
            "selected_paths": selected_paths,
            "dataloader": dataloader_summary,
        },
        "fp32": {
            "batch_size": fp32_count,
            "loss": float(loss_fp32.detach().item()),
            "logits_shape": [int(value) for value in logits_fp32.shape],
            "gradients": fp32_gradients,
        },
        "cuda_amp": {
            "batch_size": int(images.size(0)),
            "loss": float(loss_amp.detach().item()),
            "logits_shape": [int(value) for value in logits_amp.shape],
            "forward_backward_seconds": amp_seconds,
            "peak_vram_gib": peak_vram_gib,
            "device_name": torch.cuda.get_device_name(device),
            "gradients": amp_gradients,
        },
        "attention": attention,
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
