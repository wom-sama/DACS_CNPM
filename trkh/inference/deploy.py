from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
from torch.utils.data import DataLoader, Subset

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    PseudoVideoAugmenter,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.evaluate import evaluate_model, resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.metrics import build_metrics
from trkh.inference.inference import export_onnx
from trkh.models.model import (
    build_model_from_checkpoint as build_checkpoint_model,
    extract_bbox_from_model_output,
)
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    maybe_enable_dataset_image_cache,
    set_seed,
)

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover
    trt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export va benchmark deploy cho ViT-Registers.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
        help="Cach xu ly names trong data.yaml; dung raw cho dataset tuy bien hoac >4 lop.",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=0,
        help="Neu > 0, validate so class trong data.yaml truoc benchmark accuracy.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-onnx", action="store_true", default=False)
    parser.add_argument(
        "--onnx-static-int8",
        action="store_true",
        default=False,
        help="Build train-calibrated QDQ INT8 ONNX for Conv/MatMul/Gemm mobile proxy.",
    )
    parser.add_argument("--onnx-static-int8-calibration-samples", type=int, default=512)
    parser.add_argument(
        "--require-mobile-proxy-gates",
        action="store_true",
        default=False,
        help="Fail after writing deploy_summary.json unless static INT8 desktop proxy gates pass.",
    )
    parser.add_argument("--mobile-proxy-max-fp32-probability-error", type=float, default=1e-5)
    parser.add_argument("--mobile-proxy-max-macro-f1-drop", type=float, default=0.005)
    parser.add_argument("--mobile-proxy-max-class1-f1-drop", type=float, default=0.010)
    parser.add_argument("--mobile-proxy-min-int8-macro-f1", type=float, default=0.80)
    parser.add_argument("--mobile-proxy-min-int8-class1-f1", type=float, default=0.60)
    parser.add_argument("--mobile-proxy-min-int8-class1-precision", type=float, default=0.55)
    parser.add_argument("--mobile-proxy-min-int8-class1-recall", type=float, default=0.55)
    parser.add_argument("--mobile-proxy-min-qdq-target-op-coverage", type=float, default=0.90)
    parser.add_argument("--mobile-proxy-max-int8-mib", type=float, default=6.0)
    parser.add_argument("--mobile-proxy-max-parameters", type=int, default=5_000_000)
    parser.add_argument("--skip-trt-engine", action="store_true", default=False)
    parser.add_argument(
        "--allow-uncertified-precision-ensemble-tensorrt",
        action="store_true",
        default=False,
        help=(
            "Explicit research-only override. Precision-ensemble TensorRT "
            "backends currently fail the frozen probability parity gate."
        ),
    )
    parser.add_argument("--trtexec-path", type=Path, default=None)
    parser.add_argument("--trt-int8", action="store_true", default=False)
    parser.add_argument("--trt-calibration-samples", type=int, default=128)
    parser.add_argument(
        "--trt-calibration-split",
        choices=("train",),
        default="train",
        help="Reportable INT8 calibration is train-only; val remains the deployment gate.",
    )
    parser.add_argument("--trt-calibration-cache", type=Path, default=None)
    parser.add_argument("--skip-accuracy", action="store_true", default=False)
    parser.add_argument("--skip-benchmark", action="store_true", default=False)
    parser.add_argument("--split", choices=("auto", "train", "val", "test"), default="auto")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        default=False,
        help="Explicit retrospective/final-report authorization for --split test.",
    )
    return parser.parse_args()


def resolve_temporal_frames(checkpoint: Dict[str, object]) -> int:
    return max(1, int(checkpoint.get("model_config", {}).get("temporal_frames", 1)))


def validate_tensorrt_export_policy(
    checkpoint: Mapping[str, object],
    *,
    skip_trt_engine: bool,
    allow_uncertified_precision_ensemble_tensorrt: bool,
) -> None:
    model_config = checkpoint.get("model_config", {})
    model_type = (
        str(model_config.get("model_type", "")).strip().lower()
        if isinstance(model_config, Mapping)
        else ""
    )
    if (
        model_type == "precision_ensemble"
        and not bool(skip_trt_engine)
        and not bool(allow_uncertified_precision_ensemble_tensorrt)
    ):
        raise ValueError(
            "Precision-ensemble TensorRT export is blocked by default: FP16 "
            "and FP32 noTF32/O0 failed the frozen probability parity gate. "
            "Use --skip-trt-engine for the certified ONNX CPU artifact. The "
            "explicit --allow-uncertified-precision-ensemble-tensorrt override "
            "is research-only and does not certify the resulting engine."
        )


def resolve_input_shape(
    image_size: int,
    batch_size: int,
    temporal_frames: int,
) -> Tuple[int, ...]:
    if temporal_frames > 1:
        return (int(batch_size), int(temporal_frames), 3, int(image_size), int(image_size))
    return (int(batch_size), 3, int(image_size), int(image_size))


def format_trt_shape(input_shape: Sequence[int]) -> str:
    return "images:" + "x".join(str(int(dim)) for dim in input_shape)


def resolve_runtime_input_shapes(
    model: nn.Module,
    image_input_shape: Sequence[int],
) -> Dict[str, Tuple[int, ...]]:
    image_shape = tuple(int(dim) for dim in image_input_shape)
    shapes = {"images": image_shape}
    if bool(getattr(model, "requires_spatial_metadata", False)):
        if len(image_shape) != 4:
            raise ValueError(
                "Spatial-metadata classifier requires image shape [B,C,H,W]."
            )
        shapes["image_valid_mask"] = (
            image_shape[0],
            image_shape[-2],
            image_shape[-1],
        )
        shapes["bbox"] = (image_shape[0], 4)
    return shapes


def build_runtime_inputs(
    model: nn.Module,
    image_input_shape: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    shapes = resolve_runtime_input_shapes(model, image_input_shape)
    images = torch.randn(*shapes["images"], device=device, dtype=torch.float32)
    if "image_valid_mask" not in shapes:
        return (images,)
    image_valid_mask = torch.ones(
        *shapes["image_valid_mask"],
        device=device,
        dtype=torch.float32,
    )
    bbox = torch.tensor(
        ((0.5, 0.5, 1.0, 1.0),),
        device=device,
        dtype=torch.float32,
    ).expand(shapes["bbox"][0], -1)
    return images, image_valid_mask, bbox


def format_trt_shapes(input_shapes: Mapping[str, Sequence[int]]) -> str:
    return ",".join(
        f"{name}:" + "x".join(str(int(dim)) for dim in shape)
        for name, shape in input_shapes.items()
    )


def build_runtime_transform(
    checkpoint: Dict[str, object],
    image_size: int,
    temporal_frames: int,
):
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    input_mean, input_std = checkpoint_input_normalization(checkpoint)
    base_transform = build_eval_transform(
        image_size=image_size,
        resize_mode=augmentation_config.get("resize_mode", "pad"),
        illumination_normalization=bool(augmentation_config.get("illumination_normalization", False)),
        illumination_normalization_strength=float(
            augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
        ),
        foreground_crop_mode=str(augmentation_config.get("foreground_crop_mode", "none") or "none"),
        foreground_crop_margin_ratio=float(augmentation_config.get("foreground_crop_margin_ratio", 0.08) or 0.08),
        foreground_crop_min_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation_config.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(augmentation_config.get("background_suppression_mode", "none") or "none"),
        background_suppression_margin=float(augmentation_config.get("background_suppression_margin", 0.08) or 0.08),
        background_suppression_blur_radius=float(augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0),
        surface_detail_amplification_mode=str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        surface_detail_amplification_strength=float(
            augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85) or 0.85
        ),
        eval_surface_detail_amplification=bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
        mean=input_mean,
        std=input_std,
    )
    if temporal_frames > 1:
        return PseudoVideoAugmenter(
            frame_transform=base_transform,
            temporal_frames=temporal_frames,
            deterministic=True,
        )
    return base_transform


def build_runtime_model_from_checkpoint(
    checkpoint: Dict[str, object],
    device: torch.device,
    override_image_size: Optional[int] = None,
) -> nn.Module:
    model = build_checkpoint_model(
        checkpoint=checkpoint,
        num_classes=len(checkpoint["class_names"]),
        override_image_size=override_image_size,
    )
    model.to(device)
    model.eval()
    return model


def export_torchscript(model: nn.Module, input_shape: Sequence[int], output_path: Path) -> Path:
    inputs = build_runtime_inputs(
        model,
        input_shape,
        next(model.parameters()).device,
    )
    scripted = torch.jit.trace(model, inputs, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    return output_path


def export_dynamic_int8(model: nn.Module, input_shape: Sequence[int], output_path: Path) -> tuple[nn.Module, Path]:
    cpu_model = copy.deepcopy(model).cpu().eval()
    quantized_model = quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
    inputs = build_runtime_inputs(quantized_model, input_shape, torch.device("cpu"))
    quantized_script = torch.jit.trace(quantized_model, inputs, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_script.save(str(output_path))
    return quantized_model, output_path


def build_eval_loader(
    checkpoint: Dict[str, object],
    data_yaml: Path,
    batch_size: int,
    num_workers: int,
    split: str = "auto",
    override_image_size: Optional[int] = None,
    class_name_mode: Optional[str] = None,
    expected_num_classes: Optional[int] = None,
    allow_test_split: bool = False,
    stratified_max_samples: int = 0,
    sampling_seed: int = 42,
) -> tuple[DataLoader, list[str]]:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes,
    )
    class_names = [str(name) for name in checkpoint["class_names"]]
    if list(data_spec.class_names) != class_names:
        raise ValueError(
            "Checkpoint/data class order mismatch. Refusing deploy evaluation because "
            f"checkpoint={class_names!r}, data={list(data_spec.class_names)!r}."
        )
    image_size = int(override_image_size or checkpoint["model_config"].get("image_size", 224))
    temporal_frames = resolve_temporal_frames(checkpoint)
    resolved_split = "val" if split == "auto" else str(split)
    if resolved_split == "test" and not bool(allow_test_split):
        raise ValueError(
            "Deploy evaluation on test requires --allow-test-split; auto always uses val."
        )
    precision_ensemble = str(
        checkpoint.get("model_config", {}).get("model_type", "")
    ).strip().lower() == "precision_ensemble"
    runtime_transform = build_runtime_transform(
        checkpoint,
        image_size=image_size,
        temporal_frames=temporal_frames,
    )
    if data_spec.data_format == "classification_folder":
        if precision_ensemble:
            raise ValueError(
                "precision_ensemble requires bbox metadata and cannot be evaluated from "
                "canonical classification_folder data without an explicit paired source."
            )
        dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split=resolved_split,
            transform=runtime_transform,
            class_aware_augmentation=False,
        )
    else:
        dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split=resolved_split,
            transform=runtime_transform,
            crop_margin_ratio=float(
                checkpoint.get("augmentation_config", {}).get("crop_margin_ratio", 0.05)
            ),
            crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
            classification_target=True,
            classification_object_crops=True,
            classification_bbox_metadata=precision_ensemble,
        )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=False,
        context=f"deploy_{resolved_split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context=f"deploy_{resolved_split}",
    )
    print(
        "DataLoader setup:",
        {
            "loader": dataloader_summary,
            "image_cache": cache_summary,
        },
        flush=True,
    )
    sampling_summary: Dict[str, object] = {
        "mode": "full_sequential",
        "source_samples": int(len(dataset)),
        "selected_samples": int(len(dataset)),
        "per_class": {
            str(class_index): int(count)
            for class_index, count in enumerate(
                dataset.class_counts(data_spec.num_classes)
            )
        },
        "seed": None,
    }
    loader_dataset = dataset
    if int(stratified_max_samples) > 0:
        labels = dataset.labels()
        selected_indices, selected_counts = deterministic_stratified_indices(
            labels,
            max_samples=int(stratified_max_samples),
            seed=int(sampling_seed),
        )
        loader_dataset = Subset(dataset, selected_indices)
        sampling_summary = {
            "mode": "deterministic_equal_class_stratified",
            "source_samples": int(len(dataset)),
            "selected_samples": int(len(selected_indices)),
            "per_class": {
                str(class_index): int(selected_counts.get(class_index, 0))
                for class_index in range(data_spec.num_classes)
            },
            "seed": int(sampling_seed),
        }
    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(data_spec.num_classes)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    loader.trkh_sampling_summary = sampling_summary
    return loader, class_names


def deterministic_stratified_indices(
    labels: Sequence[int],
    *,
    max_samples: int,
    seed: int,
) -> tuple[List[int], Dict[int, int]]:
    """Select a deterministic, class-interleaved calibration subset.

    Equal class coverage is intentional for PTQ: it prevents a folder-sorted
    classification dataset from calibrating on only its first class and gives
    the minority class a meaningful share of activation-range observations.
    """

    target = min(max(1, int(max_samples)), len(labels))
    if target <= 0:
        return [], {}
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        class_index = int(label)
        by_class.setdefault(class_index, []).append(int(index))
    if not by_class:
        return [], {}

    class_indices = sorted(by_class)
    quotas = {class_index: 0 for class_index in class_indices}
    remaining = target
    while remaining > 0:
        progressed = False
        for class_index in class_indices:
            if remaining <= 0:
                break
            if quotas[class_index] >= len(by_class[class_index]):
                continue
            quotas[class_index] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    selected_by_class: Dict[int, List[int]] = {}
    for offset, class_index in enumerate(class_indices):
        candidates = by_class[class_index]
        generator = torch.Generator().manual_seed(
            int(seed) + 1_000_003 * int(offset + 1)
        )
        permutation = torch.randperm(len(candidates), generator=generator).tolist()
        selected_by_class[class_index] = [
            candidates[position]
            for position in permutation[: quotas[class_index]]
        ]

    selected: List[int] = []
    max_quota = max(quotas.values(), default=0)
    for position in range(max_quota):
        for class_index in class_indices:
            class_selected = selected_by_class[class_index]
            if position < len(class_selected):
                selected.append(class_selected[position])
    return selected, quotas


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> Dict[str, object]:
    return evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
        class_names=class_names,
        criterion=None,
        amp=False,
        max_batches=None,
        collect_artifact_stats=True,
    )


def benchmark_torch_module(
    model: nn.Module,
    input_shape: Sequence[int],
    device: torch.device,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    model.eval()
    inputs = build_runtime_inputs(model, input_shape, device)
    timings = []
    batch_size = int(input_shape[0]) if input_shape else 1
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(max(1, runs)):
            start = time.perf_counter()
            _ = model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    timings_tensor = torch.tensor(timings, dtype=torch.float32)
    latency_ms = float(timings_tensor.mean().item())
    p95_ms = float(torch.quantile(timings_tensor, 0.95).item())
    return {
        "latency_ms": latency_ms,
        "p95_latency_ms": p95_ms,
        "throughput_images_per_sec": float(batch_size * 1000.0 / max(latency_ms, 1e-6)),
    }


def benchmark_onnx_model(
    onnx_path: Path,
    input_shapes: Mapping[str, Sequence[int]],
    warmup: int,
    runs: int,
) -> Optional[Dict[str, float]]:
    if ort is None:
        return None

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    dummy_inputs: Dict[str, object] = {}
    for name, shape in input_shapes.items():
        if name == "image_valid_mask":
            dummy_inputs[name] = torch.ones(*shape, dtype=torch.float32).numpy()
        elif name == "bbox":
            dummy_inputs[name] = torch.tensor(
                ((0.5, 0.5, 1.0, 1.0),),
                dtype=torch.float32,
            ).expand(int(shape[0]), -1).numpy()
        else:
            dummy_inputs[name] = torch.randn(*shape).numpy()
    timings = []
    image_shape = tuple(int(dim) for dim in input_shapes["images"])
    batch_size = int(image_shape[0]) if image_shape else 1

    for _ in range(max(0, warmup)):
        session.run(None, dummy_inputs)

    for _ in range(max(1, runs)):
        start = time.perf_counter()
        session.run(None, dummy_inputs)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    timings_tensor = torch.tensor(timings, dtype=torch.float32)
    latency_ms = float(timings_tensor.mean().item())
    p95_ms = float(torch.quantile(timings_tensor, 0.95).item())
    return {
        "latency_ms": latency_ms,
        "p95_latency_ms": p95_ms,
        "throughput_images_per_sec": float(batch_size * 1000.0 / max(latency_ms, 1e-6)),
    }


def _classification_targets_for_mobile_proxy(value: object) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError("Mobile ONNX accuracy requires tensor classification targets.")
    targets = value.detach().cpu()
    if targets.ndim == 1:
        return targets.to(dtype=torch.long)
    if targets.ndim == 2 and int(targets.size(1)) > 1:
        return targets.argmax(dim=1).to(dtype=torch.long)
    raise ValueError(
        f"Mobile ONNX targets must have shape [B] or [B,C], got {tuple(targets.shape)}."
    )


def _onnx_classification_logits(
    session: "ort.InferenceSession",
    images: np.ndarray,
) -> torch.Tensor:
    inputs = session.get_inputs()
    if len(inputs) != 1 or str(inputs[0].name) != "images":
        raise ValueError(
            "Mobile ONNX proxy currently requires one input named 'images'; "
            f"observed={[entry.name for entry in inputs]}."
        )
    outputs = session.run(None, {"images": np.ascontiguousarray(images)})
    if not outputs:
        raise RuntimeError("ONNX classifier returned no outputs.")
    logits = torch.from_numpy(np.asarray(outputs[0])).to(dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError(
            f"ONNX classification logits must have shape [B,C], got {tuple(logits.shape)}."
        )
    return logits


def evaluate_mobile_onnx_pair(
    *,
    reference_model: nn.Module,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    loader: DataLoader,
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Evaluate PyTorch/FP32-ONNX/INT8-ONNX on one aligned validation stream."""

    if ort is None:
        raise ImportError("onnxruntime is required for mobile ONNX accuracy gates.")
    reference_model = reference_model.cpu().eval()
    fp32_session = ort.InferenceSession(
        str(Path(fp32_onnx_path).resolve()),
        providers=["CPUExecutionProvider"],
    )
    int8_session = ort.InferenceSession(
        str(Path(int8_onnx_path).resolve()),
        providers=["CPUExecutionProvider"],
    )

    target_chunks: List[torch.Tensor] = []
    reference_prediction_chunks: List[torch.Tensor] = []
    fp32_prediction_chunks: List[torch.Tensor] = []
    int8_prediction_chunks: List[torch.Tensor] = []
    reference_to_fp32_argmax_mismatches = 0
    fp32_to_int8_argmax_mismatches = 0
    reference_to_fp32_max_logit_error = 0.0
    reference_to_fp32_max_probability_error = 0.0
    fp32_to_int8_max_logit_error = 0.0
    fp32_to_int8_max_probability_error = 0.0
    batch_count = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError("Mobile ONNX loader must yield (images, targets, ...).")
            images = batch[0].detach().cpu().to(dtype=torch.float32).contiguous()
            targets = _classification_targets_for_mobile_proxy(batch[1])
            reference_output = reference_model(images)
            reference_logits, _ = extract_bbox_from_model_output(reference_output)
            if not torch.is_tensor(reference_logits) or reference_logits.ndim != 2:
                raise ValueError("Reference classifier must return logits [B,C].")
            reference_logits = reference_logits.detach().cpu().to(dtype=torch.float32)
            numpy_images = images.numpy()
            fp32_logits = _onnx_classification_logits(fp32_session, numpy_images)
            int8_logits = _onnx_classification_logits(int8_session, numpy_images)
            if (
                tuple(reference_logits.shape) != tuple(fp32_logits.shape)
                or tuple(fp32_logits.shape) != tuple(int8_logits.shape)
                or int(targets.numel()) != int(reference_logits.size(0))
            ):
                raise ValueError(
                    "Aligned mobile proxy shapes differ: "
                    f"targets={tuple(targets.shape)}, torch={tuple(reference_logits.shape)}, "
                    f"fp32={tuple(fp32_logits.shape)}, int8={tuple(int8_logits.shape)}."
                )

            reference_predictions = reference_logits.argmax(dim=1)
            fp32_predictions = fp32_logits.argmax(dim=1)
            int8_predictions = int8_logits.argmax(dim=1)
            reference_to_fp32_argmax_mismatches += int(
                (reference_predictions != fp32_predictions).sum().item()
            )
            fp32_to_int8_argmax_mismatches += int(
                (fp32_predictions != int8_predictions).sum().item()
            )
            reference_to_fp32_max_logit_error = max(
                reference_to_fp32_max_logit_error,
                float((reference_logits - fp32_logits).abs().max().item()),
            )
            fp32_to_int8_max_logit_error = max(
                fp32_to_int8_max_logit_error,
                float((fp32_logits - int8_logits).abs().max().item()),
            )
            reference_probabilities = torch.softmax(reference_logits, dim=1)
            fp32_probabilities = torch.softmax(fp32_logits, dim=1)
            int8_probabilities = torch.softmax(int8_logits, dim=1)
            reference_to_fp32_max_probability_error = max(
                reference_to_fp32_max_probability_error,
                float((reference_probabilities - fp32_probabilities).abs().max().item()),
            )
            fp32_to_int8_max_probability_error = max(
                fp32_to_int8_max_probability_error,
                float((fp32_probabilities - int8_probabilities).abs().max().item()),
            )
            target_chunks.append(targets)
            reference_prediction_chunks.append(reference_predictions)
            fp32_prediction_chunks.append(fp32_predictions)
            int8_prediction_chunks.append(int8_predictions)
            batch_count += 1

    if not target_chunks:
        raise ValueError("Mobile ONNX validation loader produced no samples.")
    targets = torch.cat(target_chunks)
    reference_predictions = torch.cat(reference_prediction_chunks)
    fp32_predictions = torch.cat(fp32_prediction_chunks)
    int8_predictions = torch.cat(int8_prediction_chunks)
    total = int(targets.numel())
    return {
        "artifact_role": "desktop_mobile_proxy_not_target_device_certification",
        "samples": total,
        "batches": int(batch_count),
        "seconds": float(time.perf_counter() - started),
        "torch_reference": build_metrics(targets, reference_predictions, class_names),
        "onnx_fp32": build_metrics(targets, fp32_predictions, class_names),
        "onnx_int8_qdq": build_metrics(targets, int8_predictions, class_names),
        "torch_to_onnx_fp32": {
            "argmax_mismatches": int(reference_to_fp32_argmax_mismatches),
            "argmax_mismatch_rate": float(reference_to_fp32_argmax_mismatches / total),
            "max_abs_logit_error": float(reference_to_fp32_max_logit_error),
            "max_abs_probability_error": float(reference_to_fp32_max_probability_error),
        },
        "onnx_fp32_to_int8_qdq": {
            "argmax_mismatches": int(fp32_to_int8_argmax_mismatches),
            "argmax_mismatch_rate": float(fp32_to_int8_argmax_mismatches / total),
            "max_abs_logit_error": float(fp32_to_int8_max_logit_error),
            "max_abs_probability_error": float(fp32_to_int8_max_probability_error),
        },
    }


def build_mobile_proxy_gate(
    *,
    evaluation: Mapping[str, object],
    quantization: Mapping[str, object],
    parameter_count: int,
    max_fp32_probability_error: float,
    max_macro_f1_drop: float,
    max_class1_f1_drop: float,
    max_int8_mib: float,
    max_parameters: int,
    min_int8_macro_f1: float = 0.0,
    min_int8_class1_f1: float = 0.0,
    min_int8_class1_precision: float = 0.0,
    min_int8_class1_recall: float = 0.0,
    min_qdq_target_op_coverage: float = 0.0,
) -> Dict[str, object]:
    fp32_metrics = evaluation.get("onnx_fp32", {})
    int8_metrics = evaluation.get("onnx_int8_qdq", {})
    torch_parity = evaluation.get("torch_to_onnx_fp32", {})
    if not isinstance(fp32_metrics, Mapping) or not isinstance(int8_metrics, Mapping):
        raise ValueError("Mobile gate requires FP32 and INT8 ONNX metrics.")
    fp32_per_class = fp32_metrics.get("per_class", [])
    int8_per_class = int8_metrics.get("per_class", [])
    if not isinstance(fp32_per_class, Sequence) or len(fp32_per_class) <= 1:
        raise ValueError("Mobile gate requires class-1 FP32 metrics.")
    if not isinstance(int8_per_class, Sequence) or len(int8_per_class) <= 1:
        raise ValueError("Mobile gate requires class-1 INT8 metrics.")
    fp32_macro = float(fp32_metrics.get("macro_f1", 0.0) or 0.0)
    int8_macro = float(int8_metrics.get("macro_f1", 0.0) or 0.0)
    fp32_class1 = float(fp32_per_class[1].get("f1", 0.0) or 0.0)
    int8_class1 = float(int8_per_class[1].get("f1", 0.0) or 0.0)
    int8_class1_precision = float(
        int8_per_class[1].get("precision", 0.0) or 0.0
    )
    int8_class1_recall = float(int8_per_class[1].get("recall", 0.0) or 0.0)
    int8_summary = quantization.get("int8", {})
    if not isinstance(int8_summary, Mapping):
        raise ValueError("Mobile gate requires INT8 artifact summary.")
    int8_size_mib = float(int8_summary.get("size_bytes", 0) or 0) / float(2**20)
    fully_quantized_target_op_coverage = float(
        int8_summary.get("fully_quantized_target_op_coverage", 0.0) or 0.0
    )
    checks = {
        "torch_to_fp32_onnx_argmax_mismatches_zero": int(
            torch_parity.get("argmax_mismatches", -1) or 0
        )
        == 0,
        "torch_to_fp32_onnx_probability_error": float(
            torch_parity.get("max_abs_probability_error", float("inf"))
        )
        <= float(max_fp32_probability_error),
        "int8_macro_f1_drop": (fp32_macro - int8_macro) <= float(max_macro_f1_drop),
        "int8_class1_f1_drop": (fp32_class1 - int8_class1)
        <= float(max_class1_f1_drop),
        "int8_absolute_macro_f1": int8_macro >= float(min_int8_macro_f1),
        "int8_absolute_class1_f1": int8_class1 >= float(min_int8_class1_f1),
        "int8_absolute_class1_precision": int8_class1_precision
        >= float(min_int8_class1_precision),
        "int8_absolute_class1_recall": int8_class1_recall
        >= float(min_int8_class1_recall),
        "fully_quantized_qdq_target_op_coverage": fully_quantized_target_op_coverage
        >= float(min_qdq_target_op_coverage),
        "int8_package_size": int8_size_mib <= float(max_int8_mib),
        "parameter_count": int(parameter_count) <= int(max_parameters),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "desktop_mobile_proxy_only",
        "target_device_certified": False,
        "checks": checks,
        "observed": {
            "parameter_count": int(parameter_count),
            "int8_size_mib": int8_size_mib,
            "fp32_macro_f1": fp32_macro,
            "int8_macro_f1": int8_macro,
            "macro_f1_drop": float(fp32_macro - int8_macro),
            "fp32_class1_f1": fp32_class1,
            "int8_class1_f1": int8_class1,
            "int8_class1_precision": int8_class1_precision,
            "int8_class1_recall": int8_class1_recall,
            "class1_f1_drop": float(fp32_class1 - int8_class1),
            "fully_quantized_qdq_target_op_coverage": (
                fully_quantized_target_op_coverage
            ),
            "torch_to_fp32_argmax_mismatches": int(
                torch_parity.get("argmax_mismatches", -1)
            ),
            "torch_to_fp32_max_probability_error": float(
                torch_parity.get("max_abs_probability_error", float("inf"))
            ),
        },
        "thresholds": {
            "max_parameters": int(max_parameters),
            "max_int8_mib": float(max_int8_mib),
            "max_fp32_probability_error": float(max_fp32_probability_error),
            "max_macro_f1_drop": float(max_macro_f1_drop),
            "max_class1_f1_drop": float(max_class1_f1_drop),
            "min_int8_macro_f1": float(min_int8_macro_f1),
            "min_int8_class1_f1": float(min_int8_class1_f1),
            "min_int8_class1_precision": float(min_int8_class1_precision),
            "min_int8_class1_recall": float(min_int8_class1_recall),
            "min_qdq_target_op_coverage": float(min_qdq_target_op_coverage),
        },
        "remaining_certification": (
            "Measure warm/cold p95 latency, FPS, peak RAM, package install size, "
            "energy and thermal behavior on the named physical mobile device/runtime."
        ),
    }


def resolve_trtexec_path(explicit_path: Optional[Path] = None) -> Optional[str]:
    candidates = []
    if explicit_path is not None:
        candidates.append(str(explicit_path))

    fallback = Path(r"C:\TensorRT\TensorRT-10.7.0.23\bin\trtexec.exe")
    if fallback.is_file():
        candidates.append(str(fallback))

    path_hit = shutil.which("trtexec")
    if path_hit:
        candidates.append(path_hit)

    for candidate in candidates:
        if candidate:
            return candidate
    return None


if trt is not None:  # pragma: no branch
    class TensorRTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(
            self,
            calibration_batches: Sequence[torch.Tensor],
            cache_path: Optional[Path] = None,
        ) -> None:
            super().__init__()
            if not calibration_batches:
                raise ValueError("TensorRT INT8 PTQ can it nhat 1 calibration batch.")
            self.cache_path = Path(cache_path) if cache_path is not None else None
            self.batches = [
                batch.detach().cpu().to(torch.float32).contiguous() for batch in calibration_batches
            ]
            self.batch_size = int(self.batches[0].shape[0])
            self.batch_index = 0
            self.device_input = torch.empty(
                tuple(int(dim) for dim in self.batches[0].shape),
                device="cuda",
                dtype=torch.float32,
            )

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):  # pragma: no cover
            if self.batch_index >= len(self.batches):
                return None
            batch = self.batches[self.batch_index].to(device="cuda", non_blocking=False)
            if tuple(batch.shape) != tuple(self.device_input.shape):
                self.device_input = torch.empty_like(batch, device="cuda")
            self.device_input.copy_(batch)
            self.batch_index += 1
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):  # pragma: no cover
            # Never reuse an opaque cache: it cannot prove which ONNX graph,
            # checkpoint, dataset inventory or stratified indices produced it.
            # The configured path is write-only evidence for this invocation.
            return None

        def write_calibration_cache(self, cache):  # pragma: no cover
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_bytes(cache)


def collect_calibration_batches(loader: DataLoader, max_samples: int) -> tuple[List[torch.Tensor], int]:
    batches: List[torch.Tensor] = []
    collected = 0
    target_samples = max(1, int(max_samples))
    for batch in loader:
        images = batch[0]
        if collected >= target_samples:
            break
        remaining = target_samples - collected
        batch = images[:remaining].detach().cpu().to(torch.float32).contiguous()
        if batch.numel() == 0:
            continue
        batches.append(batch)
        collected += int(batch.shape[0])
    return batches, collected


def build_trt_engine_fp16(
    onnx_path: Path,
    output_path: Path,
    input_shape: Optional[
        Union[Sequence[int], Mapping[str, Sequence[int]]]
    ] = None,
    trtexec_path: Optional[Path] = None,
) -> Optional[Path]:
    executable = resolve_trtexec_path(trtexec_path)
    if executable is None:
        return None

    command_args = [
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--fp16",
        "--skipInference",
    ]
    if input_shape is not None:
        if isinstance(input_shape, Mapping):
            shape = format_trt_shapes(input_shape)
        else:
            shape = format_trt_shape(input_shape)
        command_args[2:2] = [
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes={shape}",
        ]
    if Path(executable).suffix.lower() in {".cmd", ".bat"}:
        command = ["cmd", "/c", executable, *command_args]
    else:
        command = [executable, *command_args]
    subprocess.run(command, check=True)
    return output_path


def build_trt_engine_int8(
    onnx_path: Path,
    output_path: Path,
    input_shape: Sequence[int],
    calibration_batches: Sequence[torch.Tensor],
    calibration_cache_path: Optional[Path] = None,
) -> Path:
    if trt is None:
        raise ImportError("Khong tim thay tensorrt Python package de build INT8 PTQ engine.")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT INT8 PTQ can GPU CUDA.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    model_bytes = onnx_path.read_bytes()
    if not parser.parse(model_bytes):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("Khong parse duoc ONNX de build INT8 engine:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if getattr(builder, "platform_has_fast_fp16", False):
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    fixed_shape = tuple(int(dim) for dim in input_shape)
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)

    calibrator = TensorRTEntropyCalibrator(
        calibration_batches=calibration_batches,
        cache_path=calibration_cache_path,
    )
    config.int8_calibrator = calibrator
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT khong build duoc serialized INT8 engine.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized_engine))
    return output_path


def main() -> None:
    args = parse_args()
    if args.onnx_static_int8 and args.skip_onnx:
        raise ValueError("--onnx-static-int8 cannot be combined with --skip-onnx.")
    if int(args.onnx_static_int8_calibration_samples) <= 0:
        raise ValueError("--onnx-static-int8-calibration-samples must be positive.")
    if args.require_mobile_proxy_gates and not args.onnx_static_int8:
        raise ValueError("--require-mobile-proxy-gates requires --onnx-static-int8.")
    if args.require_mobile_proxy_gates and args.skip_accuracy:
        raise ValueError("Mobile proxy gates require full validation accuracy.")
    if int(args.mobile_proxy_max_parameters) <= 0:
        raise ValueError("--mobile-proxy-max-parameters must be positive.")
    for name in (
        "mobile_proxy_max_fp32_probability_error",
        "mobile_proxy_max_macro_f1_drop",
        "mobile_proxy_max_class1_f1_drop",
        "mobile_proxy_max_int8_mib",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    for name in (
        "mobile_proxy_min_int8_macro_f1",
        "mobile_proxy_min_int8_class1_f1",
        "mobile_proxy_min_int8_class1_precision",
        "mobile_proxy_min_int8_class1_recall",
        "mobile_proxy_min_qdq_target_op_coverage",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    set_seed(args.seed, deterministic=False)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    validate_tensorrt_export_policy(
        checkpoint,
        skip_trt_engine=bool(args.skip_trt_engine or args.skip_onnx),
        allow_uncertified_precision_ensemble_tensorrt=(
            args.allow_uncertified_precision_ensemble_tensorrt
        ),
    )
    image_size = int(args.override_image_size or checkpoint["model_config"].get("image_size", 224))
    temporal_frames = resolve_temporal_frames(checkpoint)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.resolve().parent.parent / "deploy"
    ensure_dir(output_dir)

    export_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model = build_runtime_model_from_checkpoint(
        checkpoint=checkpoint,
        device=export_device,
        override_image_size=args.override_image_size,
    )
    parameter_count = sum(int(parameter.numel()) for parameter in fp32_model.parameters())

    export_input_shape = resolve_input_shape(
        image_size=image_size,
        batch_size=1,
        temporal_frames=temporal_frames,
    )
    benchmark_input_shape = resolve_input_shape(
        image_size=image_size,
        batch_size=args.benchmark_batch_size,
        temporal_frames=temporal_frames,
    )
    export_input_shapes = resolve_runtime_input_shapes(fp32_model, export_input_shape)
    benchmark_input_shapes = resolve_runtime_input_shapes(
        fp32_model,
        benchmark_input_shape,
    )
    requires_spatial_metadata = bool(
        getattr(fp32_model, "requires_spatial_metadata", False)
    )
    supports_dynamic_batch = bool(
        getattr(fp32_model, "supports_dynamic_batch", True)
    )
    supports_torchscript_deployment = bool(
        getattr(fp32_model, "supports_torchscript_deployment", True)
    )
    if (
        not supports_dynamic_batch
        and not args.skip_benchmark
        and int(args.benchmark_batch_size) != 1
    ):
        raise ValueError(
            "This checkpoint has a fixed batch-1 deployment contract; "
            "set --benchmark-batch-size 1."
        )

    fp32_torchscript_path = None
    if supports_torchscript_deployment:
        fp32_torchscript_path = export_torchscript(
            model=fp32_model,
            input_shape=export_input_shape,
            output_path=output_dir / "model_fp32_torchscript.pt",
        )
    int8_model = None
    int8_torchscript_path = None
    if not requires_spatial_metadata:
        int8_model, int8_torchscript_path = export_dynamic_int8(
            model=fp32_model,
            input_shape=export_input_shape,
            output_path=output_dir / "model_int8_dynamic_torchscript.pt",
        )

    onnx_path = None
    if not args.skip_onnx:
        onnx_path = export_onnx(
            model=fp32_model,
            checkpoint={
                **checkpoint,
                "model_config": {
                    **checkpoint["model_config"],
                    "image_size": image_size,
                },
            },
            output_path=output_dir / "model_fp32.onnx",
        )

    onnx_static_int8_path = None
    onnx_static_int8_summary: Optional[Dict[str, object]] = None
    onnx_static_int8_calibration_samples_used = 0
    onnx_static_int8_calibration_sampling: Dict[str, object] = {}
    if args.onnx_static_int8:
        if requires_spatial_metadata or temporal_frames > 1:
            raise NotImplementedError(
                "Static mobile QDQ currently supports one-input image classifiers only."
            )
        if onnx_path is None:
            raise RuntimeError("FP32 ONNX export is required before static INT8 quantization.")
        from trkh.inference.mobile_onnx_quantization import (
            CalibrationDataReader,
            quantize_mobile_onnx_qdq,
        )

        static_calibration_loader, _ = build_eval_loader(
            checkpoint=checkpoint,
            data_yaml=args.data,
            batch_size=max(1, int(args.batch_size)),
            num_workers=args.num_workers,
            split="train",
            override_image_size=args.override_image_size,
            class_name_mode=args.class_name_mode,
            expected_num_classes=args.expected_num_classes or None,
            allow_test_split=False,
            stratified_max_samples=int(
                args.onnx_static_int8_calibration_samples
            ),
            sampling_seed=int(args.seed),
        )
        onnx_static_int8_calibration_sampling = dict(
            getattr(
                static_calibration_loader,
                "trkh_sampling_summary",
                {},
            )
        )
        static_calibration_batches, onnx_static_int8_calibration_samples_used = (
            collect_calibration_batches(
                static_calibration_loader,
                max_samples=args.onnx_static_int8_calibration_samples,
            )
        )
        static_reader = CalibrationDataReader(
            [
                {"images": batch.numpy()}
                for batch in static_calibration_batches
            ],
            calibration_split="train",
            provenance={
                "dataset": str(Path(args.data).resolve()),
                "dataset_split": "train",
                "test_used": False,
                "samples_requested": int(args.onnx_static_int8_calibration_samples),
                "samples_used": int(onnx_static_int8_calibration_samples_used),
                "sampling": onnx_static_int8_calibration_sampling,
                "checkpoint": str(Path(args.checkpoint).resolve()),
            },
        )
        onnx_static_int8_path = output_dir / "model_int8_qdq.onnx"
        onnx_static_int8_summary = quantize_mobile_onnx_qdq(
            onnx_path,
            onnx_static_int8_path,
            static_reader,
            provenance={
                "purpose": "trkh_mobile_deployment_proxy",
                "calibration_split": "train",
                "test_used": False,
                "calibration_sampling": onnx_static_int8_calibration_sampling,
            },
            overwrite=True,
            require_size_reduction=True,
            min_target_op_coverage=(
                args.mobile_proxy_min_qdq_target_op_coverage
            ),
        )

    trt_engine_path = None
    trt_int8_engine_path = None
    trt_calibration_samples_used = 0
    trt_calibration_sampling: Dict[str, object] = {}
    calibration_cache_path = None
    if onnx_path is not None and not args.skip_trt_engine:
        trt_engine_path = build_trt_engine_fp16(
            onnx_path=onnx_path,
            output_path=output_dir / "model_fp32_fp16.engine",
            input_shape=export_input_shapes if supports_dynamic_batch else None,
            trtexec_path=args.trtexec_path,
        )
        if args.trt_int8:
            if requires_spatial_metadata:
                raise NotImplementedError(
                    "INT8 calibration for multi-input spatial metadata is not implemented."
                )
            calibration_loader, trt_calibration_sampling = build_eval_loader(
                checkpoint=checkpoint,
                data_yaml=args.data,
                batch_size=export_input_shape[0],
                num_workers=args.num_workers,
                split=args.trt_calibration_split,
                override_image_size=args.override_image_size,
                class_name_mode=args.class_name_mode,
                expected_num_classes=args.expected_num_classes or None,
                allow_test_split=False,
                stratified_max_samples=int(args.trt_calibration_samples),
                sampling_seed=int(args.seed),
            )
            calibration_batches, trt_calibration_samples_used = collect_calibration_batches(
                calibration_loader,
                max_samples=args.trt_calibration_samples,
            )
            calibration_cache_path = (
                args.trt_calibration_cache
                if args.trt_calibration_cache is not None
                else output_dir / "model_int8.calib"
            )
            trt_int8_engine_path = build_trt_engine_int8(
                onnx_path=onnx_path,
                output_path=output_dir / "model_int8_ptq.engine",
                input_shape=export_input_shape,
                calibration_batches=calibration_batches,
                calibration_cache_path=calibration_cache_path,
            )

    payload: Dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "parameter_count": int(parameter_count),
        "image_size": image_size,
        "temporal_frames": temporal_frames,
        "input_shape": list(export_input_shape),
        "input_shapes": {
            name: list(shape) for name, shape in export_input_shapes.items()
        },
        "requires_spatial_metadata": requires_spatial_metadata,
        "supports_dynamic_batch": supports_dynamic_batch,
        "certified_batch_sizes": list(
            getattr(fp32_model, "certified_batch_sizes", ())
        ),
        "supports_torchscript_deployment": supports_torchscript_deployment,
        "exports": {
            "torchscript_fp32": (
                str(fp32_torchscript_path.resolve())
                if fp32_torchscript_path is not None
                else None
            ),
            "torchscript_int8_dynamic": (
                str(int8_torchscript_path.resolve())
                if int8_torchscript_path is not None
                else None
            ),
            "onnx_fp32": str(onnx_path.resolve()) if onnx_path is not None else None,
            "onnx_int8_qdq": (
                str(onnx_static_int8_path.resolve())
                if onnx_static_int8_path is not None
                else None
            ),
            "tensorrt_fp16": str(trt_engine_path.resolve()) if trt_engine_path is not None else None,
            "tensorrt_int8_ptq": (
                str(trt_int8_engine_path.resolve()) if trt_int8_engine_path is not None else None
            ),
        },
    }
    if args.trt_int8:
        payload["trt_int8"] = {
            "enabled": True,
            "calibration_split": args.trt_calibration_split,
            "calibration_samples_requested": int(args.trt_calibration_samples),
            "calibration_samples_used": int(trt_calibration_samples_used),
            "calibration_sampling": trt_calibration_sampling,
            "calibration_cache": (
                str(calibration_cache_path.resolve()) if calibration_cache_path is not None else None
            ),
        }
    else:
        payload["trt_int8"] = {"enabled": False}
    payload["onnx_static_int8"] = (
        {
            "enabled": True,
            "calibration_split": "train",
            "calibration_samples_requested": int(
                args.onnx_static_int8_calibration_samples
            ),
            "calibration_samples_used": int(
                onnx_static_int8_calibration_samples_used
            ),
            "calibration_sampling": onnx_static_int8_calibration_sampling,
            "quantization": onnx_static_int8_summary,
        }
        if onnx_static_int8_summary is not None
        else {"enabled": False}
    )

    loader = None
    class_names = None
    if not args.skip_accuracy:
        loader, class_names = build_eval_loader(
            checkpoint=checkpoint,
            data_yaml=args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split=args.split,
            override_image_size=args.override_image_size,
            class_name_mode=args.class_name_mode,
            expected_num_classes=args.expected_num_classes or None,
            allow_test_split=bool(args.allow_test_split),
        )
        fp32_cpu = build_runtime_model_from_checkpoint(
            checkpoint=checkpoint,
            device=torch.device("cpu"),
            override_image_size=args.override_image_size,
        )
        payload["accuracy"] = {
            "fp32_cpu": to_serializable(
                evaluate_accuracy(fp32_cpu, loader, class_names, torch.device("cpu"))
            ),
        }
        if int8_model is not None:
            payload["accuracy"]["int8_dynamic_cpu"] = to_serializable(
                evaluate_accuracy(int8_model, loader, class_names, torch.device("cpu"))
            )
        if onnx_static_int8_path is not None and onnx_path is not None:
            mobile_evaluation = evaluate_mobile_onnx_pair(
                reference_model=fp32_cpu,
                fp32_onnx_path=onnx_path,
                int8_onnx_path=onnx_static_int8_path,
                loader=loader,
                class_names=class_names,
            )
            payload["accuracy"]["mobile_onnx_aligned"] = to_serializable(
                mobile_evaluation
            )
            payload["mobile_proxy_gate"] = build_mobile_proxy_gate(
                evaluation=mobile_evaluation,
                quantization=onnx_static_int8_summary or {},
                parameter_count=parameter_count,
                max_fp32_probability_error=(
                    args.mobile_proxy_max_fp32_probability_error
                ),
                max_macro_f1_drop=args.mobile_proxy_max_macro_f1_drop,
                max_class1_f1_drop=args.mobile_proxy_max_class1_f1_drop,
                max_int8_mib=args.mobile_proxy_max_int8_mib,
                max_parameters=args.mobile_proxy_max_parameters,
                min_int8_macro_f1=args.mobile_proxy_min_int8_macro_f1,
                min_int8_class1_f1=args.mobile_proxy_min_int8_class1_f1,
                min_int8_class1_precision=(
                    args.mobile_proxy_min_int8_class1_precision
                ),
                min_int8_class1_recall=(
                    args.mobile_proxy_min_int8_class1_recall
                ),
                min_qdq_target_op_coverage=(
                    args.mobile_proxy_min_qdq_target_op_coverage
                ),
            )

    if not args.skip_benchmark:
        payload["benchmark"] = {
            "fp32_cpu": benchmark_torch_module(
                model=build_runtime_model_from_checkpoint(
                    checkpoint=checkpoint,
                    device=torch.device("cpu"),
                    override_image_size=args.override_image_size,
                ),
                input_shape=benchmark_input_shape,
                device=torch.device("cpu"),
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
            ),
        }
        if int8_model is not None:
            payload["benchmark"]["int8_dynamic_cpu"] = benchmark_torch_module(
                model=int8_model,
                input_shape=benchmark_input_shape,
                device=torch.device("cpu"),
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
            )
        if export_device.type == "cuda":
            payload["benchmark"]["fp32_cuda"] = benchmark_torch_module(
                model=build_runtime_model_from_checkpoint(
                    checkpoint=checkpoint,
                    device=export_device,
                    override_image_size=args.override_image_size,
                ),
                input_shape=benchmark_input_shape,
                device=export_device,
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
            )
        if onnx_path is not None:
            payload["benchmark"]["onnx_cpu"] = benchmark_onnx_model(
                onnx_path=onnx_path,
                input_shapes=benchmark_input_shapes,
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
            )
        if onnx_static_int8_path is not None:
            payload["benchmark"]["onnx_int8_qdq_cpu"] = benchmark_onnx_model(
                onnx_path=onnx_static_int8_path,
                input_shapes=benchmark_input_shapes,
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
            )

    json_dump(output_dir / "deploy_summary.json", to_serializable(payload))
    print(json.dumps(to_serializable(payload), ensure_ascii=False))
    if args.require_mobile_proxy_gates:
        gate = payload.get("mobile_proxy_gate", {})
        if not isinstance(gate, Mapping) or gate.get("status") != "passed":
            raise RuntimeError(
                "Mobile desktop-proxy gate failed; inspect deploy_summary.json. "
                "No target-device certification was claimed."
            )


if __name__ == "__main__":
    main()
