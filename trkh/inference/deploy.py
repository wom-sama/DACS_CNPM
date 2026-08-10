from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
from torch.utils.data import DataLoader

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    PseudoVideoAugmenter,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.evaluate import evaluate_model, resolve_crop_to_primary_object
from trkh.inference.inference import export_onnx
from trkh.models.model import build_model_from_checkpoint as build_checkpoint_model
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
    parser.add_argument("--trt-calibration-split", choices=("auto", "train", "val", "test"), default="train")
    parser.add_argument("--trt-calibration-cache", type=Path, default=None)
    parser.add_argument("--skip-accuracy", action="store_true", default=False)
    parser.add_argument("--skip-benchmark", action="store_true", default=False)
    parser.add_argument("--split", choices=("auto", "train", "val", "test"), default="auto")
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
) -> tuple[DataLoader, list[str]]:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes,
    )
    class_names = checkpoint["class_names"]
    image_size = int(override_image_size or checkpoint["model_config"].get("image_size", 224))
    temporal_frames = resolve_temporal_frames(checkpoint)
    resolved_split = "test" if split == "auto" and data_spec.has_test_split else split
    if resolved_split == "auto":
        resolved_split = "val"
    precision_ensemble = str(
        checkpoint.get("model_config", {}).get("model_type", "")
    ).strip().lower() == "precision_ensemble"
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=resolved_split,
        transform=build_runtime_transform(checkpoint, image_size=image_size, temporal_frames=temporal_frames),
        crop_margin_ratio=float(
            checkpoint.get("augmentation_config", {}).get("crop_margin_ratio", 0.05)
        ),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=precision_ensemble,
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    return loader, class_names


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
            if self.cache_path is not None and self.cache_path.is_file():
                return self.cache_path.read_bytes()
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

    trt_engine_path = None
    trt_int8_engine_path = None
    calibration_samples_used = 0
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
            calibration_loader, _ = build_eval_loader(
                checkpoint=checkpoint,
                data_yaml=args.data,
                batch_size=export_input_shape[0],
                num_workers=args.num_workers,
                split=args.trt_calibration_split,
                override_image_size=args.override_image_size,
            )
            calibration_batches, calibration_samples_used = collect_calibration_batches(
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
            "calibration_samples_used": int(calibration_samples_used),
            "calibration_cache": (
                str(calibration_cache_path.resolve()) if calibration_cache_path is not None else None
            ),
        }
    else:
        payload["trt_int8"] = {"enabled": False}

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

    json_dump(output_dir / "deploy_summary.json", to_serializable(payload))
    print(json.dumps(to_serializable(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
