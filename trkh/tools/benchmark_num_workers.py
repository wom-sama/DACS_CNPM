from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import autocast_context, build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    build_train_collate_fn,
    build_train_transform,
)
from trkh.models.model import create_model
from trkh.training.losses import FocalCrossEntropyLoss


def _parse_workers(raw: str) -> List[int]:
    workers: List[int] = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(max(0, int(part)))
    return workers or [0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader num_workers for classification-only training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--class-name-mode", default="auto")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--workers", default="0,2,4,6,8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-registers", type=int, default=4)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--drop-path-rate", type=float, default=0.10)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measure-batches", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--disable-pin-memory", action="store_true", default=False)
    parser.add_argument("--disable-persistent-workers", action="store_true", default=False)
    parser.add_argument("--allow-windows-workers", action="store_true", default=True)
    parser.add_argument("--no-allow-windows-workers", dest="allow_windows_workers", action="store_false")
    return parser.parse_args()


def benchmark_worker_count(
    args: argparse.Namespace,
    dataset: Union[MangoYOLOCropDataset, ClassificationFolderDataset],
    worker_count: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda" and not args.disable_pin_memory
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=worker_count,
        requested_pin_memory=pin_memory,
        context=f"benchmark_workers_{worker_count}",
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=build_train_collate_fn(num_classes=max(1, int(dataset.num_classes or 1)), batch_mix_probability=0.0),
        **dataloader_kwargs,
    )
    model = create_model(
        num_classes=max(1, int(dataset.num_classes or 1)),
        model_config={
            "model_type": "vit_registers",
            "image_size": int(args.image_size),
            "patch_size": int(args.patch_size),
            "use_cnn_stem": True,
            "stem_channels": int(args.stem_channels),
            "embed_dim": int(args.embed_dim),
            "depth": int(args.depth),
            "num_heads": int(args.num_heads),
            "num_registers": int(args.num_registers),
            "dropout": float(args.dropout),
            "drop_path_rate": float(args.drop_path_rate),
            "head_pooling": "cls_register_mean",
        },
    ).to(device)
    model.train()
    criterion = FocalCrossEntropyLoss(gamma=2.0, focal_mix=0.20, label_smoothing=0.015).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    start = time.perf_counter()
    total_batches = max(1, int(args.warmup_batches) + int(args.measure_batches))
    total_batch_times: List[float] = []
    data_wait_times: List[float] = []
    compute_times: List[float] = []
    measured_samples = 0

    iterator = iter(loader)
    for batch_index in range(total_batches):
        batch_total_start = time.perf_counter()
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        after_data = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, True):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_compute = time.perf_counter()
        if batch_index >= int(args.warmup_batches):
            data_wait_times.append(after_data - batch_total_start)
            compute_times.append(after_compute - compute_start)
            total_batch_times.append(after_compute - batch_total_start)
            measured_samples += int(images.size(0))

    total_seconds = time.perf_counter() - start
    mean_batch_seconds = sum(total_batch_times) / max(1, len(total_batch_times))
    mean_data_wait_seconds = sum(data_wait_times) / max(1, len(data_wait_times))
    mean_compute_seconds = sum(compute_times) / max(1, len(compute_times))
    samples_per_second = measured_samples / max(1e-12, sum(total_batch_times))
    return {
        "requested_workers": int(worker_count),
        "effective_workers": int(dataloader_summary["effective_num_workers"]),
        "pin_memory": bool(dataloader_summary["effective_pin_memory"]),
        "persistent_workers": bool(dataloader_summary["persistent_workers"]),
        "prefetch_factor": dataloader_summary.get("prefetch_factor"),
        "mean_batch_seconds": float(mean_batch_seconds),
        "mean_data_wait_seconds": float(mean_data_wait_seconds),
        "mean_compute_seconds": float(mean_compute_seconds),
        "data_wait_fraction": float(mean_data_wait_seconds / max(1e-12, mean_batch_seconds)),
        "samples_per_second": float(samples_per_second),
        "measured_batches": int(len(total_batch_times)),
        "measured_samples": int(measured_samples),
        "total_seconds_including_startup": float(total_seconds),
        "notes": dataloader_summary.get("notes", []),
    }


def main() -> None:
    args = parse_args()
    if args.allow_windows_workers:
        os.environ.setdefault("TRKH_ALLOW_WINDOWS_MULTIPROCESSING", "1")
        os.environ.setdefault("TRKH_ALLOW_WINDOWS_PIN_MEMORY", "1")
        os.environ.setdefault("TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS", "1")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    set_seed(int(args.seed), deterministic=False)
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    transform = build_train_transform(
        image_size=int(args.image_size),
        resize_mode="pad",
        scale_min=0.8,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        random_erasing_probability=0.0,
        random_affine_degrees=4.0,
        random_affine_translate=0.03,
        random_affine_scale_min=0.95,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.01,
        rotate90_probability=0.03,
        lighting_probability=0.0,
        randaugment_num_ops=0,
        randaugment_magnitude=0,
    )
    if data_spec.data_format == "classification_folder":
        dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=transform,
            class_aware_augmentation=True,
            class_augmentation_power=0.5,
            class_augmentation_max_scale=3.0,
        )
    else:
        dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=transform,
            crop_margin_ratio=0.08,
            crop_to_primary_object=True,
            classification_target=True,
            classification_object_crops=True,
            class_aware_augmentation=True,
            class_augmentation_power=0.5,
            class_augmentation_max_scale=3.0,
        )
    if len(dataset) == 0:
        raise ValueError("Dataset benchmark khong co sample hop le.")
    print(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataset_samples": len(dataset),
            "class_counts": dataset.class_counts(data_spec.num_classes),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "workers": _parse_workers(args.workers),
        },
        flush=True,
    )
    results = []
    for worker_count in _parse_workers(args.workers):
        result = benchmark_worker_count(args, dataset, worker_count)
        results.append(result)
        print(result, flush=True)
    best = max(results, key=lambda item: float(item["samples_per_second"]))
    print({"best_by_samples_per_second": best}, flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
