from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.tools.audit_multistage_teacher_feature_readiness import (
    _alignment_plan,
    _class1_error_direction_auc,
    _compact_alignment,
    _heat_overlay,
    _imagefolder_rows,
    _load_base_predictions,
    _projected_effective_rank,
)
from trkh.tools.audit_wavelet_scattering_readiness import (
    PCA_COMPONENTS,
    PRIMARY_READOUT,
    READOUT_NAMES,
    _assert_output_outside_datasets,
    _binary_focus_oracle,
    _compact_readout,
    _write_readout_metrics,
    fit_grouped_scattering_readouts,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
    _write_prediction_audit,
)


MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
IMAGE_SIZE = 224
PATCH_SIZE = 14
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE
SPATIAL_POOL_SIZE = 2
SEED = 20260712
FEATURE_NAMES = ("cls", "cls_plus_dense_patch")
LITERATURE = (
    "https://arxiv.org/abs/2304.07193",
    "https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md",
    "https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py",
    "https://github.com/facebookresearch/dinov2",
    "https://arxiv.org/abs/2111.07832",
)


class PathImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, int(target), str(self.samples[int(index)][0])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only readiness audit for DINOv2 dense patch "
            "surface semantics against a matched CLS-token control."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--base-train-csv", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--save-descriptors", action="store_true", default=False)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def dinov2_dense_patch_descriptors(
    tokens: Tensor,
    *,
    grid_size: Tuple[int, int],
    prefix_tokens: int = 1,
    spatial_pool_size: int = SPATIAL_POOL_SIZE,
) -> Dict[str, Tensor]:
    """Build a fixed CLS control and a spatial patch-moment candidate."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,D], got {tuple(tokens.shape)}")
    grid_height, grid_width = (int(grid_size[0]), int(grid_size[1]))
    prefix_count = int(prefix_tokens)
    pool_size = int(spatial_pool_size)
    if grid_height < 1 or grid_width < 1 or prefix_count < 1 or pool_size < 1:
        raise ValueError("grid_size, prefix_tokens, and spatial_pool_size must be positive")
    expected_tokens = prefix_count + grid_height * grid_width
    if int(tokens.size(1)) != expected_tokens:
        raise ValueError(
            f"Token/grid mismatch: tokens={tokens.size(1)}, expected={expected_tokens}"
        )

    values = tokens.float()
    cls = values[:, 0]
    patches = values[:, prefix_count:]
    patch_mean = patches.mean(dim=1)
    patch_std = patches.std(dim=1, unbiased=False)
    patch_grid = patches.reshape(
        int(values.size(0)), grid_height, grid_width, int(values.size(2))
    ).permute(0, 3, 1, 2)
    spatial_means = F.adaptive_avg_pool2d(
        patch_grid,
        output_size=(pool_size, pool_size),
    ).flatten(1)
    dense_patch = torch.cat((patch_mean, patch_std, spatial_means), dim=1)
    return {
        "cls": cls,
        "dense_patch": dense_patch,
        "cls_plus_dense_patch": torch.cat((cls, dense_patch), dim=1),
    }


def _patch_energy_map(
    tokens: Tensor,
    *,
    grid_size: Tuple[int, int],
    prefix_tokens: int,
) -> Tensor:
    patches = tokens.float()[:, int(prefix_tokens) :]
    centered = patches - patches.mean(dim=1, keepdim=True)
    energy = centered.square().mean(dim=2)
    return energy.reshape(int(tokens.size(0)), int(grid_size[0]), int(grid_size[1]))


def _energy_statistics(energy: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    values = energy.detach().float().clamp_min(0.0)
    flat = values.flatten(1)
    probabilities = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(max(2, int(flat.size(1))))
    height, width = int(values.size(1)), int(values.size(2))
    border = torch.zeros((height, width), dtype=torch.bool, device=values.device)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_mass = probabilities[:, border.flatten()].sum(dim=1)
    return (
        entropy.cpu().numpy().astype(np.float32, copy=False),
        border_mass.cpu().numpy().astype(np.float32, copy=False),
    )


def _build_alignment(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    plans: Dict[str, Dict[str, object]] = {}
    classes_by_split: Dict[str, List[str]] = {}
    for split in ("train", "val"):
        rows, classes = _imagefolder_rows(Path(args.classification_root), split)
        plans[split] = _alignment_plan(
            classification_rows=rows,
            classification_classes=classes,
            yolo_data=Path(args.yolo_data),
            split=split,
            class_name_mode=str(args.class_name_mode),
        )
        classes_by_split[split] = classes
    if list(plans["train"]["class_names"]) != list(plans["val"]["class_names"]):
        raise ValueError("Train/validation target class order differs")
    return plans, classes_by_split


def _build_model_and_transform(
    model_name: str,
    *,
    device: torch.device,
) -> Tuple[torch.nn.Module, object, Dict[str, object]]:
    model = timm.create_model(
        str(model_name),
        pretrained=True,
        num_classes=0,
        img_size=IMAGE_SIZE,
    )
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    grid_size = tuple(int(value) for value in model.patch_embed.grid_size)
    if grid_size != (PATCH_GRID, PATCH_GRID):
        raise ValueError(f"Unexpected DINOv2 patch grid: {grid_size}")
    if prefix_tokens != 1:
        raise ValueError(f"This locked protocol requires one CLS prefix token, got {prefix_tokens}")
    config = timm.data.resolve_model_data_config(model)
    mean = tuple(float(value) for value in config["mean"])
    std = tuple(float(value) for value in config["std"])
    transform = v2.Compose(
        (
            v2.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        )
    )
    model.to(device).eval()
    metadata = {
        "model_name": str(model_name),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "embedding_dimension": int(getattr(model, "num_features")),
        "prefix_tokens": prefix_tokens,
        "patch_size": PATCH_SIZE,
        "patch_grid": list(grid_size),
        "mean": list(mean),
        "std": list(std),
        "weight_url": str(getattr(model, "default_cfg", {}).get("url", "")),
        "timm_version": str(timm.__version__),
        "torch_version": str(torch.__version__),
    }
    return model, transform, metadata


def _extract_split(
    *,
    model: torch.nn.Module,
    transform,
    classification_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    model_metadata: Mapping[str, object],
) -> Dict[str, object]:
    dataset = PathImageFolder(Path(classification_root) / split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(workers) > 0),
    )
    mean = torch.tensor(model_metadata["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(model_metadata["std"], dtype=torch.float32).view(1, 3, 1, 1)
    grid_size = tuple(int(value) for value in model_metadata["patch_grid"])
    prefix_tokens = int(model_metadata["prefix_tokens"])
    feature_batches: Dict[str, List[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    labels: List[np.ndarray] = []
    paths: List[str] = []
    entropy_batches: List[np.ndarray] = []
    border_batches: List[np.ndarray] = []
    preview: Dict[int, Dict[str, object]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"dinov2-dense-{split}", dynamic_ncols=True)
        for images, targets, batch_paths in iterator:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, bool(amp)):
                tokens = model.forward_features(images)
            if not torch.is_tensor(tokens):
                raise TypeError(f"Expected TIMM token tensor, got {type(tokens)!r}")
            descriptors = dinov2_dense_patch_descriptors(
                tokens,
                grid_size=grid_size,
                prefix_tokens=prefix_tokens,
            )
            for name in FEATURE_NAMES:
                feature_batches[name].append(
                    descriptors[name].cpu().numpy().astype(np.float32, copy=False)
                )
            target_array = targets.numpy().astype(np.int64, copy=False)
            labels.append(target_array)
            paths.extend(str(path) for path in batch_paths)
            energy = _patch_energy_map(
                tokens,
                grid_size=grid_size,
                prefix_tokens=prefix_tokens,
            )
            entropy, border = _energy_statistics(energy)
            entropy_batches.append(entropy)
            border_batches.append(border)
            for row_index, target in enumerate(target_array.tolist()):
                class_index = int(target)
                if class_index in preview:
                    continue
                rgb = (
                    images[row_index].detach().float().cpu().unsqueeze(0) * std + mean
                ).squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
                preview[class_index] = {
                    "rgb": (rgb * 255.0).round().astype(np.uint8),
                    "path": str(batch_paths[row_index]),
                    "energy": energy[row_index].cpu().numpy().astype(np.float32, copy=False),
                }
    if not labels:
        raise ValueError(f"Empty split: {classification_root / split}")
    return {
        **{
            name: np.concatenate(feature_batches[name], axis=0).astype(
                np.float32, copy=False
            )
            for name in FEATURE_NAMES
        },
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "folder_classes": list(dataset.classes),
        "preview": preview,
        "energy_stats": {
            "entropy_mean": float(np.concatenate(entropy_batches).mean()),
            "entropy_p10": float(np.quantile(np.concatenate(entropy_batches), 0.10)),
            "border_mass_mean": float(np.concatenate(border_batches).mean()),
            "border_mass_p90": float(np.quantile(np.concatenate(border_batches), 0.90)),
        },
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        ),
    }


def _align_payload(
    payload: Mapping[str, object],
    plan: Mapping[str, object],
) -> Dict[str, object]:
    indices = np.asarray(plan["row_indices"], dtype=np.int64)
    aligned = {
        name: np.asarray(payload[name], dtype=np.float32)[indices]
        for name in FEATURE_NAMES
    }
    aligned.update(
        {
            "labels": np.asarray(plan["labels"], dtype=np.int64),
            "sample_index": np.asarray(plan["sample_index"], dtype=np.int64),
            "paths": np.asarray(plan["paths"], dtype=object),
            "source_stems": np.asarray(plan["source_stems"], dtype=object),
            "class_names": list(plan["class_names"]),
            "folder_classes": list(payload["folder_classes"]),
            "preview": payload["preview"],
            "energy_stats": payload["energy_stats"],
            "peak_cuda_memory_mib": payload["peak_cuda_memory_mib"],
        }
    )
    return aligned


def _write_preview(
    path: Path,
    *,
    preview: Mapping[int, Mapping[str, object]],
    folder_classes: Sequence[str],
) -> None:
    tile = 192
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index, class_name in enumerate(folder_classes):
        item = preview.get(int(class_index))
        if item is None:
            continue
        rgb = np.asarray(item["rgb"], dtype=np.uint8)
        overlay = _heat_overlay(rgb, np.asarray(item["energy"], dtype=np.float32))
        row = Image.new("RGB", (tile * 2, tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        row.paste(
            Image.fromarray(overlay).resize((tile, tile), resampling.BILINEAR),
            (tile, 0),
        )
        rows.append(row)
        manifest.append(
            {
                "row": int(len(rows) - 1),
                "class_index": int(class_index),
                "class_name": str(class_name),
                "path": str(item["path"]),
                "columns": ["rgb", "dinov2_patch_deviation_energy"],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 2, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def assess_dinov2_dense_patch_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_groups: int,
    dense_effective_rank: float,
    descriptor_finite: bool,
    energy_entropy: float,
    energy_border_mass: float,
    direct_val_metrics: Mapping[str, object],
    control_oof_metrics: Mapping[str, object],
    candidate_oof_metrics: Mapping[str, object],
    control_val_metrics: Mapping[str, object],
    candidate_val_metrics: Mapping[str, object],
    binary_oracle: Mapping[str, float],
    train_direction_auc: float,
    val_direction_auc: float,
    direct_val_transitions: Mapping[str, int],
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_source_groups": 8000,
        "min_dense_effective_rank": 24.0,
        "min_energy_entropy": 0.35,
        "max_energy_border_mass": 0.65,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_val_class1_f1": 0.70,
        "min_direct_class1_gain": 0.010,
        "max_direct_macro_drop": 0.003,
        "min_binary_oracle_class1_gain": 0.02,
        "min_direction_auc": 0.60,
        "max_direction_auc_gap": 0.15,
    }
    observed = {
        "dense_effective_rank": float(dense_effective_rank),
        "energy_entropy": float(energy_entropy),
        "energy_border_mass": float(energy_border_mass),
        "oof_macro_gain": float(candidate_oof_metrics["macro_f1"])
        - float(control_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(candidate_oof_metrics["focus_f1"])
        - float(control_oof_metrics["focus_f1"]),
        "val_macro_gain": float(candidate_val_metrics["macro_f1"])
        - float(control_val_metrics["macro_f1"]),
        "val_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(control_val_metrics["focus_f1"]),
        "direct_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(direct_val_metrics["focus_f1"]),
        "direct_macro_drop": float(direct_val_metrics["macro_f1"])
        - float(candidate_val_metrics["macro_f1"]),
        "binary_oracle_class1_gain": float(binary_oracle["f1"])
        - float(direct_val_metrics["focus_f1"]),
        "train_direction_auc": float(train_direction_auc),
        "val_direction_auc": float(val_direction_auc),
        "direction_auc_gap": abs(float(train_direction_auc) - float(val_direction_auc)),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "validation_support": int(val_samples) == int(thresholds["required_val_samples"]),
        "source_group_support": int(source_groups) >= int(thresholds["min_source_groups"]),
        "descriptor_finite": bool(descriptor_finite),
        "dense_rank": observed["dense_effective_rank"]
        >= float(thresholds["min_dense_effective_rank"]),
        "spatial_energy_noncollapsed": observed["energy_entropy"]
        >= float(thresholds["min_energy_entropy"]),
        "spatial_energy_not_border_dominated": observed["energy_border_mass"]
        <= float(thresholds["max_energy_border_mass"]),
        "oof_macro_gain": observed["oof_macro_gain"]
        >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"]
        >= float(thresholds["min_oof_class1_gain"]),
        "validation_macro_gain": observed["val_macro_gain"]
        >= float(thresholds["min_val_macro_gain"]),
        "validation_class1_gain": observed["val_class1_gain"]
        >= float(thresholds["min_val_class1_gain"]),
        "validation_class1_milestone": float(candidate_val_metrics["focus_f1"])
        >= float(thresholds["min_val_class1_f1"]),
        "direct_class1_gain": observed["direct_class1_gain"]
        >= float(thresholds["min_direct_class1_gain"]),
        "direct_macro_preserved": observed["direct_macro_drop"]
        <= float(thresholds["max_direct_macro_drop"]),
        "binary_oracle_complementarity": observed["binary_oracle_class1_gain"]
        >= float(thresholds["min_binary_oracle_class1_gain"]),
        "train_error_direction": observed["train_direction_auc"]
        >= float(thresholds["min_direction_auc"]),
        "validation_error_direction": observed["val_direction_auc"]
        >= float(thresholds["min_direction_auc"]),
        "error_direction_stability": observed["direction_auc_gap"]
        <= float(thresholds["max_direction_auc_gap"]),
        "direct_validation_net_corrections": int(direct_val_transitions["corrections"])
        >= int(direct_val_transitions["harms"]),
        "direct_validation_class1_recall_protected": int(
            direct_val_transitions["class1_fn_rescued"]
        )
        >= int(direct_val_transitions["class1_tp_broken"]),
        "direct_validation_class1_fp_control": int(
            direct_val_transitions["class1_fp_removed"]
        )
        >= int(direct_val_transitions["class1_fp_created"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "dinov2_dense_patch_target_ready": ready,
        "smoke_ready": ready,
        "smoke_permission": ready,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _save_descriptor_cache(
    path: Path,
    *,
    payload: Mapping[str, object],
) -> None:
    np.savez_compressed(
        path,
        cls=np.asarray(payload["cls"], dtype=np.float32),
        cls_plus_dense_patch=np.asarray(payload["cls_plus_dense_patch"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
        paths=np.asarray(payload["paths"], dtype=object),
        source_stems=np.asarray(payload["source_stems"], dtype=object),
        class_names=np.asarray(payload["class_names"], dtype=object),
    )


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) != 5:
        raise ValueError("Protocol is locked to five source-grouped folds")
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if str(args.model) != MODEL_NAME:
        raise ValueError(
            f"Locked protocol requires --model {MODEL_NAME!r}, got {str(args.model)!r}"
        )
    for required in (
        Path(args.classification_root) / "train",
        Path(args.classification_root) / "val",
        Path(args.yolo_data),
        Path(args.base_train_csv),
        Path(args.base_val_csv),
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    _assert_output_outside_datasets(
        Path(args.output_dir),
        classification_root=Path(args.classification_root),
        yolo_data=Path(args.yolo_data),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans, classes_by_split = _build_alignment(args)
    embedding_dimension = 384
    dense_dimension = (2 + SPATIAL_POOL_SIZE * SPATIAL_POOL_SIZE) * embedding_dimension
    preflight = {
        "mode": "dinov2_dense_patch_readiness_preflight",
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "base_train_csv": str(Path(args.base_train_csv).resolve()),
        "base_val_csv": str(Path(args.base_val_csv).resolve()),
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "protocol": {
            "model": str(args.model),
            "pretrained": True,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "patch_grid": [PATCH_GRID, PATCH_GRID],
            "control": "normalized_cls_token",
            "dense_patch_signature": "patch_mean+patch_std+2x2_spatial_means",
            "control_dimension": embedding_dimension,
            "dense_patch_dimension": dense_dimension,
            "candidate_dimension": embedding_dimension + dense_dimension,
            "pca_components": PCA_COMPONENTS,
            "primary_readout": PRIMARY_READOUT,
            "readouts": list(READOUT_NAMES),
            "folds": int(args.folds),
            "seed": SEED,
            "matched_source_folds": True,
            "candidate_selection_uses_validation": False,
            "descriptor_grid_sweep": False,
            "test_selection": False,
        },
        "literature": list(LITERATURE),
        "test_split_used": False,
        "raw_dataset_touched": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if bool(args.preflight_only):
        return preflight

    start = time.perf_counter()
    device = _resolve_device(str(args.device or ""))
    model, transform, model_metadata = _build_model_and_transform(
        str(args.model), device=device
    )
    extracted: Dict[str, Dict[str, object]] = {}
    aligned: Dict[str, Dict[str, object]] = {}
    extraction_seconds: Dict[str, float] = {}
    for split in ("train", "val"):
        split_start = time.perf_counter()
        extracted[split] = _extract_split(
            model=model,
            transform=transform,
            classification_root=Path(args.classification_root),
            split=split,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            model_metadata=model_metadata,
        )
        extraction_seconds[split] = float(time.perf_counter() - split_start)
        if list(extracted[split]["folder_classes"]) != classes_by_split[split]:
            raise ValueError(f"ImageFolder class order changed for {split}")
        aligned[split] = _align_payload(extracted[split], plans[split])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train = aligned["train"]
    val = aligned["val"]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    class_names = list(train["class_names"])
    train_groups = np.asarray(train["source_stems"], dtype=object)
    descriptor_finite = all(
        np.isfinite(np.asarray(aligned[split][name], dtype=np.float32)).all()
        for split in ("train", "val")
        for name in FEATURE_NAMES
    )
    if not descriptor_finite:
        raise ValueError("DINOv2 descriptor contains non-finite values")
    dense_train = np.asarray(train["cls_plus_dense_patch"], dtype=np.float32)[
        :, int(model_metadata["embedding_dimension"]) :
    ]
    dense_standardized = StandardScaler().fit_transform(dense_train)
    dense_effective_rank = _projected_effective_rank(
        dense_standardized, seed=SEED + 991
    )
    del dense_standardized

    readout_fits: Dict[str, Dict[str, object]] = {}
    for name in FEATURE_NAMES:
        print(f"fitting matched DINOv2 readouts: {name}", flush=True)
        readout_fits[name] = fit_grouped_scattering_readouts(
            train_features=np.asarray(train[name], dtype=np.float32),
            train_labels=train_labels,
            train_groups=train_groups,
            val_features=np.asarray(val[name], dtype=np.float32),
            val_labels=val_labels,
            class_names=class_names,
            folds=int(args.folds),
            seed=SEED,
        )
        _write_readout_metrics(
            output_dir / f"{name}_readout_metrics.csv",
            readout_fits[name]["readouts"],
        )

    control = readout_fits["cls"]["readouts"][PRIMARY_READOUT]
    candidate = readout_fits["cls_plus_dense_patch"]["readouts"][PRIMARY_READOUT]
    control_train = np.asarray(control["oof_probabilities"], dtype=np.float32)
    control_val = np.asarray(control["val_probabilities"], dtype=np.float32)
    candidate_train = np.asarray(candidate["oof_probabilities"], dtype=np.float32)
    candidate_val = np.asarray(candidate["val_probabilities"], dtype=np.float32)
    base_train = _load_base_predictions(
        Path(args.base_train_csv),
        expected_split="train",
        expected_labels=train_labels,
        class_count=len(class_names),
    )
    base_val = _load_base_predictions(
        Path(args.base_val_csv),
        expected_split="val",
        expected_labels=val_labels,
        class_count=len(class_names),
    )
    direct_train_metrics = _classification_metrics(
        train_labels, base_train, class_names=class_names
    )
    direct_val_metrics = _classification_metrics(
        val_labels, base_val, class_names=class_names
    )
    control_to_candidate_train = _transition_summary(
        train_labels, control_train, candidate_train
    )
    control_to_candidate_val = _transition_summary(
        val_labels, control_val, candidate_val
    )
    direct_to_candidate_train = _transition_summary(
        train_labels, base_train, candidate_train
    )
    direct_to_candidate_val = _transition_summary(val_labels, base_val, candidate_val)
    train_direction_auc = _class1_error_direction_auc(
        train_labels, control_train, candidate_train
    )
    val_direction_auc = _class1_error_direction_auc(
        val_labels, control_val, candidate_val
    )
    keeper_direction_auc = _class1_error_direction_auc(
        val_labels, base_val, candidate_val
    )
    binary_oracle = _binary_focus_oracle(val_labels, base_val, candidate_val)
    gate = assess_dinov2_dense_patch_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_groups=int(plans["train"]["source_groups"]),
        dense_effective_rank=dense_effective_rank,
        descriptor_finite=descriptor_finite,
        energy_entropy=float(val["energy_stats"]["entropy_mean"]),
        energy_border_mass=float(val["energy_stats"]["border_mass_mean"]),
        direct_val_metrics=direct_val_metrics,
        control_oof_metrics=control["oof_metrics"],
        candidate_oof_metrics=candidate["oof_metrics"],
        control_val_metrics=control["val_metrics"],
        candidate_val_metrics=candidate["val_metrics"],
        binary_oracle=binary_oracle,
        train_direction_auc=train_direction_auc,
        val_direction_auc=val_direction_auc,
        direct_val_transitions=direct_to_candidate_val,
    )

    _write_prediction_audit(
        output_dir / "train_oof_prediction_audit.csv",
        labels=train_labels,
        sample_index=np.asarray(train["sample_index"], dtype=np.int64),
        paths=np.asarray(train["paths"], dtype=object),
        base_probabilities=base_train,
        variants={"cls": control_train, "cls_plus_dense_patch": candidate_train},
    )
    _write_prediction_audit(
        output_dir / "val_prediction_audit.csv",
        labels=val_labels,
        sample_index=np.asarray(val["sample_index"], dtype=np.int64),
        paths=np.asarray(val["paths"], dtype=object),
        base_probabilities=base_val,
        variants={"cls": control_val, "cls_plus_dense_patch": candidate_val},
    )
    _write_preview(
        output_dir / "dinov2_patch_energy_preview.png",
        preview=val["preview"],
        folder_classes=val["folder_classes"],
    )
    if bool(args.save_descriptors):
        for split in ("train", "val"):
            _save_descriptor_cache(
                output_dir / f"{split}_descriptors.npz", payload=aligned[split]
            )

    compact_readouts = {
        feature_name: {
            "dimension": int(np.asarray(train[feature_name]).shape[1]),
            "final_pca_explained_variance": float(
                readout_fits[feature_name]["final_pca_explained_variance"]
            ),
            "fold_telemetry": readout_fits[feature_name]["folds"],
            "final_fit_elapsed_seconds": float(
                readout_fits[feature_name]["final_fit_elapsed_seconds"]
            ),
            "readouts": {
                name: _compact_readout(readout_fits[feature_name]["readouts"][name])
                for name in READOUT_NAMES
            },
        }
        for feature_name in FEATURE_NAMES
    }
    summary = {
        "mode": "dinov2_dense_patch_readiness",
        "guardrail": (
            "Frozen pretrained DINOv2-S descriptors with a matched CLS control; "
            "strict source-grouped train OOF and validation transfer only. No test, "
            "raw-data edit, target manifest, image-model training, or checkpoint write."
        ),
        "literature": list(LITERATURE),
        "protocol": preflight["protocol"],
        "model": model_metadata,
        "runtime": {
            "device": str(device),
            "amp": bool(args.amp),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "extraction_seconds": extraction_seconds,
            "peak_cuda_memory_mib": {
                split: float(aligned[split]["peak_cuda_memory_mib"])
                for split in ("train", "val")
            },
            "elapsed_seconds": float(time.perf_counter() - start),
        },
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "descriptor": {
            "finite": descriptor_finite,
            "dense_effective_rank_projected128": dense_effective_rank,
            "energy_stats": {
                split: aligned[split]["energy_stats"] for split in ("train", "val")
            },
            "saved": bool(args.save_descriptors),
        },
        "direct_base": {
            "train_metrics": direct_train_metrics,
            "val_metrics": direct_val_metrics,
        },
        "matched_readouts": compact_readouts,
        "class1_complementarity": {
            "binary_validation_oracle_vs_keeper": binary_oracle,
            "control_error_direction_auc_train_oof": train_direction_auc,
            "control_error_direction_auc_val": val_direction_auc,
            "keeper_error_direction_auc_val": keeper_direction_auc,
            "control_to_candidate_train_oof": control_to_candidate_train,
            "control_to_candidate_val": control_to_candidate_val,
            "keeper_to_candidate_train_oof": direct_to_candidate_train,
            "keeper_to_candidate_val": direct_to_candidate_val,
        },
        "gate": gate,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
        "image_model_trained": False,
        "decision": (
            "Proceed to one fixed short DINOv2 dense-patch auxiliary smoke."
            if bool(gate["smoke_permission"])
            else "Reject DINOv2 dense-patch target before image-model training."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    control_oof = control["oof_metrics"]
    control_val_metrics = control["val_metrics"]
    candidate_oof_metrics = candidate["oof_metrics"]
    candidate_val_metrics = candidate["val_metrics"]
    (output_dir / "README.md").write_text(
        "# DINOv2 dense-patch readiness audit\n\n"
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`; test is closed.\n"
        f"- Matched CLS OOF macro/class1: `{control_oof['macro_f1']:.6f}/{control_oof['focus_f1']:.6f}`.\n"
        f"- Matched dense OOF macro/class1: `{candidate_oof_metrics['macro_f1']:.6f}/{candidate_oof_metrics['focus_f1']:.6f}`.\n"
        f"- Matched CLS validation macro/class1: `{control_val_metrics['macro_f1']:.6f}/{control_val_metrics['focus_f1']:.6f}`.\n"
        f"- Matched dense validation macro/class1: `{candidate_val_metrics['macro_f1']:.6f}/{candidate_val_metrics['focus_f1']:.6f}`.\n"
        f"- Keeper validation macro/class1: `{direct_val_metrics['macro_f1']:.6f}/{direct_val_metrics['focus_f1']:.6f}`.\n"
        f"- Dense FN-vs-FP direction AUROC train/val: `{train_direction_auc:.6f}/{val_direction_auc:.6f}`.\n"
        f"- Smoke permission: `{bool(gate['smoke_permission'])}`.\n"
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n\n"
        "Descriptor settings were fixed before validation and no descriptor cache is saved by default.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="dinov2_dense_patch_readiness_evidence_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_audit(args)
    if bool(summary.get("mode") == "dinov2_dense_patch_readiness_preflight"):
        payload = summary
    else:
        payload = {
            "output_dir": str(Path(args.output_dir).resolve()),
            "smoke_permission": bool(summary["gate"]["smoke_permission"]),
            "failed_checks": summary["gate"]["failed_checks"],
            "decision": summary["decision"],
        }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
