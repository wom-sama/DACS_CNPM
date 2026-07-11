from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.data.dataset import PairedViewTrainDataset
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _effective_rank,
    _fit_grouped_readout,
    _l2_normalize,
    _transition_summary,
    _write_prediction_audit,
)
from trkh.tools.remap_classification_teacher_to_yolo import (
    _build_yolo_dataset,
    _classification_key,
)


LITERATURE = [
    "https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html",
    "https://openaccess.thecvf.com/content_ICCV_2019/html/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.html",
    "https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Masked_Autoencoders_Enable_Efficient_Knowledge_Distillers_CVPR_2023_paper.html",
    "https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html",
]

STAGE_NAMES = ("stage1", "stage2", "stage4", "stage5", "head")
READOUT_NAMES = ("head_mean", "multistage_signature", "head_plus_signature")
PROBABILITY_COLUMN = re.compile(r"^prob_(\d+)(?:_|$)")


class PathImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, int(target), str(self.samples[int(index)][0])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether label-independent ImageNet EfficientNetV2 multi-stage "
            "spatial/moment features add source-grouped OOF and validation-safe "
            "class-1 signal before implementing feature distillation. Train/val only."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--base-train-csv", type=Path, required=True)
    parser.add_argument("--base-val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="tf_efficientnetv2_s.in21k_ft_in1k",
    )
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--signature-projection-dim", type=int, default=64)
    parser.add_argument("--signature-spatial-size", type=int, default=8)
    parser.add_argument("--projection-seed", type=int, default=20260711)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--logistic-c", type=float, default=0.3)
    parser.add_argument("--logistic-max-iter", type=int, default=400)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--save-descriptors", action="store_true", default=False)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_fixed_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
) -> Tensor:
    input_dim = int(input_dim)
    output_dim = min(input_dim, max(2, int(output_dim)))
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((input_dim, output_dim), generator=generator, dtype=torch.float32)
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection.contiguous()


def multistage_signature_from_activations(
    activations: Mapping[str, Tensor],
    *,
    projection_dim: int,
    spatial_size: int,
    seed: int,
    projections: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    missing = [name for name in STAGE_NAMES if name not in activations]
    if missing:
        raise ValueError(f"Missing teacher activations: {missing}")
    projections = projections if projections is not None else {}
    stage_signatures: List[Tensor] = []
    energy_maps: Dict[str, Tensor] = {}
    for stage_index, name in enumerate(STAGE_NAMES):
        feature = activations[name]
        if feature.ndim != 4:
            raise ValueError(f"Activation {name} must be [B,C,H,W], got {tuple(feature.shape)}")
        values = feature.float()
        mean = values.mean(dim=(2, 3))
        variance = values.square().mean(dim=(2, 3)) - mean.square()
        moments = torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=1)
        projection = projections.get(name)
        if projection is None:
            projection = build_fixed_projection(
                int(moments.size(1)),
                int(projection_dim),
                seed=int(seed) + 1009 * int(stage_index),
            ).to(device=moments.device)
            projections[name] = projection
        projection = projection.to(device=moments.device, dtype=torch.float32)
        moment_signature = F.normalize(moments @ projection, dim=1, eps=1e-6)
        energy = values.square().mean(dim=1, keepdim=True)
        pooled_energy = F.adaptive_avg_pool2d(
            energy,
            (max(2, int(spatial_size)), max(2, int(spatial_size))),
        ).squeeze(1)
        energy_maps[name] = pooled_energy
        energy_signature = F.normalize(pooled_energy.flatten(1), dim=1, eps=1e-6)
        stage_signatures.append(
            F.normalize(torch.cat((moment_signature, energy_signature), dim=1), dim=1, eps=1e-6)
        )
    signature = F.normalize(torch.cat(stage_signatures, dim=1), dim=1, eps=1e-6)
    head_mean = F.normalize(activations["head"].float().mean(dim=(2, 3)), dim=1, eps=1e-6)
    return head_mean, signature, energy_maps, projections


def _energy_statistics(energy: Tensor) -> Tuple[Tensor, Tensor]:
    values = energy.float().clamp_min(0.0)
    batch, height, width = values.shape
    probabilities = values.flatten(1)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / max(math.log(max(2, height * width)), 1e-8)
    border = torch.zeros_like(values)
    border[:, 0, :] = 1.0
    border[:, -1, :] = 1.0
    border[:, :, 0] = 1.0
    border[:, :, -1] = 1.0
    border_mass = (values * border).flatten(1).sum(dim=1) / values.flatten(1).sum(dim=1).clamp_min(1e-8)
    return entropy, border_mass


def _imagefolder_rows(root: Path, split: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    dataset = ImageFolder(Path(root) / str(split))
    return [(str(path), int(target)) for path, target in dataset.samples], list(dataset.classes)


def _alignment_plan(
    *,
    classification_rows: Sequence[Tuple[str, int]],
    classification_classes: Sequence[str],
    yolo_data: Path,
    split: str,
    class_name_mode: str,
) -> Dict[str, object]:
    key_to_row: Dict[Tuple[str, int], int] = {}
    duplicate_keys: List[Tuple[str, int]] = []
    for row_index, (path, _) in enumerate(classification_rows):
        key = _classification_key(path)
        if key in key_to_row:
            duplicate_keys.append(key)
        key_to_row[key] = int(row_index)
    if duplicate_keys:
        raise ValueError(f"Duplicate classification keys for {split}: {duplicate_keys[:5]}")

    yolo_dataset, target_class_names = _build_yolo_dataset(
        yolo_data=Path(yolo_data),
        split=str(split),
        class_name_mode=str(class_name_mode),
        expected_num_classes=5,
    )
    yolo_labels = [int(value) for value in yolo_dataset.labels()]
    yolo_paths = [str(path) for path in yolo_dataset.sample_paths()]
    source_stems: List[str] = []
    row_indices: List[int] = []
    missing: List[Dict[str, object]] = []
    label_mismatches: List[Dict[str, object]] = []
    for sample_index in range(len(yolo_dataset)):
        source_id, object_index, fallback_label = PairedViewTrainDataset._sample_key(
            yolo_dataset,
            sample_index,
        )
        key = (str(source_id), int(object_index))
        row_index = key_to_row.get(key)
        target = yolo_labels[sample_index] if sample_index < len(yolo_labels) else int(fallback_label)
        if row_index is None:
            missing.append(
                {
                    "sample_index": int(sample_index),
                    "source_stem": str(source_id),
                    "object_index": int(object_index),
                    "target": int(target),
                }
            )
            continue
        classification_target = int(classification_rows[row_index][1])
        classification_name = str(classification_classes[classification_target])
        target_name = str(target_class_names[target])
        if classification_name != target_name:
            label_mismatches.append(
                {
                    "sample_index": int(sample_index),
                    "classification_name": classification_name,
                    "target_name": target_name,
                }
            )
        row_indices.append(int(row_index))
        source_stems.append(str(source_id))
    if missing or label_mismatches:
        raise ValueError(
            f"Strict {split} alignment failed: missing={len(missing)}, "
            f"label_mismatches={len(label_mismatches)}, "
            f"preview={(missing + label_mismatches)[:5]}"
        )
    if len(row_indices) != len(classification_rows) or len(row_indices) != len(yolo_dataset):
        raise ValueError(
            f"Strict {split} row count mismatch: classification={len(classification_rows)}, "
            f"mapped={len(row_indices)}, yolo={len(yolo_dataset)}"
        )
    return {
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "labels": np.asarray(yolo_labels, dtype=np.int64),
        "sample_index": np.arange(len(row_indices), dtype=np.int64),
        "paths": np.asarray(yolo_paths, dtype=object),
        "source_stems": np.asarray(source_stems, dtype=object),
        "class_names": [str(name) for name in target_class_names],
        "classification_rows": int(len(classification_rows)),
        "yolo_rows": int(len(yolo_dataset)),
        "source_groups": int(len(set(source_stems))),
        "duplicate_keys": 0,
        "missing_rows": 0,
        "label_mismatches": 0,
    }


def _compact_alignment(plan: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: plan[key]
        for key in (
            "classification_rows",
            "yolo_rows",
            "source_groups",
            "duplicate_keys",
            "missing_rows",
            "label_mismatches",
        )
    }


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
    projection_dim: int,
    spatial_size: int,
    projection_seed: int,
    data_mean: Sequence[float],
    data_std: Sequence[float],
) -> Dict[str, object]:
    dataset = PathImageFolder(Path(classification_root) / str(split), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=True,
        persistent_workers=bool(int(workers) > 0),
    )
    stage_modules = {
        "stage1": model.blocks[1],
        "stage2": model.blocks[2],
        "stage4": model.blocks[4],
        "stage5": model.blocks[5],
        "head": model.bn2,
    }
    activations: Dict[str, Tensor] = {}
    handles = []
    for name, module in stage_modules.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, stage_name=name: activations.__setitem__(
                    stage_name,
                    output,
                )
            )
        )
    head_batches: List[np.ndarray] = []
    signature_batches: List[np.ndarray] = []
    paths: List[str] = []
    folder_targets: List[np.ndarray] = []
    projections: Dict[str, Tensor] = {}
    entropy_values: Dict[str, List[np.ndarray]] = {name: [] for name in STAGE_NAMES}
    border_values: Dict[str, List[np.ndarray]] = {name: [] for name in STAGE_NAMES}
    preview: Dict[int, Dict[str, object]] = {}
    mean = torch.tensor(data_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(data_std, dtype=torch.float32).view(1, 3, 1, 1)
    model.eval()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc=f"generic-teacher-{split}", dynamic_ncols=True, leave=False)
            for images, targets, batch_paths in iterator:
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                activations.clear()
                with autocast_context(device, bool(amp)):
                    model(images)
                head, signature, energy_maps, projections = multistage_signature_from_activations(
                    activations,
                    projection_dim=int(projection_dim),
                    spatial_size=int(spatial_size),
                    seed=int(projection_seed),
                    projections=projections,
                )
                head_batches.append(head.detach().cpu().numpy())
                signature_batches.append(signature.detach().cpu().numpy())
                target_array = targets.detach().cpu().numpy().astype(np.int64, copy=False)
                folder_targets.append(target_array)
                paths.extend(str(path) for path in batch_paths)
                for name, energy in energy_maps.items():
                    entropy, border = _energy_statistics(energy)
                    entropy_values[name].append(entropy.detach().cpu().numpy())
                    border_values[name].append(border.detach().cpu().numpy())
                for row_index, target in enumerate(target_array.tolist()):
                    if int(target) in preview:
                        continue
                    rgb = (
                        images[row_index].detach().float().cpu().unsqueeze(0) * std + mean
                    ).squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)
                    preview[int(target)] = {
                        "rgb": (rgb * 255.0).round().astype(np.uint8),
                        "path": str(batch_paths[row_index]),
                        "energy": {
                            name: energy_maps[name][row_index].detach().float().cpu().numpy()
                            for name in STAGE_NAMES
                        },
                    }
    finally:
        for handle in handles:
            handle.remove()
    return {
        "head_mean": np.concatenate(head_batches, axis=0).astype(np.float32, copy=False),
        "multistage_signature": np.concatenate(signature_batches, axis=0).astype(
            np.float32,
            copy=False,
        ),
        "paths": np.asarray(paths, dtype=object),
        "folder_targets": np.concatenate(folder_targets, axis=0).astype(np.int64, copy=False),
        "folder_classes": list(dataset.classes),
        "energy_stats": {
            name: {
                "entropy_mean": float(np.concatenate(entropy_values[name]).mean()),
                "entropy_p10": float(np.quantile(np.concatenate(entropy_values[name]), 0.10)),
                "border_mass_mean": float(np.concatenate(border_values[name]).mean()),
                "border_mass_p90": float(np.quantile(np.concatenate(border_values[name]), 0.90)),
            }
            for name in STAGE_NAMES
        },
        "preview": preview,
    }


def _align_extracted_payload(
    payload: Mapping[str, object],
    plan: Mapping[str, object],
) -> Dict[str, object]:
    indices = np.asarray(plan["row_indices"], dtype=np.int64)
    return {
        "head_mean": np.asarray(payload["head_mean"], dtype=np.float32)[indices],
        "multistage_signature": np.asarray(payload["multistage_signature"], dtype=np.float32)[indices],
        "labels": np.asarray(plan["labels"], dtype=np.int64),
        "sample_index": np.asarray(plan["sample_index"], dtype=np.int64),
        "paths": np.asarray(plan["paths"], dtype=object),
        "source_stems": np.asarray(plan["source_stems"], dtype=object),
        "class_names": list(plan["class_names"]),
        "energy_stats": payload["energy_stats"],
        "preview": payload["preview"],
        "folder_classes": payload["folder_classes"],
    }


def _load_base_predictions(
    path: Path,
    *,
    expected_split: str,
    expected_labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        probability_columns: Dict[int, str] = {}
        for field in fields:
            match = PROBABILITY_COLUMN.match(str(field))
            if match:
                probability_columns[int(match.group(1))] = str(field)
        if sorted(probability_columns) != list(range(int(class_count))):
            raise ValueError(f"Invalid probability columns in {path}: {probability_columns}")
        rows = list(reader)
    indexed: Dict[int, Mapping[str, str]] = {}
    wrong_split = 0
    for row in rows:
        sample_index = int(str(row.get("sample_index", "") or "-1"))
        if sample_index in indexed:
            raise ValueError(f"Duplicate sample_index={sample_index} in {path}")
        normalized_path = str(row.get("image_path", "") or "").replace("/", "\\").casefold()
        if f"\\{str(expected_split).casefold()}\\" not in normalized_path:
            wrong_split += 1
        indexed[sample_index] = row
    if wrong_split:
        raise ValueError(f"Prediction CSV contains {wrong_split} non-{expected_split} paths: {path}")
    expected_indices = list(range(int(expected_labels.size)))
    if sorted(indexed) != expected_indices:
        missing = sorted(set(expected_indices).difference(indexed))
        extra = sorted(set(indexed).difference(expected_indices))
        raise ValueError(f"Strict sample_index mismatch in {path}: missing={missing[:5]}, extra={extra[:5]}")
    probabilities = np.zeros((expected_labels.size, int(class_count)), dtype=np.float32)
    for sample_index in expected_indices:
        row = indexed[sample_index]
        target = int(str(row.get("target_index", "") or "-1"))
        if target != int(expected_labels[sample_index]):
            raise ValueError(
                f"Target mismatch at {expected_split} sample_index={sample_index}: "
                f"{target} != {expected_labels[sample_index]}"
            )
        probabilities[sample_index] = np.asarray(
            [float(row[probability_columns[index]]) for index in range(int(class_count))],
            dtype=np.float32,
        )
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite base probabilities: {path}")
    return probabilities


def _class1_error_direction_auc(
    labels: np.ndarray,
    base_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class: int = 1,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    base = np.asarray(base_probabilities, dtype=np.float32)
    candidate = np.asarray(candidate_probabilities, dtype=np.float32)
    base_predictions = base.argmax(axis=1)
    false_negative = (labels == int(focus_class)) & (base_predictions != int(focus_class))
    false_positive = (labels != int(focus_class)) & (base_predictions == int(focus_class))
    selected = false_negative | false_positive
    direction = false_negative[selected].astype(np.int64)
    if int(np.unique(direction).size) != 2:
        return 0.5
    score = candidate[:, int(focus_class)] - base[:, int(focus_class)]
    return float(roc_auc_score(direction, score[selected]))


def _projected_effective_rank(features: np.ndarray, *, seed: int) -> float:
    values = np.asarray(features, dtype=np.float32)
    if int(values.shape[1]) > 128:
        projection = build_fixed_projection(
            int(values.shape[1]),
            128,
            seed=int(seed),
        ).numpy()
        values = values @ projection
    return float(_effective_rank(values, max_rows=2048))


def assess_multistage_teacher_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_groups: int,
    signature_effective_rank: float,
    mean_energy_entropy: float,
    max_energy_border_mass: float,
    direct_val_metrics: Mapping[str, object],
    head_oof_metrics: Mapping[str, object],
    candidate_oof_metrics: Mapping[str, object],
    head_val_metrics: Mapping[str, object],
    candidate_val_metrics: Mapping[str, object],
    train_direction_auc: float,
    val_direction_auc: float,
    val_transitions: Mapping[str, int],
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_source_groups": 8000,
        "min_signature_effective_rank": 24.0,
        "min_energy_entropy": 0.35,
        "max_energy_border_mass": 0.65,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_val_class1_f1": 0.70,
        "min_direct_class1_gain": 0.010,
        "max_direct_macro_drop": 0.003,
        "min_direction_auc": 0.60,
    }
    observed = {
        "oof_macro_gain": float(candidate_oof_metrics["macro_f1"])
        - float(head_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(candidate_oof_metrics["focus_f1"])
        - float(head_oof_metrics["focus_f1"]),
        "val_macro_gain": float(candidate_val_metrics["macro_f1"])
        - float(head_val_metrics["macro_f1"]),
        "val_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(head_val_metrics["focus_f1"]),
        "direct_class1_gain": float(candidate_val_metrics["focus_f1"])
        - float(direct_val_metrics["focus_f1"]),
        "direct_macro_drop": float(direct_val_metrics["macro_f1"])
        - float(candidate_val_metrics["macro_f1"]),
        "train_direction_auc": float(train_direction_auc),
        "val_direction_auc": float(val_direction_auc),
        "signature_effective_rank": float(signature_effective_rank),
        "mean_energy_entropy": float(mean_energy_entropy),
        "max_energy_border_mass": float(max_energy_border_mass),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "validation_support": int(val_samples) == int(thresholds["required_val_samples"]),
        "source_group_support": int(source_groups) >= int(thresholds["min_source_groups"]),
        "signature_rank": float(signature_effective_rank)
        >= float(thresholds["min_signature_effective_rank"]),
        "spatial_energy_noncollapsed": float(mean_energy_entropy)
        >= float(thresholds["min_energy_entropy"]),
        "spatial_energy_not_border_dominated": float(max_energy_border_mass)
        <= float(thresholds["max_energy_border_mass"]),
        "oof_macro_gain": observed["oof_macro_gain"] >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"]
        >= float(thresholds["min_oof_class1_gain"]),
        "validation_macro_gain": observed["val_macro_gain"]
        >= float(thresholds["min_val_macro_gain"]),
        "validation_class1_gain": observed["val_class1_gain"]
        >= float(thresholds["min_val_class1_gain"]),
        "validation_class1_milestone": float(candidate_val_metrics["focus_f1"])
        >= float(thresholds["min_val_class1_f1"]),
        "beats_direct_class1": observed["direct_class1_gain"]
        >= float(thresholds["min_direct_class1_gain"]),
        "preserves_direct_macro": observed["direct_macro_drop"]
        <= float(thresholds["max_direct_macro_drop"]),
        "train_error_direction": float(train_direction_auc)
        >= float(thresholds["min_direction_auc"]),
        "validation_error_direction": float(val_direction_auc)
        >= float(thresholds["min_direction_auc"]),
        "error_direction_no_sign_flip": (float(train_direction_auc) - 0.5)
        * (float(val_direction_auc) - 0.5)
        > 0.0,
        "validation_net_corrections": int(val_transitions["corrections"])
        >= int(val_transitions["harms"]),
        "validation_class1_recall_protected": int(val_transitions["class1_fn_rescued"])
        >= int(val_transitions["class1_tp_broken"]),
        "validation_class1_fp_control": int(val_transitions["class1_fp_removed"])
        >= int(val_transitions["class1_fp_created"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "multistage_teacher_feature_ready": not failed,
        "smoke_ready": not failed,
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _heat_overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    values = values - float(values.min())
    values = values / max(float(values.max()), 1e-8)
    red = np.clip(2.0 * values, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * values - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - values), 0.0, 1.0)
    color = np.stack((red, green, blue), axis=-1)
    image = Image.fromarray((color * 255.0).round().astype(np.uint8)).resize(
        (int(rgb.shape[1]), int(rgb.shape[0])),
        getattr(Image, "Resampling", Image).BILINEAR,
    )
    mapped = np.asarray(image, dtype=np.float32)
    return (0.58 * rgb.astype(np.float32) + 0.42 * mapped).clip(0.0, 255.0).round().astype(np.uint8)


def _write_preview(
    path: Path,
    *,
    preview: Mapping[int, Mapping[str, object]],
    folder_classes: Sequence[str],
) -> None:
    tile = 160
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index in range(len(folder_classes)):
        if class_index not in preview:
            continue
        item = preview[class_index]
        rgb = np.asarray(item["rgb"], dtype=np.uint8)
        row = Image.new("RGB", (tile * (1 + len(STAGE_NAMES)), tile), color=(255, 255, 255))
        row.paste(Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR), (0, 0))
        for column, name in enumerate(STAGE_NAMES, start=1):
            overlay = _heat_overlay(rgb, np.asarray(item["energy"][name], dtype=np.float32))
            row.paste(Image.fromarray(overlay).resize((tile, tile), resampling.BILINEAR), (column * tile, 0))
        rows.append(row)
        manifest.append(
            {
                "row": len(rows) - 1,
                "class_index": int(class_index),
                "class_name": str(folder_classes[class_index]),
                "path": str(item["path"]),
                "columns": ["teacher_input", *STAGE_NAMES],
            }
        )
    if not rows:
        return
    canvas = Image.new("RGB", (tile * (1 + len(STAGE_NAMES)), tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _compact_readout(result: Mapping[str, object]) -> Dict[str, object]:
    return {
        "oof_metrics": result["oof_metrics"],
        "val_metrics": result["val_metrics"],
        "fold_iterations": result["fold_iterations"],
        "final_iterations": result["final_iterations"],
    }


def _write_readout_metrics(path: Path, readouts: Mapping[str, Mapping[str, object]]) -> None:
    fields = [
        "variant",
        "oof_macro_f1",
        "oof_class1_f1",
        "oof_class1_precision",
        "oof_class1_recall",
        "val_macro_f1",
        "val_class1_f1",
        "val_class1_precision",
        "val_class1_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in READOUT_NAMES:
            result = readouts[name]
            oof = result["oof_metrics"]
            val = result["val_metrics"]
            writer.writerow(
                {
                    "variant": name,
                    "oof_macro_f1": oof["macro_f1"],
                    "oof_class1_f1": oof["focus_f1"],
                    "oof_class1_precision": oof["focus_precision"],
                    "oof_class1_recall": oof["focus_recall"],
                    "val_macro_f1": val["macro_f1"],
                    "val_class1_f1": val["focus_f1"],
                    "val_class1_precision": val["focus_precision"],
                    "val_class1_recall": val["focus_recall"],
                }
            )


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) < 2:
        raise ValueError("folds must be at least 2")
    if int(args.signature_projection_dim) < 2 or int(args.signature_spatial_size) < 2:
        raise ValueError("signature dimensions must be at least 2")
    if float(args.logistic_c) <= 0.0:
        raise ValueError("logistic-c must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans: Dict[str, Dict[str, object]] = {}
    rows_by_split: Dict[str, List[Tuple[str, int]]] = {}
    classes_by_split: Dict[str, List[str]] = {}
    for split in ("train", "val"):
        rows, classes = _imagefolder_rows(Path(args.classification_root), split)
        rows_by_split[split] = rows
        classes_by_split[split] = classes
        plans[split] = _alignment_plan(
            classification_rows=rows,
            classification_classes=classes,
            yolo_data=Path(args.yolo_data),
            split=split,
            class_name_mode=str(args.class_name_mode),
        )
    if list(plans["train"]["class_names"]) != list(plans["val"]["class_names"]):
        raise ValueError("Train/validation target class order mismatch")
    preflight = {
        "mode": "multistage_teacher_feature_readiness_preflight",
        "teacher_model": str(args.teacher_model),
        "teacher_local_finetune_checkpoint": None,
        "teacher_pretrained_on_trkh": False,
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "base_train_csv": str(Path(args.base_train_csv).resolve()),
        "base_val_csv": str(Path(args.base_val_csv).resolve()),
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "test_split_used": False,
        "raw_dataset_touched": False,
        "trainable_manifest_written": False,
    }
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if bool(args.preflight_only):
        return preflight

    device = _resolve_device(str(args.device or ""))
    model = timm.create_model(str(args.teacher_model), pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    pretrained_config = dict(getattr(model, "pretrained_cfg", {}) or {})
    model.to(device).eval()
    start = time.perf_counter()

    extracted: Dict[str, Dict[str, object]] = {}
    aligned: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        extracted[split] = _extract_split(
            model=model,
            transform=transform,
            classification_root=Path(args.classification_root),
            split=split,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            projection_dim=int(args.signature_projection_dim),
            spatial_size=int(args.signature_spatial_size),
            projection_seed=int(args.projection_seed),
            data_mean=data_config["mean"],
            data_std=data_config["std"],
        )
        if list(extracted[split]["folder_classes"]) != classes_by_split[split]:
            raise ValueError(f"ImageFolder class order changed during extraction for {split}")
        aligned[split] = _align_extracted_payload(extracted[split], plans[split])

    train = aligned["train"]
    val = aligned["val"]
    class_names = list(train["class_names"])
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    train_head = _l2_normalize(np.asarray(train["head_mean"], dtype=np.float32))
    val_head = _l2_normalize(np.asarray(val["head_mean"], dtype=np.float32))
    train_signature = _l2_normalize(np.asarray(train["multistage_signature"], dtype=np.float32))
    val_signature = _l2_normalize(np.asarray(val["multistage_signature"], dtype=np.float32))
    train_candidate = _l2_normalize(np.concatenate((train_head, train_signature), axis=1))
    val_candidate = _l2_normalize(np.concatenate((val_head, val_signature), axis=1))
    feature_variants = {
        "head_mean": (train_head, val_head),
        "multistage_signature": (train_signature, val_signature),
        "head_plus_signature": (train_candidate, val_candidate),
    }

    readouts: Dict[str, Dict[str, object]] = {}
    for name in READOUT_NAMES:
        print(f"fitting generic-teacher grouped OOF readout: {name}", flush=True)
        train_features, val_features = feature_variants[name]
        result = _fit_grouped_readout(
            train_features=train_features,
            train_labels=train_labels,
            train_groups=np.asarray(train["source_stems"], dtype=object),
            val_features=val_features,
            folds=int(args.folds),
            c_value=float(args.logistic_c),
            max_iter=int(args.logistic_max_iter),
            seed=42,
            class_names=class_names,
        )
        result["val_metrics"] = _classification_metrics(
            val_labels,
            np.asarray(result["val_probabilities"], dtype=np.float32),
            class_names=class_names,
        )
        readouts[name] = result

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
    candidate_train_probabilities = np.asarray(
        readouts["head_plus_signature"]["oof_probabilities"],
        dtype=np.float32,
    )
    candidate_val_probabilities = np.asarray(
        readouts["head_plus_signature"]["val_probabilities"],
        dtype=np.float32,
    )
    direct_val_metrics = _classification_metrics(
        val_labels,
        base_val,
        class_names=class_names,
    )
    train_transitions = _transition_summary(train_labels, base_train, candidate_train_probabilities)
    val_transitions = _transition_summary(val_labels, base_val, candidate_val_probabilities)
    train_direction_auc = _class1_error_direction_auc(
        train_labels,
        base_train,
        candidate_train_probabilities,
    )
    val_direction_auc = _class1_error_direction_auc(
        val_labels,
        base_val,
        candidate_val_probabilities,
    )
    signature_effective_rank = _projected_effective_rank(
        train_signature,
        seed=int(args.projection_seed) + 991,
    )
    energy_stats = {split: aligned[split]["energy_stats"] for split in ("train", "val")}
    mean_energy_entropy = float(
        np.mean(
            [
                float(energy_stats[split][stage]["entropy_mean"])
                for split in ("train", "val")
                for stage in STAGE_NAMES
            ]
        )
    )
    max_energy_border_mass = float(
        max(
            float(energy_stats[split][stage]["border_mass_mean"])
            for split in ("train", "val")
            for stage in STAGE_NAMES
        )
    )
    gate = assess_multistage_teacher_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_groups=int(plans["train"]["source_groups"]),
        signature_effective_rank=signature_effective_rank,
        mean_energy_entropy=mean_energy_entropy,
        max_energy_border_mass=max_energy_border_mass,
        direct_val_metrics=direct_val_metrics,
        head_oof_metrics=readouts["head_mean"]["oof_metrics"],
        candidate_oof_metrics=readouts["head_plus_signature"]["oof_metrics"],
        head_val_metrics=readouts["head_mean"]["val_metrics"],
        candidate_val_metrics=readouts["head_plus_signature"]["val_metrics"],
        train_direction_auc=train_direction_auc,
        val_direction_auc=val_direction_auc,
        val_transitions=val_transitions,
    )

    _write_readout_metrics(output_dir / "readout_metrics.csv", readouts)
    _write_prediction_audit(
        output_dir / "train_oof_prediction_audit.csv",
        labels=train_labels,
        sample_index=np.asarray(train["sample_index"], dtype=np.int64),
        paths=np.asarray(train["paths"], dtype=object),
        base_probabilities=base_train,
        variants={
            name: np.asarray(readouts[name]["oof_probabilities"], dtype=np.float32)
            for name in READOUT_NAMES
        },
    )
    _write_prediction_audit(
        output_dir / "val_prediction_audit.csv",
        labels=val_labels,
        sample_index=np.asarray(val["sample_index"], dtype=np.int64),
        paths=np.asarray(val["paths"], dtype=object),
        base_probabilities=base_val,
        variants={
            name: np.asarray(readouts[name]["val_probabilities"], dtype=np.float32)
            for name in READOUT_NAMES
        },
    )
    _write_preview(
        output_dir / "teacher_stage_energy_preview.png",
        preview=val["preview"],
        folder_classes=val["folder_classes"],
    )
    if bool(args.save_descriptors):
        for split, payload in (("train", train), ("val", val)):
            np.savez_compressed(
                output_dir / f"{split}_generic_teacher_descriptors.npz",
                head_mean=np.asarray(payload["head_mean"], dtype=np.float32),
                multistage_signature=np.asarray(payload["multistage_signature"], dtype=np.float32),
                labels=np.asarray(payload["labels"], dtype=np.int64),
                sample_index=np.asarray(payload["sample_index"], dtype=np.int64),
                source_stem=np.asarray(payload["source_stems"], dtype=object),
                paths=np.asarray(payload["paths"], dtype=object),
                classes=np.asarray(class_names, dtype=object),
            )

    summary = {
        "mode": "multistage_generic_teacher_feature_readiness",
        "guardrail": (
            "ImageNet-pretrained generic teacher only; no TRKH-finetuned teacher, test split, "
            "raw-data edit, checkpoint write, trainable manifest, or model training."
        ),
        "literature": LITERATURE,
        "teacher": {
            "model": str(args.teacher_model),
            "pretrained": True,
            "local_finetune_checkpoint": None,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "pretrained_config": pretrained_config,
            "data_config": data_config,
            "stages": list(STAGE_NAMES),
        },
        "protocol": {
            "candidate": "head_plus_signature",
            "candidate_predeclared": True,
            "candidate_selection_uses_validation": False,
            "folds": int(args.folds),
            "logistic_c": float(args.logistic_c),
            "projection_dim": int(args.signature_projection_dim),
            "spatial_size": int(args.signature_spatial_size),
            "projection_seed": int(args.projection_seed),
            "save_descriptors": bool(args.save_descriptors),
        },
        "alignment": {split: _compact_alignment(plan) for split, plan in plans.items()},
        "feature_dimensions": {
            "head_mean": int(train_head.shape[1]),
            "multistage_signature": int(train_signature.shape[1]),
            "head_plus_signature": int(train_candidate.shape[1]),
        },
        "feature_diagnostics": {
            "signature_effective_rank_projected128": signature_effective_rank,
            "energy_stats": energy_stats,
            "mean_energy_entropy": mean_energy_entropy,
            "max_energy_border_mass": max_energy_border_mass,
        },
        "direct_base": {
            "train_metrics": _classification_metrics(train_labels, base_train, class_names=class_names),
            "val_metrics": direct_val_metrics,
        },
        "readouts": {name: _compact_readout(readouts[name]) for name in READOUT_NAMES},
        "class1_direction": {
            "train_oof_auc": train_direction_auc,
            "val_auc": val_direction_auc,
            "train_transitions_vs_base": train_transitions,
            "val_transitions_vs_base": val_transitions,
        },
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - start),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "trainable_manifest_written": False,
        "checkpoint_written": False,
        "decision": (
            "Proceed to one fixed multi-stage spatial feature-distillation smoke."
            if bool(gate["smoke_permission"])
            else "Reject multi-stage teacher feature distillation before GPU training."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Multi-stage generic-teacher feature readiness\n\n"
        f"- Teacher: `{args.teacher_model}` ImageNet pretrained, no TRKH fine-tune.\n"
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`; test is closed.\n"
        f"- Candidate val macro/class1 F1: "
        f"`{readouts['head_plus_signature']['val_metrics']['macro_f1']:.6f}/"
        f"{readouts['head_plus_signature']['val_metrics']['focus_f1']:.6f}`.\n"
        f"- Base val macro/class1 F1: "
        f"`{direct_val_metrics['macro_f1']:.6f}/{direct_val_metrics['focus_f1']:.6f}`.\n"
        f"- FN-vs-FP direction AUROC train/val: `{train_direction_auc:.6f}/{val_direction_auc:.6f}`.\n"
        f"- Smoke permission: `{bool(gate['smoke_permission'])}`.\n"
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`.\n\n"
        "This artifact is diagnostic only and writes no training target or checkpoint.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_audit(args)
    print(json.dumps(summary if args.preflight_only else {
        "output_dir": str(Path(args.output_dir).resolve()),
        "smoke_permission": bool(summary["gate"]["smoke_permission"]),
        "failed_checks": summary["gate"]["failed_checks"],
        "decision": summary["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
