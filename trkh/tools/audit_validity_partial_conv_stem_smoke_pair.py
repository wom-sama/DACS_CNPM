from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    load_checkpoint,
    set_seed,
)
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    ValidityPartialConv2d,
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_bbox_from_model_output,
)
from trkh.tools.audit_bbox_logpolar_stem_a0 import _dataset_tree_identity
from trkh.tools.audit_visual_contrast_smoke_pair import (
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SUPPORT,
    FOCUS_CLASS,
    FOCUS_FP_CLASSES,
    _calibration,
    _metrics,
    _prepare_output_dir,
    _read_predictions,
    _sha256,
)


METHOD = "validity_partial_conv_stem"
PROTOCOL_STAGE = "B_matched_keeper_resume_120b_2e_full_validation"
SEED = 42
SOURCE_FOLDS = 5
MASK_RANK_BINS = 4
MIN_FOLD_CLASS1_SUPPORT = 25
MIN_FOLD_RESTRICTED_NEGATIVE_SUPPORT = 300
MAX_INDEPENDENT_PROBABILITY_ERROR = 5e-4
ALLOWED_ROLE_CONFIG_DIFFERENCES = frozenset(
    {
        "data.data_cartography.occurrence_output",
        "data.data_cartography.output",
        "model_config.stem_convolution",
        "run_dir",
        "run_name",
        "train_config.data_cartography_output",
    }
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked full-validation and mask-placebo audit for the matched "
            "validity partial-convolution smoke pair."
        )
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--control-predictions", type=Path)
    parser.add_argument("--candidate-predictions", type=Path)
    parser.add_argument("--control-run-dir", type=Path)
    parser.add_argument("--candidate-run-dir", type=Path)
    parser.add_argument("--stage-a-summary", type=Path)
    parser.add_argument("--locked-protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _json_dump(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _metrics_with_confusion(
    labels: np.ndarray, predictions: np.ndarray
) -> Dict[str, object]:
    result = _metrics(labels, predictions)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    result["confusion_matrix"] = confusion.tolist()
    return result


def _transition_stats(
    labels: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> Dict[str, int]:
    focus_negative = np.isin(labels, FOCUS_FP_CLASSES)
    focus_positive = labels == FOCUS_CLASS
    corrections = (reference != labels) & (candidate == labels)
    harms = (reference == labels) & (candidate != labels)
    fp_removed = focus_negative & (reference == FOCUS_CLASS) & (
        candidate != FOCUS_CLASS
    )
    fp_created = focus_negative & (reference != FOCUS_CLASS) & (
        candidate == FOCUS_CLASS
    )
    fn_rescued = focus_positive & (reference != FOCUS_CLASS) & (
        candidate == FOCUS_CLASS
    )
    tp_broken = focus_positive & (reference == FOCUS_CLASS) & (
        candidate != FOCUS_CLASS
    )
    return {
        "changed_decisions": int(np.sum(reference != candidate)),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "focus_false_positives_removed": int(fp_removed.sum()),
        "focus_false_positives_created": int(fp_created.sum()),
        "focus_false_positive_net_reduction": int(
            fp_removed.sum() - fp_created.sum()
        ),
        "focus_false_negatives_rescued": int(fn_rescued.sum()),
        "focus_true_positives_broken": int(tp_broken.sum()),
    }


def _deep_differences(left: object, right: object, path: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        output: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                output.append(child)
            else:
                output.extend(_deep_differences(left[key], right[key], child))
        return output
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        output = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            output.extend(
                _deep_differences(left_item, right_item, f"{path}[{index}]")
            )
        return output
    return [] if left == right else [path]


def _normalized_train_arguments(arguments: Sequence[object]) -> list[str]:
    normalized = [str(value) for value in arguments]
    for name in (
        "--run-name",
        "--stem-convolution",
        "--data-cartography-output",
    ):
        try:
            index = normalized.index(name)
        except ValueError as exc:
            raise ValueError(f"Locked arguments omit {name}") from exc
        if index + 1 >= len(normalized) or normalized[index + 1].startswith("--"):
            raise ValueError(f"Locked argument {name} has no value")
        normalized[index + 1] = "<matched-role>"
    return normalized


def build_mask_derangement(
    labels: np.ndarray,
    sources: np.ndarray,
    invalid_counts: np.ndarray,
    mask_sha256: np.ndarray,
    *,
    rank_bins: int = MASK_RANK_BINS,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(sources, dtype=object)
    invalid_counts = np.asarray(invalid_counts, dtype=np.int64)
    mask_sha256 = np.asarray(mask_sha256, dtype=object)
    if not (
        labels.ndim == 1
        and labels.shape == sources.shape == invalid_counts.shape == mask_sha256.shape
    ):
        raise ValueError("Mask derangement inputs must be aligned 1D arrays")
    if int(rank_bins) < 1:
        raise ValueError("rank_bins must be positive")

    mapping = np.full(labels.size, -1, dtype=np.int64)
    bins = np.full(labels.size, -1, dtype=np.int64)
    partitions: list[Dict[str, object]] = []
    for class_index in sorted(np.unique(labels).tolist()):
        class_rows = np.flatnonzero(labels == int(class_index)).tolist()
        class_rows.sort(
            key=lambda index: (
                int(invalid_counts[index]),
                str(sources[index]).casefold(),
                int(index),
            )
        )
        for rank_bin, raw_partition in enumerate(
            np.array_split(np.asarray(class_rows, dtype=np.int64), int(rank_bins))
        ):
            partition = raw_partition.tolist()
            if len(partition) < 2:
                raise ValueError(
                    f"class={class_index} bin={rank_bin} has fewer than two rows"
                )
            size = len(partition)
            invalid_cost = np.int64(10**15)
            cost = np.full((size, size), invalid_cost, dtype=np.int64)
            for row_position, row_index in enumerate(partition):
                for donor_position, donor_index in enumerate(partition):
                    if (
                        str(sources[row_index]).casefold()
                        == str(sources[donor_index]).casefold()
                        or str(mask_sha256[row_index]) == str(mask_sha256[donor_index])
                    ):
                        continue
                    distance = abs(
                        int(invalid_counts[row_index])
                        - int(invalid_counts[donor_index])
                    )
                    tie_break = (donor_position - row_position) % size
                    cost[row_position, donor_position] = np.int64(
                        distance * (size + 1) + tie_break
                    )
            selected_rows, selected_columns = linear_sum_assignment(cost)
            selected_costs = cost[selected_rows, selected_columns]
            if bool(np.any(selected_costs >= invalid_cost)):
                raise RuntimeError(
                    f"No valid mask derangement for class={class_index} bin={rank_bin}"
                )
            for row_position, donor_position in zip(
                selected_rows.tolist(), selected_columns.tolist()
            ):
                row_index = partition[int(row_position)]
                donor_index = partition[int(donor_position)]
                mapping[row_index] = donor_index
                bins[row_index] = int(rank_bin)
            partitions.append(
                {
                    "class_index": int(class_index),
                    "rank_bin": int(rank_bin),
                    "rows": int(size),
                    "unique_sources": int(
                        len({str(sources[index]).casefold() for index in partition})
                    ),
                    "unique_masks": int(
                        len({str(mask_sha256[index]) for index in partition})
                    ),
                    "invalid_minimum": int(
                        min(int(invalid_counts[index]) for index in partition)
                    ),
                    "invalid_maximum": int(
                        max(int(invalid_counts[index]) for index in partition)
                    ),
                }
            )

    if bool(np.any(mapping < 0)) or bool(np.any(bins < 0)):
        raise RuntimeError("Mask derangement is incomplete")
    checks = {
        "bijection": np.unique(mapping).size == mapping.size,
        "same_class": bool(np.array_equal(labels[mapping], labels)),
        "same_rank_bin": bool(np.array_equal(bins[mapping], bins)),
        "different_source": bool(
            np.all(
                np.char.lower(sources.astype(str))[mapping]
                != np.char.lower(sources.astype(str))
            )
        ),
        "different_mask": bool(np.all(mask_sha256[mapping] != mask_sha256)),
    }
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError(f"Mask derangement invariants failed: {checks}")
    return mapping, bins, {
        "rank_bins": int(rank_bins),
        "partitions": partitions,
        "checks": checks,
        "passed": True,
    }


def build_source_group_folds(
    labels: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.char.lower(np.asarray(sources, dtype=str))
    if labels.ndim != 1 or labels.shape != sources.shape:
        raise ValueError("Source fold inputs must be aligned 1D arrays")
    folds = np.full(labels.size, -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=SOURCE_FOLDS, shuffle=True, random_state=SEED
    )
    for fold_index, (_fit, holdout) in enumerate(
        splitter.split(np.zeros(labels.size), labels, groups=sources)
    ):
        folds[holdout] = int(fold_index)
    if bool(np.any(folds < 0)):
        raise RuntimeError("Source fold assignment is incomplete")
    source_fold_sets: Dict[str, set[int]] = {}
    for source, fold in zip(sources.tolist(), folds.tolist()):
        source_fold_sets.setdefault(str(source), set()).add(int(fold))
    overlap = sum(int(len(value) != 1) for value in source_fold_sets.values())
    fold_rows = []
    for fold_index in range(SOURCE_FOLDS):
        selected = folds == fold_index
        support = np.bincount(labels[selected], minlength=5)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "rows": int(selected.sum()),
                "sources": int(len(set(sources[selected].tolist()))),
                "support": support.tolist(),
                "class1_support": int(support[FOCUS_CLASS]),
                "restricted_negative_support": int(
                    support[list(FOCUS_FP_CLASSES)].sum()
                ),
                "adequate_support": bool(
                    int(support[FOCUS_CLASS]) >= MIN_FOLD_CLASS1_SUPPORT
                    and int(support[list(FOCUS_FP_CLASSES)].sum())
                    >= MIN_FOLD_RESTRICTED_NEGATIVE_SUPPORT
                ),
            }
        )
    return folds, {
        "folds": fold_rows,
        "source_overlap": int(overlap),
        "assignment_complete": True,
        "passed": overlap == 0
        and all(bool(row["adequate_support"]) for row in fold_rows),
    }


def _build_validation_dataset(
    data_yaml: Path, checkpoint: Mapping[str, object]
) -> MangoYOLOCropDataset:
    data_spec = load_data_spec(Path(data_yaml))
    model_config = checkpoint.get("model_config", {})
    augmentation = checkpoint.get("augmentation_config", {})
    if not isinstance(model_config, Mapping) or not isinstance(augmentation, Mapping):
        raise ValueError("Checkpoint omits model/augmentation configuration")
    mean, std = checkpoint_input_normalization(checkpoint)
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
        mean=mean,
        std=std,
    )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="val",
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
        classification_source_context_aux=False,
    )


def _make_loader(
    dataset: MangoYOLOCropDataset,
    *,
    batch_size: int,
    num_workers: int,
    context: str,
) -> tuple[DataLoader, Dict[str, object]]:
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=context,
        prefetch_factor=2,
        persistent_workers=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=build_train_collate_fn(
            num_classes=5, batch_mix_probability=0.0
        ),
        **kwargs,
    )
    return loader, summary


def _collect_mask_geometry(
    dataset: MangoYOLOCropDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    loader, loader_summary = _make_loader(
        dataset,
        batch_size=max(1, int(batch_size) * 2),
        num_workers=num_workers,
        context="validity_partial_smoke_mask_census",
    )
    mask_cache = np.empty((len(dataset), 256, 256), dtype=bool)
    labels = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("Validation loader must return image/label/metadata")
        batch_labels = batch[1]
        metadata = batch[2]
        image_mask = metadata.get("image_mask") if isinstance(metadata, Mapping) else None
        if not torch.is_tensor(batch_labels) or not torch.is_tensor(image_mask):
            raise ValueError("Validation batch omits labels or image_mask")
        count = int(batch_labels.size(0))
        mask_cache[cursor : cursor + count] = image_mask.cpu().numpy().astype(bool)
        labels[cursor : cursor + count] = (
            batch_labels.detach().cpu().to(dtype=torch.long).numpy()
        )
        cursor += count
    if cursor != len(dataset):
        raise RuntimeError("Mask census row count mismatch")
    invalid_counts = mask_cache.size // len(dataset) - mask_cache.reshape(len(dataset), -1).sum(axis=1)
    mask_hashes = np.asarray(
        [hashlib.sha256(mask.tobytes()).hexdigest() for mask in mask_cache],
        dtype=object,
    )
    sources = np.asarray(
        [sample.image_path.stem.casefold() for sample in dataset.samples], dtype=object
    )
    object_indices = np.asarray(
        [int(sample.primary_object_index) for sample in dataset.samples], dtype=np.int64
    )
    return {
        "mask_cache": mask_cache,
        "labels": labels,
        "sources": sources,
        "object_indices": object_indices,
        "invalid_counts": invalid_counts.astype(np.int64),
        "mask_sha256": mask_hashes,
        "loader": loader_summary,
    }


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    image_mask: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    bbox = metadata.get("bbox")
    crop_bbox = metadata.get("crop_bbox")
    bbox = (
        bbox.to(device=device, dtype=torch.float32, non_blocking=True)
        if torch.is_tensor(bbox)
        else None
    )
    crop_bbox = (
        crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
        if torch.is_tensor(crop_bbox)
        else bbox
    )
    image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    if hasattr(model, "forward_features") and hasattr(model, "forward_heads"):
        features = model.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=crop_bbox,
        )
        if bbox is not None:
            features["bbox"] = bbox
        output = model.forward_heads(features)
    elif hasattr(model, "forward_features") and hasattr(model, "head"):
        features = model.forward_features(
            images,
            image_valid_mask=image_mask,
            bbox_token_prior=crop_bbox,
        )
        if bbox is not None:
            features["bbox"] = bbox
        output = classification_logits_from_features(model, features)
    else:
        output = model(images)
    logits, _bbox = extract_bbox_from_model_output(output)
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.size(1) != 5:
        raise RuntimeError("Independent checkpoint replay did not produce [B,5] logits")
    return logits


def _independent_inference(
    *,
    dataset: MangoYOLOCropDataset,
    control_checkpoint: Mapping[str, object],
    candidate_checkpoint: Mapping[str, object],
    mask_cache: np.ndarray,
    derangement: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, object]:
    control_model = build_model_from_checkpoint(dict(control_checkpoint)).to(device).eval()
    candidate_model = build_model_from_checkpoint(dict(candidate_checkpoint)).to(device).eval()
    control_partial_count = sum(
        isinstance(module, ValidityPartialConv2d) for module in control_model.modules()
    )
    candidate_partial_count = sum(
        isinstance(module, ValidityPartialConv2d) for module in candidate_model.modules()
    )
    state_before = {
        "control": _model_state_sha256(control_model),
        "candidate": _model_state_sha256(candidate_model),
    }
    loader, loader_summary = _make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        context="validity_partial_smoke_independent_replay",
    )
    collected = {role: [] for role in ("control", "candidate", "placebo")}
    labels = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            images, batch_labels, metadata = batch
            count = int(images.size(0))
            indices = np.arange(cursor, cursor + count, dtype=np.int64)
            actual_mask = metadata.get("image_mask")
            if not torch.is_tensor(actual_mask):
                raise ValueError("Independent replay batch omits image_mask")
            placebo_mask = torch.from_numpy(mask_cache[derangement[indices]])
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, True):
                control_logits = _forward_logits(
                    control_model, images, metadata, actual_mask, device=device
                )
                candidate_logits = _forward_logits(
                    candidate_model, images, metadata, actual_mask, device=device
                )
                placebo_logits = _forward_logits(
                    candidate_model, images, metadata, placebo_mask, device=device
                )
            collected["control"].append(F.softmax(control_logits.float(), dim=1).cpu())
            collected["candidate"].append(
                F.softmax(candidate_logits.float(), dim=1).cpu()
            )
            collected["placebo"].append(F.softmax(placebo_logits.float(), dim=1).cpu())
            labels.append(batch_labels.detach().cpu().to(dtype=torch.long))
            cursor += count
    state_after = {
        "control": _model_state_sha256(control_model),
        "candidate": _model_state_sha256(candidate_model),
    }
    return {
        "probabilities": {
            role: torch.cat(values).numpy().astype(np.float64)
            for role, values in collected.items()
        },
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "rows": int(cursor),
        "loader": loader_summary,
        "state_before": state_before,
        "state_after": state_after,
        "state_unchanged": state_before == state_after,
        "control_partial_count": int(control_partial_count),
        "candidate_partial_count": int(candidate_partial_count),
    }


def _trace_summary(run_dir: Path, expected_convolution: str) -> Dict[str, object]:
    path = Path(run_dir) / "architecture_trace" / "trace_summary.json"
    trace = _load_json(path)
    samples = trace.get("samples")
    model_config = trace.get("model_config")
    if not isinstance(samples, list) or not isinstance(model_config, Mapping):
        raise ValueError(f"Malformed architecture trace: {path}")
    class_ids = sorted(int(sample["class_id"]) for sample in samples)
    convolution = str(model_config.get("stem_convolution", "standard"))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "sample_count": len(samples),
        "class_ids": class_ids,
        "stem_convolution": convolution,
        "passed": len(samples) == 5
        and class_ids == list(range(5))
        and convolution == expected_convolution,
    }


def _contains_test_output(run_dir: Path, summary: Mapping[str, object]) -> bool:
    if summary.get("test_summary") is not None:
        return True
    forbidden = {"final_test", "test", "test_eval", "test_predictions"}
    return any(
        any(part.casefold() in forbidden for part in path.relative_to(run_dir).parts)
        for path in Path(run_dir).rglob("*")
    )


def _read_history(path: Path) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "epoch": int(raw["epoch"]),
                    "learning_rate": float(raw["learning_rate"]),
                    "train_seconds": float(raw["train_seconds"]),
                    "epoch_seconds": float(raw["epoch_seconds"]),
                    "gpu_memory_max_allocated_mb": float(
                        raw["gpu_memory_max_allocated_mb"]
                    ),
                }
            )
    return rows


def _load_occurrence_hashes(path: Path) -> Mapping[str, object]:
    payload = _load_json(path)
    epochs = payload.get("epochs")
    if payload.get("method") != "ordered_train_sample_occurrence_sha256" or not isinstance(
        epochs, list
    ):
        raise ValueError(f"Malformed occurrence-hash artifact: {path}")
    return payload


def _source_group_metrics(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    fold_contract: Mapping[str, object],
    control: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[Dict[str, object]], Dict[str, object]]:
    rows = []
    adequate = 0
    nonworse = 0
    contract_rows = fold_contract.get("folds")
    if not isinstance(contract_rows, list):
        raise ValueError("Source fold contract omits fold rows")
    for contract in contract_rows:
        fold_index = int(contract["fold"])
        selected = folds == fold_index
        control_metrics = _metrics(labels[selected], control[selected])
        candidate_metrics = _metrics(labels[selected], candidate[selected])
        control_class1 = control_metrics["per_class"][FOCUS_CLASS]
        candidate_class1 = candidate_metrics["per_class"][FOCUS_CLASS]
        is_adequate = bool(contract["adequate_support"])
        is_nonworse = float(candidate_class1["precision"]) + 1e-12 >= float(
            control_class1["precision"]
        )
        adequate += int(is_adequate)
        nonworse += int(is_adequate and is_nonworse)
        rows.append(
            {
                **dict(contract),
                "control_class1_precision": float(control_class1["precision"]),
                "candidate_class1_precision": float(candidate_class1["precision"]),
                "class1_precision_delta": float(candidate_class1["precision"])
                - float(control_class1["precision"]),
                "control_class1_recall": float(control_class1["recall"]),
                "candidate_class1_recall": float(candidate_class1["recall"]),
                "control_class1_f1": float(control_class1["f1"]),
                "candidate_class1_f1": float(candidate_class1["f1"]),
                "precision_nonworse": bool(is_nonworse),
            }
        )
    return rows, {
        "adequate_groups": int(adequate),
        "precision_nonworse_groups": int(nonworse),
        "required_nonworse_groups": 4,
        "passed": adequate == SOURCE_FOLDS and nonworse >= 4,
    }


def assess_validity_partial_smoke_pair(
    *,
    control_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    placebo_metrics: Mapping[str, object],
    candidate_control_transitions: Mapping[str, int],
    candidate_placebo_transitions: Mapping[str, int],
    source_group_summary: Mapping[str, object],
    runtime_ratio: float,
    structural_checks: Mapping[str, bool],
) -> Dict[str, object]:
    tolerance = 1e-12
    control_per_class = control_metrics["per_class"]
    candidate_per_class = candidate_metrics["per_class"]
    placebo_per_class = placebo_metrics["per_class"]
    macro_delta = float(candidate_metrics["macro_f1"]) - float(
        control_metrics["macro_f1"]
    )
    class1_f1_delta = float(candidate_per_class[1]["f1"]) - float(
        control_per_class[1]["f1"]
    )
    class1_precision_delta = float(candidate_per_class[1]["precision"]) - float(
        control_per_class[1]["precision"]
    )
    class1_recall_delta = float(candidate_per_class[1]["recall"]) - float(
        control_per_class[1]["recall"]
    )
    worst_nonfocus_f1_delta = min(
        float(candidate_per_class[index]["f1"])
        - float(control_per_class[index]["f1"])
        for index in (0, 2, 3, 4)
    )
    placebo_f1_advantage = float(candidate_per_class[1]["f1"]) - float(
        placebo_per_class[1]["f1"]
    )
    placebo_precision_advantage = float(candidate_per_class[1]["precision"]) - float(
        placebo_per_class[1]["precision"]
    )
    checks = {
        **{str(name): bool(value) for name, value in structural_checks.items()},
        "macro_f1_preserved": macro_delta >= -0.002 - tolerance,
        "class1_f1_gain": class1_f1_delta >= 0.008 - tolerance,
        "class1_precision_gain": class1_precision_delta >= 0.015 - tolerance,
        "class1_recall_preserved": class1_recall_delta >= -0.012 - tolerance,
        "restricted_focus_fp_net_reduction": int(
            candidate_control_transitions["focus_false_positive_net_reduction"]
        )
        >= 5,
        "corrections_cover_harms": int(candidate_control_transitions["corrections"])
        >= int(candidate_control_transitions["harms"]),
        "class1_tp_breaks_bounded": int(
            candidate_control_transitions["focus_true_positives_broken"]
        )
        <= 5
        and int(candidate_control_transitions["focus_true_positives_broken"])
        <= int(candidate_control_transitions["focus_false_negatives_rescued"]) + 2,
        "nonfocus_f1_preserved": worst_nonfocus_f1_delta >= -0.012 - tolerance,
        "source_group_precision_stable": bool(source_group_summary.get("passed")),
        "aligned_beats_placebo_class1_f1": placebo_f1_advantage
        >= 0.005 - tolerance,
        "aligned_beats_placebo_class1_precision": placebo_precision_advantage
        >= 0.008 - tolerance,
        "aligned_beats_placebo_restricted_fp": int(
            candidate_placebo_transitions["focus_false_positive_net_reduction"]
        )
        >= 3,
        "runtime_ratio_bounded": math.isfinite(runtime_ratio)
        and runtime_ratio <= 1.35 + tolerance,
    }
    failed = sorted(name for name, value in checks.items() if not bool(value))
    return {
        "metric_gate_passed": not failed,
        "post_smoke_audit_required": not failed,
        "five_epoch_probe_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "route_closed": bool(failed),
        "checks": checks,
        "failed_checks": failed,
        "thresholds": {
            "minimum_macro_f1_delta": -0.002,
            "minimum_class1_f1_delta": 0.008,
            "minimum_class1_precision_delta": 0.015,
            "minimum_class1_recall_delta": -0.012,
            "minimum_restricted_fp_net_reduction": 5,
            "maximum_class1_tp_breaks": 5,
            "minimum_nonfocus_f1_delta": -0.012,
            "minimum_source_groups_nonworse": 4,
            "minimum_placebo_class1_f1_advantage": 0.005,
            "minimum_placebo_class1_precision_advantage": 0.008,
            "minimum_placebo_restricted_fp_net_reduction": 3,
            "maximum_runtime_ratio": 1.35,
        },
        "observed": {
            "macro_f1_delta": macro_delta,
            "class1_f1_delta": class1_f1_delta,
            "class1_precision_delta": class1_precision_delta,
            "class1_recall_delta": class1_recall_delta,
            "worst_nonfocus_f1_delta": worst_nonfocus_f1_delta,
            "placebo_class1_f1_advantage": placebo_f1_advantage,
            "placebo_class1_precision_advantage": placebo_precision_advantage,
            "runtime_ratio": float(runtime_ratio),
        },
    }


def _write_mask_mapping(
    path: Path,
    *,
    mapping: np.ndarray,
    bins: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    object_indices: np.ndarray,
    invalid_counts: np.ndarray,
    mask_sha256: np.ndarray,
) -> None:
    fields = [
        "sample_index",
        "target",
        "rank_bin",
        "source_stem",
        "object_index",
        "invalid_pixels",
        "mask_sha256",
        "donor_sample_index",
        "donor_source_stem",
        "donor_object_index",
        "donor_invalid_pixels",
        "donor_mask_sha256",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, donor in enumerate(mapping.tolist()):
            writer.writerow(
                {
                    "sample_index": index,
                    "target": int(labels[index]),
                    "rank_bin": int(bins[index]),
                    "source_stem": str(sources[index]),
                    "object_index": int(object_indices[index]),
                    "invalid_pixels": int(invalid_counts[index]),
                    "mask_sha256": str(mask_sha256[index]),
                    "donor_sample_index": int(donor),
                    "donor_source_stem": str(sources[donor]),
                    "donor_object_index": int(object_indices[donor]),
                    "donor_invalid_pixels": int(invalid_counts[donor]),
                    "donor_mask_sha256": str(mask_sha256[donor]),
                }
            )


def _write_source_groups(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "fold",
        "rows",
        "sources",
        "support",
        "class1_support",
        "restricted_negative_support",
        "adequate_support",
        "control_class1_precision",
        "candidate_class1_precision",
        "class1_precision_delta",
        "control_class1_recall",
        "candidate_class1_recall",
        "control_class1_f1",
        "candidate_class1_f1",
        "precision_nonworse",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            row["support"] = ",".join(str(value) for value in row["support"])
            writer.writerow(row)


def _write_placebo_predictions(
    path: Path,
    *,
    labels: np.ndarray,
    sources: np.ndarray,
    object_indices: np.ndarray,
    probabilities: np.ndarray,
    mapping: np.ndarray,
) -> None:
    fields = [
        "sample_index",
        "source_stem",
        "object_index",
        "target",
        "prediction",
        "donor_sample_index",
        *(f"prob_{index}" for index in range(5)),
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, values in enumerate(probabilities):
            row = {
                "sample_index": index,
                "source_stem": str(sources[index]),
                "object_index": int(object_indices[index]),
                "target": int(labels[index]),
                "prediction": int(np.argmax(values)),
                "donor_sample_index": int(mapping[index]),
            }
            row.update({f"prob_{class_index}": float(values[class_index]) for class_index in range(5)})
            writer.writerow(row)


def _write_changed_cases(
    path: Path,
    *,
    control: Mapping[str, object],
    candidate: Mapping[str, object],
    placebo_probabilities: np.ndarray,
    mapping: np.ndarray,
) -> int:
    fields = [
        "sample_index",
        "source_stem",
        "object_index",
        "target",
        "control_prediction",
        "candidate_prediction",
        "placebo_prediction",
        "candidate_correction",
        "candidate_harm",
        "candidate_focus_fp_removed",
        "candidate_focus_fp_created",
        "donor_sample_index",
        *(f"control_prob_{index}" for index in range(5)),
        *(f"candidate_prob_{index}" for index in range(5)),
        *(f"placebo_prob_{index}" for index in range(5)),
    ]
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (control_row, candidate_row) in enumerate(
            zip(control["rows"], candidate["rows"])
        ):
            placebo_prediction = int(np.argmax(placebo_probabilities[index]))
            control_prediction = int(control_row["prediction"])
            candidate_prediction = int(candidate_row["prediction"])
            if control_prediction == candidate_prediction == placebo_prediction:
                continue
            target = int(control_row["target"])
            row = {
                "sample_index": index,
                "source_stem": control_row["key"][1],
                "object_index": control_row["key"][2],
                "target": target,
                "control_prediction": control_prediction,
                "candidate_prediction": candidate_prediction,
                "placebo_prediction": placebo_prediction,
                "candidate_correction": int(
                    control_prediction != target and candidate_prediction == target
                ),
                "candidate_harm": int(
                    control_prediction == target and candidate_prediction != target
                ),
                "candidate_focus_fp_removed": int(
                    target in FOCUS_FP_CLASSES
                    and control_prediction == FOCUS_CLASS
                    and candidate_prediction != FOCUS_CLASS
                ),
                "candidate_focus_fp_created": int(
                    target in FOCUS_FP_CLASSES
                    and control_prediction != FOCUS_CLASS
                    and candidate_prediction == FOCUS_CLASS
                ),
                "donor_sample_index": int(mapping[index]),
            }
            for class_index in range(5):
                row[f"control_prob_{class_index}"] = float(
                    control_row["probabilities"][class_index]
                )
                row[f"candidate_prob_{class_index}"] = float(
                    candidate_row["probabilities"][class_index]
                )
                row[f"placebo_prob_{class_index}"] = float(
                    placebo_probabilities[index, class_index]
                )
            writer.writerow(row)
            count += 1
    return count


def _plot_comparison(
    path: Path,
    *,
    metrics: Mapping[str, Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
) -> None:
    roles = ("control", "candidate", "placebo")
    colors = {"control": "#4C78A8", "candidate": "#F58518", "placebo": "#9E9E9E"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(5)
    width = 0.24
    for offset, role in enumerate(roles):
        values = [float(row["f1"]) for row in metrics[role]["per_class"]]
        axes[0, 0].bar(x + (offset - 1) * width, values, width, label=role, color=colors[role])
    axes[0, 0].set_xticks(x, [str(value) for value in x])
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_title("Per-class F1")
    axes[0, 0].legend()

    names = ("precision", "recall", "f1")
    for offset, role in enumerate(roles):
        values = [float(metrics[role]["per_class"][1][name]) for name in names]
        axes[0, 1].bar(np.arange(3) + (offset - 1) * width, values, width, label=role, color=colors[role])
    axes[0, 1].set_xticks(np.arange(3), names)
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Class 1")

    confusion_delta = np.asarray(metrics["candidate"]["confusion_matrix"]) - np.asarray(
        metrics["control"]["confusion_matrix"]
    )
    limit = max(1, int(np.abs(confusion_delta).max()))
    image = axes[1, 0].imshow(confusion_delta, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[1, 0].set_title("Candidate - control confusion")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    fold_ids = [int(row["fold"]) for row in source_rows]
    fold_deltas = [float(row["class1_precision_delta"]) for row in source_rows]
    axes[1, 1].bar(fold_ids, fold_deltas, color=["#59A14F" if value >= 0 else "#E15759" for value in fold_deltas])
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(fold_ids)
    axes[1, 1].set_title("Class-1 precision delta by source fold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _align_prediction_tables(*tables: Mapping[str, object]) -> None:
    rows = [table["rows"] for table in tables]
    if len({len(value) for value in rows}) != 1:
        raise ValueError("Prediction row counts differ")
    for aligned in zip(*rows):
        first = aligned[0]
        for other in aligned[1:]:
            if (
                first["key"] != other["key"]
                or first["image_path"] != other["image_path"]
                or first["target"] != other["target"]
            ):
                raise ValueError("Prediction row/source alignment mismatch")


def _validate_run_provenance(
    *,
    control_run_dir: Path,
    candidate_run_dir: Path,
    control_summary: Mapping[str, object],
    candidate_summary: Mapping[str, object],
    control_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    control_trace: Mapping[str, object],
    candidate_trace: Mapping[str, object],
    stage_a: Mapping[str, object],
    stage_a_path: Path,
    protocol: Mapping[str, object],
    control_history: Sequence[Mapping[str, object]],
    candidate_history: Sequence[Mapping[str, object]],
    control_occurrence: Mapping[str, object],
    candidate_occurrence: Mapping[str, object],
    independent: Mapping[str, object],
    csv_control: Mapping[str, object],
    csv_candidate: Mapping[str, object],
    geometry: Mapping[str, object],
    fold_contract: Mapping[str, object],
    dataset_identity_exact: bool,
) -> tuple[Dict[str, bool], Dict[str, object]]:
    stage_gate = stage_a.get("gate")
    stage_sources = stage_a.get("sources")
    if not isinstance(stage_gate, Mapping) or not isinstance(stage_sources, Mapping):
        raise ValueError("Malformed Stage-A summary")
    control_args = protocol.get("control_arguments")
    candidate_args = protocol.get("candidate_arguments")
    if not isinstance(control_args, list) or not isinstance(candidate_args, list):
        raise ValueError("Locked pair protocol omits train arguments")
    config_differences = _deep_differences(control_config, candidate_config)
    allowed_differences = ALLOWED_ROLE_CONFIG_DIFFERENCES
    control_model_config = control_config.get("model_config")
    candidate_model_config = candidate_config.get("model_config")
    control_train_config = control_config.get("train_config")
    candidate_train_config = candidate_config.get("train_config")
    if not all(
        isinstance(value, Mapping)
        for value in (
            control_model_config,
            candidate_model_config,
            control_train_config,
            candidate_train_config,
        )
    ):
        raise ValueError("Resolved configs omit model/train configuration")

    control_lr = [float(row["learning_rate"]) for row in control_history]
    candidate_lr = [float(row["learning_rate"]) for row in candidate_history]
    control_occurrence_epochs = control_occurrence.get("epochs")
    candidate_occurrence_epochs = candidate_occurrence.get("epochs")
    independent_probabilities = independent["probabilities"]
    control_error = float(
        np.max(
            np.abs(
                np.asarray(independent_probabilities["control"])
                - np.asarray(csv_control["probabilities"])
            )
        )
    )
    candidate_error = float(
        np.max(
            np.abs(
                np.asarray(independent_probabilities["candidate"])
                - np.asarray(csv_candidate["probabilities"])
            )
        )
    )
    control_argmax_exact = np.array_equal(
        np.asarray(independent_probabilities["control"]).argmax(axis=1),
        csv_control["predictions"],
    )
    candidate_argmax_exact = np.array_equal(
        np.asarray(independent_probabilities["candidate"]).argmax(axis=1),
        csv_candidate["predictions"],
    )
    checks = {
        "stage_a_authorized": stage_a.get("method") == METHOD
        and stage_a.get("status") == "authorized"
        and bool(stage_gate.get("smoke_permission"))
        and not bool(stage_gate.get("full_train_permission")),
        "stage_a_train_only": not bool(stage_sources.get("validation_loaded"))
        and not bool(stage_sources.get("test_loaded")),
        "stage_a_hash_locked": str(protocol.get("stage_a_summary_sha256"))
        == _sha256(stage_a_path),
        "protocol_identity": protocol.get("method") == METHOD
        and protocol.get("protocol_stage") == PROTOCOL_STAGE
        and not bool(protocol.get("test_allowed")),
        "protocol_budget_locked": all(
            (
                int(protocol.get("epochs", -1)) == 2,
                int(protocol.get("scheduler_total_epochs", -1)) == 10,
                int(protocol.get("max_train_batches", -1)) == 120,
                int(protocol.get("max_val_batches", -1)) == 0,
                int(protocol.get("seed", -1)) == SEED,
                int(protocol.get("batch_size", -1)) == 32,
                int(protocol.get("grad_accum_steps", -1)) == 2,
                int(protocol.get("num_workers", -1)) == 4,
                int(protocol.get("eval_num_workers", -1)) == 2,
            )
        ),
        "arguments_matched": _normalized_train_arguments(control_args)
        == _normalized_train_arguments(candidate_args),
        "only_resolved_config_difference": set(config_differences)
        == allowed_differences,
        "convolution_values_locked": control_model_config.get("stem_convolution")
        == "standard"
        and candidate_model_config.get("stem_convolution") == "validity_partial",
        "scheduler_locked": int(control_train_config.get("scheduler_total_epochs", -1))
        == 10
        and int(candidate_train_config.get("scheduler_total_epochs", -1)) == 10
        and len(control_history) == len(candidate_history) == 2
        and control_lr == candidate_lr
        and min(control_lr + candidate_lr) > 1e-6,
        "ordered_train_rows_identical": isinstance(control_occurrence_epochs, list)
        and control_occurrence_epochs == candidate_occurrence_epochs
        and len(control_occurrence_epochs) == 2,
        "parameter_count_identical": int(control_summary.get("parameter_count", -1))
        == int(candidate_summary.get("parameter_count", -2)),
        "training_completed": int(control_summary.get("last_epoch", -1)) == 2
        and int(candidate_summary.get("last_epoch", -1)) == 2
        and str(control_summary.get("stop_reason")) == "completed"
        and str(candidate_summary.get("stop_reason")) == "completed",
        "architecture_traces_complete": bool(control_trace.get("passed"))
        and bool(candidate_trace.get("passed")),
        "no_test_output": not _contains_test_output(control_run_dir, control_summary)
        and not _contains_test_output(candidate_run_dir, candidate_summary),
        "full_validation_support": int(independent.get("rows", -1))
        == EXPECTED_VAL_ROWS,
        "independent_labels_exact": np.array_equal(
            independent["labels"], csv_control["labels"]
        )
        and np.array_equal(independent["labels"], csv_candidate["labels"]),
        "independent_control_replay": control_error
        <= MAX_INDEPENDENT_PROBABILITY_ERROR
        and control_argmax_exact,
        "independent_candidate_replay": candidate_error
        <= MAX_INDEPENDENT_PROBABILITY_ERROR
        and candidate_argmax_exact,
        "checkpoint_states_unchanged": bool(independent.get("state_unchanged")),
        "partial_conv_placement": int(independent.get("control_partial_count", -1))
        == 0
        and int(independent.get("candidate_partial_count", -1)) == 3,
        "mask_derangement_exact": bool(geometry.get("passed")),
        "source_groups_exact": bool(fold_contract.get("passed")),
        "raw_dataset_identity_exact": bool(dataset_identity_exact),
    }
    details = {
        "resolved_config_differences": config_differences,
        "allowed_config_differences": sorted(allowed_differences),
        "control_trace": control_trace,
        "candidate_trace": candidate_trace,
        "control_learning_rates": control_lr,
        "candidate_learning_rates": candidate_lr,
        "ordered_occurrence_epochs": control_occurrence_epochs,
        "independent_control_probability_max_abs_error": control_error,
        "independent_candidate_probability_max_abs_error": candidate_error,
        "independent_control_argmax_exact": bool(control_argmax_exact),
        "independent_candidate_argmax_exact": bool(candidate_argmax_exact),
        "independent_probability_threshold": MAX_INDEPENDENT_PROBABILITY_ERROR,
    }
    return checks, details


def _replay_summary(path: Path) -> Dict[str, object]:
    summary = _load_json(path)
    gate_inputs_path = Path(str(summary["gate_inputs_path"]))
    gate_inputs = _load_json(gate_inputs_path)
    replayed = assess_validity_partial_smoke_pair(**gate_inputs)
    exact = replayed == summary.get("gate")
    result = {
        "exact": bool(exact),
        "summary_path": str(Path(path).resolve()),
        "gate_inputs_path": str(gate_inputs_path.resolve()),
        "gate_inputs_sha256": _sha256(gate_inputs_path),
        "replayed_gate": replayed,
    }
    if not exact:
        raise ValueError("Validity partial-conv smoke gate replay differs")
    return result


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if args.replay_summary is not None:
        return _replay_summary(args.replay_summary)
    required = (
        args.data,
        args.control_predictions,
        args.candidate_predictions,
        args.control_run_dir,
        args.candidate_run_dir,
        args.stage_a_summary,
        args.locked_protocol,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise ValueError("Formal smoke audit requires all artifact paths")
    if int(args.batch_size) != 32 or int(args.num_workers) != 2:
        raise ValueError("Locked smoke audit requires batch-size=32 and num-workers=2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    raw_before = _dataset_tree_identity(Path(args.data))
    control_csv = _read_predictions(args.control_predictions)
    candidate_csv = _read_predictions(args.candidate_predictions)
    _align_prediction_tables(control_csv, candidate_csv)
    labels = np.asarray(control_csv["labels"], dtype=np.int64)
    if np.bincount(labels, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from the locked cohort")

    control_run_dir = Path(args.control_run_dir).resolve()
    candidate_run_dir = Path(args.candidate_run_dir).resolve()
    control_summary = _load_json(control_run_dir / "summary.json")
    candidate_summary = _load_json(candidate_run_dir / "summary.json")
    control_config = _load_json(control_run_dir / "resolved_config.json")
    candidate_config = _load_json(candidate_run_dir / "resolved_config.json")
    stage_a = _load_json(args.stage_a_summary)
    protocol = _load_json(args.locked_protocol)
    control_checkpoint_path = control_run_dir / "checkpoints" / "best.pt"
    candidate_checkpoint_path = candidate_run_dir / "checkpoints" / "best.pt"
    control_checkpoint = load_checkpoint(control_checkpoint_path, map_location="cpu")
    candidate_checkpoint = load_checkpoint(candidate_checkpoint_path, map_location="cpu")
    dataset = _build_validation_dataset(Path(args.data), candidate_checkpoint)
    if len(dataset) != EXPECTED_VAL_ROWS:
        raise ValueError("Rebuilt validation dataset row count differs")

    geometry_payload = _collect_mask_geometry(
        dataset, batch_size=int(args.batch_size), num_workers=int(args.num_workers)
    )
    dataset_labels = np.asarray(geometry_payload["labels"], dtype=np.int64)
    sources = np.asarray(geometry_payload["sources"], dtype=object)
    if not np.array_equal(dataset_labels, labels):
        raise ValueError("Rebuilt dataset labels differ from prediction CSV")
    for index, row in enumerate(control_csv["rows"]):
        expected_key = (
            index,
            str(sources[index]).casefold(),
            int(geometry_payload["object_indices"][index]),
        )
        if row["key"] != expected_key:
            raise ValueError(f"Dataset/prediction source alignment differs at row {index}")

    mapping, bins, mask_contract = build_mask_derangement(
        labels,
        sources,
        np.asarray(geometry_payload["invalid_counts"]),
        np.asarray(geometry_payload["mask_sha256"]),
    )
    folds, fold_contract = build_source_group_folds(labels, sources)
    independent = _independent_inference(
        dataset=dataset,
        control_checkpoint=control_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        mask_cache=np.asarray(geometry_payload["mask_cache"]),
        derangement=mapping,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    probabilities = independent["probabilities"]
    placebo_probabilities = np.asarray(probabilities["placebo"], dtype=np.float64)
    placebo_predictions = placebo_probabilities.argmax(axis=1)
    control_predictions = np.asarray(control_csv["predictions"], dtype=np.int64)
    candidate_predictions = np.asarray(candidate_csv["predictions"], dtype=np.int64)

    metrics = {
        "control": _metrics_with_confusion(labels, control_predictions),
        "candidate": _metrics_with_confusion(labels, candidate_predictions),
        "placebo": _metrics_with_confusion(labels, placebo_predictions),
    }
    calibration = {
        "control": _calibration(labels, np.asarray(control_csv["probabilities"])),
        "candidate": _calibration(labels, np.asarray(candidate_csv["probabilities"])),
        "placebo": _calibration(labels, placebo_probabilities),
    }
    transitions = {
        "candidate_vs_control": _transition_stats(
            labels, control_predictions, candidate_predictions
        ),
        "candidate_vs_placebo": _transition_stats(
            labels, placebo_predictions, candidate_predictions
        ),
        "placebo_vs_control": _transition_stats(
            labels, control_predictions, placebo_predictions
        ),
    }
    source_rows, source_summary = _source_group_metrics(
        labels=labels,
        folds=folds,
        fold_contract=fold_contract,
        control=control_predictions,
        candidate=candidate_predictions,
    )

    control_history = _read_history(control_run_dir / "history.csv")
    candidate_history = _read_history(candidate_run_dir / "history.csv")
    control_train_seconds = sum(float(row["train_seconds"]) for row in control_history)
    candidate_train_seconds = sum(float(row["train_seconds"]) for row in candidate_history)
    runtime_ratio = candidate_train_seconds / max(control_train_seconds, 1e-12)
    control_occurrence_path = Path(str(protocol["control_occurrence_hashes"]))
    candidate_occurrence_path = Path(str(protocol["candidate_occurrence_hashes"]))
    control_occurrence = _load_occurrence_hashes(control_occurrence_path)
    candidate_occurrence = _load_occurrence_hashes(candidate_occurrence_path)
    control_trace = _trace_summary(control_run_dir, "standard")
    candidate_trace = _trace_summary(candidate_run_dir, "validity_partial")
    raw_after = _dataset_tree_identity(Path(args.data))

    structural_checks, provenance_details = _validate_run_provenance(
        control_run_dir=control_run_dir,
        candidate_run_dir=candidate_run_dir,
        control_summary=control_summary,
        candidate_summary=candidate_summary,
        control_config=control_config,
        candidate_config=candidate_config,
        control_trace=control_trace,
        candidate_trace=candidate_trace,
        stage_a=stage_a,
        stage_a_path=Path(args.stage_a_summary),
        protocol=protocol,
        control_history=control_history,
        candidate_history=candidate_history,
        control_occurrence=control_occurrence,
        candidate_occurrence=candidate_occurrence,
        independent=independent,
        csv_control=control_csv,
        csv_candidate=candidate_csv,
        geometry=mask_contract,
        fold_contract=fold_contract,
        dataset_identity_exact=raw_before == raw_after,
    )

    mapping_path = output_dir / "mask_shape_derangement.csv"
    source_path = output_dir / "source_group_metrics.csv"
    placebo_path = output_dir / "placebo_predictions.csv"
    changed_path = output_dir / "changed_cases.csv"
    plot_path = output_dir / "metric_comparison.png"
    _write_mask_mapping(
        mapping_path,
        mapping=mapping,
        bins=bins,
        labels=labels,
        sources=sources,
        object_indices=np.asarray(geometry_payload["object_indices"]),
        invalid_counts=np.asarray(geometry_payload["invalid_counts"]),
        mask_sha256=np.asarray(geometry_payload["mask_sha256"]),
    )
    _write_source_groups(source_path, source_rows)
    _write_placebo_predictions(
        placebo_path,
        labels=labels,
        sources=sources,
        object_indices=np.asarray(geometry_payload["object_indices"]),
        probabilities=placebo_probabilities,
        mapping=mapping,
    )
    changed_rows = _write_changed_cases(
        changed_path,
        control=control_csv,
        candidate=candidate_csv,
        placebo_probabilities=placebo_probabilities,
        mapping=mapping,
    )
    _plot_comparison(plot_path, metrics=metrics, source_rows=source_rows)

    gate_inputs = {
        "control_metrics": metrics["control"],
        "candidate_metrics": metrics["candidate"],
        "placebo_metrics": metrics["placebo"],
        "candidate_control_transitions": transitions["candidate_vs_control"],
        "candidate_placebo_transitions": transitions["candidate_vs_placebo"],
        "source_group_summary": source_summary,
        "runtime_ratio": float(runtime_ratio),
        "structural_checks": structural_checks,
    }
    gate = assess_validity_partial_smoke_pair(**gate_inputs)
    gate_inputs_path = output_dir / "gate_inputs.json"
    _json_dump(gate_inputs_path, gate_inputs)

    summary = {
        "method": METHOD,
        "protocol_stage": PROTOCOL_STAGE,
        "status": "post_smoke_audit_required"
        if gate["metric_gate_passed"]
        else "route_closed",
        "sources": {
            "data_yaml": str(Path(args.data).resolve()),
            "validation_rows": int(labels.size),
            "validation_support": np.bincount(labels, minlength=5).tolist(),
            "test_used": False,
            "raw_dataset_modified": raw_before != raw_after,
            "raw_dataset_identity_before": raw_before,
            "raw_dataset_identity_after": raw_after,
            "control_checkpoint_sha256": _sha256(control_checkpoint_path),
            "candidate_checkpoint_sha256": _sha256(candidate_checkpoint_path),
            "control_predictions_sha256": _sha256(args.control_predictions),
            "candidate_predictions_sha256": _sha256(args.candidate_predictions),
            "stage_a_summary_sha256": _sha256(args.stage_a_summary),
            "locked_protocol_sha256": _sha256(args.locked_protocol),
        },
        "metrics": metrics,
        "calibration": calibration,
        "transitions": transitions,
        "source_groups": {
            "contract": fold_contract,
            "summary": source_summary,
            "rows": source_rows,
        },
        "mask_placebo": {
            "contract": mask_contract,
            "all_valid_rows": int(np.sum(np.asarray(geometry_payload["invalid_counts"]) == 0)),
            "class_unique_masks": {
                str(class_index): int(
                    len(
                        set(
                            np.asarray(geometry_payload["mask_sha256"])[
                                labels == class_index
                            ].tolist()
                        )
                    )
                )
                for class_index in range(5)
            },
            "mapping_sha256": _sha256(mapping_path),
        },
        "independent_reload": {
            key: value
            for key, value in independent.items()
            if key != "probabilities" and key != "labels"
        },
        "provenance": {
            "checks": structural_checks,
            "failed_checks": [
                name for name, value in structural_checks.items() if not value
            ],
            "details": provenance_details,
        },
        "resources": {
            "control_train_seconds": float(control_train_seconds),
            "candidate_train_seconds": float(candidate_train_seconds),
            "runtime_ratio": float(runtime_ratio),
            "control_peak_vram_gib": max(
                float(row["gpu_memory_max_allocated_mb"]) for row in control_history
            )
            / 1024.0,
            "candidate_peak_vram_gib": max(
                float(row["gpu_memory_max_allocated_mb"])
                for row in candidate_history
            )
            / 1024.0,
            "mask_census_loader": geometry_payload["loader"],
            "independent_loader": independent["loader"],
        },
        "changed_case_rows": int(changed_rows),
        "gate_inputs_path": str(gate_inputs_path.resolve()),
        "gate": gate,
        "current_best_command_updated": False,
        "test_used": False,
    }
    summary_path = output_dir / "summary.json"
    _json_dump(summary_path, summary)
    replay = _replay_summary(summary_path)
    replay_path = output_dir / "independent_replay.json"
    _json_dump(replay_path, replay)
    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            (
                "# Validity Partial-Conv Matched Smoke",
                "",
                f"- Status: `{summary['status']}`",
                f"- Gate passed: `{gate['metric_gate_passed']}`",
                f"- Failed checks: `{gate['failed_checks']}`",
                f"- Runtime ratio: `{runtime_ratio:.6f}`",
                f"- Validation rows: `{labels.size}`",
                "- Test used: `false`",
                "- Current-best command updated: `false`",
                "",
            )
        ),
        encoding="utf-8",
    )

    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    for role, path in (
        ("control_checkpoint", control_checkpoint_path),
        ("candidate_checkpoint", candidate_checkpoint_path),
        ("control_predictions", Path(args.control_predictions)),
        ("candidate_predictions", Path(args.candidate_predictions)),
        ("control_trace", control_run_dir / "architecture_trace" / "trace_summary.json"),
        ("candidate_trace", candidate_run_dir / "architecture_trace" / "trace_summary.json"),
    ):
        artifacts.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    _json_dump(
        output_dir / "artifact_manifest.json",
        {
            "artifacts": artifacts,
            "raw_dataset_modified": bool(raw_before != raw_after),
            "test_used": False,
            "full_train_authorized": False,
            "current_best_command_updated": False,
        },
    )
    return summary


def main() -> None:
    print(json.dumps(run_audit(parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
