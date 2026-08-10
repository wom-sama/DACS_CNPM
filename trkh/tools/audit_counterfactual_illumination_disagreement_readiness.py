from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import (
    IMAGE_EXTENSIONS,
    MangoYOLOCropDataset,
    build_eval_transform,
)
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    LightingShift,
    _forward_classification_with_metadata,
    _transform_classification_image,
    _unpack_classification_sample,
)
from trkh.inference.inference import load_model
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
FOCUS_CLASS = 1
CONDITIONS: Tuple[Tuple[str, float, float], ...] = (
    ("clean", 1.0, 1.0),
    ("lighting_dim", 0.7, 0.9),
    ("lighting_bright", 1.25, 1.1),
    ("low_contrast", 1.0, 0.65),
)
EVENT_NAMES: Tuple[str, ...] = (
    "focus_fn_rescue",
    "focus_tp_break",
    "focus_fp_remove_correct",
    "focus_fp_create",
    "candidate_correction",
    "candidate_harm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train-only counterfactual illumination disagreement density "
            "before any CIDT trainer smoke."
        )
    )
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-keeper-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    observed = _sha256(path)
    if observed != str(expected).strip().lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return observed


def _require_empty_output(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_balanced_indices(labels: Sequence[int], max_samples: int) -> List[int]:
    labels = [int(value) for value in labels]
    if int(max_samples) <= 0 or int(max_samples) >= len(labels):
        return list(range(len(labels)))
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    selected: List[int] = []
    positions = {label: 0 for label in by_class}
    class_order = sorted(by_class)
    while len(selected) < int(max_samples):
        made_progress = False
        for label in class_order:
            position = positions[label]
            candidates = by_class[label]
            if position >= len(candidates):
                continue
            selected.append(candidates[position])
            positions[label] = position + 1
            made_progress = True
            if len(selected) >= int(max_samples):
                break
        if not made_progress:
            break
    return selected


def directional_event_masks(
    targets: np.ndarray,
    keeper_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    focus_class: int = FOCUS_CLASS,
) -> Dict[str, np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    keeper = np.asarray(keeper_predictions, dtype=np.int64).reshape(-1)
    candidate = np.asarray(candidate_predictions, dtype=np.int64).reshape(-1)
    if targets.size != keeper.size or targets.size != candidate.size:
        raise ValueError("Directional-event arrays must have equal lengths.")
    focus = int(focus_class)
    return {
        "focus_fn_rescue": (targets == focus) & (keeper != focus) & (candidate == focus),
        "focus_tp_break": (targets == focus) & (keeper == focus) & (candidate != focus),
        "focus_fp_remove_correct": (targets != focus) & (keeper == focus) & (candidate == targets),
        "focus_fp_create": (targets != focus) & (keeper == targets) & (candidate == focus),
        "candidate_correction": (keeper != targets) & (candidate == targets),
        "candidate_harm": (keeper == targets) & (candidate != targets),
    }


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    confusion = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(1.0, support.sum())),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "support": support.astype(np.int64).tolist(),
        "predicted_support": predicted_support.astype(np.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _assign_source_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).astype(str).reshape(-1)
    if labels.size != groups.size:
        raise ValueError("Fold labels and groups do not align.")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments = np.full(labels.size, -1, dtype=np.int64)
    rows: List[Dict[str, object]] = []
    for fold_index, (fit_rows, holdout_rows) in enumerate(
        splitter.split(np.zeros(labels.size), labels, groups)
    ):
        fit_sources = set(groups[fit_rows].tolist())
        holdout_sources = set(groups[holdout_rows].tolist())
        overlap = fit_sources.intersection(holdout_sources)
        assignments[holdout_rows] = int(fold_index)
        rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(fit_rows.size),
                "holdout_rows": int(holdout_rows.size),
                "fit_sources": int(len(fit_sources)),
                "holdout_sources": int(len(holdout_sources)),
                "source_overlap": int(len(overlap)),
                "holdout_class_counts": np.bincount(
                    labels[holdout_rows], minlength=int(labels.max(initial=0) + 1)
                ).tolist(),
            }
        )
    if bool((assignments < 0).any()):
        raise RuntimeError("Source-fold assignment is incomplete.")
    return assignments, {
        "fold_count": int(folds),
        "source_overlap": int(sum(int(row["source_overlap"]) for row in rows)),
        "assignment_complete": True,
        "assignment_counts": np.bincount(assignments, minlength=int(folds)).tolist(),
        "folds": rows,
    }


class _SelectedConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset: MangoYOLOCropDataset,
        selected_indices: Sequence[int],
        *,
        corruption,
        transform,
    ) -> None:
        self.base_dataset = base_dataset
        self.selected_indices = tuple(int(index) for index in selected_indices)
        self.corruption = corruption
        self.transform = transform

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, position: int):
        sample_index = self.selected_indices[int(position)]
        image, label, metadata, bbox = _unpack_classification_sample(
            self.base_dataset[sample_index]
        )
        image = self.corruption(image)
        tensor, transformed_metadata = _transform_classification_image(
            image,
            label=label,
            metadata=metadata,
            bbox=bbox,
            transform=self.transform,
        )
        transformed_metadata["sample_index"] = torch.tensor(
            sample_index, dtype=torch.long
        )
        return tensor, label, transformed_metadata


def _build_eval_transform(semantics: Mapping[str, object]):
    return build_eval_transform(
        image_size=int(semantics["image_size"]),
        resize_mode=str(semantics["resize_mode"]),
        illumination_normalization=bool(semantics["illumination_normalization"]),
        illumination_normalization_strength=float(
            semantics["illumination_normalization_strength"]
        ),
        foreground_crop_mode=str(semantics["foreground_crop_mode"]),
        foreground_crop_margin_ratio=float(semantics["foreground_crop_margin_ratio"]),
        foreground_crop_min_mask_area_ratio=float(
            semantics["foreground_crop_min_mask_area_ratio"]
        ),
        foreground_crop_max_mask_area_ratio=float(
            semantics["foreground_crop_max_mask_area_ratio"]
        ),
        foreground_crop_max_crop_area_ratio=float(
            semantics["foreground_crop_max_crop_area_ratio"]
        ),
        background_suppression_mode=str(semantics["background_suppression_mode"]),
        background_suppression_margin=float(semantics["background_suppression_margin"]),
        background_suppression_blur_radius=float(
            semantics["background_suppression_blur_radius"]
        ),
        surface_detail_amplification_mode=str(
            semantics["surface_detail_amplification_mode"]
        ),
        surface_detail_amplification_strength=float(
            semantics["surface_detail_amplification_strength"]
        ),
        surface_detail_amplification_blur_radius=float(
            semantics["surface_detail_amplification_blur_radius"]
        ),
        surface_detail_amplification_foreground_weight=float(
            semantics["surface_detail_amplification_foreground_weight"]
        ),
        eval_surface_detail_amplification=bool(
            semantics["eval_surface_detail_amplification"]
        ),
    )


def _split_source_stems(images_dir: Path) -> set[str]:
    return {
        path.stem.casefold()
        for path in Path(images_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _dataset_identity(
    dataset: MangoYOLOCropDataset,
    selected_indices: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, Sequence) or len(samples) != len(dataset):
        raise ValueError("Dataset sample metadata is unavailable.")
    for index in selected_indices:
        sample = samples[int(index)]
        primary_index = int(getattr(sample, "primary_object_index"))
        primary_object = sample.objects[primary_index]
        row = {
            "sample_index": int(index),
            "image_path": str(Path(sample.image_path).resolve()),
            "label_path": str(Path(sample.label_path).resolve()),
            "target": int(sample.primary_label),
            "object_index": int(primary_object.object_index),
            "bbox": [float(value) for value in primary_object.bbox],
        }
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _condition_corruption(brightness: float, contrast: float):
    if math.isclose(float(brightness), 1.0) and math.isclose(float(contrast), 1.0):
        return IdentityCorruption()
    return LightingShift(brightness=float(brightness), contrast=float(contrast))


def _audit_condition(
    *,
    name: str,
    brightness: float,
    contrast: float,
    base_dataset: MangoYOLOCropDataset,
    selected_indices: Sequence[int],
    transform,
    keeper_model,
    candidate_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        selected_indices,
        corruption=_condition_corruption(brightness, contrast),
        transform=transform,
    )
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context=f"cidt_{name}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    all_indices: List[Tensor] = []
    all_targets: List[Tensor] = []
    all_keeper: List[Tensor] = []
    all_candidate: List[Tensor] = []
    started = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(dtype=torch.long)
            keeper_logits, keeper_features = _forward_classification_with_metadata(
                keeper_model, images, metadata, device=device
            )
            del keeper_features
            candidate_logits, candidate_features = _forward_classification_with_metadata(
                candidate_model, images, metadata, device=device
            )
            del candidate_features
            keeper_probabilities = F.softmax(keeper_logits.float(), dim=1)
            candidate_probabilities = F.softmax(candidate_logits.float(), dim=1)
            all_indices.append(metadata["sample_index"].detach().cpu().to(torch.long))
            all_targets.append(targets.detach().cpu())
            all_keeper.append(keeper_probabilities.detach().cpu())
            all_candidate.append(candidate_probabilities.detach().cpu())
            processed += int(targets.numel())
            if processed % 1024 < int(targets.numel()) or processed == len(dataset):
                print(
                    json.dumps(
                        {
                            "condition": name,
                            "processed": processed,
                            "rows": len(dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    indices = torch.cat(all_indices).numpy()
    targets = torch.cat(all_targets).numpy()
    keeper_probabilities = torch.cat(all_keeper).numpy()
    candidate_probabilities = torch.cat(all_candidate).numpy()
    if indices.tolist() != [int(index) for index in selected_indices]:
        raise ValueError(f"Condition {name} changed ordered sample identity.")
    for label, probabilities in (
        ("keeper", keeper_probabilities),
        ("candidate", candidate_probabilities),
    ):
        if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
            raise ValueError(f"Condition {name} has invalid {label} probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"Condition {name} has non-normalized {label} probabilities.")
    return {
        "name": name,
        "brightness": float(brightness),
        "contrast": float(contrast),
        "indices": indices,
        "targets": targets,
        "keeper_probabilities": keeper_probabilities,
        "candidate_probabilities": candidate_probabilities,
        "loader": loader_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def summarize_directional_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    fold_count: int,
) -> Dict[str, object]:
    rows = list(rows)
    nonclean = [row for row in rows if str(row["condition"]) != "clean"]
    clean = [row for row in rows if str(row["condition"]) == "clean"]

    def event_rows(items: Sequence[Mapping[str, object]], event: str):
        return [row for row in items if bool(row[event])]

    event_counts = {event: len(event_rows(nonclean, event)) for event in EVENT_NAMES}
    unique_rows = {
        event: len({int(row["sample_index"]) for row in event_rows(nonclean, event)})
        for event in EVENT_NAMES
    }
    source_counts = {
        event: len({str(row["source_stem"]) for row in event_rows(nonclean, event)})
        for event in EVENT_NAMES
    }
    rescue_source_histogram = Counter(
        str(row["source_stem"]) for row in event_rows(nonclean, "focus_fn_rescue")
    )
    rescue_events = max(1, int(event_counts["focus_fn_rescue"]))
    maximum_rescue_source_share = (
        max(rescue_source_histogram.values(), default=0) / rescue_events
    )
    fold_rows: List[Dict[str, object]] = []
    for fold in range(int(fold_count)):
        selected = [row for row in nonclean if int(row["fold"]) == fold]
        counts = {
            event: sum(1 for row in selected if bool(row[event]))
            for event in EVENT_NAMES
        }
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(selected)),
                **counts,
                "has_focus_rescue": bool(counts["focus_fn_rescue"] > 0),
                "rescue_exceeds_tp_break": bool(
                    counts["focus_fn_rescue"] > counts["focus_tp_break"]
                ),
            }
        )
    return {
        "nonclean_rows": int(len(nonclean)),
        "event_counts": event_counts,
        "unique_sample_counts": unique_rows,
        "source_group_counts": source_counts,
        "clean_unique_focus_fn_rescue": int(
            len(
                {
                    int(row["sample_index"])
                    for row in event_rows(clean, "focus_fn_rescue")
                }
            )
        ),
        "maximum_rescue_source_share": float(maximum_rescue_source_share),
        "folds_with_focus_rescue": int(
            sum(bool(row["has_focus_rescue"]) for row in fold_rows)
        ),
        "folds_rescue_exceeds_tp_break": int(
            sum(bool(row["rescue_exceeds_tp_break"]) for row in fold_rows)
        ),
        "folds": fold_rows,
    }


def _build_rows(
    condition_results: Sequence[Mapping[str, object]],
    *,
    selected_paths: Mapping[int, Path],
    fold_by_index: Mapping[int, int],
    class_names: Sequence[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    condition_summaries: List[Dict[str, object]] = []
    clean_keeper_predictions: Dict[int, int] = {}
    clean_candidate_predictions: Dict[int, int] = {}
    clean_targets: Dict[int, int] = {}

    for result in condition_results:
        condition = str(result["name"])
        indices = np.asarray(result["indices"], dtype=np.int64)
        targets = np.asarray(result["targets"], dtype=np.int64)
        keeper_probabilities = np.asarray(result["keeper_probabilities"], dtype=np.float64)
        candidate_probabilities = np.asarray(
            result["candidate_probabilities"], dtype=np.float64
        )
        keeper_predictions = keeper_probabilities.argmax(axis=1)
        candidate_predictions = candidate_probabilities.argmax(axis=1)
        masks = directional_event_masks(
            targets, keeper_predictions, candidate_predictions
        )
        if condition == "clean":
            clean_keeper_predictions = dict(zip(indices.tolist(), keeper_predictions.tolist()))
            clean_candidate_predictions = dict(
                zip(indices.tolist(), candidate_predictions.tolist())
            )
            clean_targets = dict(zip(indices.tolist(), targets.tolist()))
        for position, sample_index in enumerate(indices.tolist()):
            target = int(targets[position])
            keeper_prediction = int(keeper_predictions[position])
            candidate_prediction = int(candidate_predictions[position])
            row: Dict[str, object] = {
                "condition": condition,
                "sample_index": int(sample_index),
                "source_stem": selected_paths[int(sample_index)].stem.casefold(),
                "image_path": str(selected_paths[int(sample_index)].resolve()),
                "fold": int(fold_by_index[int(sample_index)]),
                "target_index": target,
                "target_name": str(class_names[target]),
                "keeper_prediction": keeper_prediction,
                "candidate_prediction": candidate_prediction,
                "keeper_correct": bool(keeper_prediction == target),
                "candidate_correct": bool(candidate_prediction == target),
                "keeper_target_probability": float(keeper_probabilities[position, target]),
                "candidate_target_probability": float(
                    candidate_probabilities[position, target]
                ),
                "keeper_focus_probability": float(
                    keeper_probabilities[position, FOCUS_CLASS]
                ),
                "candidate_focus_probability": float(
                    candidate_probabilities[position, FOCUS_CLASS]
                ),
            }
            for class_index in range(len(class_names)):
                row[f"keeper_prob_{class_index}"] = float(
                    keeper_probabilities[position, class_index]
                )
                row[f"candidate_prob_{class_index}"] = float(
                    candidate_probabilities[position, class_index]
                )
            for event in EVENT_NAMES:
                row[event] = bool(masks[event][position])
            rows.append(row)
        condition_summaries.append(
            {
                "condition": condition,
                "brightness": float(result["brightness"]),
                "contrast": float(result["contrast"]),
                "rows": int(indices.size),
                "keeper_metrics": _classification_metrics(
                    targets, keeper_predictions, num_classes=len(class_names)
                ),
                "candidate_metrics": _classification_metrics(
                    targets, candidate_predictions, num_classes=len(class_names)
                ),
                "events": {
                    event: int(masks[event].sum()) for event in EVENT_NAMES
                },
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "loader": result["loader"],
            }
        )

    if not clean_keeper_predictions:
        raise ValueError("Clean condition is required for retention accounting.")
    for summary, result in zip(condition_summaries, condition_results):
        condition = str(summary["condition"])
        indices = np.asarray(result["indices"], dtype=np.int64)
        targets = np.asarray(result["targets"], dtype=np.int64)
        keeper_predictions = np.asarray(result["keeper_probabilities"]).argmax(axis=1)
        candidate_predictions = np.asarray(result["candidate_probabilities"]).argmax(axis=1)
        clean_keeper_correct = np.asarray(
            [clean_keeper_predictions[int(index)] == clean_targets[int(index)] for index in indices],
            dtype=bool,
        )
        clean_candidate_correct = np.asarray(
            [clean_candidate_predictions[int(index)] == clean_targets[int(index)] for index in indices],
            dtype=bool,
        )
        clean_keeper_focus_tp = np.asarray(
            [
                clean_targets[int(index)] == FOCUS_CLASS
                and clean_keeper_predictions[int(index)] == FOCUS_CLASS
                for index in indices
            ],
            dtype=bool,
        )
        summary["keeper_clean_correct_retention"] = float(
            ((keeper_predictions == targets) & clean_keeper_correct).sum()
            / max(1, int(clean_keeper_correct.sum()))
        )
        summary["candidate_clean_correct_retention"] = float(
            ((candidate_predictions == targets) & clean_candidate_correct).sum()
            / max(1, int(clean_candidate_correct.sum()))
        )
        summary["keeper_clean_focus_tp_retention"] = float(
            ((keeper_predictions == FOCUS_CLASS) & clean_keeper_focus_tp).sum()
            / max(1, int(clean_keeper_focus_tp.sum()))
        )
        summary["condition_is_clean"] = condition == "clean"
    return rows, condition_summaries


def audit_readiness(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    if int(args.folds) < 2:
        raise ValueError("folds must be at least 2.")
    if int(args.max_samples) < 0 or 0 < int(args.max_samples) < int(args.folds) * 5:
        raise ValueError("max_samples must be 0 or at least folds * 5.")

    output_dir = _require_empty_output(args.output_dir)
    keeper_path = Path(args.keeper).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    keeper_hash = _verify_sha256(
        keeper_path, args.expected_keeper_sha256, "keeper checkpoint"
    )
    candidate_hash = _verify_sha256(
        candidate_path, args.expected_candidate_sha256, "candidate checkpoint"
    )
    data_hash = _sha256(data_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    set_seed(int(args.seed), deterministic=True)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    keeper_model, keeper_checkpoint, keeper_classes = load_model(keeper_path, device)
    candidate_model, candidate_checkpoint, candidate_classes = load_model(
        candidate_path, device
    )
    if keeper_classes != candidate_classes:
        raise ValueError("Keeper and candidate class orders differ.")
    keeper_semantics = _eval_semantics(keeper_checkpoint)
    candidate_semantics = _eval_semantics(candidate_checkpoint)
    if keeper_semantics != candidate_semantics:
        raise ValueError("Keeper and candidate evaluation semantics differ.")
    if int(keeper_semantics["temporal_frames"]) != 1:
        raise ValueError("CIDT readiness supports temporal_frames=1 only.")

    data_spec = load_data_spec(
        data_path, class_name_mode="raw", expected_num_classes=len(keeper_classes)
    )
    if list(data_spec.class_names) != list(keeper_classes):
        raise ValueError("Dataset class order differs from checkpoints.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(keeper_semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(keeper_checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    labels = [int(value) for value in base_dataset.labels()]
    paths = [Path(value) for value in base_dataset.sample_paths()]
    selected_indices = select_balanced_indices(labels, int(args.max_samples))
    selected_labels = np.asarray([labels[index] for index in selected_indices], dtype=np.int64)
    selected_groups = np.asarray(
        [paths[index].stem.casefold() for index in selected_indices], dtype=str
    )
    fold_assignments, fold_summary = _assign_source_folds(
        selected_labels,
        selected_groups,
        folds=int(args.folds),
        seed=int(args.seed),
    )
    fold_by_index = {
        int(sample_index): int(fold)
        for sample_index, fold in zip(selected_indices, fold_assignments.tolist())
    }
    selected_paths = {int(index): paths[int(index)] for index in selected_indices}

    all_train_sources = {path.stem.casefold() for path in paths}
    val_sources = _split_source_stems(data_spec.val_images)
    test_sources = (
        _split_source_stems(data_spec.test_images)
        if data_spec.test_images is not None
        else set()
    )
    train_val_overlap = sorted(all_train_sources.intersection(val_sources))
    train_test_overlap = sorted(all_train_sources.intersection(test_sources))
    dataset_identity_sha256 = _dataset_identity(base_dataset, selected_indices)

    transform = _build_eval_transform(keeper_semantics)
    condition_results: List[Dict[str, object]] = []
    for name, brightness, contrast in CONDITIONS:
        condition_results.append(
            _audit_condition(
                name=name,
                brightness=brightness,
                contrast=contrast,
                base_dataset=base_dataset,
                selected_indices=selected_indices,
                transform=transform,
                keeper_model=keeper_model,
                candidate_model=candidate_model,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
        )

    rows, condition_summaries = _build_rows(
        condition_results,
        selected_paths=selected_paths,
        fold_by_index=fold_by_index,
        class_names=keeper_classes,
    )
    directional = summarize_directional_rows(rows, fold_count=int(args.folds))
    nonclean_conditions = [
        summary for summary in condition_summaries if not summary["condition_is_clean"]
    ]
    unique_counts = directional["unique_sample_counts"]
    event_counts = directional["event_counts"]
    source_counts = directional["source_group_counts"]
    full_audit = int(args.max_samples) == 0 and len(selected_indices) == len(base_dataset)

    structural_gates = {
        "train_split_only": True,
        "checkpoint_eval_semantics_exact": keeper_semantics == candidate_semantics,
        "condition_contract_exact": len(condition_results) == len(CONDITIONS),
        "ordered_rows_complete": len(rows)
        == len(selected_indices) * len(CONDITIONS),
        "source_fold_assignment_complete": bool(fold_summary["assignment_complete"]),
        "source_fold_overlap_zero": int(fold_summary["source_overlap"]) == 0,
        "train_val_source_overlap_zero": len(train_val_overlap) == 0,
        "train_test_source_overlap_zero": len(train_test_overlap) == 0,
        "probabilities_finite_normalized": True,
    }
    readiness_gates = {
        "full_train_support": full_audit
        and len(base_dataset) == EXPECTED_TRAIN_ROWS
        and len(selected_indices) == EXPECTED_TRAIN_ROWS,
        "transform_label_retention": all(
            float(summary["keeper_clean_correct_retention"]) >= 0.80
            and float(summary["keeper_clean_focus_tp_retention"]) >= 0.65
            for summary in nonclean_conditions
        ),
        "focus_rescue_events_30": int(event_counts["focus_fn_rescue"]) >= 30,
        "focus_rescue_unique_rows_20": int(unique_counts["focus_fn_rescue"]) >= 20,
        "focus_rescue_source_groups_15": int(source_counts["focus_fn_rescue"]) >= 15,
        "focus_rescue_max_source_share_020": float(
            directional["maximum_rescue_source_share"]
        )
        <= 0.20,
        "counterfactual_rescue_expansion_10": int(unique_counts["focus_fn_rescue"])
        >= int(directional["clean_unique_focus_fn_rescue"]) + 10,
        "focus_rescue_fold_coverage_4": int(directional["folds_with_focus_rescue"])
        >= 4,
        "focus_rescue_favorable_folds_3": int(
            directional["folds_rescue_exceeds_tp_break"]
        )
        >= 3,
        "focus_tp_break_unique_rows_10": int(unique_counts["focus_tp_break"]) >= 10,
        "focus_fp_create_unique_rows_30": int(unique_counts["focus_fp_create"])
        >= 30,
    }
    scope_gates_passed = all(structural_gates.values())
    all_gates_passed = full_audit and scope_gates_passed and all(
        readiness_gates.values()
    )

    prediction_path = output_dir / "predictions_all_conditions.csv"
    cohort_path = output_dir / "directional_cohorts.csv"
    fold_path = output_dir / "fold_direction_summary.csv"
    condition_path = output_dir / "condition_summary.json"
    _write_csv(prediction_path, rows)
    _write_csv(
        cohort_path,
        [row for row in rows if any(bool(row[event]) for event in EVENT_NAMES)],
    )
    _write_csv(fold_path, directional["folds"])
    _write_json(condition_path, condition_summaries)

    summary = {
        "mode": "counterfactual_illumination_disagreement_train_readiness",
        "test_data_used": False,
        "validation_predictions_used": False,
        "split": "train",
        "audit_scope": "full" if full_audit else "probe",
        "keeper": str(keeper_path),
        "keeper_sha256": keeper_hash,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "data_yaml": str(data_path),
        "data_yaml_sha256": data_hash,
        "dataset_identity_sha256": dataset_identity_sha256,
        "device": str(device),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
        },
        "rows": int(len(selected_indices)),
        "full_expected_rows": EXPECTED_TRAIN_ROWS,
        "source_groups": int(len(set(selected_groups.tolist()))),
        "class_names": list(keeper_classes),
        "focus_class": FOCUS_CLASS,
        "eval_semantics": keeper_semantics,
        "conditions": [
            {"name": name, "brightness": brightness, "contrast": contrast}
            for name, brightness, contrast in CONDITIONS
        ],
        "cross_split_source_audit": {
            "train_sources": int(len(all_train_sources)),
            "val_sources": int(len(val_sources)),
            "test_sources": int(len(test_sources)),
            "train_val_overlap": int(len(train_val_overlap)),
            "train_test_overlap": int(len(train_test_overlap)),
            "overlap_examples": (train_val_overlap + train_test_overlap)[:20],
            "note": "Only split filenames were scanned; val/test labels, pixels, predictions, and metrics were not loaded.",
        },
        "source_folds": fold_summary,
        "condition_summaries": condition_summaries,
        "directional": directional,
        "structural_gates": structural_gates,
        "readiness_gates": readiness_gates,
        "scope_gates_passed": scope_gates_passed,
        "all_gates_passed": all_gates_passed,
        "artifacts": {
            "predictions": prediction_path.name,
            "directional_cohorts": cohort_path.name,
            "fold_summary": fold_path.name,
            "condition_summary": condition_path.name,
        },
        "guardrail": (
            "Passing authorizes one fixed no-test trainer smoke only. It does not "
            "authorize threshold, transform, loss-weight, seed, fork, or run-length sweeps."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = audit_readiness(args)
    print(
        json.dumps(
            {
                "audit_scope": summary["audit_scope"],
                "rows": summary["rows"],
                "event_counts": summary["directional"]["event_counts"],
                "unique_sample_counts": summary["directional"][
                    "unique_sample_counts"
                ],
                "scope_gates_passed": summary["scope_gates_passed"],
                "all_gates_passed": summary["all_gates_passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not bool(summary["scope_gates_passed"]):
        raise SystemExit(1)
    if summary["audit_scope"] == "full" and not bool(summary["all_gates_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
