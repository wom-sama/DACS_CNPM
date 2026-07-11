from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_head_input_from_features,
)
from trkh.models.photometric_invariant import (
    ColorInvariantW,
    invariant_response_to_model_view,
    suppress_invalid_boundary,
)
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification


LITERATURE = [
    "https://openaccess.thecvf.com/content/ICCV2021/html/Lengyel_Zero-Shot_Day-Night_Domain_Adaptation_With_a_Physics_Prior_ICCV_2021_paper.html",
    "https://github.com/Attila94/CIConv",
    "https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf",
    "https://www.cs.sfu.ca/~mark/ftp/Eccv04/",
]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a fixed CIConv-W view adds fold-safe class signal to "
            "the frozen TRKH RGB embedding. The tool reads train and validation only."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--invariant-scale", type=float, default=0.0)
    parser.add_argument("--invariant-clip", type=float, default=3.0)
    parser.add_argument("--invalid-boundary-width", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--logistic-c", type=float, default=0.3)
    parser.add_argument("--logistic-max-iter", type=int, default=600)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_rgb_cache(path: Path, *, max_samples: int = 0) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as payload:
        required = {"embeddings", "probabilities", "labels", "paths", "sample_index"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"RGB cache is missing keys {missing}: {path}")
        result: Dict[str, object] = {
            "embeddings": np.asarray(payload["embeddings"], dtype=np.float32),
            "probabilities": np.asarray(payload["probabilities"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.int64).reshape(-1),
            "paths": np.asarray(payload["paths"], dtype=object).reshape(-1),
            "sample_index": np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1),
        }
    row_count = int(np.asarray(result["labels"]).shape[0])
    for key in ("embeddings", "probabilities", "paths", "sample_index"):
        if int(np.asarray(result[key]).shape[0]) != row_count:
            raise ValueError(f"RGB cache key {key} has inconsistent row count")
    if int(max_samples) > 0:
        limit = min(row_count, int(max_samples))
        result = {key: np.asarray(value)[:limit] for key, value in result.items()}
    if not np.isfinite(np.asarray(result["embeddings"], dtype=np.float32)).all():
        raise ValueError(f"RGB cache contains non-finite embeddings: {path}")
    return result


def _normalize_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).resolve(strict=False)).replace("/", "\\").casefold()


def _source_stems(paths: Sequence[object]) -> np.ndarray:
    return np.asarray([Path(str(path)).stem.casefold() for path in paths], dtype=object)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _effective_rank(features: np.ndarray, *, max_rows: int = 4096) -> float:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] > int(max_rows):
        indices = np.linspace(0, values.shape[0] - 1, num=int(max_rows), dtype=np.int64)
        values = values[indices]
    values = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    energy = np.square(singular_values)
    probabilities = energy / max(float(energy.sum()), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return float(math.exp(entropy))


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str],
    focus_class: int = 1,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    class_count = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(class_count)),
        zero_division=0,
    )
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class = [
        {
            "class_index": int(index),
            "class_name": str(class_names[index]),
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index in range(class_count)
    ]
    return {
        "accuracy": float((predictions == labels).mean()) if labels.size else 0.0,
        "macro_f1": float(f1.mean()) if f1.size else 0.0,
        "focus_f1": float(f1[int(focus_class)]),
        "focus_precision": float(precision[int(focus_class)]),
        "focus_recall": float(recall[int(focus_class)]),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _aligned_predict_proba(estimator, features: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    final_estimator = estimator
    if hasattr(estimator, "named_steps"):
        final_estimator = list(estimator.named_steps.values())[-1]
    classes = np.asarray(getattr(final_estimator, "classes_"), dtype=np.int64)
    aligned = np.zeros((raw.shape[0], int(class_count)), dtype=np.float64)
    for column, class_index in enumerate(classes):
        aligned[:, int(class_index)] = raw[:, int(column)]
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return aligned.astype(np.float32)


def _readout_estimator(*, c_value: float, max_iter: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(max_iter),
            random_state=int(seed),
            solver="lbfgs",
        ),
    )


def _fit_grouped_readout(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    folds: int,
    c_value: float,
    max_iter: int,
    seed: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    class_count = len(class_names)
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    split_indices = list(
        splitter.split(np.zeros(train_labels.shape[0]), train_labels, groups=train_groups)
    )
    oof_probabilities = np.zeros((train_labels.shape[0], class_count), dtype=np.float32)
    template = _readout_estimator(c_value=c_value, max_iter=max_iter, seed=seed)
    fold_iterations: List[int] = []
    for fold_index, (fit_indices, holdout_indices) in enumerate(split_indices):
        estimator = clone(template)
        estimator.fit(train_features[fit_indices], train_labels[fit_indices])
        oof_probabilities[holdout_indices] = _aligned_predict_proba(
            estimator,
            train_features[holdout_indices],
            class_count,
        )
        final_estimator = list(estimator.named_steps.values())[-1]
        fold_iterations.append(int(np.asarray(final_estimator.n_iter_).max()))
        print(
            f"readout fold {fold_index + 1}/{len(split_indices)} complete; "
            f"fit={len(fit_indices)} holdout={len(holdout_indices)}",
            flush=True,
        )
    final_model = clone(template)
    final_model.fit(train_features, train_labels)
    val_probabilities = _aligned_predict_proba(final_model, val_features, class_count)
    final_estimator = list(final_model.named_steps.values())[-1]
    return {
        "oof_probabilities": oof_probabilities,
        "val_probabilities": val_probabilities,
        "oof_metrics": _classification_metrics(
            train_labels,
            oof_probabilities,
            class_names=class_names,
        ),
        "val_metrics": None,
        "fold_iterations": fold_iterations,
        "final_iterations": int(np.asarray(final_estimator.n_iter_).max()),
    }


def _transition_summary(
    labels: np.ndarray,
    source_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    focus_class: int = 1,
) -> Dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    source = np.asarray(source_probabilities).argmax(axis=1)
    target = np.asarray(target_probabilities).argmax(axis=1)
    changed = source != target
    source_correct = source == labels
    target_correct = target == labels
    return {
        "changed": int(changed.sum()),
        "corrections": int((changed & ~source_correct & target_correct).sum()),
        "harms": int((changed & source_correct & ~target_correct).sum()),
        "neutral": int((changed & (source_correct == target_correct)).sum()),
        "class1_fn_rescued": int(
            ((labels == focus_class) & (source != focus_class) & (target == focus_class)).sum()
        ),
        "class1_tp_broken": int(
            ((labels == focus_class) & (source == focus_class) & (target != focus_class)).sum()
        ),
        "class1_fp_removed": int(
            ((labels != focus_class) & (source == focus_class) & (target != focus_class)).sum()
        ),
        "class1_fp_created": int(
            ((labels != focus_class) & (source != focus_class) & (target == focus_class)).sum()
        ),
    }


def assess_photometric_invariant_readiness(
    *,
    train_samples: int,
    val_samples: int,
    alignment_ok: bool,
    rgb_effective_rank: float,
    invariant_effective_rank: float,
    invariant_border_mass: float,
    direct_metrics: Mapping[str, object],
    rgb_oof_metrics: Mapping[str, object],
    combined_oof_metrics: Mapping[str, object],
    rgb_val_metrics: Mapping[str, object],
    combined_val_metrics: Mapping[str, object],
    direct_transitions: Mapping[str, int],
    class1_error_delta_auc: float,
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_invariant_to_rgb_effective_rank_ratio": 0.75,
        "max_invariant_border_mass": 0.45,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_val_class1_f1": 0.70,
        "max_direct_macro_drop": 0.003,
        "min_class1_error_delta_auc": 0.60,
    }
    oof_macro_gain = float(combined_oof_metrics["macro_f1"]) - float(rgb_oof_metrics["macro_f1"])
    oof_class1_gain = float(combined_oof_metrics["focus_f1"]) - float(rgb_oof_metrics["focus_f1"])
    val_macro_gain = float(combined_val_metrics["macro_f1"]) - float(rgb_val_metrics["macro_f1"])
    val_class1_gain = float(combined_val_metrics["focus_f1"]) - float(rgb_val_metrics["focus_f1"])
    direct_macro_drop = float(direct_metrics["macro_f1"]) - float(combined_val_metrics["macro_f1"])
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "cache_alignment": bool(alignment_ok),
        "invariant_rank_preserved": float(invariant_effective_rank)
        / max(float(rgb_effective_rank), 1e-8)
        >= float(thresholds["min_invariant_to_rgb_effective_rank_ratio"]),
        "invariant_not_border_dominated": float(invariant_border_mass)
        <= float(thresholds["max_invariant_border_mass"]),
        "oof_macro_gain": oof_macro_gain >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": oof_class1_gain >= float(thresholds["min_oof_class1_gain"]),
        "val_macro_gain": val_macro_gain >= float(thresholds["min_val_macro_gain"]),
        "val_class1_gain": val_class1_gain >= float(thresholds["min_val_class1_gain"]),
        "val_class1_milestone": float(combined_val_metrics["focus_f1"])
        >= float(thresholds["min_val_class1_f1"]),
        "direct_macro_preserved": direct_macro_drop <= float(thresholds["max_direct_macro_drop"]),
        "net_corrections_nonnegative": int(direct_transitions["corrections"])
        >= int(direct_transitions["harms"]),
        "class1_recall_protected": int(direct_transitions["class1_fn_rescued"])
        >= int(direct_transitions["class1_tp_broken"]),
        "class1_false_positive_control": int(direct_transitions["class1_fp_removed"])
        >= int(direct_transitions["class1_fp_created"]),
        "class1_error_direction_separable": float(class1_error_delta_auc)
        >= float(thresholds["min_class1_error_delta_auc"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "photometric_invariant_target_ready": not failed,
        "smoke_ready": not failed,
        "smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": {
            "oof_macro_gain": oof_macro_gain,
            "oof_class1_gain": oof_class1_gain,
            "val_macro_gain": val_macro_gain,
            "val_class1_gain": val_class1_gain,
            "direct_macro_drop": direct_macro_drop,
            "class1_error_delta_auc": float(class1_error_delta_auc),
            "rgb_effective_rank": float(rgb_effective_rank),
            "invariant_effective_rank": float(invariant_effective_rank),
            "invariant_to_rgb_effective_rank_ratio": float(invariant_effective_rank)
            / max(float(rgb_effective_rank), 1e-8),
        },
        "thresholds": thresholds,
    }


def _border_mass(response: Tensor) -> Tensor:
    height, width = int(response.shape[-2]), int(response.shape[-1])
    border_y = max(1, int(round(height * 0.10)))
    border_x = max(1, int(round(width * 0.10)))
    mask = torch.zeros_like(response, dtype=torch.float32)
    mask[..., :border_y, :] = 1.0
    mask[..., -border_y:, :] = 1.0
    mask[..., :, :border_x] = 1.0
    mask[..., :, -border_x:] = 1.0
    energy = response.float().abs()
    return (energy * mask).flatten(1).sum(dim=1) / energy.flatten(1).sum(dim=1).clamp_min(1e-8)


def _extract_invariant_split(
    *,
    model: torch.nn.Module,
    invariant: ColorInvariantW,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    mean: Sequence[float],
    std: Sequence[float],
    clip: float,
    invalid_boundary_width: int,
    split: str,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in dataset_paths_fn()]
        if callable(dataset_paths_fn)
        else []
    )
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    embeddings: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    paths: List[str] = []
    border_values: List[np.ndarray] = []
    clip_values: List[np.ndarray] = []
    response_abs_values: List[np.ndarray] = []
    preview: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    seen = 0
    model.eval()
    invariant.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"extract-ciconv-w-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            batch_count = int(images.size(0))
            batch_paths: List[str] = []
            if isinstance(metadata, Mapping):
                raw_paths = metadata.get("paths", [])
                if isinstance(raw_paths, Sequence):
                    batch_paths = [str(path) for path in raw_paths]
            if dataset_paths and (
                len(batch_paths) != batch_count
                or not any(str(path).strip() for path in batch_paths)
            ):
                batch_paths = dataset_paths[seen : seen + batch_count]
            seen += batch_count

            bbox_metadata = None
            image_valid_mask = None
            if isinstance(metadata, Mapping):
                bbox_value = metadata.get("bbox")
                if torch.is_tensor(bbox_value):
                    bbox_metadata = bbox_value.to(device=device, dtype=torch.float32)
                mask_value = metadata.get("image_mask")
                if torch.is_tensor(mask_value):
                    image_valid_mask = mask_value.to(device=device, dtype=torch.bool)

            rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            response_raw = invariant(rgb)
            response = suppress_invalid_boundary(
                response_raw,
                image_valid_mask,
                boundary_width=int(invalid_boundary_width),
            )
            invariant_view = invariant_response_to_model_view(response, clip=float(clip))
            border_values.append(_border_mass(response).cpu().numpy())
            clip_values.append((response_raw.abs() > float(clip)).float().flatten(1).mean(dim=1).cpu().numpy())
            response_abs_values.append(response.abs().flatten(1).mean(dim=1).cpu().numpy())

            for row_index, target in enumerate(labels.detach().cpu().tolist()):
                if int(target) in preview:
                    continue
                rgb_image = (
                    rgb[row_index].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
                response_image = response[row_index, 0].detach().float().cpu().numpy()
                preview[int(target)] = (rgb_image, response_image)

            with autocast_context(device, bool(amp)):
                features = model.forward_features(
                    invariant_view,
                    image_valid_mask=image_valid_mask,
                    bbox_token_prior=bbox_metadata,
                )
                if bbox_metadata is not None:
                    features["bbox"] = bbox_metadata
                if hasattr(model, "forward_heads"):
                    model_output = model.forward_heads(features)
                else:
                    model_output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(model_output)
                head_input = extract_head_input_from_features(model, features)
            embeddings.append(head_input.detach().float().cpu().numpy())
            probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            paths.extend(batch_paths)
    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
        "probabilities": np.concatenate(probabilities, axis=0).astype(np.float32, copy=False),
        "labels": np.concatenate(labels_all, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "border_mass": np.concatenate(border_values, axis=0).astype(np.float32, copy=False),
        "clip_fraction": np.concatenate(clip_values, axis=0).astype(np.float32, copy=False),
        "response_abs_mean": np.concatenate(response_abs_values, axis=0).astype(np.float32, copy=False),
        "preview": preview,
    }


def _write_preview(
    path: Path,
    preview: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    class_names: Sequence[str],
    *,
    clip: float,
) -> None:
    tile = 192
    rows: List[Image.Image] = []
    manifest: List[Dict[str, object]] = []
    resampling = getattr(Image, "Resampling", Image)
    for class_index in range(len(class_names)):
        if class_index not in preview:
            continue
        rgb, response = preview[class_index]
        rgb_image = Image.fromarray(rgb).resize((tile, tile), resampling.BILINEAR)
        mapped = np.clip((response / (2.0 * float(clip))) + 0.5, 0.0, 1.0)
        invariant_image = Image.fromarray((mapped * 255.0).round().astype(np.uint8))
        invariant_image = invariant_image.convert("RGB").resize((tile, tile), resampling.BILINEAR)
        row = Image.new("RGB", (tile * 2, tile), color=(255, 255, 255))
        row.paste(rgb_image, (0, 0))
        row.paste(invariant_image, (tile, 0))
        rows.append(row)
        manifest.append({"row": len(rows) - 1, "class_index": class_index, "class_name": class_names[class_index]})
    if not rows:
        return
    canvas = Image.new("RGB", (tile * 2, tile * len(rows)), color=(255, 255, 255))
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, row_index * tile))
    canvas.save(path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_prediction_audit(
    path: Path,
    *,
    labels: np.ndarray,
    sample_index: np.ndarray,
    paths: np.ndarray,
    base_probabilities: np.ndarray,
    variants: Mapping[str, np.ndarray],
) -> None:
    names = ["base"] + list(variants.keys())
    payload = {"base": np.asarray(base_probabilities, dtype=np.float32)}
    payload.update({name: np.asarray(values, dtype=np.float32) for name, values in variants.items()})
    class_count = int(base_probabilities.shape[1])
    fields = ["sample_index", "path", "target"]
    for name in names:
        fields.extend([f"{name}_prediction", f"{name}_correct"])
        fields.extend([f"{name}_prob_{index}" for index in range(class_count)])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index in range(labels.shape[0]):
            row: Dict[str, object] = {
                "sample_index": int(sample_index[row_index]),
                "path": str(paths[row_index]),
                "target": int(labels[row_index]),
            }
            for name in names:
                probabilities = payload[name][row_index]
                prediction = int(probabilities.argmax())
                row[f"{name}_prediction"] = prediction
                row[f"{name}_correct"] = int(prediction == int(labels[row_index]))
                for class_index in range(class_count):
                    row[f"{name}_prob_{class_index}"] = f"{float(probabilities[class_index]):.10g}"
            writer.writerow(row)


def _save_invariant_cache(
    path: Path,
    *,
    split_payload: Mapping[str, object],
    sample_index: np.ndarray,
    class_names: Sequence[str],
) -> None:
    np.savez_compressed(
        path,
        features=np.asarray(split_payload["embeddings"], dtype=np.float32),
        probabilities=np.asarray(split_payload["probabilities"], dtype=np.float32),
        labels=np.asarray(split_payload["labels"], dtype=np.int64),
        paths=np.asarray(split_payload["paths"], dtype=object),
        sample_index=np.asarray(sample_index, dtype=np.int64),
        source_stem=_source_stems(np.asarray(split_payload["paths"], dtype=object)),
        classes=np.asarray(class_names, dtype=object),
        invariant=np.asarray(["CIConv-W"], dtype=object),
    )


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.folds) < 2:
        raise ValueError("folds must be at least 2")
    if float(args.logistic_c) <= 0.0:
        raise ValueError("logistic-c must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device).eval()
    invariant = ColorInvariantW(scale=float(args.invariant_scale), trainable_scale=False).to(device).eval()
    mean, std = checkpoint_input_normalization(checkpoint)

    rgb_cache_paths = {
        "train": Path(args.rgb_cache_dir) / "train_embeddings.npz",
        "val": Path(args.rgb_cache_dir) / "val_embeddings.npz",
    }
    max_samples = {
        "train": int(args.max_train_samples),
        "val": int(args.max_val_samples),
    }
    rgb_cache = {
        split: _load_rgb_cache(path, max_samples=max_samples[split])
        for split, path in rgb_cache_paths.items()
    }

    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    start = time.perf_counter()
    for split in ("train", "val"):
        dataset, class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples[split],
        )
        split_payloads[split] = _extract_invariant_split(
            model=model,
            invariant=invariant,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            mean=mean,
            std=std,
            clip=float(args.invariant_clip),
            invalid_boundary_width=int(args.invalid_boundary_width),
            split=split,
        )

    alignment: Dict[str, Dict[str, object]] = {}
    for split in ("train", "val"):
        cached = rgb_cache[split]
        extracted = split_payloads[split]
        cached_labels = np.asarray(cached["labels"], dtype=np.int64)
        extracted_labels = np.asarray(extracted["labels"], dtype=np.int64)
        cached_paths = np.asarray(cached["paths"], dtype=object)
        extracted_paths = np.asarray(extracted["paths"], dtype=object)
        labels_equal = bool(np.array_equal(cached_labels, extracted_labels))
        paths_equal = bool(
            len(cached_paths) == len(extracted_paths)
            and all(
                _normalize_path(left) == _normalize_path(right)
                for left, right in zip(cached_paths, extracted_paths)
            )
        )
        row_count_equal = int(cached_labels.shape[0]) == int(extracted_labels.shape[0])
        alignment[split] = {
            "row_count_equal": row_count_equal,
            "labels_equal": labels_equal,
            "paths_equal": paths_equal,
            "rows": int(extracted_labels.shape[0]),
        }
        if not (row_count_equal and labels_equal and paths_equal):
            raise ValueError(f"RGB/invariant cache alignment failed for {split}: {alignment[split]}")

    train_labels = np.asarray(rgb_cache["train"]["labels"], dtype=np.int64)
    val_labels = np.asarray(rgb_cache["val"]["labels"], dtype=np.int64)
    train_rgb = _l2_normalize(np.asarray(rgb_cache["train"]["embeddings"], dtype=np.float32))
    val_rgb = _l2_normalize(np.asarray(rgb_cache["val"]["embeddings"], dtype=np.float32))
    train_w = _l2_normalize(np.asarray(split_payloads["train"]["embeddings"], dtype=np.float32))
    val_w = _l2_normalize(np.asarray(split_payloads["val"]["embeddings"], dtype=np.float32))
    train_features = {
        "rgb": train_rgb,
        "w": train_w,
        "rgb_w": np.concatenate((train_rgb, train_w), axis=1),
    }
    val_features = {
        "rgb": val_rgb,
        "w": val_w,
        "rgb_w": np.concatenate((val_rgb, val_w), axis=1),
    }
    train_groups = _source_stems(np.asarray(rgb_cache["train"]["paths"], dtype=object))
    readouts: Dict[str, Dict[str, object]] = {}
    for name in ("rgb", "w", "rgb_w"):
        print(f"fitting grouped OOF readout: {name}", flush=True)
        result = _fit_grouped_readout(
            train_features=train_features[name],
            train_labels=train_labels,
            train_groups=train_groups,
            val_features=val_features[name],
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

    direct_train_probabilities = np.asarray(rgb_cache["train"]["probabilities"], dtype=np.float32)
    direct_val_probabilities = np.asarray(rgb_cache["val"]["probabilities"], dtype=np.float32)
    direct_metrics = _classification_metrics(
        val_labels,
        direct_val_probabilities,
        class_names=class_names,
    )
    direct_transitions = _transition_summary(
        val_labels,
        direct_val_probabilities,
        np.asarray(readouts["rgb_w"]["val_probabilities"], dtype=np.float32),
    )
    rgb_to_combined_transitions = _transition_summary(
        val_labels,
        np.asarray(readouts["rgb"]["val_probabilities"], dtype=np.float32),
        np.asarray(readouts["rgb_w"]["val_probabilities"], dtype=np.float32),
    )

    direct_predictions = direct_val_probabilities.argmax(axis=1)
    recall_mask = (val_labels == 1) & (direct_predictions != 1)
    false_positive_mask = (val_labels != 1) & (direct_predictions == 1)
    error_mask = recall_mask | false_positive_mask
    error_labels = recall_mask[error_mask].astype(np.int64)
    probability_delta = (
        np.asarray(readouts["rgb_w"]["val_probabilities"], dtype=np.float32)[:, 1]
        - np.asarray(readouts["rgb"]["val_probabilities"], dtype=np.float32)[:, 1]
    )
    class1_error_delta_auc = (
        float(roc_auc_score(error_labels, probability_delta[error_mask]))
        if int(np.unique(error_labels).size) == 2
        else 0.5
    )

    train_border = np.asarray(split_payloads["train"]["border_mass"], dtype=np.float32)
    val_border = np.asarray(split_payloads["val"]["border_mass"], dtype=np.float32)
    rgb_effective_rank = _effective_rank(train_rgb)
    invariant_effective_rank = _effective_rank(train_w)
    invariant_border_mass = float(np.mean(np.concatenate((train_border, val_border))))
    alignment_ok = all(
        bool(payload["row_count_equal"] and payload["labels_equal"] and payload["paths_equal"])
        for payload in alignment.values()
    )
    gate = assess_photometric_invariant_readiness(
        train_samples=int(train_labels.shape[0]),
        val_samples=int(val_labels.shape[0]),
        alignment_ok=alignment_ok,
        rgb_effective_rank=rgb_effective_rank,
        invariant_effective_rank=invariant_effective_rank,
        invariant_border_mass=invariant_border_mass,
        direct_metrics=direct_metrics,
        rgb_oof_metrics=readouts["rgb"]["oof_metrics"],
        combined_oof_metrics=readouts["rgb_w"]["oof_metrics"],
        rgb_val_metrics=readouts["rgb"]["val_metrics"],
        combined_val_metrics=readouts["rgb_w"]["val_metrics"],
        direct_transitions=direct_transitions,
        class1_error_delta_auc=class1_error_delta_auc,
    )

    for split in ("train", "val"):
        _save_invariant_cache(
            output_dir / f"{split}_ciconv_w_embeddings.npz",
            split_payload=split_payloads[split],
            sample_index=np.asarray(rgb_cache[split]["sample_index"], dtype=np.int64),
            class_names=class_names,
        )
        _write_prediction_audit(
            output_dir / f"{split}_readout_predictions.csv",
            labels=np.asarray(rgb_cache[split]["labels"], dtype=np.int64),
            sample_index=np.asarray(rgb_cache[split]["sample_index"], dtype=np.int64),
            paths=np.asarray(rgb_cache[split]["paths"], dtype=object),
            base_probabilities=(
                direct_train_probabilities if split == "train" else direct_val_probabilities
            ),
            variants={
                name: np.asarray(
                    readouts[name][
                        "oof_probabilities" if split == "train" else "val_probabilities"
                    ],
                    dtype=np.float32,
                )
                for name in ("rgb", "w", "rgb_w")
            },
        )
    _write_preview(
        output_dir / "ciconv_w_preview_train.png",
        split_payloads["train"]["preview"],
        class_names,
        clip=float(args.invariant_clip),
    )

    compact_readouts = {
        name: {
            "feature_dim": int(train_features[name].shape[1]),
            "oof_metrics": payload["oof_metrics"],
            "val_metrics": payload["val_metrics"],
            "fold_iterations": payload["fold_iterations"],
            "final_iterations": payload["final_iterations"],
        }
        for name, payload in readouts.items()
    }
    summary = {
        "mode": "photometric_invariant_complementarity_precheck",
        "guardrail": (
            "Frozen keeper plus train-only grouped OOF readouts; train and validation only. "
            "No raw-data edit, checkpoint write, trainable manifest, test use, or model smoke."
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "rgb_cache_dir": str(Path(args.rgb_cache_dir).resolve()),
        "invariant": {
            "name": "CIConv-W",
            "scale": float(args.invariant_scale),
            "clip": float(args.invariant_clip),
            "invalid_boundary_width": int(args.invalid_boundary_width),
            "trainable_scale": False,
            "rgb_branch_preserved": True,
        },
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "class_names": class_names,
        "alignment": alignment,
        "source_group_count": int(np.unique(train_groups).size),
        "readout": {
            "folds": int(args.folds),
            "logistic_c": float(args.logistic_c),
            "max_iter": int(args.logistic_max_iter),
            "selection": "none; one predeclared estimator and CIConv-W scale",
            "variants": compact_readouts,
        },
        "direct_keeper_val_metrics": direct_metrics,
        "invariant_diagnostics": {
            "rgb_train_effective_rank": rgb_effective_rank,
            "train_effective_rank": invariant_effective_rank,
            "train_effective_rank_ratio_to_rgb": invariant_effective_rank
            / max(rgb_effective_rank, 1e-8),
            "train_border_mass_mean": float(train_border.mean()),
            "val_border_mass_mean": float(val_border.mean()),
            "combined_border_mass_mean": invariant_border_mass,
            "train_clip_fraction_mean": float(
                np.asarray(split_payloads["train"]["clip_fraction"], dtype=np.float32).mean()
            ),
            "val_clip_fraction_mean": float(
                np.asarray(split_payloads["val"]["clip_fraction"], dtype=np.float32).mean()
            ),
            "train_response_abs_mean": float(
                np.asarray(split_payloads["train"]["response_abs_mean"], dtype=np.float32).mean()
            ),
            "val_response_abs_mean": float(
                np.asarray(split_payloads["val"]["response_abs_mean"], dtype=np.float32).mean()
            ),
        },
        "transitions": {
            "direct_keeper_to_rgb_w_readout": direct_transitions,
            "rgb_readout_to_rgb_w_readout": rgb_to_combined_transitions,
            "direct_class1_fn_support": int(recall_mask.sum()),
            "direct_class1_fp_support": int(false_positive_mask.sum()),
            "class1_error_delta_auc": class1_error_delta_auc,
        },
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - start),
        "literature": LITERATURE,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_written": False,
        "trainable_manifest_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readout_val = compact_readouts["rgb_w"]["val_metrics"]
    readme = [
        "# Photometric-Invariant Complementarity Precheck",
        "",
        f"- Train/validation rows: `{train_labels.shape[0]}/{val_labels.shape[0]}`",
        f"- Direct keeper macro/class1 F1: `{float(direct_metrics['macro_f1']):.6f}/{float(direct_metrics['focus_f1']):.6f}`",
        f"- RGB+W readout macro/class1 F1: `{float(readout_val['macro_f1']):.6f}/{float(readout_val['focus_f1']):.6f}`",
        f"- CIConv-W effective rank: `{invariant_effective_rank:.3f}`",
        f"- Class1 FN-vs-FP delta AUC: `{class1_error_delta_auc:.6f}`",
        f"- Smoke ready: `{str(bool(gate['smoke_ready'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "This diagnostic preserves the RGB branch and uses no test split or model training.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
