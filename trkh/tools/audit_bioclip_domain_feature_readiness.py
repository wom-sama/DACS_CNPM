from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.tools.audit_bbn_bilateral_classifier_readiness import (
    _apply_scaler,
    _fit_classifier,
    _fit_scaler,
    _softmax,
)
from trkh.tools.audit_multistage_teacher_feature_readiness import (
    _alignment_plan,
    _compact_alignment,
    _imagefolder_rows,
    _projected_effective_rank,
)
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _load_cache,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


SEED = 20260712
FOLDS = 5
LOGISTIC_C = 0.3
MAX_ITERATIONS = 400
BATCH_SIZE = 64
FOCUS_CLASS_INDEX = 1
FOCUS_MILESTONE = 0.70
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
EMBEDDING_DIMENSION = 512
OCCLUSION_GRID = 4
XAI_CASES = 12
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

BIOCLIP_REPO = "imageomics/bioclip"
BIOCLIP_REVISION = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c"
BIOCLIP_FILENAME = "open_clip_pytorch_model.bin"
BIOCLIP_SHA256 = "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"
CONTROL_REPO = "timm/vit_base_patch16_clip_224.openai"
CONTROL_REVISION = "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d"
CONTROL_FILENAME = "open_clip_model.safetensors"
CONTROL_SHA256 = "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"

LITERATURE = (
    "https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html",
    "https://github.com/Imageomics/bioclip",
    "https://huggingface.co/imageomics/bioclip",
    "https://github.com/mlfoundations/open_clip",
)


class PathImageFolder(ImageFolder):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image, int(target), str(self.samples[int(index)][0])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only matched OpenAI CLIP versus BioCLIP frozen-feature "
            "readiness audit. It never reads test, edits raw data, or writes a model."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--keeper-train-cache", type=Path, required=True)
    parser.add_argument("--keeper-val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(r"D:\DataAI\.cache\huggingface\hub"),
    )
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--logistic-c", type=float, default=LOGISTIC_C)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--occlusion-grid", type=int, default=OCCLUSION_GRID)
    parser.add_argument("--xai-cases", type=int, default=XAI_CASES)
    parser.add_argument("--save-embeddings", action="store_true", default=False)
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_protocol(args: argparse.Namespace) -> None:
    if int(args.folds) < 3:
        raise ValueError("At least three source-grouped folds are required")
    if float(args.logistic_c) <= 0.0 or int(args.max_iterations) < 100:
        raise ValueError("Readout settings are invalid")
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("DataLoader settings are invalid")
    if int(args.occlusion_grid) < 2 or int(args.xai_cases) < 4:
        raise ValueError("XAI settings are invalid")
    if int(args.focus_class_index) != FOCUS_CLASS_INDEX:
        raise ValueError("This locked audit supports class 1 only")
    output_dir = Path(args.output_dir).resolve()
    classification_root = Path(args.classification_root).resolve()
    yolo_root = Path(args.yolo_data).resolve().parent
    if _is_relative_to(output_dir, classification_root) or _is_relative_to(
        output_dir, yolo_root
    ):
        raise ValueError("Output directory must stay outside both raw dataset views")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    device = torch.device(requested) if requested else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(value: object) -> str:
    return str(Path(str(value)).resolve(strict=False)).replace("/", "\\").casefold()


def _normalized_source_group(value: object) -> str:
    return str(value).strip().replace("/", "\\").casefold()


def _model_spec(role: str) -> Dict[str, object]:
    normalized = str(role).strip().lower()
    if normalized == "control":
        return {
            "role": "control",
            "model_name": "ViT-B-16-quickgelu",
            "repo_id": CONTROL_REPO,
            "revision": CONTROL_REVISION,
            "filename": CONTROL_FILENAME,
            "checkpoint_sha256": CONTROL_SHA256,
            "domain": "OpenAI CLIP web image-text control",
        }
    if normalized == "bioclip":
        return {
            "role": "bioclip",
            "model_name": "ViT-B-16",
            "repo_id": BIOCLIP_REPO,
            "revision": BIOCLIP_REVISION,
            "filename": BIOCLIP_FILENAME,
            "checkpoint_sha256": BIOCLIP_SHA256,
            "domain": "TreeOfLife-10M biology-domain adaptation",
        }
    raise ValueError(f"Unknown model role: {role}")


def _build_model(
    role: str,
    *,
    device: torch.device,
    model_cache: Path,
) -> Tuple[torch.nn.Module, object, Dict[str, object]]:
    spec = _model_spec(role)
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=str(spec["repo_id"]),
            filename=str(spec["filename"]),
            revision=str(spec["revision"]),
            cache_dir=str(Path(model_cache).resolve()),
        )
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != str(spec["checkpoint_sha256"]):
        raise RuntimeError(
            f"Locked {role} checkpoint SHA-256 mismatch: {checkpoint_sha256}"
        )
    model, _, transform = open_clip.create_model_and_transforms(
        str(spec["model_name"]),
        pretrained=str(checkpoint_path),
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
        image_interpolation="bicubic",
        image_resize_mode="shortest",
    )
    model.to(device).eval()
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("OpenCLIP model has no visual encoder")
    feature_dim = int(getattr(visual, "output_dim", EMBEDDING_DIMENSION))
    first_block = visual.transformer.resblocks[0]
    metadata = {
        **spec,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_sha256": checkpoint_sha256,
        "visual_parameter_count": int(sum(p.numel() for p in visual.parameters())),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "embedding_dimension": feature_dim,
        "activation": type(first_block.mlp.gelu).__name__,
        "transform": repr(transform),
        "open_clip_version": str(getattr(open_clip, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
    }
    if feature_dim != EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected embedding dimension: {feature_dim}")
    return model, transform, metadata


def _build_alignment(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, np.ndarray]]:
    plans: Dict[str, Dict[str, object]] = {}
    aligned_classification_paths: Dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        rows, classes = _imagefolder_rows(Path(args.classification_root), split)
        plan = _alignment_plan(
            classification_rows=rows,
            classification_classes=classes,
            yolo_data=Path(args.yolo_data),
            split=split,
            class_name_mode=str(args.class_name_mode),
        )
        plans[split] = plan
        aligned_classification_paths[split] = np.asarray(
            [rows[int(index)][0] for index in np.asarray(plan["row_indices"], dtype=np.int64)],
            dtype=object,
        )
    if list(plans["train"]["class_names"]) != list(plans["val"]["class_names"]):
        raise ValueError("Train/validation class order differs")
    return plans, aligned_classification_paths


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
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    paths: List[str] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, targets, batch_paths in tqdm(
            loader,
            desc=f"clip-feature-{split}",
            dynamic_ncols=True,
        ):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, bool(amp)):
                encoded = model.encode_image(images, normalize=True)
            features.append(encoded.float().cpu().numpy().astype(np.float32, copy=False))
            labels.append(targets.numpy().astype(np.int64, copy=False))
            paths.extend(str(path) for path in batch_paths)
    if not features:
        raise ValueError(f"Empty classification split: {classification_root / split}")
    feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    norms = np.linalg.norm(feature_array.astype(np.float64), axis=1)
    return {
        "features": feature_array,
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
        "paths": np.asarray(paths, dtype=object),
        "classes": list(dataset.classes),
        "norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "finite": bool(np.isfinite(feature_array).all()),
        "seconds": float(time.perf_counter() - start),
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
    folder_labels = np.asarray(payload["labels"], dtype=np.int64)
    expected_folder_rows = int(plan["classification_rows"])
    if folder_labels.shape[0] != expected_folder_rows:
        raise ValueError("Extracted ImageFolder row count differs from alignment plan")
    aligned_folder_labels = folder_labels[indices]
    target_labels = np.asarray(plan["labels"], dtype=np.int64)
    _assert_aligned_class_names(
        aligned_folder_labels,
        payload["classes"],
        target_labels,
        plan["class_names"],
    )
    return {
        "features": np.asarray(payload["features"], dtype=np.float32)[indices],
        "labels": target_labels,
        "sample_index": np.asarray(plan["sample_index"], dtype=np.int64),
        "paths": np.asarray(plan["paths"], dtype=object),
        "source_stems": np.asarray(plan["source_stems"], dtype=object),
        "class_names": list(plan["class_names"]),
        "norm_max_abs_error": float(payload["norm_max_abs_error"]),
        "finite": bool(payload["finite"]),
        "seconds": float(payload["seconds"]),
        "peak_cuda_memory_mib": float(payload["peak_cuda_memory_mib"]),
    }


def _assert_aligned_class_names(
    folder_labels: np.ndarray,
    folder_classes: Sequence[str],
    target_labels: np.ndarray,
    target_classes: Sequence[str],
) -> None:
    folder_labels = np.asarray(folder_labels, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    if folder_labels.shape != target_labels.shape:
        raise ValueError("Aligned folder/target label arrays differ in shape")
    folder_names = np.asarray(
        [str(folder_classes[int(label)]) for label in folder_labels],
        dtype=object,
    )
    target_names = np.asarray(
        [str(target_classes[int(label)]) for label in target_labels],
        dtype=object,
    )
    if not np.array_equal(folder_names, target_names):
        mismatch = np.flatnonzero(folder_names != target_names)[:5].tolist()
        raise ValueError(f"Aligned feature class names differ at rows: {mismatch}")


def _readout_probabilities(model, features: np.ndarray) -> np.ndarray:
    return _softmax(np.asarray(model.decision_function(features), dtype=np.float64))


def fit_matched_source_readouts(
    *,
    control_train: np.ndarray,
    bioclip_train: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    control_val: np.ndarray,
    bioclip_val: np.ndarray,
    val_labels: np.ndarray,
    folds: int,
    logistic_c: float,
    max_iterations: int,
    workers: int,
    seed: int,
    focus_class_index: int,
) -> Dict[str, object]:
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_groups = np.asarray(train_groups, dtype=object)
    class_count = int(np.max(train_labels)) + 1
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    split_indices = list(splitter.split(control_train, train_labels, groups=train_groups))
    oof = {
        "control": np.zeros((len(train_labels), class_count), dtype=np.float32),
        "bioclip": np.zeros((len(train_labels), class_count), dtype=np.float32),
    }
    fold_assignment = np.full(len(train_labels), -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    fold_protocols: List[Dict[str, object]] = []
    maximum_source_overlap = 0
    converged = True
    feature_sets = {
        "control": np.asarray(control_train, dtype=np.float32),
        "bioclip": np.asarray(bioclip_train, dtype=np.float32),
    }
    for fold_index, (fit_indices, hold_indices) in enumerate(split_indices):
        fit_sources = set(train_groups[fit_indices].tolist())
        hold_sources = set(train_groups[hold_indices].tolist())
        source_overlap = len(fit_sources.intersection(hold_sources))
        maximum_source_overlap = max(maximum_source_overlap, source_overlap)
        telemetry: Dict[str, object] = {}
        for role, values in feature_sets.items():
            mean, scale = _fit_scaler(values[fit_indices])
            fit_scaled = _apply_scaler(values[fit_indices], mean, scale)
            hold_scaled = _apply_scaler(values[hold_indices], mean, scale)
            classifier = _fit_classifier(
                fit_scaled,
                train_labels[fit_indices],
                logistic_c=float(logistic_c),
                max_iterations=int(max_iterations),
                workers=int(workers),
                sample_weights=None,
            )
            probabilities = _readout_probabilities(classifier, hold_scaled)
            oof[role][hold_indices] = probabilities
            iterations = int(np.asarray(classifier.n_iter_).max())
            role_converged = iterations < int(max_iterations)
            converged = converged and role_converged
            telemetry[role] = {
                "iterations": iterations,
                "converged": bool(role_converged),
            }
        fold_assignment[hold_indices] = int(fold_index)
        control_metrics = _classification_metrics(train_labels[hold_indices], oof["control"][hold_indices])
        bioclip_metrics = _classification_metrics(train_labels[hold_indices], oof["bioclip"][hold_indices])
        control_focus = control_metrics["per_class"][int(focus_class_index)]
        bioclip_focus = bioclip_metrics["per_class"][int(focus_class_index)]
        fold_rows.append(
            {
                "fold": int(fold_index),
                "rows": int(len(hold_indices)),
                "source_overlap": int(source_overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "control_focus_f1": float(control_focus["f1"]),
                "bioclip_macro_f1": float(bioclip_metrics["macro_f1"]),
                "bioclip_focus_f1": float(bioclip_focus["f1"]),
                "macro_gain": float(bioclip_metrics["macro_f1"] - control_metrics["macro_f1"]),
                "focus_gain": float(bioclip_focus["f1"] - control_focus["f1"]),
            }
        )
        fold_protocols.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(source_overlap),
                "readouts": telemetry,
            }
        )
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF fold assignment is incomplete")

    val_probabilities: Dict[str, np.ndarray] = {}
    final_state: Dict[str, Dict[str, object]] = {}
    val_feature_sets = {
        "control": np.asarray(control_val, dtype=np.float32),
        "bioclip": np.asarray(bioclip_val, dtype=np.float32),
    }
    for role, values in feature_sets.items():
        mean, scale = _fit_scaler(values)
        train_scaled = _apply_scaler(values, mean, scale)
        val_scaled = _apply_scaler(val_feature_sets[role], mean, scale)
        classifier = _fit_classifier(
            train_scaled,
            train_labels,
            logistic_c=float(logistic_c),
            max_iterations=int(max_iterations),
            workers=int(workers),
            sample_weights=None,
        )
        val_probabilities[role] = _readout_probabilities(classifier, val_scaled)
        iterations = int(np.asarray(classifier.n_iter_).max())
        role_converged = iterations < int(max_iterations)
        converged = converged and role_converged
        final_state[role] = {
            "mean": mean,
            "scale": scale,
            "classifier": classifier,
            "iterations": iterations,
            "converged": bool(role_converged),
        }
    return {
        "oof_probabilities": oof,
        "val_probabilities": val_probabilities,
        "oof_metrics": {
            role: _classification_metrics(train_labels, probabilities)
            for role, probabilities in oof.items()
        },
        "val_metrics": {
            role: _classification_metrics(val_labels, probabilities)
            for role, probabilities in val_probabilities.items()
        },
        "fold_assignment": fold_assignment,
        "fold_rows": fold_rows,
        "fold_protocols": fold_protocols,
        "maximum_source_overlap": int(maximum_source_overlap),
        "all_readouts_converged": bool(converged),
        "final_state": final_state,
    }


def _focus(metrics: Mapping[str, object], focus_class_index: int) -> Mapping[str, object]:
    return metrics["per_class"][int(focus_class_index)]


def paired_feature_shift(
    control_features: np.ndarray,
    bioclip_features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    control = np.asarray(control_features, dtype=np.float64)
    candidate = np.asarray(bioclip_features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if control.shape != candidate.shape or control.ndim != 2:
        raise ValueError("Paired feature arrays must be aligned 2D arrays")
    cosine = np.sum(control * candidate, axis=1) / np.maximum(
        np.linalg.norm(control, axis=1) * np.linalg.norm(candidate, axis=1),
        1e-12,
    )
    displacement = np.linalg.norm(candidate - control, axis=1)
    return {
        "paired_cosine_mean": float(cosine.mean()),
        "paired_cosine_p10": float(np.quantile(cosine, 0.10)),
        "displacement_l2_mean": float(displacement.mean()),
        "per_class": [
            {
                "class_index": int(class_index),
                "samples": int(np.sum(labels == class_index)),
                "paired_cosine_mean": float(cosine[labels == class_index].mean()),
                "displacement_l2_mean": float(displacement[labels == class_index].mean()),
            }
            for class_index in sorted(np.unique(labels).tolist())
        ],
    }


def assess_bioclip_domain_readiness(
    *,
    train_rows: int,
    val_rows: int,
    train_val_source_overlap: int,
    fold_source_overlap: int,
    transform_max_abs_diff: float,
    equal_visual_parameter_count: bool,
    embedding_dimension: int,
    features_finite: bool,
    max_norm_error: float,
    minimum_effective_rank: float,
    all_readouts_converged: bool,
    folds_with_focus_gain: int,
    fold_count: int,
    control_oof: Mapping[str, object],
    bioclip_oof: Mapping[str, object],
    keeper_val: Mapping[str, object],
    control_val: Mapping[str, object],
    bioclip_val: Mapping[str, object],
    transitions_vs_keeper: Mapping[str, int],
    domain_oof_direction: Mapping[str, object],
    domain_val_direction: Mapping[str, object],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    control_oof_focus = _focus(control_oof, focus)
    bioclip_oof_focus = _focus(bioclip_oof, focus)
    keeper_focus = _focus(keeper_val, focus)
    control_val_focus = _focus(control_val, focus)
    bioclip_val_focus = _focus(bioclip_val, focus)
    oof_auc = domain_oof_direction.get("auc_fn_positive")
    val_auc = domain_val_direction.get("auc_fn_positive")
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_val_source_overlap": int(train_val_source_overlap),
        "fold_source_overlap": int(fold_source_overlap),
        "transform_max_abs_diff": float(transform_max_abs_diff),
        "equal_visual_parameter_count": bool(equal_visual_parameter_count),
        "embedding_dimension": int(embedding_dimension),
        "features_finite": bool(features_finite),
        "max_norm_error": float(max_norm_error),
        "minimum_effective_rank": float(minimum_effective_rank),
        "all_readouts_converged": bool(all_readouts_converged),
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "fold_count": int(fold_count),
        "oof_macro_gain": float(bioclip_oof["macro_f1"] - control_oof["macro_f1"]),
        "oof_focus_gain": float(bioclip_oof_focus["f1"] - control_oof_focus["f1"]),
        "val_macro_gain_vs_control": float(bioclip_val["macro_f1"] - control_val["macro_f1"]),
        "val_focus_gain_vs_control": float(bioclip_val_focus["f1"] - control_val_focus["f1"]),
        "val_macro_gain_vs_keeper": float(bioclip_val["macro_f1"] - keeper_val["macro_f1"]),
        "val_focus_gain_vs_keeper": float(bioclip_val_focus["f1"] - keeper_focus["f1"]),
        "keeper_macro_f1": float(keeper_val["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "control_val_macro_f1": float(control_val["macro_f1"]),
        "control_val_focus_f1": float(control_val_focus["f1"]),
        "bioclip_val_macro_f1": float(bioclip_val["macro_f1"]),
        "bioclip_val_focus_f1": float(bioclip_val_focus["f1"]),
        "bioclip_val_focus_precision": float(bioclip_val_focus["precision"]),
        "bioclip_val_focus_recall": float(bioclip_val_focus["recall"]),
        "transitions_vs_keeper": dict(transitions_vs_keeper),
        "oof_direction_auc": None if oof_auc is None else float(oof_auc),
        "val_direction_auc": None if val_auc is None else float(val_auc),
        "direction_auc_gap": (
            None if oof_auc is None or val_auc is None else abs(float(oof_auc) - float(val_auc))
        ),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "pixel_transform_bit_matched": observed["transform_max_abs_diff"] == 0.0,
        "visual_parameter_count_matched": bool(equal_visual_parameter_count),
        "embedding_dimension_512": observed["embedding_dimension"] == EMBEDDING_DIMENSION,
        "features_finite": bool(features_finite),
        "features_l2_normalized": observed["max_norm_error"] <= 1e-4,
        "effective_rank_ge_64": observed["minimum_effective_rank"] >= 64.0,
        "all_readouts_converged": bool(all_readouts_converged),
        "focus_gain_in_at_least_3_folds": observed["folds_with_focus_gain"]
        >= min(3, observed["fold_count"]),
        "oof_macro_gain_ge_0p005": observed["oof_macro_gain"] >= 0.005,
        "oof_focus_gain_ge_0p01": observed["oof_focus_gain"] >= 0.01,
        "val_macro_gain_vs_control_ge_0p005": observed["val_macro_gain_vs_control"] >= 0.005,
        "val_focus_gain_vs_control_ge_0p015": observed["val_focus_gain_vs_control"] >= 0.015,
        "val_macro_preserves_keeper_within_0p001": observed["val_macro_gain_vs_keeper"] >= -0.001,
        "val_focus_reaches_0p70": observed["bioclip_val_focus_f1"] >= FOCUS_MILESTONE,
        "val_focus_improves_keeper_by_0p01": observed["val_focus_gain_vs_keeper"] >= 0.01,
        "val_focus_recall_preserved_within_0p01": observed["bioclip_val_focus_recall"]
        >= observed["keeper_focus_recall"] - 0.01,
        "candidate_corrections_ge_harms_vs_keeper": int(transitions_vs_keeper["corrections"])
        >= int(transitions_vs_keeper["harms"]),
        "candidate_focus_fp_removed_ge_created_vs_keeper": int(
            transitions_vs_keeper["focus_false_positive_removed"]
        )
        >= int(transitions_vs_keeper["focus_false_positive_created"]),
        "candidate_focus_fn_rescued_ge_tp_broken_vs_keeper": int(
            transitions_vs_keeper["focus_false_negative_rescued"]
        )
        >= int(transitions_vs_keeper["focus_true_positive_broken"]),
        "oof_domain_direction_auc_ge_0p60": oof_auc is not None and float(oof_auc) >= 0.60,
        "val_domain_direction_auc_ge_0p60": val_auc is not None and float(val_auc) >= 0.60,
        "domain_direction_auc_gap_le_0p15": observed["direction_auc_gap"] is not None
        and float(observed["direction_auc_gap"]) <= 0.15,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "representation_transfer_smoke_permission": not failed,
        "image_smoke_permission": not failed,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def select_xai_cases(
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    *,
    focus_class_index: int = FOCUS_CLASS_INDEX,
    maximum_cases: int = XAI_CASES,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    keeper = np.asarray(keeper_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    focus = int(focus_class_index)
    categories = (
        (
            "focus_fn_rescued",
            (labels == focus) & (keeper != focus) & (candidate == focus),
        ),
        (
            "focus_tp_broken",
            (labels == focus) & (keeper == focus) & (candidate != focus),
        ),
        (
            "focus_fp_removed",
            (labels != focus) & (keeper == focus) & (candidate != focus),
        ),
        (
            "focus_fp_created",
            (labels != focus) & (keeper != focus) & (candidate == focus),
        ),
    )
    selected: List[Dict[str, object]] = []
    used: set[int] = set()
    quota = max(1, int(maximum_cases) // len(categories))
    for category, mask in categories:
        for index in np.flatnonzero(mask)[:quota].tolist():
            selected.append({"row_index": int(index), "category": category})
            used.add(int(index))
    changed = np.flatnonzero(keeper != candidate)
    for index in changed.tolist():
        if len(selected) >= int(maximum_cases):
            break
        if int(index) not in used:
            selected.append({"row_index": int(index), "category": "other_changed"})
            used.add(int(index))
    for index in range(len(labels)):
        if len(selected) >= int(maximum_cases):
            break
        if int(index) not in used:
            selected.append({"row_index": int(index), "category": "stable_reference"})
            used.add(int(index))
    return selected[: int(maximum_cases)]


def _focus_logits(
    features: np.ndarray,
    state: Mapping[str, object],
    *,
    focus_class_index: int,
) -> np.ndarray:
    scaled = _apply_scaler(
        np.asarray(features, dtype=np.float32),
        np.asarray(state["mean"], dtype=np.float64),
        np.asarray(state["scale"], dtype=np.float64),
    )
    logits = np.asarray(state["classifier"].decision_function(scaled), dtype=np.float64)
    return logits[:, int(focus_class_index)]


def _occlusion_maps(
    *,
    model: torch.nn.Module,
    transformed_images: torch.Tensor,
    readout_state: Mapping[str, object],
    device: torch.device,
    amp: bool,
    grid: int,
    batch_size: int,
    focus_class_index: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    image_count, _, height, width = transformed_images.shape
    grid = int(grid)
    variants: List[torch.Tensor] = []
    for image in transformed_images:
        variants.append(image.clone())
        for row in range(grid):
            y1 = int(round(row * height / grid))
            y2 = int(round((row + 1) * height / grid))
            for column in range(grid):
                x1 = int(round(column * width / grid))
                x2 = int(round((column + 1) * width / grid))
                occluded = image.clone()
                occluded[:, y1:y2, x1:x2] = 0.0
                variants.append(occluded)
    variant_tensor = torch.stack(variants, dim=0)
    feature_batches: List[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, int(variant_tensor.size(0)), max(1, int(batch_size))):
            batch = variant_tensor[start : start + int(batch_size)].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with autocast_context(device, bool(amp)):
                features = model.encode_image(batch, normalize=True)
            feature_batches.append(features.float().cpu().numpy())
    all_features = np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)
    scores = _focus_logits(all_features, readout_state, focus_class_index=focus_class_index)
    maps = np.zeros((image_count, grid, grid), dtype=np.float32)
    stride = 1 + grid * grid
    for image_index in range(image_count):
        offset = image_index * stride
        maps[image_index] = (
            scores[offset] - scores[offset + 1 : offset + stride]
        ).reshape(grid, grid)
    positive = np.maximum(maps, 0.0)
    center_start = max(0, grid // 2 - 1)
    center_end = min(grid, center_start + 2)
    total = positive.sum(axis=(1, 2))
    center = positive[:, center_start:center_end, center_start:center_end].sum(axis=(1, 2))
    center_mass = np.divide(center, np.maximum(total, 1e-12))
    return maps, {
        "cases": int(image_count),
        "grid": grid,
        "positive_center_mass_mean": float(center_mass.mean()) if image_count else 0.0,
        "positive_center_mass_per_case": center_mass.tolist(),
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        ),
    }


def _signed_overlay(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    values = np.asarray(heat, dtype=np.float32)
    maximum = max(float(np.max(np.abs(values))), 1e-8)
    normalized = np.clip(values / maximum, -1.0, 1.0)
    positive = np.maximum(normalized, 0.0)
    negative = np.maximum(-normalized, 0.0)
    color = np.zeros((*normalized.shape, 3), dtype=np.float32)
    color[..., 0] = 255.0 * positive
    color[..., 2] = 255.0 * negative
    color[..., 1] = 70.0 * (positive + negative)
    resampling = getattr(Image, "Resampling", Image)
    color_large = np.asarray(
        Image.fromarray(color.astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    strength = np.asarray(
        Image.fromarray(((np.abs(normalized)) * 255.0).astype(np.uint8)).resize(
            (int(image.shape[1]), int(image.shape[0])),
            resampling.BILINEAR,
        ),
        dtype=np.float32,
    )[..., None] / 255.0
    alpha = 0.55 * strength
    return np.clip(image.astype(np.float32) * (1.0 - alpha) + color_large * alpha, 0, 255).astype(
        np.uint8
    )


def _write_xai_contact_sheet(
    output_dir: Path,
    *,
    transformed_images: torch.Tensor,
    cases: Sequence[Mapping[str, object]],
    control_maps: np.ndarray,
    bioclip_maps: np.ndarray,
    labels: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    bioclip_probabilities: np.ndarray,
    classification_paths: np.ndarray,
    sample_indices: np.ndarray,
    control_stats: Mapping[str, object],
    bioclip_stats: Mapping[str, object],
) -> Dict[str, object]:
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
    rgb = (
        (transformed_images.float() * std + mean)
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    tile = 168
    label_height = 34
    canvas = Image.new(
        "RGB",
        (tile * 3, len(cases) * (tile + label_height)),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image)
    rows: List[Dict[str, object]] = []
    for case_index, case in enumerate(cases):
        row_index = int(case["row_index"])
        keeper_prediction = int(keeper_probabilities[row_index].argmax())
        control_prediction = int(control_probabilities[row_index].argmax())
        bioclip_prediction = int(bioclip_probabilities[row_index].argmax())
        images = (
            rgb[case_index],
            _signed_overlay(rgb[case_index], control_maps[case_index]),
            _signed_overlay(rgb[case_index], bioclip_maps[case_index]),
        )
        y = case_index * (tile + label_height)
        for column, image in enumerate(images):
            canvas.paste(
                Image.fromarray(image).resize((tile, tile), resampling.BILINEAR),
                (column * tile, y),
            )
        draw.text(
            (4, y + tile + 2),
            (
                f"{case['category']} y={int(labels[row_index])} "
                f"k/c/b={keeper_prediction}/{control_prediction}/{bioclip_prediction}"
            ),
            fill=(20, 20, 20),
        )
        rows.append(
            {
                "preview_row": int(case_index),
                "row_index": row_index,
                "sample_index": int(sample_indices[row_index]),
                "category": str(case["category"]),
                "classification_path": str(classification_paths[row_index]),
                "target_index": int(labels[row_index]),
                "keeper_prediction_index": keeper_prediction,
                "control_prediction_index": control_prediction,
                "bioclip_prediction_index": bioclip_prediction,
                "keeper_focus_probability": float(keeper_probabilities[row_index, FOCUS_CLASS_INDEX]),
                "control_focus_probability": float(control_probabilities[row_index, FOCUS_CLASS_INDEX]),
                "bioclip_focus_probability": float(bioclip_probabilities[row_index, FOCUS_CLASS_INDEX]),
                "control_occlusion_map": control_maps[case_index].tolist(),
                "bioclip_occlusion_map": bioclip_maps[case_index].tolist(),
            }
        )
    canvas.save(output_dir / "xai_mean_occlusion_class1.png")
    payload = {
        "method": "fixed_4x4_mean_occlusion_on_class1_readout_logit",
        "columns": ["official_clip_input", "openai_clip_control", "bioclip"],
        "red": "occlusion lowers class1 logit; positive class1 evidence",
        "blue": "occlusion raises class1 logit; suppressive evidence",
        "selection_uses_labels_for_audit_only": True,
        "control_stats": dict(control_stats),
        "bioclip_stats": dict(bioclip_stats),
        "rows": rows,
    }
    (output_dir / "xai_mean_occlusion_class1.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def _prediction_rows(
    *,
    split: str,
    payload: Mapping[str, object],
    fold_assignment: np.ndarray,
    keeper_probabilities: np.ndarray,
    control_probabilities: np.ndarray,
    bioclip_probabilities: np.ndarray,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    labels = np.asarray(payload["labels"], dtype=np.int64)
    for row_index in range(len(labels)):
        row: Dict[str, object] = {
            "split": split,
            "sample_index": int(payload["sample_index"][row_index]),
            "fold": int(fold_assignment[row_index]),
            "source_stem": str(payload["source_stems"][row_index]),
            "image_path": str(payload["paths"][row_index]),
            "target_index": int(labels[row_index]),
            "keeper_prediction_index": int(keeper_probabilities[row_index].argmax()),
            "control_prediction_index": int(control_probabilities[row_index].argmax()),
            "bioclip_prediction_index": int(bioclip_probabilities[row_index].argmax()),
        }
        for class_index in range(int(keeper_probabilities.shape[1])):
            row[f"keeper_prob_{class_index}"] = float(keeper_probabilities[row_index, class_index])
            row[f"control_prob_{class_index}"] = float(control_probabilities[row_index, class_index])
            row[f"bioclip_prob_{class_index}"] = float(bioclip_probabilities[row_index, class_index])
        output.append(row)
    return output


def _readout_state_metadata(state: Mapping[str, object]) -> Dict[str, object]:
    classifier = state["classifier"]
    return {
        "iterations": int(state["iterations"]),
        "converged": bool(state["converged"]),
        "coefficient_l2": float(np.linalg.norm(np.asarray(classifier.coef_, dtype=np.float64))),
        "intercept_l2": float(np.linalg.norm(np.asarray(classifier.intercept_, dtype=np.float64))),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_protocol(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    device = _resolve_device(str(args.device))
    plans, aligned_classification_paths = _build_alignment(args)
    class_names = list(plans["train"]["class_names"])
    class_count = len(class_names)
    focus_class_index = int(args.focus_class_index)
    if class_count != 5 or not 0 <= focus_class_index < class_count:
        raise ValueError("This locked audit requires five classes and a valid focus index")
    train_groups = np.asarray(
        [_normalized_source_group(value) for value in plans["train"]["source_stems"]],
        dtype=object,
    )
    val_groups = np.asarray(
        [_normalized_source_group(value) for value in plans["val"]["source_stems"]],
        dtype=object,
    )
    train_val_source_overlap = len(
        set(train_groups.tolist()).intersection(set(val_groups.tolist()))
    )
    if train_val_source_overlap:
        raise ValueError(
            f"Train/validation source groups overlap: {train_val_source_overlap}"
        )
    train_cache = _load_cache(Path(args.keeper_train_cache), split="train")
    val_cache = _load_cache(Path(args.keeper_val_cache), split="val")
    for split, cache in (("train", train_cache), ("val", val_cache)):
        plan = plans[split]
        if not np.array_equal(np.asarray(cache["labels"], dtype=np.int64), np.asarray(plan["labels"], dtype=np.int64)):
            raise ValueError(f"Keeper cache labels differ from strict {split} alignment")
        if not np.array_equal(np.asarray(cache["sample_index"], dtype=np.int64), np.asarray(plan["sample_index"], dtype=np.int64)):
            raise ValueError(f"Keeper cache sample_index differs from strict {split} alignment")
        cache_paths = np.asarray(
            [_normalized_path(path) for path in cache["paths"]],
            dtype=object,
        )
        plan_paths = np.asarray(
            [_normalized_path(path) for path in plan["paths"]],
            dtype=object,
        )
        if not np.array_equal(cache_paths, plan_paths):
            raise ValueError(f"Keeper cache paths differ from strict {split} alignment")
        if not np.array_equal(
            np.asarray(
                [_normalized_source_group(value) for value in cache["source_stems"]],
                dtype=object,
            ),
            np.asarray(
                [_normalized_source_group(value) for value in plan["source_stems"]],
                dtype=object,
            ),
        ):
            raise ValueError(f"Keeper cache source groups differ from strict {split} alignment")

    probe_path = Path(str(aligned_classification_paths["val"][0]))
    role_payloads: Dict[str, Dict[str, Dict[str, object]]] = {}
    model_metadata: Dict[str, Dict[str, object]] = {}
    probe_tensors: Dict[str, torch.Tensor] = {}
    for role in ("control", "bioclip"):
        print(f"loading locked {role} encoder", flush=True)
        model, transform, metadata = _build_model(
            role,
            device=device,
            model_cache=Path(args.model_cache),
        )
        probe_tensors[role] = transform(Image.open(probe_path).convert("RGB"))
        extracted = {
            split: _extract_split(
                model=model,
                transform=transform,
                classification_root=Path(args.classification_root),
                split=split,
                device=device,
                batch_size=int(args.batch_size),
                workers=int(args.workers),
                amp=bool(args.amp),
            )
            for split in ("train", "val")
        }
        role_payloads[role] = {
            split: _align_payload(extracted[split], plans[split])
            for split in ("train", "val")
        }
        model_metadata[role] = metadata
        model.to("cpu")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    transform_max_abs_diff = float(
        (probe_tensors["control"] - probe_tensors["bioclip"]).abs().max().item()
    )
    train_labels = np.asarray(plans["train"]["labels"], dtype=np.int64)
    val_labels = np.asarray(plans["val"]["labels"], dtype=np.int64)
    readouts = fit_matched_source_readouts(
        control_train=np.asarray(role_payloads["control"]["train"]["features"]),
        bioclip_train=np.asarray(role_payloads["bioclip"]["train"]["features"]),
        train_labels=train_labels,
        train_groups=train_groups,
        control_val=np.asarray(role_payloads["control"]["val"]["features"]),
        bioclip_val=np.asarray(role_payloads["bioclip"]["val"]["features"]),
        val_labels=val_labels,
        folds=int(args.folds),
        logistic_c=float(args.logistic_c),
        max_iterations=int(args.max_iterations),
        workers=max(1, int(args.workers)),
        seed=int(args.seed),
        focus_class_index=focus_class_index,
    )
    keeper_train_probabilities = np.asarray(train_cache["probabilities"], dtype=np.float32)
    keeper_val_probabilities = np.asarray(val_cache["probabilities"], dtype=np.float32)
    keeper_val_metrics = _classification_metrics(val_labels, keeper_val_probabilities)
    transitions = {
        "oof_bioclip_vs_control": _transition_stats(
            train_labels,
            readouts["oof_probabilities"]["control"],
            readouts["oof_probabilities"]["bioclip"],
            focus_class_index=focus_class_index,
        ),
        "val_bioclip_vs_control": _transition_stats(
            val_labels,
            readouts["val_probabilities"]["control"],
            readouts["val_probabilities"]["bioclip"],
            focus_class_index=focus_class_index,
        ),
        "val_bioclip_vs_keeper": _transition_stats(
            val_labels,
            keeper_val_probabilities,
            readouts["val_probabilities"]["bioclip"],
            focus_class_index=focus_class_index,
        ),
    }
    direction = {
        "oof_bioclip_vs_control": _direction_auc(
            train_labels,
            readouts["oof_probabilities"]["control"],
            readouts["oof_probabilities"]["bioclip"],
            focus_class_index=focus_class_index,
        ),
        "val_bioclip_vs_control": _direction_auc(
            val_labels,
            readouts["val_probabilities"]["control"],
            readouts["val_probabilities"]["bioclip"],
            focus_class_index=focus_class_index,
        ),
    }
    effective_ranks = {
        role: {
            split: float(
                _projected_effective_rank(
                    np.asarray(role_payloads[role][split]["features"], dtype=np.float32),
                    seed=int(args.seed) + (0 if role == "control" else 1000) + (0 if split == "train" else 100),
                )
            )
            for split in ("train", "val")
        }
        for role in ("control", "bioclip")
    }
    feature_shift = {
        split: paired_feature_shift(
            role_payloads["control"][split]["features"],
            role_payloads["bioclip"][split]["features"],
            plans[split]["labels"],
        )
        for split in ("train", "val")
    }
    folds_with_focus_gain = sum(float(row["focus_gain"]) > 0.0 for row in readouts["fold_rows"])
    all_norm_errors = [
        float(role_payloads[role][split]["norm_max_abs_error"])
        for role in ("control", "bioclip")
        for split in ("train", "val")
    ]
    gate = assess_bioclip_domain_readiness(
        train_rows=len(train_labels),
        val_rows=len(val_labels),
        train_val_source_overlap=train_val_source_overlap,
        fold_source_overlap=int(readouts["maximum_source_overlap"]),
        transform_max_abs_diff=transform_max_abs_diff,
        equal_visual_parameter_count=(
            int(model_metadata["control"]["visual_parameter_count"])
            == int(model_metadata["bioclip"]["visual_parameter_count"])
        ),
        embedding_dimension=int(model_metadata["bioclip"]["embedding_dimension"]),
        features_finite=all(
            bool(role_payloads[role][split]["finite"])
            for role in ("control", "bioclip")
            for split in ("train", "val")
        ),
        max_norm_error=max(all_norm_errors),
        minimum_effective_rank=min(
            effective_ranks[role][split]
            for role in ("control", "bioclip")
            for split in ("train", "val")
        ),
        all_readouts_converged=bool(readouts["all_readouts_converged"]),
        folds_with_focus_gain=folds_with_focus_gain,
        fold_count=int(args.folds),
        control_oof=readouts["oof_metrics"]["control"],
        bioclip_oof=readouts["oof_metrics"]["bioclip"],
        keeper_val=keeper_val_metrics,
        control_val=readouts["val_metrics"]["control"],
        bioclip_val=readouts["val_metrics"]["bioclip"],
        transitions_vs_keeper=transitions["val_bioclip_vs_keeper"],
        domain_oof_direction=direction["oof_bioclip_vs_control"],
        domain_val_direction=direction["val_bioclip_vs_control"],
        focus_class_index=focus_class_index,
        test_split_used=False,
    )

    cases = select_xai_cases(
        val_labels,
        keeper_val_probabilities,
        readouts["val_probabilities"]["bioclip"],
        focus_class_index=focus_class_index,
        maximum_cases=int(args.xai_cases),
    )
    locked_transform = open_clip.image_transform(
        224,
        is_train=False,
        mean=CLIP_MEAN,
        std=CLIP_STD,
        interpolation="bicubic",
        resize_mode="shortest",
    )
    transformed_images = torch.stack(
        [
            probe_tensors["control"]
            if Path(str(aligned_classification_paths["val"][int(case["row_index"])])) == probe_path
            else locked_transform(
                Image.open(
                    Path(str(aligned_classification_paths["val"][int(case["row_index"])]))
                ).convert("RGB")
            )
            for case in cases
        ],
        dim=0,
    )
    xai_maps: Dict[str, np.ndarray] = {}
    xai_stats: Dict[str, Dict[str, object]] = {}
    for role in ("control", "bioclip"):
        model, transform, _metadata = _build_model(
            role,
            device=device,
            model_cache=Path(args.model_cache),
        )
        transformed_role = torch.stack(
            [
                transform(
                    Image.open(
                        Path(str(aligned_classification_paths["val"][int(case["row_index"])]))
                    ).convert("RGB")
                )
                for case in cases
            ],
            dim=0,
        )
        if float((transformed_role - transformed_images).abs().max().item()) != 0.0:
            raise RuntimeError("XAI transforms differ between matched encoders")
        maps, stats = _occlusion_maps(
            model=model,
            transformed_images=transformed_role,
            readout_state=readouts["final_state"][role],
            device=device,
            amp=bool(args.amp),
            grid=int(args.occlusion_grid),
            batch_size=int(args.batch_size),
            focus_class_index=focus_class_index,
        )
        xai_maps[role] = maps
        xai_stats[role] = stats
        model.to("cpu")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    xai = _write_xai_contact_sheet(
        output_dir,
        transformed_images=transformed_images,
        cases=cases,
        control_maps=xai_maps["control"],
        bioclip_maps=xai_maps["bioclip"],
        labels=val_labels,
        keeper_probabilities=keeper_val_probabilities,
        control_probabilities=readouts["val_probabilities"]["control"],
        bioclip_probabilities=readouts["val_probabilities"]["bioclip"],
        classification_paths=aligned_classification_paths["val"],
        sample_indices=np.asarray(plans["val"]["sample_index"], dtype=np.int64),
        control_stats=xai_stats["control"],
        bioclip_stats=xai_stats["bioclip"],
    )

    _write_csv(output_dir / "fold_metrics.csv", readouts["fold_rows"])
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            payload=role_payloads["bioclip"]["train"],
            fold_assignment=np.asarray(readouts["fold_assignment"], dtype=np.int64),
            keeper_probabilities=keeper_train_probabilities,
            control_probabilities=readouts["oof_probabilities"]["control"],
            bioclip_probabilities=readouts["oof_probabilities"]["bioclip"],
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            payload=role_payloads["bioclip"]["val"],
            fold_assignment=np.full(len(val_labels), -1, dtype=np.int64),
            keeper_probabilities=keeper_val_probabilities,
            control_probabilities=readouts["val_probabilities"]["control"],
            bioclip_probabilities=readouts["val_probabilities"]["bioclip"],
        ),
    )
    if bool(args.save_embeddings):
        np.savez_compressed(
            output_dir / "matched_clip_embeddings_train_val.npz",
            train_control=role_payloads["control"]["train"]["features"],
            train_bioclip=role_payloads["bioclip"]["train"]["features"],
            train_labels=train_labels,
            train_sample_index=plans["train"]["sample_index"],
            train_source_stems=plans["train"]["source_stems"],
            val_control=role_payloads["control"]["val"]["features"],
            val_bioclip=role_payloads["bioclip"]["val"]["features"],
            val_labels=val_labels,
            val_sample_index=plans["val"]["sample_index"],
            val_source_stems=plans["val"]["source_stems"],
        )
    protocol = {
        "method": "bioclip_domain_feature_readiness_against_matched_openai_clip",
        "scope": "Frozen external representation diagnostic only; not a final pretrained TRKH model.",
        "literature": list(LITERATURE),
        "classification_root": str(Path(args.classification_root).resolve()),
        "yolo_data": str(Path(args.yolo_data).resolve()),
        "keeper_train_cache": str(Path(args.keeper_train_cache).resolve()),
        "keeper_val_cache": str(Path(args.keeper_val_cache).resolve()),
        "split_usage": {"train": True, "val": True, "test": False},
        "train_rows": int(len(train_labels)),
        "val_rows": int(len(val_labels)),
        "train_source_groups": int(np.unique(train_groups).size),
        "val_source_groups": int(np.unique(val_groups).size),
        "train_val_source_overlap": int(train_val_source_overlap),
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "amp": bool(args.amp),
            "torch_threads": int(args.torch_threads),
        },
        "save_embeddings": bool(args.save_embeddings),
        "transform_max_abs_diff": float(transform_max_abs_diff),
        "matched_pixels": bool(transform_max_abs_diff == 0.0),
        "matched_visual_parameter_count": bool(
            int(model_metadata["control"]["visual_parameter_count"])
            == int(model_metadata["bioclip"]["visual_parameter_count"])
        ),
        "official_activation_configs": {
            "control": str(model_metadata["control"]["activation"]),
            "bioclip": str(model_metadata["bioclip"]["activation"]),
            "note": "Official OpenAI CLIP uses QuickGELU while the released BioCLIP config uses GELU; capacity and pixels are matched, but the training-domain delta is not claimed to be activation-isolated.",
        },
        "natural_frequency_readout": True,
        "class_weight": None,
        "folds": int(args.folds),
        "logistic_c": float(args.logistic_c),
        "max_iterations": int(args.max_iterations),
        "seed": int(args.seed),
        "prompt_or_zero_shot_selection": False,
        "validation_hyperparameter_selection": False,
        "underlying_keeper_train_cache_is_not_oof": True,
        "xai_selection_is_audit_only": True,
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
        "trainable_manifest_written": False,
    }
    summary = {
        "protocol": protocol,
        "models": model_metadata,
        "alignment": {
            split: _compact_alignment(plans[split]) for split in ("train", "val")
        },
        "transform_max_abs_diff": transform_max_abs_diff,
        "extraction": {
            role: {
                split: {
                    "seconds": float(role_payloads[role][split]["seconds"]),
                    "peak_cuda_memory_mib": float(role_payloads[role][split]["peak_cuda_memory_mib"]),
                    "norm_max_abs_error": float(role_payloads[role][split]["norm_max_abs_error"]),
                    "finite": bool(role_payloads[role][split]["finite"]),
                    "effective_rank": float(effective_ranks[role][split]),
                }
                for split in ("train", "val")
            }
            for role in ("control", "bioclip")
        },
        "feature_shift": feature_shift,
        "readout": {
            "fold_protocols": readouts["fold_protocols"],
            "final": {
                role: _readout_state_metadata(readouts["final_state"][role])
                for role in ("control", "bioclip")
            },
            "all_converged": bool(readouts["all_readouts_converged"]),
        },
        "metrics": {
            "train_oof": readouts["oof_metrics"],
            "val": {
                "keeper": keeper_val_metrics,
                **readouts["val_metrics"],
            },
        },
        "transitions": transitions,
        "domain_direction": direction,
        "xai": {
            "case_count": len(cases),
            "control_stats": xai_stats["control"],
            "bioclip_stats": xai_stats["bioclip"],
            "preview": "xai_mean_occlusion_class1.png",
            "manifest": "xai_mean_occlusion_class1.json",
        },
        "gate": gate,
        "seconds": float(time.perf_counter() - start),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    observed = gate["observed"]
    readme = [
        "# BioCLIP Domain-Feature Readiness",
        "",
        f"- Train/validation rows: `{len(train_labels)}/{len(val_labels)}`; test is closed.",
        f"- OpenAI CLIP OOF macro/class1: `{float(readouts['oof_metrics']['control']['macro_f1']):.6f}/{float(_focus(readouts['oof_metrics']['control'], focus_class_index)['f1']):.6f}`.",
        f"- BioCLIP OOF macro/class1: `{float(readouts['oof_metrics']['bioclip']['macro_f1']):.6f}/{float(_focus(readouts['oof_metrics']['bioclip'], focus_class_index)['f1']):.6f}`.",
        f"- OpenAI CLIP validation macro/class1: `{float(readouts['val_metrics']['control']['macro_f1']):.6f}/{float(_focus(readouts['val_metrics']['control'], focus_class_index)['f1']):.6f}`.",
        f"- BioCLIP validation macro/class1: `{float(readouts['val_metrics']['bioclip']['macro_f1']):.6f}/{float(_focus(readouts['val_metrics']['bioclip'], focus_class_index)['f1']):.6f}`.",
        f"- Keeper validation macro/class1: `{float(keeper_val_metrics['macro_f1']):.6f}/{float(_focus(keeper_val_metrics, focus_class_index)['f1']):.6f}`.",
        f"- Domain FN-vs-FP direction AUROC OOF/val: `{observed['oof_direction_auc']}/{observed['val_direction_auc']}`.",
        f"- Representation-transfer smoke permission: `{str(bool(gate['representation_transfer_smoke_permission'])).lower()}`.",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`.",
        "- The keeper train cache is in-sample, not true current-keeper OOF; this audit does not provide a keeper-relative routing target.",
        "",
        "Decision: open one fixed domain-feature transfer smoke only when every locked check passes. Do not sweep prompts, readout C, folds, residual weights, routers, or BioCLIP variants on a rejected result.",
        "",
        "This diagnostic reads only immutable train/validation object crops and frozen keeper caches. It writes no model, checkpoint, test result, or trainable manifest.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(
        output_dir,
        mode="bioclip_domain_feature_readiness_evidence_manifest",
    )
    return {
        "gate": gate,
        "metrics": summary["metrics"],
        "artifact_manifest": {key: value for key, value in manifest.items() if key != "files"},
        "seconds": summary["seconds"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "artifact_manifest": result["artifact_manifest"],
                "seconds": result["seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
