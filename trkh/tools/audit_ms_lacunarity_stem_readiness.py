from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_deepten_stem_texture_readiness import (
    _normalized_source_group,
    build_interior_grid_mask,
)
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import _write_artifact_manifest
from trkh.tools.probe_embedding_prototypes import _build_dataset, _collate_classification


SEED = 20260720
FOCUS_CLASS_INDEX = 1
FOLDS = 5
READOUT_C = 0.3
MAX_ITERATIONS = 2000
INTERIOR_ERODE_RATIO = 0.12
EXPECTED_FIT_ROWS = 7372
EXPECTED_HOLDOUT_ROWS = 1843
EXPECTED_CONTROL_DIM = 512
EXPECTED_CANDIDATE_DIM = 1536
EXPECTED_CANDIDATE_ONLY_DIM = 1024
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)

PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/"
    "Mohan_Lacunarity_Pooling_Layers_for_Plant_Image_Classification_using_"
    "Texture_Analysis_CVPRW_2024_paper.html"
)
OFFICIAL_REPOSITORY = (
    "https://github.com/Advanced-Vision-and-Learning-Lab/"
    "2024_V4A_Lacunarity_Pooling_Layer"
)
OFFICIAL_COMMIT = "6e464b4c326513c206eb9377c7ff9139ddee65f7"
OFFICIAL_TREE = "9f1a7c618db236cd8673e2cb797308fb210e39fe"

EXPECTED_HASHES = {
    "protocol": "a6151debe79add4148ea31e5d67cba5e8ebff08cc3abfdea01580320b137cb9e",
    "paper": "a110acaf0c4b43910b0e861f657f96039dc5ed484a422d1f2adb4c7097655ce2",
    "raw_data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "fold_yaml": "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e",
    "fold_summary": "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763",
    "fold_declaration": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "current_best_commands": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-stem MS-lacunarity readiness audit on the declared TRKH "
            "train-only fold. It never opens test or writes a model checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--extract-batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--readout-c", type=float, default=READOUT_C)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--interior-erode-ratio", type=float, default=INTERIOR_ERODE_RATIO)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    device = torch.device(requested) if requested else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.folds) != FOLDS:
        raise ValueError(f"The locked protocol requires folds={FOLDS}")
    if abs(float(args.readout_c) - READOUT_C) > 1e-12:
        raise ValueError(f"The locked protocol requires readout C={READOUT_C}")
    if int(args.max_iterations) != MAX_ITERATIONS:
        raise ValueError(
            f"The locked protocol requires max_iterations={MAX_ITERATIONS}"
        )
    if abs(float(args.interior_erode_ratio) - INTERIOR_ERODE_RATIO) > 1e-12:
        raise ValueError(
            "The locked protocol requires interior_erode_ratio="
            f"{INTERIOR_ERODE_RATIO}"
        )
    if int(args.focus_class_index) != FOCUS_CLASS_INDEX:
        raise ValueError("The locked protocol supports class 1 only")
    if int(args.extract_batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("Extraction batch size/workers are invalid")
    if int(args.max_train_samples) < 0 or int(args.max_val_samples) < 0:
        raise ValueError("Sample limits cannot be negative")
    if int(args.max_train_samples) > 512 or int(args.max_val_samples) > 256:
        raise ValueError("Prefix limits cannot exceed the locked 512/256 cap")
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data).resolve().parent
    if _is_relative_to(output_dir, data_root):
        raise ValueError("Output directory must stay outside the data view")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")


def _provenance_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = _repo_root()
    data_yaml = Path(args.data).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    return {
        "protocol": root
        / "docs"
        / "TRKH_5CLASS_MS_LACUNARITY_STEM_READINESS_PROTOCOL_20260720.md",
        "paper": Path(
            "D:/DataAI/external_sources/papers/lacunarity_pooling_cvprw2024.pdf"
        ),
        "raw_data": Path("D:/DataAI/AIEx/newdataset/yolo_f/data.yaml"),
        "fold_yaml": data_yaml,
        "fold_summary": data_yaml.parent / "summary.json",
        "fold_declaration": root
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
        "keeper": checkpoint,
        "current_best_commands": root
        / "docs"
        / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "command_history": root
        / "docs"
        / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
    }


def verify_provenance(args: argparse.Namespace) -> Dict[str, object]:
    rows: Dict[str, object] = {}
    for name, path in _provenance_paths(args).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked provenance file: {path}")
        observed = _sha256(path)
        expected = EXPECTED_HASHES[name]
        if observed != expected:
            raise ValueError(
                f"Locked provenance mismatch for {name}: {observed} != {expected}"
            )
        rows[name] = {"path": str(path), "sha256": observed, "matched": True}
    return rows


def gaussian_pyrdown(values: Tensor) -> Tensor:
    """One fixed Gaussian-pyramid level matching the official 5x5 binomial form."""

    if values.ndim != 4 or min(int(values.shape[-2]), int(values.shape[-1])) < 3:
        raise ValueError("Pyrdown input must be [B,C,H,W] with H,W >= 3")
    one_d = values.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel = torch.outer(one_d, one_d).view(1, 1, 5, 5)
    kernel = kernel.expand(int(values.size(1)), 1, 5, 5)
    padded = F.pad(values, (2, 2, 2, 2), mode="reflect")
    blurred = F.conv2d(padded, kernel, groups=int(values.size(1)))
    return blurred[:, :, ::2, ::2]


def weighted_mean_lacunarity(
    values: Tensor,
    mask: Tensor,
    *,
    epsilon: float = 1e-5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return weighted mean, official normalized second moment, and support."""

    if values.ndim != 4 or mask.ndim != 4 or int(mask.size(1)) != 1:
        raise ValueError("values/mask must be [B,C,H,W]/[B,1,H,W]")
    if tuple(values.shape[0:1] + values.shape[2:4]) != tuple(
        mask.shape[0:1] + mask.shape[2:4]
    ):
        raise ValueError("values and mask spatial shapes do not align")
    weights = mask.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    support = weights.sum(dim=(2, 3)).clamp_min(1.0)
    mean = (values * weights).sum(dim=(2, 3)) / support
    second = (values.square() * weights).sum(dim=(2, 3)) / support
    lacunarity = second / (mean.square() + float(epsilon)) - 1.0
    return mean, lacunarity, support


def extract_ms_lacunarity_descriptors(
    stem: Tensor,
    crop_bbox: Tensor,
    image_valid_mask: Optional[Tensor],
    *,
    erode_ratio: float = INTERIOR_ERODE_RATIO,
) -> Dict[str, Tensor]:
    if stem.ndim != 4 or int(stem.size(1)) != 256:
        raise ValueError("Locked TRKH stem activation must have shape [B,256,H,W]")
    if crop_bbox.ndim != 2 or int(crop_bbox.size(0)) != int(stem.size(0)):
        raise ValueError("crop_bbox must align with the stem batch")

    height, width = int(stem.size(2)), int(stem.size(3))
    if height != width:
        raise ValueError("Locked stem map must be square")
    scaled = ((torch.tanh(stem.float()) + 1.0) * 0.5) * 255.0

    if image_valid_mask is None:
        full_mask = torch.ones(
            int(stem.size(0)), 1, height, width, device=stem.device, dtype=torch.float32
        )
    else:
        full_mask = image_valid_mask
        if full_mask.ndim == 3:
            full_mask = full_mask.unsqueeze(1)
        if full_mask.ndim != 4:
            raise ValueError("image_valid_mask must be [B,H,W] or [B,1,H,W]")
        full_mask = F.interpolate(
            full_mask.to(device=stem.device, dtype=torch.float32),
            size=(height, width),
            mode="area",
        )

    object_mask = build_interior_grid_mask(
        crop_bbox.to(device=stem.device, dtype=torch.float32),
        image_valid_mask.to(device=stem.device) if image_valid_mask is not None else None,
        grid_size=height,
        erode_ratio=float(erode_ratio),
    ).view(int(stem.size(0)), 1, height, width)
    object_mask = object_mask.to(dtype=torch.float32)

    down = gaussian_pyrdown(scaled)
    down_size = tuple(int(value) for value in down.shape[-2:])
    full_down_mask = F.interpolate(full_mask, size=down_size, mode="area")
    object_down_mask = F.interpolate(object_mask, size=down_size, mode="area")

    full_mean, full_l0, full_support = weighted_mean_lacunarity(scaled, full_mask)
    object_mean, object_l0, object_support = weighted_mean_lacunarity(
        scaled, object_mask
    )
    _, full_l1, full_down_support = weighted_mean_lacunarity(down, full_down_mask)
    _, object_l1, object_down_support = weighted_mean_lacunarity(
        down, object_down_mask
    )

    control = torch.cat((full_mean, object_mean), dim=1)
    candidate_only = torch.cat(
        (
            full_mean * full_l0,
            full_mean * full_l1,
            object_mean * object_l0,
            object_mean * object_l1,
        ),
        dim=1,
    )
    candidate = torch.cat((control, candidate_only), dim=1)
    if int(control.size(1)) != EXPECTED_CONTROL_DIM:
        raise RuntimeError("Unexpected control descriptor dimension")
    if int(candidate_only.size(1)) != EXPECTED_CANDIDATE_ONLY_DIM:
        raise RuntimeError("Unexpected candidate-only descriptor dimension")
    if int(candidate.size(1)) != EXPECTED_CANDIDATE_DIM:
        raise RuntimeError("Unexpected candidate descriptor dimension")
    return {
        "control": control,
        "candidate_only": candidate_only,
        "candidate": candidate,
        "full_support": full_support,
        "object_support": object_support,
        "full_down_support": full_down_support,
        "object_down_support": object_down_support,
    }


def _extract_split(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    amp: bool,
    erode_ratio: float,
    split: str,
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_classification,
        persistent_workers=bool(int(workers) > 0),
    )
    descriptor_batches: Dict[str, List[np.ndarray]] = {
        "control": [],
        "candidate_only": [],
        "candidate": [],
    }
    support_batches: Dict[str, List[np.ndarray]] = {
        "full_support": [],
        "object_support": [],
        "full_down_support": [],
        "object_down_support": [],
    }
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    sample_index_batches: List[np.ndarray] = []
    paths: List[str] = []
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = (
        [str(path) for path in dataset_paths_fn()]
        if callable(dataset_paths_fn)
        else []
    )
    captured: List[Tensor] = []

    def capture_stem(_module, _inputs, output) -> None:
        captured.append(output)

    hook = model.stem.register_forward_hook(capture_stem)
    start = time.perf_counter()
    seen = 0
    stem_shape: Optional[Tuple[int, int, int]] = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc=f"ms-lacunarity-{split}", dynamic_ncols=True)
            for images, labels, metadata in iterator:
                if not isinstance(metadata, Mapping):
                    raise ValueError("MS-lacunarity extraction requires metadata")
                images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                crop_bbox = metadata.get("crop_bbox")
                if not torch.is_tensor(crop_bbox):
                    raise ValueError("crop_bbox metadata is required")
                crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True)
                bbox = metadata.get("bbox")
                bbox = (
                    bbox.to(device=device, dtype=torch.float32, non_blocking=True)
                    if torch.is_tensor(bbox)
                    else crop_bbox
                )
                image_mask = metadata.get("image_mask")
                image_mask = (
                    image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
                    if torch.is_tensor(image_mask)
                    else None
                )

                captured.clear()
                with autocast_context(device, bool(amp)):
                    features = model.forward_features(
                        images,
                        image_valid_mask=image_mask,
                        bbox_token_prior=crop_bbox,
                    )
                    features["bbox"] = bbox
                    output = (
                        model.forward_heads(features)
                        if hasattr(model, "forward_heads")
                        else classification_logits_from_features(model, features)
                    )
                    logits, _, _ = extract_detection_from_model_output(output)
                if len(captured) != 1 or not torch.is_tensor(captured[0]):
                    raise RuntimeError("Expected exactly one CNN stem activation")
                current_shape = tuple(int(value) for value in captured[0].shape[1:])
                stem_shape = current_shape if stem_shape is None else stem_shape
                if current_shape != stem_shape or current_shape != (256, 32, 32):
                    raise RuntimeError(f"Unexpected stem shape: {current_shape}")
                descriptors = extract_ms_lacunarity_descriptors(
                    captured[0],
                    crop_bbox,
                    image_mask,
                    erode_ratio=float(erode_ratio),
                )
                for name in descriptor_batches:
                    descriptor_batches[name].append(
                        descriptors[name].detach().float().cpu().numpy()
                    )
                for name in support_batches:
                    support_batches[name].append(
                        descriptors[name].detach().float().cpu().numpy()
                    )
                probability_batches.append(
                    logits.detach().float().softmax(dim=1).cpu().numpy()
                )
                label_batches.append(labels.detach().cpu().numpy())

                batch_count = int(labels.numel())
                raw_paths = metadata.get("paths", [])
                batch_paths = (
                    [str(path) for path in raw_paths]
                    if isinstance(raw_paths, Sequence)
                    else []
                )
                if dataset_paths and (
                    len(batch_paths) != batch_count
                    or not any(path.strip() for path in batch_paths)
                ):
                    batch_paths = dataset_paths[seen : seen + batch_count]
                if len(batch_paths) != batch_count:
                    raise ValueError("Extracted path count does not match batch size")
                paths.extend(batch_paths)
                fallback = np.arange(seen, seen + batch_count, dtype=np.int64)
                sample_index = metadata.get("sample_index")
                if torch.is_tensor(sample_index) and int(sample_index.numel()) == batch_count:
                    sample_index_batches.append(
                        sample_index.detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64, copy=False)
                        .reshape(-1)
                    )
                else:
                    sample_index_batches.append(fallback)
                seen += batch_count
    finally:
        hook.remove()

    labels_array = np.concatenate(label_batches).astype(np.int64, copy=False)
    payload: Dict[str, object] = {
        name: np.concatenate(values).astype(np.float32, copy=False)
        for name, values in descriptor_batches.items()
    }
    payload["supports"] = {
        name: np.concatenate(values).astype(np.float32, copy=False)
        for name, values in support_batches.items()
    }
    payload.update(
        {
            "probabilities": np.concatenate(probability_batches).astype(
                np.float32, copy=False
            ),
            "labels": labels_array,
            "sample_index": np.concatenate(sample_index_batches).astype(
                np.int64, copy=False
            ),
            "paths": np.asarray(paths, dtype=object),
            "source_stem": np.asarray(
                [_normalized_source_group(Path(path).stem) for path in paths],
                dtype=object,
            ),
            "stem_shape": list(stem_shape or ()),
            "seconds": float(time.perf_counter() - start),
            "peak_cuda_memory_mib": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    row_counts = {
        len(paths),
        int(labels_array.size),
        int(np.asarray(payload["control"]).shape[0]),
        int(np.asarray(payload["candidate"]).shape[0]),
    }
    if len(row_counts) != 1:
        raise RuntimeError("Extracted MS-lacunarity arrays have inconsistent rows")
    return payload


def _fit_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _apply_scaler(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float64) - mean) / scale).astype(
        np.float32, copy=False
    )


def _fit_readout(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    c_value: float,
    max_iterations: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mean, scale = _fit_scaler(fit_features)
    fit_scaled = _apply_scaler(fit_features, mean, scale)
    eval_scaled = _apply_scaler(eval_features, mean, scale)
    model = LogisticRegression(
        C=float(c_value),
        class_weight=None,
        fit_intercept=True,
        max_iter=int(max_iterations),
        random_state=int(seed),
        solver="lbfgs",
        tol=1e-5,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(fit_scaled, np.asarray(fit_labels, dtype=np.int64))
    raw_probabilities = model.predict_proba(eval_scaled).astype(np.float32, copy=False)
    probabilities = np.zeros((len(eval_scaled), 5), dtype=np.float32)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if np.any((classes < 0) | (classes >= 5)):
        raise ValueError(f"Readout returned invalid class indices: {classes.tolist()}")
    probabilities[:, classes] = raw_probabilities
    iterations = int(np.max(model.n_iter_))
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    converged = bool(converged and iterations < int(max_iterations))
    return probabilities, {
        "converged": converged,
        "iterations": iterations,
        "warning_count": len(caught),
        "feature_dim": int(fit_scaled.shape[1]),
        "fit_rows": int(fit_scaled.shape[0]),
    }


def fit_source_grouped_readouts(
    train: Mapping[str, object],
    holdout: Mapping[str, object],
    *,
    folds: int,
    c_value: float,
    max_iterations: int,
    seed: int,
) -> Dict[str, object]:
    labels = np.asarray(train["labels"], dtype=np.int64)
    groups = np.asarray(train["source_stem"], dtype=object)
    holdout_labels = np.asarray(holdout["labels"], dtype=np.int64)
    roles = ("control", "candidate")
    oof = {
        role: np.zeros((len(labels), 5), dtype=np.float32) for role in roles
    }
    fold_assignment = np.full(len(labels), -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    fit_records: List[Dict[str, object]] = []
    maximum_overlap = 0
    folds_with_focus_gain = 0
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for fold_index, (fit_indices, eval_indices) in enumerate(splits):
        fit_sources = set(groups[fit_indices].tolist())
        eval_sources = set(groups[eval_indices].tolist())
        overlap = len(fit_sources.intersection(eval_sources))
        maximum_overlap = max(maximum_overlap, overlap)
        role_records: Dict[str, Mapping[str, object]] = {}
        for role in roles:
            probabilities, record = _fit_readout(
                np.asarray(train[role])[fit_indices],
                labels[fit_indices],
                np.asarray(train[role])[eval_indices],
                c_value=float(c_value),
                max_iterations=int(max_iterations),
                seed=int(seed) + fold_index,
            )
            oof[role][eval_indices] = probabilities
            role_records[role] = record
            fit_records.append(
                {"scope": f"fold_{fold_index}", "role": role, **record}
            )
        fold_assignment[eval_indices] = int(fold_index)
        control_metrics = _classification_metrics(labels[eval_indices], oof["control"][eval_indices])
        candidate_metrics = _classification_metrics(
            labels[eval_indices], oof["candidate"][eval_indices]
        )
        control_focus = control_metrics["per_class"][FOCUS_CLASS_INDEX]
        candidate_focus = candidate_metrics["per_class"][FOCUS_CLASS_INDEX]
        focus_gain = float(candidate_focus["f1"] - control_focus["f1"])
        folds_with_focus_gain += int(focus_gain > 0.0)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(eval_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(eval_sources)),
                "source_overlap": int(overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "macro_f1_gain": float(
                    candidate_metrics["macro_f1"] - control_metrics["macro_f1"]
                ),
                "control_focus_precision": float(control_focus["precision"]),
                "candidate_focus_precision": float(candidate_focus["precision"]),
                "focus_precision_gain": float(
                    candidate_focus["precision"] - control_focus["precision"]
                ),
                "control_focus_f1": float(control_focus["f1"]),
                "candidate_focus_f1": float(candidate_focus["f1"]),
                "focus_f1_gain": focus_gain,
                "control_converged": bool(role_records["control"]["converged"]),
                "candidate_converged": bool(role_records["candidate"]["converged"]),
            }
        )
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF fold assignment is incomplete")

    holdout_probabilities: Dict[str, np.ndarray] = {}
    for role in roles:
        probabilities, record = _fit_readout(
            np.asarray(train[role]),
            labels,
            np.asarray(holdout[role]),
            c_value=float(c_value),
            max_iterations=int(max_iterations),
            seed=int(seed) + 1000,
        )
        holdout_probabilities[role] = probabilities
        fit_records.append({"scope": "full", "role": role, **record})
    return {
        "oof_probabilities": oof,
        "holdout_probabilities": holdout_probabilities,
        "oof_metrics": {
            role: _classification_metrics(labels, oof[role]) for role in roles
        },
        "holdout_metrics": {
            role: _classification_metrics(holdout_labels, holdout_probabilities[role])
            for role in roles
        },
        "fold_assignment": fold_assignment,
        "fold_rows": fold_rows,
        "fit_records": fit_records,
        "maximum_source_overlap": int(maximum_overlap),
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "all_converged": all(bool(row["converged"]) for row in fit_records),
    }


def _effective_rank(features: np.ndarray, *, maximum_rows: int = 512) -> float:
    values = np.asarray(features, dtype=np.float64)
    if len(values) > int(maximum_rows):
        indices = np.linspace(0, len(values) - 1, int(maximum_rows), dtype=np.int64)
        values = values[indices]
    mean, scale = _fit_scaler(values)
    standardized = _apply_scaler(values, mean, scale).astype(np.float64, copy=False)
    singular = np.linalg.svd(standardized, compute_uv=False, full_matrices=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 0.0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _aligned_block_correlations(
    control: np.ndarray,
    candidate_only: np.ndarray,
) -> Dict[str, object]:
    control = np.asarray(control, dtype=np.float64)
    candidate_only = np.asarray(candidate_only, dtype=np.float64)
    pairs = {
        "full_l0_vs_full_mean": (control[:, :256], candidate_only[:, 0:256]),
        "full_l1_vs_full_mean": (control[:, :256], candidate_only[:, 256:512]),
        "object_l0_vs_object_mean": (
            control[:, 256:512],
            candidate_only[:, 512:768],
        ),
        "object_l1_vs_object_mean": (
            control[:, 256:512],
            candidate_only[:, 768:1024],
        ),
    }
    result: Dict[str, object] = {}
    all_values: List[np.ndarray] = []
    for name, (first, second) in pairs.items():
        first_centered = first - first.mean(axis=0, keepdims=True)
        second_centered = second - second.mean(axis=0, keepdims=True)
        denominator = np.sqrt(
            np.square(first_centered).sum(axis=0)
            * np.square(second_centered).sum(axis=0)
        )
        correlation = np.divide(
            (first_centered * second_centered).sum(axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        absolute = np.abs(correlation)
        all_values.append(absolute)
        result[name] = {
            "mean_abs": float(absolute.mean()),
            "max_abs": float(absolute.max()),
        }
    combined = np.concatenate(all_values)
    result["all"] = {
        "mean_abs": float(combined.mean()),
        "max_abs": float(combined.max()),
    }
    return result


def _restricted_fp_transitions(
    targets: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> Dict[str, int]:
    targets = np.asarray(targets, dtype=np.int64)
    control = np.asarray(control_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    restricted = np.isin(targets, np.asarray(RESTRICTED_NEGATIVE_CLASSES))
    return {
        "control": int((restricted & (control == FOCUS_CLASS_INDEX)).sum()),
        "candidate": int((restricted & (candidate == FOCUS_CLASS_INDEX)).sum()),
        "removed": int(
            (restricted & (control == FOCUS_CLASS_INDEX) & (candidate != FOCUS_CLASS_INDEX)).sum()
        ),
        "created": int(
            (restricted & (control != FOCUS_CLASS_INDEX) & (candidate == FOCUS_CLASS_INDEX)).sum()
        ),
    }


def assess_ms_lacunarity_readiness(
    *,
    train_rows: int,
    holdout_rows: int,
    train_holdout_source_overlap: int,
    fold_source_overlap: int,
    features_finite: bool,
    control_dim: int,
    candidate_dim: int,
    candidate_only_effective_rank: float,
    all_converged: bool,
    peak_cuda_memory_mib: float,
    folds_with_focus_gain: int,
    oof_metrics: Mapping[str, Mapping[str, object]],
    holdout_metrics: Mapping[str, Mapping[str, object]],
    holdout_transitions: Mapping[str, object],
    restricted_fp: Mapping[str, int],
    oof_direction_auc: Optional[float],
    holdout_direction_auc: Optional[float],
    test_split_used: bool,
    model_or_checkpoint_written: bool,
) -> Dict[str, object]:
    oof_control = oof_metrics["control"]
    oof_candidate = oof_metrics["candidate"]
    hold_control = holdout_metrics["control"]
    hold_candidate = holdout_metrics["candidate"]
    oof_control_focus = oof_control["per_class"][FOCUS_CLASS_INDEX]
    oof_candidate_focus = oof_candidate["per_class"][FOCUS_CLASS_INDEX]
    hold_control_focus = hold_control["per_class"][FOCUS_CLASS_INDEX]
    hold_candidate_focus = hold_candidate["per_class"][FOCUS_CLASS_INDEX]
    nonfocus_drops = [
        float(hold_control["per_class"][index]["f1"])
        - float(hold_candidate["per_class"][index]["f1"])
        for index in range(5)
        if index != FOCUS_CLASS_INDEX
    ]
    observed = {
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "train_holdout_source_overlap": int(train_holdout_source_overlap),
        "fold_source_overlap": int(fold_source_overlap),
        "features_finite": bool(features_finite),
        "control_dim": int(control_dim),
        "candidate_dim": int(candidate_dim),
        "candidate_only_effective_rank": float(candidate_only_effective_rank),
        "all_converged": bool(all_converged),
        "peak_cuda_memory_mib": float(peak_cuda_memory_mib),
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "oof_macro_gain": float(oof_candidate["macro_f1"] - oof_control["macro_f1"]),
        "oof_focus_f1_gain": float(oof_candidate_focus["f1"] - oof_control_focus["f1"]),
        "holdout_macro_gain": float(
            hold_candidate["macro_f1"] - hold_control["macro_f1"]
        ),
        "holdout_focus_precision_gain": float(
            hold_candidate_focus["precision"] - hold_control_focus["precision"]
        ),
        "holdout_focus_recall_delta": float(
            hold_candidate_focus["recall"] - hold_control_focus["recall"]
        ),
        "holdout_focus_f1_gain": float(
            hold_candidate_focus["f1"] - hold_control_focus["f1"]
        ),
        "maximum_nonfocus_f1_drop": float(max(nonfocus_drops, default=0.0)),
        "holdout_transitions": dict(holdout_transitions),
        "restricted_fp": dict(restricted_fp),
        "oof_direction_auc": oof_direction_auc,
        "holdout_direction_auc": holdout_direction_auc,
        "test_split_used": bool(test_split_used),
        "model_or_checkpoint_written": bool(model_or_checkpoint_written),
    }
    checks = {
        "full_support_exact": (
            observed["train_rows"] == EXPECTED_FIT_ROWS
            and observed["holdout_rows"] == EXPECTED_HOLDOUT_ROWS
        ),
        "train_holdout_sources_disjoint": observed["train_holdout_source_overlap"] == 0,
        "source_group_folds_disjoint": observed["fold_source_overlap"] == 0,
        "features_finite": observed["features_finite"],
        "descriptor_dimensions_exact": (
            observed["control_dim"] == EXPECTED_CONTROL_DIM
            and observed["candidate_dim"] == EXPECTED_CANDIDATE_DIM
        ),
        "candidate_only_effective_rank_ge_8": observed[
            "candidate_only_effective_rank"
        ]
        >= 8.0,
        "all_solvers_converged": observed["all_converged"],
        "peak_cuda_memory_le_7680_mib": observed["peak_cuda_memory_mib"] <= 7680.0,
        "oof_macro_nonnegative": observed["oof_macro_gain"] >= 0.0,
        "holdout_macro_gain_ge_0p003": observed["holdout_macro_gain"] >= 0.003,
        "oof_focus_f1_positive": observed["oof_focus_f1_gain"] > 0.0,
        "focus_gain_in_at_least_3_folds": observed["folds_with_focus_gain"] >= 3,
        "holdout_focus_f1_gain_ge_0p015": observed["holdout_focus_f1_gain"] >= 0.015,
        "holdout_focus_precision_gain_ge_0p025": observed[
            "holdout_focus_precision_gain"
        ]
        >= 0.025,
        "holdout_focus_recall_preserved": observed["holdout_focus_recall_delta"] >= -0.010,
        "restricted_fp_net_removed_ge_4": int(restricted_fp["removed"])
        - int(restricted_fp["created"])
        >= 4,
        "corrections_ge_harms": int(holdout_transitions["corrections"])
        >= int(holdout_transitions["harms"]),
        "focus_fn_rescued_ge_tp_broken": int(
            holdout_transitions["focus_false_negative_rescued"]
        )
        >= int(holdout_transitions["focus_true_positive_broken"]),
        "maximum_nonfocus_f1_drop_le_0p010": observed[
            "maximum_nonfocus_f1_drop"
        ]
        <= 0.010,
        "oof_direction_auc_ge_0p60": oof_direction_auc is not None
        and float(oof_direction_auc) >= 0.60,
        "holdout_direction_auc_ge_0p60": holdout_direction_auc is not None
        and float(holdout_direction_auc) >= 0.60,
        "test_split_closed": not observed["test_split_used"],
        "no_model_or_checkpoint_written": not observed[
            "model_or_checkpoint_written"
        ],
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "condition_replay_permission": not failed,
        "image_smoke_permission": False,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _prediction_rows(
    *,
    split: str,
    payload: Mapping[str, object],
    fold_assignment: np.ndarray,
    control_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> List[Dict[str, object]]:
    labels = np.asarray(payload["labels"], dtype=np.int64)
    keeper = np.asarray(payload["probabilities"], dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for index in range(len(labels)):
        row: Dict[str, object] = {
            "split": str(split),
            "row_index": int(index),
            "sample_index": int(np.asarray(payload["sample_index"])[index]),
            "fold": int(fold_assignment[index]),
            "source_stem": str(np.asarray(payload["source_stem"])[index]),
            "image_path": str(np.asarray(payload["paths"])[index]),
            "target_index": int(labels[index]),
            "keeper_prediction_index": int(keeper[index].argmax()),
            "control_prediction_index": int(control_probabilities[index].argmax()),
            "candidate_prediction_index": int(candidate_probabilities[index].argmax()),
        }
        for class_index in range(5):
            row[f"keeper_prob_{class_index}"] = float(keeper[index, class_index])
            row[f"control_prob_{class_index}"] = float(
                control_probabilities[index, class_index]
            )
            row[f"candidate_prob_{class_index}"] = float(
                candidate_probabilities[index, class_index]
            )
        rows.append(row)
    return rows


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_args(args)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    provenance = verify_provenance(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint))
    if not hasattr(model, "stem") or not hasattr(model, "forward_features"):
        raise ValueError("MS-lacunarity audit requires the TRKH CNN stem")
    device = _resolve_device(str(args.device))
    model.to(device).eval()

    class_names: List[str] = []
    payloads: Dict[str, Dict[str, object]] = {}
    split_specs = (
        ("train", int(args.max_train_samples)),
        ("val", int(args.max_val_samples)),
    )
    for split, maximum in split_specs:
        dataset, split_class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=split,
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=maximum,
        )
        if class_names and list(split_class_names) != class_names:
            raise ValueError("Train/holdout class order differs")
        class_names = list(split_class_names)
        payloads[split] = _extract_split(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.extract_batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            erode_ratio=float(args.interior_erode_ratio),
            split=split,
        )
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if len(class_names) != 5:
        raise ValueError("Locked audit requires exactly five classes")
    train = payloads["train"]
    holdout = payloads["val"]
    train_groups = np.asarray(train["source_stem"], dtype=object)
    holdout_groups = np.asarray(holdout["source_stem"], dtype=object)
    train_holdout_overlap = len(
        set(train_groups.tolist()).intersection(holdout_groups.tolist())
    )
    if train_holdout_overlap:
        raise ValueError(f"Train/holdout source overlap: {train_holdout_overlap}")

    readouts = fit_source_grouped_readouts(
        train,
        holdout,
        folds=int(args.folds),
        c_value=float(args.readout_c),
        max_iterations=int(args.max_iterations),
        seed=SEED,
    )
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    holdout_labels = np.asarray(holdout["labels"], dtype=np.int64)
    oof_control = readouts["oof_probabilities"]["control"]
    oof_candidate = readouts["oof_probabilities"]["candidate"]
    hold_control = readouts["holdout_probabilities"]["control"]
    hold_candidate = readouts["holdout_probabilities"]["candidate"]
    transitions = {
        "oof_candidate_vs_control": _transition_stats(
            train_labels,
            oof_control,
            oof_candidate,
            focus_class_index=FOCUS_CLASS_INDEX,
        ),
        "holdout_candidate_vs_control": _transition_stats(
            holdout_labels,
            hold_control,
            hold_candidate,
            focus_class_index=FOCUS_CLASS_INDEX,
        ),
    }
    directions = {
        "oof_candidate_vs_control": _direction_auc(
            train_labels,
            oof_control,
            oof_candidate,
            focus_class_index=FOCUS_CLASS_INDEX,
        ),
        "holdout_candidate_vs_control": _direction_auc(
            holdout_labels,
            hold_control,
            hold_candidate,
            focus_class_index=FOCUS_CLASS_INDEX,
        ),
    }
    restricted_fp = _restricted_fp_transitions(
        holdout_labels, hold_control, hold_candidate
    )
    candidate_only_rank = _effective_rank(np.asarray(train["candidate_only"]))
    features_finite = all(
        np.isfinite(np.asarray(payloads[split][name], dtype=np.float32)).all()
        for split in ("train", "val")
        for name in ("control", "candidate_only", "candidate", "probabilities")
    )
    peak_memory = max(
        float(payloads[split]["peak_cuda_memory_mib"]) for split in ("train", "val")
    )
    gate = assess_ms_lacunarity_readiness(
        train_rows=len(train_labels),
        holdout_rows=len(holdout_labels),
        train_holdout_source_overlap=train_holdout_overlap,
        fold_source_overlap=int(readouts["maximum_source_overlap"]),
        features_finite=features_finite,
        control_dim=int(np.asarray(train["control"]).shape[1]),
        candidate_dim=int(np.asarray(train["candidate"]).shape[1]),
        candidate_only_effective_rank=candidate_only_rank,
        all_converged=bool(readouts["all_converged"]),
        peak_cuda_memory_mib=peak_memory,
        folds_with_focus_gain=int(readouts["folds_with_focus_gain"]),
        oof_metrics=readouts["oof_metrics"],
        holdout_metrics=readouts["holdout_metrics"],
        holdout_transitions=transitions["holdout_candidate_vs_control"],
        restricted_fp=restricted_fp,
        oof_direction_auc=directions["oof_candidate_vs_control"]["auc_fn_positive"],
        holdout_direction_auc=directions["holdout_candidate_vs_control"][
            "auc_fn_positive"
        ],
        test_split_used=False,
        model_or_checkpoint_written=False,
    )

    _write_csv(output_dir / "fold_metrics.csv", readouts["fold_rows"])
    _write_csv(output_dir / "readout_fits.csv", readouts["fit_records"])
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            payload=train,
            fold_assignment=np.asarray(readouts["fold_assignment"], dtype=np.int64),
            control_probabilities=oof_control,
            candidate_probabilities=oof_candidate,
        ),
    )
    _write_csv(
        output_dir / "holdout_predictions.csv",
        _prediction_rows(
            split="train_only_holdout",
            payload=holdout,
            fold_assignment=np.full(len(holdout_labels), -1, dtype=np.int64),
            control_probabilities=hold_control,
            candidate_probabilities=hold_candidate,
        ),
    )

    full_support = (
        len(train_labels) == EXPECTED_FIT_ROWS
        and len(holdout_labels) == EXPECTED_HOLDOUT_ROWS
    )
    support_telemetry = {
        split: {
            name: {
                "minimum": float(np.asarray(values).min()),
                "mean": float(np.asarray(values).mean()),
                "maximum": float(np.asarray(values).max()),
            }
            for name, values in payloads[split]["supports"].items()
        }
        for split in ("train", "val")
    }
    descriptor_telemetry = {
        "control_dim": int(np.asarray(train["control"]).shape[1]),
        "candidate_only_dim": int(np.asarray(train["candidate_only"]).shape[1]),
        "candidate_dim": int(np.asarray(train["candidate"]).shape[1]),
        "candidate_only_effective_rank": float(candidate_only_rank),
        "aligned_control_correlations": _aligned_block_correlations(
            np.asarray(train["control"]), np.asarray(train["candidate_only"])
        ),
        "supports": support_telemetry,
        "finite": bool(features_finite),
    }
    protocol = {
        "method": "frozen_keeper_stem_multiscale_lacunarity_readiness",
        "support_mode": "full_decision" if full_support else "prefix_preflight_only",
        "paper": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_tree": OFFICIAL_TREE,
        "provenance": provenance,
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(checkpoint_path),
        "split_usage": {"train": True, "val": True, "test": False},
        "train_rows": int(len(train_labels)),
        "holdout_rows": int(len(holdout_labels)),
        "train_sources": int(np.unique(train_groups).size),
        "holdout_sources": int(np.unique(holdout_groups).size),
        "train_holdout_source_overlap": int(train_holdout_overlap),
        "stem_shape": train["stem_shape"],
        "activation_scaling": "((tanh(F)+1)/2)*255",
        "pyramid": "native_plus_fixed_binomial_5x5_gaussian_stride2",
        "regions": "full_valid_crop_and_12pct_eroded_object_bbox",
        "control": "native_region_means_512d",
        "candidate": "control_plus_mean_times_per_scale_lacunarity_1536d",
        "readout": {
            "type": "natural_frequency_multinomial_logistic_regression",
            "folds": int(args.folds),
            "source_grouped": True,
            "c": float(args.readout_c),
            "solver": "lbfgs",
            "max_iterations": int(args.max_iterations),
            "class_weight": None,
        },
        "keeper_predictions": "descriptive_in_sample_reference_only",
        "runtime": {
            "device": str(device),
            "extract_batch_size": int(args.extract_batch_size),
            "workers": int(args.workers),
            "amp": bool(args.amp),
            "torch_threads": int(args.torch_threads),
        },
        "raw_dataset_modified": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
        "descriptor_cache_written": False,
    }
    summary = {
        "protocol": protocol,
        "class_names": class_names,
        "extraction": {
            split: {
                "rows": int(len(np.asarray(payloads[split]["labels"]))),
                "sources": int(np.unique(payloads[split]["source_stem"]).size),
                "seconds": float(payloads[split]["seconds"]),
                "peak_cuda_memory_mib": float(
                    payloads[split]["peak_cuda_memory_mib"]
                ),
            }
            for split in ("train", "val")
        },
        "descriptor_telemetry": descriptor_telemetry,
        "metrics": {
            "keeper_descriptive": {
                "train": _classification_metrics(
                    train_labels, np.asarray(train["probabilities"])
                ),
                "holdout": _classification_metrics(
                    holdout_labels, np.asarray(holdout["probabilities"])
                ),
            },
            "train_oof": readouts["oof_metrics"],
            "holdout": readouts["holdout_metrics"],
        },
        "transitions": transitions,
        "restricted_fp": restricted_fp,
        "directions": directions,
        "gate": gate,
        "seconds": float(time.perf_counter() - start),
        "raw_dataset_modified": False,
        "test_split_used": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    observed = gate["observed"]
    readme = [
        "# MS-Lacunarity Stem Readiness",
        "",
        (
            f"- Support: `{protocol['support_mode']}`; fit/holdout "
            f"`{len(train_labels)}/{len(holdout_labels)}`; test closed."
        ),
        (
            "- OOF macro/class1 gains: "
            f"`{observed['oof_macro_gain']:+.6f}/"
            f"{observed['oof_focus_f1_gain']:+.6f}`."
        ),
        (
            "- Holdout macro/class1/precision/recall deltas: "
            f"`{observed['holdout_macro_gain']:+.6f}/"
            f"{observed['holdout_focus_f1_gain']:+.6f}/"
            f"{observed['holdout_focus_precision_gain']:+.6f}/"
            f"{observed['holdout_focus_recall_delta']:+.6f}`."
        ),
        (
            "- Restricted FP control/candidate/removed/created: "
            f"`{restricted_fp['control']}/{restricted_fp['candidate']}/"
            f"{restricted_fp['removed']}/{restricted_fp['created']}`."
        ),
        (
            "- Candidate-only effective rank: "
            f"`{candidate_only_rank:.6f}`."
        ),
        (
            "- Condition-replay permission: "
            f"`{str(bool(gate['condition_replay_permission'])).lower()}`."
        ),
        f"- Failed checks: `{','.join(gate['failed_checks'])}`.",
        "",
        (
            "Decision: run shifted-condition replay and changed-case XAI only "
            "when every fixed clean gate passes. Otherwise close this exact "
            "route without scale/readout/model sweeps."
        ),
        "",
        "No model, checkpoint, test payload, or descriptor cache is written.",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    manifest = _write_artifact_manifest(
        output_dir, mode="ms_lacunarity_stem_readiness_evidence_manifest"
    )
    return {
        "gate": gate,
        "artifact_manifest": {
            key: value for key, value in manifest.items() if key != "files"
        },
        "seconds": summary["seconds"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_audit(args)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
