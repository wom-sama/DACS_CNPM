from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from trkh.core.config import DataSpec, load_data_spec
from trkh.data.dataset import MangoYOLOCropDataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


class ClassificationTransformAdapter:
    def __init__(self, transform: nn.Module) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image, target: Optional[Mapping[str, object]] = None):
        return self.transform(image), target


class OrderedFolderDataset(Dataset):
    def __init__(
        self,
        *,
        data_spec: DataSpec,
        split: str,
        transform: nn.Module,
        max_samples_per_split: int = 0,
        max_samples_per_class: int = 0,
    ) -> None:
        self.data_spec = data_spec
        self.split = str(split)
        self.transform = transform
        self.class_names = [str(value) for value in data_spec.class_names]
        self.records = self._index_records(
            max_samples_per_split=max_samples_per_split,
            max_samples_per_class=max_samples_per_class,
        )

    def _index_records(
        self,
        *,
        max_samples_per_split: int,
        max_samples_per_class: int,
    ) -> List[ImageRecord]:
        root = self.data_spec.split_images_dir(self.split)
        records: List[ImageRecord] = []
        for class_index, class_name in enumerate(self.class_names):
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            class_count = 0
            for image_path in sorted(class_dir.rglob("*"), key=lambda item: str(item).lower()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(ImageRecord(path=image_path, label=class_index))
                    class_count += 1
                    if max_samples_per_class > 0 and class_count >= int(max_samples_per_class):
                        break
                    if max_samples_per_split > 0 and len(records) >= int(max_samples_per_split):
                        return records
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int, str]:
        record = self.records[int(index)]
        with Image.open(record.path) as image:
            rgb = image.convert("RGB")
        return (
            self.transform(rgb),
            int(record.label),
            str(record.path),
            int(index),
            str(record.path.stem),
        )


class OrderedYoloObjectDataset(Dataset):
    def __init__(
        self,
        *,
        data_spec: DataSpec,
        split: str,
        transform: nn.Module,
        max_samples_per_split: int = 0,
        max_samples_per_class: int = 0,
        crop_margin_ratio: float = 0.05,
    ) -> None:
        self.data_spec = data_spec
        self.split = str(split)
        self.class_names = [str(value) for value in data_spec.class_names]
        self.dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=ClassificationTransformAdapter(transform),
            crop_margin_ratio=float(crop_margin_ratio),
            crop_to_primary_object=True,
            fallback_to_full_image=True,
            classification_target=True,
            classification_object_crops=True,
        )
        self.indices = self._select_indices(
            max_samples_per_split=max_samples_per_split,
            max_samples_per_class=max_samples_per_class,
        )

    def _select_indices(
        self,
        *,
        max_samples_per_split: int,
        max_samples_per_class: int,
    ) -> List[int]:
        labels = [int(value) for value in self.dataset.labels()]
        selected: List[int] = []
        per_class: Dict[int, int] = {}
        for sample_index, label in enumerate(labels):
            if max_samples_per_class > 0:
                count = int(per_class.get(label, 0))
                if count >= int(max_samples_per_class):
                    continue
                per_class[label] = count + 1
            selected.append(int(sample_index))
            if max_samples_per_split > 0 and len(selected) >= int(max_samples_per_split):
                break
        return selected

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, int, str]:
        sample_index = int(self.indices[int(index)])
        item = self.dataset[sample_index]
        if not isinstance(item, tuple) or len(item) < 2:
            raise RuntimeError("Unexpected YOLO object dataset item.")
        image_tensor = item[0]
        label = int(item[1])
        sample = self.dataset.samples[sample_index]
        image_path = Path(sample.image_path)
        return image_tensor, label, str(image_path), sample_index, str(image_path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained embedding retrieval expert on class_f without "
            "training the image encoder. Hyperparameters are selected on val only."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="", help="TIMM model name, e.g. vit_small_patch14_dinov2.lvd142m.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional fine-tuned TIMM checkpoint from image_baseline_experiments.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional stratified cap per class for smoke/probe runs.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    parser.add_argument(
        "--brightness-deltas",
        default="",
        help="Comma separated RGB brightness deltas for feature TTA, e.g. -0.04,0.04.",
    )
    parser.add_argument(
        "--contrast-scales",
        default="",
        help="Comma separated RGB contrast scales for feature TTA, e.g. 0.95,1.05.",
    )
    parser.add_argument(
        "--knn-k",
        default="1,3,5,7,11,15,21,31",
        help="Comma separated k values for cosine kNN.",
    )
    parser.add_argument(
        "--temperatures",
        default="0.03,0.05,0.07,0.10,0.15,0.25,0.50",
        help="Comma separated softmax temperatures for weighted kNN/prototype.",
    )
    parser.add_argument(
        "--logreg-c",
        default="0.05,0.1,0.25,0.5,1.0,2.0,4.0",
        help="Comma separated C values for linear logistic regression on frozen embeddings.",
    )
    parser.add_argument("--disable-logreg", action="store_true")
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--focus-class-weight", type=float, default=0.35)
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    output: List[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            output.append(float(item))
    return output


def _parse_int_list(value: str) -> List[int]:
    output: List[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            output.append(max(1, int(item)))
    return output


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    return device


def _build_transform(model_name: str) -> Tuple[nn.Module, Tuple[float, ...], Tuple[float, ...], Tuple[int, int]]:
    probe = timm.create_model(str(model_name), pretrained=False)
    cfg = timm.data.resolve_model_data_config(probe)
    size = tuple(int(value) for value in cfg.get("input_size", (3, 224, 224))[-2:])
    mean = tuple(float(value) for value in cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in cfg.get("std", (0.229, 0.224, 0.225)))
    transform = transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
    )
    return transform, mean, std, size


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def _flatten_feature(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model returned an empty feature tuple.")
        output = output[-1]
    if not torch.is_tensor(output):
        raise RuntimeError(f"Model returned unsupported feature type: {type(output)!r}")
    if output.ndim == 4:
        output = output.mean(dim=(-2, -1))
    elif output.ndim == 3:
        output = output.mean(dim=1)
    elif output.ndim > 2:
        output = output.flatten(start_dim=1)
    return output.float()


def _rgb_from_normalized(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images * std + mean).clamp(0.0, 1.0)


def _normalized_from_rgb(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images - mean) / std


def _feature_tta_batches(
    images: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
) -> List[torch.Tensor]:
    batches = [images]
    if horizontal_flip:
        batches.append(torch.flip(images, dims=(-1,)))
    if brightness_deltas or contrast_scales:
        mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        rgb = _rgb_from_normalized(images, mean_tensor, std_tensor)
        for delta in brightness_deltas:
            if abs(float(delta)) <= 1e-12:
                continue
            batches.append(_normalized_from_rgb((rgb + float(delta)).clamp(0.0, 1.0), mean_tensor, std_tensor))
        for scale in contrast_scales:
            if abs(float(scale) - 1.0) <= 1e-12:
                continue
            batches.append(
                _normalized_from_rgb(((rgb - 0.5) * float(scale) + 0.5).clamp(0.0, 1.0), mean_tensor, std_tensor)
            )
    return batches


def _infer_model_name(args: argparse.Namespace, checkpoint: Optional[Mapping[str, object]]) -> str:
    requested = str(args.model or "").strip()
    if requested:
        return requested
    if checkpoint is not None:
        ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, Mapping) else {}
        if isinstance(ckpt_args, Mapping):
            inferred = str(ckpt_args.get("model", "")).strip()
            if inferred:
                return inferred
    raise ValueError("Pass --model or a checkpoint whose args contain a model name.")


def _build_feature_model(
    model_name: str,
    *,
    pretrained: bool,
    checkpoint: Optional[Mapping[str, object]] = None,
) -> nn.Module:
    if checkpoint is not None:
        classes = list(checkpoint.get("classes", [])) if isinstance(checkpoint, Mapping) else []
        if not classes:
            raise ValueError("Checkpoint missing 'classes'; cannot infer classifier shape before feature reset.")
        state_dict = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
        if not isinstance(state_dict, Mapping):
            raise ValueError("Checkpoint missing model state dict under key 'model'.")
        model = timm.create_model(str(model_name), pretrained=False, num_classes=len(classes))
        model.load_state_dict(state_dict)
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(0)
        else:
            raise ValueError(
                f"Model {model_name!r} does not expose reset_classifier; checkpoint feature extraction is unsupported."
            )
        return model
    try:
        return timm.create_model(str(model_name), pretrained=bool(pretrained), num_classes=0)
    except TypeError:
        return timm.create_model(str(model_name), pretrained=bool(pretrained), num_classes=0, global_pool="avg")


def _extract_features(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    mean: Sequence[float],
    std: Sequence[float],
    horizontal_flip: bool,
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
    description: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
    model.eval()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    features: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    sample_indices: List[int] = []
    source_stems: List[str] = []
    with torch.inference_mode():
        for images, batch_labels, batch_paths, batch_indices, batch_stems in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            feature_sum: Optional[torch.Tensor] = None
            variants = _feature_tta_batches(
                images,
                mean=mean,
                std=std,
                horizontal_flip=horizontal_flip,
                brightness_deltas=brightness_deltas,
                contrast_scales=contrast_scales,
            )
            for variant_images in variants:
                with torch.autocast(device_type=autocast_device, enabled=bool(amp) and device.type == "cuda"):
                    output = model(variant_images)
                variant_features = _flatten_feature(output)
                feature_sum = variant_features if feature_sum is None else feature_sum + variant_features
            if feature_sum is None:
                raise RuntimeError("No feature variants were produced.")
            batch_features = feature_sum / float(len(variants))
            features.append(batch_features.cpu().numpy().astype(np.float32))
            labels.extend(int(value) for value in batch_labels.tolist())
            paths.extend(str(value) for value in batch_paths)
            sample_indices.extend(int(value) for value in batch_indices.tolist())
            source_stems.extend(str(value) for value in batch_stems)
    if not features:
        raise RuntimeError(f"No features extracted for {description}.")
    return (
        np.concatenate(features, axis=0),
        np.asarray(labels, dtype=np.int64),
        paths,
        np.asarray(sample_indices, dtype=np.int64),
        source_stems,
    )


def _cache_path(output_dir: Path, split: str, model_name: str) -> Path:
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model_name)
    return output_dir / f"features_{split}_{safe_model}.npz"


def _load_or_extract_split(
    *,
    args: argparse.Namespace,
    model_name: str,
    cache_key: str,
    data_spec: DataSpec,
    split: str,
    transform: nn.Module,
    model: nn.Module,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    brightness_deltas: Sequence[float],
    contrast_scales: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
    cache_path = _cache_path(args.output_dir, split, cache_key)
    if cache_path.is_file() and not bool(args.refresh_cache):
        payload = np.load(cache_path, allow_pickle=True)
        labels = payload["labels"].astype(np.int64)
        return (
            payload["features"].astype(np.float32),
            labels,
            [str(value) for value in payload["paths"].tolist()],
            (
                payload["sample_index"].astype(np.int64)
                if "sample_index" in payload
                else np.arange(labels.shape[0], dtype=np.int64)
            ),
            (
                [str(value) for value in payload["source_stem"].tolist()]
                if "source_stem" in payload
                else [Path(str(value)).stem for value in payload["paths"].tolist()]
            ),
        )
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        dataset = OrderedFolderDataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            max_samples_per_split=int(args.max_samples_per_split),
            max_samples_per_class=int(args.max_samples_per_class),
        )
    else:
        dataset = OrderedYoloObjectDataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            max_samples_per_split=int(args.max_samples_per_split),
            max_samples_per_class=int(args.max_samples_per_class),
        )
    if len(dataset) <= 0:
        raise RuntimeError(f"Split {split!r} has no images.")
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )
    features, labels, paths, sample_index, source_stem = _extract_features(
        model=model,
        loader=loader,
        device=device,
        amp=bool(args.amp),
        mean=mean,
        std=std,
        horizontal_flip=bool(args.horizontal_flip_tta),
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
        description=f"{model_name} {split}",
    )
    np.savez_compressed(
        cache_path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        paths=np.asarray(paths, dtype=str),
        sample_index=sample_index.astype(np.int64),
        source_stem=np.asarray(source_stem, dtype=str),
        classes=np.asarray(data_spec.class_names, dtype=str),
    )
    return features, labels, paths, sample_index, source_stem


def _metrics(y_true: Sequence[int], probs: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    y_pred = np.asarray(probs, dtype=np.float32).argmax(axis=1)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_index": int(index),
                "class_name": str(class_names[index]),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in labels
        ],
    }


def _objective(metrics: Mapping[str, object], *, focus_class: int, focus_class_weight: float) -> float:
    per_class = metrics.get("per_class", [])
    focus_f1 = 0.0
    if isinstance(per_class, list) and 0 <= int(focus_class) < len(per_class):
        focus_f1 = float(per_class[int(focus_class)].get("f1", 0.0))
    return float(metrics.get("macro_f1", 0.0)) + float(focus_class_weight) * focus_f1


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    shifted = values / temp
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1e-12, None)


def _knn_probs(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    num_classes: int,
    k: int,
    temperature: float,
    chunk_size: int = 512,
) -> np.ndarray:
    train_features = _normalize_features(train_features)
    query_features = _normalize_features(query_features)
    k = max(1, min(int(k), int(train_features.shape[0])))
    output = np.zeros((query_features.shape[0], num_classes), dtype=np.float32)
    row_offset = 0
    for start in range(0, query_features.shape[0], chunk_size):
        end = min(query_features.shape[0], start + chunk_size)
        sims = query_features[start:end] @ train_features.T
        if k < sims.shape[1]:
            top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            top_sims = np.take_along_axis(sims, top_idx, axis=1)
            order = np.argsort(-top_sims, axis=1)
            top_idx = np.take_along_axis(top_idx, order, axis=1)
            top_sims = np.take_along_axis(top_sims, order, axis=1)
        else:
            top_idx = np.argsort(-sims, axis=1)
            top_sims = np.take_along_axis(sims, top_idx, axis=1)
        weights = _softmax(top_sims.astype(np.float32), temperature=temperature)
        batch_probs = np.zeros((end - start, num_classes), dtype=np.float32)
        row_indices = np.arange(end - start)[:, None]
        np.add.at(batch_probs, (row_indices, train_labels[top_idx]), weights)
        batch_probs = batch_probs / np.clip(batch_probs.sum(axis=1, keepdims=True), 1e-12, None)
        output[row_offset : row_offset + batch_probs.shape[0]] = batch_probs
        row_offset += batch_probs.shape[0]
    return output


def _prototype_probs(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    num_classes: int,
    temperature: float,
) -> np.ndarray:
    train_features = _normalize_features(train_features)
    query_features = _normalize_features(query_features)
    prototypes = np.zeros((num_classes, train_features.shape[1]), dtype=np.float32)
    for class_index in range(num_classes):
        mask = train_labels == class_index
        if not np.any(mask):
            continue
        prototypes[class_index] = train_features[mask].mean(axis=0)
    prototypes = _normalize_features(prototypes)
    sims = query_features @ prototypes.T
    return _softmax(sims.astype(np.float32), temperature=temperature).astype(np.float32)


def _fit_logreg(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    c_value: float,
    class_weight: Optional[str],
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        max_iter=1000,
        n_jobs=1,
        multi_class="auto",
        solver="lbfgs",
        class_weight=class_weight,
    )
    model.fit(_normalize_features(train_features), train_labels)
    return model


def _evaluate_candidates(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    class_names: Sequence[str],
    k_values: Sequence[int],
    temperatures: Sequence[float],
    logreg_c_values: Sequence[float],
    disable_logreg: bool,
    focus_class: int,
    focus_class_weight: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    candidates: List[Dict[str, object]] = []
    num_classes = len(class_names)
    for k in k_values:
        for temperature in temperatures:
            probs = _knn_probs(
                train_features=train_features,
                train_labels=train_labels,
                query_features=val_features,
                num_classes=num_classes,
                k=int(k),
                temperature=float(temperature),
            )
            metrics = _metrics(val_labels, probs, class_names)
            candidates.append(
                {
                    "method": "knn",
                    "k": int(k),
                    "temperature": float(temperature),
                    "metrics": metrics,
                    "objective": _objective(
                        metrics,
                        focus_class=focus_class,
                        focus_class_weight=focus_class_weight,
                    ),
                }
            )
    for temperature in temperatures:
        probs = _prototype_probs(
            train_features=train_features,
            train_labels=train_labels,
            query_features=val_features,
            num_classes=num_classes,
            temperature=float(temperature),
        )
        metrics = _metrics(val_labels, probs, class_names)
        candidates.append(
            {
                "method": "prototype",
                "temperature": float(temperature),
                "metrics": metrics,
                "objective": _objective(metrics, focus_class=focus_class, focus_class_weight=focus_class_weight),
            }
        )
    if not disable_logreg and len(set(int(value) for value in train_labels.tolist())) >= 2:
        for c_value in logreg_c_values:
            for class_weight in (None, "balanced"):
                model = _fit_logreg(
                    train_features=train_features,
                    train_labels=train_labels,
                    c_value=float(c_value),
                    class_weight=class_weight,
                )
                probs = model.predict_proba(_normalize_features(val_features)).astype(np.float32)
                metrics = _metrics(val_labels, probs, class_names)
                candidates.append(
                    {
                        "method": "logreg",
                        "c": float(c_value),
                        "class_weight": class_weight or "none",
                        "metrics": metrics,
                        "objective": _objective(
                            metrics,
                            focus_class=focus_class,
                            focus_class_weight=focus_class_weight,
                        ),
                    }
                )
    candidates = sorted(
        candidates,
        key=lambda item: (
            float(item["objective"]),
            float(item["metrics"].get("macro_f1", 0.0)),
        ),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("No retrieval candidates were evaluated.")
    return candidates[0], candidates


def _predict_with_candidate(
    candidate: Mapping[str, object],
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    method = str(candidate.get("method"))
    if method == "knn":
        return _knn_probs(
            train_features=train_features,
            train_labels=train_labels,
            query_features=query_features,
            num_classes=num_classes,
            k=int(candidate.get("k", 1)),
            temperature=float(candidate.get("temperature", 0.1)),
        )
    if method == "prototype":
        return _prototype_probs(
            train_features=train_features,
            train_labels=train_labels,
            query_features=query_features,
            num_classes=num_classes,
            temperature=float(candidate.get("temperature", 0.1)),
        )
    if method == "logreg":
        model = _fit_logreg(
            train_features=train_features,
            train_labels=train_labels,
            c_value=float(candidate.get("c", 1.0)),
            class_weight=(None if str(candidate.get("class_weight", "none")) == "none" else "balanced"),
        )
        return model.predict_proba(_normalize_features(query_features)).astype(np.float32)
    raise ValueError(f"Unsupported candidate method: {method!r}")


def _write_predictions(
    *,
    output_dir: Path,
    split: str,
    paths: Sequence[str],
    labels: Sequence[int],
    probs: np.ndarray,
    class_names: Sequence[str],
    metrics: Mapping[str, object],
    model_name: str,
    selected_candidate: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"predictions_{split}.csv"
    pred = probs.argmax(axis=1)
    fieldnames = [
        "path",
        "y_true",
        "y_pred",
        "true_name",
        "pred_name",
        "confidence",
        *[f"prob_{index}" for index in range(len(class_names))],
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, true_index, pred_index, row_probs in zip(paths, labels, pred.tolist(), probs.tolist()):
            row: Dict[str, object] = {
                "path": str(path),
                "y_true": int(true_index),
                "y_pred": int(pred_index),
                "true_name": str(class_names[int(true_index)]),
                "pred_name": str(class_names[int(pred_index)]),
                "confidence": float(max(row_probs)),
            }
            for index, value in enumerate(row_probs):
                row[f"prob_{index}"] = float(value)
            writer.writerow(row)
    metrics_payload = {
        "split": split,
        "model": model_name,
        "classes": list(class_names),
        "selected_candidate": selected_candidate,
        **dict(metrics),
    }
    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(value) for value in data_spec.class_names]
    device = _resolve_device(str(args.device))
    checkpoint_payload: Optional[Mapping[str, object]] = None
    if args.checkpoint is not None:
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
        if not isinstance(checkpoint_payload, Mapping):
            raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint_payload)!r}")
    model_name = _infer_model_name(args, checkpoint_payload)
    cache_key = model_name
    if args.checkpoint is not None:
        cache_key = f"{model_name}__{args.checkpoint.stem}"
    transform, mean, std, size = _build_transform(model_name)
    brightness_deltas = _parse_float_list(args.brightness_deltas)
    contrast_scales = _parse_float_list(args.contrast_scales)
    k_values = _parse_int_list(args.knn_k)
    temperatures = _parse_float_list(args.temperatures)
    logreg_c_values = _parse_float_list(args.logreg_c)
    if not temperatures:
        temperatures = [0.1]
    if not k_values:
        k_values = [5]
    if not logreg_c_values:
        logreg_c_values = [1.0]

    model = _build_feature_model(
        model_name,
        pretrained=not bool(args.no_pretrained),
        checkpoint=checkpoint_payload,
    )
    model.to(device)

    print(
        {
            "embedding_retrieval": {
                "model": model_name,
                "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                "pretrained": (not bool(args.no_pretrained)) and args.checkpoint is None,
                "data": str(args.data),
                "classes": class_names,
                "device": str(device),
                "input_size": list(size),
                "mean": list(mean),
                "std": list(std),
                "horizontal_flip_tta": bool(args.horizontal_flip_tta),
                "brightness_deltas": brightness_deltas,
                "contrast_scales": contrast_scales,
                "max_samples_per_split": int(args.max_samples_per_split),
                "max_samples_per_class": int(args.max_samples_per_class),
            }
        },
        flush=True,
    )

    train_features, train_labels, train_paths, train_sample_index, train_source_stem = _load_or_extract_split(
        args=args,
        model_name=model_name,
        cache_key=cache_key,
        data_spec=data_spec,
        split="train",
        transform=transform,
        model=model,
        device=device,
        mean=mean,
        std=std,
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
    )
    val_features, val_labels, val_paths, val_sample_index, val_source_stem = _load_or_extract_split(
        args=args,
        model_name=model_name,
        cache_key=cache_key,
        data_spec=data_spec,
        split="val",
        transform=transform,
        model=model,
        device=device,
        mean=mean,
        std=std,
        brightness_deltas=brightness_deltas,
        contrast_scales=contrast_scales,
    )

    train_features = _normalize_features(train_features)
    val_features = _normalize_features(val_features)
    best_candidate, candidates = _evaluate_candidates(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        class_names=class_names,
        k_values=k_values,
        temperatures=temperatures,
        logreg_c_values=logreg_c_values,
        disable_logreg=bool(args.disable_logreg),
        focus_class=int(args.focus_class),
        focus_class_weight=float(args.focus_class_weight),
    )
    val_probs = _predict_with_candidate(
        best_candidate,
        train_features=train_features,
        train_labels=train_labels,
        query_features=val_features,
        num_classes=len(class_names),
    )
    val_metrics = _metrics(val_labels, val_probs, class_names)
    _write_predictions(
        output_dir=args.output_dir,
        split="val",
        paths=val_paths,
        labels=val_labels,
        probs=val_probs,
        class_names=class_names,
        metrics=val_metrics,
        model_name=model_name,
        selected_candidate=best_candidate,
    )

    test_metrics: Optional[Dict[str, object]] = None
    if not bool(args.skip_test):
        test_features, test_labels, test_paths, test_sample_index, test_source_stem = _load_or_extract_split(
            args=args,
            model_name=model_name,
            cache_key=cache_key,
            data_spec=data_spec,
            split="test",
            transform=transform,
            model=model,
            device=device,
            mean=mean,
            std=std,
            brightness_deltas=brightness_deltas,
            contrast_scales=contrast_scales,
        )
        test_features = _normalize_features(test_features)
        test_probs = _predict_with_candidate(
            best_candidate,
            train_features=train_features,
            train_labels=train_labels,
            query_features=test_features,
            num_classes=len(class_names),
        )
        test_metrics = _metrics(test_labels, test_probs, class_names)
        _write_predictions(
            output_dir=args.output_dir,
            split="test",
            paths=test_paths,
            labels=test_labels,
            probs=test_probs,
            class_names=class_names,
            metrics=test_metrics,
            model_name=model_name,
            selected_candidate=best_candidate,
        )

    summary = {
        "model": str(model_name),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "data": str(args.data),
        "pretrained": (not bool(args.no_pretrained)) and args.checkpoint is None,
        "classes": class_names,
        "feature_shapes": {
            "train": list(train_features.shape),
            "val": list(val_features.shape),
        },
        "feature_metadata": {
            "train_sample_index_min": int(train_sample_index.min()) if train_sample_index.size else 0,
            "train_sample_index_max": int(train_sample_index.max()) if train_sample_index.size else -1,
            "val_sample_index_min": int(val_sample_index.min()) if val_sample_index.size else 0,
            "val_sample_index_max": int(val_sample_index.max()) if val_sample_index.size else -1,
            "train_unique_source_stems": int(len(set(str(value) for value in train_source_stem))),
            "val_unique_source_stems": int(len(set(str(value) for value in val_source_stem))),
        },
        "selected_by": {
            "split": "val",
            "objective": "macro_f1 + focus_class_weight * focus_class_f1",
            "focus_class": int(args.focus_class),
            "focus_class_weight": float(args.focus_class_weight),
        },
        "best_candidate": best_candidate,
        "top_candidates": candidates[:20],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    focus_index = int(args.focus_class)
    val_focus = val_metrics["per_class"][focus_index]["f1"] if 0 <= focus_index < len(class_names) else math.nan
    test_focus = (
        test_metrics["per_class"][focus_index]["f1"]
        if test_metrics is not None and 0 <= focus_index < len(class_names)
        else math.nan
    )
    print(
        {
            "summary": str(args.output_dir / "summary.json"),
            "selected": {
                key: value
                for key, value in best_candidate.items()
                if key not in {"metrics"}
            },
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_focus_f1": float(val_focus),
            "test_macro_f1": None if test_metrics is None else float(test_metrics["macro_f1"]),
            "test_focus_f1": None if test_metrics is None else float(test_focus),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
