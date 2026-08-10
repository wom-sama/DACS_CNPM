from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from trkh.core.config import load_data_spec
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    build_eval_transform,
)
from trkh.models.model import ForegroundSurfaceStatisticFusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe train-only/val foreground surface statistics with lightweight "
            "classifiers. This is diagnostic; it never reads test split."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.05)
    return parser.parse_args()


def _limited_dataset(dataset: Dataset, max_samples: int, seed: int):
    if int(max_samples) <= 0 or int(max_samples) >= len(dataset):
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return Subset(dataset, indices[: int(max_samples)].tolist())


def _sample_record(sample, sample_index: int) -> Dict[str, object]:
    image_path = getattr(sample, "image_path", "")
    label_path = getattr(sample, "label_path", "")
    label = getattr(sample, "label", getattr(sample, "primary_label", -1))
    record: Dict[str, object] = {
        "sample_index": int(sample_index),
        "image_path": str(image_path) if image_path is not None else "",
        "label": int(label),
    }
    if image_path is not None:
        record["source_stem"] = Path(image_path).stem
    if label_path is not None:
        record["label_path"] = str(label_path)
        record.setdefault("source_stem", Path(label_path).stem)
    if hasattr(sample, "primary_label"):
        record["primary_label"] = int(getattr(sample, "primary_label"))
    if hasattr(sample, "primary_object_index"):
        record["object_index"] = int(getattr(sample, "primary_object_index"))
    objects = getattr(sample, "objects", None)
    object_index = record.get("object_index")
    if objects is not None and object_index is not None:
        for obj in objects:
            if int(getattr(obj, "object_index", -1)) != int(object_index):
                continue
            bbox = getattr(obj, "bbox", None)
            if bbox is not None:
                for axis_index, value in enumerate(bbox):
                    record[f"bbox_{axis_index}"] = float(value)
            break
    yolo_object = getattr(sample, "yolo_object", None)
    if yolo_object is not None:
        record["object_index"] = int(getattr(yolo_object, "object_index", record.get("object_index", -1)))
        bbox = getattr(yolo_object, "bbox", None)
        if bbox is not None:
            for axis_index, value in enumerate(bbox):
                record[f"bbox_{axis_index}"] = float(value)
    return record


def _dataset_records(dataset) -> List[Dict[str, object]]:
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    if isinstance(dataset, Subset):
        indices = [int(index) for index in dataset.indices]
        samples = [source.samples[index] for index in indices]
    else:
        samples = getattr(source, "samples", [])
        indices = list(range(len(samples)))
    if samples:
        return [_sample_record(sample, sample_index=index) for sample, index in zip(samples, indices)]
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    sample_paths = [str(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    labels_fn = getattr(dataset, "labels", None)
    labels = [int(label) for label in labels_fn()] if callable(labels_fn) else []
    return [
        {
            "sample_index": int(index),
            "image_path": sample_paths[index] if index < len(sample_paths) else "",
            "label": int(labels[index]) if index < len(labels) else -1,
        }
        for index in range(len(dataset))
    ]


def _collate_classification(batch):
    images: List[torch.Tensor] = []
    labels: List[int] = []
    for item in batch:
        if len(item) == 3:
            image, label, _metadata = item
        elif len(item) == 2:
            image, label = item
        else:
            raise ValueError("Classification batch item must have 2 or 3 fields.")
        images.append(image)
        labels.append(int(label))
    return torch.stack(images, dim=0), torch.as_tensor(labels, dtype=torch.long)


def _build_probe_dataset(
    *,
    data_spec,
    split: str,
    transform,
    crop_margin_ratio: float,
):
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        return ClassificationFolderDataset.from_data_spec(
            data_spec,
            split,
            transform=transform,
            class_aware_augmentation=False,
        )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=transform,
        crop_margin_ratio=float(crop_margin_ratio),
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )


@torch.no_grad()
def _extract_split_stats(
    *,
    dataset,
    extractor: ForegroundSurfaceStatisticFusion,
    batch_size: int,
    workers: int,
    split_name: str,
    progress_interval: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    records = _dataset_records(dataset)
    start = time.time()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        stats = extractor.extract_stats(images)
        features.append(stats.cpu().numpy().astype(np.float32, copy=False))
        labels.append(targets.cpu().numpy().astype(np.int64, copy=False))
        if progress_interval > 0 and (batch_index % int(progress_interval) == 0):
            print(
                {
                    "split": split_name,
                    "batches": batch_index,
                    "samples": int(sum(part.shape[0] for part in labels)),
                    "seconds": round(time.time() - start, 2),
                },
                flush=True,
            )
    if not features:
        return (
            np.zeros((0, extractor.stats_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            records,
        )
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), records


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Sequence[str],
) -> Dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "per_class": [
            {
                "class_index": int(index),
                "class_name": str(class_names[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def _class1_threshold_probe(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    if probabilities.ndim != 2 or probabilities.shape[1] <= 1 or y_true.size == 0:
        return {}
    y_binary = (y_true == 1).astype(np.int64)
    scores = probabilities[:, 1]
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.linspace(0.02, 0.98, 97):
        predicted = scores >= float(threshold)
        tp = float(((predicted == 1) & (y_binary == 1)).sum())
        fp = float(((predicted == 1) & (y_binary == 0)).sum())
        fn = float(((predicted == 0) & (y_binary == 1)).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def _write_predictions_csv(
    path: Path,
    *,
    sample_records: Sequence[Mapping[str, object]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_names: Sequence[str],
) -> None:
    metadata_fields = [
        "sample_index",
        "image_path",
        "source_stem",
        "label_path",
        "object_index",
        "primary_label",
        "bbox_0",
        "bbox_1",
        "bbox_2",
        "bbox_3",
    ]
    fieldnames = [*metadata_fields, "label", "prediction"]
    fieldnames.extend([f"prob_{index}" for index in range(len(class_names))])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (label, prediction) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            record = dict(sample_records[row_index]) if row_index < len(sample_records) else {}
            row = {
                key: record.get(key, "")
                for key in metadata_fields
            }
            row.update(
                {
                    "label": int(label),
                    "prediction": int(prediction),
                }
            )
            if probabilities is not None and probabilities.ndim == 2:
                for class_index in range(len(class_names)):
                    row[f"prob_{class_index}"] = float(probabilities[row_index, class_index])
            writer.writerow(row)


def _fit_and_score(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_records: Sequence[Mapping[str, object]],
    class_names: Sequence[str],
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    classifiers = {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                multi_class="auto",
                random_state=int(seed),
            ),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        ),
    }
    results: Dict[str, object] = {}
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_val)
        probabilities = classifier.predict_proba(x_val) if hasattr(classifier, "predict_proba") else None
        metrics = _per_class_metrics(y_val, prediction, class_names=class_names)
        if probabilities is not None:
            metrics["class1_threshold_probe"] = _class1_threshold_probe(y_val, probabilities)
            _write_predictions_csv(
                output_dir / f"{name}_val_predictions.csv",
                sample_records=val_records,
                y_true=y_val,
                y_pred=prediction,
                probabilities=probabilities,
                class_names=class_names,
            )
        results[name] = metrics
    return results


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_spec = load_data_spec(args.data, class_name_mode="raw", expected_num_classes=5)
    transform = build_eval_transform(image_size=int(args.image_size), resize_mode="pad")
    train_dataset = _build_probe_dataset(
        data_spec=data_spec,
        split="train",
        transform=transform,
        crop_margin_ratio=float(args.crop_margin_ratio),
    )
    val_dataset = _build_probe_dataset(
        data_spec=data_spec,
        split="val",
        transform=transform,
        crop_margin_ratio=float(args.crop_margin_ratio),
    )
    train_dataset = _limited_dataset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = _limited_dataset(val_dataset, args.max_val_samples, args.seed + 1)
    train_records = _dataset_records(train_dataset)
    val_records = _dataset_records(val_dataset)
    extractor = ForegroundSurfaceStatisticFusion(num_classes=data_spec.num_classes).eval()

    start = time.time()
    x_train, y_train, _ = _extract_split_stats(
        dataset=train_dataset,
        extractor=extractor,
        batch_size=args.batch_size,
        workers=args.workers,
        split_name="train",
        progress_interval=args.progress_interval,
    )
    x_val, y_val, _ = _extract_split_stats(
        dataset=val_dataset,
        extractor=extractor,
        batch_size=args.batch_size,
        workers=args.workers,
        split_name="val",
        progress_interval=args.progress_interval,
    )
    np.savez_compressed(
        output_dir / "foreground_surface_stats_train_val.npz",
        x_train=x_train,
        y_train=y_train,
        train_paths=np.asarray([str(row.get("image_path", "")) for row in train_records]),
        train_sample_indices=np.asarray([int(row.get("sample_index", index)) for index, row in enumerate(train_records)]),
        x_val=x_val,
        y_val=y_val,
        val_paths=np.asarray([str(row.get("image_path", "")) for row in val_records]),
        val_sample_indices=np.asarray([int(row.get("sample_index", index)) for index, row in enumerate(val_records)]),
        class_names=np.asarray(data_spec.class_names),
    )
    results = _fit_and_score(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        val_records=val_records,
        class_names=data_spec.class_names,
        output_dir=output_dir,
        seed=args.seed,
    )
    report = {
        "data": str(args.data.resolve()),
        "data_format": str(data_spec.data_format),
        "image_size": int(args.image_size),
        "crop_margin_ratio": float(args.crop_margin_ratio),
        "stats_dim": int(extractor.stats_dim),
        "train_samples": int(y_train.size),
        "val_samples": int(y_val.size),
        "class_names": list(data_spec.class_names),
        "class_counts": {
            "train": np.bincount(y_train, minlength=data_spec.num_classes).astype(int).tolist(),
            "val": np.bincount(y_val, minlength=data_spec.num_classes).astype(int).tolist(),
        },
        "results": results,
        "seconds": float(time.time() - start),
        "note": (
            "Diagnostic only. Uses train split for fitting and val split for scoring; "
            "test split is not read."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
