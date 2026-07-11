from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.data.dataset import build_train_collate_fn
from trkh.evaluation.evaluate import (
    _build_prediction_records,
    _checkpoint_data_path_mismatch,
    _dataset_sample_metadata,
    _dataset_sample_paths,
)
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import extract_detection_from_model_output
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _class1_f1,
    _select_bbox_token_prior,
)
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
)


def parse_margin_grid(text: str) -> List[float]:
    value = str(text or "").strip()
    if not value:
        return [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
    margins: List[float] = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        margin = float(part)
        if margin < 0.0 or margin > 1.0:
            raise ValueError("--margins chi nhan gia tri trong [0,1].")
        margins.append(round(margin, 6))
    if not margins:
        raise ValueError("--margins rong.")
    return sorted(set(margins))


def margin_key(margin: float) -> str:
    text = f"{float(margin):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return "m" + text.replace("-", "neg").replace(".", "p")


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _class_metric_fields(metrics: Mapping[str, object]) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, list):
        return fields
    for item in per_class:
        if not isinstance(item, dict):
            continue
        class_index = int(item.get("class_index", -1))
        if class_index < 0:
            continue
        fields[f"class{class_index}_precision"] = float(item.get("precision", 0.0) or 0.0)
        fields[f"class{class_index}_recall"] = float(item.get("recall", 0.0) or 0.0)
        fields[f"class{class_index}_f1"] = float(item.get("f1", 0.0) or 0.0)
    return fields


def build_sweep_row(
    *,
    margin: float,
    metrics: Mapping[str, object],
    sample_count: int,
    baseline_macro_f1: Optional[float] = None,
    baseline_class1_f1: Optional[float] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "crop_margin_ratio": float(margin),
        "samples": int(sample_count),
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
        "class1_f1": _class1_f1(dict(metrics), 1),
    }
    if baseline_macro_f1 is not None:
        row["delta_macro_f1_vs_baseline"] = float(row["macro_f1"]) - float(baseline_macro_f1)
    if baseline_class1_f1 is not None:
        row["delta_class1_f1_vs_baseline"] = float(row["class1_f1"]) - float(baseline_class1_f1)
    row.update(_class_metric_fields(metrics))
    return row


def summarize_changed_predictions(
    *,
    baseline_records: Sequence[Mapping[str, object]],
    margin_records: Sequence[Mapping[str, object]],
    margin: float,
    class_names: Sequence[str],
) -> Dict[str, object]:
    rows, counts = build_changed_prediction_rows(
        baseline_records=baseline_records,
        margin_records=margin_records,
        margin=margin,
        class_names=class_names,
    )
    transitions: Dict[str, int] = {}
    for row in rows:
        transition = str(row.get("transition", ""))
        transitions[transition] = transitions.get(transition, 0) + 1
    return {
        "crop_margin_ratio": float(margin),
        "changed": len(rows),
        **counts,
        "transitions": transitions,
    }


def build_changed_prediction_rows(
    *,
    baseline_records: Sequence[Mapping[str, object]],
    margin_records: Sequence[Mapping[str, object]],
    margin: float,
    class_names: Sequence[str],
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    count = min(len(baseline_records), len(margin_records))
    rows: List[Dict[str, object]] = []
    counts = {"corrections": 0, "harms": 0, "neutral": 0}
    for index in range(count):
        base = baseline_records[index]
        current = margin_records[index]
        base_pred = int(base.get("prediction_index", -1))
        current_pred = int(current.get("prediction_index", -1))
        target = int(base.get("target_index", current.get("target_index", -1)))
        if base_pred == current_pred:
            continue
        base_correct = int(base_pred == target)
        current_correct = int(current_pred == target)
        if not base_correct and current_correct:
            change_type = "correction"
            counts["corrections"] += 1
        elif base_correct and not current_correct:
            change_type = "harm"
            counts["harms"] += 1
        else:
            change_type = "neutral"
            counts["neutral"] += 1
        target_name = str(class_names[target]) if 0 <= target < len(class_names) else ""
        base_name = str(class_names[base_pred]) if 0 <= base_pred < len(class_names) else ""
        current_name = str(class_names[current_pred]) if 0 <= current_pred < len(class_names) else ""
        row: Dict[str, object] = {
            "sample_index": int(base.get("sample_index", index)),
            "image_path": str(base.get("image_path", current.get("image_path", ""))),
            "target_index": target,
            "target_name": target_name,
            "baseline_prediction_index": base_pred,
            "baseline_prediction_name": base_name,
            "margin_prediction_index": current_pred,
            "margin_prediction_name": current_name,
            "baseline_correct": base_correct,
            "margin_correct": current_correct,
            "change_type": change_type,
            "transition": f"{target}:{base_pred}->{current_pred}",
            "crop_margin_ratio": float(margin),
            "baseline_confidence": base.get("confidence", ""),
            "margin_confidence": current.get("confidence", ""),
        }
        for class_index, class_name in enumerate(class_names):
            base_key = f"prob_{class_index}_{class_name}"
            if base_key in base:
                row[f"baseline_prob_{class_index}"] = base.get(base_key, "")
            if base_key in current:
                row[f"margin_prob_{class_index}"] = current.get(base_key, "")
        rows.append(row)
    return rows, counts


def _forward_classification_logits(
    *,
    model: nn.Module,
    images: torch.Tensor,
    image_valid_mask: Optional[torch.Tensor],
    bbox_metadata: Optional[torch.Tensor],
    bbox_token_prior: Optional[torch.Tensor],
    amp: bool,
    device: torch.device,
) -> torch.Tensor:
    with autocast_context(device, amp):
        if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            output = model.forward_heads(features)
        elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            output = classification_logits_from_features(model, features)
        else:
            output = model(images)
        logits, _, _ = extract_detection_from_model_output(output)
    return logits.float()


def _evaluate_margin(
    *,
    args: argparse.Namespace,
    checkpoint: Dict[str, object],
    model: nn.Module,
    data_spec,
    class_names: Sequence[str],
    margin: float,
    device: torch.device,
    eval_transform,
) -> Dict[str, object]:
    dataset = _build_classification_dataset(
        data_spec=data_spec,
        split=args.split,
        transform=eval_transform,
        checkpoint=checkpoint,
        full_image_detection=args.full_image_detection,
        crop_to_primary_object_override=args.crop_to_primary_object,
        disable_classification_object_crops=args.disable_classification_object_crops,
        crop_margin_ratio_override=float(margin),
    )
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"crop_margin_{args.split}_{margin_key(margin)}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    sample_paths = _dataset_sample_paths(dataset)
    sample_metadata = _dataset_sample_metadata(dataset)
    total_batches = len(loader)
    iterator: Iterable = loader
    if int(args.max_batches) > 0:
        total_batches = min(total_batches, int(args.max_batches))
        iterator = islice(loader, total_batches)

    probability_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    with torch.inference_mode():
        with tqdm(
            iterator,
            desc=f"crop margin {margin:.4f}",
            total=total_batches,
            dynamic_ncols=True,
        ) as pbar:
            for images, labels, targets in pbar:
                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                metric_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
                targets = targets if isinstance(targets, dict) else {}

                bbox = targets.get("bbox")
                crop_bbox = targets.get("crop_bbox")
                image_mask = targets.get("image_mask")
                bbox = bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(bbox) else None
                crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(crop_bbox) else None
                image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True) if torch.is_tensor(image_mask) else None
                bbox_token_prior = _select_bbox_token_prior(
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    source=args.bbox_token_prior_source,
                )
                logits = _forward_classification_logits(
                    model=model,
                    images=images,
                    image_valid_mask=image_mask,
                    bbox_metadata=bbox,
                    bbox_token_prior=bbox_token_prior,
                    amp=args.amp,
                    device=device,
                )
                probability_batches.append(torch.softmax(logits, dim=1).detach().cpu())
                target_batches.append(metric_labels.detach().cpu().to(dtype=torch.long))

    probabilities = torch.cat(probability_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    predictions = probabilities.argmax(dim=1)
    if len(sample_paths) > int(targets.numel()):
        sample_paths = sample_paths[: int(targets.numel())]
    if len(sample_metadata) > int(targets.numel()):
        sample_metadata = sample_metadata[: int(targets.numel())]
    metrics = build_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_names=class_names,
    )
    records = _build_prediction_records(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_names=class_names,
        sample_paths=sample_paths,
        sample_metadata=sample_metadata,
    )
    return {
        "margin": float(margin),
        "dataset_report": dataset.quality_report(),
        "dataloader": dataloader_summary,
        "metrics": metrics,
        "records": records,
        "samples": int(targets.numel()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether YOLO crop-margin context changes a classification checkpoint "
            "without retraining or modifying the raw dataset."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="YOLO-format data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--margins", default="0,0.02,0.05,0.08,0.12,0.18,0.25")
    parser.add_argument("--baseline-margin", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--full-image-detection", action="store_true", default=False)
    parser.add_argument("--crop-to-primary-object", action="store_true", default=False)
    parser.add_argument("--disable-classification-object-crops", action="store_true", default=False)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument(
        "--save-all-predictions",
        action="store_true",
        default=False,
        help="Save per-sample predictions for every margin, not only baseline and best rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test":
        raise ValueError("Khong dung tool nay tren test khi dang chon/tune huong. Hay dung --split val.")
    set_seed(args.seed, deterministic=False)
    output_dir = ensure_dir(args.output_dir)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    augmentation_config = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    baseline_margin = (
        float(args.baseline_margin)
        if args.baseline_margin is not None
        else float(augmentation_config.get("crop_margin_ratio", 0.05) or 0.05)
    )
    margins = parse_margin_grid(args.margins)
    if round(baseline_margin, 6) not in {round(item, 6) for item in margins}:
        margins = sorted(set(margins + [round(baseline_margin, 6)]))

    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    mismatch = _checkpoint_data_path_mismatch(checkpoint, data_spec.data_yaml)
    if mismatch is not None:
        print({"warning": "data_path_differs_from_checkpoint", **mismatch}, flush=True)
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if len(class_names) != data_spec.num_classes:
        raise ValueError("So lop trong checkpoint khong khop data.yaml.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        num_classes=len(class_names),
        override_image_size=args.override_image_size,
    )
    model.to(device)
    model.eval()
    image_size = int(
        args.override_image_size
        or checkpoint.get("model_config", {}).get("image_size", 224)
    )
    eval_transform = _build_eval_transform_from_checkpoint(
        checkpoint,
        image_size=image_size,
    )

    result_by_key: Dict[str, Dict[str, object]] = {}
    for margin in margins:
        result = _evaluate_margin(
            args=args,
            checkpoint=checkpoint,
            model=model,
            data_spec=data_spec,
            class_names=class_names,
            margin=margin,
            device=device,
            eval_transform=eval_transform,
        )
        key = margin_key(margin)
        result_by_key[key] = result
        records = result["records"]
        if args.save_all_predictions or round(margin, 6) == round(baseline_margin, 6):
            _write_rows_csv(output_dir / f"predictions_{key}.csv", records)  # type: ignore[arg-type]

    baseline_key = margin_key(baseline_margin)
    if baseline_key not in result_by_key:
        raise RuntimeError("Khong tim thay baseline margin sau khi evaluate.")
    baseline_result = result_by_key[baseline_key]
    baseline_metrics = baseline_result["metrics"]
    baseline_macro = float(dict(baseline_metrics).get("macro_f1", 0.0) or 0.0)
    baseline_class1 = _class1_f1(dict(baseline_metrics), 1)
    sweep_rows: List[Dict[str, object]] = []
    changed_summaries: List[Dict[str, object]] = []
    prediction_exports: Dict[str, str] = {baseline_key: f"predictions_{baseline_key}.csv"}

    for margin in margins:
        key = margin_key(margin)
        result = result_by_key[key]
        metrics = dict(result["metrics"])
        row = build_sweep_row(
            margin=margin,
            metrics=metrics,
            sample_count=int(result["samples"]),
            baseline_macro_f1=baseline_macro,
            baseline_class1_f1=baseline_class1,
        )
        sweep_rows.append(row)
        if round(margin, 6) == round(baseline_margin, 6):
            continue
        changed_rows, changed_counts = build_changed_prediction_rows(
            baseline_records=baseline_result["records"],  # type: ignore[arg-type]
            margin_records=result["records"],  # type: ignore[arg-type]
            margin=margin,
            class_names=class_names,
        )
        _write_rows_csv(output_dir / f"changed_vs_{baseline_key}_{key}.csv", changed_rows)
        transitions: Dict[str, int] = {}
        for changed_row in changed_rows:
            transition = str(changed_row.get("transition", ""))
            transitions[transition] = transitions.get(transition, 0) + 1
        changed_summaries.append(
            {
                "crop_margin_ratio": float(margin),
                "changed": int(len(changed_rows)),
                **changed_counts,
                "transitions": transitions,
            }
        )

    best_macro = max(sweep_rows, key=lambda item: float(item.get("macro_f1", 0.0)))
    best_class1 = max(sweep_rows, key=lambda item: float(item.get("class1_f1", 0.0)))
    for label, row in (("best_macro", best_macro), ("best_class1", best_class1)):
        key = margin_key(float(row["crop_margin_ratio"]))
        filename = f"predictions_{label}_{key}.csv"
        _write_rows_csv(output_dir / filename, result_by_key[key]["records"])  # type: ignore[arg-type]
        prediction_exports[label] = filename

    _write_rows_csv(output_dir / "crop_margin_sweep.csv", sweep_rows)
    metrics_by_margin = {
        key: dict(value["metrics"])
        for key, value in result_by_key.items()
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_spec.data_yaml),
        "split": args.split,
        "image_size": image_size,
        "baseline_margin": float(baseline_margin),
        "margins": [float(item) for item in margins],
        "samples": int(baseline_result["samples"]),
        "bbox_token_prior_source": args.bbox_token_prior_source,
        "sweep": sweep_rows,
        "changed_vs_baseline": changed_summaries,
        "best_macro": best_macro,
        "best_class1": best_class1,
        "metrics_by_margin": metrics_by_margin,
        "dataset_reports": {
            key: value["dataset_report"]
            for key, value in result_by_key.items()
        },
        "prediction_exports": prediction_exports,
        "leakage_guard": (
            "Diagnostic is intended for validation only. Do not use this tool to select "
            "crop margin from the test split."
        ),
    }
    json_dump(output_dir / "summary.json", to_serializable(summary))
    print(
        {
            "output_dir": str(output_dir),
            "samples": int(baseline_result["samples"]),
            "baseline": build_sweep_row(
                margin=baseline_margin,
                metrics=dict(baseline_metrics),
                sample_count=int(baseline_result["samples"]),
            ),
            "best_macro": best_macro,
            "best_class1": best_class1,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
