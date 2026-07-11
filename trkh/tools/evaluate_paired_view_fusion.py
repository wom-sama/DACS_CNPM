from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from trkh.data.dataset import (
    ClassificationFolderDataset,
    MangoYOLOCropDataset,
    PairedViewTrainDataset,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.evaluate import (
    DETECTION_MODEL_TYPES,
    _build_prediction_records,
    _checkpoint_data_path_mismatch,
    _stack_image_masks_from_targets,
    resolve_classification_object_crops,
    resolve_crop_to_primary_object,
)
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    extract_detection_from_model_output,
    paired_view_fused_logits_from_features,
)


def parse_weight_grid(text: str) -> List[float]:
    value = str(text or "").strip()
    if not value:
        return [round(index / 10.0, 2) for index in range(11)]
    weights: List[float] = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        weight = float(part)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("--weights chi nhan gia tri trong [0,1].")
        weights.append(float(weight))
    if not weights:
        raise ValueError("--weights rong.")
    return sorted(set(round(weight, 6) for weight in weights))


def _checkpoint_augmentation_config(checkpoint: Dict[str, object]) -> Dict[str, object]:
    config = checkpoint.get("augmentation_config", {})
    return dict(config) if isinstance(config, dict) else {}


def _build_eval_transform_from_checkpoint(
    checkpoint: Dict[str, object],
    *,
    image_size: int,
):
    augmentation_config = _checkpoint_augmentation_config(checkpoint)
    return build_eval_transform(
        image_size=int(image_size),
        resize_mode=str(augmentation_config.get("resize_mode", "pad") or "pad"),
        illumination_normalization=bool(
            augmentation_config.get("illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            augmentation_config.get("illumination_normalization_strength", 0.0) or 0.0
        ),
        foreground_crop_mode=str(augmentation_config.get("foreground_crop_mode", "none") or "none"),
        foreground_crop_margin_ratio=float(
            augmentation_config.get("foreground_crop_margin_ratio", 0.08) or 0.08
        ),
        foreground_crop_min_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation_config.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation_config.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(
            augmentation_config.get("background_suppression_mode", "none") or "none"
        ),
        background_suppression_margin=float(
            augmentation_config.get("background_suppression_margin", 0.08) or 0.08
        ),
        background_suppression_blur_radius=float(
            augmentation_config.get("background_suppression_blur_radius", 7.0) or 7.0
        ),
        surface_detail_amplification_mode=str(
            augmentation_config.get("surface_detail_amplification_mode", "none") or "none"
        ),
        surface_detail_amplification_strength=float(
            augmentation_config.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation_config.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation_config.get("surface_detail_amplification_foreground_weight", 0.85)
            or 0.85
        ),
        eval_surface_detail_amplification=bool(
            augmentation_config.get("eval_surface_detail_amplification", False)
        ),
    )


def _build_classification_dataset(
    *,
    data_spec,
    split: str,
    transform,
    checkpoint: Dict[str, object],
    paired_yolo_data_spec=None,
    full_image_detection: bool = False,
    crop_to_primary_object_override: bool = False,
    disable_classification_object_crops: bool = False,
    crop_margin_ratio_override: Optional[float] = None,
):
    checkpoint_model_type = str(
        checkpoint.get("model_config", {}).get("model_type", "")
    ).strip().lower()
    detection_mode = checkpoint_model_type in DETECTION_MODEL_TYPES
    if detection_mode:
        raise ValueError("Tool nay chi danh gia classification checkpoint.")

    augmentation_config = _checkpoint_augmentation_config(checkpoint)
    if data_spec.data_format == "classification_folder":
        return ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split=split,
            transform=transform,
            class_aware_augmentation=False,
            paired_yolo_data_spec=paired_yolo_data_spec,
            classification_bbox_metadata=paired_yolo_data_spec is not None,
        )

    crop_to_primary_object = resolve_crop_to_primary_object(
        checkpoint,
        full_image_detection=full_image_detection,
        crop_to_primary_object=crop_to_primary_object_override,
    )
    classification_object_crops = resolve_classification_object_crops(
        checkpoint,
        disable_classification_object_crops=disable_classification_object_crops,
    )
    crop_margin_ratio = (
        float(crop_margin_ratio_override)
        if crop_margin_ratio_override is not None
        else float(augmentation_config.get("crop_margin_ratio", 0.05) or 0.05)
    )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=transform,
        crop_margin_ratio=crop_margin_ratio,
        crop_to_primary_object=crop_to_primary_object,
        classification_target=True,
        classification_object_crops=classification_object_crops,
        class_aware_augmentation=False,
        classification_bbox_metadata=True,
    )


def _select_bbox_token_prior(
    *,
    bbox: Optional[torch.Tensor],
    crop_bbox: Optional[torch.Tensor],
    source: str,
) -> Optional[torch.Tensor]:
    if str(source or "bbox").strip().lower() == "crop_bbox" and torch.is_tensor(crop_bbox):
        return crop_bbox
    return bbox


def _forward_logits(
    *,
    model: nn.Module,
    images: torch.Tensor,
    image_valid_mask: Optional[torch.Tensor],
    bbox_metadata: Optional[torch.Tensor],
    bbox_token_prior: Optional[torch.Tensor],
    amp: bool,
    device: torch.device,
) -> torch.Tensor | Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    features_for_fusion = None
    with autocast_context(device, amp):
        if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            features_for_fusion = features
            output = model.forward_heads(features)
        elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            features_for_fusion = features
            output = classification_logits_from_features(model, features)
        else:
            output = model(images)
        logits, _, _ = extract_detection_from_model_output(output)
    return logits.float(), features_for_fusion


def _fused_probabilities(
    *,
    primary_logits: torch.Tensor,
    paired_logits: torch.Tensor,
    weight: float,
    mode: str,
) -> torch.Tensor:
    weight = float(weight)
    if str(mode).strip().lower() == "probability":
        return (
            (1.0 - weight) * torch.softmax(primary_logits.float(), dim=1)
            + weight * torch.softmax(paired_logits.float(), dim=1)
        )
    fused_logits = (1.0 - weight) * primary_logits.float() + weight * paired_logits.float()
    return torch.softmax(fused_logits, dim=1)


def _class1_f1(metrics: Dict[str, object], class_index: int = 1) -> float:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, list):
        return 0.0
    for item in per_class:
        if isinstance(item, dict) and int(item.get("class_index", -1)) == int(class_index):
            return float(item.get("f1", 0.0) or 0.0)
    return 0.0


def _write_rows_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate true paired-view fusion for one checkpoint: primary data view + "
            "paired object view of the same sample."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Primary data.yaml")
    parser.add_argument("--paired-data", type=Path, required=True, help="Paired view data.yaml")
    parser.add_argument(
        "--paired-classification-folder-yolo-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used to map bbox metadata for paired classification_folder data.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--fusion-mode", choices=("logit", "probability"), default="logit")
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--full-image-detection", action="store_true", default=False)
    parser.add_argument("--crop-to-primary-object", action="store_true", default=False)
    parser.add_argument("--disable-classification-object-crops", action="store_true", default=False)
    parser.add_argument("--paired-crop-margin-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    output_dir = ensure_dir(args.output_dir)
    weights = parse_weight_grid(args.weights)

    primary_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    paired_spec = load_data_spec(
        args.paired_data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or primary_spec.num_classes,
    )
    if paired_spec.class_names != primary_spec.class_names:
        raise ValueError("Class names cua primary va paired data khong khop.")
    paired_yolo_spec = None
    if args.paired_classification_folder_yolo_data is not None:
        paired_yolo_spec = load_data_spec(
            args.paired_classification_folder_yolo_data,
            class_name_mode=args.class_name_mode or "raw",
            expected_num_classes=args.expected_num_classes or primary_spec.num_classes,
        )
        if paired_yolo_spec.data_format == "classification_folder":
            raise ValueError("--paired-classification-folder-yolo-data phai la YOLO-format data.yaml.")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    mismatch = _checkpoint_data_path_mismatch(checkpoint, primary_spec.data_yaml)
    if mismatch is not None:
        print({"warning": "primary_data_path_differs_from_checkpoint", **mismatch}, flush=True)
    class_names = list(checkpoint.get("class_names", primary_spec.class_names))
    if len(class_names) != primary_spec.num_classes:
        raise ValueError("So class trong checkpoint khong khop primary data.")

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
    primary_dataset = _build_classification_dataset(
        data_spec=primary_spec,
        split=args.split,
        transform=eval_transform,
        checkpoint=checkpoint,
        full_image_detection=args.full_image_detection,
        crop_to_primary_object_override=args.crop_to_primary_object,
        disable_classification_object_crops=args.disable_classification_object_crops,
    )
    paired_dataset = _build_classification_dataset(
        data_spec=paired_spec,
        split=args.split,
        transform=eval_transform,
        checkpoint=checkpoint,
        paired_yolo_data_spec=paired_yolo_spec,
        full_image_detection=args.full_image_detection,
        crop_to_primary_object_override=args.crop_to_primary_object,
        disable_classification_object_crops=args.disable_classification_object_crops,
        crop_margin_ratio_override=args.paired_crop_margin_ratio,
    )
    dataset = PairedViewTrainDataset(
        primary_dataset=primary_dataset,
        paired_dataset=paired_dataset,
        primary_name=f"primary:{primary_spec.data_yaml.parent.name}",
        paired_name=f"paired:{paired_spec.data_yaml.parent.name}",
        require_all_matched=True,
    )

    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"paired_view_fusion_{args.split}",
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

    primary_logits_batches: List[torch.Tensor] = []
    paired_logits_batches: List[torch.Tensor] = []
    learned_fusion_prob_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    sample_paths = [str(path) for path in dataset.sample_paths()]
    total_batches = len(loader)
    iterator = loader
    if int(args.max_batches) > 0:
        total_batches = min(total_batches, int(args.max_batches))
        from itertools import islice

        iterator = islice(loader, total_batches)

    with torch.inference_mode():
        with tqdm(iterator, desc="Paired fusion eval", total=total_batches, dynamic_ncols=True) as pbar:
            for images, labels, targets in pbar:
                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                metric_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
                targets = targets if isinstance(targets, dict) else {}

                bbox = targets.get("bbox")
                crop_bbox = targets.get("crop_bbox")
                image_mask = targets.get("image_mask")
                paired_images = targets.get("paired_view_image")
                paired_bbox = targets.get("paired_view_bbox")
                paired_crop_bbox = targets.get("paired_view_crop_bbox")
                paired_image_mask = targets.get("paired_view_image_mask")
                if not torch.is_tensor(paired_images):
                    raise ValueError("Batch khong co paired_view_image.")

                bbox = bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(bbox) else None
                crop_bbox = crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(crop_bbox) else None
                image_mask = image_mask.to(device=device, dtype=torch.bool, non_blocking=True) if torch.is_tensor(image_mask) else None
                paired_images = paired_images.to(device=device, non_blocking=True)
                paired_bbox = paired_bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(paired_bbox) else None
                paired_crop_bbox = paired_crop_bbox.to(device=device, dtype=torch.float32, non_blocking=True) if torch.is_tensor(paired_crop_bbox) else None
                paired_image_mask = paired_image_mask.to(device=device, dtype=torch.bool, non_blocking=True) if torch.is_tensor(paired_image_mask) else None

                primary_prior = _select_bbox_token_prior(
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    source=args.bbox_token_prior_source,
                )
                paired_prior = _select_bbox_token_prior(
                    bbox=paired_bbox,
                    crop_bbox=paired_crop_bbox,
                    source=args.bbox_token_prior_source,
                )
                primary_logits, primary_features = _forward_logits(
                    model=model,
                    images=images,
                    image_valid_mask=image_mask,
                    bbox_metadata=bbox,
                    bbox_token_prior=primary_prior,
                    amp=args.amp,
                    device=device,
                )
                paired_logits, paired_features = _forward_logits(
                    model=model,
                    images=paired_images,
                    image_valid_mask=paired_image_mask,
                    bbox_metadata=paired_bbox,
                    bbox_token_prior=paired_prior,
                    amp=args.amp,
                    device=device,
                )
                learned_fusion = None
                if (
                    getattr(model, "paired_view_fusion_head", None) is not None
                    and isinstance(primary_features, dict)
                    and isinstance(paired_features, dict)
                ):
                    learned_fusion = paired_view_fused_logits_from_features(
                        model,
                        primary_features,
                        paired_features,
                        primary_logits,
                        paired_logits,
                    )
                primary_logits_batches.append(primary_logits.detach().cpu())
                paired_logits_batches.append(paired_logits.detach().cpu())
                if torch.is_tensor(learned_fusion):
                    learned_fusion_prob_batches.append(
                        torch.softmax(learned_fusion.float(), dim=1).detach().cpu()
                    )
                target_batches.append(metric_labels.detach().cpu().to(dtype=torch.long))

    primary_logits = torch.cat(primary_logits_batches, dim=0)
    paired_logits = torch.cat(paired_logits_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    learned_fusion_probabilities = (
        torch.cat(learned_fusion_prob_batches, dim=0)
        if learned_fusion_prob_batches
        else None
    )
    if len(sample_paths) > int(targets.numel()):
        sample_paths = sample_paths[: int(targets.numel())]

    sweep_rows: List[Dict[str, object]] = []
    metrics_by_key: Dict[str, Dict[str, object]] = {}
    for weight in weights:
        probabilities = _fused_probabilities(
            primary_logits=primary_logits,
            paired_logits=paired_logits,
            weight=weight,
            mode=args.fusion_mode,
        )
        predictions = probabilities.argmax(dim=1)
        metrics = build_metrics(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
        )
        key = f"w{weight:.6f}".rstrip("0").rstrip(".")
        metrics_by_key[key] = metrics
        row: Dict[str, object] = {
            "weight": float(weight),
            "fusion_mode": args.fusion_mode,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "class1_f1": _class1_f1(metrics, 1),
        }
        for item in metrics.get("per_class", []):
            if isinstance(item, dict):
                class_index = int(item.get("class_index", -1))
                row[f"class{class_index}_precision"] = item.get("precision", 0.0)
                row[f"class{class_index}_recall"] = item.get("recall", 0.0)
                row[f"class{class_index}_f1"] = item.get("f1", 0.0)
        sweep_rows.append(row)

    best_macro = max(sweep_rows, key=lambda row: float(row["macro_f1"]))
    best_class1 = max(sweep_rows, key=lambda row: float(row["class1_f1"]))
    learned_fusion_row = None
    learned_fusion_metrics = None
    if torch.is_tensor(learned_fusion_probabilities):
        if int(learned_fusion_probabilities.size(0)) != int(targets.numel()):
            raise ValueError(
                "learned fusion probability count khong khop targets: "
                f"{int(learned_fusion_probabilities.size(0))} vs {int(targets.numel())}"
            )
        learned_predictions = learned_fusion_probabilities.argmax(dim=1)
        learned_fusion_metrics = build_metrics(
            targets=targets,
            predictions=learned_predictions,
            probabilities=learned_fusion_probabilities,
            class_names=class_names,
        )
        learned_fusion_row = {
            "mode": "learned_paired_view_fusion",
            "accuracy": learned_fusion_metrics["accuracy"],
            "macro_f1": learned_fusion_metrics["macro_f1"],
            "weighted_f1": learned_fusion_metrics["weighted_f1"],
            "class1_f1": _class1_f1(learned_fusion_metrics, 1),
        }
        for item in learned_fusion_metrics.get("per_class", []):
            if isinstance(item, dict):
                class_index = int(item.get("class_index", -1))
                learned_fusion_row[f"class{class_index}_precision"] = item.get("precision", 0.0)
                learned_fusion_row[f"class{class_index}_recall"] = item.get("recall", 0.0)
                learned_fusion_row[f"class{class_index}_f1"] = item.get("f1", 0.0)
    _write_rows_csv(output_dir / "fusion_sweep.csv", sweep_rows)

    prediction_exports = {}
    for label, row in (("best_macro", best_macro), ("best_class1", best_class1)):
        weight = float(row["weight"])
        probabilities = _fused_probabilities(
            primary_logits=primary_logits,
            paired_logits=paired_logits,
            weight=weight,
            mode=args.fusion_mode,
        )
        predictions = probabilities.argmax(dim=1)
        records = _build_prediction_records(
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            sample_paths=sample_paths,
        )
        filename = f"predictions_{label}_w{weight:.2f}.csv"
        _write_rows_csv(output_dir / filename, records)
        prediction_exports[label] = filename
    if torch.is_tensor(learned_fusion_probabilities):
        learned_predictions = learned_fusion_probabilities.argmax(dim=1)
        records = _build_prediction_records(
            targets=targets,
            predictions=learned_predictions,
            probabilities=learned_fusion_probabilities,
            class_names=class_names,
            sample_paths=sample_paths,
        )
        filename = "predictions_learned_fusion.csv"
        _write_rows_csv(output_dir / filename, records)
        prediction_exports["learned_fusion"] = filename

    summary = {
        "checkpoint": str(args.checkpoint),
        "primary_data": str(primary_spec.data_yaml),
        "paired_data": str(paired_spec.data_yaml),
        "paired_yolo_data": str(paired_yolo_spec.data_yaml) if paired_yolo_spec is not None else None,
        "split": args.split,
        "samples": int(targets.numel()),
        "fusion_mode": args.fusion_mode,
        "weights": weights,
        "bbox_token_prior_source": args.bbox_token_prior_source,
        "dataloader": dataloader_summary,
        "primary_dataset_report": primary_dataset.quality_report(),
        "paired_dataset_report": paired_dataset.quality_report(),
        "paired_dataset_report_effective": dataset.quality_report(),
        "sweep": sweep_rows,
        "best_macro": best_macro,
        "best_class1": best_class1,
        "learned_fusion": learned_fusion_row,
        "learned_fusion_metrics": learned_fusion_metrics,
        "prediction_exports": prediction_exports,
        "metrics_by_weight": metrics_by_key,
        "leakage_guard": (
            "Only evaluates paired views on the requested split. Use val for development; "
            "do not select fusion weight on test."
        ),
    }
    json_dump(output_dir / "summary.json", to_serializable(summary))
    print(
        {
            "output_dir": str(output_dir),
            "samples": int(targets.numel()),
            "best_macro": best_macro,
            "best_class1": best_class1,
            "learned_fusion": learned_fusion_row,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
