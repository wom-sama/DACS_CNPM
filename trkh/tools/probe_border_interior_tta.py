from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.core.utils import autocast_context, ensure_dir, load_checkpoint, set_seed
from trkh.data.dataset import _pseudo_foreground_mask_array, _soft_mask_image
from trkh.evaluation.xai_audit import _build_dataset, _forward_logits_with_optional_bbox
from trkh.inference.inference import load_model
from trkh.tools.evaluate_counterfactual_color_guard import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe fixed border/interior counterfactual TTA on YOLO object crops. "
            "This diagnostic does not train, tune thresholds, edit raw data, or use test unless explicitly requested."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default=None)
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument("--mask-margin", type=float, default=0.08)
    parser.add_argument("--border-ratio", type=float, default=0.075)
    parser.add_argument("--crop-edge-ratio", type=float, default=0.075)
    parser.add_argument("--blur-radius", type=float, default=7.0)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--patch-evidence-linear-verifier-json", type=Path, default=None)
    parser.add_argument("--patch-evidence-linear-verifier-pair", type=str, default="0-1")
    parser.add_argument("--patch-evidence-linear-verifier-min-pair-probability", type=float, default=0.02)
    parser.add_argument("--patch-evidence-linear-verifier-max-pair-margin", type=float, default=0.40)
    parser.add_argument("--patch-evidence-linear-verifier-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--patch-evidence-linear-verifier-logit-boost", type=float, default=0.01)
    parser.add_argument(
        "--patch-evidence-linear-verifier-protect-right-min-probability",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _clone_target(target: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in target.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _crop_image_and_target(dataset, index: int):
    sample = dataset.samples[int(index)]
    image = dataset._load_rgb_image(sample.image_path)
    labels = torch.tensor([obj.label for obj in sample.objects], dtype=torch.long)
    boxes = torch.tensor([obj.bbox for obj in sample.objects], dtype=torch.float32)
    if getattr(dataset, "crop_to_primary_object", False):
        image, target = dataset._crop_to_primary_object(image=image, labels=labels, boxes=boxes, sample=sample)
    else:
        target = {"labels": labels, "boxes": boxes}
    target["augmentation_scale"] = torch.tensor(
        [dataset._augmentation_scale_for_labels(target["labels"])],
        dtype=torch.float32,
    )
    return image.convert("RGB"), target, sample


def _transform_image(dataset, image: Image.Image, target: Mapping[str, torch.Tensor]):
    if dataset.transform is None:
        raise ValueError("Border/interior TTA probe requires tensor eval transform.")
    transformed = dataset.transform(image, target=_clone_target(target))
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise TypeError("Expected eval transform to return (image_tensor, target).")
    tensor, transformed_target = transformed
    if not torch.is_tensor(tensor):
        raise TypeError("Eval transform did not return an image tensor.")
    return tensor, dict(transformed_target)


def _metadata_from_transformed(*, dataset, sample, transformed_target: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    primary_object = dataset._select_sample_primary_object(sample)
    metadata: Dict[str, torch.Tensor] = {
        "bbox": torch.tensor(primary_object.bbox, dtype=torch.float32),
    }
    transformed_boxes = transformed_target.get("boxes")
    if torch.is_tensor(transformed_boxes) and transformed_boxes.ndim == 2 and int(transformed_boxes.size(0)) > 0:
        areas = transformed_boxes[:, 2].clamp(min=0.0) * transformed_boxes[:, 3].clamp(min=0.0)
        crop_bbox = transformed_boxes[int(torch.argmax(areas).item())]
    else:
        crop_bbox = torch.tensor((0.5, 0.5, 1.0, 1.0), dtype=torch.float32)
    metadata["crop_bbox"] = crop_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)
    image_mask = transformed_target.get("image_mask")
    if torch.is_tensor(image_mask):
        metadata["image_mask"] = image_mask.to(dtype=torch.bool)
    return metadata


def _forward_batch(
    *,
    model,
    tensors: Sequence[torch.Tensor],
    metadata_items: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
    amp: bool,
    bbox_token_prior_source: str,
) -> torch.Tensor:
    images = torch.stack([tensor.to(dtype=torch.float32) for tensor in tensors], dim=0).to(device=device)
    bbox = torch.stack([item["bbox"] for item in metadata_items], dim=0).to(device=device, dtype=torch.float32)
    crop_bbox = torch.stack([item["crop_bbox"] for item in metadata_items], dim=0).to(device=device, dtype=torch.float32)
    masks = [item.get("image_mask") for item in metadata_items]
    image_mask = None
    if all(torch.is_tensor(mask) for mask in masks):
        image_mask = torch.stack([mask for mask in masks if torch.is_tensor(mask)], dim=0).to(
            device=device,
            dtype=torch.bool,
        )
    bbox_prior = crop_bbox if str(bbox_token_prior_source).lower() == "crop_bbox" else bbox
    with torch.no_grad():
        with autocast_context(device, amp):
            logits = _forward_logits_with_optional_bbox(
                model,
                images,
                bbox,
                bbox_prior,
                image_valid_mask=image_mask,
            )
    return F.softmax(logits.float(), dim=1).detach().cpu()


def _binary_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L")


def _filter_size_from_ratio(image: Image.Image, ratio: float) -> int:
    radius = max(1, int(round(min(image.size) * max(0.0, float(ratio)))))
    size = 2 * radius + 1
    return max(3, size if size % 2 == 1 else size + 1)


def _erode(mask: np.ndarray, image: Image.Image, ratio: float) -> np.ndarray:
    return np.asarray(_binary_image(mask).filter(ImageFilter.MinFilter(_filter_size_from_ratio(image, ratio)))) > 127


def _dilate(mask: np.ndarray, image: Image.Image, ratio: float) -> np.ndarray:
    return np.asarray(_binary_image(mask).filter(ImageFilter.MaxFilter(_filter_size_from_ratio(image, ratio)))) > 127


def _edge_ring(image: Image.Image, ratio: float) -> np.ndarray:
    width, height = image.size
    ring = np.zeros((height, width), dtype=bool)
    edge = max(1, int(round(min(width, height) * max(0.0, float(ratio)))))
    ring[:edge, :] = True
    ring[-edge:, :] = True
    ring[:, :edge] = True
    ring[:, -edge:] = True
    return ring


def _replacement_image(image: Image.Image, *, blur_radius: float) -> Image.Image:
    base = image.convert("RGB")
    return ImageOps.grayscale(base).convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=max(0.1, float(blur_radius)))
    )


def _replace_masked(image: Image.Image, mask: np.ndarray, *, blur_radius: float) -> Image.Image:
    base = image.convert("RGB")
    replacement = _replacement_image(base, blur_radius=blur_radius)
    alpha = _soft_mask_image(mask, radius=max(1.0, float(blur_radius) * 0.25))
    return Image.composite(replacement, base, alpha)


def build_counterfactual_variants(
    image: Image.Image,
    *,
    mask_margin: float,
    border_ratio: float,
    crop_edge_ratio: float,
    blur_radius: float,
) -> Tuple[Dict[str, Image.Image], Dict[str, float]]:
    base = image.convert("RGB")
    foreground = _pseudo_foreground_mask_array(base, margin=float(mask_margin))
    if float(np.mean(foreground)) < 0.03:
        foreground = np.ones((base.height, base.width), dtype=bool)
    eroded = _erode(foreground, base, float(border_ratio))
    if float(np.mean(eroded)) < 0.025:
        eroded = _erode(foreground, base, float(border_ratio) * 0.5)
    if float(np.mean(eroded)) < 0.025:
        eroded = foreground
    dilated = _dilate(foreground, base, max(float(border_ratio) * 0.5, 0.02))
    object_border = (dilated & ~eroded) | (foreground & ~eroded)
    crop_edge = _edge_ring(base, float(crop_edge_ratio))
    outside_core = ~eroded
    variants = {
        "clean": base,
        "border_suppressed": _replace_masked(base, object_border, blur_radius=blur_radius),
        "core_only": _replace_masked(base, outside_core, blur_radius=blur_radius),
        "interior_suppressed": _replace_masked(base, eroded, blur_radius=blur_radius),
        "crop_edge_suppressed": _replace_masked(base, crop_edge, blur_radius=blur_radius),
    }
    masks = {
        "foreground_fraction": float(np.mean(foreground)),
        "eroded_core_fraction": float(np.mean(eroded)),
        "object_border_fraction": float(np.mean(object_border)),
        "crop_edge_fraction": float(np.mean(crop_edge)),
    }
    return variants, masks


def _top_prediction(probabilities: torch.Tensor) -> Tuple[int, float, int, float, float]:
    values, indices = torch.topk(probabilities, k=min(2, int(probabilities.numel())))
    top1 = int(indices[0].item())
    conf1 = float(values[0].item())
    top2 = int(indices[1].item()) if int(values.numel()) > 1 else -1
    conf2 = float(values[1].item()) if int(values.numel()) > 1 else 0.0
    return top1, conf1, top2, conf2, conf1 - conf2


def _metric_summary(targets: Sequence[int], preds: Sequence[int], *, num_classes: int) -> Dict[str, object]:
    return compute_classification_metrics(targets, preds, num_classes=num_classes)


def _variant_change_summary(
    *,
    rows: Sequence[Mapping[str, object]],
    variant: str,
    focus_class: int,
    num_classes: int,
) -> Dict[str, object]:
    del num_classes
    changed = 0
    corrections = 0
    harms = 0
    wrong_to_wrong = 0
    class1_fp_corrections = 0
    class1_new_fp = 0
    class1_fn_rescues = 0
    class1_recall_harms = 0
    transitions: Counter[str] = Counter()
    target_transitions: Counter[str] = Counter()
    for row in rows:
        target = _safe_int(row.get("target_index"), -1)
        clean = _safe_int(row.get("clean_pred"), -1)
        pred = _safe_int(row.get(f"{variant}_pred"), -1)
        if pred == clean:
            continue
        changed += 1
        transitions[f"{clean}->{pred}"] += 1
        target_transitions[f"t{target}:{clean}->{pred}"] += 1
        if clean != target and pred == target:
            corrections += 1
        elif clean == target and pred != target:
            harms += 1
        elif clean != target and pred != target:
            wrong_to_wrong += 1
        if target != int(focus_class) and clean == int(focus_class) and pred == target:
            class1_fp_corrections += 1
        if target != int(focus_class) and clean != int(focus_class) and pred == int(focus_class):
            class1_new_fp += 1
        if target == int(focus_class) and clean != int(focus_class) and pred == int(focus_class):
            class1_fn_rescues += 1
        if target == int(focus_class) and clean == int(focus_class) and pred != int(focus_class):
            class1_recall_harms += 1
    return {
        "changed": int(changed),
        "corrections": int(corrections),
        "harms": int(harms),
        "wrong_to_wrong": int(wrong_to_wrong),
        "focus_fp_corrections": int(class1_fp_corrections),
        "focus_new_fp": int(class1_new_fp),
        "focus_fn_rescues": int(class1_fn_rescues),
        "focus_recall_harms": int(class1_recall_harms),
        "top_transitions": dict(transitions.most_common(12)),
        "top_target_transitions": dict(target_transitions.most_common(12)),
    }


def _prediction_rows_for_probs(
    rows: Sequence[Mapping[str, object]],
    *,
    probs_by_name: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    num_classes = len(class_names)
    for index, base_row in enumerate(rows):
        row = dict(base_row)
        for name, probabilities in probs_by_name.items():
            probs = probabilities[int(index)].to(dtype=torch.float32)
            pred, conf, top2, top2_conf, margin = _top_prediction(probs)
            row[f"{name}_pred"] = int(pred)
            row[f"{name}_confidence"] = float(conf)
            row[f"{name}_top2"] = int(top2)
            row[f"{name}_top2_confidence"] = float(top2_conf)
            row[f"{name}_margin"] = float(margin)
            row[f"{name}_correct"] = int(pred == _safe_int(row.get("target_index"), -1))
            for class_index in range(num_classes):
                row[f"{name}_prob_{class_index}"] = float(probs[class_index].item())
        output.append(row)
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_variant_probabilities(
    *,
    model,
    dataset,
    split: str,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_samples: int,
    amp: bool,
    bbox_token_prior_source: str,
    mask_margin: float,
    border_ratio: float,
    crop_edge_ratio: float,
    blur_radius: float,
) -> Tuple[List[Dict[str, object]], Dict[str, torch.Tensor], Dict[str, float]]:
    base_rows: List[Dict[str, object]] = []
    variant_tensors: Dict[str, List[torch.Tensor]] = {}
    variant_probs: Dict[str, List[torch.Tensor]] = {}
    metadata_items: List[Dict[str, torch.Tensor]] = []
    mask_totals: Counter[str] = Counter()
    mask_count = 0
    total = len(dataset) if int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    model.eval()

    def flush() -> None:
        if not metadata_items:
            return
        for variant_name, tensors in list(variant_tensors.items()):
            probabilities = _forward_batch(
                model=model,
                tensors=tensors,
                metadata_items=metadata_items,
                device=device,
                amp=amp,
                bbox_token_prior_source=bbox_token_prior_source,
            )
            variant_probs.setdefault(variant_name, []).append(probabilities)
            tensors.clear()
        metadata_items.clear()

    for index in range(total):
        crop_image, target, sample = _crop_image_and_target(dataset, index)
        variants, mask_stats = build_counterfactual_variants(
            crop_image,
            mask_margin=mask_margin,
            border_ratio=border_ratio,
            crop_edge_ratio=crop_edge_ratio,
            blur_radius=blur_radius,
        )
        clean_tensor, clean_target = _transform_image(dataset, variants["clean"], target)
        metadata = _metadata_from_transformed(dataset=dataset, sample=sample, transformed_target=clean_target)
        primary_object = dataset._select_sample_primary_object(sample)
        label = int(sample.primary_label)
        base_rows.append(
            {
                "split": str(split),
                "sample_index": int(index),
                "image_path": str(sample.image_path),
                "source_stem": str(sample.image_path.stem),
                "object_index": int(primary_object.object_index),
                "target_index": int(label),
                "target_name": str(class_names[label]) if 0 <= label < len(class_names) else str(label),
            }
        )
        for key, value in mask_stats.items():
            mask_totals[str(key)] += float(value)
        mask_count += 1
        for variant_name, variant_image in variants.items():
            if variant_name == "clean":
                tensor = clean_tensor
            else:
                tensor, _ = _transform_image(dataset, variant_image, target)
            variant_tensors.setdefault(variant_name, []).append(tensor)
        metadata_items.append(metadata)
        if len(metadata_items) >= max(1, int(batch_size)):
            flush()
        if (index + 1) % max(1, int(batch_size) * 10) == 0:
            print(f"processed {index + 1}/{total}", flush=True)
    flush()
    merged_probs = {name: torch.cat(chunks, dim=0) for name, chunks in variant_probs.items() if chunks}
    mask_summary = {key: value / max(1, mask_count) for key, value in mask_totals.items()}
    return base_rows, merged_probs, mask_summary


def _aggregate_probabilities(probs_by_name: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = probs_by_name["clean"]
    aggregates = {
        "avg_clean_border_suppressed": 0.5 * (clean + probs_by_name["border_suppressed"]),
        "avg_clean_core_only": 0.5 * (clean + probs_by_name["core_only"]),
        "avg_clean_crop_edge_suppressed": 0.5 * (clean + probs_by_name["crop_edge_suppressed"]),
        "avg_clean_border_core": (clean + probs_by_name["border_suppressed"] + probs_by_name["core_only"]) / 3.0,
    }
    return aggregates


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    if args.split == "test":
        print("WARNING: split=test requested; do not use this diagnostic for method selection.", flush=True)
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    if data_spec.data_format != "yolo":
        raise ValueError("Border/interior TTA probe expects YOLO-format yolo_f data.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    verifier_summary: Dict[str, object] = {"enabled": False}
    if args.patch_evidence_linear_verifier_json is not None:
        load_verifier = getattr(model, "load_patch_evidence_linear_verifier_export", None)
        if not callable(load_verifier):
            raise RuntimeError("Checkpoint model does not support patch-evidence linear verifier export.")
        verifier_summary = load_verifier(
            args.patch_evidence_linear_verifier_json,
            pair=args.patch_evidence_linear_verifier_pair,
            min_pair_probability=args.patch_evidence_linear_verifier_min_pair_probability,
            max_pair_margin=args.patch_evidence_linear_verifier_max_pair_margin,
            confidence_threshold=args.patch_evidence_linear_verifier_confidence_threshold,
            logit_boost=args.patch_evidence_linear_verifier_logit_boost,
            protect_right_min_probability=args.patch_evidence_linear_verifier_protect_right_min_probability,
        )
        print({"patch_evidence_linear_verifier": verifier_summary}, flush=True)
    dataset = _build_dataset(
        data_yaml=args.data,
        classification_folder_yolo_data=None,
        split=args.split,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
        checkpoint=checkpoint_payload if isinstance(checkpoint_payload, dict) else checkpoint,
    )
    output_dir = ensure_dir(args.output_dir)
    rows, variant_probs, mask_summary = collect_variant_probabilities(
        model=model,
        dataset=dataset,
        split=args.split,
        class_names=class_names,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        max_samples=max(0, int(args.max_samples)),
        amp=not bool(args.disable_amp),
        bbox_token_prior_source=str(args.bbox_token_prior_source),
        mask_margin=float(args.mask_margin),
        border_ratio=float(args.border_ratio),
        crop_edge_ratio=float(args.crop_edge_ratio),
        blur_radius=float(args.blur_radius),
    )
    aggregate_probs = _aggregate_probabilities(variant_probs)
    all_probs: Dict[str, torch.Tensor] = {**variant_probs, **aggregate_probs}
    prediction_rows = _prediction_rows_for_probs(
        rows,
        probs_by_name=all_probs,
        class_names=class_names,
    )
    predictions_path = output_dir / f"border_interior_tta_{args.split}.csv"
    _write_rows(predictions_path, prediction_rows)

    targets = [_safe_int(row.get("target_index"), -1) for row in prediction_rows]
    num_classes = len(class_names)
    metrics_by_name: Dict[str, object] = {}
    changes_by_name: Dict[str, object] = {}
    for name in all_probs:
        preds = [_safe_int(row.get(f"{name}_pred"), -1) for row in prediction_rows]
        metrics_by_name[name] = _metric_summary(targets, preds, num_classes=num_classes)
        if name != "clean":
            changes_by_name[name] = _variant_change_summary(
                rows=prediction_rows,
                variant=name,
                focus_class=int(args.focus_class),
                num_classes=num_classes,
            )

    changed_rows = [
        row
        for row in prediction_rows
        if any(_safe_int(row.get(f"{name}_pred"), -1) != _safe_int(row.get("clean_pred"), -1) for name in all_probs if name != "clean")
    ]
    _write_rows(output_dir / f"border_interior_tta_{args.split}_changed.csv", changed_rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": str(args.split),
        "rows": int(len(prediction_rows)),
        "class_names": [str(name) for name in class_names],
        "prediction_csv": str(predictions_path.resolve()),
        "changed_csv": str((output_dir / f"border_interior_tta_{args.split}_changed.csv").resolve()),
        "mask_summary": mask_summary,
        "variant_order": list(variant_probs.keys()),
        "aggregate_order": list(aggregate_probs.keys()),
        "metrics_by_name": metrics_by_name,
        "changes_vs_clean": changes_by_name,
        "patch_evidence_linear_verifier": verifier_summary,
        "note": (
            "Fixed diagnostic variants/averages only. Do not select new thresholds or use test "
            "from this output without train/OOF evidence."
        ),
    }
    summary_path = output_dir / f"border_interior_tta_{args.split}_summary.json"
    summary_path.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")
    print(json.dumps(to_serializable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
