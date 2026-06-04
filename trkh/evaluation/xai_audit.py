from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from trkh.core.config import default_data_yaml, load_data_spec, to_serializable
from trkh.core.utils import autocast_context, ensure_dir, json_dump
from trkh.data.dataset import ClassificationFolderDataset, build_eval_transform
from trkh.evaluation.attention_viz import analyze_tensor, prepare_image_and_tensor, resolve_layer_index
from trkh.inference.inference import load_model
from trkh.models.feature_hooks import count_attention_layers
from trkh.models.model import extract_bbox_from_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XAI audit nhe cho classification-only fail/low-confidence cases."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
    )
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--mistake-cases", type=int, default=16)
    parser.add_argument("--low-confidence-cases", type=int, default=8)
    parser.add_argument("--close-margin-cases", type=int, default=8)
    parser.add_argument("--per-class-cases", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head-reduction", choices=("mean", "max"), default="mean")
    parser.add_argument(
        "--query-tokens",
        choices=("cls", "registers", "cls_register_mean"),
        default="cls_register_mean",
    )
    parser.add_argument(
        "--method",
        choices=("attention", "rollout", "grad_rollout", "gradcam", "both", "all"),
        default="both",
    )
    parser.add_argument("--feature-source", choices=("auto", "patch_embed", "stem_last", "last_conv"), default="auto")
    parser.add_argument("--rollout-start-layer", type=int, default=0)
    parser.add_argument("--robustness-probes", action="store_true", default=False)
    parser.add_argument("--review-high-confidence", type=float, default=0.95)
    parser.add_argument("--review-background-threshold", type=float, default=0.12)
    parser.add_argument("--review-border-threshold", type=float, default=0.25)
    parser.add_argument("--shortcut-drop-threshold", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    return parser.parse_args()


def _build_dataset(
    *,
    data_yaml: Path,
    split: str,
    class_name_mode: Optional[str],
    expected_num_classes: int,
    checkpoint: Dict[str, object],
) -> ClassificationFolderDataset:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode=class_name_mode,
        expected_num_classes=expected_num_classes or None,
    )
    if data_spec.data_format != "classification_folder":
        raise ValueError(
            "xai_audit hien uu tien format=classification_folder de tranh lech crop. "
            "Hay dung data.yaml cua cls_crops cho nhanh phan loai."
        )
    image_size = int(checkpoint["model_config"]["image_size"])
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=checkpoint.get("augmentation_config", {}).get("resize_mode", "pad"),
    )
    return ClassificationFolderDataset.from_data_spec(
        data_spec=data_spec,
        split=split,
        transform=transform,
        class_aware_augmentation=False,
    )


def _collect_predictions(
    *,
    model,
    dataset: ClassificationFolderDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> List[Dict[str, object]]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda" and int(num_workers) > 0),
    )
    sample_paths = dataset.sample_paths()
    records: List[Dict[str, object]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            images = images.to(device=device, non_blocking=True)
            with autocast_context(device, amp):
                logits, _ = extract_bbox_from_model_output(model(images))
            probabilities = F.softmax(logits.float(), dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)
            top_values, top_indices = torch.topk(
                probabilities,
                k=min(2, int(probabilities.shape[1])),
                dim=1,
            )
            batch_size_actual = int(probabilities.shape[0])
            labels_cpu = labels.detach().cpu()
            for batch_index in range(batch_size_actual):
                sample_index = offset + batch_index
                target_index = int(labels_cpu[batch_index].item())
                prediction_index = int(predictions[batch_index].item())
                top1 = float(top_values[batch_index, 0].item())
                top2 = float(top_values[batch_index, 1].item()) if top_values.shape[1] > 1 else 0.0
                record = {
                    "sample_index": sample_index,
                    "image_path": str(sample_paths[sample_index]),
                    "target_index": target_index,
                    "target_name": dataset.class_names[target_index],
                    "prediction_index": prediction_index,
                    "prediction_name": dataset.class_names[prediction_index],
                    "confidence": top1,
                    "top2_index": int(top_indices[batch_index, 1].item()) if top_indices.shape[1] > 1 else -1,
                    "top2_confidence": top2,
                    "margin": top1 - top2,
                    "correct": int(target_index == prediction_index),
                }
                records.append(record)
            offset += batch_size_actual
    return records


def _select_cases(
    records: Sequence[Dict[str, object]],
    *,
    num_classes: int,
    max_cases: int,
    mistake_cases: int,
    low_confidence_cases: int,
    close_margin_cases: int,
    per_class_cases: int,
) -> List[Dict[str, object]]:
    selected: "OrderedDict[int, Dict[str, object]]" = OrderedDict()

    def add_many(items: Sequence[Dict[str, object]], reason: str, limit: int) -> None:
        for item in items[: max(0, int(limit))]:
            sample_index = int(item["sample_index"])
            if sample_index not in selected:
                record = dict(item)
                record["audit_reason"] = reason
                selected[sample_index] = record
            if len(selected) >= max_cases:
                return

    mistakes = [record for record in records if int(record.get("correct", 0)) == 0]
    add_many(
        sorted(mistakes, key=lambda item: float(item.get("confidence", 0.0)), reverse=True),
        "high_confidence_mistake",
        mistake_cases,
    )
    if len(selected) < max_cases:
        pair_counts: Dict[tuple[int, int], int] = {}
        for record in mistakes:
            target = int(record.get("target_index", -1))
            pred = int(record.get("prediction_index", -1))
            if 0 <= target < num_classes and 0 <= pred < num_classes and target != pred:
                pair_counts[(target, pred)] = pair_counts.get((target, pred), 0) + 1
        top_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
        for (target, pred), _count in top_pairs[: max(1, num_classes)]:
            if len(selected) >= max_cases:
                break
            pair_records = [
                record
                for record in mistakes
                if int(record.get("target_index", -1)) == target
                and int(record.get("prediction_index", -1)) == pred
            ]
            add_many(
                sorted(
                    pair_records,
                    key=lambda item: (float(item.get("confidence", 0.0)), -float(item.get("margin", 1.0))),
                    reverse=True,
                ),
                f"confusion_{target}_to_{pred}",
                max(1, per_class_cases),
            )
    if len(selected) < max_cases:
        add_many(
            sorted(records, key=lambda item: float(item.get("confidence", 0.0))),
            "low_confidence",
            low_confidence_cases,
        )
    if len(selected) < max_cases:
        add_many(
            sorted(records, key=lambda item: float(item.get("margin", 1.0))),
            "close_top2_margin",
            close_margin_cases,
        )

    for class_index in range(num_classes):
        if len(selected) >= max_cases:
            break
        class_mistakes = [
            record
            for record in mistakes
            if int(record.get("target_index", -1)) == class_index
            or int(record.get("prediction_index", -1)) == class_index
        ]
        add_many(
            sorted(class_mistakes, key=lambda item: float(item.get("confidence", 0.0)), reverse=True),
            f"class_{class_index}_mistake",
            per_class_cases,
        )

    for class_index in range(num_classes):
        if len(selected) >= max_cases:
            break
        class_correct = [
            record
            for record in records
            if int(record.get("target_index", -1)) == class_index
            and int(record.get("correct", 0)) == 1
        ]
        add_many(
            sorted(class_correct, key=lambda item: float(item.get("confidence", 0.0))),
            f"class_{class_index}_low_conf_correct",
            per_class_cases,
        )

    return list(selected.values())[:max_cases]


def _confusion_pairs(records: Sequence[Dict[str, object]], num_classes: int) -> List[Dict[str, int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for record in records:
        target = int(record.get("target_index", -1))
        pred = int(record.get("prediction_index", -1))
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            matrix[target][pred] += 1
    pairs = []
    for target in range(num_classes):
        for pred in range(num_classes):
            if target != pred and matrix[target][pred] > 0:
                pairs.append({"target_index": target, "prediction_index": pred, "count": matrix[target][pred]})
    return sorted(pairs, key=lambda item: item["count"], reverse=True)


def _pseudo_foreground_mask(crop_image: Image.Image, margin: float = 0.08) -> np.ndarray:
    rgb = np.asarray(crop_image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    median_rgb = np.median(rgb.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    median_gray = float(np.median(gray))
    color_delta = np.abs(rgb - median_rgb).mean(axis=2)
    intensity_delta = np.abs(gray - median_gray)
    mask = (color_delta + intensity_delta) > max(0.0, float(margin))
    height, width = gray.shape
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, width)
    central_ellipse = ((xx / 0.82) ** 2 + (yy / 0.92) ** 2) <= 1.0
    mask = np.logical_or(mask, central_ellipse)
    if float(mask.mean()) < 0.08:
        mask = ((xx / 0.78) ** 2 + (yy / 0.90) ** 2) <= 1.0
    return mask


def _apply_masked_background(crop_image: Image.Image, mode: str) -> Image.Image:
    image = crop_image.convert("RGB")
    mask = _pseudo_foreground_mask(image)
    base = np.asarray(image, dtype=np.uint8)
    if mode == "background_gray":
        gray = np.asarray(image.convert("L").convert("RGB"), dtype=np.uint8)
        out = np.where(mask[:, :, None], base, gray)
        return Image.fromarray(out)
    if mode == "background_blur":
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.uint8)
        out = np.where(mask[:, :, None], base, blurred)
        return Image.fromarray(out)
    if mode == "object_desaturate":
        gray = np.asarray(image.convert("L").convert("RGB"), dtype=np.uint8)
        out = np.where(mask[:, :, None], gray, base)
        return Image.fromarray(out)
    raise ValueError(f"Unknown masked background mode: {mode}")


def _center_occlusion(crop_image: Image.Image, ratio: float = 0.24) -> Image.Image:
    array = np.array(crop_image.convert("RGB"), copy=True)
    height, width = array.shape[:2]
    occ_width = max(1, int(round(width * ratio)))
    occ_height = max(1, int(round(height * ratio)))
    left = max(0, (width - occ_width) // 2)
    top = max(0, (height - occ_height) // 2)
    right = min(width, left + occ_width)
    bottom = min(height, top + occ_height)
    array[top:bottom, left:right] = np.array([127, 127, 127], dtype=array.dtype)
    return Image.fromarray(array)


def _tensor_from_crop(checkpoint: Dict[str, object], crop_image: Image.Image, device: torch.device) -> torch.Tensor:
    image_size = int(checkpoint["model_config"]["image_size"])
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=checkpoint.get("augmentation_config", {}).get("resize_mode", "pad"),
    )
    transformed = transform(crop_image)
    tensor = transformed[0] if isinstance(transformed, tuple) else transformed
    return tensor.unsqueeze(0).to(device)


def _run_robustness_probes(
    *,
    model,
    checkpoint: Dict[str, object],
    crop_image: Image.Image,
    original_prediction: int,
    target_index: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, Dict[str, object]]:
    variants = {
        "clean_recheck": crop_image,
        "background_gray": _apply_masked_background(crop_image, "background_gray"),
        "background_blur": _apply_masked_background(crop_image, "background_blur"),
        "object_desaturate": _apply_masked_background(crop_image, "object_desaturate"),
        "center_occlusion": _center_occlusion(crop_image),
    }
    results: Dict[str, Dict[str, object]] = {}
    clean_pred_probability = 0.0
    clean_target_probability = 0.0
    model.eval()
    with torch.no_grad():
        for name, image in variants.items():
            tensor = _tensor_from_crop(checkpoint, image, device)
            with autocast_context(device, amp):
                logits, _ = extract_bbox_from_model_output(model(tensor))
            probabilities = F.softmax(logits.float(), dim=1)[0].detach().cpu()
            pred_probability = float(probabilities[int(original_prediction)].item())
            target_probability = float(probabilities[int(target_index)].item())
            if name == "clean_recheck":
                clean_pred_probability = pred_probability
                clean_target_probability = target_probability
            results[name] = {
                "prediction_index": int(probabilities.argmax().item()),
                "prediction_probability": float(probabilities.max().item()),
                "original_prediction_probability": pred_probability,
                "target_probability": target_probability,
            }
    for item in results.values():
        item["original_prediction_drop"] = float(
            clean_pred_probability - float(item["original_prediction_probability"])
        )
        item["target_probability_drop"] = float(
            clean_target_probability - float(item["target_probability"])
        )
    return results


def _build_review_flags(
    case: Dict[str, object],
    *,
    high_confidence: float,
    background_threshold: float,
    border_threshold: float,
    shortcut_drop_threshold: float,
) -> List[str]:
    flags: List[str] = []
    correct = int(case.get("correct", 0)) == 1
    confidence = float(case.get("confidence", 0.0))
    margin = float(case.get("margin", 0.0))
    if not correct and confidence >= float(high_confidence):
        flags.append("high_confidence_misclassification")
    if margin <= 0.02:
        flags.append("near_tie_top2")

    viz = case.get("viz", {})
    if isinstance(viz, dict):
        focus = viz.get("heatmap_focus", {})
        if isinstance(focus, dict):
            for method_name, metrics in focus.items():
                if not isinstance(metrics, dict):
                    continue
                if float(metrics.get("background_mass", 0.0)) >= float(background_threshold):
                    flags.append(f"{method_name}_background_attention")
                if float(metrics.get("border_mass", 0.0)) >= float(border_threshold):
                    flags.append(f"{method_name}_border_attention")
        register_attention = viz.get("register_attention", {})
        if isinstance(register_attention, dict):
            similarity = float(register_attention.get("cls_register_heatmap_similarity", 0.0))
            entropy = float(register_attention.get("register_attention_entropy", 0.0))
            if similarity < 0.20:
                flags.append("register_cls_attention_divergent")
            if entropy > 0.98:
                flags.append("register_attention_diffuse")

    robustness = case.get("robustness", {})
    if isinstance(robustness, dict):
        for probe_name in ("background_gray", "background_blur"):
            probe = robustness.get(probe_name, {})
            if isinstance(probe, dict) and float(probe.get("original_prediction_drop", 0.0)) >= float(shortcut_drop_threshold):
                flags.append(f"{probe_name}_shortcut_sensitive")
        object_probe = robustness.get("object_desaturate", {})
        if isinstance(object_probe, dict) and float(object_probe.get("original_prediction_drop", 0.0)) >= float(shortcut_drop_threshold):
            flags.append("object_color_sensitive")
    return sorted(set(flags))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _aggregate_xai_metrics(cases: Sequence[Dict[str, object]]) -> Dict[str, object]:
    focus_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    register_values: Dict[str, List[float]] = defaultdict(list)
    robustness_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    flag_counts: Dict[str, int] = defaultdict(int)
    for case in cases:
        for flag in case.get("review_flags", []) if isinstance(case.get("review_flags", []), list) else []:
            flag_counts[str(flag)] += 1
        viz = case.get("viz", {})
        if isinstance(viz, dict):
            focus = viz.get("heatmap_focus", {})
            if isinstance(focus, dict):
                for method_name, metrics in focus.items():
                    if not isinstance(metrics, dict):
                        continue
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            focus_values[str(method_name)][str(metric_name)].append(float(value))
            register_attention = viz.get("register_attention", {})
            if isinstance(register_attention, dict):
                for metric_name, value in register_attention.items():
                    if isinstance(value, (int, float)):
                        register_values[str(metric_name)].append(float(value))
        robustness = case.get("robustness", {})
        if isinstance(robustness, dict):
            for probe_name, metrics in robustness.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name in ("original_prediction_drop", "target_probability_drop"):
                    if metric_name in metrics:
                        robustness_values[str(probe_name)][metric_name].append(float(metrics[metric_name]))

    return {
        "selected_cases": len(cases),
        "review_flag_counts": dict(sorted(flag_counts.items())),
        "heatmap_focus_mean": {
            method_name: {metric: _mean(values) for metric, values in metrics.items()}
            for method_name, metrics in sorted(focus_values.items())
        },
        "register_attention_mean": {
            metric: _mean(values) for metric, values in sorted(register_values.items())
        },
        "robustness_mean": {
            probe_name: {metric: _mean(values) for metric, values in metrics.items()}
            for probe_name, metrics in sorted(robustness_values.items())
        },
    }


def _write_case_table(path: Path, cases: Sequence[Dict[str, object]]) -> None:
    fields = [
        "sample_index",
        "audit_reason",
        "image_path",
        "target_index",
        "target_name",
        "prediction_index",
        "prediction_name",
        "confidence",
        "top2_index",
        "top2_confidence",
        "margin",
        "correct",
        "review_flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {field: case.get(field, "") for field in fields}
            row["review_flags"] = ";".join(str(flag) for flag in case.get("review_flags", []))
            writer.writerow(row)


def _write_review_manifest(path: Path, cases: Sequence[Dict[str, object]]) -> None:
    fields = [
        "sample_index",
        "image_path",
        "target_index",
        "prediction_index",
        "confidence",
        "margin",
        "correct",
        "review_flags",
        "max_background_mass",
        "max_border_mass",
        "background_gray_drop",
        "background_blur_drop",
        "object_desaturate_drop",
        "center_occlusion_drop",
        "cls_register_heatmap_similarity",
        "register_attention_entropy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            focus = {}
            register_attention = {}
            viz = case.get("viz", {})
            if isinstance(viz, dict):
                focus = viz.get("heatmap_focus", {}) if isinstance(viz.get("heatmap_focus", {}), dict) else {}
                register_attention = (
                    viz.get("register_attention", {}) if isinstance(viz.get("register_attention", {}), dict) else {}
                )
            background_values = []
            border_values = []
            for metrics in focus.values() if isinstance(focus, dict) else []:
                if isinstance(metrics, dict):
                    background_values.append(float(metrics.get("background_mass", 0.0)))
                    border_values.append(float(metrics.get("border_mass", 0.0)))
            robustness = case.get("robustness", {}) if isinstance(case.get("robustness", {}), dict) else {}

            def drop(name: str) -> float:
                item = robustness.get(name, {}) if isinstance(robustness, dict) else {}
                return float(item.get("original_prediction_drop", 0.0)) if isinstance(item, dict) else 0.0

            writer.writerow(
                {
                    "sample_index": case.get("sample_index", ""),
                    "image_path": case.get("image_path", ""),
                    "target_index": case.get("target_index", ""),
                    "prediction_index": case.get("prediction_index", ""),
                    "confidence": case.get("confidence", ""),
                    "margin": case.get("margin", ""),
                    "correct": case.get("correct", ""),
                    "review_flags": ";".join(str(flag) for flag in case.get("review_flags", [])),
                    "max_background_mass": max(background_values) if background_values else 0.0,
                    "max_border_mass": max(border_values) if border_values else 0.0,
                    "background_gray_drop": drop("background_gray"),
                    "background_blur_drop": drop("background_blur"),
                    "object_desaturate_drop": drop("object_desaturate"),
                    "center_occlusion_drop": drop("center_occlusion"),
                    "cls_register_heatmap_similarity": register_attention.get("cls_register_heatmap_similarity", ""),
                    "register_attention_entropy": register_attention.get("register_attention_entropy", ""),
                }
            )


def _write_markdown(
    path: Path,
    *,
    checkpoint: Path,
    data: Path,
    split: str,
    cases: Sequence[Dict[str, object]],
    confusion_pairs: Sequence[Dict[str, int]],
    xai_metrics: Dict[str, object],
) -> None:
    focus_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        viz = case.get("viz", {})
        if not isinstance(viz, dict):
            continue
        focus = viz.get("heatmap_focus", {})
        if not isinstance(focus, dict):
            continue
        for method_name, metrics in focus.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name in ("foreground_mass", "background_mass", "border_mass", "entropy"):
                if metric_name in metrics:
                    focus_values[str(method_name)][metric_name].append(float(metrics[metric_name]))

    lines = [
        "# XAI audit",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Data: `{data}`",
        f"- Split: `{split}`",
        f"- Selected cases: `{len(cases)}`",
        "",
        "## Top confusion pairs",
        "",
        "| True | Pred | Count |",
        "|---:|---:|---:|",
    ]
    for pair in confusion_pairs[:10]:
        lines.append(
            f"| {pair['target_index']} | {pair['prediction_index']} | {pair['count']} |"
        )
    if focus_values:
        lines.extend(["", "## Heatmap Focus Summary", "", "| Method | Foreground mass | Background mass | Border mass | Entropy |", "|---|---:|---:|---:|---:|"])
        for method_name, values in sorted(focus_values.items()):
            def mean_metric(key: str) -> float:
                metric_values = values.get(key, [])
                return sum(metric_values) / max(1, len(metric_values))

            lines.append(
                "| {method} | {fg:.4f} | {bg:.4f} | {border:.4f} | {entropy:.4f} |".format(
                    method=method_name,
                    fg=mean_metric("foreground_mass"),
                    bg=mean_metric("background_mass"),
                    border=mean_metric("border_mass"),
                    entropy=mean_metric("entropy"),
                )
            )
    flag_counts = xai_metrics.get("review_flag_counts", {})
    if isinstance(flag_counts, dict) and flag_counts:
        lines.extend(["", "## Review Flags", "", "| Flag | Count |", "|---|---:|"])
        for flag, count in sorted(flag_counts.items(), key=lambda item: str(item[0])):
            lines.append(f"| {flag} | {count} |")
    robustness_mean = xai_metrics.get("robustness_mean", {})
    if isinstance(robustness_mean, dict) and robustness_mean:
        lines.extend(
            [
                "",
                "## Robustness Probe Summary",
                "",
                "| Probe | Pred prob drop | Target prob drop |",
                "|---|---:|---:|",
            ]
        )
        for probe, metrics in sorted(robustness_mean.items()):
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| {probe} | {pred:.4f} | {target:.4f} |".format(
                    probe=probe,
                    pred=float(metrics.get("original_prediction_drop", 0.0)),
                    target=float(metrics.get("target_probability_drop", 0.0)),
                )
            )
    register_mean = xai_metrics.get("register_attention_mean", {})
    if isinstance(register_mean, dict) and register_mean:
        lines.extend(["", "## Register Diagnostics", "", "| Metric | Mean |", "|---|---:|"])
        for metric, value in sorted(register_mean.items()):
            lines.append(f"| {metric} | {float(value):.4f} |")
    lines.extend(["", "## Selected cases", "", "| Case | Reason | True | Pred | Conf | Margin |", "|---|---|---|---|---:|---:|"])
    for case in cases:
        case_dir = Path(str(case.get("case_dir", ""))).name
        lines.append(
            "| `{case}` | {reason} | {true} | {pred} | {conf:.4f} | {margin:.4f} |".format(
                case=case_dir,
                reason=case.get("audit_reason", ""),
                true=case.get("target_name", case.get("target_index", "")),
                pred=case.get("prediction_name", case.get("prediction_index", "")),
                conf=float(case.get("confidence", 0.0)),
                margin=float(case.get("margin", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    dataset = _build_dataset(
        data_yaml=args.data,
        split=args.split,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes,
        checkpoint=checkpoint,
    )
    amp = bool(not args.disable_amp)
    records = _collect_predictions(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=amp,
    )
    selected_cases = _select_cases(
        records,
        num_classes=len(class_names),
        max_cases=max(1, int(args.max_cases)),
        mistake_cases=args.mistake_cases,
        low_confidence_cases=args.low_confidence_cases,
        close_margin_cases=args.close_margin_cases,
        per_class_cases=args.per_class_cases,
    )
    output_dir = args.output_dir or args.checkpoint.resolve().parent.parent / f"xai_audit_{args.split}"
    output_dir = ensure_dir(output_dir)
    attention_depth = count_attention_layers(model)
    layer_index = resolve_layer_index(args.layer, attention_depth) if attention_depth > 0 else 0

    enriched_cases = []
    for order, case in enumerate(selected_cases, start=1):
        case_dir = ensure_dir(
            output_dir
            / f"case_{order:03d}_t{case['target_index']}_p{case['prediction_index']}_{case['audit_reason']}"
        )
        crop_image, tensor = prepare_image_and_tensor(
            checkpoint=checkpoint,
            image_path=Path(str(case["image_path"])),
            device=device,
        )
        viz = analyze_tensor(
            model=model,
            class_names=class_names,
            crop_image=crop_image,
            tensor=tensor,
            layer_index=layer_index,
            head_reduction=args.head_reduction,
            alpha=args.alpha,
            top_k=args.top_k,
            output_dir=case_dir,
            method=args.method,
            target_class=int(case["prediction_index"]),
            query_tokens=args.query_tokens,
            feature_source=args.feature_source,
            rollout_start_layer=args.rollout_start_layer,
        )
        enriched = dict(case)
        enriched["case_dir"] = str(case_dir.resolve())
        enriched["viz"] = viz
        if args.robustness_probes:
            enriched["robustness"] = _run_robustness_probes(
                model=model,
                checkpoint=checkpoint,
                crop_image=crop_image,
                original_prediction=int(case["prediction_index"]),
                target_index=int(case["target_index"]),
                device=device,
                amp=amp,
            )
        enriched["review_flags"] = _build_review_flags(
            enriched,
            high_confidence=args.review_high_confidence,
            background_threshold=args.review_background_threshold,
            border_threshold=args.review_border_threshold,
            shortcut_drop_threshold=args.shortcut_drop_threshold,
        )
        json_dump(case_dir / "case.json", to_serializable(enriched))
        enriched_cases.append(enriched)

    pairs = _confusion_pairs(records, len(class_names))
    xai_metrics = _aggregate_xai_metrics(enriched_cases)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "samples": len(records),
        "selected_cases": len(enriched_cases),
        "attention_depth": attention_depth,
        "layer_index": layer_index,
        "query_tokens": args.query_tokens,
        "method": args.method,
        "feature_source": args.feature_source,
        "rollout_start_layer": int(args.rollout_start_layer),
        "robustness_probes": bool(args.robustness_probes),
        "class_names": list(class_names),
        "confusion_pairs": pairs,
        "xai_metrics": xai_metrics,
        "cases": enriched_cases,
    }
    json_dump(output_dir / "xai_audit_summary.json", to_serializable(summary))
    json_dump(output_dir / "xai_metrics.json", to_serializable(xai_metrics))
    _write_case_table(output_dir / "xai_cases.csv", enriched_cases)
    _write_review_manifest(output_dir / "review_manifest.csv", enriched_cases)
    _write_markdown(
        output_dir / "xai_audit.md",
        checkpoint=args.checkpoint,
        data=args.data,
        split=args.split,
        cases=enriched_cases,
        confusion_pairs=pairs,
        xai_metrics=xai_metrics,
    )
    print(
        {
            "output_dir": str(output_dir.resolve()),
            "samples": len(records),
            "selected_cases": len(enriched_cases),
            "top_confusions": pairs[:5],
        }
    )


if __name__ == "__main__":
    main()
