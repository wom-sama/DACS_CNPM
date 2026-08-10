from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _collate_classification,
    _resolve_device,
)
from trkh.tools.probe_patch_evidence_mil import _tensor_metadata


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether TRKH global/CLS-like features score bbox/object patch "
            "tokens above background tokens. This is diagnostic-only: it writes "
            "CSV/JSON/README artifacts, refuses test unless explicitly allowed, "
            "and never writes trainable manifests."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--bbox-token-prior-source",
        type=str,
        default="bbox",
        choices=("bbox", "crop_bbox"),
    )
    parser.add_argument(
        "--query-source",
        type=str,
        default="pooled",
        choices=("pooled", "cls", "cls_register_mean"),
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.20)
    parser.add_argument("--background-threshold", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, nargs="*", default=[1, 3, 5])
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    if not row_list:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in row_list:
        for key in row:
            key_text = str(key)
            if key_text not in fieldnames:
                fieldnames.append(key_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _select_bbox_token_prior(
    *,
    metadata: Mapping[str, object],
    source: str,
    device: torch.device,
) -> Optional[Tensor]:
    bbox = _tensor_metadata(metadata, "bbox", device=device, dtype=torch.float32)
    crop_bbox = _tensor_metadata(metadata, "crop_bbox", device=device, dtype=torch.float32)
    if str(source) == "crop_bbox" and torch.is_tensor(crop_bbox):
        return crop_bbox
    return bbox


def _query_feature(features: Mapping[str, object], source: str) -> Tensor:
    source_text = str(source)
    if source_text == "pooled":
        query = features.get("pooled")
        if torch.is_tensor(query) and query.ndim == 2:
            return query
    if source_text == "cls":
        query = features.get("cls")
        if torch.is_tensor(query) and query.ndim == 2:
            return query
    if source_text == "cls_register_mean":
        cls = features.get("cls")
        registers = features.get("registers")
        if torch.is_tensor(cls) and cls.ndim == 2:
            parts = [cls.unsqueeze(1)]
            if torch.is_tensor(registers) and registers.ndim == 3 and int(registers.size(1)) > 0:
                parts.append(registers)
            return torch.cat(parts, dim=1).mean(dim=1)
    raise ValueError(f"Cannot build query feature from source={source_text!r}.")


def _valid_token_mask(features: Mapping[str, object], prior: Tensor) -> Tensor:
    key_padding_mask = features.get("memory_key_padding_mask")
    if torch.is_tensor(key_padding_mask) and tuple(key_padding_mask.shape) == tuple(prior.shape):
        valid = ~key_padding_mask.to(device=prior.device, dtype=torch.bool)
    else:
        valid = torch.ones_like(prior, dtype=torch.bool)
    empty = valid.sum(dim=1) <= 0
    if bool(empty.any().item()):
        valid = valid.clone()
        valid[empty] = True
    return valid


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(device=values.device, dtype=values.dtype)
    count = mask_float.sum(dim=1).clamp_min(1.0)
    return (values * mask_float).sum(dim=1) / count


def _compute_patch_score_statistics(
    *,
    patches: Tensor,
    query: Tensor,
    bbox_prior: Tensor,
    valid_mask: Optional[Tensor] = None,
    foreground_threshold: float = 0.20,
    background_threshold: float = 0.20,
    top_k_values: Sequence[int] = (1, 3, 5),
) -> Dict[str, Tensor]:
    if patches.ndim != 3:
        raise ValueError("patches must be [B, N, D].")
    if query.ndim != 2 or int(query.size(0)) != int(patches.size(0)):
        raise ValueError("query must be [B, D] and match patches batch size.")
    if tuple(bbox_prior.shape) != tuple(patches.shape[:2]):
        raise ValueError("bbox_prior must match patches [B, N].")
    if int(query.size(1)) != int(patches.size(2)):
        raise ValueError("query and patch embedding dimensions must match.")

    prior = bbox_prior.float().clamp(0.0, 1.0)
    valid = (
        valid_mask.to(device=prior.device, dtype=torch.bool)
        if torch.is_tensor(valid_mask)
        else torch.ones_like(prior, dtype=torch.bool)
    )
    if tuple(valid.shape) != tuple(prior.shape):
        raise ValueError("valid_mask must match bbox_prior shape.")
    empty_valid = valid.sum(dim=1) <= 0
    if bool(empty_valid.any().item()):
        valid = valid.clone()
        valid[empty_valid] = True

    scores = (F.normalize(patches.float(), dim=-1) * F.normalize(query.float(), dim=-1).unsqueeze(1)).sum(dim=-1)
    foreground = (prior >= float(foreground_threshold)) & valid
    background = (prior <= float(background_threshold)) & valid
    stats: Dict[str, Tensor] = {
        "scores": scores,
        "valid_token_count": valid.sum(dim=1).to(dtype=torch.float32),
        "foreground_token_count": foreground.sum(dim=1).to(dtype=torch.float32),
        "background_token_count": background.sum(dim=1).to(dtype=torch.float32),
        "foreground_score_mean": _masked_mean(scores, foreground),
        "background_score_mean": _masked_mean(scores, background),
    }
    stats["foreground_minus_background_score"] = (
        stats["foreground_score_mean"] - stats["background_score_mean"]
    )

    masked_scores = scores.masked_fill(~valid, -torch.inf)
    max_requested_k = max([1, *[int(value) for value in top_k_values if int(value) > 0]])
    top_count = min(max_requested_k, int(scores.size(1)))
    top_scores, top_indices = torch.topk(masked_scores, k=top_count, dim=1, largest=True, sorted=True)
    top_prior = prior.gather(1, top_indices)
    top_valid = valid.gather(1, top_indices) & torch.isfinite(top_scores)
    effective_valid_counts = valid.sum(dim=1).clamp_min(1)
    for raw_k in top_k_values:
        k_value = int(raw_k)
        if k_value <= 0:
            continue
        local_k = min(k_value, top_count)
        selected_prior = top_prior[:, :local_k]
        selected_scores = top_scores[:, :local_k]
        selected_valid = top_valid[:, :local_k]
        denominator = torch.minimum(
            effective_valid_counts,
            torch.full_like(effective_valid_counts, local_k),
        ).to(dtype=torch.float32).clamp_min(1.0)
        selected_valid_float = selected_valid.to(dtype=torch.float32)
        stats[f"top{k_value}_foreground_fraction"] = (
            ((selected_prior >= float(foreground_threshold)) & selected_valid).to(dtype=torch.float32).sum(dim=1)
            / denominator
        )
        stats[f"top{k_value}_background_fraction"] = (
            ((selected_prior <= float(background_threshold)) & selected_valid).to(dtype=torch.float32).sum(dim=1)
            / denominator
        )
        stats[f"top{k_value}_prior_mean"] = (selected_prior * selected_valid_float).sum(dim=1) / denominator
        safe_scores = torch.where(selected_valid, selected_scores, torch.zeros_like(selected_scores))
        stats[f"top{k_value}_score_mean"] = safe_scores.sum(dim=1) / denominator

    stats["top1_prior"] = top_prior[:, 0]
    stats["top1_score"] = torch.where(top_valid[:, 0], top_scores[:, 0], torch.zeros_like(top_scores[:, 0]))
    stats["top1_point_in_box"] = ((top_prior[:, 0] >= float(foreground_threshold)) & top_valid[:, 0]).to(
        dtype=torch.float32
    )
    return stats


def _confidence_margin(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [B, C].")
    if int(probabilities.size(1)) <= 1:
        return torch.zeros((probabilities.size(0),), device=probabilities.device, dtype=probabilities.dtype)
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _sample_indices_from_metadata(
    metadata: Mapping[str, object],
    *,
    fallback_start: int,
    batch_size: int,
) -> np.ndarray:
    sample_index = metadata.get("sample_index")
    if torch.is_tensor(sample_index) and int(sample_index.numel()) == int(batch_size):
        return sample_index.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.arange(fallback_start, fallback_start + batch_size, dtype=np.int64)


def _extract_patch_score_rows(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    split: str,
    class_names: Sequence[str],
    batch_size: int,
    workers: int,
    amp: bool,
    bbox_token_prior_source: str,
    query_source: str,
    foreground_threshold: float,
    background_threshold: float,
    top_k_values: Sequence[int],
) -> Dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    rows: List[Dict[str, object]] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in sample_paths_fn()] if callable(sample_paths_fn) else []
    seen_samples = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc=f"patch-score-pib-{split}", dynamic_ncols=True, leave=False)
        for images, labels, metadata in iterator:
            if not isinstance(metadata, Mapping):
                metadata = {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batch_size_value = int(images.shape[0])
            raw_paths = metadata.get("paths", [])
            batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
            if dataset_paths and (len(batch_paths) != batch_size_value or not any(path.strip() for path in batch_paths)):
                batch_paths = dataset_paths[seen_samples : seen_samples + batch_size_value]
            fallback_start = seen_samples
            seen_samples += batch_size_value
            sample_indices = _sample_indices_from_metadata(
                metadata,
                fallback_start=fallback_start,
                batch_size=batch_size_value,
            )
            bbox_prior_input = _select_bbox_token_prior(
                metadata=metadata,
                source=bbox_token_prior_source,
                device=device,
            )
            image_mask = _tensor_metadata(metadata, "image_mask", device=device, dtype=torch.bool)
            with autocast_context(device, amp):
                if not hasattr(model, "forward_features"):
                    raise TypeError("Patch-score audit requires model.forward_features().")
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=bbox_prior_input,
                )
                if torch.is_tensor(bbox_prior_input):
                    features["bbox"] = bbox_prior_input[:, :4] if bbox_prior_input.ndim == 2 else bbox_prior_input
                if hasattr(model, "forward_heads"):
                    output = model.forward_heads(features)
                else:
                    output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(output)
                probabilities = logits.float().softmax(dim=1)
                patches = features.get("patches")
                patch_prior = features.get("patch_bbox_prior")
                if not torch.is_tensor(patches) or patches.ndim != 3:
                    raise ValueError("Model features do not contain patch tokens.")
                if not torch.is_tensor(patch_prior) or tuple(patch_prior.shape) != tuple(patches.shape[:2]):
                    raise ValueError("Model features do not contain patch_bbox_prior matching patches.")
                query = _query_feature(features, query_source)
                stats = _compute_patch_score_statistics(
                    patches=patches,
                    query=query,
                    bbox_prior=patch_prior,
                    valid_mask=_valid_token_mask(features, patch_prior),
                    foreground_threshold=float(foreground_threshold),
                    background_threshold=float(background_threshold),
                    top_k_values=top_k_values,
                )
            predictions = probabilities.argmax(dim=1)
            margins = _confidence_margin(probabilities)
            probability_batches.append(probabilities.detach().cpu().numpy().astype(np.float32, copy=False))
            label_batches.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            prediction_batches.append(predictions.detach().cpu().numpy().astype(np.int64, copy=False))
            stats_cpu = {
                key: value.detach().float().cpu().numpy()
                for key, value in stats.items()
                if torch.is_tensor(value) and value.ndim == 1
            }
            probs_cpu = probabilities.detach().float().cpu().numpy()
            preds_cpu = predictions.detach().cpu().numpy().astype(np.int64, copy=False)
            labels_cpu = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            margins_cpu = margins.detach().float().cpu().numpy()
            confidences_cpu = probs_cpu.max(axis=1)
            for row_offset, target_index in enumerate(labels_cpu):
                pred_index = int(preds_cpu[row_offset])
                row: Dict[str, object] = {
                    "split": str(split),
                    "row_index": int(fallback_start + row_offset),
                    "sample_index": int(sample_indices[row_offset]) if row_offset < len(sample_indices) else int(fallback_start + row_offset),
                    "image_path": str(batch_paths[row_offset]) if row_offset < len(batch_paths) else "",
                    "target_index": int(target_index),
                    "target_name": str(class_names[int(target_index)]) if 0 <= int(target_index) < len(class_names) else str(target_index),
                    "pred_index": pred_index,
                    "pred_name": str(class_names[pred_index]) if 0 <= pred_index < len(class_names) else str(pred_index),
                    "transition": f"{int(target_index)}->{pred_index}",
                    "correct": int(pred_index == int(target_index)),
                    "confidence": float(confidences_cpu[row_offset]),
                    "margin": float(margins_cpu[row_offset]),
                    "prob_true": float(probs_cpu[row_offset, int(target_index)]),
                    "prob_pred": float(probs_cpu[row_offset, pred_index]),
                    "prob_class1": float(probs_cpu[row_offset, 1]) if probs_cpu.shape[1] > 1 else 0.0,
                }
                for key, values in stats_cpu.items():
                    row[key] = float(values[row_offset])
                rows.append(row)
    labels_array = (
        np.concatenate(label_batches, axis=0).astype(np.int64, copy=False)
        if label_batches
        else np.zeros((0,), dtype=np.int64)
    )
    predictions_array = (
        np.concatenate(prediction_batches, axis=0).astype(np.int64, copy=False)
        if prediction_batches
        else np.zeros((0,), dtype=np.int64)
    )
    probabilities_array = (
        np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)
        if probability_batches
        else np.zeros((0, len(class_names)), dtype=np.float32)
    )
    return {
        "rows": rows,
        "labels": labels_array,
        "predictions": predictions_array,
        "probabilities": probabilities_array,
    }


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([_float_value(row.get(key)) for row in rows]))


def _summarize_group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_name: str,
    group_value: str,
    top_k_values: Sequence[int],
) -> Dict[str, object]:
    support = len(rows)
    summary: Dict[str, object] = {
        "group_name": str(group_name),
        "group_value": str(group_value),
        "support": int(support),
        "accuracy": _mean(rows, "correct"),
        "confidence_mean": _mean(rows, "confidence"),
        "margin_mean": _mean(rows, "margin"),
        "top1_point_in_box_mean": _mean(rows, "top1_point_in_box"),
        "top1_prior_mean": _mean(rows, "top1_prior"),
        "foreground_score_mean": _mean(rows, "foreground_score_mean"),
        "background_score_mean": _mean(rows, "background_score_mean"),
        "foreground_minus_background_score_mean": _mean(rows, "foreground_minus_background_score"),
        "foreground_token_count_mean": _mean(rows, "foreground_token_count"),
        "background_token_count_mean": _mean(rows, "background_token_count"),
        "valid_token_count_mean": _mean(rows, "valid_token_count"),
    }
    for raw_k in top_k_values:
        k_value = int(raw_k)
        if k_value <= 0:
            continue
        summary[f"top{k_value}_foreground_fraction_mean"] = _mean(
            rows, f"top{k_value}_foreground_fraction"
        )
        summary[f"top{k_value}_background_fraction_mean"] = _mean(
            rows, f"top{k_value}_background_fraction"
        )
        summary[f"top{k_value}_prior_mean"] = _mean(rows, f"top{k_value}_prior_mean")
        summary[f"top{k_value}_score_mean"] = _mean(rows, f"top{k_value}_score_mean")
    return summary


def _build_group_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    class_names: Sequence[str],
    top_k_values: Sequence[int],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    summaries.append(
        _summarize_group_rows(rows, group_name="all", group_value="all", top_k_values=top_k_values)
    )
    for correct_value, group_value in [(1, "correct"), (0, "incorrect")]:
        group_rows = [row for row in rows if int(_float_value(row.get("correct"))) == correct_value]
        summaries.append(
            _summarize_group_rows(
                group_rows,
                group_name="correctness",
                group_value=group_value,
                top_k_values=top_k_values,
            )
        )
    for class_index, class_name in enumerate(class_names):
        target_rows = [row for row in rows if int(_float_value(row.get("target_index"), -1)) == class_index]
        pred_rows = [row for row in rows if int(_float_value(row.get("pred_index"), -1)) == class_index]
        summaries.append(
            _summarize_group_rows(
                target_rows,
                group_name="target_class",
                group_value=f"{class_index}:{class_name}",
                top_k_values=top_k_values,
            )
        )
        summaries.append(
            _summarize_group_rows(
                pred_rows,
                group_name="pred_class",
                group_value=f"{class_index}:{class_name}",
                top_k_values=top_k_values,
            )
        )
    transitions = sorted({str(row.get("transition", "")) for row in rows if str(row.get("transition", "")).strip()})
    for transition in transitions:
        transition_rows = [row for row in rows if str(row.get("transition")) == transition]
        summaries.append(
            _summarize_group_rows(
                transition_rows,
                group_name="transition",
                group_value=transition,
                top_k_values=top_k_values,
            )
        )
    return summaries


def _risk_transition_rows(group_summaries: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    transition_rows = [
        dict(row)
        for row in group_summaries
        if str(row.get("group_name")) == "transition"
        and str(row.get("group_value", "")).split("->", 1)[0]
        != str(row.get("group_value", "")).split("->", 1)[-1]
    ]
    transition_rows.sort(
        key=lambda row: (
            int(row.get("support", 0) or 0),
            -float(row.get("top1_point_in_box_mean", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return transition_rows


def _interpret_signal(
    *,
    group_summaries: Sequence[Mapping[str, object]],
    top_k_values: Sequence[int],
) -> Dict[str, object]:
    by_key = {
        (str(row.get("group_name")), str(row.get("group_value"))): row
        for row in group_summaries
    }
    all_rows = by_key.get(("all", "all"), {})
    incorrect = by_key.get(("correctness", "incorrect"), {})
    class1_target = {}
    for (group_name, group_value), row in by_key.items():
        if group_name == "target_class" and (group_value == "1" or group_value.startswith("1:")):
            class1_target = row
            break
    max_top_k = max([1, *[int(value) for value in top_k_values if int(value) > 0]])
    incorrect_pib = float(incorrect.get("top1_point_in_box_mean", 0.0) or 0.0)
    incorrect_topk_fg = float(incorrect.get(f"top{max_top_k}_foreground_fraction_mean", 0.0) or 0.0)
    incorrect_fg_minus_bg = float(incorrect.get("foreground_minus_background_score_mean", 0.0) or 0.0)
    class1_pib = float(class1_target.get("top1_point_in_box_mean", 0.0) or 0.0)
    background_shortcut_suspected = bool(
        incorrect_pib < 0.55 or incorrect_topk_fg < 0.45 or incorrect_fg_minus_bg <= 0.0
    )
    return {
        "all_top1_point_in_box_mean": float(all_rows.get("top1_point_in_box_mean", 0.0) or 0.0),
        "incorrect_top1_point_in_box_mean": incorrect_pib,
        f"incorrect_top{max_top_k}_foreground_fraction_mean": incorrect_topk_fg,
        "incorrect_foreground_minus_background_score_mean": incorrect_fg_minus_bg,
        "target_class1_top1_point_in_box_mean": class1_pib,
        "background_shortcut_suspected": background_shortcut_suspected,
        "training_permission": False,
        "recommended_use": (
            "If incorrect/class-1 rows are already foreground-high, avoid another "
            "background-filter smoke and seek a surface/boundary representation or "
            "fold-safe reliability target. If they are foreground-low, a patch-score "
            "foreground alignment loss may be worth smoke-gating."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    split = str(args.split).strip().lower()
    if split == "test" and not bool(args.allow_test):
        raise ValueError("Patch-score PiB audit refuses test unless --allow-test is set for final audit only.")
    top_k_values = sorted({int(value) for value in args.top_k if int(value) > 0})
    if not top_k_values:
        raise ValueError("--top-k must contain at least one positive integer.")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()
    dataset, class_names = _build_dataset(
        data_yaml=Path(args.data),
        split=split,
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_samples),
    )
    if int(args.expected_num_classes) > 0 and len(class_names) != int(args.expected_num_classes):
        raise ValueError("Class count in data.yaml/checkpoint does not match --expected-num-classes.")

    start_time = time.perf_counter()
    payload = _extract_patch_score_rows(
        model=model,
        dataset=dataset,
        device=device,
        split=split,
        class_names=class_names,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        amp=bool(args.amp),
        bbox_token_prior_source=str(args.bbox_token_prior_source),
        query_source=str(args.query_source),
        foreground_threshold=float(args.foreground_threshold),
        background_threshold=float(args.background_threshold),
        top_k_values=top_k_values,
    )
    rows = list(payload["rows"])
    group_summaries = _build_group_summaries(rows, class_names=class_names, top_k_values=top_k_values)
    risk_rows = _risk_transition_rows(group_summaries)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    predictions = np.asarray(payload["predictions"], dtype=np.int64)
    metrics = _classification_metrics(labels, predictions, list(class_names))
    summary: Dict[str, object] = {
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "support": int(labels.size),
        "class_names": list(class_names),
        "metrics": metrics,
        "bbox_token_prior_source": str(args.bbox_token_prior_source),
        "query_source": str(args.query_source),
        "foreground_threshold": float(args.foreground_threshold),
        "background_threshold": float(args.background_threshold),
        "top_k_values": top_k_values,
        "raw_dataset_touched": False,
        "test_accessed": bool(split == "test"),
        "trainable_manifest_written": False,
        "diagnostic_only": True,
        "group_summaries": group_summaries,
        "risk_transitions": risk_rows[:20],
        "interpretation": _interpret_signal(group_summaries=group_summaries, top_k_values=top_k_values),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    _write_rows_csv(output_dir / "patch_score_rows.csv", rows)
    _write_rows_csv(output_dir / "patch_score_group_summary.csv", group_summaries)
    _write_rows_csv(output_dir / "patch_score_risk_transitions.csv", risk_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    top_transition_lines = []
    for row in risk_rows[:8]:
        top_transition_lines.append(
            f"- `{row.get('group_value')}`: support `{row.get('support')}`, "
            f"top1 PiB `{float(row.get('top1_point_in_box_mean', 0.0) or 0.0):.4f}`, "
            f"fg-bg score `{float(row.get('foreground_minus_background_score_mean', 0.0) or 0.0):.4f}`"
        )
    readme_lines = [
        "# Patch Score Point-in-Box Audit",
        "",
        f"- Data: `{Path(args.data).resolve()}`",
        f"- Checkpoint: `{Path(args.checkpoint).resolve()}`",
        f"- Split: `{split}`",
        f"- Support: `{int(labels.size)}`",
        f"- Query source: `{args.query_source}`",
        f"- Bbox prior source: `{args.bbox_token_prior_source}`",
        f"- Test accessed: `{bool(split == 'test')}`",
        f"- Trainable manifest written: `false`",
        f"- Background shortcut suspected: `{summary['interpretation']['background_shortcut_suspected']}`",
        "",
        "## Top Error Transitions",
        "",
        *(top_transition_lines or ["- None"]),
        "",
        "Use this artifact only to decide whether the next smoke should target foreground alignment or surface/boundary representation. It is not a training manifest.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(
        {
            "output_dir": str(output_dir),
            "support": int(labels.size),
            "background_shortcut_suspected": summary["interpretation"]["background_shortcut_suspected"],
            "test_accessed": bool(split == "test"),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
