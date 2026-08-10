from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.utils import autocast_context
from trkh.evaluation.evaluate import extract_detection_from_model_output
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features
from trkh.tools.audit_patch_score_pib import (
    _confidence_margin,
    _sample_indices_from_metadata,
    _select_bbox_token_prior,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _collate_classification,
    _resolve_device,
)
from trkh.tools.probe_patch_evidence_mil import _tensor_metadata


def _variant_key(value: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    return "_".join(part for part in key.split("_") if part) or "variant"


def parse_variant_spec(text: str) -> Dict[str, object]:
    value = str(text or "").strip()
    if not value:
        raise ValueError("Locality-attention variant must not be empty.")
    name, separator, raw_spec = value.partition("=")
    if not separator:
        raise ValueError("Variant must use name=temperature_multiplier,self_suppression.")
    parts = [part.strip() for part in raw_spec.split(",")]
    if len(parts) != 2:
        raise ValueError("Variant must contain temperature_multiplier,self_suppression.")
    temperature_multiplier = float(parts[0])
    self_suppression = float(parts[1])
    if not 0.0 < temperature_multiplier <= 4.0:
        raise ValueError("temperature_multiplier must be in (0, 4].")
    if not 0.0 <= self_suppression <= 1.0:
        raise ValueError("self_suppression must be in [0, 1].")
    return {
        "name": _variant_key(name),
        "temperature_multiplier": temperature_multiplier,
        "self_suppression": self_suppression,
    }


def parse_variant_specs(values: Optional[Sequence[str]]) -> List[Dict[str, object]]:
    specs = list(values or []) or [
        "baseline=1.0,0.0",
        "smooth075=0.75,0.0",
        "sharp125=1.25,0.0",
        "diag50=1.0,0.5",
        "lsa125_diag50=1.25,0.5",
        "lsa125_diag100=1.25,1.0",
    ]
    variants: List[Dict[str, object]] = []
    seen: set[str] = set()
    for spec in specs:
        variant = parse_variant_spec(spec)
        name = str(variant["name"])
        if name in seen:
            raise ValueError(f"Duplicate locality-attention variant: {name}")
        seen.add(name)
        variants.append(variant)
    identity_index = next(
        (
            index
            for index, variant in enumerate(variants)
            if float(variant["temperature_multiplier"]) == 1.0
            and float(variant["self_suppression"]) == 0.0
        ),
        None,
    )
    if identity_index is None:
        variants.insert(0, parse_variant_spec("baseline=1.0,0.0"))
    elif identity_index != 0:
        variants.insert(0, variants.pop(identity_index))
    return variants


def parse_layer_numbers(text: str, *, block_count: int) -> List[int]:
    layers = sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})
    if not layers:
        raise ValueError("At least one attention layer is required.")
    invalid = [layer for layer in layers if layer < 1 or layer > int(block_count)]
    if invalid:
        raise ValueError(f"Attention layers out of range 1..{block_count}: {invalid}")
    return layers


def transform_patch_attention(
    attention: Tensor,
    *,
    prefix_count: int,
    temperature_multiplier: float,
    self_suppression: float,
) -> Tensor:
    """Apply an LSA-style patch-only transform while preserving patch row mass."""
    if attention.ndim != 4 or int(attention.size(-2)) != int(attention.size(-1)):
        raise ValueError("attention must be square [B, H, N, N].")
    token_count = int(attention.size(-1))
    prefix_count = max(0, min(int(prefix_count), token_count))
    patch_count = token_count - prefix_count
    temperature_multiplier = float(temperature_multiplier)
    self_suppression = float(self_suppression)
    if not 0.0 < temperature_multiplier <= 4.0:
        raise ValueError("temperature_multiplier must be in (0, 4].")
    if not 0.0 <= self_suppression <= 1.0:
        raise ValueError("self_suppression must be in [0, 1].")
    if patch_count <= 1 or (
        temperature_multiplier == 1.0 and self_suppression == 0.0
    ):
        return attention

    patch_attention = attention[:, :, prefix_count:, prefix_count:]
    patch_mass = patch_attention.float().sum(dim=-1, keepdim=True)
    conditional = patch_attention.float() / patch_mass.clamp_min(1e-12)
    if temperature_multiplier != 1.0:
        conditional = conditional.clamp_min(1e-12).pow(temperature_multiplier)
    if self_suppression > 0.0:
        diagonal_scale = 1.0 - self_suppression
        diagonal = torch.eye(
            patch_count,
            device=attention.device,
            dtype=conditional.dtype,
        ).view(1, 1, patch_count, patch_count)
        conditional = conditional * (1.0 - diagonal + diagonal * diagonal_scale)
    conditional_sum = conditional.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(conditional, 1.0 / float(patch_count))
    conditional = torch.where(
        conditional_sum > 1e-12,
        conditional / conditional_sum.clamp_min(1e-12),
        fallback,
    )
    transformed_patch = (conditional * patch_mass).to(dtype=attention.dtype)
    if prefix_count == 0:
        return transformed_patch
    patch_rows = torch.cat(
        (attention[:, :, prefix_count:, :prefix_count], transformed_patch),
        dim=-1,
    )
    return torch.cat((attention[:, :, :prefix_count, :], patch_rows), dim=-2)


def patch_attention_statistics(attention: Tensor, *, prefix_count: int) -> Dict[str, Tensor]:
    if attention.ndim != 4 or int(attention.size(-2)) != int(attention.size(-1)):
        raise ValueError("attention must be square [B, H, N, N].")
    token_count = int(attention.size(-1))
    prefix_count = max(0, min(int(prefix_count), token_count))
    patch_count = token_count - prefix_count
    if patch_count <= 0:
        raise ValueError("attention contains no patch tokens.")

    patch = attention[:, :, prefix_count:, prefix_count:].float()
    patch_mass = patch.sum(dim=-1)
    conditional = patch / patch_mass.unsqueeze(-1).clamp_min(1e-12)
    diagonal = torch.diagonal(conditional, dim1=-2, dim2=-1)
    absolute_diagonal = torch.diagonal(patch, dim1=-2, dim2=-1)
    entropy = -(conditional * conditional.clamp_min(1e-12).log()).sum(dim=-1)
    if patch_count > 1:
        entropy = entropy / math.log(float(patch_count))
    else:
        entropy = torch.zeros_like(entropy)
    top_values, top_indices = conditional.max(dim=-1)
    self_indices = torch.arange(patch_count, device=attention.device).view(1, 1, -1)
    self_top1 = (top_indices == self_indices).to(dtype=torch.float32)
    return {
        "patch_mass": patch_mass.mean(dim=(1, 2)).detach(),
        "absolute_self_mass": absolute_diagonal.mean(dim=(1, 2)).detach(),
        "conditional_self_fraction": diagonal.mean(dim=(1, 2)).detach(),
        "normalized_entropy": entropy.mean(dim=(1, 2)).detach(),
        "top1_mass": top_values.mean(dim=(1, 2)).detach(),
        "self_top1_fraction": self_top1.mean(dim=(1, 2)).detach(),
    }


class _AttentionVariantHooks:
    def __init__(
        self,
        *,
        backbone: nn.Module,
        layers: Sequence[int],
        prefix_count: int,
        temperature_multiplier: float,
        self_suppression: float,
    ) -> None:
        self.backbone = backbone
        self.layers = [int(layer) for layer in layers]
        self.prefix_count = int(prefix_count)
        self.temperature_multiplier = float(temperature_multiplier)
        self.self_suppression = float(self_suppression)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.batch_stats: Dict[int, Dict[str, Tensor]] = {}

    def reset_batch(self) -> None:
        self.batch_stats = {}

    def _hook(self, layer_number: int):
        def apply(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            if not torch.is_tensor(output):
                return output
            before = patch_attention_statistics(output, prefix_count=self.prefix_count)
            transformed = transform_patch_attention(
                output,
                prefix_count=self.prefix_count,
                temperature_multiplier=self.temperature_multiplier,
                self_suppression=self.self_suppression,
            )
            after = patch_attention_statistics(transformed, prefix_count=self.prefix_count)
            self.batch_stats[int(layer_number)] = {
                **{f"before_{key}": value for key, value in before.items()},
                **{f"after_{key}": value for key, value in after.items()},
            }
            return transformed

        return apply

    def __enter__(self):
        blocks = getattr(self.backbone, "blocks")
        for layer_number in self.layers:
            dropout = getattr(getattr(blocks[layer_number - 1], "attn"), "attention_dropout")
            self.handles.append(dropout.register_forward_hook(self._hook(layer_number)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _resolve_attention_backbone(model: nn.Module) -> nn.Module:
    candidates = [model]
    for name in ("model", "backbone", "classifier"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, nn.Module):
            candidates.append(candidate)
    for candidate in candidates:
        blocks = getattr(candidate, "blocks", None)
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
            first_attention = getattr(getattr(blocks[0], "attn", None), "attention_dropout", None)
            if isinstance(first_attention, nn.Module):
                return candidate
    raise TypeError("Could not locate TRKH transformer blocks for attention hooks.")


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in row_list:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _class_f1(metrics: Mapping[str, object], class_index: int = 1) -> float:
    per_class = metrics.get("per_class", [])
    if isinstance(per_class, Sequence):
        for row in per_class:
            if isinstance(row, Mapping) and int(row.get("class_index", -1)) == int(class_index):
                return float(row.get("f1", 0.0) or 0.0)
    return 0.0


def summarize_prediction_changes(
    *,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    variant_predictions: np.ndarray,
    class1_index: int = 1,
) -> Dict[str, int]:
    targets = np.asarray(targets, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    variant_predictions = np.asarray(variant_predictions, dtype=np.int64)
    if targets.shape != baseline_predictions.shape or targets.shape != variant_predictions.shape:
        raise ValueError("Prediction change arrays must have matching shapes.")
    changed = baseline_predictions != variant_predictions
    baseline_correct = baseline_predictions == targets
    variant_correct = variant_predictions == targets
    class1_target = targets == int(class1_index)
    non_class1_target = ~class1_target
    return {
        "changed": int(changed.sum()),
        "corrections": int((changed & ~baseline_correct & variant_correct).sum()),
        "harms": int((changed & baseline_correct & ~variant_correct).sum()),
        "neutral": int((changed & ~baseline_correct & ~variant_correct).sum()),
        "class1_fn_rescues": int(
            (class1_target & (baseline_predictions != class1_index) & (variant_predictions == class1_index)).sum()
        ),
        "class1_correct_breaks": int(
            (class1_target & (baseline_predictions == class1_index) & (variant_predictions != class1_index)).sum()
        ),
        "class1_fp_removed": int(
            (non_class1_target & (baseline_predictions == class1_index) & (variant_predictions != class1_index)).sum()
        ),
        "class1_fp_created": int(
            (non_class1_target & (baseline_predictions != class1_index) & (variant_predictions == class1_index)).sum()
        ),
    }


def _attention_summary(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    layers = sorted({int(row["layer"]) for row in rows})
    metric_keys = [
        "before_patch_mass",
        "before_absolute_self_mass",
        "before_conditional_self_fraction",
        "before_normalized_entropy",
        "before_top1_mass",
        "before_self_top1_fraction",
        "after_patch_mass",
        "after_absolute_self_mass",
        "after_conditional_self_fraction",
        "after_normalized_entropy",
        "after_top1_mass",
        "after_self_top1_fraction",
    ]
    for layer in layers:
        selected = [row for row in rows if int(row["layer"]) == layer]
        result: Dict[str, object] = {"layer": layer, "support": len(selected)}
        for key in metric_keys:
            result[key] = float(np.mean([float(row[key]) for row in selected])) if selected else 0.0
        results.append(result)
    return results


def _evaluate_variant(
    *,
    model: nn.Module,
    backbone: nn.Module,
    dataset: Dataset,
    loader: DataLoader,
    class_names: Sequence[str],
    variant: Mapping[str, object],
    layers: Sequence[int],
    prefix_count: int,
    bbox_token_prior_source: str,
    split: str,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> Dict[str, object]:
    prediction_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    probability_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    dataset_paths_fn = getattr(dataset, "sample_paths", None)
    dataset_paths = [str(path) for path in dataset_paths_fn()] if callable(dataset_paths_fn) else []
    seen_samples = 0
    model.eval()
    hooks = _AttentionVariantHooks(
        backbone=backbone,
        layers=layers,
        prefix_count=prefix_count,
        temperature_multiplier=float(variant["temperature_multiplier"]),
        self_suppression=float(variant["self_suppression"]),
    )
    with hooks, torch.inference_mode():
        iterator = tqdm(
            loader,
            desc=f"lsa-{variant['name']}-{split}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_index, (images, labels, metadata) in enumerate(iterator):
            if int(max_batches) > 0 and batch_index >= int(max_batches):
                break
            if not isinstance(metadata, Mapping):
                metadata = {}
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            batch_size = int(images.size(0))
            raw_paths = metadata.get("paths", [])
            batch_paths = [str(path) for path in raw_paths] if isinstance(raw_paths, Sequence) else []
            if dataset_paths and (len(batch_paths) != batch_size or not any(path.strip() for path in batch_paths)):
                batch_paths = dataset_paths[seen_samples : seen_samples + batch_size]
            sample_indices = _sample_indices_from_metadata(
                metadata,
                fallback_start=seen_samples,
                batch_size=batch_size,
            )
            fallback_start = seen_samples
            seen_samples += batch_size
            bbox_prior = _select_bbox_token_prior(
                metadata=metadata,
                source=bbox_token_prior_source,
                device=device,
            )
            image_mask = _tensor_metadata(metadata, "image_mask", device=device, dtype=torch.bool)
            hooks.reset_batch()
            with autocast_context(device, amp):
                features = model.forward_features(
                    images,
                    image_valid_mask=image_mask,
                    bbox_token_prior=bbox_prior,
                )
                if torch.is_tensor(bbox_prior):
                    features["bbox"] = bbox_prior[:, :4] if bbox_prior.ndim == 2 else bbox_prior
                if hasattr(model, "forward_heads"):
                    output = model.forward_heads(features)
                else:
                    output = classification_logits_from_features(model, features)
                logits, _, _ = extract_detection_from_model_output(output)
                probabilities = logits.float().softmax(dim=1)
            predictions = probabilities.argmax(dim=1)
            margins = _confidence_margin(probabilities)
            probs_cpu = probabilities.detach().cpu().numpy().astype(np.float32, copy=False)
            labels_cpu = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            preds_cpu = predictions.detach().cpu().numpy().astype(np.int64, copy=False)
            margins_cpu = margins.detach().cpu().numpy().astype(np.float32, copy=False)
            probability_batches.append(probs_cpu)
            label_batches.append(labels_cpu)
            prediction_batches.append(preds_cpu)
            for offset, target_index in enumerate(labels_cpu):
                pred_index = int(preds_cpu[offset])
                row: Dict[str, object] = {
                    "variant": str(variant["name"]),
                    "row_index": int(fallback_start + offset),
                    "sample_index": int(sample_indices[offset]),
                    "image_path": str(batch_paths[offset]) if offset < len(batch_paths) else "",
                    "target_index": int(target_index),
                    "prediction_index": pred_index,
                    "transition": f"{int(target_index)}->{pred_index}",
                    "correct": int(pred_index == int(target_index)),
                    "confidence": float(probs_cpu[offset].max()),
                    "margin": float(margins_cpu[offset]),
                    "prob_true": float(probs_cpu[offset, int(target_index)]),
                    "prob_class1": float(probs_cpu[offset, 1]) if probs_cpu.shape[1] > 1 else 0.0,
                }
                for class_index, probability in enumerate(probs_cpu[offset]):
                    row[f"prob_{class_index}"] = float(probability)
                prediction_rows.append(row)
            missing_layers = [layer for layer in layers if layer not in hooks.batch_stats]
            if missing_layers:
                raise RuntimeError(f"Attention hooks did not run for layers: {missing_layers}")
            for layer in layers:
                stats = hooks.batch_stats[layer]
                stats_cpu = {key: value.detach().float().cpu().numpy() for key, value in stats.items()}
                for offset, target_index in enumerate(labels_cpu):
                    attention_rows.append(
                        {
                            "variant": str(variant["name"]),
                            "sample_index": int(sample_indices[offset]),
                            "target_index": int(target_index),
                            "prediction_index": int(preds_cpu[offset]),
                            "correct": int(preds_cpu[offset] == int(target_index)),
                            "layer": int(layer),
                            **{key: float(values[offset]) for key, values in stats_cpu.items()},
                        }
                    )
    labels_array = np.concatenate(label_batches) if label_batches else np.zeros((0,), dtype=np.int64)
    predictions_array = (
        np.concatenate(prediction_batches) if prediction_batches else np.zeros((0,), dtype=np.int64)
    )
    probabilities_array = (
        np.concatenate(probability_batches)
        if probability_batches
        else np.zeros((0, len(class_names)), dtype=np.float32)
    )
    return {
        "prediction_rows": prediction_rows,
        "attention_rows": attention_rows,
        "attention_summary": _attention_summary(attention_rows),
        "labels": labels_array.astype(np.int64, copy=False),
        "predictions": predictions_array.astype(np.int64, copy=False),
        "probabilities": probabilities_array.astype(np.float32, copy=False),
        "metrics": _classification_metrics(labels_array, predictions_array, list(class_names)),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only pre-smoke probe for patch-only Locality Self-Attention. "
            "It measures runtime self-relation/entropy and fixed attention transforms; "
            "it never edits raw data or writes trainable manifests."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--layers", type=str, default="1,2,3,4")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument(
        "--bbox-token-prior-source",
        choices=("bbox", "crop_bbox"),
        default="bbox",
    )
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    split = str(args.split).strip().lower()
    if split == "test" and not bool(args.allow_test):
        raise ValueError("Locality-attention probe refuses test unless --allow-test is set.")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    backbone = _resolve_attention_backbone(model)
    blocks = getattr(backbone, "blocks")
    layers = parse_layer_numbers(str(args.layers), block_count=len(blocks))
    prefix_count = int(getattr(backbone, "num_prefix_tokens", 0))
    if prefix_count <= 0:
        raise ValueError("TRKH attention backbone must expose num_prefix_tokens.")
    variants = parse_variant_specs(args.variant)
    dataset, class_names = _build_dataset(
        data_yaml=Path(args.data),
        split=split,
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=int(args.max_samples),
    )
    if int(args.expected_num_classes) > 0 and len(class_names) != int(args.expected_num_classes):
        raise ValueError("Class count does not match --expected-num-classes.")
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    start_time = time.perf_counter()
    outputs: Dict[str, Dict[str, object]] = {}
    all_prediction_rows: List[Dict[str, object]] = []
    all_attention_rows: List[Dict[str, object]] = []
    for variant in variants:
        payload = _evaluate_variant(
            model=model,
            backbone=backbone,
            dataset=dataset,
            loader=loader,
            class_names=class_names,
            variant=variant,
            layers=layers,
            prefix_count=prefix_count,
            bbox_token_prior_source=str(args.bbox_token_prior_source),
            split=split,
            device=device,
            amp=bool(args.amp),
            max_batches=int(args.max_batches),
        )
        outputs[str(variant["name"])] = payload
        all_prediction_rows.extend(payload["prediction_rows"])
        all_attention_rows.extend(payload["attention_rows"])

    baseline_variant = variants[0]
    baseline_name = str(baseline_variant["name"])
    baseline = outputs[baseline_name]
    baseline_labels = np.asarray(baseline["labels"], dtype=np.int64)
    baseline_predictions = np.asarray(baseline["predictions"], dtype=np.int64)
    baseline_metrics = baseline["metrics"]
    baseline_rows = {int(row["sample_index"]): row for row in baseline["prediction_rows"]}
    variant_summaries: List[Dict[str, object]] = []
    changed_rows: List[Dict[str, object]] = []
    candidate_names: List[str] = []
    for variant in variants:
        name = str(variant["name"])
        payload = outputs[name]
        labels = np.asarray(payload["labels"], dtype=np.int64)
        predictions = np.asarray(payload["predictions"], dtype=np.int64)
        if not np.array_equal(labels, baseline_labels):
            raise RuntimeError(f"Target order changed for variant {name}.")
        metrics = payload["metrics"]
        changes = summarize_prediction_changes(
            targets=labels,
            baseline_predictions=baseline_predictions,
            variant_predictions=predictions,
        )
        macro_delta = float(metrics["macro_f1"]) - float(baseline_metrics["macro_f1"])
        class1_delta = _class_f1(metrics) - _class_f1(baseline_metrics)
        candidate = bool(
            name != baseline_name
            and macro_delta >= 0.0
            and class1_delta > 0.0
            and changes["corrections"] > changes["harms"]
            and changes["class1_fn_rescues"] >= changes["class1_correct_breaks"]
            and changes["class1_fp_created"] <= changes["class1_fp_removed"]
        )
        if candidate:
            candidate_names.append(name)
        variant_summaries.append(
            {
                "name": name,
                "temperature_multiplier": float(variant["temperature_multiplier"]),
                "self_suppression": float(variant["self_suppression"]),
                "metrics": metrics,
                "macro_f1_delta": macro_delta,
                "class1_f1_delta": class1_delta,
                "changes": changes,
                "attention_summary": payload["attention_summary"],
                "candidate_for_trainable_smoke": candidate,
            }
        )
        if name == baseline_name:
            continue
        current_rows = {int(row["sample_index"]): row for row in payload["prediction_rows"]}
        for sample_index, base_row in baseline_rows.items():
            current = current_rows[sample_index]
            if int(base_row["prediction_index"]) == int(current["prediction_index"]):
                continue
            target = int(base_row["target_index"])
            base_pred = int(base_row["prediction_index"])
            current_pred = int(current["prediction_index"])
            changed_rows.append(
                {
                    "variant": name,
                    "sample_index": sample_index,
                    "image_path": str(base_row.get("image_path", "")),
                    "target_index": target,
                    "baseline_prediction": base_pred,
                    "variant_prediction": current_pred,
                    "transition": f"{target}:{base_pred}->{current_pred}",
                    "outcome": (
                        "correction"
                        if base_pred != target and current_pred == target
                        else "harm"
                        if base_pred == target and current_pred != target
                        else "neutral"
                    ),
                    "baseline_prob_class1": float(base_row["prob_class1"]),
                    "variant_prob_class1": float(current["prob_class1"]),
                    "class1_probability_delta": float(current["prob_class1"])
                    - float(base_row["prob_class1"]),
                }
            )

    summary = {
        "mode": "validation_only_locality_self_attention_precheck",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "support": int(baseline_labels.size),
        "class_names": list(class_names),
        "layers": layers,
        "prefix_count": prefix_count,
        "bbox_token_prior_source": str(args.bbox_token_prior_source),
        "variants": variant_summaries,
        "candidate_variants": candidate_names,
        "smoke_gate_ready": bool(candidate_names),
        "training_permission": False,
        "raw_dataset_touched": False,
        "test_accessed": bool(split == "test"),
        "trainable_manifest_written": False,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "guardrail": (
            "Fixed inference diagnostic only. A candidate still requires an exact-identity "
            "trainable implementation, tests, and a separate short smoke."
        ),
    }
    _write_rows_csv(output_dir / "predictions.csv", all_prediction_rows)
    _write_rows_csv(output_dir / "attention_statistics.csv", all_attention_rows)
    _write_rows_csv(output_dir / "changed_predictions.csv", changed_rows)
    flat_summary_rows = []
    for row in variant_summaries:
        flat_summary_rows.append(
            {
                "name": row["name"],
                "temperature_multiplier": row["temperature_multiplier"],
                "self_suppression": row["self_suppression"],
                "macro_f1": row["metrics"]["macro_f1"],
                "class1_f1": _class_f1(row["metrics"]),
                "macro_f1_delta": row["macro_f1_delta"],
                "class1_f1_delta": row["class1_f1_delta"],
                **row["changes"],
                "candidate_for_trainable_smoke": row["candidate_for_trainable_smoke"],
            }
        )
    _write_rows_csv(output_dir / "variant_summary.csv", flat_summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_lines = [
        "# Locality Self-Attention Pre-Smoke Probe",
        "",
        f"- Split/support: `{split}` / `{int(baseline_labels.size)}`",
        f"- Layers: `{','.join(str(layer) for layer in layers)}`",
        f"- Prefix tokens preserved: `{prefix_count}`",
        f"- Test accessed: `{bool(split == 'test')}`",
        f"- Smoke gate ready: `{bool(candidate_names)}`",
        f"- Candidate variants: `{candidate_names}`",
        "",
        "The transform preserves prefix query rows, patch-to-prefix links, and each patch row's total patch mass. It writes no trainable parameters, manifests, or raw-data changes.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(
        {
            "output_dir": str(output_dir),
            "support": int(baseline_labels.size),
            "candidate_variants": candidate_names,
            "smoke_gate_ready": bool(candidate_names),
            "test_accessed": bool(split == "test"),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
