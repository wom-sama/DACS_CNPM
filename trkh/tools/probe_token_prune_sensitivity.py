from __future__ import annotations

import argparse
import copy
import csv
import json
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
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
)
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _class1_f1,
    _select_bbox_token_prior,
)


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
        if not isinstance(item, Mapping):
            continue
        class_index = int(item.get("class_index", -1))
        if class_index < 0:
            continue
        fields[f"class{class_index}_precision"] = float(item.get("precision", 0.0) or 0.0)
        fields[f"class{class_index}_recall"] = float(item.get("recall", 0.0) or 0.0)
        fields[f"class{class_index}_f1"] = float(item.get("f1", 0.0) or 0.0)
    return fields


def _variant_key(name: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
    key = "_".join(part for part in key.split("_") if part)
    return key or "variant"


def parse_variant_spec(text: str) -> Dict[str, object]:
    value = str(text or "").strip()
    if not value:
        raise ValueError("Variant spec rong.")
    if "=" in value:
        name, spec = value.split("=", 1)
    elif ":" in value:
        name, spec = value.split(":", 1)
    else:
        name, spec = value, "checkpoint"
    name = _variant_key(name)
    spec = str(spec or "checkpoint").strip()
    spec_lower = spec.lower()
    if spec_lower in {"checkpoint", "baseline", "keep"}:
        return {
            "name": name,
            "mode": "checkpoint",
            "token_pruning": None,
            "token_prune_layers": None,
            "token_keep_rates": None,
            "early_token_mask_keep_rate": None,
        }
    if spec_lower in {"off", "none", "no_prune", "nopruning"}:
        return {
            "name": name,
            "mode": "off",
            "token_pruning": False,
            "token_prune_layers": "",
            "token_keep_rates": "",
            "early_token_mask_keep_rate": 1.0,
        }
    if "|" in spec:
        layers, rates = spec.split("|", 1)
    elif ";" in spec:
        layers, rates = spec.split(";", 1)
    else:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                "Variant spec phai co dang name=checkpoint, name=off, "
                "hoac name=2,5|0.95,0.85."
            )
        layers, rates = parts
    layers = layers.strip()
    rates = rates.strip()
    if not layers or not rates:
        raise ValueError("Variant token-prune phai co layers va rates.")
    rate_values = [float(part.strip()) for part in rates.split(",") if part.strip()]
    if not rate_values:
        raise ValueError("Variant token_keep_rates rong.")
    for rate in rate_values:
        if rate <= 0.0 or rate > 1.0:
            raise ValueError("Moi token keep-rate phai nam trong (0,1].")
    return {
        "name": name,
        "mode": "override",
        "token_pruning": True,
        "token_prune_layers": layers,
        "token_keep_rates": ",".join(f"{rate:.6f}".rstrip("0").rstrip(".") for rate in rate_values),
        "early_token_mask_keep_rate": 1.0,
    }


def parse_variant_specs(values: Optional[Sequence[str]]) -> List[Dict[str, object]]:
    specs = list(values or [])
    if not specs:
        specs = [
            "baseline=checkpoint",
            "keep95_85=2,5|0.95,0.85",
            "keep100_85=2,5|1.0,0.85",
            "no_prune=off",
        ]
    variants: List[Dict[str, object]] = []
    seen: set[str] = set()
    for text in specs:
        variant = parse_variant_spec(text)
        name = str(variant["name"])
        if name in seen:
            raise ValueError(f"Variant bi trung ten: {name}")
        seen.add(name)
        variants.append(variant)
    if not any(str(item.get("mode")) == "checkpoint" for item in variants):
        variants.insert(0, parse_variant_spec("baseline=checkpoint"))
    return variants


def build_variant_checkpoint(
    checkpoint: Mapping[str, object],
    variant: Mapping[str, object],
) -> Dict[str, object]:
    payload = copy.deepcopy(dict(checkpoint))
    model_config = dict(payload.get("model_config", {}))
    if variant.get("token_pruning") is not None:
        model_config["token_pruning"] = bool(variant["token_pruning"])
    if variant.get("token_prune_layers") is not None:
        model_config["token_prune_layers"] = str(variant["token_prune_layers"])
    if variant.get("token_keep_rates") is not None:
        model_config["token_keep_rates"] = str(variant["token_keep_rates"])
    if variant.get("early_token_mask_keep_rate") is not None:
        model_config["early_token_mask_keep_rate"] = float(variant["early_token_mask_keep_rate"])
    payload["model_config"] = model_config
    return payload


def _forward_classification_logits(
    *,
    model: nn.Module,
    images: torch.Tensor,
    image_valid_mask: Optional[torch.Tensor],
    bbox_metadata: Optional[torch.Tensor],
    bbox_token_prior: Optional[torch.Tensor],
    amp: bool,
    device: torch.device,
    return_trace: bool,
) -> tuple[torch.Tensor, Optional[Mapping[str, object]]]:
    trace = None
    with autocast_context(device, amp):
        if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
                return_trace=return_trace,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            output = model.forward_heads(features)
            if return_trace and isinstance(features.get("trace"), Mapping):
                trace = features["trace"]
        elif hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
            features = model.forward_features(
                images,
                image_valid_mask=image_valid_mask,
                bbox_token_prior=bbox_token_prior,
                return_trace=return_trace,
            )
            if bbox_metadata is not None:
                features["bbox"] = bbox_metadata
            output = classification_logits_from_features(model, features)
            if return_trace and isinstance(features.get("trace"), Mapping):
                trace = features["trace"]
        else:
            output = model(images)
        logits = output["logits"] if isinstance(output, Mapping) and torch.is_tensor(output.get("logits")) else output
    if not torch.is_tensor(logits):
        raise TypeError("Model output khong chua logits tensor.")
    return logits.float(), trace


def _summarize_trace(trace: Mapping[str, object]) -> Dict[str, object]:
    pruning = trace.get("pruning", [])
    rows: List[Dict[str, object]] = []
    if isinstance(pruning, Sequence):
        for item in pruning:
            if not isinstance(item, Mapping):
                continue
            row: Dict[str, object] = {}
            for key in ("layer", "before_count", "after_count"):
                value = item.get(key)
                if torch.is_tensor(value):
                    row[key] = int(value.detach().cpu().reshape(-1)[0].item())
                elif value is not None:
                    row[key] = int(value)
            kept = item.get("kept_indices")
            if torch.is_tensor(kept):
                row["kept_count_mean"] = float(kept.detach().cpu().shape[-1])
            if row:
                rows.append(row)
    patch_keep_mask = trace.get("patch_keep_mask")
    if torch.is_tensor(patch_keep_mask):
        kept = patch_keep_mask.detach().float().sum(dim=1)
        return {
            "final_kept_tokens_mean": float(kept.mean().cpu().item()),
            "final_kept_tokens_min": float(kept.min().cpu().item()),
            "final_kept_tokens_max": float(kept.max().cpu().item()),
            "pruning_steps": rows,
        }
    if rows and "after_count" in rows[-1]:
        kept_count = float(rows[-1]["after_count"])
        return {
            "final_kept_tokens_mean": kept_count,
            "final_kept_tokens_min": kept_count,
            "final_kept_tokens_max": kept_count,
            "pruning_steps": rows,
        }
    patch_embedding_shape = trace.get("patch_embedding_shape")
    if isinstance(patch_embedding_shape, Sequence) and len(patch_embedding_shape) >= 2:
        kept_count = float(patch_embedding_shape[1])
        return {
            "final_kept_tokens_mean": kept_count,
            "final_kept_tokens_min": kept_count,
            "final_kept_tokens_max": kept_count,
            "pruning_steps": rows,
        }
    return {"pruning_steps": rows}


def _merge_trace_summaries(items: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not items:
        return {}
    final_means = [float(item["final_kept_tokens_mean"]) for item in items if "final_kept_tokens_mean" in item]
    merged: Dict[str, object] = {
        "trace_batches": int(len(items)),
    }
    if final_means:
        merged["final_kept_tokens_mean"] = float(sum(final_means) / len(final_means))
        merged["final_kept_tokens_min"] = float(
            min(float(item.get("final_kept_tokens_min", item["final_kept_tokens_mean"])) for item in items)
        )
        merged["final_kept_tokens_max"] = float(
            max(float(item.get("final_kept_tokens_max", item["final_kept_tokens_mean"])) for item in items)
        )
    first_steps = items[0].get("pruning_steps", [])
    if isinstance(first_steps, list):
        merged["pruning_steps_first_batch"] = first_steps
    return merged


def build_changed_prediction_rows(
    *,
    baseline_records: Sequence[Mapping[str, object]],
    current_records: Sequence[Mapping[str, object]],
    variant_name: str,
    class_names: Sequence[str],
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    count = min(len(baseline_records), len(current_records))
    rows: List[Dict[str, object]] = []
    counts = {"corrections": 0, "harms": 0, "neutral": 0, "class1_related": 0}
    for index in range(count):
        base = baseline_records[index]
        current = current_records[index]
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
        if 1 in {target, base_pred, current_pred}:
            counts["class1_related"] += 1
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
            "variant_prediction_index": current_pred,
            "variant_prediction_name": current_name,
            "baseline_correct": base_correct,
            "variant_correct": current_correct,
            "change_type": change_type,
            "transition": f"{target}:{base_pred}->{current_pred}",
            "variant": str(variant_name),
            "baseline_confidence": base.get("confidence", ""),
            "variant_confidence": current.get("confidence", ""),
        }
        for class_index, class_name in enumerate(class_names):
            base_key = f"prob_{class_index}_{class_name}"
            if base_key in base:
                row[f"baseline_prob_{class_index}"] = base.get(base_key, "")
            if base_key in current:
                row[f"variant_prob_{class_index}"] = current.get(base_key, "")
        rows.append(row)
    return rows, counts


def _evaluate_variant(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    model: nn.Module,
    dataset,
    class_names: Sequence[str],
    device: torch.device,
    variant_name: str,
) -> Dict[str, object]:
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=args.workers,
        requested_pin_memory=device.type == "cuda",
        context=f"token_prune_{variant_name}_{args.split}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(getattr(dataset, "num_classes", len(class_names)) or len(class_names))),
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
    trace_summaries: List[Dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        with tqdm(iterator, desc=f"token-prune {variant_name}", total=total_batches, dynamic_ncols=True) as pbar:
            for batch_index, (images, labels, targets) in enumerate(pbar):
                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)
                metric_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
                targets = targets if isinstance(targets, Mapping) else {}
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
                want_trace = int(args.trace_batches) > 0 and batch_index < int(args.trace_batches)
                logits, trace = _forward_classification_logits(
                    model=model,
                    images=images,
                    image_valid_mask=image_mask,
                    bbox_metadata=bbox,
                    bbox_token_prior=bbox_token_prior,
                    amp=bool(args.amp),
                    device=device,
                    return_trace=want_trace,
                )
                if trace is not None:
                    trace_summaries.append(_summarize_trace(trace))
                probability_batches.append(torch.softmax(logits, dim=1).detach().cpu())
                target_batches.append(metric_labels.detach().cpu().to(dtype=torch.long))

    probabilities = torch.cat(probability_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    predictions = probabilities.argmax(dim=1)
    sample_count = int(targets.numel())
    if len(sample_paths) > sample_count:
        sample_paths = sample_paths[:sample_count]
    if len(sample_metadata) > sample_count:
        sample_metadata = sample_metadata[:sample_count]
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
        "variant": variant_name,
        "samples": sample_count,
        "metrics": metrics,
        "records": records,
        "dataloader": dataloader_summary,
        "trace_summary": _merge_trace_summaries(trace_summaries),
        "model_config": {
            "token_pruning": bool(getattr(model, "token_pruning", False)),
            "token_prune_layers": checkpoint.get("model_config", {}).get("token_prune_layers", ""),
            "token_keep_rates": checkpoint.get("model_config", {}).get("token_keep_rates", ""),
            "early_token_mask_keep_rate": checkpoint.get("model_config", {}).get("early_token_mask_keep_rate", 1.0),
        },
    }


def _sweep_row(
    *,
    variant: Mapping[str, object],
    result: Mapping[str, object],
    baseline_macro: Optional[float],
    baseline_class1: Optional[float],
) -> Dict[str, object]:
    metrics = dict(result["metrics"])
    model_config = dict(result.get("model_config", {}))
    row: Dict[str, object] = {
        "variant": str(variant["name"]),
        "mode": str(variant.get("mode", "")),
        "samples": int(result.get("samples", 0)),
        "token_pruning": bool(model_config.get("token_pruning", False)),
        "token_prune_layers": str(model_config.get("token_prune_layers", "")),
        "token_keep_rates": str(model_config.get("token_keep_rates", "")),
        "final_kept_tokens_mean": dict(result.get("trace_summary", {})).get("final_kept_tokens_mean", ""),
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
        "class1_f1": _class1_f1(metrics, 1),
    }
    if baseline_macro is not None:
        row["delta_macro_f1_vs_baseline"] = float(row["macro_f1"]) - float(baseline_macro)
    if baseline_class1 is not None:
        row["delta_class1_f1_vs_baseline"] = float(row["class1_f1"]) - float(baseline_class1)
    row.update(_class_metric_fields(metrics))
    return row


def _compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "macro_precision": float(metrics.get("macro_precision", 0.0) or 0.0),
        "macro_recall": float(metrics.get("macro_recall", 0.0) or 0.0),
        "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
        "per_class": metrics.get("per_class", []),
        "confusion_matrix": metrics.get("confusion_matrix", []),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether changing token-prune keep rates on a frozen checkpoint "
            "recovers class-1 surface/boundary signal. Refuses test by default."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--class-name-mode", default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=0)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--trace-batches", type=int, default=2)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    parser.add_argument("--save-all-predictions", action="store_true", default=False)
    parser.add_argument("--allow-test", action="store_true", default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if str(args.split).lower() == "test" and not bool(args.allow_test):
        raise ValueError("Khong dung token-prune diagnostic tren test khi dang chon/tune huong.")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    set_seed(args.seed, deterministic=False)
    output_dir = ensure_dir(args.output_dir)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    variants = parse_variant_specs(args.variant)
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

    device_text = str(args.device or "").strip()
    device = torch.device(device_text) if device_text else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    eval_transform = _build_eval_transform_from_checkpoint(checkpoint, image_size=image_size)
    dataset = _build_classification_dataset(
        data_spec=data_spec,
        split=args.split,
        transform=eval_transform,
        checkpoint=dict(checkpoint),
    )

    results: Dict[str, Dict[str, object]] = {}
    baseline_records: Optional[Sequence[Mapping[str, object]]] = None
    baseline_macro: Optional[float] = None
    baseline_class1: Optional[float] = None
    sweep_rows: List[Dict[str, object]] = []
    changed_summaries: List[Dict[str, object]] = []

    for variant in variants:
        variant_name = str(variant["name"])
        variant_checkpoint = build_variant_checkpoint(checkpoint, variant)
        model = build_model_from_checkpoint(
            variant_checkpoint,
            num_classes=len(class_names),
        )
        model.to(device)
        result = _evaluate_variant(
            args=args,
            checkpoint=variant_checkpoint,
            model=model,
            dataset=dataset,
            class_names=class_names,
            device=device,
            variant_name=variant_name,
        )
        results[variant_name] = result
        metrics = dict(result["metrics"])
        if baseline_records is None and str(variant.get("mode")) == "checkpoint":
            baseline_records = result["records"]  # type: ignore[assignment]
            baseline_macro = float(metrics.get("macro_f1", 0.0) or 0.0)
            baseline_class1 = _class1_f1(metrics, 1)
        if bool(args.save_all_predictions) or str(variant.get("mode")) == "checkpoint":
            _write_rows_csv(output_dir / f"predictions_{variant_name}.csv", result["records"])  # type: ignore[arg-type]
        sweep_rows.append(
            _sweep_row(
                variant=variant,
                result=result,
                baseline_macro=baseline_macro,
                baseline_class1=baseline_class1,
            )
        )
        if baseline_records is not None and str(variant.get("mode")) != "checkpoint":
            changed_rows, changed_counts = build_changed_prediction_rows(
                baseline_records=baseline_records,
                current_records=result["records"],  # type: ignore[arg-type]
                variant_name=variant_name,
                class_names=class_names,
            )
            _write_rows_csv(output_dir / f"changed_vs_baseline_{variant_name}.csv", changed_rows)
            transitions: Dict[str, int] = {}
            for row in changed_rows:
                transition = str(row.get("transition", ""))
                transitions[transition] = transitions.get(transition, 0) + 1
            changed_summaries.append(
                {
                    "variant": variant_name,
                    "changed": int(len(changed_rows)),
                    **changed_counts,
                    "transitions": transitions,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if baseline_records is None:
        raise RuntimeError("Khong co baseline checkpoint variant.")
    _write_rows_csv(output_dir / "token_prune_sweep.csv", sweep_rows)
    metrics_by_variant = {
        name: _compact_metrics(dict(result["metrics"]))
        for name, result in results.items()
    }
    trace_by_variant = {
        name: dict(result.get("trace_summary", {}))
        for name, result in results.items()
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_spec.data_yaml),
        "split": str(args.split),
        "class_names": class_names,
        "variants": [dict(item) for item in variants],
        "sweep": sweep_rows,
        "changed_summaries": changed_summaries,
        "metrics_by_variant": to_serializable(metrics_by_variant),
        "trace_by_variant": to_serializable(trace_by_variant),
        "dataset_report": dataset.quality_report() if hasattr(dataset, "quality_report") else {},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(to_serializable(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
