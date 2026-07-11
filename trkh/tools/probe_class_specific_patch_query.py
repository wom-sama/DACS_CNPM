from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _classification_metrics,
    _resolve_device,
)
from trkh.tools.probe_patch_part_prototypes import _extract_split_patch_payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only diagnostic for class-specific patch queries over frozen TRKH "
            "patch tokens. Inspired by class-query FGVC heads, this fits only a tiny "
            "readout on train split and evaluates validation; it writes no trainable "
            "data manifest and refuses test unless explicitly allowed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-patches-per-sample", type=int, default=24)
    parser.add_argument("--bbox-threshold", type=float, default=0.05)
    parser.add_argument(
        "--bbox-token-prior-source",
        type=str,
        default="crop_bbox",
        choices=("bbox", "crop_bbox"),
    )
    parser.add_argument("--query-epochs", type=int, default=10)
    parser.add_argument("--query-batch-size", type=int, default=192)
    parser.add_argument("--query-lr", type=float, default=2e-3)
    parser.add_argument("--query-weight-decay", type=float, default=1e-4)
    parser.add_argument("--query-dropout", type=float, default=0.05)
    parser.add_argument("--attention-temperature", type=float, default=0.20)
    parser.add_argument("--base-logit-scale", type=float, default=1.0)
    parser.add_argument("--residual-logit-scale", type=float, default=0.12)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-test", action="store_true", default=False)
    return parser.parse_args(argv)


def build_padded_patch_cache(
    *,
    patch_tokens: np.ndarray,
    patch_sample_indices: np.ndarray,
    patch_weights: np.ndarray,
    sample_count: int,
    max_patches_per_sample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert flat selected patch rows into a stable [sample, patch, dim] cache."""

    tokens = np.asarray(patch_tokens, dtype=np.float32)
    owners = np.asarray(patch_sample_indices, dtype=np.int64).reshape(-1)
    weights = np.asarray(patch_weights, dtype=np.float32).reshape(-1)
    sample_count = int(sample_count)
    max_patches = max(1, int(max_patches_per_sample))
    if tokens.ndim != 2:
        raise ValueError("patch_tokens must be [num_rows, dim].")
    if owners.shape[0] != tokens.shape[0] or weights.shape[0] != tokens.shape[0]:
        raise ValueError("patch tokens, owners, and weights must align.")
    dim = int(tokens.shape[1])
    padded = np.zeros((sample_count, max_patches, dim), dtype=np.float32)
    mask = np.zeros((sample_count, max_patches), dtype=np.bool_)
    padded_weights = np.zeros((sample_count, max_patches), dtype=np.float32)
    counts = np.zeros((sample_count,), dtype=np.int64)
    for row_index, owner in enumerate(owners):
        owner_int = int(owner)
        if owner_int < 0 or owner_int >= sample_count:
            continue
        slot = int(counts[owner_int])
        if slot >= max_patches:
            continue
        padded[owner_int, slot] = tokens[row_index]
        mask[owner_int, slot] = True
        padded_weights[owner_int, slot] = max(float(weights[row_index]), 1e-6)
        counts[owner_int] += 1
    empty = counts <= 0
    if bool(empty.any()):
        mask[empty, 0] = True
        padded_weights[empty, 0] = 1.0
    return padded, mask, padded_weights, counts


class ClassSpecificPatchQueryReadout(nn.Module):
    """Small class-query attention readout over selected patch tokens."""

    def __init__(
        self,
        *,
        dim: int,
        num_classes: int,
        attention_temperature: float = 0.20,
        dropout: float = 0.05,
        base_logit_scale: float = 1.0,
        residual_logit_scale: float = 0.12,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.attention_temperature = float(max(1e-4, attention_temperature))
        self.base_logit_scale = float(base_logit_scale)
        self.residual_logit_scale = float(residual_logit_scale)
        self.patch_norm = nn.LayerNorm(self.dim)
        self.query = nn.Parameter(torch.empty(self.num_classes, self.dim))
        self.class_weight = nn.Parameter(torch.empty(self.num_classes, self.dim))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.dropout = nn.Dropout(float(max(0.0, dropout)))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.class_weight, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
        patch_weights: torch.Tensor,
        base_probabilities: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        if patches.ndim != 3:
            raise ValueError("patches must be [B,N,D].")
        valid = mask.to(device=patches.device, dtype=torch.bool)
        empty = ~valid.any(dim=1)
        if bool(empty.any().item()):
            valid = valid.clone()
            valid[empty, 0] = True
        norm_patches = F.normalize(self.patch_norm(patches), dim=-1)
        norm_query = F.normalize(self.query, dim=-1)
        scores = torch.einsum("bnd,cd->bcn", norm_patches, norm_query)
        scores = scores / self.attention_temperature
        if torch.is_tensor(patch_weights) and patch_weights.shape[:2] == patches.shape[:2]:
            prior = patch_weights.to(device=patches.device, dtype=torch.float32).clamp_min(1e-6)
            scores = scores + prior.log().unsqueeze(1)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        descriptor = torch.einsum("bcn,bnd->bcd", attention, self.dropout(norm_patches))
        residual = (descriptor * self.class_weight.unsqueeze(0)).sum(dim=-1) + self.bias
        logits = self.residual_logit_scale * residual
        if torch.is_tensor(base_probabilities):
            base = base_probabilities.to(device=patches.device, dtype=torch.float32).clamp_min(1e-8)
            logits = logits + self.base_logit_scale * base.log()
        if return_attention:
            return logits, attention
        return logits


def _class_weights(labels: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(num_classes)).astype(
        np.float32,
        copy=False,
    )
    non_zero = counts[counts > 0.0]
    base = float(np.median(non_zero)) if non_zero.size else 1.0
    weights = np.ones((int(num_classes),), dtype=np.float32)
    for index in range(int(num_classes)):
        if counts[index] > 0.0:
            weights[index] = (base / counts[index]) ** float(max(0.0, power))
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.from_numpy(weights.astype(np.float32, copy=False))


def _evaluate_readout(
    model: ClassSpecificPatchQueryReadout,
    *,
    patches: np.ndarray,
    mask: np.ndarray,
    patch_weights: np.ndarray,
    base_probabilities: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    logits_rows: List[np.ndarray] = []
    attention_entropy_rows: List[np.ndarray] = []
    with torch.inference_mode():
        total = int(labels.shape[0])
        for start in range(0, total, max(1, int(batch_size))):
            end = min(total, start + max(1, int(batch_size)))
            patch_tensor = torch.from_numpy(patches[start:end]).to(device=device)
            mask_tensor = torch.from_numpy(mask[start:end]).to(device=device)
            weight_tensor = torch.from_numpy(patch_weights[start:end]).to(device=device)
            prob_tensor = torch.from_numpy(base_probabilities[start:end]).to(device=device)
            logits, attention = model(
                patch_tensor,
                mask_tensor,
                weight_tensor,
                prob_tensor,
                return_attention=True,
            )
            logits_rows.append(logits.detach().float().cpu().numpy())
            entropy = -(
                attention.detach().float().clamp_min(1e-8)
                * attention.detach().float().clamp_min(1e-8).log()
            ).sum(dim=-1)
            denom = math.log(float(max(2, attention.shape[-1])))
            attention_entropy_rows.append((entropy / denom).cpu().numpy())
    logits_np = np.concatenate(logits_rows, axis=0).astype(np.float32, copy=False)
    predictions = logits_np.argmax(axis=1).astype(np.int64, copy=False)
    return {
        "logits": logits_np,
        "predictions": predictions,
        "attention_entropy": (
            np.concatenate(attention_entropy_rows, axis=0).astype(np.float32, copy=False)
            if attention_entropy_rows
            else np.zeros((0, model.num_classes), dtype=np.float32)
        ),
    }


def _train_readout(
    *,
    train_patches: np.ndarray,
    train_mask: np.ndarray,
    train_patch_weights: np.ndarray,
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    attention_temperature: float,
    base_logit_scale: float,
    residual_logit_scale: float,
    class_weight_power: float,
    seed: int,
) -> Tuple[ClassSpecificPatchQueryReadout, List[Dict[str, object]]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    readout = ClassSpecificPatchQueryReadout(
        dim=int(train_patches.shape[-1]),
        num_classes=int(num_classes),
        attention_temperature=float(attention_temperature),
        dropout=float(dropout),
        base_logit_scale=float(base_logit_scale),
        residual_logit_scale=float(residual_logit_scale),
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_patches.astype(np.float32, copy=False)),
        torch.from_numpy(train_mask.astype(np.bool_, copy=False)),
        torch.from_numpy(train_patch_weights.astype(np.float32, copy=False)),
        torch.from_numpy(train_probabilities.astype(np.float32, copy=False)),
        torch.from_numpy(train_labels.astype(np.int64, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        readout.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_labels, int(num_classes), float(class_weight_power)).to(device)
    )
    history: List[Dict[str, object]] = []
    for epoch in range(max(1, int(epochs))):
        readout.train()
        total_loss = 0.0
        total_count = 0
        correct_count = 0
        for patches, mask, weights, probabilities, labels in loader:
            patches = patches.to(device=device)
            mask = mask.to(device=device)
            weights = weights.to(device=device)
            probabilities = probabilities.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = readout(patches, mask, weights, probabilities)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), max_norm=2.0)
            optimizer.step()
            count = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_count += count
            correct_count += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(total_loss / max(1, total_count)),
                "train_accuracy": float(correct_count / max(1, total_count)),
            }
        )
    return readout, history


def _change_summary(
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    base = np.asarray(base_predictions, dtype=np.int64)
    final = np.asarray(final_predictions, dtype=np.int64)
    changed = base != final
    before_correct = base == targets
    after_correct = final == targets
    transition_counts: Dict[str, int] = {}
    for target, old, new, is_changed in zip(targets, base, final, changed):
        if not bool(is_changed):
            continue
        key = f"{int(target)}:{int(old)}->{int(new)}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
    return {
        "changed": int(changed.sum()),
        "corrections": int((changed & ~before_correct & after_correct).sum()),
        "harms": int((changed & before_correct & ~after_correct).sum()),
        "neutral": int((changed & (before_correct == after_correct)).sum()),
        "transition_counts": dict(sorted(transition_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _focus_f1(metrics: Mapping[str, object], focus_class_index: int = 1) -> float:
    per_class = metrics.get("per_class", [])
    if not isinstance(per_class, Sequence):
        return 0.0
    index = int(focus_class_index)
    if index < 0 or index >= len(per_class):
        return 0.0
    row = per_class[index]
    return float(row.get("f1", 0.0)) if isinstance(row, Mapping) else 0.0


def _write_history(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "train_loss", "train_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    final_logits: np.ndarray,
    attention_entropy: np.ndarray,
    paths: Sequence[str],
) -> None:
    class_count = int(final_logits.shape[1]) if final_logits.ndim == 2 else 0
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "query_prediction",
        "changed",
        "query_confidence",
        "query_attention_entropy_pred",
    ] + [f"query_prob_{index}" for index in range(class_count)]
    probabilities = torch.softmax(torch.from_numpy(final_logits), dim=1).numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(np.asarray(targets, dtype=np.int64)):
            pred = int(final_predictions[index])
            row = {
                "split": split,
                "sample_index": int(index),
                "image_path": str(paths[index]) if index < len(paths) else "",
                "target_index": int(target),
                "base_prediction": int(base_predictions[index]),
                "query_prediction": pred,
                "changed": int(int(base_predictions[index]) != pred),
                "query_confidence": f"{float(probabilities[index, pred]):.10g}",
                "query_attention_entropy_pred": (
                    f"{float(attention_entropy[index, pred]):.10g}"
                    if attention_entropy.ndim == 2 and pred < attention_entropy.shape[1]
                    else ""
                ),
            }
            for class_index in range(class_count):
                row[f"query_prob_{class_index}"] = f"{float(probabilities[index, class_index]):.10g}"
            writer.writerow(row)


def _write_changed_cases(
    path: Path,
    *,
    split: str,
    targets: np.ndarray,
    base_predictions: np.ndarray,
    final_predictions: np.ndarray,
    paths: Sequence[str],
) -> None:
    fieldnames = [
        "split",
        "sample_index",
        "image_path",
        "target_index",
        "base_prediction",
        "query_prediction",
        "change_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, target in enumerate(np.asarray(targets, dtype=np.int64)):
            old = int(base_predictions[index])
            new = int(final_predictions[index])
            if old == new:
                continue
            before = old == int(target)
            after = new == int(target)
            if (not before) and after:
                change_type = "correction"
            elif before and (not after):
                change_type = "harm"
            else:
                change_type = "neutral"
            writer.writerow(
                {
                    "split": split,
                    "sample_index": int(index),
                    "image_path": str(paths[index]) if index < len(paths) else "",
                    "target_index": int(target),
                    "base_prediction": old,
                    "query_prediction": new,
                    "change_type": change_type,
                }
            )


def _serializable_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(metrics))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    requested_splits = list(args.split or ["train", "val"])
    if "train" not in requested_splits:
        raise ValueError("Class-specific patch-query probe requires train split.")
    if any(str(split).lower() == "test" for split in requested_splits) and not bool(args.allow_test):
        raise ValueError("Refusing to read test split without --allow-test.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {args.checkpoint}")
    model = build_model_from_checkpoint(dict(checkpoint))
    device = _resolve_device(str(args.device or ""))
    model.to(device)
    model.eval()

    split_payloads: Dict[str, Dict[str, object]] = {}
    class_names: List[str] = []
    for split in requested_splits:
        max_samples = int(args.max_train_samples) if str(split) == "train" else int(args.max_eval_samples)
        dataset, class_names = _build_dataset(
            data_yaml=Path(args.data),
            split=str(split),
            checkpoint=checkpoint,
            class_name_mode=str(args.class_name_mode),
            max_samples=max_samples,
        )
        split_payloads[str(split)] = _extract_split_patch_payload(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            amp=bool(args.amp),
            split=str(split),
            bbox_token_prior_source=str(args.bbox_token_prior_source),
            max_patches_per_sample=int(args.max_patches_per_sample),
            bbox_threshold=float(args.bbox_threshold),
        )

    train = split_payloads["train"]
    val_key = "val" if "val" in split_payloads else requested_splits[-1]
    val = split_payloads[str(val_key)]
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    class_count = int(len(class_names))
    train_cache = build_padded_patch_cache(
        patch_tokens=np.asarray(train["patch_tokens"], dtype=np.float32),
        patch_sample_indices=np.asarray(train["patch_sample_indices"], dtype=np.int64),
        patch_weights=np.asarray(train["patch_weights"], dtype=np.float32),
        sample_count=int(train_labels.shape[0]),
        max_patches_per_sample=int(args.max_patches_per_sample),
    )
    val_cache = build_padded_patch_cache(
        patch_tokens=np.asarray(val["patch_tokens"], dtype=np.float32),
        patch_sample_indices=np.asarray(val["patch_sample_indices"], dtype=np.int64),
        patch_weights=np.asarray(val["patch_weights"], dtype=np.float32),
        sample_count=int(val_labels.shape[0]),
        max_patches_per_sample=int(args.max_patches_per_sample),
    )
    readout, history = _train_readout(
        train_patches=train_cache[0],
        train_mask=train_cache[1],
        train_patch_weights=train_cache[2],
        train_probabilities=np.asarray(train["probabilities"], dtype=np.float32),
        train_labels=train_labels,
        num_classes=class_count,
        device=device,
        epochs=int(args.query_epochs),
        batch_size=int(args.query_batch_size),
        lr=float(args.query_lr),
        weight_decay=float(args.query_weight_decay),
        dropout=float(args.query_dropout),
        attention_temperature=float(args.attention_temperature),
        base_logit_scale=float(args.base_logit_scale),
        residual_logit_scale=float(args.residual_logit_scale),
        class_weight_power=float(args.class_weight_power),
        seed=int(args.seed),
    )
    train_eval = _evaluate_readout(
        readout,
        patches=train_cache[0],
        mask=train_cache[1],
        patch_weights=train_cache[2],
        base_probabilities=np.asarray(train["probabilities"], dtype=np.float32),
        labels=train_labels,
        device=device,
        batch_size=int(args.query_batch_size),
    )
    val_eval = _evaluate_readout(
        readout,
        patches=val_cache[0],
        mask=val_cache[1],
        patch_weights=val_cache[2],
        base_probabilities=np.asarray(val["probabilities"], dtype=np.float32),
        labels=val_labels,
        device=device,
        batch_size=int(args.query_batch_size),
    )
    val_base_predictions = np.asarray(val["base_predictions"], dtype=np.int64)
    train_base_predictions = np.asarray(train["base_predictions"], dtype=np.int64)
    base_val_metrics = _classification_metrics(val_labels, val_base_predictions, class_names)
    query_val_metrics = _classification_metrics(
        val_labels,
        np.asarray(val_eval["predictions"], dtype=np.int64),
        class_names,
    )
    query_train_metrics = _classification_metrics(
        train_labels,
        np.asarray(train_eval["predictions"], dtype=np.int64),
        class_names,
    )
    base_train_metrics = _classification_metrics(train_labels, train_base_predictions, class_names)
    change = _change_summary(
        val_labels,
        val_base_predictions,
        np.asarray(val_eval["predictions"], dtype=np.int64),
    )

    _write_history(output_dir / "training_history.csv", history)
    _write_predictions(
        output_dir / "predictions_class_query.csv",
        split=str(val_key),
        targets=val_labels,
        base_predictions=val_base_predictions,
        final_predictions=np.asarray(val_eval["predictions"], dtype=np.int64),
        final_logits=np.asarray(val_eval["logits"], dtype=np.float32),
        attention_entropy=np.asarray(val_eval["attention_entropy"], dtype=np.float32),
        paths=list(val.get("paths", [])),
    )
    _write_changed_cases(
        output_dir / "changed_class_query.csv",
        split=str(val_key),
        targets=val_labels,
        base_predictions=val_base_predictions,
        final_predictions=np.asarray(val_eval["predictions"], dtype=np.int64),
        paths=list(val.get("paths", [])),
    )

    summary = {
        "method": "class-specific patch-query frozen-token readout diagnostic",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "splits": requested_splits,
        "class_names": list(class_names),
        "train_samples": int(train_labels.shape[0]),
        "val_split": str(val_key),
        "val_samples": int(val_labels.shape[0]),
        "max_patches_per_sample": int(args.max_patches_per_sample),
        "bbox_threshold": float(args.bbox_threshold),
        "bbox_token_prior_source": str(args.bbox_token_prior_source),
        "query_epochs": int(args.query_epochs),
        "query_lr": float(args.query_lr),
        "attention_temperature": float(args.attention_temperature),
        "base_logit_scale": float(args.base_logit_scale),
        "residual_logit_scale": float(args.residual_logit_scale),
        "base_train_metrics": _serializable_metrics(base_train_metrics),
        "query_train_metrics": _serializable_metrics(query_train_metrics),
        "base_val_metrics": _serializable_metrics(base_val_metrics),
        "query_val_metrics": _serializable_metrics(query_val_metrics),
        "val_change_summary": change,
        "train_patch_rows": int(np.asarray(train["patch_tokens"]).shape[0]),
        "val_patch_rows": int(np.asarray(val["patch_tokens"]).shape[0]),
        "train_empty_patch_samples": int((train_cache[3] <= 0).sum()),
        "val_empty_patch_samples": int((val_cache[3] <= 0).sum()),
        "decision_hint": (
            "Use as a no-test precheck only. Integrate a class-query head into TRKH "
            "only if validation improves class-1 recall/precision without widening "
            "0/2/4->1 false positives."
        ),
        "leakage_guard": (
            "Readout is fitted on train split only; validation is for development "
            "diagnostics; no test split, raw-data edit, relabel, sample-weight, "
            "soft-target, or targeted-margin manifest is written."
        ),
        "research_sources": [
            "https://arxiv.org/abs/2311.04157",
            "https://arxiv.org/abs/2112.09133",
        ],
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    summary["base_val_macro_f1"] = float(base_val_metrics.get("macro_f1", 0.0))
    summary["base_val_class1_f1"] = _focus_f1(base_val_metrics, 1)
    summary["query_val_macro_f1"] = float(query_val_metrics.get("macro_f1", 0.0))
    summary["query_val_class1_f1"] = _focus_f1(query_val_metrics, 1)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
