from __future__ import annotations

import argparse
import csv
import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import (
    autocast_context,
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    StrictBalancedBatchSampler,
    build_eval_transform,
    build_train_transform,
)
from trkh.evaluation.evaluate import evaluate_model, save_evaluation_artifacts
from trkh.models.model import build_model_from_checkpoint, classification_logits_from_features


def peer_small_loss_indices(losses: Tensor, remember_rate: float) -> Tensor:
    if losses.ndim != 1:
        raise ValueError("losses must be 1D per-sample losses.")
    batch_size = int(losses.numel())
    if batch_size <= 0:
        raise ValueError("losses must not be empty.")
    keep_count = max(1, min(batch_size, int(math.ceil(batch_size * float(remember_rate)))))
    return torch.topk(losses.detach(), k=keep_count, largest=False).indices


def coteaching_cross_losses(
    losses_a: Tensor,
    losses_b: Tensor,
    *,
    remember_rate: float,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    if losses_a.shape != losses_b.shape:
        raise ValueError("losses_a and losses_b must have the same shape.")
    selected_by_a = peer_small_loss_indices(losses_a, remember_rate)
    selected_by_b = peer_small_loss_indices(losses_b, remember_rate)
    loss_a_update = losses_a[selected_by_b].mean()
    loss_b_update = losses_b[selected_by_a].mean()
    overlap = len(set(int(i) for i in selected_by_a.tolist()).intersection(int(i) for i in selected_by_b.tolist()))
    keep_count = int(selected_by_a.numel())
    stats = {
        "remember_rate": float(remember_rate),
        "selected_count": float(keep_count),
        "selected_overlap_fraction": float(overlap / max(1, keep_count)),
        "loss_a_selected_by_peer": float(loss_a_update.detach().cpu().item()),
        "loss_b_selected_by_peer": float(loss_b_update.detach().cpu().item()),
    }
    return loss_a_update, loss_b_update, stats


def _class1_f1(metrics: Mapping[str, object], class_index: int = 1) -> float:
    per_class = metrics.get("per_class", [])
    if isinstance(per_class, list):
        for row in per_class:
            if isinstance(row, Mapping) and int(row.get("class_index", -1)) == int(class_index):
                return float(row.get("f1", 0.0))
    return 0.0


def _move_metadata(metadata: object, device: torch.device) -> Dict[str, Tensor]:
    if not isinstance(metadata, Mapping):
        return {}
    moved: Dict[str, Tensor] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            moved[str(key)] = value.to(device=device, non_blocking=True)
    return moved


def _forward_logits(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    bbox_token_prior_source: str,
) -> Tensor:
    bbox = metadata.get("bbox")
    crop_bbox = metadata.get("crop_bbox")
    image_mask = metadata.get("image_mask")
    token_prior = crop_bbox if bbox_token_prior_source == "crop_bbox" and torch.is_tensor(crop_bbox) else bbox
    features = model.forward_features(
        images,
        image_valid_mask=image_mask.to(dtype=torch.bool) if torch.is_tensor(image_mask) else None,
        bbox_token_prior=token_prior,
    )
    if torch.is_tensor(bbox):
        features["bbox"] = bbox.to(dtype=torch.float32)
    return classification_logits_from_features(model, features)


def _load_model(
    checkpoint_path: Path,
    *,
    class_names: Sequence[str],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location="cpu")
    checkpoint_classes = list(checkpoint.get("class_names", []))
    if checkpoint_classes and checkpoint_classes != list(class_names):
        raise ValueError(f"Class names mismatch for checkpoint: {checkpoint_path}")
    model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names))
    model.to(device)
    return model, checkpoint


def _build_loaders(
    *,
    data_yaml: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, object], List[str]]:
    data_spec = load_data_spec(data_yaml, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(name) for name in data_spec.class_names]
    train_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=build_train_transform(image_size=image_size, resize_mode="pad"),
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    val_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="val",
        transform=build_eval_transform(image_size=image_size, resize_mode="pad"),
        crop_margin_ratio=0.05,
        crop_to_primary_object=True,
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    labels = [int(sample.primary_label) for sample in train_dataset.samples]
    sampler = StrictBalancedBatchSampler(
        labels=labels,
        batch_size=int(batch_size),
        num_classes=len(class_names),
        epoch_multiplier=1.0,
        seed=int(seed),
        drop_last=False,
    )
    train_kwargs, train_loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=True,
        context="coteach_train",
    )
    val_kwargs, val_loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=True,
        context="coteach_val",
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **train_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False, **val_kwargs)
    data_summary = {
        "data_yaml": str(Path(data_yaml).resolve()),
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "balanced_sampler": sampler.exposure_summary(),
        "train_loader": train_loader_summary,
        "val_loader": val_loader_summary,
    }
    return train_loader, val_loader, data_summary, class_names


def train_coteaching(
    *,
    data_yaml: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_val_batches: int,
    learning_rate: float,
    weight_decay: float,
    remember_rate: float,
    grad_clip_norm: float,
    num_workers: int,
    seed: int,
    amp: bool,
    bbox_token_prior_source: str,
) -> Dict[str, object]:
    started = time.perf_counter()
    output_dir = ensure_dir(Path(output_dir))
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_checkpoint = load_checkpoint(Path(checkpoint_a), map_location="cpu")
    image_size = int(dict(first_checkpoint.get("model_config", {})).get("image_size", 256))
    train_loader, val_loader, data_summary, class_names = _build_loaders(
        data_yaml=Path(data_yaml),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    model_a, checkpoint_a_payload = _load_model(Path(checkpoint_a), class_names=class_names, device=device)
    model_b, checkpoint_b_payload = _load_model(Path(checkpoint_b), class_names=class_names, device=device)
    optimizer_a = optim.AdamW(model_a.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    optimizer_b = optim.AdamW(model_b.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    remember_rate = max(0.05, min(1.0, float(remember_rate)))
    bbox_token_prior_source = str(bbox_token_prior_source or "bbox").strip().lower()
    max_val_arg = None if int(max_val_batches) <= 0 else int(max_val_batches)

    history_rows: List[Dict[str, object]] = []
    best: Dict[str, object] = {"class1_f1": -1.0, "macro_f1": -1.0, "model_name": ""}

    for epoch in range(1, int(epochs) + 1):
        if hasattr(getattr(train_loader, "batch_sampler", None), "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)  # type: ignore[union-attr]
        model_a.train()
        model_b.train()
        batch_iter = train_loader
        total_batches = len(train_loader)
        if int(max_train_batches) > 0:
            total_batches = min(total_batches, int(max_train_batches))
            batch_iter = islice(train_loader, total_batches)
        totals = {
            "loss_a": 0.0,
            "loss_b": 0.0,
            "raw_loss_a": 0.0,
            "raw_loss_b": 0.0,
            "selected_overlap_fraction": 0.0,
            "class1_selected_by_a": 0.0,
            "class1_selected_by_b": 0.0,
            "class1_count": 0.0,
        }
        seen_batches = 0
        progress = tqdm(batch_iter, total=total_batches, desc=f"CoTeach epoch {epoch}", dynamic_ncols=True)
        for batch in progress:
            if len(batch) == 3:
                images, labels, metadata = batch
            elif len(batch) == 2:
                images, labels = batch
                metadata = {}
            else:
                raise ValueError("Expected classification batch with 2 or 3 items.")
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            metadata = _move_metadata(metadata, device)

            optimizer_a.zero_grad(set_to_none=True)
            optimizer_b.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits_a = _forward_logits(
                    model_a,
                    images,
                    metadata,
                    bbox_token_prior_source=bbox_token_prior_source,
                )
                logits_b = _forward_logits(
                    model_b,
                    images,
                    metadata,
                    bbox_token_prior_source=bbox_token_prior_source,
                )
                losses_a = F.cross_entropy(logits_a.float(), labels, reduction="none")
                losses_b = F.cross_entropy(logits_b.float(), labels, reduction="none")
                loss_a, loss_b, stats = coteaching_cross_losses(
                    losses_a,
                    losses_b,
                    remember_rate=remember_rate,
                )
            loss_a.backward()
            loss_b.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), float(grad_clip_norm))
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), float(grad_clip_norm))
            optimizer_a.step()
            optimizer_b.step()

            selected_by_a = peer_small_loss_indices(losses_a.detach(), remember_rate)
            selected_by_b = peer_small_loss_indices(losses_b.detach(), remember_rate)
            class1_mask = labels == 1
            class1_count = float(class1_mask.sum().detach().cpu().item())
            if class1_count > 0.0:
                totals["class1_selected_by_a"] += float(class1_mask[selected_by_a].sum().detach().cpu().item())
                totals["class1_selected_by_b"] += float(class1_mask[selected_by_b].sum().detach().cpu().item())
                totals["class1_count"] += class1_count
            totals["loss_a"] += float(loss_a.detach().cpu().item())
            totals["loss_b"] += float(loss_b.detach().cpu().item())
            totals["raw_loss_a"] += float(losses_a.mean().detach().cpu().item())
            totals["raw_loss_b"] += float(losses_b.mean().detach().cpu().item())
            totals["selected_overlap_fraction"] += float(stats["selected_overlap_fraction"])
            seen_batches += 1
            progress.set_postfix(loss_a=f"{totals['loss_a']/seen_batches:.4f}", loss_b=f"{totals['loss_b']/seen_batches:.4f}")

        eval_a = evaluate_model(
            model_a,
            val_loader,
            device,
            class_names,
            criterion=nn.CrossEntropyLoss(),
            amp=amp,
            max_batches=max_val_arg,
            collect_prediction_records=True,
            bbox_token_prior_source=bbox_token_prior_source,
        )
        eval_b = evaluate_model(
            model_b,
            val_loader,
            device,
            class_names,
            criterion=nn.CrossEntropyLoss(),
            amp=amp,
            max_batches=max_val_arg,
            collect_prediction_records=True,
            bbox_token_prior_source=bbox_token_prior_source,
        )
        for model_name, model, metrics, checkpoint_source in (
            ("peer_a", model_a, eval_a, checkpoint_a_payload),
            ("peer_b", model_b, eval_b, checkpoint_b_payload),
        ):
            class1_f1 = _class1_f1(metrics)
            macro_f1 = float(metrics.get("macro_f1", 0.0))
            if (class1_f1, macro_f1) > (float(best["class1_f1"]), float(best["macro_f1"])):
                best = {
                    "class1_f1": class1_f1,
                    "macro_f1": macro_f1,
                    "model_name": model_name,
                    "epoch": int(epoch),
                }
                payload = {
                    "checkpoint_kind": "coteaching_peer_small_loss",
                    "epoch": int(epoch),
                    "best_epoch": int(epoch),
                    "best_macro_f1": macro_f1,
                    "best_class1_f1": class1_f1,
                    "model_state": model.state_dict(),
                    "model_config": checkpoint_source.get("model_config", {}),
                    "train_config": {
                        "method": "coteaching_peer_small_loss",
                        "checkpoint_a": str(Path(checkpoint_a).resolve()),
                        "checkpoint_b": str(Path(checkpoint_b).resolve()),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "max_train_batches": int(max_train_batches),
                        "max_val_batches": int(max_val_batches),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "remember_rate": float(remember_rate),
                        "grad_clip_norm": float(grad_clip_norm),
                        "bbox_token_prior_source": bbox_token_prior_source,
                    },
                    "data_yaml": str(Path(data_yaml).resolve()),
                    "class_names": list(class_names),
                    "metrics": to_serializable(metrics),
                    "data_summary": data_summary,
                    "source_checkpoint_kind": checkpoint_source.get("checkpoint_kind", ""),
                }
                save_checkpoint(checkpoints_dir / "best.pt", payload)
        save_evaluation_artifacts(eval_a, class_names, output_dir / f"val_eval_peer_a_epoch{epoch:02d}")
        save_evaluation_artifacts(eval_b, class_names, output_dir / f"val_eval_peer_b_epoch{epoch:02d}")
        row = {
            "epoch": int(epoch),
            "train_loss_a": totals["loss_a"] / max(1, seen_batches),
            "train_loss_b": totals["loss_b"] / max(1, seen_batches),
            "train_raw_loss_a": totals["raw_loss_a"] / max(1, seen_batches),
            "train_raw_loss_b": totals["raw_loss_b"] / max(1, seen_batches),
            "selected_overlap_fraction": totals["selected_overlap_fraction"] / max(1, seen_batches),
            "class1_selected_fraction_a": totals["class1_selected_by_a"] / max(1.0, totals["class1_count"]),
            "class1_selected_fraction_b": totals["class1_selected_by_b"] / max(1.0, totals["class1_count"]),
            "val_macro_f1_a": float(eval_a.get("macro_f1", 0.0)),
            "val_class1_f1_a": _class1_f1(eval_a),
            "val_macro_f1_b": float(eval_b.get("macro_f1", 0.0)),
            "val_class1_f1_b": _class1_f1(eval_b),
            "best_model": best.get("model_name", ""),
            "best_macro_f1": best.get("macro_f1", 0.0),
            "best_class1_f1": best.get("class1_f1", 0.0),
        }
        history_rows.append(row)
        _write_history(output_dir / "history.csv", history_rows)

    summary = {
        "output_dir": str(output_dir.resolve()),
        "checkpoint_a": str(Path(checkpoint_a).resolve()),
        "checkpoint_b": str(Path(checkpoint_b).resolve()),
        "data_yaml": str(Path(data_yaml).resolve()),
        "device": str(device),
        "image_size": int(image_size),
        "class_names": list(class_names),
        "data_summary": data_summary,
        "settings": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "max_train_batches": int(max_train_batches),
            "max_val_batches": int(max_val_batches),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "remember_rate": float(remember_rate),
            "grad_clip_norm": float(grad_clip_norm),
            "amp": bool(amp),
            "bbox_token_prior_source": bbox_token_prior_source,
        },
        "best": best,
        "history": history_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    json_dump(output_dir / "summary.json", to_serializable(summary))
    return summary


def _write_history(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune two TRKH checkpoints with peer small-loss co-teaching."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-train-batches", type=int, default=24)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--remember-rate", type=float, default=0.80)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--bbox-token-prior-source", choices=("bbox", "crop_bbox"), default="crop_bbox")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = train_coteaching(
        data_yaml=args.data,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        remember_rate=args.remember_rate,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not bool(args.no_amp),
        bbox_token_prior_source=args.bbox_token_prior_source,
    )
    print(json.dumps(to_serializable(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
