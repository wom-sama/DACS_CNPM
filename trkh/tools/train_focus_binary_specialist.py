from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary image specialist for one focus class.")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val/test.")
    parser.add_argument("--focus-class-name", default="Xoai_Song_ChuaNhe_CoNguyCo")
    parser.add_argument("--model", default="mobilenetv3_large_100.ra_in1k")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument(
        "--negative-class-weight",
        action="append",
        default=[],
        metavar="CLASS=WEIGHT",
        help="Optional sample weight for negative original classes. Can be repeated.",
    )
    parser.add_argument(
        "--train-include-class-name",
        action="append",
        default=[],
        metavar="CLASS",
        help=(
            "Optional train-only class filter. Repeat to train a one-vs-one/boundary "
            "specialist while still evaluating/exporting on the full val/test split."
        ),
    )
    parser.add_argument(
        "--balanced-train-sampler",
        action="store_true",
        help="Balance positive/negative binary exposure inside the train-only specialist subset.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--eval-only-checkpoint",
        type=Path,
        default=None,
        help="Load an existing specialist checkpoint and only export val/test predictions.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_negative_class_weights(values: Sequence[str]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"--negative-class-weight expects CLASS=WEIGHT, got {value!r}")
        name, weight_text = str(value).split("=", 1)
        name = name.strip()
        weight = float(weight_text)
        if not name or weight <= 0.0:
            raise ValueError(f"Invalid negative class weight: {value!r}")
        output[name] = float(weight)
    return output


def build_transform(model_name: str, train: bool) -> transforms.Compose:
    cfg = timm.data.resolve_model_data_config(timm.create_model(model_name, pretrained=False))
    size = cfg.get("input_size", (3, 224, 224))[-2:]
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    ops: List[object] = [transforms.Resize(size, antialias=True)]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(8)], p=0.25),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.08,
                            hue=0.01,
                        )
                    ],
                    p=0.35,
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
    )
    if train:
        ops.append(transforms.RandomErasing(p=0.18, scale=(0.015, 0.08), ratio=(0.3, 3.3), value="random"))
    return transforms.Compose(ops)


class BinaryFocusImageFolder(ImageFolder):
    def __init__(
        self,
        root: Path,
        *,
        model_name: str,
        train: bool,
        focus_class_name: str,
        include_class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(str(root), transform=build_transform(model_name, train=train))
        if focus_class_name not in self.classes:
            raise ValueError(f"Unknown focus class {focus_class_name!r}; classes={self.classes}")
        self.focus_class_name = str(focus_class_name)
        self.focus_index = int(self.classes.index(focus_class_name))
        include_names = [str(value) for value in (include_class_names or []) if str(value)]
        if include_names:
            unknown = sorted(set(include_names) - set(self.classes))
            if unknown:
                raise ValueError(f"Unknown include classes: {unknown}; classes={self.classes}")
            include_indices = {int(self.classes.index(name)) for name in include_names}
            self.samples = [sample for sample in self.samples if int(sample[1]) in include_indices]
            self.imgs = self.samples
            self.targets = [int(sample[1]) for sample in self.samples]
            if not self.samples:
                raise ValueError(f"Class filter removed all samples for {root}: {include_names}")

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        binary_target = 1.0 if int(target) == self.focus_index else 0.0
        return image, torch.tensor(binary_target, dtype=torch.float32), int(target)


def make_loader(
    data: Path,
    split: str,
    *,
    model_name: str,
    focus_class_name: str,
    batch_size: int,
    workers: int,
    train: bool,
    include_class_names: Sequence[str] | None = None,
    balanced_train_sampler: bool = False,
) -> Tuple[BinaryFocusImageFolder, DataLoader]:
    dataset = BinaryFocusImageFolder(
        data / split,
        model_name=model_name,
        train=train,
        focus_class_name=focus_class_name,
        include_class_names=include_class_names if train else None,
    )
    sampler = None
    shuffle = bool(train)
    if train and bool(balanced_train_sampler):
        binary_targets = [1 if int(target) == int(dataset.focus_index) else 0 for target in dataset.targets]
        positive_count = max(1, int(sum(binary_targets)))
        negative_count = max(1, int(len(binary_targets) - positive_count))
        weights = [
            0.5 / positive_count if int(target) == 1 else 0.5 / negative_count
            for target in binary_targets
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(workers) > 0,
        drop_last=bool(train and len(dataset) >= int(batch_size)),
    )
    return dataset, loader


def _binary_metrics(
    targets: Sequence[int],
    probabilities: Sequence[float],
    *,
    thresholds: Sequence[float] | None = None,
) -> Dict[str, object]:
    y_true = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if thresholds is None:
        thresholds = [round(value, 4) for value in np.linspace(0.05, 0.95, 181)]
    rows = []
    best = None
    best_key = (-1.0, -1.0, -1.0)
    for threshold in thresholds:
        y_pred = (probs >= float(threshold)).astype(np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
        row = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "negative_precision": float(precision[0]),
            "negative_recall": float(recall[0]),
            "negative_f1": float(f1[0]),
            "negative_support": int(support[0]),
            "focus_precision": float(precision[1]),
            "focus_recall": float(recall[1]),
            "focus_f1": float(f1[1]),
            "focus_support": int(support[1]),
            "predicted_focus": int(y_pred.sum()),
        }
        rows.append(row)
        key = (row["focus_f1"], row["focus_precision"], row["focus_recall"])
        if key > best_key:
            best_key = key
            best = row
    if best is None:
        raise RuntimeError("No binary threshold metrics were computed.")
    return {"best": best, "thresholds": rows}


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    *,
    max_batches: int = 0,
) -> Tuple[float, List[int], List[int], List[float], float]:
    model.eval()
    losses: List[float] = []
    binary_targets: List[int] = []
    original_targets: List[int] = []
    probabilities: List[float] = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, targets, original) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_device, enabled=bool(amp) and device.type == "cuda"):
                logits = model(images).flatten()
                loss = criterion(logits, targets)
            losses.append(float(loss.mean().item()))
            probs = logits.sigmoid()
            binary_targets.extend(targets.long().cpu().tolist())
            original_targets.extend([int(value) for value in original])
            probabilities.extend(probs.cpu().tolist())
            if max_batches and batch_index >= int(max_batches):
                break
    elapsed = time.perf_counter() - start
    return (
        float(np.mean(losses)) if losses else 0.0,
        binary_targets,
        original_targets,
        probabilities,
        float(elapsed * 1000.0 / max(1, len(binary_targets))),
    )


def export_predictions(
    path: Path,
    dataset: BinaryFocusImageFolder,
    binary_targets: Sequence[int],
    original_targets: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples[: len(binary_targets)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "true_name",
            "original_target",
            "binary_target",
            "focus_class_name",
            "focus_probability",
            "binary_pred",
            "threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (image_path, _), binary_target, original_target, probability in zip(
            samples,
            binary_targets,
            original_targets,
            probabilities,
        ):
            writer.writerow(
                {
                    "path": str(image_path),
                    "true_name": dataset.classes[int(original_target)],
                    "original_target": int(original_target),
                    "binary_target": int(binary_target),
                    "focus_class_name": dataset.focus_class_name,
                    "focus_probability": float(probability),
                    "binary_pred": int(float(probability) >= float(threshold)),
                    "threshold": float(threshold),
                }
            )


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader = make_loader(
        args.data,
        "train",
        model_name=args.model,
        focus_class_name=args.focus_class_name,
        batch_size=args.batch_size,
        workers=args.workers,
        train=True,
        include_class_names=args.train_include_class_name,
        balanced_train_sampler=bool(args.balanced_train_sampler),
    )
    val_dataset, val_loader = make_loader(
        args.data,
        "val",
        model_name=args.model,
        focus_class_name=args.focus_class_name,
        batch_size=args.batch_size,
        workers=args.workers,
        train=False,
    )
    test_dataset, test_loader = make_loader(
        args.data,
        "test",
        model_name=args.model,
        focus_class_name=args.focus_class_name,
        batch_size=args.batch_size,
        workers=args.workers,
        train=False,
    )
    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Class order differs across splits.")
    negative_class_weights_by_name = parse_negative_class_weights(args.negative_class_weight)
    unknown_negative_weight_classes = sorted(set(negative_class_weights_by_name) - set(train_dataset.classes))
    if unknown_negative_weight_classes:
        raise ValueError(f"Unknown --negative-class-weight classes: {unknown_negative_weight_classes}")
    negative_class_weights = torch.ones(len(train_dataset.classes), dtype=torch.float32, device=device)
    for class_name, weight in negative_class_weights_by_name.items():
        negative_class_weights[int(train_dataset.classes.index(class_name))] = float(weight)

    original_targets = np.asarray(train_dataset.targets, dtype=np.int64)
    positive_count = int((original_targets == train_dataset.focus_index).sum())
    negative_count = int(len(original_targets) - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Binary focus training needs positive and negative samples.")
    pos_weight = (negative_count / positive_count) ** float(args.pos_weight_power)
    pos_weight *= float(args.pos_weight_scale)

    model = timm.create_model(
        args.model,
        pretrained=not bool(args.no_pretrained),
        num_classes=1,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    history: List[Dict[str, object]] = []
    smoothing = min(max(float(args.label_smoothing), 0.0), 0.49)
    if args.eval_only_checkpoint is not None:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_threshold = float(checkpoint.get("best_threshold", 0.5))
        best_score = float(checkpoint.get("best_val_focus_f1", -1.0))
    else:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            train_losses: List[float] = []
            for batch_index, (images, targets, original) in enumerate(
                tqdm(train_loader, desc=f"focus-binary {epoch}/{args.epochs}"),
                start=1,
            ):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if smoothing > 0.0:
                    targets = targets * (1.0 - smoothing) + (1.0 - targets) * smoothing
                optimizer.zero_grad(set_to_none=True)
                original = original.to(device, non_blocking=True)
                with torch.autocast(device_type=autocast_device, enabled=bool(args.amp) and device.type == "cuda"):
                    logits = model(images).flatten()
                    element_loss = criterion(logits, targets)
                    sample_weights = torch.ones_like(element_loss)
                    negative_mask = targets < 0.5
                    if bool(negative_mask.any()):
                        sample_weights = torch.where(
                            negative_mask,
                            negative_class_weights.index_select(0, original.long()),
                            sample_weights,
                        )
                    loss = (element_loss * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                if args.max_train_batches and batch_index >= int(args.max_train_batches):
                    break
            scheduler.step()

            val_loss, val_binary, _, val_probabilities, val_ms = run_eval(
                model,
                val_loader,
                criterion,
                device,
                bool(args.amp),
                max_batches=int(args.max_eval_batches),
            )
            val_metrics = _binary_metrics(val_binary, val_probabilities)
            val_best = val_metrics["best"]
            row = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "val_loss": float(val_loss),
                "val_focus_f1": float(val_best["focus_f1"]),
                "val_focus_precision": float(val_best["focus_precision"]),
                "val_focus_recall": float(val_best["focus_recall"]),
                "val_threshold": float(val_best["threshold"]),
                "val_predicted_focus": int(val_best["predicted_focus"]),
                "val_ms_per_image": float(val_ms),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if float(val_best["focus_f1"]) > best_score:
                best_score = float(val_best["focus_f1"])
                best_epoch = int(epoch)
                best_threshold = float(val_best["threshold"])
                stale_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "classes": train_dataset.classes,
                        "args": vars(args),
                        "focus_class_name": train_dataset.focus_class_name,
                        "focus_index": int(train_dataset.focus_index),
                        "best_epoch": best_epoch,
                        "best_threshold": best_threshold,
                        "best_val_focus_f1": best_score,
                        "pos_weight": float(pos_weight),
                    },
                    output_dir / "best.pt",
                )
            else:
                stale_epochs += 1
                if int(args.patience) > 0 and stale_epochs >= int(args.patience):
                    print(
                        json.dumps(
                            {"early_stop": True, "epoch": int(epoch), "best_epoch": best_epoch},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break

        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_threshold = float(checkpoint["best_threshold"])

    val_loss, val_binary, val_original, val_probabilities, val_ms = run_eval(
        model,
        val_loader,
        criterion,
        device,
        bool(args.amp),
        max_batches=int(args.max_eval_batches),
    )
    val_metrics = _binary_metrics(val_binary, val_probabilities)
    export_predictions(
        output_dir / "predictions_val.csv",
        val_dataset,
        val_binary,
        val_original,
        val_probabilities,
        best_threshold,
    )

    test_metrics: Dict[str, object] = {}
    if not bool(args.skip_test):
        test_loss, test_binary, test_original, test_probabilities, test_ms = run_eval(
            model,
            test_loader,
            criterion,
            device,
            bool(args.amp),
            max_batches=int(args.max_eval_batches),
        )
        test_metrics = _binary_metrics(test_binary, test_probabilities)
        export_predictions(
            output_dir / "predictions_test.csv",
            test_dataset,
            test_binary,
            test_original,
            test_probabilities,
            best_threshold,
        )
    else:
        test_loss = 0.0
        test_ms = 0.0

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "mode": "focus_binary_specialist",
        "model": str(args.model),
        "pretrained": not bool(args.no_pretrained),
        "focus_class_name": train_dataset.focus_class_name,
        "focus_index": int(train_dataset.focus_index),
        "classes": train_dataset.classes,
        "train_positive_count": positive_count,
        "train_negative_count": negative_count,
        "pos_weight": float(pos_weight),
        "negative_class_weights": negative_class_weights_by_name,
        "train_include_class_names": [str(value) for value in args.train_include_class_name],
        "balanced_train_sampler": bool(args.balanced_train_sampler),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "best_val_focus_f1": float(best_score),
        "val_loss": float(val_loss),
        "val_ms_per_image": float(val_ms),
        "val_metrics": val_metrics,
        "test_loss": float(test_loss),
        "test_ms_per_image": float(test_ms),
        "test_metrics": test_metrics,
        "probe_limits": {
            "max_train_batches": int(args.max_train_batches),
            "max_eval_batches": int(args.max_eval_batches),
            "patience": int(args.patience),
            "skip_test": bool(args.skip_test),
            "eval_only_checkpoint": None if args.eval_only_checkpoint is None else str(args.eval_only_checkpoint),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_epoch": int(best_epoch),
                "best_threshold": float(best_threshold),
                "best_val_focus_f1": float(best_score),
                "val_focus_precision": float(val_metrics["best"]["focus_precision"]),
                "val_focus_recall": float(val_metrics["best"]["focus_recall"]),
                "skip_test": bool(args.skip_test),
                "metrics": str(output_dir / "metrics.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
