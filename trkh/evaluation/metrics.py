from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def confusion_matrix_from_predictions(
    targets: Tensor,
    predictions: Tensor,
    num_classes: int,
) -> Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics_from_confusion(
    confusion: Tensor,
    class_names: Sequence[str],
) -> Dict[str, object]:
    confusion = confusion.to(torch.float32)
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    true_positive = confusion.diag()

    precision = true_positive / predicted_support.clamp(min=1.0)
    recall = true_positive / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = true_positive.sum() / confusion.sum().clamp(min=1.0)

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support / support.sum().clamp(min=1.0)).sum()

    per_class: List[Dict[str, object]] = []
    for index, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "support": int(support[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
        )

    return {
        "accuracy": float(accuracy.item()),
        "macro_precision": float(macro_precision.item()),
        "macro_recall": float(macro_recall.item()),
        "macro_f1": float(macro_f1.item()),
        "weighted_f1": float(weighted_f1.item()),
        "per_class": per_class,
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def build_metrics(
    targets: Tensor,
    predictions: Tensor,
    class_names: Sequence[str],
    probabilities: Optional[Tensor] = None,
) -> Dict[str, object]:
    confusion = confusion_matrix_from_predictions(
        targets=targets,
        predictions=predictions,
        num_classes=len(class_names),
    )
    metrics = classification_metrics_from_confusion(confusion, class_names)
    if probabilities is not None:
        metrics["confidence_curves"] = build_confidence_curves(
            targets=targets,
            probabilities=probabilities,
            class_names=class_names,
        )
    return metrics


def build_thresholded_acceptance_summary(
    targets: Tensor,
    probabilities: Tensor,
    class_names: Sequence[str],
    threshold: float,
) -> Dict[str, object]:
    if probabilities.numel() == 0:
        return {
            "threshold": float(threshold),
            "coverage": 0.0,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "accepted_metrics": {},
        }

    max_probabilities, predictions = probabilities.max(dim=1)
    accepted_mask = max_probabilities >= float(threshold)
    accepted_targets = targets[accepted_mask]
    accepted_predictions = predictions[accepted_mask]
    coverage = float(accepted_mask.float().mean().item())

    if accepted_targets.numel() == 0:
        accepted_metrics: Dict[str, object] = {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": [],
            "confusion_matrix": [
                [0 for _ in range(len(class_names))] for _ in range(len(class_names))
            ],
        }
    else:
        accepted_confusion = confusion_matrix_from_predictions(
            targets=accepted_targets,
            predictions=accepted_predictions,
            num_classes=len(class_names),
        )
        accepted_metrics = classification_metrics_from_confusion(accepted_confusion, class_names)

    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "accepted_samples": int(accepted_mask.sum().item()),
        "rejected_samples": int((~accepted_mask).sum().item()),
        "accepted_metrics": accepted_metrics,
    }


def build_confidence_curves(
    targets: Tensor,
    probabilities: Tensor,
    class_names: Sequence[str],
    num_thresholds: int = 101,
) -> Dict[str, object]:
    thresholds = torch.linspace(0.0, 1.0, steps=max(2, num_thresholds), dtype=torch.float32)
    num_classes = len(class_names)

    if targets.numel() == 0 or probabilities.numel() == 0:
        zeros = torch.zeros_like(thresholds)
        return {
            "thresholds": thresholds.tolist(),
            "macro": {
                "precision": zeros.tolist(),
                "recall": zeros.tolist(),
                "f1": zeros.tolist(),
            },
            "per_class": [
                {
                    "class_index": index,
                    "class_name": class_name,
                    "precision": zeros.tolist(),
                    "recall": zeros.tolist(),
                    "f1": zeros.tolist(),
                    "best_f1": 0.0,
                    "best_confidence": 0.0,
                }
                for index, class_name in enumerate(class_names)
            ],
            "best_macro_f1": 0.0,
            "best_macro_f1_confidence": 0.0,
        }

    probabilities = probabilities.to(torch.float32)
    if probabilities.dim() != 2 or probabilities.size(1) != num_classes:
        raise ValueError("Probabilities phai co shape [N, num_classes].")

    targets_one_hot = torch.nn.functional.one_hot(
        targets.to(torch.int64),
        num_classes=num_classes,
    ).to(torch.bool)
    predicted_positive = probabilities.unsqueeze(0) >= thresholds.view(-1, 1, 1)
    target_positive = targets_one_hot.unsqueeze(0)

    true_positive = (predicted_positive & target_positive).sum(dim=1).to(torch.float32)
    false_positive = (predicted_positive & ~target_positive).sum(dim=1).to(torch.float32)
    false_negative = ((~predicted_positive) & target_positive).sum(dim=1).to(torch.float32)

    precision = true_positive / (true_positive + false_positive).clamp(min=1.0)
    recall = true_positive / (true_positive + false_negative).clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)

    macro_precision = precision.mean(dim=1)
    macro_recall = recall.mean(dim=1)
    macro_f1 = f1.mean(dim=1)
    best_macro_index = int(torch.argmax(macro_f1).item()) if macro_f1.numel() else 0

    per_class = []
    for index, class_name in enumerate(class_names):
        class_f1 = f1[:, index]
        best_class_index = int(torch.argmax(class_f1).item()) if class_f1.numel() else 0
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "precision": precision[:, index].tolist(),
                "recall": recall[:, index].tolist(),
                "f1": class_f1.tolist(),
                "best_f1": float(class_f1[best_class_index].item()) if class_f1.numel() else 0.0,
                "best_confidence": (
                    float(thresholds[best_class_index].item()) if class_f1.numel() else 0.0
                ),
            }
        )

    return {
        "thresholds": thresholds.tolist(),
        "macro": {
            "precision": macro_precision.tolist(),
            "recall": macro_recall.tolist(),
            "f1": macro_f1.tolist(),
        },
        "per_class": per_class,
        "best_macro_f1": float(macro_f1[best_macro_index].item()) if macro_f1.numel() else 0.0,
        "best_macro_f1_confidence": (
            float(thresholds[best_macro_index].item()) if macro_f1.numel() else 0.0
        ),
    }


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: Path,
    normalize: bool = False,
    normalize_by: str = "true",
) -> None:
    matrix = torch.tensor(confusion, dtype=torch.float32)
    num_classes = len(class_names)
    if matrix.ndim != 2 or matrix.shape != (num_classes, num_classes):
        raise ValueError(
            "Confusion matrix khong khop so lop hien tai: "
            f"shape={tuple(matrix.shape)}, num_classes={num_classes}"
        )
    normalized_title = ""
    if normalize:
        normalized_axis = str(normalize_by).strip().lower()
        if normalized_axis == "true":
            matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
            normalized_title = "Row-Normalized Confusion Matrix: P(Predicted | True)"
        elif normalized_axis == "predicted":
            matrix = matrix / matrix.sum(dim=0, keepdim=True).clamp(min=1.0)
            normalized_title = "Column-Normalized Confusion Matrix: P(True | Predicted)"
        else:
            raise ValueError("normalize_by chi nhan 'true' hoac 'predicted'.")

    figure_size = max(8.0, 1.9 * num_classes)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size - 0.8))
    image = axis.imshow(matrix.numpy(), interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title(normalized_title if normalize else "Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max().item() / 2.0 if matrix.numel() else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col].item()
            text = f"{value:.2f}" if normalize else str(int(value))
            axis.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_pr_panels(
    confidence_curves: Dict[str, object],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    per_class_curves = confidence_curves.get("per_class", [])
    if not per_class_curves:
        return

    num_classes = len(class_names)
    columns = 2 if num_classes > 1 else 1
    rows = int(math.ceil(num_classes / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 5.6 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    color_map = plt.get_cmap("tab10", max(1, num_classes))

    for index, axis in enumerate(axes.ravel()):
        if index >= num_classes:
            axis.set_axis_off()
            continue
        class_curve = per_class_curves[index]
        recall = class_curve.get("recall", [])
        precision = class_curve.get("precision", [])
        class_name = class_curve.get("class_name", class_names[index])
        best_f1 = float(class_curve.get("best_f1", 0.0))
        best_confidence = float(class_curve.get("best_confidence", 0.0))
        color = color_map(index % color_map.N)

        axis.plot(recall, precision, color=color, linewidth=2.0)
        axis.set_title(str(class_name))
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        axis.text(
            0.03,
            0.05,
            f"best F1={best_f1:.3f}\nthr={best_confidence:.2f}",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pr_curve(
    confidence_curves: Dict[str, object],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    thresholds = torch.tensor(confidence_curves.get("thresholds", []), dtype=torch.float32)
    macro_curves = confidence_curves.get("macro", {})
    per_class_curves = confidence_curves.get("per_class", [])

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    pr_axis, f1_axis = axes
    color_map = plt.get_cmap("tab10", max(1, len(class_names)))

    for index, class_curve in enumerate(per_class_curves):
        color = color_map(index % color_map.N)
        recall = class_curve.get("recall", [])
        precision = class_curve.get("precision", [])
        f1 = class_curve.get("f1", [])
        class_name = class_curve.get("class_name", class_names[index])

        pr_axis.plot(recall, precision, color=color, alpha=0.45, linewidth=1.5, label=class_name)
        f1_axis.plot(thresholds.tolist(), f1, color=color, alpha=0.45, linewidth=1.5)

    pr_axis.plot(
        macro_curves.get("recall", []),
        macro_curves.get("precision", []),
        color="black",
        linewidth=2.5,
        label="macro",
    )
    f1_axis.plot(
        thresholds.tolist(),
        macro_curves.get("f1", []),
        color="black",
        linewidth=2.5,
        label="macro",
    )

    best_confidence = float(confidence_curves.get("best_macro_f1_confidence", 0.0))
    best_f1 = float(confidence_curves.get("best_macro_f1", 0.0))
    f1_axis.axvline(
        best_confidence,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"best @{best_confidence:.2f}",
    )
    f1_axis.scatter([best_confidence], [best_f1], color="tab:red", s=45)

    pr_axis.set_title("Precision-Recall")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_xlim(0.0, 1.0)
    pr_axis.set_ylim(0.0, 1.02)
    pr_axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    pr_axis.legend(loc="lower left", fontsize=8)

    f1_axis.set_title("F1-Confidence")
    f1_axis.set_xlabel("Confidence threshold")
    f1_axis.set_ylabel("F1")
    f1_axis.set_xlim(0.0, 1.0)
    f1_axis.set_ylim(0.0, 1.02)
    f1_axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    f1_axis.legend(loc="lower left", fontsize=8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_metrics(
    per_class_metrics: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    if not per_class_metrics:
        return

    class_names = [str(item.get("class_name", item.get("class_index", ""))) for item in per_class_metrics]
    precision = [float(item.get("precision", 0.0)) for item in per_class_metrics]
    recall = [float(item.get("recall", 0.0)) for item in per_class_metrics]
    f1 = [float(item.get("f1", 0.0)) for item in per_class_metrics]
    support = [int(item.get("support", 0)) for item in per_class_metrics]

    x = np.arange(len(class_names), dtype=np.float32)
    width = 0.24

    figure_width = max(10.0, 1.4 * len(class_names))
    figure, axis = plt.subplots(figsize=(figure_width, 6))
    axis.bar(x - width, precision, width=width, label="precision")
    axis.bar(x, recall, width=width, label="recall")
    axis.bar(x + width, f1, width=width, label="f1")
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("score")
    axis.set_xticks(x)
    axis.set_xticklabels(class_names, rotation=30, ha="right")
    axis.set_title("Per-Class Metrics")
    axis.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    support_axis = axis.twinx()
    support_axis.plot(x, support, color="black", marker="o", linewidth=1.8, label="support")
    support_axis.set_ylabel("support")

    axis_handles, axis_labels = axis.get_legend_handles_labels()
    support_handles, support_labels = support_axis.get_legend_handles_labels()
    axis.legend(axis_handles + support_handles, axis_labels + support_labels, loc="lower left")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_detection_confidence_curve(
    detection_curve: Dict[str, object],
    output_path: Path,
) -> None:
    thresholds = torch.tensor(detection_curve.get("thresholds", []), dtype=torch.float32)
    if thresholds.numel() == 0:
        return

    precision = detection_curve.get("precision_50", [])
    recall = detection_curve.get("recall_50", [])
    f1 = detection_curve.get("f1_50", [])
    if not precision or not recall or not f1:
        return

    best_confidence = float(detection_curve.get("best_f1_50_confidence", 0.0))
    best_f1 = float(detection_curve.get("best_f1_50", 0.0))

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(thresholds.tolist(), precision, label="precision@0.5", linewidth=1.8)
    axis.plot(thresholds.tolist(), recall, label="recall@0.5", linewidth=1.8)
    axis.plot(thresholds.tolist(), f1, label="f1@0.5", linewidth=2.4)
    axis.axvline(
        best_confidence,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"best @{best_confidence:.2f}",
    )
    axis.scatter([best_confidence], [best_f1], color="tab:red", s=45)
    axis.set_title("Detection Confidence Curve")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("score")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axis.legend(loc="lower left")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
