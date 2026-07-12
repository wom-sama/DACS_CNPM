from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from trkh.tools.audit_lbp_surface_texture_readiness import _focus
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _direction_auc,
    _load_cache,
    _transition_stats,
    _write_csv,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)


SEED = 20260712
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 512
RESNET_DIM = 2048
VIT_DIM = 768
HIDDEN_DIM_1 = 1024
HIDDEN_DIM_2 = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05
FOCUS_CLASS_INDEX = 1
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
BRANCHES = ("resnet", "vit", "fusion")
FEATURE_NORMALIZATIONS = ("raw", "block_l2")
LITERATURE = (
    "https://huggingface.co/timm/resnet50.a1_in1k",
    "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "https://arxiv.org/abs/2110.00476",
    "https://arxiv.org/abs/2106.10270",
)


class AIDTReadout(nn.Module):
    """The nonlinear classifier used by the local AIDT competitor."""

    def __init__(
        self,
        input_dim: int,
        class_count: int,
        *,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim_1)),
            nn.BatchNorm1d(int(hidden_dim_1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(int(hidden_dim_1), int(hidden_dim_2)),
            nn.BatchNorm1d(int(hidden_dim_2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(hidden_dim_2), int(class_count)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-grouped no-test readiness audit for raw-pretrained AIDT "
            "ResNet, ViT, and concatenated features. Heads are RAM-only."
        )
    )
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--keeper-val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resnet-dim", type=int, default=RESNET_DIM)
    parser.add_argument("--vit-dim", type=int, default=VIT_DIM)
    parser.add_argument("--hidden-dim-1", type=int, default=HIDDEN_DIM_1)
    parser.add_argument("--hidden-dim-2", type=int, default=HIDDEN_DIM_2)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--feature-normalization",
        choices=FEATURE_NORMALIZATIONS,
        default="raw",
    )
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().casefold()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_feature_cache(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if "test" in path.name.casefold():
        raise ValueError(f"test feature cache is forbidden: {path}")
    with np.load(path, allow_pickle=True) as payload:
        required = (
            "features",
            "labels",
            "paths",
            "sample_index",
            "source_stem",
            "object_index",
            "classes",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"feature cache is missing fields {missing}: {path}")
        result = {name: np.asarray(payload[name]) for name in required}
    result["features"] = np.asarray(result["features"], dtype=np.float32)
    result["labels"] = np.asarray(result["labels"], dtype=np.int64).reshape(-1)
    result["sample_index"] = np.asarray(result["sample_index"], dtype=np.int64).reshape(-1)
    result["object_index"] = np.asarray(result["object_index"], dtype=np.int64).reshape(-1)
    result["paths"] = np.asarray(result["paths"], dtype=object).reshape(-1)
    result["source_stem"] = np.asarray(result["source_stem"], dtype=object).reshape(-1)
    result["classes"] = np.asarray(result["classes"], dtype=object).reshape(-1)
    row_count = len(result["labels"])
    if result["features"].ndim != 2 or result["features"].shape[0] != row_count:
        raise ValueError(f"feature cache rows do not align: {path}")
    for name in ("paths", "sample_index", "source_stem", "object_index"):
        if len(result[name]) != row_count:
            raise ValueError(f"feature cache field {name} does not align: {path}")
    if not np.isfinite(result["features"]).all():
        raise ValueError(f"feature cache contains non-finite values: {path}")
    if np.unique(result["sample_index"]).size != row_count:
        raise ValueError(f"feature cache sample_index is not unique: {path}")
    class_count = len(result["classes"])
    if class_count < 2 or len(set(str(value) for value in result["classes"])) != class_count:
        raise ValueError(f"feature cache classes are invalid: {path}")
    if row_count == 0 or np.any(result["labels"] < 0) or np.any(result["labels"] >= class_count):
        raise ValueError(f"feature cache labels are out of range: {path}")
    if any(not str(value).strip() for value in result["source_stem"]):
        raise ValueError(f"feature cache contains an empty source stem: {path}")
    return result


def split_component_features(
    features: np.ndarray,
    *,
    resnet_dim: int,
    vit_dim: int,
) -> Dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or int(resnet_dim) <= 0 or int(vit_dim) <= 0:
        raise ValueError("feature matrix/dimensions are invalid")
    if values.shape[1] != int(resnet_dim) + int(vit_dim):
        raise ValueError(
            "combined feature dimension mismatch: "
            f"{values.shape[1]} != {int(resnet_dim)} + {int(vit_dim)}"
        )
    return {
        "resnet": values[:, : int(resnet_dim)],
        "vit": values[:, int(resnet_dim) :],
        "fusion": values,
    }


def normalize_component_features(
    views: Mapping[str, np.ndarray],
    *,
    mode: str,
) -> Dict[str, np.ndarray]:
    normalized_mode = str(mode).strip().casefold()
    if normalized_mode not in FEATURE_NORMALIZATIONS:
        raise ValueError(f"unsupported feature normalization: {mode}")
    resnet = np.asarray(views["resnet"], dtype=np.float32)
    vit = np.asarray(views["vit"], dtype=np.float32)
    if normalized_mode == "raw":
        return {"resnet": resnet, "vit": vit, "fusion": np.asarray(views["fusion"], dtype=np.float32)}

    def l2_normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
            raise ValueError("block L2 normalization encountered an invalid row norm")
        output = values / norms.astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("block L2 normalization produced non-finite values")
        return output.astype(np.float32, copy=False)

    normalized_resnet = l2_normalize(resnet)
    normalized_vit = l2_normalize(vit)
    return {
        "resnet": normalized_resnet,
        "vit": normalized_vit,
        "fusion": np.concatenate((normalized_resnet, normalized_vit), axis=1),
    }


def sqrt_inverse_class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(class_count))
    if np.any(counts <= 0):
        raise ValueError("sqrt-inverse weighting requires every class")
    weights = np.sqrt(float(counts.sum()) / counts.astype(np.float64))
    weights /= weights.mean()
    return weights.astype(np.float32)


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    rows: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            batch = torch.from_numpy(
                np.asarray(features[start : start + int(batch_size)], dtype=np.float32)
            ).to(device=device)
            rows.append(model(batch).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    if probabilities.shape[0] != len(features) or not np.isfinite(probabilities).all():
        raise RuntimeError("AIDT readout produced invalid probabilities")
    return probabilities


def train_readout(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    label_smoothing: float,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, object]]:
    _seed_everything(seed)
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    model = AIDTReadout(
        values.shape[1],
        class_count,
        hidden_dim_1=int(hidden_dim_1),
        hidden_dim_2=int(hidden_dim_2),
    ).to(device=device)
    class_weights = torch.from_numpy(
        sqrt_inverse_class_weights(targets, class_count)
    ).to(device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    warmup = min(int(warmup_epochs), max(0, int(epochs) - 1))
    if warmup > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=warmup,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(epochs) - warmup),
                ),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
        )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(targets)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("batch size leaves no complete AIDT readout batch")

    curve: List[Dict[str, float]] = []
    finite = True
    minimum_batch_class_support = int(batch_size)
    start = time.time()
    for epoch_index in range(int(epochs)):
        model.train()
        losses: List[float] = []
        for batch_features, batch_labels in loader:
            support = torch.bincount(batch_labels, minlength=int(class_count))
            minimum_batch_class_support = min(
                minimum_batch_class_support,
                int(support.min().item()),
            )
            batch_features = batch_features.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            if not bool(torch.isfinite(loss).item()):
                finite = False
                raise FloatingPointError("AIDT readout loss became non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        curve.append(
            {
                "epoch": int(epoch_index + 1),
                "loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()
    telemetry = {
        "finite": bool(finite),
        "epochs": int(len(curve)),
        "batches_per_epoch": int(len(loader)),
        "samples_per_epoch": int(len(loader) * int(batch_size)),
        "minimum_batch_class_support": int(minimum_batch_class_support),
        "seconds": float(time.time() - start),
        "first": dict(curve[0]),
        "last": dict(curve[-1]),
        "loss_decreased": bool(curve[-1]["loss"] < curve[0]["loss"]),
        "class_weights": [float(value) for value in class_weights.cpu().tolist()],
    }
    return model, curve, telemetry


def _align_keeper_probabilities(
    feature_cache: Mapping[str, np.ndarray],
    keeper_cache: Mapping[str, np.ndarray],
) -> np.ndarray:
    keeper_indices = np.asarray(keeper_cache["sample_index"], dtype=np.int64)
    if np.unique(keeper_indices).size != len(keeper_indices):
        raise ValueError("keeper cache sample_index is not unique")
    lookup = {int(value): index for index, value in enumerate(keeper_indices.tolist())}
    rows = []
    for sample_index in np.asarray(feature_cache["sample_index"], dtype=np.int64):
        if int(sample_index) not in lookup:
            raise ValueError(f"keeper cache is missing sample_index={int(sample_index)}")
        rows.append(int(lookup[int(sample_index)]))
    if len(set(rows)) != len(rows):
        raise ValueError("keeper alignment is not one-to-one")
    aligned_labels = np.asarray(keeper_cache["labels"], dtype=np.int64)[rows]
    if not np.array_equal(aligned_labels, np.asarray(feature_cache["labels"], dtype=np.int64)):
        raise ValueError("keeper/feature labels differ after sample-index alignment")
    return np.asarray(keeper_cache["probabilities"], dtype=np.float32)[rows]


def _selected_control(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    focus_class_index: int,
) -> str:
    component_names = ("resnet", "vit")
    return max(
        component_names,
        key=lambda name: (
            0.5 * float(metrics[name]["macro_f1"])
            + 0.5 * float(_focus(metrics[name], int(focus_class_index))["f1"]),
            name,
        ),
    )


def assess_raw_aidt_fusion_readiness(
    *,
    train_rows: int,
    val_rows: int,
    train_source_groups: int,
    train_val_source_overlap: int,
    maximum_fold_source_overlap: int,
    all_training_finite: bool,
    all_losses_decreased: bool,
    minimum_batch_class_support: int,
    folds_fusion_wins_focus: int,
    fold_count: int,
    oof_control_metrics: Mapping[str, object],
    oof_fusion_metrics: Mapping[str, object],
    val_control_metrics: Mapping[str, object],
    val_fusion_metrics: Mapping[str, object],
    val_keeper_metrics: Mapping[str, object],
    oof_transitions: Mapping[str, int],
    val_control_transitions: Mapping[str, int],
    val_keeper_transitions: Mapping[str, int],
    oof_direction: Mapping[str, object],
    val_direction: Mapping[str, object],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    oof_control_focus = _focus(oof_control_metrics, focus)
    oof_fusion_focus = _focus(oof_fusion_metrics, focus)
    val_control_focus = _focus(val_control_metrics, focus)
    val_fusion_focus = _focus(val_fusion_metrics, focus)
    keeper_focus = _focus(val_keeper_metrics, focus)
    oof_auc = oof_direction.get("auc_fn_positive")
    val_auc = val_direction.get("auc_fn_positive")
    directions_available = bool(
        oof_auc is not None
        and val_auc is not None
        and np.isfinite(float(oof_auc))
        and np.isfinite(float(val_auc))
    )
    observed = {
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_source_groups": int(train_source_groups),
        "train_val_source_overlap": int(train_val_source_overlap),
        "maximum_fold_source_overlap": int(maximum_fold_source_overlap),
        "all_training_finite": bool(all_training_finite),
        "all_losses_decreased": bool(all_losses_decreased),
        "minimum_batch_class_support": int(minimum_batch_class_support),
        "folds_fusion_wins_focus": int(folds_fusion_wins_focus),
        "fold_count": int(fold_count),
        "oof_macro_gain": float(oof_fusion_metrics["macro_f1"] - oof_control_metrics["macro_f1"]),
        "oof_focus_gain": float(oof_fusion_focus["f1"] - oof_control_focus["f1"]),
        "val_macro_gain": float(val_fusion_metrics["macro_f1"] - val_control_metrics["macro_f1"]),
        "val_focus_gain": float(val_fusion_focus["f1"] - val_control_focus["f1"]),
        "keeper_macro_f1": float(val_keeper_metrics["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "fusion_macro_f1": float(val_fusion_metrics["macro_f1"]),
        "fusion_focus_f1": float(val_fusion_focus["f1"]),
        "fusion_focus_precision": float(val_fusion_focus["precision"]),
        "fusion_focus_recall": float(val_fusion_focus["recall"]),
        "fusion_keeper_macro_gain": float(val_fusion_metrics["macro_f1"] - val_keeper_metrics["macro_f1"]),
        "fusion_keeper_focus_gain": float(val_fusion_focus["f1"] - keeper_focus["f1"]),
        "directions_available": directions_available,
        "oof_direction_auc": float(oof_auc) if directions_available else -1.0,
        "val_direction_auc": float(val_auc) if directions_available else -1.0,
        "oof_transitions": dict(oof_transitions),
        "val_control_transitions": dict(val_control_transitions),
        "val_keeper_transitions": dict(val_keeper_transitions),
        "test_split_used": bool(test_split_used),
    }
    checks = {
        "full_train_9215": observed["train_rows"] == EXPECTED_TRAIN_ROWS,
        "full_val_2606": observed["val_rows"] == EXPECTED_VAL_ROWS,
        "test_not_used": not bool(test_split_used),
        "train_source_group_support": observed["train_source_groups"] >= 8000,
        "train_val_sources_disjoint": observed["train_val_source_overlap"] == 0,
        "fold_sources_disjoint": observed["maximum_fold_source_overlap"] == 0,
        "training_finite": bool(all_training_finite),
        "all_losses_decreased": bool(all_losses_decreased),
        "all_batches_have_every_class": observed["minimum_batch_class_support"] >= 1,
        "fusion_focus_wins_at_least_3_folds": observed["folds_fusion_wins_focus"] >= min(3, observed["fold_count"]),
        "oof_macro_gain_ge_0p003": observed["oof_macro_gain"] >= 0.003,
        "oof_focus_gain_ge_0p01": observed["oof_focus_gain"] >= 0.01,
        "val_macro_gain_ge_0p002": observed["val_macro_gain"] >= 0.002,
        "val_focus_gain_ge_0p01": observed["val_focus_gain"] >= 0.01,
        "oof_corrections_ge_harms": int(oof_transitions["corrections"]) >= int(oof_transitions["harms"]),
        "val_control_corrections_ge_harms": int(val_control_transitions["corrections"]) >= int(val_control_transitions["harms"]),
        "directions_available": directions_available,
        "oof_direction_auc_ge_0p60": observed["oof_direction_auc"] >= 0.60,
        "val_direction_auc_ge_0p60": observed["val_direction_auc"] >= 0.60,
        "keeper_macro_preserved_within_0p001": observed["fusion_keeper_macro_gain"] >= -0.001,
        "keeper_focus_improved_by_0p01": observed["fusion_keeper_focus_gain"] >= 0.01,
        "fusion_focus_reaches_0p70": observed["fusion_focus_f1"] >= 0.70,
        "keeper_focus_recall_preserved_within_0p01": observed["fusion_focus_recall"] >= observed["keeper_focus_recall"] - 0.01,
        "keeper_corrections_ge_harms": int(val_keeper_transitions["corrections"]) >= int(val_keeper_transitions["harms"]),
        "keeper_fp_removed_ge_created": int(val_keeper_transitions["focus_false_positive_removed"]) >= int(val_keeper_transitions["focus_false_positive_created"]),
        "keeper_fn_rescued_ge_tp_broken": int(val_keeper_transitions["focus_false_negative_rescued"]) >= int(val_keeper_transitions["focus_true_positive_broken"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "fold_safe_teacher_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    selected_control: str,
    keeper_probabilities: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    labels = np.asarray(cache["labels"], dtype=np.int64)
    for index in range(len(labels)):
        fusion_prediction = int(probabilities["fusion"][index].argmax())
        row: Dict[str, object] = {
            "split": str(split),
            "sample_index": int(cache["sample_index"][index]),
            "fold": int(fold_assignment[index]),
            "source_stem": str(cache["source_stem"][index]),
            "object_index": int(cache["object_index"][index]),
            "image_path": str(cache["paths"][index]),
            "target_index": int(labels[index]),
            "prediction_index": fusion_prediction,
            "y_true": int(labels[index]),
            "y_pred": fusion_prediction,
            "selected_control": str(selected_control),
        }
        for branch in BRANCHES:
            row[f"{branch}_prediction_index"] = int(probabilities[branch][index].argmax())
            for class_index in range(probabilities[branch].shape[1]):
                row[f"{branch}_prob_{class_index}"] = float(
                    probabilities[branch][index, class_index]
                )
        if keeper_probabilities is not None:
            row["keeper_prediction_index"] = int(keeper_probabilities[index].argmax())
            for class_index in range(keeper_probabilities.shape[1]):
                row[f"keeper_prob_{class_index}"] = float(
                    keeper_probabilities[index, class_index]
                )
        rows.append(row)
    return rows


def _plot_summary(
    path: Path,
    *,
    curves: Sequence[Mapping[str, object]],
    oof_metrics: Mapping[str, Mapping[str, object]],
    val_metrics: Mapping[str, Mapping[str, object]],
    focus_class_index: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for branch in BRANCHES:
        rows = [
            row
            for row in curves
            if str(row["scope"]) == "full_train" and str(row["branch"]) == branch
        ]
        axes[0].plot(
            [int(row["epoch"]) for row in rows],
            [float(row["loss"]) for row in rows],
            label=branch,
        )
    axes[0].set_title("Full-train readout loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted CE")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    names = [f"OOF {name}" for name in BRANCHES] + [f"val {name}" for name in BRANCHES] + ["val keeper"]
    metric_rows = [oof_metrics[name] for name in BRANCHES] + [val_metrics[name] for name in BRANCHES] + [val_metrics["keeper"]]
    positions = np.arange(len(names))
    axes[1].bar(
        positions - 0.18,
        [float(row["macro_f1"]) for row in metric_rows],
        width=0.36,
        label="macro F1",
    )
    axes[1].bar(
        positions + 0.18,
        [float(_focus(row, focus_class_index)["f1"]) for row in metric_rows],
        width=0.36,
        label="class-1 F1",
    )
    axes[1].set_xticks(positions, names, rotation=30, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Raw-pretrained AIDT readiness")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) < 3:
        raise ValueError("folds must be at least 3")
    if not 1 <= int(args.epochs) <= 30:
        raise ValueError("epochs must stay within [1,30]")
    if int(args.batch_size) < 64:
        raise ValueError("batch-size must be at least 64")
    if int(args.hidden_dim_1) < 16 or int(args.hidden_dim_2) < 16:
        raise ValueError("hidden dimensions are too small")
    if int(args.resnet_dim) <= 0 or int(args.vit_dim) <= 0:
        raise ValueError("component dimensions must be positive")
    if (
        float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or int(args.warmup_epochs) < 0
        or not 0.0 <= float(args.label_smoothing) < 1.0
    ):
        raise ValueError("optimizer/loss settings are invalid")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = _resolve_device(str(args.device))
    start = time.time()

    train = load_feature_cache(Path(args.train_npz))
    val = load_feature_cache(Path(args.val_npz))
    if train["classes"].tolist() != val["classes"].tolist():
        raise ValueError("train/validation class order differs")
    class_count = len(train["classes"])
    if np.unique(train["labels"]).size != class_count or np.unique(val["labels"]).size != class_count:
        raise ValueError("train/validation caches do not cover every class")
    train_views = normalize_component_features(
        split_component_features(
            train["features"],
            resnet_dim=int(args.resnet_dim),
            vit_dim=int(args.vit_dim),
        ),
        mode=str(args.feature_normalization),
    )
    val_views = normalize_component_features(
        split_component_features(
            val["features"],
            resnet_dim=int(args.resnet_dim),
            vit_dim=int(args.vit_dim),
        ),
        mode=str(args.feature_normalization),
    )
    keeper_val = _load_cache(Path(args.keeper_val_cache), split="val")
    aligned_keeper = _align_keeper_probabilities(val, keeper_val)
    y_train = np.asarray(train["labels"], dtype=np.int64)
    y_val = np.asarray(val["labels"], dtype=np.int64)
    train_groups = np.asarray(
        [str(value).casefold() for value in train["source_stem"]],
        dtype=object,
    )
    val_groups = np.asarray(
        [str(value).casefold() for value in val["source_stem"]],
        dtype=object,
    )
    train_val_source_overlap = len(set(train_groups.tolist()).intersection(val_groups.tolist()))

    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    oof_probabilities = {
        branch: np.zeros((len(y_train), class_count), dtype=np.float32)
        for branch in BRANCHES
    }
    fold_assignment = np.full(len(y_train), -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    fold_protocols: List[Dict[str, object]] = []
    maximum_fold_source_overlap = 0
    all_training_finite = True
    all_losses_decreased = True
    minimum_batch_class_support = int(args.batch_size)
    folds_fusion_wins_focus = 0

    for fold_index, (fit_indices, hold_indices) in enumerate(
        splitter.split(train_views["fusion"], y_train, train_groups)
    ):
        fit_sources = set(train_groups[fit_indices].tolist())
        hold_sources = set(train_groups[hold_indices].tolist())
        overlap = len(fit_sources.intersection(hold_sources))
        maximum_fold_source_overlap = max(maximum_fold_source_overlap, overlap)
        branch_metrics: Dict[str, Mapping[str, object]] = {}
        branch_protocol: Dict[str, object] = {}
        for branch_index, branch in enumerate(BRANCHES):
            model, curve, telemetry = train_readout(
                train_views[branch][fit_indices],
                y_train[fit_indices],
                class_count=class_count,
                hidden_dim_1=int(args.hidden_dim_1),
                hidden_dim_2=int(args.hidden_dim_2),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                warmup_epochs=int(args.warmup_epochs),
                label_smoothing=float(args.label_smoothing),
                seed=int(args.seed) + int(fold_index),
                device=device,
            )
            probabilities = _predict_probabilities(
                model,
                train_views[branch][hold_indices],
                device=device,
            )
            oof_probabilities[branch][hold_indices] = probabilities
            metrics = _classification_metrics(y_train[hold_indices], probabilities)
            branch_metrics[branch] = metrics
            branch_protocol[branch] = telemetry
            all_training_finite = all_training_finite and bool(telemetry["finite"])
            all_losses_decreased = all_losses_decreased and bool(telemetry["loss_decreased"])
            minimum_batch_class_support = min(
                minimum_batch_class_support,
                int(telemetry["minimum_batch_class_support"]),
            )
            for row in curve:
                curve_rows.append(
                    {"scope": f"fold_{fold_index}", "branch": branch, **row}
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_assignment[hold_indices] = int(fold_index)
        best_component_focus = max(
            float(_focus(branch_metrics[name], int(args.focus_class_index))["f1"])
            for name in ("resnet", "vit")
        )
        fusion_focus = float(
            _focus(branch_metrics["fusion"], int(args.focus_class_index))["f1"]
        )
        folds_fusion_wins_focus += int(fusion_focus > best_component_focus)
        fold_row: Dict[str, object] = {
            "fold": int(fold_index),
            "fit_rows": int(len(fit_indices)),
            "hold_rows": int(len(hold_indices)),
            "source_overlap": int(overlap),
        }
        for branch in BRANCHES:
            fold_row[f"{branch}_macro_f1"] = float(branch_metrics[branch]["macro_f1"])
            fold_row[f"{branch}_focus_f1"] = float(
                _focus(branch_metrics[branch], int(args.focus_class_index))["f1"]
            )
        fold_row["fusion_focus_gain_vs_best_component"] = float(
            fusion_focus - best_component_focus
        )
        fold_rows.append(fold_row)
        fold_protocols.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(overlap),
                "branches": branch_protocol,
            }
        )
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF assignment is incomplete")

    val_probabilities: Dict[str, np.ndarray] = {}
    full_protocol: Dict[str, object] = {}
    for branch in BRANCHES:
        model, curve, telemetry = train_readout(
            train_views[branch],
            y_train,
            class_count=class_count,
            hidden_dim_1=int(args.hidden_dim_1),
            hidden_dim_2=int(args.hidden_dim_2),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            warmup_epochs=int(args.warmup_epochs),
            label_smoothing=float(args.label_smoothing),
            seed=int(args.seed) + int(args.folds),
            device=device,
        )
        val_probabilities[branch] = _predict_probabilities(
            model,
            val_views[branch],
            device=device,
        )
        full_protocol[branch] = telemetry
        all_training_finite = all_training_finite and bool(telemetry["finite"])
        all_losses_decreased = all_losses_decreased and bool(telemetry["loss_decreased"])
        minimum_batch_class_support = min(
            minimum_batch_class_support,
            int(telemetry["minimum_batch_class_support"]),
        )
        for row in curve:
            curve_rows.append({"scope": "full_train", "branch": branch, **row})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_metrics = {
        branch: _classification_metrics(y_train, oof_probabilities[branch])
        for branch in BRANCHES
    }
    val_metrics = {
        branch: _classification_metrics(y_val, val_probabilities[branch])
        for branch in BRANCHES
    }
    val_metrics["keeper"] = _classification_metrics(y_val, aligned_keeper)
    selected_control = _selected_control(
        oof_metrics,
        focus_class_index=int(args.focus_class_index),
    )
    transitions = {
        "oof_fusion_vs_selected_control": _transition_stats(
            y_train,
            oof_probabilities[selected_control],
            oof_probabilities["fusion"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_fusion_vs_selected_control": _transition_stats(
            y_val,
            val_probabilities[selected_control],
            val_probabilities["fusion"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_fusion_vs_keeper": _transition_stats(
            y_val,
            aligned_keeper,
            val_probabilities["fusion"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    direction = {
        "oof_fusion_vs_selected_control": _direction_auc(
            y_train,
            oof_probabilities[selected_control],
            oof_probabilities["fusion"],
            focus_class_index=int(args.focus_class_index),
        ),
        "val_fusion_vs_selected_control": _direction_auc(
            y_val,
            val_probabilities[selected_control],
            val_probabilities["fusion"],
            focus_class_index=int(args.focus_class_index),
        ),
    }
    gate = assess_raw_aidt_fusion_readiness(
        train_rows=len(y_train),
        val_rows=len(y_val),
        train_source_groups=int(np.unique(train_groups).size),
        train_val_source_overlap=int(train_val_source_overlap),
        maximum_fold_source_overlap=int(maximum_fold_source_overlap),
        all_training_finite=bool(all_training_finite),
        all_losses_decreased=bool(all_losses_decreased),
        minimum_batch_class_support=int(minimum_batch_class_support),
        folds_fusion_wins_focus=int(folds_fusion_wins_focus),
        fold_count=int(args.folds),
        oof_control_metrics=oof_metrics[selected_control],
        oof_fusion_metrics=oof_metrics["fusion"],
        val_control_metrics=val_metrics[selected_control],
        val_fusion_metrics=val_metrics["fusion"],
        val_keeper_metrics=val_metrics["keeper"],
        oof_transitions=transitions["oof_fusion_vs_selected_control"],
        val_control_transitions=transitions["val_fusion_vs_selected_control"],
        val_keeper_transitions=transitions["val_fusion_vs_keeper"],
        oof_direction=direction["oof_fusion_vs_selected_control"],
        val_direction=direction["val_fusion_vs_selected_control"],
        focus_class_index=int(args.focus_class_index),
        test_split_used=False,
    )

    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "training_curves.csv", curve_rows)
    _write_csv(
        output_dir / "train_oof_predictions.csv",
        _prediction_rows(
            split="train_oof",
            cache=train,
            fold_assignment=fold_assignment,
            probabilities=oof_probabilities,
            selected_control=selected_control,
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(len(y_val), -1, dtype=np.int64),
            probabilities=val_probabilities,
            selected_control=selected_control,
            keeper_probabilities=aligned_keeper,
        ),
    )
    np.savez_compressed(
        output_dir / "diagnostic_probabilities.npz",
        train_oof_fusion=oof_probabilities["fusion"],
        train_oof_selected_control=oof_probabilities[selected_control],
        val_fusion=val_probabilities["fusion"],
        val_selected_control=val_probabilities[selected_control],
        val_keeper=aligned_keeper,
        train_labels=y_train,
        val_labels=y_val,
        train_sample_index=train["sample_index"],
        val_sample_index=val["sample_index"],
        classes=train["classes"],
        selected_control=np.asarray([selected_control], dtype=object),
        diagnostic_only=np.asarray([True], dtype=np.bool_),
    )
    _plot_summary(
        output_dir / "raw_aidt_fusion_readiness.png",
        curves=curve_rows,
        oof_metrics=oof_metrics,
        val_metrics=val_metrics,
        focus_class_index=int(args.focus_class_index),
    )
    protocol = {
        "method": "raw_pretrained_aidt_resnet50_vitb16_fusion_readiness",
        "literature": list(LITERATURE),
        "train_npz": str(Path(args.train_npz).resolve()),
        "val_npz": str(Path(args.val_npz).resolve()),
        "keeper_val_cache": str(Path(args.keeper_val_cache).resolve()),
        "input_sha256": {
            "train_npz": _sha256(Path(args.train_npz)),
            "val_npz": _sha256(Path(args.val_npz)),
            "keeper_val_cache": _sha256(Path(args.keeper_val_cache)),
        },
        "split_usage": {"train": True, "val": True, "test": False},
        "raw_pretrained_backbones": True,
        "backbones_frozen": True,
        "feature_normalization": str(args.feature_normalization),
        "component_dimensions": {
            "resnet": int(args.resnet_dim),
            "vit": int(args.vit_dim),
            "fusion": int(args.resnet_dim) + int(args.vit_dim),
        },
        "readout": (
            f"Linear(D,{int(args.hidden_dim_1)})-BN-ReLU-Dropout0.3-"
            f"Linear({int(args.hidden_dim_1)},{int(args.hidden_dim_2)})-"
            f"BN-ReLU-Dropout0.2-Linear({int(args.hidden_dim_2)},5)"
        ),
        "epochs": int(args.epochs),
        "project_epoch_cap": 30,
        "batch_size": int(args.batch_size),
        "drop_last": True,
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "scheduler": "linear warmup then cosine, matching AIDT family",
        "label_smoothing": float(args.label_smoothing),
        "class_balance": "sqrt_inverse normalized to mean 1 per fit",
        "source_grouped_folds": int(args.folds),
        "selected_control": selected_control,
        "control_selection": "train OOF 0.5*macro_f1 + 0.5*class1_f1; no validation selection",
        "device": str(device),
        "validation_hyperparameter_tuning": False,
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "trainable_manifest_written": False,
        "test_split_used": False,
    }
    summary = {
        "protocol": protocol,
        "fold_protocols": fold_protocols,
        "full_training": full_protocol,
        "metrics": {"train_oof": oof_metrics, "val": val_metrics},
        "transitions": transitions,
        "focus_direction": direction,
        "gate": gate,
        "seconds": float(time.time() - start),
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "test_split_used": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# Raw-Pretrained AIDT Fusion Readiness",
        "",
        f"- OOF selected control: `{selected_control}`",
        f"- OOF control/fusion macro-class1: `{float(oof_metrics[selected_control]['macro_f1']):.6f}/{float(_focus(oof_metrics[selected_control], int(args.focus_class_index))['f1']):.6f} -> {float(oof_metrics['fusion']['macro_f1']):.6f}/{float(_focus(oof_metrics['fusion'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val control/fusion macro-class1: `{float(val_metrics[selected_control]['macro_f1']):.6f}/{float(_focus(val_metrics[selected_control], int(args.focus_class_index))['f1']):.6f} -> {float(val_metrics['fusion']['macro_f1']):.6f}/{float(_focus(val_metrics['fusion'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val keeper macro-class1: `{float(val_metrics['keeper']['macro_f1']):.6f}/{float(_focus(val_metrics['keeper'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Fold-safe teacher permission: `{str(bool(gate['fold_safe_teacher_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "All heads are trained only in memory and discarded. The probability NPZ is diagnostic-only until every promotion gate passes. No test input, raw-data edit, model, checkpoint, or trainable manifest is written.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(
        output_dir,
        mode="raw_pretrained_aidt_fusion_readiness_manifest",
    )
    summary["artifact_manifest"] = {key: value for key, value in manifest.items() if key != "files"}
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_audit(parse_args(argv))
    print(
        json.dumps(
            {
                "metrics": summary["metrics"],
                "transitions": summary["transitions"],
                "focus_direction": summary["focus_direction"],
                "gate": summary["gate"],
                "artifact_manifest": summary["artifact_manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
