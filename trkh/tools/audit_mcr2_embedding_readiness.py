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
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.nn import functional as F
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
    _write_artifact_manifest,
)


SEED = 20260712
FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 1000
FEATURE_DIM = 128
PCA_COMPONENTS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPSILON = 0.5
GAMMA1 = 1.0
GAMMA2 = 1.0
FOCUS_CLASS_INDEX = 1
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_VAL_ROWS = 2606
LITERATURE = (
    "https://proceedings.neurips.cc/paper_files/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html",
    "https://github.com/ryanchankh/mcr2",
    "https://openaccess.thecvf.com/content/CVPR2022/html/Baek_Efficient_Maximal_Coding_Rate_Reduction_by_Variational_Forms_CVPR_2022_paper.html",
)


class OfficialMCR2Projection(nn.Module):
    """Projection layout used by the official supervised MCR2 ResNet head."""

    def __init__(self, input_dim: int, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(int(input_dim), int(input_dim), bias=False),
            nn.BatchNorm1d(int(input_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim), int(feature_dim), bias=True),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.projection(features), dim=1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation-only MCR2 readiness audit on frozen keeper embeddings. "
            "It writes no adapter, model, checkpoint, trainable manifest, or test result."
        )
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--gamma1", type=float, default=GAMMA1)
    parser.add_argument("--gamma2", type=float, default=GAMMA2)
    parser.add_argument("--focus-class-index", type=int, default=FOCUS_CLASS_INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def maximal_coding_rate_reduction(
    features: Tensor,
    labels: Tensor,
    *,
    class_count: int,
    epsilon: float = EPSILON,
    gamma1: float = GAMMA1,
    gamma2: float = GAMMA2,
) -> Tuple[Tensor, Tensor, Tensor]:
    if features.ndim != 2 or labels.ndim != 1 or int(features.shape[0]) != int(labels.shape[0]):
        raise ValueError("features/labels must have shapes [N,D]/[N]")
    if int(features.shape[0]) < 2 or int(features.shape[1]) < 2:
        raise ValueError("MCR2 requires at least two rows and dimensions")
    if int(class_count) < 2 or float(epsilon) <= 0.0:
        raise ValueError("class_count/epsilon are invalid")
    if float(gamma1) <= 0.0 or float(gamma2) <= 0.0:
        raise ValueError("gamma values must be positive")
    if torch.any(labels < 0) or torch.any(labels >= int(class_count)):
        raise ValueError("labels are outside class_count")

    sample_count, dimension = int(features.shape[0]), int(features.shape[1])
    identity = torch.eye(dimension, device=features.device, dtype=features.dtype)
    global_matrix = identity + (
        float(gamma1) * float(dimension) / (float(sample_count) * float(epsilon))
    ) * features.transpose(0, 1).matmul(features)
    global_sign, global_logdet = torch.linalg.slogdet(global_matrix)
    if bool((global_sign <= 0).item()):
        raise RuntimeError("global coding-rate matrix is not positive definite")
    global_rate = 0.5 * global_logdet

    compressive_rate = features.new_zeros(())
    for class_index in range(int(class_count)):
        class_features = features[labels == class_index]
        class_samples = int(class_features.shape[0])
        if class_samples == 0:
            continue
        class_matrix = identity + (
            float(dimension) / (float(class_samples) * float(epsilon))
        ) * class_features.transpose(0, 1).matmul(class_features)
        class_sign, class_logdet = torch.linalg.slogdet(class_matrix)
        if bool((class_sign <= 0).item()):
            raise RuntimeError("class coding-rate matrix is not positive definite")
        compressive_rate = compressive_rate + (
            float(class_samples) / float(sample_count)
        ) * 0.5 * class_logdet
    loss = -float(gamma2) * global_rate + compressive_rate
    return loss, global_rate, compressive_rate


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


def _extract_features(
    model: nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            batch = torch.from_numpy(
                np.asarray(values[start : start + int(batch_size)], dtype=np.float32)
            ).to(device=device)
            rows.append(model(batch).cpu().numpy())
    result = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    if result.shape[0] != len(values) or not np.isfinite(result).all():
        raise RuntimeError("MCR2 projection produced invalid features")
    return result


def _coding_rate_summary(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    epsilon: float,
    gamma1: float,
    gamma2: float,
    device: torch.device,
) -> Dict[str, float]:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device=device)
    targets = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device=device)
    with torch.no_grad():
        loss, global_rate, compressive_rate = maximal_coding_rate_reduction(
            values,
            targets,
            class_count=int(class_count),
            epsilon=float(epsilon),
            gamma1=float(gamma1),
            gamma2=float(gamma2),
        )
    return {
        "loss": float(loss.cpu().item()),
        "global_rate": float(global_rate.cpu().item()),
        "compressive_rate": float(compressive_rate.cpu().item()),
        "rate_reduction": float((global_rate - compressive_rate).cpu().item()),
    }


def train_mcr2_projection(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    feature_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    epsilon: float,
    gamma1: float,
    gamma2: float,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, object]]:
    _seed_everything(seed)
    input_values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    model = OfficialMCR2Projection(input_values.shape[1], feature_dim=int(feature_dim)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(input_values), torch.from_numpy(targets)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("batch-size leaves no complete MCR2 batch")

    curve: List[Dict[str, float]] = []
    minimum_batch_class_support = int(batch_size)
    finite = True
    start = time.time()
    for epoch_index in range(int(epochs)):
        model.train()
        epoch_rows = []
        for batch_features, batch_labels in loader:
            class_support = torch.bincount(batch_labels, minlength=int(class_count))
            minimum_batch_class_support = min(
                minimum_batch_class_support,
                int(class_support.min().item()),
            )
            batch_features = batch_features.to(device=device, dtype=torch.float32)
            batch_labels = batch_labels.to(device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            projected = model(batch_features)
            loss, global_rate, compressive_rate = maximal_coding_rate_reduction(
                projected,
                batch_labels,
                class_count=int(class_count),
                epsilon=float(epsilon),
                gamma1=float(gamma1),
                gamma2=float(gamma2),
            )
            if not bool(torch.isfinite(loss).item()):
                finite = False
                raise FloatingPointError("MCR2 loss became non-finite")
            loss.backward()
            optimizer.step()
            epoch_rows.append(
                (
                    float(loss.detach().cpu().item()),
                    float(global_rate.detach().cpu().item()),
                    float(compressive_rate.detach().cpu().item()),
                )
            )
        means = np.asarray(epoch_rows, dtype=np.float64).mean(axis=0)
        curve.append(
            {
                "epoch": int(epoch_index + 1),
                "loss": float(means[0]),
                "global_rate": float(means[1]),
                "compressive_rate": float(means[2]),
                "rate_reduction": float(means[1] - means[2]),
            }
        )
    telemetry = {
        "finite": bool(finite),
        "epochs": int(len(curve)),
        "batches_per_epoch": int(len(loader)),
        "samples_per_epoch": int(len(loader) * int(batch_size)),
        "minimum_batch_class_support": int(minimum_batch_class_support),
        "seconds": float(time.time() - start),
        "first": dict(curve[0]),
        "last": dict(curve[-1]),
        "loss_decreased": bool(float(curve[-1]["loss"]) < float(curve[0]["loss"])),
    }
    return model, curve, telemetry


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.maximum(norms, 1e-12)).astype(np.float32)


def nearest_subspace_probabilities(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    class_count: int,
    pca_components: int = PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    fit = np.asarray(fit_features, dtype=np.float64)
    evaluate = np.asarray(eval_features, dtype=np.float64)
    labels = np.asarray(fit_labels, dtype=np.int64)
    if fit.ndim != 2 or evaluate.ndim != 2 or fit.shape[1] != evaluate.shape[1]:
        raise ValueError("fit/eval features must be aligned 2D matrices")
    if labels.shape != (len(fit),):
        raise ValueError("fit labels do not align with features")

    models = []
    for class_index in range(int(class_count)):
        class_features = fit[labels == class_index]
        components = min(int(pca_components), int(class_features.shape[0]) - 1, int(fit.shape[1]))
        if components < 1:
            raise ValueError(f"class {class_index} has insufficient subspace support")
        pca = PCA(n_components=components, svd_solver="full").fit(class_features)
        models.append((pca.mean_, pca.components_))

    def residual_matrix(values: np.ndarray) -> np.ndarray:
        columns = []
        for mean, components in models:
            centered = values - mean
            projected = centered @ components.T @ components
            columns.append(np.linalg.norm(centered - projected, axis=1))
        return np.stack(columns, axis=1)

    fit_residuals = residual_matrix(fit)
    eval_residuals = residual_matrix(evaluate)
    scale = max(float(np.median(fit_residuals)), 1e-6)
    logits = -eval_residuals / scale
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    protocol = {
        "pca_components": int(pca_components),
        "effective_components": [int(value[1].shape[0]) for value in models],
        "fit_residual_temperature": float(scale),
    }
    return probabilities.astype(np.float32), eval_residuals.astype(np.float32), protocol


def assess_mcr2_readiness(
    *,
    train_rows: int,
    val_rows: int,
    train_source_groups: int,
    train_val_source_overlap: int,
    maximum_fold_source_overlap: int,
    all_training_finite: bool,
    all_losses_decreased: bool,
    minimum_batch_class_support: int,
    folds_with_focus_gain: int,
    fold_count: int,
    oof_control_metrics: Mapping[str, object],
    oof_candidate_metrics: Mapping[str, object],
    val_control_metrics: Mapping[str, object],
    val_candidate_metrics: Mapping[str, object],
    val_keeper_metrics: Mapping[str, object],
    oof_transitions: Mapping[str, int],
    val_transitions: Mapping[str, int],
    keeper_transitions: Mapping[str, int],
    oof_direction: Mapping[str, object],
    val_direction: Mapping[str, object],
    focus_class_index: int,
    test_split_used: bool,
) -> Dict[str, object]:
    focus = int(focus_class_index)
    candidate_focus = _focus(val_candidate_metrics, focus)
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
        "folds_with_focus_gain": int(folds_with_focus_gain),
        "fold_count": int(fold_count),
        "oof_macro_gain": float(oof_candidate_metrics["macro_f1"] - oof_control_metrics["macro_f1"]),
        "oof_focus_gain": float(_focus(oof_candidate_metrics, focus)["f1"] - _focus(oof_control_metrics, focus)["f1"]),
        "val_macro_gain": float(val_candidate_metrics["macro_f1"] - val_control_metrics["macro_f1"]),
        "val_focus_gain": float(candidate_focus["f1"] - _focus(val_control_metrics, focus)["f1"]),
        "keeper_macro_f1": float(val_keeper_metrics["macro_f1"]),
        "keeper_focus_f1": float(keeper_focus["f1"]),
        "keeper_focus_recall": float(keeper_focus["recall"]),
        "candidate_macro_f1": float(val_candidate_metrics["macro_f1"]),
        "candidate_focus_f1": float(candidate_focus["f1"]),
        "candidate_focus_precision": float(candidate_focus["precision"]),
        "candidate_focus_recall": float(candidate_focus["recall"]),
        "candidate_keeper_macro_gain": float(val_candidate_metrics["macro_f1"] - val_keeper_metrics["macro_f1"]),
        "candidate_keeper_focus_gain": float(candidate_focus["f1"] - keeper_focus["f1"]),
        "oof_transitions": dict(oof_transitions),
        "val_transitions": dict(val_transitions),
        "keeper_transitions": dict(keeper_transitions),
        "directions_available": directions_available,
        "oof_direction_auc": float(oof_auc) if directions_available else -1.0,
        "val_direction_auc": float(val_auc) if directions_available else -1.0,
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
        "focus_gain_in_at_least_3_folds": observed["folds_with_focus_gain"] >= min(3, observed["fold_count"]),
        "oof_macro_gain_ge_0p002": observed["oof_macro_gain"] >= 0.002,
        "oof_focus_gain_ge_0p01": observed["oof_focus_gain"] >= 0.01,
        "val_macro_gain_ge_0p002": observed["val_macro_gain"] >= 0.002,
        "val_focus_gain_ge_0p015": observed["val_focus_gain"] >= 0.015,
        "oof_corrections_ge_harms": int(oof_transitions["corrections"]) >= int(oof_transitions["harms"]),
        "oof_fn_rescued_ge_tp_broken": int(oof_transitions["focus_false_negative_rescued"]) >= int(oof_transitions["focus_true_positive_broken"]),
        "val_corrections_ge_harms": int(val_transitions["corrections"]) >= int(val_transitions["harms"]),
        "val_fn_rescued_ge_tp_broken": int(val_transitions["focus_false_negative_rescued"]) >= int(val_transitions["focus_true_positive_broken"]),
        "directions_available": directions_available,
        "oof_direction_auc_ge_0p60": observed["oof_direction_auc"] >= 0.60,
        "val_direction_auc_ge_0p60": observed["val_direction_auc"] >= 0.60,
        "keeper_macro_preserved_within_0p001": observed["candidate_keeper_macro_gain"] >= -0.001,
        "keeper_focus_improved_by_0p01": observed["candidate_keeper_focus_gain"] >= 0.01,
        "candidate_focus_reaches_0p70": observed["candidate_focus_f1"] >= 0.70,
        "keeper_focus_recall_preserved_within_0p01": observed["candidate_focus_recall"] >= observed["keeper_focus_recall"] - 0.01,
        "keeper_corrections_ge_harms": int(keeper_transitions["corrections"]) >= int(keeper_transitions["harms"]),
        "keeper_fp_removed_ge_created": int(keeper_transitions["focus_false_positive_removed"]) >= int(keeper_transitions["focus_false_positive_created"]),
        "keeper_fn_rescued_ge_tp_broken": int(keeper_transitions["focus_false_negative_rescued"]) >= int(keeper_transitions["focus_true_positive_broken"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "image_smoke_permission": not failed,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
    }


def _prediction_rows(
    *,
    split: str,
    cache: Mapping[str, np.ndarray],
    fold_assignment: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
) -> List[Dict[str, object]]:
    rows = []
    source_stems = np.asarray(cache["source_stems"], dtype=object)
    for index in range(len(cache["labels"])):
        row: Dict[str, object] = {
            "split": str(split),
            "sample_index": int(cache["sample_index"][index]),
            "fold": int(fold_assignment[index]),
            "source_stem": str(source_stems[index]),
            "image_path": str(cache["paths"][index]),
            "target_index": int(cache["labels"][index]),
            "prediction_index": int(candidate[index].argmax()),
            "y_true": int(cache["labels"][index]),
            "y_pred": int(candidate[index].argmax()),
            "keeper_prediction_index": int(cache["probabilities"][index].argmax()),
            "control_prediction_index": int(control[index].argmax()),
            "candidate_prediction_index": int(candidate[index].argmax()),
        }
        for name, probabilities in (("keeper", cache["probabilities"]), ("control", control), ("candidate", candidate)):
            for class_index in range(int(probabilities.shape[1])):
                row[f"{name}_prob_{class_index}"] = float(probabilities[index, class_index])
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
    scopes = sorted({str(row["scope"]) for row in curves})
    for scope in scopes:
        rows = [row for row in curves if str(row["scope"]) == scope]
        axes[0].plot(
            [int(row["epoch"]) for row in rows],
            [float(row["rate_reduction"]) for row in rows],
            alpha=0.7,
            label=scope,
        )
    axes[0].set_title("MCR2 rate reduction")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Global - compressive rate")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7)

    names = ("OOF control", "OOF MCR2", "val control", "val MCR2", "val keeper")
    values = (
        (float(oof_metrics["control"]["macro_f1"]), float(_focus(oof_metrics["control"], focus_class_index)["f1"])),
        (float(oof_metrics["candidate"]["macro_f1"]), float(_focus(oof_metrics["candidate"], focus_class_index)["f1"])),
        (float(val_metrics["control"]["macro_f1"]), float(_focus(val_metrics["control"], focus_class_index)["f1"])),
        (float(val_metrics["candidate"]["macro_f1"]), float(_focus(val_metrics["candidate"], focus_class_index)["f1"])),
        (float(val_metrics["keeper"]["macro_f1"]), float(_focus(val_metrics["keeper"], focus_class_index)["f1"])),
    )
    positions = np.arange(len(names))
    axes[1].bar(positions - 0.18, [value[0] for value in values], width=0.36, label="macro F1")
    axes[1].bar(positions + 0.18, [value[1] for value in values], width=0.36, label="class-1 F1")
    axes[1].set_xticks(positions, names, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Source-safe readiness metrics")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) < 3:
        raise ValueError("folds must be at least three")
    if not 1 <= int(args.epochs) <= 30:
        raise ValueError("epochs must stay within the project cap [1,30]")
    if int(args.batch_size) < 128 or int(args.feature_dim) < 16:
        raise ValueError("batch-size/feature-dim are too small for MCR2")
    if int(args.pca_components) < 1 or int(args.pca_components) >= int(args.feature_dim):
        raise ValueError("pca-components must be in [1, feature-dim)")
    if (
        float(args.learning_rate) <= 0.0
        or not 0.0 <= float(args.momentum) < 1.0
        or float(args.weight_decay) < 0.0
        or float(args.epsilon) <= 0.0
        or float(args.gamma1) <= 0.0
        or float(args.gamma2) <= 0.0
    ):
        raise ValueError("optimizer/MCR2 settings are invalid")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = _resolve_device(str(args.device))
    start = time.time()

    train = _load_cache(Path(args.train_cache), split="train")
    val = _load_cache(Path(args.val_cache), split="val")
    x_train = np.asarray(train["embeddings"], dtype=np.float32)
    x_val = np.asarray(val["embeddings"], dtype=np.float32)
    y_train = np.asarray(train["labels"], dtype=np.int64)
    y_val = np.asarray(val["labels"], dtype=np.int64)
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("train/validation embedding dimensions differ")
    class_count = int(train["probabilities"].shape[1])
    if int(val["probabilities"].shape[1]) != class_count:
        raise ValueError("train/validation class counts differ")
    if np.unique(y_train).size != class_count or np.unique(y_val).size != class_count:
        raise ValueError("train/validation caches do not cover every class")
    train_groups = np.asarray(train["source_stems"], dtype=object)
    val_groups = np.asarray(val["source_stems"], dtype=object)
    train_val_source_overlap = len(set(map(str, train_groups)).intersection(map(str, val_groups)))

    raw_train = _l2_normalize(x_train)
    raw_val = _l2_normalize(x_val)
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    oof_control = np.zeros_like(train["probabilities"], dtype=np.float32)
    oof_candidate = np.zeros_like(train["probabilities"], dtype=np.float32)
    fold_assignment = np.full(len(y_train), -1, dtype=np.int64)
    fold_rows = []
    curve_rows: List[Dict[str, object]] = []
    fold_protocols = []
    maximum_fold_source_overlap = 0
    all_training_finite = True
    all_losses_decreased = True
    minimum_batch_class_support = int(args.batch_size)
    folds_with_focus_gain = 0

    for fold_index, (fit_indices, hold_indices) in enumerate(
        splitter.split(x_train, y_train, train_groups)
    ):
        fit_sources = set(map(str, train_groups[fit_indices]))
        hold_sources = set(map(str, train_groups[hold_indices]))
        overlap = len(fit_sources.intersection(hold_sources))
        maximum_fold_source_overlap = max(maximum_fold_source_overlap, overlap)
        model, curve, telemetry = train_mcr2_projection(
            x_train[fit_indices],
            y_train[fit_indices],
            class_count=class_count,
            feature_dim=int(args.feature_dim),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            epsilon=float(args.epsilon),
            gamma1=float(args.gamma1),
            gamma2=float(args.gamma2),
            seed=int(args.seed) + int(fold_index),
            device=device,
        )
        candidate_fit = _extract_features(model, x_train[fit_indices], device=device)
        candidate_hold = _extract_features(model, x_train[hold_indices], device=device)
        control_probabilities, _control_residuals, control_protocol = nearest_subspace_probabilities(
            raw_train[fit_indices],
            y_train[fit_indices],
            raw_train[hold_indices],
            class_count=class_count,
            pca_components=int(args.pca_components),
        )
        candidate_probabilities, _candidate_residuals, candidate_protocol = nearest_subspace_probabilities(
            candidate_fit,
            y_train[fit_indices],
            candidate_hold,
            class_count=class_count,
            pca_components=int(args.pca_components),
        )
        oof_control[hold_indices] = control_probabilities
        oof_candidate[hold_indices] = candidate_probabilities
        fold_assignment[hold_indices] = int(fold_index)
        control_metrics = _classification_metrics(y_train[hold_indices], control_probabilities)
        candidate_metrics = _classification_metrics(y_train[hold_indices], candidate_probabilities)
        focus_gain = float(
            _focus(candidate_metrics, int(args.focus_class_index))["f1"]
            - _focus(control_metrics, int(args.focus_class_index))["f1"]
        )
        folds_with_focus_gain += int(focus_gain > 0.0)
        fold_rows.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "source_overlap": int(overlap),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "control_focus_f1": float(_focus(control_metrics, int(args.focus_class_index))["f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "candidate_focus_f1": float(_focus(candidate_metrics, int(args.focus_class_index))["f1"]),
                "candidate_focus_gain": float(focus_gain),
                "first_rate_reduction": float(curve[0]["rate_reduction"]),
                "last_rate_reduction": float(curve[-1]["rate_reduction"]),
            }
        )
        for row in curve:
            curve_rows.append({"scope": f"fold_{fold_index}", **row})
        fold_protocols.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "hold_rows": int(len(hold_indices)),
                "fit_sources": int(len(fit_sources)),
                "hold_sources": int(len(hold_sources)),
                "source_overlap": int(overlap),
                "training": telemetry,
                "control_subspace": control_protocol,
                "candidate_subspace": candidate_protocol,
            }
        )
        all_training_finite = all_training_finite and bool(telemetry["finite"])
        all_losses_decreased = all_losses_decreased and bool(telemetry["loss_decreased"])
        minimum_batch_class_support = min(
            minimum_batch_class_support,
            int(telemetry["minimum_batch_class_support"]),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if np.any(fold_assignment < 0):
        raise RuntimeError("OOF assignment is incomplete")

    full_model, full_curve, full_telemetry = train_mcr2_projection(
        x_train,
        y_train,
        class_count=class_count,
        feature_dim=int(args.feature_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        epsilon=float(args.epsilon),
        gamma1=float(args.gamma1),
        gamma2=float(args.gamma2),
        seed=int(args.seed) + int(args.folds),
        device=device,
    )
    candidate_train = _extract_features(full_model, x_train, device=device)
    candidate_val = _extract_features(full_model, x_val, device=device)
    val_control, _val_control_residuals, full_control_protocol = nearest_subspace_probabilities(
        raw_train,
        y_train,
        raw_val,
        class_count=class_count,
        pca_components=int(args.pca_components),
    )
    val_candidate, _val_candidate_residuals, full_candidate_protocol = nearest_subspace_probabilities(
        candidate_train,
        y_train,
        candidate_val,
        class_count=class_count,
        pca_components=int(args.pca_components),
    )
    full_rate = _coding_rate_summary(
        candidate_train,
        y_train,
        class_count=class_count,
        epsilon=float(args.epsilon),
        gamma1=float(args.gamma1),
        gamma2=float(args.gamma2),
        device=device,
    )
    for row in full_curve:
        curve_rows.append({"scope": "full_train", **row})
    all_training_finite = all_training_finite and bool(full_telemetry["finite"])
    all_losses_decreased = all_losses_decreased and bool(full_telemetry["loss_decreased"])
    minimum_batch_class_support = min(
        minimum_batch_class_support,
        int(full_telemetry["minimum_batch_class_support"]),
    )

    oof_metrics = {
        "control": _classification_metrics(y_train, oof_control),
        "candidate": _classification_metrics(y_train, oof_candidate),
    }
    val_metrics = {
        "keeper": _classification_metrics(y_val, val["probabilities"]),
        "control": _classification_metrics(y_val, val_control),
        "candidate": _classification_metrics(y_val, val_candidate),
    }
    transitions = {
        "oof_candidate_vs_control": _transition_stats(
            y_train,
            oof_control,
            oof_candidate,
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _transition_stats(
            y_val,
            val_control,
            val_candidate,
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_keeper": _transition_stats(
            y_val,
            val["probabilities"],
            val_candidate,
            focus_class_index=int(args.focus_class_index),
        ),
    }
    direction = {
        "oof_candidate_vs_control": _direction_auc(
            y_train,
            oof_control,
            oof_candidate,
            focus_class_index=int(args.focus_class_index),
        ),
        "val_candidate_vs_control": _direction_auc(
            y_val,
            val_control,
            val_candidate,
            focus_class_index=int(args.focus_class_index),
        ),
    }
    gate = assess_mcr2_readiness(
        train_rows=len(y_train),
        val_rows=len(y_val),
        train_source_groups=int(np.unique(train_groups).size),
        train_val_source_overlap=train_val_source_overlap,
        maximum_fold_source_overlap=maximum_fold_source_overlap,
        all_training_finite=all_training_finite,
        all_losses_decreased=all_losses_decreased,
        minimum_batch_class_support=minimum_batch_class_support,
        folds_with_focus_gain=folds_with_focus_gain,
        fold_count=int(args.folds),
        oof_control_metrics=oof_metrics["control"],
        oof_candidate_metrics=oof_metrics["candidate"],
        val_control_metrics=val_metrics["control"],
        val_candidate_metrics=val_metrics["candidate"],
        val_keeper_metrics=val_metrics["keeper"],
        oof_transitions=transitions["oof_candidate_vs_control"],
        val_transitions=transitions["val_candidate_vs_control"],
        keeper_transitions=transitions["val_candidate_vs_keeper"],
        oof_direction=direction["oof_candidate_vs_control"],
        val_direction=direction["val_candidate_vs_control"],
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
            control=oof_control,
            candidate=oof_candidate,
        ),
    )
    _write_csv(
        output_dir / "val_predictions.csv",
        _prediction_rows(
            split="val",
            cache=val,
            fold_assignment=np.full(len(y_val), -1, dtype=np.int64),
            control=val_control,
            candidate=val_candidate,
        ),
    )
    _plot_summary(
        output_dir / "mcr2_training_and_metrics.png",
        curves=curve_rows,
        oof_metrics=oof_metrics,
        val_metrics=val_metrics,
        focus_class_index=int(args.focus_class_index),
    )
    protocol = {
        "method": "official_mcr2_frozen_keeper_embedding_readiness",
        "literature": list(LITERATURE),
        "train_cache": str(Path(args.train_cache).resolve()),
        "val_cache": str(Path(args.val_cache).resolve()),
        "split_usage": {"train": True, "val": True, "test": False},
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_source_groups": int(np.unique(train_groups).size),
        "val_source_groups": int(np.unique(val_groups).size),
        "train_val_source_overlap": int(train_val_source_overlap),
        "input_dim": int(x_train.shape[1]),
        "projection": "Linear(D,D,bias=False)-BatchNorm-ReLU-Linear(D,128)-L2Norm",
        "feature_dim": int(args.feature_dim),
        "epochs": int(args.epochs),
        "project_epoch_cap": 30,
        "official_source_training_epochs": 800,
        "official_readme_reproduction_epochs": 500,
        "batch_size": int(args.batch_size),
        "drop_last": True,
        "sampling": "natural-frequency deterministic shuffle; no class oversampling or weighting",
        "optimizer": "SGD",
        "learning_rate": float(args.learning_rate),
        "learning_rate_source": (
            "official README supervised-reproduction command (0.01); official "
            "train_sup.py parser default is 0.001"
        ),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "epsilon": float(args.epsilon),
        "gamma1": float(args.gamma1),
        "gamma2": float(args.gamma2),
        "evaluation": "official nearest-subspace PCA residual",
        "pca_components": int(args.pca_components),
        "source_grouped_folds": int(args.folds),
        "oof_scope_note": (
            "The projection/readout is source-grouped OOF, but the frozen keeper "
            "embedding cache was produced by a keeper trained on the full train split. "
            "Keeper train probabilities are in-sample references, never OOF evidence."
        ),
        "device": str(device),
        "validation_hyperparameter_tuning": False,
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "test_split_used": False,
    }
    summary = {
        "protocol": protocol,
        "fold_protocols": fold_protocols,
        "full_training": full_telemetry,
        "full_subspace_protocol": {
            "control": full_control_protocol,
            "candidate": full_candidate_protocol,
        },
        "full_train_coding_rate": full_rate,
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
        "# MCR2 Frozen-Embedding Readiness",
        "",
        f"- OOF control/MCR2 macro-class1: `{float(oof_metrics['control']['macro_f1']):.6f}/{float(_focus(oof_metrics['control'], int(args.focus_class_index))['f1']):.6f} -> {float(oof_metrics['candidate']['macro_f1']):.6f}/{float(_focus(oof_metrics['candidate'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val control/MCR2 macro-class1: `{float(val_metrics['control']['macro_f1']):.6f}/{float(_focus(val_metrics['control'], int(args.focus_class_index))['f1']):.6f} -> {float(val_metrics['candidate']['macro_f1']):.6f}/{float(_focus(val_metrics['candidate'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Val direct keeper macro-class1: `{float(val_metrics['keeper']['macro_f1']):.6f}/{float(_focus(val_metrics['keeper'], int(args.focus_class_index))['f1']):.6f}`",
        f"- Image smoke permission: `{str(bool(gate['image_smoke_permission'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "The adapter is trained only in memory and discarded. No test input, raw-data edit, model, checkpoint, or trainable manifest is written.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(
        output_dir,
        mode="mcr2_embedding_readiness_evidence_manifest",
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
