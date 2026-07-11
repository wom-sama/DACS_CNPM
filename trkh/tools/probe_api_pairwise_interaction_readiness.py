from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _l2_normalize,
    _transition_summary,
    _write_prediction_audit,
)


SEED = 20260711
LITERATURE = (
    "https://ojs.aaai.org/index.php/AAAI/article/view/7016",
    "https://arxiv.org/abs/2002.10191",
    "https://openaccess.thecvf.com/content/CVPR2022/html/"
    "Zhu_Dual_Cross-Attention_Learning_for_Fine-Grained_Visual_"
    "Categorization_and_Object_Re-Identification_CVPR_2022_paper.html",
    "https://openaccess.thecvf.com/content_ECCV_2018/html/"
    "Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.html",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-grouped train/validation readiness audit for an API-Net-style "
            "training-only attentive pairwise interaction head. Test is forbidden."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument(
        "--loss-scope",
        choices=("anchor_only", "symmetric"),
        default="anchor_only",
        help=(
            "anchor_only uses the paired image only as context and preserves natural "
            "anchor frequency; symmetric reproduces the original API loss on both endpoints."
        ),
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if "test" in path.name.casefold():
        raise ValueError(f"Test cache is forbidden: {path}")
    required = {
        "head",
        "probabilities",
        "labels",
        "sample_index",
        "paths",
        "source_stem",
        "classes",
    }
    with np.load(path, allow_pickle=True) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Cache is missing keys {missing}: {path}")
        result = {
            "features": np.asarray(payload["head"], dtype=np.float32),
            "probabilities": np.asarray(payload["probabilities"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.int64).reshape(-1),
            "sample_index": np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1),
            "paths": np.asarray(payload["paths"], dtype=object).reshape(-1),
            "groups": np.asarray(payload["source_stem"], dtype=object).reshape(-1),
            "classes": np.asarray(payload["classes"], dtype=object).reshape(-1),
        }

    row_count = int(result["labels"].shape[0])
    class_count = int(result["classes"].shape[0])
    if result["features"].ndim != 2:
        raise ValueError(f"Head features must be two-dimensional: {path}")
    if tuple(result["probabilities"].shape) != (row_count, class_count):
        raise ValueError(f"Probability shape is inconsistent: {path}")
    for key in ("features", "sample_index", "paths", "groups"):
        if int(result[key].shape[0]) != row_count:
            raise ValueError(f"Cache key {key} has inconsistent row count: {path}")
    if not np.isfinite(result["features"]).all():
        raise ValueError(f"Head features contain non-finite values: {path}")
    if not np.isfinite(result["probabilities"]).all():
        raise ValueError(f"Probabilities contain non-finite values: {path}")
    if np.unique(result["sample_index"]).size != row_count:
        raise ValueError(f"sample_index is not unique: {path}")
    if np.any(result["labels"] < 0) or np.any(result["labels"] >= class_count):
        raise ValueError(f"Labels are outside the class range: {path}")
    if any(not str(value).strip() for value in result["groups"]):
        raise ValueError(f"source_stem contains empty values: {path}")
    return result


def _pair_plan_fingerprint(
    epoch_batches: Sequence[Sequence[Mapping[str, np.ndarray]]],
) -> str:
    digest = hashlib.sha256()
    for epoch_index, batches in enumerate(epoch_batches):
        digest.update(np.asarray([epoch_index, len(batches)], dtype=np.int64).tobytes())
        for batch in batches:
            for key in ("first", "second", "pair_type"):
                values = np.asarray(batch[key], dtype=np.int64)
                digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
                digest.update(values.tobytes())
    return digest.hexdigest()


def build_pair_plan(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    class_count: Optional[int] = None,
) -> Dict[str, object]:
    """Build fixed nearest intra/inter pair batches without sample oversampling."""

    values = _l2_normalize(np.asarray(features, dtype=np.float32))
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    source_groups = np.asarray(groups, dtype=object).reshape(-1)
    if values.ndim != 2 or values.shape[0] != targets.size or targets.size != source_groups.size:
        raise ValueError("features, labels, and groups must have aligned rows")
    if int(epochs) <= 0 or int(batch_size) < 4:
        raise ValueError("epochs must be positive and batch_size must be at least four")
    if class_count is None:
        class_count = int(targets.max(initial=-1)) + 1
    class_count = int(class_count)
    rng = np.random.default_rng(int(seed))
    epoch_batches: List[List[Dict[str, np.ndarray]]] = []
    support = np.zeros((2, class_count), dtype=np.int64)
    partner_support = np.zeros((2, class_count), dtype=np.int64)
    anchor_visits = np.zeros(targets.size, dtype=np.int64)
    partner_visits = np.zeros(targets.size, dtype=np.int64)
    same_source_pairs = 0
    pair_count = 0

    for _ in range(int(epochs)):
        permutation = rng.permutation(targets.size)
        anchor_visits[permutation] += 1
        batches: List[Dict[str, np.ndarray]] = []
        for start in range(0, targets.size, int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            if indices.size < 4:
                continue
            batch_features = values[indices]
            batch_labels = targets[indices]
            batch_groups = source_groups[indices]
            distances = np.maximum(
                0.0,
                2.0 - 2.0 * np.matmul(batch_features, batch_features.T),
            )
            different_source = batch_groups[:, None] != batch_groups[None, :]
            not_self = ~np.eye(indices.size, dtype=bool)
            pair_first: List[np.ndarray] = []
            pair_second: List[np.ndarray] = []
            pair_types: List[np.ndarray] = []
            for pair_type, same_class in enumerate((True, False)):
                class_relation = (
                    batch_labels[:, None] == batch_labels[None, :]
                    if same_class
                    else batch_labels[:, None] != batch_labels[None, :]
                )
                eligible = different_source & not_self & class_relation
                valid_anchor = eligible.any(axis=1)
                if not bool(valid_anchor.any()):
                    continue
                masked = np.where(eligible, distances, np.inf)
                local_first = np.flatnonzero(valid_anchor)
                local_second = masked[local_first].argmin(axis=1)
                first = indices[local_first].astype(np.int64, copy=False)
                second = indices[local_second].astype(np.int64, copy=False)
                pair_first.append(first)
                pair_second.append(second)
                pair_types.append(np.full(first.size, pair_type, dtype=np.int64))
                np.add.at(support[pair_type], targets[first], 1)
                np.add.at(partner_support[pair_type], targets[second], 1)
                np.add.at(partner_visits, second, 1)
                same_source_pairs += int(
                    np.sum(source_groups[first] == source_groups[second])
                )
                pair_count += int(first.size)
            if not pair_first:
                continue
            batches.append(
                {
                    "first": np.concatenate(pair_first),
                    "second": np.concatenate(pair_second),
                    "pair_type": np.concatenate(pair_types),
                }
            )
        epoch_batches.append(batches)

    if not np.all(anchor_visits == int(epochs)):
        raise RuntimeError("Pair plan must visit every sample exactly once per epoch")
    payload: Dict[str, object] = {
        "epochs": epoch_batches,
        "stats": {
            "sample_count": int(targets.size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "pair_count": int(pair_count),
            "same_source_pair_count": int(same_source_pairs),
            "anchor_visit_min": int(anchor_visits.min()) if anchor_visits.size else 0,
            "anchor_visit_max": int(anchor_visits.max()) if anchor_visits.size else 0,
            "intra_anchor_support_per_class": support[0].tolist(),
            "inter_anchor_support_per_class": support[1].tolist(),
            "intra_partner_support_per_class": partner_support[0].tolist(),
            "inter_partner_support_per_class": partner_support[1].tolist(),
            "partner_visit_min": int(partner_visits.min()) if partner_visits.size else 0,
            "partner_visit_median": float(np.median(partner_visits))
            if partner_visits.size
            else 0.0,
            "partner_visit_p99": float(np.quantile(partner_visits, 0.99))
            if partner_visits.size
            else 0.0,
            "partner_visit_max": int(partner_visits.max()) if partner_visits.size else 0,
            "all_classes_have_intra_support": bool(np.all(support[0] > 0)),
            "all_classes_have_inter_support": bool(np.all(support[1] > 0)),
            "anchor_oversampling": False,
            "partner_reuse": True,
        },
    }
    payload["stats"]["fingerprint_sha256"] = _pair_plan_fingerprint(epoch_batches)
    return payload


class AttentivePairwiseInteractionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.mutual = nn.Sequential(
            nn.Linear(int(feature_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=False),
            nn.Linear(int(hidden_dim), int(feature_dim)),
        )
        self.classifier = nn.Linear(int(feature_dim), int(class_count))

    def interact(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mutual = self.mutual(torch.cat((first, second), dim=1))
        first_gate = torch.sigmoid(mutual * first)
        second_gate = torch.sigmoid(mutual * second)
        first_self = first + first * first_gate
        second_self = second + second * second_gate
        first_other = first + first * second_gate
        second_other = second + second * first_gate
        return first_self, second_self, first_other, second_other

    def inference_logits(self, features: torch.Tensor) -> torch.Tensor:
        """API-Net inference unloads pair interaction and uses the shared classifier."""

        return self.classifier(features)


def _balanced_class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(class_count))
    if np.any(counts <= 0):
        raise ValueError("Every class must be represented in a training fold")
    weights = labels.size / (float(class_count) * counts.astype(np.float64))
    return weights.astype(np.float32)


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _initial_classifier_state(
    feature_dim: int,
    class_count: int,
    *,
    seed: int,
) -> Dict[str, torch.Tensor]:
    _set_seed(seed)
    classifier = nn.Linear(int(feature_dim), int(class_count))
    return {key: value.detach().clone() for key, value in classifier.state_dict().items()}


def _api_loss(
    model: AttentivePairwiseInteractionHead,
    first: torch.Tensor,
    second: torch.Tensor,
    first_labels: torch.Tensor,
    second_labels: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    ranking_margin: float,
    ranking_weight: float,
    loss_scope: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_scope not in {"anchor_only", "symmetric"}:
        raise ValueError(f"Unsupported API loss scope: {loss_scope}")
    first_self, second_self, first_other, second_other = model.interact(first, second)
    first_self_logits = model.classifier(first_self)
    first_other_logits = model.classifier(first_other)
    first_cross_entropy = 0.5 * (
        F.cross_entropy(first_self_logits, first_labels, weight=class_weights)
        + F.cross_entropy(first_other_logits, first_labels, weight=class_weights)
    )
    first_self_probability = first_self_logits.softmax(dim=1).gather(
        1, first_labels[:, None]
    ).squeeze(1)
    first_other_probability = first_other_logits.softmax(dim=1).gather(
        1, first_labels[:, None]
    ).squeeze(1)
    first_ranking = F.relu(
        first_other_probability - first_self_probability + float(ranking_margin)
    ).mean()
    if loss_scope == "anchor_only":
        total = first_cross_entropy + float(ranking_weight) * first_ranking
        return total, first_cross_entropy.detach(), first_ranking.detach()

    second_self_logits = model.classifier(second_self)
    second_other_logits = model.classifier(second_other)
    second_cross_entropy = 0.5 * (
        F.cross_entropy(second_self_logits, second_labels, weight=class_weights)
        + F.cross_entropy(second_other_logits, second_labels, weight=class_weights)
    )
    second_self_probability = second_self_logits.softmax(dim=1).gather(
        1, second_labels[:, None]
    ).squeeze(1)
    second_other_probability = second_other_logits.softmax(dim=1).gather(
        1, second_labels[:, None]
    ).squeeze(1)
    second_ranking = F.relu(
        second_other_probability - second_self_probability + float(ranking_margin)
    ).mean()
    cross_entropy = 0.5 * (first_cross_entropy + second_cross_entropy)
    ranking = 0.5 * (first_ranking + second_ranking)
    total = cross_entropy + float(ranking_weight) * ranking
    return total, cross_entropy.detach(), ranking.detach()


def _train_pairwise_head(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    pair_plan: Mapping[str, object],
    class_count: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    ranking_margin: float,
    ranking_weight: float,
    device: torch.device,
    seed: int,
    mode: str,
    loss_scope: str,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    if mode not in {"control", "api"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if loss_scope not in {"anchor_only", "symmetric"}:
        raise ValueError(f"Unsupported loss scope: {loss_scope}")
    feature_dim = int(features.shape[1])
    initial_state = _initial_classifier_state(feature_dim, class_count, seed=seed)
    _set_seed(seed + 1)
    if mode == "api":
        model: nn.Module = AttentivePairwiseInteractionHead(
            feature_dim,
            int(hidden_dim),
            int(class_count),
        )
        model.classifier.load_state_dict(initial_state)
    else:
        model = nn.Linear(feature_dim, int(class_count))
        model.load_state_dict(initial_state)
    model.to(device)
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    class_weights = torch.from_numpy(_balanced_class_weights(labels, class_count)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    epoch_batches = pair_plan["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(epoch_batches)),
        eta_min=float(learning_rate) * 0.05,
    )
    curves: List[Dict[str, float]] = []
    model.train()
    for epoch_index, batches in enumerate(epoch_batches):
        loss_sum = 0.0
        ce_sum = 0.0
        ranking_sum = 0.0
        pair_sum = 0
        for pair_batch in batches:
            first_indices = torch.from_numpy(pair_batch["first"]).to(device=device)
            second_indices = torch.from_numpy(pair_batch["second"]).to(device=device)
            first = feature_tensor[first_indices]
            second = feature_tensor[second_indices]
            first_labels = label_tensor[first_indices]
            second_labels = label_tensor[second_indices]
            optimizer.zero_grad(set_to_none=True)
            if mode == "api":
                loss, cross_entropy, ranking = _api_loss(
                    model,
                    first,
                    second,
                    first_labels,
                    second_labels,
                    class_weights=class_weights,
                    ranking_margin=float(ranking_margin),
                    ranking_weight=float(ranking_weight),
                    loss_scope=str(loss_scope),
                )
            else:
                first_logits = model(first)
                first_loss = F.cross_entropy(
                    first_logits,
                    first_labels,
                    weight=class_weights,
                )
                if loss_scope == "anchor_only":
                    loss = first_loss
                else:
                    second_logits = model(second)
                    loss = 0.5 * (
                        first_loss
                        + F.cross_entropy(
                            second_logits,
                            second_labels,
                            weight=class_weights,
                        )
                    )
                cross_entropy = loss.detach()
                ranking = loss.detach().new_zeros(())
            loss.backward()
            optimizer.step()
            count = int(first_indices.numel())
            loss_sum += float(loss.detach().item()) * count
            ce_sum += float(cross_entropy.item()) * count
            ranking_sum += float(ranking.item()) * count
            pair_sum += count
        scheduler.step()
        denominator = max(1, pair_sum)
        row = {
            "epoch": float(epoch_index + 1),
            "loss": loss_sum / denominator,
            "cross_entropy": ce_sum / denominator,
            "ranking": ranking_sum / denominator,
            "pair_count": float(pair_sum),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        curves.append(row)
        if epoch_index == 0 or (epoch_index + 1) % 5 == 0 or epoch_index + 1 == len(epoch_batches):
            print(
                f"{mode} epoch {epoch_index + 1}/{len(epoch_batches)} "
                f"loss={row['loss']:.6f} ce={row['cross_entropy']:.6f} "
                f"rank={row['ranking']:.6f}",
                flush=True,
            )
    return model, curves


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    probabilities: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            logits = (
                model.inference_logits(batch)
                if isinstance(model, AttentivePairwiseInteractionHead)
                else model(batch)
            )
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32, copy=False)


def assess_api_pairwise_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_group_count: int,
    train_val_source_overlap: int,
    max_fold_source_overlap: int,
    pair_support_ok: bool,
    same_source_pair_count: int,
    control_oof_metrics: Mapping[str, object],
    api_oof_metrics: Mapping[str, object],
    control_val_metrics: Mapping[str, object],
    api_val_metrics: Mapping[str, object],
    direct_metrics: Mapping[str, object],
    oof_transitions: Mapping[str, int],
    val_transitions: Mapping[str, int],
    direct_transitions: Mapping[str, int],
) -> Dict[str, object]:
    thresholds = {
        "min_train_samples": 9000,
        "required_val_samples": 2606,
        "min_source_groups": 8000,
        "min_oof_macro_gain": 0.002,
        "min_oof_class1_gain": 0.010,
        "min_val_macro_gain": 0.002,
        "min_val_class1_gain": 0.015,
        "min_val_class1_f1": 0.70,
        "max_direct_macro_drop": 0.003,
    }
    observed = {
        "oof_macro_gain": float(api_oof_metrics["macro_f1"])
        - float(control_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(api_oof_metrics["focus_f1"])
        - float(control_oof_metrics["focus_f1"]),
        "val_macro_gain": float(api_val_metrics["macro_f1"])
        - float(control_val_metrics["macro_f1"]),
        "val_class1_gain": float(api_val_metrics["focus_f1"])
        - float(control_val_metrics["focus_f1"]),
        "direct_macro_drop": float(direct_metrics["macro_f1"])
        - float(api_val_metrics["macro_f1"]),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "source_group_support": int(source_group_count) >= int(thresholds["min_source_groups"]),
        "train_val_source_isolation": int(train_val_source_overlap) == 0,
        "fold_source_isolation": int(max_fold_source_overlap) == 0,
        "pair_support_all_classes": bool(pair_support_ok),
        "same_source_pairs_excluded": int(same_source_pair_count) == 0,
        "oof_macro_gain": observed["oof_macro_gain"] >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"] >= float(thresholds["min_oof_class1_gain"]),
        "val_macro_gain": observed["val_macro_gain"] >= float(thresholds["min_val_macro_gain"]),
        "val_class1_gain": observed["val_class1_gain"] >= float(thresholds["min_val_class1_gain"]),
        "oof_to_val_gain_direction": observed["oof_macro_gain"] > 0.0
        and observed["oof_class1_gain"] > 0.0
        and observed["val_macro_gain"] > 0.0
        and observed["val_class1_gain"] > 0.0,
        "val_class1_milestone": float(api_val_metrics["focus_f1"])
        >= float(thresholds["min_val_class1_f1"]),
        "direct_macro_preserved": observed["direct_macro_drop"]
        <= float(thresholds["max_direct_macro_drop"]),
        "oof_net_corrections": int(oof_transitions["corrections"])
        >= int(oof_transitions["harms"]),
        "val_net_corrections": int(val_transitions["corrections"])
        >= int(val_transitions["harms"]),
        "direct_net_corrections": int(direct_transitions["corrections"])
        >= int(direct_transitions["harms"]),
        "oof_class1_recall_protected": int(oof_transitions["class1_fn_rescued"])
        >= int(oof_transitions["class1_tp_broken"]),
        "val_class1_recall_protected": int(val_transitions["class1_fn_rescued"])
        >= int(val_transitions["class1_tp_broken"]),
        "direct_class1_recall_protected": int(direct_transitions["class1_fn_rescued"])
        >= int(direct_transitions["class1_tp_broken"]),
        "oof_class1_fp_control": int(oof_transitions["class1_fp_removed"])
        >= int(oof_transitions["class1_fp_created"]),
        "val_class1_fp_control": int(val_transitions["class1_fp_removed"])
        >= int(val_transitions["class1_fp_created"]),
        "direct_class1_fp_control": int(direct_transitions["class1_fp_removed"])
        >= int(direct_transitions["class1_fp_created"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "api_pairwise_interaction_ready": ready,
        "smoke_ready": ready,
        "smoke_permission": ready,
        "full_train_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _compact_curve(curve: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    if not curve:
        return {"epochs": 0, "first": {}, "last": {}}
    return {
        "epochs": len(curve),
        "first": dict(curve[0]),
        "last": dict(curve[-1]),
        "minimum_loss": float(min(float(row["loss"]) for row in curve)),
    }


def _write_artifact_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name.casefold()):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(str(row["name"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("ascii"))
    manifest = {
        "mode": "api_pairwise_interaction_readiness_evidence_manifest",
        "payload_count": len(rows),
        "payload_size_bytes": int(sum(int(row["size_bytes"]) for row in rows)),
        "payload_manifest_sha256": aggregate.hexdigest(),
        "files": rows,
        "contains_checkpoint": False,
        "contains_model_binary": False,
        "contains_test_payload": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) < 2:
        raise ValueError("folds must be at least two")
    if not 1 <= int(args.epochs) <= 30:
        raise ValueError("epochs must be in [1, 30]")
    if int(args.batch_size) < 16:
        raise ValueError("batch-size must be at least 16")
    if int(args.hidden_dim) <= 0:
        raise ValueError("hidden-dim must be positive")
    if float(args.learning_rate) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")
    if float(args.ranking_weight) < 0.0 or float(args.ranking_margin) < 0.0:
        raise ValueError("ranking settings must be nonnegative")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    torch.set_float32_matmul_precision("high")

    cache_dir = Path(args.cache_dir)
    train_path = cache_dir / "train_interior_second_order_descriptors.npz"
    val_path = cache_dir / "val_interior_second_order_descriptors.npz"
    train = _load_cache(train_path)
    val = _load_cache(val_path)
    if not np.array_equal(train["classes"], val["classes"]):
        raise ValueError("Train and validation class orders differ")
    class_names = [str(value) for value in train["classes"]]
    class_count = len(class_names)
    train_features = _l2_normalize(train["features"])
    val_features = _l2_normalize(val["features"])
    train_labels = train["labels"]
    val_labels = val["labels"]
    train_groups = np.asarray([str(value).casefold() for value in train["groups"]], dtype=object)
    val_groups = np.asarray([str(value).casefold() for value in val["groups"]], dtype=object)
    train_val_overlap = len(set(train_groups.tolist()).intersection(val_groups.tolist()))
    device = _resolve_device(str(args.device or ""))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds),
        shuffle=True,
        random_state=SEED,
    )
    split_indices = list(
        splitter.split(np.zeros(train_labels.size), train_labels, groups=train_groups)
    )
    control_oof = np.zeros((train_labels.size, class_count), dtype=np.float32)
    api_oof = np.zeros_like(control_oof)
    fold_rows: List[Dict[str, object]] = []
    curve_payload: Dict[str, object] = {"folds": []}
    max_fold_source_overlap = 0
    all_pair_support_ok = True
    total_same_source_pairs = 0

    for fold_index, (fit_indices, holdout_indices) in enumerate(split_indices):
        fit_groups = train_groups[fit_indices]
        holdout_groups = train_groups[holdout_indices]
        overlap = len(set(fit_groups.tolist()).intersection(holdout_groups.tolist()))
        max_fold_source_overlap = max(max_fold_source_overlap, overlap)
        fold_seed = SEED + 1000 * (fold_index + 1)
        pair_plan = build_pair_plan(
            train_features[fit_indices],
            train_labels[fit_indices],
            fit_groups,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=fold_seed,
            class_count=class_count,
        )
        pair_stats = pair_plan["stats"]
        pair_support_ok = bool(pair_stats["all_classes_have_intra_support"]) and bool(
            pair_stats["all_classes_have_inter_support"]
        )
        all_pair_support_ok = all_pair_support_ok and pair_support_ok
        total_same_source_pairs += int(pair_stats["same_source_pair_count"])
        print(
            f"fold {fold_index + 1}/{len(split_indices)} fit={fit_indices.size} "
            f"holdout={holdout_indices.size} pairs={pair_stats['pair_count']}",
            flush=True,
        )
        control_model, control_curve = _train_pairwise_head(
            features=train_features[fit_indices],
            labels=train_labels[fit_indices],
            pair_plan=pair_plan,
            class_count=class_count,
            hidden_dim=int(args.hidden_dim),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            ranking_margin=float(args.ranking_margin),
            ranking_weight=float(args.ranking_weight),
            device=device,
            seed=fold_seed + 17,
            mode="control",
            loss_scope=str(args.loss_scope),
        )
        control_oof[holdout_indices] = _predict_probabilities(
            control_model,
            train_features[holdout_indices],
            device=device,
        )
        del control_model
        api_model, api_curve = _train_pairwise_head(
            features=train_features[fit_indices],
            labels=train_labels[fit_indices],
            pair_plan=pair_plan,
            class_count=class_count,
            hidden_dim=int(args.hidden_dim),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            ranking_margin=float(args.ranking_margin),
            ranking_weight=float(args.ranking_weight),
            device=device,
            seed=fold_seed + 17,
            mode="api",
            loss_scope=str(args.loss_scope),
        )
        api_oof[holdout_indices] = _predict_probabilities(
            api_model,
            train_features[holdout_indices],
            device=device,
        )
        del api_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fold_control_metrics = _classification_metrics(
            train_labels[holdout_indices],
            control_oof[holdout_indices],
            class_names=class_names,
        )
        fold_api_metrics = _classification_metrics(
            train_labels[holdout_indices],
            api_oof[holdout_indices],
            class_names=class_names,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_samples": int(fit_indices.size),
                "holdout_samples": int(holdout_indices.size),
                "fit_source_groups": int(np.unique(fit_groups).size),
                "holdout_source_groups": int(np.unique(holdout_groups).size),
                "source_overlap": int(overlap),
                "pair_plan": pair_stats,
                "control_metrics": fold_control_metrics,
                "api_metrics": fold_api_metrics,
                "macro_gain": float(fold_api_metrics["macro_f1"])
                - float(fold_control_metrics["macro_f1"]),
                "class1_gain": float(fold_api_metrics["focus_f1"])
                - float(fold_control_metrics["focus_f1"]),
            }
        )
        curve_payload["folds"].append(
            {
                "fold": fold_index,
                "control": control_curve,
                "api": api_curve,
            }
        )

    final_seed = SEED + 9000
    final_pair_plan = build_pair_plan(
        train_features,
        train_labels,
        train_groups,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=final_seed,
        class_count=class_count,
    )
    final_pair_stats = final_pair_plan["stats"]
    all_pair_support_ok = all_pair_support_ok and bool(
        final_pair_stats["all_classes_have_intra_support"]
    ) and bool(final_pair_stats["all_classes_have_inter_support"])
    total_same_source_pairs += int(final_pair_stats["same_source_pair_count"])
    print(f"final fit pairs={final_pair_stats['pair_count']}", flush=True)
    control_model, final_control_curve = _train_pairwise_head(
        features=train_features,
        labels=train_labels,
        pair_plan=final_pair_plan,
        class_count=class_count,
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        ranking_margin=float(args.ranking_margin),
        ranking_weight=float(args.ranking_weight),
        device=device,
        seed=final_seed + 17,
        mode="control",
        loss_scope=str(args.loss_scope),
    )
    control_val = _predict_probabilities(control_model, val_features, device=device)
    del control_model
    api_model, final_api_curve = _train_pairwise_head(
        features=train_features,
        labels=train_labels,
        pair_plan=final_pair_plan,
        class_count=class_count,
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        ranking_margin=float(args.ranking_margin),
        ranking_weight=float(args.ranking_weight),
        device=device,
        seed=final_seed + 17,
        mode="api",
        loss_scope=str(args.loss_scope),
    )
    api_val = _predict_probabilities(api_model, val_features, device=device)
    del api_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    curve_payload["final"] = {
        "control": final_control_curve,
        "api": final_api_curve,
    }
    (output_dir / "training_curves.json").write_text(
        json.dumps(curve_payload, indent=2),
        encoding="utf-8",
    )

    control_oof_metrics = _classification_metrics(
        train_labels,
        control_oof,
        class_names=class_names,
    )
    api_oof_metrics = _classification_metrics(
        train_labels,
        api_oof,
        class_names=class_names,
    )
    control_val_metrics = _classification_metrics(
        val_labels,
        control_val,
        class_names=class_names,
    )
    api_val_metrics = _classification_metrics(
        val_labels,
        api_val,
        class_names=class_names,
    )
    direct_metrics = _classification_metrics(
        val_labels,
        val["probabilities"],
        class_names=class_names,
    )
    oof_transitions = _transition_summary(train_labels, control_oof, api_oof)
    val_transitions = _transition_summary(val_labels, control_val, api_val)
    direct_transitions = _transition_summary(val_labels, val["probabilities"], api_val)
    gate = assess_api_pairwise_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_group_count=int(np.unique(train_groups).size),
        train_val_source_overlap=int(train_val_overlap),
        max_fold_source_overlap=int(max_fold_source_overlap),
        pair_support_ok=bool(all_pair_support_ok),
        same_source_pair_count=int(total_same_source_pairs),
        control_oof_metrics=control_oof_metrics,
        api_oof_metrics=api_oof_metrics,
        control_val_metrics=control_val_metrics,
        api_val_metrics=api_val_metrics,
        direct_metrics=direct_metrics,
        oof_transitions=oof_transitions,
        val_transitions=val_transitions,
        direct_transitions=direct_transitions,
    )
    _write_prediction_audit(
        output_dir / "train_oof_predictions.csv",
        labels=train_labels,
        sample_index=train["sample_index"],
        paths=train["paths"],
        base_probabilities=train["probabilities"],
        variants={"control_oof": control_oof, "api_oof": api_oof},
    )
    _write_prediction_audit(
        output_dir / "val_predictions.csv",
        labels=val_labels,
        sample_index=val["sample_index"],
        paths=val["paths"],
        base_probabilities=val["probabilities"],
        variants={"control": control_val, "api": api_val},
    )

    summary = {
        "mode": "api_pairwise_interaction_readiness_precheck",
        "guardrail": (
            "Frozen in-sample keeper head cache; candidate-vs-matched-control direction must "
            "transfer from source-grouped folds to full validation. No test, image-model "
            "training, checkpoint, trainable manifest, inference pair/router, or raw-data edit."
        ),
        "cache_dir": str(cache_dir.resolve()),
        "cache_sha256": {
            "train": _sha256(train_path),
            "val": _sha256(val_path),
        },
        "train_samples": int(train_labels.size),
        "val_samples": int(val_labels.size),
        "source_group_count": int(np.unique(train_groups).size),
        "train_val_source_overlap": int(train_val_overlap),
        "class_names": class_names,
        "class_counts": {
            "train": np.bincount(train_labels, minlength=class_count).tolist(),
            "val": np.bincount(val_labels, minlength=class_count).tolist(),
        },
        "protocol": {
            "feature": "l2_normalized_keeper_head_256d",
            "candidate": "api_net_training_only_mutual_gate_score_ranking",
            "control": "plain_linear_head_with_identical_initialization_pairs_batches_optimizer",
            "mutual_mlp": [int(train_features.shape[1]) * 2, int(args.hidden_dim), int(train_features.shape[1])],
            "gate": "sigmoid(mutual_elementwise_individual)",
            "inference": "pair_module_unloaded_shared_classifier_on_single_plain_feature",
            "pairing": "nearest_intra_and_inter_class_within_natural_frequency_batch",
            "same_source_pairing": False,
            "loss_scope": str(args.loss_scope),
            "anchor_oversampling": False,
            "partner_reuse": (
                "context_only_no_label_loss"
                if str(args.loss_scope) == "anchor_only"
                else "symmetric_label_loss_original_api"
            ),
            "class_loss_weighting": "inverse_frequency_balanced_ce_for_both_control_and_api",
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "schedule": "cosine_to_5_percent",
            "ranking_margin": float(args.ranking_margin),
            "ranking_weight": float(args.ranking_weight),
            "seed": SEED,
            "candidate_sweep": False,
            "validation_selection": False,
            "known_limitation": (
                "Train embeddings are in-sample outputs of the frozen keeper, so OOF absolute "
                "metrics are optimistic; only matched API-vs-control direction plus untouched "
                "validation transfer can open a model smoke."
            ),
        },
        "pair_audit": {
            "folds": fold_rows,
            "final": final_pair_stats,
            "max_fold_source_overlap": int(max_fold_source_overlap),
            "all_pair_support_ok": bool(all_pair_support_ok),
            "same_source_pair_count": int(total_same_source_pairs),
        },
        "metrics": {
            "direct_keeper_val": direct_metrics,
            "control_oof": control_oof_metrics,
            "api_oof": api_oof_metrics,
            "control_val": control_val_metrics,
            "api_val": api_val_metrics,
        },
        "transitions": {
            "control_oof_to_api_oof": oof_transitions,
            "control_val_to_api_val": val_transitions,
            "direct_keeper_val_to_api_val": direct_transitions,
        },
        "curve_summary": {
            "folds": [
                {
                    "fold": row["fold"],
                    "control": _compact_curve(curve_payload["folds"][index]["control"]),
                    "api": _compact_curve(curve_payload["folds"][index]["api"]),
                }
                for index, row in enumerate(fold_rows)
            ],
            "final": {
                "control": _compact_curve(final_control_curve),
                "api": _compact_curve(final_api_curve),
            },
        },
        "gate": gate,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "literature": list(LITERATURE),
        "raw_dataset_touched": False,
        "test_split_used": False,
        "image_model_trained": False,
        "model_written": False,
        "checkpoint_written": False,
        "trainable_manifest_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        f"# API-Net Pairwise Interaction Readiness Precheck ({args.loss_scope})",
        "",
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`",
        f"- Direct keeper val macro/class1: `{float(direct_metrics['macro_f1']):.6f}/{float(direct_metrics['focus_f1']):.6f}`",
        f"- Matched control OOF macro/class1: `{float(control_oof_metrics['macro_f1']):.6f}/{float(control_oof_metrics['focus_f1']):.6f}`",
        f"- API OOF macro/class1: `{float(api_oof_metrics['macro_f1']):.6f}/{float(api_oof_metrics['focus_f1']):.6f}`",
        f"- Matched control val macro/class1: `{float(control_val_metrics['macro_f1']):.6f}/{float(control_val_metrics['focus_f1']):.6f}`",
        f"- API val macro/class1: `{float(api_val_metrics['macro_f1']):.6f}/{float(api_val_metrics['focus_f1']):.6f}`",
        f"- Smoke ready: `{str(bool(gate['smoke_ready'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "This is a fixed diagnostic-only gate. It writes no model and does not read test.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    summary["artifact_manifest_path"] = str(
        (output_dir / "artifact_manifest.json").resolve()
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "control_oof": control_oof_metrics,
                "api_oof": api_oof_metrics,
                "control_val": control_val_metrics,
                "api_val": api_val_metrics,
                "transitions": summary["transitions"],
                "gate": gate,
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    _write_artifact_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
