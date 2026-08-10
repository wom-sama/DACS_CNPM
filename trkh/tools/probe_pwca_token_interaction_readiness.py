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

from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _balanced_class_weights,
    _set_seed,
    _sha256,
    _write_artifact_manifest,
)
from trkh.tools.probe_photometric_invariant_complementarity import (
    _classification_metrics,
    _transition_summary,
    _write_prediction_audit,
)


SEED = 20260711
LITERATURE = (
    "https://openaccess.thecvf.com/content/CVPR2022/html/"
    "Zhu_Dual_Cross-Attention_Learning_for_Fine-Grained_Visual_"
    "Categorization_and_Object_Re-Identification_CVPR_2022_paper.html",
    "https://openaccess.thecvf.com/content/CVPR2022/supplemental/"
    "Zhu_Dual_Cross-Attention_Learning_CVPR_2022_supplemental.pdf",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-grouped frozen-token proxy for DCAL pair-wise cross-attention. "
            "PWCA is training-only; validation uses the shared self-attention branch."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--pwca-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "").strip()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if "test" in str(path).casefold():
        raise ValueError(f"Test cache is forbidden: {path}")
    required = {
        "tokens",
        "probabilities",
        "labels",
        "sample_index",
        "paths",
        "source_stem",
        "classes",
        "patch_indices",
        "selected_attention",
    }
    with np.load(path, allow_pickle=True) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"PWCA cache is missing keys {missing}: {path}")
        result = {
            "tokens": np.asarray(payload["tokens"], dtype=np.float16),
            "probabilities": np.asarray(payload["probabilities"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.int64).reshape(-1),
            "sample_index": np.asarray(payload["sample_index"], dtype=np.int64).reshape(-1),
            "paths": np.asarray(payload["paths"], dtype=object).reshape(-1),
            "groups": np.asarray(payload["source_stem"], dtype=object).reshape(-1),
            "classes": np.asarray(payload["classes"], dtype=object).reshape(-1),
            "patch_indices": np.asarray(payload["patch_indices"], dtype=np.int16),
            "selected_attention": np.asarray(
                payload["selected_attention"], dtype=np.float16
            ),
        }
    row_count = int(result["labels"].size)
    class_count = int(result["classes"].size)
    if result["tokens"].ndim != 3 or int(result["tokens"].shape[1]) < 2:
        raise ValueError(f"Token cache must be [N,S,D] with S>=2: {path}")
    if tuple(result["probabilities"].shape) != (row_count, class_count):
        raise ValueError(f"Probability shape is inconsistent: {path}")
    if tuple(result["patch_indices"].shape) != (
        row_count,
        int(result["tokens"].shape[1]) - 1,
    ):
        raise ValueError(f"Patch-index shape is inconsistent: {path}")
    if tuple(result["selected_attention"].shape) != tuple(
        result["patch_indices"].shape
    ):
        raise ValueError(f"Selected-attention shape is inconsistent: {path}")
    for key in ("tokens", "sample_index", "paths", "groups"):
        if int(result[key].shape[0]) != row_count:
            raise ValueError(f"Cache key {key} has inconsistent row count: {path}")
    if not np.isfinite(result["tokens"]).all():
        raise ValueError(f"Token cache contains non-finite values: {path}")
    if np.unique(result["sample_index"]).size != row_count:
        raise ValueError(f"sample_index is not unique: {path}")
    if any(not str(value).strip() for value in result["groups"]):
        raise ValueError(f"source_stem contains empty values: {path}")
    return result


def _partner_plan_fingerprint(
    epoch_batches: Sequence[Sequence[Mapping[str, np.ndarray]]],
) -> str:
    digest = hashlib.sha256()
    for epoch_index, batches in enumerate(epoch_batches):
        digest.update(np.asarray([epoch_index, len(batches)], dtype=np.int64).tobytes())
        for batch in batches:
            for key in ("anchor", "partner"):
                values = np.asarray(batch[key], dtype=np.int64)
                digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
                digest.update(values.tobytes())
    return digest.hexdigest()


def _random_different_source_partner(
    indices: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        raise ValueError("A PWCA batch needs at least two anchors")
    anchor_groups = groups[indices]
    if np.unique(anchor_groups).size < 2:
        raise ValueError("A PWCA batch needs at least two source groups")
    for _ in range(128):
        partner = indices[rng.permutation(indices.size)]
        if bool(np.all(groups[partner] != anchor_groups)):
            return partner
    partner = np.empty_like(indices)
    for row, anchor_index in enumerate(indices):
        candidates = indices[anchor_groups != groups[anchor_index]]
        if candidates.size == 0:
            raise ValueError("Could not find a different-source distractor")
        partner[row] = candidates[int(rng.integers(0, candidates.size))]
    return partner


def build_random_partner_plan(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    class_count: Optional[int] = None,
) -> Dict[str, object]:
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    source_groups = np.asarray(groups, dtype=object).reshape(-1)
    if targets.size != source_groups.size:
        raise ValueError("labels and groups must have aligned rows")
    if int(epochs) <= 0 or int(batch_size) < 4:
        raise ValueError("epochs must be positive and batch_size at least four")
    if class_count is None:
        class_count = int(targets.max(initial=-1)) + 1
    class_count = int(class_count)
    rng = np.random.default_rng(int(seed))
    epoch_batches: List[List[Dict[str, np.ndarray]]] = []
    anchor_visits = np.zeros(targets.size, dtype=np.int64)
    partner_visits = np.zeros(targets.size, dtype=np.int64)
    intra_count = 0
    inter_count = 0
    same_source_count = 0

    for _ in range(int(epochs)):
        permutation = rng.permutation(targets.size)
        chunks = [
            permutation[start : start + int(batch_size)]
            for start in range(0, targets.size, int(batch_size))
        ]
        if len(chunks) > 1 and chunks[-1].size < 2:
            chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
            chunks.pop()
        batches: List[Dict[str, np.ndarray]] = []
        for anchor in chunks:
            partner = _random_different_source_partner(anchor, source_groups, rng)
            anchor_visits[anchor] += 1
            np.add.at(partner_visits, partner, 1)
            same_source_count += int(
                np.sum(source_groups[anchor] == source_groups[partner])
            )
            same_class = targets[anchor] == targets[partner]
            intra_count += int(same_class.sum())
            inter_count += int((~same_class).sum())
            batches.append({"anchor": anchor, "partner": partner})
        epoch_batches.append(batches)

    if not np.all(anchor_visits == int(epochs)):
        raise RuntimeError("Every row must be an anchor exactly once per epoch")
    anchor_support = np.bincount(targets, minlength=class_count) * int(epochs)
    payload: Dict[str, object] = {
        "epochs": epoch_batches,
        "stats": {
            "samples": int(targets.size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "pair_count": int(targets.size * int(epochs)),
            "same_source_pair_count": int(same_source_count),
            "intra_class_pair_count": int(intra_count),
            "inter_class_pair_count": int(inter_count),
            "inter_class_fraction": float(inter_count / max(1, intra_count + inter_count)),
            "anchor_support_per_class": anchor_support.tolist(),
            "anchor_visit_min": int(anchor_visits.min()) if anchor_visits.size else 0,
            "anchor_visit_max": int(anchor_visits.max()) if anchor_visits.size else 0,
            "partner_visit_min": int(partner_visits.min()) if partner_visits.size else 0,
            "partner_visit_median": float(np.median(partner_visits))
            if partner_visits.size
            else 0.0,
            "partner_visit_p99": float(np.quantile(partner_visits, 0.99))
            if partner_visits.size
            else 0.0,
            "partner_visit_max": int(partner_visits.max()) if partner_visits.size else 0,
            "anchor_oversampling": False,
            "partner_label_loss": False,
            "label_blind_pairing": True,
        },
    }
    payload["stats"]["fingerprint_sha256"] = _partner_plan_fingerprint(epoch_batches)
    return payload


class SharedTokenAttentionReadout(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        mlp_ratio: float,
        class_count: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(token_dim) % int(num_heads) != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(self.token_dim)
        self.qkv = nn.Linear(self.token_dim, self.token_dim * 3)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.proj = nn.Linear(self.token_dim, self.token_dim)
        self.projection_dropout = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(self.token_dim)
        hidden_dim = max(self.token_dim, int(round(self.token_dim * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.token_dim),
            nn.Dropout(float(dropout)),
        )
        self.classifier_norm = nn.LayerNorm(self.token_dim)
        self.classifier = nn.Linear(self.token_dim, int(class_count))

    def _split_qkv(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, token_count, _ = tokens.shape
        qkv = self.qkv(self.norm1(tokens))
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(
        self,
        target_tokens: torch.Tensor,
        distractor_tokens: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
    ):
        if target_tokens.ndim != 3 or int(target_tokens.size(2)) != self.token_dim:
            raise ValueError("target_tokens must be [B,N,token_dim]")
        query, key, value = self._split_qkv(target_tokens)
        if distractor_tokens is not None:
            if tuple(distractor_tokens.shape) != tuple(target_tokens.shape):
                raise ValueError("distractor_tokens must match target_tokens")
            _, distractor_key, distractor_value = self._split_qkv(distractor_tokens)
            key = torch.cat((key, distractor_key), dim=2)
            value = torch.cat((value, distractor_value), dim=2)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = attention @ value
        output = output.transpose(1, 2).reshape_as(target_tokens)
        output = self.projection_dropout(self.proj(output))
        tokens = target_tokens + output
        tokens = tokens + self.mlp(self.norm2(tokens))
        logits = self.classifier(self.classifier_norm(tokens[:, 0]))
        if return_attention:
            return logits, attention
        return logits


def _initial_model_state(
    *,
    token_dim: int,
    num_heads: int,
    mlp_ratio: float,
    class_count: int,
    dropout: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    _set_seed(seed)
    model = SharedTokenAttentionReadout(
        token_dim,
        num_heads,
        mlp_ratio,
        class_count,
        dropout=dropout,
    )
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _train_model(
    *,
    tokens: np.ndarray,
    labels: np.ndarray,
    partner_plan: Mapping[str, object],
    class_count: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    pwca_loss_weight: float,
    device: torch.device,
    seed: int,
    mode: str,
) -> Tuple[SharedTokenAttentionReadout, List[Dict[str, float]]]:
    if mode not in {"control", "pwca"}:
        raise ValueError(f"Unsupported mode: {mode}")
    token_dim = int(tokens.shape[2])
    initial_state = _initial_model_state(
        token_dim=token_dim,
        num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio),
        class_count=int(class_count),
        dropout=float(dropout),
        seed=int(seed),
    )
    model = SharedTokenAttentionReadout(
        token_dim,
        int(num_heads),
        float(mlp_ratio),
        int(class_count),
        dropout=float(dropout),
    )
    model.load_state_dict(initial_state)
    model.to(device)
    token_tensor = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(device)
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    class_weights = torch.from_numpy(_balanced_class_weights(labels, class_count)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    epoch_batches = partner_plan["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(epoch_batches)),
        eta_min=float(learning_rate) * 0.05,
    )
    _set_seed(seed + 1)
    curves: List[Dict[str, float]] = []
    model.train()
    for epoch_index, batches in enumerate(epoch_batches):
        loss_sum = 0.0
        self_sum = 0.0
        pwca_sum = 0.0
        row_count = 0
        for pair_batch in batches:
            anchor = torch.from_numpy(pair_batch["anchor"]).to(device=device)
            partner = torch.from_numpy(pair_batch["partner"]).to(device=device)
            target_tokens = token_tensor[anchor]
            target_labels = label_tensor[anchor]
            optimizer.zero_grad(set_to_none=True)
            self_logits = model(target_tokens)
            self_loss = F.cross_entropy(
                self_logits,
                target_labels,
                weight=class_weights,
            )
            if mode == "pwca":
                pwca_logits = model(target_tokens, token_tensor[partner])
                pwca_loss = F.cross_entropy(
                    pwca_logits,
                    target_labels,
                    weight=class_weights,
                )
                loss = (
                    self_loss + float(pwca_loss_weight) * pwca_loss
                ) / (1.0 + float(pwca_loss_weight))
            else:
                pwca_loss = self_loss.detach().new_zeros(())
                loss = self_loss
            loss.backward()
            optimizer.step()
            count = int(anchor.numel())
            loss_sum += float(loss.detach().item()) * count
            self_sum += float(self_loss.detach().item()) * count
            pwca_sum += float(pwca_loss.detach().item()) * count
            row_count += count
        scheduler.step()
        denominator = max(1, row_count)
        row = {
            "epoch": float(epoch_index + 1),
            "loss": loss_sum / denominator,
            "self_cross_entropy": self_sum / denominator,
            "pwca_cross_entropy": pwca_sum / denominator,
            "anchors": float(row_count),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        curves.append(row)
        if epoch_index == 0 or (epoch_index + 1) % 5 == 0 or epoch_index + 1 == len(
            epoch_batches
        ):
            print(
                f"{mode} epoch {epoch_index + 1}/{len(epoch_batches)} "
                f"loss={row['loss']:.6f} self={row['self_cross_entropy']:.6f} "
                f"pwca={row['pwca_cross_entropy']:.6f}",
                flush=True,
            )
    return model, curves


def _predict(
    model: SharedTokenAttentionReadout,
    tokens: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    values = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
    output: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            output.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _stats(values: np.ndarray) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(data.mean()) if data.size else 0.0,
        "p05": float(np.quantile(data, 0.05)) if data.size else 0.0,
        "p50": float(np.quantile(data, 0.50)) if data.size else 0.0,
        "p95": float(np.quantile(data, 0.95)) if data.size else 0.0,
    }


def _attention_audit(
    model: SharedTokenAttentionReadout,
    tokens: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    plan = build_random_partner_plan(
        labels,
        groups,
        epochs=1,
        batch_size=int(batch_size),
        seed=int(seed),
        class_count=len(class_names),
    )
    values = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(device)
    target_labels = np.asarray(labels, dtype=np.int64)
    distractor_mass = np.zeros(target_labels.size, dtype=np.float32)
    pwca_probabilities = np.zeros((target_labels.size, len(class_names)), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for pair_batch in plan["epochs"][0]:
            anchor_np = pair_batch["anchor"]
            partner_np = pair_batch["partner"]
            anchor = torch.from_numpy(anchor_np).to(device=device)
            partner = torch.from_numpy(partner_np).to(device=device)
            logits, attention = model(
                values[anchor],
                values[partner],
                return_attention=True,
            )
            target_length = int(values.size(1))
            class_query = attention[:, :, 0]
            mass = class_query[:, :, target_length:].sum(dim=-1).mean(dim=1)
            distractor_mass[anchor_np] = mass.cpu().numpy()
            pwca_probabilities[anchor_np] = logits.softmax(dim=1).cpu().numpy()
    self_probabilities = _predict(model, tokens, device=device)
    self_predictions = self_probabilities.argmax(axis=1)
    masks = {
        "all": np.ones(target_labels.size, dtype=bool),
        "true_class1": target_labels == 1,
        "non_class1": target_labels != 1,
        "class1_false_negative": (target_labels == 1) & (self_predictions != 1),
        "class1_false_positive": (target_labels != 1) & (self_predictions == 1),
    }
    return {
        "partner_plan": plan["stats"],
        "distractor_mass": {
            name: {"support": int(mask.sum()), **_stats(distractor_mass[mask])}
            for name, mask in masks.items()
        },
        "self_metrics": _classification_metrics(
            target_labels,
            self_probabilities,
            class_names=class_names,
        ),
        "pwca_branch_metrics": _classification_metrics(
            target_labels,
            pwca_probabilities,
            class_names=class_names,
        ),
    }


def assess_pwca_readiness(
    *,
    train_samples: int,
    val_samples: int,
    source_group_count: int,
    train_val_source_overlap: int,
    max_fold_source_overlap: int,
    pair_audit_ok: bool,
    control_oof_metrics: Mapping[str, object],
    pwca_oof_metrics: Mapping[str, object],
    control_val_metrics: Mapping[str, object],
    pwca_val_metrics: Mapping[str, object],
    direct_metrics: Mapping[str, object],
    oof_transitions: Mapping[str, int],
    val_transitions: Mapping[str, int],
    direct_transitions: Mapping[str, int],
    distractor_mass_mean: float,
    distractor_mass_p95: float,
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
        "max_distractor_mass_mean": 0.35,
        "max_distractor_mass_p95": 0.55,
    }
    observed = {
        "oof_macro_gain": float(pwca_oof_metrics["macro_f1"])
        - float(control_oof_metrics["macro_f1"]),
        "oof_class1_gain": float(pwca_oof_metrics["focus_f1"])
        - float(control_oof_metrics["focus_f1"]),
        "val_macro_gain": float(pwca_val_metrics["macro_f1"])
        - float(control_val_metrics["macro_f1"]),
        "val_class1_gain": float(pwca_val_metrics["focus_f1"])
        - float(control_val_metrics["focus_f1"]),
        "direct_macro_drop": float(direct_metrics["macro_f1"])
        - float(pwca_val_metrics["macro_f1"]),
        "distractor_mass_mean": float(distractor_mass_mean),
        "distractor_mass_p95": float(distractor_mass_p95),
    }
    checks = {
        "train_support": int(train_samples) >= int(thresholds["min_train_samples"]),
        "val_support_complete": int(val_samples) == int(thresholds["required_val_samples"]),
        "source_group_support": int(source_group_count) >= int(thresholds["min_source_groups"]),
        "train_val_source_isolation": int(train_val_source_overlap) == 0,
        "fold_source_isolation": int(max_fold_source_overlap) == 0,
        "natural_pair_audit": bool(pair_audit_ok),
        "oof_macro_gain": observed["oof_macro_gain"] >= float(thresholds["min_oof_macro_gain"]),
        "oof_class1_gain": observed["oof_class1_gain"] >= float(thresholds["min_oof_class1_gain"]),
        "val_macro_gain": observed["val_macro_gain"] >= float(thresholds["min_val_macro_gain"]),
        "val_class1_gain": observed["val_class1_gain"] >= float(thresholds["min_val_class1_gain"]),
        "oof_to_val_gain_direction": observed["oof_macro_gain"] > 0.0
        and observed["oof_class1_gain"] > 0.0
        and observed["val_macro_gain"] > 0.0
        and observed["val_class1_gain"] > 0.0,
        "val_class1_milestone": float(pwca_val_metrics["focus_f1"])
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
        "distractor_attention_mean_safe": float(distractor_mass_mean)
        <= float(thresholds["max_distractor_mass_mean"]),
        "distractor_attention_tail_safe": float(distractor_mass_p95)
        <= float(thresholds["max_distractor_mass_p95"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "pwca_token_interaction_ready": ready,
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


def run_probe(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.folds) < 2:
        raise ValueError("folds must be at least two")
    if not 1 <= int(args.epochs) <= 30:
        raise ValueError("epochs must be in [1,30]")
    if int(args.batch_size) < 16:
        raise ValueError("batch-size must be at least 16")
    if float(args.pwca_loss_weight) <= 0.0:
        raise ValueError("pwca-loss-weight must be positive")
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    torch.set_float32_matmul_precision("high")
    cache_dir = Path(args.cache_dir)
    train_path = cache_dir / "train_pwca_token_cache.npz"
    val_path = cache_dir / "val_pwca_token_cache.npz"
    train = _load_cache(train_path)
    val = _load_cache(val_path)
    if not np.array_equal(train["classes"], val["classes"]):
        raise ValueError("Train and validation class orders differ")
    if tuple(train["tokens"].shape[1:]) != tuple(val["tokens"].shape[1:]):
        raise ValueError("Train and validation token shapes differ")
    class_names = [str(value) for value in train["classes"]]
    class_count = len(class_names)
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
    pwca_oof = np.zeros_like(control_oof)
    fold_rows: List[Dict[str, object]] = []
    curves: Dict[str, object] = {"folds": []}
    max_fold_overlap = 0
    pair_audit_ok = True

    for fold_index, (fit_indices, holdout_indices) in enumerate(split_indices):
        fit_groups = train_groups[fit_indices]
        holdout_groups = train_groups[holdout_indices]
        overlap = len(set(fit_groups.tolist()).intersection(holdout_groups.tolist()))
        max_fold_overlap = max(max_fold_overlap, overlap)
        fold_seed = SEED + 1000 * (fold_index + 1)
        plan = build_random_partner_plan(
            train_labels[fit_indices],
            fit_groups,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=fold_seed,
            class_count=class_count,
        )
        pair_audit_ok = pair_audit_ok and int(plan["stats"]["same_source_pair_count"]) == 0
        print(
            f"fold {fold_index + 1}/{len(split_indices)} fit={fit_indices.size} "
            f"holdout={holdout_indices.size} pairs={plan['stats']['pair_count']}",
            flush=True,
        )
        common = dict(
            tokens=train["tokens"][fit_indices],
            labels=train_labels[fit_indices],
            partner_plan=plan,
            class_count=class_count,
            num_heads=int(args.num_heads),
            mlp_ratio=float(args.mlp_ratio),
            dropout=float(args.dropout),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            pwca_loss_weight=float(args.pwca_loss_weight),
            device=device,
            seed=fold_seed + 17,
        )
        control_model, control_curve = _train_model(mode="control", **common)
        control_oof[holdout_indices] = _predict(
            control_model,
            train["tokens"][holdout_indices],
            device=device,
        )
        del control_model
        pwca_model, pwca_curve = _train_model(mode="pwca", **common)
        pwca_oof[holdout_indices] = _predict(
            pwca_model,
            train["tokens"][holdout_indices],
            device=device,
        )
        del pwca_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        control_metrics = _classification_metrics(
            train_labels[holdout_indices],
            control_oof[holdout_indices],
            class_names=class_names,
        )
        pwca_metrics = _classification_metrics(
            train_labels[holdout_indices],
            pwca_oof[holdout_indices],
            class_names=class_names,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_samples": int(fit_indices.size),
                "holdout_samples": int(holdout_indices.size),
                "source_overlap": int(overlap),
                "partner_plan": plan["stats"],
                "control_metrics": control_metrics,
                "pwca_metrics": pwca_metrics,
                "macro_gain": float(pwca_metrics["macro_f1"])
                - float(control_metrics["macro_f1"]),
                "class1_gain": float(pwca_metrics["focus_f1"])
                - float(control_metrics["focus_f1"]),
            }
        )
        curves["folds"].append(
            {"fold": fold_index, "control": control_curve, "pwca": pwca_curve}
        )

    final_seed = SEED + 9000
    final_plan = build_random_partner_plan(
        train_labels,
        train_groups,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=final_seed,
        class_count=class_count,
    )
    pair_audit_ok = pair_audit_ok and int(final_plan["stats"]["same_source_pair_count"]) == 0
    common = dict(
        tokens=train["tokens"],
        labels=train_labels,
        partner_plan=final_plan,
        class_count=class_count,
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        pwca_loss_weight=float(args.pwca_loss_weight),
        device=device,
        seed=final_seed + 17,
    )
    print(f"final fit pairs={final_plan['stats']['pair_count']}", flush=True)
    control_model, final_control_curve = _train_model(mode="control", **common)
    control_val = _predict(control_model, val["tokens"], device=device)
    del control_model
    pwca_model, final_pwca_curve = _train_model(mode="pwca", **common)
    pwca_val = _predict(pwca_model, val["tokens"], device=device)
    attention_audit = _attention_audit(
        pwca_model,
        val["tokens"],
        val_labels,
        val_groups,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        seed=final_seed + 101,
    )
    del pwca_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    curves["final"] = {
        "control": final_control_curve,
        "pwca": final_pwca_curve,
    }
    (output_dir / "training_curves.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8"
    )

    control_oof_metrics = _classification_metrics(
        train_labels, control_oof, class_names=class_names
    )
    pwca_oof_metrics = _classification_metrics(
        train_labels, pwca_oof, class_names=class_names
    )
    control_val_metrics = _classification_metrics(
        val_labels, control_val, class_names=class_names
    )
    pwca_val_metrics = _classification_metrics(
        val_labels, pwca_val, class_names=class_names
    )
    direct_metrics = _classification_metrics(
        val_labels, val["probabilities"], class_names=class_names
    )
    oof_transitions = _transition_summary(train_labels, control_oof, pwca_oof)
    val_transitions = _transition_summary(val_labels, control_val, pwca_val)
    direct_transitions = _transition_summary(val_labels, val["probabilities"], pwca_val)
    distractor_all = attention_audit["distractor_mass"]["all"]
    gate = assess_pwca_readiness(
        train_samples=int(train_labels.size),
        val_samples=int(val_labels.size),
        source_group_count=int(np.unique(train_groups).size),
        train_val_source_overlap=int(train_val_overlap),
        max_fold_source_overlap=int(max_fold_overlap),
        pair_audit_ok=bool(pair_audit_ok),
        control_oof_metrics=control_oof_metrics,
        pwca_oof_metrics=pwca_oof_metrics,
        control_val_metrics=control_val_metrics,
        pwca_val_metrics=pwca_val_metrics,
        direct_metrics=direct_metrics,
        oof_transitions=oof_transitions,
        val_transitions=val_transitions,
        direct_transitions=direct_transitions,
        distractor_mass_mean=float(distractor_all["mean"]),
        distractor_mass_p95=float(distractor_all["p95"]),
    )
    _write_prediction_audit(
        output_dir / "train_oof_predictions.csv",
        labels=train_labels,
        sample_index=train["sample_index"],
        paths=train["paths"],
        base_probabilities=train["probabilities"],
        variants={"control_oof": control_oof, "pwca_oof": pwca_oof},
    )
    _write_prediction_audit(
        output_dir / "val_predictions.csv",
        labels=val_labels,
        sample_index=val["sample_index"],
        paths=val["paths"],
        base_probabilities=val["probabilities"],
        variants={"control": control_val, "pwca": pwca_val},
    )
    summary = {
        "mode": "pwca_token_interaction_readiness_precheck",
        "guardrail": (
            "Frozen low-rank keeper token cache; shared SA/PWCA readout; PWCA removed "
            "for inference; source-grouped OOF plus full validation. No test, image-model "
            "training, checkpoint, trainable manifest, or raw-data edit."
        ),
        "cache_dir": str(cache_dir.resolve()),
        "cache_sha256": {"train": _sha256(train_path), "val": _sha256(val_path)},
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
            "token_shape": list(train["tokens"].shape[1:]),
            "shared_branch": "one_pre_norm_mhsa_ffn_classifier",
            "control": "self_attention_ce_only",
            "candidate": "equal_weight_self_attention_and_pwca_target_ce",
            "pwca": "target_Q_with_concatenated_target_and_random_distractor_KV",
            "inference": "shared_self_attention_branch_only",
            "pairing": "label_blind_natural_frequency_random_different_source",
            "anchor_oversampling": False,
            "partner_label_loss": False,
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_heads": int(args.num_heads),
            "mlp_ratio": float(args.mlp_ratio),
            "dropout": float(args.dropout),
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "pwca_loss_weight": float(args.pwca_loss_weight),
            "schedule": "cosine_to_5_percent",
            "seed": SEED,
            "candidate_sweep": False,
            "validation_selection": False,
            "known_limitation": (
                "Keeper train tokens are in-sample and projected/frozen; absolute OOF "
                "metrics are optimistic. Only matched PWCA-vs-control direction transferring "
                "to validation can open an end-to-end model smoke."
            ),
        },
        "pair_audit": {
            "folds": fold_rows,
            "final": final_plan["stats"],
            "max_fold_source_overlap": int(max_fold_overlap),
            "pair_audit_ok": bool(pair_audit_ok),
        },
        "metrics": {
            "direct_keeper_val": direct_metrics,
            "control_oof": control_oof_metrics,
            "pwca_oof": pwca_oof_metrics,
            "control_val": control_val_metrics,
            "pwca_val": pwca_val_metrics,
        },
        "transitions": {
            "control_oof_to_pwca_oof": oof_transitions,
            "control_val_to_pwca_val": val_transitions,
            "direct_keeper_val_to_pwca_val": direct_transitions,
        },
        "attention_audit": attention_audit,
        "curve_summary": {
            "folds": [
                {
                    "fold": row["fold"],
                    "control": _compact_curve(curves["folds"][index]["control"]),
                    "pwca": _compact_curve(curves["folds"][index]["pwca"]),
                }
                for index, row in enumerate(fold_rows)
            ],
            "final": {
                "control": _compact_curve(final_control_curve),
                "pwca": _compact_curve(final_pwca_curve),
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
    summary["artifact_manifest_path"] = str(
        (output_dir / "artifact_manifest.json").resolve()
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# Token-Level PWCA Readiness Precheck",
        "",
        f"- Train/validation rows: `{train_labels.size}/{val_labels.size}`",
        f"- Direct keeper val macro/class1: `{float(direct_metrics['macro_f1']):.6f}/{float(direct_metrics['focus_f1']):.6f}`",
        f"- Control OOF macro/class1: `{float(control_oof_metrics['macro_f1']):.6f}/{float(control_oof_metrics['focus_f1']):.6f}`",
        f"- PWCA OOF macro/class1: `{float(pwca_oof_metrics['macro_f1']):.6f}/{float(pwca_oof_metrics['focus_f1']):.6f}`",
        f"- Control val macro/class1: `{float(control_val_metrics['macro_f1']):.6f}/{float(control_val_metrics['focus_f1']):.6f}`",
        f"- PWCA val macro/class1: `{float(pwca_val_metrics['macro_f1']):.6f}/{float(pwca_val_metrics['focus_f1']):.6f}`",
        f"- Distractor attention mean/p95: `{float(distractor_all['mean']):.6f}/{float(distractor_all['p95']):.6f}`",
        f"- Smoke ready: `{str(bool(gate['smoke_ready'])).lower()}`",
        f"- Failed checks: `{','.join(gate['failed_checks'])}`",
        "",
        "This fixed proxy writes no model and does not read test.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": summary["metrics"],
                "transitions": summary["transitions"],
                "attention_audit": attention_audit,
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
