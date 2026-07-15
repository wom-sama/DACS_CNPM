from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

# CuBLAS needs this before its first CUDA handle is created for deterministic GEMM.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.tools.audit_class_axis_multimodal_readiness import (
    CLASS_COUNT,
    EXPECTED_FEATURE_DIM,
    EXPECTED_FIT_INDEX_SHA256,
    EXPECTED_FIT_ROWS,
    EXPECTED_HOLDOUT_INDEX_SHA256,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_SOURCE_GROUPS,
    EXPECTED_TRAIN_ROWS,
    FOCUS_CLASS,
    LOCKED_CACHE_MANIFEST_SHA256,
    LOCKED_CAGRAD_PREDICTIONS_SHA256,
    LOCKED_CAGRAD_SUMMARY_SHA256,
    LOCKED_TRAIN_CACHE_SHA256,
    _ordered_index_sha256,
    _read_comparator_holdout,
    _sha256,
    _tracked_worktree_clean,
    _verify_sha256,
    candidate_evidence,
)
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _load_cache,
)
from trkh.tools.audit_xca_dual_axis_readiness import (
    _git_commit,
    _prepare_output_dir,
)


SEED = 20260715
EPOCHS = 30
BATCH_SIZE = 42
HIDDEN_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
RECONSTRUCTION_WEIGHT = 0.01
CLASS_CENTER_WEIGHT = 0.1
SHARED_CENTER_WEIGHT = 0.1
CENTER_UPDATE_RATE = 0.01

LOCKED_PROTOCOL_SHA256 = (
    "1d5dba1afb9639db92c9930684039234d6693620bb0baece2f59f951f03eaafb"
)
LOCKED_PAPER_SHA256 = (
    "f4cf6a6c126389f39918b07c00d7d23019d25757815999c70b802b0d03b42506"
)
LOCKED_OFFICIAL_COMMIT = "8527039b0eb6e0eb7470521844a91acfac865fe5"
LOCKED_OFFICIAL_FC_SHA256 = (
    "180dfce4a5b29842a335a3dfab28b7d6d2c29224357b63938d105a1a2d9730bc"
)
LOCKED_OFFICIAL_ALL_SHA256 = (
    "dd6a63e80d258ef50946326d673ff586e47b0ee1702d81e1237cf2ad40712a51"
)


class GSFLFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = EXPECTED_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=False),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class GSFLAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = EXPECTED_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        class_count: int = CLASS_COUNT,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        # Discriminative first so paired seed resets reproduce control dropout masks.
        self.discriminative_encoder = GSFLFeatureEncoder(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(int(hidden_dim), int(class_count))
        self.shared_encoder = GSFLFeatureEncoder(input_dim, hidden_dim, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=False),
            nn.Linear(int(hidden_dim), int(input_dim)),
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        discriminative = self.discriminative_encoder(features)
        logits = self.classifier(discriminative)
        shared = self.shared_encoder(features)
        reconstruction = self.decoder(shared + discriminative)
        return {
            "logits": logits,
            "discriminative": discriminative,
            "shared": shared,
            "reconstruction": reconstruction,
        }


class GSFLCEControl(nn.Module):
    def __init__(
        self,
        input_dim: int = EXPECTED_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        class_count: int = CLASS_COUNT,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.discriminative_encoder = GSFLFeatureEncoder(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(int(hidden_dim), int(class_count))

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        discriminative = self.discriminative_encoder(features)
        return {
            "logits": self.classifier(discriminative),
            "discriminative": discriminative,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only GSFL shared/discriminative frozen-feature gate. "
            "Validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path(
            "runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/"
            "train_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path(
            "runs/diagnostic_reslt_embedding_cache_keeper_yolof_20260712/"
            "embedding_cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--cagrad-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_cagrad_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--cagrad-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/TRKH_5CLASS_GSFL_SHARED_FEATURE_READINESS_PROTOCOL_20260715.md"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\gsfl_arxiv_2004.01817.pdf"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\GSFL-Net"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_gsfl_shared_feature_readiness_20260715"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "train_cache": Path(args.train_cache).resolve(),
        "cache_manifest": Path(args.cache_manifest).resolve(),
        "cagrad_summary": Path(args.cagrad_summary).resolve(),
        "cagrad_predictions": Path(args.cagrad_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "official_root": official,
        "official_fc": official / "CompactBilinear_GSFL_fc.py",
        "official_all": official / "CompactBilinear_GSFL_all.py",
    }


def _official_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().casefold()


def _official_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if str(args.device) != "cuda":
        raise ValueError("The formal GSFL recipe locks device=cuda.")
    paths = _source_paths(args)
    hashes = {
        "train_cache": _verify_sha256(
            paths["train_cache"], LOCKED_TRAIN_CACHE_SHA256, "train embedding cache"
        ),
        "cache_manifest": _verify_sha256(
            paths["cache_manifest"], LOCKED_CACHE_MANIFEST_SHA256, "cache manifest"
        ),
        "cagrad_summary": _verify_sha256(
            paths["cagrad_summary"], LOCKED_CAGRAD_SUMMARY_SHA256, "CAGrad summary"
        ),
        "cagrad_predictions": _verify_sha256(
            paths["cagrad_predictions"],
            LOCKED_CAGRAD_PREDICTIONS_SHA256,
            "CAGrad predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "GSFL protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "GSFL paper"),
        "official_fc": _verify_sha256(
            paths["official_fc"], LOCKED_OFFICIAL_FC_SHA256, "official FC source"
        ),
        "official_all": _verify_sha256(
            paths["official_all"], LOCKED_OFFICIAL_ALL_SHA256, "official all source"
        ),
    }
    commit = _official_commit(paths["official_root"])
    license_files = sorted(
        value.name
        for value in paths["official_root"].iterdir()
        if value.is_file() and value.name.casefold().startswith("license")
    )
    checks = {
        "official_commit_exact": commit == LOCKED_OFFICIAL_COMMIT,
        "official_worktree_clean": _official_clean(paths["official_root"]),
        "official_repository_has_no_license": not license_files,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Official GSFL provenance differs: {failed}")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": hashes,
        "official_commit": commit,
        "official_checks": checks,
        "official_license_files": license_files,
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    cache = _load_cache(paths["train_cache"], split="train")
    comparator = _read_comparator_holdout(paths["cagrad_predictions"])
    cagrad_summary = json.loads(paths["cagrad_summary"].read_text(encoding="utf-8"))

    sample_index = np.asarray(cache["sample_index"], dtype=np.int64)
    holdout_indices = np.asarray(comparator["sample_index"], dtype=np.int64)
    fit_indices = np.setdiff1d(sample_index, holdout_indices, assume_unique=True)
    cache_paths = np.asarray(
        [str(Path(str(value)).resolve()) for value in cache["paths"]], dtype=object
    )
    fit_sources = set(np.asarray(cache["source_stems"])[fit_indices].tolist())
    holdout_sources = set(np.asarray(cache["source_stems"])[holdout_indices].tolist())
    cache_raw = np.asarray(cache["probabilities"])[holdout_indices]
    canonical_raw = np.asarray(comparator["raw"])
    raw_argmax_mismatches = int(
        np.sum(cache_raw.argmax(axis=1) != canonical_raw.argmax(axis=1))
    )
    raw_probability_difference = float(np.max(np.abs(cache_raw - canonical_raw)))
    cohort = cagrad_summary.get("cohort", {})
    checks = {
        "train_rows_exact": len(sample_index) == EXPECTED_TRAIN_ROWS,
        "sample_index_canonical": np.array_equal(
            sample_index, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)
        ),
        "feature_dim_exact": tuple(np.asarray(cache["embeddings"]).shape)
        == (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURE_DIM),
        "source_groups_exact": len(set(cache["source_stems"].tolist()))
        == EXPECTED_SOURCE_GROUPS,
        "fit_rows_exact": len(fit_indices) == EXPECTED_FIT_ROWS,
        "holdout_rows_exact": len(holdout_indices) == EXPECTED_HOLDOUT_ROWS,
        "fit_index_hash_exact": _ordered_index_sha256(fit_indices.tolist())
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_hash_exact": _ordered_index_sha256(holdout_indices.tolist())
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "source_groups_disjoint": not fit_sources.intersection(holdout_sources),
        "holdout_labels_exact": np.array_equal(
            np.asarray(cache["labels"])[holdout_indices], comparator["labels"]
        ),
        "holdout_paths_exact": np.array_equal(cache_paths[holdout_indices], comparator["paths"]),
        "holdout_sources_exact": np.array_equal(
            np.asarray(cache["source_stems"])[holdout_indices], comparator["source_stems"]
        ),
        "all_cache_paths_train": all(
            Path(str(value)).parent.name.casefold() == "train" for value in cache_paths
        ),
        "cache_raw_argmax_reconciliation_exact": raw_argmax_mismatches == 1,
        "cache_raw_probability_reconciliation_lte_0p04": raw_probability_difference
        <= 0.04,
        "cagrad_fit_hash_exact": cohort.get("fit_index_sha256")
        == EXPECTED_FIT_INDEX_SHA256,
        "cagrad_holdout_hash_exact": cohort.get("holdout_index_sha256")
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "cagrad_validation_not_used": cagrad_summary.get("validation_predictions_used")
        is False,
        "cagrad_test_not_used": cagrad_summary.get("test_data_used") is False,
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(f"Locked GSFL cohort contract differs: {failed}")
    return provenance, cache, comparator, {
        "checks": checks,
        "fit_indices": fit_indices,
        "holdout_indices": holdout_indices,
        "fit_rows": int(len(fit_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "fit_source_groups": int(len(fit_sources)),
        "holdout_source_groups": int(len(holdout_sources)),
        "source_overlap": [],
        "fit_index_sha256": EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_sha256": EXPECTED_HOLDOUT_INDEX_SHA256,
        "cache_raw_argmax_mismatches": raw_argmax_mismatches,
        "maximum_cache_raw_probability_difference": raw_probability_difference,
    }


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def l2_normalize_embeddings(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embedding matrix must be finite and two-dimensional.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if bool(np.any(norms <= 1e-12)):
        raise ValueError("Embedding matrix contains a zero-norm row.")
    return (values / norms).astype(np.float32)


def feature_expression_loss(
    outputs: Mapping[str, Tensor],
    features: Tensor,
    labels: Tensor,
    class_centers: Tensor,
    shared_center: Tensor,
) -> tuple[Tensor, Dict[str, Tensor]]:
    logits = outputs["logits"]
    discriminative = outputs["discriminative"]
    shared = outputs["shared"]
    reconstruction = outputs["reconstruction"]
    ce = F.cross_entropy(logits, labels)
    reconstruction_loss = F.mse_loss(reconstruction, features)
    class_center_loss = torch.mean(
        torch.square(discriminative - class_centers[labels]), dim=1
    ).sum()
    shared_center_loss = torch.mean(
        torch.square(shared - shared_center.unsqueeze(0)), dim=1
    ).sum()
    total = (
        ce
        + RECONSTRUCTION_WEIGHT * reconstruction_loss
        + CLASS_CENTER_WEIGHT * class_center_loss
        + SHARED_CENTER_WEIGHT * shared_center_loss
    )
    return total, {
        "total": total,
        "ce": ce,
        "reconstruction": reconstruction_loss,
        "class_center": class_center_loss,
        "shared_center": shared_center_loss,
    }


def update_feature_centers(
    class_centers: Tensor,
    total_class_centers: Tensor,
    discriminative: Tensor,
    labels: Tensor,
    *,
    update_rate: float = CENTER_UPDATE_RATE,
) -> Tensor:
    with torch.no_grad():
        for class_index in range(int(class_centers.shape[0])):
            mask = labels == class_index
            count = int(mask.sum().item())
            if count == 0:
                continue
            values = discriminative[mask].detach()
            for centers in (class_centers, total_class_centers):
                delta = (centers[class_index].unsqueeze(0) - values).sum(dim=0)
                delta = delta / float(1 + count)
                centers[class_index].sub_(float(update_rate) * delta)
        return total_class_centers.mean(dim=0)


def _gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        squared += float(torch.sum(torch.square(parameter.grad.detach().float())).item())
    return math.sqrt(squared)


def train_matched_adapters(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    *,
    device: torch.device,
) -> tuple[GSFLAdapter, GSFLCEControl, list[Dict[str, float]], Dict[str, object]]:
    features = np.asarray(fit_features, dtype=np.float32)
    labels = np.asarray(fit_labels, dtype=np.int64)
    _seed_torch(SEED)
    candidate = GSFLAdapter().to(device)
    control = GSFLCEControl().to(device)
    control.discriminative_encoder.load_state_dict(
        copy.deepcopy(candidate.discriminative_encoder.state_dict()), strict=True
    )
    control.classifier.load_state_dict(copy.deepcopy(candidate.classifier.state_dict()), strict=True)
    initial_candidate_discriminative_sha = _module_sha256(candidate.discriminative_encoder)
    initial_control_discriminative_sha = _module_sha256(control.discriminative_encoder)
    initial_candidate_classifier_sha = _module_sha256(candidate.classifier)
    initial_control_classifier_sha = _module_sha256(control.classifier)

    candidate.train()
    control.train()
    paired_probe = torch.from_numpy(features[: min(BATCH_SIZE, len(features))]).to(device)
    _seed_torch(SEED + 999999)
    with torch.no_grad():
        control_probe = control(paired_probe)
    _seed_torch(SEED + 999999)
    with torch.no_grad():
        candidate_probe = candidate(paired_probe)
    initial_paired_dropout_max_error = float(
        max(
            torch.max(
                torch.abs(
                    control_probe["discriminative"]
                    - candidate_probe["discriminative"]
                )
            ).item(),
            torch.max(torch.abs(control_probe["logits"] - candidate_probe["logits"])).item(),
        )
    )

    candidate_optimizer = torch.optim.Adam(
        candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    control_optimizer = torch.optim.Adam(
        control.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    class_centers = torch.zeros(CLASS_COUNT, HIDDEN_DIM, device=device)
    total_class_centers = torch.zeros_like(class_centers)
    shared_center = torch.zeros(HIDDEN_DIM, device=device)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    curve = []
    all_finite = True
    minimum_candidate_gradient_norm = float("inf")
    minimum_control_gradient_norm = float("inf")
    batch_index_sha = hashlib.sha256()
    start = time.time()

    for epoch_index in range(EPOCHS):
        candidate.train()
        control.train()
        permutation = np.random.default_rng(SEED + epoch_index).permutation(len(features))
        batch_index_sha.update(permutation.astype(np.int64).tobytes())
        totals: Dict[str, list[float]] = {
            "candidate_total": [],
            "candidate_ce": [],
            "candidate_reconstruction": [],
            "candidate_class_center": [],
            "candidate_shared_center": [],
            "control_ce": [],
        }
        for batch_number, start_index in enumerate(range(0, len(permutation), BATCH_SIZE)):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            batch_features = x[indices].to(device=device, dtype=torch.float32)
            batch_labels = y[indices].to(device=device, dtype=torch.int64)
            paired_seed = SEED + epoch_index * 10000 + batch_number

            _seed_torch(paired_seed)
            control_optimizer.zero_grad(set_to_none=True)
            control_outputs = control(batch_features)
            control_loss = F.cross_entropy(control_outputs["logits"], batch_labels)
            control_loss.backward()
            control_gradient_norm = _gradient_norm(control)
            control_optimizer.step()

            _seed_torch(paired_seed)
            candidate_optimizer.zero_grad(set_to_none=True)
            candidate_outputs = candidate(batch_features)
            candidate_loss, components = feature_expression_loss(
                candidate_outputs,
                batch_features,
                batch_labels,
                class_centers,
                shared_center,
            )
            candidate_loss.backward()
            candidate_gradient_norm = _gradient_norm(candidate)
            candidate_optimizer.step()
            shared_center = update_feature_centers(
                class_centers,
                total_class_centers,
                candidate_outputs["discriminative"],
                batch_labels,
            )

            values = [
                float(candidate_loss.detach().item()),
                float(control_loss.detach().item()),
                float(candidate_gradient_norm),
                float(control_gradient_norm),
                *[float(value.detach().item()) for value in components.values()],
            ]
            if not all(math.isfinite(value) for value in values):
                all_finite = False
                raise FloatingPointError("GSFL matched training became non-finite.")
            minimum_candidate_gradient_norm = min(
                minimum_candidate_gradient_norm, candidate_gradient_norm
            )
            minimum_control_gradient_norm = min(
                minimum_control_gradient_norm, control_gradient_norm
            )
            totals["candidate_total"].append(values[0])
            totals["control_ce"].append(values[1])
            totals["candidate_ce"].append(float(components["ce"].detach().item()))
            totals["candidate_reconstruction"].append(
                float(components["reconstruction"].detach().item())
            )
            totals["candidate_class_center"].append(
                float(components["class_center"].detach().item())
            )
            totals["candidate_shared_center"].append(
                float(components["shared_center"].detach().item())
            )
        row: Dict[str, float] = {"epoch": float(epoch_index + 1)}
        row.update({name: float(np.mean(value)) for name, value in totals.items()})
        curve.append(row)

    telemetry = {
        "all_finite": all_finite,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": int(math.ceil(len(features) / BATCH_SIZE)),
        "batch_index_sha256": batch_index_sha.hexdigest(),
        "minimum_candidate_gradient_norm": minimum_candidate_gradient_norm,
        "minimum_control_gradient_norm": minimum_control_gradient_norm,
        "initial_discriminative_state_identical": (
            initial_candidate_discriminative_sha == initial_control_discriminative_sha
        ),
        "initial_classifier_state_identical": (
            initial_candidate_classifier_sha == initial_control_classifier_sha
        ),
        "initial_paired_dropout_max_error": initial_paired_dropout_max_error,
        "initial_candidate_discriminative_sha256": initial_candidate_discriminative_sha,
        "initial_control_discriminative_sha256": initial_control_discriminative_sha,
        "initial_candidate_classifier_sha256": initial_candidate_classifier_sha,
        "initial_control_classifier_sha256": initial_control_classifier_sha,
        "final_candidate_discriminative_sha256": _module_sha256(
            candidate.discriminative_encoder
        ),
        "final_control_discriminative_sha256": _module_sha256(
            control.discriminative_encoder
        ),
        "class_center_norms": torch.linalg.vector_norm(class_centers, dim=1)
        .detach()
        .cpu()
        .tolist(),
        "shared_center_norm": float(torch.linalg.vector_norm(shared_center).item()),
        "seconds": float(time.time() - start),
        "first_epoch": dict(curve[0]),
        "last_epoch": dict(curve[-1]),
    }
    return candidate, control, curve, telemetry


def evaluate_adapters(
    candidate: GSFLAdapter,
    control: GSFLCEControl,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    candidate.eval()
    control.eval()
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    candidate_probabilities = []
    control_probabilities = []
    shared_discriminative_cosines = []
    with torch.no_grad():
        for start_index in range(0, len(values), int(batch_size)):
            batch = values[start_index : start_index + int(batch_size)].to(device)
            candidate_outputs = candidate(batch)
            control_outputs = control(batch)
            candidate_probabilities.append(
                torch.softmax(candidate_outputs["logits"], dim=1).cpu().numpy()
            )
            control_probabilities.append(
                torch.softmax(control_outputs["logits"], dim=1).cpu().numpy()
            )
            shared_discriminative_cosines.append(
                F.cosine_similarity(
                    candidate_outputs["shared"],
                    candidate_outputs["discriminative"],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
    candidate_array = np.concatenate(candidate_probabilities).astype(np.float64)
    control_array = np.concatenate(control_probabilities).astype(np.float64)
    cosines = np.concatenate(shared_discriminative_cosines)
    return candidate_array, control_array, {
        "shared_discriminative_cosine_mean": float(np.mean(cosines)),
        "shared_discriminative_cosine_std": float(np.std(cosines)),
        "candidate_probability_sum_max_error": float(
            np.max(np.abs(candidate_array.sum(axis=1) - 1.0))
        ),
        "control_probability_sum_max_error": float(
            np.max(np.abs(control_array.sum(axis=1) - 1.0))
        ),
    }


def assess_gsfl_readiness(
    candidate: Mapping[str, object],
    control_metrics: Mapping[str, object],
    *,
    structural_checks: Mapping[str, bool],
) -> Dict[str, object]:
    delta = candidate["delta_vs_raw"]
    transitions = candidate["transitions_vs_raw"]
    direction = candidate["direction_vs_raw"]
    restricted = candidate["restricted_focus_false_positives"]
    candidate_metrics = candidate["candidate"]
    candidate_focus = candidate_metrics["per_class"][FOCUS_CLASS]
    control_focus = control_metrics["per_class"][FOCUS_CLASS]
    checks = {
        "macro_f1_not_below_raw": float(delta["macro_f1"]) >= 0.0,
        "macro_f1_not_below_ce_control": float(candidate_metrics["macro_f1"])
        >= float(control_metrics["macro_f1"]),
        "class1_f1_gain_vs_raw_gte_0p005": float(delta["class1_f1"]) >= 0.005,
        "class1_f1_gain_vs_control_gte_0p005": float(candidate_focus["f1"])
        >= float(control_focus["f1"]) + 0.005,
        "class1_precision_gain_gte_0p010": float(delta["class1_precision"])
        >= 0.010,
        "class1_recall_delta_gte_minus_0p005": float(delta["class1_recall"])
        >= -0.005,
        "class1_f1_within_0p002_margin_agem": float(
            candidate["class1_f1_delta_vs_margin_agem"]
        )
        >= -0.002,
        "zero_raw_class1_tp_broken": int(transitions["focus_true_positive_broken"])
        == 0,
        "restricted_focus_fp_reduction_gte_4": int(restricted["reduction"]) >= 4,
        "corrections_gt_harms": int(transitions["corrections"])
        > int(transitions["harms"]),
        "direction_auc_gte_0p60": direction["auc_fn_positive"] is not None
        and float(direction["auc_fn_positive"]) >= 0.60,
    }
    all_checks = {**{key: bool(value) for key, value in structural_checks.items()}, **checks}
    failed = [name for name, passed in all_checks.items() if not bool(passed)]
    return {
        "structural_checks": dict(structural_checks),
        "behavior_checks": checks,
        "failed_checks": failed,
        "shared_trainer_authorized": not failed,
    }


def _prediction_rows(
    cache: Mapping[str, np.ndarray],
    comparator: Mapping[str, np.ndarray],
    control: np.ndarray,
    candidate: np.ndarray,
) -> list[Dict[str, object]]:
    indices = np.asarray(comparator["sample_index"], dtype=np.int64)
    variants = {
        "raw": np.asarray(comparator["raw"]),
        "margin_agem": np.asarray(comparator["margin_agem"]),
        "ce_control": np.asarray(control),
        "gsfl": np.asarray(candidate),
    }
    rows = []
    for offset, sample_index in enumerate(indices.tolist()):
        row: Dict[str, object] = {
            "sample_index": int(sample_index),
            "source_stem": str(cache["source_stems"][sample_index]),
            "image_path": str(Path(str(cache["paths"][sample_index])).resolve()),
            "target": int(cache["labels"][sample_index]),
        }
        for name, probabilities in variants.items():
            row[f"{name}_prediction"] = int(probabilities[offset].argmax())
            for class_index in range(CLASS_COUNT):
                row[f"{name}_prob_{class_index}"] = float(
                    probabilities[offset, class_index]
                )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, artifacts: Sequence[Path]) -> Dict[str, object]:
    files = []
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        files.append(
            {
                "name": resolved.name,
                "bytes": int(resolved.stat().st_size),
                "sha256": _sha256(resolved),
            }
        )
    payload = {
        "mode": "gsfl_shared_feature_train_only_readiness_evidence",
        "files": files,
        "contains_checkpoint": False,
        "contains_model_binary": False,
        "contains_validation_payload": False,
        "contains_test_payload": False,
        "raw_dataset_touched": False,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    raw = summary["comparators"]["raw"]
    margin = summary["comparators"]["margin_agem"]
    control = summary["control"]["metrics"]
    candidate = summary["candidate"]["evidence"]["candidate"]
    evidence = summary["candidate"]["evidence"]
    lines = [
        "# GSFL Shared-Feature Train-Only Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Shared trainer authorized: `{summary['gate']['shared_trainer_authorized']}`",
        "- Fit/holdout: `7372 / 1843`, source overlap `0`",
        "- Validation/test/model binary: `false / false / false`",
        "",
        "| Variant | Macro F1 | Class-1 P | Class-1 R | Class-1 F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Raw | {raw['macro_f1']:.6f} | {raw['per_class'][1]['precision']:.6f} | {raw['per_class'][1]['recall']:.6f} | {raw['per_class'][1]['f1']:.6f} |",
        f"| Margin A-GEM | {margin['macro_f1']:.6f} | {margin['per_class'][1]['precision']:.6f} | {margin['per_class'][1]['recall']:.6f} | {margin['per_class'][1]['f1']:.6f} |",
        f"| CE adapter | {control['macro_f1']:.6f} | {control['per_class'][1]['precision']:.6f} | {control['per_class'][1]['recall']:.6f} | {control['per_class'][1]['f1']:.6f} |",
        f"| GSFL | {candidate['macro_f1']:.6f} | {candidate['per_class'][1]['precision']:.6f} | {candidate['per_class'][1]['recall']:.6f} | {candidate['per_class'][1]['f1']:.6f} |",
        "",
        f"- GSFL restricted FP reduction: `{evidence['restricted_focus_false_positives']['reduction']}`",
        f"- GSFL class-1 TP broken: `{evidence['transitions_vs_raw']['focus_true_positive_broken']}`",
        f"- Direction AUROC: `{evidence['direction_vs_raw']['auc_fn_positive']}`",
        f"- Failed checks: `{', '.join(summary['gate']['failed_checks']) or 'none'}`",
        "",
        "The formal run used only yolo_f/train. Failure does not evaluate the original pretrained 150+200-epoch GSFL recipe.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("The formal GSFL A0 requires CUDA.")
    provenance, cache, comparator, contract = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "provenance": provenance,
            "cohort": {
                key: value
                for key, value in contract.items()
                if key not in {"fit_indices", "holdout_indices"}
            },
            "validation_predictions_used": False,
            "test_data_used": False,
        }

    output_path = Path(args.output_dir).resolve()
    raw_root = Path(r"D:\DataAI\AIEx\newdataset").resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("GSFL output cannot be written under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    normalized = l2_normalize_embeddings(cache["embeddings"])
    fit_indices = np.asarray(contract["fit_indices"], dtype=np.int64)
    holdout_indices = np.asarray(contract["holdout_indices"], dtype=np.int64)
    labels = np.asarray(cache["labels"], dtype=np.int64)

    candidate, control, curve, training = train_matched_adapters(
        normalized[fit_indices],
        labels[fit_indices],
        device=device,
    )
    candidate_probabilities, control_probabilities, evaluation = evaluate_adapters(
        candidate,
        control,
        normalized[holdout_indices],
        device=device,
    )
    control_metrics = _classification_metrics(comparator["labels"], control_probabilities)
    candidate_result = candidate_evidence(
        comparator["labels"],
        comparator["raw"],
        comparator["margin_agem"],
        candidate_probabilities,
    )
    first = training["first_epoch"]
    last = training["last_epoch"]
    structural = {
        "locked_sources_verified": True,
        "cohort_contract_exact": True,
        "source_groups_disjoint": True,
        "validation_not_used": True,
        "test_not_used": True,
        "no_hyperparameter_sweep": True,
        "tracked_worktree_clean": _tracked_worktree_clean(Path.cwd()),
        "initial_discriminative_state_identical": bool(
            training["initial_discriminative_state_identical"]
        ),
        "initial_classifier_state_identical": bool(
            training["initial_classifier_state_identical"]
        ),
        "initial_paired_dropout_exact": float(
            training["initial_paired_dropout_max_error"]
        )
        == 0.0,
        "all_training_finite": bool(training["all_finite"]),
        "candidate_gradients_nonzero": float(training["minimum_candidate_gradient_norm"])
        > 0.0,
        "control_gradients_nonzero": float(training["minimum_control_gradient_norm"])
        > 0.0,
        "candidate_objective_decreased": float(last["candidate_total"])
        < float(first["candidate_total"]),
        "control_objective_decreased": float(last["control_ce"])
        < float(first["control_ce"]),
        "reconstruction_decreased_gte_5pct": float(last["candidate_reconstruction"])
        <= 0.95 * float(first["candidate_reconstruction"]),
        "all_class_centers_updated": all(
            float(value) > 0.0 for value in training["class_center_norms"]
        ),
        "shared_center_updated": float(training["shared_center_norm"]) > 0.0,
        "candidate_differs_from_control": (
            training["final_candidate_discriminative_sha256"]
            != training["final_control_discriminative_sha256"]
        ),
        "shared_discriminative_not_identical": abs(
            float(evaluation["shared_discriminative_cosine_mean"])
        )
        < 0.999,
        "candidate_probabilities_normalized": float(
            evaluation["candidate_probability_sum_max_error"]
        )
        <= 2e-7,
        "control_probabilities_normalized": float(
            evaluation["control_probability_sum_max_error"]
        )
        <= 2e-7,
    }
    gate = assess_gsfl_readiness(
        candidate_result,
        control_metrics,
        structural_checks=structural,
    )

    predictions_path = output_dir / "predictions_holdout.csv"
    curve_path = output_dir / "training_curve.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_csv(
        predictions_path,
        _prediction_rows(
            cache,
            comparator,
            control_probabilities,
            candidate_probabilities,
        ),
    )
    _write_csv(curve_path, curve)
    summary: Dict[str, object] = {
        "status": "authorized" if gate["shared_trainer_authorized"] else "rejected",
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": {
                "reconstruction": RECONSTRUCTION_WEIGHT,
                "class_center": CLASS_CENTER_WEIGHT,
                "shared_center": SHARED_CENTER_WEIGHT,
            },
            "center_update_rate": CENTER_UPDATE_RATE,
            "class_groups": [list(range(CLASS_COUNT))],
            "selection_scope": "final_epoch30_train_holdout_once_no_sweep",
        },
        "provenance": {**provenance, "git_commit": _git_commit(Path.cwd())},
        "cohort": {
            key: value
            for key, value in contract.items()
            if key not in {"fit_indices", "holdout_indices"}
        },
        "training": training,
        "evaluation": evaluation,
        "comparators": {
            "raw": _classification_metrics(comparator["labels"], comparator["raw"]),
            "margin_agem": _classification_metrics(
                comparator["labels"], comparator["margin_agem"]
            ),
        },
        "control": {"metrics": control_metrics},
        "candidate": {"evidence": candidate_result},
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
        "binary_model_artifacts_written": False,
        "raw_dataset_touched": False,
        "artifacts": {
            "summary": str(summary_path),
            "report": str(report_path),
            "predictions": str(predictions_path),
            "training_curve": str(curve_path),
            "manifest": str(manifest_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    _write_manifest(
        manifest_path,
        [summary_path, report_path, predictions_path, curve_path],
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(json.dumps(result["gate"], indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
