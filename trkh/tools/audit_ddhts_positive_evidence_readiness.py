from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor, nn
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.inference.inference import load_checkpoint
from trkh.models.model import create_model, load_model_state
from trkh.tools.audit_class1_boundary_cagrad_readiness import (
    LOCKED_CIDT_PREDICTIONS_SHA256,
    LOCKED_CIDT_SUMMARY_SHA256,
    LOCKED_DATA_SHA256,
    LOCKED_KEEPER_SHA256,
    LOCKED_LAUNCHER_ARGS_SHA256,
)
from trkh.tools.audit_class1_boundary_vrex_readiness import _make_condition_loader
from trkh.tools.audit_class1_reference_agem_readiness import (
    EXPECTED_HARD_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_REFERENCE_ROWS,
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    _cohort_serializable,
    _locked_cohort_summary,
)
from trkh.tools.audit_class_axis_multimodal_readiness import (
    EXPECTED_FIT_INDEX_SHA256,
    EXPECTED_FIT_ROWS,
    EXPECTED_HOLDOUT_INDEX_SHA256,
    EXPECTED_SOURCE_GROUPS,
    EXPECTED_TRAIN_ROWS,
    LOCKED_CAGRAD_PREDICTIONS_SHA256,
    LOCKED_CAGRAD_SUMMARY_SHA256,
    _tracked_worktree_clean,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _build_eval_transform,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.audit_two_stage_reedl_readiness import (
    _classification_metrics,
    _transition_stats,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _git_commit,
    _prepare_output_dir,
    _verify_sha256,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


SEED = 20260716
SVM_C = 1.0
PRESERVED_CHANNELS = 32
COMMON_GRID_SIZE = 32
DESCRIPTOR_DIM = 40
THRESHOLD_EPSILON = 1e-6
CONDITIONS = ("clean", *(value[0] for value in LIGHTING_CONDITIONS))
MAX_PEAK_VRAM_GIB = 3.25

LOCKED_PROTOCOL_SHA256 = (
    "bbaec0ff74851ca5ae117520b76887062a9a73aec3d4144254584a5ada9e1970"
)
LOCKED_PAPER_SHA256 = (
    "d3fa72ef3345f4ef23403eab904855df15ebe772146b6323fe502f2d9dff9a7d"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only DDHTS positive-evidence information gate. "
            "Validation, test, trainer integration, and binary model output are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
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
        default=Path(
            "docs/TRKH_5CLASS_DDHTS_POSITIVE_EVIDENCE_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\ddhts_net_frontiers_2026.pdf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_ddhts_positive_evidence_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--svm-c", type=float, default=SVM_C)
    parser.add_argument("--preserved-channels", type=int, default=PRESERVED_CHANNELS)
    parser.add_argument("--grid-size", type=int, default=COMMON_GRID_SIZE)
    return parser.parse_args(argv)


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "cagrad_summary": Path(args.cagrad_summary).resolve(),
        "cagrad_predictions": Path(args.cagrad_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
    }


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == 16
        and int(args.num_workers) == 4
        and int(args.seed) == SEED
        and int(args.fold) == 0
        and math.isclose(float(args.svm_c), SVM_C, rel_tol=0.0, abs_tol=0.0)
        and int(args.preserved_channels) == PRESERVED_CHANNELS
        and int(args.grid_size) == COMMON_GRID_SIZE
    )


def _verify_sources(args: argparse.Namespace) -> Dict[str, object]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted DDHTS protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "cagrad_summary": _verify_sha256(
            paths["cagrad_summary"],
            LOCKED_CAGRAD_SUMMARY_SHA256,
            "CAGrad summary",
        ),
        "cagrad_predictions": _verify_sha256(
            paths["cagrad_predictions"],
            LOCKED_CAGRAD_PREDICTIONS_SHA256,
            "CAGrad predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "DDHTS protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "DDHTS paper"),
    }
    cagrad = json.loads(paths["cagrad_summary"].read_text(encoding="utf-8"))
    cidt = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cagrad.get("validation_predictions_used", True)) or bool(
        cagrad.get("test_data_used", True)
    ):
        raise ValueError("CAGrad comparator provenance used validation or test.")
    if bool(cidt.get("validation_predictions_used", True)) or bool(
        cidt.get("test_data_used", True)
    ):
        raise ValueError("CIDT cohort provenance used validation or test.")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "hashes": hashes,
        "git_commit": _git_commit(Path.cwd()),
        "tracked_worktree_clean": _tracked_worktree_clean(Path.cwd()),
        "validation_predictions_used": False,
        "test_data_used": False,
    }


def _iuwt_contrast_views(rgb: Tensor, *, epsilon: float = 1e-6) -> list[Tensor]:
    if rgb.ndim != 4 or int(rgb.size(1)) != 3:
        raise ValueError("DDHTS RGB input must have shape [B,3,H,W].")
    luminance = (
        0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
    ).clamp(0.0, 1.0)
    base = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=rgb.device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3) / 16.0
    current = luminance.float()
    outputs: list[Tensor] = []
    for dilation in (1, 2, 4):
        padded = F.pad(
            current,
            (dilation, dilation, dilation, dilation),
            mode="reflect",
        )
        low = F.conv2d(padded, base, dilation=dilation)
        high = current - low
        ratio = high / low.clamp_min(float(epsilon))
        beta = ratio.mean(dim=(2, 3), keepdim=True)
        alpha = ratio.std(dim=(2, 3), keepdim=True, unbiased=False)
        outputs.append(torch.sigmoid((ratio - beta) / (alpha + float(epsilon))))
        current = low
    return outputs


def _reduce_feature_map(
    feature: Tensor,
    *,
    preserved_channels: int,
    grid_size: int,
) -> Tensor:
    if feature.ndim != 4:
        raise ValueError("DDHTS feature map must have shape [B,C,H,W].")
    channels = int(feature.size(1))
    kept = int(preserved_channels)
    if kept <= 0 or channels < kept or channels % kept != 0:
        raise ValueError(
            f"Feature channels {channels} cannot be average-reduced to {kept}."
        )
    grouped = feature.reshape(
        int(feature.size(0)),
        kept,
        channels // kept,
        int(feature.size(2)),
        int(feature.size(3)),
    ).mean(dim=2)
    return F.adaptive_avg_pool2d(
        grouped.float(), output_size=(int(grid_size), int(grid_size))
    )


def _normalized_code_histogram(codes: Tensor, *, bins: int) -> Tensor:
    if codes.ndim != 4:
        raise ValueError("DDHTS code tensor must have shape [B,C,H,W].")
    values = [
        (codes == index).float().mean(dim=(1, 2, 3)) for index in range(int(bins))
    ]
    return torch.stack(values, dim=1)


def _stem_feature_pyramid(
    stem: nn.Module,
    images: Tensor,
    *,
    preserved_channels: int,
    grid_size: int,
    amp: bool,
) -> list[Tensor]:
    blocks = getattr(stem, "blocks", None)
    if not isinstance(blocks, nn.Sequential) or len(blocks) != 3:
        raise ValueError("Locked DDHTS gate requires the keeper's three-block HybridConvStem.")
    outputs: list[Tensor] = []
    value = images
    context = torch.autocast(
        device_type=images.device.type,
        dtype=torch.float16,
        enabled=bool(amp and images.device.type == "cuda"),
    )
    with context:
        for block in blocks:
            value = block(value)
            outputs.append(
                _reduce_feature_map(
                    value,
                    preserved_channels=int(preserved_channels),
                    grid_size=int(grid_size),
                )
            )
    return outputs


def build_ddhts_descriptor(
    stem: nn.Module,
    normalized_images: Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    preserved_channels: int = PRESERVED_CHANNELS,
    grid_size: int = COMMON_GRID_SIZE,
    amp: bool = True,
) -> tuple[Tensor, Dict[str, object], list[Tensor]]:
    mean_tensor = torch.as_tensor(
        mean, device=normalized_images.device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(
        std, device=normalized_images.device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    rgb = (normalized_images.float() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    reconstructed = (rgb - mean_tensor) / std_tensor
    reconstruction_error = float(
        (reconstructed - normalized_images.float()).abs().amax().item()
    )
    contrast = _iuwt_contrast_views(rgb)
    normalized_views = [reconstructed]
    normalized_views.extend(
        (value.repeat(1, 3, 1, 1) - mean_tensor) / std_tensor for value in contrast
    )
    pyramids = [
        _stem_feature_pyramid(
            stem,
            value,
            preserved_channels=int(preserved_channels),
            grid_size=int(grid_size),
            amp=bool(amp),
        )
        for value in normalized_views
    ]

    intra_segments: list[Tensor] = []
    for pyramid in pyramids:
        code = torch.zeros_like(pyramid[0], dtype=torch.long)
        for layer_index in range(2):
            code = code + (
                (pyramid[layer_index + 1] - pyramid[layer_index]) >= 0.0
            ).long() * (2**layer_index)
        intra_segments.append(_normalized_code_histogram(code, bins=4))

    inter_segments: list[Tensor] = []
    for layer_index in range(3):
        code = torch.zeros_like(pyramids[0][layer_index], dtype=torch.long)
        for view_index in range(1, 4):
            code = code + (
                (pyramids[view_index][layer_index] - pyramids[0][layer_index])
                >= 0.0
            ).long() * (2 ** (view_index - 1))
        inter_segments.append(_normalized_code_histogram(code, bins=8))

    descriptor = torch.cat((*inter_segments, *intra_segments), dim=1).float()
    segment_sums = torch.stack(
        [segment.sum(dim=1) for segment in (*inter_segments, *intra_segments)],
        dim=1,
    )
    diagnostics = {
        "descriptor_dim": int(descriptor.size(1)),
        "finite": bool(torch.isfinite(descriptor).all().item()),
        "input_reconstruction_max_abs_error": reconstruction_error,
        "segment_sum_max_abs_error": float((segment_sums - 1.0).abs().amax().item()),
        "segment_count": int(segment_sums.size(1)),
        "intra_dim": int(sum(int(value.size(1)) for value in intra_segments)),
        "inter_dim": int(sum(int(value.size(1)) for value in inter_segments)),
    }
    return descriptor, diagnostics, [rgb, *contrast]


def _extract_condition(
    *,
    stem: nn.Module,
    loader,
    selected_indices: Sequence[int],
    condition: str,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    preserved_channels: int,
    grid_size: int,
) -> Dict[str, object]:
    descriptor_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    index_batches: list[np.ndarray] = []
    maximum_segment_error = 0.0
    maximum_reconstruction_error = 0.0
    finite = True
    started = time.perf_counter()
    stem.eval()
    with torch.inference_mode():
        for images, labels, metadata in tqdm(
            loader, desc=f"ddhts-{condition}", dynamic_ncols=True
        ):
            if not isinstance(metadata, Mapping) or not torch.is_tensor(
                metadata.get("sample_index")
            ):
                raise ValueError("DDHTS loader requires sample_index metadata.")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            descriptor, diagnostics, _ = build_ddhts_descriptor(
                stem,
                images,
                mean=mean,
                std=std,
                preserved_channels=int(preserved_channels),
                grid_size=int(grid_size),
                amp=True,
            )
            descriptor_batches.append(descriptor.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            index_batches.append(
                metadata["sample_index"].detach().cpu().numpy().reshape(-1)
            )
            finite = finite and bool(diagnostics["finite"])
            maximum_segment_error = max(
                maximum_segment_error,
                float(diagnostics["segment_sum_max_abs_error"]),
            )
            maximum_reconstruction_error = max(
                maximum_reconstruction_error,
                float(diagnostics["input_reconstruction_max_abs_error"]),
            )
    descriptors = np.concatenate(descriptor_batches).astype(np.float32, copy=False)
    labels_array = np.concatenate(label_batches).astype(np.int64, copy=False)
    indices = np.concatenate(index_batches).astype(np.int64, copy=False)
    expected = np.asarray(selected_indices, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError(f"{condition} extraction order differs from the locked indices.")
    if descriptors.shape != (len(expected), DESCRIPTOR_DIM):
        raise ValueError(f"Unexpected DDHTS descriptor shape: {descriptors.shape}")
    return {
        "condition": str(condition),
        "sample_index": indices,
        "labels": labels_array,
        "descriptors": descriptors,
        "finite": bool(finite and np.isfinite(descriptors).all()),
        "segment_sum_max_abs_error": float(maximum_segment_error),
        "input_reconstruction_max_abs_error": float(maximum_reconstruction_error),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _read_comparators(
    path: Path,
    *,
    holdout_indices: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    by_condition: Dict[str, Dict[int, Dict[str, object]]] = {
        condition: {} for condition in CONDITIONS
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "condition",
            "sample_index",
            "source_stem",
            "image_path",
            "target",
            *{f"raw_prob_{index}" for index in range(5)},
            *{f"margin_agem_prob_{index}" for index in range(5)},
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CAGrad comparator CSV is missing: {sorted(missing)}")
        for raw in reader:
            condition = str(raw["condition"]).strip()
            if condition not in by_condition:
                raise ValueError(f"Unexpected comparator condition: {condition}")
            sample_index = int(raw["sample_index"])
            if sample_index in by_condition[condition]:
                raise ValueError("Duplicate comparator condition/sample row.")
            by_condition[condition][sample_index] = {
                "sample_index": sample_index,
                "target": int(raw["target"]),
                "source_stem": str(raw["source_stem"]).strip().casefold(),
                "image_path": str(Path(raw["image_path"]).resolve()),
                "raw": np.asarray(
                    [float(raw[f"raw_prob_{index}"]) for index in range(5)],
                    dtype=np.float32,
                ),
                "margin_agem": np.asarray(
                    [float(raw[f"margin_agem_prob_{index}"]) for index in range(5)],
                    dtype=np.float32,
                ),
            }
    ordered_indices = [int(value) for value in holdout_indices]
    result: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        if set(by_condition[condition]) != set(ordered_indices):
            raise ValueError(f"Comparator {condition} rows differ from holdout.")
        rows = [by_condition[condition][index] for index in ordered_indices]
        raw_probabilities = np.stack([row["raw"] for row in rows])
        margin_probabilities = np.stack([row["margin_agem"] for row in rows])
        if not np.allclose(raw_probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Comparator {condition} raw probabilities are not normalized.")
        result[condition] = {
            "sample_index": np.asarray(ordered_indices, dtype=np.int64),
            "targets": np.asarray([row["target"] for row in rows], dtype=np.int64),
            "source_stems": [str(row["source_stem"]) for row in rows],
            "paths": [str(row["image_path"]) for row in rows],
            "raw": raw_probabilities,
            "margin_agem": margin_probabilities,
        }
    return result


def _fit_positive_evidence_readout(
    descriptors: Mapping[int, np.ndarray],
    rows: Sequence[CleanTrainRow],
    cohorts: Mapping[str, object],
    *,
    seed: int,
    svm_c: float,
) -> Dict[str, object]:
    reference_indices = [int(value) for value in cohorts["reference_indices"]]
    hard_indices = [int(value) for value in cohorts["hard_indices"]]
    fit_indices = [*reference_indices, *hard_indices]
    matrix = np.stack([descriptors[index] for index in fit_indices]).astype(np.float64)
    labels = np.asarray(
        [1] * len(reference_indices) + [0] * len(hard_indices), dtype=np.int64
    )
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)

    def train() -> SVC:
        return SVC(
            C=float(svm_c),
            kernel="linear",
            class_weight="balanced",
            probability=False,
            random_state=int(seed),
        ).fit(standardized, labels)

    model = train()
    replay = train()
    scores = np.asarray(model.decision_function(standardized), dtype=np.float64)
    replay_scores = np.asarray(replay.decision_function(standardized), dtype=np.float64)
    score_by_index = {index: float(score) for index, score in zip(fit_indices, scores)}
    fit_true_positive_indices = [
        index
        for index in reference_indices
        if int(rows[index].keeper_prediction) == FOCUS_CLASS
    ]
    if not fit_true_positive_indices:
        raise ValueError("No fit raw class-1 true positives are available for thresholding.")
    threshold = min(score_by_index[index] for index in fit_true_positive_indices) - float(
        THRESHOLD_EPSILON
    )
    fit_tp_broken = sum(
        score_by_index[index] < threshold for index in fit_true_positive_indices
    )
    deterministic = bool(
        np.array_equal(model.support_, replay.support_)
        and np.array_equal(model.dual_coef_, replay.dual_coef_)
        and np.array_equal(model.intercept_, replay.intercept_)
        and np.array_equal(scores, replay_scores)
    )
    singular_values = np.linalg.svd(
        standardized - standardized.mean(axis=0, keepdims=True),
        compute_uv=False,
    )
    energy = singular_values**2
    effective_rank = float((energy.sum() ** 2) / np.square(energy).sum())
    return {
        "scaler": scaler,
        "model": model,
        "threshold": float(threshold),
        "fit_true_positive_rows": int(len(fit_true_positive_indices)),
        "fit_true_positive_broken": int(fit_tp_broken),
        "fit_rows": int(len(fit_indices)),
        "fit_positive_rows": int(labels.sum()),
        "fit_negative_rows": int((labels == 0).sum()),
        "support_vectors": int(model.support_.size),
        "support_vectors_by_class": model.n_support_.astype(int).tolist(),
        "deterministic_replay_exact": deterministic,
        "effective_rank": effective_rank,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def apply_positive_evidence_veto(
    raw_probabilities: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    focus_class: int = FOCUS_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_probabilities, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw.ndim != 2 or raw.shape[0] != scores.size:
        raise ValueError("DDHTS veto probability/score shapes differ.")
    focus = int(focus_class)
    predictions = raw.argmax(axis=1)
    veto = (predictions == focus) & (scores < float(threshold))
    candidate = raw.copy()
    for row_index in np.flatnonzero(veto).tolist():
        alternatives = candidate[row_index].copy()
        alternatives[focus] = -np.inf
        runner_up = int(np.argmax(alternatives))
        candidate[row_index, focus], candidate[row_index, runner_up] = (
            candidate[row_index, runner_up],
            candidate[row_index, focus],
        )
    return candidate, veto


def _focus_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return dict(metrics["per_class"][FOCUS_CLASS])


def _restricted_focus_fp(targets: np.ndarray, probabilities: np.ndarray) -> int:
    predictions = np.asarray(probabilities).argmax(axis=1)
    return int(
        np.sum(
            np.isin(np.asarray(targets, dtype=np.int64), RESTRICTED_NEGATIVE_CLASSES)
            & (predictions == FOCUS_CLASS)
        )
    )


def _comparison(
    targets: np.ndarray,
    raw: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    raw_metrics = _classification_metrics(targets, raw)
    candidate_metrics = _classification_metrics(targets, candidate)
    raw_focus = _focus_metrics(raw_metrics)
    candidate_focus = _focus_metrics(candidate_metrics)
    transitions = _transition_stats(
        targets,
        raw,
        candidate,
        focus_class_index=FOCUS_CLASS,
    )
    raw_restricted = _restricted_focus_fp(targets, raw)
    candidate_restricted = _restricted_focus_fp(targets, candidate)
    transitions["restricted_focus_fp_raw"] = int(raw_restricted)
    transitions["restricted_focus_fp_candidate"] = int(candidate_restricted)
    transitions["restricted_focus_fp_reduction"] = int(
        raw_restricted - candidate_restricted
    )
    return {
        "raw": raw_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "macro_f1": float(candidate_metrics["macro_f1"] - raw_metrics["macro_f1"]),
            "class1_f1": float(candidate_focus["f1"] - raw_focus["f1"]),
            "class1_precision": float(
                candidate_focus["precision"] - raw_focus["precision"]
            ),
            "class1_recall": float(candidate_focus["recall"] - raw_focus["recall"]),
        },
        "transitions": transitions,
    }


def assess_ddhts_gate(
    *,
    structural_checks: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, object]],
    clean_direction_auc: Optional[float],
) -> Dict[str, object]:
    clean = comparisons["clean"]
    clean_delta = clean["delta"]
    clean_transitions = clean["transitions"]
    clean_checks = {
        "clean_breaks_zero_class1_tp": int(
            clean_transitions["focus_true_positive_broken"]
        )
        == 0,
        "clean_class1_recall_exact": abs(float(clean_delta["class1_recall"])) <= 1e-12,
        "clean_restricted_fp_reduction_gte_4": int(
            clean_transitions["restricted_focus_fp_reduction"]
        )
        >= 4,
        "clean_creates_zero_class1_fp": int(
            clean_transitions["focus_false_positive_created"]
        )
        == 0,
        "clean_class1_precision_gain_gte_0p02": float(
            clean_delta["class1_precision"]
        )
        >= 0.02,
        "clean_class1_f1_gain_gte_0p005": float(clean_delta["class1_f1"]) >= 0.005,
        "clean_macro_f1_not_reduced": float(clean_delta["macro_f1"]) >= 0.0,
        "clean_corrections_gte_harms": int(clean_transitions["corrections"])
        >= int(clean_transitions["harms"]),
        "clean_direction_auc_gte_0p65": clean_direction_auc is not None
        and float(clean_direction_auc) >= 0.65,
    }
    shifted_checks: Dict[str, bool] = {}
    reductions = 0
    for condition, _, _ in LIGHTING_CONDITIONS:
        comparison = comparisons[condition]
        transitions = comparison["transitions"]
        delta = comparison["delta"]
        shifted_checks[f"{condition}_breaks_zero_class1_tp"] = int(
            transitions["focus_true_positive_broken"]
        ) == 0
        shifted_checks[f"{condition}_class1_f1_not_reduced"] = float(
            delta["class1_f1"]
        ) >= 0.0
        shifted_checks[f"{condition}_restricted_fp_not_increased"] = int(
            transitions["restricted_focus_fp_reduction"]
        ) >= 0
        reductions += int(transitions["restricted_focus_fp_reduction"] >= 1)
    shifted_checks["restricted_fp_reduced_in_at_least_two_shifts"] = reductions >= 2
    checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **clean_checks,
        **shifted_checks,
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "checks": checks,
        "failed_checks": failed,
        "clean_checks": clean_checks,
        "shifted_checks": shifted_checks,
    }


def _descriptor_rows(payloads: Mapping[str, Mapping[str, object]]) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for condition in CONDITIONS:
        payload = payloads[condition]
        for position, sample_index in enumerate(payload["sample_index"]):
            row: Dict[str, object] = {
                "condition": condition,
                "sample_index": int(sample_index),
                "target": int(payload["labels"][position]),
            }
            for feature_index, value in enumerate(payload["descriptors"][position]):
                row[f"feature_{feature_index:02d}"] = float(value)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(
    *,
    comparators: Mapping[str, Mapping[str, object]],
    payloads: Mapping[str, Mapping[str, object]],
    scores: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    vetoes: Mapping[str, np.ndarray],
    threshold: float,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for condition in CONDITIONS:
        comparator = comparators[condition]
        raw = np.asarray(comparator["raw"])
        candidate = np.asarray(candidates[condition])
        margin = np.asarray(comparator["margin_agem"])
        for position, sample_index in enumerate(comparator["sample_index"]):
            row: Dict[str, object] = {
                "condition": condition,
                "sample_index": int(sample_index),
                "source_stem": comparator["source_stems"][position],
                "image_path": comparator["paths"][position],
                "target": int(comparator["targets"][position]),
                "ddhts_score": float(scores[condition][position]),
                "threshold": float(threshold),
                "veto": bool(vetoes[condition][position]),
                "raw_prediction": int(raw[position].argmax()),
                "candidate_prediction": int(candidate[position].argmax()),
                "margin_agem_prediction": int(margin[position].argmax()),
            }
            for class_index in range(5):
                row[f"raw_prob_{class_index}"] = float(raw[position, class_index])
                row[f"candidate_prob_{class_index}"] = float(
                    candidate[position, class_index]
                )
                row[f"margin_agem_prob_{class_index}"] = float(
                    margin[position, class_index]
                )
            rows.append(row)
    return rows


def _evidence_case_rows(prediction_rows: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for row in prediction_rows:
        if str(row["condition"]) != "clean" or int(row["raw_prediction"]) != FOCUS_CLASS:
            continue
        target = int(row["target"])
        veto = bool(row["veto"])
        if target == FOCUS_CLASS:
            category = "broken_tp" if veto else "retained_tp"
        elif target in RESTRICTED_NEGATIVE_CLASSES:
            category = "removed_restricted_fp" if veto else "retained_restricted_fp"
        else:
            category = "other_false_positive"
        rows.append(
            {
                "category": category,
                "sample_index": int(row["sample_index"]),
                "target": target,
                "score": float(row["ddhts_score"]),
                "threshold": float(row["threshold"]),
                "candidate_prediction": int(row["candidate_prediction"]),
                "image_path": row["image_path"],
            }
        )
    return sorted(rows, key=lambda row: (str(row["category"]), float(row["score"])))


def _write_contact_sheet(
    path: Path,
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    cases: Sequence[Mapping[str, object]],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
) -> int:
    selected: list[Mapping[str, object]] = []
    for category in (
        "removed_restricted_fp",
        "broken_tp",
        "retained_restricted_fp",
        "retained_tp",
    ):
        category_rows = [row for row in cases if row["category"] == category]
        if category == "retained_tp":
            category_rows = sorted(category_rows, key=lambda row: float(row["score"]))
        selected.extend(category_rows[:3])
    if not selected:
        return 0
    indices = [int(row["sample_index"]) for row in selected]
    loader, _ = _make_condition_loader(
        condition="clean",
        brightness=None,
        contrast=None,
        base_dataset=base_dataset,
        transform=transform,
        indices=indices,
        args=args,
        context="ddhts_contact_sheet",
        seed=int(args.seed) + 900,
    )
    images, _, metadata = next(iter(loader))
    observed = metadata["sample_index"].detach().cpu().numpy().reshape(-1).tolist()
    if observed != indices:
        raise ValueError("DDHTS contact-sheet row order differs.")
    mean_tensor = torch.as_tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std).view(1, 3, 1, 1)
    rgb = (images.float() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    contrasts = _iuwt_contrast_views(rgb)
    views = [rgb, *[value.repeat(1, 3, 1, 1) for value in contrasts]]
    tile = 176
    caption = 34
    canvas = Image.new("RGB", (tile * 4, (tile + caption) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("RGB", "D1", "D2", "D3")
    for row_index, case in enumerate(selected):
        y = row_index * (tile + caption)
        for column, view in enumerate(views):
            array = (
                view[row_index]
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )
            image = Image.fromarray(array, mode="RGB").resize(
                (tile, tile), Image.Resampling.BILINEAR
            )
            canvas.paste(image, (column * tile, y))
            draw.text((column * tile + 4, y + 4), labels[column], fill="yellow")
        text = (
            f"{case['category']} idx={case['sample_index']} target={case['target']} "
            f"score={float(case['score']):.4f} threshold={float(case['threshold']):.4f}"
        )
        draw.text((4, y + tile + 8), text, fill="black")
    canvas.save(path, quality=92, optimize=True)
    return len(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    artifacts = []
    forbidden = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".trt"}
    for path in sorted(Path(output_dir).iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden binary model artifact was written: {path}")
        artifacts.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "artifact_count": len(artifacts),
        "total_bytes": int(sum(int(value["size_bytes"]) for value in artifacts)),
        "binary_model_artifacts_written": False,
        "validation_predictions_used": False,
        "test_data_used": False,
        "artifacts": artifacts,
    }
    path = Path(output_dir) / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_sha256": _sha256(path)}


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    clean = summary["comparisons"]["clean"]
    gate = summary["gate"]
    lines = [
        "# DDHTS Positive-Evidence Readiness Result",
        "",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Clean macro F1 delta: `{clean['delta']['macro_f1']:+.6f}`",
        f"- Clean class-1 F1 delta: `{clean['delta']['class1_f1']:+.6f}`",
        f"- Clean class-1 precision delta: `{clean['delta']['class1_precision']:+.6f}`",
        f"- Clean class-1 recall delta: `{clean['delta']['class1_recall']:+.6f}`",
        f"- Clean restricted FP reduction: `{clean['transitions']['restricted_focus_fp_reduction']}`",
        f"- Clean class-1 TP broken: `{clean['transitions']['focus_true_positive_broken']}`",
        f"- Direction AUROC: `{summary['clean_direction_auc']}`",
        "",
        "Only source-disjoint `yolo_f/train` rows were used. Validation and test were not accessed.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance = _verify_sources(args)
    paths = _source_paths(args)
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    if cohorts["fit_index_sha256"] != EXPECTED_FIT_INDEX_SHA256 or cohorts[
        "holdout_index_sha256"
    ] != EXPECTED_HOLDOUT_INDEX_SHA256:
        raise ValueError("DDHTS source split hashes differ from the locked protocol.")
    comparators = _read_comparators(
        paths["cagrad_predictions"], holdout_indices=cohorts["holdout_indices"]
    )

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    if len(class_names) != 5 or not isinstance(model_config, Mapping) or not isinstance(
        model_state, Mapping
    ):
        raise ValueError("Keeper checkpoint config/state/class order is invalid.")
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(paths["data"], class_name_mode="raw", expected_num_classes=5)
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper.")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    sample_paths = [Path(value).resolve() for value in base_dataset.sample_paths()]
    sample_paths_exact = len(sample_paths) == len(rows) and all(
        source.image_path == observed for source, observed in zip(rows, sample_paths)
    )
    train_paths_only = all(
        "train" in {part.casefold() for part in path.parts}
        and "val" not in {part.casefold() for part in path.parts}
        and "test" not in {part.casefold() for part in path.parts}
        for path in sample_paths
    )
    if not sample_paths_exact or not train_paths_only:
        raise ValueError("Dataset paths violate the locked train-only order.")

    preflight = {
        "status": "preflight_passed",
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "dataset_rows": len(base_dataset),
        "dataset_paths_exact": sample_paths_exact,
        "train_paths_only": train_paths_only,
        "comparator_condition_rows": {
            key: int(len(value["sample_index"])) for key, value in comparators.items()
        },
        "validation_predictions_used": False,
        "test_data_used": False,
    }
    if bool(args.preflight_only):
        return preflight
    if not bool(provenance["tracked_worktree_clean"]):
        raise ValueError("Formal DDHTS run requires a clean tracked worktree.")
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Locked DDHTS audit requires CUDA.")
    output_path = Path(args.output_dir).resolve()
    try:
        output_path.relative_to(paths["data"].parent)
        raise ValueError("DDHTS output cannot be inside the raw dataset tree.")
    except ValueError as error:
        if str(error).startswith("DDHTS output"):
            raise
    output_dir = _prepare_output_dir(output_path)

    set_seed(int(args.seed), deterministic=True)
    device = torch.device(str(args.device))
    model = create_model(num_classes=5, model_config=model_config).eval()
    load_model_state(model, dict(model_state), strict=True)
    initial_state_sha256 = _state_sha256(model)
    stem = model.stem.to(device).eval()
    for parameter in stem.parameters():
        parameter.requires_grad_(False)
    mean, std = checkpoint_input_normalization(checkpoint)
    transform = _build_eval_transform(semantics)

    clean_indices = sorted(
        set(int(value) for value in cohorts["reference_indices"])
        | set(int(value) for value in cohorts["hard_indices"])
        | set(int(value) for value in cohorts["holdout_indices"])
    )
    payloads: Dict[str, Dict[str, object]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    conditions = [("clean", None, None), *LIGHTING_CONDITIONS]
    for condition_index, (condition, brightness, contrast) in enumerate(conditions):
        selected = clean_indices if condition == "clean" else cohorts["holdout_indices"]
        loader, loader_summary = _make_condition_loader(
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            base_dataset=base_dataset,
            transform=transform,
            indices=selected,
            args=args,
            context=f"ddhts_{condition}",
            seed=int(args.seed) + 100 + condition_index,
        )
        payload = _extract_condition(
            stem=stem,
            loader=loader,
            selected_indices=selected,
            condition=condition,
            device=device,
            mean=mean,
            std=std,
            preserved_channels=int(args.preserved_channels),
            grid_size=int(args.grid_size),
        )
        payload["loader"] = loader_summary
        payloads[condition] = payload
    peak_vram_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )
    final_state_sha256 = _state_sha256(model)

    clean_labels_exact = all(
        int(label) == int(rows[int(index)].target)
        for index, label in zip(
            payloads["clean"]["sample_index"], payloads["clean"]["labels"]
        )
    )
    shifted_labels_exact = all(
        np.array_equal(
            np.asarray(payloads[condition]["labels"], dtype=np.int64),
            np.asarray(comparators[condition]["targets"], dtype=np.int64),
        )
        for condition, _, _ in LIGHTING_CONDITIONS
    )
    if not clean_labels_exact or not shifted_labels_exact:
        raise ValueError("DDHTS extraction labels differ from the locked rows.")

    clean_map = {
        int(index): descriptor
        for index, descriptor in zip(
            payloads["clean"]["sample_index"], payloads["clean"]["descriptors"]
        )
    }
    readout = _fit_positive_evidence_readout(
        clean_map,
        rows,
        cohorts,
        seed=int(args.seed),
        svm_c=float(args.svm_c),
    )
    scaler: StandardScaler = readout.pop("scaler")
    svm: SVC = readout.pop("model")
    threshold = float(readout["threshold"])

    scores: Dict[str, np.ndarray] = {}
    candidates: Dict[str, np.ndarray] = {}
    vetoes: Dict[str, np.ndarray] = {}
    comparisons: Dict[str, Dict[str, object]] = {}
    margin_metrics: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        payload = payloads[condition]
        descriptor_by_index = {
            int(index): descriptor
            for index, descriptor in zip(payload["sample_index"], payload["descriptors"])
        }
        holdout_descriptors = np.stack(
            [descriptor_by_index[int(index)] for index in cohorts["holdout_indices"]]
        )
        condition_scores = np.asarray(
            svm.decision_function(scaler.transform(holdout_descriptors)),
            dtype=np.float64,
        )
        candidate, veto = apply_positive_evidence_veto(
            comparators[condition]["raw"],
            condition_scores,
            threshold=threshold,
        )
        scores[condition] = condition_scores
        candidates[condition] = candidate
        vetoes[condition] = veto
        comparisons[condition] = _comparison(
            np.asarray(comparators[condition]["targets"]),
            np.asarray(comparators[condition]["raw"]),
            candidate,
        )
        margin_metrics[condition] = _classification_metrics(
            np.asarray(comparators[condition]["targets"]),
            np.asarray(comparators[condition]["margin_agem"]),
        )

    clean_targets = np.asarray(comparators["clean"]["targets"])
    clean_raw_predictions = np.asarray(comparators["clean"]["raw"]).argmax(axis=1)
    direction_mask = (clean_raw_predictions == FOCUS_CLASS) & (
        (clean_targets == FOCUS_CLASS)
        | np.isin(clean_targets, RESTRICTED_NEGATIVE_CLASSES)
    )
    direction_labels = (clean_targets[direction_mask] == FOCUS_CLASS).astype(np.int64)
    clean_direction_auc = (
        float(roc_auc_score(direction_labels, scores["clean"][direction_mask]))
        if np.unique(direction_labels).size == 2
        else None
    )

    structural_checks = {
        "tracked_worktree_clean": bool(provenance["tracked_worktree_clean"]),
        "train_rows_exact": len(rows) == EXPECTED_TRAIN_ROWS,
        "source_groups_exact": int(cohorts["unique_source_groups"])
        == EXPECTED_SOURCE_GROUPS,
        "fit_rows_exact": len(cohorts["fit_indices"]) == EXPECTED_FIT_ROWS,
        "holdout_rows_exact": len(cohorts["holdout_indices"])
        == EXPECTED_HOLDOUT_ROWS,
        "reference_rows_exact": len(cohorts["reference_indices"])
        == EXPECTED_REFERENCE_ROWS,
        "hard_rows_exact": len(cohorts["hard_indices"]) == EXPECTED_HARD_ROWS,
        "source_overlap_zero": not cohorts["source_overlap"],
        "fit_index_hash_exact": cohorts["fit_index_sha256"]
        == EXPECTED_FIT_INDEX_SHA256,
        "holdout_index_hash_exact": cohorts["holdout_index_sha256"]
        == EXPECTED_HOLDOUT_INDEX_SHA256,
        "all_conditions_complete": all(
            len(payloads[condition]["sample_index"])
            == (len(clean_indices) if condition == "clean" else EXPECTED_HOLDOUT_ROWS)
            for condition in CONDITIONS
        ),
        "all_descriptors_finite": all(
            bool(payloads[condition]["finite"]) for condition in CONDITIONS
        ),
        "histogram_segments_normalized": max(
            float(payloads[condition]["segment_sum_max_abs_error"])
            for condition in CONDITIONS
        )
        <= 1e-6,
        "input_reconstruction_exact": max(
            float(payloads[condition]["input_reconstruction_max_abs_error"])
            for condition in CONDITIONS
        )
        <= 1e-5,
        "extraction_labels_exact": bool(clean_labels_exact and shifted_labels_exact),
        "scratch_pretrained_false": not bool(model_config.get("pretrained", False)),
        "keeper_state_bit_exact": final_state_sha256 == initial_state_sha256,
        "stem_parameters_frozen": all(
            not parameter.requires_grad for parameter in stem.parameters()
        ),
        "svm_replay_exact": bool(readout["deterministic_replay_exact"]),
        "fit_true_positive_broken_zero": int(readout["fit_true_positive_broken"])
        == 0,
        "clean_raw_true_positive_rows_exact": int(
            np.sum((clean_targets == FOCUS_CLASS) & (clean_raw_predictions == FOCUS_CLASS))
        )
        == 107,
        "peak_vram_lte_3p25_gib": peak_vram_gib <= MAX_PEAK_VRAM_GIB,
        "validation_not_used": True,
        "test_not_used": True,
        "binary_model_not_written": True,
    }
    gate = assess_ddhts_gate(
        structural_checks=structural_checks,
        comparisons=comparisons,
        clean_direction_auc=clean_direction_auc,
    )

    prediction_rows = _prediction_rows(
        comparators=comparators,
        payloads=payloads,
        scores=scores,
        candidates=candidates,
        vetoes=vetoes,
        threshold=threshold,
    )
    evidence_cases = _evidence_case_rows(prediction_rows)
    _write_csv(output_dir / "descriptors.csv", _descriptor_rows(payloads))
    _write_csv(output_dir / "predictions_all_conditions.csv", prediction_rows)
    _write_csv(output_dir / "evidence_cases.csv", evidence_cases)
    contact_sheet_rows = _write_contact_sheet(
        output_dir / "ddhts_evidence_contact_sheet.jpg",
        base_dataset=base_dataset,
        transform=transform,
        cases=evidence_cases,
        mean=mean,
        std=std,
        args=args,
    )

    extraction_summary = {
        condition: {
            key: value
            for key, value in payload.items()
            if key not in {"sample_index", "labels", "descriptors"}
        }
        for condition, payload in payloads.items()
    }
    summary: Dict[str, object] = {
        "status": "stage_b_authorized" if gate["stage_b_authorized"] else "closed",
        "protocol": {
            "seed": int(args.seed),
            "svm_c": float(args.svm_c),
            "preserved_channels": int(args.preserved_channels),
            "grid_size": int(args.grid_size),
            "descriptor_dim": DESCRIPTOR_DIM,
            "threshold_epsilon": THRESHOLD_EPSILON,
            "conditions": list(CONDITIONS),
        },
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "model": {
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "state_bit_exact": final_state_sha256 == initial_state_sha256,
            "keeper_sha256": LOCKED_KEEPER_SHA256,
            "scratch_pretrained_flag": bool(model_config.get("pretrained", False)),
            "stem_class": type(stem).__name__,
            "peak_vram_gib": peak_vram_gib,
        },
        "extraction": extraction_summary,
        "readout": readout,
        "comparisons": comparisons,
        "margin_agem_metrics": margin_metrics,
        "clean_direction_auc": clean_direction_auc,
        "direction_rows": {
            "positive_tp": int(direction_labels.sum()),
            "restricted_fp": int((direction_labels == 0).sum()),
        },
        "gate": gate,
        "contact_sheet_rows": int(contact_sheet_rows),
        "artifact_manifest_path": str((output_dir / "artifact_manifest.json").resolve()),
        "validation_predictions_used": False,
        "test_data_used": False,
        "binary_model_artifacts_written": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir)
    summary["artifacts"] = manifest
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
