from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import ensure_dir, json_dump, load_checkpoint, set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import build_model_from_checkpoint
from trkh.training.friendly_adversarial import (
    build_eroded_bbox_mask,
    normalized_to_rgb,
)
from trkh.training.label_preserving_hard_positive import (
    HardPositiveResult,
    generate_projected_hard_positives,
    generate_random_feasible_hard_positives,
)
from trkh.tools.audit_friendly_foreground_adversarial_readiness import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TRAIN_COUNT,
    FOCUS_CLASS,
    FOLDS,
    _is_relative_to,
    _loader_for_indices,
    _metadata_tensors,
    _resolve_device,
    _write_csv,
    assign_source_grouped_folds,
    collect_clean_split,
    select_cohort_indices,
    sha256_file,
)
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _forward_logits,
    _select_bbox_token_prior,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


SEED = 20260724
EXPECTED_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
EPSILON = 2.0 / 255.0
STEPS = 5
LABEL_MARGIN = 0.05
BBOX_ERODE_RATIO = 0.10
PROTECT_CAP = 256
SUPPRESS_CAP = 256
CONTACT_PER_COHORT = 4
REPLAY_COUNT = 4
PAPER_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
    "8a1c4a54d73728d4d61701e320687c6d-Abstract-Conference.html"
)
PAPER_SHA256 = (
    "32e812a0f603792bb5dd68419349107078fb41319acd439077ab147a1627334e"
)
SUPPLEMENT_SHA256 = (
    "9e08ddb1b936e56af496fb2d12a69b9815ac2afbb6d37d49a9a35ec34efa1e0a"
)
REFERENCE_COMMIT = "7059ca393b5cdfa04cf3304bb86c4c79712adfa3"
VARIANTS = ("candidate", "random_feasible", "feature_only")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only source-grouped A0 audit for an independently implemented "
            "LP-A3-inspired hard-positive mechanism."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attack-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--allow-preflight", action="store_true", default=False)
    return parser.parse_args(argv)


def _stable_preflight_fold(source_stem: str, folds: int = FOLDS) -> int:
    digest = hashlib.sha256(str(source_stem).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % int(folds)


def _model_logits_features(
    *,
    model: nn.Module,
    images: Tensor,
    image_valid_mask: Tensor,
    bbox: Tensor,
    crop_bbox: Tensor,
    bbox_token_prior_source: str,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    bbox_token_prior = _select_bbox_token_prior(
        bbox=bbox,
        crop_bbox=crop_bbox,
        source=bbox_token_prior_source,
    )
    logits, features = _forward_logits(
        model=model,
        images=images,
        image_valid_mask=image_valid_mask,
        bbox_metadata=bbox,
        bbox_token_prior=bbox_token_prior,
        amp=False,
        device=device,
    )
    if not isinstance(features, Mapping) or not torch.is_tensor(features.get("pooled")):
        raise RuntimeError("keeper must expose differentiable pooled features")
    pooled = features["pooled"]
    if pooled.ndim != 2 or int(pooled.size(0)) != int(images.size(0)):
        raise RuntimeError("keeper pooled feature shape is invalid")
    return logits.float(), pooled


def _selected_cohort(
    clean_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[int], Dict[int, str]]:
    protect = select_cohort_indices(
        clean_rows,
        cohort="protect_correct",
        cap=PROTECT_CAP,
    )
    suppress = select_cohort_indices(
        clean_rows,
        cohort="suppress_correct",
        cap=SUPPRESS_CAP,
    )
    negative_errors = sorted(
        (
            (
                abs(float(row["boundary_margin"])),
                int(row["sample_index"]),
            )
            for row in clean_rows
            if str(row["cohort"]) == "negative_error"
        ),
        key=lambda item: (item[0], item[1]),
    )
    cohort_by_index: Dict[int, str] = {}
    for sample_index in protect:
        cohort_by_index[int(sample_index)] = "protect_correct"
    for sample_index in suppress:
        cohort_by_index[int(sample_index)] = "suppress_correct"
    for _, sample_index in negative_errors:
        cohort_by_index[int(sample_index)] = "negative_error"
    indices = sorted(cohort_by_index)
    return indices, cohort_by_index


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    by_cohort = {
        cohort: [row for row in rows if str(row["audit_cohort"]) == cohort]
        for cohort in ("protect_correct", "suppress_correct", "negative_error")
    }
    protect = by_cohort["protect_correct"]
    suppress = by_cohort["suppress_correct"]
    negative_errors = by_cohort["negative_error"]

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    feature_values = np.asarray(
        [float(row["feature_distance"]) for row in rows], dtype=np.float64
    )
    constraint_violations = sum(
        not bool(row["constraint_satisfied"]) for row in rows
    )
    return {
        "support": int(len(rows)),
        "cohort_support": {
            cohort: int(len(values)) for cohort, values in by_cohort.items()
        },
        "class1_tp_retention": ratio(
            sum(int(row["aug_prediction_index"]) == FOCUS_CLASS for row in protect),
            len(protect),
        ),
        "restricted_rival_retention": ratio(
            sum(
                int(row["aug_prediction_index"]) == int(row["target_index"])
                for row in suppress
            ),
            len(suppress),
        ),
        "restricted_fp_rejection": ratio(
            sum(
                int(row["aug_prediction_index"]) != FOCUS_CLASS
                for row in negative_errors
            ),
            len(negative_errors),
        ),
        "restricted_fp_correct_to_target": ratio(
            sum(
                int(row["aug_prediction_index"]) == int(row["target_index"])
                for row in negative_errors
            ),
            len(negative_errors),
        ),
        "class1_tp_broken": int(
            sum(int(row["aug_prediction_index"]) != FOCUS_CLASS for row in protect)
        ),
        "restricted_fp_removed": int(
            sum(
                int(row["aug_prediction_index"]) != FOCUS_CLASS
                for row in negative_errors
            )
        ),
        "restricted_fp_created": int(
            sum(int(row["aug_prediction_index"]) == FOCUS_CLASS for row in suppress)
        ),
        "prediction_changes": int(
            sum(
                int(row["aug_prediction_index"])
                != int(row["clean_prediction_index"])
                for row in rows
            )
        ),
        "constraint_violation_count": int(constraint_violations),
        "constraint_violation_rate": ratio(constraint_violations, len(rows)),
        "fallback_clean_count": int(
            sum(int(row["selected_step"]) == 0 for row in rows)
        ),
        "median_feature_distance": (
            float(np.median(feature_values)) if feature_values.size else float("nan")
        ),
        "mean_feature_distance": (
            float(feature_values.mean()) if feature_values.size else float("nan")
        ),
        "max_true_log_probability_drop": (
            max(float(row["true_log_probability_drop"]) for row in rows)
            if rows
            else float("nan")
        ),
        "max_delta_rgb": (
            max(float(row["max_delta_rgb"]) for row in rows)
            if rows
            else float("nan")
        ),
        "max_outside_mask_delta": (
            max(float(row["max_outside_mask_delta"]) for row in rows)
            if rows
            else float("nan")
        ),
        "empty_mask_count": int(sum(bool(row["empty_mask"]) for row in rows)),
    }


def _fold_rows(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        for fold in range(FOLDS):
            summary = summarize_rows(
                [row for row in rows if int(row["source_fold"]) == fold]
            )
            output.append({"variant": variant, "source_fold": fold, **summary})
    return output


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _check(name: str, value: object, requirement: str, passed: bool) -> Dict[str, object]:
    return {
        "name": str(name),
        "value": value,
        "requirement": str(requirement),
        "passed": bool(passed),
    }


def assess_a0_gate(
    *,
    preflight: bool,
    checkpoint_sha256: str,
    data_sha256: str,
    train_support: int,
    source_fold_crossings: int,
    replay: Mapping[str, object],
    variant_summaries: Mapping[str, Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    candidate = variant_summaries["candidate"]
    random_control = variant_summaries["random_feasible"]
    feature_only = variant_summaries["feature_only"]
    checks: List[Dict[str, object]] = [
        _check("full_protocol", not preflight, "true", not preflight),
        _check(
            "checkpoint_sha256",
            checkpoint_sha256,
            EXPECTED_CHECKPOINT_SHA256,
            checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256,
        ),
        _check(
            "data_spec_sha256",
            data_sha256,
            EXPECTED_DATA_SHA256,
            data_sha256 == EXPECTED_DATA_SHA256,
        ),
        _check(
            "train_support",
            int(train_support),
            str(EXPECTED_TRAIN_COUNT),
            int(train_support) == EXPECTED_TRAIN_COUNT,
        ),
        _check(
            "source_fold_crossings",
            int(source_fold_crossings),
            "0",
            int(source_fold_crossings) == 0,
        ),
        _check(
            "deterministic_replay",
            replay,
            "max_abs_delta=0 and steps_equal",
            float(replay.get("max_abs_delta", float("inf"))) == 0.0
            and bool(replay.get("selected_steps_equal", False)),
        ),
    ]
    for variant in VARIANTS:
        values = variant_summaries[variant]
        checks.extend(
            (
                _check(
                    f"{variant}_nonempty_masks",
                    int(values["empty_mask_count"]),
                    "0",
                    int(values["empty_mask_count"]) == 0,
                ),
                _check(
                    f"{variant}_rgb_bound",
                    float(values["max_delta_rgb"]),
                    f"<={EPSILON + 1e-7:.10f}",
                    float(values["max_delta_rgb"]) <= EPSILON + 1e-7,
                ),
                _check(
                    f"{variant}_outside_mask_delta",
                    float(values["max_outside_mask_delta"]),
                    "exactly 0",
                    float(values["max_outside_mask_delta"]) == 0.0,
                ),
            )
        )
    candidate_fp = _safe_float(candidate["restricted_fp_rejection"])
    random_fp = _safe_float(random_control["restricted_fp_rejection"])
    candidate_tp = _safe_float(candidate["class1_tp_retention"])
    feature_tp = _safe_float(feature_only["class1_tp_retention"])
    candidate_distance = _safe_float(candidate["median_feature_distance"])
    random_distance = _safe_float(random_control["median_feature_distance"])
    distance_ratio = (
        candidate_distance / random_distance
        if math.isfinite(random_distance) and random_distance > 0.0
        else float("nan")
    )
    checks.extend(
        (
            _check(
                "candidate_constraint_violations",
                int(candidate["constraint_violation_count"]),
                "0",
                int(candidate["constraint_violation_count"]) == 0,
            ),
            _check(
                "candidate_class1_tp_retention",
                candidate_tp,
                ">=0.98",
                math.isfinite(candidate_tp) and candidate_tp >= 0.98,
            ),
            _check(
                "candidate_restricted_rival_retention",
                candidate["restricted_rival_retention"],
                ">=0.98",
                _safe_float(candidate["restricted_rival_retention"]) >= 0.98,
            ),
            _check(
                "candidate_restricted_fp_rejection",
                candidate_fp,
                ">=0.05",
                math.isfinite(candidate_fp) and candidate_fp >= 0.05,
            ),
            _check(
                "candidate_minus_random_fp_rejection",
                candidate_fp - random_fp,
                ">=0.02",
                math.isfinite(candidate_fp - random_fp)
                and candidate_fp - random_fp >= 0.02,
            ),
            _check(
                "candidate_minus_feature_only_tp_retention",
                candidate_tp - feature_tp,
                ">=0.02",
                math.isfinite(candidate_tp - feature_tp)
                and candidate_tp - feature_tp >= 0.02,
            ),
            _check(
                "candidate_to_random_median_feature_distance",
                distance_ratio,
                ">=1.25",
                math.isfinite(distance_ratio) and distance_ratio >= 1.25,
            ),
        )
    )
    fold_lookup = {
        (str(row["variant"]), int(row["source_fold"])): row for row in fold_rows
    }
    stable_folds = 0
    for fold in range(FOLDS):
        candidate_fold = fold_lookup[("candidate", fold)]
        random_fold = fold_lookup[("random_feasible", fold)]
        tp = _safe_float(candidate_fold["class1_tp_retention"])
        fp_delta = _safe_float(candidate_fold["restricted_fp_rejection"]) - _safe_float(
            random_fold["restricted_fp_rejection"]
        )
        if math.isfinite(tp) and math.isfinite(fp_delta) and tp >= 0.97 and fp_delta > 0.0:
            stable_folds += 1
    checks.append(
        _check(
            "source_fold_stability",
            stable_folds,
            ">=4/5 folds",
            stable_folds >= 4,
        )
    )
    passed = sum(bool(check["passed"]) for check in checks)
    return {
        "automated_a0_permission": passed == len(checks),
        "visual_review_required": True,
        "trainer_integration_permission": False,
        "passed": int(passed),
        "total": int(len(checks)),
        "checks": checks,
    }


def _pil_image(rgb: Tensor, size: int = 144) -> Image.Image:
    array = (
        rgb.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = Image.fromarray(array)
    return image.resize((size, size), Image.Resampling.BILINEAR)


def _write_contact_sheet(
    *,
    path: Path,
    sample_indices: Sequence[int],
    visuals: Mapping[int, Mapping[str, Image.Image]],
    row_lookup: Mapping[Tuple[str, int], Mapping[str, object]],
) -> Dict[str, object]:
    columns = (
        "clean",
        "candidate",
        "candidate_delta",
        "random_feasible",
        "random_delta",
        "feature_only",
        "feature_delta",
    )
    tile = 144
    header = 34
    row_label = 36
    width = tile * len(columns)
    height = header + (tile + row_label) * len(sample_indices)
    sheet = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for column_index, name in enumerate(columns):
        draw.text((column_index * tile + 4, 8), name, fill=(0, 0, 0))
    rendered = 0
    for row_index, sample_index in enumerate(sample_indices):
        images = visuals.get(int(sample_index), {})
        if any(name not in images for name in columns):
            continue
        top = header + row_index * (tile + row_label)
        for column_index, name in enumerate(columns):
            sheet.paste(images[name], (column_index * tile, top))
        candidate = row_lookup[("candidate", int(sample_index))]
        random_row = row_lookup[("random_feasible", int(sample_index))]
        feature = row_lookup[("feature_only", int(sample_index))]
        caption = (
            f"idx={sample_index} cohort={candidate['audit_cohort']} "
            f"y={candidate['target_index']} clean={candidate['clean_prediction_index']} "
            f"cand/rand/feat={candidate['aug_prediction_index']}/"
            f"{random_row['aug_prediction_index']}/{feature['aug_prediction_index']}"
        )
        draw.text((4, top + tile + 5), caption, fill=(0, 0, 0))
        rendered += 1
    sheet.save(path, format="PNG", optimize=True)
    return {
        "path": path.name,
        "requested_rows": int(len(sample_indices)),
        "rendered_rows": int(rendered),
        "columns": list(columns),
        "sha256": sha256_file(path),
    }


def _run_variant(
    *,
    variant: str,
    dataset: Dataset,
    selected_indices: Sequence[int],
    cohort_by_index: Mapping[int, str],
    clean_by_index: Mapping[int, Mapping[str, object]],
    fold_assignments: Mapping[int, int],
    contact_indices: Sequence[int],
    visuals: Dict[int, Dict[str, Image.Image]],
    model: nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    loader, loader_summary = _loader_for_indices(
        dataset=dataset,
        indices=selected_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        context=f"lpa3_hard_positive_{variant}",
    )
    output_rows: List[Dict[str, object]] = []
    contact_set = {int(value) for value in contact_indices}
    for images, labels, metadata in tqdm(
        loader, desc=f"lpa3-a0:{variant}", unit="batch"
    ):
        if not isinstance(metadata, Mapping):
            raise ValueError("classification metadata is required")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        bbox, crop_bbox, image_valid_mask, audit_indices = _metadata_tensors(
            metadata, device=device
        )
        attack_mask = build_eroded_bbox_mask(
            crop_bbox,
            image_valid_mask,
            height=int(images.size(2)),
            width=int(images.size(3)),
            erode_ratio=BBOX_ERODE_RATIO,
        )

        def forward_logits_features(candidate_images: Tensor) -> Tuple[Tensor, Tensor]:
            return _model_logits_features(
                model=model,
                images=candidate_images,
                image_valid_mask=image_valid_mask,
                bbox=bbox,
                crop_bbox=crop_bbox,
                bbox_token_prior_source=bbox_token_prior_source,
                device=device,
            )

        if variant == "candidate":
            result = generate_projected_hard_positives(
                images=images,
                labels=labels,
                attack_mask=attack_mask,
                sample_indices=audit_indices,
                forward_logits_features=forward_logits_features,
                mean=mean,
                std=std,
                epsilon=EPSILON,
                steps=STEPS,
                label_margin=LABEL_MARGIN,
                seed=seed,
                label_penalty=True,
                require_feasible_selection=True,
            )
        elif variant == "random_feasible":
            result = generate_random_feasible_hard_positives(
                images=images,
                labels=labels,
                attack_mask=attack_mask,
                sample_indices=audit_indices,
                forward_logits_features=forward_logits_features,
                mean=mean,
                std=std,
                epsilon=EPSILON,
                candidates=STEPS,
                label_margin=LABEL_MARGIN,
                seed=seed,
            )
        elif variant == "feature_only":
            result = generate_projected_hard_positives(
                images=images,
                labels=labels,
                attack_mask=attack_mask,
                sample_indices=audit_indices,
                forward_logits_features=forward_logits_features,
                mean=mean,
                std=std,
                epsilon=EPSILON,
                steps=STEPS,
                label_margin=LABEL_MARGIN,
                seed=seed,
                label_penalty=False,
                require_feasible_selection=False,
            )
        else:
            raise ValueError(f"unknown variant: {variant}")

        with torch.no_grad():
            augmented_logits, _ = forward_logits_features(result.augmented_images)
            augmented_probabilities = torch.softmax(augmented_logits.float(), dim=1)
            augmented_predictions = augmented_probabilities.argmax(dim=1)
            clean_rgb = normalized_to_rgb(images, mean=mean, std=std)
            augmented_rgb = normalized_to_rgb(
                result.augmented_images, mean=mean, std=std
            )
        mask_float = result.attack_mask.to(dtype=result.delta_rgb.dtype)
        outside_delta = result.delta_rgb * (1.0 - mask_float)
        paths = metadata.get("paths")
        if not isinstance(paths, list) or len(paths) != int(images.size(0)):
            raise ValueError("classification paths are incomplete")
        for batch_index in range(int(images.size(0))):
            sample_index = int(audit_indices[batch_index].item())
            clean = clean_by_index[sample_index]
            row: Dict[str, object] = {
                "variant": variant,
                "sample_index": sample_index,
                "image_path": str(paths[batch_index]),
                "source_stem": str(clean["source_stem"]),
                "source_fold": int(fold_assignments[sample_index]),
                "audit_cohort": str(cohort_by_index[sample_index]),
                "target_index": int(labels[batch_index].item()),
                "clean_prediction_index": int(clean["prediction_index"]),
                "aug_prediction_index": int(
                    augmented_predictions[batch_index].item()
                ),
                "transition": (
                    f"{int(clean['prediction_index'])}->"
                    f"{int(augmented_predictions[batch_index].item())}"
                ),
                "feature_distance": float(
                    result.feature_distance[batch_index].item()
                ),
                "true_log_probability_drop": float(
                    result.true_log_probability_drop[batch_index].item()
                ),
                "constraint_satisfied": bool(
                    result.constraint_satisfied[batch_index].item()
                ),
                "selected_step": int(result.selected_step[batch_index].item()),
                "mask_fraction": float(
                    result.attack_mask[batch_index].float().mean().item()
                ),
                "empty_mask": bool(
                    result.attack_mask[batch_index].sum().item() == 0
                ),
                "max_delta_rgb": float(
                    result.delta_rgb[batch_index].abs().max().item()
                ),
                "mean_active_delta_rgb": float(
                    (
                        result.delta_rgb[batch_index].abs()
                        * mask_float[batch_index]
                    ).sum().item()
                    / max(
                        1.0,
                        float(mask_float[batch_index].sum().item())
                        * int(result.delta_rgb.size(1)),
                    )
                ),
                "max_outside_mask_delta": float(
                    outside_delta[batch_index].abs().max().item()
                ),
            }
            for class_index in range(int(augmented_probabilities.size(1))):
                row[f"clean_prob_{class_index}"] = float(
                    clean[f"prob_{class_index}"]
                )
                row[f"aug_prob_{class_index}"] = float(
                    augmented_probabilities[batch_index, class_index].item()
                )
            output_rows.append(row)
            if sample_index in contact_set:
                visual = visuals.setdefault(sample_index, {})
                visual.setdefault("clean", _pil_image(clean_rgb[batch_index]))
                visual[variant] = _pil_image(augmented_rgb[batch_index])
                delta_visual = (
                    result.delta_rgb[batch_index].float() * 16.0 + 0.5
                ).clamp(0.0, 1.0)
                delta_key = {
                    "candidate": "candidate_delta",
                    "random_feasible": "random_delta",
                    "feature_only": "feature_delta",
                }[variant]
                visual[delta_key] = _pil_image(delta_visual)
        model.zero_grad(set_to_none=True)
    return output_rows, loader_summary


def _run_replay(
    *,
    dataset: Dataset,
    indices: Sequence[int],
    model: nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
    seed: int,
) -> Dict[str, object]:
    replay_indices = list(indices[:REPLAY_COUNT])
    loader, _ = _loader_for_indices(
        dataset=dataset,
        indices=replay_indices,
        batch_size=max(1, len(replay_indices)),
        num_workers=0,
        device=device,
        context="lpa3_hard_positive_replay",
    )
    images, labels, metadata = next(iter(loader))
    if not isinstance(metadata, Mapping):
        raise ValueError("classification metadata is required")
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    bbox, crop_bbox, image_valid_mask, audit_indices = _metadata_tensors(
        metadata, device=device
    )
    attack_mask = build_eroded_bbox_mask(
        crop_bbox,
        image_valid_mask,
        height=int(images.size(2)),
        width=int(images.size(3)),
        erode_ratio=BBOX_ERODE_RATIO,
    )

    def forward_logits_features(candidate_images: Tensor) -> Tuple[Tensor, Tensor]:
        return _model_logits_features(
            model=model,
            images=candidate_images,
            image_valid_mask=image_valid_mask,
            bbox=bbox,
            crop_bbox=crop_bbox,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
        )

    def generate() -> HardPositiveResult:
        return generate_projected_hard_positives(
            images=images,
            labels=labels,
            attack_mask=attack_mask,
            sample_indices=audit_indices,
            forward_logits_features=forward_logits_features,
            mean=mean,
            std=std,
            epsilon=EPSILON,
            steps=STEPS,
            label_margin=LABEL_MARGIN,
            seed=seed,
        )

    first = generate()
    second = generate()
    return {
        "sample_indices": [int(value) for value in audit_indices.cpu().tolist()],
        "max_abs_delta": float(
            (first.delta_rgb - second.delta_rgb).abs().max().item()
        ),
        "selected_steps_equal": bool(
            torch.equal(first.selected_step, second.selected_step)
        ),
        "feature_distance_max_abs": float(
            (first.feature_distance - second.feature_distance).abs().max().item()
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.attack_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    capped = int(args.max_train_samples) > 0
    if capped and not bool(args.allow_preflight):
        raise ValueError("sample caps require --allow-preflight")

    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed), deterministic=True)
    device = _resolve_device(str(args.device))

    checkpoint_sha = sha256_file(checkpoint_path)
    data_sha = sha256_file(data_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("A0 protocol is locked to the current keeper checkpoint")
    if data_sha != EXPECTED_DATA_SHA256:
        raise ValueError("A0 protocol is locked to the yolo_f data-spec hash")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(
        data_path, class_name_mode="raw", expected_num_classes=5
    )
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        raise ValueError("A0 requires bbox-aware yolo_f")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and data class orders differ")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device=device, dtype=torch.float32)
    model.eval()
    mean, std = checkpoint_input_normalization(checkpoint)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(
        dict(checkpoint), image_size=image_size
    )
    train_dataset = _build_classification_dataset(
        data_spec=data_spec,
        split="train",
        transform=transform,
        checkpoint=dict(checkpoint),
    )
    train_indices = list(
        range(
            min(len(train_dataset), int(args.max_train_samples))
            if int(args.max_train_samples) > 0
            else len(train_dataset)
        )
    )
    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "bbox")
        or "bbox"
    )
    clean = collect_clean_split(
        split="train",
        dataset=train_dataset,
        indices=train_indices,
        model=model,
        class_names=class_names,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    clean_rows = clean["rows"]
    if not isinstance(clean_rows, list):
        raise RuntimeError("clean train rows are invalid")
    if capped:
        fold_assignments = {
            int(row["sample_index"]): _stable_preflight_fold(str(row["source_stem"]))
            for row in clean_rows
        }
    else:
        fold_assignments = assign_source_grouped_folds(
            clean_rows, folds=FOLDS, seed=int(args.seed)
        )
    for row in clean_rows:
        row["source_fold"] = int(fold_assignments[int(row["sample_index"])])
    sources_to_folds: Dict[str, set[int]] = {}
    for row in clean_rows:
        sources_to_folds.setdefault(str(row["source_stem"]), set()).add(
            int(row["source_fold"])
        )
    source_fold_crossings = sum(
        len(values) != 1 for values in sources_to_folds.values()
    )

    selected_indices, cohort_by_index = _selected_cohort(clean_rows)
    if not selected_indices:
        raise RuntimeError("locked A0 cohort is empty")
    clean_by_index = {int(row["sample_index"]): row for row in clean_rows}
    contact_indices: List[int] = []
    for cohort in ("protect_correct", "suppress_correct", "negative_error"):
        values = sorted(
            (
                (
                    abs(float(clean_by_index[index]["boundary_margin"])),
                    int(index),
                )
                for index in selected_indices
                if cohort_by_index[int(index)] == cohort
            ),
            key=lambda item: (item[0], item[1]),
        )
        contact_indices.extend(index for _, index in values[:CONTACT_PER_COHORT])

    rows_by_variant: Dict[str, List[Dict[str, object]]] = {}
    loaders: Dict[str, object] = {}
    visuals: Dict[int, Dict[str, Image.Image]] = {}
    for variant in VARIANTS:
        rows, loader_summary = _run_variant(
            variant=variant,
            dataset=train_dataset,
            selected_indices=selected_indices,
            cohort_by_index=cohort_by_index,
            clean_by_index=clean_by_index,
            fold_assignments=fold_assignments,
            contact_indices=contact_indices,
            visuals=visuals,
            model=model,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            batch_size=int(args.attack_batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )
        rows_by_variant[variant] = rows
        loaders[variant] = loader_summary
    all_rows = [
        row for variant in VARIANTS for row in rows_by_variant[variant]
    ]
    variant_summaries = {
        variant: summarize_rows(rows_by_variant[variant]) for variant in VARIANTS
    }
    fold_rows = _fold_rows(rows_by_variant)
    replay = _run_replay(
        dataset=train_dataset,
        indices=selected_indices,
        model=model,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        seed=int(args.seed),
    )

    _write_csv(output_dir / "train_clean_predictions.csv", clean_rows)
    _write_csv(output_dir / "hard_positive_rows.csv", all_rows)
    _write_csv(output_dir / "source_fold_metrics.csv", fold_rows)
    row_lookup = {
        (str(row["variant"]), int(row["sample_index"])): row for row in all_rows
    }
    contact = _write_contact_sheet(
        path=output_dir / "contact_sheet.png",
        sample_indices=contact_indices,
        visuals=visuals,
        row_lookup=row_lookup,
    )
    gate = assess_a0_gate(
        preflight=capped,
        checkpoint_sha256=checkpoint_sha,
        data_sha256=data_sha,
        train_support=int(clean["support"]),
        source_fold_crossings=int(source_fold_crossings),
        replay=replay,
        variant_summaries=variant_summaries,
        fold_rows=fold_rows,
    )
    summary = {
        "mode": "lpa3_inspired_hard_positive_a0",
        "protocol_locked": True,
        "preflight": bool(capped),
        "train_only": True,
        "validation_split_used": False,
        "test_split_used": False,
        "raw_dataset_modified": False,
        "model_fit": False,
        "checkpoint_written": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "data": str(data_path),
        "data_sha256": data_sha,
        "device": str(device),
        "seed": int(args.seed),
        "class_names": class_names,
        "image_size": image_size,
        "bbox_token_prior_source": bbox_token_prior_source,
        "sources": {
            "paper": PAPER_URL,
            "paper_sha256": PAPER_SHA256,
            "supplement_sha256": SUPPLEMENT_SHA256,
            "official_reference_commit": REFERENCE_COMMIT,
            "official_code_imported": False,
            "license_note": (
                "No repository license was visible at protocol lock; "
                "implementation is independent and paper-derived."
            ),
        },
        "protocol": {
            "epsilon_rgb": EPSILON,
            "steps": STEPS,
            "label_margin_log_probability": LABEL_MARGIN,
            "bbox_erode_ratio_each_side": BBOX_ERODE_RATIO,
            "feature": "l2_normalized_features_pooled",
            "lagrange_schedule": "10 ** (step / 5)",
            "step_schedule": "(2/255) * 0.1 ** (step / 5)",
            "initial_noise": "0.01 * N(0,1), sample-index seeded, projected",
            "protect_cap": PROTECT_CAP,
            "suppress_cap": SUPPRESS_CAP,
            "negative_error_cap": "all",
            "source_grouped_folds": FOLDS,
            "parameter_sweep": False,
        },
        "clean_train": {
            key: value for key, value in clean.items() if key != "rows"
        },
        "selected_support": int(len(selected_indices)),
        "selected_cohort_support": {
            cohort: int(sum(value == cohort for value in cohort_by_index.values()))
            for cohort in ("protect_correct", "suppress_correct", "negative_error")
        },
        "source_group_count": int(len(sources_to_folds)),
        "source_fold_crossings": int(source_fold_crossings),
        "loaders": loaders,
        "variants": variant_summaries,
        "source_fold_metrics": fold_rows,
        "replay": replay,
        "contact_sheet": contact,
        "gate": gate,
        "artifact_manifest_path": "artifact_manifest.json",
    }
    json_dump(output_dir / "summary.json", summary)
    readme = [
        "# LP-A3-Inspired Hard-Positive A0",
        "",
        "- Train-only frozen-keeper audit; validation/test were not built.",
        f"- Selected support: `{len(selected_indices)}`.",
        (
            "- Candidate TP/rival retention and restricted-FP rejection: "
            f"`{variant_summaries['candidate']['class1_tp_retention']:.6f}/"
            f"{variant_summaries['candidate']['restricted_rival_retention']:.6f}/"
            f"{variant_summaries['candidate']['restricted_fp_rejection']:.6f}`."
        ),
        (
            "- Automated A0 permission: "
            f"`{gate['automated_a0_permission']}` "
            f"(`{gate['passed']}/{gate['total']}` checks); visual review remains required."
        ),
        "",
        "Passing A0 would authorize only default-off trainer integration, not a full train.",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    _write_artifact_manifest(
        output_dir,
        mode="lpa3_inspired_hard_positive_a0_evidence_manifest",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
