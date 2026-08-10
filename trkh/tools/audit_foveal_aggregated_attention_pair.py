from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.stats import rankdata
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from trkh.core.config import load_data_spec
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    LightingShift,
    _forward_classification_with_metadata,
    _transform_classification_image,
    _unpack_classification_sample,
)
from trkh.inference.inference import load_checkpoint, load_model
from trkh.models.foveal_aggregated_attention import FovealAggregatedAttention
from trkh.models.model import classification_logits_from_features
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _batched_gradcam,
    _heat_overlay,
    _normalize_maps,
    _rgb_from_tensor,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
    _classification_metrics,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_xca_dual_axis_readiness import _comparison
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "foveal_aggregated_attention_a1_pair"
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLD = 0
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_ROWS = 7_372
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_DECLARATION_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only post-run gate for the matched block-1 Foveal "
            "Aggregated Attention A1 pair. Official validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path("runs/probe_faa_a1_control_5e_20260716/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("runs/probe_faa_a1_candidate_5e_20260716/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        default=Path("runs/probe_faa_a1_control_5e_20260716"),
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=Path("runs/probe_faa_a1_candidate_5e_20260716"),
    )
    parser.add_argument(
        "--fold-data",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=Path("runs/audit_faa_preflight_20260716/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"FAA pair audit output must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != values.size or y.size == 0:
        raise ValueError("AUROC labels/scores must be nonempty and equal length.")
    if not np.isfinite(values).all() or not np.isin(y, np.asarray([0, 1])).all():
        raise ValueError("AUROC inputs must contain finite scores and binary labels.")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative rows.")
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(ranks[y == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / float(positives * negatives)
    )


def _event_categories(target: int, control: int, candidate: int) -> list[str]:
    categories: list[str] = []
    if control != candidate and (
        target == FOCUS_CLASS or control == FOCUS_CLASS or candidate == FOCUS_CLASS
    ):
        categories.append("changed_class1_event")
    if (
        target in RESTRICTED_NEGATIVE_CLASSES
        and control == FOCUS_CLASS
        and candidate != FOCUS_CLASS
    ):
        categories.append("restricted_fp_removal")
    if (
        target in RESTRICTED_NEGATIVE_CLASSES
        and control != FOCUS_CLASS
        and candidate == FOCUS_CLASS
    ):
        categories.append("restricted_fp_creation")
    if target == FOCUS_CLASS and control == FOCUS_CLASS and candidate != FOCUS_CLASS:
        categories.append("class1_tp_break")
    if target == FOCUS_CLASS and control != FOCUS_CLASS and candidate == FOCUS_CLASS:
        categories.append("class1_fn_rescue")
    return categories


def _condition_corruption(name: str, brightness: float, contrast: float):
    if name == "clean":
        return IdentityCorruption()
    return LightingShift(brightness=float(brightness), contrast=float(contrast))


def _make_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    context: str,
    seed: int,
) -> Tuple[DataLoader, Dict[str, object]]:
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            generator=generator,
            **kwargs,
        ),
        summary,
    )


def _build_holdout_dataset(
    checkpoint: Mapping[str, object],
    *,
    fold_data: Path,
    holdout_rows: Sequence[CleanTrainRow],
) -> Tuple[MangoYOLOCropDataset, object, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    data_spec = load_data_spec(
        Path(fold_data),
        class_name_mode="raw",
        expected_num_classes=5,
    )
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if list(data_spec.class_names) != class_names:
        raise ValueError("Generated fold class order differs from the checkpoint.")
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="val",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    if len(dataset) != len(holdout_rows):
        raise ValueError(
            f"Generated holdout rows differ: {len(dataset)} != {len(holdout_rows)}"
        )
    for local_index, (sample, source) in enumerate(zip(dataset.samples, holdout_rows)):
        if sample.image_path.stem.casefold() != source.source_stem:
            raise ValueError(f"Holdout source order differs at local index {local_index}.")
        if int(sample.primary_label) != int(source.target):
            raise ValueError(f"Holdout target order differs at local index {local_index}.")
        if "val" not in {part.casefold() for part in sample.image_path.parts}:
            raise ValueError(f"Generated holdout path is not under val: {sample.image_path}")
    return dataset, _build_eval_transform(semantics), {
        "rows": len(dataset),
        "class_counts": dataset.class_counts(5),
        "source_order_exact": True,
        "semantics": semantics,
    }


def _metadata_to_device(
    metadata: Mapping[str, object], device: torch.device
) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for key, value in metadata.items():
        if not torch.is_tensor(value):
            continue
        dtype = torch.bool if key == "image_mask" else None
        result[str(key)] = value.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
    return result


def _margin(logits: Tensor) -> Tensor:
    restricted = logits[:, list(RESTRICTED_NEGATIVE_CLASSES)].amax(dim=1)
    return logits[:, FOCUS_CLASS] - restricted


def _route_accumulator() -> Dict[str, object]:
    return {
        "batches": 0,
        "samples": 0,
        "local_mass_sum": 0.0,
        "local_mass_min": math.inf,
        "local_mass_max": -math.inf,
        "pooled_mass_sum": 0.0,
        "pooled_mass_min": math.inf,
        "pooled_mass_max": -math.inf,
        "dual_route_sum": 0.0,
        "temperature_sum": 0.0,
        "finite": True,
    }


def _update_route_accumulator(
    accumulator: Dict[str, object], trace: Mapping[str, Tensor], batch_size: int
) -> None:
    required = (
        "local_mass_mean",
        "local_mass_min",
        "local_mass_max",
        "pooled_mass_mean",
        "pooled_mass_min",
        "pooled_mass_max",
        "dual_route_fraction",
        "temperature_mean",
    )
    if any(key not in trace or not torch.is_tensor(trace[key]) for key in required):
        accumulator["finite"] = False
        return
    values = {key: float(trace[key].detach().float().item()) for key in required}
    accumulator["finite"] = bool(
        accumulator["finite"] and all(math.isfinite(value) for value in values.values())
    )
    accumulator["batches"] = int(accumulator["batches"]) + 1
    accumulator["samples"] = int(accumulator["samples"]) + int(batch_size)
    accumulator["local_mass_sum"] = float(accumulator["local_mass_sum"]) + (
        values["local_mass_mean"] * int(batch_size)
    )
    accumulator["pooled_mass_sum"] = float(accumulator["pooled_mass_sum"]) + (
        values["pooled_mass_mean"] * int(batch_size)
    )
    accumulator["dual_route_sum"] = float(accumulator["dual_route_sum"]) + (
        values["dual_route_fraction"] * int(batch_size)
    )
    accumulator["temperature_sum"] = float(accumulator["temperature_sum"]) + (
        values["temperature_mean"] * int(batch_size)
    )
    for key in ("local_mass_min", "pooled_mass_min"):
        accumulator[key] = min(float(accumulator[key]), values[key])
    for key in ("local_mass_max", "pooled_mass_max"):
        accumulator[key] = max(float(accumulator[key]), values[key])


def _finalize_route_accumulator(accumulator: Mapping[str, object]) -> Dict[str, object]:
    samples = int(accumulator["samples"])
    if samples <= 0:
        raise ValueError("No FAA route observations were collected.")
    return {
        "batches": int(accumulator["batches"]),
        "samples": samples,
        "finite": bool(accumulator["finite"]),
        "local_mass_mean": float(accumulator["local_mass_sum"]) / samples,
        "local_mass_min": float(accumulator["local_mass_min"]),
        "local_mass_max": float(accumulator["local_mass_max"]),
        "pooled_mass_mean": float(accumulator["pooled_mass_sum"]) / samples,
        "pooled_mass_min": float(accumulator["pooled_mass_min"]),
        "pooled_mass_max": float(accumulator["pooled_mass_max"]),
        "dual_route_fraction": float(accumulator["dual_route_sum"]) / samples,
        "temperature_mean": float(accumulator["temperature_sum"]) / samples,
    }


def _predict_conditions(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    holdout_rows: Sequence[CleanTrainRow],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[
    Dict[str, Dict[str, list[Dict[str, object]]]],
    Dict[str, object],
    Dict[str, object],
]:
    output: Dict[str, Dict[str, list[Dict[str, object]]]] = {
        "control": {},
        "candidate": {},
    }
    loader_summaries: Dict[str, object] = {}
    route_summaries: Dict[str, object] = {}
    local_indices = list(range(len(base_dataset)))
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, brightness, contrast) in enumerate(CONDITIONS):
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            local_indices,
            corruption=_condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = _make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"faa_pair_{condition}_holdout",
            seed=seed + 100 + condition_index,
        )
        loader_summaries[condition] = loader_summary
        role_rows = {"control": [], "candidate": []}
        route_accumulator = _route_accumulator()
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = _metadata_to_device(metadata_cpu, device)
                sample_indices = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Condition metadata lacks sample_index.")
                logits_by_role = []
                for role, model in (("control", control), ("candidate", candidate)):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        logits, _ = _forward_classification_with_metadata(
                            model,
                            images,
                            metadata,
                            device=device,
                        )
                    logits = logits.float()
                    logits_by_role.append(logits)
                    if role == "candidate":
                        module = model.blocks[0].attn
                        if not isinstance(module, FovealAggregatedAttention):
                            raise TypeError("Candidate block 1 is not FAA during prediction.")
                        _update_route_accumulator(
                            route_accumulator,
                            module.trace(),
                            int(images.size(0)),
                        )
                for role, logits in zip(("control", "candidate"), logits_by_role):
                    probabilities = logits.softmax(dim=1)
                    margins = _margin(logits)
                    predictions = probabilities.argmax(dim=1)
                    for position, local_index in enumerate(sample_indices.tolist()):
                        source = holdout_rows[int(local_index)]
                        row: Dict[str, object] = {
                            "role": role,
                            "condition": condition,
                            "local_index": int(local_index),
                            "sample_index": int(source.sample_index),
                            "source_stem": source.source_stem,
                            "object_index": int(
                                base_dataset.samples[int(local_index)].primary_object_index
                            ),
                            "target": int(targets_cpu[position].item()),
                            "prediction": int(predictions[position].item()),
                            "class1_restricted_margin": float(margins[position].item()),
                        }
                        for class_index in range(5):
                            row[f"logit_{class_index}"] = float(
                                logits[position, class_index].item()
                            )
                            row[f"prob_{class_index}"] = float(
                                probabilities[position, class_index].item()
                            )
                        role_rows[role].append(row)
        expected_local = local_indices
        for role in ("control", "candidate"):
            observed = [int(row["local_index"]) for row in role_rows[role]]
            if observed != expected_local:
                raise ValueError(f"Prediction order differs for {role}/{condition}.")
            output[role][condition] = role_rows[role]
        route_summaries[condition] = _finalize_route_accumulator(route_accumulator)
    return output, loader_summaries, route_summaries


def _write_predictions(
    path: Path,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    rows = [
        row
        for role in ("control", "candidate")
        for condition, _, _ in CONDITIONS
        for row in predictions[role][condition]
    ]
    if not rows:
        raise ValueError("No FAA pair predictions to write.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selectivity(
    *,
    rows: Sequence[CleanTrainRow],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> Dict[str, object]:
    positive = {
        int(row.sample_index)
        for row in rows
        if row.fold == FOLD
        and row.target == FOCUS_CLASS
        and row.keeper_prediction == FOCUS_CLASS
    }
    hard_negative = {
        int(row.sample_index)
        for row in rows
        if row.fold == FOLD
        and row.target in RESTRICTED_NEGATIVE_CLASSES
        and row.keeper_prediction == FOCUS_CLASS
    }
    if len(positive) != 107 or len(hard_negative) != 36:
        raise ValueError(
            "Frozen selectivity cohort differs: "
            f"positive={len(positive)}, hard_negative={len(hard_negative)}."
        )
    result: Dict[str, object] = {
        "positive_rows": len(positive),
        "hard_negative_rows": len(hard_negative),
        "conditions": {},
    }
    for condition, _, _ in CONDITIONS:
        condition_result: Dict[str, object] = {}
        for role in ("control", "candidate"):
            labels: list[int] = []
            scores: list[float] = []
            observed: list[int] = []
            for row in predictions[role][condition]:
                sample_index = int(row["sample_index"])
                if sample_index in positive:
                    labels.append(1)
                    scores.append(float(row["class1_restricted_margin"]))
                    observed.append(sample_index)
                elif sample_index in hard_negative:
                    labels.append(0)
                    scores.append(float(row["class1_restricted_margin"]))
                    observed.append(sample_index)
            condition_result[role] = {
                "auroc": _binary_auroc(labels, scores),
                "rows": len(labels),
                "positive_rows": int(sum(labels)),
                "hard_negative_rows": int(len(labels) - sum(labels)),
                "ordered_sample_index_sha256": _ordered_index_sha256(observed),
            }
        condition_result["delta"] = float(condition_result["candidate"]["auroc"]) - float(
            condition_result["control"]["auroc"]
        )
        result["conditions"][condition] = condition_result
    result["positive_sample_indices"] = sorted(positive)
    result["hard_negative_sample_indices"] = sorted(hard_negative)
    return result


def _comparisons(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]
) -> Dict[str, object]:
    return {
        condition: _comparison(
            control_rows=predictions["control"][condition],
            candidate_rows=predictions["candidate"][condition],
            num_classes=5,
            focus_class=FOCUS_CLASS,
        )
        for condition, _, _ in CONDITIONS
    }


def _build_event_manifest(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]
) -> list[Dict[str, object]]:
    events: list[Dict[str, object]] = []
    for condition, _, _ in CONDITIONS:
        control = predictions["control"][condition]
        candidate = predictions["candidate"][condition]
        for control_row, candidate_row in zip(control, candidate):
            categories = _event_categories(
                int(control_row["target"]),
                int(control_row["prediction"]),
                int(candidate_row["prediction"]),
            )
            if not categories:
                continue
            events.append(
                {
                    "condition": condition,
                    "local_index": int(control_row["local_index"]),
                    "sample_index": int(control_row["sample_index"]),
                    "source_stem": str(control_row["source_stem"]),
                    "object_index": int(control_row["object_index"]),
                    "target": int(control_row["target"]),
                    "control_prediction": int(control_row["prediction"]),
                    "candidate_prediction": int(candidate_row["prediction"]),
                    "categories": categories,
                }
            )
    return events


def _sample_bbox(dataset: MangoYOLOCropDataset, local_index: int) -> Tuple[float, ...]:
    sample = dataset.samples[int(local_index)]
    primary = dataset._select_sample_primary_object(sample)
    return tuple(float(value) for value in primary.bbox)


def _representative_requests(
    *,
    dataset: MangoYOLOCropDataset,
    holdout_rows: Sequence[CleanTrainRow],
    positive_sample_indices: Sequence[int],
) -> list[Dict[str, object]]:
    geometry = []
    for local_index in range(len(dataset)):
        x, y, width, height = _sample_bbox(dataset, local_index)
        left, top = x - width / 2.0, y - height / 2.0
        right, bottom = x + width / 2.0, y + height / 2.0
        overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
        edge_clearance = min(abs(left), abs(top), abs(1.0 - right), abs(1.0 - bottom))
        geometry.append(
            {
                "local_index": local_index,
                "area": width * height,
                "overflow": overflow,
                "edge_clearance": edge_clearance,
            }
        )
    close = max(geometry, key=lambda row: (float(row["area"]), -int(row["local_index"])))
    wide = min(geometry, key=lambda row: (float(row["area"]), int(row["local_index"])))
    partial = max(
        geometry,
        key=lambda row: (
            float(row["overflow"]),
            -float(row["edge_clearance"]),
            -int(row["local_index"]),
        ),
    )
    edge_candidates = [row for row in geometry if row is not partial] or geometry
    edge = min(
        edge_candidates,
        key=lambda row: (float(row["edge_clearance"]), int(row["local_index"])),
    )
    requests = []
    for name, selected in (
        ("representative_close", close),
        ("representative_wide", wide),
        ("representative_partial", partial),
        ("representative_edge", edge),
    ):
        local_index = int(selected["local_index"])
        source = holdout_rows[local_index]
        requests.append(
            {
                "condition": "clean",
                "local_index": local_index,
                "sample_index": int(source.sample_index),
                "source_stem": source.source_stem,
                "object_index": int(dataset.samples[local_index].primary_object_index),
                "target": int(source.target),
                "control_prediction": -1,
                "candidate_prediction": -1,
                "categories": [name],
                "geometry": selected,
            }
        )
    global_to_local = {
        int(row.sample_index): local_index for local_index, row in enumerate(holdout_rows)
    }
    positive = [
        global_to_local[int(sample_index)]
        for sample_index in positive_sample_indices
        if int(sample_index) in global_to_local
    ]
    if not positive:
        raise ValueError("No frozen-positive row is available for condition representatives.")
    for offset, (condition, _, _) in enumerate(CONDITIONS[1:]):
        local_index = positive[offset % len(positive)]
        source = holdout_rows[local_index]
        requests.append(
            {
                "condition": condition,
                "local_index": local_index,
                "sample_index": int(source.sample_index),
                "source_stem": source.source_stem,
                "object_index": int(dataset.samples[local_index].primary_object_index),
                "target": int(source.target),
                "control_prediction": -1,
                "candidate_prediction": -1,
                "categories": [f"representative_{condition}"],
            }
        )
    return requests


def _merge_xai_requests(
    events: Sequence[Mapping[str, object]],
    representatives: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    merged: Dict[Tuple[str, int], Dict[str, object]] = {}
    for source in (*events, *representatives):
        key = (str(source["condition"]), int(source["local_index"]))
        if key not in merged:
            merged[key] = dict(source)
            merged[key]["categories"] = list(source.get("categories", []))
        else:
            merged[key]["categories"] = sorted(
                set(merged[key]["categories"]).union(source.get("categories", []))
            )
            if int(source.get("control_prediction", -1)) >= 0:
                merged[key]["control_prediction"] = int(source["control_prediction"])
                merged[key]["candidate_prediction"] = int(source["candidate_prediction"])
    order = {name: index for index, (name, _, _) in enumerate(CONDITIONS)}
    return sorted(
        merged.values(),
        key=lambda row: (order[str(row["condition"])], int(row["sample_index"])),
    )


def _token_gradcam(
    activations: Tensor,
    gradients: Tensor,
    *,
    prefix_count: int,
    grid_size: Tuple[int, int],
) -> Tensor:
    patch_count = int(grid_size[0] * grid_size[1])
    patch_activations = activations[:, prefix_count : prefix_count + patch_count].float()
    patch_gradients = gradients[:, prefix_count : prefix_count + patch_count].float()
    weights = patch_gradients.mean(dim=1, keepdim=True)
    heat = torch.relu((weights * patch_activations).sum(dim=-1))
    return _normalize_maps(heat.reshape(-1, grid_size[0], grid_size[1]))


def _prefix_attention_map(
    attention: Tensor,
    *,
    prefix_count: int,
    grid_size: Tuple[int, int],
) -> Tensor:
    patch_count = int(grid_size[0] * grid_size[1])
    values = attention[
        :,
        :,
        :prefix_count,
        prefix_count : prefix_count + patch_count,
    ].float().mean(dim=(1, 2))
    return _normalize_maps(values.reshape(-1, grid_size[0], grid_size[1]))


def _bbox_foreground_mass(heat: Tensor, bboxes: Tensor) -> list[float]:
    if heat.ndim != 3 or bboxes.ndim != 2 or int(bboxes.size(1)) != 4:
        raise ValueError("Foreground-mass heat/bbox shapes are invalid.")
    result = []
    height, width = int(heat.size(1)), int(heat.size(2))
    for values, bbox in zip(heat.detach().float().cpu(), bboxes.detach().float().cpu()):
        center_x, center_y, box_width, box_height = bbox.tolist()
        left = max(0, min(width - 1, int(math.floor((center_x - box_width / 2.0) * width))))
        right = max(left + 1, min(width, int(math.ceil((center_x + box_width / 2.0) * width))))
        top = max(0, min(height - 1, int(math.floor((center_y - box_height / 2.0) * height))))
        bottom = max(top + 1, min(height, int(math.ceil((center_y + box_height / 2.0) * height))))
        total = float(values.sum().item())
        inside = float(values[top:bottom, left:right].sum().item())
        result.append(inside / max(total, 1e-9))
    return result


def _collect_xai_maps(
    *,
    role: str,
    model: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    requests: Sequence[Mapping[str, object]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Dict[Tuple[str, int], Dict[str, object]], Dict[str, object]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records: Dict[Tuple[str, int], Dict[str, object]] = {}
    loaders: Dict[str, object] = {}
    by_condition: Dict[str, list[int]] = defaultdict(list)
    for request in requests:
        by_condition[str(request["condition"])].append(int(request["local_index"]))
    condition_specs = {name: (brightness, contrast) for name, brightness, contrast in CONDITIONS}
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, selected) in enumerate(by_condition.items()):
        unique_indices = sorted(set(selected))
        brightness, contrast = condition_specs[condition]
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            unique_indices,
            corruption=_condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = _make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"faa_{role}_{condition}_xai",
            seed=seed + 500 + condition_index,
        )
        loaders[condition] = loader_summary
        for images_cpu, _, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True).requires_grad_(True)
            metadata = _metadata_to_device(metadata_cpu, device)
            sample_indices = metadata_cpu.get("sample_index")
            crop_bbox = metadata_cpu.get("crop_bbox")
            if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bbox):
                raise ValueError("XAI metadata lacks sample_index/crop_bbox.")
            captured: Dict[str, Tensor] = {}

            def stem_hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError("Stem XAI hook requires a tensor output.")
                output.retain_grad()
                captured["stem"] = output

            def block_hook(_module, _inputs, output):
                tokens = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tokens):
                    raise TypeError("Block-1 XAI hook requires tensor tokens.")
                tokens.retain_grad()
                captured["block"] = tokens

            stem_handle = model.stem.register_forward_hook(stem_hook)
            block_handle = model.blocks[0].register_forward_hook(block_hook)
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                features = model.forward_features(
                    images,
                    image_valid_mask=metadata.get("image_mask"),
                    bbox_token_prior=metadata.get("bbox"),
                    return_attention=True,
                    attention_layers=[0],
                    return_trace=True,
                )
                if torch.is_tensor(metadata.get("bbox")):
                    features["bbox"] = metadata["bbox"]
                logits = classification_logits_from_features(model, features)
            logits[:, FOCUS_CLASS].float().sum().backward()
            stem_handle.remove()
            block_handle.remove()
            stem = captured.get("stem")
            block = captured.get("block")
            if stem is None or stem.grad is None or block is None or block.grad is None:
                raise RuntimeError("Stem/block-1 XAI gradients were not retained.")
            grid_size = tuple(int(value) for value in features["grid_size"])
            prefix_count = int(model.num_prefix_tokens)
            stem_heat = _batched_gradcam(
                stem,
                stem.grad,
                size=(int(images.size(-2)), int(images.size(-1))),
            ).cpu()
            block_heat = _token_gradcam(
                block,
                block.grad,
                prefix_count=prefix_count,
                grid_size=grid_size,
            ).cpu()
            attention = features.get("attentions", {}).get(0)
            if not torch.is_tensor(attention):
                raise ValueError("Block-1 attention map is missing from XAI features.")
            prefix_attention = _prefix_attention_map(
                attention,
                prefix_count=prefix_count,
                grid_size=grid_size,
            ).cpu()
            stem_foreground = _bbox_foreground_mass(stem_heat, crop_bbox)
            block_foreground = _bbox_foreground_mass(block_heat, crop_bbox)
            local_route = pooled_route = None
            if role == "candidate":
                module = model.blocks[0].attn
                if not isinstance(module, FovealAggregatedAttention):
                    raise TypeError("Candidate XAI block 1 is not FAA.")
                trace = module.trace()
                local_value = trace.get("local_mass_map")
                pooled_value = trace.get("pooled_mass_map")
                if not torch.is_tensor(local_value) or not torch.is_tensor(pooled_value):
                    raise ValueError("FAA spatial route maps are missing.")
                local_route = _normalize_maps(
                    local_value.reshape(-1, grid_size[0], grid_size[1])
                ).cpu()
                pooled_route = _normalize_maps(
                    pooled_value.reshape(-1, grid_size[0], grid_size[1])
                ).cpu()
            probabilities = logits.detach().float().softmax(dim=1).cpu()
            for position, local_index in enumerate(sample_indices.tolist()):
                record: Dict[str, object] = {
                    "rgb": _rgb_from_tensor(images_cpu[position], mean=mean, std=std),
                    "stem_gradcam": stem_heat[position].numpy(),
                    "block1_gradcam": block_heat[position].numpy(),
                    "prefix_attention": prefix_attention[position].numpy(),
                    "stem_foreground_mass": float(stem_foreground[position]),
                    "block1_foreground_mass": float(block_foreground[position]),
                    "prediction": int(probabilities[position].argmax().item()),
                }
                if local_route is not None and pooled_route is not None:
                    record["local_route"] = local_route[position].numpy()
                    record["pooled_route"] = pooled_route[position].numpy()
                records[(condition, int(local_index))] = record
            del features, logits, images
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected = {(str(row["condition"]), int(row["local_index"])) for row in requests}
    if set(records) != expected:
        raise ValueError(f"{role} XAI records differ from the locked request set.")
    finite = all(
        all(
            np.isfinite(np.asarray(row[key])).all()
            for key in ("stem_gradcam", "block1_gradcam", "prefix_attention")
        )
        for row in records.values()
    )
    return records, {"loaders": loaders, "finite": bool(finite), "rows": len(records)}


def _render_xai_pages(
    *,
    output_dir: Path,
    requests: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
) -> Dict[str, object]:
    columns = (
        "input",
        "control stem",
        "candidate stem",
        "control block1",
        "candidate block1",
        "FAA prefix attn",
        "FAA local mass",
        "FAA pooled mass",
    )
    tile = 128
    header = 28
    label_height = 50
    rows_per_page = 6
    pages = []
    manifest_rows = []
    for page_number, start in enumerate(range(0, len(requests), rows_per_page), start=1):
        page_rows = list(requests[start : start + rows_per_page])
        canvas = Image.new(
            "RGB",
            (len(columns) * tile, header + len(page_rows) * (tile + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, name in enumerate(columns):
            draw.text((column * tile + 3, 7), name, fill="black")
        for row_position, request in enumerate(page_rows):
            key = (str(request["condition"]), int(request["local_index"]))
            control = control_maps[key]
            candidate = candidate_maps[key]
            rgb = np.asarray(candidate["rgb"], dtype=np.uint8)
            views = [
                Image.fromarray(rgb),
                _heat_overlay(rgb, np.asarray(control["stem_gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate["stem_gradcam"])),
                _heat_overlay(rgb, np.asarray(control["block1_gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate["block1_gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate["prefix_attention"])),
                _heat_overlay(rgb, np.asarray(candidate["local_route"])),
                _heat_overlay(rgb, np.asarray(candidate["pooled_route"])),
            ]
            y = header + row_position * (tile + label_height)
            for column, view in enumerate(views):
                canvas.paste(
                    view.resize((tile, tile), Image.Resampling.BILINEAR),
                    (column * tile, y),
                )
            categories = ",".join(str(value) for value in request["categories"])
            line_one = (
                f"{request['condition']} idx={request['sample_index']} "
                f"y={request['target']} c/a={request['control_prediction']}/"
                f"{request['candidate_prediction']}"
            )
            line_two = f"[{categories}]"
            draw.text((4, y + tile + 3), line_one, fill="black")
            draw.text((4, y + tile + 23), line_two, fill="black")
            manifest_rows.append(
                {
                    **dict(request),
                    "page": page_number,
                    "page_row": row_position + 1,
                }
            )
        path = output_dir / f"faa_xai_contact_sheet_{page_number:03d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    _write_json(output_dir / "xai_page_manifest.json", {"rows": manifest_rows, "pages": pages})
    return {"pages": pages, "page_count": len(pages), "rows": len(manifest_rows)}


def _xai_summary(
    *,
    requests: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    render: Mapping[str, object],
) -> Dict[str, object]:
    keys = [(str(row["condition"]), int(row["local_index"])) for row in requests]
    event_keys = {(str(row["condition"]), int(row["local_index"])) for row in events}
    request_keys = set(keys)
    control_stem = [float(control_maps[key]["stem_foreground_mass"]) for key in keys]
    candidate_stem = [float(candidate_maps[key]["stem_foreground_mass"]) for key in keys]
    control_block = [float(control_maps[key]["block1_foreground_mass"]) for key in keys]
    candidate_block = [float(candidate_maps[key]["block1_foreground_mass"]) for key in keys]
    categories = {
        str(category)
        for request in requests
        for category in request.get("categories", [])
    }
    required_representatives = {
        "representative_close",
        "representative_wide",
        "representative_partial",
        "representative_edge",
        "representative_lighting_dim",
        "representative_lighting_bright",
        "representative_low_contrast",
    }
    return {
        "event_rows": len(events),
        "xai_rows": len(requests),
        "all_event_rows_covered": event_keys.issubset(request_keys),
        "representative_categories": sorted(categories.intersection(required_representatives)),
        "all_representatives_covered": required_representatives.issubset(categories),
        "control_stem_foreground_mass": float(np.mean(control_stem)),
        "candidate_stem_foreground_mass": float(np.mean(candidate_stem)),
        "stem_foreground_mass_delta": float(np.mean(candidate_stem) - np.mean(control_stem)),
        "control_block1_foreground_mass": float(np.mean(control_block)),
        "candidate_block1_foreground_mass": float(np.mean(candidate_block)),
        "block1_foreground_mass_delta": float(
            np.mean(candidate_block) - np.mean(control_block)
        ),
        "maps_finite": bool(
            all(
                np.isfinite(np.asarray(candidate_maps[key][name])).all()
                for key in keys
                for name in (
                    "stem_gradcam",
                    "block1_gradcam",
                    "prefix_attention",
                    "local_route",
                    "pooled_route",
                )
            )
        ),
        "render": dict(render),
    }


class _RegionPerturbationDataset(Dataset):
    def __init__(
        self,
        base_dataset: MangoYOLOCropDataset,
        selected_indices: Sequence[int],
        *,
        mode: str,
        transform,
    ) -> None:
        if mode not in {"clean", "object", "far_background"}:
            raise ValueError(f"Unknown perturbation mode: {mode}")
        self.base_dataset = base_dataset
        self.selected_indices = tuple(int(value) for value in selected_indices)
        self.mode = mode
        self.transform = transform

    @staticmethod
    def _perturb(image: Image.Image, bbox: Tensor, *, mode: str) -> Image.Image:
        if mode == "clean":
            return image.copy()
        source = image.convert("RGB")
        gray = ImageOps.grayscale(source).convert("RGB")
        degraded = Image.blend(source, gray, 0.88).filter(ImageFilter.GaussianBlur(3.0))
        width, height = source.size
        center_x, center_y, box_width, box_height = bbox.float().tolist()
        margin = 0.0 if mode == "object" else 0.12
        left = int(math.floor((center_x - box_width / 2.0 - margin) * width))
        right = int(math.ceil((center_x + box_width / 2.0 + margin) * width))
        top = int(math.floor((center_y - box_height / 2.0 - margin) * height))
        bottom = int(math.ceil((center_y + box_height / 2.0 + margin) * height))
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        mask = Image.new("L", source.size, 0 if mode == "object" else 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            (left, top, max(left, right - 1), max(top, bottom - 1)),
            fill=255 if mode == "object" else 0,
        )
        return Image.composite(degraded, source, mask)

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, position: int):
        sample_index = self.selected_indices[int(position)]
        image, label, metadata, bbox = _unpack_classification_sample(
            self.base_dataset[sample_index]
        )
        crop_bbox = metadata.get("crop_bbox")
        if not torch.is_tensor(crop_bbox) or crop_bbox.numel() != 4:
            crop_bbox = bbox
        image = self._perturb(image, crop_bbox.reshape(4), mode=self.mode)
        tensor, transformed_metadata = _transform_classification_image(
            image,
            label=label,
            metadata=metadata,
            bbox=bbox,
            transform=self.transform,
        )
        transformed_metadata["sample_index"] = torch.tensor(sample_index, dtype=torch.long)
        return tensor, label, transformed_metadata


def _predict_margins(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[list[int], np.ndarray]:
    indices: list[int] = []
    margins: list[float] = []
    with torch.inference_mode():
        for images_cpu, _, metadata_cpu in loader:
            sample_indices = metadata_cpu.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Perturbation metadata lacks sample_index.")
            images = images_cpu.to(device=device, non_blocking=True)
            metadata = _metadata_to_device(metadata_cpu, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _ = _forward_classification_with_metadata(
                    model,
                    images,
                    metadata,
                    device=device,
                )
            indices.extend(int(value) for value in sample_indices.tolist())
            margins.extend(float(value) for value in _margin(logits.float()).cpu().tolist())
    return indices, np.asarray(margins, dtype=np.float64)


def _perturbation_audit(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    holdout_rows: Sequence[CleanTrainRow],
    selectivity: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    global_to_local = {
        int(row.sample_index): local_index for local_index, row in enumerate(holdout_rows)
    }
    cohort_global = [
        *selectivity["positive_sample_indices"],
        *selectivity["hard_negative_sample_indices"],
    ]
    selected = sorted(global_to_local[int(value)] for value in cohort_global)
    margins: Dict[str, Dict[str, np.ndarray]] = {"control": {}, "candidate": {}}
    loader_summaries: Dict[str, object] = {}
    expected_order: Optional[list[int]] = None
    for mode_index, mode in enumerate(("clean", "object", "far_background")):
        dataset = _RegionPerturbationDataset(
            base_dataset,
            selected,
            mode=mode,
            transform=transform,
        )
        loader, loader_summary = _make_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"faa_{mode}_perturbation",
            seed=seed + 700 + mode_index,
        )
        loader_summaries[mode] = loader_summary
        for role, model in (("control", control), ("candidate", candidate)):
            indices, values = _predict_margins(model=model, loader=loader, device=device)
            if expected_order is None:
                expected_order = indices
            if indices != expected_order:
                raise ValueError(f"Perturbation order differs for {role}/{mode}.")
            margins[role][mode] = values
    result: Dict[str, object] = {
        "rows": len(selected),
        "local_index_sha256": _ordered_index_sha256(selected),
        "loaders": loader_summaries,
        "roles": {},
    }
    for role in ("control", "candidate"):
        clean = margins[role]["clean"]
        object_effect = np.abs(margins[role]["object"] - clean)
        background_effect = np.abs(margins[role]["far_background"] - clean)
        result["roles"][role] = {
            "object_mean_absolute_margin_change": float(object_effect.mean()),
            "far_background_mean_absolute_margin_change": float(background_effect.mean()),
            "object_minus_background": float(object_effect.mean() - background_effect.mean()),
            "object_more_causal": bool(object_effect.mean() > background_effect.mean()),
        }
    return result


def _write_event_manifest(path: Path, events: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "condition",
        "local_index",
        "sample_index",
        "source_stem",
        "object_index",
        "target",
        "control_prediction",
        "candidate_prediction",
        "categories",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["categories"] = ",".join(str(value) for value in event["categories"])
            writer.writerow({key: row.get(key, "") for key in fields})


def _approx(value: object, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _run_provenance(
    *,
    control_run: Path,
    candidate_run: Path,
    preflight: Mapping[str, object],
    fold_summary: Mapping[str, object],
) -> Dict[str, object]:
    run_payloads = {}
    occurrence = {}
    checks: Dict[str, bool] = {
        "preflight_permission": bool(preflight.get("gate", {}).get("formal_pair_permission")),
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fold_zero_source_overlap": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_raw_unmodified": fold_summary.get("raw_data_modified") is False,
        "fold_test_is_only_compatibility_mirror": fold_summary.get("test_mirrors_holdout")
        is True,
    }
    exact_values = {
        "epochs": 5,
        "scheduler_total_epochs": 5,
        "patience": 3,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "seed": 42,
        "disable_balanced_epoch_sampling": True,
        "learning_rate": 2.5e-4,
        "min_learning_rate": 1e-6,
        "warmup_epochs": 1,
        "weight_decay": 0.05,
        "attention_view_loss_weight": 0.0,
        "attention_crop_probability": 0.0,
        "attention_drop_probability": 0.0,
        "teacher_focus_binary_loss_weight": 0.0,
        "bbox_spatial_fusion": True,
        "data_cartography": True,
        "skip_final_test": True,
        "no_pretrained": True,
        "resume_checkpoint": "",
    }
    for role, run_dir in (("control", control_run), ("candidate", candidate_run)):
        launcher = _load_json(run_dir / "launcher_args.json")
        summary = _load_json(run_dir / "summary.json")
        resolved = _load_json(run_dir / "resolved_config.json")
        history_rows = list(
            csv.DictReader(
                (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="")
            )
        )
        occurrence_path = run_dir / "data_cartography_train_occurrence_hashes.json"
        occurrence[role] = _load_json(occurrence_path)
        role_checks = {}
        for key, expected in exact_values.items():
            observed = launcher.get(key)
            role_checks[f"launcher_{key}"] = (
                _approx(observed, expected)
                if isinstance(expected, float)
                else observed == expected
            )
        role_checks.update(
            {
                "five_history_epochs": len(history_rows) == 5,
                "last_epoch_exact": int(summary.get("last_epoch", -1)) == 5,
                "test_summary_absent": summary.get("test_summary") is None,
                "resolved_resume_not_loaded": not bool(
                    resolved.get("resume", {}).get("loaded", True)
                ),
                "checkpoint_exists": (run_dir / "checkpoints" / "best.pt").is_file(),
            }
        )
        expected_faa = role == "candidate"
        role_checks["launcher_faa_role"] = bool(
            launcher.get("foveal_aggregated_attention", False)
        ) == expected_faa
        role_checks["launcher_faa_layer"] = str(
            launcher.get("foveal_aggregated_attention_layers", "")
        ) == "1"
        checks.update({f"{role}_{key}": bool(value) for key, value in role_checks.items()})
        run_payloads[role] = {
            "launcher_args_sha256": _sha256(run_dir / "launcher_args.json"),
            "resolved_config_sha256": _sha256(run_dir / "resolved_config.json"),
            "summary": summary,
            "history_rows": len(history_rows),
            "occurrence_path": str(occurrence_path.resolve()),
            "checks": role_checks,
        }
    control_epochs = occurrence["control"].get("epochs", [])
    candidate_epochs = occurrence["candidate"].get("epochs", [])
    checks["occurrence_epoch_records_equal"] = control_epochs == candidate_epochs
    checks["occurrence_five_epochs"] = len(control_epochs) == len(candidate_epochs) == 5
    checks["occurrence_rows_exact"] = all(
        int(row.get("occurrences", -1)) == EXPECTED_FIT_ROWS
        and int(row.get("unique_sample_indices", -1)) == EXPECTED_FIT_ROWS
        and row.get("class_counts") == EXPECTED_FIT_COUNTS
        for row in control_epochs
    )
    tracked_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    upstream = subprocess.check_output(
        ["git", "rev-parse", "@{upstream}"], text=True
    ).strip()
    checks["tracked_worktree_clean"] = tracked_status == ""
    checks["head_pushed"] = head == upstream
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "repository_head": head,
        "repository_upstream": upstream,
        "runs": run_payloads,
        "occurrence": occurrence,
    }


def _gate_checks(
    *,
    provenance: Mapping[str, object],
    comparisons: Mapping[str, object],
    selectivity: Mapping[str, object],
    routes: Mapping[str, object],
    perturbation: Mapping[str, object],
    xai: Mapping[str, object],
) -> Dict[str, bool]:
    clean = comparisons["clean"]
    clean_delta = clean["delta"]
    clean_candidate = clean["candidate"]
    transitions = clean["transitions"]
    selectivity_conditions = selectivity["conditions"]
    checks: Dict[str, bool] = {
        "provenance_and_training": bool(provenance["all_checks_pass"]),
        "clean_macro_f1_delta_gte_0p003": float(clean_delta["macro_f1"]) >= 0.003,
        "clean_class1_f1_delta_gte_0p015": float(clean_delta["class1_f1"]) >= 0.015,
        "clean_candidate_class1_f1_gte_0p60": float(
            clean_candidate["per_class_f1"][FOCUS_CLASS]
        )
        >= 0.60,
        "clean_class1_precision_delta_gte_0p020": float(
            clean_delta["class1_precision"]
        )
        >= 0.020,
        "clean_class1_recall_delta_gte_minus_0p010": float(clean_delta["class1_recall"])
        >= -0.010,
        "clean_restricted_fp_reduction_gte_3": int(
            transitions["restricted_focus_fp_reduction"]
        )
        >= 3,
        "clean_corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "clean_fn_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
        >= int(transitions["focus_tp_break"]),
        "clean_maximum_nonfocus_drop_lte_0p010": float(
            clean["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
    }
    clean_selectivity = selectivity_conditions["clean"]
    checks.update(
        {
            "clean_selectivity_candidate_auroc_gte_0p65": float(
                clean_selectivity["candidate"]["auroc"]
            )
            >= 0.65,
            "clean_selectivity_delta_gte_0p020": float(clean_selectivity["delta"])
            >= 0.020,
        }
    )
    shifted = [name for name, _, _ in CONDITIONS[1:]]
    clean_candidate_auroc = float(clean_selectivity["candidate"]["auroc"])
    checks["shifted_auroc_candidate_gte_0p60"] = all(
        float(selectivity_conditions[name]["candidate"]["auroc"]) >= 0.60
        for name in shifted
    )
    checks["shifted_auroc_not_below_control"] = all(
        float(selectivity_conditions[name]["delta"]) >= 0.0 for name in shifted
    )
    checks["shifted_auroc_drop_lte_0p10"] = all(
        clean_candidate_auroc
        - float(selectivity_conditions[name]["candidate"]["auroc"])
        <= 0.10
        for name in shifted
    )
    precision_deltas = [
        float(comparisons[name]["delta"]["class1_precision"]) for name in shifted
    ]
    recall_deltas = [
        float(comparisons[name]["delta"]["class1_recall"]) for name in shifted
    ]
    shifted_fp_reductions = [
        int(comparisons[name]["transitions"]["restricted_focus_fp_reduction"])
        for name in shifted
    ]
    f1_deltas = [
        float(comparisons[name]["delta"]["class1_f1"])
        for name, _, _ in CONDITIONS
    ]
    checks.update(
        {
            "shifted_precision_not_lower": all(value >= 0.0 for value in precision_deltas),
            "shifted_precision_gain_gte_0p010_in_2_of_3": sum(
                value >= 0.010 for value in precision_deltas
            )
            >= 2,
            "shifted_recall_delta_gte_minus_0p020": all(
                value >= -0.020 for value in recall_deltas
            ),
            "shifted_restricted_fp_never_increases": all(
                value >= 0 for value in shifted_fp_reductions
            ),
            "shifted_restricted_fp_decreases_in_aggregate": sum(shifted_fp_reductions) > 0,
            "class1_f1_nonnegative_in_3_of_4": sum(value >= 0.0 for value in f1_deltas)
            >= 3,
            "class1_f1_no_delta_below_minus_0p010": min(f1_deltas) >= -0.010,
        }
    )
    clean_route = routes["clean"]
    checks.update(
        {
            "faa_routes_finite": bool(clean_route["finite"]),
            "faa_local_route_noncollapsed": 0.05
            < float(clean_route["local_mass_mean"])
            < 0.95,
            "faa_pooled_route_noncollapsed": 0.05
            < float(clean_route["pooled_mass_mean"])
            < 0.95,
            "faa_dual_route_fraction_gte_0p95": float(clean_route["dual_route_fraction"])
            >= 0.95,
            "xai_stem_foreground_delta_gte_minus_0p05": float(
                xai["stem_foreground_mass_delta"]
            )
            >= -0.05,
            "xai_block1_foreground_delta_gte_minus_0p05": float(
                xai["block1_foreground_mass_delta"]
            )
            >= -0.05,
            "xai_all_events_covered": bool(xai["all_event_rows_covered"]),
            "xai_all_representatives_covered": bool(xai["all_representatives_covered"]),
            "xai_maps_finite": bool(xai["maps_finite"]),
            "object_perturbation_more_causal_than_background": bool(
                perturbation["roles"]["candidate"]["object_more_causal"]
            ),
        }
    )
    return checks


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"FAA audit summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual review finalization requires --visual-review-result and note.")
    summary = _load_json(summary_path)
    pages = summary.get("xai", {}).get("render", {}).get("pages", [])
    if not pages or not all(Path(str(path)).is_file() for path in pages):
        raise ValueError("Visual review cannot be finalized without every XAI page.")
    passed = args.visual_review_result == "pass"
    review = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": args.visual_review_result,
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "page_count": len(pages),
        "pages_sha256": {str(path): _sha256(Path(str(path))) for path in pages},
    }
    _write_json(output_dir / "visual_review.json", review)
    gate = summary["gate"]
    gate["visual_review_completed"] = True
    gate["visual_review_passed"] = passed
    gate["promotion_permission"] = bool(gate["automated_pass"] and passed)
    summary["visual_review"] = review
    _write_json(summary_path, summary)
    return summary


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.seed) != 42 or int(args.batch_size) != 32:
        raise ValueError("FAA pair audit is locked to seed=42 and batch_size=32.")
    if int(args.xai_batch_size) <= 0 or int(args.xai_batch_size) > 4:
        raise ValueError("FAA XAI batch size must be in [1,4].")
    output_dir = _prepare_output(args.output_dir)
    declaration = Path(args.declaration).resolve()
    if _sha256(declaration) != LOCKED_DECLARATION_SHA256:
        raise ValueError("FAA declaration hash differs from the locked protocol.")
    rows = _read_clean_train_rows(declaration)
    holdout_rows = [row for row in rows if row.fold == FOLD]
    fit_rows = [row for row in rows if row.fold != FOLD]
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(fit_rows) != EXPECTED_FIT_ROWS:
        raise ValueError("FAA fit/holdout row counts differ from the protocol.")
    if _ordered_index_sha256([row.sample_index for row in fit_rows]) != EXPECTED_FIT_INDEX_SHA256:
        raise ValueError("FAA fit sample-index hash differs.")
    if (
        _ordered_index_sha256([row.sample_index for row in holdout_rows])
        != EXPECTED_HOLDOUT_INDEX_SHA256
    ):
        raise ValueError("FAA holdout sample-index hash differs.")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("FAA formal pair audit requires CUDA.")
    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    control, control_checkpoint, control_classes = load_model(
        Path(args.control_checkpoint), device
    )
    candidate, candidate_checkpoint, candidate_classes = load_model(
        Path(args.candidate_checkpoint), device
    )
    if control_classes != candidate_classes or len(control_classes) != 5:
        raise ValueError("FAA pair checkpoint class orders differ.")
    control_config = control_checkpoint.get("model_config", {})
    candidate_config = candidate_checkpoint.get("model_config", {})
    if bool(control_config.get("foveal_aggregated_attention", False)):
        raise ValueError("FAA control checkpoint unexpectedly enables FAA.")
    if not bool(candidate_config.get("foveal_aggregated_attention", False)):
        raise ValueError("FAA candidate checkpoint does not enable FAA.")
    if not isinstance(candidate.blocks[0].attn, FovealAggregatedAttention):
        raise ValueError("FAA candidate checkpoint did not reconstruct block-1 FAA.")
    if _eval_semantics(control_checkpoint) != _eval_semantics(candidate_checkpoint):
        raise ValueError("FAA pair evaluation semantics differ.")

    dataset, transform, dataset_summary = _build_holdout_dataset(
        control_checkpoint,
        fold_data=Path(args.fold_data),
        holdout_rows=holdout_rows,
    )
    preflight = _load_json(Path(args.preflight_summary))
    fold_summary = _load_json(Path(args.fold_summary))
    provenance = _run_provenance(
        control_run=Path(args.control_run_dir),
        candidate_run=Path(args.candidate_run_dir),
        preflight=preflight,
        fold_summary=fold_summary,
    )
    predictions, loader_summaries, routes = _predict_conditions(
        control=control,
        candidate=candidate,
        base_dataset=dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    prediction_path = output_dir / "predictions_all_conditions.csv"
    _write_predictions(prediction_path, predictions)
    comparisons = _comparisons(predictions)
    selectivity = _selectivity(rows=rows, predictions=predictions)
    events = _build_event_manifest(predictions)
    _write_event_manifest(output_dir / "event_manifest.csv", events)
    representatives = _representative_requests(
        dataset=dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=selectivity["positive_sample_indices"],
    )
    requests = _merge_xai_requests(events, representatives)
    semantics = dataset_summary["semantics"]
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    control_maps, control_xai = _collect_xai_maps(
        role="control",
        model=control,
        base_dataset=dataset,
        transform=transform,
        requests=requests,
        device=device,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        mean=mean,
        std=std,
    )
    candidate_maps, candidate_xai = _collect_xai_maps(
        role="candidate",
        model=candidate,
        base_dataset=dataset,
        transform=transform,
        requests=requests,
        device=device,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        mean=mean,
        std=std,
    )
    render = _render_xai_pages(
        output_dir=output_dir,
        requests=requests,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
    )
    xai = _xai_summary(
        requests=requests,
        events=events,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
        render=render,
    )
    xai["control_collection"] = control_xai
    xai["candidate_collection"] = candidate_xai
    perturbation = _perturbation_audit(
        control=control,
        candidate=candidate,
        base_dataset=dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        selectivity=selectivity,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    automated_checks = _gate_checks(
        provenance=provenance,
        comparisons=comparisons,
        selectivity=selectivity,
        routes=routes,
        perturbation=perturbation,
        xai=xai,
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
            "control_checkpoint_sha256": _sha256(Path(args.control_checkpoint)),
            "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
            "candidate_checkpoint_sha256": _sha256(Path(args.candidate_checkpoint)),
            "declaration": str(declaration),
            "declaration_sha256": _sha256(declaration),
            "fold_data": str(Path(args.fold_data).resolve()),
            "fold_summary": str(Path(args.fold_summary).resolve()),
            "preflight_summary": str(Path(args.preflight_summary).resolve()),
        },
        "dataset": dataset_summary,
        "provenance": provenance,
        "loader_summaries": loader_summaries,
        "comparisons": comparisons,
        "selectivity": selectivity,
        "routes": routes,
        "perturbation": perturbation,
        "events": {
            "rows": len(events),
            "manifest": str((output_dir / "event_manifest.csv").resolve()),
        },
        "xai": xai,
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": _sha256(prediction_path),
            "rows": sum(
                len(predictions[role][condition])
                for role in ("control", "candidate")
                for condition, _, _ in CONDITIONS
            ),
        },
        "validation_predictions_used": False,
        "test_data_used": False,
        "official_validation_permission": False,
        "gate": {
            "automated_checks": automated_checks,
            "failed_automated_checks": sorted(
                name for name, passed in automated_checks.items() if not passed
            ),
            "automated_pass": all(automated_checks.values()),
            "visual_review_completed": False,
            "visual_review_passed": False,
            "promotion_permission": False,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "visual_review_required.json",
        {
            "status": "pending",
            "required": True,
            "page_count": int(render["page_count"]),
            "finalize_command": (
                "python -m trkh.tools.audit_foveal_aggregated_attention_pair "
                f"--output-dir \"{output_dir}\" --finalize-visual-review "
                "--visual-review-result pass|fail --visual-review-note \"...\""
            ),
        },
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = (
        _finalize_visual_review(args)
        if bool(args.finalize_visual_review)
        else run_audit(args)
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True), flush=True)
    if bool(args.finalize_visual_review):
        return
    if not bool(summary["gate"]["automated_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
