from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_model
from trkh.models.deformable_spatial_attention import DeformableSpatialAttention
from trkh.models.model import classification_logits_from_features
from trkh.tools import audit_foveal_aggregated_attention_pair as common
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _batched_gradcam,
    _heat_overlay,
    _normalize_maps,
    _rgb_from_tensor,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)


METHOD = "deformable_spatial_attention_a1_pair"
FOCUS_CLASS = 1
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
LOCKED_FOLD_SUMMARY_SHA256 = (
    "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
)
CONDITIONS = common.CONDITIONS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only post-run gate for the matched block-2 DAT A1 pair. "
            "Official validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path("runs/probe_dat_a1_control_5e_20260716/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("runs/probe_dat_a1_candidate_5e_20260716/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        default=Path("runs/probe_dat_a1_control_5e_20260716"),
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=Path("runs/probe_dat_a1_candidate_5e_20260716"),
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("runs/dat_a1_pair_manifest_20260716.json"),
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
        default=Path("runs/audit_dat_preflight_a1_fix_20260716/summary.json"),
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


def _attention_key_map(
    attention: Tensor,
    *,
    prefix_count: int,
    grid_size: Tuple[int, int],
) -> Tensor:
    """Aggregate all block queries over patch-key columns."""

    patch_count = int(grid_size[0] * grid_size[1])
    if attention.ndim != 4 or int(attention.size(-1)) < prefix_count + patch_count:
        raise ValueError("Block attention cannot produce the locked patch-key map.")
    values = attention[
        :, :, :, prefix_count : prefix_count + patch_count
    ].float().mean(dim=(1, 2)).clamp_min(0.0)
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return values.reshape(-1, grid_size[0], grid_size[1])


def _standard_logits_with_block2_trace(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
    *,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, object]]:
    """Keep behavior on the deployment path while collecting audit-only trace."""

    logits, _ = _forward_classification_with_metadata(
        model,
        images,
        metadata,
        device=device,
    )
    features = model.forward_features(
        images,
        image_valid_mask=metadata.get("image_mask"),
        bbox_token_prior=metadata.get("bbox"),
        return_attention=True,
        attention_layers=[1],
        return_trace=True,
    )
    if not isinstance(features, dict):
        raise TypeError("DAT attention trace must be a feature dictionary.")
    if torch.is_tensor(metadata.get("bbox")):
        features["bbox"] = metadata["bbox"]
    return logits, features


def _position_bbox_statistics(
    positions: Tensor,
    references: Tensor,
    bboxes: Tensor,
) -> Dict[str, Tensor]:
    """Measure DAT sample positions against transformed input-space bboxes."""

    if positions.ndim != 5 or positions.shape != references.shape:
        raise ValueError("DAT position/reference tensors must have shape [B,G,H,W,2].")
    if bboxes.ndim != 2 or tuple(bboxes.shape) != (int(positions.size(0)), 4):
        raise ValueError("DAT bbox tensor must have shape [B,4].")

    def measure(points: Tensor) -> Tuple[Tensor, Tensor]:
        point_y = (points[..., 0].float() + 1.0) * 0.5
        point_x = (points[..., 1].float() + 1.0) * 0.5
        center_x, center_y, width, height = bboxes.float().unbind(dim=1)
        shape = (-1, 1, 1, 1)
        left = (center_x - width * 0.5).reshape(shape)
        right = (center_x + width * 0.5).reshape(shape)
        top = (center_y - height * 0.5).reshape(shape)
        bottom = (center_y + height * 0.5).reshape(shape)
        inside = (
            point_x.ge(left)
            & point_x.le(right)
            & point_y.ge(top)
            & point_y.le(bottom)
        )
        dx = torch.maximum(
            torch.maximum(left - point_x, point_x - right),
            torch.zeros_like(point_x),
        )
        dy = torch.maximum(
            torch.maximum(top - point_y, point_y - bottom),
            torch.zeros_like(point_y),
        )
        distance = torch.sqrt(dx.square() + dy.square())
        outside = ~inside
        outside_count = outside.sum(dim=(1, 2, 3)).clamp_min(1)
        outside_mean = (
            distance.mul(outside).sum(dim=(1, 2, 3)) / outside_count
        )
        return inside.float().mean(dim=(1, 2, 3)), outside_mean

    hit, outside = measure(positions)
    reference_hit, reference_outside = measure(references)
    return {
        "bbox_hit_fraction": hit,
        "reference_bbox_hit_fraction": reference_hit,
        "bbox_hit_delta": hit - reference_hit,
        "outside_distance": outside,
        "reference_outside_distance": reference_outside,
        "outside_distance_reduction_fraction": (
            reference_outside - outside
        ) / reference_outside.clamp_min(1e-9),
    }


def _candidate_spatial_batch(
    module: DeformableSpatialAttention,
    *,
    crop_bboxes: Tensor,
) -> Dict[str, Tensor]:
    trace = module.trace()
    positions = trace.get("positions")
    references = trace.get("reference_positions")
    offsets = trace.get("offsets")
    sampled_attention = trace.get("sample_attention")
    if not all(
        torch.is_tensor(value)
        for value in (positions, references, offsets, sampled_attention)
    ):
        raise ValueError("DAT trace lacks positions, references, offsets, or attention.")
    positions = positions.float()
    references = references.float()
    offsets = offsets.float()
    batch_size, groups, height, width, _ = positions.shape
    valid_mass = module.sampled_valid_interpolation_mass(
        sampled_attention.float(),
        positions.reshape(batch_size * groups, height, width, 2),
    )
    result = _position_bbox_statistics(positions, references, crop_bboxes)
    result.update(
        {
            "offset_rms": offsets.square().mean(dim=(1, 2, 3, 4)).sqrt(),
            "offset_rms_per_group": offsets.square().mean(dim=(2, 3, 4)).sqrt(),
            "inter_group_position_rms": (
                positions[:, 1:] - positions[:, :-1]
            ).square().mean(dim=(1, 2, 3, 4)).sqrt(),
            "valid_interpolation_mass": valid_mass.mean(dim=(1, 2)),
            "positions": positions,
            "references": references,
        }
    )
    return result


def _prediction_fields() -> Tuple[str, ...]:
    return (
        "block2_attention_bbox_mass",
        "bbox_hit_fraction",
        "reference_bbox_hit_fraction",
        "bbox_hit_delta",
        "outside_distance",
        "reference_outside_distance",
        "outside_distance_reduction_fraction",
        "offset_rms",
        "offset_rms_group_0",
        "offset_rms_group_1",
        "inter_group_position_rms",
        "valid_interpolation_mass",
    )


def _predict_conditions_and_spatial(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset,
    transform,
    holdout_rows: Sequence[CleanTrainRow],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, list[Dict[str, object]]]], Dict[str, object], Dict[str, object]]:
    output: Dict[str, Dict[str, list[Dict[str, object]]]] = {
        "control": {},
        "candidate": {},
    }
    loader_summaries: Dict[str, object] = {}
    spatial_rows: Dict[str, list[Dict[str, object]]] = {}
    local_indices = list(range(len(base_dataset)))
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, brightness, contrast) in enumerate(CONDITIONS):
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            local_indices,
            corruption=common._condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"dat_pair_{condition}_holdout",
            seed=seed + 100 + condition_index,
        )
        loader_summaries[condition] = loader_summary
        role_rows: Dict[str, list[Dict[str, object]]] = {
            "control": [],
            "candidate": [],
        }
        condition_spatial: list[Dict[str, object]] = []
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = common._metadata_to_device(metadata_cpu, device)
                sample_indices = metadata_cpu.get("sample_index")
                crop_bboxes = metadata.get("crop_bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bboxes):
                    raise ValueError("DAT condition metadata lacks sample_index/crop_bbox.")
                role_payloads = []
                for role, model in (("control", control), ("candidate", candidate)):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        logits, features = _standard_logits_with_block2_trace(
                            model,
                            images,
                            metadata,
                            device=device,
                        )
                    attention = features.get("attentions", {}).get(1)
                    if not torch.is_tensor(attention):
                        raise ValueError(f"{role} block-2 attention is missing.")
                    grid_size = tuple(int(value) for value in features["grid_size"])
                    key_heat = _attention_key_map(
                        attention,
                        prefix_count=int(model.num_prefix_tokens),
                        grid_size=grid_size,
                    )
                    attention_mass = common._bbox_foreground_mass(
                        key_heat, crop_bboxes
                    )
                    spatial = None
                    if role == "candidate":
                        module = model.blocks[1].attn
                        if not isinstance(module, DeformableSpatialAttention):
                            raise TypeError("Candidate block 2 is not DAT.")
                        spatial = _candidate_spatial_batch(
                            module,
                            crop_bboxes=crop_bboxes,
                        )
                    role_payloads.append(
                        (role, logits.float(), attention_mass, spatial)
                    )
                for role, logits, attention_mass, spatial in role_payloads:
                    probabilities = logits.softmax(dim=1)
                    margins = common._margin(logits)
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
                            "block2_attention_bbox_mass": float(attention_mass[position]),
                        }
                        for name in _prediction_fields()[1:]:
                            row[name] = float("nan")
                        if spatial is not None:
                            for name in _prediction_fields()[1:8]:
                                row[name] = float(spatial[name][position].item())
                            group_rms = spatial["offset_rms_per_group"][position]
                            row["offset_rms_group_0"] = float(group_rms[0].item())
                            row["offset_rms_group_1"] = float(group_rms[1].item())
                            row["inter_group_position_rms"] = float(
                                spatial["inter_group_position_rms"][position].item()
                            )
                            row["valid_interpolation_mass"] = float(
                                spatial["valid_interpolation_mass"][position].item()
                            )
                            condition_spatial.append(
                                {
                                    "condition": condition,
                                    "local_index": int(local_index),
                                    "sample_index": int(source.sample_index),
                                    **{
                                        name: row[name]
                                        for name in _prediction_fields()
                                    },
                                    "control_block2_attention_bbox_mass": float(
                                        role_payloads[0][2][position]
                                    ),
                                }
                            )
                        for class_index in range(5):
                            row[f"logit_{class_index}"] = float(
                                logits[position, class_index].item()
                            )
                            row[f"prob_{class_index}"] = float(
                                probabilities[position, class_index].item()
                            )
                        role_rows[role].append(row)
        for role in ("control", "candidate"):
            observed = [int(row["local_index"]) for row in role_rows[role]]
            if observed != local_indices:
                raise ValueError(f"Prediction order differs for {role}/{condition}.")
            output[role][condition] = role_rows[role]
        if [int(row["local_index"]) for row in condition_spatial] != local_indices:
            raise ValueError(f"DAT spatial row order differs for {condition}.")
        spatial_rows[condition] = condition_spatial
    return output, loader_summaries, _spatial_summary(spatial_rows)


def _spatial_summary(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    cohort_sample_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    cohort = set(int(value) for value in (cohort_sample_indices or ()))
    result: Dict[str, object] = {"conditions": {}}
    for condition, _, _ in CONDITIONS:
        rows = list(rows_by_condition[condition])
        metrics: Dict[str, object] = {"rows": len(rows)}
        for key in (
            "bbox_hit_fraction",
            "reference_bbox_hit_fraction",
            "bbox_hit_delta",
            "outside_distance",
            "reference_outside_distance",
            "outside_distance_reduction_fraction",
            "offset_rms",
            "offset_rms_group_0",
            "offset_rms_group_1",
            "inter_group_position_rms",
            "valid_interpolation_mass",
            "block2_attention_bbox_mass",
            "control_block2_attention_bbox_mass",
        ):
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            metrics[f"{key}_mean"] = float(values.mean())
            metrics[f"{key}_min"] = float(values.min())
            metrics[f"{key}_max"] = float(values.max())
            metrics[f"{key}_finite"] = bool(np.isfinite(values).all())
        metrics["positive_bbox_hit_delta_fraction"] = float(
            np.mean([float(row["bbox_hit_delta"]) > 0.0 for row in rows])
        )
        result["conditions"][condition] = metrics
    if cohort:
        clean = [
            row
            for row in rows_by_condition["clean"]
            if int(row["sample_index"]) in cohort
        ]
        if len(clean) != len(cohort):
            raise ValueError("Frozen spatial cohort coverage differs.")
        candidate = np.asarray(
            [float(row["block2_attention_bbox_mass"]) for row in clean]
        )
        control = np.asarray(
            [float(row["control_block2_attention_bbox_mass"]) for row in clean]
        )
        result["frozen_cohort"] = {
            "rows": len(clean),
            "control_block2_attention_bbox_mass": float(control.mean()),
            "candidate_block2_attention_bbox_mass": float(candidate.mean()),
            "delta": float(candidate.mean() - control.mean()),
            "candidate_not_lower_fraction": float(np.mean(candidate >= control)),
        }
    return result


def _representative_requests(
    *,
    dataset,
    holdout_rows: Sequence[CleanTrainRow],
    positive_sample_indices: Sequence[int],
) -> list[Dict[str, object]]:
    requests = common._representative_requests(
        dataset=dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=positive_sample_indices,
    )
    areas = [
        (common._sample_bbox(dataset, index)[2] * common._sample_bbox(dataset, index)[3], index)
        for index in range(len(dataset))
    ]
    _, local_index = min(areas, key=lambda item: (item[0], item[1]))
    source = holdout_rows[int(local_index)]
    requests.append(
        {
            "condition": "clean",
            "local_index": int(local_index),
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "object_index": int(dataset.samples[int(local_index)].primary_object_index),
            "target": int(source.target),
            "control_prediction": -1,
            "candidate_prediction": -1,
            "categories": ["representative_tiny_object"],
            "geometry": {"area": float(areas[int(local_index)][0])},
        }
    )
    return requests


def _position_density(positions: Tensor, grid_size: Tuple[int, int]) -> Tensor:
    if positions.ndim != 5 or int(positions.size(-1)) != 2:
        raise ValueError("DAT position density requires [B,G,H,W,2].")
    batch_size = int(positions.size(0))
    height, width = grid_size
    y = ((positions[..., 0].float() + 1.0) * 0.5 * height).floor()
    x = ((positions[..., 1].float() + 1.0) * 0.5 * width).floor()
    y = y.clamp(0, height - 1).to(dtype=torch.long)
    x = x.clamp(0, width - 1).to(dtype=torch.long)
    flat = y * width + x
    density = positions.new_zeros((batch_size, height * width), dtype=torch.float32)
    density.scatter_add_(
        1,
        flat.reshape(batch_size, -1),
        torch.ones_like(flat, dtype=torch.float32).reshape(batch_size, -1),
    )
    return _normalize_maps(density.reshape(batch_size, height, width))


def _collect_xai_maps(
    *,
    role: str,
    model: nn.Module,
    base_dataset,
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
    condition_specs = {
        name: (brightness, contrast) for name, brightness, contrast in CONDITIONS
    }
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, selected) in enumerate(by_condition.items()):
        unique_indices = sorted(set(selected))
        brightness, contrast = condition_specs[condition]
        condition_dataset = _SelectedConditionDataset(
            base_dataset,
            unique_indices,
            corruption=common._condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"dat_{role}_{condition}_xai",
            seed=seed + 500 + condition_index,
        )
        loaders[condition] = loader_summary
        for images_cpu, _, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True).requires_grad_(True)
            metadata = common._metadata_to_device(metadata_cpu, device)
            sample_indices = metadata_cpu.get("sample_index")
            crop_bbox_cpu = metadata_cpu.get("crop_bbox")
            crop_bbox = metadata.get("crop_bbox")
            if not all(
                torch.is_tensor(value)
                for value in (sample_indices, crop_bbox_cpu, crop_bbox)
            ):
                raise ValueError("DAT XAI metadata lacks sample_index/crop_bbox.")
            captured: Dict[str, Tensor] = {}

            def stem_hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError("Stem XAI hook requires a tensor output.")
                output.retain_grad()
                captured["stem"] = output

            def block_hook(_module, _inputs, output):
                tokens = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tokens):
                    raise TypeError("Block-2 XAI hook requires tensor tokens.")
                tokens.retain_grad()
                captured["block"] = tokens

            stem_handle = model.stem.register_forward_hook(stem_hook)
            block_handle = model.blocks[1].register_forward_hook(block_hook)
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
                    attention_layers=[1],
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
                raise RuntimeError("DAT stem/block-2 XAI gradients were not retained.")
            grid_size = tuple(int(value) for value in features["grid_size"])
            prefix_count = int(model.num_prefix_tokens)
            stem_heat = _batched_gradcam(
                stem,
                stem.grad,
                size=(int(images.size(-2)), int(images.size(-1))),
            ).cpu()
            block_heat = common._token_gradcam(
                block,
                block.grad,
                prefix_count=prefix_count,
                grid_size=grid_size,
            ).cpu()
            attention = features.get("attentions", {}).get(1)
            if not torch.is_tensor(attention):
                raise ValueError("Block-2 attention map is missing from DAT XAI features.")
            key_mass_map = _attention_key_map(
                attention,
                prefix_count=prefix_count,
                grid_size=grid_size,
            )
            key_heat = _normalize_maps(key_mass_map).cpu()
            stem_foreground = common._bbox_foreground_mass(stem_heat, crop_bbox_cpu)
            block_foreground = common._bbox_foreground_mass(block_heat, crop_bbox_cpu)
            key_foreground = common._bbox_foreground_mass(
                key_mass_map.cpu(), crop_bbox_cpu
            )
            positions = references = density = None
            if role == "candidate":
                module = model.blocks[1].attn
                if not isinstance(module, DeformableSpatialAttention):
                    raise TypeError("Candidate XAI block 2 is not DAT.")
                trace = module.trace()
                position_value = trace.get("positions")
                reference_value = trace.get("reference_positions")
                if not torch.is_tensor(position_value) or not torch.is_tensor(reference_value):
                    raise ValueError("Candidate DAT XAI lacks position traces.")
                positions = position_value.detach().float().cpu()
                references = reference_value.detach().float().cpu()
                density = _position_density(position_value, grid_size).cpu()
            probabilities = logits.detach().float().softmax(dim=1).cpu()
            for position, local_index in enumerate(sample_indices.tolist()):
                record: Dict[str, object] = {
                    "rgb": _rgb_from_tensor(images_cpu[position], mean=mean, std=std),
                    "bbox": crop_bbox_cpu[position].float().numpy(),
                    "stem_gradcam": stem_heat[position].numpy(),
                    "block2_gradcam": block_heat[position].numpy(),
                    "block2_attention_proxy": key_heat[position].numpy(),
                    "stem_foreground_mass": float(stem_foreground[position]),
                    "block2_foreground_mass": float(block_foreground[position]),
                    "block2_attention_foreground_mass": float(key_foreground[position]),
                    "prediction": int(probabilities[position].argmax().item()),
                }
                if positions is not None and references is not None and density is not None:
                    record["positions"] = positions[position].numpy()
                    record["references"] = references[position].numpy()
                    record["position_density"] = density[position].numpy()
                records[(condition, int(local_index))] = record
            del features, logits, images
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected = {(str(row["condition"]), int(row["local_index"])) for row in requests}
    if set(records) != expected:
        raise ValueError(f"{role} DAT XAI records differ from the request set.")
    names = ("stem_gradcam", "block2_gradcam", "block2_attention_proxy")
    finite = all(
        all(np.isfinite(np.asarray(row[name])).all() for name in names)
        for row in records.values()
    )
    if role == "candidate":
        finite = finite and all(
            np.isfinite(np.asarray(row["positions"])).all()
            and np.isfinite(np.asarray(row["position_density"])).all()
            for row in records.values()
        )
    return records, {"loaders": loaders, "finite": bool(finite), "rows": len(records)}


def _bbox_input(rgb: np.ndarray, bbox: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    center_x, center_y, box_width, box_height = [float(value) for value in bbox]
    left = int(round((center_x - box_width * 0.5) * width))
    right = int(round((center_x + box_width * 0.5) * width))
    top = int(round((center_y - box_height * 0.5) * height))
    bottom = int(round((center_y + box_height * 0.5) * height))
    draw.rectangle((left, top, right, bottom), outline=(0, 255, 0), width=2)
    return image


def _position_overlay(
    rgb: np.ndarray,
    bbox: np.ndarray,
    positions: np.ndarray,
    references: np.ndarray,
) -> Image.Image:
    image = _bbox_input(rgb, bbox)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    reference_points = np.asarray(references).reshape(-1, 2)
    for y, x in reference_points[::4]:
        px = int(round((float(x) + 1.0) * 0.5 * (width - 1)))
        py = int(round((float(y) + 1.0) * 0.5 * (height - 1)))
        draw.point((px, py), fill=(180, 180, 180))
    palette = ((255, 30, 30), (0, 220, 255), (255, 220, 0), (255, 0, 255))
    grouped = np.asarray(positions)
    for group_index, group in enumerate(grouped):
        color = palette[group_index % len(palette)]
        for y, x in group.reshape(-1, 2):
            px = int(round((float(x) + 1.0) * 0.5 * (width - 1)))
            py = int(round((float(y) + 1.0) * 0.5 * (height - 1)))
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
    return image


def _render_xai_pages(
    *,
    output_dir: Path,
    requests: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
) -> Dict[str, object]:
    columns = (
        "input+bbox",
        "control stem",
        "candidate stem",
        "control block2",
        "candidate block2",
        "control proxy",
        "candidate proxy",
        "DAT density",
        "DAT positions",
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
            bbox = np.asarray(candidate["bbox"], dtype=np.float32)
            views = [
                _bbox_input(rgb, bbox),
                _heat_overlay(rgb, np.asarray(control["stem_gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate["stem_gradcam"])),
                _heat_overlay(rgb, np.asarray(control["block2_gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate["block2_gradcam"])),
                _heat_overlay(rgb, np.asarray(control["block2_attention_proxy"])),
                _heat_overlay(rgb, np.asarray(candidate["block2_attention_proxy"])),
                _heat_overlay(rgb, np.asarray(candidate["position_density"])),
                _position_overlay(
                    rgb,
                    bbox,
                    np.asarray(candidate["positions"]),
                    np.asarray(candidate["references"]),
                ),
            ]
            y = header + row_position * (tile + label_height)
            for column, view in enumerate(views):
                canvas.paste(
                    view.resize((tile, tile), Image.Resampling.BILINEAR),
                    (column * tile, y),
                )
            categories = ",".join(str(value) for value in request["categories"])
            draw.text(
                (4, y + tile + 3),
                (
                    f"{request['condition']} idx={request['sample_index']} "
                    f"y={request['target']} c/a={request['control_prediction']}/"
                    f"{request['candidate_prediction']}"
                ),
                fill="black",
            )
            draw.text((4, y + tile + 23), f"[{categories}]", fill="black")
            manifest_rows.append(
                {**dict(request), "page": page_number, "page_row": row_position + 1}
            )
        path = output_dir / f"dat_xai_contact_sheet_{page_number:03d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    common._write_json(
        output_dir / "xai_page_manifest.json",
        {"rows": manifest_rows, "pages": pages},
    )
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
    categories = {
        str(category)
        for request in requests
        for category in request.get("categories", [])
    }
    required = {
        "representative_close",
        "representative_wide",
        "representative_partial",
        "representative_edge",
        "representative_tiny_object",
        "representative_lighting_dim",
        "representative_lighting_bright",
        "representative_low_contrast",
    }
    control_stem = np.asarray(
        [float(control_maps[key]["stem_foreground_mass"]) for key in keys]
    )
    candidate_stem = np.asarray(
        [float(candidate_maps[key]["stem_foreground_mass"]) for key in keys]
    )
    control_block = np.asarray(
        [float(control_maps[key]["block2_foreground_mass"]) for key in keys]
    )
    candidate_block = np.asarray(
        [float(candidate_maps[key]["block2_foreground_mass"]) for key in keys]
    )
    return {
        "event_rows": len(events),
        "xai_rows": len(requests),
        "all_event_rows_covered": event_keys.issubset(set(keys)),
        "representative_categories": sorted(categories.intersection(required)),
        "all_representatives_covered": required.issubset(categories),
        "control_stem_foreground_mass": float(control_stem.mean()),
        "candidate_stem_foreground_mass": float(candidate_stem.mean()),
        "stem_foreground_mass_delta": float(candidate_stem.mean() - control_stem.mean()),
        "control_block2_foreground_mass": float(control_block.mean()),
        "candidate_block2_foreground_mass": float(candidate_block.mean()),
        "block2_foreground_mass_delta": float(
            candidate_block.mean() - control_block.mean()
        ),
        "maps_finite": bool(
            all(
                np.isfinite(np.asarray(candidate_maps[key][name])).all()
                for key in keys
                for name in (
                    "stem_gradcam",
                    "block2_gradcam",
                    "block2_attention_proxy",
                    "position_density",
                    "positions",
                )
            )
        ),
        "render": dict(render),
    }


def _attach_frozen_cohort_spatial(
    spatial: Dict[str, object],
    *,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    selectivity: Mapping[str, object],
) -> None:
    cohort = {
        int(value)
        for value in (
            list(selectivity["positive_sample_indices"])
            + list(selectivity["hard_negative_sample_indices"])
        )
    }
    control_by_index = {
        int(row["sample_index"]): float(row["block2_attention_bbox_mass"])
        for row in predictions["control"]["clean"]
        if int(row["sample_index"]) in cohort
    }
    candidate_by_index = {
        int(row["sample_index"]): float(row["block2_attention_bbox_mass"])
        for row in predictions["candidate"]["clean"]
        if int(row["sample_index"]) in cohort
    }
    if set(control_by_index) != cohort or set(candidate_by_index) != cohort:
        raise ValueError("Frozen DAT attention cohort coverage differs.")
    ordered = sorted(cohort)
    control = np.asarray([control_by_index[index] for index in ordered])
    candidate = np.asarray([candidate_by_index[index] for index in ordered])
    spatial["frozen_cohort"] = {
        "rows": len(ordered),
        "ordered_sample_index_sha256": _ordered_index_sha256(ordered),
        "control_block2_attention_bbox_mass": float(control.mean()),
        "candidate_block2_attention_bbox_mass": float(candidate.mean()),
        "delta": float(candidate.mean() - control.mean()),
        "candidate_not_lower_fraction": float(np.mean(candidate >= control)),
    }


def _approx(value: object, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _run_provenance(
    *,
    control_run: Path,
    candidate_run: Path,
    pair_manifest_path: Path,
    preflight: Mapping[str, object],
    fold_summary: Mapping[str, object],
    fold_summary_path: Path,
) -> Dict[str, object]:
    pair_manifest = common._load_json(pair_manifest_path)
    preflight_head = str(preflight.get("git", {}).get("head", ""))
    recovery = pair_manifest.get("orchestration_recovery", {})
    recovery_manifest = Path(str(recovery.get("cleanup_manifest", "")))
    checks: Dict[str, bool] = {
        "preflight_permission": bool(
            preflight.get("gate", {}).get("formal_pair_permission")
        ),
        "pair_manifest_method": pair_manifest.get("method") == METHOD,
        "pair_manifest_head_matches_preflight": str(pair_manifest.get("git_head", ""))
        == preflight_head,
        "pair_manifest_validation_forbidden": pair_manifest.get(
            "official_validation_used"
        )
        is False,
        "pair_manifest_test_forbidden": pair_manifest.get("test_used") is False,
        "pair_manifest_commands_unchanged": pair_manifest.get(
            "current_best_command_updated"
        )
        is False,
        "orchestration_recovery_declared": recovery.get("required") is True,
        "orchestration_timeout_before_candidate_training": recovery.get(
            "candidate_training_started_before_timeout"
        )
        is False,
        "orchestration_exact_candidate_replay": recovery.get(
            "exact_candidate_config_replayed"
        )
        is True,
        "orchestration_cleanup_manifest_exists": recovery_manifest.is_file(),
        "orchestration_cleanup_manifest_hash": recovery_manifest.is_file()
        and common._sha256(recovery_manifest)
        == str(recovery.get("cleanup_manifest_sha256", "")),
        "fold_summary_hash": common._sha256(fold_summary_path)
        == LOCKED_FOLD_SUMMARY_SHA256,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_counts": fold_summary.get("fit_class_counts") == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fold_zero_source_overlap": int(fold_summary.get("source_overlap", -1)) == 0,
        "fold_raw_unmodified": fold_summary.get("raw_data_modified") is False,
        "fold_test_is_only_compatibility_mirror": fold_summary.get(
            "test_mirrors_holdout"
        )
        is True,
    }
    exact_values = {
        "scheduler_total_epochs": 5,
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
        "foveal_aggregated_attention": False,
        "deformable_spatial_attention_layers": "2",
        "deformable_spatial_attention_groups": 2,
        "deformable_spatial_attention_kernel_size": 5,
        "deformable_spatial_attention_offset_range": 2.0,
    }
    run_payloads = {}
    occurrence = {}
    normalized_train_args: Dict[str, list[str]] = {}
    for role, run_dir in (("control", control_run), ("candidate", candidate_run)):
        launcher = common._load_json(run_dir / "launcher_args.json")
        summary = common._load_json(run_dir / "summary.json")
        resolved = common._load_json(run_dir / "resolved_config.json")
        history_rows = list(
            csv.DictReader(
                (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="")
            )
        )
        occurrence_path = run_dir / "data_cartography_train_occurrence_hashes.json"
        occurrence[role] = common._load_json(occurrence_path)
        role_checks = {}
        for key, expected in exact_values.items():
            observed = launcher.get(key)
            role_checks[f"launcher_{key}"] = (
                _approx(observed, expected)
                if isinstance(expected, float)
                else observed == expected
            )
        train_args = [str(value) for value in launcher.get("train_args", [])]

        def cli_value(flag: str) -> Optional[str]:
            occurrences = [
                index for index, value in enumerate(train_args) if value == flag
            ]
            if len(occurrences) != 1 or occurrences[0] + 1 >= len(train_args):
                return None
            return train_args[occurrences[0] + 1]

        train_config = resolved.get("train_config", {})
        role_checks.update(
            {
                "launcher_train_args_epochs": cli_value("--epochs") == "5",
                "launcher_train_args_patience": cli_value("--patience") == "3",
                "resolved_epochs": int(train_config.get("epochs", -1)) == 5,
                "resolved_patience": int(
                    train_config.get("early_stopping_patience", -1)
                )
                == 3,
                "resolved_scheduler_total_epochs": int(
                    train_config.get("scheduler_total_epochs", -1)
                )
                == 5,
            }
        )
        dat_flag_count = train_args.count("--deformable-spatial-attention")
        if role == "candidate":
            role_checks.update(
                {
                    "train_args_dat_flag_once": dat_flag_count == 1,
                    "train_args_dat_layers": cli_value(
                        "--deformable-spatial-attention-layers"
                    )
                    == "2",
                    "train_args_dat_groups": cli_value(
                        "--deformable-spatial-attention-groups"
                    )
                    == "2",
                    "train_args_dat_kernel": cli_value(
                        "--deformable-spatial-attention-kernel-size"
                    )
                    == "5",
                    "train_args_dat_range": _approx(
                        cli_value("--deformable-spatial-attention-offset-range"),
                        2.0,
                    ),
                }
            )
        else:
            role_checks.update(
                {
                    "train_args_dat_flag_absent": dat_flag_count == 0,
                    "train_args_dat_options_absent": all(
                        cli_value(flag) is None
                        for flag in (
                            "--deformable-spatial-attention-layers",
                            "--deformable-spatial-attention-groups",
                            "--deformable-spatial-attention-kernel-size",
                            "--deformable-spatial-attention-offset-range",
                        )
                    ),
                }
            )
        normalized = list(train_args)
        run_index = normalized.index("--run-name")
        normalized[run_index + 1] = "<RUN>"
        for flag in (
            "--deformable-spatial-attention-layers",
            "--deformable-spatial-attention-groups",
            "--deformable-spatial-attention-kernel-size",
            "--deformable-spatial-attention-offset-range",
        ):
            if flag in normalized:
                option_index = normalized.index(flag)
                del normalized[option_index : option_index + 2]
        normalized = [
            value for value in normalized if value != "--deformable-spatial-attention"
        ]
        normalized_train_args[role] = normalized
        role_checks.update(
            {
                "five_history_epochs": len(history_rows) == 5,
                "last_epoch_exact": int(summary.get("last_epoch", -1)) == 5,
                "test_summary_absent": summary.get("test_summary") is None,
                "resolved_resume_not_loaded": not bool(
                    resolved.get("resume", {}).get("loaded", True)
                ),
                "checkpoint_exists": (run_dir / "checkpoints" / "best.pt").is_file(),
                "launcher_dat_role": bool(
                    launcher.get("deformable_spatial_attention", False)
                )
                == (role == "candidate"),
            }
        )
        checks.update({f"{role}_{key}": bool(value) for key, value in role_checks.items()})
        run_payloads[role] = {
            "launcher_args_sha256": common._sha256(run_dir / "launcher_args.json"),
            "resolved_config_sha256": common._sha256(run_dir / "resolved_config.json"),
            "summary": summary,
            "history_rows": len(history_rows),
            "occurrence_path": str(occurrence_path.resolve()),
            "checks": role_checks,
        }
    control_epochs = occurrence["control"].get("epochs", [])
    candidate_epochs = occurrence["candidate"].get("epochs", [])
    checks["occurrence_epoch_records_equal"] = control_epochs == candidate_epochs
    checks["normalized_train_args_equal_except_dat_role"] = (
        normalized_train_args["control"] == normalized_train_args["candidate"]
    )
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
    checks["audit_head_matches_pair"] = head == str(pair_manifest.get("git_head", ""))
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "repository_head": head,
        "repository_upstream": upstream,
        "pair_manifest": pair_manifest,
        "pair_manifest_sha256": common._sha256(pair_manifest_path),
        "runs": run_payloads,
        "occurrence": occurrence,
    }


def _gate_checks(
    *,
    provenance: Mapping[str, object],
    comparisons: Mapping[str, object],
    selectivity: Mapping[str, object],
    spatial: Mapping[str, object],
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
        "clean_class1_precision_delta_gte_0p025": float(
            clean_delta["class1_precision"]
        )
        >= 0.025,
        "clean_class1_recall_delta_gte_minus_0p010": float(clean_delta["class1_recall"])
        >= -0.010,
        "clean_restricted_fp_reduction_gte_4": int(
            transitions["restricted_focus_fp_reduction"]
        )
        >= 4,
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
    fp_reductions = [
        int(comparisons[name]["transitions"]["restricted_focus_fp_reduction"])
        for name in shifted
    ]
    f1_deltas = [
        float(comparisons[name]["delta"]["class1_f1"]) for name, _, _ in CONDITIONS
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
                value >= 0 for value in fp_reductions
            ),
            "shifted_restricted_fp_decreases_in_aggregate": sum(fp_reductions) > 0,
            "class1_f1_nonnegative_in_3_of_4": sum(value >= 0.0 for value in f1_deltas)
            >= 3,
            "class1_f1_no_delta_below_minus_0p010": min(f1_deltas) >= -0.010,
        }
    )
    spatial_conditions = spatial["conditions"]
    checks["spatial_offset_rms_finite_nonzero_lt_0p20"] = all(
        bool(spatial_conditions[name]["offset_rms_finite"])
        and 0.0 < float(spatial_conditions[name]["offset_rms_mean"]) < 0.20
        for name, _, _ in CONDITIONS
    )
    checks["spatial_valid_mass_gte_0p95"] = all(
        float(spatial_conditions[name]["valid_interpolation_mass_mean"]) >= 0.95
        for name, _, _ in CONDITIONS
    )
    checks["spatial_inter_group_mean_gte_0p005"] = all(
        float(spatial_conditions[name]["inter_group_position_rms_mean"]) >= 0.005
        for name, _, _ in CONDITIONS
    )
    checks["spatial_no_identical_groups"] = all(
        float(spatial_conditions[name]["inter_group_position_rms_min"]) > 0.0
        for name, _, _ in CONDITIONS
    )
    clean_spatial = spatial_conditions["clean"]
    checks["spatial_clean_bbox_hit_delta_gte_0p020"] = float(
        clean_spatial["bbox_hit_delta_mean"]
    ) >= 0.020
    checks["spatial_clean_positive_hit_rows_gte_0p55"] = float(
        clean_spatial["positive_bbox_hit_delta_fraction"]
    ) >= 0.55
    checks["spatial_clean_outside_distance_reduction_gte_0p05"] = float(
        clean_spatial["outside_distance_mean"]
    ) <= 0.95 * float(clean_spatial["reference_outside_distance_mean"])
    frozen = spatial["frozen_cohort"]
    checks["spatial_frozen_proxy_candidate_not_lower"] = float(
        frozen["candidate_block2_attention_bbox_mass"]
    ) >= float(frozen["control_block2_attention_bbox_mass"])
    checks["spatial_frozen_proxy_delta_gte_0p010"] = float(frozen["delta"]) >= 0.010
    checks["spatial_shifted_no_double_reversal"] = all(
        not (
            float(spatial_conditions[name]["bbox_hit_delta_mean"]) < 0.0
            and float(spatial_conditions[name]["outside_distance_mean"])
            > float(spatial_conditions[name]["reference_outside_distance_mean"])
        )
        for name in shifted
    )
    checks.update(
        {
            "xai_stem_foreground_delta_gte_minus_0p05": float(
                xai["stem_foreground_mass_delta"]
            )
            >= -0.05,
            "xai_block2_foreground_delta_gte_minus_0p05": float(
                xai["block2_foreground_mass_delta"]
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
        raise FileNotFoundError(f"DAT audit summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual review finalization requires a result and note.")
    summary = common._load_json(summary_path)
    pages = summary.get("xai", {}).get("render", {}).get("pages", [])
    if not pages or not all(Path(str(path)).is_file() for path in pages):
        raise ValueError("Visual review requires every DAT XAI page.")
    passed = args.visual_review_result == "pass"
    review = {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": args.visual_review_result,
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "page_count": len(pages),
        "pages_sha256": {
            str(path): common._sha256(Path(str(path))) for path in pages
        },
    }
    common._write_json(output_dir / "visual_review.json", review)
    gate = summary["gate"]
    gate["visual_review_completed"] = True
    gate["visual_review_passed"] = passed
    gate["promotion_permission"] = bool(gate["automated_pass"] and passed)
    summary["visual_review"] = review
    common._write_json(summary_path, summary)
    return summary


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.seed) != 42 or int(args.batch_size) != 32:
        raise ValueError("DAT pair audit is locked to seed=42 and batch_size=32.")
    if int(args.xai_batch_size) <= 0 or int(args.xai_batch_size) > 4:
        raise ValueError("DAT XAI batch size must be in [1,4].")
    output_dir = common._prepare_output(args.output_dir)
    declaration = Path(args.declaration).resolve()
    if common._sha256(declaration) != LOCKED_DECLARATION_SHA256:
        raise ValueError("DAT declaration hash differs from the locked protocol.")
    rows = _read_clean_train_rows(declaration)
    holdout_rows = [row for row in rows if row.fold == FOLD]
    fit_rows = [row for row in rows if row.fold != FOLD]
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(fit_rows) != EXPECTED_FIT_ROWS:
        raise ValueError("DAT fit/holdout row counts differ from the protocol.")
    if _ordered_index_sha256(
        [row.sample_index for row in fit_rows]
    ) != EXPECTED_FIT_INDEX_SHA256:
        raise ValueError("DAT fit sample-index hash differs.")
    if _ordered_index_sha256(
        [row.sample_index for row in holdout_rows]
    ) != EXPECTED_HOLDOUT_INDEX_SHA256:
        raise ValueError("DAT holdout sample-index hash differs.")

    if not torch.cuda.is_available():
        raise RuntimeError("DAT formal pair audit requires CUDA.")
    device = torch.device("cuda")
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
        raise ValueError("DAT pair checkpoint class orders differ.")
    control_config = control_checkpoint.get("model_config", {})
    candidate_config = candidate_checkpoint.get("model_config", {})
    if bool(control_config.get("deformable_spatial_attention", False)):
        raise ValueError("DAT control checkpoint unexpectedly enables DAT.")
    if not bool(candidate_config.get("deformable_spatial_attention", False)):
        raise ValueError("DAT candidate checkpoint does not enable DAT.")
    if not isinstance(candidate.blocks[1].attn, DeformableSpatialAttention):
        raise ValueError("DAT candidate checkpoint did not reconstruct block-2 DAT.")
    if common._eval_semantics(control_checkpoint) != common._eval_semantics(
        candidate_checkpoint
    ):
        raise ValueError("DAT pair evaluation semantics differ.")

    dataset, transform, dataset_summary = common._build_holdout_dataset(
        control_checkpoint,
        fold_data=Path(args.fold_data),
        holdout_rows=holdout_rows,
    )
    preflight = common._load_json(Path(args.preflight_summary))
    fold_summary_path = Path(args.fold_summary)
    fold_summary = common._load_json(fold_summary_path)
    provenance = _run_provenance(
        control_run=Path(args.control_run_dir),
        candidate_run=Path(args.candidate_run_dir),
        pair_manifest_path=Path(args.pair_manifest),
        preflight=preflight,
        fold_summary=fold_summary,
        fold_summary_path=fold_summary_path,
    )
    predictions, loader_summaries, spatial = _predict_conditions_and_spatial(
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
    common._write_predictions(prediction_path, predictions)
    comparisons = common._comparisons(predictions)
    selectivity = common._selectivity(rows=rows, predictions=predictions)
    _attach_frozen_cohort_spatial(
        spatial,
        predictions=predictions,
        selectivity=selectivity,
    )
    events = common._build_event_manifest(predictions)
    common._write_event_manifest(output_dir / "event_manifest.csv", events)
    representatives = _representative_requests(
        dataset=dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=selectivity["positive_sample_indices"],
    )
    requests = common._merge_xai_requests(events, representatives)
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
    perturbation = common._perturbation_audit(
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
        spatial=spatial,
        perturbation=perturbation,
        xai=xai,
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
            "control_checkpoint_sha256": common._sha256(Path(args.control_checkpoint)),
            "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
            "candidate_checkpoint_sha256": common._sha256(Path(args.candidate_checkpoint)),
            "declaration": str(declaration),
            "declaration_sha256": common._sha256(declaration),
            "fold_data": str(Path(args.fold_data).resolve()),
            "fold_summary": str(fold_summary_path.resolve()),
            "preflight_summary": str(Path(args.preflight_summary).resolve()),
            "pair_manifest": str(Path(args.pair_manifest).resolve()),
        },
        "dataset": dataset_summary,
        "provenance": provenance,
        "loader_summaries": loader_summaries,
        "comparisons": comparisons,
        "selectivity": selectivity,
        "spatial": spatial,
        "perturbation": perturbation,
        "events": {
            "rows": len(events),
            "manifest": str((output_dir / "event_manifest.csv").resolve()),
        },
        "xai": xai,
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": common._sha256(prediction_path),
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
    common._write_json(output_dir / "summary.json", summary)
    common._write_json(
        output_dir / "visual_review_required.json",
        {
            "status": "pending",
            "required": True,
            "page_count": int(render["page_count"]),
            "finalize_command": (
                "python -m trkh.tools.audit_deformable_spatial_attention_pair "
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
