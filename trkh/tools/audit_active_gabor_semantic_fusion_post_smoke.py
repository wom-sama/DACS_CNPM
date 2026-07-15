from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from trkh.core.utils import set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.xai_audit import _build_dataset, _forward_logits_with_optional_bbox
from trkh.inference.inference import load_model
from trkh.models.learnable_gabor_texture import LearnableGaborTextureEncoder
from trkh.models.model import create_model
from trkh.tools.audit_active_gabor_semantic_fusion_readiness import (
    _active_feature_diagnostics,
    _family_parameter_movement,
    _mean_off_diagonal_cosine,
)
from trkh.tools.audit_active_gabor_semantic_fusion_smoke_pair import (
    METHOD,
    _metrics_with_confusion,
)
from trkh.tools.audit_learnable_gabor_texture_post_smoke import (
    CONDITIONS,
    _attention_entropy,
    _balanced_batch,
    _condition_images,
    _move_optional,
    _pair_cosine_max,
)
from trkh.tools.audit_visual_contrast_smoke_pair import (
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SUPPORT,
    _prepare_output_dir,
    _sha256,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-validation robustness and learned-mechanism audit for active "
            "Gabor semantic fusion."
        )
    )
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def assess_robustness(
    condition_metrics: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> Dict[str, object]:
    rows = {}
    macro_wins = 0
    tp_retained = True
    for condition in CONDITIONS:
        control = condition_metrics[condition]["control"]
        candidate = condition_metrics[condition]["candidate"]
        control_tp = int(control["confusion_matrix"][1][1])
        candidate_tp = int(candidate["confusion_matrix"][1][1])
        macro_delta = float(candidate["macro_f1"]) - float(control["macro_f1"])
        tp_delta = candidate_tp - control_tp
        macro_wins += int(macro_delta >= 0.0)
        tp_retained = tp_retained and tp_delta >= -2
        rows[condition] = {
            "macro_f1_delta": macro_delta,
            "class1_tp_delta": tp_delta,
            "control_class1_tp": control_tp,
            "candidate_class1_tp": candidate_tp,
        }
    checks = {
        "macro_f1_wins_at_least_three_of_five": macro_wins >= 3,
        "class1_tp_loss_at_most_two_each_condition": tp_retained,
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "macro_f1_win_count": macro_wins,
        "conditions": rows,
    }


def _active_gabor_diagnostics(
    *,
    model,
    images: Tensor,
    labels: Tensor,
    metadata: Mapping[str, Tensor],
    initial_state: Mapping[str, Tensor],
) -> Dict[str, object]:
    module = getattr(model, "gabor_texture_semantic_encoder", None)
    if not isinstance(module, LearnableGaborTextureEncoder):
        raise ValueError("Candidate checkpoint has no active Gabor semantic encoder")
    bbox = metadata.get("crop_bbox")
    if not torch.is_tensor(bbox):
        raise RuntimeError("Balanced validation cohort lacks crop_bbox")
    bbox = bbox.float()
    module.eval()
    with torch.no_grad():
        full = module.forward_components(images, bbox)
        low = module.forward_components(images, bbox, use_high=False)
        high = module.forward_components(images, bbox, use_low=False)
        full_bbox = torch.tensor(
            [0.5, 0.5, 1.0, 1.0], device=images.device, dtype=torch.float32
        ).view(1, 4).expand(images.size(0), -1)
        full_frame = module.forward_components(images, full_bbox)
        texture, trace = module(images, bbox, return_trace=True)
        real_kernel, imaginary_kernel, parameters = module.build_kernels()
        active = _active_feature_diagnostics(model, images, metadata)

    movement = _family_parameter_movement(initial_state, module)
    named_parameters = dict(module.named_parameters())
    raw_movement = {
        name: float(
            (
                named_parameters[name].detach().float().cpu()
                - initial_state[name].float()
            )
            .abs()
            .sum()
            .item()
        )
        for name in ("raw_theta", "raw_frequency", "raw_sigma_x", "raw_sigma_y")
    }
    filter_features = full["filter_features"]
    fcm_attention = full["fcm_attention"]
    fcm_diagonal = torch.diagonal(fcm_attention.float(), dim1=-2, dim2=-1).mean()
    lho_entropy = -(full["counts"] * full["counts"].clamp_min(1e-8).log()).sum(
        dim=-1
    )
    low_response_energy = float(
        full["magnitude"][:, : module.low_filter_count]
        .float()
        .square()
        .mean()
        .item()
    )
    high_response_energy = float(
        full["magnitude"][:, module.low_filter_count :]
        .float()
        .square()
        .mean()
        .item()
    )
    low_only_delta = float(
        (full["texture_token"] - low["texture_token"]).abs().mean().item()
    )
    high_only_delta = float(
        (full["texture_token"] - high["texture_token"]).abs().mean().item()
    )
    bbox_delta = float(
        (full["texture_token"] - full_frame["texture_token"]).abs().mean().item()
    )
    filter_pair_cosine = _pair_cosine_max(filter_features.mean(dim=0))
    texture_sample_cosine = _mean_off_diagonal_cosine(texture.unsqueeze(0))
    fcm_entropy = _attention_entropy(fcm_attention)
    norm_ratio = float(active["texture_semantic_norm_ratio_mean"])
    fused_ratio = float(active["fused_semantic_norm_ratio_mean"])
    checks = {
        "all_parameter_families_updated": all(
            bool(value["passed"]) for value in movement.values()
        ),
        "all_raw_gabor_parameters_updated": all(value > 0.0 for value in raw_movement.values()),
        "low_frequency_response_live": low_response_energy > 1e-8,
        "high_frequency_response_live": high_response_energy > 1e-8,
        "low_frequency_path_material": high_only_delta > 1e-6,
        "high_frequency_path_material": low_only_delta > 1e-6,
        "bbox_path_material": bbox_delta > 1e-6,
        "lho_entropy_noncollapsed": float(lho_entropy.mean().item()) > 0.25,
        "fcm_attention_finite": bool(torch.isfinite(fcm_attention).all()),
        "fcm_attention_nonuniform": fcm_entropy < 0.9999,
        "filter_features_diverse": filter_pair_cosine < 0.9999,
        "texture_samples_diverse": texture_sample_cosine < 0.9999,
        "active_features_finite": bool(active["semantic_finite"])
        and bool(active["texture_finite"])
        and bool(active["fused_finite"]),
        "direct_semantic_sum_exact": float(active["direct_sum_max_error"]) <= 1e-6,
        "texture_logit_contribution_material": float(
            active["main_logit_mean_absolute_delta"]
        )
        > 1e-6,
        "texture_semantic_norm_ratio": 0.50 <= norm_ratio <= 3.00,
        "fused_semantic_norm_ratio": 0.75 <= fused_ratio <= 3.00,
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    per_class = []
    for row, label in enumerate(labels.detach().cpu().tolist()):
        per_class.append(
            {
                "class_index": int(label),
                "low_response_mean": float(trace["low_response_map"][row].mean().item()),
                "high_response_mean": float(trace["high_response_map"][row].mean().item()),
                "lho_entropy_mean": float(lho_entropy[row].mean().item()),
                "texture_norm": float(texture[row].float().norm().item()),
                "bbox_mask_area": float(trace["mask_area_fraction"][row].item()),
                "response_foreground_mass": float(
                    trace["response_foreground_mass"][row].mean().item()
                ),
            }
        )
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "parameter_movement_from_seed42_initialization": movement,
        "raw_parameter_absolute_movement": raw_movement,
        "low_response_energy": low_response_energy,
        "high_response_energy": high_response_energy,
        "full_minus_low_only_token_delta": low_only_delta,
        "full_minus_high_only_token_delta": high_only_delta,
        "bbox_full_frame_token_delta": bbox_delta,
        "lho_entropy_mean": float(lho_entropy.mean().item()),
        "fcm_attention_normalized_entropy": fcm_entropy,
        "fcm_attention_diagonal_mean": float(fcm_diagonal.item()),
        "texture_sample_mean_off_diagonal_cosine": texture_sample_cosine,
        "filter_feature_max_absolute_pair_cosine": filter_pair_cosine,
        "real_kernel_max_absolute_pair_cosine": _pair_cosine_max(real_kernel),
        "imaginary_kernel_max_absolute_pair_cosine": _pair_cosine_max(
            imaginary_kernel
        ),
        "frequency_min_max": [
            float(parameters["frequency"].min().item()),
            float(parameters["frequency"].max().item()),
        ],
        "sigma_x_min_max": [
            float(parameters["sigma_x"].min().item()),
            float(parameters["sigma_x"].max().item()),
        ],
        "sigma_y_min_max": [
            float(parameters["sigma_y"].min().item()),
            float(parameters["sigma_y"].max().item()),
        ],
        "active_feature": {
            key: value for key, value in active.items() if not torch.is_tensor(value)
        },
        "per_class": per_class,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Post-smoke active Gabor audit requires CUDA")
    control_model, control_checkpoint, control_names = load_model(
        args.control_checkpoint, device
    )
    candidate_model, candidate_checkpoint, candidate_names = load_model(
        args.candidate_checkpoint, device
    )
    if list(control_names) != list(candidate_names) or len(candidate_names) != 5:
        raise ValueError("Checkpoint class ordering mismatch")
    if bool(
        control_checkpoint["model_config"].get(
            "learnable_gabor_texture_semantic_fusion", False
        )
    ):
        raise ValueError("Matched control unexpectedly enables active Gabor fusion")
    if not bool(
        candidate_checkpoint["model_config"].get(
            "learnable_gabor_texture_semantic_fusion", False
        )
    ):
        raise ValueError("Candidate active Gabor fusion is disabled")
    if bool(
        candidate_checkpoint["model_config"].get(
            "learnable_gabor_texture_residual", False
        )
    ):
        raise ValueError("Candidate unexpectedly enables historical Gabor residual")

    set_seed(int(args.seed), deterministic=True)
    initial_candidate = create_model(
        len(candidate_names), dict(candidate_checkpoint["model_config"])
    )
    initial_module = getattr(initial_candidate, "gabor_texture_semantic_encoder", None)
    if not isinstance(initial_module, LearnableGaborTextureEncoder):
        raise ValueError("Seed replay has no active Gabor semantic encoder")
    initial_state = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in initial_module.named_parameters()
    }
    del initial_candidate

    dataset = _build_dataset(
        data_yaml=Path(args.data),
        classification_folder_yolo_data=None,
        split="val",
        class_name_mode="raw",
        expected_num_classes=5,
        checkpoint=candidate_checkpoint,
    )
    if len(dataset) != EXPECTED_VAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_VAL_ROWS} validation rows, found {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    mean_values, std_values = checkpoint_input_normalization(candidate_checkpoint)
    mean = torch.tensor(mean_values, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    labels_all = []
    predictions: Dict[str, Dict[str, list]] = {
        condition: {"control": [], "candidate": []} for condition in CONDITIONS
    }
    probabilities: Dict[str, Dict[str, list]] = {
        condition: {"control": [], "candidate": []} for condition in CONDITIONS
    }
    control_model.eval()
    candidate_model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device=device, non_blocking=True)
            labels = batch[1].to(device=device, dtype=torch.long, non_blocking=True)
            metadata = batch[2]
            bbox = _move_optional(metadata, "bbox", device=device, dtype=torch.float32)
            crop_bbox = _move_optional(
                metadata, "crop_bbox", device=device, dtype=torch.float32
            )
            if crop_bbox is None:
                raise RuntimeError("Validation batch lacks crop_bbox metadata")
            image_mask = _move_optional(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            source_context = _move_optional(
                metadata, "source_context_image", device=device
            )
            source_bbox = _move_optional(
                metadata, "source_context_bbox", device=device, dtype=torch.float32
            )
            variants = _condition_images(images, mean, std)
            labels_all.extend(int(value) for value in labels.cpu().tolist())
            for condition, condition_images in variants.items():
                for role, model in (
                    ("control", control_model),
                    ("candidate", candidate_model),
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = _forward_logits_with_optional_bbox(
                            model,
                            condition_images,
                            bbox,
                            bbox_token_prior=crop_bbox,
                            image_valid_mask=image_mask,
                            source_context_images=source_context,
                            source_context_bboxes=source_bbox,
                        )
                    probability = logits.float().softmax(dim=1)
                    probabilities[condition][role].append(probability.cpu())
                    predictions[condition][role].extend(
                        int(value)
                        for value in probability.argmax(dim=1).cpu().tolist()
                    )
    elapsed = time.perf_counter() - started
    labels_array = np.asarray(labels_all, dtype=np.int64)
    if np.bincount(labels_array, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from locked cohort")

    condition_metrics: Dict[str, Dict[str, object]] = {}
    prediction_path = output_dir / "robustness_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "row_index",
            "target",
            "condition",
            "role",
            "prediction",
            "probability_1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            condition_metrics[condition] = {}
            for role in ("control", "candidate"):
                prediction_array = np.asarray(
                    predictions[condition][role], dtype=np.int64
                )
                probability_array = torch.cat(
                    probabilities[condition][role], dim=0
                ).numpy()
                condition_metrics[condition][role] = _metrics_with_confusion(
                    labels_array, prediction_array
                )
                for index in range(len(labels_array)):
                    writer.writerow(
                        {
                            "row_index": index,
                            "target": int(labels_array[index]),
                            "condition": condition,
                            "role": role,
                            "prediction": int(prediction_array[index]),
                            "probability_1": float(probability_array[index, 1]),
                        }
                    )

    balanced_images, balanced_labels, balanced_metadata = _balanced_batch(
        dataset, device
    )
    mechanism = _active_gabor_diagnostics(
        model=candidate_model,
        images=balanced_images,
        labels=balanced_labels,
        metadata=balanced_metadata,
        initial_state=initial_state,
    )
    robustness_gate = assess_robustness(condition_metrics)
    summary = {
        "method": METHOD,
        "protocol_stage": "B_post_smoke_full_validation_robustness_and_mechanism",
        "sources": {
            "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
            "control_checkpoint_sha256": _sha256(args.control_checkpoint),
            "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
            "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint),
            "initialization_replay_seed": int(args.seed),
            "data": str(Path(args.data).resolve()),
            "data_sha256": _sha256(args.data),
            "split": "val",
            "validation_rows": len(labels_array),
            "test_used": False,
            "raw_dataset_modified": False,
        },
        "conditions": condition_metrics,
        "robustness_gate": robustness_gate,
        "gabor_mechanism": mechanism,
        "post_smoke_mechanism_and_robustness_passed": bool(
            robustness_gate["passed"] and mechanism["passed"]
        ),
        "runtime": {
            "seconds": elapsed,
            "batch_size": int(args.batch_size),
            "condition_count": len(CONDITIONS),
            "model_count": 2,
            "amp_dtype": "bfloat16",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "artifacts": [
            {"path": str(summary_path), "sha256": _sha256(summary_path)},
            {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        ],
        "raw_dataset_modified": False,
        "test_used": False,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run_audit(parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
