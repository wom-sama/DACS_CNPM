from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, default_collate

from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.xai_audit import _build_dataset, _forward_logits_with_optional_bbox
from trkh.core.utils import load_checkpoint, set_seed
from trkh.inference.inference import load_model
from trkh.models.learnable_gabor_texture import LearnableGaborTextureResidual
from trkh.models.model import create_model, load_model_state
from trkh.tools.audit_learnable_gabor_texture_smoke_pair import (
    _metrics_with_confusion,
)
from trkh.tools.audit_visual_contrast_smoke_pair import (
    EXPECTED_VAL_ROWS,
    EXPECTED_VAL_SUPPORT,
    _prepare_output_dir,
    _sha256,
)


CONDITIONS = ("clean", "center_occlusion", "dim", "bright", "low_contrast")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-validation robustness and learned-mechanism audit for Gabor smoke."
    )
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _move_optional(
    metadata: Mapping[str, object],
    name: str,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    value = metadata.get(name)
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _condition_images(images: Tensor, mean: Tensor, std: Tensor) -> Dict[str, Tensor]:
    rgb = (images.float() * std + mean).clamp(0.0, 1.0)
    image_mean = rgb.mean(dim=(-2, -1), keepdim=True)
    occluded = rgb.clone()
    height, width = int(rgb.size(-2)), int(rgb.size(-1))
    occ_height = max(1, int(round(height * 0.24)))
    occ_width = max(1, int(round(width * 0.24)))
    top = (height - occ_height) // 2
    left = (width - occ_width) // 2
    occluded[:, :, top : top + occ_height, left : left + occ_width] = 127.0 / 255.0
    rgb_conditions = {
        "clean": rgb,
        "center_occlusion": occluded,
        "dim": rgb * 0.75,
        "bright": (rgb * 1.15 + 0.05).clamp(0.0, 1.0),
        "low_contrast": ((rgb - image_mean) * 0.70 + image_mean).clamp(0.0, 1.0),
    }
    return {name: (value - mean) / std for name, value in rgb_conditions.items()}


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
        tp_retained = tp_retained and tp_delta >= -1
        rows[condition] = {
            "macro_f1_delta": macro_delta,
            "class1_tp_delta": tp_delta,
            "control_class1_tp": control_tp,
            "candidate_class1_tp": candidate_tp,
        }
    checks = {
        "macro_f1_wins_at_least_three_of_five": macro_wins >= 3,
        "class1_tp_loss_at_most_one_each_condition": tp_retained,
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "macro_f1_win_count": macro_wins,
        "conditions": rows,
    }


def _pair_cosine_max(values: Tensor) -> float:
    normalized = F.normalize(values.flatten(1).float(), dim=1)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity.fill_diagonal_(0.0)
    return float(similarity.abs().max().item())


def _attention_entropy(attention: Tensor) -> float:
    probability = attention.float().clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum(dim=-1)
    denominator = math.log(max(2, int(probability.size(-1))))
    return float((entropy / denominator).mean().item())


def _balanced_batch(dataset, device: torch.device) -> Tuple[Tensor, Tensor, Mapping[str, Tensor]]:
    indices: Dict[int, int] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        label = int(item[1])
        indices.setdefault(label, index)
        if len(indices) == 5:
            break
    if set(indices) != set(range(5)):
        raise RuntimeError("Could not construct one validation example per class")
    batch = default_collate([dataset[indices[index]] for index in range(5)])
    images = batch[0].to(device=device)
    labels = batch[1].to(device=device, dtype=torch.long)
    metadata = {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in batch[2].items()
    }
    return images, labels, metadata


def _gabor_diagnostics(
    *,
    model,
    images: Tensor,
    labels: Tensor,
    bbox: Tensor,
    initial_state: Mapping[str, Tensor],
) -> Dict[str, object]:
    module = getattr(model, "gabor_texture_residual", None)
    if not isinstance(module, LearnableGaborTextureResidual):
        raise ValueError("Candidate checkpoint has no learnable Gabor residual")
    module.eval()
    with torch.no_grad():
        full = module.forward_components(images, bbox)
        low = module.forward_components(images, bbox, use_high=False)
        high = module.forward_components(images, bbox, use_low=False)
        full_bbox = torch.tensor(
            [0.5, 0.5, 1.0, 1.0], device=images.device, dtype=torch.float32
        ).view(1, 4).expand(images.size(0), -1)
        full_frame = module.forward_components(images, full_bbox)
        residual, trace = module(images, bbox, return_trace=True)
        real_kernel, imaginary_kernel, parameters = module.build_kernels()
    raw_deltas = {
        name: float(
            (
                getattr(module, name).detach().float()
                - initial_state[name].to(
                    device=getattr(module, name).device,
                    dtype=torch.float32,
                )
            )
            .abs()
            .max()
            .item()
        )
        for name in ("raw_theta", "raw_frequency", "raw_sigma_x", "raw_sigma_y", "raw_gate")
    }
    filter_features = full["filter_features"].mean(dim=0)
    fcm_attention = full["fcm_attention"]
    fcm_diagonal = torch.diagonal(fcm_attention.float(), dim1=-2, dim2=-1).mean()
    lho_entropy = -(full["counts"] * full["counts"].clamp_min(1e-8).log()).sum(dim=-1)
    gate = float(module.effective_gate().detach().float().item())
    checks = {
        "effective_gate_nonzero": abs(gate) > 1e-6,
        "raw_gabor_parameters_updated": max(raw_deltas.values()) > 1e-7,
        "low_frequency_path_material": float(
            (full["texture_token"] - high["texture_token"]).abs().mean().item()
        )
        > 1e-6,
        "high_frequency_path_material": float(
            (full["texture_token"] - low["texture_token"]).abs().mean().item()
        )
        > 1e-6,
        "bbox_path_material": float(
            (full["texture_token"] - full_frame["texture_token"]).abs().mean().item()
        )
        > 1e-6,
        "lho_entropy_noncollapsed": float(lho_entropy.mean().item()) > 0.25,
        "fcm_attention_finite": bool(torch.isfinite(fcm_attention).all()),
        "residual_finite_nonzero": bool(torch.isfinite(residual).all())
        and float(residual.norm(dim=1).mean().item()) > 1e-7,
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
                "residual_norm": float(residual[row].norm().item()),
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
        "effective_gate": gate,
        "raw_parameter_max_deltas_from_initial": raw_deltas,
        "low_only_token_delta": float(
            (full["texture_token"] - low["texture_token"]).abs().mean().item()
        ),
        "high_only_token_delta": float(
            (full["texture_token"] - high["texture_token"]).abs().mean().item()
        ),
        "bbox_full_frame_token_delta": float(
            (full["texture_token"] - full_frame["texture_token"]).abs().mean().item()
        ),
        "residual_norm_mean": float(residual.norm(dim=1).mean().item()),
        "lho_entropy_mean": float(lho_entropy.mean().item()),
        "fcm_attention_normalized_entropy": _attention_entropy(fcm_attention),
        "fcm_attention_diagonal_mean": float(fcm_diagonal.item()),
        "filter_feature_max_absolute_pair_cosine": _pair_cosine_max(filter_features),
        "real_kernel_max_absolute_pair_cosine": _pair_cosine_max(real_kernel),
        "imaginary_kernel_max_absolute_pair_cosine": _pair_cosine_max(imaginary_kernel),
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
        "per_class": per_class,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = _prepare_output_dir(args.output_dir)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Post-smoke Gabor audit requires CUDA")
    control_model, control_checkpoint, control_names = load_model(
        args.control_checkpoint,
        device,
    )
    candidate_model, candidate_checkpoint, candidate_names = load_model(
        args.candidate_checkpoint, device
    )
    if list(control_names) != list(candidate_names) or len(candidate_names) != 5:
        raise ValueError("Checkpoint class ordering mismatch")
    if bool(control_checkpoint["model_config"].get("learnable_gabor_texture_residual", False)):
        raise ValueError("Matched control unexpectedly enables Gabor residual")
    if not bool(candidate_checkpoint["model_config"].get("learnable_gabor_texture_residual", False)):
        raise ValueError("Candidate Gabor residual is disabled")
    initialization_checkpoint = load_checkpoint(
        Path(args.initialization_checkpoint),
        map_location="cpu",
    )
    initialization_state = initialization_checkpoint.get("model_state")
    initialization_names = initialization_checkpoint.get("class_names")
    if not isinstance(initialization_state, Mapping):
        raise ValueError("Initialization checkpoint has no model_state")
    if list(initialization_names or []) != list(candidate_names):
        raise ValueError("Initialization checkpoint class ordering mismatch")
    set_seed(int(args.seed))
    initial_candidate = create_model(
        len(candidate_names),
        dict(candidate_checkpoint["model_config"]),
    )
    missing, unexpected = load_model_state(
        initial_candidate,
        dict(initialization_state),
        strict=False,
    )
    if not missing or any(
        not str(key).startswith("gabor_texture_residual.") for key in missing
    ):
        raise ValueError("Initialization replay did not miss only Gabor parameters")
    if unexpected:
        raise ValueError(f"Initialization replay has unexpected keys: {unexpected}")
    initial_module = getattr(initial_candidate, "gabor_texture_residual", None)
    if not isinstance(initial_module, LearnableGaborTextureResidual):
        raise ValueError("Initialization replay has no Gabor residual")
    initial_gabor_state = {
        name: getattr(initial_module, name).detach().cpu().clone()
        for name in (
            "raw_theta",
            "raw_frequency",
            "raw_sigma_x",
            "raw_sigma_y",
            "raw_gate",
        )
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
        raise ValueError(f"Expected {EXPECTED_VAL_ROWS} validation rows, found {len(dataset)}")
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
            image_mask = _move_optional(metadata, "image_mask", device=device, dtype=torch.bool)
            source_context = _move_optional(metadata, "source_context_image", device=device)
            source_bbox = _move_optional(
                metadata, "source_context_bbox", device=device, dtype=torch.float32
            )
            variants = _condition_images(images, mean, std)
            labels_all.extend(int(value) for value in labels.cpu().tolist())
            for condition, condition_images in variants.items():
                for role, model in (("control", control_model), ("candidate", candidate_model)):
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
                        int(value) for value in probability.argmax(dim=1).cpu().tolist()
                    )
    elapsed = time.perf_counter() - started
    labels_array = np.asarray(labels_all, dtype=np.int64)
    if np.bincount(labels_array, minlength=5).tolist() != list(EXPECTED_VAL_SUPPORT):
        raise ValueError("Validation support differs from locked cohort")
    condition_metrics: Dict[str, Dict[str, object]] = {}
    prediction_path = output_dir / "robustness_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["row_index", "target", "condition", "role", "prediction", "probability_1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            condition_metrics[condition] = {}
            for role in ("control", "candidate"):
                prediction_array = np.asarray(predictions[condition][role], dtype=np.int64)
                probability_array = torch.cat(probabilities[condition][role], dim=0).numpy()
                condition_metrics[condition][role] = _metrics_with_confusion(
                    labels_array,
                    prediction_array,
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
    balanced_images, balanced_labels, balanced_metadata = _balanced_batch(dataset, device)
    balanced_bbox = balanced_metadata.get("crop_bbox")
    if not torch.is_tensor(balanced_bbox):
        raise RuntimeError("Balanced validation cohort lacks crop_bbox")
    gabor = _gabor_diagnostics(
        model=candidate_model,
        images=balanced_images,
        labels=balanced_labels,
        bbox=balanced_bbox.float(),
        initial_state=initial_gabor_state,
    )
    robustness_gate = assess_robustness(condition_metrics)
    summary = {
        "method": "learnable_gabor_lho_fcm_edge_token_residual",
        "protocol_stage": "B_post_smoke_full_validation_robustness_and_mechanism",
        "sources": {
            "control_checkpoint": str(Path(args.control_checkpoint).resolve()),
            "control_checkpoint_sha256": _sha256(args.control_checkpoint),
            "initialization_checkpoint": str(
                Path(args.initialization_checkpoint).resolve()
            ),
            "initialization_checkpoint_sha256": _sha256(
                args.initialization_checkpoint
            ),
            "initialization_seed": int(args.seed),
            "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
            "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint),
            "data": str(Path(args.data).resolve()),
            "data_sha256": _sha256(args.data),
            "split": "val",
            "validation_rows": len(labels_array),
            "test_used": False,
            "raw_dataset_modified": False,
        },
        "conditions": condition_metrics,
        "robustness_gate": robustness_gate,
        "gabor_mechanism": gabor,
        "runtime": {
            "seconds": elapsed,
            "batch_size": int(args.batch_size),
            "condition_count": len(CONDITIONS),
            "model_count": 2,
            "amp_dtype": "bfloat16",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
