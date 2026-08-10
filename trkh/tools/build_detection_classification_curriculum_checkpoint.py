from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from trkh.models.model import build_model_from_checkpoint, create_model


DETECTION_MODEL_TYPES = {"detr_vit_registers", "vit_registers_hybrid"}
ENCODER_TRANSFER_PREFIXES = (
    "stem.",
    "patch_embed.",
    "blocks.",
    "norm.",
    "branch_token_fusion.",
    "detail_enhancer.",
    "cls_token",
    "register_tokens",
    "pos_embed",
)
RESET_CHECKPOINT_KEYS = {
    "adaptive_detection_loss",
    "train_model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "ema_model_state",
    "ema_updates",
    "ema_decay",
    "metrics",
    "best_macro_f1",
    "best_epoch",
    "best_selection_metric",
    "selection_metric",
    "epochs_without_improvement",
    "resume_state",
    "train_stage",
    "stage1_auto_advance",
    "stage1_auto_advance_epoch",
    "stage1_checkpoint",
    "classification_overfit_guard",
    "rare_class_recall_guard",
    "checkpoint_weight_source",
    "validation_weight_source",
    "calibration",
    "data_summary",
    "imbalance_summary",
}
DETECTOR_RUNTIME_ARGUMENTS = (
    "--full-image-detection",
    "--skip-final-test",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh-state checkpoint for a supervised full-frame detection "
            "to object-classification curriculum. Conversion never reads test data."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("classifier-to-detector", "detector-to-classifier"),
        required=True,
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-model-type", choices=tuple(sorted(DETECTION_MODEL_TYPES)), default="detr_vit_registers")
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-ffn-dim", type=int, default=512)
    parser.add_argument("--decoder-dropout", type=float, default=0.10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-grad-accum-steps", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--train-learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-backbone-lr-scale", type=float, default=0.10)
    parser.add_argument("--train-min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--train-patience", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=120)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


def _load_checkpoint(path: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a mapping: {resolved}")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError(f"Checkpoint has no selected model_state: {resolved}")
    if not isinstance(payload.get("model_config"), Mapping):
        raise ValueError(f"Checkpoint has no model_config: {resolved}")
    class_names = payload.get("class_names")
    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise ValueError(f"Checkpoint has no class_names: {resolved}")
    return payload


def is_encoder_transfer_key(key: str) -> bool:
    text = str(key)
    return any(text == prefix or text.startswith(prefix) for prefix in ENCODER_TRANSFER_PREFIXES)


def merge_matching_state(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
    *,
    encoder_only: bool,
) -> Tuple[Dict[str, Tensor], Dict[str, object]]:
    merged = {key: value.detach().cpu().clone() for key, value in target_state.items()}
    transferred: list[str] = []
    shape_mismatches: list[str] = []
    for key, value in source_state.items():
        if encoder_only and not is_encoder_transfer_key(str(key)):
            continue
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(str(key))
            continue
        merged[str(key)] = value.detach().cpu().clone()
        transferred.append(str(key))
    source_candidates = [
        str(key)
        for key in source_state
        if not encoder_only or is_encoder_transfer_key(str(key))
    ]
    missing_candidates = sorted(set(source_candidates).difference(transferred).difference(shape_mismatches))
    return merged, {
        "encoder_only": bool(encoder_only),
        "transferred_keys": transferred,
        "transferred_key_count": int(len(transferred)),
        "transferred_elements": int(sum(source_state[key].numel() for key in transferred)),
        "shape_mismatches": sorted(shape_mismatches),
        "source_candidates_without_target": missing_candidates,
    }


def fresh_checkpoint_payload(
    reference: Mapping[str, object],
    *,
    model_state: Mapping[str, Tensor],
    model_config: Mapping[str, object],
    transfer_summary: Mapping[str, object],
    train_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        key: value
        for key, value in reference.items()
        if key not in RESET_CHECKPOINT_KEYS and key != "model_state"
    }
    payload.update(
        {
            "epoch": 0,
            "model_state": {key: value.detach().cpu().clone() for key, value in model_state.items()},
            "model_config": dict(model_config),
            "curriculum_transfer": dict(transfer_summary),
            "optimizer_state": None,
            "scheduler_state": None,
            "scaler_state": None,
            "raw_dataset_touched": False,
            "test_split_used": False,
        }
    )
    if train_config is not None:
        payload["train_config"] = dict(train_config)
    return payload


def detector_train_profile(
    reference_train_config: Mapping[str, object],
    *,
    args: argparse.Namespace,
) -> Dict[str, object]:
    profile = dict(reference_train_config)
    profile.update(
        {
            "batch_size": int(args.train_batch_size),
            "grad_accum_steps": int(args.train_grad_accum_steps),
            "epochs": int(args.train_epochs),
            "learning_rate": float(args.train_learning_rate),
            "backbone_lr_scale": float(args.train_backbone_lr_scale),
            "min_learning_rate": float(args.train_min_learning_rate),
            "warmup_epochs": 0,
            "scheduler_total_epochs": int(args.train_epochs),
            "early_stopping_patience": int(args.train_patience),
            "num_workers": int(args.num_workers),
            "eval_num_workers": int(args.eval_num_workers),
            "max_train_batches": int(args.max_train_batches),
            "max_val_batches": int(args.max_val_batches),
            "stage1_epochs": 0,
            "stage1_bbox_l1_loss_weight": 0.0,
            "stage1_bbox_giou_loss_weight": 0.0,
            "best_metric": "macro_detection_hmean",
            "cls_loss_weight": 1.0,
            "bbox_l1_loss_weight": 0.20,
            "bbox_giou_loss_weight": 0.10,
            "objectness_loss_weight": 1.0,
            "matcher_class_cost": 1.0,
            "matcher_objectness_cost": 1.0,
            "classification_overfit_guard": False,
            "adaptive_detection_loss": False,
            "rare_class_recall_guard": False,
            "cardinality_loss_weight": 0.0,
            "count_loss_weight": 0.0,
            "quality_loss_weight": 0.0,
            "auxiliary_loss_weight": 0.0,
            "count_objectness_consistency_weight": 0.0,
            "metric_learning_loss_weight": 0.0,
            "teacher_focus_binary_loss_weight": 0.0,
            "distillation_weight": 0.0,
            "pretrained_distillation": False,
            "distillation_teacher_csv": "",
            "hard_sample_manifest": "",
            "sample_weight_manifest": "",
            "save_last_checkpoint": True,
        }
    )
    return profile


def _state_delta_summary(
    reference_state: Mapping[str, Tensor],
    candidate_state: Mapping[str, Tensor],
    keys: Sequence[str],
) -> Dict[str, object]:
    changed = 0
    squared_delta = 0.0
    max_absolute_delta = 0.0
    for key in keys:
        reference = reference_state[key].detach().cpu().float()
        candidate = candidate_state[key].detach().cpu().float()
        delta = candidate - reference
        maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
        if maximum > 0.0:
            changed += 1
        squared_delta += float(delta.square().sum().item())
        max_absolute_delta = max(max_absolute_delta, maximum)
    return {
        "changed_key_count": int(changed),
        "l2_delta": float(squared_delta**0.5),
        "max_absolute_delta": float(max_absolute_delta),
    }


def _verify_classifier_detector_feature_identity(
    classifier: nn.Module,
    detector: nn.Module,
    *,
    image_size: int,
) -> Dict[str, object]:
    classifier.eval()
    detector.eval()
    image = torch.rand(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        classifier_features = classifier.forward_features(image)
        detector_features = detector.forward_features(image)
        detector_output = detector.forward_heads(detector_features)
    patch_delta = float(
        (classifier_features["patches"] - detector_features["patches"]).abs().max().item()
    )
    pooled_delta = float(
        (classifier_features["pooled"] - detector_features["pooled"]).abs().max().item()
    )
    return {
        "patch_max_absolute_delta": patch_delta,
        "pooled_max_absolute_delta": pooled_delta,
        "detector_logits_shape": list(detector_output["logits"].shape),
        "detector_boxes_shape": list(detector_output["boxes"].shape),
        "detector_objectness_shape": list(detector_output["objectness_logits"].shape),
        "passed": patch_delta == 0.0 and pooled_delta == 0.0,
    }


def _verify_classifier_forward(model: nn.Module, *, image_size: int, class_count: int) -> Dict[str, object]:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.rand(1, 3, int(image_size), int(image_size)))
    if not torch.is_tensor(logits):
        raise ValueError("Round-trip classifier did not return tensor logits")
    expected = (1, int(class_count))
    return {"logits_shape": list(logits.shape), "passed": tuple(logits.shape) == expected}


def _classifier_to_detector(
    source: Mapping[str, object],
    *,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    source_type = str(source["model_config"].get("model_type", ""))
    if source_type in DETECTION_MODEL_TYPES:
        raise ValueError(f"Expected classification checkpoint, got model_type={source_type}")
    torch.manual_seed(int(args.seed))
    config = dict(source["model_config"])
    config.update(
        {
            "model_type": str(args.detector_model_type),
            "num_queries": int(args.num_queries),
            "decoder_depth": int(args.decoder_depth),
            "decoder_ffn_dim": int(args.decoder_ffn_dim),
            "decoder_dropout": float(args.decoder_dropout),
            "auxiliary_decoder_outputs": False,
            "quality_head": False,
            "count_head": False,
        }
    )
    class_count = len(source["class_names"])
    classifier = build_model_from_checkpoint(dict(source))
    detector = create_model(num_classes=class_count, model_config=config)
    source_state = source["model_state"]
    merged, transfer = merge_matching_state(
        source_state,
        detector.state_dict(),
        encoder_only=False,
    )
    source_only = transfer["source_candidates_without_target"]
    if transfer["shape_mismatches"]:
        raise ValueError(f"Classifier/detector state shape mismatch: {transfer['shape_mismatches']}")
    if sorted(source_only) != ["head.bias", "head.weight"]:
        raise ValueError(f"Unexpected classifier-only state keys: {source_only}")
    detector.load_state_dict(merged, strict=True)
    verification = _verify_classifier_detector_feature_identity(
        classifier,
        detector,
        image_size=int(config.get("image_size", 256)),
    )
    if not verification["passed"]:
        raise ValueError(f"Classifier/detector feature identity failed: {verification}")
    summary = {
        "mode": "classifier_to_detector",
        "source_model_type": source_type,
        "target_model_type": str(config["model_type"]),
        "class_names": list(source["class_names"]),
        "source_parameters": int(sum(parameter.numel() for parameter in classifier.parameters())),
        "target_parameters": int(sum(parameter.numel() for parameter in detector.parameters())),
        "transfer": transfer,
        "source_only_keys": source_only,
        "feature_identity": verification,
        "detector_config": {
            "num_queries": int(args.num_queries),
            "decoder_depth": int(args.decoder_depth),
            "decoder_ffn_dim": int(args.decoder_ffn_dim),
            "decoder_dropout": float(args.decoder_dropout),
        },
        "required_runtime": {
            "full_image_detection": True,
            "skip_final_test": True,
            "trainer_arguments": list(DETECTOR_RUNTIME_ARGUMENTS),
            "reason": (
                "The detector stage must preserve the original yolo_f frame and all "
                "object boxes, and it must not consume the held-out test split."
            ),
        },
    }
    train_profile = detector_train_profile(
        source.get("train_config", {}),
        args=args,
    )
    summary["detector_train_profile"] = {
        key: train_profile[key]
        for key in (
            "batch_size",
            "grad_accum_steps",
            "epochs",
            "learning_rate",
            "backbone_lr_scale",
            "min_learning_rate",
            "warmup_epochs",
            "scheduler_total_epochs",
            "early_stopping_patience",
            "max_train_batches",
            "max_val_batches",
            "stage1_epochs",
            "best_metric",
            "cls_loss_weight",
            "bbox_l1_loss_weight",
            "bbox_giou_loss_weight",
            "objectness_loss_weight",
        )
    }
    return merged, {
        "model_config": config,
        "train_config": train_profile,
        "summary": summary,
    }


def _detector_to_classifier(
    source: Mapping[str, object],
    reference: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    source_type = str(source["model_config"].get("model_type", ""))
    reference_type = str(reference["model_config"].get("model_type", ""))
    if source_type not in DETECTION_MODEL_TYPES:
        raise ValueError(f"Expected detection checkpoint, got model_type={source_type}")
    if reference_type in DETECTION_MODEL_TYPES:
        raise ValueError(f"Reference checkpoint must be a classifier, got {reference_type}")
    if list(source["class_names"]) != list(reference["class_names"]):
        raise ValueError("Detector and classifier class_names differ")
    reference_state = reference["model_state"]
    merged, transfer = merge_matching_state(
        source["model_state"],
        reference_state,
        encoder_only=True,
    )
    if transfer["shape_mismatches"] or transfer["source_candidates_without_target"]:
        raise ValueError(f"Unsafe detector-to-classifier transfer: {transfer}")
    transferred_keys = list(transfer["transferred_keys"])
    present_prefixes = {
        prefix
        for prefix in ENCODER_TRANSFER_PREFIXES
        if any(key == prefix or key.startswith(prefix) for key in transferred_keys)
    }
    if present_prefixes != set(ENCODER_TRANSFER_PREFIXES):
        missing = sorted(set(ENCODER_TRANSFER_PREFIXES).difference(present_prefixes))
        raise ValueError(f"Missing required encoder prefixes: {missing}")
    for head_key in ("head.weight", "head.bias"):
        if not torch.equal(merged[head_key], reference_state[head_key]):
            raise ValueError(f"Classifier head changed during detector transfer: {head_key}")
    classifier = build_model_from_checkpoint(dict(reference))
    classifier.load_state_dict(merged, strict=True)
    verification = _verify_classifier_forward(
        classifier,
        image_size=int(reference["model_config"].get("image_size", 256)),
        class_count=len(reference["class_names"]),
    )
    if not verification["passed"]:
        raise ValueError(f"Round-trip classifier forward failed: {verification}")
    summary = {
        "mode": "detector_to_classifier",
        "source_model_type": source_type,
        "target_model_type": reference_type,
        "class_names": list(reference["class_names"]),
        "transfer": transfer,
        "encoder_delta_vs_reference": _state_delta_summary(
            reference_state,
            merged,
            transferred_keys,
        ),
        "classifier_head_preserved_bit_exact": True,
        "classifier_forward": verification,
    }
    return merged, {
        "model_config": dict(reference["model_config"]),
        "train_config": dict(reference.get("train_config", {})),
        "summary": summary,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_conversion(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.num_queries) < 1 or int(args.decoder_depth) < 1 or int(args.decoder_ffn_dim) < 1:
        raise ValueError("Detector query/depth/FFN values must be positive")
    if float(args.decoder_dropout) < 0.0:
        raise ValueError("decoder-dropout must be non-negative")
    if (
        int(args.train_batch_size) < 1
        or int(args.train_grad_accum_steps) < 1
        or int(args.train_epochs) < 1
        or int(args.train_patience) < 1
        or int(args.max_train_batches) < 0
        or int(args.max_val_batches) < 0
        or int(args.num_workers) < 0
        or int(args.eval_num_workers) < 0
    ):
        raise ValueError("Invalid detector train-profile integer value")
    if (
        float(args.train_learning_rate) <= 0.0
        or float(args.train_backbone_lr_scale) <= 0.0
        or float(args.train_min_learning_rate) < 0.0
    ):
        raise ValueError("Invalid detector train-profile learning-rate value")
    source_path = Path(args.source_checkpoint).resolve()
    output_path = Path(args.output_checkpoint).resolve()
    if output_path.suffix.lower() != ".pt":
        raise ValueError("output-checkpoint must use the .pt suffix")
    if output_path == source_path:
        raise ValueError("Refusing to overwrite the source checkpoint")
    source = _load_checkpoint(source_path)
    if str(args.mode) == "classifier-to-detector":
        state, result = _classifier_to_detector(source, args=args)
        reference = source
        reference_path = source_path
    else:
        if args.reference_classifier_checkpoint is None:
            raise ValueError("detector-to-classifier requires --reference-classifier-checkpoint")
        reference_path = Path(args.reference_classifier_checkpoint).resolve()
        reference = _load_checkpoint(reference_path)
        if output_path == reference_path:
            raise ValueError("Refusing to overwrite the reference classifier checkpoint")
        state, result = _detector_to_classifier(source, reference)
    summary = dict(result["summary"])
    summary.update(
        {
            "source_checkpoint": str(source_path),
            "reference_classifier_checkpoint": str(reference_path),
            "output_checkpoint": str(output_path),
            "seed": int(args.seed),
            "preflight_only": bool(args.preflight_only),
            "raw_dataset_touched": False,
            "test_split_used": False,
            "optimizer_state_preserved": False,
            "scheduler_state_preserved": False,
            "training_progress_preserved": False,
        }
    )
    if bool(args.preflight_only):
        return summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fresh_checkpoint_payload(
        reference,
        model_state=state,
        model_config=result["model_config"],
        transfer_summary=summary,
        train_config=result["train_config"],
    )
    torch.save(payload, output_path)
    summary["output_size_bytes"] = int(output_path.stat().st_size)
    summary["output_sha256"] = _file_sha256(output_path)
    summary_path = output_path.with_suffix(output_path.suffix + ".transfer.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_conversion(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
