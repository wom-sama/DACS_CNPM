from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from trkh.models.model import build_model_from_checkpoint
from trkh.models.precision_ensemble import PrecisionEnsembleClassifier
from trkh.tools.apply_frozen_precision_ensemble import _load_frozen_protocol


EVAL_AUGMENTATION_DEFAULTS: Dict[str, object] = {
    "resize_mode": "pad",
    "crop_margin_ratio": 0.05,
    "illumination_normalization": False,
    "illumination_normalization_strength": 0.0,
    "foreground_crop_mode": "none",
    "foreground_crop_margin_ratio": 0.08,
    "foreground_crop_min_mask_area_ratio": 0.03,
    "foreground_crop_max_mask_area_ratio": 0.92,
    "foreground_crop_max_crop_area_ratio": 0.98,
    "background_suppression_mode": "none",
    "background_suppression_margin": 0.08,
    "background_suppression_blur_radius": 7.0,
    "surface_detail_amplification_mode": "none",
    "surface_detail_amplification_strength": 0.0,
    "surface_detail_amplification_blur_radius": 1.25,
    "surface_detail_amplification_foreground_weight": 0.85,
    "eval_surface_detail_amplification": False,
    "classification_source_context_aux": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(
    state: Mapping[str, Tensor],
    *,
    prefix: str = "",
    strip_prefix: bool = False,
) -> str:
    selected = sorted(name for name in state if not prefix or name.startswith(prefix))
    if not selected:
        raise ValueError(f"No state tensors matched prefix={prefix!r}.")
    digest = hashlib.sha256()
    for name in selected:
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError("model_state must map string names to tensors.")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"Nonfinite state tensor: {name}")
        canonical_name = name[len(prefix) :] if prefix and strip_prefix else name
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(canonical_name.encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(cpu_tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expected_sha256(value: str, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters.")
    return normalized


def _verify_file_hash(path: Path, expected_sha256: str, *, name: str) -> str:
    expected = _expected_sha256(expected_sha256, name=name)
    actual = sha256_file(path).lower()
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: expected={expected}, actual={actual}")
    return actual


def _checkpoint_mapping(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    for key in ("model_state", "model_config", "class_names"):
        if key not in payload:
            raise ValueError(f"Checkpoint is missing {key}: {path}")
    if not isinstance(payload["model_state"], Mapping) or not payload["model_state"]:
        raise ValueError(f"Checkpoint has an empty model_state: {path}")
    if not isinstance(payload["model_config"], Mapping):
        raise ValueError(f"Checkpoint has an invalid model_config: {path}")
    if not isinstance(payload["class_names"], Sequence) or isinstance(
        payload["class_names"], (str, bytes)
    ):
        raise ValueError(f"Checkpoint has an invalid class_names list: {path}")
    return payload


def _normalized_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_value(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Nonfinite configuration value: {value!r}")
        return parsed
    return str(value).strip().lower() if isinstance(value, str) else value


def _eval_semantics(checkpoint: Mapping[str, object]) -> Dict[str, object]:
    model_config = checkpoint["model_config"]
    assert isinstance(model_config, Mapping)
    augmentation = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation, Mapping):
        augmentation = {}
    data_summary = checkpoint.get("data_summary", {})
    if not isinstance(data_summary, Mapping):
        data_summary = {}
    result: Dict[str, object] = {
        "image_size": int(model_config.get("image_size", 224)),
        "temporal_frames": int(model_config.get("temporal_frames", 1)),
        "input_mean": _normalized_value(
            model_config.get("input_mean", (0.485, 0.456, 0.406))
        ),
        "input_std": _normalized_value(
            model_config.get("input_std", (0.229, 0.224, 0.225))
        ),
        "crop_to_primary_object": bool(data_summary.get("crop_to_primary_object", True)),
        "classification_object_crops": bool(
            data_summary.get("classification_object_crops", True)
        ),
    }
    for key, default in EVAL_AUGMENTATION_DEFAULTS.items():
        result[key] = _normalized_value(augmentation.get(key, default))
    return result


def _validate_member_compatibility(
    keeper: Mapping[str, object],
    candidate: Mapping[str, object],
    protocol: Mapping[str, object],
) -> Tuple[Sequence[str], Dict[str, object]]:
    keeper_classes = [str(value) for value in keeper["class_names"]]
    candidate_classes = [str(value) for value in candidate["class_names"]]
    protocol_classes = [str(value) for value in protocol.get("class_names", [])]
    if keeper_classes != candidate_classes or keeper_classes != protocol_classes:
        raise ValueError(
            "Class order mismatch across keeper, candidate, and frozen protocol: "
            f"keeper={keeper_classes}, candidate={candidate_classes}, "
            f"protocol={protocol_classes}"
        )
    keeper_semantics = _eval_semantics(keeper)
    candidate_semantics = _eval_semantics(candidate)
    if keeper_semantics != candidate_semantics:
        differences = {
            key: {"keeper": keeper_semantics.get(key), "candidate": candidate_semantics.get(key)}
            for key in sorted(set(keeper_semantics) | set(candidate_semantics))
            if keeper_semantics.get(key) != candidate_semantics.get(key)
        }
        raise ValueError(f"Evaluation semantics differ between members: {differences}")
    if int(keeper_semantics["temporal_frames"]) != 1:
        raise ValueError("Precision ensemble is locked to temporal_frames=1.")
    return keeper_classes, keeper_semantics


def build_precision_ensemble_payload(
    keeper_checkpoint: Mapping[str, object],
    candidate_checkpoint: Mapping[str, object],
    protocol: Mapping[str, object],
    *,
    keeper_path: Path,
    candidate_path: Path,
    protocol_path: Path,
    keeper_sha256: str,
    candidate_sha256: str,
    protocol_sha256: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    class_names, eval_semantics = _validate_member_compatibility(
        keeper_checkpoint,
        candidate_checkpoint,
        protocol,
    )
    locked = protocol.get("locked")
    if not isinstance(locked, Mapping):
        raise ValueError("Frozen protocol has no locked rule.")
    candidate_weight = float(locked["candidate_weight"])
    keeper_weight = float(locked["keeper_weight"])
    focus_margin_offset = float(locked["focus_margin_offset"])
    focus_class = int(protocol["focus_class"])
    if abs((candidate_weight + keeper_weight) - 1.0) > 1e-12:
        raise ValueError("Frozen member weights do not sum to one.")

    keeper_model = build_model_from_checkpoint(dict(keeper_checkpoint)).eval()
    candidate_model = build_model_from_checkpoint(dict(candidate_checkpoint)).eval()
    model = PrecisionEnsembleClassifier(
        keeper_model,
        candidate_model,
        num_classes=len(class_names),
        candidate_weight=candidate_weight,
        focus_class=focus_class,
        focus_margin_offset=focus_margin_offset,
    ).eval()
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    keeper_state = keeper_checkpoint["model_state"]
    candidate_state = candidate_checkpoint["model_state"]
    assert isinstance(keeper_state, Mapping)
    assert isinstance(candidate_state, Mapping)
    keeper_source_state_sha256 = state_dict_sha256(keeper_state)  # type: ignore[arg-type]
    candidate_source_state_sha256 = state_dict_sha256(candidate_state)  # type: ignore[arg-type]
    keeper_packaged_state_sha256 = state_dict_sha256(
        model_state,
        prefix="keeper_model.",
        strip_prefix=True,
    )
    candidate_packaged_state_sha256 = state_dict_sha256(
        model_state,
        prefix="candidate_model.",
        strip_prefix=True,
    )
    if keeper_packaged_state_sha256 != keeper_source_state_sha256:
        raise RuntimeError("Packaged keeper state is not bit-identical to its source.")
    if candidate_packaged_state_sha256 != candidate_source_state_sha256:
        raise RuntimeError("Packaged candidate state is not bit-identical to its source.")

    keeper_model_config = dict(keeper_checkpoint["model_config"])
    candidate_model_config = dict(candidate_checkpoint["model_config"])
    model_config: Dict[str, object] = {
        "model_type": "precision_ensemble",
        "image_size": int(eval_semantics["image_size"]),
        "temporal_frames": 1,
        "input_mean": list(eval_semantics["input_mean"]),
        "input_std": list(eval_semantics["input_std"]),
        "precision_ensemble": {
            "format_version": 1,
            "candidate_weight": candidate_weight,
            "keeper_weight": keeper_weight,
            "focus_class": focus_class,
            "focus_margin_offset": focus_margin_offset,
            "minimum_probability": 1e-8,
            "requires_spatial_metadata": True,
            "spatial_metadata_inputs": ["image_valid_mask", "bbox"],
            "supports_dynamic_batch": False,
            "certified_batch_sizes": [1],
            "certified_deployment_backends": ["onnx", "tensorrt"],
            "member_model_configs": {
                "keeper": keeper_model_config,
                "candidate": candidate_model_config,
            },
        },
    }
    provenance: Dict[str, object] = {
        "mode": "frozen_precision_ensemble_single_checkpoint",
        "format_version": 1,
        "test_data_used": False,
        "selection_parameters_frozen": True,
        "keeper_checkpoint": str(Path(keeper_path).resolve()),
        "keeper_checkpoint_sha256": keeper_sha256,
        "candidate_checkpoint": str(Path(candidate_path).resolve()),
        "candidate_checkpoint_sha256": candidate_sha256,
        "frozen_protocol": str(Path(protocol_path).resolve()),
        "frozen_protocol_sha256": protocol_sha256,
        "class_names": list(class_names),
        "candidate_weight": candidate_weight,
        "keeper_weight": keeper_weight,
        "focus_class": focus_class,
        "focus_margin_offset": focus_margin_offset,
        "requires_spatial_metadata": True,
        "spatial_metadata_inputs": ["image_valid_mask", "bbox"],
        "supports_dynamic_batch": False,
        "certified_batch_sizes": [1],
        "certified_deployment_backends": ["onnx", "tensorrt"],
        "eval_semantics": eval_semantics,
        "keeper_source_state_sha256": keeper_source_state_sha256,
        "candidate_source_state_sha256": candidate_source_state_sha256,
        "keeper_packaged_state_sha256": keeper_packaged_state_sha256,
        "candidate_packaged_state_sha256": candidate_packaged_state_sha256,
        "packaged_state_sha256": state_dict_sha256(model_state),
    }
    payload: Dict[str, object] = {
        "checkpoint_kind": "precision_ensemble_inference",
        "checkpoint_weight_source": "frozen_precision_ensemble",
        "validation_weight_source": "frozen_precision_ensemble",
        "class_names": list(class_names),
        "model_config": model_config,
        "model_state": model_state,
        "augmentation_config": dict(keeper_checkpoint.get("augmentation_config", {})),
        "train_config": dict(keeper_checkpoint.get("train_config", {})),
        "data_summary": dict(keeper_checkpoint.get("data_summary", {})),
        "data_yaml": keeper_checkpoint.get("data_yaml"),
        "precision_ensemble_provenance": provenance,
    }
    reloaded = build_model_from_checkpoint(payload).eval()
    if not isinstance(reloaded, PrecisionEnsembleClassifier):
        raise RuntimeError("Packaged checkpoint did not reload as PrecisionEnsembleClassifier.")
    return payload, provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the validation-frozen precision ensemble as one checkpoint."
    )
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--expected-keeper-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    keeper_path = args.keeper.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    protocol_path = args.frozen_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (keeper_path, candidate_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if keeper_path == candidate_path:
        raise ValueError("Keeper and candidate checkpoints must differ.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output_dir}")

    keeper_sha256 = _verify_file_hash(
        keeper_path,
        args.expected_keeper_sha256,
        name="expected_keeper_sha256",
    )
    candidate_sha256 = _verify_file_hash(
        candidate_path,
        args.expected_candidate_sha256,
        name="expected_candidate_sha256",
    )
    protocol_sha256 = _verify_file_hash(
        protocol_path,
        args.expected_summary_sha256,
        name="expected_summary_sha256",
    )
    protocol = _load_frozen_protocol(protocol_path)
    keeper_checkpoint = _checkpoint_mapping(keeper_path)
    candidate_checkpoint = _checkpoint_mapping(candidate_path)
    payload, provenance = build_precision_ensemble_payload(
        keeper_checkpoint,
        candidate_checkpoint,
        protocol,
        keeper_path=keeper_path,
        candidate_path=candidate_path,
        protocol_path=protocol_path,
        keeper_sha256=keeper_sha256,
        candidate_sha256=candidate_sha256,
        protocol_sha256=protocol_sha256,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "precision_ensemble.pt"
    summary_path = output_dir / "build_summary.json"
    torch.save(payload, checkpoint_path)
    saved_payload = _checkpoint_mapping(checkpoint_path)
    saved_model = build_model_from_checkpoint(dict(saved_payload)).eval()
    if not isinstance(saved_model, PrecisionEnsembleClassifier):
        raise RuntimeError("Saved checkpoint did not strictly reload as precision ensemble.")
    summary = {
        **provenance,
        "output_checkpoint": str(checkpoint_path),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "output_bytes": int(checkpoint_path.stat().st_size),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
