from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import timm

from trkh.core.config import load_data_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a timm ImageFolder baseline checkpoint into TRKH checkpoint format, "
            "with classifier row remapping from checkpoint class order to data.yaml order."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("auto", "raw", "mango"), default="raw")
    parser.add_argument("--expected-num-classes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--research-track",
        choices=("pretrained", "no_pretrain"),
        required=True,
    )
    parser.add_argument("--expected-source-checkpoint-sha256", default="")
    parser.add_argument("--pretrained-source-url", default="")
    parser.add_argument("--pretrained-source-revision", default="")
    parser.add_argument("--pretrained-source-license", default="")
    parser.add_argument("--pretrained-initializer-sha256", default="")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid baseline checkpoint: {path}")
    return checkpoint


def _reorder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    baseline_classes: List[str],
    target_classes: List[str],
) -> Dict[str, torch.Tensor]:
    index_by_name = {name: index for index, name in enumerate(baseline_classes)}
    missing = [name for name in target_classes if name not in index_by_name]
    if missing:
        raise ValueError(f"Baseline checkpoint thieu class trong data.yaml: {missing}")
    row_order = torch.tensor([index_by_name[name] for name in target_classes], dtype=torch.long)

    remapped: Dict[str, torch.Tensor] = {}
    num_classes = len(target_classes)
    for key, value in state_dict.items():
        key_lower = str(key).lower()
        is_classifier_tensor = any(
            token in key_lower for token in ("classifier", "head", "fc", "last_linear")
        )
        if (
            is_classifier_tensor
            and torch.is_tensor(value)
            and value.ndim >= 1
            and int(value.shape[0]) == num_classes
        ):
            remapped[key] = value.index_select(0, row_order)
        else:
            remapped[key] = value
    return remapped


def main() -> None:
    args = parse_args()
    source_checkpoint_sha256 = _sha256_file(args.checkpoint)
    expected_source_sha256 = str(args.expected_source_checkpoint_sha256 or "").strip().lower()
    if expected_source_sha256 and source_checkpoint_sha256 != expected_source_sha256:
        raise ValueError(
            "Source checkpoint SHA-256 mismatch: "
            f"observed={source_checkpoint_sha256}, expected={expected_source_sha256}."
        )
    if args.research_track == "pretrained":
        required_provenance = {
            "--expected-source-checkpoint-sha256": expected_source_sha256,
            "--pretrained-source-url": args.pretrained_source_url,
            "--pretrained-source-revision": args.pretrained_source_revision,
            "--pretrained-source-license": args.pretrained_source_license,
            "--pretrained-initializer-sha256": args.pretrained_initializer_sha256,
        }
        missing = [name for name, value in required_provenance.items() if not str(value).strip()]
        if missing:
            raise ValueError(
                "Pretrained import provenance is incomplete: " + ", ".join(missing)
            )
    baseline = _load_checkpoint(args.checkpoint)
    if "model" not in baseline or "classes" not in baseline:
        raise ValueError("Checkpoint baseline phai co keys 'model' va 'classes'.")
    state_dict = baseline["model"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint baseline key 'model' khong phai state_dict.")
    baseline_classes = [str(name) for name in baseline["classes"]]
    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    target_classes = list(data_spec.class_names)
    args_dict = baseline.get("args", {})
    model_name = (
        str(args_dict.get("model", "")).strip()
        if isinstance(args_dict, Mapping)
        else ""
    )
    if not model_name:
        raise ValueError("Khong tim thay args.model trong checkpoint baseline.")
    model_data_config = timm.data.resolve_model_data_config(
        timm.create_model(model_name, pretrained=False)
    )
    input_mean = tuple(float(value) for value in model_data_config.get("mean", (0.485, 0.456, 0.406)))
    input_std = tuple(float(value) for value in model_data_config.get("std", (0.229, 0.224, 0.225)))

    remapped_state = _reorder_state_dict(state_dict, baseline_classes, target_classes)
    payload = {
        "model_state": remapped_state,
        "class_names": target_classes,
        "data_yaml": str(data_spec.data_yaml.resolve()),
        "model_config": {
            "model_type": "timm_classifier",
            "timm_model_name": model_name,
            "research_track": str(args.research_track),
            "pretrained": args.research_track == "pretrained",
            "pretrained_source_url": str(args.pretrained_source_url),
            "pretrained_source_revision": str(args.pretrained_source_revision),
            "pretrained_source_license": str(args.pretrained_source_license),
            "image_size": int(args.image_size),
            "input_mean": input_mean,
            "input_std": input_std,
        },
        "train_config": {
            "classification_loss": "cross_entropy",
            "label_smoothing": 0.0,
            "source": "imported_timm_baseline_checkpoint",
        },
        "augmentation_config": {
            "resize_mode": "stretch",
        },
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_classes": baseline_classes,
        "class_remap": {
            target_name: int(baseline_classes.index(target_name))
            for target_name in target_classes
        },
    }
    if args.research_track == "pretrained":
        payload["pretrained_provenance"] = {
            "schema_version": 1,
            "research_track": "pretrained",
            "initialization_source": "imported_finetuned_timm_checkpoint",
            "external_initialization_replayed": False,
            "fine_tuned_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": source_checkpoint_sha256,
            },
            "imagenet_initializer": {
                "model_name": model_name,
                "source_url": str(args.pretrained_source_url),
                "source_revision": str(args.pretrained_source_revision),
                "license_id": str(args.pretrained_source_license),
                "sha256": str(args.pretrained_initializer_sha256).strip().lower(),
            },
            "class_remap": dict(payload["class_remap"]),
            "input_contract": {
                "image_size": int(args.image_size),
                "resize_mode": "stretch",
                "mean": list(input_mean),
                "std": list(input_std),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "model_name": model_name,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "research_track": str(args.research_track),
        "target_data_yaml": str(data_spec.data_yaml.resolve()),
        "target_classes": target_classes,
        "source_classes": baseline_classes,
        "class_remap": payload["class_remap"],
    }
    (args.output.parent / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
