from __future__ import annotations

import argparse
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
    return parser.parse_args()


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
            "pretrained": False,
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
        "source_classes": baseline_classes,
        "class_remap": {
            target_name: int(baseline_classes.index(target_name))
            for target_name in target_classes
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "model_name": model_name,
        "source_checkpoint": str(args.checkpoint.resolve()),
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
