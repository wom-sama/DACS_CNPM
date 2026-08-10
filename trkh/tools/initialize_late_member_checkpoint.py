from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor

from trkh.core.utils import ensure_dir, json_dump
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.build_late_member_splice_checkpoint import sha256_file


def _validated_model_state(checkpoint: Mapping[str, object]) -> Mapping[str, Tensor]:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Keeper checkpoint is missing nonempty model_state.")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError("Keeper model_state must map string names to tensors.")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"Keeper tensor is nonfinite: {name}")
    return state  # type: ignore[return-value]


def initialize_late_member_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    source_path: Path,
    source_sha256: str,
    fork_after_block: int = 6,
    candidate_weight: float = 0.40,
    focus_class: int = 1,
    focus_margin_offset: float = 0.034,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    source_state = _validated_model_state(checkpoint)
    model_config = checkpoint.get("model_config")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping):
        raise ValueError("Keeper checkpoint is missing model_config.")
    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise ValueError("Keeper checkpoint is missing class_names.")
    if str(model_config.get("model_type", "")).strip().lower() != "vit_registers":
        raise ValueError("Late-member initializer currently supports vit_registers only.")
    if not 0 <= int(focus_class) < len(class_names):
        raise ValueError("focus_class is outside the checkpoint class range.")

    updated_config = dict(model_config)
    updated_config.update(
        {
            "late_member_branch": True,
            "late_member_fork_after_block": int(fork_after_block),
            "late_member_candidate_weight": float(candidate_weight),
            "late_member_focus_class": int(focus_class),
            "late_member_focus_margin_offset": float(focus_margin_offset),
        }
    )
    build_payload = dict(checkpoint)
    build_payload["model_config"] = updated_config
    build_payload["model_state"] = dict(source_state)
    model = build_model_from_checkpoint(
        build_payload,
        num_classes=len(class_names),
    )
    output_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    for name, tensor in source_state.items():
        if name not in output_state or not torch.equal(
            output_state[name],
            tensor.detach().cpu(),
        ):
            raise RuntimeError(f"Primary keeper tensor changed during initialization: {name}")

    late_keys = sorted(name for name in output_state if name.startswith("late_member_"))
    if not late_keys:
        raise RuntimeError("Late-member initializer produced no late-member state tensors.")
    source_parameter_count = int(
        sum(tensor.numel() for tensor in source_state.values())
    )
    output_parameter_count = int(sum(tensor.numel() for tensor in output_state.values()))
    provenance: Dict[str, object] = {
        "mode": "keeper_initialized_late_member",
        "test_data_used": False,
        "source_checkpoint": str(Path(source_path).resolve()),
        "source_checkpoint_sha256": str(source_sha256),
        "source_state_key": "model_state",
        "fork_after_block": int(fork_after_block),
        "candidate_weight": float(candidate_weight),
        "focus_class": int(focus_class),
        "focus_margin_offset": float(focus_margin_offset),
        "source_tensor_count": len(source_state),
        "late_member_tensor_count": len(late_keys),
        "output_tensor_count": len(output_state),
        "source_state_numel": source_parameter_count,
        "output_state_numel": output_parameter_count,
        "state_numel_ratio": (
            output_parameter_count / float(max(1, source_parameter_count))
        ),
        "primary_tensors_bit_identical": True,
        "late_member_keys": late_keys,
    }

    output: Dict[str, object] = dict(checkpoint)
    output["model_config"] = updated_config
    output["model_state"] = output_state
    for key in (
        "ema_model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "train_model_state",
        "metrics",
        "calibration",
        "resume_state",
    ):
        output.pop(key, None)
    output["epoch"] = 0
    output["best_epoch"] = 0
    output["best_macro_f1"] = 0.0
    output["epochs_without_improvement"] = 0
    output["checkpoint_kind"] = "late_member_keeper_init"
    output["checkpoint_weight_source"] = "keeper_model_state"
    output["validation_weight_source"] = "keeper_model_state"
    output["late_member_initialization"] = provenance
    return output, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a resume-safe TRKH late-member checkpoint initialized exactly "
            "from the audited keeper model_state."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fork-after-block", type=int, default=6)
    parser.add_argument("--candidate-weight", type=float, default=0.40)
    parser.add_argument("--focus-class", type=int, default=1)
    parser.add_argument("--focus-margin-offset", type=float, default=0.034)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path == source_path:
        raise ValueError("Late-member output must differ from the keeper checkpoint.")
    if output_path.exists():
        raise FileExistsError(output_path)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)

    source_sha256 = sha256_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    output, provenance = initialize_late_member_checkpoint(
        checkpoint,
        source_path=source_path,
        source_sha256=source_sha256,
        fork_after_block=args.fork_after_block,
        candidate_weight=args.candidate_weight,
        focus_class=args.focus_class,
        focus_margin_offset=args.focus_margin_offset,
    )
    ensure_dir(output_path.parent)
    torch.save(output, output_path)
    summary = {
        **provenance,
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
    }
    json_dump(summary_path, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
