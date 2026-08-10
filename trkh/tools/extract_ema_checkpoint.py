from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor

from trkh.core.utils import ensure_dir, json_dump


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_state(checkpoint: Mapping[str, object], key: str) -> Mapping[str, Tensor]:
    state = checkpoint.get(key)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint is missing nonempty {key}.")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError(f"{key} must be a string-to-tensor state dict.")
    return state  # type: ignore[return-value]


def build_ema_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    source_path: Path,
    source_sha256: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    model_state = _validated_state(checkpoint, "model_state")
    ema_state = _validated_state(checkpoint, "ema_model_state")
    if set(model_state) != set(ema_state):
        raise ValueError("model_state and ema_model_state keys do not match.")
    for name, tensor in model_state.items():
        ema_tensor = ema_state[name]
        if tuple(tensor.shape) != tuple(ema_tensor.shape):
            raise ValueError(f"EMA tensor shape mismatch for {name}.")

    output: Dict[str, object] = dict(checkpoint)
    output["model_state"] = {
        name: tensor.detach().cpu().clone() for name, tensor in ema_state.items()
    }
    for key in (
        "ema_model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "train_model_state",
    ):
        output.pop(key, None)
    provenance = {
        "source_checkpoint": str(Path(source_path).resolve()),
        "source_checkpoint_sha256": str(source_sha256),
        "source_state_key": "ema_model_state",
        "epoch": checkpoint.get("epoch"),
        "ema_updates": checkpoint.get("ema_updates"),
        "tensor_count": len(ema_state),
    }
    output["checkpoint_kind"] = "ema_export"
    output["checkpoint_weight_source"] = "ema"
    output["validation_weight_source"] = "ema"
    output["ema_export"] = provenance
    return output, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the validation EMA weights from a TRKH last checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path == source_path:
        raise ValueError("EMA export output must differ from the source checkpoint.")
    if output_path.exists():
        raise FileExistsError(output_path)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)

    source_sha256 = sha256_file(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    output, provenance = build_ema_checkpoint(
        checkpoint,
        source_path=source_path,
        source_sha256=source_sha256,
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
