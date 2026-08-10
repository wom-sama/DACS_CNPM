from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from trkh.core.utils import ensure_dir, json_dump


READOUT_PREFIXES: Tuple[str, ...] = (
    "norm.",
    "head.",
    "fine_grained_pool.",
    "pairwise_margin_norm.",
    "pairwise_margin_head.",
    "cnn_fusion_norm.",
    "cnn_fusion_head.",
    "bbox_spatial_fusion_head.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_state(
    checkpoint: Mapping[str, object],
    *,
    checkpoint_name: str,
) -> Mapping[str, Tensor]:
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{checkpoint_name} checkpoint is missing nonempty model_state.")
    for name, tensor in state.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError(
                f"{checkpoint_name} model_state must map string names to tensors."
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"{checkpoint_name} tensor is nonfinite: {name}")
    return state  # type: ignore[return-value]


def _model_config(
    checkpoint: Mapping[str, object],
    *,
    checkpoint_name: str,
) -> Mapping[str, object]:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{checkpoint_name} checkpoint is missing model_config.")
    return config


def _parse_one_based_layers(value: object) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise ValueError(f"Unsupported layer specification: {value!r}")
    layers = tuple(int(part) for part in parts)
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"Layer numbers must be one-based positive integers: {layers}")
    return layers


def splice_prefixes(*, depth: int, fork_after_block: int) -> Tuple[str, ...]:
    if depth < 2:
        raise ValueError("Late-member splice requires depth >= 2.")
    if not 1 <= int(fork_after_block) < int(depth):
        raise ValueError(
            f"fork_after_block must be in [1, {int(depth) - 1}], got {fork_after_block}."
        )
    block_prefixes = tuple(
        f"blocks.{block_index}."
        for block_index in range(int(fork_after_block), int(depth))
    )
    return block_prefixes + READOUT_PREFIXES


def _state_sha256(state: Mapping[str, Tensor], keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(keys):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_late_member_splice_checkpoint(
    keeper_checkpoint: Mapping[str, object],
    candidate_checkpoint: Mapping[str, object],
    *,
    keeper_path: Path,
    candidate_path: Path,
    keeper_sha256: str,
    candidate_sha256: str,
    fork_after_block: int = 6,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    keeper_state = _validated_state(keeper_checkpoint, checkpoint_name="keeper")
    candidate_state = _validated_state(
        candidate_checkpoint,
        checkpoint_name="candidate",
    )
    keeper_config = _model_config(keeper_checkpoint, checkpoint_name="keeper")
    candidate_config = _model_config(candidate_checkpoint, checkpoint_name="candidate")

    if set(keeper_state) != set(candidate_state):
        missing = sorted(set(keeper_state) - set(candidate_state))
        unexpected = sorted(set(candidate_state) - set(keeper_state))
        raise ValueError(
            "Keeper/candidate state schemas differ: "
            f"missing_in_candidate={missing}, unexpected_in_candidate={unexpected}"
        )
    for name, keeper_tensor in keeper_state.items():
        candidate_tensor = candidate_state[name]
        if tuple(keeper_tensor.shape) != tuple(candidate_tensor.shape):
            raise ValueError(f"Keeper/candidate tensor shape mismatch for {name}.")
        if keeper_tensor.dtype != candidate_tensor.dtype:
            raise ValueError(f"Keeper/candidate tensor dtype mismatch for {name}.")

    depth = int(keeper_config.get("depth", 0) or 0)
    candidate_depth = int(candidate_config.get("depth", 0) or 0)
    if depth != candidate_depth:
        raise ValueError(
            f"Keeper/candidate depth mismatch: keeper={depth}, candidate={candidate_depth}."
        )
    prefixes = splice_prefixes(depth=depth, fork_after_block=fork_after_block)
    prune_layers = _parse_one_based_layers(keeper_config.get("token_prune_layers"))
    if prune_layers and max(prune_layers) > int(fork_after_block):
        raise ValueError(
            "Late-member splice requires all token pruning to finish at or before "
            f"the fork; prune_layers={prune_layers}, fork_after_block={fork_after_block}."
        )

    prefix_matches = {
        prefix: sorted(name for name in keeper_state if name.startswith(prefix))
        for prefix in prefixes
    }
    missing_prefixes = [prefix for prefix, names in prefix_matches.items() if not names]
    if missing_prefixes:
        raise ValueError(f"Splice prefixes have no state tensors: {missing_prefixes}")
    copied_keys = sorted(
        name for name in keeper_state if any(name.startswith(prefix) for prefix in prefixes)
    )
    retained_keys = sorted(set(keeper_state) - set(copied_keys))
    if not copied_keys or not retained_keys:
        raise ValueError("Splice must contain both copied and retained tensors.")

    output_state: Dict[str, Tensor] = {}
    for name, keeper_tensor in keeper_state.items():
        source = candidate_state[name] if name in copied_keys else keeper_tensor
        output_state[name] = source.detach().cpu().clone()

    for name in copied_keys:
        if not torch.equal(output_state[name], candidate_state[name].detach().cpu()):
            raise RuntimeError(f"Copied tensor verification failed for {name}.")
    for name in retained_keys:
        if not torch.equal(output_state[name], keeper_state[name].detach().cpu()):
            raise RuntimeError(f"Retained tensor verification failed for {name}.")

    provenance: Dict[str, object] = {
        "mode": "late_member_direct_splice_readiness",
        "test_data_used": False,
        "keeper_checkpoint": str(Path(keeper_path).resolve()),
        "keeper_checkpoint_sha256": str(keeper_sha256),
        "candidate_checkpoint": str(Path(candidate_path).resolve()),
        "candidate_checkpoint_sha256": str(candidate_sha256),
        "fork_after_block": int(fork_after_block),
        "depth": int(depth),
        "token_prune_layers": list(prune_layers),
        "copied_prefixes": list(prefixes),
        "copied_tensor_count": len(copied_keys),
        "retained_tensor_count": len(retained_keys),
        "copied_keys": copied_keys,
        "retained_keys": retained_keys,
        "copied_candidate_state_sha256": _state_sha256(candidate_state, copied_keys),
        "retained_keeper_state_sha256": _state_sha256(keeper_state, retained_keys),
        "output_state_sha256": _state_sha256(output_state, sorted(output_state)),
        "untouched_tensors_bit_identical": True,
    }

    output: Dict[str, object] = dict(keeper_checkpoint)
    output["model_state"] = output_state
    for key in (
        "ema_model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "train_model_state",
        "metrics",
        "calibration",
    ):
        output.pop(key, None)
    output["checkpoint_kind"] = "late_member_splice_diagnostic"
    output["checkpoint_weight_source"] = "late_member_splice"
    output["validation_weight_source"] = "late_member_splice"
    output["late_member_splice"] = provenance
    return output, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the one locked keeper/scratch late-member splice checkpoint for "
            "validation-only architecture readiness."
        )
    )
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fork-after-block", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keeper_path = args.keeper.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not keeper_path.is_file():
        raise FileNotFoundError(keeper_path)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if keeper_path == candidate_path:
        raise ValueError("Keeper and candidate checkpoints must differ.")
    if output_path in {keeper_path, candidate_path}:
        raise ValueError("Splice output must differ from both source checkpoints.")
    if output_path.exists():
        raise FileExistsError(output_path)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)

    keeper_sha256 = sha256_file(keeper_path)
    candidate_sha256 = sha256_file(candidate_path)
    keeper_checkpoint = torch.load(keeper_path, map_location="cpu", weights_only=False)
    candidate_checkpoint = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(keeper_checkpoint, Mapping):
        raise ValueError("Keeper checkpoint payload must be a mapping.")
    if not isinstance(candidate_checkpoint, Mapping):
        raise ValueError("Candidate checkpoint payload must be a mapping.")
    output, provenance = build_late_member_splice_checkpoint(
        keeper_checkpoint,
        candidate_checkpoint,
        keeper_path=keeper_path,
        candidate_path=candidate_path,
        keeper_sha256=keeper_sha256,
        candidate_sha256=candidate_sha256,
        fork_after_block=args.fork_after_block,
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
