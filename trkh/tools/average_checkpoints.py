from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch import Tensor

from trkh.core.utils import ensure_dir, json_dump


AVERAGE_STATE_KEYS = ("model_state", "ema_model_state")


def _parse_weights(raw: str, count: int) -> List[float]:
    if not raw:
        return [1.0 / float(count) for _ in range(count)]
    values = [float(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"--weights needs {count} values, got {len(values)}.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("--weights sum must be positive.")
    return [value / total for value in values]


def _load_checkpoint(path: Path) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dict: {path}")
    if "model_state" not in checkpoint or not isinstance(checkpoint["model_state"], Mapping):
        raise ValueError(f"Checkpoint is missing model_state: {path}")
    return checkpoint


def _compatible_state_dicts(
    checkpoints: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> List[Mapping[str, Tensor]]:
    states: List[Mapping[str, Tensor]] = []
    for index, checkpoint in enumerate(checkpoints):
        state = checkpoint.get(key)
        if state is None:
            return []
        if not isinstance(state, Mapping):
            raise ValueError(f"{key} in checkpoint {index} is not a state dict.")
        states.append(state)  # type: ignore[arg-type]
    first_keys = set(states[0].keys())
    for index, state in enumerate(states[1:], start=1):
        if set(state.keys()) != first_keys:
            missing = sorted(first_keys - set(state.keys()))[:8]
            extra = sorted(set(state.keys()) - first_keys)[:8]
            raise ValueError(
                f"{key} key mismatch at checkpoint {index}: missing={missing}, extra={extra}"
            )
    for name, first_tensor in states[0].items():
        if not torch.is_tensor(first_tensor):
            raise ValueError(f"{key}.{name} is not a tensor in first checkpoint.")
        for index, state in enumerate(states[1:], start=1):
            tensor = state[name]
            if not torch.is_tensor(tensor):
                raise ValueError(f"{key}.{name} is not a tensor in checkpoint {index}.")
            if tuple(tensor.shape) != tuple(first_tensor.shape):
                raise ValueError(
                    f"{key}.{name} shape mismatch at checkpoint {index}: "
                    f"{tuple(tensor.shape)} != {tuple(first_tensor.shape)}"
                )
    return states


def average_state_dicts(
    states: Sequence[Mapping[str, Tensor]],
    weights: Sequence[float],
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    if not states:
        raise ValueError("No state dicts to average.")
    if len(states) != len(weights):
        raise ValueError("states and weights length mismatch.")
    averaged: Dict[str, Tensor] = {}
    copied_non_float = 0
    copied_different_non_float = 0
    for name, first_tensor in states[0].items():
        if first_tensor.is_floating_point() or first_tensor.is_complex():
            acc = torch.zeros_like(first_tensor, dtype=torch.float32)
            for state, weight in zip(states, weights):
                acc = acc + state[name].detach().to(dtype=torch.float32) * float(weight)
            averaged[name] = acc.to(dtype=first_tensor.dtype)
            continue

        copied_non_float += 1
        same = all(torch.equal(first_tensor, state[name]) for state in states[1:])
        if not same:
            copied_different_non_float += 1
        averaged[name] = first_tensor.detach().clone()
    stats = {
        "tensor_count": len(averaged),
        "copied_non_float": copied_non_float,
        "copied_different_non_float": copied_different_non_float,
    }
    return averaged, stats


def build_average_checkpoint(
    checkpoint_paths: Sequence[Path],
    *,
    weights: Sequence[float],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    checkpoints = [_load_checkpoint(path) for path in checkpoint_paths]
    output: Dict[str, object] = dict(checkpoints[0])
    source_metrics = []
    state_stats: Dict[str, Dict[str, int]] = {}
    for key in AVERAGE_STATE_KEYS:
        states = _compatible_state_dicts(checkpoints, key=key)
        if not states:
            continue
        averaged, stats = average_state_dicts(states, weights)
        output[key] = averaged
        state_stats[key] = stats
    if "ema_model_state" in output:
        output["checkpoint_weight_source"] = "model_soup"
        output["validation_weight_source"] = "model_soup"
    for path, checkpoint, weight in zip(checkpoint_paths, checkpoints, weights):
        source_metrics.append(
            {
                "path": str(path),
                "weight": float(weight),
                "epoch": checkpoint.get("epoch"),
                "best_epoch": checkpoint.get("best_epoch"),
                "metrics": checkpoint.get("metrics", {}),
            }
        )
    output["checkpoint_kind"] = "model_soup"
    output["model_soup"] = {
        "sources": [str(path) for path in checkpoint_paths],
        "weights": [float(weight) for weight in weights],
        "state_stats": state_stats,
    }
    output["model_soup_source_metrics"] = source_metrics
    output.pop("optimizer_state", None)
    output.pop("scheduler_state", None)
    output.pop("scaler_state", None)
    summary = {
        "checkpoint_kind": "model_soup",
        "sources": [str(path) for path in checkpoint_paths],
        "weights": [float(weight) for weight in weights],
        "state_stats": state_stats,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average compatible TRKH checkpoints into one model-soup checkpoint."
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [path.expanduser().resolve() for path in args.checkpoints]
    if len(checkpoint_paths) < 2:
        raise ValueError("At least two checkpoints are required.")
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    weights = _parse_weights(str(args.weights), len(checkpoint_paths))
    checkpoint, summary = build_average_checkpoint(checkpoint_paths, weights=weights)
    output_path = args.output.expanduser()
    ensure_dir(output_path.parent)
    torch.save(checkpoint, output_path)
    summary_path = output_path.with_name(output_path.stem + "_model_soup_summary.json")
    json_dump(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
