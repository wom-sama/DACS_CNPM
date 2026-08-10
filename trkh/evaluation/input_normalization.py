from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trkh.core.config import IMAGENET_MEAN, IMAGENET_STD


def _coerce_float_triplet(value: object, default: Sequence[float], *, positive: bool = False) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return tuple(float(item) for item in default)  # type: ignore[return-value]
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return tuple(float(item) for item in default)  # type: ignore[return-value]
    if len(values) != 3:
        return tuple(float(item) for item in default)  # type: ignore[return-value]
    if positive and any(item <= 0.0 for item in values):
        return tuple(float(item) for item in default)  # type: ignore[return-value]
    return values  # type: ignore[return-value]


def checkpoint_input_normalization(checkpoint: Mapping[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, Mapping):
        model_config = {}
    mean = _coerce_float_triplet(model_config.get("input_mean"), IMAGENET_MEAN)
    std = _coerce_float_triplet(model_config.get("input_std"), IMAGENET_STD, positive=True)
    return mean, std
