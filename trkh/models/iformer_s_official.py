"""Pinned adapter for the official iFormer-S v0.9 implementation.

The architecture remains in the verified official checkout.  This module only
constructs it directly, loads the verified checkpoint safely, and exposes the
official pooled tensor that the upstream ``forward_features`` hides behind its
classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


OFFICIAL_REPOSITORY = "https://github.com/ChuanyangZheng/iFormer"
OFFICIAL_REVISION = "063d2afd851f77c84be0f15c67c9bb5d828c5f2b"
OFFICIAL_TAG = "v0.9"
OFFICIAL_LICENSE = "MIT"
OFFICIAL_LICENSE_SHA256 = (
    "63e8210e6bf3e8c032dc0c69b1d1d2e3ab72c14b02cabcc0dada2618bb188b97"
)
OFFICIAL_SOURCE_SHA256 = (
    "a83a7903a9b9e68c072f219974f3a99f5bd80d24965a3437179e8c9ea029a652"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "dba81d99d9b6491b18fccd022f795d2b43e31bcb75f7aeea59be37b00bc7d7a1"
)
OFFICIAL_CHECKPOINT_URL = (
    "https://github.com/ChuanyangZheng/iFormer/releases/download/v0.9/"
    "iFormer_s.pth"
)
OFFICIAL_CHECKPOINT_ASSET_ID = 210_177_750
OFFICIAL_CHECKPOINT_BYTES = 79_363_642
OFFICIAL_STATE_KEY = "model"
OFFICIAL_STATE_TENSORS = 505
OFFICIAL_CLASSES = 1000
TRANSFER_CLASSES = 5
DESCRIPTOR_DIM = 320
PARAMETERS_1000 = 6_563_368
PARAMETERS_5 = 6_243_973
_CONSTRUCTION_SEED = 20260804
_LINEAR_HEAD_KEYS = {
    "classifier.classifier.l.weight",
    "classifier.classifier.l.bias",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_source(root: Path) -> dict[str, object]:
    """Validate the pinned source file, and the license for a repository root."""

    resolved = Path(root).resolve(strict=True)
    source = resolved if resolved.is_file() else resolved / "models" / "iformer.py"
    source = source.resolve(strict=True)
    observed = _sha256(source)
    if observed != OFFICIAL_SOURCE_SHA256:
        raise ValueError(
            f"iFormer source SHA-256 mismatch: {observed} != "
            f"{OFFICIAL_SOURCE_SHA256}"
        )
    result: dict[str, object] = {
        "root": str(resolved if resolved.is_dir() else source.parent.parent),
        "source": str(source),
        "source_sha256": observed,
        "repository": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "tag": OFFICIAL_TAG,
        "license": OFFICIAL_LICENSE,
    }
    if resolved.is_dir():
        license_path = (resolved / "LICENSE").resolve(strict=True)
        license_sha = _sha256(license_path)
        if license_sha != OFFICIAL_LICENSE_SHA256:
            raise ValueError("iFormer LICENSE SHA-256 mismatch")
        result.update(license_path=str(license_path), license_sha256=license_sha)
    return result


@lru_cache(maxsize=2)
def _official_constructor(source_text: str) -> Any:
    source = Path(source_text).resolve(strict=True)
    validate_official_source(source)

    # Execute the pinned source directly.  This avoids timm.create_model, whose
    # current registry injects kwargs unsupported by the v0.9 constructor, and
    # avoids writing bytecode into the immutable official checkout.
    module = types.ModuleType("_trkh_pinned_iformer_v09")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    constructor = getattr(module, "iFormer_s", None)
    if not callable(constructor):
        raise TypeError("Pinned iFormer source has no callable iFormer_s constructor")
    return constructor


def _construct(source: Path, *, num_classes: int) -> nn.Module:
    constructor = _official_constructor(str(source.resolve(strict=True)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_CONSTRUCTION_SEED)
        model = constructor(
            pretrained=False,
            num_classes=int(num_classes),
            drop_path_rate=0.0,
            layer_scale_init_value=0.0,
            head_init_scale=1.0,
            distillation=False,
        )
    if not isinstance(model, nn.Module):
        raise TypeError("Official iFormer-S constructor did not return nn.Module")
    expected = PARAMETERS_1000 if num_classes == OFFICIAL_CLASSES else PARAMETERS_5
    observed = sum(parameter.numel() for parameter in model.parameters())
    if num_classes not in {OFFICIAL_CLASSES, TRANSFER_CLASSES} or observed != expected:
        raise ValueError(
            f"iFormer-S parameter contract changed: classes={num_classes}, "
            f"parameters={observed}, expected={expected}"
        )
    return model


def load_official_checkpoint(path: Path) -> dict[str, object]:
    """Safely load only the pinned released model state and its provenance."""

    checkpoint_path = path.resolve(strict=True)
    observed = _sha256(checkpoint_path)
    if observed != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(
            f"iFormer checkpoint SHA-256 mismatch: {observed} != "
            f"{OFFICIAL_CHECKPOINT_SHA256}"
        )

    allowlist = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        argparse.Namespace,
    ]
    with torch.serialization.safe_globals(allowlist):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("iFormer checkpoint root must be a mapping")
    state = checkpoint.get(OFFICIAL_STATE_KEY)
    if not isinstance(state, Mapping):
        raise TypeError("iFormer checkpoint has no mapping-valued 'model' state")
    if len(state) != OFFICIAL_STATE_TENSORS or any(
        not isinstance(key, str) or not isinstance(value, Tensor)
        for key, value in state.items()
    ):
        raise ValueError("iFormer checkpoint model state contract changed")
    # Deliberately ignore checkpoint['args']: its stale internal model name and
    # layer-scale value do not describe this strict-loadable released state.
    return {
        "path": str(checkpoint_path),
        "sha256": observed,
        "state_key": OFFICIAL_STATE_KEY,
        "state_tensors": len(state),
        "state": state,
    }


def _safe_checkpoint(path: Path) -> Mapping[str, Tensor]:
    return load_official_checkpoint(path)["state"]  # type: ignore[return-value]


class IFormerPooledDescriptor(nn.Module):
    """Official iFormer-S plus its exact pre-classifier GAP descriptor."""

    def __init__(self, model: nn.Module, *, num_classes: int | None = None) -> None:
        super().__init__()
        self.model = model
        linear = getattr(getattr(model.classifier, "classifier", None), "l", None)
        observed_classes = int(getattr(linear, "out_features", -1))
        self.num_classes = int(
            observed_classes if num_classes is None else num_classes
        )
        if self.num_classes != observed_classes:
            raise ValueError("iFormer-S wrapper class count does not match its model")
        self.descriptor_dim = DESCRIPTOR_DIM
        self.source_sha256 = OFFICIAL_SOURCE_SHA256
        self.checkpoint_sha256 = OFFICIAL_CHECKPOINT_SHA256

    def pooled_descriptor(self, images: Tensor) -> Tensor:
        value: Any = images
        for index in range(4):
            if isinstance(value, tuple):
                spatial, other = value
                value = (self.model.downsample_layers[index](spatial), other)
            else:
                value = self.model.downsample_layers[index](value)
            value = self.model.stages[index](value)
        if isinstance(value, tuple):
            value = value[0]
        descriptor = F.adaptive_avg_pool2d(value, 1).flatten(1)
        if descriptor.ndim != 2 or descriptor.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(
                f"iFormer-S descriptor geometry changed: {tuple(descriptor.shape)}"
            )
        return descriptor

    def forward_with_descriptor(self, images: Tensor) -> tuple[Tensor, Tensor]:
        descriptor = self.pooled_descriptor(images)
        logits = self.model.classifier(descriptor)
        return logits, descriptor

    def forward(self, images: Tensor) -> Tensor:
        return self.pooled_descriptor(images)


OfficialIFormerS = IFormerPooledDescriptor


def build_official_iformer_s(root: Path, checkpoint: Path) -> nn.Module:
    """Return the exact strict-loaded released 1000-class official module."""

    source_info = validate_official_source(root)
    model = _construct(Path(str(source_info["source"])), num_classes=OFFICIAL_CLASSES)
    artifact = load_official_checkpoint(checkpoint)
    model.load_state_dict(artifact["state"], strict=True)  # type: ignore[arg-type]
    model.trkh_official_source_path = str(source_info["source"])
    model.trkh_official_source_sha256 = OFFICIAL_SOURCE_SHA256
    model.trkh_official_checkpoint_sha256 = OFFICIAL_CHECKPOINT_SHA256
    return model


def copy_iformer_s_backbone_5class(released: nn.Module) -> nn.Module:
    """Copy every compatible released state and reset only the 1000-way linear."""

    source_text = getattr(released, "trkh_official_source_path", "")
    if not source_text:
        raise ValueError("Released iFormer-S module has no pinned source provenance")
    model = _construct(Path(source_text), num_classes=TRANSFER_CLASSES)
    transferable = {
        key: value
        for key, value in released.state_dict().items()
        if key not in _LINEAR_HEAD_KEYS
    }
    result = model.load_state_dict(transferable, strict=False)
    if set(result.missing_keys) != _LINEAR_HEAD_KEYS or result.unexpected_keys:
        raise RuntimeError(f"Unexpected iFormer-S 5-class transfer result: {result}")
    model.trkh_official_source_path = source_text
    model.trkh_official_source_sha256 = OFFICIAL_SOURCE_SHA256
    model.trkh_official_checkpoint_sha256 = OFFICIAL_CHECKPOINT_SHA256
    return model


def load_official_iformer_s_1000(
    root: Path,
    checkpoint: Path,
) -> IFormerPooledDescriptor:
    """Strict-load the released 1000-class state before any transfer surgery."""

    model = build_official_iformer_s(root, checkpoint)
    return IFormerPooledDescriptor(model, num_classes=OFFICIAL_CLASSES)


def build_official_iformer_s_5class(
    root: Path,
    checkpoint: Path,
) -> IFormerPooledDescriptor:
    """Copy the exact pretrained backbone/BN and replace only the linear head."""

    released = build_official_iformer_s(root, checkpoint)
    model = copy_iformer_s_backbone_5class(released)
    return IFormerPooledDescriptor(model, num_classes=TRANSFER_CLASSES)


__all__ = [
    "DESCRIPTOR_DIM",
    "OFFICIAL_CHECKPOINT_ASSET_ID",
    "OFFICIAL_CHECKPOINT_BYTES",
    "OFFICIAL_CHECKPOINT_SHA256",
    "OFFICIAL_CHECKPOINT_URL",
    "OFFICIAL_LICENSE",
    "OFFICIAL_LICENSE_SHA256",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_REVISION",
    "OFFICIAL_SOURCE_SHA256",
    "OFFICIAL_TAG",
    "PARAMETERS_1000",
    "PARAMETERS_5",
    "IFormerPooledDescriptor",
    "OfficialIFormerS",
    "build_official_iformer_s",
    "build_official_iformer_s_5class",
    "copy_iformer_s_backbone_5class",
    "load_official_checkpoint",
    "load_official_iformer_s_1000",
    "validate_official_source",
]
