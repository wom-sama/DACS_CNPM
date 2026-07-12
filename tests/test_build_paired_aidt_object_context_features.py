from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trkh.tools.build_paired_aidt_object_context_features import (
    CONTEXT_MARGIN_RATIO,
    FEATURE_DIM,
    align_object_features_to_context,
    parse_args,
    resolve_encoder_metadata,
)


def test_protocol_defaults_lock_feature_width_and_context_margin() -> None:
    args = parse_args(
        [
            "--object-train-npz",
            "object_train.npz",
            "--object-val-npz",
            "object_val.npz",
            "--context-train-npz",
            "context_train.npz",
            "--context-val-npz",
            "context_val.npz",
            "--output-dir",
            "out",
        ]
    )
    assert args.feature_dim == FEATURE_DIM == 2816
    assert args.expected_context_margin_ratio == CONTEXT_MARGIN_RATIO == 0.5


def _cache(
    sample_index: list[int],
    *,
    features: np.ndarray,
    labels: list[int],
    sources: list[str],
    objects: list[int],
) -> dict:
    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray([f"image_{value}.jpg" for value in sample_index], dtype=object),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "source_stem": np.asarray(sources, dtype=object),
        "object_index": np.asarray(objects, dtype=np.int64),
        "classes": np.asarray(["c0", "c1"], dtype=object),
    }


def test_object_features_are_reordered_by_sample_index_and_keys() -> None:
    object_cache = _cache(
        [2, 7, 4],
        features=np.asarray([[2, 20], [7, 70], [4, 40]], dtype=np.float32),
        labels=[0, 1, 1],
        sources=["Source_2", "SOURCE_7", "source_4"],
        objects=[0, 1, 0],
    )
    context_cache = _cache(
        [4, 2],
        features=np.asarray([[40, 400], [20, 200]], dtype=np.float32),
        labels=[1, 0],
        sources=["SOURCE_4", "source_2"],
        objects=[0, 0],
    )
    aligned, summary = align_object_features_to_context(
        object_cache,
        context_cache,
        feature_dim=2,
    )
    assert np.array_equal(aligned, [[4, 40], [2, 20]])
    assert summary["rows"] == 2
    assert summary["object_extra_rows"] == 1
    assert summary["source_groups"] == 2


def test_alignment_fails_closed_on_object_identity_mismatch() -> None:
    object_cache = _cache(
        [3],
        features=np.ones((1, 2), dtype=np.float32),
        labels=[1],
        sources=["source_3"],
        objects=[0],
    )
    context_cache = _cache(
        [3],
        features=np.ones((1, 2), dtype=np.float32),
        labels=[1],
        sources=["source_3"],
        objects=[1],
    )
    with pytest.raises(ValueError, match="object indices differ"):
        align_object_features_to_context(object_cache, context_cache, feature_dim=2)


def test_encoder_metadata_follows_remap_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source_train.npz"
    np.savez_compressed(
        source,
        resnet_state_dict_sha256=np.asarray(["resnet-hash"], dtype=object),
        vit_state_dict_sha256=np.asarray(["vit-hash"], dtype=object),
        data_format=np.asarray(["classification_folder"], dtype=object),
    )
    remapped = tmp_path / "remapped_train.npz"
    np.savez_compressed(
        remapped,
        source_feature_npz=np.asarray([str(source)], dtype=object),
    )
    metadata = resolve_encoder_metadata(remapped)
    assert metadata["resnet_state_dict_sha256"] == "resnet-hash"
    assert metadata["vit_state_dict_sha256"] == "vit-hash"
    assert metadata["metadata_npz"] == str(source.resolve())
    assert metadata["remapped_npz"] == str(remapped.resolve())
