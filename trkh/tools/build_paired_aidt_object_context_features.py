from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from trkh.tools.audit_raw_aidt_fusion_readiness import load_feature_cache
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _sha256,
    _write_artifact_manifest,
)


FEATURE_DIM = 2816
EXPECTED_ROWS = {"train": 9215, "val": 2606}
CONTEXT_MARGIN_RATIO = 0.50


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly align raw-pretrained AIDT object and YOLO bbox-context "
            "features by sample_index, then concatenate them for a no-test audit."
        )
    )
    parser.add_argument("--object-train-npz", type=Path, required=True)
    parser.add_argument("--object-val-npz", type=Path, required=True)
    parser.add_argument("--context-train-npz", type=Path, required=True)
    parser.add_argument("--context-val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM)
    parser.add_argument(
        "--expected-context-margin-ratio",
        type=float,
        default=CONTEXT_MARGIN_RATIO,
    )
    return parser.parse_args(argv)


def _one_scalar(payload: Mapping[str, np.ndarray], name: str) -> object:
    values = np.asarray(payload[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"metadata field {name} must contain one value")
    return values[0]


def resolve_encoder_metadata(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as payload:
        fields = set(payload.files)
        if {"resnet_state_dict_sha256", "vit_state_dict_sha256"}.issubset(fields):
            metadata: Dict[str, object] = {
                "metadata_npz": str(path),
                "resnet_state_dict_sha256": str(
                    _one_scalar(payload, "resnet_state_dict_sha256")
                ),
                "vit_state_dict_sha256": str(
                    _one_scalar(payload, "vit_state_dict_sha256")
                ),
            }
            for name in (
                "resnet_name",
                "vit_name",
                "data_format",
                "yolo_crop_margin_ratio",
            ):
                if name in fields:
                    value = _one_scalar(payload, name)
                    metadata[name] = float(value) if name.endswith("ratio") else str(value)
            return metadata
        if "source_feature_npz" not in fields:
            raise ValueError(f"feature cache lacks encoder provenance: {path}")
        source = Path(str(_one_scalar(payload, "source_feature_npz"))).resolve()
    if source == path:
        raise ValueError(f"feature cache provenance points to itself: {path}")
    metadata = resolve_encoder_metadata(source)
    metadata["remapped_npz"] = str(path)
    return metadata


def _normalized_sources(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(value).casefold() for value in values], dtype=object)


def align_object_features_to_context(
    object_cache: Mapping[str, np.ndarray],
    context_cache: Mapping[str, np.ndarray],
    *,
    feature_dim: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    object_features = np.asarray(object_cache["features"], dtype=np.float32)
    context_features = np.asarray(context_cache["features"], dtype=np.float32)
    if object_features.shape[1] != int(feature_dim):
        raise ValueError(
            f"object feature dimension differs from {int(feature_dim)}: "
            f"{object_features.shape[1]}"
        )
    if context_features.shape[1] != int(feature_dim):
        raise ValueError(
            f"context feature dimension differs from {int(feature_dim)}: "
            f"{context_features.shape[1]}"
        )
    if object_cache["classes"].tolist() != context_cache["classes"].tolist():
        raise ValueError("object/context class order differs")

    object_indices = np.asarray(object_cache["sample_index"], dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(object_indices.tolist())}
    aligned_rows = []
    for sample_index in np.asarray(context_cache["sample_index"], dtype=np.int64):
        if int(sample_index) not in lookup:
            raise ValueError(f"object cache is missing sample_index={int(sample_index)}")
        aligned_rows.append(int(lookup[int(sample_index)]))
    if len(set(aligned_rows)) != len(aligned_rows):
        raise ValueError("object/context sample-index alignment is not one-to-one")

    aligned_labels = np.asarray(object_cache["labels"], dtype=np.int64)[aligned_rows]
    context_labels = np.asarray(context_cache["labels"], dtype=np.int64)
    if not np.array_equal(aligned_labels, context_labels):
        raise ValueError("object/context labels differ after sample-index alignment")
    aligned_sources = _normalized_sources(
        np.asarray(object_cache["source_stem"], dtype=object)[aligned_rows]
    )
    context_sources = _normalized_sources(
        np.asarray(context_cache["source_stem"], dtype=object)
    )
    if not np.array_equal(aligned_sources, context_sources):
        raise ValueError("object/context source stems differ after alignment")
    aligned_objects = np.asarray(object_cache["object_index"], dtype=np.int64)[
        aligned_rows
    ]
    context_objects = np.asarray(context_cache["object_index"], dtype=np.int64)
    if not np.array_equal(aligned_objects, context_objects):
        raise ValueError("object/context object indices differ after alignment")

    return object_features[aligned_rows], {
        "rows": int(len(aligned_rows)),
        "object_rows": int(len(object_indices)),
        "object_extra_rows": int(len(object_indices) - len(aligned_rows)),
        "sample_index_min": int(np.min(context_cache["sample_index"])),
        "sample_index_max": int(np.max(context_cache["sample_index"])),
        "source_groups": int(np.unique(context_sources).size),
        "class_counts": np.bincount(
            context_labels,
            minlength=len(context_cache["classes"]),
        ).tolist(),
    }


def _validate_encoder_metadata(
    metadata: Mapping[str, Mapping[str, object]],
    *,
    expected_context_margin_ratio: float,
) -> Dict[str, str]:
    resnet_hashes = {
        str(value["resnet_state_dict_sha256"]) for value in metadata.values()
    }
    vit_hashes = {str(value["vit_state_dict_sha256"]) for value in metadata.values()}
    if len(resnet_hashes) != 1 or len(vit_hashes) != 1:
        raise ValueError("object/context encoder state hashes differ")
    for split in ("context_train", "context_val"):
        values = metadata[split]
        if str(values.get("data_format", "")).casefold() != "yolo":
            raise ValueError(f"{split} is not a direct YOLO feature cache")
        margin = float(values.get("yolo_crop_margin_ratio", -1.0))
        if not np.isclose(margin, float(expected_context_margin_ratio), atol=1e-7):
            raise ValueError(
                f"{split} context margin {margin} != {expected_context_margin_ratio}"
            )
    return {
        "resnet_state_dict_sha256": next(iter(resnet_hashes)),
        "vit_state_dict_sha256": next(iter(vit_hashes)),
    }


def run_builder(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.feature_dim) <= 0:
        raise ValueError("feature-dim must be positive")
    if float(args.expected_context_margin_ratio) < 0.0:
        raise ValueError("expected-context-margin-ratio must be non-negative")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "object_train": Path(args.object_train_npz).resolve(),
        "object_val": Path(args.object_val_npz).resolve(),
        "context_train": Path(args.context_train_npz).resolve(),
        "context_val": Path(args.context_val_npz).resolve(),
    }
    metadata = {name: resolve_encoder_metadata(path) for name, path in paths.items()}
    encoder_hashes = _validate_encoder_metadata(
        metadata,
        expected_context_margin_ratio=float(args.expected_context_margin_ratio),
    )

    split_summaries: Dict[str, Dict[str, object]] = {}
    output_paths: Dict[str, str] = {}
    split_sources: Dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        object_cache = load_feature_cache(paths[f"object_{split}"])
        context_cache = load_feature_cache(paths[f"context_{split}"])
        aligned_object, alignment = align_object_features_to_context(
            object_cache,
            context_cache,
            feature_dim=int(args.feature_dim),
        )
        context_features = np.asarray(context_cache["features"], dtype=np.float32)
        paired = np.concatenate((aligned_object, context_features), axis=1)
        if not np.isfinite(paired).all():
            raise ValueError(f"{split} paired features contain non-finite values")
        output_path = output_dir / f"features_{split}_raw_aidt_object_context_m050.npz"
        np.savez_compressed(
            output_path,
            features=paired,
            labels=context_cache["labels"],
            paths=context_cache["paths"],
            sample_index=context_cache["sample_index"],
            source_stem=context_cache["source_stem"],
            object_index=context_cache["object_index"],
            classes=context_cache["classes"],
            component_a_dim=np.asarray([int(args.feature_dim)], dtype=np.int64),
            component_b_dim=np.asarray([int(args.feature_dim)], dtype=np.int64),
            component_a_name=np.asarray(["object"], dtype=object),
            component_b_name=np.asarray(["context"], dtype=object),
            candidate_name=np.asarray(["paired"], dtype=object),
            audit_mode=np.asarray(["object_context_fusion"], dtype=object),
            context_margin_ratio=np.asarray(
                [float(args.expected_context_margin_ratio)], dtype=np.float32
            ),
            object_source_npz=np.asarray([str(paths[f"object_{split}"])], dtype=object),
            context_source_npz=np.asarray(
                [str(paths[f"context_{split}"])], dtype=object
            ),
            resnet_state_dict_sha256=np.asarray(
                [encoder_hashes["resnet_state_dict_sha256"]], dtype=object
            ),
            vit_state_dict_sha256=np.asarray(
                [encoder_hashes["vit_state_dict_sha256"]], dtype=object
            ),
            diagnostic_only=np.asarray([True], dtype=np.bool_),
        )
        split_sources[split] = _normalized_sources(context_cache["source_stem"])
        split_summaries[split] = {
            **alignment,
            "expected_full_rows": int(EXPECTED_ROWS[split]),
            "full_support": bool(int(alignment["rows"]) == EXPECTED_ROWS[split]),
            "feature_shape": list(paired.shape),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
        }
        output_paths[split] = str(output_path)

    source_overlap = len(
        set(split_sources["train"].tolist()).intersection(split_sources["val"].tolist())
    )
    summary = {
        "method": "raw_pretrained_aidt_object_context_feature_pairing",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "encoder_metadata": metadata,
        "encoder_hashes": encoder_hashes,
        "component_order": ["object_2816d", "context_2816d"],
        "combined_feature_dim": int(args.feature_dim) * 2,
        "context_margin_ratio": float(args.expected_context_margin_ratio),
        "splits": split_summaries,
        "train_val_source_overlap": int(source_overlap),
        "split_usage": {"train": True, "val": True, "test": False},
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
        "test_split_used": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# Raw AIDT Object + Context Paired Feature Cache",
        "",
        f"- Rows train/val: `{split_summaries['train']['rows']}/{split_summaries['val']['rows']}`",
        f"- Feature order: `{summary['component_order']}`",
        f"- Context bbox margin ratio: `{summary['context_margin_ratio']}`",
        f"- Train/validation source overlap: `{source_overlap}`",
        "",
        "Rows are aligned strictly by sample_index and verified by label, source stem, object index, class order, and encoder state hashes. This cache is diagnostic-only and contains no test input, model, or checkpoint.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = _write_artifact_manifest(
        output_dir,
        mode="raw_pretrained_aidt_object_context_feature_pair_manifest",
    )
    summary["artifact_manifest"] = {
        key: value for key, value in manifest.items() if key != "files"
    }
    summary["outputs"] = output_paths
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_builder(parse_args(argv))
    print(
        json.dumps(
            {
                "splits": summary["splits"],
                "encoder_hashes": summary["encoder_hashes"],
                "train_val_source_overlap": summary["train_val_source_overlap"],
                "artifact_manifest": summary["artifact_manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
