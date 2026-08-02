from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Must be set before the first CUDA/cuBLAS operation in deterministic mode.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.models.dinov3_xcnorm_pair_adapter_a0 import (
    LOCAL_CONV_A0_MODE,
    LOCAL_XCNORM_A0_MODE,
    XCNORM_PAIR_ADDED_PARAMETER_COUNT,
    XCNORM_PAIR_RESIDUAL_RATIO_CAP,
    DinoV3XCNormPairAdapterA0,
)
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.build_classf_integrity_review import _HammingBKTree
from trkh.tools.export_timm_predictions import _state_dict_sha256
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)
PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_XCNORM_A0_TRAINFOLD_20260802"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
EXPECTED_CLASSES = ("Xoai_Song_Chua_KhoDap", "Xoai_Song_ChuaNhe_CoNguyCo", "Xoai_Chin_NgotThanh_DeDap", "Xoai_ChinGia_NgotGat_KhongVanChuyen", "Xoai_Hu_KhongAnDuoc")
EXPECTED_TRAIN_SAMPLES = 8_278
EXPECTED_DATA_SHA256 = "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8"
EXPECTED_MANIFEST_SHA256 = "59cf846b69f73c72bc415feaa3ce19fae13ca99f2c58ddb1119c0aaacc77a870"
EXPECTED_INTEGRITY_MANIFEST_SHA256 = "fb4dd603851fb5d1f84e9afa589534835bc9587ddfbf32034de3a1a7dce0118b"
EXPECTED_CHECKPOINT_SHA256 = "52de1b4dd45281d3531f36974f615fd71ba7c108747cca54b4052ef19e506d10"
EXPECTED_SELECTED_STATE_SHA256 = "efe735ccbd70df9d056da8f9576203b326063678e311160b5d700be26fa2f9ad"
EXPECTED_DATASET_TREE_SHA256 = "70a1b7d2b4c6f80e28fe3f0f714f1ba3e8ab654a90ce50b1e9cae8e4dac4a503"
EXPECTED_PRETRAINED_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
EXPECTED_B9_RECIPE_SHA256 = "a2936069361a8b2c23b96adf763bd0a948b6d55fd6583a22f10f6050a0917068"
EXPECTED_B9_PROTOCOL = "TRKH_PRETRAINED_CLASSF_B9_B2_REFERENCE_COMPLETION_20260801"

FOLDS = 5
SEED = 20260731
EPOCHS = 5
LR = 1.5e-4
# XCNorm is scale-normalized; weight decay would make the Conv/XCNorm pair
# asymmetric without regularizing the effective XCNorm function.
WEIGHT_DECAY = 0.0
TEMPERED_POWER = 0.5
PATCH_SHAPE = (256, 384)
RIVALS = (0, 2, 4)
PHASH_RADIUS = 3
SOURCE_ADJACENCY = 1
EXPECTED_ASSIGNMENT_INT64_SHA256 = "ff718e7d102f966fe27271692ce843daaca3dc6af342d217101983761da557ed"
EXPECTED_PATH_FOLD_SHA256 = "d13d8b70397aac942b8a1949cd7b9b8ca1a380a351efc6fde6db7c2b5928223b"
EXPECTED_HELD_CLASS_COUNTS = ((394, 96, 359, 501, 473), (401, 102, 234, 444, 513), (388, 95, 236, 296, 506), (415, 98, 213, 472, 457), (389, 106, 284, 367, 439))

CACHE_FILES = {"tokens": "train_postnorm_patch_tokens_f32.npy", "logits": "train_base_logits_f32.npy", "labels": "train_labels_i64.npy", "paths": "train_paths.json", "manifest": "cache_manifest.json"}
PHOTOMETRIC_CONDITIONS = {"dim": {"brightness": 0.70, "contrast": 0.90}, "bright": {"brightness": 1.25, "contrast": 1.10}, "low_contrast": {"brightness": 1.00, "contrast": 0.65}}
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only cached-token OOF screen for capacity-matched "
            "DINOv3 local Conv A0 versus XCNorm A0. No validation/test dataset "
            "is constructed or opened."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-integrity-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adapter-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_train_path(path: Path, data_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"train path is outside canonical data root: {resolved}") from exc
    if not relative.parts or relative.parts[0].casefold() != "train":
        raise ValueError(f"non-train path is forbidden: {resolved}")
    return relative.as_posix()


def assert_train_only_paths(paths: Sequence[Path | str], train_root: Path) -> None:
    root = train_root.expanduser().resolve()
    for raw in paths:
        resolved = Path(raw).expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"precheck path is not inside canonical train: {resolved}") from exc


def _data_root_from_yaml(data_yaml: Path) -> Path:
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if str(raw.get("format", raw.get("data_format", ""))).strip().lower() != "classification_folder":
        raise ValueError("XCNorm A0 precheck requires classification_folder data")
    configured = Path(str(raw.get("path", ".")))
    root = configured if configured.is_absolute() else data_yaml.parent / configured
    train_value = Path(str(raw.get("train", "train")))
    train_root = train_value if train_value.is_absolute() else root / train_value
    if train_root.expanduser().resolve() != (root / "train").expanduser().resolve():
        raise ValueError("canonical XCNorm precheck is locked to the train directory")
    return root.expanduser().resolve()


def _read_train_rows(
    manifest_path: Path,
    data_root: Path,
) -> List[Dict[str, object]]:
    """Read only canonical train identities/labels from the source manifest."""

    rows: List[Dict[str, object]] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "output_image", "class_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"manifest lacks fields: {sorted(required - set(reader.fieldnames or ()))}"
            )
        for raw in reader:
            if str(raw.get("split", "")).strip().casefold() != "train":
                continue
            relative = _relative_train_path(
                Path(str(raw["output_image"])),
                data_root,
            )
            key = relative.casefold()
            if key in seen:
                raise ValueError(f"duplicate canonical train path: {relative}")
            seen.add(key)
            rows.append(
                {
                    "relative_path": relative,
                    "label": int(raw["class_id"]),
                }
            )
    if len(rows) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f"canonical train support mismatch: {len(rows)}")
    return rows


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(int(size)))
        self.size = [1] * int(size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(int(left)), self.find(int(right))
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _read_integrity_train_rows(
    path: Path, canonical_rows: Sequence[Mapping[str, object]], data_root: Path
) -> List[Dict[str, object]]:
    expected = {str(row["relative_path"]).casefold(): int(row["label"]) for row in canonical_rows}
    values: List[Dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "relative_path", "class_id", "source_image_name", "source", "source_split", "source_prefix", "source_number", "leakage_group", "phash64", "sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"integrity manifest lacks fields: {sorted(required - set(reader.fieldnames or ()))}")
        for row in reader:
            if str(row.get("split", "")).strip().casefold() != "train":
                continue
            relative = str(row.get("relative_path", "")).replace("\\", "/")
            key = relative.casefold()
            if key not in expected:
                raise ValueError(f"integrity manifest has non-canonical train row: {relative}")
            if key in seen:
                raise ValueError(f"duplicate integrity train path: {relative}")
            seen.add(key)
            label = int(row["class_id"])
            if label != expected[key]:
                raise ValueError(f"integrity/canonical label mismatch: {relative}")
            image_path = data_root / relative
            if _sha256(image_path) != str(row.get("sha256", "")).casefold():
                raise ValueError(f"current train image differs from integrity attestation: {relative}")
            number_text = str(row.get("source_number", "")).strip()
            values.append({
                "relative_path": relative, "label": label,
                "source": str(row.get("source", "")).casefold(),
                "source_split": str(row.get("source_split", "")).casefold(),
                "source_root": Path(str(row.get("source_image_name", ""))).stem.casefold(),
                "source_prefix": str(row.get("source_prefix", "")).casefold(),
                "source_number": int(number_text) if number_text else None,
                "leakage_group": str(row.get("leakage_group", "")).casefold(),
                "phash_value": int(str(row.get("phash64", "")), 16),
            })
    if seen != set(expected):
        raise ValueError("integrity manifest does not exactly cover canonical train")
    return values


def build_train_union_groups(
    rows: Sequence[Mapping[str, object]],
    phash_values: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(rows) != len(phash_values) or not rows:
        raise ValueError("rows/phash_values must be non-empty and aligned")
    union = _UnionFind(len(rows))
    source_roots: Dict[Tuple[str, str, str], int] = {}
    leakage: Dict[str, int] = {}
    cohorts: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        provenance = (str(row["source"]), str(row["source_split"]))
        root_key = (*provenance, str(row["source_root"]))
        if root_key in source_roots:
            union.union(index, source_roots[root_key])
        else:
            source_roots[root_key] = index
        declared = str(row.get("leakage_group", ""))
        if declared:
            if declared in leakage:
                union.union(index, leakage[declared])
            else:
                leakage[declared] = index
        number = row.get("source_number")
        if number is not None:
            cohorts[(*provenance, str(row["source_prefix"]))].append((int(number), index))
    for members in cohorts.values():
        ordered = sorted(members)
        for (left_number, left), (right_number, right) in zip(ordered, ordered[1:]):
            if right_number - left_number <= SOURCE_ADJACENCY:
                union.union(left, right)
    tree = _HammingBKTree()
    for index, value in enumerate(phash_values):
        for previous, _distance in tree.query(int(value), PHASH_RADIUS):
            union.union(index, int(previous))
        tree.add(int(value), index)

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    groups = np.asarray([union.find(index) for index in range(len(rows))], dtype=np.int64)
    sizes = sorted((len(value) for value in components.values()), reverse=True)
    mixed = sum(
        len({int(rows[index]["label"]) for index in members}) > 1
        for members in components.values()
    )
    stats = {
        "components": len(components),
        "largest_component_sizes": sizes[:10],
        "components_over_20": int(sum(size > 20 for size in sizes)),
        "mixed_label_components": int(mixed),
        "size_quantiles": {
            str(q): float(np.quantile(np.asarray(sizes, dtype=np.float64), q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
    }
    return groups, stats


def assign_locked_folds(
    rows: Sequence[Mapping[str, object]], groups: np.ndarray, *, canonical_lock: bool = False
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[str, str]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if labels.size != groups.size:
        raise ValueError("rows/groups mismatch")
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    assignments = np.full(labels.size, -1, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    for fold, (fit, held) in enumerate(splitter.split(np.zeros(labels.size), labels, groups)):
        overlap = set(groups[fit].tolist()) & set(groups[held].tolist())
        counts = np.bincount(labels[held], minlength=5)
        if overlap or bool((counts <= 0).any()):
            raise RuntimeError(f"fold {fold} is unsafe: overlap={len(overlap)}, counts={counts.tolist()}")
        assignments[held] = fold
        fold_rows.append(
            {
                "fold": fold,
                "fit_samples": int(fit.size),
                "held_samples": int(held.size),
                "held_class_counts": counts.astype(int).tolist(),
                "group_overlap": 0,
            }
        )
    if bool((assignments < 0).any()):
        raise RuntimeError("incomplete fold assignment")
    path_fold_payload = "\n".join(
        f"{str(row['relative_path'])}\t{int(assignments[index])}"
        for index, row in enumerate(rows)
    ) + "\n"
    hashes = {
        "assignment_int64_sha256": hashlib.sha256(assignments.astype("<i8").tobytes()).hexdigest(),
        "path_fold_sha256": hashlib.sha256(path_fold_payload.encode("utf-8")).hexdigest(),
    }
    if canonical_lock:
        observed_counts = tuple(tuple(int(value) for value in row["held_class_counts"]) for row in fold_rows)
        if (
            hashes["assignment_int64_sha256"] != EXPECTED_ASSIGNMENT_INT64_SHA256
            or hashes["path_fold_sha256"] != EXPECTED_PATH_FOLD_SHA256
            or observed_counts != EXPECTED_HELD_CLASS_COUNTS
        ):
            raise RuntimeError("canonical SGKF assignment changed; refuse silent fold drift")
    return assignments, fold_rows, hashes


def _validate_b9_checkpoint(checkpoint: Mapping[str, object], checkpoint_sha256: str) -> Dict[str, object]:
    model_config = checkpoint.get("model_config", {})
    train_config = checkpoint.get("train_config", {})
    provenance = checkpoint.get("pretrained_provenance", {})
    if not all(isinstance(value, Mapping) for value in (model_config, train_config, provenance)):
        raise ValueError("B9 checkpoint metadata mappings are missing")
    initialization = provenance.get("initialization", {})
    if not isinstance(initialization, Mapping):
        raise ValueError("B9 pretrained initialization provenance is missing")
    pretrained_checkpoint = initialization.get("checkpoint", {})
    if not isinstance(pretrained_checkpoint, Mapping):
        raise ValueError("B9 pretrained checkpoint provenance is missing")
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_type": "timm_classifier",
        "model_name": MODEL_NAME,
        "epoch": 9,
        "best_epoch": 9,
        "checkpoint_weight_source": "ema",
        "validation_weight_source": "ema",
        "protocol_id": EXPECTED_B9_PROTOCOL,
        "dataset_tree_sha256": EXPECTED_DATASET_TREE_SHA256,
        "recipe_sha256": EXPECTED_B9_RECIPE_SHA256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "pretrained_offline": True,
    }
    observed = {
        "checkpoint_sha256": checkpoint_sha256,
        "model_type": str(model_config.get("model_type", "")),
        "model_name": str(model_config.get("timm_model_name", "")),
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "checkpoint_weight_source": str(checkpoint.get("checkpoint_weight_source", "")),
        "validation_weight_source": str(checkpoint.get("validation_weight_source", "")),
        "protocol_id": str(train_config.get("experiment_protocol_id", "")),
        "dataset_tree_sha256": str(train_config.get("dataset_image_tree_sha256", "")),
        "recipe_sha256": str(train_config.get("recipe_train_contract_sha256", "")),
        "pretrained_sha256": str(pretrained_checkpoint.get("sha256", "")),
        "pretrained_offline": bool(initialization.get("offline_verified_load", False)),
    }
    mismatches = {key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value}
    state = checkpoint.get("model_state")
    ema_state = checkpoint.get("ema_model_state")
    if not isinstance(state, Mapping) or not isinstance(ema_state, Mapping):
        raise ValueError("B9 selected EMA state is absent")
    selected_hash = _state_dict_sha256(state)
    ema_hash = _state_dict_sha256(ema_state)
    if selected_hash != EXPECTED_SELECTED_STATE_SHA256 or ema_hash != selected_hash:
        mismatches["selected_ema_state_sha256"] = {
            "expected": EXPECTED_SELECTED_STATE_SHA256,
            "observed": {"model_state": selected_hash, "ema_model_state": ema_hash},
        }
    if mismatches:
        raise ValueError(f"locked B9 checkpoint contract mismatch: {mismatches}")
    return {"expected": expected, "observed": observed, "selected_state_sha256": selected_hash}


def build_paired_adapters(classifier_weight: Tensor, *, seed: int = SEED) -> Tuple[nn.Module, nn.Module, Dict[str, object]]:
    torch.manual_seed(int(seed))
    control = DinoV3XCNormPairAdapterA0(mode=LOCAL_CONV_A0_MODE)
    control.bind_classifier_weight(classifier_weight)
    candidate = DinoV3XCNormPairAdapterA0(mode=LOCAL_XCNORM_A0_MODE)
    candidate.load_state_dict(control.state_dict(), strict=True)
    for module in (control, candidate):
        if module.added_parameter_count(trainable_only=True) != XCNORM_PAIR_ADDED_PARAMETER_COUNT:
            raise RuntimeError("adapter parameter/freeze contract changed")
        if not module.classifier_binding_matches(classifier_weight):
            raise RuntimeError("adapter classifier binding mismatch")
    control_hash = _state_dict_sha256(control.state_dict())
    candidate_hash = _state_dict_sha256(candidate.state_dict())
    if control_hash != candidate_hash:
        raise RuntimeError("paired adapters did not receive identical initialization")
    return control, candidate, {
        "paired_initial_state_sha256": control_hash,
        "control_parameters": control.added_parameter_count(),
        "candidate_parameters": candidate.added_parameter_count(),
        "classifier_weight_requires_grad": bool(classifier_weight.requires_grad),
    }


def _cache_paths(output_dir: Path) -> Dict[str, Path]:
    return {key: output_dir / name for key, name in CACHE_FILES.items()}


def _path_hash(paths: Sequence[str]) -> str:
    return _json_sha256([str(value).replace("\\", "/").casefold() for value in paths])


def _array_finite(array: np.ndarray, chunk: int = 64) -> bool:
    return all(np.isfinite(array[start : start + chunk]).all() for start in range(0, int(array.shape[0]), chunk))


def _load_or_extract_cache(
    model: nn.Module,
    dataset: Dataset,
    relative_paths: Sequence[str],
    canonical_labels: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    paths = _cache_paths(output_dir)
    present = {key: value.is_file() for key, value in paths.items()}
    expected_common = {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "paths_sha256": _path_hash(relative_paths),
        "token_shape": [EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE],
        "token_dtype": "float32",
        "logit_shape": [EXPECTED_TRAIN_SAMPLES, 5],
        "logit_dtype": "float32",
        "label_shape": [EXPECTED_TRAIN_SAMPLES],
        "label_dtype": "int64",
        "label_content_sha256": hashlib.sha256(np.asarray(canonical_labels, dtype="<i8").tobytes()).hexdigest(),
        "class_counts": np.bincount(np.asarray(canonical_labels, dtype=np.int64), minlength=5).astype(int).tolist(),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "head_state_sha256": provenance["head_state_sha256"],
        "backbone_state_sha256": provenance["backbone_state_sha256"],
    }
    if any(present.values()):
        if not all(present.values()):
            raise RuntimeError(f"partial cache is forbidden; use a new output: {present}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        mismatch = {key: [value, manifest.get(key)] for key, value in expected_common.items() if manifest.get(key) != value}
        for key in ("tokens", "logits", "labels", "paths"):
            if _sha256(paths[key]) != str(manifest.get(f"{key}_sha256", "")):
                mismatch[f"{key}_sha256"] = "cache content hash mismatch"
        if mismatch:
            raise ValueError(f"cache manifest mismatch: {mismatch}")
        stored_paths = json.loads(paths["paths"].read_text(encoding="utf-8"))
        if list(stored_paths) != list(relative_paths):
            raise ValueError("cached paths differ from canonical train order")
        tokens = np.load(paths["tokens"], mmap_mode="r", allow_pickle=False)
        logits = np.load(paths["logits"], mmap_mode="r", allow_pickle=False)
        labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
        if tokens.shape != (EXPECTED_TRAIN_SAMPLES, *PATCH_SHAPE) or tokens.dtype != np.float32:
            raise ValueError("cached post-norm token shape/dtype mismatch")
        if logits.shape != (EXPECTED_TRAIN_SAMPLES, 5) or logits.dtype != np.float32:
            raise ValueError("cached base-logit shape/dtype mismatch")
        if labels.shape != (EXPECTED_TRAIN_SAMPLES,) or labels.dtype != np.int64:
            raise ValueError("cached label shape/dtype mismatch")
        if not _array_finite(tokens) or not _array_finite(logits):
            raise ValueError("cached tensors contain non-finite values")
        if not np.array_equal(np.asarray(labels, dtype=np.int64), np.asarray(canonical_labels, dtype=np.int64)):
            raise ValueError("cached labels differ from live canonical train labels")
        if float(manifest.get("maximum_manual_head_parity_error", float("inf"))) > 1e-6:
            raise ValueError("cached manual average-patch head parity failed")
        return {"tokens": tokens, "logits": logits, "labels": labels, "manifest": manifest}

    partial = {key: value.with_suffix(value.suffix + ".partial") for key, value in paths.items() if key in {"tokens", "logits", "labels"}}
    if any(value.exists() for value in partial.values()):
        raise RuntimeError("partial token cache exists; use a new output directory")
    tokens = np.lib.format.open_memmap(partial["tokens"], mode="w+", dtype=np.float32, shape=(len(dataset), *PATCH_SHAPE))
    logits = np.lib.format.open_memmap(partial["logits"], mode="w+", dtype=np.float32, shape=(len(dataset), 5))
    labels = np.lib.format.open_memmap(partial["labels"], mode="w+", dtype=np.int64, shape=(len(dataset),))
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=max(0, workers), collate_fn=_collate_classification)
    cursor = 0
    max_head_parity_error = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets, _metadata in tqdm(loader, desc="B9 train post-norm token cache", dynamic_ncols=True):
            images = images.to(device=device, dtype=torch.float32)
            features = model.forward_features(images)
            if not torch.is_tensor(features) or features.ndim != 3:
                raise TypeError("B9 forward_features must return post-norm tokens [B,N,D]")
            prefix = int(getattr(model, "num_prefix_tokens", 0) or 0)
            patches = features.float()[:, prefix:]
            if tuple(patches.shape[1:]) != PATCH_SHAPE:
                raise ValueError(f"post-norm patch shape mismatch: {tuple(patches.shape)}")
            base_logits = model.forward_head(features, pre_logits=False).float()
            if tuple(base_logits.shape[1:]) != (5,):
                raise ValueError("B9 base logits must have width five")
            manual_logits = model.head(patches.mean(dim=1)).float()
            parity_error = float((manual_logits - base_logits).abs().max().cpu())
            max_head_parity_error = max(max_head_parity_error, parity_error)
            if parity_error > 1e-6:
                raise RuntimeError(f"manual average-patch head parity failed: {parity_error}")
            patch_np = patches.cpu().numpy().astype(np.float32, copy=False)
            logit_np = base_logits.cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(patch_np).all() or not np.isfinite(logit_np).all():
                raise FloatingPointError("non-finite B9 cache values")
            end = cursor + int(targets.numel())
            tokens[cursor:end], logits[cursor:end] = patch_np, logit_np
            labels[cursor:end] = targets.numpy().astype(np.int64)
            cursor = end
    if cursor != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(f"cache row mismatch: {cursor}")
    for array in (tokens, logits, labels):
        array.flush()
    del tokens, logits, labels
    for key in ("tokens", "logits", "labels"):
        partial[key].replace(paths[key])
    paths["paths"].write_text(json.dumps(list(relative_paths), indent=2), encoding="utf-8")
    manifest = {
        **expected_common,
        "schema_version": 1,
        "lossy_token_quantization": False,
        "maximum_manual_head_parity_error": max_head_parity_error,
        "head_state_sha256": provenance["head_state_sha256"],
        "backbone_state_sha256": provenance["backbone_state_sha256"],
    }
    for key in ("tokens", "logits", "labels", "paths"):
        manifest[f"{key}_sha256"] = _sha256(paths[key])
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "tokens": np.load(paths["tokens"], mmap_mode="r", allow_pickle=False),
        "logits": np.load(paths["logits"], mmap_mode="r", allow_pickle=False),
        "labels": np.load(paths["labels"], mmap_mode="r", allow_pickle=False),
        "manifest": manifest,
    }


def _tempered_indices(fit_indices: np.ndarray, labels: np.ndarray, seed: int, batch_size: int) -> np.ndarray:
    fit_indices = fit_indices[labels[fit_indices] != 3]
    fit_labels = labels[fit_indices]
    counts = np.bincount(fit_labels, minlength=5).astype(np.float64)
    weights = np.power(np.maximum(counts[fit_labels], 1.0), -TEMPERED_POWER)
    rng = np.random.default_rng(int(seed))
    batches = (fit_indices.size + batch_size - 1) // batch_size
    schedule = rng.choice(fit_indices, size=batches * batch_size, replace=True, p=weights / weights.sum())
    for start in range(0, schedule.size, batch_size):
        for offset, class_index in enumerate((0, 1, 2, 4)):
            support = fit_indices[fit_labels == class_index]
            schedule[start + offset] = rng.choice(support)
    return schedule


def _metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, object]:
    predicted = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(labels, dtype=np.int64)
    f1 = f1_score(labels, predicted, labels=list(range(5)), average=None, zero_division=0)
    fp_by_class = {str(rival): int(np.sum((labels == rival) & (predicted == 1))) for rival in RIVALS}
    return {
        "macro_f1": float(np.mean(f1)),
        "class1_f1": float(f1[1]),
        "class1_tp": int(np.sum((labels == 1) & (predicted == 1))),
        "restricted_fp": int(sum(fp_by_class.values())),
        "fp_to_class1": fp_by_class,
    }


def _pair_aurocs(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for rival in RIVALS:
        mask = np.logical_or(labels == 1, labels == rival)
        target = (labels[mask] == 1).astype(np.int64)
        result[str(rival)] = float(roc_auc_score(target, logits[mask, 1] - logits[mask, rival]))
    return result


def _pairwise_bce(logits: Tensor, targets: Tensor) -> Tensor:
    losses: List[Tensor] = []
    for rival in RIVALS:
        mask = torch.logical_or(targets == 1, targets == rival)
        count = int(mask.sum().item())
        if count:
            margin = logits[mask, 1] - logits[mask, rival]
            binary = (targets[mask] == 1).to(dtype=margin.dtype)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(margin, binary, reduction="mean"))
    if len(losses) != len(RIVALS):
        raise RuntimeError("paired batch has no 1-vs-{0,2,4} support")
    return torch.stack(losses).mean()


def _adapter_logits(
    adapter: nn.Module, tokens: Tensor, base_logits: Tensor, classifier_weight: Tensor
) -> Tuple[Tensor, Tensor]:
    logits, trace = adapter.logits_from_cached_tokens(
        tokens, classifier_weight, base_logits=base_logits, return_trace=True
    )
    return logits, trace["residual_norm_ratio"]


def run_oof_screen(
    *, tokens: np.ndarray, base_logits: np.ndarray, labels: np.ndarray,
    folds: np.ndarray, classifier_weight: Tensor, device: torch.device,
    batch_size: int, state_dir: Path, relative_paths: Sequence[str],
    assignment_hashes: Mapping[str, str], group_fold_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    control_oof = np.empty_like(base_logits, dtype=np.float32)
    candidate_oof = np.empty_like(base_logits, dtype=np.float32)
    residual_ratios: List[np.ndarray] = []
    fold_rows: List[Dict[str, object]] = []
    training_rows: List[Dict[str, object]] = []
    branch_off_error = 0.0
    paired_contracts: List[Dict[str, object]] = []
    nonfinite_updates = 0
    skipped_updates = 0
    completed_updates = 0
    state_dir.mkdir(parents=True, exist_ok=False)
    expected_updates = 0
    frozen_weight = classifier_weight.detach().to(device=device, dtype=torch.float32)
    if frozen_weight.requires_grad:
        raise RuntimeError("classifier weight must remain frozen")

    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        eligible_count = int(np.sum(labels[fit] != 3))
        expected_updates += EPOCHS * ((eligible_count + batch_size - 1) // batch_size)
        control, candidate, paired = build_paired_adapters(classifier_weight.detach().cpu(), seed=SEED + fold)
        paired_contracts.append({"fold": fold, **paired})
        control, candidate = control.to(device), candidate.to(device)
        control.bind_classifier_weight(frozen_weight)
        candidate.bind_classifier_weight(frozen_weight)
        optimizers = (
            torch.optim.AdamW(control.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
            torch.optim.AdamW(candidate.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        )
        for epoch in range(EPOCHS):
            schedule = _tempered_indices(fit, labels, SEED + fold * 100 + epoch, batch_size)
            losses = [0.0, 0.0]
            seen = 0
            control.train(); candidate.train()
            for start in range(0, schedule.size, batch_size):
                selected = schedule[start : start + batch_size]
                token_batch = torch.from_numpy(np.asarray(tokens[selected], dtype=np.float32)).to(device)
                base_batch = torch.from_numpy(np.asarray(base_logits[selected], dtype=np.float32)).to(device)
                target_batch = torch.from_numpy(labels[selected]).to(device)
                outputs = []
                for module in (control, candidate):
                    output = module.logits_from_cached_tokens(token_batch, frozen_weight, base_logits=base_batch)
                    outputs.append(output)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                batch_losses = [_pairwise_bce(output, target_batch) for output in outputs]
                if not all(bool(torch.isfinite(loss).item()) for loss in batch_losses):
                    nonfinite_updates += 1; skipped_updates += 1
                    continue
                batch_losses[0].backward(); batch_losses[1].backward()
                gradients_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                    for module in (control, candidate) for parameter in module.parameters()
                )
                if not gradients_finite:
                    nonfinite_updates += 1; skipped_updates += 1
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    continue
                for optimizer in optimizers:
                    optimizer.step()
                if not all(bool(torch.isfinite(parameter).all().item()) for module in (control, candidate) for parameter in module.parameters()):
                    raise FloatingPointError("adapter parameter became non-finite after AdamW step")
                completed_updates += 1
                count = int(selected.size)
                for index, loss in enumerate(batch_losses):
                    losses[index] += float(loss.detach().cpu()) * count
                seen += count
            training_rows.append({
                "fold": fold, "epoch": epoch + 1, "samples": seen,
                "control_loss": losses[0] / max(1, seen), "candidate_loss": losses[1] / max(1, seen),
            })

        control.eval(); candidate.eval()
        control_state = {key: value.detach().cpu() for key, value in control.state_dict().items()}
        candidate_state = {key: value.detach().cpu() for key, value in candidate.state_dict().items()}
        state_path = state_dir / f"fold_{fold}_final_adapters.pt"
        torch.save(
            {
                "protocol_id": PROTOCOL_ID, "fold": fold, "epochs": EPOCHS,
                "held_fold_selection": False, "adapter_ema": False,
                "control_state": control_state, "candidate_state": candidate_state,
            },
            state_path,
        )
        paired_contracts[-1].update({
            "control_final_state_sha256": _state_dict_sha256(control_state),
            "candidate_final_state_sha256": _state_dict_sha256(candidate_state),
            "state_artifact": str(state_path), "state_artifact_sha256": _sha256(state_path),
        })
        control_chunks: List[np.ndarray] = []
        candidate_chunks: List[np.ndarray] = []
        ratio_chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, held.size, batch_size):
                selected = held[start : start + batch_size]
                token_batch = torch.from_numpy(np.asarray(tokens[selected], dtype=np.float32)).to(device)
                base_batch = torch.from_numpy(np.asarray(base_logits[selected], dtype=np.float32)).to(device)
                control_result, _ = _adapter_logits(control, token_batch, base_batch, frozen_weight)
                candidate_result, ratios = _adapter_logits(candidate, token_batch, base_batch, frozen_weight)
                off = candidate.logits_from_cached_tokens(
                    token_batch, frozen_weight, base_logits=base_batch, branch_off=True
                )
                branch_off_error = max(branch_off_error, float((off - base_batch).abs().max().cpu()))
                control_chunks.append(control_result.cpu().numpy())
                candidate_chunks.append(candidate_result.cpu().numpy())
                ratio_chunks.append(ratios.cpu().numpy().reshape(-1))
        control_fold = np.concatenate(control_chunks)
        candidate_fold = np.concatenate(candidate_chunks)
        control_oof[held], candidate_oof[held] = control_fold, candidate_fold
        residual_ratios.extend(ratio_chunks)
        control_auc = _pair_aurocs(labels[held], control_fold)
        candidate_auc = _pair_aurocs(labels[held], candidate_fold)
        fold_rows.append({
            "fold": fold,
            "control_pair_auroc": control_auc,
            "candidate_pair_auroc": candidate_auc,
            "mean_pair_auroc_gain": float(np.mean([candidate_auc[str(r)] - control_auc[str(r)] for r in RIVALS])),
        })
    base_metrics = _metrics(labels, base_logits)
    control_metrics = _metrics(labels, control_oof)
    candidate_metrics = _metrics(labels, candidate_oof)
    control_auc = _pair_aurocs(labels, control_oof)
    candidate_auc = _pair_aurocs(labels, candidate_oof)
    pair_gains = {str(r): float(candidate_auc[str(r)] - control_auc[str(r)]) for r in RIVALS}
    oof_path = state_dir.parent / "train_oof_logits.npz"
    np.savez_compressed(
        oof_path,
        relative_paths=np.asarray(list(relative_paths), dtype=np.str_),
        labels=labels.astype(np.int64), folds=np.asarray(folds, dtype=np.int64),
        base_logits=np.asarray(base_logits, dtype=np.float32),
        control_logits=control_oof, candidate_logits=candidate_oof,
    )
    return {
        "base": base_metrics,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "control_pair_auroc": control_auc,
        "candidate_pair_auroc": candidate_auc,
        "pair_auroc_gains": pair_gains,
        "mean_pair_auroc_gain": float(np.mean(list(pair_gains.values()))),
        "active_off": {
            "macro_f1_delta_vs_b9": float(candidate_metrics["macro_f1"] - base_metrics["macro_f1"]),
            "class1_f1_delta_vs_b9": float(candidate_metrics["class1_f1"] - base_metrics["class1_f1"]),
            "class1_tp_delta_vs_b9": int(candidate_metrics["class1_tp"] - base_metrics["class1_tp"]),
            "restricted_fp_delta_vs_b9": int(candidate_metrics["restricted_fp"] - base_metrics["restricted_fp"]),
        },
        "fold_rows": fold_rows,
        "fold_wins": int(sum(float(row["mean_pair_auroc_gain"]) > 0.0 for row in fold_rows)),
        "residual_ratio_p95": float(np.quantile(np.concatenate(residual_ratios), 0.95)),
        "branch_off_max_abs_error": branch_off_error,
        "training_rows": training_rows,
        "paired_contracts": paired_contracts,
        "completed_updates": completed_updates,
        "expected_updates": expected_updates,
        "nonfinite_updates": nonfinite_updates,
        "skipped_updates": skipped_updates,
        "assignment_hashes": dict(assignment_hashes),
        "group_fold_contract": [dict(row) for row in group_fold_rows],
        "oof_artifact": str(oof_path),
        "oof_artifact_sha256": _sha256(oof_path),
    }


def assess_readiness(screen: Mapping[str, object]) -> Dict[str, object]:
    base = screen["base"]
    candidate = screen["candidate"]
    assert isinstance(base, Mapping) and isinstance(candidate, Mapping)
    base_fp = base["fp_to_class1"]
    candidate_fp = candidate["fp_to_class1"]
    assert isinstance(base_fp, Mapping) and isinstance(candidate_fp, Mapping)
    tp_retention = float(candidate["class1_tp"]) / max(1.0, float(base["class1_tp"]))
    fp_reduction = (float(base["restricted_fp"]) - float(candidate["restricted_fp"])) / max(1.0, float(base["restricted_fp"]))
    pair_gains = {str(key): float(value) for key, value in dict(screen.get("pair_auroc_gains", {})).items()}
    fold_rows = list(screen.get("group_fold_contract", []))
    assignment_hashes = screen.get("assignment_hashes", {})
    if not isinstance(assignment_hashes, Mapping):
        assignment_hashes = {}
    checks = {
        "five_complete_group_disjoint_folds": len(fold_rows) == FOLDS and all(int(row.get("group_overlap", 1)) == 0 for row in fold_rows),
        "canonical_fold_assignment_locked": assignment_hashes.get("assignment_int64_sha256") == EXPECTED_ASSIGNMENT_INT64_SHA256 and assignment_hashes.get("path_fold_sha256") == EXPECTED_PATH_FOLD_SHA256,
        "capacity_exact_3672": len(screen.get("paired_contracts", [])) == FOLDS and all(
            int(row.get("control_parameters", -1)) == XCNORM_PAIR_ADDED_PARAMETER_COUNT
            and int(row.get("candidate_parameters", -1)) == XCNORM_PAIR_ADDED_PARAMETER_COUNT
            for row in screen.get("paired_contracts", [])
        ),
        "candidate_vs_conv_pair_auroc_gain": float(screen["mean_pair_auroc_gain"]) >= 0.020,
        "two_of_three_pair_gains": sum(value >= 0.010 for value in pair_gains.values()) >= 2,
        "no_pair_auroc_collapse": len(pair_gains) == len(RIVALS) and min(pair_gains.values(), default=float("-inf")) >= -0.005,
        "candidate_fold_wins": int(screen["fold_wins"]) >= 4,
        "b9_macro_preserved": float(candidate["macro_f1"]) >= float(base["macro_f1"]) - 0.002,
        "b9_class1_tp_retained": tp_retention >= 0.98,
        "b9_restricted_fp_reduced": fp_reduction >= 0.10,
        "b9_0_to_1_not_increased": int(candidate_fp["0"]) <= int(base_fp["0"]),
        "b9_2_to_1_not_increased": int(candidate_fp["2"]) <= int(base_fp["2"]),
        "residual_active": float(screen["residual_ratio_p95"]) >= 1e-3,
        "residual_cap": float(screen["residual_ratio_p95"]) <= XCNORM_PAIR_RESIDUAL_RATIO_CAP + 1e-6,
        "active_off_class1_f1_gain": float(candidate["class1_f1"]) - float(base["class1_f1"]) >= 0.005,
        "branch_off_exact": float(screen["branch_off_max_abs_error"]) == 0.0,
        "finite_updates": int(screen.get("nonfinite_updates", -1)) == 0,
        "zero_skipped_updates": int(screen.get("skipped_updates", -1)) == 0,
        "all_fixed_updates_completed": int(screen.get("completed_updates", -1)) == int(screen.get("expected_updates", -2)),
        "photometric_train_recaches_complete": False,
    }
    clean_names = [name for name in checks if name != "photometric_train_recaches_complete"]
    clean_pass = all(bool(checks[name]) for name in clean_names)
    return {
        "conditional_adapter_screen": True,
        "end_to_end_oof_or_generalization_claim": False,
        "clean_train_gates_passed": clean_pass,
        "photometric_recache_permission": clean_pass,
        "full_validation_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not bool(value)],
        "observed": {"class1_tp_retention_vs_b9": tp_retention, "restricted_fp_reduction_vs_b9": fp_reduction},
    }


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    if args.batch_size <= 0 or args.adapter_batch_size < 4 or args.workers < 0:
        raise ValueError("cache batch must be positive, adapter batch >=4, workers non-negative")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    data_yaml = args.data.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("data YAML is not the locked canonical class_f contract")
    data_root = _data_root_from_yaml(data_yaml)
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must be outside immutable class_f")
    manifest_path = (args.manifest or data_root / "manifest.csv").expanduser().resolve()
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest is not the locked canonical class_f manifest")
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    checkpoint_contract = _validate_b9_checkpoint(checkpoint, checkpoint_sha)
    dataset, classes = _build_dataset(
        data_yaml=data_yaml, split="train", checkpoint=checkpoint,
        class_name_mode="raw", max_samples=0,
    )
    if tuple(classes) != EXPECTED_CLASSES or len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError("canonical class order/train support changed")
    dataset_paths = [Path(path) for path in dataset.sample_paths()]
    assert_train_only_paths(dataset_paths, data_root / "train")
    relative_paths = [_relative_train_path(path, data_root) for path in dataset_paths]
    dataset_labels = np.asarray(dataset.labels(), dtype=np.int64)
    canonical_rows = [
        {"relative_path": path, "label": int(dataset_labels[index])}
        for index, path in enumerate(relative_paths)
    ]

    integrity_path = args.train_integrity_manifest.expanduser().resolve()
    if _sha256(integrity_path) != EXPECTED_INTEGRITY_MANIFEST_SHA256:
        raise ValueError("train integrity manifest is not the locked radius-3 artifact")
    integrity_rows = _read_integrity_train_rows(integrity_path, canonical_rows, data_root)
    phash_vector = [int(row["phash_value"]) for row in integrity_rows]
    phash_vector_sha256 = hashlib.sha256(np.asarray(phash_vector, dtype="<u8").tobytes()).hexdigest()
    groups_sorted, group_stats = build_train_union_groups(integrity_rows, phash_vector)
    assignments_sorted, fold_rows, assignment_hashes = assign_locked_folds(integrity_rows, groups_sorted, canonical_lock=True)
    fold_by_path = {str(row["relative_path"]).casefold(): int(assignments_sorted[index]) for index, row in enumerate(integrity_rows)}
    group_by_path = {str(row["relative_path"]).casefold(): str(groups_sorted[index]) for index, row in enumerate(integrity_rows)}
    dataset_folds = np.asarray([fold_by_path[path.casefold()] for path in relative_paths], dtype=np.int64)

    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("B9 backbone/head did not freeze")
    runtime_contract = {
        "global_pool": str(getattr(model, "global_pool", "")),
        "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", 0) or 0),
        "fc_norm": type(getattr(model, "fc_norm", None)).__name__,
        "head_drop_class": type(getattr(model, "head_drop", None)).__name__,
        "head_drop_probability": float(getattr(getattr(model, "head_drop", None), "p", -1.0)),
        "head_class": type(getattr(model, "head", None)).__name__,
        "head_shape": [int(model.head.out_features), int(model.head.in_features)],
    }
    expected_runtime = {"global_pool": "avg", "num_prefix_tokens": 5, "fc_norm": "Identity", "head_drop_class": "Dropout", "head_drop_probability": 0.0, "head_class": "Linear", "head_shape": [5, 384]}
    if runtime_contract != expected_runtime:
        raise ValueError(f"B9 runtime pooling contract mismatch: {runtime_contract}")
    state = checkpoint["model_state"]
    head_state = {key: value for key, value in state.items() if str(key).startswith("head.")}
    backbone_state = {key: value for key, value in state.items() if not str(key).startswith("head.")}
    provenance = {
        "selected_state_sha256": _state_dict_sha256(state),
        "head_state_sha256": _state_dict_sha256(head_state),
        "backbone_state_sha256": _state_dict_sha256(backbone_state),
        "backbone_eval": not model.training,
        "all_backbone_head_parameters_frozen": True,
        "runtime_contract": runtime_contract,
    }
    classifier_weight = model.head.weight.detach().cpu().float()
    _, _, paired_preflight = build_paired_adapters(classifier_weight)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "conditional_cached_postnorm_adapter_screen_only",
        "checkpoint_contract": checkpoint_contract,
        "frozen_model": provenance,
        "paired_adapter": paired_preflight,
        "source_files": {
            "tool_sha256": _sha256(Path(__file__).resolve()),
            "model_sha256": _sha256(Path(__file__).resolve().parents[1] / "models" / "dinov3_xcnorm_pair_adapter_a0.py"),
        },
        "grouping": {**group_stats, **assignment_hashes, "phash_vector_u64_sha256": phash_vector_sha256, "folds": fold_rows},
        "locked_optimization": {
            "folds": FOLDS, "seed": SEED, "epochs": EPOCHS,
            "optimizer": "AdamW", "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY, "tempered_sampling_power": TEMPERED_POWER,
            "early_stopping": False, "adapter_ema": False,
            "held_fold_selection": False, "paired_batch_schedule": True,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
        "photometric_conditions": {
            key: {**value, "status": "deferred_requires_separate_transformed_train_image_recache"}
            for key, value in PHOTOMETRIC_CONDITIONS.items()
        },
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "full_validation_permission": False,
        "test_permission": False,
    }
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    assignment_rows = [
        {
            "relative_path": path,
            "label": int(dataset_labels[index]),
            "union_group": group_by_path[path.casefold()],
            "fold": int(dataset_folds[index]),
        }
        for index, path in enumerate(relative_paths)
    ]
    with (output_dir / "train_fold_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader(); writer.writerows(assignment_rows)
    if args.preflight_only:
        return preflight

    device = _resolve_device(args.device)
    model.to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    cache = _load_or_extract_cache(
        model, dataset, relative_paths, dataset_labels, output_dir, device,
        int(args.batch_size), int(args.workers), provenance,
    )
    if not np.array_equal(np.asarray(cache["labels"], dtype=np.int64), dataset_labels):
        raise ValueError("cached labels differ from canonical train labels")
    del model
    screen = run_oof_screen(
        tokens=cache["tokens"], base_logits=cache["logits"], labels=cache["labels"],
        folds=dataset_folds, classifier_weight=classifier_weight,
        device=device, batch_size=int(args.adapter_batch_size),
        state_dir=output_dir / "fold_adapter_states",
        relative_paths=relative_paths, assignment_hashes=assignment_hashes,
        group_fold_rows=fold_rows,
    )
    readiness = assess_readiness(screen)
    summary = {
        **preflight,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "cache_manifest": cache["manifest"],
        "screen": screen,
        "readiness": readiness,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_precheck(args)
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "preflight_only": bool(args.preflight_only),
        "full_validation_permission": bool(summary.get("readiness", {}).get("full_validation_permission", False)),
        "validation_split_used": False,
        "test_split_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
