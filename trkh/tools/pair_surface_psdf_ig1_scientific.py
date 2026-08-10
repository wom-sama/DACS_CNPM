from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from trkh.tools.pair_surface_ddf_v2_scientific import (
    ActionMetrics, ActionPolicy, ActionSelection, AppliedAction, BootstrapResult, CalibratorState, CIDTDiagnosticResult,
    ConjunctiveGateEvaluationV2, DerivedRunEvidence, FrozenArrayIdentity, FrozenTargetSlice, GateResult, MapStatistics,
    NamedArtifactIdentity, NamedCalibratorState, RawScientificRun, RepeatActionComparisonV2, ScientificComparisonRecord,
    SidecarFoldPreimage, TargetShardLock, TrainingTraceEvidence, _apply_derived_actions, _build_comparison_records,
    _derive_run, _fit_all_calibrators, _gates_1_to_13, _validate_run,
)

ROWS = 763
CIDT_ROWS = 9215
ROLES = (
    "ddf_full",
    "static_matched",
    "ddf_spatial_only",
    "ddf_channel_only",
    "ddf_full_repeat",
)
RUN_NAMES = ("primary", "replay")
_ARTIFACT_NAMES = ("sidecar_preimages", "causal_scores", "invalid_fill_scores", "invalid_fill_maps", "attention_maps", "evidence_maps", "spatial_filters32", "spatial_filters16", "channel_filters16", "channel_filters32", "donor_assignment", "nonwrap_offsets")
_ARTIFACT_ARRAYS = _ARTIFACT_NAMES[1:10] + ("nonwrap_offsets",)
SHA_RE = re.compile(r"[0-9a-f]{64}")
_SENTINEL_SHA256 = {"0" * 64, "f" * 64, hashlib.sha256(b"").hexdigest()}
LOCK_SCHEMA = "trkh_psdf_ig1_lineage_lock/v1"
PROTOCOL_ID = "trkh_psdf_information_gate_a0_r1"

_SOURCE_IDENTITIES = (
    ("docs/TRKH_5CLASS_PSDF_INFORMATION_GATE_A0_R1_PROTOCOL_20260729.md", "1365a9aa022ce20d0c5d8429365e89b0c891c16fd61b65c4642ffbf650a4d78b"),
    ("docs/TRKH_5CLASS_PSDF_INFORMATION_GATE_A0_R1_MECHANISM_REGISTRY_20260729.json", "cd1d16f7b0c6a906de8b5f43e84b64b0a38a8f817892fcba957befa567c24e70"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md", "b5fa5e2ada7e078a08276902f5897a0358d21ae03658c0d398402c4277c0b679"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_ERRATUM_20260729.md", "0da08a193ca50f775047e28fcca763f8aafcc2391d54934480a3574ba1c89aeb"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2R2_FOLD_MANIFEST_20260729.json", "81a404bedc25dc7b3e7dc3dbb8d79b4ca5e97d74374083c79e0686ececb7fb2d"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2R2_FOLD_ERRATUM_20260729.md", "1e9c5e64b259b909c0d69591b7ea5dd42cfc72f7c8db1489f89de29d3f9206c7"),
    ("trkh/tools/pair_surface_ddf_v2_engine.py", "e6558d8cf9c00ec3d5f89b549902817a25afca21bff399a4766c397e80ba2774"),
    ("trkh/tools/pair_surface_ddf_v2_scientific.py", "a6168d9d438fe78dd77f87e48f4005c418c1a218e1811edbe3609a034ece850e"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_PRELOCK_PROCESS_INCIDENT_20260729.md", "c4c5eba21733d85b98b76c68312f1223ede3f6d5d1919ba25753037736b232f7"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_PRELOCK_PROCESS_INCIDENT_INDEPENDENT_REVIEW_ERRATUM_20260729.md", "b7ef5a3847b125273d7771ec35dc935ba52a570279db18f3d3ae928a95b9b96c"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_RUNTIME_GUARD_TEST_ATTESTATION_INVALIDATION_20260729.md", "5b52066708f419ab5ce728f64328585829ab5eaf84954da74c1909e2b58bd558"),
    ("docs/TRKH_5CLASS_MECHANISM_REGISTRY_20260729.json", "49b1d3422fb0d592e56c7814289fde8fe058986cc7be671357bb95d4b297566d"),
    ("docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_PRELOCK_ACTIVATION_CLOSURE_20260729.md", "d6343c7dbd5858a58935b61329a5ca355cb1c0f3c4d16efef6bfb38479037cdd"),
)

_RAW_ARRAYS = {
    "sample_indices": ("<i8", (ROWS,), False),
    "held_folds": ("<i8", (ROWS,), False),
    "bootstrap_draws": ("<i8", (2000, 158), False),
    "keeper_probabilities": ("<f8", (ROWS, 5), True),
    "causal_scores": ("<f8", (6, ROWS), True),
    "invalid_fill_scores": ("<f8", (2, ROWS), True),
    "invalid_fill_maps": ("<f8", (2, ROWS, 4, 16, 16), True),
    "attention_maps": ("<f8", (ROWS, 4, 16, 16), True),
    "evidence_maps": ("<f8", (ROWS, 4, 16, 16), True),
    "valid64": ("bool", (ROWS, 64, 64), False),
    "valid32": ("bool", (ROWS, 32, 32), False),
    "valid16": ("bool", (ROWS, 16, 16), False),
    "bbox_valid16": ("bool", (ROWS, 16, 16), False),
    "nonwrap_offsets": ("<i8", (ROWS, 2, 2), False),
    "spatial_filters32": ("<f8", (ROWS, 9, 32, 32), True),
    "spatial_filters16": ("<f8", (ROWS, 9, 16, 16), True),
    "channel_filters16": ("<f8", (ROWS, 16, 9), True),
    "channel_filters32": ("<f8", (ROWS, 32, 9), True),
    "cidt_sample_indices": ("<i8", (CIDT_ROWS,), False),
}
_RAW_KEYS = frozenset(_RAW_ARRAYS) | {
    "component_ids", "component_order", "sidecar_preimages",
    "donor_artifact_sha256", "artifact_identities", "trace",
}
_PREIMAGE_KEYS = {
    "role", "outer_fold", "calibration_sample_indices",
    "calibration_raw_scores", "held_sample_indices", "held_raw_scores",
}
_TRACE_SPECS = {
    "losses": ("<f8", (5, 5, 160), True),
    "gradient_sha256": ("<U64", (5, 5, 160), False),
    "optimizer_state_sha256": ("<U64", (5, 5, 161), False),
    "initialized_model_state_sha256": ("<U64", (5, 5), False),
    "final_model_state_sha256": ("<U64", (5, 5), False),
    "prepared_input_sha256": ("<U64", (2, ROWS), False),
}

@dataclass(frozen=True)
class IG1LineageLock:
    schema: str
    protocol_id: str
    repository_root: str
    authorization_sha256: str
    machine_lock_sha256: str
    environment_lock_sha256: str
    access_policy_sha256: str
    target_blind_root_sha256: str
    primary_raw_root_sha256: str
    replay_raw_root_sha256: str
    primary_access_root_sha256: str
    replay_access_root_sha256: str
    component_order_sha256: str
    sample_component_mapping_sha256: str
    bootstrap_draws: FrozenArrayIdentity
    keeper_probabilities: FrozenArrayIdentity
    valid64: FrozenArrayIdentity
    valid32: FrozenArrayIdentity
    valid16: FrozenArrayIdentity
    bbox_valid16: FrozenArrayIdentity
    nonwrap_offsets: FrozenArrayIdentity
    donor_artifact_sha256: str
    sidecar_preimage_registry_sha256: str
    artifact_registry_sha256: str
    bbox_unusable_sample_indices: FrozenArrayIdentity
    bbox_unusable_records_sha256: str
    target_shards: Tuple[TargetShardLock, ...]
    target_shard_registry_sha256: str
    cidt_sample_indices: FrozenArrayIdentity
    cidt_targets: FrozenArrayIdentity
    cidt_baseline_predictions: FrozenArrayIdentity
    cidt_registry_sha256: str
    source_identities: Tuple[NamedArtifactIdentity, ...]
    frozen_record_sha256: str

@dataclass(frozen=True)
class PreHeldBundle:
    run_name: str
    union_probabilities: np.ndarray
    pair_probabilities: np.ndarray
    action_probabilities: np.ndarray
    calibrators: Tuple[NamedCalibratorState, ...]
    action_selections: Tuple[Tuple[ActionSelection, ...], ...]
    primary_action: AppliedAction
    repeat_action: AppliedAction

@dataclass(frozen=True)
class PreHeldBundleReceipt:
    run_name: str
    path: str
    byte_count: int
    sha256: str

@dataclass(frozen=True)
class IG1ScientificResult:
    outcome: str
    evaluation: ConjunctiveGateEvaluationV2
    preheld_receipts: Tuple[PreHeldBundleReceipt, ...]
    decision_summary: Mapping[str, object]

def _keys(value: object, expected: Sequence[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    keys = tuple(value)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(f"{name} keys must be strings")
    observed = set(keys)
    wanted = set(expected)
    if observed != wanted:
        raise ValueError(f"{name} keys differ: missing={sorted(wanted-observed)}, extra={sorted(observed-wanted)}")
    return value

def _digest(value: object, name: str) -> str:
    if not isinstance(value, str): raise TypeError(f"{name} must be a string")
    if SHA_RE.fullmatch(value) is None or value in _SENTINEL_SHA256:
        raise ValueError(f"{name} is not a canonical SHA-256")
    return value

def _arr(value: object, dtype: str, shape: Tuple[int, ...], name: str, finite: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise TypeError(f"{name} must be ndarray {np.dtype(dtype)} {shape}")
    if finite and not bool(np.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    output = np.array(value, dtype=np.dtype(dtype), order="C", copy=True)
    output.setflags(write=False)
    return output

def _array_identity(value: np.ndarray) -> FrozenArrayIdentity:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(array.tobytes())
    return FrozenArrayIdentity(str(array.dtype), tuple(int(item) for item in array.shape), digest.hexdigest())

def _identity(value: object, name: str) -> FrozenArrayIdentity:
    item = _keys(value, ("dtype", "shape", "array_sha256"), name)
    shape = item["shape"]
    if not isinstance(shape, (list, tuple)) or any(type(x) is not int or x < 0 for x in shape):
        raise TypeError(f"{name}.shape is invalid")
    dtype = str(item["dtype"])
    np.dtype(dtype)
    return FrozenArrayIdentity(dtype, tuple(shape), _digest(item["array_sha256"], name))

def _encode(value: object, include_array_bytes: bool) -> object:
    if isinstance(value, np.ndarray):
        identity = _array_identity(value)
        if not include_array_bytes:
            return _plain_identity(identity)
        output = {"dtype": identity.dtype, "shape": list(identity.shape), "sha256": identity.array_sha256}
        output["data_b64"] = base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii")
        return {"__ndarray__": output}
    if is_dataclass(value):
        return {"__type__": type(value).__name__, "fields": {field.name: _encode(getattr(value, field.name), include_array_bytes) for field in fields(value)}}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical mappings require string keys")
        return {key: _encode(value[key], include_array_bytes) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_encode(item, include_array_bytes) for item in value]
    if isinstance(value, np.generic):
        return _encode(value.item(), include_array_bytes)
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Canonical payload contains a non-finite float")
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")

def _canonical_bytes(value: object, include_array_bytes: bool = False) -> bytes:
    return json.dumps(_encode(value, include_array_bytes), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _deep_freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = np.array(value, order="C", copy=True)
        if output.dtype.kind in "fc" and not bool(np.isfinite(output).all()):
            raise FloatingPointError("Pinned output contains non-finite values")
        output.setflags(write=False)
        return output
    if is_dataclass(value):
        return type(value)(**{field.name: _deep_freeze(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _deep_freeze(item) for key, item in value.items()}
    return value

def _freeze_commit(value: Any) -> Tuple[Any, Tuple[bytes, str]]:
    frozen = _deep_freeze(value); payload = _canonical_bytes(frozen, include_array_bytes=True)
    return frozen, (payload, _sha(payload))

def _verify_commit(value: object, commit: Tuple[bytes, str], name: str) -> None:
    observed = _canonical_bytes(value, include_array_bytes=True)
    if observed != commit[0] or _sha(observed) != commit[1]: raise RuntimeError(name + " changed across pinned call")

def _plain_identity(value: FrozenArrayIdentity) -> Mapping[str, object]:
    return {"dtype": value.dtype, "shape": list(value.shape), "array_sha256": value.array_sha256}

def _json_sha(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))

def _slice_lock(value: object, name: str) -> FrozenTargetSlice:
    item = _keys(value, ("sample_indices", "targets"), name)
    return FrozenTargetSlice(_identity(item["sample_indices"], name + ".sample_indices"), _identity(item["targets"], name + ".targets"))

def _parse_lock(value: object) -> IG1LineageLock:
    expected = tuple(field.name for field in fields(IG1LineageLock))
    item = _keys(value, expected, "IG1 lineage lock")
    frozen_wire = {name: item[name] for name in expected if name != "frozen_record_sha256"}
    frozen_wire_root = _sha(_canonical_bytes(frozen_wire))
    identity_names = ("bootstrap_draws", "keeper_probabilities", "valid64", "valid32", "valid16", "bbox_valid16", "nonwrap_offsets", "bbox_unusable_sample_indices", "cidt_sample_indices", "cidt_targets", "cidt_baseline_predictions")
    parsed = dict(item)
    for name in ("schema", "protocol_id", "repository_root"):
        if not isinstance(item[name], str):
            raise TypeError("lock." + name + " must be a string")
    for name in (field.name for field in fields(IG1LineageLock) if field.name.endswith("sha256")):
        parsed[name] = _digest(item[name], "lock." + name)
    for name in identity_names:
        parsed[name] = _identity(item[name], "lock." + name)
    shards = item["target_shards"]
    if not isinstance(shards, (list, tuple)) or len(shards) != 5:
        raise ValueError("lock.target_shards must contain five folds")
    parsed_shards = []
    for position, shard_value in enumerate(shards):
        shard = _keys(shard_value, ("outer_fold", "fit", "calibration", "held"), f"target_shards[{position}]")
        if type(shard["outer_fold"]) is not int or shard["outer_fold"] != position:
            raise ValueError("Target shard order changed")
        parsed_shards.append(TargetShardLock(position, _slice_lock(shard["fit"], "fit"), _slice_lock(shard["calibration"], "calibration"), _slice_lock(shard["held"], "held")))
    parsed["target_shards"] = tuple(parsed_shards)
    sources = item["source_identities"]
    if not isinstance(sources, (list, tuple)):
        raise TypeError("source_identities must be a sequence")
    parsed_sources = []
    for source in sources:
        source_item = _keys(source, ("name", "sha256"), "source identity")
        if not isinstance(source_item["name"], str):
            raise TypeError("source identity name must be a string")
        parsed_sources.append(NamedArtifactIdentity(source_item["name"], _digest(source_item["sha256"], "source identity")))
    parsed["source_identities"] = tuple(parsed_sources)
    lock = IG1LineageLock(**parsed)
    _validate_lock(lock)
    if frozen_wire_root != lock.frozen_record_sha256:
        raise ValueError("IG1 frozen-record root changed")
    return lock

def _validate_lock(lock: IG1LineageLock) -> None:
    if lock.schema != LOCK_SCHEMA or lock.protocol_id != PROTOCOL_ID:
        raise ValueError("IG1 schema/protocol identity changed")
    root = Path(lock.repository_root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("repository_root must be an absolute resolved path")
    if lock.primary_access_root_sha256 != lock.replay_access_root_sha256:
        raise ValueError("Primary/replay normalized access roots differ")
    expected_sources = tuple(NamedArtifactIdentity(*item) for item in _SOURCE_IDENTITIES)
    if lock.source_identities != expected_sources:
        raise ValueError("Retained scientific source registry changed")
    for source in lock.source_identities:
        path = (root / Path(source.name)).resolve()
        if os.path.commonpath((str(root), str(path))) != str(root) or not path.is_file():
            raise ValueError(f"Retained source is outside/missing: {source.name}")
        if _sha(path.read_bytes()) != source.sha256:
            raise ValueError(f"Retained source hash changed: {source.name}")
    shard_payload = [{"outer_fold": shard.outer_fold, "fit": {"sample_indices": _plain_identity(shard.fit.sample_indices), "targets": _plain_identity(shard.fit.targets)}, "calibration": {"sample_indices": _plain_identity(shard.calibration.sample_indices), "targets": _plain_identity(shard.calibration.targets)}, "held": {"sample_indices": _plain_identity(shard.held.sample_indices), "targets": _plain_identity(shard.held.targets)}} for shard in lock.target_shards]
    if _json_sha(shard_payload) != lock.target_shard_registry_sha256:
        raise ValueError("Target-shard registry root changed")
    cidt_payload = {"sample_indices": _plain_identity(lock.cidt_sample_indices), "targets": _plain_identity(lock.cidt_targets), "baseline_predictions": _plain_identity(lock.cidt_baseline_predictions)}
    if _json_sha(cidt_payload) != lock.cidt_registry_sha256:
        raise ValueError("CIDT registry root changed")

def _parse_raw(value: object, lock: IG1LineageLock, name: str) -> Tuple[Dict[str, object], str]:
    item = _keys(value, _RAW_KEYS, name + ".raw")
    wire_root = _sha(_canonical_bytes(item))
    output: Dict[str, object] = {}
    for field, (dtype, shape, finite) in _RAW_ARRAYS.items():
        output[field] = _arr(item[field], dtype, shape, name + "." + field, finite)
    for field in ("bootstrap_draws", "keeper_probabilities", "valid64", "valid32", "valid16", "bbox_valid16", "nonwrap_offsets", "cidt_sample_indices"):
        if _array_identity(output[field]) != getattr(lock, field):
            raise ValueError(name + "." + field + " identity changed")
    if not isinstance(item["component_ids"], (list, tuple)) or not isinstance(item["component_order"], (list, tuple)) or any(not isinstance(part, str) for part in tuple(item["component_ids"]) + tuple(item["component_order"])):
        raise TypeError(name + " component axes must be string sequences")
    components = tuple(item["component_ids"])
    order = tuple(item["component_order"])
    if len(components) != ROWS or len(order) != 158:
        raise ValueError(name + " component axes are invalid")
    components, order = tuple(_digest(part, name + ".component") for part in components), tuple(_digest(part, name + ".component_order") for part in order)
    if tuple(sorted(set(components))) != order or _json_sha(list(order)) != lock.component_order_sha256 or _json_sha([[int(index), component] for index, component in zip(output["sample_indices"], components)]) != lock.sample_component_mapping_sha256:
        raise ValueError(name + " component lineage changed")
    output["component_ids"], output["component_order"] = components, order
    preimages = item["sidecar_preimages"]
    if not isinstance(preimages, (list, tuple)) or len(preimages) != 25:
        raise ValueError(name + " must contain 25 sidecar preimages")
    parsed_preimages = []
    for position, raw_preimage in enumerate(preimages):
        preimage = _keys(raw_preimage, _PREIMAGE_KEYS, f"{name}.preimage[{position}]")
        role, fold = ROLES[position // 5], position % 5
        if not isinstance(preimage["role"], str) or type(preimage["outer_fold"]) is not int or preimage["role"] != role or preimage["outer_fold"] != fold:
            raise ValueError(name + " sidecar axes changed")
        cal_ids = preimage["calibration_sample_indices"]
        held_ids = preimage["held_sample_indices"]
        if not isinstance(cal_ids, np.ndarray) or not isinstance(held_ids, np.ndarray):
            raise TypeError(name + " sidecar indices must be ndarrays")
        cal_ids = _arr(cal_ids, "<i8", cal_ids.shape, name + ".cal_ids")
        held_ids = _arr(held_ids, "<i8", held_ids.shape, name + ".held_ids")
        if cal_ids.ndim != 1 or held_ids.ndim != 1:
            raise ValueError(name + " sidecar indices must be one-dimensional")
        cal_scores = _arr(preimage["calibration_raw_scores"], "<f8", (cal_ids.size, 4), name + ".cal_scores", True)
        held_scores = _arr(preimage["held_raw_scores"], "<f8", (held_ids.size, 4), name + ".held_scores", True)
        parsed_preimages.append(SidecarFoldPreimage(role, fold, cal_ids, cal_scores, held_ids, held_scores))
    output["sidecar_preimages"] = tuple(parsed_preimages)
    indices, folds = output["sample_indices"], output["held_folds"]
    if not bool(np.all(indices[1:] > indices[:-1])) or not bool(np.isin(folds, np.arange(5)).all()):
        raise ValueError(name + " sample/fold axes are invalid")
    for preimage in output["sidecar_preimages"]:
        expected_cal = indices[folds == ((preimage.outer_fold + 1) % 5)]
        expected_held = indices[folds == preimage.outer_fold]
        shard = lock.target_shards[preimage.outer_fold]
        if not np.array_equal(preimage.calibration_sample_indices, expected_cal) or not np.array_equal(preimage.held_sample_indices, expected_held) or _array_identity(expected_cal) != shard.calibration.sample_indices or _array_identity(expected_held) != shard.held.sample_indices:
            raise ValueError(name + " sidecar target-role indexing changed")
    output["donor_artifact_sha256"] = _digest(item["donor_artifact_sha256"], name + ".donor")
    if output["donor_artifact_sha256"] != lock.donor_artifact_sha256:
        raise ValueError(name + " donor artifact changed")
    artifacts = item["artifact_identities"]
    if not isinstance(artifacts, (list, tuple)):
        raise TypeError(name + ".artifact_identities must be a sequence")
    parsed_artifacts = []
    for entry in artifacts:
        artifact = _keys(entry, ("name", "sha256"), name + ".artifact")
        if not isinstance(artifact["name"], str):
            raise TypeError(name + " artifact names must be strings")
        parsed_artifacts.append(NamedArtifactIdentity(artifact["name"], _digest(artifact["sha256"], name + ".artifact")))
    output["artifact_identities"] = tuple(parsed_artifacts)
    trace = _keys(item["trace"], _TRACE_SPECS, name + ".trace")
    trace_value = TrainingTraceEvidence(**{field: _arr(trace[field], spec[0], spec[1], name + ".trace." + field, spec[2]) for field, spec in _TRACE_SPECS.items()})
    for field in tuple(_TRACE_SPECS)[1:]:
        for part in getattr(trace_value, field).flat:
            _digest(str(part), name + ".trace." + field)
    if not np.array_equal(trace_value.prepared_input_sha256[0], trace_value.prepared_input_sha256[1]):
        raise ValueError(name + " invalid-fill trace changed")
    output["trace"] = trace_value
    sidecar_root = _json_sha([{"role": part.role, "outer_fold": part.outer_fold, "calibration_sample_indices": _plain_identity(_array_identity(part.calibration_sample_indices)), "calibration_raw_scores": _plain_identity(_array_identity(part.calibration_raw_scores)), "held_sample_indices": _plain_identity(_array_identity(part.held_sample_indices)), "held_raw_scores": _plain_identity(_array_identity(part.held_raw_scores))} for part in output["sidecar_preimages"]])
    artifact_digests = {field: _array_identity(output[field]).array_sha256 for field in _ARTIFACT_ARRAYS}
    artifact_digests.update({"sidecar_preimages": sidecar_root, "donor_assignment": output["donor_artifact_sha256"]})
    expected_artifacts = tuple(NamedArtifactIdentity(field, artifact_digests[field]) for field in _ARTIFACT_NAMES)
    if sidecar_root != lock.sidecar_preimage_registry_sha256 or output["artifact_identities"] != expected_artifacts or _json_sha([{"name": part.name, "sha256": part.sha256} for part in expected_artifacts]) != lock.artifact_registry_sha256:
        raise ValueError(name + " sidecar/artifact registry changed")
    return output, wire_root

def _target_slices(value: object, lock: IG1LineageLock, base: Mapping[str, object], role: str, name: str) -> np.ndarray:
    if role not in ("calibration", "held") or not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError(name + " target slices are invalid")
    indices = base["sample_indices"]
    targets = np.full(ROWS, -1, dtype=np.dtype("<i8"))
    seen = np.zeros(ROWS, dtype=np.bool_)
    for fold, raw_slice in enumerate(value):
        item = _keys(raw_slice, ("outer_fold", "sample_indices", "targets"), f"{name}[{fold}]")
        if type(item["outer_fold"]) is not int or item["outer_fold"] != fold:
            raise ValueError(name + " target slice order changed")
        expected = getattr(lock.target_shards[fold], role)
        ids_value, targets_value = item["sample_indices"], item["targets"]
        if not isinstance(ids_value, np.ndarray) or not isinstance(targets_value, np.ndarray):
            raise TypeError(name + " target slices require ndarrays")
        ids = _arr(ids_value, "<i8", ids_value.shape, name + ".ids")
        truth = _arr(targets_value, "<i8", targets_value.shape, name + ".targets")
        if ids.ndim != 1 or truth.shape != ids.shape or _array_identity(ids) != expected.sample_indices or _array_identity(truth) != expected.targets:
            raise ValueError(name + " target slice identity changed")
        positions = np.searchsorted(indices, ids)
        if bool((positions >= ROWS).any()) or not np.array_equal(indices[positions], ids) or bool(seen[positions].any()):
            raise ValueError(name + " target slices overlap or omit row identities")
        targets[positions], seen[positions] = truth, True
    if not bool(seen.all()) or not bool(np.isin(targets, np.arange(5)).all()):
        raise ValueError(name + " target union is incomplete")
    targets.setflags(write=False)
    return targets

def _make_run(base: Mapping[str, object], targets: np.ndarray, cidt_targets: np.ndarray, cidt_baseline: np.ndarray) -> RawScientificRun:
    values = dict(base)
    values["targets"] = targets
    values["cidt_targets"] = cidt_targets
    values["cidt_baseline_predictions"] = cidt_baseline
    return RawScientificRun(**values)

def _parse_calibration(value: object, lock: IG1LineageLock, name: str) -> Tuple[RawScientificRun, str]:
    item = _keys(value, ("target_blind_root_sha256", "raw", "calibration_target_slices"), name)
    if _digest(item["target_blind_root_sha256"], name) != lock.target_blind_root_sha256:
        raise ValueError(name + " target-blind root changed")
    base, raw_root = _parse_raw(item["raw"], lock, name)
    if raw_root != getattr(lock, name + "_raw_root_sha256"):
        raise ValueError(name + " raw root changed")
    targets = _target_slices(item["calibration_target_slices"], lock, base, "calibration", name)
    unusable = ~base["bbox_valid16"].any(axis=(1, 2))
    unusable_ids = np.ascontiguousarray(base["sample_indices"][unusable], dtype=np.dtype("<i8"))
    unusable_records = [{"sample_index": int(index), "target": int(target)} for index, target in zip(unusable_ids, targets[unusable])]
    if _array_identity(unusable_ids) != lock.bbox_unusable_sample_indices or _json_sha(unusable_records) != lock.bbox_unusable_records_sha256:
        raise ValueError(name + " bbox-unusable lineage changed")
    sentinel = _arr(np.full(CIDT_ROWS, -1, dtype=np.dtype("<i8")), "<i8", (CIDT_ROWS,), name + ".cidt_sentinel")
    return _make_run(base, targets, sentinel, sentinel), raw_root

def _parse_held(value: object, lock: IG1LineageLock, calibration: RawScientificRun, raw_root: str, name: str) -> Dict[str, object]:
    item = _keys(value, ("target_blind_root_sha256", "raw", "held_target_slices"), name)
    if _digest(item["target_blind_root_sha256"], name) != lock.target_blind_root_sha256:
        raise ValueError(name + " target-blind root changed after release")
    base, observed_root = _parse_raw(item["raw"], lock, name + ".complete")
    if observed_root != raw_root or observed_root != getattr(lock, name + "_raw_root_sha256"):
        raise ValueError(name + " raw bytes changed across release")
    held_targets = _target_slices(item["held_target_slices"], lock, base, "held", name)
    if not np.array_equal(held_targets, calibration.targets):
        raise ValueError(name + " calibration/held target roles disagree")
    return base

def _parse_cidt(value: object, lock: IG1LineageLock) -> Tuple[np.ndarray, np.ndarray]:
    item = _keys(value, ("cidt_targets", "cidt_baseline_predictions"), "CIDT release")
    cidt_targets = _arr(item["cidt_targets"], "<i8", (CIDT_ROWS,), "cidt.targets")
    cidt_baseline = _arr(item["cidt_baseline_predictions"], "<i8", (CIDT_ROWS,), "cidt.baseline")
    if _array_identity(cidt_targets) != lock.cidt_targets or _array_identity(cidt_baseline) != lock.cidt_baseline_predictions:
        raise ValueError("CIDT identity changed")
    return cidt_targets, cidt_baseline

def _bundle(run_name: str, run: RawScientificRun) -> PreHeldBundle:
    frozen_run, run_commit = _freeze_commit(run)
    fit = _fit_all_calibrators(frozen_run)
    _verify_commit(frozen_run, run_commit, run_name + " calibration run")
    fit, fit_commit = _freeze_commit(fit)
    union, pair, calibrators, selections = fit
    action_probabilities, primary_action, repeat_action = _apply_derived_actions(frozen_run, union, selections)
    _verify_commit(fit, fit_commit, run_name + " fit result"); _verify_commit(frozen_run, run_commit, run_name + " calibration run after actions")
    bundle = _deep_freeze(PreHeldBundle(run_name, union, pair, action_probabilities, calibrators, selections, primary_action, repeat_action))
    if bundle.union_probabilities.shape != (5, 4, ROWS) or bundle.pair_probabilities.shape != (5, 3, 2, ROWS) or bundle.action_probabilities.shape != (2, ROWS):
        raise ValueError("Pinned probability output shape changed")
    if len(bundle.calibrators) != 250 or tuple(len(items) for items in bundle.action_selections) != (5, 5):
        raise ValueError("Pinned calibrator/action cardinality changed")
    if not all(isinstance(item, NamedCalibratorState) for item in bundle.calibrators) or not all(isinstance(item, ActionSelection) for group in bundle.action_selections for item in group):
        raise TypeError("Pinned calibrator/action output type changed")
    if not isinstance(bundle.primary_action, AppliedAction) or not isinstance(bundle.repeat_action, AppliedAction):
        raise TypeError("Pinned AppliedAction output type changed")
    return bundle

def _persist_bundle(bundle: PreHeldBundle, path_value: object) -> Tuple[PreHeldBundleReceipt, bytes]:
    path = Path(str(path_value)).resolve()
    payload = _canonical_bytes(bundle, include_array_bytes=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    observed = path.read_bytes()
    if observed != payload:
        raise IOError("Pre-held bundle readback changed")
    receipt = PreHeldBundleReceipt(bundle.run_name, str(path), len(payload), _sha(payload))
    return receipt, payload

def _verify_bundles(receipts: Sequence[PreHeldBundleReceipt], persisted: Mapping[str, bytes]) -> None:
    for receipt in receipts:
        observed = Path(receipt.path).read_bytes()
        if observed != persisted[receipt.run_name] or len(observed) != receipt.byte_count or _sha(observed) != receipt.sha256:
            raise IOError("Pre-held bundle changed")

def _summary_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _plain_identity(_array_identity(value))
    if is_dataclass(value):
        return {field.name: _summary_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary_value(item) for item in value]
    if isinstance(value, np.generic):
        return _summary_value(value.item())
    return value

def _numeric_array_summary(value: np.ndarray) -> Mapping[str, object]:
    return {"identity": _plain_identity(_array_identity(value)), "values": np.asarray(value).tolist()}

def _decision_summary(lineage: IG1LineageLock, evaluation: ConjunctiveGateEvaluationV2, receipts: Tuple[PreHeldBundleReceipt, ...]) -> Mapping[str, object]:
    failed = set(evaluation.gates[14].details["failed_records"])
    comparisons = [{"category": record.category, "name": record.name, "kind": record.kind, "exact": f"{record.category}/{record.name}/{record.kind}" not in failed} for record in evaluation.comparison_records]
    if len(comparisons) != 69: raise RuntimeError("Gate-14 comparison count changed")
    derived = {}
    for name, evidence in (("primary", evaluation.formal), ("replay", evaluation.replay)):
        numeric = ("pooled_auroc", "pooled_auprc", "fold_auroc", "fold_auprc", "pair_auroc", "pair_auprc", "causal_auroc", "causal_auprc", "filter_dispersions")
        derived[name] = {field: _numeric_array_summary(getattr(evidence, field)) for field in numeric}
        derived[name].update({
            "bootstrap": {"auroc_delta": _plain_identity(_array_identity(evidence.bootstrap.auroc_delta)), "auprc_delta": _plain_identity(_array_identity(evidence.bootstrap.auprc_delta)), "auroc_interval": list(evidence.bootstrap.auroc_interval), "auprc_interval": list(evidence.bootstrap.auprc_interval)},
            "primary_action_metrics": _summary_value(evidence.primary_action_metrics),
            "repeat_action_metrics": _summary_value(evidence.repeat_action_metrics),
            "repeat_comparison": _summary_value(evidence.repeat_comparison),
            "cidt": _summary_value(evidence.cidt),
            "derived_array_identities": {field: _plain_identity(_array_identity(getattr(evidence, field))) for field in ("held_raw_scores", "union_probabilities", "pair_probabilities", "action_probabilities")},
            "calibrators_sha256": _sha(_canonical_bytes(evidence.calibrators)),
            "action_selections_sha256": _sha(_canonical_bytes(evidence.action_selections)),
            "map_statistics_sha256": _sha(_canonical_bytes(evidence.map_statistics)),
        })
    return {
        "schema": "trkh_psdf_ig1_decision_summary/v1",
        "protocol_id": PROTOCOL_ID,
        "outcome": "information_pass" if evaluation.all_passed else "scientific_fail",
        "all_passed": evaluation.all_passed,
        "lineage": {
            "frozen_record_sha256": lineage.frozen_record_sha256,
            "authorization_sha256": lineage.authorization_sha256,
            "target_blind_root_sha256": lineage.target_blind_root_sha256,
            "raw_roots": {"primary": lineage.primary_raw_root_sha256, "replay": lineage.replay_raw_root_sha256},
            "normalized_access_roots": {"primary": lineage.primary_access_root_sha256, "replay": lineage.replay_access_root_sha256},
            "target_shard_registry_sha256": lineage.target_shard_registry_sha256,
            "cidt_registry_sha256": lineage.cidt_registry_sha256,
            "source_registry_sha256": _json_sha([{"name": part.name, "sha256": part.sha256} for part in lineage.source_identities]),
            "source_identities": _summary_value(lineage.source_identities),
        },
        "gates": {str(index): {"passed": evaluation.gates[index].passed, "details": _summary_value(evaluation.gates[index].details)} for index in range(1, 15)},
        "derived": derived,
        "comparison_records": comparisons,
        "preheld_receipts": _summary_value(receipts),
    }

def evaluate_ig1_pair(
    *,
    lock: Mapping[str, object],
    calibration_runs: Mapping[str, Mapping[str, object]],
    release_held_runs: Callable[[], Mapping[str, Mapping[str, object]]],
    release_cidt: Callable[[], Mapping[str, object]],
    preheld_paths: Mapping[str, str],
) -> IG1ScientificResult:
    lineage = _parse_lock(lock)
    calibration_map = _keys(calibration_runs, RUN_NAMES, "calibration runs")
    path_map = _keys(preheld_paths, RUN_NAMES, "pre-held paths")
    if not callable(release_held_runs) or not callable(release_cidt): raise TypeError("release callbacks must be callable")
    resolved_paths = tuple(str(Path(str(path_map[name])).resolve()) for name in RUN_NAMES)
    if len(set(resolved_paths)) != 2 or any(Path(path).exists() for path in resolved_paths): raise FileExistsError("Pre-held paths must be distinct and absent")
    calibration: Dict[str, RawScientificRun] = {}
    raw_roots: Dict[str, str] = {}
    for name in RUN_NAMES:
        calibration[name], raw_roots[name] = _parse_calibration(calibration_map[name], lineage, name)
    receipts_list = []
    persisted: Dict[str, bytes] = {}
    for name in RUN_NAMES:
        receipt, payload = _persist_bundle(_bundle(name, calibration[name]), resolved_paths[RUN_NAMES.index(name)])
        receipts_list.append(receipt)
        persisted[name] = payload
    receipts = tuple(receipts_list)
    _verify_bundles(receipts, persisted)
    held_map = _keys(release_held_runs(), RUN_NAMES, "held runs")
    held = {name: _parse_held(held_map[name], lineage, calibration[name], raw_roots[name], name) for name in RUN_NAMES}
    _verify_bundles(receipts, persisted)
    cidt_targets, cidt_baseline = _parse_cidt(release_cidt(), lineage)
    _verify_bundles(receipts, persisted)
    complete = {name: _freeze_commit(_make_run(held[name], calibration[name].targets, cidt_targets, cidt_baseline)) for name in RUN_NAMES}
    for name in RUN_NAMES:
        _validate_run(complete[name][0], lineage, name=name, exact_a0=True)
        _verify_commit(complete[name][0], complete[name][1], name + " complete run after validation")
    derived = {}
    for name in RUN_NAMES:
        evidence = _derive_run(complete[name][0], lineage)
        _verify_commit(complete[name][0], complete[name][1], name + " complete run after derivation")
        derived[name] = _freeze_commit(evidence)
    for name in RUN_NAMES:
        evidence = derived[name][0]
        refit = PreHeldBundle(name, evidence.union_probabilities, evidence.pair_probabilities, evidence.action_probabilities, evidence.calibrators, evidence.action_selections, evidence.primary_action, evidence.repeat_action)
        if _canonical_bytes(_deep_freeze(refit), include_array_bytes=True) != persisted[name]: raise ValueError(name + " refit differs from its pre-held bundle")
    gates = {}
    for name in RUN_NAMES:
        gate_map = _gates_1_to_13(complete[name][0], derived[name][0])
        _verify_commit(derived[name][0], derived[name][1], name + " derived evidence after gates"); _verify_commit(complete[name][0], complete[name][1], name + " complete run after gates")
        gate_items, gate_commit = _freeze_commit(tuple(gate_map.items()))
        gates[name] = (dict(gate_items), gate_commit)
    records, gate14 = _build_comparison_records(lineage, complete["primary"][0], complete["replay"][0], derived["primary"][0], derived["replay"][0], gates["primary"][0], gates["replay"][0])
    for name in RUN_NAMES:
        _verify_commit(complete[name][0], complete[name][1], name + " complete run after comparison")
        _verify_commit(derived[name][0], derived[name][1], name + " derived evidence after comparison")
        _verify_commit(tuple(gates[name][0].items()), gates[name][1], name + " gates after comparison")
    primary_gates = {**gates["primary"][0], 14: gate14}
    evaluation = ConjunctiveGateEvaluationV2(bool(all(gate.passed for gate in primary_gates.values())), primary_gates, derived["primary"][0], derived["replay"][0], records)
    summary = _decision_summary(lineage, evaluation, receipts)
    _verify_bundles(receipts, persisted)
    return IG1ScientificResult(str(summary["outcome"]), evaluation, receipts, summary)
