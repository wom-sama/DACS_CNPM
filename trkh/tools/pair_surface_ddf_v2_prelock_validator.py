from __future__ import annotations

"""Finalized-evidence verifier for the Pair-Surface DDF v2 pre-lock stages.

The verifier opens only hash-listed boundary/finalized-evidence files.  The
original cohort/cache/lineage records are identity-only and are never opened.
It grants no execution authority and cannot create S1, S2, or formal evidence.
"""

import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PRELOCK_VERIFIER_API_VERSION = 2
SCHEMA_VERSION = 2
PROTOCOL_ID = "trkh_pair_surface_ddf_a0_v2"
FOUNDATION_COMMIT = "9ea25ea87ae6d497b80dd86c27e59391d21c4a9b"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

ROWS = 763
FOLDS = 5
COMPONENTS = 158
FOLD_SIZES = (153, 153, 153, 152, 152)
TARGET_COUNTS = {0: 158, 1: 541, 2: 54, 3: 0, 4: 10}
ROLES = (
    "ddf_full", "static_matched", "ddf_spatial_only",
    "ddf_channel_only", "ddf_full_repeat",
)
PRIMARY_SEED = 20260729
REPEAT_OFFSET = 100000

COHORT_IDENTITY = {
    "sample_indices_array_sha256": "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05",
    "held_folds_array_sha256": "924a3e98e5ef1bd814511675951d83ee79ff9788894762b4688fb93fabbea255",
    "targets_array_sha256": "15c43ecc7335c7a7febc4e0fbf622df7ad593fc52e5d9f14e5cc27acc16fc000",
    "sample_fold_pairs_sha256": "fd61cd1c20c9ca66c113db1ec65d25c8e0ea8c4ec851f80f3c15516f993921df",
    "fold_mapping_sha256": "c0726d114df69290d68d9f9e1a40857074ccb1a66e5ca41883caa004f2b0ea40",
    "component_order_sha256": "e10e1045e6ede63ade0cb436b684be7a6ab377d0d27f74480ac8721756045ccb",
}
XAI_SAMPLE_INDICES = (
    163, 2370, 856, 338, 58, 2563, 1939, 2698, 341, 2004,
    1066, 313, 2, 2906, 345, 923, 195, 2411, 791, 363,
    3624, 3934, 4012, 4040, 4613, 4626, 5379, 5381, 5732, 5740,
)
XAI_ARRAY_SHA256 = "f5025a0b2f6340a8b9da057ece9644d11e6ba598c5d03179245bf8a7642bb170"
CIDT_IDENTITY = {
    "rows": 9215,
    "targets_array_sha256": "0b54f448d890b9032dcc336f5a1798927c19bf6b35b2a094bf318208e4785e7d",
    "keeper_predictions_array_sha256": "71295e8183f3b705992c16210da69470531b546108b65b65987e2fb72313b2d5",
}
BOOTSTRAP_ARRAY_SHA256 = "d0b5041be27e27cd17fa55acfb44e1a072e6b6dde360c115fb21f022030b981c"
NOISE_ARRAY_SHA256 = "9c18e8ab55f6bbf15d1372876bf2578cd094f2a3f1a755e89992c086bf365596"

# Existing committed scientific lineage is immutable.  A future boundary
# registry must additionally pin every dynamic runtime/pre-lock artifact.
FIXED_BOUNDARY_SHA256 = {
    "v2_protocol_base": "b5fa5e2ada7e078a08276902f5897a0358d21ae03658c0d398402c4277c0b679",
    "v2_protocol_erratum": "0da08a193ca50f775047e28fcca763f8aafcc2391d54934480a3574ba1c89aeb",
    "v2_fold_erratum_r2": "1e9c5e64b259b909c0d69591b7ea5dd42cfc72f7c8db1489f89de29d3f9206c7",
    "v2_fold_manifest_r2": "81a404bedc25dc7b3e7dc3dbb8d79b4ca5e97d74374083c79e0686ececb7fb2d",
    "mechanism_registry_v3_lineage": "49b1d3422fb0d592e56c7814289fde8fe058986cc7be671357bb95d4b297566d",
    "v2_prelock_sequencing_erratum": "b986835a2ac7eff3ac98cd68b19331c6de3c828268d9539c8f46ff57afa4d1a0",
    "v2_prelock_process_incident": "c4c5eba21733d85b98b76c68312f1223ede3f6d5d1919ba25753037736b232f7",
    "v2_engine": "e6558d8cf9c00ec3d5f89b549902817a25afca21bff399a4766c397e80ba2774",
    "v2_engine_tests": "c310e0918933004b54921ceb5793965d508caafb65156e066e6134c26f3124d4",
    "v2_scientific": "a6168d9d438fe78dd77f87e48f4005c418c1a218e1811edbe3609a034ece850e",
    "v2_scientific_tests": "945fbfd95d74fc63a4434cb3d7cd51f50a9a319569163325007aab7acaeae7cd",
}
RUNTIME_BOUNDARY_ROLES = frozenset({
    "v2_contracts_source", "v2_runtime_source", "v2_handoff_source",
    "v2_execution_guard", "v2_runtime_tests_source", "v2_runtime_test_launcher",
    "v2_test_authority_manifest", "v2_runtime_tests_manifest",
})
DYNAMIC_BOUNDARY_ROLES = frozenset({
    "mechanism_registry_v4", "v2_prelock_boundary_registry",
    "v2_guard_identity_evidence", "v2_fold_projection_target_free",
    "v2_geometry_projection_target_free", "v2_prelock_validator_source",
    "v2_prelock_validator_tests", "v2_prelock_test_launcher",
    "v2_prelock_test_authority_manifest", "v2_prelock_tests_manifest",
})
BOUNDARY_ROLES = frozenset(FIXED_BOUNDARY_SHA256) | RUNTIME_BOUNDARY_ROLES | DYNAMIC_BOUNDARY_ROLES
STAGE_ROLES = frozenset({
    "v2_s1_contract", "v2_s1_authorization", "v2_s1_bundle_manifest",
    "v2_s1_finalization_root_record", "v2_s2_contract", "v2_s2_authorization",
    "v2_s2_bundle_manifest", "v2_s2_finalization_root_record",
})
NO_REOPEN_ROLES = frozenset({
    "train_ccr_cohort_arrays", "train_ccr_model_srgb",
    "train_ccr_valid_masks_packbits", "dataset_manifest_train_metadata",
    "cidt_clean_train_table", "keeper_checkpoint_lineage_only",
    "keeper_config_lineage_only",
})

S2A_MEMBER_REGISTRY = (
    ("train_ccr_cohort_arrays", "sample_indices"),
    ("train_ccr_cohort_arrays", "model_boxes"),
    ("train_ccr_cohort_arrays", "keeper_probabilities"),
    ("train_ccr_valid_masks_packbits", "valid_masks_packbits"),
)
S2B_MEMBER_REGISTRY = (
    ("train_ccr_cohort_arrays", "targets"),
    ("cidt_clean_train_table", "targets"),
    ("cidt_clean_train_table", "keeper_predictions"),
)
TARGET_ALIASES = frozenset({
    "target", "targets", "label", "labels", "y", "ground_truth", "class_id",
    "xai", "xai_ids", "target_strata", "target_counts",
})
S1_GATE_REGISTRY = frozenset({
    "forward_batch_1", "forward_batch_2", "forward_batch_32",
    "finite_nonzero_gradients", "finite_parameter_update",
    "onnx_opset", "onnx_checker", "onnx_ort_batch_1", "onnx_ort_batch_2",
    "onnx_ort_batch_32", "trt_plugin_free", "trt_batch_1", "trt_batch_2",
    "trt_batch_32", "batch1_latency", "batch32_throughput",
    "peak_cuda", "peak_rss",
})
ALLOWED_ONNX_OPS = frozenset({
    "Abs", "Add", "And", "Cast", "Clip", "Concat", "Constant", "Conv",
    "Div", "Equal", "Exp", "Gather", "Greater", "MatMul", "MaxPool",
    "Mul", "Not", "ReduceMax", "ReduceMean", "ReduceSum", "Relu", "Reshape",
    "Shape", "Sigmoid", "Slice", "Softmax", "Sqrt", "Sub", "Transpose",
    "Unsqueeze", "Where",
})

S1_PHASE = "S1_SYNTHETIC_ONLY_GPU"
S2_PHASE = "S2_SCORE_INDEPENDENT_DERIVATION_CPU"
ZERO_AUTHORITY = {
    "formal_runs": 0,
    "candidate_runs": 0,
    "validation_runs": 0,
    "test_runs": 0,
    "conditional_xai_runs": 0,
}
PARAMETER_COUNTS = {
    "ddf_full": 9380,
    "static_matched": 9435,
    "ddf_spatial_only": 9380,
    "ddf_channel_only": 9380,
    "ddf_full_repeat": 9380,
}
TARGET_SHARD_KEYS = tuple(
    f"fold_{fold}_{split}"
    for fold in range(FOLDS)
    for split in ("fit", "calibration", "held")
)
NOISE_SHAPE = (ROWS, 3, 256, 256)
BOOTSTRAP_SHAPE = (2000, COMPONENTS)
EPOCHS = 20
S1_SPATIAL_SHAPE = (3, 64, 64)
S1_OUTPUT_WIDTH = 4

S1_READ_ROLES = (
    "v2_s1_contract",
    "v2_s1_authorization",
    "v2_protocol_base",
    "v2_protocol_erratum",
    "v2_prelock_sequencing_erratum",
    "v2_engine",
)
S2_FIXED_READ_ROLES = (
    "v2_s2_contract",
    "v2_s2_authorization",
    "v2_s1_bundle_manifest",
    "v2_s1_finalization_root_record",
    "v2_protocol_base",
    "v2_protocol_erratum",
    "v2_fold_erratum_r2",
    "v2_fold_manifest_r2",
    "v2_prelock_sequencing_erratum",
    "v2_engine",
    "v2_scientific",
)
S1_OUTPUT_ROLES = (
    "native_reference",
    "onnx_model",
    "onnx_outputs",
    "tensorrt_plan",
    "tensorrt_outputs",
    "engineering_report",
)
S2_OUTPUT_ROLES = (
    "cohort_lock",
    "target_shards",
    "xai_lock",
    "cidt_lock",
    "random_lock",
    "parameter_lock",
    "scientific_report",
)
ALL_ARTIFACT_ROLES = BOUNDARY_ROLES | STAGE_ROLES | NO_REOPEN_ROLES
PRELOCK_REGISTRY_HASH_ROLES = BOUNDARY_ROLES - {"v2_prelock_boundary_registry"}

DOCUMENT_STATES = {
    "v2_prelock_boundary_registry": "prelock_boundary_registry_v2_finalized",
    "v2_guard_identity_evidence": "guard_identity_evidence_finalized",
    "v2_prelock_test_authority_manifest": "prelock_synthetic_test_authority_finalized",
    "v2_prelock_tests_manifest": "prelock_verifier_tests_finalized",
    "v2_s1_contract": "prospective_s1_contract_finalized",
    "v2_s1_authorization": "s1_single_use_authorization_consumed",
    "v2_s1_bundle_manifest": "s1_primary_replay_bundle_finalized",
    "v2_s1_finalization_root_record": "s1_outer_root_finalized",
    "v2_s2_contract": "prospective_s2_contract_finalized_after_s1",
    "v2_s2_authorization": "s2_single_use_authorization_consumed",
    "v2_s2_bundle_manifest": "s2_primary_replay_bundle_finalized",
    "v2_s2_finalization_root_record": "s2_outer_root_finalized",
}


class PrelockValidationError(ValueError):
    """The finalized pre-lock boundary is incomplete or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrelockValidationError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise PrelockValidationError(f"{label} must be a list")
    return value


def _exact(value: Mapping[str, object], keys: Iterable[str], label: str) -> None:
    wanted, observed = set(keys), set(value)
    if wanted != observed:
        raise PrelockValidationError(
            f"{label} keys differ: missing={sorted(wanted-observed)}, extra={sorted(observed-wanted)}"
        )


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    text = str(value)
    if pattern.fullmatch(text) is None:
        raise PrelockValidationError(f"{label} is not canonical")
    return text


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(value: object, label: str) -> Dict[str, object]:
    row = _mapping(value, label)
    _exact(row, {"path", "bytes", "sha256", "access_class"}, label)
    path = str(row["path"])
    if not os.path.isabs(path) or isinstance(row["bytes"], bool) or int(row["bytes"]) < 0:
        raise PrelockValidationError(f"{label} path/bytes differ")
    return {
        "path": os.path.abspath(path), "bytes": int(row["bytes"]),
        "sha256": _hex(row["sha256"], HEX64, f"{label}.sha256"),
        "access_class": str(row["access_class"]),
    }


def _read_record(record: Mapping[str, object], label: str) -> bytes:
    path = Path(str(record["path"]))
    try:
        absolute, resolved = os.path.abspath(str(path)), os.path.realpath(str(path))
        if os.path.normcase(absolute) != os.path.normcase(resolved):
            raise PrelockValidationError(f"{label} uses a reparse/symlink alias")
        data = path.read_bytes()
    except OSError as exc:
        raise PrelockValidationError(f"cannot read {label}") from exc
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise PrelockValidationError(f"{label} byte identity differs")
    return data


def _json_bytes(data: bytes, label: str) -> Mapping[str, object]:
    try:
        value = _mapping(json.loads(data.decode("utf-8")), label)
    except (UnicodeError, ValueError) as exc:
        raise PrelockValidationError(f"{label} is not JSON") from exc
    if data not in {_canonical_bytes(value), _canonical_bytes(value) + b"\n"}:
        raise PrelockValidationError(f"{label} is not canonical JSON")
    return value


def _read_json(record: Mapping[str, object], label: str) -> Mapping[str, object]:
    return _json_bytes(_read_record(record, label), label)


def _relative_run_out(value: object, label: str) -> str:
    text = str(value)
    if not text.startswith("$RUN_OUT/") or "\\" in text:
        raise PrelockValidationError(f"{label} is not a canonical RUN_OUT path")
    relative = text[len("$RUN_OUT/"):]
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.as_posix() != relative:
        raise PrelockValidationError(f"{label} is not a normalized RUN_OUT relative path")
    return relative


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise PrelockValidationError("NumPy is required for scientific preimages") from exc
    return np


def _array_sha(array: object) -> str:
    np = _numpy()
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _typed_array(value: object, label: str, *, dtype: str | None = None, shape: Tuple[int, ...] | None = None):
    np = _numpy()
    row = _mapping(value, label)
    _exact(row, {"dtype", "shape", "data_hex", "array_sha256"}, label)
    observed_shape = tuple(int(item) for item in _sequence(row["shape"], f"{label}.shape"))
    if not observed_shape or any(item < 0 for item in observed_shape):
        raise PrelockValidationError(f"{label} shape differs")
    try:
        observed_dtype = np.dtype(str(row["dtype"]))
        raw = bytes.fromhex(str(row["data_hex"]))
        array = np.frombuffer(raw, dtype=observed_dtype).reshape(observed_shape)
    except (TypeError, ValueError) as exc:
        raise PrelockValidationError(f"{label} typed bytes differ") from exc
    if dtype is not None and str(observed_dtype) != dtype:
        raise PrelockValidationError(f"{label} dtype differs")
    if shape is not None and observed_shape != shape:
        raise PrelockValidationError(f"{label} shape differs")
    if row["array_sha256"] != _array_sha(array):
        raise PrelockValidationError(f"{label} array SHA-256 differs")
    return array


def _utc(value: object, label: str) -> _datetime.datetime:
    try:
        parsed = _datetime.datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise PrelockValidationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PrelockValidationError(f"{label} is not timezone-aware")
    return parsed.astimezone(_datetime.timezone.utc)


def _document(
    value: Mapping[str, object], role: str, *, state: str | None = None
) -> Mapping[str, object]:
    _exact(value, {"schema_version", "protocol_id", "kind", "state", "payload"}, role)
    expected_state = DOCUMENT_STATES.get(role) if state is None else state
    if (
        int(value["schema_version"]) != SCHEMA_VERSION
        or value["protocol_id"] != PROTOCOL_ID
        or value["kind"] != role
        or value["state"] != expected_state
    ):
        raise PrelockValidationError(f"{role} document identity differs")
    return _mapping(value["payload"], f"{role}.payload")


def _zero_authority(value: object, label: str) -> None:
    row = _mapping(value, label)
    _exact(row, ZERO_AUTHORITY, label)
    if dict(row) != ZERO_AUTHORITY:
        raise PrelockValidationError(f"{label} is not zero")


def _record_identity(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        "path": str(record["path"]),
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
        "access_class": str(record["access_class"]),
    }


class _EvidenceReader:
    """Open finalized files while rejecting aliases, reparses, and hardlinks."""

    def __init__(self, records: Mapping[str, Mapping[str, object]]) -> None:
        self.records = records
        self._physical: Dict[Tuple[int, int], str] = {}
        self._paths: Dict[str, str] = {}

    def raw_record(self, record: Mapping[str, object], label: str) -> bytes:
        checked = _artifact(record, label)
        path = Path(str(checked["path"]))
        try:
            absolute = os.path.abspath(str(path))
            resolved = os.path.realpath(absolute)
            if os.path.normcase(absolute) != os.path.normcase(resolved):
                raise PrelockValidationError(f"{label} uses a reparse/symlink alias")
            stat = path.lstat()
            attributes = int(getattr(stat, "st_file_attributes", 0))
            reparse_flag = int(getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if path.is_symlink() or attributes & reparse_flag:
                raise PrelockValidationError(f"{label} is a reparse point")
            if not path.is_file() or int(getattr(stat, "st_nlink", 1)) != 1:
                raise PrelockValidationError(f"{label} is not a single-link regular file")
            path_key = os.path.normcase(resolved)
            owner = self._paths.get(path_key)
            if owner is not None and owner != label:
                raise PrelockValidationError(f"finalized artifact path alias: {owner} and {label}")
            self._paths[path_key] = label
            identity = (int(stat.st_dev), int(stat.st_ino))
            owner = self._physical.get(identity)
            if owner is not None and owner != label:
                raise PrelockValidationError(f"finalized artifact physical alias: {owner} and {label}")
            self._physical[identity] = label
            data = path.read_bytes()
        except PrelockValidationError:
            raise
        except OSError as exc:
            raise PrelockValidationError(f"cannot read {label}") from exc
        if len(data) != checked["bytes"] or hashlib.sha256(data).hexdigest() != checked["sha256"]:
            raise PrelockValidationError(f"{label} byte identity differs")
        return data

    def raw(self, role: str) -> bytes:
        if role in NO_REOPEN_ROLES:
            raise PrelockValidationError(f"{role} is provenance-only and must not be reopened")
        return self.raw_record(self.records[role], role)

    def json_record(self, record: Mapping[str, object], label: str) -> Mapping[str, object]:
        return _json_bytes(self.raw_record(record, label), label)

    def json(self, role: str) -> Mapping[str, object]:
        return _json_bytes(self.raw(role), role)

    def payload(self, role: str, *, state: str | None = None) -> Mapping[str, object]:
        return _document(self.json(role), role, state=state)


def _prepare_artifacts(
    artifacts: object,
) -> Tuple[Dict[str, Dict[str, object]], _EvidenceReader]:
    source = _mapping(artifacts, "artifacts")
    _exact(source, ALL_ARTIFACT_ROLES, "artifacts")
    records = {str(role): _artifact(record, f"artifact {role}") for role, record in source.items()}
    for role in NO_REOPEN_ROLES:
        if records[role]["access_class"] not in {
            "provenance_only_no_reopen", "lineage_only_forbidden_runtime"
        }:
            raise PrelockValidationError(f"{role} access class does not forbid reopening")
    reader = _EvidenceReader(records)
    # Opening in one deterministic pass makes path/file aliases observable
    # before any semantic conclusion is reached.
    for role in sorted(ALL_ARTIFACT_ROLES - NO_REOPEN_ROLES):
        reader.raw(role)
    return records, reader


def _forbid_aliases(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in TARGET_ALIASES:
                raise PrelockValidationError(f"{label} exposes target/XAI alias {key}")
            _forbid_aliases(item, label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _forbid_aliases(item, label)
    elif isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in TARGET_ALIASES:
            raise PrelockValidationError(f"{label} exposes target/XAI alias {value}")


def _validate_guard_identity(
    reader: _EvidenceReader, records: Mapping[str, Mapping[str, object]]
) -> None:
    payload = reader.payload("v2_guard_identity_evidence")
    _exact(
        payload,
        {
            "guard_sha256", "protected_records", "physical_identities",
            "all_paths_canonical", "reparse_points_zero", "hardlinks_zero",
            "identity_collisions_zero", "delegation_scope", "downstream_authority",
        },
        "guard identity evidence",
    )
    if payload["guard_sha256"] != records["v2_execution_guard"]["sha256"]:
        raise PrelockValidationError("guard identity evidence is not bound to the guard")
    protected = _mapping(payload["protected_records"], "guard protected records")
    identities = _mapping(payload["physical_identities"], "guard physical identities")
    _exact(protected, NO_REOPEN_ROLES, "guard protected records")
    _exact(identities, NO_REOPEN_ROLES, "guard physical identities")
    seen: set[Tuple[str, str]] = set()
    for role in sorted(NO_REOPEN_ROLES):
        if _artifact(protected[role], f"delegated {role}") != records[role]:
            raise PrelockValidationError(f"delegated identity differs for {role}")
        row = _mapping(identities[role], f"physical identity {role}")
        _exact(
            row,
            {"canonical_path", "real_path", "volume_serial", "file_id", "link_count", "reparse_tag"},
            f"physical identity {role}",
        )
        if (
            os.path.normcase(str(row["canonical_path"])) != os.path.normcase(str(records[role]["path"]))
            or os.path.normcase(str(row["canonical_path"])) != os.path.normcase(str(row["real_path"]))
            or int(row["link_count"]) != 1
            or int(row["reparse_tag"]) != 0
            or not str(row["volume_serial"])
            or not str(row["file_id"])
        ):
            raise PrelockValidationError(f"physical identity differs for {role}")
        identity = (str(row["volume_serial"]), str(row["file_id"]))
        if identity in seen:
            raise PrelockValidationError("delegated protected files share a physical identity")
        seen.add(identity)
    if (
        payload["all_paths_canonical"] is not True
        or int(payload["reparse_points_zero"]) != 0
        or int(payload["hardlinks_zero"]) != 0
        or int(payload["identity_collisions_zero"]) != 0
        or payload["delegation_scope"] != "identity_only_no_content_reopen_by_prelock_verifier"
    ):
        raise PrelockValidationError("guard physical-identity delegation failed")
    _zero_authority(payload["downstream_authority"], "guard identity authority")


def _validate_test_boundary(
    reader: _EvidenceReader, records: Mapping[str, Mapping[str, object]]
) -> None:
    authority = reader.payload("v2_prelock_test_authority_manifest")
    _exact(
        authority,
        {
            "test_source_sha256", "launcher_sha256", "allowed_nodeids",
            "fixture_scope", "precollection_tripwire_required", "forbidden_access_classes",
            "live_integration", "downstream_authority",
        },
        "prelock test authority",
    )
    nodes = [str(item) for item in _sequence(authority["allowed_nodeids"], "allowed nodeids")]
    if (
        authority["test_source_sha256"] != records["v2_prelock_validator_tests"]["sha256"]
        or authority["launcher_sha256"] != records["v2_prelock_test_launcher"]["sha256"]
        or len(nodes) < 12
        or len(nodes) != len(set(nodes))
        or any(not node.startswith("tests/test_pair_surface_ddf_v2_prelock_validator.py::") for node in nodes)
        or authority["fixture_scope"] != "synthetic_tmp_only_no_repository_cache_dataset_gpu"
        or authority["precollection_tripwire_required"] is not True
        or authority["forbidden_access_classes"] != ["cache", "raw", "validation", "test", "gpu"]
        or authority["live_integration"] != {"collection_authorized": False, "execution_authorized": False}
    ):
        raise PrelockValidationError("prelock test authority differs")
    _zero_authority(authority["downstream_authority"], "prelock test authority downstream")

    manifest = reader.payload("v2_prelock_tests_manifest")
    _exact(
        manifest,
        {
            "authority_sha256", "production_sha256", "tests_sha256", "launcher_sha256",
            "passed", "test_count", "negative_categories", "positive_staged_e2e",
            "fresh_subprocess_passed", "precollection_tripwire_passed",
            "forbidden_access_counts", "live_integration", "downstream_authority",
        },
        "prelock tests manifest",
    )
    negatives = [str(item) for item in _sequence(manifest["negative_categories"], "negative categories")]
    if (
        manifest["authority_sha256"] != records["v2_prelock_test_authority_manifest"]["sha256"]
        or manifest["production_sha256"] != records["v2_prelock_validator_source"]["sha256"]
        or manifest["tests_sha256"] != records["v2_prelock_validator_tests"]["sha256"]
        or manifest["launcher_sha256"] != records["v2_prelock_test_launcher"]["sha256"]
        or manifest["passed"] is not True
        or int(manifest["test_count"]) < 12
        or len(set(negatives)) < 11
        or manifest["positive_staged_e2e"] is not True
        or manifest["fresh_subprocess_passed"] is not True
        or manifest["precollection_tripwire_passed"] is not True
        or manifest["forbidden_access_counts"] != {"cache": 0, "raw": 0, "validation": 0, "test": 0, "gpu": 0}
        or manifest["live_integration"] != {"collected": 0, "executed": 0}
    ):
        raise PrelockValidationError("prelock verifier test gate failed")
    _zero_authority(manifest["downstream_authority"], "prelock tests downstream")


def _validate_boundary(
    reader: _EvidenceReader, records: Mapping[str, Mapping[str, object]]
) -> None:
    for role, expected in FIXED_BOUNDARY_SHA256.items():
        if records[role]["sha256"] != expected:
            raise PrelockValidationError(f"fixed boundary hash differs for {role}")
    registry = reader.payload("v2_prelock_boundary_registry")
    _exact(
        registry,
        {
            "registry_version", "foundation_commit", "incident_sha256", "hashes",
            "no_reopen_roles", "downstream_authority",
        },
        "prelock boundary registry",
    )
    hashes = _mapping(registry["hashes"], "prelock boundary hashes")
    _exact(hashes, PRELOCK_REGISTRY_HASH_ROLES, "prelock boundary hashes")
    expected_hashes = {role: records[role]["sha256"] for role in sorted(PRELOCK_REGISTRY_HASH_ROLES)}
    if (
        int(registry["registry_version"]) != 2
        or registry["foundation_commit"] != FOUNDATION_COMMIT
        or registry["incident_sha256"] != records["v2_prelock_process_incident"]["sha256"]
        or dict(hashes) != expected_hashes
        or registry["no_reopen_roles"] != sorted(NO_REOPEN_ROLES)
    ):
        raise PrelockValidationError("prelock boundary registry differs")
    _zero_authority(registry["downstream_authority"], "prelock boundary authority")
    _validate_guard_identity(reader, records)
    _validate_test_boundary(reader, records)


def _parameter_contracts(value: object, label: str) -> Dict[str, Tuple[Mapping[str, object], ...]]:
    rows = _mapping(value, label)
    _exact(rows, ROLES, label)
    result: Dict[str, Tuple[Mapping[str, object], ...]] = {}
    for family in ROLES:
        expected_numel = PARAMETER_COUNTS[family]
        items = _sequence(rows[family], f"{label}.{family}")
        if not items:
            raise PrelockValidationError(f"{label}.{family} is empty")
        parsed = []
        names: set[str] = set()
        total = 0
        for position, raw in enumerate(items):
            row = _mapping(raw, f"{label}.{family}[{position}]")
            _exact(row, {"name", "dtype", "shape", "requires_grad"}, f"{label}.{family}[{position}]")
            name = str(row["name"])
            shape = tuple(int(item) for item in _sequence(row["shape"], f"parameter shape {name}"))
            if (
                not name
                or name in names
                or not shape
                or any(item <= 0 for item in shape)
                or str(row["dtype"]) not in {"float32", "int64"}
                or not isinstance(row["requires_grad"], bool)
            ):
                raise PrelockValidationError(f"parameter contract differs for {name}")
            names.add(name)
            total += math.prod(shape)
            parsed.append({
                "name": name,
                "dtype": str(row["dtype"]),
                "shape": list(shape),
                "requires_grad": bool(row["requires_grad"]),
            })
        if total != expected_numel:
            raise PrelockValidationError(f"{family} parameter count differs")
        result[family] = tuple(parsed)
    full = {str(row["name"]): row for row in result["ddf_full"]}
    repeat = {str(row["name"]): row for row in result["ddf_full_repeat"]}
    spatial = {str(row["name"]): row for row in result["ddf_spatial_only"]}
    channel = {str(row["name"]): row for row in result["ddf_channel_only"]}
    if set(full) != set(repeat) or set(full) != set(spatial) or set(full) != set(channel):
        raise PrelockValidationError("DDF role parameter names differ")
    spatial_frozen = channel_frozen = 0
    for name, reference in full.items():
        for candidate in (repeat[name], spatial[name], channel[name]):
            if candidate["dtype"] != reference["dtype"] or candidate["shape"] != reference["shape"]:
                raise PrelockValidationError(f"DDF role parameter shape differs for {name}")
        expected_spatial = bool(reference["requires_grad"]) and not any(
            token in name for token in ("channel_reduce", "channel_expand", "channel_scale")
        )
        expected_channel = bool(reference["requires_grad"]) and "spatial_projection" not in name
        if bool(spatial[name]["requires_grad"]) != expected_spatial or bool(channel[name]["requires_grad"]) != expected_channel:
            raise PrelockValidationError(f"DDF causal-control trainability differs for {name}")
        spatial_frozen += int(bool(reference["requires_grad"]) and not expected_spatial)
        channel_frozen += int(bool(reference["requires_grad"]) and not expected_channel)
    if spatial_frozen == 0 or channel_frozen == 0:
        raise PrelockValidationError("DDF causal controls do not freeze their named mechanism")
    return result


def _validate_contract(
    reader: _EvidenceReader,
    records: Mapping[str, Mapping[str, object]],
    stage: str,
    *,
    verified_s1: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    role = f"v2_{stage}_contract"
    payload = reader.payload(role)
    common = {
        "phase", "source_revision", "canonical_output_root", "boundary_registry_sha256",
        "parameter_contracts", "resource_limits", "downstream_authority",
    }
    if stage == "s1":
        _exact(
            payload,
            common | {"allowed_input_roles", "workload", "deployment_limits"},
            "S1 contract",
        )
        if payload["phase"] != S1_PHASE or payload["allowed_input_roles"] != list(S1_READ_ROLES):
            raise PrelockValidationError("S1 phase/input contract differs")
        workload = _mapping(payload["workload"], "S1 workload")
        _exact(
            workload,
            {"dtype", "spatial_shape", "batches", "input_seed", "warmup_iterations", "measured_iterations"},
            "S1 workload",
        )
        if (
            workload["dtype"] != "float32"
            or workload["spatial_shape"] != list(S1_SPATIAL_SHAPE)
            or workload["batches"] != [1, 2, 32]
            or int(workload["input_seed"]) != PRIMARY_SEED
            or int(workload["warmup_iterations"]) <= 0
            or int(workload["measured_iterations"]) < 3
        ):
            raise PrelockValidationError("S1 workload differs")
        limits = _mapping(payload["deployment_limits"], "S1 deployment limits")
        _exact(
            limits,
            {
                "onnx_opset", "parity_max_abs", "batch1_mean_ceiling_ms",
                "batch1_p95_ceiling_ms", "batch32_min_throughput_sps",
                "peak_cuda_bytes", "peak_rss_bytes",
            },
            "S1 deployment limits",
        )
        if (
            int(limits["onnx_opset"]) < 17
            or float(limits["parity_max_abs"]) <= 0
            or min(float(limits[key]) for key in (
                "batch1_mean_ceiling_ms", "batch1_p95_ceiling_ms",
                "batch32_min_throughput_sps",
            )) <= 0
            or int(limits["peak_cuda_bytes"]) <= 0
            or int(limits["peak_rss_bytes"]) <= 0
        ):
            raise PrelockValidationError("S1 deployment limits are invalid")
    else:
        _exact(
            payload,
            common
            | {
                "verified_s1", "s2a_contract", "s2b_contract", "target_release_order",
                "cohort_contract", "random_contract", "initial_state_sha256",
                "allowed_input_roles",
            },
            "S2 contract",
        )
        if payload["phase"] != S2_PHASE or payload["allowed_input_roles"] != list(S2_FIXED_READ_ROLES):
            raise PrelockValidationError("S2 phase/input contract differs")
        if verified_s1 is None:
            raise PrelockValidationError("S2 requires verified S1")
        expected_s1 = {
            "contract_sha256": verified_s1["contract_sha256"],
            "authorization_sha256": verified_s1["authorization_sha256"],
            "bundle_sha256": verified_s1["bundle_sha256"],
            "root_record_sha256": verified_s1["root_record_sha256"],
            "outer_root_sha256": verified_s1["outer_root_sha256"],
            "finalized_utc": verified_s1["finalized_utc"],
        }
        if payload["verified_s1"] != expected_s1:
            raise PrelockValidationError("S2 does not bind the verified S1 result")
        s2a = _mapping(payload["s2a_contract"], "S2A contract")
        _exact(s2a, {"state", "member_registry", "score_independent", "target_blind"}, "S2A contract")
        _forbid_aliases(s2a, "S2A contract")
        if (
            s2a["state"] != "target_blind_archive_phase"
            or s2a["member_registry"] != [list(item) for item in S2A_MEMBER_REGISTRY]
            or s2a["score_independent"] is not True
            or s2a["target_blind"] is not True
        ):
            raise PrelockValidationError("S2A contract differs")
        s2b = _mapping(payload["s2b_contract"], "S2B contract")
        _exact(s2b, {"state", "member_registry", "release_scope"}, "S2B contract")
        if (
            s2b["state"] != "sealed_target_metadata_phase"
            or s2b["member_registry"] != [list(item) for item in S2B_MEMBER_REGISTRY]
            or s2b["release_scope"] != "exact_target_shards_cidt_and_xai_only"
            or payload["target_release_order"] != ["S2A_FINALIZED", "S2B_RELEASED"]
        ):
            raise PrelockValidationError("S2B/target transition contract differs")
        cohort = _mapping(payload["cohort_contract"], "S2 cohort contract")
        _exact(
            cohort,
            {
                "rows", "folds", "components", "fold_sizes", "target_counts",
                "identity_sha256", "xai_sample_indices", "xai_sample_indices_array_sha256",
                "cidt_identity", "target_shard_keys",
            },
            "S2 cohort contract",
        )
        if (
            int(cohort["rows"]) != ROWS
            or int(cohort["folds"]) != FOLDS
            or int(cohort["components"]) != COMPONENTS
            or cohort["fold_sizes"] != list(FOLD_SIZES)
            or {int(key): int(value) for key, value in _mapping(cohort["target_counts"], "target counts").items()} != TARGET_COUNTS
            or cohort["identity_sha256"] != COHORT_IDENTITY
            or cohort["xai_sample_indices"] != list(XAI_SAMPLE_INDICES)
            or cohort["xai_sample_indices_array_sha256"] != XAI_ARRAY_SHA256
            or cohort["cidt_identity"] != CIDT_IDENTITY
            or cohort["target_shard_keys"] != list(TARGET_SHARD_KEYS)
        ):
            raise PrelockValidationError("S2 cohort identity contract differs")
        random_contract = _mapping(payload["random_contract"], "S2 random contract")
        _exact(
            random_contract,
            {
                "generator", "primary_seed", "repeat_offset", "epochs",
                "noise_shape", "noise_array_sha256", "bootstrap_shape",
                "bootstrap_array_sha256",
            },
            "S2 random contract",
        )
        if random_contract != {
            "generator": "numpy.random.PCG64",
            "primary_seed": PRIMARY_SEED,
            "repeat_offset": REPEAT_OFFSET,
            "epochs": EPOCHS,
            "noise_shape": list(NOISE_SHAPE),
            "noise_array_sha256": NOISE_ARRAY_SHA256,
            "bootstrap_shape": list(BOOTSTRAP_SHAPE),
            "bootstrap_array_sha256": BOOTSTRAP_ARRAY_SHA256,
        }:
            raise PrelockValidationError("S2 random contract differs")
        state_hashes = _mapping(payload["initial_state_sha256"], "initial state contract")
        _exact(state_hashes, {f"{role}.fold_{fold}" for role in ROLES for fold in range(FOLDS)}, "initial state contract")
        for key, digest in state_hashes.items():
            _hex(digest, HEX64, f"initial state {key}")
    revision = _mapping(payload["source_revision"], f"{stage} source revision")
    _exact(revision, {"commit", "pushed", "committed_utc"}, f"{stage} source revision")
    _hex(revision["commit"], HEX40, f"{stage} source commit")
    _utc(revision["committed_utc"], f"{stage} source timestamp")
    if revision["pushed"] is not True:
        raise PrelockValidationError(f"{stage} source revision is not pushed")
    if stage == "s2" and verified_s1 is not None and _utc(
        revision["committed_utc"], "S2 source timestamp"
    ) <= _utc(verified_s1["finalized_utc"], "S1 finalization timestamp"):
        raise PrelockValidationError("S2 source revision does not follow finalized S1")
    if payload["canonical_output_root"] != "$RUN_OUT" or payload["boundary_registry_sha256"] != records["v2_prelock_boundary_registry"]["sha256"]:
        raise PrelockValidationError(f"{stage} canonical/boundary binding differs")
    _parameter_contracts(payload["parameter_contracts"], f"{stage} parameter contracts")
    resources = _mapping(payload["resource_limits"], f"{stage} resources")
    _exact(resources, {"wall_seconds", "peak_rss_bytes", "peak_cuda_bytes", "retained_bytes"}, f"{stage} resources")
    if min(int(resources[key]) for key in ("wall_seconds", "peak_rss_bytes", "retained_bytes")) <= 0:
        raise PrelockValidationError(f"{stage} resources differ")
    if (stage == "s1" and int(resources["peak_cuda_bytes"]) <= 0) or (stage == "s2" and int(resources["peak_cuda_bytes"]) != 0):
        raise PrelockValidationError(f"{stage} CUDA resource limit differs")
    _zero_authority(payload["downstream_authority"], f"{stage} contract authority")
    return payload


def _validate_authorization(
    reader: _EvidenceReader,
    records: Mapping[str, Mapping[str, object]],
    stage: str,
    contract: Mapping[str, object],
) -> Mapping[str, object]:
    role = f"v2_{stage}_authorization"
    payload = reader.payload(role)
    _exact(
        payload,
        {
            "phase", "contract_sha256", "source_commit", "authorization_commit",
            "source_commit_pushed", "authorization_commit_pushed", "source_is_ancestor",
            "source_committed_utc", "authorization_committed_utc", "primary_output_root",
            "replay_output_root", "quotas", "consumed", "downstream_authority",
            "authorization_id",
        },
        f"{stage} authorization",
    )
    source = _hex(payload["source_commit"], HEX40, f"{stage} authorization source commit")
    authorization = _hex(payload["authorization_commit"], HEX40, f"{stage} authorization commit")
    source_contract = _mapping(contract["source_revision"], f"{stage} source revision")
    primary = os.path.abspath(str(payload["primary_output_root"]))
    replay = os.path.abspath(str(payload["replay_output_root"]))
    try:
        overlap = os.path.commonpath([os.path.normcase(primary), os.path.normcase(replay)])
    except ValueError:
        overlap = ""
    core = dict(payload)
    observed_id = core.pop("authorization_id")
    if (
        payload["phase"] != (S1_PHASE if stage == "s1" else S2_PHASE)
        or payload["contract_sha256"] != records[f"v2_{stage}_contract"]["sha256"]
        or source != source_contract["commit"]
        or source == authorization
        or payload["source_commit_pushed"] is not True
        or payload["authorization_commit_pushed"] is not True
        or payload["source_is_ancestor"] is not True
        or _utc(payload["source_committed_utc"], f"{stage} auth source time") != _utc(source_contract["committed_utc"], f"{stage} contract source time")
        or _utc(payload["source_committed_utc"], f"{stage} auth source time") >= _utc(payload["authorization_committed_utc"], f"{stage} authorization time")
        or not os.path.isabs(str(payload["primary_output_root"]))
        or not os.path.isabs(str(payload["replay_output_root"]))
        or os.path.normcase(primary) == os.path.normcase(replay)
        or overlap in {os.path.normcase(primary), os.path.normcase(replay)}
        or payload["quotas"] != {"primary_processes": 1, "fresh_replays": 1}
        or payload["consumed"] is not True
        or observed_id != _sha(core)
    ):
        raise PrelockValidationError(f"{stage} authorization revision/order/identity differs")
    _zero_authority(payload["downstream_authority"], f"{stage} authorization downstream")
    return payload


def _claim(
    value: object,
    label: str,
    *,
    stage: str,
    process_role: str,
    contract_sha256: str,
    authorization_sha256: str,
    expected_root: str,
) -> Mapping[str, object]:
    row = _mapping(value, label)
    _exact(
        row,
        {
            "phase", "process_role", "contract_sha256", "authorization_sha256",
            "process_id", "process_start_identity", "physical_output_root",
            "started_utc", "finished_utc", "claim_id",
        },
        label,
    )
    core = dict(row)
    claim_id = core.pop("claim_id")
    if (
        row["phase"] != (S1_PHASE if stage == "s1" else S2_PHASE)
        or row["process_role"] != process_role
        or row["contract_sha256"] != contract_sha256
        or row["authorization_sha256"] != authorization_sha256
        or isinstance(row["process_id"], bool)
        or int(row["process_id"]) <= 0
        or not str(row["process_start_identity"])
        or os.path.normcase(os.path.abspath(str(row["physical_output_root"])))
        != os.path.normcase(os.path.abspath(expected_root))
        or _utc(row["started_utc"], f"{label} start") >= _utc(row["finished_utc"], f"{label} finish")
        or claim_id != _sha(core)
    ):
        raise PrelockValidationError(f"{label} differs")
    return row


def _expected_ledger(
    stage: str,
    process_role: str,
    artifacts: Mapping[str, Mapping[str, object]],
    output_paths: Mapping[str, str],
) -> list[Dict[str, object]]:
    events: list[Dict[str, object]] = [
        {
            "sequence": 0,
            "operation": "claim",
            "stage": stage.upper(),
            "role": f"v2_{stage}_authorization",
            "member": None,
            "path": "$RUN_OUT/.control/process_claim.json",
            "mode": "create_exclusive",
        }
    ]
    reads = S1_READ_ROLES if stage == "s1" else S2_FIXED_READ_ROLES
    for role in reads:
        events.append(
            {
                "sequence": len(events),
                "operation": "read_artifact",
                "stage": stage.upper(),
                "role": role,
                "member": None,
                "path": artifacts[role]["path"],
                "mode": "rb",
            }
        )
    if stage == "s2":
        for container, member in S2A_MEMBER_REGISTRY:
            events.append(
                {
                    "sequence": len(events),
                    "operation": "read_archive_member",
                    "stage": "S2A",
                    "role": container,
                    "member": member,
                    "path": f"{artifacts[container]['path']}::{member}",
                    "mode": "rb_member_exact",
                }
            )
        events.append(
            {
                "sequence": len(events),
                "operation": "target_release_transition",
                "stage": "S2A_TO_S2B",
                "role": "v2_s2_authorization",
                "member": None,
                "path": "$RUN_OUT/.control/s2a_to_s2b.json",
                "mode": "create_exclusive",
            }
        )
        for container, member in S2B_MEMBER_REGISTRY:
            events.append(
                {
                    "sequence": len(events),
                    "operation": "read_archive_member",
                    "stage": "S2B",
                    "role": container,
                    "member": member,
                    "path": f"{artifacts[container]['path']}::{member}",
                    "mode": "rb_member_exact",
                }
            )
    outputs = S1_OUTPUT_ROLES if stage == "s1" else S2_OUTPUT_ROLES
    for role in outputs:
        events.append(
            {
                "sequence": len(events),
                "operation": "write_output",
                "stage": stage.upper(),
                "role": role,
                "member": None,
                "path": f"$RUN_OUT/{output_paths[role]}",
                "mode": "create_exclusive",
            }
        )
    return events


def _validate_ledger(
    value: object,
    label: str,
    *,
    stage: str,
    process_role: str,
    artifacts: Mapping[str, Mapping[str, object]],
    output_paths: Mapping[str, str],
    claim_id: str,
) -> str:
    row = _mapping(value, label)
    _exact(
        row,
        {
            "process_role", "claim_id", "events", "event_count", "events_sha256",
            "blocked_events", "extra_reads", "extra_writes", "cache_or_lineage_reopened",
        },
        label,
    )
    expected = _expected_ledger(stage, process_role, artifacts, output_paths)
    events = list(_sequence(row["events"], f"{label}.events"))
    for position, event in enumerate(events):
        parsed = _mapping(event, f"{label}.event[{position}]")
        _exact(parsed, {"sequence", "operation", "stage", "role", "member", "path", "mode"}, f"{label}.event[{position}]")
        if int(parsed["sequence"]) != position:
            raise PrelockValidationError(f"{label} event sequence differs")
        if parsed["operation"] == "write_output":
            _relative_run_out(parsed["path"], f"{label}.event[{position}].path")
    if (
        row["process_role"] != process_role
        or row["claim_id"] != claim_id
        or events != expected
        or int(row["event_count"]) != len(expected)
        or row["events_sha256"] != _sha(expected)
        or int(row["blocked_events"]) != 0
        or int(row["extra_reads"]) != 0
        or int(row["extra_writes"]) != 0
        or int(row["cache_or_lineage_reopened"]) != 0
    ):
        raise PrelockValidationError(f"{label} exact-set ledger differs")
    # This explicit assertion guards future registry edits from silently making
    # a lineage checkpoint readable.
    if any(event["role"] in NO_REOPEN_ROLES for event in expected if event["operation"] == "read_artifact"):
        raise PrelockValidationError(f"{label} reopens a no-reopen role")
    return str(row["events_sha256"])


def _validate_output_records(
    reader: _EvidenceReader,
    value: object,
    label: str,
    *,
    stage: str,
    physical_root: str,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    rows = _mapping(value, label)
    expected_roles = S1_OUTPUT_ROLES if stage == "s1" else S2_OUTPUT_ROLES
    _exact(rows, expected_roles, label)
    result: Dict[str, Dict[str, object]] = {}
    paths: Dict[str, str] = {}
    physical_keys: set[str] = set()
    for role in expected_roles:
        item = _mapping(rows[role], f"{label}.{role}")
        _exact(item, {"canonical_path", "record"}, f"{label}.{role}")
        relative = _relative_run_out(item["canonical_path"], f"{label}.{role}.canonical_path")
        if relative in paths.values():
            raise PrelockValidationError(f"{label} has duplicate canonical paths")
        record = _artifact(item["record"], f"{label}.{role}.record")
        expected = os.path.abspath(os.path.join(physical_root, *PurePosixPath(relative).parts))
        if os.path.normcase(record["path"]) != os.path.normcase(expected):
            raise PrelockValidationError(f"{label}.{role} is outside its physical RUN_OUT root")
        key = os.path.normcase(record["path"])
        if key in physical_keys:
            raise PrelockValidationError(f"{label} has duplicate physical output paths")
        physical_keys.add(key)
        reader.raw_record(record, f"{label}.{role}.record")
        result[role] = record
        paths[role] = relative
    return result, paths


def _process_payload(
    reader: _EvidenceReader,
    record: Mapping[str, object],
    label: str,
    *,
    stage: str,
    process_role: str,
    artifacts: Mapping[str, Mapping[str, object]],
    authorization: Mapping[str, object],
) -> Mapping[str, object]:
    document = reader.json_record(record, label)
    payload = _document(
        document,
        f"v2_{stage}_{process_role}_record",
        state=f"{stage}_{process_role}_finalized",
    )
    _exact(
        payload,
        {
            "phase", "contract_sha256", "authorization_sha256", "claim", "access_ledger",
            "environment", "outputs", "evidence", "scientific_root_sha256",
        },
        label,
    )
    contract_sha = artifacts[f"v2_{stage}_contract"]["sha256"]
    authorization_sha = artifacts[f"v2_{stage}_authorization"]["sha256"]
    root = str(authorization[f"{process_role}_output_root"])
    claim = _claim(
        payload["claim"], f"{label}.claim", stage=stage, process_role=process_role,
        contract_sha256=contract_sha, authorization_sha256=authorization_sha,
        expected_root=root,
    )
    outputs, paths = _validate_output_records(
        reader, payload["outputs"], f"{label}.outputs", stage=stage, physical_root=root
    )
    ledger_sha = _validate_ledger(
        payload["access_ledger"], f"{label}.access_ledger", stage=stage,
        process_role=process_role, artifacts=artifacts, output_paths=paths,
        claim_id=str(claim["claim_id"]),
    )
    environment = _mapping(payload["environment"], f"{label}.environment")
    _exact(
        environment,
        {
            "python", "numpy", "torch", "onnx", "onnxruntime", "platform",
            "dependency_lock_sha256", "deterministic", "cuda_used",
        },
        f"{label}.environment",
    )
    if not all(str(environment[key]) for key in ("python", "numpy", "torch", "onnx", "onnxruntime", "platform")):
        raise PrelockValidationError(f"{label} environment is incomplete")
    _hex(environment["dependency_lock_sha256"], HEX64, f"{label} dependency lock")
    if environment["deterministic"] is not True or environment["cuda_used"] is not (stage == "s1"):
        raise PrelockValidationError(f"{label} environment differs")
    evidence = _mapping(payload["evidence"], f"{label}.evidence")
    identity = {
        "phase": payload["phase"],
        "contract_sha256": payload["contract_sha256"],
        "authorization_sha256": payload["authorization_sha256"],
        "claim_id": claim["claim_id"],
        "ledger_sha256": ledger_sha,
        "environment_sha256": _sha(environment),
        "outputs": {
            role: {
                "canonical_path": f"$RUN_OUT/{paths[role]}",
                "bytes": outputs[role]["bytes"],
                "sha256": outputs[role]["sha256"],
            }
            for role in sorted(outputs)
        },
        "evidence_sha256": _sha(evidence),
    }
    if (
        payload["phase"] != (S1_PHASE if stage == "s1" else S2_PHASE)
        or payload["contract_sha256"] != contract_sha
        or payload["authorization_sha256"] != authorization_sha
        or payload["scientific_root_sha256"] != _sha(identity)
    ):
        raise PrelockValidationError(f"{label} scientific root differs")
    return {
        "payload": payload,
        "claim": claim,
        "outputs": outputs,
        "paths": paths,
        "environment": environment,
        "evidence": evidence,
        "scientific_root_sha256": payload["scientific_root_sha256"],
    }


def _validate_temporal_chain(
    *,
    stage: str,
    authorization_committed_utc: object,
    primary_started_utc: object,
    primary_finished_utc: object,
    handoff_finalized_utc: object,
    replay_started_utc: object,
    replay_finished_utc: object,
    comparison_finalized_utc: object,
    bundle_finalized_utc: object,
    root_finalized_utc: object,
) -> None:
    ordered = (
        _utc(authorization_committed_utc, f"{stage} authorization commit"),
        _utc(primary_started_utc, f"{stage} primary start"),
        _utc(primary_finished_utc, f"{stage} primary finish"),
        _utc(handoff_finalized_utc, f"{stage} handoff finalization"),
        _utc(replay_started_utc, f"{stage} replay start"),
        _utc(replay_finished_utc, f"{stage} replay finish"),
        _utc(comparison_finalized_utc, f"{stage} comparison finalization"),
        _utc(bundle_finalized_utc, f"{stage} bundle finalization"),
        _utc(root_finalized_utc, f"{stage} root finalization"),
    )
    if any(left >= right for left, right in zip(ordered, ordered[1:])):
        raise PrelockValidationError(f"{stage} temporal finalization chain differs")


def _validate_chain(
    reader: _EvidenceReader,
    artifacts: Mapping[str, Mapping[str, object]],
    stage: str,
    authorization: Mapping[str, object],
) -> Mapping[str, object]:
    bundle_role = f"v2_{stage}_bundle_manifest"
    root_role = f"v2_{stage}_finalization_root_record"
    bundle = reader.payload(bundle_role)
    _exact(
        bundle,
        {
            "phase", "contract_sha256", "authorization_sha256", "primary_record",
            "primary_handoff_record", "replay_record", "comparison_record",
            "primary_scientific_root_sha256", "handoff_root_sha256",
            "replay_scientific_root_sha256", "comparison_root_sha256",
            "combined_outer_root_sha256", "finalized_utc", "downstream_authority",
        },
        f"{stage} bundle",
    )
    member_records = {
        name: _artifact(bundle[f"{name}_record"], f"{stage} {name} record")
        for name in ("primary", "primary_handoff", "replay", "comparison")
    }
    primary = _process_payload(
        reader, member_records["primary"], f"{stage} primary record", stage=stage,
        process_role="primary", artifacts=artifacts, authorization=authorization,
    )

    handoff_doc = reader.json_record(member_records["primary_handoff"], f"{stage} primary handoff")
    handoff = _document(
        handoff_doc, f"v2_{stage}_primary_handoff_record",
        state=f"{stage}_primary_only_replay_handoff_finalized",
    )
    _exact(
        handoff,
        {
            "phase", "contract_sha256", "authorization_sha256", "primary_record",
            "primary_scientific_root_sha256", "ordered_primary_output_records",
            "handoff_root_sha256", "finalized_utc",
        },
        f"{stage} primary handoff",
    )
    ordered = [
        {
            "role": role,
            "canonical_path": f"$RUN_OUT/{primary['paths'][role]}",
            "record": primary["outputs"][role],
        }
        for role in sorted(primary["outputs"])
    ]
    handoff_core = {
        "phase": handoff["phase"],
        "contract_sha256": handoff["contract_sha256"],
        "authorization_sha256": handoff["authorization_sha256"],
        "primary_record": member_records["primary"],
        "primary_scientific_root_sha256": primary["scientific_root_sha256"],
        "ordered_primary_output_records": ordered,
        "finalized_utc": handoff["finalized_utc"],
    }
    if (
        handoff_core != {key: handoff[key] for key in handoff_core}
        or handoff["handoff_root_sha256"] != _sha(handoff_core)
    ):
        raise PrelockValidationError(f"{stage} primary-only handoff differs")

    replay = _process_payload(
        reader, member_records["replay"], f"{stage} replay record", stage=stage,
        process_role="replay", artifacts=artifacts, authorization=authorization,
    )
    # The exact process schema deliberately has no optional key.  Bind replay
    # through its evidence object so a primary record cannot masquerade as replay.
    replay_evidence = replay["evidence"]
    binding = _mapping(replay_evidence.get("replay_binding"), f"{stage} replay binding")
    _exact(binding, {"handoff_record_sha256", "handoff_root_sha256", "primary_record_sha256"}, f"{stage} replay binding")
    if binding != {
        "handoff_record_sha256": member_records["primary_handoff"]["sha256"],
        "handoff_root_sha256": handoff["handoff_root_sha256"],
        "primary_record_sha256": member_records["primary"]["sha256"],
    }:
        raise PrelockValidationError(f"{stage} replay is not bound to primary handoff")

    if (
        primary["claim"]["process_id"] == replay["claim"]["process_id"]
        or primary["claim"]["process_start_identity"] == replay["claim"]["process_start_identity"]
        or os.path.normcase(str(primary["claim"]["physical_output_root"]))
        == os.path.normcase(str(replay["claim"]["physical_output_root"]))
        or primary["claim"]["claim_id"] == replay["claim"]["claim_id"]
    ):
        raise PrelockValidationError(f"{stage} primary/replay process identity reused")

    comparison_doc = reader.json_record(member_records["comparison"], f"{stage} comparison record")
    comparison = _document(
        comparison_doc, f"v2_{stage}_comparison_record",
        state=f"{stage}_primary_replay_comparison_finalized",
    )
    _exact(
        comparison,
        {
            "phase", "contract_sha256", "authorization_sha256", "primary_record_sha256",
            "handoff_record_sha256", "replay_record_sha256", "primary_scientific_root_sha256",
            "handoff_root_sha256", "replay_scientific_root_sha256", "gate_decisions",
            "comparison_root_sha256", "finalized_utc",
        },
        f"{stage} comparison",
    )
    comparison_core = {
        "phase": comparison["phase"],
        "contract_sha256": comparison["contract_sha256"],
        "authorization_sha256": comparison["authorization_sha256"],
        "primary_record_sha256": member_records["primary"]["sha256"],
        "handoff_record_sha256": member_records["primary_handoff"]["sha256"],
        "replay_record_sha256": member_records["replay"]["sha256"],
        "primary_scientific_root_sha256": primary["scientific_root_sha256"],
        "handoff_root_sha256": handoff["handoff_root_sha256"],
        "replay_scientific_root_sha256": replay["scientific_root_sha256"],
        "gate_decisions": comparison["gate_decisions"],
        "finalized_utc": comparison["finalized_utc"],
    }
    if (
        comparison_core != {key: comparison[key] for key in comparison_core}
        or comparison["comparison_root_sha256"] != _sha(comparison_core)
    ):
        raise PrelockValidationError(f"{stage} comparison binding/root differs")
    outer_core = {
        "phase": comparison["phase"],
        "contract_sha256": comparison["contract_sha256"],
        "authorization_sha256": comparison["authorization_sha256"],
        "member_records": {name: member_records[name] for name in ("primary", "primary_handoff", "replay", "comparison")},
        "inner_roots": {
            "primary": primary["scientific_root_sha256"],
            "handoff": handoff["handoff_root_sha256"],
            "replay": replay["scientific_root_sha256"],
            "comparison": comparison["comparison_root_sha256"],
        },
    }
    if (
        bundle["phase"] != (S1_PHASE if stage == "s1" else S2_PHASE)
        or bundle["contract_sha256"] != artifacts[f"v2_{stage}_contract"]["sha256"]
        or bundle["authorization_sha256"] != artifacts[f"v2_{stage}_authorization"]["sha256"]
        or bundle["primary_scientific_root_sha256"] != primary["scientific_root_sha256"]
        or bundle["handoff_root_sha256"] != handoff["handoff_root_sha256"]
        or bundle["replay_scientific_root_sha256"] != replay["scientific_root_sha256"]
        or bundle["comparison_root_sha256"] != comparison["comparison_root_sha256"]
        or bundle["combined_outer_root_sha256"] != _sha(outer_core)
    ):
        raise PrelockValidationError(f"{stage} combined outer root differs")
    _zero_authority(bundle["downstream_authority"], f"{stage} bundle downstream")

    root = reader.payload(root_role)
    _exact(
        root,
        {
            "phase", "bundle_record", "member_records", "inner_roots",
            "combined_outer_root_sha256", "finalized_utc", "downstream_authority",
        },
        f"{stage} root record",
    )
    if (
        root["phase"] != bundle["phase"]
        or _artifact(root["bundle_record"], f"{stage} root bundle record") != artifacts[bundle_role]
        or root["member_records"] != outer_core["member_records"]
        or root["inner_roots"] != outer_core["inner_roots"]
        or root["combined_outer_root_sha256"] != bundle["combined_outer_root_sha256"]
    ):
        raise PrelockValidationError(f"{stage} finalization root record differs")
    _validate_temporal_chain(
        stage=stage,
        authorization_committed_utc=authorization["authorization_committed_utc"],
        primary_started_utc=primary["claim"]["started_utc"],
        primary_finished_utc=primary["claim"]["finished_utc"],
        handoff_finalized_utc=handoff["finalized_utc"],
        replay_started_utc=replay["claim"]["started_utc"],
        replay_finished_utc=replay["claim"]["finished_utc"],
        comparison_finalized_utc=comparison["finalized_utc"],
        bundle_finalized_utc=bundle["finalized_utc"],
        root_finalized_utc=root["finalized_utc"],
    )
    _zero_authority(root["downstream_authority"], f"{stage} root downstream")
    return {
        "bundle": bundle,
        "root": root,
        "primary": primary,
        "replay": replay,
        "comparison": comparison,
        "outer_root_sha256": bundle["combined_outer_root_sha256"],
        "finalized_utc": root["finalized_utc"],
    }


def _finite(array: object, label: str) -> None:
    np = _numpy()
    if not bool(np.isfinite(array).all()):
        raise PrelockValidationError(f"{label} contains non-finite values")


def _max_abs(left: object, right: object, label: str) -> float:
    np = _numpy()
    if left.shape != right.shape:
        raise PrelockValidationError(f"{label} shapes differ")
    _finite(left, f"{label}.left")
    _finite(right, f"{label}.right")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0


def _validate_forward_cases(value: object, label: str) -> Dict[int, object]:
    cases = _mapping(value, label)
    _exact(cases, {"batch_1", "batch_2", "batch_32"}, label)
    outputs: Dict[int, object] = {}
    for batch in (1, 2, 32):
        row = _mapping(cases[f"batch_{batch}"], f"{label}.batch_{batch}")
        _exact(row, {"image", "valid64", "valid32", "valid16", "scores"}, f"{label}.batch_{batch}")
        channels, height, width = S1_SPATIAL_SHAPE
        image = _typed_array(row["image"], f"{label}.batch_{batch}.image", dtype="float32", shape=(batch, channels, height, width))
        valid64 = _typed_array(row["valid64"], f"{label}.batch_{batch}.valid64", dtype="uint8", shape=(batch, 1, height, width))
        valid32 = _typed_array(row["valid32"], f"{label}.batch_{batch}.valid32", dtype="uint8", shape=(batch, 1, height // 2, width // 2))
        valid16 = _typed_array(row["valid16"], f"{label}.batch_{batch}.valid16", dtype="uint8", shape=(batch, 1, height // 4, width // 4))
        scores = _typed_array(row["scores"], f"{label}.batch_{batch}.scores", dtype="float32", shape=(batch, S1_OUTPUT_WIDTH))
        for array_name, array in (("image", image), ("scores", scores)):
            _finite(array, f"{label}.batch_{batch}.{array_name}")
        for mask_name, mask in (("valid64", valid64), ("valid32", valid32), ("valid16", valid16)):
            if not bool(((mask == 0) | (mask == 1)).all()) or not bool((mask.sum(axis=(1, 2, 3)) > 0).all()):
                raise PrelockValidationError(f"{label}.batch_{batch}.{mask_name} differs")
        outputs[batch] = scores
    return outputs


def _validate_backend_outputs(
    value: object,
    label: str,
    *,
    reference: Mapping[int, object],
    limit: float,
) -> Dict[int, bool]:
    np = _numpy()
    rows = _mapping(value, label)
    _exact(rows, {"batch_1", "batch_2", "batch_32"}, label)
    decisions: Dict[int, bool] = {}
    for batch in (1, 2, 32):
        row = _mapping(rows[f"batch_{batch}"], f"{label}.batch_{batch}")
        _exact(row, {"scores", "max_abs", "argmax_equal"}, f"{label}.batch_{batch}")
        scores = _typed_array(row["scores"], f"{label}.batch_{batch}.scores", dtype="float32", shape=(batch, S1_OUTPUT_WIDTH))
        observed = _max_abs(reference[batch], scores, f"{label}.batch_{batch}")
        argmax_equal = bool(np.array_equal(np.argmax(reference[batch], axis=1), np.argmax(scores, axis=1)))
        if not math.isclose(float(row["max_abs"]), observed, rel_tol=0.0, abs_tol=1e-12) or row["argmax_equal"] is not argmax_equal:
            raise PrelockValidationError(f"{label}.batch_{batch} derived parity differs")
        decisions[batch] = observed <= limit and argmax_equal
    return decisions


def _percentile95(values: Sequence[float]) -> float:
    np = _numpy()
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95, method="linear"))


def _validate_s1_run(
    evidence: Mapping[str, object],
    label: str,
    *,
    contract: Mapping[str, object],
    outputs: Mapping[str, Mapping[str, object]],
    replay: bool,
) -> Tuple[Dict[str, bool], str]:
    expected_keys = {
        "engineering", "replay_binding"
    } if replay else {"engineering"}
    _exact(evidence, expected_keys, label)
    engineering = _mapping(evidence["engineering"], f"{label}.engineering")
    _exact(
        engineering,
        {
            "forward_cases", "gradient", "parameter_update", "onnx", "tensorrt",
            "timing", "resources", "exact_scientific_preimage", "declared_gates",
        },
        f"{label}.engineering",
    )
    native = _validate_forward_cases(engineering["forward_cases"], f"{label}.forward")
    gradient = _mapping(engineering["gradient"], f"{label}.gradient")
    _exact(gradient, {"parameter_name", "values"}, f"{label}.gradient")
    grad = _typed_array(gradient["values"], f"{label}.gradient.values", dtype="float32")
    _finite(grad, f"{label}.gradient")
    gradient_gate = bool(grad.size and (_numpy().abs(grad) > 0).any()) and bool(str(gradient["parameter_name"]))
    update = _mapping(engineering["parameter_update"], f"{label}.parameter_update")
    _exact(update, {"parameter_name", "before", "after"}, f"{label}.parameter_update")
    before = _typed_array(update["before"], f"{label}.update.before", dtype="float32")
    after = _typed_array(update["after"], f"{label}.update.after", dtype="float32", shape=tuple(before.shape))
    _finite(before, f"{label}.update.before")
    _finite(after, f"{label}.update.after")
    update_gate = bool(str(update["parameter_name"])) and not bool(_numpy().array_equal(before, after))

    limits = _mapping(contract["deployment_limits"], "S1 deployment limits")
    parity_limit = float(limits["parity_max_abs"])
    onnx = _mapping(engineering["onnx"], f"{label}.onnx")
    _exact(
        onnx,
        {"opset", "checker_passed", "checker_error", "operators", "outputs"},
        f"{label}.onnx",
    )
    operators = [str(item) for item in _sequence(onnx["operators"], f"{label}.onnx.operators")]
    if len(operators) != len(set(operators)) or any(op not in ALLOWED_ONNX_OPS for op in operators):
        raise PrelockValidationError(f"{label} ONNX operator set differs")
    onnx_parity = _validate_backend_outputs(
        onnx["outputs"], f"{label}.onnx.outputs", reference=native, limit=parity_limit
    )
    trt = _mapping(engineering["tensorrt"], f"{label}.tensorrt")
    _exact(
        trt,
        {"plugin_layers", "unsupported_ops", "plan_sha256", "plan_bytes", "outputs"},
        f"{label}.tensorrt",
    )
    trt_parity = _validate_backend_outputs(
        trt["outputs"], f"{label}.tensorrt.outputs", reference=native, limit=parity_limit
    )
    if (
        trt["plugin_layers"] != []
        or trt["unsupported_ops"] != []
        or trt["plan_sha256"] != outputs["tensorrt_plan"]["sha256"]
        or int(trt["plan_bytes"]) != outputs["tensorrt_plan"]["bytes"]
    ):
        raise PrelockValidationError(f"{label} TensorRT plan/plugin identity differs")

    timing = _mapping(engineering["timing"], f"{label}.timing")
    _exact(timing, {"warmup_iterations", "batch1_ms", "batch32_ms", "derived"}, f"{label}.timing")
    batch1 = [float(item) for item in _sequence(timing["batch1_ms"], f"{label}.batch1 timing")]
    batch32 = [float(item) for item in _sequence(timing["batch32_ms"], f"{label}.batch32 timing")]
    measured = int(_mapping(contract["workload"], "S1 workload")["measured_iterations"])
    if len(batch1) != measured or len(batch32) != measured or any(not math.isfinite(item) or item <= 0 for item in (*batch1, *batch32)):
        raise PrelockValidationError(f"{label} timing samples differ")
    derived = {
        "batch1_mean_ms": sum(batch1) / len(batch1),
        "batch1_p95_ms": _percentile95(batch1),
        "batch32_throughput_sps": 32.0 * 1000.0 / (sum(batch32) / len(batch32)),
    }
    declared_derived = _mapping(timing["derived"], f"{label}.timing.derived")
    _exact(declared_derived, derived, f"{label}.timing.derived")
    if any(not math.isclose(float(declared_derived[key]), value, rel_tol=0.0, abs_tol=1e-9) for key, value in derived.items()):
        raise PrelockValidationError(f"{label} derived timing differs")
    if int(timing["warmup_iterations"]) != int(_mapping(contract["workload"], "workload")["warmup_iterations"]):
        raise PrelockValidationError(f"{label} warmup differs")
    resources = _mapping(engineering["resources"], f"{label}.resources")
    _exact(resources, {"cuda_samples_bytes", "rss_samples_bytes", "derived"}, f"{label}.resources")
    cuda = [int(item) for item in _sequence(resources["cuda_samples_bytes"], f"{label}.cuda samples")]
    rss = [int(item) for item in _sequence(resources["rss_samples_bytes"], f"{label}.rss samples")]
    if not cuda or not rss or any(item < 0 for item in (*cuda, *rss)):
        raise PrelockValidationError(f"{label} resource samples differ")
    resource_derived = {"peak_cuda_bytes": max(cuda), "peak_rss_bytes": max(rss)}
    if resources["derived"] != resource_derived:
        raise PrelockValidationError(f"{label} derived resources differ")

    gates = {
        "forward_batch_1": bool(_numpy().isfinite(native[1]).all()),
        "forward_batch_2": bool(_numpy().isfinite(native[2]).all()),
        "forward_batch_32": bool(_numpy().isfinite(native[32]).all()),
        "finite_nonzero_gradients": gradient_gate,
        "finite_parameter_update": update_gate,
        "onnx_opset": int(onnx["opset"]) == int(limits["onnx_opset"]),
        "onnx_checker": onnx["checker_passed"] is True and onnx["checker_error"] == "",
        "onnx_ort_batch_1": onnx_parity[1],
        "onnx_ort_batch_2": onnx_parity[2],
        "onnx_ort_batch_32": onnx_parity[32],
        "trt_plugin_free": trt["plugin_layers"] == [] and trt["unsupported_ops"] == [],
        "trt_batch_1": trt_parity[1],
        "trt_batch_2": trt_parity[2],
        "trt_batch_32": trt_parity[32],
        "batch1_latency": derived["batch1_mean_ms"] <= float(limits["batch1_mean_ceiling_ms"]) and derived["batch1_p95_ms"] <= float(limits["batch1_p95_ceiling_ms"]),
        "batch32_throughput": derived["batch32_throughput_sps"] >= float(limits["batch32_min_throughput_sps"]),
        "peak_cuda": resource_derived["peak_cuda_bytes"] <= int(limits["peak_cuda_bytes"]),
        "peak_rss": resource_derived["peak_rss_bytes"] <= int(limits["peak_rss_bytes"]),
    }
    if set(gates) != S1_GATE_REGISTRY or engineering["declared_gates"] != gates or not all(gates.values()):
        raise PrelockValidationError(f"{label} derived S1 engineering gates failed")
    preimage = _mapping(engineering["exact_scientific_preimage"], f"{label}.scientific preimage")
    _exact(preimage, {"workload_sha256", "initial_state_sha256", "native_outputs_sha256"}, f"{label}.scientific preimage")
    native_hash = _sha({f"batch_{batch}": _array_sha(array) for batch, array in native.items()})
    if (
        preimage["workload_sha256"] != _sha(contract["workload"])
        or preimage["native_outputs_sha256"] != native_hash
    ):
        raise PrelockValidationError(f"{label} exact scientific preimage differs")
    _hex(preimage["initial_state_sha256"], HEX64, f"{label} initial state")
    return gates, _sha(preimage)


def _validate_s1_engineering(
    chain: Mapping[str, object], contract: Mapping[str, object]
) -> None:
    primary = _mapping(chain["primary"], "S1 primary")
    replay = _mapping(chain["replay"], "S1 replay")
    primary_gates, primary_preimage = _validate_s1_run(
        _mapping(primary["evidence"], "S1 primary evidence"), "S1 primary",
        contract=contract, outputs=_mapping(primary["outputs"], "S1 primary outputs"), replay=False,
    )
    replay_gates, replay_preimage = _validate_s1_run(
        _mapping(replay["evidence"], "S1 replay evidence"), "S1 replay",
        contract=contract, outputs=_mapping(replay["outputs"], "S1 replay outputs"), replay=True,
    )
    if primary_preimage != replay_preimage:
        raise PrelockValidationError("S1 primary/replay exact scientific preimages differ")
    decisions = _mapping(_mapping(chain["comparison"], "S1 comparison")["gate_decisions"], "S1 comparison gates")
    expected = {
        "primary": primary_gates,
        "replay": replay_gates,
        "exact_scientific_preimage_equal": True,
        "tensorrt_plan_equality_required": False,
    }
    if decisions != expected:
        raise PrelockValidationError("S1 comparison gates differ from recomputation")


def _validate_cohort(value: object, label: str) -> Mapping[str, object]:
    row = _mapping(value, label)
    _exact(
        row,
        {
            "sample_indices", "held_folds", "targets", "component_ids",
            "component_order", "sample_fold_pairs_sha256", "fold_mapping_sha256",
            "component_order_sha256",
        },
        label,
    )
    indices = _typed_array(row["sample_indices"], f"{label}.sample_indices", dtype="int64", shape=(ROWS,))
    folds = _typed_array(row["held_folds"], f"{label}.held_folds", dtype="int64", shape=(ROWS,))
    targets = _typed_array(row["targets"], f"{label}.targets", dtype="int64", shape=(ROWS,))
    if len(set(map(int, indices.tolist()))) != ROWS or not bool(((folds >= 0) & (folds < FOLDS)).all()):
        raise PrelockValidationError(f"{label} indices/folds differ")
    if not bool(((targets >= 0) & (targets <= 4)).all()):
        raise PrelockValidationError(f"{label} targets differ")
    if _array_sha(indices) != COHORT_IDENTITY["sample_indices_array_sha256"] or _array_sha(folds) != COHORT_IDENTITY["held_folds_array_sha256"] or _array_sha(targets) != COHORT_IDENTITY["targets_array_sha256"]:
        raise PrelockValidationError(f"{label} locked array identity differs")
    observed_fold_sizes = tuple(int((folds == fold).sum()) for fold in range(FOLDS))
    observed_targets = {target: int((targets == target).sum()) for target in range(5)}
    if observed_fold_sizes != FOLD_SIZES or observed_targets != TARGET_COUNTS:
        raise PrelockValidationError(f"{label} fold/target counts differ")
    components = [str(item) for item in _sequence(row["component_ids"], f"{label}.component_ids")]
    order = [str(item) for item in _sequence(row["component_order"], f"{label}.component_order")]
    if len(components) != ROWS or len(order) != COMPONENTS or len(set(order)) != COMPONENTS or set(components) != set(order):
        raise PrelockValidationError(f"{label} component identities differ")
    for position, digest in enumerate((*components, *order)):
        _hex(digest, HEX64, f"{label}.component[{position}]")
    component_fold: Dict[str, int] = {}
    for component, fold in zip(components, map(int, folds.tolist())):
        prior = component_fold.setdefault(component, fold)
        if prior != fold:
            raise PrelockValidationError(f"{label} component crosses held folds")
    pairs = [[int(index), int(fold)] for index, fold in zip(indices.tolist(), folds.tolist())]
    mapping = {str(int(index)): int(fold) for index, fold in zip(indices.tolist(), folds.tolist())}
    if (
        row["sample_fold_pairs_sha256"] != _sha(pairs)
        or row["sample_fold_pairs_sha256"] != COHORT_IDENTITY["sample_fold_pairs_sha256"]
        or row["fold_mapping_sha256"] != _sha(mapping)
        or row["fold_mapping_sha256"] != COHORT_IDENTITY["fold_mapping_sha256"]
        or row["component_order_sha256"] != _sha(order)
        or row["component_order_sha256"] != COHORT_IDENTITY["component_order_sha256"]
    ):
        raise PrelockValidationError(f"{label} pair/component preimage differs")
    return {"indices": indices, "folds": folds, "targets": targets, "components": components, "order": order}


def _fold_split_indices(indices: object, folds: object, held: int, split: str):
    np = _numpy()
    calibration = (held + 1) % FOLDS
    if split == "held":
        mask = folds == held
    elif split == "calibration":
        mask = folds == calibration
    else:
        mask = (folds != held) & (folds != calibration)
    return np.asarray(indices[mask], dtype=np.int64)


def _validate_target_shards(value: object, label: str, cohort: Mapping[str, object]) -> Dict[str, str]:
    rows = _mapping(value, label)
    _exact(rows, TARGET_SHARD_KEYS, label)
    indices = cohort["indices"]
    folds = cohort["folds"]
    targets = cohort["targets"]
    target_by_index = {int(index): int(target) for index, target in zip(indices.tolist(), targets.tolist())}
    digests: Dict[str, str] = {}
    for held in range(FOLDS):
        for split in ("fit", "calibration", "held"):
            key = f"fold_{held}_{split}"
            shard = _mapping(rows[key], f"{label}.{key}")
            _exact(shard, {"sample_indices", "targets", "pair_sha256"}, f"{label}.{key}")
            expected_indices = _fold_split_indices(indices, folds, held, split)
            observed_indices = _typed_array(
                shard["sample_indices"], f"{label}.{key}.sample_indices",
                dtype="int64", shape=(int(expected_indices.size),),
            )
            observed_targets = _typed_array(
                shard["targets"], f"{label}.{key}.targets",
                dtype="int64", shape=(int(expected_indices.size),),
            )
            expected_targets = _numpy().asarray([target_by_index[int(index)] for index in expected_indices.tolist()], dtype=_numpy().int64)
            if not bool(_numpy().array_equal(observed_indices, expected_indices)) or not bool(_numpy().array_equal(observed_targets, expected_targets)):
                raise PrelockValidationError(f"{label}.{key} target shard differs")
            pair_sha = _sha([[int(index), int(target)] for index, target in zip(observed_indices.tolist(), observed_targets.tolist())])
            if shard["pair_sha256"] != pair_sha:
                raise PrelockValidationError(f"{label}.{key} pair SHA differs")
            digests[key] = pair_sha
    return digests


def _validate_xai(value: object, label: str, cohort: Mapping[str, object]) -> str:
    row = _mapping(value, label)
    _exact(row, {"sample_indices", "targets", "strata_counts", "pair_sha256"}, label)
    indices = _typed_array(row["sample_indices"], f"{label}.sample_indices", dtype="int64", shape=(len(XAI_SAMPLE_INDICES),))
    targets = _typed_array(row["targets"], f"{label}.targets", dtype="int64", shape=(len(XAI_SAMPLE_INDICES),))
    expected_indices = _numpy().asarray(XAI_SAMPLE_INDICES, dtype=_numpy().int64)
    target_by_index = {int(index): int(target) for index, target in zip(cohort["indices"].tolist(), cohort["targets"].tolist())}
    try:
        expected_targets = _numpy().asarray([target_by_index[int(index)] for index in XAI_SAMPLE_INDICES], dtype=_numpy().int64)
    except KeyError as exc:
        raise PrelockValidationError(f"{label} XAI ID is outside cohort") from exc
    strata = {str(target): int((expected_targets == target).sum()) for target in sorted(set(map(int, expected_targets.tolist())))}
    pair_sha = _sha([[int(index), int(target)] for index, target in zip(indices.tolist(), targets.tolist())])
    if (
        not bool(_numpy().array_equal(indices, expected_indices))
        or _array_sha(indices) != XAI_ARRAY_SHA256
        or not bool(_numpy().array_equal(targets, expected_targets))
        or row["strata_counts"] != strata
        or row["pair_sha256"] != pair_sha
    ):
        raise PrelockValidationError(f"{label} XAI identity/target strata differ")
    return pair_sha


def _validate_cidt(value: object, label: str) -> str:
    row = _mapping(value, label)
    _exact(row, {"targets", "keeper_predictions", "pair_sha256"}, label)
    targets = _typed_array(row["targets"], f"{label}.targets", dtype="int64", shape=(CIDT_IDENTITY["rows"],))
    predictions = _typed_array(row["keeper_predictions"], f"{label}.keeper_predictions", dtype="int64", shape=(CIDT_IDENTITY["rows"],))
    if (
        _array_sha(targets) != CIDT_IDENTITY["targets_array_sha256"]
        or _array_sha(predictions) != CIDT_IDENTITY["keeper_predictions_array_sha256"]
        or not bool(((targets >= 0) & (targets <= 4)).all())
        or not bool(((predictions >= 0) & (predictions <= 4)).all())
    ):
        raise PrelockValidationError(f"{label} CIDT typed identity differs")
    pair_sha = _sha([[int(target), int(prediction)] for target, prediction in zip(targets.tolist(), predictions.tolist())])
    if row["pair_sha256"] != pair_sha:
        raise PrelockValidationError(f"{label} CIDT pair preimage differs")
    return pair_sha


def _pcg_integers(seed: int, low: int, high: int, shape: Tuple[int, ...], dtype: str):
    np = _numpy()
    return np.random.Generator(np.random.PCG64(seed)).integers(
        low, high, size=shape, dtype=np.dtype(dtype), endpoint=False
    )


def _validate_random(value: object, label: str, cohort: Mapping[str, object]) -> Mapping[str, object]:
    row = _mapping(value, label)
    _exact(row, {"noise_array_sha256", "bootstrap_array_sha256", "epoch_orders"}, label)
    noise = _pcg_integers(PRIMARY_SEED, 0, 256, NOISE_SHAPE, "uint8")
    bootstrap = _pcg_integers(PRIMARY_SEED, 0, COMPONENTS, BOOTSTRAP_SHAPE, "int64")
    noise_sha, bootstrap_sha = _array_sha(noise), _array_sha(bootstrap)
    if noise_sha != NOISE_ARRAY_SHA256 or row["noise_array_sha256"] != noise_sha:
        raise PrelockValidationError(f"{label} PCG noise preimage differs")
    if bootstrap_sha != BOOTSTRAP_ARRAY_SHA256 or row["bootstrap_array_sha256"] != bootstrap_sha:
        raise PrelockValidationError(f"{label} PCG bootstrap preimage differs")
    orders = _mapping(row["epoch_orders"], f"{label}.epoch_orders")
    expected_keys = {f"{role}.fold_{fold}" for role in ROLES for fold in range(FOLDS)}
    _exact(orders, expected_keys, f"{label}.epoch_orders")
    order_sha: Dict[str, str] = {}
    for role in ROLES:
        for fold in range(FOLDS):
            key = f"{role}.fold_{fold}"
            fit = _fold_split_indices(cohort["indices"], cohort["folds"], fold, "fit")
            seed = PRIMARY_SEED + 100 * fold + (REPEAT_OFFSET if role == "ddf_full_repeat" else 0) + 1
            generator = _numpy().random.Generator(_numpy().random.PCG64(seed))
            expected = _numpy().stack([generator.permutation(fit) for _ in range(EPOCHS)]).astype(_numpy().int64, copy=False)
            observed = _typed_array(
                orders[key], f"{label}.epoch_orders.{key}", dtype="int64",
                shape=(EPOCHS, int(fit.size)),
            )
            if not bool(_numpy().array_equal(observed, expected)):
                raise PrelockValidationError(f"{label} PCG epoch order differs for {key}")
            order_sha[key] = _array_sha(observed)
    return {"noise": noise_sha, "bootstrap": bootstrap_sha, "orders": order_sha}


def _state_tensors(
    value: object,
    label: str,
    contract: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, object], str]:
    rows = _sequence(value, label)
    if len(rows) != len(contract):
        raise PrelockValidationError(f"{label} parameter tensor count differs")
    tensors: Dict[str, object] = {}
    identities = []
    for position, (raw, spec) in enumerate(zip(rows, contract)):
        row = _mapping(raw, f"{label}[{position}]")
        _exact(row, {"name", "dtype", "shape", "requires_grad", "data_hex", "array_sha256"}, f"{label}[{position}]")
        if (
            row["name"] != spec["name"]
            or row["dtype"] != spec["dtype"]
            or row["shape"] != spec["shape"]
            or row["requires_grad"] is not spec["requires_grad"]
        ):
            raise PrelockValidationError(f"{label}[{position}] parameter metadata differs")
        typed = {key: row[key] for key in ("dtype", "shape", "data_hex", "array_sha256")}
        array = _typed_array(
            typed, f"{label}[{position}].tensor", dtype=str(spec["dtype"]),
            shape=tuple(int(item) for item in spec["shape"]),
        )
        if str(spec["dtype"]).startswith("float"):
            _finite(array, f"{label}[{position}]")
        tensors[str(spec["name"])] = array
        identities.append({
            "name": spec["name"], "dtype": spec["dtype"], "shape": spec["shape"],
            "requires_grad": spec["requires_grad"], "array_sha256": row["array_sha256"],
        })
    return tensors, _sha(identities)


def _validate_initial_states(
    value: object,
    label: str,
    *,
    contract_payload: Mapping[str, object],
) -> Dict[str, str]:
    rows = _mapping(value, label)
    keys = {f"{role}.fold_{fold}" for role in ROLES for fold in range(FOLDS)}
    _exact(rows, keys, label)
    contracts = _parameter_contracts(contract_payload["parameter_contracts"], "S2 parameter contracts")
    expected_hashes = _mapping(contract_payload["initial_state_sha256"], "S2 initial hashes")
    states: Dict[str, Dict[str, object]] = {}
    digests: Dict[str, str] = {}
    for role in ROLES:
        for fold in range(FOLDS):
            key = f"{role}.fold_{fold}"
            tensors, digest = _state_tensors(rows[key], f"{label}.{key}", contracts[role])
            if digest != expected_hashes[key]:
                raise PrelockValidationError(f"{label}.{key} state hash differs")
            states[key] = tensors
            digests[key] = digest
    np = _numpy()
    for fold in range(FOLDS):
        full = states[f"ddf_full.fold_{fold}"]
        for role in ("ddf_spatial_only", "ddf_channel_only"):
            candidate = states[f"{role}.fold_{fold}"]
            if set(candidate) != set(full) or any(not np.array_equal(full[name], candidate[name]) for name in full):
                raise PrelockValidationError(f"{label} matching DDF state differs for {role}.fold_{fold}")
        repeat = states[f"ddf_full_repeat.fold_{fold}"]
        if set(repeat) != set(full) or not any(
            not np.array_equal(full[name], repeat[name])
            for name in full if str(full[name].dtype).startswith("float")
        ):
            raise PrelockValidationError(f"{label} repeat state is not independently initialized")
        static = states[f"static_matched.fold_{fold}"]
        common = set(full) & set(static)
        if not common or any(full[name].shape != static[name].shape or not np.array_equal(full[name], static[name]) for name in common):
            raise PrelockValidationError(f"{label} DDF/static matching state differs")
    return digests


def _validate_s2_run(
    evidence: Mapping[str, object],
    label: str,
    *,
    contract: Mapping[str, object],
    replay: bool,
) -> Tuple[Dict[str, bool], str]:
    _exact(evidence, {"scientific", "replay_binding"} if replay else {"scientific"}, label)
    scientific = _mapping(evidence["scientific"], f"{label}.scientific")
    _exact(
        scientific,
        {"cohort", "target_shards", "xai", "cidt", "random", "initial_states", "declared_gates"},
        f"{label}.scientific",
    )
    cohort = _validate_cohort(scientific["cohort"], f"{label}.cohort")
    shards = _validate_target_shards(scientific["target_shards"], f"{label}.target_shards", cohort)
    xai_sha = _validate_xai(scientific["xai"], f"{label}.xai", cohort)
    cidt_sha = _validate_cidt(scientific["cidt"], f"{label}.cidt")
    random = _validate_random(scientific["random"], f"{label}.random", cohort)
    states = _validate_initial_states(scientific["initial_states"], f"{label}.initial_states", contract_payload=contract)
    gates = {
        "cohort_identity": True,
        "target_shards_exact": len(shards) == len(TARGET_SHARD_KEYS),
        "xai_exact": bool(xai_sha),
        "cidt_exact": bool(cidt_sha),
        "pcg_noise_exact": random["noise"] == NOISE_ARRAY_SHA256,
        "pcg_bootstrap_exact": random["bootstrap"] == BOOTSTRAP_ARRAY_SHA256,
        "pcg_epoch_orders_exact": len(random["orders"]) == len(ROLES) * FOLDS,
        "initial_states_exact": len(states) == len(ROLES) * FOLDS,
        "s2a_before_s2b": True,
        "no_lineage_checkpoint_read": True,
    }
    if scientific["declared_gates"] != gates or not all(gates.values()):
        raise PrelockValidationError(f"{label} derived S2 scientific gates failed")
    root = _sha({
        "cohort": {
            "indices": _array_sha(cohort["indices"]),
            "folds": _array_sha(cohort["folds"]),
            "targets": _array_sha(cohort["targets"]),
            "components": _sha(cohort["components"]),
        },
        "target_shards": shards,
        "xai": xai_sha,
        "cidt": cidt_sha,
        "random": random,
        "initial_states": states,
    })
    return gates, root


def _validate_s2_scientific(
    chain: Mapping[str, object], contract: Mapping[str, object]
) -> None:
    primary = _mapping(chain["primary"], "S2 primary")
    replay = _mapping(chain["replay"], "S2 replay")
    primary_gates, primary_root = _validate_s2_run(
        _mapping(primary["evidence"], "S2 primary evidence"), "S2 primary",
        contract=contract, replay=False,
    )
    replay_gates, replay_root = _validate_s2_run(
        _mapping(replay["evidence"], "S2 replay evidence"), "S2 replay",
        contract=contract, replay=True,
    )
    if primary_root != replay_root:
        raise PrelockValidationError("S2 primary/replay scientific preimages differ")
    decisions = _mapping(_mapping(chain["comparison"], "S2 comparison")["gate_decisions"], "S2 comparison gates")
    expected = {
        "primary": primary_gates,
        "replay": replay_gates,
        "scientific_preimage_equal": True,
    }
    if decisions != expected:
        raise PrelockValidationError("S2 comparison gates differ from recomputation")


def _s1_receipt(
    artifacts: Mapping[str, Mapping[str, object]], chain: Mapping[str, object]
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "s1_finalized_verified_no_execution_authority",
        "contract_sha256": artifacts["v2_s1_contract"]["sha256"],
        "authorization_sha256": artifacts["v2_s1_authorization"]["sha256"],
        "bundle_sha256": artifacts["v2_s1_bundle_manifest"]["sha256"],
        "root_record_sha256": artifacts["v2_s1_finalization_root_record"]["sha256"],
        "outer_root_sha256": chain["outer_root_sha256"],
        "finalized_utc": chain["finalized_utc"],
        "formal_runs_authorized": 0,
        "candidate_runs_authorized": 0,
        "validation_runs_authorized": 0,
        "test_runs_authorized": 0,
        "conditional_xai_runs_authorized": 0,
        "cache_or_lineage_reopened": 0,
    }


def _s2_receipt(
    artifacts: Mapping[str, Mapping[str, object]], chain: Mapping[str, object]
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "s2_finalized_verified_after_s1_no_execution_authority",
        "contract_sha256": artifacts["v2_s2_contract"]["sha256"],
        "authorization_sha256": artifacts["v2_s2_authorization"]["sha256"],
        "bundle_sha256": artifacts["v2_s2_bundle_manifest"]["sha256"],
        "root_record_sha256": artifacts["v2_s2_finalization_root_record"]["sha256"],
        "outer_root_sha256": chain["outer_root_sha256"],
        "finalized_utc": chain["finalized_utc"],
        "formal_runs_authorized": 0,
        "candidate_runs_authorized": 0,
        "validation_runs_authorized": 0,
        "test_runs_authorized": 0,
        "conditional_xai_runs_authorized": 0,
        "cache_or_lineage_reopened": 0,
    }


def _verify_s1_details(
    artifacts: Mapping[str, Mapping[str, object]], reader: _EvidenceReader
) -> Tuple[Dict[str, object], Mapping[str, object]]:
    contract = _validate_contract(reader, artifacts, "s1")
    authorization = _validate_authorization(reader, artifacts, "s1", contract)
    chain = _validate_chain(reader, artifacts, "s1", authorization)
    _validate_s1_engineering(chain, contract)
    return _s1_receipt(artifacts, chain), chain


def _verify_s1_prepared(
    artifacts: Mapping[str, Mapping[str, object]], reader: _EvidenceReader
) -> Dict[str, object]:
    receipt, _ = _verify_s1_details(artifacts, reader)
    return receipt


def _validate_verified_s1(value: object) -> Mapping[str, object]:
    row = _mapping(value, "verified_s1")
    _exact(
        row,
        {
            "schema_version", "state", "contract_sha256", "authorization_sha256",
            "bundle_sha256", "root_record_sha256", "outer_root_sha256", "finalized_utc",
            "formal_runs_authorized", "candidate_runs_authorized", "validation_runs_authorized",
            "test_runs_authorized", "conditional_xai_runs_authorized", "cache_or_lineage_reopened",
        },
        "verified_s1",
    )
    for key in (
        "contract_sha256", "authorization_sha256", "bundle_sha256",
        "root_record_sha256", "outer_root_sha256",
    ):
        _hex(row[key], HEX64, f"verified_s1.{key}")
    _utc(row["finalized_utc"], "verified_s1.finalized_utc")
    if (
        int(row["schema_version"]) != SCHEMA_VERSION
        or row["state"] != "s1_finalized_verified_no_execution_authority"
        or any(int(row[key]) != 0 for key in (
            "formal_runs_authorized", "candidate_runs_authorized", "validation_runs_authorized",
            "test_runs_authorized", "conditional_xai_runs_authorized", "cache_or_lineage_reopened",
        ))
    ):
        raise PrelockValidationError("verified_s1 receipt differs")
    return row


def _verify_s2_prepared(
    artifacts: Mapping[str, Mapping[str, object]],
    reader: _EvidenceReader,
    verified_s1: object,
    s1_chain: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    verified = _validate_verified_s1(verified_s1)
    expected = {
        "contract_sha256": artifacts["v2_s1_contract"]["sha256"],
        "authorization_sha256": artifacts["v2_s1_authorization"]["sha256"],
        "bundle_sha256": artifacts["v2_s1_bundle_manifest"]["sha256"],
        "root_record_sha256": artifacts["v2_s1_finalization_root_record"]["sha256"],
    }
    if any(verified[key] != digest for key, digest in expected.items()):
        raise PrelockValidationError("verified_s1 artifact binding differs")
    contract = _validate_contract(reader, artifacts, "s2", verified_s1=verified)
    authorization = _validate_authorization(reader, artifacts, "s2", contract)
    chain = _validate_chain(reader, artifacts, "s2", authorization)
    _validate_s2_scientific(chain, contract)
    if s1_chain is not None:
        _validate_cross_phase(s1_chain, chain)
    return _s2_receipt(artifacts, chain)


def _validate_cross_phase(
    s1_chain: Mapping[str, object], s2_chain: Mapping[str, object]
) -> None:
    s1_claims = [
        _mapping(_mapping(s1_chain[role], f"S1 {role}")["claim"], f"S1 {role} claim")
        for role in ("primary", "replay")
    ]
    s2_claims = [
        _mapping(_mapping(s2_chain[role], f"S2 {role}")["claim"], f"S2 {role} claim")
        for role in ("primary", "replay")
    ]
    for field, normalize in (
        ("process_id", lambda value: str(int(value))),
        ("process_start_identity", str),
        ("physical_output_root", lambda value: os.path.normcase(os.path.abspath(str(value)))),
        ("claim_id", str),
    ):
        s1_values = {normalize(claim[field]) for claim in s1_claims}
        s2_values = {normalize(claim[field]) for claim in s2_claims}
        if s1_values & s2_values:
            raise PrelockValidationError(f"cross-phase {field} reused")
    s1_roots = {
        str(s1_chain["outer_root_sha256"]),
        *[str(_mapping(s1_chain[role], f"S1 {role}")["scientific_root_sha256"]) for role in ("primary", "replay")],
    }
    s2_roots = {
        str(s2_chain["outer_root_sha256"]),
        *[str(_mapping(s2_chain[role], f"S2 {role}")["scientific_root_sha256"]) for role in ("primary", "replay")],
    }
    if s1_roots & s2_roots:
        raise PrelockValidationError("cross-phase scientific/final root reused")


def verify_finalized_s1_evidence(*, artifacts: object) -> Dict[str, object]:
    """Verify finalized S1 without issuing or implying any execution authority."""
    records, reader = _prepare_artifacts(artifacts)
    _validate_boundary(reader, records)
    return _verify_s1_prepared(records, reader)


def verify_finalized_s2_evidence(
    *, artifacts: object, verified_s1: object
) -> Dict[str, object]:
    """Verify finalized S2 only after an exact finalized-S1 receipt is supplied."""
    records, reader = _prepare_artifacts(artifacts)
    _validate_boundary(reader, records)
    # Do not trust a caller-created receipt: recompute S1 from the same bytes.
    recomputed_s1, s1_chain = _verify_s1_details(records, reader)
    if verified_s1 != recomputed_s1:
        raise PrelockValidationError("supplied verified_s1 differs from recomputation")
    return _verify_s2_prepared(records, reader, recomputed_s1, s1_chain)


def verify_finalized_prelock_evidence(*, artifacts: object) -> Dict[str, object]:
    """Guard-compatible v2 entry point; returns the exact no-authority receipt."""
    records, reader = _prepare_artifacts(artifacts)
    _validate_boundary(reader, records)
    verified_s1, s1_chain = _verify_s1_details(records, reader)
    _verify_s2_prepared(records, reader, verified_s1, s1_chain)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "s1_s2_finalized_verified_no_execution_authority",
        "sequencing_erratum_sha256": records["v2_prelock_sequencing_erratum"]["sha256"],
        "verifier_source_sha256": records["v2_prelock_validator_source"]["sha256"],
        "verifier_tests_sha256": records["v2_prelock_validator_tests"]["sha256"],
        "s1_bundle_sha256": records["v2_s1_bundle_manifest"]["sha256"],
        "s1_root_sha256": records["v2_s1_finalization_root_record"]["sha256"],
        "s2_bundle_sha256": records["v2_s2_bundle_manifest"]["sha256"],
        "s2_root_sha256": records["v2_s2_finalization_root_record"]["sha256"],
        "formal_runs_authorized": 0,
        "candidate_runs_authorized": 0,
        "validation_runs_authorized": 0,
        "test_runs_authorized": 0,
        "conditional_xai_runs_authorized": 0,
        "cache_or_lineage_reopened": 0,
    }


__all__ = [
    "PRELOCK_VERIFIER_API_VERSION",
    "PrelockValidationError",
    "verify_finalized_s1_evidence",
    "verify_finalized_s2_evidence",
    "verify_finalized_prelock_evidence",
]
