"""Pure, fail-closed serialization schema for pair-surface DDF v2 pre-lock v3.

This module performs no I/O and imports no project, data, array, training, or
runtime package.  It freezes record/envelope shapes only; scientific values are
still recomputed by a later verifier from the referenced canonical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping, Sequence


DOCUMENT_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 3
PROTOCOL_ID = "trkh_pair_surface_ddf_a0_v2"
FOUNDATION_COMMIT = "9ea25ea87ae6d497b80dd86c27e59391d21c4a9b"
BOUNDARY_REVIEW_COMMIT = "c8fd6ca080effcfd78bae8fb9fc389056bd27ad9"
INVALIDATED_MANIFEST_SHA256 = (
    "f36f95e2654a8bf8708f418716a99b6156947864632427176a00baa2d9e4c6af"
)
INVALIDATION_DOCUMENT_SHA256 = (
    "5b52066708f419ab5ce728f64328585829ab5eaf84954da74c1909e2b58bd558"
)
INVALIDATION_REASON = "incomplete_transitive_import_closure"
ROWS = 763
COMPONENTS = 158
BOOTSTRAP_REPLICATES = 2000
FOLDS = 5
ROLES = 5
EPOCHS = 20
TASKS = 4
RIVALS = 3
GATE14_RECORDS = 69
CALIBRATORS = 250
CIDT_ROWS = 9215
S1_BATCH = 32
BATCH1_TIMING_SAMPLES = 300
BATCH32_TIMING_SAMPLES = 200
ROLE_NAMES = (
    "ddf_full", "static_matched", "ddf_spatial_only", "ddf_channel_only",
    "ddf_full_repeat",
)
TASK_NAMES = ("union", "class1_vs_0", "class1_vs_2", "class1_vs_4")
PARTITIONS = ("fit", "calibration", "held")
FOLD_ROW_COUNTS = (153, 153, 153, 152, 152)
OUTER_TO_CALIBRATION_FOLD = (1, 2, 3, 4, 0)
FIT_ROW_COUNTS = (457, 457, 458, 459, 458)
CALIBRATOR_NAMES = (
    "K", "KU", "KUH", "KUP", "K_0", "KP_0", "K_2", "KP_2", "K_4", "KP_4",
)
CALIBRATOR_FEATURE_NAMES = MappingProxyType({
    "K": ("keeper_margin",),
    "KU": ("keeper_margin", "raw_union"),
    "KUH": ("keeper_margin", "raw_union", "rival_is_2", "rival_is_4"),
    "KUP": ("keeper_margin", "raw_union", "rival_is_2", "rival_is_4",
            "selected_pair_0", "selected_pair_2", "selected_pair_4"),
    "K_0": ("keeper_margin_1v0",), "KP_0": ("keeper_margin_1v0", "raw_pair_1v0"),
    "K_2": ("keeper_margin_1v2",), "KP_2": ("keeper_margin_1v2", "raw_pair_1v2"),
    "K_4": ("keeper_margin_1v4",), "KP_4": ("keeper_margin_1v4", "raw_pair_1v4"),
})
LOCKED_NONWRAP_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)
WEIGHT_KINDS = (
    "classification_positive", "classification_negative",
    "bbox_positive", "bbox_negative",
)
GATE14_REGISTRY_SHA256 = "689730f64f918e3b482c1f2d565cf71be4e03a89cafafc09626c08008cbe2dd6"
EXACT_CIDT_ARRAY_SHA256 = MappingProxyType({
    "cidt_sample_indices": "44a4e9101166892870890c4c24e54a1a240b70f9706ddd12603d85dda68d5bc7",
    "cidt_targets": "0b54f448d890b9032dcc336f5a1798927c19bf6b35b2a094bf318208e4785e7d",
    "cidt_baseline_predictions": "71295e8183f3b705992c16210da69470531b546108b65b65987e2fb72313b2d5",
})
EXACT_A0_ARRAY_SHA256 = MappingProxyType({
    "sample_indices": "ad51a9bdbf6acc7449ad8d8f65b3dc69971c318d7f64e2effd814bac8f77fe05",
    "targets": "15c43ecc7335c7a7febc4e0fbf622df7ad593fc52e5d9f14e5cc27acc16fc000",
    "held_folds": "924a3e98e5ef1bd814511675951d83ee79ff9788894762b4688fb93fabbea255",
})

FIXED_PROTOCOL_AND_INCIDENT_SHA256 = MappingProxyType({
    "v2_protocol_base": "b5fa5e2ada7e078a08276902f5897a0358d21ae03658c0d398402c4277c0b679",
    "v2_protocol_erratum": "0da08a193ca50f775047e28fcca763f8aafcc2391d54934480a3574ba1c89aeb",
    "v2_fold_erratum_r2": "1e9c5e64b259b909c0d69591b7ea5dd42cfc72f7c8db1489f89de29d3f9206c7",
    "v2_fold_manifest_r2": "81a404bedc25dc7b3e7dc3dbb8d79b4ca5e97d74374083c79e0686ececb7fb2d",
    "mechanism_registry_v3_lineage": "49b1d3422fb0d592e56c7814289fde8fe058986cc7be671357bb95d4b297566d",
    "v2_prelock_sequencing_erratum": "b986835a2ac7eff3ac98cd68b19331c6de3c828268d9539c8f46ff57afa4d1a0",
    "v2_prelock_process_incident": "c4c5eba21733d85b98b76c68312f1223ede3f6d5d1919ba25753037736b232f7",
    "v2_prelock_process_incident_independent_review_erratum":
        "b7ef5a3847b125273d7771ec35dc935ba52a570279db18f3d3ae928a95b9b96c",
    "v2_runtime_guard_test_attestation_invalidation": INVALIDATION_DOCUMENT_SHA256,
    "v2_engine": "e6558d8cf9c00ec3d5f89b549902817a25afca21bff399a4766c397e80ba2774",
    "v2_engine_tests": "c310e0918933004b54921ceb5793965d508caafb65156e066e6134c26f3124d4",
    "v2_scientific": "a6168d9d438fe78dd77f87e48f4005c418c1a218e1811edbe3609a034ece850e",
    "v2_scientific_tests": "945fbfd95d74fc63a4434cb3d7cd51f50a9a319569163325007aab7acaeae7cd",
})

OUTER_FIELDS = ("schema_version", "protocol_id", "kind", "state", "payload")
ARTIFACT_FIELDS = (
    "role", "logical_path", "media_type", "access_class", "bytes", "sha256",
)
ARRAY_FIELDS = ARTIFACT_FIELDS + ("dtype", "shape", "array_bytes", "array_sha256")
ARCHIVE_MEMBER_EVENT_FIELDS = (
    "event_index", "archive_role", "archive_expected_bytes",
    "archive_expected_sha256", "member_name", "archive_directory_index",
    "compression_method", "compressed_bytes", "uncompressed_bytes",
    "uncompressed_sha256", "dtype", "shape", "array_sha256", "stage",
    "outer_fold", "target_bearing", "transition_sha256", "read_count",
    "snapshot_artifact_role", "snapshot_array_sha256",
)
WRITE_RECEIPT_FIELDS = (
    "sequence", "role", "logical_path", "bytes", "sha256",
    "creator_claim_sha256", "process_identity_sha256", "physical_identity",
    "media_type",
)

STAGE_ROLE_STATES = MappingProxyType({
    "v2_s1_contract": "s1_contract_v3_frozen",
    "v2_s1_authorization": "s1_authorization_v3_pending_unconsumed",
    "v2_s1_bundle_manifest": "s1_primary_replay_bundle_v3_finalized",
    "v2_s1_finalization_root_record": "s1_outer_root_finalized",
    "v2_s2_contract": "s2_contract_v3_frozen",
    "v2_s2_authorization": "s2_authorization_v3_pending_unconsumed",
    "v2_s2_bundle_manifest": "s2_primary_replay_bundle_v3_finalized",
    "v2_s2_finalization_root_record": "s2_outer_root_finalized",
})

S1_GROUP_ROLES = (
    "s1_environment_preimage", "s1_synthetic_inputs",
    "s1_ddf_fp64_equation_preimage", "s1_validity_pool_softmax_preimage",
    "s1_masked_bn_preimage", "s1_gradient_update_preimage",
    "s1_onnx_model_and_node_registry", "s1_ort_reference_outputs",
    "s1_trt_build_preimage", "s1_trt_plan_identity_and_outputs",
    "s1_raw_timing_vectors", "s1_raw_cuda_allocation_trace",
)
S2A_GROUP_ROLES = (
    "s2a_target_free_source_snapshot", "s2a_validity_bbox_preimages",
    "s2a_keeper_rival_and_rank_preimages", "s2a_donor_assignment_preimages",
    "s2a_nonwrap_offset_preimages", "s2a_seed_epoch_order_preimages",
    "s2a_parameter_schema_and_init_identities", "s2a_bootstrap_draws",
    "s2a_invalid_fill_noise", "s2a_formula_and_formal_schema",
)
S2B_GROUP_ROLES = (
    "s2b_target_source_snapshot", "s2b_task_and_bbox_weight_preimages",
    "s2b_fold_target_shards", "s2b_cidt_projection", "s2b_deferred_xai_ids",
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_ROLE = re.compile(r"[a-z][a-z0-9_]*")
_DTYPE = re.compile(r"(?:bool|uint8|int64|float32|float64|<U64)")
_DTYPE_BYTES = MappingProxyType({
    "bool": 1, "uint8": 1, "int64": 8, "float32": 4, "float64": 8, "<U64": 256,
})
_SENTINELS = {"0" * 64, "f" * 64,
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}


class PrelockSchemaV3Error(ValueError):
    """Raised when a value is not the one canonical v3 schema shape."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrelockSchemaV3Error(f"{label} must be a mapping")
    if not all(type(key) is str for key in value):
        raise PrelockSchemaV3Error(f"{label} keys must be strings")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise PrelockSchemaV3Error(f"{label} must be a sequence")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _exact(row: Mapping[str, object], fields: Sequence[str], label: str) -> None:
    if set(row) != set(fields):
        missing = sorted(set(fields) - set(row))
        extra = sorted(set(row) - set(fields))
        raise PrelockSchemaV3Error(f"{label} keys differ: missing={missing}, extra={extra}")


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PrelockSchemaV3Error(f"{label} must be an integer >= {minimum}")
    return value


def _exact_integer(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise PrelockSchemaV3Error(f"{label} must be exact integer {expected}")
    return value


def _digest(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise PrelockSchemaV3Error(f"{label} must be a lowercase SHA-256")
    if not allow_empty and value in _SENTINELS:
        raise PrelockSchemaV3Error(f"{label} uses a forbidden sentinel")
    if value == INVALIDATED_MANIFEST_SHA256:
        raise PrelockSchemaV3Error(f"{label} uses the invalidated f36 manifest")
    return value


def _commit(value: object, label: str) -> str:
    if (type(value) is not str or _HEX40.fullmatch(value) is None
            or value in {"0" * 40, "f" * 40}):
        raise PrelockSchemaV3Error(f"{label} must be a non-sentinel Git SHA-1")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PrelockSchemaV3Error(f"{label} must be nonempty canonical text")
    return value


def _relative(value: object, label: str) -> str:
    text = _text(value, label).replace("\\", "/")
    if text.startswith(("/", "~")) or text != str(value) or "//" in text:
        raise PrelockSchemaV3Error(f"{label} must be a canonical relative path")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise PrelockSchemaV3Error(f"{label} contains a path alias")
    return text


def _logical_path(value: object, label: str) -> str:
    text = _text(value, label)
    if text.startswith("$RUN_OUT/"):
        _relative(text[len("$RUN_OUT/"):], label)
        return text
    if re.fullmatch(r"[A-Za-z]:\\[^\r\n]+", text) is None:
        raise PrelockSchemaV3Error(f"{label} is neither $RUN_OUT nor absolute Windows identity")
    if any(part in {"", ".", ".."} for part in text[3:].split("\\")):
        raise PrelockSchemaV3Error(f"{label} contains an absolute path alias")
    return text


def _shape(value: object, label: str) -> tuple[int, ...]:
    parts = _sequence(value, label)
    if not parts:
        raise PrelockSchemaV3Error(f"{label} cannot be scalar/empty")
    return tuple(_integer(item, f"{label}[{index}]", 1) for index, item in enumerate(parts))


@dataclass(frozen=True)
class ArtifactRecordV3:
    role: str
    logical_path: str
    media_type: str
    access_class: str
    bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str = "artifact") -> "ArtifactRecordV3":
        row = _mapping(value, label)
        _exact(row, ARTIFACT_FIELDS, label)
        role = _text(row["role"], f"{label}.role")
        if _ROLE.fullmatch(role) is None:
            raise PrelockSchemaV3Error(f"{label}.role differs")
        media = _text(row["media_type"], f"{label}.media_type")
        if media not in {
            "application/json", "application/octet-stream", "application/x-npy",
            "text/markdown", "text/plain", "text/x-python",
        }:
            raise PrelockSchemaV3Error(f"{label}.media_type differs")
        return cls(role, _logical_path(row["logical_path"], f"{label}.logical_path"), media,
                   _text(row["access_class"], f"{label}.access_class"),
                   _integer(row["bytes"], f"{label}.bytes", 1),
                   _digest(row["sha256"], f"{label}.sha256"))

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in ARTIFACT_FIELDS}


@dataclass(frozen=True)
class ArrayRecordV3(ArtifactRecordV3):
    dtype: str
    shape: tuple[int, ...]
    array_bytes: int
    array_sha256: str

    @classmethod
    def parse(cls, value: object, label: str = "array") -> "ArrayRecordV3":
        row = _mapping(value, label)
        _exact(row, ARRAY_FIELDS, label)
        base = ArtifactRecordV3.parse({field: row[field] for field in ARTIFACT_FIELDS}, label)
        dtype = _text(row["dtype"], f"{label}.dtype")
        if _DTYPE.fullmatch(dtype) is None:
            raise PrelockSchemaV3Error(f"{label}.dtype is not a frozen ASCII NumPy dtype")
        if base.media_type != "application/x-npy":
            raise PrelockSchemaV3Error(f"{label}.media_type is not canonical NPY")
        shape = _shape(row["shape"], f"{label}.shape")
        expected_bytes = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        array_bytes = _integer(row["array_bytes"], f"{label}.array_bytes", 1)
        if array_bytes != expected_bytes:
            raise PrelockSchemaV3Error(f"{label}.array_bytes differs from dtype/shape")
        if base.bytes <= array_bytes:
            raise PrelockSchemaV3Error(f"{label}.bytes cannot contain a canonical NPY header")
        return cls(*tuple(getattr(base, field) for field in ARTIFACT_FIELDS), dtype,
                   shape, array_bytes,
                   _digest(row["array_sha256"], f"{label}.array_sha256"))

    def as_dict(self) -> dict[str, object]:
        row = ArtifactRecordV3.as_dict(self)
        row.update(dtype=self.dtype, shape=list(self.shape), array_bytes=self.array_bytes,
                   array_sha256=self.array_sha256)
        return row


@dataclass(frozen=True)
class ArchiveMemberEventV3:
    event_index: int
    archive_role: str
    archive_expected_bytes: int
    archive_expected_sha256: str
    member_name: str
    archive_directory_index: int
    compression_method: int
    compressed_bytes: int
    uncompressed_bytes: int
    uncompressed_sha256: str
    dtype: str
    shape: tuple[int, ...]
    array_sha256: str
    stage: str
    outer_fold: int | None
    target_bearing: bool
    transition_sha256: str | None
    read_count: int
    snapshot_artifact_role: str
    snapshot_array_sha256: str

    @classmethod
    def parse(cls, value: object, label: str = "archive_event") -> "ArchiveMemberEventV3":
        row = _mapping(value, label)
        _exact(row, ARCHIVE_MEMBER_EVENT_FIELDS, label)
        fold = row["outer_fold"]
        if fold is not None and (type(fold) is not int or fold not in range(FOLDS)):
            raise PrelockSchemaV3Error(f"{label}.outer_fold differs")
        target = row["target_bearing"]
        if type(target) is not bool:
            raise PrelockSchemaV3Error(f"{label}.target_bearing differs")
        transition = row["transition_sha256"]
        if (target and transition is None) or (not target and transition is not None):
            raise PrelockSchemaV3Error(f"{label}.transition target binding differs")
        transition_digest = None if transition is None else _digest(transition, f"{label}.transition_sha256")
        dtype = _text(row["dtype"], f"{label}.dtype")
        if _DTYPE.fullmatch(dtype) is None:
            raise PrelockSchemaV3Error(f"{label}.dtype differs")
        shape = _shape(row["shape"], f"{label}.shape")
        compression_method = _integer(row["compression_method"], f"{label}.compression_method")
        uncompressed_bytes = _integer(
            row["uncompressed_bytes"], f"{label}.uncompressed_bytes", 1
        )
        raw_array_bytes = _DTYPE_BYTES[dtype]
        for dimension in shape:
            raw_array_bytes *= dimension
        if uncompressed_bytes <= raw_array_bytes:
            raise PrelockSchemaV3Error(f"{label}.uncompressed_bytes cannot contain canonical NPY")
        if compression_method not in {0, 8}:
            raise PrelockSchemaV3Error(f"{label}.compression_method differs")
        array_digest = _digest(row["array_sha256"], f"{label}.array_sha256")
        snapshot_digest = _digest(row["snapshot_array_sha256"], f"{label}.snapshot_array_sha256")
        if snapshot_digest != array_digest:
            raise PrelockSchemaV3Error(f"{label} snapshot array identity differs")
        _exact_integer(row["read_count"], 1, f"{label}.read_count")
        if row["stage"] not in {"s2a", "s2b"}:
            raise PrelockSchemaV3Error(f"{label} stage/read_count differs")
        if row["stage"] == "s2a" and target:
            raise PrelockSchemaV3Error(f"{label} releases target-bearing content in S2A")
        if row["stage"] == "s2b" and not target:
            raise PrelockSchemaV3Error(f"{label} S2B event is not target-bearing")
        archive_role = _text(row["archive_role"], f"{label}.archive_role")
        snapshot_role = _text(
            row["snapshot_artifact_role"], f"{label}.snapshot_artifact_role"
        )
        if _ROLE.fullmatch(archive_role) is None or _ROLE.fullmatch(snapshot_role) is None:
            raise PrelockSchemaV3Error(f"{label} archive/snapshot role differs")
        return cls(_integer(row["event_index"], f"{label}.event_index"), archive_role,
                   _integer(row["archive_expected_bytes"], f"{label}.archive_expected_bytes", 1),
                   _digest(row["archive_expected_sha256"], f"{label}.archive_expected_sha256"),
                   _relative(row["member_name"], f"{label}.member_name"),
                   _integer(row["archive_directory_index"], f"{label}.archive_directory_index"),
                   compression_method,
                   _integer(row["compressed_bytes"], f"{label}.compressed_bytes", 1),
                   uncompressed_bytes,
                   _digest(row["uncompressed_sha256"], f"{label}.uncompressed_sha256"),
                   dtype, shape, array_digest,
                   str(row["stage"]), fold, target, transition_digest, 1,
                   snapshot_role,
                   snapshot_digest)


@dataclass(frozen=True)
class WriteReceiptV3:
    sequence: int
    role: str
    logical_path: str
    bytes: int
    sha256: str
    creator_claim_sha256: str
    process_identity_sha256: str
    physical_identity: str
    media_type: str

    @classmethod
    def parse(cls, value: object, label: str = "write_receipt") -> "WriteReceiptV3":
        row = _mapping(value, label)
        _exact(row, WRITE_RECEIPT_FIELDS, label)
        role = _text(row["role"], f"{label}.role")
        path = _logical_path(row["logical_path"], f"{label}.logical_path")
        media = _text(row["media_type"], f"{label}.media_type")
        if _ROLE.fullmatch(role) is None or not path.startswith("$RUN_OUT/"):
            raise PrelockSchemaV3Error(f"{label} role/path differs")
        if media not in {"application/json", "application/octet-stream", "application/x-npy"}:
            raise PrelockSchemaV3Error(f"{label}.media_type differs")
        return cls(_integer(row["sequence"], f"{label}.sequence"), role, path,
                   _integer(row["bytes"], f"{label}.bytes", 1),
                   _digest(row["sha256"], f"{label}.sha256"),
                   _digest(row["creator_claim_sha256"], f"{label}.creator_claim_sha256"),
                   _digest(row["process_identity_sha256"], f"{label}.process_identity_sha256"),
                   _text(row["physical_identity"], f"{label}.physical_identity"),
                   media)


@dataclass(frozen=True)
class FieldSpecV3:
    name: str
    kind: str
    role: str
    media_type: str
    access_class: str
    replay_class: str
    dtype: str | None = None
    shape: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class GroupSchemaV3:
    role: str
    fields: tuple[FieldSpecV3, ...]


def _a(name: str, media_type: str = "application/json") -> FieldSpecV3:
    return FieldSpecV3(name, "artifact", name, media_type, "", "")


def _b(name: str) -> FieldSpecV3:
    return _a(name, "application/octet-stream")


def _n(name: str, dtype: str, *shape: int | str) -> FieldSpecV3:
    return FieldSpecV3(name, "array", name, "application/x-npy", "", "", dtype, tuple(shape))


_GROUP_FIELDS = {
    "s1_environment_preimage": (_a("environment_registry"), _a("deployment_limits")),
    "s1_synthetic_inputs": (_a("synthetic_generation_spec"),
                            _n("synthetic_inputs", "float32", S1_BATCH, 3, 64, 64),
                            _n("synthetic_valid64", "bool", S1_BATCH, 64, 64),
                            _n("synthetic_invalid_fill", "float32", S1_BATCH, 3, 64, 64)),
    "s1_ddf_fp64_equation_preimage": (_a("ddf_fp64_tensor_registry"),),
    "s1_validity_pool_softmax_preimage": (_a("validity_pool_tensor_registry"),),
    "s1_masked_bn_preimage": (_a("masked_bn_tensor_registry"),),
    "s1_gradient_update_preimage": (_a("parameter_update_registry"),
                                     _a("optimizer_state_registry")),
    "s1_onnx_model_and_node_registry": (_b("onnx_opset17_model"), _a("onnx_domain_node_registry")),
    "s1_ort_reference_outputs": (_a("ort_input_registry"), _a("ort_output_registry")),
    "s1_trt_build_preimage": (_a("trt_build_configuration"), _a("trt_build_log"),
                               _a("trt_workload_registry")),
    "s1_trt_plan_identity_and_outputs": (_b("trt_plan"), _a("trt_binding_registry")),
    "s1_raw_timing_vectors": (
        _a("timing_protocol"),
        _n("batch1_raw_timing_seconds", "float64", BATCH1_TIMING_SAMPLES),
        _n("batch32_raw_timing_seconds", "float64", BATCH32_TIMING_SAMPLES),
    ),
    "s1_raw_cuda_allocation_trace": (_a("cuda_allocation_event_registry"),
                                      _a("cuda_allocation_raw_trace")),
    "s2a_target_free_source_snapshot": (
        _n("sample_indices", "int64", ROWS), _n("sample_component_ids", "<U64", ROWS),
        _n("component_order", "<U64", COMPONENTS), _n("held_folds", "int64", ROWS),
        _n("outer_to_calibration_fold_map", "int64", FOLDS),
        _n("model_boxes", "float32", ROWS, 4), _n("validity_packbits", "uint8", ROWS, 8192),
        _n("keeper_probabilities", "float64", ROWS, 5), _a("source_archive_member_registry")),
    "s2a_validity_bbox_preimages": (
        _n("valid64", "bool", ROWS, 64, 64), _n("valid32", "bool", ROWS, 32, 32),
        _n("valid16", "bool", ROWS, 16, 16), _n("bbox16", "bool", ROWS, 16, 16),
        _n("bbox_valid16", "bool", ROWS, 16, 16),
        _n("bbox_unusable_sample_indices", "int64", 12)),
    "s2a_keeper_rival_and_rank_preimages": (
        _n("strongest_rivals", "int64", ROWS), _a("fold_rank_partition_registry")),
    "s2a_donor_assignment_preimages": (_a("donor_cost_matrix_registry"),
                                        _a("donor_dominance_proof_registry"),
                                        _a("donor_assignment_registry")),
    "s2a_nonwrap_offset_preimages": (_n("nonwrap_offsets", "int64", ROWS, 2, 2),
                                      _a("nonwrap_slice_oracle_registry")),
    "s2a_seed_epoch_order_preimages": (_n("role_fold_seeds", "int64", ROLES, FOLDS),
                                        _a("twenty_epoch_order_registry")),
    "s2a_parameter_schema_and_init_identities": (_a("parameter_schema_registry"),
                                                  _a("initialization_identity_registry"),
                                                  _a("primary_tensor_match_registry")),
    "s2a_bootstrap_draws": (_n("component_bootstrap_draws", "int64",
                                BOOTSTRAP_REPLICATES, COMPONENTS),),
    "s2a_invalid_fill_noise": (_n("invalid_fill_noise", "uint8", ROWS, 3, 256, 256),),
    "s2a_formula_and_formal_schema": (_a("formula_registry"), _a("formal_evidence_schema"),
                                       _a("calibrator_250_registry"), _a("gate14_69_record_registry")),
    "s2b_target_source_snapshot": (_n("sample_indices", "int64", ROWS),
                                    _n("targets", "int64", ROWS),
                                    _a("target_archive_member_registry")),
    "s2b_task_and_bbox_weight_preimages": (_a("fold_task_support_registry"),
                                            _a("fold_bbox_support_registry"),
                                            _a("fp32_weight_bitpattern_registry"),
                                            _a("one_class_bootstrap_registry")),
    "s2b_fold_target_shards": (_a("fifteen_target_shard_registry"),),
    "s2b_cidt_projection": (_n("cidt_sample_indices", "int64", CIDT_ROWS),
                             _n("cidt_targets", "int64", CIDT_ROWS),
                             _n("cidt_baseline_predictions", "int64", CIDT_ROWS),
                             _a("cidt_projection_registry")),
    "s2b_deferred_xai_ids": (_n("deferred_xai_sample_indices", "int64", 30),
                              _a("deferred_xai_strata_registry")),
}
STRUCTURAL_FIELD_NAMES = frozenset({
    "trt_build_log", "trt_plan", "batch1_raw_timing_seconds",
    "batch32_raw_timing_seconds", "cuda_allocation_event_registry",
    "cuda_allocation_raw_trace",
})


def _group_access_class(role: str, replay_class: str) -> str:
    if role.startswith("s1_"):
        return f"s1_synthetic_only_{replay_class}"
    if role.startswith("s2a_"):
        return f"s2a_target_blind_{replay_class}"
    return f"s2b_target_metadata_{replay_class}"


GROUP_SCHEMAS = MappingProxyType({
    group_role: GroupSchemaV3(
        group_role,
        tuple(
            FieldSpecV3(
                field.name,
                field.kind,
                f"{group_role}__{field.name}",
                field.media_type,
                _group_access_class(
                    group_role,
                    "structural_observation" if field.name in STRUCTURAL_FIELD_NAMES
                    else "scientific_exact",
                ),
                "structural_observation" if field.name in STRUCTURAL_FIELD_NAMES
                else "scientific_exact",
                field.dtype,
                field.shape,
            )
            for field in _GROUP_FIELDS[group_role]
        ),
    )
    for group_role in S1_GROUP_ROLES + S2A_GROUP_ROLES + S2B_GROUP_ROLES
})


@dataclass(frozen=True)
class RegistrySpecV3:
    columns: tuple[str, ...]
    order_by: tuple[str, ...]
    count_rule: int | str
    required_values: tuple[tuple[str, object], ...] = ()


def _r(columns: str, order_by: str, count_rule: int | str,
       *required_values: tuple[str, object]) -> RegistrySpecV3:
    return RegistrySpecV3(
        tuple(columns.split(",")), tuple(order_by.split(",")), count_rule,
        tuple(required_values),
    )


REGISTRY_SPECS = MappingProxyType({
    "environment_registry": _r("key:str,value_type:str,value_json:json", "key", "nonempty_environment"),
    "deployment_limits": _r("resource:str,value_json:json,unit:str", "resource", "nonempty_limits"),
    "synthetic_generation_spec": _r("case_index:int,batch_size:int,seed:int,generator:str,formula_sha256:sha256", "case_index", 3),
    "ddf_fp64_tensor_registry": _r("case_index:int,tensor_name:str,array:array", "case_index,tensor_name", "all_oracle_tensors"),
    "validity_pool_tensor_registry": _r("case_index:int,tensor_name:str,array:array", "case_index,tensor_name", "all_pool_softmax_tensors"),
    "masked_bn_tensor_registry": _r("case_index:int,mode:str,tensor_name:str,array:array", "case_index,mode,tensor_name", "all_masked_bn_tensors"),
    "parameter_update_registry": _r("parameter_index:int,parameter_name:str,shape:shape,before:array,gradient:array,after:array", "parameter_index", "all_intended_parameters"),
    "optimizer_state_registry": _r("parameter_index:int,state_name:str,array:array", "parameter_index,state_name", "all_optimizer_states"),
    "onnx_domain_node_registry": _r("node_index:int,domain:onnx_domain,op_type:str,inputs:str_list,outputs:str_list", "node_index", "all_onnx_nodes"),
    "ort_input_registry": _r("batch_size:int,tensor_name:str,array:array", "batch_size,tensor_name", "all_ort_inputs"),
    "ort_output_registry": _r("batch_size:int,tensor_name:str,array:array", "batch_size,tensor_name", "all_ort_outputs"),
    "trt_build_configuration": _r("key:str,value_type:str,value_json:json", "key", "all_trt_build_options"),
    "trt_build_log": _r("sequence:int,severity:str,message:str", "sequence", "complete_build_log"),
    "trt_workload_registry": _r("batch_size:int,input_name:str,dtype:trt_dtype,shape:shape,precision:str", "batch_size,input_name", 2),
    "trt_binding_registry": _r("batch_size:int,binding_index:int,name:str,io_mode:str,dtype:trt_dtype,shape:shape,tensor:array", "batch_size,binding_index", "all_trt_bindings"),
    "timing_protocol": _r("batch_size:int,warmup_iterations:int,timed_iterations:int,clock:str,synchronization:str", "batch_size", 2),
    "cuda_allocation_event_registry": _r("event_index:int,batch_size:int,operation:str,allocated_bytes:int,reserved_bytes:int", "event_index", "complete_allocation_trace"),
    "cuda_allocation_raw_trace": _r("sequence:int,event_type:str,payload_json:json", "sequence", "complete_cuda_trace"),
    "source_archive_member_registry": _r("member_index:int,archive_role:str,member_name:str,container_bytes:int,container_sha256:sha256,member_bytes:int,member_sha256:sha256,dtype:numpy_dtype,shape:shape,target_bearing:bool", "member_index", "exact_s2a_source_members", ("target_bearing", False)),
    "fold_rank_partition_registry": _r("outer_fold:int,partition:str,expected_rows:int,sample_indices:array,bbox_ranks:array,valid_ranks:array", "outer_fold,partition", 10),
    "donor_cost_matrix_registry": _r("outer_fold:int,victim_ids:array,donor_ids:array,costs:array,maxima:array,coefficients:array", "outer_fold", 5),
    "donor_dominance_proof_registry": _r("outer_fold:int,priority_index:int,maximum:int,coefficient:int,lower_max_total:int,assignments:int,dominates:bool", "outer_fold,priority_index", 30, ("dominates", True)),
    "donor_assignment_registry": _r("outer_fold:int,lower_capacity:int,upper_capacity:int,victim_ids:array,donor_pool_ids:array,selected_donor_ids:array,donor_positions:array,use_counts:array,selected_costs:array,aggregate_costs:array,use_count_histogram:array,same_component_pairs:int,fixed_points:int,post_target_mismatch_count:optional_int", "outer_fold", 5, ("same_component_pairs", 0), ("fixed_points", 0), ("post_target_mismatch_count", None)),
    "nonwrap_slice_oracle_registry": _r("offset_index:int,dy:int,dx:int,source_rule:str,destination_rule:str", "offset_index", 8),
    "twenty_epoch_order_registry": _r("role_index:int,role_name:str,outer_fold:int,epoch:int,order:array", "role_index,outer_fold,epoch", 500),
    "parameter_schema_registry": _r("role_index:int,parameter_index:int,name:str,dtype:numpy_dtype,shape:shape,numel:int,requires_grad:bool", "role_index,parameter_index", "all_role_parameters"),
    "initialization_identity_registry": _r("role_index:int,outer_fold:int,parameter_index:int,array_sha256:sha256", "role_index,outer_fold,parameter_index", "all_initialized_parameters"),
    "primary_tensor_match_registry": _r("left_role_index:int,right_role_index:int,outer_fold:int,parameter_index:int,left_sha256:sha256,right_sha256:sha256,exact:bool", "left_role_index,right_role_index,outer_fold,parameter_index", "all_required_primary_matches", ("exact", True)),
    "formula_registry": _r("formula_index:int,name:str,source_symbol:str,source_sha256:sha256", "formula_index", "all_frozen_formulas"),
    "formal_evidence_schema": _r("record_index:int,role:str,dtype:numpy_dtype,shape:shape,row_order:str,source_group:str", "record_index", "all_formal_records"),
    "calibrator_250_registry": _r("calibrator_index:int,role_name:str,outer_fold:int,calibration_fold:int,calibrator_name:str,feature_names:str_list,scaler_with_mean:bool,scaler_with_std:bool,logistic_c_hex:str,class_weight:str,solver:str,max_iter:int,tolerance_hex:str,random_state:int,fitted_state_present:bool,probabilities_present:bool,threshold_present:bool", "calibrator_index", CALIBRATORS, ("scaler_with_mean", True), ("scaler_with_std", True), ("logistic_c_hex", "0x1.999999999999ap-4"), ("class_weight", "balanced"), ("solver", "lbfgs"), ("max_iter", 1000), ("tolerance_hex", "0x1.5798ee2308c3ap-27"), ("random_state", 20260729), ("fitted_state_present", False), ("probabilities_present", False), ("threshold_present", False)),
    "gate14_69_record_registry": _r("record_index:int,category:str,name:str,comparison_rule:str", "record_index", GATE14_RECORDS),
    "target_archive_member_registry": _r("member_index:int,archive_role:str,member_name:str,container_bytes:int,container_sha256:sha256,member_bytes:int,member_sha256:sha256,dtype:numpy_dtype,shape:shape,target_bearing:bool", "member_index", "exact_s2b_target_members", ("target_bearing", True)),
    "fold_task_support_registry": _r("outer_fold:int,task_index:int,task_name:str,partition:str,partition_rows:int,positive_support:int,negative_support:int", "outer_fold,task_index,partition", 60),
    "fold_bbox_support_registry": _r("outer_fold:int,task_index:int,task_name:str,partition:str,partition_rows:int,active_positive_support:int,active_negative_support:int,usable_positive_support:int,usable_negative_support:int", "outer_fold,task_index,partition", 60),
    "fp32_weight_bitpattern_registry": _r("outer_fold:int,task_index:int,task_name:str,weight_kind:str,uint32_bits:uint32", "outer_fold,task_index,weight_kind", 80),
    "one_class_bootstrap_registry": _r("replicate_count:int,one_class_replicates:int", "replicate_count", 1, ("replicate_count", BOOTSTRAP_REPLICATES), ("one_class_replicates", 0)),
    "fifteen_target_shard_registry": _r("shard_index:int,outer_fold:int,partition:str,expected_rows:int,sample_indices:array,targets:array", "shard_index", 15),
    "cidt_projection_registry": _r("field_index:int,name:str,top_level_field:str,array:array,expected_array_sha256:sha256", "field_index", 3),
    "deferred_xai_strata_registry": _r("position:int,sample_index:int,stratum:str,bbox_supervision_valid:bool", "position", 30),
})

REGISTRY_AXES = MappingProxyType({
    "synthetic_generation_spec": (("case_index", (0, 1, 2)),),
    "trt_workload_registry": (("batch_size", (1, 32)),),
    "timing_protocol": (("batch_size", (1, 32)),),
    "fold_rank_partition_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("partition", ("held", "calibration")),
    ),
    "donor_cost_matrix_registry": (("outer_fold", tuple(range(FOLDS))),),
    "donor_dominance_proof_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("priority_index", tuple(range(6))),
    ),
    "donor_assignment_registry": (("outer_fold", tuple(range(FOLDS))),),
    "nonwrap_slice_oracle_registry": (("offset_index", tuple(range(8))),),
    "twenty_epoch_order_registry": (
        ("role_index", tuple(range(ROLES))), ("outer_fold", tuple(range(FOLDS))),
        ("epoch", tuple(range(EPOCHS))),
    ),
    "calibrator_250_registry": (
        ("role_name", ROLE_NAMES), ("outer_fold", tuple(range(FOLDS))),
        ("calibrator_name", CALIBRATOR_NAMES),
    ),
    "gate14_69_record_registry": (("record_index", tuple(range(GATE14_RECORDS))),),
    "fold_task_support_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("task_index", tuple(range(TASKS))),
        ("partition", PARTITIONS),
    ),
    "fold_bbox_support_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("task_index", tuple(range(TASKS))),
        ("partition", PARTITIONS),
    ),
    "fp32_weight_bitpattern_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("task_index", tuple(range(TASKS))),
        ("weight_kind", WEIGHT_KINDS),
    ),
    "fifteen_target_shard_registry": (
        ("outer_fold", tuple(range(FOLDS))), ("partition", PARTITIONS),
    ),
    "cidt_projection_registry": (("field_index", (0, 1, 2)),),
    "deferred_xai_strata_registry": (("position", tuple(range(30))),),
})


def registry_count_rule_payload(field_name: str) -> Mapping[str, object]:
    spec = REGISTRY_SPECS.get(field_name)
    if spec is None:
        raise PrelockSchemaV3Error("registry count-rule field differs")
    if type(spec.count_rule) is int:
        return MappingProxyType({"mode": "exact", "count": spec.count_rule})
    return MappingProxyType({
        "mode": "source_derived", "resolver": spec.count_rule,
        "resolution_authority": "verifier_recomputed_from_pinned_artifacts",
        "minimum": 1,
    })


def _validate_registry_count_rule(value: object, field_name: str) -> None:
    row = _mapping(value, "count_rule")
    expected = dict(registry_count_rule_payload(field_name))
    _exact(row, tuple(expected), "count_rule")
    if row["mode"] != expected["mode"]:
        raise PrelockSchemaV3Error("count-rule mode differs")
    if expected["mode"] == "exact":
        _exact_integer(row["count"], int(expected["count"]), "count_rule.count")
    else:
        if (row["resolver"] != expected["resolver"]
                or row["resolution_authority"] != expected["resolution_authority"]):
            raise PrelockSchemaV3Error("source-derived count-rule identity differs")
        _exact_integer(row["minimum"], 1, "count_rule.minimum")

BINARY_CONTENT_BINDINGS = MappingProxyType({
    "onnx_opset17_model": "onnx_domain_node_registry+onnx_checker_opset17",
    "trt_plan": "trt_binding_registry+tensorrt_parser",
})


@dataclass(frozen=True)
class ScientificGroupPayloadV3:
    group_role: str
    records: tuple[ArtifactRecordV3 | ArrayRecordV3, ...]


def _registry_value(value: object, kind: str, label: str) -> object:
    if kind == "str":
        return _text(value, label)
    if kind == "int":
        return _integer(value, label)
    if kind == "uint32":
        observed = _integer(value, label)
        if observed >= 2 ** 32:
            raise PrelockSchemaV3Error(f"{label} exceeds uint32")
        return observed
    if kind == "bool":
        if type(value) is not bool:
            raise PrelockSchemaV3Error(f"{label} must be bool")
        return value
    if kind == "numpy_dtype":
        observed = _text(value, label)
        if _DTYPE.fullmatch(observed) is None:
            raise PrelockSchemaV3Error(f"{label} is not a frozen NumPy dtype")
        return observed
    if kind == "trt_dtype":
        observed = _text(value, label)
        if observed not in {"bool", "float32"}:
            raise PrelockSchemaV3Error(f"{label} is not a frozen TensorRT dtype")
        return observed
    if kind == "onnx_domain":
        if type(value) is not str or value != "":
            raise PrelockSchemaV3Error(f"{label} is not the standard ONNX domain")
        return value
    if kind == "optional_int":
        return None if value is None else _integer(value, label)
    if kind == "sha256":
        return _digest(value, label)
    if kind == "shape":
        return _shape(value, label)
    if kind == "str_list":
        return tuple(_text(item, f"{label}[{index}]")
                     for index, item in enumerate(_sequence(value, label)))
    if kind == "array":
        return ArrayRecordV3.parse(value, label)
    if kind == "artifact":
        return ArtifactRecordV3.parse(value, label)
    if kind == "json":
        raw = _text(value, label)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrelockSchemaV3Error(f"{label} is not canonical JSON text") from exc
        if canonical_json_bytes(decoded).decode("ascii") != raw:
            raise PrelockSchemaV3Error(f"{label} is not canonical JSON text")
        return decoded
    raise PrelockSchemaV3Error(f"{label} has an unknown registry kind")


def _registry_array(row: Mapping[str, object], name: str, dtype: str,
                    shape: tuple[int, ...], label: str) -> ArrayRecordV3:
    array = ArrayRecordV3.parse(row[name], f"{label}.{name}")
    if array.dtype != dtype or array.shape != shape:
        raise PrelockSchemaV3Error(f"{label}.{name} dependent dtype/shape differs")
    return array


def _partition_rows(outer_fold: int, partition: str) -> int:
    if partition == "fit":
        return FIT_ROW_COUNTS[outer_fold]
    fold = outer_fold if partition == "held" else OUTER_TO_CALIBRATION_FOLD[outer_fold]
    return FOLD_ROW_COUNTS[fold]


def _validate_scientific_registry(
    field_name: str,
    rows: Sequence[Mapping[str, object]],
    group_payload: ScientificGroupPayloadV3 | None,
) -> None:
    if field_name == "fold_rank_partition_registry":
        for index, row in enumerate(rows):
            count = _partition_rows(int(row["outer_fold"]), str(row["partition"]))
            if row["expected_rows"] != count:
                raise PrelockSchemaV3Error("rank partition row count differs")
            for name in ("sample_indices", "bbox_ranks", "valid_ranks"):
                _registry_array(row, name, "int64", (count,), f"rank[{index}]")
    elif field_name == "donor_cost_matrix_registry":
        for row in rows:
            fold = int(row["outer_fold"])
            held = FOLD_ROW_COUNTS[fold]
            calibration = FOLD_ROW_COUNTS[OUTER_TO_CALIBRATION_FOLD[fold]]
            _registry_array(row, "victim_ids", "int64", (held,), f"donor_cost[{fold}]")
            _registry_array(row, "donor_ids", "int64", (calibration,), f"donor_cost[{fold}]")
            _registry_array(row, "costs", "int64", (held, calibration, 6), f"donor_cost[{fold}]")
            _registry_array(row, "maxima", "int64", (6,), f"donor_cost[{fold}]")
            _registry_array(row, "coefficients", "int64", (6,), f"donor_cost[{fold}]")
    elif field_name == "donor_dominance_proof_registry":
        for row in rows:
            if row["assignments"] != FOLD_ROW_COUNTS[int(row["outer_fold"])]:
                raise PrelockSchemaV3Error("donor dominance assignment count differs")
    elif field_name == "donor_assignment_registry":
        for row in rows:
            fold = int(row["outer_fold"])
            held = FOLD_ROW_COUNTS[fold]
            calibration = FOLD_ROW_COUNTS[OUTER_TO_CALIBRATION_FOLD[fold]]
            lower, remainder = divmod(held, calibration)
            upper = lower + int(remainder > 0)
            if row["lower_capacity"] != lower or row["upper_capacity"] != upper:
                raise PrelockSchemaV3Error("donor capacity differs")
            shapes = {
                "victim_ids": (held,), "donor_pool_ids": (calibration,),
                "selected_donor_ids": (held,), "donor_positions": (held,),
                "use_counts": (calibration,), "selected_costs": (held, 6),
                "aggregate_costs": (6,), "use_count_histogram": (upper + 1,),
            }
            for name, shape in shapes.items():
                _registry_array(row, name, "int64", shape, f"donor_assignment[{fold}]")
    elif field_name == "nonwrap_slice_oracle_registry":
        for index, row in enumerate(rows):
            if (row["dy"], row["dx"]) != LOCKED_NONWRAP_OFFSETS[index]:
                raise PrelockSchemaV3Error("nonwrap offset axis differs")
            if (row["source_rule"] != "max(0,-d):size-max(0,d)"
                    or row["destination_rule"] != "max(0,d):size-max(0,-d)"):
                raise PrelockSchemaV3Error("nonwrap executable slice rule differs")
    elif field_name == "twenty_epoch_order_registry":
        for row in rows:
            role = int(row["role_index"])
            fold = int(row["outer_fold"])
            if row["role_name"] != ROLE_NAMES[role]:
                raise PrelockSchemaV3Error("epoch-order role name differs")
            _registry_array(row, "order", "int64", (FIT_ROW_COUNTS[fold],), "epoch_order")
    elif field_name == "calibrator_250_registry":
        for index, row in enumerate(rows):
            fold = int(row["outer_fold"])
            name = str(row["calibrator_name"])
            if (row["calibrator_index"] != index
                    or row["calibration_fold"] != OUTER_TO_CALIBRATION_FOLD[fold]
                    or tuple(row["feature_names"]) != CALIBRATOR_FEATURE_NAMES[name]):
                raise PrelockSchemaV3Error("calibrator exact axis/features differ")
    elif field_name == "gate14_69_record_registry":
        projection = [[row["category"], row["name"], row["comparison_rule"]] for row in rows]
        if hashlib.sha256(canonical_json_bytes(projection)).hexdigest() != GATE14_REGISTRY_SHA256:
            raise PrelockSchemaV3Error("Gate-14 exact comparison registry differs")
    elif field_name in {"fold_task_support_registry", "fold_bbox_support_registry"}:
        for row in rows:
            fold, task = int(row["outer_fold"]), int(row["task_index"])
            if (row["task_name"] != TASK_NAMES[task]
                    or row["partition_rows"] != _partition_rows(fold, str(row["partition"]))):
                raise PrelockSchemaV3Error("task support axis/count differs")
            keys = ("positive_support", "negative_support") if field_name.startswith("fold_task") else (
                "active_positive_support", "active_negative_support",
                "usable_positive_support", "usable_negative_support",
            )
            if any(type(row[key]) is not int or row[key] <= 0 for key in keys):
                raise PrelockSchemaV3Error("task support is not strictly positive")
            active_sum = (
                row["positive_support"] + row["negative_support"]
                if field_name.startswith("fold_task") else
                row["active_positive_support"] + row["active_negative_support"]
            )
            if (task == 0 and active_sum != row["partition_rows"]) or active_sum > row["partition_rows"]:
                raise PrelockSchemaV3Error("task active support differs from partition")
            if field_name.startswith("fold_bbox") and (
                row["usable_positive_support"] > row["active_positive_support"]
                or row["usable_negative_support"] > row["active_negative_support"]
            ):
                raise PrelockSchemaV3Error("bbox usable support exceeds active support")
    elif field_name == "fp32_weight_bitpattern_registry":
        for row in rows:
            task = int(row["task_index"])
            bits = int(row["uint32_bits"])
            if row["task_name"] != TASK_NAMES[task] or not 0 < bits < 0x7F800000:
                raise PrelockSchemaV3Error("FP32 task-weight identity differs")
    elif field_name == "fifteen_target_shard_registry":
        for index, row in enumerate(rows):
            fold = int(row["outer_fold"])
            count = _partition_rows(fold, str(row["partition"]))
            if row["shard_index"] != index or row["expected_rows"] != count:
                raise PrelockSchemaV3Error("target-shard axis/count differs")
            _registry_array(row, "sample_indices", "int64", (count,), f"target_shard[{index}]")
            _registry_array(row, "targets", "int64", (count,), f"target_shard[{index}]")
    elif field_name == "cidt_projection_registry":
        if group_payload is None or group_payload.group_role != "s2b_cidt_projection":
            raise PrelockSchemaV3Error("CIDT registry lacks its typed group payload")
        schema = GROUP_SCHEMAS[group_payload.group_role]
        top_level = {field.name: record for field, record in zip(schema.fields, group_payload.records)}
        expected_names = (
            ("sample_indices", "cidt_sample_indices"),
            ("targets", "cidt_targets"),
            ("baseline_predictions", "cidt_baseline_predictions"),
        )
        for index, (name, top_name) in enumerate(expected_names):
            row = rows[index]
            array = ArrayRecordV3.parse(row["array"], f"cidt[{index}].array")
            if (row["name"] != name or row["top_level_field"] != top_name
                    or array != top_level[top_name]
                    or row["expected_array_sha256"] != EXACT_CIDT_ARRAY_SHA256[top_name]
                    or array.array_sha256 != EXACT_CIDT_ARRAY_SHA256[top_name]):
                raise PrelockSchemaV3Error("CIDT top-level cross-binding differs")
    elif field_name == "deferred_xai_strata_registry":
        sample_indices = [int(row["sample_index"]) for row in rows]
        if len(set(sample_indices)) != 30:
            raise PrelockSchemaV3Error("deferred XAI sample IDs are not unique")


def parse_registry_bytes(payload: object, expected_group_role: str,
                         expected_field_name: str,
                         group_payload: object,
                         ) -> tuple[Mapping[str, object], ...]:
    if expected_group_role not in GROUP_SCHEMAS:
        raise PrelockSchemaV3Error("registry group role differs")
    fields = {field.name: field for field in GROUP_SCHEMAS[expected_group_role].fields}
    if expected_field_name not in fields:
        raise PrelockSchemaV3Error("registry field is absent from the exact group")
    field = fields[expected_field_name]
    if field.kind != "artifact" or field.media_type != "application/json":
        raise PrelockSchemaV3Error("field is not a canonical JSON registry")
    spec = REGISTRY_SPECS.get(expected_field_name)
    if spec is None:
        raise PrelockSchemaV3Error("registry has no frozen inner content schema")
    if type(payload) is not bytes:
        raise PrelockSchemaV3Error("registry payload must be bytes")
    typed_group = parse_group_payload(group_payload, expected_group_role)
    field_index = tuple(item.name for item in GROUP_SCHEMAS[expected_group_role].fields).index(
        expected_field_name
    )
    group_record = typed_group.records[field_index]
    if (not isinstance(group_record, ArtifactRecordV3)
            or isinstance(group_record, ArrayRecordV3)
            or group_record.bytes != len(payload)
            or group_record.sha256 != hashlib.sha256(payload).hexdigest()):
        raise PrelockSchemaV3Error("registry bytes differ from the typed group artifact")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrelockSchemaV3Error("registry payload is not UTF-8 JSON") from exc
    if canonical_json_bytes(value) != payload:
        raise PrelockSchemaV3Error("registry payload is not canonical no-newline JSON")
    row = _mapping(value, expected_field_name)
    outer_fields = (
        "evidence_schema_version", "schema_id", "columns", "order_by",
        "count_rule", "row_count", "rows",
    )
    _exact(row, outer_fields, expected_field_name)
    _validate_registry_count_rule(row["count_rule"], expected_field_name)
    _exact_integer(row["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION,
                   "registry.evidence_schema_version")
    if (row["schema_id"] != field.role
            or tuple(_sequence(row["columns"], "columns")) != spec.columns
            or tuple(_sequence(row["order_by"], "order_by")) != spec.order_by):
        raise PrelockSchemaV3Error("registry schema identity differs")
    rows = _sequence(row["rows"], "rows")
    if _integer(row["row_count"], "row_count") != len(rows):
        raise PrelockSchemaV3Error("registry row count differs")
    if not rows:
        raise PrelockSchemaV3Error("registry cannot be empty")
    if type(spec.count_rule) is int:
        if len(rows) != spec.count_rule:
            raise PrelockSchemaV3Error("registry fixed row count differs")
    names = tuple(column.split(":", 1)[0] for column in spec.columns)
    kinds = tuple(column.split(":", 1)[1] for column in spec.columns)
    parsed: list[Mapping[str, object]] = []
    order_keys: list[tuple[object, ...]] = []
    for index, raw_row in enumerate(rows):
        item = _mapping(raw_row, f"rows[{index}]")
        _exact(item, names, f"rows[{index}]")
        for name, kind in zip(names, kinds):
            observed = _registry_value(item[name], kind, f"rows[{index}].{name}")
            if isinstance(observed, ArtifactRecordV3) and not (
                expected_field_name == "cidt_projection_registry" and name == "array"
            ):
                suffix = ".npy" if isinstance(observed, ArrayRecordV3) else ".bin"
                expected_child_role = f"{field.role}__row_{index:06d}__{name}"
                expected_child_path = (
                    f"$RUN_OUT/{expected_group_role}/{expected_field_name}/"
                    f"row_{index:06d}/{name}{suffix}"
                )
                if (observed.role != expected_child_role
                        or observed.logical_path != expected_child_path
                        or observed.access_class != field.access_class):
                    raise PrelockSchemaV3Error(
                        f"rows[{index}].{name} nested artifact identity differs"
                    )
        for name, expected in spec.required_values:
            if item[name] != expected:
                raise PrelockSchemaV3Error(f"rows[{index}].{name} required value differs")
        order_keys.append(tuple(item[name] for name in spec.order_by))
        parsed.append(_freeze(dict(item)))
    axes = REGISTRY_AXES.get(expected_field_name)
    if axes is not None:
        expected_axes = list(product(*(values for _, values in axes)))
        observed_axes = [tuple(item[name] for name, _ in axes) for item in rows]
        if observed_axes != expected_axes:
            raise PrelockSchemaV3Error("registry Cartesian axes/order differ")
    elif order_keys != sorted(order_keys) or len(set(order_keys)) != len(order_keys):
        raise PrelockSchemaV3Error("registry rows are not uniquely canonically ordered")
    _validate_scientific_registry(expected_field_name, tuple(parsed), typed_group)
    if type(spec.count_rule) is str:
        raise PrelockSchemaV3Error(
            "source-derived registry is unavailable until its stage-specific verifier"
        )
    return tuple(parsed)


def parse_group_payload(value: object, expected_role: str) -> ScientificGroupPayloadV3:
    if expected_role not in GROUP_SCHEMAS:
        raise PrelockSchemaV3Error("unknown scientific group role")
    row = _mapping(value, expected_role)
    _exact(row, ("evidence_schema_version", "group_role", "records"), expected_role)
    _exact_integer(row["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION,
                   f"{expected_role}.evidence_schema_version")
    if row["group_role"] != expected_role:
        raise PrelockSchemaV3Error(f"{expected_role} identity differs")
    records = _mapping(row["records"], f"{expected_role}.records")
    schema = GROUP_SCHEMAS[expected_role]
    _exact(records, tuple(field.name for field in schema.fields), f"{expected_role}.records")
    parsed: list[ArtifactRecordV3 | ArrayRecordV3] = []
    observed_paths: set[str] = set()
    for field in schema.fields:
        record = (ArrayRecordV3 if field.kind == "array" else ArtifactRecordV3).parse(
            records[field.name], f"{expected_role}.{field.name}")
        if record.role != field.role:
            raise PrelockSchemaV3Error(f"{expected_role}.{field.name} role differs")
        if record.media_type != field.media_type or record.access_class != field.access_class:
            raise PrelockSchemaV3Error(f"{expected_role}.{field.name} media/access differs")
        suffix = ".npy" if isinstance(record, ArrayRecordV3) else (
            ".json" if field.media_type == "application/json" else ".bin"
        )
        expected_path = f"$RUN_OUT/{expected_role}/{field.name}{suffix}"
        if record.logical_path != expected_path:
            raise PrelockSchemaV3Error(f"{expected_role}.{field.name} output path differs")
        if record.logical_path in observed_paths:
            raise PrelockSchemaV3Error(f"{expected_role} output path aliases")
        observed_paths.add(record.logical_path)
        if isinstance(record, ArrayRecordV3):
            expected_shape = tuple(field.shape)
            if record.dtype != field.dtype or record.shape != expected_shape:
                raise PrelockSchemaV3Error(f"{expected_role}.{field.name} dtype/shape differs")
            if (expected_role == "s2b_cidt_projection"
                    and field.name in EXACT_CIDT_ARRAY_SHA256
                    and record.array_sha256 != EXACT_CIDT_ARRAY_SHA256[field.name]):
                raise PrelockSchemaV3Error(f"{expected_role}.{field.name} exact CIDT identity differs")
            expected_a0_sha = None
            if expected_role == "s2a_target_free_source_snapshot" and field.name in {
                "sample_indices", "held_folds",
            }:
                expected_a0_sha = EXACT_A0_ARRAY_SHA256[field.name]
            elif expected_role == "s2b_target_source_snapshot" and field.name in {
                "sample_indices", "targets",
            }:
                expected_a0_sha = EXACT_A0_ARRAY_SHA256[field.name]
            if expected_a0_sha is not None and record.array_sha256 != expected_a0_sha:
                raise PrelockSchemaV3Error(f"{expected_role}.{field.name} exact A0 identity differs")
        elif (field.media_type == "application/json" and field.name not in REGISTRY_SPECS) or (
            field.media_type == "application/octet-stream"
            and field.name not in BINARY_CONTENT_BINDINGS
        ):
            raise PrelockSchemaV3Error(f"{expected_role}.{field.name} content schema is absent")
        parsed.append(record)
    return ScientificGroupPayloadV3(expected_role, tuple(parsed))


COMMON_CONTRACT_LINEAGE_FIELDS = (
    "foundation_commit", "boundary_review_commit", "source_commit", "branch",
    "upstream_ref", "upstream_commit", "repository_tree_sha256", "launcher",
    "v2_prelock_evidence_schema_v3", "v2_prelock_evidence_schema_v3_source",
    "full_local_import_closure", "package_files",
    "fixed_protocol_and_incident_hashes", "expanded_test_node_registry_sha256",
    "rejected_lineage",
)
LINEAGE_MATRIX_FIELDS = MappingProxyType({
    "contract": (), "authorization": ("contract",),
    "bundle": ("contract", "authorization"),
    "claim": ("contract", "authorization"),
    "verifier_receipt": ("contract", "authorization", "bundle"),
})


def parse_common_contract_lineage(value: object, stage: str) -> Mapping[str, object]:
    if stage not in {"s1", "s2"}:
        raise PrelockSchemaV3Error("contract lineage stage differs")
    row = _mapping(value, "contract_lineage")
    fields = COMMON_CONTRACT_LINEAGE_FIELDS + (("verified_s1_receipt",) if stage == "s2" else ())
    _exact(row, fields, "contract_lineage")
    if row["foundation_commit"] != FOUNDATION_COMMIT or row["boundary_review_commit"] != BOUNDARY_REVIEW_COMMIT:
        raise PrelockSchemaV3Error("fixed commit lineage differs")
    source_commit = _commit(row["source_commit"], "source_commit")
    upstream_commit = _commit(row["upstream_commit"], "upstream_commit")
    if upstream_commit != source_commit:
        raise PrelockSchemaV3Error("upstream_commit must equal source_commit")
    launcher = ArtifactRecordV3.parse(row["launcher"], "contract_lineage.launcher")
    if (launcher.role != "v2_execution_launcher" or launcher.media_type != "text/x-python"
            or launcher.access_class != "committed_pushed_local_import_identity"
            or launcher.logical_path.startswith("$RUN_OUT/")):
        raise PrelockSchemaV3Error("launcher artifact role differs")
    schema = ArtifactRecordV3.parse(row["v2_prelock_evidence_schema_v3"], "contract_lineage.schema")
    if (schema.role != "v2_prelock_evidence_schema_v3" or schema.media_type != "text/markdown"
            or schema.access_class != "committed_pushed_boundary_document_identity"
            or schema.logical_path.startswith("$RUN_OUT/")):
        raise PrelockSchemaV3Error("schema artifact role differs")
    schema_source = ArtifactRecordV3.parse(
        row["v2_prelock_evidence_schema_v3_source"], "contract_lineage.schema_source"
    )
    if (schema_source.role != "v2_prelock_evidence_schema_v3_source"
            or schema_source.media_type != "text/x-python"
            or schema_source.access_class != "committed_pushed_local_import_identity"
            or schema_source.logical_path.startswith("$RUN_OUT/")):
        raise PrelockSchemaV3Error("schema-source artifact role differs")
    if stage == "s2":
        receipt = ArtifactRecordV3.parse(row["verified_s1_receipt"], "contract_lineage.s1_receipt")
        if (receipt.role != "v2_s1_finalization_root_record"
                or receipt.media_type != "application/json"
                or receipt.access_class != "committed_pushed_stage_receipt_identity"
                or receipt.logical_path.startswith("$RUN_OUT/")):
            raise PrelockSchemaV3Error("S2 contract does not bind the verified S1 receipt")
    closure_records: Mapping[str, object] | None = None
    for name in ("full_local_import_closure", "package_files"):
        records = _mapping(row[name], f"contract_lineage.{name}")
        if not records or tuple(records) != tuple(sorted(records)):
            raise PrelockSchemaV3Error(f"contract_lineage.{name} must be nonempty and ordered")
        for key, record in records.items():
            parsed_record = ArtifactRecordV3.parse(record, f"contract_lineage.{name}.{key}")
            if parsed_record.logical_path.startswith("$RUN_OUT/"):
                raise PrelockSchemaV3Error(f"contract_lineage.{name}.{key} is not an input identity")
            expected_access = (
                "committed_pushed_local_import_identity"
                if name == "full_local_import_closure" else "pinned_package_file_identity"
            )
            if parsed_record.access_class != expected_access:
                raise PrelockSchemaV3Error(f"contract_lineage.{name}.{key} access class differs")
        if name == "full_local_import_closure":
            closure_records = records
    if closure_records is None:
        raise PrelockSchemaV3Error("full import closure was not parsed")
    closure_schema_source = closure_records.get(
        "trkh.tools.pair_surface_ddf_v2_prelock_schema_v3"
    )
    if closure_schema_source is None or ArtifactRecordV3.parse(
        closure_schema_source, "contract_lineage.schema_source_closure"
    ) != schema_source:
        raise PrelockSchemaV3Error("schema source is not exact in the full import closure")
    fixed = _mapping(row["fixed_protocol_and_incident_hashes"], "fixed hashes")
    if dict(fixed) != dict(FIXED_PROTOCOL_AND_INCIDENT_SHA256):
        raise PrelockSchemaV3Error("fixed protocol/incident hashes differ")
    rejected = _sequence(row["rejected_lineage"], "rejected_lineage")
    exact_rejection = {"sha256": INVALIDATED_MANIFEST_SHA256, "reason": INVALIDATION_REASON,
                       "scientific_effect": 0, "authority_effect": 0}
    if len(rejected) != 1:
        raise PrelockSchemaV3Error("invalidated f36 lineage rejection differs")
    rejection = _mapping(rejected[0], "rejected_lineage[0]")
    _exact(rejection, tuple(exact_rejection), "rejected_lineage[0]")
    if (type(rejection["sha256"]) is not str
            or rejection["sha256"] != INVALIDATED_MANIFEST_SHA256
            or type(rejection["reason"]) is not str
            or rejection["reason"] != INVALIDATION_REASON):
        raise PrelockSchemaV3Error("invalidated f36 lineage rejection differs")
    _exact_integer(rejection["scientific_effect"], 0, "rejection.scientific_effect")
    _exact_integer(rejection["authority_effect"], 0, "rejection.authority_effect")
    accepted = {key: value for key, value in row.items() if key != "rejected_lineage"}
    if INVALIDATED_MANIFEST_SHA256 in canonical_json_bytes(accepted).decode("ascii"):
        raise PrelockSchemaV3Error("invalidated f36 appears in accepted lineage")
    _digest(row["repository_tree_sha256"], "repository_tree_sha256")
    _digest(row["expanded_test_node_registry_sha256"], "expanded_test_node_registry_sha256")
    _text(row["branch"], "branch")
    _text(row["upstream_ref"], "upstream_ref")
    return _freeze(dict(row))


def parse_lineage_binding(value: object, kind: str, stage: str) -> tuple[ArtifactRecordV3, ...]:
    if kind not in LINEAGE_MATRIX_FIELDS or stage not in {"s1", "s2"}:
        raise PrelockSchemaV3Error("lineage binding kind/stage differs")
    row = _mapping(value, f"{kind}_binding")
    fields = LINEAGE_MATRIX_FIELDS[kind]
    _exact(row, fields, f"{kind}_binding")
    records = tuple(ArtifactRecordV3.parse(row[field], f"{kind}_binding.{field}") for field in fields)
    expected = tuple(
        f"v2_{stage}_bundle_manifest" if field == "bundle" else f"v2_{stage}_{field}"
        for field in fields
    )
    if tuple(record.role for record in records) != expected:
        raise PrelockSchemaV3Error(f"{kind} does not bind the exact {stage} records")
    return records


DOWNSTREAM_AUTHORITY_FIELDS = (
    "formal_runs_authorized", "candidate_runs_authorized", "validation_runs_authorized",
    "test_runs_authorized", "conditional_xai_runs_authorized",
)
S1_FINALIZATION_PAYLOAD_FIELDS = (
    "evidence_schema_version", "foundation_commit", "boundary_review_commit",
    "schema_document_sha256", "schema_source_sha256", "source_commit",
    "s1_bundle_sha256", "s1_contract_sha256", "s1_authorization_sha256",
    "primary_operational_root_sha256", "replay_operational_root_sha256",
    "exact_scientific_root_sha256", "primary_structural_root_sha256",
    "replay_structural_root_sha256", "engineering_decision_root_sha256",
    "fresh_process_replay", "s2_contract_may_be_created", "downstream_authority",
    "cache_or_lineage_reopened",
)
S2_FINALIZATION_PAYLOAD_FIELDS = (
    "evidence_schema_version", "foundation_commit", "boundary_review_commit",
    "schema_document_sha256", "schema_source_sha256", "source_commit",
    "s1_finalization_record_sha256", "s2_bundle_sha256", "s2_contract_sha256",
    "s2_authorization_sha256", "primary_operational_root_sha256",
    "replay_operational_root_sha256", "s2a_inner_root_sha256",
    "exact_scientific_root_sha256", "primary_structural_root_sha256",
    "replay_structural_root_sha256", "derivation_decision_root_sha256",
    "fresh_process_replay", "downstream_authority", "cache_or_lineage_reopened",
)


def parse_finalization_payload(value: object, stage: str) -> Mapping[str, object]:
    fields = S1_FINALIZATION_PAYLOAD_FIELDS if stage == "s1" else S2_FINALIZATION_PAYLOAD_FIELDS if stage == "s2" else ()
    if not fields:
        raise PrelockSchemaV3Error("finalization stage differs")
    row = _mapping(value, f"{stage}_finalization")
    _exact(row, fields, f"{stage}_finalization")
    _exact_integer(row["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION,
                   f"{stage}.evidence_schema_version")
    if row["foundation_commit"] != FOUNDATION_COMMIT or row["boundary_review_commit"] != BOUNDARY_REVIEW_COMMIT:
        raise PrelockSchemaV3Error(f"{stage} finalization fixed lineage differs")
    _commit(row["source_commit"], f"{stage} finalization source commit")
    for name in fields:
        if name.endswith("_sha256"):
            _digest(row[name], f"{stage}_finalization.{name}")
    if row["fresh_process_replay"] is not True:
        raise PrelockSchemaV3Error(f"{stage} finalization replay is not fresh")
    if stage == "s1" and row["s2_contract_may_be_created"] is not True:
        raise PrelockSchemaV3Error("S1 finalization does not permit the S2 contract")
    authority = _mapping(row["downstream_authority"], f"{stage}.downstream_authority")
    _exact(authority, DOWNSTREAM_AUTHORITY_FIELDS, f"{stage}.downstream_authority")
    if any(type(authority[name]) is not int or authority[name] != 0
           for name in DOWNSTREAM_AUTHORITY_FIELDS):
        raise PrelockSchemaV3Error(f"{stage} finalization grants authority")
    if type(row["cache_or_lineage_reopened"]) is not int or row["cache_or_lineage_reopened"] != 0:
        raise PrelockSchemaV3Error(f"{stage} finalization reopens cache or lineage")
    return _freeze(dict(row))


def parse_outer_document(value: object, expected_kind: str | None = None) -> Mapping[str, object]:
    row = _mapping(value, "outer_document")
    _exact(row, OUTER_FIELDS, "outer_document")
    kind = _text(row["kind"], "outer_document.kind")
    if expected_kind is not None and kind != expected_kind:
        raise PrelockSchemaV3Error("outer kind differs")
    if kind not in STAGE_ROLE_STATES or row["state"] != STAGE_ROLE_STATES[kind]:
        raise PrelockSchemaV3Error("outer role/state differs")
    _exact_integer(row["schema_version"], DOCUMENT_SCHEMA_VERSION,
                   "outer_document.schema_version")
    if row["protocol_id"] != PROTOCOL_ID:
        raise PrelockSchemaV3Error("outer protocol identity differs")
    payload = _mapping(row["payload"], "outer_document.payload")
    if "evidence_schema_version" not in payload:
        raise PrelockSchemaV3Error("outer evidence schema version differs")
    _exact_integer(payload["evidence_schema_version"], EVIDENCE_SCHEMA_VERSION,
                   "outer payload evidence_schema_version")
    if kind == "v2_s1_finalization_root_record":
        parse_finalization_payload(payload, "s1")
    elif kind == "v2_s2_finalization_root_record":
        parse_finalization_payload(payload, "s2")
    else:
        raise PrelockSchemaV3Error(
            f"{kind} payload schema is not frozen in this fail-closed source"
        )
    return _freeze(dict(row))


def canonical_json_bytes(value: object) -> bytes:
    def json_ready(item: object) -> object:
        if isinstance(item, Mapping):
            if not all(type(key) is str for key in item):
                raise PrelockSchemaV3Error("canonical JSON mapping keys must be strings")
            return {key: json_ready(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_ready(child) for child in item]
        return item

    ready = json_ready(value)
    try:
        return json.dumps(ready, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PrelockSchemaV3Error("value is not canonical JSON-compatible") from exc


def serialize_outer_document(value: object, expected_kind: str | None = None) -> bytes:
    parse_outer_document(value, expected_kind)
    return canonical_json_bytes(value)


def parse_outer_bytes(payload: object, expected_kind: str | None = None) -> Mapping[str, object]:
    if type(payload) is not bytes:
        raise PrelockSchemaV3Error("outer payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrelockSchemaV3Error("outer payload is not UTF-8 JSON") from exc
    parsed = parse_outer_document(value, expected_kind)
    if canonical_json_bytes(value) != payload:
        raise PrelockSchemaV3Error("outer payload is not exact no-newline canonical JSON")
    return parsed


__all__ = [
    "ArchiveMemberEventV3", "ArrayRecordV3", "ArtifactRecordV3",
    "BATCH1_TIMING_SAMPLES", "BATCH32_TIMING_SAMPLES", "BINARY_CONTENT_BINDINGS",
    "BOOTSTRAP_REPLICATES", "BOUNDARY_REVIEW_COMMIT", "CALIBRATOR_NAMES",
    "CIDT_ROWS", "COMMON_CONTRACT_LINEAGE_FIELDS",
    "DOCUMENT_SCHEMA_VERSION", "DOWNSTREAM_AUTHORITY_FIELDS", "EVIDENCE_SCHEMA_VERSION",
    "EXACT_A0_ARRAY_SHA256", "EXACT_CIDT_ARRAY_SHA256", "FIT_ROW_COUNTS",
    "FIXED_PROTOCOL_AND_INCIDENT_SHA256", "FOLD_ROW_COUNTS", "FOUNDATION_COMMIT",
    "GATE14_REGISTRY_SHA256", "GROUP_SCHEMAS",
    "INVALIDATED_MANIFEST_SHA256", "INVALIDATION_DOCUMENT_SHA256", "INVALIDATION_REASON",
    "LINEAGE_MATRIX_FIELDS", "OUTER_TO_CALIBRATION_FOLD", "PARTITIONS",
    "PrelockSchemaV3Error", "PROTOCOL_ID", "REGISTRY_AXES", "REGISTRY_SPECS",
    "ROLE_NAMES", "ROWS", "S1_BATCH", "S1_FINALIZATION_PAYLOAD_FIELDS",
    "S1_GROUP_ROLES",
    "S2A_GROUP_ROLES", "S2B_GROUP_ROLES",
    "S2_FINALIZATION_PAYLOAD_FIELDS", "STAGE_ROLE_STATES", "TASK_NAMES",
    "WEIGHT_KINDS", "WriteReceiptV3",
    "canonical_json_bytes", "parse_common_contract_lineage", "parse_finalization_payload",
    "parse_group_payload", "parse_lineage_binding", "parse_outer_bytes", "parse_registry_bytes",
    "parse_outer_document", "registry_count_rule_payload", "serialize_outer_document",
]
