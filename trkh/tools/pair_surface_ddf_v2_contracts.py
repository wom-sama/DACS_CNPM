from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 3
PROTOCOL_ID = "trkh_pair_surface_ddf_a0_v2"
MACHINE_LOCK_STATE = "trusted_fresh_process_runner_locked_no_execution_authority"
PENDING_AUTHORIZATION_STATE = "pending_pair_run_authorization_committed_pushed_unconsumed"
PRELOCK_VERIFIER_REVIEWED = False
PRELOCK_DOCUMENT_SCHEMA_VERSION = 2
SCIENTIFIC_FOUNDATION_COMMIT = "9ea25ea87ae6d497b80dd86c27e59391d21c4a9b"
BOUNDARY_REVIEW_COMMIT = "c8fd6ca080effcfd78bae8fb9fc389056bd27ad9"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
TARGET_STAGES = ("fit", "calibration", "held", "cidt", "xai")
SCIENTIFIC_OUTPUT_KINDS = (*TARGET_STAGES[:2], "held_prediction", *TARGET_STAGES[2:])
PHASES = ("primary", "replay")
REQUIRED_LOCAL_RUNTIME_MODULES = frozenset(
    {
        "trkh",
        "trkh.tools",
        "trkh.tools.pair_surface_ddf_v2_contracts",
        "trkh.tools.pair_surface_ddf_v2_runtime",
        "trkh.tools.pair_surface_ddf_v2_handoff",
        "trkh.tools.pair_surface_ddf_v2_execution_guard",
    }
)
REQUIRED_PACKAGE_MODULES = frozenset({"numpy", "threadpoolctl", "torch"})
DETERMINISTIC_ENVIRONMENT = MappingProxyType(
    {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
)
MANDATORY_CONTROL_ROLES = frozenset(
    {
        "claim",
        "runtime_attestation",
        "access_ledger",
        "member_ledger",
        "transition_chain",
        "consumption",
        "tombstone",
        "handoff",
        "final_root",
    }
)
HANDOFF_DISPOSITIONS = frozenset(
    {"replay_input", "compare_output", "retained_evidence", "control_evidence"}
)
PRELOCK_FIXED_EVIDENCE_SHA256 = MappingProxyType(
    {
        "v2_protocol_base": "b5fa5e2ada7e078a08276902f5897a0358d21ae03658c0d398402c4277c0b679",
        "v2_protocol_erratum": "0da08a193ca50f775047e28fcca763f8aafcc2391d54934480a3574ba1c89aeb",
        "v2_fold_erratum_r2": "1e9c5e64b259b909c0d69591b7ea5dd42cfc72f7c8db1489f89de29d3f9206c7",
        "v2_fold_manifest_r2": "81a404bedc25dc7b3e7dc3dbb8d79b4ca5e97d74374083c79e0686ececb7fb2d",
        "mechanism_registry_v3_lineage": "49b1d3422fb0d592e56c7814289fde8fe058986cc7be671357bb95d4b297566d",
        "v2_prelock_sequencing_erratum": "b986835a2ac7eff3ac98cd68b19331c6de3c828268d9539c8f46ff57afa4d1a0",
        "v2_prelock_process_incident": "c4c5eba21733d85b98b76c68312f1223ede3f6d5d1919ba25753037736b232f7",
        "v2_prelock_process_incident_independent_review_erratum":
            "b7ef5a3847b125273d7771ec35dc935ba52a570279db18f3d3ae928a95b9b96c",
        "v2_runtime_guard_test_attestation_invalidation":
            "5b52066708f419ab5ce728f64328585829ab5eaf84954da74c1909e2b58bd558",
        "v2_engine": "e6558d8cf9c00ec3d5f89b549902817a25afca21bff399a4766c397e80ba2774",
        "v2_engine_tests": "c310e0918933004b54921ceb5793965d508caafb65156e066e6134c26f3124d4",
        "v2_scientific": "a6168d9d438fe78dd77f87e48f4005c418c1a218e1811edbe3609a034ece850e",
        "v2_scientific_tests": "945fbfd95d74fc63a4434cb3d7cd51f50a9a319569163325007aab7acaeae7cd",
    }
)
PRELOCK_FIXED_BOUNDARY_ROLES = frozenset(PRELOCK_FIXED_EVIDENCE_SHA256)
PRELOCK_RUNTIME_BOUNDARY_ROLES = frozenset(
    {
        "v2_contracts_source", "v2_runtime_source", "v2_handoff_source",
        "v2_execution_guard", "v2_runtime_tests_source", "v2_runtime_test_launcher",
        "v2_test_authority_manifest", "v2_runtime_tests_manifest",
        "v2_execution_launcher",
    }
)
PRELOCK_DYNAMIC_BOUNDARY_ROLES = frozenset(
    {
        "mechanism_registry_v4", "v2_prelock_boundary_registry",
        "v2_guard_identity_evidence", "v2_fold_projection_target_free",
        "v2_geometry_projection_target_free", "v2_prelock_validator_source",
        "v2_prelock_validator_tests", "v2_prelock_test_launcher",
        "v2_prelock_test_authority_manifest", "v2_prelock_tests_manifest",
    }
)
PRELOCK_STAGE_ROLES = frozenset(
    {
        "v2_s1_contract", "v2_s1_authorization", "v2_s1_bundle_manifest",
        "v2_s1_finalization_root_record", "v2_s2_contract", "v2_s2_authorization",
        "v2_s2_bundle_manifest", "v2_s2_finalization_root_record",
    }
)
PRELOCK_NO_REOPEN_ROLES = frozenset(
    {
        "train_ccr_cohort_arrays", "train_ccr_model_srgb",
        "train_ccr_valid_masks_packbits", "dataset_manifest_train_metadata",
        "cidt_clean_train_table", "keeper_checkpoint_lineage_only",
        "keeper_config_lineage_only",
    }
)
PRELOCK_ARTIFACT_ROLES = frozenset(
    PRELOCK_FIXED_BOUNDARY_ROLES
    | PRELOCK_RUNTIME_BOUNDARY_ROLES
    | PRELOCK_DYNAMIC_BOUNDARY_ROLES
    | PRELOCK_STAGE_ROLES
    | PRELOCK_NO_REOPEN_ROLES
)


class ContractError(ValueError):
    pass


class ReviewedPrelockRequired(ContractError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decode_prelock_document_bytes(
    payload: bytes,
    *,
    role: str,
    state: str,
) -> Mapping[str, object]:
    """Decode the verifier's one exact wrapped-document serialization.

    Pre-lock documents use canonical JSON bytes with no trailing newline.  The
    machine-lock and trusted-runner output formats are separate versioned
    formats and remain newline terminated.
    """
    try:
        value = mapping(json.loads(payload.decode("utf-8")), role)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedPrelockRequired(f"{role} is not JSON") from exc
    if payload != canonical_json_bytes(value):
        raise ReviewedPrelockRequired(
            f"{role} does not use the exact no-newline prelock serializer"
        )
    exact_keys(
        value,
        {"schema_version", "protocol_id", "kind", "state", "payload"},
        role,
    )
    if (
        int(value["schema_version"]) != PRELOCK_DOCUMENT_SCHEMA_VERSION
        or value["protocol_id"] != PROTOCOL_ID
        or value["kind"] != role
        or value["state"] != state
    ):
        raise ReviewedPrelockRequired(f"{role} wrapped document identity differs")
    return mapping(value["payload"], f"{role}.payload")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{label} must be a list")
    return value


def exact_keys(value: Mapping[str, object], expected: Iterable[str], label: str) -> None:
    observed, wanted = set(value), set(expected)
    if observed != wanted:
        raise ContractError(
            f"{label} keys differ: missing={sorted(wanted-observed)}, "
            f"extra={sorted(observed-wanted)}"
        )


def hex_digest(value: object, pattern: re.Pattern[str], label: str) -> str:
    text = str(value)
    if pattern.fullmatch(text) is None:
        raise ContractError(f"{label} is not canonical")
    return text


def canonical_path(
    value: object,
    *,
    label: str = "path",
    must_exist: bool = False,
    reject_alias: bool = True,
) -> str:
    raw = os.path.expanduser(os.fspath(value))
    if not os.path.isabs(raw):
        raise ContractError(f"{label} is not absolute")
    absolute = os.path.abspath(raw)
    resolved = os.path.realpath(absolute)
    if reject_alias and os.path.normcase(absolute) != os.path.normcase(resolved):
        raise ContractError(f"{label} uses a symlink/junction alias")
    if must_exist and not os.path.exists(resolved):
        raise ContractError(f"{label} is missing")
    return resolved


def path_key(value: object) -> str:
    return os.path.normcase(canonical_path(value, reject_alias=False))


def path_within(path: object, root: object) -> bool:
    try:
        return os.path.commonpath((path_key(path), path_key(root))) == path_key(root)
    except ValueError:
        return False


def relative_path(value: object, label: str) -> str:
    text = str(value).replace("\\", "/")
    parsed = PurePosixPath(text)
    if (
        not text
        or parsed.is_absolute()
        or parsed.as_posix() != text
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ContractError(f"{label} is not a canonical relative path")
    return text


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str
    access_class: str

    @classmethod
    def parse(
        cls, value: object, label: str, *, allowed_access: Optional[Iterable[str]] = None
    ) -> "ArtifactRecord":
        row = mapping(value, label)
        exact_keys(row, {"path", "bytes", "sha256", "access_class"}, label)
        size = int(row["bytes"])
        if isinstance(row["bytes"], bool) or size < 0:
            raise ContractError(f"{label}.bytes differs")
        access = str(row["access_class"])
        if allowed_access is not None and access not in set(allowed_access):
            raise ContractError(f"{label}.access_class differs")
        return cls(
            path=canonical_path(row["path"], label=f"{label}.path", must_exist=False),
            bytes=size,
            sha256=hex_digest(row["sha256"], HEX64, f"{label}.sha256"),
            access_class=access,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "access_class": self.access_class,
        }


@dataclass(frozen=True)
class WriteSpec:
    relative_path: str
    media_type: str
    disposition: str

    @classmethod
    def parse(cls, value: object, label: str) -> "WriteSpec":
        row = mapping(value, label)
        exact_keys(row, {"relative_path", "media_type", "disposition"}, label)
        disposition = str(row["disposition"])
        if disposition not in HANDOFF_DISPOSITIONS:
            raise ContractError(f"{label}.disposition differs")
        media_type = str(row["media_type"])
        if media_type not in {"application/json", "application/octet-stream"}:
            raise ContractError(f"{label}.media_type differs")
        return cls(relative_path(row["relative_path"], label), media_type, disposition)


@dataclass(frozen=True)
class ArchiveMemberSpec:
    archive_role: str
    member_name: str
    dtype: str
    shape: Tuple[int, ...]
    uncompressed_bytes: int
    uncompressed_sha256: str
    order: int
    stage: str
    fold: Optional[int]
    target_bearing: bool

    @classmethod
    def parse(cls, value: object, label: str) -> "ArchiveMemberSpec":
        row = mapping(value, label)
        exact_keys(
            row,
            {
                "archive_role", "member_name", "dtype", "shape", "uncompressed_bytes",
                "uncompressed_sha256", "order", "stage", "fold", "target_bearing",
            },
            label,
        )
        member_name = relative_path(row["member_name"], f"{label}.member_name")
        shape = tuple(int(item) for item in sequence(row["shape"], f"{label}.shape"))
        if any(item < 0 for item in shape):
            raise ContractError(f"{label}.shape differs")
        fold = row["fold"]
        if fold is not None and (type(fold) is not int or fold not in range(5)):
            raise ContractError(f"{label}.fold differs")
        stage = str(row["stage"])
        if stage not in {*TARGET_STAGES, "shared"}:
            raise ContractError(f"{label}.stage differs")
        if type(row["target_bearing"]) is not bool:
            raise ContractError(f"{label}.target_bearing differs")
        size, order = int(row["uncompressed_bytes"]), int(row["order"])
        if size <= 0 or order < 0:
            raise ContractError(f"{label} size/order differs")
        return cls(
            archive_role=str(row["archive_role"]),
            member_name=member_name,
            dtype=str(row["dtype"]),
            shape=shape,
            uncompressed_bytes=size,
            uncompressed_sha256=hex_digest(
                row["uncompressed_sha256"], HEX64, f"{label}.uncompressed_sha256"
            ),
            order=order,
            stage=stage,
            fold=fold,
            target_bearing=bool(row["target_bearing"]),
        )


@dataclass(frozen=True)
class Subcontract:
    phase: str
    root: str
    reads: Mapping[str, ArtifactRecord]
    members: Mapping[str, ArchiveMemberSpec]
    targets: Mapping[int, Mapping[str, str]]
    forbidden_member_names: Tuple[str, ...]
    writes: Mapping[str, WriteSpec]
    mandatory_final_roles: Tuple[str, ...]
    cidt_expected: Mapping[int, Mapping[str, object]]
    replay_input_roles: Tuple[str, ...]


@dataclass(frozen=True)
class MachineLock:
    path: str
    file_sha256: str
    lock_id: str
    repository: Mapping[str, object]
    runtime: Mapping[str, object]
    prelock: Mapping[str, object]
    genesis_root: ArtifactRecord
    primary: Subcontract
    replay: Subcontract
    raw: Mapping[str, object]


@dataclass(frozen=True)
class PairRunAuthorization:
    path: str
    bytes: int
    file_sha256: str
    authorization_id: str
    machine_lock_sha256: str
    machine_lock_commit: str
    phases: Mapping[str, Mapping[str, object]]
    raw: Mapping[str, object]


def _parse_artifact_map(value: object, label: str) -> Dict[str, ArtifactRecord]:
    result: Dict[str, ArtifactRecord] = {}
    paths: Dict[str, str] = {}
    for role, raw in mapping(value, label).items():
        record = ArtifactRecord.parse(raw, f"{label}.{role}")
        key = path_key(record.path)
        if key in paths:
            raise ContractError(f"{label} path alias: {paths[key]} and {role}")
        paths[key], result[str(role)] = str(role), record
    return result


def validate_current_prelock_receipt(
    value: object, artifacts: Mapping[str, ArtifactRecord]
) -> Mapping[str, object]:
    receipt = mapping(value, "prelock verifier receipt")
    exact_keys(
        receipt,
        {
            "schema_version", "state", "sequencing_erratum_sha256",
            "verifier_source_sha256", "verifier_tests_sha256", "s1_bundle_sha256",
            "s1_root_sha256", "s2_bundle_sha256", "s2_root_sha256",
            "formal_runs_authorized", "candidate_runs_authorized",
            "validation_runs_authorized", "test_runs_authorized",
            "conditional_xai_runs_authorized", "cache_or_lineage_reopened",
        },
        "prelock verifier receipt",
    )
    expected = {
        "schema_version": 2,
        "state": "s1_s2_finalized_verified_no_execution_authority",
        "sequencing_erratum_sha256": artifacts["v2_prelock_sequencing_erratum"].sha256,
        "verifier_source_sha256": artifacts["v2_prelock_validator_source"].sha256,
        "verifier_tests_sha256": artifacts["v2_prelock_validator_tests"].sha256,
        "s1_bundle_sha256": artifacts["v2_s1_bundle_manifest"].sha256,
        "s1_root_sha256": artifacts["v2_s1_finalization_root_record"].sha256,
        "s2_bundle_sha256": artifacts["v2_s2_bundle_manifest"].sha256,
        "s2_root_sha256": artifacts["v2_s2_finalization_root_record"].sha256,
        "formal_runs_authorized": 0,
        "candidate_runs_authorized": 0,
        "validation_runs_authorized": 0,
        "test_runs_authorized": 0,
        "conditional_xai_runs_authorized": 0,
        "cache_or_lineage_reopened": 0,
    }
    if receipt != expected:
        raise ReviewedPrelockRequired("current prelock verifier receipt differs")
    return receipt


def _parse_subcontract(value: object, phase: str) -> Subcontract:
    row = mapping(value, f"{phase} subcontract")
    exact_keys(
        row,
        {
            "phase", "root", "reads", "archive_members", "targets",
            "forbidden_member_names", "writes", "mandatory_final_roles",
            "cidt_expected", "replay_input_roles",
        },
        f"{phase} subcontract",
    )
    if row["phase"] != phase:
        raise ContractError(f"{phase} subcontract phase differs")
    root = canonical_path(row["root"], label=f"{phase}.root", must_exist=False)
    reads = _parse_artifact_map(row["reads"], f"{phase}.reads")
    members = {
        str(role): ArchiveMemberSpec.parse(raw, f"{phase}.archive_members.{role}")
        for role, raw in mapping(row["archive_members"], f"{phase}.archive_members").items()
    }
    member_names, member_orders = set(), set()
    for role, member in members.items():
        if member.archive_role not in reads:
            raise ContractError(f"{phase} member {role} archive role differs")
        identity = (member.archive_role, member.member_name)
        order_identity = (member.archive_role, member.order)
        if identity in member_names or order_identity in member_orders:
            raise ContractError(f"{phase} archive member name/order alias")
        member_names.add(identity)
        member_orders.add(order_identity)
    forbidden = tuple(
        relative_path(item, f"{phase}.forbidden_member_names")
        for item in sequence(row["forbidden_member_names"], f"{phase}.forbidden_member_names")
    )
    if not forbidden or len(set(forbidden)) != len(forbidden):
        raise ContractError(f"{phase} forbidden all-row member set differs")
    if any(member.member_name in forbidden for member in members.values()):
        raise ContractError(f"{phase} exposes a permanently forbidden member")
    targets_raw = mapping(row["targets"], f"{phase}.targets")
    if set(targets_raw) != {str(fold) for fold in range(5)}:
        raise ContractError(f"{phase} target folds differ")
    targets: Dict[int, Mapping[str, str]] = {}
    used_targets = set()
    for fold in range(5):
        stages = mapping(targets_raw[str(fold)], f"{phase}.targets.{fold}")
        exact_keys(stages, TARGET_STAGES, f"{phase}.targets.{fold}")
        parsed = {}
        for stage in TARGET_STAGES:
            role = str(stages[stage])
            if role in used_targets or role not in members:
                raise ContractError(f"{phase} target role is missing or aliased")
            member = members[role]
            if not member.target_bearing or member.stage != stage or member.fold != fold:
                raise ContractError(f"{phase} target member stage/fold differs")
            used_targets.add(role)
            parsed[stage] = role
        targets[fold] = MappingProxyType(parsed)
    writes = {
        str(role): WriteSpec.parse(raw, f"{phase}.writes.{role}")
        for role, raw in mapping(row["writes"], f"{phase}.writes").items()
    }
    if not MANDATORY_CONTROL_ROLES.issubset(writes):
        raise ContractError(f"{phase} mandatory control writes are missing")
    required_scientific_roles = {
        f"fold_{fold}_{kind}_output"
        for fold in range(5)
        for kind in SCIENTIFIC_OUTPUT_KINDS
    }
    if not required_scientific_roles.issubset(writes):
        raise ContractError(f"{phase} fixed scientific output roles are missing")
    output_path_keys = []
    for spec in writes.values():
        output = canonical_path(Path(root, *PurePosixPath(spec.relative_path).parts), must_exist=False)
        if not path_within(output, root):
            raise ContractError(f"{phase} output escapes its fixed root")
        output_path_keys.append(path_key(output))
    if len(set(output_path_keys)) != len(output_path_keys):
        raise ContractError(f"{phase} output paths alias")
    mandatory = tuple(str(item) for item in sequence(row["mandatory_final_roles"], f"{phase}.mandatory_final_roles"))
    if set(mandatory) != set(writes) or len(mandatory) != len(writes):
        raise ContractError(f"{phase} final physical roles must equal all fixed writes")
    cidt_raw = mapping(row["cidt_expected"], f"{phase}.cidt_expected")
    if set(cidt_raw) != {str(fold) for fold in range(5)}:
        raise ContractError(f"{phase} CIDT folds differ")
    cidt: Dict[int, Mapping[str, object]] = {}
    for fold in range(5):
        outcome = mapping(cidt_raw[str(fold)], f"{phase}.cidt_expected.{fold}")
        exact_keys(outcome, {"rows", "outcome_sha256", "clean", "causal_pass"}, f"{phase}.cidt_expected.{fold}")
        if (
            type(outcome["rows"]) is not int
            or int(outcome["rows"]) <= 0
            or type(outcome["clean"]) is not bool
            or type(outcome["causal_pass"]) is not bool
            or outcome["clean"] is not True
            or outcome["causal_pass"] is not True
        ):
            raise ContractError(f"{phase} CIDT outcome differs")
        hex_digest(outcome["outcome_sha256"], HEX64, f"{phase}.cidt_expected.{fold}.outcome_sha256")
        cidt[fold] = MappingProxyType(dict(outcome))
    replay_inputs = tuple(str(item) for item in sequence(row["replay_input_roles"], f"{phase}.replay_input_roles"))
    if len(set(replay_inputs)) != len(replay_inputs):
        raise ContractError(f"{phase} replay input roles differ")
    return Subcontract(
        phase=phase,
        root=root,
        reads=MappingProxyType(reads),
        members=MappingProxyType(members),
        targets=MappingProxyType(targets),
        forbidden_member_names=forbidden,
        writes=MappingProxyType(writes),
        mandatory_final_roles=mandatory,
        cidt_expected=MappingProxyType(cidt),
        replay_input_roles=replay_inputs,
    )


def parse_machine_lock(value: object, *, path: str, file_digest: str) -> MachineLock:
    row = mapping(value, "machine lock")
    exact_keys(
        row,
        {
            "schema_version", "protocol_id", "state", "lock_id", "repository",
            "runtime", "prelock", "genesis_root", "subcontracts",
        },
        "machine lock",
    )
    if (
        int(row["schema_version"]) != SCHEMA_VERSION
        or row["protocol_id"] != PROTOCOL_ID
        or row["state"] != MACHINE_LOCK_STATE
    ):
        raise ContractError("machine lock identity differs")
    lock_id = hex_digest(row["lock_id"], HEX64, "machine lock id")
    core = dict(row)
    del core["lock_id"]
    if canonical_json_sha256(core) != lock_id:
        raise ContractError("machine lock object hash differs")
    repository = mapping(row["repository"], "repository contract")
    exact_keys(
        repository,
        {
            "root", "branch", "commit", "upstream_ref", "upstream_commit",
            "protected_registry", "protected_untracked_files",
        },
        "repository contract",
    )
    repository_root = canonical_path(
        repository["root"], label="repository.root", must_exist=True
    )
    hex_digest(repository["commit"], HEX40, "repository.commit")
    hex_digest(repository["upstream_commit"], HEX40, "repository.upstream_commit")
    if (
        repository["commit"] != repository["upstream_commit"]
        or not str(repository["branch"])
        or not str(repository["upstream_ref"])
    ):
        raise ContractError("machine-lock source revision was not pushed exactly")
    ArtifactRecord.parse(repository["protected_registry"], "repository.protected_registry")
    protected_untracked = mapping(
        repository["protected_untracked_files"], "repository.protected_untracked_files"
    )
    protected_keys = set()
    for raw_relative, raw_identity in protected_untracked.items():
        relative = relative_path(raw_relative, "protected untracked relative path")
        key = os.path.normcase(relative)
        if key in protected_keys or relative.startswith(".git/"):
            raise ContractError("protected untracked paths alias or enter .git")
        protected_keys.add(key)
        identity = mapping(raw_identity, f"protected untracked {relative}")
        exact_keys(identity, {"bytes", "sha256"}, f"protected untracked {relative}")
        if type(identity["bytes"]) is not int or int(identity["bytes"]) < 0:
            raise ContractError(f"protected untracked size differs for {relative}")
        hex_digest(identity["sha256"], HEX64, f"protected untracked {relative} SHA-256")
    runtime = mapping(row["runtime"], "runtime contract")
    exact_keys(
        runtime,
        {
            "source_root", "s1_machine_identity_roles", "package_files",
            "local_import_closure", "environment", "python_flags",
        },
        "runtime contract",
    )
    source_root = canonical_path(
        runtime["source_root"], label="runtime.source_root", must_exist=True
    )
    if path_key(source_root) != path_key(repository_root):
        raise ContractError("runtime source root differs from the locked repository root")
    if runtime["s1_machine_identity_roles"] != [
        "v2_s1_finalization_root_record", "v2_guard_identity_evidence"
    ]:
        raise ContractError("runtime S1 machine identity roles differ")
    package_files = _parse_artifact_map(runtime["package_files"], "runtime.package_files")
    local_closure = _parse_artifact_map(
        runtime["local_import_closure"], "runtime.local_import_closure"
    )
    if not REQUIRED_PACKAGE_MODULES.issubset(package_files):
        raise ContractError("runtime package hash set is incomplete")
    if not REQUIRED_LOCAL_RUNTIME_MODULES.issubset(local_closure):
        raise ContractError("runtime local import closure is incomplete")
    if mapping(runtime["environment"], "runtime.environment") != DETERMINISTIC_ENVIRONMENT:
        raise ContractError("runtime deterministic environment differs")
    if mapping(runtime["python_flags"], "runtime.python_flags") != {
        "isolated": True,
        "dont_write_bytecode": True,
    }:
        raise ContractError("runtime Python flags differ")
    prelock = mapping(row["prelock"], "prelock contract")
    exact_keys(
        prelock,
        {"artifacts", "verifier_receipt", "boundary_registry_role", "guard_identity_role"},
        "prelock contract",
    )
    prelock_artifacts = _parse_artifact_map(prelock["artifacts"], "prelock.artifacts")
    if set(prelock_artifacts) != set(PRELOCK_ARTIFACT_ROLES):
        raise ReviewedPrelockRequired("prelock immutable artifact role set differs")
    for role, expected_sha256 in PRELOCK_FIXED_EVIDENCE_SHA256.items():
        if prelock_artifacts[role].sha256 != expected_sha256:
            raise ReviewedPrelockRequired(
                f"prelock fixed boundary evidence differs for {role}"
            )
    validate_current_prelock_receipt(prelock["verifier_receipt"], prelock_artifacts)
    if (
        prelock["boundary_registry_role"] != "v2_prelock_boundary_registry"
        or prelock["guard_identity_role"] != "v2_guard_identity_evidence"
    ):
        raise ReviewedPrelockRequired("reviewed prelock boundary roles differ")
    genesis = ArtifactRecord.parse(row["genesis_root"], "genesis_root")
    subcontracts = mapping(row["subcontracts"], "subcontracts")
    exact_keys(subcontracts, PHASES, "subcontracts")
    primary = _parse_subcontract(subcontracts["primary"], "primary")
    replay = _parse_subcontract(subcontracts["replay"], "replay")
    if path_key(primary.root) == path_key(replay.root):
        raise ContractError("primary and replay roots must be physically distinct")
    if path_within(primary.root, replay.root) or path_within(replay.root, primary.root):
        raise ContractError("primary and replay roots must not be nested")
    if path_within(genesis.path, primary.root) or path_within(genesis.path, replay.root):
        raise ContractError("genesis root cannot live inside an execution root")
    if path_within(path, primary.root) or path_within(path, replay.root):
        raise ContractError("machine lock cannot live inside an execution root")
    if primary.replay_input_roles:
        raise ContractError("primary cannot declare replay inputs")
    if set(replay.replay_input_roles) - set(primary.writes):
        raise ContractError("replay inputs are not fixed primary handoff roles")
    if not replay.replay_input_roles:
        raise ContractError("replay requires at least one finalized primary handoff input")
    primary_replay_outputs = {
        role
        for role, spec in primary.writes.items()
        if spec.disposition == "replay_input"
    }
    if primary_replay_outputs != set(replay.replay_input_roles):
        raise ContractError(
            "primary replay-output dispositions differ from replay input roles"
        )
    normalized_subcontracts = []
    for phase in PHASES:
        normalized = dict(mapping(subcontracts[phase], f"subcontracts.{phase}"))
        normalized.pop("phase")
        normalized.pop("root")
        normalized.pop("replay_input_roles")
        normalized_subcontracts.append(canonical_json_sha256(normalized))
    if normalized_subcontracts[0] != normalized_subcontracts[1]:
        raise ContractError("primary and replay scientific subcontracts differ")
    return MachineLock(
        path=path,
        file_sha256=file_digest,
        lock_id=lock_id,
        repository=MappingProxyType(dict(repository)),
        runtime=MappingProxyType(dict(runtime)),
        prelock=MappingProxyType(dict(prelock)),
        genesis_root=genesis,
        primary=primary,
        replay=replay,
        raw=MappingProxyType(dict(row)),
    )


def load_machine_lock(path: object, expected_sha256: object) -> MachineLock:
    canonical = canonical_path(path, label="machine lock path", must_exist=True)
    expected = hex_digest(expected_sha256, HEX64, "expected machine lock SHA-256")
    payload = Path(canonical).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ContractError("machine lock exact bytes differ")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("machine lock is not JSON") from exc
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ContractError("machine lock is not canonical newline-terminated JSON")
    return parse_machine_lock(value, path=canonical, file_digest=observed)


def load_pair_run_authorization(
    path: object,
    expected_sha256: object,
    lock: MachineLock,
) -> PairRunAuthorization:
    """Load the one pre-primary authorization for both fixed phases."""
    canonical = canonical_path(path, label="pair authorization path", must_exist=True)
    expected = hex_digest(expected_sha256, HEX64, "expected pair authorization SHA-256")
    payload = Path(canonical).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ContractError("pair authorization exact bytes differ")
    try:
        value = mapping(json.loads(payload.decode("utf-8")), "pair authorization")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("pair authorization is not JSON") from exc
    if payload != canonical_json_bytes(value) + b"\n":
        raise ContractError("pair authorization is not canonical newline-terminated JSON")
    exact_keys(
        value,
        {
            "schema_version", "protocol_id", "state", "authorization_id",
            "machine_lock_sha256", "machine_lock_commit",
            "reviewed_prelock_receipt_sha256", "boundary_registry_sha256",
            "phases", "execution_contract",
        },
        "pair authorization",
    )
    if (
        int(value["schema_version"]) != SCHEMA_VERSION
        or value["protocol_id"] != PROTOCOL_ID
        or value["state"] != PENDING_AUTHORIZATION_STATE
        or value["machine_lock_sha256"] != lock.file_sha256
    ):
        raise ContractError("pair authorization identity differs")
    authorization_id = hex_digest(
        value["authorization_id"], HEX64, "pair authorization id"
    )
    core = dict(value)
    del core["authorization_id"]
    if canonical_json_sha256(core) != authorization_id:
        raise ContractError("pair authorization object hash differs")
    machine_lock_commit = hex_digest(
        value["machine_lock_commit"], HEX40, "machine-lock commit"
    )
    if machine_lock_commit == lock.repository["commit"]:
        raise ContractError("machine lock was not committed after its source revision")
    if value["reviewed_prelock_receipt_sha256"] != canonical_json_sha256(
        lock.prelock["verifier_receipt"]
    ):
        raise ContractError("pair authorization reviewed-prelock binding differs")
    prelock_artifacts = _parse_artifact_map(lock.prelock["artifacts"], "prelock.artifacts")
    if value["boundary_registry_sha256"] != prelock_artifacts[
        "v2_prelock_boundary_registry"
    ].sha256:
        raise ContractError("pair authorization boundary-registry binding differs")
    phases = mapping(value["phases"], "pair authorization phases")
    exact_keys(phases, PHASES, "pair authorization phases")
    parsed_phases: Dict[str, Mapping[str, object]] = {}
    for phase, contract in (("primary", lock.primary), ("replay", lock.replay)):
        row = mapping(phases[phase], f"pair authorization {phase}")
        exact_keys(
            row,
            {
                "subcontract_sha256", "root", "claim_relative_path", "run_quota",
                "prior_root_role", "prior_handoff_role",
            },
            f"pair authorization {phase}",
        )
        expected_prior = "genesis_root" if phase == "primary" else "primary.final_root"
        expected_handoff = None if phase == "primary" else "primary.handoff"
        if (
            row["subcontract_sha256"]
            != canonical_json_sha256(lock.raw["subcontracts"][phase])
            or path_key(row["root"]) != path_key(contract.root)
            or row["claim_relative_path"] != contract.writes["claim"].relative_path
            or type(row["run_quota"]) is not int
            or row["run_quota"] != 1
            or row["prior_root_role"] != expected_prior
            or row["prior_handoff_role"] != expected_handoff
        ):
            raise ContractError(f"pair authorization {phase} fixed grant differs")
        parsed_phases[phase] = MappingProxyType(dict(row))
    execution = mapping(value["execution_contract"], "pair authorization execution")
    exact_keys(
        execution,
        {
            "launcher_role", "entrypoint", "python_flags", "phase_processes",
            "child_processes", "network_access", "phase_resource_ceilings",
            "stop_rule", "retention_rule",
        },
        "pair authorization execution",
    )
    if (
        execution["launcher_role"] != "v2_execution_launcher"
        or execution["entrypoint"]
        != "trkh.tools.pair_surface_ddf_v2_execution_guard:TrustedPairSurfaceRunner"
        or execution["python_flags"] != ["-I", "-B"]
        or execution["phase_processes"] != {"primary": 1, "replay": 1}
        or type(execution["child_processes"]) is not int
        or execution["child_processes"] != 0
        or execution["network_access"] is not False
        or execution["stop_rule"]
        != "stop_on_first_contract_or_scientific_failure_no_retry"
        or execution["retention_rule"]
        != "retain_primary_replay_roots_and_all_fixed_evidence_no_cleanup"
    ):
        raise ContractError("pair authorization execution policy differs")
    ceilings = mapping(
        execution["phase_resource_ceilings"], "phase resource ceilings"
    )
    exact_keys(ceilings, PHASES, "phase resource ceilings")
    for phase in PHASES:
        ceiling = mapping(ceilings[phase], f"{phase} resource ceiling")
        exact_keys(
            ceiling,
            {"wall_seconds", "disk_bytes", "cpu_threads", "gpu_device"},
            f"{phase} resource ceiling",
        )
        if (
            type(ceiling["wall_seconds"]) is not int
            or int(ceiling["wall_seconds"]) <= 0
            or type(ceiling["disk_bytes"]) is not int
            or int(ceiling["disk_bytes"]) <= 0
            or type(ceiling["cpu_threads"]) is not int
            or int(ceiling["cpu_threads"]) != 1
            or type(ceiling["gpu_device"]) is not int
            or int(ceiling["gpu_device"]) < 0
        ):
            raise ContractError(f"pair authorization {phase} resource ceiling differs")
    if path_within(canonical, lock.primary.root) or path_within(canonical, lock.replay.root):
        raise ContractError("pair authorization cannot live inside an execution root")
    if path_key(canonical) == path_key(lock.path):
        raise ContractError("pair authorization aliases the machine lock")
    return PairRunAuthorization(
        path=canonical,
        bytes=len(payload),
        file_sha256=observed,
        authorization_id=authorization_id,
        machine_lock_sha256=lock.file_sha256,
        machine_lock_commit=machine_lock_commit,
        phases=MappingProxyType(parsed_phases),
        raw=MappingProxyType(dict(value)),
    )


def artifact_record(path: Path, access_class: str) -> Dict[str, object]:
    canonical = Path(canonical_path(path, must_exist=True))
    return ArtifactRecord(
        path=str(canonical),
        bytes=canonical.stat().st_size,
        sha256=file_sha256(canonical),
        access_class=access_class,
    ).as_dict()


def write_exclusive_json(path: object, value: object) -> str:
    canonical = canonical_path(path, reject_alias=False)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        canonical,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()
