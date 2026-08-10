from __future__ import annotations

"""Thin facade for the trusted Pair-Surface DDF fresh-process runner."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from types import MappingProxyType
from typing import Dict, Mapping, Optional

from trkh.tools.pair_surface_ddf_v2_contracts import (
    ArtifactRecord,
    BOUNDARY_REVIEW_COMMIT,
    ContractError,
    HEX64,
    PairRunAuthorization,
    PRELOCK_ARTIFACT_ROLES,
    PRELOCK_DYNAMIC_BOUNDARY_ROLES,
    PRELOCK_FIXED_BOUNDARY_ROLES,
    PRELOCK_NO_REOPEN_ROLES,
    PRELOCK_RUNTIME_BOUNDARY_ROLES,
    PRELOCK_VERIFIER_REVIEWED,
    ReviewedPrelockRequired,
    SCIENTIFIC_FOUNDATION_COMMIT,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_path,
    decode_prelock_document_bytes,
    exact_keys,
    hex_digest,
    load_machine_lock,
    load_pair_run_authorization,
    mapping,
    path_key,
)
from trkh.tools.pair_surface_ddf_v2_handoff import finalize_session
from trkh.tools.pair_surface_ddf_v2_runtime import (
    ExternalBoundArtifact,
    ProcessIdentity,
    RootHandle,
    TrustedRuntimeError,
    TrustedSession,
    current_process_identity,
    load_reviewed_prelock_receipt,
    load_s1_machine_receipt,
)


class LiveRepositoryError(TrustedRuntimeError):
    pass


def _require_reviewed_execution_boundary() -> None:
    if not PRELOCK_VERIFIER_REVIEWED:
        raise ReviewedPrelockRequired(
            "trusted-runner execution is hard-blocked until the rewritten prelock verifier and combined pair authorization are independently reviewed"
        )


def _git(root: str, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", root, *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise LiveRepositoryError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.rstrip("\r\n")


def _read_output_json(record: ArtifactRecord, label: str) -> Mapping[str, object]:
    artifact = ExternalBoundArtifact(record, label, record)
    payload = artifact.read_current_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LiveRepositoryError(f"{label} is not JSON") from exc
    if canonical_json_bytes(value) + b"\n" != payload:
        raise LiveRepositoryError(f"{label} is not canonical JSON")
    return mapping(value, label)


def _read_prelock_document(
    record: ArtifactRecord,
    *,
    role: str,
    state: str,
) -> Mapping[str, object]:
    artifact = ExternalBoundArtifact(record, role, record)
    payload = artifact.read_current_bytes()
    try:
        return decode_prelock_document_bytes(payload, role=role, state=state)
    except ContractError as exc:
        raise LiveRepositoryError(f"{role} wrapped document differs") from exc


def _repository_relative(root: str, path: str, label: str) -> str:
    canonical = canonical_path(path, label=label, must_exist=True)
    try:
        relative = os.path.relpath(canonical, root)
    except ValueError as exc:
        raise LiveRepositoryError(f"{label} is outside the repository") from exc
    if relative == ".." or relative.startswith(f"..{os.sep}"):
        raise LiveRepositoryError(f"{label} is outside the repository")
    return relative.replace("\\", "/")


def validate_live_repository(lock, authorization: PairRunAuthorization) -> None:
    repository = lock.repository
    root = canonical_path(repository["root"], must_exist=True)
    if not Path(root, ".git").exists():
        raise LiveRepositoryError("repository root has no live .git metadata")
    head = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", str(repository["upstream_ref"]))
    if head != upstream:
        raise LiveRepositoryError("live HEAD is not pushed to the locked upstream")
    if _git(root, "symbolic-ref", "--short", "HEAD") != repository["branch"]:
        raise LiveRepositoryError("live branch differs from the machine lock")
    porcelain = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all", "-z"
    )
    entries = [item for item in porcelain.split("\0") if item]
    protected_untracked = mapping(
        repository["protected_untracked_files"], "protected untracked files"
    )
    observed_untracked = set()
    for entry in entries:
        if not entry.startswith("?? "):
            raise LiveRepositoryError(f"live repository has a tracked change: {entry}")
        relative = entry[3:].replace("\\", "/")
        if relative not in protected_untracked:
            raise LiveRepositoryError(f"unprotected untracked path is present: {relative}")
        observed_untracked.add(relative)
    if observed_untracked != set(protected_untracked):
        raise LiveRepositoryError("protected-untracked inventory differs from porcelain")
    for relative, raw_identity in protected_untracked.items():
        identity = mapping(raw_identity, f"protected untracked {relative}")
        record = ArtifactRecord(
            canonical_path(Path(root, *PurePosixPath(relative).parts), must_exist=True),
            int(identity["bytes"]),
            str(identity["sha256"]),
            "protected_untracked_preserve_exact",
        )
        ExternalBoundArtifact(record, f"protected_untracked:{relative}", record)
    machine_lock_relative = _repository_relative(root, lock.path, "machine lock")
    authorization_relative = _repository_relative(
        root, authorization.path, "pair authorization"
    )
    _git(root, "ls-files", "--error-unmatch", "--", machine_lock_relative)
    _git(root, "ls-files", "--error-unmatch", "--", authorization_relative)
    lock_commit = _git(
        root, "log", "-1", "--format=%H", "--", machine_lock_relative
    )
    authorization_commit = _git(
        root, "log", "-1", "--format=%H", "--", authorization_relative
    )
    if lock_commit != authorization.machine_lock_commit:
        raise LiveRepositoryError("machine-lock commit differs from the pair authorization")
    if authorization_commit != head or lock_commit == head:
        raise LiveRepositoryError(
            "pair authorization is not a later committed and pushed artifact"
        )
    _git(root, "merge-base", "--is-ancestor", str(repository["commit"]), lock_commit)
    _git(root, "merge-base", "--is-ancestor", lock_commit, head)
    artifacts = {
        str(role): ArtifactRecord.parse(raw, f"prelock.artifacts.{role}")
        for role, raw in mapping(lock.prelock["artifacts"], "prelock artifacts").items()
    }
    registry_record = ArtifactRecord.parse(
        repository["protected_registry"], "repository.protected_registry"
    )
    expected_registry = artifacts["v2_prelock_boundary_registry"]
    if registry_record != expected_registry:
        raise LiveRepositoryError("protected registry is not the pinned prelock boundary registry")
    registry_relative = _repository_relative(
        root, registry_record.path, "protected boundary registry"
    )
    _git(root, "ls-files", "--error-unmatch", "--", registry_relative)
    registry_commit = _git(
        root, "log", "-1", "--format=%H", "--", registry_relative
    )
    _git(root, "merge-base", "--is-ancestor", registry_commit, lock_commit)
    registry = _read_prelock_document(
        registry_record,
        role="v2_prelock_boundary_registry",
        state="prelock_boundary_registry_v2_finalized",
    )
    exact_keys(
        registry,
        {
            "registry_version", "foundation_commit", "boundary_review_commit",
            "incident_sha256", "hashes", "no_reopen_roles", "downstream_authority",
        },
        "protected boundary registry",
    )
    boundary_roles = (
        PRELOCK_FIXED_BOUNDARY_ROLES
        | PRELOCK_RUNTIME_BOUNDARY_ROLES
        | PRELOCK_DYNAMIC_BOUNDARY_ROLES
    ) - {"v2_prelock_boundary_registry"}
    hashes = mapping(registry["hashes"], "protected boundary hashes")
    if (
        int(registry["registry_version"]) != 2
        or registry["foundation_commit"] != SCIENTIFIC_FOUNDATION_COMMIT
        or registry["boundary_review_commit"] != BOUNDARY_REVIEW_COMMIT
        or registry["incident_sha256"] != artifacts["v2_prelock_process_incident"].sha256
        or set(hashes) != set(boundary_roles)
        or dict(hashes) != {role: artifacts[role].sha256 for role in sorted(boundary_roles)}
        or registry["no_reopen_roles"] != sorted(PRELOCK_NO_REOPEN_ROLES)
    ):
        raise LiveRepositoryError("protected boundary registry projection differs")
    expected_zero_authority = {
        "formal_runs": 0,
        "candidate_runs": 0,
        "validation_runs": 0,
        "test_runs": 0,
        "conditional_xai_runs": 0,
    }
    if registry["downstream_authority"] != expected_zero_authority:
        raise LiveRepositoryError("protected boundary registry authority is not zero")
    if set(artifacts) != set(PRELOCK_ARTIFACT_ROLES):
        raise LiveRepositoryError("protected prelock artifact registry is incomplete")


def _fixed_output_path(contract, role: str) -> str:
    spec = contract.writes[role]
    return canonical_path(
        Path(contract.root, *PurePosixPath(spec.relative_path).parts),
        must_exist=True,
    )


def _external_record(path: str, expected_sha256: str, access_class: str) -> ArtifactRecord:
    expected = hex_digest(expected_sha256, HEX64, "expected external SHA-256")
    canonical = Path(canonical_path(path, must_exist=True))
    payload = canonical.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise TrustedRuntimeError("external finalized artifact SHA-256 differs")
    return ArtifactRecord(str(canonical), len(payload), expected, access_class)


def _validate_bound_output_identity(
    runner: "TrustedPairSurfaceRunner",
    contract,
    role: str,
    value: object,
) -> tuple[Mapping[str, object], ExternalBoundArtifact]:
    identity = mapping(value, f"primary bound output {role}")
    exact_keys(
        identity,
        {
            "role", "path", "bytes", "sha256", "physical_device",
            "physical_inode", "physical_link_count", "session_nonce_sha256",
            "creator_claim_sha256", "creator_process_sha256", "sequence",
            "member_name",
        },
        f"primary bound output {role}",
    )
    expected_path = _fixed_output_path(contract, role)
    if (
        identity["role"] != role
        or path_key(identity["path"]) != path_key(expected_path)
        or identity["member_name"] is not None
        or type(identity["bytes"]) is not int
        or int(identity["bytes"]) < 0
        or type(identity["sequence"]) is not int
        or int(identity["sequence"]) < 0
    ):
        raise TrustedRuntimeError(f"primary bound output identity differs for {role}")
    for field in (
        "sha256", "session_nonce_sha256", "creator_claim_sha256",
        "creator_process_sha256",
    ):
        hex_digest(identity[field], HEX64, f"primary {role}.{field}")
    record = ArtifactRecord(
        expected_path,
        int(identity["bytes"]),
        str(identity["sha256"]),
        "finalized_primary_fixed_output",
    )
    artifact = ExternalBoundArtifact(runner, role, record)
    observed = artifact.identity()
    if (
        observed["physical_device"] != identity["physical_device"]
        or observed["physical_inode"] != identity["physical_inode"]
        or observed["physical_link_count"] != identity["physical_link_count"]
    ):
        raise TrustedRuntimeError(f"primary physical identity differs for {role}")
    return identity, artifact


def _validate_primary_inventory(
    contract,
    inventory: object,
    identities: Mapping[str, Mapping[str, object]],
    roles: set[str],
    label: str,
) -> None:
    rows = mapping(inventory, label)
    expected_paths = {
        contract.writes[role].relative_path: role for role in roles
    }
    if set(rows) != set(expected_paths):
        raise TrustedRuntimeError(f"{label} path set differs")
    for relative, role in expected_paths.items():
        row = mapping(rows[relative], f"{label}.{relative}")
        exact_keys(
            row,
            {
                "bytes", "sha256", "physical_device", "physical_inode",
                "physical_link_count",
            },
            f"{label}.{relative}",
        )
        identity = identities[role]
        if dict(row) != {
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "physical_device": identity["physical_device"],
            "physical_inode": identity["physical_inode"],
            "physical_link_count": identity["physical_link_count"],
        }:
            raise TrustedRuntimeError(f"{label} identity differs for {role}")


def _canonical_role_set_sha256(contract) -> str:
    return canonical_json_sha256(sorted(contract.writes))


def _validate_primary_final_files(contract) -> None:
    expected = {spec.relative_path for spec in contract.writes.values()}
    observed = set()
    for current, directories, files in os.walk(contract.root, followlinks=False):
        for directory in directories:
            if Path(current, directory).is_symlink():
                raise TrustedRuntimeError("primary final root contains a symlink directory")
        for filename in files:
            path = Path(current, filename)
            if path.is_symlink():
                raise TrustedRuntimeError("primary final root contains a symlink file")
            observed.add(path.relative_to(contract.root).as_posix())
    if observed != expected:
        raise TrustedRuntimeError("primary final physical file set differs")


def _validate_primary_controls(
    runner: "TrustedPairSurfaceRunner",
    artifacts: Mapping[str, ExternalBoundArtifact],
    process: Mapping[str, object],
    claim_sha256: str,
    session_nonce_sha256: str,
) -> None:
    lock = runner.lock
    claim = _read_output_json(artifacts["claim"].record, "primary claim")
    exact_keys(
        claim,
        {
            "schema_version", "protocol_id", "state", "phase",
            "machine_lock_sha256", "machine_lock_id", "pair_authorization_id",
            "pair_authorization_sha256", "session_nonce_sha256", "pid",
            "process_start_token", "process_identity_sha256", "prior_root_sha256",
        },
        "primary claim",
    )
    if (
        int(claim["schema_version"]) != 3
        or claim["protocol_id"] != "trkh_pair_surface_ddf_a0_v2"
        or claim["state"] != "session_claimed_o_excl"
        or claim["phase"] != "primary"
        or claim["machine_lock_sha256"] != lock.file_sha256
        or claim["machine_lock_id"] != lock.lock_id
        or claim["pair_authorization_id"] != runner.authorization.authorization_id
        or claim["pair_authorization_sha256"] != runner.authorization.file_sha256
        or claim["session_nonce_sha256"] != session_nonce_sha256
        or claim["pid"] != process["pid"]
        or claim["process_start_token"] != process["start_token"]
        or claim["process_identity_sha256"] != process["digest"]
        or claim["prior_root_sha256"] != lock.genesis_root.sha256
    ):
        raise TrustedRuntimeError("primary claim payload differs")
    runtime = _read_output_json(
        artifacts["runtime_attestation"].record, "primary runtime attestation"
    )
    if (
        runtime.get("state") != "actual_runtime_and_import_closure_attested"
        or runtime.get("machine_lock_sha256") != lock.file_sha256
        or runtime.get("process_identity_sha256") != process["digest"]
    ):
        raise TrustedRuntimeError("primary runtime attestation payload differs")
    access = _read_output_json(artifacts["access_ledger"].record, "primary access ledger")
    members = _read_output_json(artifacts["member_ledger"].record, "primary member ledger")
    if (
        access.get("state") != "exact_access_ledger_finalized"
        or access.get("claim_sha256") != claim_sha256
        or members.get("state") != "typed_archive_member_ledger_finalized"
        or members.get("claim_sha256") != claim_sha256
        or not isinstance(access.get("events"), list)
        or not isinstance(members.get("events"), list)
    ):
        raise TrustedRuntimeError("primary access/member ledger payload differs")
    member_roles = {str(event.get("role")) for event in members["events"]}
    required_target_roles = {
        role for stages in lock.primary.targets.values() for role in stages.values()
    }
    if not required_target_roles.issubset(member_roles):
        raise TrustedRuntimeError("primary member ledger omits a released target")
    chain = _read_output_json(
        artifacts["transition_chain"].record, "primary transition chain"
    )
    transitions = chain.get("transitions")
    if (
        chain.get("state") != "all_five_fold_target_chain_finalized"
        or chain.get("claim_sha256") != claim_sha256
        or not isinstance(transitions, list)
    ):
        raise TrustedRuntimeError("primary transition chain payload differs")
    prior = "0" * 64
    fold_actions = {fold: [] for fold in range(5)}
    barrier_sha256 = None
    for index, stored in enumerate(transitions):
        row = dict(mapping(stored, f"primary transition {index}"))
        digest = str(row.pop("transition_sha256", ""))
        if (
            row.get("index") != index
            or row.get("prior_transition_sha256") != prior
            or row.get("claim_sha256") != claim_sha256
            or row.get("process_identity_sha256") != process["digest"]
            or canonical_json_sha256(row) != digest
        ):
            raise TrustedRuntimeError("primary transition hash/identity differs")
        prior = digest
        if row.get("fold") is None:
            if row.get("action") != "freeze_global_all_five_fold_clean_causal_barrier":
                raise TrustedRuntimeError("primary transition has an unknown global action")
            barrier_sha256 = digest
        else:
            fold = int(row["fold"])
            if fold not in fold_actions:
                raise TrustedRuntimeError("primary transition fold differs")
            fold_actions[fold].append(str(row.get("action")))
    expected_actions = [
        "release_fit_target", "freeze_fit", "release_calibration_target",
        "freeze_calibration", "freeze_held_outputs_before_target",
        "release_held_target", "freeze_held", "release_cidt_target",
        "freeze_cidt", "release_xai_target", "freeze_xai",
    ]
    if (
        any(actions != expected_actions for actions in fold_actions.values())
        or barrier_sha256 is None
        or chain.get("global_xai_barrier_sha256") != barrier_sha256
        or chain.get("final_transition_sha256") != prior
    ):
        raise TrustedRuntimeError("primary finalized target-transition semantics differ")
    consumption = _read_output_json(
        artifacts["consumption"].record, "primary consumption"
    )
    tombstone = _read_output_json(artifacts["tombstone"].record, "primary tombstone")
    if (
        consumption.get("state") != "single_run_authority_consumed"
        or consumption.get("claim_sha256") != claim_sha256
        or consumption.get("process_identity_sha256") != process["digest"]
        or tombstone.get("state") != "science_outputs_closed_control_finalization_only"
        or tombstone.get("claim_sha256") != claim_sha256
        or tombstone.get("consumption_sha256") != artifacts["consumption"].sha256
    ):
        raise TrustedRuntimeError("primary consumption/tombstone payload differs")


def _parse_primary_root(
    runner: "TrustedPairSurfaceRunner",
    expected_sha256: str,
) -> tuple[RootHandle, Mapping[str, ExternalBoundArtifact]]:
    lock = runner.lock
    root_path = _fixed_output_path(lock.primary, "final_root")
    root_record = _external_record(root_path, expected_sha256, "finalized_primary_root")
    root_artifact = ExternalBoundArtifact(runner, "primary_final_root", root_record)
    root = _read_output_json(root_record, "primary final root")
    exact_keys(
        root,
        {
            "schema_version", "protocol_id", "state", "phase",
            "machine_lock_sha256", "pair_authorization_id",
            "pair_authorization_sha256", "process_identity", "claim",
            "runtime_attestation", "access_ledger", "member_ledger",
            "transition_chain", "consumption", "tombstone", "handoff",
            "fixed_role_artifacts", "physical_inventory_before_root",
            "expected_final_physical_paths", "fixed_role_set_sha256",
        },
        "primary final root",
    )
    if (
        int(root.get("schema_version", -1)) != 3
        or root.get("protocol_id") != "trkh_pair_surface_ddf_a0_v2"
        or root.get("state") != "trusted_session_final_root"
        or root.get("phase") != "primary"
        or root.get("machine_lock_sha256") != lock.file_sha256
        or root.get("pair_authorization_id") != runner.authorization.authorization_id
        or root.get("pair_authorization_sha256") != runner.authorization.file_sha256
    ):
        raise TrustedRuntimeError("primary final root identity differs")
    fixed_raw = mapping(root["fixed_role_artifacts"], "primary fixed-role artifacts")
    expected_fixed_roles = set(lock.primary.writes) - {"final_root"}
    if set(fixed_raw) != expected_fixed_roles:
        raise TrustedRuntimeError("primary fixed-role artifact set differs")
    fixed: Dict[str, Mapping[str, object]] = {}
    fixed_artifacts: Dict[str, ExternalBoundArtifact] = {}
    for role in sorted(expected_fixed_roles):
        identity, artifact = _validate_bound_output_identity(
            runner, lock.primary, role, fixed_raw[role]
        )
        fixed[role], fixed_artifacts[role] = identity, artifact
    if root["expected_final_physical_paths"] != sorted(
        spec.relative_path for spec in lock.primary.writes.values()
    ) or root["fixed_role_set_sha256"] != _canonical_role_set_sha256(lock.primary):
        raise TrustedRuntimeError("primary final physical-role projection differs")
    _validate_primary_inventory(
        lock.primary,
        root["physical_inventory_before_root"],
        fixed,
        expected_fixed_roles,
        "primary inventory before root",
    )
    _validate_primary_final_files(lock.primary)
    process = mapping(root.get("process_identity"), "primary process identity")
    exact_keys(process, {"pid", "start_token", "digest"}, "primary process identity")
    expected_process_digest = hashlib.sha256(
        f"{int(process['pid'])}\0{process['start_token']}".encode("utf-8")
    ).hexdigest()
    if process["digest"] != expected_process_digest:
        raise TrustedRuntimeError("primary process identity digest differs")
    claim_identity = fixed["claim"]
    claim_sha256 = str(claim_identity["sha256"])
    session_nonce = str(claim_identity["session_nonce_sha256"])
    for role, identity in fixed.items():
        if (
            identity["creator_process_sha256"] != process["digest"]
            or identity["session_nonce_sha256"] != session_nonce
            or identity["creator_claim_sha256"] != claim_sha256
        ):
            raise TrustedRuntimeError(f"primary creator receipt differs for {role}")
    for role in (
        "claim", "runtime_attestation", "access_ledger", "member_ledger",
        "transition_chain", "consumption", "tombstone", "handoff",
    ):
        if root[role] != fixed[role]:
            raise TrustedRuntimeError(f"primary root control binding differs for {role}")
    _validate_primary_controls(
        runner,
        fixed_artifacts,
        process,
        claim_sha256,
        session_nonce,
    )
    handoff_identity = mapping(root.get("handoff"), "primary handoff identity")
    handoff_path = _fixed_output_path(lock.primary, "handoff")
    if path_key(handoff_identity.get("path")) != path_key(handoff_path):
        raise TrustedRuntimeError("primary handoff path differs")
    if dict(handoff_identity) != dict(fixed["handoff"]):
        raise TrustedRuntimeError("primary root/handoff identity differs")
    handoff_artifact = fixed_artifacts["handoff"]
    handoff_record = handoff_artifact.record
    handoff = _read_output_json(handoff_record, "primary handoff")
    exact_keys(
        handoff,
        {
            "schema_version", "protocol_id", "state", "phase",
            "machine_lock_sha256", "pair_authorization_id",
            "pair_authorization_sha256", "claim_sha256",
            "process_identity_sha256", "prior_root", "members",
            "dispositions", "replay_inputs", "physical_inventory",
        },
        "primary handoff",
    )
    if (
        int(handoff.get("schema_version", -1)) != 3
        or handoff.get("protocol_id") != "trkh_pair_surface_ddf_a0_v2"
        or handoff.get("state") != "trusted_exhaustive_primary_or_replay_handoff"
        or handoff.get("phase") != "primary"
        or handoff.get("machine_lock_sha256") != lock.file_sha256
        or handoff.get("pair_authorization_id") != runner.authorization.authorization_id
        or handoff.get("pair_authorization_sha256") != runner.authorization.file_sha256
        or handoff.get("claim_sha256") != claim_sha256
        or handoff.get("process_identity_sha256") != process["digest"]
    ):
        raise TrustedRuntimeError("primary handoff identity differs")
    handoff_members = mapping(handoff["members"], "primary handoff members")
    handoff_roles = set(lock.primary.writes) - {"handoff", "final_root"}
    if set(handoff_members) != handoff_roles or any(
        dict(mapping(handoff_members[role], f"handoff member {role}")) != dict(fixed[role])
        for role in handoff_roles
    ):
        raise TrustedRuntimeError("primary handoff exhaustive members differ")
    dispositions = {
        role: lock.primary.writes[role].disposition for role in sorted(handoff_roles)
    }
    if handoff["dispositions"] != dispositions:
        raise TrustedRuntimeError("primary handoff dispositions differ")
    prior = mapping(handoff["prior_root"], "primary handoff prior root")
    if (
        prior.get("path") != lock.genesis_root.path
        or prior.get("bytes") != lock.genesis_root.bytes
        or prior.get("sha256") != lock.genesis_root.sha256
    ):
        raise TrustedRuntimeError("primary handoff genesis-root binding differs")
    _validate_primary_inventory(
        lock.primary,
        handoff["physical_inventory"],
        fixed,
        handoff_roles,
        "primary handoff inventory",
    )
    replay_rows = mapping(handoff.get("replay_inputs"), "primary replay inputs")
    if set(replay_rows) != set(lock.replay.replay_input_roles):
        raise TrustedRuntimeError("primary handoff replay-input role set differs")
    replay_inputs: Dict[str, ExternalBoundArtifact] = {}
    for role in lock.replay.replay_input_roles:
        identity = mapping(replay_rows[role], f"primary replay input {role}")
        if dict(identity) != dict(fixed[role]):
            raise TrustedRuntimeError(f"primary replay input binding differs for {role}")
        expected_path = _fixed_output_path(lock.primary, role)
        if path_key(identity.get("path")) != path_key(expected_path):
            raise TrustedRuntimeError(f"primary replay input path differs for {role}")
        record = ArtifactRecord(
            expected_path,
            int(identity["bytes"]),
            hex_digest(identity["sha256"], HEX64, f"primary replay input {role} SHA-256"),
            "finalized_primary_replay_input",
        )
        replay_inputs[role] = ExternalBoundArtifact(runner, role, record)
    process_identity = ProcessIdentity(
        int(process["pid"]), str(process["start_token"]), str(process["digest"])
    )
    return (
        RootHandle(None, "primary", root_artifact, handoff_artifact, process_identity),
        MappingProxyType(replay_inputs),
    )


class TrustedPairSurfaceRunner:
    """Loads one exact lock and internally owns one fixed phase session."""

    def __init__(self, lock, authorization: PairRunAuthorization, mode: str) -> None:
        _require_reviewed_execution_boundary()
        self.lock = lock
        self.authorization = authorization
        self.mode = mode
        self.prelock_receipt = load_reviewed_prelock_receipt(lock)
        self.s1_receipt = load_s1_machine_receipt(lock)
        self._issued = False
        self._completed = False
        self._prior_root: Optional[RootHandle] = None
        self._replay_inputs: Mapping[str, ExternalBoundArtifact] = MappingProxyType({})
        authorization_record = ArtifactRecord(
            authorization.path,
            authorization.bytes,
            authorization.file_sha256,
            "committed_pushed_pair_authorization",
        )
        self.authorization_artifact = ExternalBoundArtifact(
            self, "pair_run_authorization", authorization_record
        )
        self._reviewed_boundary = True

    @classmethod
    def open_primary(
        cls,
        machine_lock_path: object,
        expected_machine_lock_sha256: object,
        pair_authorization_path: object,
        expected_pair_authorization_sha256: object,
    ) -> "TrustedPairSurfaceRunner":
        _require_reviewed_execution_boundary()
        lock = load_machine_lock(machine_lock_path, expected_machine_lock_sha256)
        authorization = load_pair_run_authorization(
            pair_authorization_path, expected_pair_authorization_sha256, lock
        )
        validate_live_repository(lock, authorization)
        runner = cls(lock, authorization, "primary")
        genesis = ExternalBoundArtifact(runner, "genesis_root", lock.genesis_root)
        runner._prior_root = RootHandle(
            None,
            "genesis",
            genesis,
            None,
            ProcessIdentity(0, "prelock-finalized", lock.genesis_root.sha256),
        )
        return runner

    @classmethod
    def open_replay(
        cls,
        machine_lock_path: object,
        expected_machine_lock_sha256: object,
        pair_authorization_path: object,
        expected_pair_authorization_sha256: object,
        expected_primary_root_sha256: object,
    ) -> "TrustedPairSurfaceRunner":
        _require_reviewed_execution_boundary()
        lock = load_machine_lock(machine_lock_path, expected_machine_lock_sha256)
        authorization = load_pair_run_authorization(
            pair_authorization_path, expected_pair_authorization_sha256, lock
        )
        validate_live_repository(lock, authorization)
        runner = cls(lock, authorization, "replay")
        prior, replay_inputs = _parse_primary_root(runner, str(expected_primary_root_sha256))
        if prior.process_identity.digest == current_process_identity().digest:
            raise TrustedRuntimeError("replay runner requires a fresh process identity")
        runner._prior_root = prior
        runner._replay_inputs = replay_inputs
        return runner

    def issue(self) -> TrustedSession:
        if getattr(self, "_reviewed_boundary", False) is not True:
            raise ReviewedPrelockRequired("runner issue requires the reviewed execution boundary")
        if self._issued or self._completed:
            raise TrustedRuntimeError("runner issue is exactly-once")
        if self._prior_root is None:
            raise TrustedRuntimeError("runner prior root is not bound")
        contract = self.lock.primary if self.mode == "primary" else self.lock.replay
        session = TrustedSession(
            self,
            self.lock,
            contract,
            self._prior_root,
            self.prelock_receipt,
            self.s1_receipt,
            self._replay_inputs,
        )
        self._issued = True
        return session

    def finalize(self, session: TrustedSession) -> RootHandle:
        if self._completed or not self._issued or session.runner is not self:
            raise TrustedRuntimeError("runner finalization session differs or was consumed")
        root = finalize_session(session)
        self._completed = True
        return root


def build_machine_lock(*args, **kwargs):
    del args, kwargs
    if not PRELOCK_VERIFIER_REVIEWED:
        raise ReviewedPrelockRequired(
            "machine-lock builder is hard-blocked until the final prelock verifier receipt is independently reviewed"
        )
    raise ReviewedPrelockRequired("reviewed prelock receipt artifact was not supplied")


def build_pending_authorization(*args, **kwargs):
    del args, kwargs
    raise ReviewedPrelockRequired(
        "authorization issuer is hard-blocked until a reviewed machine-lock receipt exists"
    )


__all__ = [
    "LiveRepositoryError",
    "TrustedPairSurfaceRunner",
    "TrustedRuntimeError",
    "build_machine_lock",
    "build_pending_authorization",
    "validate_live_repository",
]
