from __future__ import annotations

"""Acyclic handoff and final-root construction for trusted sessions."""

from types import MappingProxyType
from typing import Dict, Mapping

from trkh.tools.pair_surface_ddf_v2_contracts import (
    MANDATORY_CONTROL_ROLES,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    canonical_json_sha256,
)
from trkh.tools.pair_surface_ddf_v2_runtime import (
    BoundArtifact,
    RootHandle,
    TrustedRuntimeError,
    TrustedSession,
)


def _artifact_rows(session: TrustedSession, *, exclude: set[str]) -> Dict[str, Mapping[str, object]]:
    rows: Dict[str, Mapping[str, object]] = {}
    for role in sorted(session.contract.writes):
        if role in exclude:
            continue
        if role not in session._artifacts:
            raise TrustedRuntimeError(f"mandatory fixed artifact {role} is missing")
        rows[role] = MappingProxyType(session._artifacts[role].identity())
    return rows


def _validate_inventory(
    session: TrustedSession,
    *,
    expected_roles: set[str],
) -> Mapping[str, Mapping[str, object]]:
    expected_paths = {
        session.contract.writes[role].relative_path: role for role in expected_roles
    }
    if len(expected_paths) != len(expected_roles):
        raise TrustedRuntimeError("fixed physical paths alias")
    inventory = session.physical_inventory()
    if set(inventory) != set(expected_paths):
        raise TrustedRuntimeError(
            f"physical inventory differs: missing={sorted(set(expected_paths)-set(inventory))}, "
            f"extra={sorted(set(inventory)-set(expected_paths))}"
        )
    expected_directories = set()
    for relative in expected_paths:
        parts = relative.split("/")[:-1]
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    observed_directories = set(session.physical_directories())
    if observed_directories != expected_directories:
        raise TrustedRuntimeError(
            "physical directory inventory differs: "
            f"missing={sorted(expected_directories-observed_directories)}, "
            f"extra={sorted(observed_directories-expected_directories)}"
        )
    for relative, role in expected_paths.items():
        if role not in session._artifacts:
            raise TrustedRuntimeError(f"physical role {role} has no session artifact")
        identity = session._artifacts[role].identity()
        observed = inventory[relative]
        if (
            observed["bytes"] != identity["bytes"]
            or observed["sha256"] != identity["sha256"]
            or observed["physical_device"] != identity["physical_device"]
            or observed["physical_inode"] != identity["physical_inode"]
            or observed["physical_link_count"] != identity["physical_link_count"]
        ):
            raise TrustedRuntimeError(f"physical inventory identity differs for {role}")
    return inventory


def finalize_handoff(session: TrustedSession) -> BoundArtifact:
    session._require_running()
    if "handoff" in session._artifacts:
        raise TrustedRuntimeError("handoff is create-once")
    session.prior_root.artifact.read_current_bytes()
    before_handoff = set(session.contract.writes) - {"handoff", "final_root"}
    inventory = _validate_inventory(session, expected_roles=before_handoff)
    members = _artifact_rows(session, exclude={"handoff", "final_root"})
    dispositions = {
        role: session.contract.writes[role].disposition for role in sorted(members)
    }
    replay_inputs = {
        role: members[role]
        for role, disposition in dispositions.items()
        if disposition == "replay_input"
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state": "trusted_exhaustive_primary_or_replay_handoff",
        "phase": session.contract.phase,
        "machine_lock_sha256": session.lock.file_sha256,
        "pair_authorization_id": session.authorization.authorization_id,
        "pair_authorization_sha256": session.authorization.file_sha256,
        "claim_sha256": session.claim_sha256,
        "process_identity_sha256": session.process_identity.digest,
        "prior_root": session.prior_root.artifact.identity(),
        "members": {role: dict(value) for role, value in members.items()},
        "dispositions": dispositions,
        "replay_inputs": {role: dict(value) for role, value in replay_inputs.items()},
        "physical_inventory": {path: dict(value) for path, value in inventory.items()},
    }
    return session._write_control("handoff", payload)


def finalize_session(session: TrustedSession) -> RootHandle:
    session._require_running()
    if session.state == "finalized" or "final_root" in session._artifacts:
        raise TrustedRuntimeError("session finalization is exactly-once")
    if "runtime_attestation" not in session._artifacts:
        raise TrustedRuntimeError("runtime attestation is mandatory")
    if "transition_chain" not in session._artifacts:
        raise TrustedRuntimeError("target transition chain is mandatory")
    if "access_ledger" not in session._artifacts or "member_ledger" not in session._artifacts:
        session.emit_access_and_member_ledgers()
    session._write_control(
        "consumption",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "state": "single_run_authority_consumed",
            "phase": session.contract.phase,
            "claim_sha256": session.claim_sha256,
            "process_identity_sha256": session.process_identity.digest,
        },
    )
    session._write_control(
        "tombstone",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "state": "science_outputs_closed_control_finalization_only",
            "phase": session.contract.phase,
            "claim_sha256": session.claim_sha256,
            "consumption_sha256": session._artifacts["consumption"].sha256,
        },
    )
    handoff = finalize_handoff(session)
    before_root_roles = set(session.contract.writes) - {"final_root"}
    before_root_inventory = _validate_inventory(session, expected_roles=before_root_roles)
    fixed = _artifact_rows(session, exclude={"final_root"})
    if not MANDATORY_CONTROL_ROLES.difference({"final_root"}).issubset(fixed):
        raise TrustedRuntimeError("final root mandatory control roles differ")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state": "trusted_session_final_root",
        "phase": session.contract.phase,
        "machine_lock_sha256": session.lock.file_sha256,
        "pair_authorization_id": session.authorization.authorization_id,
        "pair_authorization_sha256": session.authorization.file_sha256,
        "process_identity": {
            "pid": session.process_identity.pid,
            "start_token": session.process_identity.start_token,
            "digest": session.process_identity.digest,
        },
        "claim": dict(fixed["claim"]),
        "runtime_attestation": dict(fixed["runtime_attestation"]),
        "access_ledger": dict(fixed["access_ledger"]),
        "member_ledger": dict(fixed["member_ledger"]),
        "transition_chain": dict(fixed["transition_chain"]),
        "consumption": dict(fixed["consumption"]),
        "tombstone": dict(fixed["tombstone"]),
        "handoff": handoff.identity(),
        "fixed_role_artifacts": {role: dict(value) for role, value in fixed.items()},
        "physical_inventory_before_root": {
            path: dict(value) for path, value in before_root_inventory.items()
        },
        "expected_final_physical_paths": sorted(
            spec.relative_path for spec in session.contract.writes.values()
        ),
        "fixed_role_set_sha256": canonical_json_sha256(sorted(session.contract.writes)),
    }
    root = session._write_control("final_root", payload)
    final_inventory = _validate_inventory(
        session, expected_roles=set(session.contract.mandatory_final_roles)
    )
    if set(final_inventory) != {
        spec.relative_path for spec in session.contract.writes.values()
    }:
        raise TrustedRuntimeError("final physical inventory is not exact")
    session.state = "finalized"
    session.close_import_finder()
    return RootHandle(
        session=session,
        phase=session.contract.phase,
        artifact=root,
        handoff_artifact=handoff,
        process_identity=session.process_identity,
    )


__all__ = ["finalize_handoff", "finalize_session"]
