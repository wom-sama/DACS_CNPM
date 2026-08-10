from __future__ import annotations

"""Synthetic-only red-team tests for the schema-v3 trusted runner.

These tests deliberately exercise ordinary Python objects.  They verify a
hash-pinned correctness protocol; they do not claim that Python is a hostile
code sandbox or that an in-process token can be made unforgeable.
"""

from dataclasses import replace
import hashlib
import inspect
import importlib.machinery
import io
import os
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest

from trkh.tools import pair_surface_ddf_v2_execution_guard as guard
from trkh.tools import pair_surface_ddf_v2_runtime as runtime
from trkh.tools.pair_surface_ddf_v2_contracts import (
    ArchiveMemberSpec,
    ArtifactRecord,
    ContractError,
    MANDATORY_CONTROL_ROLES,
    MachineLock,
    PairRunAuthorization,
    PRELOCK_ARTIFACT_ROLES,
    PRELOCK_VERIFIER_REVIEWED,
    ReviewedPrelockRequired,
    Subcontract,
    WriteSpec,
    _parse_subcontract,
    artifact_record,
    canonical_json_bytes,
    load_machine_lock,
)
from trkh.tools.pair_surface_ddf_v2_execution_guard import (
    LiveRepositoryError,
    TrustedPairSurfaceRunner,
    build_machine_lock,
    build_pending_authorization,
    validate_live_repository,
)
from trkh.tools.pair_surface_ddf_v2_handoff import finalize_handoff
from trkh.tools.pair_surface_ddf_v2_runtime import (
    CIDTOutcome,
    ExternalBoundArtifact,
    ProcessIdentity,
    ReviewedPrelockReceipt,
    RootHandle,
    S1MachineIdentityReceipt,
    TrustedRuntimeError,
    attest_runtime,
    current_process_identity,
)


SYNTHETIC_PRELOCK_V1 = MappingProxyType(
    {
        "schema_version": 1,
        "state": "immutable_synthetic_v1_dependency_no_live_import",
        "source_sha256": hashlib.sha256(b"synthetic-prelock-v1\n").hexdigest(),
    }
)
STAGES = ("fit", "calibration", "held", "cidt", "xai")
OUTPUT_KINDS = ("fit", "calibration", "held_prediction", "held", "cidt", "xai")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _record(path: Path, access_class: str = "synthetic_read") -> ArtifactRecord:
    return ArtifactRecord.parse(artifact_record(path, access_class), path.name)


def _npy_bytes(fold: int, stage_index: int) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray([fold, stage_index], dtype=np.int64), allow_pickle=False)
    return stream.getvalue()


def _write_archive(
    path: Path,
) -> tuple[dict[str, ArchiveMemberSpec], dict[int, dict[str, str]]]:
    members: dict[str, ArchiveMemberSpec] = {}
    targets: dict[int, dict[str, str]] = {fold: {} for fold in range(5)}
    order = 0
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for fold in range(5):
            for stage_index, stage in enumerate(STAGES):
                role = f"fold_{fold}_{stage}_target"
                member_name = f"targets/fold_{fold}_{stage}.npy"
                payload = _npy_bytes(fold, stage_index)
                archive.writestr(member_name, payload)
                members[role] = ArchiveMemberSpec(
                    archive_role="target_archive",
                    member_name=member_name,
                    dtype=np.dtype(np.int64).str,
                    shape=(2,),
                    uncompressed_bytes=len(payload),
                    uncompressed_sha256=hashlib.sha256(payload).hexdigest(),
                    order=order,
                    stage=stage,
                    fold=fold,
                    target_bearing=True,
                )
                targets[fold][stage] = role
                order += 1
    return members, targets


def _writes() -> dict[str, WriteSpec]:
    result = {
        role: WriteSpec(
            relative_path=f"control/{role}.json",
            media_type="application/json",
            disposition="control_evidence",
        )
        for role in MANDATORY_CONTROL_ROLES
    }
    for fold in range(5):
        for kind in OUTPUT_KINDS:
            role = f"fold_{fold}_{kind}_output"
            result[role] = WriteSpec(
                relative_path=f"science/fold_{fold}/{kind}.json",
                media_type="application/json",
                disposition="compare_output",
            )
    result["primary_replay_seed"] = WriteSpec(
        relative_path="science/replay_seed.json",
        media_type="application/json",
        disposition="replay_input",
    )
    return result


def _subcontract(
    root: Path,
    phase: str,
    archive_record: ArtifactRecord,
    members: dict[str, ArchiveMemberSpec],
    targets: dict[int, dict[str, str]],
) -> Subcontract:
    writes = _writes()
    cidt = {
        fold: MappingProxyType(
            {
                "rows": 2,
                "outcome_sha256": hashlib.sha256(f"cidt-{fold}".encode()).hexdigest(),
                "clean": True,
                "causal_pass": True,
            }
        )
        for fold in range(5)
    }
    return Subcontract(
        phase=phase,
        root=str(root.resolve()),
        reads=MappingProxyType({"target_archive": archive_record}),
        members=MappingProxyType(members),
        targets=MappingProxyType(
            {fold: MappingProxyType(stages) for fold, stages in targets.items()}
        ),
        forbidden_member_names=("targets/all_rows.npy",),
        writes=MappingProxyType(writes),
        mandatory_final_roles=tuple(sorted(writes)),
        cidt_expected=MappingProxyType(cidt),
        replay_input_roles=() if phase == "primary" else ("primary_replay_seed",),
    )


def _build_lock(tmp_path: Path, suffix: str):
    base = tmp_path / suffix
    base.mkdir()
    archive_path = base / "targets.zip"
    members, targets = _write_archive(archive_path)
    archive_record = _record(archive_path, "typed_archive_read")
    primary = _subcontract(base / "primary", "primary", archive_record, members, targets)
    replay_members = dict(members)
    replay = _subcontract(base / "replay", "replay", archive_record, replay_members, targets)
    source_root = base / "empty_runtime_source"
    source_root.mkdir()
    genesis_path = base / "genesis.json"
    _write_json(genesis_path, {"state": "synthetic_finalized_genesis"})
    lock = MachineLock(
        path=str((base / "machine_lock.json").resolve()),
        file_sha256="b" * 64,
        lock_id="a" * 64,
        repository=MappingProxyType({}),
        runtime=MappingProxyType(
            {
                "source_root": str(source_root.resolve()),
                "s1_machine_identity_roles": [
                    "v2_s1_finalization_root_record",
                    "v2_guard_identity_evidence",
                ],
                "package_files": {},
                "local_import_closure": {},
                "environment": {},
                "python_flags": {"isolated": True, "dont_write_bytecode": True},
            }
        ),
        prelock=MappingProxyType({}),
        genesis_root=_record(genesis_path, "finalized_genesis"),
        primary=primary,
        replay=replay,
        raw=MappingProxyType({}),
    )
    return lock, members, replay_members


def _receipts(lock: MachineLock):
    prelock = ReviewedPrelockReceipt(
        lock.lock_id,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
    )
    s1 = S1MachineIdentityReceipt(
        lock.lock_id,
        "5" * 64,
        "6" * 64,
        MappingProxyType({"state": "synthetic_s1_machine_identity"}),
    )
    return prelock, s1


def _raw_runner(
    lock: MachineLock,
    mode: str,
    prior_root: RootHandle | None = None,
) -> TrustedPairSurfaceRunner:
    runner = object.__new__(TrustedPairSurfaceRunner)
    runner.lock = lock
    runner.mode = mode
    authorization_path = Path(lock.path).parent / "pair_authorization.json"
    if not authorization_path.exists():
        _write_json(authorization_path, {"state": "synthetic_pair_authorization"})
    authorization_record = _record(
        authorization_path, "committed_pushed_pair_authorization"
    )
    phases = MappingProxyType(
        {
            phase: MappingProxyType(
                {
                    "root": getattr(lock, phase).root,
                    "run_quota": 1,
                }
            )
            for phase in ("primary", "replay")
        }
    )
    runner.authorization = PairRunAuthorization(
        path=authorization_record.path,
        bytes=authorization_record.bytes,
        file_sha256=authorization_record.sha256,
        authorization_id="8" * 64,
        machine_lock_sha256=lock.file_sha256,
        machine_lock_commit="8" * 40,
        phases=phases,
        raw=MappingProxyType(
            {
                "execution_contract": {
                    "phase_resource_ceilings": {
                        phase: {
                            "wall_seconds": 3600,
                            "disk_bytes": 100_000_000,
                            "cpu_threads": 1,
                            "gpu_device": 0,
                        }
                        for phase in ("primary", "replay")
                    }
                }
            }
        ),
    )
    runner.authorization_artifact = ExternalBoundArtifact(
        runner, "pair_run_authorization", authorization_record
    )
    runner.prelock_receipt, runner.s1_receipt = _receipts(lock)
    runner._reviewed_boundary = True
    runner._issued = False
    runner._completed = False
    runner._replay_inputs = MappingProxyType({})
    if prior_root is None:
        genesis = ExternalBoundArtifact(runner, "genesis_root", lock.genesis_root)
        prior_root = RootHandle(
            None,
            "genesis",
            genesis,
            None,
            ProcessIdentity(0, "synthetic-finalized-prelock", "7" * 64),
        )
    runner._prior_root = prior_root
    return runner


def _context(tmp_path: Path, suffix: str = "case"):
    lock, members, _ = _build_lock(tmp_path, suffix)
    runner = _raw_runner(lock, "primary")
    session = runner.issue()
    return SimpleNamespace(
        lock=lock,
        members=members,
        runner=runner,
        session=session,
        prelock=runner.prelock_receipt,
    )


def _start(ctx, *, synthetic_runtime_ready: bool = True) -> None:
    ctx.session.claim()
    ctx.session.start_run()
    if synthetic_runtime_ready:
        ctx.session._write_control(
            "runtime_attestation",
            {"state": "synthetic_runtime_attestation_for_unit_boundary"},
        )


def _output_receipt(ctx, fold: int, kind: str, *, cidt: CIDTOutcome | None = None):
    role = f"fold_{fold}_{kind}_output"
    artifact = ctx.session.write_json_output(role, {"fold": fold, "kind": kind})
    return ctx.session.make_scientific_receipt(kind, fold, artifact, cidt=cidt)


def _advance_to_cidt_release(ctx, fold: int) -> None:
    targets = ctx.session.targets
    targets.release_fit(fold, ctx.prelock)
    ctx.session.read_archive_member(f"fold_{fold}_fit_target")
    targets.freeze_fit(_output_receipt(ctx, fold, "fit"))
    targets.release_calibration(fold)
    ctx.session.read_archive_member(f"fold_{fold}_calibration_target")
    targets.freeze_calibration(_output_receipt(ctx, fold, "calibration"))
    targets.freeze_held_outputs(_output_receipt(ctx, fold, "held_prediction"))
    targets.release_held(fold)
    ctx.session.read_archive_member(f"fold_{fold}_held_target")
    targets.freeze_held(_output_receipt(ctx, fold, "held"))
    targets.release_cidt(fold)


def _freeze_cidt(ctx, fold: int, outcome: CIDTOutcome | None = None) -> None:
    ctx.session.read_archive_member(f"fold_{fold}_cidt_target")
    if outcome is None:
        expected = ctx.session.contract.cidt_expected[fold]
        outcome = CIDTOutcome(fold=fold, **dict(expected))
    ctx.session.targets.freeze_cidt(_output_receipt(ctx, fold, "cidt", cidt=outcome))


def _complete_science(ctx) -> None:
    ctx.session.write_json_output("primary_replay_seed", {"seed": "fixed-primary-output"})
    for fold in range(5):
        _advance_to_cidt_release(ctx, fold)
        _freeze_cidt(ctx, fold)
    for fold in range(5):
        ctx.session.targets.release_xai(fold)
        ctx.session.read_archive_member(f"fold_{fold}_xai_target")
        ctx.session.targets.freeze_xai(_output_receipt(ctx, fold, "xai"))
    ctx.session.targets.finalize()


def _subcontract_raw(contract: Subcontract) -> dict[str, object]:
    return {
        "phase": contract.phase,
        "root": contract.root,
        "reads": {role: record.as_dict() for role, record in contract.reads.items()},
        "archive_members": {
            role: {
                "archive_role": spec.archive_role,
                "member_name": spec.member_name,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "uncompressed_bytes": spec.uncompressed_bytes,
                "uncompressed_sha256": spec.uncompressed_sha256,
                "order": spec.order,
                "stage": spec.stage,
                "fold": spec.fold,
                "target_bearing": spec.target_bearing,
            }
            for role, spec in contract.members.items()
        },
        "targets": {
            str(fold): dict(stages) for fold, stages in contract.targets.items()
        },
        "forbidden_member_names": list(contract.forbidden_member_names),
        "writes": {
            role: {
                "relative_path": spec.relative_path,
                "media_type": spec.media_type,
                "disposition": spec.disposition,
            }
            for role, spec in contract.writes.items()
        },
        "mandatory_final_roles": list(contract.mandatory_final_roles),
        "cidt_expected": {
            str(fold): dict(value) for fold, value in contract.cidt_expected.items()
        },
        "replay_input_roles": list(contract.replay_input_roles),
    }


def test_01_runner_facade_has_no_caller_ledger_or_policy_surface():
    assert set(inspect.signature(TrustedPairSurfaceRunner.open_primary).parameters) == {
        "machine_lock_path",
        "expected_machine_lock_sha256",
        "pair_authorization_path",
        "expected_pair_authorization_sha256",
    }
    assert set(inspect.signature(TrustedPairSurfaceRunner.issue).parameters) == {"self"}
    assert not hasattr(runtime, "ExactAccessLedger")
    assert not hasattr(runtime.TrustedSession, "bind_written")


def test_02_claim_precedes_every_sensitive_operation(tmp_path):
    ctx = _context(tmp_path)
    with pytest.raises(TrustedRuntimeError, match="claim and run start"):
        ctx.session.read_artifact("target_archive")
    with pytest.raises(TrustedRuntimeError, match="claim and run start"):
        ctx.session.targets.release_fit(0, ctx.prelock)
    with pytest.raises(TrustedRuntimeError, match="claim and run start"):
        ctx.session.write_json_output("fold_0_fit_output", {})
    with pytest.raises(TrustedRuntimeError, match="claim and run start"):
        finalize_handoff(ctx.session)
    with pytest.raises(TrustedRuntimeError, match="claim and run start"):
        ctx.runner.finalize(ctx.session)


def test_03_issue_claim_and_start_are_exactly_once(tmp_path):
    ctx = _context(tmp_path)
    with pytest.raises(TrustedRuntimeError, match="issue is exactly-once"):
        ctx.runner.issue()
    ctx.session.claim()
    with pytest.raises(TrustedRuntimeError, match="claim is exactly-once"):
        ctx.session.claim()
    ctx.session.start_run()
    with pytest.raises(TrustedRuntimeError, match="run start is exactly-once"):
        ctx.session.start_run()


def test_04_roots_and_claim_paths_must_be_new_physical_locations(tmp_path):
    ctx = _context(tmp_path, "root")
    with pytest.raises(FileExistsError, match="session root must be new"):
        _raw_runner(ctx.lock, "primary").issue()
    other = _context(tmp_path, "claim")
    claim_path = Path(other.session._role_path("claim"))
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_bytes(b"preexisting")
    with pytest.raises(FileExistsError):
        other.session.claim()


def test_05_outputs_are_internal_o_excl_create_once_only(tmp_path):
    ctx = _context(tmp_path)
    _start(ctx)
    role = "fold_0_fit_output"
    output_path = Path(ctx.session._role_path(role))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"attacker-preexisting")
    with pytest.raises(FileExistsError):
        ctx.session.write_output(role, b"trusted")
    written = ctx.session.write_output("fold_0_calibration_output", b"first")
    assert written.read_current_bytes() == b"first"
    with pytest.raises(TrustedRuntimeError, match="already written"):
        ctx.session.write_output("fold_0_calibration_output", b"second")
    with pytest.raises(TrustedRuntimeError, match="control roles"):
        ctx.session.write_json_output("claim", {})


def _bad_member_case(tmp_path: Path, suffix: str, mutation) -> None:
    ctx = _context(tmp_path, suffix)
    _start(ctx)
    ctx.session.targets.release_fit(0, ctx.prelock)
    role = "fold_0_fit_target"
    ctx.members[role] = mutation(ctx.members[role])
    with pytest.raises(TrustedRuntimeError):
        ctx.session.read_archive_member(role)


def test_06_typed_archive_reader_checks_name_order_dtype_shape_size_and_hash(tmp_path):
    _bad_member_case(tmp_path, "order", lambda spec: replace(spec, order=1))
    _bad_member_case(tmp_path, "dtype", lambda spec: replace(spec, dtype="<f4"))
    _bad_member_case(tmp_path, "shape", lambda spec: replace(spec, shape=(1, 2)))
    _bad_member_case(
        tmp_path,
        "size",
        lambda spec: replace(spec, uncompressed_bytes=spec.uncompressed_bytes + 1),
    )
    _bad_member_case(
        tmp_path,
        "hash",
        lambda spec: replace(spec, uncompressed_sha256="0" * 64),
    )


def test_07_contract_rejects_stage_mismatch_and_all_row_member(tmp_path):
    ctx = _context(tmp_path)
    raw = _subcontract_raw(ctx.lock.primary)
    raw["archive_members"]["fold_0_fit_target"]["stage"] = "calibration"
    with pytest.raises(ContractError, match="stage/fold"):
        _parse_subcontract(raw, "primary")
    raw = _subcontract_raw(ctx.lock.primary)
    raw["archive_members"]["fold_0_fit_target"]["member_name"] = "targets/all_rows.npy"
    with pytest.raises(ContractError, match="permanently forbidden"):
        _parse_subcontract(raw, "primary")
    _start(ctx)
    with pytest.raises(TrustedRuntimeError, match="not fixed"):
        ctx.session.read_archive_member("all_rows")


def test_08_artifacts_bind_session_creator_physical_identity_and_current_bytes(tmp_path):
    first = _context(tmp_path, "first")
    second = _context(tmp_path, "second")
    _start(first)
    _start(second)
    artifact = first.session.write_output("fold_0_fit_output", b"bound")
    with pytest.raises(TrustedRuntimeError, match="another trusted session"):
        second.session.make_scientific_receipt("fit", 0, artifact)
    Path(artifact.path).write_bytes(b"mutated")
    with pytest.raises(TrustedRuntimeError, match="physical identity/current bytes"):
        artifact.read_current_bytes()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"hardlink")
    hardlink_path = Path(second.session._role_path("fold_0_fit_output"))
    hardlink_path.parent.mkdir(parents=True, exist_ok=True)
    os.link(outside, hardlink_path)
    with pytest.raises(FileExistsError):
        second.session.write_output("fold_0_fit_output", b"new")


def test_09_live_repository_checks_head_upstream_porcelain_and_registry(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    pinned = tmp_path / "registry.json"
    _write_json(pinned, {"synthetic": True})
    record = _record(pinned, "protected_registry")
    artifacts = {role: record.as_dict() for role in PRELOCK_ARTIFACT_ROLES}
    repository = {
        "root": str(repo.resolve()),
        "branch": "main",
        "commit": "1" * 40,
        "upstream_ref": "refs/remotes/origin/main",
        "upstream_commit": "2" * 40,
        "protected_registry": record.as_dict(),
        "protected_untracked_files": {},
    }
    machine_lock_path = repo / "machine_lock.json"
    authorization_path = repo / "pair_authorization.json"
    _write_json(machine_lock_path, {"lock": True})
    _write_json(authorization_path, {"authorization": True})
    lock = SimpleNamespace(
        path=str(machine_lock_path.resolve()),
        repository=repository,
        prelock={"artifacts": artifacts},
    )
    authorization = SimpleNamespace(
        path=str(authorization_path.resolve()), machine_lock_commit="4" * 40
    )
    monkeypatch.setattr(
        guard,
        "_git",
        lambda root, *args: "0" * 40 if args[-1] == "HEAD" else "3" * 40,
    )
    with pytest.raises(LiveRepositoryError, match="not pushed"):
        validate_live_repository(lock, authorization)

    def dirty_git(root, *args):
        del root
        return {
            ("rev-parse", "HEAD"): "1" * 40,
            ("rev-parse", "refs/remotes/origin/main"): "1" * 40,
            ("symbolic-ref", "--short", "HEAD"): "main",
            ("status", "--porcelain=v1", "--untracked-files=all", "-z"):
                "?? dirty.txt\0",
        }[args]

    monkeypatch.setattr(guard, "_git", dirty_git)
    with pytest.raises(LiveRepositoryError, match="unprotected untracked"):
        validate_live_repository(lock, authorization)

    def clean_git(root, *args):
        del root
        if args[0] in {"status", "ls-files", "merge-base"}:
            return ""
        if args[:2] == ("rev-parse", "HEAD") or args[:2] == (
            "rev-parse",
            "refs/remotes/origin/main",
        ):
            return "3" * 40
        if args[:3] == ("symbolic-ref", "--short", "HEAD"):
            return "main"
        if args[:3] == ("log", "-1", "--format=%H"):
            return "4" * 40 if args[-1].endswith("machine_lock.json") else "3" * 40
        raise AssertionError(args)

    different = tmp_path / "different_registry.json"
    _write_json(different, {"different": True})
    lock.repository = {**repository, "protected_registry": _record(different).as_dict()}
    monkeypatch.setattr(guard, "_git", clean_git)
    with pytest.raises(LiveRepositoryError, match="pinned prelock boundary registry"):
        validate_live_repository(lock, authorization)


def test_10_target_fsm_accepts_only_typed_runner_receipts_and_exact_cidt(tmp_path):
    ctx = _context(tmp_path, "typed")
    _start(ctx)
    with pytest.raises(TrustedRuntimeError, match="runner-loaded reviewed prelock"):
        ctx.session.targets.release_fit(0, {})
    ctx.session.targets.release_fit(0, ctx.prelock)
    ctx.session.read_archive_member("fold_0_fit_target")
    output = ctx.session.write_json_output("fold_0_fit_output", {})
    with pytest.raises(TrustedRuntimeError, match="typed scientific receipt"):
        ctx.session.targets.freeze_fit({})
    ctx.session.targets.freeze_fit(ctx.session.make_scientific_receipt("fit", 0, output))

    cidt = _context(tmp_path, "cidt")
    _start(cidt)
    _advance_to_cidt_release(cidt, 0)
    cidt.session.read_archive_member("fold_0_cidt_target")
    wrong = CIDTOutcome(0, 2, "0" * 64, True, True)
    receipt = _output_receipt(cidt, 0, "cidt", cidt=wrong)
    with pytest.raises(TrustedRuntimeError, match="typed outcome differs"):
        cidt.session.targets.freeze_cidt(receipt)


def test_11_xai_waits_for_all_five_clean_and_causal_cidt_outcomes(tmp_path):
    ctx = _context(tmp_path)
    _start(ctx)
    _advance_to_cidt_release(ctx, 0)
    _freeze_cidt(ctx, 0)
    with pytest.raises(TrustedRuntimeError, match="all five CIDT clean\+causal"):
        ctx.session.targets.release_xai(0)


def _primary_root_handle(tmp_path: Path, runner, *, handoff: bool, same_process: bool):
    root_path = tmp_path / "synthetic_primary_root.json"
    _write_json(root_path, {"state": "synthetic_finalized_primary_root"})
    root_artifact = ExternalBoundArtifact(runner, "primary_final_root", _record(root_path))
    handoff_artifact = None
    if handoff:
        handoff_path = tmp_path / "synthetic_primary_handoff.json"
        _write_json(handoff_path, {"state": "synthetic_finalized_primary_handoff"})
        handoff_artifact = ExternalBoundArtifact(runner, "primary_handoff", _record(handoff_path))
    process = current_process_identity() if same_process else ProcessIdentity(9, "other", "9" * 64)
    return RootHandle(None, "primary", root_artifact, handoff_artifact, process)


def test_12_replay_requires_finalized_handoff_exact_inputs_and_fresh_process(tmp_path):
    lock, _, _ = _build_lock(tmp_path, "missing_handoff")
    runner = _raw_runner(lock, "replay")
    prior = _primary_root_handle(tmp_path / "missing_handoff", runner, handoff=False, same_process=False)
    runner._prior_root = prior
    session = runner.issue()
    with pytest.raises(TrustedRuntimeError, match="finalized primary handoff"):
        session.claim()

    lock, _, _ = _build_lock(tmp_path, "same_process")
    runner = _raw_runner(lock, "replay")
    prior = _primary_root_handle(tmp_path / "same_process", runner, handoff=True, same_process=True)
    runner._prior_root = prior
    session = runner.issue()
    with pytest.raises(TrustedRuntimeError, match="exact finalized primary replay-input set"):
        session.claim()
    runner._replay_inputs = MappingProxyType(
        {"primary_replay_seed": prior.handoff_artifact}
    )
    session.replay_inputs = runner._replay_inputs
    with pytest.raises(TrustedRuntimeError, match="fresh process identity"):
        session.claim()


def test_13_handoff_and_final_root_require_exact_physical_inventory(tmp_path):
    incomplete = _context(tmp_path, "incomplete")
    _start(incomplete)
    with pytest.raises(TrustedRuntimeError, match="mandatory fixed artifact"):
        finalize_handoff(incomplete.session)

    complete = _context(tmp_path, "extra")
    _start(complete, synthetic_runtime_ready=False)
    complete.session.attest_runtime()
    _complete_science(complete)
    extra = Path(complete.session.contract.root) / "unexpected.bin"
    extra.write_bytes(b"not in the fixed inventory")
    try:
        with pytest.raises(TrustedRuntimeError, match="physical inventory differs"):
            complete.runner.finalize(complete.session)
    finally:
        complete.session.close_import_finder()


def test_14_runtime_attestation_rejects_fake_sys_modules_package(monkeypatch, tmp_path):
    ctx = _context(tmp_path)
    fake_source = tmp_path / "fake_package.py"
    fake_source.write_text("VALUE = 1\n", encoding="utf-8")
    fake_name = "synthetic_fake_runtime_package"
    runtime_contract = dict(ctx.lock.runtime)
    runtime_contract["package_files"] = {fake_name: _record(fake_source).as_dict()}
    forged_lock = replace(ctx.lock, runtime=MappingProxyType(runtime_contract))
    forged = ModuleType(fake_name)
    forged.__file__ = str(fake_source)
    forged.__spec__ = importlib.machinery.ModuleSpec(fake_name, loader=None)
    forged.__spec__.origin = str(fake_source)
    forged.__loader__ = None
    monkeypatch.setitem(sys.modules, fake_name, forged)
    with pytest.raises(TrustedRuntimeError, match="no real origin"):
        attest_runtime(forged_lock, ctx.runner.s1_receipt)


def test_15_machine_lock_loader_requires_exact_canonical_bytes_and_hash(tmp_path):
    path = tmp_path / "machine_lock.json"
    _write_json(path, {})
    with pytest.raises(ContractError, match="exact bytes differ"):
        load_machine_lock(path, "0" * 64)
    path.write_bytes(b'{ "noncanonical": true }\n')
    noncanonical_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ContractError, match="not canonical"):
        load_machine_lock(path, noncanonical_digest)


def test_52_immutable_synthetic_v1_dependency_keeps_builders_hard_blocked():
    assert PRELOCK_VERIFIER_REVIEWED is False
    assert "trkh.tools.pair_surface_ddf_v2_prelock_validator" not in sys.modules
    with pytest.raises(ReviewedPrelockRequired, match="independently reviewed"):
        build_machine_lock(prelock_verifier=SYNTHETIC_PRELOCK_V1)
    with pytest.raises(ReviewedPrelockRequired, match="reviewed machine-lock receipt"):
        build_pending_authorization(machine_lock=SYNTHETIC_PRELOCK_V1)


def test_99_positive_trusted_primary_lifecycle_finalizes_exact_root(tmp_path):
    ctx = _context(tmp_path)
    _start(ctx, synthetic_runtime_ready=False)
    ctx.session.attest_runtime()
    _complete_science(ctx)
    root = ctx.runner.finalize(ctx.session)
    assert root.phase == "primary"
    assert root.handoff_artifact is not None
    assert ctx.session.state == "finalized"
    assert set(ctx.session.physical_inventory()) == {
        spec.relative_path for spec in ctx.session.contract.writes.values()
    }
    assert set(ctx.session._artifacts) == set(ctx.session.contract.mandatory_final_roles)
    with pytest.raises(TrustedRuntimeError, match="consumed"):
        ctx.runner.finalize(ctx.session)
