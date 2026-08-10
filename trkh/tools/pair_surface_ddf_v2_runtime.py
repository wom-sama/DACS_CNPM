from __future__ import annotations

"""Trusted, hash-pinned correctness runtime.

This module is deliberately not a hostile-code sandbox.  Its objects are
ordinary Python references used by a reviewed fresh-process runner.  Safety
comes from fixed contracts, live byte/identity checks, create-once outputs and
independent replay evidence, not from claims that Python tokens are unforgeable.
"""

from dataclasses import dataclass
import ctypes
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat as statlib
import sys
import time
from types import MappingProxyType, ModuleType
from typing import Dict, Mapping, Optional, Sequence, Tuple
import zipfile

from trkh.tools.pair_surface_ddf_v2_contracts import (
    ArtifactRecord,
    ContractError,
    MANDATORY_CONTROL_ROLES,
    MachineLock,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    Subcontract,
    TARGET_STAGES,
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_path,
    decode_prelock_document_bytes,
    mapping,
    path_key,
    path_within,
)


class TrustedRuntimeError(RuntimeError):
    pass


def _windows_start_token() -> str:
    class FILETIME(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    values = [FILETIME() for _ in range(4)]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, *([ctypes.POINTER(FILETIME)] * 4)]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(), *(ctypes.byref(value) for value in values)
    ):
        raise TrustedRuntimeError("cannot capture process start identity")
    creation = (int(values[0].high) << 32) | int(values[0].low)
    return f"win-filetime:{creation}"


def _process_start_token() -> str:
    if os.name == "nt":
        return _windows_start_token()
    try:
        with open("/proc/self/stat", "rb") as handle:
            fields = handle.read().split()
        if len(fields) >= 22:
            return f"proc-start:{fields[21].decode('ascii')}"
    except (OSError, UnicodeError):
        pass
    return f"fallback:{os.getpid()}:{id(sys.modules)}"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_token: str
    digest: str


_PROCESS_PID = os.getpid()
_PROCESS_START = _process_start_token()


def current_process_identity() -> ProcessIdentity:
    if os.getpid() != _PROCESS_PID:
        raise TrustedRuntimeError("runner object crossed a process boundary")
    digest = hashlib.sha256(f"{_PROCESS_PID}\0{_PROCESS_START}".encode("utf-8")).hexdigest()
    return ProcessIdentity(_PROCESS_PID, _PROCESS_START, digest)


@dataclass(frozen=True)
class PhysicalIdentity:
    device: int
    inode: int
    link_count: int


@dataclass(frozen=True)
class ReviewedPrelockReceipt:
    lock_id: str
    verifier_receipt_sha256: str
    boundary_registry_sha256: str
    guard_identity_sha256: str
    target_contract_sha256: str


@dataclass(frozen=True)
class S1MachineIdentityReceipt:
    lock_id: str
    artifact_sha256: str
    machine_identity_sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class CIDTOutcome:
    fold: int
    rows: int
    outcome_sha256: str
    clean: bool
    causal_pass: bool


@dataclass(frozen=True)
class ScientificReceipt:
    session: "TrustedSession"
    kind: str
    fold: int
    artifact: "BoundArtifact"
    cidt: Optional[CIDTOutcome] = None


class BoundArtifact:
    """A session-owned byte identity; revalidated whenever consumed."""

    __slots__ = (
        "session", "role", "path", "bytes", "sha256", "physical_identity",
        "session_nonce", "creator_claim_sha256", "creator_process_sha256",
        "sequence", "member_name", "_member_payload_sha256",
    )

    def __init__(
        self,
        *,
        session: "TrustedSession",
        role: str,
        path: str,
        payload: bytes,
        physical_identity: PhysicalIdentity,
        sequence: int,
        member_name: Optional[str] = None,
    ) -> None:
        self.session = session
        self.role = role
        self.path = path
        self.bytes = len(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.physical_identity = physical_identity
        self.session_nonce = session.session_nonce
        self.creator_claim_sha256 = session.claim_sha256
        self.creator_process_sha256 = session.process_identity.digest
        self.sequence = sequence
        self.member_name = member_name
        self._member_payload_sha256 = self.sha256 if member_name is not None else None

    def identity(self) -> Dict[str, object]:
        self.session.assert_current(self)
        return {
            "role": self.role,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "physical_device": self.physical_identity.device,
            "physical_inode": self.physical_identity.inode,
            "physical_link_count": self.physical_identity.link_count,
            "session_nonce_sha256": hashlib.sha256(self.session_nonce.encode("ascii")).hexdigest(),
            "creator_claim_sha256": self.creator_claim_sha256,
            "creator_process_sha256": self.creator_process_sha256,
            "sequence": self.sequence,
            "member_name": self.member_name,
        }

    def read_current_bytes(self) -> bytes:
        return self.session.assert_current(self)


class ExternalBoundArtifact:
    """Exact finalized control bytes loaded by the trusted runner before a claim."""

    def __init__(self, owner: object, role: str, record: ArtifactRecord) -> None:
        payload, identity = _read_record(record, role)
        self.owner = owner
        self.role = role
        self.record = record
        self.path = record.path
        self.bytes = len(payload)
        self.sha256 = record.sha256
        self.physical_identity = identity

    def read_current_bytes(self) -> bytes:
        payload, identity = _read_record(self.record, self.role)
        if identity != self.physical_identity:
            raise TrustedRuntimeError(f"external control {self.role} physical identity differs")
        return payload

    def identity(self) -> Dict[str, object]:
        self.read_current_bytes()
        return {
            "role": self.role,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "physical_device": self.physical_identity.device,
            "physical_inode": self.physical_identity.inode,
            "physical_link_count": self.physical_identity.link_count,
        }


@dataclass(frozen=True)
class RootHandle:
    session: Optional["TrustedSession"]
    phase: str
    artifact: object
    handoff_artifact: Optional[object]
    process_identity: ProcessIdentity


def _read_file(path: str) -> Tuple[bytes, PhysicalIdentity]:
    canonical = canonical_path(path, must_exist=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        stat = os.fstat(descriptor)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    if not statlib.S_ISREG(stat.st_mode) or int(getattr(stat, "st_nlink", 1)) != 1:
        raise TrustedRuntimeError("trusted artifact is not a single-link regular file")
    return b"".join(chunks), PhysicalIdentity(
        int(stat.st_dev), int(stat.st_ino), int(getattr(stat, "st_nlink", 1))
    )


def _read_record(record: ArtifactRecord, label: str) -> Tuple[bytes, PhysicalIdentity]:
    payload, identity = _read_file(record.path)
    if len(payload) != record.bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise TrustedRuntimeError(f"{label} exact bytes differ")
    return payload, identity


def _prelock_document_record(
    record: ArtifactRecord,
    role: str,
    state: str,
) -> Mapping[str, object]:
    payload, _ = _read_record(record, role)
    try:
        return decode_prelock_document_bytes(payload, role=role, state=state)
    except ContractError as exc:
        raise TrustedRuntimeError(f"{role} wrapped document differs") from exc


def load_reviewed_prelock_receipt(lock: MachineLock) -> ReviewedPrelockReceipt:
    artifacts = _artifact_map(lock.prelock["artifacts"], "prelock.artifacts")
    receipt = mapping(lock.prelock["verifier_receipt"], "prelock verifier receipt")
    expected_target = canonical_json_sha256(
        {phase: lock.raw["subcontracts"][phase]["targets"] for phase in ("primary", "replay")}
    )
    return ReviewedPrelockReceipt(
        lock.lock_id,
        canonical_json_sha256(receipt),
        artifacts["v2_prelock_boundary_registry"].sha256,
        artifacts["v2_guard_identity_evidence"].sha256,
        expected_target,
    )


def load_s1_machine_receipt(lock: MachineLock) -> S1MachineIdentityReceipt:
    artifacts = _artifact_map(lock.prelock["artifacts"], "prelock.artifacts")
    s1_record = artifacts["v2_s1_finalization_root_record"]
    guard_record = artifacts["v2_guard_identity_evidence"]
    s1_payload = _prelock_document_record(
        s1_record, "v2_s1_finalization_root_record", "s1_outer_root_finalized"
    )
    guard_payload = _prelock_document_record(
        guard_record, "v2_guard_identity_evidence", "guard_identity_evidence_finalized"
    )
    identity_sha = canonical_json_sha256(
        {
            "s1_finalization_root_sha256": s1_record.sha256,
            "guard_identity_evidence_sha256": guard_record.sha256,
        }
    )
    return S1MachineIdentityReceipt(
        lock.lock_id,
        s1_record.sha256,
        identity_sha,
        MappingProxyType({"s1_root": dict(s1_payload), "guard_identity": dict(guard_payload)}),
    )


class PinnedLocalImportFinder:
    """Prospective correctness check for local modules, not a security boundary."""

    def __init__(self, source_root: str, pins: Mapping[str, ArtifactRecord]) -> None:
        self.source_root = source_root
        self.pins = pins

    def find_spec(self, fullname: str, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
            return spec
        try:
            origin = canonical_path(spec.origin, must_exist=True)
        except ContractError:
            return spec
        if path_within(origin, self.source_root):
            if fullname not in self.pins:
                raise ImportError(f"local import {fullname} is absent from the locked closure")
            record = self.pins[fullname]
            if path_key(origin) != path_key(record.path) or hashlib.sha256(Path(origin).read_bytes()).hexdigest() != record.sha256:
                raise ImportError(f"local import {fullname} differs from its pin")
        return spec


def _artifact_map(value: object, label: str) -> Dict[str, ArtifactRecord]:
    return {
        str(role): ArtifactRecord.parse(raw, f"{label}.{role}")
        for role, raw in mapping(value, label).items()
    }


def attest_runtime(
    lock: MachineLock,
    s1_receipt: S1MachineIdentityReceipt,
) -> Tuple[Mapping[str, object], PinnedLocalImportFinder]:
    runtime = lock.runtime
    expected_flags = mapping(runtime["python_flags"], "runtime.python_flags")
    if expected_flags != {"isolated": True, "dont_write_bytecode": True}:
        raise TrustedRuntimeError("runtime Python flag contract differs")
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise TrustedRuntimeError("runner requires Python -I -B")
    environment = mapping(runtime["environment"], "runtime.environment")
    if {str(key): os.environ.get(str(key)) for key in environment} != dict(environment):
        raise TrustedRuntimeError("runtime environment differs")
    package_pins = _artifact_map(runtime["package_files"], "runtime.package_files")
    for module_name, record in package_pins.items():
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            raise TrustedRuntimeError(f"package module {module_name} is missing or fake")
        spec = getattr(module, "__spec__", None)
        module_file = getattr(module, "__file__", None)
        loader = getattr(spec, "loader", None) if spec is not None else None
        if (
            spec is None
            or not spec.origin
            or module_file is None
            or not isinstance(loader, importlib.machinery.SourceFileLoader)
            or getattr(module, "__loader__", None) is not loader
        ):
            raise TrustedRuntimeError(f"package module {module_name} has no real origin")
        if path_key(spec.origin) != path_key(record.path) or path_key(module_file) != path_key(record.path):
            raise TrustedRuntimeError(f"package module {module_name} origin differs")
        _read_record(record, f"package module {module_name}")
    source_root = canonical_path(runtime["source_root"], must_exist=True)
    closure = _artifact_map(runtime["local_import_closure"], "runtime.local_import_closure")
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            local = path_within(module_file, source_root)
        except (ContractError, TypeError):
            local = False
        if local:
            if module_name not in closure or not isinstance(module, ModuleType):
                raise TrustedRuntimeError(f"loaded local module {module_name} is outside the closure")
            record = closure[module_name]
            spec = getattr(module, "__spec__", None)
            loader = getattr(spec, "loader", None) if spec is not None else None
            if (
                spec is None
                or not isinstance(loader, importlib.machinery.SourceFileLoader)
                or getattr(module, "__loader__", None) is not loader
                or path_key(spec.origin) != path_key(record.path)
                or path_key(module_file) != path_key(record.path)
            ):
                raise TrustedRuntimeError(f"loaded local module {module_name} origin differs")
            _read_record(record, f"local module {module_name}")
    if "torch" in package_pins:
        torch = sys.modules["torch"]
        try:
            if (
                not torch.are_deterministic_algorithms_enabled()
                or torch.get_num_threads() != 1
                or torch.get_num_interop_threads() != 1
                or torch.backends.cudnn.benchmark
                or not torch.backends.cudnn.deterministic
                or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32
            ):
                raise TrustedRuntimeError("actual PyTorch determinism differs")
        except AttributeError as exc:
            raise TrustedRuntimeError("actual PyTorch module surface differs") from exc
    if "threadpoolctl" in package_pins:
        module = sys.modules["threadpoolctl"]
        for pool in module.threadpool_info():
            if int(pool.get("num_threads", -1)) != 1:
                raise TrustedRuntimeError("native threadpool is not single-threaded")
    finder = PinnedLocalImportFinder(source_root, MappingProxyType(closure))
    sys.meta_path.insert(0, finder)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state": "actual_runtime_and_import_closure_attested",
        "machine_lock_sha256": lock.file_sha256,
        "s1_machine_receipt_sha256": s1_receipt.artifact_sha256,
        "s1_machine_identity_sha256": s1_receipt.machine_identity_sha256,
        "process_identity_sha256": current_process_identity().digest,
        "package_sha256": {name: pin.sha256 for name, pin in sorted(package_pins.items())},
        "local_import_closure_sha256": {name: pin.sha256 for name, pin in sorted(closure.items())},
        "environment": dict(environment),
        "python_flags": dict(expected_flags),
    }
    return MappingProxyType(payload), finder


class TargetReleaseFSM:
    def __init__(self, session: "TrustedSession") -> None:
        self.session = session
        self.states = {fold: "locked" for fold in range(5)}
        self.transitions: list[Dict[str, object]] = []
        self.prior_sha256 = "0" * 64
        self._xai_barrier_sha256: Optional[str] = None

    def _transition(
        self,
        fold: Optional[int],
        action: str,
        before: str,
        after: str,
        evidence: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        self.session._require_science_ready()
        if fold is not None and self.states[fold] != before:
            raise TrustedRuntimeError(f"illegal target transition from {self.states[fold]}")
        row = {
            "schema_version": SCHEMA_VERSION,
            "index": len(self.transitions),
            "fold": fold,
            "action": action,
            "from_state": before,
            "to_state": after,
            "prior_transition_sha256": self.prior_sha256,
            "claim_sha256": self.session.claim_sha256,
            "process_identity_sha256": self.session.process_identity.digest,
            "evidence": [dict(item) for item in evidence],
        }
        digest = canonical_json_sha256(row)
        row["transition_sha256"] = digest
        self.transitions.append(row)
        self.prior_sha256 = digest
        if fold is not None:
            self.states[fold] = after
        return MappingProxyType(json.loads(canonical_json_bytes(row).decode("utf-8")))

    def _release(self, fold: int, stage: str, before: str) -> Mapping[str, object]:
        role = self.session.contract.targets[fold][stage]
        result = self._transition(
            fold,
            f"release_{stage}_target",
            before,
            f"{stage}_released",
            [{"member_role": role, "member_sha256": self.session.contract.members[role].uncompressed_sha256}],
        )
        self.session._released_members.add(role)
        return result

    def _freeze(
        self,
        receipt: ScientificReceipt,
        stage: str,
        after: str,
    ) -> Mapping[str, object]:
        self.session.assert_scientific_receipt(receipt, stage)
        fold = receipt.fold
        role = self.session.contract.targets[fold][stage]
        if role not in self.session._read_member_roles:
            raise TrustedRuntimeError(f"{stage} target was not read through the typed member reader")
        if receipt.artifact.sequence <= self.session._member_read_sequences[role]:
            raise TrustedRuntimeError(
                f"{stage} scientific artifact predates its typed target read"
            )
        result = self._transition(
            fold,
            f"freeze_{stage}",
            f"{stage}_released",
            after,
            [receipt.artifact.identity()],
        )
        self.session._released_members.discard(role)
        return result

    def release_fit(self, fold: int, receipt: ReviewedPrelockReceipt) -> Mapping[str, object]:
        self.session.assert_prelock_receipt(receipt)
        return self._release(fold, "fit", "locked")

    def freeze_fit(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        return self._freeze(receipt, "fit", "fit_frozen")

    def release_calibration(self, fold: int) -> Mapping[str, object]:
        return self._release(fold, "calibration", "fit_frozen")

    def freeze_calibration(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        return self._freeze(receipt, "calibration", "calibration_frozen")

    def freeze_held_outputs(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        self.session.assert_scientific_receipt(receipt, "held_prediction")
        return self._transition(
            receipt.fold,
            "freeze_held_outputs_before_target",
            "calibration_frozen",
            "held_outputs_frozen",
            [receipt.artifact.identity()],
        )

    def release_held(self, fold: int) -> Mapping[str, object]:
        return self._release(fold, "held", "held_outputs_frozen")

    def freeze_held(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        return self._freeze(receipt, "held", "held_frozen")

    def release_cidt(self, fold: int) -> Mapping[str, object]:
        return self._release(fold, "cidt", "held_frozen")

    def freeze_cidt(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        self.session.assert_scientific_receipt(receipt, "cidt")
        if (
            not isinstance(receipt.cidt, CIDTOutcome)
            or receipt.cidt.fold != receipt.fold
        ):
            raise TrustedRuntimeError("CIDT requires a typed exact outcome")
        expected = self.session.contract.cidt_expected[receipt.fold]
        observed = {
            "rows": receipt.cidt.rows,
            "outcome_sha256": receipt.cidt.outcome_sha256,
            "clean": receipt.cidt.clean,
            "causal_pass": receipt.cidt.causal_pass,
        }
        if observed != dict(expected):
            raise TrustedRuntimeError("CIDT typed outcome differs from the lock")
        return self._freeze(receipt, "cidt", "cidt_clean_causal_frozen")

    def _ensure_global_xai_barrier(self) -> None:
        if self._xai_barrier_sha256 is not None:
            return
        if any(state != "cidt_clean_causal_frozen" for state in self.states.values()):
            raise TrustedRuntimeError("XAI requires all five CIDT clean+causal outcomes")
        row = self._transition(
            None,
            "freeze_global_all_five_fold_clean_causal_barrier",
            "all_five_cidt_pending",
            "global_xai_released",
            [
                {"fold": fold, **dict(self.session.contract.cidt_expected[fold])}
                for fold in range(5)
            ],
        )
        self._xai_barrier_sha256 = str(row["transition_sha256"])

    def release_xai(self, fold: int) -> Mapping[str, object]:
        self._ensure_global_xai_barrier()
        return self._release(fold, "xai", "cidt_clean_causal_frozen")

    def freeze_xai(self, receipt: ScientificReceipt) -> Mapping[str, object]:
        return self._freeze(receipt, "xai", "complete")

    def finalize(self) -> BoundArtifact:
        self.session._require_science_ready()
        if any(state != "complete" for state in self.states.values()):
            raise TrustedRuntimeError("target transition chain is incomplete")
        prior = "0" * 64
        for index, stored in enumerate(self.transitions):
            row = dict(stored)
            digest = str(row.pop("transition_sha256"))
            if row["index"] != index or row["prior_transition_sha256"] != prior:
                raise TrustedRuntimeError("target transition ordering differs")
            if canonical_json_sha256(row) != digest:
                raise TrustedRuntimeError("target transition hash differs")
            prior = digest
        artifact = self.session._write_control(
            "transition_chain",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "state": "all_five_fold_target_chain_finalized",
                "claim_sha256": self.session.claim_sha256,
                "global_xai_barrier_sha256": self._xai_barrier_sha256,
                "transitions": self.transitions,
                "final_transition_sha256": prior,
            },
        )
        self.session._science_closed = True
        return artifact


class TrustedSession:
    """One fixed primary or replay session created only by the trusted facade."""

    def __init__(
        self,
        runner: object,
        lock: MachineLock,
        contract: Subcontract,
        prior_root: RootHandle,
        prelock_receipt: ReviewedPrelockReceipt,
        s1_receipt: S1MachineIdentityReceipt,
        replay_inputs: Optional[Mapping[str, ExternalBoundArtifact]] = None,
    ) -> None:
        self.runner = runner
        self.lock = lock
        self.contract = contract
        self.authorization = getattr(runner, "authorization", None)
        self.authorization_artifact = getattr(runner, "authorization_artifact", None)
        if (
            self.authorization is None
            or not isinstance(self.authorization_artifact, ExternalBoundArtifact)
            or self.authorization_artifact.owner is not runner
            or self.authorization.machine_lock_sha256 != lock.file_sha256
            or contract.phase not in self.authorization.phases
        ):
            raise TrustedRuntimeError(
                "a committed pushed pair authorization is required before issue"
            )
        try:
            execution = mapping(
                self.authorization.raw["execution_contract"],
                "pair authorization execution",
            )
            ceilings = mapping(
                execution["phase_resource_ceilings"], "phase resource ceilings"
            )
            self.resource_ceiling = MappingProxyType(dict(ceilings[contract.phase]))
        except (KeyError, TypeError, ContractError) as exc:
            raise TrustedRuntimeError("pair authorization resource contract is missing") from exc
        self.prior_root = prior_root
        self.prelock_receipt = prelock_receipt
        self.s1_receipt = s1_receipt
        self.replay_inputs = MappingProxyType(dict(replay_inputs or {}))
        self.process_identity = current_process_identity()
        self.session_nonce = secrets.token_hex(32)
        self.claim_sha256 = ""
        self.state = "issued"
        self._artifacts: Dict[str, BoundArtifact] = {}
        self._access_events: list[Dict[str, object]] = []
        self._member_events: list[Dict[str, object]] = []
        self._sequence = 0
        self._released_members: set[str] = set()
        self._read_member_roles: set[str] = set()
        self._member_read_sequences: Dict[str, int] = {}
        self._science_closed = False
        self._run_started_monotonic: Optional[float] = None
        self._finder: Optional[PinnedLocalImportFinder] = None
        self.targets = TargetReleaseFSM(self)
        root = Path(contract.root)
        if root.exists():
            raise FileExistsError(f"{contract.phase} session root must be new")
        root.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(root)
        if any(root.iterdir()):
            raise TrustedRuntimeError("new session root is not empty")

    def _role_path(self, role: str) -> str:
        if role not in self.contract.writes:
            raise TrustedRuntimeError(f"output role {role} is not fixed by the subcontract")
        parts = PurePosixPath(self.contract.writes[role].relative_path).parts
        path = canonical_path(Path(self.contract.root, *parts), must_exist=False)
        if not path_within(path, self.contract.root):
            raise TrustedRuntimeError("fixed output path escapes the session root")
        return path

    def _prepare_parent(self, path: str) -> None:
        root = Path(self.contract.root)
        parent = Path(path).parent
        relative = parent.relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.exists():
                if not cursor.is_dir() or cursor.is_symlink():
                    raise TrustedRuntimeError("output parent is not a physical directory")
            else:
                os.mkdir(cursor)

    def _write_bytes(self, role: str, payload: bytes, *, claim: bool = False) -> BoundArtifact:
        if role in self._artifacts:
            raise TrustedRuntimeError(f"output role {role} was already written")
        if not claim:
            self._require_running()
        projected_bytes = sum(item.bytes for item in self._artifacts.values()) + len(payload)
        if projected_bytes > int(self.resource_ceiling["disk_bytes"]):
            raise TrustedRuntimeError("phase disk-byte ceiling would be exceeded")
        path = self._role_path(role)
        self._prepare_parent(path)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            view = memoryview(bytes(payload))
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("short write")
                view = view[count:]
            os.fsync(descriptor)
            stat = os.fstat(descriptor)
            if (
                not statlib.S_ISREG(stat.st_mode)
                or int(getattr(stat, "st_nlink", 1)) != 1
            ):
                raise TrustedRuntimeError(
                    "new output is not a single-link regular file"
                )
        finally:
            os.close(descriptor)
        artifact = BoundArtifact(
            session=self,
            role=role,
            path=path,
            payload=payload,
            physical_identity=PhysicalIdentity(
                int(stat.st_dev),
                int(stat.st_ino),
                int(getattr(stat, "st_nlink", 1)),
            ),
            sequence=self._sequence,
        )
        self._sequence += 1
        self._artifacts[role] = artifact
        return artifact

    def _write_json(self, role: str, value: object, *, claim: bool = False) -> BoundArtifact:
        return self._write_bytes(role, canonical_json_bytes(value) + b"\n", claim=claim)

    def claim(self) -> BoundArtifact:
        if self.state != "issued":
            raise TrustedRuntimeError("session claim is exactly-once and must follow issue")
        if (
            not isinstance(self.prior_root.artifact, ExternalBoundArtifact)
            or self.prior_root.artifact.owner is not self.runner
        ):
            raise TrustedRuntimeError("prior root is not a runner-bound object")
        self.authorization_artifact.read_current_bytes()
        self.prior_root.artifact.read_current_bytes()
        if self.contract.phase == "primary" and self.prior_root.phase != "genesis":
            raise TrustedRuntimeError("primary claim requires the locked genesis root")
        if self.contract.phase == "replay" and self.prior_root.phase != "primary":
            raise TrustedRuntimeError("replay claim requires the finalized primary root")
        if self.prior_root.phase == "primary" and self.contract.phase == "replay":
            handoff = self.prior_root.handoff_artifact
            if (
                not isinstance(handoff, ExternalBoundArtifact)
                or handoff.owner is not self.runner
            ):
                raise TrustedRuntimeError(
                    "replay claim requires the finalized primary handoff"
                )
            handoff.read_current_bytes()
            if set(self.replay_inputs) != set(self.contract.replay_input_roles):
                raise TrustedRuntimeError(
                    "replay claim requires the exact finalized primary replay-input set"
                )
            for artifact in self.replay_inputs.values():
                if (
                    not isinstance(artifact, ExternalBoundArtifact)
                    or artifact.owner is not self.runner
                ):
                    raise TrustedRuntimeError(
                        "replay input is not bound to the finalized primary handoff"
                    )
                artifact.read_current_bytes()
            if self.prior_root.process_identity.digest == self.process_identity.digest:
                raise TrustedRuntimeError("replay must be claimed by a fresh process identity")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "state": "session_claimed_o_excl",
            "phase": self.contract.phase,
            "machine_lock_sha256": self.lock.file_sha256,
            "machine_lock_id": self.lock.lock_id,
            "pair_authorization_id": self.authorization.authorization_id,
            "pair_authorization_sha256": self.authorization.file_sha256,
            "session_nonce_sha256": hashlib.sha256(self.session_nonce.encode("ascii")).hexdigest(),
            "pid": self.process_identity.pid,
            "process_start_token": self.process_identity.start_token,
            "process_identity_sha256": self.process_identity.digest,
            "prior_root_sha256": self.prior_root.artifact.sha256,
        }
        artifact = self._write_json("claim", payload, claim=True)
        self.claim_sha256 = artifact.sha256
        artifact.creator_claim_sha256 = artifact.sha256
        self.state = "claimed"
        return artifact

    def start_run(self) -> None:
        if self.state != "claimed":
            raise TrustedRuntimeError("run start is exactly-once and must follow claim")
        self.assert_current(self._artifacts["claim"])
        self.authorization_artifact.read_current_bytes()
        self.state = "running"
        self._run_started_monotonic = time.monotonic()

    def _require_running(self) -> None:
        if self.state != "running":
            raise TrustedRuntimeError("claim and run start must precede this operation")
        if (
            self._run_started_monotonic is None
            or time.monotonic() - self._run_started_monotonic
            > int(self.resource_ceiling["wall_seconds"])
        ):
            raise TrustedRuntimeError("phase wall-time ceiling was exceeded")
        if current_process_identity() != self.process_identity:
            raise TrustedRuntimeError("trusted session process identity changed")
        if "claim" not in self._artifacts:
            raise TrustedRuntimeError("physical claim is missing")
        self.assert_current(self._artifacts["claim"])
        self.prior_root.artifact.read_current_bytes()
        if self.contract.phase == "replay":
            assert isinstance(self.prior_root.handoff_artifact, ExternalBoundArtifact)
            self.prior_root.handoff_artifact.read_current_bytes()

    def _require_science_ready(self) -> None:
        self._require_running()
        if "runtime_attestation" not in self._artifacts:
            raise TrustedRuntimeError("runtime attestation must precede scientific access")
        if self._science_closed:
            raise TrustedRuntimeError("scientific access is closed after transition finalization")

    def assert_prelock_receipt(self, receipt: ReviewedPrelockReceipt) -> None:
        self._require_running()
        if receipt is not self.prelock_receipt or receipt.lock_id != self.lock.lock_id:
            raise TrustedRuntimeError("target release requires the runner-loaded reviewed prelock receipt")

    def _record_access(self, role: str, record: ArtifactRecord, identity: PhysicalIdentity) -> None:
        self._access_events.append(
            {
                "index": len(self._access_events),
                "role": role,
                "path": record.path,
                "sha256": record.sha256,
                "bytes": record.bytes,
                "physical_device": identity.device,
                "physical_inode": identity.inode,
                "claim_sha256": self.claim_sha256,
            }
        )

    def read_artifact(self, role: str) -> BoundArtifact:
        self._require_science_ready()
        if role not in self.contract.reads:
            raise TrustedRuntimeError(f"read role {role} is not fixed by the subcontract")
        if role in {spec.archive_role for spec in self.contract.members.values()}:
            raise TrustedRuntimeError(
                "archive containers are readable only through the typed member reader"
            )
        record = self.contract.reads[role]
        payload, identity = _read_record(record, f"read role {role}")
        self._record_access(role, record, identity)
        artifact = BoundArtifact(
            session=self,
            role=role,
            path=record.path,
            payload=payload,
            physical_identity=identity,
            sequence=self._sequence,
        )
        self._sequence += 1
        return artifact

    def read_replay_input(self, role: str) -> ExternalBoundArtifact:
        self._require_science_ready()
        if self.contract.phase != "replay" or role not in self.contract.replay_input_roles:
            raise TrustedRuntimeError("role is not a fixed replay input")
        if role not in self.replay_inputs:
            raise TrustedRuntimeError("replay input is absent from the finalized primary handoff")
        artifact = self.replay_inputs[role]
        if artifact.owner is not self.runner:
            raise TrustedRuntimeError("replay input belongs to another trusted runner")
        artifact.read_current_bytes()
        self._access_events.append(
            {
                "index": len(self._access_events),
                "role": role,
                "path": artifact.path,
                "sha256": artifact.sha256,
                "bytes": artifact.bytes,
                "physical_device": artifact.physical_identity.device,
                "physical_inode": artifact.physical_identity.inode,
                "claim_sha256": self.claim_sha256,
                "source": "finalized_primary_handoff",
            }
        )
        return artifact

    def _member_bytes(
        self, role: str, *, record_event: bool
    ) -> Tuple[bytes, PhysicalIdentity, str, Tuple[int, ...]]:
        spec = self.contract.members[role]
        if spec.member_name in self.contract.forbidden_member_names:
            raise TrustedRuntimeError("original all-row target member is permanently forbidden")
        archive = self.contract.reads[spec.archive_role]
        archive_payload, identity = _read_record(archive, f"archive {spec.archive_role}")
        if record_event:
            self._record_access(spec.archive_role, archive, identity)
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as handle:
            infos = handle.infolist()
            expected_inventory = sorted(
                (member.order, member.member_name)
                for member in self.contract.members.values()
                if member.archive_role == spec.archive_role
            )
            observed_inventory = [
                (position, info.filename) for position, info in enumerate(infos)
            ]
            if observed_inventory != expected_inventory:
                raise TrustedRuntimeError("archive physical member inventory differs")
            if any(info.filename in self.contract.forbidden_member_names for info in infos):
                raise TrustedRuntimeError(
                    "archive physically contains a permanently forbidden member"
                )
            if spec.order >= len(infos) or infos[spec.order].filename != spec.member_name:
                raise TrustedRuntimeError("archive member order/name differs")
            if len({item.filename for item in infos}) != len(infos):
                raise TrustedRuntimeError("archive contains duplicate member names")
            info = infos[spec.order]
            if info.file_size != spec.uncompressed_bytes:
                raise TrustedRuntimeError("archive member uncompressed size differs")
            payload = handle.read(info)
        if hashlib.sha256(payload).hexdigest() != spec.uncompressed_sha256:
            raise TrustedRuntimeError("archive member uncompressed SHA-256 differs")
        try:
            import numpy as np

            array = np.load(io.BytesIO(payload), allow_pickle=False)
            dtype, shape = array.dtype.str, tuple(int(item) for item in array.shape)
        except Exception as exc:
            raise TrustedRuntimeError("archive member is not a non-pickle NPY array") from exc
        if dtype != spec.dtype or shape != spec.shape:
            raise TrustedRuntimeError("archive member dtype/shape differs")
        return payload, identity, dtype, shape

    def read_archive_member(self, role: str) -> BoundArtifact:
        self._require_science_ready()
        if role not in self.contract.members:
            raise TrustedRuntimeError(f"archive member role {role} is not fixed")
        spec = self.contract.members[role]
        if spec.target_bearing and role not in self._released_members:
            raise TrustedRuntimeError("target-bearing member has not reached its release stage")
        payload, identity, dtype, shape = self._member_bytes(role, record_event=True)
        self._member_events.append(
            {
                "index": len(self._member_events),
                "role": role,
                "archive_role": spec.archive_role,
                "member_name": spec.member_name,
                "order": spec.order,
                "dtype": dtype,
                "shape": list(shape),
                "uncompressed_sha256": spec.uncompressed_sha256,
                "stage": spec.stage,
                "fold": spec.fold,
                "claim_sha256": self.claim_sha256,
            }
        )
        self._read_member_roles.add(role)
        artifact = BoundArtifact(
            session=self,
            role=role,
            path=self.contract.reads[spec.archive_role].path,
            payload=payload,
            physical_identity=identity,
            sequence=self._sequence,
            member_name=spec.member_name,
        )
        self._sequence += 1
        self._member_read_sequences[role] = artifact.sequence
        return artifact

    def write_output(self, role: str, payload: bytes) -> BoundArtifact:
        self._require_science_ready()
        if role in MANDATORY_CONTROL_ROLES:
            raise TrustedRuntimeError("control roles are emitted only by the trusted session")
        return self._write_bytes(role, payload)

    def write_json_output(self, role: str, value: object) -> BoundArtifact:
        self._require_science_ready()
        if role in MANDATORY_CONTROL_ROLES:
            raise TrustedRuntimeError("control roles are emitted only by the trusted session")
        return self._write_json(role, value)

    def _write_control(self, role: str, value: object) -> BoundArtifact:
        if role not in MANDATORY_CONTROL_ROLES:
            raise TrustedRuntimeError("role is not a fixed control artifact")
        return self._write_json(role, value)

    def make_scientific_receipt(
        self,
        kind: str,
        fold: int,
        artifact: BoundArtifact,
        *,
        cidt: Optional[CIDTOutcome] = None,
    ) -> ScientificReceipt:
        self._require_science_ready()
        if kind not in {*TARGET_STAGES, "held_prediction"} or fold not in range(5):
            raise TrustedRuntimeError("scientific receipt kind/fold differs")
        self.assert_current(artifact)
        expected_role = f"fold_{fold}_{kind}_output"
        if artifact.role != expected_role or expected_role not in self.contract.writes:
            raise TrustedRuntimeError("scientific receipt artifact role differs")
        return ScientificReceipt(self, kind, fold, artifact, cidt)

    def assert_scientific_receipt(self, receipt: ScientificReceipt, kind: str) -> None:
        self._require_science_ready()
        if (
            not isinstance(receipt, ScientificReceipt)
            or receipt.session is not self
            or receipt.kind != kind
            or receipt.fold not in range(5)
        ):
            raise TrustedRuntimeError("typed scientific receipt differs")
        self.assert_current(receipt.artifact)

    def assert_current(self, artifact: BoundArtifact) -> bytes:
        if not isinstance(artifact, BoundArtifact) or artifact.session is not self:
            raise TrustedRuntimeError("artifact belongs to another trusted session")
        is_claim = self._artifacts.get("claim") is artifact
        expected_creator_claim = artifact.sha256 if is_claim else self.claim_sha256
        if (
            artifact.session_nonce != self.session_nonce
            or artifact.creator_process_sha256 != self.process_identity.digest
            or artifact.creator_claim_sha256 != expected_creator_claim
        ):
            raise TrustedRuntimeError("artifact creator receipt/session identity differs")
        if artifact.member_name is None:
            payload, identity = _read_file(artifact.path)
        else:
            payload, identity, _, _ = self._member_bytes(artifact.role, record_event=False)
        if (
            identity != artifact.physical_identity
            or len(payload) != artifact.bytes
            or hashlib.sha256(payload).hexdigest() != artifact.sha256
        ):
            raise TrustedRuntimeError("artifact physical identity/current bytes differ")
        return payload

    def attest_runtime(self) -> BoundArtifact:
        self._require_running()
        if self._science_closed or "runtime_attestation" in self._artifacts:
            raise TrustedRuntimeError("runtime attestation is create-once before science")
        payload, finder = attest_runtime(self.lock, self.s1_receipt)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(
            self.resource_ceiling["gpu_device"]
        ):
            raise TrustedRuntimeError("authorized GPU device differs from the process")
        self._finder = finder
        return self._write_control("runtime_attestation", dict(payload))

    def emit_access_and_member_ledgers(self) -> Tuple[BoundArtifact, BoundArtifact]:
        self._require_running()
        if not self._science_closed or "transition_chain" not in self._artifacts:
            raise TrustedRuntimeError("access ledgers require finalized scientific transitions")
        access = self._write_control(
            "access_ledger",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "exact_access_ledger_finalized",
                "claim_sha256": self.claim_sha256,
                "events": self._access_events,
            },
        )
        members = self._write_control(
            "member_ledger",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "typed_archive_member_ledger_finalized",
                "claim_sha256": self.claim_sha256,
                "events": self._member_events,
            },
        )
        return access, members

    def physical_inventory(self) -> Mapping[str, Mapping[str, object]]:
        inventory: Dict[str, Mapping[str, object]] = {}
        root = Path(self.contract.root)
        for current, directories, files in os.walk(root, followlinks=False):
            for directory in directories:
                path = Path(current, directory)
                if path.is_symlink():
                    raise TrustedRuntimeError("session inventory contains a symlink directory")
            for filename in files:
                path = Path(current, filename)
                if path.is_symlink():
                    raise TrustedRuntimeError("session inventory contains a symlink file")
                relative = path.relative_to(root).as_posix()
                payload, identity = _read_file(str(path))
                inventory[relative] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "physical_device": identity.device,
                    "physical_inode": identity.inode,
                    "physical_link_count": identity.link_count,
                }
        return MappingProxyType(inventory)

    def physical_directories(self) -> Tuple[str, ...]:
        directories: set[str] = set()
        root = Path(self.contract.root)
        for current, names, _ in os.walk(root, followlinks=False):
            for name in names:
                path = Path(current, name)
                if path.is_symlink() or not path.is_dir():
                    raise TrustedRuntimeError(
                        "session inventory contains a non-physical directory"
                    )
                directories.add(path.relative_to(root).as_posix())
        return tuple(sorted(directories))

    def expected_relative_paths(self) -> Mapping[str, str]:
        return MappingProxyType(
            {role: spec.relative_path for role, spec in self.contract.writes.items()}
        )

    def close_import_finder(self) -> None:
        if self._finder is not None:
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass
            self._finder = None
