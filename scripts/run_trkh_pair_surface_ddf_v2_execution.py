from __future__ import annotations

"""Fresh-process DDF-v2 execution bootstrap; intentionally fail closed today.

This is a correctness boundary for reviewed, hash-pinned Python, not a hostile
Python sandbox.  The fixed producer module named below does not exist yet.
"""

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys


PROTOCOL_ID = "trkh_pair_surface_ddf_a0_v2"
PRODUCER_MODULE = "trkh.tools.pair_surface_ddf_v2_producer"
PRODUCER_CALLABLE = "run_trusted_pair_surface_ddf_v2"
REQUIRED_RUNNER_MODULE = "trkh.tools.pair_surface_ddf_v2_execution_guard"
HEX64 = frozenset("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hex64(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in HEX64 for character in value):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def _path(value: object, *, exists: bool) -> str:
    absolute = os.path.abspath(os.path.expanduser(os.fspath(value)))
    resolved = os.path.realpath(absolute)
    if os.path.normcase(absolute) != os.path.normcase(resolved):
        raise RuntimeError("execution path uses a symlink/junction alias")
    if exists and not os.path.isfile(resolved):
        raise RuntimeError(f"required execution file is missing: {resolved}")
    return resolved


class _ProcessAuditBoundary:
    """Catches accidental direct I/O by the reviewed producer."""

    _CHILD_EVENTS = {
        "subprocess.Popen", "os.system", "os.fork", "os.forkpty",
    }
    _DESTRUCTIVE_EVENTS = {
        "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
        "os.link", "os.symlink", "os.truncate",
    }

    def __init__(self, bootstrap_files: set[str]) -> None:
        self.read_files = {os.path.normcase(path) for path in bootstrap_files}
        self.read_directories: set[str] = set()
        self.output_files: set[str] = set()
        self.output_directories: set[str] = set()
        self.event_count = 0
        self.event_digest = hashlib.sha256()

    @staticmethod
    def _normal(value: object) -> str:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))

    def configure(
        self,
        *,
        read_files: set[str],
        output_files: set[str],
        output_directories: set[str],
    ) -> None:
        self.read_files.update(self._normal(path) for path in read_files)
        self.output_files = {self._normal(path) for path in output_files}
        self.output_directories = {
            self._normal(path) for path in output_directories
        }
        for path in self.read_files | self.output_files:
            cursor = os.path.dirname(path)
            while cursor and cursor != os.path.dirname(cursor):
                self.read_directories.add(cursor)
                cursor = os.path.dirname(cursor)
        self.read_directories.update(self.output_directories)

    def _record(self, event: str, path: str | None = None) -> None:
        self.event_count += 1
        self.event_digest.update(_canonical_bytes([event, path]) + b"\n")

    def audit(self, event: str, arguments: tuple[object, ...]) -> None:
        if (
            event in self._CHILD_EVENTS
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
            or event.startswith("os.posix_spawn")
        ):
            self._record("denied_child")
            raise RuntimeError("child processes are forbidden by the execution launcher")
        if event.startswith("socket."):
            self._record("denied_network")
            raise RuntimeError("network access is forbidden by the execution launcher")
        if event in self._DESTRUCTIVE_EVENTS:
            self._record("denied_destructive_mutation")
            raise RuntimeError("destructive filesystem mutation is forbidden")
        if event == "open" and arguments and not isinstance(arguments[0], int):
            path = self._normal(arguments[0])
            mode = arguments[1] if len(arguments) > 1 else None
            flags = arguments[2] if len(arguments) > 2 else 0
            write_like = (
                isinstance(mode, str) and any(mark in mode for mark in "wax+")
            ) or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            )
            permitted = (
                path in self.output_files if write_like
                else path in self.read_files or path in self.output_files
            )
            self._record("write" if write_like else "read", path)
            if not permitted:
                raise RuntimeError("direct file access is outside the exact contract")
        if event in {"os.listdir", "os.scandir"} and arguments:
            path = self._normal(arguments[0])
            self._record("directory_read", path)
            if path not in self.read_directories:
                raise RuntimeError("directory read is outside the exact contract")
        if event == "os.mkdir" and arguments:
            path = self._normal(arguments[0])
            self._record("mkdir", path)
            if path not in self.output_directories:
                raise RuntimeError("directory creation is outside fixed output paths")

    def receipt(self) -> dict[str, object]:
        return {
            "event_count": self.event_count,
            "ordered_event_sha256": self.event_digest.hexdigest(),
            "child_processes": 0,
            "network_access": False,
        }


class _PinnedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname: str, path: str, expected_sha256: str) -> None:
        super().__init__(fullname, path)
        self.expected_sha256 = expected_sha256

    def get_code(self, fullname: str):
        payload = Path(self.path).read_bytes()
        if _digest(payload) != self.expected_sha256:
            raise ImportError(f"pinned source changed for {fullname}")
        return compile(payload, self.path, "exec", dont_inherit=True)


class _PinnedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, modules: dict[str, dict[str, object]]) -> None:
        self.modules = modules

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if fullname.startswith("trkh") and fullname not in self.modules:
            raise ImportError(f"local module is outside the pinned closure: {fullname}")
        if fullname not in self.modules:
            return None
        row = self.modules[fullname]
        source = str(row["path"])
        loader = _PinnedLoader(fullname, source, str(row["sha256"]))
        locations = [os.path.dirname(source)] if bool(row["package"]) else None
        return importlib.util.spec_from_file_location(
            fullname, source, loader=loader, submodule_search_locations=locations
        )


def _read_document(path: str, expected_sha256: str, label: str) -> dict[str, object]:
    payload = Path(path).read_bytes()
    if _digest(payload) != _hex64(expected_sha256, f"expected {label} SHA-256"):
        raise RuntimeError(f"{label} exact bytes differ")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or payload != _canonical_bytes(value) + b"\n":
        raise RuntimeError(f"{label} is not canonical newline-terminated JSON")
    return value


def _record_paths(records: object) -> set[str]:
    if not isinstance(records, dict):
        raise RuntimeError("artifact record mapping differs")
    result = set()
    for row in records.values():
        if not isinstance(row, dict) or set(row) != {
            "path", "bytes", "sha256", "access_class"
        }:
            raise RuntimeError("artifact record schema differs")
        result.add(_path(row["path"], exists=True))
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen DDF-v2 phase.")
    parser.add_argument("--mode", choices=("primary", "replay"), required=True)
    parser.add_argument("--machine-lock", required=True)
    parser.add_argument("--expected-machine-lock-sha256", required=True)
    parser.add_argument("--pair-authorization", required=True)
    parser.add_argument("--expected-pair-authorization-sha256", required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--expected-source-closure-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--watchdog-contract", required=True)
    parser.add_argument("--expected-watchdog-contract-sha256", required=True)
    parser.add_argument("--expected-primary-root-sha256")
    return parser.parse_args()


def main() -> int:
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise RuntimeError("execution launcher requires Python -I -B")
    args = _parse_arguments()
    launcher = _path(__file__, exists=True)
    launcher_sha256 = _digest(Path(launcher).read_bytes())
    if launcher_sha256 != _hex64(args.expected_launcher_sha256, "launcher SHA-256"):
        raise RuntimeError("launcher exact bytes differ")
    paths = {
        "lock": _path(args.machine_lock, exists=True),
        "authorization": _path(args.pair_authorization, exists=True),
        "closure": _path(args.source_closure, exists=True),
        "watchdog": _path(args.watchdog_contract, exists=True),
    }
    audit = _ProcessAuditBoundary({launcher, *paths.values()})
    sys.addaudithook(audit.audit)
    lock = _read_document(
        paths["lock"], args.expected_machine_lock_sha256, "machine lock"
    )
    authorization = _read_document(
        paths["authorization"],
        args.expected_pair_authorization_sha256,
        "combined pair authorization",
    )
    closure = _read_document(
        paths["closure"], args.expected_source_closure_sha256, "source closure"
    )
    watchdog = _read_document(
        paths["watchdog"],
        args.expected_watchdog_contract_sha256,
        "external watchdog contract",
    )
    if closure.get("protocol_id") != PROTOCOL_ID or closure.get("state") != (
        "prospective_execution_source_closure_with_reviewed_producer"
    ):
        raise RuntimeError("source closure is not a reviewed producer boundary")
    if closure.get("launcher_sha256") != launcher_sha256:
        raise RuntimeError("source closure launcher binding differs")
    fixed = closure.get("fixed_producer")
    if fixed != {"module": PRODUCER_MODULE, "callable": PRODUCER_CALLABLE}:
        raise RuntimeError("fixed producer role is absent or differs")
    modules = closure.get("local_modules")
    if not isinstance(modules, dict) or PRODUCER_MODULE not in modules:
        raise RuntimeError("fixed reviewed producer source is absent (expected today)")
    for name, row in modules.items():
        if not isinstance(row, dict) or set(row) != {
            "path", "bytes", "sha256", "package"
        }:
            raise RuntimeError(f"source closure row differs for {name}")
        row["path"] = _path(row["path"], exists=True)
    runtime = lock.get("runtime")
    subcontracts = lock.get("subcontracts")
    if not isinstance(runtime, dict) or not isinstance(subcontracts, dict):
        raise RuntimeError("machine-lock runtime/subcontract schema differs")
    lock_modules = runtime.get("local_import_closure")
    if not isinstance(lock_modules, dict) or set(lock_modules) != set(modules):
        raise RuntimeError("source closure is not the exact machine-lock closure")
    for name, row in modules.items():
        locked = lock_modules[name]
        if not isinstance(locked, dict) or (
            _path(locked.get("path"), exists=True) != row["path"]
            or locked.get("bytes") != row["bytes"]
            or locked.get("sha256") != row["sha256"]
        ):
            raise RuntimeError(f"machine-lock source identity differs for {name}")
    phase = subcontracts.get(args.mode)
    if not isinstance(phase, dict) or not isinstance(phase.get("writes"), dict):
        raise RuntimeError("selected phase subcontract differs")
    output_root = _path(phase.get("root"), exists=False)
    if not os.path.isdir(os.path.dirname(output_root)):
        raise RuntimeError("fixed output-root parent must already exist")
    output_files = set()
    output_directories = {output_root}
    for spec in phase["writes"].values():
        relative = str(spec["relative_path"]).replace("/", os.sep)
        destination = _path(os.path.join(output_root, relative), exists=False)
        if os.path.commonpath((os.path.normcase(destination), os.path.normcase(output_root))) != os.path.normcase(output_root):
            raise RuntimeError("fixed output path escapes the selected root")
        output_files.add(destination)
        cursor = os.path.dirname(destination)
        while cursor != output_root:
            output_directories.add(cursor)
            cursor = os.path.dirname(cursor)
    read_files = set(paths.values()) | {launcher}
    read_files |= {str(row["path"]) for row in modules.values()}
    read_files |= _record_paths(runtime.get("package_files"))
    read_files |= _record_paths(phase.get("reads"))
    read_files |= _record_paths(lock.get("prelock", {}).get("artifacts"))
    if isinstance(lock.get("genesis_root"), dict):
        read_files.add(_path(lock["genesis_root"]["path"], exists=True))
    audit.configure(
        read_files=read_files,
        output_files=output_files,
        output_directories=output_directories,
    )
    for name, row in modules.items():
        payload = Path(row["path"]).read_bytes()
        if len(payload) != row["bytes"] or _digest(payload) != row["sha256"]:
            raise RuntimeError(f"source closure bytes differ for {name}")
    execution = authorization.get("execution_contract")
    limits = execution.get("phase_resource_ceilings", {}).get(args.mode) if isinstance(execution, dict) else None
    expected_watchdog = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "state": "external_parent_watchdog_required_and_active",
        "mode": args.mode,
        "pair_authorization_sha256": args.expected_pair_authorization_sha256,
        "limits": limits,
        "launcher_spawns_processes": False,
    }
    if watchdog != expected_watchdog:
        raise RuntimeError("external wall/disk/GPU watchdog contract differs")
    if not isinstance(execution, dict) or (
        execution.get("python_flags") != ["-I", "-B"]
        or execution.get("child_processes") != 0
        or execution.get("network_access") is not False
    ):
        raise RuntimeError("combined authorization execution policy differs")
    if args.mode == "primary" and args.expected_primary_root_sha256 is not None:
        raise RuntimeError("primary mode forbids a primary-root argument")
    if args.mode == "replay" and args.expected_primary_root_sha256 is None:
        raise RuntimeError("replay mode requires the finalized primary root SHA-256")
    sys.meta_path.insert(0, _PinnedFinder(modules))
    guard = importlib.import_module(REQUIRED_RUNNER_MODULE)
    runner_type = getattr(guard, "TrustedPairSurfaceRunner")
    if args.mode == "primary":
        runner = runner_type.open_primary(
            paths["lock"], args.expected_machine_lock_sha256,
            paths["authorization"], args.expected_pair_authorization_sha256,
        )
    else:
        runner = runner_type.open_replay(
            paths["lock"], args.expected_machine_lock_sha256,
            paths["authorization"], args.expected_pair_authorization_sha256,
            _hex64(args.expected_primary_root_sha256, "primary root SHA-256"),
        )
    session = runner.issue()
    session.claim()
    session.start_run()
    session.attest_runtime()
    producer = getattr(importlib.import_module(PRODUCER_MODULE), PRODUCER_CALLABLE)
    if producer(session) is not None:
        raise RuntimeError("fixed producer must return None and expose no raw archive bytes")
    root = runner.finalize(session)
    receipt = {
        "schema_version": 1,
        "state": "execution_launcher_completed_external_capture_required",
        "mode": args.mode,
        "machine_lock_sha256": args.expected_machine_lock_sha256,
        "pair_authorization_sha256": args.expected_pair_authorization_sha256,
        "source_closure_sha256": args.expected_source_closure_sha256,
        "launcher_sha256": launcher_sha256,
        "final_root_sha256": root.artifact.sha256,
        "audit": audit.receipt(),
    }
    print("TRKH_DDF_V2_EXECUTION_RECEIPT=" + _canonical_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
