from __future__ import annotations

"""Stdlib bootstrap for the schema-v3 trusted-runner test boundary."""

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {
    "cache": r"D:\DataAI\AIEx\TRKH\cache",
    "raw": r"D:\DataAI\AIEx\newdataset",
    "validation": r"D:\DataAI\AIEx\newdataset\class_f\val",
    "test": r"D:\DataAI\AIEx\newdataset\class_f\test",
    "gpu": r"D:\DataAI\AIEx\TRKH\runs",
}
REQUIRED_TEST_NAMES = {
    "test_01_runner_facade_has_no_caller_ledger_or_policy_surface",
    "test_02_claim_precedes_every_sensitive_operation",
    "test_03_issue_claim_and_start_are_exactly_once",
    "test_04_roots_and_claim_paths_must_be_new_physical_locations",
    "test_05_outputs_are_internal_o_excl_create_once_only",
    "test_06_typed_archive_reader_checks_name_order_dtype_shape_size_and_hash",
    "test_07_contract_rejects_stage_mismatch_and_all_row_member",
    "test_08_artifacts_bind_session_creator_physical_identity_and_current_bytes",
    "test_09_live_repository_checks_head_upstream_porcelain_and_registry",
    "test_10_target_fsm_accepts_only_typed_runner_receipts_and_exact_cidt",
    "test_11_xai_waits_for_all_five_clean_and_causal_cidt_outcomes",
    "test_12_replay_requires_finalized_handoff_exact_inputs_and_fresh_process",
    "test_13_handoff_and_final_root_require_exact_physical_inventory",
    "test_14_runtime_attestation_rejects_fake_sys_modules_package",
    "test_15_machine_lock_loader_requires_exact_canonical_bytes_and_hash",
    "test_52_immutable_synthetic_v1_dependency_keeps_builders_hard_blocked",
    "test_99_positive_trusted_primary_lifecycle_finalizes_exact_root",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _BootstrapBoundary:
    def __init__(self) -> None:
        self.roots = {
            name: os.path.normcase(os.path.realpath(os.path.abspath(path)))
            for name, path in FORBIDDEN_ROOTS.items()
        }
        self.counts = {name: 0 for name in self.roots}
        self.allowed_local_files: set[str] = set()
        self.enforce_local_closure = False
        self.forbid_children = False

    def audit(self, event, args) -> None:
        if self.forbid_children and (
            event in {"subprocess.Popen", "os.system", "os.fork", "os.forkpty"}
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
            or event.startswith("os.posix_spawn")
        ):
            raise RuntimeError("child processes are forbidden by this exact test authority")
        if event not in {"open", "os.listdir", "os.scandir"} or not args or isinstance(args[0], int):
            return
        try:
            candidate = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(args[0]))))
        except (OSError, TypeError, ValueError):
            return
        for name, root in self.roots.items():
            try:
                inside = os.path.commonpath((candidate, root)) == root
            except ValueError:
                inside = False
            if inside:
                self.counts[name] += 1
                raise RuntimeError(f"pre-import tripwire blocked forbidden {name} access")
        if self.enforce_local_closure:
            try:
                local = os.path.commonpath((candidate, os.path.normcase(str(REPOSITORY_ROOT)))) == os.path.normcase(str(REPOSITORY_ROOT))
            except ValueError:
                local = False
            if local and (candidate.endswith(".py") or candidate.endswith(".pyc")):
                if candidate not in self.allowed_local_files:
                    raise RuntimeError("local import/read is outside the prospective pinned closure")


class _PinnedSourceLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, path: Path, expected_sha256: str) -> None:
        self.fullname, self.path, self.expected_sha256 = fullname, path, expected_sha256

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        source = self.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected_sha256:
            raise ImportError(f"pinned local source changed for {self.fullname}")
        module.__file__ = str(self.path)
        module.__package__ = (
            self.fullname
            if self.path.name == "__init__.py"
            else self.fullname.rpartition(".")[0]
        )
        exec(compile(source, str(self.path), "exec", dont_inherit=True), module.__dict__)


class _PinnedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, closure) -> None:
        self.closure = closure

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname not in self.closure:
            return None
        row = self.closure[fullname]
        source = REPOSITORY_ROOT / row["relative_path"]
        loader = _PinnedSourceLoader(fullname, source, row["sha256"])
        package_locations = [str(source.parent)] if source.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=loader,
            submodule_search_locations=package_locations,
        )


class _ExactCounter:
    def __init__(self, registry) -> None:
        self.registry = {str(row["nodeid"]): str(row["classification"]) for row in registry}
        self.collected: list[str] = []
        self.passed: set[str] = set()

    def pytest_collection_finish(self, session) -> None:
        self.collected = [item.nodeid.replace("\\", "/") for item in session.items]
        if set(self.collected) != set(self.registry) or len(self.collected) != len(self.registry):
            raise RuntimeError("expanded pytest node set differs from the exact authority")

    def pytest_runtest_logreport(self, report) -> None:
        nodeid = report.nodeid.replace("\\", "/")
        if report.when == "call" and report.passed:
            self.passed.add(nodeid)


def _load_authority(
    path: Path,
    expected_sha256: str,
    bootstrap: _BootstrapBoundary,
):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("v3 test authority differs from its external expected SHA-256")
    value = json.loads(payload.decode("utf-8"))
    if _canonical_bytes(value) + b"\n" != payload:
        raise RuntimeError("v3 test authority is not canonical newline-terminated JSON")
    expected_keys = {
        "schema_version", "state", "launcher_sha256", "test_source",
        "local_import_closure", "expanded_nodes", "fixture_scope",
        "preimport_tripwire_required", "forbidden_roots", "child_process_policy",
    }
    if set(value) != expected_keys or int(value["schema_version"]) != 3:
        raise RuntimeError("v3 test authority schema differs")
    if value["state"] != "trusted_runner_v3_test_authority_frozen_after_source_commit":
        raise RuntimeError("v3 test authority is not frozen after source commit")
    if value["launcher_sha256"] != _sha(Path(__file__).resolve()):
        raise RuntimeError("launcher differs from the exact authority")
    if value["forbidden_roots"] != FORBIDDEN_ROOTS or value["preimport_tripwire_required"] is not True:
        raise RuntimeError("pre-import tripwire authority differs")
    if value["fixture_scope"] != "synthetic_tmp_only_no_repository_cache_dataset_gpu":
        raise RuntimeError("test fixture scope differs")
    if value["child_process_policy"] != "forbidden_during_pytest":
        raise RuntimeError("child process policy differs")
    test_source = value["test_source"]
    test_path = REPOSITORY_ROOT / test_source["relative_path"]
    if test_source["relative_path"] != "tests/test_pair_surface_ddf_v2_execution_guard.py" or _sha(test_path) != test_source["sha256"]:
        raise RuntimeError("test source differs from the exact authority")
    closure = value["local_import_closure"]
    if not closure:
        raise RuntimeError("prospective local import closure is empty")
    allowed = {
        os.path.normcase(os.path.realpath(str(candidate.resolve())))
        for candidate in (path, test_path, Path(__file__).resolve())
    }
    for module, row in closure.items():
        if set(row) != {"relative_path", "sha256"} or not module:
            raise RuntimeError("local import closure row differs")
        source = (REPOSITORY_ROOT / row["relative_path"]).resolve()
        try:
            inside = os.path.commonpath((str(source), str(REPOSITORY_ROOT))) == str(REPOSITORY_ROOT)
        except ValueError:
            inside = False
        if not inside or _sha(source) != row["sha256"]:
            raise RuntimeError(f"local import closure differs for {module}")
        allowed.add(os.path.normcase(os.path.realpath(str(source))))
    registry = value["expanded_nodes"]
    if not registry or len({row["nodeid"] for row in registry}) != len(registry):
        raise RuntimeError("expanded test-node registry differs")
    for row in registry:
        if set(row) != {"nodeid", "classification"}:
            raise RuntimeError("expanded test-node row differs")
        if not row["nodeid"].startswith("tests/test_pair_surface_ddf_v2_execution_guard.py::"):
            raise RuntimeError("expanded test node escapes the isolated test source")
        if row["classification"] not in {"negative", "positive"}:
            raise RuntimeError("test classification differs")
    expected_nodeids = {
        f"tests/test_pair_surface_ddf_v2_execution_guard.py::{name}"
        for name in REQUIRED_TEST_NAMES
    }
    if {row["nodeid"] for row in registry} != expected_nodeids:
        raise RuntimeError("expanded test nodes differ from the mandatory v3 registry")
    classifications = {row["nodeid"]: row["classification"] for row in registry}
    if (
        sum(value == "negative" for value in classifications.values()) < 10
        or classifications[
            "tests/test_pair_surface_ddf_v2_execution_guard.py::test_99_positive_trusted_primary_lifecycle_finalizes_exact_root"
        ] != "positive"
    ):
        raise RuntimeError("mandatory negative/positive classification differs")
    bootstrap.allowed_local_files = allowed
    bootstrap.enforce_local_closure = True
    return value


def _write_manifest(path: Path, value: object) -> str:
    if "V3" not in path.name.upper() or path.exists() or path.with_suffix(".sha256").exists():
        raise RuntimeError("v3 result manifest path must be new and versioned")
    payload = _canonical_bytes(value) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    descriptor = os.open(
        str(path.resolve()), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while finalizing the v3 test manifest")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def main() -> int:
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise RuntimeError("test launcher must be started with Python -I -B")
    parser = argparse.ArgumentParser(description="Run the frozen schema-v3 trusted-runner tests.")
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    args = parser.parse_args()
    bootstrap = _BootstrapBoundary()
    sys.addaudithook(bootstrap.audit)
    authority = _load_authority(
        args.authority.resolve(), args.expected_authority_sha256, bootstrap
    )
    bootstrap.forbid_children = True
    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.meta_path.insert(0, _PinnedSourceFinder(authority["local_import_closure"]))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTEST_ADDOPTS"] = ""
    os.environ["PYTEST_PLUGINS"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import pytest

    counter = _ExactCounter(authority["expanded_nodes"])
    nodeids = [row["nodeid"] for row in authority["expanded_nodes"]]
    result = pytest.main(
        ["-q", "-c", os.devnull, "-p", "no:cacheprovider", *nodeids],
        plugins=[counter],
    )
    bootstrap.forbid_children = False
    if result != pytest.ExitCode.OK or set(counter.passed) != set(nodeids):
        raise RuntimeError("trusted-runner exact test execution failed")
    if any(bootstrap.counts.values()):
        raise RuntimeError("forbidden access occurred during trusted-runner tests")
    negatives = sorted(
        nodeid for nodeid, classification in counter.registry.items() if classification == "negative"
    )
    manifest = {
        "schema_version": 3,
        "state": "trusted_runner_v3_tests_finalized",
        "authority_sha256": _sha(args.authority.resolve()),
        "launcher_sha256": _sha(Path(__file__).resolve()),
        "test_source_sha256": authority["test_source"]["sha256"],
        "local_import_closure": authority["local_import_closure"],
        "expanded_node_set_sha256": hashlib.sha256(_canonical_bytes(sorted(nodeids))).hexdigest(),
        "test_count": len(nodeids),
        "negative_nodeids": negatives,
        "negative_test_count": len(negatives),
        "passed": True,
        "fresh_isolated_process": True,
        "child_processes": {"policy": "forbidden_during_pytest", "observed": 0},
        "forbidden_access_counts": bootstrap.counts,
        "live_integration": {"collected": 0, "executed": 0},
    }
    digest = _write_manifest(args.result_manifest.resolve(), manifest)
    print(f"trusted_runner_v3_manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
