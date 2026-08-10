from __future__ import annotations

"""Static/tmp-only checks for the fail-closed DDF-v2 execution launcher."""

import ast
from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_trkh_pair_surface_ddf_v2_execution.py"
)


def _source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(LAUNCHER))


def test_launcher_has_stdlib_only_static_imports() -> None:
    imported = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "trkh" not in imported
    assert "torch" not in imported
    assert "numpy" not in imported


def test_launcher_requires_isolated_no_bytecode_and_all_external_hashes() -> None:
    source = _source()
    assert "sys.flags.isolated" in source
    assert "sys.flags.dont_write_bytecode" in source
    for option in (
        "--expected-machine-lock-sha256",
        "--expected-pair-authorization-sha256",
        "--expected-source-closure-sha256",
        "--expected-launcher-sha256",
        "--expected-watchdog-contract-sha256",
    ):
        assert option in source


def test_audit_hook_precedes_every_dynamic_project_import() -> None:
    main = next(
        node for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    hook_lines = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "addaudithook"
    ]
    import_lines = [
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
    ]
    assert len(hook_lines) == 1
    assert import_lines and hook_lines[0] < min(import_lines)


def test_fixed_runner_sequence_and_producer_are_literal() -> None:
    source = _source()
    assert 'PRODUCER_MODULE = "trkh.tools.pair_surface_ddf_v2_producer"' in source
    assert 'PRODUCER_CALLABLE = "run_trusted_pair_surface_ddf_v2"' in source
    sequence = [
        "runner.issue()",
        "session.claim()",
        "session.start_run()",
        "session.attest_runtime()",
        "producer(session)",
        "runner.finalize(session)",
    ]
    positions = [source.index(fragment) for fragment in sequence]
    assert positions == sorted(positions)


def test_launcher_never_reads_or_passes_archive_payloads() -> None:
    source = _source()
    assert "read_archive_member" not in source
    assert "read_current_bytes" not in source
    assert "zipfile" not in source
    assert "producer(session)" in source


def test_launcher_contains_no_child_or_network_implementation() -> None:
    tree = _tree()
    names = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" not in names
    assert "socket" not in names
    source = _source()
    assert 'event.startswith("socket.")' in source
    assert '"subprocess.Popen"' in source


def test_primary_and_replay_share_one_combined_authorization_surface() -> None:
    source = _source()
    assert source.count('parser.add_argument("--pair-authorization"') == 1
    assert source.count('parser.add_argument("--expected-pair-authorization-sha256"') == 1
    assert 'choices=("primary", "replay")' in source
    assert "open_primary" in source and "open_replay" in source
