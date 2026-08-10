from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_trkh_mechanism_registry.py"
REGISTRY_PATH = REPO_ROOT / "docs" / "TRKH_5CLASS_MECHANISM_REGISTRY_20260729.json"
SIDECAR_PATH = REGISTRY_PATH.with_suffix(".sha256")


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_trkh_mechanism_registry", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def test_heading_normalization_is_stable_and_accent_insensitive(builder):
    assert builder.normalize_heading("  Cập nhật `DDF` / A0 ###  ") == "cap-nhat-ddf-a0"
    assert builder.normalize_heading("PAIR–Surface  DDF") == "pair-surface-ddf"
    with pytest.raises(builder.RegistryError, match="empty"):
        builder.normalize_heading("###")


def test_build_reads_only_the_pinned_git_object_database(builder, monkeypatch):
    def _forbid_worktree_read(_self):
        raise AssertionError("build_registry must not read any working-tree Path")

    monkeypatch.setattr(Path, "read_bytes", _forbid_worktree_read)
    registry = builder.build_registry(REPO_ROOT)
    assert registry["source_snapshot"]["read_mode"] == "git_object_database_only_no_worktree"
    assert registry["source_snapshot"]["git_commit"] == builder.DEFAULT_SOURCE_REVISION


def test_every_required_source_pins_full_bytes_sha_and_git_blob(builder):
    registry = builder.build_registry(REPO_ROOT)
    source_records = registry["source_snapshot"]["files"]
    assert tuple(record["path"] for record in source_records) == builder.PINNED_SOURCES
    for record in source_records:
        payload, object_id = builder.read_git_blob(
            REPO_ROOT,
            builder.DEFAULT_SOURCE_REVISION,
            record["path"],
        )
        assert record["byte_representation"] == "canonical_committed_git_blob"
        assert record["bytes"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["git_blob_oid"] == object_id


def test_heading_index_contains_every_h2_h3_from_journal_and_todo(builder):
    registry = builder.build_registry(REPO_ROOT)
    actual = registry["headings"]
    expected = []
    for path in builder.HEADING_SOURCES:
        payload, _ = builder.read_git_blob(REPO_ROOT, builder.DEFAULT_SOURCE_REVISION, path)
        expected.extend(builder.extract_headings(path, payload))
    expected.sort(key=lambda item: (item["document"], item["line"]))
    assert actual == expected
    assert {heading["level"] for heading in actual} == {2, 3}
    assert len(actual) > 500


def test_curated_no_repeat_entries_are_present_and_canonically_referenced(builder):
    registry = builder.build_registry(REPO_ROOT)
    by_id = {entry["id"]: entry for entry in registry["mechanisms"]}
    required = {
        "pair_surface_ddf_a0_v1",
        "pair_surface_ddf_a0_v2",
        "odconv",
        "condconv",
        "involution",
        "dynamic_convolution_generic",
        "pairwise_confusion_regularizer",
        "api_net_pooled_interaction",
        "pwca_token_natural_distractor",
        "pwca_keeper_relative_scalar_residual",
    }
    assert required <= set(by_id)
    for entry_id in required:
        entry = by_id[entry_id]
        assert entry["status"]
        assert entry["decision"]
        assert entry["reopen_or_advance_criterion"]
        assert entry["source_heading_refs"]
        for ref in entry["source_heading_refs"]:
            assert ref["level"] in (2, 3)
            assert ref["normalized_heading"] == builder.normalize_heading(ref["title"])
    assert "pair_surface_ddf_a0" not in by_id
    assert by_id["pair_surface_ddf_a0_v1"]["status"] == "superseded_before_formal"
    assert by_id["pair_surface_ddf_a0_v2"]["status"] == (
        "prospective_v2r2_engine_pinned_no_machine_lock_or_formal_authorization"
    )


def test_registry_discloses_bounded_coverage_and_cannot_authorize_gpu(builder):
    registry = builder.build_registry(REPO_ROOT)
    assert registry["schema_version"] == 3
    assert registry["state"] == (
        "critical_family_no_repeat_registry_no_training_authorization"
    )
    assert registry["coverage"] == {
        "authorizes_formal_or_gpu": False,
        "curated_semantic_scope": (
            "critical_mechanism_families_only_not_an_exhaustive_ontology"
        ),
        "heading_index_complete_for_pinned_markdown": True,
        "proves_novelty": False,
    }
    assert "not an exhaustive semantic ontology" in registry["purpose"]
    boundary = registry["v2_effective_boundary"]
    assert boundary["state"] == (
        "v2r2_protocol_and_synthetic_engine_frozen_pre_machine_lock"
    )
    assert boundary["authorizes_formal_or_gpu"] is False
    assert boundary["execution_evidence"] == {
        "candidate_scores_observed": False,
        "formal_runs": 0,
        "gpu_executions": 0,
        "replay_runs": 0,
        "validation_or_test_results": 0,
    }
    assert boundary["fold_identity"] == {
        "component_count": 158,
        "fold_rows": [153, 153, 153, 152, 152],
        "manifest_schema": "trkh_pair_surface_ddf_v2_fold_manifest/v2",
        "mapping_sha256": builder.EXPECTED_V2_MAPPING_SHA256,
    }
    assert boundary["scientific_sources"] == [
        {"path": path, "sha256": sha256}
        for path, sha256 in builder.EXPECTED_V2_EFFECTIVE_SHA256.items()
    ]


def test_missing_duplicate_ids_and_duplicate_refs_fail_closed(builder):
    headings = builder.build_registry(REPO_ROOT)["headings"]

    duplicated_id = [
        copy.deepcopy(builder.CURATED_MECHANISMS[0]),
        copy.deepcopy(builder.CURATED_MECHANISMS[0]),
    ]
    with pytest.raises(builder.RegistryError, match="duplicate curated mechanism id"):
        builder._resolve_curated_entries(headings, duplicated_id)

    missing_ref = copy.deepcopy(builder.CURATED_MECHANISMS[0])
    missing_ref["source_heading_refs"][0]["line"] += 1
    with pytest.raises(builder.RegistryError, match="resolved 0 times"):
        builder._resolve_curated_entries(headings, [missing_ref])

    duplicated_ref = copy.deepcopy(builder.CURATED_MECHANISMS[0])
    duplicated_ref["source_heading_refs"].append(
        copy.deepcopy(duplicated_ref["source_heading_refs"][0])
    )
    with pytest.raises(builder.RegistryError, match="duplicate source ref"):
        builder._resolve_curated_entries(headings, [duplicated_ref])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorizes_formal_or_gpu", True),
        ("state", "formal_authorized"),
    ],
)
def test_v2_effective_boundary_tamper_fails_closed(builder, field, value):
    registry = builder.build_registry(REPO_ROOT)
    registry["v2_effective_boundary"][field] = value
    with pytest.raises(builder.RegistryError, match="v2 effective boundary differs"):
        builder.validate_registry(registry)


def test_v2_effective_source_hash_tamper_fails_closed(builder):
    registry = builder.build_registry(REPO_ROOT)
    registry["v2_effective_boundary"]["scientific_sources"][-1]["sha256"] = "0" * 64
    with pytest.raises(builder.RegistryError, match="v2 effective boundary differs"):
        builder.validate_registry(registry)


def test_deterministic_render_matches_checked_in_registry_and_sidecar(builder):
    first = builder.render_registry(builder.build_registry(REPO_ROOT))
    second = builder.render_registry(builder.build_registry(REPO_ROOT))
    assert first == second
    assert REGISTRY_PATH.read_bytes() == first
    expected_sidecar = builder.render_sidecar(first, REGISTRY_PATH.name)
    assert SIDECAR_PATH.read_bytes() == expected_sidecar
    assert json.loads(first)["schema_version"] == 3


def test_cli_requires_explicit_write_and_check_fails_on_tamper(builder, tmp_path):
    output = tmp_path / "registry.json"
    base_command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--repo-root",
        str(REPO_ROOT),
        "--output",
        str(output),
    ]

    inspect = subprocess.run(base_command, check=False, capture_output=True)
    assert inspect.returncode == 0, inspect.stderr.decode(errors="replace")
    assert not output.exists()
    assert not output.with_suffix(".sha256").exists()
    assert json.loads(inspect.stdout)["schema_version"] == 3

    write = subprocess.run([*base_command, "--write"], check=False, capture_output=True)
    assert write.returncode == 0, write.stderr.decode(errors="replace")
    assert output.is_file()
    assert output.with_suffix(".sha256").is_file()

    check = subprocess.run([*base_command, "--check"], check=False, capture_output=True)
    assert check.returncode == 0, check.stderr.decode(errors="replace")

    output.with_suffix(".sha256").write_text("0" * 64 + "  registry.json\n", encoding="ascii")
    tampered = subprocess.run([*base_command, "--check"], check=False, capture_output=True)
    assert tampered.returncode == 2
    assert b"sidecar differs" in tampered.stderr
