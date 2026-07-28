"""Build the prospective TRKH critical-family no-repeat registry.

The registry is intentionally derived from committed Git blobs, never from the
working tree.  This makes a build reproducible even while the research journal
and TODO contain unrelated, uncommitted work.  It is a machine-checkable
critical-family index, not an exhaustive semantic ontology, novelty proof, or
training authorization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Iterable


DEFAULT_SOURCE_REVISION = "742fce03700b17f8e8f4449eac8a49dc0a03e211"
DEFAULT_OUTPUT = Path("docs/TRKH_5CLASS_MECHANISM_REGISTRY_20260729.json")

HEADING_SOURCES = (
    "docs/TRKH_5CLASS_RESEARCH_JOURNAL.md",
    "docs/TODO_TRKH_5CLASS.md",
    "docs/TRKH_5CLASS_POST_CAP_SUCCESSOR_NO_RUN_SCREEN_20260725.md",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_PROTOCOL_20260725.md",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V1_SUPERSESSION_20260729.md",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md",
)

PINNED_SOURCES = (
    *HEADING_SOURCES,
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.sha256",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V1_SUPERSESSION_20260729.json",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.json",
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.sha256",
)

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class RegistryError(RuntimeError):
    """Raised when the prospective registry cannot be proven internally sound."""


def normalize_heading(value: str) -> str:
    """Return an ASCII, case-insensitive stable key for a Markdown heading."""

    value = value.strip()
    value = re.sub(r"\s+#+\s*$", "", value)
    value = value.replace("`", "")
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not value:
        raise RegistryError("heading normalization produced an empty key")
    return value


def _run_git(repo_root: Path, *args: str) -> bytes:
    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RegistryError(f"cannot execute git: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RegistryError(f"git command failed ({' '.join(command)}): {stderr}")
    return completed.stdout


def resolve_commit(repo_root: Path, revision: str) -> str:
    if not revision or revision.strip() != revision:
        raise RegistryError("source revision must be a non-empty, trimmed Git ref")
    raw = _run_git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    commit = raw.decode("ascii", errors="strict").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RegistryError(f"Git did not resolve a full SHA-1 commit: {commit!r}")
    return commit


def read_git_blob(repo_root: Path, commit: str, path: str) -> tuple[bytes, str]:
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise RegistryError(f"source path must be repository-relative: {path!r}")
    object_raw = _run_git(repo_root, "rev-parse", "--verify", f"{commit}:{path}")
    object_id = object_raw.decode("ascii", errors="strict").strip()
    if not _HEX_OBJECT_RE.fullmatch(object_id):
        raise RegistryError(f"invalid Git blob object id for {path}: {object_id!r}")
    object_type = _run_git(repo_root, "cat-file", "-t", object_id).decode("ascii").strip()
    if object_type != "blob":
        raise RegistryError(f"pinned source is not a blob: {path} ({object_type})")
    return _run_git(repo_root, "cat-file", "blob", object_id), object_id


def extract_headings(path: str, payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RegistryError(f"heading source is not strict UTF-8: {path}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _HEADING_RE.fullmatch(line)
        if match is None:
            continue
        title = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
        normalized = normalize_heading(title)
        records.append(
            {
                "document": path,
                "level": len(match.group(1)),
                "line": line_number,
                "normalized_heading": normalized,
                "title": title,
            }
        )
    if not records:
        raise RegistryError(f"no H2/H3 headings found in {path}")
    return records


def _heading_ref(document: str, line: int, title: str) -> dict[str, Any]:
    return {
        "document": document,
        "line": line,
        "normalized_heading": normalize_heading(title),
    }


JOURNAL = "docs/TRKH_5CLASS_RESEARCH_JOURNAL.md"
V1_SUPERSESSION = (
    "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V1_SUPERSESSION_20260729.md"
)
V2_PROTOCOL = "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V2_PROTOCOL_20260729.md"

# These entries are deliberately explicit rather than inferred from keywords.
# Their source lines are resolved against the pinned Git snapshot below; a line
# move or heading edit requires an intentional registry revision.
CURATED_MECHANISMS: tuple[dict[str, Any], ...] = (
    {
        "id": "pair_surface_ddf_a0_v1",
        "family": "position_conditioned_dynamic_filter",
        "aliases": ["DDF v1", "pair-surface DDF v1", "2026-07-25 DDF A0"],
        "status": "superseded_before_formal",
        "decision": (
            "The v1 protocol, lock and synthetic engine remain immutable engineering "
            "lineage, but its folds split manifest groups and numeric neighborhoods. "
            "It consumed zero formal and replay runs and cannot support an OOF claim."
        ),
        "reopen_or_advance_criterion": (
            "Never reopen the exact v1 fold assignment or formal authorization. Preserve "
            "it only as superseded lineage; any advance must use the separately pinned v2 "
            "protocol, fold manifest, engine, machine lock and authorization."
        ),
        "source_heading_refs": [
            _heading_ref(
                V1_SUPERSESSION,
                3,
                "Decision",
            )
        ],
    },
    {
        "id": "pair_surface_ddf_a0_v2",
        "family": "position_conditioned_dynamic_filter",
        "aliases": ["DDF v2", "pair-surface DDF v2", "validity-aware DDF sidecar"],
        "status": "prospective_protocol_and_fold_only_no_engine_or_formal_authorization",
        "decision": (
            "The component-disjoint fold manifest, validity-aware protocol and geometry "
            "preflight are prospectively pinned. No v2 engine, machine lock, candidate "
            "score, formal run, replay, validation result, test result or GPU run exists."
        ),
        "reopen_or_advance_criterion": (
            "Advance only after this registry is pushed, then implement and test the v2 "
            "engine/access ledger/replay, write a machine lock and explicit authorization, "
            "and consume exactly one formal plus one replay if every preflight passes."
        ),
        "source_heading_refs": [
            _heading_ref(
                V2_PROTOCOL,
                3,
                "Prospective state and supersession",
            )
        ],
    },
    {
        "id": "odconv",
        "family": "sample_conditioned_dynamic_convolution",
        "aliases": ["ODConv", "omni-dimensional dynamic convolution"],
        "status": "closed_no_run_overlap",
        "decision": (
            "Multiplicative sample-conditioned convolution overlaps failed stem routes "
            "and its official recipe requires schedules outside the current value gate."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only for a prospectively specified mechanism that is not an ODConv "
            "kernel/weight/schedule sweep and first proves new class-1-vs-0/2/4 "
            "train-only information within the locked resource envelope."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                16208,
                "ViG Max-Relative Graph Closure 2026-07-15 - Active Non-Local Mixing Broke Class-1 TP",
            )
        ],
    },
    {
        "id": "condconv",
        "family": "sample_conditioned_dynamic_convolution",
        "aliases": ["CondConv", "conditionally parameterized convolution"],
        "status": "closed_no_repeat_family",
        "decision": (
            "Generic CondConv-style sample-conditioned kernels remain inside the closed "
            "dynamic-convolution family and are not a DDF control or successor."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only if the operator has a mathematically distinct conditioning axis, "
            "a matched static control, prospective precision/recall gates, and evidence "
            "that cannot be obtained by changing expert count, routing, or kernel width."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                22226,
                "Pair-Surface DDF A0 Prospective Selection - 2026-07-25",
            )
        ],
    },
    {
        "id": "involution",
        "family": "position_specific_channel_shared_kernel",
        "aliases": ["Involution", "official involution replacement"],
        "status": "rejected_engineering_exact_geometry",
        "decision": (
            "Rejected at the keeper's 256-channel third-stem geometry because measured "
            "runtime and memory multipliers were prohibitive and the fast path needs "
            "CuPy/custom CUDA."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only with a standard-operator implementation measured at the exact "
            "deployment geometry that passes the predeclared runtime/memory limits and "
            "adds a distinct, train-only class boundary signal."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                17864,
                "Meta-ACON Activation Signal A0 Closure 2026-07-17 - Active But Not Selective Enough",
            )
        ],
    },
    {
        "id": "dynamic_convolution_generic",
        "family": "sample_conditioned_dynamic_convolution",
        "aliases": ["dynamic convolution", "dynamic kernels", "conditional kernels"],
        "status": "closed_no_repeat_family",
        "decision": (
            "Generic multiplicative or expert-mixed dynamic convolution is closed; the "
            "v1 DDF route is superseded and the narrowly specified v2 route is prospective "
            "only, not an authorized exception to execute."
        ),
        "reopen_or_advance_criterion": (
            "A proposal must prove a new conditioning/evidence pathway, identify why it "
            "does not reduce to ODConv/CondConv/Involution or a failed stem route, and pass "
            "a train-only information gate before trainer integration."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                22226,
                "Pair-Surface DDF A0 Prospective Selection - 2026-07-25",
            )
        ],
    },
    {
        "id": "pairwise_confusion_regularizer",
        "family": "pairwise_feature_regularization",
        "aliases": ["Pairwise Confusion", "pairwise confusion loss"],
        "status": "rejected_probe_precision_regression",
        "decision": (
            "The exact regularizer raised class-1 recall pressure but worsened precision "
            "and false positives; weight/source variants are not a new mechanism."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only with a genuinely new representation or causal signal that has a "
            "prospective false-positive control and matched class-1 precision/recall gate; "
            "do not repeat weight=0.005 with head/patch/register/logit sources."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                219,
                "Pairwise Confusion regularization worsened class-1 precision",
            )
        ],
    },
    {
        "id": "api_net_pooled_interaction",
        "family": "training_only_cross_image_interaction",
        "aliases": ["API-Net", "pooled-head API", "API pairwise interaction"],
        "status": "closed_train_only_readiness",
        "decision": (
            "Both symmetric and corrected anchor-only pooled API failed matched transfer "
            "and produced class-unsafe probability shifts; no image-model smoke was authorized."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only for a non-pooled, fold-safe interaction with natural anchor "
            "exposure, no partner-label amplification, a matched control, and locked "
            "class-1 false-positive plus true-positive preservation gates."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                13379,
                "Diagnostic 2026-07-11 - Pooled-Head API-Net Interaction Rejected",
            )
        ],
    },
    {
        "id": "pwca_token_natural_distractor",
        "family": "training_only_cross_image_token_attention",
        "aliases": ["PWCA", "DCAL PWCA", "natural-distractor PWCA"],
        "status": "closed_below_gate",
        "decision": (
            "PWCA showed a small positive relative direction but missed macro/class-1 "
            "gates and broke keeper class-1 true positives; no end-to-end smoke was authorized."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only from a prospectively locked keeper-preserving representation path "
            "with explicit recall protection; do not sweep projection rank, top-k, heads, "
            "dropout, optimizer, loss weight, pair distribution, or epochs on the closed proxy."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                13490,
                "Diagnostic 2026-07-11 - Token-Level Natural-Distractor PWCA Below Gate",
            )
        ],
    },
    {
        "id": "pwca_keeper_relative_scalar_residual",
        "family": "posthoc_pairwise_residual_calibration",
        "aliases": ["PWCA scalar residual", "keeper-relative PWCA residual"],
        "status": "closed_no_repeat",
        "decision": (
            "Only alpha=0 preserved the keeper constraints; scalar interpolation contains "
            "no usable class-1 correction signal and is closed."
        ),
        "reopen_or_advance_criterion": (
            "Reopen only for a new learned representation with fold-safe causal evidence; "
            "no alpha-grid, threshold, or other post-hoc interpolation sweep may reuse this closure."
        ),
        "source_heading_refs": [
            _heading_ref(
                JOURNAL,
                13573,
                "Diagnostic 2026-07-11 - Keeper-Relative PWCA Scalar Residual Rejected",
            )
        ],
    },
)


def materialize_committed_text(payload: bytes) -> bytes:
    """Reproduce a historical Windows CRLF lock payload without reading it.

    The 2026-07-25 v1 lock sidecar was computed on its CRLF worktree bytes.
    Registry source records themselves use canonical committed Git blobs; this
    helper exists only to verify that historical sidecar without rewriting it.
    """

    if b"\r" in payload:
        raise RegistryError("committed text blob unexpectedly contains CR bytes")
    return payload.replace(b"\n", b"\r\n")


def _source_record(path: str, payload: bytes, object_id: str) -> dict[str, Any]:
    return {
        "byte_representation": "canonical_committed_git_blob",
        "bytes": len(payload),
        "git_blob_oid": object_id,
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _parse_lock_sidecar(payload: bytes, expected_name: str) -> str:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise RegistryError("DDF lock SHA sidecar is not strict ASCII") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\r?\n?", text)
    if match is None or match.group(2) != expected_name:
        raise RegistryError("DDF lock SHA sidecar has an invalid fail-closed format")
    return match.group(1)


def _resolve_curated_entries(
    headings: list[dict[str, Any]],
    curated_entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    index: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for heading in headings:
        key = (
            heading["document"],
            heading["line"],
            heading["normalized_heading"],
        )
        index.setdefault(key, []).append(heading)

    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_entry in curated_entries:
        entry = copy.deepcopy(raw_entry)
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise RegistryError("curated mechanism id must be a non-empty string")
        if entry_id in seen_ids:
            raise RegistryError(f"duplicate curated mechanism id: {entry_id}")
        seen_ids.add(entry_id)

        refs = entry.get("source_heading_refs")
        if not isinstance(refs, list) or not refs:
            raise RegistryError(f"curated mechanism has no source refs: {entry_id}")
        seen_refs: set[tuple[str, int, str]] = set()
        resolved_refs: list[dict[str, Any]] = []
        for ref in refs:
            try:
                key = (
                    str(ref["document"]),
                    int(ref["line"]),
                    str(ref["normalized_heading"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryError(f"invalid source ref in {entry_id}: {ref!r}") from exc
            if key in seen_refs:
                raise RegistryError(f"duplicate source ref in {entry_id}: {key!r}")
            seen_refs.add(key)
            matches = index.get(key, [])
            if len(matches) != 1:
                raise RegistryError(
                    f"source ref in {entry_id} resolved {len(matches)} times: {key!r}"
                )
            resolved_refs.append(copy.deepcopy(matches[0]))
        entry["source_heading_refs"] = resolved_refs
        resolved.append(entry)
    return sorted(resolved, key=lambda item: item["id"])


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != 2:
        raise RegistryError("unsupported mechanism-registry schema")
    if registry.get("state") != (
        "critical_family_no_repeat_registry_no_training_authorization"
    ):
        raise RegistryError("mechanism-registry state is missing or unsafe")
    coverage = registry.get("coverage")
    if not isinstance(coverage, dict):
        raise RegistryError("registry coverage contract is missing")
    expected_coverage = {
        "authorizes_formal_or_gpu": False,
        "curated_semantic_scope": (
            "critical_mechanism_families_only_not_an_exhaustive_ontology"
        ),
        "heading_index_complete_for_pinned_markdown": True,
        "proves_novelty": False,
    }
    if coverage != expected_coverage:
        raise RegistryError("registry coverage contract differs")
    snapshot = registry.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise RegistryError("source_snapshot is missing")
    if snapshot.get("read_mode") != "git_object_database_only_no_worktree":
        raise RegistryError("registry does not fail closed to committed Git blobs")
    commit = snapshot.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RegistryError("source snapshot does not pin a full Git commit")

    sources = snapshot.get("files")
    if not isinstance(sources, list) or len(sources) != len(PINNED_SOURCES):
        raise RegistryError("source snapshot has an incomplete pinned-file set")
    source_paths: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            raise RegistryError("source record is not an object")
        path = source.get("path")
        source_paths.append(path)
        if not isinstance(source.get("bytes"), int) or source["bytes"] <= 0:
            raise RegistryError(f"source byte count is invalid: {path}")
        if not isinstance(source.get("sha256"), str) or not _HEX_64_RE.fullmatch(
            source["sha256"]
        ):
            raise RegistryError(f"source SHA-256 is invalid: {path}")
        if not isinstance(source.get("git_blob_oid"), str) or not _HEX_OBJECT_RE.fullmatch(
            source["git_blob_oid"]
        ):
            raise RegistryError(f"source Git blob id is invalid: {path}")
        if source.get("byte_representation") != "canonical_committed_git_blob":
            raise RegistryError(f"source byte representation is invalid: {path}")
    if len(source_paths) != len(set(source_paths)):
        raise RegistryError("duplicate pinned source paths")
    if tuple(source_paths) != PINNED_SOURCES:
        raise RegistryError("pinned source paths/order do not match the schema")

    headings = registry.get("headings")
    if not isinstance(headings, list) or not headings:
        raise RegistryError("heading index is empty")
    heading_keys: set[tuple[str, int]] = set()
    for heading in headings:
        key = (heading.get("document"), heading.get("line"))
        if key in heading_keys:
            raise RegistryError(f"duplicate heading location: {key!r}")
        heading_keys.add(key)
        if heading.get("document") not in HEADING_SOURCES:
            raise RegistryError(f"heading came from an unauthorized source: {key!r}")
        if heading.get("level") not in (2, 3):
            raise RegistryError(f"non-H2/H3 heading in registry: {key!r}")
        if heading.get("normalized_heading") != normalize_heading(heading.get("title", "")):
            raise RegistryError(f"unstable heading normalization: {key!r}")

    mechanisms = registry.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        raise RegistryError("mechanism list is empty")
    required_ids = {
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
    ids = [entry.get("id") for entry in mechanisms]
    if len(ids) != len(set(ids)):
        raise RegistryError("duplicate mechanism ids")
    if not required_ids.issubset(ids):
        raise RegistryError(f"required curated mechanisms are missing: {required_ids - set(ids)}")
    if "pair_surface_ddf_a0" in ids:
        raise RegistryError("ambiguous unsuffixed DDF registry id is forbidden")
    by_id = {entry["id"]: entry for entry in mechanisms}
    expected_ddf_statuses = {
        "pair_surface_ddf_a0_v1": "superseded_before_formal",
        "pair_surface_ddf_a0_v2": (
            "prospective_protocol_and_fold_only_no_engine_or_formal_authorization"
        ),
    }
    for entry_id, expected_status in expected_ddf_statuses.items():
        if by_id[entry_id].get("status") != expected_status:
            raise RegistryError(f"unsafe DDF status for {entry_id}")
    heading_index = {
        (
            heading["document"],
            heading["line"],
            heading["normalized_heading"],
        ): heading
        for heading in headings
    }
    for entry in mechanisms:
        refs = entry.get("source_heading_refs")
        if not isinstance(refs, list) or not refs:
            raise RegistryError(f"mechanism lacks source headings: {entry.get('id')}")
        ref_keys: list[tuple[str, int, str]] = []
        for ref in refs:
            key = (ref.get("document"), ref.get("line"), ref.get("normalized_heading"))
            ref_keys.append(key)
            if key not in heading_index or ref != heading_index[key]:
                raise RegistryError(
                    f"mechanism source heading is missing or non-canonical: {entry.get('id')}"
                )
        if len(ref_keys) != len(set(ref_keys)):
            raise RegistryError(f"duplicate source refs in mechanism: {entry.get('id')}")
        for field in ("status", "decision", "reopen_or_advance_criterion"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise RegistryError(f"mechanism {entry.get('id')} lacks {field}")


def build_registry(repo_root: Path, source_revision: str = DEFAULT_SOURCE_REVISION) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    commit = resolve_commit(repo_root, source_revision)
    blobs: dict[str, bytes] = {}
    source_records: list[dict[str, Any]] = []
    for path in PINNED_SOURCES:
        payload, object_id = read_git_blob(repo_root, commit, path)
        blobs[path] = payload
        source_records.append(_source_record(path, payload, object_id))

    lock_path = "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.json"
    sidecar_path = "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_LOCK_20260725.sha256"
    expected_lock_sha = _parse_lock_sidecar(blobs[sidecar_path], Path(lock_path).name)
    actual_lock_sha = hashlib.sha256(materialize_committed_text(blobs[lock_path])).hexdigest()
    if actual_lock_sha != expected_lock_sha:
        raise RegistryError(
            f"DDF lock sidecar mismatch: expected {expected_lock_sha}, got {actual_lock_sha}"
        )
    try:
        json.loads(blobs[lock_path].decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError("DDF lock is not valid strict UTF-8 JSON") from exc

    v2_fold_path = "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.json"
    v2_fold_sidecar = (
        "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_V2_FOLD_MANIFEST_20260729.sha256"
    )
    expected_v2_fold_sha = _parse_lock_sidecar(
        blobs[v2_fold_sidecar], Path(v2_fold_path).name
    )
    actual_v2_fold_sha = hashlib.sha256(blobs[v2_fold_path]).hexdigest()
    if actual_v2_fold_sha != expected_v2_fold_sha:
        raise RegistryError(
            "v2 fold-manifest sidecar mismatch: "
            f"expected {expected_v2_fold_sha}, got {actual_v2_fold_sha}"
        )
    for json_path in (
        "docs/TRKH_5CLASS_PAIR_SURFACE_DDF_A0_V1_SUPERSESSION_20260729.json",
        v2_fold_path,
    ):
        try:
            value = json.loads(blobs[json_path].decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError(f"pinned source is not strict UTF-8 JSON: {json_path}") from exc
        if not isinstance(value, dict):
            raise RegistryError(f"pinned JSON source is not an object: {json_path}")

    headings: list[dict[str, Any]] = []
    for path in HEADING_SOURCES:
        headings.extend(extract_headings(path, blobs[path]))
    headings.sort(key=lambda item: (item["document"], item["line"]))

    registry = {
        "coverage": {
            "authorizes_formal_or_gpu": False,
            "curated_semantic_scope": (
                "critical_mechanism_families_only_not_an_exhaustive_ontology"
            ),
            "heading_index_complete_for_pinned_markdown": True,
            "proves_novelty": False,
        },
        "heading_normalization": {
            "algorithm": "trim-trailing-hashes-casefold-nfkd-strip-combining-ascii-alnum-hyphen",
            "scope": "all Markdown H2/H3 headings in every pinned Markdown heading source",
        },
        "mechanisms": _resolve_curated_entries(headings, CURATED_MECHANISMS),
        "purpose": (
            "Prospective machine-checkable critical-family no-repeat boundary for TRKH "
            "5-class research. It is not an exhaustive semantic ontology, novelty proof, "
            "training, validation, test, candidate, formal, replay, or GPU authorization."
        ),
        "schema_version": 2,
        "state": "critical_family_no_repeat_registry_no_training_authorization",
        "source_snapshot": {
            "files": source_records,
            "git_commit": commit,
            "read_mode": "git_object_database_only_no_worktree",
        },
        "headings": headings,
    }
    validate_registry(registry)
    return registry


def render_registry(registry: dict[str, Any]) -> bytes:
    validate_registry(registry)
    return (json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def render_sidecar(registry_bytes: bytes, registry_name: str) -> bytes:
    digest = hashlib.sha256(registry_bytes).hexdigest()
    return f"{digest}  {registry_name}\n".encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _output_paths(repo_root: Path, output: Path) -> tuple[Path, Path]:
    registry_path = output if output.is_absolute() else repo_root / output
    if registry_path.suffix.lower() != ".json":
        raise RegistryError("registry output must use a .json suffix")
    return registry_path, registry_path.with_suffix(".sha256")


def write_registry(repo_root: Path, output: Path, registry_bytes: bytes) -> tuple[Path, Path]:
    registry_path, sidecar_path = _output_paths(repo_root, output)
    _atomic_write(registry_path, registry_bytes)
    _atomic_write(sidecar_path, render_sidecar(registry_bytes, registry_path.name))
    return registry_path, sidecar_path


def check_registry(repo_root: Path, output: Path, registry_bytes: bytes) -> tuple[Path, Path]:
    registry_path, sidecar_path = _output_paths(repo_root, output)
    if not registry_path.is_file() or not sidecar_path.is_file():
        raise RegistryError("registry or SHA-256 sidecar is missing")
    actual_registry = registry_path.read_bytes()
    actual_sidecar = sidecar_path.read_bytes()
    expected_sidecar = render_sidecar(registry_bytes, registry_path.name)
    if actual_registry != registry_bytes:
        raise RegistryError("registry differs from deterministic rebuild")
    if actual_sidecar != expected_sidecar:
        raise RegistryError("registry SHA-256 sidecar differs from deterministic rebuild")
    return registry_path, sidecar_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="TRKH Git repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="committed Git revision to read; working-tree files are never read",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="explicitly write the deterministic JSON and SHA-256 sidecar",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify existing files byte-for-byte without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        registry = build_registry(args.repo_root, args.source_revision)
        payload = render_registry(registry)
        if args.write:
            registry_path, sidecar_path = write_registry(args.repo_root, args.output, payload)
            print(f"wrote {registry_path}")
            print(f"wrote {sidecar_path}")
        elif args.check:
            registry_path, sidecar_path = check_registry(args.repo_root, args.output, payload)
            print(f"verified {registry_path}")
            print(f"verified {sidecar_path}")
        else:
            # Inspection is safe by default and intentionally has no filesystem side effect.
            sys.stdout.buffer.write(payload)
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
