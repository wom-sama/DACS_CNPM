from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from trkh.tools import pair_surface_ddf_v2_prelock_validator as validator


H40_SOURCE_S1 = "1" * 40
H40_AUTH_S1 = "2" * 40
H40_SOURCE_S2 = "3" * 40
H40_AUTH_S2 = "4" * 40
T = {
    "s1_source": "2026-07-29T00:00:00+00:00",
    "s1_auth": "2026-07-29T00:01:00+00:00",
    "s1_primary_start": "2026-07-29T00:02:00+00:00",
    "s1_primary_finish": "2026-07-29T00:03:00+00:00",
    "s1_handoff": "2026-07-29T00:04:00+00:00",
    "s1_replay_start": "2026-07-29T00:05:00+00:00",
    "s1_replay_finish": "2026-07-29T00:06:00+00:00",
    "s1_comparison": "2026-07-29T00:07:00+00:00",
    "s1_bundle": "2026-07-29T00:08:00+00:00",
    "s1_root": "2026-07-29T00:09:00+00:00",
    "s2_source": "2026-07-29T01:00:00+00:00",
    "s2_auth": "2026-07-29T01:01:00+00:00",
    "s2_primary_start": "2026-07-29T01:02:00+00:00",
    "s2_primary_finish": "2026-07-29T01:03:00+00:00",
    "s2_handoff": "2026-07-29T01:04:00+00:00",
    "s2_replay_start": "2026-07-29T01:05:00+00:00",
    "s2_replay_finish": "2026-07-29T01:06:00+00:00",
    "s2_comparison": "2026-07-29T01:07:00+00:00",
    "s2_bundle": "2026-07-29T01:08:00+00:00",
    "s2_root": "2026-07-29T01:09:00+00:00",
}
AUTHORIZED_NODE_NAMES = (
    "test_staged_s1_then_s2_then_guard_receipt",
    "test_s2_rejects_missing_or_forged_verified_s1",
    "test_boundary_rejects_extra_role_and_fixed_hash_drift",
    "test_authorization_requires_distinct_later_pushed_commit",
    "test_s1_rejects_arbitrary_pass_boolean",
    "test_s2a_target_alias_and_extra_ledger_access_fail_closed",
    "test_exact_cohort_fold_target_component_preimages",
    "test_exact_xai_and_cidt_typed_preimages",
    "test_pcg_epoch_order_and_parameter_schema_initial_state_are_recomputed",
    "test_temporal_handoff_and_final_root_order_is_strict",
    "test_cross_phase_process_and_root_reuse_is_rejected",
    "test_run_out_rejects_parent_absolute_duplicate_and_physical_alias",
    "test_no_reopen_delegation_requires_guard_bound_unique_single_link_identities",
)


def _doc(role: str, payload: object, state: str | None = None) -> dict[str, object]:
    return {
        "schema_version": validator.SCHEMA_VERSION,
        "protocol_id": validator.PROTOCOL_ID,
        "kind": role,
        "state": validator.DOCUMENT_STATES.get(role) if state is None else state,
        "payload": payload,
    }


def _record(path: Path, data: bytes, access_class: str = "finalized_evidence") -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "access_class": access_class,
    }


def _json_record(path: Path, value: object, access_class: str = "finalized_evidence") -> dict[str, object]:
    return _record(path, validator._canonical_bytes(value), access_class)


def _typed(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data_hex": value.tobytes().hex(),
        "array_sha256": validator._array_sha(value),
    }


def _state_digest(rows: list[dict[str, object]]) -> str:
    return validator._sha(
        [
            {
                "name": row["name"],
                "dtype": row["dtype"],
                "shape": row["shape"],
                "requires_grad": row["requires_grad"],
                "array_sha256": row["array_sha256"],
            }
            for row in rows
        ]
    )


def _tensor_row(spec: dict[str, object], value: float) -> dict[str, object]:
    array = np.full(tuple(spec["shape"]), value, dtype=np.dtype(str(spec["dtype"])))
    return {"name": spec["name"], "requires_grad": spec["requires_grad"], **_typed(array)}


def _zero_authority() -> dict[str, int]:
    return dict(validator.ZERO_AUTHORITY)


class StagedFixture:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self.records: dict[str, dict[str, object]] = {}
        self.s1_contract: dict[str, object] = {}
        self.s2_contract: dict[str, object] = {}
        self.s1_engineering: dict[str, object] = {}
        self.s2_scientific: dict[str, object] = {}
        self.s2_cohort: dict[str, object] = {}
        self.initial_states: dict[str, list[dict[str, object]]] = {}
        self.parameter_contracts = self._parameter_contracts()
        self._lock_small_constants()
        self._build_boundary()
        self._build_s1()
        self.verified_s1 = validator.verify_finalized_s1_evidence(artifacts=self.records)
        self._build_s2()

    def _parameter_contracts(self) -> dict[str, list[dict[str, object]]]:
        full = [
            {"name": "common.weight", "dtype": "float32", "shape": [1], "requires_grad": True},
            {"name": "block.spatial_projection.weight", "dtype": "float32", "shape": [1], "requires_grad": True},
            {"name": "block.channel_reduce.weight", "dtype": "float32", "shape": [1], "requires_grad": True},
        ]
        spatial = copy.deepcopy(full)
        spatial[2]["requires_grad"] = False
        channel = copy.deepcopy(full)
        channel[1]["requires_grad"] = False
        static = [
            {"name": "common.weight", "dtype": "float32", "shape": [1], "requires_grad": True},
            {"name": "static.extra", "dtype": "float32", "shape": [1], "requires_grad": True},
        ]
        return {
            "ddf_full": full,
            "static_matched": static,
            "ddf_spatial_only": spatial,
            "ddf_channel_only": channel,
            "ddf_full_repeat": copy.deepcopy(full),
        }

    def _lock_small_constants(self) -> None:
        indices = np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64)
        folds = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        targets = np.asarray([0, 1, 1, 2, 1, 4], dtype=np.int64)
        components = ["a" * 64, "a" * 64, "b" * 64, "b" * 64, "c" * 64, "c" * 64]
        order = ["a" * 64, "b" * 64, "c" * 64]
        pairs = [[int(index), int(fold)] for index, fold in zip(indices, folds)]
        mapping = {str(int(index)): int(fold) for index, fold in zip(indices, folds)}
        cohort_identity = {
            "sample_indices_array_sha256": validator._array_sha(indices),
            "held_folds_array_sha256": validator._array_sha(folds),
            "targets_array_sha256": validator._array_sha(targets),
            "sample_fold_pairs_sha256": validator._sha(pairs),
            "fold_mapping_sha256": validator._sha(mapping),
            "component_order_sha256": validator._sha(order),
        }
        xai = (10, 11, 15)
        cidt_targets = targets.copy()
        cidt_predictions = np.asarray([0, 1, 1, 2, 0, 4], dtype=np.int64)
        noise_shape = (6, 3, 2, 2)
        bootstrap_shape = (4, 3)
        noise = validator._pcg_integers(validator.PRIMARY_SEED, 0, 256, noise_shape, "uint8")
        bootstrap = validator._pcg_integers(validator.PRIMARY_SEED, 0, 3, bootstrap_shape, "int64")
        values = {
            "ROWS": 6,
            "FOLDS": 3,
            "COMPONENTS": 3,
            "FOLD_SIZES": (2, 2, 2),
            "TARGET_COUNTS": {0: 1, 1: 3, 2: 1, 3: 0, 4: 1},
            "TARGET_SHARD_KEYS": tuple(
                f"fold_{fold}_{split}"
                for fold in range(3)
                for split in ("fit", "calibration", "held")
            ),
            "XAI_SAMPLE_INDICES": xai,
            "XAI_ARRAY_SHA256": validator._array_sha(np.asarray(xai, dtype=np.int64)),
            "CIDT_IDENTITY": {
                "rows": 6,
                "targets_array_sha256": validator._array_sha(cidt_targets),
                "keeper_predictions_array_sha256": validator._array_sha(cidt_predictions),
            },
            "COHORT_IDENTITY": cohort_identity,
            "NOISE_SHAPE": noise_shape,
            "BOOTSTRAP_SHAPE": bootstrap_shape,
            "NOISE_ARRAY_SHA256": validator._array_sha(noise),
            "BOOTSTRAP_ARRAY_SHA256": validator._array_sha(bootstrap),
            "EPOCHS": 2,
            "S1_SPATIAL_SHAPE": (3, 4, 4),
            "PARAMETER_COUNTS": {
                "ddf_full": 3,
                "static_matched": 2,
                "ddf_spatial_only": 3,
                "ddf_channel_only": 3,
                "ddf_full_repeat": 3,
            },
        }
        for name, value in values.items():
            self.monkeypatch.setattr(validator, name, value)
        self.indices = indices
        self.folds = folds
        self.targets = targets
        self.components = components
        self.component_order = order
        self.cidt_targets = cidt_targets
        self.cidt_predictions = cidt_predictions

    def _role_bytes(self, role: str) -> bytes:
        return validator._canonical_bytes({"synthetic_role": role})

    def _write_role(self, role: str, value: object) -> None:
        self.records[role] = _json_record(self.root / "top" / f"{role}.json", value)

    def _build_boundary(self) -> None:
        for role in sorted(validator.ALL_ARTIFACT_ROLES - validator.NO_REOPEN_ROLES):
            self.records[role] = _record(self.root / "top" / f"{role}.json", self._role_bytes(role))
        for role in sorted(validator.NO_REOPEN_ROLES):
            self.records[role] = {
                "path": str((self.root / "sealed_never_created" / f"{role}.bin").resolve()),
                "bytes": 17,
                "sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
                "access_class": (
                    "lineage_only_forbidden_runtime"
                    if "checkpoint" in role or "config" in role
                    else "provenance_only_no_reopen"
                ),
            }
        self.monkeypatch.setattr(
            validator,
            "FIXED_BOUNDARY_SHA256",
            {role: self.records[role]["sha256"] for role in validator.FIXED_BOUNDARY_SHA256},
        )
        identities = {
            role: {
                "canonical_path": self.records[role]["path"],
                "real_path": self.records[role]["path"],
                "volume_serial": "synthetic-volume",
                "file_id": f"synthetic-{position}",
                "link_count": 1,
                "reparse_tag": 0,
            }
            for position, role in enumerate(sorted(validator.NO_REOPEN_ROLES))
        }
        guard_payload = {
            "guard_sha256": self.records["v2_execution_guard"]["sha256"],
            "protected_records": {role: self.records[role] for role in sorted(validator.NO_REOPEN_ROLES)},
            "physical_identities": identities,
            "all_paths_canonical": True,
            "reparse_points_zero": 0,
            "hardlinks_zero": 0,
            "identity_collisions_zero": 0,
            "delegation_scope": "identity_only_no_content_reopen_by_prelock_verifier",
            "downstream_authority": _zero_authority(),
        }
        self._write_role("v2_guard_identity_evidence", _doc("v2_guard_identity_evidence", guard_payload))
        nodeids = [
            f"tests/test_pair_surface_ddf_v2_prelock_validator.py::{name}"
            for name in AUTHORIZED_NODE_NAMES
        ]
        authority = {
            "test_source_sha256": self.records["v2_prelock_validator_tests"]["sha256"],
            "launcher_sha256": self.records["v2_prelock_test_launcher"]["sha256"],
            "allowed_nodeids": nodeids,
            "fixture_scope": "synthetic_tmp_only_no_repository_cache_dataset_gpu",
            "precollection_tripwire_required": True,
            "forbidden_access_classes": ["cache", "raw", "validation", "test", "gpu"],
            "live_integration": {"collection_authorized": False, "execution_authorized": False},
            "downstream_authority": _zero_authority(),
        }
        self._write_role(
            "v2_prelock_test_authority_manifest",
            _doc("v2_prelock_test_authority_manifest", authority),
        )
        tests_manifest = {
            "authority_sha256": self.records["v2_prelock_test_authority_manifest"]["sha256"],
            "production_sha256": self.records["v2_prelock_validator_source"]["sha256"],
            "tests_sha256": self.records["v2_prelock_validator_tests"]["sha256"],
            "launcher_sha256": self.records["v2_prelock_test_launcher"]["sha256"],
            "passed": True,
            "test_count": 15,
            "negative_categories": [f"blocker_{index}" for index in range(11)],
            "positive_staged_e2e": True,
            "fresh_subprocess_passed": True,
            "precollection_tripwire_passed": True,
            "forbidden_access_counts": {"cache": 0, "raw": 0, "validation": 0, "test": 0, "gpu": 0},
            "live_integration": {"collected": 0, "executed": 0},
            "downstream_authority": _zero_authority(),
        }
        self._write_role("v2_prelock_tests_manifest", _doc("v2_prelock_tests_manifest", tests_manifest))
        registry = {
            "registry_version": 2,
            "foundation_commit": validator.FOUNDATION_COMMIT,
            "incident_sha256": self.records["v2_prelock_process_incident"]["sha256"],
            "hashes": {
                role: self.records[role]["sha256"]
                for role in sorted(validator.PRELOCK_REGISTRY_HASH_ROLES)
            },
            "no_reopen_roles": sorted(validator.NO_REOPEN_ROLES),
            "downstream_authority": _zero_authority(),
        }
        self._write_role("v2_prelock_boundary_registry", _doc("v2_prelock_boundary_registry", registry))

    def _authorization(
        self,
        stage: str,
        contract_sha: str,
        primary_root: Path,
        replay_root: Path,
    ) -> dict[str, object]:
        core = {
            "phase": validator.S1_PHASE if stage == "s1" else validator.S2_PHASE,
            "contract_sha256": contract_sha,
            "source_commit": H40_SOURCE_S1 if stage == "s1" else H40_SOURCE_S2,
            "authorization_commit": H40_AUTH_S1 if stage == "s1" else H40_AUTH_S2,
            "source_commit_pushed": True,
            "authorization_commit_pushed": True,
            "source_is_ancestor": True,
            "source_committed_utc": T[f"{stage}_source"],
            "authorization_committed_utc": T[f"{stage}_auth"],
            "primary_output_root": str(primary_root.resolve()),
            "replay_output_root": str(replay_root.resolve()),
            "quotas": {"primary_processes": 1, "fresh_replays": 1},
            "consumed": True,
            "downstream_authority": _zero_authority(),
        }
        return {**core, "authorization_id": validator._sha(core)}

    def _output_records(self, stage: str, process_role: str) -> tuple[dict[str, object], dict[str, str]]:
        roles = validator.S1_OUTPUT_ROLES if stage == "s1" else validator.S2_OUTPUT_ROLES
        physical = self.root / f"{stage}_{process_role}"
        rows: dict[str, object] = {}
        paths: dict[str, str] = {}
        for role in roles:
            relative = f"outputs/{role}.bin"
            record = _record(
                physical / "outputs" / f"{role}.bin",
                f"{stage}:{process_role}:{role}".encode("utf-8"),
                "output",
            )
            rows[role] = {"canonical_path": f"$RUN_OUT/{relative}", "record": record}
            paths[role] = relative
        return rows, paths

    def _claim(self, stage: str, process_role: str, pid: int) -> dict[str, object]:
        core = {
            "phase": validator.S1_PHASE if stage == "s1" else validator.S2_PHASE,
            "process_role": process_role,
            "contract_sha256": self.records[f"v2_{stage}_contract"]["sha256"],
            "authorization_sha256": self.records[f"v2_{stage}_authorization"]["sha256"],
            "process_id": pid,
            "process_start_identity": f"synthetic-process-start-{pid}",
            "physical_output_root": str((self.root / f"{stage}_{process_role}").resolve()),
            "started_utc": T[f"{stage}_{process_role}_start"],
            "finished_utc": T[f"{stage}_{process_role}_finish"],
        }
        return {**core, "claim_id": validator._sha(core)}

    def _environment(self, stage: str) -> dict[str, object]:
        return {
            "python": "3.11.synthetic",
            "numpy": "2.synthetic",
            "torch": "2.synthetic",
            "onnx": "1.synthetic",
            "onnxruntime": "1.synthetic",
            "platform": "synthetic-tmp",
            "dependency_lock_sha256": "d" * 64,
            "deterministic": True,
            "cuda_used": stage == "s1",
        }

    def _process_record(
        self,
        stage: str,
        process_role: str,
        pid: int,
        evidence: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        outputs, paths = self._output_records(stage, process_role)
        claim = self._claim(stage, process_role, pid)
        events = validator._expected_ledger(stage, process_role, self.records, paths)
        ledger = {
            "process_role": process_role,
            "claim_id": claim["claim_id"],
            "events": events,
            "event_count": len(events),
            "events_sha256": validator._sha(events),
            "blocked_events": 0,
            "extra_reads": 0,
            "extra_writes": 0,
            "cache_or_lineage_reopened": 0,
        }
        environment = self._environment(stage)
        identity = {
            "phase": validator.S1_PHASE if stage == "s1" else validator.S2_PHASE,
            "contract_sha256": self.records[f"v2_{stage}_contract"]["sha256"],
            "authorization_sha256": self.records[f"v2_{stage}_authorization"]["sha256"],
            "claim_id": claim["claim_id"],
            "ledger_sha256": ledger["events_sha256"],
            "environment_sha256": validator._sha(environment),
            "outputs": {
                role: {
                    "canonical_path": outputs[role]["canonical_path"],
                    "bytes": outputs[role]["record"]["bytes"],
                    "sha256": outputs[role]["record"]["sha256"],
                }
                for role in sorted(outputs)
            },
            "evidence_sha256": validator._sha(evidence),
        }
        payload = {
            "phase": identity["phase"],
            "contract_sha256": identity["contract_sha256"],
            "authorization_sha256": identity["authorization_sha256"],
            "claim": claim,
            "access_ledger": ledger,
            "environment": environment,
            "outputs": outputs,
            "evidence": evidence,
            "scientific_root_sha256": validator._sha(identity),
        }
        document = _doc(
            f"v2_{stage}_{process_role}_record",
            payload,
            f"{stage}_{process_role}_finalized",
        )
        record = _json_record(
            self.root / "members" / stage / f"{process_role}.json",
            document,
        )
        return record, payload

    def _s1_forward(self) -> dict[str, object]:
        rows: dict[str, object] = {}
        for batch in (1, 2, 32):
            image = np.arange(batch * 3 * 4 * 4, dtype=np.float32).reshape(batch, 3, 4, 4) / 100.0
            scores = np.tile(np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), (batch, 1))
            rows[f"batch_{batch}"] = {
                "image": _typed(image),
                "valid64": _typed(np.ones((batch, 1, 4, 4), dtype=np.uint8)),
                "valid32": _typed(np.ones((batch, 1, 2, 2), dtype=np.uint8)),
                "valid16": _typed(np.ones((batch, 1, 1, 1), dtype=np.uint8)),
                "scores": _typed(scores),
            }
        return rows

    def _s1_engineering(self, plan: dict[str, object]) -> dict[str, object]:
        forward = self._s1_forward()
        outputs = {
            f"batch_{batch}": {
                "scores": copy.deepcopy(forward[f"batch_{batch}"]["scores"]),
                "max_abs": 0.0,
                "argmax_equal": True,
            }
            for batch in (1, 2, 32)
        }
        native_hash = validator._sha(
            {
                f"batch_{batch}": forward[f"batch_{batch}"]["scores"]["array_sha256"]
                for batch in (1, 2, 32)
            }
        )
        gates = {key: True for key in validator.S1_GATE_REGISTRY}
        return {
            "forward_cases": forward,
            "gradient": {
                "parameter_name": "common.weight",
                "values": _typed(np.asarray([1.0], dtype=np.float32)),
            },
            "parameter_update": {
                "parameter_name": "common.weight",
                "before": _typed(np.asarray([0.0], dtype=np.float32)),
                "after": _typed(np.asarray([0.1], dtype=np.float32)),
            },
            "onnx": {
                "opset": 17,
                "checker_passed": True,
                "checker_error": "",
                "operators": ["Add"],
                "outputs": copy.deepcopy(outputs),
            },
            "tensorrt": {
                "plugin_layers": [],
                "unsupported_ops": [],
                "plan_sha256": plan["sha256"],
                "plan_bytes": plan["bytes"],
                "outputs": copy.deepcopy(outputs),
            },
            "timing": {
                "warmup_iterations": 1,
                "batch1_ms": [1.0, 1.0, 1.0],
                "batch32_ms": [2.0, 2.0, 2.0],
                "derived": {
                    "batch1_mean_ms": 1.0,
                    "batch1_p95_ms": 1.0,
                    "batch32_throughput_sps": 16000.0,
                },
            },
            "resources": {
                "cuda_samples_bytes": [10, 11],
                "rss_samples_bytes": [20, 21],
                "derived": {"peak_cuda_bytes": 11, "peak_rss_bytes": 21},
            },
            "exact_scientific_preimage": {
                "workload_sha256": validator._sha(self.s1_contract["workload"]),
                "initial_state_sha256": "e" * 64,
                "native_outputs_sha256": native_hash,
            },
            "declared_gates": gates,
        }

    def _build_chain(
        self,
        stage: str,
        primary_evidence_factory: Callable[[dict[str, object]], dict[str, object]],
        replay_evidence_factory: Callable[[dict[str, object], dict[str, object]], dict[str, object]],
        comparison_gates: dict[str, object],
        primary_pid: int,
        replay_pid: int,
    ) -> None:
        primary_outputs, _ = self._output_records(stage, "primary")
        primary_evidence = primary_evidence_factory(primary_outputs)
        primary_record, primary_payload = self._process_record(
            stage, "primary", primary_pid, primary_evidence
        )
        ordered = [
            {
                "role": role,
                "canonical_path": primary_payload["outputs"][role]["canonical_path"],
                "record": primary_payload["outputs"][role]["record"],
            }
            for role in sorted(primary_payload["outputs"])
        ]
        handoff_core = {
            "phase": primary_payload["phase"],
            "contract_sha256": self.records[f"v2_{stage}_contract"]["sha256"],
            "authorization_sha256": self.records[f"v2_{stage}_authorization"]["sha256"],
            "primary_record": primary_record,
            "primary_scientific_root_sha256": primary_payload["scientific_root_sha256"],
            "ordered_primary_output_records": ordered,
            "finalized_utc": T[f"{stage}_handoff"],
        }
        handoff_payload = {**handoff_core, "handoff_root_sha256": validator._sha(handoff_core)}
        handoff_record = _json_record(
            self.root / "members" / stage / "primary_handoff.json",
            _doc(
                f"v2_{stage}_primary_handoff_record",
                handoff_payload,
                f"{stage}_primary_only_replay_handoff_finalized",
            ),
        )
        binding = {
            "handoff_record_sha256": handoff_record["sha256"],
            "handoff_root_sha256": handoff_payload["handoff_root_sha256"],
            "primary_record_sha256": primary_record["sha256"],
        }
        replay_outputs, _ = self._output_records(stage, "replay")
        replay_evidence = replay_evidence_factory(replay_outputs, binding)
        replay_record, replay_payload = self._process_record(
            stage, "replay", replay_pid, replay_evidence
        )
        comparison_core = {
            "phase": primary_payload["phase"],
            "contract_sha256": self.records[f"v2_{stage}_contract"]["sha256"],
            "authorization_sha256": self.records[f"v2_{stage}_authorization"]["sha256"],
            "primary_record_sha256": primary_record["sha256"],
            "handoff_record_sha256": handoff_record["sha256"],
            "replay_record_sha256": replay_record["sha256"],
            "primary_scientific_root_sha256": primary_payload["scientific_root_sha256"],
            "handoff_root_sha256": handoff_payload["handoff_root_sha256"],
            "replay_scientific_root_sha256": replay_payload["scientific_root_sha256"],
            "gate_decisions": comparison_gates,
            "finalized_utc": T[f"{stage}_comparison"],
        }
        comparison_payload = {
            **comparison_core,
            "comparison_root_sha256": validator._sha(comparison_core),
        }
        comparison_record = _json_record(
            self.root / "members" / stage / "comparison.json",
            _doc(
                f"v2_{stage}_comparison_record",
                comparison_payload,
                f"{stage}_primary_replay_comparison_finalized",
            ),
        )
        members = {
            "primary": primary_record,
            "primary_handoff": handoff_record,
            "replay": replay_record,
            "comparison": comparison_record,
        }
        inner = {
            "primary": primary_payload["scientific_root_sha256"],
            "handoff": handoff_payload["handoff_root_sha256"],
            "replay": replay_payload["scientific_root_sha256"],
            "comparison": comparison_payload["comparison_root_sha256"],
        }
        outer_core = {
            "phase": primary_payload["phase"],
            "contract_sha256": self.records[f"v2_{stage}_contract"]["sha256"],
            "authorization_sha256": self.records[f"v2_{stage}_authorization"]["sha256"],
            "member_records": members,
            "inner_roots": inner,
        }
        bundle_payload = {
            "phase": primary_payload["phase"],
            "contract_sha256": outer_core["contract_sha256"],
            "authorization_sha256": outer_core["authorization_sha256"],
            "primary_record": primary_record,
            "primary_handoff_record": handoff_record,
            "replay_record": replay_record,
            "comparison_record": comparison_record,
            "primary_scientific_root_sha256": inner["primary"],
            "handoff_root_sha256": inner["handoff"],
            "replay_scientific_root_sha256": inner["replay"],
            "comparison_root_sha256": inner["comparison"],
            "combined_outer_root_sha256": validator._sha(outer_core),
            "finalized_utc": T[f"{stage}_bundle"],
            "downstream_authority": _zero_authority(),
        }
        self._write_role(
            f"v2_{stage}_bundle_manifest",
            _doc(f"v2_{stage}_bundle_manifest", bundle_payload),
        )
        root_payload = {
            "phase": primary_payload["phase"],
            "bundle_record": self.records[f"v2_{stage}_bundle_manifest"],
            "member_records": members,
            "inner_roots": inner,
            "combined_outer_root_sha256": bundle_payload["combined_outer_root_sha256"],
            "finalized_utc": T[f"{stage}_root"],
            "downstream_authority": _zero_authority(),
        }
        self._write_role(
            f"v2_{stage}_finalization_root_record",
            _doc(f"v2_{stage}_finalization_root_record", root_payload),
        )
        self.stage_payloads = getattr(self, "stage_payloads", {})
        self.stage_payloads[stage] = {
            "primary": primary_payload,
            "replay": replay_payload,
            "handoff": handoff_payload,
            "comparison": comparison_payload,
            "bundle": bundle_payload,
            "root": root_payload,
        }

    def _build_s1(self) -> None:
        self.s1_contract = {
            "phase": validator.S1_PHASE,
            "source_revision": {
                "commit": H40_SOURCE_S1,
                "pushed": True,
                "committed_utc": T["s1_source"],
            },
            "canonical_output_root": "$RUN_OUT",
            "boundary_registry_sha256": self.records["v2_prelock_boundary_registry"]["sha256"],
            "parameter_contracts": self.parameter_contracts,
            "resource_limits": {
                "wall_seconds": 100,
                "peak_rss_bytes": 1000,
                "peak_cuda_bytes": 1000,
                "retained_bytes": 1000,
            },
            "allowed_input_roles": list(validator.S1_READ_ROLES),
            "workload": {
                "dtype": "float32",
                "spatial_shape": list(validator.S1_SPATIAL_SHAPE),
                "batches": [1, 2, 32],
                "input_seed": validator.PRIMARY_SEED,
                "warmup_iterations": 1,
                "measured_iterations": 3,
            },
            "deployment_limits": {
                "onnx_opset": 17,
                "parity_max_abs": 1e-5,
                "batch1_mean_ceiling_ms": 5.0,
                "batch1_p95_ceiling_ms": 5.0,
                "batch32_min_throughput_sps": 100.0,
                "peak_cuda_bytes": 100,
                "peak_rss_bytes": 100,
            },
            "downstream_authority": _zero_authority(),
        }
        self._write_role("v2_s1_contract", _doc("v2_s1_contract", self.s1_contract))
        authorization = self._authorization(
            "s1",
            self.records["v2_s1_contract"]["sha256"],
            self.root / "s1_primary",
            self.root / "s1_replay",
        )
        self._write_role("v2_s1_authorization", _doc("v2_s1_authorization", authorization))

        def primary_factory(outputs: dict[str, object]) -> dict[str, object]:
            engineering = self._s1_engineering(outputs["tensorrt_plan"]["record"])
            self.s1_engineering = engineering
            return {"engineering": engineering}

        def replay_factory(outputs: dict[str, object], binding: dict[str, object]) -> dict[str, object]:
            return {
                "engineering": self._s1_engineering(outputs["tensorrt_plan"]["record"]),
                "replay_binding": binding,
            }

        gates = {key: True for key in validator.S1_GATE_REGISTRY}
        comparison = {
            "primary": gates,
            "replay": gates,
            "exact_scientific_preimage_equal": True,
            "tensorrt_plan_equality_required": False,
        }
        self._build_chain("s1", primary_factory, replay_factory, comparison, 101, 102)

    def _initial_state_rows(self) -> dict[str, list[dict[str, object]]]:
        rows: dict[str, list[dict[str, object]]] = {}
        for role in validator.ROLES:
            for fold in range(validator.FOLDS):
                values = [float(fold), float(fold + 1), float(fold + 2)]
                if role == "static_matched":
                    values = [float(fold), float(fold + 3)]
                elif role == "ddf_full_repeat":
                    values = [float(fold + 100), float(fold + 101), float(fold + 102)]
                rows[f"{role}.fold_{fold}"] = [
                    _tensor_row(spec, value)
                    for spec, value in zip(self.parameter_contracts[role], values)
                ]
        return rows

    def _cohort_payload(self) -> dict[str, object]:
        return {
            "sample_indices": _typed(self.indices),
            "held_folds": _typed(self.folds),
            "targets": _typed(self.targets),
            "component_ids": self.components,
            "component_order": self.component_order,
            "sample_fold_pairs_sha256": validator.COHORT_IDENTITY["sample_fold_pairs_sha256"],
            "fold_mapping_sha256": validator.COHORT_IDENTITY["fold_mapping_sha256"],
            "component_order_sha256": validator.COHORT_IDENTITY["component_order_sha256"],
        }

    def _target_shards(self) -> dict[str, object]:
        rows: dict[str, object] = {}
        target_by_index = {int(index): int(target) for index, target in zip(self.indices, self.targets)}
        for held in range(validator.FOLDS):
            for split in ("fit", "calibration", "held"):
                key = f"fold_{held}_{split}"
                indices = validator._fold_split_indices(self.indices, self.folds, held, split)
                targets = np.asarray([target_by_index[int(index)] for index in indices], dtype=np.int64)
                rows[key] = {
                    "sample_indices": _typed(indices),
                    "targets": _typed(targets),
                    "pair_sha256": validator._sha(
                        [[int(index), int(target)] for index, target in zip(indices, targets)]
                    ),
                }
        return rows

    def _random_payload(self) -> dict[str, object]:
        orders: dict[str, object] = {}
        for role in validator.ROLES:
            for fold in range(validator.FOLDS):
                fit = validator._fold_split_indices(self.indices, self.folds, fold, "fit")
                seed = validator.PRIMARY_SEED + 100 * fold + (
                    validator.REPEAT_OFFSET if role == "ddf_full_repeat" else 0
                ) + 1
                generator = np.random.Generator(np.random.PCG64(seed))
                order = np.stack(
                    [generator.permutation(fit) for _ in range(validator.EPOCHS)]
                ).astype(np.int64, copy=False)
                orders[f"{role}.fold_{fold}"] = _typed(order)
        return {
            "noise_array_sha256": validator.NOISE_ARRAY_SHA256,
            "bootstrap_array_sha256": validator.BOOTSTRAP_ARRAY_SHA256,
            "epoch_orders": orders,
        }

    def _scientific_payload(self) -> dict[str, object]:
        cohort = self._cohort_payload()
        self.s2_cohort = cohort
        xai_indices = np.asarray(validator.XAI_SAMPLE_INDICES, dtype=np.int64)
        target_by_index = {int(index): int(target) for index, target in zip(self.indices, self.targets)}
        xai_targets = np.asarray([target_by_index[int(index)] for index in xai_indices], dtype=np.int64)
        xai = {
            "sample_indices": _typed(xai_indices),
            "targets": _typed(xai_targets),
            "strata_counts": {
                str(target): int((xai_targets == target).sum())
                for target in sorted(set(map(int, xai_targets.tolist())))
            },
            "pair_sha256": validator._sha(
                [[int(index), int(target)] for index, target in zip(xai_indices, xai_targets)]
            ),
        }
        cidt = {
            "targets": _typed(self.cidt_targets),
            "keeper_predictions": _typed(self.cidt_predictions),
            "pair_sha256": validator._sha(
                [[int(target), int(prediction)] for target, prediction in zip(self.cidt_targets, self.cidt_predictions)]
            ),
        }
        gates = {
            "cohort_identity": True,
            "target_shards_exact": True,
            "xai_exact": True,
            "cidt_exact": True,
            "pcg_noise_exact": True,
            "pcg_bootstrap_exact": True,
            "pcg_epoch_orders_exact": True,
            "initial_states_exact": True,
            "s2a_before_s2b": True,
            "no_lineage_checkpoint_read": True,
        }
        return {
            "cohort": cohort,
            "target_shards": self._target_shards(),
            "xai": xai,
            "cidt": cidt,
            "random": self._random_payload(),
            "initial_states": self.initial_states,
            "declared_gates": gates,
        }

    def _build_s2(self) -> None:
        self.initial_states = self._initial_state_rows()
        initial_hashes = {key: _state_digest(rows) for key, rows in self.initial_states.items()}
        cohort_contract = {
            "rows": validator.ROWS,
            "folds": validator.FOLDS,
            "components": validator.COMPONENTS,
            "fold_sizes": list(validator.FOLD_SIZES),
            "target_counts": validator.TARGET_COUNTS,
            "identity_sha256": validator.COHORT_IDENTITY,
            "xai_sample_indices": list(validator.XAI_SAMPLE_INDICES),
            "xai_sample_indices_array_sha256": validator.XAI_ARRAY_SHA256,
            "cidt_identity": validator.CIDT_IDENTITY,
            "target_shard_keys": list(validator.TARGET_SHARD_KEYS),
        }
        self.s2_contract = {
            "phase": validator.S2_PHASE,
            "source_revision": {
                "commit": H40_SOURCE_S2,
                "pushed": True,
                "committed_utc": T["s2_source"],
            },
            "canonical_output_root": "$RUN_OUT",
            "boundary_registry_sha256": self.records["v2_prelock_boundary_registry"]["sha256"],
            "parameter_contracts": self.parameter_contracts,
            "resource_limits": {
                "wall_seconds": 100,
                "peak_rss_bytes": 1000,
                "peak_cuda_bytes": 0,
                "retained_bytes": 1000,
            },
            "verified_s1": {
                key: self.verified_s1[key]
                for key in (
                    "contract_sha256", "authorization_sha256", "bundle_sha256",
                    "root_record_sha256", "outer_root_sha256", "finalized_utc",
                )
            },
            "s2a_contract": {
                "state": "target_blind_archive_phase",
                "member_registry": [list(item) for item in validator.S2A_MEMBER_REGISTRY],
                "score_independent": True,
                "target_blind": True,
            },
            "s2b_contract": {
                "state": "sealed_target_metadata_phase",
                "member_registry": [list(item) for item in validator.S2B_MEMBER_REGISTRY],
                "release_scope": "exact_target_shards_cidt_and_xai_only",
            },
            "target_release_order": ["S2A_FINALIZED", "S2B_RELEASED"],
            "cohort_contract": cohort_contract,
            "random_contract": {
                "generator": "numpy.random.PCG64",
                "primary_seed": validator.PRIMARY_SEED,
                "repeat_offset": validator.REPEAT_OFFSET,
                "epochs": validator.EPOCHS,
                "noise_shape": list(validator.NOISE_SHAPE),
                "noise_array_sha256": validator.NOISE_ARRAY_SHA256,
                "bootstrap_shape": list(validator.BOOTSTRAP_SHAPE),
                "bootstrap_array_sha256": validator.BOOTSTRAP_ARRAY_SHA256,
            },
            "initial_state_sha256": initial_hashes,
            "allowed_input_roles": list(validator.S2_FIXED_READ_ROLES),
            "downstream_authority": _zero_authority(),
        }
        self._write_role("v2_s2_contract", _doc("v2_s2_contract", self.s2_contract))
        authorization = self._authorization(
            "s2",
            self.records["v2_s2_contract"]["sha256"],
            self.root / "s2_primary",
            self.root / "s2_replay",
        )
        self._write_role("v2_s2_authorization", _doc("v2_s2_authorization", authorization))

        def primary_factory(_outputs: dict[str, object]) -> dict[str, object]:
            scientific = self._scientific_payload()
            self.s2_scientific = scientific
            return {"scientific": scientific}

        def replay_factory(_outputs: dict[str, object], binding: dict[str, object]) -> dict[str, object]:
            return {"scientific": self._scientific_payload(), "replay_binding": binding}

        gates = self._scientific_payload()["declared_gates"]
        comparison = {
            "primary": gates,
            "replay": gates,
            "scientific_preimage_equal": True,
        }
        self._build_chain("s2", primary_factory, replay_factory, comparison, 201, 202)


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StagedFixture:
    return StagedFixture(tmp_path, monkeypatch)


def test_staged_s1_then_s2_then_guard_receipt(staged: StagedFixture) -> None:
    s1 = validator.verify_finalized_s1_evidence(artifacts=staged.records)
    s2 = validator.verify_finalized_s2_evidence(artifacts=staged.records, verified_s1=s1)
    receipt = validator.verify_finalized_prelock_evidence(artifacts=staged.records)
    assert s1["state"] == "s1_finalized_verified_no_execution_authority"
    assert s2["state"] == "s2_finalized_verified_after_s1_no_execution_authority"
    assert receipt == {
        "schema_version": 2,
        "state": "s1_s2_finalized_verified_no_execution_authority",
        "sequencing_erratum_sha256": staged.records["v2_prelock_sequencing_erratum"]["sha256"],
        "verifier_source_sha256": staged.records["v2_prelock_validator_source"]["sha256"],
        "verifier_tests_sha256": staged.records["v2_prelock_validator_tests"]["sha256"],
        "s1_bundle_sha256": staged.records["v2_s1_bundle_manifest"]["sha256"],
        "s1_root_sha256": staged.records["v2_s1_finalization_root_record"]["sha256"],
        "s2_bundle_sha256": staged.records["v2_s2_bundle_manifest"]["sha256"],
        "s2_root_sha256": staged.records["v2_s2_finalization_root_record"]["sha256"],
        "formal_runs_authorized": 0,
        "candidate_runs_authorized": 0,
        "validation_runs_authorized": 0,
        "test_runs_authorized": 0,
        "conditional_xai_runs_authorized": 0,
        "cache_or_lineage_reopened": 0,
    }
    assert all(not Path(staged.records[role]["path"]).exists() for role in validator.NO_REOPEN_ROLES)


def test_s2_rejects_missing_or_forged_verified_s1(staged: StagedFixture) -> None:
    with pytest.raises(validator.PrelockValidationError, match="verified_s1"):
        validator.verify_finalized_s2_evidence(artifacts=staged.records, verified_s1={})


def test_boundary_rejects_extra_role_and_fixed_hash_drift(staged: StagedFixture) -> None:
    extra = {**staged.records, "unexpected": staged.records["v2_protocol_base"]}
    with pytest.raises(validator.PrelockValidationError, match="artifacts keys differ"):
        validator.verify_finalized_prelock_evidence(artifacts=extra)
    drift = copy.deepcopy(staged.records)
    path = staged.root / "drift.json"
    drift["v2_protocol_base"] = _record(path, b"drift")
    with pytest.raises(validator.PrelockValidationError, match="fixed boundary hash differs"):
        validator.verify_finalized_prelock_evidence(artifacts=drift)


def test_authorization_requires_distinct_later_pushed_commit(staged: StagedFixture) -> None:
    payload = copy.deepcopy(staged.stage_payloads["s1"])
    authorization = validator._document(
        validator._json_bytes(Path(staged.records["v2_s1_authorization"]["path"]).read_bytes(), "auth"),
        "v2_s1_authorization",
    )
    forged = dict(authorization)
    forged["authorization_commit"] = forged["source_commit"]
    forged["authorization_id"] = validator._sha({key: value for key, value in forged.items() if key != "authorization_id"})

    class Reader:
        def payload(self, _role: str) -> dict[str, object]:
            return forged

    del payload
    with pytest.raises(validator.PrelockValidationError, match="revision/order/identity"):
        validator._validate_authorization(Reader(), staged.records, "s1", staged.s1_contract)


def test_s1_rejects_arbitrary_pass_boolean(staged: StagedFixture) -> None:
    evidence = {"engineering": copy.deepcopy(staged.s1_engineering)}
    evidence["engineering"]["automatic_passed"] = True
    outputs = {
        role: item["record"]
        for role, item in staged.stage_payloads["s1"]["primary"]["outputs"].items()
    }
    with pytest.raises(validator.PrelockValidationError, match="keys differ"):
        validator._validate_s1_run(
            evidence,
            "forged S1",
            contract=staged.s1_contract,
            outputs=outputs,
            replay=False,
        )


def test_s2a_target_alias_and_extra_ledger_access_fail_closed(staged: StagedFixture) -> None:
    with pytest.raises(validator.PrelockValidationError, match="target/XAI alias"):
        validator._forbid_aliases({"nested": {"labels": [1]}}, "S2A")
    primary = staged.stage_payloads["s2"]["primary"]
    output_paths = {
        role: item["canonical_path"].removeprefix("$RUN_OUT/")
        for role, item in primary["outputs"].items()
    }
    extras = (
        {
            "operation": "read_artifact",
            "stage": "S2",
            "role": "keeper_checkpoint_lineage_only",
            "member": None,
            "path": staged.records["keeper_checkpoint_lineage_only"]["path"],
            "mode": "rb",
        },
        {
            "operation": "write_output",
            "stage": "S2",
            "role": "undeclared_output",
            "member": None,
            "path": "$RUN_OUT/undeclared.bin",
            "mode": "create_exclusive",
        },
    )
    for extra in extras:
        ledger = copy.deepcopy(primary["access_ledger"])
        ledger["events"].append({"sequence": len(ledger["events"]), **extra})
        with pytest.raises(validator.PrelockValidationError, match="exact-set ledger differs"):
            validator._validate_ledger(
                ledger,
                "forged S2 ledger",
                stage="s2",
                process_role="primary",
                artifacts=staged.records,
                output_paths=output_paths,
                claim_id=primary["claim"]["claim_id"],
            )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("fold", "fold/target counts differ"),
        ("target", "locked array identity differs"),
        ("component", "component crosses held folds"),
    ],
)
def test_exact_cohort_fold_target_component_preimages(
    staged: StagedFixture, mutation: str, match: str
) -> None:
    cohort = copy.deepcopy(staged.s2_cohort)
    if mutation == "fold":
        folds = np.asarray([0, 1, 1, 1, 2, 2], dtype=np.int64)
        cohort["held_folds"] = _typed(folds)
        staged.monkeypatch.setitem(
            validator.COHORT_IDENTITY, "held_folds_array_sha256", validator._array_sha(folds)
        )
    elif mutation == "target":
        targets = staged.targets.copy()
        targets[0] = 1
        cohort["targets"] = _typed(targets)
    else:
        cohort["component_ids"][2] = cohort["component_ids"][0]
    with pytest.raises(validator.PrelockValidationError, match=match):
        validator._validate_cohort(cohort, "forged cohort")


def test_exact_xai_and_cidt_typed_preimages(staged: StagedFixture) -> None:
    scientific = staged.s2_scientific
    cohort = validator._validate_cohort(scientific["cohort"], "cohort")
    xai = copy.deepcopy(scientific["xai"])
    xai["targets"] = _typed(np.asarray([1, 1, 4], dtype=np.int64))
    with pytest.raises(validator.PrelockValidationError, match="XAI identity/target strata"):
        validator._validate_xai(xai, "forged XAI", cohort)
    cidt = copy.deepcopy(scientific["cidt"])
    cidt["keeper_predictions"]["data_hex"] = _typed(np.zeros(6, dtype=np.int64))["data_hex"]
    with pytest.raises(validator.PrelockValidationError, match="array SHA-256 differs"):
        validator._validate_cidt(cidt, "forged CIDT")


def test_pcg_epoch_order_and_parameter_schema_initial_state_are_recomputed(
    staged: StagedFixture,
) -> None:
    scientific = staged.s2_scientific
    cohort = validator._validate_cohort(scientific["cohort"], "cohort")
    random = copy.deepcopy(scientific["random"])
    key = "ddf_full.fold_0"
    order = np.frombuffer(bytes.fromhex(random["epoch_orders"][key]["data_hex"]), dtype=np.int64).copy()
    order[0], order[1] = order[1], order[0]
    random["epoch_orders"][key] = _typed(order.reshape(random["epoch_orders"][key]["shape"]))
    with pytest.raises(validator.PrelockValidationError, match="PCG epoch order differs"):
        validator._validate_random(random, "forged random", cohort)
    contracts = copy.deepcopy(staged.parameter_contracts)
    contracts["ddf_channel_only"][1]["requires_grad"] = True
    with pytest.raises(validator.PrelockValidationError, match="trainability differs"):
        validator._parameter_contracts(contracts, "forged parameter contracts")
    states = copy.deepcopy(staged.initial_states)
    states["ddf_spatial_only.fold_0"][0]["data_hex"] = _typed(np.asarray([99.0], dtype=np.float32))["data_hex"]
    with pytest.raises(validator.PrelockValidationError, match="array SHA-256 differs"):
        validator._validate_initial_states(
            states, "forged initial states", contract_payload=staged.s2_contract
        )


def test_temporal_handoff_and_final_root_order_is_strict(staged: StagedFixture) -> None:
    del staged
    with pytest.raises(validator.PrelockValidationError, match="temporal finalization chain differs"):
        validator._validate_temporal_chain(
            stage="s1",
            authorization_committed_utc=T["s1_auth"],
            primary_started_utc=T["s1_primary_start"],
            primary_finished_utc=T["s1_primary_finish"],
            handoff_finalized_utc=T["s1_replay_start"],
            replay_started_utc=T["s1_handoff"],
            replay_finished_utc=T["s1_replay_finish"],
            comparison_finalized_utc=T["s1_comparison"],
            bundle_finalized_utc=T["s1_bundle"],
            root_finalized_utc=T["s1_root"],
        )


def test_cross_phase_process_and_root_reuse_is_rejected(staged: StagedFixture) -> None:
    s1 = {
        "primary": {"claim": staged.stage_payloads["s1"]["primary"]["claim"], "scientific_root_sha256": "a" * 64},
        "replay": {"claim": staged.stage_payloads["s1"]["replay"]["claim"], "scientific_root_sha256": "b" * 64},
        "outer_root_sha256": "c" * 64,
    }
    reused = copy.deepcopy(s1)
    with pytest.raises(validator.PrelockValidationError, match="cross-phase process_id reused"):
        validator._validate_cross_phase(s1, reused)


def test_run_out_rejects_parent_absolute_duplicate_and_physical_alias(tmp_path: Path) -> None:
    for value in ("$RUN_OUT/../escape", str((tmp_path / "absolute").resolve())):
        with pytest.raises(validator.PrelockValidationError, match="RUN_OUT"):
            validator._relative_run_out(value, "output")
    first = _record(tmp_path / "first.bin", b"same")
    second_path = tmp_path / "second.bin"
    os.link(first["path"], second_path)
    second = {
        "path": str(second_path.resolve()),
        "bytes": first["bytes"],
        "sha256": first["sha256"],
        "access_class": "finalized_evidence",
    }
    reader = validator._EvidenceReader({})
    with pytest.raises(validator.PrelockValidationError, match="single-link regular file"):
        reader.raw_record(first, "first")
    assert second["path"] != first["path"]
    physical = tmp_path / "outputs"
    rows = {
        role: {
            "canonical_path": f"$RUN_OUT/{role}.bin",
            "record": _record(physical / f"{role}.bin", role.encode("utf-8"), "output"),
        }
        for role in validator.S1_OUTPUT_ROLES
    }
    rows[validator.S1_OUTPUT_ROLES[1]]["canonical_path"] = rows[validator.S1_OUTPUT_ROLES[0]]["canonical_path"]
    with pytest.raises(validator.PrelockValidationError, match="duplicate canonical paths"):
        validator._validate_output_records(
            validator._EvidenceReader({}),
            rows,
            "duplicate outputs",
            stage="s1",
            physical_root=str(physical.resolve()),
        )


def test_no_reopen_delegation_requires_guard_bound_unique_single_link_identities(
    staged: StagedFixture,
) -> None:
    document = validator._json_bytes(
        Path(staged.records["v2_guard_identity_evidence"]["path"]).read_bytes(),
        "guard evidence",
    )
    payload = copy.deepcopy(document["payload"])
    roles = sorted(validator.NO_REOPEN_ROLES)
    payload["physical_identities"][roles[1]]["file_id"] = payload["physical_identities"][roles[0]]["file_id"]
    forged_record = _json_record(
        staged.root / "forged_guard_identity.json",
        _doc("v2_guard_identity_evidence", payload),
    )
    records = {**staged.records, "v2_guard_identity_evidence": forged_record}
    reader = validator._EvidenceReader(records)
    with pytest.raises(validator.PrelockValidationError, match="share a physical identity"):
        validator._validate_guard_identity(reader, records)
