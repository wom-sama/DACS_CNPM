"""Developmental pure-schema tests; these never attest production pre-lock PASS."""

from __future__ import annotations

import copy
import hashlib
import math

import pytest

from trkh.tools import pair_surface_ddf_v2_prelock_schema_v3 as schema


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CLAIM = "d" * 64
PROCESS = "1" * 64


def _artifact(role: str, *, array: bool = False, dtype: str = "int64",
              shape: list[int] | None = None, logical_path: str | None = None,
              media_type: str | None = None,
              access_class: str = "retained_evidence") -> dict[str, object]:
    shape = shape or [1]
    itemsize = {"bool": 1, "uint8": 1, "int64": 8, "float32": 4,
                "float64": 8, "<U64": 256}[dtype]
    array_bytes = math.prod(shape) * itemsize
    row: dict[str, object] = {
        "role": role,
        "logical_path": logical_path or (f"$RUN_OUT/{role}.npy" if array else f"$RUN_OUT/{role}.json"),
        "media_type": media_type or ("application/x-npy" if array else "application/json"),
        "access_class": access_class,
        "bytes": array_bytes + 128 if array else 101,
        "sha256": SHA_A,
    }
    if array:
        row.update(dtype=dtype, shape=shape, array_bytes=array_bytes, array_sha256=SHA_B)
    return row


def _input_artifact(role: str, suffix: str = ".json",
                    access_class: str = "committed_pushed_local_import_identity") -> dict[str, object]:
    media = "text/x-python" if suffix == ".py" else "text/markdown" if suffix == ".md" else "application/json"
    return _artifact(
        role, logical_path=f"D:\\repo\\{role}{suffix}", media_type=media,
        access_class=access_class,
    )


def _outer(kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "protocol_id": schema.PROTOCOL_ID,
        "kind": kind,
        "state": schema.STAGE_ROLE_STATES[kind],
        "payload": payload,
    }


def _group(role: str) -> dict[str, object]:
    records: dict[str, object] = {}
    for field in schema.GROUP_SCHEMAS[role].fields:
        if field.kind == "artifact":
            suffix = ".json" if field.media_type == "application/json" else ".bin"
            records[field.name] = _artifact(
                field.role,
                logical_path=f"$RUN_OUT/{role}/{field.name}{suffix}",
                media_type=field.media_type,
                access_class=field.access_class,
            )
        else:
            records[field.name] = _artifact(
                field.role, array=True, dtype=str(field.dtype), shape=list(field.shape),
                logical_path=f"$RUN_OUT/{role}/{field.name}.npy",
                access_class=field.access_class,
            )
            if role == "s2b_cidt_projection" and field.name in schema.EXACT_CIDT_ARRAY_SHA256:
                records[field.name]["array_sha256"] = schema.EXACT_CIDT_ARRAY_SHA256[field.name]
            if role == "s2a_target_free_source_snapshot" and field.name in {"sample_indices", "held_folds"}:
                records[field.name]["array_sha256"] = schema.EXACT_A0_ARRAY_SHA256[field.name]
            if role == "s2b_target_source_snapshot" and field.name in {"sample_indices", "targets"}:
                records[field.name]["array_sha256"] = schema.EXACT_A0_ARRAY_SHA256[field.name]
    return {"evidence_schema_version": 3, "group_role": role, "records": records}


def _bind_registry(group_payload: dict[str, object], field: str,
                   payload: bytes) -> dict[str, object]:
    bound = copy.deepcopy(group_payload)
    record = bound["records"][field]
    record["bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    return bound


def _lineage(stage: str = "s1") -> dict[str, object]:
    schema_source = _input_artifact("v2_prelock_evidence_schema_v3_source", ".py")
    row = {
        "foundation_commit": schema.FOUNDATION_COMMIT,
        "boundary_review_commit": schema.BOUNDARY_REVIEW_COMMIT,
        "source_commit": "2" * 40,
        "branch": "classification-only-research",
        "upstream_ref": "origin/classification-only-research",
        "upstream_commit": "2" * 40,
        "repository_tree_sha256": SHA_C,
        "launcher": _input_artifact("v2_execution_launcher", ".py"),
        "v2_prelock_evidence_schema_v3": _input_artifact(
            "v2_prelock_evidence_schema_v3", ".md",
            "committed_pushed_boundary_document_identity",
        ),
        "v2_prelock_evidence_schema_v3_source": schema_source,
        "full_local_import_closure": {
            "trkh": _input_artifact("trkh_package", ".py"),
            "trkh.tools.pair_surface_ddf_v2_prelock_schema_v3": schema_source,
        },
        "package_files": {"trkh.__init__": _input_artifact(
            "trkh_package_init", ".py", "pinned_package_file_identity"
        )},
        "fixed_protocol_and_incident_hashes": dict(schema.FIXED_PROTOCOL_AND_INCIDENT_SHA256),
        "expanded_test_node_registry_sha256": SHA_B,
        "rejected_lineage": [{
            "sha256": schema.INVALIDATED_MANIFEST_SHA256,
            "reason": schema.INVALIDATION_REASON,
            "scientific_effect": 0,
            "authority_effect": 0,
        }],
    }
    if stage == "s2":
        row["verified_s1_receipt"] = _input_artifact(
            "v2_s1_finalization_root_record", ".json",
            "committed_pushed_stage_receipt_identity",
        )
    return row


def _finalization(stage: str) -> dict[str, object]:
    fields = (schema.S1_FINALIZATION_PAYLOAD_FIELDS if stage == "s1"
              else schema.S2_FINALIZATION_PAYLOAD_FIELDS)
    row: dict[str, object] = {}
    for field in fields:
        if field == "evidence_schema_version":
            row[field] = 3
        elif field == "foundation_commit":
            row[field] = schema.FOUNDATION_COMMIT
        elif field == "boundary_review_commit":
            row[field] = schema.BOUNDARY_REVIEW_COMMIT
        elif field == "source_commit":
            row[field] = "2" * 40
        elif field in {"fresh_process_replay", "s2_contract_may_be_created"}:
            row[field] = True
        elif field == "downstream_authority":
            row[field] = {name: 0 for name in schema.DOWNSTREAM_AUTHORITY_FIELDS}
        elif field == "cache_or_lineage_reopened":
            row[field] = 0
        else:
            row[field] = SHA_C
    return row


def test_group_role_counts_and_order_are_exact_developmental_only() -> None:
    assert len(schema.S1_GROUP_ROLES) == 12
    assert len(schema.S2A_GROUP_ROLES) == 10
    assert len(schema.S2B_GROUP_ROLES) == 5
    assert tuple(schema.GROUP_SCHEMAS) == (
        schema.S1_GROUP_ROLES + schema.S2A_GROUP_ROLES + schema.S2B_GROUP_ROLES
    )
    assert "targets" not in {
        field.name for role in schema.S2A_GROUP_ROLES
        for field in schema.GROUP_SCHEMAS[role].fields
    }
    role = "s1_synthetic_inputs"
    payload = _group(role)
    payload["records"]["synthetic_inputs"]["shape"] = [3, 3, 64, 64]
    with pytest.raises(schema.PrelockSchemaV3Error, match="dtype/shape"):
        schema.parse_group_payload(payload, role)
    assert schema.GROUP_SCHEMAS["s2a_nonwrap_offset_preimages"].fields[0].shape == (
        schema.ROWS, 2, 2
    )
    assert schema.GROUP_SCHEMAS["s2b_cidt_projection"].fields[0].shape == (
        schema.CIDT_ROWS,
    )


def test_outer_wrapper_is_canonical_no_newline_and_exact() -> None:
    row = _outer("v2_s1_finalization_root_record", _finalization("s1"))
    payload = schema.serialize_outer_document(row, "v2_s1_finalization_root_record")
    assert not payload.endswith(b"\n")
    parsed = schema.parse_outer_bytes(payload, "v2_s1_finalization_root_record")
    assert parsed["kind"] == "v2_s1_finalization_root_record"
    assert schema.serialize_outer_document(
        parsed, "v2_s1_finalization_root_record"
    ) == payload
    with pytest.raises(TypeError):
        parsed["payload"]["downstream_authority"]["formal_runs_authorized"] = 0
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.parse_outer_bytes(payload + b"\n", "v2_s1_finalization_root_record")
    with pytest.raises(schema.PrelockSchemaV3Error, match="keys must be strings"):
        schema.canonical_json_bytes({1: "aliased"})
    changed = copy.deepcopy(row)
    changed["extra"] = 1
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.serialize_outer_document(changed)
    changed = copy.deepcopy(row)
    changed["state"] = "passed"
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.serialize_outer_document(changed)
    changed = copy.deepcopy(row)
    changed["kind"] = []
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.serialize_outer_document(changed)
    changed = copy.deepcopy(row)
    changed["schema_version"] = 2.0
    with pytest.raises(schema.PrelockSchemaV3Error, match="exact integer"):
        schema.serialize_outer_document(changed)
    changed = copy.deepcopy(row)
    changed["payload"]["evidence_schema_version"] = 3.0
    with pytest.raises(schema.PrelockSchemaV3Error, match="exact integer"):
        schema.serialize_outer_document(changed)
    with pytest.raises(schema.PrelockSchemaV3Error, match="not frozen"):
        schema.serialize_outer_document(
            _outer("v2_s1_contract", {"evidence_schema_version": 3})
        )


def test_artifact_array_and_write_records_reject_aliases_and_extra_keys() -> None:
    artifact = schema.ArtifactRecordV3.parse(_artifact("raw_input"))
    assert artifact.role == "raw_input"
    array = schema.ArrayRecordV3.parse(_artifact("indices", array=True, shape=[763]))
    assert array.shape == (763,)
    bad = _artifact("raw_input")
    bad["logical_path"] = "$RUN_OUT/../escape.json"
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.ArtifactRecordV3.parse(bad)
    bad = _artifact("indices", array=True)
    bad["dtype"] = ">i8"
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.ArrayRecordV3.parse(bad)
    bad = _artifact("indices", array=True)
    bad["array_bytes"] = 7
    with pytest.raises(schema.PrelockSchemaV3Error, match="dtype/shape"):
        schema.ArrayRecordV3.parse(bad)
    bad = _artifact("indices", array=True)
    bad["bytes"] = bad["array_bytes"]
    with pytest.raises(schema.PrelockSchemaV3Error, match="NPY header"):
        schema.ArrayRecordV3.parse(bad)
    bad = _artifact("invalidated")
    bad["sha256"] = schema.INVALIDATED_MANIFEST_SHA256
    with pytest.raises(schema.PrelockSchemaV3Error, match="invalidated"):
        schema.ArtifactRecordV3.parse(bad)
    receipt = {
        "sequence": 0, "role": "result", "logical_path": "$RUN_OUT/result.json",
        "bytes": 11, "sha256": SHA_A, "creator_claim_sha256": CLAIM,
        "process_identity_sha256": PROCESS, "physical_identity": "volume:file:1",
        "media_type": "application/json",
    }
    assert schema.WriteReceiptV3.parse(receipt).sequence == 0
    receipt["unbound"] = True
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.WriteReceiptV3.parse(receipt)


def test_archive_member_event_requires_one_read_snapshot_and_target_transition() -> None:
    event = {
        "event_index": 0, "archive_role": "target_archive", "archive_expected_bytes": 1000,
        "archive_expected_sha256": SHA_A, "member_name": "fold0/targets.npy",
        "archive_directory_index": 2, "compression_method": 8, "compressed_bytes": 400,
        "uncompressed_bytes": 928, "uncompressed_sha256": SHA_B, "dtype": "int64",
        "shape": [100], "array_sha256": SHA_C, "stage": "s2b", "outer_fold": 0,
        "target_bearing": True, "transition_sha256": CLAIM, "read_count": 1,
        "snapshot_artifact_role": "fold0_targets_snapshot", "snapshot_array_sha256": SHA_C,
    }
    assert schema.ArchiveMemberEventV3.parse(event).target_bearing is True
    for field, value in (("transition_sha256", None), ("read_count", 2),
                         ("snapshot_array_sha256", SHA_A)):
        bad = dict(event)
        bad[field] = value
        with pytest.raises(schema.PrelockSchemaV3Error):
            schema.ArchiveMemberEventV3.parse(bad)
    blind = dict(event, stage="s2a", target_bearing=False, transition_sha256=None,
                 member_name="source/sample_indices.npy")
    assert schema.ArchiveMemberEventV3.parse(blind).transition_sha256 is None
    leaked = dict(event, stage="s2a")
    with pytest.raises(schema.PrelockSchemaV3Error, match="S2A"):
        schema.ArchiveMemberEventV3.parse(leaked)


def test_all_fixed_shape_groups_parse_and_reject_field_or_role_drift() -> None:
    for role in schema.S1_GROUP_ROLES + schema.S2A_GROUP_ROLES + schema.S2B_GROUP_ROLES:
        payload = _group(role)
        parsed = schema.parse_group_payload(payload, role)
        assert parsed.group_role == role
        first = schema.GROUP_SCHEMAS[role].fields[0].name
        changed = copy.deepcopy(payload)
        changed["records"][first]["role"] = "drifted_role"
        with pytest.raises(schema.PrelockSchemaV3Error):
            schema.parse_group_payload(changed, role)
        changed = copy.deepcopy(payload)
        changed["records"]["extra"] = _artifact("extra")
        with pytest.raises(schema.PrelockSchemaV3Error):
            schema.parse_group_payload(changed, role)


def test_registry_inner_rows_are_exact_ordered_and_target_blind() -> None:
    group = "s1_synthetic_inputs"
    field = "synthetic_generation_spec"
    spec = schema.REGISTRY_SPECS[field]
    role = next(item.role for item in schema.GROUP_SCHEMAS[group].fields if item.name == field)
    registry = {
        "evidence_schema_version": 3, "schema_id": role, "columns": list(spec.columns),
        "order_by": list(spec.order_by),
        "count_rule": dict(schema.registry_count_rule_payload(field)),
        "row_count": 3,
        "rows": [
            {"case_index": index, "batch_size": schema.S1_BATCH,
             "seed": 20260729 + index, "generator": "PCG64",
             "formula_sha256": SHA_B}
            for index in range(3)
        ],
    }
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    assert len(schema.parse_registry_bytes(payload, group, field, group_payload)) == 3
    unbound = copy.deepcopy(group_payload)
    unbound["records"][field]["sha256"] = SHA_A
    with pytest.raises(schema.PrelockSchemaV3Error, match="typed group artifact"):
        schema.parse_registry_bytes(payload, group, field, unbound)

    group = "s2a_target_free_source_snapshot"
    field = "source_archive_member_registry"
    spec = schema.REGISTRY_SPECS[field]
    role = next(item.role for item in schema.GROUP_SCHEMAS[group].fields if item.name == field)
    row = {
        "member_index": 0, "archive_role": "target_free_source", "member_name": "sample.npy",
        "container_bytes": 100, "container_sha256": SHA_A, "member_bytes": 80,
        "member_sha256": SHA_B, "dtype": "int64", "shape": [10],
        "target_bearing": False,
    }
    registry = {
        "evidence_schema_version": 3, "schema_id": role, "columns": list(spec.columns),
        "order_by": list(spec.order_by),
        "count_rule": dict(schema.registry_count_rule_payload(field)),
        "row_count": 1, "rows": [row],
    }
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="stage-specific verifier"):
        schema.parse_registry_bytes(payload, group, field, group_payload)
    registry["rows"][0]["dtype"] = ">i8"
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="frozen NumPy dtype"):
        schema.parse_registry_bytes(payload, group, field, group_payload)
    registry["rows"][0]["dtype"] = "int64"
    registry["rows"][0]["target_bearing"] = True
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="required value"):
        schema.parse_registry_bytes(payload, group, field, group_payload)

    group, field = "s1_ort_reference_outputs", "ort_output_registry"
    spec = schema.REGISTRY_SPECS[field]
    field_spec = next(item for item in schema.GROUP_SCHEMAS[group].fields if item.name == field)
    array = _artifact(
        f"{field_spec.role}__row_000000__array", array=True, dtype="float32", shape=[1, 4],
        logical_path=f"$RUN_OUT/{group}/{field}/row_000000/array.npy",
        access_class=field_spec.access_class,
    )
    registry = {"evidence_schema_version": 3, "schema_id": field_spec.role,
                "columns": list(spec.columns), "order_by": list(spec.order_by),
                "count_rule": dict(schema.registry_count_rule_payload(field)), "row_count": 1,
                "rows": [{"batch_size": 1, "tensor_name": "union", "array": array}]}
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="stage-specific verifier"):
        schema.parse_registry_bytes(payload, group, field, group_payload)
    registry["rows"][0]["array"]["role"] = "aliased"
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="nested artifact"):
        schema.parse_registry_bytes(payload, group, field, group_payload)

    group, field = "s2b_cidt_projection", "cidt_projection_registry"
    spec = schema.REGISTRY_SPECS[field]
    field_spec = next(item for item in schema.GROUP_SCHEMAS[group].fields if item.name == field)
    group_payload = _group(group)
    names = (
        ("sample_indices", "cidt_sample_indices"),
        ("targets", "cidt_targets"),
        ("baseline_predictions", "cidt_baseline_predictions"),
    )
    registry = {
        "evidence_schema_version": 3, "schema_id": field_spec.role,
        "columns": list(spec.columns), "order_by": list(spec.order_by),
        "count_rule": dict(schema.registry_count_rule_payload(field)), "row_count": 3,
        "rows": [
            {"field_index": index, "name": name, "top_level_field": top_name,
             "array": copy.deepcopy(group_payload["records"][top_name]),
             "expected_array_sha256": schema.EXACT_CIDT_ARRAY_SHA256[top_name]}
            for index, (name, top_name) in enumerate(names)
        ],
    }
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(group_payload, field, payload)
    assert len(schema.parse_registry_bytes(payload, group, field, group_payload)) == 3
    registry["rows"][0]["expected_array_sha256"] = SHA_A
    payload = schema.canonical_json_bytes(registry)
    group_payload = _bind_registry(_group(group), field, payload)
    with pytest.raises(schema.PrelockSchemaV3Error, match="CIDT"):
        schema.parse_registry_bytes(payload, group, field, group_payload)


def test_common_lineage_requires_full_closure_invalidation_and_acyclic_bindings() -> None:
    lineage = _lineage()
    assert schema.parse_common_contract_lineage(lineage, "s1")["source_commit"] == "2" * 40
    bad = copy.deepcopy(lineage)
    bad["expanded_test_node_registry_sha256"] = schema.INVALIDATED_MANIFEST_SHA256
    with pytest.raises(schema.PrelockSchemaV3Error, match="invalidated"):
        schema.parse_common_contract_lineage(bad, "s1")
    bad = copy.deepcopy(lineage)
    bad["full_local_import_closure"] = {}
    with pytest.raises(schema.PrelockSchemaV3Error, match="nonempty"):
        schema.parse_common_contract_lineage(bad, "s1")
    bad = copy.deepcopy(lineage)
    bad["rejected_lineage"][0]["reason"] = "superseded"
    with pytest.raises(schema.PrelockSchemaV3Error, match="rejection"):
        schema.parse_common_contract_lineage(bad, "s1")
    bad = copy.deepcopy(lineage)
    bad["rejected_lineage"][0]["scientific_effect"] = False
    with pytest.raises(schema.PrelockSchemaV3Error, match="exact integer"):
        schema.parse_common_contract_lineage(bad, "s1")
    assert schema.parse_common_contract_lineage(_lineage("s2"), "s2")[
        "verified_s1_receipt"
    ]["role"] == "v2_s1_finalization_root_record"
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.parse_common_contract_lineage(_lineage("s1"), "s2")
    assert schema.parse_lineage_binding({}, "contract", "s1") == ()
    contract = _artifact("v2_s1_contract")
    authorization = _artifact("v2_s1_authorization")
    assert len(schema.parse_lineage_binding({"contract": contract}, "authorization", "s1")) == 1
    assert len(schema.parse_lineage_binding(
        {"contract": contract, "authorization": authorization}, "bundle", "s1")) == 2
    bundle = _artifact("v2_s1_bundle_manifest")
    assert len(schema.parse_lineage_binding(
        {"contract": contract, "authorization": authorization, "bundle": bundle},
        "verifier_receipt", "s1",
    )) == 3
    wrong_bundle = _artifact("v2_s1_bundle")
    with pytest.raises(schema.PrelockSchemaV3Error, match="exact s1 records"):
        schema.parse_lineage_binding(
            {"contract": contract, "authorization": authorization, "bundle": wrong_bundle},
            "verifier_receipt", "s1",
        )
    with pytest.raises(schema.PrelockSchemaV3Error):
        schema.parse_lineage_binding({"authorization": authorization}, "contract", "s1")


def test_verifier_owned_finalization_payloads_are_exact_and_zero_authority() -> None:
    for stage in ("s1", "s2"):
        row = _finalization(stage)
        assert schema.parse_finalization_payload(row, stage)["fresh_process_replay"] is True
        changed = copy.deepcopy(row)
        changed["downstream_authority"]["formal_runs_authorized"] = 1
        with pytest.raises(schema.PrelockSchemaV3Error, match="authority"):
            schema.parse_finalization_payload(changed, stage)
        changed = copy.deepcopy(row)
        changed["fresh_process_replay"] = 1
        with pytest.raises(schema.PrelockSchemaV3Error, match="fresh"):
            schema.parse_finalization_payload(changed, stage)
        changed = copy.deepcopy(row)
        changed["producer_passed"] = True
        with pytest.raises(schema.PrelockSchemaV3Error, match="keys differ"):
            schema.parse_finalization_payload(changed, stage)
