from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_trkh_pair_surface_ddf_v2_fold_manifest.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ddf_v2_fold_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module() -> ModuleType:
    return _module()


@pytest.fixture(scope="module")
def generated(module: ModuleType):
    return module.build_manifest()


def _row(
    sample_index: int,
    image_n: int,
    *,
    leakage_group: str,
    target: int = 0,
):
    return {
        "sample_index": sample_index,
        "stem": f"image_{image_n}",
        "target": target,
        "image_n": image_n,
        "leakage_group": leakage_group,
        "image_rel": f"images/train/Image_{image_n}.jpg",
    }


def test_cohort_loader_accesses_only_allowlisted_metadata_keys(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = {
        "sample_indices": np.asarray([7, 8], dtype=np.int64),
        "targets": np.asarray([1, 4], dtype=np.int64),
        "source_stems": np.asarray(["image_7", "image_8"]),
        "image_paths": np.asarray([r"C:\x\Image_7.jpg", r"C:\x\Image_8.jpg"]),
        "folds": np.asarray([99, 99]),
        "keeper_probabilities": np.full((2, 5), np.nan),
        "model_boxes": np.full((2, 4), np.nan),
        "crop_boxes": np.full((2, 4), np.nan),
        "label_paths": np.asarray(["forbidden", "forbidden"]),
    }
    accessed = []
    expected_contract = {
        key: {
            "shape": tuple(int(value) for value in arrays[key].shape),
            "dtype": str(arrays[key].dtype),
            "sha256": module._array_sha256(arrays[key]),
        }
        for key in module.METADATA_ARRAY_KEYS
    }

    class FakeArchive:
        files = list(arrays)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            accessed.append(key)
            if key in module.FORBIDDEN_ARRAY_KEYS:
                raise AssertionError(f"forbidden array accessed: {key}")
            return arrays[key]

    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: FakeArchive())
    monkeypatch.setattr(module, "EXPECTED_METADATA_ARRAYS", expected_contract)
    loaded = module._load_cohort_metadata(Path("unused.npz"))

    assert tuple(accessed) == module.METADATA_ARRAY_KEYS
    assert set(loaded) == set(module.METADATA_ARRAY_KEYS)
    assert set(accessed).isdisjoint(module.FORBIDDEN_ARRAY_KEYS)


def test_union_is_transitive_and_order_independent(module: ModuleType) -> None:
    # 1--4 and 4--7 are w3 edges, so 1 and 7 share a component even though
    # their direct numeric gap is six. Image 100 joins the same component only
    # through the exact leakage group of Image 1. The second Image 1 row also
    # exercises exact-stem grouping.
    rows = [
        _row(10, 1, leakage_group="remote-link"),
        _row(11, 4, leakage_group="g4"),
        _row(12, 7, leakage_group="g7"),
        _row(13, 100, leakage_group="remote-link"),
        _row(14, 200, leakage_group="singleton"),
        _row(15, 1, leakage_group="second-object"),
    ]
    graph = module._build_relation_graph(rows)
    replay = module._build_relation_graph(list(reversed(rows)))

    assert graph["component_count"] == 2
    assert graph["largest_component_rows"] == 5
    assert graph["relation_edge_counts"] == {
        "exact_stem": 1,
        "exact_leakage_group": 1,
        "image_n_w3": 4,
    }
    assert graph["edge_set_sha256"] == replay["edge_set_sha256"]
    assert graph["component_set_sha256"] == replay["component_set_sha256"]
    assert graph["component_order_sha256"] == replay["component_order_sha256"]


def test_cohort_loader_rejects_archive_member_schema_before_deserializing(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadArchive:
        files = [*module.METADATA_ARRAY_KEYS, "unexpected_scores"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            raise AssertionError(f"member must not be deserialized: {key}")

    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: BadArchive())
    with pytest.raises(ValueError, match="member schema differs"):
        module._load_cohort_metadata(Path("unused.npz"))


def test_lexicographic_milp_has_a_unique_score_independent_tie_break(
    module: ModuleType,
) -> None:
    components = [
        {
            "component_sha256": f"{index:064x}",
            "rows_internal": [
                _row(index, 1000 + index, leakage_group=f"g{index}", target=4)
            ],
        }
        for index in range(10)
    ]
    assignment = module._assign_components(
        components,
        fold_count=5,
        row_bounds=(2, 2),
        class_bounds={4: (2, 2)},
    )
    replay = module._assign_components(
        components,
        fold_count=5,
        row_bounds=(2, 2),
        class_bounds={4: (2, 2)},
    )

    assert assignment["status"] == 0
    assert assignment["fold_vector"] == "0011223344"
    assert replay["fold_vector"] == assignment["fold_vector"]
    assert assignment["integrality_max_abs"] == 0.0


def test_milp_fails_closed_when_a_component_exceeds_a_fold(
    module: ModuleType,
) -> None:
    component = {
        "component_sha256": "a" * 64,
        "rows_internal": [
            _row(index, index, leakage_group="one") for index in range(3)
        ],
    }
    with pytest.raises(ValueError, match="component exceeds"):
        module._assign_components(
            [component],
            fold_count=2,
            row_bounds=(1, 2),
            class_bounds={0: (0, 2)},
        )


def test_manifest_join_rejects_duplicate_outputs(
    module: ModuleType,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "split,leakage_group,output_image\n"
        "train,g1,C:\\data\\Image_1.jpg\n"
        "train,g2,C:\\data\\Image_1.jpg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate manifest output_image"):
        module._load_yolo_manifest(csv_path)


def test_manifest_loader_rejects_malformed_short_row(
    module: ModuleType,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "split,leakage_group,output_image\n"
        "train,g1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed manifest row"):
        module._load_yolo_manifest(csv_path)


def test_canonical_join_rejects_group_crossing_output_splits(
    module: ModuleType,
) -> None:
    path = r"C:\data\images\train\Image_1.jpg"
    cohort = {
        "sample_indices": np.asarray([1], dtype=np.int64),
        "targets": np.asarray([1], dtype=np.int64),
        "source_stems": np.asarray(["Image_1"]),
        "image_paths": np.asarray([path]),
    }
    manifest = {
        "by_output": {
            module._normalized_output_path(path): {
                "output_image": path,
                "leakage_group": "g1",
                "split": "train",
            }
        },
        "group_splits": {"g1": {"train", "val"}},
        "group_rows": {"g1": 2},
        "total_rows": 2,
    }
    with pytest.raises(ValueError, match="crosses YOLO output splits"):
        module._canonical_rows(cohort, manifest)


def test_real_manifest_locks_graph_solver_mapping_and_counts(
    module: ModuleType,
    generated,
) -> None:
    assert generated["scope"]["metadata_array_members_deserialized"] == list(
        module.METADATA_ARRAY_KEYS
    )
    assert generated["scope"]["forbidden_array_members_not_deserialized"] == list(
        module.FORBIDDEN_ARRAY_KEYS
    )
    assert generated["scope"][
        "whole_cohort_container_bytes_hashed_for_opaque_lineage"
    ] is True
    assert generated["scope"][
        "whole_cohort_container_hash_used_as_assignment_feature"
    ] is False
    assert generated["scope"][
        "keeper_probabilities_or_candidate_scores_deserialized"
    ] is False
    assert generated["scope"][
        "cross_split_manifest_identity_metadata_read_for_group_audit"
    ] is True
    assert generated["inputs"]["predecessor_manifest"]["assignment_influence"] is False
    assert generated["supersession"]["mapping_changed"] is False
    selected = generated["inputs"]["selected_npz_members"]
    expected_selected = {
        key: {
            "shape": list(module.EXPECTED_METADATA_ARRAYS[key]["shape"]),
            "dtype": module.EXPECTED_METADATA_ARRAYS[key]["dtype"],
            "sha256": module.EXPECTED_METADATA_ARRAYS[key]["sha256"],
        }
        for key in module.METADATA_ARRAY_KEYS
    }
    assert selected["members"] == expected_selected
    assert selected["selected_member_set_sha256"] == module._json_sha256(
        expected_selected
    )
    assert generated["identity_with_predecessor"]["all_equal"] is True
    assert generated["identity_with_predecessor"]["fold_vector"] == generated[
        "solver"
    ]["fold_vector"]
    exporter = generated["inputs"]["leakage_group_exporter"]
    assert exporter["sha256"] == module.CVAT_EXPORTER_SHA256
    assert exporter["git_commit"] == module.CVAT_EXPORTER_GIT_COMMIT
    assert exporter["git_blob_oid"] == module.CVAT_EXPORTER_GIT_BLOB
    assert "absolute resolved source paths" in exporter["portability_caveat"]
    assert generated["canonical_metadata"]["sha256"] == (
        module.CANONICAL_METADATA_SHA256
    )
    assert generated["canonical_metadata"]["target_counts"] == {
        "0": 158,
        "1": 541,
        "2": 54,
        "3": 0,
        "4": 10,
    }
    graph = generated["relation_graph"]
    assert graph["edge_count"] == 1372
    assert graph["edge_set_sha256"] == module.EDGE_SET_SHA256
    assert graph["component_count"] == 158
    assert graph["component_set_sha256"] == module.COMPONENT_SET_SHA256
    assert graph["component_order_sha256"] == module.COMPONENT_ORDER_SHA256

    solver = generated["solver"]
    assert solver["status"] == 0
    assert "HiGHS Status 7: Optimal" in solver["status_text"]
    assert solver["feasibility_calls"] == 451
    assert solver["integrality_max_abs"] == 0.0
    assignment = generated["assignment"]
    assert assignment["mapping_sha256"] == module.MAPPING_SHA256
    assert assignment["component_assignments_sha256"] == (
        module.COMPONENT_ASSIGNMENTS_SHA256
    )
    assert assignment["cross_fold_relation_edges"] == 0
    assert assignment["cross_fold_components"] == 0
    assert len(assignment["rows"]) == 763
    assert len({row["sample_index"] for row in assignment["rows"]}) == 763
    assert [fold["rows"] for fold in assignment["folds"]] == [153, 153, 153, 152, 152]
    assert [fold["target_counts"]["4"] for fold in assignment["folds"]] == [
        2,
        2,
        2,
        2,
        2,
    ]


def test_runtime_and_calibration_are_pinned_for_research_use(
    module: ModuleType,
    generated,
) -> None:
    solver = generated["solver"]
    assert solver["python_version"] == module.EXPECTED_PYTHON
    assert solver["numpy_version"] == module.EXPECTED_NUMPY
    assert solver["scipy_version"] == module.EXPECTED_SCIPY
    assert solver["highs_wrapper_sha256"] == (
        module.EXPECTED_HIGHS_WRAPPER_SHA256
    )
    assert solver["solver_options"] == module.SOLVER_OPTIONS

    outer = generated["calibration"]["outer_folds"]
    assert generated["state"] == (
        "frozen_v2r2_provenance_hardened_train_metadata_"
        "manifest_group_numeric_neighborhood_disjoint"
    )
    assert generated["calibration"]["authorized_mode"] == (
        "fixed_simple_one_fold_calibration_only"
    )
    assert generated["calibration"]["mapping"] == (
        "calibration_fold=(held_fold+1)%5"
    )
    assert generated["calibration"]["fit_fold_count"] == 3
    assert generated["calibration"]["alternative_calibration_modes_authorized"] is False
    assert [item["calibration_fold"] for item in outer] == [1, 2, 3, 4, 0]
    assert all(item["calibration"]["target_counts"]["4"] == 2 for item in outer)
    assert all(item["fit"]["target_counts"]["4"] == 6 for item in outer)
    assert all(
        "same fitted model predicts both calibration and held rows"
        in item["candidate_model_rule"]
        for item in outer
    )
    assert all(item["component_overlap_fit_calibration_held"] == 0 for item in outer)
    assert "nested" not in json.dumps(generated["calibration"]).casefold()
    assert generated["canonical_metadata"]["target_counts"]["3"] == 0
    assert generated["legacy_fold_diagnostics"]["assignment_influence"] is False


def test_check_only_replays_frozen_json_and_sha(
    module: ModuleType,
    generated,
) -> None:
    module._assert_check_only_matches_frozen(generated)
    parts = module.OUTPUT_SHA_PATH.read_text(encoding="ascii").split()
    assert parts == [module._sha256(module.OUTPUT_PATH), module.OUTPUT_PATH.name]
    frozen = json.loads(module.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert frozen == generated
    assert frozen["builder"]["sha256"] == module._sha256(SCRIPT)


def test_check_only_rejects_a_bad_sha_record(
    module: ModuleType,
    generated,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / module.OUTPUT_PATH.name
    sidecar = output.with_suffix(".sha256")
    output.write_text(json.dumps(generated), encoding="utf-8")
    sidecar.write_text(f"{'0' * 64}  {output.name}\n", encoding="ascii")
    monkeypatch.setattr(module, "OUTPUT_PATH", output)
    monkeypatch.setattr(module, "OUTPUT_SHA_PATH", sidecar)
    with pytest.raises(ValueError, match="SHA differs"):
        module._assert_check_only_matches_frozen(generated)


def test_cli_is_check_only_by_default_and_write_path_is_guarded(
    module: ModuleType,
    generated,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert module.parse_args([]).write is False
    assert module.parse_args(["--check-only"]).write is False
    assert module.parse_args(["--write"]).write is True
    assert generated["write_contract"]["default_check_only"] is True
    assert generated["write_contract"]["pair_transactional"] is False
    assert generated["write_contract"]["partial_pair_fails_closed"] is True

    monkeypatch.setattr(module, "EXPECTED_REPOSITORY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="expected repository"):
        module._assert_expected_repository_path()


def test_explicit_write_uses_fsync_and_atomic_replace(
    module: ModuleType,
    generated,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / module.OUTPUT_PATH.name
    sidecar = output.with_suffix(".sha256")
    replacements = []
    fsync_calls = []
    real_replace = module.os.replace
    real_fsync = module.os.fsync

    def tracked_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    def tracked_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(module, "OUTPUT_PATH", output)
    monkeypatch.setattr(module, "OUTPUT_SHA_PATH", sidecar)
    monkeypatch.setattr(module, "_assert_expected_repository_path", lambda: None)
    monkeypatch.setattr(module.os, "replace", tracked_replace)
    monkeypatch.setattr(module.os, "fsync", tracked_fsync)

    digest = module.write_manifest(generated)

    assert digest == module._sha256(output)
    assert len(replacements) == 2
    assert [destination for _, destination in replacements] == [output, sidecar]
    assert len(fsync_calls) == 2
    assert sidecar.read_text(encoding="ascii").split() == [digest, output.name]
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(FileExistsError, match="one-shot write refuses overwrite"):
        module.write_manifest(generated)
