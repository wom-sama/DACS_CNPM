from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trkh.tools import audit_pair_surface_ddf_v2_geometry_metadata as audit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
)


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rows = audit.EXPECTED_ROWS
    cohort_path = tmp_path / "cohort_arrays.npz"
    np.savez(
        cohort_path,
        sample_indices=np.arange(rows, dtype=np.int64),
        targets=(np.arange(rows, dtype=np.int64) % 4)[
            np.newaxis, ...
        ].reshape(-1),
        model_boxes=np.tile(
            np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
            (rows, 1),
        ),
    )
    # Replace synthetic class index 3 with the fourth supported TRKH class 4.
    with np.load(cohort_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    payload["targets"][payload["targets"] == 3] = 4
    np.savez(cohort_path, **payload)

    valid256 = np.ones((rows, 256, 256), dtype=np.bool_)
    packed = np.packbits(valid256.reshape(rows, -1), axis=1, bitorder="little")
    valid_path = tmp_path / "image_valid_masks_packbits.npy"
    np.save(valid_path, packed)
    return cohort_path, valid_path


def test_numpy_and_torch_validity_pipelines_are_exact_for_irregular_masks() -> None:
    valid = np.zeros((3, 256, 256), dtype=np.bool_)
    valid[0, 0, 0] = True
    valid[0, 255, 255] = True
    valid[1, 31:193, 47:211] = True
    valid[2, ::7, ::11] = True

    numpy_masks = audit.propagate_valid_masks_numpy(valid)
    torch_masks = audit.propagate_valid_masks_torch(valid)

    assert [value.shape for value in numpy_masks] == [
        (3, 64, 64),
        (3, 32, 32),
        (3, 16, 16),
    ]
    assert all(
        np.array_equal(numpy_value, torch_value)
        for numpy_value, torch_value in zip(numpy_masks, torch_masks)
    )
    assert numpy_masks[0][0, 0, 0]
    assert numpy_masks[0][0, -1, -1]


def test_model_bbox_rasterization_uses_locked_floor_ceil_rule() -> None:
    boxes = np.asarray(
        [
            [0.5, 0.5, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    masks = audit.rasterize_model_boxes(boxes)

    assert masks.shape == (3, 16, 16)
    assert masks[0].sum() == 16
    assert masks[0, 6:10, 6:10].all()
    assert masks[1].sum() == 1 and masks[1, 0, 0]
    assert masks[2].sum() == 1 and masks[2, 15, 15]


def test_report_discloses_geometry_only_access_and_exact_denominator(
    tmp_path: Path,
) -> None:
    cohort_path, valid_path = _synthetic_inputs(tmp_path)
    report = audit.build_report(
        cohort_path,
        valid_path,
        enforce_locked_inputs=False,
    )

    assert report["access_contract"] == {
        "candidate_scores_read": 0,
        "cohort_members_read": ["sample_indices", "targets", "model_boxes"],
        "fold_assignment_performed": False,
        "image_or_label_paths_opened": 0,
        "keeper_probability_arrays_read": 0,
        "old_or_v2_fold_arrays_read": 0,
        "rgb_images_read": 0,
        "test_data_used": False,
        "train_table_only": True,
        "training_launched": False,
        "validation_data_used": False,
    }
    assert report["definitions"]["q"] == (
        "sum(bbox16 AND valid16) / sum(valid16)"
    )
    assert report["cohort"]["bbox_usable_rows"] == audit.EXPECTED_ROWS
    assert report["cohort"]["bbox_unusable_rows"] == 0
    assert report["bbox_geometry"]["q_all_763_rows"]["mean"] == pytest.approx(
        16.0 / 256.0
    )
    assert report["automatic_passed"] is False


def test_cohort_loader_does_not_read_scores_folds_or_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = audit.EXPECTED_ROWS
    expected = {
        "sample_indices": np.arange(rows, dtype=np.int64),
        "targets": np.ones(rows, dtype=np.int64),
        "model_boxes": np.ones((rows, 4), dtype=np.float32),
    }
    accessed: list[str] = []

    class GuardedArchive:
        files = [
            *audit.REQUIRED_COHORT_ARRAYS,
            "keeper_probabilities",
            "folds",
            "image_paths",
            "label_paths",
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __getitem__(self, name: str) -> np.ndarray:
            accessed.append(name)
            if name not in audit.REQUIRED_COHORT_ARRAYS:
                raise AssertionError(f"Forbidden cohort member read: {name}")
            return expected[name]

    monkeypatch.setattr(audit.np, "load", lambda *_args, **_kwargs: GuardedArchive())
    loaded = audit._load_cohort_geometry(tmp_path / "guarded.npz")

    assert set(loaded) == set(audit.REQUIRED_COHORT_ARRAYS)
    assert accessed == list(audit.REQUIRED_COHORT_ARRAYS)


@pytest.mark.skipif(
    not (CACHE_ROOT / "cohort_arrays.npz").is_file(),
    reason="Materialized train-table cache is not distributed with source checkout",
)
def test_locked_train_table_reproduces_protocol_geometry_numbers() -> None:
    report = audit.build_report(
        CACHE_ROOT / "cohort_arrays.npz",
        CACHE_ROOT / "image_valid_masks_packbits.npy",
    )

    assert report["automatic_passed"] is True
    assert report["cohort"]["bbox_usable_rows"] == 751
    assert report["cohort"]["bbox_unusable_rows"] == 12
    assert report["cohort"]["bbox_unusable_target_counts"] == {
        "0": 2,
        "1": 9,
        "4": 1,
    }
    assert report["cohort"]["bbox_unusable_records_sha256"] == (
        audit.EXPECTED_BBOX_UNUSABLE_RECORD_SHA256
    )
    assert report["bbox_geometry"]["q_all_763_rows"]["mean"] == pytest.approx(
        audit.EXPECTED_ALL_Q_MEAN,
        abs=1e-15,
    )
    assert report["bbox_geometry"]["q_bbox_usable_751_rows"][
        "mean"
    ] == pytest.approx(audit.EXPECTED_USABLE_Q_MEAN, abs=1e-15)
    regression = report["bitorder_regression"]
    assert regression["wrong_big_q_all_763_rows"]["mean"] == pytest.approx(
        audit.LEGACY_WRONG_BIG_ALL_Q_MEAN,
        abs=1e-15,
    )
    assert regression["wrong_big_and_correct_unusable_records_match"] is True
    assert regression["correct_and_wrong_valid16_sha256_differ"] is True
    assert report["locked_checks"]["legacy_wrong_big_signature"] is True


def test_input_path_with_test_component_is_rejected(tmp_path: Path) -> None:
    forbidden = tmp_path / "test" / "cohort_arrays.npz"
    forbidden.parent.mkdir(parents=True)
    np.savez(forbidden, placeholder=np.asarray([1]))
    with pytest.raises(ValueError, match="Validation/test path components"):
        audit._assert_train_table_input(forbidden, suffix=".npz")
