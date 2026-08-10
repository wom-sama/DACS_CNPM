from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "build_trkh_cap_integral_region_context_a0_lock.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("cap_lock_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cap_region_geometry_is_exact_and_full_image_is_last() -> None:
    module = _module()
    boxes = module._region_boxes()
    assert boxes.shape == (27, 4)
    assert boxes.dtype == np.int64
    assert boxes[-1].tolist() == [0, 0, 42, 42]
    assert len({tuple(row) for row in boxes.tolist()}) == 27
    assert all(
        width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= 42
        and y + height <= 42
        for x, y, width, height in boxes.tolist()
    )


def test_cap_visual_rows_are_fixed_unique_and_cover_each_fold_stratum() -> None:
    module = _module()
    with np.load(module.GEOMETRY_CACHE, allow_pickle=False) as archive:
        samples = np.asarray(archive["sample_indices"])
        targets = np.asarray(archive["targets"])
        folds = np.asarray(archive["folds"])
        predictions = np.asarray(archive["keeper_probabilities"]).argmax(axis=1)
    selected = module.VISUAL_SAMPLE_INDICES
    assert selected.shape == (20,)
    assert len(set(selected.tolist())) == 20
    positions = np.asarray(
        [int(np.flatnonzero(samples == value)[0]) for value in selected],
        dtype=np.int64,
    )
    for fold in range(5):
        current = positions[fold * 4 : (fold + 1) * 4]
        assert np.all(folds[current] == fold)
        pairs = [(int(targets[index]), int(predictions[index])) for index in current]
        assert pairs[0] == (1, 1)
        assert pairs[1][0] == 1 and pairs[1][1] != 1
        assert pairs[2] == (0, 1)
        assert pairs[3][0] in {2, 4} and pairs[3][1] == 1


def test_cap_fold_roles_are_source_disjoint_and_hash_stable() -> None:
    module = _module()
    arrays = module._load_inputs()
    folds = arrays["folds"].astype(np.int64)
    targets = arrays["targets"].astype(np.int64)
    samples = arrays["sample_indices"].astype(np.int64)
    probabilities = arrays["keeper_probabilities"].astype(np.float32)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    sources = arrays["source_stems"].astype(str).tolist()
    payloads = [
        module._fold_payload(
            fold,
            folds,
            targets,
            predictions,
            samples,
            sources,
        )
        for fold in range(5)
    ]
    assert [item["calibration_fold"] for item in payloads] == [1, 2, 3, 4, 0]
    assert all(item["source_overlap"] == 0 for item in payloads)
    assert all(
        item["cross_sample_derangement_source_overlap"] == 0
        for item in payloads
    )
    assert all(
        item["cross_sample_derangement_partition_escape"] == 0
        for item in payloads
    )
    assert all(
        all(
            detail["outside_partition"] == 0
            and detail["source_overlap"] == 0
            and detail["self_pairs"] == 0
            and detail["unique_partners"] == detail["rows"]
            for detail in item["cross_sample_derangement_partitions"].values()
        )
        for item in payloads
    )
    assert all(len(item["primary_orders"]["per_epoch_sha256"]) == 30 for item in payloads)
    assert all(len(item["repeat_orders"]["per_epoch_sha256"]) == 30 for item in payloads)


def test_cap_spatial_derangement_and_valid_support_are_strict() -> None:
    module = _module()
    arrays = module._load_inputs()
    permutations = module._spatial_permutations(arrays["sample_indices"])
    expected = np.arange(27, dtype=np.int64)
    assert permutations.shape == (763, 27)
    assert all(
        np.array_equal(np.sort(permutation), expected)
        for permutation in permutations
    )
    assert not np.any(permutations == expected[None, :])
    boxes = module._valid_support_boxes(arrays["valid_masks"])
    assert boxes.shape == (763, 4)
    for mask, (x0, y0, x1, y1) in zip(arrays["valid_masks"], boxes.tolist()):
        assert mask[y0:y1, x0:x1].all()
        assert int(mask.sum()) == (y1 - y0) * (x1 - x0)


def test_cap_lock_build_is_prospective_and_contains_no_candidate_metric() -> None:
    module = _module()
    lock = module.build_lock()
    assert lock["state"] == "prospective_no_candidate_metric"
    assert lock["cohort"]["counts"] == {
        "tp1": 528,
        "fn1": 13,
        "restricted_fp": 222,
        "0_to_1": 158,
        "2_to_1": 54,
        "4_to_1": 10,
    }
    assert lock["geometry"]["regions"] == 27
    assert lock["geometry"]["spatial_derangement"]["fixed_points"] == 0
    assert lock["architecture"]["netvlad_clusters"] == 32
    assert lock["architecture"]["padding_enters_cap"] is False
    assert lock["architecture"]["pre_cap_channel_recalibration"] == (
        "none_paper_equation"
    )
    assert lock["architecture"]["pixel_projection_normalization"] == (
        "none_paper_equation"
    )
    assert lock["architecture"]["region_attention_pre_softmax_activation"] == (
        "none_paper_equation"
    )
    assert lock["architecture"]["cross_sample_context_control"] == {
        "queries": "own_regions",
        "keys": "partition_mapped_different_source_regions",
        "values": "own_regions",
        "mapping_uses_labels": False,
    }
    assert lock["optimization"]["epochs"] == 30
    assert lock["calibration"]["fn_rescue_enabled"] is False
    text = str(lock).lower()
    assert "candidate_metric" not in text.replace(
        "prospective_no_candidate_metric", ""
    )
    assert "validation_metric" not in text
    assert "test_metric" not in text
