from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof as b6

from trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof import (
    CACHE_MANIFEST_SCHEMA_VERSION,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DATA_SHA256,
    EXPECTED_HEAD_PARAMETERS,
    EXPECTED_TRAIN_SAMPLES,
    FOLDS,
    GLOBAL_LOGIT_WIDTH,
    MAX_HEAD_MACS,
    PAIR_COUNT,
    PAIR_RIVALS,
    PATCH_TOKENS,
    PROTOCOL_ID,
    TOKEN_WIDTH,
    _PairTokenDataset,
    _cache_paths,
    _load_or_extract_cache,
    _parse_args,
    _path_rows_sha256,
    _sha256,
    _source_group_manifest_fields,
    _train_readout,
    _validate_cache_manifest,
    _validate_fold_assignment_contract,
    _validate_oof_prediction_coverage,
    architecture_self_check,
    assess_deepsets_readiness,
    expand_pair_rows,
    permute_fit_targets_within_pairs,
    six_stratum_weights,
)


def _passing_rows():
    pairs = []
    folds = []
    for rival in PAIR_RIVALS:
        pairs.append(
            {
                "pair": f"{rival}-1",
                "rival_class": rival,
                "delta_auroc": 0.02,
                "delta_f1_class1": 0.02,
                "delta_recall_class1": -0.005,
                "control_tp": 100.0,
                "candidate_tp": 99.0,
                "control_fp": 50.0,
                "candidate_fp": 40.0,
                "rival_fp_reduction": 0.20,
            }
        )
        for fold in range(FOLDS):
            folds.append(
                {
                    "pair": f"{rival}-1",
                    "rival_class": rival,
                    "fold": fold,
                    "source_overlap": 0,
                    "control_auroc": 0.70,
                    "candidate_auroc": 0.72,
                    "placebo_auroc": 0.50,
                    "delta_auroc": 0.02,
                    "control_f1_class1": 0.65,
                    "candidate_f1_class1": 0.67,
                }
            )
    global_folds = [{"fold": fold, "source_overlap": 0} for fold in range(FOLDS)]
    return pairs, folds, global_folds


@pytest.fixture(scope="module")
def locked_architecture():
    return architecture_self_check()


def test_architecture_matches_locked_capacity_and_invariances(
    locked_architecture,
) -> None:
    result = locked_architecture
    assert result["candidate_parameters"] == EXPECTED_HEAD_PARAMETERS == 10_577
    assert result["control_parameters"] == EXPECTED_HEAD_PARAMETERS
    assert result["estimated_head_macs"] <= MAX_HEAD_MACS
    assert result["token_permutation_max_abs_error"] <= 1e-6
    assert result["constant_bag_control_max_abs_error"] <= 1e-6
    assert result["all_pairs_path_max_abs_error"] <= 1e-6
    assert result["initial_state_max_abs_error"] == 0.0


def test_pair_expansion_balancing_and_pairwise_placebo() -> None:
    labels = np.asarray([0, 1, 2, 3, 4, 1], dtype=np.int64)
    rows = expand_pair_rows(labels)
    assert rows["sample_indices"].size == 9
    assert np.bincount(rows["pair_ids"], minlength=PAIR_COUNT).tolist() == [3, 3, 3]
    assert np.bincount(rows["targets"], minlength=2).tolist() == [3, 6]

    weights = six_stratum_weights(rows["pair_ids"], rows["targets"])
    totals = []
    for pair_id in range(PAIR_COUNT):
        for target in (0, 1):
            mask = np.logical_and(
                rows["pair_ids"] == pair_id,
                rows["targets"] == target,
            )
            totals.append(float(weights[mask].sum()))
    assert np.allclose(totals, totals[0])

    placebo = permute_fit_targets_within_pairs(
        rows["pair_ids"], rows["targets"], seed=20260731
    )
    for pair_id in range(PAIR_COUNT):
        mask = rows["pair_ids"] == pair_id
        assert int(placebo[mask].sum()) == int(rows["targets"][mask].sum())


def test_six_stratum_balance_guard_scales_for_locked_float32_support() -> None:
    # Fold-0 fit support from the locked class_f source-group assignment.  The
    # old absolute 1e-5 guard rejected these correctly balanced float32
    # weights solely because their total is about 954.5 per stratum.
    counts = [1582, 388, 1067, 388, 1914, 388]
    pair_ids = []
    targets = []
    for stratum, count in enumerate(counts):
        pair_ids.extend([stratum // 2] * count)
        targets.extend([stratum % 2] * count)
    pairs = np.asarray(pair_ids, dtype=np.int64)
    values = np.asarray(targets, dtype=np.int64)

    weights = six_stratum_weights(pairs, values)
    expected_total = float(values.size) / float(2 * PAIR_COUNT)
    totals = [
        float(
            weights[np.logical_and(pairs == pair, values == target)].sum(
                dtype=np.float64
            )
        )
        for pair in range(PAIR_COUNT)
        for target in (0, 1)
    ]

    assert max(abs(total - expected_total) for total in totals) <= (
        2.0 * np.finfo(np.float32).eps * expected_total
    )


def test_pooled_control_repeats_the_deployed_pool_256_times() -> None:
    labels = np.asarray([0, 1, 2, 4], dtype=np.int64)
    rows = expand_pair_rows(labels)
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(4, PATCH_TOKENS, TOKEN_WIDTH)).astype(np.float16)
    logits = rng.normal(size=(4, GLOBAL_LOGIT_WIDTH)).astype(np.float32)
    control = _PairTokenDataset(
        token_cache=tokens,
        global_logits=logits,
        rows=rows,
        pooled_control=True,
    )
    item_tokens = control[0][0].numpy()
    expected = tokens[int(rows["sample_indices"][0])].astype(np.float32).mean(axis=0)
    assert item_tokens.shape == (PATCH_TOKENS, TOKEN_WIDTH)
    assert np.allclose(item_tokens, expected[None, :])


def test_readout_training_is_deterministic_for_identical_seed_and_order() -> None:
    labels = np.asarray([0, 1, 2, 4, 0, 1, 2, 4], dtype=np.int64)
    rows = expand_pair_rows(labels)
    rng = np.random.default_rng(11)
    tokens = rng.normal(size=(8, 4, TOKEN_WIDTH)).astype(np.float16)
    logits = rng.normal(size=(8, GLOBAL_LOGIT_WIDTH)).astype(np.float32)
    dataset = _PairTokenDataset(
        token_cache=tokens,
        global_logits=logits,
        rows=rows,
        pooled_control=False,
    )
    first, first_report = _train_readout(
        dataset=dataset,
        device=torch.device("cpu"),
        seed=123,
        epochs=1,
        batch_size=len(dataset),
    )
    second, second_report = _train_readout(
        dataset=dataset,
        device=torch.device("cpu"),
        seed=123,
        epochs=1,
        batch_size=len(dataset),
    )
    assert first_report == second_report
    assert first_report["successful_optimizer_updates"] == 1
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_cache_manifest_uses_distinct_file_and_label_content_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b6, "EXPECTED_TRAIN_SAMPLES", 5)
    cache_paths = {
        "tokens": tmp_path / "tokens.npy",
        "logits": tmp_path / "logits.npy",
        "labels": tmp_path / "labels.npy",
    }
    for index, path in enumerate(cache_paths.values()):
        path.write_bytes(f"cache-{index}".encode("ascii"))
    expected_labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    source_groups = np.asarray(
        [f"source-{index}" for index in range(expected_labels.size)],
        dtype=object,
    )
    labels_content = np.asarray(expected_labels, dtype="<i8").tobytes()
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": "locked-paths",
        "token_shape": [5, PATCH_TOKENS, TOKEN_WIDTH],
        "token_dtype": "float16",
        "global_logits_shape": [5, GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [5],
        "labels_dtype": "int64",
        "tokens_sha256": _sha256(cache_paths["tokens"]),
        "logits_sha256": _sha256(cache_paths["logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
        "labels_content_sha256": hashlib.sha256(labels_content).hexdigest(),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    manifest.update(_source_group_manifest_fields(source_groups))
    _validate_cache_manifest(
        manifest=manifest,
        cache_paths=cache_paths,
        expected_paths_sha256="locked-paths",
        expected_source_groups=source_groups,
        expected_labels=expected_labels,
    )
    cache_paths["labels"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        _validate_cache_manifest(
            manifest=manifest,
            cache_paths=cache_paths,
            expected_paths_sha256="locked-paths",
            expected_source_groups=source_groups,
            expected_labels=expected_labels,
        )


def test_legacy_cache_manifest_upgrades_without_reextracting_or_touching_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b6, "EXPECTED_TRAIN_SAMPLES", 2)
    monkeypatch.setattr(b6, "PATCH_TOKENS", 2)
    monkeypatch.setattr(b6, "TOKEN_WIDTH", 3)
    paths = [str(tmp_path / "image_0.jpg"), str(tmp_path / "image_1.jpg")]
    source_groups = np.asarray(["image_0", "image_1"], dtype=object)
    expected_labels = np.asarray([0, 1], dtype=np.int64)
    cache_paths = _cache_paths(tmp_path)
    np.save(
        cache_paths["tokens"],
        np.arange(12, dtype=np.float16).reshape(2, 2, 3),
        allow_pickle=False,
    )
    np.save(
        cache_paths["logits"],
        np.arange(10, dtype=np.float32).reshape(2, GLOBAL_LOGIT_WIDTH),
        allow_pickle=False,
    )
    np.save(cache_paths["labels"], expected_labels, allow_pickle=False)
    legacy_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_content_sha256": hashlib.sha256(
            np.asarray(expected_labels, dtype="<i8").tobytes()
        ).hexdigest(),
        "token_shape": [2, 2, 3],
        "token_dtype": "float16",
        "global_logits_shape": [2, GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [2],
        "labels_dtype": "int64",
        "tokens_sha256": _sha256(cache_paths["tokens"]),
        "logits_sha256": _sha256(cache_paths["logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    cache_paths["manifest"].write_text(
        json.dumps(legacy_manifest, indent=2),
        encoding="utf-8",
    )
    cache_hashes = {
        key: _sha256(cache_paths[key]) for key in ("tokens", "logits", "labels")
    }
    cache_mtimes = {
        key: cache_paths[key].stat().st_mtime_ns
        for key in ("tokens", "logits", "labels")
    }

    def _unexpected_extraction(**_kwargs):
        raise AssertionError("legacy cache reuse must not invoke extraction")

    monkeypatch.setattr(b6, "_extract_train_cache", _unexpected_extraction)
    loaded = _load_or_extract_cache(
        model=object(),
        dataset=object(),
        paths=paths,
        source_groups=source_groups,
        expected_labels=expected_labels,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=1,
        workers=0,
    )
    upgraded = loaded["manifest"]
    assert upgraded["schema_version"] == CACHE_MANIFEST_SCHEMA_VERSION
    assert upgraded["legacy_schema_upgraded_from"] == 1
    assert upgraded["source_group_count"] == 2
    assert upgraded["source_group_rows"] == 2
    assert {
        key: _sha256(cache_paths[key]) for key in cache_hashes
    } == cache_hashes
    assert {
        key: cache_paths[key].stat().st_mtime_ns for key in cache_mtimes
    } == cache_mtimes
    del loaded

    with pytest.raises(ValueError, match="cache manifest mismatch"):
        _load_or_extract_cache(
            model=object(),
            dataset=object(),
            paths=paths,
            source_groups=source_groups,
            expected_labels=expected_labels[::-1],
            output_dir=tmp_path,
            device=torch.device("cpu"),
            batch_size=1,
            workers=0,
        )


def test_fold_and_oof_contracts_reject_split_or_prediction_gaps() -> None:
    labels = np.asarray([0, 1, 2, 3, 4, 1], dtype=np.int64)
    source_groups = np.asarray(
        [f"source-{index}" for index in range(labels.size)],
        dtype=object,
    )
    assignments = np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64)
    paths = [f"image-{index}.jpg" for index in range(labels.size)]
    fold_report = _validate_fold_assignment_contract(
        source_groups,
        assignments,
    )
    assert fold_report["assignment_rows"] == labels.size
    assert fold_report["assignment_folds"] == list(range(FOLDS))

    rows = []
    for sample_index, target_index in enumerate(labels.tolist()):
        rivals = (
            PAIR_RIVALS
            if target_index == 1
            else (target_index,)
            if target_index in PAIR_RIVALS
            else ()
        )
        for rival in rivals:
            rows.append(
                {
                    "sample_index": sample_index,
                    "image_path": paths[sample_index],
                    "source_group": str(source_groups[sample_index]),
                    "fold": int(assignments[sample_index]),
                    "target_index": target_index,
                    "pair": f"{rival}-1",
                    "binary_target_class1": int(target_index == 1),
                    "control_probability_class1": 0.5,
                    "candidate_probability_class1": 0.5,
                    "placebo_probability_class1": 0.5,
                }
            )
    coverage = _validate_oof_prediction_coverage(
        prediction_rows=rows,
        labels=labels,
        source_groups=source_groups,
        fold_assignments=assignments,
        paths=paths,
    )
    assert coverage["expected_prediction_rows"] == 9
    assert coverage["unique_prediction_keys"] == 9

    with pytest.raises(RuntimeError, match="duplicated"):
        _validate_oof_prediction_coverage(
            prediction_rows=[*rows, dict(rows[0])],
            labels=labels,
            source_groups=source_groups,
            fold_assignments=assignments,
            paths=paths,
        )

    split_group_values = source_groups.copy()
    split_group_values[1] = split_group_values[0]
    with pytest.raises(RuntimeError, match="multiple holdout folds"):
        _validate_fold_assignment_contract(split_group_values, assignments)

    invalid_assignments = assignments.copy()
    invalid_assignments[-1] = FOLDS
    with pytest.raises(ValueError, match="exactly the locked folds"):
        _validate_fold_assignment_contract(source_groups, invalid_assignments)


def test_readiness_passes_all_locked_gates_and_rejects_duplicate_coverage(
    locked_architecture,
) -> None:
    pairs, folds, global_folds = _passing_rows()
    result = assess_deepsets_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=7751,
        architecture=locked_architecture,
        maximum_pool_parity_error=0.0,
        cache_finite=True,
    )
    assert result["deepsets_patch_distribution_ready"]
    assert result["failed_checks"] == []
    assert not result["validation_permission"]
    assert not result["test_permission"]

    folds[-1] = dict(folds[0])
    rejected = assess_deepsets_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=7751,
        architecture=locked_architecture,
        maximum_pool_parity_error=0.0,
        cache_finite=True,
    )
    assert not rejected["deepsets_patch_distribution_ready"]
    assert "complete_pair_fold_coverage" in rejected["failed_checks"]

    pairs, folds, global_folds = _passing_rows()
    duplicate_pairs = [dict(pairs[0]), dict(pairs[1]), dict(pairs[0])]
    rejected = assess_deepsets_readiness(
        pair_rows=duplicate_pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=7751,
        architecture=locked_architecture,
        maximum_pool_parity_error=0.0,
        cache_finite=True,
    )
    assert not rejected["deepsets_patch_distribution_ready"]
    assert "complete_pair_coverage" in rejected["failed_checks"]

    pairs, folds, global_folds = _passing_rows()
    pairs[0]["pair"] = "wrong"
    folds[0]["pair"] = "wrong"
    rejected = assess_deepsets_readiness(
        pair_rows=pairs,
        fold_metric_rows=folds,
        global_fold_rows=global_folds,
        train_samples=EXPECTED_TRAIN_SAMPLES,
        source_groups=7751,
        architecture=locked_architecture,
        maximum_pool_parity_error=0.0,
        cache_finite=True,
    )
    assert not rejected["deepsets_patch_distribution_ready"]
    assert "pair_labels_consistent" in rejected["failed_checks"]
    assert "complete_pair_fold_coverage" in rejected["failed_checks"]


def test_cli_and_launcher_have_no_implicit_readiness_run() -> None:
    args = _parse_args(
        [
            "--data",
            "train.yaml",
            "--checkpoint",
            "best.pt",
            "--output-dir",
            "run",
        ]
    )
    assert not args.preflight_only
    assert not hasattr(args, "validation")
    assert not hasattr(args, "test")
    assert not hasattr(args, "full_train")

    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_pretrained_classf_b6_deepsets_mil_readiness.ps1"
    ).read_text(encoding="utf-8")
    assert "[switch]$Run" in launcher
    assert "if (-not $Run)" in launcher
    assert '$Arguments += "--preflight-only"' in launcher
    assert "validation/test dataset" in launcher
