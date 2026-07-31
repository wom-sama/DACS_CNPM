from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.precheck_dinov3_classconditional_deepsets_train_oof import (
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
    _parse_args,
    _sha256,
    _train_readout,
    _validate_cache_manifest,
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


def test_cache_manifest_uses_distinct_file_and_label_content_hashes(tmp_path: Path) -> None:
    cache_paths = {
        "tokens": tmp_path / "tokens.npy",
        "logits": tmp_path / "logits.npy",
        "labels": tmp_path / "labels.npy",
    }
    for index, path in enumerate(cache_paths.values()):
        path.write_bytes(f"cache-{index}".encode("ascii"))
    labels_content = np.asarray([0, 1, 2, 3, 4], dtype="<i8").tobytes()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": "locked-paths",
        "token_shape": [EXPECTED_TRAIN_SAMPLES, PATCH_TOKENS, TOKEN_WIDTH],
        "token_dtype": "float16",
        "global_logits_shape": [EXPECTED_TRAIN_SAMPLES, GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [EXPECTED_TRAIN_SAMPLES],
        "labels_dtype": "int64",
        "tokens_sha256": _sha256(cache_paths["tokens"]),
        "logits_sha256": _sha256(cache_paths["logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
        "labels_content_sha256": hashlib.sha256(labels_content).hexdigest(),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
    }
    _validate_cache_manifest(
        manifest=manifest,
        cache_paths=cache_paths,
        expected_paths_sha256="locked-paths",
    )
    cache_paths["labels"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        _validate_cache_manifest(
            manifest=manifest,
            cache_paths=cache_paths,
            expected_paths_sha256="locked-paths",
        )


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
