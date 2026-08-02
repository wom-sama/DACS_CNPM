from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from timm.models.eva import Eva

from trkh.tools import precheck_dinov3_xcnorm_a1_sourcefold as a1_preflight
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (
    ADAPTER_BATCH_SIZE,
    EXPECTED_ASSIGNMENT_INT64_SHA256,
    EXPECTED_PATH_FOLD_SHA256,
    FOLDS,
)
from trkh.tools.screen_dinov3_xcnorm_a1_sourcefold import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _cache_expected,
    _promote_memmap_cache,
    assess_readiness,
    fold_stratified_union_group_bootstrap,
    metrics_all_fp,
    run_screen,
    validate_or_write_assignment_csv,
    validate_preflight_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_hashes() -> dict[str, str]:
    return {
        "selected_full_state_sha256": a1_preflight.EXPECTED_SELECTED_STATE_SHA256,
        "final_block_state_sha256": a1_preflight.EXPECTED_FINAL_BLOCK_STATE_SHA256,
        "gamma1_state_sha256": a1_preflight.EXPECTED_GAMMA1_STATE_SHA256,
        "gamma2_state_sha256": a1_preflight.EXPECTED_GAMMA2_STATE_SHA256,
        "final_norm_head_state_sha256": a1_preflight.EXPECTED_FINAL_NORM_HEAD_STATE_SHA256,
    }


def _source_hashes() -> dict[str, str]:
    tool = Path(a1_preflight.__file__).resolve()
    return {
        "tool_sha256": _sha256(tool),
        "a1_model_sha256": _sha256(
            tool.parents[1] / "models" / "dinov3_xcnorm_pair_adapter_a1.py"
        ),
        "a0_operator_model_sha256": _sha256(
            tool.parents[1] / "models" / "dinov3_xcnorm_pair_adapter_a0.py"
        ),
        "a0_shared_protocol_sha256": _sha256(
            tool.with_name("precheck_dinov3_xcnorm_a0_sourcefold.py")
        ),
        "model_builder_sha256": _sha256(tool.parents[1] / "models" / "model.py"),
        "dataset_pipeline_sha256": _sha256(tool.parents[1] / "data" / "dataset.py"),
        "probe_loader_sha256": _sha256(
            tool.with_name("probe_embedding_prototypes.py")
        ),
        "timm_eva_source_sha256": _sha256(
            Path(inspect.getsourcefile(Eva) or "").resolve()
        ),
    }


def _preflight_payload() -> dict[str, object]:
    return {
        "protocol_id": a1_preflight.PROTOCOL_ID,
        "preflight_passed": True,
        "smoke": {
            "actual_b9_train_smoke_passed": True,
            "parity": {"branch_off_logit_max_abs_error": 0.0},
            "smoke_paths_sha256": a1_preflight.EXPECTED_SMOKE_PATHS_SHA256,
            "state_hashes_before": _state_hashes(),
            "state_hashes_after": _state_hashes(),
            "frozen_model_gradients_present": False,
            "residual_gradient_through_frozen_tail_norm": 0.1,
        },
        "smoke_paths_sha256": a1_preflight.EXPECTED_SMOKE_PATHS_SHA256,
        "canonical_train": {
            "samples": 8_278,
            "assignment_int64_sha256": EXPECTED_ASSIGNMENT_INT64_SHA256,
            "path_fold_sha256": EXPECTED_PATH_FOLD_SHA256,
            "manifest_sha256": a1_preflight.EXPECTED_MANIFEST_SHA256,
            "integrity_manifest_sha256": a1_preflight.EXPECTED_INTEGRITY_MANIFEST_SHA256,
        },
        "future_locked_oof_optimization": {
            "folds": 5,
            "seed": a1_preflight.SEED,
            "epochs": 5,
            "learning_rate": a1_preflight.LR,
            "adapter_batch_size": ADAPTER_BATCH_SIZE,
            "weight_decay": a1_preflight.WEIGHT_DECAY,
            "tempered_sampling_power": a1_preflight.TEMPERED_POWER,
            "status": "not_run_preflight_only",
        },
        "source_files": _source_hashes(),
        "locked_state_hashes": _state_hashes(),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "large_cache_created": False,
        "oof_training_run": False,
        "full_validation_permission": False,
        "test_permission": False,
    }


def test_preflight_artifact_binds_sha_smoke_and_all_eight_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_preflight_payload()), encoding="utf-8")
    payload, digest = validate_preflight_artifact(path, _sha256(path))
    assert payload["preflight_passed"]
    assert digest == _sha256(path)
    assert len(payload["source_files"]) == 8

    tampered = _preflight_payload()
    tampered["source_files"]["dataset_pipeline_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="eight_preflight_sources"):
        validate_preflight_artifact(path, _sha256(path))


def test_memmaps_close_hash_and_promote_atomically(tmp_path: Path) -> None:
    partials = {
        name: tmp_path / f"{name}.npy.partial" for name in ("left", "right")
    }
    finals = {name: tmp_path / f"{name}.npy" for name in partials}
    arrays = {
        name: np.lib.format.open_memmap(
            partials[name], mode="w+", dtype=np.float32, shape=(3,)
        )
        for name in partials
    }
    arrays["left"][:] = [1, 2, 3]
    arrays["right"][:] = [4, 5, 6]
    mappings = {name: array._mmap for name, array in arrays.items()}
    hashes = _promote_memmap_cache(arrays, partials, finals)
    assert all(mapping.closed for mapping in mappings.values())
    assert all(_sha256(finals[name]) == hashes[name] for name in finals)
    np.testing.assert_array_equal(np.load(finals["left"]), [1, 2, 3])


def test_assignment_resume_requires_exact_rows(tmp_path: Path) -> None:
    path = tmp_path / "train_fold_assignments.csv"
    rows = [
        {"relative_path": "train/c0/a.jpg", "label": 0, "union_group": 7, "fold": 2},
        {"relative_path": "train/c1/b.jpg", "label": 1, "union_group": 8, "fold": 3},
    ]
    first = validate_or_write_assignment_csv(path, rows)
    second = validate_or_write_assignment_csv(path, rows)
    assert first == second == _sha256(path)

    changed = [dict(row) for row in rows]
    changed[1]["fold"] = 4
    with pytest.raises(ValueError, match="rows changed"):
        validate_or_write_assignment_csv(path, changed)


def test_cache_identity_separates_path_order_from_paths_file_hash() -> None:
    expected = _cache_expected(
        relative_paths=["train/c0/a.jpg", "train/c1/b.jpg"],
        labels=np.asarray([0, 1], dtype=np.int64),
        preflight_sha256="a" * 64,
        state_hashes=_state_hashes(),
    )
    assert len(expected["path_order_sha256"]) == 64
    assert "paths_file_sha256" not in expected
    assert "paths_sha256" not in expected


def _bootstrap_fixture():
    rng = np.random.default_rng(29)
    labels = np.tile(np.repeat(np.arange(5), 20), FOLDS)
    folds = np.repeat(np.arange(FOLDS), 100)
    groups = np.arange(labels.size, dtype=np.int64)
    control = rng.normal(0.0, 0.8, size=(labels.size, 5))
    candidate = rng.normal(0.0, 0.25, size=(labels.size, 5))
    base = rng.normal(0.0, 0.35, size=(labels.size, 5))
    candidate[np.arange(labels.size), labels] += 2.0
    base[np.arange(labels.size), labels] += 1.6
    control[np.arange(labels.size), labels] += 0.35
    return labels, folds, groups, base, control, candidate


def test_fold_group_bootstrap_is_deterministic_and_hash_bound(
    tmp_path: Path,
) -> None:
    labels, folds, groups, base, control, candidate = _bootstrap_fixture()
    oof = tmp_path / "oof.npz"
    np.savez(oof, labels=labels)
    kwargs = dict(
        labels=labels,
        folds=folds,
        groups=groups,
        base_logits=base,
        control_logits=control,
        candidate_logits=candidate,
        oof_artifact=oof,
        replicates=24,
        seed=41,
    )
    first = fold_stratified_union_group_bootstrap(**kwargs)
    second = fold_stratified_union_group_bootstrap(**kwargs)
    assert first == second
    assert first["replicates"] == 24
    assert first["oof_artifact_sha256"] == _sha256(oof)
    assert len(first["draws_int64_sha256"]) == 64
    assert first["mean_pair_auroc_delta"]["lower"] > 0.0

    crossed = groups.copy()
    crossed[100] = crossed[0]
    with pytest.raises(ValueError, match="crosses held folds"):
        fold_stratified_union_group_bootstrap(**{**kwargs, "groups": crossed})


def _passing_screen() -> tuple[dict[str, object], dict[str, object]]:
    base = {
        "macro_f1": 0.80,
        "class1_f1": 0.70,
        "class1_tp": 100,
        "total_fp_to_class1": 100,
        "fp_to_class1": {"0": 20, "2": 30, "3": 10, "4": 40},
    }
    control = {
        "macro_f1": 0.80,
        "class1_f1": 0.70,
        "class1_tp": 99,
        "total_fp_to_class1": 90,
        "fp_to_class1": {"0": 18, "2": 28, "3": 8, "4": 36},
    }
    candidate = {
        "macro_f1": 0.801,
        "class1_f1": 0.71,
        "class1_tp": 99,
        "total_fp_to_class1": 85,
        "fp_to_class1": {"0": 17, "2": 26, "3": 8, "4": 34},
    }
    digest = "a" * 64
    training_rows = [
        {
            "samples": 128,
            "schedule_int64_sha256": digest,
            "class_counts": [32, 32, 32, 0, 32],
        }
        for _ in range(25)
    ]
    screen = {
        "base": base,
        "control": control,
        "candidate": candidate,
        "preflight_artifact_verified": True,
        "assignment_hashes": {
            "assignment_int64_sha256": EXPECTED_ASSIGNMENT_INT64_SHA256,
            "path_fold_sha256": EXPECTED_PATH_FOLD_SHA256,
        },
        "group_fold_contract": [
            {"fold": fold, "group_overlap": 0} for fold in range(FOLDS)
        ],
        "paired_contracts": [
            {"control_parameters": 3_672, "candidate_parameters": 3_672}
            for _ in range(FOLDS)
        ],
        "training_rows": training_rows,
        "schedule_contract_sha256": digest,
        "pair_auroc_gains": {"0": 0.0003, "2": 0.0003, "4": 0.0003},
        "mean_pair_auroc_gain": 0.0003,
        "fold_wins": 4,
        "telemetry": {
            arm: {
                "injection_ratio": {"p95": 0.039},
                "final_token_perturbation_ratio": {"p95": 0.079},
            }
            for arm in ("control", "candidate")
        },
        "branch_off_max_abs_error": 0.0,
        "frozen_state_hashes_before": _state_hashes(),
        "frozen_state_hashes_after": _state_hashes(),
        "completed_updates": 100,
        "expected_updates": 100,
        "nonfinite_updates": 0,
        "skipped_updates": 0,
        "oof_artifact_sha256": digest,
        "group_vector_int64_sha256": "b" * 64,
        "fold_vector_int64_sha256": "c" * 64,
    }
    bootstrap = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "oof_artifact_sha256": digest,
        "group_vector_int64_sha256": "b" * 64,
        "fold_vector_int64_sha256": "c" * 64,
        "mean_pair_auroc_delta": {"lower": 0.00001, "upper": 0.0005},
        "class1_f1_delta_vs_b9": {"lower": 0.001, "upper": 0.02},
        "macro_f1_delta_vs_b9": {"lower": -0.001, "upper": 0.003},
        "total_fp_rate_delta_vs_control": {"lower": -0.01, "upper": 0.0},
    }
    return screen, bootstrap


def test_preregistered_gates_pass_only_photometric_train_permission() -> None:
    screen, bootstrap = _passing_screen()
    result = assess_readiness(screen, bootstrap)
    assert result["clean_train_gates_passed"]
    assert result["photometric_train_permission"]
    assert not result["full_validation_permission"]
    assert not result["test_permission"]

    screen["candidate"]["fp_to_class1"]["3"] = 9
    failed = assess_readiness(screen, bootstrap)
    assert not failed["clean_train_gates_passed"]
    assert "each_source_fp_no_higher_than_base_and_control" in failed["failed_checks"]


def test_metrics_count_class3_false_positives_into_class1() -> None:
    labels = np.asarray([0, 1, 2, 3, 4])
    logits = np.eye(5, dtype=np.float32)
    logits[3] = [0, 2, 0, 1, 0]
    metrics = metrics_all_fp(labels, logits)
    assert metrics["fp_to_class1"]["3"] == 1
    assert metrics["total_fp_to_class1"] == 1


def test_adapter_batch_lock_fails_before_artifact_or_data_io() -> None:
    with pytest.raises(ValueError, match=f"locked to {ADAPTER_BATCH_SIZE}"):
        run_screen(
            SimpleNamespace(
                adapter_batch_size=64,
                batch_size=1,
                workers=0,
            )
        )
