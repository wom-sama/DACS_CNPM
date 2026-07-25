from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageEnhance
import pytest
import torch

import trkh.tools.audit_cross_colour_ratio_surface_a0 as audit
from trkh.tools.cross_colour_ratio_surface_a0_engine import (
    EPOCHS,
    VIEW_SIZE,
    array_sha256,
    build_dephase_offsets,
    build_epoch_orders,
    map_targets_to_head_indices,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CACHE = (
    REPOSITORY
    / "runs"
    / "audit_cross_colour_ratio_surface_a0_materialized_20260725"
)


def _lock() -> dict[str, object]:
    return json.loads(audit.LOCK_PATH.read_text(encoding="utf-8"))


def test_runner_does_not_import_broad_dataset_or_trainer() -> None:
    tree = ast.parse(audit.RUNNER_PATH.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert "trkh.data.dataset" not in imports
    assert not any("train" in name and name.startswith("trkh.") for name in imports)


def test_locked_cohort_folds_and_epoch_orders_replay() -> None:
    lock = _lock()
    cohort = audit._load_cohort_arrays(
        CACHE / audit.COHORT_ARRAYS_NAME,
        lock,
    )
    assert cohort.sample_indices.shape == (audit.ROWS,)
    assert len(set(cohort.source_stems.tolist())) == 735
    for fold in range(audit.FOLDS):
        partitions = audit._fold_indices(fold, cohort, lock)
        assert sum(len(values) for values in partitions.values()) == audit.ROWS
        assert not set(partitions["fit"]).intersection(partitions["held"])
        for role in (
            "colour_ratio_control",
            "cross_colour_ratio_candidate",
            "cross_colour_ratio_seed_repeat",
            "cross_colour_ratio_spatial_dephased_control",
        ):
            orders = audit._verify_epoch_orders(
                role=role,
                fold=fold,
                fit_indices=partitions["fit"],
                lock=lock,
            )
            assert len(orders) == EPOCHS


def test_dephase_offsets_match_the_prospective_lock() -> None:
    lock = _lock()
    cohort = audit._load_cohort_arrays(
        CACHE / audit.COHORT_ARRAYS_NAME,
        lock,
    )
    offsets = build_dephase_offsets(cohort.sample_indices)
    assert offsets.shape == (audit.ROWS, 6, 2)
    assert array_sha256(offsets) == lock["descriptor"]["dephase_offsets_sha256"]
    assert np.all(offsets >= 1)
    assert np.all(offsets < VIEW_SIZE)


def test_condition_srgb_matches_pillow_brightness_then_contrast() -> None:
    rng = np.random.default_rng(20260725)
    values = rng.integers(
        0,
        256,
        size=(2, 3, 17, 19),
        dtype=np.uint8,
    )
    actual = audit._condition_srgb(
        values,
        brightness=0.70,
        contrast=0.90,
    )
    expected_rows = []
    for row in values:
        image = Image.fromarray(np.transpose(row, (1, 2, 0)))
        image = ImageEnhance.Brightness(image).enhance(0.70)
        image = ImageEnhance.Contrast(image).enhance(0.90)
        expected_rows.append(np.transpose(np.asarray(image), (2, 0, 1)))
    np.testing.assert_array_equal(actual, np.stack(expected_rows))
    np.testing.assert_array_equal(
        audit._condition_srgb(
            values,
            brightness=1.0,
            contrast=1.0,
        ),
        values,
    )


def test_fit_ledger_blocks_undeclared_split_and_writes(
    tmp_path: Path,
) -> None:
    train = tmp_path / "dataset" / "train" / "allowed.bin"
    validation = tmp_path / "dataset" / "validation" / "blocked.bin"
    run = tmp_path / "runs" / "unknown.bin"
    for path in (train, validation, run):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    roots = {
        "dataset": tmp_path / "dataset",
        "runs": tmp_path / "runs",
        "external": tmp_path / "external",
    }
    with audit.FitDataAccessLedger(
        allowed_paths=(train,),
        domain_roots=roots,
    ) as ledger:
        assert train.read_bytes() == b"x"
        with pytest.raises(PermissionError, match="forbidden_split"):
            validation.read_bytes()
        with pytest.raises(PermissionError, match="undeclared_input"):
            run.read_bytes()
        with pytest.raises(PermissionError, match="write_to_data_domain"):
            train.write_bytes(b"y")
    snapshot = ledger.snapshot()
    assert snapshot["hook_installed"]
    assert snapshot["hook_probe_seen"]
    assert snapshot["blocked_attempt_count"] == 3
    assert snapshot["validation_open_count"] >= 1
    assert snapshot["test_open_count"] == 0


def test_same_weight_action_reuses_clean_threshold() -> None:
    output = audit._empty_role_output()
    held = np.array([0, 1], dtype=np.int64)
    cohort = audit.CohortArrays(
        sample_indices=np.array([3, 7], dtype=np.int64),
        targets=np.array([1, 0], dtype=np.int64),
        folds=np.array([0, 0], dtype=np.int64),
        source_stems=np.array(["a", "b"]),
        keeper_probabilities=np.array(
            [
                [0.1, 0.7, 0.1, 0.05, 0.05],
                [0.2, 0.6, 0.1, 0.05, 0.05],
            ],
            dtype=np.float32,
        ),
        model_boxes=np.zeros((2, 4), dtype=np.float32),
        crop_boxes=np.zeros((2, 4), dtype=np.float32),
        image_paths=np.array(["a", "b"]),
        label_paths=np.array(["a", "b"]),
    )
    result = {
        "logits": np.zeros((2, 4), dtype=np.float32),
        "scores": np.array([0.9, 0.2], dtype=np.float64),
        "pair_maps": np.zeros((2, 3, 24, 24), dtype=np.float32),
        "evidence_reliability": np.ones(
            (2, 1, 24, 24),
            dtype=np.float32,
        ),
    }
    audit._place_same_weight_fold_output(
        output=output,
        fold=0,
        held=held,
        held_result=result,
        threshold=0.5,
        cohort=cohort,
    )
    assert output["thresholds"][0] == 0.5
    assert output["suppressed"][held].tolist() == [False, True]
    assert output["predictions"][held].tolist() == [1, 0]


def test_small_cpu_head_fit_has_finite_gradients_and_updates() -> None:
    generator = torch.Generator().manual_seed(71)
    rows = 8
    descriptors = torch.randn(
        rows,
        6,
        VIEW_SIZE,
        VIEW_SIZE,
        generator=generator,
    )
    reliability = torch.ones(rows, 1, VIEW_SIZE, VIEW_SIZE)
    targets = map_targets_to_head_indices(
        np.array([0, 1, 2, 4, 0, 1, 2, 4], dtype=np.int64)
    )
    offsets = torch.from_numpy(
        build_dephase_offsets(np.arange(rows, dtype=np.int64))
    )
    orders = build_epoch_orders(
        np.arange(rows, dtype=np.int64),
        seed=20260725,
    )
    model, state, record = audit._train_head(
        role="cross_colour_ratio_candidate",
        fold=0,
        descriptors=descriptors,
        reliability=reliability,
        head_targets=targets,
        offsets=offsets,
        orders=orders,
        device=torch.device("cpu"),
        num_workers=0,
        require_effective_workers=None,
    )
    assert record["updates"] == EPOCHS
    assert record["first_gradient"]["all_present"]
    assert record["first_gradient"]["all_finite"]
    assert record["all_parameters_changed"]
    assert record["optimizer_state_fp32"]
    assert all(np.isfinite(value).all() for value in state.values())
    del model


def test_npz_replay_comparison_enforces_tolerance(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    audit._write_npz(
        first,
        {
            "float": np.array([1.0, 2.0], dtype=np.float32),
            "integer": np.array([1, 2], dtype=np.int64),
        },
    )
    audit._write_npz(
        second,
        {
            "float": np.array(
                [1.0 + 5e-8, 2.0],
                dtype=np.float32,
            ),
            "integer": np.array([1, 2], dtype=np.int64),
        },
    )
    result = audit._compare_npz_artifacts(
        first,
        second,
        tolerance=1e-7,
    )
    assert result["passed"]
    audit._write_npz(
        second,
        {
            "float": np.array([1.0 + 5e-5, 2.0], dtype=np.float32),
            "integer": np.array([1, 2], dtype=np.int64),
        },
    )
    with pytest.raises(RuntimeError, match="replay array differs"):
        audit._compare_npz_artifacts(
            first,
            second,
            tolerance=1e-7,
        )


def test_manifests_reject_unlisted_or_modified_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "payload.json").write_text("{}\n", encoding="utf-8")
    manifest = audit._directory_manifest(
        tmp_path,
        excluded_names=(audit.FORMAL_MANIFEST_NAME,),
    )
    audit._write_json(tmp_path / audit.FORMAL_MANIFEST_NAME, manifest)
    assert audit._verify_manifest(
        tmp_path,
        audit.FORMAL_MANIFEST_NAME,
    )["passed"]
    (tmp_path / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="manifest differs"):
        audit._verify_manifest(
            tmp_path,
            audit.FORMAL_MANIFEST_NAME,
        )


def test_training_record_scrub_removes_only_worker_pids() -> None:
    value = {
        "effective_worker_count": 4,
        "effective_worker_pids": [1, 2, 3, 4],
        "nested": {"effective_worker_pids": [5], "updates": 160},
    }
    assert audit._scrub_training_record(value) == {
        "effective_worker_count": 4,
        "nested": {"updates": 160},
    }


def test_process_snapshot_reports_live_python_child() -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )
    try:
        snapshot = audit._process_snapshot()
        child_pids = {
            int(row["pid"]) for row in snapshot["compute_children"]
        }
        assert child.pid in child_pids
        assert snapshot["compute_child_process_count"] >= 1
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])
    args = audit.parse_args(["--preflight-only"])
    assert args.preflight_only
    assert not args.formal
    assert not args.replay


def test_protected_paths_remain_the_only_expected_untracked_contract() -> None:
    assert audit.PROTECTED_REPOSITORY_UNTRACKED == {
        "BaoCao/",
        "deep-research-report (9).md",
        "deep-research-report (10).md",
    }
    assert os.path.isabs(str(audit.DEFAULT_CACHE))


def test_launcher_parses_and_avoids_native_stderr_pipeline() -> None:
    script = audit.LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "& $Python @RunnerArgs" in script
    assert "$LASTEXITCODE" in script
    assert "2>&1" not in script
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{audit.LAUNCHER_PATH}', [ref]$tokens, [ref]$errors) | "
        "Out-Null; if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_implementation_note_keeps_downstream_work_closed() -> None:
    text = audit.IMPLEMENTATION_NOTE_PATH.read_text(encoding="utf-8")
    assert "does not edit the image trainer" in text
    assert "negative-but-reusable" in text
    assert "current-best command updates" in text
    assert "different process" in text
