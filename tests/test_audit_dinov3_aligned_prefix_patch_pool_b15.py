from __future__ import annotations

import inspect
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import trkh.tools.audit_dinov3_aligned_prefix_patch_pool_b15 as b15


def _tokens(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    return torch.randn(
        batch,
        b15.TOKEN_COUNT,
        b15.DESCRIPTOR_DIM,
        generator=generator,
    )


def _valid_operator_payload() -> dict[str, object]:
    latency = {
        "uniform": {
            "median_ms": 10.0,
            "p95_ms": 10.0,
            "trial_mean_ms": [10.0] * b15.ORT_TRIALS,
        },
        "aligned": {
            "median_ms": 10.2,
            "p95_ms": 10.4,
            "trial_mean_ms": [10.2] * b15.ORT_TRIALS,
        },
    }
    export = {
        "bytes": 10,
        "sha256": "a" * 64,
        "operator_domains": [""],
        "operators": ["Add"],
        "export_signature": "single_tensor_forward_wrapper",
    }
    return {
        "settings": {
            "runtime": "onnxruntime_cpu",
            "threads": b15.ORT_THREADS,
            "batch": 1,
            "warmups": b15.ORT_WARMUPS,
            "trials": b15.ORT_TRIALS,
            "iterations_per_trial": b15.ORT_ITERATIONS,
            "input": [1, 3, b15.DINO_IMAGE_SIZE, b15.DINO_IMAGE_SIZE],
        },
        "model_contract": {
            "runtime_class": "timm.models.eva.Eva",
            "features": b15.DESCRIPTOR_DIM,
            "prefix_tokens": b15.DINO_PREFIX_TOKENS,
            "grid": [16, 16],
            "weight_bytes": b15.EXPECTED_DINO_WEIGHT_BYTES,
            "weight_sha256": b15.DINO_WEIGHT_SHA256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "parameters": {
            "operator": 0,
            "operator_buffers": 0,
            "uniform_wrapper": 21_000_000,
            "aligned_wrapper": 21_000_000,
        },
        "exports": {"uniform": dict(export), "aligned": dict(export)},
        "parity": {
            "uniform": {"max_abs": 1e-6, "argmax_equal": True},
            "aligned": {"max_abs": 1e-6, "argmax_equal": True},
        },
        "latency": latency,
        "ratios": {"median": 1.02, "p95": 1.04},
        "checks": {name: True for name in b15.MOBILE_OPERATOR_CHECK_NAMES},
        "passed": True,
    }


def _valid_preflight_structure() -> dict[str, object]:
    checks = b15._preflight_checks(
        git={"branch": b15.EXPECTED_BRANCH, "tracked_worktree_clean": True},
        synthetic_passed=True,
        operator_passed=True,
    )
    return {
        "schema_version": 1,
        "protocol_id": b15.PROTOCOL_ID,
        "mode": "synthetic_preflight_no_dataset",
        "focused_tests": {
            "command": b15._focused_test_command(),
            "returncode": 0,
            "output": "12 passed in 1.0s",
        },
        "synthetic": {
            "formal_arrays_or_labels_read": False,
            "checks": {name: True for name in b15.SYNTHETIC_CHECK_NAMES},
        },
        "operator": _valid_operator_payload(),
        "operator_checks": {name: True for name in b15.OPERATOR_CHECK_NAMES},
        "permissions": {
            "train_descriptor_read": False,
            "validation_not_constructed": True,
            "test_not_constructed": True,
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
        },
        "checks": checks,
        "passed": True,
    }


def test_descriptor_formula_identities_and_zero_parameter_contract() -> None:
    tokens = _tokens()
    uniform, prefix_only, aligned, weights = b15._descriptor_details(tokens)
    assert torch.equal(
        uniform, tokens[:, b15.DINO_PREFIX_TOKENS :].mean(dim=1)
    )
    assert torch.equal(
        prefix_only, tokens[:, : b15.DINO_PREFIX_TOKENS].mean(dim=1)
    )
    assert tuple(aligned.shape) == (3, b15.DESCRIPTOR_DIM)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0)
    operator = b15.AlignedPrefixPatchPool()
    assert sum(parameter.numel() for parameter in operator.parameters()) == 0
    assert sum(buffer.numel() for buffer in operator.buffers()) == 0

    zero_query = tokens.clone()
    zero_query[:, : b15.DINO_PREFIX_TOKENS] = uniform[:, None]
    zero_uniform, _prefix, zero_aligned, zero_weights = b15._descriptor_details(
        zero_query
    )
    assert torch.equal(
        zero_weights,
        torch.full_like(zero_weights, 1.0 / b15.PATCH_TOKENS),
    )
    assert float(torch.max(torch.abs(zero_aligned - zero_uniform))) <= 1e-6


def test_aligned_is_prefix_order_and_joint_patch_permutation_invariant() -> None:
    tokens = _tokens()
    _uniform, _prefix, aligned, weights = b15._descriptor_details(tokens)
    prefix = tokens[:, : b15.DINO_PREFIX_TOKENS]
    patches = tokens[:, b15.DINO_PREFIX_TOKENS :]
    prefix_permutation = torch.tensor([3, 0, 4, 1, 2])
    _u, _p, reordered, reordered_weights = b15._descriptor_details(
        tokens, prefix_tokens=prefix[:, prefix_permutation]
    )
    assert torch.allclose(aligned, reordered, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weights, reordered_weights, atol=1e-6, rtol=1e-6)

    patch_permutation = torch.arange(b15.PATCH_TOKENS - 1, -1, -1)
    permuted_tokens = torch.cat((prefix, patches[:, patch_permutation]), dim=1)
    _u, _p, patch_reordered, patch_weights = b15._descriptor_details(
        permuted_tokens
    )
    assert torch.allclose(aligned, patch_reordered, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        weights[:, patch_permutation], patch_weights, atol=1e-6, rtol=1e-6
    )


def test_deranged_uses_donor_center_and_rejects_current_center_semantics() -> None:
    tokens = _tokens()
    uniform = tokens[:, b15.DINO_PREFIX_TOKENS :].mean(dim=1)
    prefixes = tokens[:, : b15.DINO_PREFIX_TOKENS]
    donor_prefix = torch.roll(prefixes, shifts=1, dims=0)
    donor_center = torch.roll(uniform, shifts=1, dims=0)
    _u, _p, correct, _w = b15._descriptor_details(
        tokens,
        prefix_tokens=donor_prefix,
        prefix_center=donor_center,
    )
    offset = torch.randn_like(donor_center)
    _u, _p, offset_result, _w = b15._descriptor_details(
        tokens,
        prefix_tokens=donor_prefix + offset[:, None],
        prefix_center=donor_center + offset,
    )
    _u, _p, forbidden, _w = b15._descriptor_details(
        tokens,
        prefix_tokens=donor_prefix,
        prefix_center=uniform,
    )
    reconstructed_prefix = uniform[:, None] + (
        donor_prefix - donor_center[:, None]
    )
    _u, _p, reconstructed, _w = b15._descriptor_details(
        tokens,
        prefix_tokens=reconstructed_prefix,
        prefix_center=uniform,
    )
    assert torch.allclose(correct, offset_result, atol=1e-6, rtol=1e-6)
    assert torch.allclose(correct, reconstructed, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(correct, forbidden, atol=1e-6, rtol=1e-6)


def test_synthetic_contract_reports_entropy_and_no_formal_labels() -> None:
    report = b15._synthetic_label_free_contract()
    assert report["formal_arrays_or_labels_read"] is False
    assert all(report["checks"].values())
    entropy = np.asarray(report["attention_entropy"])
    effective = np.asarray(report["effective_patch_count"])
    assert np.all((entropy >= 0) & (entropy <= math.log(b15.PATCH_TOKENS) + 1e-6))
    assert np.all((effective >= 1) & (effective <= b15.PATCH_TOKENS + 1e-5))


def test_union_group_donor_map_is_exact_repeatable_and_label_free() -> None:
    folds = np.repeat(np.arange(b15.FOLDS, dtype=np.int64), 6)
    groups = np.concatenate(
        [np.repeat(np.arange(3, dtype=np.int64) + fold * 10, 2) for fold in range(5)]
    )
    first, report = b15.build_fold_contained_donor_map(folds, groups)
    second, repeated = b15.build_fold_contained_donor_map(folds, groups)
    assert np.array_equal(first, second)
    assert report["donor_int64_sha256"] == repeated["donor_int64_sha256"]
    assert not np.any(first == np.arange(first.size))
    assert np.all(folds[first] == folds)
    assert np.all(groups[first] != groups)
    assert all(
        set(first[folds == fold]) == set(np.flatnonzero(folds == fold))
        for fold in range(b15.FOLDS)
    )
    assert "labels" not in inspect.signature(
        b15.build_fold_contained_donor_map
    ).parameters


def test_donor_map_fails_when_union_group_disjoint_map_is_impossible() -> None:
    folds = np.repeat(np.arange(b15.FOLDS, dtype=np.int64), 5)
    groups = np.concatenate(
        [np.array([fold, fold, fold, fold + 10, fold + 20]) for fold in range(5)]
    )
    with pytest.raises(ValueError, match="no union-group-disjoint derangement"):
        b15.build_fold_contained_donor_map(folds, groups)


def test_staged_archive_readers_never_request_forbidden_arrays() -> None:
    class GuardedArchive:
        def __init__(self) -> None:
            self.accessed: list[str] = []
            self.values = {
                "folds": np.arange(4, dtype=np.int64),
                "groups": np.arange(4, dtype=np.int64),
                "labels": np.arange(4, dtype=np.int64),
                "dino_scores": np.zeros((4, b15.CLASSES), dtype=np.float64),
            }

        def __getitem__(self, key: str) -> np.ndarray:
            self.accessed.append(key)
            if key in {
                "efficientvim_scores",
                "iformer_scores",
                "b9_logits",
                "validation_scores",
                "test_scores",
            }:
                raise AssertionError(f"forbidden B15 array requested: {key}")
            return self.values[key]

    archive = GuardedArchive()
    assert set(b15._read_b13_fold_group_arrays(archive)) == {"folds", "groups"}
    assert archive.accessed == ["folds", "groups"]
    archive.accessed.clear()
    assert set(b15._read_b13_label_score_arrays(archive)) == {
        "labels",
        "dino_scores",
    }
    assert archive.accessed == ["labels", "dino_scores"]


def test_path_only_guard_rejects_val_test_traversal_and_absolute_paths(
    tmp_path: Path,
) -> None:
    opaque_class = "opaque_class_directory"
    valid = [f"train/{opaque_class}/image.jpg"]
    resolved = b15._resolve_train_paths_without_labels(
        valid, tmp_path, require_files=False
    )
    assert resolved[0].is_relative_to((tmp_path / "train").resolve())
    invalid = (
        f"val/{opaque_class}/image.jpg",
        f"test/{opaque_class}/image.jpg",
        f"train/{opaque_class}/../other.jpg",
        "../train/image.jpg",
        r"C:\outside\image.jpg",
    )
    for raw in invalid:
        with pytest.raises(ValueError, match="non-TRAIN path"):
            b15._resolve_train_paths_without_labels(
                [raw], tmp_path, require_files=False
            )
    assert "labels" not in inspect.signature(b15._TrainPathDataset).parameters


def test_uniform_npy_contract_is_in_memory_and_only_three_descriptors_are_retained() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    contract = b15._in_memory_npy_contract(values)
    import io
    import hashlib

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    assert contract == {
        "sha256": hashlib.sha256(buffer.getbuffer()).hexdigest(),
        "bytes": len(buffer.getbuffer()),
        "retained_on_disk": False,
    }
    assert set(b15.RETAINED_DESCRIPTOR_FILENAMES) == {
        "prefix_only",
        "aligned",
        "deranged",
    }
    assert tuple(b15.RETAINED_DESCRIPTOR_FILENAMES.values()) == (
        "train_prefix_only_f32.npy",
        "train_aligned_prefix_patch_pool_f32.npy",
        "train_deranged_prefix_patch_pool_f32.npy",
    )
    assert "uniform" not in b15.RETAINED_DESCRIPTOR_FILENAMES


def test_fixed_point_json_projection_matches_exact_lf_encoded_total() -> None:
    payload: dict[str, object] = {"final_total_bytes": 0, "value": [1, 2, 3]}

    def set_total(target: dict[str, object], total: int) -> None:
        target["final_total_bytes"] = total

    encoded, total = b15._fixed_point_json_bytes(
        payload,
        base_bytes=123,
        set_final_total=set_total,
    )
    assert b"\r\n" not in encoded
    assert total == 123 + len(encoded)
    assert payload["final_total_bytes"] == total


def test_timed_fixed_point_uses_post_serialization_clock_and_closes_late_run() -> None:
    payload: dict[str, object] = {
        "wall_seconds": 0.0,
        "final_total_bytes": 0,
        "wall_check": True,
    }
    clock_values = iter((899.2, 900.1, 900.1, 900.1))

    def set_total(target: dict[str, object], total: int) -> None:
        target["final_total_bytes"] = total
        target["wall_check"] = float(target["wall_seconds"]) <= 900.0

    def set_wall(target: dict[str, object], seconds: float) -> None:
        target["wall_seconds"] = seconds

    encoded, total, wall_upper = b15._timed_fixed_point_json_bytes(
        payload,
        base_bytes=321,
        started=0.0,
        set_final_total=set_total,
        set_wall_seconds=set_wall,
        clock=lambda: next(clock_values),
    )
    assert wall_upper == 901.0
    assert payload["wall_seconds"] == 901.0
    assert payload["wall_check"] is False
    assert total == 321 + len(encoded)


def test_formal_resets_cuda_peak_before_source_and_artifact_validation() -> None:
    source = inspect.getsource(b15._run_formal)
    reset = source.index("torch.cuda.reset_peak_memory_stats(device)")
    data_yaml = source.index("data_yaml = args.data_yaml")
    preflight = source.index("accepted = _validate_preflight")
    anchors = source.index("anchors = _verify_formal_evidence_anchors")
    assert reset < data_yaml < preflight < anchors
    gate_write = source.index(
        '_atomic_json(partial / "pre_metric_resource_gate.json", pre_metric_gate)'
    )
    post_gate = source.index("post_gate_elapsed = time.perf_counter() - started")
    sentinel = source.index(
        'partial / "stage_metric_construction_started.json"'
    )
    assert gate_write < post_gate < sentinel


def test_bootstrap_contract_excludes_deranged_scores() -> None:
    rows = 10
    base = {
        name: np.zeros((rows, b15.CLASSES), dtype=np.float64)
        for name in ("uniform", "prefix_only", "aligned", "deranged")
    }
    with pytest.raises(ValueError, match="arms/count/seed changed"):
        b15.paired_component_bootstrap_b15(
            labels=np.arange(rows) % b15.CLASSES,
            folds=np.arange(rows) % b15.FOLDS,
            groups=np.arange(rows),
            scores=base,
        )


def _gate_metrics(deranged_pair2: float = 0.81) -> dict[str, dict[str, object]]:
    matrix_uniform = [[10, 0, 0, 0, 0] for _ in range(b15.CLASSES)]
    matrix_aligned = deepcopy(matrix_uniform)
    matrix_uniform[1][2] = 5
    matrix_aligned[1][2] = 5
    matrix_uniform[2][1] = 20
    matrix_aligned[2][1] = 18

    def arm(
        pair2: float,
        mean_pair: float,
        c1_f1: float,
        macro: float,
        recall: float,
        matrix: list[list[int]],
    ) -> dict[str, object]:
        return {
            "pairs": {"0": {"auroc": mean_pair}, "2": {"auroc": pair2}, "4": {"auroc": mean_pair}},
            "mean_pair_auroc": mean_pair,
            "class1_f1": c1_f1,
            "macro_f1": macro,
            "class1_recall": recall,
            "confusion_matrix": matrix,
        }

    return {
        "uniform": arm(0.80, 0.80, 0.50, 0.80, 0.80, matrix_uniform),
        "prefix_only": arm(0.79, 0.79, 0.49, 0.79, 0.79, matrix_uniform),
        "aligned": arm(0.82, 0.81, 0.52, 0.81, 0.80, matrix_aligned),
        "deranged": arm(deranged_pair2, 0.80, 0.50, 0.80, 0.80, matrix_uniform),
    }


def test_gate_uses_point_only_deranged_check_and_no_deranged_interval() -> None:
    intervals = {
        comparator: {
            "pair2_auroc_delta": {"lower": 0.001, "upper": 0.03},
            "mean_pair_auroc_delta": {"lower": 0.001, "upper": 0.03},
            "class1_f1_delta": {"lower": 0.001, "upper": 0.03},
            "macro_f1_delta": {"lower": -0.001, "upper": 0.03},
        }
        for comparator in ("uniform", "prefix_only")
    }
    gate = b15.assess_b15_gate(
        metrics=_gate_metrics(),
        bootstrap={"intervals": intervals, "all_gate_replicates_finite": True},
        uniform_parity={"passed": True},
        fold_wins={"pair2_wins": 4, "all_pair_fold_wins": 10},
        readouts_converged=True,
        integrity_complete=True,
    )
    assert gate["signal_gate_passed"]
    assert "pair2_point_gt_deranged" in gate["checks"]
    assert "deranged" not in intervals

    failed = b15.assess_b15_gate(
        metrics=_gate_metrics(deranged_pair2=0.83),
        bootstrap={"intervals": intervals, "all_gate_replicates_finite": True},
        uniform_parity={"passed": True},
        fold_wins={"pair2_wins": 4, "all_pair_fold_wins": 10},
        readouts_converged=True,
        integrity_complete=True,
    )
    assert failed["checks"]["pair2_point_gt_deranged"] is False


def test_preflight_structure_rejects_tampering_behind_passed_true() -> None:
    valid = _valid_preflight_structure()
    assert all(b15._preflight_payload_structure_checks(valid).values())

    tampered = deepcopy(valid)
    tampered["operator"]["ratios"]["p95"] = 1.06  # type: ignore[index]
    assert tampered["passed"] is True
    assert not b15._preflight_payload_structure_checks(tampered)[
        "operator_contract_exact"
    ]

    forbidden = deepcopy(valid)
    forbidden["synthetic"]["formal_arrays_or_labels_read"] = True  # type: ignore[index]
    assert not b15._preflight_payload_structure_checks(forbidden)["synthetic_exact"]


def test_preflight_structure_rejects_empty_missing_false_extra_and_cross_mismatch_checks() -> None:
    for nested_arm in ("synthetic", "operator"):
        empty = deepcopy(_valid_preflight_structure())
        empty[nested_arm]["checks"] = {}  # type: ignore[index]
        checks = b15._preflight_payload_structure_checks(empty)
        assert not all(checks.values())

        missing = deepcopy(_valid_preflight_structure())
        missing_checks = missing[nested_arm]["checks"]  # type: ignore[index]
        missing_checks.pop(next(iter(missing_checks)))
        checks = b15._preflight_payload_structure_checks(missing)
        assert not all(checks.values())

        false_value = deepcopy(_valid_preflight_structure())
        false_checks = false_value[nested_arm]["checks"]  # type: ignore[index]
        false_checks[next(iter(false_checks))] = False
        checks = b15._preflight_payload_structure_checks(false_value)
        assert not all(checks.values())

        extra = deepcopy(_valid_preflight_structure())
        extra[nested_arm]["checks"]["forged_extra_check"] = True  # type: ignore[index]
        checks = b15._preflight_payload_structure_checks(extra)
        assert not all(checks.values())

    cross_mismatch = deepcopy(_valid_preflight_structure())
    cross_mismatch["operator_checks"]["uniform_identity_exact"] = False  # type: ignore[index]
    checks = b15._preflight_payload_structure_checks(cross_mismatch)
    assert checks["synthetic_exact"] is True
    assert checks["operator_contract_exact"] is True
    assert checks["operator_checks_exact"] is False


def test_output_scope_and_quarantine_are_owned_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b15, "_repository_root", lambda: tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    output = runs / f"{b15.FORMAL_OUTPUT_PREFIX}test_r1"
    assert b15._validated_b15_output_path(output) == output.resolve()
    with pytest.raises(ValueError, match="fresh direct child"):
        b15._validated_b15_output_path(runs / "nested" / output.name)
    with pytest.raises(ValueError, match="fresh direct child"):
        b15._validated_b15_output_path(runs / "unrelated")

    partial = output.with_name(output.name + ".partial")
    partial.mkdir()
    (partial / "run_owner.json").write_text(
        json.dumps({"protocol_id": b15.PROTOCOL_ID}), encoding="utf-8"
    )
    (partial / "stage_metric_construction_started.json").write_text(
        "{}", encoding="utf-8"
    )
    b15._quarantine_partial(
        SimpleNamespace(output_dir=output), RuntimeError("after metric boundary")
    )
    manifest = json.loads(
        (partial / "failure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["metric_boundary_crossed"] is True
    assert manifest["retry_allowed"] is False


def test_locked_timm_source_contract_uses_actual_eva_implementation() -> None:
    contract = b15._dino_source_contract()
    assert set(contract["sha256"]) == {"eva.py", "_factory.py"}
    assert contract["sha256"] == b15.EXPECTED_TIMM_SOURCE_HASHES
