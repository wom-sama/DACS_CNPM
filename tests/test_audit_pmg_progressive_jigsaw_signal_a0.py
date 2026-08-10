from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trkh.tools import audit_pmg_progressive_jigsaw_signal_a0 as pmg


def _probability(primary: int, secondary: int, primary_value: float) -> np.ndarray:
    result = np.full(5, 0.01, dtype=np.float64)
    result[primary] = float(primary_value)
    result[secondary] = 1.0 - float(primary_value) - 0.03
    return result


def _synthetic_gate_inputs():
    targets = []
    folds = []
    keeper = []
    control = []
    candidate = []
    for fold in range(5):
        for _ in range(10):
            targets.append(1)
            folds.append(fold)
            base = _probability(1, 0, 0.78)
            improved = _probability(1, 0, 0.84)
            keeper.append(base)
            control.append(base)
            candidate.append(improved)
        for _ in range(2):
            targets.append(1)
            folds.append(fold)
            base = _probability(0, 1, 0.52)
            improved = _probability(1, 0, 0.62)
            keeper.append(base)
            control.append(base)
            candidate.append(improved)
        for target in pmg.RESTRICTED_NEGATIVE_CLASSES:
            for _ in range(3):
                targets.append(target)
                folds.append(fold)
                base = _probability(1, target, 0.56)
                improved = _probability(target, 1, 0.70)
                keeper.append(base)
                control.append(base)
                candidate.append(improved)
        for target in (0, 2, 3, 4):
            for _ in range(4):
                targets.append(target)
                folds.append(fold)
                base = _probability(target, (target + 1) % 5, 0.82)
                keeper.append(base)
                control.append(base)
                candidate.append(base)
    target_array = np.asarray(targets, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=np.int64)
    keeper_array = np.asarray(keeper, dtype=np.float64)
    control_array = np.asarray(control, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    roles = {
        "clean_control": control_array,
        "pmg_aligned": candidate_array,
        "reverse_placebo": control_array.copy(),
        "deepest_placebo": control_array.copy(),
    }
    stats = {
        role: {
            "effective_rank_512_row_sketch": 64.0,
            "blocks": [
                {"block": index, "constant": False} for index in range(4)
            ],
        }
        for role in pmg.ROLE_NAMES
    }
    return target_array, fold_array, keeper_array, roles, stats


def test_permutation_bank_is_deterministic_bijective_and_sample_specific() -> None:
    first = pmg.build_permutation_bank([7, 19], grid=8, seed=20260714)
    second = pmg.build_permutation_bank([7, 19], grid=8, seed=20260714)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    expected = torch.arange(64).expand(2, -1)
    assert torch.equal(torch.sort(first, dim=1).values, expected)


def test_patch_permutation_roundtrip_preserves_image_and_mask() -> None:
    image = torch.arange(2 * 3 * 32 * 32, dtype=torch.int64).reshape(2, 3, 32, 32)
    mask = (image[:, :1] % 5) != 0
    permutation = pmg.build_permutation_bank([2, 4], grid=4, seed=20260714)
    inverse = torch.argsort(permutation, dim=1)
    shuffled_image = pmg.apply_patch_permutation(image, permutation, grid=4)
    shuffled_mask = pmg.apply_patch_permutation(mask, permutation, grid=4)
    assert torch.equal(
        pmg.apply_patch_permutation(shuffled_image, inverse, grid=4), image
    )
    assert torch.equal(
        pmg.apply_patch_permutation(shuffled_mask, inverse, grid=4), mask
    )


def test_stage_descriptor_excludes_prefix_and_matches_formula() -> None:
    tokens = torch.tensor(
        [
            [
                [100.0, 100.0],
                [-100.0, -100.0],
                [1.0, 3.0],
                [5.0, 7.0],
            ]
        ]
    )
    observed = pmg.stage_local_descriptor(tokens, nn.Identity(), prefix_count=2)
    patches = tokens[:, 2:]
    expected = torch.nn.functional.normalize(
        0.5 * patches.mean(dim=1) + 0.5 * patches.amax(dim=1), dim=1
    )
    assert torch.allclose(observed, expected)


def test_role_assembly_uses_locked_stage_grid_alignment() -> None:
    batch = 2
    clean = {
        layer: torch.full((batch, 256), float(layer)) for layer in pmg.STAGE_LAYERS
    }
    jigsaw = {
        grid: {
            layer: torch.full((batch, 256), float(grid * 100 + layer))
            for layer in pmg.STAGE_LAYERS
        }
        for grid in pmg.JIGSAW_GRIDS
    }
    global_feature = torch.full((batch, 256), 999.0)
    roles = pmg.assemble_role_descriptors(clean, jigsaw, global_feature)
    assert tuple(roles["pmg_aligned"].shape) == (2, 1024)
    assert torch.equal(roles["pmg_aligned"][:, :256], jigsaw[8][2])
    assert torch.equal(roles["pmg_aligned"][:, 256:512], jigsaw[4][5])
    assert torch.equal(roles["pmg_aligned"][:, 512:768], jigsaw[2][8])
    assert torch.equal(roles["reverse_placebo"][:, :256], jigsaw[2][2])
    assert torch.equal(roles["deepest_placebo"][:, :256], jigsaw[8][8])
    assert all(
        torch.equal(value[:, -256:], global_feature) for value in roles.values()
    )


def test_oof_readout_predicts_every_row_once_and_converges() -> None:
    rng = np.random.default_rng(20260714)
    rows = 250
    targets = np.arange(rows, dtype=np.int64) % 5
    folds = (np.arange(rows, dtype=np.int64) // 5) % 5
    descriptors = rng.normal(0.0, 0.05, size=(rows, pmg.ROLE_DIM)).astype(np.float32)
    descriptors[np.arange(rows), targets] += 3.0
    probabilities, telemetry = pmg.fit_oof_readout(
        descriptors, targets, folds, seed=20260714
    )
    assert probabilities.shape == (rows, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert telemetry["every_row_predicted_once"] is True
    assert telemetry["all_converged"] is True
    assert len(telemetry["folds"]) == 5


def test_analysis_passes_only_for_selective_progressive_signal() -> None:
    targets, folds, keeper, roles, stats = _synthetic_gate_inputs()
    analysis = pmg.build_analysis(
        targets=targets,
        folds=folds,
        keeper_probabilities=keeper,
        role_probabilities=roles,
        descriptor_stats=stats,
    )
    assert analysis["mechanism_gates_passed"] is True
    assert all(analysis["mechanism_gates"].values())
    assert (
        analysis["candidate_vs_control_transitions"]["restricted_fp_net_removal"]
        >= 10
    )


def test_analysis_rejects_broad_class1_recall_contraction() -> None:
    targets, folds, keeper, roles, stats = _synthetic_gate_inputs()
    candidate = roles["pmg_aligned"].copy()
    class1 = targets == 1
    candidate[class1] = _probability(0, 1, 0.80)
    roles["pmg_aligned"] = candidate
    analysis = pmg.build_analysis(
        targets=targets,
        folds=folds,
        keeper_probabilities=keeper,
        role_probabilities=roles,
        descriptor_stats=stats,
    )
    assert analysis["mechanism_gates_passed"] is False
    assert analysis["mechanism_gates"]["class1_recall_delta_ge_minus0005"] is False
    assert analysis["mechanism_gates"]["tp_break_le_fn_rescue_plus_1"] is False


def test_model_state_hash_supports_scalar_buffers_and_is_mutation_sensitive() -> None:
    model = nn.BatchNorm1d(4)
    before = pmg._model_state_sha256(model)
    model.num_batches_tracked.add_(1)
    after = pmg._model_state_sha256(model)
    assert before != after


def test_manifest_and_probability_replay_reconstruct_analysis(tmp_path) -> None:
    targets, folds, keeper, roles, stats = _synthetic_gate_inputs()
    analysis = pmg.build_analysis(
        targets=targets,
        folds=folds,
        keeper_probabilities=keeper,
        role_probabilities=roles,
        descriptor_stats=stats,
    )
    np.savez_compressed(
        tmp_path / "oof_probabilities.npz",
        targets=targets,
        folds=folds,
        keeper_probabilities=keeper,
        **{f"probabilities_{role}": roles[role] for role in pmg.ROLE_NAMES},
    )
    pmg._write_json(tmp_path / "descriptor_statistics.json", stats)
    pmg._write_json(tmp_path / "summary.json", {"analysis": analysis})
    pmg._write_manifest(tmp_path)
    replay = pmg.replay_summary(tmp_path / "summary.json")
    assert replay["replay_passed"] is True
    assert replay["maximum_numeric_difference"] <= 1e-12
    assert replay["payload_count"] == 3


def test_manifest_detects_payload_mutation(tmp_path) -> None:
    (tmp_path / "payload.txt").write_text("locked\n", encoding="utf-8")
    pmg._write_manifest(tmp_path)
    (tmp_path / "payload.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload differs"):
        pmg._verify_manifest(tmp_path)


def _write_visual_review_fixture(tmp_path, *, automated_passed: bool) -> Path:
    contact_path = tmp_path / "jigsaw_contact_sheet.png"
    contact_path.write_bytes(b"fixed-sheet")
    summary_path = tmp_path / "summary.json"
    pmg._write_json(
        summary_path,
        {
            "mode": "pmg_progressive_jigsaw_signal_a0_train_information_gate",
            "status": "awaiting_visual_review"
            if automated_passed
            else "rejected_automated_gate",
            "automated_gate_passed": automated_passed,
            "visual_review": {
                "required": True,
                "completed": False,
                "geometry_passed": None,
            },
            "artifacts": {
                "jigsaw_contact_sheet": pmg._artifact_record(
                    contact_path, root=tmp_path
                )
            },
            "trainer_integration_authorized": False,
            "short_pair_authorized": False,
            "full_train_authorized": False,
            "current_command_update_authorized": False,
        },
    )
    pmg._write_manifest(tmp_path)
    return summary_path


def test_visual_review_pass_is_hash_locked_and_authorizes_only_short_pair(
    tmp_path,
) -> None:
    summary_path = _write_visual_review_fixture(tmp_path, automated_passed=True)
    reviewed_sha = pmg._sha256(summary_path)
    result = pmg.finalize_visual_review(
        summary_path,
        result="pass",
        expected_summary_sha256=reviewed_sha,
    )
    finalized = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["all_a0_gates_passed"] is True
    assert finalized["status"] == "passed_all_a0_gates"
    assert finalized["short_pair_authorized"] is True
    assert finalized["full_train_authorized"] is False
    assert finalized["current_command_update_authorized"] is False
    assert finalized["visual_review"]["reviewed_summary_sha256"] == reviewed_sha
    assert pmg._verify_manifest(tmp_path)["payload_count"] == 2


def test_visual_review_cannot_override_automated_rejection(tmp_path) -> None:
    summary_path = _write_visual_review_fixture(tmp_path, automated_passed=False)
    result = pmg.finalize_visual_review(
        summary_path,
        result="pass",
        expected_summary_sha256=pmg._sha256(summary_path),
    )
    finalized = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result["all_a0_gates_passed"] is False
    assert finalized["status"] == "rejected_automated_gate"
    assert finalized["short_pair_authorized"] is False


def test_visual_review_rejects_stale_summary_hash(tmp_path) -> None:
    summary_path = _write_visual_review_fixture(tmp_path, automated_passed=True)
    with pytest.raises(ValueError, match="summary hash differs"):
        pmg.finalize_visual_review(
            summary_path,
            result="pass",
            expected_summary_sha256="0" * 64,
        )


def test_output_directory_must_not_preexist(tmp_path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        pmg._prepare_output_dir(output)


def test_descriptor_statistics_reports_rank_and_nonconstant_blocks() -> None:
    rng = np.random.default_rng(7)
    descriptors = rng.normal(size=(600, pmg.ROLE_DIM)).astype(np.float32)
    stats = pmg.descriptor_statistics(descriptors)
    assert stats["shape"] == [600, pmg.ROLE_DIM]
    assert stats["effective_rank_512_row_sketch"] > 16.0
    assert not any(row["constant"] for row in stats["blocks"])
    json.dumps(stats)
