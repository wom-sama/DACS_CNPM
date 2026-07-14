from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from trkh.tools.audit_pdisco_semantic_part_readiness import (
    EXPECTED_KEEPER_SHA256,
    SemanticPartHead,
    SplitCache,
    _write_artifact_manifest,
    align_cw90_patch_indices_to_original,
    assign_source_grouped_folds,
    classification_metrics,
    dataset_sample_paths,
    evaluate_gates,
    focus_direction_auc,
    part_equivariance_loss,
    pixel_assignment_entropy_loss,
    rotate_bbox_cw90,
    semantic_part_losses,
    sha256_state_dict,
    total_variation_loss,
    transition_counts,
)


class _PathDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> int:
        return int(index)

    def sample_paths(self) -> list[str]:
        return [f"image_{index}.jpg" for index in range(len(self))]


def test_clockwise_rotation_transforms_bbox_and_patch_indices() -> None:
    bbox = torch.tensor([[0.20, 0.30, 0.40, 0.50]], dtype=torch.float32)
    rotated = rotate_bbox_cw90(bbox)
    assert torch.allclose(rotated, torch.tensor([[0.70, 0.20, 0.50, 0.40]]))

    original_grid = torch.arange(16).reshape(4, 4)
    rotated_grid = torch.rot90(original_grid, k=-1, dims=(-2, -1))
    rotated_indices = torch.arange(16).reshape(1, -1)
    aligned = align_cw90_patch_indices_to_original(
        rotated_indices,
        grid_size=(4, 4),
    )
    assert torch.equal(
        original_grid.flatten()[aligned.flatten()],
        rotated_grid.flatten(),
    )


def test_dataset_sample_paths_preserve_subset_order() -> None:
    dataset = Subset(_PathDataset(), [4, 1, 3])
    assert dataset_sample_paths(dataset) == [
        "image_4.jpg",
        "image_1.jpg",
        "image_3.jpg",
    ]


def test_semantic_part_head_reconstructs_missing_positions_as_background() -> None:
    torch.manual_seed(7)
    head = SemanticPartHead(feature_dim=4, num_classes=5, num_parts=2, part_dropout=0.0)
    head.eval()
    tokens = torch.randn(1, 3, 4)
    indices = torch.tensor([[0, 5, 10]], dtype=torch.long)
    valid = torch.tensor([[True, False, True]])
    output = head(
        tokens,
        indices,
        valid,
        grid_size=(4, 4),
        stochastic_assignment=False,
    )
    maps = output["maps"]
    assert tuple(maps.shape) == (1, 3, 4, 4)
    assert tuple(output["logits"].shape) == (1, 5)
    assert torch.allclose(maps.sum(dim=1), torch.ones(1, 4, 4), atol=1e-6)
    assert float(maps[0, -1, 1, 1]) == pytest.approx(1.0)
    assert float(maps[0, -1, 0, 1]) == pytest.approx(1.0)
    assert torch.isfinite(output["logits"]).all()


def test_semantic_part_losses_are_finite_and_identity_equivariance_is_zero() -> None:
    torch.manual_seed(11)
    head = SemanticPartHead(feature_dim=8, num_classes=5, num_parts=2, part_dropout=0.0)
    head.eval()
    tokens = torch.randn(3, 12, 8)
    indices = torch.arange(12).reshape(1, -1).expand(3, -1)
    valid = torch.ones(3, 12, dtype=torch.bool)
    output = head(
        tokens,
        indices,
        valid,
        grid_size=(4, 4),
        stochastic_assignment=False,
    )
    losses = semantic_part_losses(output, output)
    assert set(losses) == {
        "presence",
        "equivariance",
        "orthogonality",
        "total_variation",
        "enforced_background",
        "pixel_entropy",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(losses["equivariance"]) == pytest.approx(0.0, abs=1e-6)
    assert float(part_equivariance_loss(output["maps"], output["maps"])) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert float(total_variation_loss(output["maps"])) >= 0.0
    assert float(pixel_assignment_entropy_loss(output["maps"])) >= 0.0


def test_source_grouped_folds_never_split_a_source() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2, dtype=np.int64)
    groups = np.asarray([f"source_{index // 2}" for index in range(20)])
    splits, summary = assign_source_grouped_folds(labels, groups, folds=2, seed=19)
    assert summary["source_overlap"] == 0
    assert summary["assignment_complete"] is True
    assert len(splits) == 2
    for fit, holdout in splits:
        assert set(groups[fit]).isdisjoint(set(groups[holdout]))


def test_transition_and_direction_statistics_use_class1_fn_as_positive() -> None:
    labels = np.asarray([1, 0, 1, 4], dtype=np.int64)
    keeper = np.asarray(
        [
            [0.6, 0.4],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    control = keeper.copy()
    candidate = np.asarray(
        [
            [0.3, 0.7],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    transitions = transition_counts(labels, keeper, candidate, focus_class=1)
    assert transitions["focus_fn_rescue"] == 1
    assert transitions["focus_fp_remove"] == 1
    assert transitions["focus_tp_break"] == 1
    direction = focus_direction_auc(labels, keeper, control, candidate, focus_class=1)
    assert direction["focus_fn_rows"] == 1
    assert direction["focus_fp_rows"] == 1
    assert direction["auc_fn_positive"] == pytest.approx(1.0)


def _fake_cache(tmp_path: Path, prefix: str) -> SplitCache:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.eye(5, dtype=np.float32)[labels]
    return SplitCache(
        root=tmp_path / prefix,
        tokens_original=np.zeros((4, 2, 3), dtype=np.float16),
        tokens_rot90=np.zeros((4, 2, 3), dtype=np.float16),
        patch_indices_original=np.zeros((4, 2), dtype=np.int16),
        patch_indices_rot90_aligned=np.zeros((4, 2), dtype=np.int16),
        token_valid_original=np.ones((4, 2), dtype=np.uint8),
        token_valid_rot90=np.ones((4, 2), dtype=np.uint8),
        grid_valid=np.ones((4, 4), dtype=np.uint8),
        labels=labels,
        sample_index=np.arange(4, dtype=np.int64),
        keeper_probabilities=probabilities,
        bbox=np.ones((4, 4), dtype=np.float32) * 0.5,
        paths=np.asarray([f"{prefix}_{index}.jpg" for index in range(4)]),
        source_stems=np.asarray([f"{prefix}_{index}" for index in range(4)]),
        grid_size=(2, 2),
    )


def test_full_gate_fails_closed_when_locked_support_is_missing(tmp_path: Path) -> None:
    train = _fake_cache(tmp_path, "train")
    val = _fake_cache(tmp_path, "val")
    class_names = [str(index) for index in range(5)]
    metrics = classification_metrics(
        np.asarray(val.labels),
        np.asarray(val.keeper_probabilities),
        class_names=class_names,
    )
    transition = {
        "changed": 0,
        "corrections": 0,
        "harms": 0,
        "neutral": 0,
        "focus_fn_rescue": 0,
        "focus_tp_break": 0,
        "focus_fp_remove": 0,
        "focus_fp_create": 0,
    }
    part_metrics = {
        "foreground_part_mean_mass": [0.3, 0.3],
        "foreground_part_use_entropy_normalized": 1.0,
        "background_outside_minus_inside": 0.2,
        "foreground_equivariance_cosine": 0.9,
        "total_variation": 0.1,
        "pixel_entropy": 0.1,
    }
    gate = evaluate_gates(
        preflight=False,
        checkpoint_sha256=EXPECTED_KEEPER_SHA256,
        train_cache=train,
        val_cache=val,
        fold_summary={"source_overlap": 0, "assignment_complete": True},
        oof_metrics={"control": metrics, "candidate": metrics},
        val_metrics={"keeper": metrics, "control": metrics, "candidate": metrics},
        fold_results=[{"control": metrics, "candidate": metrics} for _ in range(5)],
        transitions={"candidate_vs_keeper_val": transition},
        direction={
            "train_oof_candidate_vs_control": {"auc_fn_positive": 0.6},
            "val_candidate_vs_control": {"auc_fn_positive": 0.6},
        },
        part_metrics={"control": part_metrics, "candidate": part_metrics},
        initialization_equal=True,
        all_finite=True,
    )
    assert gate["all_automated_checks_passed"] is False
    assert "train_support" in gate["failed_checks"]
    assert "validation_support" in gate["failed_checks"]
    assert gate["image_smoke_permitted"] is False


def test_state_hash_is_stable_for_matched_initialization() -> None:
    torch.manual_seed(23)
    first = SemanticPartHead(feature_dim=4, num_classes=5).state_dict()
    clone = {key: value.clone() for key, value in first.items()}
    assert sha256_state_dict(first) == sha256_state_dict(clone)
    clone["prototypes"][0, 0] += 0.01
    assert sha256_state_dict(first) != sha256_state_dict(clone)


def test_artifact_manifest_uses_retained_cache_summary_after_cleanup(
    tmp_path: Path,
) -> None:
    (tmp_path / "cache_summary_retained.json").write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    manifest = _write_artifact_manifest(tmp_path)
    assert manifest["cache_payloads_are_hashed_in"] == "cache_summary_retained.json"
