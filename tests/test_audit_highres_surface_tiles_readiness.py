import argparse

import pytest
import torch

from trkh.tools.audit_highres_surface_tiles_readiness import (
    CANDIDATE_TILE_WEIGHT,
    TILE_FRACTION,
    assess_smoke_permission,
    clipped_bbox_for_tile,
    fixed_tile_boxes,
    fuse_global_and_tiles,
    object_crop_geometry,
    transition_audit,
)


def test_fixed_tile_boxes_cover_five_locked_positions():
    boxes = fixed_tile_boxes(100, 80)
    assert TILE_FRACTION == 0.70
    assert [name for name, _ in boxes] == [
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "center",
    ]
    assert boxes[0][1] == (0, 0, 70, 56)
    assert boxes[1][1] == (30, 0, 100, 56)
    assert boxes[-1][1] == (15, 12, 85, 68)


def test_object_crop_geometry_matches_bbox_margin_contract():
    crop, object_xyxy = object_crop_geometry(
        image_size=(200, 100),
        bbox=(0.5, 0.5, 0.5, 0.4),
        margin_ratio=0.05,
    )
    assert crop == (45, 28, 155, 72)
    assert object_xyxy == pytest.approx((5.0, 2.0, 105.0, 42.0))


def test_clipped_bbox_for_tile_is_normalized_and_clipped():
    bbox = clipped_bbox_for_tile((5.0, 2.0, 105.0, 42.0), (30, 0, 100, 40))
    assert bbox == pytest.approx((0.5, 0.525, 1.0, 0.95), abs=1e-5)


def test_probability_fusion_is_fixed_mean_and_normalized():
    global_probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    tile_probabilities = torch.tensor(
        [
            [[0.4, 0.6], [0.6, 0.4]],
            [[0.5, 0.5], [0.1, 0.9]],
        ]
    )
    fused = fuse_global_and_tiles(global_probabilities, tile_probabilities)
    assert CANDIDATE_TILE_WEIGHT == 0.50
    assert torch.allclose(fused, torch.tensor([[0.65, 0.35], [0.3, 0.7]]))
    assert torch.allclose(fused.sum(dim=1), torch.ones(2))


def test_transition_audit_counts_recall_and_false_positive_actions():
    targets = torch.tensor([1, 1, 0, 2, 0])
    global_probabilities = torch.tensor(
        [
            [0.6, 0.4, 0.0],
            [0.1, 0.8, 0.1],
            [0.4, 0.6, 0.0],
            [0.1, 0.6, 0.3],
            [0.8, 0.1, 0.1],
        ]
    )
    candidate_probabilities = torch.tensor(
        [
            [0.3, 0.7, 0.0],
            [0.6, 0.3, 0.1],
            [0.7, 0.3, 0.0],
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
        ]
    )
    audit = transition_audit(targets, global_probabilities, candidate_probabilities)
    assert audit["class1_fn_rescued"] == 1
    assert audit["class1_tp_broken"] == 1
    assert audit["class1_fp_removed"] == 2
    assert audit["class1_fp_created"] == 1
    assert audit["corrections"] == 3
    assert audit["harms"] == 2


def _result(*, macro, focus_f1, recall, transitions, auc, count):
    return {
        "sample_count": count,
        "global_metrics": {
            "macro_f1": 0.883,
            "class1": {"precision": 0.60, "recall": 0.78, "f1": 0.678},
        },
        "candidate_metrics": {
            "macro_f1": macro,
            "class1": {"precision": 0.66, "recall": recall, "f1": focus_f1},
        },
        "transitions": transitions,
        "direction": {"fn_vs_fp_delta_auroc": auc},
    }


def test_smoke_permission_requires_every_locked_gate():
    transitions = {
        "corrections": 8,
        "harms": 3,
        "class1_fp_removed": 5,
        "class1_fp_created": 2,
        "class1_fn_rescued": 4,
        "class1_tp_broken": 1,
    }
    train = _result(
        macro=0.89,
        focus_f1=0.71,
        recall=0.79,
        transitions=transitions,
        auc=0.70,
        count=100,
    )
    val = _result(
        macro=0.884,
        focus_f1=0.71,
        recall=0.79,
        transitions=transitions,
        auc=0.66,
        count=40,
    )
    accepted = assess_smoke_permission(
        train_result=train,
        val_result=val,
        source_overlap_count=0,
        expected_train_count=100,
        expected_val_count=40,
        preflight=False,
    )
    assert accepted["smoke_permission"] is True

    rejected = assess_smoke_permission(
        train_result=train,
        val_result=val,
        source_overlap_count=0,
        expected_train_count=100,
        expected_val_count=40,
        preflight=True,
    )
    assert rejected["checks"]["not_preflight"] is False
    assert rejected["smoke_permission"] is False


def test_preflight_caps_require_explicit_opt_in(tmp_path):
    from trkh.tools.audit_highres_surface_tiles_readiness import run_audit

    args = argparse.Namespace(
        batch_size=1,
        num_workers=0,
        max_train_samples=1,
        max_val_samples=0,
        allow_preflight=False,
        output_dir=tmp_path / "audit",
    )
    with pytest.raises(ValueError, match="allow-preflight"):
        run_audit(args)
