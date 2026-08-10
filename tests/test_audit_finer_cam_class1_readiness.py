import argparse
import csv

import pytest
import torch

from trkh.tools.audit_finer_cam_class1_readiness import (
    FINER_ALPHA,
    MASK_FRACTION,
    PrefixDataset,
    apply_fixed_contrast_gain,
    assess_smoke_permission,
    batched_gradcam,
    finer_weighted_target,
    relative_confidence_drop,
    run_audit,
    select_reference_categories,
    top_fraction_mask,
    transition_audit,
)
from trkh.tools.review_finer_cam_changed_cases import (
    SelectedIndexDataset,
    _assert_reproduction,
    expand_batch_context_indices,
    load_changed_case_rows,
)


class _PathDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

    def sample_paths(self):
        return [f"sample_{index}.jpg" for index in range(len(self))]


def test_prefix_dataset_preserves_path_alignment():
    dataset = PrefixDataset(_PathDataset(), max_samples=2)
    assert len(dataset) == 2
    assert dataset[1] == 1
    assert dataset.sample_paths() == ["sample_0.jpg", "sample_1.jpg"]


def test_selected_index_dataset_preserves_requested_order_and_paths():
    dataset = SelectedIndexDataset(_PathDataset(), [3, 1])
    assert len(dataset) == 2
    assert dataset[0] == 3
    assert dataset[1] == 1
    assert dataset.sample_paths() == ["sample_3.jpg", "sample_1.jpg"]


def test_batch_context_expansion_recreates_original_windows():
    indices = expand_batch_context_indices(
        [3, 9, 11],
        dataset_length=12,
        batch_size=4,
    )
    assert indices == list(range(0, 4)) + list(range(8, 12))


def test_changed_case_csv_filter_and_reproduction(tmp_path):
    path = tmp_path / "cases.csv"
    rows = [
        {
            "split": "val",
            "sample_index": "7",
            "target_index": "1",
            "global_prediction_index": "0",
            "candidate_prediction_index": "1",
        },
        {
            "split": "val",
            "sample_index": "8",
            "target_index": "0",
            "global_prediction_index": "0",
            "candidate_prediction_index": "0",
        },
        {
            "split": "train",
            "sample_index": "9",
            "target_index": "1",
            "global_prediction_index": "0",
            "candidate_prediction_index": "1",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = load_changed_case_rows(path)
    assert [int(row["sample_index"]) for row in changed] == [7]
    reproduced, positions, maximum_difference = _assert_reproduction(
        [
            {
                "sample_index": 0,
                "target_index": 1,
                "global_prediction_index": 0,
                "candidate_prediction_index": 1,
                "standard_relative_drop": 0.01,
                "finer_relative_drop": 0.02,
                "finer_minus_standard_relative_drop": 0.01,
            }
        ],
        [
            {
                **changed[0],
                "standard_relative_drop": 0.01,
                "finer_relative_drop": 0.02,
                "finer_minus_standard_relative_drop": 0.01,
            }
        ],
        [7],
    )
    assert reproduced[0]["sample_index"] == 7
    assert reproduced[0]["review_position"] == 0
    assert positions == [0]
    assert maximum_difference == 0.0


def test_reference_selection_uses_logit_distance_and_excludes_target():
    logits = torch.tensor(
        [
            [0.8, 1.0, 0.9, -1.0, 1.4],
            [2.0, 0.0, -0.1, 0.3, -2.0],
        ]
    )
    references = select_reference_categories(logits, target_class=1, reference_count=3)
    assert references.tolist() == [[2, 0, 4], [2, 3, 0]]
    assert not references.eq(1).any()


def test_finer_weighted_target_matches_official_objective_and_keeps_gradients():
    logits = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
    references = torch.tensor([[0, 2]])
    target = finer_weighted_target(logits, references, target_class=1)
    probabilities = torch.softmax(logits, dim=1)
    expected = (
        probabilities[0, 0] * (logits[0, 1] - FINER_ALPHA * logits[0, 0])
        + probabilities[0, 2] * (logits[0, 1] - FINER_ALPHA * logits[0, 2])
    ) / (probabilities[0, 0] + probabilities[0, 2])
    assert torch.allclose(target[0], expected)
    target.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_batched_gradcam_applies_relu_after_channel_weighting():
    activations = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]]]
    )
    gradients = torch.tensor(
        [[[[2.0, 2.0], [2.0, 2.0]], [[-1.0, -1.0], [-1.0, -1.0]]]]
    )
    heatmap = batched_gradcam(activations, gradients)
    expected_raw = torch.relu(2.0 * activations[0, 0] - activations[0, 1])
    expected = expected_raw / expected_raw.max()
    assert torch.allclose(heatmap[0], expected)
    assert heatmap.min().item() >= 0.0
    assert heatmap.max().item() == pytest.approx(1.0)


def test_top_fraction_mask_respects_valid_pixels_and_zero_cam():
    heatmap = torch.arange(20, dtype=torch.float32).view(1, 4, 5)
    valid = torch.ones_like(heatmap, dtype=torch.bool)
    valid[:, 3, :] = False
    mask = top_fraction_mask(heatmap, valid, fraction=0.20)
    assert MASK_FRACTION == 0.05
    assert int(mask.sum().item()) == 3
    assert not bool((mask & ~valid).any())
    assert mask[0, 2, 4]

    zero_mask = top_fraction_mask(torch.zeros_like(heatmap), valid, fraction=0.20)
    assert not bool(zero_mask.any())


def test_relative_confidence_drop_uses_frozen_weighted_reference():
    original = torch.tensor([[0.2, 0.6, 0.2]])
    masked = torch.tensor([[0.3, 0.4, 0.3]])
    references = torch.tensor([[0, 2]])
    weights = torch.tensor([[0.75, 0.25]])
    drop = relative_confidence_drop(
        original,
        masked,
        references,
        weights,
        target_class=1,
    )
    assert drop.item() == pytest.approx(0.3)


def test_fixed_contrast_gain_only_changes_top2_class1_rows():
    probabilities = torch.tensor(
        [
            [0.45, 0.40, 0.15],
            [0.60, 0.10, 0.30],
            [0.20, 0.60, 0.20],
        ]
    )
    candidate = apply_fixed_contrast_gain(
        probabilities,
        torch.tensor([0.10, 0.50, -0.50]),
        target_class=1,
    )
    assert candidate[0].argmax().item() == 1
    assert torch.allclose(candidate[1], probabilities[1])
    assert candidate[2].argmax().item() != 1
    assert torch.allclose(candidate.sum(dim=1), torch.ones(3))


def test_transition_audit_counts_class1_actions():
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


def _result(*, macro, class1_f1, recall, auc, transitions, count, rd_gain=True):
    standard_rd = 0.02
    finer_rd = 0.03 if rd_gain else 0.01
    return {
        "sample_count": count,
        "global_metrics": {
            "macro_f1": 0.883,
            "class1": {"precision": 0.61, "recall": 0.78, "f1": 0.678},
        },
        "candidate_metrics": {
            "macro_f1": macro,
            "class1": {"precision": 0.66, "recall": recall, "f1": class1_f1},
        },
        "transitions": transitions,
        "direction": {
            "fn_vs_fp_auroc": {"finer_minus_standard_relative_drop": auc}
        },
        "groups": {
            "all": {
                "standard_relative_drop": standard_rd,
                "finer_relative_drop": finer_rd,
            }
        },
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
        class1_f1=0.71,
        recall=0.79,
        auc=0.70,
        transitions=transitions,
        count=100,
    )
    val = _result(
        macro=0.884,
        class1_f1=0.71,
        recall=0.79,
        auc=0.66,
        transitions=transitions,
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
