from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from trkh.models.model import MultiHeadSelfAttention
from trkh.tools.audit_validity_aware_attention_a0 import (
    PATCH_COUNT,
    ValidityAwareAttentionController,
    assess_validity_aware_attention_a0,
    build_patch_key_padding_mask,
    build_same_class_fold_source_derangement,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _TinyBlock(
                    MultiHeadSelfAttention(
                        dim=8,
                        num_heads=2,
                        attention_dropout=0.0,
                        projection_dropout=0.0,
                    )
                )
                for _ in range(8)
            ]
        )


class _TinyBlock(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attn = attention


def test_patch_mask_uses_locked_fraction_and_fallback() -> None:
    mask = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    mask[0, :, :, :8] = False
    mask[1] = False
    result = build_patch_key_padding_mask(mask)
    assert result.shape == (2, PATCH_COUNT)
    assert int(result[0].sum()) == 128
    assert not bool(result[1].any())


def test_same_class_fold_derangement_is_bijective_and_source_disjoint() -> None:
    target = np.repeat(np.asarray([0, 1, 0, 1]), 4)
    fold = np.repeat(np.asarray([0, 0, 1, 1]), 4)
    source = np.asarray([f"source_{index}" for index in range(target.size)], dtype=object)
    first = build_same_class_fold_source_derangement(target, fold, source, seed=20260720)
    second = build_same_class_fold_source_derangement(target, fold, source, seed=20260720)
    assert np.array_equal(first, second)
    assert np.unique(first).size == target.size
    assert np.array_equal(target[first], target)
    assert np.array_equal(fold[first], fold)
    assert np.all(source[first] != source)


def test_all_valid_attention_is_numerically_identical() -> None:
    torch.manual_seed(4)
    model = _TinyModel().eval()
    controller = ValidityAwareAttentionController(model)
    module = model.blocks[0].attn
    tokens = torch.randn(2, 4, 8)
    patch_indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    expected = module(
        tokens,
        prefix_count=1,
        patch_indices=patch_indices,
    )
    controller.reset(collect=True)
    controller.set_batch(
        torch.zeros(2, PATCH_COUNT, dtype=torch.bool), torch.tensor([3, 7])
    )
    observed = controller._forward(
        0,
        module,
        tokens,
        prefix_count=1,
        patch_indices=patch_indices,
    )
    assert torch.equal(expected, observed)
    invalid_count = torch.cat(controller.records[0]["invalid_count"]).numpy()
    assert np.array_equal(invalid_count, np.zeros(2))


def test_masked_key_has_zero_attention_and_pruned_index_is_gathered() -> None:
    torch.manual_seed(8)
    model = _TinyModel().eval()
    controller = ValidityAwareAttentionController(model)
    patch_mask = torch.zeros(1, PATCH_COUNT, dtype=torch.bool)
    patch_mask[0, 7] = True
    controller.reset(collect=True)
    controller.set_batch(patch_mask, torch.tensor([11]))
    tokens = torch.randn(1, 3, 8)
    output, attention = controller._forward(
        0,
        model.blocks[0].attn,
        tokens,
        return_attention=True,
        prefix_count=1,
        patch_indices=torch.tensor([[7, 9]], dtype=torch.long),
    )
    assert output.shape == tokens.shape
    assert torch.count_nonzero(attention[..., 1]) == 0
    invalid_count = torch.cat(controller.records[0]["invalid_count"]).numpy()
    pre_mass = torch.cat(controller.records[0]["pre_mass"])
    post_mass = torch.cat(controller.records[0]["post_mass"]).numpy()
    module = model.blocks[0].attn
    qkv = module.qkv(tokens).reshape(1, 3, 3, module.num_heads, module.head_dim)
    query, key = qkv.permute(2, 0, 3, 1, 4)[:2]
    full_attention = ((query @ key.transpose(-2, -1)) * module.scale).softmax(dim=-1)
    expected_pre_mass = full_attention[:, :, 0, 1].mean(dim=1)
    assert invalid_count.tolist() == [1]
    assert torch.allclose(pre_mass, expected_pre_mass, atol=0.0, rtol=0.0)
    assert post_mass.tolist() == [0.0]


def test_installed_context_restores_forward_method() -> None:
    model = _TinyModel().eval()
    controller = ValidityAwareAttentionController(model)
    module = model.blocks[0].attn
    assert "forward" not in module.__dict__
    with controller.installed():
        assert "forward" in module.__dict__
    assert "forward" not in module.__dict__


def _metrics(macro: float, precision: float, recall: float, f1: float):
    return {
        "accuracy": macro,
        "macro_f1": macro,
        "per_class_precision": [0.9, precision, 0.9, 0.9, 0.9],
        "per_class_recall": [0.9, recall, 0.9, 0.9, 0.9],
        "per_class_f1": [0.9, f1, 0.9, 0.9, 0.9],
        "support": [10, 10, 10, 10, 10],
        "predicted_support": [10, 10, 10, 10, 10],
        "confusion_matrix": np.eye(5, dtype=int).tolist(),
        "nll": 0.2,
        "brier": 0.1,
        "ece": 0.02,
    }


def _passing_analysis():
    control = _metrics(0.90, 0.70, 0.80, 0.75)
    candidate = _metrics(0.905, 0.72, 0.80, 0.77)
    placebo = _metrics(0.901, 0.705, 0.80, 0.755)
    candidate_events = {
        "corrections": 20,
        "harms": 10,
        "class1_fn_rescues": 3,
        "class1_tp_breaks": 2,
        "restricted_fp_removals": 20,
        "restricted_fp_creations": 3,
        "net_restricted_fp_removals": 17,
    }
    placebo_events = {
        **candidate_events,
        "class1_tp_breaks": 3,
        "restricted_fp_removals": 8,
        "restricted_fp_creations": 2,
        "net_restricted_fp_removals": 6,
    }
    conditions = {}
    for name in ("clean", "lighting_dim", "lighting_bright", "low_contrast"):
        conditions[name] = {
            "roles": {"control": control, "candidate": candidate, "placebo": placebo},
            "candidate_vs_control": {
                "deltas": {
                    "macro_f1": 0.005,
                    "class1_f1": 0.02,
                    "class1_precision": 0.02,
                    "class1_recall": 0.0,
                },
                "events": candidate_events,
            },
            "placebo_vs_control": {
                "deltas": {
                    "macro_f1": 0.001,
                    "class1_f1": 0.005,
                    "class1_precision": 0.005,
                    "class1_recall": 0.0,
                },
                "events": placebo_events,
            },
        }
    folds = [
        {
            "candidate_vs_control": {
                "class1_precision": 0.01,
                "class1_f1": 0.01,
            }
        }
        for _ in range(5)
    ]
    return {
        "conditions": conditions,
        "clean_folds": folds,
        "direction": {
            "candidate": {"direction_auc": 0.66},
            "placebo": {"direction_auc": 0.60},
        },
    }


def _passing_structural():
    return {
        "provenance_exact": True,
        "ordered_rows_complete": True,
        "class_fold_counts_exact": True,
        "derangement_exact": True,
        "geometry_census_exact": True,
        "no_all_masked_fallback": True,
        "standard_attention_contract": True,
        "control_probability_max_abs_difference": 0.0,
        "all_valid_logit_max_abs_difference": 0.0,
        "all_valid_probability_max_abs_difference": 0.0,
        "candidate_differs_from_keeper": True,
        "maximum_post_mask_invalid_attention_mass": 0.0,
        "model_state_unchanged": True,
        "raw_dataset_unchanged": True,
        "validation_data_unused": True,
        "test_data_unused": True,
        "model_checkpoint_writes_false": True,
        "replay_verified": True,
    }


def test_gate_is_conjunctive() -> None:
    analysis = _passing_analysis()
    result = assess_validity_aware_attention_a0(
        structural=_passing_structural(), analysis=analysis
    )
    assert result["all_passed"]
    analysis["conditions"]["clean"]["candidate_vs_control"]["deltas"][
        "class1_precision"
    ] = 0.009
    result = assess_validity_aware_attention_a0(
        structural=_passing_structural(), analysis=analysis
    )
    assert not result["all_passed"]
    assert "clean.class1_precision_delta_ge_010" in result["failed"]


def test_derangement_rejects_single_source_partition() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        build_same_class_fold_source_derangement(
            np.asarray([0, 0]),
            np.asarray([0, 0]),
            np.asarray(["same", "same"], dtype=object),
        )
