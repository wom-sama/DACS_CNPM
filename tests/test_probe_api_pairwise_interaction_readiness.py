from __future__ import annotations

import numpy as np
import torch

from trkh.tools.probe_api_pairwise_interaction_readiness import (
    AttentivePairwiseInteractionHead,
    _api_loss,
    assess_api_pairwise_readiness,
    build_pair_plan,
)


def _metrics(macro: float, focus: float):
    return {"macro_f1": macro, "focus_f1": focus}


def _transitions(*, corrections: int = 10, harms: int = 3):
    return {
        "corrections": corrections,
        "harms": harms,
        "class1_fn_rescued": 5,
        "class1_tp_broken": 2,
        "class1_fp_removed": 6,
        "class1_fp_created": 2,
    }


def test_pair_plan_visits_each_sample_once_and_excludes_same_source() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    groups = np.asarray(["shared", "shared", "a", "b", "b", "c"], dtype=object)
    plan = build_pair_plan(
        features,
        labels,
        groups,
        epochs=2,
        batch_size=6,
        seed=7,
        class_count=2,
    )
    stats = plan["stats"]
    assert stats["anchor_visit_min"] == 2
    assert stats["anchor_visit_max"] == 2
    assert stats["same_source_pair_count"] == 0
    assert stats["anchor_oversampling"] is False
    assert stats["partner_reuse"] is True
    assert stats["all_classes_have_intra_support"] is True
    assert stats["all_classes_have_inter_support"] is True
    for batches in plan["epochs"]:
        for batch in batches:
            assert np.all(groups[batch["first"]] != groups[batch["second"]])


def test_api_zero_mutual_uses_expected_residual_gate_and_plain_inference() -> None:
    model = AttentivePairwiseInteractionHead(4, 3, 2)
    for parameter in model.mutual.parameters():
        torch.nn.init.zeros_(parameter)
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    second = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    first_self, second_self, first_other, second_other = model.interact(first, second)
    assert torch.allclose(first_self, 1.5 * first)
    assert torch.allclose(first_other, 1.5 * first)
    assert torch.allclose(second_self, 1.5 * second)
    assert torch.allclose(second_other, 1.5 * second)
    assert torch.equal(model.inference_logits(first), model.classifier(first))


def test_anchor_only_loss_uses_partner_as_context_not_as_label_target() -> None:
    model = AttentivePairwiseInteractionHead(2, 2, 2)
    for parameter in model.mutual.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()
    first = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    second = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    first_labels = torch.tensor([0, 1])
    partner_labels_a = torch.tensor([0, 0])
    partner_labels_b = torch.tensor([1, 1])
    weights = torch.ones(2)
    anchor_a, _, _ = _api_loss(
        model,
        first,
        second,
        first_labels,
        partner_labels_a,
        class_weights=weights,
        ranking_margin=0.05,
        ranking_weight=1.0,
        loss_scope="anchor_only",
    )
    anchor_b, _, _ = _api_loss(
        model,
        first,
        second,
        first_labels,
        partner_labels_b,
        class_weights=weights,
        ranking_margin=0.05,
        ranking_weight=1.0,
        loss_scope="anchor_only",
    )
    symmetric_a, _, _ = _api_loss(
        model,
        first,
        second,
        first_labels,
        partner_labels_a,
        class_weights=weights,
        ranking_margin=0.05,
        ranking_weight=1.0,
        loss_scope="symmetric",
    )
    symmetric_b, _, _ = _api_loss(
        model,
        first,
        second,
        first_labels,
        partner_labels_b,
        class_weights=weights,
        ranking_margin=0.05,
        ranking_weight=1.0,
        loss_scope="symmetric",
    )
    assert torch.equal(anchor_a, anchor_b)
    assert not torch.equal(symmetric_a, symmetric_b)


def test_readiness_gate_requires_transfer_and_class1_safety() -> None:
    ready = assess_api_pairwise_readiness(
        train_samples=9215,
        val_samples=2606,
        source_group_count=8064,
        train_val_source_overlap=0,
        max_fold_source_overlap=0,
        pair_support_ok=True,
        same_source_pair_count=0,
        control_oof_metrics=_metrics(0.930, 0.790),
        api_oof_metrics=_metrics(0.934, 0.805),
        control_val_metrics=_metrics(0.880, 0.670),
        api_val_metrics=_metrics(0.890, 0.710),
        direct_metrics=_metrics(0.885, 0.686),
        oof_transitions=_transitions(),
        val_transitions=_transitions(),
        direct_transitions=_transitions(),
    )
    assert ready["smoke_permission"] is True

    rejected = assess_api_pairwise_readiness(
        train_samples=9215,
        val_samples=2606,
        source_group_count=8064,
        train_val_source_overlap=0,
        max_fold_source_overlap=0,
        pair_support_ok=True,
        same_source_pair_count=0,
        control_oof_metrics=_metrics(0.930, 0.790),
        api_oof_metrics=_metrics(0.929, 0.780),
        control_val_metrics=_metrics(0.880, 0.670),
        api_val_metrics=_metrics(0.875, 0.650),
        direct_metrics=_metrics(0.885, 0.686),
        oof_transitions=_transitions(corrections=2, harms=8),
        val_transitions=_transitions(corrections=1, harms=9),
        direct_transitions=_transitions(corrections=1, harms=9),
    )
    assert rejected["smoke_permission"] is False
    assert "oof_class1_gain" in rejected["failed_checks"]
    assert "val_net_corrections" in rejected["failed_checks"]
