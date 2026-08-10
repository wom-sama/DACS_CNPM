from __future__ import annotations

import numpy as np
import torch

from trkh.tools.probe_pwca_token_interaction_readiness import (
    SharedTokenAttentionReadout,
    assess_pwca_readiness,
    build_random_partner_plan,
)


def _metrics(macro: float, focus: float):
    return {"macro_f1": macro, "focus_f1": focus}


def _transitions(*, corrections: int = 12, harms: int = 4):
    return {
        "corrections": corrections,
        "harms": harms,
        "class1_fn_rescued": 5,
        "class1_tp_broken": 2,
        "class1_fp_removed": 6,
        "class1_fp_created": 2,
    }


def test_random_partner_plan_is_natural_frequency_and_source_safe() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    groups = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g"], dtype=object)
    plan = build_random_partner_plan(
        labels,
        groups,
        epochs=3,
        batch_size=8,
        seed=9,
        class_count=3,
    )
    stats = plan["stats"]
    assert stats["anchor_visit_min"] == 3
    assert stats["anchor_visit_max"] == 3
    assert stats["same_source_pair_count"] == 0
    assert stats["anchor_oversampling"] is False
    assert stats["partner_label_loss"] is False
    for batches in plan["epochs"]:
        for batch in batches:
            assert np.all(groups[batch["anchor"]] != groups[batch["partner"]])


def test_shared_pwca_attention_has_expected_shape_and_mass() -> None:
    model = SharedTokenAttentionReadout(8, 2, 2.0, 3, dropout=0.0)
    for parameter in model.qkv.parameters():
        torch.nn.init.zeros_(parameter)
    target = torch.randn(4, 5, 8)
    distractor = torch.randn(4, 5, 8)
    self_logits = model(target)
    pwca_logits, attention = model(target, distractor, return_attention=True)
    assert self_logits.shape == (4, 3)
    assert pwca_logits.shape == (4, 3)
    assert attention.shape == (4, 2, 5, 10)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(4, 2, 5))
    distractor_mass = attention[:, :, 0, 5:].sum(dim=-1)
    assert torch.allclose(distractor_mass, torch.full((4, 2), 0.5))


def test_pwca_gate_requires_transfer_direction_and_attention_safety() -> None:
    ready = assess_pwca_readiness(
        train_samples=9215,
        val_samples=2606,
        source_group_count=8064,
        train_val_source_overlap=0,
        max_fold_source_overlap=0,
        pair_audit_ok=True,
        control_oof_metrics=_metrics(0.930, 0.790),
        pwca_oof_metrics=_metrics(0.934, 0.805),
        control_val_metrics=_metrics(0.880, 0.670),
        pwca_val_metrics=_metrics(0.890, 0.710),
        direct_metrics=_metrics(0.885, 0.686),
        oof_transitions=_transitions(),
        val_transitions=_transitions(),
        direct_transitions=_transitions(),
        distractor_mass_mean=0.20,
        distractor_mass_p95=0.40,
    )
    assert ready["smoke_permission"] is True

    rejected = assess_pwca_readiness(
        train_samples=9215,
        val_samples=2606,
        source_group_count=8064,
        train_val_source_overlap=0,
        max_fold_source_overlap=0,
        pair_audit_ok=True,
        control_oof_metrics=_metrics(0.930, 0.790),
        pwca_oof_metrics=_metrics(0.929, 0.780),
        control_val_metrics=_metrics(0.880, 0.670),
        pwca_val_metrics=_metrics(0.875, 0.650),
        direct_metrics=_metrics(0.885, 0.686),
        oof_transitions=_transitions(corrections=2, harms=8),
        val_transitions=_transitions(corrections=2, harms=8),
        direct_transitions=_transitions(corrections=2, harms=8),
        distractor_mass_mean=0.48,
        distractor_mass_p95=0.65,
    )
    assert rejected["smoke_permission"] is False
    assert "oof_class1_gain" in rejected["failed_checks"]
    assert "distractor_attention_tail_safe" in rejected["failed_checks"]
