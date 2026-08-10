from __future__ import annotations

import numpy as np
import torch

from trkh.tools import run_b46_pareto_margin_distillation_fold0 as b46


def test_balanced_step_indices_are_exact_and_nearly_uniform() -> None:
    labels = torch.arange(5).repeat_interleave(b46.BASES_PER_CLASS)
    exposure = np.zeros(labels.numel(), dtype=np.int64)
    for step in range(b46.CONSOLIDATION_STEPS):
        indices = b46._balanced_step_indices(labels, step)
        assert len(indices) == 5 * b46.SOURCES_PER_CLASS_PER_STEP
        selected_labels = labels[indices]
        assert torch.equal(
            torch.bincount(selected_labels, minlength=5),
            torch.full((5,), b46.SOURCES_PER_CLASS_PER_STEP, dtype=torch.long),
        )
        exposure[indices] += 1
    assert int(exposure.sum()) == 300
    assert int(exposure.max() - exposure.min()) <= 1


def test_formal_budget_matches_metric_free_preflight_budget() -> None:
    formal_views = (
        b46.CONSOLIDATION_STEPS * 5 * b46.SOURCES_PER_CLASS_PER_STEP
    )
    preflight_views = b46.b46.STEPS * b46.b46.BASES_PER_CLASS * 5
    assert formal_views == preflight_views == 300
    assert b46.LEARNING_RATE == b46.b46.LEARNING_RATE
    assert b46.MICROBATCH_SIZE == b46.b46.BATCH_SIZE
