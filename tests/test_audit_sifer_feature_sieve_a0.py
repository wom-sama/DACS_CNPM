from __future__ import annotations

from argparse import Namespace
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from trkh.tools.audit_sifer_feature_sieve_a0 import (
    EXPECTED_AUXILIARY_PARAMETERS,
    EXPECTED_FORGET_STEPS,
    SiferAuxiliary,
    SiferBasicBlock,
    _equation_diagnostics,
    _make_auxiliary,
    _locked_args_exact,
    _module_parameter_sha256,
    _select_visual_rows,
    assess_a0,
    uniform_cross_entropy,
)
from trkh.tools.replay_sifer_feature_sieve_a0 import _assert_close


def _locked_namespace() -> Namespace:
    return Namespace(
        device="cuda",
        batch_size=32,
        num_workers=4,
        seed=42,
        fold=0,
        max_train_batches=60,
        main_learning_rate=1e-5,
        main_weight_decay=0.05,
        aux_learning_rate=1e-2,
        aux_weight_decay=1e-4,
        forget_learning_rate=1e-4,
        forget_interval=5,
        aux_seed=42042,
        aux_depth=2,
        aux_width=256,
        focus_class=1,
        contact_rows=8,
        max_runtime_ratio=1.60,
        max_peak_vram_gib=6.50,
    )


def test_locked_arguments_accept_only_protocol_values() -> None:
    args = _locked_namespace()
    assert _locked_args_exact(args)
    changed = copy.deepcopy(args)
    changed.forget_interval = 6
    assert not _locked_args_exact(changed)


def test_auxiliary_topology_has_locked_parameter_count_and_shape() -> None:
    auxiliary = SiferAuxiliary()
    assert sum(parameter.numel() for parameter in auxiliary.parameters()) == (
        EXPECTED_AUXILIARY_PARAMETERS
    )
    auxiliary.eval()
    output = auxiliary(torch.randn(2, 256, 11, 11))
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()


def test_basic_block_skip_contract_is_exact() -> None:
    inputs = torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]]]])
    without_skip = SiferBasicBlock(1, 1, apply_skip=False).eval()
    with_skip = SiferBasicBlock(1, 1, apply_skip=True).eval()
    for block in (without_skip, with_skip):
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
    assert torch.equal(without_skip(inputs), torch.zeros_like(inputs))
    assert torch.equal(with_skip(inputs), F.relu(inputs))


def test_uniform_cross_entropy_matches_soft_target_cross_entropy() -> None:
    logits = torch.tensor(
        [[0.2, -0.4, 1.1, 0.7, -0.9], [1.0, 0.0, -0.3, 0.4, 0.2]],
        requires_grad=True,
    )
    targets = torch.full_like(logits, 0.2)
    observed = uniform_cross_entropy(logits)
    expected = F.cross_entropy(logits, targets)
    assert torch.allclose(observed, expected, rtol=0.0, atol=1e-7)
    gradient = torch.autograd.grad(observed, logits)[0]
    assert torch.isfinite(gradient).all()


def test_equation_diagnostics_are_exact_and_use_twelve_forget_steps() -> None:
    diagnostics = _equation_diagnostics(_locked_namespace())
    assert diagnostics["uniform_ce_max_abs_error"] <= 1e-7
    assert diagnostics["hard_ce_max_abs_error"] <= 1e-7
    assert diagnostics["gradient_finite"]
    assert diagnostics["auxiliary_parameters"] == EXPECTED_AUXILIARY_PARAMETERS
    assert diagnostics["forget_steps"] == list(EXPECTED_FORGET_STEPS)


def test_auxiliary_initialization_preserves_cpu_rng() -> None:
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    auxiliary = _make_auxiliary(_locked_namespace(), torch.device("cpu"))
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)
    assert sum(parameter.numel() for parameter in auxiliary.parameters()) == (
        EXPECTED_AUXILIARY_PARAMETERS
    )


def test_detached_identify_updates_auxiliary_but_not_stem() -> None:
    torch.manual_seed(7)
    stem = nn.Conv2d(3, 256, kernel_size=1, bias=False)
    auxiliary = SiferAuxiliary()
    optimizer = torch.optim.SGD(auxiliary.parameters(), lr=1e-2)
    images = torch.randn(2, 3, 7, 7)
    targets = torch.tensor([0, 1])
    stem_before = _module_parameter_sha256(stem)
    auxiliary_before = _module_parameter_sha256(auxiliary)
    with torch.no_grad():
        features = stem(images).detach()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(auxiliary(features), targets)
    loss.backward()
    optimizer.step()
    assert _module_parameter_sha256(stem) == stem_before
    assert _module_parameter_sha256(auxiliary) != auxiliary_before


def test_uniform_forget_updates_stem_but_not_auxiliary_parameters() -> None:
    torch.manual_seed(11)
    stem = nn.Conv2d(3, 256, kernel_size=1, bias=False)
    auxiliary = SiferAuxiliary().eval()
    optimizer = torch.optim.SGD(stem.parameters(), lr=1e-4)
    images = torch.randn(2, 3, 7, 7)
    stem_before = _module_parameter_sha256(stem)
    auxiliary_before = _module_parameter_sha256(auxiliary)
    for parameter in auxiliary.parameters():
        parameter.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    loss = uniform_cross_entropy(auxiliary(stem(images)))
    loss.backward()
    optimizer.step()
    assert _module_parameter_sha256(stem) != stem_before
    assert _module_parameter_sha256(auxiliary) == auxiliary_before


def _prediction_row(
    sample_index: int,
    target: int,
    prediction: int,
    p1: float,
) -> dict[str, object]:
    remaining = (1.0 - p1) / 4.0
    return {
        "sample_index": sample_index,
        "target": target,
        "prediction": prediction,
        **{
            f"prob_{index}": p1 if index == 1 else remaining
            for index in range(5)
        },
    }


def test_visual_selection_uses_locked_transition_priority() -> None:
    control = [
        _prediction_row(10, 0, 1, 0.70),
        _prediction_row(11, 1, 1, 0.80),
        _prediction_row(12, 1, 0, 0.20),
        _prediction_row(13, 2, 2, 0.10),
        _prediction_row(14, 3, 3, 0.01),
    ]
    candidate = [
        _prediction_row(10, 0, 0, 0.20),
        _prediction_row(11, 1, 0, 0.30),
        _prediction_row(12, 1, 1, 0.70),
        _prediction_row(13, 2, 1, 0.65),
        _prediction_row(14, 3, 3, 0.02),
    ]
    selected = _select_visual_rows(
        control_rows=control, candidate_rows=candidate, limit=5
    )
    assert [row["category"] for row in selected] == [
        "restricted_fp_removed",
        "class1_tp_broken",
        "class1_fn_rescued",
        "restricted_fp_created",
        "largest_p1_shift",
    ]


def test_visual_selection_reserves_safety_categories_before_filling() -> None:
    control = [
        _prediction_row(index, 0, 1, 0.90 - index * 0.01)
        for index in range(8)
    ]
    candidate = [
        _prediction_row(index, 0, 0, 0.10 + index * 0.01)
        for index in range(8)
    ]
    control.append(_prediction_row(20, 1, 1, 0.80))
    candidate.append(_prediction_row(20, 1, 0, 0.20))
    selected = _select_visual_rows(
        control_rows=control, candidate_rows=candidate, limit=8
    )
    assert selected[0]["category"] == "restricted_fp_removed"
    assert selected[1]["category"] == "class1_tp_broken"
    assert len(selected) == 8


def _metrics() -> dict[str, object]:
    return {
        "predicted_support": [100, 100, 100, 100, 100],
    }


def test_assess_a0_requires_precision_and_tp_safety() -> None:
    clean = {
        "control": _metrics(),
        "candidate": _metrics(),
        "delta": {
            "macro_f1": 0.001,
            "class1_f1": 0.006,
            "class1_precision": 0.006,
            "class1_recall": 0.0,
        },
        "transitions": {
            "restricted_focus_fp_reduction": 2,
            "focus_fn_rescue": 2,
            "focus_tp_break": 1,
            "candidate_correction": 4,
            "candidate_harm": 1,
        },
        "maximum_nonfocus_f1_drop": 0.001,
    }
    illumination = [
        {
            "delta": {
                "class1_f1": 0.0,
                "class1_precision": 0.0,
                "class1_recall": 0.0,
            },
            "transitions": {
                "focus_fn_rescue": 1,
                "focus_tp_break": 0,
                "restricted_focus_fp_reduction": 1,
            },
        }
        for _ in range(3)
    ]
    mechanism = {
        "identify_decrease_count": 8,
        "uniform_ce_nonincrease_count": 8,
        "entropy_nondecrease_count": 8,
        "stem_mean_absolute_difference": 1e-3,
        "probability_mean_absolute_difference": 1e-3,
    }
    passed = assess_a0(
        structural_checks={"all": True},
        mechanism=mechanism,
        clean=clean,
        illumination=illumination,
    )
    assert passed["automatic_gates_passed"]
    failed_clean = copy.deepcopy(clean)
    failed_clean["delta"]["class1_recall"] = -0.1
    failed = assess_a0(
        structural_checks={"all": True},
        mechanism=mechanism,
        clean=failed_clean,
        illumination=illumination,
    )
    assert not failed["automatic_gates_passed"]
    assert "clean.class1_recall_delta_gte_minus_0p005" in failed[
        "automatic_failed_checks"
    ]


def test_probability_rows_remain_normalized_fixture() -> None:
    row = _prediction_row(1, 0, 1, 0.6)
    probabilities = np.asarray([row[f"prob_{index}"] for index in range(5)])
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)


def test_replay_tolerance_is_wider_only_for_float32_stem_aggregate() -> None:
    formal = 0.004924195830225621
    row_replay = 0.004924195810138767
    assert np.float32(formal) == np.float32(row_replay)
    _assert_close(
        row_replay,
        formal,
        "mechanism.stem_mean_absolute_difference",
    )
    with pytest.raises(AssertionError):
        _assert_close(
            formal + 6e-10,
            formal,
            "mechanism.stem_mean_absolute_difference",
        )
    with pytest.raises(AssertionError):
        _assert_close(
            1.0 + 2e-12,
            1.0,
            "mechanism.probability_mean_absolute_difference",
        )
