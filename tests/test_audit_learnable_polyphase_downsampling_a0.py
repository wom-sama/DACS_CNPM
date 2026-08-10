from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from trkh.tools.audit_learnable_polyphase_downsampling_a0 import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    EXPECTED_DECLARATION_EXCEPTION,
    HIDDEN_CHANNELS,
    SHIFT_DIRECTIONS,
    LearnablePolyphaseMaxPool2d,
    LearnablePolyphaseSelector,
    _load_module,
    _load_official_split,
    _phase0_legacy_error,
    _translate_bbox,
    _translate_tensor,
    assess_shift_signal,
    combine_polyphase,
    parse_args,
    split_polyphase_v2,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 7
    assert args.seed == 42
    assert HIDDEN_CHANNELS == (8, 16, 32)
    assert SHIFT_DIRECTIONS == (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )


def test_phase_split_uses_locked_v2_order() -> None:
    source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    components = split_polyphase_v2(source)
    assert components.shape == (4, 1, 1, 2, 2)
    assert torch.equal(components[0, 0, 0], torch.tensor([[0.0, 2.0], [8.0, 10.0]]))
    assert torch.equal(components[1, 0, 0], torch.tensor([[1.0, 3.0], [9.0, 11.0]]))
    assert torch.equal(components[2, 0, 0], torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    assert torch.equal(components[3, 0, 0], torch.tensor([[5.0, 7.0], [13.0, 15.0]]))

    with pytest.raises(ValueError, match="even"):
        split_polyphase_v2(torch.zeros(1, 1, 3, 4))


def test_official_v2_split_and_selector_replay_when_source_is_available() -> None:
    root = Path(
        r"D:\DataAI\external_sources\official\learnable_polyphase_sampling\learn_poly_sampling\layers"
    )
    if not root.is_dir():
        pytest.skip("Official LPS source is not installed.")
    official_split = _load_official_split(root / "polydown.py")
    source = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    expected = official_split(
        x=source,
        stride=2,
        in_channels=3,
        num_components=4,
    )
    observed = split_polyphase_v2(source)
    assert torch.equal(observed, expected)

    module = _load_module("_test_locked_lps_logits", root / "lps_logit_layers.py")
    torch.manual_seed(17)
    official = module.LPSLogitLayersV2(3, 5, padding_mode="circular").double()
    local = LearnablePolyphaseSelector(3, 5).double()
    local.load_state_dict(official.state_dict(), strict=True)
    assert torch.allclose(local(observed), official(expected), atol=1e-12, rtol=0.0)


def test_fixed_phase_zero_is_bit_exact_legacy_maxpool() -> None:
    for shape in ((1, 3, 16, 16), (1, 8, 32, 32), (1, 16, 64, 64)):
        error, exact = _phase0_legacy_error(shape)
        assert exact is True
        assert error == 0.0


def test_selector_permutation_and_hard_global_mean_invariance() -> None:
    torch.manual_seed(23)
    source = torch.randn(3, 4, 12, 12, dtype=torch.float64)
    selector = LearnablePolyphaseSelector(4, 6).double().eval()
    components = split_polyphase_v2(source)
    logits = selector(components)
    output, _, _ = combine_polyphase(components, logits, training=False)

    shifted = torch.roll(source, shifts=(-1, -1), dims=(-2, -1))
    shifted_components = split_polyphase_v2(shifted)
    shifted_logits = selector(shifted_components)
    permutation = torch.tensor([3, 2, 1, 0])
    assert torch.allclose(
        logits,
        shifted_logits.index_select(1, permutation),
        atol=1e-12,
        rtol=0.0,
    )
    shifted_output, _, _ = combine_polyphase(
        shifted_components, shifted_logits, training=False
    )
    assert torch.allclose(
        output.mean(dim=(-1, -2)),
        shifted_output.mean(dim=(-1, -2)),
        atol=1e-12,
        rtol=0.0,
    )


def test_gumbel_training_reaches_both_selector_convolutions() -> None:
    torch.manual_seed(29)
    pool = LearnablePolyphaseMaxPool2d(4, 6).train()
    source = torch.randn(3, 4, 16, 16, requires_grad=True)
    output = pool(source)
    output.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    for parameter in pool.selector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(pool.selector.conv1.weight.grad) > 0
    assert torch.count_nonzero(pool.selector.conv2.weight.grad) > 0


def test_translation_moves_content_bbox_and_mask_in_the_same_direction() -> None:
    source = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    shifted = _translate_tensor(source, dx=1, dy=0, replicate=True)
    expected = torch.tensor(
        [[[[0.0, 0.0, 1.0], [3.0, 3.0, 4.0], [6.0, 6.0, 7.0]]]]
    )
    assert torch.equal(shifted, expected)

    mask = torch.ones(1, 3, 3, dtype=torch.uint8)
    shifted_mask = _translate_tensor(mask, dx=1, dy=0, replicate=False)
    assert torch.equal(shifted_mask[:, :, 0], torch.zeros(1, 3, dtype=torch.uint8))
    assert torch.equal(shifted_mask[:, :, 1:], torch.ones(1, 3, 2, dtype=torch.uint8))

    bbox = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
    shifted_bbox = _translate_bbox(bbox, dx=1, dy=-1)
    assert shifted_bbox[0, 0].item() == pytest.approx(0.5 + 1.0 / 256.0)
    assert shifted_bbox[0, 1].item() == pytest.approx(0.5 - 1.0 / 256.0)
    assert torch.equal(shifted_bbox[:, 2:], bbox[:, 2:])


def _passing_shift_summary() -> dict[str, object]:
    directions = {
        "shift_dx_m1_dy_0": 2,
        "shift_dx_p1_dy_0": 2,
        "shift_dx_0_dy_m1": 0,
        "shift_dx_0_dy_p1": 0,
        "shift_dx_m1_dy_m1": 2,
        "shift_dx_m1_dy_p1": 2,
        "shift_dx_p1_dy_m1": 0,
        "shift_dx_p1_dy_p1": 0,
    }
    return {
        "rows": 607 * 9,
        "samples": 607,
        "replay_stable_samples": 606,
        "replay_stable_tp": 421,
        "replay_stable_fp": 185,
        "declaration_mismatches": [dict(EXPECTED_DECLARATION_EXCEPTION)],
        "any_class1_exit_count": 18,
        "tp_exit_count": 8,
        "restricted_fp_exit_count": 10,
        "fold_exit_counts": {"1": 4, "2": 4, "3": 4, "4": 6},
        "direction_exit_counts": directions,
        "cardinal_directions_with_exit": 2,
        "diagonal_directions_with_exit": 2,
        "focus_probability_span_median": 0.010,
        "focus_probability_span_p90": 0.030,
        "all_numeric_finite": True,
    }


def test_shift_signal_gate_is_conjunctive() -> None:
    summary = _passing_shift_summary()
    assert assess_shift_signal(summary)["passed"] is True
    summary["restricted_fp_exit_count"] = 3
    result = assess_shift_signal(summary)
    assert result["passed"] is False
    assert result["checks"]["at_least_4_restricted_fp_exits"] is False
