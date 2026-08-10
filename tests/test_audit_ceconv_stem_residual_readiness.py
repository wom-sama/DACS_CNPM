from __future__ import annotations

import os

import torch

from trkh.models.model import ConvStemBlock
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    BATCH_SIZE,
    EXPECTED_HARD_ROWS,
    EXPECTED_REFERENCE_ROWS,
    HALF_BATCH,
    ROTATIONS,
    TRAIN_STEPS,
    CEConvStemResidual,
    assess_stage_a,
    balanced_boundary_order,
    hue_rotation_matrix,
    hue_rotation_powers,
    official_equation_replay,
    parse_args,
    transform_input_filter,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 32
    assert args.rotations == ROTATIONS == 3
    assert args.max_train_batches == TRAIN_STEPS == 60
    assert args.learning_rate == 5e-4
    assert args.weight_decay == 0.01


def test_hue_rotation_is_proper_rgb_diagonal_cycle() -> None:
    rotation = hue_rotation_matrix(ROTATIONS)
    identity = torch.eye(3)
    diagonal = torch.ones(3) / (3.0**0.5)
    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-6, rtol=0.0)
    assert torch.allclose(rotation @ diagonal, diagonal, atol=1e-6, rtol=0.0)
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(torch.matrix_power(rotation, ROTATIONS), identity, atol=1e-6)


def test_filter_transform_matches_official_equation_and_identity_control() -> None:
    weights = torch.linspace(-0.4, 0.5, steps=4 * 3 * 1 * 3 * 3).reshape(
        4, 3, 1, 3, 3
    )
    candidate = transform_input_filter(weights, hue_rotation_powers())
    replay = official_equation_replay(weights)
    control = transform_input_filter(
        weights,
        hue_rotation_powers(identity_control=True),
    )
    assert torch.equal(candidate, replay)
    assert torch.equal(candidate[:, 0, :, 0], weights[:, :, 0])
    assert torch.equal(control[:, 0], control[:, 1])
    assert torch.equal(control[:, 1], control[:, 2])
    assert not torch.equal(candidate, control)


def test_zero_gate_preserves_native_block_and_hue_cycle_is_distinct() -> None:
    torch.manual_seed(17)
    native = ConvStemBlock(3, 4, pooling_mode="max").eval()
    candidate = CEConvStemResidual(native, identity_control=False).eval()
    control = CEConvStemResidual(native, identity_control=True).eval()
    with torch.no_grad():
        candidate.branch_weight.copy_(
            torch.linspace(-0.3, 0.4, steps=candidate.branch_weight.numel()).reshape_as(
                candidate.branch_weight
            )
        )
        control.branch_weight.copy_(candidate.branch_weight)
    images = torch.linspace(-1.0, 1.0, steps=2 * 3 * 17 * 19).reshape(2, 3, 17, 19)
    with torch.inference_mode():
        expected = native(images)
        assert torch.equal(candidate(images), expected)
        assert torch.equal(control(images), expected)
        rotation = hue_rotation_matrix()
        rotated = torch.einsum("ij,bjhw->bihw", rotation, images)
        groups = candidate.pre_norm_groups(images)
        rotated_groups = candidate.pre_norm_groups(rotated)
        candidate_error = min(
            float(
                (rotated_groups - torch.roll(groups, shifts=shift, dims=2))
                .abs()
                .max()
            )
            for shift in (-1, 1)
        )
        control_groups = control.pre_norm_groups(images)
        control_rotated = control.pre_norm_groups(rotated)
        control_error = min(
            float(
                (control_rotated - torch.roll(control_groups, shifts=shift, dims=2))
                .abs()
                .max()
            )
            for shift in (-1, 1)
        )
    assert candidate_error <= 1e-5
    assert control_error >= 1e-4


def test_balanced_boundary_order_is_deterministic_and_exact() -> None:
    reference = list(range(EXPECTED_REFERENCE_ROWS))
    hard = list(range(1000, 1000 + EXPECTED_HARD_ROWS))
    first = balanced_boundary_order(reference, hard)
    second = balanced_boundary_order(reference, hard)
    assert first == second
    assert len(first) == TRAIN_STEPS * BATCH_SIZE
    reference_set = set(reference)
    hard_set = set(hard)
    for start in range(0, len(first), BATCH_SIZE):
        batch = first[start : start + BATCH_SIZE]
        assert sum(value in reference_set for value in batch) == HALF_BATCH
        assert sum(value in hard_set for value in batch) == HALF_BATCH


def _comparison(
    *,
    macro: float,
    f1: float,
    precision: float,
    recall: float,
    fp_reduction: int,
    corrections: int,
    harms: int,
    breaks: int = 0,
    rescues: int = 0,
) -> dict[str, object]:
    return {
        "delta": {
            "macro_f1": macro,
            "class1_f1": f1,
            "class1_precision": precision,
            "class1_recall": recall,
        },
        "transitions": {
            "focus_tp_break": breaks,
            "focus_fn_rescue": rescues,
            "restricted_focus_fp_reduction": fp_reduction,
            "candidate_correction": corrections,
            "candidate_harm": harms,
            "new_nonfocus_3_to_2_harms": 0,
            "focus_fp_remove_correct": max(0, fp_reduction),
            "focus_fp_create": 0,
        },
        "maximum_nonfocus_f1_drop": 0.002,
    }


def _passing_comparisons() -> dict[str, dict[str, object]]:
    output = {}
    for condition in ("clean", "lighting_dim", "lighting_bright", "low_contrast"):
        output[condition] = {
            "raw_control": _comparison(
                macro=0.0,
                f1=0.0,
                precision=0.0,
                recall=0.0,
                fp_reduction=0,
                corrections=0,
                harms=0,
            ),
            "raw_candidate": _comparison(
                macro=0.003,
                f1=0.006,
                precision=0.010,
                recall=0.0,
                fp_reduction=3,
                corrections=5,
                harms=1,
            ),
            "control_candidate": _comparison(
                macro=0.002,
                f1=0.004,
                precision=0.006,
                recall=0.0,
                fp_reduction=2,
                corrections=4,
                harms=1,
            ),
        }
    return output


def test_gate_accepts_safe_gain_and_rejects_tp_suppression() -> None:
    comparisons = _passing_comparisons()
    result = assess_stage_a(
        structural_checks={"structure": True},
        comparisons=comparisons,
    )
    assert result["stage_b_authorized"] is True
    assert result["failed_checks"] == []

    comparisons = _passing_comparisons()
    comparisons["clean"]["raw_candidate"] = _comparison(
        macro=0.003,
        f1=0.006,
        precision=0.020,
        recall=-0.004,
        fp_reduction=6,
        corrections=8,
        harms=2,
        breaks=2,
        rescues=0,
    )
    result = assess_stage_a(
        structural_checks={"structure": True},
        comparisons=comparisons,
    )
    assert result["stage_b_authorized"] is False
    assert "zero_net_raw_class1_tp_break" in result["failed_checks"]

