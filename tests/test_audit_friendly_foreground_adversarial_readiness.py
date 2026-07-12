from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from trkh.tools.audit_friendly_foreground_adversarial_readiness import (
    ATTACK_EPSILON,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_KEEPER_VAL_CLASS1,
    EXPECTED_KEEPER_VAL_MACRO,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VAL_COUNT,
    assess_smoke_permission,
    assign_source_grouped_folds,
    boundary_margin_from_row,
    build_eroded_bbox_mask,
    class1_attack_margin,
    classify_row_cohort,
    combine_gradients,
    generate_friendly_adversarial_examples,
    gradient_cosine,
    select_cohort_indices,
    trainable_parameter_signature,
)
from trkh.training.train import (
    _friendly_foreground_adversarial_loss,
    _parse_friendly_adversarial_negative_classes,
    _select_friendly_adversarial_indices,
)


def _row(
    sample_index: int,
    source: str,
    target: int,
    prediction: int,
    probabilities: list[float],
) -> dict[str, object]:
    order = sorted(range(5), key=lambda index: probabilities[index], reverse=True)
    row: dict[str, object] = {
        "sample_index": sample_index,
        "source_stem": source,
        "target_index": target,
        "prediction_index": prediction,
        "top1_index": order[0],
        "top2_index": order[1],
    }
    for index, probability in enumerate(probabilities):
        row[f"prob_{index}"] = probability
    return row


def test_eroded_bbox_mask_intersects_valid_pixels() -> None:
    bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    valid[:, :, :2] = False
    mask = build_eroded_bbox_mask(bbox, valid, height=8, width=8, erode_ratio=0.25)
    assert mask.shape == (1, 1, 8, 8)
    assert not bool(mask[:, :, :, :2].any())
    assert int(mask.sum()) == 16


def test_eroded_bbox_mask_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="erode_ratio"):
        build_eroded_bbox_mask(
            torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
            torch.ones((1, 8, 8), dtype=torch.bool),
            height=8,
            width=8,
            erode_ratio=0.5,
        )


def test_class1_attack_margin_has_locked_directions() -> None:
    logits = torch.tensor(
        [
            [0.5, 0.8, 0.2, -1.0, 0.1],
            [0.7, 0.6, 0.1, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 0], dtype=torch.long)
    protect = class1_attack_margin(logits[:1], labels[:1], direction="protect")
    suppress = class1_attack_margin(logits[1:], labels[1:], direction="suppress")
    assert protect.item() == pytest.approx(-0.3)
    assert suppress.item() == pytest.approx(-0.1)


@pytest.mark.parametrize("direction,label", [("protect", 1), ("suppress", 0)])
def test_friendly_attack_stops_at_first_crossing_and_respects_mask(
    direction: str,
    label: int,
) -> None:
    images = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    mask[:, :, 2:6, 2:6] = True

    def forward_logits(candidate: torch.Tensor) -> torch.Tensor:
        red_score = candidate[:, 0].mean(dim=(1, 2)) * 100.0
        logits = torch.full((candidate.size(0), 5), -20.0, dtype=candidate.dtype)
        if direction == "protect":
            logits[:, 0] = red_score
            logits[:, 1] = 50.05
        else:
            logits[:, 0] = 50.05
            logits[:, 1] = red_score
        return logits

    result = generate_friendly_adversarial_examples(
        images=images,
        labels=torch.tensor([label]),
        attack_mask=mask,
        forward_logits=forward_logits,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        direction=direction,
    )
    assert bool(result.crossed.item())
    assert int(result.crossing_step.item()) == 1
    assert float(result.delta_rgb.abs().max()) <= ATTACK_EPSILON + 1e-7
    outside = result.delta_rgb * (~mask).to(dtype=result.delta_rgb.dtype)
    assert torch.equal(outside, torch.zeros_like(outside))


def test_friendly_attack_rejects_step_larger_than_epsilon() -> None:
    images = torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32)

    with pytest.raises(ValueError, match="cannot exceed epsilon"):
        generate_friendly_adversarial_examples(
            images=images,
            labels=torch.tensor([1]),
            attack_mask=torch.ones((1, 1, 4, 4), dtype=torch.bool),
            forward_logits=lambda value: torch.zeros((value.size(0), 5)),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            direction="protect",
            epsilon=1.0 / 255.0,
            step_size=2.0 / 255.0,
        )


def test_cohorts_and_margin_ranking_are_deterministic() -> None:
    protect = _row(3, "a", 1, 1, [0.20, 0.45, 0.25, 0.05, 0.05])
    positive_error = _row(4, "b", 1, 0, [0.45, 0.40, 0.05, 0.05, 0.05])
    suppress = _row(5, "c", 0, 0, [0.46, 0.44, 0.04, 0.03, 0.03])
    negative_error = _row(6, "d", 4, 1, [0.05, 0.48, 0.05, 0.02, 0.40])
    assert classify_row_cohort(protect) == "protect_correct"
    assert classify_row_cohort(positive_error) == "positive_error"
    assert classify_row_cohort(suppress) == "suppress_correct"
    assert classify_row_cohort(negative_error) == "negative_error"
    assert boundary_margin_from_row(suppress) == pytest.approx(0.02)
    selected = select_cohort_indices(
        [
            suppress,
            _row(2, "e", 2, 2, [0.05, 0.39, 0.40, 0.10, 0.06]),
        ],
        cohort="suppress_correct",
        cap=1,
    )
    assert selected == [2]


def test_source_grouped_folds_never_split_a_source() -> None:
    rows = []
    patterns = [
        (1, 1, [0.1, 0.7, 0.1, 0.05, 0.05]),
        (1, 0, [0.5, 0.4, 0.04, 0.03, 0.03]),
        (0, 0, [0.5, 0.4, 0.04, 0.03, 0.03]),
        (0, 1, [0.4, 0.5, 0.04, 0.03, 0.03]),
        (3, 3, [0.05, 0.05, 0.05, 0.8, 0.05]),
    ]
    for index in range(50):
        target, prediction, probabilities = patterns[index % len(patterns)]
        rows.append(
            _row(index, f"source_{index // 2:02d}", target, prediction, probabilities)
        )
    assignments = assign_source_grouped_folds(rows, folds=5, seed=20260712)
    assert len(assignments) == len(rows)
    by_source: dict[str, set[int]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_stem"]), set()).add(
            assignments[int(row["sample_index"])]
        )
    assert all(len(folds) == 1 for folds in by_source.values())


def test_gradient_helpers_and_parameter_signature() -> None:
    first = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0])}
    second = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0])}
    assert gradient_cosine(first, second) == pytest.approx(1.0)
    combined = combine_gradients(first, {"a": -first["a"], "b": first["b"]})
    assert torch.equal(combined["a"], torch.zeros(2))
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    parameters, signature = trainable_parameter_signature(model)
    assert len(parameters) == 4
    assert signature["numel"] == sum(parameter.numel() for _, parameter in parameters)
    assert len(str(signature["sha256"])) == 64


def _passing_gate_inputs():
    metrics = {
        "macro_f1": EXPECTED_KEEPER_VAL_MACRO,
        "per_class": [
            {"f1": 0.9},
            {"f1": EXPECTED_KEEPER_VAL_CLASS1},
            {"f1": 0.9},
            {"f1": 0.9},
            {"f1": 0.9},
        ],
    }
    attack = {
        "eligible_count": 10,
        "crossing_rate": 0.5,
        "max_delta_rgb": ATTACK_EPSILON,
        "max_outside_mask_delta": 0.0,
        "empty_mask_count": 0,
    }
    attacks = {
        split: {direction: dict(attack) for direction in ("protect", "suppress")}
        for split in ("train", "val")
    }
    gradients = {
        "folds": [
            {
                "complete": True,
                "protect_correction_cosine": 0.08,
                "suppress_correction_cosine": 0.09,
            }
            for _ in range(5)
        ],
        "transfer": {
            "protect_validation_correction_cosine": 0.03,
            "suppress_validation_correction_cosine": 0.04,
            "train_combined_clean_cosine": 0.11,
            "val_combined_clean_cosine": 0.01,
        },
    }
    return metrics, attacks, gradients


def test_smoke_gate_passes_only_when_every_locked_check_passes() -> None:
    metrics, attacks, gradients = _passing_gate_inputs()
    gate = assess_smoke_permission(
        preflight=False,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        train_support=EXPECTED_TRAIN_COUNT,
        val_support=EXPECTED_VAL_COUNT,
        source_overlap_count=0,
        val_metrics=metrics,
        attacks=attacks,
        gradients=gradients,
    )
    assert gate["smoke_permission"] is True
    assert gate["passed"] == gate["total"]

    attacks["val"]["protect"]["crossing_rate"] = 0.09
    failed = assess_smoke_permission(
        preflight=False,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        train_support=EXPECTED_TRAIN_COUNT,
        val_support=EXPECTED_VAL_COUNT,
        source_overlap_count=0,
        val_metrics=metrics,
        attacks=attacks,
        gradients=gradients,
    )
    assert failed["smoke_permission"] is False
    assert failed["passed"] == failed["total"] - 1


def test_gradient_cosine_returns_nan_for_zero_snapshot() -> None:
    value = gradient_cosine({"a": torch.zeros(2)}, {"a": torch.ones(2)})
    assert math.isnan(value)


def test_training_selector_uses_clean_correct_nearest_boundary_rows() -> None:
    logits = torch.tensor(
        [
            [4.90, 5.00, -2.0, -2.0, -2.0],
            [4.98, 5.00, -2.0, -2.0, -2.0],
            [5.10, 5.00, -2.0, -2.0, -2.0],
            [5.02, 5.00, -2.0, -2.0, -2.0],
            [4.90, 5.00, -2.0, -2.0, -2.0],
        ]
    )
    targets = torch.tensor([1, 1, 0, 0, 0])
    protect, suppress = _select_friendly_adversarial_indices(
        logits,
        targets,
        focus_class=1,
        negative_classes=(0, 2, 4),
        max_per_direction=1,
    )
    assert protect.tolist() == [1]
    assert suppress.tolist() == [3]


def test_training_negative_class_parser_is_fail_closed() -> None:
    assert _parse_friendly_adversarial_negative_classes(
        "0,2,4,2", num_classes=5, focus_class=1
    ) == [0, 2, 4]
    with pytest.raises(ValueError, match="focus class"):
        _parse_friendly_adversarial_negative_classes(
            "0,1,2", num_classes=5, focus_class=1
        )


class _ToyBoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.state_probe = nn.Dropout(p=0.5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0].mean(dim=(1, 2)) * self.scale
        logits = torch.full(
            (images.size(0), 5),
            -20.0,
            device=images.device,
            dtype=images.dtype,
        )
        logits[:, 0] = red
        logits[:, 1] = 5.0
        return logits


def test_training_friendly_loss_is_active_and_restores_module_modes() -> None:
    model = _ToyBoundaryModel()
    model.train()
    model.state_probe.eval()
    images = torch.stack(
        [
            torch.full((3, 8, 8), 0.498),
            torch.full((3, 8, 8), 0.497),
            torch.full((3, 8, 8), 0.502),
            torch.full((3, 8, 8), 0.503),
        ]
    )
    targets = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    clean_logits = model(images)
    bbox = torch.tensor([[0.5, 0.5, 1.0, 1.0]]).repeat(4, 1)
    valid_mask = torch.ones((4, 8, 8), dtype=torch.bool)
    loss, stats = _friendly_foreground_adversarial_loss(
        model=model,
        images=images,
        logits=clean_logits,
        target_indices=targets,
        image_valid_mask=valid_mask,
        bbox_metadata=bbox,
        crop_bbox_metadata=bbox,
        bbox_token_prior_metadata=bbox,
        device=torch.device("cpu"),
        amp=False,
        input_mean=(0.0, 0.0, 0.0),
        input_std=(1.0, 1.0, 1.0),
        epsilon=2.0 / 255.0,
        step_size=1.0 / 255.0,
        steps=2,
        bbox_erode_ratio=0.10,
        focus_class=1,
        negative_classes="0,2,4",
        max_per_direction=8,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert stats["friendly_adversarial_protect_count"] == 2.0
    assert stats["friendly_adversarial_suppress_count"] == 2.0
    assert stats["friendly_adversarial_direction_count"] == 2.0
    assert stats["friendly_adversarial_max_delta_rgb"] <= 2.0 / 255.0 + 1e-7
    assert model.training is True
    assert model.state_probe.training is False
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_v8_launcher_wires_friendly_adversarial_flags_fail_closed() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    assert "[double]$FriendlyAdversarialLossWeight = 0.0" in script
    assert '"--friendly-adversarial-loss-weight", "$FriendlyAdversarialLossWeight"' in script
    assert '"--friendly-adversarial-negative-classes", "$FriendlyAdversarialNegativeClasses"' in script
    assert "friendly_adversarial_loss_weight = $FriendlyAdversarialLossWeight" in script
    assert "if ($FriendlyAdversarialStartEpoch -lt 1)" in script
