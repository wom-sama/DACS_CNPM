import pytest
import torch

from trkh.models.model import VisionTransformerWithRegisters
from trkh.training.train import (
    _build_pretrained_distillation_teacher,
    _late_member_precision_rule_distillation_loss_from_features,
    _late_member_specialist_losses_from_features,
)


def _logits_from_focus_probability(probability: float) -> list[float]:
    probability = float(probability)
    focus_logit = torch.log(
        torch.tensor(4.0 * probability / (1.0 - probability))
    ).item()
    return [0.0, focus_logit, 0.0, 0.0, 0.0]


def test_late_member_directional_loss_selects_teacher_complements() -> None:
    primary = torch.tensor(
        [
            _logits_from_focus_probability(0.30),
            _logits_from_focus_probability(0.42),
            _logits_from_focus_probability(0.55),
            _logits_from_focus_probability(0.20),
        ],
        requires_grad=True,
    )
    candidate = torch.tensor(
        [
            _logits_from_focus_probability(0.42),
            _logits_from_focus_probability(0.30),
            _logits_from_focus_probability(0.30),
            _logits_from_focus_probability(0.34),
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            _logits_from_focus_probability(0.72),
            _logits_from_focus_probability(0.12),
            _logits_from_focus_probability(0.66),
            _logits_from_focus_probability(0.08),
        ],
        requires_grad=True,
    )
    labels = torch.tensor([1, 0, 1, 4])

    auxiliary, directional, preservation, stats = (
        _late_member_specialist_losses_from_features(
            features={
                "late_member_primary_logits": primary,
                "late_member_logits": candidate,
            },
            teacher_logits=teacher,
            hard_labels=labels,
            targets=labels,
            focus_class=1,
            negative_classes="0,2,4",
            min_teacher_gap=0.04,
            hard_target_blend=0.25,
            preservation_slack=0.01,
        )
    )

    assert auxiliary.item() > 0.0
    assert directional.item() > 0.0
    assert preservation.item() > 0.0
    assert stats["positive_count"] == pytest.approx(2.0)
    assert stats["negative_count"] == pytest.approx(2.0)
    assert stats["protect_positive_fraction"] == pytest.approx(0.50)
    assert stats["protect_negative_fraction"] == pytest.approx(0.25)

    (directional + preservation).backward()
    assert candidate.grad is not None
    assert float(candidate.grad.abs().sum()) > 0.0
    assert primary.grad is None
    assert teacher.grad is None


def test_late_member_preservation_is_zero_inside_primary_slack() -> None:
    primary = torch.tensor(
        [
            _logits_from_focus_probability(0.70),
            _logits_from_focus_probability(0.20),
        ]
    )
    candidate = torch.tensor(
        [
            _logits_from_focus_probability(0.72),
            _logits_from_focus_probability(0.18),
        ],
        requires_grad=True,
    )
    labels = torch.tensor([1, 4])

    _, _, preservation, _ = _late_member_specialist_losses_from_features(
        features={
            "late_member_primary_logits": primary,
            "late_member_logits": candidate,
        },
        teacher_logits=None,
        hard_labels=labels,
        targets=labels,
        focus_class=1,
        negative_classes="0,2,4",
        min_teacher_gap=0.04,
        hard_target_blend=0.25,
        preservation_slack=0.01,
    )

    assert preservation.item() == pytest.approx(0.0)


@pytest.mark.parametrize("distance", ["kl", "logit_l2"])
def test_late_member_precision_rule_target_matches_model_fusion(distance: str) -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        late_member_branch=True,
        late_member_fork_after_block=1,
        late_member_candidate_weight=0.40,
        late_member_focus_class=1,
        late_member_focus_margin_offset=0.034,
    )
    primary = torch.tensor(
        [[2.0, 1.0, 0.5, -0.2, 0.1], [0.2, 1.7, 0.8, -0.4, 0.3]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.5, 2.1, 0.2, -0.1, 0.0], [1.3, 0.4, 0.9, -0.2, 0.1]],
        requires_grad=True,
    )
    fused = model.fuse_late_member_logits(primary, teacher).detach().requires_grad_(True)
    labels = torch.tensor([0, 1])

    loss, stats = _late_member_precision_rule_distillation_loss_from_features(
        features={
            "late_member_primary_logits": primary,
            "late_member_fused_logits": fused,
        },
        teacher_logits=teacher,
        hard_labels=labels,
        targets=labels,
        candidate_weight=0.40,
        focus_class=1,
        focus_margin_offset=0.034,
        correct_boost=5.0,
        distance=distance,
    )

    assert abs(loss.item()) < 1e-6
    assert stats["precision_rule_dense_fraction"] == pytest.approx(1.0)
    assert stats["precision_rule_mean_absolute_probability_error"] < 1e-6


def test_late_member_precision_rule_loss_updates_only_fused_path() -> None:
    primary = torch.tensor(
        [[2.0, 0.1, 0.3, -0.2, 0.0], [0.1, 2.0, 0.5, -0.3, 0.2]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.2, 2.2, 0.1, -0.1, 0.0], [1.8, 0.3, 0.4, -0.2, 0.1]],
        requires_grad=True,
    )
    fused = primary.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 1])

    loss, stats = _late_member_precision_rule_distillation_loss_from_features(
        features={
            "late_member_primary_logits": primary,
            "late_member_fused_logits": fused,
        },
        teacher_logits=teacher,
        hard_labels=labels,
        targets=labels,
        candidate_weight=0.40,
        focus_class=1,
        focus_margin_offset=0.034,
        correct_boost=5.0,
        distance="logit_l2",
    )
    loss.backward()

    assert loss.item() > 0.0
    assert fused.grad is not None
    assert float(fused.grad.abs().sum()) > 0.0
    assert primary.grad is None
    assert teacher.grad is None
    assert stats["precision_rule_primary_correct_fraction"] == pytest.approx(1.0)


def test_trkh_distillation_teacher_requires_class_names(tmp_path) -> None:
    checkpoint_path = tmp_path / "missing_classes.pt"
    torch.save({"model_config": {}, "model_state": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="class_names"):
        _build_pretrained_distillation_teacher(
            checkpoint_path=checkpoint_path,
            target_class_names=["0", "1", "2", "3", "4"],
            device=torch.device("cpu"),
        )


def test_trkh_checkpoint_loads_as_frozen_distillation_teacher(tmp_path) -> None:
    model = VisionTransformerWithRegisters(
        image_size=32,
        patch_size=8,
        use_cnn_stem=False,
        num_classes=5,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_registers=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save(
        {
            "model_config": {
                "model_type": "vit_registers",
                "image_size": 32,
                "patch_size": 8,
                "use_cnn_stem": False,
                "embed_dim": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "num_registers": 2,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "drop_path_rate": 0.0,
            },
            "model_state": model.state_dict(),
            "class_names": ["0", "1", "2", "3", "4"],
        },
        checkpoint_path,
    )

    teacher, class_indices, summary = _build_pretrained_distillation_teacher(
        checkpoint_path=checkpoint_path,
        target_class_names=["0", "1", "2", "3", "4"],
        device=torch.device("cpu"),
    )

    assert summary["backend"] == "trkh"
    assert summary["trainable_parameters"] == 0
    assert class_indices.tolist() == [0, 1, 2, 3, 4]
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
