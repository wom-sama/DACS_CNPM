from __future__ import annotations

import copy

import pytest
import timm
import torch
import torch.nn.functional as F

from trkh.models.swiftformer_surface_b16 import (
    SURFACE_BRANCH_LAYER_SCALE_INIT,
    SURFACE_MEAN_CONTROL_B16_MODE,
    SURFACE_OFF_B16_MODE,
    SURFACE_SPATIAL_B16_MODE,
    RELATION_SMOOTH_L1_BETA,
    SurfaceResidualBranchB16,
    SwiftFormerSurfaceB16,
    SwiftFormerSurfaceB16ContractError,
    cosine_neighbor_relation_field,
    surface_relation_distillation_loss,
)


def _stock() -> torch.nn.Module:
    return timm.create_model(
        "swiftformer_xs",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )


def _paired_models() -> tuple[SwiftFormerSurfaceB16, SwiftFormerSurfaceB16]:
    torch.manual_seed(1601)
    shared = _stock()
    torch.manual_seed(1602)
    candidate = SwiftFormerSurfaceB16(
        copy.deepcopy(shared), SURFACE_SPATIAL_B16_MODE
    )
    torch.manual_seed(1602)
    control = SwiftFormerSurfaceB16(
        copy.deepcopy(shared), SURFACE_MEAN_CONTROL_B16_MODE
    )
    return candidate, control


def test_b16_shapes_and_stock_xs_stage_geometry() -> None:
    torch.manual_seed(1603)
    model = SwiftFormerSurfaceB16(_stock(), SURFACE_SPATIAL_B16_MODE).eval()
    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits, s2 = model(image, return_s2=True)
        final = model.forward_features(image)

    assert logits.shape == (1, 5)
    assert s2.shape == (1, 112, 14, 14)
    assert final.shape == (1, 220, 7, 7)
    assert torch.isfinite(logits).all()


def test_b16_branch_off_is_bit_exact_stock_eval_forward() -> None:
    torch.manual_seed(1605)
    stock = _stock().eval()
    model = SwiftFormerSurfaceB16(
        copy.deepcopy(stock), SURFACE_OFF_B16_MODE
    ).eval()
    image = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        expected = stock(image)
        actual = model(image)

    assert torch.equal(actual, expected)


def test_b16_candidate_and_control_have_identical_initial_state_and_capacity() -> None:
    candidate, control = _paired_models()
    candidate_state = candidate.state_dict()
    control_state = control.state_dict()

    assert candidate_state.keys() == control_state.keys()
    assert all(
        torch.equal(candidate_state[name], control_state[name])
        for name in candidate_state
    )
    assert {
        name: tuple(parameter.shape)
        for name, parameter in candidate.named_parameters()
    } == {
        name: tuple(parameter.shape)
        for name, parameter in control.named_parameters()
    }
    assert sum(parameter.numel() for parameter in candidate.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )


def test_b16_surface_branch_scale_and_spatial_control_do_not_collapse() -> None:
    torch.manual_seed(1607)
    branch = SurfaceResidualBranchB16().eval()
    s2 = torch.randn(2, 112, 14, 14)

    with torch.no_grad():
        spatial = branch(s2)
        control = branch(s2, mean_broadcast=True)

    assert spatial.shape == control.shape == (2, 220, 7, 7)
    assert torch.equal(
        branch.layer_scale,
        torch.full_like(branch.layer_scale, SURFACE_BRANCH_LAYER_SCALE_INIT),
    )
    assert torch.isfinite(spatial).all()
    assert torch.count_nonzero(spatial) > 0
    assert float(spatial.var(dim=(2, 3)).mean()) > 0.0
    assert torch.equal(control, control[:, :, :1, :1].expand_as(control))
    assert torch.equal(
        control,
        spatial.mean(dim=(2, 3), keepdim=True).expand_as(control),
    )
    assert not torch.equal(spatial, control)


def test_b16_active_candidate_has_finite_nonzero_branch_gradients() -> None:
    torch.manual_seed(1611)
    model = SwiftFormerSurfaceB16(_stock(), SURFACE_SPATIAL_B16_MODE).train()
    image = torch.randn(2, 3, 64, 64)

    logits = model(image)
    loss = logits.float().square().mean() + logits.float().mean()
    loss.backward()

    branch_parameters = list(model.surface_branch.parameters())
    assert branch_parameters
    assert all(parameter.grad is not None for parameter in branch_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in branch_parameters
        if parameter.grad is not None
    )
    assert all(
        int(torch.count_nonzero(parameter.grad)) == parameter.numel()
        for parameter in branch_parameters
        if parameter.grad is not None
    )


def test_b16_relation_field_is_exact_scale_invariant_and_noncollapsed() -> None:
    features = torch.tensor(
        [
            [
                [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        ]
    )
    field = cosine_neighbor_relation_field(features)
    normalized = F.normalize(features.float(), p=2.0, dim=1, eps=1.0e-6)
    expected_right = (
        normalized[:, :, :, :-1] * normalized[:, :, :, 1:]
    ).sum(dim=1)
    expected_down = (
        normalized[:, :, :-1, :] * normalized[:, :, 1:, :]
    ).sum(dim=1)
    expected = torch.cat(
        (expected_right.flatten(1), expected_down.flatten(1)), dim=1
    )

    assert field.shape == (1, 12)
    assert torch.equal(field, expected)
    assert torch.allclose(
        cosine_neighbor_relation_field(7.25 * features),
        field,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert float(field.std()) > 0.0
    assert not torch.allclose(field, torch.ones_like(field))


def test_b16_relation_field_keeps_all_edges_and_is_horizontal_flip_equivariant() -> None:
    torch.manual_seed(1612)
    features = torch.randn(2, 7, 14, 14)
    field = cosine_neighbor_relation_field(features)
    flipped = cosine_neighbor_relation_field(features.flip(dims=(3,)))
    horizontal_count = 14 * 13
    right = field[:, :horizontal_count].reshape(2, 14, 13)
    down = field[:, horizontal_count:].reshape(2, 13, 14)
    flipped_right = flipped[:, :horizontal_count].reshape(2, 14, 13)
    flipped_down = flipped[:, horizontal_count:].reshape(2, 13, 14)

    assert field.shape == (2, 364)
    assert torch.allclose(
        flipped_right, right.flip(dims=(2,)), atol=1.0e-7, rtol=1.0e-6
    )
    assert torch.allclose(
        flipped_down, down.flip(dims=(2,)), atol=1.0e-7, rtol=1.0e-6
    )


def test_b16_relation_loss_is_finite_aligned_and_derangement_sensitive() -> None:
    torch.manual_seed(1613)
    latent_teacher = torch.randn(2, 4, 16, 16)
    raw_teacher = latent_teacher.repeat(1, 96, 1, 1).requires_grad_()
    resized_latent = F.interpolate(
        latent_teacher,
        size=(14, 14),
        mode="bilinear",
        align_corners=False,
    )
    student = resized_latent.repeat(1, 28, 1, 1).detach().requires_grad_()

    aligned = surface_relation_distillation_loss(student, raw_teacher)
    deranged = surface_relation_distillation_loss(student, raw_teacher.flip(0))
    aligned.backward()

    assert aligned.ndim == 0
    assert torch.isfinite(aligned)
    assert torch.isfinite(deranged)
    assert float(aligned) < 1.0e-11
    assert float(deranged) > float(aligned) + 1.0e-5
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert raw_teacher.grad is None


def test_b16_relation_loss_matches_exact_fp32_smooth_l1_formula() -> None:
    torch.manual_seed(1615)
    student = torch.randn(2, 7, 5, 6, dtype=torch.float16)
    teacher = torch.randn(2, 11, 8, 9, dtype=torch.float16)
    resized_teacher = F.interpolate(
        teacher.detach().float(),
        size=(5, 6),
        mode="bilinear",
        align_corners=False,
    )
    expected = F.smooth_l1_loss(
        cosine_neighbor_relation_field(student),
        cosine_neighbor_relation_field(resized_teacher),
        beta=RELATION_SMOOTH_L1_BETA,
        reduction="mean",
    )

    actual = surface_relation_distillation_loss(student, teacher)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    with pytest.raises(SwiftFormerSurfaceB16ContractError, match="beta"):
        surface_relation_distillation_loss(student, teacher, beta=0.0)
    with pytest.raises(SwiftFormerSurfaceB16ContractError, match="beta"):
        surface_relation_distillation_loss(student, teacher, beta=float("nan"))


def test_b16_registers_no_teacher_parameters_or_buffers() -> None:
    model = SwiftFormerSurfaceB16(_stock(), SURFACE_SPATIAL_B16_MODE)
    names = [name.lower() for name, _ in model.named_parameters()]
    names.extend(name.lower() for name, _ in model.named_buffers())

    assert names
    assert not hasattr(model, "teacher")
    assert all("teacher" not in name and "dino" not in name for name in names)


@pytest.mark.parametrize(
    "mode",
    (SURFACE_SPATIAL_B16_MODE, SURFACE_MEAN_CONTROL_B16_MODE, SURFACE_OFF_B16_MODE),
)
def test_b16_state_dict_strict_round_trip(mode: str) -> None:
    torch.manual_seed(1617)
    source = SwiftFormerSurfaceB16(_stock(), mode).eval()
    state = copy.deepcopy(source.state_dict())
    torch.manual_seed(1619)
    restored = SwiftFormerSurfaceB16(_stock(), mode).eval()

    incompatible = restored.load_state_dict(state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert all(
        torch.equal(source.state_dict()[name], restored.state_dict()[name])
        for name in state
    )


def test_b16_rejects_invalid_relation_geometry() -> None:
    with pytest.raises(SwiftFormerSurfaceB16ContractError, match="H,W>=2"):
        cosine_neighbor_relation_field(torch.randn(1, 8, 1, 4))
    with pytest.raises(SwiftFormerSurfaceB16ContractError, match="batch sizes"):
        surface_relation_distillation_loss(
            torch.randn(2, 112, 14, 14),
            torch.randn(1, 384, 16, 16),
        )
