import math

import pytest
import torch

from trkh.training.representation_self_challenging import (
    locate_protected_rsc,
    select_positive_drop_rows,
    signed_top_gradient_mask,
    true_logit_gradient,
)


def test_true_logit_gradient_matches_linear_equation() -> None:
    representation = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0], [2.0, -3.0, 0.25]]
    )
    logits = representation @ weight.T
    targets = torch.tensor([0, 2])

    gradient = true_logit_gradient(
        logits=logits,
        targets=targets,
        representation=representation,
    )

    assert torch.equal(gradient, weight.index_select(0, targets))


def test_signed_top_gradient_mask_uses_signed_values_and_protection() -> None:
    gradient = torch.tensor(
        [
            [-100.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        ]
    )
    mask, drop_count = signed_top_gradient_mask(
        gradient,
        drop_fraction=1.0 / 3.0,
        protected_rows=torch.tensor([False, True]),
    )

    assert drop_count == 2
    assert torch.equal(mask[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(mask[1], torch.ones(6))


def test_positive_drop_selection_is_stable_and_limited() -> None:
    clean_logits = torch.log(
        torch.tensor(
            [
                [0.80, 0.20],
                [0.70, 0.30],
                [0.60, 0.40],
                [0.55, 0.45],
                [0.40, 0.60],
            ]
        )
    )
    challenged_logits = torch.log(
        torch.tensor(
            [
                [0.50, 0.50],
                [0.40, 0.60],
                [0.45, 0.55],
                [0.55, 0.45],
                [0.20, 0.80],
            ]
        )
    )
    targets = torch.tensor([0, 0, 0, 0, 1])

    selected, drop, eligible, positive, limit = select_positive_drop_rows(
        clean_logits=clean_logits,
        challenged_logits=challenged_logits,
        targets=targets,
        eligible_classes=(0,),
        batch_fraction=1.0 / 3.0,
        minimum_drop=1e-4,
    )

    assert limit == math.ceil(4.0 / 3.0)
    assert torch.equal(eligible, torch.tensor([True, True, True, True, False]))
    assert torch.equal(positive, torch.tensor([True, True, True, False, False]))
    assert torch.equal(selected, torch.tensor([True, True, False, False, False]))
    assert drop[0] == pytest.approx(0.2999, abs=1e-6)


def test_locate_protected_rsc_masks_only_selected_nonfocus_rows() -> None:
    representation = torch.tensor(
        [
            [2.0, 1.0, 0.5, -1.0, 0.1, 0.2],
            [1.0, 3.0, -0.5, 0.5, 0.2, -0.1],
            [0.5, -1.0, 2.0, 1.0, -0.2, 0.3],
            [1.5, 0.2, 0.1, 2.0, 0.5, -0.4],
        ],
        requires_grad=True,
    )
    weight = torch.tensor(
        [
            [2.0, 1.5, 1.0, 0.5, 0.0, -0.5],
            [1.0, 2.0, 1.5, 0.0, 0.5, -1.0],
            [0.5, 0.0, 2.0, 1.5, 1.0, -0.5],
        ]
    )
    targets = torch.tensor([0, 1, 2, 0])

    def forward(value: torch.Tensor) -> torch.Tensor:
        return value @ weight.T

    clean_logits = forward(representation)
    result = locate_protected_rsc(
        representation=representation,
        clean_logits=clean_logits,
        targets=targets,
        forward_from_representation=forward,
        focus_class=1,
        eligible_classes=(0, 2),
        drop_fraction=1.0 / 3.0,
        batch_fraction=1.0 / 3.0,
        minimum_drop=0.0,
    )

    assert result.drop_count == 2
    assert result.selection_limit == 1
    assert int(result.selected.sum()) == 1
    assert not bool(result.selected[1])
    assert torch.equal(result.preliminary_mask[1], torch.ones(6))
    assert torch.equal(result.final_mask[1], torch.ones(6))
    assert torch.equal(
        (result.final_mask == 0).sum(dim=1),
        result.selected.long() * 2,
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_signed_top_gradient_mask_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        signed_top_gradient_mask(torch.ones(2, 4), drop_fraction=fraction)
