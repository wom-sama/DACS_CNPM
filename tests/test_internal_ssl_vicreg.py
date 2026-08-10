import torch
import pytest

from trkh.tools.pretrain_internal_barlow import barlow_twins_loss, vicreg_loss


def test_vicreg_loss_is_finite_and_reports_parts() -> None:
    torch.manual_seed(7)
    z_a = torch.randn(8, 16, requires_grad=True)
    z_b = z_a.detach().clone() + 0.05 * torch.randn(8, 16)

    loss, parts = vicreg_loss(z_a, z_b)

    assert torch.isfinite(loss)
    assert set(parts) == {"invariance", "variance", "covariance"}
    assert all(torch.isfinite(value) for value in parts.values())
    loss.backward()
    assert z_a.grad is not None
    assert torch.isfinite(z_a.grad).all()


def test_vicreg_penalizes_collapsed_projection_more_than_varied_projection() -> None:
    collapsed_a = torch.zeros(8, 16)
    collapsed_b = torch.zeros(8, 16)
    varied_a = torch.randn(8, 16)
    varied_b = varied_a + 0.01 * torch.randn(8, 16)

    collapsed_loss, collapsed_parts = vicreg_loss(collapsed_a, collapsed_b)
    varied_loss, varied_parts = vicreg_loss(varied_a, varied_b)

    assert collapsed_parts["variance"] > varied_parts["variance"]
    assert collapsed_loss > varied_loss


def test_ssl_losses_validate_projection_shape() -> None:
    z_a = torch.randn(4, 8)
    z_b = torch.randn(4, 7)

    with pytest.raises(ValueError):
        _ = barlow_twins_loss(z_a, z_b)
    with pytest.raises(ValueError):
        _ = vicreg_loss(z_a, z_b)
