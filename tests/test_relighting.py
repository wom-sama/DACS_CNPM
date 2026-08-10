import torch

from trkh.training.relighting import balanced_polarities, relight_luminance


def test_balanced_polarities_are_reproducible_and_nearly_equal() -> None:
    left = balanced_polarities(
        9,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(17),
    )
    right = balanced_polarities(
        9,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(left, right)
    assert abs(int((left > 0).sum()) - int((left < 0).sum())) == 1


def test_relighting_preserves_unclipped_chroma() -> None:
    rgb = torch.full((4, 3, 8, 8), 0.45)
    rgb[:, 0] += 0.04
    rgb[:, 2] -= 0.03
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    relit_rgb = relight_luminance(
        (rgb - mean) / std,
        torch.tensor([-1.0, -1.0, 1.0, 1.0]),
    ) * std + mean
    torch.testing.assert_close(relit_rgb[:, 0] - relit_rgb[:, 1], rgb[:, 0] - rgb[:, 1])
    torch.testing.assert_close(relit_rgb[:, 2] - relit_rgb[:, 1], rgb[:, 2] - rgb[:, 1])
