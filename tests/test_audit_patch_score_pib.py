import torch

from trkh.tools.audit_patch_score_pib import (
    _compute_patch_score_statistics,
    _interpret_signal,
    _summarize_group_rows,
)


def test_compute_patch_score_statistics_tracks_topk_foreground_mass() -> None:
    patches = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    prior = torch.tensor([[0.9, 0.0, 0.7], [0.0, 0.8, 0.6]], dtype=torch.float32)

    stats = _compute_patch_score_statistics(
        patches=patches,
        query=query,
        bbox_prior=prior,
        foreground_threshold=0.5,
        background_threshold=0.2,
        top_k_values=[1, 2],
    )

    assert torch.allclose(stats["top1_point_in_box"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(stats["top2_foreground_fraction"], torch.tensor([1.0, 0.5]))
    assert stats["foreground_minus_background_score"][0] > 0


def test_compute_patch_score_statistics_respects_invalid_tokens() -> None:
    patches = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    query = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    prior = torch.tensor([[0.0, 0.9]], dtype=torch.float32)
    valid = torch.tensor([[False, True]])

    stats = _compute_patch_score_statistics(
        patches=patches,
        query=query,
        bbox_prior=prior,
        valid_mask=valid,
        foreground_threshold=0.5,
        background_threshold=0.2,
        top_k_values=[1],
    )

    assert torch.allclose(stats["top1_prior"], torch.tensor([0.9]))
    assert torch.allclose(stats["top1_point_in_box"], torch.tensor([1.0]))
    assert torch.allclose(stats["valid_token_count"], torch.tensor([1.0]))


def test_group_summary_and_interpretation_flag_background_shortcut() -> None:
    rows = [
        {
            "correct": 0,
            "target_index": 1,
            "pred_index": 2,
            "transition": "1->2",
            "top1_point_in_box": 0.0,
            "top1_prior": 0.1,
            "top3_foreground_fraction": 0.33,
            "foreground_minus_background_score": -0.2,
            "foreground_score_mean": 0.1,
            "background_score_mean": 0.3,
        },
        {
            "correct": 1,
            "target_index": 1,
            "pred_index": 1,
            "transition": "1->1",
            "top1_point_in_box": 1.0,
            "top1_prior": 0.8,
            "top3_foreground_fraction": 0.67,
            "foreground_minus_background_score": 0.2,
            "foreground_score_mean": 0.4,
            "background_score_mean": 0.2,
        },
    ]

    summary = _summarize_group_rows(
        rows,
        group_name="all",
        group_value="all",
        top_k_values=[1, 3],
    )
    assert summary["support"] == 2
    assert summary["top1_point_in_box_mean"] == 0.5

    group_summaries = [
        summary,
        _summarize_group_rows(
            [rows[0]],
            group_name="correctness",
            group_value="incorrect",
            top_k_values=[1, 3],
        ),
        _summarize_group_rows(
            rows,
            group_name="target_class",
            group_value="1:1",
            top_k_values=[1, 3],
        ),
    ]
    interpretation = _interpret_signal(group_summaries=group_summaries, top_k_values=[1, 3])
    assert interpretation["background_shortcut_suspected"] is True
    assert interpretation["training_permission"] is False
