from __future__ import annotations

import copy
import os

import torch

from trkh.tools.audit_augself_color_adapter_readiness import (
    ADAPTER_HIDDEN_DIM,
    AUXILIARY_WEIGHT,
    BATCH_SIZE,
    EMBED_DIM,
    EXPECTED_ADAPTER_PARAMETERS,
    EXPECTED_PREDICTOR_PARAMETERS,
    HeadResidualAdapter,
    AdaptedLinearHead,
    AugSelfColorPredictor,
    TRAIN_STEPS,
    _required_xai_indices,
    assess_stage_a,
    make_color_pair_batch,
    parse_args,
    private_color_seed,
    symmetric_color_loss,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 32
    assert args.max_train_batches == TRAIN_STEPS == 60
    assert args.learning_rate == 5e-4
    assert args.weight_decay == 0.01
    assert args.auxiliary_weight == AUXILIARY_WEIGHT == 1.0


def test_adapter_and_predictor_parameter_contract_and_zero_init() -> None:
    torch.manual_seed(13)
    classifier = torch.nn.Linear(EMBED_DIM, 5)
    head = AdaptedLinearHead(classifier).eval()
    predictor = AugSelfColorPredictor().eval()
    adapter_parameters = sum(value.numel() for value in head.adapter.parameters())
    predictor_parameters = sum(value.numel() for value in predictor.parameters())
    assert ADAPTER_HIDDEN_DIM == 64
    assert adapter_parameters == EXPECTED_ADAPTER_PARAMETERS == 32_768
    assert predictor_parameters == EXPECTED_PREDICTOR_PARAMETERS == 528_388
    assert torch.count_nonzero(head.adapter.up.weight).item() == 0
    embeddings = torch.randn(7, EMBED_DIM)
    with torch.inference_mode():
        assert torch.equal(head.adapter(embeddings), embeddings)
        assert torch.equal(head(embeddings), classifier(embeddings))


def test_private_color_views_are_deterministic_symmetric_and_rng_isolated() -> None:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    torch.manual_seed(19)
    rgb = torch.rand(6, 3, 24, 20)
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    images = (rgb - mean_tensor) / std_tensor
    indices = [2, 5, 7, 11, 13, 17]
    torch.manual_seed(101)
    expected_global = torch.rand(5)
    torch.manual_seed(101)
    first = make_color_pair_batch(
        images,
        indices,
        phase="unit",
        seed=42,
        step=3,
        mean=mean,
        std=std,
    )
    observed_global = torch.rand(5)
    second = make_color_pair_batch(
        images,
        indices,
        phase="unit",
        seed=42,
        step=3,
        mean=mean,
        std=std,
    )
    assert torch.equal(expected_global, observed_global)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert first[3] == second[3]
    assert all(record["reversal_exact"] for record in first[3])
    assert torch.all(first[2].abs() <= 1.0)
    for record in first[3]:
        for view in (record["view_1"], record["view_2"]):
            assert all(-0.5 <= value <= 0.5 for value in view["centered_parameters"])
    assert private_color_seed(
        phase="unit", seed=42, step=3, sample_index=2, view=1
    ) != private_color_seed(
        phase="unit", seed=42, step=3, sample_index=2, view=2
    )


def test_detached_control_blocks_only_adapter_auxiliary_gradient() -> None:
    torch.manual_seed(23)
    candidate_adapter = HeadResidualAdapter()
    control_adapter = copy.deepcopy(candidate_adapter)
    candidate_predictor = AugSelfColorPredictor().train()
    control_predictor = copy.deepcopy(candidate_predictor).train()
    first_base = torch.randn(12, EMBED_DIM)
    second_base = torch.randn(12, EMBED_DIM)
    target = torch.empty(12, 4).uniform_(-1.0, 1.0)

    candidate_first = candidate_adapter(first_base)
    candidate_second = candidate_adapter(second_base)
    candidate_loss, _, _ = symmetric_color_loss(
        candidate_predictor,
        candidate_first,
        candidate_second,
        target,
        detach_embeddings=False,
    )
    candidate_loss.backward()

    control_first = control_adapter(first_base)
    control_second = control_adapter(second_base)
    control_loss, _, _ = symmetric_color_loss(
        control_predictor,
        control_first,
        control_second,
        target,
        detach_embeddings=True,
    )
    control_loss.backward()

    candidate_adapter_norm = sum(
        0.0 if parameter.grad is None else float(parameter.grad.square().sum())
        for parameter in candidate_adapter.parameters()
    )
    control_adapter_norm = sum(
        0.0 if parameter.grad is None else float(parameter.grad.square().sum())
        for parameter in control_adapter.parameters()
    )
    assert candidate_adapter_norm > 0.0
    assert control_adapter_norm == 0.0
    for candidate_parameter, control_parameter in zip(
        candidate_predictor.parameters(), control_predictor.parameters()
    ):
        assert candidate_parameter.grad is not None
        assert control_parameter.grad is not None
        assert torch.equal(candidate_parameter.grad, control_parameter.grad)


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


def _passing_representation() -> dict[str, object]:
    return {
        "zero_predictor_mse": 1.0,
        "control": {"mse": 0.90},
        "candidate": {"mse": 0.80, "explained_variance_vs_zero": 0.20},
        "two_view_classification": {
            "control": {"accuracy": 0.80, "per_class_f1": [0.8] * 5},
            "candidate": {"accuracy": 0.81, "per_class_f1": [0.81] * 5},
        },
    }


def test_gate_accepts_full_safe_gain_and_rejects_weak_color_learning() -> None:
    passing = assess_stage_a(
        structural_checks={"structure": True},
        representation=_passing_representation(),
        comparisons=_passing_comparisons(),
    )
    assert passing["stage_b_authorized"] is True
    assert passing["failed_checks"] == []

    weak = _passing_representation()
    weak["candidate"] = {"mse": 0.97, "explained_variance_vs_zero": 0.03}
    rejected = assess_stage_a(
        structural_checks={"structure": True},
        representation=weak,
        comparisons=_passing_comparisons(),
    )
    assert rejected["stage_b_authorized"] is False
    assert "candidate_color_mse_lte_0p95_zero" in rejected["failed_checks"]
    assert "candidate_color_mse_lte_0p995_control" in rejected["failed_checks"]


def _row(index: int, target: int, prediction: int, p1: float) -> dict[str, object]:
    return {
        "sample_index": index,
        "target": target,
        "prediction": prediction,
        "prob_1": p1,
    }


def test_xai_selection_covers_all_required_events_then_reaches_twelve() -> None:
    raw = [_row(index, 0, 0, 0.1 + index * 0.01) for index in range(20)]
    control = copy.deepcopy(raw)
    candidate = copy.deepcopy(raw)
    raw[0] = _row(0, 1, 1, 0.8)
    control[0] = _row(0, 1, 1, 0.8)
    candidate[0] = _row(0, 1, 0, 0.3)
    raw[1] = _row(1, 1, 0, 0.3)
    control[1] = _row(1, 1, 0, 0.3)
    candidate[1] = _row(1, 1, 1, 0.7)
    raw[2] = _row(2, 2, 1, 0.7)
    control[2] = _row(2, 2, 1, 0.7)
    candidate[2] = _row(2, 2, 2, 0.2)
    raw[3] = _row(3, 4, 4, 0.2)
    control[3] = _row(3, 4, 4, 0.2)
    candidate[3] = _row(3, 4, 1, 0.7)
    selected, categories, required = _required_xai_indices(
        raw, control, candidate, minimum_rows=12
    )
    assert set(required) == {0, 1, 2, 3}
    assert set(required).issubset(selected)
    assert len(selected) == 12
    assert "class1_tp_break_vs_raw" in categories[0]
    assert "class1_tp_rescue_vs_control" in categories[1]
    assert "restricted_fp_remove_vs_raw" in categories[2]
    assert "restricted_fp_create_vs_control" in categories[3]
