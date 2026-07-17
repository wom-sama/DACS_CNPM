from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trkh.tools.audit_global_response_normalization_signal import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    CONDITIONS,
    EPOCHS,
    FEATURE_WIDTH,
    LEARNING_RATE,
    MIN_FIT_TP_RETENTION,
    PREFIX_TOKENS,
    ROLES,
    TOKEN_WIDTH,
    TOKENS,
    WEIGHT_DECAY,
    GlobalResponseNormalization,
    GlobalResponseReadout,
    MatchedBlock2FFN,
    PatchGlobalResponseNormalization,
    _independent_grn,
    _load_official_grn_class,
    assess_declaration_replay,
    assess_information_gate,
    normalized_l2_response,
    object_patch_mask,
    parse_args,
    select_recall_constrained_threshold,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.epochs == EPOCHS == 12
    assert args.learning_rate == LEARNING_RATE == 0.003
    assert args.weight_decay == WEIGHT_DECAY == 0.0001
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 3
    assert MIN_FIT_TP_RETENTION == 0.95
    assert ROLES == (
        "identity",
        "grn_l2",
        "grn_object_response",
        "grn_outside_response",
    )
    assert len(CONDITIONS) == 4


def test_official_ast_loader_ignores_unavailable_sparse_import(tmp_path: Path) -> None:
    source = tmp_path / "utils.py"
    source.write_text(
        "\n".join(
            (
                "import torch",
                "import torch.nn as nn",
                "from definitely_missing_package import SparseTensor",
                "class Unrelated:",
                "    pass",
                "class GRN(nn.Module):",
                "    def __init__(self, dim):",
                "        super().__init__()",
                "        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))",
                "        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))",
                "    def forward(self, x):",
                "        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)",
                "        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)",
                "        return self.gamma * (x * nx) + self.beta + x",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    official = _load_official_grn_class(source)(8).double()
    local = GlobalResponseNormalization(8).double()
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    assert torch.equal(official(x), local(x))
    assert not (tmp_path / "__pycache__").exists()


def test_local_grn_matches_independent_equation_and_gradients() -> None:
    torch.manual_seed(17)
    module = GlobalResponseNormalization(8).double()
    with torch.no_grad():
        module.gamma.uniform_(-0.2, 0.2)
        module.beta.uniform_(-0.1, 0.1)
    x = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    observed = module(x)
    expected, response = _independent_grn(
        expected_x, module.gamma, module.beta
    )
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)
    assert torch.isfinite(response).all()
    probe = torch.randn_like(observed)
    observed_grad = torch.autograd.grad(
        (observed * probe).sum(), [x, module.gamma, module.beta], retain_graph=True
    )
    expected_grad = torch.autograd.grad(
        (expected * probe).sum(), [expected_x, module.gamma, module.beta]
    )
    for left, right in zip(observed_grad, expected_grad):
        assert torch.allclose(left, right, atol=1e-11, rtol=0.0)


def test_patch_grn_zero_init_is_identity_and_preserves_prefixes() -> None:
    torch.manual_seed(19)
    module = PatchGlobalResponseNormalization(16, PREFIX_TOKENS)
    tokens = torch.randn(2, TOKENS, 16)
    assert torch.equal(module(tokens), tokens)
    with torch.no_grad():
        module.gamma.fill_(0.25)
        module.beta.fill_(0.1)
    changed = module(tokens)
    assert torch.equal(changed[:, :PREFIX_TOKENS], tokens[:, :PREFIX_TOKENS])
    assert not torch.equal(changed[:, PREFIX_TOKENS:], tokens[:, PREFIX_TOKENS:])


def test_bbox_intersection_masks_and_region_responses_are_distinct() -> None:
    boxes = torch.tensor(
        [[0.5, 0.5, 0.25, 0.25], [0.08, 0.08, 0.05, 0.05]],
        dtype=torch.float32,
    )
    object_mask, outside_mask = object_patch_mask(boxes)
    assert object_mask.shape == outside_mask.shape == (2, 256)
    assert torch.equal(object_mask, ~outside_mask)
    assert object_mask.any(1).all() and outside_mask.any(1).all()

    patches = torch.ones(2, 256, 4)
    patches[:, :, 1] = 2.0
    patches[0, object_mask[0], 0] = 10.0
    object_response = normalized_l2_response(patches, object_mask)
    outside_response = normalized_l2_response(patches, outside_mask)
    assert object_response.shape == outside_response.shape == (2, 1, 4)
    assert not torch.equal(object_response[0], outside_response[0])


def test_readout_zero_init_matches_identity_then_has_live_grn_gradients() -> None:
    torch.manual_seed(23)
    identity = GlobalResponseReadout("identity", 16)
    candidate = GlobalResponseReadout("grn_l2", 16)
    candidate.head.load_state_dict(identity.head.state_dict(), strict=True)
    object_mean = torch.randn(8, 16)
    response = torch.rand(8, 16) + 0.5
    assert torch.equal(identity(object_mean, response)[0], candidate(object_mean, response)[0])
    target = torch.tensor([1.0, 0.0] * 4)
    loss = nn.functional.binary_cross_entropy_with_logits(
        candidate(object_mean, response)[0], target
    )
    loss.backward()
    assert candidate.gamma is not None and candidate.gamma.grad is not None
    assert candidate.beta is not None and candidate.beta.grad is not None
    assert torch.count_nonzero(candidate.gamma.grad) > 0
    assert torch.count_nonzero(candidate.beta.grad) > 0


def test_matched_ffn_zero_init_output_and_parameter_delta() -> None:
    source = nn.Sequential(
        nn.Linear(TOKEN_WIDTH, FEATURE_WIDTH),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(FEATURE_WIDTH, TOKEN_WIDTH),
        nn.Dropout(0.1),
    )
    native = MatchedBlock2FFN(source, use_grn=False).eval()
    candidate = MatchedBlock2FFN(source, use_grn=True).eval()
    tokens = torch.randn(1, TOKENS, TOKEN_WIDTH)
    with torch.inference_mode():
        assert torch.equal(native(tokens), candidate(tokens))
    added = sum(value.numel() for value in candidate.grn.parameters())
    assert added == FEATURE_WIDTH * 2


def test_recall_constrained_threshold_uses_highest_valid_positive_score() -> None:
    scores = np.asarray(
        [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90,
         0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79],
        dtype=np.float64,
    )
    labels = np.asarray([1] * 20 + [0], dtype=np.int64)
    threshold = select_recall_constrained_threshold(scores, labels)
    assert threshold == 0.81
    assert np.mean(scores[labels == 1] >= threshold) == 0.95


def test_declaration_replay_requires_exact_locked_near_tie() -> None:
    row = CleanTrainRow(
        sample_index=3657,
        source_stem="sample",
        image_path=Path("train/sample.jpg"),
        fold=2,
        target=2,
        keeper_prediction=1,
        keeper_probabilities=(0.0, 0.5, 0.5, 0.0, 0.0),
    )
    passing = assess_declaration_replay(
        [{"sample_index": 3657, "target": 2, "prediction": 2}], [row]
    )
    assert passing["passed"] is True
    failing = assess_declaration_replay(
        [{"sample_index": 3657, "target": 2, "prediction": 1}], [row]
    )
    assert failing["passed"] is False


def _metric(
    auroc: float,
    *,
    tp: float = 0.96,
    fp: float = 0.25,
    tp_median: float = 0.75,
    fp_median: float = 0.30,
) -> dict[str, float]:
    return {
        "auroc": auroc,
        "tp_retention": tp,
        "fp_rejection": fp,
        "tp_median_score": tp_median,
        "fp_median_score": fp_median,
    }


def _passing_gate_inputs() -> tuple[dict[str, object], dict[str, object]]:
    names = [name for name, _, _ in CONDITIONS]
    metrics: dict[str, object] = {
        "identity": {name: _metric(0.62) for name in names},
        "grn_l2": {name: _metric(0.67) for name in names},
        "grn_object_response": {name: _metric(0.66, tp=0.94, fp=0.20) for name in names},
        "grn_outside_response": {name: _metric(0.64) for name in names},
    }
    diagnostics = {
        "fold_deltas_grn_vs_identity": [0.04, 0.03, 0.02, -0.01],
        "activity": {
            "mean_oof_gamma_rms": 0.006,
            "mean_object_vector_delta_rms": 0.007,
            "between_sample_full_response_rms": 0.02,
        },
    }
    return metrics, diagnostics


def test_information_gate_enforces_precision_and_object_control() -> None:
    metrics, diagnostics = _passing_gate_inputs()
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is True
    assert result["failed_checks"] == []

    metrics["grn_l2"]["clean"]["fp_rejection"] = 0.19
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is False
    assert "clean_fp_rejection_gte_0p20" in result["failed_checks"]

    metrics, diagnostics = _passing_gate_inputs()
    metrics["grn_object_response"]["clean"]["auroc"] = 0.63
    result = assess_information_gate(metrics, diagnostics)
    assert result["passed"] is False
    assert "object_response_delta_vs_identity_gte_0p025" in result["failed_checks"]
