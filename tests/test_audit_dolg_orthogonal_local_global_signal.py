from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_dolg_orthogonal_local_global_signal import (
    BATCH_SIZE,
    BENCHMARK_REPEATS,
    CHANNELS,
    CONDITIONS,
    DOLGObjectOrthogonalSidecar,
    FIT_FOLDS,
    GRID_SIZE,
    PATCH_TOKENS,
    _independent_orthogonal,
    _load_official_orthogonal_function,
    assess_signal_gate,
    descriptor_roles,
    object_center_mask,
    orthogonal_component,
    parse_args,
    select_recall_constrained_threshold,
)


def test_protocol_defaults_match_locked_recipe() -> None:
    args = parse_args([])
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert args.device == "cuda"
    assert args.batch_size == BATCH_SIZE == 64
    assert args.num_workers == 4
    assert args.benchmark_repeats == BENCHMARK_REPEATS == 5
    assert args.seed == 42
    assert len(CONDITIONS) == 4


def test_orthogonal_component_is_orthogonal_and_matches_independent_oracle() -> None:
    torch.manual_seed(17)
    local = torch.randn(3, 11, 7, dtype=torch.float64)
    global_feature = torch.randn(3, 7, dtype=torch.float64)
    observed = orthogonal_component(local, global_feature)
    dot = (observed * global_feature[:, None, :]).sum(dim=-1)
    assert float(dot.abs().max()) <= 1e-12
    expected = _independent_orthogonal(local.numpy(), global_feature.numpy())
    assert np.max(np.abs(observed.numpy() - expected)) <= 1e-12


def test_projection_commutes_with_uniform_pooling() -> None:
    torch.manual_seed(19)
    local = torch.randn(4, 23, 9, dtype=torch.float64)
    global_feature = torch.randn(4, 9, dtype=torch.float64)
    projected_then_pool = orthogonal_component(local, global_feature).mean(dim=1)
    pooled_then_project = orthogonal_component(local.mean(dim=1), global_feature)
    assert torch.allclose(
        projected_then_pool, pooled_then_project, atol=1e-12, rtol=0.0
    )


def test_sidecar_shape_gradients_and_exact_role_candidate() -> None:
    torch.manual_seed(23)
    local = torch.randn(2, PATCH_TOKENS, CHANNELS, requires_grad=True)
    global_feature = torch.randn(2, CHANNELS, requires_grad=True)
    boxes = torch.tensor(
        [[0.5, 0.5, 0.70, 0.80], [0.45, 0.55, 0.60, 0.72]],
        dtype=torch.float32,
    )
    weights = object_center_mask(boxes).float()
    sidecar = DOLGObjectOrthogonalSidecar()
    candidate = sidecar(local, global_feature, weights)
    roles, mechanism = descriptor_roles(local, global_feature, boxes)
    assert candidate.shape == (2, 2 * CHANNELS)
    assert torch.equal(candidate, roles["global_object_orth"])
    assert mechanism["cell_orth_energy"].shape == (2, PATCH_TOKENS)
    candidate.square().mean().backward()
    assert local.grad is not None and torch.isfinite(local.grad).all()
    assert global_feature.grad is not None and torch.isfinite(global_feature.grad).all()
    assert torch.count_nonzero(local.grad) > 0
    assert torch.count_nonzero(global_feature.grad) > 0


def test_object_center_mask_has_locked_layout_semantics() -> None:
    boxes = torch.tensor(
        [[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    mask = object_center_mask(boxes, GRID_SIZE)
    assert mask.shape == (2, GRID_SIZE * GRID_SIZE)
    assert int(mask[0].sum()) == GRID_SIZE * GRID_SIZE
    assert int(mask[1].sum()) == 64


def test_official_ast_loader_replays_author_source_when_available() -> None:
    path = Path(
        r"D:\DataAI\external_sources\official\dolg-iccv2021\core\model\dolg_model.py"
    )
    if not path.is_file():
        pytest.skip("Official DOLG source is not installed in this environment.")
    official, assignments = _load_official_orthogonal_function(path)
    assert assignments == ["fg_norm", "proj", "proj", "proj", "orth_comp"]
    torch.manual_seed(29)
    global_feature = torch.randn(2, 5, dtype=torch.float64)
    local_map = torch.randn(2, 5, 3, 4, dtype=torch.float64)
    expected = official(global_feature, local_map)
    local = local_map.permute(0, 2, 3, 1).reshape(2, 12, 5)
    observed = orthogonal_component(local, global_feature)
    observed = observed.reshape(2, 3, 4, 5).permute(0, 3, 1, 2)
    torch.testing.assert_close(expected, observed, rtol=0.0, atol=1e-12)


def test_recall_constrained_threshold_uses_highest_valid_score() -> None:
    scores = np.asarray([0.95, 0.90, 0.80, 0.70, 0.60, 0.55])
    labels = np.asarray([1, 1, 1, 1, 1, 0])
    threshold = select_recall_constrained_threshold(
        scores, labels, min_tp_retention=0.80
    )
    assert threshold == 0.70


def _role_metric(
    auroc: float,
    *,
    tp: float = 0.92,
    fp: float = 0.30,
    precision: float = 0.76,
    f1: float = 0.83,
) -> dict[str, object]:
    return {
        "auroc": auroc,
        "tp_retention": tp,
        "fp_rejection": fp,
        "precision": precision,
        "f1": f1,
        "fold_auroc": {str(fold): auroc for fold in FIT_FOLDS},
    }


def _passing_signal_inputs() -> tuple[dict[str, object], dict[str, object]]:
    conditions: dict[str, object] = {}
    for name, _, _ in CONDITIONS:
        candidate_auroc = 0.72 if name == "clean" else 0.715
        conditions[name] = {
            "global": _role_metric(0.68),
            "full_local": _role_metric(0.64),
            "object_local": _role_metric(0.66),
            "global_object_concat": _role_metric(
                0.70, fp=0.24, precision=0.74, f1=0.82
            ),
            "global_full_orth": _role_metric(0.70),
            "global_object_orth": _role_metric(candidate_auroc),
            "bbox_geometry": _role_metric(0.65),
        }
    clean = conditions["clean"]
    clean["global_object_orth"]["fold_auroc"] = {
        "1": 0.73,
        "2": 0.72,
        "3": 0.71,
        "4": 0.70,
    }
    clean["global_object_concat"]["fold_auroc"] = {
        "1": 0.70,
        "2": 0.70,
        "3": 0.70,
        "4": 0.70,
    }
    telemetry = {
        "all_finite": True,
        "global_norm_min": 1.0,
        "candidate_concat_max_abs_delta": 0.2,
        "candidate_features_all_nonzero_variance": True,
        "hook_calls_exact": True,
    }
    return {"conditions": conditions}, telemetry


def test_signal_gate_accepts_all_locked_gains_and_rejects_precision_failure() -> None:
    metrics, telemetry = _passing_signal_inputs()
    result = assess_signal_gate(metrics, telemetry)
    assert result["passed"] is True

    metrics["conditions"]["clean"]["global_object_orth"]["precision"] = 0.739
    result = assess_signal_gate(metrics, telemetry)
    assert result["passed"] is False
    assert result["checks"]["clean_precision_at_least_074"] is False


def test_three_phase_launcher_is_vscode_native_command_safe() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_dolg_orthogonal_local_global_signal_a0.ps1"
    )
    text = launcher.read_text(encoding="utf-8")
    assert '[ValidateSet("Preflight", "Formal", "Finalize")]' in text
    assert "$LASTEXITCODE" in text
    assert "2>&1" not in text
    assert "ForEach-Object" not in text
    assert "Get-CimInstance Win32_Process" in text
    assert "--query-gpu=utilization.gpu,memory.used" in text
    assert "ExpectedSummarySha256" in text
    assert "--expected-summary-sha256" in text
