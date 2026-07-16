from __future__ import annotations

from dataclasses import replace

import torch

from trkh.core.config import ModelConfig
from trkh.models.diverse_branch_stem import DiverseBranchConvStem
from trkh.models.model import HybridConvStem, create_model
from trkh.tools.audit_diverse_branch_stem_preflight import (
    _candidate_config,
    _control_config,
    _deployment_structure,
    _fold_control_stem,
    _formal_gate,
    _independent_equation_replay,
    _state_pairing_summary,
)


def _model_config(stem_architecture: str) -> ModelConfig:
    return ModelConfig(
        model_type="vit_registers",
        image_size=256,
        patch_size=16,
        stem_architecture=stem_architecture,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_registers=2,
        dropout=0.0,
        drop_path_rate=0.0,
        multi_branch_fusion=False,
        token_pruning=False,
    )


def test_dbb_preflight_configs_change_only_the_stem() -> None:
    source = vars(_model_config("conv_pool"))
    control = _control_config(source)
    candidate = _candidate_config(source)
    differences = sorted(
        key
        for key in set(control).union(candidate)
        if control.get(key) != candidate.get(key)
    )

    assert differences == ["stem_architecture"]
    assert control["stem_architecture"] == "conv_pool"
    assert candidate["stem_architecture"] == "dbb_conv_pool"
    assert candidate["stem_pooling_mode"] == "max"
    assert candidate["pretrained"] is False


def test_dbb_preflight_state_pairing_and_control_fold_are_exact() -> None:
    control_config = _model_config("conv_pool")
    candidate_config = replace(control_config, stem_architecture="dbb_conv_pool")
    torch.manual_seed(42)
    control = create_model(num_classes=5, model_config=control_config).eval()
    torch.manual_seed(42)
    candidate = create_model(num_classes=5, model_config=candidate_config).eval()
    pairing = _state_pairing_summary(control, candidate)
    inputs = torch.randn(2, 3, 256, 256)
    with torch.inference_mode():
        expected = control(inputs)
        folded = _fold_control_stem(control, inplace=False).eval()
        observed = folded(inputs)

    assert isinstance(control.stem, HybridConvStem)
    assert isinstance(candidate.stem, DiverseBranchConvStem)
    assert pairing["all_nonstem_bit_exact"] is True
    assert pairing["all_origin_bit_exact"] is True
    assert pairing["candidate_only_finite"] is True
    assert pairing["terminal_gamma_one"] is True
    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-5)
    assert torch.equal(expected.argmax(dim=1), observed.argmax(dim=1))
    assert _deployment_structure(folded.stem)["passed"] is True


def test_dbb_preflight_independent_equations_stay_within_locked_tolerance() -> None:
    replay = _independent_equation_replay()

    assert replay["passed"] is True
    assert replay["maximum_error"] <= 1e-6
    assert set(replay["errors"]["path_output"]) == {
        "origin",
        "pointwise",
        "sequential",
        "average",
    }


def test_dbb_preflight_gate_fails_closed_and_sorts_failures() -> None:
    rejected = _formal_gate(
        {
            "source_hash": True,
            "gradient": False,
            "deployment": False,
        }
    )
    accepted = _formal_gate({"source_hash": True, "gradient": True})

    assert rejected["formal_pair_permission"] is False
    assert rejected["failed_checks"] == ["deployment", "gradient"]
    assert rejected["validation_permission"] is False
    assert rejected["test_permission"] is False
    assert accepted["formal_pair_permission"] is True
    assert accepted["failed_checks"] == []
