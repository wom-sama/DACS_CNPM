from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from trkh.models.model import create_model
from trkh.tools.audit_cropr_token_selector_preflight import (
    LOCKED_HASHES,
    LOCKED_OFFICIAL_COMMIT,
    LOCKED_OFFICIAL_TREE,
    _detach_check,
    _direct_equation_replay,
    _official_equation_replay,
    _selector_config,
    _sha256,
    _state_summary,
    _synthetic_learnability,
    parse_args,
)
from trkh.tools.audit_foveal_aggregated_attention_preflight import _load_json


OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\cropr-cvpr2025")


def _locked_paths() -> dict[str, Path]:
    return {
        "protocol": Path(
            "docs/TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
        ),
        "fit_only_data": Path("configs/trkh_cropr_a0_fitonly_20260716.yaml"),
        "resolved_config": Path(
            "runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json"
        ),
        "raw_data": Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
        "declaration": Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
        "declaration_summary": Path(
            "runs/audit_cidt_readiness_full_train_20260714/summary.json"
        ),
        "fold_data": Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
        "fold_summary": Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
        "keeper_checkpoint": Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ),
        "command_history": Path("docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"),
        "v8_launcher": Path("scripts/run_trkh_5class_attention_views_v8.ps1"),
        "selector_runtime": Path("trkh/models/cropr_token_selector.py"),
        "model_runtime": Path("trkh/models/model.py"),
        "train_runtime": Path("trkh/training/train.py"),
        "config_runtime": Path("trkh/core/config.py"),
        "selector_tests": Path("tests/test_cropr_token_selector.py"),
        "official_cropr": OFFICIAL_ROOT / "cls" / "cropr.py",
        "official_vit": OFFICIAL_ROOT / "cls" / "vision_transformer.py",
        "official_engine": OFFICIAL_ROOT / "cls" / "engine.py",
        "official_recipe": OFFICIAL_ROOT / "cls" / "CLASSIFICATION.md",
        "official_license": OFFICIAL_ROOT / "LICENSE",
        "paper": Path(
            r"D:\DataAI\external_sources\official\cropr-cvpr2025-paper.pdf"
        ),
    }


def test_cropr_preflight_locked_hashes_and_official_repository_match() -> None:
    paths = _locked_paths()
    assert set(paths) == set(LOCKED_HASHES)
    for name, path in paths.items():
        assert path.is_file(), (name, path)
        assert _sha256(path) == LOCKED_HASHES[name], name

    commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD^{tree}"],
        text=True,
        encoding="utf-8",
    ).strip()
    assert commit == LOCKED_OFFICIAL_COMMIT
    assert tree == LOCKED_OFFICIAL_TREE


def test_cropr_preflight_independent_oracles_pass() -> None:
    assert _official_equation_replay(OFFICIAL_ROOT / "cls" / "cropr.py")["passed"]
    assert _direct_equation_replay()["passed"]
    assert _detach_check()["passed"]
    synthetic = _synthetic_learnability()
    assert synthetic["loss_reduction_fraction"] >= 0.50
    assert synthetic["relevant_token_top1_hit_rate"] >= 0.95
    assert synthetic["passed"]


def test_cropr_preflight_roles_are_state_identical() -> None:
    resolved = _load_json(
        Path("runs/full_v8_yolof_randominit_30e_20260714_105524/resolved_config.json")
    )
    source = resolved["model_config"]
    torch.manual_seed(42)
    control = create_model(
        num_classes=5,
        model_config=_selector_config(source, enabled=True, routing=False),
    )
    torch.manual_seed(42)
    candidate = create_model(
        num_classes=5,
        model_config=_selector_config(source, enabled=True, routing=True),
    )
    state = _state_summary(control, candidate)
    assert state["bit_exact"]
    assert state["finite"]
    assert state["left_parameter_count"] == state["right_parameter_count"]
    assert not any("routing" in name for name in candidate.state_dict())


def test_cropr_fit_only_view_and_pair_wrapper_block_holdout_leakage() -> None:
    fit_only = Path("configs/trkh_cropr_a0_fitonly_20260716.yaml").read_text(
        encoding="utf-8"
    )
    assert "train: images/train" in fit_only
    assert "val: images/train" in fit_only
    assert "test: images/train" in fit_only
    assert "images/val" not in fit_only
    wrapper = Path("scripts/run_trkh_cropr_token_selector_a0.ps1").read_text(
        encoding="utf-8"
    )
    assert "DataYaml = $FitOnlyDataYaml" in wrapper
    assert "checkpoints\\last.pt" in wrapper
    assert "data_cartography_train_occurrence_hashes.json" in wrapper
    assert "CroprTokenSelectorRouting $false" in wrapper
    assert "CroprTokenSelectorRouting $true" in wrapper
    assert "formal_pair_permission" in wrapper
    assert "SkipFinalTest = $true" in wrapper


def test_cropr_preflight_cli_defaults_are_locked() -> None:
    args = parse_args(["--output-dir", "runs/unit_cropr_preflight"])
    assert args.seed == 42
    assert args.batch_size == 32
    assert args.fp32_batch_size == 2
    assert args.warmup_iterations == 2
    assert args.timed_iterations == 5
    assert args.fit_only_data == Path("configs/trkh_cropr_a0_fitonly_20260716.yaml")
